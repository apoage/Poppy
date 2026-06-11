"""
token_timeline.py — Trader-facing per-bar causal token annotation (TASK_800, branch C).

THE PROBLEM (see TASK_800): `tokenizer_api.encode_asset` yields one token chunk per source
window at stride=1 — consecutive chunks re-encode 63 of the same 64 bars. That is the right object
for corpus statistics (the A8 gate) and the WRONG object for a trader: Poppy needs ONE causal token
annotation per (bar, asset), sliceable by the same `window_start` indexing the encoder uses
(`mcts_trader/env/jax_encoder.py:722`).

This module turns a tokenizer + a real asset's minute stream into per-bar int arrays:

    annotate_asset(db, asset_id, tokenizer, cfg) -> TimelineResult
        ts         (T,) int64    one entry per MINUTE BAR (the feature-bar timestamps, ts[k+1])
        token_id   (T,) int32    id of the token covering bar t, computed from bars <= t ONLY
        codelength (T,) float32  bits of the current token under the BUILD bigram (eval_battery reuse)
        valid      (T,) bool     False at gaps (dt!=60) and pre-first-completed-token bars

CONVENTION (DECIDED, do not revisit): emit the LAST COMPLETED token at or before t (prefix-causal).
There is NO <PENDING> vocab entry. Bars before the first completed token get token_id = -1,
valid = False. Downstream (TASK_801) maps -1/invalid to a zero-masked embedding row.

The feature substrate (volunit.features_for_run) is itself causal: feats[k] describes bar ts[k+1]
from bars <= ts[k+1] only, with a causal EWMA sigma re-anchored at every UTC day. So the only
causality hazard is BOUNDARY SENSITIVITY of the tokenizer's segmentation — a BPE merge covering bar
t can change once bar t+1 arrives. The "last completed token at or before t" convention removes the
in-flight (still-growing) tail token from the per-bar annotation; the prefix-stability audit
(audit_causality) measures whatever firm-up lag remains.

Two languages are built (TASK_800, in this order):
  * L8 Haar (window-rate)  : one token per TRAILING 64-bar window, timestamped at the window END
                             (`db_stream.end_ts`). Trivially per-bar causal: the token whose window
                             ends at bar t is the last-completed token at t. churn must be == 0.
  * L1 BPE (variable-rate) : encode each contiguous dt==60 run ONCE, walk tokens with their `bars`
                             coverage map (l1_bpe.numerosity_reduce -> per-unit raw-bar coverage),
                             emit the last COMPLETED token per bar. A bounded firm-up lag is expected.

L2 ABBA / L5 matrix-profile are NOT built — see the feasibility memo in runs/800_timeline/RESULT.md.

ONE PASS PER ASSET -> RAM int arrays (~2 MB/asset-year). Downstream training loops index from RAM;
ZERO tokenization and ZERO DB I/O in any training loop (CLAUDE.md anti-patterns).

PURE NUMPY / CPU ONLY. No JAX, no GPU. Reuses tokenizer_api / l1_bpe / l8_wavelet / db_stream /
volunit / eval_battery — no parallel implementations.
"""
from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np

from . import config as C
from . import base_alphabet as BA
from . import db_stream as DS
from . import eval_battery as EB
from .volunit import features_for_run
from .symbolic import l1_bpe as L1
from .token_window_slice import input_lag_for, lag_shift_per_bar, INVALID_TOKEN_ID


# --------------------------------------------------------------------------- #
# result container
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class TimelineResult:
    """Per-bar causal token annotation for ONE asset under ONE language.

    All four arrays are length T (one entry per MINUTE feature bar, in chronological order). A feature
    bar is bar ts[k+1] of the raw stream (volunit drops the first raw bar of each run). `valid[t]` is
    False at gaps (a feature bar whose dt to the previous feature bar != 60) and at bars before the
    first completed token of that bar's run.
    """
    ts: np.ndarray          # (T,) int64
    token_id: np.ndarray    # (T,) int32
    codelength: np.ndarray  # (T,) float32
    valid: np.ndarray       # (T,) bool
    asset_id: int
    language: str

    def __post_init__(self):
        self.ts = np.ascontiguousarray(self.ts, dtype=np.int64)
        self.token_id = np.ascontiguousarray(self.token_id, dtype=np.int32)
        self.codelength = np.ascontiguousarray(self.codelength, dtype=np.float32)
        self.valid = np.ascontiguousarray(self.valid, dtype=bool)
        T = self.ts.shape[0]
        assert self.token_id.shape[0] == T
        assert self.codelength.shape[0] == T
        assert self.valid.shape[0] == T

    def __len__(self):
        return int(self.ts.shape[0])


# --------------------------------------------------------------------------- #
# codelength: bits of a token under the BUILD bigram (reuse eval_battery)
# --------------------------------------------------------------------------- #
class _CodelengthModel:
    """Per-token codelength under a BUILD unigram model (Laplace-smoothed), reusing the eval_battery
    bigram counting machinery to fit. `bits[id]` = -log2 p(id) where p is the smoothed BUILD marginal.

    We use the UNIGRAM marginal (not the conditional p(curr|prev)) for the per-bar `codelength`
    float, because the per-bar annotation has no canonical predecessor (the predecessor is whatever
    token last completed, which for window-rate vs variable-rate members is not the natural Markov
    prev). The bits float is the "informativeness" signal TASK_803 consumes; a rarer current token is
    more bits. Counts are folded with eval_battery._fold_with_bigram (the SAME ordered-pass counter
    the A8 gate uses) so this is not a parallel implementation.
    """

    def __init__(self, codebook_size: int, alpha: float = 1.0):
        self.K = int(codebook_size)
        self.alpha = float(alpha)
        self._counts = np.zeros(self.K, dtype=np.int64)
        self._bits = None  # lazily finalized

    def fold_ids(self, ids: np.ndarray) -> None:
        """Fold a 1-D token-id stream into the BUILD unigram counts (one ordered pass)."""
        # _fold_with_bigram consumes an ITERATOR of chunks; wrap the single array as one chunk.
        uni, _big = EB._fold_with_bigram([ids], self.K, max_k=0)  # max_k=0 -> skip the dense KxK
        self._counts += uni

    def finalize(self) -> None:
        """Freeze the smoothed -log2 p(id) table (bits per token)."""
        c = self._counts.astype(np.float64)
        n = float(c.sum())
        p = (c + self.alpha) / (n + self.alpha * self.K) if n > 0 else np.full(self.K, 1.0 / self.K)
        self._bits = (-np.log2(np.maximum(p, 1e-300))).astype(np.float64)

    def bits_for(self, token_id: int) -> float:
        """Bits of one token id under the frozen BUILD unigram model. id<0 (-1 invalid) -> 0.0."""
        if self._bits is None:
            self.finalize()
        if token_id < 0 or token_id >= self.K:
            return 0.0
        return float(self._bits[int(token_id)])

    def bits_array(self, token_ids: np.ndarray) -> np.ndarray:
        """Vectorized bits lookup for a (T,) token-id array; id<0 -> 0.0."""
        if self._bits is None:
            self.finalize()
        ids = np.asarray(token_ids, dtype=np.int64)
        out = np.zeros(ids.shape[0], dtype=np.float64)
        ok = (ids >= 0) & (ids < self.K)
        out[ok] = self._bits[ids[ok]]
        return out


# --------------------------------------------------------------------------- #
# run segmentation shared by both members: the per-asset list of contiguous
# dt==60 feature-bar runs (so the per-bar timeline can mark gaps as valid=False).
# --------------------------------------------------------------------------- #
def _iter_feature_runs(db, asset_id, cfg, max_windows=None):
    """Yield (feats(F,N_FEAT)f32, ts_out(F,)i64, finite(F,)bool) per maximal dt==60 raw run.

    feats[k] describes feature bar ts_out[k] (== raw ts[k+1]); F == L-1 for a length-L raw run.
    This is the per-bar grid the timeline annotates: within ONE run all consecutive feature bars are
    dt==60 apart, so a timeline assembled per-run has NO internal gaps; gaps appear only BETWEEN runs
    and are encoded by the run boundaries (the caller marks the inter-run dt and emits valid=False for
    the first feature bar of every run after the first when its dt to the prior run's last bar != 60).

    `max_windows` is honored only as a coarse cap on the number of feature bars retained (kept from the
    TAIL — most-recent — to mirror db_stream's most-recent-window cap), used by the smoke path.
    """
    for ts, o, h, l, c in DS.iter_runs(db, asset_id, cfg):
        feats, _close, ts_out, finite = features_for_run(ts, o, h, l, c, cfg)
        if feats.shape[0] == 0:
            continue
        yield feats, ts_out, finite


def _split_inmem_runs(ts, n):
    """Split an in-memory (ts,) int64 timestamp grid of length n into the (start,end) bar-index
    spans of its maximal dt==C.BAR_SECONDS contiguous runs, mirroring db_stream.iter_runs' run rule
    (a run breaks at any dt!=60, including non-monotone/duplicate timestamps). Returns a list of
    (s, e) half-open raw-bar index spans with e-s >= 2 (a run needs >=2 raw bars to yield >=1 feature
    bar). Bar-validity of OHLC is the CALLER's contract (synthetic slots are dense + finite); the
    db path's _bar_valid guard does not apply to in-memory OHLCV."""
    ts = np.asarray(ts, dtype=np.int64)
    if n < 2:
        return []
    dt = ts[1:] - ts[:-1]
    # break points: a new run starts at index i+1 whenever dt[i] != BAR_SECONDS.
    breaks = np.flatnonzero(dt != C.BAR_SECONDS) + 1   # indices where a new run begins
    bounds = np.concatenate([[0], breaks, [n]]).astype(np.int64)
    spans = []
    for k in range(bounds.shape[0] - 1):
        s, e = int(bounds[k]), int(bounds[k + 1])
        if e - s >= 2:                                 # need >=2 raw bars for >=1 feature bar
            spans.append((s, e))
    return spans


def _iter_feature_runs_inmem(ohlcv, ts, cfg):
    """In-memory mirror of `_iter_feature_runs`: yield (feats(F,N_FEAT)f32, ts_out(F,)i64, finite(F,))
    per maximal dt==C.BAR_SECONDS run of an in-RAM OHLCV slot, calling the SAME features_for_run per
    run as the DB path. This is the ONLY substrate difference between annotate_asset (DB rows via
    db_stream.iter_runs) and annotate_ohlcv (in-memory rows split here) — the per-run tokenization core
    downstream is byte-identical.

    ohlcv : (T, >=4) float — columns 0:4 = O,H,L,C (col 4 = volume, ignored by volunit).
    ts    : (T,)     int64 — per-bar timestamps. Runs break at any dt!=60 (gaps -> valid=False, R2).
    """
    ohlcv = np.asarray(ohlcv, dtype=np.float64)
    ts = np.asarray(ts, dtype=np.int64)
    T = ohlcv.shape[0]
    o = ohlcv[:, 0]; h = ohlcv[:, 1]; l = ohlcv[:, 2]; c = ohlcv[:, 3]
    for (s, e) in _split_inmem_runs(ts, T):
        feats, _close, ts_out, finite = features_for_run(ts[s:e], o[s:e], h[s:e], l[s:e], c[s:e], cfg)
        if feats.shape[0] == 0:
            continue
        yield feats, ts_out, finite


# --------------------------------------------------------------------------- #
# L8 Haar window-rate timeline (the trivial case — build this first)
# --------------------------------------------------------------------------- #
def _annotate_l8(run_iter, l8_vocab, cfg, asset_id, code_model: Optional[_CodelengthModel] = None,
                 max_feature_bars: Optional[int] = None) -> TimelineResult:
    """Per-bar L8 timeline. L8 mints ONE token per trailing window of W feature bars; the token is
    timestamped at the window END bar (db_stream end_ts / the last bar of the window). For a per-bar
    causal channel: bar t's token is the token of the window ENDING at t (covers bars [t-W+1 .. t]),
    which is by construction the last-completed token at or before t. Bars before the first full
    window in their run get token_id=-1, valid=False. Gaps (between runs) get valid=False.

    L8 churn must be EXACTLY 0 (the window ending at t never changes when bar t+1 arrives — t+1's
    window is a NEW token timestamped at t+1; t's token is final the moment bar t exists).

    `run_iter` yields (feats(F,N_FEAT), ts_out(F,), finite(F,)) per maximal dt==60 run — supplied by
    `_iter_feature_runs` (DB / annotate_asset) or `_iter_feature_runs_inmem` (annotate_ohlcv). The
    per-run tokenization below is IDENTICAL for both substrates (the R3 cross-producer invariant).
    """
    from .symbolic import l8_wavelet as L8
    W = int(l8_vocab.win_len)

    ts_chunks, id_chunks, valid_chunks = [], [], []
    prev_run_last_ts = None

    for feats, ts_out, finite in run_iter:
        F = feats.shape[0]
        mv = feats[:, 0].astype(np.float64)               # vol-unit move channel (L8 substrate)
        ids = np.full(F, -1, dtype=np.int64)
        valid = np.zeros(F, dtype=bool)
        # the window ending at feature bar s (0-based within the run) covers [s-W+1 .. s]; it exists
        # for s >= W-1. Require the whole window finite (matches _iter_mv_windows' finite guard).
        for s in range(W - 1, F):
            lo = s - W + 1
            if not finite[lo:s + 1].all():
                continue
            tok = int(l8_vocab.encode_window(mv[lo:s + 1]))
            ids[s] = tok
            valid[s] = True
        # gap handling: the FIRST feature bar of a run that does not continue dt==60 from the prior
        # run's last bar is a gap boundary, but within a run all bars are dt==60 already; the
        # pre-first-window bars are simply valid=False via the loop above. Nothing extra to mark here
        # because valid is False by default until a full window completes.
        ts_chunks.append(ts_out)
        id_chunks.append(ids)
        valid_chunks.append(valid)
        prev_run_last_ts = int(ts_out[-1])
        if max_feature_bars is not None:
            tot = sum(c.shape[0] for c in ts_chunks)
            if tot >= max_feature_bars:
                break

    return _assemble(ts_chunks, id_chunks, valid_chunks, asset_id, "L8_haar",
                     code_model, max_feature_bars)


# --------------------------------------------------------------------------- #
# L1 BPE variable-rate timeline (encode each run ONCE; walk the bars coverage map)
# --------------------------------------------------------------------------- #
def _bpe_prefix_token_per_bar(syms: np.ndarray, vocab) -> np.ndarray:
    """Given one run's base-symbol sequence `syms` (F,), return token_per_bar (F,) int64: for each
    raw feature bar k, the id of the LAST COMPLETED token at or before k under the prefix-causal
    convention.

    'Last completed token at or before k' = encode the FULL run, walk the resulting tokens with their
    raw-bar coverage (numerosity run-lengths summed per token), and assign each token to ALL its
    covered bars EXCEPT we must NOT credit the still-in-flight final token to bars that, from a
    prefix ending earlier, would not yet have completed it. The per-bar convention is: a token is
    'completed' at the bar where its LAST covered raw bar lands. So bar k is annotated with the most
    recent token whose last covered bar index <= k. Concretely: build, per token, its [first_bar,
    last_bar] coverage; the annotation for bar k is the token with the GREATEST last_bar <= k.

    This is computed once on the full run (the audit re-runs it on prefixes to measure firm-up lag).
    """
    F = int(syms.shape[0])
    out = np.full(F, -1, dtype=np.int64)
    if F == 0:
        return out
    # numerosity reduce -> leaf units + per-unit raw-bar coverage (run-lengths).
    units, bars = L1.numerosity_reduce(syms)                 # units(M,), bars(M,) sum==F
    if units.size == 0:
        return out
    # per-leaf-unit cumulative LAST raw-bar index (0-based within the run).
    unit_last_bar = np.cumsum(bars) - 1                      # (M,) last covered bar per reduced unit
    # encode the reduced units -> tokens, tracking how many UNITS each token spans (vocab.unit_length).
    token_ids = L1.encode_units_bpe(units, vocab)            # (NT,)
    # walk tokens left-to-right; each consumes unit_length(tok) reduced units; the token's last raw
    # bar is unit_last_bar[unit_cursor + n_units - 1].
    unit_cursor = 0
    tok_last_bar = np.empty(token_ids.shape[0], dtype=np.int64)
    for ti, tok in enumerate(token_ids.tolist()):
        nu = int(vocab.unit_length(int(tok)))
        last_unit = unit_cursor + nu - 1
        tok_last_bar[ti] = int(unit_last_bar[last_unit])
        unit_cursor += nu
    # assign: bar k -> the token whose last_bar is the greatest value <= k. tok_last_bar is strictly
    # increasing (tokens cover disjoint consecutive bar spans), so a step-fill does it: for each
    # token ti completing at bar b=tok_last_bar[ti], all bars in (prev_b .. b] get token ti EXCEPT
    # the "last completed at or before k" rule means bar k carries the token that COMPLETED on/before
    # k — i.e. bars strictly between two completion points carry the EARLIER token (it is the last
    # one fully completed). So bar k in [tok_last_bar[ti-1]+1 .. tok_last_bar[ti]-1] still has only
    # ti-1 completed; bar tok_last_bar[ti] is the first bar where ti is completed.
    prev_b = -1
    for ti in range(token_ids.shape[0]):
        b = int(tok_last_bar[ti])
        # bars (prev_b .. b-1]  -> last completed is ti-1 (already set by the previous iteration's
        # fill of [.. prev_b]); bar b and onward (until next completion) -> ti.
        out[b:] = int(token_ids[ti])  # ti is completed from bar b onward (overwritten by later toks)
        prev_b = b
    # bars before the first token completes (b0 = tok_last_bar[0]) stay -1 (no completed token yet).
    out[: int(tok_last_bar[0])] = -1
    return out


def _annotate_l1(run_iter, l1_vocab, cfg, asset_id, code_model: Optional[_CodelengthModel] = None,
                 max_feature_bars: Optional[int] = None) -> TimelineResult:
    """Per-bar L1 BPE timeline. Encode each contiguous dt==60 run ONCE (NOT per-window): base symbols
    over the whole run -> numerosity reduce -> BPE -> walk the tokens' raw-bar coverage -> emit the
    last-completed token per bar (_bpe_prefix_token_per_bar). Pre-first-token bars and gaps are
    valid=False.

    `run_iter` yields (feats(F,N_FEAT), ts_out(F,), finite(F,)) per maximal dt==60 run — supplied by
    `_iter_feature_runs` (DB / annotate_asset) or `_iter_feature_runs_inmem` (annotate_ohlcv). The
    per-run tokenization below is IDENTICAL for both substrates (the R3 cross-producer invariant)."""
    table = l1_vocab.table
    ts_chunks, id_chunks, valid_chunks = [], [], []

    for feats, ts_out, finite in run_iter:
        F = feats.shape[0]
        # encode the WHOLE run's base symbols once (run is gap-free dt==60 by construction).
        syms = BA.encode_symbols(feats.astype(np.float64), table)   # (F,) base symbols
        token_per_bar = _bpe_prefix_token_per_bar(syms, l1_vocab)    # (F,) last-completed token / bar
        valid = (token_per_bar >= 0) & finite
        ids = token_per_bar.copy()
        ids[~valid] = -1
        ts_chunks.append(ts_out)
        id_chunks.append(ids)
        valid_chunks.append(valid)
        if max_feature_bars is not None:
            tot = sum(c.shape[0] for c in ts_chunks)
            if tot >= max_feature_bars:
                break

    return _assemble(ts_chunks, id_chunks, valid_chunks, asset_id, "L1_bpe",
                     code_model, max_feature_bars)


# --------------------------------------------------------------------------- #
# assembly: concat runs into one per-bar timeline, mark inter-run gaps, codelength
# --------------------------------------------------------------------------- #
def _assemble(ts_chunks, id_chunks, valid_chunks, asset_id, language,
              code_model, max_feature_bars):
    """Concatenate per-run arrays into one chronological timeline. Inter-run boundaries are gaps:
    consecutive runs are separated by dt!=60 (else db_stream would have kept them one run), so the
    FIRST feature bar of every run after the first has dt!=60 to the previous bar -> valid=False at
    that bar (no forward-fill). Codelength is filled from the BUILD model if provided, else zeros."""
    if not ts_chunks:
        empty_i64 = np.zeros(0, dtype=np.int64)
        return TimelineResult(empty_i64, np.zeros(0, np.int32), np.zeros(0, np.float32),
                              np.zeros(0, bool), asset_id, language)

    ts = np.concatenate(ts_chunks).astype(np.int64)
    ids = np.concatenate(id_chunks).astype(np.int64)
    valid = np.concatenate(valid_chunks).astype(bool)

    # explicit gap mask from the per-bar dt (any feature bar whose dt to the previous bar != 60 is a
    # gap boundary -> valid=False, matching config.py:36-37 NO forward-fill). The first bar overall
    # has no predecessor; it is a pre-first-token bar anyway (its run hasn't completed a token yet).
    if ts.shape[0] >= 2:
        dt = np.empty(ts.shape[0], dtype=np.int64)
        dt[0] = 0
        dt[1:] = ts[1:] - ts[:-1]
        gap = np.empty(ts.shape[0], dtype=bool)
        gap[0] = True                      # no predecessor -> treat as a boundary (valid stays via token rule)
        gap[1:] = dt[1:] != C.BAR_SECONDS
        valid = valid & ~gap
        ids[~valid] = -1

    if max_feature_bars is not None and ts.shape[0] > max_feature_bars:
        # keep the MOST-RECENT max_feature_bars (mirror db_stream's tail-keep) for the smoke path.
        ts = ts[-max_feature_bars:]
        ids = ids[-max_feature_bars:]
        valid = valid[-max_feature_bars:]

    if code_model is not None:
        codelength = code_model.bits_array(ids).astype(np.float32)
    else:
        codelength = np.zeros(ts.shape[0], dtype=np.float32)
    codelength[~valid] = 0.0

    return TimelineResult(ts, ids.astype(np.int32), codelength, valid, asset_id, language)


# --------------------------------------------------------------------------- #
# public entry point
# --------------------------------------------------------------------------- #
def annotate_asset(db, asset_id, tokenizer, cfg, *, code_model=None,
                   max_feature_bars=None) -> TimelineResult:
    """Per-bar causal token annotation for ONE asset under ONE language.

    db         : DB path (str).
    asset_id   : int.
    tokenizer  : a frozen tokenizer object. Dispatch is by type:
                   * tokenizer_api.CandleTokenizer (or an object exposing `.vocab` BPEVocab + `.table`)
                     -> L1 BPE timeline.
                   * l8_wavelet.HaarVocab (exposes `.win_len` + `.encode_window`) -> L8 Haar timeline.
    cfg        : CandleConfig (window / halflife / sigma_floor MUST match what the vocab was fit under).
    code_model : optional _CodelengthModel (BUILD unigram). If None, codelength is zeros.
    max_feature_bars : optional cap on retained feature bars (kept from the TAIL); for the smoke path.

    Returns a TimelineResult (ts, token_id, codelength, valid — all length T, one entry per bar).
    """
    cfg = cfg if cfg is not None else getattr(tokenizer, "cfg", None) or C.CandleConfig()
    lang = _language_of(tokenizer)
    aid = int(asset_id)
    # the DB run-iterator: identical (feats, ts_out, finite) tuples that annotate_ohlcv builds in-mem.
    run_iter = _iter_feature_runs(db, aid, cfg)
    if lang == "L8_haar":
        return _annotate_l8(run_iter, tokenizer, cfg, aid, code_model, max_feature_bars)
    if lang == "L1_bpe":
        vocab = getattr(tokenizer, "vocab", tokenizer)  # CandleTokenizer.vocab or a raw BPEVocab
        # carry the table onto the vocab if the caller passed a bare BPEVocab (it already has .table)
        return _annotate_l1(run_iter, vocab, cfg, aid, code_model, max_feature_bars)
    raise TypeError(f"annotate_asset: unrecognized tokenizer type {type(tokenizer)!r}")


# --------------------------------------------------------------------------- #
# public entry point — in-memory OHLCV (synthetic / Bella slots; TASK_801b-i Fork A)
# --------------------------------------------------------------------------- #
def annotate_ohlcv(ohlcv, ts, tokenizer, cfg, *, code_model=None,
                   max_feature_bars=None) -> TimelineResult:
    """Per-bar causal token annotation for ONE in-memory OHLCV slot under ONE language.

    The in-memory twin of `annotate_asset`: instead of reading DB rows via db_stream.iter_runs, it
    splits the provided (ts, OHLC) into maximal dt==C.BAR_SECONDS runs (`_iter_feature_runs_inmem`)
    and feeds the SAME L1/L8 per-run tokenization core. So real (DB) and synthetic (Bella) slots are
    byte-identical token producers up to substrate + head warm-up; the R3 audit isolates exactly that.

    ohlcv  : (T, >=4) array — columns 0:4 = O,H,L,C (col 4 = volume, IGNORED by volunit).
    ts     : (T,) int64 timestamps, OR None.
             R1 (FROZEN synthetic-slot convention): ts=None -> ts = arange(T) * C.BAR_SECONDS.
             That gives dt==60 throughout (one contiguous run, no gaps) and, with t0=0, the volunit
             UTC-day sigma re-anchor (ts//86400) flips every 86400/60 = 1440 bars, anchored at bar 0.
             This is the canonical convention for Bella / synthetic slots that carry no real wall-clock
             timestamps; pass a real ts array for real-bar-derived slots.
    tokenizer : same dispatch as annotate_asset (HaarVocab -> L8; CandleTokenizer/BPEVocab -> L1).
    cfg    : CandleConfig (window/halflife/sigma_floor MUST match the vocab's fit substrate).
    code_model : optional _CodelengthModel (BUILD unigram). If None, codelength is zeros.
    max_feature_bars : optional tail cap on retained feature bars (mirrors annotate_asset).

    Returns a TimelineResult on the FEATURE-bar grid (volunit drops the first raw bar of each run),
    identical structure to annotate_asset. R2: dt!=60 breaks a run -> boundary feature bars valid=False,
    NO forward-fill ever.
    """
    ohlcv = np.asarray(ohlcv, dtype=np.float64)
    if ohlcv.ndim != 2 or ohlcv.shape[1] < 4:
        raise ValueError(f"annotate_ohlcv: ohlcv must be (T,>=4) [O,H,L,C,...]; got {ohlcv.shape}")
    T = ohlcv.shape[0]
    if ts is None:
        # R1 FROZEN convention: synthetic minute grid anchored at t0=0 (dt==60; day re-anchor / 1440).
        ts = np.arange(T, dtype=np.int64) * C.BAR_SECONDS
    else:
        ts = np.asarray(ts, dtype=np.int64)
        if ts.shape[0] != T:
            raise ValueError(f"annotate_ohlcv: ts length {ts.shape[0]} != ohlcv rows {T}")

    cfg = cfg if cfg is not None else getattr(tokenizer, "cfg", None) or C.CandleConfig()
    lang = _language_of(tokenizer)
    # asset_id is bookkeeping only for an in-memory slot; -1 marks "synthetic / no DB id".
    aid = -1
    run_iter = _iter_feature_runs_inmem(ohlcv, ts, cfg)
    if lang == "L8_haar":
        return _annotate_l8(run_iter, tokenizer, cfg, aid, code_model, max_feature_bars)
    if lang == "L1_bpe":
        vocab = getattr(tokenizer, "vocab", tokenizer)
        return _annotate_l1(run_iter, vocab, cfg, aid, code_model, max_feature_bars)
    raise TypeError(f"annotate_ohlcv: unrecognized tokenizer type {type(tokenizer)!r}")


def _language_of(tokenizer) -> str:
    """Identify the member from the tokenizer object's surface (no isinstance import gymnastics)."""
    # L8 HaarVocab: has win_len + encode_window + radixes, NO BPE merges.
    if hasattr(tokenizer, "win_len") and hasattr(tokenizer, "encode_window"):
        return "L8_haar"
    # L1: CandleTokenizer (.vocab is a BPEVocab) or a bare BPEVocab (.merges + .table + .unit_length).
    v = getattr(tokenizer, "vocab", tokenizer)
    if hasattr(v, "merges") and hasattr(v, "unit_length") and hasattr(v, "table"):
        return "L1_bpe"
    raise TypeError(f"_language_of: cannot classify tokenizer {type(tokenizer)!r}")


# --------------------------------------------------------------------------- #
# timestamp-join onto a slot grid (TASK_801b-i Fork A, Deliverable 2)
# --------------------------------------------------------------------------- #
def join_timeline_to_grid(timeline: "TimelineResult", grid_ts, language):
    """Timestamp-join a per-bar token TIMELINE onto a slot's `valid_timestamps` GRID.

    The slot's grid (`grid_ts`, == HistoricalMarket.valid_timestamps for a real slot, or the synthetic
    slot's own dt==60 grid) is a DENSE union grid whose per-asset gaps are FORWARD-FILLED upstream
    (~476k fills/period). Forward-filled rows carry NO real-bar timestamp, so they have no entry in the
    timeline's {ts -> token} map and land -1/invalid here — this AUTO-SOLVES the no-forward-fill rule
    (R2): we never serve a token for a filled (non-real) bar.

    Steps:
      1. Build {ts -> (token_id, is_valid)} from the timeline's VALID entries only (invalid timeline
         bars — gaps / pre-first-token — are excluded so they cannot match a grid row).
      2. For each grid row r: grid_ts[r] in the map AND valid -> its token_id; else -> -1, invalid.
         This masks BOTH forward-filled grid rows (no real bar) AND timeline gaps.
      3. Apply the serving lag ON THE GRID (R4): an INDEX-shift by `input_lag_for(language)` over the
         grid index axis (the grid index IS the model's window_start space), leading `lag` rows -> -1.
         `input_lag_for` is the ONLY lag source (long-form 'L1_bpe'/'L8_haar' accepted). Targets are
         NOT this function's concern.

    Args:
        timeline : TimelineResult (ts, token_id, valid) from annotate_asset / annotate_ohlcv.
        grid_ts  : (G,) int — the slot grid timestamps (valid_timestamps).
        language : 'L1_bpe' / 'L8_haar' (long-form) or 'L1' / 'L8' — resolved via input_lag_for.
    Returns:
        token_ids : (G,) int32 — lag-applied grid token ids, -1 at filled/gap/leading-lag rows.
        valid     : (G,) bool  — True only where a real, lag-firmed token landed.
    """
    grid_ts = np.asarray(grid_ts, dtype=np.int64)
    G = grid_ts.shape[0]

    # 1) {ts -> token_id} over VALID timeline entries only. A real-bar grid ts matches exactly one
    #    timeline ts (both are real-bar timestamps on the same minute grid); duplicates in the timeline
    #    cannot occur (a feature-bar grid is strictly increasing within a run, runs are disjoint in ts).
    t_ts = np.asarray(timeline.ts, dtype=np.int64)
    t_tok = np.asarray(timeline.token_id, dtype=np.int64)
    t_val = np.asarray(timeline.valid, dtype=bool)
    ts_to_tok = {int(ts): int(tok) for ts, tok, v in zip(t_ts, t_tok, t_val) if v}

    # 2) per-grid-row lookup (pre-lag). Filled rows (ts not a real bar) + any non-valid -> -1.
    grid_tok_raw = np.full(G, INVALID_TOKEN_ID, dtype=np.int32)
    for r in range(G):
        tok = ts_to_tok.get(int(grid_ts[r]))
        if tok is not None:
            grid_tok_raw[r] = tok

    # 3) serving lag as a grid-index shift (R4). input_lag_for is the SINGLE lag source.
    lag = input_lag_for(language)
    token_ids = lag_shift_per_bar(grid_tok_raw, lag).astype(np.int32)
    valid = token_ids >= 0
    return token_ids, valid
