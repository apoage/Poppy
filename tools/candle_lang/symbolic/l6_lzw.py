"""
L6 — LZW compression-tokenizer panel member (design 01 §6 / TASK_798, branch C).

L6 is the panel's *dictionary-compression* member. Where L1 (BPE) and L3 (RePair grammar) grow a
vocabulary by repeatedly merging the most-frequent adjacent pair, L6 grows it the LZW way: a single
left-to-right pass that, whenever it sees a phrase NOT in the dictionary, mints a new id for
(longest-known-prefix + next-symbol) and restarts the phrase at that next symbol. Of the classic
compression algorithms, LZW is the one that yields a clean, deterministic, INTEGER token-id stream
(no byte packing, no entropy-coder bit-fiddling) — so it slots straight into the shared, NON-predictive
candle-language battery (A0/A1/A2/A8/A9) exactly like the other members.

Pipeline (per asset):

    minute OHLC  -> vol-unit substrate (volunit.features_for_run, shared with every member)
                 -> base alphabet (base_alphabet.encode_symbols): each (W,6) feature row -> one of
                    A = N_SYMBOLS = 288 base symbols (ids 0..A-1), rescale-invariant by construction
                 -> FROZEN two-phase LZW: a dict fit on the pooled BUILD base-symbol streams is
                    FROZEN, then any stream is parsed greedily against it to an id stream. NO dict
                    growth at inference (the TASK_797 frozen-tokenizer contract). Base symbols keep
                    their ids 0..A-1; learned phrases get ids A, A+1, ... capped at --max-dict.

NO numerosity reduction (unlike L3): LZW's own phrase growth IS the run-collapsing mechanism — a long
flat run of one base symbol becomes ever-longer dictionary phrases, so we feed the raw base-symbol
stream and let LZW discover the structure.

`mv_t` is rescale-invariant (volunit / eval A0), and the base alphabet + LZW dict are fit ONCE on the
pooled BUILD corpus, so the SAME relative-shape run lands on the SAME token across assets
($0.007 vs $105,747). Judged by the shared NON-predictive battery:
  A0 (rescale-invariance, HARD pre-gate — must be byte-identical),
  A1 (MDL bits-per-candle), A2 (reconstruction — LZW is LOSSLESS on base symbols),
  A8 (cross-asset transfer vs the freq-matched random-codebook null + reported substrate controls),
  A9 (length non-degeneracy; token length = phrase length in base symbols).

CPU / numpy ONLY — no GPU, no JAX (this path never imports JAX). Streaming-bounded: the dict is fit on
a bounded most-recent window cap per asset, every battery reduction is the eval_battery online
O(codebook) fold.

CLI:
    python -m tools.candle_lang.symbolic.l6_lzw \
        --db <path> --out runs/789_token_c0/L6_RESULT.md \
        --max-windows-per-asset 25000 [--max-dict 4096] [--transfer-matrix]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

# Package-relative imports (symbolic/ is a subpackage -> two dots to the package root).
from .. import config as C
from ..base_alphabet import fit_alphabet, encode_symbols as ba_encode_symbols, N_SYMBOLS
from .. import db_stream as DS
from .. import eval_battery as EB
from ..result_writer import write_result
from ..volunit import features_for_run


# --------------------------------------------------------------------------- #
# constants / tunables (kept private; never a contracted signature)
# --------------------------------------------------------------------------- #
# Default cap on the frozen LZW dictionary (THE "word-size limit" knob). Base symbols 0..A-1 consume
# the first A=288 ids; learned phrases fill the remaining (max_dict - A) slots. 4096 mirrors the
# classic 12-bit LZW dictionary cap and L3/L1's vocab order of magnitude.
_DEFAULT_MAX_DICT = 4096
# Window-seam sentinel: phrases never span the seam between two windows (a window is the unit of a
# contiguous dt==60 sub-run slice; spanning it would mint phrases across a discontinuity). The frozen
# LZW parse is reset at each window boundary, so we never need a sentinel id in the stream itself —
# we simply re-seed the LZW state per window (see _lzw_encode_stream).


# --------------------------------------------------------------------------- #
# frozen two-phase LZW (build dict on BUILD -> freeze -> parse to ids)
# --------------------------------------------------------------------------- #
def lzw_build_dict(symbol_streams, A: int, max_dict: int) -> dict:
    """Phase 1 — build the LZW dictionary on the pooled BUILD base-symbol streams, then FREEZE.

    Operates on INTEGER base symbols directly (no byte packing). Base symbols 0..A-1 are seeded as
    single-symbol phrases mapping to their own ids; learned multi-symbol phrases get the next ids
    A, A+1, ... until the dict reaches `max_dict` (no further growth, classic LZW dict-full behavior).

    Each stream is parsed with the standard LZW build loop: extend the current phrase `w` by the next
    symbol; if `w+sym` is known keep extending, else mint a new id for `w+sym` (if room) and restart
    the phrase at `sym`. Streams are independent (phrases never span a window seam) — each stream is a
    single window's base-symbol sequence.

    Returns d : dict{tuple(base symbols) -> id}. d has exactly min(max_dict, A + #learned) entries.
    """
    A = int(A)
    max_dict = int(max_dict)
    d = {(s,): s for s in range(A)}
    nxt = A
    for stream in symbol_streams:
        w = ()
        for sym in stream:
            sym = int(sym)
            wk = w + (sym,)
            if wk in d:
                w = wk
            else:
                if nxt < max_dict:
                    d[wk] = nxt
                    nxt += 1
                w = (sym,)
    return d


def lzw_encode(stream, d: dict, A: int):
    """Phase 2 — encode ONE base-symbol stream to ids against the FROZEN dict `d` (no growth).

    Greedy longest-match LZW parse: extend `w` while `w+sym` is a known phrase; when it isn't, emit
    the id of `w` (the longest known phrase) and restart at `sym`. Because the dict is frozen and
    seeded with every single base symbol (ids 0..A-1), every symbol is always emittable — a novel
    motif simply falls back to a SHORTER known phrase (down to the single base symbol), which is the
    OOV/novelty fallback we log. Returns (ids:list[int], phrase_lengths:list[int]) where
    phrase_lengths[k] = number of base symbols covered by ids[k] (>=1).

    LOSSLESS on base symbols: concatenating the phrases of the emitted ids reproduces `stream`.
    """
    out = []
    lens = []
    w = ()
    for sym in stream:
        sym = int(sym)
        wk = w + (sym,)
        if wk in d:
            w = wk
        else:
            # emit longest known phrase w (always in d: it is a prefix that was extendable, or a
            # single base symbol seeded at A-seed time). Guard the empty-w edge (first symbol).
            emit = w if w else (sym,)
            out.append(d[emit])
            lens.append(len(emit))
            w = (sym,)
    if w:
        out.append(d[w])
        lens.append(len(w))
    return out, lens


# --------------------------------------------------------------------------- #
# frozen LZW vocab object (exposes the eval-battery contract)
# --------------------------------------------------------------------------- #
class L6LZW:
    """Frozen L6 LZW vocabulary: the base alphabet table + the frozen LZW dict, plus the per-id
    phrase expansion (to base symbols) and per-id bar-length the battery needs.

    Public attributes used by the eval battery / A8:
        codebook_size : int    contiguous token-id alphabet size (== max id in dict + 1)
        base_card     : int    number of base symbols A (= N_SYMBOLS, 288)
        token_bar_length : (codebook_size,) int64   phrase length in base symbols per id (A9/A1)
        halflife / sigma_floor : float  vol-unit substrate constants (A0 single-window path)
    Public methods:
        encode_stream(base_symbols(int64,)) -> (ids(int64,), lens(int64,))
        expand(token_id) -> np.ndarray[base symbols]   (A2 reconstruction)
    """

    def __init__(self, table: dict, d: dict, base_card: int, mv_canon: np.ndarray,
                 halflife: float, sigma_floor: float):
        self.table = table
        self.base_card = int(base_card)
        self.d = d
        self.halflife = float(halflife)
        self.sigma_floor = float(sigma_floor)
        # ids are already a dense [0, nxt) range by construction (base symbols 0..A-1, then A, A+1...).
        self.codebook_size = int(max(d.values()) + 1) if d else int(base_card)
        # per-id phrase (tuple of base symbols) -> for expansion + bar-length.
        self._id_to_phrase = [None] * self.codebook_size
        for phrase, tid in d.items():
            self._id_to_phrase[int(tid)] = np.asarray(phrase, dtype=np.int64)
        # any unfilled id (shouldn't happen — dict ids are contiguous) falls back to a length-1 placeholder.
        self.token_bar_length = np.ones(self.codebook_size, dtype=np.int64)
        for tid in range(self.codebook_size):
            ph = self._id_to_phrase[tid]
            if ph is None:
                self._id_to_phrase[tid] = np.asarray([tid % base_card], dtype=np.int64)
                ph = self._id_to_phrase[tid]
            self.token_bar_length[tid] = int(ph.shape[0])
        # canonical per-base-symbol vol-unit move (A2 reconstruction template, mirrors L1).
        self.mv_canon = np.asarray(mv_canon, dtype=np.float64)

    def encode_stream(self, base_symbols: np.ndarray):
        """Encode a base-symbol sequence (one window) to ids + phrase lengths against the frozen dict."""
        seq = np.asarray(base_symbols, dtype=np.int64).tolist()
        ids, lens = lzw_encode(seq, self.d, self.base_card)
        return (np.asarray(ids, dtype=np.int64), np.asarray(lens, dtype=np.int64))

    def expand(self, token_id: int) -> np.ndarray:
        """Token id -> the base-symbol sequence it represents (A2 reconstruction)."""
        tid = int(token_id)
        if 0 <= tid < self.codebook_size:
            return self._id_to_phrase[tid]
        return np.asarray([tid % self.base_card], dtype=np.int64)


# --------------------------------------------------------------------------- #
# substrate streaming -> base-symbol sequences per window
# --------------------------------------------------------------------------- #
def _phase_shuffle_feats(feats: np.ndarray, rng) -> np.ndarray:
    """Donor phase_shuffle_returns surrogate on a (W,6) feature window (mirror of l3/l1 controls):
    phase-randomize the move channel (idx 0), time-permute the geometry channels (idx 1..5). Keeps
    per-bar marginals, destroys temporal order — the A8 'shuffle' substrate null on the base symbols."""
    feats = np.asarray(feats, dtype=np.float64).copy()
    T = feats.shape[0]
    if T >= 8:
        x = feats[:, 0]
        Xf = np.fft.rfft(x)
        mag = np.abs(Xf)
        ph = np.angle(Xf)
        rand_ph = rng.uniform(-np.pi, np.pi, size=ph.shape)
        rand_ph[0] = ph[0]
        if T % 2 == 0:
            rand_ph[-1] = ph[-1]
        Xnew = mag * np.exp(1j * rand_ph)
        feats[:, 0] = np.fft.irfft(Xnew, n=T)
        perm = rng.permutation(T)
        feats[:, 1:] = feats[:, 1:][perm]
    return feats.astype(np.float32)


def _iter_base_symbol_windows(db, asset, cfg, table, max_windows, control=None, seed=0):
    """Yield ONE base-symbol sequence (1-D int64) per source window for `asset`.

    control=None        -> real vol-unit substrate base symbols.
    control='shuffle'   -> donor phase-shuffle on the window features BEFORE base-symbol encode (the
                           A8 structure-destroying null #1 on the substrate, EXACTLY like L1/L3).
    control='random_merge' -> permute the base-symbol sequence in time (same marginals, no order;
                           null #2). Streamed: never materializes the asset's full stream.
    """
    rng = np.random.default_rng(seed + int(asset))
    for chunk in DS.iter_windows(db, asset, cfg, max_windows=max_windows):
        X = chunk["X"]                                  # (n, W, 6)
        for w in range(X.shape[0]):
            feats = X[w]
            if control == "shuffle":
                feats = _phase_shuffle_feats(feats, rng)
            syms = ba_encode_symbols(np.asarray(feats, dtype=np.float64), table)
            if control == "random_merge":
                syms = syms[rng.permutation(syms.size)]
            if syms.size:
                yield syms


# --------------------------------------------------------------------------- #
# fit (pooled BUILD corpus) -> frozen LZW vocab
# --------------------------------------------------------------------------- #
def _fit_move_template(db, assets, cfg, table, max_windows):
    """Online per-base-symbol mean of the mv channel — the A2 reconstruction template (mirrors L1's
    fit_move_template). Two O(N_SYMBOLS) accumulators; never stores the stream. mv_canon[unseen]=0."""
    ns = int(N_SYMBOLS(table))
    s_sum = np.zeros(ns, dtype=np.float64)
    s_cnt = np.zeros(ns, dtype=np.int64)
    for a in assets:
        for chunk in DS.iter_windows(db, a, cfg, max_windows=max_windows):
            X = chunk["X"]
            for w in range(X.shape[0]):
                feats = X[w]
                syms = ba_encode_symbols(np.asarray(feats, dtype=np.float64), table)
                mv = np.asarray(feats, dtype=np.float64)[:, 0]
                np.add.at(s_sum, syms, mv)
                np.add.at(s_cnt, syms, 1)
    return np.where(s_cnt > 0, s_sum / np.maximum(s_cnt, 1), 0.0)


def fit_lzw(db, cfg, build_assets, max_windows_per_asset, max_dict, fit_window_cap=4000):
    """Fit the L6 frozen LZW vocab on the pooled BUILD corpus.

    1) Pool vol-unit feature rows across BUILD assets (bounded by fit_window_cap windows/asset) and
       fit the shared base alphabet (A = N_SYMBOLS = 288 symbols).
    2) Build the LZW dict on the pooled BUILD base-symbol streams (one stream per window so phrases
       never span a seam), FREEZE it, cap at max_dict.
    3) Fit the per-base-symbol canonical mv template for A2.

    Returns (vocab:L6LZW, table, n_fit_windows).
    """
    # ---- 1) base-alphabet fit (pooled vol-unit rows) ----
    pooled_rows = []
    pooled_windows = 0
    for aid in build_assets:
        held = 0
        for chunk in DS.iter_windows(db, int(aid), cfg, max_windows=fit_window_cap):
            X = chunk["X"]
            pooled_rows.append(X.reshape(-1, X.shape[-1]))
            held += X.shape[0]
            pooled_windows += X.shape[0]
            if held >= fit_window_cap:
                break
    if not pooled_rows:
        raise RuntimeError("L6 fit: no BUILD windows streamed — check --db / build_assets.")
    feats_pooled = np.concatenate(pooled_rows, axis=0)
    table = fit_alphabet(feats_pooled, cfg)
    A = int(N_SYMBOLS(table))
    del pooled_rows, feats_pooled

    # ---- 2) build + freeze the LZW dict over BUILD base-symbol streams ----
    def _build_streams():
        for aid in build_assets:
            for syms in _iter_base_symbol_windows(db, int(aid), cfg, table, max_windows_per_asset):
                yield syms.tolist()

    n_fit = 0
    # count fit windows while building (generator is single-pass, so wrap to also count).
    def _counting_streams():
        nonlocal n_fit
        for s in _build_streams():
            n_fit += 1
            yield s

    d = lzw_build_dict(_counting_streams(), A, max_dict)

    # ---- 3) A2 reconstruction template ----
    mv_canon = _fit_move_template(db, build_assets, cfg, table, max_windows_per_asset)

    vocab = L6LZW(table, d, A, mv_canon,
                  halflife=cfg.ewma_halflife, sigma_floor=cfg.sigma_floor)
    return vocab, table, n_fit


# --------------------------------------------------------------------------- #
# encode streams for the battery (token-id chunks, recon pairs, length counts, novelty)
# --------------------------------------------------------------------------- #
def _token_id_chunks(vocab: L6LZW, db, assets, cfg, max_windows, control=None, seed=0):
    """Iterator of token-id CHUNKS (1-D int64), ONE per window — the A1/A8 stream fold input."""
    for a in assets:
        for syms in _iter_base_symbol_windows(db, int(a), cfg, vocab.table, max_windows,
                                              control=control, seed=seed):
            ids, _lens = vocab.encode_stream(syms)
            if ids.size:
                yield ids


def _length_counts(vocab: L6LZW, db, assets, cfg, max_windows):
    """Accumulate token BAR-LENGTH counts {length -> n_tokens} over the stream (A9 input). Online —
    only the small {length->count} dict grows."""
    counts = {}
    for a in assets:
        for syms in _iter_base_symbol_windows(db, int(a), cfg, vocab.table, max_windows):
            _ids, lens = vocab.encode_stream(syms)
            for L in lens.tolist():
                counts[int(L)] = counts.get(int(L), 0) + 1
    return counts


def _recon_pairs(vocab: L6LZW, db, assets, cfg, max_windows):
    """Iterator of (recon_mv, true_mv) PER-BAR pairs for A2 (mode='additive', abs_tol=0.25).

    LZW is LOSSLESS on base symbols: decode the id stream -> base-symbol sequence (== the original),
    map each base symbol to its canonical per-symbol mv (the same template L1 uses) -> per-bar signed
    increments. Compared against the TRUE per-bar mv channel. A2 is therefore the symbol-quantization
    ceiling (the only error is mv -> base-symbol bucketing, NOT a serving bug). Streamed one window
    at a time."""
    for a in assets:
        for chunk in DS.iter_windows(db, int(a), cfg, max_windows=max_windows):
            X = chunk["X"]
            for w in range(X.shape[0]):
                feats = X[w]
                true_mv = np.asarray(feats, dtype=np.float64)[:, 0]
                syms = ba_encode_symbols(np.asarray(feats, dtype=np.float64), vocab.table)
                if syms.size == 0:
                    continue
                ids, _lens = vocab.encode_stream(syms)
                # decode ids back to base symbols (LOSSLESS) then to canonical mv per bar.
                base = np.concatenate([vocab.expand(int(t)) for t in ids]) if ids.size \
                    else np.zeros(0, dtype=np.int64)
                recon_mv = vocab.mv_canon[base].astype(np.float64)
                yield recon_mv, true_mv


def _novelty_stats(vocab: L6LZW, db, assets, cfg, max_windows, control=None, seed=0):
    """OOV/novelty fallback rate on `assets` under the frozen dict: how often the encoder is forced
    to emit a SHORTER phrase because the longer motif was not in the frozen dict. We proxy this by the
    fraction of emitted tokens that are SINGLE BASE SYMBOLS (phrase length 1) — the maximal fallback —
    plus the mean phrase length. On BUILD these are the dict's natural stats; on HOLDOUT a higher
    length-1 fraction / lower mean length == the holdout hits more novel motifs (a useful signal)."""
    n_tokens = 0
    n_len1 = 0
    sum_len = 0
    for a in assets:
        for syms in _iter_base_symbol_windows(db, int(a), cfg, vocab.table, max_windows,
                                              control=control, seed=seed):
            _ids, lens = vocab.encode_stream(syms)
            n_tokens += int(lens.size)
            n_len1 += int(np.count_nonzero(lens == 1))
            sum_len += int(lens.sum())
    return dict(
        n_tokens=int(n_tokens),
        len1_fraction=float(n_len1 / n_tokens) if n_tokens else float("nan"),
        mean_phrase_len=float(sum_len / n_tokens) if n_tokens else float("nan"),
    )


# --------------------------------------------------------------------------- #
# A0 single-window tokenizer (rescale-invariance check)
# --------------------------------------------------------------------------- #
def _make_a0_tokenizer(vocab: L6LZW, cfg):
    """Return tokenize_window_fn(ohlc(W,4)) -> int for eval_battery.a0_rescale_invariance.

    Runs the FULL member path on ONE window via the canonical vol-unit substrate (features_for_run),
    base-symbol encodes, frozen-LZW encodes, and returns the FIRST token id (a stable scalar
    fingerprint). Because the substrate is rescale-invariant and the dict is frozen, x and lam*x yield
    byte-identical ids -> A0 PASS."""
    table = vocab.table

    def tokenize(ohlc):
        ohlc = np.asarray(ohlc, dtype=np.float64)
        W = ohlc.shape[0]
        ts = np.arange(W, dtype=np.int64) * C.BAR_SECONDS
        o, h, l, c = ohlc[:, 0], ohlc[:, 1], ohlc[:, 2], ohlc[:, 3]
        feats, _close, _ts, finite = features_for_run(ts, o, h, l, c, cfg)
        if feats.shape[0] == 0:
            return 0
        feats = feats[finite] if not finite.all() else feats
        if feats.shape[0] == 0:
            return 0
        syms = ba_encode_symbols(np.asarray(feats, dtype=np.float64), table)
        ids, _lens = vocab.encode_stream(syms)
        return int(ids[0]) if ids.size else 0

    return tokenize


def _sample_a0_windows(db, cfg, build_assets, n=24):
    """A SMALL diagnostic sample of raw-price OHLC windows (W,4) for A0 (unit check, not a fold).
    Mirrors L1/L3: build a plausible OHLC envelope around a real close path (rescale-invariance does
    not depend on the envelope, only on it being a fixed multiplicative function of price)."""
    wins = []
    for a in build_assets:
        for chunk in DS.iter_windows(db, int(a), cfg, max_windows=64):
            Cp = chunk["C"]
            for r in range(Cp.shape[0]):
                close = Cp[r].astype(np.float64)
                if close.size < 4 or not np.all(np.isfinite(close)) or np.any(close <= 0):
                    continue
                op = np.empty_like(close); op[0] = close[0]; op[1:] = close[:-1]
                hi = np.maximum(op, close) * 1.0005
                lo = np.minimum(op, close) * 0.9995
                wins.append(np.stack([op, hi, lo, close], axis=1))
                if len(wins) >= n:
                    return wins
            if len(wins) >= n:
                break
        if len(wins) >= n:
            break
    return wins


# --------------------------------------------------------------------------- #
# CLI / main
# --------------------------------------------------------------------------- #
def _default_out() -> str:
    """ABSOLUTE default --out path (repo-root-anchored, like config.db_path())."""
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    return os.path.join(repo_root, "runs", "789_token_c0", "L6_RESULT.md")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="L6 LZW compression candle-language member (CPU).")
    p.add_argument("--db", type=str, default=None, help="DB path (default: config.db_path()).")
    p.add_argument("--out", type=str, default=_default_out(),
                   help="RESULT.md output path (.json sidecar alongside; absolute by default).")
    p.add_argument("--max-windows-per-asset", type=int, default=25000,
                   help="cap windows kept per asset (most-recent ring; RAM bound).")
    p.add_argument("--fit-window-cap", type=int, default=4000,
                   help="windows/asset pooled for the base-alphabet fit.")
    p.add_argument("--max-dict", type=int, default=_DEFAULT_MAX_DICT,
                   help="frozen LZW dictionary cap (the word-size limit knob; base symbols 0..287 "
                        "consume the first 288 ids, phrases fill the rest).")
    p.add_argument("--window", type=int, default=C.DEFAULT_WINDOW)
    p.add_argument("--abs-tol", type=float, default=0.25,
                   help="A2 additive vol-unit tolerance (shared panel default).")
    p.add_argument("--build-assets", nargs="+", type=int, default=None,
                   help="explicit BUILD asset ids (default: CandleConfig build_assets).")
    p.add_argument("--holdout-assets", nargs="+", type=int, default=None,
                   help="explicit HOLDOUT asset ids (default: CandleConfig holdout_assets).")
    p.add_argument("--transfer-matrix", action="store_true",
                   help="compute the per-asset A8 A-by-B transfer matrix + per-holdout breakdown.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    t0 = time.time()
    db = args.db if args.db else C.db_path()
    cfg = C.CandleConfig(
        window=int(args.window), db=db,
        build_assets=tuple(args.build_assets) if args.build_assets else C.DEFAULT_BUILD_ASSETS,
        holdout_assets=tuple(args.holdout_assets) if args.holdout_assets is not None
        else C.DEFAULT_HOLDOUT_ASSETS,
    )
    mwpa = int(args.max_windows_per_asset)
    max_dict = int(args.max_dict)

    print(f"[L6] LZW member | db={cfg.db} window={cfg.window} "
          f"max_windows/asset={mwpa} max_dict={max_dict}", flush=True)
    if not DS.quick_check(cfg.db):
        print(f"[L6] WARNING: PRAGMA quick_check did not return 'ok' for {cfg.db}", flush=True)

    build_assets = list(cfg.build_assets)
    holdout_assets = [a for a in cfg.holdout_assets]

    # ---- fit frozen LZW dict on pooled BUILD ----
    vocab, table, n_fit = fit_lzw(cfg.db, cfg, build_assets, mwpa, max_dict,
                                  fit_window_cap=int(args.fit_window_cap))
    A = int(N_SYMBOLS(table))
    n_phrases = vocab.codebook_size - A
    print(f"[L6] fit done: base_card={A} phrases_learned={n_phrases} "
          f"codebook_size={vocab.codebook_size} (cap {max_dict}) fit_windows={n_fit}", flush=True)

    # ---- A0 rescale-invariance (HARD pre-gate; must be byte-identical) ----
    a0_windows = _sample_a0_windows(cfg.db, cfg, build_assets, n=24)
    if a0_windows:
        a0 = EB.a0_rescale_invariance(_make_a0_tokenizer(vocab, cfg), a0_windows,
                                      lambdas=(0.01, 100.0))
    else:
        a0 = dict(agree=float("nan"), n_trials=0, note="no A0 windows streamed", **{"pass": False})
    print(f"[L6] A0: {a0}", flush=True)

    # ---- A1 MDL over the full BUILD stream ----
    null_bits_per_bar = float(np.log2(max(A, 2)))            # per-bar null = base-symbol entropy
    codebook_bits = float(vocab.codebook_size) * float(np.log2(max(vocab.codebook_size, 2)))
    n_tokens_total = 0
    n_bars_covered = 0
    for chunk in _token_id_chunks(vocab, cfg.db, build_assets, cfg, mwpa):
        ids = np.asarray(chunk, dtype=np.int64)
        n_tokens_total += int(ids.size)
        n_bars_covered += int(vocab.token_bar_length[ids].sum())
    a1 = EB.a1_mdl(
        _token_id_chunks(vocab, cfg.db, build_assets, cfg, mwpa),
        codebook_size=vocab.codebook_size,
        null_bits_per_bar=null_bits_per_bar,
        n_boundaries=n_tokens_total,
        codebook_bits=codebook_bits,
        n_bars_covered=n_bars_covered,
    )
    print(f"[L6] A1: bits/candle={a1['bits_per_candle']:.4f} "
          f"compression={a1['compression_ratio']:.3f} n_tokens={a1['n_tokens']} "
          f"bars_covered={a1['n_bars_covered']}", flush=True)

    # ---- A2 reconstruction (LOSSLESS on base symbols -> symbol-quantization ceiling) ----
    a2 = EB.a2_reconstruction(
        _recon_pairs(vocab, cfg.db, build_assets, cfg, mwpa), mode="additive",
        abs_tol=float(args.abs_tol)
    )
    print(f"[L6] A2: frac_within_tol={a2['frac_within_tol']:.4f} "
          f"mean_abs_err={a2['mean_abs_err']:.5f} (mode={a2['mode']})", flush=True)

    # ---- A9 length non-degeneracy ----
    a9 = EB.a9_length_nondegeneracy(_length_counts(vocab, cfg.db, build_assets, cfg, mwpa))
    print(f"[L6] A9: median_len={a9['median_len']} max_share={a9['max_token_share']:.4f} "
          f"pass={a9['pass']}", flush=True)

    # ---- novelty / OOV fallback rate (BUILD vs HOLDOUT) ----
    nov_build = _novelty_stats(vocab, cfg.db, build_assets, cfg, mwpa)
    nov_hold = _novelty_stats(vocab, cfg.db, holdout_assets, cfg, mwpa) if holdout_assets else {}
    print(f"[L6] novelty BUILD: {nov_build}", flush=True)
    if nov_hold:
        print(f"[L6] novelty HOLDOUT: {nov_hold}", flush=True)

    # ---- A8 cross-asset transfer (only if holdout assets present) ----
    a8 = None
    if holdout_assets:
        def build_vocab_fn(assets=None):
            return vocab  # frozen vocab already fit on BUILD

        def encode_assets_fn(_vocab, assets, control):
            return _token_id_chunks(_vocab, cfg.db, list(assets), cfg, mwpa, control=control)

        a8 = EB.a8_cross_asset_transfer(
            build_vocab_fn, encode_assets_fn,
            build_assets=build_assets, holdout_assets=holdout_assets,
            controls=("shuffle", "random_merge"), exclude_top_k=8,
            per_asset_matrix=args.transfer_matrix,
        )
        a8["a8_power"] = EB.run_a8_power()
        print(f"[L6] A8: transfer_bits={a8['transfer_bits']:.4f} "
              f"perm_min={a8['perm_min_bits']:.4f} "
              f"beats_random_codebook={a8['beats_random_codebook']} "
              f"beats_random_codebook_bigram={a8.get('beats_random_codebook_bigram')} "
              f"a8_verdict={EB.a8_verdict(a8)}", flush=True)

    # ---- verdict (A0 hard pre-gate, then A8 + A2 + A1 + A9) ----
    a0_ok = bool(a0.get("pass", False))
    a8_ok = bool(EB.a8_verdict(a8))
    a2_ok = bool(a2["frac_within_tol"] >= 0.90)
    a1_ok = bool(a1["compression_ratio"] > 1.0)
    a9_ok = bool(a9["pass"])
    proceed = a0_ok and a8_ok and a2_ok and a1_ok and a9_ok
    verdict = "PROCEED" if proceed else "KILL"

    # ---- write RESULT.md + .json ----
    sections = [
        ("Member", [
            "- **L6 — LZW compression tokenizer** (dictionary-compression candle language; CPU/numpy).",
            "- vol-unit substrate -> 288 base symbols -> FROZEN two-phase LZW (dict fit on BUILD,",
            "  frozen, parsed to ids on holdout; NO dict growth at inference — TASK_797 contract).",
            "- Base symbols keep ids 0..287; learned phrases get ids 288.. up to the `--max-dict` cap.",
            "- No numerosity reduction: LZW's own phrase growth is the run-collapsing mechanism.",
        ]),
        ("Configuration", [
            f"- DB: `{cfg.db}`",
            f"- Window {cfg.window} bars | base alphabet {A} | max windows/asset {mwpa}",
            f"- max_dict {max_dict} | phrases learned {n_phrases} | codebook size {vocab.codebook_size}",
            f"- Build assets: {build_assets}",
            f"- Holdout assets: {holdout_assets}",
        ]),
        ("A0 — rescale invariance (HARD pre-gate)", [
            f"- agreement {a0.get('agree')} over {a0.get('n_trials', 0)} trials "
            f"(lambdas {a0.get('lambdas', '(0.01,100)')}) -> pass={a0.get('pass')}",
            "- Must be byte-identical: base symbols come from the rescale-invariant vol-unit substrate.",
        ]),
        ("A1 — MDL (bits per candle, full BUILD stream)", [
            f"- bits/candle {a1['bits_per_candle']:.4f} | entropy/token "
            f"{a1['entropy_bits_per_token']:.4f}",
            f"- null bits/bar {null_bits_per_bar:.4f} | compression ratio "
            f"{a1['compression_ratio']:.4f} (>1 = cheaper than per-bar null)",
            f"- tokens {a1['n_tokens']} | bars covered {a1['n_bars_covered']} | "
            f"codebook {a1['codebook_size']}",
        ]),
        ("A2 — reconstruction (vol-unit mv, per-bar additive)", [
            f"- mode `{a2['mode']}` | frac within tol (±{a2['abs_tol']}) {a2['frac_within_tol']:.4f} "
            f"| mean abs err {a2['mean_abs_err']:.5f}",
            "- LZW is LOSSLESS on base symbols -> A2 == the symbol-quantization ceiling "
            "(error is mv->base-symbol bucketing, not a serving bug).",
            f"- bars compared {a2['n_bars']} over {a2['n_pairs']} windows.",
        ]),
        ("A9 — length non-degeneracy", [
            f"- median token bar-length {a9['median_len']} | max single-length share "
            f"{a9['max_token_share']:.4f}",
            f"- all-length-1 {a9['all_len1']} | distinct lengths {a9.get('n_lengths')} | "
            f"pass {a9['pass']}",
        ]),
        ("Novelty / OOV fallback (frozen-dict diagnostic)", [
            f"- BUILD: len1_fraction {nov_build.get('len1_fraction')} | "
            f"mean_phrase_len {nov_build.get('mean_phrase_len')} | tokens {nov_build.get('n_tokens')}",
        ] + ([
            f"- HOLDOUT: len1_fraction {nov_hold.get('len1_fraction')} | "
            f"mean_phrase_len {nov_hold.get('mean_phrase_len')} | tokens {nov_hold.get('n_tokens')}",
            "- A higher HOLDOUT len1_fraction == the holdout hits more motifs absent from the frozen "
            "dict (forced fallback to a shorter/single-symbol phrase) — the OOV/novelty signal.",
        ] if nov_hold else [])),
    ]
    if a8 is not None:
        a8_lines = list(EB.render_a8_lines(a8))
        a8_lines += list(EB.render_transfer_matrix(a8))
        a8_lines.append(
            f"- **A8 verdict (§186 gate: random-codebook null on unigram AND bigram): "
            f"{EB.a8_verdict(a8)}** "
            f"(substrate controls are entropy-confounded -> reported, NOT gated)"
        )
        sections.append((
            "A8 — cross-asset transfer (headline = non-saturating transfer codelength)", a8_lines))

    meta = dict(
        member="L6_lzw",
        window=int(cfg.window),
        base_card=int(A),
        max_dict=int(max_dict),
        phrases_learned=int(n_phrases),
        codebook_size=int(vocab.codebook_size),
        fit_windows=int(n_fit),
        max_windows_per_asset=int(mwpa),
        novelty_build=nov_build,
        novelty_holdout=(nov_hold if nov_hold else {}),
        a0=a0, a1=a1, a2=a2, a9=a9,
        a8=(a8 if a8 is not None else {}),
        gates=dict(a0=a0_ok, a8=a8_ok, a2=a2_ok, a1=a1_ok, a9=a9_ok),
        wallclock_s=float(time.time() - t0),
        db=cfg.db,
    )

    write_result(args.out, "TASK_798 L6 — LZW Compression Candle Language",
                 sections, verdict, meta)
    print(f"[L6] VERDICT {verdict}  wall {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
