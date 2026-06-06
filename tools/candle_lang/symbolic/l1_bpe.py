"""
L1 BPE-delta — the PRIMARY first-experiment candle language (design 01 §2.3 / §6, branch C).

Pipeline (all CPU, pure numpy, NO GPU/JAX needed):

    vol-unit features (db_stream.iter_windows)
      -> base symbols              (base_alphabet.encode_symbols, frozen pooled product alphabet)
      -> numerosity reduction      (collapse identical-symbol runs to one symbol + COARSE 3-bucket
                                    run-length tag: short/med/long — "FLAT×short" != "FLAT×long",
                                    design §2.3 step 1 / F5)
      -> frequency-greedy BPE      (RePair/BPE merges pooled across BUILD assets, p_min floor,
         merges                     Götz 2025 / Tavabi-Lerman style, ~2-4x compression; design §2.3
                                    step 2 / §6 E2)
      -> frozen vocab
      -> encode via EXACT BPE MERGE RANKING (NOT greedy longest-match — longest-match silently
         breaks rescale-invariance at serving, design §2.6 / critique #69).

Then the shared non-predictive battery A0/A1/A2/A8/A9 (eval_battery) + result_writer.write_result.

Token alphabet layout (one flat integer id space, FIXED):
    [0,             N_BASE_TAGGED)          : "leaf" reduced symbols  = base_symbol * N_LEN_BUCKETS + bucket
    [N_BASE_TAGGED, N_BASE_TAGGED + |merges|): "merge" tokens         = leaf-pair composites (BPE non-terminals)
codebook_size = N_BASE_TAGGED + len(merges). A0/A1/A8 fold token ids into an O(codebook) bincount.

Numerosity-reduction note: a reduced UNIT is (base_symbol, len_bucket). BPE then merges adjacent
reduced units. A "token length" (for A9) is the number of reduced units a token spans (1 for a leaf,
sum of children for a merge). Bars covered (for A1/A2 boundary accounting) is the number of raw
feature bars the token spans (the run-lengths summed) — tracked separately from unit-length.

HARD RULES honored: writes ONLY this file; imports the frozen foundation; CPU-only; no GPU; all
battery folds are streamed O(codebook); the BUILD-pooled corpus for fitting is bounded by
--max-windows-per-asset (the most-recent windows, db_stream's ring), never the whole DB.
"""
from __future__ import annotations

import argparse
import os
import time
from collections import Counter

import numpy as np

from .. import config as C
from .. import base_alphabet as BA
from .. import db_stream as DS
from .. import eval_battery as EB
from .. import result_writer as RW
from ..volunit import features_for_run


# --------------------------------------------------------------------------- #
# numerosity reduction — coarse 3-bucket run-length tag (design §2.3 / F5)
# --------------------------------------------------------------------------- #
N_LEN_BUCKETS = 3                       # short / med / long (coarse; "FLAT×short" != "FLAT×long")
_LEN_EDGES = (2, 5)                     # run-length thresholds: 1 -> short(0); 2..4 -> med(1); >=5 -> long(2)


def _len_bucket(run_len: int) -> int:
    """Coarse 3-bucket run-length tag. 1 bar -> short, 2-4 -> med, >=5 -> long."""
    if run_len < _LEN_EDGES[0]:
        return 0
    if run_len < _LEN_EDGES[1]:
        return 1
    return 2


def numerosity_reduce(symbols: np.ndarray):
    """Collapse maximal runs of an identical base symbol into one (symbol, len_bucket) reduced unit.

    symbols : (T,) int64 base-symbol ids (one run of contiguous feature bars).
    Returns (units(M,) int64 leaf-token ids, bars(M,) int64 raw-bar coverage per unit).
        leaf id = base_symbol * N_LEN_BUCKETS + len_bucket   (the flat alphabet's leaf region).
    Symbol-level noise filter: long flat stretches become a single 'long' unit rather than many bars.
    """
    symbols = np.asarray(symbols, dtype=np.int64).ravel()
    T = symbols.size
    if T == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    # run-boundaries where the symbol changes
    change = np.empty(T, dtype=bool)
    change[0] = True
    change[1:] = symbols[1:] != symbols[:-1]
    starts = np.flatnonzero(change)
    run_syms = symbols[starts]
    run_lens = np.diff(np.append(starts, T)).astype(np.int64)
    buckets = np.array([_len_bucket(int(rl)) for rl in run_lens], dtype=np.int64)
    leaves = run_syms * N_LEN_BUCKETS + buckets
    return leaves, run_lens


def n_base_tagged(table: dict) -> int:
    """Size of the leaf (reduced-symbol) region of the flat alphabet = N_SYMBOLS * N_LEN_BUCKETS."""
    return int(BA.N_SYMBOLS(table)) * N_LEN_BUCKETS


# --------------------------------------------------------------------------- #
# frozen vocab object (what A8 / encode consume)
# --------------------------------------------------------------------------- #
class BPEVocab:
    """Frozen BPE vocabulary: ordered merge list + the flat-id layout.

    merges      : list of (left_id:int, right_id:int) in MERGE RANK ORDER (rank 0 applied first).
                  Each merge i mints a new token id = n_base_tagged + i.
    ranks       : dict (left,right) -> rank (== merge index), for O(1) serving merge selection.
    children    : dict new_id -> (left,right) for length/bar expansion.
    n_base      : n_base_tagged(table) (leaf region size).
    table       : the base alphabet table (carried for A8 re-fit / encode).
    """

    def __init__(self, merges, table):
        self.table = table
        self.n_base = n_base_tagged(table)
        self.merges = list(merges)
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.children = {self.n_base + i: pair for i, pair in enumerate(self.merges)}
        # precompute, per token id, its unit-length and raw-bar span for A1/A9 (leaves resolved
        # lazily against actual run-lengths at encode time; merges expand recursively into leaves).
        self._unit_len_cache = {}

    @property
    def codebook_size(self) -> int:
        return int(self.n_base + len(self.merges))

    @property
    def vocab_size(self) -> int:
        return self.codebook_size

    def __len__(self) -> int:
        return self.codebook_size

    def unit_length(self, token_id: int) -> int:
        """Number of reduced units (leaves) a token spans. 1 for a leaf, sum of children else."""
        token_id = int(token_id)
        if token_id < self.n_base:
            return 1
        if token_id in self._unit_len_cache:
            return self._unit_len_cache[token_id]
        l, r = self.children[token_id]
        v = self.unit_length(l) + self.unit_length(r)
        self._unit_len_cache[token_id] = v
        return v


# --------------------------------------------------------------------------- #
# fit: pool reduced unit streams across BUILD, frequency-greedy BPE merge
# --------------------------------------------------------------------------- #
def _stream_reduced_units(db, asset, cfg, table, max_windows):
    """Yield (reduced-unit ids, per-unit raw-bar coverage) pairs, ONE per window-chunk run, for one
    asset.

    A window from db_stream is a (W,6) all-finite vol-unit slice; we encode it to base symbols and
    numerosity-reduce WITHIN the window (windows never straddle a gap -> reduction is run-local).
    `bars` (the numerosity run-lengths, summing to the window's bar count) is yielded so callers can
    accumulate the token stream's true raw-bar coverage for the A1 per-bar null (TASK_796).
    """
    for chunk in DS.iter_windows(db, asset, cfg, max_windows=max_windows):
        X = chunk["X"]                              # (n, W, 6) f32
        for w in range(X.shape[0]):
            syms = BA.encode_symbols(X[w], table)   # (W,) base symbols
            units, bars = numerosity_reduce(syms)
            if units.size >= 1:
                yield units, bars


def fit_bpe(cfg, table, build_assets, max_windows_per_asset, p_min, max_merges, verbose=True):
    """Frequency-greedy BPE/RePair over the POOLED numerosity-reduced base-symbol stream.

    Pools every BUILD asset's reduced unit sequences (kept as a list of small int arrays — sequences,
    NOT the raw bar stream, so this is already ~window-compressed), then repeatedly merges the most
    frequent adjacent reduced-unit PAIR into a new composite token, provided the pair's pooled count
    is >= p_min (the viable-token floor, design §2.4). Stops at max_merges or when no pair clears
    p_min. Pair counts are maintained incrementally (RePair-style) so each round is cheap.

    Returns a frozen BPEVocab.
    """
    n_base = n_base_tagged(table)

    # pool reduced sequences across build assets (list of np.int64 arrays -> python lists for mutation)
    seqs = []
    for a in build_assets:
        for units, _bars in _stream_reduced_units(cfg.db, a, cfg, table, max_windows_per_asset):
            seqs.append(list(int(u) for u in units))
    if verbose:
        n_units = sum(len(s) for s in seqs)
        print(f"[L1] pooled {len(seqs)} reduced sequences, {n_units} reduced units "
              f"(n_base leaf alphabet = {n_base})", flush=True)

    # adjacency pair counts over all sequences
    def _count_pairs(sequences):
        cnt = Counter()
        for s in sequences:
            for i in range(len(s) - 1):
                cnt[(s[i], s[i + 1])] += 1
        return cnt

    pair_counts = _count_pairs(seqs)
    merges = []

    for m in range(int(max_merges)):
        if not pair_counts:
            break
        # most frequent adjacent pair
        best_pair, best_n = max(pair_counts.items(), key=lambda kv: kv[1])
        if best_n < int(p_min):
            if verbose:
                print(f"[L1] stop: top pair count {best_n} < p_min {p_min} at merge {m}", flush=True)
            break
        new_id = n_base + len(merges)
        merges.append(best_pair)
        # apply the merge in every sequence, incrementally updating pair_counts (RePair idiom).
        _apply_merge(seqs, best_pair, new_id, pair_counts)
        if verbose and (m < 5 or (m + 1) % 50 == 0):
            print(f"[L1] merge {m}: {best_pair} -> {new_id}  (count {best_n}, "
                  f"vocab {n_base + len(merges)})", flush=True)

    if verbose:
        print(f"[L1] BPE done: {len(merges)} merges, codebook_size {n_base + len(merges)}", flush=True)
    return BPEVocab(merges, table)


def _apply_merge(seqs, pair, new_id, pair_counts):
    """In-place replace every adjacent occurrence of `pair` with `new_id` across all sequences,
    updating the incremental `pair_counts` for the neighbours touched (classic BPE/RePair update).
    Left-to-right, non-overlapping (the canonical BPE application order)."""
    a, b = pair
    pair_counts.pop(pair, None)
    for s in seqs:
        i = 0
        out = []
        n = len(s)
        while i < n:
            if i < n - 1 and s[i] == a and s[i + 1] == b:
                # decrement the pairs that straddle the merged region (left & right neighbours)
                if out:
                    left = out[-1]
                    pair_counts[(left, a)] -= 1
                    if pair_counts[(left, a)] <= 0:
                        pair_counts.pop((left, a), None)
                    pair_counts[(left, new_id)] += 1
                if i + 2 < n:
                    right = s[i + 2]
                    pair_counts[(b, right)] -= 1
                    if pair_counts[(b, right)] <= 0:
                        pair_counts.pop((b, right), None)
                    pair_counts[(new_id, right)] += 1
                out.append(new_id)
                i += 2
            else:
                out.append(s[i])
                i += 1
        s[:] = out


# --------------------------------------------------------------------------- #
# encode: EXACT BPE MERGE-RANK serving (NOT greedy longest-match, design §2.6)
# --------------------------------------------------------------------------- #
def encode_units_bpe(units, vocab: BPEVocab) -> np.ndarray:
    """Apply the frozen merge list to a reduced-unit sequence in MERGE-RANK order (rank 0 first),
    exactly as fit (serving == training segmentation). Returns token-id array (1-D int64).

    Repeatedly: find the adjacent pair with the lowest merge rank present in the sequence, merge ALL
    its non-overlapping occurrences, repeat until no mergeable pair remains. This is the canonical
    BPE serving algorithm (deterministic, rescale-invariant — A0 holds because identical base symbols
    -> identical units -> identical merge path)."""
    seq = [int(u) for u in np.asarray(units, dtype=np.int64).ravel()]
    if len(seq) < 2 or not vocab.ranks:
        return np.asarray(seq, dtype=np.int64)
    ranks = vocab.ranks
    n_base = vocab.n_base
    while True:
        # lowest-rank mergeable adjacent pair
        best_rank = None
        best_pair = None
        for i in range(len(seq) - 1):
            pr = ranks.get((seq[i], seq[i + 1]))
            if pr is not None and (best_rank is None or pr < best_rank):
                best_rank = pr
                best_pair = (seq[i], seq[i + 1])
        if best_pair is None:
            break
        new_id = n_base + best_rank
        a, b = best_pair
        out = []
        i = 0
        n = len(seq)
        while i < n:
            if i < n - 1 and seq[i] == a and seq[i + 1] == b:
                out.append(new_id)
                i += 2
            else:
                out.append(seq[i])
                i += 1
        seq = out
    return np.asarray(seq, dtype=np.int64)


# --------------------------------------------------------------------------- #
# streaming encoders (token-id chunks for the O(codebook) battery folds)
# --------------------------------------------------------------------------- #
def _encode_asset_token_chunks(db, asset, cfg, vocab, max_windows, control=None, seed=0):
    """Yield token-id chunks (1-D int64) for one asset, ONE chunk per source window.

    control=None      : real vol-unit substrate.
    control='shuffle' : donor phase_shuffle_returns surrogate on the window features (A8 null #1).
    control='random_merge' : permute the reduced-unit sequence before BPE (destroys good adjacency,
                      A8 null #2 — isolates good merges from any merges; design §4 control (ii)).
    Streamed: never materializes the asset's full token stream on the host."""
    from tools.tokenizer_c0.data import phase_shuffle_returns  # donor (data.py:165), used for control
    table = vocab.table
    rng = np.random.default_rng(seed + int(asset))
    for chunk in DS.iter_windows(db, asset, cfg, max_windows=max_windows):
        X = chunk["X"]                              # (n, W, 6)
        for w in range(X.shape[0]):
            feats = X[w]
            if control == "shuffle":
                fvalid = np.ones(feats.shape[0], dtype=bool)
                feats = phase_shuffle_returns(feats.astype(np.float32), fvalid, rng)
            syms = BA.encode_symbols(feats, table)
            units, _bars = numerosity_reduce(syms)
            if control == "random_merge":
                units = units[rng.permutation(units.size)]
            ids = encode_units_bpe(units, vocab)
            if ids.size:
                yield ids


# --------------------------------------------------------------------------- #
# A2 reconstruction template — canonical vol-unit move per base symbol
# --------------------------------------------------------------------------- #
def fit_move_template(cfg, table, build_assets, max_windows_per_asset):
    """Online per-base-symbol mean of the mv (vol-unit move) channel — the reconstruction template.

    A token's reconstruction of the mv path replays each covered bar's base-symbol canonical move.
    Accumulated streaming with two O(N_SYMBOLS) accumulators (sum, count) — never stores the stream.
    Returns mv_mean : (N_SYMBOLS,) float64 (the canonical move for each base symbol; 0 where unseen).
    """
    ns = int(BA.N_SYMBOLS(table))
    s_sum = np.zeros(ns, dtype=np.float64)
    s_cnt = np.zeros(ns, dtype=np.int64)
    for a in build_assets:
        for chunk in DS.iter_windows(cfg.db, a, cfg, max_windows=max_windows_per_asset):
            X = chunk["X"]
            for w in range(X.shape[0]):
                feats = X[w]
                syms = BA.encode_symbols(feats, table)          # (W,)
                mv = feats[:, 0].astype(np.float64)             # vol-unit move channel
                np.add.at(s_sum, syms, mv)
                np.add.at(s_cnt, syms, 1)
    mv_mean = np.where(s_cnt > 0, s_sum / np.maximum(s_cnt, 1), 0.0)
    return mv_mean


def _recon_pairs(db, asset, cfg, table, mv_mean, max_windows):
    """Yield PER-BAR (recon_mv, true_mv) increment pairs for A2, ONE per window (TASK_796).

    Reconstruction lives in the vol-unit move space (mv = logret/sigma, std ~1). A2 now scores each
    bar additively (|recon_mv - true_mv| <= abs_tol) — well-posed at zero-crossings and sign-sensitive,
    unlike the old cumsum path-level log-ratio (which read 0.071 purely as a missing-offset artifact at
    |true_path|->0). We therefore yield the RAW signed per-bar increments (recon = base-symbol canonical
    move replayed bar-by-bar, true = the actual mv channel), NOT cumulative paths and NO +offset hack.
    NEVER materializes all windows."""
    for chunk in DS.iter_windows(db, asset, cfg, max_windows=max_windows):
        X = chunk["X"]
        for w in range(X.shape[0]):
            feats = X[w]
            true_mv = feats[:, 0].astype(np.float64)         # signed vol-unit move per bar
            syms = BA.encode_symbols(feats, table)
            recon_mv = mv_mean[syms].astype(np.float64)      # canonical per-bar move per base symbol
            # per-bar signed increments (NOT cumsum) for mode="additive".
            yield recon_mv, true_mv


# --------------------------------------------------------------------------- #
# A0 single-window tokenize_fn (raw OHLC (W,4) -> single token id)
# --------------------------------------------------------------------------- #
def make_tokenize_window_fn(cfg, vocab: BPEVocab):
    """Build a tokenize_window_fn(ohlc(W,4)) -> int for eval A0. Runs the FULL member path on ONE
    window (volunit -> base symbols -> reduce -> BPE) and returns the FIRST token id (a stable
    single-id summary; rescale-invariance must make it byte-identical under c->lam*c).

    The window's OHLC is treated as one contiguous dt==60 run; features_for_run yields W-1 feature
    bars. A0 only asks for byte-identity under rescale, which holds for ANY deterministic reduction
    of the (rescale-invariant) feature window."""
    table = vocab.table

    def _fn(ohlc):
        ohlc = np.asarray(ohlc, dtype=np.float64)
        W = ohlc.shape[0]
        ts = np.arange(W, dtype=np.int64) * C.BAR_SECONDS
        o, h, l, c = ohlc[:, 0], ohlc[:, 1], ohlc[:, 2], ohlc[:, 3]
        feats, _close, _ts, finite = features_for_run(ts, o, h, l, c, cfg)
        if feats.shape[0] == 0:
            return 0
        # keep finite bars only (A0 windows are clean synthetic price paths)
        feats = feats[finite] if not finite.all() else feats
        if feats.shape[0] == 0:
            return 0
        syms = BA.encode_symbols(feats, table)
        units, _bars = numerosity_reduce(syms)
        ids = encode_units_bpe(units, vocab)
        return int(ids[0]) if ids.size else 0

    return _fn


def _sample_a0_windows(cfg, table, build_assets, n=24):
    """Pull a SMALL diagnostic sample of raw-price OHLC windows for A0 (a unit check, not a fold).

    A0 needs raw OHLC windows, but db_stream yields vol-unit features. We instead synthesize clean
    rescale-test windows from a real run's CLOSE path geometry is unnecessary — A0 only requires that
    the SAME window tokenizes identically under rescale. We reconstruct approximate OHLC from the raw
    close path of real windows (chunk['C']) by using close for o/h/l/c offsets; that is sufficient for
    the byte-identity test (the substrate is rescale-invariant for any real OHLC). Returns list[(W,4)].
    """
    wins = []
    W = int(cfg.window)
    for a in build_assets:
        for chunk in DS.iter_windows(cfg.db, a, cfg, max_windows=64):
            Cp = chunk["C"]                          # (n, W) raw close path
            for r in range(Cp.shape[0]):
                close = Cp[r].astype(np.float64)
                if close.size < 4 or not np.all(np.isfinite(close)) or np.any(close <= 0):
                    continue
                # build a plausible OHLC window from the real close path: open = prev close,
                # high/low = small symmetric envelope. Rescale-invariance does not depend on the
                # envelope width, only on it being a fixed multiplicative function of price.
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
# length-distribution fold for A9 (streamed, O(codebook))
# --------------------------------------------------------------------------- #
def _length_counts(db, assets, cfg, vocab, max_windows):
    """Stream every encoded token, tally token UNIT-LENGTH counts into a dict (A9 input). Host memory
    is O(distinct lengths) — bounded, never O(windows)."""
    lc = {}
    for a in assets:
        for ids in _encode_asset_token_chunks(db, a, cfg, vocab, max_windows, control=None):
            for tid in ids.tolist():
                L = vocab.unit_length(int(tid))
                lc[L] = lc.get(L, 0) + 1
    return lc


# --------------------------------------------------------------------------- #
# CLI / orchestration
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="L1 BPE-delta candle language (CPU, branch C).")
    p.add_argument("--db", type=str, default=None, help="DB path (default: config.db_path()).")
    p.add_argument("--out", type=str,
                   default="/home/lukas/Trader/runs/789_token_c0/L1_BPE_RESULT.md")
    p.add_argument("--max-windows-per-asset", type=int, default=20000)
    p.add_argument("--window", type=int, default=C.DEFAULT_WINDOW)
    p.add_argument("--p-min", type=int, default=50, help="viable-token floor (pooled pair count).")
    p.add_argument("--max-merges", type=int, default=512, help="cap on BPE merges (vocab size).")
    p.add_argument("--ratio-tol", type=float, default=0.05,
                   help="A2 per-bar log-ratio tolerance (mode=ratio; unused in additive mode).")
    p.add_argument("--abs-tol", type=float, default=0.25,
                   help="A2 additive vol-unit tolerance (TASK_796 mode=additive; shared panel default).")
    # TASK_796 scale-up: explicit asset lists + per-asset transfer matrix for the 150/150 run.
    p.add_argument("--build-assets", nargs="+", type=int, default=None,
                   help="explicit BUILD asset ids (default: config.build_assets).")
    p.add_argument("--holdout-assets", nargs="+", type=int, default=None,
                   help="explicit HOLDOUT asset ids (default: config.holdout_assets).")
    p.add_argument("--transfer-matrix", action="store_true",
                   help="compute the per-asset A8 A-by-B transfer matrix (per_asset_matrix=True).")
    return p.parse_args(argv)


def build_cfg(args) -> "C.CandleConfig":
    """Assemble a CandleConfig from CLI (db override via --db or $CANDLE_DB through config.db_path)."""
    kw = dict(window=int(args.window))
    if args.db:
        kw["db"] = args.db
    return C.CandleConfig(**kw)


def main(argv=None):
    args = parse_args(argv)
    cfg = build_cfg(args)
    t0 = time.time()
    mwpa = int(args.max_windows_per_asset)

    print(f"[L1] db={cfg.db}  window={cfg.window}  max_windows/asset={mwpa}  "
          f"p_min={args.p_min}  max_merges={args.max_merges}", flush=True)
    if not DS.quick_check(cfg.db):
        print(f"[L1] WARNING: PRAGMA quick_check did not return 'ok' for {cfg.db}", flush=True)

    # TASK_796 scale-up: explicit CLI asset lists override config defaults (bare run == unchanged).
    build_assets = list(tuple(args.build_assets) if args.build_assets else cfg.build_assets)
    holdout_assets = list(tuple(args.holdout_assets) if args.holdout_assets is not None
                          else cfg.holdout_assets)

    # ----- 1) fit the shared base alphabet on a POOLED vol-unit sample (bounded) ----- #
    print("[L1] fitting base alphabet on pooled build sample...", flush=True)
    pooled = []
    pooled_rows = 0
    pooled_cap = 2_000_000                              # cap pooled fit rows (bounded RAM)
    for a in build_assets:
        for chunk in DS.iter_windows(cfg.db, a, cfg, max_windows=mwpa):
            X = chunk["X"]                              # (n, W, 6)
            flat = X.reshape(-1, C.N_FEAT)
            pooled.append(flat)
            pooled_rows += flat.shape[0]
            if pooled_rows >= pooled_cap:
                break
        if pooled_rows >= pooled_cap:
            break
    if not pooled:
        raise RuntimeError("[L1] no windows pooled for alphabet fit — check DB / assets / window.")
    feats_pooled = np.concatenate(pooled, axis=0)
    table = BA.fit_alphabet(feats_pooled, cfg)
    del pooled, feats_pooled
    print(f"[L1] base alphabet fitted: N_SYMBOLS={BA.N_SYMBOLS(table)}  "
          f"n_base_tagged={n_base_tagged(table)}", flush=True)

    # ----- 2) fit BPE merges pooled across BUILD ----- #
    vocab = fit_bpe(cfg, table, build_assets, mwpa,
                    p_min=int(args.p_min), max_merges=int(args.max_merges))
    cb = vocab.codebook_size

    # ----- 3) A2 reconstruction template ----- #
    print("[L1] fitting reconstruction (mv) template...", flush=True)
    mv_mean = fit_move_template(cfg, table, build_assets, mwpa)

    # ----- 4) eval battery (all streamed) ----- #
    # A0 — rescale invariance (small diagnostic sample, unit check)
    print("[L1] A0 rescale-invariance...", flush=True)
    tok_fn = make_tokenize_window_fn(cfg, vocab)
    a0_windows = _sample_a0_windows(cfg, table, build_assets, n=24)
    a0 = EB.a0_rescale_invariance(tok_fn, a0_windows, lambdas=(0.01, 100.0))

    # A1 — MDL over the FULL build stream (online bincount)
    print("[L1] A1 MDL...", flush=True)
    n_base = n_base_tagged(table)
    null_bits_per_bar = float(np.log2(max(int(BA.N_SYMBOLS(table)), 2)))
    # count tokens (= boundaries) for the boundary charge, streamed.
    def _build_token_iter():
        for a in build_assets:
            yield from _encode_asset_token_chunks(cfg.db, a, cfg, vocab, mwpa, control=None)
    n_tokens = 0
    for ids in _build_token_iter():
        n_tokens += int(ids.size)
    # TASK_796: true raw-bar coverage of the build stream = sum of numerosity-reduction run-lengths
    # (the `_bars` value currently discarded at every reduce call site). BPE merges group reduced
    # units but do NOT change bar coverage, so summing run_lens over the reduced stream IS the token
    # stream's bar coverage. Charged in a1_mdl as n_bars_covered so the per-bar null is like-for-like.
    n_bars_covered = 0
    for a in build_assets:
        for units, bars in _stream_reduced_units(cfg.db, a, cfg, table, mwpa):
            n_bars_covered += int(bars.sum())
    codebook_bits = float(cb) * float(np.log2(max(cb, 2)))   # crude vocab-shipping cost
    a1 = EB.a1_mdl(_build_token_iter(), codebook_size=cb,
                   null_bits_per_bar=null_bits_per_bar,
                   n_boundaries=n_tokens, codebook_bits=codebook_bits,
                   n_bars_covered=n_bars_covered)

    # A2 — reconstruction (streamed PER-BAR signed increment pairs over build; additive mode).
    print("[L1] A2 reconstruction...", flush=True)
    def _recon_iter():
        for a in build_assets:
            yield from _recon_pairs(cfg.db, a, cfg, table, mv_mean, mwpa)
    a2 = EB.a2_reconstruction(_recon_iter(), mode="additive", abs_tol=float(args.abs_tol))

    # A8 — cross-asset transfer (build vocab is already fit; re-encode build/holdout + control)
    print("[L1] A8 cross-asset transfer...", flush=True)

    def build_vocab_fn(_assets=None):
        return vocab                                   # frozen vocab fit on BUILD

    def encode_assets_fn(_vocab, assets, control):
        for a in assets:
            yield from _encode_asset_token_chunks(cfg.db, a, cfg, _vocab, mwpa, control=control)

    a8 = None
    if holdout_assets:
        a8 = EB.a8_cross_asset_transfer(
            build_vocab_fn, encode_assets_fn,
            build_assets=build_assets, holdout_assets=holdout_assets,
            controls=("shuffle", "random_merge"), exclude_top_k=8,
            per_asset_matrix=bool(args.transfer_matrix),
        )

    # A9 — length non-degeneracy (streamed unit-length tally)
    print("[L1] A9 length non-degeneracy...", flush=True)
    lc = _length_counts(cfg.db, build_assets, cfg, vocab, mwpa)
    a9 = EB.a9_length_nondegeneracy(lc)

    # A8 metric power harnesses (synthetic, no DB/GPU) — stored on the a8 dict (so render_a8_lines
    # surfaces them) and in meta. None when no holdout (a8 is None / empty).
    a8_power = EB.run_a8_power()
    if a8:
        a8["a8_power"] = a8_power

    wall = time.time() - t0

    # ----- 5) verdict + write result ----- #
    # A8 go/no-go via the SHARED §186-aligned verdict: gate on the FAIR random-codebook null on BOTH
    # unigram AND bigram axes; the entropy-confounded substrate controls are reported, not gated.
    a8_ok = EB.a8_verdict(a8) if a8 else True
    passes = [
        bool(a0["pass"]),
        bool(a1["compression_ratio"] > 1.0),
        bool(a2["frac_within_tol"] >= 0.50),
        a8_ok,
        bool(a9["pass"]),
    ]
    verdict = "PROCEED" if all(passes) else "REVIEW"

    sections = [
        ("Configuration", [
            f"- DB: `{cfg.db}`",
            f"- Window {cfg.window} bars | max windows/asset {mwpa}",
            f"- Base alphabet N_SYMBOLS = {BA.N_SYMBOLS(table)} | leaf alphabet (tagged) = {n_base}",
            f"- BPE merges = {len(vocab.merges)} | codebook_size = {cb}",
            f"- p_min = {args.p_min} | max_merges = {args.max_merges} | "
            f"len-buckets = {N_LEN_BUCKETS} (edges {_LEN_EDGES})",
            f"- Build assets: {', '.join(str(x) for x in build_assets)}",
            f"- Holdout assets: {', '.join(str(x) for x in holdout_assets)}",
            "- Serving = EXACT BPE merge-rank order (not greedy longest-match).",
        ]),
        ("A0 — rescale invariance (hard pre-gate)", [
            f"- agree fraction: {a0['agree']:.4f} over {a0['n_trials']} trials "
            f"(lambdas {a0['lambdas']})",
            f"- **PASS: {bool(a0['pass'])}**",
        ]),
        ("A1 — MDL / description length (full build stream)", [
            f"- bits per candle: {a1['bits_per_candle']:.4f}",
            f"- entropy bits/token: {a1['entropy_bits_per_token']:.4f}",
            f"- null bits/bar (base-symbol entropy): {null_bits_per_bar:.4f}",
            f"- tokens: {a1['n_tokens']} | codebook_bits: {a1['codebook_bits']:.1f}",
            f"- **compression ratio vs null: {a1['compression_ratio']:.4f}** "
            f"(>1 = cheaper than per-bar null)",
        ]),
        ("A2 — reconstruction fidelity (mv vol-unit, per-bar additive)", [
            f"- mode: {a2['mode']} | abs_tol (vol-units): {a2['abs_tol']}",
            f"- frac within tol: {a2['frac_within_tol']:.4f}",
            f"- mean abs err (vol-units): {a2['mean_abs_err']:.6f} | MSE: {a2['mean_log_ratio_mse']:.6f}",
            f"- bars: {a2['n_bars']} | pairs: {a2['n_pairs']}",
        ]),
        ("A8 — cross-asset transfer (build vocab -> holdout)", (
            EB.render_a8_lines(a8) + EB.render_transfer_matrix(a8)
            if a8 else ["- skipped (no holdout assets configured)."]
        )),
        ("A9 — length non-degeneracy", [
            f"- median token length (units): {a9['median_len']:.2f}",
            f"- max length-bucket share: {a9['max_token_share']:.4f}",
            f"- all length-1: {bool(a9['all_len1'])} | distinct lengths: {a9['n_lengths']}",
            f"- **PASS: {bool(a9['pass'])}**",
        ]),
        ("Gate summary", [
            f"- A0 pass: {passes[0]}",
            f"- A1 compression>1: {passes[1]}",
            f"- A2 frac_within>=0.50: {passes[2]}",
            f"- A8 verdict (random-codebook null, uni+bigram): "
            f"{a8_ok if a8 else 'n/a (no holdout)'}",
            f"- A9 non-degenerate: {passes[4]}",
            f"- wallclock: {wall:.0f}s",
        ]),
    ]

    meta = dict(
        member="L1_BPE",
        db=cfg.db,
        window=int(cfg.window),
        max_windows_per_asset=mwpa,
        p_min=int(args.p_min),
        max_merges=int(args.max_merges),
        n_symbols=int(BA.N_SYMBOLS(table)),
        n_base_tagged=int(n_base),
        n_merges=int(len(vocab.merges)),
        codebook_size=int(cb),
        len_buckets=int(N_LEN_BUCKETS),
        a0=a0, a1=a1, a2=a2, a8=(a8 if a8 is not None else {}), a9=a9,
        a8_power=a8_power,
        wallclock_s=float(wall),
    )

    RW.write_result(args.out, "TASK_789 L1 BPE-delta — Candle Language (branch C)",
                    sections, verdict, meta)

    print("\n========= L1 BPE SUMMARY =========", flush=True)
    print(f"codebook_size {cb}  merges {len(vocab.merges)}", flush=True)
    print(f"A0 agree {a0['agree']:.4f} pass {bool(a0['pass'])}", flush=True)
    print(f"A1 bits/candle {a1['bits_per_candle']:.4f} compression {a1['compression_ratio']:.4f}",
          flush=True)
    print(f"A2 frac_within {a2['frac_within_tol']:.4f}", flush=True)
    if a8:
        print(f"A8 transfer_bits {a8['transfer_bits']:.4f} verdict {a8_ok} "
              f"(rand_cb {bool(a8['beats_random_codebook'])}, "
              f"rand_cb_bigram {bool(a8.get('beats_random_codebook_bigram'))}, "
              f"power_ok {bool(a8_power.get('power_ok'))})",
              flush=True)
    print(f"A9 median_len {a9['median_len']:.2f} pass {bool(a9['pass'])}", flush=True)
    print(f"VERDICT {verdict}  wall {wall:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
