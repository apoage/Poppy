"""
L2 ABBA — Adaptive Brownian-Bridge-based symbolic Aggregation over the vol-unit substrate
(candle-language panel member, branch C / TASK_789 / design 01 §6, build plan 04 §4).

CPU-only. No JAX, no GPU. Pure numpy + a tiny self-contained k-means.

----------------------------------------------------------------------------------------------
WHAT ABBA IS (Elsworth & Güttel 2020; mapped to our SPEC in design 03 / _raw_research):
  1. COMPRESSION (adaptive polygonal segmentation). Walk the polygonal chain of the cumulative
     vol-unit move (the `mv` channel integrated -> a scale-free "log-price-in-vol-units" path) and
     greedily extend a straight piece for as long as the squared reconstruction error of the chord
     stays <= a per-piece tolerance. Emit one (len, inc) tuple per piece: len = #bars in the piece,
     inc = net rise of the chord over the piece (in vol-units). Piece LENGTH is data-adaptive ->
     variable-length segments with ONE tolerance knob (the SPEC's variable-length requirement).
  2. DIGITIZATION (clustering). Standardize the pooled (len, inc) tuple set, apply ABBA's `scl`
     weight so SHAPE (inc) dominates DURATION (len) — design wants price-shape tokens, not duration
     tokens — and k-means into k clusters. Each cluster centre is a canonical (len, inc) template;
     its index is the symbol id.

WHY IT TRANSFERS / IS RESCALE-INVARIANT (eval A0, the load-bearing property):
  We segment the VOL-UNIT `mv` channel, never raw price. A pure price rescale c -> lambda*c leaves
  every vol-unit feature byte-identical (config.py §RESCALE INVARIANCE, volunit.py), so the cumulative
  vol-unit path, every (len, inc) tuple, the standardization, and the cluster assignment are ALL
  byte-identical -> the symbol stream is byte-identical. A0 passes by construction.
  The codebook (means/std + cluster centres) is fitted ONCE on the POOLED build-asset tuple set, so
  the same shape lands on the same symbol across assets at extreme price levels (transfer is
  STRUCTURAL — a property of the shared vol-unit encoding + shared codebook — NOT learned coupling).

PIPELINE (this file):
  fit:  stream build assets -> vol-unit feats (db_stream) -> ABBA pieces (per run) -> pooled tuples
        -> standardize (scl-weighted) -> k-means -> frozen AbbaVocab.
  encode: stream an asset -> feats -> ABBA pieces -> standardize with the FROZEN stats -> nearest
        cluster -> token-id stream (one id per piece). Lengths (in bars) ride alongside for A1/A9.

EVAL (TASK_792 non-predictive battery): A0 (rescale-invariance, byte-identical re-encode of a
  rescaled stream), A1 (corrected MDL, null = base-alphabet entropy/bar), A2 (reconstruction of the
  vol-unit move path from (len, inc) tokens), A8 (cross-asset transfer vs phase_shuffle control),
  A9 (length non-degeneracy). Then result_writer.write_result -> runs/789_token_c0/.

LEAK/OOM (HARD-REQ #2/#3): we stream per asset via db_stream.iter_runs (paged cursor, bounded RAM),
  reduce A1/A8 ONLINE with O(codebook) np.bincount accumulators (never O(windows)/O(tokens) host
  lists), cap the fit tuple pool, and never materialize a whole asset's token stream on the host.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

from .. import config as C
from ..base_alphabet import N_SYMBOLS, encode_symbols, fit_alphabet
from ..db_stream import iter_runs, quick_check, survey_row_counts
from ..volunit import features_for_run
from .. import eval_battery as EB
from ..result_writer import write_result

# Donor null for A8 'shuffle' control (data.py:165-203) — structure-destroying surrogate.
from tools.tokenizer_c0.data import phase_shuffle_returns  # noqa: E402


# --------------------------------------------------------------------------- #
# tunables (ABBA knobs; design 03 borrows c/d)
# --------------------------------------------------------------------------- #
# Default symbol alphabet size for the (len, inc) clustering. Chosen of the same order as the base
# alphabet (288) so A1's MDL comparison vs the per-bar base-symbol null is like-for-like in scale.
DEFAULT_K = 256
# ABBA per-piece squared-error tolerance on the vol-unit cumulative path. The cumulative path moves
# ~O(1) vol-units per bar, so a tolerance of a few vol-units^2 yields multi-bar pieces (the SPEC's
# variable-length tokens) without collapsing to per-bar (len==1) pieces.
DEFAULT_TOL = 0.5
# ABBA 'scl': weight on the (standardized) LENGTH dimension relative to the INCREMENT dimension in
# the k-means metric. scl<1 makes SHAPE (inc) dominate DURATION (len) — design wants shape tokens.
DEFAULT_SCL = 0.3
# Max pooled (len, inc) tuples kept for the k-means fit (bounded RAM; reservoir-capped).
DEFAULT_MAX_FIT_TUPLES = 600_000
# Min pieces required before we trust a k-means fit.
_MIN_FIT_TUPLES = 64


# --------------------------------------------------------------------------- #
# ABBA compression: adaptive polygonal segmentation of the vol-unit move path
# --------------------------------------------------------------------------- #
def _cumulative_move(mv: np.ndarray) -> np.ndarray:
    """Cumulative vol-unit move path P (the polygonal chain we segment). P[0]=0, P[k]=sum(mv[:k]).
    This is the scale-free analogue of ABBA's cumulative log-price chain; built from the `mv`
    channel (config.CHANNELS[0]) so it is rescale-invariant by construction."""
    mv = np.asarray(mv, dtype=np.float64).reshape(-1)
    P = np.empty(mv.size + 1, dtype=np.float64)
    P[0] = 0.0
    np.cumsum(mv, out=P[1:])
    return P


def _segment_run(P: np.ndarray, tol: float):
    """Greedy adaptive segmentation of one cumulative path P (length n+1 for n move bars).

    Walk forward extending a straight chord from the current anchor; a piece ends at the LAST bar
    for which every interior point's squared deviation from the chord stays <= tol. Emits one
    (length, increment) tuple per piece, where length = #bars covered and increment = chord rise
    (P[end] - P[start]) in vol-units. Mirrors ABBA's tol-bounded greedy extension (research §borrow).

    Returns (lengths(P,)int64, incs(P,)float64) — P = number of pieces. No piece is ever len 0; the
    final partial piece is always flushed so the whole run is covered (A1 codes every bar).
    """
    n = P.size - 1  # number of move bars
    if n <= 0:
        return np.zeros((0,), np.int64), np.zeros((0,), np.float64)
    tol = float(tol)
    lengths = []
    incs = []
    start = 0
    while start < n:
        # extend [start .. end] (inclusive end index into P, end>=start+1) while the chord fits.
        end = start + 1
        last_ok = end
        while end <= n:
            # chord from P[start] to P[end]; check max squared deviation of interior points.
            seg_len = end - start
            if seg_len <= 1:
                last_ok = end
                end += 1
                continue
            x = np.arange(seg_len + 1, dtype=np.float64)  # 0..seg_len local index over P[start:end+1]
            y = P[start:end + 1]
            slope = (y[-1] - y[0]) / float(seg_len)
            approx = y[0] + slope * x
            err = np.max((y - approx) ** 2)
            if err <= tol:
                last_ok = end
                end += 1
            else:
                break
        seg_len = last_ok - start
        lengths.append(int(seg_len))
        incs.append(float(P[last_ok] - P[start]))
        start = last_ok
    return np.asarray(lengths, dtype=np.int64), np.asarray(incs, dtype=np.float64)


def _pieces_for_run(ts, o, h, l, c, cfg):
    """Vol-unit features for ONE dt==60 run -> ABBA (len, inc) pieces over the finite `mv` path.

    Only the maximal all-finite prefix-runs of the mv channel are segmented (a non-finite feature
    bar breaks the path; we never bridge a non-finite gap). Returns (lengths, incs) concatenated
    across the finite subsegments of this run, plus the finite mv array used (for A2 reconstruction).
    """
    tol = float(getattr(cfg, "abba_tol", DEFAULT_TOL))
    feats, close, ts_out, finite = features_for_run(ts, o, h, l, c, cfg)
    if feats.shape[0] == 0:
        return (np.zeros((0,), np.int64), np.zeros((0,), np.float64), np.zeros((0,), np.float64))
    mv_all = feats[:, 0].astype(np.float64)
    fin = np.asarray(finite, dtype=bool)
    all_len, all_inc = [], []
    # segment each maximal finite subsegment independently (no bridging across non-finite bars).
    i = 0
    n = mv_all.size
    while i < n:
        if not fin[i]:
            i += 1
            continue
        j = i
        while j < n and fin[j]:
            j += 1
        mv_seg = mv_all[i:j]
        if mv_seg.size >= 1:
            P = _cumulative_move(mv_seg)
            lens, incs = _segment_run(P, tol)
            if lens.size:
                all_len.append(lens)
                all_inc.append(incs)
        i = j
    if not all_len:
        return (np.zeros((0,), np.int64), np.zeros((0,), np.float64),
                mv_all[fin].astype(np.float64))
    return (np.concatenate(all_len), np.concatenate(all_inc), mv_all[fin].astype(np.float64))


# --------------------------------------------------------------------------- #
# digitization: standardize (scl-weighted) + tiny k-means over (len, inc) tuples
# --------------------------------------------------------------------------- #
def _standardize(tuples: np.ndarray, mean: np.ndarray, std: np.ndarray, scl: float) -> np.ndarray:
    """Map raw (len, inc) tuples into the scl-weighted standardized clustering space.
    Column 0 = length, column 1 = increment. scl down-weights length so increment (shape) dominates
    the k-means distance (ABBA 'scl'). std is floored (CLAUDE.md jnp.maximum guard analogue)."""
    z = (np.asarray(tuples, dtype=np.float64) - mean) / std
    w = np.array([float(scl), 1.0], dtype=np.float64)  # length weighted by scl, inc weight 1
    return z * w


def _kmeans(X: np.ndarray, k: int, n_iter: int = 50, seed: int = 0):
    """Self-contained Lloyd k-means (no scipy/sklearn dependency). X: (N, 2) standardized tuples.
    Returns (centres(k,2), labels(N,)). k-means++ seeding for stability; empty clusters re-seeded to
    the farthest point. Deterministic given seed -> reproducible codebook (A0 re-encode stability)."""
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    k = int(min(k, N))
    if k <= 0:
        return np.zeros((0, X.shape[1]), np.float64), np.zeros((0,), np.int64)
    # k-means++ init
    centres = np.empty((k, X.shape[1]), dtype=np.float64)
    first = int(rng.integers(N))
    centres[0] = X[first]
    d2 = np.sum((X - centres[0]) ** 2, axis=1)
    for ci in range(1, k):
        total = float(d2.sum())
        if total <= 0:
            centres[ci] = X[int(rng.integers(N))]
        else:
            probs = d2 / total
            centres[ci] = X[int(rng.choice(N, p=probs))]
        nd = np.sum((X - centres[ci]) ** 2, axis=1)
        d2 = np.minimum(d2, nd)

    labels = np.zeros(N, dtype=np.int64)
    for _ in range(int(n_iter)):
        # assign: nearest centre (chunked to bound the (N,k) distance matrix RAM).
        new_labels = _assign(X, centres)
        if np.array_equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels
        # update
        for ci in range(k):
            mask = labels == ci
            if np.any(mask):
                centres[ci] = X[mask].mean(axis=0)
            else:
                # re-seed an empty cluster to the current farthest point from its centre.
                dd = np.sum((X - centres[ci]) ** 2, axis=1)
                centres[ci] = X[int(np.argmax(dd))]
    return centres, labels


def _assign(X: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """Nearest-centre labels for X, chunked so the (chunk, k) distance block stays bounded."""
    N = X.shape[0]
    k = centres.shape[0]
    if k == 0 or N == 0:
        return np.zeros(N, dtype=np.int64)
    out = np.empty(N, dtype=np.int64)
    chunk = max(1, min(N, 1 << 16))
    c2 = np.sum(centres ** 2, axis=1)  # (k,)
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        xb = X[s:e]
        # ||x-c||^2 = ||x||^2 - 2 x.c + ||c||^2 ; ||x||^2 is constant per row -> drop for argmin.
        d = c2[None, :] - 2.0 * (xb @ centres.T)
        out[s:e] = np.argmin(d, axis=1).astype(np.int64)
    return out


# --------------------------------------------------------------------------- #
# frozen vocab object
# --------------------------------------------------------------------------- #
class AbbaVocab:
    """Frozen ABBA codebook: standardization stats + cluster centres. Encodes a (len, inc) tuple
    set to symbol ids by nearest centre in the scl-weighted standardized space. Exposes
    `codebook_size` for the eval battery (a8 introspection)."""

    def __init__(self, mean, std, scl, centres, raw_centres, tol):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.std = np.asarray(std, dtype=np.float64)
        self.scl = float(scl)
        self.centres = np.asarray(centres, dtype=np.float64)     # standardized-space centres
        self.raw_centres = np.asarray(raw_centres, dtype=np.float64)  # (k,2) raw (len, inc) templates
        self.tol = float(tol)

    @property
    def codebook_size(self) -> int:
        return int(self.centres.shape[0])

    def encode_tuples(self, lengths: np.ndarray, incs: np.ndarray) -> np.ndarray:
        """(len, inc) arrays -> symbol-id array (one id per piece)."""
        if lengths.size == 0:
            return np.zeros((0,), np.int64)
        tuples = np.stack([lengths.astype(np.float64), incs.astype(np.float64)], axis=1)
        Z = _standardize(tuples, self.mean, self.std, self.scl)
        return _assign(Z, self.centres)


# --------------------------------------------------------------------------- #
# fit
# --------------------------------------------------------------------------- #
def fit_vocab(cfg, assets, k=DEFAULT_K, scl=DEFAULT_SCL, max_fit_tuples=DEFAULT_MAX_FIT_TUPLES,
              max_windows_per_asset=None, seed=0) -> AbbaVocab:
    """Fit the ABBA codebook on the POOLED (len, inc) tuples from `assets` (the build corpus).

    Streams each asset run-by-run (db_stream.iter_runs — paged, bounded RAM), accumulates pooled
    (len, inc) tuples up to `max_fit_tuples` (reservoir cap so RAM stays bounded on a huge corpus),
    standardizes with scl weighting, and k-means clusters into k symbols.

    `max_windows_per_asset` is interpreted as a per-asset cap on PIECES contributed to the fit pool
    (keeps the fit balanced across assets; None = no cap).
    """
    rng = np.random.default_rng(seed)
    pool = []
    pooled = 0
    cap = int(max_fit_tuples)
    per_asset_cap = None if max_windows_per_asset is None else int(max_windows_per_asset)

    for a in assets:
        got = 0
        for ts, o, h, l, c in iter_runs(cfg.db, int(a), cfg):
            lens, incs, _ = _pieces_for_run(ts, o, h, l, c, cfg)
            if lens.size == 0:
                continue
            if per_asset_cap is not None and got + lens.size > per_asset_cap:
                take = per_asset_cap - got
                if take <= 0:
                    break
                lens = lens[:take]
                incs = incs[:take]
            got += int(lens.size)
            pool.append(np.stack([lens.astype(np.float64), incs.astype(np.float64)], axis=1))
            pooled += int(lens.size)
            if pooled >= cap:
                break
            if per_asset_cap is not None and got >= per_asset_cap:
                break
        if pooled >= cap:
            break

    if not pool:
        raise RuntimeError("ABBA fit: no pieces produced from build assets (empty corpus?)")
    tuples = np.concatenate(pool, axis=0)
    if tuples.shape[0] > cap:
        # uniform subsample down to the cap (bounded k-means input).
        idx = rng.choice(tuples.shape[0], size=cap, replace=False)
        tuples = tuples[idx]
    if tuples.shape[0] < _MIN_FIT_TUPLES:
        raise RuntimeError(f"ABBA fit: only {tuples.shape[0]} pieces (< {_MIN_FIT_TUPLES}); raise "
                           f"--max-windows-per-asset or lower --abba-tol")

    mean = tuples.mean(axis=0)
    std = np.maximum(tuples.std(axis=0), C.DIV_EPS)  # CLAUDE.md guard
    Z = _standardize(tuples, mean, std, scl)
    centres, _ = _kmeans(Z, k, seed=seed)
    # recover raw (len, inc) template per centre (un-standardize, un-weight) for A2 reconstruction.
    w = np.array([float(scl), 1.0], dtype=np.float64)
    raw_centres = (centres / w) * std + mean
    return AbbaVocab(mean, std, scl, centres, raw_centres, float(getattr(cfg, "abba_tol", DEFAULT_TOL)))


# --------------------------------------------------------------------------- #
# encode (streaming, per asset)
# --------------------------------------------------------------------------- #
def encode_asset_ids(cfg, asset_id, vocab: AbbaVocab, control=None, max_windows_per_asset=None):
    """Yield token-id CHUNKS (1-D int64) for ONE asset, ONE chunk per dt==60 run.

    control=None       -> real vol-unit substrate.
    control='shuffle'  -> donor phase_shuffle_returns surrogate (data.py:165) applied to the run's
                          vol-unit features before ABBA (the A8 structure-destroying null).
    control='random_merge' -> permute the move channel within the run (destroys piece structure but
                          preserves the marginal), a second cheap null.
    Never materializes the asset's full token stream — the caller folds chunks online.
    """
    cap = None if max_windows_per_asset is None else int(max_windows_per_asset)
    emitted = 0
    rng = np.random.default_rng(int(asset_id))  # deterministic per asset
    for ts, o, h, l, c in iter_runs(cfg.db, int(asset_id), cfg):
        lens, incs = _pieces_for_run_control(ts, o, h, l, c, cfg, control, rng)
        if lens.size == 0:
            continue
        ids = vocab.encode_tuples(lens, incs)
        if cap is not None and emitted + ids.size > cap:
            ids = ids[: max(0, cap - emitted)]
        if ids.size:
            yield ids
            emitted += int(ids.size)
        if cap is not None and emitted >= cap:
            break


def _pieces_for_run_control(ts, o, h, l, c, cfg, control, rng):
    """ABBA pieces for one run under an optional A8 control transform of the vol-unit features."""
    if control is None:
        lens, incs, _ = _pieces_for_run(ts, o, h, l, c, cfg)
        return lens, incs
    feats, close, ts_out, finite = features_for_run(ts, o, h, l, c, cfg)
    if feats.shape[0] == 0:
        return np.zeros((0,), np.int64), np.zeros((0,), np.float64)
    fin = np.asarray(finite, dtype=bool)
    if control == "shuffle":
        feats = phase_shuffle_returns(feats.astype(np.float32), fin, rng)
        mv_all = np.asarray(feats[:, 0], dtype=np.float64)
    elif control == "random_merge":
        mv_all = feats[:, 0].astype(np.float64).copy()
        idx = np.flatnonzero(fin)
        if idx.size:
            mv_all[idx] = mv_all[idx][rng.permutation(idx.size)]
    else:
        raise ValueError(f"unknown control {control!r}")
    # segment each finite subsegment of the (possibly transformed) mv path.
    all_len, all_inc = [], []
    i, n = 0, mv_all.size
    tol = float(getattr(cfg, "abba_tol", DEFAULT_TOL))
    while i < n:
        if not fin[i]:
            i += 1
            continue
        j = i
        while j < n and fin[j]:
            j += 1
        seg = mv_all[i:j]
        if seg.size >= 1:
            lens, incs = _segment_run(_cumulative_move(seg), tol)
            if lens.size:
                all_len.append(lens)
                all_inc.append(incs)
        i = j
    if not all_len:
        return np.zeros((0,), np.int64), np.zeros((0,), np.float64)
    return np.concatenate(all_len), np.concatenate(all_inc)


# --------------------------------------------------------------------------- #
# A2 reconstruction: rebuild the vol-unit move path from (len, inc) tokens
# --------------------------------------------------------------------------- #
def _recon_pairs_for_asset(cfg, asset_id, vocab: AbbaVocab, max_windows_per_asset=None):
    """Yield (recon_mv_path, true_mv_path) pairs (one per dt==60 run) for A2.

    Decode: each token id -> its raw (len, inc) cluster template; spread `inc` uniformly over `len`
    bars -> a per-bar recon mv; concatenate. The true path is the original finite mv sequence. A2
    measures the multiplicative agreement of the recon vs true cumulative move (scale-free)."""
    cap = None if max_windows_per_asset is None else int(max_windows_per_asset)
    seen = 0
    for ts, o, h, l, c in iter_runs(cfg.db, int(asset_id), cfg):
        feats, close, ts_out, finite = features_for_run(ts, o, h, l, c, cfg)
        if feats.shape[0] == 0:
            continue
        fin = np.asarray(finite, dtype=bool)
        i, n = 0, feats.shape[0]
        while i < n:
            if not fin[i]:
                i += 1
                continue
            j = i
            while j < n and fin[j]:
                j += 1
            mv_true = feats[i:j, 0].astype(np.float64)
            if mv_true.size >= 1:
                lens, incs = _segment_run(_cumulative_move(mv_true), float(getattr(cfg, "abba_tol", DEFAULT_TOL)))
                if lens.size:
                    ids = vocab.encode_tuples(lens, incs)
                    recon_mv = _decode_mv(ids, lens, vocab)
                    m = min(recon_mv.size, mv_true.size)
                    if m > 0:
                        # cumulative vol-unit move paths (offset so both start at a fixed positive base
                        # -> the A2 ratio metric is well-defined and scale-free).
                        base = 10.0
                        true_path = base + np.cumsum(mv_true[:m])
                        recon_path = base + np.cumsum(recon_mv[:m])
                        yield recon_path, true_path
                        seen += 1
                        if cap is not None and seen >= cap:
                            return
            i = j


def _decode_mv(ids: np.ndarray, lens: np.ndarray, vocab: AbbaVocab) -> np.ndarray:
    """Token ids + the encoded piece lengths -> per-bar reconstructed mv (inc spread over its bars).
    We use the ACTUAL encoded `lens` (length is part of the emitted tuple) and the cluster's template
    increment, so the reconstruction is the ABBA decode of the symbol stream."""
    out = []
    for tok, L in zip(ids.tolist(), lens.tolist()):
        L = int(L)
        if L <= 0:
            continue
        inc = float(vocab.raw_centres[int(tok), 1])
        out.append(np.full(L, inc / float(L), dtype=np.float64))
    if not out:
        return np.zeros((0,), np.float64)
    return np.concatenate(out)


# --------------------------------------------------------------------------- #
# length-distribution accumulator (A9) — bar-lengths of emitted tokens, online
# --------------------------------------------------------------------------- #
def _accumulate_lengths(cfg, assets, vocab: AbbaVocab, max_len_bucket=4096, max_windows_per_asset=None):
    """Online np.bincount of emitted-token bar-lengths over `assets` (A9 input). Host memory is
    O(max_len_bucket), never O(tokens) — we fold each run's piece-length array straight into the
    fixed accumulator."""
    acc = np.zeros(int(max_len_bucket) + 1, dtype=np.int64)
    for a in assets:
        seen = 0
        cap = None if max_windows_per_asset is None else int(max_windows_per_asset)
        for ts, o, h, l, c in iter_runs(cfg.db, int(a), cfg):
            lens, _incs, _ = _pieces_for_run(ts, o, h, l, c, cfg)
            if lens.size == 0:
                continue
            if cap is not None and seen + lens.size > cap:
                lens = lens[: max(0, cap - seen)]
            if lens.size == 0:
                continue
            clipped = np.clip(lens, 0, int(max_len_bucket)).astype(np.int64)
            acc += np.bincount(clipped, minlength=int(max_len_bucket) + 1)
            seen += int(lens.size)
            if cap is not None and seen >= cap:
                break
    return acc


# --------------------------------------------------------------------------- #
# A0 windows (raw OHLC) + a per-window tokenize fn
# --------------------------------------------------------------------------- #
def _sample_ohlc_windows(cfg, asset_id, n_windows=24):
    """A small diagnostic sample of raw-price OHLC windows (W,4) from ONE asset for the A0 unit
    check. We slice consecutive dt==60 windows of length cfg.window+2 (>= a couple of pieces)."""
    W = int(cfg.window)
    out = []
    for ts, o, h, l, c in iter_runs(cfg.db, int(asset_id), cfg):
        n = len(ts)
        if n < W + 2:
            continue
        s = 0
        while s + W + 2 <= n and len(out) < n_windows:
            win = np.stack([o[s:s + W + 2], h[s:s + W + 2],
                            l[s:s + W + 2], c[s:s + W + 2]], axis=1).astype(np.float64)
            out.append(win)
            s += W + 2
        if len(out) >= n_windows:
            break
    return out


def _make_tokenize_window_fn(cfg, vocab: AbbaVocab):
    """Return tokenize_window_fn(ohlc(W,4)) -> single int token id, for A0. We ABBA-segment the
    window's vol-unit mv path and return the FIRST piece's symbol id (a deterministic per-window
    summary token). Rescale-invariant by construction (vol-unit substrate)."""
    tol = float(getattr(cfg, "abba_tol", DEFAULT_TOL))

    def _fn(ohlc):
        ohlc = np.asarray(ohlc, dtype=np.float64)
        L = ohlc.shape[0]
        ts = (np.arange(L, dtype=np.int64) * C.BAR_SECONDS) + 1_600_000_000
        feats, _close, _ts, finite = features_for_run(
            ts, ohlc[:, 0], ohlc[:, 1], ohlc[:, 2], ohlc[:, 3], cfg)
        fin = np.asarray(finite, dtype=bool)
        if feats.shape[0] == 0 or not fin.any():
            return 0
        # first maximal finite subsegment
        idx = np.flatnonzero(fin)
        i0 = int(idx[0])
        i1 = i0
        while i1 < feats.shape[0] and fin[i1]:
            i1 += 1
        mv = feats[i0:i1, 0].astype(np.float64)
        lens, incs = _segment_run(_cumulative_move(mv), tol)
        if lens.size == 0:
            return 0
        ids = vocab.encode_tuples(lens[:1], incs[:1])
        return int(ids[0])

    return _fn


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def run(cfg, build_assets, holdout_assets, k, scl, max_windows_per_asset, out_md, seed=0):
    """Fit ABBA, run the A0/A1/A2/A8/A9 battery, write the RESULT.md + .json."""
    t0 = time.time()
    print(f"[L2-ABBA] db={cfg.db}", flush=True)
    print(f"[L2-ABBA] build_assets={tuple(build_assets)} holdout_assets={tuple(holdout_assets)}", flush=True)
    print(f"[L2-ABBA] k={k} scl={scl} tol={getattr(cfg,'abba_tol',DEFAULT_TOL)} "
          f"max_windows_per_asset={max_windows_per_asset}", flush=True)

    if not quick_check(cfg.db):
        print(f"[L2-ABBA] WARNING: PRAGMA quick_check did not return 'ok' for {cfg.db}", flush=True)

    # base-alphabet null: A1 'null_bits_per_bar' = entropy of the per-bar base symbol (= log2 of its
    # cardinality upper bound). We fit the base alphabet on a pooled feature sample from build assets.
    base_table = _fit_base_alphabet_sample(cfg, build_assets, max_windows_per_asset)
    null_bits_per_bar = float(np.log2(max(N_SYMBOLS(base_table), 2)))

    # ---- fit ABBA codebook ----
    vocab = fit_vocab(cfg, build_assets, k=k, scl=scl,
                      max_windows_per_asset=max_windows_per_asset, seed=seed)
    cb = vocab.codebook_size
    print(f"[L2-ABBA] fitted codebook_size={cb}", flush=True)

    # ---- A1 MDL (online over the full build stream) ----
    def _token_stream():
        for a in build_assets:
            for chunk in encode_asset_ids(cfg, int(a), vocab, control=None,
                                          max_windows_per_asset=max_windows_per_asset):
                yield chunk

    # n_boundaries == number of pieces == number of tokens (each token is one ABBA piece/boundary).
    n_tokens = 0
    for a in build_assets:
        for chunk in encode_asset_ids(cfg, int(a), vocab, control=None,
                                      max_windows_per_asset=max_windows_per_asset):
            n_tokens += int(chunk.size)
    codebook_bits = float(cb) * 2.0 * 32.0  # (len, inc) float32 template per code (model cost)
    a1 = EB.a1_mdl(_token_stream(), codebook_size=cb, null_bits_per_bar=null_bits_per_bar,
                   n_boundaries=n_tokens, codebook_bits=codebook_bits)

    # ---- A2 reconstruction (streamed pairs) ----
    def _recon_pairs():
        for a in build_assets:
            yield from _recon_pairs_for_asset(cfg, int(a), vocab,
                                              max_windows_per_asset=max_windows_per_asset)
    a2 = EB.a2_reconstruction(_recon_pairs(), ratio_tol=0.02)

    # ---- A9 length non-degeneracy ----
    len_acc = _accumulate_lengths(cfg, build_assets, vocab,
                                  max_windows_per_asset=max_windows_per_asset)
    a9 = EB.a9_length_nondegeneracy(len_acc)

    # ---- A0 rescale invariance ----
    a0_windows = _sample_ohlc_windows(cfg, int(build_assets[0]))
    tok_fn = _make_tokenize_window_fn(cfg, vocab)
    a0 = EB.a0_rescale_invariance(tok_fn, a0_windows, lambdas=(0.01, 100.0))

    # ---- A8 cross-asset transfer (real vs shuffle control) ----
    a8 = None
    if holdout_assets:
        def _build_vocab_fn(assets):
            return fit_vocab(cfg, assets, k=k, scl=scl,
                             max_windows_per_asset=max_windows_per_asset, seed=seed)

        def _encode_assets_fn(vc, assets, control):
            for a in assets:
                yield from encode_asset_ids(cfg, int(a), vc, control=control,
                                            max_windows_per_asset=max_windows_per_asset)
        a8 = EB.a8_cross_asset_transfer(_build_vocab_fn, _encode_assets_fn,
                                        build_assets=tuple(build_assets),
                                        holdout_assets=tuple(holdout_assets),
                                        control="shuffle", exclude_top_k=8)

    wall = time.time() - t0

    # ---- verdict (design 03 re-weighting: A8 transfer + A2 fidelity + corrected A1 + A0 + A9) ----
    a0_pass = bool(a0.get("pass", False))
    a9_pass = bool(a9.get("pass", False))
    a1_beats_null = bool(a1["compression_ratio"] > 1.0)
    a8_beats = bool(a8["beats_control"]) if a8 is not None else None
    verdict = _verdict(a0_pass, a1_beats_null, a9_pass, a8_beats)

    sections = [
        ("Configuration", [
            f"- Member: L2 ABBA (adaptive polygonal segmentation of the vol-unit move path).",
            f"- DB: `{cfg.db}`",
            f"- Build assets: {', '.join(str(a) for a in build_assets)}",
            f"- Holdout assets: {', '.join(str(a) for a in holdout_assets) if holdout_assets else '(none)'}",
            f"- k (symbols)={cb} | scl={scl} | tol={getattr(cfg,'abba_tol',DEFAULT_TOL)} "
            f"| max_windows_per_asset={max_windows_per_asset}",
            f"- Window={cfg.window} | base-null bits/bar={null_bits_per_bar:.4f} "
            f"(N_SYMBOLS base={N_SYMBOLS(base_table)})",
            f"- Tokens (build stream)={n_tokens}",
        ]),
        ("A0 — rescale invariance", [
            f"- agreement={a0['agree']:.4f} over {a0['n_trials']} trials  (lambdas={a0['lambdas']})",
            f"- **pass={a0_pass}**  (vol-unit substrate => byte-identical tokens by construction)",
        ]),
        ("A1 — corrected MDL (bits/candle)", [
            f"- bits_per_candle={a1['bits_per_candle']:.4f}",
            f"- null_bits/bar(base)={null_bits_per_bar:.4f} | entropy_bits/token={a1['entropy_bits_per_token']:.4f}",
            f"- covered_bits={a1['covered_bits']:.1f} | boundary_bits={a1['boundary_bits']:.1f} "
            f"| codebook_bits={a1['codebook_bits']:.1f}",
            f"- compression_ratio(null/code)={a1['compression_ratio']:.4f}  "
            f"(**beats null={a1_beats_null}**)",
        ]),
        ("A2 — reconstruction fidelity (vol-unit move path)", [
            f"- frac_within_tol={a2['frac_within_tol']:.4f}  (ratio_tol={a2['ratio_tol']})",
            f"- mean_log_ratio_mse={a2['mean_log_ratio_mse']:.6f} over {a2['n_bars']} bars / {a2['n_pairs']} pieces",
        ]),
        ("A9 — length non-degeneracy", [
            f"- median_len={a9['median_len']:.2f} | max_length_share={a9['max_token_share']:.4f} "
            f"| n_lengths={a9['n_lengths']}",
            f"- all_len1={a9['all_len1']} | total_tokens={a9['total_tokens']}  (**pass={a9_pass}**)",
        ]),
    ]
    if a8 is not None:
        sections.append(("A8 — cross-asset transfer (vs phase-shuffle control)", [
            f"- coverage={a8['coverage']:.4f} (control={a8['coverage_control']:.4f})",
            f"- jaccard={a8['jaccard']:.4f} (control={a8['jaccard_control']:.4f})",
            f"- spearman={a8['spearman']:.4f} (control={a8['spearman_control']:.4f})",
            f"- exclude_top_k={a8['exclude_top_k']} | **beats_control={a8['beats_control']}**",
        ]))
    sections.append(("Timing", [f"- wallclock={wall:.1f}s"]))

    meta = dict(
        member="L2_ABBA", db=cfg.db, build_assets=list(build_assets),
        holdout_assets=list(holdout_assets), k=int(cb), scl=float(scl),
        tol=float(getattr(cfg, "abba_tol", DEFAULT_TOL)), window=int(cfg.window),
        max_windows_per_asset=max_windows_per_asset, n_tokens=int(n_tokens),
        null_bits_per_bar=null_bits_per_bar, wallclock_s=float(wall),
        a0=a0, a1=a1, a2=a2, a9=a9, a8=a8,
    )
    write_result(out_md, "TASK_789 L2 ABBA — Candle-Language Panel Member", sections, verdict, meta)
    return verdict


def _verdict(a0_pass, a1_beats_null, a9_pass, a8_beats):
    """Re-weighted go/no-go (design 03 §6): A0 is the cheap correctness pre-gate; the language is
    PROMISING iff it is rescale-correct, non-degenerate, MDL-beats-null, and (if holdout present)
    transfers above the structure-destroying control. Anything short of that is INCONCLUSIVE; an
    A0 failure is a hard FAIL (the load-bearing property broke)."""
    if not a0_pass:
        return "FAIL"
    gates = [a1_beats_null, a9_pass]
    if a8_beats is not None:
        gates.append(a8_beats)
    if all(gates):
        return "PROMISING"
    return "INCONCLUSIVE"


# --------------------------------------------------------------------------- #
# base-alphabet sample fit (for A1 null bits/bar)
# --------------------------------------------------------------------------- #
def _fit_base_alphabet_sample(cfg, assets, max_windows_per_asset, max_rows=400_000):
    """Fit base_alphabet on a bounded pooled feature sample from `assets` (only used to derive the
    per-bar NULL cardinality for A1). Bounded RAM: we stop accumulating at max_rows."""
    rows = []
    got = 0
    for a in assets:
        for ts, o, h, l, c in iter_runs(cfg.db, int(a), cfg):
            feats, _close, _ts, finite = features_for_run(ts, o, h, l, c, cfg)
            if feats.shape[0] == 0:
                continue
            f = feats[np.asarray(finite, dtype=bool)]
            if f.shape[0]:
                rows.append(f.astype(np.float64))
                got += f.shape[0]
            if got >= max_rows:
                break
        if got >= max_rows:
            break
    if not rows:
        # degenerate fallback: a tiny synthetic pooled corpus so fit_alphabet is well-defined.
        rows = [np.zeros((C.N_FEAT, C.N_FEAT), dtype=np.float64) + 1e-3]
    pooled = np.concatenate(rows, axis=0)
    if pooled.shape[0] > max_rows:
        pooled = pooled[:max_rows]
    return fit_alphabet(pooled, cfg)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _build_cfg(args):
    """Construct a CandleConfig and attach the ABBA tol knob (private attr; frozen dataclass can't
    hold it, so we stash it on a thin subclass-free wrapper via object.__setattr__ won't work on a
    frozen dataclass -> use a lightweight namespace that quacks like CandleConfig for our needs)."""
    base = C.CandleConfig(
        window=args.window,
        db=args.db if args.db else C.db_path(),
        build_assets=tuple(args.build_assets) if args.build_assets else C.DEFAULT_BUILD_ASSETS,
        holdout_assets=tuple(args.holdout_assets) if args.holdout_assets is not None
        else C.DEFAULT_HOLDOUT_ASSETS,
    )
    return _CfgWithTol(base, float(args.abba_tol))


class _CfgWithTol:
    """Thin read-through wrapper over a frozen CandleConfig that also carries `abba_tol`.
    Exposes every attribute the streaming/feature code reads (db, window, page_rows, ewma_halflife,
    sigma_floor, batch, prefetch, build_assets, holdout_assets) by delegation, so it can stand in for
    a CandleConfig everywhere in this module without mutating the frozen dataclass."""

    def __init__(self, base: "C.CandleConfig", abba_tol: float):
        self._base = base
        self.abba_tol = float(abba_tol)

    def __getattr__(self, name):
        return getattr(self._base, name)


def main(argv=None):
    p = argparse.ArgumentParser(description="L2 ABBA candle-language panel member (CPU)")
    p.add_argument("--db", default=None, help="DB path (default: config.db_path())")
    p.add_argument("--out", default="runs/789_token_c0/l2_abba_RESULT.md",
                   help="RESULT.md output path (.json sidecar written alongside)")
    p.add_argument("--window", type=int, default=C.DEFAULT_WINDOW)
    p.add_argument("--k", type=int, default=DEFAULT_K, help="symbol alphabet size (k-means clusters)")
    p.add_argument("--scl", type=float, default=DEFAULT_SCL,
                   help="ABBA scl: length weight vs increment (small => shape dominates)")
    p.add_argument("--abba-tol", type=float, default=DEFAULT_TOL,
                   help="ABBA per-piece squared-error tolerance on the vol-unit move path")
    p.add_argument("--max-windows-per-asset", type=int, default=None,
                   help="cap pieces/tokens contributed per asset (bounded RAM); None = no cap")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--build-assets", type=int, nargs="*", default=None)
    p.add_argument("--holdout-assets", type=int, nargs="*", default=None)
    p.add_argument("--smoke", action="store_true",
                   help="tiny self-contained run: survey DB, pick the top asset, cap windows, no holdout")
    args = p.parse_args(argv)

    cfg = _build_cfg(args)

    if args.smoke:
        rc = survey_row_counts(cfg.db, limit=3)
        print(f"[L2-ABBA smoke] top rowcounts: {rc}", flush=True)
        if rc:
            top = [a for a, _ in rc[:2]]
            cfg = _CfgWithTol(
                C.CandleConfig(window=args.window, db=cfg.db,
                               build_assets=tuple(top), holdout_assets=()),
                float(args.abba_tol))
            return run(cfg, list(top), [], k=min(args.k, 64), scl=args.scl,
                       max_windows_per_asset=args.max_windows_per_asset or 2000,
                       out_md=args.out, seed=args.seed)
        print("[L2-ABBA smoke] empty DB survey; nothing to do", flush=True)
        return "INCONCLUSIVE"

    return run(cfg, list(cfg.build_assets), list(cfg.holdout_assets),
               k=args.k, scl=args.scl,
               max_windows_per_asset=args.max_windows_per_asset,
               out_md=args.out, seed=args.seed)


if __name__ == "__main__":
    main()
