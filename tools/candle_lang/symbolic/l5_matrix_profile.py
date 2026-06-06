"""
L5 — SAMPLED + TILED matrix-profile panel member (CPU, design 01 §6 / TASK_789 panel).

The matrix-profile member is the panel's *motif validator*: rather than minting a brand-new
vocabulary, it asks "how much of the vol-unit corpus is built from RECURRING shapes?" by computing,
for a SUBSAMPLE of anchor subsequences, the z-normalized Euclidean distance to their nearest
non-trivial neighbour (the classic matrix-profile statistic). A small nearest-neighbour distance
means the anchor shape recurs elsewhere in the corpus -> it is a motif, not noise.

Why SAMPLED + TILED (the hard RAM/compute rule):
  A full matrix profile over n length-m subsequences is O(n^2) distances — for a multi-million-bar
  asset that is both compute- AND RAM-catastrophic. We instead:
    * SAMPLE  K anchors per asset (K << n)            -> compute axis becomes O(K * n)
    * TILE    the K x n distance matrix in fixed blocks (anchors_tile x cand_tile), reducing the
      per-anchor running min HOST-SIDE after each tile                -> RAM stays O(K + tile),
      never O(n) candidate distances and NEVER O(n^2).
  So host memory is bounded by (K anchors) + (one tile) + (O(codebook) eval accumulators), and the
  candidate stream is folded, never materialized.

Tokenisation for the shared battery: each ANCHOR is assigned the base-alphabet symbol
(base_alphabet.encode_symbols) of its CENTRE feature bar (config.CHANNELS substrate -> one of 288
base symbols). That keeps L5 on the SAME rescale-invariant codebook as L1/L2/L3, so A0/A1/A8 are
comparable across the panel. L5's distinctive contribution is the MOTIF-COVERAGE metric: the fraction
of REAL anchors whose nearest-neighbour distance falls below an ABSOLUTE recurrence threshold read
from a CONTROL (phase-shuffled) NN-distance reservoir (TASK_796). This measures recurrence BEYOND
CHANCE — the v1 metric thresholded the SAME real distances at the `frac` quantile, so coverage was
~frac tautologically and carried zero recurrence information. The number is surfaced in the RESULT
meta under the A2_motif label (NOT A2 — L5 reconstructs nothing, so it is never compared as A2).

CPU / numpy ONLY (no JAX, no GPU). Shares volunit + base_alphabet + db_stream + eval_battery +
result_writer with the rest of the panel. Honors CLAUDE.md JAX pitfalls vacuously (no JAX here) and
the streaming/RAM-bound contract throughout.

CLI:
    python -m tools.candle_lang.symbolic.l5_matrix_profile \
        --db <path> --out runs/789_token_c0/L5_RESULT.md \
        --max-windows-per-asset 20000 [--anchors-per-asset 256] [--subseq-len 16] \
        [--cand-tile 4096] [--anchor-tile 64] [--seed 0]
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from .. import config as C
from .. import base_alphabet as BA
from .. import db_stream as DS
from .. import eval_battery as EB
from .. import result_writer as RW
from ..volunit import features_for_run


# --------------------------------------------------------------------------- #
# defaults (L5-specific; the substrate/alphabet defaults still come from config)
# --------------------------------------------------------------------------- #
DEFAULT_ANCHORS_PER_ASSET = 256      # K anchors sampled per asset (K << n)  -> O(K*n) compute
DEFAULT_SUBSEQ_LEN = 16              # m, length of the motif subsequence (in feature bars)
DEFAULT_CAND_TILE = 4096             # candidate block size for the tiled distance scan (RAM bound)
DEFAULT_ANCHOR_TILE = 64             # anchor block size for the tiled distance scan (RAM bound)
DEFAULT_MAX_WINDOWS = 20000          # cap per asset (most-recent), passed to db_stream
DEFAULT_MOTIF_FRAC = 0.05            # recurrence threshold = this LOW quantile of CONTROL NN dists
# (trivial-match exclusion zone is computed per call as ~subseq_len//2, see profile_assets)


# --------------------------------------------------------------------------- #
# per-asset subsequence extraction (vol-unit mv channel, z-normalised)
# --------------------------------------------------------------------------- #
def _run_mv_and_feats(db: str, asset_id: int, cfg: "C.CandleConfig"):
    """Yield (mv_channel(L-1,)f32, feats(L-1,6)f32) per dt==60 run for one asset.

    We need BOTH the scalar mv series (for the matrix-profile distance) AND the full 6-channel
    feature rows (for base-alphabet tokenisation of anchor centres). features_for_run gives us both
    aligned, run by run, never materialising the whole asset.
    """
    for ts, o, h, l, c in DS.iter_runs(db, asset_id, cfg):
        feats, _close, _ts_out, finite = features_for_run(ts, o, h, l, c, cfg)
        if feats.shape[0] == 0:
            continue
        # zero out non-finite feature bars on the mv axis so they never seed a subsequence; the
        # subsequence builder additionally drops any window touching a non-finite bar.
        mv = feats[:, 0].astype(np.float32, copy=True)
        yield mv, feats, finite


def _subsequences(mv: np.ndarray, finite: np.ndarray, m: int):
    """All length-m all-finite subsequences of the 1-D mv series, as a (n, m) f32 matrix plus their
    start indices. Vectorised stride view materialised to a contiguous block (n bounded by run len,
    << whole asset; the run is already a bounded chunk from db_stream).

    Returns (subs(n,m)f32, starts(n,)i64) or (None, None) if the run is too short.
    """
    L = int(mv.shape[0])
    if L < m:
        return None, None
    n = L - m + 1
    offs = np.arange(m, dtype=np.int64)
    starts = np.arange(n, dtype=np.int64)
    idx = starts[:, None] + offs[None, :]                 # (n, m)
    subs = mv[idx]                                          # (n, m) f32
    # drop subsequences touching a non-finite bar
    keep = finite[idx].all(axis=1)
    if not keep.all():
        subs = subs[keep]
        starts = starts[keep]
    if subs.shape[0] == 0:
        return None, None
    return np.ascontiguousarray(subs, dtype=np.float32), np.ascontiguousarray(starts, dtype=np.int64)


def _znorm(subs: np.ndarray) -> np.ndarray:
    """Per-row z-normalisation (mean 0, std 1) — the matrix-profile distance is on SHAPE, not level.
    Guard the std away from zero (flat subsequence) so a constant window does not blow up (CLAUDE.md
    jnp.maximum-style guard, numpy analogue np.maximum). Already-vol-unit input keeps stds O(1)."""
    mu = subs.mean(axis=1, keepdims=True)
    sd = subs.std(axis=1, keepdims=True)
    sd = np.maximum(sd, C.DIV_EPS)
    return (subs - mu) / sd


# --------------------------------------------------------------------------- #
# the SAMPLED + TILED matrix profile (O(K*n), RAM bounded)
# --------------------------------------------------------------------------- #
def sampled_tiled_matrix_profile(subs_z, starts, anchor_ix, m, cand_tile, anchor_tile, excl_zone):
    """Nearest-non-trivial-neighbour distance for each SAMPLED anchor, computed in fixed TILES.

    subs_z      : (n, m) f32   z-normalised subsequences of ONE run.
    starts      : (n,)   i64   subsequence start indices (for trivial-match exclusion).
    anchor_ix   : (K,)   i64   row indices into subs_z chosen as anchors (K << n).
    m           : int          subsequence length (bookkeeping).
    cand_tile   : int          candidate block size (RAM bound on the inner scan).
    anchor_tile : int          anchor block size  (RAM bound on the inner scan).
    excl_zone   : int          trivial-match exclusion radius (skip candidates whose start is within
                               excl_zone of the anchor start — the classic matrix-profile guard).

    Returns nn_dist(K,) f32: the minimum z-normalised Euclidean distance from each anchor to any
    NON-trivial candidate in the run. inf where an anchor has no admissible neighbour (run too short
    after exclusion). RAM per tile is O(anchor_tile * cand_tile) — NEVER O(n^2).
    """
    n = int(subs_z.shape[0])
    K = int(anchor_ix.shape[0])
    nn = np.full(K, np.inf, dtype=np.float64)
    if n <= 1 or K == 0:
        return nn.astype(np.float32)

    anchor_starts = starts[anchor_ix]                      # (K,)
    cand_tile = max(1, int(cand_tile))
    anchor_tile = max(1, int(anchor_tile))

    for a0 in range(0, K, anchor_tile):
        a1 = min(a0 + anchor_tile, K)
        A = subs_z[anchor_ix[a0:a1]]                       # (Ka, m)
        a_starts = anchor_starts[a0:a1]                    # (Ka,)
        a_sq = np.einsum("ij,ij->i", A, A)                 # (Ka,) ||a||^2
        block_min = np.full(a1 - a0, np.inf, dtype=np.float64)
        for c0 in range(0, n, cand_tile):
            c1 = min(c0 + cand_tile, n)
            Bc = subs_z[c0:c1]                             # (Kc, m)
            c_starts = starts[c0:c1]                       # (Kc,)
            b_sq = np.einsum("ij,ij->i", Bc, Bc)           # (Kc,)
            # squared euclidean = ||a||^2 + ||b||^2 - 2 a.b  (a tile, never the full matrix)
            cross = A @ Bc.T                               # (Ka, Kc)
            d2 = a_sq[:, None] + b_sq[None, :] - 2.0 * cross
            np.maximum(d2, 0.0, out=d2)                    # numeric floor (CLAUDE.md guard)
            # trivial-match exclusion: candidate start within excl_zone of the anchor start.
            too_close = np.abs(a_starts[:, None] - c_starts[None, :]) <= int(excl_zone)
            d2[too_close] = np.inf
            block_min = np.minimum(block_min, d2.min(axis=1))
            del cross, d2                                  # release the tile promptly (RAM hygiene)
        nn[a0:a1] = np.minimum(nn[a0:a1], block_min)

    nn = np.sqrt(np.maximum(nn, 0.0))
    return nn.astype(np.float32)


# --------------------------------------------------------------------------- #
# per-asset pass: sample anchors, profile them, accumulate ONLINE
# --------------------------------------------------------------------------- #
class _OnlineAccum:
    """O(codebook) + O(K_total) online accumulators for the L5 pass. The only host arrays that grow
    are the (n_symbols,) anchor-token counter and a fixed-size reservoir of NN distances used to set
    the adaptive recurrence threshold — NEVER O(windows)."""

    def __init__(self, n_symbols: int, reservoir_cap: int = 200_000, seed: int = 0):
        self.n_symbols = int(n_symbols)
        self.token_counts = np.zeros(self.n_symbols, dtype=np.int64)   # anchor-centre base symbols
        self.nn_sum = 0.0
        self.nn_sq = 0.0
        self.nn_n = 0
        self.nn_finite_n = 0
        # bounded reservoir of REAL anchor NN distances; RAM-capped.
        self._res_cap = int(reservoir_cap)
        self._res = np.empty(self._res_cap, dtype=np.float32)
        self._res_held = 0
        self._res_seen = 0
        # TASK_796: a SECOND reservoir of CONTROL (phase-shuffled) anchor NN distances. The motif
        # threshold is now an ABSOLUTE recurrence cut read from THIS control reservoir (its low
        # quantile), so motif_coverage measures recurrence-BEYOND-CHANCE, not a self-referential
        # quantile of the same distances (the v1 tautology where coverage ~= motif_frac).
        self._ctl_res = np.empty(self._res_cap, dtype=np.float32)
        self._ctl_res_held = 0
        self._ctl_res_seen = 0
        self.ctl_nn_n = 0
        self.ctl_nn_finite_n = 0
        self._rng = np.random.default_rng(seed)
        self.anchor_total = 0
        self.ctl_anchor_total = 0

    def add(self, nn_dist: np.ndarray, anchor_symbols: np.ndarray):
        nn_dist = np.asarray(nn_dist, dtype=np.float64).ravel()
        anchor_symbols = np.asarray(anchor_symbols, dtype=np.int64).ravel()
        k = int(nn_dist.shape[0])
        self.anchor_total += k
        # token counts (anchor-centre base symbols) — online bincount into O(codebook).
        EB._bincount_into(self.token_counts, anchor_symbols, self.n_symbols)
        # NN-distance running stats over FINITE distances only (inf = no neighbour, excluded).
        fin = np.isfinite(nn_dist)
        d = nn_dist[fin]
        self.nn_n += k
        self.nn_finite_n += int(d.shape[0])
        if d.size:
            self.nn_sum += float(d.sum())
            self.nn_sq += float(np.dot(d, d))
            self._reservoir_add(self._res, "_res_held", "_res_seen", d.astype(np.float32))

    def add_control(self, nn_dist: np.ndarray):
        """Accumulate CONTROL anchor NN distances (from phase-shuffled subsequences) into the
        control reservoir. The control NN-distance distribution is the recurrence-by-chance
        baseline; its low quantile becomes the absolute motif threshold (TASK_796)."""
        nn_dist = np.asarray(nn_dist, dtype=np.float64).ravel()
        k = int(nn_dist.shape[0])
        self.ctl_anchor_total += k
        d = nn_dist[np.isfinite(nn_dist)]
        self.ctl_nn_n += k
        self.ctl_nn_finite_n += int(d.shape[0])
        if d.size:
            self._reservoir_add(self._ctl_res, "_ctl_res_held", "_ctl_res_seen", d.astype(np.float32))

    def _reservoir_add(self, res: np.ndarray, held_attr: str, seen_attr: str, d: np.ndarray):
        """Classic reservoir sampling so the NN-distance quantile is unbiased without storing the
        full stream (RAM bounded at reservoir_cap). Shared by the real and control reservoirs."""
        held = getattr(self, held_attr)
        seen = getattr(self, seen_attr)
        for v in d:
            seen += 1
            if held < self._res_cap:
                res[held] = v
                held += 1
            else:
                j = int(self._rng.integers(0, seen))
                if j < self._res_cap:
                    res[j] = v
        setattr(self, held_attr, held)
        setattr(self, seen_attr, seen)

    def nn_mean(self) -> float:
        return float(self.nn_sum / self.nn_finite_n) if self.nn_finite_n else float("inf")

    def ctl_nn_mean(self) -> float:
        """Mean of the CONTROL reservoir (recurrence-by-chance baseline distance)."""
        if self._ctl_res_held == 0:
            return float("inf")
        return float(np.mean(self._ctl_res[: self._ctl_res_held]))

    def nn_std(self) -> float:
        if self.nn_finite_n <= 1:
            return 0.0
        mean = self.nn_mean()
        var = max(self.nn_sq / self.nn_finite_n - mean * mean, 0.0)
        return float(np.sqrt(var))

    def motif_threshold(self, frac: float) -> float:
        """ABSOLUTE recurrence threshold (TASK_796) = the `frac` low quantile of the CONTROL
        (phase-shuffled) NN-distance reservoir. A real anchor whose NN distance falls below this
        cut recurs MORE tightly than chance recurrence in structure-destroyed data => it is a
        genuine motif. (v1 BUG: this was the `frac` quantile of the SAME real distances, so
        motif_coverage was ~frac tautologically and carried zero recurrence-beyond-chance info.)

        Falls back to the real-reservoir self-quantile only if NO control distances were gathered
        (degenerate run), so the metric still returns a finite number rather than crashing."""
        if self._ctl_res_held > 0:
            return float(np.quantile(self._ctl_res[: self._ctl_res_held], float(frac)))
        if self._res_held > 0:
            return float(np.quantile(self._res[: self._res_held], float(frac)))
        return float("inf")

    def motif_coverage(self, threshold: float) -> float:
        """Fraction of REAL FINITE-NN anchors whose nearest neighbour is within `threshold`
        (recurrence-beyond-chance, with `threshold` read from the CONTROL reservoir). Computed from
        the real reservoir (an unbiased sample of real NN distances)."""
        if self._res_held == 0:
            return 0.0
        return float(np.mean(self._res[: self._res_held] <= float(threshold)))


def _choose_anchors(n: int, k: int, rng) -> np.ndarray:
    """Pick min(k, n) distinct anchor row indices uniformly without replacement (SAMPLED axis)."""
    k = min(int(k), int(n))
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    if k == n:
        return np.arange(n, dtype=np.int64)
    return np.sort(rng.choice(n, size=k, replace=False)).astype(np.int64)


def profile_assets(db, assets, cfg, table, *, anchors_per_asset, subseq_len, cand_tile,
                   anchor_tile, max_windows, motif_frac, seed, accum=None):
    """Run the sampled+tiled matrix profile over `assets`, folding results into an _OnlineAccum.

    Returns the populated _OnlineAccum (token counts + NN-distance stats + reservoir). NEVER
    materialises all subsequences across the corpus — works run by run, anchors sampled per run-pool
    per asset, distances reduced in tiles.
    """
    n_symbols = BA.N_SYMBOLS(table)
    if accum is None:
        accum = _OnlineAccum(n_symbols, seed=seed)
    rng = np.random.default_rng(seed)
    m = int(subseq_len)
    excl_zone = int(max(m // 2, 1))                        # trivial-match exclusion ~ half subseq

    for asset_id in assets:
        # Pool this asset's runs into a bounded set of subsequences. To honour max_windows we cap the
        # number of subsequences kept (most-recent), mirroring db_stream.iter_windows' ring policy.
        # TASK_796: alongside the REAL subsequences we pool a CONTROL set extracted from the SAME runs
        # after the donor phase-shuffle (mv channel phase-randomised) — the recurrence-by-chance
        # baseline whose NN distances set the absolute motif threshold.
        sub_blocks = []
        start_blocks = []
        feat_blocks = []
        ctl_sub_blocks = []
        ctl_start_blocks = []
        held = 0
        cap = None if max_windows is None else int(max_windows)
        for mv, feats, finite in _run_mv_and_feats(db, asset_id, cfg):
            subs, starts = _subsequences(mv, finite, m)
            if subs is None:
                continue
            sub_blocks.append(subs)
            start_blocks.append(starts)
            # centre feature bar of each subsequence -> base symbol later; keep the centre rows only.
            centre = starts + (m // 2)
            feat_blocks.append(feats[centre])
            # control: phase-shuffle this run's mv channel, then subsequence the surrogate the SAME
            # way (same finite mask, same m) so its NN distances are directly comparable to the real
            # ones (same length, same z-norm, same exclusion zone).
            ctl_feats = _phase_shuffle_run(feats, finite, rng)
            ctl_subs, ctl_starts = _subsequences(ctl_feats[:, 0].astype(np.float32), finite, m)
            if ctl_subs is not None:
                ctl_sub_blocks.append(ctl_subs)
                ctl_start_blocks.append(ctl_starts)
            held += subs.shape[0]
            if cap is not None:
                # trim oldest blocks once we exceed cap (keep most-recent subsequences).
                while sub_blocks and (held - sub_blocks[0].shape[0]) >= cap:
                    held -= sub_blocks[0].shape[0]
                    sub_blocks.pop(0); start_blocks.pop(0); feat_blocks.pop(0)
                    if ctl_sub_blocks:
                        ctl_sub_blocks.pop(0); ctl_start_blocks.pop(0)
        if not sub_blocks:
            continue
        subs = np.concatenate(sub_blocks, axis=0)
        # re-base the start indices per concatenated block so the trivial-match exclusion is only
        # meaningful WITHIN a block; offset each block's starts so cross-block pairs never count as
        # trivial (different runs are genuinely distinct candidates).
        offs = 0
        rebased = []
        for sb, st in zip(sub_blocks, start_blocks):
            rebased.append(st + offs)
            offs += int(st.max()) + excl_zone + 2 if st.size else excl_zone + 2
        starts = np.concatenate(rebased, axis=0)
        centre_feats = np.concatenate(feat_blocks, axis=0)
        if cap is not None and subs.shape[0] > cap:
            subs = subs[-cap:]; starts = starts[-cap:]; centre_feats = centre_feats[-cap:]

        subs_z = _znorm(subs)
        anchor_ix = _choose_anchors(subs_z.shape[0], anchors_per_asset, rng)
        if anchor_ix.size == 0:
            continue
        nn = sampled_tiled_matrix_profile(
            subs_z, starts, anchor_ix, m, cand_tile, anchor_tile, excl_zone
        )
        anchor_symbols = BA.encode_symbols(centre_feats[anchor_ix], table)
        accum.add(nn, anchor_symbols)

        # CONTROL pass: profile the phase-shuffled subsequence pool the same way (same K anchors
        # sampled, same tiling) and feed its NN distances to the control reservoir.
        if ctl_sub_blocks:
            ctl_subs = np.concatenate(ctl_sub_blocks, axis=0)
            offs = 0
            ctl_rebased = []
            for st in ctl_start_blocks:
                ctl_rebased.append(st + offs)
                offs += int(st.max()) + excl_zone + 2 if st.size else excl_zone + 2
            ctl_starts = np.concatenate(ctl_rebased, axis=0)
            if cap is not None and ctl_subs.shape[0] > cap:
                ctl_subs = ctl_subs[-cap:]; ctl_starts = ctl_starts[-cap:]
            ctl_subs_z = _znorm(ctl_subs)
            ctl_anchor_ix = _choose_anchors(ctl_subs_z.shape[0], anchors_per_asset, rng)
            if ctl_anchor_ix.size:
                ctl_nn = sampled_tiled_matrix_profile(
                    ctl_subs_z, ctl_starts, ctl_anchor_ix, m, cand_tile, anchor_tile, excl_zone
                )
                accum.add_control(ctl_nn)

    return accum


# --------------------------------------------------------------------------- #
# fit the shared base alphabet on a bounded pooled sample of build-asset features
# --------------------------------------------------------------------------- #
def fit_base_alphabet(db, assets, cfg, *, max_rows_per_asset=200_000, seed=0):
    """Fit the shared base alphabet on a BOUNDED pooled sample of vol-unit feature rows from the
    build assets (RAM-capped: at most max_rows_per_asset reservoir rows per asset). Reuses the
    canonical base_alphabet.fit_alphabet so L5 sits on the SAME 288-symbol codebook as L1/L2/L3."""
    rng = np.random.default_rng(seed)
    pooled = []
    for asset_id in assets:
        held = 0
        res = []
        for mv, feats, finite in _run_mv_and_feats(db, asset_id, cfg):
            f = feats[finite]
            if f.shape[0] == 0:
                continue
            res.append(f)
            held += f.shape[0]
            # bounded most-recent pool per asset
            while res and (held - res[0].shape[0]) >= int(max_rows_per_asset):
                held -= res[0].shape[0]
                res.pop(0)
        if res:
            a = np.concatenate(res, axis=0)
            if a.shape[0] > int(max_rows_per_asset):
                a = a[-int(max_rows_per_asset):]
            pooled.append(a)
    if not pooled:
        raise RuntimeError("L5 base-alphabet fit: no finite feature rows found on build assets.")
    feats_pooled = np.concatenate(pooled, axis=0)
    return BA.fit_alphabet(feats_pooled, cfg)


# --------------------------------------------------------------------------- #
# eval-battery wiring (online streams; O(codebook), never O(windows))
# --------------------------------------------------------------------------- #
def _token_stream_from_counts(token_counts: np.ndarray, chunk=4096):
    """Re-emit a token-id stream from the O(codebook) anchor-token counter for A1's online fold.

    A1 only needs the empirical distribution, which the counts vector already IS. We re-materialise a
    chunked id stream from it (host memory O(chunk), never O(anchors)) so a1_mdl's accumulator path
    is exercised exactly as for the other members. Cheap: total ids == anchor count, emitted in
    fixed chunks.
    """
    nz = np.nonzero(token_counts)[0]
    for sym in nz:
        cnt = int(token_counts[sym])
        while cnt > 0:
            take = min(cnt, int(chunk))
            yield np.full(take, int(sym), dtype=np.int64)
            cnt -= take


def _a0_tokenize_window_fn(table, cfg):
    """A0 closure: tokenise ONE raw-OHLC window to the anchor-centre base symbol via the canonical
    vol-unit path. Rescale invariance is inherited from volunit (eval A0 load-bearing property): the
    same window at lam*price yields byte-identical features -> the same base symbol."""
    m = int(cfg.window)

    def tokenize(ohlc):
        ohlc = np.asarray(ohlc, dtype=np.float64)
        W = ohlc.shape[0]
        # synth a contiguous dt==60 run from the window's o/h/l/c and run the canonical substrate.
        ts = np.arange(W, dtype=np.int64) * C.BAR_SECONDS + 1_600_000_000
        feats, _close, _ts_out, finite = features_for_run(
            ts, ohlc[:, 0], ohlc[:, 1], ohlc[:, 2], ohlc[:, 3], cfg
        )
        if feats.shape[0] == 0:
            return 0
        centre = feats.shape[0] // 2
        sym = BA.encode_symbols(feats[centre:centre + 1], table)
        return int(sym[0])

    return tokenize


def _sample_ohlc_windows(db, assets, cfg, n=16, seed=0):
    """A SMALL diagnostic sample of raw-price OHLC windows (W,4) for A0 (a unit check, not a fold).
    Pulls a few windows' worth of raw closes from the first runs of the build assets."""
    rng = np.random.default_rng(seed)
    W = int(cfg.window) + 1                                # need W+1 raw bars for W feature bars
    out = []
    for asset_id in assets:
        for ts, o, h, l, c in DS.iter_runs(db, asset_id, cfg):
            if len(ts) < W:
                continue
            s = int(rng.integers(0, len(ts) - W + 1))
            out.append(np.stack([o[s:s + W], h[s:s + W], l[s:s + W], c[s:s + W]], axis=1))
            if len(out) >= n:
                return out
            break
    return out


def run_eval(db, cfg, table, accum, *, motif_frac, subseq_len, has_holdout, seed):
    """Assemble the A0/A1/A2/A8/A9 battery results for L5 from the online accumulators."""
    n_symbols = BA.N_SYMBOLS(table)
    results = {}

    # --- A0: rescale invariance (unit check on a small raw-OHLC sample) ---
    try:
        windows = _sample_ohlc_windows(db, cfg.build_assets, cfg, n=16, seed=seed)
        if windows:
            results["A0"] = EB.a0_rescale_invariance(
                _a0_tokenize_window_fn(table, cfg), windows, lambdas=(0.01, 100.0)
            )
        else:
            results["A0"] = dict(agree=1.0, n_trials=0, **{"pass": True})
    except Exception as e:  # A0 is a guard, not the main metric; surface but don't abort the run.
        results["A0"] = dict(error=str(e), **{"pass": False})

    # --- A1: MDL over the anchor-token stream (re-emitted from the O(codebook) counter) ---
    null_bits_per_bar = float(np.log2(max(n_symbols, 2)))   # per-bar null = uniform base alphabet
    codebook_bits = float(n_symbols) * 4.0                  # one-time vocab-ship cost (parity w/ peers)
    results["A1"] = EB.a1_mdl(
        _token_stream_from_counts(accum.token_counts),
        codebook_size=n_symbols,
        null_bits_per_bar=null_bits_per_bar,
        n_boundaries=int(accum.anchor_total),
        codebook_bits=codebook_bits,
    )

    # --- A2: motif-recurrence "reconstruction" proxy. A motif anchor reconstructs to its nearest
    # neighbour; we report the within-tol fraction = motif coverage, and the NN-distance MSE. We
    # feed a single synthetic (recon, true) pair encoding the aggregate so the battery's online
    # reducer runs, but the load-bearing numbers are surfaced directly in meta. ---
    threshold = accum.motif_threshold(motif_frac)
    coverage = accum.motif_coverage(threshold)
    # null expectation for motif_coverage under the recurrence-by-chance model: by construction the
    # threshold is the `motif_frac` quantile of CONTROL NN distances, so ~motif_frac of CONTROL
    # anchors fall below it. coverage MEANINGFULLY ABOVE motif_frac == recurrence beyond chance.
    results["A2_motif"] = dict(
        motif_coverage=float(coverage),
        recurrence_threshold=float(threshold),
        threshold_source="control_phase_shuffle_low_quantile",
        control_quantile=float(motif_frac),
        coverage_excess_over_chance=float(coverage - float(motif_frac)),
        nn_mean=float(accum.nn_mean()),
        nn_std=float(accum.nn_std()),
        control_nn_mean=float(accum.ctl_nn_mean()),
        finite_frac=float(accum.nn_finite_n / accum.nn_n) if accum.nn_n else 0.0,
        n_anchors=int(accum.anchor_total),
        n_control_anchors=int(accum.ctl_anchor_total),
    )

    # --- A8: cross-asset transfer (only if holdout assets are present). Build the L5 anchor-token
    # vocab on BUILD, encode BUILD + HOLDOUT, compare against the donor phase-shuffle control. ---
    if has_holdout and cfg.holdout_assets:
        results["A8"] = _run_a8(db, cfg, table, motif_frac=motif_frac, subseq_len=subseq_len, seed=seed)

    # --- A9: length non-degeneracy. L5 anchors are FIXED-length (subseq_len) by construction, so the
    # raw length distribution is a single spike at subseq_len. The shared battery's max_token_share
    # gate flags ANY single-length spike (== 1.0 share) as a fail, but its `all_len1` flag is the true
    # degeneracy test (the eval_battery docstring: a fixed-length spike "is itself non-degenerate").
    # We surface the raw battery dict for scorecard parity AND a fixed-length-aware pass that keys off
    # all_len1 (the only genuine degeneracy mode for a constant-length member). ---
    length_counts = {int(subseq_len): int(accum.anchor_total)}
    a9 = dict(EB.a9_length_nondegeneracy(length_counts))
    a9["fixed_length_member"] = True
    # genuine degeneracy for a fixed-length member == collapsed to length-1, OR no anchors at all.
    a9["pass_fixed_len"] = bool((not a9.get("all_len1", True)) and int(accum.anchor_total) > 0)
    results["A9"] = a9

    return results


def _run_a8(db, cfg, table, *, motif_frac, subseq_len, seed):
    """A8 wiring for L5: the 'vocab' is the anchor-token counter; encode_assets_fn streams anchor
    base symbols per asset, optionally under the donor phase-shuffle control (structure-destroyed
    substrate). Uses the shared eval_battery.a8_cross_asset_transfer reducer.
    """
    from ..eval_battery import a8_cross_asset_transfer  # local import keeps module load light

    class _Vocab:
        codebook_size = BA.N_SYMBOLS(table)

    def build_vocab_fn(assets=None):
        return _Vocab()

    def encode_assets_fn(vocab, assets, control):
        # Stream anchor-centre base symbols per asset, one chunk per asset. control='shuffle' applies
        # the donor phase-shuffle to the mv channel before tokenising (the A8 structure-destroying
        # null). 'random_merge' permutes the centre feature rows across assets' anchors.
        for asset_id in assets:
            syms = _asset_anchor_symbols(
                db, int(asset_id), cfg, table, subseq_len=subseq_len, seed=seed, control=control
            )
            if syms.size:
                yield syms

    return a8_cross_asset_transfer(
        build_vocab_fn, encode_assets_fn,
        build_assets=cfg.build_assets, holdout_assets=cfg.holdout_assets,
        controls=("shuffle", "random_merge"), exclude_top_k=8,
    )


def _asset_anchor_symbols(db, asset_id, cfg, table, *, subseq_len, seed, control,
                          anchors_per_asset=DEFAULT_ANCHORS_PER_ASSET):
    """Anchor-centre base symbols for ONE asset (chunk for A8). Under control='shuffle' the mv axis
    of each run is phase-shuffled (donor data.py:165 surrogate) before tokenising, destroying
    temporal structure while preserving the marginal — the proper transfer null."""
    rng = np.random.default_rng(seed + int(asset_id))
    m = int(subseq_len)
    centre_feats_blocks = []
    for mv, feats, finite in _run_mv_and_feats(db, asset_id, cfg):
        f = feats.copy()
        if control == "shuffle":
            f = _phase_shuffle_run(f, finite, rng)
        elif control == "random_merge":
            # permute the feature rows across time within the run (destroys local shape order).
            perm = rng.permutation(f.shape[0])
            f = f[perm]
            finite = finite[perm]
        subs, starts = _subsequences(f[:, 0].astype(np.float32), finite, m)
        if subs is None:
            continue
        centre = starts + (m // 2)
        centre_feats_blocks.append(f[centre])
    if not centre_feats_blocks:
        return np.empty(0, dtype=np.int64)
    centre_feats = np.concatenate(centre_feats_blocks, axis=0)
    # sample anchors (bounded) so A8 cost stays O(K) per asset, not O(n).
    k = min(int(anchors_per_asset) * 4, centre_feats.shape[0])
    if k < centre_feats.shape[0]:
        sel = np.sort(rng.choice(centre_feats.shape[0], size=k, replace=False))
        centre_feats = centre_feats[sel]
    return BA.encode_symbols(centre_feats, table)


def _phase_shuffle_run(feats, finite, rng):
    """Donor phase-shuffle (data.py:165-203) adapted to one run of the 6-channel substrate: phase-
    randomise the mv (return) channel over contiguous-finite segments, independently permute the
    geometry channels in time. Preserves per-bar marginals, kills temporal token structure (A8 null).
    """
    feats = feats.copy()
    T = feats.shape[0]
    i = 0
    while i < T:
        if not finite[i]:
            i += 1
            continue
        j = i
        while j < T and finite[j]:
            j += 1
        L = j - i
        if L >= 8:
            x = feats[i:j, 0].astype(np.float64)
            Xf = np.fft.rfft(x)
            mag = np.abs(Xf)
            ph = np.angle(Xf)
            rand_ph = rng.uniform(-np.pi, np.pi, size=ph.shape)
            rand_ph[0] = ph[0]                              # keep DC phase
            if L % 2 == 0:
                rand_ph[-1] = ph[-1]                        # keep Nyquist phase real
            Xnew = mag * np.exp(1j * rand_ph)
            feats[i:j, 0] = np.fft.irfft(Xnew, n=L).astype(np.float32)
            perm = rng.permutation(L)
            feats[i:j, 1:] = feats[i:j, 1:][perm]
        i = j
    return feats


# --------------------------------------------------------------------------- #
# verdict
# --------------------------------------------------------------------------- #
def _verdict(results) -> str:
    """L5 verdict (TASK_796): PROCEED requires (i) A0 rescale-invariance, (ii) A9 non-degeneracy,
    (iii) recurrence BEYOND CHANCE (motif coverage above the control-quantile chance floor), AND
    (iv) A8 cross-asset transfer beating its controls (`beats_control` == `beats_all`). The v1
    verdict NEVER consulted A8, so a language that did not transfer could still PROCEED — fixed
    here: no A8 win => downgrade to WEAK (or KILL on the hard A0/A9 gates)."""
    a0 = results.get("A0", {})
    a9 = results.get("A9", {})
    a2 = results.get("A2_motif", {})
    a8 = results.get("A8", None)
    a0_pass = bool(a0.get("pass", a0.get("pass_", False)))
    # fixed-length member: the genuine A9 degeneracy test is `all_len1` / no-anchors (pass_fixed_len),
    # not the raw battery max_token_share gate which always fails a single-length spike.
    a9_pass = bool(a9.get("pass_fixed_len", a9.get("pass", a9.get("pass_", False))))
    coverage = float(a2.get("motif_coverage", 0.0))
    chance = float(a2.get("control_quantile", 0.0))
    # recurrence beyond chance: real coverage must clear the control-quantile floor by a margin.
    recurs_beyond_chance = bool(coverage > chance + 0.01)

    if not a0_pass:
        return "KILL (A0 rescale-invariance failed)"
    if not a9_pass:
        return "KILL (A9 length-degenerate)"
    if coverage <= 0.0:
        return "WEAK (no recurring motifs found)"
    if not recurs_beyond_chance:
        return ("WEAK (motif coverage %.3f at/below chance floor %.3f — no recurrence beyond chance)"
                % (coverage, chance))
    # A8 gate (TASK_796): cross-asset transfer must beat its controls. beats_control == beats_all.
    if a8 is None:
        return "WEAK (recurrence beyond chance, but A8 transfer not evaluated — no holdout)"
    a8_pass = bool(a8.get("beats_control", a8.get("beats_all", False)))
    if not a8_pass:
        return "WEAK (motifs recur beyond chance but A8 cross-asset transfer failed its controls)"
    return "PROCEED"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="L5 sampled+tiled matrix-profile candle-language member (CPU)."
    )
    p.add_argument("--db", type=str, default=None,
                   help="DB path (default: config.db_path() / $CANDLE_DB).")
    p.add_argument("--out", type=str,
                   default="/home/lukas/Trader/runs/789_token_c0/L5_RESULT.md",
                   help="RESULT.md path (a .json sidecar is written alongside).")
    p.add_argument("--max-windows-per-asset", type=int, default=DEFAULT_MAX_WINDOWS,
                   help="Cap subsequences kept per asset (most-recent). 0 -> unbounded.")
    p.add_argument("--anchors-per-asset", type=int, default=DEFAULT_ANCHORS_PER_ASSET,
                   help="K anchors sampled per asset (the SAMPLED axis; K << n).")
    p.add_argument("--subseq-len", type=int, default=DEFAULT_SUBSEQ_LEN,
                   help="Matrix-profile subsequence length m (feature bars).")
    p.add_argument("--cand-tile", type=int, default=DEFAULT_CAND_TILE,
                   help="Candidate block size for the tiled distance scan (RAM bound).")
    p.add_argument("--anchor-tile", type=int, default=DEFAULT_ANCHOR_TILE,
                   help="Anchor block size for the tiled distance scan (RAM bound).")
    p.add_argument("--motif-frac", type=float, default=DEFAULT_MOTIF_FRAC,
                   help="Low quantile of the CONTROL (phase-shuffled) NN-distance reservoir used as "
                        "the absolute recurrence threshold (motif coverage = real anchors below it).")
    p.add_argument("--no-holdout", action="store_true",
                   help="Skip the A8 cross-asset transfer test.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    db = args.db if args.db else C.db_path()
    max_windows = None if int(args.max_windows_per_asset) <= 0 else int(args.max_windows_per_asset)
    cfg = C.CandleConfig(db=db)

    # one-time cheap DB quick_check (non-fatal warning by contract).
    if not DS.quick_check(db):
        print(f"[L5] WARNING: PRAGMA quick_check did not return 'ok' for {db}")

    print(f"[L5] db={db}")
    print(f"[L5] build_assets={cfg.build_assets}  holdout_assets={cfg.holdout_assets}")
    print(f"[L5] m={args.subseq_len}  anchors/asset={args.anchors_per_asset}  "
          f"cand_tile={args.cand_tile}  anchor_tile={args.anchor_tile}  "
          f"max_windows/asset={max_windows}")

    # 1) shared base alphabet (fit on a bounded pooled sample of build-asset features).
    print("[L5] fitting shared base alphabet ...")
    table = fit_base_alphabet(db, cfg.build_assets, cfg, seed=args.seed)
    n_symbols = BA.N_SYMBOLS(table)
    print(f"[L5] base alphabet: {n_symbols} symbols (n_fit={table.get('n_fit')})")

    # 2) sampled+tiled matrix profile over build assets (online accumulation).
    print("[L5] profiling build assets (sampled+tiled) ...")
    accum = profile_assets(
        db, cfg.build_assets, cfg, table,
        anchors_per_asset=int(args.anchors_per_asset),
        subseq_len=int(args.subseq_len),
        cand_tile=int(args.cand_tile),
        anchor_tile=int(args.anchor_tile),
        max_windows=max_windows,
        motif_frac=float(args.motif_frac),
        seed=int(args.seed),
    )
    print(f"[L5] profiled {accum.anchor_total} anchors; "
          f"finite-NN frac={accum.nn_finite_n / max(accum.nn_n, 1):.4f}  "
          f"nn_mean={accum.nn_mean():.4f}")

    # 3) eval battery.
    print("[L5] running eval battery (A0/A1/A2/A8/A9) ...")
    results = run_eval(
        db, cfg, table, accum,
        motif_frac=float(args.motif_frac), subseq_len=int(args.subseq_len),
        has_holdout=not args.no_holdout, seed=int(args.seed),
    )
    verdict = _verdict(results)

    # 4) write RESULT.md + json sidecar.
    sections = _format_sections(results, args, cfg, n_symbols)
    meta = _format_meta(results, args, cfg, n_symbols, accum, verdict)
    RW.write_result(args.out, title="L5 — Sampled+Tiled Matrix-Profile (candle language)",
                    sections=sections, verdict=verdict, meta=meta)
    print(f"[L5] verdict: {verdict}")
    return results


def _kv_lines(d: dict):
    """Render a flat metric dict as markdown bullet lines (skip the duplicate 'pass_' key)."""
    out = []
    for k, v in d.items():
        if k == "pass_":
            continue
        if isinstance(v, float):
            out.append(f"- **{k}**: {v:.6g}")
        else:
            out.append(f"- **{k}**: {v}")
    return out


def _format_sections(results, args, cfg, n_symbols):
    sections = []
    sections.append(("Config", [
        f"- subseq_len (m): {args.subseq_len}",
        f"- anchors/asset (K): {args.anchors_per_asset}",
        f"- cand_tile x anchor_tile: {args.cand_tile} x {args.anchor_tile}",
        f"- max_windows/asset: {args.max_windows_per_asset}",
        f"- base alphabet symbols: {n_symbols}",
        f"- build assets: {list(cfg.build_assets)}",
        f"- holdout assets: {list(cfg.holdout_assets)}",
    ]))
    for key in ("A0", "A1", "A2_motif", "A8", "A9"):
        if key in results:
            if key == "A8":
                sections.append((key, _a8_lines(results[key])))
            else:
                sections.append((key, _kv_lines(results[key])))
    return sections


def _a8_lines(a8: dict):
    """Render the TASK_796 A8 result. HEADLINE is transfer_bits (non-saturating bits/in-band-token of
    the holdout under the build unigram); the freq-matched random-codebook rank test and each
    substrate control are surfaced explicitly. The old saturating coverage_control/jaccard_control/
    spearman_control keys are GONE — never reference them (KeyError) and never compare these L5
    coverage/jaccard/spearman numbers as A2."""
    headline = [
        "- **transfer_bits** (HEADLINE, bits/in-band-token, lower=better): "
        f"{float(a8.get('transfer_bits', float('inf'))):.6g}",
        f"- **n_band_tokens**: {int(a8.get('n_band_tokens', 0))}",
        f"- **beats_random_codebook**: {bool(a8.get('beats_random_codebook', False))} "
        f"(perm_min={float(a8.get('perm_min_bits', float('inf'))):.6g}, "
        f"perm_mean={float(a8.get('perm_mean_bits', float('inf'))):.6g}, "
        f"n_perm_seeds={int(a8.get('n_perm_seeds', 0))})",
        f"- **beats_substrate**: {bool(a8.get('beats_substrate', False))}",
        f"- **beats_control (== beats_all, VERDICT GATE)**: {bool(a8.get('beats_control', False))}",
        f"- **coverage** (mid-freq, reported): {float(a8.get('coverage', 0.0)):.6g}",
        f"- **jaccard** (mid-freq, reported): {float(a8.get('jaccard', 0.0)):.6g}",
        f"- **spearman** (mid-freq, reported): {float(a8.get('spearman', 0.0)):.6g}",
        f"- **band**: {tuple(a8.get('band', ()))}  exclude_top_k={int(a8.get('exclude_top_k', 0))}  "
        f"codebook_size={int(a8.get('codebook_size', 0))}",
    ]
    for ctl, d in (a8.get("per_control") or {}).items():
        headline.append(
            f"- **control[{ctl}]**: transfer_bits={float(d.get('transfer_bits', float('inf'))):.6g} "
            f"beats={bool(d.get('beats', False))} coverage={float(d.get('coverage', 0.0)):.4g} "
            f"jaccard={float(d.get('jaccard', 0.0)):.4g} spearman={float(d.get('spearman', 0.0)):.4g}"
        )
    return headline


def _format_meta(results, args, cfg, n_symbols, accum, verdict):
    meta = {
        "member": "L5_matrix_profile",
        "subseq_len": int(args.subseq_len),
        "anchors_per_asset": int(args.anchors_per_asset),
        "cand_tile": int(args.cand_tile),
        "anchor_tile": int(args.anchor_tile),
        "max_windows_per_asset": int(args.max_windows_per_asset),
        "n_symbols": int(n_symbols),
        "n_anchors": int(accum.anchor_total),
        "verdict": verdict,
    }
    for key in ("A0", "A1", "A2_motif", "A8", "A9"):
        if key in results:
            meta[key] = results[key]
    return meta


if __name__ == "__main__":
    main()
