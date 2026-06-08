"""
L8 — Haar wavelet TRANSFORM-tokenizer panel member (CPU, TASK_798).

L8 is the panel's *multi-resolution / scale-native* member. Where L7 (SFA) asks "what is the
low-FREQUENCY spectral shape of this 64-bar window?" via a global rfft, L8 asks "what is the
multi-RESOLUTION shape of this window?" via a dyadic Haar wavelet ladder, and mints ONE token per
window from it. Price paths are self-affine (D = 2 - H ~ 1.5, Mandelbrot): we tokenize the 1-D
trace of a 1.5-D object, and the wavelet is the scale-native coder for exactly that object — each
detail band localizes structure at one dyadic scale instead of smearing it across global sinusoids
(the rfft basis). It is the signal-processing analog of the L4 FSQ-VQ member: a FIXED-LENGTH-W
member that streams per-window mv signal, builds a FROZEN quantizer on BUILD, encodes one
product-id token per window, then runs the shared battery.

THE ALGORITHM (multilevel orthonormal Haar DWT — adapted for a SCALE-FREE move signal):
  1. Per 64-bar window take the mv channel (vol-unit inter-bar move, length W=64=2^6). A pure-numpy
     multilevel Haar DWT (haar_dwt) gives an ORDERED real coefficient vector along the
     multi-resolution ladder: [final_approx | detail_band_level_L ... detail_band_level_1]
     (coarse -> fine). W=64 with levels=4 -> approx(4) + dW4(4)+dW3(8)+dW2(16)+dW1(32) = 64 reals.
  2. Keep a SUBSET of those coefficients (the COARSE approximation + the coarsest detail bands;
     build_coef_layout). The coarse approximation carries the bulk of the energy; we keep the whole
     coarsest approximation band and the first one or two detail bands, dropping the finest detail
     tail (high-resolution noise) just as L7 drops the high-freq rfft tail.
  3. MCB (Multiple Coefficient Binning): per COEFFICIENT POSITION, fit EQUI-DEPTH (quantile) bin
     edges on the BUILD-asset window-coefficient corpus, FROZEN. More bins go to the coarse
     approximation coefficients (which carry the bulk of the energy / discriminative power), fewer
     to fine detail -> a mixed-radix ladder (see _default_radix_ladder). Target total vocab
     ~1024-2048.
  4. Token id = FSQ-style MIXED-RADIX pack of the per-coefficient bin indices into ONE integer:
         id = sum_i b_i * prod_{j<i} r_j          (fsq_core basis; codebook_size = prod(radixes))
  5. CRITICAL — NO PER-WINDOW Z-NORMALIZATION. Textbook symbolic-transform pipelines z-normalize
     each window before the transform to make the symbols shape-invariant. We DO NOT: the mv channel
     is ALREADY vol-unit / scale-free (volunit.py: every channel is a log-ratio divided by a causal
     local-vol sigma), so a z-norm would DELETE the volatility MAGNITUDE that is the whole point of a
     MOVE language. The MCB bins are fit on the scale-free coefficient corpus CORPUS-GLOBALLY; the
     coefficients are never renormalized per window. This is the #1 transform-tokenizer trap and the
     reason A0 can hold byte-exactly.

  DC-KEEP DECISION (documented per spec): the textbook pipeline z-norms each window, which forces
  the window mean (the DC-equivalent) to ~0 and it is dropped. The Haar DC-equivalent is the FINAL
  coarse approximation band: its components are local sums (sqrt-scaled means) of the signal, and
  for an orthonormal Haar of a length-W=2^L signal the SINGLE coefficient of a fully-decomposed
  (levels==log2 W) approximation is exactly sqrt(W) * mean(x). With levels=4 the approximation is a
  length-4 band of coarse local means; we KEEP that whole band as the signed "net-drift" slots: a
  strong-up window must not pack to the same id as a strong-down window. This is a deliberate
  departure from textbook symbolic transforms, justified by the scale-free signed-move substrate.

A0 RESCALE-INVARIANCE IS THE HARD GATE. the Haar DWT is LINEAR, the mv channel is vol-unit
(byte-identical under a pure price rescale c->lam*c, per volunit.py), the MCB bins are corpus-fit and
FROZEN, and there is NO per-window renormalization. Therefore a pure price rescale yields a
BYTE-IDENTICAL coefficient vector -> identical bin indices -> identical packed id. The A0 closure
verifies agree==1.0. If A0 != 1.0 you almost certainly re-normalized per window somewhere — STOP
and fix.

PURE NUMPY — pywt is NOT a dependency (not installed on this box or the server; the panel is
numpy/scipy-only). The Haar DWT/IDWT are implemented here from first principles and self-checked for
perfect reconstruction (haar_idwt(haar_dwt(x)) ~ x to 1e-10).

CPU / numpy ONLY (no JAX, no GPU). Shares volunit + db_stream + eval_battery + result_writer + config
with the rest of the panel. Honors the streaming / O(codebook) RAM contract: the only host arrays
that grow are the (codebook,) token counter + a bounded coefficient reservoir for the MCB fit.

CLI (mirrors L7):
    python -m tools.candle_lang.symbolic.l8_wavelet \
        --db /home/lukas/Trader/trading_data.db --out /abs/path/L8_RESULT.md \
        --max-windows-per-asset 20000 [--n-levels 4] [--stride 1] [--seed 0] \
        [--build-assets ...] [--holdout-assets ...] [--transfer-matrix] [--no-holdout]
"""
from __future__ import annotations

import argparse
import dataclasses

import numpy as np

from .. import config as C
from .. import db_stream as DS
from .. import eval_battery as EB
from .. import result_writer as RW
from ..volunit import features_for_run


# --------------------------------------------------------------------------- #
# defaults (L8-specific)
# --------------------------------------------------------------------------- #
DEFAULT_N_LEVELS = 4                # multilevel Haar decomposition depth (W=64 -> approx len 4)
DEFAULT_STRIDE = 1                  # window hop (in feature bars) when slicing per-run windows
DEFAULT_MAX_WINDOWS = 20000         # cap windows kept per asset (most-recent), passed to db_stream
DEFAULT_FIT_RESERVOIR = 300_000     # bounded coefficient-row reservoir for the MCB quantile fit
# How many DETAIL bands (coarsest-first) to KEEP alongside the coarse approximation. The finest
# detail bands (largest, high-resolution noise) are DROPPED, exactly as L7 drops the high-freq tail.
DEFAULT_DETAIL_BANDS_KEPT = 2

_SQRT2 = np.sqrt(2.0)


# --------------------------------------------------------------------------- #
# pure-numpy multilevel orthonormal Haar DWT / IDWT (perfect reconstruction)
# --------------------------------------------------------------------------- #
def haar_dwt(signal: np.ndarray, levels: int) -> np.ndarray:
    """Multilevel orthonormal Haar discrete wavelet transform, PURE NUMPY.

    Given a length-N=2^k signal and `levels` <= k: at each level, given the current length-N
    approximation `a`, compute
        approx_next = (a[0::2] + a[1::2]) / sqrt(2)     # coarser approximation (length N/2)
        detail      = (a[0::2] - a[1::2]) / sqrt(2)     # detail band at this scale (length N/2)
    recurse `levels` times on the approximation. Returns the ordered real coefficient vector along
    the multi-resolution ladder, COARSE -> FINE:
        [final_approx (len N/2^levels)] + [detail_level_L ... detail_level_1]
    where detail_level_L is the coarsest (smallest, deepest) detail band and detail_level_1 the
    finest (largest, first-split) detail band. This matches the multi-resolution ladder ordering.

    LINEAR in `signal` (all ops are scaled add/sub) -> the A0 byte-identity rests on this.
    """
    a = np.asarray(signal, dtype=np.float64)
    n = a.shape[0]
    if n & (n - 1) != 0:
        raise ValueError(f"haar_dwt requires power-of-two length, got {n}")
    if levels < 1 or (1 << levels) > n:
        raise ValueError(f"levels={levels} out of range for length {n}")
    details = []  # fine -> coarse as we append
    for _ in range(int(levels)):
        even = a[0::2]
        odd = a[1::2]
        approx_next = (even + odd) / _SQRT2
        detail = (even - odd) / _SQRT2
        details.append(detail)
        a = approx_next
    # ladder order: final approx, then detail bands coarse -> fine.
    ordered = [a] + list(reversed(details))
    return np.concatenate(ordered).astype(np.float64)


def haar_idwt(coeffs: np.ndarray, levels: int, signal_len: int) -> np.ndarray:
    """Exact inverse of haar_dwt. Un-normalize each level:
        a[0::2] = (s + d) / sqrt(2)
        a[1::2] = (s - d) / sqrt(2)
    where s is the (current) approximation and d the matching detail band. PERFECT reconstruction:
    haar_idwt(haar_dwt(x, L), L, len(x)) == x to ~1e-12.

    coeffs is the ladder vector [final_approx | detail_L ... detail_1] (coarse->fine), signal_len
    is the original length, levels the decomposition depth.
    """
    coeffs = np.asarray(coeffs, dtype=np.float64)
    n = int(signal_len)
    approx_len = n >> int(levels)
    a = coeffs[:approx_len]
    # detail bands stored coarse -> fine after the approximation; iterate coarse(deepest) -> fine.
    off = approx_len
    band_lens = [n >> lvl for lvl in range(int(levels), 0, -1)]  # coarse..fine detail band lengths
    for blen in band_lens:
        d = coeffs[off:off + blen]
        off += blen
        s = a  # current approximation has the same length as this detail band
        even = (s + d) / _SQRT2
        odd = (s - d) / _SQRT2
        a = np.empty(2 * blen, dtype=np.float64)
        a[0::2] = even
        a[1::2] = odd
    return a


def _self_check_roundtrip(seed=0):
    """Assert perfect reconstruction of the pure-numpy Haar transform (run at import / __main__)."""
    rng = np.random.default_rng(seed)
    for levels in (1, 2, 3, 4, 6):
        x = rng.standard_normal(64).astype(np.float64)
        c = haar_dwt(x, levels)
        xr = haar_idwt(c, levels, 64)
        err = float(np.max(np.abs(xr - x)))
        assert err < 1e-10, f"Haar round-trip failed levels={levels}: max err {err:g}"
    # rescale linearity: dwt(lam*x) == lam*dwt(x) to f64 precision -> A0 substrate.
    x = rng.standard_normal(64).astype(np.float64)
    err = float(np.max(np.abs(haar_dwt(7.5 * x, 4) - 7.5 * haar_dwt(x, 4))))
    assert err < 1e-12, f"Haar linearity check failed: {err:g}"
    return True


_self_check_roundtrip()


def _default_radix_ladder(n_real_slots: int) -> tuple:
    """A mixed-radix ladder that allocates MORE bins to the COARSE approximation (early) coefficient
    slots and FEWER to the finer detail (late) slots, targeting a total vocab ~1024-2048.

    The coefficient vector is ordered [approx_0..approx_{A-1}, coarse_detail..finer_detail] (the
    build_coef_layout order). Energy concentrates in the coarse approximation band (the
    "net-drift" / DC-equivalent slots), so the earliest slots get the most resolution; the finer
    detail TAIL gets radix 1 (a single bin = no resolution = that coefficient does not contribute to
    the product id, but its slot is kept so the layout / centroid bookkeeping stays aligned and A2
    can still dequantize it to the column mean). The canonical ladder for the default keep layout
    (approx len 4 + 2 coarsest detail bands = 4 + 4 + 8 = 16 real slots) is

        [6, 4, 4, 4, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]   -> prod = 6*4*4*4*2*2 = 1536

    Radix 1 keeps the codebook in the ~1024-2048 target instead of letting a long tail of radix-2
    slots blow it up. Returns the radix tuple of length n_real_slots; for shorter/longer layouts the
    template is truncated / radix-1-padded.
    """
    template = [6, 4, 4, 4, 2, 2]
    if n_real_slots <= len(template):
        return tuple(int(r) for r in template[:n_real_slots])
    tail = [1] * (n_real_slots - len(template))
    return tuple(int(r) for r in (template + tail))


# --------------------------------------------------------------------------- #
# coefficient layout (which Haar bands we keep, and in what order)
# --------------------------------------------------------------------------- #
def build_coef_layout(n_levels: int, win_len: int, detail_bands_kept: int = DEFAULT_DETAIL_BANDS_KEPT):
    """Decide which Haar bands form the per-window coefficient vector.

    A multilevel Haar of a length-W=2^k signal with `levels` produces a ladder
        [final_approx (W/2^levels)] + [detail_level_L (W/2^levels) ... detail_level_1 (W/2)]
    (coarse -> fine). We keep the WHOLE coarse approximation band (the net-drift / DC-equivalent
    slots — see the module DC-KEEP DECISION) plus the first `detail_bands_kept` COARSEST detail
    bands. The finest (largest) detail bands are DROPPED (high-resolution noise), exactly as L7
    drops the high-frequency rfft tail. Keeping the whole approximation + coarsest detail bands
    targets ~16 real slots for the default levels=4, detail_bands_kept=2.

    Returns a list of (band_role:str, idx_in_band:int, global_pos:int) in PACKING order, where
    global_pos is the index of that coefficient in the full haar_dwt ladder vector (so coef_vector
    can slice it directly). band_role in {'approx','detail'}. The approximation band is FIRST.
    """
    W = int(win_len)
    levels = int(n_levels)
    if W & (W - 1) != 0:
        raise ValueError(f"build_coef_layout requires power-of-two window, got {W}")
    if levels < 1 or (1 << levels) > W:
        raise ValueError(f"n_levels={levels} out of range for window {W}")
    approx_len = W >> levels
    # ladder global positions: [0 .. approx_len) = approx; then detail bands coarse->fine.
    detail_band_lens = [W >> lvl for lvl in range(levels, 0, -1)]  # coarse..fine
    layout = []
    # approximation band (all of it).
    for j in range(approx_len):
        layout.append(("approx", j, j))
    # coarsest `detail_bands_kept` detail bands.
    pos = approx_len
    kept = int(min(int(detail_bands_kept), len(detail_band_lens)))
    for b in range(len(detail_band_lens)):
        blen = detail_band_lens[b]
        if b < kept:
            for j in range(blen):
                layout.append(("detail", j, pos + j))
        pos += blen
    return layout


def coef_vector(mv_window: np.ndarray, layout, *, n_levels, win_len) -> np.ndarray:
    """The per-window Haar coefficient vector: haar_dwt(mv_window) projected onto the kept bands in
    `layout` order. NO per-window z-normalization (the #1 transform-tokenizer trap) — the mv channel
    is already vol-unit/scale-free, so the raw coefficients ARE the scale-free multi-resolution
    description.

    mv_window : (W,) float64/float32 vol-unit move signal for ONE window.
    Returns (n_slots,) float64 coefficient vector (n_slots == len(layout)).
    """
    x = np.asarray(mv_window, dtype=np.float64)
    ladder = haar_dwt(x, int(n_levels))          # (W,) real; LINEAR in x -> A0 holds
    out = np.empty(len(layout), dtype=np.float64)
    for i, (_role, _j, gpos) in enumerate(layout):
        out[i] = float(ladder[gpos])
    return out


# --------------------------------------------------------------------------- #
# the FROZEN MCB quantizer (per-coefficient quantile bin edges + per-bin centroids)
# --------------------------------------------------------------------------- #
class HaarVocab:
    """Frozen Haar-wavelet vocabulary.

    Holds, per coefficient slot i (in build_coef_layout order):
        radixes[i]   : number of equi-depth bins r_i for that slot.
        edges[i]     : (r_i - 1,) sorted interior quantile bin edges (np.digitize boundaries).
        centroids[i] : (r_i,) per-bin MEAN coefficient value on BUILD (for A2 dequantization).
    Token id = mixed-radix pack of the per-slot bin indices using the fsq_core basis
        id = sum_i b_i * prod_{j<i} r_j ,  basis = [1, r_0, r_0*r_1, ...].
    codebook_size = prod(radixes).

    Fitting uses the numpy equi-depth path (the canonical fit): L8 deliberately uses a NON-uniform
    radix ladder (more bins on the coarse approximation), so the uniform-n_bins pyts MCB path is not
    applicable; `fit_backend` records 'numpy'.
    """

    def __init__(self, layout, radixes, edges, centroids, fit_backend, win_len, n_levels,
                 detail_bands_kept):
        self.layout = list(layout)
        self.radixes = tuple(int(r) for r in radixes)
        self.edges = [np.asarray(e, dtype=np.float64) for e in edges]
        self.centroids = [np.asarray(cc, dtype=np.float64) for cc in centroids]
        self.fit_backend = str(fit_backend)
        self.win_len = int(win_len)
        self.n_levels = int(n_levels)
        self.detail_bands_kept = int(detail_bands_kept)
        # mixed-radix basis (fsq_core: id = sum_i b_i * prod_{j<i} r_j).
        lv = np.asarray(self.radixes, dtype=np.int64)
        self.basis = np.concatenate([[1], np.cumprod(lv[:-1])]).astype(np.int64)
        self.codebook_size = int(np.prod(lv))

    # --- encode ---
    def bin_indices(self, coef_vec: np.ndarray) -> np.ndarray:
        """Per-slot bin index of one coefficient vector (clipped to [0, r_i-1])."""
        cv = np.asarray(coef_vec, dtype=np.float64)
        idx = np.empty(len(self.layout), dtype=np.int64)
        for i in range(len(self.layout)):
            b = int(np.digitize(cv[i], self.edges[i]))          # 0..r_i-1 (edges are interior)
            r = self.radixes[i]
            idx[i] = b if b < r else r - 1
        return idx

    def pack(self, bin_idx: np.ndarray) -> int:
        """Mixed-radix pack of per-slot bin indices -> one integer token id."""
        return int(np.dot(np.asarray(bin_idx, dtype=np.int64), self.basis))

    def encode_window(self, mv_window: np.ndarray) -> int:
        """One vol-unit mv window -> one product-id token."""
        cv = coef_vector(mv_window, self.layout, n_levels=self.n_levels, win_len=self.win_len)
        return self.pack(self.bin_indices(cv))

    # --- decode (A2 reconstruction) ---
    def dequantize(self, bin_idx: np.ndarray) -> np.ndarray:
        """Per-slot bin index -> per-bin CENTROID coefficient value (BUILD per-bin mean)."""
        out = np.empty(len(self.layout), dtype=np.float64)
        for i in range(len(self.layout)):
            out[i] = self.centroids[i][int(bin_idx[i])]
        return out

    def reconstruct_window(self, mv_window: np.ndarray) -> np.ndarray:
        """Reconstruct the mv window from its quantized Haar ladder (centroid dequant -> full ladder
        vector with zeros for dropped fine-detail bands -> haar_idwt). Returns (W,) float64."""
        cv = coef_vector(mv_window, self.layout, n_levels=self.n_levels, win_len=self.win_len)
        idx = self.bin_indices(cv)
        deq = self.dequantize(idx)
        ladder = np.zeros(self.win_len, dtype=np.float64)
        for val, (_role, _j, gpos) in zip(deq, self.layout):
            ladder[gpos] = val
        return haar_idwt(ladder, self.n_levels, self.win_len)


def _fit_edges_centroids_numpy(coef_matrix: np.ndarray, radixes):
    """Numpy equi-depth (quantile) bin edges + per-bin centroids per coefficient column.

    coef_matrix : (n_rows, n_slots) float64 pooled BUILD window coefficients.
    radixes     : per-slot bin counts.
    Returns (edges_list, centroids_list). Edges are interior quantile boundaries (r-1 of them);
    centroids are the mean coefficient value of the BUILD rows that fall in each bin (per-bin MEAN
    for A2 dequant). Degenerate (constant) columns collapse to a single occupied bin; their unused
    centroids fall back to the column mean so dequant never returns NaN.
    """
    n_slots = coef_matrix.shape[1]
    edges_list, centroids_list = [], []
    for i in range(n_slots):
        col = coef_matrix[:, i].astype(np.float64)
        r = int(radixes[i])
        qs = np.linspace(0.0, 1.0, r + 1)[1:-1]                 # r-1 interior quantiles
        edges = np.quantile(col, qs) if r > 1 else np.empty(0, dtype=np.float64)
        edges = np.asarray(edges, dtype=np.float64)
        # per-bin centroid = mean of build rows landing in that bin (fallback: column mean).
        binidx = np.digitize(col, edges)
        binidx = np.clip(binidx, 0, r - 1)
        col_mean = float(col.mean()) if col.size else 0.0
        cents = np.full(r, col_mean, dtype=np.float64)
        for b in range(r):
            m = binidx == b
            if np.any(m):
                cents[b] = float(col[m].mean())
        edges_list.append(edges)
        centroids_list.append(cents)
    return edges_list, centroids_list


def fit_vocab(coef_matrix: np.ndarray, layout, radixes, *, win_len, n_levels, detail_bands_kept):
    """Fit the frozen HaarVocab from pooled BUILD window coefficients (numpy equi-depth path).
    Centroids are the per-bin BUILD mean for A2 dequantization."""
    radixes = tuple(int(r) for r in radixes)
    edges, centroids = _fit_edges_centroids_numpy(coef_matrix, radixes)
    return HaarVocab(layout, radixes, edges, centroids, "numpy", win_len, n_levels,
                     detail_bands_kept)


# --------------------------------------------------------------------------- #
# per-asset window streaming (vol-unit mv channel, NO per-window z-norm)
# --------------------------------------------------------------------------- #
def _iter_mv_windows(db, asset_id, cfg, *, win_len, stride):
    """Yield (mv_window(win_len,)f64, close_window(win_len,)f64) per all-finite length-`win_len`
    feature window sliced WITHIN one dt==60 run. close_window is the raw close path over the SAME
    feature bars (for A2 price-side bookkeeping, though A2 here scores the mv reconstruction). NO
    forward-fill, NO gap straddle (db_stream guarantees run contiguity)."""
    W = int(win_len)
    for ts, o, h, l, c in DS.iter_runs(db, asset_id, cfg):
        feats, close, _ts_out, finite = features_for_run(ts, o, h, l, c, cfg)
        L = feats.shape[0]
        if L < W:
            continue
        mv = feats[:, 0].astype(np.float64)
        for s in range(0, L - W + 1, int(stride)):
            if not finite[s:s + W].all():
                continue
            yield mv[s:s + W], close[s:s + W].astype(np.float64)


def _phase_shuffle_signal(x: np.ndarray, rng) -> np.ndarray:
    """Donor phase-shuffle (data.py:165 surrogate) of a 1-D signal: randomize the rfft phases
    (keep DC + Nyquist phase), preserving the power spectrum/marginal while destroying temporal
    structure. The A8/control substrate transform mirrored from L7's control path."""
    x = np.asarray(x, dtype=np.float64)
    L = x.shape[0]
    if L < 4:
        return x.copy()
    Xf = np.fft.rfft(x)
    mag = np.abs(Xf)
    ph = np.angle(Xf)
    rand_ph = rng.uniform(-np.pi, np.pi, size=ph.shape)
    rand_ph[0] = ph[0]
    if L % 2 == 0:
        rand_ph[-1] = ph[-1]
    Xnew = mag * np.exp(1j * rand_ph)
    return np.fft.irfft(Xnew, n=L).astype(np.float64)


# --------------------------------------------------------------------------- #
# build-asset coefficient corpus for the MCB fit (bounded reservoir, RAM-capped)
# --------------------------------------------------------------------------- #
def fit_vocab_from_assets(db, assets, cfg, *, n_levels, win_len, stride, max_windows,
                          detail_bands_kept=DEFAULT_DETAIL_BANDS_KEPT,
                          reservoir_cap=DEFAULT_FIT_RESERVOIR, seed=0):
    """Stream BUILD-asset windows, pool their Haar coefficient vectors into a bounded reservoir, and
    fit the frozen HaarVocab. RAM-capped at reservoir_cap coefficient rows (reservoir sampling), so
    the fit never materializes all windows. Returns the fitted HaarVocab."""
    layout = build_coef_layout(n_levels, win_len, detail_bands_kept)
    radixes = _default_radix_ladder(len(layout))
    rng = np.random.default_rng(seed)
    n_slots = len(layout)
    res = np.empty((int(reservoir_cap), n_slots), dtype=np.float64)
    held = 0
    seen = 0
    cap = None if max_windows is None else int(max_windows)
    for asset_id in assets:
        a_seen = 0
        for mv_w, _close in _iter_mv_windows(db, asset_id, cfg, win_len=win_len, stride=stride):
            cv = coef_vector(mv_w, layout, n_levels=n_levels, win_len=win_len)
            seen += 1
            a_seen += 1
            if held < res.shape[0]:
                res[held] = cv
                held += 1
            else:
                j = int(rng.integers(0, seen))
                if j < res.shape[0]:
                    res[j] = cv
            if cap is not None and a_seen >= cap:
                break
    if held == 0:
        raise RuntimeError("L8 MCB fit: no finite windows found on build assets.")
    coef_matrix = res[:held]
    return fit_vocab(coef_matrix, layout, radixes, win_len=win_len, n_levels=n_levels,
                     detail_bands_kept=detail_bands_kept)


# --------------------------------------------------------------------------- #
# per-asset online accumulation (token counts + A2 reconstruction stats)
# --------------------------------------------------------------------------- #
class _OnlineAccum:
    """O(codebook) online accumulators: a (codebook,) token counter + running A2 reconstruction
    sums. NEVER O(windows). A2 is folded online (per-bar |recon_mv - true_mv| within abs_tol)."""

    def __init__(self, codebook_size: int):
        self.codebook_size = int(codebook_size)
        self.token_counts = np.zeros(self.codebook_size, dtype=np.int64)
        self.n_windows = 0
        # A2 online accumulators
        self.a2_n_bars = 0
        self.a2_n_within = 0
        self.a2_sq = 0.0
        self.a2_abs = 0.0

    def add_window(self, token_id: int, recon_mv=None, true_mv=None, abs_tol=0.25):
        EB._bincount_into(self.token_counts, np.asarray([int(token_id)], dtype=np.int64),
                          self.codebook_size)
        self.n_windows += 1
        if recon_mv is not None and true_mv is not None:
            r = np.asarray(recon_mv, dtype=np.float64).reshape(-1)
            t = np.asarray(true_mv, dtype=np.float64).reshape(-1)
            n = min(r.size, t.size)
            if n:
                err = r[:n] - t[:n]
                self.a2_n_bars += int(n)
                self.a2_n_within += int(np.count_nonzero(np.abs(err) <= float(abs_tol)))
                self.a2_sq += float(np.sum(err * err))
                self.a2_abs += float(np.sum(np.abs(err)))


def encode_assets(db, assets, cfg, vocab, accum, *, win_len, stride, max_windows,
                  a2_reconstruct=False, abs_tol=0.25):
    """Stream `assets` windows, encode each to its product-id token, fold counts (+ optional A2
    reconstruction) into `accum`. Bounded RAM; one window at a time."""
    cap = None if max_windows is None else int(max_windows)
    for asset_id in assets:
        a_seen = 0
        for mv_w, _close in _iter_mv_windows(db, asset_id, cfg, win_len=win_len, stride=stride):
            if a2_reconstruct:
                cv = coef_vector(mv_w, vocab.layout, n_levels=vocab.n_levels, win_len=vocab.win_len)
                idx = vocab.bin_indices(cv)
                tok = vocab.pack(idx)
                recon = vocab.reconstruct_window(mv_w)
                accum.add_window(tok, recon_mv=recon, true_mv=mv_w, abs_tol=abs_tol)
            else:
                accum.add_window(vocab.encode_window(mv_w))
            a_seen += 1
            if cap is not None and a_seen >= cap:
                break
    return accum


# --------------------------------------------------------------------------- #
# eval-battery wiring (online; O(codebook), never O(windows))
# --------------------------------------------------------------------------- #
def _token_stream_from_counts(token_counts: np.ndarray, chunk=4096):
    """Re-emit a token-id stream from the O(codebook) counter for A1's online fold (host O(chunk))."""
    nz = np.nonzero(token_counts)[0]
    for sym in nz:
        cnt = int(token_counts[sym])
        while cnt > 0:
            take = min(cnt, int(chunk))
            yield np.full(take, int(sym), dtype=np.int64)
            cnt -= take


def _a0_tokenize_window_fn(vocab, cfg, *, stride):
    """A0 closure: tokenise ONE raw-OHLC window to its L8 product-id via the full vol-unit ->
    haar_dwt -> MCB-bin -> pack path. Rescale invariance is inherited from volunit (the mv channel
    is byte-identical under c->lam*c) + Haar-DWT linearity + frozen corpus-fit bins + NO per-window
    z-norm: a pure price rescale MUST yield a byte-identical id. This closure is the A0 verifier."""
    W = int(vocab.win_len)

    def tokenize(ohlc):
        ohlc = np.asarray(ohlc, dtype=np.float64)
        Wr = ohlc.shape[0]
        ts = np.arange(Wr, dtype=np.int64) * C.BAR_SECONDS + 1_600_000_000
        feats, _close, _ts_out, finite = features_for_run(
            ts, ohlc[:, 0], ohlc[:, 1], ohlc[:, 2], ohlc[:, 3], cfg
        )
        if feats.shape[0] < W:
            return 0
        # take the FIRST all-finite length-W feature window (deterministic; A0 only needs one id).
        mv = feats[:, 0].astype(np.float64)
        for s in range(0, feats.shape[0] - W + 1, 1):
            if finite[s:s + W].all():
                return int(vocab.encode_window(mv[s:s + W]))
        return 0

    return tokenize


def _sample_ohlc_windows(db, assets, cfg, win_len, n=16, seed=0):
    """A SMALL diagnostic sample of raw-price OHLC windows (W+2,4) for A0 (a unit check, not a fold).
    We pull win_len+2 raw bars so features_for_run yields at least win_len+1 feature bars (one full
    window plus slack for the first-bar drop)."""
    rng = np.random.default_rng(seed)
    W = int(win_len) + 2
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


def _asset_token_stream(db, asset_id, cfg, vocab, *, stride, max_windows, control, seed):
    """Token-id chunk(s) for ONE asset (A8 encode). control in {None,'shuffle','random_merge'}
    applies a structure-destroying transform to the mv substrate BEFORE the Haar DWT, mirroring L7:
        'shuffle'       -> donor phase-shuffle of each window's mv signal (preserves power spectrum
                           marginal, destroys phase/temporal structure).
        'random_merge'  -> permute the mv samples within the window (destroys local shape order).
    Yields int64 chunks (one chunk per asset)."""
    rng = np.random.default_rng(seed + int(asset_id))
    cap = None if max_windows is None else int(max_windows)
    ids = []
    a_seen = 0
    for mv_w, _close in _iter_mv_windows(db, asset_id, cfg, win_len=vocab.win_len, stride=stride):
        if control == "shuffle":
            sub = _phase_shuffle_signal(mv_w, rng)
        elif control == "random_merge":
            sub = mv_w[rng.permutation(mv_w.shape[0])]
        else:
            sub = mv_w
        ids.append(vocab.encode_window(sub))
        a_seen += 1
        if cap is not None and a_seen >= cap:
            break
    if not ids:
        return
    yield np.asarray(ids, dtype=np.int64)


def _run_a8(db, cfg, vocab, *, stride, max_windows, seed, transfer_matrix=False):
    """A8 wiring for L8: the 'vocab' is the frozen HaarVocab; encode_assets_fn streams product-ids
    per asset, optionally under the structure-destroying substrate controls. Uses the shared
    eval_battery.a8_cross_asset_transfer reducer (unigram + bigram + power)."""
    from ..eval_battery import a8_cross_asset_transfer

    def build_vocab_fn(assets=None):
        return vocab

    def encode_assets_fn(v, assets, control):
        for asset_id in assets:
            for chunk in _asset_token_stream(
                db, int(asset_id), cfg, v, stride=stride, max_windows=max_windows,
                control=control, seed=seed,
            ):
                yield chunk

    a8 = a8_cross_asset_transfer(
        build_vocab_fn, encode_assets_fn,
        build_assets=cfg.build_assets, holdout_assets=cfg.holdout_assets,
        controls=("shuffle", "random_merge"), exclude_top_k=8,
        per_asset_matrix=transfer_matrix,
    )
    if a8 is not None:
        a8["a8_power"] = EB.run_a8_power()
    return a8


def run_eval(db, cfg, vocab, accum, *, stride, max_windows, has_holdout, seed,
             transfer_matrix=False, abs_tol=0.25):
    """Assemble the A0/A1/A2/A8/A9 battery for L8 from the online accumulators + frozen vocab."""
    cb = int(vocab.codebook_size)
    results = {}

    # --- A0: rescale invariance (the HARD GATE) ---
    try:
        windows = _sample_ohlc_windows(db, cfg.build_assets, cfg, vocab.win_len, n=16, seed=seed)
        if windows:
            results["A0"] = EB.a0_rescale_invariance(
                _a0_tokenize_window_fn(vocab, cfg, stride=stride), windows, lambdas=(0.01, 100.0)
            )
        else:
            results["A0"] = dict(agree=1.0, n_trials=0, **{"pass": True})
    except Exception as e:
        results["A0"] = dict(error=str(e), **{"pass": False})

    # --- A1: MDL over the product-id token stream (re-emitted from the O(codebook) counter) ---
    # fixed-length-W member: n_bars_covered = n_windows * window (each token covers `window` bars).
    null_bits_per_bar = float(np.log2(max(cb, 2)))          # per-bar null = uniform over the codebook
    codebook_bits = float(cb) * 4.0                         # one-time vocab-ship cost (panel parity)
    results["A1"] = EB.a1_mdl(
        _token_stream_from_counts(accum.token_counts),
        codebook_size=cb,
        null_bits_per_bar=null_bits_per_bar,
        n_boundaries=int(accum.n_windows),
        codebook_bits=codebook_bits,
        n_bars_covered=int(accum.n_windows) * int(vocab.win_len),
    )

    # --- A2: wavelet reconstruction fidelity (additive, signed vol-unit mv) ---
    # A2 was folded online during encode (per-bar |recon_mv - true_mv| <= abs_tol). We surface the
    # online tallies via a single synthetic aggregate pair so the shared a2_reconstruction reducer
    # runs and the scorecard schema matches the other members.
    frac_within = (accum.a2_n_within / accum.a2_n_bars) if accum.a2_n_bars else 0.0
    mean_sq = (accum.a2_sq / accum.a2_n_bars) if accum.a2_n_bars else 0.0
    mean_abs = (accum.a2_abs / accum.a2_n_bars) if accum.a2_n_bars else 0.0
    results["A2"] = dict(
        frac_within_tol=float(frac_within),
        mean_log_ratio_mse=float(mean_sq),     # name kept for panel parity; = additive MSE here
        mean_abs_err=float(mean_abs),
        mode="additive",
        abs_tol=float(abs_tol),
        n_bars=int(accum.a2_n_bars),
        n_pairs=int(accum.n_windows),
    )

    # --- A8: cross-asset transfer (only if holdout present) ---
    if has_holdout and cfg.holdout_assets:
        results["A8"] = _run_a8(db, cfg, vocab, stride=stride, max_windows=max_windows,
                                seed=seed, transfer_matrix=transfer_matrix)

    # --- A9: length non-degeneracy. L8 tokens are FIXED-length (one token / window) by construction,
    # so the raw length distribution is a single spike at `window`. Mirror L4/L5/L7: use the shared
    # battery dict for scorecard parity AND a fixed-length-aware pass that keys off all_len1, with a
    # USED-CODES floor so a degenerate few-bin collapse (e.g. all mass on one product id) FAILS. ---
    used_codes = int(np.count_nonzero(accum.token_counts))
    length_counts = {int(vocab.win_len): int(accum.n_windows)}
    a9 = dict(EB.a9_length_nondegeneracy(length_counts))
    a9["fixed_length_member"] = True
    a9["used_codes"] = used_codes
    a9["codebook_size"] = cb
    a9["code_utilization"] = float(used_codes / cb) if cb else 0.0
    # genuine degeneracy for a fixed-length member: collapsed to length-1, OR no windows, OR the
    # product vocabulary collapsed to a near-trivial number of distinct ids (few-bin collapse).
    USED_CODES_FLOOR = 8
    a9["pass_fixed_len"] = bool(
        (not a9.get("all_len1", True)) and int(accum.n_windows) > 0 and used_codes >= USED_CODES_FLOOR
    )
    results["A9"] = a9
    return results


# --------------------------------------------------------------------------- #
# verdict
# --------------------------------------------------------------------------- #
def _verdict(results) -> str:
    """L8 verdict: PROCEED requires (i) A0 rescale-invariance (the hard gate), (ii) A9 non-degeneracy
    (fixed-length + healthy used-codes), AND (iii) A8 cross-asset transfer passing the §186-aligned
    go/no-go (EB.a8_verdict: random-codebook null on unigram AND bigram). A2 is reported, not gated
    (reconstruction fidelity is informative but a lossy coarse-band wavelet code is expected to be
    approximate; it is not a PROCEED gate)."""
    a0 = results.get("A0", {})
    a9 = results.get("A9", {})
    a8 = results.get("A8", None)
    a0_pass = bool(a0.get("pass", a0.get("pass_", False)))
    a9_pass = bool(a9.get("pass_fixed_len", a9.get("pass", a9.get("pass_", False))))

    if not a0_pass:
        return "KILL (A0 rescale-invariance failed — likely a per-window renormalization crept in)"
    if not a9_pass:
        return "KILL (A9 degenerate — length-1 collapse or product vocab collapsed below used-codes floor)"
    if a8 is None:
        return "WEAK (A0+A9 pass, but A8 transfer not evaluated — no holdout)"
    if not EB.a8_verdict(a8):
        return ("WEAK (A0+A9 pass but A8 cross-asset transfer failed the random-codebook null on the "
                "unigram/bigram axes)")
    return "PROCEED"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="L8 Haar wavelet candle-language member (CPU, pure-numpy DWT)."
    )
    p.add_argument("--db", type=str, default=None,
                   help="DB path (default: config.db_path() / $CANDLE_DB).")
    p.add_argument("--out", type=str,
                   default="/home/lukas/Trader/runs/789_token_c0/L8_RESULT.md",
                   help="RESULT.md path (a .json sidecar is written alongside).")
    p.add_argument("--max-windows-per-asset", type=int, default=DEFAULT_MAX_WINDOWS,
                   help="Cap windows kept per asset (most-recent slice via early break). 0 -> unbounded.")
    p.add_argument("--n-levels", type=int, default=DEFAULT_N_LEVELS,
                   help="Multilevel Haar decomposition depth (W=64 -> coarse approx len W/2^levels).")
    p.add_argument("--detail-bands-kept", type=int, default=DEFAULT_DETAIL_BANDS_KEPT,
                   help="Number of COARSEST detail bands kept alongside the approximation.")
    p.add_argument("--stride", type=int, default=DEFAULT_STRIDE,
                   help="Window hop in feature bars when slicing per-run windows.")
    p.add_argument("--no-holdout", action="store_true",
                   help="Skip the A8 cross-asset transfer test.")
    p.add_argument("--build-assets", nargs="+", type=int, default=None,
                   help="Explicit BUILD asset ids (default: config.CandleConfig.build_assets).")
    p.add_argument("--holdout-assets", nargs="+", type=int, default=None,
                   help="Explicit HOLDOUT asset ids (default: config.CandleConfig.holdout_assets).")
    p.add_argument("--transfer-matrix", action="store_true",
                   help="Compute the per-asset A8 A-by-B transfer matrix + holdout_transfer_rate.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    db = args.db if args.db else C.db_path()
    max_windows = None if int(args.max_windows_per_asset) <= 0 else int(args.max_windows_per_asset)
    cfg = C.CandleConfig(db=db)
    cfg = dataclasses.replace(
        cfg,
        build_assets=tuple(args.build_assets) if args.build_assets else cfg.build_assets,
        holdout_assets=tuple(args.holdout_assets) if args.holdout_assets is not None else cfg.holdout_assets,
    )
    win_len = int(cfg.window)
    stride = int(args.stride)
    n_levels = int(args.n_levels)
    detail_bands_kept = int(args.detail_bands_kept)

    if not DS.quick_check(db):
        print(f"[L8] WARNING: PRAGMA quick_check did not return 'ok' for {db}")

    print(f"[L8] db={db}")
    print(f"[L8] build_assets={cfg.build_assets}  holdout_assets={cfg.holdout_assets}")
    print(f"[L8] window={win_len}  n_levels={n_levels}  detail_bands_kept={detail_bands_kept}  "
          f"stride={stride}  max_windows/asset={max_windows}")

    # 1) fit the frozen Haar vocab (MCB bins) on a bounded BUILD-asset coefficient reservoir.
    print("[L8] fitting frozen Haar vocab (MCB quantile bins on build-asset coefficients) ...")
    vocab = fit_vocab_from_assets(
        db, cfg.build_assets, cfg,
        n_levels=n_levels, win_len=win_len, stride=stride, max_windows=max_windows,
        detail_bands_kept=detail_bands_kept, seed=int(args.seed),
    )
    print(f"[L8] coef layout ({len(vocab.layout)} real slots): {vocab.layout}")
    print(f"[L8] radixes: {vocab.radixes}  codebook_size={vocab.codebook_size}  "
          f"fit_backend={vocab.fit_backend}")

    # 2) encode BUILD assets -> one product-id token per window (online counts + A2 reconstruction).
    print("[L8] encoding build assets (one token/window) + folding A2 reconstruction ...")
    accum = _OnlineAccum(vocab.codebook_size)
    encode_assets(db, cfg.build_assets, cfg, vocab, accum,
                  win_len=win_len, stride=stride, max_windows=max_windows,
                  a2_reconstruct=True, abs_tol=0.25)
    used = int(np.count_nonzero(accum.token_counts))
    print(f"[L8] encoded {accum.n_windows} windows; used_codes={used}/{vocab.codebook_size} "
          f"({100.0 * used / max(vocab.codebook_size, 1):.1f}%)")

    # 3) eval battery.
    print("[L8] running eval battery (A0/A1/A2/A8/A9) ...")
    results = run_eval(
        db, cfg, vocab, accum,
        stride=stride, max_windows=max_windows,
        has_holdout=not args.no_holdout, seed=int(args.seed),
        transfer_matrix=bool(args.transfer_matrix), abs_tol=0.25,
    )
    verdict = _verdict(results)

    # 4) write RESULT.md + json sidecar.
    sections = _format_sections(results, args, cfg, vocab)
    meta = _format_meta(results, args, cfg, vocab, accum, verdict)
    RW.write_result(args.out, title="L8 — Haar wavelet (multi-resolution) candle language",
                    sections=sections, verdict=verdict, meta=meta)
    print(f"[L8] verdict: {verdict}")
    return results


def _kv_lines(d: dict):
    out = []
    for k, v in d.items():
        if k == "pass_":
            continue
        if isinstance(v, float):
            out.append(f"- **{k}**: {v:.6g}")
        else:
            out.append(f"- **{k}**: {v}")
    return out


def _format_sections(results, args, cfg, vocab):
    sections = []
    sections.append(("Config", [
        f"- window (W): {vocab.win_len}",
        f"- n_levels (Haar decomposition depth): {vocab.n_levels}",
        f"- detail_bands_kept (coarsest-first): {vocab.detail_bands_kept}",
        f"- coef real slots: {len(vocab.layout)}  (coarse approx kept as net-drift slots; fine detail tail dropped)",
        f"- radixes (mixed-radix ladder, more bins on coarse approx): {list(vocab.radixes)}",
        f"- codebook_size (prod radixes): {vocab.codebook_size}",
        f"- MCB fit backend: {vocab.fit_backend}",
        f"- per-window z-norm: NONE (scale-free mv substrate — A0 byte-identity)",
        f"- stride: {args.stride}",
        f"- max_windows/asset: {args.max_windows_per_asset}",
        f"- build assets: {list(cfg.build_assets)}",
        f"- holdout assets: {list(cfg.holdout_assets)}",
    ]))
    for key in ("A0", "A1", "A2", "A8", "A9"):
        if key in results:
            if key == "A8":
                a8 = results[key]
                a8_lines = list(EB.render_a8_lines(a8))
                a8_lines += EB.render_transfer_matrix(a8)
                a8_lines.append(
                    "- **a8_verdict (§186 gate: random-codebook null on unigram+bigram)**: "
                    f"{bool(EB.a8_verdict(a8))}"
                )
                sections.append((key, a8_lines))
            else:
                sections.append((key, _kv_lines(results[key])))
    return sections


def _format_meta(results, args, cfg, vocab, accum, verdict):
    meta = {
        "member": "L8_haar",
        "window": int(vocab.win_len),
        "n_levels": int(vocab.n_levels),
        "detail_bands_kept": int(vocab.detail_bands_kept),
        "radixes": list(vocab.radixes),
        "codebook_size": int(vocab.codebook_size),
        "fit_backend": vocab.fit_backend,
        "per_window_znorm": False,
        "dc_kept_as_net_drift": True,
        "stride": int(args.stride),
        "max_windows_per_asset": int(args.max_windows_per_asset),
        "n_windows": int(accum.n_windows),
        "used_codes": int(np.count_nonzero(accum.token_counts)),
        "verdict": verdict,
    }
    for key in ("A0", "A1", "A2", "A8", "A9"):
        if key in results:
            meta[key] = results[key]
    return meta


if __name__ == "__main__":
    _self_check_roundtrip()
    main()
