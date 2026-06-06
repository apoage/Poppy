"""
L7 — SFA (Symbolic Fourier Approximation) TRANSFORM-tokenizer panel member (CPU, TASK_798).

L7 is the panel's *spectral* member. Where L5 (matrix profile) asks "does this shape RECUR?"
on the raw vol-unit signal, L7 asks "what is the low-frequency SPECTRAL SHAPE of this 64-bar
window?" and mints ONE token per window from it. It is the signal-processing analog of the
L4 FSQ-VQ member: a FIXED-LENGTH-W member that streams per-window mv signal, builds a FROZEN
quantizer on BUILD, encodes one product-id token per window, then runs the shared battery.

THE ALGORITHM (classic SFA — Schäfer & Högqvist 2012 — adapted for a SCALE-FREE move signal):
  1. Per 64-bar window take the mv channel (vol-unit inter-bar move, length ~W). scipy.fft.rfft
     -> ~W//2+1 complex coefficients.
  2. Keep the first l=N_COEF_KEEP LOW-frequency coefficients. Split into REAL & IMAG parts. The
     DC term (coef 0) has imag==0 and the Nyquist term (only present when the rfft length makes
     it real) has imag==0 -> those imag slots are DROPPED (they carry no information). We KEEP
     the DC REAL part as a "net-drift" slot (see DC-KEEP DECISION below). The resulting ordered
     real-valued vector is the per-window SFA coefficient vector.
  3. MCB (Multiple Coefficient Binning): per COEFFICIENT POSITION, fit EQUI-DEPTH (quantile) bin
     edges on the BUILD-asset window-coefficient corpus, FROZEN. More levels go to low-freq
     coefficients (which carry the bulk of the energy / discriminative power), fewer to high-freq
     -> a mixed-radix ladder (see _default_radix_ladder). Target total vocab ~1024-2048.
  4. Token id = FSQ-style MIXED-RADIX pack of the per-coefficient bin indices into ONE integer:
         id = sum_i b_i * prod_{j<i} r_j          (fsq_core basis; codebook_size = prod(radixes))
  5. CRITICAL — NO PER-WINDOW Z-NORMALIZATION. Textbook SFA z-normalizes each window before the
     DFT to make the symbols shape-invariant. We DO NOT: the mv channel is ALREADY vol-unit /
     scale-free (volunit.py: every channel is a log-ratio divided by a causal local-vol sigma),
     so SFA's z-norm would DELETE the volatility MAGNITUDE that is the whole point of a MOVE
     language. The MCB bins are fit on the scale-free coefficient corpus CORPUS-GLOBALLY; the
     coefficients are never renormalized per window. This is the #1 SFA trap and the reason A0
     can hold byte-exactly.

  DC-KEEP DECISION (documented per spec): the textbook SFA pipeline z-norms each window, which
  forces the DC coefficient to ~0 and it is dropped. We KEEP the DC REAL part as an explicit
  "net-drift" coefficient slot: this is a signed MOVE signal and the net direction/drift of a
  64-bar window over its local-vol baseline is genuine, discriminative structure (a strong-up
  window must not pack to the same id as a strong-down window). The DC IMAG is identically 0 and
  is dropped. This is a deliberate departure from textbook SFA, justified by the scale-free
  signed-move substrate.

A0 RESCALE-INVARIANCE IS THE HARD GATE. rfft is LINEAR, the mv channel is vol-unit (byte-identical
under a pure price rescale c->lam*c, per volunit.py), the MCB bins are corpus-fit and FROZEN, and
there is NO per-window renormalization. Therefore a pure price rescale yields a BYTE-IDENTICAL
coefficient vector -> identical bin indices -> identical packed id. The A0 closure verifies
agree==1.0. If A0 != 1.0 you almost certainly re-normalized per window somewhere — STOP and fix.

CPU / numpy + scipy ONLY (no JAX, no GPU). Shares volunit + db_stream + eval_battery + result_writer
+ config with the rest of the panel. Honors the streaming / O(codebook) RAM contract: the only host
arrays that grow are the (codebook,) token counter + a bounded coefficient reservoir for the MCB fit.

CLI (mirrors L5):
    python -m tools.candle_lang.symbolic.l7_sfa \
        --db /home/lukas/Trader/trading_data.db --out /abs/path/L7_RESULT.md \
        --max-windows-per-asset 20000 [--n-coef 8] [--stride 1] [--seed 0] \
        [--build-assets ...] [--holdout-assets ...] [--transfer-matrix] [--no-holdout]
"""
from __future__ import annotations

import argparse
import dataclasses

import numpy as np
from scipy.fft import rfft, irfft

from .. import config as C
from .. import db_stream as DS
from .. import eval_battery as EB
from .. import result_writer as RW
from ..volunit import features_for_run


# --------------------------------------------------------------------------- #
# defaults (L7-specific)
# --------------------------------------------------------------------------- #
DEFAULT_N_COEF = 8                  # l = number of LOW-frequency rfft coefficients kept
DEFAULT_STRIDE = 1                  # window hop (in feature bars) when slicing per-run windows
DEFAULT_MAX_WINDOWS = 20000         # cap windows kept per asset (most-recent), passed to db_stream
DEFAULT_FIT_RESERVOIR = 300_000     # bounded coefficient-row reservoir for the MCB quantile fit


def _default_radix_ladder(n_real_slots: int) -> tuple:
    """A mixed-radix ladder that allocates MORE bins to the low-frequency (early) coefficient
    slots and FEWER to high-frequency (late) slots, targeting a total vocab ~1024-2048.

    The real-valued coefficient vector is ordered [DC_real, c1_real, c1_imag, c2_real, c2_imag, ...]
    (the build_coef_layout order). Energy concentrates in the low-frequency coefficients, so the
    earliest slots get the most resolution; the HIGH-FREQUENCY TAIL gets radix 1 (a single bin =
    no resolution = that coefficient does not contribute to the product id, but its slot is kept so
    the layout / centroid bookkeeping stays aligned and A2 can still dequantize it to the column
    mean). The canonical ladder for the default l=8 layout (15 real slots: DC + 7*(real,imag)) is

        [8, 6, 4, 4, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]   -> prod = 1536  (in the 1024-2048 target)

    Radix 1 keeps the codebook in the ~1024-2048 target instead of letting a long tail of radix-2
    slots blow it up (15 radix-2 slots would be 2^15). Returns the radix tuple of length
    n_real_slots; for shorter/longer layouts the template is truncated / radix-1-padded.
    """
    template = [8, 6, 4, 4, 2]
    if n_real_slots <= len(template):
        return tuple(int(r) for r in template[:n_real_slots])
    tail = [1] * (n_real_slots - len(template))
    return tuple(int(r) for r in (template + tail))


# --------------------------------------------------------------------------- #
# coefficient layout (which rfft real/imag parts we keep, and in what order)
# --------------------------------------------------------------------------- #
def build_coef_layout(n_coef: int, win_len: int):
    """Decide which rfft real/imag parts form the per-window SFA coefficient vector.

    rfft of a length-L real signal gives L//2+1 complex coefficients C[0..L//2]:
        C[0]    (DC)      : imag == 0 by construction.
        C[L//2] (Nyquist) : imag == 0 IFF L is even (then C[L//2] is real).
    We keep the first `n_coef` coefficients (capped at the available L//2+1). For each kept
    coefficient k we emit its REAL part always, and its IMAG part EXCEPT where imag is structurally
    zero (k==0 DC; and k==L//2 Nyquist when L is even). Keeping a structurally-zero imag slot would
    waste a radix on a constant and could even break A0's bin assignment, so we drop them.

    Returns a list of (coef_index:int, part:str in {'re','im'}) in packing order, e.g. for n_coef=8,
    L=64 (even, L//2=32, Nyquist not among first 8): [(0,'re'),(1,'re'),(1,'im'),...,(7,'re'),(7,'im')].
    DC ('net-drift') is the FIRST slot — see the module DC-KEEP DECISION.
    """
    n_avail = win_len // 2 + 1                  # number of rfft coefficients for a length-L signal
    keep = int(min(int(n_coef), n_avail))
    nyq_index = win_len // 2 if (win_len % 2 == 0) else None
    layout = []
    for k in range(keep):
        layout.append((k, "re"))
        imag_is_zero = (k == 0) or (nyq_index is not None and k == nyq_index)
        if not imag_is_zero:
            layout.append((k, "im"))
    return layout


def coef_vector(mv_window: np.ndarray, layout) -> np.ndarray:
    """The per-window SFA coefficient vector: rfft(mv_window) projected onto the kept real/imag
    parts in `layout` order. NO per-window z-normalization (the #1 SFA trap) — the mv channel is
    already vol-unit/scale-free, so the raw coefficients ARE the scale-free spectral description.

    mv_window : (W,) float64/float32 vol-unit move signal for ONE window.
    Returns (n_slots,) float64 coefficient vector (n_slots == len(layout)).
    """
    x = np.asarray(mv_window, dtype=np.float64)
    spec = rfft(x)                              # (W//2+1,) complex; LINEAR in x -> A0 holds
    out = np.empty(len(layout), dtype=np.float64)
    for i, (k, part) in enumerate(layout):
        out[i] = float(spec[k].real) if part == "re" else float(spec[k].imag)
    return out


# --------------------------------------------------------------------------- #
# the FROZEN MCB quantizer (per-coefficient quantile bin edges + per-bin centroids)
# --------------------------------------------------------------------------- #
class SFAVocab:
    """Frozen Symbolic-Fourier-Approximation vocabulary.

    Holds, per coefficient slot i (in build_coef_layout order):
        radixes[i]   : number of equi-depth bins r_i for that slot.
        edges[i]     : (r_i - 1,) sorted interior quantile bin edges (np.digitize boundaries).
        centroids[i] : (r_i,) per-bin MEAN coefficient value on BUILD (for A2 dequantization).
    Token id = mixed-radix pack of the per-slot bin indices using the fsq_core basis
        id = sum_i b_i * prod_{j<i} r_j ,  basis = [1, r_0, r_0*r_1, ...].
    codebook_size = prod(radixes).

    Fitting uses pyts.approximation.MultipleCoefficientBinning when it is importable AND the layout
    is uniform-radix (MCB forces ONE n_bins across all positions); since L7 deliberately uses a
    NON-uniform radix ladder (more bins low-freq), the numpy equi-depth path is the canonical fit.
    `fit_backend` records which path was used.
    """

    def __init__(self, layout, radixes, edges, centroids, fit_backend, win_len, n_coef):
        self.layout = list(layout)
        self.radixes = tuple(int(r) for r in radixes)
        self.edges = [np.asarray(e, dtype=np.float64) for e in edges]
        self.centroids = [np.asarray(cc, dtype=np.float64) for cc in centroids]
        self.fit_backend = str(fit_backend)
        self.win_len = int(win_len)
        self.n_coef = int(n_coef)
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
        cv = coef_vector(mv_window, self.layout)
        return self.pack(self.bin_indices(cv))

    # --- decode (A2 reconstruction) ---
    def dequantize(self, bin_idx: np.ndarray) -> np.ndarray:
        """Per-slot bin index -> per-bin CENTROID coefficient value (BUILD per-bin mean)."""
        out = np.empty(len(self.layout), dtype=np.float64)
        for i in range(len(self.layout)):
            out[i] = self.centroids[i][int(bin_idx[i])]
        return out

    def reconstruct_window(self, mv_window: np.ndarray) -> np.ndarray:
        """Reconstruct the mv window from its quantized low-freq spectrum (centroid dequant ->
        Hermitian spectrum with zeros for dropped high freqs -> irfft). Returns (W,) float64."""
        cv = coef_vector(mv_window, self.layout)
        idx = self.bin_indices(cv)
        deq = self.dequantize(idx)
        n_freq = self.win_len // 2 + 1
        spec = np.zeros(n_freq, dtype=np.complex128)
        for val, (k, part) in zip(deq, self.layout):
            if part == "re":
                spec[k] += val
            else:
                spec[k] += 1j * val
        return irfft(spec, n=self.win_len)


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


def _fit_edges_pyts(coef_matrix: np.ndarray, n_bins: int):
    """pyts MultipleCoefficientBinning edges (uniform n_bins across slots). Only usable when the
    radix ladder is UNIFORM (MCB forces one n_bins). Returns the per-slot interior edge list, or
    None if pyts is unavailable. Centroids are still computed in numpy (MCB exposes only edges)."""
    try:
        from pyts.approximation import MultipleCoefficientBinning
    except Exception:
        return None
    mcb = MultipleCoefficientBinning(n_bins=int(n_bins), strategy="quantile")
    mcb.fit(coef_matrix.astype(np.float64))
    be = np.asarray(mcb.bin_edges_, dtype=np.float64)           # (n_slots, n_bins-1)
    return [be[i] for i in range(be.shape[0])]


def fit_vocab(coef_matrix: np.ndarray, layout, radixes, *, win_len, n_coef, prefer_pyts=True):
    """Fit the frozen SFAVocab from pooled BUILD window coefficients.

    Uses pyts MCB ONLY when the radix ladder is uniform AND pyts imports (records 'pyts'); else the
    numpy equi-depth path (records 'numpy'). Centroids are always computed in numpy (per-bin BUILD
    mean) for A2 dequantization."""
    radixes = tuple(int(r) for r in radixes)
    uniform = len(set(radixes)) == 1
    fit_backend = "numpy"
    edges = centroids = None
    if prefer_pyts and uniform:
        e = _fit_edges_pyts(coef_matrix, radixes[0])
        if e is not None:
            edges = e
            # centroids from build rows per pyts edges
            cents = []
            for i, ed in enumerate(edges):
                col = coef_matrix[:, i].astype(np.float64)
                bi = np.clip(np.digitize(col, np.asarray(ed)), 0, radixes[i] - 1)
                cm = float(col.mean()) if col.size else 0.0
                cc = np.full(radixes[i], cm, dtype=np.float64)
                for b in range(radixes[i]):
                    msk = bi == b
                    if np.any(msk):
                        cc[b] = float(col[msk].mean())
                cents.append(cc)
            centroids = cents
            fit_backend = "pyts"
    if edges is None:
        edges, centroids = _fit_edges_centroids_numpy(coef_matrix, radixes)
    return SFAVocab(layout, radixes, edges, centroids, fit_backend, win_len, n_coef)


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
    structure. The A8/control substrate transform mirrored from L5's control path."""
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
def fit_vocab_from_assets(db, assets, cfg, *, n_coef, win_len, stride, max_windows,
                          reservoir_cap=DEFAULT_FIT_RESERVOIR, seed=0, prefer_pyts=True):
    """Stream BUILD-asset windows, pool their SFA coefficient vectors into a bounded reservoir, and
    fit the frozen SFAVocab. RAM-capped at reservoir_cap coefficient rows (reservoir sampling), so
    the fit never materializes all windows. Returns the fitted SFAVocab."""
    layout = build_coef_layout(n_coef, win_len)
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
            cv = coef_vector(mv_w, layout)
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
        raise RuntimeError("L7 MCB fit: no finite windows found on build assets.")
    coef_matrix = res[:held]
    return fit_vocab(coef_matrix, layout, radixes, win_len=win_len, n_coef=n_coef,
                     prefer_pyts=prefer_pyts)


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
                cv = coef_vector(mv_w, vocab.layout)
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
    """A0 closure: tokenise ONE raw-OHLC window to its L7 product-id via the full vol-unit -> rfft
    -> MCB-bin -> pack path. Rescale invariance is inherited from volunit (the mv channel is
    byte-identical under c->lam*c) + rfft linearity + frozen corpus-fit bins + NO per-window
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
    applies a structure-destroying transform to the mv substrate BEFORE the rfft, mirroring L5:
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
    """A8 wiring for L7: the 'vocab' is the frozen SFAVocab; encode_assets_fn streams product-ids
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
    """Assemble the A0/A1/A2/A8/A9 battery for L7 from the online accumulators + frozen vocab."""
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

    # --- A2: spectral reconstruction fidelity (additive, signed vol-unit mv) ---
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

    # --- A9: length non-degeneracy. L7 tokens are FIXED-length (one token / window) by construction,
    # so the raw length distribution is a single spike at `window`. Mirror L4/L5: use the shared
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
    """L7 verdict: PROCEED requires (i) A0 rescale-invariance (the hard gate), (ii) A9 non-degeneracy
    (fixed-length + healthy used-codes), AND (iii) A8 cross-asset transfer passing the §186-aligned
    go/no-go (EB.a8_verdict: random-codebook null on unigram AND bigram). A2 is reported, not gated
    (reconstruction fidelity is informative but a lossy low-freq spectral code is expected to be
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
        description="L7 SFA (Symbolic Fourier Approximation) candle-language member (CPU)."
    )
    p.add_argument("--db", type=str, default=None,
                   help="DB path (default: config.db_path() / $CANDLE_DB).")
    p.add_argument("--out", type=str,
                   default="/home/lukas/Trader/runs/789_token_c0/L7_RESULT.md",
                   help="RESULT.md path (a .json sidecar is written alongside).")
    p.add_argument("--max-windows-per-asset", type=int, default=DEFAULT_MAX_WINDOWS,
                   help="Cap windows kept per asset (most-recent slice via early break). 0 -> unbounded.")
    p.add_argument("--n-coef", type=int, default=DEFAULT_N_COEF,
                   help="l = number of LOW-frequency rfft coefficients kept.")
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
    p.add_argument("--no-pyts", action="store_true",
                   help="Force the numpy quantile-bin path even when pyts is importable.")
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
    n_coef = int(args.n_coef)

    if not DS.quick_check(db):
        print(f"[L7] WARNING: PRAGMA quick_check did not return 'ok' for {db}")

    print(f"[L7] db={db}")
    print(f"[L7] build_assets={cfg.build_assets}  holdout_assets={cfg.holdout_assets}")
    print(f"[L7] window={win_len}  n_coef={n_coef}  stride={stride}  "
          f"max_windows/asset={max_windows}")

    # 1) fit the frozen SFA vocab (MCB bins) on a bounded BUILD-asset coefficient reservoir.
    print("[L7] fitting frozen SFA vocab (MCB quantile bins on build-asset coefficients) ...")
    vocab = fit_vocab_from_assets(
        db, cfg.build_assets, cfg,
        n_coef=n_coef, win_len=win_len, stride=stride, max_windows=max_windows,
        seed=int(args.seed), prefer_pyts=not args.no_pyts,
    )
    print(f"[L7] coef layout ({len(vocab.layout)} real slots): {vocab.layout}")
    print(f"[L7] radixes: {vocab.radixes}  codebook_size={vocab.codebook_size}  "
          f"fit_backend={vocab.fit_backend}")

    # 2) encode BUILD assets -> one product-id token per window (online counts + A2 reconstruction).
    print("[L7] encoding build assets (one token/window) + folding A2 reconstruction ...")
    accum = _OnlineAccum(vocab.codebook_size)
    encode_assets(db, cfg.build_assets, cfg, vocab, accum,
                  win_len=win_len, stride=stride, max_windows=max_windows,
                  a2_reconstruct=True, abs_tol=0.25)
    used = int(np.count_nonzero(accum.token_counts))
    print(f"[L7] encoded {accum.n_windows} windows; used_codes={used}/{vocab.codebook_size} "
          f"({100.0 * used / max(vocab.codebook_size, 1):.1f}%)")

    # 3) eval battery.
    print("[L7] running eval battery (A0/A1/A2/A8/A9) ...")
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
    RW.write_result(args.out, title="L7 — SFA (Symbolic Fourier Approximation) candle language",
                    sections=sections, verdict=verdict, meta=meta)
    print(f"[L7] verdict: {verdict}")
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
        f"- n_coef (l, low-freq kept): {vocab.n_coef}",
        f"- coef real slots: {len(vocab.layout)}  (DC kept as net-drift slot; structural-zero imag dropped)",
        f"- radixes (mixed-radix ladder, more bins low-freq): {list(vocab.radixes)}",
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
        "member": "L7_sfa",
        "window": int(vocab.win_len),
        "n_coef": int(vocab.n_coef),
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
    main()
