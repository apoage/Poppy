"""
Vol-unit substrate (design 01 §2.1) — the per-bar feature kernel shared by EVERY panel member.

Six causal, scale-free channels per bar t (t>=1), aligned so feats[k] describes the bar at ts[k+1]:
    mv, gap, body, range  = log-ratios divided by a causal local-vol estimate sigma_t
    uw, lw                = wick fractions in [0,1] (ratios of price differences; no vol norm)

sigma_t = causal EWMA(|log(c/c_{t-1})|, halflife=H), floored, RE-ANCHORED at every UTC-day boundary
(design §2.5: 24/7 corpus -> fixed 1440-min cadence, not "per session").

RESCALE INVARIANCE (eval A0, the load-bearing property): under c->lam*c (and o/h/l likewise), every
log-ratio is unchanged, sigma_t is unchanged (it is built from log-ratios), and uw/lw are ratios of
price differences -> unchanged. So the 6-vector is byte-identical under rescale, for any lam where the
wick guard (WICK_EPS) does not bite. That invariance is what makes "the same shape -> the same token"
hold across assets at $0.007 vs $105,747.

Canonical path = the numpy builder (`features_for_run`), which produces the SHIPPED features. The JAX
`volunit_window_jax` mirror is a window-local (gap-free, no day re-anchor) recompute for optional
on-device eval; it is NOT used to produce shipped tokens.
"""
import numpy as np
from scipy.signal import lfilter

from . import config as C  # noqa: E402  (package-relative)


# --------------------------------------------------------------------------- #
# numpy builder (CPU worker — the canonical path)
# --------------------------------------------------------------------------- #
def _ewma_causal(x, day, alpha):
    """Causal EWMA of x with the state RESET at every change in `day` (UTC-day re-anchor).
    Within each day-segment: s[0]=x[0] (fresh seed), s[k]=alpha*x[k]+(1-alpha)*s[k-1].
    Implemented per-segment with scipy.lfilter (C-fast) so streaming millions of bars is cheap."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    if x.size == 0:
        return out
    bnd = np.flatnonzero(np.diff(day)) + 1            # indices where the day changes
    starts = np.concatenate([[0], bnd])
    ends = np.concatenate([bnd, [x.size]])
    b = [alpha]
    a = [1.0, -(1.0 - alpha)]
    for s, e in zip(starts, ends):
        seg = x[s:e]
        if seg.size == 0:
            continue
        zi = [(1.0 - alpha) * seg[0]]                 # makes y[0] == seg[0] (seed = first value)
        out[s:e], _ = lfilter(b, a, seg, zi=zi)
    return out


def features_for_run(ts, o, h, l, c, cfg: "C.CandleConfig"):
    """Build the 6-channel vol-unit features for ONE dt==60 contiguous run (caller guarantees
    contiguity + bar validity; NO forward-fill, NO gap straddle).

    Returns (all length L-1, where L = len(ts)):
        feats  : (L-1, N_FEAT) float32   channels in config.CHANNELS order
        close  : (L-1,)        float32   raw close at each feature bar (= c[1:]) — for A2 denorm
        ts_out : (L-1,)        int64     timestamp at each feature bar (= ts[1:])
        finite : (L-1,)        bool      all-finite guard (drop any non-finite feature bar)
    """
    ts = np.asarray(ts, dtype=np.int64)
    o = np.asarray(o, dtype=np.float64); h = np.asarray(h, dtype=np.float64)
    l = np.asarray(l, dtype=np.float64); c = np.asarray(c, dtype=np.float64)
    prev_c = c[:-1]
    cur_o, cur_h, cur_l, cur_c = o[1:], h[1:], l[1:], c[1:]

    logret = np.log(np.clip(cur_c, C.EPS, None) / np.clip(prev_c, C.EPS, None))
    day = ts[1:] // 86400
    alpha = 1.0 - 2.0 ** (-1.0 / float(cfg.ewma_halflife))
    sigma = _ewma_causal(np.abs(logret), day, alpha)
    sigma = np.maximum(sigma, cfg.sigma_floor)
    den = np.maximum(sigma, C.DIV_EPS)

    mv = logret / den
    gap = np.log(np.clip(cur_o, C.EPS, None) / np.clip(prev_c, C.EPS, None)) / den
    body = np.log(np.clip(cur_c, C.EPS, None) / np.clip(cur_o, C.EPS, None)) / den
    rng = np.log(np.clip(cur_h, C.EPS, None) / np.clip(cur_l, C.EPS, None)) / den
    hl = np.maximum(cur_h - cur_l, C.WICK_EPS)
    uw = (cur_h - np.maximum(cur_o, cur_c)) / hl
    lw = (np.minimum(cur_o, cur_c) - cur_l) / hl

    feats = np.stack([mv, gap, body, rng, uw, lw], axis=1).astype(np.float32)
    finite = np.all(np.isfinite(feats), axis=1)
    return feats, cur_c.astype(np.float32), ts[1:], finite


# --------------------------------------------------------------------------- #
# JAX vmap mirror (window-local recompute for on-device eval; gap-free, no day re-anchor)
# --------------------------------------------------------------------------- #
def volunit_window_jax(o, h, l, c, halflife, sigma_floor):
    """Window-local mirror of the substrate for on-device recompute. o/h/l/c: (W,) for ONE
    gap-free window. Returns (W-1, N_FEAT). NO day re-anchor (a window is << 1 day); use only
    for the optional on-device A0 cross-check, never to mint shipped tokens. JAX pitfalls honored
    (jnp.maximum guards; lax.scan forward-only -> no backward, no checkpoint needed)."""
    import jax
    import jax.numpy as jnp
    logret = jnp.log(jnp.maximum(c[1:], C.EPS) / jnp.maximum(c[:-1], C.EPS))
    absr = jnp.abs(logret)
    alpha = 1.0 - 2.0 ** (-1.0 / float(halflife))

    def step(s_prev, x):
        s = alpha * x + (1.0 - alpha) * s_prev
        return s, s

    _, sigma_rest = jax.lax.scan(step, absr[0], absr[1:])
    sigma = jnp.concatenate([absr[:1], sigma_rest])
    sigma = jnp.maximum(sigma, sigma_floor)
    den = jnp.maximum(sigma, C.DIV_EPS)

    mv = logret / den
    gap = jnp.log(jnp.maximum(o[1:], C.EPS) / jnp.maximum(c[:-1], C.EPS)) / den
    body = jnp.log(jnp.maximum(c[1:], C.EPS) / jnp.maximum(o[1:], C.EPS)) / den
    rng = jnp.log(jnp.maximum(h[1:], C.EPS) / jnp.maximum(l[1:], C.EPS)) / den
    hl = jnp.maximum(h[1:] - l[1:], C.WICK_EPS)
    uw = (h[1:] - jnp.maximum(o[1:], c[1:])) / hl
    lw = (jnp.minimum(o[1:], c[1:]) - l[1:]) / hl
    return jnp.stack([mv, gap, body, rng, uw, lw], axis=1)


if __name__ == "__main__":
    # A0 self-test (CPU, no JAX): rescale-invariance must hold byte-identically for reasonable lam.
    rng = np.random.default_rng(0)
    L = 5000
    ts = np.arange(L, dtype=np.int64) * C.BAR_SECONDS + 1_600_000_000   # contiguous dt==60
    steps = rng.normal(0, 0.002, size=L).cumsum()
    close = 100.0 * np.exp(steps)
    op = close * np.exp(rng.normal(0, 0.0005, size=L))
    hi = np.maximum(op, close) * (1 + np.abs(rng.normal(0, 0.0008, size=L)))
    lo = np.minimum(op, close) * (1 - np.abs(rng.normal(0, 0.0008, size=L)))
    cfg = C.CandleConfig()
    f1, c1, t1, ok1 = features_for_run(ts, op, hi, lo, close, cfg)
    for lam in (0.1, 1e-3, 1e3, 7777.0):
        f2, _, _, _ = features_for_run(ts, op * lam, hi * lam, lo * lam, close * lam, cfg)
        max_abs = float(np.max(np.abs(f1 - f2)))
        print(f"lam={lam:>10g}  max|Δfeature|={max_abs:.3e}  (PASS if ~0)")
    print(f"finite fraction: {ok1.mean():.4f}  feats shape {f1.shape}")
