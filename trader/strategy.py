#!/usr/bin/env python3
"""TASK_820 strategy layer — faithful port of the audited runs/819_single_asset/portfolio.py
trend-following construction (leak-free): trend_signal, rolling_vol, first_active,
present_mask, weights, build_positions. Plus the continuous tanh variant (available,
non-default).

BACKEND: every kernel is written against a swappable array module `xp` (config.get_backend()
-> numpy by default, jax.numpy via --backend jax). The functional (scatter-free) rewrites are
numerically identical to 819's numpy versions on float64 — proven by trader parity/selftest,
which force the numpy backend. The originals used numpy in-place assignment (`rv[t]=...`,
`acc[m]+=...`); those are replaced here by concatenate/stack/where so the SAME arithmetic runs
under jax without lax control flow.

Leak-free contract (unchanged from 819):
  - signal s_t = sign(logC_t - logC_{t-L})  uses close <= t only
  - trailing vol realized INTO row t uses returns <= t only
  - the forward label (close(t+H)) is applied later in backtest.regime_series, never here.
"""

import numpy as np

from .config import VOL_FLOOR, VOL_WINDOW, get_backend


def _xp(xp):
    return get_backend() if xp is None else xp


# =============================================================================
# leak-free construction — port of 819 portfolio.py fns (scatter-free / xp-generic)
# =============================================================================

def rolling_vol(logc, window, xp=None):
    """Trailing std of per-bar log returns over the last `window` returns realized INTO row t
    (rows <= t only -> leak-free). Returns (T, A); NaN for t < window.

    819 computed rows t>=window via cumulative sums and scattered into `rv[t]`. Here the same
    cumulative-sum arithmetic is evaluated for all t and the t<window rows are masked to NaN
    (identical values for t>=window; no scatter -> xp-generic)."""
    xp = _xp(xp)
    T, A = logc.shape
    r = logc[1:] - logc[:-1]                        # r[k] = logc[k+1]-logc[k]; r[t-1] known at t
    fin = xp.isfinite(r)
    r0 = xp.where(fin, r, 0.0)
    zrow = xp.zeros((1, A), dtype=logc.dtype)
    c1 = xp.concatenate([zrow, xp.cumsum(r0, axis=0)], axis=0)          # (T, A)
    c2 = xp.concatenate([zrow, xp.cumsum(r0 * r0, axis=0)], axis=0)
    cc = xp.concatenate([zrow, xp.cumsum(fin.astype(logc.dtype), axis=0)], axis=0)
    tt = xp.arange(T)
    lo = tt - window
    loc = xp.where(lo < 0, 0, lo)                    # t<window rows masked below; index safely
    s1 = c1 - c1[loc]                               # row t: c1[t]-c1[t-window] for t>=window
    s2 = c2 - c2[loc]
    n = xp.maximum(cc - cc[loc], 1.0)
    var = xp.maximum(s2 / n - (s1 / n) ** 2, 0.0)
    rv = xp.where((tt >= window)[:, None], xp.sqrt(var), xp.nan)
    return rv


def trend_signal(logc, lookbacks, xp=None):
    """Mean-sign trend over the lookback set: s_t = mean_L sign(logc_t - logc_{t-L}) in [-1,1].
    Single lookback -> plain sign in {-1,0,+1}. Uses close <= t only (leak-free).

    Reproduces 819's acc/cnt masked-mean exactly (rows where no lookback is warm -> NaN)."""
    xp = _xp(xp)
    T, A = logc.shape
    parts = []
    for L in lookbacks:
        warm = xp.full((L, A), xp.nan, dtype=logc.dtype)
        body = xp.sign(logc[L:] - logc[:-L])
        parts.append(xp.concatenate([warm, body], axis=0))
    stack = xp.stack(parts, axis=0)                 # (nL, T, A), NaN in each lookback's warmup
    fin = xp.isfinite(stack)
    cnt = fin.sum(axis=0)
    acc = xp.where(fin, stack, 0.0).sum(axis=0)
    return xp.where(cnt > 0, acc / xp.maximum(cnt, 1.0), xp.nan)


def trend_signal_tanh(logc, lookbacks, window, xp=None):
    """Continuous CTA variant (AVAILABLE, non-default): s_t = mean_L tanh(zscore_L),
    zscore_L = (logc_t-logc_{t-L}) / (trailing-bar-vol * sqrt(L)). Bounded [-1,1]. Leak-free."""
    xp = _xp(xp)
    T, A = logc.shape
    rv = rolling_vol(logc, window, xp=xp)
    parts = []
    for L in lookbacks:
        warm = xp.full((L, A), xp.nan, dtype=logc.dtype)
        mom = logc[L:] - logc[:-L]
        denom = rv[L:] * np.sqrt(L)
        z = xp.where(denom > VOL_FLOOR, mom / xp.maximum(denom, VOL_FLOOR), 0.0)
        parts.append(xp.concatenate([warm, xp.tanh(z)], axis=0))
    stack = xp.stack(parts, axis=0)
    fin = xp.isfinite(stack)
    cnt = fin.sum(axis=0)
    acc = xp.where(fin, stack, 0.0).sum(axis=0)
    return xp.where(cnt > 0, acc / xp.maximum(cnt, 1.0), xp.nan)


def first_active(logc, xp=None):
    """First row index per asset with a real (nonzero) return — its 'listing' row. Rows before
    it are back-filled constant and excluded from the present mask. Returns (A,) int; T if the
    asset never moves. 819 looped per column; here it's a vectorized argmin-of-first-True."""
    xp = _xp(xp)
    T, A = logc.shape
    r = logc[1:] - logc[:-1]
    nz = xp.abs(r) > 1e-12                           # (T-1, A)
    rows = xp.arange(T - 1)[:, None]
    cand = xp.where(nz, rows, T)                     # T where not nonzero
    return xp.min(cand, axis=0)                      # first nonzero r-row, else T


def present_mask(logc, rv, fa, max_lb, window, xp=None):
    """present[t,i]: asset i tradeable at decision t iff it has >= max(max_lb, window) bars of
    active history AND trailing vol > floor (drops pre-listing back-fill and dead/delisted
    constant segments, e.g. LUNA after 2022)."""
    xp = _xp(xp)
    T, A = logc.shape
    age = xp.arange(T)[:, None] - fa[None, :]
    warm = max(max_lb, window)
    return (age >= warm) & xp.isfinite(rv) & (rv > VOL_FLOOR)


def weights(pres, rv, sizing, xp=None):
    """Risk-budget weights summing to 1 over present assets each row.
    equal: 1/n_present. invvol (risk parity): (1/rv)/sum(1/rv)."""
    xp = _xp(xp)
    if sizing == "equal":
        raw = xp.where(pres, 1.0, 0.0)
    elif sizing == "invvol":
        raw = xp.where(pres, 1.0 / xp.maximum(rv, VOL_FLOOR), 0.0)
    else:
        raise ValueError(sizing)
    denom = raw.sum(axis=1, keepdims=True)
    return xp.where(denom > 0, raw / xp.maximum(denom, 1e-300), 0.0)


def build_positions(logc, lookbacks, sizing, window=VOL_WINDOW, signal="sign", xp=None):
    """positions pos[t,i] = w[t,i] * s[t,i]. w = risk budget (sums to 1 over present);
    s in [-1,1] = trend direction/conviction; gross exposure Sum|pos| <= 1 (a $1 long/short
    trend book). Returns (pos, pres, w, warm). Faithful to 819 build_positions."""
    xp = _xp(xp)
    max_lb = max(lookbacks)
    rv = rolling_vol(logc, window, xp=xp)
    fa = first_active(logc, xp=xp)
    if signal == "tanh":
        sig = trend_signal_tanh(logc, lookbacks, window, xp=xp)
    else:
        sig = trend_signal(logc, lookbacks, xp=xp)
    pres = present_mask(logc, rv, fa, max_lb, window, xp=xp)
    sig = xp.where(pres, sig, 0.0)
    w = weights(pres, rv, sizing, xp=xp)
    pos = w * sig
    warm = max(max_lb, window) + 1
    return pos, pres, w, warm


def positions_from_config(logc, scfg, xp=None):
    """Convenience: build positions for a StrategyConfig. Returns (pos, pres, w, warm)."""
    return build_positions(
        logc, list(scfg.lookbacks), scfg.sizing,
        window=scfg.vol_window, signal=scfg.signal, xp=xp,
    )
