#!/usr/bin/env python3
"""TASK_918 tactic library — K=8 position matrices, built ONLY from unmodified
`trader.strategy` primitives (`rolling_vol`, `first_active`, `trend_signal`, `present_mask`,
`weights`) plus a small amount of composition glue in this file. `trader/` itself is never
edited or monkeypatched.

All 8 arms share ONE present-mask / warmup grid (see `shared_grid`, max_lb=90 -- driven by
the beta-hedge arm's 90d trailing BTC-beta window, the longest lookback in the library) so
the hull's mixed-book turnover netting and every arm's H-strided decision grid are directly
comparable and combinable. This is a deliberate, pre-registered simplification (SPEC.md
"Library construction"): shorter-lookback arms (7d/21d reversal) lose a few weeks of
per-asset eligibility right after listing that their own native lookback would have granted;
it costs nothing mid-window and is disclosed rather than silently varying the grid per arm.

Leak-free contract: every signal at row t uses close[<=t] only; the forward label is applied
downstream by the scorer, never here (same contract as `trader.strategy`).
"""
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from trader import strategy as tstrat   # noqa: E402
from trader.config import VOL_FLOOR     # noqa: E402

SHARED_MAX_LB = 90        # longest lookback in the library (beta-hedge's 90d BTC beta)
SHARED_VOL_WINDOW = 30    # 819 convention; also the vol-target / equal-weight sizing window
BETA_WINDOW = 90
VOL_TARGET_WINDOW = 30
VOL_TARGET_ANNUAL = 0.40  # 40%/yr portfolio vol target (sizing report's T5, tactics_gaps.md:14)

TACTIC_NAMES = [
    "flat", "trend_819", "long_flat_trend", "equal_weight_cm", "beta_hedged_cm",
    "vol_targeted_trend", "reversal_7d_ts", "reversal_21d_xs",
]


def shared_grid(logc, xp=None):
    """The ONE (pres, warm) every arm in the library shares. Returns (rv30, fa, pres, warm)."""
    xp = tstrat._xp(xp)
    rv30 = tstrat.rolling_vol(logc, SHARED_VOL_WINDOW, xp=xp)
    fa = tstrat.first_active(logc, xp=xp)
    pres = tstrat.present_mask(logc, rv30, fa, SHARED_MAX_LB, SHARED_VOL_WINDOW, xp=xp)
    warm = SHARED_MAX_LB + 1
    return rv30, fa, pres, warm


def _signal_positions(logc, pres, rv, sig, sizing="equal", xp=None):
    """Compose an arbitrary signal `sig` (T,A) with the shared present-mask into a position
    matrix, via the SAME primitives `trader.strategy.build_positions` calls internally
    (`weights`) -- the only thing varying arm-to-arm is `sig` and, for sizing, `rv`."""
    xp = tstrat._xp(xp)
    sig = xp.where(pres, sig, 0.0)
    w = tstrat.weights(pres, rv, sizing, xp=xp)
    return w * sig


def build_flat(logc, pres, rv, xp=None):
    """Arm 1: flat. Zero positions everywhere (a cash book) on the shared grid."""
    xp = tstrat._xp(xp)
    return xp.zeros_like(logc)


def build_trend_819(logc, pres, rv, xp=None):
    """Arm 2: the frozen 819/820 rule (single 30d sign trend, equal weight), rebuilt on the
    SHARED present-mask (a strict narrowing of the rule's own native 30d-warm mask -- never an
    inflation, see module docstring)."""
    xp = tstrat._xp(xp)
    sig = tstrat.trend_signal(logc, [30], xp=xp)
    return _signal_positions(logc, pres, rv, sig, "equal", xp=xp)


def build_long_flat_trend(logc, pres, rv, xp=None):
    """Arm 3: TASK_898's long-flat variant (shorts of the 30d trend signal replaced by flat),
    reproduced from the SAME `trader.strategy` primitives 898's `build_long_flat_positions`
    uses (`experiments/898_trend_decay/decay_harness.py:99-120`, attributed; not imported
    across the digit-prefixed package -- this is the identical 3-line recipe)."""
    xp = tstrat._xp(xp)
    sig = tstrat.trend_signal(logc, [30], xp=xp)
    sig = xp.maximum(sig, 0.0)
    return _signal_positions(logc, pres, rv, sig, "equal", xp=xp)


def build_equal_weight_cm(logc, pres, rv, xp=None):
    """Arm 4: equal-weight constant-mix -- always-long, equal weight over present assets,
    daily-rebalanced to the constant target 1/n_present. Same construction as
    `trader.backtest.regime_series`'s internal `bh` column (`wbh = pr/pr.sum()`), but built
    here as a genuine TRADEABLE position matrix (real turnover as membership/weights shift),
    not an internal zero-turnover benchmark column."""
    xp = tstrat._xp(xp)
    sig = xp.ones_like(logc)
    return _signal_positions(logc, pres, rv, sig, "equal", xp=xp)


def rolling_cov_beta(logc, pres, btc_col, window=BETA_WINDOW, xp=None):
    """Trailing-`window`-day OLS beta of every asset's daily log return against BTC's, using
    only returns realized INTO row t (leak-free) -- the same cumulative-sum construction
    `trader.strategy.rolling_vol` uses, generalized to a bivariate covariance. Returns (T,A)
    beta_i,t = Cov(r_i, r_btc)/Var(r_btc) over the trailing window; NaN before `window` returns
    are available."""
    xp = tstrat._xp(xp)
    T, A = logc.shape
    r = logc[1:] - logc[:-1]
    fin = xp.isfinite(r)
    r0 = xp.where(fin, r, 0.0)
    r_btc = r0[:, btc_col:btc_col + 1]                      # (T-1, 1)
    fin_btc = fin[:, btc_col:btc_col + 1]
    zrow = xp.zeros((1, A), dtype=logc.dtype)
    zrow1 = xp.zeros((1, 1), dtype=logc.dtype)

    def _cum(x):
        return xp.concatenate([zrow if x.shape[1] == A else zrow1, xp.cumsum(x, axis=0)], axis=0)

    c_r = _cum(r0)                       # sum r_i
    c_b = xp.concatenate([zrow1, xp.cumsum(r_btc, axis=0)], axis=0)         # sum r_btc
    c_rb = _cum(r0 * r_btc)              # sum r_i*r_btc
    c_bb = xp.concatenate([zrow1, xp.cumsum(r_btc * r_btc, axis=0)], axis=0)  # sum r_btc^2
    c_n = _cum(fin.astype(logc.dtype) * fin_btc.astype(logc.dtype))          # paired-finite count

    tt = xp.arange(T)
    lo = tt - window
    loc = xp.where(lo < 0, 0, lo)
    n = xp.maximum(c_n - c_n[loc], 1.0)
    s_r = c_r - c_r[loc]
    s_b = c_b - c_b[loc]
    s_rb = c_rb - c_rb[loc]
    s_bb = c_bb - c_bb[loc]
    cov = s_rb / n - (s_r / n) * (s_b / n)
    var = xp.maximum(s_bb / n - (s_b / n) ** 2, 1e-18)
    beta = cov / var
    beta = xp.where((tt >= window)[:, None], beta, xp.nan)
    return beta


def build_beta_hedged_cm(logc, pres, rv, syms, xp=None):
    """Arm 5: the equal-weight constant-mix (arm 4) hedged against trailing-90d BTC beta
    (K-tactic explorer's T4, `lore/harvest_6b32_appendix/tactics_gaps.md:14`): the book's
    aggregate trailing beta to BTC/XBT is shorted directly in the BTC column,
    pos_hedge[t,BTC] = pos_ew[t,BTC] - Sum_i w_ew[t,i]*beta_i,t. Gross is NOT re-capped after
    the hedge leg (disclosed -- the hedge can push BTC's column beyond the base book's own
    per-asset weight; the aggregate is still O(1) since portfolio beta is O(1)). If BTC/XBT is
    absent at row t (present_mask False that day), the hedge term is 0 for that row."""
    xp = tstrat._xp(xp)
    pos_ew = build_equal_weight_cm(logc, pres, rv, xp=xp)
    btc_candidates = [j for j, s in enumerate(syms) if s in ("XBT", "BTC")]
    if not btc_candidates:
        return pos_ew, {"btc_present": False, "note": "no BTC/XBT column in this panel"}
    btc_col = btc_candidates[0]
    beta = rolling_cov_beta(logc, pres, btc_col, window=BETA_WINDOW, xp=xp)
    port_beta = xp.sum(xp.where(xp.isfinite(beta), pos_ew * beta, 0.0), axis=1)
    pos = pos_ew.copy() if hasattr(pos_ew, "copy") else np.array(pos_ew, copy=True)
    btc_present = np.asarray(pres)[:, btc_col]
    hedge = np.where(btc_present, port_beta, 0.0)
    pos = np.asarray(pos, dtype=np.float64).copy()
    pos[:, btc_col] = pos[:, btc_col] - hedge
    return pos, {"btc_present": True, "btc_col": int(btc_col), "btc_symbol": syms[btc_col]}


def _trailing_std_1d(x, window):
    """Trailing std of a 1-D series over the last `window` values realized strictly BEFORE
    row t (x[t-window:t], i.e. NOT including x[t] -- the vol-target multiplier at t must be
    computable from information available before t's own return is known). NaN-safe via the
    same masked cumulative-sum trick as `rolling_vol`."""
    x = np.asarray(x, dtype=np.float64)
    T = x.size
    fin = np.isfinite(x)
    x0 = np.where(fin, x, 0.0)
    c1 = np.concatenate([[0.0], np.cumsum(x0)])
    c2 = np.concatenate([[0.0], np.cumsum(x0 * x0)])
    cc = np.concatenate([[0.0], np.cumsum(fin.astype(np.float64))])
    tt = np.arange(T)
    hi = tt                      # exclusive of t itself: window is x[t-window : t]
    lo = np.maximum(tt - window, 0)
    n = np.maximum(cc[hi] - cc[lo], 1.0)
    s1 = c1[hi] - c1[lo]
    s2 = c2[hi] - c2[lo]
    var = np.maximum(s2 / n - (s1 / n) ** 2, 0.0)
    out = np.where(tt >= window, np.sqrt(var), np.nan)
    return out


def build_vol_targeted_trend(logc, pres, rv, base_pos, xp=None):
    """Arm 6: the 819 trend book (arm 2, passed in as `base_pos`) rescaled to a 40%/yr
    PORTFOLIO vol target using the book's OWN trailing-30d realized daily P&L (leak-free: the
    scale applied to pos[t] uses only P&L realized over rows < t), gross-capped at 1 (multiplier
    <= 1, never levers up -- `sizing report's T5 gross cap 1,
    tactics_gaps.md:14; analytic ceiling sqrt(1+CV^2), net Delta in [-0.10,+0.08],
    fable_tactics_panel_2026-09-04.md:36`)."""
    logc_np = np.asarray(logc, dtype=np.float64)
    pos_np = np.asarray(base_pos, dtype=np.float64)
    T = logc_np.shape[0]
    fwd1 = logc_np[1:] - logc_np[:-1]
    fwd1 = np.where(np.isfinite(fwd1), fwd1, 0.0)
    daily_pnl = np.full(T, np.nan)
    # pnl realized INTO row s (s=1..T-1) used position set at s-1 against the s-1->s move.
    daily_pnl[1:] = np.sum(pos_np[:-1] * fwd1, axis=1)
    trailing_vol = _trailing_std_1d(daily_pnl, VOL_TARGET_WINDOW) * np.sqrt(365.0)
    scale = np.where(np.isfinite(trailing_vol) & (trailing_vol > 1e-12),
                     np.minimum(VOL_TARGET_ANNUAL / np.maximum(trailing_vol, 1e-12), 1.0), 1.0)
    return pos_np * scale[:, None], {"scale_mean": float(np.nanmean(scale)),
                                     "scale_min": float(np.nanmin(scale)),
                                     "scale_at_cap_frac": float(np.mean(scale >= 0.999999))}


def build_reversal_7d_ts(logc, pres, rv, xp=None):
    """Arm 7: 7d TIME-SERIES reversal = -sign(logC_t - logC_{t-7}), equal weight over present
    assets (label correction per `tactics_gaps.md:7` -- NOT cross-sectional)."""
    xp = tstrat._xp(xp)
    sig = -tstrat.trend_signal(logc, [7], xp=xp)
    return _signal_positions(logc, pres, rv, sig, "equal", xp=xp)


def build_reversal_21d_xs(logc, pres, rv, xp=None):
    """Arm 8: 21d CROSS-SECTIONAL reversal, tercile long/short. Each day, rank PRESENT assets
    by 21d log return; long the bottom tercile (worst performers) equal-weight summing to
    +0.5 gross, short the top tercile equal-weight summing to -0.5 gross, middle tercile flat.
    Dollar-neutral by construction (gross = 1 when both legs are non-empty; a day with < 3
    present assets, or where one tercile is empty, degrades to whichever leg exists, capped at
    its own 0.5 -- disclosed, not silently re-normalized to gross 1)."""
    logc_np = np.asarray(logc, dtype=np.float64)
    pres_np = np.asarray(pres, dtype=bool)
    T, A = logc_np.shape
    q21 = np.full((T, A), np.nan)
    q21[21:] = logc_np[21:] - logc_np[:-21]
    pos = np.zeros((T, A), dtype=np.float64)
    n_present_hist = np.zeros(T, dtype=np.int64)
    for t in range(T):
        mask = pres_np[t] & np.isfinite(q21[t])
        idx = np.where(mask)[0]
        n = idx.size
        n_present_hist[t] = n
        if n < 3:
            continue
        order = idx[np.argsort(q21[t, idx])]     # ascending 21d return
        k = n // 3
        if k == 0:
            continue
        bottom = order[:k]                        # worst performers -> long (reversal)
        top = order[-k:]                           # best performers -> short
        pos[t, bottom] = 0.5 / k
        pos[t, top] = -0.5 / k
    return pos, {"days_with_lt3_present": int(np.sum(n_present_hist < 3))}


def build_all(logc, syms, xp=None):
    """Build all 8 arms on the shared grid. Returns (positions dict name->(T,A) ndarray,
    pres (T,A) bool, warm int, diag dict of per-arm construction notes)."""
    xp = tstrat._xp(xp)
    rv30, fa, pres, warm = shared_grid(logc, xp=xp)
    diag = {}
    pos = {}
    pos["flat"] = np.asarray(build_flat(logc, pres, rv30, xp=xp), dtype=np.float64)
    pos["trend_819"] = np.asarray(build_trend_819(logc, pres, rv30, xp=xp), dtype=np.float64)
    pos["long_flat_trend"] = np.asarray(build_long_flat_trend(logc, pres, rv30, xp=xp),
                                        dtype=np.float64)
    pos["equal_weight_cm"] = np.asarray(build_equal_weight_cm(logc, pres, rv30, xp=xp),
                                        dtype=np.float64)
    p5, d5 = build_beta_hedged_cm(logc, pres, rv30, syms, xp=xp)
    pos["beta_hedged_cm"] = np.asarray(p5, dtype=np.float64)
    diag["beta_hedged_cm"] = d5
    p6, d6 = build_vol_targeted_trend(logc, pres, rv30, pos["trend_819"], xp=xp)
    pos["vol_targeted_trend"] = np.asarray(p6, dtype=np.float64)
    diag["vol_targeted_trend"] = d6
    pos["reversal_7d_ts"] = np.asarray(build_reversal_7d_ts(logc, pres, rv30, xp=xp),
                                       dtype=np.float64)
    p8, d8 = build_reversal_21d_xs(logc, pres, rv30, xp=xp)
    pos["reversal_21d_xs"] = np.asarray(p8, dtype=np.float64)
    diag["reversal_21d_xs"] = d8
    assert list(pos.keys()) == TACTIC_NAMES, (list(pos.keys()), TACTIC_NAMES)
    return pos, np.asarray(pres, dtype=bool), int(warm), diag
