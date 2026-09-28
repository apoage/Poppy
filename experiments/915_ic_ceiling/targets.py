#!/usr/bin/env python3
"""TASK_915 targets (memo Sec.4.2). Four named targets, each a function of the causal panel
at date t (uses data <= t) mapped to the FORWARD h-day outcome (data starting strictly after
t) -- leak-free by construction (the label reads r_{t->t+h}; every feature/vol/beta reads
<= t only, checked in tests/test_targets.py).

T1 PRIMARY -- market-residual, Yang-Zhang-standardised forward return:
    y1_{i,t,h} = (r_{i,t->t+h} - beta_hat_{i,t} * r_{mkt,t->t+h}) / sigma_hat_YZ_{i,t}
  beta_hat from a trailing 90-day causal regression of asset log-returns on the equal-weight
  market log-return; sigma_hat_YZ the Yang-Zhang range estimator on the trailing 30 true daily
  OHLC bars (yang_zhang.yang_zhang_vol). Market-neutral by construction (BEAR-positive in
  expectation if any edge exists).

T2 CONTROL (A-2 test) -- same market-residual construction, standardised by CLOSE-TO-CLOSE
  sigma instead of Yang-Zhang:
    y2_{i,t,h} = (r_{i,t->t+h} - beta_hat_{i,t} * r_{mkt,t->t+h}) / sigma_hat_c2c_{i,t}
  Tests whether the range-based (OHLC-native) vol estimator earns its keep over a close-only
  one -- the memo's A-2 claim, made testable rather than assumed.

T3 CONTROL -- absolute forward return (unstandardised, NOT residualised):
    y3_{i,t,h} = | r_{i,t->t+h} |
  A magnitude (volatility-direction) control: if features "predict" y3 but not y1/y2, the
  apparent edge is a vol-clustering effect, not directional alpha.

T4 "sign x vol-scaled" arm (Barroso-Santa-Clara / Daniel-Moskowitz vol-managed momentum,
  memo Sec.4.6) -- inverse-variance-scaled RAW forward return (not residualised: this arm
  tests the vol-scaling effect on outright returns, the crypto vol-managed-momentum
  replication the memo cites):
    y4_{i,t,h} = r_{i,t->t+h} / sigma_hat_YZ_{i,t}^2

A separate SINGLE-SERIES market-factor-timing arm (predicting r_mkt itself, breadth ~= 1) is
reported but never pooled into the 160-cell grid (memo Sec.4.2) -- see `market_timing_target`.
"""
import numpy as np

import yang_zhang as yz


def log_close(close):
    return np.log(np.where(close > 0, close, np.nan))


def forward_log_return(close, h):
    """r_{t->t+h} = log(close_{t+h}) - log(close_t); last h rows are NaN (no forward data)."""
    lc = log_close(close)
    T = lc.shape[0]
    out = np.full_like(lc, np.nan)
    out[: T - h] = lc[h:] - lc[:-h]
    return out


def equal_weight_market_return(close, present_mask):
    """(T,) equal-weight market log-return: cross-sectional mean of 1-day close-to-close
    log-returns over PRESENT assets only (present at BOTH t-1 and t)."""
    lc = log_close(close)
    r1 = np.vstack([np.full((1, close.shape[1]), np.nan), lc[1:] - lc[:-1]])
    pres_pair = present_mask & np.vstack([np.zeros((1, close.shape[1]), dtype=bool),
                                           present_mask[:-1]])
    r1m = np.where(pres_pair, r1, np.nan)
    with np.errstate(invalid="ignore"):
        mkt = np.nanmean(r1m, axis=1)
    n_live = pres_pair.sum(axis=1)
    mkt[n_live < 2] = np.nan
    return mkt


def rolling_beta(asset_ret_1d, mkt_ret_1d, window=90):
    """Causal trailing-window OLS beta of `asset_ret_1d` (T,A) on `mkt_ret_1d` (T,), ending at
    each t (t itself included). NaN until a full window of jointly-finite pairs exists."""
    T, A = asset_ret_1d.shape
    out = np.full((T, A), np.nan)
    mkt = mkt_ret_1d
    for t in range(window - 1, T):
        seg_r = asset_ret_1d[t - window + 1: t + 1]
        seg_m = mkt[t - window + 1: t + 1]
        ok_m = np.isfinite(seg_m)
        if ok_m.sum() < window:
            continue
        m = seg_m[ok_m]
        mc = m - m.mean()
        var_m = float(mc @ mc)
        if var_m <= 0:
            continue
        seg_ok = seg_r[ok_m]                      # (window, A)
        finite_cols = np.isfinite(seg_ok).all(axis=0)
        rc = seg_ok[:, finite_cols] - seg_ok[:, finite_cols].mean(axis=0, keepdims=True)
        cov = mc @ rc / window
        out[t, finite_cols] = cov / (var_m / window)
    return out


def one_day_log_returns(close):
    lc = log_close(close)
    return np.vstack([np.full((1, close.shape[1]), np.nan), lc[1:] - lc[:-1]])


def build_all_targets(panel, present_mask, h, beta_window=90, vol_window=30):
    """panel: dict with open/high/low/close (T,A). Returns dict of the four (T,A) target
    arrays plus the intermediates (beta, sigma_yz, sigma_c2c, mkt_ret) needed by G-B/features."""
    close = panel["close"]
    r_fwd = forward_log_return(close, h)
    r1d = one_day_log_returns(close)
    mkt1d = equal_weight_market_return(close, present_mask)
    r_mkt_fwd = np.full(close.shape[0], np.nan)
    T = close.shape[0]
    mkt_cum = np.concatenate([[0.0], np.nancumsum(np.where(np.isfinite(mkt1d), mkt1d, 0.0))])
    # forward market return over the same non-overlapping h-day window as the asset return
    r_mkt_fwd[: T - h] = mkt_cum[h + 1: T + 1] - mkt_cum[1: T - h + 1]

    beta = rolling_beta(r1d, mkt1d, window=beta_window)
    sigma_yz = yz.yang_zhang_vol(panel["open"], panel["high"], panel["low"], close,
                                  window=vol_window)
    sigma_c2c = yz.close_to_close_vol(close, window=vol_window)

    resid_fwd = r_fwd - beta * r_mkt_fwd[:, None]
    y1 = resid_fwd / np.maximum(sigma_yz, 1e-10)
    y2 = resid_fwd / np.maximum(sigma_c2c, 1e-10)
    y3 = np.abs(r_fwd)
    y4 = r_fwd / np.maximum(sigma_yz ** 2, 1e-10)

    return {
        "y1_primary_resid_yz": y1, "y2_control_resid_c2c": y2,
        "y3_control_abs_return": y3, "y4_sign_volscaled": y4,
        "beta": beta, "sigma_yz": sigma_yz, "sigma_c2c": sigma_c2c,
        "mkt_ret_1d": mkt1d, "r_mkt_fwd": r_mkt_fwd, "r_fwd": r_fwd,
        "resid_fwd": resid_fwd,   # market-residual forward return (numerator of y1/y2) -- the
                                   # correlation spectrum G-B's N_eff is measured on.
    }


def market_timing_target(mkt_ret_1d, h):
    """Single-series market-factor-timing arm: forward h-day market return itself (breadth=1,
    reported separately, never pooled into the 160-cell grid)."""
    T = mkt_ret_1d.size
    cum = np.concatenate([[0.0], np.nancumsum(np.where(np.isfinite(mkt_ret_1d), mkt_ret_1d, 0.0))])
    out = np.full(T, np.nan)
    out[: T - h] = cum[h + 1: T + 1] - cum[1: T - h + 1]
    return out


TARGET_NAMES = ("y1_primary_resid_yz", "y2_control_resid_c2c", "y3_control_abs_return",
                "y4_sign_volscaled")
