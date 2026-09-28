#!/usr/bin/env python3
"""TASK_915 — range-based volatility estimators on true daily OHLC (memo Sec.4.2/4.6).

All functions are CAUSAL: the value at row t uses only bars <= t (rows 0..t inclusive of the
trailing `window`). Panels are (T, A) float64 arrays, NaN where an asset is not yet listed /
already delisted (present-mask handles exclusion downstream — these functions do not).

Yang-Zhang (2000): sigma_YZ^2 = sigma_overnight^2 + k*sigma_oc^2 + (1-k)*sigma_RS^2
  overnight_t  = log(open_t / close_{t-1})
  oc_t         = log(close_t / open_t)
  RS_t (Rogers-Satchell, single-day) = log(H/C)*log(H/O) + log(L/C)*log(L/O)
  k = 0.34 / (1.34 + (n+1)/(n-1)), n = trailing window length (Yang-Zhang's bias-optimal k)

Parkinson (1980): sigma_P^2 = (1/(4 ln 2)) * mean( log(H/L)^2 )  [high-low range only]
Garman-Klass (1980): sigma_GK^2 = 0.5*mean(log(H/L)^2) - (2 ln2 - 1)*mean(log(C/O)^2)

Unit tests: tests/test_yang_zhang.py (constant-OHLC -> 0; a hand-built 2-bar path against a
manual YZ computation; Parkinson/GK non-negativity; NW SE consistency is tested separately in
stats_utils).
"""
import numpy as np


def _log(x):
    return np.log(np.where(x > 0, x, np.nan))


def _rolling_mean(x, window):
    """Causal trailing mean over `window` rows ending at t (t itself included), row t needs
    >= window valid rows in [t-window+1, t] else NaN. Uses a NaN-aware cumulative-sum trick
    (fast, no python loop); ties are not an issue since diffs of cumsum are exact only when no
    NaNs sit inside the window -- we require a full window of finite values."""
    T, A = x.shape
    finite = np.isfinite(x)
    xz = np.where(finite, x, 0.0)
    cs = np.cumsum(xz, axis=0)
    cc = np.cumsum(finite.astype(np.int64), axis=0)
    cs = np.vstack([np.zeros((1, A)), cs])
    cc = np.vstack([np.zeros((1, A), dtype=np.int64), cc])
    out = np.full((T, A), np.nan)
    for t in range(window - 1, T):
        lo = t - window + 1
        s = cs[t + 1] - cs[lo]
        n = cc[t + 1] - cc[lo]
        ok = n == window
        out[t, ok] = s[ok] / window
    return out


def _rolling_var(x, window, ddof=1):
    """Causal trailing sample variance (mean-centred within the window)."""
    m = _rolling_mean(x, window)
    x2 = x * x
    m2 = _rolling_mean(x2, window)
    var = m2 - m * m
    if ddof:
        var = var * (window / max(window - ddof, 1))
    return np.maximum(var, 0.0)


def overnight_return(open_, close):
    """log(open_t / close_{t-1}); row 0 is NaN (no previous close)."""
    prev_close = np.vstack([np.full((1, close.shape[1]), np.nan), close[:-1]])
    return _log(open_) - _log(prev_close)


def open_close_return(open_, close):
    return _log(close) - _log(open_)


def rogers_satchell_term(open_, high, low, close):
    lh, ll, lc, lo = _log(high), _log(low), _log(close), _log(open_)
    return (lh - lc) * (lh - lo) + (ll - lc) * (ll - lo)


def close_to_close_return(close):
    lc = _log(close)
    prev = np.vstack([np.full((1, close.shape[1]), np.nan), lc[:-1]])
    return lc - prev


def yang_zhang_variance(open_, high, low, close, window=30):
    n = window
    k = 0.34 / (1.34 + (n + 1) / max(n - 1, 1))
    ov = overnight_return(open_, close)
    oc = open_close_return(open_, close)
    rs = rogers_satchell_term(open_, high, low, close)
    var_o = _rolling_var(ov, window)
    var_c = _rolling_var(oc, window)
    mean_rs = _rolling_mean(rs, window)
    yz2 = var_o + k * var_c + (1 - k) * mean_rs
    return np.maximum(yz2, 0.0)


def yang_zhang_vol(open_, high, low, close, window=30):
    return np.sqrt(yang_zhang_variance(open_, high, low, close, window=window))


def close_to_close_vol(close, window=30):
    r = close_to_close_return(close)
    return np.sqrt(_rolling_var(r, window))


def parkinson_variance(high, low, window=30):
    hl2 = (_log(high) - _log(low)) ** 2
    m = _rolling_mean(hl2, window)
    return m / (4.0 * np.log(2.0))


def parkinson_vol(high, low, window=30):
    return np.sqrt(parkinson_variance(high, low, window=window))


def garman_klass_variance(open_, high, low, close, window=30):
    hl2 = (_log(high) - _log(low)) ** 2
    co2 = (_log(close) - _log(open_)) ** 2
    m_hl = _rolling_mean(hl2, window)
    m_co = _rolling_mean(co2, window)
    gk = 0.5 * m_hl - (2 * np.log(2) - 1) * m_co
    return np.maximum(gk, 0.0)


def garman_klass_vol(open_, high, low, close, window=30):
    return np.sqrt(garman_klass_variance(open_, high, low, close, window=window))


def range_to_close_ratio(high, low, close):
    """(H-L)/C, same-day, causal (uses only day-t OHLC, no rolling)."""
    return (high - low) / np.where(close != 0, close, np.nan)


def har_rv_terms(close, day_window=1, week_window=5, month_window=22):
    """Corsi (2009) HAR-RV: realised variance of close-to-close returns aggregated at
    daily/weekly/monthly windows (each causal trailing mean of squared returns)."""
    r2 = close_to_close_return(close) ** 2
    rv_d = _rolling_mean(r2, day_window) if day_window > 1 else r2
    rv_w = _rolling_mean(r2, week_window)
    rv_m = _rolling_mean(r2, month_window)
    return rv_d, rv_w, rv_m


def vol_of_vol(vol_series, window=30):
    """Rolling std of a (T,A) vol series itself (vol-of-vol feature)."""
    return np.sqrt(_rolling_var(vol_series, window))
