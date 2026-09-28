#!/usr/bin/env python3
"""TASK_915 feature families F1-F4 (memo Sec.4.2). Every feature reads data <= t only (raw,
un-standardised here -- cross-sectional standardisation per date happens centrally in
grid.py, over the live universe of THAT date, never here, so a feature computed today does not
depend on which OTHER dates are in the run).

F1 trend/reversal: lagged log-returns at {1,3,7,21,63}d, PLUS 30d (explicitly, to literally
  include the 819 single-30-day-trend signal as a control per the memo's instruction).
F2 range/vol family: Yang-Zhang / Parkinson / Garman-Klass vol, their ratios (OHLC-native vol
  efficiency, A-2's untested content), vol-of-vol, range-to-close, HAR-RV (Corsi) terms.
F3 cross-sectional: beta, idiosyncratic vol, dollar-volume (rank done centrally in grid.py
  alongside the shared per-date z-score, since a rank IS a cross-sectional operation).
F4 truncated (level<=3) log-signature of the trailing 30-day path of
  (log-close, log-range, log-volume, time) -- `signature.py`, Chen's identity, numpy only
  (no iisignature/esig on gpu-server's trader_env, verified 2026-09-28 -- disclosed
  substitution per the task instruction).
"""
import numpy as np

import yang_zhang as yz
import signature as sig
import targets as tg

F1_LOOKBACKS = (1, 3, 7, 21, 30, 63)


def build_f1(close, lookbacks=F1_LOOKBACKS):
    lc = tg.log_close(close)
    T, A = close.shape
    out = {}
    for lb in lookbacks:
        col = np.full((T, A), np.nan)
        col[lb:] = lc[lb:] - lc[:-lb]
        out[f"f1_trend_{lb}d"] = col
    return out


def build_f2(open_, high, low, close, window=30):
    yz_v = yz.yang_zhang_vol(open_, high, low, close, window=window)
    pk_v = yz.parkinson_vol(high, low, window=window)
    gk_v = yz.garman_klass_vol(open_, high, low, close, window=window)
    vov = yz.vol_of_vol(yz_v, window=window)
    r2c = yz.range_to_close_ratio(high, low, close)
    rv_d, rv_w, rv_m = yz.har_rv_terms(close)
    with np.errstate(divide="ignore", invalid="ignore"):
        pk_ratio = pk_v / np.maximum(yz_v, 1e-12)
        gk_ratio = gk_v / np.maximum(yz_v, 1e-12)
    return {
        "f2_yz_vol": yz_v, "f2_parkinson_vol": pk_v, "f2_gk_vol": gk_v,
        "f2_parkinson_over_yz": pk_ratio, "f2_gk_over_yz": gk_ratio,
        "f2_vol_of_vol": vov, "f2_range_to_close": r2c,
        "f2_har_rv_d": rv_d, "f2_har_rv_w": rv_w, "f2_har_rv_m": rv_m,
    }


def build_f3(close, beta, mkt1d, r1d, volume, window=30):
    idio = r1d - beta * mkt1d[:, None]
    from yang_zhang import _rolling_var
    idio_vol = np.sqrt(_rolling_var(idio, window))
    dollar_vol = _rolling_mean_local(close * np.maximum(volume, 0.0), window)
    return {"f3_beta": beta, "f3_idio_vol": idio_vol, "f3_dollar_volume": dollar_vol}


def _rolling_mean_local(x, window):
    from yang_zhang import _rolling_mean
    return _rolling_mean(x, window)


def build_f4(close, high, low, volume, window=30):
    """Per-asset rolling log-signature of the 4-dim path (log-close, log-range, log-volume,
    time), level<=3, flattened to columns f4_l1_*, f4_l2_*, f4_l3_*."""
    T, A = close.shape
    lc = tg.log_close(close)
    lr = np.log(np.maximum(high - low, 1e-12))
    lv = np.log(np.maximum(volume, 0.0) + 1.0)
    tix = np.arange(T, dtype=np.float64) / float(window)

    d = 4
    L1 = np.full((T, A, d), np.nan)
    L2 = np.full((T, A, d, d), np.nan)
    L3 = np.full((T, A, d, d, d), np.nan)
    for a in range(A):
        path = np.stack([lc[:, a], lr[:, a], lv[:, a], tix], axis=1)
        finite = np.isfinite(path).all(axis=1)
        if finite.sum() < window + 1:
            continue
        # rolling_log_signatures needs a fully-finite path; fill any interior gap (should not
        # occur post ffill/bfill) with the previous finite row to keep the recursion defined.
        p = path.copy()
        for j in range(1, T):
            if not finite[j]:
                p[j] = p[j - 1]
        l1, l2, l3 = sig.rolling_log_signatures(p, window)
        L1[:, a] = l1
        L2[:, a] = l2
        L3[:, a] = l3

    out = {}
    for i in range(d):
        out[f"f4_l1_{i}"] = L1[:, :, i]
    for i in range(d):
        for j in range(d):
            out[f"f4_l2_{i}_{j}"] = L2[:, :, i, j]
    for i in range(d):
        for j in range(d):
            for k in range(d):
                out[f"f4_l3_{i}_{j}_{k}"] = L3[:, :, i, j, k]
    return out


def build_all_features(panel, target_intermediates):
    """panel: dict with open/high/low/close/volume (T,A). target_intermediates: the dict
    `targets.build_all_targets` returns (reused for beta/mkt_ret/r1d so F3 stays consistent
    with the target construction -- one beta estimate, not two)."""
    feats = {}
    feats.update(build_f1(panel["close"]))
    feats.update(build_f2(panel["open"], panel["high"], panel["low"], panel["close"]))
    feats.update(build_f3(panel["close"], target_intermediates["beta"],
                          target_intermediates["mkt_ret_1d"],
                          tg.one_day_log_returns(panel["close"]), panel["volume"]))
    feats.update(build_f4(panel["close"], panel["high"], panel["low"], panel["volume"]))
    return feats


FEATURE_SET_MEMBERS = {
    "F1": lambda names: [n for n in names if n.startswith("f1_")],
    "F2": lambda names: [n for n in names if n.startswith("f2_")],
    "F3": lambda names: [n for n in names if n.startswith("f3_")],
    "F4": lambda names: [n for n in names if n.startswith("f4_")],
}


def feature_set_columns(all_feature_names, set_name):
    if set_name == "ALL":
        return list(all_feature_names)
    return FEATURE_SET_MEMBERS[set_name](all_feature_names)
