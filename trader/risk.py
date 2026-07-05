#!/usr/bin/env python3
"""TASK_820 risk overlay — the "with risk management" fork of TASK_819. Execution-level, NOT
signal shaping: it caps, throttles, bands, and staleness-guards the frozen signal's positions.

Overlay (spec §Risk overlay):
  1. Per-asset position cap: |pos_i| <= per_asset_cap of book.
  2. Drawdown throttle vs a high-water-mark. TWO MODES (RiskConfig.dd_mode):
     - 'shadow_scale' (default): the drawdown that drives the throttle is the SHADOW
       equity curve — the uncapped frozen strategy's own overlay-OFF net equity, which the
       caller tracks and passes as (equity, hwm). Levels: dd > dd_halve -> 1.0;
       dd_halve >= dd > dd_flat -> 0.5; dd <= dd_flat -> dd_deep_scale (0.25, never flat).
       The shadow book keeps trading, recovers, and the throttle re-raises automatically.
     - 'realized_flat' (legacy): (equity, hwm) = the throttled book's realized curve;
       dd <= dd_flat -> flat + halt flag. Dead-state hazard: once flat, realized equity
       freezes so dd never recovers and the book never re-enters (kept for comparison).
  3. Rebalance band: skip trades with |Delta pos_i| < rebalance_band of book (hold prev).
  4. Staleness guard: newest daily bar older than staleness_days -> no NEW positions (hold
     current book), log alert.

Defaults OFF (RiskConfig.enabled=False) so backtest == 819 (gate 1a). Paper flips it ON.
All kernels are numpy (small A-vectors; the overlay is a per-decision scalar/vector op, not a
batch hot path)."""

import numpy as np

from .config import RISK, SEC_PER_DAY


def apply_position_cap(pos, cap):
    """Overlay 1: clip each position to +/- cap of book."""
    return np.clip(pos, -cap, cap)


def dd_scale(equity, hwm, rcfg=RISK):
    """Overlay 2: drawdown throttle. dd = equity/hwm - 1 (<= 0). Returns (scale, halt, dd).

    dd_mode='shadow_scale' (default; equity/hwm = the SHADOW overlay-OFF curve):
      dd <= dd_flat  -> scale dd_deep_scale (0.25), halt False   (never flat -> no dead state)
      dd <= dd_halve -> scale 0.5, halt False
      else           -> scale 1.0, halt False
    dd_mode='realized_flat' (legacy; equity/hwm = the throttled book's realized curve):
      dd <= dd_flat  -> scale 0.0, halt True
      dd <= dd_halve -> scale 0.5, halt False
      else           -> scale 1.0, halt False"""
    hwm = max(hwm, 1e-12)
    dd = equity / hwm - 1.0
    if rcfg.dd_mode == "realized_flat":
        if dd <= rcfg.dd_flat:
            return 0.0, True, dd
        if dd <= rcfg.dd_halve:
            return 0.5, False, dd
        return 1.0, False, dd
    if rcfg.dd_mode != "shadow_scale":
        raise ValueError(f"unknown dd_mode {rcfg.dd_mode!r}")
    if dd <= rcfg.dd_flat:
        return rcfg.dd_deep_scale, False, dd
    if dd <= rcfg.dd_halve:
        return 0.5, False, dd
    return 1.0, False, dd


def apply_rebalance_band(target, prev, band):
    """Overlay 3: skip sub-band trades — where |target_i - prev_i| < band, hold prev_i."""
    target = np.asarray(target, dtype=np.float64)
    prev = np.asarray(prev, dtype=np.float64)
    return np.where(np.abs(target - prev) < band, prev, target)


def bar_age_days(latest_ts, now_ts):
    """Age (days) of the newest daily bar relative to `now_ts` (epoch seconds)."""
    return (float(now_ts) - float(latest_ts)) / SEC_PER_DAY


def is_stale(latest_ts, now_ts, rcfg=RISK):
    """Overlay 4 predicate: True if the newest bar is older than staleness_days."""
    return bar_age_days(latest_ts, now_ts) > rcfg.staleness_days


def apply_overlay(raw_target, prev_pos, equity, hwm, latest_ts, now_ts, rcfg=RISK):
    """Apply overlays 1-4 to a single decision's target position vector.

    (equity, hwm) is the curve DRIVING the DD throttle: in dd_mode='shadow_scale' the caller
    passes the SHADOW (uncapped overlay-OFF) equity/HWM; in 'realized_flat' the throttled
    book's realized equity/HWM (legacy).

    Returns (final_pos, info). If the overlay is disabled, returns raw_target unchanged with
    info['enabled']=False (backtest parity path). Ordering: staleness guard first (data problem
    -> hold current book, no new positions), else cap -> DD throttle -> rebalance band."""
    raw_target = np.asarray(raw_target, dtype=np.float64)
    prev_pos = np.asarray(prev_pos, dtype=np.float64)
    info = {"enabled": rcfg.enabled, "stale": False, "halt": False, "dd": 0.0,
            "dd_scale": 1.0, "n_capped": 0, "n_banded": 0}

    if not rcfg.enabled:
        return raw_target, info

    age = bar_age_days(latest_ts, now_ts)
    info["bar_age_days"] = age
    if age > rcfg.staleness_days:
        # data too old -> do NOT open/adjust positions; hold the current book, alert.
        info["stale"] = True
        return prev_pos.copy(), info

    # 1. per-asset cap
    capped = apply_position_cap(raw_target, rcfg.per_asset_cap)
    info["n_capped"] = int(np.sum(np.abs(raw_target) > rcfg.per_asset_cap + 1e-15))

    # 2. drawdown throttle
    scale, halt, dd = dd_scale(equity, hwm, rcfg)
    info["dd"], info["dd_scale"], info["halt"] = dd, scale, halt
    throttled = capped * scale

    # 3. rebalance band (skip sub-band changes -> hold prev)
    final = apply_rebalance_band(throttled, prev_pos, rcfg.rebalance_band)
    info["n_banded"] = int(np.sum(np.abs(throttled - prev_pos) < rcfg.rebalance_band))
    return final, info
