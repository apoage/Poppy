#!/usr/bin/env python3
"""TASK_918 scorer — calls ONLY unmodified `trader.backtest` primitives
(`regime_series`, `net_series`, `_sharpe`, `_block_bootstrap_sharpe_ci`, `gross_simple_series`,
`borrow_cost_series`, `net_series_with_borrow`). Never uses `cell_metrics` (per the task file:
"it cannot score these windows"). This is the same set of primitives
`experiments/898_trend_decay/decay_harness.py:score_window` calls -- reproduced here (not
imported across the digit-prefixed package) so any externally-built (pos, pres, warm) triple,
including a MIXED hull book, can be scored on an arbitrary [start_ts, end_ts) window.

Fee tiers reuse the exact July-2026 numbers TASK_916 verified and shipped
(`experiments/916_cost_truth/build_benchmark.py:36`, cited not re-fetched): legacy21 (the
record's historical continuity tier), base_taker80 (verified Level-1 taker), base_maker40
(verified Level-1 maker). Borrow model reuses `trader.backtest.borrow_cost_series` /
`net_series_with_borrow` (TASK_916 leg 1) unmodified, at both its 'low'/'high' scenarios.
"""
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from trader import backtest as bt        # noqa: E402
from trader.config import K, MINUTES_PER_YEAR   # noqa: E402

# TASK_916 build_benchmark.py:36 -- cited, not re-fetched.
FEE_TIERS_BP = {"legacy21": 21, "base_maker40": 40, "base_taker80": 80}
GATE_COST_BP = 21          # the +0.97 bar's own denomination (819/`docs/pathway.md:12`)
BORROW_SCENARIOS = ("low", "high")
N_BOOT = 2000
SEED = 918


def h_grid(T, warm, r_lo, r_hi, H):
    """The EXACT H-strided entry-row grid `trader.backtest.regime_series` derives internally
    (`backtest.py:212-218`, reproduced verbatim here as bookkeeping only -- the actual scoring
    arithmetic below still goes through `regime_series`/`net_series`/`_sharpe` unmodified).
    Returns None under the identical thinness contract (< 6 decisions)."""
    lo = max(warm, r_lo)
    hi = min(r_hi - H, T - 1 - H)
    if hi - lo < H:
        return None
    t = np.arange(lo, hi + 1, H)
    if t.size < 6:
        return None
    return t


def score_arm(logc, pos, pres, ts_sub, warm, syms, H=1, start_ts=None, end_ts=None,
             n_decisions_monthly=None, n_boot=N_BOOT, seed=SEED):
    """Score one (pos, pres, warm) triple over [start_ts, end_ts). Returns None if too thin
    (same thinness contract as `regime_series`), else a dict with log AND simple gross/net at
    every fee tier, borrow-charged net@21 at both scenarios, block-bootstrap CI at net@21
    (log, no borrow -- the PRIMARY STEP-0 kill reading), turnover/exposure diagnostics, and
    `n_decisions_monthly` attached to every number per D81(c)."""
    logc_np = bt.config.to_numpy(logc)
    pos_np = bt.config.to_numpy(pos)
    pres_np = np.asarray(pres)
    ts_sub = np.asarray(ts_sub)
    T = ts_sub.size
    r_lo = int(np.searchsorted(ts_sub, start_ts)) if start_ts is not None else 0
    r_hi = int(np.searchsorted(ts_sub, end_ts)) if end_ts is not None else T

    res = bt.regime_series(logc_np, pos_np, pres_np, r_lo, r_hi, H, warm, xp=np)
    if res is None:
        return None
    gross_log, turn, bh = res
    t = h_grid(logc_np.shape[0], warm, r_lo, r_hi, H)
    assert t is not None and t.size == gross_log.size, "h_grid must reproduce regime_series's own grid"

    gross_simple = bt.gross_simple_series(logc_np, pos_np, t, H)
    period = H * K
    n = int(gross_log.size)
    years = (n * H * K) / MINUTES_PER_YEAR

    d = {
        "n_daily_decisions": n, "n_decisions_monthly": n_decisions_monthly,
        "years": float(years), "H": int(H),
        "turn_mean": float(np.mean(turn)), "ann_turn": float(np.sum(turn) / years) if years > 0 else float("nan"),
        "gross_exposure_mean": float(np.mean(np.sum(np.abs(pos_np[t]), axis=1))),
        "flat_frac": float(np.mean(np.sum(np.abs(pos_np[t]), axis=1) < 1e-12)),
        "gross_log_sharpe": bt._sharpe(gross_log, period),
        "gross_simple_sharpe": bt._sharpe(gross_simple, period),
        "bh_sharpe": bt._sharpe(bh, period),
        "net_log": {}, "net_simple": {},
        "borrow": {},
    }
    for label, c in FEE_TIERS_BP.items():
        d["net_log"][label] = bt._sharpe(bt.net_series(gross_log, turn, c), period)
        d["net_simple"][label] = bt._sharpe(bt.net_series(gross_simple, turn, c), period)
    d["net_log"]["zero"] = bt._sharpe(gross_log, period)
    d["net_simple"]["zero"] = bt._sharpe(gross_simple, period)

    nh21 = bt.net_series(gross_log, turn, GATE_COST_BP)
    ci = bt._block_bootstrap_sharpe_ci(nh21, period, period, n_boot, seed)
    d["ci_net21_log_lo"], d["ci_net21_log_hi"] = float(ci[0]), float(ci[1])
    eq = np.exp(np.cumsum(nh21))
    peak = np.maximum.accumulate(eq)
    d["mult_net21_log"] = float(eq[-1])
    d["maxdd_net21_log"] = float(np.min(eq / peak - 1.0))

    for scen in BORROW_SCENARIOS:
        borrow_cost = bt.borrow_cost_series(pos_np, syms, t, H, rate_scenario=scen)
        scen_block = {}
        for label, c in FEE_TIERS_BP.items():
            nh = bt.net_series_with_borrow(gross_log, turn, c, borrow_cost)
            scen_block[f"net_log_{label}"] = bt._sharpe(nh, period)
        nh21b = bt.net_series_with_borrow(gross_log, turn, GATE_COST_BP, borrow_cost)
        ci_b = bt._block_bootstrap_sharpe_ci(nh21b, period, period, n_boot, seed)
        scen_block["ci_net21_lo"], scen_block["ci_net21_hi"] = float(ci_b[0]), float(ci_b[1])
        scen_block["borrow_bp_per_day"] = float(np.mean(borrow_cost)) * 1e4 / H
        scen_block["short_frac_mean"] = float(np.mean(
            np.sum(np.maximum(-pos_np[t], 0.0), axis=1) /
            np.maximum(np.sum(np.abs(pos_np[t]), axis=1), 1e-300)))
        d["borrow"][scen] = scen_block

    return d


def by_year_windows():
    """Calendar-year + half-year sub-windows inside the STEP-0 scoring window
    (2024-01-01 .. 2026-07-01), for the by-year disclosure the task requires."""
    return [
        ("2024", bt._to_epoch("2024-01-01"), bt._to_epoch("2025-01-01")),
        ("2025", bt._to_epoch("2025-01-01"), bt._to_epoch("2026-01-01")),
        ("2026H1", bt._to_epoch("2026-01-01"), bt._to_epoch("2026-07-01")),
    ]
