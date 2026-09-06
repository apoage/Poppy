#!/usr/bin/env python3
"""TASK_916 leg 3 — restated benchmark table generator.

Calls ONLY existing public `trader.backtest` / `trader.data` functions (no modification, no
monkeypatch) to build the borrow-charged restatement table G2 requires: both borrow models
(none / kraken-margin) x >=2 fee tiers x both panels (survivor npz / PIT db-fixture) x the
named BEAR-sensitive re-grades (BEAR22, H=3d, 898 by-year). Read-only against
fixtures/trading_fixture.db (never trading_data.db, never regenerated) per the project's
verification-run invariant. CPU-only, seconds of runtime.

Usage: trader_env/bin/python experiments/916_cost_truth/build_benchmark.py
"""
import dataclasses
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from trader import config as tconfig
from trader import backtest as bt
from trader import data as tdata
from trader import cli as tcli

FIXTURE_DB = str(_REPO_ROOT / "fixtures" / "trading_fixture.db")

# Fee tiers (leg 2), bp/side one-sided. Verified 2026-09-06 against kraken.com/features/fee-schedule
# (primary source) -- see TASK_916 report for the fetch. 'legacy21' = the record's historical
# taker16+slip5 assumption, kept for continuity; 'base_taker'/'base_maker' = the July-2026
# ladder's Level-1 (<$10k/30d) tier, confirmed against the primary schedule.
FEE_TIERS_BP = {
    "legacy21": 21,
    "base_taker80": 80,
    "base_maker40": 40,
}
GATE_COSTS = (21,)


def _print_row(label, d, borrow_key=None):
    if d is None:
        print(f"    {label:34s}  (window too thin)")
        return
    net21 = d.get("net21")
    net21b = d.get("net21_borrow") if borrow_key else None
    line = f"    {label:34s}  n={d['n']:5d}  net21={net21:+.3f}"
    if borrow_key:
        line += f" -> borrow[{borrow_key}] {net21b:+.3f}"
    print(line)


def panel_npz():
    logc, ts_d, syms, dropped = tcli._npz_frozen_logc()
    return logc, ts_d, syms


def panel_db_fixture():
    ts_d, close_d, ids, syms, dinfo = tdata.load_db_backtest_panel(db_path=FIXTURE_DB,
                                                                   include_dead=False)
    with np.errstate(divide="ignore", invalid="ignore"):
        logc = np.log(np.where(close_d > 0, close_d, np.nan))
    return logc, ts_d, syms


def regime_table(panel_name, logc, ts_d, syms, scfg, rate_scenario="low"):
    print(f"\n=== {panel_name} panel -- H={scfg.H_rebal}d -- borrow scenario={rate_scenario} ===")
    print(f"    {'regime':34s}  {'n':>6s}  {'net21(none)':>12s}  -> borrow-charged / short-frac")
    rows = bt.regime_rows(np.asarray(ts_d))
    out = {}
    for name in tconfig.REG_NAMES:
        r0, r1 = rows[name]
        d = bt.borrow_window_report(logc, ts_d, scfg, r0, r1, syms,
                                    costs_bp=tuple(FEE_TIERS_BP.values()) + (0,),
                                    gate_costs=GATE_COSTS, rate_scenario=rate_scenario)
        out[name] = d
        if d is None:
            print(f"    {name:34s}  (window too thin)")
            continue
        print(f"    {name:8s}  n={d['n']:5d}  net21={d['net21']:+.3f} -> "
              f"{d['net21_borrow']:+.3f}  short_frac={d['short_frac_mean']*100:5.1f}%  "
              f"fee_tiers: " + "  ".join(
                  f"{tier}@{bp}bp={d[f'net{bp}']:+.3f}->{d[f'net{bp}_borrow']:+.3f}"
                  for tier, bp in FEE_TIERS_BP.items()))
    return out


def by_year_table(logc, ts_d, syms, scfg, rate_scenario="low"):
    """898-methodology re-grade: score each calendar year separately on the PIT (db) panel,
    unmodified scfg (H=1d, default STRATEGY), same construction as
    experiments/898_trend_decay/decay_harness.py (borrow_window_report internally reuses
    _decision_grid -> strategy.positions_from_config, same primitives 898 calls)."""
    print(f"\n=== PIT panel -- 898-methodology by-year re-grade -- scenario={rate_scenario} ===")
    years = [2021, 2022, 2023, 2024, 2025, 2026]
    for y in years:
        start = bt._to_epoch(f"{y}-01-01")
        end = bt._to_epoch(f"{y+1}-01-01") if y < 2026 else bt._to_epoch("2026-07-01")
        r_lo = int(np.searchsorted(ts_d, start))
        r_hi = int(np.searchsorted(ts_d, end))
        d = bt.borrow_window_report(logc, ts_d, scfg, r_lo, r_hi, syms,
                                    costs_bp=(8, 21), gate_costs=GATE_COSTS,
                                    rate_scenario=rate_scenario)
        label = f"{y}" if y < 2026 else "2026H1"
        if d is None:
            print(f"    {label:8s}  (window too thin)")
            continue
        print(f"    {label:8s}  n={d['n']:5d}  net8={d['net8']:+.3f} -> {d['net8_borrow']:+.3f}  "
              f"net21={d['net21']:+.3f} -> {d['net21_borrow']:+.3f}  "
              f"short_frac={d['short_frac_mean']*100:5.1f}%")


def main():
    print("# TASK_916 leg 3 -- restated benchmark table")
    print(f"# fixture db: {FIXTURE_DB}")

    logc_s, ts_s, syms_s = panel_npz()
    logc_p, ts_p, syms_p = panel_db_fixture()

    for scenario in ("low", "high"):
        regime_table("SURVIVOR (npz)", logc_s, ts_s, syms_s, tconfig.STRATEGY,
                     rate_scenario=scenario)
        regime_table("PIT (db/fixture)", logc_p, ts_p, syms_p, tconfig.STRATEGY,
                     rate_scenario=scenario)

    # H=3d robustness re-grade (survivor panel -- the anchor "H=3d survivor +1.35" lives here)
    scfg3 = dataclasses.replace(tconfig.STRATEGY, H_rebal=3)
    regime_table("SURVIVOR (npz)", logc_s, ts_s, syms_s, scfg3, rate_scenario="low")

    # 898 by-year re-grade (PIT panel)
    by_year_table(logc_p, ts_p, syms_p, tconfig.STRATEGY, rate_scenario="low")


if __name__ == "__main__":
    main()
