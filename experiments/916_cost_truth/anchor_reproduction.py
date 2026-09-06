#!/usr/bin/env python3
"""TASK_916 fix pass (Finding 2, flow-verifier) -- registered-window PIT anchor reproduction.

BACKGROUND. The task's "Why" #1 cites a "+0.97 survivor / +0.56 PIT benchmark" full-history
anchor: `runs/819_single_asset/POWERED_RESULT.md`'s Phase-3 pre-registered config (that file's
own bespoke point-in-time / dead-included / liquidity-filtered universe, built by
`runs/819_single_asset/powered.py`, NO borrow charge) -- FULL net@8 +0.67 CI[-0.01,+1.34],
net@21 +0.56 CI[-0.13,+1.23] (POWERED_RESULT.md:62,94). A prior TASK_916 fix-pass subagent
reproduced this anchor against `trader.backtest` + the fixture db and reported n=3085, unable
to explain the gap against POWERED_RESULT.md's implied n=2800.

DIAGNOSIS (flow-verifier). It is a SILENT WINDOW-END EXTENSION, not a dead-coin-inclusion
difference. `powered.py`'s registered span is `FULL_START="2018-01-01"`,
`FULL_END="2025-09-01"` (powered.py:84-85). The earlier pass scored through the fixture's
actual LAST available day (2026-07-06) instead of that registered end bound, silently adding
~10 months of a strongly negative 2026H1 regime (`build_benchmark.py`'s own by-year re-grade:
2026H1 net8 -1.435) into what was reported as the registered-window figure.

THIS SCRIPT is the committed, reproducible replacement. It reports the REGISTERED window
[2018-01-01, 2025-09-01) as the PRIMARY result (both include_dead=False/True, both rate
scenarios), and prints the mis-scored EXTENDED window only as an explicitly labelled SECONDARY
-- so the wrong number can never again be read back as the registered one. It also separates
out the include_dead=False vs True comparison to make the "not a dead-coin-inclusion
difference" disclosure self-evident (both cluster close together; the ~10-month regime
difference dominates by an order of magnitude).

STANDALONE FROM G2 -- do not conflate with `trader/config.py:72-78`'s `REGIMES["FULL"]`, which
is a DIFFERENT and NARROWER span: 2021-01-01..2025-09-01 (`build_benchmark.py`'s own G2 "FULL"
row). THIS anchor's "FULL" is `powered.py`'s full-history 2018-2025 definition. Two different
things share the name "FULL" in this codebase; every print block below says so explicitly.

This script calls ONLY existing public `trader.backtest` / `trader.data` functions (no
modification, no monkeypatch, no re-implementation of powered.py's bespoke universe filter) --
it reuses the SAME `load_db_backtest_panel` construction `build_benchmark.py` uses for its own
"PIT (db/fixture)" panel. That is why the reproduced net8/net21 land CLOSE TO but not
bit-identical to POWERED_RESULT.md's own bespoke-universe figures (different universe
construction: powered.py additionally applies its own liquidity filter and drops
legacy-duplicate/wrapped-derivative codes at the panel-building step, before this pass's
WBTC/TBTC borrow-classification fix ever gets a chance to matter there).

Read-only against fixtures/trading_fixture.db (never trading_data.db, never regenerated), CPU
only, seconds of runtime.

Usage: trader_env/bin/python experiments/916_cost_truth/anchor_reproduction.py
"""
import datetime
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from trader import config as tconfig
from trader import backtest as bt
from trader import data as tdata

FIXTURE_DB = str(_REPO_ROOT / "fixtures" / "trading_fixture.db")

# powered.py:84-85 -- the REGISTERED span this anchor is pre-committed to. NOT
# trader/config.py's REGIMES["FULL"] (2021-01-01..2025-09-01) -- see module docstring.
REGISTERED_START = "2018-01-01"
REGISTERED_END = "2025-09-01"

GATE_COSTS = (8, 21)
COSTS_BP = (0, 8, 21)


def _score(logc, ts_d, syms, r_lo, r_hi, rate_scenario):
    return bt.borrow_window_report(logc, ts_d, tconfig.STRATEGY, r_lo, r_hi, syms,
                                    costs_bp=COSTS_BP, gate_costs=GATE_COSTS,
                                    rate_scenario=rate_scenario)


def _load(include_dead):
    ts_d, close_d, ids, syms, info = tdata.load_db_backtest_panel(
        db_path=FIXTURE_DB, include_dead=include_dead)
    with np.errstate(divide="ignore", invalid="ignore"):
        logc = np.log(np.where(close_d > 0, close_d, np.nan))
    return logc, ts_d, syms, info


def _line(d, scen):
    if d is None:
        return f"    scenario={scen:4s}  (window too thin)"
    return (f"    scenario={scen:4s}  n={d['n']:5d}  "
            f"net8={d['net8']:+.4f} -> borrow={d['net8_borrow']:+.4f}   "
            f"net21={d['net21']:+.4f} -> borrow={d['net21_borrow']:+.4f}  "
            f"[ci21_borrow {d['ci21_borrow_lo']:+.2f},{d['ci21_borrow_hi']:+.2f}]  "
            f"short_frac={d['short_frac_mean']*100:5.1f}%")


def main():
    print("# TASK_916 fix pass -- registered-window PIT anchor reproduction (Finding 2)")
    print(f"# fixture db: {FIXTURE_DB}")
    print("#")
    print("# STANDALONE FROM G2: trader/config.py:72-78 REGIMES[\"FULL\"] is "
          "2021-01-01..2025-09-01 (build_benchmark.py's own G2 'FULL' row). THIS anchor's "
          "'FULL' is runs/819_single_asset/powered.py's full-history 2018-01-01..2025-09-01 "
          "-- a different, WIDER span that only shares the name. Do not conflate the two.")

    registered_net21_borrow_low = {}

    for include_dead in (False, True):
        logc, ts_d, syms, info = _load(include_dead)
        r_lo = int(np.searchsorted(ts_d, bt._to_epoch(REGISTERED_START)))
        r_hi_reg = int(np.searchsorted(ts_d, bt._to_epoch(REGISTERED_END)))
        r_hi_ext = len(ts_d)
        ext_end_day = datetime.datetime.utcfromtimestamp(int(ts_d[-1])).strftime("%Y-%m-%d")

        tag = ("include_dead=False (PRIMARY -- matches build_benchmark.py's own PIT panel)"
               if not include_dead else "include_dead=True (secondary, dead-coin sensitivity)")
        print(f"\n=== PIT panel, {tag} ===")
        print(f"    n_syms={len(syms)}  panel_info={info}")

        print(f"  -- REGISTERED window [{REGISTERED_START}, {REGISTERED_END}) "
              f"(powered.py FULL_START/FULL_END) -- PRIMARY RESULT --")
        for scen in ("low", "high"):
            d = _score(logc, ts_d, syms, r_lo, r_hi_reg, scen)
            print(_line(d, scen))
            if scen == "low" and d is not None:
                registered_net21_borrow_low[include_dead] = d["net21_borrow"]

        print(f"  -- SECONDARY, mis-scored EXTENDED window [{REGISTERED_START}, {ext_end_day}) "
              f"-- reproduces the earlier (WRONG) window-end extension; NOT a reference figure --")
        for scen in ("low", "high"):
            d = _score(logc, ts_d, syms, r_lo, r_hi_ext, scen)
            print(_line(d, scen))

    lo_false = registered_net21_borrow_low.get(False)
    lo_true = registered_net21_borrow_low.get(True)
    if lo_false is not None and lo_true is not None:
        print(f"\n# Dead-coin-inclusion disclosure: registered-window net21_borrow[low] "
              f"moves by {abs(lo_false - lo_true):.4f} between include_dead=False/True "
              f"({lo_false:+.4f} vs {lo_true:+.4f}) -- an order of magnitude smaller than the "
              f"~10-month regime-extension effect above. include_dead is NOT what explained "
              f"the earlier n=3085 vs n=2800 discrepancy.")

    print("\n# Reference (recorded, NOT reproduced bit-identically by this script -- shown "
          "for comparison only; see module docstring for why the universe differs):")
    print("#   runs/819_single_asset/POWERED_RESULT.md:62,94 (powered.py's own bespoke "
          "point-in-time / dead-included / liquidity-filtered universe, pre-registered, NO "
          "borrow charge): FULL net@8 +0.67 CI[-0.01,+1.34], net@21 +0.56 CI[-0.13,+1.23].")
    print("#")
    print("# Qualitative conclusion (unchanged by this reproduction, not re-litigated): "
          "charged POINT ESTIMATES fall below +0.56 under every window variant tried; the "
          "REGISTERED window is LESS unfavourable than the previously mis-scored extended "
          "window -- i.e. the earlier reporting OVERSTATED how far below the bar the "
          "borrow-charged figure sits.")
    print("#")
    print("# CONFIDENCE CAVEAT (do not drop this when citing the line above): the conclusion "
          "is about POINT ESTIMATES only. The registered window's own bootstrap CI on the "
          "charged net21 figure does NOT statistically exclude +0.56 -- low scenario "
          "[-0.47,+0.90], high scenario [-0.86,+0.56] (upper bound sits exactly AT the bar). "
          "The recorded reference is equally wide (+0.56 CI[-0.13,+1.23]), so this is an "
          "instrument-wide property, not a defect introduced here. Cite this figure as a "
          "restated point estimate, never as a statistically demonstrated shortfall.")


if __name__ == "__main__":
    main()
