#!/usr/bin/env python3
"""TASK_923 data layer -- reuses TASK_917 P3's panel loader and market-residual construction
UNCHANGED (task instruction: "same `daily_panel.py` builder, unchanged; same listing-date
source"). No copy, no reimplementation: this module imports `daily_panel.py` and
`p3_post_listing_event_study.py` directly from
`experiments/917_second_moment_and_event_instruments/` via sys.path insertion -- the same
cross-experiment-reuse pattern `experiments/918_tactic_hull/loader.py` uses (there it
re-implements a helper instead because of a circular-import risk that does not exist here;
here a plain import is safe since neither 917 module has a competing top-level name and
neither is executed as `__main__` on import).

Read-only everywhere: `daily_panel.load_raw_pit_panel` opens
`mcts_trader.utils.db._ro_connect` (`file:...?mode=ro`). Fixture only
(`fixtures/trading_fixture.db`); never `trading_data.db`.
"""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
_P3_DIR = _REPO_ROOT / "experiments" / "917_second_moment_and_event_instruments"
for _p in (str(_REPO_ROOT), str(_P3_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import daily_panel as dp                 # noqa: E402  (917, unchanged, read-only import)
import p3_post_listing_event_study as p3  # noqa: E402  (917, unchanged, read-only import)

FIXTURE = str(_REPO_ROOT / "fixtures" / "trading_fixture.db")
AGE_MAX = p3.AGE_MAX   # 89 -- P3's "first 90 days" window; provenance only in this row,
                       # the event book's own exit family is independent (see SPEC.md).


def load_panel(db_path=FIXTURE, include_dead=True):
    """Raw PIT daily OHLCV panel -- 917's `load_raw_pit_panel`, unchanged. include_dead=True
    matches P3's headline (dead-coin retention is load-bearing per the P3 task file)."""
    return dp.load_raw_pit_panel(db_path, include_dead=include_dead)


def market_residual_inputs(panel):
    """Same construction P3's `true_ic_for_panel` uses: daily log return, equal-weight PIT
    market (beta=1 demeaning), per-asset active-life bounds (first/last real print row).
    Returns dict: dr (T,A) daily log return, logc (T,A) log close, mkt (T,) equal-weight
    market daily log return, first (A,) first active row idx (-1 if never listed),
    last (A,) last active row idx."""
    dr, logc = dp.daily_log_return(panel["close"])
    mkt = p3.equal_weight_market(dr)
    first, last = p3.first_last_active(panel["close"])
    return {"dr": dr, "logc": logc, "mkt": mkt, "first": first, "last": last}
