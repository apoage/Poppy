#!/usr/bin/env python3
"""TASK_915 D85 MDE-at-spec-time: minimum detectable mean IC at NW t>=3 after Bonferroni, per
horizon, computed from the PANEL ALONE (live-asset counts per date over the G-A window,
2024-01-01..2026-07-01) -- NO fitting, run BEFORE the first real fit and pinned into SPEC.md.

NW inflation assumption (stated explicitly, per the task instruction): h-day forward returns
computed on EVERY day overlap by up to h-1 days, inducing MA(h-1)-type serial correlation in
the per-date IC series even under the pure null. The standard overlapping-window variance
inflation factor for a Bartlett-kernel NW SE with lag=h on such a series is `2h-1` (h=1 -> 1,
no inflation; h=21 -> 41x) -- this is the assumed `nw_inflation` plugged into
`stats_utils.mde_ic`. It is a CONVENTION (the exact inflation depends on the true
autocorrelation structure, measured only after fitting), disclosed rather than derived.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stats_utils as su   # noqa: E402
import gates as gt          # noqa: E402

ALPHA = 0.05
N_TESTS = 160
BAND = 0.02


def nw_inflation_for_h(h):
    return float(2 * h - 1)


def mde_table(present_mask, day_ts, horizons=(1, 3, 7, 21)):
    lo, hi = gt._window_rows(day_ts, *gt.GA_WINDOW)
    n_live = present_mask[lo:hi].sum(axis=1)
    n_dates = int((n_live >= 3).sum())
    n_bar = float(n_live[n_live >= 3].mean()) if n_dates else np.nan
    out = {}
    for h in horizons:
        infl = nw_inflation_for_h(h)
        mde, t_crit = su.mde_ic(n_bar - 1, n_dates, ALPHA, N_TESTS, infl, power_t=3.0)
        out[str(h)] = {"n_dates": n_dates, "n_bar_live_assets": n_bar, "nw_inflation": infl,
                        "t_crit_bonferroni": t_crit, "mde_ic_at_t3": mde,
                        "underpowered_vs_0.02_band": bool(mde > BAND)}
    return out


if __name__ == "__main__":
    # standalone usage: python mde_report.py <panel_npz_or_via_loader>
    import loader
    panel = loader.load_panel()
    panel = loader.apply_liquidity_filter(panel)
    import grid
    present, fa = grid.build_present_mask(panel)
    tbl = mde_table(present, panel["day_ts"])
    print(json.dumps(tbl, indent=2))
