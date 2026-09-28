#!/usr/bin/env python3
"""AD-HOC diagnostic (not part of the pytest suite): the fixture smoke measured surprisingly
HIGH IC (0.11-0.125, NW-t ~20) for a plain F1/h=1 cell -- 3x the prior literature ceiling
(819 CAPACITY's pooled max 0.037). Before trusting that number for a launch decision, run the
SAME cell with the target's DATE axis independently permuted (a same-real-data negative
control, same logic as tests/test_positive_control.py but on the real panel instead of
synthetic) -- if the permuted version ALSO shows high |IC|, something in the pipeline is
leaking/misaligned; if it collapses to noise, the real measurement is more likely genuine
(cross-sectional per-date IC vs E-2's weaker pooled time-series IC are different, not
necessarily contradictory, statistics)."""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import loader   # noqa: E402
import grid     # noqa: E402
import gates    # noqa: E402

t0 = time.time()
panel = loader.load_panel()
panel = loader.apply_liquidity_filter(panel)
print(f"panel loaded n_assets={len(panel['syms'])} n_days={panel['day_ts'].size} "
      f"({time.time()-t0:.0f}s)", flush=True)

built = grid.build_panel_features_and_targets(panel)
print(f"features/targets built ({time.time()-t0:.0f}s)", flush=True)

h = 1
tname = "y1_primary_resid_yz"
res = grid.fit_cell(built, panel["day_ts"], h, tname, "F1", "ridge",
                     first_test_date="2024-01-01", last_date="2024-12-31")
target_real = built["targets_by_h"][h][tname]
stats_real = gates.cell_ic_stats(res["forecast"], target_real, panel["day_ts"], h)
print("REAL cell:", json.dumps({k: v for k, v in stats_real.items()
                                 if k not in ("ic_series_ga_window", "day_ts_ga_window")}))

rng = np.random.default_rng(0)
perm = rng.permutation(target_real.shape[0])
target_shuffled = target_real[perm]
stats_shuf = gates.cell_ic_stats(res["forecast"], target_shuffled, panel["day_ts"], h)
print("DATE-SHUFFLED target cell (negative control):",
      json.dumps({k: v for k, v in stats_shuf.items()
                  if k not in ("ic_series_ga_window", "day_ts_ga_window")}))

# also: same cell, cross-sectional (per-date asset-axis) permutation null already implemented
ga_lo, ga_hi = gates._window_rows(panel["day_ts"], *gates.GA_WINDOW)
import stats_utils as su
null = su.per_date_permutation_null(res["forecast"][ga_lo:ga_hi], target_real[ga_lo:ga_hi],
                                     n_draws=100, rng=rng)
print("per-date cross-sectional permutation null: mean=%.4f sd=%.4f (real mean_ic=%.4f)" %
      (np.nanmean(null), np.nanstd(null), stats_real["mean_ic"]))
print(f"done ({time.time()-t0:.0f}s)")
