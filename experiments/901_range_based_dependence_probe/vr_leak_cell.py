#!/usr/bin/env python3
"""TASK_901 fix cycle 3 — power up the ONE cell where `variance_ratio` over-certifies.

Fix cycle 2 read `variance_ratio / persistent_component phi=0.999 / n=3000` (8 certifications,
0/8 contained) as "not readable as pass or fail at all" under its own n_certified < 20 rule. That
rule is correct for ESTABLISHING a pass and wrong for REFUTING one: 0/8 against a 0.93 bar has
p = 0.07^8 = 5.8e-10. Rather than argue the rule, this re-runs the SAME cell (same estimator,
same family, same n_dec, same everything) at 5x the trials on a fresh seed, so the cell's own
n_certified clears the threshold cycle 2 set for itself. Nothing else in the sweep is re-run; no
parameter is changed.

Reproduce (gpu-server, wrapped, own tmux):
    tools/safe_run.sh --name t901vrleak --cpu --mem 16G -- \
        trader_env/bin/python experiments/901_range_based_dependence_probe/vr_leak_cell.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from scipy import stats

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent))

import range_probe as rp        # noqa: E402
import coverage_harness as ch   # noqa: E402

N_TRIALS = 5000
SEED = 901


def main() -> int:
    rp.prewarm_calibration_table(blocks=rp.BLOCK_STRATA, n_decs=(3000,), n_jobs=8,
                                 n_trials_calib=200, n_trials_valid=200, n_boot=300)
    row = ch.run_coverage_cell("variance_ratio", "persistent_component", 3000,
                               {"phi": 0.999, "weight": 0.2}, N_TRIALS, seed=SEED, n_boot_ci=300)
    n, cov = row["n_certified"], row["coverage"]
    hits = int(round((cov or 0.0) * n))
    row["n_contained"] = hits
    # One-sided binomial test of H0: coverage >= 0.93 (the gate's bar).
    row["p_value_vs_bar_0.93"] = float(stats.binom.cdf(hits, n, 0.93)) if n else None
    lo, hi = (stats.beta.ppf(0.025, hits, n - hits + 1) if hits else 0.0,
              stats.beta.ppf(0.975, hits + 1, n - hits) if hits < n else 1.0)
    row["clopper_pearson_95"] = [float(lo), float(hi)]
    print(json.dumps(row, indent=2, default=str), flush=True)
    (_THIS.parent / "VR_LEAK_CELL.json").write_text(json.dumps(row, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
