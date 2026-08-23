#!/usr/bin/env python3
"""TASK_901 — the full out-of-family certification run.

Reproduce (server, wrapped, see the task file's compute section):
    tools/safe_run.sh --name t901validate --mem 16G -- \
        trader_env/bin/python experiments/901_range_based_dependence_probe/run_validation.py \
        --out experiments/901_range_based_dependence_probe/RESULTS.json

Prints the coverage table (gate i) and the persistent-component readout (gate ii) and writes the
full machine-readable record to `--out`. Gate (iii) — the real-data leg — lives in
`run_real_leg.py` (workstation, data-local; see its docstring) and is MERGED in by
`combine_results.py`; passing `--skip-real` (the server invocation) omits it here entirely. There
is no gate-check/threshold logic in this file: it MEASURES and records, and the reading is done in
`measured_findings.py`. An earlier revision of `coverage_harness.py` pointed at a "gate (iii)
check" here that never existed; that dangling reference is removed rather than back-filled.

TRIAL COUNT (fix cycle 2): 1000 trials/cell, up from the shipped 200. At 200 the PASSING cells
were not resolved against the 0.93 bar (e.g. acf_band/triangular_dep22/n=1000 read 0.9436 +- 0.0165
— z=0.74, indistinguishable from the bar); at 1000 the binomial SE near p=0.95 is ~0.0069, which
separates 0.95 from 0.93 at ~3 SE. The FAILING cells were already conclusive at 200 (8.8-22 sigma)
— the raise buys resolution on the cells that were being read as clean, not on the verdict.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent))

import range_probe as rp        # noqa: E402
import coverage_harness as ch   # noqa: E402
# `families` / `real_series` are NOT imported here: the cell list below names families as
# strings (resolved inside `coverage_harness`), and the real-data leg is delegated to
# `run_real_leg` inside the one branch that needs it — keeping `probe`/`decide` (which
# `real_series` pulls in, and which need the 806/808/809 run trees on disk) off the import path
# of the server's `--skip-real` invocation entirely.

CALIB_N_DEC_GRID = (200, 500, 1000, 3000, 10000)
N_TRIALS = 1000
N_BOOT_CI = 300

# (family, kwargs, n_dec) cells. `triangular` uses dep values DELIBERATELY NOT in
# `range_probe.BLOCK_STRATA` (the calibration draws' own dep set) — a genuinely held-out
# in-shape validation, not a re-check of the calibration draws themselves.
HELD_OUT_TRIANGULAR_DEPS = (4, 10, 15, 22, 32, 50)

CELLS = []
for _n_dec in (200, 1000, 3000):
    CELLS.append(("iid", {}, _n_dec))
for _dep in HELD_OUT_TRIANGULAR_DEPS:
    for _n_dec in (1000, 3000):
        CELLS.append(("triangular", {"dep": _dep}, _n_dec))
for _phi in (0.9, 0.99, 0.999):
    CELLS.append(("ar1", {"phi": _phi}, 3000))
for _phi in (0.99, 0.999):
    CELLS.append(("persistent_component", {"phi": _phi, "weight": 0.2}, 3000))
CELLS.append(("seasonal_regime", {"period": 200, "phi_low": 0.1, "phi_high": 0.9}, 3000))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(_THIS.parent / "RESULTS.json"))
    ap.add_argument("--n-jobs-calib", type=int, default=8)
    ap.add_argument("--n-trials", type=int, default=N_TRIALS)
    ap.add_argument("--skip-real", action="store_true",
                    help="skip the real-data leg (gate iii) — synthetic-only run")
    args = ap.parse_args()

    t0 = time.time()
    print(f"[901] prewarming calibration table: {len(rp.BLOCK_STRATA)} blocks x "
          f"{len(CALIB_N_DEC_GRID)} n_dec = {len(rp.BLOCK_STRATA) * len(CALIB_N_DEC_GRID)} cells, "
          f"n_jobs={args.n_jobs_calib}", flush=True)
    rp.prewarm_calibration_table(blocks=rp.BLOCK_STRATA, n_decs=CALIB_N_DEC_GRID,
                                 n_jobs=args.n_jobs_calib, n_trials_calib=200,
                                 n_trials_valid=200, n_boot=300)
    print(f"[901] calibration table ready in {time.time() - t0:.0f}s "
          f"({len(rp._K_CACHE)} entries)", flush=True)

    coverage_rows = []
    for estimator in rp.ESTIMATORS:
        for family, kwargs, n_dec in CELLS:
            tcell = time.time()
            row = ch.run_coverage_cell(estimator, family, n_dec, kwargs, args.n_trials, seed=42,
                                       n_boot_ci=N_BOOT_CI)
            coverage_rows.append(row)
            print(f"[901] {estimator:15s} {family:22s} n_dec={n_dec:6d} {kwargs} "
                  f"certify={row['certify_rate']:.2f} cov={row['coverage']} "
                  f"se={row['coverage_se']} med_range={row['median_range_est']} "
                  f"({time.time() - tcell:.1f}s)", flush=True)

    real_leg = None
    if not args.skip_real:
        # ONE implementation of the real-data leg, not two: `run_real_leg.py` owns it (all three
        # horizons + the stationarity diagnostics), and is invoked here rather than duplicated.
        import run_real_leg   # noqa: PLC0415  (deliberately local — only this branch needs it)
        run_real_leg.main()
        real_leg = json.loads((_THIS.parent / "REAL_LEG.json").read_text())["real_leg"]

    out = {
        "n_trials_per_cell": args.n_trials, "n_boot_ci": N_BOOT_CI,
        "calib_n_dec_grid": list(CALIB_N_DEC_GRID), "block_strata": list(rp.BLOCK_STRATA),
        "coverage_rows": coverage_rows, "real_leg": real_leg,
        "runtime_s": time.time() - t0,
    }
    Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    print(f"[901] wrote {args.out} ({time.time() - t0:.0f}s total)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
