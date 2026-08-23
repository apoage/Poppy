#!/usr/bin/env python3
"""TASK_901 fix cycle 3 — the two DIAGNOSTIC measurements this cycle's corrections require.

This file adds NO estimator, changes NO parameter, and re-runs NO coverage cell. It measures two
things the shipped artifact asserted without separating them:

  A. `RULE_VS_STATISTIC` (fix for cycle-3 MAJOR 1). `range_probe.first_levelled_index` compares
     every later grid point against a FIXED relative tolerance (`plateau_tol = 0.10`) while the
     SAMPLING SD of VR(q) grows with q. This block measures that sd on i.i.d. draws (where the
     population VR is exactly 1 at every q, so all dispersion is sampling noise), the refusal rate
     the fixed rule produces there, which grid points drive the refusals, and — as a
     COUNTERFACTUAL ONLY, never wired into `range_probe` — what an SE-aware tolerance would have
     done on the SAME draws. The point is to separate "the rule at this grid refuses" from "the
     variance-ratio statistic is conservative". Nothing here is a tuning proposal; see
     `measured_findings.RULE_IS_NOT_THE_STATISTIC` for the reading.

  B. `REFUSAL_CHANNELS` (fix for cycle-3 MAJOR 2). A "refusal" in `coverage_harness.run_coverage_
     cell` is a single count that merges FIVE distinct events: the estimator returning NaN, the
     range exceeding `MAX_BLOCK_CAP=60`, too few non-overlapping blocks, the width calibration
     not achieving target, and a bootstrap that returned a non-finite Sharpe or CI bound. This
     module splits the FIRST FOUR; the fifth cannot appear here because this leg draws no
     bootstrap, and it measured ZERO in the runs that could show it (0 `unusable_bootstrap` in
     228 real-leg per-chunk records) -- which is why the four-way split below reconciles against
     that five-branch counter. Those are different facts about different components — in particular a
     cap refusal says the estimator DID return a number and the BLOCK CEILING rejected it. This
     block re-runs the estimator + `block_rule` (the real code path, imported, not re-implemented)
     over the same 63 cells and splits the refusal count into those first four channels. No bootstrap CI
     is drawn here, so this leg is cheap; `certifiable_rate` is the analogue of the sweep's
     `certify_rate` and is cross-checked against it.

FRESH DRAWS, DISCLOSED: this runs at `seed=901`, not the sweep's `seed=42`, and the sweep's RNG
stream is additionally advanced by bootstrap draws that this leg does not make. So these numbers
are an INDEPENDENT Monte Carlo estimate of the same quantities, not a replay — every rate carries
its binomial MC SE, and B's `certifiable_rate` vs the sweep's `certify_rate` is a real consistency
check rather than an identity.

Reproduce (gpu-server, wrapped, own tmux):
    tools/safe_run.sh --name t901readoff --cpu --mem 16G -- \
        trader_env/bin/python experiments/901_range_based_dependence_probe/readoff_diagnostics.py \
        --out experiments/901_range_based_dependence_probe/READOFF_DIAG.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent))

import range_probe as rp        # noqa: E402
import families as fam          # noqa: E402
import run_validation as rv     # noqa: E402  (the cell list, imported so it can never drift)

N_DRAWS = 1000
SEED = 901


def _binom_se(p: float, n: int) -> float:
    if n <= 0:
        return float("nan")
    return float(math.sqrt(max(0.0, p * (1.0 - p)) / n))


def _vr_grid(n: int):
    max_q = max(2, n // 10)
    g, q = [], 2
    while q <= max_q:
        g.append(q)
        q *= 2
    return g


def rule_vs_statistic(n_dec: int, n_draws: int = N_DRAWS, seed: int = SEED) -> dict:
    """A: the fixed-tolerance read-off rule measured against the sampling sd of its own input."""
    rng = np.random.default_rng(seed)
    grid = _vr_grid(n_dec)
    curves = np.empty((n_draws, len(grid)), dtype=np.float64)
    for t in range(n_draws):
        x = fam.simulate_iid(n_dec, rng=rng)
        curves[t] = [rp.variance_ratio(x, q) for q in grid]

    sd = curves.std(axis=0, ddof=1)
    # SE of a sample sd under approximate normality: sd / sqrt(2*(N-1)).
    sd_se = sd / math.sqrt(2.0 * (n_draws - 1))

    # The rule EXACTLY as shipped (imported, not re-implemented).
    idx = [rp.first_levelled_index(list(c), 0.10) for c in curves]
    n_nan = sum(1 for i in idx if i is None)
    returned_q = [grid[i] for i in idx if i is not None]

    # Which grid point is furthest (in the rule's own relative units) from the FIRST grid point,
    # on the draws the rule refused — i.e. what the refusal is being driven by.
    driver = Counter()
    for c, i in zip(curves, idx):
        if i is not None:
            continue
        base = max(abs(c[0]), 1e-9)
        j = int(np.argmax(np.abs(c[1:] - c[0]) / base)) + 1
        driver[grid[j]] += 1

    # COUNTERFACTUAL, NOT SHIPPED: same curves, tolerance made aware of the compared point's own
    # sampling sd (0.10*|v_i| + 2*sd_iid(q_j)). `range_probe` is untouched.
    def _se_aware(c):
        for i in range(len(c) - 1):
            base = max(abs(c[i]), 1e-9)
            if all(abs(c[j] - c[i]) < 0.10 * base + 2.0 * sd[j] for j in range(i + 1, len(c))):
                return i
        return None

    cf_idx = [_se_aware(c) for c in curves]
    cf_found = sum(1 for i in cf_idx if i is not None)
    cf_correct = sum(1 for i in cf_idx if i == 0)          # grid[0] == 2 == the true i.i.d. range
    fixed_correct = sum(1 for q in returned_q if q == 2)

    return {
        "n_dec": int(n_dec), "n_draws": int(n_draws), "seed": int(seed),
        "q_grid": grid,
        "sd_iid_by_q": {int(q): round(float(s), 5) for q, s in zip(grid, sd)},
        "sd_iid_by_q_se": {int(q): round(float(s), 5) for q, s in zip(grid, sd_se)},
        "plateau_tol": 0.10,
        "max_sd_over_tol": round(float(sd.max() / 0.10), 3),
        "fixed_rule_nan_rate": n_nan / n_draws,
        "fixed_rule_nan_rate_se": _binom_se(n_nan / n_draws, n_draws),
        "fixed_rule_n_finite": n_draws - n_nan,
        "fixed_rule_returned_q_hist": dict(sorted(Counter(returned_q).items())),
        "fixed_rule_median_returned_q": float(np.median(returned_q)) if returned_q else None,
        "fixed_rule_correct_q2_of_finite": f"{fixed_correct}/{n_draws - n_nan}",
        "refusal_driver_hist_q": dict(sorted(driver.items())),
        "refusal_driver_top2_share": (
            (driver[grid[-1]] + driver[grid[-2]]) / n_nan if n_nan else None),
        "counterfactual_se_aware_found_rate": cf_found / n_draws,
        "counterfactual_se_aware_found_rate_se": _binom_se(cf_found / n_draws, n_draws),
        "counterfactual_se_aware_correct_q2_rate": cf_correct / n_draws,
        "counterfactual_se_aware_correct_q2_rate_se": _binom_se(cf_correct / n_draws, n_draws),
    }


def refusal_channels(estimator_name: str, family_name: str, n_dec: int, family_kwargs: dict,
                     n_draws: int = N_DRAWS, seed: int = SEED) -> dict:
    """B: split one cell's single `n_refused` count into its four distinct channels."""
    estimator = rp.ESTIMATORS[estimator_name]
    generator = fam.FAMILIES[family_name]
    rng = np.random.default_rng(seed)
    ch = Counter()
    finite_ranges = []
    t0 = time.time()
    for _ in range(int(n_draws)):
        x = generator(n_dec, rng=rng, **family_kwargs)
        range_est, _info = estimator(x)
        if np.isfinite(range_est):
            finite_ranges.append(float(range_est))
        rule = rp.block_rule(n_dec, range_est)
        if not rule.refused:
            ch["certifiable"] += 1
        elif not np.isfinite(range_est):
            ch["estimator_nan"] += 1
        elif "calibratable ceiling" in rule.reason:
            ch["cap_refusal"] += 1
        elif "non-overlapping blocks" in rule.reason:
            ch["min_blocks_refusal"] += 1
        else:
            ch["calibration_refusal"] += 1
    out = {
        "estimator": estimator_name, "family": family_name, "n_dec": int(n_dec),
        "family_kwargs": family_kwargs, "n_draws": int(n_draws), "seed": int(seed),
        "median_range_est_finite": (float(np.median(finite_ranges)) if finite_ranges else None),
        "n_finite_range": len(finite_ranges),
        "runtime_s": round(time.time() - t0, 2),
    }
    for k in ("certifiable", "estimator_nan", "cap_refusal", "min_blocks_refusal",
              "calibration_refusal"):
        p = ch[k] / n_draws
        out[k] = ch[k]
        out[k + "_rate"] = p
        out[k + "_rate_se"] = _binom_se(p, n_draws)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(_THIS.parent / "READOFF_DIAG.json"))
    ap.add_argument("--n-draws", type=int, default=N_DRAWS)
    ap.add_argument("--n-jobs-calib", type=int, default=8)
    args = ap.parse_args()

    t0 = time.time()
    n_decs = sorted({nd for _f, _kw, nd in rv.CELLS})
    print(f"[901/diag] prewarming calibration: {len(rp.BLOCK_STRATA)} blocks x {n_decs}",
          flush=True)
    rp.prewarm_calibration_table(blocks=rp.BLOCK_STRATA, n_decs=n_decs,
                                 n_jobs=args.n_jobs_calib, n_trials_calib=200,
                                 n_trials_valid=200, n_boot=300)
    print(f"[901/diag] calibration ready in {time.time() - t0:.0f}s "
          f"({len(rp._K_CACHE)} entries)", flush=True)

    rule_rows = []
    for nd in n_decs:
        r = rule_vs_statistic(nd, n_draws=args.n_draws)
        rule_rows.append(r)
        print(f"[901/diag] A n_dec={nd:5d} grid={r['q_grid']} sd={list(r['sd_iid_by_q'].values())} "
              f"nan={r['fixed_rule_nan_rate']:.3f} med_q={r['fixed_rule_median_returned_q']} "
              f"cf_found={r['counterfactual_se_aware_found_rate']:.3f} "
              f"cf_q2={r['counterfactual_se_aware_correct_q2_rate']:.3f}", flush=True)

    chan_rows = []
    for estimator in rp.ESTIMATORS:
        for family, kwargs, n_dec in rv.CELLS:
            row = refusal_channels(estimator, family, n_dec, kwargs, n_draws=args.n_draws)
            chan_rows.append(row)
            print(f"[901/diag] B {estimator:15s} {family:22s} n_dec={n_dec:6d} {kwargs} "
                  f"cert={row['certifiable_rate']:.3f} nan={row['estimator_nan_rate']:.3f} "
                  f"cap={row['cap_refusal_rate']:.3f} minblk={row['min_blocks_refusal_rate']:.3f} "
                  f"calib={row['calibration_refusal_rate']:.3f} "
                  f"med_range={row['median_range_est_finite']} ({row['runtime_s']}s)", flush=True)

    out = {"n_draws": args.n_draws, "seed": SEED, "rule_vs_statistic": rule_rows,
           "refusal_channels": chan_rows, "runtime_s": time.time() - t0}
    Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    print(f"[901/diag] wrote {args.out} ({time.time() - t0:.0f}s total)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
