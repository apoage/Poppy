#!/usr/bin/env python3
"""TASK_897 — MEASURED null coverage of `honest_estimator.block_bootstrap_sharpe_ci`.

WHY THIS EXISTS
---------------
The TASK_897 pre-registered band has three criteria: post-cost Sharpe >= 1.0, PnL > 0, and
**CI lo > 0**. The third one is only meaningful if the interval's coverage is what it claims.
DECISIONS **D49** (2026-08-19) is the standing precedent that in this repo that assumption has
already been measured FALSE once: the 892 block bootstrap, run with a block length equal to the
dependence length, plateaus near 88% two-sided coverage against a 93% target and never reaches
nominal at any attainable sample size.

Rather than assume this module's bootstrap inherits or escapes that defect, this probe MEASURES
it — through the REAL `block_bootstrap_sharpe_ci` code path, never a re-implementation (the same
discipline `experiments/892_action_ev/calibration.py` follows), under a TRUE NULL where the
population Sharpe is exactly 0, in two scenarios:

  * IID          — per-decision returns i.i.d. N(0,1). The optimistic bound: no dependence at all.
  * OVERLAPPING  — each observation is a rolling sum of `dep` i.i.d. innovations, so consecutive
                   observations share up to dep-1 of them. This is the dependence model 892's
                   harness uses, and it is the relevant stress case here: per-decision SEGMENTS
                   are non-overlapping by construction, but consecutive HOLDS are not independent
                   bets (signal persistence and regime structure carry across rebalances), and
                   `compute_honest_metrics` CAPS the block length at
                   `honest_estimator.MAX_BLOCK_DEFAULT` — so dependence longer than that cap is
                   structurally outside what this bootstrap can capture. The `MEASURED_COVERAGE`
                   numbers below were taken when that cap was 20; D55(a) has since raised it (see
                   the UPDATE note at the end of this docstring), and the CURRENT code path's
                   certified coverage lives in `calibration.MEASURED_CALIBRATION_TABLE` +
                   `calibration.MEASURED_BORROWED_PAIRING_COVERAGE`, not here.

Coverage is measured against the NOMINAL 95% the percentile interval claims (2.5/97.5).

REPRODUCE (cheap, CPU-only, seconds):
    trader_env/bin/python experiments/897_corrected_809/ci_coverage_probe.py

The numbers this printed on 2026-08-22 are recorded verbatim in `MEASURED_COVERAGE` below and
quoted in `honest_estimator.CI_UNCALIBRATED_DISCLOSURE`. They are a CHEAP measurement
(400 trials x 400 resamples; binomial MC SE ~1.1-2.5pp near p=0.9), which is ample to establish
the DIRECTION and ROUGH SIZE of the shortfall but is not a certification of any particular value.

TASK_897 D55(a) UPDATE (2026-08-22, same day): `honest_estimator.MAX_BLOCK_DEFAULT` was RAISED
from 20 to 60 after `MEASURED_COVERAGE` below was recorded — `block_for` here reads that live
constant (never a hardcoded 20) so THIS probe cannot drift from the estimator it measures, but
it also means **re-running the reproduce command above today prints DIFFERENT numbers than
`MEASURED_COVERAGE`** at n_dec=500/1000 (block becomes 25/50 instead of 20/20) — that is
expected, not a regression: `MEASURED_COVERAGE` is the PRE-calibration baseline that motivated
raising the cap and building `calibration.py`'s calibrated-width correction in the first place,
kept verbatim as the historical evidence. The CURRENT code path's certified coverage (raised
cap + calibrated k) lives in `calibration.MEASURED_CALIBRATION_TABLE`, not here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import honest_estimator as hest  # noqa: E402

NOMINAL_COVERAGE = 0.95


def simulate_iid(n_dec: int, dep: int, rng: np.random.Generator) -> np.ndarray:
    """Scenario IID — population Sharpe exactly 0, no serial dependence."""
    return rng.normal(0.0, 1.0, size=int(n_dec))


def simulate_overlapping(n_dec: int, dep: int, rng: np.random.Generator) -> np.ndarray:
    """Scenario OVERLAPPING — rolling sum of `dep` i.i.d. innovations (population Sharpe still
    exactly 0, but consecutive observations share up to dep-1 innovations)."""
    n_dec, dep = int(n_dec), int(dep)
    innov = rng.normal(0.0, 1.0, size=n_dec + dep - 1)
    csum = np.concatenate([[0.0], np.cumsum(innov)])
    return csum[dep:dep + n_dec] - csum[0:n_dec]


SCENARIOS = {"iid": simulate_iid, "overlapping": simulate_overlapping}


def block_for(n_dec: int, max_block: int = hest.MAX_BLOCK_DEFAULT) -> int:
    """The block length `compute_honest_metrics` would itself select at this decision count —
    read off the same expression AND the same live cap constant (`hest.MAX_BLOCK_DEFAULT`, D55(a)
    raised it from 20 to 60), so the probe cannot drift from the estimator it measures. The
    `MEASURED_COVERAGE` numbers below were recorded BEFORE the D55(a) cap raise (block cap was 20
    at measurement time) — they remain the motivating evidence for calibrating
    (`calibration.py`), not a live re-check of the raised cap; `test_calibration.py` and
    `calibration.MEASURED_CALIBRATION_TABLE` are what certify the current (cap=60) code path."""
    return max(2, min(int(max_block), int(n_dec) // 20))


def measure_coverage(n_dec: int, scenario: str = "iid", dep: int = 20, n_trials: int = 400,
                     n_boot: int = 400, seed: int = 0) -> dict:
    """Fraction of TRUE-NULL trials whose nominal-95% interval actually contains 0."""
    sim = SCENARIOS[scenario]
    rng = np.random.default_rng(seed)
    block = block_for(n_dec)
    hits = 0
    usable = 0
    for t in range(int(n_trials)):
        r = sim(n_dec, dep, rng)
        lo, hi = hest.block_bootstrap_sharpe_ci(r, ppy=1.0, block=block, n_boot=n_boot, seed=t)
        if not (np.isfinite(lo) and np.isfinite(hi)):
            continue          # a bootstrap that never ran cannot be scored either way
        usable += 1
        hits += int(lo <= 0.0 <= hi)
    return {"n_dec": int(n_dec), "scenario": scenario, "block": block, "dep": int(dep),
            "coverage": (hits / usable) if usable else None, "n_usable": usable,
            "nominal": NOMINAL_COVERAGE}


# Recorded verbatim from the reproduce command above on 2026-08-22 (seed=0, 400 trials,
# n_boot=400, dep=20) UNDER THE THEN-PREVAILING BLOCK CAP OF 20. EVERY cell is BELOW the nominal
# 0.95 the interval advertises; the overlapping-dependence column is catastrophically below it at
# the decision counts a thin cell actually produces. This is the evidence behind
# `honest_estimator.CI_UNCALIBRATED_DISCLOSURE`.
#
# *** WHAT RE-RUNNING THE COMMAND TODAY PRINTS, CELL BY CELL (D57 fix pass, item 4) ***
# The docstring above says this at length ~70 lines up; it is repeated HERE because the
# "recorded verbatim" claim sits on THIS constant and a reader checking it must not have to go
# find the caveat. `block_for` reads the LIVE `honest_estimator.MAX_BLOCK_DEFAULT`, which D55(a)
# raised from 20 to 60 AFTER these numbers were taken, so `block` changes for any cell whose
# n_dec // 20 exceeds 20 — and the coverage changes with it:
#
#   * STILL REPRODUCE (block unaffected by the cap: 2, 5 and 10, all below either cap) —
#     ("iid",40)=0.905, ("iid",100)=0.902, ("iid",200)=0.920 and the LOAD-BEARING
#     ("overlapping",100)=0.530, the number D55/D49 and `CI_UNCALIBRATED_DISCLOSURE` quote.
#     Re-measured 2026-08-22 at the raised cap: identical, at the 3 decimals recorded here.
#     PRECISION: 3dp is a DISPLAY rounding of hits/400 — ("iid",100) reads 0.9025 live and is
#     recorded as 0.902; `round(live, 3) == recorded` is the exact claim, pinned by
#     `test_honest_estimator.py::test_measured_coverage_probe_load_bearing_cells_still_reproduce`.
#   * NO LONGER REPRODUCE (the cap raise moved their block) — ("iid",500) and
#     ("overlapping",500) now run at block 25 instead of 20, ("iid",1000) and
#     ("overlapping",1000) at block 50 instead of 20. Measured at the raised cap they read
#     roughly 0.930 / 0.877 / 0.930 / 0.910 against the 0.935 / 0.865 / 0.948 / 0.868 recorded
#     (the second figure was written as 0.878 and is CORRECTED to 0.877, D57 fix cycle 4 item C4:
#     the ("overlapping",500) cell measures 351/400 = 0.8775 exactly, and the nearest double to
#     0.8775 lies just BELOW it, so the command's `f"{...:.3f}"` renders 0.877 — which is what a
#     reader checking this comment against the output actually sees. Re-run 2026-08-22: the cell
#     reads 0.8775 (400 usable trials) and prints 0.877.)
#     here. That is EXPECTED — a longer block is a different estimator, which is the whole reason
#     the cap was raised — and it is why these four cells are historical evidence only.
# NOTHING here certifies the current code path; `calibration.MEASURED_CALIBRATION_TABLE` +
# `calibration.MEASURED_BORROWED_PAIRING_COVERAGE` do that. The four still-reproducing cells are
# NOT a certification either: every one of them is BELOW nominal, which is the point.
MEASURED_COVERAGE = {
    ("iid", 40): 0.905,
    ("iid", 100): 0.902,
    ("iid", 200): 0.920,
    ("iid", 500): 0.935,
    ("iid", 1000): 0.948,
    ("overlapping", 100): 0.530,
    ("overlapping", 500): 0.865,
    ("overlapping", 1000): 0.868,
}

# The split above, machine-readable so a test can pin it instead of a reader trusting the comment.
# A cell reproduces iff the cap raise did not move its block, i.e. iff n_dec // 20 <= 20.
MEASURED_COVERAGE_STILL_REPRODUCES = (("iid", 40), ("iid", 100), ("iid", 200),
                                      ("overlapping", 100))
MEASURED_COVERAGE_SUPERSEDED_BY_CAP_RAISE = (("iid", 500), ("iid", 1000),
                                             ("overlapping", 500), ("overlapping", 1000))


def main() -> None:
    print(f"nominal two-sided coverage claimed by the interval: {NOMINAL_COVERAGE:.2f}\n")
    for scenario, counts in (("iid", (40, 100, 200, 500, 1000)),
                             ("overlapping", (100, 500, 1000))):
        for n_dec in counts:
            r = measure_coverage(n_dec, scenario=scenario)
            print(f"{r['scenario']:12s} n_dec={r['n_dec']:5d} block={r['block']:2d} "
                  f"coverage={r['coverage']:.3f} (nominal {r['nominal']:.2f}, "
                  f"{r['n_usable']} usable trials)")
    print("\nEvery measured cell is BELOW nominal. The interval is UNCALIBRATED; the TASK_897 "
          "band's 'CI lo > 0' criterion cannot be read off it. See "
          "honest_estimator.CI_UNCALIBRATED_DISCLOSURE and DECISIONS D49.")


if __name__ == "__main__":
    main()
