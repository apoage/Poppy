#!/usr/bin/env python3
"""TASK_897 D55(a) — calibrate `honest_estimator.block_bootstrap_sharpe_ci` before the srv leg.

D49/D55's doctrine, applied here rather than assumed: `ci_coverage_probe.py` MEASURED (not
argued) that this repo's block-bootstrap intervals under-cover when block length is capped
below the data's real dependence length, and that no sample size alone closes the gap — the
remedy is the block length, the interval method, or a calibrated width correction, never more
data. TASK_892 (`experiments/892_action_ev/calibration.py`) shipped a calibrated-width remedy
for its OWN statistic (a delta-of-means, at a FIXED block length of 21, keyed by block COUNT).

THIS MODULE REUSES 892's METHOD, NOT ITS TABLE — D55/D56's explicit instruction, because both
things 892's calibration held fixed have changed here: the statistic (a Sharpe ratio — a
variance-driven ratio statistic, not a mean difference) AND the block-length regime (2..60,
SELECTED FROM the decision count by `honest_estimator.compute_honest_metrics`'s own
`n_dec // 20` rule, not a fixed block=21). Importing 892's `k` values across both changes
at once is exactly the unvalidated estimator reuse that produced the retracted "809 trap" this
whole task exists to correct — so every number below is measured fresh, through the REAL
`honest_estimator.block_bootstrap_sharpe_ci` code path, never a re-implementation, exactly as
892's own `calibration.py` measures its own numbers through `report.block_bootstrap_delta_ci`.

THE METHOD, carried over unchanged in SHAPE from 892 (`k_needed_for_trial` / `gather_k_needed`
/ `measured_coverage_at_k` / `calibrate_block` below are the direct analogs of 892's
`k_needed_for_trial` / `gather_k_needed` / `measured_coverage_at_k` / `calibrate_block_count` —
same closed-form percentile argument, same calibrate-on-one-batch/validate-on-an-INDEPENDENT-
batch discipline, same finite-sample safety margin — applied to a DIFFERENT point estimate
(the Sharpe ratio itself, not a delta) and a DIFFERENT stratification key: the REAL
`(block, n_dec)` PAIRING the cell will run at. (An earlier revision keyed on `n_dec` alone and
let the lookup borrow the nearest-below stratum. That is unsound here because `block_for` is a
STEP function of n_dec that saturates at the cap, so nearest-below-by-n_dec silently handed
block-60 cells a `k` measured at block 50 — see the STRATIFICATION KEY block below, and
`MEASURED_BORROWED_PAIRING_COVERAGE` for the re-measurement at the pairings cells actually use.)

  1. For a fixed trial (point-estimate `sh`, raw `ci_lo`, `ci_hi` from ONE real bootstrap run),
     the k-inflated interval `[sh - k*(sh-ci_lo), sh + k*(ci_hi-sh)]`'s containment of 0 is
     MONOTONIC in k, so the smallest k achieving a target coverage over N trials is exactly the
     target-th percentile of the per-trial "k needed to just barely contain 0" — no grid search.
  2. The calibrated k is chosen on a CALIBRATION batch and RE-MEASURED on an INDEPENDENT
     VALIDATION batch before being reported as "achieves coverage C" (never graded on the same
     draws it was chosen from).
  3. The calibration-batch percentile is taken at `target + calibration_margin`, not `target`
     itself — 892's own measured finite-sample regression toward the true quantile applies here
     too (a percentile from ~300 draws re-graded on an independent ~300-draw batch is noisy near
     the target line by construction); the VALIDATION acceptance test still checks the real
     `target` with no margin.

BLOCK CAP: `honest_estimator.py`'s `compute_honest_metrics(..., max_block=20)` capped block
length at 20, making dependence longer than 20 decisions structurally uncapturable at ANY
n_dec (D55's citation of D49's finding, re-measured on this second estimator by
`ci_coverage_probe.py`: overlapping-dependence coverage 0.530/0.865/0.868 at n_dec=100/500/1000,
never closing on more data). `NEW_MAX_BLOCK` below is the raised cap this module's own
measurement (`measure_block_cap_candidates`) selected — see its printed rationale and
`BLOCK_CAP_SELECTION_RATIONALE`.

WHAT THIS CERTIFICATION DOES **NOT** COVER — read `CERTIFICATION_IS_CONDITIONAL_ON_DEPENDENCE`
before reading any `k` below. Every `k` here was measured at `dep == block`, which is the EASIEST
point of this module's own dependence sweep and NOT (as three earlier strings in this file said)
"the worst case". Held fixed against dependence 8x its block, the shipped k covers 0.52-0.55 —
the magnitude that caused D55 to be raised. The certification is therefore CONDITIONAL on the
data's real dependence being no longer than the block, and `honest_estimator.dependence_length` +
`sweep_gated_maker.apply_ci_calibration`'s refusal (`DEPENDENCE_REFUSAL_MECHANISM`) ATTEMPT to
measure that condition per cell rather than assume it.

TASK_897 D57 (2026-08-22) NARROWS THAT LAST CLAIM: the measurement (tau, the integrated
autocorrelation time) equals the dependence RANGE a block bootstrap actually needs only inside
the triangular-ACF family `simulate_overlapping` generates, and under-reads badly out of family —
see `MEASURED_TAU_OUT_OF_FAMILY_HALTED_CYCLE` / `MEASURED_REAL_DATA_TAU_INERT_HALTED_CYCLE` below
and `DEPENDENCE_REFUSAL_MECHANISM`'s OUT-OF-FAMILY paragraph. On this row's real per-decision
data the guard is EXPECTED NOT TO BITE (every real series reads tau=1-2 against block 60), so a
near-100% certification rate here is NOT evidence the condition holds. TASK_901 (a range-based
dependence probe, validated out-of-family) is what re-earns this conjunct, retroactively, FROM
THE SAME ARTIFACTS.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve()
if str(_THIS.parent) not in sys.path:
    sys.path.insert(0, str(_THIS.parent))

import honest_estimator as hest  # noqa: E402  (the REAL code path — never re-implemented)

TARGET_COVERAGE = 0.93          # same target 892 certified to (SPEC.md [A1]-equivalent bar).

# --- D55: the raised block-length cap, measured (see measure_block_cap_candidates below).
# Owned by `honest_estimator.MAX_BLOCK_DEFAULT` (imported, not duplicated) so the estimator's
# actual cap and the number this module's certification table was measured against can never
# silently drift apart — `test_calibration.py::test_new_max_block_matches_estimator_default`
# pins the equality.
NEW_MAX_BLOCK = hest.MAX_BLOCK_DEFAULT
OLD_MAX_BLOCK = 20              # the D49/D55-cited defect this module raises.
BLOCK_CAP_SELECTION_RATIONALE = (
    "Measured 2026-08-22, both sweeps reproducible verbatim (`python calibration.py "
    "--sweep-caps`; `measure_dependence_mismatch()` below), NOT assumed. "
    "SWEEP 1 (`measure_block_cap_candidates`, defaults: n_trials=150, n_boot=400, seed=42): "
    "candidate caps {20,40,60,80,100} at n_dec in {500,2000,8000}, dependence MATCHED to the "
    "block (dep==block). **THAT IS NOT THE WORST CASE — IT IS THE EASIEST POINT OF THIS "
    "MODULE'S OWN SWEEP, and an earlier revision of this string wrongly called it 'the worst "
    "case D49/892 characterize'.** Matched dependence is the one point where a block bootstrap "
    "has exactly the block length it needs; SWEEP 2 below pushes dependence PAST the block and "
    "coverage falls monotonically. Read this sweep as 'the caps are indistinguishable at the "
    "matched point', never as 'the caps were stressed'. Every candidate cleared or "
    "nearly cleared the 0.93 target (0.907-0.993 across all 15 cells; TWO of the 15 read BELOW "
    "target — cap=80/n_dec=500 at 0.9067 and cap=100/n_dec=8000 at 0.9267 — and BOTH sit at a "
    "cap ABOVE the 60 selected below, i.e. neither is at cap 20 or cap 60. At 150 validation "
    "trials the binomial SE near the bar is ~2.1pp, so 0.9067 is ~1.1 SE under it and 0.9267 "
    "~0.2 SE under it: both are ordinary draws, not demonstrated holes. CORRECTED 2026-08-22 "
    "(D57 fix cycle 4, item C1) — this clause previously said 'ONLY cap=80/n_dec=500 read below "
    "target', which the sweep's own re-run contradicts; the 0.907-0.993 range was and is "
    "correct. Every cell is now pinned cell-by-cell in `MEASURED_BLOCK_CAP_SWEEP1` below, so "
    "this count cannot drift from the measurement again. "
    "READ HONESTLY: this sweep does NOT show any cap failing outright once the calibrated-k "
    "widening is applied — the widening can numerically compensate for a too-short block within "
    "this synthetic dependence model, so 'raise the cap' is not shown to be STRICTLY REQUIRED "
    "for the calibrated interval to hit nominal coverage at these decision counts. "
    "SWEEP 2 (`measure_dependence_mismatch`, n_dec=8000, n_trials=150, n_boot=400, seed=11): "
    "holding n_dec fixed and pushing the REAL dependence to 1x/2x/4x/8x/16x the block, at "
    "cap=20 (block=20) k needed grows 1.38 -> 1.94 -> 2.03 -> 3.37 -> 3.73. **THE COVERAGE "
    "FIGURE THAT USED TO FOLLOW HERE ('coverage held 0.91-0.99 throughout, never failing "
    "outright even at 16x') DESCRIBED A k RE-CALIBRATED AT EACH DEPENDENCE LEVEL — it is what "
    "`measure_dependence_mismatch` reports, because that function re-derives k at every point "
    "before grading it. IT IS NOT THE COVERAGE OF THE k THIS MODULE SHIPS.** With the SHIPPED "
    "k held fixed, coverage collapses in exactly the D49 direction: see "
    "`MEASURED_SHIPPED_K_AT_DEPENDENCE_MISMATCH` below (0.52-0.55 at dep = 8x block). The k "
    "growth above is the honest reading of SWEEP 2: it says how much MORE widening a longer "
    "dependence would need, i.e. how far the shipped k is from sufficient there. At cap=60 "
    "(block=60) the SAME "
    "16x sweep reaches an ABSOLUTE dependence length 3x longer (dep=960 vs dep=320) at a "
    "comparable k (4.22 vs 3.73) and comparable coverage (0.94 vs 0.91 — again, coverage of the "
    "k RE-DERIVED at that dependence, not of the shipped k) — i.e. cap=60 tolerates "
    "3x more absolute dependence before needing the same size of correction, because a longer "
    "block genuinely captures more of a given dependence length structurally, rather than the "
    "calibration papering over all of it with an ever-larger multiplier (the module's own "
    "doctrine: a widened-until-it-can't-fail interval stops being a calibration). Neither sweep "
    "shows k blowing past k_cap=8 in the tested range at either cap — the classic 'uncapturable "
    "at any n' plateau (D49's raw, UNcalibrated finding) is not reproduced here once the k "
    "correction is RE-DERIVED AT THE ACTUAL DEPENDENCE; what raising the cap changes is HOW MUCH "
    "correction is needed for a given real dependence length, not whether target coverage is "
    "reachable at all. THE CATCH, STATED HERE BECAUSE IT IS THE WHOLE CONDITION ON THIS "
    "CERTIFICATION: re-deriving k at the actual dependence requires KNOWING the actual "
    "dependence, and the shipped table does not — every `k` in `MEASURED_CALIBRATION_TABLE` was "
    "measured at dep == block. See `CERTIFICATION_IS_CONDITIONAL_ON_DEPENDENCE`. "
    "SELECTION: 60 is chosen as the smallest cap that (a) is not shown worse than 20 by either "
    "sweep, (b) literally satisfies D55's instruction ('dependence longer than 20 decisions is "
    "capturable') by tripling the block length the estimator can structurally represent, and "
    "(c) never leaves the bootstrap short of non-overlapping blocks anywhere in the certified "
    "range — recomputed 2026-08-22 after a review found the previous wording of this clause "
    "arithmetically FALSE twice over (it claimed 'at least 8 non-overlapping blocks at the "
    "smallest calibrated stratum, n_dec=500 -> 500//60=8', but 500 was not the SMALLEST stratum "
    "— 500 IS a stratum, `MEASURED_CALIBRATION_TABLE[500]` = block 25, k 1.5979, achieved; the "
    "SMALLEST stratum is n_dec=40 — and block_for(500, cap=60) is 25, not 60, giving "
    "500//25 = 20 blocks. (The retraction itself previously read '500 was not a stratum at all', "
    "which is false against the shipped table and is corrected here: a retraction that is itself "
    "wrong, shipping verbatim into the artifact, is worse than the error it retracts.) "
    "THE REAL ARITHMETIC: "
    "because the estimator's own rule is `block = n_dec // 20`, the non-overlapping block count "
    "is PINNED AT 20 everywhere the block sits below the cap — including at the smallest "
    "certified stratum (n_dec=40, block 2 -> 40//2 = 20 blocks) and at the exact point where "
    "any cap first binds (n_dec = 20*cap -> 20 blocks, for cap=60, cap=100, or any other value). "
    "That clause therefore CANNOT distinguish 60 from 100 at the transition — both give 20. "
    "Where the caps do differ is ABOVE the transition, and there the LOWER cap leaves MORE "
    "blocks, not fewer: at n_dec=3000, cap=60 gives block 60 and 50 non-overlapping blocks while "
    "cap=100 gives block 100 and only 30; at n_dec=300000 it is 5000 vs 3000. So the block-count "
    "argument mildly FAVOURS the smaller cap over 100 across the whole certified range, and the "
    "certified minimum block count anywhere in the shipped table is 20. This is a DISCLOSED, "
    "reasoned choice, not a value forced by a sharp measured failure at 20 — the honest caveat "
    "is recorded here rather than a cleaner story than the numbers support."
)

# Below this many decisions, `compute_honest_metrics` itself sets block=min_block=2 regardless
# of the cap (n_dec // 20 < 2), and `MIN_DECISIONS_FOR_SHARPE=2` already refuses anything
# smaller — there is no cap-dependent behaviour to certify under 40 decisions.
MIN_N_DEC_FOR_CALIBRATION = 40

# =============================================================================================
# SWEEP 1, PINNED CELL BY CELL (D57 fix cycle 4, item C1). `BLOCK_CAP_SELECTION_RATIONALE` above
# is the ONE measured block in this module that shipped with no test at all, and it shipped a
# false count: it said "only cap=80/n_dec=500 read below target", while its own advertised
# command reports TWO below-target cells. The narrative is now pinned against a machine-readable
# record, and `test_calibration.py::test_block_cap_sweep1_prose_matches_the_pinned_record`
# derives every figure the prose quotes FROM this dict, so a re-measurement that moves a cell
# turns the prose red instead of leaving it wrong.
#
# THIS IS THE EXACT STDOUT OF THE ADVERTISED COMMAND, field for field:
#   trader_env/bin/python experiments/897_corrected_809/calibration.py --sweep-caps
# i.e. of `measure_block_cap_candidates()` at every default (cap_candidates=(20,40,60,80,100),
# n_dec_probe=(500,2000,8000), n_trials=150, n_boot=400, target=0.93, margin=0.03, seed=42),
# re-run 2026-08-22, MEASURED COST 1m49s single-threaded. The path is bit-deterministic at the
# fixed seed, so a re-run reproduces this BIT-IDENTICALLY and NO Monte-Carlo tolerance applies to
# one. Only `block`, `k` and `validation_coverage` are pinned — those are the three fields the
# command prints, i.e. exactly what a reader can check.
#
# THE RNG IS ONE SHARED STREAM ACROSS THE WHOLE NESTED LOOP (a single
# `np.random.default_rng(42)`), so a cell can only be reproduced together with every cell BEFORE
# it in (cap, n_dec) order. That is why the exact-reproduction test pins the FIRST cap's three
# cells (~19s measured) rather than one interior cell or all 15 (~109s): the leading prefix of the
# stream is the cheapest exactly-reproducible unit there is. The other 12 cells are pinned as a
# record and checked for internal consistency, not re-run in the suite.
MEASURED_BLOCK_CAP_SWEEP1 = {
    20: {500:  {"block": 20, "k": 1.809675794654463,  "validation_coverage": 0.9933333333333333},
         2000: {"block": 20, "k": 1.5537648164535303, "validation_coverage": 0.9866666666666667},
         8000: {"block": 20, "k": 1.5079169512770938, "validation_coverage": 0.9866666666666667}},
    40: {500:  {"block": 25, "k": 1.4989351191419749, "validation_coverage": 0.9533333333333334},
         2000: {"block": 40, "k": 1.383675085594595,  "validation_coverage": 0.96},
         8000: {"block": 40, "k": 1.4386718518296742, "validation_coverage": 0.9733333333333334}},
    60: {500:  {"block": 25, "k": 1.809032961421371,  "validation_coverage": 0.9733333333333334},
         2000: {"block": 60, "k": 1.286444259116037,  "validation_coverage": 0.98},
         8000: {"block": 60, "k": 1.3665555359398756, "validation_coverage": 0.9733333333333334}},
    80: {500:  {"block": 25, "k": 1.3051761012856111, "validation_coverage": 0.9066666666666666},
         2000: {"block": 80, "k": 1.6086270236923725, "validation_coverage": 0.9733333333333334},
         8000: {"block": 80, "k": 1.2258580467962914, "validation_coverage": 0.9533333333333334}},
    100: {500:  {"block": 25,  "k": 1.4928032812838627,
                 "validation_coverage": 0.9733333333333334},
          2000: {"block": 100, "k": 1.466465863433753,
                 "validation_coverage": 0.9466666666666667},
          8000: {"block": 100, "k": 1.2275819106426353,
                 "validation_coverage": 0.9266666666666666}},
}
# The prefix the exact-reproduction test re-runs (the shared-RNG argument above).
BLOCK_CAP_SWEEP1_REPRODUCIBLE_PREFIX_CAPS = (20,)

# =============================================================================================
# THE STRATIFICATION KEY — corrected 2026-08-22 (fix pass, MAJOR 1).
# ---------------------------------------------------------------------------------------------
# A previous revision stratified on n_dec ALONE and let `calibrated_k_for` borrow the
# nearest-below stratum's k. That is unsound here, because the thing that actually drives this
# bootstrap's coverage is the BLOCK LENGTH, and `block_for` is a STEP FUNCTION of n_dec that
# saturates at the cap:
#
#     block_for(n_dec) = max(2, min(NEW_MAX_BLOCK, n_dec // 20))
#
#   * REGIME A (n_dec < 20*NEW_MAX_BLOCK == 1200): block == n_dec // 20, so the block length
#     CHANGES with n_dec and the non-overlapping block COUNT stays ~20 throughout.
#   * REGIME B (n_dec >= 1200): block is pinned at the cap (60) and the block COUNT grows.
#
# Borrowing across that step is what broke: a cell with n_dec=1200-2000 runs at block=60 but a
# nearest-below-by-n_dec lookup handed it the n_dec=1000 / block=50 stratum's k. THE VALUE OF THAT
# k, CORRECTED 2026-08-22 (D57 fix cycle 4, item C4): the defect was DEMONSTRATED at k=1.3413,
# which came from the PRE-RESTRATIFICATION grid, and this comment used to attribute 1.3413 to the
# n_dec=1000 / block=50 stratum — which in the SHIPPED table carries
# k=1.5309881369858398. No shipped stratum has a k rounding to 1.3413 (checked over all 67 rows;
# nothing in the table even falls in [1.33, 1.35]). The nine coverage figures below DO reproduce
# exactly at k=1.3413, so it is the ATTRIBUTION that was wrong, not the measurement: read 1.3413
# as "the k the old lookup handed a block-60 cell at the time this defect was measured", not as a
# value the shipped table serves.
# Re-measured achieved coverage at that REAL (borrowed-k, real-block) pairing — a review found
# 0.930 / 0.9375 / 0.9475 (SEEDS NEVER RECORDED for that triple, and no command was advertised
# for it, so it is quoted as a historical reading and is NOT re-derivable — the same limitation
# `MEASURED_TABLE_SEED_JITTER_SUPERSEDED` carries; D57 item C4). What IS re-derivable is this fix
# pass's independent reproduction through `measured_coverage_at_k`, at named seeds, re-verified
# 2026-08-22 (~36s for all nine cells):
#
#     for s in (0, 1, 2):
#         for nd in (1200, 1350, 1500):
#             measured_coverage_at_k(nd, 1.3413, "overlapping", 200, 400,
#                                    np.random.default_rng(s))     # A FRESH rng PER CELL
#
#     seed 0: 0.9250 / 0.9400 / 0.9500
#     seed 1: 0.9550 / 0.9400 / 0.9350
#     seed 2: 0.9100 / 0.9250 / 0.9200
#
# (Pinned by `test_calibration.py::test_borrowed_across_block_boundary_nine_cell_record_
# reproduces`, which re-runs a named 1-cell subset of that grid exactly — the fresh-rng-per-cell
# rule is what makes a single cell reproducible on its own here.)
#
# — mean ~0.933, with 4 of 9 cells UNDER the 0.93 bar. That is sitting ON the target rather than
# inside the margin the pinned strata advertise. HOW WIDE THAT MARGIN ACTUALLY IS, CORRECTED
# 2026-08-22 (D57 item C4): this sentence used to name "the 0.943-0.977 margin". Measured off the
# shipped table, the 62 `achieved` strata's `validation_coverage_overlapping` spans
# 0.9333-0.9833, and FOUR achieved strata sit BELOW 0.943 (n_dec=1020, 30000 and 100000 at 0.9333;
# n_dec=3000 at 0.9367). The contrast being drawn here is therefore REAL but WEAKER than it was
# stated: the borrowed pairing sits at the bottom of the achieved band, not below it.
# And real sweep cells essentially never
# land exactly on a pinned n_dec, so that borrowing is the TYPICAL case, not an edge case.
#
# THE FIX: the calibration key is the REAL `(block, n_dec)` pairing the cell will use.
# `calibrated_k_for` now matches `block` EXACTLY and only then takes the nearest-below n_dec
# WITHIN that block — so a stratum and the cell that borrows from it always agree on the block
# length. To make that lookup total rather than mostly-empty, the strata are generated FROM
# `block_for`'s own step function (`n_dec_strata_for_grid` below): one stratum at the BOTTOM of
# every reachable regime-A block (n_dec = 20*b, the smallest n_dec whose block is b — i.e. a
# stratum sits exactly at every block-transition point), plus a ladder of n_dec strata inside
# the single saturated regime-B block. Coverage is then RE-MEASURED at borrowed (non-stratum)
# pairings — see `DEFAULT_BORROWED_PROBE_POINTS` / `MEASURED_BORROWED_PAIRING_COVERAGE`.
# =============================================================================================

# n_dec ladder inside REGIME B (block pinned at the cap). Spans the block COUNT from 20
# (n_dec=1200, the transition) to 5000 (n_dec=300000) — 300000 is also the CERTIFIED CEILING:
# above the largest stratum there is no measurement, so `calibrated_k_for` refuses rather than
# extrapolating (see `certified_region_description`).
DEFAULT_CAP_REGIME_N_DEC = (1200, 1500, 2000, 3000, 5000, 10000, 30000, 100000, 300000)


def n_dec_strata_for_grid(max_block: int = NEW_MAX_BLOCK,
                          min_n_dec: int = MIN_N_DEC_FOR_CALIBRATION,
                          cap_regime_n_dec=DEFAULT_CAP_REGIME_N_DEC):
    """Every n_dec stratum the certification needs, DERIVED from `block_for`'s own step function
    rather than hand-picked — so there is a stratum at every block-transition point and no cell
    can ever borrow a k that was measured at a different block length.

    Regime A: one stratum per reachable block b in [2, max_block), placed at n_dec = 20*b, which
    is the SMALLEST n_dec whose block is b (`block_for(20*b) == b`). That stratum therefore sits
    at the bottom of the half-open n_dec range [20*b, 20*b+19] that shares block b, and every
    n_dec in that range borrows from it — a borrow of at most 19 decisions at an IDENTICAL block.
    Regime B: the explicit `cap_regime_n_dec` ladder, all of which have block == max_block.
    """
    max_block, min_n_dec = int(max_block), int(min_n_dec)
    strata = set()
    for b in range(2, max_block):
        nd = 20 * b
        if nd >= min_n_dec:
            strata.add(nd)
    for nd in cap_regime_n_dec:
        strata.add(int(nd))
    return tuple(sorted(strata))


DEFAULT_N_DEC_STRATA = n_dec_strata_for_grid()


def simulate_iid(n_dec, rng):
    return rng.normal(0.0, 1.0, size=int(n_dec))


def simulate_overlapping(n_dec, dep, rng):
    """Rolling sum of `dep` i.i.d. innovations — the SAME dependence model
    `ci_coverage_probe.py` and 892's `calibration.py` both use, so the certification here is
    against the identical synthetic dependence structure the rest of this repo's calibration
    work is judged against (disclosed as a scenario caveat, not eliminated by it — see
    `CALIBRATION_SCENARIO_CAVEAT`)."""
    n_dec, dep = int(n_dec), int(dep)
    innov = rng.normal(0.0, 1.0, size=n_dec + dep - 1)
    csum = np.concatenate([[0.0], np.cumsum(innov)])
    return csum[dep:dep + n_dec] - csum[0:n_dec]


def block_for(n_dec: int, max_block: int = NEW_MAX_BLOCK, min_block: int = 2) -> int:
    """The block length `compute_honest_metrics` would itself select at this decision count and
    cap — read off the identical expression, so this harness cannot drift from the estimator it
    calibrates. Mirrors `honest_estimator.compute_honest_metrics`'s own
    `max(min_block, min(max_block, n_dec // 20))`."""
    return max(int(min_block), min(int(max_block), int(n_dec) // 20))


def k_needed_for_trial(point_estimate: float, ci_lo: float, ci_hi: float) -> float:
    """Smallest k >= 0 such that 0 lies in the k-inflated interval
    `[pe - k*(pe-ci_lo), pe + k*(ci_hi-pe)]`, widened around the POINT ESTIMATE (never around 0).
    Identical closed form to `experiments/892_action_ev/calibration.py::k_needed_for_trial`
    (the method, not a borrowed number) — see that module's docstring for the derivation and the
    fixed k=0-at-boundary bug it documents; re-derived independently here so this module never
    imports from 892's."""
    pe, lo, hi = float(point_estimate), float(ci_lo), float(ci_hi)
    if pe == 0.0:
        return 0.0
    if pe > 0.0:
        d_lo = pe - lo
        return np.inf if d_lo <= 0.0 else pe / d_lo
    d_hi = hi - pe
    return np.inf if d_hi <= 0.0 else -pe / d_hi


def gather_k_needed(n_dec, scenario, n_trials, n_boot, rng, max_block=NEW_MAX_BLOCK, dep=None):
    """Run n_trials independent TRUE-NULL trials (population Sharpe exactly 0) through the REAL
    `honest_estimator.sharpe` + `honest_estimator.block_bootstrap_sharpe_ci` — never a
    re-implementation — and return the array of per-trial k_needed.

    `dep` defaults to the block length itself (dep == block). THAT IS NOT A WORST CASE — an
    earlier revision of this docstring called it "the matched-dependence worst case", and that
    was wrong in the direction that flatters the certification. dep == block is the point where
    the block bootstrap has EXACTLY the block length the dependence requires; it is the EASIEST
    point of this module's own dependence sweep, and coverage falls monotonically as `dep` rises
    above it (`MEASURED_SHIPPED_K_AT_DEPENDENCE_MISMATCH`). Pass a larger `dep` to measure
    dependence LONGER than the block can represent — that is the direction that fails."""
    block = block_for(n_dec, max_block=max_block)
    eff_dep = block if dep is None else int(dep)
    ks = []
    for _ in range(int(n_trials)):
        if scenario == "iid":
            r = simulate_iid(n_dec, rng)
        elif scenario == "overlapping":
            r = simulate_overlapping(n_dec, eff_dep, rng)
        else:
            raise ValueError(f"unknown scenario {scenario!r}")
        sh = hest.sharpe(r, periods_per_year=1.0)
        lo, hi = hest.block_bootstrap_sharpe_ci(
            r, ppy=1.0, block=block, n_boot=n_boot, seed=int(rng.integers(0, 2 ** 31 - 1)))
        if not (np.isfinite(sh) and np.isfinite(lo) and np.isfinite(hi)):
            continue  # a bootstrap/point-estimate that never ran cannot be scored either way
        ks.append(k_needed_for_trial(sh, lo, hi))
    return np.asarray(ks, dtype=np.float64)


def measured_coverage_at_k(n_dec, k, scenario, n_trials, n_boot, rng, max_block=NEW_MAX_BLOCK,
                            dep=None):
    """Independent re-measurement (a FRESH batch, never the one `k` was chosen from): fraction
    of trials whose k-inflated interval contains 0."""
    ks = gather_k_needed(n_dec, scenario, n_trials, n_boot, rng, max_block=max_block, dep=dep)
    if ks.size == 0:
        return None, 0
    return float(np.mean(ks <= float(k))), int(ks.size)


def calibrate_block(n_dec, target=TARGET_COVERAGE, n_trials_calib=300, n_trials_valid=300,
                     n_boot=500, seed=0, k_cap=8.0, calibration_margin=0.03,
                     max_block=NEW_MAX_BLOCK):
    """Full calibrate-then-validate pass for ONE decision-count stratum, in BOTH scenarios (iid,
    overlapping with dep==block).

    **THE CERTIFICATION THIS PRODUCES IS AT dep == block ONLY** — never at longer dependence, and
    dep == block is the easiest, not the worst, point of the dependence sweep (see
    `gather_k_needed` and `CERTIFICATION_IS_CONDITIONAL_ON_DEPENDENCE`). Direct analog of
    892's `calibrate_block_count`; see that function's docstring for why `k_cap` and
    `calibration_margin` exist (an interval widened past `k_cap` is refused as no longer a
    calibration; the margin absorbs percentile-estimation regression between the calibration and
    validation batches without relaxing the acceptance test itself)."""
    rng = np.random.default_rng(seed)
    block = block_for(n_dec, max_block=max_block)
    calib_percentile = min(0.999, target + calibration_margin)
    calib_k = {}
    for scenario in ("iid", "overlapping"):
        ks = gather_k_needed(n_dec, scenario, n_trials_calib, n_boot, rng, max_block=max_block)
        calib_k[scenario] = (
            float(np.percentile(ks, 100.0 * calib_percentile, method="higher"))
            if ks.size else None
        )
    valid_ks = [v for v in calib_k.values() if v is not None]
    if not valid_ks:
        return {"n_dec": int(n_dec), "block": int(block), "k": None, "achieved": False,
                "reason": "no calibration trial produced a bandable raw CI at this n_dec"}
    k = max(valid_ks)
    uncapped_k = k
    capped = k > k_cap
    if capped:
        k = float(k_cap)
    out = {"n_dec": int(n_dec), "block": int(block), "k": float(k),
           "k_uncapped": float(uncapped_k), "capped_at_k_cap": bool(capped),
           "calibration_percentile_k": calib_k}
    for scenario in ("iid", "overlapping"):
        cov, n_valid = measured_coverage_at_k(n_dec, k, scenario, n_trials_valid, n_boot, rng,
                                              max_block=max_block)
        out[f"validation_coverage_{scenario}"] = cov
        out[f"validation_n_{scenario}"] = n_valid
    out["achieved"] = (not capped) and (out["validation_coverage_overlapping"] is not None
                                        and out["validation_coverage_overlapping"] >= target)
    if capped:
        out["reason"] = (f"overlapping-scenario calibration percentile needed k="
                         f"{uncapped_k:.2f} > k_cap={k_cap} — refused as 'not a calibration "
                         f"anymore', reported UNACHIEVED rather than applied")
    elif not out["achieved"]:
        out["reason"] = (f"validation-batch overlapping coverage "
                         f"{out['validation_coverage_overlapping']!r} < target {target} despite "
                         f"a calibration-batch percentile of k={k:.3f} at n_dec={n_dec} "
                         f"(block={block}) — the calibration did not reproduce out-of-sample "
                         f"at this stratum")
    else:
        out["reason"] = None
    return out


def _calibrate_one(args):
    """Top-level (picklable) worker for `build_calibration_table`'s process pool."""
    nd, kwargs = args
    return nd, calibrate_block(nd, **kwargs)


def pool_workers(n_jobs, n_units):
    """How many worker PROCESSES a pool of `n_units` jobs may use, given the operator's
    `--n-jobs`. Returns 1 (meaning: run serially) whenever a pool would be pointless.

    D57 fix cycle 4, item A2 — `--n-jobs` WAS UNVALIDATED UPWARD AND THAT IS A MEASURED FORK
    BOMB, not a theoretical one: `--n-jobs 100000` spawned 7,549 processes and was still
    climbing at 65 seconds. This machine has ~54 GiB and a documented hard-crash history under
    memory pressure ([[workstation-syncflood-crashes]]), and every `--n-jobs` docstring in this
    module tells the operator to pass `<ncpu>` — so `1400` for `14` is the realistic typo, not
    an adversarial input. TWO defences, because they answer different mistakes:
      * a value ABOVE `os.cpu_count()` is REFUSED, naming the cap — oversubscribing this many
        NumPy processes cannot be what anyone meant, and silently clamping a `1400` would hide
        the typo instead of surfacing it;
      * a value at or below the cap is CLAMPED to the job count, so 14 workers are never
        forked for 3 probe points.
    Values <= 1 already degraded safely to the serial path and still do."""
    n = int(n_jobs)
    units = int(n_units)
    cpu = os.cpu_count() or 1
    if n > cpu:
        raise SystemExit(
            f"REFUSAL: --n-jobs {n} exceeds this machine's CPU count ({cpu}). Each worker is a "
            f"full NumPy process, so oversubscribing forks a load this host will not survive "
            f"(measured: --n-jobs 100000 spawned 7,549 processes in 65s). Pass at most "
            f"--n-jobs {cpu}."
        )
    if n <= 1 or units <= 1:
        return 1
    return max(1, min(n, cpu, units))


def stratum_seed(n_dec, seed=0, n_dec_strata=DEFAULT_N_DEC_STRATA):
    """The DETERMINISTIC seed `build_calibration_table` assigns to one stratum: `seed + 1000*i`,
    where `i` is that stratum's index in the SORTED, de-duplicated strata list. Defined once and
    used by both the harness and the tests that reproduce it, so the rule cannot drift into two
    slightly different versions (D57 fix pass, item 1)."""
    strata = sorted(set(int(n) for n in n_dec_strata))
    return int(seed) + 1000 * strata.index(int(n_dec))


def reproduce_strata(n_dec_seq, n_dec_strata=DEFAULT_N_DEC_STRATA, n_jobs=1, **kwargs):
    """Re-run `calibrate_block` for a SUBSET of strata at exactly the seeds the full
    `build_calibration_table` run would have given them — so each returned row is bit-identical to
    the corresponding row of a full run, and hence to `MEASURED_CALIBRATION_TABLE` (except at
    n_dec=620, whose `achieved`/`reason` carry the borrowed-pairing demotion overlay; see the
    table's header). This is what makes an EXACT subset check affordable: a full 67-stratum
    reproduction costs tens of CPU-minutes, one stratum costs ~21-45s.

    Any keyword forwarded to `calibrate_block` must be left at its default for the result to match
    the shipped table — the defaults here are the ones the table was pinned at. `seed` is NOT
    among them: it is derived, and passing one is refused rather than silently ignored."""
    if "seed" in kwargs:
        raise TypeError("reproduce_strata derives each stratum's seed from `stratum_seed`; "
                        "passing `seed` would break the exact-reproduction property it exists for")
    # D57 fix cycle 4, item D1: `n_dec_strata` RE-KEYS THE SEED just as effectively as `seed`
    # does, and it was not refused. `stratum_seed` is `seed + 1000 * strata.index(n_dec)`, so the
    # INDEX — hence the whole strata list — is part of the seed: `stratum_seed(1000)` is 48000
    # against DEFAULT_N_DEC_STRATA but 1000 against `n_dec_strata=(40, 1000)`. Measured, the
    # latter returns validation_coverage_overlapping=1.0 where the shipped table says 0.9333 —
    # a DIFFERENT measurement wearing this function's exact-reproduction name. Refused, not
    # silently honoured; a caller who wants a different stratification wants `build_calibration_
    # table`, which takes `n_dec_strata` legitimately because it is not claiming to reproduce
    # anything.
    if tuple(int(n) for n in n_dec_strata) != tuple(int(n) for n in DEFAULT_N_DEC_STRATA):
        raise TypeError(
            "reproduce_strata reproduces the SHIPPED table, whose seeds are keyed off "
            "DEFAULT_N_DEC_STRATA (stratum_seed = seed + 1000 * index-in-that-list). Passing a "
            "different `n_dec_strata` silently re-keys every seed and measures something else "
            "under this function's name — refused. Use build_calibration_table(n_dec_strata=...) "
            "for a different stratification."
        )
    targets = [int(n) for n in n_dec_seq]
    jobs = [(nd, dict(kwargs, seed=stratum_seed(nd, n_dec_strata=n_dec_strata)))
            for nd in targets]
    workers = pool_workers(n_jobs, len(jobs))
    if workers <= 1:
        return {nd: calibrate_block(nd, **kw) for nd, kw in jobs}
    from concurrent.futures import ProcessPoolExecutor
    out = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for nd, entry in ex.map(_calibrate_one, jobs):
            out[nd] = entry
    return {nd: out[nd] for nd in targets}


def build_calibration_table(n_dec_strata=DEFAULT_N_DEC_STRATA, target=TARGET_COVERAGE,
                            n_trials_calib=300, n_trials_valid=300, n_boot=500, seed=0,
                            k_cap=8.0, calibration_margin=0.03, max_block=NEW_MAX_BLOCK,
                            n_jobs=1):
    """The reproducible harness entry point — one `calibrate_block` call per n_dec stratum.
    `python calibration.py` regenerates it end to end and prints the table
    `MEASURED_CALIBRATION_TABLE` below was pinned from.

    Each stratum gets a DETERMINISTIC seed (`stratum_seed`: `seed + 1000*index` over the sorted
    strata), so `n_jobs > 1` (a process pool, CPU-only) changes only the wall clock, never the
    draws — and the whole harness is BIT-DETERMINISTIC: re-running it at the same arguments
    reproduces `MEASURED_CALIBRATION_TABLE` exactly, field for field. It is pinned as a constant
    because it is EXPENSIVE (see RUNTIME below), not because it drifts. `reproduce_strata` runs a
    named subset at the same seeds when the full sweep is unaffordable (e.g. in a test).

    RUNTIME (D57 fix pass item 5): bare `python calibration.py` is NOT an interactive command —
    it is a single-threaded sweep over 67 strata up to n_dec=300000, each at the default 300+300
    trials x 500-resample bootstraps. Per-stratum cost is NOT constant (measured, isolated,
    single-core: ~21s at n_dec=40-120, 24s at n_dec=500, 34s at n_dec=5000; the largest strata
    cost substantially more) — a full end-to-end run was still too slow to time for this
    disclosure, but 67 strata at the SMALLEST measured per-stratum cost is already >= 23
    CPU-minutes, and the n_dec >= 10000 strata dominate, so read the full sweep as
    tens-of-CPU-minutes and never as something a test can pay (D57 fix pass, item 1: that is why
    the exact-reproduction test pins a named 3-stratum subset via `reproduce_strata`). Use
    `--n-jobs <ncpu>` (never MORE than ncpu — `pool_workers` refuses that, D57 item A2) to
    parallelize across strata (changes wall clock only, never the draws). For a FAST check that
    the harness itself still runs — NOT a reproduction of `MEASURED_CALIBRATION_TABLE`, which
    needs the real trial/boot budget above — a reduced-parameter spot check, verified at ~2.8s:
        python calibration.py --n-dec-strata 100 500 3000 --n-trials-calib 50 \
            --n-trials-valid 50 --n-boot 100"""
    strata = sorted(set(int(n) for n in n_dec_strata))
    jobs = [(nd, dict(target=target, n_trials_calib=n_trials_calib, n_trials_valid=n_trials_valid,
                      n_boot=n_boot, seed=stratum_seed(nd, seed=seed, n_dec_strata=strata),
                      k_cap=k_cap, calibration_margin=calibration_margin, max_block=max_block))
            for nd in strata]
    workers = pool_workers(n_jobs, len(jobs))
    if workers <= 1:
        return {nd: calibrate_block(nd, **kw) for nd, kw in jobs}
    from concurrent.futures import ProcessPoolExecutor
    table = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for nd, entry in ex.map(_calibrate_one, jobs):
            table[nd] = entry
    return {nd: table[nd] for nd in strata}


# ---------------------------------------------------------------------------------------------
# BORROWED-PAIRING VERIFICATION (MAJOR-1 fix, second half): the certification must be true for
# the cells that USE it, not only at the pinned strata. Real cells essentially never land on a
# stratum, so these points are the typical case — each one is a n_dec that is NOT a stratum,
# run at ITS OWN real block, using the k it would actually BORROW via `calibrated_k_for`.
# Regime A points sit at the TOP of a block's n_dec range (20*b + 19 — the largest borrow gap
# available at that block); regime B points sit between the cap-regime ladder rungs.
# Format: (n_dec, n_trials, n_boot) — the two largest points use a reduced budget (disclosed on
# the record itself via `n_trials`/`n_boot`/`validation_n`), because their cost scales with n_dec.
# ---------------------------------------------------------------------------------------------
DEFAULT_BORROWED_PROBE_POINTS = (
    (59, 300, 500),      # block 2,  borrows from stratum 40
    (79, 300, 500),      # block 3,  borrows from stratum 60
    (119, 300, 500),     # block 5,  borrows from stratum 100
    (219, 300, 500),     # block 10, borrows from stratum 200
    (419, 300, 500),     # block 20, borrows from stratum 400
    (619, 300, 500),     # block 30 — its stratum (600) FAILED validation, so this point must
                         #            come back UNCERTIFIED. Kept deliberately: the probe has to
                         #            demonstrate the refusal, not only the happy path.
    (639, 300, 500),     # block 31, borrows from stratum 620
    (919, 300, 500),     # block 45, borrows from stratum 900
    (1199, 300, 500),    # block 59, borrows from stratum 1180 (last regime-A block)
    (1250, 300, 500),    # block 60, borrows from stratum 1200 (JUST above the transition —
                         #           the exact pairing the old n_dec-only lookup got wrong)
    (1400, 300, 500),
    (1800, 300, 500),
    (2500, 300, 500),
    (4000, 300, 500),
    (7000, 300, 500),
    (20000, 300, 500),
    (50000, 200, 300),
    (200000, 200, 300),
)


def _verify_one(args):
    n_dec, n_trials, n_boot, k, src_nd, target, seed, max_block = args
    rng = np.random.default_rng(seed)
    block = block_for(n_dec, max_block=max_block)
    cov_ov, n_ov = measured_coverage_at_k(n_dec, k, "overlapping", n_trials, n_boot, rng,
                                          max_block=max_block)
    cov_iid, n_iid = measured_coverage_at_k(n_dec, k, "iid", n_trials, n_boot, rng,
                                            max_block=max_block)
    return n_dec, {
        "n_dec": int(n_dec), "block": int(block), "borrowed_k": float(k),
        "borrowed_from_n_dec": int(src_nd),
        "coverage_overlapping": cov_ov, "coverage_iid": cov_iid,
        "validation_n_overlapping": n_ov, "validation_n_iid": n_iid,
        "n_trials": int(n_trials), "n_boot": int(n_boot),
        "meets_target": bool(cov_ov is not None and cov_ov >= target),
    }


def verify_borrowed_pairings(table, probe_points=DEFAULT_BORROWED_PROBE_POINTS,
                             target=TARGET_COVERAGE, seed=7, max_block=NEW_MAX_BLOCK, n_jobs=1):
    """Re-measure ACHIEVED coverage at the REAL (block, n_dec) pairings cells will use.

    For each probe n_dec: look the k up exactly the way `sweep_gated_maker.apply_ci_calibration`
    does (`calibrated_k_for`, same-block nearest-below), then run a FRESH true-null batch at that
    n_dec — hence at its OWN real block — and report the fraction of k-inflated intervals that
    contain 0. A probe point whose overlapping coverage falls below `target` means the region
    around it is NOT certified, and the strata covering it must be demoted (or a stratum added)
    rather than the shortfall being absorbed silently.

    RUNTIME (D57 fix pass item 5): `python calibration.py --verify-borrowed` (the advertised
    reproduction command) is NOT quick — measured end-to-end ~8 minutes single-threaded (the
    default `n_jobs=1`) on a shared machine. `--n-jobs <ncpu>` — never MORE than ncpu, which
    `pool_workers` refuses (D57 item A2) — parallelizes across the 18 probe
    points (changes wall clock only, never the draws). There is no reduced-trial-count flag for
    this command: every probe point's (n_trials, n_boot) is baked into
    `DEFAULT_BORROWED_PROBE_POINTS` because these ARE the certified probe points, and running them
    at a smaller budget would not be probing what the artifact actually certifies. A per-point
    progress line is printed to stderr as each probe completes, so a live run is distinguishable
    from a hang."""
    jobs, skipped = [], {}
    for i, (nd, n_trials, n_boot) in enumerate(probe_points):
        k, achieved, src = calibrated_k_for(nd, table, max_block=max_block)
        if not achieved:
            skipped[int(nd)] = {"n_dec": int(nd), "block": block_for(nd, max_block=max_block),
                                "borrowed_k": None, "borrowed_from_n_dec": None,
                                "coverage_overlapping": None, "coverage_iid": None,
                                "validation_n_overlapping": 0, "validation_n_iid": 0,
                                "n_trials": int(n_trials), "n_boot": int(n_boot),
                                "meets_target": False,
                                "reason": "no certified stratum at this block at or below this "
                                          "n_dec — stays UNCALIBRATED by construction"}
            continue
        jobs.append((int(nd), int(n_trials), int(n_boot), float(k), int(src), float(target),
                     seed + 1000 * i, int(max_block)))
    out = dict(skipped)
    if not jobs:
        return out
    # D57 fix pass item 5: this call is the slow leg of `--verify-borrowed` (minutes, all in this
    # loop) and previously printed NOTHING until it finished — indistinguishable from a hang.
    # A per-point progress line to stderr costs nothing (one print per already-expensive probe
    # point) and does not touch stdout (the machine-readable table / --out JSON).
    total = len(jobs)
    workers = pool_workers(n_jobs, total)
    if workers <= 1:
        for i, j in enumerate(jobs, 1):
            nd, rec = _verify_one(j)
            out[nd] = rec
            print(f"[verify-borrowed] {i}/{total} n_dec={nd} done", file=sys.stderr, flush=True)
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for i, (nd, rec) in enumerate(ex.map(_verify_one, jobs), 1):
                out[nd] = rec
                print(f"[verify-borrowed] {i}/{total} n_dec={nd} done", file=sys.stderr,
                      flush=True)
    return {nd: out[nd] for nd in sorted(out)}


DISCLOSURE_FIELDS = ("block", "k", "achieved", "validation_coverage_overlapping",
                    "validation_coverage_iid", "validation_n_overlapping",
                    "validation_n_iid", "reason")


def trim_table_for_disclosure(table):
    return {int(nd): {f: e.get(f) for f in DISCLOSURE_FIELDS} for nd, e in table.items()}


def calibrated_k_for(n_dec, table, max_block: int = NEW_MAX_BLOCK):
    """SAME-BLOCK, nearest-below floor lookup — the MAJOR-1 fix.

    The query cell will run at `block_for(n_dec)`. Only strata that were measured AT THAT SAME
    BLOCK are eligible; among those, take the largest n_dec at or below the query (nearest below,
    not the largest k — k does not move monotonically with n_dec in the measured table, so 'take
    the max k' would silently prefer an unrelated stratum). This is what makes the certification
    true for the cell that consumes it: the stratum and the cell agree on the block length, which
    is the quantity that actually drives coverage, and the residual borrow is a small n_dec gap
    at a FIXED block (at most 19 decisions in regime A; a measured-and-verified gap in regime B —
    see `MEASURED_BORROWED_PAIRING_COVERAGE`).

    A query above the largest certified stratum for its block returns UNCERTIFIED: there is no
    measurement up there, and extrapolating one is precisely the unvalidated estimator reuse this
    module exists to refuse.

    Returns `(k, achieved, source_n_dec)`, or `(None, False, None)` when nothing at this block and
    at-or-below this n_dec was ever certified."""
    n_dec = int(n_dec)
    block = block_for(n_dec, max_block=max_block)
    same_block = [(int(nd), e) for nd, e in table.items()
                  if e.get("achieved") and int(e.get("block", -1)) == block]
    if not same_block:
        return None, False, None
    # How far above its own n_dec a stratum is allowed to reach:
    #   * below the cap, a block owns exactly the n_dec range [20*block, 20*block+19] and the
    #     single stratum at its bottom covers all of it (verified at the range TOP by
    #     `DEFAULT_BORROWED_PROBE_POINTS`), so the reach is that range's top;
    #   * AT the cap the block never changes again, so there is no structural boundary to stop
    #     at — the reach is the LARGEST certified stratum, and above it nothing is certified
    #     (extrapolating past the last measurement is the thing this module refuses to do).
    reach = (20 * block + 19) if block < int(max_block) else max(nd for nd, _ in same_block)
    if n_dec > reach:
        return None, False, None
    candidates = [(nd, e) for nd, e in same_block if nd <= n_dec]
    if not candidates:
        return None, False, None
    nd_src, entry = max(candidates, key=lambda kv: kv[0])
    return float(entry["k"]), True, int(nd_src)


def certified_region_description(table=None, max_block: int = NEW_MAX_BLOCK) -> str:
    """Human-readable statement of exactly which `(block, n_dec)` regions this table certifies
    and which keep the UNCALIBRATED qualifier — computed FROM the table, never asserted, so the
    artifact's claim about its own coverage cannot drift from the numbers."""
    table = MEASURED_CALIBRATION_TABLE if table is None else table
    ok = {int(nd): e for nd, e in table.items() if e.get("achieved")}
    if not ok:
        return "NOTHING is certified — every cell keeps the UNCALIBRATED qualifier."
    by_block = {}
    for nd, e in ok.items():
        by_block.setdefault(int(e["block"]), []).append(nd)
    reg_a = sorted(b for b in by_block if b < max_block)
    lo_a = min(min(by_block[b]) for b in reg_a) if reg_a else None
    hi_a = max(max(by_block[b]) for b in reg_a) if reg_a else None
    cap_nds = sorted(by_block.get(max_block, []))
    # Blocks that HAVE a stratum but whose stratum did not validate — genuine holes, not gaps in
    # the grid. Their whole n_dec range keeps the qualifier.
    failed = sorted(int(e["block"]) for nd, e in table.items() if not e.get("achieved"))
    parts = []
    if reg_a:
        parts.append(
            f"REGIME A (block = n_dec//20, block in [{min(reg_a)}, {max(reg_a)}]): CERTIFIED for "
            f"n_dec in [{lo_a}, {hi_a + 19}] EXCEPT the holes below — one stratum at every "
            f"block-transition point (n_dec = 20*block), so a cell borrows at most 19 decisions "
            f"at an IDENTICAL block.")
    if cap_nds:
        parts.append(
            f"REGIME B (block pinned at the cap {max_block}): CERTIFIED for n_dec in "
            f"[{cap_nds[0]}, {cap_nds[-1]}] via strata {cap_nds}.")
    holes = ", ".join(f"block {b} (n_dec {20 * b}-{20 * b + 19})" for b in failed) or "none"
    parts.append(
        f"UNCALIBRATED (qualifier stands, `ci_advisory=False`, `ci_lo_ADVISORY`/`ci_hi_ADVISORY` "
        f"NaN): "
        f"n_dec < {MIN_N_DEC_FOR_CALIBRATION} (below the smallest stratum — block is pinned at "
        f"min_block=2 there and `MIN_DECISIONS_FOR_SHARPE` already refuses n_dec<2); "
        f"n_dec > {cap_nds[-1] if cap_nds else hi_a} (above the largest certified stratum — no "
        f"measurement exists up there and this module refuses to extrapolate one); and the "
        f"strata that failed EITHER their own validation batch OR the borrowed-pairing check, "
        f"kept as measured rather than re-rolled until they passed: {holes}.")
    return "  ".join(parts)


CALIBRATION_SCENARIO_CAVEAT = (
    "The OVERLAPPING scenario this table certifies against is a SYNTHETIC dependence model (a "
    "rolling sum of i.i.d. innovations) — the same model `ci_coverage_probe.py` and 892's "
    "calibration.py both use, chosen for consistency across this repo's calibration work, not "
    "because it is known to match real per-decision return dependence. Real forward returns "
    "additionally carry fat tails, heteroskedasticity, cross-sectional correlation and regime "
    "structure this model does not. A calibrated k is only as good as the scenario it was "
    "measured against: certifying here relocates the honesty problem from 'coverage is "
    "~53-87%' to 'coverage is nominal UNDER THIS DEPENDENCE MODEL', it does not eliminate it. "
    "Every artifact reporting a calibrated k for this estimator must disclose this caveat "
    "alongside it."
)


# =============================================================================================
# THE CONDITION THIS ENTIRE CERTIFICATION RESTS ON — fix cycle 3, MAJOR 1.
# ---------------------------------------------------------------------------------------------
# Every `k` in `MEASURED_CALIBRATION_TABLE` and every coverage number in
# `MEASURED_BORROWED_PAIRING_COVERAGE` was measured with the simulated dependence set EQUAL to
# the block length (`gather_k_needed(..., dep=None)` -> `dep = block`). Earlier revisions of this
# module called that "the matched-dependence worst case" in three places, including inside
# `BLOCK_CAP_SELECTION_RATIONALE`, which ships VERBATIM into the artifact where a reader checks
# the calibration's legitimacy. It is not the worst case. It is the EASIEST point of this
# module's own sweep, and the numbers below are what the SHIPPED k does past it.
# =============================================================================================
CERTIFICATION_IS_CONDITIONAL_ON_DEPENDENCE = (
    "**THE CALIBRATION IS CERTIFIED AT `dep == block` ONLY, AND THAT IS A CONDITION, NOT A "
    "WORST CASE.** Every `k` in `MEASURED_CALIBRATION_TABLE` was measured against a synthetic "
    "dependence length exactly equal to the block length the estimator would select. That is the "
    "single point at which a block bootstrap has exactly the block it needs — the EASIEST point "
    "of this module's own dependence sweep, not the hardest. WHAT HAPPENS BEYOND IT, MEASURED "
    "WITH THE SHIPPED k HELD FIXED (`MEASURED_SHIPPED_K_AT_DEPENDENCE_MISMATCH`, reproducible "
    "via `python calibration.py --sweep-shipped-k-mismatch`, which reproduces the record "
    "BIT-IDENTICALLY — the path is deterministic at fixed seeds): coverage falls monotonically "
    "and steeply — at n_dec=3000/block=60/k=1.2410 it reads 0.948/0.900 at dep=1x block, "
    "0.868/0.816 at 2x, 0.688/0.680 at 4x, 0.528/0.516 at 8x; at n_dec=1000/block=50/k=1.5310 it "
    "reads 0.968/0.952 at 1x, 0.908/0.872 at 2x, 0.764/0.736 at 4x, 0.528/0.552 at 8x (two "
    "independent seeds each — 20001 and 20105 — 250 trials x 400 resamples, through the real code "
    "path; the two seeds differ by at most 5.2pp on any one cell, ~1.6 SE of a difference of two "
    "250-trial estimates, so read the pair rather than either number alone). Note that even the "
    "CERTIFIED point is only AT the bar within that sampling error: n_dec=3000 at dep=1x block "
    "means 0.924 against the 0.93 target, about half a standard error below it. "
    "**READ THAT 0.924 FOR WHAT IT IS — A TWO-SEED, 250-TRIAL SPOT CHECK WITH THE SHIPPED k HELD "
    "FIXED, NOT A RE-MEASUREMENT OF THE CERTIFICATION.** The certification was validated on "
    "INDEPENDENT 300-trial batches, and re-validated at the BORROWED pairings real cells actually "
    "read, at >= 0.93 (`MEASURED_CALIBRATION_TABLE`, `MEASURED_BORROWED_PAIRING_COVERAGE`). These "
    "are two different measurements at two different budgets, and a reader must not conclude from "
    "the 0.924 that the certification failed its own bar: what it says is that the matched point "
    "sits AT the bar rather than comfortably above it, which is exactly why this leg withholds "
    "the band's third conjunct instead of leaning on it. "
    "**0.52-0.55 is the same magnitude as the 0.530 that caused D55 to be raised in the first "
    "place** — i.e. past its certified dependence the calibrated interval is no better than the "
    "uncalibrated one D55 refused to read a band off. The k GROWTH in "
    "`measure_dependence_mismatch` (1.38 -> 3.73 at cap 20; 4.22 at cap 60/16x) is the same "
    "finding seen from the other side: it is how much MORE widening the longer dependence would "
    "have needed, and the shipped k does not have it. "
    "**NOTHING IN THE 809 LINEAGE EVER MEASURED THE STRATEGY'S REAL PER-DECISION DEPENDENCE, "
    "AND `block = n_dec // 20` IS CHOSEN FROM SAMPLE SIZE, NOT FROM THE DATA** — so 'dependence "
    "== block' would be a self-fulfilling assumption if it were merely assumed. "
    "`honest_estimator.dependence_length` ATTEMPTS to measure each cell's realized per-decision "
    "dependence via tau, the integrated autocorrelation time — but tau equals the dependence's "
    "RANGE (what a block bootstrap actually needs) ONLY inside the triangular-ACF family this "
    "table's own dep==block scenario is drawn from, and UNDER-READS out of family (TASK_897 D57, "
    "2026-08-22 — see `DEPENDENCE_REFUSAL_MECHANISM`'s OUT-OF-FAMILY paragraph): a 4%-of-variance "
    "persistent component reads tau=1, certifies 100%, and covers only 0.805-0.376 "
    "(`MEASURED_TAU_OUT_OF_FAMILY_HALTED_CYCLE`); on THIS row's real per-decision data every "
    "series reads tau=1-2 against block=60 (`MEASURED_REAL_DATA_TAU_INERT_HALTED_CYCLE`), so the "
    "guard is EXPECTED NOT TO BITE HERE and a near-100% certification rate is NOT evidence the "
    "dep<=block condition holds; an empirical containment check on the real long/short H=1 "
    "series (97 disjoint 1000-decision chunks) contained the grand Sharpe only 0.639 of the time "
    "while the guard certified ~100%. `sweep_gated_maker.apply_ci_calibration` still REFUSES to "
    "certify any cell whose measured dependence exceeds its own block, or whose dependence "
    "cannot be measured at all — that refusal remains fail-closed and worth keeping. A refused "
    "cell keeps `ci_advisory=False` and the UNCALIBRATED qualifier; A CELL THAT PASSES THE GUARD "
    "IS ONLY `ci_advisory=True` — its `ci_lo_ADVISORY`/`ci_hi_ADVISORY` are recorded for "
    "**TASK_901** (a range-based dependence probe, validated out-of-family) to re-read "
    "retroactively FROM THE SAME ARTIFACTS, and are NEVER band-readable on this leg. EITHER WAY "
    "the band's 'CI lo > 0' criterion is **UNAVAILABLE-because-uncalibratable** on this leg — not "
    "merely unreadable for refused cells. THAT FALLBACK IS STILL STRICTLY BETTER THAN THE "
    "ALTERNATIVE: an interval labelled 'certified' whose certifying assumption is violated (or "
    "cannot be established out-of-family) is a false statement about coverage that a verdict "
    "would then be read off, while the qualifier is merely a criterion the instrument declines "
    "to supply — the same trade D55 already made once."
)


def measure_shipped_k_at_dependence_mismatch(
        points=((3000, None), (1000, None)), dep_multiples=(1, 2, 4, 8), seeds=(20001, 20105),
        n_trials=250, n_boot=400, table=None):
    """MAJOR 1(a)'s measurement, reproducible: hold the SHIPPED `k` fixed (the one a real cell at
    this n_dec would actually be handed) and re-measure coverage as the REAL dependence is pushed
    to 1x/2x/4x/8x the block. This is the complement of `measure_dependence_mismatch`, which
    RE-DERIVES k at every dependence level and therefore cannot show the shipped interval failing.

    `points` is a sequence of `(n_dec, k)`; `k=None` reads the shipped k out of `table`
    (default `MEASURED_CALIBRATION_TABLE`) so the sweep can never drift from what is shipped.

    DETERMINISM — read this before writing any tolerance against this function's output. Every
    draw comes from `np.random.default_rng(s)` for `s` in `seeds`, freshly constructed inside the
    loop, so this is a PURE function of its arguments: the same arguments return the same numbers
    on every invocation and every machine. Called with no overrides at all — which is exactly
    what the advertised `python calibration.py --sweep-shipped-k-mismatch` does — it returns
    `MEASURED_SHIPPED_K_AT_DEPENDENCE_MISMATCH` below BIT-IDENTICALLY, and
    `test_shipped_k_mismatch_cli_default_reproduces_the_pinned_record_exactly` asserts that with
    NO tolerance. There is no run-to-run Monte Carlo noise here to disclose; the only real
    variability is BETWEEN the two default seeds, and both seeds' numbers are published per cell
    in the record so a reader sees that spread directly instead of being handed an average."""
    table = MEASURED_CALIBRATION_TABLE if table is None else table
    out = {}
    for n_dec, k in points:
        n_dec = int(n_dec)
        k = float(table[n_dec]["k"]) if k is None else float(k)
        block = block_for(n_dec)
        out[n_dec] = {"block": block, "shipped_k": k, "n_trials": int(n_trials),
                      "n_boot": int(n_boot), "by_dep_multiple": {}}
        for mult in dep_multiples:
            covs = []
            for s in seeds:
                rng = np.random.default_rng(int(s))
                cov, _ = measured_coverage_at_k(n_dec, k, "overlapping", n_trials, n_boot, rng,
                                                dep=block * int(mult))
                covs.append(cov)
            out[n_dec]["by_dep_multiple"][int(mult)] = {"dep": block * int(mult),
                                                        "coverage_by_seed": covs}
    return out


# THIS RECORD IS THE EXACT OUTPUT OF `python calibration.py --sweep-shipped-k-mismatch`, i.e. of
# `measure_shipped_k_at_dependence_mismatch()` at its own defaults: seeds (20001, 20105),
# dep multiples 1/2/4/8, 250 trials x 400 resamples per cell, `k` read out of the shipped
# `MEASURED_CALIBRATION_TABLE`. Re-measured 2026-08-22 (D57 fix pass, item 4 — REDONE).
# The path is DETERMINISTIC (every draw from `np.random.default_rng(seed)` at a fixed seed), so
# that command reproduces these numbers BIT-IDENTICALLY, and
# `test_shipped_k_mismatch_cli_default_reproduces_the_pinned_record_exactly` asserts equality
# with NO tolerance. NO MONTE-CARLO TOLERANCE APPLIES TO A RE-RUN AT FIXED SEEDS: there is
# nothing for one to absorb. `k` is asserted equal to the shipped table's k by
# `test_calibration.py`, so this record cannot drift from what is actually served.
#
# THE REAL VARIABILITY IS BETWEEN SEEDS, AND IT IS PUBLISHED RATHER THAN AVERAGED AWAY: the two
# seeds differ by at most 5.2pp on any one cell (n_dec=3000 at dep = 2x block: 0.868 vs 0.816),
# which is ~1.6 standard errors of a difference of two independent 250-trial estimates (per-
# estimate binomial SE ~2.3pp near p=0.84). Read the pair, never one of the numbers alone. Note
# in particular that at the CERTIFIED point (dep = 1x block) n_dec=3000 reads 0.948 / 0.900 —
# a 2-seed mean of 0.924 against the 0.93 bar, about half a standard error below it. The
# certified point is AT the bar within 250-trial sampling error, not comfortably above it.
#
# WHY IT WAS RE-PINNED — and the superseded numbers are kept, not discarded, in
# `MEASURED_SHIPPED_K_AT_DEPENDENCE_MISMATCH_HALTED_CYCLE` below. The record originally carried
# here was measured during the halted cycle with its seed provenance written down only as the
# RANGE "20001..20105"; the specific pair inside that range is not recoverable (seed 20001 does
# reproduce the first value of the n_dec=3000/1x cell, 0.948, but seed 20105 gives 0.900 where
# that cell's second value reads 0.944 — and the n_dec=1000/2x cell holds the SAME two values the
# default pair produces, in the opposite order, which no single fixed pair can do). Two
# earlier passes tried to close the resulting gap by moving the disclosed tolerance (~3pp, then
# ~6pp) and by naming "MC noise" as the cause. Both were wrong: this path has no run-to-run
# noise, and the cause was that the record and the code no longer corresponded. Widening a bar
# until a mismatch fits inside it is the move this program rejects, so the bar was left alone and
# the measurement was redone at a NAMED, published seed pair instead.
#
# THE LOAD-BEARING CONCLUSION DOES NOT MOVE — which is exactly why both records are kept side by
# side. Old and new agree on the finding every citing sentence rests on: with the shipped `k`
# held FIXED, coverage falls monotonically and steeply from 0.92-0.96 at dep = 1x block (2-seed
# means: 0.924 at n_dec=3000, 0.960 at n_dec=1000) to ~0.52 at 8x, and 0.52-0.55 is the same
# magnitude as the 0.530 that raised D55.
# HOW THE RE-MEASUREMENT MOVED THE CELLS, COUNTED RATHER THAN CHARACTERISED (corrected
# 2026-08-22, D57 fix cycle 4 item C4 — this used to read "marginally WORSE at every multiple past
# 1x, never better", which is false against the two records it compares, and the "~0.95 at 1x" in
# the sentence above described the SUPERSEDED record). Of the SIX cells past 1x, comparing 2-seed
# means: WORSE at four (n_dec=3000 at 2x/4x/8x and n_dec=1000 at 8x), UNCHANGED at one
# (n_dec=1000 / 2x — 0.890 both, literally the same two values 0.908 and 0.872 in the opposite
# order, which is also the tell that no single fixed seed pair produced the old record), and
# BETTER at one (n_dec=1000 / 4x — 0.750 new vs 0.702 old). Both 1x cells came in LOWER as well
# (0.924 vs 0.946, 0.960 vs 0.968), so across all EIGHT cells it is six worse, one unchanged, one
# better. What both records agree on is the DIRECTION and MAGNITUDE of the collapse; the per-cell
# drift is 250-trial sampling error between two different seed pairs and it does not run one way.
MEASURED_SHIPPED_K_AT_DEPENDENCE_MISMATCH = {
    3000: {"block": 60, "shipped_k": 1.2409929802023711, "n_trials": 250, "n_boot": 400,
           "by_dep_multiple": {
               1: {"dep": 60, "coverage_by_seed": [0.948, 0.900]},
               2: {"dep": 120, "coverage_by_seed": [0.868, 0.816]},
               4: {"dep": 240, "coverage_by_seed": [0.688, 0.680]},
               8: {"dep": 480, "coverage_by_seed": [0.528, 0.516]}}},
    1000: {"block": 50, "shipped_k": 1.5309881369858398, "n_trials": 250, "n_boot": 400,
           "by_dep_multiple": {
               1: {"dep": 50, "coverage_by_seed": [0.968, 0.952]},
               2: {"dep": 100, "coverage_by_seed": [0.908, 0.872]},
               4: {"dep": 200, "coverage_by_seed": [0.764, 0.736]},
               8: {"dep": 400, "coverage_by_seed": [0.528, 0.552]}}},
}

# THE SUPERSEDED MEASUREMENT, KEPT VISIBLE — provenance-stamped, alongside this module's other
# `*_HALTED_CYCLE` constants, and NOT overwritten out of existence. These were a real run through
# the same code path during the halted cycle (fail-closed panel, 2026-08-22, 250 trials x 400
# resamples, k = the same shipped k); what was lost is only WHICH two seeds inside the recorded
# range 20001..20105 produced them, so they cannot be re-derived on demand and therefore cannot
# serve as the reproducible pin. Per this repo's D39 doctrine — a repair tool that refreshes its
# own pin pins nothing — the old numbers stay next to the new ones so a reader can check for
# themselves that the re-measurement did not move the conclusion, only its provenance.
MEASURED_SHIPPED_K_AT_DEPENDENCE_MISMATCH_HALTED_CYCLE = {
    "provenance": "halted cycle, fail-closed panel cycle 3, 2026-08-22 — same code path and the "
                  "same shipped k, 250 trials x 400 resamples; seed provenance recorded ONLY as "
                  "the range 20001..20105 and NOT recoverable (seed 20001 reproduces the first "
                  "value of the n_dec=3000/1x cell, seed 20105 does not reproduce its second, "
                  "and the n_dec=1000/2x cell is recorded with the two seeds in the opposite "
                  "order), so this record is quoted as a historical measurement and is NOT "
                  "re-derivable. SUPERSEDED as the reproducible pin by "
                  "MEASURED_SHIPPED_K_AT_DEPENDENCE_MISMATCH above (D57 fix pass, item 4), which "
                  "IS the exact output of `python calibration.py --sweep-shipped-k-mismatch`. "
                  "The two records agree on the finding: 0.92-0.97 at dep = 1x block (2-seed "
                  "means — THIS record 0.946/0.968, the reproducible pin 0.924/0.960) collapsing "
                  "to 0.52-0.56 at 8x, the magnitude that raised D55. (The '~0.95 at 1x' this "
                  "sentence used to quote described THIS record only and did not hold for the "
                  "pin it claims to agree with — corrected 2026-08-22, D57 item C4.)",
    "record": {
        3000: {"block": 60, "shipped_k": 1.2409929802023711, "n_trials": 250, "n_boot": 400,
               "by_dep_multiple": {
                   1: {"dep": 60, "coverage_by_seed": [0.948, 0.944]},
                   2: {"dep": 120, "coverage_by_seed": [0.844, 0.852]},
                   4: {"dep": 240, "coverage_by_seed": [0.692, 0.700]},
                   8: {"dep": 480, "coverage_by_seed": [0.556, 0.516]}}},
        1000: {"block": 50, "shipped_k": 1.5309881369858398, "n_trials": 250, "n_boot": 400,
               "by_dep_multiple": {
                   1: {"dep": 50, "coverage_by_seed": [0.968, 0.968]},
                   2: {"dep": 100, "coverage_by_seed": [0.872, 0.908]},
                   4: {"dep": 200, "coverage_by_seed": [0.724, 0.680]},
                   8: {"dep": 400, "coverage_by_seed": [0.556, 0.552]}}},
    },
}


def shipped_k_mismatch_rows(record=None):
    """Flatten `MEASURED_SHIPPED_K_AT_DEPENDENCE_MISMATCH` into artifact table rows — rendered
    FROM the measurement so the artifact's numbers cannot drift from the record."""
    record = MEASURED_SHIPPED_K_AT_DEPENDENCE_MISMATCH if record is None else record
    rows = []
    for n_dec in sorted(record):
        e = record[n_dec]
        for mult in sorted(e["by_dep_multiple"]):
            d = e["by_dep_multiple"][mult]
            rows.append({"n_dec": n_dec, "block": e["block"], "shipped_k": e["shipped_k"],
                         "dep_multiple": mult, "dep": d["dep"],
                         "coverage_by_seed": list(d["coverage_by_seed"]),
                         "n_trials": e["n_trials"], "n_boot": e["n_boot"]})
    return rows


# =============================================================================================
# THE REFUSAL — fix cycle 3, MAJOR 1(b). The wording above is only half a close; this is the
# other half. `honest_estimator.dependence_length` measures each cell's realized per-decision
# dependence and `sweep_gated_maker.apply_ci_calibration` withholds certification where it
# exceeds the cell's own block, or cannot be measured.
# =============================================================================================
DEPENDENCE_REFUSAL_MECHANISM = (
    "MECHANISM. For every cell, `honest_estimator.compute_honest_metrics` measures the REALIZED "
    "dependence of that cell's OWN per-decision return series and reports it as "
    "`measured_dependence_len`, alongside `block_used` (the block length its CI was actually "
    "bootstrapped at). The estimate is the integrated autocorrelation time "
    "tau = 1 + 2*sum|rho(h)|, truncated at the first lag whose sample autocorrelation falls "
    "inside the white-noise band 2/sqrt(n). tau is a LINEAR-ACF statistic: it is in the SAME "
    "units as `block` and equals the dependence's RANGE ONLY inside the exact triangular-ACF "
    "family this table is certified against (a rolling sum of `dep` i.i.d. innovations, "
    "`simulate_overlapping`, where the population autocorrelation is triangular and tau equals "
    "`dep` exactly) — INSIDE that family only, `tau <= block` reads 'this cell's dependence is no "
    "longer than the dependence its k was certified at'. See the OUT-OF-FAMILY paragraph below "
    "for why that reading does not generalise. "
    "THE RULE: `sweep_gated_maker.apply_ci_calibration` certifies a cell ONLY IF a certified "
    "same-block stratum exists AND `measured_dependence_len` is finite AND "
    "`measured_dependence_len <= block_used`. Otherwise `ci_advisory=False`, "
    "`ci_lo_ADVISORY`/`ci_hi_ADVISORY` are NaN, the UNCALIBRATED qualifier stands, and 'CI lo > 0' "
    "remains unreadable for that cell — the same fallback D55 already accepted. A cell that DOES "
    "pass this rule is only `ci_advisory=True`: per D57 below, that is STILL never band-readable "
    "on this leg, strictly better than certifying an interval whose certifying assumption is "
    "violated OR cannot be established out-of-family. "
    "FAIL-CLOSED, IN THREE PLACES: (1) `dependence_length` returns NaN — never a number — when "
    "the series is shorter than `MIN_DECISIONS_FOR_DEPENDENCE`, when it is degenerate "
    "(zero-variance), or when its autocorrelation never returns to the white-noise band inside "
    "the probed lag window (dependence too long to resolve is UNKNOWN dependence); every NaN "
    "refuses. (2) `apply_ci_calibration`'s dependence argument DEFAULTS TO None, and None "
    "refuses — a caller that forgets to pass the measurement gets no certification rather than a "
    "silent pass. (3) The comparison is against `block_used` reported by the estimator itself, "
    "not a re-derived block, so the two cannot drift. "
    "OPERATING CHARACTERISTIC, MEASURED IN-FAMILY (`MEASURED_DEPENDENCE_REFUSAL_OC`, reproducible "
    "via `python calibration.py --sweep-dependence-refusal`, all draws from "
    "`simulate_overlapping` — the triangular-ACF family, NOT a general statement): the guard is "
    "CONSERVATIVE at the certified point and near-total past it, IN THAT FAMILY. On i.i.d. series "
    "it refuses ~0% (tau reads 1) so it does not vacuously refuse everything; at dep == block — "
    "the certified point, where the SHIPPED k's stress-sweep coverage is 0.90-0.97 per seed "
    "(2-seed means 0.924 at n_dec=3000 and 0.960 at n_dec=1000, i.e. AT the 0.93 bar, not "
    "comfortably above it; that sweep is a 250-trial spot check, NOT the certification's own "
    "300-trial validation, which cleared >= 0.93) — it still refuses a minority of draws "
    "because tau is noisy; at dep >= 2x block — where the shipped k's coverage is 0.82-0.91 and "
    "falling — it refuses essentially always, IN THAT FAMILY. The false-refusal rate at the "
    "matched point is the price of the fail-closed direction and is disclosed rather than tuned "
    "away. "
    "OUT OF FAMILY, THIS OPERATING CHARACTERISTIC DOES NOT HOLD (TASK_897 D57, 2026-08-22): tau "
    "conflates dependence RANGE with AMPLITUDE, so a low-amplitude, long-range persistent "
    "component under-reads it badly. Measured on a synthetic series whose persistent component "
    "is only 4% of variance (`simulate_persistent_component`, weight=0.2, phi=0.99 — population "
    "tau 8.6, genuinely under block 60): the guard reads tau=1, certifies 100% of draws, and the "
    "achieved coverage is 0.805; at phi=0.999 coverage falls to 0.376 — BELOW the 0.530 that "
    "raised D55 (`MEASURED_TAU_OUT_OF_FAMILY_HALTED_CYCLE`; direction reproduced at reduced "
    "budget by `measure_tau_out_of_family_mismatch`). ON THIS ROW'S REAL PER-DECISION DATA THE "
    "GUARD IS EXPECTED NOT TO BITE: every measured series (7 majors, OOS 2024-11 -> 2025-04, H in "
    "{1,5,20}, long-only and long-short) reads tau=1-2 against block=60 "
    "(`MEASURED_REAL_DATA_TAU_INERT_HALTED_CYCLE` — real-data, provenance-stamped, NOT re-run in "
    "this loop) — so a ~100% certification rate on this leg's real cells is NOT evidence the "
    "dep<=block condition holds, it is the guard measured INERT exactly where it will be used. "
    "An empirical containment check on the real long/short H=1 series (97 disjoint 1000-decision "
    "chunks) contained the grand Sharpe only 0.639 of the time while the guard certified ~100%. "
    "**TASK_901** (a range-based dependence probe — first-lag-in-band / variance-ratio / "
    "spectral-density-at-zero, validated out-of-family) is what upgrades this conjunct, "
    "retroactively, FROM THE SAME ARTIFACTS."
)


# =============================================================================================
# TASK_897 D57 (2026-08-22) — THE TAU GUARD IS OUT-OF-FAMILY, MEASURED. `dependence_length`'s
# tau (the integrated autocorrelation time) equals the dependence RANGE a block bootstrap
# actually needs only inside the exact triangular-ACF family `simulate_overlapping` generates.
# tau CONFLATES range with AMPLITUDE, so a low-amplitude but long-range persistent component
# under-reads it. `simulate_persistent_component` below is that out-of-family generator;
# `measure_tau_out_of_family_mismatch` is the FAST (seconds, CPU-only) reproducible probe —
# it reproduces the SIGN/DIRECTION of the halted cycle's finding, not its exact figures (that
# would need the same trial budget/seed the halted cycle used, which is not on record). The
# halted cycle's own numbers are pinned verbatim, with provenance, in
# `MEASURED_TAU_OUT_OF_FAMILY_HALTED_CYCLE` below. `MEASURED_REAL_DATA_TAU_INERT_HALTED_CYCLE`
# is the REAL-data half of the same finding — a measurement against `trading_data.db`, staged on
# the workstation during the halted cycle, NOT reproduced here: expensive real-data validation is
# scheduled, never re-run inside the factory loop (project invariant), and this row's scope is
# narrowing the claim, not re-measuring real data.
# =============================================================================================

def simulate_persistent_component(n_dec, phi, weight=0.2, rng=None):
    """An OUT-OF-FAMILY synthetic series, deliberately NOT `simulate_overlapping`: `weight` of
    the variance is a slow, unit-variance AR(1) with persistence `phi` (the RANGE — the
    autocorrelation never hits exactly zero, unlike a finite rolling sum); the rest is i.i.d.
    noise. Both components are constructed to unit variance, so
        Var(x) = weight**2 * Var(AR1) + (1 - weight**2) * Var(iid) = weight**2 + (1 - weight**2) = 1
    regardless of `phi`, and `weight**2` reads directly as "fraction of variance that is
    persistent" (weight=0.2 -> 4%). This is the shape `dependence_length`'s white-noise-band
    truncation under-reads: the sample ACF drops inside the band almost immediately (low
    amplitude at every lag) even though the series is genuinely persistent arbitrarily far out
    (high RANGE) — the opposite failure mode from `simulate_overlapping`, whose ACF is exactly
    zero past `dep` by construction."""
    rng = np.random.default_rng(0) if rng is None else rng
    n_dec = int(n_dec)
    phi = float(phi)
    innov_sd = math.sqrt(max(1e-12, 1.0 - phi * phi))
    z = np.empty(n_dec, dtype=np.float64)
    z[0] = rng.normal(0.0, 1.0)
    innov = rng.normal(0.0, innov_sd, size=n_dec)
    for t in range(1, n_dec):
        z[t] = phi * z[t - 1] + innov[t]
    e = rng.normal(0.0, math.sqrt(max(0.0, 1.0 - weight * weight)), size=n_dec)
    return float(weight) * z + e


def measure_tau_out_of_family_mismatch(n_dec=3000, phis=(0.99, 0.999), weight=0.2,
                                       n_trials=60, n_boot=150, seed=97, table=None):
    """FAST reproducible probe for D57's finding (`python calibration.py
    --sweep-tau-out-of-family`; seconds, CPU-only): does the shipped tau guard's near-100%
    certification rate on a low-amplitude, long-range persistent series correspond to real
    coverage? At THIS trial count it reproduces the SIGN/DIRECTION only — NOT a bit-identical
    match to `MEASURED_TAU_OUT_OF_FAMILY_HALTED_CYCLE` (a much larger, unreproduced-here budget).
    Direction asserted by `test_calibration.py`: tau reads near 1 (the guard 'certifies') while
    achieved coverage is materially below target, and gets WORSE — not better — as phi -> 1
    (more persistence, same 4% variance share)."""
    table = MEASURED_CALIBRATION_TABLE if table is None else table
    n_dec = int(n_dec)
    block = block_for(n_dec)
    k = float(table[n_dec]["k"])
    out = {"n_dec": n_dec, "block": block, "weight": float(weight), "k": k,
          "n_trials": int(n_trials), "n_boot": int(n_boot), "by_phi": {}}
    for phi in phis:
        rng = np.random.default_rng(int(seed) + int(round(phi * 100000)))
        taus, hits, usable = [], 0, 0
        for _ in range(int(n_trials)):
            r = simulate_persistent_component(n_dec, phi, weight, rng)
            taus.append(hest.dependence_length(r))
            sh = hest.sharpe(r, periods_per_year=1.0)
            lo, hi = hest.block_bootstrap_sharpe_ci(
                r, ppy=1.0, block=block, n_boot=n_boot, seed=int(rng.integers(0, 2 ** 31 - 1)))
            if not (np.isfinite(sh) and np.isfinite(lo) and np.isfinite(hi)):
                continue
            usable += 1
            clo = sh - k * (sh - lo)
            chi = sh + k * (hi - sh)
            hits += int(clo <= 0.0 <= chi)
        t = np.asarray(taus, dtype=np.float64)
        finite = t[np.isfinite(t)]
        out["by_phi"][float(phi)] = {
            "median_tau": (float(np.median(finite)) if finite.size else None),
            "certify_rate": float(np.mean(np.isfinite(t) & (t <= block))),
            "coverage": (hits / usable) if usable else None, "n_usable": usable,
        }
    return out


# Pinned, PROVENANCE-STAMPED result of the halted cycle's fail-closed panel (fix cycle 3,
# 2026-08-22) — a SYNTHETIC AR(1) mismatch measurement (not real data), at a trial/boot budget
# not on record and therefore NOT bit-identically reproduced by `measure_tau_out_of_family_
# mismatch` above (that function reproduces the DIRECTION at a fast, disclosed budget instead).
# Method: series = weight*standardized-AR(1)(phi) + iid noise, unit total variance, so the
# persistent component is exactly `weight**2` of variance (weight=0.2 -> 4%); measured at
# n_dec=3000 (block=60, the shipped k for that stratum).
MEASURED_TAU_OUT_OF_FAMILY_HALTED_CYCLE = {
    "provenance": "halted cycle, fail-closed panel cycle 3, 2026-08-22 — SYNTHETIC (not real "
                  "data); exact trial/boot budget not recorded by the halt summary, so this is "
                  "quoted as a historical measurement, not re-run bit-identically here",
    "n_dec": 3000, "block": 60, "weight": 0.2,
    "by_phi": {
        0.99: {"population_tau": 8.6, "median_tau": 1.0, "certify_rate": 1.0, "coverage": 0.805},
        0.999: {"median_tau": 1.0, "certify_rate": 1.0, "coverage": 0.376},
    },
}

# Pinned, PROVENANCE-STAMPED result of the halted cycle's REAL-data measurement (real
# `trading_data.db`, workstation, 2026-08-22) — NOT reproduced in this loop (expensive real-data
# validation is scheduled, never polled inside the factory loop, per the project invariants).
# Reproduction recipe: 7 majors, OOS window 2024-11 -> 2025-04, per-decision gated_maker series
# at H in {1,5,20}, long_only and long_short, through THIS module's `honest_estimator.
# dependence_length` against block=60; separately, the real long/short H=1 series split into 97
# disjoint 1000-decision chunks, each chunk's own bootstrap CI checked for containment of the
# GRAND (whole-series) Sharpe.
MEASURED_REAL_DATA_TAU_INERT_HALTED_CYCLE = {
    "provenance": "halted cycle, fail-closed panel cycle 3, 2026-08-22, real minute-table data "
                  "(trading_data.db, workstation) — read-only, NOT re-run in this loop",
    "series_measured": "7 majors, per-decision gated_maker series, H in {1,5,20}, long_only and "
                       "long_short, OOS 2024-11 -> 2025-04",
    "block": 60,
    "measured_dependence_len_range": (1, 2),         # EVERY series, every H, both variants
    "guard_certification_rate": 1.0,                  # ~100% — tau<=block on every series measured
    "empirical_containment_check": {
        "series": "real long_short H=1, OOS 2024-11 -> 2025-04",
        "n_chunks": 97, "chunk_size_decisions": 1000,
        "contains_grand_sharpe_fraction": 0.639,
        "guard_certified_fraction_of_these_cells": 1.0,
    },
}


def measure_dependence_refusal_operating_characteristic(
        points=((3000, 60), (1000, 50), (400, 20), (100, 5), (40, 2)),
        dep_multiples=(0, 1, 2, 4, 8), n_trials=400, seed=5):
    """Measure what the refusal guard actually does, in BOTH directions: how often it refuses a
    series whose dependence really is <= block (a false refusal — conservative, the safe error)
    and how often it refuses one whose dependence exceeds the block (a true refusal — the whole
    point). `dep_multiple == 0` means the i.i.d. scenario.

    Drives `honest_estimator.dependence_length` directly — the same function the sweep calls."""
    out = {}
    for n_dec, block in points:
        n_dec, block = int(n_dec), int(block)
        assert block == block_for(n_dec), (n_dec, block, block_for(n_dec))
        out[n_dec] = {"block": block, "n_trials": int(n_trials), "by_dep_multiple": {}}
        for mult in dep_multiples:
            rng = np.random.default_rng(seed + 17 * int(mult))
            taus = []
            for _ in range(int(n_trials)):
                r = (simulate_iid(n_dec, rng) if int(mult) == 0
                     else simulate_overlapping(n_dec, block * int(mult), rng))
                taus.append(hest.dependence_length(r))
            t = np.asarray(taus, dtype=np.float64)
            refused = ~np.isfinite(t) | (t > block)
            finite = t[np.isfinite(t)]
            out[n_dec]["by_dep_multiple"][int(mult)] = {
                "dep": ("iid" if int(mult) == 0 else block * int(mult)),
                "refusal_rate": float(np.mean(refused)),
                "unmeasurable_rate": float(np.mean(~np.isfinite(t))),
                "median_tau": (float(np.median(finite)) if finite.size else None),
            }
    return out


# The pinned result of the run above (2026-08-22, defaults: 400 trials/point, seed=5), i.e. of
# `python calibration.py --sweep-dependence-refusal`, which calls it with NO overrides.
# `refusal` = fraction of draws the guard refuses (tau unmeasurable OR tau > block);
# `unmeasurable` = the sub-fraction that refused because tau could not be estimated at all.
#
# PRECISION, DISCLOSED (D57 fix pass, item 3): every rate below is ROUNDED TO 3 DECIMALS. The
# path is bit-deterministic at the recorded seed (400 trials means the raw rates are multiples of
# 1/400 = 0.0025), so 12 of the 25 cells' live values carry a 4th decimal this record does not
# print — e.g. n_dec=40 / dep=1x reads 0.3675 live and is recorded as 0.367, and n_dec=3000 /
# dep=8x reads 0.9475 live and is recorded as 0.948. That is a DISPLAY rounding, not a tolerance
# and not MC noise: `round(live, 3) == recorded` holds for all 25 cells, and
# `test_dependence_refusal_oc_record_reproduces_exactly_at_the_recorded_3dp` pins exactly that
# (whole record, ~1.4s) rather than leaving the agreement assumed. `median_tau`, `dep`, `block`
# and `n_trials` are exact.
# READ IT AS: not vacuous (0.00 refusal on i.i.d. — the guard does not simply refuse
# everything), CONSERVATIVE at the certified point (0.27-0.42 false refusals at dep == block —
# "false" because dep <= block there, NOT because the shipped k has margin at that point: the
# shipped-k stress sweep puts it AT the 0.93 bar there, 2-seed means 0.924 / 0.960, and that
# sweep is a 250-trial spot check rather than a re-run of the certification's own 300-trial
# validation, which cleared >= 0.93), and near-total exactly where the shipped k breaks
# (>= 0.94 at dep >= 2x block, where measured coverage is 0.82-0.91 and falling).
MEASURED_DEPENDENCE_REFUSAL_OC = {
    3000: {"block": 60, "n_trials": 400, "by_dep_multiple": {
        0: {"dep": "iid", "refusal_rate": 0.000, "unmeasurable_rate": 0.000, "median_tau": 1.0},
        1: {"dep": 60, "refusal_rate": 0.422, "unmeasurable_rate": 0.025, "median_tau": 59.0},
        2: {"dep": 120, "refusal_rate": 1.000, "unmeasurable_rate": 0.110, "median_tau": 109.5},
        4: {"dep": 240, "refusal_rate": 1.000, "unmeasurable_rate": 0.287, "median_tau": 196.0},
        8: {"dep": 480, "refusal_rate": 1.000, "unmeasurable_rate": 0.948, "median_tau": 207.0}}},
    1000: {"block": 50, "n_trials": 400, "by_dep_multiple": {
        0: {"dep": "iid", "refusal_rate": 0.000, "unmeasurable_rate": 0.000, "median_tau": 1.0},
        1: {"dep": 50, "refusal_rate": 0.318, "unmeasurable_rate": 0.000, "median_tau": 46.0},
        2: {"dep": 100, "refusal_rate": 0.980, "unmeasurable_rate": 0.020, "median_tau": 82.0},
        4: {"dep": 200, "refusal_rate": 0.995, "unmeasurable_rate": 0.048, "median_tau": 136.0},
        8: {"dep": 400, "refusal_rate": 0.995, "unmeasurable_rate": 0.297, "median_tau": 166.0}}},
    400: {"block": 20, "n_trials": 400, "by_dep_multiple": {
        0: {"dep": "iid", "refusal_rate": 0.000, "unmeasurable_rate": 0.000, "median_tau": 1.0},
        1: {"dep": 20, "refusal_rate": 0.287, "unmeasurable_rate": 0.000, "median_tau": 18.0},
        2: {"dep": 40, "refusal_rate": 0.990, "unmeasurable_rate": 0.000, "median_tau": 34.0},
        4: {"dep": 80, "refusal_rate": 0.978, "unmeasurable_rate": 0.000, "median_tau": 57.0},
        8: {"dep": 160, "refusal_rate": 0.993, "unmeasurable_rate": 0.000, "median_tau": 79.0}}},
    100: {"block": 5, "n_trials": 400, "by_dep_multiple": {
        0: {"dep": "iid", "refusal_rate": 0.000, "unmeasurable_rate": 0.000, "median_tau": 1.0},
        1: {"dep": 5, "refusal_rate": 0.270, "unmeasurable_rate": 0.000, "median_tau": 5.0},
        2: {"dep": 10, "refusal_rate": 0.970, "unmeasurable_rate": 0.000, "median_tau": 9.0},
        4: {"dep": 20, "refusal_rate": 0.995, "unmeasurable_rate": 0.000, "median_tau": 14.0},
        8: {"dep": 40, "refusal_rate": 0.995, "unmeasurable_rate": 0.000, "median_tau": 19.0}}},
    40: {"block": 2, "n_trials": 400, "by_dep_multiple": {
        0: {"dep": "iid", "refusal_rate": 0.000, "unmeasurable_rate": 0.000, "median_tau": 1.0},
        1: {"dep": 2, "refusal_rate": 0.367, "unmeasurable_rate": 0.000, "median_tau": 2.0},
        2: {"dep": 4, "refusal_rate": 0.945, "unmeasurable_rate": 0.000, "median_tau": 4.0},
        4: {"dep": 8, "refusal_rate": 0.973, "unmeasurable_rate": 0.000, "median_tau": 5.0},
        8: {"dep": 16, "refusal_rate": 0.993, "unmeasurable_rate": 0.000, "median_tau": 7.0}}},
}


# TASK_902 item 8 — MEASURED_DEPENDENCE_REFUSAL_OC discloses the refusal-rate operating
# characteristic at dependence multiples {0, 1, 2, 4, 8}x the block ONLY. Factory run 40 measured
# an intermediate point NOT in that grid: at 1.5x the guard certifies (does not refuse) 8-13% of
# draws at 0.905 coverage — a materially worse operating point than the 1x neighbour shown above
# (0.270-0.422 refusal, i.e. 58-73% still certify, but AT the calibration's own dep==block anchor)
# and worse than what a reader interpolating linearly between the 1x and 2x rows (0.270-0.422 and
# 0.945-1.000 refusal respectively) would guess. This constant is the disclosed GAP, not a new
# table row: TASK_902's own discipline is that this row changes claims and rendering, never adds
# new measured numbers, so the 1.5x point is named here rather than folded into
# `MEASURED_DEPENDENCE_REFUSAL_OC` as if it had been measured under the same protocol. A reader
# must not assume the grid interpolates linearly, or monotonically in refusal rate, between its
# disclosed multiples.
DEPENDENCE_REFUSAL_OC_GRID_GAP = (
    "MEASURED_DEPENDENCE_REFUSAL_OC discloses the refusal-rate operating characteristic at "
    "dependence multiples {0, 1, 2, 4, 8}x the block ONLY. A known intermediate point is NOT in "
    "this grid and is disclosed here instead (TASK_902 item 8, factory run 40): at 1.5x the "
    "guard certifies (fails to refuse) 8-13% of draws at 0.905 coverage -- below the "
    f"{TARGET_COVERAGE} target, and a materially worse operating point than the 1x row's own "
    "dep==block anchor. Do not interpolate linearly or assume monotonicity between the "
    "disclosed multiples above; this constant names the gap rather than filling it."
)


def measure_block_cap_candidates(cap_candidates=(20, 40, 60, 80, 100),
                                 n_dec_probe=(500, 2000, 8000), n_trials=150, n_boot=400,
                                 target=TARGET_COVERAGE, margin=0.03, seed=42):
    """The measurement `BLOCK_CAP_SELECTION_RATIONALE` cites — reproduced here (not just quoted)
    so a future re-run can re-derive the same cap choice rather than trust the docstring on
    faith. For each candidate cap, calibrate+validate at each probe n_dec under the
    matched-dependence (dep==block) overlapping scenario and report the validation coverage.
    Cheap relative to `build_calibration_table` (fewer trials, 3 strata) — this is a SELECTION
    sweep, not the certified table itself."""
    rng = np.random.default_rng(seed)
    out = {}
    for cap in cap_candidates:
        out[cap] = {}
        for nd in n_dec_probe:
            block = block_for(nd, max_block=cap)
            ks = gather_k_needed(nd, "overlapping", n_trials, n_boot, rng, max_block=cap)
            if ks.size == 0:
                out[cap][nd] = {"block": block, "achieved": False, "reason": "no usable trials"}
                continue
            k = float(np.percentile(ks, 100.0 * min(0.999, target + margin), method="higher"))
            cov, n_valid = measured_coverage_at_k(nd, k, "overlapping", n_trials, n_boot, rng,
                                                  max_block=cap)
            out[cap][nd] = {"block": block, "k": k, "validation_coverage": cov,
                            "validation_n": n_valid}
    return out


# ---------------------------------------------------------------------------------------------
# THE NAMED POINT SETS the pinned jitter record below was measured at (D57 fix pass, item 2).
# ORDER IS LOAD-BEARING: `measure_table_seed_jitter` assigns each point's seeds from its INDEX in
# the sequence (`seed0 + 1000*i + s`), so re-ordering these tuples changes every draw. They are
# kept in ascending n_dec, which is also the order the record and the artifact table read in.
# ---------------------------------------------------------------------------------------------
DEFAULT_SEED_JITTER_DEMOTED_POINTS = ((520, 520), (600, 600), (639, 620), (660, 660), (740, 740))
DEFAULT_SEED_JITTER_ACHIEVED_POINTS = tuple(
    (nd, nd) for nd in (100, 200, 300, 400, 500, 640, 720, 760, 880, 1000, 1080, 1160, 1200, 1500))


def seed_jitter_jobs(points, n_seeds=6, n_trials=300, n_boot=500, seed0=0, table=None):
    """The exact `_verify_one` job list `measure_table_seed_jitter` runs — factored out so the
    seeding rule (`seed0 + 1000*i + s`, i = the point's INDEX in `points`) lives in exactly one
    place and a test can rebuild a single point's jobs without re-deriving it."""
    table = MEASURED_CALIBRATION_TABLE if table is None else table
    jobs = []
    for i, (nd, src) in enumerate(points):
        k = float(table[int(src)]["k"])
        for s in range(int(n_seeds)):
            jobs.append((int(nd), int(n_trials), int(n_boot), k, int(src), TARGET_COVERAGE,
                         seed0 + 1000 * i + s, NEW_MAX_BLOCK))
    return jobs


def measure_table_seed_jitter(points, n_seeds=6, n_trials=300, n_boot=500, seed0=0, table=None,
                              n_jobs=1, only=None):
    """Re-measure a NAMED set of strata/pairings at several independent seeds, holding the
    SHIPPED k fixed — the measurement `DEMOTION_IS_MC_JITTER_NOT_A_DEMONSTRATED_HOLE` cites.

    `points` is a sequence of `(n_dec, source_stratum_n_dec)`; the k is read from `table` at
    `source_stratum_n_dec` (which equals `n_dec` for a stratum's own re-measurement, and is the
    stratum a borrowed pairing reads for a pairing re-measurement). This function CHANGES NOTHING:
    it re-measures, it does not re-roll — no `achieved` flag anywhere is derived from its output.

    `only` — restrict which n_dec are actually RUN while leaving every point's seed assignment
    exactly as it is in the full run (the seeds key off the point's index in `points`, which
    `only` does not touch). A subset re-run therefore reproduces the full run's rows for those
    n_dec BIT-IDENTICALLY, which is what lets a test pin the record exactly without paying for
    all of it."""
    jobs = seed_jitter_jobs(points, n_seeds=n_seeds, n_trials=n_trials, n_boot=n_boot,
                            seed0=seed0, table=table)
    if only is not None:
        keep = {int(nd) for nd in only}
        # D57 fix cycle 4, item D3: an `only` value that is not among `points` used to filter the
        # job list down to NOTHING and return `{}` silently. The exact-equality test that uses
        # this would then pass VACUOUSLY over zero rows if its `assert set(res) == {639}` line
        # were ever removed or weakened — a test that cannot fail. Refuse instead.
        available = sorted({int(nd) for nd, _ in points})
        absent = sorted(nd for nd in keep if nd not in available)
        if absent:
            raise ValueError(
                f"only={absent} is not in `points` (available: {available}). Filtering to an "
                "absent selector would run zero jobs and return {}, which reads as a pass to "
                "any caller that does not separately assert WHICH rows came back."
            )
        jobs = [j for j in jobs if j[0] in keep]
    recs = {}
    workers = pool_workers(n_jobs, len(jobs))
    if workers <= 1:
        results = [_verify_one(j) for j in jobs]
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_verify_one, jobs))
    for nd, rec in results:
        recs.setdefault(nd, []).append(rec["coverage_overlapping"])
    out = {}
    for nd, covs in recs.items():
        arr = np.asarray([c for c in covs if c is not None], dtype=np.float64)
        pooled_n = int(arr.size * n_trials)
        mean = float(arr.mean()) if arr.size else float("nan")
        se = (float(np.sqrt(max(mean * (1.0 - mean), 0.0) / pooled_n)) if pooled_n else
              float("nan"))
        out[nd] = {"coverage_by_seed": [float(c) for c in arr], "mean": mean,
                   "pooled_n_trials": pooled_n, "pooled_se": se,
                   "sigma_vs_target": ((mean - TARGET_COVERAGE) / se) if se else float("nan")}
    return {nd: out[nd] for nd in sorted(out)}


# The fields of a row that are MEASUREMENT OUTPUT (and so must reproduce exactly) as opposed to
# annotation (`recorded_in_table`, `note`, which are copied from the shipped table / written by
# hand). The exact-equality test compares these and only these.
SEED_JITTER_MEASURED_FIELDS = ("coverage_by_seed", "mean", "pooled_n_trials", "pooled_se",
                               "sigma_vs_target")

# RE-PINNED 2026-08-22 (D57 fix pass, item 2). THIS RECORD IS THE EXACT OUTPUT OF
#   trader_env/bin/python experiments/897_corrected_809/calibration.py --sweep-seed-jitter \
#       --n-jobs <ncpu>      (<= ncpu; `pool_workers` REFUSES more, D57 item A2)
# i.e. of `measure_table_seed_jitter(DEFAULT_SEED_JITTER_{DEMOTED,ACHIEVED}_POINTS, n_seeds=6/3)`
# at every other default (n_trials=300, n_boot=500, seed0=0, shipped k read from
# `MEASURED_CALIBRATION_TABLE`), through the real `measured_coverage_at_k` path. The path is
# DETERMINISTIC at fixed seeds, so that command reproduces these numbers BIT-IDENTICALLY and
# `test_seed_jitter_record_reproduces_exactly_at_the_named_pairing` asserts equality with NO
# tolerance (on a disclosed 1-point subset — see that test for the cost argument). NO MONTE-CARLO
# TOLERANCE APPLIES TO A RE-RUN: there is nothing for one to absorb. Verified end to end
# 2026-08-22: that command reproduced all 19 rows, every field, with zero mismatches. COST,
# MEASURED: 24.8 CPU-minutes (72 measurement jobs; 2.0 minutes wall at --n-jobs 14), which is why
# the whole record is not re-derived in the suite.
# Values are pinned at FULL PRECISION (a coverage is hits/300, which is not a short decimal), so
# nothing here is a rounded display of something else. The artifact renders them at 4dp.
#
# WHY IT WAS RE-PINNED — the superseded numbers are kept, not discarded, in
# `MEASURED_TABLE_SEED_JITTER_SUPERSEDED` below. The record originally carried here advertised NO
# command and recorded NO seeds; the D57 audit could not re-derive its load-bearing row (pairing
# 639) at any plausible seed index, so the numbers could not be re-checked by anyone. The
# DIRECTION was robust — every index probed put the 6-seed mean above the bar — but a figure that
# cannot be re-derived cannot be the pin, so the measurement was redone at a NAMED, published seed
# rule and every citing sentence was rewritten off the new output. One rhetorical clause did NOT
# survive and is RETRACTED rather than re-hunted: the old narrative's "one of those six seeds
# landed on 0.9267 exactly" — those six being the 639 PAIRING's, which is the scope of that
# retraction and the only scope it ever supported (D57 fix cycle 4 item C3; 0.9266666666666666
# does occur three times elsewhere in the record below — see
# `DEMOTION_IS_MC_JITTER_NOT_A_DEMONSTRATED_HOLE`).
#
# NOTHING IN THIS RECORD CHANGES AN `achieved` FLAG — it exists so the artifact's narrative about
# the demotions is measured rather than asserted. Group 1: the five demoted strata (620
# re-measured at its borrowed pairing 639, which is what demoted it), 6 seeds each. Group 2: a
# 14-stratum sample of ACHIEVED rows, 3 seeds each — the MIRROR direction, so the table's weakness
# is not reported only where it flatters.
MEASURED_TABLE_SEED_JITTER = {
    "demoted": {
        520: {"coverage_by_seed": [0.9266666666666666, 0.91, 0.91, 0.9233333333333333,
                                   0.9033333333333333, 0.95],
              "mean": 0.9205555555555556, "pooled_n_trials": 1800,
              "pooled_se": 0.00637412410938676, "sigma_vs_target": -1.481685057016109,
              "recorded_in_table": 0.9233},
        600: {"coverage_by_seed": [0.9666666666666667, 0.96, 0.9133333333333333,
                                   0.9466666666666667, 0.9433333333333334, 0.9533333333333334],
              "mean": 0.9472222222222223, "pooled_n_trials": 1800,
              "pooled_se": 0.005270056079746802, "sigma_vs_target": 3.267939081029606,
              "recorded_in_table": 0.9200},
        639: {"coverage_by_seed": [0.9533333333333334, 0.94, 0.94, 0.92, 0.9433333333333334,
                                   0.9666666666666667],
              "mean": 0.9438888888888889, "pooled_n_trials": 1800,
              "pooled_se": 0.0054243614227845426, "sigma_vs_target": 2.560465243069868,
              "recorded_in_table": 0.9267,
              "note": "the borrowed pairing that demoted stratum 620"},
        660: {"coverage_by_seed": [0.9233333333333333, 0.9366666666666666, 0.9466666666666667,
                                   0.9466666666666667, 0.9266666666666666, 0.94],
              "mean": 0.9366666666666665, "pooled_n_trials": 1800,
              "pooled_se": 0.005740800477586454, "sigma_vs_target": 1.1612782385827292,
              "recorded_in_table": 0.9267},
        740: {"coverage_by_seed": [0.9133333333333333, 0.96, 0.9233333333333333,
                                   0.9533333333333334, 0.92, 0.9366666666666666],
              "mean": 0.9344444444444444, "pooled_n_trials": 1800,
              "pooled_se": 0.005833715448977895, "sigma_vs_target": 0.7618548561917016,
              "recorded_in_table": 0.9167},
    },
    "achieved_sample": {
        100: {"coverage_by_seed": [0.9566666666666667, 0.96, 0.94],
              "mean": 0.9522222222222222, "pooled_n_trials": 900,
              "pooled_se": 0.0071098571425244304, "sigma_vs_target": 3.125551157604822},
        200: {"coverage_by_seed": [0.9733333333333334, 0.9866666666666667, 0.9466666666666667],
              "mean": 0.9688888888888889, "pooled_n_trials": 900,
              "pooled_se": 0.005787266662110994, "sigma_vs_target": 6.719733366269932},
        300: {"coverage_by_seed": [0.9566666666666667, 0.94, 0.96],
              "mean": 0.9522222222222222, "pooled_n_trials": 900,
              "pooled_se": 0.0071098571425244304, "sigma_vs_target": 3.125551157604822},
        400: {"coverage_by_seed": [0.98, 0.9766666666666667, 0.9633333333333334],
              "mean": 0.9733333333333333, "pooled_n_trials": 900,
              "pooled_se": 0.005370242654930927, "sigma_vs_target": 8.069157413873057},
        500: {"coverage_by_seed": [0.9666666666666667, 0.9766666666666667, 0.9733333333333334],
              "mean": 0.9722222222222223, "pooled_n_trials": 900,
              "pooled_se": 0.005477851651018154, "sigma_vs_target": 7.707806803124093},
        640: {"coverage_by_seed": [0.9533333333333334, 0.9366666666666666, 0.9266666666666666],
              "mean": 0.938888888888889, "pooled_n_trials": 900,
              "pooled_se": 0.007984467087892624, "sigma_vs_target": 1.1132726569025198},
        720: {"coverage_by_seed": [0.98, 0.9833333333333333, 0.9766666666666667],
              "mean": 0.98, "pooled_n_trials": 900,
              "pooled_se": 0.004666666666666669, "sigma_vs_target": 10.714285714285696},
        760: {"coverage_by_seed": [0.9766666666666667, 0.9666666666666667, 0.9666666666666667],
              "mean": 0.9700000000000001, "pooled_n_trials": 900,
              "pooled_se": 0.005686240703077319, "sigma_vs_target": 7.034524581126606},
        880: {"coverage_by_seed": [0.9166666666666666, 0.9366666666666666, 0.9433333333333334],
              "mean": 0.9322222222222223, "pooled_n_trials": 900,
              "pooled_se": 0.008378805975481225, "sigma_vs_target": 0.26521943922858365},
        1000: {"coverage_by_seed": [0.9766666666666667, 0.9566666666666667, 0.9566666666666667],
               "mean": 0.9633333333333334, "pooled_n_trials": 900,
               "pooled_se": 0.006264735715115752, "sigma_vs_target": 5.320788433725242},
        1080: {"coverage_by_seed": [0.9466666666666667, 0.9566666666666667, 0.9633333333333334],
               "mean": 0.9555555555555556, "pooled_n_trials": 900,
               "pooled_se": 0.0068693470337005185, "sigma_vs_target": 3.7202306755186267},
        1160: {"coverage_by_seed": [0.97, 0.9366666666666666, 0.96],
               "mean": 0.9555555555555556, "pooled_n_trials": 900,
               "pooled_se": 0.0068693470337005185, "sigma_vs_target": 3.7202306755186267},
        1200: {"coverage_by_seed": [0.97, 0.98, 0.9866666666666667],
               "mean": 0.9788888888888888, "pooled_n_trials": 900,
               "pooled_se": 0.0047918254580953355, "sigma_vs_target": 10.202560447249933},
        1500: {"coverage_by_seed": [0.9633333333333334, 0.96, 0.9666666666666667],
               "mean": 0.9633333333333334, "pooled_n_trials": 900,
               "pooled_se": 0.006264735715115752, "sigma_vs_target": 5.320788433725242},
    },
}

# THE SUPERSEDED MEASUREMENT, KEPT VISIBLE — provenance-stamped and NOT overwritten out of
# existence, the same D39 treatment `MEASURED_SHIPPED_K_AT_DEPENDENCE_MISMATCH_HALTED_CYCLE`
# gets. These were a real run through the same code path (fix cycle 3, 2026-08-22, 300 trials x
# 500 resamples per seed, shipped k held fixed); what is missing is any record of the seeds or of
# a command that would regenerate them, so they are NOT re-derivable and cannot serve as the pin.
# Kept so a reader can check for themselves what the re-measurement did and did not move.
MEASURED_TABLE_SEED_JITTER_SUPERSEDED = {
    "provenance": "fix cycle 3, 2026-08-22 — same code path, 300 trials x 500 resamples per seed, "
                  "shipped k held fixed. NO reproduce command was advertised and NO seeds were "
                  "recorded; the D57 audit could not re-derive the load-bearing row (pairing 639, "
                  "read here as 0.9200/0.9500/0.9267/0.9700/0.9500/0.9633) at any seed index it "
                  "probed, so this record is quoted as a historical measurement and is NOT "
                  "recoverable. SUPERSEDED as the pin by MEASURED_TABLE_SEED_JITTER above (D57 "
                  "fix pass, item 2), which IS the exact output of `calibration.py "
                  "--sweep-seed-jitter`. THE TWO RECORDS AGREE ON THE CONCLUSION — near the bar, "
                  "achieved-vs-failed is largely 300-trial Monte-Carlo jitter, four of the five "
                  "demotions measure at or above the bar at depth, and the jitter's errors in "
                  "this sample were conservative-only — and they DISAGREE on individual figures: "
                  "the old 639 mean reads 0.9467 where the reproducible run reads 0.9439, the old "
                  "660 reads 0.9500 where it now reads 0.9367, and the old narrative's "
                  "'one of those six seeds landed on 0.9267 exactly' is RETRACTED — NO SEED OF "
                  "THE 639 PAIRING produces it; that pairing's lowest, 0.9200, is lower still. "
                  "(Re-scoped 2026-08-22, D57 fix cycle 4 item C3: this clause used to say 'no "
                  "seed of the named run', which is false — 0.9266666666666666 appears three "
                  "times elsewhere in `MEASURED_TABLE_SEED_JITTER`, at demoted 520 seed 0, "
                  "demoted 660 seed 3004 and achieved 640 seed 5002. The retraction is about the "
                  "639 pairing, and only about it.)",
    "demoted": {
        520: {"coverage_by_seed": [0.9300, 0.9167, 0.9167, 0.9133, 0.9167, 0.9233],
              "mean": 0.9194, "pooled_n_trials": 1800, "recorded_in_table": 0.9233},
        600: {"coverage_by_seed": [0.9433, 0.9667, 0.9567, 0.9533, 0.9400, 0.9400],
              "mean": 0.9500, "pooled_n_trials": 1800, "recorded_in_table": 0.9200},
        639: {"coverage_by_seed": [0.9200, 0.9500, 0.9267, 0.9700, 0.9500, 0.9633],
              "mean": 0.9467, "pooled_n_trials": 1800, "recorded_in_table": 0.9267,
              "note": "the borrowed pairing that demoted stratum 620"},
        660: {"coverage_by_seed": [0.9367, 0.9600, 0.9500, 0.9300, 0.9567, 0.9667],
              "mean": 0.9500, "pooled_n_trials": 1800, "recorded_in_table": 0.9267},
        740: {"coverage_by_seed": [0.9400, 0.9167, 0.9233, 0.9367, 0.9300, 0.9333],
              "mean": 0.9300, "pooled_n_trials": 1800, "recorded_in_table": 0.9167},
    },
    "achieved_sample": {
        100: {"coverage_by_seed": [0.9467, 0.9600, 0.9500], "mean": 0.9522},
        200: {"coverage_by_seed": [0.9900, 0.9667, 0.9667], "mean": 0.9744},
        300: {"coverage_by_seed": [0.9733, 0.9633, 0.9533], "mean": 0.9633},
        400: {"coverage_by_seed": [0.9767, 0.9767, 0.9967], "mean": 0.9833},
        500: {"coverage_by_seed": [0.9767, 0.9667, 0.9600], "mean": 0.9678},
        640: {"coverage_by_seed": [0.9400, 0.9600, 0.9567], "mean": 0.9522},
        720: {"coverage_by_seed": [0.9633, 0.9567, 0.9367], "mean": 0.9522},
        760: {"coverage_by_seed": [0.9667, 0.9633, 0.9633], "mean": 0.9644},
        880: {"coverage_by_seed": [0.9400, 0.9500, 0.9467], "mean": 0.9456},
        1000: {"coverage_by_seed": [0.9600, 0.9667, 0.9600], "mean": 0.9622},
        1080: {"coverage_by_seed": [0.9667, 0.9667, 0.9600], "mean": 0.9644},
        1160: {"coverage_by_seed": [0.9733, 0.9600, 0.9700], "mean": 0.9678},
        1200: {"coverage_by_seed": [0.9633, 0.9633, 0.9700], "mean": 0.9656},
        1500: {"coverage_by_seed": [0.9533, 0.9733, 0.9500], "mean": 0.9589},
    },
}

DEMOTION_IS_MC_JITTER_NOT_A_DEMONSTRATED_HOLE = (
    "HOW TO READ THE `achieved=False` ROWS — every figure below RE-MEASURED and RE-PINNED "
    "2026-08-22 at a NAMED, published seed rule (`MEASURED_TABLE_SEED_JITTER`, exact output of "
    "`calibration.py --sweep-seed-jitter`), because the numbers this paragraph used to quote came "
    "from a run whose seeds were never recorded and could not be re-derived. The superseded "
    "figures are kept in `MEASURED_TABLE_SEED_JITTER_SUPERSEDED`; the conclusion is unchanged, "
    "the individual figures are not, and they are restated here from the reproducible run. "
    "Stratum 620 / block 31 was demoted on a SINGLE measured 0.9267 at its borrowed pairing "
    "n_dec=639. Re-measured at 6 independent seeds (300 trials x 500 resamples each, shipped k "
    "held fixed, 1800 pooled trials) that pairing reads "
    "0.9533 / 0.9400 / 0.9400 / 0.9200 / 0.9433 / 0.9667 — mean **0.9439**, about +2.6 pooled "
    "standard errors ABOVE the 0.93 bar. The LOWEST of those six, 0.9200, is BELOW the 0.9267 "
    "that caused the demotion: a draw at least that low is an ordinary member of this "
    "distribution (a single 300-trial batch has an SE of ~1.3pp), not a coverage defect. "
    "(RETRACTED, and not re-hunted: the previous version of this sentence claimed one of the six "
    "seeds landed on 0.9267 *exactly*. **No seed of the 639 PAIRING reproduces that value** — "
    "those six read 0.9533/0.9400/0.9400/0.9200/0.9433/0.9667 — so the clause is withdrawn "
    "rather than a seed being sought that resurrects it. RE-SCOPED 2026-08-22, D57 fix cycle 4 "
    "item C3: this retraction previously said 'no seed of the NAMED RUN reproduces that value', "
    "which is false against the run's own output and shipped verbatim into the artifact. "
    "0.9266666666666666 occurs THREE times in `MEASURED_TABLE_SEED_JITTER` — demoted stratum 520 "
    "at seed 0 (1st of its six), demoted stratum 660 at seed 3004 (5th of six) and achieved "
    "stratum 640 at seed 5002 (3rd of three) — and this very paragraph cites the last of them, "
    "four sentences below, as one of the two below-bar achieved-stratum seeds. It is the 639 "
    "pairing specifically that is free of it, which is also what "
    "`test_seed_jitter_record_reproduces_exactly_at_the_named_pairing` actually asserts. A "
    "retraction that is itself wrong, shipping verbatim into the artifact, is worse than the "
    "error it retracts — this module's own words, applied to this module.) "
    "Strata 600 (6-seed mean 0.9472, recorded 0.9200) and 639 above are the two that come back "
    "clearly above the bar; 660 (0.9367) and 740 (0.9344) come back barely above it (+1.2 and "
    "+0.8 pooled SE) and should be read as MARGINAL, not as certified-by-re-measurement; and "
    "**520 (0.9206)** comes back BELOW it, 1.5 pooled SE under, the one demotion of the five this "
    "re-measurement positively supports. "
    "**THE DEMOTIONS ARE KEPT ANYWAY** — they fail closed, they propagate correctly, and "
    "re-rolling a failure until it passes is selection on noise — but the artifact must not tell "
    "a reader that block 31 has a real hole, because the measurement says it does not. "
    "THE MIRROR DIRECTION, MEASURED AND REPORTED BECAUSE IT CUTS THE OTHER WAY: 14 ACHIEVED "
    "strata were re-measured at 3 seeds each (100/200/300/400/500/640/720/760/880/1000/1080/"
    "1160/1200/1500) and every one's 3-seed MEAN sat at or above the bar, the weakest being "
    "n_dec=880 at 0.9322 (+0.3 pooled SE, i.e. indistinguishable from the bar) — but 2 of those "
    "42 individual seeds fell BELOW it (0.9167 at n_dec=880 and 0.9267 at n_dec=640), which "
    "sharpens rather than softens the point: a single 300-trial batch at an ACHIEVED stratum can "
    "read as low as the batches that got other strata demoted. "
    "CONCLUSION, STATED PLAINLY: near the bar, whether a stratum reads achieved or failed is "
    "LARGELY MONTE-CARLO JITTER AT 300 TRIALS, so **the shipped table is weaker than it reads "
    "PER STRATUM** — an individual `achieved=True` is a 300-trial pass, not a proof, and an "
    "individual `achieved=False` is not a proof of a hole either. In THIS sample the jitter "
    "produced only CONSERVATIVE errors: it demoted four regions whose 6-seed mean is at or above "
    "the bar and promoted none whose 3-seed mean is below it. That is the direction it must err "
    "in, and it is what was observed — it is not a guarantee about strata nobody re-measured."
)


# =============================================================================================
# THE CERTIFIED TABLE — D55(a)'s deliverable. Measured 2026-08-22, reproducible verbatim:
#   trader_env/bin/python experiments/897_corrected_809/calibration.py
# (defaults: DEFAULT_N_DEC_STRATA, target=0.93, n_trials_calib=n_trials_valid=300, n_boot=500,
# seed=0, k_cap=8.0, calibration_margin=0.03, max_block=NEW_MAX_BLOCK=60 — every default above).
# 67 STRATA, 62 ACHIEVED, 5 NOT — "achieved" requires validated overlapping-dependence coverage
# >= 0.93 on an INDEPENDENT 300-trial batch from the one `k` was chosen on, AND (where a probe
# exists) at the BORROWED pairing a real cell reads. Achieved k stays in [1.2317, 1.6905] — min
# 1.231702348964501 at n_dec=100000, max 1.6904650613309085 at n_dec=980 — comfortably
# under k_cap=8, i.e. a genuine calibration, never an interval "widened until it can't fail."
# (Corrected 2026-08-22, D57 fix cycle 4 item C4: this read "[1.23, 1.69]", and the max achieved k
# is 1.69046..., i.e. OUTSIDE the interval the sentence stated. The figures are now quoted at
# enough digits to contain the values they describe, and
# `test_achieved_k_interval_claim_contains_every_achieved_k` derives them from the table.)
#
# ONE of the five was demoted by the borrowed-pairing check despite passing its own batch:
# n_dec=620 / block 31 — see MEASURED_BORROWED_PAIRING_COVERAGE below for that row and what it
# cost. The other FOUR failed their own validation batch, AND ARE LEFT THAT WAY:
# n_dec = 520 / 600 / 660 / 740 (blocks 26 / 30 / 33 / 37) came in at
# 0.9233 / 0.9200 / 0.9267 / 0.9167 against the 0.93 bar.
#
# ALL FIVE DEMOTIONS STAND — and the reason they stand is NOT that a coverage defect was
# demonstrated at any of them. See `DEMOTION_IS_MC_JITTER_NOT_A_DEMONSTRATED_HOLE` above: every
# one of those five single-batch numbers was re-measured at 6 independent seeds (300 trials x 500
# resamples each, 1800 pooled trials per stratum) and four of the five came back AT OR ABOVE the
# bar — 600 at 0.9472 and 620/639 at 0.9439 clearly so (+3.3 / +2.6 pooled SE), 660 at 0.9367 and
# 740 at 0.9344 only barely (+1.2 / +0.8 SE, read them as marginal); only 520 (0.9206) came back
# under it. The demotions are kept because they FAIL CLOSED and
# propagate correctly (a demoted stratum is simply refused service, which costs nothing but a
# qualifier), NOT because block 31 or block 30 has a hole. Anyone reading the `achieved=False`
# rows as "these regions were shown to under-cover" is reading them wrong.
# At 300 validation trials the binomial SE against a true coverage of ~0.95 is ~1.3pp, so a
# stratum whose true coverage is fine still fails this bar about 5% of the time; over 67 strata
# roughly 3 such failures are EXPECTED, and 4 is what a fair coin gave. THE OBVIOUS TEMPTATION —
# re-run those four with a different seed until they pass — IS REFUSED: re-rolling only the
# failures and keeping only the passes is selection on noise, and it would turn a certification
# into a search. (The 6-seed re-measurement above is NOT that move: it re-measures a NAMED,
# pre-specified set at fixed depth, changes no `achieved` flag in either direction, and is
# reported in full including the two strata it did NOT rescue.) They are recorded as
# `achieved=False` with their measured coverage and reason,
# `calibrated_k_for` refuses to serve them, and every cell whose n_dec lands in
# [520,539] / [600,619] / [660,679] / [740,759] keeps the UNCALIBRATED qualifier and says so
# per-cell. If a future pass wants those four regions certified, the honest move is to raise
# `n_trials_valid` (or `calibration_margin`) FOR EVERY STRATUM and re-measure the whole table.
#
# `sweep_gated_maker.py` looks this table up via `calibrated_k_for` (SAME-BLOCK nearest-below
# floor).
#
# REPRODUCTION, CORRECTED (D57 fix pass, item 1). An earlier revision of this header advertised
# the table as "reproducible verbatim" and then said the re-run would "NOT be bit-identical
# (fresh random draws each run)" and that "WHICH strata fall under the bar will move run to run".
# BOTH OF THOSE WERE FALSE, and they are retracted here: `build_calibration_table` seeds every
# stratum DETERMINISTICALLY (`seed + 1000*index` over the sorted strata — see `stratum_seed`), so
# the advertised command above reproduces this table BIT-IDENTICALLY, and nothing about which
# strata fall under the bar moves between runs of it. There is NO run-to-run Monte Carlo noise on
# this path and therefore no MC tolerance may be disclosed against a re-run of it. Verified at
# named strata (40 / 620 / 1000 and, out of band, 100 / 500 / 520 / 600 / 660 / 740 / 1200 /
# 3000): every MEASURED field of every one reproduced exactly — 11 of 11, the only difference
# anywhere being stratum 620's `achieved`/`reason`, which are an overlay and not a measurement
# (next paragraph). See
# `test_measured_calibration_table_reproduces_exactly_at_named_strata`, which pins a 3-stratum
# subset because the full 67-stratum run costs tens of CPU-minutes (>= 23 at the smallest
# measured per-stratum cost) and no test may pay that.
# The reason this measurement is pinned as a constant rather than regenerated at import time is
# therefore COST, not drift.
#
# THE ONE PLACE THIS TABLE IS NOT A VERBATIM `calibrate_block` DUMP: n_dec=620's `achieved` and
# `reason` are OVERWRITTEN by the borrowed-pairing demotion (its own batch passed at 0.96; the
# pairing at 639 did not). Re-running the harness at stratum 620 returns achieved=True; every
# measured field — block, k, both validation coverages, both validation n — still matches
# exactly. That overlay is deliberate, is stated on the row itself, and is what the subset test
# asserts at 620 rather than papering over.
#
# WHAT DOES vary run-to-run is a DIFFERENT, hypothetical experiment: a re-measurement at seeds
# nobody recorded (the binomial SE paragraph above, and `MEASURED_TABLE_SEED_JITTER`, which
# measures exactly that). Neither is a property of re-running the command above.
# =============================================================================================
def _E(block, k, cov_ov, cov_iid, n_ov=300, n_iid=300, achieved=True, reason=None):
    """Row constructor for `MEASURED_CALIBRATION_TABLE`. The corrected stratification (one
    stratum at every block-transition point, MAJOR-1 fix) needs 67 rows, which is too many to
    spell out field-by-field without the table becoming unreadable — every `DISCLOSURE_FIELD` is
    still present on every row, and the artifact reads it through `trim_table_for_disclosure`."""
    return {"block": int(block), "k": float(k), "achieved": bool(achieved),
            "validation_coverage_overlapping": cov_ov, "validation_coverage_iid": cov_iid,
            "validation_n_overlapping": int(n_ov), "validation_n_iid": int(n_iid),
            "reason": reason}


MEASURED_CALIBRATION_TABLE = {
    40: _E(2, 1.379384151362865, 0.9566666666666667, 0.9833333333333333),
    60: _E(3, 1.3831617402092362, 0.9533333333333334, 0.9933333333333333),
    80: _E(4, 1.437983407419496, 0.9566666666666667, 0.99),
    100: _E(5, 1.3977296267458934, 0.9566666666666667, 0.9833333333333333),
    120: _E(6, 1.6063885335058676, 0.9633333333333334, 0.9933333333333333),
    140: _E(7, 1.4736880764878186, 0.96, 0.9933333333333333),
    160: _E(8, 1.4946011076029946, 0.9633333333333334, 0.9966666666666667),
    180: _E(9, 1.582773475724553, 0.9666666666666667, 0.9933333333333333),
    200: _E(10, 1.6203425093333903, 0.98, 0.9966666666666667),
    220: _E(11, 1.660072642524351, 0.9833333333333333, 0.9966666666666667),
    240: _E(12, 1.5381623569302085, 0.9666666666666667, 0.9933333333333333),
    260: _E(13, 1.449326814895103, 0.96, 0.9933333333333333),
    280: _E(14, 1.458772356030053, 0.9533333333333334, 0.99),
    300: _E(15, 1.433384717345944, 0.9533333333333334, 0.9866666666666667),
    320: _E(16, 1.5277667936389125, 0.9733333333333334, 0.99),
    340: _E(17, 1.4589228101326053, 0.96, 0.9933333333333333),
    360: _E(18, 1.5291004592601227, 0.9533333333333334, 0.9966666666666667),
    380: _E(19, 1.6087271288498413, 0.9633333333333334, 0.9933333333333333),
    400: _E(20, 1.6183406664478082, 0.96, 0.9966666666666667),
    420: _E(21, 1.5100596386400145, 0.9566666666666667, 0.9933333333333333),
    440: _E(22, 1.537399937295367, 0.97, 0.9966666666666667),
    460: _E(23, 1.449435282547826, 0.9566666666666667, 0.9833333333333333),
    480: _E(24, 1.50649412669772, 0.97, 0.9866666666666667),
    500: _E(25, 1.597888555118712, 0.9766666666666667, 1.0),
    520: _E(26, 1.2789902814398504, 0.9233333333333333, 0.9633333333333334, 300, 300, achieved=False, reason='validation-batch overlapping coverage 0.9233333333333333 < target 0.93 despite a calibration-batch percentile of k=1.279 at n_dec=520 (block=26) — the calibration did not reproduce out-of-sample at this stratum'),
    540: _E(27, 1.5102929562248057, 0.97, 0.9833333333333333),
    560: _E(28, 1.6211278666052396, 0.97, 0.9933333333333333),
    580: _E(29, 1.5382955001151943, 0.9633333333333334, 0.9933333333333333),
    600: _E(30, 1.378373100611602, 0.92, 0.9866666666666667, 300, 300, achieved=False, reason='validation-batch overlapping coverage 0.92 < target 0.93 despite a calibration-batch percentile of k=1.378 at n_dec=600 (block=30) — the calibration did not reproduce out-of-sample at this stratum'),
    # DEMOTED by the borrowed-pairing check, not by its own batch: this stratum's independent
    # validation batch passed at 0.96, but the pairing a real cell at n_dec=639 actually reads
    # (same k, same block 31, that cell's own n_dec) measured 0.9267 < 0.93. See
    # MEASURED_BORROWED_PAIRING_COVERAGE[639]. Certification is two-sided; this is what that costs.
    # THE DEMOTION STANDS BUT IT IS MC JITTER, NOT A DEMONSTRATED HOLE: pairing 639 re-measured at
    # 6 NAMED seeds means 0.9439 (1800 pooled trials), +2.6 pooled SE ABOVE the bar, and its
    # LOWEST seed (0.9200) is below the 0.9267 that demoted the stratum — a draw that low is an
    # ordinary member of the distribution. See DEMOTION_IS_MC_JITTER_NOT_A_DEMONSTRATED_HOLE.
    # (`achieved`/`reason` on THIS row are the demotion OVERLAY, not `calibrate_block` output —
    # re-running the harness at stratum 620 returns achieved=True with these same measured
    # numbers. See the table header.)
    620: _E(31, 1.3623645161993703, 0.96, 0.9866666666666667, achieved=False,
            reason="own validation batch passed (overlapping 0.96) but the BORROWED pairing at "
                   "n_dec=639 — same block 31, same k, the interval a real cell in [620,639] "
                   "actually reads — measured 0.9267 < target 0.93; demoted rather than "
                   "certified on the stratum-only check. NOT a demonstrated coverage defect: "
                   "that pairing re-measured at 6 named seeds gives mean 0.9439 over 1800 pooled "
                   "trials (+2.6 pooled SE above the bar), its lowest seed reading 0.9200 — the "
                   "demotion is kept because it fails CLOSED, not because block 31 "
                   "has a hole (DEMOTION_IS_MC_JITTER_NOT_A_DEMONSTRATED_HOLE)"),
    640: _E(32, 1.4010469078434356, 0.9633333333333334, 0.99),
    660: _E(33, 1.385631434766136, 0.9266666666666666, 0.9933333333333333, 300, 300, achieved=False, reason='validation-batch overlapping coverage 0.9266666666666666 < target 0.93 despite a calibration-batch percentile of k=1.386 at n_dec=660 (block=33) — the calibration did not reproduce out-of-sample at this stratum'),
    680: _E(34, 1.5575758817729444, 0.9633333333333334, 0.9866666666666667),
    700: _E(35, 1.5039875091639507, 0.97, 1.0),
    720: _E(36, 1.530618858381558, 0.9433333333333334, 0.99),
    740: _E(37, 1.3057566112986687, 0.9166666666666666, 0.9766666666666667, 300, 300, achieved=False, reason='validation-batch overlapping coverage 0.9166666666666666 < target 0.93 despite a calibration-batch percentile of k=1.306 at n_dec=740 (block=37) — the calibration did not reproduce out-of-sample at this stratum'),
    760: _E(38, 1.5032845896649152, 0.95, 0.9966666666666667),
    780: _E(39, 1.4795724400095958, 0.95, 0.9966666666666667),
    800: _E(40, 1.3976290742378474, 0.9633333333333334, 0.9833333333333333),
    820: _E(41, 1.5163440214372863, 0.96, 0.9866666666666667),
    840: _E(42, 1.5521381772895928, 0.9766666666666667, 0.98),
    860: _E(43, 1.5017468322818073, 0.9633333333333334, 1.0),
    880: _E(44, 1.3802036809802014, 0.9533333333333334, 0.98),
    900: _E(45, 1.6308147481699817, 0.9833333333333333, 0.9933333333333333),
    920: _E(46, 1.429663455984171, 0.97, 0.98),
    940: _E(47, 1.4121203394119448, 0.9533333333333334, 0.9966666666666667),
    960: _E(48, 1.5544160741967818, 0.97, 1.0),
    980: _E(49, 1.6904650613309085, 0.9833333333333333, 0.9866666666666667),
    1000: _E(50, 1.5309881369858398, 0.98, 0.9866666666666667),
    1020: _E(51, 1.404536210354013, 0.9333333333333333, 0.9866666666666667),
    1040: _E(52, 1.5656816390687511, 0.97, 0.9966666666666667),
    1060: _E(53, 1.3788581276311338, 0.9666666666666667, 0.98),
    1080: _E(54, 1.4712692754403387, 0.9433333333333334, 0.99),
    1100: _E(55, 1.5505771153020906, 0.9533333333333334, 0.99),
    1120: _E(56, 1.4171279156254761, 0.9533333333333334, 0.99),
    1140: _E(57, 1.4764159448114051, 0.95, 0.9966666666666667),
    1160: _E(58, 1.4347897234000555, 0.9466666666666667, 0.99),
    1180: _E(59, 1.5492018598331803, 0.9633333333333334, 0.9933333333333333),
    1200: _E(60, 1.533885528906261, 0.9833333333333333, 0.9966666666666667),
    1500: _E(60, 1.4067042524197753, 0.9466666666666667, 0.98),
    2000: _E(60, 1.4726387190135317, 0.9766666666666667, 1.0),
    3000: _E(60, 1.2409929802023711, 0.9366666666666666, 0.97),
    5000: _E(60, 1.3651455756818023, 0.9533333333333334, 0.98),
    10000: _E(60, 1.3084008911104212, 0.9666666666666667, 0.9833333333333333),
    30000: _E(60, 1.2823151125283507, 0.9333333333333333, 0.9866666666666667),
    100000: _E(60, 1.231702348964501, 0.9333333333333333, 0.98),
    300000: _E(60, 1.3646346799480558, 0.9666666666666667, 0.9966666666666667),
}


# =============================================================================================
# ACHIEVED COVERAGE AT THE BORROWED PAIRINGS — the half of the certification that has to be true
# for the cells that will USE it, not merely at the pinned strata. Measured 2026-08-22,
# reproducible verbatim EXCEPT FOR ONE ROW (see the next paragraph):
#   trader_env/bin/python experiments/897_corrected_809/calibration.py --verify-borrowed
# Each row is a NON-stratum n_dec (a real cell essentially never lands on a stratum) run at ITS
# OWN real block with the k `calibrated_k_for` actually hands it.
#
# THE ONE ROW THAT CANNOT REPRODUCE — n_dec=639, and it is the load-bearing one (D57 fix cycle 4,
# item C2). The 639 row below is a PRE-DEMOTION measurement: it was taken while stratum 620 was
# still `achieved=True`, which is what let `calibrated_k_for(639)` hand back k=1.3624 to be
# graded at 0.9267. That 0.9267 is WHY stratum 620 was then demoted — and with the demotion in
# the shipped table, `calibrated_k_for(639)` now returns `(None, False, None)`, so
# `verify_borrowed_pairings` routes 639 into `skipped` and the command prints it IDENTICALLY to
# 619, the deliberate negative control: `borrowed_k=None from_stratum=None cov_overlap=None
# cov_iid=None meets_target=False`. Re-run and verified 2026-08-22: 16 probe jobs ran instead of
# 18, and 17 of the 18 rows reproduced EVERY field exactly; the 639 row is the sole difference.
# This is the same self-referential overlay `MEASURED_CALIBRATION_TABLE` discloses on its own 620
# row, in the other direction: there, a measured row is overwritten by the demotion it caused;
# here, a measured row is the demotion's CAUSE and the demotion has since made the measurement
# unrepeatable. The row is KEPT rather than deleted — it is the evidence for the demotion, and
# deleting it would leave the `achieved=False` on stratum 620 unexplained — but the "reproducible
# verbatim" claim above is scoped to the other 17 rows, and
# `test_borrowed_pairing_639_is_a_pre_demotion_row_the_command_now_refuses` pins that scoping so
# it cannot drift back into an unqualified "verbatim".
#
# THE CERTIFICATION RULE IS TWO-SIDED: a stratum is `achieved=True` only if BOTH its own
# independent validation batch AND every borrowed probe that reads it cleared the target. That
# rule cost a stratum here and the cost was paid rather than argued away:
#
#   * n_dec=639 (block 31) borrows k=1.3624 from stratum 620, whose OWN validation batch passed
#     at 0.96 — but the borrowed pairing measured 0.9267, UNDER the 0.93 bar. Stratum 620 is
#     therefore DEMOTED to `achieved=False`, and every cell with n_dec in [620, 639] keeps the
#     UNCALIBRATED qualifier.
#     READ THIS ROW CORRECTLY: the demotion is a FAIL-CLOSED response to one low draw, not
#     evidence that block 31 under-covers. Re-measured at 6 NAMED seeds the same pairing means
#     0.9439 over 1800 pooled trials (+2.6 pooled SE above the bar) and its lowest seed reads
#     0.9200, below the draw that demoted it.
#     See `DEMOTION_IS_MC_JITTER_NOT_A_DEMONSTRATED_HOLE`, which also reports the mirror
#     direction (14 achieved strata re-measured at 3 seeds, every MEAN at or above the bar but 2
#     of 42 individual seeds below it) and the
#     consequence: near the bar, achieved-vs-failed is largely MC jitter at 300 trials, so this
#     table is weaker than it reads PER STRATUM — in this sample the jitter erred only
#     conservatively.
#   * n_dec=619 (block 30) is a deliberate negative control: its stratum (600) had already
#     failed validation, so the probe must come back with NO coverage claim at all. It does.
#
# HONEST LIMIT OF THIS CHECK: only the 8 regime-A blocks and the regime-B rungs listed in
# `DEFAULT_BORROWED_PROBE_POINTS` got the second (borrowed) test. The other ~50 regime-A blocks
# were certified on their own validation batch alone, and block 31 is direct evidence that the
# weaker check can pass where the stronger one fails. Read the un-probed blocks accordingly.
# =============================================================================================
def _B(block, k, src, cov_ov, cov_iid, n_trials, n_boot, meets_target, reason=None):
    return {"block": int(block), "borrowed_k": k, "borrowed_from_n_dec": src,
            "coverage_overlapping": cov_ov, "coverage_iid": cov_iid,
            "n_trials": n_trials, "n_boot": n_boot, "meets_target": bool(meets_target),
            "reason": reason}


MEASURED_BORROWED_PAIRING_COVERAGE = {
    59: _B(2, 1.379384151362865, 40, 0.96, 0.99, 300, 500, True),
    79: _B(3, 1.3831617402092362, 60, 0.9466666666666667, 0.9866666666666667, 300, 500, True),
    119: _B(5, 1.3977296267458934, 100, 0.9366666666666666, 0.9966666666666667, 300, 500, True),
    219: _B(10, 1.6203425093333903, 200, 0.97, 0.9966666666666667, 300, 500, True),
    419: _B(20, 1.6183406664478082, 400, 0.9933333333333333, 1.0, 300, 500, True),
    619: _B(30, None, None, None, None, 300, 500, False,
            "no certified stratum at this block at or below this n_dec (stratum 600 failed its "
            "own validation batch) — stays UNCALIBRATED by construction"),
    639: _B(31, 1.3623645161993703, 620, 0.9266666666666666, 0.97, 300, 500, False,
            "BORROWED-PAIRING FAILURE: stratum 620 passed its own validation batch (0.96) but "
            "the pairing a real cell at n_dec=639 actually reads came in at 0.9267 < 0.93 — "
            "stratum 620 is demoted to achieved=False and block 31 keeps the qualifier. "
            "RE-MEASURED AT DEPTH (re-pinned at NAMED seeds, D57 fix pass item 2): 6 independent "
            "seeds give 0.9533/0.9400/0.9400/0.9200/0.9433/0.9667, mean 0.9439 over 1800 pooled "
            "trials — ABOVE the bar. The 0.9267 recorded here is a low draw of that distribution "
            "(the lowest of the six, 0.9200, is lower still), NOT a demonstrated hole at block "
            "31. The demotion is kept because it "
            "fails closed; see DEMOTION_IS_MC_JITTER_NOT_A_DEMONSTRATED_HOLE. "
            "PROVENANCE (D57 item C2): this row is a PRE-DEMOTION measurement and is the one row "
            "of the 18 that `--verify-borrowed` no longer reproduces — now that stratum 620 is "
            "`achieved=False`, `calibrated_k_for(639)` returns no k and the command prints this "
            "point exactly like 619, the negative control. Kept because it is the evidence FOR "
            "the demotion, not because it can be re-run."),
    919: _B(45, 1.6308147481699817, 900, 0.96, 0.99, 300, 500, True),
    1199: _B(59, 1.5492018598331803, 1180, 0.9633333333333334, 0.99, 300, 500, True),
    1250: _B(60, 1.533885528906261, 1200, 0.9666666666666667, 0.9966666666666667, 300, 500, True),
    1400: _B(60, 1.533885528906261, 1200, 0.9466666666666667, 0.9933333333333333, 300, 500, True),
    1800: _B(60, 1.4067042524197753, 1500, 0.9666666666666667, 0.9866666666666667, 300, 500, True),
    2500: _B(60, 1.4726387190135317, 2000, 0.9766666666666667, 0.9933333333333333, 300, 500, True),
    4000: _B(60, 1.2409929802023711, 3000, 0.95, 0.97, 300, 500, True),
    7000: _B(60, 1.3651455756818023, 5000, 0.9566666666666667, 0.9966666666666667, 300, 500, True),
    20000: _B(60, 1.3084008911104212, 10000, 0.9666666666666667, 0.9866666666666667, 300, 500,
              True),
    50000: _B(60, 1.2823151125283507, 30000, 0.965, 0.985, 200, 300, True),
    200000: _B(60, 1.231702348964501, 100000, 0.965, 0.965, 200, 300, True),
}


def measure_dependence_mismatch(n_dec=8000, caps=(20, 60), dep_multiples=(1, 2, 4, 8, 16),
                                n_trials=150, n_boot=400, target=TARGET_COVERAGE, margin=0.03,
                                seed=11):
    """The SWEEP 2 measurement `BLOCK_CAP_SELECTION_RATIONALE` cites, reproducible verbatim.
    Holds `n_dec` fixed and pushes the REAL dependence `dep` to `dep_multiples[i] * block` for
    each candidate cap's own block length, reporting the calibrated k and validation coverage
    at each point — the stress test for whether a capped block structurally fails once
    dependence exceeds it, as opposed to `measure_block_cap_candidates`'s matched-dependence
    (dep==block) case."""
    rng = np.random.default_rng(seed)
    out = {}
    for cap in caps:
        block = block_for(n_dec, max_block=cap)
        out[cap] = {"block": block, "by_dep_multiple": {}}
        for mult in dep_multiples:
            dep = block * mult
            ks = gather_k_needed(n_dec, "overlapping", n_trials, n_boot, rng, max_block=cap,
                                 dep=dep)
            if ks.size == 0:
                out[cap]["by_dep_multiple"][mult] = {"dep": dep, "achieved": False,
                                                      "reason": "no usable trials"}
                continue
            k = float(np.percentile(ks, 100.0 * min(0.999, target + margin), method="higher"))
            cov, n_valid = measured_coverage_at_k(n_dec, k, "overlapping", n_trials, n_boot, rng,
                                                  max_block=cap, dep=dep)
            out[cap]["by_dep_multiple"][mult] = {"dep": dep, "k": k,
                                                 "validation_coverage": cov,
                                                 "validation_n": n_valid}
    return out


def main():
    import argparse
    import json
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-trials-calib", type=int, default=300)
    ap.add_argument("--n-trials-valid", type=int, default=300)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-dec-strata", type=int, nargs="+", default=list(DEFAULT_N_DEC_STRATA))
    ap.add_argument("--calibration-margin", type=float, default=0.03, dest="calibration_margin")
    ap.add_argument("--max-block", type=int, default=NEW_MAX_BLOCK)
    ap.add_argument("--out", default=None, help="write the full table as JSON")
    ap.add_argument("--n-jobs", type=int, default=1,
                    help="process-pool width (CPU only); seeds are per-stratum so this changes "
                         "wall clock, never the draws. REFUSED above this machine's CPU count "
                         f"({os.cpu_count() or 1} here) and clamped down to the job count — see "
                         "`pool_workers`.")
    ap.add_argument("--verify-borrowed", action="store_true",
                    help="re-measure achieved coverage at BORROWED (non-stratum) (block, n_dec) "
                         "pairings using MEASURED_CALIBRATION_TABLE — the MAJOR-1 certification")
    ap.add_argument("--sweep-caps", action="store_true",
                    help="run measure_block_cap_candidates instead of the certification table")
    ap.add_argument("--sweep-mismatch", action="store_true",
                    help="run measure_dependence_mismatch instead of the certification table "
                         "(k RE-DERIVED at each dependence level — NOT the shipped k)")
    ap.add_argument("--sweep-shipped-k-mismatch", action="store_true",
                    dest="sweep_shipped_k_mismatch",
                    help="MAJOR 1(a): hold the SHIPPED k fixed and re-measure coverage as the "
                         "real dependence exceeds the block — the certification's condition")
    ap.add_argument("--sweep-dependence-refusal", action="store_true",
                    dest="sweep_dependence_refusal",
                    help="MAJOR 1(b): operating characteristic of the dependence-refusal guard "
                         "(`honest_estimator.dependence_length`), both directions")
    ap.add_argument("--reproduce-strata", type=int, nargs="+", default=None,
                    dest="reproduce_strata",
                    help="D57 item 1: re-run a SUBSET of strata at exactly the seeds the full "
                         "sweep gives them (`stratum_seed`) and diff against "
                         "MEASURED_CALIBRATION_TABLE — an exact check without the full "
                         "tens-of-CPU-minutes sweep")
    ap.add_argument("--sweep-seed-jitter", action="store_true", dest="sweep_seed_jitter",
                    help="D57 item 2: reproduce MEASURED_TABLE_SEED_JITTER — the named demoted "
                         "and achieved point sets re-measured at 6/3 seeds with the shipped k "
                         "held fixed (slow: use --n-jobs)")
    ap.add_argument("--sweep-tau-out-of-family", action="store_true",
                    dest="sweep_tau_out_of_family",
                    help="D57: fast direction-only reproduction of the tau guard's out-of-family "
                         "mismatch (a low-amplitude, long-range persistent series)")
    args = ap.parse_args()

    if args.reproduce_strata:
        # D57 fix cycle 4, item D2: `--reproduce-strata 61` used to die on
        # `ValueError: 61 is not in list` from `stratum_seed`'s `strata.index(...)` — a traceback,
        # not a refusal — and the `"<not a shipped stratum>"` branch below could never be reached
        # because the crash came first. Validate here, name the mistake, and drop the dead branch.
        unknown = [nd for nd in args.reproduce_strata if nd not in DEFAULT_N_DEC_STRATA]
        if unknown:
            raise SystemExit(
                f"REFUSAL: --reproduce-strata {unknown} — not shipped strata. This command "
                "re-runs strata at the seed the FULL sweep gives them, which is keyed off the "
                "value's INDEX in DEFAULT_N_DEC_STRATA, so a value that is not in that list has "
                "no seed to reproduce. Shipped strata: "
                f"{', '.join(str(n) for n in DEFAULT_N_DEC_STRATA)}."
            )
        res = reproduce_strata(args.reproduce_strata, n_jobs=args.n_jobs)
        for nd in args.reproduce_strata:
            e, p = res[nd], MEASURED_CALIBRATION_TABLE[nd]
            diffs = [f for f in DISCLOSURE_FIELDS if e.get(f) != p.get(f)]
            print(f"n_dec={nd:6d} block={e['block']:3d} k={e['k']!r} achieved={e['achieved']} "
                  f"cov_overlap={e.get('validation_coverage_overlapping')!r} "
                  f"cov_iid={e.get('validation_coverage_iid')!r} "
                  f"diff_vs_shipped={diffs or 'NONE (bit-identical)'}")
        return

    if args.sweep_seed_jitter:
        for group, points, n_seeds in (
                ("demoted", DEFAULT_SEED_JITTER_DEMOTED_POINTS, 6),
                ("achieved_sample", DEFAULT_SEED_JITTER_ACHIEVED_POINTS, 3)):
            res = measure_table_seed_jitter(points, n_seeds=n_seeds, n_jobs=args.n_jobs)
            for nd in sorted(res):
                e = res[nd]
                print(f"{group:16s} n_dec={nd:5d} by_seed={e['coverage_by_seed']} "
                      f"mean={e['mean']!r} pooled_n={e['pooled_n_trials']} "
                      f"pooled_se={e['pooled_se']!r} sigma={e['sigma_vs_target']!r}")
        return

    if args.sweep_tau_out_of_family:
        res = measure_tau_out_of_family_mismatch()
        for phi, d in res["by_phi"].items():
            print(f"n_dec={res['n_dec']:6d} block={res['block']:3d} weight={res['weight']:.2f} "
                  f"phi={phi:.4f} median_tau={d['median_tau']!r} "
                  f"certify_rate={d['certify_rate']:.3f} coverage={d['coverage']!r} "
                  f"n_usable={d['n_usable']}")
        return

    if args.sweep_shipped_k_mismatch:
        res = measure_shipped_k_at_dependence_mismatch()
        for n_dec, e in res.items():
            for mult, d in e["by_dep_multiple"].items():
                print(f"n_dec={n_dec:7d} block={e['block']:3d} shipped_k={e['shipped_k']:.4f} "
                      f"dep_mult={mult:2d} dep={d['dep']:5d} "
                      f"coverage_by_seed={d['coverage_by_seed']}")
        return

    if args.sweep_dependence_refusal:
        res = measure_dependence_refusal_operating_characteristic()
        for n_dec, e in res.items():
            for mult, d in e["by_dep_multiple"].items():
                print(f"n_dec={n_dec:7d} block={e['block']:3d} dep_mult={mult:2d} "
                      f"dep={d['dep']!r:>6} refusal={d['refusal_rate']:.3f} "
                      f"unmeasurable={d['unmeasurable_rate']:.3f} median_tau={d['median_tau']!r}")
        return

    if args.sweep_caps:
        res = measure_block_cap_candidates()
        for cap, per_nd in res.items():
            for nd, e in per_nd.items():
                print(f"cap={cap:4d} n_dec={nd:6d} block={e.get('block')} k={e.get('k')!r} "
                     f"val_cov={e.get('validation_coverage')!r}")
        return

    if args.sweep_mismatch:
        res = measure_dependence_mismatch()
        for cap, entry in res.items():
            block = entry["block"]
            for mult, e in entry["by_dep_multiple"].items():
                print(f"cap={cap:4d} block={block:4d} dep_mult={mult:3d} dep={e.get('dep')} "
                     f"k={e.get('k')!r} val_cov={e.get('validation_coverage')!r}")
        return

    if args.verify_borrowed:
        res = verify_borrowed_pairings(MEASURED_CALIBRATION_TABLE, n_jobs=args.n_jobs)
        for nd in sorted(res):
            e = res[nd]
            print(f"n_dec={nd:7d} block={e['block']:3d} borrowed_k={e['borrowed_k']!r} "
                  f"from_stratum={e['borrowed_from_n_dec']!r} "
                  f"cov_overlap={e['coverage_overlapping']!r} cov_iid={e['coverage_iid']!r} "
                  f"meets_target={e['meets_target']}")
        if args.out:
            with open(args.out, "w") as fh:
                json.dump({str(k): v for k, v in res.items()}, fh, indent=2, default=float)
            print(f"wrote {args.out}")
        return

    table = build_calibration_table(args.n_dec_strata, n_trials_calib=args.n_trials_calib,
                                    n_trials_valid=args.n_trials_valid, n_boot=args.n_boot,
                                    seed=args.seed, calibration_margin=args.calibration_margin,
                                    max_block=args.max_block, n_jobs=args.n_jobs)
    for nd in sorted(table):
        e = table[nd]
        print(f"n_dec={nd:6d} block={e['block']:3d} k={e['k']!r} achieved={e['achieved']} "
             f"cov_overlap={e.get('validation_coverage_overlapping')!r} "
             f"cov_iid={e.get('validation_coverage_iid')!r} reason={e.get('reason')}")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(table, fh, indent=2, default=float)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
