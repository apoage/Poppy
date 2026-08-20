#!/usr/bin/env python3
"""TASK_892 — the DISCLOSURE block every 892 artifact carries inline.

One definition, imported by both ``trace.py`` (stamped into every trace's ``provenance``) and
``report.py`` (stamped at the TOP of every report), so a reader of EITHER artifact cannot miss
what this instrument delivers that its pre-registration did not ask for. Duplicating the text in
two modules would let them drift; a planner reading only the report must see the same words a
planner reading only a trace sees.

``experiments/892_action_ev/SPEC.md`` IS A FROZEN PRE-REGISTRATION AND IS NOT AMENDED TO MATCH
THIS BUILD. Editing a pre-registration so it describes what was actually built is the
self-certification this program forbids; the deviations below are therefore recorded as
DEVIATIONS, in the artifact, for the planner to rule on (seeded as a planner decision row).
"""

SPEC_PATH = "experiments/892_action_ev/SPEC.md"

# ---------------------------------------------------------------------------------------------
# The headline. Kept to one screen because it is meant to be READ, not skimmed past.
# ---------------------------------------------------------------------------------------------
READ_THIS_FIRST = (
    "READ FIRST — THIS ARTIFACT DEVIATES FROM ITS PRE-REGISTRATION. The delivered forward-return "
    "estimator is NOT the quantity SPEC.md pre-registers. SPEC.md asks for 'the realized forward "
    "H=21 net return attributable to the decision (the env's own accounting; PV conservation "
    "exact)'; what is delivered is a COUNTERFACTUAL hold-through-H, ENTRY-COST-ONLY price return "
    "computed outside the env's compounding path. At the PRIMARY cost tier (cost0, fee=slip=0) "
    "the cost half of that difference is exactly nil, so the primary band is unaffected by it; "
    "the SECONDARY fee16/slip5 sign-persistence read is BIASED OPTIMISTIC (see "
    "pre_registration_deviations[0].bias_direction). Do not cite this artifact's numbers as the "
    "pre-registered quantity. SPEC.md was subsequently AMENDED to accept this estimator for the "
    "cost0 primary read (Amendment 1, 2026-08-19, D49(a), planner act, body sha256 8408970d…); "
    "the fee16/slip5 read is a disclosed, non-binding diagnostic under that ruling."
)

PRE_REGISTRATION_DEVIATIONS = [
    {
        "id": "PRD-1",
        "severity": "MAJOR",
        "spec_clause": f"{SPEC_PATH} — Design 1 ('Policy walks') and the bands' cost tiers",
        "spec_pre_registers": (
            "'the realized forward H=21 net return attributable to the decision (the env's own "
            "accounting; PV conservation exact)' — i.e. the quantity read out of the walk's own "
            "compounded PV path, with conservation asserted per walk."
        ),
        "delivered": (
            "A counterfactual event-study forward return computed in numpy OUTSIDE the env: "
            "forward_return[k,a] = direction[k,a] * (price[k+H]/price[k] - 1) - entry_cost, with "
            "entry_cost = (fee_bps + slip_bps)/1e4 charged ONCE. direction[k,a] is the env's own "
            "post-decision realized book side, so the DIRECTION is env accounting; the P&L "
            "attribution is not. It does not stop early when a later decision flips or flattens "
            "the position, and it never re-plays the walk's compounding."
        ),
        "cost0_impact": (
            "NEUTRAL at the primary tier. At cost0 fee_bps = slip_bps = 0, so entry_cost == 0 "
            "EXACTLY and no cost term of any kind enters the estimator — the cost-accounting half "
            "of this deviation cannot move the cost0 Delta, its CI, or the primary band."
        ),
        "bias_direction": (
            "OPTIMISTIC at the secondary fee16/slip5 tier. A decision that CLOSES a position to "
            "flat realizes direction == 0 and is EXCLUDED (NaN) from the pooled values, so the "
            "exit cost of that round trip is charged to NO decision at all: the opening decision "
            "pays one entry cost and the closing decision pays nothing. Total costs attributed "
            "are therefore strictly LESS than an honest walk replay would charge, which can only "
            "make the fee16/slip5 SIGN-PERSISTENCE half of the EDGE band read HEALTHIER (a Delta "
            "closer to / further above zero) than the pre-registered quantity would. It cannot "
            "manufacture an ANTI-EDGE."
        ),
        "why_not_amended": (
            "SPEC.md is a frozen pre-registration. Editing it so it matches the delivered "
            "estimator would be self-certification; the deviation is disclosed here instead and "
            "ruled on by the planner."
        ),
    },
    {
        "id": "PRD-2",
        "severity": "SCOPE LIMITATION",
        "spec_clause": f"{SPEC_PATH} — the 'PV conservation violation' instrument-fail check",
        "spec_pre_registers": "'PV conservation exact per walk' as an instrument-fail condition.",
        "delivered": (
            "A REALIZED-ALLOCATION conservation check, not a bar-by-bar PV reconciliation: the "
            "walk's own post-step allocated fraction (1 - cash/PV, read from the env state that "
            "produced the PV) must never exceed 1.0, AND must be a finite number. Because "
            "PV == cash + Σ position_values + reserved by construction, this is exactly the "
            "statement 'committed capital never exceeded portfolio value / cash never went "
            "negative' — the env's own conservation identity, computed under WHATEVER allocation "
            "semantics the run selected."
        ),
        "cost0_impact": "None — the check is a walk property, not part of the estimator.",
        "bias_direction": (
            "CONSERVATIVE in coverage, not in value: it can miss a hypothetical PV defect that "
            "keeps cash >= 0 and finite, and it cannot miss an overdraft/leverage defect. It is "
            "the conservative HALF of SPEC.md's claim, reported as such."
        ),
        "cycle3_correction": (
            "The 'cannot miss an overdraft/leverage defect' half of the sentence above was FALSE "
            "AS WRITTEN until cycle 3. The test was the bare ``allocated > 1.0 + tol``, and "
            "``NaN > x`` is False in IEEE/numpy, so a non-finite allocation was silently NOT a "
            "violation (measured: [0.5, nan, 0.9] -> 0 violations). No route to a NaN allocation "
            "is proven reachable today — a bankruptcy clamps cash to 0 — but a fail-closed guard "
            "may not rest on the unreachability of its own blind spot, and the disclosed guarantee "
            "was stated unconditionally. The check is now "
            "``(allocated > 1.0 + tol) | ~isfinite(allocated)``: an allocation that is not a "
            "finite number is not a bounded allocation, so it counts as a violation and the "
            "comparison reads INSTRUMENT FAIL."
        ),
        "why_not_amended": "Same reason as PRD-1.",
    },
]

# ---------------------------------------------------------------------------------------------
# SPEC Amendment 1's co-primary split: entries/exits vs maintenance re-trims.
# ---------------------------------------------------------------------------------------------
POSITION_CHANGE_SPLIT_DISCLOSURE = (
    "SPEC.md AMENDMENT 1, Co-primary: the primary subjects' action streams are mostly deadband "
    "re-trims of a standing book, so the pooled headline Delta mostly measures book QUALITY "
    "(close to the teacher's own signal), not DECISION quality. This artifact reports "
    "entries/exits (a sign change or a 0<->nonzero transition — the decision-quality read) and "
    "maintenance re-trims (the book's side is unchanged — completeness) as TWO SEPARATE "
    "co-primary comparisons, each independently calibrated and banded (never one band standing "
    "in for both). DISCLOSED LIMIT: `realized_book` is a 3-STATE encoding {flat, long, short} "
    "(trace.py's module docstring) — a change in WEIGHT MAGNITUDE within the same side (e.g. "
    "60% long -> 40% long, still long) is INVISIBLE to it. Under this encoding "
    "`position_delta != 0` <=> entry/exit and `non_hold & position_delta == 0` <=> re-trim "
    "EXACTLY, so the split is total and unambiguous over the recorded state — but 'maintenance "
    "re-trim' as reported here means 'side-preserving', NOT 'literally the same allocation'; "
    "magnitude-only re-trims are NOT separately resolved by this instrument, and this artifact "
    "does not claim a finer read than the encoding it was built on supports. If a finer "
    "realized-allocation signal becomes available on a future walk, it should be preferred and "
    "this disclosure updated to say so — nothing here should be read as though it already is."
)

# ---------------------------------------------------------------------------------------------
# D49(a) / SPEC Amendment 1: fee16/slip5 is a REPORTED DIAGNOSTIC ONLY, never a band condition.
# ---------------------------------------------------------------------------------------------
FEE_TIER_DIAGNOSTIC_DISCLOSURE = (
    "D49(a), operator-ruled 2026-08-19 (SPEC Amendment 1): the delivered event-study estimator "
    "is ACCEPTED for the cost0 PRIMARY read (provably cost-neutral there — see PRD-1's "
    "cost0_impact), but the fee16/slip5 sign-persistence check is DEMOTED to a reported, "
    "NON-BINDING diagnostic — it can never make or break a band. This is enforced in CODE, not "
    "merely in prose: report.band_for's `force_diagnostic` argument replaces any would-be "
    "SUBSTANTIVE band (EDGE/ANTI-EDGE/NULL) computed from a fee16/slip5-tier trace with "
    "BAND_FEE_TIER_DIAGNOSTIC, structurally, before it is ever returned — a fee16 comparison "
    "cannot pass this module's own banding function and come out as a citable band regardless "
    "of caller discipline. The TIER ITSELF is read the same way (fix pass, D51-re-cut review "
    "MAJOR 1): `tier_bp`/`fee_bps`/`slip_bps` are in report.REQUIRED_TRACE_PROVENANCE and are "
    "hard-subscripted off the graded trace's own provenance, so a document that OMITS its cost "
    "tier is REFUSED rather than read as cost0 — previously that one omission was enough to turn "
    "a genuine fee16 comparison into a citable EDGE, which made this paragraph's own claim "
    "false. The DELTA and CI are still computed and reported (the sign is the "
    "diagnostic's whole point), alongside PRD-1's disclosed bias direction: fee16 costs are "
    "understated by this estimator (an exit leg the estimator never charges), so a healthier- "
    "than-honest sign-persistence read is the expected DIRECTION of the diagnostic's own bias, "
    "not evidence the diagnostic corroborates cost0."
)

# ---------------------------------------------------------------------------------------------
# SPEC Amendment 1 / never-pooled: an OPEN ITEM, stated as one. Fix pass (D51-re-cut review
# MAJOR 2): the earlier wording called this enforcement "structural", which it is not.
# ---------------------------------------------------------------------------------------------
NEVER_POOLED_OPEN_ITEM = (
    "OPEN ITEM (not a closed guarantee). SPEC.md Amendment 1 requires the 887 A/D subject class "
    "never to be pooled with the imprint primaries. What EXISTS today: (1) each report is built "
    "from exactly ONE subject trace and stamps that trace's own `subject_class` and "
    "`in_d47_evidence_base`, so no single report can pool anything — report.py has no "
    "multi-report aggregator at all, and its CLI takes one --subject; (2) "
    "`report.assert_never_pooled` is the enforcement point, implemented and unit-tested, that "
    "raises when two distinct D47-evidence classes would be combined. What does NOT exist: any "
    "live call to it. It is a library function a future D47 aggregator MUST call, and nothing in "
    "this module calls it, so pooling is today PREVENTED BY THE ABSENCE OF AGGREGATION CODE, not "
    "by a structural guard — do not read it as 'pooling is structurally impossible'. Two further "
    "gaps a caller must own until an aggregator lands: (a) `subject_class` is OPTIONAL on a "
    "trace and `None` is filtered out of the comparison, so a MISTAGGED or untagged primary "
    "sitting beside an 887 A/D report would not trip the guard either (an untagged subject is "
    "reported as NOT in the D47 evidence base, which is the only protection there); and (b) "
    "combining Deltas by hand, outside this module, bypasses all of the above. When the "
    "aggregator is built, `assert_never_pooled` must be called on its inputs BEFORE any "
    "averaging, and this disclosure replaced with the structural claim it can then support."
)

# ---------------------------------------------------------------------------------------------
# SPEC Amendment 1: per-decision net exposure + forward market return of the held assets.
# ---------------------------------------------------------------------------------------------
NET_EXPOSURE_DISCLOSURE = (
    "SPEC.md AMENDMENT 1, Design 1: every trace now records, per decision, `net_exposure` "
    "(mean realized book side over assets, in [-1, 1]: fully short .. fully long) and, per "
    "held (decision, asset) cell, `market_return` (the UNDIRECTED forward H-return of that "
    "asset over the SAME horizon/mask as `forward_return` — trace.py's compute_forward_returns "
    "docstring). Purpose: separate a DIRECTIONAL ARTIFACT (a long book that happens to realize "
    "well because the broad market rallied over the walk, something the state-blind "
    "matched-activity null does NOT share, since it draws each asset's SIDE from the subject's "
    "own marginal but carries no information about WHEN a rally happens) from TIMING skill (the "
    "signed, cost-inclusive `forward_return` the primary Delta is built from). `market_return_"
    "delta` (subject mean market_return minus null mean market_return, both over the exact same "
    "held cells as the headline comparison) is the size of that separable directional artifact: "
    "a large `market_return_delta` alongside a small or NULL signed Delta corroborates 887's "
    "own bull-blindness finding rather than contradicting it; a positive market_return_delta "
    "does NOT by itself make the signed Delta an artifact (the null's own book realizes the SAME "
    "market during the SAME walk — see the matched-activity construction — so this is a "
    "diagnostic to READ alongside the band, never a correction applied TO it, and it does not "
    "gate or adjust `band` in any way)."
)

# ---------------------------------------------------------------------------------------------
# SPEC.md:46's third matched moment. Promised, previously never computed — now computed AND
# disclosed, because a matched MEAN is not a matched DISTRIBUTION.
# ---------------------------------------------------------------------------------------------
BOOK_WIDTH_DISCLOSURE = (
    "SPEC.md:46 promises the null matches the subject's activity rate, long/short mix AND "
    "AVERAGE BOOK WIDTH. All three are now computed and reported (book width = number of "
    "non-flat slots in the realized book at a decision). DISCLOSED LIMIT: the null draws each "
    "asset's action i.i.d. from that asset's own marginal, which preserves every PER-ASSET "
    "marginal (hence the MEAN book width, up to sampling noise) but DESTROYS cross-asset "
    "correlation. The book-width DISTRIBUTION therefore differs by construction even when the "
    "means agree: the subject's book is bursty/clustered (correlated opens), the null's is "
    "CLT-concentrated around its mean. This does not invalidate the null — state-blindness is "
    "precisely the property being contrasted against — but a reader must not read 'matched' as "
    "'distributionally identical'. A book-width mean mismatch is reported as a WARNING, never "
    "folded into instrument_fail_reasons: SPEC.md's instrument-fail list is pre-registered and "
    "closed (activity/mix mismatch > 0.5pp, PV violation, < 30 non-HOLD decisions), and this "
    "module does not get to add conditions to a frozen contract."
)

# ---------------------------------------------------------------------------------------------
# CYCLE-3 finding 2 — THE MEASURED CALIBRATION OF THE BLOCK BOOTSTRAP.
#
# The shipped power floor (MIN_BOOTSTRAP_BLOCKS = 2) killed the LITERALLY degenerate case and the
# disclosure claimed only non-degeneracy — it never said what the coverage of a NON-degenerate CI
# actually is, and the regression gate asserted only `ci_hi > ci_lo`. So the module was banding at
# a power it had never measured. It is measured here.
#
# HOW (reproducible): report.block_bootstrap_delta_ci itself is called (no re-implementation) on
# two INDEPENDENT samples, so the true Delta is exactly 0; coverage = the fraction of trials whose
# 95% CI contains 0. block = 21 (SPEC.md), n_boot = 2000, 1000 trials per row, the data seed and
# the bootstrap seed both varied per trial. Scenario IID: each side i.i.d. N(0,1). Scenario
# OVERLAPPING: each observation is a rolling sum of 21 i.i.d. innovations, so consecutive
# observations share up to 20 of them — the autocorrelation structure the real overlapping-horizon
# forward returns have, and the case a block bootstrap exists for. Monte-Carlo SE ~0.7-1.3pp.
NULL_COVERAGE_TABLE = {          # blocks per side -> measured coverage, scenario IID
    2: 0.785, 3: 0.863, 4: 0.886, 5: 0.897, 6: 0.899, 8: 0.914, 10: 0.933,
    12: 0.926, 15: 0.933, 20: 0.935, 25: 0.938, 30: 0.944, 40: 0.947,
}
NULL_COVERAGE_TABLE_OVERLAPPING = {   # scenario OVERLAPPING — the structure the real data HAS
    2: 0.567, 4: 0.778, 8: 0.806, 10: 0.851, 12: 0.854, 15: 0.863, 20: 0.863,
    30: 0.888, 40: 0.876,
}

# The i.i.d. coverage the banding bar must still exhibit — a regression tripwire on the
# calibration, NOT the rule that derives the bar (that rule is the plateau rule below).
NULL_COVERAGE_BAND_MIN = 0.93

# DERIVED, NOT ASSERTED — THE PLATEAU RULE, applied to BOTH measured scenarios:
#   the smallest measured block count B such that, in each scenario, coverage at B and at EVERY
#   larger measured count is within 2pp of that scenario's own maximum measured coverage.
# (Within 2pp because the Monte-Carlo SE is 0.7-1.6pp; requiring more would be reading noise. The
# "and every larger count" clause stops one lucky row from buying a cheaper bar.)
#   * IID       — max 94.7% (40 blocks) => need >= 92.7%. 10 measures 93.3% but 12 falls back to
#                 92.6%, so 10 and 12 both fail the clause; 15/20/25/30/40 are all >= 92.7%  => 15.
#   * OVERLAP   — max 88.8% (30 blocks) => need >= 86.8%. 15 and 20 measure 86.3%, below it; 30
#                 (88.8%) and 40 (87.6%) clear it                                            => 30.
# The bar is the MORE demanding of the two, because the overlapping scenario is the one the real
# forward returns actually live in: 30 full blocks per side, n >= 630.
CALIBRATED_BOOTSTRAP_BLOCKS = 30

# ESCALATION, recorded in the artifact rather than decided here: the plateau in the OVERLAPPING
# scenario is ~88%, not 95%, and MORE DATA DOES NOT CLOSE IT — coverage is flat from 30 to 40
# blocks (840 observations/side). So above the bar this estimator is still anti-conservative by
# ~6-7pp under realistic dependence, and NO sample-size bar can fix that: the cause is that the
# block LENGTH equals the dependence length (block = 21 against forward windows that overlap by up
# to 20), and a block bootstrap needs a block substantially LONGER than the dependence it has to
# reproduce. The remedy is therefore the block length or the interval method (studentized /
# calibrated / double bootstrap), not the sample size. SPEC.md PRE-REGISTERS "block = 21, 10k
# resamples" and is FROZEN, so this build does not get to change it: the residual is measured,
# disclosed here, and escalated to the planner as a SPEC-level question.
NULL_COVERAGE_ESCALATION = (
    "ESCALATED, NOT DECIDED HERE. Above the banding bar this CI is STILL anti-conservative under "
    "the dependence the real data has: measured coverage plateaus at ~88% (not 95%) in the "
    "overlapping-horizon scenario and does NOT improve from 30 to 40 blocks, so no sample-size bar "
    "removes it. The cause is structural — SPEC.md pre-registers block = 21 while consecutive "
    "decisions' forward windows overlap by up to 20, i.e. the block length EQUALS the dependence "
    "length, and a block bootstrap needs a block substantially longer than the dependence it must "
    "reproduce. The remedy is the block length or the interval method (studentized / calibrated / "
    "double bootstrap), NOT more decisions. SPEC.md is a frozen pre-registration, so this build "
    "does not change either: a band emitted at or above the bar should be read as ~88%-covering, "
    "not 95%-covering, and the planner rules on whether that is good enough for D47 or whether the "
    "interval method needs re-specifying."
)


# ---------------------------------------------------------------------------------------------
# AMENDMENT 1 / D51 RE-CUT — THE CALIBRATED-WIDTH CORRECTION (remedy (iii) of the pre-registered
# interval-method ladder). D49's escalation measured that the block bootstrap's coverage
# PLATEAUS at ~88% under realistic overlapping dependence and never reaches the nominal 95% (or
# even the 93% Amendment 1 asks for) at ANY sample size, because block=21 (SPEC.md, frozen)
# equals the dependence length. The fixed-bar approach above (CALIBRATED_BOOTSTRAP_BLOCKS) could
# only ever answer "is n big enough" — never "is the METHOD calibrated" — and the plateau proves
# it structurally cannot be, at any n. This table is the replacement: a per-BLOCK-COUNT widening
# factor k, measured by Monte Carlo through ``report.block_bootstrap_delta_ci`` itself (never a
# re-implementation — see calibration.py) under a TRUE NULL (independent sides, delta=0 exactly),
# then INDEPENDENTLY RE-VALIDATED on a fresh trial batch (never the same draws the factor was
# chosen from) before being trusted. ``python calibration.py`` regenerates this table end to end;
# the numbers below are its most recent run (parameters and date in the header comment beside the
# table itself).
#
# A calibrated k is only as good as the scenario it was calibrated against. The OVERLAPPING
# scenario is a SYNTHETIC model (each observation a rolling sum of H=21 i.i.d. innovations) --
# real forward returns additionally carry fat tails, heteroskedasticity, cross-sectional
# correlation and regime structure this model does not. Calibrating does NOT eliminate the
# interval-method caveat NULL_COVERAGE_ESCALATION named; it RELOCATES it from "coverage is ~88%"
# to "coverage is nominal UNDER THIS DEPENDENCE MODEL" — see CALIBRATED_WIDTH_DISCLOSURE below,
# carried inline in every report exactly like its predecessor.
CALIBRATION_TARGET_COVERAGE = 0.93          # SPEC.md [A1]: ">= 93% two-sided coverage"

# MEASURED 2026-08-20, IN-LOOP PRECISION (disclosed, not hidden — see CALIBRATION_PRECISION_
# DISCLOSURE below for why this run is deliberately CHEAP and what that costs). Command:
#   python calibration.py --n-trials-calib 200 --n-trials-valid 200 --n-boot 400 \
#     --seed 31337 --calibration-margin 0.03 --block-counts 2 3 4 5 6 10 20 80
# Runtime: 1m35s. Block counts are the strata SPEC.md Amendment 1 actually names: the thin
# co-primary splits (2, 3, 4), the OOS binding leg's ~5-6 blocks (5, 6), one mid anchor (10, 20),
# and the sanity leg's pooled ~79 blocks (80). ``calibrated_k_for``'s floor-lookup (max-k among
# achieved entries at or below n_blocks) stays fail-safe over the gaps in this sparser table by
# construction — that is the design property that makes a restricted block-count list safe here.
# ALL EIGHT probed counts ACHIEVED the target with margin (validation-batch overlapping coverage
# 0.94-0.975, against a 0.93 bar); k ranges 3.49 (2 blocks) down to 1.38 (80 blocks), consistent
# in shape and magnitude with an independent cross-check across three dependence models
# (Gaussian rolling-sum / AR(1) phi=0.909 / fat-tailed t3) reporting k*~3.25 at 2 blocks, ~2.0 at
# 3-5, ~1.5 at 30 -- this table's own numbers sit close to and mostly above that cross-check
# (MORE conservative, not less), which is expected: this harness calibrates to target+0.03
# margin and validates on an INDEPENDENT fresh batch, both of which push k up relative to a
# same-batch measurement.
# n_trials=200/200 per scenario is a CHEAP, DISCLOSED precision, not a final measurement: MC SE
# on a coverage estimate near 0.93-0.97 at n=200 is on the order of 1.5-2.5pp (binomial
# se = sqrt(p(1-p)/n)). CALIBRATION_PRECISION_DISCLOSURE states this explicitly, in every
# artifact that carries this table. A HIGHER-PRECISION RE-RUN (larger n_trials/n_boot, and/or
# the block counts skipped here: 8, 12, 15, 25, 30, 40, 50, 65, 100) IS REQUIRED before the real
# leg launches — this is scheduled, not implied: SPEC.md's own requirement is certification
# "before the real leg runs", and the real leg itself is planner-scheduled on gpu-server per
# the operator's standing 2026-08-19 guidance (no CPU-heavy work on the workstation).
CALIBRATION_TABLE = {
    2: {"k": 3.494949538093774, "achieved": True, "validation_coverage_overlapping": 0.94,
       "validation_coverage_iid": 1.0, "validation_n_overlapping": 200,
       "validation_n_iid": 200, "reason": None},
    3: {"k": 2.4018255532637416, "achieved": True, "validation_coverage_overlapping": 0.96,
       "validation_coverage_iid": 0.995, "validation_n_overlapping": 200,
       "validation_n_iid": 200, "reason": None},
    4: {"k": 2.275514112668692, "achieved": True, "validation_coverage_overlapping": 0.975,
       "validation_coverage_iid": 1.0, "validation_n_overlapping": 200,
       "validation_n_iid": 200, "reason": None},
    5: {"k": 2.064163704869456, "achieved": True, "validation_coverage_overlapping": 0.97,
       "validation_coverage_iid": 1.0, "validation_n_overlapping": 200,
       "validation_n_iid": 200, "reason": None},
    6: {"k": 2.0551280013786157, "achieved": True, "validation_coverage_overlapping": 0.955,
       "validation_coverage_iid": 0.99, "validation_n_overlapping": 200,
       "validation_n_iid": 200, "reason": None},
    10: {"k": 1.532438458917841, "achieved": True, "validation_coverage_overlapping": 0.955,
        "validation_coverage_iid": 0.985, "validation_n_overlapping": 200,
        "validation_n_iid": 200, "reason": None},
    20: {"k": 1.463321901624038, "achieved": True, "validation_coverage_overlapping": 0.975,
        "validation_coverage_iid": 1.0, "validation_n_overlapping": 200,
        "validation_n_iid": 200, "reason": None},
    80: {"k": 1.37601365581144, "achieved": True, "validation_coverage_overlapping": 0.96,
        "validation_coverage_iid": 0.985, "validation_n_overlapping": 200,
        "validation_n_iid": 200, "reason": None},
}

CALIBRATION_PRECISION_DISCLOSURE = (
    "CALIBRATION_TABLE was measured at CHEAP, IN-LOOP precision (200 calibration trials + 200 "
    "validation trials per scenario, n_boot=400, 8 block counts spanning the strata SPEC.md "
    "Amendment 1 names) so the factory build did not run a multi-hour Monte-Carlo job inside "
    "the implementation loop. Every probed count ACHIEVED >= 93% validated coverage with real "
    "margin (0.94-0.975), consistent with an independent cross-check across three dependence "
    "models. But Monte-Carlo SE at n=200 per scenario is ~1.5-2.5pp, and only 8 of the block "
    "counts a real leg could present were probed (calibrated_k_for's floor-lookup covers the "
    "gaps fail-safely, but a gap is still a gap). A HIGHER-PRECISION RE-RUN is REQUIRED before "
    "the real leg launches (SPEC.md [A1]: certification 'before the real leg runs'); that "
    "re-run is planner-scheduled on gpu-server, not implied by this table's presence."
)


def calibrated_k_for(n_blocks, table=None):
    """NEAREST-BELOW floor-lookup (same convention as ``measured_null_coverage`` below, and
    ``build_calibration_table``'s theoretical expectation: k DECREASES as n_blocks increases,
    since more blocks means less bootstrap sampling variance to correct for). Among every table
    entry whose block count is <= ``n_blocks`` AND whose calibration was measured ACHIEVED
    (independently-validated coverage >= CALIBRATION_TARGET_COVERAGE under the overlapping
    scenario), take the one with the LARGEST block count (nearest below), not the largest k.

    BUG FIXED (measured, not assumed): an earlier version took the MAX k among all qualifying
    entries instead of the nearest-below one, reasoning that Monte-Carlo noise could make the
    k-vs-blocks curve non-monotonic. Measured consequence: because k is DECREASING with n_blocks
    in every table this harness has produced, "take the max" silently degenerates to "always use
    the SMALLEST calibrated block count's k" for every query, however large — e.g.
    ``calibrated_k_for(80)`` returned the k measured at 2 blocks (3.49), not the k measured AT 80
    blocks (1.38), discarding the more precise, less-widened measurement the table exists to
    provide. Nearest-below is both the historically-established convention in this module
    (matches ``measured_null_coverage``) and the correct one given the table's own measured
    trend; local Monte-Carlo noise between adjacent entries is a smaller residual risk now that
    every entry validates with real margin (0.94-0.975 against a 0.93 target — see
    CALIBRATION_PRECISION_DISCLOSURE), not the borderline (~0.90-0.93) case a max-based guard
    was originally reacting to. Returns ``(k, achieved, source_n_blocks)``, or
    ``(None, False, None)`` if no achieved entry at or below ``n_blocks`` exists (e.g. block
    count below the smallest calibrated entry, or every entry at-or-below failed to achieve
    target)."""
    table = CALIBRATION_TABLE if table is None else table
    candidates = [(nb, e) for nb, e in table.items() if nb <= n_blocks and e.get("achieved")]
    if not candidates:
        return None, False, None
    nb_src, entry = max(candidates, key=lambda kv: kv[0])       # nearest-below: largest nb <= n_blocks
    return float(entry["k"]), True, int(nb_src)


CALIBRATED_WIDTH_DISCLOSURE = (
    "AMENDMENT 1's INTERVAL-METHOD REMEDY (D49's escalation): the pre-registered block bootstrap "
    "(block=21, 10k resamples) is measured to PLATEAU at ~88% two-sided coverage under realistic "
    "overlapping dependence and to NEVER reach nominal at any sample size (block=21 equals the "
    "dependence length; see NULL_COVERAGE_ESCALATION). The remedy applied here is (iii) of the "
    "pre-registered ladder: every reported CI is WIDENED by a factor k, chosen per block count by "
    "Monte Carlo simulation through this instrument's own block_bootstrap_delta_ci (never a "
    "re-implementation) and INDEPENDENTLY RE-VALIDATED on a fresh trial batch, so that the "
    "widened interval reaches >= " + f"{100.0*CALIBRATION_TARGET_COVERAGE:.0f}%" +
    " two-sided coverage under the realistic-dependence (overlapping) scenario at THAT block "
    "count. Remedies (i) (block length > dependence length) and (ii) (subsampled non-overlapping "
    "windows at stride H) were considered first, per the pre-registered order, and are "
    "arithmetically inapplicable to the thinnest strata this instrument reports (the OOS binding "
    "leg's ~5-6 blocks cannot simultaneously satisfy block > 21 and carry enough blocks to "
    "resample; stride-H subsampling leaves too few independent observations for any interval "
    "method). CRITICAL CAVEAT, not eliminated by calibrating, only RELOCATED: the widening factor "
    "is calibrated against a SYNTHETIC dependence model (each observation a rolling sum of H "
    "i.i.d. innovations) — real forward returns additionally carry fat tails, "
    "heteroskedasticity, cross-sectional correlation and regime structure this model does not, so "
    "a calibrated band should be read as 'nominal under this dependence model', not "
    "unconditionally nominal. A stratum whose block count has NO independently-validated k "
    "achieving the target (or one whose required k exceeded the harness's own sanity cap, "
    "refused as no-longer-a-calibration) ships LOW-POWER — CI reported, band refused — never a "
    "manufactured band. A wide, calibrated interval that spans zero is a legitimate NULL "
    "(SPEC.md's own publishable band), not a defect of this method."
)


def measured_null_coverage(n_blocks, scenario="iid"):
    """The MEASURED true-null coverage to quote for an observed block count: the table entry at or
    below ``n_blocks`` (conservative — never credit a sample with a larger sample's coverage).
    ``scenario`` selects the i.i.d. table (the optimistic bound) or the overlapping-horizon table
    (the structure the real forward returns have). ``None`` when the count is below the smallest
    measured row — there is no measurement to quote, and such a sample is refused a CI anyway."""
    table = (NULL_COVERAGE_TABLE_OVERLAPPING if scenario == "overlapping"
             else NULL_COVERAGE_TABLE)
    try:
        b = int(n_blocks)
    except (TypeError, ValueError):
        return None
    keys = [k for k in sorted(table) if k <= b]
    if not keys:
        return None
    return table[keys[-1]]


def _coverage_table_text(table=None):
    return "; ".join(f"{k} blocks (n={21 * k}): {100 * v:.1f}%"
                     for k, v in sorted((table or NULL_COVERAGE_TABLE).items()))


BOOTSTRAP_POWER_DISCLOSURE_CALIBRATION = (
    "MEASURED CALIBRATION (cycle-3 finding 2) -- HISTORICAL RECORD, SUPERSEDED BY AMENDMENT 1. "
    "Clearing the degeneracy floor is NOT the same guarantee as having the nominal 95% coverage; "
    "the pre-cycle-3 disclosure conflated them, arguing only that a CI at >= 2 blocks is "
    "non-degenerate while the regression gate checked only that ci_hi > ci_lo. Cycle 3 measured "
    "coverage, by calling THIS module's own block_bootstrap_delta_ci on two independent samples "
    "(true Delta = 0 exactly), block = 21, n_boot = 2000, 1000 trials per row, MC SE ~0.7-1.6pp, "
    "in TWO scenarios. SCENARIO IID (each side i.i.d. N(0,1) -- the optimistic bound): "
    + _coverage_table_text(NULL_COVERAGE_TABLE) + ". "
    "SCENARIO OVERLAPPING (each observation a rolling sum of 21 innovations, so consecutive "
    "observations share up to 20 -- the autocorrelation the real overlapping-horizon forward "
    "returns HAVE): " + _coverage_table_text(NULL_COVERAGE_TABLE_OVERLAPPING) + ". "
    "READ THE FIRST ROWS: at 2 blocks, RAW (uncalibrated) coverage of the nominal 95% interval is "
    "78.5% i.i.d. and 56.7% under realistic dependence -- so between one in five and over four in "
    "ten comparisons would band EDGE or ANTI-EDGE UNDER A TRUE NULL if banded at that power "
    "unadjusted, against a nominal one in twenty. CYCLE 3'S RESPONSE was a FIXED BAR "
    "(CALIBRATED_BOOTSTRAP_BLOCKS = 30 full blocks, n >= 630) below which a comparison read "
    "LOW-POWER (CI reported, band refused). THAT BAR IS NOW DEAD CODE, KEPT ONLY AS A REGRESSION "
    "REFERENCE: a fixed n-floor cannot fix a coverage PLATEAU (NULL_COVERAGE_ESCALATION measured "
    "the overlapping scenario topping out at ~88%, never reaching 95%, at ANY block count "
    "including far above 30) -- the bar was buying a number that was never going to arrive. "
    "AMENDMENT 1 replaced it with the calibrated-width correction: see "
    "CALIBRATED_WIDTH_DISCLOSURE for the live mechanism. " + NULL_COVERAGE_ESCALATION
)

BOOTSTRAP_POWER_DISCLOSURE_ZERO_WIDTH = (
    "ZERO-WIDTH INTERVALS ARE A SEPARATE, ALWAYS-LIVE CONDITION (cycle-3 finding 3, REBUILT under "
    "D51). The block floor guarantees 'enough BLOCKS'; it does not guarantee 'enough VARIANCE', "
    "and an earlier comment in report.py claimed 'Nothing here can emit a zero-width interval', "
    "which was false. Cycle 3's own fix then only caught BIT-IDENTICAL inputs "
    "(``not (hi > lo)``) -- a NEAR-duplicate input (e.g. the same values plus per-block jitter at "
    "1e-9 or 1e-6 scale, still bit-DISTINCT) produces a real, nonzero interval width that is "
    "STILL degenerate in every way that matters, and read EDGE with the same fake certainty. "
    "MEASURED: this module's CI width scales LINEARLY with its own input's scale (a circular "
    "block bootstrap is homogeneous of degree 1) -- so no test built only from the sample's own "
    "dispersion or magnitude can ever separate a genuinely-tiny-but-real effect from an "
    "artificially-shrunk one; the guard now compares the FINAL (possibly calibration-widened) "
    "interval's width against ZERO_WIDTH_FLOOR, a fixed anchor EXTERNAL to the sample -- one "
    "basis point (1e-4), this instrument's own smallest cost-relevant unit (report.py's module "
    "comment above ZERO_WIDTH_FLOOR carries the full argument and measured margins). The "
    "plausible real route to this condition remains a per-regime bucket of highly autocorrelated "
    "/ near-duplicate overlapping-horizon cells, or duplicated/stale price data. Any comparison "
    "whose FINAL interval comes back at or below that floor is labelled 'ZERO-WIDTH-CI' and "
    "excluded from banding, regardless of how many blocks it carries or whether it was widened."
)

BOOTSTRAP_POWER_DISCLOSURE = (
    "UNDER-POWERED SAMPLES ARE A REPORTED CONDITION, NEVER A BAND. The circular block bootstrap "
    "at block = 21 (SPEC.md) cannot produce ANY variation when a side has n <= block: "
    "n_blocks = ceil(n/block) == 1, and one circular block of length >= n covers every index "
    "exactly once, so each 'resample' is a mere ROTATION of the full sample — and a mean is "
    "rotation-invariant. All 10,000 resamples then return the identical mean and the CI collapses "
    "to a genuinely ZERO-WIDTH interval, which would read EDGE or ANTI-EDGE with total, fake "
    "certainty and could NEVER read NULL. THE DEGENERACY FLOOR: a side must carry at least "
    "MIN_BOOTSTRAP_BLOCKS (2) FULL blocks — n >= 42 observations — before a CI is emitted at all. "
    "Two full blocks is the minimum at which the resample genuinely RECOMBINES blocks rather than "
    "rotating one block (at n == block + 1 the variation comes from a single tail observation). "
    "Anything thinner is labelled 'INSUFFICIENT-N', reports its point-estimate delta with "
    "ci_lo/ci_hi = null, and is EXCLUDED from banding. "
    "THAT FLOOR IS NOT THE BANDING BAR, AND MUST NOT BE READ AS ONE — it buys non-degeneracy, "
    "nothing more; the measured banding bar is stated in the calibration paragraph below. "
    "READ POWER IN BLOCKS, NOT IN CELLS: what this bootstrap consumes is BLOCKS of 21, so every "
    "report states n_blocks_subject / n_blocks_null (floor(n/block)) beside the raw n, measured "
    "from the data at hand. Do NOT reason about power from SPEC.md's own thin-n sentence: that "
    "sentence ('non-HOLD decisions ... in the low hundreds per seed') is a pre-registered estimate "
    "that a 2026-08-19 planner/operator audit measured WRONG — the primary subjects carry ~95k "
    "non-HOLD action CELLS per seed pre-blocking, and the binding constraint is the ~80 blocks of "
    "21 decisions, not the cell count. This module therefore derives and reports the block count "
    "rather than repeating that figure. Per-regime buckets are still where thinness bites, because "
    "a bucket's block count, not the pooled cell count, is what has to clear the bar. "
    "Both bars are STRICTER than SPEC.md's pre-registered n >= 30 instrument-fail floor; the "
    "pre-registered floor is still enforced separately and unchanged. "
    + BOOTSTRAP_POWER_DISCLOSURE_CALIBRATION + " " + BOOTSTRAP_POWER_DISCLOSURE_ZERO_WIDTH
    + " " + CALIBRATED_WIDTH_DISCLOSURE
)
