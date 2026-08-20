#!/usr/bin/env python3
"""TASK_892 / action-EV audit — scope 3: the comparison report.

Reads one SUBJECT trace (trace.py, --decision-rule argmax or the search-operator secondary)
and one-or-more NULL traces (--decision-rule random-matched, distinct --eval-seed each) and
computes the pre-registered quantity (SPEC.md): Delta = mean realized forward H-return after
the subject's non-HOLD decisions minus the same quantity over the null's non-HOLD decisions, a
block-bootstrap 95% CI on Delta (block=21, 10k resamples — SPEC.md's exact numbers), optional
per-regime-bucket breakdown (831-EXT windows, via eval_vector.load_ext_windows), and the
instrument-fail checks SPEC.md names explicitly: null activity-rate/long-short-mix mismatch
> 0.5pp, < 30 non-HOLD decisions pooled, and any PV/budget violation.

This module computes the mechanical BAND LABEL (EDGE / ANTI-EDGE / NULL / INSTRUMENT FAIL /
DIAGNOSTIC-ONLY) the SPEC's bands (+ Amendment 1's fee-tier carve-out) define, but that label is
NOT a program verdict — ruling on it (seeding D47) is explicitly the planner's job (TASK_892's
"Out of scope"). Treat `band` as what the numbers say under the pre-registered rule, not as this
module claiming authority to close the question.

FIVE NON-BANDS (D51 re-cut: was four). None is one of SPEC.md's three publishable outcomes; each
means "this comparison is not interpretable [as a band]", and each EXCLUDES the comparison from
banding at the top level, in every per-regime bucket, AND in the [A1] co-primary split:
  * ``INSTRUMENT FAIL`` — SPEC.md's own list (activity/mix mismatch > 0.5pp, PV/budget violation,
    < 30 non-HOLD decisions pooled).
  * ``INSUFFICIENT-N`` — the sample is below the block bootstrap's DEGENERACY floor
    (``MIN_BOOTSTRAP_BLOCKS`` full blocks per side). At n <= block the circular block bootstrap is
    DEGENERATE: every resample is a rotation of the whole sample and a mean is rotation-
    invariant, so the CI collapses to zero width and any nonzero delta would read EDGE or
    ANTI-EDGE with fake certainty and could never read NULL.
  * ``LOW-POWER`` — above the degeneracy floor, but no measured, independently-validated
    calibration factor (Amendment 1's interval-method remedy — disclosures.CALIBRATION_TABLE)
    reached the target coverage at this block count. The CI is still reported, with its measured
    coverage; the BAND is refused.
  * ``ZERO-WIDTH-CI`` — the FINAL (possibly calibration-widened) interval has no width relative
    to this instrument's own smallest cost-relevant unit (D51 rebuild: a bit-identity test used
    to catch only literally-equal floats; see report.ZERO_WIDTH_FLOOR).
  * ``DIAGNOSTIC-ONLY (fee16/slip5, non-binding per D49(a))`` — [A1] the graded trace ran at a
    nonzero cost tier. The delta/CI are still computed and reported (that IS the diagnostic); the
    band is structurally replaced so a fee16 trace can never make or break a substantive verdict
    (see ``band_for``'s ``force_diagnostic``).
``disclosures.BOOTSTRAP_POWER_DISCLOSURE`` / ``CALIBRATED_WIDTH_DISCLOSURE`` /
``FEE_TIER_DIAGNOSTIC_DISCLOSURE`` carry the full arguments, inline in every emitted report.

EVERY REPORT CARRIES ITS PRE-REGISTRATION DEVIATIONS INLINE, FIRST
(``disclosures.READ_THIS_FIRST`` / ``pre_registration_deviations``): the delivered forward-return
estimator is NOT the quantity SPEC.md pre-registers verbatim; SPEC Amendment 1 (2026-08-19,
D49(a)) accepts it for the cost0 primary read and demotes the fee16/slip5 read to a disclosed
non-binding diagnostic. Read that block before citing any number here.
"""
import argparse
import json
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
EXP_DIR = os.path.dirname(os.path.abspath(__file__))
if EXP_DIR not in sys.path:
    sys.path.insert(0, EXP_DIR)

import disclosures                                            # noqa: E402  (path set above)
# CYCLE-3 BLOCKER 1: this module must VALIDATE the schema of every trace it reads, so it needs
# the one authoritative TRACE_SCHEMA constant rather than a second copy of the string. EXP_DIR is
# at sys.path[0] above, so this resolves to experiments/892_action_ev/trace.py, not stdlib trace.
import trace as trace_mod                                     # noqa: E402  (path set above)

REPORT_SCHEMA = "892/action-ev-report/v3"        # v3 (D51 re-cut, Amendment 1): calibrated-width
# correction fields, the entry/exit vs re-trim co-primary split, net_exposure/market_return
# diagnostics, and the --leg binding/sanity structural marker below.
ACTIVITY_MISMATCH_TOL_PP = 0.5
MIN_NON_HOLD_DECISIONS = 30

# [A1] SPEC.md Amendment 1's subject classes (trace.py's closed vocabulary — the same tuple,
# imported rather than redeclared so the two modules cannot drift). The 887 A/D subject class must
# NEVER be pooled with the imprint primaries; the 832 contrast leg is EXCLUDED from the D47
# evidence base entirely (inherits the unresolved 882 epoch confound).
#
# FIX PASS (D51 re-cut review, MAJOR 2) — WHAT IS AND IS NOT ENFORCED HERE. The 832/search
# EXCLUSION is structural: it is computed from the graded trace's own provenance on every report
# (in_d47_evidence_base, below, stamped into the returned dict). The NEVER-POOLED rule is NOT:
# `assert_never_pooled` exists and is tested, but nothing in this module calls it, because nothing
# in this module aggregates across reports at all (one --subject per invocation). The earlier
# wording here claimed both were "enforced STRUCTURALLY", which was false for pooling and is now
# stated as the OPEN ITEM it is — disclosures.NEVER_POOLED_OPEN_ITEM, carried inline in every
# emitted report so a reader is never left inferring a guarantee this module does not provide.
SUBJECT_CLASSES = trace_mod.SUBJECT_CLASSES
D47_EVIDENCE_CLASSES = (trace_mod.SUBJECT_CLASS_PRIMARY, trace_mod.SUBJECT_CLASS_887_A,
                        trace_mod.SUBJECT_CLASS_887_D)
# search-secondary is a CAPACITY-CEILING probe (SPEC.md [A1]: "an R-step LOOKAHEAD ... NOT a
# deployable comparison"), reported separately; 832 is excluded for the epoch-confound reason.
D47_EXCLUDED_CLASSES = (trace_mod.SUBJECT_CLASS_SEARCH_SECONDARY,
                        trace_mod.SUBJECT_CLASS_832_CONTRAST)


def in_d47_evidence_base(subject_class):
    """Whether a report's subject_class is part of the D47 evidence base. ``None`` (an untagged
    trace) is NOT in the evidence base — an untagged subject cannot be assumed primary."""
    return subject_class in D47_EVIDENCE_CLASSES


def assert_never_pooled(reports):
    """[A1] The 887 A/D subject class must never be pooled with the imprint primaries (SPEC.md
    Amendment 1: "one 887 A-arm ... and one 887 D-arm ... [never] pooled with the imprint
    primaries"). Given a list of report ``compare()`` dicts (or bare subject_class strings),
    raises if more than one DISTINCT subject_class among the D47 evidence classes appears.

    CALLED BY NOTHING IN THIS MODULE — SAY SO PLAINLY (fix pass, D51 re-cut review MAJOR 2). This
    is the enforcement point a future D47 aggregator MUST call before averaging/combining Deltas
    across reports; there is no such aggregator today (``main`` takes exactly one ``--subject``
    and ``compare`` grades exactly one subject trace), so this function has no live caller and
    pooling is prevented today by the ABSENCE of aggregation code, not by a guard that runs. Do
    not describe it as making pooling "structurally impossible": it is a checkable rule with a
    tested implementation, waiting to be wired in. The report itself carries
    ``disclosures.NEVER_POOLED_OPEN_ITEM`` saying this, and that disclosure also names the two
    residual gaps this function cannot close on its own: an untagged/MISTAGGED subject
    (``subject_class`` is optional, and ``None`` is filtered out below, so an untagged primary
    beside an 887 A/D report does not trip it — it is merely stamped as NOT in the evidence
    base), and any hand-combination of Deltas outside this module.

    A single report's own subject vs its OWN matched nulls is never "pooling" in this sense
    (nulls carry no subject_class of their own — see trace.py's build_trace_doc); this function
    is for combining MULTIPLE subjects' reports."""
    classes = set()
    for r in reports:
        sc = r.get("subject_class") if isinstance(r, dict) else r
        if sc is not None:
            classes.add(sc)
    evidence_classes = classes & set(D47_EVIDENCE_CLASSES)
    if len(evidence_classes) > 1:
        raise ValueError(
            f"assert_never_pooled: {sorted(evidence_classes)} would be POOLED into one D47 "
            f"evidence read — SPEC.md Amendment 1 requires the 887 A/D subject class and the "
            f"imprint primaries to be reported SEPARATELY, never combined. Refusing.")

# [A1] SPEC.md Design 1: "Policy walks, two legs." The OOS window is the BINDING read D47 is
# drawn from; the sanity leg is reported but is NOT a verdict source. These are UTC epoch
# SECONDS for the two pre-registered window boundaries (`date -u -d '<date>' +%s`), used to
# STRUCTURALLY validate (not merely comment) that a report tagged with a leg actually carries
# decisions from that leg's own window — a mislabeled leg is exactly the kind of "note describing
# a run that never happened" cycle 3 already found once (trace.py's universe_pin_note history).
LEG_BINDING = "oos-binding"      # 2025-10-05 .. 2026-03-31 (861 scorecard's oos_full window)
LEG_SANITY = "sanity"            # 2021-01-01 .. 2025-10-04 (882 Leg-A window)
LEGS = (LEG_BINDING, LEG_SANITY)
LEG_WINDOW_EPOCH = {
    LEG_BINDING: (1759622400, 1774915200),     # 2025-10-05T00:00:00Z, 2026-03-31T00:00:00Z
    LEG_SANITY: (1609459200, 1759536000),       # 2021-01-01T00:00:00Z, 2025-10-04T00:00:00Z
}
LEG_ROLE_NOTE = {
    LEG_BINDING: ("BINDING — SPEC.md Amendment 1: the D47 verdict is read from THIS leg. An "
                 "in-window EDGE here is near-foregone and nearly information-free; only an "
                 "OOS EDGE (or a decisive ANTI-EDGE/NULL) can move D47."),
    LEG_SANITY: ("SANITY — reported for completeness, NEVER the D47 verdict source. An "
                "in-window NULL/ANTI-EDGE here would be devastating; an in-window EDGE is "
                "near-foregone and does not, by itself, re-open the line."),
}


def validate_leg_window(leg, timestamps, label):
    """[A1] Structural (not commentary-only) enforcement that a report tagged with a leg
    actually carries decisions from that leg's own pre-registered window. Raises rather than
    silently mislabeling — the exact failure class trace.py's universe_pin_note history already
    demonstrated once (a note describing a run that never happened)."""
    if leg not in LEGS:
        raise ValueError(f"validate_leg_window: leg={leg!r} is not one of {LEGS!r}")
    ts = np.asarray(timestamps, dtype=np.int64)
    if ts.size == 0:
        return
    lo, hi = LEG_WINDOW_EPOCH[leg]
    bad = (ts < lo) | (ts > hi)
    if bad.any():
        raise ValueError(
            f"validate_leg_window: --leg {leg!r} was declared for {label}, but "
            f"{int(bad.sum())}/{ts.size} decision timestamp(s) fall outside that leg's "
            f"pre-registered window [{lo}, {hi}] (epoch seconds). Refusing to emit a report "
            f"whose declared leg does not match the data it was computed from.")

# TASK_892 fix pass (BLOCKER 1) — the bootstrap's own power floor. A circular block bootstrap
# with n <= block is DEGENERATE: n_blocks == 1, the single wrapped block covers every index
# exactly once, every resample is a rotation of the full sample, and a mean is rotation-
# invariant => a genuinely ZERO-WIDTH CI that can only read EDGE/ANTI-EDGE and never NULL.
# Require at least this many FULL blocks per side before a CI is emitted at all; anything
# thinner is labelled INSUFFICIENT-N and EXCLUDED from banding (see
# disclosures.BOOTSTRAP_POWER_DISCLOSURE for the full argument, which every report carries
# inline). STRICTER than, and independent of, SPEC.md's pre-registered n >= 30 floor.
MIN_BOOTSTRAP_BLOCKS = 2

# CYCLE-3 MAJOR (finding 2) / D51 RE-CUT, SUPERSEDED BY AMENDMENT 1's CALIBRATED-WIDTH CORRECTION.
# History: MIN_BOOTSTRAP_BLOCKS above kills only the LITERALLY degenerate case; it says nothing
# about whether the CI it emits has its nominal 95% coverage. Cycle 3 measured coverage under a
# TRUE NULL and found it well below nominal at low block counts (56.7%-78.5% at 2 blocks
# depending on scenario) and, MORE FUNDAMENTALLY (D49's escalation, adopted by SPEC Amendment 1):
# it PLATEAUS at ~88% under realistic overlapping dependence and NEVER reaches 95% at ANY sample
# size, because block=21 (SPEC.md, frozen) EQUALS the dependence length (consecutive forward-
# return windows overlap by up to H-1=20). A fixed "you need N blocks before banding" bar cannot
# fix a plateau -- MORE DATA DOES NOT CLOSE IT (disclosures.NULL_COVERAGE_ESCALATION).
#
# AMENDMENT 1 (D49's interval-method escalation, adopted): the remedy is the interval method, not
# the sample size. This module now applies remedy (iii) of the pre-registered ladder -- a
# CALIBRATED-WIDTH CORRECTION, measured by Monte Carlo through THIS module's own
# block_bootstrap_delta_ci (calibration.py; never a re-implementation) at every block count the
# real leg's strata can plausibly present (the OOS binding leg's ~5-6 blocks included), then
# INDEPENDENTLY RE-VALIDATED on a fresh trial batch before being trusted (disclosures.
# CALIBRATION_TABLE / calibrated_k_for). CALIBRATED_BOOTSTRAP_BLOCKS below is KEPT as a documented
# regression reference to the OLD fixed-bar approach it superseded, not as a live gate: the old
# bar is dead code now (calibration achieves >= 93% well below 30 blocks -- measured as low as 2
# blocks -- so nothing observed at n_blocks >= MIN_BOOTSTRAP_BLOCKS should ever need it), and this
# comment says so explicitly rather than leaving a floor whose claim quietly stopped being true
# (cycle 2's own failure mode: report.py:234 "Nothing here can emit a zero-width interval").
NULL_COVERAGE_BAND_MIN = disclosures.NULL_COVERAGE_BAND_MIN
CALIBRATED_BOOTSTRAP_BLOCKS = disclosures.CALIBRATED_BOOTSTRAP_BLOCKS

# D51 RE-CUT, PRIORITY 1 -- the zero_width guard, REBUILT under D51's transferable rule ("a
# tolerance relative to the data's own scale, never an exact-equality or bit-identity test, and
# its test must construct the NEAR case"). See ``_is_zero_width`` below for the full argument;
# ZERO_WIDTH_FLOOR is the one number that argument is built on.
#
# WHY NO STATISTIC COMPUTED PURELY FROM s/n's OWN DISPERSION OR MAGNITUDE CAN DO THIS JOB
# (measured, not assumed): a circular block bootstrap's CI is HOMOGENEOUS OF DEGREE 1 in its own
# input -- rescale s and n by any constant c and the whole interval (hence its width) rescales by
# c too. Measured directly (reproduced here and cross-checked against an independent probe from
# the factory orchestrator): holding the two sides' relative shape fixed and varying ONLY the
# overall scale across four decades (per-block jitter std 1e-9, 1e-6, 1e-4, and an O(1) "healthy"
# case), ci_width / jitter_std is CONSTANT to within Monte-Carlo noise. So a "width relative to
# this sample's own std/variance/magnitude" test is invariant to the exact rescaling that
# separates a genuinely-tiny-but-real effect from an artificially-shrunk (near-duplicate /
# stale-price / duplicated-window) one -- it is mathematically unable to tell them apart, no
# matter how the ratio is built. Discriminating them needs an ANCHOR EXTERNAL to the sample: each
# block's near-identical value is then an EFFECTIVE sample-size collapse the raw block count does
# not reflect, regardless of how many blocks are counted.
#
# THE ANCHOR: a forward return is a DIMENSIONLESS, already-normalized quantity (unlike a raw
# price, which needs per-asset rescaling) -- this whole instrument already expresses cost/
# tolerance quantities in the SAME unit (``cost = (fee_bps+slip_bps)/1e4`` in trace.py;
# FEE_BPS=16, SLIP_BPS=5 in eval_vector.py). ZERO_WIDTH_FLOOR = 1 basis point (1e-4): a CI
# narrower than this draws a distinction FINER than any cost/fee granularity this instrument
# computes anywhere, at ANY block count -- the "near-duplicate autocorrelated cell" signature
# disclosures.py's escalation note already names as the plausible real route to a collapsed
# interval. MEASURED MARGIN: at 1e-9-scale per-block jitter the emitted width is O(1e-9 to 1e-10)
# -- five to six orders of magnitude below this floor; at 1e-6-scale jitter it is O(1e-7) -- three
# orders below. A healthy simulated CI at a realistic forward-return scale is O(1e-1) to O(1e-2)
# -- three to four orders of magnitude ABOVE it. The smallest nonzero cost tier this instrument
# treats as economically real (SLIP_BPS=5) is 5e-4, still 5x above this floor; a boundary at 5e-4
# or looser would start swallowing that tier as "no width", so 1e-4 is the largest floor that
# stays clear of every cost unit this instrument computes, not an arbitrarily tuned number. (Per
# the same measurement, a per-block jitter of 1e-4 itself is a JUDGMENT BOUNDARY: its emitted
# width is O(1e-5), below this floor, so it reads ZERO-WIDTH-CI here -- deliberately, because 1e-4
# is this module's own smallest cost-relevant unit and a distinction that fine is not one this
# instrument's accounting can act on even if the resampling technically produced it.)
ZERO_WIDTH_FLOOR = 1e-4

# Reported-warning tolerance for SPEC.md:46's third matched moment (mean book width), expressed
# as a fraction of the universe so it means the same thing at 5 fixture assets and 64 real ones.
# A WARNING only — never an instrument_fail (SPEC.md's fail list is frozen).
BOOK_WIDTH_MISMATCH_WARN_FRAC = 0.005            # 0.5pp of the universe, mirroring the mix bar

BAND_INSUFFICIENT_N = "INSUFFICIENT-N"
BAND_INSTRUMENT_FAIL = "INSTRUMENT FAIL"
# CYCLE-3: two further NON-BANDS. Neither is one of SPEC.md's three publishable outcomes; both
# mean "this comparison is not interpretable" and both exclude it from banding.
BAND_LOW_POWER = "LOW-POWER"            # finding 2: measured coverage well below nominal 95%
BAND_ZERO_WIDTH = "ZERO-WIDTH-CI"       # finding 3: the CI has no width -> fake certainty
# [A1] D49(a): the fee16/slip5 tier is a REPORTED DIAGNOSTIC ONLY — it must never make or break a
# band. Enforced in CODE (compare(), below): whenever the graded trace's own tier_bp != 0, a
# would-be substantive band is structurally REPLACED by this marker before it ever reaches the
# returned dict, rather than merely being described as non-binding in prose.
BAND_FEE_TIER_DIAGNOSTIC = "DIAGNOSTIC-ONLY (fee16/slip5, non-binding per D49(a))"
SUBSTANTIVE_BANDS = ("EDGE", "ANTI-EDGE", "NULL")
NON_BANDS = (BAND_INSTRUMENT_FAIL, BAND_INSUFFICIENT_N, BAND_LOW_POWER, BAND_ZERO_WIDTH,
            BAND_FEE_TIER_DIAGNOSTIC)
RECOGNIZED_BANDS = SUBSTANTIVE_BANDS + NON_BANDS + ("UNDEFINED",)


BAND_NOTE = ("mechanical label under the SPEC.md pre-registered bands; the PROGRAM VERDICT "
             "(D47) is the planner's, not this module's. 'INSTRUMENT FAIL', 'INSUFFICIENT-N', "
             "'LOW-POWER' and 'ZERO-WIDTH-CI' are NOT bands — they are reported conditions that "
             "EXCLUDE this comparison from banding entirely (SPEC.md: an instrument failure is "
             "never folded into a band; and this module does not emit a band it cannot back).")


def min_bootstrap_n(block):
    """Minimum observations per SIDE for the block bootstrap to have any sampling variation."""
    return int(MIN_BOOTSTRAP_BLOCKS) * int(block)


def calibrated_bootstrap_n(block):
    """Minimum observations per SIDE for a CI at (approximately) its nominal coverage — the
    MEASURED bar (finding 2), above which this module is willing to emit a band."""
    return int(CALIBRATED_BOOTSTRAP_BLOCKS) * int(block)


def _load(path):
    with open(path) as fh:
        return json.load(fh)


# The provenance fields this module HARD-READS off every input trace. A trace that does not carry
# them is not a trace this module can grade — see _load_trace. Each one is paired with WHY its
# absence is a refusal rather than a default, because the refusal message names it.
REQUIRED_TRACE_PROVENANCE_WHY = {
    "budget_conservation_violations": (
        "SPEC.md makes a PV/budget violation an instrument-fail condition, and an absent field "
        "is NOT zero violations"),
    # FIX PASS (D51 re-cut review, MAJOR 1): the SAME defect class as the budget field, one
    # mechanism later. ``tier_bp`` is the SOLE gate on D49(a)'s fee-tier carve-out, and it was
    # read as ``.get("tier_bp", 0)`` — so a schema-v3 document that simply OMITS it read as
    # cost0 and a genuine fee16/slip5 comparison came back as a full citable EDGE with
    # ``instrument_fail: []`` instead of DIAGNOSTIC-ONLY (reproduced on real fee16 traces:
    # deleting this one key flipped identical numbers, delta 0.0207906523, from
    # DIAGNOSTIC-ONLY to EDGE). An absent cost tier is NOT cost0.
    "tier_bp": (
        "D49(a)/SPEC Amendment 1 makes a nonzero cost tier a DIAGNOSTIC-ONLY (never-binding) "
        "condition, and an absent tier is NOT cost0"),
    # ``fee_bps``/``slip_bps`` are the actual costs the tier LABEL stands for, and they gate the
    # same carve-out (see _cost_tier_of): a document claiming tier_bp=0 while carrying real
    # per-decision costs must not be graded as the cost0 primary either.
    "fee_bps": (
        "it is the cost the tier label stands for, and this module refuses to read a stated "
        "cost tier it cannot cross-check"),
    "slip_bps": (
        "it is the cost the tier label stands for, and this module refuses to read a stated "
        "cost tier it cannot cross-check"),
}
REQUIRED_TRACE_PROVENANCE = tuple(REQUIRED_TRACE_PROVENANCE_WHY)
REQUIRED_TRACE_SECTIONS = ("provenance", "decisions")
REQUIRED_TRACE_DECISION_KEYS = ("non_hold", "forward_return", "chosen_vectors", "realized_book",
                                "timestamp", "position_delta",
                                # [A1] Amendment 1's measurement surface (TRACE_SCHEMA v3):
                                # net exposure + the undirected market return over the same
                                # horizon/mask as forward_return, and position_delta (already
                                # emitted since v1/v2, now HARD-required) for the entry/exit vs
                                # re-trim co-primary split.
                                "market_return", "net_exposure")


def _load_trace(path):
    """Load ONE input trace and REFUSE anything that is not a declared ``trace.TRACE_SCHEMA``
    document carrying every field this comparison hard-reads.

    CYCLE-3 BLOCKER 1 — WHY THIS EXISTS. ``compare`` used to read the budget field softly:
    ``subject_doc["provenance"].get("budget_conservation_violations", 0)``. It was the ONLY
    ``.get(..., default)`` in the whole comparison (every other field is a hard ``dec["key"]``),
    and nothing validated ``doc.get("schema")`` against ``TRACE_SCHEMA`` at all. Measured against
    the real ``compare()``: two documents with NO ``schema`` key and NO
    ``budget_conservation_violations`` produced ``band=EDGE``, ``instrument_fail=[]``,
    ``violations=0`` — i.e. A TRACE THAT NEVER ASSERTED ITS BUDGET WAS CHECKED READ AS CLEAN and
    yielded a complete, citable EDGE verdict. For a fail-closed instrument whose whole job is to
    refuse uninterpretable comparisons, "the field is missing" must never be spelled "zero
    violations": absence of evidence is not evidence of conservation.

    A soft default here is uniquely dangerous because the PV/budget check is one of SPEC.md's
    three instrument-fail conditions, so silently defaulting it to 0 disables a pre-registered
    refusal from the outside, by feeding in a document that simply omits the field.

    FIX PASS (D51 re-cut review, MAJOR 1) — THE SAME DEFECT, ONE MECHANISM LATER. The fee-tier
    carve-out built in this cut re-introduced exactly the pattern this docstring argues against:
    ``compare`` read ``provenance.get("tier_bp", 0)`` while reading the budget field by hard
    subscript two lines away. ``tier_bp`` is the SOLE gate on D49(a)'s guarantee, so a v3
    document that omits it read as cost0 and a real fee16/slip5 comparison came back a citable
    EDGE. The rule this module now actually holds to, with no exception on the trace-read path:
    every provenance field a refusal depends on is REQUIRED here and subscripted (never
    ``.get``) downstream. ``fee_bps``/``slip_bps`` join it so a stated tier can be cross-checked
    against the costs it claims (see ``_cost_tier_of``)."""
    doc = _load(path)
    schema = doc.get("schema")
    if schema != trace_mod.TRACE_SCHEMA:
        raise RuntimeError(
            f"report.py: {path!r} declares schema {schema!r}, not {trace_mod.TRACE_SCHEMA!r}. "
            f"Refusing to grade a document whose format this module cannot vouch for — an "
            f"unvalidated input is how a trace that never ran the budget check reads as clean.")
    for section in REQUIRED_TRACE_SECTIONS:
        if section not in doc:
            raise RuntimeError(f"report.py: {path!r} carries schema "
                               f"{trace_mod.TRACE_SCHEMA!r} but no {section!r} section.")
    for key in REQUIRED_TRACE_PROVENANCE:
        if key not in doc["provenance"]:
            raise RuntimeError(
                f"report.py: {path!r} has no provenance.{key} — this comparison HARD-READS it "
                f"({REQUIRED_TRACE_PROVENANCE_WHY[key]}). Absence of the field is not the "
                f"healthy value. Re-emit the trace with trace.py.")
    for key in REQUIRED_TRACE_DECISION_KEYS:
        if key not in doc["decisions"]:
            raise RuntimeError(f"report.py: {path!r} has no decisions.{key}.")
    return doc


# [A1] D49(a) — the cost tier a trace DECLARES, read by hard subscript off provenance _load_trace
# has already validated. Returned as the whole (tier_bp, fee_bps, slip_bps) triple so the
# fee-tier carve-out cannot be defeated by an incoherent label either: a document stating
# tier_bp=0 while carrying real per-decision costs is still a fee-tier read, and a subject/null
# pair whose triples differ is not a reading of either tier.
_COST_TIER_ZERO = (0, 0.0, 0.0)


def _cost_tier_of(doc):
    prov = doc["provenance"]
    return (int(prov["tier_bp"]), float(prov["fee_bps"]), float(prov["slip_bps"]))


def _fmt_cost_tier(tier):
    return f"tier_bp={tier[0]} fee_bps={tier[1]:g} slip_bps={tier[2]:g}"


def _to_float_grid(rows):
    return np.asarray([[np.nan if x is None else x for x in row] for row in rows],
                      dtype=np.float64)


def _non_hold_forward_returns(trace_doc):
    """Flatten a trace's (decision, asset) grid to the pooled 1-D array of forward returns at
    non-HOLD cells with a defined (non-NaN) forward_return — the SPEC's exact unit of pooling.
    Also returns the per-cell DECISION timestamp aligned 1:1 with `values` (regime bucketing),
    the raw non_hold mask, per-asset action codes (activity-match check) and the row count
    (min-decisions check).

    [A1] ALSO splits the pooled cells into the co-primary partition SPEC.md's Amendment 1
    requires: ENTRY/EXIT (``position_delta != 0`` — a sign change or 0<->nonzero transition,
    the decision-quality read) vs MAINTENANCE RE-TRIM (``non_hold & position_delta == 0`` — the
    book's side is unchanged, reported for completeness). ``realized_book`` is a 3-STATE
    encoding {flat, long, short} (module docstring, trace.py); a WEIGHT-MAGNITUDE change within
    the same side is invisible to it, so "re-trim" here means "side-preserving", NOT "no change
    at all" — a magnitude-only re-trim is not separately resolved by this encoding, and this is
    disclosed rather than silently let the 3-state proxy stand in for a finer claim (see
    disclosures.POSITION_CHANGE_SPLIT_DISCLOSURE, carried inline in every report)."""
    dec = trace_doc["decisions"]
    non_hold = np.asarray(dec["non_hold"], dtype=bool)
    fwd = _to_float_grid(dec["forward_return"])
    mkt = _to_float_grid(dec["market_return"])
    pos_delta = np.asarray(dec["position_delta"], dtype=np.float64)
    chosen = np.asarray(dec["chosen_vectors"], dtype=np.int64)
    book = np.asarray(dec["realized_book"], dtype=np.int64)
    ts = np.asarray(dec["timestamp"], dtype=np.int64)
    n_assets = non_hold.shape[1] if non_hold.ndim == 2 else 0
    ts_grid = np.broadcast_to(ts[:, None], non_hold.shape)
    defined = non_hold & np.isfinite(fwd)
    entry_exit = defined & (pos_delta != 0)
    re_trim = defined & (pos_delta == 0)
    return {
        "values": fwd[defined],
        "timestamps": ts_grid[defined],
        "market_return_values": mkt[defined],
        "n_non_hold": int(non_hold.sum()),
        "n_non_hold_with_forward": int(defined.sum()),
        "n_decisions": int(non_hold.shape[0]),
        "n_assets": int(n_assets),
        "chosen_vectors": chosen,
        "net_exposure": np.asarray([np.nan if x is None else x
                                    for x in dec["net_exposure"]], dtype=np.float64),
        # [A1] co-primary split — see docstring.
        "values_entry_exit": fwd[entry_exit],
        "timestamps_entry_exit": ts_grid[entry_exit],
        "values_re_trim": fwd[re_trim],
        "timestamps_re_trim": ts_grid[re_trim],
        "n_entry_exit": int(entry_exit.sum()),
        "n_re_trim": int(re_trim.sum()),
        # SPEC.md:46's third matched moment — book WIDTH per decision (count of non-flat slots
        # in the realized book). Kept per-decision (not pre-averaged) so the distribution the
        # i.i.d. null cannot match is reportable, not just its mean.
        "book_width": (book != 0).sum(axis=1) if book.ndim == 2 else np.array([], dtype=np.int64),
    }


def book_width_summary(widths, n_assets):
    """Mean/spread of realized book width — SPEC.md:46's third matched moment. The MEAN is what
    SPEC.md promises to match; the spread is reported beside it because the i.i.d.-per-asset null
    matches per-asset marginals and therefore the mean, but destroys cross-asset correlation and
    therefore the DISTRIBUTION (disclosures.BOOK_WIDTH_DISCLOSURE, carried in every report)."""
    w = np.asarray(widths, dtype=np.float64)
    if w.size == 0:
        return {"mean": None, "std": None, "p05": None, "p50": None, "p95": None,
                "max": None, "n_decisions": 0, "n_assets": int(n_assets),
                "mean_frac_of_universe": None}
    p05, p50, p95 = (float(x) for x in np.percentile(w, [5, 50, 95]))
    mean = float(np.mean(w))
    return {"mean": mean, "std": float(np.std(w, ddof=1)) if w.size > 1 else 0.0,
            "p05": p05, "p50": p50, "p95": p95, "max": float(np.max(w)),
            "n_decisions": int(w.size), "n_assets": int(n_assets),
            "mean_frac_of_universe": (mean / n_assets) if n_assets else None}


def regime_bucket_report(subj, null_flat_list, ext_windows, block, n_boot, seed,
                         instrument_fail=None, force_diagnostic=False):
    """Per-831-EXT-window (+ an unconditional 'ALL') Delta/CI/band — scope 3's per-regime-bucket
    breakdown. A window with no file (fixture-scale runs — 831-EXT is a real-history artifact,
    `experiments/835_stage4_eval/eval_vector.py:load_ext_windows`) or no observations degrades
    to a reported None, matching `score_tier`'s own per_window convention rather than crashing.

    TWO fix-pass conditions, both of which USED TO LEAK A SUBSTANTIVE BAND out of this function:

    * BLOCKER 1 — this function is where thinness actually bites (a bucket's BLOCK count, not
      the pooled cell count, is what the bootstrap consumes), and it is UNGUARDED by
      ``compare``'s pooled-count gate. A bucket below the bootstrap's power floor now reads
      INSUFFICIENT-N with ci_lo/ci_hi = null (``block_bootstrap_delta_ci``'s refusal) instead of
      a zero-width, fake-certainty CI. Every bucket reports ``n_blocks_subject``/``n_blocks_null``
      beside its raw n, derived from the data — deliberately NOT quoting SPEC.md's own thin-n
      sentence, which a 2026-08-19 planner/operator audit measured wrong (see
      ``disclosures.BOOTSTRAP_POWER_DISCLOSURE``).
    * CYCLE-3 findings 2 and 3 — the same argument, one level further in. Clearing the DEGENERACY
      floor is not clearing the CALIBRATION bar: measured against a true null, this module's
      bootstrap covers ~78.5% at 2 blocks and ~89% at 4, so a per-bucket EDGE just above the old
      floor was closer to a 1-in-5 coin flip than to a 95% claim. Buckets below
      ``CALIBRATED_BOOTSTRAP_BLOCKS`` now read LOW-POWER (CI reported, band refused, measured
      coverage stated), and any bucket whose interval comes back with ZERO WIDTH reads
      ZERO-WIDTH-CI regardless of its block count. Per-regime buckets are exactly where this
      bites — SPEC.md itself says the within-regime match "thins further".
    * finding 6 — a bucket knew nothing about ``instrument_fail``, so a report whose TOP-LEVEL
      band was correctly suppressed to INSTRUMENT FAIL still published e.g.
      ``per_window["ALL"]["band"] == "EDGE"``. ``instrument_fail`` is now threaded in and
      preempts every bucket's band, and EVERY bucket carries the same ``band_note`` disclaimer
      the top level carries — a reader extracting only ``per_window`` cannot read a verdict the
      top level just declared uninterpretable."""
    null_values_all = (np.concatenate([nf["values"] for nf in null_flat_list])
                       if null_flat_list else np.array([]))
    null_ts_all = (np.concatenate([nf["timestamps"] for nf in null_flat_list])
                  if null_flat_list else np.array([], dtype=np.int64))
    buckets = {"ALL": None}
    for name, wd in (ext_windows or {}).items():
        buckets[name] = wd
    out = {}
    for name, wd in buckets.items():
        if name != "ALL" and wd is None:
            out[name] = {"window": None, "note": "831-EXT window file missing",
                         "band": "UNDEFINED", "band_note": BAND_NOTE}
            continue
        if name == "ALL":
            s_vals, n_vals = subj["values"], null_values_all
        else:
            s_sel = ((subj["timestamps"] >= wd["start_ts"]) &
                    (subj["timestamps"] <= wd["end_ts"]))
            n_sel = ((null_ts_all >= wd["start_ts"]) & (null_ts_all <= wd["end_ts"]))
            s_vals, n_vals = subj["values"][s_sel], null_values_all[n_sel]
        ci = block_bootstrap_delta_ci(s_vals, n_vals, block=block, n_boot=n_boot, seed=seed)
        band = band_for(ci, instrument_fail=instrument_fail, force_diagnostic=force_diagnostic)
        entry = {"window": wd["window"] if wd else None, "n_subject": int(s_vals.size),
                 "n_null": int(n_vals.size), "delta": ci, "band": band,
                 "band_note": BAND_NOTE}
        if band == BAND_INSTRUMENT_FAIL:
            entry["excluded_from_banding"] = True
            entry["excluded_reason"] = ("the INSTRUMENT FAILED for this comparison (see "
                                        "instrument_fail_reasons); no bucket of a failed "
                                        "instrument carries a substantive band")
        elif band == BAND_FEE_TIER_DIAGNOSTIC:
            entry["excluded_from_banding"] = True
            entry["excluded_reason"] = ("D49(a): this comparison ran at a nonzero cost tier — "
                                        "reported as a diagnostic (delta/CI above), never a band")
        elif band in NON_BANDS:
            # INSUFFICIENT-N / LOW-POWER / ZERO-WIDTH-CI all carry their own measured `note`.
            entry["excluded_from_banding"] = True
            entry["excluded_reason"] = ci.get("note", band)
        else:
            entry["excluded_from_banding"] = False
        out[name] = entry
    return out


def activity_summary(chosen_vectors):
    """activity_rate = fraction of (decision, asset) cells that are non-HOLD; long/short mix =
    fraction of non-HOLD cells that are long-side (Buy=2/LimitBuy=4) vs short-side
    (Sell=0/LimitSell=3) — the env code convention eval_vector.py already documents."""
    HOLD = 1
    total = chosen_vectors.size
    non_hold = chosen_vectors != HOLD
    n_non_hold = int(non_hold.sum())
    activity_rate = n_non_hold / total if total else 0.0
    n_long = int(np.isin(chosen_vectors, [2, 4]).sum())
    n_short = int(np.isin(chosen_vectors, [0, 3]).sum())
    long_frac = n_long / n_non_hold if n_non_hold else 0.0
    return {"activity_rate": activity_rate, "long_frac_of_non_hold": long_frac,
            "n_non_hold": n_non_hold, "n_long": n_long, "n_short": n_short}


def _is_zero_width(ci_lo, ci_hi):
    """D51 rebuild of the zero_width guard (see ZERO_WIDTH_FLOOR's module-level comment for the
    full argument): a tolerance ANCHORED to this instrument's own smallest cost-relevant unit
    (1bp), not an exact-equality/bit-identity test and not a tolerance relative to the sample's
    own dispersion (provably unable to discriminate a real tiny effect from an artificially
    shrunk one — see the ZERO_WIDTH_FLOOR comment). Applied to the FINAL (possibly
    calibration-widened) interval, since that is what is actually reported and banded."""
    return (float(ci_hi) - float(ci_lo)) <= ZERO_WIDTH_FLOOR


def block_bootstrap_delta_ci(subject_vals, null_vals, block=21, n_boot=10000, seed=0):
    """Circular block bootstrap CI on Delta = mean(subject_vals) - mean(null_vals). Each side is
    resampled independently (its own circular block scheme, trader/backtest.py's
    `_block_bootstrap_sharpe_ci` house style) — block length is FIXED at 21 (SPEC.md), not
    derived from the series length, because H=21 is the horizon the forward returns already
    overlap by construction (consecutive decisions' forward windows share up to H-1 days), and
    a block at least as long as that overlap is what block-bootstrap needs to respect it.

    AMENDMENT 1 / D51 RE-CUT — the RAW percentile interval (``ci_lo_raw``/``ci_hi_raw``) is now
    ALWAYS widened by a MEASURED, INDEPENDENTLY-VALIDATED calibration factor
    (``disclosures.calibrated_k_for``, built by ``calibration.py``'s two-scenario Monte Carlo
    harness) before it is reported as ``ci_lo``/``ci_hi`` — the interval-method remedy D49's
    escalation and SPEC Amendment 1 both call for, replacing the old fixed
    ``n_blocks >= CALIBRATED_BOOTSTRAP_BLOCKS`` bar (which a plateaued-at-88% coverage curve can
    never clear at any sample size; see disclosures.NULL_COVERAGE_ESCALATION).

    FOUR reported conditions can come back instead of a bandable CI (each one is also an
    ``excluded_from_banding`` label in ``band_for``):
      * ``insufficient_n``       — below MIN_BOOTSTRAP_BLOCKS full blocks per side: the
        bootstrap is DEGENERATE (a single circular block is a rotation of the whole sample) and
        no CI is emitted at all.
      * ``low_power``            — at/above that floor, but no calibration entry for this block
        count achieved >= the target coverage (disclosures.CALIBRATION_TABLE): the RAW CI is
        reported for information, with the measured (failed) validation coverage stated, and the
        band is refused.
      * ``zero_width_ci``        — the FINAL (possibly widened) interval has no width relative to
        this instrument's own smallest cost unit (see ZERO_WIDTH_FLOOR): possible at ANY n when
        the inputs carry no distinguishable within-side variance at that resolution, e.g.
        constant / near-duplicate cells.
      * ``calibration_capped``   — the calibration harness found a widening factor but refused to
        apply it because it exceeded ``calibration.k_cap`` (an interval "calibrated" past that
        point is being inflated until it can't fail, not calibrated); treated the same as
        low_power for banding purposes.
    """
    # CYCLE-3 finding 7: validate the two shape parameters. Unreachable through the pinned
    # invocation (``block``/``n_boot`` come from argparse defaults or SPEC.md's fixed 21/10000),
    # but the failures were raw and confusing rather than loud: measured, ``block=0`` ->
    # ZeroDivisionError, ``block=-5`` -> numpy "negative dimensions are not allowed",
    # ``n_boot=0`` -> IndexError from np.percentile on an empty array.
    block = int(block)
    n_boot = int(n_boot)
    if block < 1:
        raise ValueError(f"block_bootstrap_delta_ci: block must be >= 1, got {block} "
                         f"(SPEC.md pins block = 21, the forward-return horizon)")
    if n_boot < 1:
        raise ValueError(f"block_bootstrap_delta_ci: n_boot must be >= 1, got {n_boot} "
                         f"(SPEC.md pins 10000 resamples)")
    s = np.asarray(subject_vals, dtype=np.float64)
    s = s[np.isfinite(s)]
    n = np.asarray(null_vals, dtype=np.float64)
    n = n[np.isfinite(n)]
    min_n = min_bootstrap_n(block)
    if s.size < min_n or n.size < min_n:
        # BLOCKER 1 (fix pass): an under-powered sample is a REPORTED CONDITION, never a band.
        # The point estimate is still reported when both sides have anything at all — it is the
        # CI, and with it the band, that is refused.
        # CYCLE-3 finding 3: the comment here used to read "Nothing here can emit a zero-width
        # interval", which was FALSE — this floor guarantees "enough BLOCKS", not "enough
        # VARIANCE", and the two are different guarantees. A zero-width interval is still
        # reachable at/above the floor from constant or near-duplicate inputs (measured:
        # s = full(42, 0.02), n = full(42, 0.0) -> ci_lo == ci_hi == 0.02, band EDGE). That case
        # is caught explicitly AFTER the resampling, below.
        delta = (float(np.mean(s) - np.mean(n)) if (s.size and n.size) else None)
        return {"delta": delta, "ci_lo": None, "ci_hi": None,
                "ci_lo_raw": None, "ci_hi_raw": None, "n_boot": 0,
                "n_subject": int(s.size), "n_null": int(n.size),
                # POWER IN BLOCKS, derived from the data at hand — the quantity this bootstrap
                # actually consumes (disclosures.BOOTSTRAP_POWER_DISCLOSURE; SPEC.md's own thin-n
                # sentence is NOT quoted here, a 2026-08-19 audit measured it wrong).
                "n_blocks_subject": int(s.size // int(block)),
                "n_blocks_null": int(n.size // int(block)),
                "block": int(block),
                "insufficient_n": True, "min_n_for_bootstrap": int(min_n),
                "low_power": True, "zero_width_ci": False,
                "calibration_k": None, "calibration_achieved": False,
                "calibration_source_n_blocks": None,
                "calibrated_min_blocks": int(CALIBRATED_BOOTSTRAP_BLOCKS),
                "calibrated_min_n": int(calibrated_bootstrap_n(block)),
                "measured_null_coverage": disclosures.measured_null_coverage(
                    min(int(s.size // int(block)), int(n.size // int(block)))),
                "measured_null_coverage_overlapping": disclosures.measured_null_coverage(
                    min(int(s.size // int(block)), int(n.size // int(block))),
                    scenario="overlapping"),
                "note": (f"INSUFFICIENT-N: the circular block bootstrap needs >= "
                         f"{MIN_BOOTSTRAP_BLOCKS} full blocks of {int(block)} "
                         f"(n >= {min_n}) per side; got n_subject={int(s.size)}, "
                         f"n_null={int(n.size)}. No CI and NO BAND are emitted — a single "
                         f"circular block cannot produce variation (every resample is a "
                         f"rotation of the full sample and a mean is rotation-invariant), so "
                         f"a CI here would be a zero-width, fake-certainty interval.")}
    rng = np.random.RandomState(seed)

    def _resample(x, n_boot_):
        L = x.size
        n_blocks = int(np.ceil(L / block))
        out = np.empty(n_boot_)
        for b in range(n_boot_):
            starts = rng.randint(0, L, size=n_blocks)
            idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % L
            out[b] = np.mean(x[idx[:L]])
        return out

    s_means = _resample(s, n_boot)
    n_means = _resample(n, n_boot)
    deltas = s_means - n_means
    lo_raw, hi_raw = (float(x) for x in np.percentile(deltas, [2.5, 97.5]))
    delta = float(np.mean(s) - np.mean(n))
    n_blocks_s, n_blocks_n = int(s.size // block), int(n.size // block)
    n_blocks_min = min(n_blocks_s, n_blocks_n)

    # AMENDMENT 1 / D51 RE-CUT — remedy (iii) of the pre-registered ladder: widen the RAW
    # interval by a factor measured (and INDEPENDENTLY validated) to reach the calibration
    # harness's target coverage at this block count, disclosures.CALIBRATION_TABLE /
    # calibrated_k_for (calibration.py builds the table; never re-derived inline here).
    # Widened AROUND THE POINT ESTIMATE (delta), not around 0, so a genuinely nonzero delta is
    # not itself read as evidence against widening.
    k, achieved, source_nb = disclosures.calibrated_k_for(n_blocks_min)
    capped = (not achieved) and (k is not None)   # calibration.py already refused a >k_cap fit
    if achieved:
        half_lo, half_hi = max(0.0, delta - lo_raw), max(0.0, hi_raw - delta)
        lo, hi = delta - k * half_lo, delta + k * half_hi
    else:
        lo, hi = lo_raw, hi_raw
    low_power = not achieved
    # finding 3 / D51 rebuild: "enough blocks" (or even a calibrated width) is not "enough
    # variance". Constant / near-duplicate inputs still collapse the FINAL interval below this
    # instrument's own smallest cost-relevant unit — see ZERO_WIDTH_FLOOR's module comment for
    # why this must be an EXTERNALLY anchored floor, not a tolerance relative to this sample's
    # own dispersion. Checked on the interval actually reported (post-widening).
    zero_width = _is_zero_width(lo, hi)
    out = {"delta": delta, "ci_lo": float(lo), "ci_hi": float(hi),
           "ci_lo_raw": float(lo_raw), "ci_hi_raw": float(hi_raw),
           "n_boot": int(n_boot), "n_subject": int(s.size), "n_null": int(n.size),
           "n_blocks_subject": n_blocks_s, "n_blocks_null": n_blocks_n,
           "block": int(block),
           "insufficient_n": False, "min_n_for_bootstrap": int(min_n),
           # ---- Amendment 1: the calibrated-width correction actually applied here ----
           "calibration_k": (float(k) if k is not None else None),
           "calibration_achieved": bool(achieved),
           "calibration_capped": bool(capped),
           "calibration_source_n_blocks": (int(source_nb) if source_nb is not None else None),
           "calibration_target_coverage": float(disclosures.CALIBRATION_TARGET_COVERAGE),
           "low_power": bool(low_power),
           # legacy fields, kept for the regression tripwire against the OLD fixed-bar approach
           # (report.py module comment above CALIBRATED_BOOTSTRAP_BLOCKS): NOT a live gate.
           "calibrated_min_blocks": int(CALIBRATED_BOOTSTRAP_BLOCKS),
           "calibrated_min_n": int(calibrated_bootstrap_n(block)),
           "measured_null_coverage": disclosures.measured_null_coverage(n_blocks_min),
           "measured_null_coverage_overlapping": disclosures.measured_null_coverage(
               n_blocks_min, scenario="overlapping"),
           # the RAW percentile interval's own nominal level (2.5/97.5, unconditionally) — distinct
           # from calibration_target_coverage (0.93), which is the ACHIEVED-coverage bar the
           # calibrated widening is measured against. A nominal-95% interval with sub-nominal
           # ACTUAL coverage is exactly the miscalibration Amendment 1 exists to correct.
           "ci_nominal_coverage": 0.95,
           # ---- finding 3 / D51: the interval's realized (post-widening) width ----
           "zero_width_ci": bool(zero_width),
           "ci_width": float(hi) - float(lo),
           "ci_width_raw": float(hi_raw) - float(lo_raw),
           "resample_delta_spread": float(np.max(deltas) - np.min(deltas)),
           "subject_variance": float(np.var(s)) if s.size else None,
           "null_variance": float(np.var(n)) if n.size else None}
    if low_power:
        if capped:
            out["note"] = (
                f"LOW-POWER (calibration CAPPED): at {n_blocks_min} full block(s) of {block} on "
                f"the weaker side, the calibration harness measured that the widening factor "
                f"needed to reach {100.0*disclosures.CALIBRATION_TARGET_COVERAGE:.0f}% coverage "
                f"exceeds calibration.k_cap — an interval widened that far is being inflated "
                f"until it can't fail, not calibrated. The RAW CI is reported for information; "
                f"NO BAND is emitted.")
        else:
            out["note"] = (
                f"LOW-POWER: no entry in disclosures.CALIBRATION_TABLE at or below "
                f"{n_blocks_min} full block(s) of {block} achieved "
                f"{100.0*disclosures.CALIBRATION_TARGET_COVERAGE:.0f}% validated coverage under "
                f"the realistic-dependence (overlapping) scenario. The RAW CI is reported for "
                f"information (measured true-null coverage at this power: "
                f"~{100.0*(out['measured_null_coverage'] or 0.0):.0f}% i.i.d. / "
                f"~{100.0*(out['measured_null_coverage_overlapping'] or 0.0):.0f}% overlapping); "
                f"NO BAND is emitted. See bootstrap_power_disclosure / "
                f"calibrated_width_disclosure for the full argument.")
    elif capped:
        # unreachable given `low_power = not achieved` above, kept only so a future refactor that
        # decouples the two flags cannot silently drop this branch's message.
        pass
    if zero_width:
        out["note"] = (
            f"ZERO-WIDTH-CI: the {'calibrated ' if achieved else ''}95% interval has width "
            f"{float(hi) - float(lo)!r}, at or below this instrument's own smallest cost-relevant "
            f"unit (ZERO_WIDTH_FLOOR={ZERO_WIDTH_FLOOR!r}) — despite {n_blocks_min} full "
            f"block(s) per side, the inputs carry no within-side variance DISTINGUISHABLE at any "
            f"resolution this instrument's cost accounting can act on (subject_variance="
            f"{out['subject_variance']!r}, null_variance={out['null_variance']!r}). A CI this "
            f"narrow reads EDGE or ANTI-EDGE with fake certainty and can never read NULL; NO BAND "
            f"is emitted. Neither the block floor nor a calibrated width guarantees 'enough "
            f"distinguishable variance' — see ZERO_WIDTH_FLOOR's module comment.")
    return out


def band_for(ci, instrument_fail=None, force_diagnostic=False):
    """The SPEC's mechanical band label, from a ``block_bootstrap_delta_ci`` RESULT DICT.

    Takes the whole dict (not a bare ci_lo/ci_hi pair) so the conditions that must PREEMPT a band
    cannot be dropped by a caller:
      * the instrument itself failed (SPEC.md: "any failure of the instrument ... is an
        INSTRUMENT FAIL, reported as such, NOT folded into a band") — passed in by the caller;
      * the sample is degenerate for the block bootstrap (BLOCKER 1) => INSUFFICIENT-N;
      * the emitted (possibly calibration-widened) interval has NO WIDTH relative to this
        instrument's own smallest cost-relevant unit (cycle-3 finding 3 / D51 rebuild) =>
        ZERO-WIDTH-CI;
      * the sample is above the degeneracy floor but no measured, independently-validated
        calibration factor reached the target coverage at this block count (Amendment 1's
        interval-method remedy; cycle-3 finding 2's successor) => LOW-POWER.
    Only after all of them are ruled out does a CI get read as EDGE / ANTI-EDGE / NULL — and even
    then, [A1] D49(a): if the caller marks this comparison as a fee16/slip5 tier
    (``force_diagnostic=True``), a would-be SUBSTANTIVE band is replaced by
    BAND_FEE_TIER_DIAGNOSTIC before it is ever returned. This is the CODE enforcement of "fee16 is
    a reported diagnostic, never a band condition": the override happens HERE, the one function
    every band computation in this module (pooled, per-window, co-primary) routes through, so a
    fee16 trace cannot make or break a band by construction, not merely by convention."""
    if instrument_fail:
        return BAND_INSTRUMENT_FAIL
    if ci.get("insufficient_n"):
        return BAND_INSUFFICIENT_N
    # finding 3 before finding 2: a zero-width interval is a hard degeneracy, not merely thin.
    if ci.get("zero_width_ci"):
        return BAND_ZERO_WIDTH
    if ci.get("low_power"):
        return BAND_LOW_POWER
    ci_lo, ci_hi = ci.get("ci_lo"), ci.get("ci_hi")
    if ci_lo is None or ci_hi is None:
        return "UNDEFINED"
    if force_diagnostic:
        return BAND_FEE_TIER_DIAGNOSTIC
    if ci_lo > 0:
        return "EDGE"
    if ci_hi < 0:
        return "ANTI-EDGE"
    return "NULL"


def co_primary_split_report(subj, null_flat_list, split, block, n_boot, seed,
                            instrument_fail=None, force_diagnostic=False):
    """[A1] Co-primary split (SPEC.md Amendment 1): bands ENTRY/EXIT and MAINTENANCE RE-TRIM
    Delta SEPARATELY, each through the SAME calibrated block_bootstrap_delta_ci/band_for
    machinery as the pooled headline — so each gets its OWN gate-(viii) calibration
    certification at its OWN (thinner) block count, never inheriting the pooled read's power.
    ``split`` is ``"entry_exit"`` or ``"re_trim"`` (the two ``_non_hold_forward_returns`` keys)."""
    vkey, tkey = f"values_{split}", f"timestamps_{split}"
    s_vals = subj[vkey]
    n_vals = (np.concatenate([nf[vkey] for nf in null_flat_list]) if null_flat_list
             else np.array([]))
    ci = block_bootstrap_delta_ci(s_vals, n_vals, block=block, n_boot=n_boot, seed=seed)
    band = band_for(ci, instrument_fail=instrument_fail, force_diagnostic=force_diagnostic)
    if band in SUBSTANTIVE_BANDS:
        excluded_reason = None
    elif band == BAND_INSTRUMENT_FAIL:
        excluded_reason = "the INSTRUMENT FAILED for this comparison"
    elif band == BAND_FEE_TIER_DIAGNOSTIC:
        excluded_reason = ("D49(a): this comparison ran at a nonzero cost tier — reported as a "
                           "diagnostic (delta/CI above), never a band")
    else:
        excluded_reason = ci.get("note", band)
    return {"delta": ci, "band": band, "band_note": BAND_NOTE,
           "n_subject": int(s_vals.size), "n_null": int(n_vals.size),
           "excluded_from_banding": band in NON_BANDS,
           "excluded_reason": excluded_reason}


def compare(subject_path, null_paths, block=21, n_boot=10000, seed=0, ext_windows=None,
           leg=None):
    # BLOCKER 1: schema-validated, hard-read loads — never a bare json.load whose missing fields
    # default to the healthy value.
    subject_doc = _load_trace(subject_path)
    subj = _non_hold_forward_returns(subject_doc)

    null_docs = [_load_trace(p) for p in null_paths]
    null_flat = [_non_hold_forward_returns(d) for d in null_docs]

    # [A1] structural leg binding — see validate_leg_window. Every decision (subject AND every
    # null seed) must fall inside the declared leg's pre-registered window.
    if leg is not None:
        validate_leg_window(leg, subj["timestamps"], "the subject trace")
        for i, nf in enumerate(null_flat):
            validate_leg_window(leg, nf["timestamps"], f"null trace #{i}")

    # [A1] D49(a): fee16/slip5 is a REPORTED DIAGNOSTIC ONLY, never a band condition. Detected
    # from the graded trace's OWN provenance (never a CLI flag a caller could forget to pass) —
    # if EITHER side ran at a nonzero cost tier, every band this compare() call could emit is
    # structurally forced to BAND_FEE_TIER_DIAGNOSTIC via band_for's force_diagnostic (never by
    # prose alone). A tier MISMATCH between subject and null is not silently ignored either.
    # [A1] subject class, read from the trace this report was actually built from (never a
    # caller-supplied override) — see assert_never_pooled / in_d47_evidence_base above.
    subject_class = subject_doc["provenance"].get("subject_class")
    if subject_class is not None and subject_class not in SUBJECT_CLASSES:
        raise ValueError(
            f"compare: subject trace declares subject_class={subject_class!r}, not one of "
            f"{SUBJECT_CLASSES!r}. Refusing to grade a trace tagged with a class this module's "
            f"never-pooled enforcement cannot recognize.")

    # FIX PASS (MAJOR 1): HARD reads, exactly like the budget field below — never
    # `.get("tier_bp", 0)`. `_load_trace` has already refused any document that omits any part of
    # the cost triple, so these subscripts can only fail if that guard is bypassed. The soft
    # default this replaces was the one spelling that turned a real fee16/slip5 comparison into a
    # citable cost0 EDGE simply by omitting a field.
    subj_cost_tier = _cost_tier_of(subject_doc)
    null_cost_tiers = sorted({_cost_tier_of(d) for d in null_docs})
    subj_tier_bp = subj_cost_tier[0]
    is_fee_tier = any(t != _COST_TIER_ZERO for t in [subj_cost_tier] + null_cost_tiers)
    if null_docs and null_cost_tiers != [subj_cost_tier]:
        instrument_fail_tier_note = (
            f"subject {_fmt_cost_tier(subj_cost_tier)} but null cost tier(s) "
            f"[{'; '.join(_fmt_cost_tier(t) for t in null_cost_tiers)}] — a tier-mismatched "
            f"comparison is not a reading of either tier")
    else:
        instrument_fail_tier_note = None

    null_values = np.concatenate([nf["values"] for nf in null_flat]) if null_flat else np.array([])
    null_chosen = np.concatenate([nf["chosen_vectors"].ravel() for nf in null_flat]) \
        if null_flat else np.array([], dtype=np.int64)

    subj_activity = activity_summary(subj["chosen_vectors"])
    null_activity = activity_summary(null_chosen) if null_chosen.size else \
        {"activity_rate": None, "long_frac_of_non_hold": None, "n_non_hold": 0}

    instrument_fail = []
    if (subj_activity["activity_rate"] is not None and
            null_activity["activity_rate"] is not None):
        d_rate_pp = 100.0 * abs(subj_activity["activity_rate"] - null_activity["activity_rate"])
        d_mix_pp = 100.0 * abs(subj_activity["long_frac_of_non_hold"] -
                               null_activity["long_frac_of_non_hold"])
        if d_rate_pp > ACTIVITY_MISMATCH_TOL_PP:
            instrument_fail.append(
                f"null activity-rate mismatch {d_rate_pp:.3f}pp > {ACTIVITY_MISMATCH_TOL_PP}pp")
        if d_mix_pp > ACTIVITY_MISMATCH_TOL_PP:
            instrument_fail.append(
                f"null long/short-mix mismatch {d_mix_pp:.3f}pp > {ACTIVITY_MISMATCH_TOL_PP}pp")
    else:
        d_rate_pp = d_mix_pp = None

    n_pooled_non_hold = subj["n_non_hold_with_forward"]
    if n_pooled_non_hold < MIN_NON_HOLD_DECISIONS:
        instrument_fail.append(
            f"only {n_pooled_non_hold} subject non-HOLD decisions with a defined forward "
            f"return (< {MIN_NON_HOLD_DECISIONS})")

    # BLOCKER 1: HARD reads (KeyError, not a 0 default). _load_trace has already refused any
    # document that lacks the field, so these subscripts can only fail if that guard is bypassed.
    pv_violations = int(subject_doc["provenance"]["budget_conservation_violations"])
    for d in null_docs:
        pv_violations += int(d["provenance"]["budget_conservation_violations"])
    if pv_violations:
        instrument_fail.append(f"{pv_violations} budget-conservation violation(s) across traces")

    # SPEC.md:46's THIRD promised matched moment (fix pass, MAJOR 3): average book width. It was
    # promised and never computed; it is now computed for both sides, reported, and its
    # distributional caveat disclosed. A mismatch is a WARNING, never an instrument_fail —
    # SPEC.md's fail list is a frozen pre-registration and this module does not extend it.
    subj_width = book_width_summary(subj["book_width"], subj["n_assets"])
    null_widths = (np.concatenate([nf["book_width"] for nf in null_flat])
                   if null_flat else np.array([], dtype=np.int64))
    null_width = book_width_summary(null_widths, subj["n_assets"])
    warnings = []
    if subj_width["mean"] is not None and null_width["mean"] is not None:
        d_width = float(subj_width["mean"] - null_width["mean"])
        d_width_frac = (abs(d_width) / subj["n_assets"]) if subj["n_assets"] else None
        if d_width_frac is not None and d_width_frac > BOOK_WIDTH_MISMATCH_WARN_FRAC:
            warnings.append(
                f"null mean book-width mismatch {d_width:+.4f} slots "
                f"({100.0 * d_width_frac:.3f}pp of the {subj['n_assets']}-asset universe > "
                f"{100.0 * BOOK_WIDTH_MISMATCH_WARN_FRAC}pp) — SPEC.md:46 promises matched "
                f"AVERAGE book width; reported as a WARNING, not an instrument_fail (SPEC.md's "
                f"instrument-fail list is pre-registered and closed)")
    else:
        d_width = d_width_frac = None

    if instrument_fail_tier_note:
        instrument_fail.append(instrument_fail_tier_note)

    ci = block_bootstrap_delta_ci(subj["values"], null_values, block=block, n_boot=n_boot,
                                  seed=seed)
    band = band_for(ci, instrument_fail=instrument_fail, force_diagnostic=is_fee_tier)
    per_window = regime_bucket_report(subj, null_flat, ext_windows, block, n_boot, seed,
                                      instrument_fail=instrument_fail,
                                      force_diagnostic=is_fee_tier)

    def _windows_with(label):
        return sorted(k for k, v in per_window.items() if v.get("band") == label)

    insufficient_n_windows = _windows_with(BAND_INSUFFICIENT_N)

    # [A1] Amendment 1's co-primary split: entries/exits (the decision-quality read) vs
    # maintenance re-trims (completeness), EACH independently calibrated/banded — never a single
    # pooled band standing in for both (see co_primary_split_report docstring).
    entry_exit_report = co_primary_split_report(subj, null_flat, "entry_exit", block, n_boot,
                                                seed, instrument_fail=instrument_fail,
                                                force_diagnostic=is_fee_tier)
    re_trim_report = co_primary_split_report(subj, null_flat, "re_trim", block, n_boot, seed,
                                             instrument_fail=instrument_fail,
                                             force_diagnostic=is_fee_tier)

    # [A1] net exposure / market return — SEPARATE a directional artifact (long book in a rally)
    # from timing skill. Reported as disclosed diagnostics alongside the band, never gating one:
    # SPEC.md's instrument-fail list is closed and this module does not extend it.
    def _mean_or_none(x):
        x = np.asarray(x, dtype=np.float64)
        x = x[np.isfinite(x)]
        return float(np.mean(x)) if x.size else None

    subj_net_exposure_mean = _mean_or_none(subj["net_exposure"])
    null_net_exposure_mean = _mean_or_none(
        np.concatenate([nf["net_exposure"] for nf in null_flat]) if null_flat else [])
    subj_market_return_mean = _mean_or_none(subj["market_return_values"])
    null_market_return_mean = _mean_or_none(
        np.concatenate([nf["market_return_values"] for nf in null_flat]) if null_flat else [])
    # the directional-artifact read itself: how much of the subject's headline Delta is simply
    # "the held assets' own (unsigned) market moved", i.e. beta the null does NOT share because
    # it is state-blind, vs `ci["delta"]` (the SIGNED, cost-inclusive quantity actually banded).
    market_return_delta = (None if (subj_market_return_mean is None or
                                    null_market_return_mean is None) else
                           float(subj_market_return_mean - null_market_return_mean))

    return {
        "schema": REPORT_SCHEMA,
        # ---- [A1] which leg this report is, and whether it can decide D47. Structural, not a
        # comment: validate_leg_window already raised above if the data didn't match. ----
        "leg": leg,
        "leg_is_binding": (leg == LEG_BINDING),
        "leg_role_note": (LEG_ROLE_NOTE.get(leg) if leg is not None else
                          "NO --leg DECLARED: this report does not assert which SPEC.md leg it "
                          "is, and therefore CANNOT be cited as the D47 binding read (Amendment "
                          "1 requires the binding leg to be structurally unambiguous)."),
        # ---- [A1] D49(a): fee16/slip5 is disclosed, never binding — enforced via band_for's
        # force_diagnostic, not merely by this note ----
        "tier_bp": subj_tier_bp,
        # the whole declared cost triple, so the artifact shows what the carve-out was decided on
        # (a hard read off provenance — see _cost_tier_of / REQUIRED_TRACE_PROVENANCE).
        "fee_bps": subj_cost_tier[1],
        "slip_bps": subj_cost_tier[2],
        "is_fee_tier_diagnostic": is_fee_tier,
        "fee_tier_disclosure": disclosures.FEE_TIER_DIAGNOSTIC_DISCLOSURE if is_fee_tier else None,
        # ---- [A1] subject class / D47 evidence-base membership. The membership read IS
        # structural (computed here from the graded trace's own provenance); the never-pooled
        # rule is NOT — assert_never_pooled has no live caller, so it ships as a disclosed OPEN
        # ITEM below (fix pass MAJOR 2), for a future multi-report aggregator to call. ----
        "subject_class": subject_class,
        "in_d47_evidence_base": in_d47_evidence_base(subject_class),
        "d47_evidence_base_note": (
            None if subject_class is None else
            "part of the D47 evidence base" if in_d47_evidence_base(subject_class) else
            ("EXCLUDED from the D47 evidence base: SPEC.md Amendment 1 excludes the 832 "
             "contrast leg (inherits the unresolved 882 epoch confound) and reports the "
             "search-secondary capacity-ceiling leg separately, never as deployable evidence."
             if subject_class in D47_EXCLUDED_CLASSES else
             f"subject_class={subject_class!r} is not a recognized D47 evidence or excluded "
             f"class")),
        # [A1] fix pass MAJOR 2: the never-pooled rule is an OPEN ITEM, not a closed structural
        # guarantee — stated in the artifact itself rather than left to a reader's inference.
        "never_pooled_enforcement": "OPEN ITEM — see never_pooled_open_item",
        "never_pooled_open_item": disclosures.NEVER_POOLED_OPEN_ITEM,
        "subject_class_is_untagged": subject_class is None,
        # ---- DISCLOSURES FIRST. A reader must hit these before any number. ----
        "READ_THIS_FIRST": disclosures.READ_THIS_FIRST,
        "pre_registration_deviations": disclosures.PRE_REGISTRATION_DEVIATIONS,
        "book_width_disclosure": disclosures.BOOK_WIDTH_DISCLOSURE,
        "bootstrap_power_disclosure": disclosures.BOOTSTRAP_POWER_DISCLOSURE,
        "calibrated_width_disclosure": disclosures.CALIBRATED_WIDTH_DISCLOSURE,
        "calibration_precision_disclosure": disclosures.CALIBRATION_PRECISION_DISCLOSURE,
        "band": band,
        "band_note": BAND_NOTE,
        "instrument_fail_reasons": instrument_fail,
        "warnings": warnings,
        # ---- inputs / provenance ----
        "subject_trace": subject_path,
        "null_traces": list(null_paths),
        "block": int(block), "n_boot": int(n_boot), "bootstrap_seed": int(seed),
        "min_n_for_bootstrap": min_bootstrap_n(block),
        "min_bootstrap_blocks": int(MIN_BOOTSTRAP_BLOCKS),
        # the honest power quantity, measured: how many blocks of `block` each side carries.
        "n_blocks_subject": int(ci["n_blocks_subject"]),
        "n_blocks_null": int(ci["n_blocks_null"]),
        # ---- cycle-3 finding 2: the CALIBRATED banding bar and the table it came from ----
        "calibrated_min_blocks": int(CALIBRATED_BOOTSTRAP_BLOCKS),
        "calibrated_min_n_for_banding": calibrated_bootstrap_n(block),
        "null_coverage_band_min": float(NULL_COVERAGE_BAND_MIN),
        "measured_null_coverage_table": disclosures.NULL_COVERAGE_TABLE,
        "measured_null_coverage_table_overlapping":
            disclosures.NULL_COVERAGE_TABLE_OVERLAPPING,
        "measured_null_coverage_at_this_power": ci.get("measured_null_coverage"),
        "measured_null_coverage_at_this_power_overlapping":
            ci.get("measured_null_coverage_overlapping"),
        "bootstrap_calibration_escalation": disclosures.NULL_COVERAGE_ESCALATION,
        "low_power": bool(ci.get("low_power")),
        "zero_width_ci": bool(ci.get("zero_width_ci")),
        # ---- Amendment 1: the calibrated-width correction actually applied to the pooled CI ----
        "calibration_k": ci.get("calibration_k"),
        "calibration_achieved": bool(ci.get("calibration_achieved")),
        "calibration_target_coverage": float(disclosures.CALIBRATION_TARGET_COVERAGE),
        "calibration_table": disclosures.CALIBRATION_TABLE,
        # ---- the three matched moments SPEC.md:46 promises ----
        "subject_activity": subj_activity,
        "null_activity": null_activity,
        "activity_rate_mismatch_pp": d_rate_pp,
        "long_short_mix_mismatch_pp": d_mix_pp,
        "subject_book_width": subj_width,
        "null_book_width": null_width,
        "book_width_mismatch_slots": d_width,
        "book_width_mismatch_frac_of_universe": d_width_frac,
        # ---- counts, checks, the quantity ----
        "n_subject_non_hold_with_forward": n_pooled_non_hold,
        "n_null_non_hold_with_forward": int(null_values.size),
        "budget_conservation_violations": pv_violations,
        "delta": ci,
        # ---- [A1] co-primary split: entries/exits (decision quality) vs re-trims (completeness),
        # each independently calibrated/banded (co_primary_split_report) ----
        "position_change_split_disclosure": disclosures.POSITION_CHANGE_SPLIT_DISCLOSURE,
        "entry_exit": entry_exit_report,
        "maintenance_re_trim": re_trim_report,
        "n_subject_entry_exit": subj["n_entry_exit"], "n_subject_re_trim": subj["n_re_trim"],
        # ---- [A1] net exposure / forward market return of the held assets: separates a
        # directional artifact (beta) from timing; NEVER gates a band ----
        "net_exposure_disclosure": disclosures.NET_EXPOSURE_DISCLOSURE,
        "subject_net_exposure_mean": subj_net_exposure_mean,
        "null_net_exposure_mean": null_net_exposure_mean,
        "subject_market_return_mean": subj_market_return_mean,
        "null_market_return_mean": null_market_return_mean,
        "market_return_delta": market_return_delta,
        "per_window": per_window,
        "insufficient_n_windows": insufficient_n_windows,
        "low_power_windows": _windows_with(BAND_LOW_POWER),
        "zero_width_ci_windows": _windows_with(BAND_ZERO_WIDTH),
        "non_banded_windows": sorted(k for k, v in per_window.items()
                                     if v.get("band") in NON_BANDS),
        "n_windows_banded": sum(1 for v in per_window.values()
                                if v.get("band") in SUBSTANTIVE_BANDS),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", required=True, help="subject trace JSON (trace.py output)")
    ap.add_argument("--null", action="append", required=True, dest="nulls",
                    help="a null trace JSON (repeat for multiple seeds)")
    ap.add_argument("--block", type=int, default=21)
    ap.add_argument("--n-boot", type=int, default=10000, dest="n_boot")
    ap.add_argument("--bootstrap-seed", type=int, default=0, dest="bootstrap_seed")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-regime-windows", action="store_true", default=False, dest="no_regime",
                    help="skip loading the 831-EXT regime windows (fixture-only runs have no "
                         "overlap with them anyway; ALL bucket is always computed)")
    ap.add_argument("--leg", choices=LEGS, default=None,
                    help="[A1] which SPEC.md leg this report is (oos-binding = the D47 verdict "
                         "source; sanity = reported, never the verdict source). STRUCTURALLY "
                         "validated against every trace's decision timestamps (validate_leg_"
                         "window) — a mismatch raises rather than mislabeling. Optional so "
                         "fixture-smoke runs (whose window matches neither pre-registered leg) "
                         "keep working; a real-leg invocation MUST pass this to be citable as "
                         "binding.")
    args = ap.parse_args()

    ext_windows = None
    if not args.no_regime:
        EV_PY = os.path.join(_REPO, "experiments", "835_stage4_eval", "eval_vector.py")
        import importlib.util
        spec = importlib.util.spec_from_file_location("eval_vector835_892r", EV_PY)
        ev = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ev)
        ext_windows = ev.load_ext_windows()

    report = compare(args.subject, args.nulls, block=args.block, n_boot=args.n_boot,
                     seed=args.bootstrap_seed, ext_windows=ext_windows, leg=args.leg)
    # cycle-3 minor: atomic (tmp + os.replace) — every refusal above happens before the write.
    trace_mod.write_json_atomic(args.out, report, indent=2, default=float)
    print(f"[892/report] wrote {args.out} — band={report['band']} "
         f"delta={report['delta']['delta']} ci=[{report['delta']['ci_lo']}, "
         f"{report['delta']['ci_hi']}] instrument_fail={report['instrument_fail_reasons']}")
    print(f"[892/report] power in BLOCKS of {report['block']} (what the bootstrap consumes): "
         f"subject={report['n_blocks_subject']} null={report['n_blocks_null']} "
         f"(degeneracy floor: {report['min_bootstrap_blocks']}; MEASURED banding bar: "
         f"{report['calibrated_min_blocks']} full blocks per side. Measured true-null coverage at "
         f"this power: ~{100.0 * (report['measured_null_coverage_at_this_power'] or 0.0):.0f}% "
         f"i.i.d. / ~"
         f"{100.0 * (report['measured_null_coverage_at_this_power_overlapping'] or 0.0):.0f}% "
         f"under realistic overlapping-horizon dependence, vs a nominal 95%)")
    if report["insufficient_n_windows"]:
        print(f"[892/report] INSUFFICIENT-N (no band, no CI) buckets: "
             f"{report['insufficient_n_windows']}")
    if report["low_power_windows"]:
        print(f"[892/report] LOW-POWER (CI reported, NO BAND — measured coverage below "
             f"{100.0 * report['null_coverage_band_min']:.0f}% of nominal) buckets: "
             f"{report['low_power_windows']}")
    if report["zero_width_ci_windows"]:
        print(f"[892/report] ZERO-WIDTH-CI (no band — the inputs carry no within-side variance) "
             f"buckets: {report['zero_width_ci_windows']}")
    for w in report["warnings"]:
        print(f"[892/report] WARNING: {w}")
    print(f"[892/report] leg={report['leg']!r} leg_is_binding={report['leg_is_binding']} — "
         f"{report['leg_role_note']}")
    if report["is_fee_tier_diagnostic"]:
        print(f"[892/report] FEE-TIER DIAGNOSTIC ONLY (tier_bp={report['tier_bp']}): band "
             f"forced to {BAND_FEE_TIER_DIAGNOSTIC!r} per D49(a) — never binding")
    print(f"[892/report] subject_class={report['subject_class']!r} "
         f"in_d47_evidence_base={report['in_d47_evidence_base']}"
         + (f" ({report['d47_evidence_base_note']})" if report['d47_evidence_base_note'] else ""))
    print(f"[892/report] never-pooled: {report['never_pooled_enforcement']} — "
         f"assert_never_pooled exists and is tested but has NO live caller in this module (no "
         f"multi-report aggregator exists); an aggregator MUST call it, and an untagged subject "
         f"(subject_class_is_untagged={report['subject_class_is_untagged']}) would not trip it.")
    print(f"[892/report] co-primary split: entry/exit band={report['entry_exit']['band']} "
         f"(n={report['n_subject_entry_exit']}) / maintenance re-trim "
         f"band={report['maintenance_re_trim']['band']} (n={report['n_subject_re_trim']})")
    print(f"[892/report] net_exposure subject={report['subject_net_exposure_mean']!r} "
         f"null={report['null_net_exposure_mean']!r}; market_return_delta="
         f"{report['market_return_delta']!r} (directional-artifact diagnostic, never bands)")
    # The deviation must be impossible to miss on the console too, not only inside the JSON.
    print(f"[892/report] {disclosures.READ_THIS_FIRST}")


if __name__ == "__main__":
    main()
