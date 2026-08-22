#!/usr/bin/env python3
"""TASK_897 D55(a) — `calibration.py`'s method (reused from 892) and the certified table it
produced for `honest_estimator.block_bootstrap_sharpe_ci`.

Run: python -m pytest experiments/897_corrected_809/test_calibration.py -v
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calibration as cal  # noqa: E402
import honest_estimator as hest  # noqa: E402


def test_new_max_block_matches_estimator_default():
    """The cap this module certified against and the cap the estimator actually uses must be
    the SAME number, by construction (import, not duplication) — a drift here would mean the
    certified table no longer describes the code path it claims to."""
    assert cal.NEW_MAX_BLOCK == hest.MAX_BLOCK_DEFAULT == 60
    assert cal.NEW_MAX_BLOCK > cal.OLD_MAX_BLOCK


def test_block_for_matches_compute_honest_metrics_formula():
    """`calibration.block_for` must be the exact expression
    `compute_honest_metrics` uses (`max(min_block, min(max_block, n_dec // 20))`), not a
    reimplementation that could drift."""
    for n_dec in (1, 10, 39, 40, 100, 1199, 1200, 1201, 50000):
        expected = max(2, min(60, n_dec // 20))
        assert cal.block_for(n_dec) == expected


@pytest.mark.parametrize("pe,lo,hi,expected", [
    (0.0, -1.0, 1.0, 0.0),
    (2.0, 1.0, 3.0, 2.0),        # d_lo = 1.0 -> k = 2/1 = 2.0
    (2.0, 2.0, 3.0, np.inf),     # d_lo <= 0 -> degenerate, no finite k rescues it
    (-2.0, -3.0, -1.0, 2.0),     # symmetric case: d_hi = 1.0 -> k = 2/1 = 2.0
    (-2.0, -3.0, -2.0, np.inf),  # d_hi <= 0 -> degenerate
])
def test_k_needed_for_trial_closed_form(pe, lo, hi, expected):
    got = cal.k_needed_for_trial(pe, lo, hi)
    if np.isinf(expected):
        assert np.isinf(got)
    else:
        assert got == pytest.approx(expected)


def test_k_needed_for_trial_is_verified_against_a_brute_force_scan():
    """Independent cross-check: for a handful of (pe, lo, hi) triples, brute-force scan k on a
    fine grid and confirm the SMALLEST k at which the inflated interval contains 0 matches the
    closed form to within the grid's resolution."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        pe = rng.uniform(-5, 5)
        lo = pe - abs(rng.uniform(0.01, 5))
        hi = pe + abs(rng.uniform(0.01, 5))
        closed = cal.k_needed_for_trial(pe, lo, hi)
        if not np.isfinite(closed):
            continue
        grid = np.linspace(0, closed + 1.0, 20000)
        grid_step = grid[1] - grid[0]
        contains = (pe - grid * (pe - lo) <= 0) & (pe + grid * (hi - pe) >= 0)
        if not contains.any():
            continue
        brute = grid[np.argmax(contains)]
        assert abs(brute - closed) < 2 * grid_step, (pe, lo, hi, closed, brute, grid_step)


def _entry(n_dec, k, achieved=True):
    """A table entry in the shipped shape — `block` is the REAL block at that n_dec, which is
    what the same-block lookup keys on."""
    return {"block": cal.block_for(n_dec), "k": float(k), "achieved": bool(achieved)}


def test_calibrated_k_for_nearest_below_within_the_same_block_and_achieved_only():
    # All of these are regime-B (block == cap) strata, so nearest-below applies within one block.
    table = {
        1200: _entry(1200, 5.0),
        2000: _entry(2000, 2.0, achieved=False),   # NOT achieved -> must never be selected
        3000: _entry(3000, 1.5),
        10000: _entry(10000, 1.1),
    }
    assert {e["block"] for e in table.values()} == {60}
    # exactly at an achieved entry
    assert cal.calibrated_k_for(3000, table) == (1.5, True, 3000)
    # between two achieved entries -> nearest BELOW, not nearest overall
    assert cal.calibrated_k_for(4000, table) == (1.5, True, 3000)
    # sits above an UNACHIEVED entry (2000) but below the next achieved one (3000) -> falls back
    # to 1200 (achieved), never silently uses 2000's unachieved k
    assert cal.calibrated_k_for(2500, table) == (5.0, True, 1200)
    # below every entry at this block -> nothing certified
    assert cal.calibrated_k_for(1201, {10000: _entry(10000, 1.1)}) == (None, False, None)
    # above the largest certified stratum in the saturated block -> refused, never extrapolated
    assert cal.calibrated_k_for(50000, table) == (None, False, None)


def test_calibrated_k_for_never_borrows_across_a_BLOCK_BOUNDARY():
    """THE MAJOR-1 FIX. `block_for` is a STEP function of n_dec that saturates at the cap, so a
    nearest-below-by-n_dec lookup handed a block-60 cell the k measured at block 50. A stratum is
    now eligible only if it was measured at the QUERY's OWN block."""
    assert cal.block_for(1000) == 50 and cal.block_for(1200) == 60   # the step
    block50_only = {1000: _entry(1000, 1.34)}
    assert cal.calibrated_k_for(1000, block50_only) == (1.34, True, 1000)
    for n_dec in (1200, 1500, 2000, 30000):
        assert cal.calibrated_k_for(n_dec, block50_only) == (None, False, None), n_dec
    # and the reverse direction: a cap-block stratum must not certify a sub-cap cell
    block60_only = {1200: _entry(1200, 1.47), 3000: _entry(3000, 1.5)}
    for n_dec in (40, 300, 1000, 1199):
        assert cal.calibrated_k_for(n_dec, block60_only) == (None, False, None), n_dec


def test_calibrated_k_for_does_not_take_the_largest_k():
    """Regression for the documented bug this convention avoids: k DECREASES with n_dec inside
    the cap block, so 'take the largest achieved k at or below n_dec' would silently collapse to
    always using the smallest calibrated stratum. The correct rule is nearest-BELOW by n_dec
    (within the same block), not largest-by-k."""
    table = {1200: _entry(1200, 5.0), 10000: _entry(10000, 1.2)}
    k, achieved, src = cal.calibrated_k_for(10000, table)
    assert (k, src) == (1.2, 10000)   # NOT (5.0, 1200), which "largest k" would wrongly pick


def test_strata_are_generated_from_block_for_so_every_block_has_one():
    """The strata are DERIVED from `block_for`'s own step function, not hand-picked: there is a
    stratum at every block-transition point, so no reachable (block, n_dec) pairing is left
    without a same-block stratum to borrow from."""
    strata = cal.DEFAULT_N_DEC_STRATA
    blocks_with_a_stratum = {cal.block_for(nd) for nd in strata}
    reachable = {cal.block_for(nd) for nd in range(cal.MIN_N_DEC_FOR_CALIBRATION, 20 * 60 + 40)}
    assert reachable <= blocks_with_a_stratum, sorted(reachable - blocks_with_a_stratum)
    # every regime-A stratum sits at the BOTTOM of its block's n_dec range (n_dec == 20*block)
    for nd in strata:
        b = cal.block_for(nd)
        if b < cal.NEW_MAX_BLOCK:
            assert nd == 20 * b, (nd, b)
    # ...and the cap regime carries an explicit ladder, starting exactly at the transition point
    cap_strata = sorted(nd for nd in strata if cal.block_for(nd) == cal.NEW_MAX_BLOCK)
    assert cap_strata[0] == 20 * cal.NEW_MAX_BLOCK == 1200
    assert len(cap_strata) >= 5


def test_every_n_dec_in_the_certified_range_resolves_to_a_same_block_stratum():
    """Exhaustive over regime A plus a dense regime-B sample: whenever `calibrated_k_for`
    certifies, the stratum it picked was measured at EXACTLY the block the cell will run at."""
    tbl = cal.MEASURED_CALIBRATION_TABLE
    n_certified = 0
    probe = list(range(2, 1400)) + [1500, 2500, 4000, 9000, 25000, 90000, 299999, 300001, 10 ** 7]
    for n_dec in probe:
        k, achieved, src = cal.calibrated_k_for(n_dec, tbl)
        if not achieved:
            continue
        n_certified += 1
        assert tbl[src]["block"] == cal.block_for(n_dec), (n_dec, src)
        assert src <= n_dec
    assert n_certified > 1000


# ---------------------------------------------------------------- the certified table itself

def test_measured_calibration_table_claims_exactly_what_it_measured():
    """D55(a)'s deliverable: NOTHING is rounded up. Every stratum flagged `achieved=True` really
    validated coverage >= TARGET_COVERAGE on an independent batch, and every stratum that came
    in under the bar is flagged False WITH its measured number and reason — not re-rolled until
    it passed (selection on noise), not quietly dropped from the table."""
    assert set(cal.MEASURED_CALIBRATION_TABLE) == set(cal.DEFAULT_N_DEC_STRATA), (
        "the shipped table must cover exactly the strata the grid generator derives from "
        "`block_for` — a missing block is a (block, n_dec) region with no same-block stratum")
    n_failed = 0
    for n_dec, entry in cal.MEASURED_CALIBRATION_TABLE.items():
        assert entry["block"] == cal.block_for(n_dec)    # block matches the live formula
        assert entry["k"] < 8.0, (n_dec, entry)          # comfortably under k_cap
        if entry["achieved"]:
            assert entry["validation_coverage_overlapping"] >= cal.TARGET_COVERAGE, (n_dec, entry)
            assert entry["reason"] is None, (n_dec, entry)
        else:
            n_failed += 1
            assert entry["reason"], (n_dec, entry)
            # a stratum is uncertified either because its OWN batch missed, or because the
            # borrowed pairing that reads it missed — both must be recorded, never inferred
            own_missed = entry["validation_coverage_overlapping"] < cal.TARGET_COVERAGE
            borrowed_missed = any(
                e["borrowed_from_n_dec"] == n_dec and not e["meets_target"]
                for e in cal.MEASURED_BORROWED_PAIRING_COVERAGE.values())
            assert own_missed or borrowed_missed, (n_dec, entry)
            # and the lookup must actually refuse it
            assert cal.calibrated_k_for(n_dec, cal.MEASURED_CALIBRATION_TABLE) == (None, False,
                                                                                   None)
    # the shipped table is mostly certified, but not unanimously — a table with zero failures at
    # 300 validation trials over 67 strata would itself be the suspicious outcome.
    assert n_failed <= 8, f"{n_failed}/67 strata failed validation — re-measure the table"
    assert sum(1 for e in cal.MEASURED_CALIBRATION_TABLE.values() if e["achieved"]) >= 55


def test_stratum_seed_is_the_rule_build_calibration_table_actually_uses():
    """The premise of the exact test below. `stratum_seed` must be the SAME rule the harness
    applies (`seed + 1000*index` over the sorted strata), or a subset re-run would be a different
    measurement wearing the same name."""
    strata = sorted(set(int(n) for n in cal.DEFAULT_N_DEC_STRATA))
    assert cal.stratum_seed(strata[0]) == 0
    assert cal.stratum_seed(1000) == 1000 * strata.index(1000) == 48000
    assert cal.stratum_seed(3000, seed=7) == 7 + 1000 * strata.index(3000)
    # and it is index-based, not value-based: a custom strata list re-indexes
    assert cal.stratum_seed(1000, n_dec_strata=(40, 1000)) == 1000
    # the harness's own docstring must not have drifted back to promising MC noise on a re-run
    doc = cal.build_calibration_table.__doc__
    assert "BIT-DETERMINISTIC" in doc


# NAMED SUBSET for the exact-reproduction test below, chosen by a stated rule and NOT by which
# strata happen to agree (all 11 probed strata agreed — 100/500/520/600/660/740/1200/3000 too):
#   * 40   — the SMALLEST certified stratum (block 2), the cheapest row there is;
#   * 620  — the one row whose `achieved`/`reason` are a demotion OVERLAY rather than
#            `calibrate_block` output, so the test pins that distinction instead of hiding it;
#   * 1000 — block 50, the last regime-A stratum the prose cites (the k a block-60 cell used to
#            borrow across the step, i.e. the MAJOR-1 bug's own example).
EXACT_REPRO_STRATA = (40, 620, 1000)


def test_measured_calibration_table_reproduces_exactly_at_named_strata():
    """The header claims the advertised command reproduces this table VERBATIM. That is now a
    testable claim rather than a hedged one: `build_calibration_table` seeds every stratum
    deterministically (`stratum_seed`), so a re-run is BIT-IDENTICAL and exact equality is the
    honest bar. An earlier revision claimed the opposite in the same breath as "reproducible
    verbatim" — that the re-run would NOT be bit-identical and that which strata fall under the
    bar would move run to run. Both were false; this test is what makes them un-reassertable.

    SUBSET, DISCLOSED: 3 of the 67 strata (`EXACT_REPRO_STRATA`, chosen by the rule above), run at
    exactly the seeds the full sweep gives them. The full 67-stratum reproduction costs tens of
    CPU-minutes; one stratum costs ~21-45s. COST OF THIS TEST, MEASURED: ~27s wall (the three run
    in a process pool, so wall clock is the slowest of them, not the sum). A subset test that is
    EXACT and says it is a subset beats a full test nobody can afford to run.

    620 is the disclosed exception and is asserted AS an exception: every MEASURED field matches
    exactly, while `achieved`/`reason` differ because the shipped row carries the borrowed-pairing
    demotion, which `calibrate_block` (a stratum-only measurement) cannot know about."""
    fresh = cal.reproduce_strata(EXACT_REPRO_STRATA, n_jobs=len(EXACT_REPRO_STRATA))
    measured = ("block", "k", "validation_coverage_overlapping", "validation_coverage_iid",
                "validation_n_overlapping", "validation_n_iid")
    for n_dec in EXACT_REPRO_STRATA:
        pinned, got = cal.MEASURED_CALIBRATION_TABLE[n_dec], fresh[n_dec]
        for f in measured:
            assert got[f] == pinned[f], (n_dec, f, got[f], pinned[f])
        if n_dec == 620:
            # the overlay: the harness alone would have certified this stratum
            assert got["achieved"] is True and pinned["achieved"] is False
            assert got["reason"] is None and "BORROWED pairing" in pinned["reason"]
            assert not cal.MEASURED_BORROWED_PAIRING_COVERAGE[639]["meets_target"]
        else:
            assert got["achieved"] == pinned["achieved"], n_dec
            assert got["reason"] == pinned["reason"], n_dec


def test_trim_table_for_disclosure_exposes_only_the_documented_fields():
    trimmed = cal.trim_table_for_disclosure(cal.MEASURED_CALIBRATION_TABLE)
    for entry in trimmed.values():
        assert set(entry.keys()) == set(cal.DISCLOSURE_FIELDS)


def test_min_n_dec_for_calibration_is_below_every_shipped_stratum():
    assert all(nd >= cal.MIN_N_DEC_FOR_CALIBRATION for nd in cal.MEASURED_CALIBRATION_TABLE)


# ------------------------------------------- the BORROWED pairings (MAJOR-1 certification half)

def test_borrowed_pairing_coverage_is_measured_at_the_pairings_cells_actually_use():
    """A pinned stratum is not what a real cell reads — real cells land BETWEEN strata and borrow.
    This pins that achieved coverage was RE-MEASURED at those borrowed (block, n_dec) pairings,
    at the cell's OWN block, using the k `calibrated_k_for` would actually hand it — and that
    every certified one cleared the target."""
    tbl = cal.MEASURED_BORROWED_PAIRING_COVERAGE
    probe_n = {int(p[0]) for p in cal.DEFAULT_BORROWED_PROBE_POINTS}
    assert set(tbl) == probe_n
    n_certified = 0
    for n_dec, e in tbl.items():
        assert e["block"] == cal.block_for(n_dec), (n_dec, e)
        # none of these points may BE a stratum — the whole point is that they are borrowers
        assert n_dec not in cal.MEASURED_CALIBRATION_TABLE, n_dec
        k, achieved, src = cal.calibrated_k_for(n_dec, cal.MEASURED_CALIBRATION_TABLE)
        # THE LOAD-BEARING INVARIANT, in both directions: a probe point that did not meet target
        # must NOT be servable by the lookup (whether it was never certified, or whether THIS
        # failure is what demoted its stratum), and a probe point the lookup does serve must have
        # measured >= target at the exact pairing it serves.
        if not e["meets_target"]:
            assert achieved is False, (
                n_dec, "borrowed coverage missed the target but the lookup still serves this "
                       "pairing — the stratum it reads must be demoted")
            assert e.get("reason"), (n_dec, e)
            continue
        assert achieved is True, (n_dec, e)
        n_certified += 1
        assert e["borrowed_from_n_dec"] == src
        assert e["borrowed_k"] == pytest.approx(k)
        assert cal.MEASURED_CALIBRATION_TABLE[src]["block"] == e["block"]
        assert e["coverage_overlapping"] >= cal.TARGET_COVERAGE, (n_dec, e)
    assert n_certified >= 12, "the borrowed-pairing certification must cover most of the grid"
    # and the check must have real teeth: at least one probe DID fail and cost a stratum
    demoted = [nd for nd, e in cal.MEASURED_BORROWED_PAIRING_COVERAGE.items()
               if not e["meets_target"] and e["borrowed_from_n_dec"] is not None]
    assert demoted, ("the borrowed-pairing check must be able to fail; if a re-measurement ever "
                     "passes everywhere, relax this assertion deliberately, do not delete it")
    for nd in demoted:
        src = cal.MEASURED_BORROWED_PAIRING_COVERAGE[nd]["borrowed_from_n_dec"]
        assert cal.MEASURED_CALIBRATION_TABLE[src]["achieved"] is False, (nd, src)


def test_the_block_transition_pairing_that_the_old_lookup_got_wrong_is_covered():
    """The specific MAJOR-1 defect: n_dec just above 1200 runs at block 60 but the old n_dec-only
    lookup handed it the n_dec=1000 / block=50 stratum's k, whose re-measured coverage sat ON the
    0.93 target (0.930/0.9375/0.9475) instead of inside the advertised margin. There must now be
    a stratum AT the transition and a borrowed probe just above it."""
    assert cal.block_for(1199) == 59 and cal.block_for(1200) == cal.NEW_MAX_BLOCK
    assert 1200 in cal.MEASURED_CALIBRATION_TABLE
    assert cal.MEASURED_CALIBRATION_TABLE[1200]["block"] == cal.NEW_MAX_BLOCK
    for n_dec in (1250, 1400):
        k, achieved, src = cal.calibrated_k_for(n_dec, cal.MEASURED_CALIBRATION_TABLE)
        assert achieved and cal.MEASURED_CALIBRATION_TABLE[src]["block"] == cal.NEW_MAX_BLOCK
    assert 1250 in cal.MEASURED_BORROWED_PAIRING_COVERAGE
    assert (cal.MEASURED_BORROWED_PAIRING_COVERAGE[1250]["coverage_overlapping"]
            >= cal.TARGET_COVERAGE)


def test_certified_region_description_matches_the_table_it_describes():
    txt = cal.certified_region_description()
    assert "REGIME A" in txt and "REGIME B" in txt and "UNCALIBRATED" in txt
    cap_nds = sorted(nd for nd, e in cal.MEASURED_CALIBRATION_TABLE.items()
                     if e["block"] == cal.NEW_MAX_BLOCK and e["achieved"])
    assert str(cap_nds[-1]) in txt                       # the certified ceiling is stated
    assert str(cal.MIN_N_DEC_FOR_CALIBRATION) in txt     # and the floor
    # the stated ceiling is real: one past it is refused
    assert cal.calibrated_k_for(cap_nds[-1], cal.MEASURED_CALIBRATION_TABLE)[1] is True
    assert cal.calibrated_k_for(cap_nds[-1] + 1, cal.MEASURED_CALIBRATION_TABLE)[1] is False


# --------------------------------------------------------- the block-cap rationale (MAJOR 4)

def test_block_cap_rationale_arithmetic_is_recomputed_not_asserted():
    """The rationale ships VERBATIM into the artifact. Its previous item (c) claimed 'at least 8
    non-overlapping blocks at the smallest calibrated stratum (n_dec=500 -> 500//60=8 blocks)' —
    false twice over (500 was not a stratum, and block_for(500) is 25, giving 20 blocks). Every
    number the corrected clause states is recomputed here from the live code."""
    r = cal.BLOCK_CAP_SELECTION_RATIONALE
    # the false claim may appear ONLY inside its own retraction, never as a live assertion
    assert "arithmetically FALSE" in r
    assert r.index("arithmetically FALSE") < r.index("500//60=8")
    # MINOR 3 (fix cycle 3): the RETRACTION itself used to be false. It said "500 was not a
    # stratum at all" — but 500 IS a stratum in the shipped table and IS in DEFAULT_N_DEC_STRATA.
    # The true claim is that 500 was not the SMALLEST stratum (40 is). A wrong retraction shipping
    # verbatim into the artifact is worse than the error it retracts, so both directions are
    # pinned: the corrected wording present, the false wording ABSENT.
    assert 500 in cal.MEASURED_CALIBRATION_TABLE
    assert cal.MEASURED_CALIBRATION_TABLE[500]["block"] == 25
    assert cal.MEASURED_CALIBRATION_TABLE[500]["achieved"] is True
    assert 500 in cal.DEFAULT_N_DEC_STRATA
    assert min(cal.MEASURED_CALIBRATION_TABLE) == 40 != 500
    assert "500 was not the SMALLEST stratum" in r
    # the false wording survives ONLY as a quotation inside its own retraction, never as a claim
    assert r.index("500 was not the SMALLEST stratum") < r.index("500 was not a stratum at all")
    assert "previously read '500 was not a stratum at all', which is false" in r
    assert r.count("500 was not a stratum at all") == 1
    assert "block_for(500, cap=60) is 25, not 60, giving 500//25 = 20 blocks" in r
    assert cal.block_for(500, max_block=60) == 25 and 500 // 25 == 20
    # the corrected claims, each re-derived:
    smallest = min(cal.MEASURED_CALIBRATION_TABLE)
    assert smallest == 40 and cal.block_for(40) == 2 and 40 // cal.block_for(40) == 20
    assert "n_dec=40, block 2 -> 40//2 = 20 blocks" in r
    transition = 20 * cal.NEW_MAX_BLOCK
    assert transition // cal.block_for(transition) == 20
    assert "n_dec = 20*cap -> 20 blocks" in r
    assert 3000 // cal.block_for(3000, max_block=60) == 50
    assert 3000 // cal.block_for(3000, max_block=100) == 30
    assert "cap=60 gives block 60 and 50 non-overlapping blocks" in r
    assert "cap=100 gives block 100 and only 30" in r
    assert 300000 // cal.block_for(300000, 60) == 5000
    assert 300000 // cal.block_for(300000, 100) == 3000
    assert "at n_dec=300000 it is 5000 vs 3000" in r
    # the honest caveat that raising the cap was NOT shown strictly required must SURVIVE
    assert "not shown to be STRICTLY REQUIRED" in r
    assert "not a value forced by a sharp measured failure at 20" in r
    # ...and the minimum block count anywhere in the shipped table really is 20
    assert min(nd // e["block"] for nd, e in cal.MEASURED_CALIBRATION_TABLE.items()) == 20
    assert "certified minimum block count anywhere in the shipped table is 20" in r


# ------------------------- the certification's CONDITION (fix cycle 3, MAJOR 1(a)) -----------

def test_no_shipped_string_calls_dep_equals_block_the_worst_case():
    """MAJOR 1(a). Every `k` was measured at dep == block, which is the EASIEST point of this
    module's own dependence sweep. Three strings used to call it "the worst case", one of them
    `BLOCK_CAP_SELECTION_RATIONALE`, which ships VERBATIM into the artifact where a reader checks
    the calibration's legitimacy. The phrase may now appear only inside its own retraction."""
    import inspect
    src = inspect.getsource(cal)
    for m in ("the matched-dependence worst case", "the worst case D49/892 characterize"):
        assert m not in src.replace("'" + m + "'", "").replace('"' + m + '"', ""), m
    r = cal.BLOCK_CAP_SELECTION_RATIONALE
    assert "THAT IS NOT THE WORST CASE" in r
    assert "EASIEST POINT" in r
    # and the retraction of the coverage sentence is present and correctly attributed
    assert "RE-CALIBRATED AT EACH DEPENDENCE LEVEL" in r
    assert "IT IS NOT THE COVERAGE OF THE k THIS MODULE SHIPS" in r
    assert r.index("coverage held") > r.index("THE COVERAGE")   # only inside its retraction


def test_certification_conditionality_is_stated_with_the_measured_numbers():
    t = cal.CERTIFICATION_IS_CONDITIONAL_ON_DEPENDENCE
    assert "dep == block` ONLY" in t
    assert "0.530" in t                       # tied back to the number that raised D55
    for needle in ("0.528", "0.516", "0.552", "EASIEST point", "FROM SAMPLE SIZE, NOT FROM THE"):
        assert needle in t, needle
    # the safe-fallback claim (MAJOR 1(c)) must be stated, not implied
    assert "STRICTLY BETTER" in t.upper()
    # and the numbers quoted in prose must be the numbers in the measured record
    rec = cal.MEASURED_SHIPPED_K_AT_DEPENDENCE_MISMATCH
    for n_dec, e in rec.items():
        assert f"{e['shipped_k']:.4f}" in t, n_dec
        for mult in (1, 2, 4, 8):
            for c in e["by_dep_multiple"][mult]["coverage_by_seed"]:
                assert f"{c:.3f}" in t, (n_dec, mult, c)


def test_shipped_k_mismatch_record_uses_the_k_THAT_IS_ACTUALLY_SHIPPED():
    """The stress table is worthless if it stresses some other k. Every row's `shipped_k` must be
    the exact value `calibrated_k_for` would hand a cell at that n_dec, and its `block` the block
    that cell would run at."""
    for n_dec, e in cal.MEASURED_SHIPPED_K_AT_DEPENDENCE_MISMATCH.items():
        k, achieved, src = cal.calibrated_k_for(n_dec, cal.MEASURED_CALIBRATION_TABLE)
        assert achieved and src == n_dec
        assert e["shipped_k"] == pytest.approx(k)
        assert e["block"] == cal.block_for(n_dec)
        assert e["by_dep_multiple"][1]["dep"] == e["block"]


@pytest.mark.parametrize("record_name", ["MEASURED_SHIPPED_K_AT_DEPENDENCE_MISMATCH",
                                         "MEASURED_SHIPPED_K_AT_DEPENDENCE_MISMATCH_HALTED_CYCLE"])
def test_shipped_k_coverage_collapses_past_its_certified_dependence(record_name):
    """The finding itself, pinned off the record: at dep == block the shipped k sits at the bar;
    by 8x the block it is at the 0.53 magnitude that caused D55 to be raised. Monotone, and never
    allowed to be quietly re-recorded as 'fine'.

    PARAMETRIZED OVER BOTH RECORDS ON PURPOSE (D57 fix pass item 4, redone): the re-measured pin
    and the superseded halted-cycle measurement it replaced must BOTH support this conclusion, so
    the re-pinning is demonstrably a provenance fix and not a finding that moved."""
    record = getattr(cal, record_name)
    record = record["record"] if "record" in record else record
    for n_dec, e in record.items():
        by = e["by_dep_multiple"]
        matched = np.mean(by[1]["coverage_by_seed"])
        worst = np.mean(by[8]["coverage_by_seed"])
        # AT the bar, not comfortably above it. Each seed is a 250-trial binomial estimate, so
        # its SE near p=0.93 is sqrt(.93*.07/250) = 1.61pp and the 2-seed mean's is 1.14pp; the
        # allowance below is 0.033, i.e. 2.89 SE of that mean — the test is STRICTER than the
        # "3 SE" this comment used to claim (3 SE would be 3.42pp). The allowance is kept as it
        # is and the sentence is corrected to match it, never the other way round: widening a
        # bar so a claim about it becomes true is the move this row exists to eliminate.
        # (D57 fix cycle 4, item C4.) (Reproducible pin: 0.924 at
        # n_dec=3000 and 0.960 at n_dec=1000 — one half a SE under the 0.93 bar, one 2.6 over.)
        # This is a statement about 250-trial sampling error ACROSS SEEDS, NOT a tolerance on
        # re-running the sweep: at fixed seeds the sweep is bit-deterministic (see
        # `test_shipped_k_mismatch_cli_default_reproduces_the_pinned_record_exactly`).
        assert matched >= cal.TARGET_COVERAGE - 0.033, (n_dec, matched)
        # the collapse, by contrast, is ~30 SE — unambiguous under any such allowance
        assert worst < 0.60, (n_dec, worst)
        assert np.mean(by[2]["coverage_by_seed"]) < matched
        assert np.mean(by[4]["coverage_by_seed"]) < np.mean(by[2]["coverage_by_seed"])


def test_superseded_shipped_k_record_is_kept_and_provenance_stamped():
    """D39 doctrine, applied to this re-pin: the halted-cycle numbers the reproducible pin
    replaced are a real measurement whose exact seed provenance is unrecoverable. They must stay
    VISIBLE next to the new ones, stamped, rather than being overwritten out of existence — and
    they must actually be the superseded values, not a copy of the new pin."""
    hist = cal.MEASURED_SHIPPED_K_AT_DEPENDENCE_MISMATCH_HALTED_CYCLE
    prov = hist["provenance"]
    for needle in ("halted cycle", "2026-08-22", "20001..20105", "NOT recoverable", "SUPERSEDED"):
        assert needle in prov, needle
    # same shape as the live record, so `shipped_k_mismatch_rows` can render either
    assert set(hist["record"]) == set(cal.MEASURED_SHIPPED_K_AT_DEPENDENCE_MISMATCH)
    assert len(cal.shipped_k_mismatch_rows(hist["record"])) == len(cal.shipped_k_mismatch_rows())
    # it is genuinely the OLD measurement, and it stresses the SAME shipped k
    assert hist["record"] != cal.MEASURED_SHIPPED_K_AT_DEPENDENCE_MISMATCH
    for n_dec, e in hist["record"].items():
        live = cal.MEASURED_SHIPPED_K_AT_DEPENDENCE_MISMATCH[n_dec]
        assert e["shipped_k"] == live["shipped_k"] and e["block"] == live["block"]
    # and the artifact/prose figures are read off the LIVE pin, never off the historical one
    assert "0.944" not in cal.CERTIFICATION_IS_CONDITIONAL_ON_DEPENDENCE


def test_shipped_k_mismatch_reproduces_live_at_a_reduced_budget():
    """Cheap live re-derivation through the real code path, so the pinned record cannot go stale
    relative to the estimator it describes. A DIFFERENT (much smaller) trial/boot budget than the
    pinned record, so it checks the DIRECTION only, not the pinned values — it is nonetheless
    itself bit-deterministic at seed 3, like every path in this function."""
    res = cal.measure_shipped_k_at_dependence_mismatch(
        points=((1000, None),), dep_multiples=(1, 8), seeds=(3,), n_trials=60, n_boot=150)
    by = res[1000]["by_dep_multiple"]
    assert res[1000]["shipped_k"] == cal.MEASURED_CALIBRATION_TABLE[1000]["k"]
    assert by[1]["coverage_by_seed"][0] > by[8]["coverage_by_seed"][0] + 0.15


def test_shipped_k_mismatch_cli_default_seeds_match_the_pinned_record():
    """D57 fix pass item 4: `python calibration.py --sweep-shipped-k-mismatch` calls
    `measure_shipped_k_at_dependence_mismatch()` with NO overrides, so the function's default
    `seeds` IS what that advertised command runs at. The comment pinning
    `MEASURED_SHIPPED_K_AT_DEPENDENCE_MISMATCH` names the seeds it was measured at — the default
    must equal that pair, or the advertised command does not reproduce the constant it claims to
    (it once read seeds=(0, 1), a different pair entirely)."""
    import inspect
    default_seeds = inspect.signature(cal.measure_shipped_k_at_dependence_mismatch).parameters[
        "seeds"].default
    assert tuple(default_seeds) == (20001, 20105)


def test_measured_coverage_at_k_is_bit_deterministic_at_a_fixed_seed():
    """The premise of the exact-equality test below, proved rather than assumed: the measurement
    path is a pure function of its arguments, so re-running it at a recorded seed reproduces the
    recorded number BIT-IDENTICALLY. There is no run-to-run Monte Carlo noise on this path, and
    therefore no MC tolerance may ever be disclosed against a re-run of it (an earlier revision
    disclosed one, at ~3pp and then ~6pp, to absorb a seed-provenance gap — that framing named a
    mechanism that does not exist here)."""
    k = cal.MEASURED_CALIBRATION_TABLE[3000]["k"]
    vals = [cal.measured_coverage_at_k(3000, k, "overlapping", 40, 80,
                                      np.random.default_rng(20001), dep=60) for _ in range(3)]
    assert len(set(vals)) == 1, vals


def test_shipped_k_mismatch_cli_default_reproduces_the_pinned_record_exactly():
    """The advertised `python calibration.py --sweep-shipped-k-mismatch` must reproduce
    `MEASURED_SHIPPED_K_AT_DEPENDENCE_MISMATCH` EXACTLY — no tolerance. The path is deterministic
    at fixed seeds (test above), the default seeds are the recorded ones (test above), so exact
    equality is the honest bar and anything looser would be a bar moved to fit a result.

    COST: this runs the real default sweep (2 n_dec x 4 dep multiples x 2 seeds, 250 trials x 400
    resamples each) and is by far the slowest test in this module, ~73s single-core. It replaced a
    tolerance-based test that ran the SAME sweep, so it is runtime-neutral against the previous
    revision — the cost is the price of the claim 'the advertised command reproduces the constant',
    which cannot be checked at a reduced budget because a reduced budget is a different measurement.
    """
    res = cal.measure_shipped_k_at_dependence_mismatch()
    assert res == cal.MEASURED_SHIPPED_K_AT_DEPENDENCE_MISMATCH


# --------------------- the dependence REFUSAL (fix cycle 3, MAJOR 1(b)) ----------------------

def test_dependence_length_reads_out_the_true_dependence_of_the_certified_model():
    """tau must be in the SAME units as `block`: under `simulate_overlapping(n, dep)` the
    population ACF is triangular and the integrated autocorrelation time is exactly `dep`."""
    rng = np.random.default_rng(11)
    for dep in (5, 20, 60):
        vals = [hest.dependence_length(cal.simulate_overlapping(4000, dep, rng)) for _ in range(9)]
        med = float(np.median([v for v in vals if np.isfinite(v)]))
        assert 0.6 * dep <= med <= 1.6 * dep, (dep, med, vals)
    # an i.i.d. series reads ~1 — the guard must not refuse everything
    iid = [hest.dependence_length(cal.simulate_iid(4000, rng)) for _ in range(9)]
    assert max(iid) <= 3.0, iid


@pytest.mark.parametrize("x", [
    np.zeros(0),                                                   # empty
    np.random.default_rng(9).normal(size=hest.MIN_DECISIONS_FOR_DEPENDENCE - 1),
                                                                    # shorter than the floor
                                                                    # (non-degenerate on purpose)
    np.ones(500),                                                  # degenerate: zero variance
    np.cumsum(np.random.default_rng(0).normal(size=20000)),        # random walk: the ACF never
                                                                    # returns to the band inside
                                                                    # the probed lag window
])
def test_dependence_length_is_nan_when_it_cannot_measure(x):
    """FAIL-CLOSED at the source: unmeasurable dependence is NaN, never a number — and every
    consumer treats NaN as refuse."""
    assert np.isnan(hest.dependence_length(x))


def test_dependence_refusal_operating_characteristic_is_not_vacuous_and_bites_where_it_must():
    """The guard must (a) NOT refuse i.i.d. series — otherwise it is a blanket refusal dressed up
    as a measurement — and (b) refuse essentially always once the real dependence is >= 2x the
    block, where the shipped k demonstrably under-covers. Both directions re-derived live."""
    res = cal.measure_dependence_refusal_operating_characteristic(
        points=((1000, 50),), dep_multiples=(0, 1, 2, 8), n_trials=120, seed=3)
    by = res[1000]["by_dep_multiple"]
    assert by[0]["refusal_rate"] <= 0.02, by[0]
    assert by[2]["refusal_rate"] >= 0.85, by[2]
    assert by[8]["refusal_rate"] >= 0.85, by[8]
    assert by[1]["refusal_rate"] < by[2]["refusal_rate"]
    # the pinned record agrees in shape with the live one
    for nd, e in cal.MEASURED_DEPENDENCE_REFUSAL_OC.items():
        assert e["by_dep_multiple"][0]["refusal_rate"] <= 0.02, nd
        assert e["by_dep_multiple"][2]["refusal_rate"] >= 0.90, nd
        assert e["by_dep_multiple"][8]["refusal_rate"] >= 0.90, nd


def test_dependence_refusal_oc_record_reproduces_exactly_at_the_recorded_3dp():
    """D57 fix pass item 3. `MEASURED_DEPENDENCE_REFUSAL_OC` is the output of the advertised
    `python calibration.py --sweep-dependence-refusal` (no overrides), a bit-deterministic path —
    so the record must reproduce, and the ONLY slack allowed is the 3-decimal DISPLAY rounding the
    constant now discloses. 12 of the 25 cells carry a 4th decimal live (400 trials makes the raw
    rates multiples of 0.0025), so `round(live, 3) == recorded` is the exact claim, asserted here
    at full record coverage rather than left assumed. Cheap: the whole sweep is ~1.4s.

    Anything looser would be a tolerance, and there is no run-to-run noise on this path for a
    tolerance to absorb."""
    live = cal.measure_dependence_refusal_operating_characteristic()
    pinned = cal.MEASURED_DEPENDENCE_REFUSAL_OC
    assert set(live) == set(pinned)
    n_rounded = 0
    for n_dec, e in live.items():
        p = pinned[n_dec]
        assert (e["block"], e["n_trials"]) == (p["block"], p["n_trials"]), n_dec
        assert set(e["by_dep_multiple"]) == set(p["by_dep_multiple"]), n_dec
        for mult, d in e["by_dep_multiple"].items():
            q = p["by_dep_multiple"][mult]
            assert d["dep"] == q["dep"] and d["median_tau"] == q["median_tau"], (n_dec, mult)
            for f in ("refusal_rate", "unmeasurable_rate"):
                assert round(d[f], 3) == q[f], (n_dec, mult, f, d[f], q[f])
                n_rounded += int(d[f] != q[f])
    # the disclosure is not hypothetical: cells really do get rounded, and it says how many
    assert n_rounded == 12, n_rounded
    import inspect
    src = inspect.getsource(cal)
    assert "ROUNDED TO 3 DECIMALS" in src
    assert "0.3675 live and is recorded as 0.367" in src


def test_dependence_refusal_mechanism_text_states_the_three_fail_closed_paths():
    t = cal.DEPENDENCE_REFUSAL_MECHANISM
    assert "measured_dependence_len <= block_used" in t
    assert "DEFAULTS TO None" in t
    assert "UNKNOWN dependence" in t
    assert "CONSERVATIVE" in t


# ------------------------- the demotion narrative (fix cycle 3, MINOR 2) ---------------------

def test_demotion_narrative_says_jitter_not_hole_and_keeps_the_demotion():
    """MINOR 2. The demotion of stratum 620 (on one 0.9267 draw) STANDS — it fails closed — but
    the artifact must not tell a reader block 31 has a real hole, because 6 seeds say otherwise.
    Both halves are pinned: the demotion is still in force, and the narrative says why."""
    assert cal.MEASURED_CALIBRATION_TABLE[620]["achieved"] is False
    assert cal.calibrated_k_for(639, cal.MEASURED_CALIBRATION_TABLE) == (None, False, None)
    t = cal.DEMOTION_IS_MC_JITTER_NOT_A_DEMONSTRATED_HOLE
    assert "THE DEMOTIONS ARE KEPT ANYWAY" in t
    assert "not a coverage defect" in t
    assert "weaker than it reads" in t.lower() or "WEAKER THAN IT READS" in t
    assert "CONSERVATIVE" in t
    # the numbers in the narrative are the numbers in the record
    j = cal.MEASURED_TABLE_SEED_JITTER
    assert f"{j['demoted'][639]['mean']:.4f}" in t
    assert j["demoted"][639]["mean"] > cal.TARGET_COVERAGE
    assert j["demoted"][639]["recorded_in_table"] == pytest.approx(
        cal.MEASURED_BORROWED_PAIRING_COVERAGE[639]["coverage_overlapping"], abs=1e-3)
    # the one the re-measurement does NOT rescue is NOT dressed up as jitter, and the two that
    # clear the bar only barely are called marginal rather than certified (D57 item 2: these are
    # the RE-PINNED figures; the old 0.9194/0.9300 pair came from an unrecoverable run)
    assert j["demoted"][520]["mean"] < cal.TARGET_COVERAGE
    assert f"520 ({j['demoted'][520]['mean']:.4f})" in t
    for nd in (660, 740):
        assert cal.TARGET_COVERAGE < j["demoted"][nd]["mean"] < cal.TARGET_COVERAGE + 0.01, nd
        assert 0 < j["demoted"][nd]["sigma_vs_target"] < 2.0, nd
    assert "660 (0.9367) and 740 (0.9344) come back barely above it" in t
    assert "read as MARGINAL, not as certified-by-re-measurement" in t
    # the retracted clause may appear ONLY inside its own retraction, and the record must not
    # contain the value it claimed (which is what makes the retraction correct)
    assert "landed on 0.9267 exactly" not in t          # the unqualified claim, gone
    assert "RETRACTED, and not re-hunted" in t
    assert t.index("RETRACTED, and not re-hunted") < t.index("0.9267 *exactly*")
    assert all(round(c, 4) != 0.9267 for c in j["demoted"][639]["coverage_by_seed"])
    # ...and the replacement figure it now leans on is real: the lowest seed IS below the draw
    # that caused the demotion
    assert min(j["demoted"][639]["coverage_by_seed"]) < j["demoted"][639]["recorded_in_table"]


def test_seed_jitter_record_covers_both_directions_and_is_self_consistent():
    """The mirror half of MINOR 2: the re-measurement must include ACHIEVED strata too, so the
    table's weakness is not reported only where it flatters the demotions."""
    j = cal.MEASURED_TABLE_SEED_JITTER
    assert set(j["demoted"]) == {520, 600, 639, 660, 740}
    assert len(j["achieved_sample"]) >= 12
    below = []
    for nd, e in j["achieved_sample"].items():
        assert cal.MEASURED_CALIBRATION_TABLE[nd]["achieved"] is True, nd
        assert e["mean"] >= cal.TARGET_COVERAGE, (nd, e)     # the reported finding
        below += [(nd, c) for c in e["coverage_by_seed"] if c < cal.TARGET_COVERAGE]
    # INDIVIDUAL seeds at achieved strata DO fall under the bar (2 of 42 at the re-pinned seeds),
    # which is the sharper form of the finding — so the narrative must not claim otherwise, and
    # this test must not assert the stronger per-seed claim an earlier revision did.
    assert 0 < len(below) <= 4, below
    assert min(c for _, c in below) < cal.TARGET_COVERAGE - 0.01     # 0.9167 at n_dec=880
    t = cal.DEMOTION_IS_MC_JITTER_NOT_A_DEMONSTRATED_HOLE
    assert f"{len(below)} of those" in t and "individual seeds fell BELOW it" in t
    for nd, c in below:
        assert f"{c:.4f} at n_dec={nd}" in t, (nd, c)
    for group in ("demoted", "achieved_sample"):
        for nd, e in j[group].items():
            assert e["mean"] == pytest.approx(float(np.mean(e["coverage_by_seed"])), abs=5e-4)
            # pooled n and SE are the ones the record's own arithmetic implies, not decoration
            assert e["pooled_n_trials"] == 300 * len(e["coverage_by_seed"])
            assert e["sigma_vs_target"] == pytest.approx(
                (e["mean"] - cal.TARGET_COVERAGE) / e["pooled_se"], rel=1e-9)
    # 639 is the borrowed pairing, not a stratum
    assert 639 not in cal.MEASURED_CALIBRATION_TABLE


def test_seed_jitter_re_measurement_reproduces_live_at_a_reduced_budget():
    """The pinned jitter record must not go stale relative to the code path: re-run the demoted
    pairing cheaply and confirm it lands in the same neighbourhood (not bit-identical)."""
    res = cal.measure_table_seed_jitter(((639, 620),), n_seeds=2, n_trials=80, n_boot=200,
                                        seed0=4321)
    assert 0.85 < res[639]["mean"] < 1.0
    assert res[639]["pooled_n_trials"] == 160


def test_seed_jitter_points_are_named_constants_in_the_order_the_seeds_depend_on():
    """The seeds key off each point's INDEX in the points sequence, so the point sets are pinned
    as module constants (not spelled out at the call site) and their ORDER is part of the record.
    Re-ordering or inserting a point silently re-rolls every draw after it."""
    dem, ach = (cal.DEFAULT_SEED_JITTER_DEMOTED_POINTS, cal.DEFAULT_SEED_JITTER_ACHIEVED_POINTS)
    assert [nd for nd, _ in dem] == sorted(nd for nd, _ in dem)
    assert [nd for nd, _ in ach] == sorted(nd for nd, _ in ach)
    assert set(nd for nd, _ in dem) == set(cal.MEASURED_TABLE_SEED_JITTER["demoted"])
    assert set(nd for nd, _ in ach) == set(cal.MEASURED_TABLE_SEED_JITTER["achieved_sample"])
    # each point reads the k of the stratum it is supposed to: itself, except the borrowed pairing
    for nd, src in dem + ach:
        assert (src == nd) or (nd, src) == (639, 620), (nd, src)
        assert src in cal.MEASURED_CALIBRATION_TABLE
    # the seeding rule, spelled out once, is what the record was measured at
    jobs = cal.seed_jitter_jobs(dem, n_seeds=6)
    assert len(jobs) == 30
    assert [j[6] for j in jobs[:7]] == [0, 1, 2, 3, 4, 5, 1000]      # seed0 + 1000*i + s
    assert {j[3] for j in jobs if j[0] == 639} == {cal.MEASURED_CALIBRATION_TABLE[620]["k"]}


def test_seed_jitter_record_reproduces_exactly_at_the_named_pairing():
    """D57 fix pass item 2. `MEASURED_TABLE_SEED_JITTER` is the exact output of
    `calibration.py --sweep-seed-jitter`, a bit-deterministic path at fixed seeds, so the pinned
    row must reproduce with NO tolerance. The record it replaced advertised no command and
    recorded no seeds, and its load-bearing row could not be re-derived at all — this test is what
    stops that from recurring.

    SUBSET, DISCLOSED: only n_dec=639 — the borrowed pairing that demoted stratum 620, i.e. the
    one row every citing sentence in `DEMOTION_IS_MC_JITTER_NOT_A_DEMONSTRATED_HOLE` and in
    stratum 620's own `reason` rests on. The full record is 72 measurement jobs / 24.8 CPU-minutes
    (2.0 min wall at --n-jobs 14, verified to reproduce every row); this is 6 of them (~15s wall in
    a pool of 6). `only=` keeps the seed assignment identical to
    the full run, so this subset is bit-identical to the full run's 639 row, not a cheaper
    approximation of it."""
    res = cal.measure_table_seed_jitter(cal.DEFAULT_SEED_JITTER_DEMOTED_POINTS, n_seeds=6,
                                        only=(639,), n_jobs=6)
    assert set(res) == {639}
    pinned = cal.MEASURED_TABLE_SEED_JITTER["demoted"][639]
    for f in cal.SEED_JITTER_MEASURED_FIELDS:
        assert res[639][f] == pinned[f], (f, res[639][f], pinned[f])
    # the annotations are annotations, and are NOT claimed to come out of the measurement
    assert set(pinned) - set(res[639]) == {"recorded_in_table", "note"}


def test_superseded_seed_jitter_record_is_kept_and_provenance_stamped():
    """D39 doctrine, applied to this re-pin too: the numbers the reproducible run replaced were a
    real measurement whose seeds were never written down. They stay VISIBLE, stamped as
    unrecoverable, rather than being overwritten out of existence — and they must genuinely be the
    OLD values, not a copy of the new ones."""
    old = cal.MEASURED_TABLE_SEED_JITTER_SUPERSEDED
    for needle in ("NO seeds were recorded", "NOT recoverable", "SUPERSEDED",
                   "--sweep-seed-jitter", "RETRACTED", "2026-08-22"):
        assert needle in old["provenance"], needle
    for group in ("demoted", "achieved_sample"):
        assert set(old[group]) == set(cal.MEASURED_TABLE_SEED_JITTER[group])
        for nd, e in old[group].items():
            assert e["coverage_by_seed"] != cal.MEASURED_TABLE_SEED_JITTER[group][nd][
                "coverage_by_seed"], (group, nd)
    # the superseded values are NOT what any shipped prose quotes any more
    t = cal.DEMOTION_IS_MC_JITTER_NOT_A_DEMONSTRATED_HOLE
    assert f"{old['demoted'][639]['mean']:.4f}" not in t          # 0.9467, the old 639 mean
    assert f"{old['achieved_sample'][880]['mean']:.4f}" not in t  # 0.9456, the old weakest mean
    # ...and the conclusion the two records share is stated as shared, not silently carried over
    assert "the conclusion is unchanged" in t
    assert "THE TWO RECORDS AGREE ON THE CONCLUSION" in old["provenance"]


# =============================================================================================
# TASK_897 D57 (2026-08-22) — the tau guard is out-of-family, narrowed. Every test below is the
# 881G "gate (i) doc-only-author" shape applied a third time in this row's history: a claim in a
# shipped string the code does not back, rendered exactly where a reader checks legitimacy.
# =============================================================================================

def test_no_shipped_string_claims_tau_unconditionally_measures_dependence():
    """The pre-D57 generalisation — 'MEASURES each cell's realized per-decision dependence' as an
    unqualified fact, and 'tau <= block' read as a general truth rather than an in-family one —
    must be gone from every module that ships the claim. `honest_estimator.dependence_length`'s
    own in-family algebra ('tau <= block` there reads') is fine; the FALSE generalisation is not.
    """
    import inspect
    for mod in (cal, hest):
        src = inspect.getsource(mod)
        assert "MEASURES each cell's realized per-decision dependence and" not in src, mod.__name__
        assert '`tau <= block` is thus precisely "this series' not in src, mod.__name__
    # the narrowed replacement phrasing is present, in both modules that make the claim
    assert "ATTEMPTS to measure each cell's realized per-decision dependence" in cal.CERTIFICATION_IS_CONDITIONAL_ON_DEPENDENCE
    assert "ATTEMPT TO MEASURE" in hest.dependence_length.__doc__.upper()


def test_no_shipped_string_asserts_refusal_operating_characteristic_without_in_family_qualifier():
    """The OPERATING CHARACTERISTIC sentence in `DEPENDENCE_REFUSAL_MECHANISM` ('at dep >= 2x
    block ... it refuses essentially always') is an IN-FAMILY-ONLY measurement
    (`simulate_overlapping`) and must carry that qualifier at the point of the claim, immediately
    followed by the OUT-OF-FAMILY paragraph that says it does not generalise."""
    t = cal.DEPENDENCE_REFUSAL_MECHANISM
    assert "MEASURED IN-FAMILY" in t
    assert "IN THAT FAMILY" in t
    assert "OUT OF FAMILY, THIS OPERATING CHARACTERISTIC DOES NOT HOLD" in t
    assert (t.index("MEASURED IN-FAMILY")
            < t.index("OUT OF FAMILY, THIS OPERATING CHARACTERISTIC DOES NOT HOLD"))


def test_dependence_length_docstring_narrows_the_tau_claim_and_names_TASK_901():
    """Site 1 of TASK_897 D57's scope. The in-family algebra (tau == dep exactly under
    `simulate_overlapping`) must survive; the generalisation that followed it must not, and the
    AR1 out-of-family numbers + real-data inertness + TASK_901 handoff must all be present."""
    d = hest.dependence_length.__doc__
    assert "tau = 1 + 2 * sum_{h=1..dep-1}" in d          # the correct in-family algebra survives
    assert "INSIDE THE CALIBRATED FAMILY, AND ONLY THERE" in d
    assert "LINEAR-ACF" in d
    assert "CONFLATES" in d.upper()
    assert "0.805" in d and "0.376" in d and "8.6" in d
    assert "EXPECTED NOT TO BITE" in d
    assert "0.639" in d and "97 disjoint" in d
    assert "TASK_901" in d
    assert '`tau <= block` is thus precisely "this series' not in d


def test_certification_conditional_and_refusal_mechanism_name_the_out_of_family_finding():
    """Sites 4 and 5 of TASK_897 D57's scope. Both load-bearing strings must state (1) the guard
    is a linear-ACF statistic valid only in-family, (2) the AR1 mismatch numbers, (3) the
    real-data inertness numbers with provenance, (4) the guard is EXPECTED NOT TO BITE here, and
    (5) TASK_901 by name."""
    for t in (cal.CERTIFICATION_IS_CONDITIONAL_ON_DEPENDENCE, cal.DEPENDENCE_REFUSAL_MECHANISM):
        assert "0.805" in t and "0.376" in t
        assert "tau=1-2" in t or "tau =1-2" in t
        assert "EXPECTED NOT TO BITE" in t
        assert "0.639" in t
        assert "TASK_901" in t
        assert "ci_advisory" in t
    # the numbers quoted in prose must match the pinned, provenance-stamped records exactly
    ar1 = cal.MEASURED_TAU_OUT_OF_FAMILY_HALTED_CYCLE
    assert f"{ar1['by_phi'][0.99]['coverage']}" in cal.CERTIFICATION_IS_CONDITIONAL_ON_DEPENDENCE
    assert f"{ar1['by_phi'][0.999]['coverage']}" in cal.CERTIFICATION_IS_CONDITIONAL_ON_DEPENDENCE
    real = cal.MEASURED_REAL_DATA_TAU_INERT_HALTED_CYCLE
    cc = real["empirical_containment_check"]
    assert f"{cc['contains_grand_sharpe_fraction']}" in cal.DEPENDENCE_REFUSAL_MECHANISM
    assert f"{cc['n_chunks']}" in cal.DEPENDENCE_REFUSAL_MECHANISM
    assert f"{cc['chunk_size_decisions']}" in cal.DEPENDENCE_REFUSAL_MECHANISM


def test_ci_advisory_never_presented_as_band_readable_anywhere_shipped():
    """No shipped string may say the band's 'CI lo > 0' criterion 'may be read off' an
    advisory/calibrated interval, or call `ci_advisory`/`ci_calibrated`=True sufficient for the
    third conjunct — this is the exact retracted claim
    (`sweep_gated_maker.py`'s old "THIS is the interval the band's 'CI lo > 0' criterion may be
    read off"). Generic English uses of "read off" elsewhere (e.g. widening guidance) are not the
    target and must not false-positive this test."""
    import inspect
    import sweep_gated_maker as sw
    for mod in (cal, hest, sw):
        src = inspect.getsource(mod)
        # the retracted AFFIRMATIVE claim ("THIS is the interval ... may be read off") must be
        # gone; a NEGATED use of the same words ("NEVER ... may be read off") is the correct,
        # narrowed claim and must NOT trip this check
        assert "THIS is the interval the band's" not in src, mod.__name__
        assert "estimate and carries `ci_calibrated=True`" not in src, mod.__name__
    # the narrowed (negated) replacement is proven at the RENDERED-artifact level in
    # test_sweep_guards.py::test_artifact_never_presents_an_advisory_interval_as_band_readable
    # (source-level grep here would see the raw, still-adjacent string literals, not the
    # runtime-concatenated sentence)
    assert "UNAVAILABLE-because-uncalibratable" in hest.CI_UNCALIBRATED_DISCLOSURE
    assert "REGARDLESS of whether" in hest.CI_UNCALIBRATED_DISCLOSURE.upper() \
        or "REGARDLESS OF WHETHER" in hest.CI_UNCALIBRATED_DISCLOSURE.upper()


def test_measured_tau_out_of_family_halted_cycle_is_provenance_stamped_and_matches_prose():
    """The AR1 out-of-family numbers are a HISTORICAL measurement (halted cycle) at a budget not
    reproduced bit-identically in this loop — pinned with provenance, per the task's instruction,
    and cross-checked against the fast probe's DIRECTION only."""
    rec = cal.MEASURED_TAU_OUT_OF_FAMILY_HALTED_CYCLE
    assert "halted cycle" in rec["provenance"] and "2026-08-22" in rec["provenance"]
    assert rec["n_dec"] == 3000 and rec["block"] == 60 and rec["weight"] == 0.2
    assert rec["by_phi"][0.99]["coverage"] == 0.805
    assert rec["by_phi"][0.999]["coverage"] == 0.376
    assert rec["by_phi"][0.999]["coverage"] < rec["by_phi"][0.99]["coverage"]
    assert rec["by_phi"][0.99]["certify_rate"] == 1.0


def test_measured_real_data_tau_inert_is_provenance_stamped_real_data_not_rerun():
    """The real-data numbers are provenance-stamped from the halted cycle's real-DB measurement
    (workstation, 2026-08-22) and must NOT be re-derivable by a function this test could call —
    there must be no live real-data reproduction path in this module (real-data validation is
    scheduled, never polled in-loop, per the project invariants)."""
    rec = cal.MEASURED_REAL_DATA_TAU_INERT_HALTED_CYCLE
    assert "trading_data.db" in rec["provenance"] and "NOT re-run in this loop" in rec["provenance"]
    assert rec["measured_dependence_len_range"] == (1, 2)
    assert rec["block"] == 60
    assert rec["guard_certification_rate"] == 1.0
    cc = rec["empirical_containment_check"]
    assert cc["n_chunks"] == 97 and cc["chunk_size_decisions"] == 1000
    assert cc["contains_grand_sharpe_fraction"] == 0.639
    assert not hasattr(cal, "measure_real_data_tau_inert")   # no in-loop real-data reproduction


def test_simulate_persistent_component_carries_the_claimed_variance_fraction():
    """`weight**2` must actually be the variance fraction of the persistent part — otherwise the
    '4% of variance' claim quoted verbatim in three shipped strings would be unverified prose."""
    rng = np.random.default_rng(3)
    for phi in (0.9, 0.99, 0.999):
        x = cal.simulate_persistent_component(200000, phi, weight=0.2, rng=rng)
        assert x.std() == pytest.approx(1.0, abs=0.02)      # unit total variance, by construction
    # tau reads near-1 (the guard would certify) at the SHIPPED weight=0.2, phi=0.99 — this is
    # the exact reading `MEASURED_TAU_OUT_OF_FAMILY_HALTED_CYCLE` records
    rng2 = np.random.default_rng(4)
    tau = hest.dependence_length(cal.simulate_persistent_component(3000, 0.99, 0.2, rng2))
    assert tau <= 3.0


def test_measure_tau_out_of_family_mismatch_reproduces_the_direction_fast():
    """FAST (seconds) reproduction of D57's finding, asserting DIRECTION only per the task's
    instruction (the halted cycle's exact 0.805/0.376 are not reproduced at this budget): tau
    reads near-iid (the guard would certify) while achieved coverage is materially below target
    and gets WORSE as phi -> 1 (more persistence, same 4% variance share)."""
    res = cal.measure_tau_out_of_family_mismatch(n_dec=3000, phis=(0.99, 0.999), n_trials=50,
                                                  n_boot=120, seed=123)
    by = res["by_phi"]
    assert by[0.99]["median_tau"] <= 3.0 and by[0.999]["median_tau"] <= 3.0
    assert by[0.99]["certify_rate"] >= 0.9 and by[0.999]["certify_rate"] >= 0.9
    assert by[0.99]["coverage"] is not None and by[0.999]["coverage"] is not None
    assert by[0.99]["coverage"] < cal.TARGET_COVERAGE
    assert by[0.999]["coverage"] < by[0.99]["coverage"]           # WORSE, not better, as phi -> 1
    # uses the shipped k, not an ad-hoc one
    assert res["k"] == cal.MEASURED_CALIBRATION_TABLE[3000]["k"]


# =============================================================================================
# D57 FIX CYCLE 4 — the surrounding instrument. Every test below pins a figure or a refusal that
# shipped WRONG or MISSING, i.e. a claim in a shipped string that the code did not back.
# =============================================================================================

# ------------------------------------------------------------------- A2: --n-jobs fork bomb

def test_pool_workers_refuses_more_workers_than_this_machine_has_cpus():
    """D57 item A2. `--n-jobs` was unvalidated upward and it is a MEASURED fork bomb:
    `--n-jobs 100000` spawned 7,549 processes and was still climbing at 65s, on a host with
    ~54 GiB and a documented hard-crash history under memory pressure. The realistic input is a
    typo (`1400` for `14`), which is why the refusal NAMES the cap instead of silently clamping.

    PERTURBATION: return `int(n_jobs)` unconditionally and the 100000 case forks 100000 workers."""
    cpu = os.cpu_count() or 1
    for n in (cpu + 1, 1400, 100000):
        with pytest.raises(SystemExit) as e:
            cal.pool_workers(n, 64)
        msg = str(e.value)
        assert f"exceeds this machine's CPU count ({cpu})" in msg
        assert f"--n-jobs {cpu}" in msg          # names the cap, actionably


def test_pool_workers_clamps_to_the_job_count_and_keeps_serial_serial():
    """At or below the cap, the width is clamped DOWN to the number of jobs (never 14 workers for
    3 probe points), and <= 1 still means "run serially", which is the pre-existing safe path."""
    cpu = os.cpu_count() or 1
    assert cal.pool_workers(cpu, 3) == min(cpu, 3)
    assert cal.pool_workers(4, 2) == 2
    assert cal.pool_workers(2, 1000) == 2
    for n in (1, 0, -7):
        assert cal.pool_workers(n, 1000) == 1       # serial, as before
    assert cal.pool_workers(cpu, 0) == 1            # no jobs -> no pool


def test_every_process_pool_in_this_module_goes_through_pool_workers():
    """The clamp is only a defence if NOTHING constructs a pool around it. All four
    `ProcessPoolExecutor` sites must take their width from `pool_workers`, checked in the source
    so a fifth pool added later cannot quietly bypass it."""
    src = Path(cal.__file__).read_text()
    n_pools = src.count("ProcessPoolExecutor(max_workers=")
    assert n_pools == 4, f"expected 4 pool sites, found {n_pools} — new pool needs the clamp"
    assert src.count("ProcessPoolExecutor(max_workers=workers)") == 4
    assert "max_workers=int(n_jobs)" not in src


# --------------------------------------------------- C1: SWEEP 1's false below-target count

def test_block_cap_sweep1_prose_matches_the_pinned_record():
    """D57 item C1. `BLOCK_CAP_SELECTION_RATIONALE` said SWEEP 1 had "only cap=80/n_dec=500 read
    below target, at 0.907". Its own advertised command (`--sweep-caps`, re-run 2026-08-22,
    1m49s) reports TWO below-target cells: (80,500)=0.9067 and (100,8000)=0.9267. This was the
    one measured block in the module with NO test at all.

    Every figure the prose quotes is DERIVED from `MEASURED_BLOCK_CAP_SWEEP1` here, so a
    re-measurement that moves a cell turns the prose red instead of leaving it silently wrong.

    PERTURBATION: put "only cap=80/n_dec=500" back and the count assertions below fail."""
    rec = cal.MEASURED_BLOCK_CAP_SWEEP1
    text = cal.BLOCK_CAP_SELECTION_RATIONALE
    covs = [(cap, nd, e["validation_coverage"]) for cap in rec for nd, e in rec[cap].items()]
    assert len(covs) == 15
    below = sorted((cap, nd, c) for cap, nd, c in covs if c < cal.TARGET_COVERAGE)
    # the measurement itself
    assert [(cap, nd) for cap, nd, _ in below] == [(80, 500), (100, 8000)]
    assert f"{min(c for _, _, c in covs):.4f}" == "0.9067"
    assert f"{max(c for _, _, c in covs):.4f}" == "0.9933"
    # ...and the prose agrees with it, cell by cell
    assert "TWO of the 15 read BELOW target" in text
    assert "cap=80/n_dec=500 at 0.9067" in text
    assert "cap=100/n_dec=8000 at 0.9267" in text
    assert "0.907-0.993 across all 15 cells" in text     # the range WAS correct; keep it
    # the retracted wording must be gone
    assert "only cap=80/n_dec=500" not in text
    # both below-target cells sit ABOVE the selected cap, which the prose also claims
    assert all(cap > cal.NEW_MAX_BLOCK for cap, _, _ in below)
    assert "BOTH sit at a cap ABOVE the 60 selected below" in text
    # every cell's `block` is what block_for would pick at that cap (the record is self-consistent)
    for cap in rec:
        for nd, e in rec[cap].items():
            assert e["block"] == cal.block_for(nd, max_block=cap)


def test_block_cap_sweep1_reproduces_exactly_at_the_first_cap():
    """The record is the EXACT stdout of `calibration.py --sweep-caps`. The RNG is one shared
    stream across the nested loop, so a cell is only reproducible together with everything before
    it — the cheapest exactly-reproducible unit is therefore the FIRST cap's three cells, which is
    what this pins (~19s measured; all 15 cost ~109s, which no test may pay). No tolerance:
    the path is bit-deterministic at seed=42."""
    caps = cal.BLOCK_CAP_SWEEP1_REPRODUCIBLE_PREFIX_CAPS
    live = cal.measure_block_cap_candidates(cap_candidates=caps)
    for cap in caps:
        for nd, pinned in cal.MEASURED_BLOCK_CAP_SWEEP1[cap].items():
            got = live[cap][nd]
            assert got["block"] == pinned["block"]
            assert got["k"] == pinned["k"], (cap, nd)
            assert got["validation_coverage"] == pinned["validation_coverage"], (cap, nd)


# ------------------------------------ C2: the borrowed-pairing record's unqualified "verbatim"

def test_borrowed_pairing_639_is_a_pre_demotion_row_the_command_now_refuses():
    """D57 item C2. `MEASURED_BORROWED_PAIRING_COVERAGE` claimed "reproducible verbatim" via
    `--verify-borrowed`, but row 639 CANNOT reproduce: stratum 620 is now `achieved=False`
    (demoted BY this very row), so `calibrated_k_for(639)` returns `(None, False, None)`, 639
    routes into `skipped`, and the command prints it identically to 619 — the deliberate negative
    control. Re-run 2026-08-22: 16 probe jobs instead of 18, and 17 of the 18 rows reproduced
    every field exactly.

    The row is KEPT (it is the evidence for the demotion) and the "verbatim" claim is SCOPED, the
    same overlay disclosure `MEASURED_CALIBRATION_TABLE` already carries for its 620 row.

    PERTURBATION: restore an unqualified "reproducible verbatim" and the source assertions fail."""
    table = cal.MEASURED_CALIBRATION_TABLE
    assert table[620]["achieved"] is False
    assert cal.calibrated_k_for(639, table) == (None, False, None)
    assert cal.calibrated_k_for(619, table) == (None, False, None)   # the negative control
    # the two points route identically through the real function's own job/skip split
    jobs, skipped = [], {}
    for i, (nd, n_trials, n_boot) in enumerate(cal.DEFAULT_BORROWED_PROBE_POINTS):
        k, achieved, _src = cal.calibrated_k_for(nd, table)
        (skipped.setdefault(nd, True) if not achieved else jobs.append(nd))
    assert set(skipped) == {619, 639}
    assert len(jobs) == 16 and len(cal.DEFAULT_BORROWED_PROBE_POINTS) == 18
    # the record still carries the pre-demotion measurement, and says what it is
    row = cal.MEASURED_BORROWED_PAIRING_COVERAGE[639]
    assert row["coverage_overlapping"] == 0.9266666666666666
    assert row["borrowed_k"] == table[620]["k"]
    assert "PRE-DEMOTION measurement" in row["reason"]
    assert "no longer reproduces" in row["reason"]
    # ...and the module comment scopes the verbatim claim rather than asserting it whole
    src = Path(cal.__file__).read_text()
    assert "reproducible verbatim EXCEPT FOR ONE ROW" in src
    assert "17 of the 18 rows reproduced EVERY field exactly" in src


# ------------------------------------------------- C3: the retraction that contradicted itself

def test_the_0_9267_retraction_is_scoped_to_the_639_pairing():
    """D57 item C3. The shipped (and RENDERED) retraction said "No seed of the named run
    reproduces that value" of 0.9267 — while the named run produces 0.9266666666666666 THREE
    times, and the same paragraph cites one of them four sentences later. Only the 639 pairing is
    free of it, which is what the test suite actually asserts. This module's own words: a
    retraction that is itself wrong, shipping verbatim into the artifact, is worse than the error
    it retracts.

    PERTURBATION: restore "No seed of the named run reproduces that value" and this fails."""
    j = cal.MEASURED_TABLE_SEED_JITTER
    target = 0.9266666666666666
    hits = [(group, nd, i + 1)
            for group in ("demoted", "achieved_sample")
            for nd, e in j[group].items()
            for i, c in enumerate(e["coverage_by_seed"]) if c == target]
    # the value the old retraction said no seed produces — three of them do
    assert sorted(hits) == [("achieved_sample", 640, 3), ("demoted", 520, 1), ("demoted", 660, 5)]
    # ...but NOT the 639 pairing, which is the true scope
    assert target not in j["demoted"][639]["coverage_by_seed"]
    assert min(j["demoted"][639]["coverage_by_seed"]) == 0.92

    text = cal.DEMOTION_IS_MC_JITTER_NOT_A_DEMONSTRATED_HOLE
    assert "**No seed of the 639 PAIRING reproduces that value**" in text
    assert "No seed of the named run reproduces that value" not in text
    # the retraction names where 0.9267 DOES occur, so it cannot read as a contradiction again
    assert "demoted stratum 520 at seed 0" in text
    assert "demoted stratum 660 at seed 3004" in text
    assert "achieved stratum 640 at seed 5002" in text
    # ...and those seeds are the real ones the harness assigns
    seeds = {(nd, s) for nd, _n, _b, _k, _src, _t, s, _mb
             in cal.seed_jitter_jobs(cal.DEFAULT_SEED_JITTER_DEMOTED_POINTS, n_seeds=6)}
    assert (520, 0) in seeds and (660, 3004) in seeds
    seeds_a = {(nd, s) for nd, _n, _b, _k, _src, _t, s, _mb
               in cal.seed_jitter_jobs(cal.DEFAULT_SEED_JITTER_ACHIEVED_POINTS, n_seeds=3)}
    assert (640, 5002) in seeds_a
    # the superseded record's provenance carried the same over-broad clause; it is scoped too
    prov = cal.MEASURED_TABLE_SEED_JITTER_SUPERSEDED["provenance"]
    assert "NO SEED OF THE 639 PAIRING produces it" in prov
    assert "no seed of the named run produces it" not in prov


# ---------------------------------------------------------------- C4: the remaining false figures

def test_shipped_k_repin_movement_is_counted_not_characterised():
    """D57 item C4. The comment said the re-measurement "makes the collapse marginally WORSE at
    every multiple past 1x, never better". Measured against the record it compares: of the six
    cells past 1x, four are worse, n_dec=1000/2x is UNCHANGED (the same two values in the opposite
    order) and n_dec=1000/4x is BETTER (0.750 vs 0.702).

    PERTURBATION: restore "WORSE at every multiple past 1x, never better" and this fails."""
    new = cal.MEASURED_SHIPPED_K_AT_DEPENDENCE_MISMATCH
    old = cal.MEASURED_SHIPPED_K_AT_DEPENDENCE_MISMATCH_HALTED_CYCLE["record"]
    worse = same = better = 0
    for nd in new:
        for m in new[nd]["by_dep_multiple"]:
            if m == 1:
                continue
            n = float(np.mean(new[nd]["by_dep_multiple"][m]["coverage_by_seed"]))
            o = float(np.mean(old[nd]["by_dep_multiple"][m]["coverage_by_seed"]))
            worse += n < o
            same += n == o
            better += n > o
    assert (worse, same, better) == (4, 1, 1)
    # the two named cells
    assert (sorted(new[1000]["by_dep_multiple"][2]["coverage_by_seed"])
            == sorted(old[1000]["by_dep_multiple"][2]["coverage_by_seed"]))
    assert np.mean(new[1000]["by_dep_multiple"][4]["coverage_by_seed"]) == 0.75
    assert np.mean(old[1000]["by_dep_multiple"][4]["coverage_by_seed"]) == pytest.approx(0.702)

    src = Path(cal.__file__).read_text()
    assert "WORSE at four" in src
    assert "marginally WORSE at every multiple past 1x, never better" not in src
    # the 1x claim must cover BOTH rows of the reproducible pin, not just the old record's ~0.95
    means_1x = sorted(float(np.mean(new[nd]["by_dep_multiple"][1]["coverage_by_seed"]))
                      for nd in new)
    assert [round(m, 3) for m in means_1x] == [0.924, 0.96]
    assert "0.92-0.96 at dep = 1x block" in src
    assert "~0.95 at dep = 1x block to ~0.52" not in src


def test_achieved_k_interval_claim_contains_every_achieved_k():
    """D57 item C4. The certified-table header said "Achieved k stays in [1.23, 1.69]" while the
    max achieved k is 1.6904650613309085 — OUTSIDE the stated interval.

    PERTURBATION: restore "[1.23, 1.69]" and the containment assertion fails."""
    ks = {nd: e["k"] for nd, e in cal.MEASURED_CALIBRATION_TABLE.items() if e["achieved"]}
    lo, hi = min(ks.values()), max(ks.values())
    assert (lo, hi) == (1.231702348964501, 1.6904650613309085)
    src = Path(cal.__file__).read_text()
    assert "Achieved k stays in [1.2317, 1.6905]" in src
    assert "Achieved k stays in [1.23, 1.69]" not in src
    # the stated interval must actually CONTAIN every achieved k
    assert all(1.2317 <= k <= 1.6905 for k in ks.values())
    # and the strata the prose names are the real argmin/argmax
    assert min(ks, key=ks.get) == 100000 and max(ks, key=ks.get) == 980
    assert "min\n# 1.231702348964501 at n_dec=100000, max 1.6904650613309085 at n_dec=980" in src
    assert hi < 8.0                                     # still a calibration, not a widening


def test_achieved_coverage_band_claim_matches_the_shipped_table():
    """D57 item C4. The stratification comment contrasted a borrowed pairing against "the
    0.943-0.977 margin the pinned strata advertised". The achieved strata actually span
    0.9333-0.9833, with FOUR achieved strata below 0.943 — the contrast is real but weaker.

    PERTURBATION: restore "the 0.943-0.977 margin" and this fails."""
    ach = {nd: e["validation_coverage_overlapping"]
           for nd, e in cal.MEASURED_CALIBRATION_TABLE.items() if e["achieved"]}
    assert len(ach) == 62
    assert (round(min(ach.values()), 4), round(max(ach.values()), 4)) == (0.9333, 0.9833)
    below = sorted(nd for nd, c in ach.items() if c < 0.943)
    assert below == [1020, 3000, 30000, 100000]
    src = Path(cal.__file__).read_text()
    assert "spans\n# 0.9333-0.9833, and FOUR achieved strata sit BELOW 0.943" in src
    assert "0.943-0.977 margin the pinned strata advertised" not in src


def test_the_1_3413_k_is_not_attributed_to_a_shipped_stratum():
    """D57 item C4. The stratification comment attributed k=1.3413 to the n_dec=1000 / block=50
    stratum, which carries k=1.5309881369858398. NO shipped stratum has a k rounding to 1.3413 —
    it came from the pre-restratification grid. The nine coverage values beside it DO reproduce at
    k=1.3413, so the ATTRIBUTION was the defect, not the measurement.

    PERTURBATION: restore "the n_dec=1000 / block=50 stratum's k (=1.3413)" and this fails."""
    assert cal.MEASURED_CALIBRATION_TABLE[1000]["block"] == 50
    assert cal.MEASURED_CALIBRATION_TABLE[1000]["k"] == 1.5309881369858398
    assert [nd for nd, e in cal.MEASURED_CALIBRATION_TABLE.items()
            if round(e["k"], 4) == 1.3413] == []
    assert [nd for nd, e in cal.MEASURED_CALIBRATION_TABLE.items()
            if 1.33 <= e["k"] <= 1.35] == []
    src = Path(cal.__file__).read_text()
    assert "stratum's k (=1.3413)" not in src
    assert "came from the PRE-RESTRATIFICATION grid" in src
    assert "k=1.5309881369858398" in src


def test_borrowed_across_block_boundary_nine_cell_record_reproduces():
    """D57 item C4. The nine borrowed-across-the-block-boundary coverage figures rendered into
    the artifact with NO seeds recorded. The recipe is now written down — a FRESH
    `np.random.default_rng(seed)` per cell, k=1.3413, "overlapping", 200 trials x 400 resamples,
    n_dec 1200/1350/1500 x seeds 0/1/2 — which is exactly what makes a SINGLE cell reproducible on
    its own. One named cell is re-run here (~4s); all nine cost ~36s, verified 2026-08-22.

    NO TOLERANCE: at a fixed seed this path is bit-deterministic."""
    pinned = {(0, 1200): 0.925, (0, 1350): 0.94, (0, 1500): 0.95,
              (1, 1200): 0.955, (1, 1350): 0.94, (1, 1500): 0.935,
              (2, 1200): 0.91, (2, 1350): 0.925, (2, 1500): 0.92}
    src = Path(cal.__file__).read_text()
    # the record as rendered in the comment, and the recipe that regenerates it
    assert "A FRESH rng PER CELL" in src
    assert "seed 0: 0.9250 / 0.9400 / 0.9500" in src
    assert "SEEDS NEVER RECORDED for that triple" in src     # the un-seeded review triple, scoped
    # 4 of the 9 cells are UNDER the bar — the claim the comment makes about them
    assert sum(v < cal.TARGET_COVERAGE for v in pinned.values()) == 4
    assert round(float(np.mean(list(pinned.values()))), 3) == 0.933
    # ...and one named cell re-measured exactly
    seed, n_dec = 2, 1500
    cov, n = cal.measured_coverage_at_k(n_dec, 1.3413, "overlapping", 200, 400,
                                       np.random.default_rng(seed))
    assert (cov, n) == (pinned[(seed, n_dec)], 200)


def test_dependence_refusal_range_at_dep_equals_block_is_stated_correctly():
    """D57 item C4. `sweep_gated_maker.apply_ci_calibration`'s docstring said "~30-42%
    conservative refusal at dep == block"; the measured range in the module's own record starts at
    0.270 (n_dec=100). `calibration.py` already stated it correctly as "0.27-0.42".

    PERTURBATION: restore "~30-42%" in the sweep docstring and this fails."""
    import sweep_gated_maker as sw
    rates = sorted(e["by_dep_multiple"][1]["refusal_rate"]
                   for e in cal.MEASURED_DEPENDENCE_REFUSAL_OC.values())
    assert (min(rates), max(rates)) == (0.270, 0.422)
    doc = sw.apply_ci_calibration.__doc__
    assert "0.27-0.42 conservative refusal at dep == block" in doc
    # the false figure may survive only inside its own retraction, never as the live claim
    assert "~30-42% conservative refusal" not in doc
    assert doc.count("30-42%") == 1 and 'corrected from "~30-42%"' in doc
    # calibration.py already stated the same range correctly, in the comment above its own record
    src_cal = Path(cal.__file__).read_text()
    assert "0.27-0.42 false refusals at dep == block" in src_cal


def test_ci_coverage_probe_comment_matches_what_the_command_prints():
    """D57 item C4. `ci_coverage_probe.py` said the command prints 0.878 for
    ("overlapping",500); it prints 0.877, because the cell is 351/400 = 0.8775 and the nearest
    double lies just below, so `f"{...:.3f}"` renders 0.877. The comment describes what a reader
    SEES, so it must match the formatting, not the unrounded value."""
    import ci_coverage_probe as probe
    src = Path(probe.__file__).read_text()
    assert "0.930 / 0.877 / 0.930 / 0.910" in src
    assert "0.930 / 0.878 / 0.930 / 0.910" not in src
    assert f"{0.8775:.3f}" == "0.877"       # the formatting fact the correction rests on
    assert probe.MEASURED_COVERAGE[("overlapping", 500)] == 0.865   # the historical record stands


# --------------------------------------------------- D: the new subset machinery's refusals

def test_reproduce_strata_refuses_a_re_keying_n_dec_strata():
    """D57 item D1. `reproduce_strata` refused a `seed=` kwarg but not `n_dec_strata=`, which
    re-keys the seed just as effectively (`stratum_seed` is `seed + 1000 * index-in-the-list`):
    `stratum_seed(1000)` is 48000 by default but 1000 under `n_dec_strata=(40, 1000)`, and the
    latter measured validation_coverage_overlapping=1.0 against the shipped 0.9333 — a DIFFERENT
    measurement wearing the exact-reproduction name.

    PERTURBATION: drop the check and `reproduce_strata([1000], n_dec_strata=(40, 1000))` runs and
    returns a coverage that does not match the shipped table."""
    assert cal.stratum_seed(1000) == 48000
    assert cal.stratum_seed(1000, n_dec_strata=(40, 1000)) == 1000
    with pytest.raises(TypeError) as e:
        cal.reproduce_strata([1000], n_dec_strata=(40, 1000))
    assert "silently re-keys every seed" in str(e.value)
    # the sibling refusal still stands
    with pytest.raises(TypeError):
        cal.reproduce_strata([1000], seed=5)
    # ...and the DEFAULT spelling is still accepted (no honest call starts refusing)
    import inspect
    assert (inspect.signature(cal.reproduce_strata).parameters["n_dec_strata"].default
            is cal.DEFAULT_N_DEC_STRATA)


def test_reproduce_strata_cli_refuses_a_non_stratum_cleanly(capsys):
    """D57 item D2. `--reproduce-strata 61` raised `ValueError: 61 is not in list` from
    `stratum_seed`'s `.index()` — a traceback, not a refusal — and `main()`'s
    `"<not a shipped stratum>"` branch was therefore unreachable dead code.

    PERTURBATION: remove the validation and this raises ValueError instead of SystemExit."""
    import sweep_gated_maker  # noqa: F401  (keeps sys.path shape identical to a real CLI run)
    argv = sys.argv
    try:
        sys.argv = ["calibration.py", "--reproduce-strata", "61"]
        with pytest.raises(SystemExit) as e:
            cal.main()
        msg = str(e.value)
        assert "not shipped strata" in msg and "[61]" in msg
        assert "40" in msg          # lists what IS available
    finally:
        sys.argv = argv
    # the dead branch is gone
    src = Path(cal.__file__).read_text()
    # the dead branch is gone — the string survives only inside the comment that retires it
    assert 'diff_vs_shipped=' in src and '"<not a shipped stratum>"]' not in src
    # and every shipped stratum really is a key of the table the CLI then indexes
    assert set(cal.DEFAULT_N_DEC_STRATA) == set(cal.MEASURED_CALIBRATION_TABLE)


def test_measure_table_seed_jitter_refuses_an_absent_only_selector():
    """D57 item D3. `only=` with a selector not present in `points` returned `{}` SILENTLY, so
    the exact-equality test would pass vacuously over zero rows if its `assert set(res) == {639}`
    line were ever removed.

    PERTURBATION: drop the check and `only=[61]` returns {} — a test that cannot fail."""
    with pytest.raises(ValueError) as e:
        cal.measure_table_seed_jitter(cal.DEFAULT_SEED_JITTER_DEMOTED_POINTS, only=[61])
    assert "is not in `points`" in str(e.value)
    assert "[520, 600, 639, 660, 740]" in str(e.value)
    # a PARTIALLY valid selector refuses too (639 is present, 61 is not)
    with pytest.raises(ValueError):
        cal.measure_table_seed_jitter(cal.DEFAULT_SEED_JITTER_DEMOTED_POINTS, only=[639, 61])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
