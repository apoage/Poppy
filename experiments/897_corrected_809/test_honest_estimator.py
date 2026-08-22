#!/usr/bin/env python3
"""
TASK_897 gate (i): the honest per-decision estimator passes a hand-computed worked example,
and a mutation re-introducing per-bar annualization is caught by a test.

Run: python -m pytest experiments/897_corrected_809/test_honest_estimator.py -v
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import honest_estimator as hest  # noqa: E402


# =============================================================================
# WORKED EXAMPLE — every number below is computed BY HAND (plain arithmetic, no numpy,
# no call into honest_estimator) so this test is a genuine independent cross-check, not a
# tautology against the implementation it is testing.
#
# Setup: 12 one-minute bars (bar_sec=60), two decisions (rebalances) at bar 0 and bar 6,
# i.e. two 6-bar holds. Per-bar NET returns are constant within each hold:
#   hold 1 (bars 0-5):  net = +0.001 / bar  -> hold-1 total return = 6*0.001 = 0.006
#   hold 2 (bars 6-11): net = +0.002 / bar  -> hold-2 total return = 6*0.002 = 0.012
#
# Hand-derived per-decision series: reb_ret = [0.006, 0.012]
#   mean = (0.006+0.012)/2                         = 0.009
#   deviations: -0.003, +0.003 -> squares 9e-6 each -> mean sq = 9e-6 -> sd = 0.003
#   n_bars=12, bar_sec=60 -> window_years = 720 / 31_536_000 = 1/43800
#   n_decisions=2 -> decisions_per_year = 2 * 43800 = 87600
#   sharpe_decision = (0.009/0.003) * sqrt(87600) = 3 * sqrt(87600) = 887.9189152169246...
#
# Contrast — the RETRACTED per-bar-annualized ("809 trap") number computed on the same
# 12 raw bar returns [0.001]*6+[0.002]*6, annualized by bars/year = 525600 (bar_sec=60):
#   mean_bar = 0.0015, sd_bar = 0.0005 (population, ddof=0)
#   bars_per_year = 31_536_000/60 = 525600
#   sharpe_bar_TRAP = (0.0015/0.0005) * sqrt(525600) = 3 * sqrt(525600) = 2174.948275247023...
#   ratio sharpe_bar_TRAP / sharpe_decision = sqrt(6) = 2.449489742783178
#     (exactly sqrt(mean_hold_bars) — the "809 trap" inflation this module exists to remove)
# =============================================================================

BAR_SEC = 60
NET_BAR = np.array([0.001] * 6 + [0.002] * 6, dtype=np.float64)
TURN_BAR = np.array([0.10, 0, 0, 0, 0, 0, 0.10, 0, 0, 0, 0, 0], dtype=np.float64)
GROSS_BAR = NET_BAR  # cost_per_side=0 -> gross == net, isolates the annualization logic

EXPECTED_REB_RET = [0.006, 0.012]
EXPECTED_N_DECISIONS = 2
EXPECTED_DECISIONS_PER_YEAR = 87600.0
EXPECTED_SHARPE_DECISION = 3.0 * math.sqrt(87600.0)          # 887.9189152169246
EXPECTED_SHARPE_BAR_TRAP = 3.0 * math.sqrt(525600.0)         # 2174.948275247023
EXPECTED_TRAP_RATIO = math.sqrt(6.0)                          # 2.449489742783178


def test_worked_example_segmentation():
    reb = hest.segment_by_decision(NET_BAR, TURN_BAR)
    assert reb.tolist() == pytest.approx(EXPECTED_REB_RET, abs=1e-12)
    assert len(reb) == EXPECTED_N_DECISIONS


def test_worked_example_decisions_per_year():
    ppy = hest.decisions_per_year(EXPECTED_N_DECISIONS, n_bars=12, bar_sec=BAR_SEC)
    assert ppy == pytest.approx(EXPECTED_DECISIONS_PER_YEAR, rel=1e-9)


def test_worked_example_honest_sharpe_matches_hand_computation():
    """The load-bearing hand-check (gate i, first half): compute_honest_metrics's
    `sharpe_decision` must match the independently hand-derived value above."""
    m = hest.compute_honest_metrics(GROSS_BAR, TURN_BAR, cost_per_side=0.0, bar_sec=BAR_SEC,
                                     seed=0, n_boot=200)
    assert m.n_bars == 12
    assert m.n_decisions == EXPECTED_N_DECISIONS
    assert m.decisions_per_year == pytest.approx(EXPECTED_DECISIONS_PER_YEAR, rel=1e-9)
    assert m.mean_hold_bars == pytest.approx(6.0)
    assert m.sharpe_decision == pytest.approx(EXPECTED_SHARPE_DECISION, rel=1e-9)
    # the diagnostic per-bar number must equal the hand-derived TRAP value...
    assert m.sharpe_per_bar_DIAGNOSTIC_DO_NOT_GATE_ON == pytest.approx(EXPECTED_SHARPE_BAR_TRAP, rel=1e-9)
    # ...and must NOT be what compute_honest_metrics reports as sharpe_decision (the two
    # differ by exactly sqrt(mean_hold_bars) in this constructed example — the 809 trap's
    # inflation factor, made visible on purpose).
    assert m.sharpe_decision != pytest.approx(m.sharpe_per_bar_DIAGNOSTIC_DO_NOT_GATE_ON, rel=1e-6)
    assert (m.sharpe_per_bar_DIAGNOSTIC_DO_NOT_GATE_ON / m.sharpe_decision) == pytest.approx(
        EXPECTED_TRAP_RATIO, rel=1e-9
    )


def test_leading_flat_segment_is_not_counted_as_a_decision():
    """A zero-turnover, no-position stretch before the first trade is NOT a realized decision
    and must NOT be counted.

    This REPLACES an earlier test that asserted the opposite. The earlier reasoning ("dropping
    the leading segment biases decisions_per_year upward") had the direction backwards:
    n_decisions is the NUMERATOR of decisions_per_year, so KEEPING the phantom segment inflates
    decisions_per_year and hence sqrt(decisions_per_year) — the same inflation direction as the
    retracted 809 trap. The spec's wording is "annualized by REALIZED decisions-per-year"; a flat
    pre-trade stretch is not a realized decision.

    Hand-computed: bars [0,1] flat (no book, turn=0), a single decision at bar 2, hold runs
    [2,5) with net 0.001/bar -> the ONE segment's return is 3*0.001 = 0.003."""
    net = np.array([0.0, 0.0, 0.001, 0.001, 0.001], dtype=np.float64)
    turn = np.array([0.0, 0.0, 0.10, 0.0, 0.0], dtype=np.float64)
    reb = hest.segment_by_decision(net, turn)
    assert reb.tolist() == pytest.approx([0.003], abs=1e-12)
    assert len(reb) == 1               # ONE real decision, not two


def test_leading_flat_segment_cannot_fabricate_a_sharpe_from_one_bet():
    """The concrete exploit the counting bug enabled (constructed by the TASK_897 fix-pass
    review): 2000 bars, exactly ONE real decision at bar 500, constant net 5e-4 thereafter.

    Counting the leading flat stretch made n_decisions=2 and injected a synthetic 0.0 sample
    into the per-decision series, which DEFEATED `sharpe()`'s own `x.size < 2 -> nan` degeneracy
    guard: it produced sharpe_decision=+22.93 with a zero-width CI [+22.93, +22.93], which
    trivially satisfies the pre-registered band's "CI lo > 0" off a SINGLE bet.

    Correct behaviour: one real decision -> n_decisions == 1 -> NaN Sharpe and NaN CI."""
    n_bars = 2000
    gross = np.zeros(n_bars, dtype=np.float64)
    gross[500:1200] = 0.0005
    turn = np.zeros(n_bars, dtype=np.float64)
    turn[500] = 0.2
    m = hest.compute_honest_metrics(gross, turn, cost_per_side=0.0, bar_sec=60, seed=0, n_boot=200)
    assert m.n_decisions == 1
    assert math.isnan(m.sharpe_decision)
    assert math.isnan(m.ci_lo_UNCALIBRATED) and math.isnan(m.ci_hi_UNCALIBRATED)


def test_fewer_than_two_real_decisions_is_nan():
    """The general rule behind the case above: below MIN_DECISIONS_FOR_SHARPE REAL decisions
    there is no sample standard deviation, so there is no Sharpe and no CI — NaN, never a
    fabricated number a band could be read off. Checked at 0 and at 1 decision."""
    assert hest.MIN_DECISIONS_FOR_SHARPE == 2
    for turn_bar_idx in ([], [7]):
        turn = np.zeros(40, dtype=np.float64)
        for i in turn_bar_idx:
            turn[i] = 0.3
        gross = np.full(40, 0.001, dtype=np.float64)
        m = hest.compute_honest_metrics(gross, turn, cost_per_side=0.0, bar_sec=60, seed=0,
                                         n_boot=200)
        assert m.n_decisions == len(turn_bar_idx)
        assert math.isnan(m.sharpe_decision)
        assert math.isnan(m.ci_lo_UNCALIBRATED) and math.isnan(m.ci_hi_UNCALIBRATED)


def test_degenerate_zero_turnover_does_not_crash():
    """An all-flat book (no decisions ever) must return a finite (nan-Sharpe, no-crash) result,
    not raise — this is the shape the fixture-scale sweep run actually hits (see
    sweep_gated_maker.py's smoke run: too few present assets to open any position)."""
    net = np.zeros(50, dtype=np.float64)
    turn = np.zeros(50, dtype=np.float64)
    m = hest.compute_honest_metrics(net, turn, cost_per_side=0.0008, bar_sec=60, seed=0, n_boot=50)
    assert m.n_decisions == 0          # ZERO decisions — a flat window made no decision at all
    assert math.isnan(m.sharpe_decision)   # no decisions -> nan, not an exception
    assert math.isnan(m.mean_hold_bars)
    assert m.decisions_per_year == pytest.approx(0.0)


def test_decisions_per_year_uses_full_window_span_not_traded_span():
    """The annualization denominator is the FULL window's calendar span, including an untraded
    lead-in — using only the traded span would raise decisions/year and hence the annualized
    Sharpe (the trap direction). 100 bars at bar_sec=60, first trade at bar 50, 2 decisions:
    window_years = 6000/31_536_000 -> decisions_per_year = 2 * 31_536_000/6000 = 10512.0
    (a traded-span denominator would have given 2 * 31_536_000/3000 = 21024.0)."""
    gross = np.full(100, 0.001, dtype=np.float64)
    turn = np.zeros(100, dtype=np.float64)
    turn[50] = turn[75] = 0.1
    m = hest.compute_honest_metrics(gross, turn, cost_per_side=0.0, bar_sec=60, seed=0, n_boot=200)
    assert m.n_decisions == 2
    assert m.decisions_per_year == pytest.approx(10512.0, rel=1e-9)
    # mean hold = covered span / n_dec = (100-50)/2 = 25 bars, NOT n_bars/n_dec = 50.
    assert m.mean_hold_bars == pytest.approx(25.0)


def test_positive_sharpe_can_coexist_with_capital_destruction():
    """The volatility-drag cell (constructed by the TASK_897 fix-pass review) that motivates the
    band's THIRD criterion, PnL > 0: per-decision returns alternating +20% / -19% have a POSITIVE
    arithmetic mean (+0.5%) and therefore a strongly positive Sharpe, while the GEOMETRIC
    compounding destroys capital (1.20 * 0.81 = 0.972 per pair).

    So a Sharpe-and-CI-only gate can pass a cell that lost everything. This test pins the fact;
    `runs/809_decision_rule_rescue/recompute_best.py` is what acts on it."""
    n = 400
    gross = np.empty(n, dtype=np.float64)
    gross[0::2] = 0.20
    gross[1::2] = -0.19
    turn = np.full(n, 0.1, dtype=np.float64)      # every bar is a decision -> one bar per hold
    m = hest.compute_honest_metrics(gross, turn, cost_per_side=0.0, bar_sec=60, seed=0, n_boot=200)
    assert m.n_decisions == n
    assert m.sharpe_decision > 1.0                 # would pass criterion 1
    assert m.ci_lo_UNCALIBRATED > 0.0              # would pass criterion 3
    assert m.pnl < 0.0                             # ...having lost capital. Criterion 2 catches it.
    assert m.maxdd < 0.0


# =============================================================================
# GATE (i), second half: prove a mutation re-introducing per-bar annualization is CAUGHT.
#
# This test does NOT patch honest_estimator — it defines the exact retracted computation
# INLINE (the "809 trap": annualize by bars/year instead of decisions/year) and asserts the
# honest module's output must NOT match it. This is the machine-checkable form of "a mutation
# re-introducing per-bar annualization is caught by a test": if `decisions_per_year()` were
# mutated to return `n_bars / window_years` (bars-per-year) instead of `n_decisions /
# window_years`, `compute_honest_metrics(...).sharpe_decision` would collapse onto
# `_trap_sharpe_per_bar_annualization(...)` and this test would go RED.
#
# The actual RED/GREEN demonstration (mutate honest_estimator.py's decisions_per_year to
# `return n_bars / window_years`, run this file, observe the failure, then revert) is recorded
# verbatim in the TASK_897 implementation report — this test is what caught it.
# =============================================================================

def _trap_sharpe_per_bar_annualization(gross_bar, turn_bar, cost_per_side, bar_sec):
    """The RETRACTED 809 methodology, reproduced independently here (not imported from
    honest_estimator) purely as the forbidden comparison point."""
    net = np.asarray(gross_bar, dtype=np.float64) - np.asarray(turn_bar, dtype=np.float64) * cost_per_side
    bars_per_year = hest.SECONDS_PER_YEAR / bar_sec
    return hest.sharpe(net, bars_per_year)


def test_gate_i_mutation_is_caught():
    m = hest.compute_honest_metrics(GROSS_BAR, TURN_BAR, cost_per_side=0.0, bar_sec=BAR_SEC,
                                     seed=0, n_boot=200)
    trap = _trap_sharpe_per_bar_annualization(GROSS_BAR, TURN_BAR, 0.0, BAR_SEC)
    # The honest, gate-facing number must differ materially from the per-bar trap number.
    assert abs(m.sharpe_decision - trap) > 100.0
    # And it must equal the independently hand-derived honest value, not the trap value.
    assert m.sharpe_decision == pytest.approx(EXPECTED_SHARPE_DECISION, rel=1e-9)
    assert m.sharpe_decision != pytest.approx(trap, rel=1e-3)


# =============================================================================
# The CI is UNCALIBRATED and must never be silently readable as the band's "CI lo > 0".
# These tests pin the *un-silencing* mechanism itself, so a future refactor cannot quietly
# rename the qualifier away and restore a bare-looking interval.
# =============================================================================

def test_ci_fields_are_named_uncalibrated():
    """The interval reaches every caller under a name that carries its own qualifier — the same
    idiom as `sharpe_per_bar_DIAGNOSTIC_DO_NOT_GATE_ON`. A plain `ci_lo`/`ci_hi` must not exist."""
    fields = set(hest.HonestMetrics.__dataclass_fields__)
    assert "ci_lo_UNCALIBRATED" in fields and "ci_hi_UNCALIBRATED" in fields
    assert "ci_lo" not in fields and "ci_hi" not in fields


def test_ci_uncalibrated_disclosure_states_the_consequence():
    """The disclosure must say the thing that matters: the band's criterion is UNAVAILABLE off
    this interval (TASK_897 D57 narrowed 'cannot be read off THIS interval' to 'unavailable on
    this leg, off either interval'), and it must cite the D49 precedent it rests on."""
    d = hest.CI_UNCALIBRATED_DISCLOSURE
    assert "UNCALIBRATED" in d
    assert "D49" in d
    assert "CI lo > 0" in d and "UNAVAILABLE-because-uncalibratable" in d


def test_ci_uncalibrated_disclosure_states_the_certifications_own_condition():
    """Fix cycle 3, MAJOR 1(a); narrowed by TASK_897 D57 (2026-08-22). This string ships VERBATIM
    into every artifact, at exactly the spot a reader checks the calibration's legitimacy — so it
    is where the certification's CONDITION, and the fact that the guard checking it is measured
    inert on this row's real data, both have to be stated."""
    d = hest.CI_UNCALIBRATED_DISCLOSURE
    assert "CONDITIONAL ON MEASURED DEPENDENCE" in d
    assert "dep == block" in d
    assert "NOT a worst case" in d
    assert "0.52-0.55" in d and "0.530" in d
    assert "REFUSES to certify" in d
    assert "cannot be measured at all" in d
    # the condition names the LIVE cap, by construction, so it cannot go stale
    assert f"capped at {hest.MAX_BLOCK_DEFAULT}" in d
    # D57: the tau guard is narrowed to a linear-ACF, in-family-only statistic, and the real-data
    # inertness + TASK_901 handoff must be named right here, not only in calibration.py
    assert "LINEAR-ACF" in d
    assert "0.805" in d and "0.376" in d
    assert "EXPECTED NOT TO BITE" in d
    assert "0.639" in d
    assert "TASK_901" in d
    assert "UNAVAILABLE-because-uncalibratable" in d


# ------------------------- the realized-dependence measurement (fix cycle 3, MAJOR 1(b))

def test_honest_metrics_reports_the_block_it_used_and_the_dependence_it_measured():
    """The two fields the refusal is built on must come off the ESTIMATOR, not be re-derived by
    the consumer: `block_used` is the block the CI was actually bootstrapped at, and
    `measured_dependence_len` is this cell's own realized per-decision dependence."""
    fields = set(hest.HonestMetrics.__dataclass_fields__)
    assert {"block_used", "measured_dependence_len"} <= fields
    rng = np.random.default_rng(0)
    n_dec, hold = 400, 3
    gross = np.repeat(rng.normal(0, 1e-4, n_dec), hold)
    turn = np.zeros(gross.size)
    turn[::hold] = 0.1
    m = hest.compute_honest_metrics(gross, turn, 0.0, 60, seed=0, n_boot=100)
    assert m.n_decisions == n_dec
    assert m.block_used == max(2, min(hest.MAX_BLOCK_DEFAULT, n_dec // 20)) == 20
    assert np.isfinite(m.measured_dependence_len) and m.measured_dependence_len <= 3


def test_block_used_is_minus_one_when_no_ci_was_computed():
    """Below `MIN_DECISIONS_FOR_SHARPE` no CI exists, so there is no block to compare against —
    the record must say so rather than report a plausible-looking block."""
    gross = np.zeros(100)
    turn = np.zeros(100)
    turn[10] = 0.1                       # exactly ONE decision
    m = hest.compute_honest_metrics(gross, turn, 0.0, 60, seed=0, n_boot=50)
    assert m.n_decisions == 1
    assert m.block_used == -1
    assert np.isnan(m.measured_dependence_len)


def test_dependence_length_refuses_rather_than_guesses():
    """FAIL-CLOSED at the source. NaN — never a number — for a series too short to estimate an
    ACF from, a degenerate one, or one whose ACF never returns to the white-noise band."""
    assert np.isnan(hest.dependence_length(np.zeros(0)))
    # too short to estimate an ACF from — NON-degenerate on purpose, so this pins the length
    # floor itself and not the zero-variance guard below it
    short = np.random.default_rng(3).normal(size=hest.MIN_DECISIONS_FOR_DEPENDENCE - 1)
    assert short.std() > 0
    assert np.isnan(hest.dependence_length(short))
    assert np.isnan(hest.dependence_length(np.ones(500)))                    # zero variance
    walk = np.cumsum(np.random.default_rng(1).normal(size=20000))            # unresolved memory
    assert np.isnan(hest.dependence_length(walk))
    # ...and it DOES measure an ordinary series (the guard is not a blanket refusal)
    iid = np.random.default_rng(2).normal(size=2000)
    assert hest.dependence_length(iid) == pytest.approx(1.0)


def test_ci_uncalibrated_disclosure_quotes_the_LIVE_block_cap_not_a_stale_one():
    """MAJOR (fix pass 2026-08-22): this disclosure ships VERBATIM into every artifact, a few
    lines below a calibrated table showing `block=60, achieved=True` rows — and it still asserted,
    present tense, that the estimator "CAPS the block length at 20, so dependence longer than 20
    decisions cannot be captured at any n", and that calibrating was still "a planner-scheduled
    prerequisite". Both were FALSE at ship time. The artifact contradicted itself at exactly the
    spot a reader checks the calibration's legitimacy.

    The previous version of this test only pinned a qualitative phrase, which is why the
    staleness sailed through. This one asserts THE CAP VALUE THE CODE ACTUALLY USES — the
    disclosure is built as an f-string over `MAX_BLOCK_DEFAULT`, so the claim and the constant
    cannot drift apart again — and that the retracted claims are gone."""
    d = hest.CI_UNCALIBRATED_DISCLOSURE
    cap = hest.MAX_BLOCK_DEFAULT
    # (1) the live cap is quoted, by construction, in the sentence that describes the mechanism
    assert f"`MAX_BLOCK_DEFAULT` = {cap} decisions" in d
    assert f"min({cap}, n_dec // 20)" in d
    # (2) the stale present-tense claims are gone
    assert "CAPS the block length at 20" not in d
    assert "cannot be captured at any n" not in d
    assert "planner-scheduled prerequisite" not in d
    # (3) the historical measurement is still quoted, but explicitly dated to the OLD cap
    assert "0.530" in d and "THEN-" in d and "PREVAILING BLOCK CAP OF 20" in d
    # (4) the CURRENT mechanism (where an advisory interval does exist) is described — renamed
    # from ci_calibrated/ci_lo to ci_advisory/ci_lo_ADVISORY by TASK_897 D57 (2026-08-22)
    assert "ci_advisory" in d and "`ci_lo_ADVISORY`" in d
    assert "ci_calibrated" not in d and "`ci_lo`" not in d
    # (5) and the disclosure never claims a cap the code does not use
    for wrong in (20, 21, 40, 80, 100):
        if wrong == cap:
            continue
        assert f"`MAX_BLOCK_DEFAULT` = {wrong}" not in d


def test_ci_uncalibrated_disclosure_does_not_contradict_the_calibration_table():
    """Cross-module consistency: the disclosure ships in the same artifact as
    `calibration.MEASURED_CALIBRATION_TABLE`. If that table certifies rows at the raised cap,
    the disclosure beside it must not be describing a lower one."""
    import calibration as cal
    assert cal.NEW_MAX_BLOCK == hest.MAX_BLOCK_DEFAULT
    shipped_blocks = {e["block"] for e in cal.MEASURED_CALIBRATION_TABLE.values()}
    assert max(shipped_blocks) == hest.MAX_BLOCK_DEFAULT
    assert f"{hest.MAX_BLOCK_DEFAULT} decisions" in hest.CI_UNCALIBRATED_DISCLOSURE


def test_measured_coverage_probe_reproduces_below_nominal():
    """The disclosure's quantitative claim is re-measured here at low cost through the REAL CI
    code path: under a true null with overlapping dependence of length 20 at 100 decisions, the
    nominal-95% interval's actual coverage is far below 0.95. (The recorded probe value is 0.530;
    this test uses a coarse budget and asserts only the direction-and-size claim, so it pins the
    finding without pinning Monte-Carlo noise.)"""
    import ci_coverage_probe as probe
    r = probe.measure_coverage(100, scenario="overlapping", n_trials=120, n_boot=200, seed=0)
    assert r["block"] == 5                      # what compute_honest_metrics would itself pick
    assert r["coverage"] < 0.80                 # nominal is 0.95
    assert probe.MEASURED_COVERAGE[("overlapping", 100)] < 0.95
    assert all(v < 0.95 for v in probe.MEASURED_COVERAGE.values())


def test_measured_coverage_probe_caveat_lives_on_the_constant_not_70_lines_away():
    """D57 fix pass item 4. `MEASURED_COVERAGE`'s comment says "recorded verbatim from the
    reproduce command above", and since the D55(a) cap raise (20 -> 60) four of its eight cells no
    longer reproduce that command. The module docstring said so; the comment ON THE CONSTANT did
    not, and that is where a reader checks. This pins the caveat's LOCATION and its CONTENT: which
    cells still reproduce, which do not, and why."""
    import inspect

    import ci_coverage_probe as probe
    src = inspect.getsource(probe)
    head, _, tail = src.partition("MEASURED_COVERAGE = {")
    # the caveat must sit in the ~30 lines immediately above the constant, not only in the docstring
    near = "\n".join(head.splitlines()[-32:])
    assert "WHAT RE-RUNNING THE COMMAND TODAY PRINTS" in near
    assert "STILL REPRODUCE" in near and "NO LONGER REPRODUCE" in near
    assert "0.530" in near and "LOAD-BEARING" in near      # the one that still reproduces
    for n_dec in (500, 1000):
        assert f'("iid",{n_dec})' in near and f'("overlapping",{n_dec})' in near
    # the split is machine-readable and covers the record exactly once
    a, b = probe.MEASURED_COVERAGE_STILL_REPRODUCES, probe.MEASURED_COVERAGE_SUPERSEDED_BY_CAP_RAISE
    assert set(a) | set(b) == set(probe.MEASURED_COVERAGE)
    assert not set(a) & set(b)
    # ...and it is the split the code implies: a cell reproduces iff the cap raise left its block
    for scenario, n_dec in a:
        assert probe.block_for(n_dec) == min(20, n_dec // 20), (scenario, n_dec)
    for scenario, n_dec in b:
        assert probe.block_for(n_dec) > 20, (scenario, n_dec)
    assert tail                                            # the constant really follows the caveat


def test_measured_coverage_probe_load_bearing_cells_still_reproduce():
    """The other half of item 4: the cells the caveat says still reproduce, actually re-measured
    through the live code path at the recorded settings — exact at the 3 decimals the record
    prints (a coverage is hits/400, so 3dp is a display rounding: ("iid",100) reads 0.9025 live).

    SUBSET, DISCLOSED: 2 of the 4 unaffected cells — ("overlapping",100), the load-bearing 0.530
    that D49/D55 and `CI_UNCALIBRATED_DISCLOSURE` quote, and ("iid",100), the one cell whose
    record is a rounding of a 4th decimal, so the disclosed precision claim is pinned and not just
    asserted. Each cell costs ~5.5s (400 trials x 400 resamples); the other two unaffected cells
    (("iid",40) and ("iid",200)) were verified out of band in the same fix pass and reproduce
    identically, but are not paid for on every run. The four SUPERSEDED cells are deliberately
    NOT re-measured here — under the raised cap they are a different estimator, which is the
    caveat's whole point."""
    import ci_coverage_probe as probe
    for cell in (("overlapping", 100), ("iid", 100)):
        scenario, n_dec = cell
        r = probe.measure_coverage(n_dec, scenario=scenario)     # every arg at the recorded default
        assert r["n_usable"] == 400, cell
        assert round(r["coverage"], 3) == probe.MEASURED_COVERAGE[cell], (cell, r["coverage"])
    assert probe.MEASURED_COVERAGE[("overlapping", 100)] == 0.530


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
