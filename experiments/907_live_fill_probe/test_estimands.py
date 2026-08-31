import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from estimands import _percentile, compute_estimand1, compute_estimand2  # noqa: E402
from shadow_engine import MarkoutSample, ShadowPairResult  # noqa: E402


def _result_with_markouts(values_1m):
    r = ShadowPairResult(pair="BTC/USD")
    for i, v in enumerate(values_1m):
        r.markouts.append(MarkoutSample(pair="BTC/USD", side=1, fill_price=100.0,
                                         fill_ts=float(i), horizon="1m", adverse_selection_bp=v))
    return r


def test_estimand1_median_and_deciles():
    vals = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]  # n=11, sorted already
    r = _result_with_markouts(vals)
    e1 = compute_estimand1(r)
    h = e1.by_horizon["1m"]
    assert h.n_samples == 11
    assert h.n_used == 11
    assert h.n_dropped_nonfinite == 0
    assert h.median_bp == pytest.approx(5.0)
    assert h.mean_bp == pytest.approx(5.0)
    assert h.deciles_bp[10] == pytest.approx(1.0)
    assert h.deciles_bp[90] == pytest.approx(9.0)


def test_estimand1_model_tier_comparison():
    # median = 3.0 bp: model's 0bp tier is BELOW it, 5bp and 10bp tiers meet/exceed it
    r = _result_with_markouts([1.0, 3.0, 5.0])
    e1 = compute_estimand1(r)
    h = e1.by_horizon["1m"]
    assert h.median_bp == pytest.approx(3.0)
    assert h.model_tiers_meet_or_exceed_median[0.0] is False
    assert h.model_tiers_meet_or_exceed_median[5.0] is True
    assert h.model_tiers_meet_or_exceed_median[10.0] is True


def test_estimand1_zero_sample_horizons_are_reported_never_omitted():
    """B3 (run-45), superseding the old `test_estimand1_empty_horizon_omitted`: a pre-registered
    horizon that drew zero samples is a RESULT about the instrument, not the absence of a row.
    All four pre-registered horizons (`HORIZONS_S`) must appear, each explicitly zero-sample and
    not-evaluable, even when the pair had NO markouts at all."""
    r = ShadowPairResult(pair="BTC/USD")  # no markouts at all
    e1 = compute_estimand1(r)
    assert set(e1.by_horizon) == {"1m", "5m", "20m", "1h"}
    for h in e1.by_horizon.values():
        assert h.n_samples == 0
        assert h.n_used == 0
        assert h.is_evaluable is False
        assert h.median_bp is None


def test_estimand2_rates():
    r = ShadowPairResult(pair="BTC/USD", n_posted=10, n_touch_fills=4, n_trade_cross_fills=6)
    e2 = compute_estimand2(r)
    assert e2.touch_fill_rate == pytest.approx(0.4)
    assert e2.trade_cross_fill_rate == pytest.approx(0.6)


def test_estimand2_no_postings_is_none_not_zero():
    r = ShadowPairResult(pair="BTC/USD", n_posted=0)
    e2 = compute_estimand2(r)
    assert e2.touch_fill_rate is None
    assert e2.trade_cross_fill_rate is None


# --- BLOCKER 1 (run-43): a non-finite sample must never reach a number or a verdict ---
#
# The defect: one NaN among 11 finite markouts left `n` at 11, produced mean=nan and median=nan,
# left the NaN mid-sequence in `sorted()` so several deciles rendered PLAUSIBLE FINITE NUMBERS
# derived from a corrupted rank ordering, and — because `tier >= nan` is always False — emitted
# a confidently-worded "BELOW measured" verdict for every model tier.

NAN = float("nan")
INF = float("inf")


def test_one_nan_among_n_is_dropped_counted_and_never_corrupts_the_deciles():
    vals = [0.0, 1.0, 2.0, 3.0, 4.0, NAN, 6.0, 7.0, 8.0, 9.0, 10.0]
    e1 = compute_estimand1(_result_with_markouts(vals))
    h = e1.by_horizon["1m"]

    assert h.n_samples == 11
    assert h.n_used == 10               # the NaN did NOT stay in the sample count
    assert h.n_dropped_nonfinite == 1   # and it is DISCLOSED, not silently excluded
    assert e1.total_dropped_nonfinite == 1

    # every reported number is finite and matches the finite-only sample exactly
    finite = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    assert h.mean_bp == pytest.approx(sum(finite) / len(finite))
    assert h.median_bp == pytest.approx(5.0)
    assert all(math.isfinite(v) for v in h.deciles_bp.values())
    # the pre-fix bug rendered p40/p50 as nan while p30/p60 were plausible-but-corrupt
    assert h.deciles_bp[40] == pytest.approx(_percentile(finite, 40))
    assert h.deciles_bp[60] == pytest.approx(_percentile(finite, 60))
    # a real comparison ran, so a real verdict is emitted
    assert h.model_tiers_meet_or_exceed_median[0.0] is False
    assert h.model_tiers_meet_or_exceed_median[10.0] is True


def test_all_nan_horizon_is_reported_as_not_evaluable_never_as_a_verdict():
    e1 = compute_estimand1(_result_with_markouts([NAN, NAN, NAN]))
    h = e1.by_horizon["1m"]  # NOT omitted — the artifact must say these samples existed
    assert h.n_samples == 3
    assert h.n_used == 0
    assert h.n_dropped_nonfinite == 3
    assert h.mean_bp is None
    assert h.median_bp is None
    assert h.deciles_bp == {}
    assert h.is_evaluable is False
    # every tier is NOT-EVALUABLE — never False, which would render as "BELOW measured"
    for tier, ok in h.model_tiers_meet_or_exceed_median.items():
        assert ok is None, f"tier {tier} emitted a verdict from a non-finite median"


def test_infinities_are_treated_as_non_finite_too():
    e1 = compute_estimand1(_result_with_markouts([1.0, INF, -INF, 3.0]))
    h = e1.by_horizon["1m"]
    assert h.n_used == 2 and h.n_dropped_nonfinite == 2
    assert h.median_bp == pytest.approx(2.0)
    assert math.isfinite(h.mean_bp)


def test_percentile_refuses_a_nonfinite_sample_as_a_backstop():
    with pytest.raises(ValueError, match="non-finite"):
        _percentile([1.0, NAN, 3.0], 50)
    with pytest.raises(ValueError, match="empty"):
        _percentile([], 50)


def test_estimand1_zero_samples_and_all_nan_render_the_same_reported_shape():
    """Superseding the old `test_estimand1_still_omits_a_horizon_with_no_samples_at_all` (B3,
    run-45): zero samples and all-non-finite samples are now the SAME reported shape (n_used=0,
    nothing evaluable, no verdict) — they differ only in `n_samples`, which is exactly what a
    reader needs to tell "we measured nothing" apart from "we measured only non-finite junk"."""
    r = ShadowPairResult(pair="BTC/USD")
    h = compute_estimand1(r).by_horizon["1m"]
    assert h.n_samples == 0
    assert h.n_used == 0
    assert h.is_evaluable is False


def test_estimand2_carries_the_drop_counters_for_disclosure():
    r = ShadowPairResult(pair="BTC/USD", n_posted=10, n_touch_fills=4, n_trade_cross_fills=6,
                          n_markouts_dropped_nonfinite=3, n_markouts_dropped_gap=2,
                          n_markouts_dropped_no_mid=1, n_markouts_dropped_outside_span=4)
    e2 = compute_estimand2(r)
    assert e2.n_markouts_dropped_nonfinite == 3
    assert e2.n_markouts_dropped_gap == 2
    assert e2.n_markouts_dropped_no_mid == 1
    assert e2.n_markouts_dropped_outside_span == 4
