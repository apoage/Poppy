#!/usr/bin/env python3
"""Fast, deterministic unit tests for TASK_901's estimators and block rule — NOT the Monte Carlo
coverage certification itself (that is `run_validation.py`, which is expensive by design and
runs on gpu-server; see the task file). These tests are cheap enough to run anywhere."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import range_probe as rp   # noqa: E402
import families as fam     # noqa: E402
import coverage_harness as ch  # noqa: E402
import gate_iv_hook         # noqa: E402


def test_range_acf_band_iid_reads_out_near_one():
    rng = np.random.default_rng(1)
    x = fam.simulate_iid(4000, rng)
    r, info = rp.range_acf_band(x)
    assert np.isfinite(r) and r <= 3, (r, info)


def test_range_acf_band_triangular_reads_out_close_to_dep():
    rng = np.random.default_rng(2)
    x = fam.simulate_triangular(4000, 20, rng)
    r, info = rp.range_acf_band(x)
    assert np.isfinite(r)
    # Bound WIDENED (and made one-sided at the bottom) in fix cycle 2, when the fail-open
    # Bonferroni band was replaced by the standard 1.96/sqrt(n) one: against a dependent series
    # the sample ACF's own sd exceeds 1/sqrt(n) (Bartlett), so out-of-band lags beyond `dep` keep
    # resetting the k_consec run and the readout lands ABOVE the true dep. That is the
    # CONSERVATIVE direction — longer block, wider interval — and is the price of not
    # fail-opening; recorded here rather than tuned away.
    #
    # THIS ASSERTION HOLDS ONLY BECAUSE THE SEED IS PINNED, AND THE CYCLE-2 ANNOTATION UNDERSTATED
    # THAT (fix cycle 3). It said "measured 20-44 over 8 draws at dep=20, median ~28"; over 200
    # draws of this exact cell (`spread_check.py` -> SPREAD_CHECK.json) the readout is median 29,
    # p10 = 20.0, p90 = 63.1, min 19, max 149 — only 84.5% of draws land inside [20, 60]. So this
    # is a PINNED-SEED REGRESSION GUARD on one draw, not a statement that the estimator reads
    # 20-60 on this family; roughly one draw in six would fail it. It is left as-is deliberately:
    # widening the bound to cover the p90 tail would make it assert nothing, and re-seeding until
    # it passes comfortably would be tuning. The distributional claim lives in
    # `measured_findings.ACF_BAND_READOUT_SPREAD`, where it belongs.
    assert 20 <= r <= 60, (r, info)


def test_range_acf_band_refuses_on_short_series():
    r, info = rp.range_acf_band(np.zeros(0))
    assert math.isnan(r)
    r, info = rp.range_acf_band(np.arange(10, dtype=float))
    assert math.isnan(r)
    assert "min_n" in info["reason"]


def test_range_acf_band_refuses_on_degenerate_series():
    r, info = rp.range_acf_band(np.ones(100))
    assert math.isnan(r)


def test_variance_ratio_iid_near_one():
    rng = np.random.default_rng(3)
    x = fam.simulate_iid(4000, rng)
    vr = rp.variance_ratio(x, 16)
    assert 0.7 < vr < 1.3


def test_variance_ratio_triangular_grows_with_q_then_flattens_near_dep():
    rng = np.random.default_rng(4)
    x = fam.simulate_triangular(20000, 20, rng)
    vr_small = rp.variance_ratio(x, 4)
    vr_at_dep = rp.variance_ratio(x, 40)
    vr_large = rp.variance_ratio(x, 400)
    assert vr_small < vr_at_dep
    # once q >> dep, VR should have leveled off near the population tau (~dep)
    assert abs(vr_large - vr_at_dep) / vr_at_dep < 0.5


def _population_vr_persistent(q: int, weight: float, phi: float) -> float:
    """POPULATION variance ratio of `weight*AR1(phi) + sqrt(1-weight^2)*iid` — closed form,
    no sampling: rho(h) = weight^2 * phi^h, VR(q) = 1 + 2*sum_{h<q} (1 - h/q) rho(h)."""
    return 1.0 + 2.0 * sum((1.0 - h / q) * (weight ** 2) * phi ** h for h in range(1, q))


def test_readoff_rule_is_not_the_one_step_rule_on_a_known_population_curve():
    """THE BLOCKER-1 REGRESSION TEST. Deterministic, no RNG: on the POPULATION VR curve of the
    4%-variance persistent component (phi=0.99) the one-step rule that shipped in cycle 1 stops
    at q=4 (VR 1.03960 -> 1.11801 is a 7.5% step, under the 10% tolerance) while the curve goes
    on climbing to 6.06 at q=256 against an asymptote of 8.92. The corrected rule must refuse."""
    qs = [2, 4, 8, 16, 32, 64, 128, 256]
    vr = [_population_vr_persistent(q, 0.2, 0.99) for q in qs]
    assert vr[0] == pytest.approx(1.03960, abs=1e-5)
    assert vr[1] == pytest.approx(1.11801, abs=1e-5)
    one_step_rel = abs(vr[1] - vr[0]) / abs(vr[0])
    assert one_step_rel < 0.10                      # the shipped rule WOULD have stopped here
    assert rp.first_levelled_index(vr, 0.10) is None  # the corrected rule does not


def test_readoff_rule_finds_the_smallest_levelled_point_and_never_the_last():
    assert rp.first_levelled_index([1.0, 1.01, 1.0, 0.99], 0.10) == 0
    assert rp.first_levelled_index([1.0, 2.0, 2.05, 2.02], 0.10) == 1
    # rising throughout -> None, even though the LAST point trivially has nothing above it
    assert rp.first_levelled_index([1.0, 2.0, 4.0, 8.0], 0.10) is None
    assert rp.first_levelled_index([1.0], 0.10) is None


def test_acf_band_is_the_standard_per_lag_band_not_a_widened_one():
    """MAJOR-2 REGRESSION TEST: a family-wise (Bonferroni) widening makes the 'no more
    dependence' verdict EASIER -> shorter blocks -> narrower intervals, the fail-open direction.
    The band must be the plain 1.96/sqrt(n)."""
    assert rp._band_z(0.05) == pytest.approx(1.959964, abs=1e-5)
    rng = np.random.default_rng(101)
    x = fam.simulate_iid(3000, rng)
    _r, info = rp.range_acf_band(x)
    assert info["band"] == pytest.approx(rp._band_z(0.05) / math.sqrt(3000), rel=1e-12)
    assert info["band"] == pytest.approx(0.0357839, abs=1e-6)
    # and it must be TIGHTER than what the retired Bonferroni split produced (z=3.71 over 240
    # lags), which is what let a population rho(1)=0.0396 be called noise.
    assert info["band"] < 0.0678


def test_acf_band_reads_the_low_amplitude_persistent_component_above_one():
    """The mechanism claim being corrected: the persistent component's population rho(1) =
    0.04*0.99 = 0.0396 is ABOVE the 1.96/sqrt(3000)=0.0358 noise floor, so a correctly-banded
    first-crossing rule DOES see it (the interval still under-covers — that is measured in
    `measured_findings`, and is a different failure from 'the signal is invisible')."""
    rng = np.random.default_rng(202)
    reads = []
    for _ in range(8):
        x = fam.simulate_persistent_component(3000, 0.99, rng=rng, weight=0.2)
        r, _i = rp.range_acf_band(x)
        if np.isfinite(r):
            reads.append(r)
    assert reads, "estimator refused every draw"
    assert float(np.median(reads)) > 1.0


def test_spectral_zero_returns_nan_on_bandwidth_refusal():
    """MINOR: the refusal path used to return a FINITE bandwidth with info['refuse']=True, which
    a caller keying off the return value (every caller in this module, `block_rule` included)
    would read as a certification."""
    rng = np.random.default_rng(303)
    x = fam.simulate_ar1(300, 0.99, rng)
    bw, info = rp.range_spectral_zero(x, max_bandwidth_frac=0.01)
    assert math.isnan(bw), (bw, info)
    assert info.get("refuse") is True
    assert np.isfinite(info["bandwidth"])
    assert rp.block_rule(100000, bw).refused


def test_range_variance_ratio_refuses_when_never_plateaus():
    rng = np.random.default_rng(5)
    # dep close to the grid ceiling -> VR should still be climbing at the last grid point.
    x = fam.simulate_triangular(2000, 150, rng)
    r, info = rp.range_variance_ratio(x)
    assert math.isnan(r)
    assert "reason" in info


def test_range_spectral_zero_iid_small():
    rng = np.random.default_rng(6)
    x = fam.simulate_iid(4000, rng)
    bw, info = rp.range_spectral_zero(x)
    assert np.isfinite(bw) and bw < 20


def test_range_spectral_zero_refuses_on_short_series():
    r, info = rp.range_spectral_zero(np.arange(5, dtype=float))
    assert math.isnan(r)


def test_snap_block_never_rounds_down():
    for r in (1.2, 2.0, 2.1, 7.5, 59.9, 60.0):
        b = rp.snap_block(r)
        assert b >= math.ceil(r), (r, b)
        assert b in rp.BLOCK_STRATA


def test_block_rule_refuses_above_calibratable_ceiling_never_extrapolates():
    rule = rp.block_rule(3000, rp.MAX_BLOCK_CAP + 1.0)
    assert rule.refused
    assert rule.k is None
    assert "ceiling" in rule.reason


def test_block_rule_refuses_on_nonfinite_range():
    rule = rp.block_rule(1000, float("nan"))
    assert rule.refused
    assert rule.block is None and rule.k is None


def test_block_rule_refuses_on_insufficient_non_overlapping_blocks():
    # block=60 (snapped) against n_dec=100 leaves only 1 non-overlapping block.
    rule = rp.block_rule(100, 60.0)
    assert rule.refused
    assert "non-overlapping blocks" in rule.reason


def test_block_rule_calibrates_and_caches():
    rp._K_CACHE.clear()
    rule1 = rp.block_rule(2000, 5.0, n_trials_calib=15, n_trials_valid=15, n_boot=40)
    n_after_first = len(rp._K_CACHE)
    rule2 = rp.block_rule(2000, 5.0, n_trials_calib=15, n_trials_valid=15, n_boot=40)
    assert len(rp._K_CACHE) == n_after_first, "second identical call should hit the cache"
    if not rule1.refused:
        assert rule1.k == rule2.k


def test_prewarm_calibration_table_serial_matches_direct_call():
    rp._K_CACHE.clear()
    table = rp.prewarm_calibration_table(blocks=(2, 5), n_decs=(500,), n_jobs=1,
                                         n_trials_calib=10, n_trials_valid=10, n_boot=30, seed=99)
    assert set(table.keys()) == {(2, 500, 99), (5, 500, 99)}
    for v in table.values():
        assert "achieved" in v


def test_run_coverage_cell_basic_shape():
    rp._K_CACHE.clear()
    rp.prewarm_calibration_table(blocks=rp.BLOCK_STRATA, n_decs=(500,), n_jobs=1,
                                 n_trials_calib=15, n_trials_valid=15, n_boot=40)
    row = ch.run_coverage_cell("acf_band", "iid", 500, {}, n_trials=15, seed=11, n_boot_ci=40,
                               calib_kwargs=dict(n_trials_calib=15, n_trials_valid=15, n_boot=40))
    assert row["n_trials"] == 15
    assert row["n_certified"] + row["n_refused"] == 15
    if row["n_certified"] > 0:
        assert 0.0 <= row["coverage"] <= 1.0
        assert row["coverage_se"] is not None


def test_mc_se_matches_binomial_formula():
    assert ch.mc_se(0.5, 100) == pytest.approx(0.05)
    assert ch.mc_se(None, 100) is None
    assert ch.mc_se(0.5, 0) is None


def test_real_data_chunk_containment_shape_on_synthetic_stand_in():
    # Uses a synthetic AR(1) series in place of the real minute data purely to test the
    # CONTAINMENT-CHECK MACHINERY's shape/arithmetic — the real-data leg itself is exercised
    # separately by `run_validation.py` against `real_series.build_real_series` on gpu-server.
    rng = np.random.default_rng(21)
    x = fam.simulate_ar1(5000, 0.3, rng)
    rp._K_CACHE.clear()
    rp.prewarm_calibration_table(blocks=rp.BLOCK_STRATA, n_decs=(1000,), n_jobs=1,
                                 n_trials_calib=15, n_trials_valid=15, n_boot=40)
    cc = ch.real_data_chunk_containment(x, "acf_band", chunk_size=1000, n_boot_ci=40, seed=3,
                                        calib_kwargs=dict(n_trials_calib=15, n_trials_valid=15,
                                                          n_boot=40))
    assert cc["n_chunks"] == 5
    assert cc["n_certified"] + cc["n_refused"] == 5
    assert len(cc["per_chunk"]) == 5


def test_stationarity_diagnostics_shape_and_estimator_independence():
    """Gate (iii)'s re-scoped instrument: it must produce the between/within comparison and the
    VR curves WITHOUT consulting any range estimator (a diagnostic about the SERIES)."""
    rng = np.random.default_rng(41)
    x = fam.simulate_iid(4000, rng)
    d = ch.stationarity_diagnostics(x, chunk_size=1000, n_boot=120, seed=2)
    assert d["n_chunks"] == 4
    assert len(d["chunk_sharpes"]) == 4
    assert set(d["boot_se_by_block"]) == {2, 3, 5, 8, 12, 18, 26, 38, 60}
    assert d["chi2_df"] == 3
    assert 0.0 <= d["chi2_p_value"] <= 1.0
    # i.i.d. => VR ~ 1 at every q, and between-chunk dispersion ~ within-chunk SE
    assert 0.6 < d["chunk_vr_mean"][8] < 1.4
    assert 0.1 < d["between_within_variance_ratio"] < 6.0


def test_gate_iv_hook_refuses_when_artifact_absent():
    with pytest.raises(gate_iv_hook.Gate4NotYetRunnable):
        gate_iv_hook.reread_897_conjunct("/no/such/path/artifact.md", "variance_ratio")


def test_families_are_mean_zero_in_population_by_construction():
    # Sanity: sample mean should be small relative to sample std at large n (not asserting
    # exact zero — these are stochastic draws).
    rng = np.random.default_rng(31)
    for name, kwargs in (("iid", {}), ("triangular", {"dep": 10}), ("ar1", {"phi": 0.9}),
                        ("persistent_component", {"phi": 0.9, "weight": 0.2}),
                        ("seasonal_regime", {"period": 50, "phi_low": 0.1, "phi_high": 0.8})):
        x = fam.FAMILIES[name](20000, rng=rng, **kwargs)
        se = x.std() / math.sqrt(x.size)
        assert abs(x.mean()) < 6 * se, (name, x.mean(), se)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
