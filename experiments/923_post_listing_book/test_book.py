#!/usr/bin/env python3
"""TASK_923 unit tests -- fixture smoke (tiny synthetic listings; no DB touched) covering
the book mechanics (entry timing, cost application, clustering, MDE) and the mandatory
zero/one-event degenerate guards, PLUS a small placebo/LOO/capacity/by-year smoke on
synthetic data. Run via pytest, wrapped, on gpu-server (never bare on the workstation).

No network, no DB, no fixture read here -- pure numpy synthetic panels, matching the task's
"fixture smoke first (tiny synthetic listings) -> then the real panel" instruction (this
file is the smoke half; `run_book.py` against `fixtures/trading_fixture.db` is the real
half).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import book as bk        # noqa: E402
import battery            # noqa: E402
import loader             # noqa: E402

SEC_PER_DAY = 86400
DAY0 = 1577836800  # 2020-01-01T00:00:00Z


def make_synthetic_panel(T=400, A=4, seed=0):
    """4-asset synthetic panel:
      asset 0: listed day 0 (LEFT-CENSORED -- must be excluded from the event universe)
      asset 1: listed day 50, active through day 399 (long-lived: eligible for every exit
               and for placebo admissibility, life=350 >= 2*90)
      asset 2: listed day 60, active through day 399 (same, second long-lived asset so
               cohort-week clustering has >1 cluster)
      asset 3: listed day 380, active through day 399 (short-lived: life=20, ineligible for
               horizons > ~15 days and for any placebo shift -- degeneracy exercise)
    Asset 1 is given a strong NEGATIVE post-listing residual drift (market-neutral short
    should profit); asset 2 is close to flat (near-zero edge); asset 3 is too short-lived to
    contribute past h=5.
    """
    rng = np.random.default_rng(seed)
    day_ts = DAY0 + np.arange(T, dtype=np.int64) * SEC_PER_DAY
    close = np.full((T, A), np.nan)
    listed_from = [0, 50, 60, 380]
    listed_to = [399, 399, 399, 399]
    drift = [0.0, -0.01, 0.0002, -0.005]     # daily log-return drift, asset-specific
    for i in range(A):
        f, l = listed_from[i], listed_to[i]
        n = l - f + 1
        noise = rng.normal(0, 0.02, size=n)
        logret = drift[i] + noise
        logc = np.cumsum(logret)
        close[f:l + 1, i] = np.exp(logc) * 100.0
    volume = np.where(np.isfinite(close), 1000.0 + 10.0 * np.arange(T)[:, None], np.nan)
    return {"day_ts": day_ts, "close": close, "volume": volume,
            "ids": list(range(A)), "syms": [f"SYN{i}" for i in range(A)], "info": {}}


@pytest.fixture
def panel():
    return make_synthetic_panel()


@pytest.fixture
def mkt_inputs(panel):
    return loader.market_residual_inputs(panel)


# =========================================================================================
# Entry timing / listing-event universe
# =========================================================================================

def test_entry_indices_same_close_vs_t1(mkt_inputs):
    first = mkt_inputs["first"]
    same = bk.entry_indices(first, "same_close")
    t1 = bk.entry_indices(first, "t1")
    assert np.array_equal(same, first)
    assert np.array_equal(t1, first + 1)


def test_entry_indices_bad_rule_raises(mkt_inputs):
    with pytest.raises(ValueError):
        bk.entry_indices(mkt_inputs["first"], "bogus")


def test_listing_events_excludes_left_censored(mkt_inputs):
    asset_cols, n_excluded = bk.listing_events(mkt_inputs["first"], mkt_inputs["last"])
    assert 0 not in asset_cols.tolist()          # asset 0 listed at panel-start row 0
    assert n_excluded == 1
    assert set(asset_cols.tolist()) == {1, 2, 3}


def test_listing_events_all_excluded_is_empty_not_crash():
    first = np.array([0, 0, -1])
    last = np.array([10, 20, -1])
    asset_cols, n_excluded = bk.listing_events(first, last)
    assert asset_cols.size == 0
    assert n_excluded == 2


# =========================================================================================
# compute_event_returns / net_markout_bp mechanics
# =========================================================================================

def test_compute_event_returns_matches_manual_calc(mkt_inputs, panel):
    asset_cols = np.array([1, 2])
    entry_idx = bk.entry_indices(mkt_inputs["first"], "t1")[asset_cols]
    horizon = 10
    ev = bk.compute_event_returns(mkt_inputs, panel["day_ts"], asset_cols, entry_idx, horizon)
    assert ev["valid"].all()
    logc = mkt_inputs["logc"]
    mkt = mkt_inputs["mkt"]
    for k, i in enumerate(asset_cols):
        e, x = int(entry_idx[k]), int(entry_idx[k] + horizon)
        assert ev["asset_log_ret"][k] == pytest.approx(logc[x, i] - logc[e, i])
        assert ev["mkt_log_ret"][k] == pytest.approx(float(np.sum(mkt[e + 1:x + 1])))
        assert ev["residual"][k] == pytest.approx(ev["asset_log_ret"][k] - ev["mkt_log_ret"][k])
        assert ev["hold_days"][k] == pytest.approx(horizon)


def test_compute_event_returns_drops_out_of_range_exit(mkt_inputs, panel):
    # asset 3 (life=20) cannot support a 60-day horizon from its T+1 entry.
    asset_cols = np.array([3])
    entry_idx = bk.entry_indices(mkt_inputs["first"], "t1")[asset_cols]
    ev = bk.compute_event_returns(mkt_inputs, panel["day_ts"], asset_cols, entry_idx, 60)
    assert not ev["valid"][0]
    assert np.isnan(ev["residual"][0])


def test_net_markout_bp_hedge_doubles_fee_leg(mkt_inputs, panel):
    asset_cols = np.array([1])
    entry_idx = bk.entry_indices(mkt_inputs["first"], "t1")[asset_cols]
    ev = bk.compute_event_returns(mkt_inputs, panel["day_ts"], asset_cols, entry_idx, 21)
    hedged = bk.net_markout_bp(ev, hedge=True, fee_bp=80.0, borrow_annual=0.0)
    unhedged = bk.net_markout_bp(ev, hedge=False, fee_bp=80.0, borrow_annual=0.0)
    assert hedged["fee_total_bp"] == 160.0
    assert unhedged["fee_total_bp"] == 80.0
    # gross differs (residual vs raw asset return); net = gross - fee, so the DIFFERENCE
    # between hedged and unhedged net must equal (unhedged_gross - hedged_gross) - 80bp.
    diff = unhedged["net_bp"][0] - hedged["net_bp"][0]
    expected = (unhedged["gross_bp"][0] - hedged["gross_bp"][0]) + 80.0
    assert diff == pytest.approx(expected)


def test_net_markout_bp_borrow_scales_with_hold_days(mkt_inputs, panel):
    asset_cols = np.array([1])
    entry_idx = bk.entry_indices(mkt_inputs["first"], "t1")[asset_cols]
    ev10 = bk.compute_event_returns(mkt_inputs, panel["day_ts"], asset_cols, entry_idx, 10)
    ev20 = bk.compute_event_returns(mkt_inputs, panel["day_ts"], asset_cols, entry_idx, 20)
    borrow10 = bk.net_markout_bp(ev10, hedge=True, fee_bp=0.0, borrow_annual=0.876)["borrow_bp"][0]
    borrow20 = bk.net_markout_bp(ev20, hedge=True, fee_bp=0.0, borrow_annual=0.876)["borrow_bp"][0]
    assert borrow20 == pytest.approx(2 * borrow10)
    assert borrow10 == pytest.approx(0.876 * (10 / 365.0) * 1e4)


def test_net_markout_bp_never_zero_notional_is_zero_borrow(mkt_inputs, panel):
    asset_cols = np.array([1])
    entry_idx = bk.entry_indices(mkt_inputs["first"], "t1")[asset_cols]
    ev = bk.compute_event_returns(mkt_inputs, panel["day_ts"], asset_cols, entry_idx, 21)
    out = bk.net_markout_bp(ev, hedge=True, fee_bp=0.0, borrow_annual=0.876, notional=0.0)
    assert out["borrow_bp"][0] == 0.0


# =========================================================================================
# Clustering / MDE / degenerate guards (fail loud, never crash, never fabricate a number)
# =========================================================================================

def test_clustered_mean_se_zero_events_guard():
    stat = bk.clustered_mean_se(np.array([]), np.array([]))
    assert stat["n_events"] == 0
    assert stat["degenerate"] is True
    assert np.isnan(stat["mean"])
    assert np.isnan(stat["se"])


def test_clustered_mean_se_one_event_guard():
    stat = bk.clustered_mean_se(np.array([42.0]), np.array([7]))
    assert stat["n_events"] == 1
    assert stat["degenerate"] is True
    assert stat["mean"] == 42.0
    assert np.isnan(stat["sd"])
    assert np.isnan(stat["se"])


def test_clustered_mean_se_single_cluster_degenerate():
    stat = bk.clustered_mean_se(np.array([10.0, 20.0, 30.0]), np.array([1, 1, 1]))
    assert stat["n_events"] == 3
    assert stat["n_clusters"] == 1
    assert stat["degenerate"] is True
    assert np.isnan(stat["se"])


def test_clustered_mean_se_multi_cluster_finite():
    values = np.array([10.0, 20.0, 30.0, 40.0])
    cohorts = np.array([1, 1, 2, 2])
    stat = bk.clustered_mean_se(values, cohorts)
    assert stat["degenerate"] is False
    assert stat["n_clusters"] == 2
    assert stat["mean"] == pytest.approx(25.0)
    assert stat["se"] == pytest.approx(stat["sd"] / np.sqrt(2))


def test_clustered_mean_se_ignores_nan():
    values = np.array([10.0, np.nan, 20.0, 30.0])
    cohorts = np.array([1, 1, 2, 2])
    stat = bk.clustered_mean_se(values, cohorts)
    assert stat["n_events"] == 3


def test_mde_nan_when_se_nan():
    assert np.isnan(bk.mde(float("nan")))


def test_mde_scales_with_z():
    assert bk.mde(2.0) == pytest.approx(bk.Z_MDE_80PWR_5PCT_TWOSIDED * 2.0)


def test_ci_bonferroni_degenerate_returns_nan():
    stat = bk.clustered_mean_se(np.array([1.0]), np.array([1]))
    lo, hi = bk.ci_bonferroni(stat)
    assert np.isnan(lo) and np.isnan(hi)


def test_z_bonferroni_matches_hand_value():
    # alpha_family=0.05 over 5 exits -> per-test alpha 0.01 -> two-sided z ~= 2.5758
    z = bk.z_bonferroni(n_family=5, alpha_family=0.05)
    assert z == pytest.approx(2.5758, abs=5e-3)


# =========================================================================================
# Honesty battery smoke
# =========================================================================================

def test_placebo_draw_respects_admissibility(mkt_inputs):
    rng = np.random.default_rng(1)
    asset_cols, placebo_first = battery.placebo_draw_asset_cols_and_first(
        mkt_inputs["first"], mkt_inputs["last"], rng, min_days=90)
    # asset 3 (life=20) can never be admissible at min_days=90; assets 1/2 (life=350/340) are.
    assert 3 not in asset_cols.tolist()
    assert set(asset_cols.tolist()) <= {1, 2}
    for k, i in enumerate(asset_cols):
        assert mkt_inputs["first"][i] <= placebo_first[k] <= mkt_inputs["last"][i]


def test_run_placebo_smoke(mkt_inputs, panel):
    out = battery.run_placebo(mkt_inputs, panel["day_ts"], horizon_days=10, n_draws=5,
                              seed=1, fee_bp=80.0, borrow_annual=0.876)
    assert out["n_draws"] == 5
    assert len(out["draws"]) == 5


def test_placebo_clean_degenerate_when_no_finite_draws():
    out = battery.placebo_clean(10.0, {"draws": [float("nan")] * 3, "mean": float("nan"),
                                      "sd": float("nan"), "max_abs": float("nan")})
    assert out["clean"] is False


def test_leave_one_out_hand_computed():
    values = np.array([10.0, 20.0, 30.0, 40.0])
    cohorts = np.array([1, 1, 2, 2])
    out = battery.leave_one_out(values, cohorts)
    assert out["degenerate"] is False
    # drop 10 -> mean(20,30,40)=30; drop 20 -> mean(10,30,40)=26.667; etc.
    assert sorted(out["loo_means"]) == pytest.approx(
        sorted([np.mean([20, 30, 40]), np.mean([10, 30, 40]),
               np.mean([10, 20, 40]), np.mean([10, 20, 30])]))


def test_leave_one_out_degenerate_below_3():
    out = battery.leave_one_out(np.array([10.0, 20.0]), np.array([1, 2]))
    assert out["degenerate"] is True
    assert out["loo_means"] == []


def test_capacity_disclosure_basic(panel, mkt_inputs):
    asset_cols = np.array([1, 2])
    entry_idx = bk.entry_indices(mkt_inputs["first"], "t1")[asset_cols]
    ev = bk.compute_event_returns(mkt_inputs, panel["day_ts"], asset_cols, entry_idx, 5)
    out = battery.capacity_disclosure(panel, panel["day_ts"], asset_cols, entry_idx,
                                      ev["valid"], window_days=5,
                                      assumed_notional_usd=10000.0, participation_rate=0.01)
    assert out["n_events"] == 2
    assert out["ceiling_usd"]["min"] > 0


def test_capacity_disclosure_zero_events_no_crash(panel):
    out = battery.capacity_disclosure(panel, panel["day_ts"], np.array([], dtype=np.int64),
                                      np.array([], dtype=np.int64), np.array([], dtype=bool))
    assert out["n_events"] == 0
    assert out["ceiling_usd"] is None


def test_by_cohort_year_buckets_correctly(panel, mkt_inputs):
    asset_cols = np.array([1, 2])
    entry_idx = bk.entry_indices(mkt_inputs["first"], "t1")[asset_cols]
    values = np.array([100.0, -50.0])
    out = battery.by_cohort_year(panel["day_ts"], entry_idx, values)
    assert "2020" in out
    assert out["2020"]["n_events"] == 2


# =========================================================================================
# Cycle-2 fix: POST-HOC corrected G2 (task wording) vs the as-implemented |.|-about-zero test
# =========================================================================================

def _synthetic_book(rng, edge_bp, cost_bp, n_events=400, n_clusters=100, sd_bp=500.0):
    """Per-event NET markouts of a book with a constant per-event cost: gross = edge + noise,
    net = gross - cost. Returns (net_values, cohort_ids)."""
    gross = edge_bp + rng.normal(0.0, sd_bp, size=n_events)
    cohorts = np.arange(n_events) % n_clusters
    return gross - cost_bp, cohorts


def _synthetic_placebo_summary(rng, cost_bp, n_draws=200):
    """What run_placebo returns for a ZERO-edge null book carrying the same constant cost:
    per-draw clustered means centred at -cost (not at zero)."""
    means = np.array([bk.clustered_mean_se(*_synthetic_book(rng, 0.0, cost_bp))["mean"]
                      for _ in range(n_draws)])
    return {"n_draws": n_draws, "n_finite": n_draws, "mean": float(np.mean(means)),
            "sd": float(np.std(means, ddof=1)), "max_abs": float(np.max(np.abs(means))),
            "draws": means.tolist()}


def _real_cell(values, cohorts):
    stat = bk.clustered_mean_se(values, cohorts)
    lo, hi = bk.ci_bonferroni(stat)
    return stat["mean"], lo, hi


def test_corrected_g2_separates_real_edge_under_large_constant_cost_old_criterion_does_not():
    rng = np.random.default_rng(923)
    cost = 2000.0
    placebo = _synthetic_placebo_summary(rng, cost)
    real_mean, lo, hi = _real_cell(*_synthetic_book(rng, edge_bp=1500.0, cost_bp=cost))
    # real book: net ~ -500bp (still cost-negative) but +1500bp gross edge over the null.
    assert real_mean < 0
    # The defect: |real| (~500) never exceeds max|placebo| (~2000) -> old test says not clean.
    old = battery.placebo_clean(real_mean, placebo)
    assert old["clean"] is False
    # Corrected (task wording): the real CI lies wholly above the placebo's central 95%.
    new = battery.placebo_separation_task_g2(real_mean, lo, hi, placebo["draws"])
    assert new["excludes"] is True
    assert new["direction"] == "above"
    assert new["k_ge"] == 0 and new["p_one_sided"] == 0.0
    assert new["placebo_mean"] == pytest.approx(-cost, abs=50.0)


def test_corrected_g2_zero_edge_real_book_does_not_separate():
    rng = np.random.default_rng(924)
    cost = 2000.0
    placebo = _synthetic_placebo_summary(rng, cost)
    real_mean, lo, hi = _real_cell(*_synthetic_book(rng, edge_bp=0.0, cost_bp=cost))
    new = battery.placebo_separation_task_g2(real_mean, lo, hi, placebo["draws"])
    assert new["excludes"] is False
    assert new["direction"] is None
    assert 0.05 < new["p_one_sided"] < 0.95


def test_corrected_g2_degenerate_inputs_do_not_separate():
    out = battery.placebo_separation_task_g2(10.0, float("nan"), float("nan"), [1.0, 2.0])
    assert out["excludes"] is False and out["direction"] is None
    out = battery.placebo_separation_task_g2(10.0, 0.0, 20.0, [float("nan")] * 3)
    assert out["excludes"] is False and out["n_placebo_finite"] == 0


def test_corrected_g2_reports_unfavourable_side():
    out = battery.placebo_separation_task_g2(-100.0, -150.0, -50.0,
                                             np.linspace(0.0, 100.0, 201))
    assert out["excludes"] is True and out["direction"] == "below"
    assert out["p_one_sided"] == 1.0
