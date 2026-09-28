#!/usr/bin/env python3
"""Unit tests for stats_utils.py: NW SE reduces to iid SE at lag=0 and against a hand-built
AR(1) case; Ledoit-Wolf shrinkage sanity vs a hand case; N_eff bounds; Spearman IC vs scipy
directly; Bonferroni t_crit; MDE formula sanity."""
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sstats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import stats_utils as su  # noqa: E402


def test_nw_se_lag0_equals_iid_se():
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    se_iid = x.std(ddof=1) / np.sqrt(x.size)  # note: population variance /n at lag0 in NW,
    se_nw = su.newey_west_se_of_mean(x, lag=0)
    # NW's gamma0 uses /n (population, ddof=0); iid uses ddof=1/sqrt(n) -- reconcile explicitly.
    se_iid_pop = x.std(ddof=0) / np.sqrt(x.size)
    assert abs(se_nw - se_iid_pop) < 1e-12, (se_nw, se_iid_pop, se_iid)


def test_nw_se_hand_ar1_case():
    """Hand case: x_t iid so autocovariances at h>0 are ~0 in EXPECTATION; NW with lag>0 should
    stay close to the lag-0 value for a large iid sample (statistical, not exact, tolerance)."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=2000)
    se0 = su.newey_west_se_of_mean(x, lag=0)
    se5 = su.newey_west_se_of_mean(x, lag=5)
    assert abs(se0 - se5) / se0 < 0.15


def test_nw_se_detects_positive_autocorrelation():
    """A strongly autocorrelated AR(1) series has a NW SE well ABOVE the naive iid SE (the
    whole point of the correction)."""
    rng = np.random.default_rng(2)
    n = 3000
    eps = rng.normal(size=n)
    x = np.zeros(n)
    phi = 0.8
    for t in range(1, n):
        x[t] = phi * x[t - 1] + eps[t]
    se_iid = x.std(ddof=0) / np.sqrt(n)
    se_nw = su.newey_west_se_of_mean(x, lag=20)
    assert se_nw > se_iid * 1.5


def test_spearman_ic_matches_scipy():
    rng = np.random.default_rng(3)
    f = rng.normal(size=30)
    y = f + rng.normal(size=30) * 0.5
    ic, n = su.spearman_ic(f, y)
    want, _ = sstats.spearmanr(f, y)
    assert n == 30
    assert abs(ic - want) < 1e-12


def test_ledoit_wolf_shrinkage_bounds_and_diagonal_case():
    rng = np.random.default_rng(4)
    # true covariance is diagonal (independent assets) -> sample off-diagonals are pure noise;
    # LW should shrink meaningfully (shrinkage > 0) and the shrunk matrix should be closer to
    # diagonal than the raw sample covariance (Frobenius distance to its own diagonal).
    T, A = 60, 12
    true_sd = rng.uniform(0.5, 2.0, size=A)
    X = rng.normal(size=(T, A)) * true_sd
    sample_cov = np.cov(X, rowvar=False)
    shrunk, shrinkage = su.ledoit_wolf_cov(X)
    assert 0.0 <= shrinkage <= 1.0
    off_sample = sample_cov - np.diag(np.diag(sample_cov))
    off_shrunk = shrunk - np.diag(np.diag(shrunk))
    assert np.abs(off_shrunk).sum() < np.abs(off_sample).sum()


def test_n_eff_bounds_and_single_factor_case():
    A = 20
    ident = np.eye(A)
    assert abs(su.n_eff_participation_ratio(ident) - A) < 1e-8       # fully independent -> N_eff=A
    ones = np.ones((A, A))
    assert abs(su.n_eff_participation_ratio(ones) - 1.0) < 1e-6      # perfectly collinear -> N_eff=1
    # single dominant factor model: corr = (1-rho)*I + rho*ones, rho close to 1 -> N_eff close to 1
    rho = 0.9
    corr = (1 - rho) * ident + rho * ones
    neff = su.n_eff_participation_ratio(corr)
    assert 1.0 < neff < A


def test_bonferroni_t_crit_matches_norm_ppf():
    t = su.bonferroni_t_crit(0.05, 160)
    # alpha/(2*160) two-sided upper tail
    want = sstats.norm.ppf(1 - 0.05 / (2 * 160))
    assert abs(t - want) < 1e-8
    assert t > sstats.norm.ppf(0.975)  # stricter than the uncorrected 95% bar


def test_mde_formula_sane_scaling():
    mde_more_dates, tcrit1 = su.mde_ic(n_bar_minus1_assets=99, n_dates=900, alpha=0.05,
                                       n_tests=160, nw_inflation=2.0)
    mde_fewer_dates, _ = su.mde_ic(n_bar_minus1_assets=99, n_dates=225, alpha=0.05,
                                    n_tests=160, nw_inflation=2.0)
    # MDE ~ 1/sqrt(n_dates): quartering n_dates should roughly double the MDE
    assert 1.8 < (mde_fewer_dates / mde_more_dates) < 2.2
    mde_more_assets, _ = su.mde_ic(n_bar_minus1_assets=399, n_dates=900, alpha=0.05,
                                    n_tests=160, nw_inflation=2.0)
    assert mde_more_assets < mde_more_dates


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)}/{len(fns)} stats_utils tests passed")
