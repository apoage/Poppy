#!/usr/bin/env python3
"""TASK_915 statistics: per-date cross-sectional Spearman IC, Newey-West SE, Ledoit-Wolf
shrinkage, N_eff (participation ratio), Bonferroni, the 2026-09-05 amendment null family
(block-Rademacher / circular-shift + per-date permutation), MDE at spec time.

Two SDs, BOTH stated per the 2026-09-05 amendment (never conflate them):
  - per-date cross-sectional IC null SD ~= 1/sqrt(N-1) over LIVE assets that date -- this is
    what `ic_mean_and_nw_se` tests IC != 0 against (via Newey-West on the date-series of ICs).
  - N_eff (this file's `n_eff_participation_ratio`) governs BREADTH in the fundamental law
    (BR = N_eff * decisions/yr) ONLY. Never divide by sqrt(N_eff) when testing IC != 0.
"""
import numpy as np
from scipy import stats as sstats


# ---------------------------------------------------------------------------
# per-date cross-sectional Spearman IC
# ---------------------------------------------------------------------------

def spearman_ic(forecast, target):
    """Spearman rank correlation between two 1-D arrays over the SAME live-asset set (NaNs
    dropped pairwise). Returns (ic, n_live) with ic=NaN if n_live<3."""
    ok = np.isfinite(forecast) & np.isfinite(target)
    n = int(ok.sum())
    if n < 3:
        return np.nan, n
    rho, _ = sstats.spearmanr(forecast[ok], target[ok])
    return float(rho), n


def per_date_ic_series(forecast_panel, target_panel):
    """forecast_panel, target_panel: (T, A). Returns (ic (T,), n_live (T,) int)."""
    T = forecast_panel.shape[0]
    ic = np.full(T, np.nan)
    n_live = np.zeros(T, dtype=int)
    for t in range(T):
        ic[t], n_live[t] = spearman_ic(forecast_panel[t], target_panel[t])
    return ic, n_live


# ---------------------------------------------------------------------------
# Newey-West standard error of a mean
# ---------------------------------------------------------------------------

def newey_west_se_of_mean(x, lag):
    """NW-HAC standard error of the SAMPLE MEAN of a (possibly serially correlated) 1-D series
    x (NaNs dropped). Bartlett kernel, max lag `lag`.

    var(mean) = (1/n^2) * sum_{h=-lag}^{lag} w(h) * sum_t (x_t-mean)(x_{t-h}-mean)
    with Bartlett weight w(h) = 1 - |h|/(lag+1). Reduces EXACTLY to the iid SE (std/sqrt(n))
    at lag=0 -- tested in tests/test_stats_utils.py against numpy's own variance."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return np.nan
    xc = x - x.mean()
    gamma0 = float(np.dot(xc, xc)) / n
    s = gamma0
    for h in range(1, min(lag, n - 1) + 1):
        w = 1.0 - h / (lag + 1)
        gamma_h = float(np.dot(xc[h:], xc[:-h])) / n
        s += 2.0 * w * gamma_h
    var_mean = s / n
    return float(np.sqrt(max(var_mean, 0.0)))


def newey_west_t(x, lag):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.nan, np.nan, np.nan
    mean = float(x.mean())
    se = newey_west_se_of_mean(x, lag)
    t = mean / se if se > 0 else np.nan
    return mean, se, t


# ---------------------------------------------------------------------------
# Ledoit-Wolf shrinkage (sklearn wrapper; test vs a hand case)
# ---------------------------------------------------------------------------

def ledoit_wolf_cov(returns):
    """returns: (T, A) with NaNs already handled by the caller (e.g. a common-window slice
    with no NaN columns). Returns (Sigma_shrunk (A,A), shrinkage in [0,1])."""
    from sklearn.covariance import LedoitWolf
    lw = LedoitWolf().fit(returns)
    return lw.covariance_, float(lw.shrinkage_)


# ---------------------------------------------------------------------------
# N_eff -- participation ratio of the residual correlation spectrum
# ---------------------------------------------------------------------------

def n_eff_participation_ratio(corr_or_cov):
    """N_eff = (sum(eig))^2 / sum(eig^2) of a correlation (or covariance) matrix's eigenvalue
    spectrum -- the effective number of independent bets (Grinold-Kahn breadth term). For a
    correlation matrix (unit diagonal, trace=A) this is bounded in [1, A]."""
    m = np.asarray(corr_or_cov, dtype=np.float64)
    eig = np.linalg.eigvalsh(0.5 * (m + m.T))
    eig = np.clip(eig, 0.0, None)
    s1 = eig.sum()
    s2 = (eig ** 2).sum()
    if s2 <= 0:
        return float(m.shape[0])
    return float((s1 ** 2) / s2)


def cov_to_corr(cov):
    d = np.sqrt(np.clip(np.diag(cov), 1e-300, None))
    return cov / np.outer(d, d)


# ---------------------------------------------------------------------------
# Bonferroni
# ---------------------------------------------------------------------------

def bonferroni_alpha(alpha, n_tests):
    return alpha / n_tests


def bonferroni_t_crit(alpha, n_tests, two_sided=True):
    a = bonferroni_alpha(alpha, n_tests)
    p = 1 - a / 2 if two_sided else 1 - a
    return float(sstats.norm.ppf(p))  # NW t-stats are asymptotically normal; large-n panel


# ---------------------------------------------------------------------------
# 2026-09-05 amendment: block-Rademacher / circular-shift + per-date permutation null family
# ---------------------------------------------------------------------------

def circular_shift_null(forecast_panel, target_panel, n_draws=500, block=None, rng=None):
    """Dependence-preserving null: circularly shift the DATE axis of `forecast_panel` relative
    to `target_panel` by a random offset (preserves each panel's own serial + cross-sectional
    dependence; breaks the forecast<->target alignment, which is the null hypothesis). Returns
    an array of `n_draws` null mean-IC statistics (same per-date-IC-then-mean-over-dates
    estimator as the real statistic)."""
    rng = rng or np.random.default_rng(0)
    T = forecast_panel.shape[0]
    out = np.full(n_draws, np.nan)
    for d in range(n_draws):
        shift = int(rng.integers(1, T))  # never 0 (that would be the real alignment)
        shifted = np.roll(forecast_panel, shift, axis=0)
        ic, _ = per_date_ic_series(shifted, target_panel)
        out[d] = np.nanmean(ic)
    return out


def block_rademacher_null(forecast_panel, target_panel, n_draws=500, block=21, rng=None):
    """Block-Rademacher placebo: multiply the FORECAST panel by a random +-1 sign, held fixed
    over blocks of `block` consecutive dates (preserves within-block serial dependence and the
    forecast's own cross-sectional structure at each date; a sign-flip null is PnL-invariant
    for a rank statistic like Spearman IC only through its SIGN, which is exactly the point --
    it destroys the true forecast<->target relationship at the block level while a pure
    reflection/time-reversal surrogate would not, per the 2026-09-05 amendment)."""
    rng = rng or np.random.default_rng(0)
    T = forecast_panel.shape[0]
    n_blocks = int(np.ceil(T / block))
    out = np.full(n_draws, np.nan)
    for d in range(n_draws):
        signs = rng.choice([-1.0, 1.0], size=n_blocks)
        sign_series = np.repeat(signs, block)[:T]
        flipped = forecast_panel * sign_series[:, None]
        ic, _ = per_date_ic_series(flipped, target_panel)
        out[d] = np.nanmean(ic)
    return out


def per_date_permutation_null(forecast_panel, target_panel, n_draws=500, rng=None):
    """Per-date cross-sectional permutation: shuffle asset labels of the target WITHIN each
    date independently (breaks the cross-sectional forecast<->target pairing at every date,
    keeps each date's own marginal distribution and the forecast's serial structure)."""
    rng = rng or np.random.default_rng(0)
    T, A = forecast_panel.shape
    out = np.full(n_draws, np.nan)
    for d in range(n_draws):
        ic_t = np.full(T, np.nan)
        for t in range(T):
            row_f = forecast_panel[t]
            row_y = target_panel[t]
            ok = np.isfinite(row_f) & np.isfinite(row_y)
            if ok.sum() < 3:
                continue
            idx = np.where(ok)[0]
            perm = rng.permutation(idx)
            yv_perm = row_y.copy()
            yv_perm[idx] = row_y[perm]
            ic_t[t], _ = spearman_ic(row_f, yv_perm)
        out[d] = np.nanmean(ic_t)
    return out


def null_family_summary(forecast_panel, target_panel, n_draws=200, rng_seed=0):
    """Runs all three null families and returns their mean/SD -- used as a SANITY CHECK on the
    NW SE (per the amendment), never as the gate itself (the gate is NW t >= 3 after
    Bonferroni on the REAL per-date IC series)."""
    rng = np.random.default_rng(rng_seed)
    cs = circular_shift_null(forecast_panel, target_panel, n_draws=n_draws, rng=rng)
    br = block_rademacher_null(forecast_panel, target_panel, n_draws=n_draws, rng=rng)
    pp = per_date_permutation_null(forecast_panel, target_panel, n_draws=n_draws, rng=rng)
    def _s(a):
        a = a[np.isfinite(a)]
        return {"mean": float(a.mean()) if a.size else np.nan,
                "sd": float(a.std(ddof=1)) if a.size > 1 else np.nan, "n": int(a.size)}
    return {"circular_shift": _s(cs), "block_rademacher": _s(br), "per_date_permutation": _s(pp)}


# ---------------------------------------------------------------------------
# monthly-IC autocorrelation tau_IC (2026-09-05 amendment, overlap-aware null)
# ---------------------------------------------------------------------------

def tau_ic(ic_series, dates_per_month=21):
    """Monthly-aggregated IC series' lag-1 autocorrelation (persistence statistic). Returns
    (tau, monthly_series)."""
    ic = np.asarray(ic_series, dtype=np.float64)
    T = ic.size
    n_months = T // dates_per_month
    if n_months < 3:
        return np.nan, np.array([])
    monthly = np.array([np.nanmean(ic[m * dates_per_month:(m + 1) * dates_per_month])
                         for m in range(n_months)])
    ok = np.isfinite(monthly)
    monthly = monthly[ok]
    if monthly.size < 3:
        return np.nan, monthly
    a, b = monthly[:-1], monthly[1:]
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0, monthly
    tau = float(np.corrcoef(a, b)[0, 1])
    return tau, monthly


def tau_ic_overlap_aware_se(monthly_series, lag=1):
    """NW SE of the lag-1 autocorrelation estimate, overlap-aware (Bartlett kernel over the
    monthly series itself, since consecutive monthly means overlap through shared within-month
    dates only at the boundary -- treated conservatively as lag-1 dependent)."""
    a, b = monthly_series[:-1], monthly_series[1:]
    if a.size < 4:
        return np.nan
    prod = (a - a.mean()) * (b - b.mean())
    return newey_west_se_of_mean(prod, lag=lag) / (a.std() * b.std() + 1e-12)


# ---------------------------------------------------------------------------
# SR0(N) multiplicity line (oracle-null selection inflation, 2026-09-05 amendment)
# ---------------------------------------------------------------------------

def sr0_multiplicity(n_cells, n_dates=None, alpha=0.05):
    """Expected max-of-N_cells Sharpe-ratio-equivalent inflation under the null (order
    statistics of N iid standard normals, converted to an annualised-Sharpe-like unit via
    1/sqrt(n_dates)). Reported beside every best cell per the amendment -- NOT a gate."""
    # E[max of n_cells iid N(0,1)] ~= sqrt(2*ln(n_cells)) (classical extreme-value approx)
    if n_cells <= 1:
        return 0.0
    return float(np.sqrt(2.0 * np.log(n_cells)))


# ---------------------------------------------------------------------------
# D85 MDE at spec time (population-only, no fitting)
# ---------------------------------------------------------------------------

def mde_ic(n_bar_minus1_assets, n_dates, alpha, n_tests, nw_inflation, power_t=3.0):
    """Minimum detectable mean IC at NW t>=`power_t` after Bonferroni, from the PANEL alone
    (asset counts per date, no fitting): MDE ~= t_crit * (1/sqrt(N_bar-1)) *
    sqrt(nw_inflation) / sqrt(n_dates). `power_t` defaults to the gate's own t>=3 bar (D85's
    "MDE at NW t=3 after Bonferroni")."""
    t_crit = bonferroni_t_crit(alpha, n_tests)
    per_date_sd = 1.0 / np.sqrt(max(n_bar_minus1_assets, 1))
    return float(power_t * per_date_sd * np.sqrt(nw_inflation) / np.sqrt(max(n_dates, 1))), t_crit
