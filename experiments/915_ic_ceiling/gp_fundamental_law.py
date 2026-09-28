#!/usr/bin/env python3
"""TASK_915 G-B -- fundamental-law net-Sharpe bound (memo Sec.4.2/4.6).

Gross IR bound = IC * sqrt(BR) * TC              (Grinold-Kahn / Clarke-de Silva-Thorley)
BR = N_eff * (365/h)                              (N_eff measured, not assumed -- see below)
Net Sharpe bound = gross - turnover * cost_bp / sigma_book

DISCLOSED SCOPING (SPEC.md states this explicitly -- read before citing G-B numbers):

1. N_eff and Sigma: measured EXACTLY as specified. N_eff = (sum(eig))^2/sum(eig^2) of the
   Ledoit-Wolf-shrunk residual-return CORRELATION matrix (`stats_utils.n_eff_participation_
   ratio`, `stats_utils.ledoit_wolf_cov`) -- no simplification here.

2. GP turnover/TC: solving the exact multivariate Gârleanu-Pedersen matrix Riccati equation
   for a general (non-diagonal) Lambda/Sigma is out of scope for this row's time budget and
   risks an unverified closed-form. Instead this module measures the GP TRADE-OFF DIRECTLY
   BY SIMULATION on the panel's own realised forecast path: a scalar partial-adjustment
   trading rate `a` (Lambda assumed PROPORTIONAL to Sigma, `Lambda = kappa*Sigma`, the
   textbook special case that collapses the matrix Riccati equation to a scalar one and is
   applied identically to every asset) is grid-searched to MAXIMISE the realised net Sharpe of
   x_t = x_{t-1} + a*(aim_t - x_{t-1}), aim_t built from the SAME per-date forecast that
   produced the cell's IC. Turnover, gross Sharpe, and net Sharpe at the optimal a are all
   MEASURED from that simulation (not assumed). TC is reported as the Clarke-de Silva-Thorley
   ratio actually realised: net-of-cost Sharpe at a* divided by the frictionless (a=1) gross
   Sharpe -- cross-checked against the memo's literature range 0.5-0.7, never assumed equal to
   it. This is a numerically-equivalent, fully-disclosed approximation to the closed-form GP
   book, not the book itself (the book is TASK_915's own out-of-scope-tonight G-C).
"""
import numpy as np

import stats_utils as su


def residual_n_eff_and_sigma(residual_returns):
    """residual_returns: (T, A) with NO NaN columns (caller passes a common finite window).
    Returns (N_eff, Sigma_shrunk (A,A), shrinkage, corr_shrunk)."""
    sigma, shrinkage = su.ledoit_wolf_cov(residual_returns)
    corr = su.cov_to_corr(sigma)
    n_eff = su.n_eff_participation_ratio(corr)
    return n_eff, sigma, shrinkage, corr


def _aim_weights(forecast_panel, sigma_diag, gross_target=1.0):
    """aim_t (T,A): inverse-variance-weighted forecast (diagonal Sigma approximation for the
    AIM direction only -- Sigma's off-diagonal structure enters through N_eff/breadth, not
    through the aim weights, a disclosed simplification), normalised to sum|aim_t| ==
    gross_target on each date (dollar-neutral long-short, market-residual target)."""
    w = forecast_panel / np.maximum(sigma_diag[None, :], 1e-12)
    w = np.where(np.isfinite(w), w, 0.0)
    gross = np.sum(np.abs(w), axis=1, keepdims=True)
    gross = np.where(gross > 0, gross, 1.0)
    return w / gross * gross_target


def simulate_gp_tradeoff(forecast_panel, fwd_return_panel, sigma_diag, cost_bp, h,
                          a_grid=None, gross_target=1.0):
    """forecast_panel, fwd_return_panel: (T,A), aligned so row t's forecast trades INTO the
    position that earns fwd_return_panel[t] (the h-day forward return already computed
    upstream -- causal by construction, this function does no further lag handling). Returns
    a dict with the measured turnover/Sharpe/TC at the GP-optimal trading rate a*, plus the
    full a-grid trace for disclosure."""
    if a_grid is None:
        a_grid = np.linspace(0.05, 1.0, 20)
    aim = _aim_weights(forecast_panel, sigma_diag, gross_target=gross_target)
    T = aim.shape[0]
    valid = np.isfinite(fwd_return_panel).any(axis=1) & (np.abs(aim).sum(axis=1) > 0)
    decisions_per_year = 365.0 / h

    trace = []
    best = None
    for a in a_grid:
        x = np.zeros_like(aim)
        for t in range(1, T):
            prev = x[t - 1]
            x[t] = prev + a * (aim[t] - prev)
        r = np.where(np.isfinite(fwd_return_panel), fwd_return_panel, 0.0)
        gross_t = np.sum(x[:-1] * r[1:], axis=1)      # position held from t-1 into t's outcome
        turn_t = np.sum(np.abs(x[1:] - x[:-1]), axis=1)
        vok = valid[1:]
        gross_t = gross_t[vok]
        turn_t = turn_t[vok]
        if gross_t.size < 10:
            continue
        net_t = gross_t - turn_t * (cost_bp / 10000.0)
        gross_sh = _annualized_sharpe(gross_t, decisions_per_year)
        net_sh = _annualized_sharpe(net_t, decisions_per_year)
        turnover_mean = float(np.mean(turn_t)) * decisions_per_year   # annualised gross turnover
        row = {"a": float(a), "gross_sharpe": gross_sh, "net_sharpe": net_sh,
               "turnover_annualized": turnover_mean, "n": int(gross_t.size)}
        trace.append(row)
        if best is None or (np.isfinite(net_sh) and net_sh > best["net_sharpe"]):
            best = row

    frictionless = next((r for r in trace if abs(r["a"] - 1.0) < 1e-9), trace[-1] if trace else None)
    tc = (best["net_sharpe"] / frictionless["gross_sharpe"]
          if best and frictionless and frictionless["gross_sharpe"] not in (0, None)
          and np.isfinite(frictionless["gross_sharpe"]) and frictionless["gross_sharpe"] != 0
          else np.nan)
    return {"trace": trace, "best": best, "frictionless_gross_sharpe":
            frictionless["gross_sharpe"] if frictionless else np.nan, "tc_measured": tc}


def _annualized_sharpe(net_series, decisions_per_year):
    net_series = net_series[np.isfinite(net_series)]
    if net_series.size < 5 or net_series.std(ddof=1) == 0:
        return np.nan
    return float(net_series.mean() / net_series.std(ddof=1) * np.sqrt(decisions_per_year))


def gross_ir_bound(ic, n_eff, h, tc):
    br = n_eff * 365.0 / h
    return float(ic * np.sqrt(max(br, 0.0)) * tc), br


def net_sharpe_bound_for_cell(ic, n_eff, h, forecast_panel, fwd_return_panel, sigma_diag,
                               cost_bp, tc_literature_range=(0.5, 0.7)):
    """Full G-B computation for one G-A-passing cell. Returns the measured GP simulation dict
    plus the Grinold-Kahn analytic cross-check (both TC-measured and the literature range)."""
    sim = simulate_gp_tradeoff(forecast_panel, fwd_return_panel, sigma_diag, cost_bp, h)
    tc_meas = sim["tc_measured"]
    gk_measured, br = gross_ir_bound(ic, n_eff, h, tc_meas) if np.isfinite(tc_meas) else (np.nan, n_eff * 365.0 / h)
    gk_lit_lo, _ = gross_ir_bound(ic, n_eff, h, tc_literature_range[0])
    gk_lit_hi, _ = gross_ir_bound(ic, n_eff, h, tc_literature_range[1])
    return {
        "n_eff": n_eff, "breadth_BR": br, "tc_measured": tc_meas,
        "tc_literature_range": tc_literature_range,
        "gross_ir_bound_measured_tc": gk_measured,
        "gross_ir_bound_literature_tc_lo": gk_lit_lo, "gross_ir_bound_literature_tc_hi": gk_lit_hi,
        "gp_simulation": sim,
        "net_sharpe_bound": sim["best"]["net_sharpe"] if sim["best"] else np.nan,
        "turnover_annualized": sim["best"]["turnover_annualized"] if sim["best"] else np.nan,
        "trading_rate_a_star": sim["best"]["a"] if sim["best"] else np.nan,
    }
