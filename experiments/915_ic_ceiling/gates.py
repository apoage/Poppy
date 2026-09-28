#!/usr/bin/env python3
"""TASK_915 G-A / G-B gate evaluation from one cell's OOS forecast panel."""
import datetime as dt

import numpy as np

import stats_utils as su
import gp_fundamental_law as gpl
import grid

GA_WINDOW = ("2024-01-01", "2026-07-01")   # [lo, hi)
YEAR_WINDOWS = {"2024": ("2024-01-01", "2025-01-01"), "2025": ("2025-01-01", "2026-01-01"),
                "2026H1": ("2026-01-01", "2026-07-01")}
ALPHA = 0.05
# POST-RUN CHANGE (TASK_915 cycle 2, 2026-09-28): derived from the grid instead of a duplicated
# literal. Value unchanged (160); the official run used the frozen literal (gates.py sha
# 4d9ffad1...), see SPEC.md POST-HOC ERRATA.
N_TESTS = len(grid.ALL_CELLS)


def _epoch(date_str):
    y, m, d = (int(x) for x in date_str.split("-"))
    return int(dt.datetime(y, m, d, tzinfo=dt.timezone.utc).timestamp())


def _window_rows(day_ts, lo, hi):
    return int(np.searchsorted(day_ts, _epoch(lo))), int(np.searchsorted(day_ts, _epoch(hi)))


def cell_ic_stats(forecast, target, day_ts, h):
    """Per-date cross-sectional Spearman IC over the G-A window, NW t (lag=h), Bonferroni
    threshold, per-year sign, tau_IC. Returns a flat dict (JSON-serialisable)."""
    lo, hi = _window_rows(day_ts, *GA_WINDOW)
    ic_full, n_live_full = su.per_date_ic_series(forecast, target)
    ic = ic_full[lo:hi]
    n_live = n_live_full[lo:hi]
    mean_ic, se, t_nw = su.newey_west_t(ic, lag=h)
    t_crit = su.bonferroni_t_crit(ALPHA, N_TESTS)

    by_year = {}
    for name, (ylo, yhi) in YEAR_WINDOWS.items():
        r0, r1 = _window_rows(day_ts, ylo, yhi)
        seg = ic_full[r0:r1]
        seg = seg[np.isfinite(seg)]
        by_year[name] = {"mean_ic": float(seg.mean()) if seg.size else np.nan, "n_dates": int(seg.size)}

    tau, monthly = su.tau_ic(ic[np.isfinite(ic)])
    tau_se = su.tau_ic_overlap_aware_se(monthly) if monthly.size >= 4 else np.nan

    n_bar = float(np.nanmean(n_live)) if np.isfinite(n_live).any() else np.nan
    n_dates = int(np.isfinite(ic).sum())

    return {
        "mean_ic": mean_ic, "nw_se": se, "nw_t": t_nw, "bonferroni_t_crit": t_crit,
        "n_dates_ga_window": n_dates, "n_bar_live_assets": n_bar,
        "by_year": by_year, "tau_ic": tau, "tau_ic_se": tau_se,
        "ic_series_ga_window": ic.tolist(), "day_ts_ga_window": day_ts[lo:hi].tolist(),
    }


def ga_pass(stats):
    if not (np.isfinite(stats["mean_ic"]) and stats["mean_ic"] >= 0.02):
        return False, "mean_ic < 0.02"
    if not (np.isfinite(stats["nw_t"]) and stats["nw_t"] >= stats["bonferroni_t_crit"]):
        return False, "NW t below Bonferroni-corrected critical value"
    for name, d in stats["by_year"].items():
        if not (np.isfinite(d["mean_ic"]) and d["mean_ic"] > 0):
            return False, f"IC not > 0 in {name}"
    return True, "PASS"


def gb_result(ic_mean, n_eff, h, forecast_panel, fwd_return_panel, sigma_diag, cost_bp_restated,
              cost_bp_legacy21=21.0):
    restated = gpl.net_sharpe_bound_for_cell(ic_mean, n_eff, h, forecast_panel, fwd_return_panel,
                                              sigma_diag, cost_bp_restated)
    legacy = gpl.net_sharpe_bound_for_cell(ic_mean, n_eff, h, forecast_panel, fwd_return_panel,
                                            sigma_diag, cost_bp_legacy21)
    binding = restated  # task precondition: grade at the 916-restated bar, not legacy21
    net = binding["net_sharpe_bound"]
    if np.isfinite(net) and net >= 0.97:
        verdict = "PASS"
    elif np.isfinite(net) and net >= 0.56:
        verdict = "MARGINAL"
    else:
        verdict = "FAIL"
    # POST-RUN CHANGE (TASK_915 cycle 2): machine-readable borrow disclosure. G-B charges the
    # 80bp base-taker rate on turnover only; 916's margin-rollover borrow on short notional is
    # NOT charged (no book exists at G-B, SPEC.md Sec.8). A PASS here is provisional until
    # re-graded with borrow -- consumers must not read `verdict` without `borrow_charged`.
    return {"restated_916": restated, "legacy_21bp": legacy, "binding": "restated_916",
            "verdict": verdict, "borrow_charged": False,
            "provisional_if_pass": True}
