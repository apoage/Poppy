#!/usr/bin/env python3
"""TASK_923 honesty battery: circular-shift placebo (G2), leave-one-listing-out, capacity
disclosure, by-cohort-year table. All four run through the SAME book code
(`book.compute_event_returns` / `book.net_markout_bp` / `book.clustered_mean_se`) that scores
the binding table -- only the listing-date input (placebo) or the event subset (LOO) differs.
"""
import datetime

import numpy as np

import book as bk


# =========================================================================================
# G2 — circular-shift placebo on the LISTING DATE (not just the IC — the whole book)
# =========================================================================================

def placebo_draw_asset_cols_and_first(first, last, rng, frac_lo=0.10, frac_hi=0.90,
                                      min_days=None):
    """One i.i.d. draw of a placebo listing date per admissible asset: each asset's OWN
    listing date is replaced by `first[i] + shift_days[i]`, `shift_days[i]` a uniform
    fraction of its own life in [frac_lo, frac_hi], clipped into the per-asset admissible
    band `min_days <= shift_days <= life - min_days` (same per-asset day-count condition
    P3 revision 3 uses, `per_asset_shift_bounds` reimplemented here rather than imported
    across the digit-prefixed 917 package for a single one-line bound formula -- the
    formula, not the IC-specific machinery, is what's shared). `min_days` defaults to
    `AGE_MAX+1` (90) so even the longest pre-registered exit (age* <= 89) always has 90 days
    of post-shift life to score against, mirroring P3's own bound exactly.

    Because `first[i]+shift_days[i] <= last[i]-min_days` and `shift_days[i] >= min_days`,
    the shifted point never needs circular wraparound (unlike P3's age-relabeling placebo,
    which relabels EVERY row in the asset's life): the book only ever reads forward from one
    entry day, so a straight forward shift already stays inside [first[i], last[i]] with
    margin. Returns (asset_cols admissible-only, placebo_first (same length)).
    """
    if min_days is None:
        min_days = 90
    first = np.asarray(first)
    last = np.asarray(last)
    # Same universe `book.listing_events` scores (left-censored panel-start rows excluded):
    # the placebo must draw from the SAME population the real binding table draws from, or
    # it is not a null of the real book, just a null of a differently-shaped population.
    listed = (first >= 0) & (first != bk.EXCLUDE_FIRST_ROW_IDX)
    life = np.where(listed, last - first + 1, 0)
    lo = np.full(life.shape, min_days, dtype=np.int64)
    hi = life - min_days
    ok = listed & (last > first) & (hi >= lo)
    asset_cols = np.nonzero(ok)[0].astype(np.int64)
    frac = rng.uniform(frac_lo, frac_hi, size=asset_cols.size)
    shift_days = np.rint(frac * life[asset_cols]).astype(np.int64)
    shift_days = np.clip(shift_days, lo[asset_cols], np.maximum(hi[asset_cols], lo[asset_cols]))
    placebo_first = first[asset_cols] + shift_days
    return asset_cols, placebo_first


def run_placebo(mkt_inputs, day_ts, horizon_days, n_draws, seed, fee_bp, borrow_annual,
                hedge=True, min_days=90):
    """N i.i.d. placebo draws of the binding book's clustered mean net markout at one
    horizon. Returns dict: draws (n_draws,) array of per-draw clustered means (NaN where a
    draw yields <2 events), plus the usual max/quantile/parametric summary P3's own
    revision-3 iid null reports (same reading style, applied to the BOOK statistic instead
    of the IC)."""
    rng = np.random.default_rng(seed)
    first, last = mkt_inputs["first"], mkt_inputs["last"]
    means = np.full(n_draws, np.nan)
    for d in range(n_draws):
        asset_cols, placebo_first = placebo_draw_asset_cols_and_first(
            first, last, rng, min_days=min_days)
        # entry rule mirrors the binding rule (T+1) regardless of `hedge` (hedge only
        # controls whether the hedge leg is charged/scored, not entry timing).
        entry_idx = placebo_first + 1
        ev = bk.compute_event_returns(mkt_inputs, day_ts, asset_cols, entry_idx, horizon_days)
        costed = bk.net_markout_bp(ev, hedge=hedge, fee_bp=fee_bp, borrow_annual=borrow_annual)
        cohort = bk.cohort_week_ids(day_ts, entry_idx)
        stat = bk.clustered_mean_se(costed["net_bp"], cohort)
        means[d] = stat["mean"] if not stat["degenerate"] else np.nan
    finite = means[np.isfinite(means)]
    if finite.size == 0:
        return {"n_draws": int(n_draws), "n_finite": 0, "mean": float("nan"),
                "sd": float("nan"), "max_abs": float("nan"), "draws": means.tolist()}
    return {"n_draws": int(n_draws), "n_finite": int(finite.size),
            "mean": float(np.mean(finite)), "sd": float(np.std(finite, ddof=1))
            if finite.size > 1 else float("nan"),
            "max_abs": float(np.max(np.abs(finite))),
            "q99_abs": float(np.percentile(np.abs(finite), 99)),
            "draws": means.tolist()}


def placebo_clean(real_mean, placebo_summary):
    """Same criterion family P3 revision 3 uses: NOT clean unless |real| > max|placebo| AND
    the empirical two-sided p <= 0.05. Returns dict with p_empirical, z, clean (bool).
    Degenerate (no finite placebo draws, or real_mean non-finite) -> clean=False, NULL-coded
    per G2's own rule ('if the placebo book matches the real book, the row's verdict is
    NULL')."""
    draws = np.asarray(placebo_summary["draws"], dtype=np.float64)
    finite = draws[np.isfinite(draws)]
    if not np.isfinite(real_mean) or finite.size == 0:
        return {"p_empirical": float("nan"), "z": float("nan"), "clean": False,
               "reason": "no finite real mean or no finite placebo draws"}
    n_ge = int(np.sum(np.abs(finite) >= abs(real_mean)))
    p = (1 + n_ge) / (1 + finite.size)
    sd = placebo_summary.get("sd", float("nan"))
    z = ((real_mean - placebo_summary["mean"]) / sd) if np.isfinite(sd) and sd > 0 \
        else float("nan")
    clean = bool(abs(real_mean) > placebo_summary["max_abs"] and p <= 0.05)
    return {"p_empirical": float(p), "z": float(z), "clean": clean,
            "max_abs_placebo": placebo_summary["max_abs"]}


def placebo_separation_task_g2(real_mean, ci_lo, ci_hi, placebo_draws, q_lo=2.5, q_hi=97.5):
    """POST-HOC CORRECTION (cycle-2 fix, see SPEC.md 'POST-HOC ERRATA'): TASK_923's own G2
    wording -- 'the circular-shift placebo ... must produce a distribution the real book's CI
    excludes'. `placebo_clean` above compares |real| to max|placebo| about ZERO, which is
    only a null test for a statistic the null centres at zero (P3's IC); a book with a
    constant per-horizon cost has a placebo centred at -cost, not zero. This compares the
    real book's (Bonferroni) CI against the placebo distribution ITSELF, wherever it sits.

    Returns placebo mean, placebo q_lo/q_hi quantiles of the per-draw means, whether the real
    CI excludes the placebo central interval (and on which side: 'above' = real book more
    favourable than placebo, 'below' = less), and the one-sided empirical p = share of finite
    placebo means >= real mean (raw share, k/N, plus k and N). Degenerate (non-finite real
    CI/mean or no finite placebo draws) -> excludes=False, direction=None, reason given."""
    draws = np.asarray(placebo_draws, dtype=np.float64)
    finite = draws[np.isfinite(draws)]
    if (finite.size == 0 or not np.isfinite(real_mean) or not np.isfinite(ci_lo)
            or not np.isfinite(ci_hi)):
        return {"n_placebo_finite": int(finite.size), "placebo_mean": float("nan"),
                "placebo_q_lo": float("nan"), "placebo_q_hi": float("nan"),
                "excludes": False, "direction": None, "k_ge": None,
                "p_one_sided": float("nan"),
                "reason": "non-finite real mean/CI or no finite placebo draws"}
    p_lo = float(np.percentile(finite, q_lo))
    p_hi = float(np.percentile(finite, q_hi))
    if ci_lo > p_hi:
        direction = "above"
    elif ci_hi < p_lo:
        direction = "below"
    else:
        direction = None
    k_ge = int(np.sum(finite >= real_mean))
    return {"n_placebo_finite": int(finite.size), "placebo_mean": float(np.mean(finite)),
            "placebo_q_lo": p_lo, "placebo_q_hi": p_hi,
            "excludes": direction is not None, "direction": direction,
            "k_ge": k_ge, "p_one_sided": k_ge / finite.size}


# =========================================================================================
# Leave-one-listing-out
# =========================================================================================

def leave_one_out(net_bp_values, cohort_ids):
    """Jackknife over EVENTS (not clusters): for each finite event k, the clustered mean
    recomputed with event k removed. Returns dict: loo_means (list), min, max,
    max_abs_delta_from_full (how far any single event can move the headline)."""
    values = np.asarray(net_bp_values, dtype=np.float64)
    cohort_ids = np.asarray(cohort_ids)
    finite_idx = np.nonzero(np.isfinite(values))[0]
    n = finite_idx.size
    full = bk.clustered_mean_se(values, cohort_ids)
    if n < 3:
        return {"n": n, "loo_means": [], "min": float("nan"), "max": float("nan"),
               "max_abs_delta": float("nan"), "full_mean": full["mean"],
               "degenerate": True}
    loo_means = []
    for k in finite_idx:
        mask = np.ones(values.shape, dtype=bool)
        mask[k] = False
        sub_vals = values[mask]
        sub_cohort = cohort_ids[mask]
        stat = bk.clustered_mean_se(sub_vals, sub_cohort)
        loo_means.append(stat["mean"])
    loo_means = np.asarray(loo_means, dtype=np.float64)
    lf = loo_means[np.isfinite(loo_means)]
    if lf.size == 0:
        return {"n": n, "loo_means": [], "min": float("nan"), "max": float("nan"),
               "max_abs_delta": float("nan"), "full_mean": full["mean"], "degenerate": True}
    max_abs_delta = float(np.max(np.abs(lf - full["mean"])))
    return {"n": n, "loo_means": lf.tolist(), "min": float(np.min(lf)),
            "max": float(np.max(lf)), "max_abs_delta": max_abs_delta,
            "full_mean": full["mean"], "degenerate": False}


# =========================================================================================
# Capacity disclosure
# =========================================================================================

def capacity_disclosure(panel, day_ts, asset_cols, entry_idx, valid, window_days=5,
                        assumed_notional_usd=10000.0, participation_rate=0.01):
    """n_events, ASSUMED notional per event (disclosed assumption, not measured), and a
    first-`window_days`-days dollar-volume ceiling read from the panel's own `volume` column
    (read-only; if it were absent this would say so instead of fabricating a number --
    the fixture's `ohlcv_daily` DOES carry a volume column, verified by schema inspection).
    Ceiling = `participation_rate` (1% default, a standard disclosed participation-rate
    heuristic, NOT a measured limit) of the summed first-window_days dollar volume
    (volume*close) at each event, read-only, requiring both finite in the same row."""
    close = panel["close"]
    volume = panel["volume"]
    T = close.shape[0]
    rows = []
    for k in range(len(asset_cols)):
        if not valid[k]:
            continue
        i = int(asset_cols[k])
        e = int(entry_idx[k])
        x = min(e + window_days, T - 1)
        c = close[e:x + 1, i]
        v = volume[e:x + 1, i]
        both = np.isfinite(c) & np.isfinite(v)
        dollar_vol = float(np.sum(c[both] * v[both])) if both.any() else float("nan")
        rows.append(dollar_vol)
    rows = np.asarray(rows, dtype=np.float64)
    finite = rows[np.isfinite(rows) & (rows > 0)]
    n_zero_or_nodata = int(rows.size - finite.size)
    if finite.size == 0:
        return {"n_events": int(rows.size), "n_zero_or_nodata_volume": n_zero_or_nodata,
               "assumed_notional_usd": assumed_notional_usd,
               "participation_rate": participation_rate, "ceiling_usd": None}
    ceiling = finite * participation_rate
    return {"n_events": int(rows.size), "n_zero_or_nodata_volume": n_zero_or_nodata,
            "assumed_notional_usd": assumed_notional_usd,
            "participation_rate": participation_rate,
            "first_window_dollar_volume_usd": {
                "min": float(np.min(finite)), "median": float(np.median(finite)),
                "mean": float(np.mean(finite)), "max": float(np.max(finite))},
            "ceiling_usd": {
                "min": float(np.min(ceiling)), "median": float(np.median(ceiling)),
                "mean": float(np.mean(ceiling)), "max": float(np.max(ceiling))},
            "n_events_ceiling_below_assumed_notional":
                int(np.sum(ceiling < assumed_notional_usd))}


# =========================================================================================
# By-cohort-year
# =========================================================================================

def by_cohort_year(day_ts, entry_idx, net_bp_values):
    day_ts = np.asarray(day_ts)
    entry_idx = np.asarray(entry_idx)
    values = np.asarray(net_bp_values, dtype=np.float64)
    years = np.array([
        datetime.datetime.utcfromtimestamp(int(day_ts[e])).year if e < day_ts.size else -1
        for e in entry_idx
    ])
    out = {}
    for yr in sorted(set(int(y) for y in years if y > 0)):
        m = (years == yr) & np.isfinite(values)
        vals = values[m]
        out[str(yr)] = {"n_events": int(vals.size),
                        "mean_net_bp": float(np.mean(vals)) if vals.size else None}
    return out
