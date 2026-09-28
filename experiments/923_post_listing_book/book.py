#!/usr/bin/env python3
"""TASK_923 — post-listing event book engine.

SHORT the new listing from its own entry, market-hedged at beta=1 against the SAME
equal-weight PIT market P3's residual construction uses (`loader.market_residual_inputs`).
Direction is SHORT because P3 measured NEGATIVE market-residual returns in every age bucket
(new listings UNDER-perform; see `experiments/917_second_moment_and_event_instruments/
p3_result.json` `profile`, sha256 17dcf3cc424f81400ec9d210b788f8f5cf2b25b5fe3318efc1a247abb8e1e0c9,
commit 8f411d1).

One book-simulation code path serves every reading this row needs: the binding table, every
non-binding sensitivity (same-close entry, unhedged short, legacy21 fee), and the honesty
battery (placebo listing dates, leave-one-out) -- only the INPUTS (entry rule, cost params,
listing dates) change between calls, never the mechanics, per the task's "both through the
same book code" instruction.

Estimand: per-EVENT (listing) net markout in bp over a fixed exit horizon (trading-day row
offset from entry in the daily grid -- consistent with P3's own H=1-day forward-return
convention; no skip-NaN adjustment, disclosed). Clustering: listing cohort WEEK (D85 --
SE = event-level SD / sqrt(n_clusters), the same simplified formula used for the SPEC-time
MDE and the observed CI, for consistency between them; disclosed in SPEC.md).

No DB writes anywhere in this module (pure array arithmetic over panel arrays already loaded
by `loader.py`).
"""
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from trader.config import SEC_PER_DAY               # noqa: E402
from trader.backtest import BORROW_RATE_SCENARIOS    # noqa: E402  (916-verified, read-only)

# ---------------------------------------------------------------------------------------
# Pre-registered constants (SPEC.md is the source of truth; these mirror it exactly -- any
# edit here must be mirrored there and vice versa, checked by `run_book.py`'s sha256 pin).
# ---------------------------------------------------------------------------------------
EXIT_HORIZONS_FIXED = (5, 10, 21, 60)          # trading-day row offsets from entry
FEE_BASE_TAKER_BP = 80.0                       # 916-verified binding round-trip taker tier
FEE_LEGACY_BP = 21.0                           # NON-BINDING, record-keeping only (task file)
BORROW_ANNUAL_ALT_HIGH = BORROW_RATE_SCENARIOS["high"]["ALT"]   # ~87.6%/yr -- BINDING
BORROW_ANNUAL_ALT_LOW = BORROW_RATE_SCENARIOS["low"]["ALT"]     # ~43.8%/yr -- non-binding
BAND_BP = 100.0                                # pre-registered tradability band (net > +100bp)
N_FAMILY_EXITS = 5                             # 4 fixed horizons + 1 age-decay exit
ALPHA_FAMILY = 0.05
Z_MDE_80PWR_5PCT_TWOSIDED = 2.8016             # 1.9600 (alpha=0.05 two-sided) + 0.8416 (80% power)
# left-censoring exclusion: an asset whose first real print is the PANEL'S OWN grid-start row
# is not an observed listing event (917's own audit: only 2/398 assets on the include_dead=
# True panel "start near panel start", and excluding them left the P3 IC unchanged to 6
# decimals -- RESULTS.md:798-799). Excluded here on the SAME criterion (first row index == 0),
# reported explicitly rather than silently dropped.
EXCLUDE_FIRST_ROW_IDX = 0


def z_bonferroni(n_family=N_FAMILY_EXITS, alpha_family=ALPHA_FAMILY):
    """Two-sided normal quantile at the Bonferroni-adjusted per-test alpha."""
    from scipy.stats import norm
    alpha_test = alpha_family / n_family
    return float(norm.ppf(1.0 - alpha_test / 2.0))


# =========================================================================================
# Events
# =========================================================================================

def listing_events(first, last, exclude_first_row_idx=EXCLUDE_FIRST_ROW_IDX):
    """Every asset column with a real listing row, minus the left-censored panel-start
    exclusion. Returns (asset_cols (n,) int, n_excluded_left_censored int)."""
    first = np.asarray(first)
    last = np.asarray(last)
    listed = (first >= 0) & (last > first)
    excluded = listed & (first == exclude_first_row_idx)
    keep = listed & ~excluded
    return np.nonzero(keep)[0].astype(np.int64), int(excluded.sum())


def cohort_week_ids(day_ts, entry_idx):
    """Integer cohort-week id = floor(entry timestamp / 7 days) -- the clustering unit D85
    names ('listing cohort WEEK')."""
    entry_idx = np.asarray(entry_idx)
    ts = np.asarray(day_ts)[entry_idx]
    return (ts // (7 * SEC_PER_DAY)).astype(np.int64)


def entry_indices(first, rule):
    """rule='same_close' -> the first PIT-tradable bar itself (non-binding sensitivity);
    rule='t1' -> the close of the bar AFTER it (binding, 918-lesson execution timing)."""
    first = np.asarray(first)
    if rule == "same_close":
        return first.copy()
    if rule == "t1":
        return first + 1
    raise ValueError(f"unknown entry rule: {rule!r}")


# =========================================================================================
# Per-event returns
# =========================================================================================

def compute_event_returns(mkt_inputs, day_ts, asset_cols, entry_idx, horizon_days):
    """For each event k: exit_idx = entry_idx[k] + horizon_days. Requires both endpoints
    finite AND every intervening daily market-return finite (so the hedge leg is fully
    computable -- an event with any NaN day in its window is dropped, not silently
    zero-filled). Returns dict of (n,) arrays: exit_idx, valid (bool), asset_log_ret,
    mkt_log_ret, residual (= asset_log_ret - mkt_log_ret, the market-neutral LONG-asset
    return; the short+hedge book's gross return is -residual), hold_days (calendar days,
    for the borrow charge)."""
    logc = mkt_inputs["logc"]
    mkt = mkt_inputs["mkt"]
    last = mkt_inputs["last"]
    T = logc.shape[0]
    asset_cols = np.asarray(asset_cols, dtype=np.int64)
    entry_idx = np.asarray(entry_idx, dtype=np.int64)
    n = asset_cols.size
    exit_idx = entry_idx + int(horizon_days)
    valid = (entry_idx >= 0) & (exit_idx < T) & (exit_idx <= last[asset_cols])
    asset_log_ret = np.full(n, np.nan)
    mkt_log_ret = np.full(n, np.nan)
    hold_days = np.full(n, np.nan)
    day_ts = np.asarray(day_ts)
    for k in range(n):
        if not valid[k]:
            continue
        i = int(asset_cols[k])
        e = int(entry_idx[k])
        x = int(exit_idx[k])
        a0, a1 = logc[e, i], logc[x, i]
        if not (np.isfinite(a0) and np.isfinite(a1)):
            valid[k] = False
            continue
        window_mkt = mkt[e + 1:x + 1]
        if window_mkt.size == 0 or not np.all(np.isfinite(window_mkt)):
            valid[k] = False
            continue
        asset_log_ret[k] = a1 - a0
        mkt_log_ret[k] = float(np.sum(window_mkt))
        hold_days[k] = float(day_ts[x] - day_ts[e]) / SEC_PER_DAY
    residual = asset_log_ret - mkt_log_ret
    return {"exit_idx": exit_idx, "valid": valid, "asset_log_ret": asset_log_ret,
            "mkt_log_ret": mkt_log_ret, "residual": residual, "hold_days": hold_days}


# =========================================================================================
# Costs
# =========================================================================================

def net_markout_bp(ev, hedge=True, fee_bp=FEE_BASE_TAKER_BP, borrow_annual=BORROW_ANNUAL_ALT_HIGH,
                   notional=1.0):
    """Per-event net markout in bp. gross = -(residual) if hedged (short + long-market hedge,
    beta=1) else -(asset_log_ret) (naked short). Costs: `fee_bp` round-trip taker charged on
    the short leg, AND AGAIN on the hedge leg if hedged (both legs really trade); borrow on
    the short notional for the full holding period (916 model, D81(c) denominator -- per
    notional per unit time, independent of the fee/turnover cost). Every quantity is finite
    only where `ev['valid']` is True elsewhere NaN (propagates, never silently zeroed)."""
    residual = ev["residual"]
    asset_log_ret = ev["asset_log_ret"]
    hold_days = ev["hold_days"]
    gross_bp = np.where(ev["valid"], -(residual if hedge else asset_log_ret) * 1e4, np.nan)
    n_legs = 2 if hedge else 1
    fee_total_bp = fee_bp * n_legs
    borrow_bp = np.where(ev["valid"], notional * borrow_annual * (hold_days / 365.0) * 1e4,
                         np.nan)
    net_bp = gross_bp - fee_total_bp - borrow_bp
    return {"gross_bp": gross_bp, "fee_total_bp": fee_total_bp, "borrow_bp": borrow_bp,
            "net_bp": net_bp}


# =========================================================================================
# Clustered stats / power
# =========================================================================================

def clustered_mean_se(values, cohort_ids):
    """D85's simplified clustering formula: mean over EVENTS (equal-weighted), dispersion =
    event-level SD, SE = event-level SD / sqrt(n_clusters) -- the same formula used both at
    SPEC time (for the pre-registered MDE) and here (for the observed CI), so the two are
    directly comparable. Degenerate guards: n=0 -> all-NaN dict with n_events=0; n=1 or
    n_clusters=1 -> SD/SE reported as NaN (never divide-by-zero, never a silently fabricated
    number) with a `degenerate` flag."""
    values = np.asarray(values, dtype=np.float64)
    cohort_ids = np.asarray(cohort_ids)
    finite = np.isfinite(values)
    v = values[finite]
    c = cohort_ids[finite]
    n = int(v.size)
    if n == 0:
        return {"n_events": 0, "n_clusters": 0, "mean": float("nan"), "sd": float("nan"),
                "se": float("nan"), "degenerate": True}
    mean = float(np.mean(v))
    n_clusters = int(np.unique(c).size)
    if n < 2 or n_clusters < 2:
        return {"n_events": n, "n_clusters": n_clusters, "mean": mean, "sd": float("nan"),
                "se": float("nan"), "degenerate": True}
    sd = float(np.std(v, ddof=1))
    se = sd / np.sqrt(n_clusters)
    return {"n_events": n, "n_clusters": n_clusters, "mean": mean, "sd": sd, "se": se,
            "degenerate": False}


def mde(se, z=Z_MDE_80PWR_5PCT_TWOSIDED):
    if not np.isfinite(se):
        return float("nan")
    return float(z * se)


def ci_bonferroni(stat, z=None):
    """stat = clustered_mean_se(...) output. Returns (lo, hi) at the Bonferroni-adjusted
    family-wise 95% level (N_FAMILY_EXITS exits), or (nan, nan) if degenerate."""
    if z is None:
        z = z_bonferroni()
    if stat["degenerate"] or not np.isfinite(stat["se"]):
        return float("nan"), float("nan")
    return stat["mean"] - z * stat["se"], stat["mean"] + z * stat["se"]
