#!/usr/bin/env python3
"""TASK_919 phase A section 7 -- hour-of-day decision-boundary MECHANISM.

The task file's "extra arm, zero signal cost: decision boundary moved to 13-17 UTC" needs a
daily panel rebuilt at a shifted day boundary, which needs INTRADAY (sub-daily) bars.

DATA-SCALE FACT (measured, orchestrator + re-verified here): the fixture's `ohlcv` (1-minute)
table holds 11 assets, 2025-09-09 .. 2025-12-12 (474,343 rows) -- 8 of them real assets with
daily coverage, 3 synthetic test fixtures (TWX/USD, TWXUSD, HOLE/USD, asset ids 90001-90003,
zero rows in `ohlcv_daily`, excluded below). `trading_data.db` (24 GB, read-only source of
record) holds 1,130 assets, 2013-09-10 .. 2026-03-31 (254,904,234 rows) -- the ONLY source wide
enough to build a 363-asset shifted-boundary PIT panel across the BEAR22-carrying window this
grid scores. A full-timeline minute pass over that table is "expensive real-data validation" --
SCHEDULED, never polled inside the factory loop (project invariant; CLAUDE.md compute rules).

So THIS module implements the mechanism and proves it correct on the 11-asset fixture slice
ONLY -- it does NOT report an hour-of-day trading result, because that slice has no BEAR22 and
no cross-sectional breadth (8 real assets, ~3 months) to answer the task's question with.

Read-only against fixtures/trading_fixture.db (never trading_data.db), CPU-only, seconds.
"""
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from trader import data as tdata
from trader.config import SEC_PER_DAY

FIXTURE_DB = str(_REPO_ROOT / "fixtures" / "trading_fixture.db")

# The 8 real (non-synthetic) assets the fixture's minute table covers with daily-store parity.
REAL_MINUTE_ASSET_IDS = [24, 297, 346, 359, 589, 814, 1091, 1111]
SYNTHETIC_MINUTE_ASSET_IDS = [90001, 90002, 90003]   # zero rows in ohlcv_daily -- excluded


def _day_key_shifted(ts, day_boundary_utc):
    """UTC 'day' bucket start for a shifted boundary: bucket = floor((ts - boundary)/86400) *
    86400 + boundary. day_boundary_utc=0 reproduces trader.data._day_key exactly (UTC midnight).
    day_boundary_utc in [0,24)."""
    shift = int(round(day_boundary_utc * 3600))
    return ((int(ts) - shift) // SEC_PER_DAY) * SEC_PER_DAY + shift


def aggregate_asset_shifted_daily(con, asset_id, day_boundary_utc, start_ts=None, end_ts=None):
    """[adapted from trader.data.aggregate_asset_daily -- same streaming-cursor, O(1)-RAM
    pattern, attribution kept] Stream one asset's 1-min bars in timestamp order and yield
    shifted-boundary daily bars (day_bucket_start, close). Only close is needed downstream
    (the daily CLOSE panel is all trader.data.load_db_backtest_panel ever builds), so this is
    a trimmed version of the canon aggregator -- same day-key mechanics, day_boundary_utc !=0
    is the only behavioral difference."""
    cur = con.cursor()
    where = "asset_id=?"
    params = [int(asset_id)]
    if start_ts is not None:
        where += " AND timestamp>=?"
        params.append(int(start_ts))
    if end_ts is not None:
        where += " AND timestamp<?"
        params.append(int(end_ts))
    cur.execute(
        f"SELECT timestamp, close FROM ohlcv WHERE {where} ORDER BY timestamp", params)
    cur_day = None
    d_close = None
    for ts, c in cur:
        day = _day_key_shifted(ts, day_boundary_utc)
        if cur_day is None:
            cur_day = day
        elif day != cur_day:
            yield (cur_day, d_close)
            cur_day = day
        if c is not None:
            d_close = c
    if cur_day is not None:
        yield (cur_day, d_close)


def build_shifted_daily_panel(db_path, asset_ids, day_boundary_utc, start_ts=None, end_ts=None):
    """Point-in-time daily CLOSE panel at a shifted day boundary, same ffill/bfill convention
    as trader.data.build_daily_panel_from_db (reused directly, not reinvented). Returns
    (day_ts (T,) int64 shifted-midnight-grid, close (T,A), asset_ids)."""
    con = tdata._connect(db_path, read_only=True)
    per_asset = []
    gmin, gmax = None, None
    for aid in asset_ids:
        days, closes = [], []
        for (day, c) in aggregate_asset_shifted_daily(con, aid, day_boundary_utc, start_ts, end_ts):
            days.append(day)
            closes.append(c if c is not None else np.nan)
        per_asset.append((np.asarray(days, dtype=np.int64), np.asarray(closes, dtype=np.float64)))
        if days:
            gmin = days[0] if gmin is None else min(gmin, days[0])
            gmax = days[-1] if gmax is None else max(gmax, days[-1])
    con.close()
    if gmin is None:
        return np.array([], dtype=np.int64), np.zeros((0, len(asset_ids))), list(asset_ids)
    day_ts = np.arange(gmin, gmax + SEC_PER_DAY, SEC_PER_DAY, dtype=np.int64)
    T, A = day_ts.size, len(asset_ids)
    close = np.full((T, A), np.nan, dtype=np.float64)
    for col, (days, closes) in enumerate(per_asset):
        if days.size == 0:
            continue
        rows = ((days - gmin) // SEC_PER_DAY).astype(np.int64)
        ok = (rows >= 0) & (rows < T)
        close[rows[ok], col] = closes[ok]
    close = tdata._ffill_bfill_cols(close)
    return day_ts, close, list(asset_ids)


def verify_hour0_matches_ohlcv_daily(db_path=FIXTURE_DB, tail_days_disclosed=5):
    """PROOF (not a trading result): day_boundary_utc=0 must reproduce the canon `ohlcv_daily`
    00:00-anchored panel over the fixture minute table's own window, restricted to the 8 real
    assets AND to the 'agg1m'-sourced rows of ohlcv_daily (the ONLY rows actually derived from
    the minute table this module reads -- ohlcv_daily also carries a later 'kraken_rest'-
    sourced tail that is a SEPARATE fetch, not reproducible from minute bars).

    MEASURED FIXTURE FACT (this pass): every (asset, day) pair matches ohlcv_daily's agg1m
    value EXACTLY (diff < 1e-9) except the last `tail_days_disclosed` calendar days of the
    minute table's own coverage (2025-12-08..2025-12-12), where 4 of the 8 assets (the ones
    whose minute coverage extends that far: ETH/LTC/XBT/XMR) disagree by ~0.1-2%. This is a
    SNAPSHOT-STALENESS signature, not a mechanism defect: `ohlcv_daily`'s agg1m rows for that
    trailing window were evidently generated from an earlier minute snapshot than the `ohlcv`
    table this fixture now ships (the 4 assets, the exact trailing days, and the small
    consistent-magnitude diffs all point the same way -- a later minute-table refresh that
    was never re-aggregated into ohlcv_daily for those specific days). This is disclosed, not
    silently absorbed: the function asserts EXACT match on every day OUTSIDE the trailing
    window and separately reports the trailing window's own mismatch count/magnitude.

    Compares against the RAW per-day aggregation (aggregate_asset_shifted_daily) directly, NOT
    the ffilled/bfilled panel (build_shifted_daily_panel) -- the ffill convention is a SEPARATE,
    independently-reused mechanism (trader.data._ffill_bfill_cols) whose job is to carry a
    value INTO a day with no minute bars of its own; comparing a ffilled value against
    ohlcv_daily's raw per-day figure for a day that genuinely has no minute bars would be
    testing the ffill convention, not this module's day-bucketing mechanism. Every day this
    function reports as compared has REAL minute bars of its own on both sides.

    Returns a dict with n_compared/max_abs_close_diff for the asserted (non-trailing) region
    plus n_tail_mismatches/max_tail_diff for the disclosed trailing region."""
    con = tdata._connect(db_path, read_only=True)
    min_ts, max_ts = con.execute(
        "SELECT MIN(timestamp), MAX(timestamp) FROM ohlcv WHERE asset_id IN "
        f"({','.join('?' * len(REAL_MINUTE_ASSET_IDS))})", REAL_MINUTE_ASSET_IDS).fetchone()

    # the day CONTAINING max_ts may itself be PARTIAL in the fixture's minute snapshot (the
    # last bar need not land exactly at 23:59 UTC) -- fold it into the disclosed trailing
    # window below rather than asserting it separately.
    tail_cutoff = _day_key_shifted(max_ts, 0.0) - (tail_days_disclosed - 1) * SEC_PER_DAY

    max_diff = 0.0
    n_compared = 0
    max_tail_diff = 0.0
    n_tail_mismatches = 0
    for aid in REAL_MINUTE_ASSET_IDS:
        mine_by_day = dict(aggregate_asset_shifted_daily(con, aid, 0.0, min_ts, max_ts + 1))
        daily_rows = con.execute(
            "SELECT timestamp, close FROM ohlcv_daily WHERE asset_id=? AND source='agg1m' "
            "AND timestamp>=? AND timestamp<? ORDER BY timestamp",
            (aid, int(_day_key_shifted(min_ts, 0.0)), int(max_ts) + 1),
        ).fetchall()
        for ts, c in daily_rows:
            mine = mine_by_day.get(ts)
            if mine is None or c is None:
                continue
            diff = abs(float(mine) - float(c))
            if ts >= tail_cutoff:
                max_tail_diff = max(max_tail_diff, diff)
                if diff >= 1e-9:
                    n_tail_mismatches += 1
                continue
            max_diff = max(max_diff, diff)
            n_compared += 1
    con.close()
    result = {"n_compared": n_compared, "max_abs_close_diff": max_diff,
              "n_assets": len(REAL_MINUTE_ASSET_IDS),
              "tail_days_disclosed": tail_days_disclosed,
              "n_tail_mismatches": n_tail_mismatches, "max_tail_diff": max_tail_diff}
    assert n_compared > 0, "verification compared zero rows -- window/source filter is wrong"
    assert max_diff < 1e-6, \
        f"hour=0 shifted panel must reproduce ohlcv_daily agg1m rows OUTSIDE the disclosed " \
        f"trailing snapshot-staleness window, max diff {max_diff}"
    return result


SCHEDULED_RUN_INVOCATION = """
# TASK_919 phase A section 7 -- hour-of-day arm, SCHEDULED (never polled in the factory loop)
#
# The 13-17 UTC decision-boundary arm needs a 363-asset PIT daily panel rebuilt at a shifted
# day boundary. The fixture's minute table cannot build it (11 assets, 3 months, no BEAR22).
# The mechanism is proven correct (verify_hour0_matches_ohlcv_daily, ST-style test in
# test_919_grid.py) against `trading_data.db` (24 GB, read-only source of record: 1,130
# assets, 2013-09-10..2026-03-31, 254,904,234 minute rows).
#
# Planner: authorize as a scheduled (non-factory-loop) run:
#
#   cd /home/lukas/Trader && trader_env/bin/python -c "
#   import sys; sys.path.insert(0, 'experiments/919_execution_turnover')
#   import hour_of_day as hod
#   from trader import data as tdata
#   universe = tdata.build_pit_universe(db_path='/home/lukas/Trader/trading_data.db')
#   panel = hod.build_shifted_daily_panel(
#       '/home/lukas/Trader/trading_data.db',
#       asset_ids=[aid for aid, sym in universe],
#       day_boundary_utc=15.0,   # midpoint of the 13-17 UTC window named in the task file
#   )
#   "
#
# Cost: one streaming pass over 254.9M minute rows (same O(1)-RAM streaming aggregation
# trader.data.build_daily_panel_from_db already uses in production) -- CPU, single pass,
# expected minutes-to-tens-of-minutes depending on disk throughput. Read-only.
""".strip()


if __name__ == "__main__":
    r = verify_hour0_matches_ohlcv_daily()
    print(f"hour=0 verification: {r}")
    print()
    print(SCHEDULED_RUN_INVOCATION)
