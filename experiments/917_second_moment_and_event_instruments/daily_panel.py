#!/usr/bin/env python3
"""TASK_917 shared data layer — raw (un-ffilled) PIT daily OHLCV panel.

`trader.data.read_ohlcv_daily` / `load_db_backtest_panel` return a CLOSE-only panel with the
819 ffill/leading-bfill convention applied (real gaps and post-listing/pre-listing rows are
fabricated forward). That convention is correct for a TRADING backtest (a strategy needs SOME
price to mark a position against) but wrong for the second-moment / event instruments in this
pack: a variance-ratio or a shock-return computed across a fabricated (zero-return) gap is
biased toward the random-walk null (VR->1) or silently invents a "shock" of zero. So this
module re-reads `ohlcv_daily` RAW (NaN on any day the store has no row for that asset — no
ffill, no bfill) and adds open/high/low/volume, which the canon close-only reader drops.

Universe selection (USD-quoted, non-stable/fiat, kraken_rest-covered by default, or
+dead/delisted with include_dead=True) is NOT re-derived here — it delegates to
`trader.data.load_db_backtest_panel` for the id/symbol list and the day-grid bounds, so the
PIT universe membership is identical to the canon backtest path (819/820 convention); this
module only changes how PRICES inside that universe are read (raw vs ffilled) and adds O/H/L/V.

Read-only: uses `mcts_trader.utils.db._ro_connect` (mode=ro) on whichever db_path is passed —
never writes, never DDLs. Callers point this at `fixtures/trading_fixture.db` (verified below
to carry the FULL `ohlcv_daily` table, byte-identical to `trading_data.db` for this table — see
TASK_917 report) or at `trading_data.db` directly.
"""
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from trader import data as tdata
from trader.config import SEC_PER_DAY
from mcts_trader.utils.db import _ro_connect


def load_raw_pit_panel(db_path, include_dead=True, start_ts=None, end_ts=None,
                       source_filter="kraken_rest"):
    """Raw PIT daily OHLCV panel, no fill.

    Returns dict with:
      day_ts    (T,) int64 UTC-midnight grid
      open/high/low/close/volume  (T, A) float64, NaN where `ohlcv_daily` has no row
      ids       [A] asset_id
      syms      [A] base symbol (matches trader.data's convention: base only, quote stripped)
      info      the `load_db_backtest_panel` skip-accounting dict

    include_dead=True (default here — P1/P3 pre-registration explicitly wants dead coins
    retained; P2 defaults the same for consistency, callers can override per-instrument).
    """
    day_ts, _close_filled, ids, syms, info = tdata.load_db_backtest_panel(
        db_path=db_path, start_ts=start_ts, end_ts=end_ts,
        source_filter=source_filter, include_dead=include_dead)
    T = day_ts.size
    A = len(ids)
    if T == 0 or A == 0:
        empty = np.zeros((0, 0), dtype=np.float64)
        return {"day_ts": day_ts, "open": empty, "high": empty, "low": empty,
                "close": empty, "volume": empty, "ids": ids, "syms": syms, "info": info}
    gmin, gmax = int(day_ts[0]), int(day_ts[-1])
    o = np.full((T, A), np.nan, dtype=np.float64)
    h = np.full((T, A), np.nan, dtype=np.float64)
    l = np.full((T, A), np.nan, dtype=np.float64)
    c = np.full((T, A), np.nan, dtype=np.float64)
    v = np.full((T, A), np.nan, dtype=np.float64)
    con = _ro_connect(db_path)
    cur = con.cursor()
    for col, aid in enumerate(ids):
        cur.execute(
            "SELECT timestamp, open, high, low, close, volume FROM ohlcv_daily "
            "WHERE asset_id=? AND timestamp>=? AND timestamp<=? ORDER BY timestamp",
            (int(aid), gmin, gmax),
        )
        rows = cur.fetchall()
        if not rows:
            continue
        ts = np.asarray([r[0] for r in rows], dtype=np.int64)
        rowidx = ((ts - gmin) // SEC_PER_DAY).astype(np.int64)
        ok = (rowidx >= 0) & (rowidx < T)
        rowidx = rowidx[ok]
        vals = np.asarray(rows, dtype=np.float64)[ok]
        o[rowidx, col] = vals[:, 1]
        h[rowidx, col] = vals[:, 2]
        l[rowidx, col] = vals[:, 3]
        c[rowidx, col] = vals[:, 4]
        v[rowidx, col] = vals[:, 5]
    con.close()
    return {"day_ts": day_ts, "open": o, "high": h, "low": l, "close": c, "volume": v,
            "ids": ids, "syms": syms, "info": info}


def daily_log_return(close_raw):
    """(T,A) log return between CONSECUTIVE grid rows. NaN wherever either endpoint is
    missing (raw panel — a missing day is NOT bridged, so a return spanning a real gap in the
    store is NaN, never a fabricated multi-day return silently treated as one day)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        logc = np.log(np.where(close_raw > 0, close_raw, np.nan))
    r = np.full_like(logc, np.nan)
    r[1:] = logc[1:] - logc[:-1]
    return r, logc


def qday_return_nogap(logc, dr_finite, q):
    """(T,A) q-DAY compounded log return logc[t]-logc[t-q], valid ONLY where every one of the
    q intermediate daily returns is finite (no gap bridged). `dr_finite` = np.isfinite(daily
    log return) from `daily_log_return`. Implemented via a cumulative bad-day counter so a
    single missing day inside the window invalidates every q-return that spans it."""
    T, A = logc.shape
    bad = (~dr_finite).astype(np.int64)          # (T,A), row0 always "bad" (no return there)
    csum = np.cumsum(bad, axis=0)
    out = np.full((T, A), np.nan)
    if T <= q:
        return out
    span_bad = csum[q:] - csum[:-q]               # bad daily-return count strictly inside (t-q, t]
    endpoints_ok = np.isfinite(logc[q:]) & np.isfinite(logc[:-q])
    valid = (span_bad == 0) & endpoints_ok
    diff = logc[q:] - logc[:-q]
    out[q:] = np.where(valid, diff, np.nan)
    return out
