#!/usr/bin/env python3
"""TASK_820 data layer — daily close panels for the trend CTA.

Three sources:
  (a) load_npz_panel        — the cached 819 broad daily universe (gate 1a determinism ref).
  (b) build_daily_panel_from_db — POINT-IN-TIME daily panel from the 1-min `ohlcv` table,
        aggregated to UTC-midnight daily bars via STREAMING SQL (iterate the cursor, bucket
        by day; the (asset_id,timestamp) PK makes per-asset range scans cheap). Never loads
        the 246M-row table into RAM.
  (c) read_ohlcv_daily      — read the `ohlcv_daily` daily-bar store (populated by the separate
        tools/fetch_daily_kraken.py fetcher). Tolerates the table not existing yet.

update_data() writes aggregated daily bars into `ohlcv_daily` (source='agg1m'), idempotent
(INSERT OR REPLACE on the (asset_id,timestamp) PK).

Gap-fill (ffill then leading bfill) reproduces the audited 819/806 convention: carry the last
real bar forward, back-fill the leading gap with the first real bar — same-series past/adjacent
bars only, no cross-time leak. Listing = first real (nonzero-return) bar (strategy.first_active).
"""

import sqlite3

import numpy as np

from .config import DB_PATH, NPZ_CACHE, SEC_PER_DAY, STABLE_FIAT

# The daily fetcher (tools/fetch_daily_kraken.py) may hold write transactions on the DB while
# we read; give every connection a generous busy timeout instead of failing on a brief lock.
SQLITE_TIMEOUT_S = 60


def _connect(db_path, read_only=False):
    """sqlite connection with busy-timeout resilience (fetcher may be mid-insert)."""
    if read_only:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=SQLITE_TIMEOUT_S)
    else:
        con = sqlite3.connect(db_path, timeout=SQLITE_TIMEOUT_S)
    return con


# =============================================================================
# (a) cached npz panel — gate 1a
# =============================================================================

def load_npz_panel(path=NPZ_CACHE):
    """Load the cached 819 broad daily close panel. Returns (close_d (T,A) float64,
    ts_d (T,) int64, syms list). ts_d is the 819-frozen daily grid (sampled at the 816
    minute-grid offset, not midnight) — used only for gate-1a parity."""
    z = np.load(str(path))
    close_d = np.ascontiguousarray(z["close_d"]).astype(np.float64)
    ts_d = np.ascontiguousarray(z["ts_d"]).astype(np.int64)
    syms = [str(s) for s in z["syms"]]
    return close_d, ts_d, syms


# =============================================================================
# liquidity filter — port of 819 portfolio.liquidity_filter
# =============================================================================

def liquidity_filter(close_d):
    """[copied from runs/819 portfolio.liquidity_filter] Active-window zero-daily-return
    fraction per asset (filters stablecoins / gappy series). Returns (zerofrac, active_days);
    zerofrac measured between an asset's first and last real return (PIT within the panel)."""
    T, A = close_d.shape
    with np.errstate(divide="ignore", invalid="ignore"):
        logc = np.log(np.where(close_d > 0, close_d, np.nan))
    r = np.diff(logc, axis=0)
    zf = np.full(A, np.nan)
    act = np.zeros(A, dtype=int)
    for i in range(A):
        nz = np.where(np.abs(r[:, i]) > 1e-12)[0]
        if nz.size < 2:
            zf[i] = 1.0
            continue
        lo, hi = nz[0], nz[-1]
        seg = r[lo:hi + 1, i]
        zf[i] = float(np.mean(np.abs(seg) <= 1e-12))
        act[i] = hi - lo + 1
    return zf, act


def apply_liquidity_filter(close_d, syms, zf_max=0.15, min_active=365):
    """Apply the 819 liquidity gate: keep assets with zero-return frac < zf_max AND
    active_days >= min_active. Returns (close_kept, syms_kept, dropped, keep_mask)."""
    zf, act = liquidity_filter(close_d)
    keep = (zf < zf_max) & (act >= min_active)
    close_kept = np.ascontiguousarray(close_d[:, keep])
    syms_kept = [syms[i] for i in range(len(syms)) if keep[i]]
    dropped = [syms[i] for i in range(len(syms)) if not keep[i]]
    return close_kept, syms_kept, dropped, keep


# =============================================================================
# gap fill — vectorized ffill + leading bfill (819 convention)
# =============================================================================

def _ffill_bfill_cols(a):
    """[copied from runs/819 portfolio._ffill_bfill_cols] Forward-fill then leading back-fill
    per column. Uses only same-series past/adjacent bars (no cross-time leak)."""
    T, A = a.shape
    out = a.copy()
    valid = ~np.isnan(out)
    idx = np.where(valid, np.arange(T)[:, None], 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    out = out[idx, np.arange(A)[None, :]]
    for col in range(A):
        vr = np.where(valid[:, col])[0]
        if vr.size and vr[0] > 0:
            out[:vr[0], col] = a[vr[0], col]
        elif vr.size == 0:
            out[:, col] = np.nan
    return out


# =============================================================================
# (b) streaming 1-min -> UTC-midnight daily aggregation
# =============================================================================

def _day_key(ts):
    """UTC-midnight epoch for a 1-min bar timestamp."""
    return (int(ts) // SEC_PER_DAY) * SEC_PER_DAY


def aggregate_asset_daily(con, asset_id, start_ts=None, end_ts=None):
    """Stream one asset's 1-min bars in timestamp order and yield UTC-midnight daily bars
    (day_ts, open, high, low, close, volume, trade_count). GROUP-BY-over-cursor in Python:
    the cursor is iterated lazily (sqlite fetches in batches) so RAM stays O(1) per asset,
    never the full row set. Daily open/close are the first/last 1-min bars of the day."""
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
        f"SELECT timestamp, open, high, low, close, volume, trade_count "
        f"FROM ohlcv WHERE {where} ORDER BY timestamp",
        params,
    )
    cur_day = None
    d_open = d_high = d_low = d_close = None
    d_vol = 0.0
    d_tc = 0
    for ts, o, h, lo, c, v, tc in cur:
        day = _day_key(ts)
        if cur_day is None:
            cur_day = day
            d_open, d_high, d_low, d_close, d_vol, d_tc = o, h, lo, c, 0.0, 0
        elif day != cur_day:
            yield (cur_day, d_open, d_high, d_low, d_close, d_vol, d_tc)
            cur_day = day
            d_open, d_high, d_low, d_close, d_vol, d_tc = o, h, lo, c, 0.0, 0
        # accumulate within the day
        if h is not None:
            d_high = h if d_high is None else max(d_high, h)
        if lo is not None:
            d_low = lo if d_low is None else min(d_low, lo)
        if c is not None:
            d_close = c                          # last close of the day
        d_vol += (v or 0.0)
        d_tc += (tc or 0)
    if cur_day is not None:
        yield (cur_day, d_open, d_high, d_low, d_close, d_vol, d_tc)


def build_daily_panel_from_db(db_path=DB_PATH, asset_ids=None, start_ts=None, end_ts=None,
                              syms=None):
    """Point-in-time daily CLOSE panel from the 1-min table via streaming aggregation.
    Returns (day_ts (T,) int64 UTC-midnight, close_panel (T, A) float64, syms/ids used).
    Each asset is aggregated to daily (streaming), then placed on a regular midnight day grid
    and ffill/bfill'd per the 819 convention. Only the daily arrays live in RAM."""
    if asset_ids is None:
        raise ValueError("asset_ids required (use build_pit_universe to discover)")
    con = _connect(db_path)
    per_asset = []
    gmin, gmax = None, None
    for aid in asset_ids:
        days, closes = [], []
        for (day, o, h, lo, c, v, tc) in aggregate_asset_daily(con, aid, start_ts, end_ts):
            days.append(day)
            closes.append(c if c is not None else np.nan)
        per_asset.append((np.asarray(days, dtype=np.int64), np.asarray(closes, dtype=np.float64)))
        if days:
            gmin = days[0] if gmin is None else min(gmin, days[0])
            gmax = days[-1] if gmax is None else max(gmax, days[-1])
    con.close()
    if gmin is None:
        return np.array([], dtype=np.int64), np.zeros((0, len(asset_ids))), (syms or asset_ids)
    day_ts = np.arange(gmin, gmax + SEC_PER_DAY, SEC_PER_DAY, dtype=np.int64)
    T = day_ts.size
    A = len(asset_ids)
    close = np.full((T, A), np.nan, dtype=np.float64)
    for col, (days, closes) in enumerate(per_asset):
        if days.size == 0:
            continue
        rows = ((days - gmin) // SEC_PER_DAY).astype(np.int64)
        ok = (rows >= 0) & (rows < T)
        close[rows[ok], col] = closes[ok]
    close = _ffill_bfill_cols(close)
    return day_ts, close, (syms or list(asset_ids))


# =============================================================================
# (c) ohlcv_daily store — reader + writer (tolerate the table not existing yet)
# =============================================================================

_OHLCV_DAILY_DDL = """
CREATE TABLE IF NOT EXISTS ohlcv_daily (
    asset_id    INTEGER NOT NULL,
    timestamp   INTEGER NOT NULL,
    open        REAL,
    high        REAL,
    low         REAL,
    close       REAL,
    volume      REAL,
    trade_count INTEGER,
    source      TEXT,
    PRIMARY KEY (asset_id, timestamp)
)
"""


def ohlcv_daily_exists(db_path=DB_PATH):
    con = _connect(db_path)
    try:
        r = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='ohlcv_daily'"
        ).fetchone()
        return r is not None
    finally:
        con.close()


def create_ohlcv_daily(con):
    con.execute(_OHLCV_DAILY_DDL)
    con.commit()


def read_ohlcv_daily(db_path=DB_PATH, asset_ids=None, start_ts=None, end_ts=None):
    """Read daily bars from `ohlcv_daily` into a close panel. Returns
    (day_ts, close_panel (T,A), asset_ids). Raises RuntimeError with a clear message if the
    table does not exist yet (the fetcher / update-data has not populated it)."""
    if not ohlcv_daily_exists(db_path):
        raise RuntimeError(
            "ohlcv_daily table does not exist yet — run `python -m trader update-data "
            "--asset-ids ...` or wait for tools/fetch_daily_kraken.py to populate it."
        )
    con = _connect(db_path)
    cur = con.cursor()
    if asset_ids is None:
        asset_ids = [r[0] for r in cur.execute(
            "SELECT DISTINCT asset_id FROM ohlcv_daily ORDER BY asset_id")]
    per_asset = []
    gmin, gmax = None, None
    for aid in asset_ids:
        where = "asset_id=?"
        params = [int(aid)]
        if start_ts is not None:
            where += " AND timestamp>=?"
            params.append(int(start_ts))
        if end_ts is not None:
            where += " AND timestamp<?"
            params.append(int(end_ts))
        rows = cur.execute(
            f"SELECT timestamp, close FROM ohlcv_daily WHERE {where} ORDER BY timestamp",
            params,
        ).fetchall()
        days = np.asarray([r[0] for r in rows], dtype=np.int64)
        closes = np.asarray([r[1] for r in rows], dtype=np.float64)
        per_asset.append((days, closes))
        if days.size:
            gmin = days[0] if gmin is None else min(gmin, days[0])
            gmax = days[-1] if gmax is None else max(gmax, days[-1])
    con.close()
    if gmin is None:
        return np.array([], dtype=np.int64), np.zeros((0, len(asset_ids))), list(asset_ids)
    day_ts = np.arange(gmin, gmax + SEC_PER_DAY, SEC_PER_DAY, dtype=np.int64)
    T = day_ts.size
    A = len(asset_ids)
    close = np.full((T, A), np.nan, dtype=np.float64)
    for col, (days, closes) in enumerate(per_asset):
        if days.size == 0:
            continue
        r = ((days - gmin) // SEC_PER_DAY).astype(np.int64)
        ok = (r >= 0) & (r < T)
        close[r[ok], col] = closes[ok]
    close = _ffill_bfill_cols(close)
    return day_ts, close, list(asset_ids)


def load_db_backtest_panel(db_path=DB_PATH, start_ts=None, end_ts=None,
                           source_filter="kraken_rest"):
    """Backtest/gate-2 panel from `ohlcv_daily`.

    UNIVERSE = assets that `source_filter` covers (default 'kraken_rest' — assets present
    only via 'agg1m', e.g. the aggregation smokes or REST-delisted pairs, are skipped
    cleanly and counted), then filtered to USD-quoted non-stablecoin/fiat via `assets`.
    ROWS are read unfiltered: the store is canonical — the (asset_id,timestamp) PK holds
    exactly one bar per (asset, day) regardless of which source wrote it last, and the
    fetch contract cross-checks agg1m vs kraken_rest closes on overlap.

    Returns (day_ts, close (T,A), asset_ids, syms, info) where info counts what was
    skipped. Columns with no rows in [start_ts, end_ts) are trimmed. PIT membership
    (listing/delisting) is handled downstream by the present-mask."""
    if not ohlcv_daily_exists(db_path):
        raise RuntimeError(
            "ohlcv_daily table does not exist yet — run `python -m trader update-data "
            "--asset-ids ...` or wait for tools/fetch_daily_kraken.py to populate it."
        )
    con = _connect(db_path, read_only=True)
    cur = con.cursor()
    all_ids = {r[0] for r in cur.execute("SELECT DISTINCT asset_id FROM ohlcv_daily")}
    if source_filter:
        src_ids = [r[0] for r in cur.execute(
            "SELECT DISTINCT asset_id FROM ohlcv_daily WHERE source=?", (source_filter,))]
    else:
        src_ids = sorted(all_ids)
    n_skipped_source = len(all_ids) - len(src_ids)
    sym_map = {i: s for (i, s) in cur.execute("SELECT id, symbol FROM assets")}
    con.close()

    keep_ids, keep_syms = [], []
    n_skipped_nonusd = 0
    for aid in sorted(src_ids):
        sym = sym_map.get(aid, "")
        if not sym or "/" not in sym:
            n_skipped_nonusd += 1
            continue
        base, quote = sym.split("/", 1)
        if quote.upper() != "USD" or base.upper() in STABLE_FIAT:
            n_skipped_nonusd += 1
            continue
        keep_ids.append(aid)
        keep_syms.append(base)

    day_ts, close, _ids = read_ohlcv_daily(db_path, keep_ids, start_ts, end_ts)
    # trim assets with zero rows in the window (all-NaN columns after fill)
    if close.size:
        has = np.isfinite(close).any(axis=0)
    else:
        has = np.zeros(len(keep_ids), dtype=bool)
    n_skipped_empty = int((~has).sum())
    close = np.ascontiguousarray(close[:, has])
    keep_ids = [i for i, h in zip(keep_ids, has) if h]
    keep_syms = [s for s, h in zip(keep_syms, has) if h]
    info = {"n_skipped_source": n_skipped_source, "n_skipped_nonusd": n_skipped_nonusd,
            "n_skipped_empty": n_skipped_empty, "source_filter": source_filter}
    return day_ts, close, keep_ids, keep_syms, info


def update_data(db_path=DB_PATH, asset_ids=None, start_ts=None, end_ts=None, source="agg1m"):
    """Aggregate 1-min -> UTC-midnight daily bars and write them to `ohlcv_daily` (streaming;
    INSERT OR REPLACE, idempotent). Returns {asset_id: n_days_written}. Requires an explicit
    asset_ids list (never a blind full-DB run)."""
    if not asset_ids:
        raise ValueError("update_data requires an explicit asset_ids list (no blind full run)")
    con = _connect(db_path)
    create_ohlcv_daily(con)
    written = {}
    # separate read/write connections would be cleaner, but INSERT after fully consuming each
    # asset's aggregation generator is safe on one connection.
    for aid in asset_ids:
        rows = list(aggregate_asset_daily(con, aid, start_ts, end_ts))
        if not rows:
            written[aid] = 0
            continue
        con.executemany(
            "INSERT OR REPLACE INTO ohlcv_daily "
            "(asset_id, timestamp, open, high, low, close, volume, trade_count, source) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            [(int(aid), int(d), o, h, lo, c, v, tc, source)
             for (d, o, h, lo, c, v, tc) in rows],
        )
        written[aid] = len(rows)
    con.commit()
    con.close()
    return written


# =============================================================================
# point-in-time universe discovery
# =============================================================================

def build_pit_universe(db_path=DB_PATH, min_bars=0):
    """Discover the USD-quoted, non-stablecoin, non-fiat crypto universe from `assets`.
    Returns [(asset_id, symbol)] sorted by asset_id. If min_bars>0, keep only assets with at
    least that many 1-min bars (a coverage floor). PIT membership (listing/delisting) is handled
    downstream by the present-mask, not here."""
    con = _connect(db_path)
    cur = con.cursor()
    out = []
    for aid, sym in cur.execute("SELECT id, symbol FROM assets"):
        if not sym or "/" not in sym:
            continue
        base, quote = sym.split("/", 1)
        if quote.upper() != "USD":
            continue
        if base.upper() in STABLE_FIAT:
            continue
        out.append((aid, sym))
    if min_bars > 0:
        kept = []
        for aid, sym in out:
            n = cur.execute("SELECT COUNT(*) FROM ohlcv WHERE asset_id=?", (aid,)).fetchone()[0]
            if n >= min_bars:
                kept.append((aid, sym))
        out = kept
    con.close()
    out.sort(key=lambda t: t[0])
    return out
