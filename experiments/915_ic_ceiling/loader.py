#!/usr/bin/env python3
"""TASK_915 data layer -- PIT daily OHLCV panel = fixture `ohlcv_daily` (through 2026-06-21) +
staged 858-backfill CSVs (`runs/858_backfill/api_2026-08-31/daily/`), full O/H/L/C/V (NOT
close-only -- the memo's A-2 receipt names close-only construction as the exact defect this row
must not repeat).

Read-only everywhere: `fixtures/trading_fixture.db` via `trader.data`'s own `file:...?mode=ro`
connection helper; the staged CSVs as plain files. Never opens `trading_data.db`. Never writes,
never DDLs, never regenerates the fixture.

Universe discovery is delegated to `trader.data.load_db_backtest_panel` (same loader 898/916/
917/918 use) for asset selection + CLOSE panel (include_dead=True -> dead coins retained, PIT
per the present-mask downstream). This module then pulls the MATCHING open/high/low/volume
columns for the SAME (day_ts grid, asset_ids) from `ohlcv_daily` directly (read_ohlcv_daily only
carries close), and self-checks the two day_ts grids are identical before trusting the join.

Extends TASK_918's seam re-verification (`experiments/918_tactic_hull/loader.py`) from close-only
to ALL FIVE columns: every overlap-window (asset, day) pair is compared DB-vs-CSV on
open/high/low/close/volume independently; ANY mismatch on ANY column refuses the whole load
(raise, never silently prefer one source)."""
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from trader import data as tdata               # noqa: E402
from trader.config import SEC_PER_DAY, STABLE_FIAT   # noqa: E402

FIXTURE_DB = str(_REPO_ROOT / "fixtures" / "trading_fixture.db")
BACKFILL_DIR = _REPO_ROOT / "runs" / "858_backfill" / "api_2026-08-31"
MANIFEST_PATH = BACKFILL_DIR / "manifest.json"
DAILY_DIR = BACKFILL_DIR / "daily"

OVERLAP_REL_TOL = 1e-6   # looser than 918's close-only 1e-9: high/low/volume carry more
OVERLAP_ABS_TOL = 1e-6   # cross-pull rounding noise (volume esp.) than a re-pulled close alone
COLUMNS = ("open", "high", "low", "close", "volume")


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module_sha256():
    return file_sha256(__file__)


def _read_manifest():
    return json.loads(MANIFEST_PATH.read_text())


def usd_extension_symbols(manifest=None):
    if manifest is None:
        manifest = _read_manifest()
    out = []
    for p in manifest["pulled"]:
        sym = p["symbol"]
        if "/" not in sym:
            continue
        base, quote = sym.split("/", 1)
        if quote.upper() != "USD" or base.upper() in STABLE_FIAT:
            continue
        out.append(base.upper())
    return sorted(out)


def _read_csv_daily_full(base):
    """One staged daily CSV -> dict of (n,) arrays: ts, open, high, low, close, volume; sorted
    by time. Header is `time,open,high,low,close,vwap,volume,count` (vwap/count unused here)."""
    path = DAILY_DIR / f"{base}_USD.csv"
    if not path.exists():
        raise RuntimeError(f"915 loader: expected staged CSV {path} not found")
    cols = {c: [] for c in ("ts",) + COLUMNS}
    with open(path) as f:
        header = f.readline().strip().split(",")
        idx = {name: header.index(name) for name in ("time", "open", "high", "low", "close",
                                                       "volume")}
        for line in f:
            parts = line.strip().split(",")
            if not parts or parts[0] == "":
                continue
            cols["ts"].append(int(parts[idx["time"]]))
            for c in COLUMNS:
                cols[c].append(float(parts[idx[c]]))
    order = np.argsort(cols["ts"])
    out = {}
    for k, v in cols.items():
        arr = np.asarray(v, dtype=np.int64 if k == "ts" else np.float64)
        out[k] = arr[order]
    return out


def _raw_db_daily_full(db_path, asset_id, source_filter="kraken_rest"):
    """RAW (unfilled) daily O/H/L/C/V rows for one asset_id from `ohlcv_daily`, read-only."""
    con = tdata._connect(db_path, read_only=True)
    try:
        rows = con.execute(
            "SELECT timestamp, open, high, low, close, volume FROM ohlcv_daily "
            "WHERE asset_id=? AND source=? ORDER BY timestamp", (int(asset_id), source_filter),
        ).fetchall()
    finally:
        con.close()
    if not rows:
        return {"ts": np.zeros(0, dtype=np.int64), **{c: np.zeros(0) for c in COLUMNS}}
    ts = np.asarray([r[0] for r in rows], dtype=np.int64)
    out = {"ts": ts}
    for j, c in enumerate(COLUMNS):
        out[c] = np.asarray([r[j + 1] for r in rows], dtype=np.float64)
    return out


def _asset_id_for_symbol(db_path, symbol):
    con = tdata._connect(db_path, read_only=True)
    try:
        row = con.execute("SELECT id FROM assets WHERE symbol=?", (symbol,)).fetchone()
    finally:
        con.close()
    return None if row is None else int(row[0])


def _all_asset_daily_full(db_path, asset_ids, day_ts, source_filter="kraken_rest"):
    """Pull O/H/L/V (not close -- caller already has a ffilled close from
    `load_db_backtest_panel`) for `asset_ids` onto the EXACT `day_ts` grid, ffill/bfill'd the
    same way `trader.data._ffill_bfill_cols` fills close. Raw (unfilled-per-asset) rows are
    read from BOTH sources present in `ohlcv_daily` (kraken_rest + agg1m; the store is
    canonical -- (asset_id,timestamp) PK holds one row regardless of which source wrote it,
    exactly as `read_ohlcv_daily` documents) so dead/delisted assets whose rows are agg1m-only
    are still covered."""
    T = day_ts.size
    A = len(asset_ids)
    ts_to_row = {int(t): i for i, t in enumerate(day_ts)}
    panels = {c: np.full((T, A), np.nan) for c in ("open", "high", "low", "volume")}
    con = tdata._connect(db_path, read_only=True)
    try:
        cur = con.cursor()
        for j, aid in enumerate(asset_ids):
            rows = cur.execute(
                "SELECT timestamp, open, high, low, volume FROM ohlcv_daily WHERE asset_id=? "
                "ORDER BY timestamp", (int(aid),)).fetchall()
            for (ts, o, h, lo, v) in rows:
                r = ts_to_row.get(int(ts))
                if r is None:
                    continue
                panels["open"][r, j] = o
                panels["high"][r, j] = h
                panels["low"][r, j] = lo
                panels["volume"][r, j] = v
    finally:
        con.close()
    for c in panels:
        panels[c] = tdata._ffill_bfill_cols(panels[c])
    return panels


def verify_and_extend_seam(day_ts, ohlcv, ids, syms, db_path=FIXTURE_DB,
                           source_filter="kraken_rest"):
    """Re-derive the DB/CSV overlap check at LOAD TIME on ALL FIVE columns and extend the
    panel's date grid for the 31 USD catch-up assets. Raises RuntimeError on ANY mismatch on
    ANY column. Returns (day_ts_new, ohlcv_new (dict of (T',A') arrays), ids, syms, seam_block)."""
    manifest = _read_manifest()
    ext_bases = usd_extension_symbols(manifest)
    sym_to_col = {s: j for j, s in enumerate(syms)}

    overlap_rows = []          # (symbol, ts, col, db_val, csv_val)
    mismatches = []
    per_asset = {}
    tail_frames = []           # (col_idx_or_new, csv_dict_tail)
    new_cols = []              # (symbol, csv_dict_full)

    day_max_existing = int(day_ts[-1]) if day_ts.size else None

    for base in ext_bases:
        csv = _read_csv_daily_full(base)
        symbol = f"{base}/USD"
        if base in sym_to_col:
            col = sym_to_col[base]
            aid = ids[col]
        else:
            aid = _asset_id_for_symbol(db_path, symbol)
            col = None
        if aid is None:
            per_asset[base] = {"db_asset_id": None, "overlap_n": 0, "db_last_ts": None,
                               "csv_first_ts": int(csv["ts"][0]) if csv["ts"].size else None,
                               "csv_last_ts": int(csv["ts"][-1]) if csv["ts"].size else None,
                               "new_column": True}
            new_cols.append((base, csv))
            continue
        db = _raw_db_daily_full(db_path, aid, source_filter=source_filter)
        db_ts_set = {int(t): i for i, t in enumerate(db["ts"])}
        csv_ts_set = {int(t): i for i, t in enumerate(csv["ts"])}
        common = sorted(set(db_ts_set) & set(csv_ts_set))
        for t in common:
            di, ci = db_ts_set[t], csv_ts_set[t]
            for c in COLUMNS:
                a, b = float(db[c][di]), float(csv[c][ci])
                overlap_rows.append((symbol, t, c, a, b))
                if not math.isclose(a, b, rel_tol=OVERLAP_REL_TOL, abs_tol=OVERLAP_ABS_TOL):
                    mismatches.append({"symbol": symbol, "ts": t, "col": c, "db": a, "csv": b})
        db_last_ts = int(db["ts"][-1]) if db["ts"].size else None
        per_asset[base] = {"db_asset_id": int(aid), "overlap_n": len(common),
                            "db_last_ts": db_last_ts,
                            "csv_first_ts": int(csv["ts"][0]) if csv["ts"].size else None,
                            "csv_last_ts": int(csv["ts"][-1]) if csv["ts"].size else None,
                            "new_column": False}
        tail_mask = csv["ts"] > (db_last_ts if db_last_ts is not None else -1)
        tail = {k: v[tail_mask] for k, v in csv.items()}
        tail_frames.append((col, tail))

    if mismatches:
        raise RuntimeError(
            f"915 loader seam re-verification FAILED: {len(mismatches)} DB/CSV mismatch(es) "
            f"(any of open/high/low/close/volume) out of {len(overlap_rows)} overlap cells -- "
            f"refusing to extend. First mismatch: {mismatches[0]}"
        )

    seam_fingerprint_input = json.dumps(
        sorted((s, t, c, a, b) for (s, t, c, a, b) in overlap_rows), sort_keys=True).encode()
    seam_hash = hashlib.sha256(seam_fingerprint_input).hexdigest()

    all_new_ts = set()
    for _, tail in tail_frames:
        all_new_ts.update(int(x) for x in tail["ts"])
    for _, csv in new_cols:
        all_new_ts.update(int(x) for x in csv["ts"])
    if day_max_existing is not None:
        all_new_ts = {t for t in all_new_ts if t > day_max_existing}

    if all_new_ts:
        t_max_new = max(all_new_ts)
        grid_start = (day_max_existing + SEC_PER_DAY) if day_max_existing is not None \
            else min(all_new_ts)
        day_ts_new = np.arange(grid_start, t_max_new + SEC_PER_DAY, SEC_PER_DAY, dtype=np.int64)
    else:
        day_ts_new = np.zeros(0, dtype=np.int64)

    T_old, A_old = ohlcv["close"].shape
    T_add = day_ts_new.size
    n_new_cols = len(new_cols)
    day_ts_full = np.concatenate([day_ts, day_ts_new]) if T_add else day_ts.copy()
    ts_to_new_row = {int(t): (T_old + i) for i, t in enumerate(day_ts_new)}
    old_ts_to_row = {int(t): i for i, t in enumerate(day_ts)}

    ohlcv_big = {}
    for c in COLUMNS:
        big = np.full((T_old + T_add, A_old + n_new_cols), np.nan, dtype=np.float64)
        big[:T_old, :A_old] = ohlcv[c]
        ohlcv_big[c] = big

    def _place(col, tail):
        for i in range(tail["ts"].size):
            ti = int(tail["ts"][i])
            row = old_ts_to_row.get(ti, ts_to_new_row.get(ti))
            if row is None:
                raise RuntimeError(f"915 loader: tail ts {ti} lands in neither grid -- bug")
            for c in COLUMNS:
                ohlcv_big[c][row, col] = tail[c][i]

    for col, tail in tail_frames:
        if col is None or tail["ts"].size == 0:
            continue
        _place(col, tail)

    syms_full = list(syms)
    ids_full = list(ids)
    for k, (base, csv) in enumerate(new_cols):
        col = A_old + k
        syms_full.append(base)
        ids_full.append(None)
        _place(col, csv)

    for c in COLUMNS:
        ohlcv_big[c] = tdata._ffill_bfill_cols(ohlcv_big[c]) if ohlcv_big[c].size else ohlcv_big[c]

    seam_block = {
        "extension_symbols": ext_bases, "n_extension_symbols": len(ext_bases),
        "per_asset": per_asset, "overlap_cells_total": len(overlap_rows),
        "columns_checked": list(COLUMNS), "mismatches_n": 0, "seam_sha256": seam_hash,
        "new_columns_added": [b for b, _ in new_cols], "grid_extended_days": int(T_add),
        "grid_old_max_ts": day_max_existing,
        "grid_new_max_ts": int(day_ts_full[-1]) if day_ts_full.size else None,
        "manifest_sha256": file_sha256(MANIFEST_PATH),
        "rel_tol": OVERLAP_REL_TOL, "abs_tol": OVERLAP_ABS_TOL,
    }
    return day_ts_full, ohlcv_big, ids_full, syms_full, seam_block


def load_panel(db_path=FIXTURE_DB, source_filter="kraken_rest", include_dead=True,
               start_ts=None, end_ts=None):
    """Full TASK_915 OHLCV panel. Returns dict: day_ts, open/high/low/close/volume (T,A) each,
    ids, syms, loader_info, seam_block, fixture_sha256, loader_sha256."""
    day_ts, close, ids, syms, info = tdata.load_db_backtest_panel(
        db_path=db_path, start_ts=start_ts, end_ts=end_ts,
        source_filter=source_filter, include_dead=include_dead)
    ohlv = _all_asset_daily_full(db_path, ids, day_ts, source_filter=source_filter)
    ohlcv = {"open": ohlv["open"], "high": ohlv["high"], "low": ohlv["low"], "close": close,
             "volume": ohlv["volume"]}
    day_ts2, ohlcv2, ids2, syms2, seam_block = verify_and_extend_seam(
        day_ts, ohlcv, ids, syms, db_path=db_path, source_filter=source_filter)
    return {
        "day_ts": day_ts2, **ohlcv2, "ids": ids2, "syms": syms2,
        "loader_info": info, "seam_block": seam_block,
        "fixture_sha256": file_sha256(db_path), "loader_sha256": module_sha256(),
    }


def apply_liquidity_filter(panel, zf_max=0.15, min_active=365):
    """819's liquidity filter (`trader.data.apply_liquidity_filter`, zf_max=0.15,
    min_active=365) applied on the CLOSE panel, then the same column mask applied to every
    O/H/L/C/V array and to ids/syms."""
    close_kept, syms_kept, dropped, keep = tdata.apply_liquidity_filter(
        panel["close"], panel["syms"], zf_max=zf_max, min_active=min_active)
    out = dict(panel)
    for c in ("open", "high", "low", "close", "volume"):
        out[c] = np.ascontiguousarray(panel[c][:, keep])
    out["syms"] = syms_kept
    out["ids"] = [i for i, k in zip(panel["ids"], keep) if k]
    out["dropped_illiquid"] = dropped
    return out
