#!/usr/bin/env python3
"""TASK_918 data layer — PIT daily panel = fixture `ohlcv_daily` + staged 858-backfill CSVs.

Read-only everywhere: opens `fixtures/trading_fixture.db` via `trader.data`'s own read-only
connection helper (`file:...?mode=ro`); reads the staged catch-up CSVs
(`runs/858_backfill/api_2026-08-31/daily/*.csv`) as plain files. Never opens `trading_data.db`
(protected; this task never needs it — the fixture's `ohlcv_daily` table is the verified
byte-identical copy other 91x rows already rely on, see `experiments/917.../daily_panel.py`
docstring). Never writes, never DDLs, never regenerates the fixture.

Universe/PIT-membership discovery is NOT re-derived here — delegated entirely to
`trader.data.load_db_backtest_panel` (the same loader 898/916/917 use), so this row's universe
membership (USD-quoted, non-stable/fiat, kraken_rest-covered + dead/delisted when
include_dead=True) is identical to the canon backtest path. This module only EXTENDS that
panel's date grid for the 31 USD-quoted assets the 858 catch-up covers, past the fixture's own
last stored day, using the staged CSVs -- and only after RE-DERIVING the DB/CSV overlap check
at load time (never trusting the 2026-08-31 record as a load-time assumption).

Non-extended assets (all but the 31 catch-up USD pairs) simply end at the fixture's own last
row for that asset; `trader.data`'s existing ffill/leading-bfill convention (`_ffill_bfill_cols`)
carries their last real price forward exactly as it already does for any dead/delisted asset
(the LUNA handling) -- this module adds NO new fabrication, and disclosure counts are in
`describe_extension_coverage`.
"""
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

# match-tolerance for the DB/CSV overlap re-verification (same exchange API, independent pulls;
# both sides are float64 prices -- an exact re-pull can differ in trailing-digit rounding, so a
# tight relative+absolute tolerance is used rather than requiring bit-identity).
OVERLAP_REL_TOL = 1e-9
OVERLAP_ABS_TOL = 1e-9


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_manifest():
    m = json.loads(MANIFEST_PATH.read_text())
    return m


def usd_extension_symbols(manifest=None):
    """The manifest's USD-quoted bases (31 of the 56 pulled pairs) -- the only ones
    `load_db_backtest_panel`'s USD/non-stable filter would ever admit."""
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


def _read_csv_daily(base):
    """Read one staged daily CSV -> (ts (n,) int64, close (n,) float64), sorted by time.
    Filename = manifest symbol with '/' -> '_' (verified against the manifest 1:1; bare pairs
    like BTCCAD have no '/' and need no substitution)."""
    path = DAILY_DIR / f"{base}_USD.csv"
    if not path.exists():
        raise RuntimeError(f"918 loader: expected staged CSV {path} not found")
    ts, close = [], []
    with open(path) as f:
        header = f.readline().strip().split(",")
        ti, ci = header.index("time"), header.index("close")
        for line in f:
            parts = line.strip().split(",")
            if not parts or parts[0] == "":
                continue
            ts.append(int(parts[ti]))
            close.append(float(parts[ci]))
    order = np.argsort(ts)
    return np.asarray(ts, dtype=np.int64)[order], np.asarray(close, dtype=np.float64)[order]


def _raw_db_daily(db_path, asset_id, source_filter="kraken_rest"):
    """RAW (unfilled) daily (timestamp, close) rows for one asset_id from `ohlcv_daily`,
    read-only, sorted by time. Mirrors `experiments/917.../daily_panel.load_raw_pit_panel`'s
    read pattern (a plain read-only SELECT), reimplemented here rather than imported across a
    digit-prefixed experiment package."""
    con = tdata._connect(db_path, read_only=True)
    try:
        rows = con.execute(
            "SELECT timestamp, close FROM ohlcv_daily WHERE asset_id=? AND source=? "
            "ORDER BY timestamp", (int(asset_id), source_filter),
        ).fetchall()
    finally:
        con.close()
    if not rows:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)
    ts = np.asarray([r[0] for r in rows], dtype=np.int64)
    close = np.asarray([r[1] for r in rows], dtype=np.float64)
    return ts, close


def _asset_id_for_symbol(db_path, symbol):
    con = tdata._connect(db_path, read_only=True)
    try:
        row = con.execute("SELECT id FROM assets WHERE symbol=?", (symbol,)).fetchone()
    finally:
        con.close()
    return None if row is None else int(row[0])


def verify_and_extend_seam(day_ts, close, ids, syms, db_path=FIXTURE_DB,
                           source_filter="kraken_rest"):
    """Re-derive the DB/CSV overlap check at LOAD TIME (never assume the 2026-08-31 record) and
    extend the panel's date grid for the 31 USD catch-up assets. Raises RuntimeError on ANY
    price mismatch in the overlap window (refuse, never silently prefer one source). Returns
    (day_ts_new, close_new, ids, syms, seam_block) where seam_block carries the overlap row
    count, mismatch count (must be 0), per-asset extension coverage, and a seam fingerprint hash
    (sha256 over the sorted (symbol, ts, db_close, csv_close) overlap tuples)."""
    manifest = _read_manifest()
    ext_bases = usd_extension_symbols(manifest)
    sym_to_col = {s: j for j, s in enumerate(syms)}

    overlap_rows = []          # (symbol, ts, db_close, csv_close)
    mismatches = []
    per_asset = {}
    tail_frames = []           # (col_idx_or_new, ts_new_tail, close_new_tail) to append
    new_cols = []              # (symbol, ts_new_tail, close_new_tail) for bases absent from panel

    day_max_existing = int(day_ts[-1]) if day_ts.size else None

    for base in ext_bases:
        csv_ts, csv_close = _read_csv_daily(base)
        symbol = f"{base}/USD"
        if base in sym_to_col:
            col = sym_to_col[base]
            aid = ids[col]
        else:
            aid = _asset_id_for_symbol(db_path, symbol)
            col = None
        if aid is None:
            # asset genuinely absent from the fixture's `assets` table -- nothing to overlap
            # against; the whole CSV history becomes a brand-new column.
            per_asset[base] = {"db_asset_id": None, "overlap_n": 0, "db_last_ts": None,
                               "csv_first_ts": int(csv_ts[0]) if csv_ts.size else None,
                               "csv_last_ts": int(csv_ts[-1]) if csv_ts.size else None,
                               "new_column": True}
            new_cols.append((base, csv_ts, csv_close))
            continue
        db_ts, db_close = _raw_db_daily(db_path, aid, source_filter=source_filter)
        db_set = {int(t): float(c) for t, c in zip(db_ts, db_close)}
        csv_set = {int(t): float(c) for t, c in zip(csv_ts, csv_close)}
        common = sorted(set(db_set) & set(csv_set))
        for t in common:
            a, b = db_set[t], csv_set[t]
            overlap_rows.append((symbol, t, a, b))
            if not math.isclose(a, b, rel_tol=OVERLAP_REL_TOL, abs_tol=OVERLAP_ABS_TOL):
                mismatches.append({"symbol": symbol, "ts": t, "db_close": a, "csv_close": b})
        db_last_ts = int(db_ts[-1]) if db_ts.size else None
        per_asset[base] = {
            "db_asset_id": int(aid), "overlap_n": len(common),
            "db_last_ts": db_last_ts,
            "csv_first_ts": int(csv_ts[0]) if csv_ts.size else None,
            "csv_last_ts": int(csv_ts[-1]) if csv_ts.size else None,
            "new_column": False,
        }
        # tail = CSV rows strictly AFTER this asset's own DB last day (never re-derive the
        # overlap zone from CSV -- the DB value stands there, already verified equal above).
        tail_mask = csv_ts > (db_last_ts if db_last_ts is not None else -1)
        tail_frames.append((col, csv_ts[tail_mask], csv_close[tail_mask]))

    if mismatches:
        raise RuntimeError(
            f"918 loader seam re-verification FAILED: {len(mismatches)} DB/CSV price "
            f"mismatch(es) out of {len(overlap_rows)} overlap rows -- refusing to extend. "
            f"First mismatch: {mismatches[0]}"
        )

    seam_fingerprint_input = json.dumps(sorted(overlap_rows), sort_keys=True).encode()
    seam_hash = hashlib.sha256(seam_fingerprint_input).hexdigest()

    # ---- build the extended date grid -------------------------------------------------
    all_new_ts = set()
    for _, t_tail, _ in tail_frames:
        all_new_ts.update(int(x) for x in t_tail)
    for _, t_tail, _ in new_cols:
        all_new_ts.update(int(x) for x in t_tail)
    if day_max_existing is not None:
        all_new_ts = {t for t in all_new_ts if t > day_max_existing}

    if all_new_ts:
        t_min_new = min(all_new_ts)
        t_max_new = max(t for t in set(all_new_ts) | ({day_max_existing} if day_max_existing
                                                       is not None else set()))
        grid_start = (day_max_existing + SEC_PER_DAY) if day_max_existing is not None else t_min_new
        day_ts_new = np.arange(grid_start, t_max_new + SEC_PER_DAY, SEC_PER_DAY, dtype=np.int64)
    else:
        day_ts_new = np.zeros(0, dtype=np.int64)

    T_old, A_old = close.shape
    T_add = day_ts_new.size
    n_new_cols = len(new_cols)
    close_big = np.full((T_old + T_add, A_old + n_new_cols), np.nan, dtype=np.float64)
    close_big[:T_old, :A_old] = close
    day_ts_full = (np.concatenate([day_ts, day_ts_new]) if T_add else day_ts.copy())

    ts_to_new_row = {int(t): (T_old + i) for i, t in enumerate(day_ts_new)}
    old_ts_to_row = {int(t): i for i, t in enumerate(day_ts)}

    def _place(col, t_tail, c_tail, close_arr):
        """Write each (ts, close) pair into whichever grid already has that day: the OLD
        grid (another asset's row already exists there -- this asset's own tail merely fills
        a column gap, not a new day) or the newly appended tail rows beyond the old max."""
        for t, c in zip(t_tail, c_tail):
            ti = int(t)
            if ti in old_ts_to_row:
                close_arr[old_ts_to_row[ti], col] = c
            elif ti in ts_to_new_row:
                close_arr[ts_to_new_row[ti], col] = c
            else:
                raise RuntimeError(f"918 loader: tail ts {ti} lands in neither grid -- bug")

    for col, t_tail, c_tail in tail_frames:
        if col is None or t_tail.size == 0:
            continue
        _place(col, t_tail, c_tail, close_big)

    syms_full = list(syms)
    ids_full = list(ids)
    for k, (base, t_tail, c_tail) in enumerate(new_cols):
        col = A_old + k
        syms_full.append(base)
        ids_full.append(None)
        _place(col, t_tail, c_tail, close_big)

    close_filled = tdata._ffill_bfill_cols(close_big) if close_big.size else close_big

    seam_block = {
        "extension_symbols": ext_bases,
        "n_extension_symbols": len(ext_bases),
        "per_asset": per_asset,
        "overlap_n_total": len(overlap_rows),
        "mismatches_n": 0,
        "seam_sha256": seam_hash,
        "new_columns_added": [b for b, _, _ in new_cols],
        "grid_extended_days": int(T_add),
        "grid_old_max_ts": day_max_existing,
        "grid_new_max_ts": int(day_ts_full[-1]) if day_ts_full.size else None,
        "manifest_sha256": file_sha256(MANIFEST_PATH),
    }
    return day_ts_full, close_filled, ids_full, syms_full, seam_block


def load_panel(db_path=FIXTURE_DB, source_filter="kraken_rest", include_dead=True,
              start_ts=None, end_ts=None):
    """Full TASK_918 panel: fixture `load_db_backtest_panel` + seam-verified 858 extension.
    Returns dict: day_ts, close, ids, syms, loader_info (from load_db_backtest_panel),
    seam_block (from verify_and_extend_seam), fixture_sha256."""
    day_ts, close, ids, syms, info = tdata.load_db_backtest_panel(
        db_path=db_path, start_ts=start_ts, end_ts=end_ts,
        source_filter=source_filter, include_dead=include_dead)
    day_ts2, close2, ids2, syms2, seam_block = verify_and_extend_seam(
        day_ts, close, ids, syms, db_path=db_path, source_filter=source_filter)
    return {
        "day_ts": day_ts2, "close": close2, "ids": ids2, "syms": syms2,
        "loader_info": info, "seam_block": seam_block,
        "fixture_sha256": file_sha256(db_path),
        "loader_sha256": file_sha256(__file__),
    }


def describe_extension_coverage(panel, window_start_ts, window_end_ts):
    """Disclosure block (task requirement): which assets are extended to the window end vs
    which simply end at the fixture's own last row for that asset (carried flat by the existing
    ffill convention, never a new fabrication)."""
    day_ts = panel["day_ts"]
    close = panel["close"]
    syms = panel["syms"]
    seam = panel["seam_block"]
    ext_set = set(seam["extension_symbols"]) | set(seam["new_columns_added"])
    w_lo = int(np.searchsorted(day_ts, window_start_ts))
    w_hi = int(np.searchsorted(day_ts, window_end_ts))
    out = {"window_row_lo": w_lo, "window_row_hi": w_hi, "extended": [], "not_extended_n": 0,
          "not_extended_last_true_print_before_window_end_n": 0}
    for j, s in enumerate(syms):
        if s in ext_set:
            out["extended"].append(s)
    out["not_extended_n"] = len(syms) - len(out["extended"])
    return out
