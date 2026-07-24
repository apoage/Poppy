#!/usr/bin/env python3
"""
TASK_858 — Guarded minute-table backfill 2025-10 -> present.

The ONLY sanctioned write path to trading_data.db (the 24GB read-only source of
record). Everything else in the repo treats that DB as read-only; D22 admits this
one guarded, APPEND-ONLY ingest path.

Data path (measured, drives the design):
  - Kraken's public OHLC endpoint returns only the last ~720 candles/pair and can
    NEVER backfill 9 months. The backfill therefore ingests Kraken's quarterly
    OHLCVT CSV dumps ({SYMBOL}_1.csv, 1-min) via the DB's original build format.
  - The existing incremental fetcher (mcts_trader/data/fetcher.py) covers only the
    <=12h tail after the dumps; wired here as an opt-in top-up (never in the loop).

Guards (the load-bearing part — all fail LOUD):
  - Backup gate: refuses to run unless a fresh backup of the target DB exists
    (present AND at least as new as the DB). The backup is the recovery path;
    zstd->rsync does NOT detect source corruption, so the backup is mandatory.
  - Append-only invariant: never UPDATE/DELETE. An incoming (asset, timestamp) that
    already exists with DIFFERENT values aborts loud with a diff sample. Identical
    re-supplied candles are skipped (idempotent). Only brand-new keys are inserted.
  - Integrity checks post-ingest: OHLC sanity (l <= o,c <= h; v >= 0), no duplicate
    keys, monotone timestamps, row counts before/after.

Coverage report (the acceptance artifact) -> runs/858_backfill/coverage_report.{md,json}.

Usage:
    # real ingest (operator-run; requires downloaded dumps + a fresh backup):
    python tools/backfill_minute.py run \
        --db trading_data.db \
        --backup /path/to/trading_data.db.backup \
        --dump-dir /path/Kraken_OHLCVT_Q4_2025 /path/Kraken_OHLCVT_Q1_2026 ... \
        --universe runs/832_e5/frame_provenance_imprint.json

    # fixture leg (acceptance i + ii) — self-contained, no network, no real data:
    python tools/backfill_minute.py fixture-selftest
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Reuse the canonical filename->symbol parser + OHLC validation idioms from the
# original importer so the backfill matches the DB's original build path exactly.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from import_kraken_csv import parse_symbol_from_filename  # noqa: E402

DEFAULT_UNIVERSE = "runs/832_e5/frame_provenance_imprint.json"
DEFAULT_REPORT_DIR = "runs/858_backfill"
# 2025-10-01T00:00:00Z — the backfill coverage window start (see TASK_858 "Why").
WINDOW_START_EPOCH = int(datetime(2025, 10, 1, tzinfo=timezone.utc).timestamp())
ONE_DAY = 86400
FLOAT_TOL = 1e-9  # relative tolerance for "identical re-supplied candle"


# ---------------------------------------------------------------------------
# Loud, typed failures (no silent skip-count, no silent overwrite)
# ---------------------------------------------------------------------------
class BackfillError(Exception):
    """Base for all fail-loud backfill guard violations."""


class BackupGateError(BackfillError):
    """The mandatory backup gate refused the run."""


class AppendConflictError(BackfillError):
    """An incoming candle would overwrite a differing existing row."""


class IntegrityError(BackfillError):
    """A post-ingest integrity check failed."""


class UniverseError(BackfillError):
    """The 64-symbol universe could not be resolved."""


# ---------------------------------------------------------------------------
# Backup gate
# ---------------------------------------------------------------------------
def check_backup_gate(db_path: str, backup_path: Optional[str]) -> Dict:
    """Refuse to run unless a fresh, REAL, complete backup of db_path exists.

    A valid backup is a distinct file (never the DB itself), non-empty, at least
    as large as the DB, and at least as new (mtime). Every one of these fails
    LOUD with a specific reason — a fail-OPEN gate that waves through a
    self-reference, a 0-byte file, or a stale-but-touched file is worthless as
    the recovery path (zstd->rsync does not detect source corruption).
    """
    if not backup_path:
        raise BackupGateError(
            "BACKUP GATE: no --backup provided. This is the only write path to the "
            "source-of-record; a fresh backup is MANDATORY (zstd->rsync does not "
            "detect source corruption). Refusing to run."
        )
    if not os.path.exists(db_path):
        raise BackfillError(f"target DB does not exist: {db_path}")
    if not os.path.exists(backup_path):
        raise BackupGateError(
            f"BACKUP GATE: backup not found at {backup_path!r}. Refusing to run."
        )
    # Reject a backup that is really the DB itself (same path, symlink, or hardlink).
    db_stat = os.stat(db_path)
    bk_stat = os.stat(backup_path)
    same_path = os.path.realpath(db_path) == os.path.realpath(backup_path)
    same_inode = (db_stat.st_ino == bk_stat.st_ino
                  and db_stat.st_dev == bk_stat.st_dev)
    if same_path or same_inode:
        raise BackupGateError(
            f"BACKUP GATE: --backup ({backup_path!r}) points at the DB itself "
            f"({db_path!r}) (same path/inode). A backup must be an INDEPENDENT copy "
            "— it cannot be the recovery path for its own corruption. Refusing to run."
        )
    db_size = db_stat.st_size
    bk_size = bk_stat.st_size
    # Reject an empty backup (a `touch`ed placeholder / truncated file).
    if bk_size == 0:
        raise BackupGateError(
            f"BACKUP GATE: backup at {backup_path!r} is EMPTY (0 bytes). Refusing to run."
        )
    # Reject a backup smaller than the DB (partial copy, or a stale/old snapshot
    # from before the DB grew, rescued only by a `touch` on its mtime).
    if bk_size < db_size:
        raise BackupGateError(
            f"BACKUP GATE: backup at {backup_path!r} is SMALLER than the DB "
            f"({bk_size:,} < {db_size:,} bytes) — it is partial or stale (a `touch` "
            "cannot make an incomplete file whole). Refusing to run."
        )
    db_mtime = db_stat.st_mtime
    bk_mtime = bk_stat.st_mtime
    if bk_mtime < db_mtime:
        raise BackupGateError(
            "BACKUP GATE: backup is STALE — its mtime "
            f"({datetime.fromtimestamp(bk_mtime)}) predates the DB's "
            f"({datetime.fromtimestamp(db_mtime)}). Take a fresh backup before "
            "running. Refusing to run."
        )
    return {
        "db_path": os.path.abspath(db_path),
        "backup_path": os.path.abspath(backup_path),
        "db_mtime": db_mtime,
        "backup_mtime": bk_mtime,
        "db_size": db_size,
        "backup_size": bk_size,
    }


# ---------------------------------------------------------------------------
# Universe resolution (join through the assets table = canonical identity)
# ---------------------------------------------------------------------------
def load_universe_symbols(universe_path: str) -> List[str]:
    if not os.path.exists(universe_path):
        raise UniverseError(
            f"UNIVERSE: frame provenance not found at {universe_path!r}. This file "
            "defines the 64-symbol slot-locked universe and is REQUIRED (it is pulled "
            "from gpu-server under runs/). Provide the exact path via --universe; do "
            "NOT guess the universe."
        )
    with open(universe_path, "r") as f:
        data = json.load(f)
    syms = data.get("frame_symbols")
    if not syms or not isinstance(syms, list):
        raise UniverseError(
            f"UNIVERSE: {universe_path!r} has no non-empty 'frame_symbols' list."
        )
    return list(syms)


def _symbol_variants(symbol: str) -> List[str]:
    """Candidate DB-symbol spellings for one universe/parsed symbol."""
    s = symbol.strip()
    cands = [
        s,
        s.lstrip("/"),
        s.replace("XBT", "BTC"),
        s.replace("BTC", "XBT"),
        s.replace("/", ""),
        s.lstrip("/").replace("/", ""),
    ]
    out = []
    for c in cands:
        if c and c not in out:
            out.append(c)
    return out


def _canon_key(symbol: str) -> str:
    """Spelling-insensitive match key for routing a dump file to a universe slot.

    Drops ALL slashes, normalizes XBT->BTC, uppercases. This round-trips the DB
    spellings that `_symbol_variants` alone cannot: leading-slash forms
    (`/USDTEUR`, `/USDC/USD`) and BTC-spelled forms (`BTCGBP`, `ETHBTC`) — a dump
    file `USDTEUR_1.csv` parses to `USDT/EUR` and `BTCGBP_1.csv` to `XBT/GBP`,
    both of which canonicalize to the same key as their DB symbol. Matching by
    resolved asset_id via this key (not by string-variant guessing) reaches all
    64 universe slots instead of 53.
    """
    return symbol.upper().replace("/", "").replace("XBT", "BTC")


def resolve_universe_asset_ids(
    cursor: sqlite3.Cursor, universe_symbols: List[str]
) -> Tuple[Dict[str, int], List[str]]:
    """Map each universe symbol to a DB asset_id (via variants).

    Returns (symbol -> asset_id, unresolved_symbols). Unresolved symbols are a
    coverage concern surfaced in the report, not a hard abort (some slots may be
    absent from a given DB / fixture).
    """
    cursor.execute("SELECT id, symbol FROM assets")
    symbol_map = {row[1]: row[0] for row in cursor.fetchall()}
    resolved: Dict[str, int] = {}
    unresolved: List[str] = []
    for sym in universe_symbols:
        aid = None
        for v in _symbol_variants(sym):
            if v in symbol_map:
                aid = symbol_map[v]
                break
        if aid is None:
            unresolved.append(sym)
        else:
            resolved[sym] = aid
    return resolved, unresolved


# ---------------------------------------------------------------------------
# CSV parsing (mirrors import_kraken_csv row validation)
# ---------------------------------------------------------------------------
def _parse_csv_rows(filepath: Path) -> Tuple[List[Tuple], int]:
    """Parse a Kraken OHLCVT 1-min CSV — STRUCTURAL parsing only.

    Returns (rows, malformed) where each row is
    (timestamp, open, high, low, close, volume, trade_count) and `malformed`
    counts rows that could not be parsed at all (too few columns, non-numeric).

    Value SANITY (price > 0, OHLC ordering, volume >= 0) is deliberately NOT
    filtered here: silently dropping a suspect candle before the guard would hide
    bad incoming data from the source-of-record write. Numeric-but-invalid rows
    flow through to `integrity_check`, which aborts the whole run LOUD (and rolls
    back) rather than skipping them silently.
    """
    rows: List[Tuple] = []
    malformed = 0
    with open(filepath, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 6:
                malformed += 1
                continue
            try:
                ts = int(row[0])
                o = float(row[1])
                h = float(row[2])
                low = float(row[3])
                c = float(row[4])
                v = float(row[5])
                tc = int(row[6]) if len(row) > 6 else 0
            except (ValueError, IndexError):
                malformed += 1
                continue
            rows.append((ts, o, h, low, c, v, tc))
    return rows, malformed


def _values_differ(a: Tuple, b: Tuple) -> bool:
    """Compare (o,h,l,c,v) with relative tolerance; trade_count exact."""
    for x, y in zip(a[:5], b[:5]):
        if abs(x - y) > FLOAT_TOL * max(1.0, abs(x), abs(y)):
            return True
    if int(a[5]) != int(b[5]):
        return True
    return False


# ---------------------------------------------------------------------------
# Append-only ingest with conflict detection
# ---------------------------------------------------------------------------
def ingest_file(
    cursor: sqlite3.Cursor,
    filepath: Path,
    asset_id: int,
    symbol: str,
    inserted_keys: Dict[int, set],
    preexist_vals: Dict[int, Dict[int, Tuple]],
    new_rows: Dict[int, List[Tuple]],
    max_conflict_samples: int = 10,
) -> Dict:
    """Append-only ingest of one CSV into ohlcv for asset_id (uncommitted).

    Aborts LOUD (AppendConflictError) BEFORE inserting anything from this file if
    any incoming (asset_id, ts) already exists with DIFFERENT values.

    Threads three run-scoped accumulators (mutated in place) so the orchestrator
    can prove append-only atomically across ALL files/dirs:
      - `inserted_keys[aid]`: timestamps WE inserted this run (so a later dump dir
        that overlaps our own inserts is not mistaken for pre-existing data).
      - `preexist_vals[aid]`: the TRUE pre-existing {ts -> (o,h,l,c,v,tc)} map for
        keys in the incoming range — captured before we touch them, excluding our
        own inserts. This is the subset-proof anchor.
      - `new_rows[aid]`: the newly-inserted candle tuples, for integrity scoping.
    """
    rows, malformed = _parse_csv_rows(filepath)
    if not rows:
        return {
            "file": filepath.name, "symbol": symbol, "asset_id": asset_id,
            "parsed": 0, "inserted": 0, "idempotent": 0,
            "malformed": malformed,
            "min_ts": None, "max_ts": None,
        }

    min_ts = min(r[0] for r in rows)
    max_ts = max(r[0] for r in rows)

    # Load only the existing rows in the incoming range (bounds memory).
    cursor.execute(
        "SELECT timestamp, open, high, low, close, volume, trade_count "
        "FROM ohlcv WHERE asset_id = ? AND timestamp BETWEEN ? AND ?",
        (asset_id, min_ts, max_ts),
    )
    existing = {r[0]: r[1:] for r in cursor.fetchall()}

    # Capture the TRUE pre-existing values (exclude keys we inserted earlier this
    # run; record each key only once across files) for the append-only proof.
    own = inserted_keys.setdefault(asset_id, set())
    pre_map = preexist_vals.setdefault(asset_id, {})
    for ts, vals in existing.items():
        if ts not in own and ts not in pre_map:
            pre_map[ts] = vals

    to_insert: List[Tuple] = []
    idempotent = 0
    conflicts: List[Dict] = []
    for ts, o, h, low, c, v, tc in rows:
        incoming_vals = (o, h, low, c, v, tc)
        if ts in existing:
            if _values_differ(incoming_vals, existing[ts]):
                if len(conflicts) < max_conflict_samples:
                    ex = existing[ts]
                    conflicts.append({
                        "asset_id": asset_id, "symbol": symbol, "timestamp": ts,
                        "utc": datetime.fromtimestamp(ts, timezone.utc).isoformat(),
                        "existing": {"open": ex[0], "high": ex[1], "low": ex[2],
                                     "close": ex[3], "volume": ex[4], "trade_count": ex[5]},
                        "incoming": {"open": o, "high": h, "low": low,
                                     "close": c, "volume": v, "trade_count": tc},
                    })
            else:
                idempotent += 1
        else:
            row = (asset_id, ts, o, h, low, c, v, tc)
            to_insert.append(row)
            # Reflect our own insert so an intra-run duplicate ts is handled as
            # idempotent/conflict rather than a raw PK crash.
            existing[ts] = incoming_vals
            own.add(ts)
            new_rows.setdefault(asset_id, []).append((ts, o, h, low, c, v, tc))

    if conflicts:
        raise AppendConflictError(
            f"APPEND-ONLY VIOLATION on {symbol} ({filepath.name}): "
            f"{len(conflicts)}+ incoming candle(s) conflict with existing rows. "
            "Refusing to overwrite the source-of-record. Diff sample:\n"
            + json.dumps(conflicts, indent=2)
        )

    # Pure append: brand-new keys only. INSERT (no OR REPLACE/IGNORE) so a race
    # that made a key appear after our read raises IntegrityError, not silence.
    if to_insert:
        cursor.executemany(
            "INSERT INTO ohlcv "
            "(asset_id, timestamp, open, high, low, close, volume, trade_count) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            to_insert,
        )

    return {
        "file": filepath.name, "symbol": symbol, "asset_id": asset_id,
        "parsed": len(rows), "inserted": len(to_insert), "idempotent": idempotent,
        "malformed": malformed,
        "min_ts": min_ts, "max_ts": max_ts,
    }


# ---------------------------------------------------------------------------
# Integrity checks
# ---------------------------------------------------------------------------
def integrity_check(new_rows: List[Tuple], asset_id: int, symbol: str,
                    max_samples: int = 10) -> Dict:
    """Integrity of the NEWLY-INSERTED candles for one symbol (in memory).

    Scoped to the incoming rows only — NEVER the pre-existing series, so a
    pre-existing quirk this task is not here to audit cannot abort the run.
    Raises IntegrityError (LOUD, pre-commit) on any failure so the orchestrator
    rolls the whole transaction back. Each `new_rows` tuple is
    (timestamp, open, high, low, close, volume, trade_count).

    Checks: price > 0 (closes the zero/negative-price gap), OHLC ordering
    (l <= o,c and h >= o,c and h >= l), volume >= 0, and no duplicate timestamp
    within the batch (a PK crash otherwise, but caught here with a clear reason).
    """
    bad: List[Dict] = []
    seen = set()
    n_dup = 0
    for ts, o, h, low, c, v, tc in new_rows:
        reason = None
        if o <= 0 or h <= 0 or low <= 0 or c <= 0:
            reason = "non-positive price"
        elif low > o or low > c or h < o or h < c or h < low:
            reason = "OHLC ordering (need l<=o,c and h>=o,c and h>=l)"
        elif v < 0:
            reason = "negative volume"
        if reason is not None and len(bad) < max_samples:
            bad.append({
                "asset_id": asset_id, "symbol": symbol, "timestamp": ts,
                "utc": datetime.fromtimestamp(ts, timezone.utc).isoformat(),
                "open": o, "high": h, "low": low, "close": c, "volume": v,
                "reason": reason,
            })
        if ts in seen:
            n_dup += 1
        seen.add(ts)
    if bad:
        raise IntegrityError(
            f"INTEGRITY: {symbol} (asset_id={asset_id}) — "
            f"{len(bad)}+ incoming candle(s) fail OHLC/volume sanity. "
            "Refusing to write bad data to the source-of-record. Sample:\n"
            + json.dumps(bad, indent=2)
        )
    if n_dup:
        raise IntegrityError(
            f"INTEGRITY: {symbol} (asset_id={asset_id}) — {n_dup} duplicate "
            "timestamp(s) among incoming candles."
        )
    return {"checked": len(new_rows), "ohlc_bad": 0, "duplicates": 0}


def _row_count(cursor: sqlite3.Cursor, asset_id: int) -> int:
    cursor.execute("SELECT COUNT(*) FROM ohlcv WHERE asset_id = ?", (asset_id,))
    return cursor.fetchone()[0]


def _last_ts(cursor: sqlite3.Cursor, asset_id: int) -> Optional[int]:
    cursor.execute("SELECT MAX(timestamp) FROM ohlcv WHERE asset_id = ?", (asset_id,))
    return cursor.fetchone()[0]


def window_row_hash(cursor: sqlite3.Cursor, asset_id: int,
                    start_ts: int, end_ts: int) -> Tuple[str, int]:
    """Stable hash of the exact rows for asset_id in [start_ts, end_ts].

    Used to PROVE append-only: a pre-existing window's hash must be identical
    pre- vs post-ingest (existing rows never mutated).
    """
    cursor.execute(
        "SELECT timestamp, open, high, low, close, volume, trade_count "
        "FROM ohlcv WHERE asset_id = ? AND timestamp BETWEEN ? AND ? "
        "ORDER BY timestamp",
        (asset_id, start_ts, end_ts),
    )
    h = hashlib.sha256()
    n = 0
    for r in cursor.fetchall():
        h.update(repr(r).encode())
        n += 1
    return h.hexdigest(), n


def find_gaps(cursor: sqlite3.Cursor, asset_id: int,
              start_ts: int, end_ts: int, threshold: int = ONE_DAY) -> List[Dict]:
    """Residual gaps > threshold seconds between consecutive candles in window."""
    cursor.execute(
        "SELECT timestamp FROM ohlcv WHERE asset_id = ? AND timestamp BETWEEN ? AND ? "
        "ORDER BY timestamp",
        (asset_id, start_ts, end_ts),
    )
    gaps: List[Dict] = []
    prev = None
    for (ts,) in cursor.fetchall():
        if prev is not None and (ts - prev) > threshold:
            gaps.append({
                "from_ts": prev, "to_ts": ts, "gap_seconds": ts - prev,
                "gap_days": round((ts - prev) / ONE_DAY, 2),
                "from_utc": datetime.fromtimestamp(prev, timezone.utc).isoformat(),
                "to_utc": datetime.fromtimestamp(ts, timezone.utc).isoformat(),
            })
        prev = ts
    return gaps


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_backfill(
    db_path: str,
    dump_dirs: List[str],
    universe_path: str = DEFAULT_UNIVERSE,
    backup_path: Optional[str] = None,
    report_dir: str = DEFAULT_REPORT_DIR,
    window_start: int = WINDOW_START_EPOCH,
    max_conflict_samples: int = 10,
    write_report: bool = True,
) -> Dict:
    """Run the guarded, append-only minute backfill. Returns the coverage dict."""
    # --- Guard 1: backup gate (fail loud before touching the DB) ---
    backup_info = check_backup_gate(db_path, backup_path)

    universe_symbols = load_universe_symbols(universe_path)

    conn = sqlite3.connect(db_path)
    conn.row_factory = None
    cursor = conn.cursor()

    try:
        resolved, unresolved = resolve_universe_asset_ids(cursor, universe_symbols)
        if not resolved:
            raise UniverseError(
                "UNIVERSE: none of the universe symbols resolved to a DB asset — "
                "wrong DB or wrong universe file. Refusing to run."
            )
        allowed_ids = set(resolved.values())
        id_to_symbol = {v: k for k, v in resolved.items()}
        # Route dump files to a universe SLOT by resolved asset_id, not by string
        # guessing: canon_key(DB spelling) -> asset_id. This reaches leading-slash
        # and BTC-spelled forms that _symbol_variants misses. Collisions are fatal.
        canon_to_aid: Dict[str, int] = {}
        for sym, aid in resolved.items():
            k = _canon_key(sym)
            if canon_to_aid.get(k, aid) != aid:
                raise UniverseError(
                    f"UNIVERSE: canonical dump key {k!r} collides between "
                    f"asset_id={canon_to_aid[k]} and {aid} ({sym}) — cannot route "
                    "dump files unambiguously. Refusing to run."
                )
            canon_to_aid[k] = aid

        # --- Pre-ingest snapshot (row counts, last ts) for touched-symbol report ---
        pre = {aid: {"row_count": _row_count(cursor, aid),
                     "last_ts": _last_ts(cursor, aid)}
               for aid in allowed_ids}

        # --- Ingest each dump dir (append-only, UNCOMMITTED) ---
        # Run-scoped accumulators for the append-only proof + integrity scoping.
        inserted_keys: Dict[int, set] = {}
        preexist_vals: Dict[int, Dict[int, Tuple]] = {}
        new_rows: Dict[int, List[Tuple]] = {}
        file_stats: List[Dict] = []
        seen_files = 0
        unmatched_files: List[Dict] = []
        for d in dump_dirs:
            if not os.path.isdir(d):
                raise BackfillError(f"dump dir not found: {d}")
            for filepath in sorted(Path(d).glob("*_1.csv")):
                sym = parse_symbol_from_filename(filepath.name)
                aid = canon_to_aid.get(_canon_key(sym))
                if aid is None:
                    # Not in the 64-universe — surfaced in the report, NOT silent,
                    # so "dump present but unmatched" != "no dump".
                    unmatched_files.append({
                        "file": filepath.name, "dir": os.path.abspath(d),
                        "parsed_symbol": sym, "canon_key": _canon_key(sym),
                    })
                    continue
                seen_files += 1
                st = ingest_file(cursor, filepath, aid, id_to_symbol.get(aid, sym),
                                 inserted_keys, preexist_vals, new_rows,
                                 max_conflict_samples)
                file_stats.append(st)

        # === GUARDS run against the UNCOMMITTED transaction. Commit ONLY if all
        # pass; any failure rolls the entire run back (see except clause). ===

        # --- Guard: integrity of the NEWLY-INSERTED rows only (per touched sym) ---
        touched = sorted({s["asset_id"] for s in file_stats})
        for aid in touched:
            integrity_check(new_rows.get(aid, []), aid, id_to_symbol.get(aid, str(aid)))

        # --- Guard: append-only SUBSET proof. Every pre-existing (asset,ts) must
        # still map to its ORIGINAL values (new interior/tail keys are allowed).
        # This PASSES on a legitimate gap-fill and FIRES on any real mutation. ---
        append_only_proof = []
        for aid, pre_map in preexist_vals.items():
            if not pre_map:
                continue
            lo, hi = min(pre_map), max(pre_map)
            cursor.execute(
                "SELECT timestamp, open, high, low, close, volume, trade_count "
                "FROM ohlcv WHERE asset_id = ? AND timestamp BETWEEN ? AND ?",
                (aid, lo, hi))
            post = {r[0]: r[1:] for r in cursor.fetchall()}
            pre_h = hashlib.sha256()
            post_h = hashlib.sha256()
            mismatches: List[Dict] = []
            for ts in sorted(pre_map):
                pv = pre_map[ts]
                pre_h.update(repr((ts,) + tuple(pv)).encode())
                if ts not in post or _values_differ(post[ts], pv):
                    if len(mismatches) < max_conflict_samples:
                        mismatches.append({
                            "timestamp": ts,
                            "original": list(pv),
                            "now": list(post.get(ts)) if ts in post else "DELETED",
                        })
                else:
                    post_h.update(repr((ts,) + tuple(post[ts])).encode())
            identical = not mismatches
            append_only_proof.append({
                "asset_id": aid, "symbol": id_to_symbol.get(aid, str(aid)),
                "window": [lo, hi], "preexisting_keys": len(pre_map),
                "pre_hash": pre_h.hexdigest(), "post_hash": post_h.hexdigest(),
                "identical": identical,
            })
            if not identical:
                raise IntegrityError(
                    "APPEND-ONLY PROOF FAILED for asset_id="
                    f"{aid} ({id_to_symbol.get(aid)}): {len(mismatches)}+ "
                    "pre-existing row(s) were mutated or deleted. Sample:\n"
                    + json.dumps(mismatches, indent=2)
                )

        # --- All guards passed: commit the single atomic transaction. ---
        conn.commit()

        # --- Coverage report (reads the now-durable state) ---
        now = int(time.time())
        per_symbol = []
        for aid in sorted(allowed_ids):
            sym = id_to_symbol.get(aid, str(aid))
            post_count = _row_count(cursor, aid)
            post_last = _last_ts(cursor, aid)
            added = post_count - pre[aid]["row_count"]
            gaps = find_gaps(cursor, aid, window_start, now) if post_last else []
            per_symbol.append({
                "symbol": sym, "asset_id": aid,
                "old_last_ts": pre[aid]["last_ts"],
                "old_last_utc": (datetime.fromtimestamp(pre[aid]["last_ts"], timezone.utc).isoformat()
                                 if pre[aid]["last_ts"] else None),
                "new_last_ts": post_last,
                "new_last_utc": (datetime.fromtimestamp(post_last, timezone.utc).isoformat()
                                 if post_last else None),
                "rows_added": added,
                "residual_gaps_gt_1d": gaps,
                "n_residual_gaps": len(gaps),
            })

        coverage = {
            "task": "TASK_858",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "db_path": os.path.abspath(db_path),
            "universe_path": os.path.abspath(universe_path),
            "backup": backup_info,
            "dump_dirs": [os.path.abspath(d) for d in dump_dirs],
            "window_start_ts": window_start,
            "window_start_utc": datetime.fromtimestamp(window_start, timezone.utc).isoformat(),
            "n_universe": len(universe_symbols),
            "n_resolved": len(resolved),
            "unresolved_symbols": unresolved,
            "files_ingested": seen_files,
            "files_unmatched": len(unmatched_files),
            "unmatched_files": unmatched_files,
            "total_rows_added": sum(s["inserted"] for s in file_stats),
            "total_idempotent_skipped": sum(s["idempotent"] for s in file_stats),
            "per_symbol": per_symbol,
            "append_only_proof": append_only_proof,
            "file_stats": file_stats,
        }
    except Exception:
        # Any failure — a guard, a mid-ingest abort, anything — rolls the whole
        # transaction back so the source-of-record stays byte-identical.
        conn.rollback()
        raise
    finally:
        conn.close()

    if write_report:
        _write_report(coverage, report_dir)

    return coverage


def _write_report(coverage: Dict, report_dir: str) -> None:
    os.makedirs(report_dir, exist_ok=True)
    json_path = os.path.join(report_dir, "coverage_report.json")
    md_path = os.path.join(report_dir, "coverage_report.md")
    with open(json_path, "w") as f:
        json.dump(coverage, f, indent=2)

    lines = []
    lines.append("# TASK_858 minute-backfill coverage report\n")
    lines.append(f"- generated: {coverage['generated_utc']}")
    lines.append(f"- db: `{coverage['db_path']}`")
    lines.append(f"- universe: `{coverage['universe_path']}` "
                 f"({coverage['n_resolved']}/{coverage['n_universe']} resolved)")
    lines.append(f"- window start: {coverage['window_start_utc']}")
    lines.append(f"- dumps: {', '.join('`%s`' % d for d in coverage['dump_dirs']) or '(none)'}")
    lines.append(f"- files ingested: {coverage['files_ingested']}")
    lines.append(f"- files present but UNMATCHED to the universe: {coverage['files_unmatched']}")
    lines.append(f"- total rows added: {coverage['total_rows_added']:,}")
    lines.append(f"- idempotent rows skipped: {coverage['total_idempotent_skipped']:,}")
    if coverage["unresolved_symbols"]:
        lines.append(f"- UNRESOLVED universe symbols ({len(coverage['unresolved_symbols'])}): "
                     f"{', '.join(coverage['unresolved_symbols'])}")
    if coverage["unmatched_files"]:
        lines.append(f"- UNMATCHED dump files ({len(coverage['unmatched_files'])}): "
                     + ", ".join(f"`{u['file']}`(->{u['parsed_symbol']})"
                                 for u in coverage["unmatched_files"][:20])
                     + (" ..." if len(coverage["unmatched_files"]) > 20 else ""))
    all_identical = all(p["identical"] for p in coverage["append_only_proof"]) \
        if coverage["append_only_proof"] else True
    lines.append(f"- append-only proof: "
                 f"{'ALL IDENTICAL' if all_identical else 'FAILED'} "
                 f"({len(coverage['append_only_proof'])} pre-existing windows checked)")
    lines.append("")
    lines.append("| symbol | old_last | new_last | rows_added | gaps>1d |")
    lines.append("|---|---|---|---|---|")
    for r in coverage["per_symbol"]:
        lines.append(
            f"| {r['symbol']} | {r['old_last_utc'] or '-'} | "
            f"{r['new_last_utc'] or '-'} | {r['rows_added']:,} | {r['n_residual_gaps']} |")
    lines.append("")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Tail top-up (opt-in; wraps the existing incremental fetcher)
# ---------------------------------------------------------------------------
def run_tail_topup(db_path: str, universe_path: str, days_back: int = 1) -> None:
    """Run the existing Kraken OHLC fetcher for the <=12h tail after the dumps.

    Opt-in only. Hits the live Kraken public API — NEVER run inside the factory
    loop (scheduled/operator step). No secrets, public endpoint, read-only fetch.
    """
    import asyncio
    from mcts_trader.data.kraken_client import KrakenClient
    from mcts_trader.data.database import Database
    from mcts_trader.data.fetcher import DataFetcher

    universe_symbols = load_universe_symbols(universe_path)

    async def _go():
        client = KrakenClient()
        db = Database(db_path)
        await db.connect()
        try:
            fetcher = DataFetcher(client, db)
            assets = [(s, s) for s in universe_symbols]
            await fetcher.fetch_all_assets(assets, days_back=days_back)
        finally:
            await db.close()

    asyncio.run(_go())


# ---------------------------------------------------------------------------
# Fixture self-test (acceptance i + ii) — self-contained
# ---------------------------------------------------------------------------
def _write_dump(dump_dir: Path, filename: str, rows: List[Tuple]) -> None:
    dump_dir.mkdir(parents=True, exist_ok=True)
    with open(dump_dir / filename, "w", newline="") as f:
        w = csv.writer(f)
        for r in rows:
            w.writerow(r)


def _touch_newer(path: str, ref: str) -> None:
    """Set path's mtime just after ref's, so a backup reads as fresh."""
    t = os.path.getmtime(ref) + 1
    os.utime(path, (t, t))


def fixture_selftest(fixture_db: str = "fixtures/trading_fixture.db",
                     universe_path: str = DEFAULT_UNIVERSE) -> int:
    """Exercise the whole orchestrator + guards against a COPY of the fixture.

    Covers acceptance (i) guards bite + report schema, and (ii) append-only proof.
    The later scenarios (E onward) are the FIX-PASS regressions that FAIL against
    the pre-fix code:
      E  interior gap-fill (new key with ts < existing max) SUCCEEDS, proof PASSES
         — the old whole-window hash false-fired on the added interior row.
      F  the subset proof FIRES on a simulated mutation of a pre-existing value.
      G  a guard failure leaves the DB row-count byte-identical (atomic rollback)
         — the old code committed BEFORE the guards, so a failure was durable.
      H  backup-gate bypasses (self-reference, 0-byte, touched-stale) each REFUSE.
      I  an incoming zero/negative-price candle aborts pre-commit (integrity bites
         on NEW data only), DB unchanged.
      J  the leading-slash (/USDTEUR) and BTC-spelled (BTCGBP) dump forms ingest.
    Returns 0 on full pass, non-zero otherwise.
    """
    if not os.path.exists(fixture_db):
        print(f"[fixture-selftest] fixture not found: {fixture_db}", file=sys.stderr)
        return 2

    workdir = Path(tempfile.mkdtemp(prefix="backfill858_"))
    print(f"[fixture-selftest] workdir: {workdir}")
    results = []

    def record(name: str, ok: bool, detail: str = "") -> None:
        results.append((name, ok, detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
              + (f" — {detail}" if detail else ""))

    # Build a small fixture universe from real fixture symbols with in-DB data.
    src = sqlite3.connect(fixture_db)
    cur = src.cursor()
    cur.execute(
        "SELECT a.symbol, o.asset_id, MIN(o.timestamp), MAX(o.timestamp), COUNT(*) "
        "FROM ohlcv o JOIN assets a ON a.id = o.asset_id "
        "GROUP BY o.asset_id ORDER BY COUNT(*) DESC LIMIT 3")
    picks = cur.fetchall()
    # grab an anchor existing row for the top symbol (for conflict + idempotent tests)
    top_sym, top_aid, top_min, top_max, _ = picks[0]
    cur.execute(
        "SELECT timestamp, open, high, low, close, volume, trade_count "
        "FROM ohlcv WHERE asset_id = ? ORDER BY timestamp LIMIT 8", (top_aid,))
    first_rows = cur.fetchall()
    anchor = first_rows[0]
    # Symbols whose DB spelling the old dump matcher could NOT round-trip
    # (leading-slash + BTC-spelled). Present in the fixture assets table.
    slash_sym = cur.execute(
        "SELECT symbol FROM assets WHERE symbol = '/USDTEUR'").fetchone()
    btc_sym = cur.execute(
        "SELECT symbol FROM assets WHERE symbol = 'BTCGBP'").fetchone()
    src.close()

    fixture_universe = [row[0] for row in picks]
    uni_path = workdir / "fixture_universe.json"
    with open(uni_path, "w") as f:
        json.dump({"frame_symbols": fixture_universe, "n_assets": len(fixture_universe),
                   "source": "fixture-selftest synthetic universe"}, f)

    def sym_to_filename(sym: str) -> str:
        return sym.replace("/", "") + "_1.csv"

    def make_db_copy(tag: str) -> str:
        p = str(workdir / f"copy_{tag}.db")
        shutil.copy(fixture_db, p)
        return p

    def make_backup(db: str, fresh: bool) -> str:
        b = db + ".backup"
        shutil.copy(db, b)
        if fresh:
            _touch_newer(b, db)
        else:
            # stale: older than the DB
            t = os.path.getmtime(db) - 100
            os.utime(b, (t, t))
        return b

    def total_rows(db: str) -> int:
        c = sqlite3.connect(db)
        n = c.execute("SELECT COUNT(*) FROM ohlcv").fetchone()[0]
        c.close()
        return n

    # New candles strictly AFTER the top symbol's current max (pure append),
    # plus a re-supply of the anchor row with IDENTICAL values (idempotent skip).
    step = 60
    new_rows = [(top_max + step * i,
                 anchor[1], anchor[2], anchor[3], anchor[4], anchor[5], anchor[6])
                for i in range(1, 6)]
    idempotent_row = (anchor[0], anchor[1], anchor[2], anchor[3],
                      anchor[4], anchor[5], anchor[6])

    # ---- Scenario A: missing-backup refusal ----
    try:
        dbA = make_db_copy("A")
        dumpA = workdir / "dumpA"
        _write_dump(dumpA, sym_to_filename(top_sym), new_rows)
        run_backfill(dbA, [str(dumpA)], str(uni_path), backup_path=None,
                     report_dir=str(workdir / "reportA"), write_report=False)
        record("missing-backup refusal", False, "run did NOT raise")
    except BackupGateError:
        record("missing-backup refusal", True, "BackupGateError raised")
    except Exception as e:
        record("missing-backup refusal", False, f"wrong error: {type(e).__name__}: {e}")

    # ---- Scenario B: stale-backup refusal ----
    try:
        dbB = make_db_copy("B")
        bkB = make_backup(dbB, fresh=False)
        run_backfill(dbB, [str(dumpA)], str(uni_path), backup_path=bkB,
                     report_dir=str(workdir / "reportB"), write_report=False)
        record("stale-backup refusal", False, "run did NOT raise")
    except BackupGateError:
        record("stale-backup refusal", True, "BackupGateError raised")
    except Exception as e:
        record("stale-backup refusal", False, f"wrong error: {type(e).__name__}: {e}")

    # ---- Scenario C: conflicting-candle abort ----
    try:
        dbC = make_db_copy("C")
        bkC = make_backup(dbC, fresh=True)
        dumpC = workdir / "dumpC"
        conflict_row = (anchor[0], anchor[1] + 999.0, anchor[2] + 999.0,
                        anchor[3], anchor[4] + 999.0, anchor[5], anchor[6])
        _write_dump(dumpC, sym_to_filename(top_sym), [conflict_row] + new_rows)
        run_backfill(dbC, [str(dumpC)], str(uni_path), backup_path=bkC,
                     report_dir=str(workdir / "reportC"), write_report=False)
        record("conflicting-candle abort", False, "run did NOT raise")
    except AppendConflictError as e:
        has_diff = "existing" in str(e) and "incoming" in str(e)
        record("conflicting-candle abort", has_diff,
               "AppendConflictError with diff sample" if has_diff
               else "raised but missing diff sample")
    except Exception as e:
        record("conflicting-candle abort", False, f"wrong error: {type(e).__name__}: {e}")

    # ---- Scenario D: happy path + append-only proof + report schema ----
    try:
        dbD = make_db_copy("D")
        bkD = make_backup(dbD, fresh=True)
        dumpD = workdir / "dumpD"
        _write_dump(dumpD, sym_to_filename(top_sym), new_rows + [idempotent_row])

        # capture the pre-existing full-range hash independently for (ii)
        pc = sqlite3.connect(dbD)
        pcur = pc.cursor()
        pre_hash, pre_n = window_row_hash(pcur, top_aid, top_min, top_max)
        pc.close()

        cov = run_backfill(dbD, [str(dumpD)], str(uni_path), backup_path=bkD,
                           report_dir=str(workdir / "reportD"), write_report=True)

        # append-only proof identical
        proof_ok = bool(cov["append_only_proof"]) and \
            all(p["identical"] for p in cov["append_only_proof"])
        record("append-only proof identical (ii)", proof_ok,
               f"{len(cov['append_only_proof'])} windows, all identical={proof_ok}")

        # independent pre/post hash over pre-existing window
        pc2 = sqlite3.connect(dbD)
        pcur2 = pc2.cursor()
        post_hash, post_n = window_row_hash(pcur2, top_aid, top_min, top_max)
        pc2.close()
        record("pre/post window hash identical (ii)",
               (post_hash == pre_hash and post_n == pre_n),
               f"pre={pre_hash[:12]} post={post_hash[:12]} n {pre_n}->{post_n}")

        # rows actually appended (5 new)
        added = next(r["rows_added"] for r in cov["per_symbol"]
                     if r["symbol"] == top_sym)
        record("new candles appended", added == 5, f"rows_added={added} (expected 5)")

        # idempotent row skipped, not double-inserted
        idem = cov["total_idempotent_skipped"]
        record("idempotent re-supply skipped", idem >= 1,
               f"idempotent_skipped={idem}")

        # report schema stable (files written + required keys)
        rj = workdir / "reportD" / "coverage_report.json"
        rm = workdir / "reportD" / "coverage_report.md"
        required = {"task", "per_symbol", "append_only_proof", "file_stats",
                    "window_start_ts", "total_rows_added", "n_resolved"}
        schema_ok = rj.exists() and rm.exists() and required.issubset(cov.keys())
        record("report schema stable + written (i)", schema_ok,
               f"json={rj.exists()} md={rm.exists()}")

        # integrity ran clean (no raise) — implied by reaching here
        record("integrity checks passed", True, "no IntegrityError")
    except Exception as e:
        import traceback
        traceback.print_exc()
        record("happy path", False, f"{type(e).__name__}: {e}")

    # ---- Scenario E: interior gap-fill SUCCEEDS + append-only proof PASSES ----
    # (FAILS pre-fix: old whole-window hash false-fires on the added interior row.)
    try:
        dbE = make_db_copy("E")
        bkE = make_backup(dbE, fresh=True)
        dumpE = workdir / "dumpE"
        # New key at a NON-grid interior timestamp (< existing max), i.e. a genuine
        # gap-fill. Re-supply real existing rows around it so the incoming range
        # overlaps pre-existing data and the subset proof actually exercises.
        interior_ts = first_rows[0][0] + 30  # between grid points -> did not exist
        assert interior_ts < top_max, "interior_ts must be inside the window"
        interior_row = (interior_ts, first_rows[0][1], first_rows[0][2],
                        first_rows[0][3], first_rows[0][4], first_rows[0][5],
                        first_rows[0][6])
        resupply = [(r[0], r[1], r[2], r[3], r[4], r[5], r[6]) for r in first_rows]
        _write_dump(dumpE, sym_to_filename(top_sym), resupply + [interior_row])
        covE = run_backfill(dbE, [str(dumpE)], str(uni_path), backup_path=bkE,
                            report_dir=str(workdir / "reportE"), write_report=False)
        addedE = next(r["rows_added"] for r in covE["per_symbol"]
                      if r["symbol"] == top_sym)
        proofE_ok = bool(covE["append_only_proof"]) and \
            all(p["identical"] for p in covE["append_only_proof"])
        gapfill_ok = (addedE == 1) and proofE_ok
        record("interior gap-fill succeeds + proof passes (E)", gapfill_ok,
               f"rows_added={addedE} (expect 1), proof_identical={proofE_ok}, "
               f"interior_ts<max={interior_ts < top_max}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        record("interior gap-fill succeeds + proof passes (E)", False,
               f"{type(e).__name__}: {e}")

    # ---- Scenario F: subset proof FIRES on a simulated mutation ----
    # A plain INSERT cannot mutate a PK row, so exercise the proof invariant
    # directly: capture pre-existing values, mutate one in a copy, re-read, and
    # assert the subset comparison flags it (this is what run_backfill asserts).
    try:
        dbF = make_db_copy("F")
        cf = sqlite3.connect(dbF)
        cfc = cf.cursor()
        pre_map = {r[0]: r[1:] for r in first_rows}
        mutate_ts = first_rows[0][0]
        cfc.execute("UPDATE ohlcv SET close = close + 123.45 "
                    "WHERE asset_id = ? AND timestamp = ?", (top_aid, mutate_ts))
        cf.commit()
        lo, hi = min(pre_map), max(pre_map)
        cfc.execute(
            "SELECT timestamp, open, high, low, close, volume, trade_count "
            "FROM ohlcv WHERE asset_id = ? AND timestamp BETWEEN ? AND ?",
            (top_aid, lo, hi))
        post = {r[0]: r[1:] for r in cfc.fetchall()}
        cf.close()
        fired = any(ts not in post or _values_differ(post[ts], pv)
                    for ts, pv in pre_map.items())
        detected_ts = mutate_ts in post and _values_differ(post[mutate_ts],
                                                            pre_map[mutate_ts])
        record("subset proof fires on mutation (F)", fired and detected_ts,
               f"fired={fired} detected_mutated_key={detected_ts}")
    except Exception as e:
        record("subset proof fires on mutation (F)", False,
               f"{type(e).__name__}: {e}")

    # ---- Scenario G/I: bad-OHLC/zero-price incoming aborts pre-commit +
    #                    atomic rollback leaves the DB row-count identical ----
    try:
        dbG = make_db_copy("G")
        bkG = make_backup(dbG, fresh=True)
        rows_before = total_rows(dbG)
        dumpG = workdir / "dumpG"
        # A brand-NEW key (reaches insert) carrying a zero-price candle. Parse is
        # structural-only now, so it flows to integrity_check, which must bite.
        # ts must not collide with the good new_rows (top_max+60..+300).
        bad_row = (top_max + 600, 0.0, 0.0, 0.0, 0.0, 5.0, 3)
        _write_dump(dumpG, sym_to_filename(top_sym),
                    new_rows + [bad_row])  # some good + one bad
        raised = None
        try:
            run_backfill(dbG, [str(dumpG)], str(uni_path), backup_path=bkG,
                         report_dir=str(workdir / "reportG"), write_report=False)
        except IntegrityError as e:
            raised = e
        rows_after = total_rows(dbG)
        top_after = None
        cg = sqlite3.connect(dbG)
        top_after = cg.execute("SELECT COUNT(*) FROM ohlcv WHERE asset_id = ?",
                               (top_aid,)).fetchone()[0]
        cg.close()
        record("zero-price incoming aborts pre-commit (I)", raised is not None,
               "IntegrityError raised" if raised else "did NOT raise")
        record("guard failure leaves DB row-count identical (G)",
               rows_after == rows_before,
               f"before={rows_before} after={rows_after} (top_aid rows {top_after})")
    except Exception as e:
        import traceback
        traceback.print_exc()
        record("zero-price abort + atomic rollback (G/I)", False,
               f"{type(e).__name__}: {e}")

    # ---- Scenario H: backup-gate bypasses each REFUSE ----
    # (a) self-reference: --backup IS the DB itself
    try:
        dbH = make_db_copy("H")
        dumpH = workdir / "dumpH"
        _write_dump(dumpH, sym_to_filename(top_sym), new_rows)
        run_backfill(dbH, [str(dumpH)], str(uni_path), backup_path=dbH,
                     report_dir=str(workdir / "reportH"), write_report=False)
        record("backup-gate refuses self-reference (H)", False, "did NOT raise")
    except BackupGateError:
        record("backup-gate refuses self-reference (H)", True, "BackupGateError")
    except Exception as e:
        record("backup-gate refuses self-reference (H)", False,
               f"wrong error: {type(e).__name__}: {e}")
    # (b) 0-byte backup, touched newer
    try:
        dbH2 = make_db_copy("H2")
        bkH2 = dbH2 + ".backup"
        open(bkH2, "w").close()  # 0 bytes
        _touch_newer(bkH2, dbH2)
        run_backfill(dbH2, [str(dumpH)], str(uni_path), backup_path=bkH2,
                     report_dir=str(workdir / "reportH2"), write_report=False)
        record("backup-gate refuses 0-byte backup (H)", False, "did NOT raise")
    except BackupGateError:
        record("backup-gate refuses 0-byte backup (H)", True, "BackupGateError")
    except Exception as e:
        record("backup-gate refuses 0-byte backup (H)", False,
               f"wrong error: {type(e).__name__}: {e}")
    # (c) stale/smaller backup rescued by a `touch` (fresh mtime, short content)
    try:
        dbH3 = make_db_copy("H3")
        bkH3 = dbH3 + ".backup"
        with open(bkH3, "wb") as f:
            f.write(b"SQLite format 3\x00")  # tiny, < db size
        _touch_newer(bkH3, dbH3)  # touch makes mtime fresh
        run_backfill(dbH3, [str(dumpH)], str(uni_path), backup_path=bkH3,
                     report_dir=str(workdir / "reportH3"), write_report=False)
        record("backup-gate refuses touched-stale backup (H)", False, "did NOT raise")
    except BackupGateError:
        record("backup-gate refuses touched-stale backup (H)", True, "BackupGateError")
    except Exception as e:
        record("backup-gate refuses touched-stale backup (H)", False,
               f"wrong error: {type(e).__name__}: {e}")

    # ---- Scenario J: leading-slash + BTC-spelled dump forms ingest ----
    # (FAILS pre-fix: _symbol_variants cannot round-trip these DB spellings.)
    try:
        if not (slash_sym and btc_sym):
            record("spelling-form dumps ingest (J)", False,
                   "fixture lacks /USDTEUR or BTCGBP assets")
        else:
            dbJ = make_db_copy("J")
            bkJ = make_backup(dbJ, fresh=True)
            uni_j = workdir / "universe_J.json"
            with open(uni_j, "w") as f:
                json.dump({"frame_symbols": [slash_sym[0], btc_sym[0]],
                           "n_assets": 2, "source": "selftest spelling-form"}, f)
            dumpJ = workdir / "dumpJ"
            base = 1760000000
            jrows = [(base + i * 60, 100.0, 101.0, 99.0, 100.5, 5.0, 3)
                     for i in range(4)]
            # DB '/USDTEUR' -> Kraken altname dump file 'USDTEUR_1.csv'
            _write_dump(dumpJ, "USDTEUR_1.csv", jrows)
            # DB 'BTCGBP'   -> dump file 'BTCGBP_1.csv' (parses to XBT/GBP)
            _write_dump(dumpJ, "BTCGBP_1.csv", jrows)
            covJ = run_backfill(dbJ, [str(dumpJ)], str(uni_j), backup_path=bkJ,
                                report_dir=str(workdir / "reportJ"), write_report=False)
            slash_added = next((r["rows_added"] for r in covJ["per_symbol"]
                                if r["symbol"] == slash_sym[0]), 0)
            btc_added = next((r["rows_added"] for r in covJ["per_symbol"]
                              if r["symbol"] == btc_sym[0]), 0)
            ok = (covJ["files_ingested"] == 2 and covJ["files_unmatched"] == 0
                  and slash_added == 4 and btc_added == 4)
            record("spelling-form dumps ingest (J)", ok,
                   f"files_ingested={covJ['files_ingested']} "
                   f"unmatched={covJ['files_unmatched']} "
                   f"{slash_sym[0]}+{slash_added} {btc_sym[0]}+{btc_added}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        record("spelling-form dumps ingest (J)", False,
               f"{type(e).__name__}: {e}")

    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    print(f"\n[fixture-selftest] {passed}/{total} checks PASS")
    # keep workdir for inspection only on failure
    if passed == total:
        shutil.rmtree(workdir, ignore_errors=True)
    else:
        print(f"[fixture-selftest] workdir retained for inspection: {workdir}")
    return 0 if passed == total else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="TASK_858 guarded append-only minute-table backfill")
    sub = parser.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="run the guarded backfill (writes to --db)")
    pr.add_argument("--db", default="trading_data.db", help="target DB (source of record)")
    pr.add_argument("--backup", required=True,
                    help="path to a FRESH backup of --db (mandatory backup gate)")
    pr.add_argument("--dump-dir", nargs="+", required=True,
                    help="Kraken quarterly OHLCVT dump dir(s) ({SYMBOL}_1.csv)")
    pr.add_argument("--universe", default=DEFAULT_UNIVERSE,
                    help="frame provenance JSON with 'frame_symbols'")
    pr.add_argument("--report-dir", default=DEFAULT_REPORT_DIR)
    pr.add_argument("--max-conflict-samples", type=int, default=10)
    pr.add_argument("--tail-topup", action="store_true",
                    help="after dumps, run the live Kraken OHLC fetcher for the "
                         "<=12h tail (opt-in; hits public API; never in factory loop)")
    pr.add_argument("--tail-days", type=int, default=1)

    ps = sub.add_parser("fixture-selftest",
                        help="acceptance (i)+(ii): guards + append-only vs fixture copy")
    ps.add_argument("--fixture", default="fixtures/trading_fixture.db")

    args = parser.parse_args(argv)

    if args.cmd == "fixture-selftest":
        return fixture_selftest(args.fixture)

    if args.cmd == "run":
        cov = run_backfill(
            db_path=args.db,
            dump_dirs=args.dump_dir,
            universe_path=args.universe,
            backup_path=args.backup,
            report_dir=args.report_dir,
            max_conflict_samples=args.max_conflict_samples,
        )
        print(f"[backfill] {cov['total_rows_added']:,} rows added across "
              f"{cov['files_ingested']} files; report -> {args.report_dir}")
        if args.tail_topup:
            print("[backfill] running tail top-up via live Kraken fetcher...")
            run_tail_topup(args.db, args.universe, days_back=args.tail_days)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
