#!/usr/bin/env python3
"""
Daily-bar fetcher: Kraken public REST -> ohlcv_daily table in trading_data.db.

Standalone script — no imports from mcts_trader. CPU/network only, RAM-trivial.
Fills the daily-bar gap for a daily-trend strategy using Kraken's public
OHLC endpoint (interval=1440), incremental + resumable + rate-limited.

See tasks/TASK_820_SHIP_TREND_CTA.md, section "Daily-bar store + fetch contract".

Usage:
    python tools/fetch_daily_kraken.py --limit 5          # first 5 mapped assets (testing)
    python tools/fetch_daily_kraken.py --asset-ids 61,359  # explicit DB asset ids
    python tools/fetch_daily_kraken.py --all               # full mapped universe
"""

import argparse
import json
import logging
import os
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

LOG = logging.getLogger("fetch_daily_kraken")

KRAKEN_BASE = "https://api.kraken.com"
ASSETPAIRS_URL = f"{KRAKEN_BASE}/0/public/AssetPairs"
OHLC_URL = f"{KRAKEN_BASE}/0/public/OHLC"

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_DB_PATH = os.path.join(REPO_ROOT, "trading_data.db")

# Kraken's AssetPairs "quote" field for USD-quoted pairs. Historically always
# "ZUSD"; some newer pair listings could plausibly use a bare "USD" — keep both.
USD_QUOTES = {"ZUSD", "USD"}

RATE_LIMIT_SECONDS = 1.0
MAX_RETRIES = 6
BACKOFF_BASE = 5.0
BACKOFF_MAX = 60.0
RETRYABLE_HTTP_STATUS = {429, 500, 502, 503, 504, 520, 521, 522, 524, 525, 526, 529}

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS ohlcv_daily (
    asset_id INTEGER NOT NULL,
    timestamp INTEGER NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume REAL,
    trade_count INTEGER,
    source TEXT,
    PRIMARY KEY (asset_id, timestamp)
)
"""


# --------------------------------------------------------------------------- #
# HTTP with retry/backoff
# --------------------------------------------------------------------------- #

def http_get_json(url, params, timeout=30):
    """GET url?params, parse JSON. Retries on 429/5xx/network errors with
    exponential backoff. Returns the parsed dict (may contain a non-empty
    'error' list on unrecoverable Kraken-side errors, e.g. unknown pair)."""
    qs = urllib.parse.urlencode(params)
    full_url = f"{url}?{qs}"
    backoff = BACKOFF_BASE
    last_err = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(
                full_url, headers={"User-Agent": "trader-daily-fetch/1.0"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read()
            data = json.loads(body)
            errors = data.get("error") or []
            if errors:
                msg = " ".join(errors)
                if "Too many requests" in msg or "Rate limit" in msg.lower():
                    last_err = RuntimeError(f"rate limited: {msg}")
                else:
                    # Non-retryable Kraken-side error (bad pair, etc.) - surface it.
                    return data
            else:
                return data
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code not in RETRYABLE_HTTP_STATUS:
                LOG.warning("Non-retryable HTTP %s for %s", e.code, full_url)
                return {"error": [f"HTTP {e.code}"]}
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            last_err = e

        LOG.warning(
            "Request attempt %d/%d failed (%s); backing off %.1fs",
            attempt, MAX_RETRIES, last_err, backoff,
        )
        time.sleep(backoff)
        backoff = min(backoff * 1.5, BACKOFF_MAX)

    LOG.error("Max retries exhausted for %s params=%s: %s", url, params, last_err)
    return {"error": [f"max retries exhausted: {last_err}"]}


def fetch_asset_pairs():
    LOG.info("Fetching Kraken AssetPairs (one-time)...")
    data = http_get_json(ASSETPAIRS_URL, {})
    if data.get("error"):
        raise RuntimeError(f"AssetPairs fetch failed: {data['error']}")
    return data["result"]


def fetch_daily_ohlc(pair_key, since=None):
    """Fetch interval=1440 OHLC series for a Kraken pair key. Returns
    (rows, error) where rows includes the trailing UNFINISHED current day
    (caller must drop it)."""
    params = {"pair": pair_key, "interval": 1440}
    if since is not None:
        params["since"] = since
    data = http_get_json(OHLC_URL, params)
    if data.get("error"):
        return None, data["error"]
    result = data.get("result", {})
    series = [v for k, v in result.items() if k != "last"]
    if not series:
        return [], None
    return series[0], None


# --------------------------------------------------------------------------- #
# Kraken pair <-> assets-table symbol resolution
# --------------------------------------------------------------------------- #

def usd_pairs(all_pairs):
    return {k: v for k, v in all_pairs.items() if v.get("quote") in USD_QUOTES}


def _btc_xbt_variants(name):
    out = {name}
    if "XBT" in name:
        out.add(name.replace("XBT", "BTC"))
    if "BTC" in name:
        out.add(name.replace("BTC", "XBT"))
    return out


def candidate_names(pair_key, info):
    """All plausible `assets.symbol` spellings for a Kraken pair: the pair
    key itself, altname, wsname, each with/without the wsname '/' separator,
    and with XBT<->BTC swapped (Kraken uses XBT/XXBT internally; the DB and
    common usage mix XBT and BTC spellings)."""
    bases = set()
    for raw in (pair_key, info.get("altname", ""), info.get("wsname", "")):
        if not raw:
            continue
        bases.add(raw)
        bases.add(raw.replace("/", ""))
    names = set()
    for b in bases:
        names |= _btc_xbt_variants(b)
    return names


def load_db_symbol_map(conn):
    cur = conn.execute("SELECT id, symbol FROM assets")
    return {sym: aid for aid, sym in cur.fetchall()}


def ohlcv_1m_row_count(conn, asset_id):
    cur = conn.execute("SELECT COUNT(*) FROM ohlcv WHERE asset_id=?", (asset_id,))
    return cur.fetchone()[0]


def resolve_universe(conn, all_pairs):
    """Map Kraken USD-quoted pairs to `assets` rows.

    Returns (resolved, skipped_unmapped, skipped_duplicates):
      resolved            - list of dicts: asset_id, symbol, pair_key, altname, wsname
      skipped_unmapped     - [(asset_id, symbol)] USD-looking DB rows not found on
                              Kraken at all -> delisted, PIT universe handles them
      skipped_duplicates   - [(asset_id, symbol, pair_key, reason)] DB rows that
                              duplicate an already-chosen asset for the same Kraken
                              pair (the DB has some near-empty duplicate rows
                              alongside the real historical ones; we keep whichever
                              row already has more 1-min ohlcv history)
    """
    sym_to_id = load_db_symbol_map(conn)
    id_to_symbol = {v: k for k, v in sym_to_id.items()}
    pairs = usd_pairs(all_pairs)

    resolved = {}
    skipped_duplicates = []
    matched_asset_ids = set()

    for pair_key, info in sorted(pairs.items()):
        names = candidate_names(pair_key, info)
        hit_ids = sorted({sym_to_id[n] for n in names if n in sym_to_id})
        if not hit_ids:
            continue

        if len(hit_ids) > 1:
            counts = [(aid, ohlcv_1m_row_count(conn, aid)) for aid in hit_ids]
            counts.sort(key=lambda t: (-t[1], t[0]))
            chosen_id = counts[0][0]
            for aid, cnt in counts[1:]:
                skipped_duplicates.append((
                    aid, id_to_symbol[aid], pair_key,
                    f"duplicate row for Kraken pair {pair_key}; kept asset_id="
                    f"{chosen_id} ({id_to_symbol[chosen_id]}, more 1m history)",
                ))
        else:
            chosen_id = hit_ids[0]

        matched_asset_ids.add(chosen_id)
        resolved[chosen_id] = {
            "asset_id": chosen_id,
            "symbol": id_to_symbol[chosen_id],
            "pair_key": pair_key,
            "altname": info.get("altname"),
            "wsname": info.get("wsname"),
        }

    dup_ids = {d[0] for d in skipped_duplicates}
    skipped_unmapped = [
        (aid, sym) for sym, aid in sym_to_id.items()
        if aid not in matched_asset_ids
        and aid not in dup_ids
        and (sym.endswith("USD") or sym.endswith("/USD"))
    ]

    return (
        sorted(resolved.values(), key=lambda r: r["asset_id"]),
        sorted(skipped_unmapped),
        skipped_duplicates,
    )


# --------------------------------------------------------------------------- #
# ohlcv_daily read/write
# --------------------------------------------------------------------------- #

def utc_midnight_ts(dt=None):
    dt = dt or datetime.now(timezone.utc)
    return int(datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc).timestamp())


def get_existing_max_ts(conn, asset_id):
    cur = conn.execute(
        "SELECT MAX(timestamp) FROM ohlcv_daily WHERE asset_id=?", (asset_id,)
    )
    row = cur.fetchone()
    return row[0] if row and row[0] is not None else None


def insert_daily_rows(conn, asset_id, rows):
    """INSERT OR IGNORE the given Kraken OHLC rows. Returns count actually
    inserted (rows already present are silently ignored, not counted)."""
    if not rows:
        return 0
    before = conn.total_changes
    conn.executemany(
        "INSERT OR IGNORE INTO ohlcv_daily "
        "(asset_id, timestamp, open, high, low, close, volume, trade_count, source) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        [
            (
                asset_id, int(r[0]), float(r[1]), float(r[2]), float(r[3]),
                float(r[4]), float(r[6]), int(r[7]), "kraken_rest",
            )
            for r in rows
        ],
    )
    conn.commit()
    return conn.total_changes - before


def process_asset(conn, asset, rate_limit_seconds):
    """Fetch + insert daily bars for one asset. Returns a result dict and
    whether a network request was actually made (for rate-limit pacing)."""
    asset_id = asset["asset_id"]
    symbol = asset["symbol"]
    pair_key = asset["pair_key"]
    pair_display = asset.get("wsname") or pair_key

    today_mid = utc_midnight_ts()
    yesterday_mid = today_mid - 86400

    existing_max = get_existing_max_ts(conn, asset_id)
    if existing_max is not None and existing_max >= yesterday_mid:
        newest_str = datetime.fromtimestamp(existing_max, tz=timezone.utc).strftime("%Y-%m-%d")
        LOG.info("%-12s %-14s  up-to-date (newest=%s) — skip", symbol, pair_display, newest_str)
        return {"asset_id": asset_id, "symbol": symbol, "status": "up_to_date",
                "inserted": 0, "newest": newest_str}, False

    since = existing_max if existing_max is not None else None
    rows, err = fetch_daily_ohlc(pair_key, since=since)

    if err:
        LOG.warning("%-12s %-14s  FETCH ERROR: %s", symbol, pair_display, err)
        return {"asset_id": asset_id, "symbol": symbol, "status": "error",
                "error": str(err), "inserted": 0, "newest": None}, True

    # Drop the trailing UNFINISHED current day (Kraken always returns it last).
    if rows:
        rows = rows[:-1]
    # Belt-and-suspenders vs. `since` boundary semantics: never store a row
    # we already have, and never store "today" even if boundary math differs.
    if since is not None:
        rows = [r for r in rows if int(r[0]) > since]
    rows = [r for r in rows if int(r[0]) < today_mid]

    inserted = insert_daily_rows(conn, asset_id, rows)

    new_max = get_existing_max_ts(conn, asset_id)
    newest_str = (
        datetime.fromtimestamp(new_max, tz=timezone.utc).strftime("%Y-%m-%d")
        if new_max else "n/a"
    )
    LOG.info("%-12s %-14s  +%-4d rows  newest=%s", symbol, pair_display, inserted, newest_str)
    return {"asset_id": asset_id, "symbol": symbol, "status": "ok",
            "inserted": inserted, "newest": newest_str}, True


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Fetch daily (interval=1440) OHLC bars from Kraken's public "
                     "REST API into trading_data.db:ohlcv_daily."
    )
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="Path to trading_data.db")
    parser.add_argument("--limit", type=int, default=None,
                         help="Process only the first N resolved assets (testing)")
    parser.add_argument("--asset-ids", default=None,
                         help="Comma-separated DB asset ids to restrict to")
    parser.add_argument("--all", action="store_true",
                         help="Process the entire mapped universe (default behavior "
                              "when neither --limit nor --asset-ids is given)")
    parser.add_argument("--rate-limit", type=float, default=RATE_LIMIT_SECONDS,
                         help="Seconds to sleep between Kraken requests")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    conn = sqlite3.connect(args.db)
    conn.execute(CREATE_TABLE_SQL)
    conn.commit()

    all_pairs = fetch_asset_pairs()
    LOG.info("Kraken reports %d total pairs", len(all_pairs))

    resolved, skipped_unmapped, skipped_duplicates = resolve_universe(conn, all_pairs)
    LOG.info(
        "Resolved %d assets to fetch; %d unmapped (delisted/not-on-kraken); "
        "%d duplicate DB rows skipped",
        len(resolved), len(skipped_unmapped), len(skipped_duplicates),
    )
    if skipped_unmapped:
        preview = ", ".join(f"{sym}(id={aid})" for aid, sym in skipped_unmapped[:20])
        LOG.info("Skip list preview (delisted/unmapped): %s%s",
                 preview, " ..." if len(skipped_unmapped) > 20 else "")

    if args.asset_ids:
        wanted = {int(x) for x in args.asset_ids.split(",") if x.strip()}
        targets = [a for a in resolved if a["asset_id"] in wanted]
    else:
        targets = resolved

    if args.limit is not None:
        targets = targets[: args.limit]

    LOG.info("Processing %d assets this run", len(targets))

    n_ok = n_err = n_skip = n_ins = 0
    for i, asset in enumerate(targets, 1):
        result, made_request = process_asset(conn, asset, args.rate_limit)
        if result["status"] == "ok":
            n_ok += 1
            n_ins += result["inserted"]
        elif result["status"] == "up_to_date":
            n_skip += 1
        else:
            n_err += 1
        if made_request and i < len(targets):
            time.sleep(args.rate_limit)

    conn.close()
    LOG.info(
        "Done. fetched=%d up_to_date=%d errors=%d rows_inserted=%d (this run)",
        n_ok, n_skip, n_err, n_ins,
    )


if __name__ == "__main__":
    main()
