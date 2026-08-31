#!/usr/bin/env python3
"""Kraken OHLC pull habit (TASK_899 scope 4) — runs on the apothecary VPS via cron.

Public API only, stdlib only, no credentials. Two jobs:
  default        : daily candles (since now-30d, seam overlap) + minute tails
                   (the rolling 720-candle window) for the 56 canon pairs.
  --dump-watch   : fetch Kraken's OHLCVT downloads page once a day and alert
                   (DUMP_FOUND file + loud log line) when Q2/Q3 2026 zips appear.

Data lands in ~/kraken_pull/staging/<UTC-date>/{daily,minute_tail}/ + manifest.json.
Retention: last 14 staging dirs. The workstation syncs with
tools/pull_from_apothecary.sh whenever it is awake. Never writes anywhere else.
Planted 2026-09-01 by the planner; canon list from runs/858_backfill/api_2026-08-31.
"""
import json, os, pathlib, sys, time, urllib.request, datetime

HOME = pathlib.Path(__file__).parent
STAGING = HOME / "staging"
LOCK = HOME / ".lock"
LOG = HOME / "runs.log"

# canon minute-table symbols (DB spelling); API spelling = slash stripped
SYMBOLS = [
    "AAVE/USD", "ADA/EUR", "ADA/USD", "AKT/USD", "ALGO/USD", "ARB/USD", "ATOM/USD",
    "AVAX/EUR", "BCH/EUR", "BCH/USD", "BTCCAD", "BTCCHF", "BTCGBP", "DOT/EUR",
    "DOT/USDT", "ETH/USD", "ETHBTC", "FET/USD", "GALA/USD", "GRT/USD", "ICP/USD",
    "IMX/USD", "INJ/USD", "KSM/USD", "LDO/USD", "LINK/USD", "LTC/EUR", "LTC/USD",
    "LTC/USDT", "LTCGBP", "LUNA/USD", "NEAR/USD", "PEPE/EUR", "PEPE/USD", "RUNE/USD",
    "SHIB/EUR", "SHIB/USD", "SOLGBP", "STX/USD", "TIA/USD", "TRX/EUR", "TRX/USD",
    "XBT/EUR", "XBT/USD", "XBT/USDC", "XBT/USDT", "XDG/EUR", "XDG/USD", "XLM/USD",
    "XMR/EUR", "XMR/USD", "XRP/EUR", "XRP/USD", "XRP/USDT", "XRPBTC", "XRPGBP",
]

DUMP_WATCH_URLS = [
    "https://support.kraken.com/hc/en-us/articles/360000477842-Downloadable-historical-OHLCVT-data",
]
DUMP_PATTERNS = ("Q2_2026", "Q3_2026")


def log(msg):
    line = f"{datetime.datetime.now(datetime.timezone.utc).isoformat()} {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def fetch(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": "trader-data-habit/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def ohlc(api_sym, interval, since=None):
    url = f"https://api.kraken.com/0/public/OHLC?pair={api_sym}&interval={interval}"
    if since:
        url += f"&since={since}"
    d = json.loads(fetch(url))
    if d.get("error"):
        return None, ";".join(d["error"])
    key = [k for k in d["result"] if k != "last"]
    return (d["result"][key[0]] if key else []), None


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("time,open,high,low,close,vwap,volume,count\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")


def pull():
    day = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    out = STAGING / day
    since_daily = int(time.time()) - 30 * 86400
    manifest = {"date": day, "pulled": [], "failed": []}
    for db_sym in SYMBOLS:
        api_sym = db_sym.replace("/", "")
        daily, e1 = ohlc(api_sym, 1440, since=since_daily)
        time.sleep(1.0)
        tail, e2 = ohlc(api_sym, 1)
        time.sleep(1.0)
        if e1 or e2 or daily is None or tail is None:
            manifest["failed"].append({"symbol": db_sym, "reason": str(e1 or e2)})
            continue
        safe = db_sym.replace("/", "_")
        write_csv(out / "daily" / f"{safe}.csv", daily)
        write_csv(out / "minute_tail" / f"{safe}.csv", tail)
        manifest["pulled"].append({
            "symbol": db_sym, "daily_rows": len(daily), "daily_last": daily[-1][0],
            "minute_rows": len(tail), "minute_last": tail[-1][0] if tail else None,
        })
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    log(f"pull {day}: {len(manifest['pulled'])} ok, {len(manifest['failed'])} failed")
    # retention: keep the newest 14 staging dirs
    dirs = sorted(STAGING.iterdir()) if STAGING.exists() else []
    for old in dirs[:-14]:
        import shutil; shutil.rmtree(old, ignore_errors=True)
    return 1 if manifest["failed"] and not manifest["pulled"] else 0


def dump_watch():
    import re
    found = []
    for url in DUMP_WATCH_URLS:
        try:
            html = fetch(url).decode("utf-8", "replace")
        except Exception as e:
            log(f"dump-watch: fetch failed {url}: {e}")
            continue
        for pat in DUMP_PATTERNS:
            if pat in html:
                links = re.findall(r'https?://[^\s"\'<>]+\.zip', html) or ["(link extraction failed - visit page)"]
                found.append((url, pat, links))
        time.sleep(2.0)
    if found:
        day = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
        msg = "\n".join(f"{u}\n  {p}: {l}" for u, p, l in found)
        (HOME / f"DUMP_FOUND_{day}.txt").write_text(msg + "\n")
        log(f"DUMP-WATCH: *** NEW DUMPS VISIBLE *** {msg}")
    else:
        log("dump-watch: no Q2/Q3 2026 zips visible yet")
    return 0


def main():
    if LOCK.exists():
        log("another run holds the lock; skipping")
        return 0
    LOCK.write_text(str(os.getpid()))
    try:
        return dump_watch() if "--dump-watch" in sys.argv else pull()
    finally:
        LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
