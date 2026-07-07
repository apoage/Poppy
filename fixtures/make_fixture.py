#!/usr/bin/env python
"""Regenerate fixtures/trading_fixture.db from trading_data.db (read-only source).

Contents: full `assets` + `ohlcv_daily` panel + `fetch_log`, plus a 90-day 1-min
`ohlcv` slice for the 8 assets with the longest minute coverage. ~50MB, instant-load.
This is the dark-factory verification fixture (`.dark-factory/config.json paths.fixture`).
The fixture db itself is gitignored (*.db rule) — regenerate with this script.
"""
import sqlite3, os, sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "trading_data.db"
DST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trading_fixture.db")
DAYS = 90
TOP_N = 8

if os.path.exists(DST):
    os.remove(DST)
src = sqlite3.connect(f"file:{SRC}?mode=ro", uri=True)

top = [r[0] for r in src.execute(
    "select asset_id from fetch_log where data_type='ohlcv_1' "
    "order by (newest_timestamp-oldest_timestamp) desc limit ?", (TOP_N,))]
if not top:
    top = [r[0] for r in src.execute(
        "select asset_id from ohlcv_daily group by asset_id order by count(*) desc limit ?", (TOP_N,))]
cutoff = src.execute("select max(timestamp) - ?*86400 from ohlcv", (DAYS,)).fetchone()[0]

dst = sqlite3.connect(DST)
for (sql,) in src.execute("select sql from sqlite_master where type='table' and sql is not null"):
    dst.execute(sql)

for t in ("assets", "ohlcv_daily", "fetch_log"):
    ncols = len(src.execute(f"pragma table_info({t})").fetchall())
    dst.executemany(f"insert into {t} values ({','.join('?' * ncols)})",
                    src.execute(f"select * from {t}"))

cur = src.execute(
    f"select * from ohlcv where timestamp>=? and asset_id in ({','.join('?' * len(top))})",
    [cutoff] + top)
ncols = len(cur.description)
dst.executemany(f"insert into ohlcv values ({','.join('?' * ncols)})", cur)
dst.commit()

n_min = dst.execute("select count(*) from ohlcv").fetchone()[0]
n_day = dst.execute("select count(*) from ohlcv_daily").fetchone()[0]
dst.execute("vacuum")
dst.close()
print(f"fixture: top{TOP_N}={top} minute_rows={n_min} daily_rows={n_day} "
      f"size={os.path.getsize(DST)/1e6:.1f}MB -> {DST}")
