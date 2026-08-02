#!/usr/bin/env python
"""Regenerate fixtures/trading_fixture.db from trading_data.db (read-only source).

Contents: full `assets` + `ohlcv_daily` panel + `fetch_log`, plus a 90-day 1-min
`ohlcv` slice for the 8 assets with the longest minute coverage. ~50MB, instant-load.
This is the dark-factory verification fixture (`.dark-factory/config.json paths.fixture`).
The fixture db itself is gitignored (*.db rule) — regenerate with this script.

TASK_868 (D31 scope 4): three SYNTHETIC minute symbols are appended after the copy — the
twin-identity pair TWX/USD (90001) + TWXUSD (90002) with an orphaned span, and HOLE/USD
(90003) with a multi-bar interior hole — all on 2025-09-10..2025-09-12, i.e. BEFORE the
copied slice and outside every smoke window, so no existing panel or universe ranking moves.
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
# TASK_868: the slice window is PINNED, not max-relative. trading_data.db keeps accruing
# minute rows (its max had already moved ~3.5 months past 2025-12-12 by 2026-08), so the old
# `max(timestamp) - DAYS*86400` cutoff silently re-based every fixture panel on each regen.
# The verification suites pin their windows inside 2025-09-13..2025-12-12; re-pin the slice
# deliberately here, never implicitly.
from datetime import datetime as _dt
SLICE_END = "2025-12-12"            # inclusive local-calendar end day of the minute slice
end_ts = int(_dt.strptime(SLICE_END, "%Y-%m-%d").timestamp()) + 86399
cutoff = end_ts - DAYS * 86400 + 1

dst = sqlite3.connect(DST)
for (sql,) in src.execute("select sql from sqlite_master where type='table' and sql is not null"):
    dst.execute(sql)

for t in ("assets", "ohlcv_daily", "fetch_log"):
    ncols = len(src.execute(f"pragma table_info({t})").fetchall())
    dst.executemany(f"insert into {t} values ({','.join('?' * ncols)})",
                    src.execute(f"select * from {t}"))

cur = src.execute(
    f"select * from ohlcv where timestamp>=? and timestamp<=? "
    f"and asset_id in ({','.join('?' * len(top))})",
    [cutoff, end_ts] + top)
ncols = len(cur.description)
dst.executemany(f"insert into ohlcv values ({','.join('?' * ncols)})", cur)

# ---- TASK_868 (D31 scope 4): synthetic identity/hole phenomena, minute table only ----------
# Both mechanisms the task repairs must be fixture-provable forever. All synthetic rows land
# BEFORE the copied slice (2025-09-10..2025-09-12), so every smoke window's coverage-ranked
# universe and every existing panel is untouched (zero synthetic rows in any smoke window).
# The ohlcv_daily table is NOT touched (daily paths stay bit-identical).
from datetime import datetime
_T868 = int(datetime.strptime("2025-09-10", "%Y-%m-%d").timestamp())
def _t868_rows(aid, minutes, price):
    return [(aid, _T868 + 60 * m, price, price, price, price, 1.0, 1) for m in minutes]

# (i) a TWIN-IDENTITY pair with an orphaned span (the D31 ingest-split pattern):
# original prints minutes 0..1439 then goes dark; the twin (spelling variant, added "later")
# prints minutes 720..2879 — agreeing in the 720..1439 overlap, and carrying the ORPHANED
# span 1440..2879 the original lacks. The loader's alias union must read one continuous
# 2880-minute series; the agreement gate must pass (and refuse a tampered overlap).
TWIN_A_ID, TWIN_B_ID = 90001, 90002
dst.execute("insert into assets (id, symbol) values (?, ?)", (TWIN_A_ID, "TWX/USD"))
dst.execute("insert into assets (id, symbol) values (?, ?)", (TWIN_B_ID, "TWXUSD"))
dst.executemany("insert into ohlcv values (?,?,?,?,?,?,?,?)",
                _t868_rows(TWIN_A_ID, range(0, 1440), 100.0))
dst.executemany("insert into ohlcv values (?,?,?,?,?,?,?,?)",
                _t868_rows(TWIN_B_ID, range(720, 2880), 100.0))

# (ii) a genuine MULTI-BAR INTERIOR HOLE (run-23 rec (C)): prints every minute of the same
# two days EXCEPT minutes 1000..1599 (a 600-minute hole), re-anchoring +60% on resumption —
# the cell class D31's causal open-gate mask force-HOLDs.
HOLE_ID = 90003
dst.execute("insert into assets (id, symbol) values (?, ?)", (HOLE_ID, "HOLE/USD"))
dst.executemany("insert into ohlcv values (?,?,?,?,?,?,?,?)",
                _t868_rows(HOLE_ID, range(0, 1000), 100.0))
dst.executemany("insert into ohlcv values (?,?,?,?,?,?,?,?)",
                _t868_rows(HOLE_ID, range(1600, 2880), 160.0))

dst.commit()

n_min = dst.execute("select count(*) from ohlcv").fetchone()[0]
n_day = dst.execute("select count(*) from ohlcv_daily").fetchone()[0]
dst.execute("vacuum")
dst.close()
print(f"fixture: top{TOP_N}={top} minute_rows={n_min} daily_rows={n_day} "
      f"size={os.path.getsize(DST)/1e6:.1f}MB -> {DST}")
