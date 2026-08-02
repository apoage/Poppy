#!/usr/bin/env python3
"""TASK_868 gate (i) harvest — the 861 leg-1 frame read through the alias union (mode=ro).

Replicates the 865 smoke's cell-honesty measurement but with each frame symbol resolved to
its alias GROUP (mcts_trader.data.asset_aliases) and the per-bar real mask unioned across
member ids. Prints the numbers the updated leg_real_db_coverage pins. READ-ONLY.
"""
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
from mcts_trader.data.asset_aliases import (  # noqa: E402
    alias_agreement_gate, resolve_requested_symbols)

REAL_DB = os.path.join(_REPO, "trading_data.db")
REAL_FRAME_JSON = os.path.join(_REPO, "runs", "832_e5", "frame_provenance_imprint.json")
REAL_WINDOW = ("2021-01-01", "2025-10-04")
STEP = 1440
WARM = 60

prov = json.load(open(REAL_FRAME_JSON))
frame = prov.get("frame_symbols") or prov.get("symbols")
s_ts = int(datetime.strptime(REAL_WINDOW[0], "%Y-%m-%d").timestamp())
e_ts = int(datetime.strptime(REAL_WINDOW[1], "%Y-%m-%d").timestamp())
con = sqlite3.connect(f"file:{REAL_DB}?mode=ro", uri=True, timeout=60)
try:
    all_assets = con.execute("SELECT id, symbol FROM assets").fetchall()
    resolution, unknown = resolve_requested_symbols(all_assets, frame)
    id_groups = [r["asset_ids"] for r in resolution]
    print("unknown:", unknown)
    aliased = [(i, frame[i], g) for i, g in enumerate(id_groups) if len(g) > 1]
    print(f"aliased slots ({len(aliased)}):")
    for i, sym, g in aliased:
        print(f"  slot {i:>3} {sym:<12} -> asset_ids {g}")
    # agreement gate on the window (assert 0.00 overlap diff for the twins)
    disc = alias_agreement_gate(con, "ohlcv", id_groups, s_ts, e_ts)
    for d in disc:
        print(f"  gate: ids {d['asset_ids']} overlaps {d['overlaps']}")
    ids = sorted({a for g in id_groups for a in g})
    ph2 = ",".join("?" * len(ids))
    rows = con.execute(
        f"""SELECT asset_id, COUNT(*), MIN(timestamp), MAX(timestamp) FROM ohlcv
            WHERE asset_id IN ({ph2}) AND timestamp >= ? AND timestamp <= ?
              AND close IS NOT NULL AND close != 0
            GROUP BY asset_id""", (*ids, s_ts, e_ts)).fetchall()
    axis = np.fromiter((r[0] for r in con.execute(
        f"""SELECT DISTINCT timestamp FROM ohlcv
            WHERE asset_id IN ({ph2}) AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp""", (*ids, s_ts, e_ts))), dtype=np.int64)
    n_full = int(axis.size)
    n_panel = len(range(0, n_full, STEP))
    panel_ts = axis[::STEP]
    idx_of = {int(t): i for i, t in enumerate(axis)}
    col_of = {a: k for k, g in enumerate(id_groups) for a in g}
    real_bar = np.zeros((n_panel, len(frame)), dtype=bool)
    real_exact = np.zeros((n_panel, len(frame)), dtype=bool)
    per_id_inwin = {a: 0 for a in ids}
    for aid, t in con.execute(
        f"""SELECT asset_id, timestamp FROM ohlcv WHERE asset_id IN ({ph2})
            AND timestamp >= ? AND timestamp <= ? AND close IS NOT NULL AND close != 0""",
            (*ids, s_ts, e_ts)):
        per_id_inwin[aid] += 1
        i = idx_of.get(t)
        if i is None:
            continue
        c = col_of[aid]
        b = (i + STEP - 1) // STEP if i else 0
        if b < n_panel:
            real_bar[b, c] = True
        if i % STEP == 0 and i // STEP < n_panel:
            real_exact[i // STEP, c] = True
finally:
    con.close()

per_id = {r[0]: (r[1], r[2], r[3]) for r in rows}
fmt = lambda t: datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%d")  # noqa: E731
# per-slot union span: min first, max last over member ids with rows
filled = np.zeros_like(real_bar)
tailm = np.zeros_like(real_bar)
for k, g in enumerate(id_groups):
    spans = [(per_id[a][1], per_id[a][2]) for a in g if per_id.get(a, (0,))[0]]
    if not spans:
        continue
    mn = min(s[0] for s in spans)
    mx = max(s[1] for s in spans)
    filled[:, k] = panel_ts >= mn
    tailm[:, k] = panel_ts > mx
frozen = filled & ~real_bar
frozen_exact = filled & ~real_exact
D = slice(WARM, n_panel)
n_dec = n_panel - WARM
n_cells = n_dec * len(frame)
head = int((~filled[D]).sum())
fz_tail = int((frozen & tailm)[D].sum())
fz_int = int((frozen & ~tailm)[D].sum())
live = int((filled & real_bar)[D].sum())
pct = lambda n: 100.0 * n / max(n_cells, 1)  # noqa: E731
print(f"\nn_full {n_full}  n_panel {n_panel}  n_dec {n_dec}  cells {n_cells}")
print(f"INERT  {head}  ({pct(head):.4f}%)")
print(f"FROZEN {fz_int + fz_tail} ({pct(fz_int + fz_tail):.4f}%) = "
      f"{fz_int} INTERIOR ({pct(fz_int):.4f}%) + {fz_tail} trailing ({pct(fz_tail):.4f}%)")
print(f"LIVE   {live} ({pct(live):.4f}%)   NOT REAL {head + fz_int + fz_tail} "
      f"({pct(head + fz_int + fz_tail):.4f}%)")
print(f"exact-minute diagnostic: {int(frozen_exact[D].sum())}")
print("\nper-slot interior-frozen cells (top 16):")
rec = []
for k, sym in enumerate(frame):
    fi = int((frozen & ~tailm)[D][:, k].sum())
    ft = int((frozen & tailm)[D][:, k].sum())
    hd = int((~filled[D])[:, k].sum())
    rec.append((fi, ft, hd, k, sym, id_groups[k],
                {a: per_id_inwin[a] for a in id_groups[k]}))
for fi, ft, hd, k, sym, g, counts in sorted(rec, reverse=True)[:16]:
    print(f"  slot {k:>3} {sym:<12} interior {fi:>5} tail {ft:>4} head {hd:>4} ids {g} {counts}")
