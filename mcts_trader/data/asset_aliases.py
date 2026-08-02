"""TASK_868 (DECISIONS D31, opt (a1)) — asset identity repair: normalized-alias resolution +
fail-loud union loading across twin ``assets`` rows.

THE DEFECT: the 858 minute backfill ingested 13 series a SECOND time under spelling-variant
``assets`` rows (``/USDTEUR`` id 1016 vs ``USDTEUR`` id 1144, twins added 2025-12-12), splitting
one feed's minute history across two ``asset_id``s. On the 861 leg-1 frame 6 of 64 slots
(/USDTEUR, /USDTCHF, /USDC/USD, /USDCGBP, /USDC/USDT, /USDTAUD) consequently read near-empty
2025-03-31 -> 2025-10-01 while the twin id carries 128k-368k real rows spanning exactly that
gap — and in the measured overlap the two closes agree with max relative diff 0.00 (same feed,
split key; independently confirmed by the run-23 orchestrator and again for this task).

THE REPAIR (loader-level, NO DB WRITES): at load time, resolve each requested symbol to the SET
of ``assets`` rows whose NORMALIZED form matches, and union their rows into one series:

  * NORMALIZATION is slash-only: ``symbol.upper().replace("/", "")``. It deliberately does NOT
    fold XBT->BTC (tools/backfill_minute._canon_key does, for dump-file routing): that would
    merge ``BTCCHF`` (id 1085) with ``XBTCHF`` (id 1175) — DIFFERENT feeds, and BTCCHF is one of
    the 5 genuine sparse slots D31 keeps. Slash-only grouping reproduces the planner's measured
    55 duplicate groups table-wide exactly; adding XBT->BTC yields a 56th (BTCCHF/XBTCHF), which
    is precisely the wrong merge.
  * AGREEMENT GATE (fail-loud): where aliased series overlap in time, any close disagreement
    (rel > 1e-12) on a shared timestamp REFUSES with both asset_ids and the first disagreeing
    timestamp named — never average, never prefer silently. Duplicate timestamps with AGREEING
    prices dedupe cleanly (first writer in canonical id order wins).
  * CANONICAL ID ORDER: the exact-spelling id first, then remaining ids ascending. Deterministic;
    the exact DB spelling of the requested symbol owns overlapping timestamps.
  * DISCLOSED: every consumer records per-slot ``asset_ids`` used + per-id row counts in the
    artifact (the D31 "disclosed" rider).

Applied by BOTH minute framed loaders + the canon unpinned minute loader
(``VectorBTTradingEnv._load_database_prices``) + the daily loaders (a no-op where no alias id
carries ``ohlcv_daily`` rows — on the real DB only ``USDCUSD`` id 1145 (721 rows) and ``USDGUSD``
id 1182 (609) do, both verified disjoint-or-agreeing against their originals' zero daily rows).

Table-wide duplicate sweep (D31 scope 3) is ``duplicate_sweep`` below: read-only, report-only.
"""
from __future__ import annotations

import sqlite3

# The agreement gate's tolerance. Overlapping closes of the same feed agree to float64-exact 0.0
# on every measured pair (125,737 shared rows on the largest pair); 1e-12 admits float dust only.
ALIAS_AGREEMENT_REL_TOL = 1e-12


class AliasDisagreementError(RuntimeError):
    """Two aliased asset_ids carry DIFFERENT closes for the same timestamp. Never averaged,
    never silently preferred — the load refuses."""


def normalize_symbol_key(symbol: str) -> str:
    """Slash-only normalized match key. NOT XBT-folding (see module docstring)."""
    return str(symbol).upper().replace("/", "")


def group_asset_rows(asset_rows):
    """[(id, symbol), ...] -> {normalized_key: [(id, symbol), ...] sorted by id}."""
    groups = {}
    for aid, sym in asset_rows:
        groups.setdefault(normalize_symbol_key(sym), []).append((int(aid), sym))
    for g in groups.values():
        g.sort()
    return groups


def resolve_requested_symbols(asset_rows, requested_symbols):
    """Resolve each requested symbol to its alias group.

    Returns (resolution, unknown): one entry per requested symbol (order preserved) ::
    {"symbol", "asset_ids" (canonical order: exact-spelling id first, then ascending),
     "db_symbols", "exact_id"}; ``unknown`` lists requested symbols with NO group at all
    (callers decide: the canon loader warns+skips, the framed loaders refuse)."""
    groups = group_asset_rows(asset_rows)
    resolution, unknown = [], []
    for s in requested_symbols:
        g = groups.get(normalize_symbol_key(s))
        if not g:
            unknown.append(s)
            continue
        exact = [aid for aid, sym in g if sym == s]
        exact_id = exact[0] if exact else None
        ids = ([exact_id] if exact_id is not None else []) \
            + [aid for aid, _ in g if aid != exact_id]
        resolution.append({"symbol": s, "asset_ids": ids, "exact_id": exact_id,
                           "db_symbols": [sym for _, sym in g]})
    return resolution, unknown


def ranked_minute_universe(cur, s_ts, e_ts, n_assets):
    """The canon coverage-ranked universe, REPAIRED: per-id in-window minute row counts grouped
    by normalized key and SUMMED, so a split series ranks on its whole history instead of each
    twin ranking separately (D31: "the split slots' coverage ranking is itself distorted today").

    Returns up to ``n_assets`` groups, ordered by unioned count desc (ties: lowest member id —
    deterministic), each ::
    {"symbol" (representative = the in-window max-count member's DB spelling, tie -> lowest id),
     "asset_ids" (ascending), "count", "per_id_counts"}."""
    rows = cur.execute(
        """SELECT a.id, a.symbol, COUNT(*) FROM ohlcv o JOIN assets a ON o.asset_id = a.id
           WHERE o.timestamp >= ? AND o.timestamp <= ? GROUP BY a.id""",
        (s_ts, e_ts)).fetchall()
    groups = {}
    for aid, sym, cnt in rows:
        g = groups.setdefault(normalize_symbol_key(sym), {"members": [], "count": 0})
        g["members"].append((int(aid), sym, int(cnt)))
        g["count"] += int(cnt)
    ordered = sorted(groups.values(),
                     key=lambda g: (-g["count"], min(m[0] for m in g["members"])))
    out = []
    for g in ordered[:int(n_assets)]:
        members = sorted(g["members"], key=lambda m: (-m[2], m[0]))
        out.append({"symbol": members[0][1],
                    "asset_ids": sorted(m[0] for m in members),
                    "count": g["count"],
                    "per_id_counts": {m[0]: m[2] for m in members}})
    return out


def alias_agreement_gate(conn, table, id_groups, s_ts, e_ts, rel_tol=ALIAS_AGREEMENT_REL_TOL):
    """Fail-loud agreement gate, checked ONCE per load (not per streamed row): for every alias
    group with >1 id, every pair's shared-timestamp closes in [s_ts, e_ts] must agree within
    ``rel_tol`` (relative). A disagreement REFUSES with both asset_ids and the first disagreeing
    timestamp named. Returns a JSON-safe per-group disclosure of the checked overlaps."""
    disclosure = []
    for group in id_groups:
        ids = [int(i) for i in group]
        if len(ids) < 2:
            continue
        pairs = [(ids[i], ids[j]) for i in range(len(ids)) for j in range(i + 1, len(ids))]
        overlap = []
        for a, b in pairs:
            bad = conn.execute(
                f"""SELECT a.timestamp, a.close, b.close FROM {table} a
                    JOIN {table} b ON a.timestamp = b.timestamp
                    WHERE a.asset_id = ? AND b.asset_id = ?
                      AND a.timestamp >= ? AND a.timestamp <= ?
                      AND a.close IS NOT NULL AND b.close IS NOT NULL
                      AND a.close != 0 AND b.close != 0
                      AND ABS(a.close - b.close)
                          / MAX(ABS(a.close), ABS(b.close)) > ?
                    ORDER BY a.timestamp LIMIT 1""",
                (a, b, s_ts, e_ts, rel_tol)).fetchone()
            if bad is not None:
                raise AliasDisagreementError(
                    f"alias agreement gate: asset_ids {a} and {b} (same normalized symbol, "
                    f"table `{table}`) DISAGREE on shared timestamp {int(bad[0])}: closes "
                    f"{bad[1]!r} vs {bad[2]!r} (rel tol {rel_tol}). An ingest identity split is "
                    f"only union-safe where the two feeds are the SAME series; refusing to "
                    f"average or silently prefer one. Resolve the assets rows first.")
            n, mx = conn.execute(
                f"""SELECT COUNT(*), MAX(ABS(a.close - b.close)
                          / MAX(ABS(a.close), ABS(b.close))) FROM {table} a
                    JOIN {table} b ON a.timestamp = b.timestamp
                    WHERE a.asset_id = ? AND b.asset_id = ?
                      AND a.timestamp >= ? AND a.timestamp <= ?
                      AND a.close IS NOT NULL AND b.close IS NOT NULL
                      AND a.close != 0 AND b.close != 0""",
                (a, b, s_ts, e_ts)).fetchone()
            overlap.append({"pair": [a, b], "shared_ts": int(n),
                            "max_rel_diff": (float(mx) if mx is not None else None)})
        disclosure.append({"asset_ids": ids, "overlaps": overlap})
    return disclosure


def real_bar_mask(raw_nz, n_steps, step):
    """(n_steps, ...) bool — did this slot print a REAL close anywhere inside panel row r's BAR?
    ``raw_nz`` is the 1-min-resolution non-zero mask (1-D or 2-D with rows on axis 0). Panel row
    r is the ``[::step]`` sample at 1-min index ``r*step``, so its bar is the ``step`` minutes
    ``((r-1)*step, r*step]`` (row 0 is the single minute 0). Shared by the canon loader
    (env_factory, on the raw mask captured before the fill) and eval_vector's framed minute
    loader (its private ``_real_bar_mask`` delegates here — see its docstring for why the
    exact-minute reading is wrong on real data)."""
    import numpy as np
    nz = np.asarray(raw_nz, dtype=bool)
    if int(step) <= 1:
        return nz[:int(n_steps)]
    n_full = nz.shape[0]
    cs = np.concatenate([np.zeros((1,) + nz.shape[1:], dtype=np.int64),
                         np.cumsum(nz, axis=0, dtype=np.int64)], axis=0)  # cs[i] = prints in [0,i)
    r = np.arange(int(n_steps), dtype=np.int64)
    hi = np.minimum(r * int(step), n_full - 1)
    lo = np.where(r == 0, 0, (r - 1) * int(step) + 1)
    return (cs[hi + 1] - cs[lo]) > 0


def duplicate_sweep(db_path, timeout_s=60):
    """D31 scope 3 — the table-wide duplicate sweep. READ-ONLY (mode=ro URI), report-only:
    every normalized-duplicate ``assets`` group with per-id row counts + spans in BOTH price
    tables and the pairwise overlap agreement. Fix nothing — this is the measurement the wrap
    cites. Returns a JSON-safe dict."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=timeout_s)
    try:
        rows = conn.execute("SELECT id, symbol FROM assets").fetchall()
        groups = group_asset_rows(rows)
        dups = {k: v for k, v in groups.items() if len(v) > 1}
        out = {"db": str(db_path), "normalization": "upper().replace('/', '') — slash-only, "
               "NOT XBT-folding (BTCCHF/XBTCHF are different feeds)",
               "n_assets_rows": len(rows), "n_duplicate_groups": len(dups), "groups": []}
        for key in sorted(dups):
            members = []
            for aid, sym in sorted(dups[key]):
                rec = {"asset_id": aid, "symbol": sym}
                for table in ("ohlcv", "ohlcv_daily"):
                    n, mn, mx = conn.execute(
                        f"SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM {table} "
                        f"WHERE asset_id = ? AND close IS NOT NULL AND close != 0",
                        (aid,)).fetchone()
                    rec[table] = {"rows": int(n), "first_ts": mn, "last_ts": mx}
                members.append(rec)
            ids = [m["asset_id"] for m in members]
            overlaps = []
            for table in ("ohlcv", "ohlcv_daily"):
                for i in range(len(ids)):
                    for j in range(i + 1, len(ids)):
                        n, mx = conn.execute(
                            f"""SELECT COUNT(*), MAX(ABS(a.close - b.close)
                                      / MAX(ABS(a.close), ABS(b.close))) FROM {table} a
                                JOIN {table} b ON a.timestamp = b.timestamp
                                WHERE a.asset_id = ? AND b.asset_id = ?
                                  AND a.close IS NOT NULL AND b.close IS NOT NULL
                                  AND a.close != 0 AND b.close != 0""",
                            (ids[i], ids[j])).fetchone()
                        overlaps.append({"table": table, "pair": [ids[i], ids[j]],
                                         "shared_ts": int(n),
                                         "max_rel_diff": (float(mx) if mx is not None else None)})
            n_live = sum(1 for m in members
                         if m["ohlcv"]["rows"] or m["ohlcv_daily"]["rows"])
            out["groups"].append({
                "normalized_key": key, "members": members, "overlaps": overlaps,
                "classification": ("empty_stub_pair" if n_live <= 1 else "live_twin_pair")})
        return out
    finally:
        conn.close()
