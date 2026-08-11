#!/usr/bin/env python3
"""TASK_877 — XBT/XDG ticker-alias fold, acceptance gates (ii) and (iii), the EVIDENCE FLOOR
(fix pass), plus a full-``assets``-table regression guard for the scope-2 sweep finding.

READ-ONLY against the real DB / fixture (both opened ``mode=ro``); the agreement-gate and
evidence-floor demonstrations use throwaway in-memory SQLite databases they build themselves —
no shared file is ever touched, let alone written.

Run: ``python runs/877_xbt_alias_fold/test_xbt_alias_fold.py``
     ``python runs/877_xbt_alias_fold/test_xbt_alias_fold.py --real-db``  (adds the full-table
     sweep against trading_data.db AND the "the floor refuses nothing already shipped"
     measurement; opt-in, one bounded read-only pass over ``assets`` + small indexed queries —
     no panel build, no GPU)
"""
import argparse
import os
import sqlite3
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from mcts_trader.data.asset_aliases import (   # noqa: E402
    MIN_ALIAS_OVERLAP_TS, AliasDisagreementError, AliasInsufficientOverlapError, TICKER_ALIASES,
    alias_agreement_gate, group_asset_rows, normalize_symbol_key, resolve_requested_symbols)

FIXTURE_DB = os.path.join(_REPO, "fixtures", "trading_fixture.db")
REAL_DB = os.path.join(_REPO, "trading_data.db")

failures = []


def check(name, ok, detail=""):
    print(f"  {'OK  ' if ok else 'FAIL'} {name}" + (f" — {detail}" if detail and not ok else ""))
    if not ok:
        failures.append(f"{name}: {detail}")


# ---------------------------------------------------------------------------------------------
# Gate (iii): the alias table is explicit and TOKEN-EXACT — a symbol that merely CONTAINS "xbt"
# as a substring of a DIFFERENT token must not fold. Two of these are REAL rows in `assets`
# today (AIXBT/EUR id 78, XBTPY/USD id 1090) — not hypothetical.
# ---------------------------------------------------------------------------------------------
print("[877] gate (iii) — substring-trap symbols must NOT fold ...")
SUBSTRING_TRAPS = [
    ("AIXBT/EUR", "AIXBTEUR"),   # real row id 78 — "xbt" at position 2, a DIFFERENT ticker
    ("AIXBT/USD", "AIXBTUSD"),   # real row id 79
    ("XBTPY/USD", "XBTPYUSD"),   # real row id 1090 — base token is "XBTPY", not "XBT"
    ("WBTC/EUR", "WBTCEUR"),     # real row id 1045 — wrapped BTC, a DIFFERENT instrument
    ("WBTC/USD", "WBTCUSD"),     # real row id 1046
    ("TBTC/USD", "TBTCUSD"),     # real row id 933 — tBTC, a DIFFERENT instrument
    ("XXBTZUSD", "XXBTZUSD"),    # real row id 1139 — legacy Kraken REST spelling, "XX" not "XBT"
]
for sym, expected in SUBSTRING_TRAPS:
    got = normalize_symbol_key(sym)
    check(f"normalize_symbol_key({sym!r}) does not fold (-> {expected!r})", got == expected,
          f"got {got!r}")
# The gate iii requirement stated literally: no blanket `.replace` anywhere in the module.
import inspect                                                        # noqa: E402
from mcts_trader.data import asset_aliases as _aa_mod                 # noqa: E402
_src = inspect.getsource(_aa_mod)
check("no blanket str.replace('XBT'/'XDG', ...) anywhere in the module",
      ("replace(\"XBT\"" not in _src and "replace('XBT'" not in _src
       and "replace(\"XDG\"" not in _src and "replace('XDG'" not in _src),
      "found a substring-replace call on an alias token")

# ---------------------------------------------------------------------------------------------
# The fold itself: known collisions must unify, and TICKER_ALIASES must be exactly {XBT->BTC,
# XDG->DOGE} per the task's explicit scope (not a wider guess).
# ---------------------------------------------------------------------------------------------
print("[877] the fold — known aliases unify to one key ...")
FOLD_CASES = [
    ("XBT/CHF", "BTC/CHF"), ("XBTCHF", "BTCCHF"), ("XBT/CHF", "BTCCHF"),
    ("XDG/USD", "XDGUSD"), ("XDGBTC", "XDG/BTC"),
]
for a, b in FOLD_CASES:
    ka, kb = normalize_symbol_key(a), normalize_symbol_key(b)
    check(f"normalize_symbol_key({a!r}) == normalize_symbol_key({b!r})  ({ka!r})", ka == kb,
          f"{ka!r} != {kb!r}")
check("TICKER_ALIASES is exactly the task's scoped table",
      TICKER_ALIASES == {"XBT": "BTC", "XDG": "DOGE"}, repr(TICKER_ALIASES))

# ---------------------------------------------------------------------------------------------
# Gate (ii): the agreement gate demonstrated BOTH ways, on a throwaway in-memory DB (never the
# real DB or the fixture — this never writes to anything shared).
# ---------------------------------------------------------------------------------------------
print("[877] gate (ii) — agreement gate, both directions (throwaway in-memory DB) ...")


SHARED_TS = 100          # shared timestamps in the gate-(ii) fixture; > MIN_ALIAS_OVERLAP_TS (64)


def _make_synthetic_db(tamper_ts=None, tamper_close=None):
    """A minimal in-memory DB with one BTCCHF/XBTCHF-shaped alias pair: asset_id 2 = 'XBTCHF'
    (dense: 200 minute rows), asset_id 1 = 'BTCCHF' (sparser: the first ``SHARED_TS`` of those
    timestamps, agreeing on every one of them). Optionally corrupts ONE of asset 2's closes at
    ``tamper_ts`` to ``tamper_close`` to exercise the REFUSAL path.

    The overlap is deliberately ``SHARED_TS`` > ``MIN_ALIAS_OVERLAP_TS``: gate (ii) is about
    what the gate does with AGREEMENT and DISAGREEMENT, so its fixture must clear the evidence
    floor — otherwise this demonstration would itself be the untested union the floor refuses
    (it was, before the fix pass: the original fixture shared exactly 1 timestamp)."""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE assets (id INTEGER PRIMARY KEY, symbol TEXT UNIQUE NOT NULL)")
    conn.execute("CREATE TABLE ohlcv (asset_id INTEGER, timestamp INTEGER, close REAL, "
                 "PRIMARY KEY (asset_id, timestamp))")
    conn.executemany("INSERT INTO assets VALUES (?, ?)", [(1, "BTCCHF"), (2, "XBTCHF")])
    xbtchf_closes = {1000 + 60 * k: 50000.0 + 10.0 * k for k in range(200)}
    rows = [(1, ts, xbtchf_closes[ts])                            # BTCCHF: agrees on its subset
            for ts in sorted(xbtchf_closes)[:SHARED_TS]]
    if tamper_ts is not None:
        xbtchf_closes[tamper_ts] = tamper_close
    rows += [(2, ts, c) for ts, c in sorted(xbtchf_closes.items())]
    conn.executemany("INSERT INTO ohlcv VALUES (?, ?, ?)", rows)
    conn.commit()
    return conn


# (ii-a) AGREEING overlap folds/gates CLEANLY — no raise, disclosure names the pair + shared ts.
conn = _make_synthetic_db()
try:
    all_assets = conn.execute("SELECT id, symbol FROM assets").fetchall()
    resolution, unknown = resolve_requested_symbols(all_assets, ["BTCCHF"])
    id_groups = [r["asset_ids"] for r in resolution]
    check("BTCCHF resolves to BOTH synthetic ids [1, 2]", id_groups == [[1, 2]], str(id_groups))
    raised = False
    disclosure = None
    try:
        disclosure = alias_agreement_gate(conn, "ohlcv", id_groups, 0, 9999999999)
    except AliasDisagreementError:
        raised = True
    check("agreeing overlap does NOT raise", not raised)
    check("disclosure names the pair with a non-zero shared-timestamp overlap and 0.0 max diff",
          disclosure and disclosure[0]["asset_ids"] == [1, 2]
          and disclosure[0]["overlaps"][0]["shared_ts"] == SHARED_TS
          and disclosure[0]["overlaps"][0]["max_rel_diff"] == 0.0,
          str(disclosure))
finally:
    conn.close()

# (ii-b) TAMPERED overlap REFUSES, naming both ids and the FIRST disagreeing timestamp.
conn = _make_synthetic_db(tamper_ts=1000, tamper_close=999999.0)   # corrupt the SHARED ts
try:
    all_assets = conn.execute("SELECT id, symbol FROM assets").fetchall()
    resolution, _ = resolve_requested_symbols(all_assets, ["BTCCHF"])
    id_groups = [r["asset_ids"] for r in resolution]
    raised_msg = None
    try:
        alias_agreement_gate(conn, "ohlcv", id_groups, 0, 9999999999)
    except AliasDisagreementError as exc:
        raised_msg = str(exc)
    check("tampered overlap RAISES AliasDisagreementError", raised_msg is not None)
    if raised_msg:
        check("refusal names BOTH asset_ids (1 and 2)",
              "1" in raised_msg.split("DISAGREE")[0] and "2" in raised_msg.split("DISAGREE")[0],
              raised_msg)
        check("refusal names the FIRST disagreeing timestamp (1000)", "1000" in raised_msg,
              raised_msg)
        check("refusal never mentions an averaged/preferred value (no silent pick)",
              "average" not in raised_msg.lower() or "refusing to average" in raised_msg.lower(),
              raised_msg)
finally:
    conn.close()

# ---------------------------------------------------------------------------------------------
# EVIDENCE FLOOR (TASK_877 fix pass, MAJOR 1). Before it, the gate blessed a merge sharing 0 or 1
# timestamps exactly as confidently as one sharing 48,780: "no disagreement found" over an empty
# overlap is not evidence, it is silence. The floor makes an untested union REFUSE.
#
# The measurement that bounds the threshold (see MIN_ALIAS_OVERLAP_TS): the smallest overlap any
# CURRENTLY-SHIPPED fold has is 720 shared timestamps (fixture, TWXUSD ids 90001/90002) and
# 10,657 (real DB, USDGUSD ids 1001/1182) — so 64 cannot retroactively refuse one. Both bounds
# are re-measured below (the real-DB one under --real-db), not asserted from a comment.
# ---------------------------------------------------------------------------------------------
print("[877] evidence floor — red then green (throwaway in-memory DBs) ...")


def _floor_db(rows_a, rows_b):
    """In-memory DB with ids 1 ('BTCCHF') and 2 ('XBTCHF') carrying the given {ts: close} maps."""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE assets (id INTEGER PRIMARY KEY, symbol TEXT UNIQUE NOT NULL)")
    conn.execute("CREATE TABLE ohlcv (asset_id INTEGER, timestamp INTEGER, close REAL, "
                 "PRIMARY KEY (asset_id, timestamp))")
    conn.executemany("INSERT INTO assets VALUES (?, ?)", [(1, "BTCCHF"), (2, "XBTCHF")])
    conn.executemany("INSERT INTO ohlcv VALUES (?, ?, ?)",
                     [(1, t, c) for t, c in sorted(rows_a.items())]
                     + [(2, t, c) for t, c in sorted(rows_b.items())])
    conn.commit()
    return conn


def _gate(conn, s_ts=0, e_ts=9999999999, **kw):
    all_assets = conn.execute("SELECT id, symbol FROM assets").fetchall()
    resolution, _ = resolve_requested_symbols(all_assets, ["BTCCHF"])
    return alias_agreement_gate(conn, "ohlcv", [r["asset_ids"] for r in resolution],
                                s_ts, e_ts, **kw)


# (F-a) RED — two live ids whose spans NEVER meet. Nothing disagrees (nothing is comparable),
#       and pre-fix this passed silently. It must now REFUSE, naming both ids and the count.
disjoint_a = {t: 50000.0 + t for t in range(1, 101)}          # ts 1..100
disjoint_b = {t: 60000.0 + t for t in range(1000, 1100)}      # ts 1000..1099, no shared ts
conn = _floor_db(disjoint_a, disjoint_b)
try:
    msg = None
    try:
        _gate(conn)
    except AliasInsufficientOverlapError as exc:
        msg = str(exc)
    check("RED: zero-overlap union of two LIVE ids REFUSES (AliasInsufficientOverlapError)",
          msg is not None, "the gate blessed a union with no shared timestamp at all")
    if msg:
        check("refusal names BOTH asset_ids and the shared-timestamp count",
              "1" in msg and "2" in msg and "0 timestamp(s)" in msg, msg)
    # (F-b) GREEN — the SAME database with the floor lowered to 0 passes. This is what makes the
    #       red above attributable to the floor and nothing else: no other check changed.
    raised = False
    try:
        disc = _gate(conn, min_shared_ts=0)
    except (AliasInsufficientOverlapError, AliasDisagreementError):
        raised = True
    check("GREEN: same DB, min_shared_ts=0 -> passes (the floor is the only thing that fired)",
          not raised)
finally:
    conn.close()

# (F-c) A one-shared-timestamp union is just as untested — the second half of "0 or 1".
conn = _floor_db({1000: 50000.0}, {1000: 50000.0, 1060: 50010.0})
try:
    msg = None
    try:
        _gate(conn)
    except AliasInsufficientOverlapError as exc:
        msg = str(exc)
    check("RED: a ONE-shared-timestamp union also REFUSES (agreement on n=1 is not evidence)",
          msg is not None and "1 timestamp(s)" in msg, str(msg))
finally:
    conn.close()

# (F-d) GREEN — an EMPTY STUB pair (one id carries no rows at all) is a pure relabel: nothing is
#       blended, nothing can be substituted. It must NOT refuse — 42 of the 56 real-DB duplicate
#       groups are exactly this shape, and 11 of the 18 aliased frame slots depend on it.
conn = _floor_db({t: 50000.0 + t for t in range(1, 101)}, {})
try:
    raised = None
    try:
        disc = _gate(conn)
    except Exception as exc:                                      # noqa: BLE001
        raised = repr(exc)
    check("GREEN: empty-stub pair (0 rows on one side) passes as 'relabel_only'",
          raised is None and disc[0]["overlaps"][0]["evidence"]["basis"] == "relabel_only",
          raised or str(disc))
finally:
    conn.close()

# (F-e) GREEN — the case that would otherwise be a REGRESSION on already-shipped folds: both ids
#       live inside the requested window, ZERO shared timestamps inside it, but a long agreeing
#       overlap outside it. Measured on the real DB: on 2025-09-14..2025-12-12 SEVEN shipped twin
#       pairs (incl. BTCCHF/XBTCHF) share exactly 0 timestamps while sharing 10,657-125,737 over
#       their whole history. Identity is a property of the ids, not of the window.
hist_a = {t: 50000.0 + t for t in range(0, 200)}                  # ts 0..199
hist_b = dict(hist_a)                                            # 200 agreeing shared ts ...
hist_b.update({t: 50000.0 + t for t in range(5000, 5100)})       # ... plus its own later span
hist_a.update({t: 50000.0 + t for t in range(6000, 6100)})       # window 4900..6100: both live,
conn = _floor_db(hist_a, hist_b)                                 # zero SHARED inside it
try:
    raised = None
    try:
        disc = _gate(conn, s_ts=4900, e_ts=6100)
    except Exception as exc:                                      # noqa: BLE001
        raised = repr(exc)
    ev = None if raised else disc[0]["overlaps"][0]["evidence"]
    check("GREEN: zero in-window overlap + agreeing FULL history passes as 'full_history'",
          raised is None and ev["basis"] == "full_history" and ev["shared_ts"] == 0
          and ev["full_history_shared_ts"] == 200, raised or str(disc))
finally:
    conn.close()

# (F-f) RED — the same fallback, but the full history DISAGREES: refuse as a disagreement (the
#       evidence was found and it REFUTES identity), naming the disagreeing timestamp.
hist_b2 = dict(hist_b)
hist_b2[100] = 1.0                                               # tamper OUTSIDE the window
conn = _floor_db(hist_a, hist_b2)
try:
    msg = None
    try:
        _gate(conn, s_ts=4900, e_ts=6100)
    except AliasDisagreementError as exc:
        msg = str(exc)
    check("RED: full-history fallback that DISAGREES refuses as AliasDisagreementError (ts 100)",
          msg is not None and "100" in msg and "OUTSIDE the requested window" in msg, str(msg))
finally:
    conn.close()

# (F-g) GREEN — a plain, well-evidenced in-window union is unchanged: basis 'window', no extra
#       queries, no behaviour change. This is the path every shipped fold takes on a wide window.
plain = {t: 50000.0 + t for t in range(0, 500)}
conn = _floor_db(plain, dict(plain))
try:
    disc = _gate(conn)
    ev = disc[0]["overlaps"][0]["evidence"]
    check("GREEN: a 500-shared-timestamp agreeing union passes on basis 'window'",
          ev["basis"] == "window" and ev["shared_ts"] == 500
          and ev["min_required"] == MIN_ALIAS_OVERLAP_TS, str(ev))
finally:
    conn.close()

# (ii-c) SUPERSEDED BY THE FLOOR, kept explicit so the change of contract is on the record: a
# disjoint pair used to be asserted here as "never refuses". It now REFUSES when BOTH ids are
# live (F-a) — that was the MAJOR-1 hole — and still does not refuse when the pair is a stub
# relabel (F-d) or when the evidence lives outside the window (F-e). What remains true, and is
# re-checked here, is the narrower original point: the gate never invents a DISAGREEMENT out of
# non-overlapping rows.
conn = _floor_db(disjoint_a, disjoint_b)
try:
    kind = None
    try:
        _gate(conn)
    except Exception as exc:                                      # noqa: BLE001
        kind = type(exc).__name__
    check("disjoint spans are never reported as a price DISAGREEMENT",
          kind == "AliasInsufficientOverlapError", str(kind))
finally:
    conn.close()

# ---------------------------------------------------------------------------------------------
# Full-`assets`-table regression guard (scope 2's sweep) — read-only against the fixture always,
# and against the real DB under --real-db. Confirms the ONE known live merge and that nothing
# else moves.
# ---------------------------------------------------------------------------------------------
def sweep(db_path):
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=60)
    try:
        rows = conn.execute("SELECT id, symbol FROM assets").fetchall()
        old_groups = {}
        for aid, sym in rows:
            k = str(sym).upper().replace("/", "")
            old_groups.setdefault(k, set()).add(aid)
        new_groups = group_asset_rows(rows)
        merges = []
        for nk, members in new_groups.items():
            ids = {aid for aid, _ in members}
            old_keys_here = {str(sym).upper().replace("/", "") for _, sym in members}
            if len(old_keys_here) > 1:
                live = []
                for aid, sym in members:
                    n = conn.execute(
                        "SELECT COUNT(*) FROM ohlcv WHERE asset_id=? AND close IS NOT NULL "
                        "AND close!=0", (aid,)).fetchone()[0]
                    live.append(n)
                merges.append((nk, sorted(ids), live))
        return merges
    finally:
        conn.close()


def both_live_overlaps(db_path, table="ohlcv"):
    """Every duplicate-group pair where BOTH ids carry rows (the pairs the evidence floor
    actually judges), with their WHOLE-history shared-timestamp count. Read-only."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=120)
    try:
        groups = group_asset_rows(conn.execute("SELECT id, symbol FROM assets").fetchall())
        out = []
        for key, members in sorted(groups.items()):
            if len(members) < 2:
                continue
            ids = [aid for aid, _ in members]
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    a, b = ids[i], ids[j]
                    live = [conn.execute(
                        f"SELECT 1 FROM {table} WHERE asset_id=? AND close IS NOT NULL "
                        f"AND close!=0 LIMIT 1", (x,)).fetchone() is not None for x in (a, b)]
                    if not all(live):
                        continue
                    n, mx = conn.execute(
                        f"""SELECT COUNT(*), MAX(ABS(a.close-b.close)
                                  /MAX(ABS(a.close),ABS(b.close))) FROM {table} a
                            JOIN {table} b ON a.timestamp=b.timestamp
                            WHERE a.asset_id=? AND b.asset_id=? AND a.close IS NOT NULL
                              AND b.close IS NOT NULL AND a.close!=0 AND b.close!=0""",
                        (a, b)).fetchone()
                    out.append((key, a, b, int(n), mx))
        return out
    finally:
        conn.close()


def check_floor_refuses_nothing_shipped(db_path, label):
    """THE CONSTRAINT ON THE THRESHOLD: a floor that retroactively refuses an already-shipped
    fold is a regression, not a fix. Measured, not assumed — every both-live pair on this DB."""
    pairs = both_live_overlaps(db_path)
    worst = min(pairs, key=lambda p: p[3]) if pairs else None
    print(f"  {label}: {len(pairs)} BOTH-LIVE alias pair(s); smallest whole-history overlap "
          + (f"{worst[3]:,} shared ts ({worst[0]}, ids [{worst[1]}, {worst[2]}])"
             if worst else "n/a"))
    check(f"[{label}] the {MIN_ALIAS_OVERLAP_TS}-ts floor refuses NO currently-shipped fold",
          all(n >= MIN_ALIAS_OVERLAP_TS for _, _, _, n, _ in pairs),
          str([(k, a, b, n) for k, a, b, n, _ in pairs if n < MIN_ALIAS_OVERLAP_TS]))
    check(f"[{label}] every both-live pair still agrees exactly (max rel diff 0.0)",
          all(mx == 0.0 for *_, mx in pairs),
          str([(k, a, b, mx) for k, a, b, _, mx in pairs if mx != 0.0]))
    return pairs


print(f"[877] full-assets-table sweep (fixture: {os.path.relpath(FIXTURE_DB, _REPO)}) ...")
fixture_merges = sweep(FIXTURE_DB)
check("exactly one new merge, key 'BTCCHF', ids [1085, 1175]",
      len(fixture_merges) == 1 and fixture_merges[0][0] == "BTCCHF"
      and fixture_merges[0][1] == [1085, 1175],
      str(fixture_merges))
check("on the FIXTURE both sides are empty (0 rows) — the fold is a structural no-op there, "
      "numerically invisible to every fixture-based smoke suite",
      bool(fixture_merges) and fixture_merges[0][2] == [0, 0],
      str(fixture_merges))
check_floor_refuses_nothing_shipped(FIXTURE_DB, "fixture")

parser = argparse.ArgumentParser()
parser.add_argument("--real-db", action="store_true",
                     help="also sweep trading_data.db (read-only, bounded)")
args = parser.parse_args()
if args.real_db:
    print(f"[877] full-assets-table sweep (real DB: {REAL_DB}) ...")
    real_merges = sweep(REAL_DB)
    check("exactly one new merge, key 'BTCCHF', ids [1085, 1175], BOTH sides carry real rows",
          len(real_merges) == 1 and real_merges[0][0] == "BTCCHF"
          and real_merges[0][1] == [1085, 1175] and all(n > 0 for n in real_merges[0][2]),
          str(real_merges))
    real_pairs = check_floor_refuses_nothing_shipped(REAL_DB, "real DB")
    check("the real DB's both-live set is the 13 TASK_868 twins + the one TASK_877 pair",
          len(real_pairs) == 14 and any(k == "BTCCHF" and [a, b] == [1085, 1175]
                                        for k, a, b, _, _ in real_pairs),
          str([(k, a, b) for k, a, b, _, _ in real_pairs]))
    # NB: the daily LOADERS do not apply the alias union at all (they resolve by exact
    # `assets.symbol` match — see asset_aliases' "NOT APPLIED BY THE DAILY LOADERS" paragraph and
    # the /USDC/USD miscount disclosed there). This pins only the narrower claim that paragraph
    # does make: on `ohlcv_daily` a union WOULD be a pure relabel, because no group has two
    # daily-live members. "Nothing is blended" is not "the requested spelling reaches the data".
    check("the daily table never blends two series (no group has 2 daily-live members) — the "
          "'pure relabel' claim asset_aliases' docstring makes about ohlcv_daily",
          both_live_overlaps(REAL_DB, table="ohlcv_daily") == [],
          str(both_live_overlaps(REAL_DB, table="ohlcv_daily")))
else:
    print("[877] real-DB sweep SKIPPED (pass --real-db to run it)")

print()
if failures:
    print(f"{len(failures)} FAILURE(S):")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("ALL GATES PASS")
sys.exit(0)
