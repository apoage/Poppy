#!/usr/bin/env python3
"""
TASK_897 D56(a) — the canon alias union applied as a READ-TIME LAYER, owned by THIS runner.

WHY THIS FILE EXISTS (and why it is not a change to the 806/808/809 lineage)
-----------------------------------------------------------------------------
D56 measured a real contradiction: TASK_897's own gate (iii) requires the TASK_868/877 alias
fold applied, while TASK_897's own Out-of-scope forbids modifying the 809 lineage
(`runs/806_predictability_probe/probe.py`, `runs/808_linear_backtest/backtest.py`,
`runs/809_decision_rule_rescue/decide.py`) beyond the estimator fix. Those files have zero
alias handling and predate TASK_877 (verified, `test_sweep_guards.test_the_lineage_really_
has_no_alias_handling`). D56 ruled (a): measure the blast radius, then apply the union AT READ
TIME, from THIS runner, without touching the lineage. The blast radius is nonzero — 6 of 291
candidate symbols are the TASK_868/877 identity-split class (`/USDC/USD`, `/USDC/USDT`,
`/USDCGBP`, `/USDTAUD`, `/USDTCHF`, `/USDTEUR`) — so this module exists.

WHAT THIS MODULE DOES, EXACTLY
-------------------------------
Two functions, each a READ-TIME REPLACEMENT for the equivalent call `sweep_gated_maker.py`
used to make directly into `probe.py` — never a copy-and-tweak of `probe.py`'s own functions,
and `probe.py`/`backtest.py`/`decide.py` are imported here only for their tiny, already-public
helpers (`_to_epoch`, `Config`) that this module's own callers already relied on:

  * `rank_universe_with_alias_union(cfg)` — mirrors `probe.load_universe_trainlocked`'s
    candidate-pool + train-window-locked ranking, EXCEPT each candidate symbol's rank key is
    the SUM of in-train-window bar counts over every id in its `mcts_trader.data.asset_aliases`
    normalized group (the same "rank on the whole history, not each twin separately" rule
    `asset_aliases.ranked_minute_universe` already ships for the canon minute loaders) rather
    than the single exact-spelling id's count alone. Returns both the aliased top-N AND the
    ORIGINAL exact-spelling top-N (computed for free from the same per-id counts) so a caller
    can name every rank change without a second DB pass.
  * `build_aligned_ohlcv_with_alias_union(cfg, asset_ids, alias_groups, bar_sec)` — mirrors
    `probe.build_aligned_ohlcv` cell for cell (identical grid construction, identical
    forward/back-fill policy — `_forward_back_fill` below is a literal duplicate of that loop,
    kept in sync by `test_alias_read_layer.py::test_single_id_slot_matches_probe_byte_for_byte`),
    except the per-slot row LOAD unions every id in that slot's alias group instead of reading
    only the exact-spelling id, and passes every multi-id group through
    `asset_aliases.alias_agreement_gate` (fail-loud on disagreement, fail-closed on an
    untested/insufficient overlap) before trusting the union.

Both functions open the DB `mode=ro` (belt-and-suspenders on top of `sweep_gated_maker.py`'s
own `assert_db_path_allowed` gate) and never write the DB file. Sidecars are mode-dependent, not
unconditional: on a WAL-mode source a `mode=ro` read still CREATES `-shm`/`-wal`. See
`_connect_ro`'s docstring — the protected `trading_data.db` is verified rollback-journal, so
nothing is created on it.

WHY DUPLICATE THE GRID/FILL LOOP INSTEAD OF IMPORTING IT: `probe.build_aligned_ohlcv` couples
the grid construction AND the fill policy to its own single-id-per-column row-load loop in one
function body; there is no seam to call "the fill, not the load" without editing `probe.py`,
which Out-of-scope forbids. Duplicating ~15 lines of pure-numpy, no-DB fill logic here (and
pinning it against drift with a byte-for-byte parity test) is the wrap-at-the-read-boundary
this row was ruled to do, not a fork of the lineage: nothing in `probe.py` changes, and nothing
here reaches back into it except the two tiny helpers named above.
"""
from __future__ import annotations

import os
import sqlite3
import stat
import sys
import urllib.parse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_PROBE_DIR = _REPO / "runs" / "806_predictability_probe"
if str(_PROBE_DIR) not in sys.path:
    sys.path.insert(0, str(_PROBE_DIR))

import probe  # noqa: E402  (806 lineage — used ONLY for _to_epoch/Config, never edited/forked)
from mcts_trader.data import asset_aliases as aa  # noqa: E402

# The 868/877 identity-split slots D56 measured MATERIAL in the 809 candidate universe (exact
# spelling id misses 70k-203k minute rows per OOS window vs the union, 33-56%; see
# `alias_material_slot_report` below for the per-window remeasurement this row ran itself).
MATERIAL_ALIAS_SYMBOLS = (
    "/USDC/USD", "/USDC/USDT", "/USDCGBP", "/USDTAUD", "/USDTCHF", "/USDTEUR",
)


def normalize_db_path(db_path) -> str:
    """The EXACT byte-string sqlite will open, given `db_path` as a `file:<path>?mode=ro` URI.

    D59 — A CALLER'S STRING AND THE FILE SQLITE OPENS COULD BE DIFFERENT FILES, AND THAT DEFEATED
    THE PROTECTED-DB GATE. `sqlite3ParseUri` discards a leading `//<authority>` when the authority
    is EMPTY or exactly `localhost` (lowercase, byte-compared), so `file://localhost/x/db?mode=ro`
    opens `/x/db`. Any belt reasoning about the raw string is therefore reasoning about a path
    sqlite will not open — `os.path.realpath("//localhost/x/db")` is not the protected realpath,
    and `os.path.samefile` raises `OSError` on it. Measured against a decoy standing in for
    `trading_data.db`, before this function existed:

        assert_db_path_allowed('<decoy>',            scheduled=False) -> REFUSED (correct)
        assert_db_path_allowed('//localhost<decoy>', scheduled=False) -> PASSED  (gate defeated)
        ...and _connect_ro on the returned string then OPENED the protected decoy.

    The fix is not another special case in a belt: normalize ONCE, here, and hand the SAME string
    to the percent-quote and to the open, so no two consumers can ever see different strings —
    that divergence is the whole bug class. `sweep_gated_maker.assert_db_path_allowed` separately
    refuses a leading `//` outright, so the two belts do not rely on each other.

    THIS MIRRORS SQLITE'S PARSER EXACTLY, not Python's path semantics. Only an empty authority
    (`///x` -> `/x`) and the byte-exact lowercase `localhost` (`//localhost/x` -> `/x`) are
    stripped, because those are the only two `sqlite3ParseUri` accepts; anything else
    (`//LOCALHOST/x`, `//tmp/x`) is left UNTOUCHED and sqlite refuses it loudly with
    `invalid uri authority: ...` — verified on both spellings. A returned value can therefore
    still begin with `//` only in that refused-by-sqlite case, so no residual spelling exists
    where sqlite successfully opens a path the guard did not stat.

    `sweep_gated_maker.assert_db_path_allowed` refuses a leading `//` outright (belt 1) AND
    normalizes through this function before its realpath/samefile identity check (belt 2);
    neither belt depends on the other."""
    p = str(db_path)
    if p.startswith("//"):
        rest = p[2:]
        cut = rest.find("/")
        authority = rest if cut == -1 else rest[:cut]
        remainder = "" if cut == -1 else rest[cut:]
        if authority == "" or authority == "localhost":
            return remainder
    return p


def _connect_ro(db_path: str) -> sqlite3.Connection:
    """Read-only handle on `db_path`.

    NOTE ON THE SCOPE OF "READ-ONLY" (D59 fix cycle 2, FINDING 2). `mode=ro` guarantees no write
    to the DB FILE ITSELF in every case, but on a **WAL-mode** source a read still CREATES the
    `-shm` and `-wal` sidecars where none existed (demonstrated on a WAL copy of the fixture: a
    single SELECT created `-shm` 32768 B and `-wal` 0 B). That is mode-dependent, not universal:
    on a rollback-journal (`journal_mode=delete`) source no sidecar is created and the directory
    is left byte-identical. The protected `trading_data.db` is verified rollback-journal today
    (file-header bytes 18/19 both `1`; `2` would mean WAL), so no sidecar is created on it — and
    an end-to-end `--smoke` run was checked to leave it byte-identical (sha256 + full listing).
    Callers on a WAL source should expect sidecars.

    D57 fix cycle 4, item A1 — THE `mode=ro` USED TO BE DEFEATABLE BY THE PATH ITSELF. The old
    body interpolated the raw path into `f"file:{db_path}?mode=ro"`, so a path containing a `?`
    (e.g. `trading_data.db?x=1`) produced `file:trading_data.db?x=1?mode=ro`: sqlite's URI
    parser reads that as the SINGLE parameter `x="1?mode=ro"`, `mode` is never set, and the
    handle opens READ-WRITE-CREATE — a writable handle on the protected read-only source of
    record (a judge demonstrated `CREATE TABLE` through it). Percent-quoting the path makes the
    `?` an ordinary path character (`%3F`), so `mode=ro` is always the first and only parameter.
    `safe="/"` keeps directory separators intact, which is what the URI form requires.
    `sweep_gated_maker.assert_db_path_allowed` independently refuses `?`/`#`/`file:`/`//` in a
    `--db-path`; this quoting is the second belt and does not rely on the first.

    D59 — ONE STRING, TWO USES. `normalize_db_path` runs FIRST and its single result feeds BOTH
    the percent-quote and the open, so no later belt can ever be reasoning about a different file
    than the one sqlite opens. That divergence was a real defect: a leading `//localhost` is
    accepted by sqlite's URI parser as an ignored authority and stripped, so an un-normalized
    caller-supplied string and the string sqlite actually opened were two different paths. It is
    also why `sweep_gated_maker.assert_db_path_allowed` refuses a leading `//` outright — without
    normalization, `//localhost/…/trading_data.db` walked straight past the protected-DB gate
    (demonstrated against a decoy). Do not reintroduce `str(db_path)` anywhere past this line.

    KNOWN, DELIBERATELY NOT GUARDED — a non-regular `--db-path` hangs. A `mode=ro` URI open on a
    FIFO blocks forever, where the pre-fix bare read-write connect failed fast ('disk I/O error');
    the same is true of a FIFO `<db>-journal` sidecar beside an otherwise valid DB, and that one
    hangs identically under the pre-fix form too, so it is a property of opening sqlite at all.
    An earlier cycle added an `S_ISREG` type guard for this and it was REMOVED on operator call:
    `--db-path` is an operator-supplied path to a real database on a scheduled run, a named pipe
    is not a plausible input, and each cycle spent chasing a more exotic spelling (FIFO →
    symlink-to-FIFO → `//localhost`-prefixed FIFO) was the TASK_891 failure mode — chasing total
    coverage of an unbounded input space — reappearing. Complexity inside a fail-closed guard is
    itself risk. This is a documented limitation, not a defence: pass a real sqlite file."""
    path = normalize_db_path(db_path)
    quoted = urllib.parse.quote(path, safe="/")
    return sqlite3.connect(f"file:{quoted}?mode=ro", uri=True)


def _candidate_symbols(cfg: "probe.Config") -> List[str]:
    """The 291-symbol candidate pool `probe.load_universe_trainlocked` ranks over: the union of
    every benchmark period's `benchmark_assets`, exactly as that function reads it."""
    import json
    with open(cfg.benchmark_periods) as f:
        periods = json.load(f)["periods"]
    return sorted({s for p in periods for s in p.get("benchmark_assets", [])})


def rank_universe_with_alias_union(cfg: "probe.Config") -> Dict:
    """Read-time replacement for `probe.load_universe_trainlocked(cfg)`.

    Returns a dict:
      symbols, asset_ids, alias_groups      — the aliased top-N (asset_ids[i] is the
                                                exact-spelling id of slot i; alias_groups[i] is
                                                that slot's full alias-group id list, exact id
                                                first — the shape `build_aligned_ohlcv_with_
                                                alias_union` needs).
      ranking_aliased                        — full candidate list, sorted desc by union count.
      ranking_exact                          — full candidate list, sorted desc by exact-id-only
                                                count (byte-identical to what
                                                `probe.load_universe_trainlocked` itself would
                                                rank, computed from the SAME per-id counts so
                                                there is no second DB pass and no chance of the
                                                two rankings drifting from independently-run
                                                queries).
      rank_changes                           — every candidate symbol whose position in
                                                `ranking_aliased` differs from its position in
                                                `ranking_exact`, `{symbol, exact_rank, union_rank}`
                                                (0-indexed, ascending = higher-ranked).
      top64_entered / top64_left             — symbols that enter/leave the top-N specifically
                                                under the union read (the membership-changing
                                                subset of `rank_changes`, the one the D56 ruling
                                                calls out by name: "can change which assets make
                                                the top-64").
      unknown                                — candidate symbols with no alias group at all
                                                (skipped, same convention `probe.py` uses for a
                                                symbol absent from `assets` entirely).
    """
    candidates = _candidate_symbols(cfg)
    con = _connect_ro(cfg.db_path)
    try:
        cur = con.cursor()
        cur.execute("SELECT id, symbol FROM assets")
        asset_rows = cur.fetchall()
        resolution, unknown = aa.resolve_requested_symbols(asset_rows, candidates)

        tr_s = probe._to_epoch(cfg.train_start)
        tr_e = probe._to_epoch(cfg.val_start)

        entries = []
        for r in resolution:
            exact_id = r["exact_id"]
            if exact_id is None:
                # Same convention `probe.load_universe_trainlocked` uses for a requested
                # spelling absent from `assets`: contributes nothing, ranked nowhere. An alias
                # group matched only via a DIFFERENT spelling never got requested by that exact
                # string, so nothing changes here relative to the un-aliased loader.
                continue
            per_id_counts = {}
            for aid in r["asset_ids"]:
                cur.execute(
                    "SELECT COUNT(*) FROM ohlcv WHERE asset_id=? AND timestamp>=? AND timestamp<?",
                    (aid, tr_s, tr_e),
                )
                per_id_counts[aid] = int(cur.fetchone()[0])
            entries.append({
                "symbol": r["symbol"], "exact_id": exact_id, "asset_ids": r["asset_ids"],
                "per_id_train_bar_count": per_id_counts,
                "exact_train_bar_count": per_id_counts[exact_id],
                "union_train_bar_count": sum(per_id_counts.values()),
            })
    finally:
        con.close()

    ranking_exact = sorted(entries, key=lambda e: (-e["exact_train_bar_count"], e["exact_id"]))
    ranking_aliased = sorted(entries, key=lambda e: (-e["union_train_bar_count"], e["exact_id"]))

    exact_pos = {e["symbol"]: i for i, e in enumerate(ranking_exact)}
    union_pos = {e["symbol"]: i for i, e in enumerate(ranking_aliased)}
    rank_changes = [
        {"symbol": s, "exact_rank": exact_pos[s], "union_rank": union_pos[s]}
        for s in exact_pos if exact_pos[s] != union_pos[s]
    ]

    n = cfg.n_universe
    exact_top = {e["symbol"] for e in ranking_exact[:n]}
    union_top = {e["symbol"] for e in ranking_aliased[:n]}
    top_entered = sorted(union_top - exact_top)
    top_left = sorted(exact_top - union_top)

    chosen = ranking_aliased[:n]
    return {
        "symbols": [e["symbol"] for e in chosen],
        "asset_ids": [e["exact_id"] for e in chosen],
        "alias_groups": [e["asset_ids"] for e in chosen],
        "ranking_aliased": ranking_aliased,
        "ranking_exact": ranking_exact,
        "rank_changes": rank_changes,
        "top64_entered": top_entered,
        "top64_left": top_left,
        "unknown": unknown,
    }


def _forward_back_fill(ohlcv: np.ndarray) -> None:
    """In-place forward-fill (carry last real bar) then leading-gap back-fill, per asset column.
    Byte-for-byte the same policy as `probe.build_aligned_ohlcv` (pinned by
    `test_alias_read_layer.py::test_single_id_slot_matches_probe_byte_for_byte`) — duplicated here
    because that function couples this fill loop to its own single-id row-load loop with no
    seam to reuse just the fill, and editing `probe.py` to add one is out of scope (module
    docstring)."""
    T, n_assets, _ = ohlcv.shape
    close = ohlcv[:, :, 3]
    have = ~np.isnan(close)
    for col in range(n_assets):
        valid_rows = np.where(have[:, col])[0]
        if len(valid_rows) == 0:
            continue
        last = None
        for r in range(T):
            if have[r, col]:
                last = ohlcv[r, col].copy()
            elif last is not None:
                ohlcv[r, col] = last
        first_r = valid_rows[0]
        if first_r > 0:
            ohlcv[:first_r, col] = ohlcv[first_r, col]


def build_aligned_ohlcv_with_alias_union(
    cfg: "probe.Config", asset_ids: Sequence[int], alias_groups: Sequence[Sequence[int]],
    bar_sec: int,
) -> Tuple[np.ndarray, np.ndarray, List[Dict], List[Dict[int, int]]]:
    """Read-time replacement for `probe.build_aligned_ohlcv(cfg, asset_ids, bar_sec)`.

    For a slot whose alias group has exactly one id this is byte-for-byte what
    `probe.build_aligned_ohlcv` would have produced for that column. For a slot with >1 id, the
    900k-row `asset_aliases.alias_agreement_gate` runs ONCE over the full grid span for every
    multi-id group (fail-loud on disagreement, fail-CLOSED on an untested/insufficient overlap —
    `AliasDisagreementError` / `AliasInsufficientOverlapError` propagate uncaught: a group this
    gate cannot certify must not silently union), and every id's rows in the group are then
    loaded into that one column (an agreeing duplicate timestamp is written twice per the
    module's own DEDUPE convention — order-independent since every admitted pair agrees to
    <=1e-12 relative, per `asset_aliases`'s module docstring).

    Returns (ohlcv, grid_ts, alias_gate_disclosure, per_slot_id_row_counts) — the last two purely
    for artifact disclosure (gate iii's per-cell requirement)."""
    n_assets = len(asset_ids)
    grid_start = probe._to_epoch(cfg.train_start) - cfg.W * bar_sec
    grid_end = probe._to_epoch(cfg.test_end)
    grid_ts = np.arange(grid_start, grid_end, bar_sec, dtype=np.int64)
    T = len(grid_ts)
    ts_to_row = {int(t): i for i, t in enumerate(grid_ts)}

    ohlcv = np.full((T, n_assets, 5), np.nan, dtype=np.float64)

    con = _connect_ro(cfg.db_path)
    try:
        multi_groups = [list(g) for g in alias_groups if len(g) > 1]
        alias_gate_disclosure = (
            aa.alias_agreement_gate(con, "ohlcv", multi_groups, grid_start, grid_end)
            if multi_groups else []
        )

        cur = con.cursor()
        per_slot_id_row_counts: List[Dict[int, int]] = []
        for col, ids in enumerate(alias_groups):
            counts: Dict[int, int] = {}
            for aid in ids:
                cur.execute(
                    "SELECT timestamp, open, high, low, close, volume FROM ohlcv "
                    "WHERE asset_id=? AND timestamp>=? AND timestamp<? ORDER BY timestamp",
                    (int(aid), int(grid_start), int(grid_end)),
                )
                rows = cur.fetchall()
                counts[int(aid)] = len(rows)
                for ts, o, h, lo, c, v in rows:
                    r = ts_to_row.get(int(ts))
                    if r is not None:
                        ohlcv[r, col, 0] = o
                        ohlcv[r, col, 1] = h
                        ohlcv[r, col, 2] = lo
                        ohlcv[r, col, 3] = c
                        ohlcv[r, col, 4] = v
            per_slot_id_row_counts.append(counts)
    finally:
        con.close()

    _forward_back_fill(ohlcv)
    return ohlcv, grid_ts, alias_gate_disclosure, per_slot_id_row_counts


def alias_material_slot_report(
    cfg: "probe.Config", windows: Sequence[Tuple[str, str, str]],
    material_symbols: Sequence[str] = MATERIAL_ALIAS_SYMBOLS,
) -> List[Dict]:
    """Per-window before/after row-count disclosure for the 6 D56-material slots — exact-spelling
    id alone vs the alias-union total, over EACH window this sweep actually runs (not just the
    original 809 window). Read-only, cheap (indexed COUNT queries), used only to populate the
    artifact; the sweep's own panel build (`build_aligned_ohlcv_with_alias_union`) is what
    actually feeds the cells."""
    con = _connect_ro(cfg.db_path)
    try:
        cur = con.cursor()
        cur.execute("SELECT id, symbol FROM assets")
        asset_rows = cur.fetchall()
        resolution, _ = aa.resolve_requested_symbols(asset_rows, material_symbols)
        out = []
        for r in resolution:
            exact_id = r["exact_id"]
            if exact_id is None:
                continue
            twin_ids = [aid for aid in r["asset_ids"] if aid != exact_id]
            for wname, wstart, wend in windows:
                s_ts, e_ts = probe._to_epoch(wstart), probe._to_epoch(wend)
                cur.execute(
                    "SELECT COUNT(*) FROM ohlcv WHERE asset_id=? AND timestamp>=? AND timestamp<?",
                    (exact_id, s_ts, e_ts))
                exact_rows = int(cur.fetchone()[0])
                twin_rows_total = 0
                shared_total = 0
                for tid in twin_ids:
                    cur.execute(
                        "SELECT COUNT(*) FROM ohlcv WHERE asset_id=? AND timestamp>=? "
                        "AND timestamp<?", (tid, s_ts, e_ts))
                    twin_rows_total += int(cur.fetchone()[0])
                    cur.execute(
                        "SELECT COUNT(*) FROM ohlcv a JOIN ohlcv b ON a.timestamp=b.timestamp "
                        "WHERE a.asset_id=? AND b.asset_id=? AND a.timestamp>=? "
                        "AND a.timestamp<?", (exact_id, tid, s_ts, e_ts))
                    shared_total += int(cur.fetchone()[0])
                union_rows = exact_rows + twin_rows_total - shared_total
                missing_via_twin = twin_rows_total - shared_total
                out.append({
                    "symbol": r["symbol"], "window": wname, "exact_id": exact_id,
                    "twin_ids": twin_ids, "exact_rows": exact_rows, "twin_rows": twin_rows_total,
                    "shared_rows": shared_total, "union_rows": union_rows,
                    "missing_rows_via_twin": missing_via_twin,
                    "missing_pct_of_union": (100.0 * missing_via_twin / union_rows)
                    if union_rows else 0.0,
                })
        return out
    finally:
        con.close()
