#!/usr/bin/env python3
"""TASK_897 D56(a) — the read-time alias-union layer (`alias_read_layer.py`).

Builds throwaway SQLite databases (no shared file touched, let alone written — the same
discipline `runs/877_xbt_alias_fold/test_xbt_alias_fold.py`'s in-memory demonstrations use)
and a scratch `benchmark_periods.json`, so these tests never depend on `trading_data.db` or
`fixtures/trading_fixture.db` carrying any particular row.

Run: python -m pytest experiments/897_corrected_809/test_alias_read_layer.py -v
"""
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import alias_read_layer as al  # noqa: E402
import probe  # noqa: E402  (806 lineage — read-only helpers `Config`/`_to_epoch`, never edited)

from mcts_trader.data.asset_aliases import (  # noqa: E402
    AliasDisagreementError, AliasInsufficientOverlapError,
)

DAY = 24 * 3600
BAR_SEC = 60


def _make_db(path, rows_by_asset, assets=None):
    """rows_by_asset: {asset_id: [(ts, o, h, l, c, v), ...]}. `assets` defaults to
    {aid: f"SYM{aid}"} — callers needing specific spellings pass an explicit {id: symbol} map."""
    con = sqlite3.connect(str(path))
    cur = con.cursor()
    cur.execute("CREATE TABLE assets (id INTEGER PRIMARY KEY, symbol TEXT)")
    cur.execute("CREATE TABLE ohlcv (asset_id INTEGER, timestamp INTEGER, open REAL, high REAL, "
               "low REAL, close REAL, volume REAL)")
    if assets is None:
        assets = {aid: f"SYM{aid}" for aid in rows_by_asset}
    cur.executemany("INSERT INTO assets VALUES (?, ?)", list(assets.items()))
    for aid, rows in rows_by_asset.items():
        cur.executemany(
            "INSERT INTO ohlcv VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(aid,) + tuple(r) for r in rows],
        )
    con.commit()
    con.close()


def _make_periods_file(path, benchmark_assets):
    path.write_text(json.dumps({"periods": [{"benchmark_assets": benchmark_assets}]}))


def _bar(ts, close, vol=1.0):
    return (ts, close, close, close, close, vol)


# ---------------------------------------------------------------- rank_universe_with_alias_union

def test_union_can_flip_top_n_membership(tmp_path):
    """AAA/USD (exact) alone has fewer train-window rows than BBB/USD, so the un-aliased ranking
    would pick BBB/USD for a top-1 universe. AAA/USD's alias twin AAAUSD carries enough
    additional train-window rows (non-overlapping timestamps, so the agreement gate is not even
    exercised at ranking time — ranking never calls the gate, matching
    `asset_aliases.ranked_minute_universe`'s own convention) that the UNIONED count overtakes
    BBB/USD. This is exactly the D56 finding stated generally: the universe is rank-ordered by
    bar count, so a union read can change top-N membership, not just patch holes in a fixed set."""
    db = tmp_path / "db.sqlite"
    train_start = "2023-01-01"
    val_start = "2023-02-01"
    tr_s = probe._to_epoch(train_start)
    aaa_rows = [_bar(tr_s + i * BAR_SEC, 100.0) for i in range(5)]              # 5 rows
    aaa_twin_rows = [_bar(tr_s + (1000 + i) * BAR_SEC, 100.0) for i in range(10)]  # +10, disjoint ts
    bbb_rows = [_bar(tr_s + i * BAR_SEC, 50.0) for i in range(8)]               # 8 rows
    _make_db(db, {1: aaa_rows, 2: aaa_twin_rows, 3: bbb_rows},
             assets={1: "AAA/USD", 2: "AAAUSD", 3: "BBB/USD"})
    periods = tmp_path / "benchmark_periods.json"
    _make_periods_file(periods, ["AAA/USD", "BBB/USD"])

    cfg = probe.Config(db_path=str(db), benchmark_periods=str(periods),
                       train_start=train_start, val_start=val_start,
                       test_start="2023-02-01", test_end="2023-02-02", n_universe=1)

    result = al.rank_universe_with_alias_union(cfg)
    assert result["symbols"] == ["AAA/USD"]          # union (15) beats BBB/USD's exact (8)
    assert result["asset_ids"] == [1]
    assert sorted(result["alias_groups"][0]) == [1, 2]
    assert result["top64_entered"] == ["AAA/USD"]
    assert result["top64_left"] == ["BBB/USD"]
    changed = {c["symbol"] for c in result["rank_changes"]}
    assert {"AAA/USD", "BBB/USD"} <= changed


def test_no_alias_twin_reproduces_the_unaliased_ranking(tmp_path):
    """A candidate pool with no duplicate normalized keys at all must rank identically to
    `probe.load_universe_trainlocked` — the aliased path must not perturb the ordinary case."""
    db = tmp_path / "db.sqlite"
    train_start = "2023-01-01"
    tr_s = probe._to_epoch(train_start)
    _make_db(db, {1: [_bar(tr_s + i * BAR_SEC, 1.0) for i in range(9)],
                 2: [_bar(tr_s + i * BAR_SEC, 1.0) for i in range(4)]},
             assets={1: "AAA/USD", 2: "BBB/USD"})
    periods = tmp_path / "benchmark_periods.json"
    _make_periods_file(periods, ["AAA/USD", "BBB/USD"])
    cfg = probe.Config(db_path=str(db), benchmark_periods=str(periods),
                       train_start=train_start, val_start="2023-02-01",
                       test_start="2023-02-01", test_end="2023-02-02", n_universe=2)

    aliased = al.rank_universe_with_alias_union(cfg)
    orig_symbols, orig_asset_ids, _ = probe.load_universe_trainlocked(cfg)
    assert aliased["symbols"] == orig_symbols == ["AAA/USD", "BBB/USD"]
    assert aliased["asset_ids"] == orig_asset_ids == [1, 2]
    assert aliased["rank_changes"] == []
    assert aliased["top64_entered"] == aliased["top64_left"] == []


def test_requested_symbol_absent_from_assets_is_unknown(tmp_path):
    db = tmp_path / "db.sqlite"
    tr_s = probe._to_epoch("2023-01-01")
    _make_db(db, {1: [_bar(tr_s, 1.0)]}, assets={1: "AAA/USD"})
    periods = tmp_path / "benchmark_periods.json"
    _make_periods_file(periods, ["AAA/USD", "ZZZ/USD"])
    cfg = probe.Config(db_path=str(db), benchmark_periods=str(periods),
                       train_start="2023-01-01", val_start="2023-02-01",
                       test_start="2023-02-01", test_end="2023-02-02", n_universe=2)
    result = al.rank_universe_with_alias_union(cfg)
    assert result["unknown"] == ["ZZZ/USD"]
    assert result["symbols"] == ["AAA/USD"]


# ---------------------------------------------------------------- build_aligned_ohlcv_with_alias_union

def _base_cfg(db_path, train_start="2023-01-01", val_start="2023-01-02",
             test_start="2023-01-02", test_end="2023-01-03", W=2):
    return probe.Config(db_path=str(db_path), train_start=train_start, val_start=val_start,
                       test_start=test_start, test_end=test_end, W=W)


def test_single_id_slot_matches_probe_byte_for_byte(tmp_path):
    """A slot whose alias group has exactly one id must reproduce
    `probe.build_aligned_ohlcv` byte-for-byte — the read-time layer changes nothing for the
    ordinary (non-aliased) case."""
    db = tmp_path / "db.sqlite"
    tr_s = probe._to_epoch("2023-01-01")
    rows = [_bar(tr_s + i * BAR_SEC, 100.0 + i) for i in range(0, 200, 7)]  # gappy on purpose
    _make_db(db, {1: rows}, assets={1: "AAA/USD"})
    cfg = _base_cfg(db)

    ohlcv_probe, grid_ts_probe = probe.build_aligned_ohlcv(cfg, [1], BAR_SEC)
    ohlcv_alias, grid_ts_alias, gate_disclosure, counts = al.build_aligned_ohlcv_with_alias_union(
        cfg, [1], [[1]], BAR_SEC)

    assert np.array_equal(grid_ts_probe, grid_ts_alias)
    assert np.allclose(ohlcv_probe, ohlcv_alias, equal_nan=True)
    assert gate_disclosure == []  # no multi-id group -> gate never runs
    assert counts == [{1: len(rows)}]


def test_multi_id_slot_unions_rows_from_both_ids(tmp_path):
    """Two ids in one alias group, DISJOINT in time WITHIN THE GRID WINDOW (the real 868/877
    shape: the twin carries the history the exact spelling is missing over the span of
    interest) — the panel column must carry BOTH spans, not just the exact-spelling id's.

    The evidence floor (`MIN_ALIAS_OVERLAP_TS`) is satisfied out-of-band: >=64 agreeing shared
    timestamps FAR BEFORE the grid window (basis 'full_history', exactly the real 868/877
    fallback for pairs whose live spans don't overlap in the window being read) — the grid
    query itself never touches those out-of-window rows (`WHERE timestamp>=grid_start AND
    timestamp<grid_end`), so they exist purely to let the agreement gate certify the union
    without perturbing what lands in `ohlcv`."""
    db = tmp_path / "db.sqlite"
    tr_s = probe._to_epoch("2023-01-01")
    evidence_ts0 = tr_s - 10_000_000
    evidence = [_bar(evidence_ts0 + i * BAR_SEC, 5.0) for i in range(70)]         # far pre-grid
    early = [_bar(tr_s + i * BAR_SEC, 10.0) for i in range(5)]                    # id 1 (exact)
    late = [_bar(tr_s + (50 + i) * BAR_SEC, 20.0) for i in range(5)]              # id 2 (twin)
    _make_db(db, {1: early + evidence, 2: late + evidence},
             assets={1: "AAA/USD", 2: "AAAUSD"})
    cfg = _base_cfg(db, test_end="2023-01-02")

    ohlcv, grid_ts, gate_disclosure, counts = al.build_aligned_ohlcv_with_alias_union(
        cfg, [1], [[1, 2]], BAR_SEC)

    ts_to_row = {int(t): i for i, t in enumerate(grid_ts)}
    for ts, _, _, _, close, _ in early:
        assert ohlcv[ts_to_row[ts], 0, 3] == 10.0
    for ts, _, _, _, close, _ in late:
        assert ohlcv[ts_to_row[ts], 0, 3] == 20.0
    assert counts == [{1: 5, 2: 5}]  # the out-of-window evidence rows never enter this count
    assert len(gate_disclosure) == 1
    assert gate_disclosure[0]["asset_ids"] == [1, 2]
    assert gate_disclosure[0]["overlaps"][0]["evidence"]["basis"] == "full_history"


def test_agreeing_overlap_is_gated_and_passes(tmp_path):
    """Enough agreeing shared timestamps (>= MIN_ALIAS_OVERLAP_TS) certifies the union; the
    disclosure records the evidence basis."""
    from mcts_trader.data.asset_aliases import MIN_ALIAS_OVERLAP_TS
    db = tmp_path / "db.sqlite"
    tr_s = probe._to_epoch("2023-01-01")
    n = MIN_ALIAS_OVERLAP_TS + 10
    shared = [_bar(tr_s + i * BAR_SEC, 42.0) for i in range(n)]  # both ids agree exactly
    _make_db(db, {1: shared, 2: shared}, assets={1: "AAA/USD", 2: "AAAUSD"})
    cfg = _base_cfg(db, test_end="2023-01-02")

    ohlcv, grid_ts, gate_disclosure, counts = al.build_aligned_ohlcv_with_alias_union(
        cfg, [1], [[1, 2]], BAR_SEC)
    assert gate_disclosure[0]["overlaps"][0]["evidence"]["basis"] == "window"
    assert gate_disclosure[0]["overlaps"][0]["shared_ts"] == n


def test_disagreeing_overlap_refuses_fail_loud(tmp_path):
    """A real disagreement on a shared timestamp must REFUSE, never average or silently prefer
    one side — `AliasDisagreementError` propagates uncaught out of the panel builder."""
    from mcts_trader.data.asset_aliases import MIN_ALIAS_OVERLAP_TS
    db = tmp_path / "db.sqlite"
    tr_s = probe._to_epoch("2023-01-01")
    n = MIN_ALIAS_OVERLAP_TS + 10
    rows_a = [_bar(tr_s + i * BAR_SEC, 42.0) for i in range(n)]
    rows_b = [_bar(tr_s + i * BAR_SEC, 42.0) for i in range(n)]
    rows_b[3] = _bar(tr_s + 3 * BAR_SEC, 999.0)  # disagreement on ts index 3
    _make_db(db, {1: rows_a, 2: rows_b}, assets={1: "AAA/USD", 2: "AAAUSD"})
    cfg = _base_cfg(db, test_end="2023-01-02")

    with pytest.raises(AliasDisagreementError):
        al.build_aligned_ohlcv_with_alias_union(cfg, [1], [[1, 2]], BAR_SEC)


def test_insufficient_overlap_refuses_fail_closed(tmp_path):
    """Both ids carry rows (a real blend, not a pure relabel) but share too few timestamps for
    the agreement gate to have tested the 'same instrument' premise — refuses rather than bless
    an untested union."""
    db = tmp_path / "db.sqlite"
    tr_s = probe._to_epoch("2023-01-01")
    rows_a = [_bar(tr_s + i * BAR_SEC, 1.0) for i in range(5)]
    rows_b = [_bar(tr_s + (10000 + i) * BAR_SEC, 2.0) for i in range(5)]  # far outside the grid
    _make_db(db, {1: rows_a, 2: rows_b}, assets={1: "AAA/USD", 2: "AAAUSD"})
    cfg = _base_cfg(db, test_end="2023-01-02")

    with pytest.raises(AliasInsufficientOverlapError):
        al.build_aligned_ohlcv_with_alias_union(cfg, [1], [[1, 2]], BAR_SEC)


# ---------------------------------------------------------------- alias_material_slot_report

def test_material_slot_report_matches_hand_computed_counts(tmp_path):
    db = tmp_path / "db.sqlite"
    w_start, w_end = "2023-01-01", "2023-01-02"
    s, e = probe._to_epoch(w_start), probe._to_epoch(w_end)
    exact_rows = [_bar(s + i * BAR_SEC, 1.0) for i in range(0, 100, 2)]      # 50 rows, evens
    twin_rows = [_bar(s + i * BAR_SEC, 1.0) for i in range(0, 100, 3)]       # rows at multiples of 3
    _make_db(db, {1: exact_rows, 2: twin_rows}, assets={1: "AAA/USD", 2: "AAAUSD"})
    cfg = _base_cfg(db)

    shared = len(set(r[0] for r in exact_rows) & set(r[0] for r in twin_rows))
    union_expected = len(exact_rows) + len(twin_rows) - shared
    missing_expected = len(twin_rows) - shared

    out = al.alias_material_slot_report(cfg, [("w", w_start, w_end)],
                                        material_symbols=["AAA/USD"])
    assert len(out) == 1
    r = out[0]
    assert r["exact_rows"] == len(exact_rows)
    assert r["twin_rows"] == len(twin_rows)
    assert r["shared_rows"] == shared
    assert r["union_rows"] == union_expected
    assert r["missing_rows_via_twin"] == missing_expected


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
