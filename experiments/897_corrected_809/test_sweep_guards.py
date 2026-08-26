#!/usr/bin/env python3
"""TASK_897 — the sweep harness's CLI refusals and the artifact's honesty claims.

Four things are pinned here, all of them fix-pass findings:

  * the 806/808/809 lineage itself must stay free of alias handling — D56's union is applied as
    a READ-TIME layer (`alias_read_layer.py`) precisely so the lineage is not edited, and the
    artifact says so; this test re-reads the lineage source so that wording cannot drift;
  * the artifact must state PROMINENTLY that its RAW CI is UNCALIBRATED, and must not drop the
    qualifier on any cell whose `(block, n_dec)` pairing the calibration does not cover;
  * `apply_ci_calibration` — the function that decides EVERY real cell's `ci_lo_ADVISORY`/
    `ci_hi_ADVISORY`/`ci_advisory` (renamed from `ci_lo`/`ci_hi`/`ci_calibrated` by TASK_897 D57,
    2026-08-22 — the pair is never band-readable on this leg even when finite) — must be pinned
    directly, guard by guard (D55(a) fix pass, MAJOR 2);
  * `--db-path` must not be able to reach the protected 24GB `trading_data.db` BY OMISSION,
    and every CLI refusal must be a clean message rather than a raw traceback.
"""
import os
import stat
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

pytest.importorskip("numpy")

_LINEAGE = _REPO / "runs" / "806_predictability_probe" / "probe.py"
pytestmark = pytest.mark.skipif(
    not _LINEAGE.exists(),
    reason=f"806/808/809 lineage not present at {_LINEAGE} (gitignored runs/)",
)


def _sweep():
    import sweep_gated_maker
    return sweep_gated_maker


def _artifact_meta(n_cells=1, tiers=(8.0,)):
    """The minimal `meta` `write_artifact` reads, for tests that drive it with hand-built cells."""
    sw = _sweep()
    return {
        "attainable_maker_fee_bp_tiers": list(tiers), "db_path": "fixtures/trading_fixture.db",
        "bar_sec": 60, "n_assets": 1, "n_universe": 1, "symbols": ["A"],
        "alias_union_applied": True, "alias_groups": {"A": [1]},
        "alias_material_slots": list(sw.alias_layer.MATERIAL_ALIAS_SYMBOLS),
        "alias_gate_disclosure": [], "alias_per_slot_id_row_counts": {"A": {1: 10}},
        "alias_material_slot_report": [], "alias_rank_changes": [], "alias_top64_entered": [],
        "alias_top64_left": [], "alias_unknown_symbols": [],
        "train_start": "2023-05-16", "val_start": "2024-06-01",
        "windows": [{"name": "a", "start": "x", "end": "y"}], "grid": {"H": [5]},
        "grid_epoch": {"start": 0, "end": 1}, "grid_start_iso": "1970-01-01T00:00:00Z",
        "grid_end_iso": "1970-01-01T00:00:01Z", "seed": 897, "n_boot": 10,
        "n_cells_expected": n_cells, "n_cells_recorded": n_cells, "n_cells_ok": n_cells,
        "n_cells_dropped_or_failed": 0, "runtime_s": 0.0, "estimator": "per-decision",
        "maker_fill_model": "OHLC-touch",
    }


# ---------------------------------------------------------------- CLI refusals (MINOR fixes)

def test_db_path_cannot_be_reached_by_omission():
    sw = _sweep()
    with pytest.raises(SystemExit) as e:
        sw.assert_db_path_allowed(None, False)
    assert "--db-path is required and has no default" in str(e.value)


def test_protected_real_db_refuses_without_the_explicit_acknowledgement():
    sw = _sweep()
    real = str(sw.PROTECTED_REAL_DB)
    with pytest.raises(SystemExit) as e:
        sw.assert_db_path_allowed(real, False)
    assert "PROTECTED read-only source of record" in str(e.value)
    # ...and is reachable only when explicitly named AND acknowledged. TASK_902 item 10 (TOCTOU
    # fix): the guard now returns the RESOLVED path (here, the protected DB's own canonical
    # realpath — a fixed constant) rather than echoing back the raw operator spelling, so a
    # symlink cannot be re-targeted between this check and the caller's later open.
    assert sw.assert_db_path_allowed(real, True) == os.path.realpath(real)
    # a relative spelling of the same file resolves to the same refusal
    with pytest.raises(SystemExit):
        sw.assert_db_path_allowed("./trading_data.db", False)


def test_fixture_db_path_is_allowed():
    sw = _sweep()
    p = str(_REPO / "fixtures" / "trading_fixture.db")
    # TASK_902 item 10: resolved, not raw-echoed (see the protected-DB test above).
    assert sw.assert_db_path_allowed(p, False) == os.path.realpath(p)


# ------------------------------------------------------------ A1: the guard was URI-defeatable
#
# THE DEFECT, D57 fix cycle 4 item A1: `assert_db_path_allowed('trading_data.db?x=1', False)`
# PASSED, because `Path("trading_data.db?x=1") != Path("trading_data.db")`. Both belts failed on
# the SAME string: `alias_read_layer._connect_ro` built `file:trading_data.db?x=1?mode=ro`, which
# sqlite parses as the single parameter `x="1?mode=ro"` — `mode` is never set, so the handle opens
# READ-WRITE-CREATE. A judge demonstrated `CREATE TABLE` through it. That is a writable handle on
# the protected read-only source of record.

@pytest.mark.parametrize("spelling", [
    "trading_data.db?x=1",          # the demonstrated defeat
    "trading_data.db?mode=rw",      # the explicit one
    "trading_data.db#frag",         # URI fragment
    "file:trading_data.db",         # a URI, not a path
    "FILE:trading_data.db",         # ...case-insensitively
    "fixtures/trading_fixture.db?x=1",   # even on an otherwise-honest file
    # D59 fix cycle 3, MAJOR — the URI-AUTHORITY spelling. sqlite discards a leading
    # `//<authority>` when the authority is empty or exactly `localhost`, so these name the
    # protected DB to sqlite while looking like a different path to `os.path.realpath`.
    "//localhost/home/lukas/Trader/trading_data.db",
    "///home/lukas/Trader/trading_data.db",
    "//localhost/anything_at_all.db",     # ...refused on its spelling, not on its target
])
def test_db_path_with_uri_syntax_is_refused_outright(spelling):
    """A DB path is a filesystem path, never a URI. Each spelling refuses with an actionable
    message. PERTURBATION: drop the `?`/`#`/`file:` check and the first spelling PASSES the
    guard, which is the whole finding."""
    sw = _sweep()
    with pytest.raises(SystemExit) as e:
        sw.assert_db_path_allowed(spelling, False)
    msg = str(e.value)
    assert "is not a filesystem path" in msg
    assert "READ-WRITE-CREATE" in msg          # says WHY, not just "no"
    assert "fixtures/trading_fixture.db" in msg  # says what to do instead


def test_uri_syntax_is_refused_even_with_the_scheduled_run_acknowledgement():
    """The acknowledgement flag authorizes the REAL DB, not URI syntax — a shifted `mode=ro` is a
    writable handle whether or not the operator meant to touch the real DB."""
    sw = _sweep()
    with pytest.raises(SystemExit):
        sw.assert_db_path_allowed("trading_data.db?x=1", True)


def test_a_uri_authority_prefix_cannot_walk_a_protected_db_past_the_gate(tmp_path, monkeypatch):
    """D59 fix cycle 3, MAJOR. THE DEFECT, measured with `PROTECTED_REAL_DB` monkeypatched to a
    decoy so nothing points at the real 24GB DB:

        assert_db_path_allowed('<decoy>',            scheduled=False) -> REFUSED  (correct)
        assert_db_path_allowed('//localhost<decoy>', scheduled=False) -> PASSED
           ...and `_connect_ro` on the returned string OPENED the protected decoy.

    `realpath('//localhost/…')` is not the protected realpath and `samefile` raises `OSError`
    (swallowed), so BOTH identity belts missed it — the same one-string-defeats-both-belts class
    `assert_db_path_allowed`'s own D57 note says must never recur. Read-only, so the source of
    record is never written, but it defeats what this gate is FOR: real-data validation is
    SCHEDULED, never polled from a factory or interactive session.

    PERTURBATION THIS IS RED AGAINST: remove `"//"` from `DB_PATH_FORBIDDEN_PREFIXES` AND drop
    the `normalize_db_path` call from the identity check, and the `//localhost` spelling PASSES
    (mutation evidence recorded in the task report). Removing EITHER one alone still refuses —
    that is the point of two independent belts."""
    import sqlite3
    sw = _sweep()
    decoy = tmp_path / "decoy_protected.db"
    sqlite3.connect(str(decoy)).execute("CREATE TABLE t (a int)")
    monkeypatch.setattr(sw, "PROTECTED_REAL_DB", decoy)

    with pytest.raises(SystemExit) as plain:            # the honest spelling already refused
        sw.assert_db_path_allowed(str(decoy), False)
    assert "PROTECTED read-only source of record" in str(plain.value)

    for spelling in ("//localhost" + str(decoy), "//" + str(decoy).lstrip("/"),
                     "///" + str(decoy).lstrip("/")):
        with pytest.raises(SystemExit) as e:
            sw.assert_db_path_allowed(spelling, False)
        assert "is not a filesystem path" in str(e.value), spelling

    # ...and belt 2 alone (identity on the normalized string) would catch it too: the normalized
    # spelling IS the protected path, which is the comparison the gate now makes.
    assert sw.alias_layer.normalize_db_path("//localhost" + str(decoy)) == str(decoy)


def test_connect_ro_cannot_be_made_writable_by_a_question_mark_in_the_path(tmp_path):
    """BELT 2, independent of belt 1: `_connect_ro` percent-quotes the path, so a `?` becomes an
    ordinary path character and `mode=ro` is always the first and only URI parameter.

    PERTURBATION THIS IS RED AGAINST — the pre-fix body, asserted here to have been genuinely
    exploitable so the test is not vacuous: `sqlite3.connect(f"file:{p}?x=1?mode=ro", uri=True)`
    on the same decoy file DOES create a table."""
    import sqlite3
    import alias_read_layer as al
    decoy = tmp_path / "decoy.db"
    sqlite3.connect(str(decoy)).close()

    # the old body really was writable (documents the perturbation, on a throwaway file)
    old = sqlite3.connect(f"file:{decoy}?x=1?mode=ro", uri=True)
    old.execute("CREATE TABLE pwned_by_the_old_body (a int)")
    old.commit()
    old.close()

    # the fixed body refuses: the quoted name is a path that does not exist, and mode=ro will
    # not create one.
    with pytest.raises(sqlite3.OperationalError):
        al._connect_ro(str(decoy) + "?x=1").execute("CREATE TABLE pwned (a int)")
    # ...and the honest path still opens, still read-only.
    con = al._connect_ro(str(decoy))
    assert con.execute("select 1").fetchone() == (1,)
    with pytest.raises(sqlite3.OperationalError):
        con.execute("CREATE TABLE nope (a int)")


def test_connect_ro_reads_the_real_fixture():
    """The honest case, on the real fixture: quoting must not break a normal path (nor an
    absolute one with directory separators in it)."""
    import alias_read_layer as al
    fx = _REPO / "fixtures" / "trading_fixture.db"
    if not fx.exists():
        pytest.skip("fixture not present")
    con = al._connect_ro(str(fx))
    n = con.execute("select count(*) from sqlite_master").fetchone()[0]
    assert n > 0


# ------------------------------------------------------- B1: gate (iv)'s unenforced grid subset

def test_run_sweep_refuses_attainable_tiers_the_grid_never_measures():
    """D57 fix cycle 4, item B1. `GRID_FEE_BP = (0, 8, 16)` and the ONLY gate-(iv) check was
    non-emptiness, so Kraken's real ladder `--attainable-maker-fee-bp 14,12,10` ran to completion
    with NOT ONE declared tier measured — while the artifact told the reader the tiers were "the
    Kraken-reachable subset of the fee grid below" and the planner read CELL CLEARS off 8bp cells.
    8 < 10: STRICTLY BETTER THAN REALITY, i.e. the optimistic direction.

    PERTURBATION: delete the `missing_tiers` check and this call proceeds to open the DB."""
    sw = _sweep()
    with pytest.raises(ValueError) as e:
        sw.run_sweep(attainable_maker_fee_bp=[14.0, 12.0, 10.0],
                     db_path=str(_REPO / "fixtures" / "trading_fixture.db"))
    msg = str(e.value)
    assert "GATE (iv) REFUSAL" in msg
    assert "[14.0, 12.0, 10.0]" in msg          # names the offending tiers
    assert "[0.0, 8.0, 16.0]" in msg            # ...and the grid
    assert "optimistic" in msg
    # a PARTIAL miss refuses too — one measured tier does not license the unmeasured ones
    with pytest.raises(ValueError) as e2:
        sw.run_sweep(attainable_maker_fee_bp=[8.0, 10.0],
                     db_path=str(_REPO / "fixtures" / "trading_fixture.db"))
    assert "[10.0]" in str(e2.value)


def test_run_sweep_accepts_tiers_the_grid_does_measure():
    """The honest case must get PAST gate (iv). Asserted without paying for a real run: every
    tier of the default grid, and every subset of it, passes the subset check."""
    sw = _sweep()
    for tiers in ([8.0], [16.0], [0.0], [8.0, 16.0], list(sw.GRID_FEE_BP)):
        missing = [t for t in tiers if t not in [float(x) for x in sw.GRID_FEE_BP]]
        assert missing == []
    # ...and the artifact's "subset" sentence now names the enforcement rather than asserting it
    import inspect
    src = inspect.getsource(sw.write_artifact)
    assert "'subset' is ENFORCED, not asserted" in src


# -------------------------------------------------------- B3: window parsing accepted garbage

@pytest.mark.parametrize("raw,needle", [
    ("not-a-date,also-not", "is not an ISO date"),
    ("2024-11-01,also-not", "is not an ISO date"),
    ("2024-13-01,2025-04-01", "is not an ISO date"),
    ("20241101,2025-04-01", "is not an ISO date"),
    ("2025-04-01,2024-11-01", "START must be strictly BEFORE END"),
    ("2024-11-01,2024-11-01", "START must be strictly BEFORE END"),
])
def test_window_parsing_refuses_unparseable_and_inverted_bounds(raw, needle):
    """D57 fix cycle 4, item B3. `_parse_window` used to accept both spellings; `_window_slice`
    then selected ZERO decisions with no refusal, every cell was still recorded `ok`, and the
    artifact reported `recorded: 144 (ok=144, dropped/failed=0)` with no GATE (ii) violation and
    exit 0 — the 863 green-looking-artifact shape, one operator typo away on the srv leg.

    Exit-code semantics are deliberately NOT touched (a planner ruling, out of scope): this is a
    parse-time refusal before any data is read."""
    sw = _sweep()
    with pytest.raises(SystemExit) as e:
        sw._parse_window("--window-a", raw)
    assert needle in str(e.value)


def test_window_parsing_still_accepts_the_shipped_defaults():
    """No honest window may start refusing as a side effect — including the two REAL_WINDOWS the
    srv leg runs, read back through the parser in the exact CLI spelling."""
    sw = _sweep()
    assert sw._parse_window("--window-a", "2024-11-01,2025-04-01") == ("2024-11-01", "2025-04-01")
    assert sw._parse_window("--window-b", "2025-04-01,2026-03-01") == ("2025-04-01", "2026-03-01")
    for _name, start, end in sw.REAL_WINDOWS:
        assert sw._parse_window("--window-a", f"{start},{end}") == (start, end)
    # whitespace around the bounds is still tolerated
    assert sw._parse_window("--window-a", " 2024-11-01 , 2025-04-01 ") == ("2024-11-01",
                                                                          "2025-04-01")


def test_relative_db_path_refuses_the_protected_db_regardless_of_cwd(tmp_path, monkeypatch):
    """D57 fix pass item 3: `assert_db_path_allowed` used to resolve a relative `--db-path`
    against the PROCESS CWD only, so a caller invoked from a staging directory (the srv leg
    legitimately stages files elsewhere) typing a repo-relative spelling of the protected DB was
    NOT refused unless the process happened to be running from the repo root. Refuse if EITHER
    the cwd-relative OR the `_REPO`-relative resolution lands on the protected DB — strictly more
    refusing, never less. Both relative spellings are checked, from a non-repo-root cwd."""
    sw = _sweep()
    monkeypatch.chdir(tmp_path)
    for spelling in ("trading_data.db", "./trading_data.db"):
        with pytest.raises(SystemExit) as e:
            sw.assert_db_path_allowed(spelling, False)
        assert "PROTECTED read-only source of record" in str(e.value)
    # ...and still reachable when explicitly acknowledged, from that same non-repo-root cwd.
    # TASK_902 item 10 (TOCTOU fix): the guard returns the RESOLVED path (here, the protected
    # DB's own canonical realpath) rather than the raw "trading_data.db" spelling — see the
    # dedicated protected-DB test above for why.
    assert sw.assert_db_path_allowed("trading_data.db", True) == os.path.realpath(
        str(sw.PROTECTED_REAL_DB))


def test_relative_fixture_db_path_is_still_allowed_from_a_non_repo_root_cwd(tmp_path, monkeypatch):
    """An honest relative path — protected under NEITHER interpretation — must not be caught by
    the widened guard (no honest path may start refusing as a side effect of this fix)."""
    import os
    sw = _sweep()
    monkeypatch.chdir(tmp_path)
    rel = os.path.relpath(_REPO / "fixtures" / "trading_fixture.db", tmp_path)
    # TASK_902 item 10 (TOCTOU fix): the guard returns the RESOLVED (realpath) string, computed
    # from the SAME cwd this check ran under, not the raw relative spelling — see the fixture/
    # protected-DB tests above.
    assert sw.assert_db_path_allowed(rel, False) == os.path.realpath(rel)


@pytest.mark.parametrize("raw,needle", [
    (None, "required and has no default"),
    ("", "required and has no default"),
    ("   ", "required and has no default"),
    (",", "contains no tier values"),
    (" , , ", "contains no tier values"),
    ("abc", "is not a number"),
    ("8,abc", "is not a number"),
    # D57 fix pass item 2: garbage numeric tiers used to slip through and land VERBATIM in the
    # artifact's operator-facing metadata disclosure line — now refused with the same clean
    # SystemExit pattern as the other three refusal paths.
    ("-8", "is negative"),
    ("8,-1", "is negative"),
    ("nan", "is not finite"),
    ("inf", "is not finite"),
    ("-inf", "is not finite"),
    ("1e400", "is not finite"),        # float("1e400") == inf
])
def test_fee_tier_parse_refuses_cleanly(raw, needle):
    """Each of these used to be (or would have been) a raw ValueError traceback out of the
    parse expression; every one is now a SystemExit carrying an actionable message."""
    sw = _sweep()
    with pytest.raises(SystemExit) as e:
        sw.parse_fee_tiers(raw)
    assert needle in str(e.value)


def test_fee_tier_parse_accepts_a_real_list():
    sw = _sweep()
    assert sw.parse_fee_tiers("8,16") == [8.0, 16.0]
    assert sw.parse_fee_tiers(" 8 , 16 ,") == [8.0, 16.0]


def test_fee_tier_parse_allows_zero():
    """D57 fix pass item 2: 0bp is economically meaningful (a rebate-neutral venue) and is NOT
    treated as a parsing failure — only negative and non-finite entries are refused."""
    sw = _sweep()
    assert _sweep().parse_fee_tiers("0") == [0.0]
    assert sw.parse_fee_tiers("0,8,16") == [0.0, 8.0, 16.0]


@pytest.mark.parametrize("raw", ["2024-11-01", "", "a,b,c", "2024-11-01,"])
def test_window_parse_refuses_cleanly(raw):
    sw = _sweep()
    with pytest.raises(SystemExit) as e:
        sw._parse_window("--window-a", raw)
    assert "must be exactly 'START,END'" in str(e.value)


def test_default_windows_produce_the_original_label_strings_byte_identical():
    """The fix must not perturb the pre-registered default-window launch: with the default
    --window-a/--window-b dates, the DERIVED labels must equal WINDOW_A[0]/WINDOW_B[0]
    byte-for-byte (the srv launch's artifact labels stay exactly what they are today)."""
    sw = _sweep()
    windows = sw.build_windows("2024-11-01", "2025-04-01", "2025-04-01", "2026-03-01")
    assert windows == (sw.WINDOW_A, sw.WINDOW_B)
    assert windows[0][0] == "a_2024-11_2025-04" == sw.WINDOW_A[0]
    assert windows[1][0] == "b_2025-04_2026-03" == sw.WINDOW_B[0]


def test_overridden_windows_get_a_label_derived_from_their_own_dates():
    """Regression for the defect: an overridden --window-a/--window-b used to keep the
    hardcoded 'a_2024-11_2025-04'/'b_2025-04_2026-03' labels regardless of the actual dates
    supplied, so the artifact's own metadata line could assert a label that contradicted the
    dates recorded right beside it. The label must now track the dates it is tagging."""
    sw = _sweep()
    windows = sw.build_windows("2024-01-04", "2024-02-01", "2025-06-01", "2025-09-01")
    assert windows[0] == ("a_2024-01_2024-02", "2024-01-04", "2024-02-01")
    assert windows[1] == ("b_2025-06_2025-09", "2025-06-01", "2025-09-01")
    # the previously-hardcoded, now-wrong labels must NOT appear
    assert windows[0][0] != sw.WINDOW_A[0]
    assert windows[1][0] != sw.WINDOW_B[0]


# ---------------------------------------------------------------- artifact honesty (MAJORs)

def _artifact_text(tmp_path, **meta_overrides):
    sw = _sweep()
    meta = {
        "attainable_maker_fee_bp_tiers": [8.0, 16.0], "db_path": "fixtures/trading_fixture.db",
        "bar_sec": 60, "n_assets": 3, "n_universe": 3, "symbols": ["A", "B", "C"],
        "alias_union_applied": True,
        "alias_groups": {"A": [1], "B": [2], "C": [3]},
        "alias_material_slots": list(sw.alias_layer.MATERIAL_ALIAS_SYMBOLS),
        "alias_gate_disclosure": [],
        "alias_per_slot_id_row_counts": {"A": {1: 10}, "B": {2: 10}, "C": {3: 10}},
        "alias_material_slot_report": [],
        "alias_rank_changes": [],
        "alias_top64_entered": [],
        "alias_top64_left": [],
        "alias_unknown_symbols": [],
        "train_start": "2023-05-16", "val_start": "2024-06-01",
        "windows": [{"name": "a", "start": "x", "end": "y"}],
        "grid": {"H": [5]}, "grid_epoch": {"start": 0, "end": 1},
        "grid_start_iso": "1970-01-01T00:00:00Z", "grid_end_iso": "1970-01-01T00:00:01Z",
        "seed": 897, "n_boot": 10, "n_cells_expected": 0, "n_cells_recorded": 0,
        "n_cells_ok": 0, "n_cells_dropped_or_failed": 0, "runtime_s": 0.0,
        "estimator": "per-decision", "maker_fill_model": "OHLC-touch",
    }
    meta.update(meta_overrides)
    out = tmp_path / "result.md"
    sw.write_artifact([], meta, out)
    return out.read_text()


def test_artifact_claims_the_877_alias_fold_is_applied_via_read_time_layer(tmp_path):
    """D56(a): the fold IS applied now, but as a READ-TIME layer owned by this runner, not by
    editing the 809 lineage. The artifact must say so, name the material slots, and still assert
    the lineage itself carries zero alias handling (re-verified by
    `test_the_lineage_really_has_no_alias_handling` below, never taken on faith here)."""
    text = _artifact_text(tmp_path)
    assert "877 alias fold: APPLIED, as a READ-TIME LAYER" in text
    assert "the 806/808/809 lineage itself is UNCHANGED" in text
    assert "still contain zero references to" in text
    for sym in ("/USDC/USD", "/USDC/USDT", "/USDCGBP", "/USDTAUD", "/USDTCHF", "/USDTEUR"):
        assert sym in text


def test_artifact_names_universe_rank_changes(tmp_path):
    """D56's explicit disclosure requirement: any slot whose universe rank CHANGES under the
    union read must be named."""
    text = _artifact_text(
        tmp_path,
        alias_rank_changes=[{"symbol": "X/USD", "exact_rank": 5, "union_rank": 2}],
        alias_top64_entered=["X/USD"], alias_top64_left=["Y/USD"],
    )
    assert "X/USD" in text
    assert "Y/USD" in text
    assert "1 candidate symbol(s) changed rank position" in text


def test_the_lineage_really_has_no_alias_handling():
    """The claim above is re-verified against the SOURCE on every run, so the artifact's wording
    can never drift back into truthful-sounding-but-false if the lineage changes underneath."""
    names = ("alias", "asset_aliases", "normalize_symbol_key", "resolve_requested_symbols",
             "group_asset_rows")
    for rel in ("runs/806_predictability_probe/probe.py",
                "runs/808_linear_backtest/backtest.py",
                "runs/809_decision_rule_rescue/decide.py"):
        p = _REPO / rel
        if not p.exists():
            pytest.skip(f"{rel} not present")
        src = p.read_text()
        for n in names:
            assert n not in src, (
                f"{rel} now references {n!r} — the artifact states the alias union is applied "
                "as a READ-TIME layer (`alias_read_layer.py`) with the 809 lineage left "
                "unmodified. That statement is no longer accurate if the lineage itself now "
                "does alias handling; re-verify the wording before changing this test."
            )


def test_artifact_states_the_ci_is_uncalibrated_prominently(tmp_path):
    """MAJOR: never ship an uncalibrated interval silently underneath a band that depends on it.
    The disclosure must appear ABOVE the cell table and the CI column must be labelled."""
    text = _artifact_text(tmp_path)
    assert "UNCALIBRATED" in text
    assert "CI lo > 0" in text
    assert "D49" in text
    assert "CI(UNCALIBRATED)" in text                       # the column header itself
    banner_at = text.index("UNCALIBRATED")
    table_at = text.index("## Cells")
    assert banner_at < table_at, "the disclosure must precede the cell table"


def test_artifact_states_the_certification_is_conditional_on_measured_dependence(tmp_path):
    """MAJOR 1(a)/(c), fix cycle 3. The artifact is where the calibration's legitimacy is read,
    so it must (1) say plainly that every k was certified at dep == block ONLY, (2) carry the
    SHIPPED-k-past-its-block coverage NUMBERS, (3) never call dep == block the worst case, and
    (4) state that the safe fallback beats a certified interval whose assumption is violated."""
    import calibration as cal
    text = _artifact_text(tmp_path)
    assert "CONDITIONAL ON MEASURED DEPENDENCE <= BLOCK" in text
    assert "dep == block` ONLY" in text
    assert "the SHIPPED k, stressed past its certified dependence".lower() in text.lower()
    # the measured collapse numbers are IN the artifact, not merely referenced
    for r in cal.shipped_k_mismatch_rows():
        for c in r["coverage_by_seed"]:
            assert f"{c:.3f}" in text, (r["n_dec"], r["dep_multiple"], c)
    assert "0.530" in text                      # tied to the number that raised D55
    # the refusal mechanism and its operating characteristic (TASK_897 D57: "certification" ->
    # "advisory interval", since even a pass is never band-readable on this leg)
    assert "REFUSED an advisory interval on dependence grounds" in text
    assert "measured_dependence_len <= block_used" in text
    assert "dep<=block" in text                 # the per-cell column header
    # (4) the safe-fallback statement
    assert "strictly better" in text.lower()
    # and the retracted framing survives ONLY as a quotation inside its own retraction
    assert text.count("the matched-dependence worst case") == 1
    assert "wrongly called it 'the matched-dependence worst case'" in text
    assert "THAT IS NOT THE WORST CASE" in text
    assert text.count("the worst case D49/892 characterize") == 1
    assert (text.index("THAT IS NOT THE WORST CASE")
            < text.index("the worst case D49/892 characterize"))


def test_artifact_states_the_tau_guard_is_out_of_family_and_names_TASK_901(tmp_path):
    """TASK_897 D57 (2026-08-22). The artifact must (1) state the third conjunct is UNAVAILABLE,
    (2) state plainly the guard is EXPECTED NOT TO BITE on this row's real data, (3) carry the
    real-data containment number (0.639) and the out-of-family AR1 numbers (0.805/0.376), and
    (4) name TASK_901 as the row that upgrades this conjunct. Mutation-tested: deleting the
    'EXPECTED NOT TO BITE' clause from the banner (the real drop site,
    `write_artifact`'s dependence box) turns this RED — demonstrated by hand during
    implementation, not asserted here (that would just be this same test)."""
    text = _artifact_text(tmp_path)
    assert "UNAVAILABLE-because-uncalibratable ON THIS LEG" in text
    assert "EXPECTED NOT TO BITE" in text
    assert "0.639" in text
    assert "0.805" in text and "0.376" in text
    assert "TASK_901" in text
    assert "LINEAR-ACF" in text or "linear-ACF" in text
    # the never-band-readable claim must survive even for an advisory=True cell
    assert "NEVER" in text and "band-readable" in text.lower()


def test_artifact_never_presents_an_advisory_interval_as_band_readable(tmp_path):
    """The retracted AFFIRMATIVE claim ('THIS is the interval the band's ... may be read off')
    must not survive anywhere in the RENDERED artifact (checked on the joined runtime string, not
    module source — adjacent string literals only concatenate at import/render time), and its
    narrowed, NEGATED replacement must be present instead."""
    text = _artifact_text(tmp_path)
    assert "THIS is the interval the band's 'CI lo > 0' criterion may be read off" not in text
    assert ("NEVER the interval the band's 'CI lo > 0' criterion may be read off" in text
            or "NEVER** WHERE THE BAND'S 'CI lo > 0' CRITERION MAY BE READ" in text)
    assert "estimate and carries `ci_calibrated=True`" not in text
    assert "carries `ci_advisory=True`" in text


def test_artifact_demotion_narrative_is_jitter_not_a_hole(tmp_path):
    """MINOR 2. The demotion stands, but the artifact must not tell a reader block 31 has a real
    coverage hole — and it must report the mirror finding (achieved strata re-measured) too."""
    import calibration as cal
    text = _artifact_text(tmp_path)
    assert "How to read the `achieved=False` rows" in text
    assert "THE DEMOTIONS ARE KEPT ANYWAY" in text
    assert "not a coverage defect" in text
    assert "MONTE-CARLO JITTER" in text
    assert "weaker than it reads" in text.lower()
    # both re-measurement groups are tabulated, with the recorded value beside them
    for nd in cal.MEASURED_TABLE_SEED_JITTER["demoted"]:
        assert f"| {nd} | demoted |" in text
    for nd in cal.MEASURED_TABLE_SEED_JITTER["achieved_sample"]:
        assert f"| {nd} | achieved_sample |" in text
    # the strata the re-measurement does NOT rescue are still named, at the RE-PINNED figures
    # (D57 fix pass item 2 — the old 0.9194/0.9300 pair came from an unrecoverable run)
    assert "520 (0.9206)" in text
    assert "660 (0.9367) and 740 (0.9344) come back barely above it" in text
    assert "0.9194" not in text and "0.9456" not in text


def test_artifact_counts_and_names_dependence_refused_cells(tmp_path):
    """The per-cell disclosure, driven through `write_artifact` with REAL CellRecords: a cell
    refused on dependence must say so in its own row, and be counted in the banner."""
    sw = _sweep()
    import numpy as np

    def _cell(n_dec, dep, block, calibrated):
        return sw.CellRecord(
            window="a", H=5, k=16, variant="long_short", tau=0.9, N=0, fee_bp=8.0, adv_bp=5.0,
            status="ok", reason="", n_bars=9000, n_decisions=n_dec, decisions_per_year=1000.0,
            mean_hold_bars=3.0, turnover_mean=0.03, fill_rate=0.5, sharpe_decision=1.2,
            ci_lo_UNCALIBRATED=0.4, ci_hi_UNCALIBRATED=2.0, ci_advisory=calibrated,
            ci_calibration_k=1.4 if calibrated else float("nan"),
            ci_calibration_source_n_dec=3000 if calibrated else -1,
            ci_lo_ADVISORY=0.1 if calibrated else float("nan"),
            ci_hi_ADVISORY=2.3 if calibrated else float("nan"),
            block_used=block, measured_dependence_len=dep,
            pnl=0.01, maxdd=-0.02, sharpe_per_bar_diagnostic=5.0)

    cells = [_cell(3000, 12.0, 60, True),                  # dependence inside the block
             _cell(3000, 480.0, 60, False),                # dependence 8x the block -> refused
             _cell(3000, float("nan"), 60, False)]         # unmeasurable -> refused
    out = tmp_path / "r.md"
    meta = {
        "attainable_maker_fee_bp_tiers": [8.0], "db_path": "fixtures/trading_fixture.db",
        "bar_sec": 60, "n_assets": 1, "n_universe": 1, "symbols": ["A"],
        "alias_union_applied": True, "alias_groups": {"A": [1]},
        "alias_material_slots": list(sw.alias_layer.MATERIAL_ALIAS_SYMBOLS),
        "alias_gate_disclosure": [], "alias_per_slot_id_row_counts": {"A": {1: 10}},
        "alias_material_slot_report": [], "alias_rank_changes": [], "alias_top64_entered": [],
        "alias_top64_left": [], "alias_unknown_symbols": [],
        "train_start": "2023-05-16", "val_start": "2024-06-01",
        "windows": [{"name": "a", "start": "x", "end": "y"}], "grid": {"H": [5]},
        "grid_epoch": {"start": 0, "end": 1}, "grid_start_iso": "1970-01-01T00:00:00Z",
        "grid_end_iso": "1970-01-01T00:00:01Z", "seed": 897, "n_boot": 10,
        "n_cells_expected": 3, "n_cells_recorded": 3, "n_cells_ok": 3,
        "n_cells_dropped_or_failed": 0, "runtime_s": 0.0, "estimator": "per-decision",
        "maker_fill_model": "OHLC-touch",
    }
    sw.write_artifact(cells, meta, out)
    text = out.read_text()
    assert ("**This artifact: 2/3 ok cells THE GUARD ACTUALLY RAN ON were REFUSED an advisory "
            "interval on dependence grounds** (1 of them") in text
    assert "REFUSED ON DEPENDENCE**: measured dependence 480 > block 60" in text
    assert ("REFUSED ON DEPENDENCE**: this cell's realized per-decision dependence could NOT be "
            "measured") in text
    assert "| UNMEASURABLE | False |" in text
    assert "measured dependence 12 <= block 60" in text
    assert not np.isnan(cells[0].measured_dependence_len)
    # all three cells carried decisions, so none is in the "guard never ran" bucket
    assert "The guard ran on 3 of the 3 ok cells; the other 0 carried fewer than" in text


def test_a_no_decision_cell_is_NOT_counted_as_a_dependence_refusal(tmp_path):
    """D57 fix cycle 4, item B2 — THE DEFECT: `_dependence_ok` returned False for a cell with
    `block_used=-1` / `n_decisions=0`, so every such cell was counted in the banner's
    `n_dep_refused` while its own row (correctly) said "no CI computed". On the fixture that
    printed "144/144 ok cells were REFUSED an advisory interval on dependence grounds (144 of
    them because the dependence could not be measured at all)" over a table with ZERO
    `REFUSED ON DEPENDENCE` rows — the reader told the safety guard bit everywhere, which reads
    as evidence of health, when nothing had been measured at all.

    PERTURBATION THIS IS RED AGAINST: revert `_dependence_status` to the old
    `blk is None -> not ok` collapse and the banner reads "2/2 ... REFUSED" here while the table
    shows one refusal — the disagreement this test exists to forbid.

    The invariant asserted is the strong one: for EVERY category, the banner's count equals the
    number of table rows in that category, counted from the rendered text."""
    sw = _sweep()
    import numpy as np

    def _cell(n_dec, dep, block):
        return sw.CellRecord(
            window="a", H=5, k=16, variant="long_short", tau=0.9, N=0, fee_bp=8.0, adv_bp=5.0,
            status="ok", reason="", n_bars=9000, n_decisions=n_dec, decisions_per_year=1000.0,
            mean_hold_bars=3.0, turnover_mean=0.03, fill_rate=0.5,
            sharpe_decision=1.2 if n_dec else float("nan"),
            ci_lo_UNCALIBRATED=0.4 if n_dec else float("nan"),
            ci_hi_UNCALIBRATED=2.0 if n_dec else float("nan"), ci_advisory=False,
            ci_calibration_k=float("nan"), ci_calibration_source_n_dec=-1,
            ci_lo_ADVISORY=float("nan"), ci_hi_ADVISORY=float("nan"),
            block_used=block, measured_dependence_len=dep,
            pnl=0.01, maxdd=-0.02, sharpe_per_bar_diagnostic=5.0)

    # one real refusal (dependence past the block) + one cell that never had an interval at all
    cells = [_cell(3000, 480.0, 60), _cell(0, float("nan"), -1)]
    for c in cells:
        assert sw._dependence_ok(c) is False        # both still fail CLOSED
    assert sw._dependence_status(cells[0]) == sw.DEP_REFUSED
    assert sw._dependence_status(cells[1]) == sw.DEP_NO_CI

    out = tmp_path / "r.md"
    sw.write_artifact(cells, _artifact_meta(n_cells=2), out)
    text = out.read_text()
    # the banner counts ONE refusal out of ONE measured cell, and names the other separately
    assert ("**This artifact: 1/1 ok cells THE GUARD ACTUALLY RAN ON were REFUSED an advisory "
            "interval on dependence grounds** (0 of them") in text
    assert "The guard ran on 1 of the 2 ok cells; the other 1 carried fewer than 2 decisions" \
        in text
    # ...and the banner agrees with its own table, row for row
    rows = [ln for ln in text.splitlines() if ln.startswith("| a | 5 | 16 |")]
    assert len(rows) == 2
    assert sum("REFUSED ON DEPENDENCE" in r for r in rows) == 1
    assert sum("no CI computed" in r for r in rows) == 1
    # the no-interval cell must NOT claim its dependence was measured-and-rejected
    no_ci_row = [r for r in rows if "no CI computed" in r][0]
    assert "REFUSED ON DEPENDENCE" not in no_ci_row
    assert np.isnan(cells[1].measured_dependence_len)


def test_a_run_with_no_decisions_says_so_prominently(tmp_path):
    """D57 fix cycle 4, item B3 (second half). A run in which not one `ok` cell carried a
    decision used to render as `recorded: N (ok=N, dropped/failed=0)` with no gate violation
    anywhere — the documented 863 green-looking-artifact shape. The artifact must SAY it holds no
    measurement. It stays a STATEMENT, never a verdict and never an error (this instrument records;
    the planner rules), and exit codes are untouched — on the fixture, zero decisions is normal.

    PERTURBATION: give either cell n_decisions>0 and the line must disappear (asserted below)."""
    sw = _sweep()

    def _cell(n_dec):
        return sw.CellRecord(
            window="a", H=5, k=16, variant="long_short", tau=0.9, N=0, fee_bp=8.0, adv_bp=5.0,
            status="ok", reason="", n_bars=9000, n_decisions=n_dec, decisions_per_year=float("nan"),
            mean_hold_bars=float("nan"), turnover_mean=float("nan"), fill_rate=float("nan"),
            sharpe_decision=float("nan"), ci_lo_UNCALIBRATED=float("nan"),
            ci_hi_UNCALIBRATED=float("nan"), ci_advisory=False, ci_calibration_k=float("nan"),
            ci_calibration_source_n_dec=-1, ci_lo_ADVISORY=float("nan"),
            ci_hi_ADVISORY=float("nan"), block_used=-1, measured_dependence_len=float("nan"),
            pnl=float("nan"), maxdd=float("nan"), sharpe_per_bar_diagnostic=float("nan"))

    empty = tmp_path / "empty.md"
    sw.write_artifact([_cell(0), _cell(0)], _artifact_meta(n_cells=2), empty)
    t = empty.read_text()
    assert "NO DECISIONS IN THIS RUN" in t and "THIS ARTIFACT CONTAINS NO MEASUREMENT" in t
    assert "All 2 `ok` cells recorded ZERO decisions" in t
    # ...and the dependence banner must NOT render "0/0 ok cells were REFUSED", which is true but
    # reads like "no refusals, all good" on exactly the run where nothing was measured
    assert "**This artifact: THE DEPENDENCE GUARD RAN ON NO CELLS.**" in t
    assert "an ABSENCE OF MEASUREMENT, not a clean bill of health" in t
    # the counting form ("N/M ok cells THE GUARD ACTUALLY RAN ON were REFUSED") must be absent —
    # its 0/0 rendering is the misread this branch exists to prevent
    assert "ok cells THE GUARD ACTUALLY RAN ON were REFUSED" not in t
    assert "measured ok cells PASSING the guard here" not in t
    assert "with no cell measured there is no pass rate here" in t
    # a STATEMENT, not a verdict: no pass/fail language, and the no-verdict header still stands
    assert "**No verdict in this file.**" in t
    for verdict_word in ("CELL CLEARS", "ALL FAIL"):
        # the only mentions of these are in the standing "that is the planner's job" sentences,
        # never attached to this run
        assert f"{verdict_word}:" not in t
    # ...and the line is absent as soon as ONE cell carried a decision
    mixed = tmp_path / "mixed.md"
    sw.write_artifact([_cell(0), _cell(3000)], _artifact_meta(n_cells=2), mixed)
    assert "NO DECISIONS IN THIS RUN" not in mixed.read_text()


# ------------------------------------------------- apply_ci_calibration (MAJOR 2, D55(a) fix)
#
# This is the function that decides EVERY real cell's ci_lo_ADVISORY / ci_hi_ADVISORY /
# ci_advisory (renamed from ci_lo/ci_hi/ci_calibrated by TASK_897 D57, 2026-08-22 — the pair is
# never band-readable on this leg regardless). Before this block it had ZERO dedicated tests: its
# guards looked right on inspection and nothing pinned them. Every test below drives the REAL
# function, never a re-implementation.

def _tbl(*entries):
    """Build a calibration table in the shipped shape: {n_dec: {block, k, achieved}}."""
    import calibration as cal
    return {int(nd): {"block": int(cal.block_for(nd)), "k": float(k), "achieved": bool(ach)}
            for nd, k, ach in entries}


def _apply(sw, sh, lo_u, hi_u, n_dec, table, dep=1.0, block=None):
    """Call the REAL `apply_ci_calibration` with a dependence measurement supplied.

    `dep=1.0` is an i.i.d.-like cell — the only kind that can certify at every block — so these
    tests exercise the stratum/superset guards rather than the dependence guard, which has its
    own tests below. The dependence argument is NEVER defaulted here on purpose: the default is
    None and None REFUSES (fix cycle 3, MAJOR 1(b)), which is pinned by
    `test_apply_ci_calibration_refuses_when_no_dependence_measurement_is_supplied`."""
    return sw.apply_ci_calibration(sh, lo_u, hi_u, n_dec, table,
                                   measured_dependence_len=dep, block_used=block)


def test_apply_ci_calibration_calibrates_a_cell_with_an_achieved_stratum():
    sw = _sweep()
    import calibration as cal
    n_dec = 3000                                   # block 60 under the shipped cap
    table = _tbl((3000, 1.5, True))
    assert cal.block_for(n_dec) == 60
    cal_ok, k, src, lo, hi = _apply(sw, 2.0, 1.0, 3.0, n_dec, table)
    assert cal_ok is True and k == 1.5 and src == 3000
    # widened around the POINT ESTIMATE (2.0), never around 0:
    assert lo == pytest.approx(2.0 - 1.5 * 1.0)    # 0.5
    assert hi == pytest.approx(2.0 + 1.5 * 1.0)    # 3.5


def test_apply_ci_calibration_leaves_the_qualifier_when_no_stratum_is_certified():
    sw = _sweep()
    import numpy as np
    table = _tbl((3000, 1.5, False))               # certified NOTHING
    cal_ok, k, src, lo, hi = _apply(sw, 2.0, 1.0, 3.0, 3000, table)
    assert cal_ok is False and src == -1
    assert np.isnan(k) and np.isnan(lo) and np.isnan(hi)
    # ...and an empty table is the same refusal, not a crash.
    assert _apply(sw, 2.0, 1.0, 3.0, 3000, {})[0] is False


@pytest.mark.parametrize("n_dec", [0, 1, float("nan"), -5, None])
def test_apply_ci_calibration_refuses_degenerate_decision_counts(n_dec):
    """0 decisions (the degenerate-smoke case), 1 decision (below MIN_DECISIONS_FOR_SHARPE),
    NaN, a negative count, and a missing count all refuse — never a calibrated band."""
    sw = _sweep()
    import numpy as np
    table = _tbl((40, 1.4, True), (3000, 1.5, True))
    cal_ok, k, src, lo, hi = _apply(sw, 2.0, 1.0, 3.0, n_dec, table)
    assert cal_ok is False and src == -1
    assert np.isnan(k) and np.isnan(lo) and np.isnan(hi)


@pytest.mark.parametrize("sh,lo_u,hi_u", [
    (float("nan"), 1.0, 3.0),
    (2.0, float("nan"), 3.0),
    (2.0, 1.0, float("nan")),
    (float("inf"), 1.0, 3.0),
    (2.0, -float("inf"), 3.0),
])
def test_apply_ci_calibration_refuses_non_finite_sharpe_or_raw_ci(sh, lo_u, hi_u):
    sw = _sweep()
    import numpy as np
    table = _tbl((3000, 1.5, True))
    cal_ok, k, src, lo, hi = _apply(sw, sh, lo_u, hi_u, 3000, table)
    assert cal_ok is False and src == -1
    assert np.isnan(k) and np.isnan(lo) and np.isnan(hi)


def test_apply_ci_calibration_between_strata_never_crosses_a_BLOCK_boundary():
    """THE MAJOR-1 CASE, pinned at the consumer. `block_for` jumps to the cap at n_dec=1200, so
    a cell at n_dec=1500 runs at block=60 while the n_dec=1000 stratum was measured at block=50.
    A k measured at block 50 must NEVER be handed to a block-60 cell."""
    sw = _sweep()
    import calibration as cal
    assert cal.block_for(1000) == 50 and cal.block_for(1500) == 60   # the step function
    only_block50 = _tbl((1000, 1.34, True))
    cal_ok, k, src, lo, hi = _apply(sw, 2.0, 1.0, 3.0, 1500, only_block50)
    assert cal_ok is False, "a block-50 stratum must not certify a block-60 cell"
    # add the block-60 ladder (a stratum AT the transition point, plus one above the query so the
    # certified region reaches it) and the same cell now certifies from the block-60 stratum
    with_block60 = dict(only_block50)
    with_block60.update(_tbl((1200, 1.47, True), (3000, 1.5, True)))
    cal_ok, k, src, lo, hi = _apply(sw, 2.0, 1.0, 3.0, 1500, with_block60)
    assert (cal_ok, k, src) == (True, 1.47, 1200)
    # ...but a query ABOVE the largest certified stratum in the saturated block is refused, not
    # extrapolated: there is no measurement up there.
    assert _apply(sw, 2.0, 1.0, 3.0, 50000, with_block60)[0] is False
    # and within regime A the borrow stays inside one block: n_dec=419 -> block 20 -> stratum 400
    reg_a = _tbl((400, 1.41, True), (300, 1.39, True))
    assert cal.block_for(419) == 20 and cal.block_for(300) == 15
    assert _apply(sw, 2.0, 1.0, 3.0, 419, reg_a)[:3] == (True, 1.41, 400)


def test_apply_ci_calibration_refuses_an_inverted_or_narrowing_widening():
    """MINOR 5. Widening must never make the band EASIER to clear. If the raw interval failed to
    bracket its own point estimate the arithmetic inverts and k>1 would RAISE the lower bound; a
    k<1 would narrow a well-formed interval. Both are refused, not published."""
    sw = _sweep()
    import numpy as np
    table = _tbl((3000, 1.5, True))
    # sharpe BELOW its own raw ci_lo -> naive arithmetic gives lo = 1.0 - 1.5*(1.0-2.0) = 2.5,
    # i.e. a RAISED lower bound off a raw lower bound of 2.0. Must refuse.
    cal_ok, k, src, lo, hi = _apply(sw, 1.0, 2.0, 3.0, 3000, table)
    assert cal_ok is False and src == -1 and np.isnan(lo) and np.isnan(hi)
    # sharpe ABOVE its own raw ci_hi -> mirror case
    assert _apply(sw, 4.0, 1.0, 3.0, 3000, table)[0] is False
    # a k < 1 narrows a perfectly well-formed interval -> also refused
    assert _apply(sw, 2.0, 1.0, 3.0, 3000, _tbl((3000, 0.5, True)))[0] is False
    # the exact-touch boundary (sharpe == ci_lo) is degenerate but not inverted: it widens only
    # upward and the superset invariant still holds, so it is allowed through.
    ok, _, _, lo2, hi2 = _apply(sw, 1.0, 1.0, 3.0, 3000, table)
    assert ok is True and lo2 == pytest.approx(1.0) and hi2 > 3.0


def test_apply_ci_calibration_refuses_a_ZERO_WIDTH_raw_interval():
    """MINOR 4 (fix cycle 3). The superset guard admits EQUALITY, so a degenerate raw interval
    `[+5,+5]` around a point estimate of +5 satisfied `lo <= lo_u and hi >= hi_u` trivially and
    certified — handing the pre-registered band a `CI lo = +5 > 0` off an interval carrying no
    information at all. Same class as the original BLOCKER (n_dec=2, CI [+22.93,+22.93]).
    Not demonstrated reachable through `compute_honest_metrics`, but closed anyway."""
    sw = _sweep()
    import numpy as np
    import calibration as cal
    # the exact reported reproducer, against the SHIPPED table
    ok, k, src, lo, hi = sw.apply_ci_calibration(5.0, 5.0, 5.0, 1000,
                                                 cal.MEASURED_CALIBRATION_TABLE,
                                                 measured_dependence_len=1.0)
    assert ok is False and src == -1
    assert np.isnan(k) and np.isnan(lo) and np.isnan(hi)
    # any zero-width or inverted raw interval refuses, at any offset from the point estimate
    for sh, lo_u, hi_u in ((5.0, 5.0, 5.0), (0.0, 0.0, 0.0), (-3.0, -3.0, -3.0),
                           (2.0, 2.5, 2.5), (2.0, 3.0, 1.0)):
        assert _apply(sw, sh, lo_u, hi_u, 3000, _tbl((3000, 1.5, True)))[0] is False, (sh, lo_u)
    # the guard is real: widen the raw interval by the smallest amount and it certifies again
    ok2 = _apply(sw, 5.0, 5.0 - 1e-9, 5.0 + 1e-9, 3000, _tbl((3000, 1.5, True)))[0]
    assert ok2 is True


# ------------------------------- the DEPENDENCE refusal (fix cycle 3, MAJOR 1(b))

def test_apply_ci_calibration_refuses_when_no_dependence_measurement_is_supplied():
    """FAIL-CLOSED BY DEFAULT. Every shipped `k` was certified at dep == block ONLY. A caller who
    does not supply the cell's measured dependence has not established that condition, so the
    default (`measured_dependence_len=None`) must REFUSE — the omission can never be a silent
    pass. This is the guard's real drop site: delete the None check and this goes red."""
    sw = _sweep()
    import numpy as np
    import calibration as cal
    ok, k, src, lo, hi = sw.apply_ci_calibration(2.0, 1.0, 3.0, 3000,
                                                 cal.MEASURED_CALIBRATION_TABLE)
    assert ok is False and src == -1
    assert np.isnan(k) and np.isnan(lo) and np.isnan(hi)


@pytest.mark.parametrize("dep", [float("nan"), float("inf"), -float("inf")])
def test_apply_ci_calibration_refuses_unmeasurable_dependence(dep):
    """`honest_estimator.dependence_length` returns NaN when the dependence cannot be measured
    (series too short, degenerate, or an ACF that never returns to the white-noise band). Unknown
    dependence must refuse, never certify."""
    sw = _sweep()
    import numpy as np
    ok, k, src, lo, hi = _apply(sw, 2.0, 1.0, 3.0, 3000, _tbl((3000, 1.5, True)), dep=dep)
    assert ok is False and src == -1
    assert np.isnan(k) and np.isnan(lo) and np.isnan(hi)


def test_apply_ci_calibration_refuses_dependence_longer_than_the_block():
    """The substantive half of MAJOR 1(b): at dep > block the SHIPPED k's measured coverage runs
    0.82-0.91 -> 0.52 (`calibration.MEASURED_SHIPPED_K_AT_DEPENDENCE_MISMATCH`), so a cell whose
    OWN per-decision dependence exceeds its block must not be certified off that k."""
    sw = _sweep()
    import numpy as np
    import calibration as cal
    table = _tbl((3000, 1.5, True))
    assert cal.block_for(3000) == 60
    for dep in (61, 120, 480, 10000):
        ok, k, src, lo, hi = _apply(sw, 2.0, 1.0, 3.0, 3000, table, dep=dep)
        assert ok is False and src == -1, dep
        assert np.isnan(k) and np.isnan(lo) and np.isnan(hi)
    # at or below the block it certifies — the guard is a threshold, not a blanket refusal
    for dep in (1, 30, 59, 60):
        assert _apply(sw, 2.0, 1.0, 3.0, 3000, table, dep=dep)[0] is True, dep


def test_dependence_refusal_uses_the_ESTIMATORS_OWN_block_not_a_re_derived_one():
    """`block_used` comes off `HonestMetrics`, i.e. the block the CI was ACTUALLY bootstrapped
    at. Passing it explicitly must be what the comparison uses, so the guard cannot drift from
    the estimator if the block rule ever changes underneath."""
    sw = _sweep()
    table = _tbl((3000, 1.5, True))
    # dep=40 is fine against the real block 60 ...
    assert _apply(sw, 2.0, 1.0, 3.0, 3000, table, dep=40)[0] is True
    # ... but refused if the estimator reports it actually bootstrapped at block 20
    assert _apply(sw, 2.0, 1.0, 3.0, 3000, table, dep=40, block=20)[0] is False


def test_dependence_refusal_end_to_end_through_compute_honest_metrics():
    """Drive the REAL producer, not a hand-built argument: build one bar series whose
    per-decision returns are i.i.d. and one whose per-decision returns carry dependence far
    longer than the block, run both through `compute_honest_metrics`, and check that only the
    first can certify."""
    sw = _sweep()
    import numpy as np
    import calibration as cal
    import honest_estimator as hest
    rng = np.random.default_rng(897)
    n_dec, hold = 3000, 3
    for label, per_dec in (("iid", cal.simulate_iid(n_dec, rng)),
                           ("dependent", cal.simulate_overlapping(n_dec, 60 * 8, rng))):
        gross = np.repeat(per_dec / hold, hold) * 1e-3
        turn = np.zeros(gross.size)
        turn[::hold] = 0.1
        m = hest.compute_honest_metrics(gross, turn, 0.0, 60, seed=0, n_boot=200)
        assert m.n_decisions == n_dec
        assert m.block_used == cal.block_for(n_dec) == 60
        ok = sw.apply_ci_calibration(m.sharpe_decision, m.ci_lo_UNCALIBRATED,
                                     m.ci_hi_UNCALIBRATED, m.n_decisions,
                                     cal.MEASURED_CALIBRATION_TABLE,
                                     measured_dependence_len=m.measured_dependence_len,
                                     block_used=m.block_used)[0]
        if label == "iid":
            assert m.measured_dependence_len <= 60 and ok is True
        else:
            assert not (m.measured_dependence_len <= 60), m.measured_dependence_len
            assert ok is False, "a cell with 8x-block dependence must not certify"


def test_apply_ci_calibration_output_is_always_a_superset_of_the_raw_interval():
    """Property sweep over random well-formed cells: whenever the function DOES certify, the
    calibrated band is strictly no easier to clear than the raw one."""
    sw = _sweep()
    import numpy as np
    rng = np.random.default_rng(897)
    table = _tbl((40, 1.4, True), (400, 1.41, True), (1200, 1.47, True), (3000, 1.5, True))
    n_certified = 0
    for _ in range(3000):
        sh = float(rng.uniform(-5, 5))
        lo_u = sh - abs(float(rng.uniform(1e-6, 4)))
        hi_u = sh + abs(float(rng.uniform(1e-6, 4)))
        n_dec = int(rng.integers(2, 9000))
        ok, k, src, lo, hi = _apply(sw, sh, lo_u, hi_u, n_dec, table)
        if not ok:
            assert np.isnan(lo) and np.isnan(hi) and src == -1
            continue
        n_certified += 1
        assert lo <= lo_u + 1e-12 and hi >= hi_u - 1e-12, (sh, lo_u, hi_u, n_dec, lo, hi)
    assert n_certified > 100, "the sweep must actually exercise the certified path"


def test_no_cell_can_read_ci_lo_while_ci_advisory_is_false():
    """The structural claim the artifact rests on: `ci_lo_ADVISORY`/`ci_hi_ADVISORY` are NaN on
    EVERY uncertified cell, so a band read can never silently fall back to the raw bounds
    (renamed from ci_lo/ci_hi/ci_calibrated by TASK_897 D57, 2026-08-22 — the pair is never
    band-readable on this leg even when finite). Checked on the two producers of a CellRecord:
    `_empty_cells` (the dropped/failed path) and `apply_ci_calibration` (the ok path) — across a
    table that certifies only part of the n_dec range."""
    sw = _sweep()
    import numpy as np
    table = _tbl((1200, 1.47, True), (3000, 1.5, True))     # nothing certified below n_dec=1200
    for n_dec in (0, 1, 2, 39, 40, 100, 419, 1199, 1200, 1500, 3000, 8000, 10 ** 7):
        for dep in (1.0, 10 ** 6, float("nan"), None):
            ok, k, src, lo, hi = sw.apply_ci_calibration(2.0, 1.0, 3.0, n_dec, table,
                                                          measured_dependence_len=dep)
            if ok:
                assert np.isfinite(lo) and np.isfinite(hi) and np.isfinite(k) and src > 0
            else:
                assert np.isnan(lo) and np.isnan(hi) and np.isnan(k) and src == -1
    for c in sw._empty_cells("w", 1, [16], ["long_short"], [0.9], [0], [0.0], [0.0],
                             "dropped", "reason"):
        assert c.ci_advisory is False
        assert np.isnan(c.ci_lo_ADVISORY) and np.isnan(c.ci_hi_ADVISORY) \
            and np.isnan(c.ci_calibration_k)
        assert c.ci_calibration_source_n_dec == -1
        # a dropped/failed cell has no dependence measurement either — and that also refuses
        assert np.isnan(c.measured_dependence_len) and c.block_used == -1
        assert sw._dependence_ok(c) is False


def test_apply_ci_calibration_against_the_SHIPPED_table_at_real_cell_sizes():
    """Drives the real `calibration.MEASURED_CALIBRATION_TABLE` (not a stub) at decision counts
    the non-degenerate smoke actually produces, and pins that the certified/uncertified split
    matches `certified_region_description`'s stated region."""
    sw = _sweep()
    import calibration as cal
    import numpy as np
    for n_dec in (1000, 1500, 2000, 4320):
        ok, k, src, lo, hi = _apply(sw, 1.2, 0.4, 2.0, n_dec, cal.MEASURED_CALIBRATION_TABLE)
        assert ok is True, f"n_dec={n_dec} should be inside the certified region"
        assert cal.MEASURED_CALIBRATION_TABLE[src]["block"] == cal.block_for(n_dec)
        assert lo <= 0.4 and hi >= 2.0
        # ...and the SAME cell, with dependence past its block, must NOT certify
        assert _apply(sw, 1.2, 0.4, 2.0, n_dec, cal.MEASURED_CALIBRATION_TABLE,
                      dep=cal.block_for(n_dec) + 1)[0] is False
    # below the smallest stratum and above the largest one, the qualifier stands
    for n_dec in (2, 39, 10 ** 9):
        ok, k, src, lo, hi = _apply(sw, 1.2, 0.4, 2.0, n_dec, cal.MEASURED_CALIBRATION_TABLE)
        assert ok is False and np.isnan(lo) and np.isnan(hi)
    # ...and so does a cell landing in one of the four strata whose validation batch came in
    # under target (they were recorded as measured, never re-rolled until they passed)
    unachieved_blocks = {e["block"] for e in cal.MEASURED_CALIBRATION_TABLE.values()
                         if not e["achieved"]}
    assert unachieved_blocks, "this pin is only meaningful while some stratum is uncertified"
    for n_dec in (nd for nd, e in cal.MEASURED_CALIBRATION_TABLE.items() if not e["achieved"]):
        ok, k, src, lo, hi = _apply(sw, 1.2, 0.4, 2.0, n_dec, cal.MEASURED_CALIBRATION_TABLE)
        assert ok is False, f"n_dec={n_dec} sits in an UNCERTIFIED stratum"
        assert np.isnan(lo) and np.isnan(hi)


# ------------------------------------------- TASK_897 D57(D): block_used hardening -----------
#
# `DEPENDENCE_REFUSAL_MECHANISM` asserts "(3) The comparison is against `block_used` reported by
# the estimator itself, not a re-derived block, so the two cannot drift" — nothing enforced that
# before this row; `apply_ci_calibration` trusted a caller-supplied `block_used` outright. Same
# defect shape as the three retracted strings: a claim the code did not back. Fixed here as
# cheap, fail-closed defence-in-depth: REFUSE rather than silently pass when `block_used !=
# calibration.block_for(n_decisions)`.

def test_apply_ci_calibration_refuses_a_block_used_mismatch():
    """The real drop site: delete/comment the `expected_block` check in `apply_ci_calibration`
    (restoring `block = cal.block_for(...) if block_used is None else int(block_used)` with no
    guard) and this goes RED — demonstrated by hand during implementation (temporarily disabling
    the check, re-running this test, observing failure, then restoring the check; recorded in the
    TASK_897 D57 implementation report, not re-asserted here as that would just be this test)."""
    sw = _sweep()
    import calibration as cal
    table = _tbl((3000, 1.5, True))
    assert cal.block_for(3000) == 60
    # a block_used that DISAGREES with block_for(n_decisions) must refuse outright, regardless of
    # whether the dependence measurement would otherwise have passed
    assert _apply(sw, 2.0, 1.0, 3.0, 3000, table, dep=1.0, block=61)[0] is False
    assert _apply(sw, 2.0, 1.0, 3.0, 3000, table, dep=1.0, block=59)[0] is False
    assert _apply(sw, 2.0, 1.0, 3.0, 3000, table, dep=1.0, block=0)[0] is False
    # a MATCHED block_used (or none supplied, which derives it) still certifies normally
    assert _apply(sw, 2.0, 1.0, 3.0, 3000, table, dep=1.0, block=60)[0] is True
    assert _apply(sw, 2.0, 1.0, 3.0, 3000, table, dep=1.0, block=None)[0] is True


def test_apply_ci_calibration_refuses_nonfinite_block_used_instead_of_raising():
    """D57 fix pass item 1: the docstring promises 'anything else refuses', but
    `int(block_used)` on a non-finite `block_used` (nan/inf) used to raise an uncaught
    `ValueError: cannot convert float NaN to integer` instead of returning the documented
    refusal tuple. Not reachable from today's call site (`HonestMetrics.block_used` is always an
    int) — this is defence-in-depth, so it must change no numerics for the matched-pair path
    (covered by `test_block_used_from_the_real_estimator_always_matches_block_for` below)."""
    sw = _sweep()
    import calibration as cal
    table = _tbl((3000, 1.5, True))
    assert cal.block_for(3000) == 60
    for bad in (float("nan"), float("inf"), float("-inf")):
        ok, k, src, lo, hi = _apply(sw, 2.0, 1.0, 3.0, 3000, table, dep=1.0, block=bad)
        assert ok is False
        assert src == -1
        import math
        assert math.isnan(k) and math.isnan(lo) and math.isnan(hi)
    # a matched, finite block_used still certifies (no numerics changed by this guard)
    assert _apply(sw, 2.0, 1.0, 3.0, 3000, table, dep=1.0, block=60)[0] is True


def test_block_used_from_the_real_estimator_always_matches_block_for():
    """Defence-in-depth proof: the ONE caller today (`run_sweep`, via `HonestMetrics.block_used`)
    always hands `apply_ci_calibration` a MATCHED pair, so the D57(D) hardening changes no
    numerics for the shipped path — `compute_honest_metrics` selects
    `max(min_block, min(max_block, n_dec // 20))` with `min_block=2, max_block=MAX_BLOCK_DEFAULT`,
    the SAME formula `calibration.block_for` mirrors. Checked directly through the real
    estimator, not just by formula inspection, across both regime A and regime B."""
    sw = _sweep()
    import numpy as np
    import calibration as cal
    import honest_estimator as hest
    for n_dec, hold in ((59, 3), (400, 2), (1200, 2), (1500, 2), (8000, 1)):
        gross = np.zeros(n_dec * hold)
        turn = np.zeros(n_dec * hold)
        turn[::hold] = 0.1
        m = hest.compute_honest_metrics(gross, turn, 0.0, 60, seed=0, n_boot=50)
        assert m.n_decisions == n_dec
        assert m.block_used == cal.block_for(n_dec), (n_dec, m.block_used, cal.block_for(n_dec))
        # and that matched pair, fed straight through, is never refused BY THE HARDENING alone
        ok, *_ = sw.apply_ci_calibration(1.0, 0.5, 1.5, n_dec,
                                         {n_dec: {"block": cal.block_for(n_dec), "k": 1.5,
                                                  "achieved": True}},
                                         measured_dependence_len=1.0, block_used=m.block_used)
        assert ok is True


# ------------------------------------------------------------------------------------ D59 fix
#
# `probe.report_bar_frequency` is this harness's FIRST data touch on every run, and its body
# (`runs/806_predictability_probe/probe.py:162`, byte-untouched — the 809-lineage doctrine)
# opens `sqlite3.connect(cfg.db_path)` with no `mode=ro`: READ-WRITE-CREATE. This falsified the
# module's own docstring sentence ("this harness never writes to it") and, on the real 24GB DB,
# a read-write open of a file carrying a hot journal performs rollback RECOVERY and writes —
# something `mode=ro` cannot do. `_ReadOnlySqlite3ForProbe` scope-patches `probe.sqlite3` for
# the duration of that one call, routed through the existing `alias_read_layer._connect_ro`.

def test_a_nonexistent_db_path_refuses_instead_of_creating_a_file(tmp_path):
    """(i) D59. A nonexistent `--db-path`, run through the harness's real entrypoint
    (`run_sweep`, which reaches `probe.report_bar_frequency` first), must REFUSE — not have a
    zero-byte file CREATED at it, which is what the pre-fix bare `sqlite3.connect` did.

    PERTURBATION THIS IS RED AGAINST: with the fix reverted (`_ReadOnlySqlite3ForProbe` removed
    / the scope-patch deleted from `_run_sweep_inner`), this call SUCCEEDS past
    `report_bar_frequency` (an empty/nonexistent sqlite file opens READ-WRITE-CREATE with no
    tables, `sqlite_master` query on it raises further downstream, but the FILE is already
    created by then) — see the mutation evidence recorded in the task report."""
    import sqlite3
    sw = _sweep()
    decoy = tmp_path / "does_not_exist_yet.db"
    assert not decoy.exists()
    with pytest.raises(sqlite3.OperationalError) as e:
        sw.run_sweep(attainable_maker_fee_bp=[8.0], db_path=str(decoy))
    assert "unable to open database file" in str(e.value)
    assert not decoy.exists(), "a mode=ro connect must never CREATE the target file"


def test_write_through_the_harness_connection_path_fails(tmp_path):
    """(ii) D59. A write/DDL attempt through the exact connection form the harness now uses
    (`_ReadOnlySqlite3ForProbe.connect`, i.e. `probe.sqlite3.connect` for the duration of the
    scope-patch) fails with sqlite's own read-only refusal.

    PERTURBATION THIS IS RED AGAINST: reverting to a bare `sqlite3.connect(db)` (no `mode=ro`)
    lets `CREATE TABLE` commit successfully on the same file — see the mutation evidence in the
    task report."""
    import sqlite3
    sw = _sweep()
    decoy = tmp_path / "writable_decoy.db"
    sqlite3.connect(str(decoy)).close()   # an honest, pre-existing, empty sqlite file

    con = sw._ReadOnlySqlite3ForProbe.connect(str(decoy))
    with pytest.raises(sqlite3.OperationalError) as e:
        con.execute("CREATE TABLE pwned (a int)")
    assert "attempt to write a readonly database" in str(e.value)


def test_bar_sec_is_unchanged_vs_the_read_write_form_on_the_fixture_db():
    """(iii) D59. The CONNECTION FORM must not move any number: `report_bar_frequency`'s
    result on the fixture DB is byte-identical (same int) whether read through the read-only
    scope-patch or the original bare read-write connect — the fix changes how the file is
    opened, never what is read from it."""
    sw = _sweep()
    fx = _REPO / "fixtures" / "trading_fixture.db"
    if not fx.exists():
        pytest.skip("fixture not present")
    import probe
    cfg = probe.Config(db_path=str(fx))

    # read-write form (the pre-fix body, called directly — never through the patched name)
    import sqlite3
    con = sqlite3.connect(str(fx))
    cur = con.cursor()
    cur.execute("SELECT asset_id, COUNT(*) c FROM ohlcv GROUP BY asset_id ORDER BY c DESC LIMIT 1")
    aid = cur.fetchone()[0]
    cur.execute("SELECT timestamp FROM ohlcv WHERE asset_id=? ORDER BY timestamp LIMIT 5000", (aid,))
    ts = [r[0] for r in cur.fetchall()]
    con.close()
    import numpy as np
    deltas = np.diff(np.asarray(ts, dtype=np.int64))
    deltas = deltas[deltas > 0]
    bar_sec_rw = int(np.bincount(deltas).argmax()) if len(deltas) else 60

    # read-only form, through the harness's own scope-patch mechanism
    _orig = probe.sqlite3
    probe.sqlite3 = sw._ReadOnlySqlite3ForProbe()
    try:
        bar_sec_ro = probe.report_bar_frequency(cfg)
    finally:
        probe.sqlite3 = _orig

    assert bar_sec_ro == bar_sec_rw
    assert isinstance(bar_sec_ro, int) and isinstance(bar_sec_rw, int)


# ------------------------------------------------------------------------------ D59 — path shapes
#
# A NON-REGULAR `--db-path` HANGS, AND THAT IS A DOCUMENTED LIMITATION, NOT A GUARDED CASE.
# A `mode=ro` URI open on a FIFO blocks forever where the pre-fix bare connect failed fast
# ('disk I/O error'); a FIFO `<db>-journal` sidecar beside a valid DB hangs under BOTH forms, so
# that half is a property of opening sqlite at all. An earlier cycle added an `S_ISREG` type guard
# plus subprocess-timeout tests for this; both were REMOVED on operator call. `--db-path` is an
# operator-supplied path to a real database on a scheduled run — a named pipe is not a plausible
# input, and each cycle chasing a more exotic spelling (FIFO -> symlink-to-FIFO -> `//localhost`
# -prefixed FIFO) was TASK_891's failure mode returning: chasing total coverage of an unbounded
# input space. Complexity inside a fail-closed guard is itself risk.
#
# WHAT REMAINS PINNED BELOW IS THE PART THAT PROTECTS THE 24GB SOURCE OF RECORD: `normalize_db_path`
# runs ONCE in `_connect_ro` and its single result feeds both the percent-quote and the open, so no
# belt can ever reason about a different file than sqlite opens. That divergence was real — sqlite
# strips a leading `//` authority when it is empty or exactly `localhost`, and un-normalized,
# `//localhost/…/trading_data.db` walked straight past `assert_db_path_allowed`'s protected-DB gate
# (demonstrated against a decoy). These tests pin the normalization, NOT a file-type guard.


def test_a_symlink_to_a_regular_db_still_opens(tmp_path):
    """(3) The other side of following symlinks: a symlink whose TARGET is a real sqlite file is
    an ordinary, honest path and must keep working."""
    import sqlite3
    import alias_read_layer as al
    real = tmp_path / "real.db"
    sqlite3.connect(str(real)).execute("CREATE TABLE t (a int)")
    link = tmp_path / "link.db"
    link.symlink_to(real)
    assert al._connect_ro(str(link)).execute("select 1").fetchone() == (1,)


def test_a_directory_db_path_fails_fast_and_loudly(tmp_path):
    """A directory `--db-path` does NOT hang — sqlite refuses it fast and loudly with its own
    `disk I/O error`. Pinned so the documented limitation stays accurate about which shapes wedge
    (FIFOs) and which simply fail (directories)."""
    import sqlite3
    import alias_read_layer as al
    d = tmp_path / "a_directory.db"
    d.mkdir()
    with pytest.raises(sqlite3.OperationalError) as e:
        al._connect_ro(str(d))
    assert "disk I/O error" in str(e.value)


def test_nonexistent_and_dangling_paths_refuse_without_creating_anything(tmp_path):
    """NO-REGRESSION. A nonexistent path and a dangling symlink refuse fast with sqlite's own
    `OperationalError: unable to open database file`, and — the part that matters for the D59
    invariant — neither path is CREATED. That "never creates" property is the whole point of the
    read-only handle: the pre-fix bare connect created a zero-byte file at whatever it was given.
    `test_a_nonexistent_db_path_refuses_instead_of_creating_a_file` pins the same thing through
    the real `run_sweep` entrypoint; this pins it directly on `_connect_ro`."""
    import sqlite3
    import alias_read_layer as al
    missing = tmp_path / "not_here.db"
    with pytest.raises(sqlite3.OperationalError) as e:
        al._connect_ro(str(missing))
    assert "unable to open database file" in str(e.value)
    assert not missing.exists()

    dangling = tmp_path / "dangling.db"
    dangling.symlink_to(tmp_path / "nothing_at_all.db")
    with pytest.raises(sqlite3.OperationalError) as e:
        al._connect_ro(str(dangling))
    assert "unable to open database file" in str(e.value)
    assert not (tmp_path / "nothing_at_all.db").exists()


# -------------------------------------------------------- D59 — the URI-authority normalization
#
# sqlite DISCARDS a leading `//<authority>` when the authority is empty or exactly `localhost`,
# so an un-normalized caller string and the path sqlite actually opens can be DIFFERENT FILES.
# That is not cosmetic: it defeated the protected-DB gate. Measured against a decoy standing in
# for `trading_data.db`:
#
#     assert_db_path_allowed('<decoy>',            scheduled=False) -> REFUSED  (correct)
#     assert_db_path_allowed('//localhost<decoy>', scheduled=False) -> PASSED   (gate defeated)
#
# because `realpath('//localhost/…')` is not the protected realpath and `samefile` raises an
# `OSError` that was swallowed. `normalize_db_path` now runs ONCE in `_connect_ro` and its single
# result feeds both the percent-quote and the open, and `assert_db_path_allowed` refuses a leading
# `//` outright. Two belts, neither relying on the other.


def test_an_invalid_uri_authority_still_fails_loudly_and_fast(tmp_path):
    """Only an EMPTY authority and the byte-exact lowercase `localhost` are stripped, because
    those are the only two sqlite's parser accepts. `//LOCALHOST/…` is NOT stripped: sqlite
    refuses it loudly (`invalid uri authority`) and never OPENS it, so the un-normalized spelling
    cannot reach a file by that route either."""
    import sqlite3
    import alias_read_layer as al
    missing = tmp_path / "nope.db"
    with pytest.raises(sqlite3.OperationalError) as e:
        al._connect_ro("//LOCALHOST" + str(missing))
    assert "invalid uri authority" in str(e.value)
    assert al.normalize_db_path("//LOCALHOST/x") == "//LOCALHOST/x"


def test_the_normalized_path_is_the_file_sqlite_opens(tmp_path):
    """THE INVARIANT ITSELF, not a spelling-by-spelling patch. Whatever `_connect_ro` is handed,
    the path `normalize_db_path` produces must be the file sqlite actually OPENS. Divergence
    between those two is the entire bug class — it is what let `//localhost/…/trading_data.db`
    walk past the protected-DB gate while sqlite opened the real thing.

    The comparison is EMPIRICAL, not a re-derivation of the fix: sqlite is asked which file it
    opened (`PRAGMA database_list` reports the main database's resolved filename), so this cannot
    agree with a wrong normalization merely by sharing its code.

    WHAT THIS DOES AND DOES NOT PIN — measured, not assumed. It pins that `normalize_db_path` is a
    correct MIRROR of `sqlite3ParseUri`: break the mirror (strip `//LOCALHOST`, or stop stripping
    `//localhost`) and this goes red. It does NOT pin that `_connect_ro` actually CALLS it —
    mutation-tested on gpu-server: dropping `normalize_db_path` from `_connect_ro` and quoting
    `str(db_path)` left this GREEN, because sqlite performs the identical authority-stripping
    itself, so both sides still name the same file. An earlier revision of this docstring claimed
    that perturbation as its red case; that claim was FALSE and is retracted here.

    The consumer that genuinely needs the normalization is the PROTECTED-DB GATE, which
    `realpath`/`samefile`s the string — see
    `test_a_uri_authority_prefix_cannot_walk_a_protected_db_past_the_gate`."""
    import sqlite3
    import alias_read_layer as al
    real = tmp_path / "real.db"
    sqlite3.connect(str(real)).execute("CREATE TABLE t (a int)")
    weird = tmp_path / "we ird?x=1%2F#frag.db"        # every URI metacharacter at once
    sqlite3.connect(str(weird)).execute("CREATE TABLE t (a int)")

    for spelling in [str(real), "//localhost" + str(real), "//" + str(real),
                     str(weird), "//localhost" + str(weird)]:
        normalized = al.normalize_db_path(spelling)
        con = al._connect_ro(spelling)
        opened = con.execute("PRAGMA database_list").fetchone()[2]
        con.close()
        assert os.path.realpath(normalized) == os.path.realpath(opened), (
            f"{spelling!r}: normalized to {normalized!r} but sqlite opened {opened!r}"
        )


@pytest.mark.parametrize("raw,expected", [
    ("/a/b.db", "/a/b.db"),                       # untouched
    ("rel/b.db", "rel/b.db"),                     # untouched
    ("//localhost/a/b.db", "/a/b.db"),            # the wedge spelling
    ("///a/b.db", "/a/b.db"),                     # empty authority
    ("//LOCALHOST/a/b.db", "//LOCALHOST/a/b.db"),  # sqlite refuses it; do not pretend otherwise
    ("//host/a/b.db", "//host/a/b.db"),           # ditto
    ("/a//localhost/b.db", "/a//localhost/b.db"),  # authority parsing is LEADING-only
])
def test_normalize_db_path_mirrors_sqlites_authority_rules(raw, expected):
    """(9) D59 fix cycle 3. The normalization is a mirror of `sqlite3ParseUri`, not a
    Python-flavoured path cleanup: strip ONLY an empty or byte-exact-lowercase-`localhost`
    leading authority. Getting this wrong in either direction reopens the divergence — stripping
    too much makes the guard stat a file sqlite will not open, stripping too little makes sqlite
    open a file the guard did not stat."""
    import alias_read_layer as al
    assert al.normalize_db_path(raw) == expected


# ============================================================================================
# TASK_897 D60 — the inherited-IC guard re-based to the pins' own claim (queue row 1).
#
# The srv leg's FIRST launch recorded an honest NULL: the guard asserted 806's June-era pins
# against the harness's SWEEP panel, whose test slice runs a year past the pins' basis. D60's
# ruling re-bases the guard to a SEPARATE pin-era panel (`probe.Config`'s own split defaults —
# the pins' actual basis), asserted exactly as before (tol UNCHANGED), while the sweep-panel IC
# becomes a disclosed, non-asserting MEASUREMENT. These tests pin that wiring WITHOUT paying for
# a real DB or real ridge regression: every DB-touching call in the pipeline is stubbed with a
# lightweight synthetic-panel generator that reproduces the REAL grid-construction shape (so
# `probe.make_splits`/`honest_estimator.compute_honest_metrics` still run for real on
# well-formed arrays), and the PIN vs SWEEP panels are built with DIFFERENT asset counts so a
# test can tell, from the `close` array a stubbed `decide.fit_xs_signal_chunked`/
# `inherited_signal_check_chunked` call receives, WHICH panel fed it.
# ============================================================================================

def _stub_897_pipeline(monkeypatch, sw, *, pin_universe_n=1, sweep_universe_n=2,
                        pin_ic=0.111111, sweep_ic=0.222222, pin_ic_by_h=None,
                        check_proxy=False):
    """Stubs every DB-touching call the pin-era guard and the sweep panel build reach
    (`probe.report_bar_frequency`/`load_universe_trainlocked`/`build_aligned_ohlcv` for the
    EXACT-SPELLING pin-era read; `alias_layer.rank_universe_with_alias_union`/
    `build_aligned_ohlcv_with_alias_union`/`alias_material_slot_report` for the sweep panel) with
    synthetic-but-real-shaped generators, and stubs `decide.fit_xs_signal_chunked`/
    `inherited_signal_check_chunked`/`_chunked_predict`/`_run_decision` so no real regression
    runs. The PIN panel always has `pin_universe_n` assets; the SWEEP panel always has
    `sweep_universe_n` assets (default 2, deliberately different from the pin's default 1) — a
    stub tells the two apart by `close.shape[1]`, never by guessing which call site it is, so a
    test can catch the guard reading the WRONG panel.

    `pin_ic` (or `pin_ic_by_h: Dict[int, float|Exception]`, overriding it per-H) is what the
    stubbed pin-era `inherited_signal_check_chunked` call returns; the stubbed sweep-panel call
    always returns `sweep_ic`. If `check_proxy` is True, `load_universe_trainlocked`/
    `build_aligned_ohlcv` additionally record, per call, whether `probe.sqlite3` was the
    read-only proxy at call time (`calls['proxy']`).

    Returns `calls`: `{"fit_check": [...], "proxy": [...]}`. Each `fit_check` entry is
    `{"fn": "fit"|"check", "H": int, "which": "pin"|"sweep", "cfg": <decide.CFG at call time>,
    "pins_snapshot": dict(backtest.SIGNAL_806_IC_FULL) at call time}`."""
    import numpy as np
    import probe
    import decide
    import backtest

    calls = {"fit_check": [], "proxy": []}

    def _panel(cfg, n_assets, seed):
        grid_start = probe._to_epoch(cfg.train_start) - cfg.W * 60
        grid_end = probe._to_epoch(cfg.test_end)
        grid_ts = np.arange(grid_start, grid_end, 60, dtype=np.int64)
        T = len(grid_ts)
        rng = np.random.default_rng(seed)
        close = 200.0 + rng.normal(scale=0.001, size=(T, n_assets)).cumsum(axis=0)
        ohlcv = np.zeros((T, n_assets, 5), dtype=np.float64)
        ohlcv[:, :, 0] = close
        ohlcv[:, :, 1] = close + 0.05
        ohlcv[:, :, 2] = close - 0.05
        ohlcv[:, :, 3] = close
        ohlcv[:, :, 4] = 1.0
        return ohlcv, grid_ts

    def fake_report_bar_frequency(cfg):
        return 60

    def fake_load_universe_trainlocked(cfg):
        if check_proxy:
            calls["proxy"].append(("load_universe_trainlocked",
                                    isinstance(probe.sqlite3, sw._ReadOnlySqlite3ForProbe)))
        aids = list(range(pin_universe_n))
        return [f"P{i}" for i in aids], aids, [(f"P{i}", i, 100) for i in aids]

    def fake_build_aligned_ohlcv(cfg, asset_ids, bar_sec):
        if check_proxy:
            calls["proxy"].append(("build_aligned_ohlcv",
                                    isinstance(probe.sqlite3, sw._ReadOnlySqlite3ForProbe)))
        return _panel(cfg, len(asset_ids), seed=1)

    def fake_rank_universe_with_alias_union(cfg):
        aids = list(range(sweep_universe_n))
        return {"symbols": [f"S{i}" for i in aids], "asset_ids": aids,
                "alias_groups": [[i] for i in aids], "rank_changes": [],
                "top64_entered": [], "top64_left": [], "unknown": []}

    def fake_build_aligned_ohlcv_with_alias_union(cfg, asset_ids, alias_groups, bar_sec):
        ohlcv, grid_ts = _panel(cfg, len(asset_ids), seed=2)
        return ohlcv, grid_ts, [], [10 for _ in asset_ids]

    def fake_alias_material_slot_report(cfg, windows):
        return []

    def fake_fit(close, grid_ts, H):
        which = "pin" if close.shape[1] == pin_universe_n else "sweep"
        calls["fit_check"].append({"fn": "fit", "H": H, "which": which, "cfg": decide.CFG,
                                    "pins_snapshot": dict(backtest.SIGNAL_806_IC_FULL)})
        A = close.shape[1]
        return np.zeros((1, A)), np.zeros(A), np.ones(A), 1.0

    def fake_check(close, grid_ts, H, w, mu, sd):
        which = "pin" if close.shape[1] == pin_universe_n else "sweep"
        calls["fit_check"].append({"fn": "check", "H": H, "which": which, "cfg": decide.CFG,
                                    "pins_snapshot": dict(backtest.SIGNAL_806_IC_FULL)})
        if which == "sweep":
            return sweep_ic
        result = (pin_ic_by_h.get(H, pin_ic) if pin_ic_by_h is not None else pin_ic)
        if isinstance(result, BaseException):
            raise result
        return result

    def fake_chunked_predict(close, rows, w, mu, sd, W, chunk=20000):
        return np.zeros((len(rows), close.shape[1]))

    def fake_run_decision(close, high, low, pred, present, dec_rows, k, variant, tau, N,
                           exec_model):
        # Return contract mirrored from the 809 lineage (`decide._run_decision`, docstring +
        # `fill_frac = (filled / intended) ...`): (gross_bar ARRAY, turnover_bar ARRAY,
        # fill_frac SCALAR). The scalar matters — `CellRecord.fill_rate` is rendered `{:.3f}`
        # by `write_artifact`, so a stubbed ARRAY here makes any artifact-rendering test die
        # on `TypeError: unsupported format string passed to numpy.ndarray.__format__`
        # (D60 fix cycle 2: found while adding the H2 artifact-wording test).
        n = len(dec_rows)
        return np.zeros(n), np.zeros(n), 1.0

    monkeypatch.setattr(probe, "report_bar_frequency", fake_report_bar_frequency)
    monkeypatch.setattr(probe, "load_universe_trainlocked", fake_load_universe_trainlocked)
    monkeypatch.setattr(probe, "build_aligned_ohlcv", fake_build_aligned_ohlcv)
    monkeypatch.setattr(sw.alias_layer, "rank_universe_with_alias_union",
                         fake_rank_universe_with_alias_union)
    monkeypatch.setattr(sw.alias_layer, "build_aligned_ohlcv_with_alias_union",
                         fake_build_aligned_ohlcv_with_alias_union)
    monkeypatch.setattr(sw.alias_layer, "alias_material_slot_report",
                         fake_alias_material_slot_report)
    monkeypatch.setattr(decide, "fit_xs_signal_chunked", fake_fit)
    monkeypatch.setattr(decide, "inherited_signal_check_chunked", fake_check)
    monkeypatch.setattr(decide, "_chunked_predict", fake_chunked_predict)
    monkeypatch.setattr(decide, "_run_decision", fake_run_decision)
    return calls


_D60_SWEEP_WINDOWS = (("t", "2024-01-04", "2024-01-06"),)


def test_pin_era_cfg_is_806s_own_basis_regardless_of_operator_overrides(monkeypatch):
    """D60: `cfg_pin` must be `probe.Config`'s OWN split defaults — train_start=2023-05-16,
    val_start=2024-06-01, test_start=2024-11-01, test_end=2025-04-01, the pins' actual basis —
    even when the CALLER overrides --train-start/--val-start/--window-a/--window-b to something
    completely different (here: 2024-01-01..2024-01-06, nowhere near the pins' basis).

    PERTURBATION THIS IS RED AGAINST: build `cfg_pin` from the sweep `cfg` (e.g. `probe.Config(
    db_path=db_path, train_start=train_start, val_start=val_start, test_start=..., test_end=...)`
    — deriving it from the caller's own overrides, exactly the drift D60 forbids) instead of
    `probe.Config(db_path=db_path)`, and the captured pin-era `cfg.test_end` below reads
    "2024-01-06" (the override) instead of "2025-04-01" — this assertion catches it immediately.
    Mutation-verified by hand (see task report): reverted `cfg_pin` to
    `probe.Config(db_path=db_path, train_start=train_start, val_start=val_start,
    test_start=windows[0][1], test_end=windows[-1][2])` and this test failed on the `test_end`
    assertion; reverted back, test passes again."""
    sw = _sweep()
    calls = _stub_897_pipeline(monkeypatch, sw)
    sw.run_sweep(
        attainable_maker_fee_bp=[8.0], db_path="fixtures/trading_fixture.db",
        train_start="2024-01-01", val_start="2024-01-03", windows=_D60_SWEEP_WINDOWS,
        grid_h=(5,), grid_k=(16,), grid_variant=("long_only",), grid_tau=(0.75,), grid_n=(0,),
        grid_fee_bp=(8.0,), grid_adv_bp=(0.0,), n_boot=5,
    )
    pin_calls = [c for c in calls["fit_check"] if c["which"] == "pin"]
    sweep_calls = [c for c in calls["fit_check"] if c["which"] == "sweep"]
    assert pin_calls and sweep_calls
    for c in pin_calls:
        cfg = c["cfg"]
        assert cfg.train_start == "2023-05-16"
        assert cfg.val_start == "2024-06-01"
        assert cfg.test_start == "2024-11-01"
        assert cfg.test_end == "2025-04-01"
    for c in sweep_calls:
        cfg = c["cfg"]
        assert cfg.train_start == "2024-01-01"
        assert cfg.test_end == "2024-01-06"


def test_pin_era_guard_reads_the_pin_era_panel_not_the_sweep_panel(monkeypatch):
    """THE LOAD-BEARING MUTATION GATE (queue row 1): the pin-era guard's recorded IC must come
    from the PIN-ERA panel (exact-spelling `probe.load_universe_trainlocked`/`build_aligned_
    ohlcv`), and the disclosed sweep-panel measurement must come from the SWEEP panel (the
    D56(a) alias-union read) — never crossed. The two panels are built with different asset
    counts here (`pin_universe_n=1` vs `sweep_universe_n=2`) purely so a stub can tell which one
    it received; `pin_ic`/`sweep_ic` are set to distinct sentinels so a crossed wire is
    unmistakable in the artifact's own metadata.

    PERTURBATION THIS IS RED AGAINST, MUTATION-VERIFIED BY HAND (see task report): reverted
    `_run_sweep_inner` to the PRE-D60 shape — deleted the separate pin-era panel build entirely
    and pointed the (still-asserting) guard at the SWEEP panel instead
    (`decide.inherited_signal_check_chunked(close, grid_ts, H, w, mu, sd)` on the sweep `close`/
    `grid_ts`, inside the existing per-H `try` block, exactly as D59 shipped it). Under that
    revert `meta['inherited_ic_pin_era']` does not exist and the sentinel below is unreachable —
    RED. Restored the D60 shape and this test passes again."""
    sw = _sweep()
    calls = _stub_897_pipeline(monkeypatch, sw, pin_ic=0.111111, sweep_ic=0.222222)
    cells, meta = sw.run_sweep(
        attainable_maker_fee_bp=[8.0], db_path="fixtures/trading_fixture.db",
        train_start="2024-01-01", val_start="2024-01-03", windows=_D60_SWEEP_WINDOWS,
        grid_h=(5,), grid_k=(16,), grid_variant=("long_only",), grid_tau=(0.75,), grid_n=(0,),
        grid_fee_bp=(8.0,), grid_adv_bp=(0.0,), n_boot=5,
    )
    assert meta["inherited_ic_pin_era"][5]["status"] == "pass"
    assert meta["inherited_ic_pin_era"][5]["ic"] == pytest.approx(0.111111)
    assert meta["signal_ic_sweep_panel"][5] == pytest.approx(0.222222)
    # ...and every cell that reached the decision engine did so under the sweep panel's shape
    # (sweep_universe_n=2), never the pin panel's (pin_universe_n=1) — the guard's panel never
    # leaks into the cells themselves either.
    assert any(c.status == "ok" for c in cells)


def test_pin_era_read_goes_through_the_read_only_proxy(monkeypatch):
    """The pin-era read reaches `probe.load_universe_trainlocked`/`build_aligned_ohlcv`, and both
    are, in real life, bare `sqlite3.connect(cfg.db_path)` calls (READ-WRITE-CREATE — the exact
    D59 hazard already closed on `report_bar_frequency`). This asserts the SAME
    `_ReadOnlySqlite3ForProbe` scope-patch is installed for the duration of BOTH pin-era calls,
    and that `probe.sqlite3` is back to the real module immediately after.

    PERTURBATION THIS IS RED AGAINST: delete the `probe.sqlite3 = _ReadOnlySqlite3ForProbe()` /
    restore around the pin-era panel build (keeping everything else). Mutation-verified by hand
    (see task report): with the rebind removed, `isinstance(probe.sqlite3,
    sw._ReadOnlySqlite3ForProbe)` reads False for both stubbed calls — RED. Restored, both read
    True again."""
    import sqlite3 as real_sqlite3
    sw = _sweep()
    calls = _stub_897_pipeline(monkeypatch, sw, check_proxy=True)
    sw.run_sweep(
        attainable_maker_fee_bp=[8.0], db_path="fixtures/trading_fixture.db",
        train_start="2024-01-01", val_start="2024-01-03", windows=_D60_SWEEP_WINDOWS,
        grid_h=(5,), grid_k=(16,), grid_variant=("long_only",), grid_tau=(0.75,), grid_n=(0,),
        grid_fee_bp=(8.0,), grid_adv_bp=(0.0,), n_boot=5,
    )
    assert calls["proxy"], "the pin-era panel build never ran through the stubbed loaders"
    for fn_name, is_proxy in calls["proxy"]:
        assert is_proxy, f"{fn_name} ran WITHOUT the read-only proxy installed"
    # ...and it is restored: nothing left `probe.sqlite3` patched after the run.
    import probe
    assert probe.sqlite3 is real_sqlite3


def test_pin_era_guard_failure_or_nan_drops_exactly_that_hs_cells_and_keeps_counts_consistent(
        monkeypatch):
    """Gate (ii): a failing OR non-finite pin-era guard at one H must drop every cell of THAT H
    only (reason names the D60 pin-era basis), never crash the run, and `n_cells_recorded` must
    still equal `n_cells_expected`. H=1 gets a NaN IC (the D60 rider on TASK_902's vacuous-on-NaN
    item — `nan > tol` is False, so an unguarded compare would let it slip through as a pass);
    H=5 gets a normal passing IC and its cells proceed to the (stubbed) decision engine.

    PERTURBATION THIS IS RED AGAINST: delete the `if not np.isfinite(ic_pin): raise
    AssertionError(...)` guard in `_run_sweep_inner`'s pin-era loop. Mutation-verified by hand
    (see task report): with it removed, H=1's `inherited_ic_pin_era` status reads "pass" with
    `ic=nan` instead of "failed", and H=1's cells are NOT dropped (status "ok") — RED against
    both assertions below. Restored, this test passes again."""
    sw = _sweep()
    calls = _stub_897_pipeline(monkeypatch, sw, pin_ic_by_h={1: float("nan"), 5: 0.1511})
    cells, meta = sw.run_sweep(
        attainable_maker_fee_bp=[8.0], db_path="fixtures/trading_fixture.db",
        train_start="2024-01-01", val_start="2024-01-03", windows=_D60_SWEEP_WINDOWS,
        grid_h=(1, 5), grid_k=(16,), grid_variant=("long_only",), grid_tau=(0.75,), grid_n=(0,),
        grid_fee_bp=(8.0,), grid_adv_bp=(0.0,), n_boot=5,
    )
    assert meta["inherited_ic_pin_era"][1]["status"] == "failed"
    assert "non-finite" in meta["inherited_ic_pin_era"][1]["reason"]
    assert meta["inherited_ic_pin_era"][5]["status"] == "pass"

    h1_cells = [c for c in cells if c.H == 1]
    h5_cells = [c for c in cells if c.H == 5]
    assert h1_cells and all(c.status == "dropped" for c in h1_cells)
    assert all("pin-era inherited-IC GUARD FAILED" in c.reason for c in h1_cells)
    assert h5_cells and any(c.status == "ok" for c in h5_cells)
    assert meta["n_cells_recorded"] == meta["n_cells_expected"]


def test_sweep_panel_ic_is_never_gated_and_the_pins_module_attribute_is_restored(monkeypatch):
    """D60 item 3: the sweep-panel IC call must go through the SAME lineage function
    (`decide.inherited_signal_check_chunked`) but NON-ASSERTING — achieved by rebinding
    `backtest.SIGNAL_806_IC_FULL` to a new empty dict for that one call and restoring the
    ORIGINAL object afterwards. This is proven two ways: (1) the stub's own snapshot of
    `backtest.SIGNAL_806_IC_FULL` at call time is EMPTY for every sweep-panel call and
    NON-EMPTY (the real pins) for every pin-era call; (2) after the run, the module attribute IS
    (identity, not just equality) the exact object captured before the call.

    PERTURBATION THIS IS RED AGAINST: delete the `backtest.SIGNAL_806_IC_FULL = {}` /
    `finally: backtest.SIGNAL_806_IC_FULL = _orig_pins` scope-patch around the sweep-panel call
    in `_run_sweep_inner`'s per-H loop. Mutation-verified by hand (see task report): with it
    removed, the sweep-panel call's `pins_snapshot` reads the real non-empty pins dict instead of
    `{}` — RED against assertion (1). Restored, this test passes again."""
    sw = _sweep()   # MUST run before `import backtest` — this is what puts
                    # runs/808_linear_backtest on sys.path in an otherwise-fresh process.
    import backtest
    orig_pins_obj = backtest.SIGNAL_806_IC_FULL
    assert orig_pins_obj  # sanity: the real pins are non-empty going in
    calls = _stub_897_pipeline(monkeypatch, sw)
    sw.run_sweep(
        attainable_maker_fee_bp=[8.0], db_path="fixtures/trading_fixture.db",
        train_start="2024-01-01", val_start="2024-01-03", windows=_D60_SWEEP_WINDOWS,
        grid_h=(5,), grid_k=(16,), grid_variant=("long_only",), grid_tau=(0.75,), grid_n=(0,),
        grid_fee_bp=(8.0,), grid_adv_bp=(0.0,), n_boot=5,
    )
    check_calls = [c for c in calls["fit_check"] if c["fn"] == "check"]
    pin_checks = [c for c in check_calls if c["which"] == "pin"]
    sweep_checks = [c for c in check_calls if c["which"] == "sweep"]
    assert pin_checks and sweep_checks
    for c in pin_checks:
        assert c["pins_snapshot"] == orig_pins_obj
    for c in sweep_checks:
        assert c["pins_snapshot"] == {}
    assert backtest.SIGNAL_806_IC_FULL is orig_pins_obj


# ============================================================================================
# TASK_897 D60 fix pass, cycle 2 — three holes the review panel found in the re-based guard.
# Each test below is mutation-verified RED-then-GREEN against ONE named perturbation, and each
# is run BOTH in the full suite AND under `-k` isolation (the previous cycle's trap: a test
# that only passes by accident of full-suite import/patch ordering).
# ============================================================================================

def test_pin_era_guard_refuses_an_H_that_carries_no_806_pin_instead_of_passing_vacuously(
        monkeypatch):
    """H1 (MAJOR) — the SECOND, independent vacuous-pass path, distinct from the non-finite one.

    `decide.inherited_signal_check_chunked` (809 lineage, byte-frozen) compares its IC only
    `if tgt is not None`, where `tgt = backtest.SIGNAL_806_IC_FULL.get(H)`. For any H outside
    the pinned set {1, 5, 20} that short-circuits and the function returns a real, FINITE,
    NEVER-COMPARED IC — which the `np.isfinite` rider happily waves through as
    `{"ic": <real number>, "status": "pass"}`, under an artifact bullet that calls it "the
    lineage-fidelity proof; GATED". Reproduced live at H=2 before the fix
    (`ic: -0.0036549..., status: 'pass'`). Not reachable from today's CLI (`GRID_H` /
    `SMOKE_GRID_H` are subsets of the pinned set) but fully reachable through the public
    `run_sweep(grid_h=...)` API — the exact reuse surface D60 points the 898-class decay work at.

    PERTURBATION THIS IS RED AGAINST: delete the `tgt_pin = backtest.SIGNAL_806_IC_FULL.get(H)`
    / `if tgt_pin is None: raise AssertionError(...)` block from `_run_sweep_inner`'s pin-era
    loop. Mutation-verified by hand (see task report): with it removed, H=2's
    `inherited_ic_pin_era` status reads "pass" carrying the stub's finite IC, and H=2's cells
    are NOT dropped — RED against the first and third assertion groups below."""
    sw = _sweep()
    import backtest
    # The precondition, read from the LINEAGE itself rather than asserted from prose: the pins
    # are exactly {1, 5, 20}, and the lineage's compare really is `tgt is not None`-gated.
    assert 2 not in backtest.SIGNAL_806_IC_FULL
    lineage_src = (_REPO / "runs" / "809_decision_rule_rescue" / "decide.py").read_text()
    assert "tgt is not None and abs(ic - tgt) > backtest.SIGNAL_IC_TOL" in lineage_src, (
        "the 809 lineage's short-circuit changed shape — re-derive H1 before trusting this test"
    )

    calls = _stub_897_pipeline(monkeypatch, sw, pin_ic=-0.0036549)
    cells, meta = sw.run_sweep(
        attainable_maker_fee_bp=[8.0], db_path="fixtures/trading_fixture.db",
        train_start="2024-01-01", val_start="2024-01-03", windows=_D60_SWEEP_WINDOWS,
        grid_h=(2, 5), grid_k=(16,), grid_variant=("long_only",), grid_tau=(0.75,), grid_n=(0,),
        grid_fee_bp=(8.0,), grid_adv_bp=(0.0,), n_boot=5,
    )
    # (1) the off-grid H is recorded NOT-EVALUABLE, never "pass", and carries no IC.
    assert meta["inherited_ic_pin_era"][2]["status"] == "failed"
    assert meta["inherited_ic_pin_era"][2]["ic"] is None
    reason = meta["inherited_ic_pin_era"][2]["reason"]
    assert "NOT-EVALUABLE" in reason
    assert "no 806 IC pin" in reason
    assert "LINEAGE FIDELITY WAS NOT ESTABLISHED" in reason
    # (2) the guard refuses BEFORE producing an IC nothing would compare — no pin-era lineage
    #     call is made for H=2 at all.
    assert not [c for c in calls["fit_check"] if c["H"] == 2 and c["which"] == "pin"]
    # (3) a pinned H alongside it is unaffected, its cells proceed, and gate (ii) holds:
    #     the off-grid H's cells DROP and the recorded/expected counts still match.
    assert meta["inherited_ic_pin_era"][5]["status"] == "pass"
    h2_cells = [c for c in cells if c.H == 2]
    h5_cells = [c for c in cells if c.H == 5]
    assert h2_cells and all(c.status == "dropped" for c in h2_cells)
    assert all("pin-era inherited-IC GUARD FAILED" in c.reason for c in h2_cells)
    assert h5_cells and any(c.status == "ok" for c in h5_cells)
    assert meta["n_cells_recorded"] == meta["n_cells_expected"]


def test_artifact_sweep_panel_ic_bullet_states_what_the_code_actually_does_under_skip(
        monkeypatch, tmp_path):
    """H2 (MINOR) — the shipped string must not claim something the code does not do.

    `meta.setdefault("signal_ic_sweep_panel", {})[H] = ic` runs UNCONDITIONALLY for every H that
    reaches the fit, with `ic = None` under `--skip-inherited-ic-check` — so the dict renders as
    `{5: None}`, NOT `{}`. The bullet claimed "(empty if `--skip-inherited-ic-check`)". The
    OTHER new bullet's "empty if skip" claim is true (the pin-era guard dict really is never
    populated under the flag) and is asserted here too, so the fix cannot be over-applied.
    This project has escalated this exact shape twice (D53, D57): a false claim in a shipped
    string, rendered exactly where a reader checks legitimacy.

    PERTURBATION THIS IS RED AGAINST: restore the sweep-panel bullet's wording to
    "(empty if `--skip-inherited-ic-check`)". Mutation-verified by hand (see task report): the
    rendered bullet then asserts "empty" beside a literal `{5: None}` on the same line — RED
    against the `"empty if" not in bullet` assertion."""
    sw = _sweep()
    _stub_897_pipeline(monkeypatch, sw)
    cells, meta = sw.run_sweep(
        attainable_maker_fee_bp=[8.0], db_path="fixtures/trading_fixture.db",
        train_start="2024-01-01", val_start="2024-01-03", windows=_D60_SWEEP_WINDOWS,
        grid_h=(5,), grid_k=(16,), grid_variant=("long_only",), grid_tau=(0.75,), grid_n=(0,),
        grid_fee_bp=(8.0,), grid_adv_bp=(0.0,), n_boot=5, verify_inherited_ic=False,
    )
    # GROUND TRUTH, measured off the run itself — the two dicts behave DIFFERENTLY under skip.
    assert meta["signal_ic_sweep_panel"] == {5: None}   # keyed, valued None — NOT empty
    assert meta["inherited_ic_pin_era"] == {}           # this one genuinely IS empty

    out = tmp_path / "result_h2.md"
    sw.write_artifact(cells, meta, out)
    lines = out.read_text().splitlines()
    sweep_bullets = [ln for ln in lines if "Signal IC, SWEEP-PANEL basis" in ln]
    guard_bullets = [ln for ln in lines if "Inherited-IC GUARD (pin-era basis" in ln]
    assert len(sweep_bullets) == 1 and len(guard_bullets) == 1
    sweep_bullet, guard_bullet = sweep_bullets[0], guard_bullets[0]
    # The sweep-panel bullet renders the real dict and describes it truthfully.
    assert "{5: None}" in sweep_bullet
    assert "None per H if" in sweep_bullet
    assert "empty if" not in sweep_bullet
    # The pin-era GUARD bullet's own "empty if skip" claim is TRUE and stays.
    assert "empty if" in guard_bullet
    assert "{}" in guard_bullet


def test_pin_era_panel_build_failure_frees_the_panel_before_the_sweep_build(monkeypatch):
    """H3 (MINOR) — cleanup symmetry between the pin-era build's success and except paths.

    The success path does `del ohlcv_pin; gc.collect()`. An exception raised AFTER the
    `ohlcv_pin, grid_ts_pin = build_aligned_ohlcv(...)` bind but BEFORE that `del` (driven here,
    as the reviewing judge drove it, by a stubbed build returning a malformed 3-channel array,
    so `ohlcv_pin[:, :, 3]` raises) left the full pin-era panel alive in `_run_sweep_inner`'s
    frame for the rest of the function — straight through the much larger sweep-panel build
    immediately below, the worst possible moment for the scheduled leg's `safe_run --mem 64G`
    budget. The judge did NOT prove this reachable from the real 806-lineage
    `build_aligned_ohlcv`; the asymmetry is closed anyway because it is cheap and the memory
    window is the scheduled leg's.

    The liveness probe fires AT THE SWEEP-PANEL BUILD, not after `run_sweep` returns — after
    return the frame dies and the reference is released either way, so a post-hoc check would be
    green under the bug and prove nothing.

    PERTURBATION THIS IS RED AGAINST: delete the `ohlcv_pin = close_pin = grid_ts_pin = None;
    gc.collect()` lines from the pin-era build's `except` branch (and the `ohlcv_pin =` from the
    pre-`try` initialisation). Mutation-verified by hand (see task report): the weakref then
    reads ALIVE at the sweep-panel build — RED against the load-bearing assertion below."""
    import weakref

    import numpy as np
    sw = _sweep()
    import probe
    _stub_897_pipeline(monkeypatch, sw)
    holder = {}

    def malformed_build_aligned_ohlcv(cfg, asset_ids, bar_sec):
        """Returns a 3-CHANNEL panel: the bind succeeds, `[:, :, 3]` one line later raises."""
        grid_start = probe._to_epoch(cfg.train_start) - cfg.W * 60
        grid_end = probe._to_epoch(cfg.test_end)
        grid_ts = np.arange(grid_start, grid_end, 60, dtype=np.int64)
        bad = np.zeros((len(grid_ts), len(asset_ids), 3), dtype=np.float64)
        holder["wref"] = weakref.ref(bad)
        holder["nbytes"] = bad.nbytes
        return bad, grid_ts

    monkeypatch.setattr(probe, "build_aligned_ohlcv", malformed_build_aligned_ohlcv)

    stubbed_sweep_build = sw.alias_layer.build_aligned_ohlcv_with_alias_union

    def probing_sweep_build(cfg, asset_ids, alias_groups, bar_sec):
        # The moment that matters: the next large allocation in the same frame.
        holder["alive_at_sweep_build"] = holder["wref"]() is not None
        return stubbed_sweep_build(cfg, asset_ids, alias_groups, bar_sec)

    monkeypatch.setattr(sw.alias_layer, "build_aligned_ohlcv_with_alias_union",
                        probing_sweep_build)

    cells, meta = sw.run_sweep(
        attainable_maker_fee_bp=[8.0], db_path="fixtures/trading_fixture.db",
        train_start="2024-01-01", val_start="2024-01-03", windows=_D60_SWEEP_WINDOWS,
        grid_h=(5,), grid_k=(16,), grid_variant=("long_only",), grid_tau=(0.75,), grid_n=(0,),
        grid_fee_bp=(8.0,), grid_adv_bp=(0.0,), n_boot=5,
    )
    assert "wref" in holder, "the malformed pin-era build never ran"
    assert holder["nbytes"] > 1_000_000, "the probe panel is too small to be a meaningful check"
    assert "alive_at_sweep_build" in holder, "the sweep-panel build never ran"
    # THE LOAD-BEARING ASSERTION: no reference to the failed pin-era panel survives into the
    # sweep-panel build.
    assert holder["alive_at_sweep_build"] is False, (
        "the failed pin-era panel is STILL REFERENCED when the (much larger) sweep panel is "
        "built — the except branch did not match the success path's cleanup discipline"
    )
    # ...and the failure is still recorded per gate (ii): every H dropped, counts consistent.
    assert meta["inherited_ic_pin_era"][5]["status"] == "failed"
    assert "pin-era PANEL BUILD failed" in meta["inherited_ic_pin_era"][5]["reason"]
    assert cells and all(c.status == "dropped" for c in cells)
    assert meta["n_cells_recorded"] == meta["n_cells_expected"]


# ============================================================================================
# TASK_902 — 897 instrument hardening. Items 1-12; items 2/10 are demonstrated by the EXISTING
# D60/D59 tests above (item 2's two vacuous-pass causes were already pinned by
# `test_pin_era_guard_refuses_an_H_that_carries_no_806_pin_instead_of_passing_vacuously` and
# `test_pin_era_guard_failure_or_nan_drops_...`; item 10's TOCTOU fix is demonstrated by the
# updated return-value assertions in the fixture/protected/relative `assert_db_path_allowed`
# tests near the top of this file). Everything below is new.
# ============================================================================================

# ---------------------------------------------------------------------------------- item 1 (D58)

def test_exit_code_for_flags_a_gate_ii_violation_and_clears_an_honest_empty_result():
    """A gate-(ii) completeness violation (recorded != expected) exits non-zero; an HONEST empty
    result (every cell accounted for, none reached "ok" -- the fixture's normal state) exits 0,
    and so does a fully-successful run."""
    sw = _sweep()
    assert sw.exit_code_for({"n_cells_expected": 144, "n_cells_recorded": 100}) == 1
    assert sw.exit_code_for({"n_cells_expected": 144, "n_cells_recorded": 144,
                              "n_cells_ok": 0}) == 0
    assert sw.exit_code_for({"n_cells_expected": 144, "n_cells_recorded": 144,
                              "n_cells_ok": 144}) == 0


def test_main_exits_nonzero_on_a_forced_gate_ii_violation(tmp_path, monkeypatch):
    """End-to-end through `main()`: a run whose own meta fails gate (ii) must exit non-zero, and
    the artifact is still written (D58/orchestration parses the artifact, not just the code)."""
    sw = _sweep()
    out = tmp_path / "violation.md"

    def fake_run_sweep(**kwargs):
        meta = _artifact_meta(n_cells=144)
        meta["n_cells_recorded"] = 100          # forced violation: 100 != 144 expected
        return [], meta

    monkeypatch.setattr(sw, "run_sweep", fake_run_sweep)
    monkeypatch.setattr(sys, "argv", [
        "prog", "--attainable-maker-fee-bp", "8", "--db-path", "fixtures/trading_fixture.db",
        "--out", str(out), "--smoke"])
    with pytest.raises(SystemExit) as e:
        sw.main()
    assert e.value.code == 1
    assert out.exists()


def test_main_does_not_exit_nonzero_on_an_honest_empty_result(tmp_path, monkeypatch):
    """The complementary direction: recorded == expected (every cell the grid promised WAS
    recorded) with ok == 0 -- an HONEST EMPTY RESULT, the normal fixture state -- must NOT raise
    SystemExit at all.

    TASK_902 fix cycle 2, MAJOR: this test used to encode the honest-empty case as
    `expected == recorded == 0`, which is NOT an honest empty result -- it is a run with NO CELLS
    AT ALL, where gate (ii)'s `recorded == expected` check is satisfied trivially by `0 == 0` and
    a degenerate grid reads as healthy. That case is now LOUD (its own banner + exit 2, see
    `test_a_zero_cell_run_is_loud_instead_of_reading_as_healthy`). D58's honest-empty result is
    the one pinned here: cells recorded, none measured."""
    sw = _sweep()
    out = tmp_path / "empty.md"

    def fake_run_sweep(**kwargs):
        meta = _artifact_meta(n_cells=144)       # expected == recorded == 144
        meta["n_cells_ok"] = 0                   # ...and NOT ONE of them measured anything
        meta["n_cells_dropped_or_failed"] = 144
        return [], meta

    monkeypatch.setattr(sw, "run_sweep", fake_run_sweep)
    monkeypatch.setattr(sys, "argv", [
        "prog", "--attainable-maker-fee-bp", "8", "--db-path", "fixtures/trading_fixture.db",
        "--out", str(out), "--smoke"])
    sw.main()                    # must complete normally -- no SystemExit
    assert out.exists()


# --------------------------------------------------------------------------------- item 2b (D58)

def test_absent_db_path_writes_a_refusal_artifact_and_exits_nonzero_instead_of_crashing(
        tmp_path, monkeypatch):
    """An absent/unreadable --db-path used to crash UNCAUGHT with NO artifact written at all
    (`sqlite3.OperationalError` propagating straight out of `report_bar_frequency`, before
    `write_artifact` was ever reachable). It must now refuse in a CONTROLLED way: a legible
    artifact at --out, and a non-zero exit."""
    sw = _sweep()
    missing = tmp_path / "does_not_exist.db"
    out = tmp_path / "refusal.md"
    monkeypatch.setattr(sys, "argv", [
        "prog", "--attainable-maker-fee-bp", "8", "--db-path", str(missing),
        "--out", str(out), "--smoke"])
    with pytest.raises(SystemExit) as e:
        sw.main()
    assert e.value.code == 1
    assert out.exists(), "no artifact was written -- the exact defect item 2b closes"
    text = out.read_text()
    assert "REFUSED BEFORE ANY CELL WAS MEASURED" in text
    assert "NO CELL IN THIS ARTIFACT WAS EVER MEASURED" in text
    assert "OperationalError" in text
    assert "unable to open database file" in text
    assert not missing.exists(), "a mode=ro connect must never CREATE the target file"


def test_a_systemexit_refusal_still_propagates_unwrapped_through_the_new_except(monkeypatch):
    """The new `except Exception` in `main()` must NOT swallow the harness's own clean
    `SystemExit` refusals (gate (iv), `assert_db_path_allowed`, etc.) -- `SystemExit` is a
    `BaseException`, not an `Exception`, and this pins that it is never caught here."""
    sw = _sweep()
    monkeypatch.setattr(sys, "argv", [
        "prog", "--attainable-maker-fee-bp", "8", "--db-path", "trading_data.db?x=1"])
    with pytest.raises(SystemExit) as e:
        sw.main()
    assert "not a filesystem path" in str(e.value)   # the ORIGINAL clean refusal, unwrapped


# ---------------------------------------------------------------------------------- item 2c

def test_realized_book_width_is_recorded_and_the_instrumentation_cleans_up_after_itself(
        monkeypatch):
    """`realized_book_width_mean` is measured by scope-patching `decide._decide_step` for the
    duration of ONE `_run_decision` call (the same rebind-the-name pattern this file already
    uses for `decide.CFG`/`probe.sqlite3`) -- it must (a) record a sane, finite, <=k book width
    for a cell with real decisions, and (b) leave `decide._decide_step` restored to the ORIGINAL
    function afterward, exactly like those other scope-patches."""
    import numpy as np
    sw = _sweep()
    import decide
    orig_decide_step = decide._decide_step
    orig_run_decision = decide._run_decision
    _stub_897_pipeline(monkeypatch, sw)
    # `_stub_897_pipeline` stubs `decide._run_decision` too (for OTHER tests' speed); restore the
    # REAL engine here so the instrumentation wraps the genuine `_decide_step`.
    monkeypatch.setattr(decide, "_run_decision", orig_run_decision)
    # `_stub_897_pipeline`'s `_chunked_predict` stub returns all-ZERO predictions, which
    # `_decide_step` (`p[a] > 0.0`) never treats as eligible -- no cell would ever realize a
    # decision. Override with reliably-positive predictions so the REAL decision engine actually
    # holds something to instrument.
    rng = np.random.default_rng(0)

    def fake_chunked_predict_nonzero(close, rows, w, mu, sd, W, chunk=20000):
        return rng.normal(loc=1.0, scale=0.1, size=(len(rows), close.shape[1]))

    monkeypatch.setattr(decide, "_chunked_predict", fake_chunked_predict_nonzero)

    cells, meta = sw.run_sweep(
        attainable_maker_fee_bp=[8.0], db_path="fixtures/trading_fixture.db",
        train_start="2024-01-01", val_start="2024-01-03", windows=_D60_SWEEP_WINDOWS,
        grid_h=(5,), grid_k=(1,), grid_variant=("long_only",), grid_tau=(0.75,), grid_n=(0,),
        grid_fee_bp=(8.0,), grid_adv_bp=(0.0,), n_boot=5,
    )
    measured = [c for c in cells if c.status == "ok" and c.n_decisions and c.n_decisions > 0]
    assert measured, "expected at least one cell with real decisions to demonstrate this on"
    for c in measured:
        assert np.isfinite(c.realized_book_width_mean)
        assert 0.0 < c.realized_book_width_mean <= c.k + 1e-9
    # the instrumentation cleans up after itself -- no leaked wrapper survives the run.
    assert decide._decide_step is orig_decide_step


def test_a_zero_decision_ok_cell_states_that_it_is_empty(tmp_path):
    """A cell can reach `status='ok'` (not a fit/engine failure) while realizing ZERO decisions
    (`decide._decide_step`'s `npr < (2*k if long_short else k)` empty-target floor) -- that used
    to render indistinguishably from a real measurement. It must now say EMPTY in its own row."""
    sw = _sweep()
    empty_cell = sw.CellRecord(
        window="a", H=5, k=16, variant="long_short", tau=0.9, N=0, fee_bp=8.0, adv_bp=5.0,
        status="ok", reason="", n_bars=9000, n_decisions=0, decisions_per_year=float("nan"),
        mean_hold_bars=float("nan"), turnover_mean=float("nan"), fill_rate=float("nan"),
        sharpe_decision=float("nan"), ci_lo_UNCALIBRATED=float("nan"),
        ci_hi_UNCALIBRATED=float("nan"), ci_advisory=False, ci_calibration_k=float("nan"),
        ci_calibration_source_n_dec=-1, ci_lo_ADVISORY=float("nan"), ci_hi_ADVISORY=float("nan"),
        block_used=-1, measured_dependence_len=float("nan"), pnl=float("nan"),
        maxdd=float("nan"), sharpe_per_bar_diagnostic=float("nan"),
        realized_book_width_mean=0.0,
    )
    measured_cell = sw.CellRecord(
        window="a", H=5, k=16, variant="long_short", tau=0.9, N=0, fee_bp=8.0, adv_bp=5.0,
        status="ok", reason="", n_bars=9000, n_decisions=50, decisions_per_year=1000.0,
        mean_hold_bars=3.0, turnover_mean=0.03, fill_rate=0.5, sharpe_decision=1.2,
        ci_lo_UNCALIBRATED=0.4, ci_hi_UNCALIBRATED=2.0, ci_advisory=False,
        ci_calibration_k=float("nan"), ci_calibration_source_n_dec=-1,
        ci_lo_ADVISORY=float("nan"), ci_hi_ADVISORY=float("nan"), block_used=-1,
        measured_dependence_len=float("nan"), pnl=0.01, maxdd=-0.02,
        sharpe_per_bar_diagnostic=5.0, realized_book_width_mean=3.5,
    )
    out = tmp_path / "r.md"
    sw.write_artifact([empty_cell, measured_cell], _artifact_meta(n_cells=2), out)
    rows = [ln for ln in out.read_text().splitlines() if ln.startswith("| a | 5 | 16 |")]
    assert len(rows) == 2
    empty_row = [r for r in rows if r.split("|")[11].strip() == "0"][0]     # n_dec column == 0
    other_row = [r for r in rows if r.split("|")[11].strip() == "50"][0]
    assert "EMPTY (0 decisions" in empty_row
    assert "EMPTY" not in other_row
    assert "3.50" in other_row       # realized_book_width_mean rendered in the new column


# ------------------------------------------------------------------------------------- item 3

def test_borrowed_pairing_table_renders_its_own_reason_column(tmp_path):
    """The borrowed-pairing table used to render no `reason` column even though pairing 639
    CANNOT REPRODUCE by construction and `calibration.py` already carries a rich `reason` string
    for it -- the per-stratum table right beside it had a `reason` column, this one did not."""
    sw = _sweep()
    out = tmp_path / "r.md"
    sw.write_artifact([], _artifact_meta(n_cells=0), out)
    lines = out.read_text().splitlines()
    header_idx = [i for i, ln in enumerate(lines)
                  if ln.startswith("  | n_dec (borrower)")][0]
    assert "reason" in lines[header_idx]
    # There is ANOTHER table earlier in the artifact (the seed-jitter demotion narrative) whose
    # rows also start with "  | 639 |" -- restrict the search to lines AFTER this table's own
    # header so this test cannot accidentally match that other table.
    row_639 = [ln for ln in lines[header_idx:] if ln.startswith("  | 639 |")][0]
    assert "BORROWED-PAIRING FAILURE" in row_639


# ------------------------------------------------------------------------------------- item 4

def test_scope_honesty_prose_states_no_pass_rate_on_an_unmeasured_run(tmp_path):
    """The Scope-honesty section used to assert an UNCONDITIONAL 'near-100% pass rate' even on a
    run that measured NO pass rate at all (e.g. the empty fixture run) -- a specific numeric
    claim that run does not back."""
    sw = _sweep()
    out = tmp_path / "empty.md"
    sw.write_artifact([], _artifact_meta(n_cells=0), out)
    text = out.read_text()
    assert "near-100% pass rate" not in text
    assert "recorded NO measured pass rate at all" in text


def test_scope_honesty_prose_states_the_real_pass_rate_when_one_exists(tmp_path):
    sw = _sweep()

    def _cell(dep):
        return sw.CellRecord(
            window="a", H=5, k=16, variant="long_short", tau=0.9, N=0, fee_bp=8.0, adv_bp=5.0,
            status="ok", reason="", n_bars=9000, n_decisions=3000, decisions_per_year=1000.0,
            mean_hold_bars=3.0, turnover_mean=0.03, fill_rate=0.5, sharpe_decision=1.2,
            ci_lo_UNCALIBRATED=0.4, ci_hi_UNCALIBRATED=2.0, ci_advisory=False,
            ci_calibration_k=float("nan"), ci_calibration_source_n_dec=-1,
            ci_lo_ADVISORY=float("nan"), ci_hi_ADVISORY=float("nan"),
            block_used=60, measured_dependence_len=dep,
            pnl=0.01, maxdd=-0.02, sharpe_per_bar_diagnostic=5.0)

    cells = [_cell(12.0), _cell(480.0)]     # one passes (12<=60), one refused (480>60)
    out = tmp_path / "r.md"
    sw.write_artifact(cells, _artifact_meta(n_cells=2), out)
    text = out.read_text()
    assert ("this artifact's 1/2 pass rate on the ok cells the guard ran on is NOT evidence"
            in text)


# ------------------------------------------------------------------------------------- item 5

def test_dep_le_block_column_renders_not_run_distinctly_from_false(tmp_path):
    """A cell the dependence guard never ran on (too few decisions for any interval to exist)
    must render `not-run` in the `dep<=block` column, not a bare `False` -- which used to read
    as a MEASUREMENT (the guard ran and refused) rather than an absence of one."""
    sw = _sweep()

    def _cell(n_dec, block, dep):
        return sw.CellRecord(
            window="a", H=5, k=16, variant="long_short", tau=0.9, N=0, fee_bp=8.0, adv_bp=5.0,
            status="ok", reason="", n_bars=9000, n_decisions=n_dec, decisions_per_year=1000.0,
            mean_hold_bars=3.0, turnover_mean=0.03, fill_rate=0.5,
            sharpe_decision=1.2 if n_dec else float("nan"),
            ci_lo_UNCALIBRATED=0.4 if n_dec else float("nan"),
            ci_hi_UNCALIBRATED=2.0 if n_dec else float("nan"), ci_advisory=False,
            ci_calibration_k=float("nan"), ci_calibration_source_n_dec=-1,
            ci_lo_ADVISORY=float("nan"), ci_hi_ADVISORY=float("nan"),
            block_used=block, measured_dependence_len=dep,
            pnl=0.01, maxdd=-0.02, sharpe_per_bar_diagnostic=5.0)

    no_ci_cell = _cell(0, -1, float("nan"))       # DEP_NO_CI: guard never ran
    refused_cell = _cell(3000, 60, 480.0)         # DEP_REFUSED: guard ran, refused
    out = tmp_path / "r.md"
    sw.write_artifact([no_ci_cell, refused_cell], _artifact_meta(n_cells=2), out)
    rows = [ln for ln in out.read_text().splitlines() if ln.startswith("| a | 5 | 16 |")]
    assert len(rows) == 2
    no_ci_row = [r for r in rows if "no CI computed" in r][0]
    refused_row = [r for r in rows if "REFUSED ON DEPENDENCE" in r][0]
    assert "| not-run |" in no_ci_row
    assert "| not-run |" not in refused_row
    assert "| False |" in refused_row


# ------------------------------------------------------------------------------------- item 6

def test_a_100_percent_failed_run_gets_a_top_level_banner(tmp_path):
    """A 0-decision run (every `ok` cell had zero decisions) gets a large banner; a run where
    every cell's decision engine or upstream guard FAILED (no cell ever reached `status="ok"`)
    used to get none. Both must now get a top-level statement, and they must not be conflated."""
    sw = _sweep()
    dropped_cell = sw.CellRecord(
        window="a", H=5, k=16, variant="long_short", tau=0.9, N=0, fee_bp=8.0, adv_bp=5.0,
        status="dropped", reason="pin-era inherited-IC GUARD FAILED", n_bars=0, n_decisions=0,
        decisions_per_year=float("nan"), mean_hold_bars=float("nan"), turnover_mean=float("nan"),
        fill_rate=float("nan"), sharpe_decision=float("nan"), ci_lo_UNCALIBRATED=float("nan"),
        ci_hi_UNCALIBRATED=float("nan"), ci_advisory=False, ci_calibration_k=float("nan"),
        ci_calibration_source_n_dec=-1, ci_lo_ADVISORY=float("nan"), ci_hi_ADVISORY=float("nan"),
        block_used=-1, measured_dependence_len=float("nan"), pnl=float("nan"),
        maxdd=float("nan"), sharpe_per_bar_diagnostic=float("nan"),
    )
    out = tmp_path / "r.md"
    meta = _artifact_meta(n_cells=1)
    meta["n_cells_ok"] = 0
    meta["n_cells_dropped_or_failed"] = 1
    sw.write_artifact([dropped_cell], meta, out)
    text = out.read_text()
    assert "EVERY CELL FAILED" in text
    assert "THIS ARTIFACT CONTAINS NO `ok` CELL AT ALL" in text
    assert "NO DECISIONS IN THIS RUN" not in text   # the OTHER banner must not also fire


# ------------------------------------------------------------------------------------- item 7

def test_out_default_points_into_the_untracked_runs_tree(monkeypatch, tmp_path):
    """A bare run (no --out) must default into the UNTRACKED runs/ tree, never into the tracked
    experiments/897_corrected_809/ directory it used to write into by default."""
    import subprocess
    sw = _sweep()
    captured = {}

    def fake_run_sweep(**kwargs):
        # n_cells=144, not 0: a zero-cell run is now a LOUD non-zero exit (fix cycle 2, MAJOR),
        # and this test is about the DEFAULT --out path, not about the exit-code contract.
        return [], _artifact_meta(n_cells=144)

    def fake_write_artifact(cells, meta, path):
        captured["path"] = path

    monkeypatch.setattr(sw, "run_sweep", fake_run_sweep)
    monkeypatch.setattr(sw, "write_artifact", fake_write_artifact)
    monkeypatch.setattr(sys, "argv", [
        "prog", "--attainable-maker-fee-bp", "8", "--db-path", "fixtures/trading_fixture.db",
        "--smoke"])
    sw.main()
    assert "path" in captured
    default_out = captured["path"]
    assert "experiments/897_corrected_809" not in str(default_out)
    assert str(default_out) == str(_REPO / "runs" / "897_corrected_809" / "result.md")
    res = subprocess.run(["git", "check-ignore", "-q", str(default_out)], cwd=str(_REPO))
    assert res.returncode == 0, f"{default_out} is NOT gitignored -- item 7 is not actually safe"


# ------------------------------------------------------------------------------------- item 8

def test_dependence_refusal_oc_grid_gap_is_disclosed(tmp_path):
    """`MEASURED_DEPENDENCE_REFUSAL_OC` only discloses dependence multiples {0,1,2,4,8}x; the
    known worse intermediate point (1.5x, 8-13% certify at 0.905 coverage) is now named
    explicitly rather than silently absent from the grid."""
    sw = _sweep()
    out = tmp_path / "r.md"
    sw.write_artifact([], _artifact_meta(n_cells=0), out)
    text = out.read_text()
    assert "1.5x the guard certifies" in text
    assert "8-13%" in text
    assert "{0, 1, 2, 4, 8}x the block ONLY" in text


# ------------------------------------------------------------------------------------- item 9

def test_run_sweep_itself_refuses_the_protected_db_without_the_acknowledgement():
    """`assert_db_path_allowed` used to be bound ONLY in `main()`; a PROGRAMMATIC caller of
    `run_sweep` (the reuse surface D60 points the 898-class decay work at) could reach the
    protected DB without ever passing the acknowledgement flag. Bind it in `run_sweep` too."""
    sw = _sweep()
    real = str(sw.PROTECTED_REAL_DB)
    with pytest.raises(SystemExit) as e:
        sw.run_sweep(attainable_maker_fee_bp=[8.0], db_path=real)
    assert "PROTECTED read-only source of record" in str(e.value)


def test_run_sweep_accepts_the_protected_db_when_acknowledged(monkeypatch):
    """The complementary direction, fully stubbed so no real file is ever touched (every
    DB-touching call `_stub_897_pipeline` intercepts is monkeypatched to a synthetic generator
    regardless of `cfg.db_path`'s value): the acknowledgement flag DOES reach `run_sweep`'s own
    internal gate (not just `main()`'s), and once passed, the run proceeds normally -- while the
    SAME call without it still refuses, everything else held identical."""
    sw = _sweep()
    _stub_897_pipeline(monkeypatch, sw)
    real = str(sw.PROTECTED_REAL_DB)
    kwargs = dict(
        attainable_maker_fee_bp=[8.0], db_path=real,
        train_start="2024-01-01", val_start="2024-01-03", windows=_D60_SWEEP_WINDOWS,
        grid_h=(5,), grid_k=(16,), grid_variant=("long_only",), grid_tau=(0.75,), grid_n=(0,),
        grid_fee_bp=(8.0,), grid_adv_bp=(0.0,), n_boot=5,
    )
    with pytest.raises(SystemExit):
        sw.run_sweep(**kwargs)                                    # not acknowledged -> refuses
    cells, meta = sw.run_sweep(scheduled_real_data_run=True, **kwargs)   # acknowledged -> proceeds
    assert meta["n_cells_recorded"] == meta["n_cells_expected"]


# ------------------------------------------------------------------------------------ item 11

def test_db_path_refusal_echo_is_clipped():
    """A huge --db-path used to be echoed VERBATIM into the refusal message (measured: a
    900k-char --db-path produced a 900k-char refusal). Uses the URI-syntax refusal path
    specifically, which never touches the filesystem (no realpath/stat calls), so this test
    cannot be flaky on OS path-length limits."""
    sw = _sweep()
    huge = "a" * 900_000 + "?x=1"
    with pytest.raises(SystemExit) as e:
        sw.assert_db_path_allowed(huge, False)
    msg = str(e.value)
    assert len(msg) < 5000, f"refusal message is {len(msg)} chars -- not clipped"
    assert "aaaa" in msg              # the value is still visibly named, not replaced outright
    assert "CLIPPED" in msg


def test_clip_echo_unit():
    sw = _sweep()
    assert sw._clip_echo("short") == "short"
    long_s = "b" * 5000
    clipped = sw._clip_echo(long_s)
    assert len(clipped) < len(long_s)
    assert clipped.startswith("b" * sw.ECHO_CLIP_LEN)
    assert "CLIPPED" in clipped and "5000 total" in clipped


# ------------------------------------------------------------------------------------ item 12

def test_grid_coverage_label_distinguishes_smoke_full_and_custom(tmp_path):
    """The artifact must state WHICH pre-registered grid ran -- smoke is normally 144 cells, the
    full pre-registered grid is normally 864, and a bare cell count must never be readable as
    full coverage on its own."""
    sw = _sweep()

    smoke_meta = _artifact_meta(n_cells=0)
    smoke_meta["grid"] = {"H": list(sw.SMOKE_GRID_H), "k": list(sw.SMOKE_GRID_K),
                           "variant": list(sw.SMOKE_GRID_VARIANT),
                           "tau": list(sw.SMOKE_GRID_TAU), "N": list(sw.SMOKE_GRID_N)}
    out1 = tmp_path / "smoke.md"
    sw.write_artifact([], smoke_meta, out1)
    assert "**SMOKE** grid" in out1.read_text()

    full_meta = _artifact_meta(n_cells=0)
    full_meta["grid"] = {"H": list(sw.GRID_H), "k": list(sw.GRID_K),
                          "variant": list(sw.GRID_VARIANT), "tau": list(sw.GRID_TAU),
                          "N": list(sw.GRID_N)}
    out2 = tmp_path / "full.md"
    sw.write_artifact([], full_meta, out2)
    assert "**FULL** grid" in out2.read_text()

    custom_meta = _artifact_meta(n_cells=0)      # grid = {"H": [5]} only, from the helper
    out3 = tmp_path / "custom.md"
    sw.write_artifact([], custom_meta, out3)
    assert "**CUSTOM** grid" in out3.read_text()


# ============================================================================================
# TASK_902 FIX CYCLE 2 — the findings from the two verification lenses on the first cycle.
# ============================================================================================

# ------------------------------------------------------- BLOCKER: refusal vs DEFECT, + traceback

_INJECTED_BUGS = [
    KeyError("nonexistent_key"),
    TypeError("unsupported operand type(s) for +: 'NoneType' and 'int'"),
]


@pytest.mark.parametrize("injected", _INJECTED_BUGS, ids=["KeyError", "TypeError"])
def test_an_arbitrary_bug_renders_as_a_DEFECT_never_as_a_governed_refusal(
        tmp_path, monkeypatch, capsys, injected):
    """THE BLOCKER. `main()`'s single `except Exception` rendered EVERY escape from `run_sweep` as
    the item-2b CONTROLLED REFUSAL artifact, in byte-identical prose: an injected `KeyError` or
    `TypeError` in `probe.report_bar_frequency` (an unrelated code defect, NOT a refusal
    condition) produced a file titled `REFUSED BEFORE ANY CELL WAS MEASURED` with exit 1, so a
    shipped bug and a missing DB were indistinguishable to the operator AND to the orchestrator
    (whose standing contract is to parse the artifact, never the exit code alone). Before item 2b
    the same bug produced a loud uncaught traceback -- strictly more diagnosable -- so the fix had
    made this failure mode worse. And `traceback.format_exc()` was called NOWHERE in the module,
    so the stack was destroyed and logged nowhere.

    Both halves are pinned here: the DEFECT shape, and the traceback surviving in the artifact AND
    on stderr. No test covered this path at all before (the only item-2b test uses the DB-not-found
    case, which *reads* like a real refusal), which is how the gap shipped green."""
    sw = _sweep()
    import probe
    out = tmp_path / "defect.md"

    def boom(cfg):
        raise injected

    monkeypatch.setattr(probe, "report_bar_frequency", boom)
    monkeypatch.setattr(sys, "argv", [
        "prog", "--attainable-maker-fee-bp", "8", "--db-path", "fixtures/trading_fixture.db",
        "--out", str(out), "--smoke"])
    with pytest.raises(SystemExit) as e:
        sw.main()
    assert e.value.code == 1
    assert out.exists(), "an artifact must still be written -- item 2b's contract"
    text = out.read_text()
    # (a) DEFECT-shaped...
    assert "UNEXPECTED INTERNAL ERROR" in text
    assert "A DEFECT, NOT A GOVERNED REFUSAL" in text
    assert "THIS IS A DEFECT IN THE INSTRUMENT" in text
    # (b) ...and NOT refusal-shaped. These are the exact strings an orchestrator greps for; the
    # two artifacts deliberately never quote each other's banner text.
    assert "REFUSED BEFORE ANY CELL WAS MEASURED" not in text
    assert "CONTROLLED REFUSAL record" not in text
    assert "NO CELL IN THIS ARTIFACT WAS EVER MEASURED" not in text
    # (c) the traceback is NOT destroyed -- in the artifact...
    assert "Traceback (most recent call last)" in text
    assert type(injected).__name__ in text
    assert "report_bar_frequency" in text, "the failing call site must be visible"
    # ...and on stderr, so an operator sees it without opening the file.
    err = capsys.readouterr().err
    assert "UNEXPECTED INTERNAL ERROR" in err
    assert "Traceback (most recent call last)" in err


def test_a_governed_gate_refusal_still_renders_as_a_controlled_refusal(tmp_path, monkeypatch):
    """The complementary direction of the BLOCKER: a GOVERNED refusal (here gate (iv)'s
    attainable-tier subset check, which raises `GateRefusal` from inside `run_sweep`) must still
    produce the item-2b CONTROLLED REFUSAL artifact -- narrowing the `except` must not have turned
    a designed refusal into a 'defect'."""
    sw = _sweep()
    out = tmp_path / "refusal.md"
    # 10 bp is not in the smoke fee grid (0, 8, 16) -> gate (iv)'s subset refusal, raised before
    # any data is touched.
    monkeypatch.setattr(sys, "argv", [
        "prog", "--attainable-maker-fee-bp", "10", "--db-path", "fixtures/trading_fixture.db",
        "--out", str(out), "--smoke"])
    with pytest.raises(SystemExit) as e:
        sw.main()
    assert e.value.code == 1
    text = out.read_text()
    assert text.splitlines()[0].startswith(
        "# TASK_897 -- gated_maker sweep artifact -- REFUSED BEFORE ANY CELL WAS MEASURED")
    assert "CONTROLLED REFUSAL record" in text
    assert "A GOVERNED REFUSAL" in text
    assert "GateRefusal" in text
    assert "GATE (iv) REFUSAL" in text
    # NOT defect-shaped: neither artifact quotes the other's banner.
    assert "UNEXPECTED INTERNAL ERROR" not in text
    assert "THIS IS A DEFECT IN THE INSTRUMENT" not in text


def test_gate_refusal_is_a_valueerror_subclass_so_existing_callers_are_unaffected():
    """`GateRefusal` narrows the TYPE of this harness's own gate refusals so they can be told
    apart from a stdlib/numpy `ValueError` raised by a real bug -- but it subclasses `ValueError`,
    so every existing caller and test that catches `ValueError` from `run_sweep` still works."""
    sw = _sweep()
    assert issubclass(sw.GateRefusal, ValueError)
    assert sw.GateRefusal in sw.REFUSAL_EXCEPTIONS
    import sqlite3 as _sq
    assert _sq.DatabaseError in sw.REFUSAL_EXCEPTIONS and OSError in sw.REFUSAL_EXCEPTIONS
    # The whole taxonomy, through the ONE predicate `main()` uses.
    assert sw._is_governed_refusal(sw.GateRefusal("gate (iv)"))
    # The missing-DB case (item 2b's own test): OperationalError is a DatabaseError.
    assert sw._is_governed_refusal(_sq.OperationalError("unable to open database file"))
    assert sw._is_governed_refusal(_sq.DatabaseError("file is not a database"))
    assert sw._is_governed_refusal(FileNotFoundError("no such file"))
    # A plain ValueError is NOT a governed refusal: it is the single most common way a real bug
    # (numpy shape mismatch, int() of garbage) surfaces, so it must render as a defect.
    assert not sw._is_governed_refusal(ValueError("x"))
    assert not sw._is_governed_refusal(KeyError("nonexistent_key"))
    assert not sw._is_governed_refusal(TypeError("NoneType + int"))
    # ...and sqlite3 API MISUSE is a DEFECT even though CPython makes `ProgrammingError` a
    # SUBCLASS of `DatabaseError` -- narrowing the tuple alone could not have excluded it, which
    # is exactly why the discrimination is a predicate and not just an `except` tuple.
    assert issubclass(_sq.ProgrammingError, _sq.DatabaseError)
    assert not sw._is_governed_refusal(_sq.ProgrammingError("bad API use"))


@pytest.mark.parametrize("base_exc", [SystemExit("clean refusal"), KeyboardInterrupt()],
                          ids=["SystemExit", "KeyboardInterrupt"])
def test_baseexceptions_pass_through_unwrapped_and_write_no_artifact(
        tmp_path, monkeypatch, base_exc):
    """`SystemExit` (every clean refusal in this file) and `KeyboardInterrupt` are
    `BaseException`, not `Exception`, so NEITHER handler in `main()` may see them: they propagate
    unwrapped and write no artifact of either kind. Pinned because the BLOCKER fix touched exactly
    that `except` structure."""
    sw = _sweep()
    import probe
    out = tmp_path / "never_written.md"

    def boom(cfg):
        raise base_exc

    monkeypatch.setattr(probe, "report_bar_frequency", boom)
    monkeypatch.setattr(sys, "argv", [
        "prog", "--attainable-maker-fee-bp", "8", "--db-path", "fixtures/trading_fixture.db",
        "--out", str(out), "--smoke"])
    with pytest.raises(type(base_exc)):
        sw.main()
    assert not out.exists(), "a BaseException must not be wrapped into an artifact"


# --------------------------------------------------- MINOR 2: the double fault must still speak

def test_a_double_fault_writing_the_refusal_artifact_still_reaches_the_operator(
        tmp_path, capsys):
    """`write_refusal_artifact`'s own `path.parent.mkdir()` was unguarded and OUTSIDE `main()`'s
    `except`: `--out <plainfile>/result.md` raises `FileExistsError` there, so NO artifact was
    written at all and the original refusal was buried behind an unrelated traceback -- reviving
    the exact 'traceback, exit 1, nothing to parse' state item 2b closed."""
    sw = _sweep()
    plain = tmp_path / "not_a_dir"
    plain.write_text("i am a file\n")
    out = plain / "result.md"           # parent is an existing PLAIN FILE
    sw.write_refusal_artifact(out, "OperationalError: unable to open database file",
                               "fixtures/trading_fixture.db", [8.0],
                               (("a", "2024-11-01", "2025-04-01"),), (5,), (16,),
                               ("long_only",), (0.75,), (0,))
    assert not out.exists()
    err = capsys.readouterr().err
    assert "DOUBLE FAULT" in err
    assert "FileExistsError" in err or "NotADirectoryError" in err
    # the record itself is not lost -- the whole artifact body goes to stderr
    assert "REFUSED BEFORE ANY CELL WAS MEASURED" in err
    assert "unable to open database file" in err
    assert "NOTHING WAS WRITTEN AT --out" in err


def test_a_double_fault_end_to_end_through_main_still_exits_cleanly(tmp_path, monkeypatch, capsys):
    """The same double fault, reached the way an operator would reach it: a missing --db-path
    (governed refusal) AND an --out whose parent is a plain file. `main()` must still exit 1 with
    a legible stderr record, never a raw `FileExistsError` traceback out of the handler."""
    sw = _sweep()
    plain = tmp_path / "not_a_dir"
    plain.write_text("i am a file\n")
    out = plain / "result.md"
    missing = tmp_path / "does_not_exist.db"
    monkeypatch.setattr(sys, "argv", [
        "prog", "--attainable-maker-fee-bp", "8", "--db-path", str(missing),
        "--out", str(out), "--smoke"])
    with pytest.raises(SystemExit) as e:
        sw.main()
    assert e.value.code == 1
    err = capsys.readouterr().err
    assert "DOUBLE FAULT" in err
    assert "unable to open database file" in err


# ------------------------------------------------- MAJOR: every echoing refusal site is clipped

_HUGE = "a" * 900_000


def test_every_echoing_refusal_site_is_clipped():
    """`_clip_echo` reached only 4 of the >=11 echoing sites after fix cycle 1. Measured before
    this fix: a 900,000-char `--attainable-maker-fee-bp` produced a 1,800,148-char `SystemExit`
    (the value echoed TWICE per message) and a 900,000-char `--window-a` produced 2,700,344 chars
    (THREE times) -- both LARGER than the 900k->1.6M TASK_888 defect item 11 cites as its own
    precedent. Every echoing refusal in the module is exercised here, by name."""
    sw = _sweep()
    checked = []

    def _refuses_clipped(label, marker, fn, *a, exc_type=SystemExit):
        """`marker` is a short prefix of the offending value: a clipped echo must still NAME the
        value it rejected (item 11's rule -- a real prefix, never a placeholder)."""
        with pytest.raises(exc_type) as e:
            fn(*a)
        msg = str(e.value)
        assert len(msg) < 20_000, f"{label}: refusal message is {len(msg)} chars -- not clipped"
        assert "CLIPPED" in msg, f"{label}: nothing was clipped"
        assert marker in msg, f"{label}: the value is no longer identifiably named"
        checked.append(label)

    # parse_fee_tiers -- 4 echoing refusals
    _refuses_clipped("fee: separators only", ",,,,", sw.parse_fee_tiers, "," * 900_000)
    _refuses_clipped("fee: not a number", "aaaa", sw.parse_fee_tiers, _HUGE)
    _refuses_clipped("fee: not finite", "aaaa", sw.parse_fee_tiers, "nan," + _HUGE + ",8")
    _refuses_clipped("fee: negative", "aaaa", sw.parse_fee_tiers, "-1," + _HUGE + ",8")
    # _parse_window -- 3 echoing refusals (the third, the INVERTED-window one, cannot carry a huge
    # echo at all: it is only reachable once BOTH bounds have parsed as ISO dates, so its echoes
    # are bounded by construction. Clipped anyway, for uniformity.)
    _refuses_clipped("window: not START,END", "aaaa", sw._parse_window, "--window-a", _HUGE)
    _refuses_clipped("window: not an ISO date", "aaaa", sw._parse_window, "--window-a",
                      _HUGE + ",x")
    # assert_db_path_allowed -- 2 echoing refusals (already clipped in cycle 1; re-pinned)
    _refuses_clipped("db-path: URI syntax", "aaaa",
                      lambda p: sw.assert_db_path_allowed(p, False), _HUGE + "?x=1")
    # run_sweep gate (iv) -- the attainable-tier subset refusal echoes two SEQUENCES, unbounded
    # through the public API.
    _refuses_clipped("gate iv: tier not in grid", "1.0,",
                      lambda: sw.run_sweep(attainable_maker_fee_bp=[float(i) for i in range(4000)],
                                            db_path="fixtures/trading_fixture.db"),
                      exc_type=sw.GateRefusal)
    assert len(checked) == 8, checked


def test_the_fee_tier_refusal_is_not_larger_than_its_own_input():
    """The specific measured regression: the echo used to MULTIPLY the input (1.8M chars out of a
    900k-char input, because `raw` is echoed twice per message)."""
    sw = _sweep()
    with pytest.raises(SystemExit) as e:
        sw.parse_fee_tiers(_HUGE)
    assert len(str(e.value)) < len(_HUGE) / 100


def test_window_refusal_is_not_larger_than_its_own_input():
    sw = _sweep()
    with pytest.raises(SystemExit) as e:
        sw._parse_window("--window-a", _HUGE + ",x")
    assert len(str(e.value)) < len(_HUGE) / 100


def test_the_refusal_and_defect_artifacts_clip_their_sequence_echoes(tmp_path):
    """Both failure artifacts echo caller-supplied SEQUENCES (tiers, windows, every grid axis),
    which are unbounded through the public API -- e.g. `run_sweep(grid_h=range(10**6))`."""
    sw = _sweep()
    big_axis = tuple(range(200_000))
    for name, writer, extra in (("refusal", sw.write_refusal_artifact, ()),
                                 ("defect", sw.write_internal_error_artifact,
                                  ("Traceback (most recent call last): ...",))):
        out = tmp_path / f"{name}.md"
        writer(out, "reason: " + _HUGE, *extra, _HUGE,
               [float(i) for i in range(50_000)],
               (("a", "2024-11-01", "2025-04-01"),), big_axis, (16,), ("long_only",), (0.75,), (0,))
        text = out.read_text()
        assert len(text) < 60_000, f"{name} artifact is {len(text)} chars -- echoes unclipped"
        assert "CLIPPED" in text


def test_clip_echo_cannot_raise_on_hostile_input():
    """A clipper that raises inside a refusal path turns a clean refusal into a traceback -- the
    whole failure class this module is closing. Non-str, None, bytes, a broken `__str__`, and lone
    SURROGATES (legal in a `str`, but `UnicodeEncodeError` the moment the message is written to a
    file or stderr) must all be survivable."""
    sw = _sweep()

    class _Hostile:
        def __str__(self):
            raise RuntimeError("no string for you")

    assert sw._clip_echo(None) == "None"
    assert sw._clip_echo(17) == "17"
    assert sw._clip_echo(b"bytes") == "b'bytes'"
    assert sw._clip_echo([1, 2, 3]) == "[1, 2, 3]"
    assert "unprintable _Hostile" in sw._clip_echo(_Hostile())
    assert sw._clip_echo(float("nan")) == "nan"
    # a lone surrogate survives clipping AND is safe to write out afterwards
    surrogate = "bad\udcff" + "a" * 5000
    clipped = sw._clip_echo(surrogate)
    assert "CLIPPED" in clipped
    clipped.encode("utf-8")          # would raise UnicodeEncodeError if surrogates survived
    # well-formed non-ASCII round-trips untouched
    assert sw._clip_echo("héllo — ✓") == "héllo — ✓"


# ------------------------------------------- MAJOR: a zero-cell run must be loud, not "healthy"

def test_a_zero_cell_run_is_loud_instead_of_reading_as_healthy(monkeypatch, tmp_path):
    """`run_sweep(grid_k=())` is API-reachable (not CLI-reachable -- the same surface item 2's
    second root cause lived on) and produced `n_cells_expected == n_cells_recorded ==
    n_cells_ok == 0` with `len(cells) == 0`. Every loud path missed it: gate (ii)'s completeness
    check is satisfied TRIVIALLY by `0 == 0`, `no_measurement` needs `bool(n_ok)` (False),
    `all_cells_failed` needs `n_cells_total > 0` (False), and `exit_code_for` returned 0. A
    degenerate run read as healthy from top to bottom. It must now carry its OWN banner and exit
    non-zero -- while a genuine HONEST EMPTY result (cells recorded, none measured: D58's
    protected case, the normal fixture state) still exits 0."""
    sw = _sweep()
    _stub_897_pipeline(monkeypatch, sw)
    cells, meta = sw.run_sweep(
        attainable_maker_fee_bp=[8.0], db_path="fixtures/trading_fixture.db",
        train_start="2024-01-01", val_start="2024-01-03", windows=_D60_SWEEP_WINDOWS,
        grid_h=(5,), grid_k=(), grid_variant=("long_only",), grid_tau=(0.75,), grid_n=(0,),
        grid_fee_bp=(8.0,), grid_adv_bp=(0.0,), n_boot=5,
    )
    # the degenerate state, reproduced exactly as reported
    assert len(cells) == 0
    assert meta["n_cells_expected"] == 0 and meta["n_cells_recorded"] == 0
    assert meta["n_cells_ok"] == 0
    # ...is now LOUD in both channels
    assert sw.exit_code_for(meta) != 0
    out = tmp_path / "zero.md"
    sw.write_artifact(cells, meta, out)
    text = out.read_text()
    assert "ZERO CELLS" in text
    assert "THE GRID ITSELF WAS EMPTY" in text
    assert "satisfied trivially by `0 == 0`" in text.replace("**", "")
    # and it is NOT conflated with either of the other two banners
    assert "EVERY CELL FAILED" not in text
    assert "NO DECISIONS IN THIS RUN" not in text


def test_exit_code_separates_a_zero_cell_run_from_an_honest_empty_one():
    """D58's line, drawn explicitly: cells recorded + none measured = HONEST EMPTY = exit 0 (a
    non-zero code there would fire on every fixture run and train operators to ignore it); NO
    cells at all = the instrument did nothing = non-zero."""
    sw = _sweep()
    assert sw.exit_code_for({"n_cells_expected": 144, "n_cells_recorded": 144,
                              "n_cells_ok": 0}) == 0        # honest empty -- unchanged
    assert sw.exit_code_for({"n_cells_expected": 0, "n_cells_recorded": 0,
                              "n_cells_ok": 0}) != 0        # zero-cell -- now loud
    assert sw.exit_code_for({"n_cells_expected": 144, "n_cells_recorded": 100}) == 1


def test_main_exits_nonzero_on_a_zero_cell_run(tmp_path, monkeypatch, capsys):
    """End-to-end through `main()`, and the artifact is still written (orchestration parses the
    artifact, not just the code)."""
    sw = _sweep()
    out = tmp_path / "zero.md"

    def fake_run_sweep(**kwargs):
        return [], _artifact_meta(n_cells=0)

    monkeypatch.setattr(sw, "run_sweep", fake_run_sweep)
    monkeypatch.setattr(sys, "argv", [
        "prog", "--attainable-maker-fee-bp", "8", "--db-path", "fixtures/trading_fixture.db",
        "--out", str(out), "--smoke"])
    with pytest.raises(SystemExit) as e:
        sw.main()
    assert e.value.code != 0
    assert "ZERO CELLS" in out.read_text()
    assert "ZERO CELLS" in capsys.readouterr().err


# -------------------------------------- MINOR 1: the protected-DB acknowledgement is keyword-only

def test_scheduled_real_data_run_is_keyword_only():
    """D57's bar is that the acknowledgement flag reflects EXPLICIT operator intent, never an
    accident. It was the 16th positional-or-keyword parameter of `run_sweep` -- and this very
    hardening row appended it there, which is exactly how a future parameter insertion above it
    would silently shift a positional argument into it and grant protected-DB access nobody asked
    for. It (and every other flag past the `*`) is keyword-only now."""
    import inspect
    sw = _sweep()
    params = inspect.signature(sw.run_sweep).parameters
    for name in ("scheduled_real_data_run", "verify_inherited_ic", "seed", "n_boot"):
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY, name
    # nothing positional can reach it: the positional tail ends at grid_adv_bp
    positional = [n for n, p in params.items()
                  if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD]
    assert positional[-1] == "grid_adv_bp"
    with pytest.raises(TypeError):
        # 12 positional args (the whole remaining tail) + 2 more: the 13th USED to land on `seed`
        # and one insertion above would have been enough for one to land on the acknowledgement
        # flag. Now every one of them is rejected.
        sw.run_sweep([8.0], "fixtures/trading_fixture.db", "2023-05-16", "2024-06-01",
                      sw.REAL_WINDOWS, (5,), (16,), ("long_only",), (0.75,), (0,), (8.0,), (0.0,),
                      897, True)


# ------------------------------- MINOR 3: the module-global monkeypatch non-reentrancy is stated

def test_the_monkeypatch_nonreentrancy_is_documented_at_every_patch_site_and_in_run_sweep():
    """Item 2c's `decide._decide_step` patch joins `decide.CFG`, `probe.sqlite3` and
    `backtest.SIGNAL_806_IC_FULL`; none is thread-safe, so a concurrent `run_sweep` (the 898-class
    reuse surface this instrument is pointed at) would interleave the save/restore pairs and could
    leave a module attribute permanently mispointed. Pre-existing in kind. This row DOCUMENTS it
    (a locking redesign is explicitly out of scope) -- pinned so the documentation cannot quietly
    disappear while the hazard stays."""
    sw = _sweep()
    doc = sw.run_sweep.__doc__ or ""
    assert "NOT THREAD-SAFE" in doc and "NOT REENTRANT" in doc
    for attr in ("decide.CFG", "probe.sqlite3", "backtest.SIGNAL_806_IC_FULL",
                 "decide._decide_step"):
        assert attr in doc, f"{attr} is not named in run_sweep's non-reentrancy docstring"
    src = Path(sw.__file__).read_text()
    # every patch site carries the warning LOCALLY too, not only in the docstring
    assert src.count("NOT REENTRANT (MINOR 3)") >= 5, src.count("NOT REENTRANT (MINOR 3)")


def test_the_fixture_cannot_cover_the_nonzero_decision_book_width_path():
    """The caveat a reader must not over-read past: the fixture universe is 7 assets and never
    clears `k=16`'s `2k` floor for `long_short`, so NO cell of the 144-cell SMOKE grid ever
    realizes `n_decisions > 0`. The gate-(ii) fixture rerun is therefore SILENT on
    `realized_book_width_mean` for nonzero-decision cells -- that path is covered only by a
    partly-stubbed unit test. Pinned at the field's own docstring so the disclosure ships with the
    code rather than living only in a verdict record."""
    sw = _sweep()
    field_doc = Path(sw.__file__).read_text()
    assert "THE FIXTURE CANNOT COVER THE NONZERO-DECISION PATH OF THIS FIELD" in field_doc
    assert "test_realized_book_width_is_recorded_and_the_instrumentation_cleans_up_after_itself" \
        in field_doc


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
