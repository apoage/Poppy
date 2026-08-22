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
    # ...and is reachable only when explicitly named AND acknowledged.
    assert sw.assert_db_path_allowed(real, True) == real
    # a relative spelling of the same file resolves to the same refusal
    with pytest.raises(SystemExit):
        sw.assert_db_path_allowed("./trading_data.db", False)


def test_fixture_db_path_is_allowed():
    sw = _sweep()
    p = str(_REPO / "fixtures" / "trading_fixture.db")
    assert sw.assert_db_path_allowed(p, False) == p


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
    assert sw.assert_db_path_allowed("trading_data.db", True) == "trading_data.db"


def test_relative_fixture_db_path_is_still_allowed_from_a_non_repo_root_cwd(tmp_path, monkeypatch):
    """An honest relative path — protected under NEITHER interpretation — must not be caught by
    the widened guard (no honest path may start refusing as a side effect of this fix)."""
    import os
    sw = _sweep()
    monkeypatch.chdir(tmp_path)
    rel = os.path.relpath(_REPO / "fixtures" / "trading_fixture.db", tmp_path)
    assert sw.assert_db_path_allowed(rel, False) == rel


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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
