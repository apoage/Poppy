#!/usr/bin/env python3
"""TASK_897 — the sweep harness's CLI refusals and the artifact's honesty claims.

Three things are pinned here, all of them fix-pass findings:

  * the artifact must NOT claim the 877 XBT/BTC alias fold is applied — it is not, and the
    claim was verified false against the 806/808/809 lineage's own source;
  * the artifact must state PROMINENTLY that its CI is UNCALIBRATED, so the pre-registered
    band's "CI lo > 0" criterion cannot be read off it;
  * `--db-path` must not be able to reach the protected 24GB `trading_data.db` BY OMISSION,
    and every CLI refusal must be a clean message rather than a raw traceback.
"""
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


@pytest.mark.parametrize("raw,needle", [
    (None, "required and has no default"),
    ("", "required and has no default"),
    ("   ", "required and has no default"),
    (",", "contains no tier values"),
    (" , , ", "contains no tier values"),
    ("abc", "is not a number"),
    ("8,abc", "is not a number"),
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

def _artifact_text(tmp_path):
    sw = _sweep()
    meta = {
        "attainable_maker_fee_bp_tiers": [8.0, 16.0], "db_path": "fixtures/trading_fixture.db",
        "bar_sec": 60, "n_assets": 3, "symbols": ["A", "B", "C"],
        "train_start": "2023-05-16", "val_start": "2024-06-01",
        "windows": [{"name": "a", "start": "x", "end": "y"}],
        "grid": {"H": [5]}, "grid_epoch": {"start": 0, "end": 1},
        "grid_start_iso": "1970-01-01T00:00:00Z", "grid_end_iso": "1970-01-01T00:00:01Z",
        "seed": 897, "n_boot": 10, "n_cells_expected": 0, "n_cells_recorded": 0,
        "n_cells_ok": 0, "n_cells_dropped_or_failed": 0, "runtime_s": 0.0,
        "estimator": "per-decision", "maker_fill_model": "OHLC-touch",
    }
    out = tmp_path / "result.md"
    sw.write_artifact([], meta, out)
    return out.read_text()


def test_artifact_does_not_claim_the_877_alias_fold_is_applied(tmp_path):
    """MAJOR: the metadata asserted the fold "applies transitively via
    `probe.load_universe_trainlocked`/`build_aligned_ohlcv` (inherited, never reimplemented
    here)". That is false — the lineage has zero alias handling and predates TASK_877. The
    artifact must now say so instead."""
    text = _artifact_text(tmp_path)
    assert "877 alias fold: NOT APPLIED by this lineage" in text
    assert "gate (iii) is NOT met on this point" in text
    assert "applies transitively" not in text.split("An earlier revision")[0]
    assert "SELECT id, symbol FROM assets" in text     # the verified mechanism, named


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
                f"{rel} now references {n!r} — the artifact's 'alias fold NOT applied' "
                "statement may no longer be accurate; re-verify before changing it."
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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
