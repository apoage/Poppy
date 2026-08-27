"""TASK_911 (D69) -- startup check: the 806/808/809 lineage files this suite imports/reads must
match their pin in ``config/gitignored_pins.json``, checked BEFORE anything downstream trusts
their content.

Why this suite first (task scope item 2): ``recompute_best.py`` is the exact file D69 is about
-- experiments/897_corrected_809/test_recompute_gate.py imports it directly, and a silent
mtime-preserving restore reverted it to stale content on 2026-08-25 with no in-repo detector.
``probe.py``/``backtest.py``/``decide.py`` are the rest of the 806/808/809 lineage this suite's
own alias_read_layer.py, sweep_gated_maker.py and test_sweep_guards.py import or read-and-assert
against (see config/gitignored_pins.json's ``why_load_bearing`` for each).

Two states, never conflated (mirrors test_recompute_gate.py's own ``_LINEAGE.exists()``
skipif, which this file does NOT weaken or replace):

  * WHOLESALE ABSENCE -- none of the four lineage files exist at all (a fresh checkout that has
    never staged the gitignored ``runs/`` tree). This is the SAME, already-sanctioned condition
    test_recompute_gate.py / test_sweep_guards.py already skip on -- nothing to gate-test, and
    it is not the D69 defect (D69 was a lineage that IS present silently going stale, not a
    lineage that was never staged). Reported as a loud, explicit SKIP -- never silent.

  * ANYTHING ELSE -- the lineage directory tree is present (at least one file exists), so this
    host has real gitignored content and every pinned file in it is expected to be both PRESENT
    and byte-identical to its pin. A pinned file that's missing here is exactly as loud a
    refusal as one whose sha256 diverges (non-negotiable: "skip-with-silence nothing") -- this
    is the partial-corruption / D69 class this row exists to catch.
"""
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.verify_gitignored_pins import verify_all, refuse_on_divergence, PinVerificationError

# The 806/808/809 lineage files THIS suite depends on (see config/gitignored_pins.json for
# provenance/why_load_bearing on each). runs/875_wf/* is a separate lineage this suite never
# imports or reads -- out of scope for this suite's startup check per the task's scope fence.
_LINEAGE_PATHS = (
    "runs/806_predictability_probe/probe.py",
    "runs/808_linear_backtest/backtest.py",
    "runs/809_decision_rule_rescue/decide.py",
    "runs/809_decision_rule_rescue/recompute_best.py",
)


def _lineage_results():
    return verify_all(paths=list(_LINEAGE_PATHS), repo_root=_REPO)


def test_lineage_pins_startup_check():
    """The 897 suite's startup identity check (TASK_911 scope item 2). See module docstring for
    the wholesale-absence-vs-partial distinction."""
    results = _lineage_results()
    present = [r for r in results if r["status"] != "MISSING"]
    if not present:
        pytest.skip(
            "the gitignored 806/808/809 lineage is not staged on this host at all (0/"
            f"{len(results)} pinned files present) -- nothing to gate-test, same condition "
            "test_recompute_gate.py / test_sweep_guards.py already skip on. Not the D69 "
            "defect (a present-but-stale file), so this is a loud, explicit SKIP, not a "
            "silent one.")
    try:
        refuse_on_divergence(results, label="[897 startup check]")
    except PinVerificationError as e:
        pytest.fail(str(e), pytrace=False)
    unpinned = [r for r in results if r["status"] == "UNPINNED"]
    assert not unpinned, (
        f"lineage file(s) present with no recorded pin (should never happen for the 806/808/809 "
        f"lineage, which is seeded in full): {[r['path'] for r in unpinned]}")


def test_noconftest_bypass_stays_documented():
    """The startup gate lives in ``pytest_sessionstart``, so ``pytest --noconftest`` skips it
    entirely -- structural to any conftest-hosted gate, shared with the FIXTURE_IDENTITY
    precedent, and documented rather than fought (2026-08-27 correctness judge, MINOR 3). This
    test exists so the documentation cannot be quietly deleted, leaving a limit that is real but
    no longer stated, and so the unbypassable alternative stays named next to it."""
    src = (Path(__file__).resolve().parent / "conftest.py").read_text()
    assert "--noconftest" in src, (
        "conftest.py no longer documents the --noconftest bypass of its own startup gate")
    assert "python -m tools.verify_gitignored_pins" in src, (
        "conftest.py documents the bypass but no longer names the invocation that cannot be "
        "bypassed -- an honest limit needs its escape hatch named next to it")


def test_lineage_pins_manifest_has_all_four_files():
    """The manifest itself must still name all four 806/808/809 lineage files this suite
    depends on -- catches an accidental entry deletion from config/gitignored_pins.json, which
    would otherwise make the startup check above silently check fewer files."""
    from tools.verify_gitignored_pins import load_manifest
    manifest = load_manifest()
    pinned_paths = {e["path"] for e in manifest["pins"]}
    missing_entries = [p for p in _LINEAGE_PATHS if p not in pinned_paths]
    assert not missing_entries, (
        f"config/gitignored_pins.json no longer pins: {missing_entries}")
