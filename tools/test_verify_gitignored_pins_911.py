"""TASK_911 (D69) -- tests for tools/verify_gitignored_pins.py, the identity-pin verifier for
load-bearing gitignored artifacts.

Covers: hashing, manifest loading (fail-closed on missing/corrupt/incomplete), the four
per-file verify_pin() statuses (OK/DIVERGED/MISSING/UNPINNED), refuse_on_divergence()'s
raise/no-raise boundary, promote_pin()'s evidence-required discipline, the CLI's exit-code
contract (0/1/2), and a sanity check against the REAL shipped config/gitignored_pins.json.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.verify_gitignored_pins import (
    sha256_of, load_manifest, verify_pin, verify_all, refuse_on_divergence, promote_pin, main,
    PinVerificationError, DEFAULT_MANIFEST,
)


def _write_manifest(tmp_path, pins):
    p = tmp_path / "pins.json"
    p.write_text(json.dumps({"pins": pins}))
    return p


def _write_file(tmp_path, rel, content=b"hello world\n"):
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content)
    return p


def _entry(path, sha256=None, **extra):
    e = {"path": path, "sha256": sha256, "provenance": "test", "why_load_bearing": "test"}
    e.update(extra)
    return e


# --------------------------------------------------------------------------------------------
# sha256_of
# --------------------------------------------------------------------------------------------
def test_sha256_of_matches_hashlib(tmp_path):
    p = _write_file(tmp_path, "f.txt", b"some bytes\n" * 100)
    assert sha256_of(p) == hashlib.sha256(b"some bytes\n" * 100).hexdigest()


def test_sha256_of_large_file_chunked(tmp_path):
    """Exercise the chunked read path (chunk_size default 1MB) with content that spans several
    chunks, to catch an off-by-one in the read loop."""
    content = (b"x" * 1000) * 5000  # 5,000,000 bytes
    p = _write_file(tmp_path, "big.bin", content)
    assert sha256_of(p, chunk_size=1024) == hashlib.sha256(content).hexdigest()


# --------------------------------------------------------------------------------------------
# load_manifest -- fail-closed
# --------------------------------------------------------------------------------------------
def test_load_manifest_missing_file_refuses(tmp_path):
    with pytest.raises(PinVerificationError, match="missing"):
        load_manifest(tmp_path / "nope.json")


def test_load_manifest_corrupt_json_refuses(tmp_path):
    p = tmp_path / "pins.json"
    p.write_text("{not json")
    with pytest.raises(PinVerificationError, match="unreadable/corrupt"):
        load_manifest(p)


def test_load_manifest_no_pins_key_refuses(tmp_path):
    p = tmp_path / "pins.json"
    p.write_text(json.dumps({"_meta": {}}))
    with pytest.raises(PinVerificationError, match="no 'pins' list"):
        load_manifest(p)


def test_load_manifest_entry_missing_required_field_refuses(tmp_path):
    p = _write_manifest(tmp_path, [{"path": "x", "sha256": "abc"}])  # no provenance/why
    with pytest.raises(PinVerificationError, match="missing field"):
        load_manifest(p)


def test_load_manifest_valid_roundtrip(tmp_path):
    p = _write_manifest(tmp_path, [_entry("x", "abc")])
    m = load_manifest(p)
    assert m["pins"][0]["path"] == "x"


# --------------------------------------------------------------------------------------------
# verify_pin -- the four statuses
# --------------------------------------------------------------------------------------------
def test_verify_pin_ok(tmp_path):
    p = _write_file(tmp_path, "runs/foo.py", b"content")
    entry = _entry("runs/foo.py", sha256_of(p))
    r = verify_pin(entry, repo_root=tmp_path)
    assert r["status"] == "OK"
    assert r["expected"] == r["actual"]


def test_verify_pin_diverged_names_expected_and_actual(tmp_path):
    p = _write_file(tmp_path, "runs/foo.py", b"NEW stale-reverted content")
    entry = _entry("runs/foo.py", "0" * 64)  # a hash that cannot match
    r = verify_pin(entry, repo_root=tmp_path)
    assert r["status"] == "DIVERGED"
    assert r["expected"] == "0" * 64
    assert r["actual"] == sha256_of(p)
    assert r["expected"] in r["message"] and r["actual"] in r["message"]


def test_verify_pin_missing_is_never_silent(tmp_path):
    """The D69/orch_wf875 class: a pinned file simply isn't there. Must be its own named
    status, distinct from DIVERGED, and must state the expected hash in the message."""
    entry = _entry("runs/never_created.py", "a" * 64)
    r = verify_pin(entry, repo_root=tmp_path)
    assert r["status"] == "MISSING"
    assert r["actual"] is None
    assert "a" * 64 in r["message"]
    assert "MISSING" in r["message"]


def test_verify_pin_missing_with_no_recorded_hash_still_names_expected_as_none(tmp_path):
    """orch_wf875.sh's exact shape: absent AND sha256 is null in the manifest. Must still
    produce a MISSING status (never silently OK, never silently skipped)."""
    entry = _entry("runs/875_wf/orch_wf875.sh", sha256=None)
    r = verify_pin(entry, repo_root=tmp_path)
    assert r["status"] == "MISSING"
    assert "none ever recorded" in r["message"]


def test_verify_pin_unpinned_present_but_no_hash_recorded(tmp_path):
    _write_file(tmp_path, "runs/new_thing.py", b"stuff")
    entry = _entry("runs/new_thing.py", sha256=None)
    r = verify_pin(entry, repo_root=tmp_path)
    assert r["status"] == "UNPINNED"
    assert r["actual"] is not None


# --------------------------------------------------------------------------------------------
# verify_all
# --------------------------------------------------------------------------------------------
def test_verify_all_checks_every_pin_in_order(tmp_path):
    p1 = _write_file(tmp_path, "a.py", b"aaa")
    p2 = _write_file(tmp_path, "b.py", b"bbb")
    manifest = _write_manifest(tmp_path, [
        _entry("a.py", sha256_of(p1)), _entry("b.py", sha256_of(p2)),
    ])
    results = verify_all(manifest_path=manifest, repo_root=tmp_path)
    assert [r["path"] for r in results] == ["a.py", "b.py"]
    assert all(r["status"] == "OK" for r in results)


def test_verify_all_paths_subset(tmp_path):
    p1 = _write_file(tmp_path, "a.py", b"aaa")
    p2 = _write_file(tmp_path, "b.py", b"bbb")
    manifest = _write_manifest(tmp_path, [
        _entry("a.py", sha256_of(p1)), _entry("b.py", sha256_of(p2)),
    ])
    results = verify_all(manifest_path=manifest, repo_root=tmp_path, paths=["b.py"])
    assert [r["path"] for r in results] == ["b.py"]


def test_verify_all_unknown_path_refuses(tmp_path):
    manifest = _write_manifest(tmp_path, [_entry("a.py", "x" * 64)])
    with pytest.raises(PinVerificationError, match="not present"):
        verify_all(manifest_path=manifest, repo_root=tmp_path, paths=["nonexistent.py"])


# --------------------------------------------------------------------------------------------
# refuse_on_divergence
# --------------------------------------------------------------------------------------------
def test_refuse_on_divergence_raises_on_diverged():
    results = [{"path": "x", "status": "DIVERGED", "message": "DIVERGED: x"}]
    with pytest.raises(PinVerificationError, match="REFUSED"):
        refuse_on_divergence(results)


def test_refuse_on_divergence_raises_on_missing():
    results = [{"path": "x", "status": "MISSING", "message": "MISSING: x"}]
    with pytest.raises(PinVerificationError, match="REFUSED"):
        refuse_on_divergence(results)


def test_refuse_on_divergence_does_not_raise_on_ok_or_unpinned():
    results = [
        {"path": "x", "status": "OK", "message": "OK: x"},
        {"path": "y", "status": "UNPINNED", "message": "UNPINNED: y"},
    ]
    refuse_on_divergence(results)  # must not raise


# --------------------------------------------------------------------------------------------
# promote_pin -- deliberate-only re-pin
# --------------------------------------------------------------------------------------------
def test_promote_pin_refuses_without_evidence(tmp_path):
    p = _write_file(tmp_path, "runs/foo.py", b"content")
    manifest = _write_manifest(tmp_path, [])
    with pytest.raises(PinVerificationError, match="evidence"):
        promote_pin("runs/foo.py", "prov", "why", evidence="", manifest_path=manifest,
                     repo_root=tmp_path)
    with pytest.raises(PinVerificationError, match="evidence"):
        promote_pin("runs/foo.py", "prov", "why", evidence=None, manifest_path=manifest,
                     repo_root=tmp_path)


def test_promote_pin_refuses_on_absent_file(tmp_path):
    manifest = _write_manifest(tmp_path, [])
    with pytest.raises(PinVerificationError, match="does not exist"):
        promote_pin("runs/never_made.py", "prov", "why", evidence="re-derived by hand",
                     manifest_path=manifest, repo_root=tmp_path)


def test_promote_pin_creates_new_entry(tmp_path):
    p = _write_file(tmp_path, "runs/foo.py", b"content")
    manifest = _write_manifest(tmp_path, [])
    promote_pin("runs/foo.py", "TASK_X", "gates the thing", evidence="measured by hand",
                manifest_path=manifest, repo_root=tmp_path)
    m = json.loads(manifest.read_text())
    assert len(m["pins"]) == 1
    assert m["pins"][0]["sha256"] == sha256_of(p)
    assert m["pins"][0]["promotion_evidence"] == "measured by hand"


def test_promote_pin_updates_existing_entry_never_silent_about_old_value(tmp_path):
    p = _write_file(tmp_path, "runs/foo.py", b"content v2")
    manifest = _write_manifest(tmp_path, [_entry("runs/foo.py", "deadbeef" * 8)])
    msg = promote_pin("runs/foo.py", "TASK_X", "gates the thing", evidence="re-derived, checked",
                       manifest_path=manifest, repo_root=tmp_path)
    assert "deadbeef" * 8 in msg  # old value named, not silently dropped
    m = json.loads(manifest.read_text())
    assert m["pins"][0]["sha256"] == sha256_of(p)


# --------------------------------------------------------------------------------------------
# CLI (main()) exit-code contract: 0 = all OK, 1 = DIVERGED/MISSING present, 2 = only UNPINNED
# --------------------------------------------------------------------------------------------
def test_cli_exit_0_when_all_ok(tmp_path, capsys):
    p = _write_file(tmp_path, "a.py", b"aaa")
    manifest = _write_manifest(tmp_path, [_entry("a.py", sha256_of(p))])
    rc = main(["--manifest", str(manifest), "--repo-root", str(tmp_path), "--path", "a.py"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "OK" in captured.out


def test_cli_exit_1_on_diverged(tmp_path, capsys):
    _write_file(tmp_path, "a.py", b"aaa")
    manifest = _write_manifest(tmp_path, [_entry("a.py", "0" * 64)])
    rc = main(["--manifest", str(manifest), "--repo-root", str(tmp_path)])
    assert rc == 1
    captured = capsys.readouterr()
    assert "DIVERGED" in captured.out
    assert "REFUSED" in captured.err


def test_cli_exit_1_on_missing_file(tmp_path, capsys):
    manifest = _write_manifest(tmp_path, [_entry("never_created.py", "a" * 64)])
    rc = main(["--manifest", str(manifest), "--repo-root", str(tmp_path)])
    assert rc == 1
    captured = capsys.readouterr()
    assert "MISSING" in captured.out
    assert "REFUSED" in captured.err


def test_cli_exit_2_on_unpinned_only(tmp_path, capsys):
    _write_file(tmp_path, "a.py", b"aaa")
    manifest = _write_manifest(tmp_path, [_entry("a.py", sha256=None)])
    rc = main(["--manifest", str(manifest), "--repo-root", str(tmp_path)])
    assert rc == 2
    captured = capsys.readouterr()
    assert "UNPINNED" in captured.out
    assert "INCOMPLETE" in captured.err


def test_cli_exit_1_on_missing_manifest(capsys):
    rc = main(["--manifest", "/nonexistent/path/pins.json"])
    assert rc == 1
    captured = capsys.readouterr()
    assert "missing" in captured.err.lower()


def test_cli_promote_requires_evidence(tmp_path, capsys):
    p = _write_file(tmp_path, "a.py", b"aaa")
    manifest = _write_manifest(tmp_path, [])
    rc = main(["--manifest", str(manifest), "--promote-pin", "a.py",
               "--provenance", "p", "--why", "w"])
    assert rc == 1
    captured = capsys.readouterr()
    assert "evidence" in captured.err.lower()


# --------------------------------------------------------------------------------------------
# MAJOR (2026-08-27 correctness judge): the promote path must not be able to launder a re-pin
# with no audit trail. Two halves, both required -- refuse the WRITE (missing/falsy
# --provenance/--why), and catch an already-written null entry on the READ, so a laundered
# entry cannot live forever just because its keys exist.
# --------------------------------------------------------------------------------------------
def test_promote_pin_refuses_without_provenance(tmp_path):
    _write_file(tmp_path, "runs/foo.py", b"content")
    manifest = _write_manifest(tmp_path, [])
    for bad in (None, "", "   "):
        with pytest.raises(PinVerificationError, match="provenance"):
            promote_pin("runs/foo.py", bad, "why", evidence="re-derived, checked",
                        manifest_path=manifest, repo_root=tmp_path)
    assert json.loads(manifest.read_text())["pins"] == []  # nothing written


def test_promote_pin_refuses_without_why(tmp_path):
    _write_file(tmp_path, "runs/foo.py", b"content")
    manifest = _write_manifest(tmp_path, [])
    for bad in (None, "", "   "):
        with pytest.raises(PinVerificationError, match="why"):
            promote_pin("runs/foo.py", "prov", bad, evidence="re-derived, checked",
                        manifest_path=manifest, repo_root=tmp_path)
    assert json.loads(manifest.read_text())["pins"] == []  # nothing written


def test_cli_promote_without_provenance_or_why_cannot_launder_a_repin(tmp_path, capsys):
    """The judge's exact reproduction: `--promote-pin runs/foo/bar.py --evidence '...'` with no
    --provenance/--why previously exited 0, printed RE-PINNED, and wrote null into both audit
    fields. It must now exit 1 and leave the manifest byte-identical."""
    _write_file(tmp_path, "runs/foo/bar.py", b"content")
    manifest = _write_manifest(tmp_path, [_entry("runs/foo/bar.py", "0" * 64)])
    before = manifest.read_bytes()
    rc = main(["--manifest", str(manifest), "--repo-root", str(tmp_path),
               "--promote-pin", "runs/foo/bar.py",
               "--evidence", "just evidence, no provenance/why"])
    assert rc == 1
    assert manifest.read_bytes() == before, "a refused promote must not touch the manifest"
    err = capsys.readouterr().err
    assert "--provenance" in err


def test_cli_promote_with_full_audit_trail_succeeds(tmp_path, capsys):
    """The refusal above must not have broken the legitimate path."""
    p = _write_file(tmp_path, "runs/foo/bar.py", b"content")
    manifest = _write_manifest(tmp_path, [_entry("runs/foo/bar.py", "0" * 64)])
    rc = main(["--manifest", str(manifest), "--repo-root", str(tmp_path),
               "--promote-pin", "runs/foo/bar.py", "--provenance", "TASK_X re-derivation",
               "--why", "gates the 897 suite", "--evidence", "re-ran the deriving script"])
    assert rc == 0
    assert "RE-PINNED" in capsys.readouterr().out
    entry = json.loads(manifest.read_text())["pins"][0]
    assert entry["sha256"] == sha256_of(p)
    assert entry["provenance"] and entry["why_load_bearing"]


@pytest.mark.parametrize("field", ["provenance", "why_load_bearing"])
@pytest.mark.parametrize("value", [None, "", "   "])
def test_load_manifest_catches_an_already_laundered_entry_on_read(tmp_path, field, value):
    """The second half of the MAJOR: key PRESENCE is not enough. An entry already carrying a
    null/blank audit field -- e.g. one written by the old promote path before it was fixed --
    must be refused when the manifest is READ, not tolerated forever."""
    entry = {"path": "runs/foo/bar.py", "sha256": "0" * 64, "provenance": "p",
             "why_load_bearing": "w", "promotion_evidence": "just evidence"}
    entry[field] = value
    p = _write_manifest(tmp_path, [entry])
    with pytest.raises(PinVerificationError, match=field):
        load_manifest(p)


def test_load_manifest_still_allows_null_sha256(tmp_path):
    """orch_wf875.sh's shape: sha256 null is a DOCUMENTED value (no known-good hash was ever
    recorded) and must keep loading -- the truthiness rule applies to the audit fields only."""
    p = _write_manifest(tmp_path, [_entry("runs/875_wf/orch_wf875.sh", sha256=None)])
    assert load_manifest(p)["pins"][0]["sha256"] is None


def test_load_manifest_refuses_blank_sha256(tmp_path):
    """...but an empty-string sha256 is not "unrecorded", it is a broken pin."""
    p = _write_manifest(tmp_path, [_entry("runs/foo.py", sha256="")])
    with pytest.raises(PinVerificationError, match="sha256"):
        load_manifest(p)


def test_cli_exit_1_on_laundered_manifest_entry(tmp_path, capsys):
    """The read-side refusal reaches the CLI too (so the gate-(iii) entry point catches it)."""
    _write_file(tmp_path, "a.py", b"aaa")
    p = _write_manifest(tmp_path, [{"path": "a.py", "sha256": "0" * 64, "provenance": None,
                                    "why_load_bearing": None}])
    rc = main(["--manifest", str(p), "--repo-root", str(tmp_path)])
    assert rc == 1
    assert "provenance" in capsys.readouterr().err


# --------------------------------------------------------------------------------------------
# MINOR 1 (same judge): sha256_of must not make verify_pin raise. verify_pin's contract is a
# structured result for every ordinary outcome -- an uncaught OSError inside
# pytest_sessionstart surfaces as an INTERNALERROR, losing the legible refusal exactly when it
# matters most.
# --------------------------------------------------------------------------------------------
def test_verify_pin_on_a_directory_returns_structured_result(tmp_path):
    (tmp_path / "runs" / "a_dir").mkdir(parents=True)
    r = verify_pin(_entry("runs/a_dir", "a" * 64), repo_root=tmp_path)  # must NOT raise
    assert r["status"] == "DIVERGED"
    assert r["actual"] is None
    assert "IsADirectoryError" in r["message"] or "Is a directory" in r["message"]


@pytest.mark.skipif(getattr(os, "geteuid", lambda: 1)() == 0,
                    reason="root ignores the permission bits this test relies on")
def test_verify_pin_on_an_unreadable_file_returns_structured_result(tmp_path):
    p = _write_file(tmp_path, "runs/locked.py", b"content")
    p.chmod(0o000)
    try:
        r = verify_pin(_entry("runs/locked.py", "a" * 64), repo_root=tmp_path)  # must NOT raise
    finally:
        p.chmod(0o644)
    assert r["status"] == "DIVERGED"
    assert r["actual"] is None
    assert "PermissionError" in r["message"] or "Permission denied" in r["message"]


def test_cli_exit_1_on_an_unhashable_pin(tmp_path, capsys):
    (tmp_path / "runs" / "a_dir").mkdir(parents=True)
    manifest = _write_manifest(tmp_path, [_entry("runs/a_dir", "a" * 64)])
    rc = main(["--manifest", str(manifest), "--repo-root", str(tmp_path)])
    assert rc == 1
    assert "REFUSED" in capsys.readouterr().err


# --------------------------------------------------------------------------------------------
# MINOR 2 (same judge): a manifest path must resolve INSIDE repo_root. Containment is checked
# on realpath of BOTH sides (TASK_886/888 realpath-vs-abspath history; same shape as TASK_905's
# _require_within_fold_root), and the refusal happens BEFORE any bytes are read.
# --------------------------------------------------------------------------------------------
def test_verify_pin_refuses_a_dotdot_escape(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "path_test_outside"
    outside.mkdir()
    (outside / "secret.txt").write_bytes(b"secret")
    entry = _entry("../path_test_outside/secret.txt",
                   sha256_of(outside / "secret.txt"))  # would have verified OK before the fix
    r = verify_pin(entry, repo_root=repo)
    assert r["status"] == "DIVERGED"
    assert "OUTSIDE" in r["message"]
    assert str(repo) in r["message"]


def test_verify_pin_refuses_a_symlink_escape(tmp_path):
    """A literal `..` is the easy case; a symlink planted inside the tree is the one a
    startswith-on-unnormalized-strings check misses."""
    repo = tmp_path / "repo"
    (repo / "runs").mkdir(parents=True)
    outside = tmp_path / "path_test_outside"
    outside.mkdir()
    (outside / "secret.txt").write_bytes(b"secret")
    (repo / "runs" / "link.txt").symlink_to(outside / "secret.txt")
    entry = _entry("runs/link.txt", sha256_of(outside / "secret.txt"))
    r = verify_pin(entry, repo_root=repo)
    assert r["status"] == "DIVERGED"
    assert "OUTSIDE" in r["message"]


def test_verify_pin_refuses_an_absolute_path_entry(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "path_test_outside"
    outside.mkdir()
    (outside / "secret.txt").write_bytes(b"secret")
    r = verify_pin(_entry(str(outside / "secret.txt"), "a" * 64), repo_root=repo)
    assert r["status"] == "DIVERGED"
    assert "OUTSIDE" in r["message"]


def test_verify_pin_escape_refusal_beats_absence(tmp_path):
    """An escaping path that doesn't exist must refuse as an escape, not report a bland
    MISSING -- the manifest defect is the finding."""
    repo = tmp_path / "repo"
    repo.mkdir()
    r = verify_pin(_entry("../nowhere/nothing.txt", "a" * 64), repo_root=repo)
    assert r["status"] == "DIVERGED"
    assert "OUTSIDE" in r["message"]


def test_promote_pin_refuses_an_escaping_path(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "path_test_outside"
    outside.mkdir()
    (outside / "secret.txt").write_bytes(b"secret")
    manifest = _write_manifest(tmp_path, [])
    with pytest.raises(PinVerificationError, match="OUTSIDE"):
        promote_pin("../path_test_outside/secret.txt", "prov", "why", evidence="ev",
                    manifest_path=manifest, repo_root=repo)
    assert json.loads(manifest.read_text())["pins"] == []


def test_verify_pin_allows_a_normal_nested_path(tmp_path):
    """Containment must not become a false refusal for ordinary entries."""
    p = _write_file(tmp_path, "runs/809_decision_rule_rescue/recompute_best.py", b"x")
    r = verify_pin(_entry("runs/809_decision_rule_rescue/recompute_best.py", sha256_of(p)),
                   repo_root=tmp_path)
    assert r["status"] == "OK"


# --------------------------------------------------------------------------------------------
# Sanity against the REAL shipped manifest -- catches a broken/incomplete
# config/gitignored_pins.json before it reaches a suite.
# --------------------------------------------------------------------------------------------
def test_real_manifest_every_pin_resolves_inside_the_repo():
    """No shipped entry may name a path outside the repo root (MINOR 2's standing guard)."""
    repo = Path(__file__).resolve().parents[1]
    for e in load_manifest(DEFAULT_MANIFEST)["pins"]:
        r = verify_pin(e, repo_root=repo)
        assert "OUTSIDE" not in r["message"], f"pin escapes the repo root: {e['path']}"


def test_real_manifest_audit_fields_are_all_non_empty():
    """Every shipped entry carries a real provenance + why_load_bearing (the MAJOR's subject),
    which load_manifest now enforces -- asserted explicitly so the intent is visible."""
    for e in load_manifest(DEFAULT_MANIFEST)["pins"]:
        assert e["provenance"].strip(), e["path"]
        assert e["why_load_bearing"].strip(), e["path"]



def test_real_manifest_loads_and_has_required_fields():
    manifest = load_manifest(DEFAULT_MANIFEST)
    assert len(manifest["pins"]) >= 6
    paths = {e["path"] for e in manifest["pins"]}
    for expected in (
        "runs/806_predictability_probe/probe.py",
        "runs/808_linear_backtest/backtest.py",
        "runs/809_decision_rule_rescue/decide.py",
        "runs/809_decision_rule_rescue/recompute_best.py",
        "runs/875_wf/grade_correct.py",
        "runs/875_wf/orch_wf875.sh",
    ):
        assert expected in paths, f"expected pin missing from real manifest: {expected}"


def test_real_manifest_recompute_best_pin_matches_the_D69_repaired_value():
    """The exact value the task handed off as the repaired, currently-deployed content."""
    manifest = load_manifest(DEFAULT_MANIFEST)
    entry = next(e for e in manifest["pins"]
                 if e["path"] == "runs/809_decision_rule_rescue/recompute_best.py")
    assert entry["sha256"] == (
        "783b2ea35754d057bed2509b6ea3ffa1f3022473eb9247c706cd0091f4931ffc")
