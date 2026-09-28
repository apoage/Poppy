"""TASK_922 G1/G2/G3. All run trees are temporary; CLI and in-run calls are dry runs.

Run on the approved verification host, not the workstation. The G1 positive control
really authorizes via SPEC.md + experiment covers before each refusal is planted.
The existing 881E suite remains unchanged and retains its consent positive controls.
"""

import ast
from datetime import date
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import mock_open

import pytest

REPO = Path(__file__).resolve().parents[2]
GC_PATH = REPO / "tools" / "ckpt_gc.py"
WRITER_PATH = REPO / "mcts_trader" / "training" / "run_declaration.py"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gc = _load("ckpt_gc_922", GC_PATH)
writer = _load("run_declaration_922", WRITER_PATH)


def _authorized_tree(tmp_path, relative="target_run"):
    run = tmp_path / "runs" / relative
    for epoch in (1, 2, 3):
        ckpt = run / f"checkpoint_{epoch}"
        ckpt.mkdir(parents=True)
        (ckpt / "checkpoint").write_bytes(b"fixture")
        os.utime(ckpt, (epoch, epoch))
    owner = tmp_path / "experiments" / "owner"
    owner.mkdir(parents=True)
    (owner / "PROTECTED.txt").write_text(f"covers: {run}\n", encoding="utf-8")
    (run / "SPEC.md").write_text(f"covers:\n- {owner}\n", encoding="utf-8")
    return run


def _bound(run, roots):
    manifests = gc.load_manifests([str(run.parents[1] / "experiments")])
    return gc.resolve_bound_coverage(
        [str(run)], manifests.protect | manifests.covers, manifests.dangling,
        citations=manifests.citations, roots=[str(r) for r in roots])[str(run)]


def _cli(tmp_path):
    result = subprocess.run(
        [sys.executable, str(GC_PATH), "--roots", "runs", "--experiments-dir", "experiments",
         "--min-free-gb", "1e18", "--keep-unscored", "0", "--keep-latest", "1"],
        cwd=tmp_path, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout + result.stderr


def _assert_cli_authorized(tmp_path):
    output = _cli(tmp_path)
    assert "REFUSE run dir" not in output, output
    assert "would delete" in output, output  # Not a vacuous no-candidates green.


@pytest.mark.parametrize("marker", ["", "# name: target_run\n# reason: retain\n"])
def test_g1_cli_self_declaration_refuses_otherwise_authorized_run(tmp_path, marker):
    run = _authorized_tree(tmp_path)
    _assert_cli_authorized(tmp_path)
    (run / "PROTECTED.txt").write_text(marker, encoding="utf-8")
    output = _cli(tmp_path)
    assert "REFUSE run dir" in output and "self-protected" in output, output
    assert "would delete" not in output, output
    assert len(list(run.glob("checkpoint_*"))) == 3
    (run / "PROTECTED.txt").unlink()
    _assert_cli_authorized(tmp_path)


@pytest.mark.parametrize("marker, diagnostic", [
    (b"not a directive\n", "malformed"),
    (b"covers: experiments/owner\ninvalid\n", "malformed"),
    (b"\xff", "UTF-8"),
    (b"\xef\xbb\xbf# name: target\n", "BYTE-ORDER MARK"),
    (b"# name: target\x00\n", "NUL"),
])
def test_g1_cli_malformed_marker_overrides_real_spec_consent(tmp_path, marker, diagnostic):
    run = _authorized_tree(tmp_path)
    _assert_cli_authorized(tmp_path)
    (run / "PROTECTED.txt").write_bytes(marker)
    output = _cli(tmp_path)
    assert "REFUSE run dir" in output and diagnostic in output, output
    assert str(run / "PROTECTED.txt") in output, output
    assert "would delete" not in output, output


@pytest.mark.parametrize("shape, diagnostic", [
    ("directory", "DIRECTORY"), ("fifo", "not a regular file"),
    ("dangling", "does not resolve"), ("file_link", "SYMLINK"),
    ("directory_link", "SYMLINK TO A DIRECTORY"),
])
def test_g1_wrong_shapes_refuse_without_opening(tmp_path, monkeypatch, shape, diagnostic):
    run = _authorized_tree(tmp_path)
    roots = [tmp_path / "runs"]
    assert _bound(run, roots)
    marker = run / "PROTECTED.txt"
    if shape == "directory":
        marker.mkdir()
    elif shape == "fifo":
        os.mkfifo(marker)
    elif shape == "dangling":
        marker.symlink_to(run / "missing")
    elif shape == "file_link":
        marker.symlink_to(run / "SPEC.md")
    else:
        marker.symlink_to(run / "checkpoint_1", target_is_directory=True)
    original = gc._read_manifest_text

    def guarded_read(path):
        assert str(path) != str(marker), "nonregular marker must never be opened"
        return original(path)

    monkeypatch.setattr(gc, "_read_manifest_text", guarded_read)
    assert not _bound(run, roots)
    hint = gc.authority_binding_refusal_hint(str(run), list(map(str, roots)), set(), {})
    assert diagnostic in hint and str(marker) in hint
    output = _cli(tmp_path)  # Timeout also catches an accidental blocking FIFO open.
    assert "REFUSE run dir" in output and diagnostic in output, output
    assert "would delete" not in output


def test_g1_unreadable_marker_refuses_with_diagnostic(tmp_path, monkeypatch):
    run = _authorized_tree(tmp_path)
    roots = [str(tmp_path / "runs")]
    assert _bound(run, roots)
    marker = run / "PROTECTED.txt"
    marker.write_text("# retained\n", encoding="utf-8")
    original_open = open

    def denied(path, *args, **kwargs):
        if str(path) == str(marker):
            raise PermissionError("922 read denied")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", denied)
    assert not _bound(run, roots)
    hint = gc.authority_binding_refusal_hint(str(run), roots, set(), {})
    assert "REFUSED" in hint and "922 read denied" in hint and str(marker) in hint


@pytest.mark.parametrize("marker", ["# retain nested run\n", "malformed declaration\n"])
def test_g1_nested_ancestor_and_deepest_root_boundary(tmp_path, marker):
    run = _authorized_tree(tmp_path, "group/inner/leaf")
    outer = tmp_path / "runs"
    inner = outer / "group" / "inner"
    _assert_cli_authorized(tmp_path)
    declaration = outer / "group" / "PROTECTED.txt"
    declaration.write_text(marker, encoding="utf-8")
    assert "REFUSE run dir" in _cli(tmp_path)
    manifests = gc.load_manifests([str(tmp_path / "experiments")])

    def bound(roots):
        return gc.resolve_bound_coverage(
            [str(run)], manifests.covers, citations=manifests.citations,
            roots=list(map(str, roots)))[str(run)]

    assert not bound([outer])
    assert bound([outer, inner])  # Marker ABOVE deepest root must not leak in.
    assert bound([inner, outer])
    (inner / "PROTECTED.txt").write_text(marker, encoding="utf-8")
    assert not bound([outer, inner])  # Deepest root itself IS included.
    assert not bound([inner, outer])


@pytest.mark.parametrize("directive", ["covers", "protect"])
def test_g2_existing_directive_consent_stays_881e_authorized(tmp_path, directive):
    run = _authorized_tree(tmp_path)
    (run / "SPEC.md").unlink()
    (run / "PROTECTED.txt").write_text(
        f"# legacy consent\n{directive}: {tmp_path / 'experiments' / 'owner'}\n",
        encoding="utf-8")
    assert _bound(run, [tmp_path / "runs"])
    _assert_cli_authorized(tmp_path)


def test_g2_citations_none_escape_hatch_is_exact(tmp_path, monkeypatch):
    run = _authorized_tree(tmp_path)
    (run / "PROTECTED.txt").write_text("# protected\n", encoding="utf-8")

    def forbidden(*args):
        pytest.fail("citations=None must not consult the declaration")

    monkeypatch.setattr(gc, "_run_declaration_refusal", forbidden)
    named = {str(run)}
    assert gc.resolve_bound_coverage([str(run)], named, citations=None) == \
        gc.resolve_coverage([str(run)], named)


def test_g1_in_run_parity_preserves_declared_run_not_sibling(tmp_path, monkeypatch):
    from mcts_trader.training import safe_checkpoint as sc

    run = _authorized_tree(tmp_path)
    sibling = run.parent / "sibling"
    sibling.mkdir()
    ckpt = sibling / "checkpoint_1"
    ckpt.mkdir()
    (ckpt / "checkpoint").write_bytes(b"fixture")
    owner = tmp_path / "experiments" / "owner"
    with (owner / "PROTECTED.txt").open("a", encoding="utf-8") as stream:
        stream.write(f"covers: {sibling}\n")
    (sibling / "SPEC.md").write_text(f"covers:\n- {owner}\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    sc._reset_protection_cache_for_test()
    try:
        def prune(path):
            messages, failed = sc.prune_run_dir_by_score(
                str(path), roots=(str(run.parent),), scan_dirs=(str(tmp_path / "experiments"),),
                apply=False, authority_binding=True)
            assert not failed, messages
            return "\n".join(messages)

        assert "REFUSE" not in prune(run)
        assert "REFUSE" not in prune(sibling)
        (run / "PROTECTED.txt").write_text("# retain\n", encoding="utf-8")
        assert "self-protected" in prune(run)
        assert "REFUSE" not in prune(sibling)
        (run / "PROTECTED.txt").write_text("malformed\n", encoding="utf-8")
        assert "malformed PROTECTED.txt" in prune(run)
        assert len(list(run.glob("checkpoint_*"))) == 3
    finally:
        sc._reset_protection_cache_for_test()


META = dict(name="922_poppy", task="unspecified", reason="Retain café checkpoints")


def test_g3_writer_new_directory_utf8_comment_metadata(tmp_path):
    run = tmp_path / "new_parent" / "run"
    assert writer.create_protected_run_dir(run, **META) is True
    text = (run / "PROTECTED.txt").read_text(encoding="utf-8")
    assert all(line.startswith("# ") for line in text.splitlines())
    for key, value in META.items():
        assert f"# {key}: {value}\n" in text
    date.fromisoformat(next(line.removeprefix("# date: ") for line in text.splitlines()
                           if line.startswith("# date: ")))
    entries, _, malformed = gc.parse_protected_txt(text, str(run / "PROTECTED.txt"))
    assert not entries and not malformed
    assert "self-protected" in gc._run_declaration_refusal(str(run), [str(tmp_path)])


@pytest.mark.parametrize("manifest", [None, b"# old metadata\n", b"covers: experiments/owner\n"])
def test_g2_writer_existing_directory_and_manifests_untouched(tmp_path, manifest):
    run = tmp_path / "existing"
    run.mkdir()
    (run / "SPEC.md").write_bytes(b"existing spec\n")
    if manifest is not None:
        (run / "PROTECTED.txt").write_bytes(manifest)
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in run.iterdir()}
    dir_stat = run.stat().st_mtime_ns
    assert writer.create_protected_run_dir(run, **META) is False
    assert before == {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in run.iterdir()}
    assert run.stat().st_mtime_ns == dir_stat


@pytest.mark.parametrize("field", ["name", "task", "reason"])
@pytest.mark.parametrize("bad", ["", "   ", "a\nb", "a\rb", "a\x00b", "a\u2028b", "a\x85b", "a\tb", 922])
def test_g3_invalid_metadata_refuses_before_any_mkdir(tmp_path, field, bad):
    run = tmp_path / "absent_parent" / "run"
    metadata = dict(META, **{field: bad})
    with pytest.raises(ValueError, match=field):
        writer.create_protected_run_dir(run, **metadata)
    assert not run.parent.exists()


def test_g3_creation_failure_is_loud(tmp_path):
    run = tmp_path / "file_not_directory"
    run.write_bytes(b"untouched")
    with pytest.raises(FileExistsError):
        writer.create_protected_run_dir(run, **META)
    assert run.read_bytes() == b"untouched"


def test_g3_mkdir_loser_does_not_write_into_concurrent_run(tmp_path, monkeypatch):
    run = tmp_path / "raced"
    original = Path.mkdir

    def race(path, *args, **kwargs):
        if path == run:
            original(path)
            (path / "PROTECTED.txt").write_bytes(b"# other creator\n")
            raise FileExistsError("other creator won")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", race)
    assert writer.create_protected_run_dir(run, **META) is False
    assert (run / "PROTECTED.txt").read_bytes() == b"# other creator\n"


def test_g3_exclusive_manifest_creation_failure_is_loud(tmp_path, monkeypatch):
    run = tmp_path / "new"
    original = Path.open

    def raced_open(path, *args, **kwargs):
        if path == run / "PROTECTED.txt":
            with original(path, "wb") as stream:
                stream.write(b"# concurrent declaration\n")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", raced_open)
    with pytest.raises(FileExistsError):
        writer.create_protected_run_dir(run, **META)
    with original(run / "PROTECTED.txt", "rb") as stream:
        assert stream.read() == b"# concurrent declaration\n"


def test_g3_write_failure_is_loud_and_preserves_new_directory(tmp_path, monkeypatch):
    stream = mock_open()
    stream.return_value.write.side_effect = OSError("922 disk full")
    monkeypatch.setattr(Path, "open", stream)
    run = tmp_path / "new"
    with pytest.raises(OSError, match="922 disk full"):
        writer.create_protected_run_dir(run, **META)
    assert run.is_dir()
    stream.assert_called_once_with("x", encoding="utf-8", newline="\n")


def test_g3_actual_train_jax_fast_has_single_creation_hook():
    tree = ast.parse((REPO / "train_jax_fast.py").read_text(encoding="utf-8"))
    imports = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
               and node.module == "mcts_trader.training.run_declaration"]
    assert len(imports) == 1
    assert [(alias.name, alias.asname) for alias in imports[0].names] == [
        ("create_protected_run_dir", None)]
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "create_protected_run_dir"]
    assert len(calls) == 1
    call = calls[0]
    assert len(call.args) == 1 and ast.unparse(call.args[0]) == "checkpoint_dir"
    keywords = {kw.arg: kw.value for kw in call.keywords}
    assert set(keywords) == {"name", "task", "reason"}
    assert ast.unparse(keywords["name"]) == "config.name"
    assert ast.literal_eval(keywords["task"]) == "unspecified"
    assert ast.literal_eval(keywords["reason"])
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "os.makedirs":
            assert not node.args or ast.unparse(node.args[0]) != "checkpoint_dir"
    # The hook is the immediate statement after the actual checkpoint_dir assignment,
    # not an unused wrapper or a conditional hook that misses real creation.
    for parent in ast.walk(tree):
        body = getattr(parent, "body", None)
        if not isinstance(body, list):
            continue
        for before, after in zip(body, body[1:]):
            if isinstance(after, ast.Expr) and after.value is call:
                assert isinstance(before, ast.Assign)
                assert [ast.unparse(t) for t in before.targets] == ["checkpoint_dir"]
                assert "jax_ckpt_fast/" in ast.unparse(before.value)
                return
    pytest.fail("creation hook must directly replace the actual mkdir statement")
