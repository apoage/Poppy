"""TASK_905 — ``validate --fold-root`` must audit the tree it was ACTUALLY POINTED AT,
not the tree that originally generated the manifest it is reading.

The defect (measured, factory run 41, flow-verifier): ``validate_folds`` resolved every
periods-file path from the manifest's own RECORDED repo-relative string re-joined onto
this script's OWN ``_REPO``, so ``validate --fold-root <copy>`` silently re-read the
ORIGINAL tree no matter what directory it was pointed at — a ``cp -r``'d copy with a
TAMPERED ``period_*.json`` validated clean (``ALL 36 checks PASS``, exit 0) because the
tampered file was never opened.

Fix (decision recorded in the module banner above ``validate_folds`` in
``walkforward.py``): resolve every per-fold artifact against ``fold_root`` itself
(``fold_root/<fold_id>/<recorded-basename>`` — materialize's own write layout), not
against ``_REPO``. A resolved path that lands outside the named ``--fold-root`` (a
tampered ``fold_id``, or a symlink planted inside the copy) is a LOUD refusal
(``LeakError``), never a silent redirect. ``_clone_fold_tree`` (smoke's own negative-test
helper) no longer needs its own bespoke path-rewrite for this — one implementation now
serves the CLI and the smoke helper.

This suite proves, directly against the API (no CLI subprocess needed for most cases —
one CLI-level test is kept because it is the literal repro shape the task measured):
  (i)   a COPY with a tampered period file is REJECTED and the rejection NAMES the fold/
        file involved (gate i);
  (ii)  a PRISTINE copy at a different path still PASSES — no false refusal, the opposite
        failure the ``_clone_fold_tree`` MAJOR-3 history warns about (gate ii);
  (iii) in-place (unmoved) validation is unaffected (a cheap API-level complement to the
        full CLI byte-diff proof recorded in the TASK_905 implementation report);
  the fail-closed escape guard: a tampered ``fold_id`` or a symlinked fold directory
        that would resolve OUTSIDE ``--fold-root`` is refused, never silently followed;
  ``_clone_fold_tree`` no longer rewrites manifest text (scope item 3).

Run (workstation, wrapped, CPU-only):
    tools/safe_run.sh --name 905 --mem 8G --cpu -- \
        python -m pytest experiments/837_walkforward/test_905_validate_fold_root.py -q
"""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))


def _load_wf():
    spec = importlib.util.spec_from_file_location(
        "wf905", os.path.join(_HERE, "walkforward.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


wf = _load_wf()


@pytest.fixture
def scratch():
    """A UNIQUE in-repo scratch root (mkdtemp under folds/, matching run_smoke /
    test_904's convention) — materialize() writes here, never into the tracked trees."""
    folds_dir = os.path.join(_HERE, "folds")
    os.makedirs(folds_dir, exist_ok=True)
    d = tempfile.mkdtemp(prefix="real_smoke_scratch_t905_", dir=folds_dir)
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _materialize_smoke(root, **kw):
    return wf.materialize(wf.FIXTURE_DB, wf.SMOKE_FOLDS, "smoke", root,
                          n_assets=wf.SMOKE_PIT_N_ASSETS, min_days=wf.PIT_MIN_DAYS,
                          jobs_db=wf._rel(wf.FIXTURE_DB), **kw)


def _tamper_train_into_eval(fold_dir):
    """Push a fold's train end_date into its own eval window — the exact corruption
    shape the task's own measured repro used, and enough to trip ASSERT-1 AND the
    manifest cross-check AND POLICY-PIN in one edit."""
    p = os.path.join(fold_dir, "period_train.json")
    with open(p) as fh:
        d = json.load(fh)
    d["periods"][0]["end_date"] = "2099-01-01"
    with open(p, "w") as fh:
        json.dump(d, fh, indent=2)
    return p


# ------------------------------------------------------------- gate (i) / (ii) : API ----

def test_relocated_tampered_copy_is_rejected_and_names_the_file(scratch, tmp_path):
    """Gate (i), API level. ``tmp_path`` (pytest's per-test /tmp dir) is genuinely
    OUTSIDE the repo — the strong case the task calls out (a restored backup, a staged
    copy on gpu-server are not in-repo either)."""
    _materialize_smoke(scratch)
    copy = tmp_path / "relocated_tampered"
    shutil.copytree(scratch, copy)
    tampered_path = _tamper_train_into_eval(os.path.join(copy, "smokefold1"))

    with pytest.raises(wf.LeakError) as exc:
        wf.validate_folds(str(copy), wf.FIXTURE_DB, verbose=False)
    msg = str(exc.value)
    assert "smokefold1" in msg, msg
    assert os.path.basename(tampered_path) in msg or "ASSERT-1" in msg, msg


def test_relocated_pristine_copy_at_a_different_path_passes(scratch, tmp_path):
    """Gate (ii): the opposite failure mode. A pristine copy at a NEW path — never
    touched by _clone_fold_tree's old rewrite trick — must still validate clean. The
    _clone_fold_tree MAJOR-3 history is the standing warning that over-strict path
    handling makes negative tests vacuous by making EVERYTHING fail, including the
    innocent case."""
    m = _materialize_smoke(scratch)
    copy = tmp_path / "relocated_pristine"
    shutil.copytree(scratch, copy)

    in_place_checks = wf.validate_folds(scratch, wf.FIXTURE_DB, verbose=False)
    relocated_checks = wf.validate_folds(str(copy), wf.FIXTURE_DB, verbose=False)
    assert len(relocated_checks) == len(in_place_checks)
    assert all(c["ok"] for c in relocated_checks)
    assert len(m["folds"]) == 2   # sanity: smoke set is 2 folds, nothing silently skipped


def test_in_place_unmoved_validation_still_passes(scratch):
    """Gate (iii) complement at the API level: the ordinary in-place call (fold_root ==
    the directory materialize just wrote to — what every existing caller does) must be
    completely unaffected by the relocation fix."""
    _materialize_smoke(scratch)
    checks = wf.validate_folds(scratch, wf.FIXTURE_DB, verbose=False)
    assert all(c["ok"] for c in checks)


def test_cli_cp_r_copy_tampered_then_pristine(scratch, tmp_path):
    """The literal repro shape the task measured, first-hand, through the CLI: a plain
    ``cp -r`` (not _clone_fold_tree — no manifest rewrite of any kind) then
    ``validate --fold-root <copy>``. Tampered -> reject (exit 1, names the file);
    pristine at the same kind of relocated path -> accept (exit 0)."""
    _materialize_smoke(scratch)
    bad = tmp_path / "cli_tampered"
    good = tmp_path / "cli_pristine"
    shutil.copytree(scratch, bad)
    shutil.copytree(scratch, good)
    _tamper_train_into_eval(os.path.join(bad, "smokefold1"))

    env = dict(os.environ)
    r_bad = subprocess.run(
        [sys.executable, os.path.join(_HERE, "walkforward.py"), "validate",
         "--fold-root", str(bad)],
        cwd=_REPO, capture_output=True, text=True, env=env)
    assert r_bad.returncode == 1, r_bad.stdout + r_bad.stderr
    assert "LEAK GATE" in r_bad.stderr
    assert "smokefold1" in r_bad.stderr

    r_good = subprocess.run(
        [sys.executable, os.path.join(_HERE, "walkforward.py"), "validate",
         "--fold-root", str(good)],
        cwd=_REPO, capture_output=True, text=True, env=env)
    assert r_good.returncode == 0, r_good.stdout + r_good.stderr
    assert "ALL" in r_good.stdout and "PASS" in r_good.stdout


# ------------------------------------------------------ fail-closed escape guard ----

def test_tampered_fold_id_cannot_escape_fold_root(scratch, tmp_path):
    """Scope item 2: a validated path that resolves OUTSIDE --fold-root must be a loud
    refusal, never a silent redirect. ``fold_id`` feeds directly into the reconstructed
    path (fold_root/fold_id/basename) and is attacker-controlled JSON — a `..`-laden
    fold_id is the direct escape vector (a `..` in the recorded FILE path alone cannot
    escape: only its basename survives)."""
    _materialize_smoke(scratch)
    copy = tmp_path / "escape_via_fold_id"
    shutil.copytree(scratch, copy)
    man_path = os.path.join(copy, "fold_manifest.json")
    with open(man_path) as fh:
        man = json.load(fh)
    escape = os.path.join("..", "..", "..", "..", "..", "..", "tmp", "t905_escape_target")
    man["folds"][0]["fold_id"] = escape
    with open(man_path, "w") as fh:
        json.dump(man, fh, indent=2)

    with pytest.raises(wf.LeakError) as exc:
        wf.validate_folds(str(copy), wf.FIXTURE_DB, verbose=False)
    msg = str(exc.value)
    assert "OUTSIDE" in msg and "fold-root" in msg.lower(), msg


def test_symlinked_fold_directory_cannot_escape_fold_root(scratch, tmp_path):
    """Same guard, the OTHER escape shape (TASK_886/888 precedent: a `..` string check
    alone misses a symlink escape) — a fold_id directory that is itself a symlink
    pointing outside fold_root must be refused after BOTH sides are realpath'd, not
    silently followed."""
    _materialize_smoke(scratch)
    copy = tmp_path / "escape_via_symlink"
    shutil.copytree(scratch, copy)
    outside = tmp_path / "outside_fold_root"
    outside.mkdir()
    fold1_dir = os.path.join(copy, "smokefold1")
    shutil.rmtree(fold1_dir)
    os.symlink(str(outside), fold1_dir)

    with pytest.raises(wf.LeakError) as exc:
        wf.validate_folds(str(copy), wf.FIXTURE_DB, verbose=False)
    msg = str(exc.value)
    assert "OUTSIDE" in msg and "fold-root" in msg.lower(), msg


def _two_problem_tree(src, dest, escape_first):
    """Build a copy carrying TWO independent problems at once — the exact scenario
    ``--fold-root`` exists for (auditing a partially restored backup that has more than
    one thing wrong with it):
      * one fold with a GENUINE train/eval leak (``smokefold1``, untouched fold_id), and
      * one fold whose ``fold_id`` is a containment ESCAPE (resolves outside fold_root).
    ``escape_first`` controls the manifest's fold ORDER, because the pre-fix bug was
    order-dependent in two DIFFERENT ways: escape-first swallowed the leak by never
    reaching it, leak-first swallowed the leak by discarding an already-populated
    ``fails`` list when the escape raised out of the loop. Returns the leaking fold_id."""
    shutil.copytree(src, dest)
    man_path = os.path.join(dest, "fold_manifest.json")
    with open(man_path) as fh:
        man = json.load(fh)
    leak_fid, escape_fid = (f["fold_id"] for f in man["folds"])
    _tamper_train_into_eval(os.path.join(dest, leak_fid))
    escape = os.path.join(*([".."] * 8), "tmp", "t905_escape_target")
    for f in man["folds"]:
        if f["fold_id"] == escape_fid:
            f["fold_id"] = escape
    if escape_first:
        man["folds"] = list(reversed(man["folds"]))
    with open(man_path, "w") as fh:
        json.dump(man, fh, indent=2)
    return leak_fid


@pytest.mark.parametrize("escape_first", [True, False],
                         ids=["escape-fold-first", "leak-fold-first"])
def test_escape_and_real_leak_are_BOTH_reported_in_either_order(scratch, tmp_path,
                                                                escape_first):
    """FINDING 1 (panel, correctness lens). ``validate_folds``' own docstring promises it
    "Raises LeakError listing EVERY violation (fail loud)" — every check goes through
    ``rec()`` and the single aggregate raise fires only after the WHOLE scan. The
    containment guard must obey that contract too: refusing to read out of root is
    right, raising out of the loop on the spot is not.

    The raised message is the ONLY thing the CLI puts on stderr (walkforward.py's
    ``except LeakError`` handler) and the only thing a CI grep ever sees, so an
    incomplete message is an incomplete audit no matter what streamed to stdout.

    RED before the fix in BOTH orders: escape-first never evaluates the leaking fold at
    all; leak-first collects the leak but then discards ``fails`` when the guard raises
    directly. Either way ``leak_fid not in str(e)``."""
    _materialize_smoke(scratch)
    copy = tmp_path / f"two_problems_{'escape' if escape_first else 'leak'}_first"
    leak_fid = _two_problem_tree(scratch, str(copy), escape_first)

    with pytest.raises(wf.LeakError) as exc:
        wf.validate_folds(str(copy), wf.FIXTURE_DB, verbose=False)
    msg = str(exc.value)
    # the escape is still refused, loudly
    assert "OUTSIDE" in msg and "fold-root" in msg.lower(), msg
    # ...and the unrelated REAL leak in the other fold is reported in the SAME message
    assert leak_fid in msg, msg
    assert "ASSERT-1" in msg, msg


def test_escape_plus_leak_still_exits_nonzero_through_the_cli(scratch, tmp_path):
    """FINDING 1 corollary: moving the containment refusal from an immediate raise onto
    the aggregate ``fails`` list must NOT soften it — the CLI must still exit 1 and put
    BOTH violations on stderr (the aggregate raise reaches the same handler)."""
    _materialize_smoke(scratch)
    copy = tmp_path / "cli_two_problems"
    leak_fid = _two_problem_tree(scratch, str(copy), escape_first=True)

    r = subprocess.run(
        [sys.executable, os.path.join(_HERE, "walkforward.py"), "validate",
         "--fold-root", str(copy)],
        cwd=_REPO, capture_output=True, text=True, env=dict(os.environ))
    assert r.returncode == 1, r.stdout + r.stderr
    assert "LEAK GATE" in r.stderr, r.stderr
    assert "OUTSIDE" in r.stderr, r.stderr
    assert leak_fid in r.stderr, r.stderr


# ------------------------------------------------------------- scope item 3 ----

def test_clone_fold_tree_no_longer_rewrites_manifest_text(scratch):
    """Scope item 3: _clone_fold_tree's old_rel->new_rel text-rewrite existed ONLY to
    work around validate_folds' _REPO resolution; now that validate_folds resolves
    against fold_root itself, the rewrite is dead weight and has been dropped — the
    clone's fold_manifest.json/job manifests are BYTE-IDENTICAL to the original's, and
    the clone still validates clean (proving the CLI and the smoke helper now share one
    resolution implementation instead of two divergent ones)."""
    _materialize_smoke(scratch)
    neg_root = wf._clone_fold_tree(scratch, "t905_no_rewrite_check")
    for name in ("fold_manifest.json", "train_jobs.manifest", "eval_jobs.manifest"):
        with open(os.path.join(scratch, name), "rb") as fh:
            orig = fh.read()
        with open(os.path.join(neg_root, name), "rb") as fh:
            clone = fh.read()
        assert orig == clone, f"{name} was rewritten by _clone_fold_tree"
    checks = wf.validate_folds(neg_root, wf.FIXTURE_DB, verbose=False)
    assert all(c["ok"] for c in checks)
