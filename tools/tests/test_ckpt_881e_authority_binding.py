"""TASK_881E — authority binding for the manifest coverage predicate (D64's amendment, D72's
re-cut mechanism).

D64 (ruled 2026-08-25) froze `ckpt_gc.py --apply` because the coverage predicate
(`resolve_coverage`) has no AUTHORITY BINDING: a grammatically perfect `covers:`/`protect:` line
in ANY `experiments/<id>/PROTECTED.txt` grants coverage over ANY run dir it names, with no check
that `<id>` has any relationship to that run dir. TASK_881E adds the binding as a CONJUNCT
(`resolve_bound_coverage`, `tools/ckpt_gc.py`).

D64's ORIGINAL amendment text proposed id-correlation (a run dir's path-derived task number
matched against the citing manifest's own `experiments/<id>/` directory name). D72 (operator,
2026-08-29) KILLED that design after factory run 46 proved it provably ambiguous:
`runs/875_wf/f1_s0` and `runs/000_smoke/adhoc_debug_run` decompose IDENTICALLY under every rule
tried, yet one must cover and the other must refuse. D72's replacement, SHIPPED HERE: authority
= the run dir's OWN creation record — a PROTECTED.txt/SPEC.md physically written INSIDE the
run's own tree (at the run dir itself, or an ancestor of it up to its checkpoint root) that
names the citing experiment back. No id, no number, is ever extracted from any path anywhere in
this mechanism (`tools/ckpt_gc.py`'s `_own_manifest_entries` / `_own_manifest_names`).

D70 (ruled 2026-08-28, planner; OVERRIDES this row's original "flag-gated, default OFF" framing
and config.json's general invariant for THIS row specifically): the binding ships **DEFAULT ON**
(`--bind-authority` / `authority_binding=` both default True) — "a flag-gated safety binding on a
remote deleter is a safety feature that is off ... this row changes no numeric — it narrows who
may authorize a delete." The pre-existing 881/881D/881I/881G suites, which use placeholder
manifest ids (`999_x`, `e1`, `e2`, `hostile_unrelated`, ...) throughout, by design, to isolate
COVERAGE from AUTHORITY, keep passing via D70's own required mechanism: a named, TEST-ONLY escape
hatch (`--no-bind-authority` / `authority_binding=False`) those suites opt into explicitly.

Gate-numbering: this file uses TASK_881E's own re-cut acceptance gates, exactly as written in
`tasks/TASK_881E_MANIFEST_AUTHORITY_BINDING.md` (2026-08-29 re-cut):
  (i)   the D64 repro flips from "covers, deletes" to "refuses, deletes nothing"
  (ii)  every protection the CURRENT corpus grants is re-derived under the binding (a diff) —
        this is a READ-ONLY scan of the real tree, run as a standalone dry-run invocation, never
        as a pytest here, so a real-tree scan is never accidentally exercised by
        `pytest tools/tests`. Its RESULT is not deferred to any external document: the full
        enumeration (REFUSE 15 -> 18; the three run dirs that lose standing; the host/mount
        caveat on the absolute counts) is written INLINE in `tools/ckpt_gc.py`, in the
        "GATE (ii) — THE ENUMERATION" block of the TASK_881E section banner, and
        `test_gate_ii_enumeration_is_stated_in_the_tool_itself` below pins that it stays there.
  (iii) the 881D/881I/881G suites stay green; canon pins re-pinned only where legitimate
  (iv)  the D72 proof pair (`runs/875_wf/f1_s0` vs `runs/000_smoke/adhoc_debug_run`) codified as
        a fixture

Sandboxed under tmp_path throughout. Nothing here ever touches the real `runs/`,
`jax_ckpt_fast/`, or `experiments/`.
"""

import argparse
import ast
import collections
import hashlib
import importlib.util
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GC_PATH = os.path.join(REPO, "tools", "ckpt_gc.py")
if REPO not in sys.path:
    sys.path.insert(0, REPO)

_spec = importlib.util.spec_from_file_location("ckpt_gc", GC_PATH)
gc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gc)

from mcts_trader.training import safe_checkpoint as sc  # noqa: E402

D64_OVERRIDE_FLAG = "--i-have-read-d64-and-accept-the-unbound-coverage-risk"
D64_OVERRIDE_TOKEN = "I-HAVE-READ-D64"
D64_OVERRIDE = f"{D64_OVERRIDE_FLAG}={D64_OVERRIDE_TOKEN}"


def _mk_ckpt(root, name, mtime=None):
    p = os.path.join(root, name)
    os.makedirs(p, exist_ok=True)
    (open(os.path.join(p, "checkpoint"), "w")).close()
    if mtime is not None:
        os.utime(p, (mtime, mtime))
    return p


def _write_manifest(dirpath, text, basename="PROTECTED.txt"):
    os.makedirs(dirpath, exist_ok=True)
    with open(os.path.join(dirpath, basename), "w") as fh:
        fh.write(text)
    return os.path.join(dirpath, basename)


def _run_gc(args, cwd):
    return subprocess.run([sys.executable, GC_PATH] + args, cwd=cwd,
                          capture_output=True, text=True)


def _d64_hostile_tree(tmp_path):
    """D64's OWN hostile scenario (mirrors test_ckpt_881_protection_model.py's own
    `_d64_hostile_tree`, rebuilt here so this file is self-contained): a run dir whose only
    coverage comes from a `covers:` line in a totally unrelated experiment id, and which carries
    NO manifest of its own."""
    runs = tmp_path / "runs"
    for i in (1, 2, 3):
        _mk_ckpt(str(runs / "target_run"), f"checkpoint_{i}")
    hostile = tmp_path / "experiments" / "hostile_unrelated"
    hostile.mkdir(parents=True)
    (hostile / "PROTECTED.txt").write_text("covers: runs/target_run\n")
    return runs / "target_run"


_D64_APPLY_ARGS = ["--roots", "runs", "--experiments-dir", "experiments",
                   "--min-free-gb", "99999", "--keep-unscored", "0", "--keep-latest", "1"]


def _survivors(run_dir):
    return sorted(p.name for p in run_dir.iterdir())


# TASK_881D: `--apply` now implies `--write-index` (an auditable `checkpoints_index.json` per
# run dir) and appends `prunes.jsonl` — sanctioned artifacts a DELETING run writes into run_dir
# itself, not checkpoints, so a survivors comparison after an actual delete must exclude them.
_AUDIT_FILES = {"checkpoints_index.json", "prunes.jsonl"}


def _survivor_checkpoints(run_dir):
    return [n for n in _survivors(run_dir) if n not in _AUDIT_FILES]


# ==================================================================== gate (i): the D64 repro
# "the unrelated-hostile_unrelated manifest no longer covers ... a manifest with no declared
# relationship refuses" — reproduced BOTH directions, on isolated scratch trees, using the exact
# grammatically-well-formed shape the module docstring and D64 itself name: a `covers:` line in
# an UNRELATED `experiments/<id>/PROTECTED.txt`, with the run dir carrying no manifest of ITS own.

def test_d64_repro_flag_omitted_now_refuses_and_deletes_nothing(tmp_path):
    """D70 (2026-08-28): the binding is DEFAULT ON, so the flag OMITTED is the PRODUCTION path —
    it must refuse and delete nothing. This is gate (i)'s repro run with ZERO flags beyond what
    production would actually pass."""
    run_dir = _d64_hostile_tree(tmp_path)
    r = _run_gc(_D64_APPLY_ARGS + ["--apply", D64_OVERRIDE], cwd=str(tmp_path))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "REFUSE run dir" in r.stdout, r.stdout
    assert _survivors(run_dir) == ["checkpoint_1", "checkpoint_2", "checkpoint_3"], (
        f"default-ON must delete NOTHING even with no flags given: {r.stdout}")


def test_d64_repro_no_bind_authority_escape_hatch_reproduces_pre_d72_covers_and_deletes(tmp_path):
    """The named, test-only escape hatch (D70): `--no-bind-authority` explicitly restores the
    PRE-881E flat-namespace behaviour on demand — the hostile manifest covers again, and --apply
    (with the D64 override) deletes again."""
    run_dir = _d64_hostile_tree(tmp_path)
    r = _run_gc(_D64_APPLY_ARGS + ["--apply", D64_OVERRIDE, "--no-bind-authority"],
               cwd=str(tmp_path))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "REFUSE run dir" not in r.stdout, r.stdout
    assert _survivor_checkpoints(run_dir) == ["checkpoint_3"], (
        f"the escape hatch must reproduce pre-881E behaviour exactly: {r.stdout}")


def test_d64_repro_bind_authority_refuses_and_deletes_nothing(tmp_path):
    """`--bind-authority` given EXPLICITLY (redundant with the D70 default, kept as its own
    test): the run dir carries no manifest of its own at all, so it refuses wholesale and every
    checkpoint survives, and the refusal text is scope item 2's wording — naming what WOULD
    confer standing (a manifest INSIDE the run dir's own tree) — never the pre-881E generic
    "an experiments/<id>/PROTECTED.txt" (any id), and never the removed `_AUTHORITY_ALLOWLIST`
    text (D72 killed the allow-list along with id-correlation)."""
    run_dir = _d64_hostile_tree(tmp_path)
    r = _run_gc(_D64_APPLY_ARGS + ["--apply", D64_OVERRIDE, "--bind-authority"], cwd=str(tmp_path))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "REFUSE run dir" in r.stdout, r.stdout
    assert _survivors(run_dir) == ["checkpoint_1", "checkpoint_2", "checkpoint_3"], (
        f"post-fix must delete NOTHING: {r.stdout}")
    assert "hostile_unrelated" in r.stdout
    assert "no declared relationship" in r.stdout
    assert "carries no PROTECTED.txt/SPEC.md of its own" in r.stdout
    assert "_AUTHORITY_ALLOWLIST" not in r.stdout, (
        "D72 removed the allow-list entirely — the refusal text must not still point at it")
    assert "an experiments/<id>/PROTECTED.txt" not in r.stdout


def test_d64_repro_own_manifest_naming_back_still_covers_under_the_binding(tmp_path):
    """Scope item 3's second clause: "the same content under the manifest the run's own record
    names still covers". Identical `covers:` line from `experiments/real_owner/`, but now
    `runs/target_run` carries ITS OWN PROTECTED.txt naming `experiments/real_owner` back —
    D72's authority — and `--bind-authority` does NOT refuse it."""
    runs = tmp_path / "runs"
    _mk_ckpt(str(runs / "target_run"), "checkpoint_1")
    _mk_ckpt(str(runs / "target_run"), "checkpoint_2")
    owner = tmp_path / "experiments" / "real_owner"
    owner.mkdir(parents=True)
    (owner / "PROTECTED.txt").write_text("covers: runs/target_run\n")
    _write_manifest(str(runs / "target_run"), "covers: experiments/real_owner\n")
    r = _run_gc(["--roots", "runs", "--experiments-dir", "experiments", "--min-free-gb", "1e18",
               "--bind-authority"], cwd=str(tmp_path))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "REFUSE run dir" not in r.stdout, (
        f"a run dir whose OWN manifest names the citing experiment back must still cover: "
        f"{r.stdout}")


def test_d64_repro_own_manifest_naming_someone_else_still_refuses(tmp_path):
    """D83(i) mutation proof #1 — the blind spot a lazy implementation would miss: the run dir
    DOES carry its own manifest, so a broken check that only asks 'does ANY own manifest exist'
    (rather than 'does it name THIS SPECIFIC citing experiment') would wrongly grant coverage
    here. `runs/target_run`'s own manifest names `experiments/some_other_experiment` — NOT
    `experiments/real_owner`, which is the one actually citing it — so the grant must still
    refuse."""
    runs = tmp_path / "runs"
    _mk_ckpt(str(runs / "target_run"), "checkpoint_1")
    owner = tmp_path / "experiments" / "real_owner"
    owner.mkdir(parents=True)
    (owner / "PROTECTED.txt").write_text("covers: runs/target_run\n")
    (tmp_path / "experiments" / "some_other_experiment").mkdir(parents=True)
    _write_manifest(str(runs / "target_run"), "covers: experiments/some_other_experiment\n")
    r = _run_gc(["--roots", "runs", "--experiments-dir", "experiments", "--min-free-gb", "1e18",
               "--bind-authority"], cwd=str(tmp_path))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "REFUSE run dir" in r.stdout, (
        f"an own manifest that names a DIFFERENT experiment must not authorize this citer: "
        f"{r.stdout}")


# ============================================================== unit-level: resolve_bound_coverage

def test_resolve_bound_coverage_with_no_citations_is_byte_identical_to_resolve_coverage(
        monkeypatch):
    """The "additive by construction" claim, exercised directly: omitting citations returns
    EXACTLY resolve_coverage's own dict — the flag-off case every existing test relies on."""
    named = {"/a/runs/x", "/a/runs/y/ckpt"}
    run_dirs = ["/a/runs/x", "/a/runs/y"]
    monkeypatch.setattr(gc.os.path, "exists", lambda p: True)
    base = gc.resolve_coverage(run_dirs, named)
    bound = gc.resolve_bound_coverage(run_dirs, named)
    assert bound == base


def test_resolve_bound_coverage_can_only_remove_never_add(tmp_path, monkeypatch):
    """A direct, constructive proof of the non-negotiable: build a tree where the base predicate
    covers 2 run dirs (one with its own naming-back manifest, one without) and confirm the bound
    predicate's covered set is a SUBSET of the base predicate's — never a superset, never a
    disjoint addition."""
    monkeypatch.chdir(tmp_path)
    runs = tmp_path / "runs"
    _mk_ckpt(str(runs / "authorized"), "checkpoint_1")
    _mk_ckpt(str(runs / "unauthorized"), "checkpoint_1")
    exp = tmp_path / "experiments" / "owner"
    exp.mkdir(parents=True)
    (exp / "PROTECTED.txt").write_text("covers: runs/authorized\ncovers: runs/unauthorized\n")
    _write_manifest(str(runs / "authorized"), "covers: experiments/owner\n")

    manifests = gc.load_manifests(["experiments"])
    named = manifests.protect | manifests.covers
    run_dirs = [str(runs / "authorized"), str(runs / "unauthorized")]

    base = gc.resolve_coverage(run_dirs, named, set(manifests.dangling))
    bound = gc.resolve_bound_coverage(run_dirs, named, set(manifests.dangling),
                                      citations=manifests.citations, roots=["runs"])

    assert base[str(runs / "authorized")] is True
    assert base[str(runs / "unauthorized")] is True
    assert bound[str(runs / "authorized")] is True, "the authorized grant must survive the binding"
    assert bound[str(runs / "unauthorized")] is False, "the unauthorized grant must be removed"
    for rd in run_dirs:
        assert bound[rd] <= base[rd], f"{rd}: bound coverage exceeded base coverage"


def test_protect_file_entries_are_untouched_by_the_binding(tmp_path):
    """Scope item 1 binds `covers:`/`protect:` MANIFEST entries only — an operator
    `--protect-file` entry (a different, explicit trust mechanism) has no citing manifest to
    correlate and must not be refused by the binding."""
    runs = tmp_path / "runs"
    ck = _mk_ckpt(str(runs / "unrelated_name"), "checkpoint_1")
    pf = tmp_path / "protect.txt"
    pf.write_text(f"{ck}\n")
    r = _run_gc(["--roots", "runs", "--protect-file", str(pf), "--min-free-gb", "1e18",
               "--bind-authority"], cwd=str(tmp_path))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "REFUSE run dir" not in r.stdout, (
        f"a --protect-file entry must satisfy coverage under the binding too: {r.stdout}")


def test_resolve_bound_coverage_without_roots_refuses_rather_than_grants(tmp_path, monkeypatch):
    """The binding ACTIVE (citations given) but `roots` omitted must fail CLOSED: with no roots,
    `_own_manifest_entries` finds no owning root for any run dir and therefore no own manifest
    at all, however real one on disk actually is. It is keyword-optional only so a caller
    omission can never raise inside the in-run pruner's apply=True path — the cost of that is
    paid in refusals, never in grants."""
    monkeypatch.chdir(tmp_path)
    runs = tmp_path / "runs"
    _mk_ckpt(str(runs / "authorized"), "checkpoint_1")
    exp = tmp_path / "experiments" / "owner"
    exp.mkdir(parents=True)
    (exp / "PROTECTED.txt").write_text("covers: runs/authorized\n")
    _write_manifest(str(runs / "authorized"), "covers: experiments/owner\n")
    manifests = gc.load_manifests(["experiments"])
    named = manifests.protect | manifests.covers
    rd = [str(runs / "authorized")]
    kw = dict(citations=manifests.citations)
    assert gc.resolve_bound_coverage(rd, named, set(manifests.dangling), **kw)[rd[0]] is False
    assert gc.resolve_bound_coverage(rd, named, set(manifests.dangling), roots=["runs"],
                                     **kw)[rd[0]] is True


# ========================================================= gate (iii) sibling: the flag itself

def test_bind_authority_flag_exists_with_an_unused_first_letter():
    """Sanity check for the NAMING CONSTRAINT this row's own addition must respect: `--b` was
    unused before this row (no HEAD long option starts with 'b'), so it cannot steal an
    abbreviation."""
    r = _run_gc(["--help"], cwd=REPO)
    assert "--bind-authority" in r.stdout


def test_refusal_text_default_is_the_scope_item_2_wording_not_the_pre_881e_one(tmp_path):
    """The binding is DEFAULT ON, so the DEFAULT refusal text is scope item 2's wording (naming
    what WOULD confer standing — a manifest inside the run's own tree), not the pre-881E generic
    "an experiments/<id>/PROTECTED.txt" (any id) text."""
    runs = tmp_path / "runs"
    _mk_ckpt(str(runs / "my_run"), "checkpoint_1")
    r = _run_gc(["--roots", "runs", "--min-free-gb", "1e18"], cwd=str(tmp_path))
    assert "to an experiments/<id>/PROTECTED.txt" not in r.stdout, r.stdout
    assert "no manifest names it at all" in r.stdout, r.stdout
    assert "carries no PROTECTED.txt/SPEC.md of its own" in r.stdout, r.stdout


def test_no_bind_authority_escape_hatch_reproduces_the_pre_881e_refusal_wording(tmp_path):
    """The flag-off refusal text (via the named, test-only escape hatch) is UNCHANGED from
    pre-881E — this is what makes the flag genuinely additive at the text level too, not just at
    the coverage-boolean level."""
    runs = tmp_path / "runs"
    _mk_ckpt(str(runs / "my_run"), "checkpoint_1")
    r = _run_gc(["--roots", "runs", "--min-free-gb", "1e18", "--no-bind-authority"],
               cwd=str(tmp_path))
    assert "to an experiments/<id>/PROTECTED.txt" in r.stdout, r.stdout


# =============================================================== scope item 4: the in-run pruner

def test_in_run_pruner_authority_binding_default_on_refuses_unauthorized_manifest(
        tmp_path, monkeypatch):
    """D70 (2026-08-28): `authority_binding` defaults **True** — the in-run pruner is a live,
    unattended, `apply=True` deleter on every training run's score-retention save, so leaving it
    off by default would leave exactly the path D70 is about still unbound. A citing manifest
    with no reply from the run's own tree no longer covers with NO kwarg passed at all — the
    production call shape (`train_jax_fast.py` never passes `authority_binding`)."""
    sc._reset_protection_cache_for_test()
    monkeypatch.chdir(tmp_path)
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "my_run"
    _mk_ckpt(str(run_dir), "checkpoint_1")
    exp_root = tmp_path / "experiments"
    (exp_root / "999_x").mkdir(parents=True)
    (exp_root / "999_x" / "PROTECTED.txt").write_text("covers: runs/my_run\n")
    msgs, had_failure = sc.prune_run_dir_by_score(
        str(run_dir), roots=(str(runs_root),), scan_dirs=(str(exp_root),), apply=False)
    assert any("REFUSE" in m for m in msgs), msgs
    assert not had_failure
    sc._reset_protection_cache_for_test()


def test_in_run_pruner_authority_binding_false_escape_hatch_is_unaffected(tmp_path, monkeypatch):
    """The named, test-only escape hatch for Python-level callers: `authority_binding=False`
    passed EXPLICITLY reproduces the pre-D70/pre-881E flat-namespace behaviour exactly — an
    unauthorized manifest still covers."""
    sc._reset_protection_cache_for_test()
    monkeypatch.chdir(tmp_path)
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "my_run"
    _mk_ckpt(str(run_dir), "checkpoint_1")
    exp_root = tmp_path / "experiments"
    (exp_root / "999_x").mkdir(parents=True)
    (exp_root / "999_x" / "PROTECTED.txt").write_text("covers: runs/my_run\n")
    msgs, had_failure = sc.prune_run_dir_by_score(
        str(run_dir), roots=(str(runs_root),), scan_dirs=(str(exp_root),), apply=False,
        authority_binding=False)
    assert not any("REFUSE" in m for m in msgs), msgs
    assert not had_failure
    sc._reset_protection_cache_for_test()


def test_in_run_pruner_authority_binding_on_refuses_unauthorized_manifest(tmp_path, monkeypatch):
    """`authority_binding=True` wires the SAME binding into the in-run pruner: a citing manifest
    with no reply from the run's own tree no longer covers."""
    sc._reset_protection_cache_for_test()
    monkeypatch.chdir(tmp_path)
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "my_run"
    _mk_ckpt(str(run_dir), "checkpoint_1")
    exp_root = tmp_path / "experiments"
    (exp_root / "999_x").mkdir(parents=True)
    (exp_root / "999_x" / "PROTECTED.txt").write_text("covers: runs/my_run\n")
    msgs, had_failure = sc.prune_run_dir_by_score(
        str(run_dir), roots=(str(runs_root),), scan_dirs=(str(exp_root),), apply=False,
        authority_binding=True)
    assert any("REFUSE" in m for m in msgs), msgs
    assert any("no declared relationship" in m or "no manifest names it at all" in m
              for m in msgs), msgs
    assert not had_failure
    sc._reset_protection_cache_for_test()


def test_in_run_pruner_own_manifest_naming_back_still_covers(tmp_path, monkeypatch):
    """Regression guard: the narrowing must not remove REAL, DECLARED coverage from the live
    deleter. `runs/my_run` carries its own manifest naming `experiments/999_x` back."""
    sc._reset_protection_cache_for_test()
    monkeypatch.chdir(tmp_path)
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "my_run"
    _mk_ckpt(str(run_dir), "checkpoint_1")
    exp_root = tmp_path / "experiments"
    (exp_root / "999_x").mkdir(parents=True)
    (exp_root / "999_x" / "PROTECTED.txt").write_text("covers: runs/my_run\n")
    _write_manifest(str(run_dir), "covers: experiments/999_x\n")
    msgs, had_failure = sc.prune_run_dir_by_score(
        str(run_dir), roots=(str(runs_root),), scan_dirs=(str(exp_root),), apply=False,
        authority_binding=True)
    assert not any("REFUSE" in m for m in msgs), msgs
    assert not had_failure
    sc._reset_protection_cache_for_test()


# ========================= scope item 4's TEETH: proof, not a design argument (see task report)

def test_authority_binding_refusal_never_sets_had_failure_or_raises(tmp_path, monkeypatch):
    """THE PROOF: construct a refusal caused SPECIFICALLY by the authority binding (a manifest
    that covers under the OLD predicate but not the bound one) and demonstrate the deleting call
    — `apply=True`, the exact call shape train_jax_fast.py uses — returns cleanly with
    `had_failure=False` and a non-empty, informative message. No exception anywhere in the call
    stack; nothing resembling `sys.exit`."""
    sc._reset_protection_cache_for_test()
    monkeypatch.chdir(tmp_path)
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "my_run"
    ck = _mk_ckpt(str(run_dir), "checkpoint_1")
    exp_root = tmp_path / "experiments"
    (exp_root / "999_x").mkdir(parents=True)
    (exp_root / "999_x" / "PROTECTED.txt").write_text("covers: runs/my_run\n")

    msgs, had_failure = sc.prune_run_dir_by_score(
        str(run_dir), roots=(str(runs_root),), scan_dirs=(str(exp_root),), apply=True,
        authority_binding=True)

    assert had_failure is False, (
        "an authority-binding refusal must never be reported as a delete FAILURE — that is the "
        "flag `train_jax_fast.py` turns into sys.exit(1)")
    assert len(msgs) > 0 and any(len(m) > 0 for m in msgs), "the refusal must be a NON-EMPTY log line"
    assert any("REFUSE" in m for m in msgs), msgs
    assert os.path.isdir(ck), "the checkpoint must survive an authority-binding refusal"
    sc._reset_protection_cache_for_test()


def test_authority_binding_refusal_leaves_the_training_exit_path_unchanged(tmp_path, monkeypatch):
    """One level up: reconstruct train_jax_fast.py's own `ckpt_prune_failed` bookkeeping
    (train_jax_fast.py:747-771) around the call and confirm the flag it feeds `sys.exit(1)` from
    stays False after an authority-binding refusal — the exact mechanism the task asks to be
    proven, not merely argued."""
    sc._reset_protection_cache_for_test()
    monkeypatch.chdir(tmp_path)
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "my_run"
    _mk_ckpt(str(run_dir), "checkpoint_1")
    exp_root = tmp_path / "experiments"
    (exp_root / "999_x").mkdir(parents=True)
    (exp_root / "999_x" / "PROTECTED.txt").write_text("covers: runs/my_run\n")

    ckpt_prune_failed = False
    try:
        msgs, had_failure = sc.prune_run_dir_by_score(
            str(run_dir), roots=(str(runs_root),), scan_dirs=(str(exp_root),), apply=True,
            authority_binding=True)
    except Exception as e:  # mirrors train_jax_fast.py's own defense-in-depth catch
        ckpt_prune_failed = True
        msgs, had_failure = [f"raised {type(e).__name__}: {e}"], True
    if had_failure:
        ckpt_prune_failed = True

    assert ckpt_prune_failed is False, (
        f"train_jax_fast.py's own exit-code flag would have gone True — the run would "
        f"sys.exit(1) after finishing: {msgs}")
    sc._reset_protection_cache_for_test()


def test_collect_protection_returns_citations_as_a_fourth_element(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runs_root = tmp_path / "runs"
    _mk_ckpt(str(runs_root / "my_run"), "checkpoint_1")
    exp_root = tmp_path / "experiments"
    (exp_root / "999_x").mkdir(parents=True)
    (exp_root / "999_x" / "PROTECTED.txt").write_text("covers: runs/my_run\n")
    got = sc.collect_protection(roots=(str(runs_root),), scan_dirs=(str(exp_root),))
    assert got is not None
    named, protect, missing, citations = got
    assert len(named) == 1
    assert citations, "citations must be populated from the SAME scan that produced named"


def test_get_protection_return_arity_is_unchanged(tmp_path, monkeypatch):
    """The PUBLIC `get_protection` contract stays a 3-tuple — existing callers (this repo's own
    test_ckpt_881_protection_model.py included) must not need to change."""
    sc._reset_protection_cache_for_test()
    monkeypatch.chdir(tmp_path)
    runs_root = tmp_path / "runs"
    _mk_ckpt(str(runs_root / "my_run"), "checkpoint_1")
    exp_root = tmp_path / "experiments"
    (exp_root / "999_x").mkdir(parents=True)
    (exp_root / "999_x" / "PROTECTED.txt").write_text("covers: runs/my_run\n")
    named, protect, missing = sc.get_protection(roots=(str(runs_root),),
                                                scan_dirs=(str(exp_root),), ttl_seconds=1000)
    assert len(named) == 1
    assert sc.get_protection_citations(), "citations reachable via the new accessor"
    sc._reset_protection_cache_for_test()


# ========================================================== D83(i) mutation proofs — own-manifest
# lookup mechanics. Each test below names the SPECIFIC blind spot a plausible-but-wrong
# implementation would have, and proves the shipped one does not have it.

def test_own_manifest_found_at_an_ancestor_not_only_at_the_run_dir_itself(tmp_path, monkeypatch):
    """Blind spot: an implementation that only checks `run_dir` itself (never walks up) would
    miss the real, live corpus shape — one manifest for an owner dir with un-numbered children
    (`runs/875_wf/PROTECTED.txt` covering `runs/875_wf/f1_s0`, `f1_s1`, ... without duplicating
    the file in every leaf). The manifest here lives at the PARENT of the run dir being
    resolved, not at the run dir itself."""
    monkeypatch.chdir(tmp_path)
    runs = tmp_path / "runs"
    run_dir = runs / "family" / "leaf_s0"
    _mk_ckpt(str(run_dir), "checkpoint_1")
    exp = tmp_path / "experiments" / "owner"
    exp.mkdir(parents=True)
    (exp / "PROTECTED.txt").write_text("covers: runs/family/leaf_s0\n")
    _write_manifest(str(runs / "family"), "covers: experiments/owner\n")  # at the ANCESTOR
    r = _run_gc(["--roots", "runs", "--experiments-dir", "experiments", "--min-free-gb", "1e18",
               "--bind-authority"], cwd=str(tmp_path))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "REFUSE run dir" not in r.stdout, (
        f"an ancestor's own manifest must still be found by the walk-up: {r.stdout}")


def test_own_manifest_above_the_checkpoint_root_does_not_count(tmp_path, monkeypatch):
    """Blind spot: an implementation that walks up WITHOUT stopping at the configured root would
    let a manifest ABOVE `runs/` (e.g. at the repo/scratch root) launder standing onto every run
    dir underneath it — exactly the kind of blanket, unreviewed authority D64 exists to close.
    The rogue manifest here lives one level ABOVE the `runs` root itself."""
    monkeypatch.chdir(tmp_path)
    runs = tmp_path / "runs"
    run_dir = runs / "target_run"
    _mk_ckpt(str(run_dir), "checkpoint_1")
    exp = tmp_path / "experiments" / "owner"
    exp.mkdir(parents=True)
    (exp / "PROTECTED.txt").write_text("covers: runs/target_run\n")
    _write_manifest(str(tmp_path), "covers: experiments/owner\n")  # ABOVE the `runs` root
    r = _run_gc(["--roots", "runs", "--experiments-dir", "experiments", "--min-free-gb", "1e18",
               "--bind-authority"], cwd=str(tmp_path))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "REFUSE run dir" in r.stdout, (
        f"a manifest ABOVE the configured root must NOT confer standing on everything under "
        f"it: {r.stdout}")


def test_own_manifest_naming_the_citing_manifest_file_directly_also_counts(tmp_path, monkeypatch):
    """The run's own manifest may name either the citing experiment's DIRECTORY or its manifest
    FILE directly — both are accepted (`_own_manifest_names` checks both `citing_file` and
    `citing_dir`). This pins the file-level form, which the directory-level form (the main line
    of tests above) does not exercise."""
    monkeypatch.chdir(tmp_path)
    runs = tmp_path / "runs"
    run_dir = runs / "target_run"
    _mk_ckpt(str(run_dir), "checkpoint_1")
    exp = tmp_path / "experiments" / "owner"
    exp.mkdir(parents=True)
    (exp / "PROTECTED.txt").write_text("covers: runs/target_run\n")
    _write_manifest(str(run_dir), "covers: experiments/owner/PROTECTED.txt\n")
    r = _run_gc(["--roots", "runs", "--experiments-dir", "experiments", "--min-free-gb", "1e18",
               "--bind-authority"], cwd=str(tmp_path))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "REFUSE run dir" not in r.stdout, r.stdout


def test_own_manifest_via_spec_md_block_also_counts(tmp_path, monkeypatch):
    """The run's own creation record may be a SPEC.md `covers:` block, not only a
    PROTECTED.txt — `_own_manifest_entries` calls `parse_spec_manifest` for that basename too,
    the same UNCHANGED 881I grammar function `load_manifests` itself uses for SPEC.md."""
    monkeypatch.chdir(tmp_path)
    runs = tmp_path / "runs"
    run_dir = runs / "target_run"
    _mk_ckpt(str(run_dir), "checkpoint_1")
    exp = tmp_path / "experiments" / "owner"
    exp.mkdir(parents=True)
    (exp / "PROTECTED.txt").write_text("covers: runs/target_run\n")
    _write_manifest(str(run_dir), "covers:\n\n- experiments/owner\n", basename="SPEC.md")
    r = _run_gc(["--roots", "runs", "--experiments-dir", "experiments", "--min-free-gb", "1e18",
               "--bind-authority"], cwd=str(tmp_path))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "REFUSE run dir" not in r.stdout, (
        f"a SPEC.md-based own manifest must be honoured identically to PROTECTED.txt: {r.stdout}")


def test_malformed_own_manifest_contributes_nothing_fail_closed(tmp_path, monkeypatch):
    """Blind spot: a broken malformed-line handler could either raise (reaching the in-run
    pruner's apply=True path) or, worse, silently include the malformed file's entries anyway.
    `_own_manifest_entries` must withdraw the WHOLE file's entries on a malformed line (matching
    881I's own "a partially-trustworthy manifest is not trustworthy" rule) and the run dir must
    refuse — never grant on a file that could not be trusted."""
    monkeypatch.chdir(tmp_path)
    runs = tmp_path / "runs"
    run_dir = runs / "target_run"
    _mk_ckpt(str(run_dir), "checkpoint_1")
    exp = tmp_path / "experiments" / "owner"
    exp.mkdir(parents=True)
    (exp / "PROTECTED.txt").write_text("covers: runs/target_run\n")
    # "covers:" with two tokens is 881G's own MALFORMED shape (unanchored subset-grant) — the
    # whole file's entries are withdrawn, including the one that would otherwise name it back.
    _write_manifest(str(run_dir), "covers: experiments/owner extra_token\n")
    r = _run_gc(["--roots", "runs", "--experiments-dir", "experiments", "--min-free-gb", "1e18",
               "--bind-authority"], cwd=str(tmp_path))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "REFUSE run dir" in r.stdout, (
        f"a malformed own-manifest must contribute NOTHING, not silently authorize: {r.stdout}")


def test_symlinked_own_manifest_is_not_followed(tmp_path, monkeypatch):
    """Blind spot: a broken symlink check could follow a symlinked manifest BASENAME the same
    way `load_manifests` explicitly refuses to elsewhere (a manifest's target may be untracked
    or outside the tree entirely). `_own_manifest_entries` uses `os.path.isfile(...) and not
    os.path.islink(...)`, so a symlinked PROTECTED.txt inside the run dir must be treated as
    ABSENT, not read."""
    monkeypatch.chdir(tmp_path)
    runs = tmp_path / "runs"
    run_dir = runs / "target_run"
    _mk_ckpt(str(run_dir), "checkpoint_1")
    exp = tmp_path / "experiments" / "owner"
    exp.mkdir(parents=True)
    (exp / "PROTECTED.txt").write_text("covers: runs/target_run\n")
    real = tmp_path / "real_own_manifest.txt"
    real.write_text("covers: experiments/owner\n")
    os.symlink(str(real), str(run_dir / "PROTECTED.txt"))
    r = _run_gc(["--roots", "runs", "--experiments-dir", "experiments", "--min-free-gb", "1e18",
               "--bind-authority"], cwd=str(tmp_path))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "REFUSE run dir" in r.stdout, (
        f"a symlinked own-manifest must not be followed: {r.stdout}")


# ============================================================== gate (iv): the D72 proof pair
# "runs/875_wf/f1_s0 (its own manifest names 875 -> covers) vs runs/000_smoke/adhoc_debug_run
# (no such manifest -> refuses)" — the exact wall that killed name-derived ownership in run 46,
# now codified as a fixture under the D72 mechanism.

def test_d72_proof_pair_875_wf_f1_s0_covers_000_smoke_adhoc_debug_run_refuses(tmp_path,
                                                                              monkeypatch):
    """The two run dirs are built with IDENTICAL path shapes to the real corpus wall (an owner
    dir with un-numbered leaf children under `875_wf/`; a shared scratch container `000_smoke/`
    holding an unrelated leaf) — the previous, name-derived design decomposed them IDENTICALLY.
    Under the D72 binding they decompose DIFFERENTLY, and for a reason that has nothing to do
    with either path's name: `runs/875_wf` carries its own manifest naming
    `experiments/875_search_teacher_imprint` back; `runs/000_smoke` carries no manifest of its
    own naming ANYTHING back (even though something else, unrelated, cites the leaf)."""
    monkeypatch.chdir(tmp_path)
    runs = tmp_path / "runs"

    # runs/875_wf/f1_s0 — the real corpus shape (an owner dir, un-numbered children)
    _mk_ckpt(str(runs / "875_wf" / "f1_s0"), "checkpoint_1")
    exp_875 = tmp_path / "experiments" / "875_search_teacher_imprint"
    exp_875.mkdir(parents=True)
    (exp_875 / "PROTECTED.txt").write_text("covers: runs/875_wf/f1_s0\n")
    _write_manifest(str(runs / "875_wf"), "covers: experiments/875_search_teacher_imprint\n")

    # runs/000_smoke/adhoc_debug_run — the shared scratch container, no own record at all
    _mk_ckpt(str(runs / "000_smoke" / "adhoc_debug_run"), "checkpoint_1")
    exp_000 = tmp_path / "experiments" / "000_ci"
    exp_000.mkdir(parents=True)
    (exp_000 / "PROTECTED.txt").write_text("covers: runs/000_smoke/adhoc_debug_run\n")
    # deliberately NO PROTECTED.txt/SPEC.md anywhere under runs/000_smoke

    r = _run_gc(["--roots", "runs", "--experiments-dir", "experiments", "--min-free-gb", "1e18",
               "--bind-authority"], cwd=str(tmp_path))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "REFUSE run dir runs/875_wf/f1_s0" not in r.stdout, (
        f"the run dir carrying its own creation record must cover: {r.stdout}")
    assert "REFUSE run dir runs/000_smoke/adhoc_debug_run" in r.stdout, (
        f"the run dir with no creation record of its own must refuse, even though an unrelated "
        f"manifest names it: {r.stdout}")


def test_d72_proof_pair_unit_level_own_manifest_entries(tmp_path, monkeypatch):
    """The same wall, exercised directly at the unit level against `_own_manifest_entries` /
    `_own_manifest_names` — no CLI, no subprocess."""
    monkeypatch.chdir(tmp_path)
    runs = tmp_path / "runs"
    _mk_ckpt(str(runs / "875_wf" / "f1_s0"), "checkpoint_1")
    _mk_ckpt(str(runs / "000_smoke" / "adhoc_debug_run"), "checkpoint_1")
    (tmp_path / "experiments" / "875_search_teacher_imprint").mkdir(parents=True)
    _write_manifest(str(runs / "875_wf"), "covers: experiments/875_search_teacher_imprint\n")

    own_875 = gc._own_manifest_entries(str(runs / "875_wf" / "f1_s0"), ["runs"])
    assert own_875, "875_wf/f1_s0 must find its ancestor's own manifest"
    assert gc._own_manifest_names(own_875, "experiments/875_search_teacher_imprint/PROTECTED.txt")

    own_000 = gc._own_manifest_entries(str(runs / "000_smoke" / "adhoc_debug_run"), ["runs"])
    assert own_000 == [], "000_smoke/adhoc_debug_run must carry no manifest of its own"


# ============================== cycle-2 BLOCKER: THE ESCAPE HATCH'S OWN ABBREVIATION IMMUNITY
# WHAT SHIPPED IN CYCLE 1, AND WHY 28 GREEN TESTS DID NOT SEE IT. `main()`'s parser is built
# without `allow_abbrev=False` (deliberately — disabling it globally breaks every HEAD
# abbreviation, pinned by test_ckpt_881i_manifest_ingestion.py), so argparse resolves any
# UNAMBIGUOUS long-option prefix. `--no-bind-authority` is the ONLY long option in ckpt_gc.py
# beginning with `n`, so a bare `--n` resolved to it and turned the ENTIRE D72 binding off.
#
# Measured on a scratch tree before the fix (2026-09-06, this working tree), on D64's own hostile
# scenario (`runs/target_run` with no manifest of its own, covered only by
# `experiments/hostile_unrelated/PROTECTED.txt`):
#   * correct production `--apply` invocation, binding ON by default -> REFUSE, 3/3 checkpoints
#     survive;
#   * the SAME invocation plus a bare `--n` -> binding off, hostile manifest grants coverage,
#     `checkpoint_1` and `checkpoint_2` DELETED, `checkpoints_index.json` + `prunes.jsonl`
#     written, exit 0.
# That is byte-for-byte the pre-D64 vulnerability this row exists to close, reachable by a typo.
#
# NOT ONE TEST IN THIS FILE OR IN test_ckpt_881_protection_model.py INVOKED ANYTHING SHORTER THAN
# THE FULL FLAG. The mechanism was strong exactly where every test pointed (the predicate) and
# blind exactly where none did (the SPELLING that reaches the predicate) — D83(i)'s "plant the
# poison where the guard is BLIND". This section is that lens, permanently.
#
# THE FIX (ckpt_gc.py): `_bind_authority_setting` honours the hatch only when
# `_safety_flag_spelled_in_full` says it was given EXACTLY ONCE and IN FULL, and otherwise FAILS
# CLOSED — binding stays ON, loud advisory on stderr. The spelling rule is the D64 override's
# own, called rather than copied, so the RECURRENCE (a new safety flag inheriting none of the
# discipline) is what is actually closed.

# Every spelling argparse resolves to `--no-bind-authority` today, plus the full-name
# one-character-short form. `--n` is the reported two-keystroke bypass.
_BIND_OFF_PREFIXES = [
    "--n",
    "--no",
    "--no-b",
    "--no-bind",
    "--no-bind-auth",
    "--no-bind-authorit",     # one character short of full
]

BIND_OFF_FLAG = "--no-bind-authority"


def _bind_hostile_apply(tmp_path, extra):
    """D64's hostile tree + the production `--apply` invocation + whatever `extra` argv is under
    test. Returns (result, run_dir)."""
    run_dir = _d64_hostile_tree(tmp_path)
    r = _run_gc(_D64_APPLY_ARGS + ["--apply", D64_OVERRIDE] + list(extra), cwd=str(tmp_path))
    return r, run_dir


@pytest.mark.parametrize("prefix", _BIND_OFF_PREFIXES)
def test_bind_authority_hatch_is_immune_to_argparse_abbreviation(tmp_path, prefix):
    """THE BLOCKER, reproduced as a test: every one of these prefixes resolved to
    `--no-bind-authority` at argparse level and disarmed the whole D72 binding before the fix.
    Each must now FAIL CLOSED — the binding stays ON, the hostile grant is refused, and all three
    checkpoints survive."""
    r, run_dir = _bind_hostile_apply(tmp_path, [prefix])
    assert r.returncode == 0, r.stdout + r.stderr
    assert "REFUSE run dir runs/target_run" in r.stdout, (
        f"{prefix!r} disarmed the authority binding: {r.stdout}")
    assert _survivor_checkpoints(run_dir) == ["checkpoint_1", "checkpoint_2", "checkpoint_3"], \
        _survivors(run_dir)
    # ...and no `--apply` artifact was written either, which proves the delete path was never
    # reached rather than merely that this plan happened to keep everything.
    assert _survivors(run_dir) == ["checkpoint_1", "checkpoint_2", "checkpoint_3"], \
        _survivors(run_dir)


@pytest.mark.parametrize("prefix", _BIND_OFF_PREFIXES)
def test_bind_authority_hatch_abbreviation_is_reported_never_silent(tmp_path, prefix):
    """Failing closed silently would be a silent correction — the operator asked for one thing
    and got another. The advisory names the flag, the consequence, and the fact that nothing was
    deleted on its account."""
    r, _run_dir = _bind_hostile_apply(tmp_path, [prefix])
    assert "IGNORING an abbreviated/repeated --no-bind-authority" in r.stderr, r.stderr
    assert "STAYS ON" in r.stderr, r.stderr
    assert "Nothing was deleted on account of this flag." in r.stderr, r.stderr


@pytest.mark.parametrize("prefix", _BIND_OFF_PREFIXES)
def test_bind_authority_hatch_abbreviation_fails_closed_in_dry_run_too(tmp_path, prefix):
    """The gate is not `--apply`-conditional (unlike D64's, which only guards deletion): an
    abbreviated hatch in a DRY RUN must also leave the binding on, or a dry run would report a
    coverage picture the real run would not have."""
    run_dir = _d64_hostile_tree(tmp_path)
    r = _run_gc(_D64_APPLY_ARGS + [prefix], cwd=str(tmp_path))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "REFUSE run dir runs/target_run" in r.stdout, r.stdout
    assert "IGNORING an abbreviated/repeated" in r.stderr, r.stderr
    assert _survivor_checkpoints(run_dir) == ["checkpoint_1", "checkpoint_2", "checkpoint_3"]


@pytest.mark.parametrize("argv", [
    [BIND_OFF_FLAG, BIND_OFF_FLAG],            # the same full spelling twice
    [BIND_OFF_FLAG, "--n"],                    # full spelling then an abbreviation
    ["--n", BIND_OFF_FLAG],                    # ...and the other order
])
def test_bind_authority_hatch_must_appear_exactly_once(tmp_path, argv):
    """The EXACTLY-ONCE half, inherited from the D64 override's cycle-3 multiplicity rule via the
    shared helper. Not defended as "someone might type it twice" but as the general property that
    made the D64 split-token bypass possible: with more than one occurrence, the token that is
    CHECKED need not be the token that takes EFFECT. Refusing multiplicity outright removes the
    class rather than reasoning about each shape of it."""
    r, run_dir = _bind_hostile_apply(tmp_path, argv)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "REFUSE run dir runs/target_run" in r.stdout, r.stdout
    assert "IGNORING an abbreviated/repeated" in r.stderr, r.stderr
    assert _survivor_checkpoints(run_dir) == ["checkpoint_1", "checkpoint_2", "checkpoint_3"]


def test_the_full_hatch_spelling_really_does_disarm_the_binding(tmp_path):
    """POSITIVE CONTROL — without it the section above could pass by refusing everything. The
    full spelling, given exactly once, still reaches the pre-881E flat namespace and still
    deletes, which is precisely why the abbreviations above must not."""
    r, run_dir = _bind_hostile_apply(tmp_path, [BIND_OFF_FLAG])
    assert r.returncode == 0, r.stdout + r.stderr
    assert "REFUSE run dir runs/target_run" not in r.stdout, r.stdout
    assert "IGNORING an abbreviated/repeated" not in r.stderr, r.stderr
    assert _survivor_checkpoints(run_dir) == ["checkpoint_3"], _survivors(run_dir)


def test_bind_authority_on_side_is_the_default_with_no_flag_at_all(tmp_path):
    """The other positive control: no binding flag whatsoever is the PRODUCTION path (D70,
    default ON) and prints no advisory — the advisory must fire on abbreviation, not on silence."""
    r, run_dir = _bind_hostile_apply(tmp_path, [])
    assert "IGNORING an abbreviated/repeated" not in r.stderr, r.stderr
    assert "REFUSE run dir runs/target_run" in r.stdout, r.stdout
    assert _survivor_checkpoints(run_dir) == ["checkpoint_1", "checkpoint_2", "checkpoint_3"]


@pytest.mark.parametrize("argv,expected", [
    ([BIND_OFF_FLAG], True),
    (["--bind-authority", BIND_OFF_FLAG], True),        # `--bind-authority` is not a prefix of it
    ([BIND_OFF_FLAG, "--roots", "runs"], True),
    (["--no-bind-authority=x"], True),                  # NAME-only comparison; argparse rejects
    ([], False),
    (["--n"], False),
    (["--no"], False),
    (["--"], False),                                    # the end-of-options marker is not a flag
    (["--roots", BIND_OFF_FLAG], True),                 # a VALUE that looks like the flag...
    ([BIND_OFF_FLAG, BIND_OFF_FLAG], False),
])
def test_safety_flag_spelled_in_full_unit(argv, expected):
    """The shared helper itself, at unit level. The `--roots <flag>` row documents a known,
    accepted limitation stated rather than hidden: the rule reads raw argv positionally and
    cannot tell an option NAME from a VALUE that happens to be spelled like one. It is safe in
    this direction — the only way to exploit it is to spell the flag IN FULL, which is the thing
    being permitted anyway — and closing it would require re-implementing argparse's own
    tokenizer, which is what the split-token BLOCKER already taught us not to do."""
    assert gc._safety_flag_spelled_in_full(BIND_OFF_FLAG, argv) is expected


# ===================================== cycle-2 ROOT CAUSE: ONE SPELLING RULE, NOT ONE PER FLAG
# The BLOCKER above was not "someone forgot a check on one flag" — it was that the check existed
# only as three hand-rolled conditions inside `_d64_apply_refusal_reason`, so the NEXT safety flag
# added to this file was structurally guaranteed to ship without them. These tests pin the
# factoring, so a third safety flag cannot repeat it quietly.

_SPELLING_HELPERS = ("_safety_flag_occurrences", "_safety_flag_spelled_in_full")


# ------------------------------------------------------- the CONTRACT each gate must satisfy
# CYCLE-3 FIX (D83(i)). The AST reach-check further down used to be the ONLY pin on this
# property, and a re-verification lens refuted it by construction: it wrote this gate and the
# AST check stayed GREEN —
#
#     def _bind_authority_setting(a, argv):
#         _unused = _safety_flag_spelled_in_full(_BIND_AUTHORITY_OFF_FLAG, argv)  # called, IGNORED
#         if getattr(a, "bind_authority", True):
#             return True, None
#         return False, None
#
# — while fully reinstating cycle 2's BLOCKER, because the gate no longer BRANCHES on the
# helper's answer: a bare `--n` disarms the D72 binding again. "Reaches the helper" is not the
# property that matters; "the helper's answer decides the outcome" is. So the load-bearing guard
# is BEHAVIOURAL: every registered gate is DRIVEN with an abbreviated argv and must return its
# own fail-closed answer.
#
# The table has to say, per gate, what "the safety is OFF" means in that gate's OWN return type,
# because the two gates answer in opposite shapes and opposite directions (D64 returns None to
# ARM a deleter; the binding gate returns False to DISARM a check).

_GateContract = collections.namedtuple(
    "_GateContract", "gate flag_const namespace argv_prefix render abbreviations "
                     "safety_off safety_off_meaning")

_SAFETY_FLAG_GATE_CONTRACTS = (
    _GateContract(
        gate="_d64_apply_refusal_reason",
        flag_const="_D64_OVERRIDE_FLAG",
        # The namespace argparse WOULD produce: --apply given, the override's value correct.
        # Everything except the SPELLING is arranged to permit the delete, so the spelling is
        # the only variable between the positive control and the fail-closed rows.
        namespace=lambda mod: argparse.Namespace(
            apply=True, verify_manifests=False,
            **{mod._D64_OVERRIDE_DEST: mod._D64_OVERRIDE_TOKEN}),
        argv_prefix=("--apply",),
        render=lambda mod, spelling: f"{spelling}={mod._D64_OVERRIDE_TOKEN}",
        abbreviations=("--i", "--i-have", "--i-have-read-d64"),
        safety_off=lambda r: r is None,
        safety_off_meaning="--apply is ARMED (the gate returned no refusal text)",
    ),
    _GateContract(
        gate="_bind_authority_setting",
        flag_const="_BIND_AUTHORITY_OFF_FLAG",
        namespace=lambda mod: argparse.Namespace(bind_authority=False),
        argv_prefix=(),
        render=lambda mod, spelling: spelling,
        abbreviations=("--n", "--no", "--no-bind", "--no-bind-authorit"),
        safety_off=lambda r: r[0] is False,
        safety_off_meaning="the TASK_881E/D72 authority binding is DISARMED",
    ),
)

_SAFETY_FLAG_GATES = tuple(c.gate for c in _SAFETY_FLAG_GATE_CONTRACTS)


def _drive_safety_flag_gate(mod, contract, spellings):
    """Call `contract.gate` on `mod` with a parsed namespace that says the safety flag WAS given,
    and a raw argv carrying it under `spellings`. Returns (result, argv)."""
    argv = list(contract.argv_prefix) + [contract.render(mod, s) for s in spellings]
    return getattr(mod, contract.gate)(contract.namespace(mod), argv), argv


def check_safety_flag_gate_contract(mod, contract):
    """The behavioural contract as a plain function over ANY module object exposing the gate, so
    a poison-proof harness can drive it against a MUTATED COPY of `ckpt_gc.py` (the real file is
    never edited) without going through pytest. Raises AssertionError on violation."""
    flag = getattr(mod, contract.flag_const)

    # POSITIVE CONTROL first, or every row below could be satisfied by a gate that refuses
    # unconditionally: the EXACT full spelling, given exactly once, must really turn the safety
    # off. (`test_the_full_hatch_spelling_really_does_disarm_the_binding` proves the same thing
    # end-to-end through a subprocess for the binding; this is the per-gate unit of it.)
    result, argv = _drive_safety_flag_gate(mod, contract, [flag])
    assert contract.safety_off(result), (
        f"POSITIVE CONTROL FAILED: {contract.gate} driven with argv={argv!r} returned "
        f"{result!r}; the full spelling given exactly once must mean "
        f"{contract.safety_off_meaning}, else the fail-closed rows prove nothing")

    cases = [([s], f"the argparse abbreviation {s!r}") for s in contract.abbreviations]
    cases.append(([flag, flag], "a REPEATED full spelling (the multiplicity rule)"))
    for spellings, why in cases:
        result, argv = _drive_safety_flag_gate(mod, contract, spellings)
        assert not contract.safety_off(result), (
            f"{contract.gate} FAILED OPEN on {why}: driven with argv={argv!r} it returned "
            f"{result!r} — i.e. {contract.safety_off_meaning}. The gate is not BRANCHING on "
            f"the {contract.flag_const} spelling check. Calling `_safety_flag_spelled_in_full` "
            f"and discarding its answer satisfies the AST reach-check below while reinstating "
            f"the cycle-2 BLOCKER: two keystrokes silently reopen the pre-881E hole.")


def _entity_text(src, name):
    lines = src.splitlines(keepends=True)
    for node in ast.parse(src).body:
        nm = getattr(node, "name", None)
        if nm is None and isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == name:
                    nm = name
        if nm == name:
            return "".join(lines[node.lineno - 1:node.end_lineno])
    raise AssertionError(f"{name!r} not found as a top-level def/class/assignment in ckpt_gc.py")


def _argv_reading_functions(src):
    """Top-level functions in ckpt_gc.py that take a raw `argv` parameter — i.e. every function
    positioned to police (or fail to police) an argv SPELLING."""
    out = set()
    for node in ast.parse(src).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names = [a.arg for a in
                     node.args.posonlyargs + node.args.args + node.args.kwonlyargs]
            if "argv" in names:
                out.add(node.name)
    return out


@pytest.mark.parametrize("contract", _SAFETY_FLAG_GATE_CONTRACTS, ids=_SAFETY_FLAG_GATES)
def test_every_safety_flag_gate_fails_closed_on_an_abbreviated_spelling(contract):
    """THE anti-recurrence guard for the cycle-2 root cause, asserted as EFFECT rather than as
    syntax.

    WHAT THIS PROVES, for each gate in `_SAFETY_FLAG_GATE_CONTRACTS`: with the parsed namespace
    saying the safety flag was given, an argparse ABBREVIATION of it — and a repetition of the
    full spelling — makes the gate return its FAIL-CLOSED answer, while the exact full spelling
    given exactly once returns the permissive one. A gate that calls the shared helper and
    ignores the result therefore goes RED here even though the AST reach-check below stays
    green; that exact poison is re-planted against a mutated copy in the cycle-3 fix record.

    WHAT IT DOES NOT PROVE: it cannot certify that some FUTURE gate uses the helper. It binds
    the gates in the table; `test_every_argv_reading_gate_is_registered_in_the_contract_table`
    is what forces a new argv-reading gate INTO the table, and a hypothetical future gate that
    reads `sys.argv` directly rather than taking an `argv` parameter would be outside the reach
    of both. That residue is stated, not papered over."""
    check_safety_flag_gate_contract(gc, contract)


def test_every_argv_reading_gate_is_registered_in_the_contract_table():
    """The completeness half: a new safety-flag gate must not be able to dodge the behavioural
    contract above by simply not being listed in it. Every top-level function in `ckpt_gc.py`
    that takes a raw `argv` parameter is either one of the shared spelling helpers or is
    registered as a gate with its own contract row."""
    src = open(GC_PATH).read()
    unregistered = (_argv_reading_functions(src)
                    - set(_SPELLING_HELPERS) - set(_SAFETY_FLAG_GATES))
    assert not unregistered, (
        f"{sorted(unregistered)} read raw argv but carry no row in "
        f"_SAFETY_FLAG_GATE_CONTRACTS, so nothing drives them with an abbreviated spelling. "
        f"Add a contract row (namespace, spellings, and what 'the safety is OFF' means in that "
        f"gate's return type) rather than trusting the AST reach-check alone.")


@pytest.mark.parametrize("gate", _SAFETY_FLAG_GATES)
def test_every_safety_flag_gate_uses_the_shared_spelling_discipline(gate):
    """CHEAP SMOKE, not the load-bearing guard. It proves only that the gate CONTAINS a call to
    one of the shared helpers — by call, not by copy (a substring grep would be satisfied by the
    helper's name in a comment). It deliberately says nothing about whether the call's answer is
    used: a bare `_unused = _safety_flag_spelled_in_full(...)` passes this test, which is why
    `test_every_safety_flag_gate_fails_closed_on_an_abbreviated_spelling` above exists and is
    the one that actually holds the property. Kept because it localises the failure — it names
    the gate that re-derived the rule, where the behavioural test only reports the wrong
    answer."""
    src = open(GC_PATH).read()
    called = {n.func.id for n in ast.walk(ast.parse(_entity_text(src, gate)))
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert called & set(_SPELLING_HELPERS), (
        f"{gate} does not call any of {_SPELLING_HELPERS} — it is re-deriving argparse's "
        f"abbreviation rule locally, which is exactly how --no-bind-authority shipped without "
        f"the discipline the D64 override already had")


def test_the_argv_token_enumeration_has_exactly_one_implementation():
    """THE ANTI-RECURRENCE PIN. The enumeration's distinctive shape — splitting each argv token on
    `=` to compare the NAME half — must appear ONCE in the file. A second copy means a new safety
    flag re-derived the rule instead of calling the helper, which is the defect class this cycle
    closed, and this test is how it gets caught at authoring time rather than by a scratch-tree
    delete."""
    src = open(GC_PATH).read()
    n = src.count('tok.split("=", 1)[0]')
    assert n == 1, (
        f"the argv-token enumeration appears {n} times; it must live only in "
        f"`_safety_flag_occurrences` and be CALLED by every safety-flag gate")
    assert _entity_text(src, "_safety_flag_occurrences").count('tok.split("=", 1)[0]') == 1


def test_the_hatch_flag_has_exactly_one_spelling_in_the_source():
    """`add_argument` and the gate must not be able to drift apart: the flag's literal string is
    written ONCE (`_BIND_AUTHORITY_OFF_FLAG`) and referenced everywhere else. Anything else and a
    rename hardens a flag that no longer exists while leaving the real one bare — the same
    single-spelling discipline `_D64_OVERRIDE_FLAG` already uses."""
    src = open(GC_PATH).read()
    assert src.count('"--no-bind-authority"') == 1, (
        "the literal '--no-bind-authority' must appear exactly once, in "
        "`_BIND_AUTHORITY_OFF_FLAG`")
    assert '_BIND_AUTHORITY_OFF_FLAG = "--no-bind-authority"' in src
    assert "ap.add_argument(_BIND_AUTHORITY_OFF_FLAG" in src


# ------------------------------------------------------------------ the guard is LOAD-BEARING
# D83(i): a guard is a claim, and the claim is only proven by planting the defect where the guard
# sits and watching it fire. These run the tool from a MUTATED COPY of its source (the real
# `tools/ckpt_gc.py` is never edited) with the spelling discipline severed, and confirm the
# BLOCKER's exact deletion comes back. Without this, "the tests are green" would be indis-
# tinguishable from "the gate was never reached".

_HATCH_SEVERANCE = (
    "    if _safety_flag_spelled_in_full(_BIND_AUTHORITY_OFF_FLAG, argv):\n"
    "        return False, None\n"
    "    return True, _BIND_AUTHORITY_ABBREVIATION_ADVISORY",
    "    return False, None")


def test_red_when_the_hatch_spelling_gate_is_severed_at_its_real_drop_site(tmp_path):
    """SEVER THE GATE -> the measured BLOCKER returns: `--apply` plus a bare `--n` deletes
    `checkpoint_1` and `checkpoint_2` on D64's own hostile tree. This is the poison planted where
    the guard sits."""
    src = open(GC_PATH).read()
    old, new = _HATCH_SEVERANCE
    assert src.count(old) == 1, f"anchor not unique ({src.count(old)})"
    mutated = tmp_path / "mutated_ckpt_gc.py"
    mutated.write_text(src.replace(old, new))

    run_dir = _d64_hostile_tree(tmp_path)
    r = subprocess.run([sys.executable, str(mutated)] + _D64_APPLY_ARGS
                       + ["--apply", D64_OVERRIDE, "--n"],
                       cwd=str(tmp_path), capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert _survivor_checkpoints(run_dir) == ["checkpoint_3"], (
        f"severing the spelling gate must restore the BLOCKER's deletion, else this suite is "
        f"not testing the gate at all: {r.stdout}")


def test_green_on_the_same_argv_with_the_gate_intact(tmp_path):
    """The control for the severance above, run from an UNMUTATED copy at the same path shape, so
    the difference between the two is the gate and nothing else about how the tool was invoked."""
    src = open(GC_PATH).read()
    copy = tmp_path / "copy_ckpt_gc.py"
    copy.write_text(src)
    run_dir = _d64_hostile_tree(tmp_path)
    r = subprocess.run([sys.executable, str(copy)] + _D64_APPLY_ARGS
                       + ["--apply", D64_OVERRIDE, "--n"],
                       cwd=str(tmp_path), capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert _survivor_checkpoints(run_dir) == ["checkpoint_1", "checkpoint_2", "checkpoint_3"]


# ==================================== cycle-2 MAJOR: NESTED ROOTS MUST NOT DECIDE THE VERDICT
# `_own_manifest_entries` picked the FIRST root in `roots` containing the run dir, not the
# deepest. With `outer/inner/run_dir` and a PROTECTED.txt at `outer/`, `roots=[outer, inner]`
# walked up to `outer` and found it (authorize) while `roots=[inner, outer]` stopped at `inner`
# and found nothing (refuse) — identical bytes on disk, opposite authority verdicts, decided by
# the order an argument list happened to arrive in. D72 and the docstring both claim the walk
# stops "up to but never above its configured checkpoint root"; with nested roots that boundary
# was not well-defined, so the claim was false. Not reachable on today's corpus (`jax_ckpt_fast`
# and `runs` are disjoint), which is why it is a MAJOR and not a BLOCKER — but the file already
# carries `_dedupe_nested_scan_dirs` for `--experiments-dir`, i.e. nesting was known to be an
# operational reality and the discipline simply had not been extended to `--roots`.
#
# FIX: the LONGEST matching prefix. Deterministic, and the fail-closed one of the two — the
# deepest containing root is the shortest walk, so it can only ever find FEWER own-manifests.

def _nested_roots_tree(tmp_path, manifest_at):
    """`outer/inner/run_dir` with one checkpoint, and the run's own creation record written at
    `manifest_at` (a path relative to tmp_path). Returns (run_dir, outer, inner)."""
    outer = tmp_path / "outer"
    inner = outer / "inner"
    run = inner / "run_dir"
    _mk_ckpt(str(run), "checkpoint_1")
    (tmp_path / "experiments" / "e_owner").mkdir(parents=True)
    _write_manifest(str(tmp_path / manifest_at), "covers: experiments/e_owner\n")
    return str(run), str(outer), str(inner)


@pytest.mark.parametrize("manifest_at,expect_found", [
    ("outer", False),                     # ABOVE the deepest root -> never consulted
    ("outer/inner", True),                # AT the deepest root -> consulted
    ("outer/inner/run_dir", True),        # the run dir itself
])
def test_nested_roots_give_the_same_verdict_in_either_order(tmp_path, manifest_at, expect_found):
    """The MAJOR, as a test: same disk, both `roots` orderings, one verdict. The `outer` row is
    the one that was order-dependent before the fix; the other two rows are the controls that
    stop this passing by finding nothing everywhere."""
    run, outer, inner = _nested_roots_tree(tmp_path, manifest_at)
    forward = gc._own_manifest_entries(run, [outer, inner])
    reverse = gc._own_manifest_entries(run, [inner, outer])
    assert [e.path for e in forward] == [e.path for e in reverse], (
        f"authority verdict depends on --roots ORDER: {[e.path for e in forward]} vs "
        f"{[e.path for e in reverse]}")
    assert bool(forward) is expect_found


def test_nested_roots_stop_at_the_deepest_root_which_is_the_fail_closed_choice(tmp_path):
    """Determinism alone would be satisfied by always taking the SHALLOWEST root too. The choice
    is the deepest, on purpose: it is the shorter walk, so it can only ever find fewer
    own-manifests than the alternative, i.e. it refuses more and never less (D72). Pinned so a
    future 'fix' cannot quietly flip it to the granting direction."""
    run, outer, inner = _nested_roots_tree(tmp_path, "outer")
    assert gc._own_manifest_entries(run, [outer, inner]) == []
    assert gc._own_manifest_entries(run, [outer]) != [], (
        "sanity: with `outer` as the ONLY root the same manifest IS this run's own record — "
        "so the empty result above really is the nesting boundary, not an unreadable file")


_FIRST_MATCH_SEVERANCE = (
    """        if rp == rr or rp.startswith(rr + os.sep):
            if owning_root is None or len(rr) > len(owning_root):
                owning_root = rr""",
    """        if rp == rr or rp.startswith(rr + os.sep):
            owning_root = rr
            break""")


def test_red_when_root_selection_is_reverted_to_first_match(tmp_path):
    """MUTATION at the real drop site, in a copy: restore the first-match `break` and the two
    orderings really do disagree again. Proves the parametrized test above is a check on the
    selection rule and not merely on a fixture that happens to be unambiguous."""
    src = open(GC_PATH).read()
    old, new = _FIRST_MATCH_SEVERANCE
    assert src.count(old) == 1, f"anchor not unique ({src.count(old)})"
    mutated_path = tmp_path / "mutated_first_match.py"
    mutated_path.write_text(src.replace(old, new))
    _s = importlib.util.spec_from_file_location("ckpt_gc_first_match", str(mutated_path))
    mut = importlib.util.module_from_spec(_s)
    _s.loader.exec_module(mut)

    run, outer, inner = _nested_roots_tree(tmp_path, "outer")
    assert mut._own_manifest_entries(run, [outer, inner]) != [], \
        "first-match selection should walk up to `outer` and find the manifest"
    assert mut._own_manifest_entries(run, [inner, outer]) == [], \
        "first-match selection should stop at `inner` and find nothing"


# ================================================= gate (ii): THE ENUMERATION IS A KEPT ARTIFACT
# MINOR (cycle 2): `tools/ckpt_gc.py` and this file both deferred the real-corpus coverage
# enumeration to "the task report", which is not a tracked file anywhere — so the gate the task
# states ("everything the binding removes is NAMED in the artifact") pointed at nothing. The
# enumeration now lives INLINE in the tool's own TASK_881E section banner, with the host/mount
# caveat D83(ii) requires, and this test keeps it there.

_GATE_II_REQUIRED = (
    "GATE (ii) — THE ENUMERATION",
    "runs/832_e5",
    "experiments/832_linear_imprint/PROTECTED.txt",
    "runs/861_imprint",
    "runs/861_ref",
    "experiments/861_search_teacher_imprint/PROTECTED.txt",
    "15 (binding OFF)",
    "18 (binding ON)",
    "DELTA = 3",
    "BROKEN symlink",          # the host/mount caveat on the ABSOLUTE counts
    "18 -> 24",
    "workstation",                # the host the numbers were measured on (D83(ii): artifact+revision)
    "2026-09-06",
)


@pytest.mark.parametrize("needle", _GATE_II_REQUIRED)
def test_gate_ii_enumeration_is_stated_in_the_tool_itself(needle):
    """Every element of gate (ii)'s answer — the three run dirs that lose standing, the citing
    manifests, the 15->18 refusal counts, the host/mount caveat that bounds those absolute
    counts, and the measurement date — is present in `tools/ckpt_gc.py`, not deferred."""
    assert needle in open(GC_PATH).read(), (
        f"gate (ii)'s enumeration must be self-contained in the tool; missing: {needle!r}")


def test_gate_ii_is_not_deferred_to_an_untracked_document():
    """The specific regression: 'see the task report' as the ANSWER to gate (ii). The task report
    may still be referenced for narrative, but the enumeration itself must not live only there."""
    src = open(GC_PATH).read()
    assert "see the task report's gate-(ii) enumeration" not in src, (
        "gate (ii)'s enumeration is deferred to an untracked document again")
