"""TASK_881 — the checkpoint protection model: explicit, tracked, fail-loud.

Covers the row's acceptance gates plus the two things folded in from other rows:

  * the SCORE READER scope addition (banner at the top of the task file): path-first score
    resolution for `checkpoint_<step>` and `checkpoint_epoch_<N>`, and the `_is_ckpt_dir`
    fallback that used to collapse a whole run dir into one opaque "checkpoint".
  * the disputed `_order_newest_first` finding — a THIRD independent measurement, by execution.

GATE-NUMBERING NOTE (D62(a) item 4, 2026-08-25): this suite carries TWO INDEPENDENT (i)-(vi)
sequences, and every section header / gate reference below is now prefixed to say which one it
means — the roman numerals collide (881(parent)'s (iii) is "both deleters agree", 881D's own
(iii) is "881I's malformed corpus, INTEGRATION"; 881(parent)'s (vi) is the `--ckpt-retention`
default-flip, 881D's own (vi) is the scope-item-0 corpus appearing in every report) and reading
them as one scheme silently picks the wrong definition half the time.

  881(PARENT) gate numbering — tasks/TASK_881_CHECKPOINT_PROTECTION_MODEL.md's own "Acceptance
  (gates)", prefixed "881(parent) gate (n)" below:
    (i)   no-manifest refusal, including under --apply
    (ii)  descendant-aware protection (inner path protects the enclosing checkpoint)
    (iii) both deleters (offline + in-run) agree on the same tree, TTL re-scan picks up a
          manifest added mid-run
    (iv)  advisory ledger rescues can only ADD
    (v)   a failed rmtree produces a FAILED audit row + non-zero exit, never a silent success
    (vi)  is NOT tested here — the task explicitly permits leaving --ckpt-retention at 'recency'
          if gate (vi)'s end-to-end collapse proof is not honestly reached (see the task report).

  TASK_881D gate numbering — tasks/TASK_881D_PROTECTION_DELETER.md's own "Acceptance (gates)",
  the task file THIS suite actually ships under, prefixed "TASK_881D gate (n)" below:
    (i)   every kill from the verified list reproduced on the final tree, independently rebuilt
    (ii)  a --apply dry-run on the REAL tree keeps all 7 load-bearing checkpoints
    (iii) 881I's malformed-input corpus still refuses identically through 881D's callers
          (integration, not just unit)
    (iv)  byte-identity of every non---apply code path vs HEAD, AMENDED by D62(a) — see
          test_ckpt_881i_manifest_ingestion.py
    (v)   standing clauses: guards go red when removed at the real drop site, real CLI
          invocations, mutations independently rebuilt
    (vi)  the corpus of scope item 0 exists, is tracked-or-deliberately-untracked with the reason
          recorded, and appears in every report the tool emits

All sandboxed under tmp_path / a temp `experiments/` tree. Nothing here ever touches the real
`runs/`, `jax_ckpt_fast/`, `jax_ckpt_easy/`, or `/mnt/scratch`.
"""

import ast
import importlib.util
import json
import os
import shutil
import stat
import subprocess
import sys
import time

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GC_PATH = os.path.join(REPO, "tools", "ckpt_gc.py")
if REPO not in sys.path:
    sys.path.insert(0, REPO)

_spec = importlib.util.spec_from_file_location("ckpt_gc", GC_PATH)
gc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gc)

from mcts_trader.training import safe_checkpoint as sc  # noqa: E402

# D64 (2026-08-25, TASK_881D resume): every CLI `--apply` invocation now needs this override too
# — the freeze is CODE, not convention (tools/ckpt_gc.py::_D64_APPLY_FREEZE_REFUSAL). Tests that
# exercise a DIFFERENT refusal further down the `--apply` path (a malformed manifest, an
# unusable --protect-file, ...) still need this so they reach THAT refusal rather than D64's.
#
# The override is `--<flag>=<exact token>`, a SINGLE argv token, and both halves are load-bearing
# (cycle-2 BLOCKER): the first shape of this gate was a bare `store_true` flag, which argparse's
# default `allow_abbrev=True` let `--apply --i` arm — measured deleting a real checkpoint on a
# scratch tree at exit 0 with no D64 text printed. Resistance is asserted explicitly by
# `test_d64_*_abbreviation*` / `test_d64_*_token*` below, not assumed.
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


def _run_gc(args, cwd):
    # TASK_881E/D70 (2026-08-28): the authority binding is DEFAULT ON since D70's ruling. This
    # suite's placeholder manifest ids (999_x, e1, hostile_unrelated, my_run, ...) are fixture
    # CONVENIENCE for testing D64/gate-i/ii/iii/iv/v mechanics that are ORTHOGONAL to authority
    # correlation (that is exclusively test_ckpt_881e_authority_binding.py's job) -- none of them
    # encode a real protection relationship. `--no-bind-authority` is the named, test-only escape
    # hatch D70 requires: applied here, once, at this suite's single subprocess entry point, so
    # this suite keeps exercising exactly what it always exercised rather than the new binding.
    return subprocess.run([sys.executable, GC_PATH] + args + ["--no-bind-authority"], cwd=cwd,
                          capture_output=True, text=True)


# ==================================================================== TASK_881D resume: D64 GATE
# `--apply` refuses at parse/entry time unless the override flag is present. Reachable with ZERO
# filesystem side effects — no run dir, no manifest, nothing on disk at all in these first two.

def test_d64_apply_without_override_refuses_before_any_filesystem_work(tmp_path):
    """Nothing exists under tmp_path at all — not even `runs/` or `experiments/` — so any
    scanning/discovery/planning would itself raise or print something about a missing dir. It
    doesn't: the refusal fires first."""
    r = _run_gc(["--apply"], cwd=str(tmp_path))
    assert r.returncode == 2, r.stdout + r.stderr
    assert "D64" in r.stderr, r.stderr
    assert "TASK_881E" in r.stderr, r.stderr
    assert "TASK_880's D35 gate opens" in r.stderr, r.stderr
    assert "--i-have-read-d64-and-accept-the-unbound-coverage-risk" in r.stderr, r.stderr
    # nothing scanned, nothing planned, nothing on disk to show for it
    assert r.stdout == "", r.stdout


def test_d64_refusal_does_not_overstate_the_no_manifest_refusal(tmp_path):
    """The lens's own correction, preserved as a test: the refusal text must not claim the
    no-manifest refusal is broken — only that manifest COVERAGE lacks an authority binding."""
    r = _run_gc(["--apply"], cwd=str(tmp_path))
    assert "no manifest declares coverage" not in r.stderr
    assert "AUTHORITY BINDING" in r.stderr, r.stderr
    assert "no check that" in r.stderr, r.stderr


def test_d64_apply_with_override_reaches_the_normal_refusal(tmp_path):
    """With the override, `--apply` proceeds to whatever the tool would normally do next — here,
    the pre-existing no-manifest refusal, NOT the D64 gate. Proves the override is a pass-through,
    not a second license to skip real gates."""
    runs = tmp_path / "runs"
    _mk_ckpt(str(runs / "my_run"), "checkpoint_100")
    (tmp_path / "experiments").mkdir()
    r = _run_gc(["--roots", "runs", "--apply", D64_OVERRIDE], cwd=str(tmp_path))
    assert "D64" not in r.stderr, r.stderr
    assert "REFUSE run dir" in r.stdout, r.stdout
    assert "no manifest declares coverage" in r.stdout, r.stdout


def test_d64_dry_run_completely_unaffected(tmp_path):
    """A plain dry-run (no --apply at all) must be byte-for-byte unaffected by this gate — proof
    the freeze targets only `--apply`."""
    runs = tmp_path / "runs"
    _mk_ckpt(str(runs / "my_run"), "checkpoint_100")
    (tmp_path / "experiments").mkdir()
    r = _run_gc(["--roots", "runs"], cwd=str(tmp_path))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "D64" not in r.stdout and "D64" not in r.stderr


def test_d64_write_index_without_apply_completely_unaffected(tmp_path):
    """`--write-index` alone (no `--apply`) must reach exactly its normal behaviour — the D64
    gate is keyed on `a.apply`, never on `a.write_index`."""
    runs = tmp_path / "runs"
    _mk_ckpt(str(runs / "my_run"), "checkpoint_100")
    (tmp_path / "experiments" / "e1").mkdir(parents=True)
    (tmp_path / "experiments" / "e1" / "PROTECTED.txt").write_text("covers: runs/my_run\n")
    r = _run_gc(["--roots", "runs", "--experiments-dir", "experiments", "--write-index"],
               cwd=str(tmp_path))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "D64" not in r.stdout and "D64" not in r.stderr
    assert os.path.exists(str(tmp_path / "runs" / "my_run" / "checkpoints_index.json"))


def test_d64_verify_manifests_with_apply_is_not_gated(tmp_path):
    """`--verify-manifests` is documented as never reaching the delete path even when `--apply`
    is also given ("Mutually exclusive in effect ... this mode never reaches it"). The D64 gate
    must not refuse a combination that was never going to delete anything."""
    (tmp_path / "runs").mkdir()
    (tmp_path / "experiments").mkdir()
    r = _run_gc(["--experiments-dir", "experiments", "--verify-manifests", "--apply"],
               cwd=str(tmp_path))
    assert "D64" not in r.stdout and "D64" not in r.stderr, r.stdout + r.stderr


def test_d64_override_flag_alone_without_apply_is_a_silent_noop(tmp_path):
    """The override flag by itself, without `--apply`, must not change dry-run behaviour at
    all — it exists to unlock `--apply`, nothing else."""
    runs = tmp_path / "runs"
    _mk_ckpt(str(runs / "my_run"), "checkpoint_100")
    (tmp_path / "experiments").mkdir()
    plain = _run_gc(["--roots", "runs"], cwd=str(tmp_path))
    with_override = _run_gc(["--roots", "runs", D64_OVERRIDE], cwd=str(tmp_path))
    assert plain.returncode == with_override.returncode == 0
    # strip the timestamp/free-space header line -- it legitimately differs between two
    # subprocess invocations a moment apart, same convention as this suite's other stdout diffs.
    _strip = lambda out: [l for l in out.splitlines() if "free=" not in l]
    assert _strip(plain.stdout) == _strip(with_override.stdout)


# ======================================== TASK_881D resume cycle 2: D64 ABBREVIATION RESISTANCE
# THE BLOCKER THIS SECTION CLOSES, measured before the fix: `tools/ckpt_gc.py`'s parser is built
# WITHOUT `allow_abbrev=False`, argparse therefore resolves unambiguous long-option prefixes, and
# no other long option here begins with `i` — so `--apply --i` satisfied a `store_true` override
# in two keystrokes. On an isolated scratch tree carrying D64's own hostile scenario (a
# `covers: runs/target_run` line in an unrelated `experiments/hostile_unrelated/PROTECTED.txt`) it
# reached `mode=APPLY`, printed NO D64 text, exited 0, DELETED `runs/target_run/checkpoint_1` and
# wrote `checkpoints_index.json` + `prunes.jsonl`. Nothing in the suite covered it.
#
# THE FIX IS NOT `allow_abbrev=False` (that would break every abbreviation that resolved at HEAD —
# `--c` -> `--critical-free-gb` most visibly — which is a canon-parity change AND is pinned by
# test_ckpt_881i_manifest_ingestion.py::{test_head_option_abbreviations_still_resolve,
# test_c_abbreviation_resolves_to_critical_free_gb}, whose own comment records the deliberate prior
# decision to preserve abbreviations). Instead the OVERRIDE is made immune by two independent
# conditions a prefix cannot satisfy: it TAKES A VALUE that must equal `I-HAVE-READ-D64` exactly,
# and the option must appear spelled IN FULL in argv. Both halves are asserted below, and the
# preserved abbreviations are asserted still-working in the same section.

_D64_PREFIXES = [
    "--i",                                                        # the reported two-keystroke form
    "--i-h",
    "--i-have",
    "--i-have-read",
    "--i-have-read-d64",
    "--i-have-read-d64-and-accept-the-unbound-coverage-ris",       # one character short of full
]


def _d64_hostile_tree(tmp_path):
    """D64's OWN hostile scenario, rebuilt: a run dir whose only coverage comes from a
    `covers:` line in a totally unrelated experiment. Coverage resolves, so the no-manifest
    refusal does NOT fire and the only thing standing between argv and an irreversible delete is
    the D64 gate — which is exactly what these tests need to exercise."""
    runs = tmp_path / "runs"
    for i in (1, 2, 3):
        _mk_ckpt(str(runs / "target_run"), f"checkpoint_{i}")
    hostile = tmp_path / "experiments" / "hostile_unrelated"
    hostile.mkdir(parents=True)
    (hostile / "PROTECTED.txt").write_text("covers: runs/target_run\n")
    return runs / "target_run"


_D64_APPLY_ARGS = ["--roots", "runs", "--experiments-dir", "experiments",
                   "--min-free-gb", "99999", "--keep-unscored", "0", "--keep-latest", "1"]


def _d64_tree_is_untouched(run_dir):
    """Every checkpoint still present AND neither `--apply`-only artifact written. The artifacts
    matter as much as the checkpoints: `checkpoints_index.json`/`prunes.jsonl` are written only
    once the delete path is reached, so their absence proves the gate fired BEFORE it, not merely
    that this particular plan happened to keep everything."""
    names = sorted(p.name for p in run_dir.iterdir())
    return names == ["checkpoint_1", "checkpoint_2", "checkpoint_3"]


@pytest.mark.parametrize("prefix", _D64_PREFIXES)
def test_d64_override_is_immune_to_argparse_abbreviation(tmp_path, prefix):
    """AN ABBREVIATED OPTION SPELLING MUST REFUSE EVEN WITH THE CORRECT VALUE. This is the
    BLOCKER, reproduced as a test: every one of these prefixes resolved to the override at
    argparse level and armed `--apply` before the fix."""
    run_dir = _d64_hostile_tree(tmp_path)
    r = _run_gc(["--apply", f"{prefix}={D64_OVERRIDE_TOKEN}"] + _D64_APPLY_ARGS,
                cwd=str(tmp_path))
    assert r.returncode == 2, r.stdout + r.stderr
    assert "D64" in r.stderr, r.stderr
    assert "ABBREVIATED" in r.stderr, r.stderr
    assert D64_OVERRIDE in r.stderr, r.stderr          # the refusal states the required full form
    assert _d64_tree_is_untouched(run_dir), sorted(p.name for p in run_dir.iterdir())


@pytest.mark.parametrize("prefix", _D64_PREFIXES)
def test_d64_bare_abbreviated_override_refuses_and_deletes_nothing(tmp_path, prefix):
    """The literal reported bypass argv — an abbreviated flag with NO value at all. Refused by
    argparse itself ("expected one argument"), which is a refusal in the right direction; what
    this test pins is the CONSEQUENCE, that nothing was deleted and no `--apply` artifact was
    written."""
    run_dir = _d64_hostile_tree(tmp_path)
    r = _run_gc(["--apply", prefix] + _D64_APPLY_ARGS, cwd=str(tmp_path))
    assert r.returncode == 2, r.stdout + r.stderr
    assert "mode=APPLY" not in r.stdout, r.stdout
    assert _d64_tree_is_untouched(run_dir), sorted(p.name for p in run_dir.iterdir())


@pytest.mark.parametrize("value", ["", "i-have-read-d64", "I-HAVE-READ-D65", "I-HAVE-READ-D64 ",
                                   "yes", "true", "I-HAVE-READ-D64-AND-MORE"])
def test_d64_override_requires_its_exact_literal_value(tmp_path, value):
    """The other half of the immunity: the FULL option spelling with anything but the exact,
    case-sensitive token refuses. Lower-case, off-by-one-decision, trailing-space and
    truncated/extended forms all included — a truncated argv truncates the TOKEN too, which is
    why the token is what makes a prefix useless."""
    run_dir = _d64_hostile_tree(tmp_path)
    r = _run_gc(["--apply", f"{D64_OVERRIDE_FLAG}={value}"] + _D64_APPLY_ARGS, cwd=str(tmp_path))
    assert r.returncode == 2, r.stdout + r.stderr
    assert "D64" in r.stderr, r.stderr
    assert "exact literal" in r.stderr, r.stderr
    assert _d64_tree_is_untouched(run_dir), sorted(p.name for p in run_dir.iterdir())


@pytest.mark.parametrize("argv_form", ["equals", "space"])
def test_d64_only_the_full_exact_override_arms_apply(tmp_path, argv_form):
    """POSITIVE CONTROL — without this the section above could pass by refusing everything. The
    full option spelling plus the exact token really does reach `mode=APPLY` and really does
    delete, in both argparse spellings of a valued option (`--flag=V` and `--flag V`)."""
    run_dir = _d64_hostile_tree(tmp_path)
    override = ([D64_OVERRIDE] if argv_form == "equals"
                else [D64_OVERRIDE_FLAG, D64_OVERRIDE_TOKEN])
    r = _run_gc(["--apply"] + override + _D64_APPLY_ARGS, cwd=str(tmp_path))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "mode=APPLY" in r.stdout, r.stdout
    assert "D64" not in r.stderr, r.stderr
    assert not _d64_tree_is_untouched(run_dir), r.stdout


# ============================== TASK_881D resume cycle 3: D64 REPEATED-OVERRIDE (SPLIT-TOKEN) GAP
# THE CYCLE-3 BLOCKER, measured before the fix. Cycle 2's gate had two halves that read DIFFERENT
# SOURCES: "spelled in full" was an `any()` over raw argv, "value is the exact token" came from the
# parsed namespace — and argparse's default `store` action records only the LAST occurrence. So the
# two halves could be satisfied by two DIFFERENT tokens:
#
#     --apply ... --i-have-read-d64-and-accept-the-unbound-coverage-risk=NOT_THE_TOKEN \
#                 --i=I-HAVE-READ-D64
#
# reached `mode=APPLY`, printed NO D64 text, exited 0, and DELETED `checkpoint_1` + `checkpoint_2`
# on the hostile tree. The full-form occurrence vouched for the SPELLING while the two-character
# abbreviation supplied the VALUE — which falsified cycle 2's own shipped refusal text ("nothing
# shorter than the full spelling plus the exact token can arm --apply"). Not merely an exploit: a
# mistyped full-form attempt left on the line and "corrected" by appending a short one is exactly
# the shell-history reuse the refusal warns about. NOTHING in the ~15 cycle-2 D64 tests supplied
# the option more than once; that test gap WAS the implementation gap.
#
# THE FIX (option (a) of the two offered): the option must appear EXACTLY ONCE, counting every
# argparse abbreviation of it, and more than one occurrence refuses outright in either order and
# whatever the values — including two CORRECT full forms and the wrong-then-right self-correction.
# Deliberate: repeating a safety override has no legitimate use, refusing is the fail-closed
# direction, and it makes the two halves describe the same token by construction rather than by an
# argument about argparse's internals. The property established is the one in
# `_d64_apply_refusal_reason`'s docstring: THE OCCURRENCE THAT SUPPLIES THE ARMING VALUE IS ITSELF
# THE FULL SPELLING, and no other occurrence can vouch for it.

_D64_REPEATED_SHAPES = {
    # the exact reported bypass argv
    "decoy_full_then_abbreviated":
        [f"{D64_OVERRIDE_FLAG}=NOT_THE_TOKEN", f"--i={D64_OVERRIDE_TOKEN}"],
    # ... and the same two tokens the other way round (argparse would keep the junk value here, so
    # this one never armed — pinned anyway so the rule is order-independent, not luck)
    "abbreviated_then_decoy_full":
        [f"--i={D64_OVERRIDE_TOKEN}", f"{D64_OVERRIDE_FLAG}=NOT_THE_TOKEN"],
    # two occurrences that are BOTH the full spelling with the CORRECT value: refused. Chosen
    # semantics, asserted rather than inherited.
    "full_form_twice_both_correct": [D64_OVERRIDE, D64_OVERRIDE],
    # the "typo, then fix it on the same line" shape. Cycle 2 ARMED this (the last occurrence is
    # full and correct); it now refuses, deliberately — re-editing a half-typed safety override is
    # the shell-history-reuse shape the override exists to make expensive.
    "full_form_wrong_value_then_full_form_correct":
        [f"{D64_OVERRIDE_FLAG}=NOT_THE_TOKEN", D64_OVERRIDE],
    # three occurrences, mixed spellings
    "three_occurrences_mixed":
        [f"{D64_OVERRIDE_FLAG}=junk", f"--i-have={D64_OVERRIDE_TOKEN}", D64_OVERRIDE],
    # the space-separated spelling plus an abbreviation
    "space_form_then_abbreviated":
        [D64_OVERRIDE_FLAG, D64_OVERRIDE_TOKEN, f"--i={D64_OVERRIDE_TOKEN}"],
}


@pytest.mark.parametrize("shape", sorted(_D64_REPEATED_SHAPES))
def test_d64_a_repeated_override_refuses_and_deletes_nothing(tmp_path, shape):
    """Every multi-occurrence shape refuses, on the hostile tree where a bypass really deletes.
    Checkpoint SURVIVAL is the assertion that matters — exit code alone would not distinguish
    "refused before the delete loop" from "deleted, then failed"."""
    run_dir = _d64_hostile_tree(tmp_path)
    r = _run_gc(["--apply"] + _D64_REPEATED_SHAPES[shape] + _D64_APPLY_ARGS, cwd=str(tmp_path))
    assert r.returncode == 2, r.stdout + r.stderr
    assert "mode=APPLY" not in r.stdout, r.stdout
    assert "D64" in r.stderr, r.stderr
    assert "MORE THAN ONCE" in r.stderr, r.stderr
    assert _d64_tree_is_untouched(run_dir), sorted(p.name for p in run_dir.iterdir())


def test_d64_the_reported_split_token_bypass_argv_verbatim(tmp_path):
    """The reported argv reproduced VERBATIM, flag order and all, as its own named test so the
    regression is greppable by its shape rather than only via the parametrised table. Before the
    fix: exit 0, `mode=APPLY`, `checkpoint_1` and `checkpoint_2` gone, `checkpoints_index.json` and
    `prunes.jsonl` written."""
    run_dir = _d64_hostile_tree(tmp_path)
    r = _run_gc(["--apply", "--roots", "runs", "--experiments-dir", "experiments",
                 "--min-free-gb", "99999", "--keep-unscored", "0", "--keep-latest", "1",
                 f"{D64_OVERRIDE_FLAG}=NOT_THE_TOKEN",
                 f"--i={D64_OVERRIDE_TOKEN}"], cwd=str(tmp_path))
    assert r.returncode == 2, r.stdout + r.stderr
    assert "mode=APPLY" not in r.stdout, r.stdout
    assert "MORE THAN ONCE" in r.stderr, r.stderr
    assert sorted(p.name for p in run_dir.iterdir()) == \
        ["checkpoint_1", "checkpoint_2", "checkpoint_3"], sorted(p.name for p in run_dir.iterdir())


def test_d64_repeated_override_refusal_states_the_repeat_rule(tmp_path):
    """The refusal an operator actually meets must TEACH the rule it enforced (give it once, in
    full) and must not read as the abbreviation refusal, which would send them to fix the wrong
    thing."""
    run_dir = _d64_hostile_tree(tmp_path)
    r = _run_gc(["--apply", D64_OVERRIDE, D64_OVERRIDE] + _D64_APPLY_ARGS, cwd=str(tmp_path))
    assert "EXACTLY ONCE" in r.stderr, r.stderr
    assert D64_OVERRIDE in r.stderr, r.stderr
    assert "OFFLINE CLI ONLY" in r.stderr, r.stderr        # the scope clause travels with it
    assert _d64_tree_is_untouched(run_dir)


def test_d64_freeze_refusal_text_states_the_exactly_once_rule(tmp_path):
    """The bare-`--apply` refusal is where an operator reads what the override buys. Cycle 2's
    wording — "nothing shorter than the full spelling plus the exact token can arm --apply" —
    was FALSE while the split-token bypass existed. It is true now, and the text now also states
    the rule that makes it true, so the shipped safety string does not overstate again."""
    r = _run_gc(["--apply"], cwd=str(tmp_path))
    assert r.returncode == 2
    assert "EXACTLY ONCE" in r.stderr, r.stderr
    assert "given twice in any spelling" in r.stderr, r.stderr
    assert "the single occurrence that supplies the arming value must ITSELF be the full " \
           "spelling" in r.stderr, r.stderr


def test_d64_a_single_occurrence_is_still_the_only_thing_that_arms(tmp_path):
    """NEGATIVE CONTROL on the fix itself: the multiplicity rule must not have been implemented as
    "refuse whenever the flag looks repeated" in a way that also refuses the ONE legitimate shape.
    Distinct from the positive control above in that it runs right beside the repeat cases, so a
    change that made the counter over-count would fail here rather than pass vacuously."""
    run_dir = _d64_hostile_tree(tmp_path)
    r = _run_gc(["--apply", D64_OVERRIDE] + _D64_APPLY_ARGS, cwd=str(tmp_path))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "mode=APPLY" in r.stdout, r.stdout
    assert not _d64_tree_is_untouched(run_dir), r.stdout


def test_d64_fix_preserved_every_head_abbreviation_it_could_have_broken(tmp_path):
    """THE ENUMERATED PRESERVATION CHECK: the chosen fix keeps global abbreviation ON, so every
    long-option abbreviation that resolved before the fix still resolves to the same option.
    `allow_abbrev=False` — the other candidate — would have broken all of these at once, and the
    prior decision to preserve them is recorded in
    test_ckpt_881i_manifest_ingestion.py:819-821. Pinned here too because that comment predates
    the D64 flag and this is the row that had to weigh the two against each other.

    (Deliberately checks the flags themselves, not the abbreviation MACHINERY: the class-closing
    "no new flag stole a HEAD abbreviation" test lives in the 881i suite and is not duplicated.)"""
    expected = {
        "--c": "--critical-free-gb",     # the named prior instance, D53's `--check-manifests` scar
        "--r": "--roots",
        "--m": "--min-free-gb",
        "--s": "--score-regex",
        "--l": "--log-glob",
        "--p": "--protect-file",
        "--exp": "--experiments-dir",
        "--kn": "--known-dangling",
    }
    for pref, opt in expected.items():
        r = _run_gc([pref], cwd=str(tmp_path))
        assert "ambiguous option" not in r.stderr, f"{pref} became ambiguous: {r.stderr}"
        assert f"argument {opt}:" in r.stderr, f"{pref} no longer resolves to {opt}: {r.stderr}"
    # ... and abbreviation really is still globally enabled, i.e. the fix was NOT
    # `allow_abbrev=False` on the parser. Read off the real construction line, not off a grep of
    # the whole file (which legitimately DISCUSSES the rejected option in a comment).
    src = open(GC_PATH).read()
    ctor = next(l for l in src.splitlines() if "argparse.ArgumentParser(description=__doc__" in l)
    assert "allow_abbrev" not in ctor, \
        f"the D64 fix must not disable argparse abbreviation globally: {ctor}"


def test_d64_refusal_text_claims_only_what_actually_holds(tmp_path):
    """The shipped safety string used to claim the flag's name "cannot be reused by muscle memory
    or copy-paste from an old cron line". The first half was FALSE (`--i` is two keystrokes) and
    the second half is STILL false as a matter of mechanism (a verbatim copy of a complete command
    line does arm it). A false claim in a shipped safety string is this project's named recurring
    defect class, so the claim is now the narrower one that is true, and says so."""
    r = _run_gc(["--apply"], cwd=str(tmp_path))
    assert "cannot be reused by muscle memory" not in r.stderr, r.stderr
    assert "an argparse abbreviation such as --i is REFUSED" in r.stderr, r.stderr
    assert "a verbatim copy of a COMPLETE old command line still arms it" in r.stderr, r.stderr


def test_d64_refusal_discloses_that_the_in_run_deleter_is_not_behind_this_gate(tmp_path):
    """MAJOR (fail-closed lens, cycle 2): D64's ruling text says "this row is what first makes it
    gate an irreversible deleter" without differentiating, so a reader of the ruling OR of this
    refusal would conclude the freeze covers deletion generally. It covers the offline CLI only —
    `safe_checkpoint.py` calls `load_manifests`/`plan`/`collect` directly and
    `train_jax_fast.py` calls `prune_run_dir_by_score(..., apply=True)` unconditionally under
    `--ckpt-retention score`. The gap is disclosed where a reader meets the refusal."""
    r = _run_gc(["--apply"], cwd=str(tmp_path))
    assert "OFFLINE CLI ONLY" in r.stderr, r.stderr
    assert "safe_checkpoint.py" in r.stderr, r.stderr
    assert "train_jax_fast.py" in r.stderr, r.stderr
    assert "never through argparse" in r.stderr, r.stderr


def test_the_in_run_deleter_gap_is_disclosed_in_all_three_places_a_reader_meets_it():
    """The same disclosure, at the two SOURCE locations a reader arrives at from the other
    directions: `ckpt_gc.py`'s module docstring (the tool's own reference text, also its
    `--help` description) and `safe_checkpoint.py`'s own module docstring + the
    `prune_run_dir_by_score` call site. A grep for `D64` in `safe_checkpoint.py` returned NOTHING
    before this cycle."""
    gc_doc = gc.__doc__
    assert "THE FREEZE COVERS THIS OFFLINE CLI ONLY" in gc_doc, "ckpt_gc.py module docstring"
    assert "safe_checkpoint.py" in gc_doc

    sc_src = open(os.path.join(REPO, "mcts_trader", "training", "safe_checkpoint.py")).read()
    assert "D64" in sc_src, "safe_checkpoint.py must name D64 — it did not before this cycle"
    assert "D64" in (sc.__doc__ or ""), "safe_checkpoint.py's MODULE docstring must name D64"
    assert "D64" in (sc.prune_run_dir_by_score.__doc__ or ""), \
        "the in-run deleter's own docstring must name D64"


# ==================================================== TASK_881D resume cycle 2: PIN THE D64 GATE
# `main()` is 382 lines — 10.0% of `tools/ckpt_gc.py`'s 3829 — and is allowlisted WHOLESALE by
# `_881D_KNOWN_REWRITES`, so nothing checks its content. Measured on this tree: 1549 of 3829 lines
# (40.5%) are inside allowlisted entities and thus content-unchecked; 2280 (59.5%) stay protected
# against unexplained removal. The D64 gate was living inline in `main()`, i.e. in the unpinned
# 40.5% — a safety gate nothing pins is one careless edit from vanishing with every test still
# green. It is now a top-level function, `_d64_apply_refusal_reason`, pinned to a HARDCODED
# sha256 of its exact AST entity text (a HEAD-diff pin is impossible: the function does not exist
# at HEAD at all), the same mechanism `AMENDED_INGESTION_NAMES` uses in the 881i suite.

D64_GATE_PINNED_NAMES = (
    "_d64_apply_refusal_reason",   # the WHOLE gate predicate
    "_D64_OVERRIDE_FLAG",
    "_D64_OVERRIDE_TOKEN",         # the exact literal that makes an abbreviation useless
    "_D64_OVERRIDE_DEST",
)


def _sha256_of_entity(src, name):
    import hashlib
    return hashlib.sha256(_top_level_entity(src, name).encode()).hexdigest()


# Captured at the moment this cycle shipped. An intentional, reviewed change updates the digest;
# an accidental one fails loudly. NOT self-derived (a digest computed from its own input would be
# a tautology) — see test_red_when_the_d64_gate_is_mutated_at_its_real_drop_site.
D64_GATE_DIGESTS = {
    # RE-PINNED cycle 3: the gate grew the EXACTLY-ONCE multiplicity rule (the split-token
    # BLOCKER — see `_D64_REPEATED_SHAPES` above). Cycle-2 digest was
    # d0ae868964b4c0f909664c43b06752a733ee63f6330a86406b53eb22f78191ab.
    #
    # RE-PINNED AGAIN by TASK_881E cycle 2 (2026-09-06). SANCTIONED CHANGE, and deliberately the
    # SMALLEST one that closes its BLOCKER: the four-line inline argv-token enumeration is now a
    # call to `_safety_flag_occurrences` (ckpt_gc.py's SAFETY-FLAG SPELLING DISCIPLINE section).
    # The RULE is unchanged — same tokens, same order, same `len(name) > 2` guard — it simply
    # lives in one place now, because `--no-bind-authority` shipped WITHOUT this discipline and
    # was armed by a bare `--n`, i.e. the recurrence, not the individual hole, was the defect.
    # Every behavioural test in this section, and every mutation anchor below, is untouched:
    # `occurrences`, the multiplicity rule and `spelled_in_full` keep their exact bytes.
    # Cycle-3 digest was e29f1a266709b45f3046fdd30fde38338718982905b7e431696e8230a4ea920c.
    "_d64_apply_refusal_reason":
        "f5eccefb9695f3cb65ec9e1f9aae5958c557402b302fa6a6b47bce7fed0dd886",
    "_D64_OVERRIDE_FLAG":
        "df7e8778cced5d9a2670315e3cc1c3ff5a4a3381813ee8696482fea479c9f8f6",
    "_D64_OVERRIDE_TOKEN":
        "9fc79ee4dead07ac706f35deb600e730a1c729cbb2251f6e2a6b4f294f836fe7",
    "_D64_OVERRIDE_DEST":
        "8072ed27da8dff52cfc38c25f36ed8e21f960f622f2641562f41e0af8314c347",
}


@pytest.mark.parametrize("name", D64_GATE_PINNED_NAMES)
def test_the_d64_gate_is_byte_identity_pinned(name):
    actual = _sha256_of_entity(open(GC_PATH).read(), name)
    assert actual == D64_GATE_DIGESTS[name], (
        f"the D64 `--apply` gate entity {name!r} drifted from its pin (now sha256={actual}). "
        f"If this is an intentional, reviewed change, update D64_GATE_DIGESTS — but read D64 "
        f"first: this is the code gate that keeps an irreversible deleter frozen.")


@pytest.mark.parametrize("name,old,new", [
    # Remove the abbreviation half of the gate — the exact regression that shipped as the BLOCKER.
    ("_d64_apply_refusal_reason",
     "        return None if spelled_in_full else _D64_ABBREVIATED_OVERRIDE_REFUSAL",
     "        return None"),
    # The two lines that close the cycle-3 split-token BLOCKER, each pinned. MEASURED, and stated
    # as measured rather than as assumed: severing EITHER one alone still refuses the bypass argv
    # (they are mutually redundant — see
    # test_the_cycle3_split_token_closure_is_doubly_redundant_and_both_halves_are_load_bearing,
    # which runs all three severances); severing BOTH restores the bypass and deletes checkpoints.
    # These two entries pin the BYTES; that test pins the BEHAVIOUR.
    ("_d64_apply_refusal_reason",
     "    if len(occurrences) > 1:\n        return _D64_REPEATED_OVERRIDE_REFUSAL",
     "    if False:\n        return _D64_REPEATED_OVERRIDE_REFUSAL"),
    ("_d64_apply_refusal_reason",
     "    spelled_in_full = occurrences == [_D64_OVERRIDE_FLAG]",
     "    spelled_in_full = _D64_OVERRIDE_FLAG in occurrences"),
    # Neuter the whole gate.
    ("_d64_apply_refusal_reason",
     '    if not getattr(a, "apply", False) or getattr(a, "verify_manifests", False):',
     "    if True:"),
    # Weaken the token to something a prefix-plus-guess could hit.
    ("_D64_OVERRIDE_TOKEN", '_D64_OVERRIDE_TOKEN = "I-HAVE-READ-D64"',
     '_D64_OVERRIDE_TOKEN = "y"'),
])
def test_red_when_the_d64_gate_is_mutated_at_its_real_drop_site(name, old, new):
    """MUTATION, at the gate's real drop site, against a COPY of the source string — the real
    tool is never edited. Proves the pin above is a check and not a tautology, and that each
    mutation really moves the pinned bytes."""
    src = open(GC_PATH).read()
    assert src.count(old) == 1, f"anchor not unique ({src.count(old)}): {old!r}"
    mutated = src.replace(old, new)
    assert _sha256_of_entity(mutated, name) != D64_GATE_DIGESTS[name], \
        f"mutating {name!r} at its real drop site did not move the pinned digest"


_D64_MULTIPLICITY_SEVERANCE = (
    "    if len(occurrences) > 1:\n        return _D64_REPEATED_OVERRIDE_REFUSAL",
    "    if False:\n        return _D64_REPEATED_OVERRIDE_REFUSAL")
_D64_SAME_OCCURRENCE_SEVERANCE = (
    "    spelled_in_full = occurrences == [_D64_OVERRIDE_FLAG]",
    "    spelled_in_full = _D64_OVERRIDE_FLAG in occurrences")   # cycle 2's `any()`, restored


@pytest.mark.parametrize("severed,expect_bypass", [
    ((), False),                                                   # the shipped tool: refuses
    ((_D64_MULTIPLICITY_SEVERANCE,), False),                       # redundant half #1 removed
    ((_D64_SAME_OCCURRENCE_SEVERANCE,), False),                    # redundant half #2 removed
    ((_D64_MULTIPLICITY_SEVERANCE, _D64_SAME_OCCURRENCE_SEVERANCE), True),   # BOTH -> bypass back
])
def test_the_cycle3_split_token_closure_is_doubly_redundant_and_both_halves_are_load_bearing(
        tmp_path, severed, expect_bypass):
    """BEHAVIOURAL mutation, on a COPY of the tool written into tmp_path — `tools/ckpt_gc.py` is
    never edited. Runs the reported split-token argv against four variants and asserts what each
    one really does to the checkpoints.

    Stated as MEASURED, because the tempting claim is wrong: severing the multiplicity rule alone
    does NOT restore the bypass. The two lines are mutually redundant — with only one occurrence
    permitted, `occurrences == [FLAG]` and `FLAG in occurrences` agree; with the equality in
    place, a second occurrence makes the list unequal and refuses as ABBREVIATED even when the
    count check is gone. Only removing BOTH brings the deletion back. The last row is the
    anti-tautology proof: without it the first three could pass on a tool that refuses everything.
    """
    src = open(GC_PATH).read()
    for old, new in severed:
        assert src.count(old) == 1, f"anchor not unique ({src.count(old)}): {old!r}"
        src = src.replace(old, new)
    variant = tmp_path / "ckpt_gc_variant.py"
    variant.write_text(src)
    run_dir = _d64_hostile_tree(tmp_path)
    # TASK_881E/D70 escape hatch: this mutation test is about the D64 override multiplicity
    # rule, not authority correlation — 'hostile_unrelated' is fixture convenience here too.
    r = subprocess.run([sys.executable, str(variant), "--apply"]
                       + [f"{D64_OVERRIDE_FLAG}=NOT_THE_TOKEN", f"--i={D64_OVERRIDE_TOKEN}"]
                       + _D64_APPLY_ARGS + ["--no-bind-authority"],
                       cwd=str(tmp_path), capture_output=True, text=True)
    if expect_bypass:
        assert r.returncode == 0, r.stdout + r.stderr
        assert "mode=APPLY" in r.stdout, r.stdout
        assert not _d64_tree_is_untouched(run_dir), \
            "severing BOTH halves must restore the measured bypass, else these anchors are not " \
            "what closes it and the pin is guarding the wrong lines"
    else:
        assert r.returncode == 2, r.stdout + r.stderr
        assert "mode=APPLY" not in r.stdout, r.stdout
        assert _d64_tree_is_untouched(run_dir), sorted(p.name for p in run_dir.iterdir())


def test_the_d64_gate_is_the_only_place_apply_is_armed():
    """Structural: `main()` must reach the freeze through the pinned function and nowhere else, so
    a future edit inside the unpinned 40.5% cannot quietly grow a second arming path. Exactly one
    call to `_d64_apply_refusal_reason`, and no other read of the override's dest anywhere in the
    file."""
    src = open(GC_PATH).read()
    assert src.count("_d64_apply_refusal_reason(") == 2, \
        "expected exactly the definition + one call site"
    # the override's dest is read ONLY inside the pinned gate
    gate = _top_level_entity(src, "_d64_apply_refusal_reason")
    assert src.count("_D64_OVERRIDE_DEST") == 2, \
        "the override's dest name must be referenced only by its definition and the pinned gate"
    assert "_D64_OVERRIDE_DEST" in gate


# ================================================ 881(parent) gate (i): no manifest -> FAIL LOUD

def test_gate_i_unresolved_run_dir_is_refused_dry_run(tmp_path):
    runs = tmp_path / "runs"
    run_dir = runs / "my_run"
    _mk_ckpt(str(run_dir), "checkpoint_100")
    (tmp_path / "experiments").mkdir()
    r = _run_gc(["--roots", "runs"], cwd=str(tmp_path))
    assert "REFUSE run dir" in r.stdout
    assert "no manifest declares coverage" in r.stdout
    assert "1 run dir(s) REFUSED" in r.stdout


def test_gate_i_unresolved_run_dir_nothing_deleted_under_apply(tmp_path):
    runs = tmp_path / "runs"
    run_dir = runs / "my_run"
    ck = _mk_ckpt(str(run_dir), "checkpoint_100")
    (tmp_path / "experiments").mkdir()
    r = _run_gc(["--roots", "runs", "--apply", D64_OVERRIDE, "--min-free-gb", "1e18"],
               cwd=str(tmp_path))
    assert os.path.isdir(ck), "an unresolved run dir must not be touched even under --apply"
    assert "REFUSE run dir" in r.stdout


def test_gate_i_covers_directive_resolves_with_nothing_to_protect(tmp_path):
    runs = tmp_path / "runs"
    run_dir = runs / "my_run"
    ck = _mk_ckpt(str(run_dir), "checkpoint_100")
    exp = tmp_path / "experiments" / "999_x"
    exp.mkdir(parents=True)
    (exp / "PROTECTED.txt").write_text(f"covers: runs/my_run\n")
    r = _run_gc(["--roots", "runs", "--min-free-gb", "1e18"], cwd=str(tmp_path))
    assert "REFUSE run dir" not in r.stdout
    assert os.path.isdir(ck)


def test_gate_i_protect_line_also_resolves_coverage(tmp_path):
    runs = tmp_path / "runs"
    run_dir = runs / "my_run"
    _mk_ckpt(str(run_dir), "checkpoint_100")
    exp = tmp_path / "experiments" / "999_x"
    exp.mkdir(parents=True)
    (exp / "PROTECTED.txt").write_text("protect: runs/my_run/checkpoint_100\n")
    r = _run_gc(["--roots", "runs", "--min-free-gb", "1e18"], cwd=str(tmp_path))
    assert "REFUSE run dir" not in r.stdout


def test_manifest_malformed_line_is_ignored_and_reported_not_silently_dropped(tmp_path, monkeypatch):
    """881I/D53 SETTLED SEMANTICS (re-scoped from this test's original stash-era premise): a
    malformed line no longer leaves the OTHER directives in the same file trusted — it WITHDRAWS
    THE WHOLE FILE (the cycle-3 fix this row consumes as settled, not re-opens). "Ignored and
    reported, not silently dropped" now means: the malformed line is named in the raised
    refusal's `.errors`, and NOTHING in the file — including the otherwise-well-formed
    `protect:` line — is trusted. `runs/foo/ck` is created on disk so the ONLY reason it fails
    to protect is the whole-file withdrawal, never a `.dangling` false negative."""
    monkeypatch.chdir(tmp_path)
    exp = tmp_path / "experiments" / "999_x"
    exp.mkdir(parents=True)
    os.makedirs("runs/foo/ck", exist_ok=True)
    (exp / "PROTECTED.txt").write_text("this is not a directive\nprotect: runs/foo/ck\n")
    with pytest.raises(gc.ManifestScanIncomplete) as ei:
        gc.load_manifests(["experiments"])
    assert "this is not a directive" in str(ei.value) or "this is not a directive" in \
        "; ".join(ei.value.errors)
    assert os.path.realpath("runs/foo/ck") not in ei.value.partial.protect


# ======================= TASK_881D gate (iii): 881I's malformed corpus, INTEGRATION ===========
# ("881I's malformed-input corpus still refuses identically when driven through 881D's callers
# — integration, not just unit.") Every malformed-manifest test above this point calls
# `gc.load_manifests` directly, in-process — real coverage of the PARSER, but it never proves
# the malformed-file refusal actually reaches `main()`'s real `--apply`/dry-run CLI path (the
# thing this row wires `load_manifests` into). These two drive the same malformed shapes through
# a REAL subprocess invocation of `tools/ckpt_gc.py` and check it refuses IDENTICALLY to what
# `--verify-manifests` already proves in `test_ckpt_881i_manifest_ingestion.py`: a dry run
# continues, labelled INCOMPLETE, with the malformed file's run dir REFUSED (no coverage); an
# `--apply` run refuses outright at exit 2 (the module docstring's own promise: "`--apply` is
# refused outright (exit 2) while any of the tree-wide ones is raised").

def test_gate_iii_malformed_protected_txt_refuses_through_dry_run(tmp_path):
    runs = tmp_path / "runs"
    run_dir = runs / "my_run"
    ck = _mk_ckpt(str(run_dir), "checkpoint_100")
    exp = tmp_path / "experiments" / "999_x"
    exp.mkdir(parents=True)
    # 881I's own canonical malformed shape: two tokens after the directive (the unanchored
    # subset-grant D53 closed) — the SAME shape `test_ckpt_881i_manifest_ingestion.py` exercises
    # against `--verify-manifests`, driven here through the dry-run delete path instead.
    (exp / "PROTECTED.txt").write_text("protect: runs/my_run/checkpoint_100 extra_token\n")
    r = _run_gc(["--roots", "runs", "--experiments-dir", "experiments", "--min-free-gb", "1e18"],
               cwd=str(tmp_path))
    # A bare dry-run is not required to exit non-zero on an INCOMPLETE manifest scan (only
    # `--apply` refuses outright at exit 2 — see the sibling `..._apply_outright` test below);
    # this test's claim is narrower: the malformed file's coverage is withdrawn and the scan is
    # LOUDLY labelled INCOMPLETE, never silently treated as "this run dir has no manifest".
    assert r.returncode == 0, r.stdout + r.stderr
    assert "CRITICAL" in r.stderr and "manifest scan INCOMPLETE" in r.stderr, r.stderr
    assert "dry-run continues with the INCOMPLETE manifest set" in r.stderr, r.stderr
    assert "REFUSE run dir" in r.stdout, r.stdout
    assert os.path.isdir(ck), "a malformed manifest must never un-refuse the run dir it covers"


def test_gate_iii_malformed_protected_txt_refuses_apply_outright(tmp_path):
    runs = tmp_path / "runs"
    run_dir = runs / "my_run"
    ck = _mk_ckpt(str(run_dir), "checkpoint_100")
    exp = tmp_path / "experiments" / "999_x"
    exp.mkdir(parents=True)
    (exp / "PROTECTED.txt").write_text("protect: runs/my_run/checkpoint_100 extra_token\n")
    r = _run_gc(["--roots", "runs", "--experiments-dir", "experiments", "--apply", D64_OVERRIDE,
               "--min-free-gb", "1e18"], cwd=str(tmp_path))
    assert r.returncode == 2, r.stdout + r.stderr
    assert "REFUSING to --apply" in r.stderr, r.stderr
    assert os.path.isdir(ck), "REFUSED --apply must delete nothing"


def test_spec_md_protected_block_is_parsed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    exp = tmp_path / "experiments" / "999_x"
    exp.mkdir(parents=True)
    # 881I/D53: a well-formed entry naming a path that DOES NOT EXIST lands in `.dangling`, not
    # `.protect`/`.covers` — so this parse-correctness fixture needs its targets to be real.
    os.makedirs("runs/foo/ck0", exist_ok=True)
    os.makedirs("runs/foo/ck1", exist_ok=True)
    os.makedirs("runs/bar", exist_ok=True)
    (exp / "SPEC.md").write_text(
        "# some doc\n\nprotected:\n  - runs/foo/ck0\n  - runs/foo/ck1\n\ncovers:\n  - runs/bar\n")
    ms = gc.load_manifests(["experiments"])
    assert os.path.realpath("runs/foo/ck0") in ms.protect
    assert os.path.realpath("runs/foo/ck1") in ms.protect
    assert os.path.realpath("runs/bar") in ms.covers


# ================================================== 881(parent) gate (ii): descendant-aware

def test_gate_ii_inner_path_protects_enclosing_checkpoint(tmp_path):
    runs = tmp_path / "runs"
    run_dir = runs / "my_run"
    enclosing = _mk_ckpt(str(run_dir), "snapshot_ep10")
    inner = os.path.join(enclosing, "checkpoint_500")
    os.makedirs(inner)
    newest = _mk_ckpt(str(run_dir), "snapshot_ep20", mtime=2_000_000)
    # Set the enclosing dir's mtime AFTER creating its child -- mkdir() bumps a parent dir's
    # own mtime, so setting it before would be silently overwritten back to "now".
    os.utime(enclosing, (1_000_000, 1_000_000))
    exp = tmp_path / "experiments" / "999_x"
    exp.mkdir(parents=True)
    # Manifest names a path INSIDE the checkpoint, exactly the form restore_checkpoint accepts.
    (exp / "PROTECTED.txt").write_text(f"protect: runs/my_run/snapshot_ep10/checkpoint_500\n")
    r = _run_gc(["--roots", "runs", "--apply", D64_OVERRIDE, "--keep-best", "1",
                "--keep-latest", "1", "--keep-unscored", "0", "--min-free-gb", "1e18"],
               cwd=str(tmp_path))
    assert os.path.isdir(enclosing), (
        "a manifest naming a path INSIDE a checkpoint must protect the enclosing checkpoint; "
        f"stdout={r.stdout} stderr={r.stderr}")
    assert os.path.isdir(newest)


def test_is_protected_ancestor_and_descendant(tmp_path):
    ck = str(tmp_path / "run" / "snapshot_ep10")
    os.makedirs(ck)
    inner = os.path.join(ck, "checkpoint_500")
    os.makedirs(inner)
    protect = {os.path.realpath(inner)}
    assert gc._is_protected(ck, protect), "descendant protection must reach upward"
    assert gc._is_protected(inner, protect)
    protect2 = {os.path.realpath(ck)}
    assert gc._is_protected(inner, protect2), "ancestor protection must reach downward"


# ================================================ 881(parent) gate (iii): both deleters agree

def test_gate_iii_offline_and_inrun_deleters_agree(tmp_path, monkeypatch):
    sc._reset_protection_cache_for_test()
    # Manifest paths are repo-relative (matching the real convention: `covers: runs/foo`
    # resolved from the repo root) -- chdir so both deleters resolve them the same way the
    # real trainer (run from the repo root) would.
    monkeypatch.chdir(tmp_path)
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "my_run"
    keep_new = _mk_ckpt(str(run_dir), "checkpoint_200", mtime=2_000_000)
    drop_old = _mk_ckpt(str(run_dir), "checkpoint_100", mtime=1_000_000)
    exp_root = tmp_path / "experiments"
    (exp_root / "999_x").mkdir(parents=True)
    (exp_root / "999_x" / "PROTECTED.txt").write_text(f"covers: runs/my_run\n")
    with open(os.path.join(str(run_dir), "scores.jsonl"), "a") as f:
        f.write(json.dumps({"epoch": 1, "score": 0.5, "path": keep_new, "step": 200}) + "\n")
        f.write(json.dumps({"epoch": 1, "score": 0.1, "path": drop_old, "step": 100}) + "\n")

    # In-run pruner, dry-run, pointed at the sandbox roots/experiments — never the real ones.
    # TASK_881E/D70 escape hatch: '999_x'/'my_run' is fixture convenience for gate (iii) parity,
    # not a real protection relationship — see this file's `_run_gc` comment.
    msgs, had_failure = sc.prune_run_dir_by_score(
        str(run_dir), roots=(str(runs_root),), scan_dirs=(str(exp_root),),
        keep_best=1, keep_latest=1, keep_unscored=0, apply=False, authority_binding=False)
    assert not had_failure
    in_run_would_delete = {m for m in msgs if "would delete" in m}
    assert any("checkpoint_100" in m for m in in_run_would_delete)
    assert not any("checkpoint_200" in m for m in in_run_would_delete)

    # Offline tool, same tree, same manifest dirs.
    r = _run_gc(["--roots", "runs", "--keep-best", "1", "--keep-latest", "1",
                "--keep-unscored", "0", "--min-free-gb", "1e18"], cwd=str(tmp_path))
    assert "would delete" in r.stdout
    assert "checkpoint_100" in r.stdout
    assert "checkpoint_200" not in r.stdout.split("would delete")[1] if "would delete" in r.stdout else True
    sc._reset_protection_cache_for_test()


def test_gate_iii_ttl_rescan_picks_up_manifest_added_mid_run(tmp_path, monkeypatch):
    sc._reset_protection_cache_for_test()
    monkeypatch.chdir(tmp_path)
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "my_run"
    _mk_ckpt(str(run_dir), "checkpoint_100")
    exp_root = tmp_path / "experiments"
    exp_root.mkdir()
    # No manifest yet -> refused.
    msgs, _ = sc.prune_run_dir_by_score(
        str(run_dir), roots=(str(runs_root),), scan_dirs=(str(exp_root),), ttl_seconds=0.05,
        apply=False, authority_binding=False)
    assert any("REFUSE" in m for m in msgs)

    # A manifest is dropped mid-"run" (SERVER.md's documented normal workflow).
    (exp_root / "999_x").mkdir(parents=True)
    (exp_root / "999_x" / "PROTECTED.txt").write_text("covers: runs/my_run\n")

    # Immediately after (within the TTL window) the cache is still fresh from the first attempt,
    # so `named` is served from cache and does not yet know about the new manifest. This test
    # drives the DRY-RUN path (`apply=False`), where that is a reporting lag and nothing more --
    # it is NOT "correctly empty", as this comment used to claim, it is STALE, and the earlier
    # wording is what let the dangerous direction go untested for a whole cycle: the same lag on
    # a DELETING call deleted a manifest-protected checkpoint (see
    # `test_a_deleting_call_at_the_default_ttl_honours_a_manifest_dropped_mid_run` at the bottom
    # of this file, which is the same scenario with `apply=True` and no `sleep` at all). Here the
    # tiny TTL is waited out on purpose, to assert only that a re-scan DOES pick the manifest up.
    time.sleep(0.06)
    msgs2, _ = sc.prune_run_dir_by_score(
        str(run_dir), roots=(str(runs_root),), scan_dirs=(str(exp_root),), ttl_seconds=0.05,
        apply=False, authority_binding=False)
    assert not any("REFUSE" in m for m in msgs2), (
        f"manifest added mid-run must be honoured within the TTL: {msgs2}")
    sc._reset_protection_cache_for_test()


def test_protection_cache_stays_last_known_good_on_scan_failure(tmp_path, monkeypatch):
    sc._reset_protection_cache_for_test()
    exp_root = tmp_path / "experiments"
    (exp_root / "999_x").mkdir(parents=True)
    # 881G/D53: entry paths resolve against the PROCESS'S CWD, not the manifest's own directory —
    # this test does not chdir, so the entry is written ABSOLUTE (avoids CWD-dependence entirely).
    # 881I/D53: a well-formed entry naming a path that does not exist lands in `.dangling`, not
    # `.covers` — the run dir must actually exist for `named` to be non-empty.
    (tmp_path / "runs" / "my_run").mkdir(parents=True)
    (exp_root / "999_x" / "PROTECTED.txt").write_text(f"covers: {tmp_path / 'runs' / 'my_run'}\n")
    named, protect, missing = sc.get_protection(roots=(str(tmp_path / "runs"),),
                                                scan_dirs=(str(exp_root),), ttl_seconds=1000)
    assert not sc.protection_cache_is_stale()
    assert len(named) == 1

    def _boom(*a, **k):
        return None
    monkeypatch.setattr(sc, "collect_protection", _boom)
    sc._PROTECTION_CACHE["ts"] = 0.0  # force a re-scan attempt
    named2, protect2, missing2 = sc.get_protection(roots=(str(tmp_path / "runs"),),
                                                   scan_dirs=(str(exp_root),), ttl_seconds=1000)
    assert sc.protection_cache_is_stale()
    assert named2 == named, "a failed re-scan must keep the last-known-good set"
    sc._reset_protection_cache_for_test()


# ================================================ 881(parent) gate (iv): advisory can only ADD

def test_gate_iv_ledger_cites_path_manifest_omits_still_kept(tmp_path):
    """Isolates the rescue mechanism with THREE siblings in one family: `newest` (kept trivially
    as the resume point -- proves the run wasn't refused outright), `cited` (older, kept ONLY by
    the advisory ledger citation), and `unprotected` (older still, named by nothing -- must
    actually be DELETED, proving the run really planned and executed deletions rather than
    vacuously keeping everything)."""
    runs = tmp_path / "runs"
    run_dir = runs / "my_run"
    unprotected = _mk_ckpt(str(run_dir), "ckpt_unprotected", mtime=1_000_000)
    cited = _mk_ckpt(str(run_dir), "ckpt_special", mtime=1_500_000)
    newest = _mk_ckpt(str(run_dir), "ckpt_newest", mtime=2_000_000)
    exp = tmp_path / "experiments" / "999_x"
    exp.mkdir(parents=True)
    # An explicit "covers" manifest for the run dir (satisfies gate i) that says NOTHING about
    # ckpt_special -- the only thing keeping it must be the ledger citation.
    (exp / "PROTECTED.txt").write_text("covers: runs/my_run\n")
    (exp / "SPEC.md").write_text("Some experiment. See `runs/my_run/ckpt_special` for weights.\n")
    r = _run_gc(["--roots", "runs", "--apply", D64_OVERRIDE, "--keep-best", "0",
                "--keep-latest", "1", "--keep-unscored", "0", "--min-free-gb", "1e18"],
               cwd=str(tmp_path))
    assert os.path.isdir(newest)
    assert os.path.isdir(cited), (
        f"an advisory ledger citation must rescue a checkpoint the manifest omits: {r.stdout}")
    assert not os.path.isdir(unprotected), (
        f"an unrelated, unprotected sibling must still be pruned (proves deletion actually "
        f"ran, not that everything survives vacuously): {r.stdout}")


def test_gate_iv_manifest_cites_path_ledger_omits_still_kept(tmp_path):
    """Same three-sibling isolation, but the ONLY protection on `manifest_only` is a manifest
    `protect:` line (no ledger/prose citation anywhere)."""
    runs = tmp_path / "runs"
    run_dir = runs / "my_run"
    unprotected = _mk_ckpt(str(run_dir), "ckpt_unprotected", mtime=1_000_000)
    manifest_only = _mk_ckpt(str(run_dir), "ckpt_special", mtime=1_500_000)
    newest = _mk_ckpt(str(run_dir), "ckpt_newest", mtime=2_000_000)
    exp = tmp_path / "experiments" / "999_x"
    exp.mkdir(parents=True)
    (exp / "PROTECTED.txt").write_text("protect: runs/my_run/ckpt_special\ncovers: runs/my_run\n")
    r = _run_gc(["--roots", "runs", "--apply", D64_OVERRIDE, "--keep-best", "0",
                "--keep-latest", "1", "--keep-unscored", "0", "--min-free-gb", "1e18"],
               cwd=str(tmp_path))
    assert os.path.isdir(newest)
    assert os.path.isdir(manifest_only), f"a manifest protect: line must be honoured: {r.stdout}"
    assert not os.path.isdir(unprotected), (
        f"an unrelated, unprotected sibling must still be pruned: {r.stdout}")


def test_advisory_scan_alone_never_satisfies_coverage(tmp_path):
    """Prose-only citation (no PROTECTED.txt/SPEC.md protected:/covers: block) must NOT resolve
    881(parent) gate (i) -- otherwise the exact defect this row exists to close resurfaces."""
    runs = tmp_path / "runs"
    run_dir = runs / "my_run"
    _mk_ckpt(str(run_dir), "checkpoint_100")
    exp = tmp_path / "experiments" / "999_x"
    exp.mkdir(parents=True)
    # A SPEC.md that mentions the path in prose but has no `protected:`/`covers:` block.
    (exp / "SPEC.md").write_text("This experiment trains on `runs/my_run/checkpoint_100`.\n")
    r = _run_gc(["--roots", "runs", "--min-free-gb", "1e18"], cwd=str(tmp_path))
    assert "REFUSE run dir" in r.stdout


# ============================================================== gate (v): failed rmtree

@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_gate_v_failed_rmtree_records_failed_prune_and_nonzero_exit(tmp_path):
    runs = tmp_path / "runs"
    run_dir = runs / "my_run"
    keep_new = _mk_ckpt(str(run_dir), "checkpoint_200", mtime=2_000_000)
    victim = _mk_ckpt(str(run_dir), "checkpoint_100")
    with open(os.path.join(victim, "blob"), "w") as f:
        f.write("x")
    # mtime set AFTER writing the blob file -- creating a file bumps its parent dir's own
    # mtime back to "now", which would otherwise make `victim` look newer than `keep_new`.
    os.utime(victim, (1_000_000, 1_000_000))
    exp = tmp_path / "experiments" / "999_x"
    exp.mkdir(parents=True)
    (exp / "PROTECTED.txt").write_text("covers: runs/my_run\n")
    # Strip all permissions on the victim so `shutil.rmtree` cannot list it after the rename
    # (the rename itself only needs write+exec on the PARENT dir, so it still succeeds).
    os.chmod(victim, 0o000)
    try:
        r = _run_gc(["--roots", "runs", "--apply", D64_OVERRIDE, "--keep-best", "1",
                    "--keep-latest", "1", "--keep-unscored", "0", "--min-free-gb", "1e18"],
                   cwd=str(tmp_path))
        assert r.returncode != 0, f"a failed rmtree must exit non-zero: {r.stdout} {r.stderr}"
        assert "FAILED" in r.stdout
        idx_path = os.path.join(str(run_dir), "checkpoints_index.json")
        assert os.path.exists(idx_path)
        idx = json.load(open(idx_path))
        victim_row = next(row for row in idx["checkpoints"]
                          if row["path"] == os.path.realpath(victim))
        assert victim_row["kept"] is True
        assert "DELETE FAILED" in victim_row["reason"]
    finally:
        # cleanup so tmp_path teardown can remove it -- the trash dir carries the
        # SUBPROCESS's pid, not this test's, so glob for it rather than guess the name.
        import glob as _glob
        for cand in [victim] + _glob.glob(f"{victim}.gc-trash-*"):
            if os.path.isdir(cand):
                os.chmod(cand, 0o755)


def test_finalize_index_rows_marks_rmtree_failure_explicitly():
    rows = [{"path": "/x/ck1", "kept": False, "reason": "score too low"},
            {"path": "/x/ck2", "kept": False, "reason": "score too low"}]
    out = gc.finalize_index_rows(rows, actually_deleted={"/x/ck2"}, rmtree_failed={"/x/ck1"})
    ck1 = next(r for r in out if r["path"] == "/x/ck1")
    ck2 = next(r for r in out if r["path"] == "/x/ck2")
    assert ck1["kept"] is True
    assert "DELETE FAILED" in ck1["reason"]
    assert ck2["kept"] is False


# ============================================================== SCOPE ADDITION: score reader

def test_score_reader_resolves_flax_checkpoint_step_dir_by_path(tmp_path):
    """The exact repro from the task banner: checkpoint_<step> names a STEP, not an epoch --
    the reader must resolve it via `path`, not by parsing the digits out of the basename."""
    run_dir = tmp_path / "run"
    ck = _mk_ckpt(str(run_dir), "checkpoint_9000")  # step 9000, NOT epoch 9000
    with open(os.path.join(str(run_dir), "scores.jsonl"), "w") as f:
        f.write(json.dumps({"epoch": 3, "score": 0.75, "metric": "reward", "path": ck,
                            "step": 9000}) + "\n")
    by_path, by_epoch = gc._load_scores(str(run_dir), None, "*.log")
    assert by_path[os.path.realpath(ck)].score == 0.75
    # And it must NOT leak into the epoch map (a different `checkpoint_3` dir must not steal it).
    assert 3 not in by_epoch or by_epoch[3].score != 0.75 or True  # path rows never populate by_epoch
    assert 3 not in by_epoch


def test_score_reader_resolves_safe_manager_checkpoint_epoch_dir(tmp_path):
    run_dir = tmp_path / "run"
    ck = _mk_ckpt(str(run_dir), "checkpoint_epoch_7")
    with open(os.path.join(str(run_dir), "scores.jsonl"), "w") as f:
        f.write(json.dumps({"epoch": 7, "score": 0.42, "metric": "avg_reward", "path": ck}) + "\n")
    runs = {str(run_dir): [ck]}
    deletions, summary, index_rows, refused = gc.plan(
        runs, lambda rd: gc._load_scores(rd, None, "*.log"), keep_best=1, keep_latest=1,
        under_floor=False, keep_unscored=0)
    assert not refused
    row = index_rows[str(run_dir)][0]
    assert row["score"] == 0.42
    assert row["family"] == "safe_manager"


def test_is_ckpt_dir_does_not_collapse_a_run_dir_with_real_checkpoint_subdirs(tmp_path):
    """The `1 ckpts, 0 scored` bug: train_jax_fast.py writes `checkpoint_<step>` dirs straight
    under the run dir; the fallback used to match the RUN dir itself as one opaque checkpoint."""
    run_dir = tmp_path / "jax_run"
    _mk_ckpt(str(run_dir), "checkpoint_100")
    _mk_ckpt(str(run_dir), "checkpoint_200")
    assert not gc._is_ckpt_dir(str(run_dir)), "a dir holding checkpoint subdirs is a RUN dir"
    runs = gc.collect([str(tmp_path)])
    assert str(run_dir) in runs
    assert len(runs[str(run_dir)]) == 2


def test_never_delete_basenames_checkpoint_golden(tmp_path):
    run_dir = tmp_path / "run"
    golden = _mk_ckpt(str(run_dir), "checkpoint_golden", mtime=1)
    newer_scored_out = _mk_ckpt(str(run_dir), "checkpoint_epoch_5", mtime=2)
    with open(os.path.join(str(run_dir), "scores.jsonl"), "w") as f:
        f.write(json.dumps({"epoch": 5, "score": -999, "path": newer_scored_out}) + "\n")
    runs = {str(run_dir): [golden, newer_scored_out]}
    deletions, summary, index_rows, refused = gc.plan(
        runs, lambda rd: gc._load_scores(rd, None, "*.log"), keep_best=0, keep_latest=0,
        under_floor=True, keep_unscored=0)
    assert not refused
    deleted_paths = {p for p, _b, _r in deletions}
    assert os.path.realpath(golden) not in deleted_paths


# ============================== TASK_881D resume, correctness-lens MAJOR: `--keep-best` is a
# PER-METRIC cap, not a global one. Direction chosen: (b), an HONEST disclosure — see plan()'s
# docstring. These reproduce the two measured shapes from the D64/correctness-lens finding and
# assert the (unchanged, deliberately over-retentive) survivor counts plus the new disclosure.

def _mk_scored(run_dir, name, mtime, score, metric):
    p = _mk_ckpt(str(run_dir), name, mtime=mtime)
    return p, {"epoch": mtime, "score": score, "metric": metric, "path": p}


def test_major_keep_best_is_per_metric_not_global_two_metrics_8_and_2(tmp_path):
    """10 ckpts scored under 2 metrics (8 + 2, exactly train_jax_fast.py's own
    train_reward_ma20 / eval_reward_L(n) shape), `--keep-best 3 --keep-latest 1` -> 5 survive,
    not 3. Over-retention, unchanged by this row's fix (fail-safe direction); what changes is
    that the summary line now says so."""
    run_dir = tmp_path / "run"
    rows = []
    train_paths = []
    for i, score in enumerate([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]):
        p, row = _mk_scored(run_dir, f"checkpoint_epoch_{i}", i + 1, score, "train_reward_ma20")
        train_paths.append(p)
        rows.append(row)
    eval_paths = []
    for i, score in enumerate([50.0, 49.0]):
        p, row = _mk_scored(run_dir, f"checkpoint_epoch_{8 + i}", 9 + i, score, "eval_reward_L4")
        eval_paths.append(p)
        rows.append(row)
    with open(os.path.join(str(run_dir), "scores.jsonl"), "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    all_ckpts = train_paths + eval_paths
    runs = {str(run_dir): all_ckpts}
    deletions, summary, index_rows, refused = gc.plan(
        runs, lambda rd: gc._load_scores(rd, None, "*.log"), keep_best=3, keep_latest=1,
        under_floor=True, keep_unscored=0)
    assert not refused
    assert len(deletions) == 5, [r for _p, _b, r in deletions]
    assert len(all_ckpts) - len(deletions) == 5

    # the three BEST-scored of the 8-member train group survive, and BOTH eval-group members
    # survive (only 2 exist, both within top-3 of that metric) -- 3 + 2 = 5.
    deleted_paths = {os.path.realpath(p) for p, _b, _r in deletions}
    survived = {os.path.realpath(p) for p in all_ckpts} - deleted_paths
    expected = {os.path.realpath(p) for p in train_paths[:3]} | {
        os.path.realpath(p) for p in eval_paths}
    assert survived == expected

    line = next(s for s in summary if s.startswith(str(run_dir)))
    assert "keeping 5" in line, line
    assert "PER METRIC across 2 metrics" in line, line
    assert "up to 6 may survive" in line, line


def test_major_keep_best_is_per_metric_not_global_five_distinct_metrics(tmp_path):
    """Extreme reachable form: 5 ckpts, 5 distinct metrics, `--keep-best 1` -> ALL 5 survive."""
    run_dir = tmp_path / "run"
    rows = []
    paths = []
    for i in range(5):
        p, row = _mk_scored(run_dir, f"checkpoint_epoch_{i}", i + 1, float(i), f"metric_{i}")
        paths.append(p)
        rows.append(row)
    with open(os.path.join(str(run_dir), "scores.jsonl"), "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    runs = {str(run_dir): paths}
    deletions, summary, index_rows, refused = gc.plan(
        runs, lambda rd: gc._load_scores(rd, None, "*.log"), keep_best=1, keep_latest=1,
        under_floor=True, keep_unscored=0)
    assert not refused
    assert deletions == [], deletions

    line = next(s for s in summary if s.startswith(str(run_dir)))
    assert "keeping 5" in line, line
    assert "PER METRIC across 5 metrics" in line, line
    assert "up to 5 may survive" in line, line


def test_major_single_metric_summary_line_carries_no_per_metric_caveat(tmp_path):
    """The disclosure clause must NOT appear when there is only one metric in the run dir --
    `--keep-best N` really is the cap there, and saying otherwise would be its own false claim."""
    run_dir = tmp_path / "run"
    rows = []
    paths = []
    for i, score in enumerate([3.0, 2.0, 1.0]):
        p, row = _mk_scored(run_dir, f"checkpoint_epoch_{i}", i + 1, score, "reward")
        paths.append(p)
        rows.append(row)
    with open(os.path.join(str(run_dir), "scores.jsonl"), "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    runs = {str(run_dir): paths}
    _deletions, summary, _rows, refused = gc.plan(
        runs, lambda rd: gc._load_scores(rd, None, "*.log"), keep_best=1, keep_latest=1,
        under_floor=True, keep_unscored=0)
    assert not refused
    line = next(s for s in summary if s.startswith(str(run_dir)))
    assert "PER METRIC" not in line, line


# ============================== TASK_881D resume, correctness-lens MINOR: exact-score ties are
# broken by a documented deterministic recency rule (newest survives), reusing
# `_order_newest_first` rather than the filesystem-order-dependent stable sort.

def test_minor_exact_score_tie_break_is_recency_not_input_order(tmp_path):
    """4 exactly-tied checkpoints, `--keep-best 2 --keep-latest 1`. Constructed so the INPUT
    order (the `ckpts` list handed to `plan()`, standing in for `collect()`'s unsorted
    `os.walk`/`os.listdir` order) is OLDEST-FIRST -- the exact shape that used to keep the two
    OLDEST under Python's stable sort. The fix must keep the two NEWEST regardless."""
    run_dir = tmp_path / "run"
    oldest = _mk_ckpt(str(run_dir), "checkpoint_epoch_0", mtime=1_000_000)
    second = _mk_ckpt(str(run_dir), "checkpoint_epoch_1", mtime=1_000_100)
    third = _mk_ckpt(str(run_dir), "checkpoint_epoch_2", mtime=1_000_200)
    newest = _mk_ckpt(str(run_dir), "checkpoint_epoch_3", mtime=1_000_300)
    with open(os.path.join(str(run_dir), "scores.jsonl"), "w") as f:
        for ep, p in enumerate([oldest, second, third, newest]):
            f.write(json.dumps({"epoch": ep, "score": 5.0, "metric": "r", "path": p}) + "\n")

    # INPUT order is oldest-first -- the adversarial shape for a stable, input-order tiebreak.
    runs = {str(run_dir): [oldest, second, third, newest]}
    deletions, summary, index_rows, refused = gc.plan(
        runs, lambda rd: gc._load_scores(rd, None, "*.log"), keep_best=2, keep_latest=1,
        under_floor=True, keep_unscored=0)
    assert not refused
    deleted_paths = {os.path.realpath(p) for p, _b, _r in deletions}
    assert deleted_paths == {os.path.realpath(oldest), os.path.realpath(second)}, (
        f"the NEWEST two of an exact tie must survive: deleted={deleted_paths}")
    assert os.path.realpath(third) not in deleted_paths
    assert os.path.realpath(newest) not in deleted_paths


def test_minor_tie_break_reuses_order_newest_first_not_a_second_ordering_rule(tmp_path):
    """Cross-check: the tie-break's own ranking of the 4 tied checkpoints must be IDENTICAL to
    `_order_newest_first`'s own ordering of the same 4 paths -- proving the fix reuses that
    function rather than inventing an independent one that could disagree with it."""
    run_dir = tmp_path / "run"
    a = _mk_ckpt(str(run_dir), "checkpoint_epoch_0", mtime=5_000_000)
    b = _mk_ckpt(str(run_dir), "checkpoint_epoch_1", mtime=5_000_050)
    c = _mk_ckpt(str(run_dir), "checkpoint_epoch_2", mtime=5_000_100)
    with open(os.path.join(str(run_dir), "scores.jsonl"), "w") as f:
        for ep, p in enumerate([a, b, c]):
            f.write(json.dumps({"epoch": ep, "score": 1.0, "metric": "r", "path": p}) + "\n")

    expected_order = gc._order_newest_first([a, b, c])
    runs = {str(run_dir): [a, b, c]}       # deliberately NOT in newest-first input order
    _deletions, _summary, index_rows, refused = gc.plan(
        runs, lambda rd: gc._load_scores(rd, None, "*.log"), keep_best=1, keep_latest=1,
        under_floor=True, keep_unscored=0)
    assert not refused
    kept_row = next(r for r in index_rows[str(run_dir)] if r["kept"] and
                    r["path"] == os.path.realpath(expected_order[0]))
    assert kept_row is not None


# ============================================================== disputed `_order_newest_first`

def test_order_newest_first_third_measurement_original_tie_bug(tmp_path):
    """Original motivating bug: same whole-second mtime, alphabetical fallback used to delete
    the true newest (highest step) checkpoint."""
    p100 = _mk_ckpt(str(tmp_path), "checkpoint_100", mtime=1_000_000)
    p150 = _mk_ckpt(str(tmp_path), "checkpoint_150", mtime=1_000_000)
    p200 = _mk_ckpt(str(tmp_path), "checkpoint_200", mtime=1_000_000)
    order = gc._order_newest_first([p100, p150, p200])
    assert os.path.basename(order[0]) == "checkpoint_200"


def test_order_newest_first_third_measurement_disputed_cross_family_bug(tmp_path):
    """THIRD INDEPENDENT MEASUREMENT (by execution) of the D33-disputed finding: an earlier
    candidate implementation assigned each family member a mtime BORROWED from its own family's
    sorted slots by rank, which could make a genuinely-newer checkpoint in one family look older
    than a genuinely-staler one in another. Confirmed real here: `checkpoint_200` (highest step,
    but the OLDEST real mtime of the three) must NOT be ranked newest, and the true newest
    (`checkpoint_100`, real mtime T+999000) must not be the one recency-pruned."""
    p200 = _mk_ckpt(str(tmp_path), "checkpoint_200", mtime=1_000_001)   # real-oldest, highest step
    p10 = _mk_ckpt(str(tmp_path), "snapshot_ep10", mtime=1_000_300)     # real-middle
    p100 = _mk_ckpt(str(tmp_path), "checkpoint_100", mtime=1_001_000)   # real-newest, lower step
    order = gc._order_newest_first([p200, p10, p100])
    assert os.path.basename(order[0]) == "checkpoint_100", (
        f"the true newest-by-mtime file must sort first, got order={[os.path.basename(p) for p in order]}")
    assert os.path.basename(order[-1]) == "checkpoint_200"

    runs = {str(tmp_path): [p200, p10, p100]}
    deletions, summary, _rows, _refused = gc.plan(
        runs, lambda rd: ({}, {}), keep_best=1, keep_latest=1, under_floor=True,
        keep_unscored=0)
    deleted_names = {os.path.basename(p) for p, _b, _r in deletions}
    assert "checkpoint_100" not in deleted_names, (
        "the true newest checkpoint on disk must never be the one deleted under the floor")
    assert "checkpoint_200" in deleted_names, (
        "the real-stalest checkpoint should be the one recency-pruned, not the newest")


# ==================================== BLOCKER: a path the scan cannot read is silently dropped
#
# This row's own named historical BLOCKER, reproduced end-to-end under --apply in cycle 2 (two
# judges, independently): a symlink that the scan declined was never recorded as an error, so
# `ManifestScanIncomplete` / `ProtectScanIncomplete` never raised, the protect set silently
# shrank, and a protected checkpoint was DELETED with exit 0 and no stderr.
#
# Every test below runs the REAL tool end-to-end under --apply against a sandbox and asserts on
# what is on disk afterwards. Each was confirmed RED against the unfixed code (see the task
# report): variants A and B both deleted the protected checkpoint and exited 0.

def _seed_two_ckpts(tmp_path):
    """A run dir with an old and a new checkpoint. Under --keep-latest 1 --keep-unscored 0 with
    the disk floor tripped, the OLD one is the deletion candidate."""
    run_dir = tmp_path / "runs" / "my_run"
    old = _mk_ckpt(str(run_dir), "checkpoint_100", mtime=1_000_000)
    new = _mk_ckpt(str(run_dir), "checkpoint_200", mtime=2_000_000)
    return run_dir, old, new


_APPLY = ["--roots", "runs", "--apply", D64_OVERRIDE, "--keep-best", "0", "--keep-latest", "1",
          "--keep-unscored", "0", "--min-free-gb", "1e18"]


def test_blocker_variant_a_symlinked_manifest_file_refuses_instead_of_deleting(tmp_path):
    """Variant A: `experiments/999_x/SPEC.md` is a SYMLINK. Coverage is legitimately satisfied by
    a real PROTECTED.txt, so the run dir is NOT refused by 881(parent) gate (i) — the only thing
    standing
    between `checkpoint_100` and deletion is the citation inside the symlinked SPEC.md, which
    the old `if os.path.islink(fp): continue` dropped without a word."""
    _run, old, new = _seed_two_ckpts(tmp_path)
    exp = tmp_path / "experiments" / "999_x"
    exp.mkdir(parents=True)
    (exp / "PROTECTED.txt").write_text("covers: runs/my_run\n")
    real_spec = tmp_path / "outside" / "SPEC_real.md"
    real_spec.parent.mkdir()
    real_spec.write_text("Weights live in `runs/my_run/checkpoint_100`.\n")
    os.symlink(str(real_spec), str(exp / "SPEC.md"))

    r = _run_gc(_APPLY, cwd=str(tmp_path))
    assert os.path.isdir(old), (
        f"a symlinked manifest must never be silently skipped into a deletion: "
        f"stdout={r.stdout} stderr={r.stderr}")
    assert os.path.isdir(new)
    assert r.returncode == 2, f"--apply must be refused, got {r.returncode}"
    assert "manifest scan INCOMPLETE" in r.stderr
    assert "SPEC.md" in r.stderr and "SYMLINK" in r.stderr


def test_blocker_variant_b_symlinked_experiments_dir_refuses_instead_of_deleting(tmp_path):
    """Variant B: `experiments/999_x` is a symlinked DIRECTORY holding the `protect:` line.
    `os.walk(followlinks=False)` omits it and `onerror` never fires, so the whole manifest was
    invisible. A second, REAL manifest dir supplies coverage so 881(parent) gate (i) does not
    mask the bug."""
    _run, old, new = _seed_two_ckpts(tmp_path)
    exp = tmp_path / "experiments"
    (exp / "998_real").mkdir(parents=True)
    (exp / "998_real" / "PROTECTED.txt").write_text("covers: runs/my_run\n")
    hidden = tmp_path / "outside" / "999_x"
    hidden.mkdir(parents=True)
    (hidden / "PROTECTED.txt").write_text("protect: runs/my_run/checkpoint_100\n")
    os.symlink(str(hidden), str(exp / "999_x"))

    r = _run_gc(_APPLY, cwd=str(tmp_path))
    assert os.path.isdir(old), (
        f"a manifest behind a symlinked directory must never be silently invisible: "
        f"stdout={r.stdout} stderr={r.stderr}")
    assert os.path.isdir(new)
    assert r.returncode == 2
    assert "SYMLINKED DIRECTORY" in r.stderr


def test_symlinked_advisory_artifact_is_READ_not_silently_dropped(tmp_path):
    """The other half of the read-vs-refuse decision, and the one that proves it is a DECISION
    rather than a blanket refusal: a symlinked NON-manifest artifact is FOLLOWED, because the
    advisory scan can only ADD rescues. Its citation must actually rescue the checkpoint — which
    also means the tool still applies (exit 0) instead of refusing the whole tree over a doc."""
    _run, old, new = _seed_two_ckpts(tmp_path)
    exp = tmp_path / "experiments" / "999_x"
    exp.mkdir(parents=True)
    (exp / "PROTECTED.txt").write_text("covers: runs/my_run\n")
    real_doc = tmp_path / "outside" / "result_real.md"
    real_doc.parent.mkdir()
    real_doc.write_text("Best weights: `runs/my_run/checkpoint_100`.\n")
    os.symlink(str(real_doc), str(exp / "result.md"))

    r = _run_gc(_APPLY, cwd=str(tmp_path))
    assert r.returncode == 0, f"a symlinked advisory doc must not block --apply: {r.stderr}"
    assert os.path.isdir(old), (
        f"the citation inside a symlinked advisory artifact must rescue: {r.stdout} {r.stderr}")
    assert os.path.isdir(new)


def test_dangling_symlinked_advisory_artifact_is_an_incompleteness_not_a_silent_skip(tmp_path):
    """Following is not the same as ignoring failures: a symlink that cannot be resolved is a
    path the scan FAILED to read, and must refuse rather than shrink the advisory set."""
    exp = tmp_path / "experiments" / "999_x"
    exp.mkdir(parents=True)
    os.symlink(str(tmp_path / "nowhere.md"), str(exp / "result.md"))
    (tmp_path / "runs").mkdir()
    with pytest.raises(gc.ProtectScanIncomplete) as ei:
        gc.scan_referenced_paths([str(exp.parent)], [str(tmp_path / "runs")])
    assert "result.md" in str(ei.value)


def test_self_referential_symlinked_dir_is_not_a_false_refusal(tmp_path, monkeypatch):
    """Cycle detection, and a REAL case: this repo ships
    `experiments/832_linear_imprint/ckpt_smoke/ckpt_smoke -> .../ckpt_smoke`. A symlink pointing
    at a directory the scan already covered can hide nothing, so it must NOT be reported as
    incompleteness — otherwise the fix for the blocker becomes a permanent denial of service on
    the real tree."""
    monkeypatch.chdir(tmp_path)   # manifest paths are repo-root-relative by convention
    exp = tmp_path / "experiments" / "999_x"
    (exp / "sub").mkdir(parents=True)
    # 881I/D53: a well-formed entry naming a path that DOES NOT EXIST lands in `.dangling`, not
    # `.protect` — create the real checkpoint dir so this test exercises cycle detection, not
    # the (separately-tested) dangling-entry path.
    os.makedirs("runs/my_run/checkpoint_100", exist_ok=True)
    (exp / "sub" / "PROTECTED.txt").write_text("protect: runs/my_run/checkpoint_100\n")
    os.symlink(str(exp / "sub"), str(exp / "sub" / "loop"))
    ms = gc.load_manifests([str(tmp_path / "experiments")])
    assert os.path.realpath(str(tmp_path / "runs/my_run/checkpoint_100")) in ms.protect

    # ...and the same on the REAL repo tree, which is where the self-loop actually lives.
    # 881G/D53: entry paths resolve against the PROCESS'S CWD, not the manifest's own directory
    # or the scan dir — this test's own `monkeypatch.chdir(tmp_path)` above is still in effect,
    # so the real corpus's relative `protect: runs/...` entries would resolve under `tmp_path`
    # (dangling) rather than under `REPO`, exactly the CWD trap the grammar warns authors about.
    # chdir back to REPO before reading REPO's own manifests.
    monkeypatch.chdir(REPO)
    real = gc.load_manifests([os.path.join(REPO, "experiments")])
    assert real.protect, "the real experiments/ manifests must still resolve"


# ============================== MAJOR: offline scan and in-run pruner disagree about a symlink

def test_collect_refuses_symlinked_dir_discovered_under_a_root(tmp_path):
    """`os.walk` skips a symlinked child, so `--roots runs` reported `0 checkpoints` for a
    symlinked `runs/my_run` with no diagnostic at all."""
    moved = tmp_path / "moved" / "my_run"
    _mk_ckpt(str(moved), "checkpoint_100")
    _mk_ckpt(str(moved), "checkpoint_200")
    (tmp_path / "runs").mkdir()
    os.symlink(str(moved), str(tmp_path / "runs" / "my_run"))
    with pytest.raises(gc.DiscoveryScanIncomplete) as ei:
        gc.collect([str(tmp_path / "runs")])
    assert "my_run" in str(ei.value) and "symlinked DIRECTORY" in str(ei.value)


def test_explicit_checkpoint_root_is_RESOLVED_not_refused(tmp_path):
    """MAJOR D: an explicitly-passed CHECKPOINT root RESOLVES through its symlink.

    `jax_ckpt_fast` is this repo's documented default root and IS a symlink to
    `/mnt/scratch`; refusing it refused the normal case, and globally — one symlinked
    root disabled reclaim for the entirely healthy `runs/` tree too."""
    moved = tmp_path / "moved" / "ckroot"
    _mk_ckpt(str(moved / "run_a"), "checkpoint_100")
    link = tmp_path / "ckroot"
    os.symlink(str(moved), str(link))
    found = gc.collect([str(link)])                      # must NOT raise
    assert list(found) == [os.path.join(str(moved), "run_a")], found
    assert gc.root_resolution(str(link)) == (str(moved), True)


def test_resolved_root_landing_outside_every_configured_root_is_an_ESCAPE(tmp_path):
    """Resolving an honest spelling is not the same as following a link out of the world this
    tool may touch. `runs/my_run -> /elsewhere` still REFUSES."""
    moved = tmp_path / "moved" / "my_run"
    _mk_ckpt(str(moved), "checkpoint_100")
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    link = runs_root / "my_run"
    os.symlink(str(moved), str(link))
    with pytest.raises(gc.DiscoveryScanIncomplete) as ei:
        gc.collect([str(link)], allowed_roots=[str(runs_root)])
    assert "ESCAPE" in str(ei.value)


def test_resolved_root_into_a_forbidden_component_is_an_ESCAPE(tmp_path):
    """The other escape: FORBIDDEN_PARTS. A root resolving into a `backups`/`trading_data`/
    `.git`/`fixtures` component refuses however honest the spelling looks."""
    moved = tmp_path / "backups" / "my_run"
    _mk_ckpt(str(moved), "checkpoint_100")
    link = tmp_path / "ckroot"
    os.symlink(str(moved), str(link))
    with pytest.raises(gc.DiscoveryScanIncomplete) as ei:
        gc.collect([str(link)])
    assert "ESCAPE" in str(ei.value) and "backups" in str(ei.value)


def test_major_d_parity_offline_and_inrun_agree_on_a_symlinked_checkpoint_root(tmp_path,
                                                                               monkeypatch):
    """THE PARITY PROOF MAJOR D demands, in the exact scratch disk shape: the ROOT is a symlink,
    the run dir beneath it is real. Executed against cycle 2's code this was the asymmetry the
    fail-closed judge named — `ckpt_gc.py --apply` refused the whole tree with
    `DiscoveryScanIncomplete` (exit 2, deleting NOTHING) while `prune_run_dir_by_score` on the
    SAME tree deleted, i.e. the refusal did not prevent deletions, it MIGRATED them from the
    supervised tool to the unsupervised per-save one. Measured under `--apply` on both sides for
    that reason: in dry run cycle 2 looked fine, because the offline refusal still reported from
    its partial result. Both deleters must now remove the SAME checkpoint."""
    sc._reset_protection_cache_for_test()
    monkeypatch.chdir(tmp_path)
    real_root = tmp_path / "media" / "ckroot"
    run_dir = real_root / "run_a"
    old = _mk_ckpt(str(run_dir), "checkpoint_100", mtime=1_000_000)
    new = _mk_ckpt(str(run_dir), "checkpoint_200", mtime=2_000_000)
    os.symlink(str(real_root), str(tmp_path / "ckroot"))     # the `jax_ckpt_fast` shape
    exp_root = tmp_path / "experiments"
    (exp_root / "999_x").mkdir(parents=True)
    (exp_root / "999_x" / "PROTECTED.txt").write_text("covers: ckroot/run_a\n")
    argv = ["--roots", "ckroot", "--apply", D64_OVERRIDE, "--keep-best", "0", "--keep-latest", "1",
            "--keep-unscored", "0", "--min-free-gb", "1e18"]

    r = _run_gc(argv, cwd=str(tmp_path))
    assert r.returncode == 0, f"the supervised tool must not refuse: {r.stdout} {r.stderr}"
    assert "(resolved)" in r.stdout, f"the resolution must be DISCLOSED: {r.stdout}"
    assert "REFUSE run dir" not in r.stdout, r.stdout
    assert not os.path.isdir(old) and os.path.isdir(new), r.stdout

    _mk_ckpt(str(run_dir), "checkpoint_100", mtime=1_000_000)
    sc._reset_protection_cache_for_test()
    msgs, had_failure = sc.prune_run_dir_by_score(
        str(tmp_path / "ckroot" / "run_a"), roots=(str(tmp_path / "ckroot"),),
        scan_dirs=(str(exp_root),), keep_best=0, keep_latest=1, keep_unscored=0,
        under_floor=True, apply=True, authority_binding=False)
    assert not had_failure, msgs
    assert not any("REFUSE" in m for m in msgs), msgs
    assert not os.path.isdir(old) and os.path.isdir(new), (
        f"the two deleters must agree on the same tree: {msgs}")
    sc._reset_protection_cache_for_test()


def test_inrun_pruner_refuses_a_symlinked_run_dir_the_offline_tool_cannot_see(tmp_path, monkeypatch):
    """End-to-end on the deleter that actually deletes: with the run dir symlinked OUT of the
    configured roots, the offline tool sees NOTHING under `runs` while this pruner (which passes
    the run dir as a root) used to plan a deletion. It must refuse, and nothing may be removed.

    MAJOR D narrowed the reason without weakening the outcome: the refusal is now the ESCAPE
    check (`/tmp/.../moved/my_run` is outside every configured root), not a blanket
    "the root is a symlink" — an honest spelling INSIDE the roots resolves, see
    `test_major_d_parity_offline_and_inrun_agree_on_a_symlinked_checkpoint_root`."""
    sc._reset_protection_cache_for_test()
    monkeypatch.chdir(tmp_path)
    moved = tmp_path / "moved" / "my_run"
    old = _mk_ckpt(str(moved), "checkpoint_100", mtime=1_000_000)
    new = _mk_ckpt(str(moved), "checkpoint_200", mtime=2_000_000)
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    link = runs_root / "my_run"
    os.symlink(str(moved), str(link))
    exp_root = tmp_path / "experiments"
    (exp_root / "999_x").mkdir(parents=True)
    (exp_root / "999_x" / "PROTECTED.txt").write_text("covers: runs/my_run\n")

    msgs, had_failure = sc.prune_run_dir_by_score(
        str(link), roots=(str(runs_root),), scan_dirs=(str(exp_root),),
        keep_best=0, keep_latest=1, keep_unscored=0, under_floor=True, apply=True,
        authority_binding=False)
    assert not had_failure
    assert any("ESCAPE" in m for m in msgs), msgs
    assert os.path.isdir(old) and os.path.isdir(new), (
        f"the in-run pruner must delete nothing the offline tool cannot even see: {msgs}")
    sc._reset_protection_cache_for_test()


# ================================ MAJOR: a torn scores.jsonl line is swallowed, best gets pruned

def test_torn_scores_row_refuses_the_run_dir_and_the_best_checkpoint_survives(tmp_path):
    """End-to-end under --apply. `scores.jsonl` is appended with NO locking, so a torn row is
    realistic; the peak checkpoint's row is the torn one. Unfixed, the bare `except Exception:
    continue` swallowed it, the true best (42.0) read as UNSCORED and was recency-pruned under
    the disk floor with exit 0 and no trace."""
    run_dir = tmp_path / "runs" / "my_run"
    best = _mk_ckpt(str(run_dir), "snapshot_ep100", mtime=1_000_000)
    mid = _mk_ckpt(str(run_dir), "snapshot_ep50", mtime=1_500_000)
    newest = _mk_ckpt(str(run_dir), "snapshot_ep10", mtime=2_000_000)
    exp = tmp_path / "experiments" / "999_x"
    exp.mkdir(parents=True)
    (exp / "PROTECTED.txt").write_text("covers: runs/my_run\n")
    with open(os.path.join(str(run_dir), "scores.jsonl"), "w") as f:
        f.write('{"epoch": 100, "score": 42.0, "metric":\n')             # TORN — the true best
        f.write(json.dumps({"epoch": 50, "score": 1.0, "metric": "r", "path": mid}) + "\n")
        f.write(json.dumps({"epoch": 10, "score": 2.0, "metric": "r", "path": newest}) + "\n")

    r = _run_gc(_APPLY, cwd=str(tmp_path))
    assert os.path.isdir(best), (
        f"an unreadable score row must never turn the best checkpoint into a deletion "
        f"candidate: stdout={r.stdout} stderr={r.stderr}")
    assert os.path.isdir(mid) and os.path.isdir(newest)
    assert "REFUSED" in r.stdout and "scores.jsonl:1" in r.stdout, r.stdout


def test_load_scores_names_the_offending_file_and_line(tmp_path):
    """The diagnostic discipline the rest of this file already uses: name it, never `continue`."""
    run_dir = tmp_path / "run"
    ck = _mk_ckpt(str(run_dir), "snapshot_ep1")
    with open(os.path.join(str(run_dir), "scores.jsonl"), "w") as f:
        f.write(json.dumps({"epoch": 1, "score": 1.0, "path": ck}) + "\n")
        f.write("{not json\n")
    with pytest.raises(gc.CanonScoreReadIncomplete) as ei:
        gc._load_scores(str(run_dir), None, "*.log")
    assert "scores.jsonl:2" in str(ei.value)


def test_unreadable_checkpoints_index_is_also_a_refusal_not_a_shrug(tmp_path):
    """`checkpoints_index.json` is a score source too, and a corrupt one shrinks the score set
    exactly the same way — it used to be a `[warn]` that still let the deletion proceed."""
    run_dir = tmp_path / "run"
    _mk_ckpt(str(run_dir), "snapshot_ep1")
    with open(os.path.join(str(run_dir), "checkpoints_index.json"), "w") as f:
        f.write('{"checkpoints": [')
    with pytest.raises(gc.CanonScoreReadIncomplete):
        gc._load_scores(str(run_dir), None, "*.log")


def test_intact_scores_file_still_resolves_normally(tmp_path):
    """The refusal must be caused by the torn row, not by the run dir having scores at all."""
    run_dir = tmp_path / "run"
    ck = _mk_ckpt(str(run_dir), "snapshot_ep1")
    with open(os.path.join(str(run_dir), "scores.jsonl"), "w") as f:
        f.write(json.dumps({"epoch": 1, "score": 1.25, "metric": "r", "path": ck}) + "\n")
    by_path, _by_epoch = gc._load_scores(str(run_dir), None, "*.log")
    assert by_path[os.path.realpath(ck)].score == 1.25


# ============================ MAJOR: the CLI's CONDITIONAL refusal of --ckpt-retention score

def test_retention_policy_unavailable_when_ckpt_gc_is_missing_from_the_deployment(tmp_path):
    """The refusal's INPUT, exercised for real and unstubbed: a partial deploy (a copied-out
    trainer whose sibling `tools/ckpt_gc.py` did not come along) must report UNAVAILABLE.
    `retention_policy_available` resolves the policy path relative to its own __file__, so a
    genuine copy into a tree with no `tools/` is the honest way to produce that state."""
    dest = tmp_path / "deploy" / "mcts_trader" / "training"
    dest.mkdir(parents=True)
    src = os.path.join(REPO, "mcts_trader", "training", "safe_checkpoint.py")
    shutil.copy2(src, str(dest / "safe_checkpoint.py"))
    assert not os.path.exists(str(tmp_path / "deploy" / "tools" / "ckpt_gc.py"))

    spec = importlib.util.spec_from_file_location("_copied_sc", str(dest / "safe_checkpoint.py"))
    copied = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(copied)
    ok, where = copied.retention_policy_available()
    assert ok is False, "a deployment without tools/ckpt_gc.py must report UNAVAILABLE"
    assert "ckpt_gc.py" in where
    # ...and the in-run pruner degrades to a REFUSAL, never an unprotected prune.
    msgs, had_failure = copied.prune_run_dir_by_score(str(tmp_path / "nope"), apply=True)
    assert not had_failure
    assert any("UNAVAILABLE" in m for m in msgs), msgs


def test_cli_refuses_ckpt_retention_score_when_the_policy_module_is_unavailable(monkeypatch):
    """The refusal BRANCH itself (mcts_trader/training/cli.py). It converts what used to be an
    UNCONDITIONAL refusal into a conditional one, and it explicitly must not fall back to
    'recency' silently — so it needs a test, or a future edit deletes the guard invisibly."""
    from mcts_trader.training.cli import parse_args
    monkeypatch.setattr(sc, "retention_policy_available",
                        lambda: (False, "forced-unavailable-for-test"))
    monkeypatch.setattr(sys, "argv", ["train_jax_fast.py", "--ckpt-retention", "score"])
    with pytest.raises(SystemExit) as ei:
        parse_args()
    assert ei.value.code != 0, "argparse's error() must make this a non-zero exit, not a warning"


def test_cli_refusal_does_not_silently_fall_back_to_recency(monkeypatch, capsys):
    """The specific inversion the guard exists to prevent: returning a config that says
    'recency' would look like success to every caller."""
    from mcts_trader.training.cli import parse_args
    monkeypatch.setattr(sc, "retention_policy_available",
                        lambda: (False, "forced-unavailable-for-test"))
    monkeypatch.setattr(sys, "argv", ["train_jax_fast.py", "--ckpt-retention", "score"])
    with pytest.raises(SystemExit):
        parse_args()
    err = capsys.readouterr().err
    assert "UNAVAILABLE" in err and "ckpt-retention score" in err


def test_cli_accepts_score_when_the_policy_module_is_loadable(monkeypatch):
    """The ACCEPT branch, for contrast — the guard must be conditional, not a resurrected
    blanket refusal."""
    from mcts_trader.training.cli import parse_args
    assert sc.retention_policy_available()[0]
    monkeypatch.setattr(sys, "argv", ["train_jax_fast.py", "--ckpt-retention", "score"])
    assert parse_args().ckpt_retention == "score"


def test_in_run_prune_call_site_cannot_kill_a_healthy_training_run():
    """The startup availability check promises that an unavailable/failing policy must not crash
    an otherwise healthy multi-hour run. That promise has to be enforced at the per-save CALL
    SITE, not only at startup. Asserted structurally (the closure lives inside `main()` and only
    manifests across a full training run): the `prune_run_dir_by_score(...)` call must sit inside
    a `try`, and the handler must set the non-zero-exit flag rather than swallow the failure."""
    src = open(os.path.join(REPO, "train_jax_fast.py")).read()
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "_maybe_prune_by_score")
    tries = [n for n in ast.walk(fn) if isinstance(n, ast.Try)]
    guarded = [t for t in tries
               if any(isinstance(c, ast.Call) and getattr(c.func, "id", None) ==
                      "prune_run_dir_by_score" for c in ast.walk(t))]
    assert guarded, "prune_run_dir_by_score is not wrapped — an opportunistic prune can kill the run"
    handlers = [h for t in guarded for h in t.handlers]
    assert handlers, "the wrapping try has no except handler"
    assert any(getattr(h.type, "id", None) == "Exception" for h in handlers), (
        "the handler must catch the UNKNOWN failure, which is the whole point")
    body = ast.get_source_segment(src, handlers[0]) or ""
    assert "ckpt_prune_failed = True" in body, (
        "a swallowed prune failure must still make the run exit non-zero (gate v)")


# ============================================================== 4th deletion site (scope 5)

def test_safe_checkpoint_manager_default_off_and_bounded():
    """Scope item 5: the SafeCheckpointManager's own keep_best_n rmtree is documented as
    deliberately outside the protection model (default OFF, bounded, legacy path) rather than
    silently left unaddressed."""
    import inspect
    src = inspect.getsource(sc.SafeCheckpointManager.save_checkpoint)
    assert "TASK_881 scope item 5" in src
    assert "keep_best_n" in src


# ================= BLOCKER A: the refusal destroys the score data, then deletes the peak
#
# Cycle-3 findings. Every test below runs the REAL tool end-to-end and asserts on what is on
# disk afterwards; each was confirmed RED by reverting the fix at its real drop site (see the
# task report for the executed proofs).

def _seed_scored_run(tmp_path, scores=((10, 1.0), (20, 42.0), (30, 2.0))):
    """A covered run dir whose PEAK (ep20, 42.0) is neither the newest nor the oldest."""
    run_dir = tmp_path / "runs" / "my_run"
    mt = {10: 1_000_000, 20: 1_500_000, 30: 2_000_000}
    paths = {ep: _mk_ckpt(str(run_dir), f"snapshot_ep{ep}") for ep, _ in scores}
    exp = tmp_path / "experiments" / "999_x"
    exp.mkdir(parents=True)
    (exp / "PROTECTED.txt").write_text("covers: runs/my_run\n")
    with open(os.path.join(str(run_dir), "scores.jsonl"), "w") as f:
        for ep, s in scores:
            f.write(json.dumps({"epoch": ep, "score": s, "metric": "r",
                                "path": paths[ep]}) + "\n")
    for ep in paths:
        os.utime(paths[ep], (mt[ep], mt[ep]))
    return run_dir, paths


def _index_scores(run_dir):
    idx = json.load(open(os.path.join(str(run_dir), "checkpoints_index.json")))
    return {os.path.basename(r["path"]): r.get("score") for r in idx["checkpoints"]}


def test_blocker_a_refusal_does_not_rewrite_the_index_and_the_peak_survives_the_repair(tmp_path):
    """THE reproduction, in full, exactly as the judges executed it.

    1. a healthy `--write-index` run records `snapshot_ep20 = 42.0` in `checkpoints_index.json`
       (which `_load_scores` reads as a SCORE SOURCE);
    2. ep20's `scores.jsonl` row is torn mid-write (no locking, realistic) -> `--apply`
       correctly REFUSES the run dir... and used to rewrite the index to all-`None`;
    3. the operator follows `CanonScoreReadIncomplete`'s OWN documented remediation ("repair or
       truncate the offending line and re-run") -> with the index destroyed, ep20 reads UNSCORED
       and is recency-pruned, exit 0. The peak is gone, and the tool's own fix-instructions are
       what deleted it.

    The refusal write is the entire difference: with the index intact step 3 reports
    "3 scored, keeping 3"."""
    keep = ["--roots", "runs", "--keep-best", "1", "--keep-latest", "1",
            "--keep-unscored", "0", "--min-free-gb", "1e18"]
    run_dir, paths = _seed_scored_run(tmp_path)
    r1 = _run_gc(keep + ["--write-index"], cwd=str(tmp_path))
    assert _index_scores(run_dir)["snapshot_ep20"] == 42.0, r1.stdout

    lines = open(os.path.join(str(run_dir), "scores.jsonl")).read().splitlines()
    lines[1] = '{"epoch": 20, "score": 42.0, "metric":'          # torn mid-write
    open(os.path.join(str(run_dir), "scores.jsonl"), "w").write("\n".join(lines) + "\n")

    r2 = _run_gc(keep + ["--apply", D64_OVERRIDE], cwd=str(tmp_path))
    assert "REFUSED" in r2.stdout, r2.stdout
    assert _index_scores(run_dir)["snapshot_ep20"] == 42.0, (
        f"a REFUSAL must be SIDE-EFFECT-FREE — it rewrote the index it refuses to trust: "
        f"{_index_scores(run_dir)}")

    # step 3: the documented remediation.
    open(os.path.join(str(run_dir), "scores.jsonl"), "w").write(
        "\n".join(lines[:1] + lines[2:]) + "\n")
    r3 = _run_gc(keep + ["--apply", D64_OVERRIDE], cwd=str(tmp_path))
    assert os.path.isdir(paths[20]), (
        f"the tool's own remediation must not delete the peak: {r3.stdout}")
    assert "3 scored" in r3.stdout, r3.stdout


def test_blocker_a_write_index_dry_run_never_wipes_scores_of_an_unresolved_run_dir(tmp_path):
    """No corruption needed, and this is the DEFAULT state of most run dirs: a scored index and
    NO manifest. `--write-index` reads as audit-only (and `--apply` implies it), the tool prints
    "Nothing under runs/... will be pruned" — and used to wipe every recorded score on the way
    out, because the UNRESOLVED rows it wrote carry no `score` key at all."""
    run_dir, _paths = _seed_scored_run(tmp_path)
    keep = ["--roots", "runs", "--min-free-gb", "1e18"]
    _run_gc(keep + ["--write-index"], cwd=str(tmp_path))
    before = _index_scores(run_dir)
    assert before["snapshot_ep20"] == 42.0

    os.remove(str(tmp_path / "experiments" / "999_x" / "PROTECTED.txt"))   # coverage removed
    r = _run_gc(keep + ["--write-index"], cwd=str(tmp_path))
    assert "REFUSE run dir" in r.stdout
    assert _index_scores(run_dir) == before, (
        f"an UNRESOLVED run dir must not be written to at all: {_index_scores(run_dir)}")


def test_blocker_a_inrun_pruner_refusal_also_leaves_the_index_alone(tmp_path, monkeypatch):
    """881(parent) gate (iii): the same rule on the deleter that runs unsupervised, once per save. Its
    `write_run_index(run_dir, index_rows.get(run_dir, []), ...)` wrote an EMPTY checkpoint list
    over the index whenever `plan()` refused."""
    sc._reset_protection_cache_for_test()
    monkeypatch.chdir(tmp_path)
    run_dir, _paths = _seed_scored_run(tmp_path)
    _run_gc(["--roots", "runs", "--write-index", "--min-free-gb", "1e18"], cwd=str(tmp_path))
    before = _index_scores(run_dir)
    assert before["snapshot_ep20"] == 42.0

    lines = open(os.path.join(str(run_dir), "scores.jsonl")).read().splitlines()
    lines[1] = '{"epoch": 20, "score": 42.0, "metric":'
    open(os.path.join(str(run_dir), "scores.jsonl"), "w").write("\n".join(lines) + "\n")

    msgs, had_failure = sc.prune_run_dir_by_score(
        str(run_dir), roots=(str(tmp_path / "runs"),),
        scan_dirs=(str(tmp_path / "experiments"),),
        keep_best=1, keep_latest=1, keep_unscored=0, under_floor=True, apply=True,
        authority_binding=False)
    assert not had_failure
    assert any("REFUSED" in m for m in msgs), msgs
    assert _index_scores(run_dir) == before, (
        f"the in-run pruner's refusal must be side-effect-free too: {_index_scores(run_dir)}")
    sc._reset_protection_cache_for_test()


def test_blocker_a_write_run_index_is_atomic(tmp_path):
    """The index is a SCORE SOURCE, so an interrupted write manufactures exactly the torn index
    that triggers its own refusal. Asserted on the mechanism: the destination never exists in a
    partial state, because the bytes are written to a temp file and `os.replace`d."""
    d = tmp_path / "run"
    d.mkdir()
    dest = d / "checkpoints_index.json"
    gc.write_run_index(str(d), [{"path": "/x/ck", "kept": True, "score": 1.0}], {"generated": "t"})
    seen = []
    real_replace = os.replace

    def _spy(src, dst):
        seen.append((os.path.exists(dst), json.load(open(dst))["checkpoints"][0]["score"]))
        return real_replace(src, dst)

    os.replace = _spy
    try:
        gc.write_run_index(str(d), [{"path": "/x/ck", "kept": True, "score": 2.0}],
                           {"generated": "t2"})
    finally:
        os.replace = real_replace
    assert seen == [(True, 1.0)], (
        "at the instant before the swap the OLD index must still be complete and readable — "
        f"a truncating open() would have destroyed it already: {seen}")
    assert json.load(open(dest))["checkpoints"][0]["score"] == 2.0
    assert not [p for p in os.listdir(str(d)) if ".tmp-" in p], "temp file left behind"


# ================= BLOCKER B: a missing --protect-file is silently "nothing to protect"

def test_blocker_b_missing_protect_file_is_a_hard_error_not_an_empty_set(tmp_path):
    """`--protect-file` was PROMOTED into 881(parent) gate (i)'s coverage predicate this cycle and is
    documented as "treated as explicit and tracked, like a manifest". A missing MANIFEST refuses;
    a mistyped `--protect-file` used to read as an empty set and delete everything with exit 0
    and zero stderr."""
    _run, old, new = _seed_two_ckpts(tmp_path)
    (tmp_path / "experiments").mkdir()
    (tmp_path / "protect_real.txt").write_text("runs/my_run/checkpoint_100\n")
    r = _run_gc(_APPLY + ["--protect-file", "protect_TYPO.txt"], cwd=str(tmp_path))
    assert r.returncode == 2, f"a named-but-absent protect file must refuse --apply: {r.stdout}"
    assert "protect_TYPO.txt" in r.stderr and "CRITICAL" in r.stderr
    assert os.path.isdir(old) and os.path.isdir(new), r.stdout


def test_blocker_b_symlinked_protect_file_gets_the_manifest_treatment(tmp_path):
    """A symlinked protect file can un-refuse a run dir exactly like a `covers:` line, so it is
    REFUSED, not silently followed — the same reasoning `load_manifests` applies."""
    _run, old, new = _seed_two_ckpts(tmp_path)
    (tmp_path / "experiments").mkdir()
    real = tmp_path / "outside" / "p.txt"
    real.parent.mkdir()
    real.write_text("runs/my_run\n")
    os.symlink(str(real), str(tmp_path / "protect.txt"))
    r = _run_gc(_APPLY + ["--protect-file", "protect.txt"], cwd=str(tmp_path))
    assert r.returncode == 2, r.stdout
    assert "SYMLINK" in r.stderr
    assert os.path.isdir(old) and os.path.isdir(new)


def test_protect_file_that_exists_still_protects_and_covers(tmp_path):
    """The ACCEPT branch — the guard must be a check, not a resurrected blanket refusal."""
    _run, old, new = _seed_two_ckpts(tmp_path)
    (tmp_path / "experiments").mkdir()
    (tmp_path / "p.txt").write_text("# a comment\nruns/my_run/checkpoint_100\n")
    r = _run_gc(_APPLY + ["--protect-file", "p.txt"], cwd=str(tmp_path))
    assert r.returncode == 0, f"{r.stdout} {r.stderr}"
    assert os.path.isdir(old) and os.path.isdir(new)


# ============ BLOCKER C: the symlinked-manifest refusal is bypassed by rooting AT the symlink

def test_blocker_c_symlinked_scan_root_is_refused_not_read(tmp_path):
    """Both judges found this independently. The manifest lives in an UNTRACKED, out-of-tree
    directory; `experiments/999_x` links to it. The default `--experiments-dir experiments`
    refuses correctly — and the refusal message actively suggests the spelling that bypassed it:
    `--experiments-dir experiments/999_x` made `os.walk`'s "always list the top path" behaviour
    read that out-of-tree manifest as AUTHORITATIVE, its `covers:` line un-refused the run dir,
    and the checkpoint was deleted with exit 0."""
    _run, old, new = _seed_two_ckpts(tmp_path)
    hidden = tmp_path / "untracked" / "999_x"
    hidden.mkdir(parents=True)
    (hidden / "PROTECTED.txt").write_text("covers: runs/my_run\n")
    (tmp_path / "experiments").mkdir()
    os.symlink(str(hidden), str(tmp_path / "experiments" / "999_x"))

    r = _run_gc(_APPLY + ["--experiments-dir", "experiments/999_x"], cwd=str(tmp_path))
    assert r.returncode == 2, f"rooting the scan AT the symlink must not launder it: {r.stdout}"
    assert "manifest scan INCOMPLETE" in r.stderr
    assert os.path.isdir(old) and os.path.isdir(new), r.stdout


def test_blocker_c_a_real_scan_root_still_reads_normally(tmp_path, monkeypatch):
    """...and the fix must not make every scan root refuse."""
    monkeypatch.chdir(tmp_path)
    exp = tmp_path / "experiments" / "999_x"
    exp.mkdir(parents=True)
    _mk_ckpt(str(tmp_path / "runs" / "my_run"), "checkpoint_100")
    (exp / "PROTECTED.txt").write_text("protect: runs/my_run/checkpoint_100\n")
    ms = gc.load_manifests([str(exp)])
    assert os.path.realpath("runs/my_run/checkpoint_100") in ms.protect


# ================= MAJOR E: a `protect:` line naming nothing grants coverage and protects nothing

def test_major_e_dangling_protect_line_refuses_the_run_dir(tmp_path):
    """One typo un-refuses the run dir AND voids the protection in the same stroke: the author
    wrote `checkpoint_10` for `snapshot_ep10`, the entry realpath'd fine, `resolve_coverage` only
    asked whether SOME named path lay under the run dir — and `snapshot_ep10` was deleted with
    exit 0 and no diagnostic. A malformed line is reported at file:line; this well-formed one was
    not reported at all."""
    runs = tmp_path / "runs"
    run_dir = runs / "my_run"
    victim = _mk_ckpt(str(run_dir), "snapshot_ep10", mtime=1_000_000)
    newest = _mk_ckpt(str(run_dir), "snapshot_ep20", mtime=2_000_000)
    exp = tmp_path / "experiments" / "999_x"
    exp.mkdir(parents=True)
    (exp / "PROTECTED.txt").write_text("protect: runs/my_run/checkpoint_10\n")
    r = _run_gc(_APPLY, cwd=str(tmp_path))
    assert os.path.isdir(victim), (
        f"a manifest entry that names nothing must not un-refuse the run dir: {r.stdout}")
    assert os.path.isdir(newest)
    assert "NAMES NOTHING" in r.stderr, r.stderr
    assert "REFUSE run dir" in r.stdout, r.stdout


def test_major_e_dangling_entry_withdraws_coverage_even_beside_a_valid_covers(tmp_path):
    """The stronger half, and the reason a WARNING is not enough: coverage is legitimately
    granted by `covers:`, so 881(parent) gate (i) is satisfied and the deleter runs — against a
    protect set
    the author believes is in force. The dangling entry WITHDRAWS coverage rather than merely
    printing."""
    runs = tmp_path / "runs"
    run_dir = runs / "my_run"
    victim = _mk_ckpt(str(run_dir), "snapshot_ep10", mtime=1_000_000)
    _mk_ckpt(str(run_dir), "snapshot_ep20", mtime=2_000_000)
    exp = tmp_path / "experiments" / "999_x"
    exp.mkdir(parents=True)
    (exp / "PROTECTED.txt").write_text(
        "covers: runs/my_run\nprotect: runs/my_run/checkpoint_10\n")
    r = _run_gc(_APPLY, cwd=str(tmp_path))
    assert os.path.isdir(victim), r.stdout
    assert "REFUSE run dir" in r.stdout, r.stdout


def test_major_e_a_dangling_entry_naming_no_discovered_run_dir_is_only_a_warning(tmp_path):
    """Scoped, not global: the shipped `experiments/828_stage3_train/PROTECTED.txt` names three
    `jax_ckpt_fast/...` paths that do not currently resolve (scratch disk unmounted). Those refuse
    NOTHING, because no discovered run dir lies under them — otherwise one stale manifest would
    disable reclaim for the whole healthy tree, the global-refusal failure mode MAJOR D rejects."""
    _run, old, new = _seed_two_ckpts(tmp_path)
    exp = tmp_path / "experiments" / "999_x"
    exp.mkdir(parents=True)
    (exp / "PROTECTED.txt").write_text(
        "covers: runs/my_run\nprotect: jax_ckpt_fast/long_gone_run\n")
    r = _run_gc(_APPLY, cwd=str(tmp_path))
    assert "NAMES NOTHING" in r.stderr
    assert r.returncode == 0, f"{r.stdout} {r.stderr}"
    assert os.path.isdir(new)
    assert not os.path.isdir(old), f"the healthy run dir must still be prunable: {r.stdout}"


def test_major_e_inrun_pruner_withdraws_coverage_on_the_same_dangling_entry(tmp_path,
                                                                            monkeypatch):
    """881(parent) gate (iii): a dangling manifest entry must not mean one deleter refuses while the other
    prunes — `missing` is carried through `collect_protection`/`get_protection` for that reason."""
    sc._reset_protection_cache_for_test()
    monkeypatch.chdir(tmp_path)
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "my_run"
    _mk_ckpt(str(run_dir), "snapshot_ep10", mtime=1_000_000)
    _mk_ckpt(str(run_dir), "snapshot_ep20", mtime=2_000_000)
    exp_root = tmp_path / "experiments"
    (exp_root / "999_x").mkdir(parents=True)
    (exp_root / "999_x" / "PROTECTED.txt").write_text(
        "covers: runs/my_run\nprotect: runs/my_run/checkpoint_10\n")
    msgs, _ = sc.prune_run_dir_by_score(
        str(run_dir), roots=(str(runs_root),), scan_dirs=(str(exp_root),),
        keep_best=0, keep_latest=1, keep_unscored=0, under_floor=True, apply=True)
    assert any("REFUSE" in m for m in msgs), msgs
    assert os.path.isdir(str(run_dir / "snapshot_ep10")), msgs
    sc._reset_protection_cache_for_test()


# ============================================================== the cheap ones

def test_minor_covers_block_inside_a_spec_md_does_not_advisory_rescue(tmp_path):
    """The two documented manifest FORMS behaved OPPOSITELY: the prose scan skipped
    `PROTECTED.txt` but not `SPEC.md`, so a `covers:` block inside a SPEC.md — which MEANS
    "reviewed, nothing here needs protecting" — was also a path-shaped string, advisory-rescued
    the run dir, and made every checkpoint under it immortal via ancestor protection."""
    _run, old, new = _seed_two_ckpts(tmp_path)
    exp = tmp_path / "experiments" / "999_x"
    exp.mkdir(parents=True)
    (exp / "SPEC.md").write_text("# exp\n\ncovers:\n  - runs/my_run\n\nsome prose.\n")
    r = _run_gc(_APPLY, cwd=str(tmp_path))
    assert os.path.isdir(new)
    assert not os.path.isdir(old), (
        f"a `covers:` block must not rescue what it declares needs no rescue: {r.stdout}")


def test_minor_spec_md_prose_citation_outside_the_block_still_rescues(tmp_path):
    """...and stripping the block must not disarm the advisory net 881(parent) gate (iv) tests."""
    _run, old, new = _seed_two_ckpts(tmp_path)
    exp = tmp_path / "experiments" / "999_x"
    exp.mkdir(parents=True)
    (exp / "SPEC.md").write_text(
        "# exp\n\ncovers:\n  - runs/my_run\n\nWeights: `runs/my_run/checkpoint_100`.\n")
    r = _run_gc(_APPLY, cwd=str(tmp_path))
    assert os.path.isdir(old), f"prose outside the block is still advisory: {r.stdout}"
    assert os.path.isdir(new)


def test_minor_apply_exits_nonzero_when_a_run_dir_was_refused_for_an_unreadable_score(tmp_path):
    """The module docstring promised `--apply` is refused while any incompleteness is raised;
    `plan()` swallowed `CanonScoreReadIncomplete` per run dir and `main()` never learned, so a run
    whose best checkpoint was unevaluable exited 0 — indistinguishable, to any cron, from a clean
    prune. The per-run-dir blast radius is deliberate, so the EXIT STATUS is what is made true."""
    run_dir, _paths = _seed_scored_run(tmp_path)
    lines = open(os.path.join(str(run_dir), "scores.jsonl")).read().splitlines()
    lines[1] = '{"epoch": 20, "score": 42.0, "metric":'
    open(os.path.join(str(run_dir), "scores.jsonl"), "w").write("\n".join(lines) + "\n")
    r = _run_gc(["--roots", "runs", "--apply", D64_OVERRIDE, "--min-free-gb", "1e18"],
               cwd=str(tmp_path))
    assert r.returncode != 0, f"a partial prune must not look like a clean one: {r.stdout}"
    assert "REFUSALS" in r.stderr, r.stderr


def test_minor_non_utf8_scores_file_is_a_designed_refusal_not_a_traceback(tmp_path):
    """`UnicodeDecodeError` is a `ValueError`, not an `OSError`, and it is raised by the file
    ITERATION, not the per-line json handler — so a non-UTF-8 `scores.jsonl` escaped the designed
    per-run-dir refusal, aborted the whole plan with a traceback, and raised straight out of
    `prune_run_dir_by_score` against its own "never raises for a designed refusal" docstring."""
    run_dir = tmp_path / "run"
    ck = _mk_ckpt(str(run_dir), "snapshot_ep1")
    with open(os.path.join(str(run_dir), "scores.jsonl"), "wb") as f:
        f.write(json.dumps({"epoch": 1, "score": 1.0, "path": ck}).encode() + b"\n")
        f.write(b'{"epoch": 2, "score": \xff\xfe 3.0}\n')
    with pytest.raises(gc.CanonScoreReadIncomplete):
        gc._load_scores(str(run_dir), None, "*.log")


def test_minor_non_utf8_scores_file_does_not_raise_out_of_the_inrun_pruner(tmp_path,
                                                                          monkeypatch):
    sc._reset_protection_cache_for_test()
    monkeypatch.chdir(tmp_path)
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "my_run"
    _mk_ckpt(str(run_dir), "snapshot_ep1", mtime=1_000_000)
    _mk_ckpt(str(run_dir), "snapshot_ep2", mtime=2_000_000)
    exp_root = tmp_path / "experiments"
    (exp_root / "999_x").mkdir(parents=True)
    (exp_root / "999_x" / "PROTECTED.txt").write_text("covers: runs/my_run\n")
    with open(os.path.join(str(run_dir), "scores.jsonl"), "wb") as f:
        f.write(b'{"epoch": 1, "score": \xff\xfe 3.0}\n')
    msgs, had_failure = sc.prune_run_dir_by_score(          # must not raise
        str(run_dir), roots=(str(runs_root),), scan_dirs=(str(exp_root),),
        keep_best=0, keep_latest=1, keep_unscored=0, under_floor=True, apply=True,
        authority_binding=False)
    assert any("REFUSED" in m for m in msgs), msgs
    assert os.path.isdir(str(run_dir / "snapshot_ep1"))
    sc._reset_protection_cache_for_test()


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_nit_rename_failure_is_counted_in_the_failure_line(tmp_path):
    """`had_failure` was set by a failed RENAME but the summary only ever printed
    `len(rmtree_failed)` — "FAILED: 0 delete(s) left orphaned"."""
    runs = tmp_path / "runs"
    run_dir = runs / "my_run"
    _mk_ckpt(str(run_dir), "snapshot_ep20", mtime=2_000_000)
    victim = _mk_ckpt(str(run_dir), "snapshot_ep10", mtime=1_000_000)
    exp = tmp_path / "experiments" / "999_x"
    exp.mkdir(parents=True)
    (exp / "PROTECTED.txt").write_text("covers: runs/my_run\n")
    os.chmod(str(run_dir), 0o555)          # rename needs write on the PARENT -> EACCES
    try:
        r = _run_gc(_APPLY, cwd=str(tmp_path))
        assert r.returncode != 0, r.stdout
        assert "rename failed" in r.stdout, r.stdout
        assert "0 delete(s) left orphaned bytes and 1 could not be renamed" in r.stdout, r.stdout
        assert os.path.isdir(victim)
    finally:
        os.chmod(str(run_dir), 0o755)


# ============================================================================================
# TASK_881D CYCLE 2 — the fail-closed panel's findings, each with its dangerous direction and a
# mutation at the real drop site.
# ============================================================================================

def _mutant_sc(tmp_path, name, old, new):
    """A COPY of `mcts_trader/training/safe_checkpoint.py` with one exact substring replaced,
    importable as its own module (its own `_PROTECTION_CACHE`).

    Laid out at `<tmp>/<name>/mcts_trader/training/safe_checkpoint.py` with
    `<tmp>/<name>/tools/ckpt_gc.py` symlinked to the real tool, because the module resolves the
    retention policy as `Path(__file__).resolve().parents[2] / "tools" / "ckpt_gc.py"` — the
    mutant must load the SAME policy module the real one does, so the only difference between
    the two runs is the mutated line.
    """
    src = open(SC_PATH, encoding="utf-8").read()
    assert src.count(old) == 1, f"anchor not unique ({src.count(old)}): {old!r}"
    root = tmp_path / name
    (root / "mcts_trader" / "training").mkdir(parents=True)
    (root / "tools").mkdir(parents=True)
    os.symlink(GC_PATH, str(root / "tools" / "ckpt_gc.py"))
    dst = root / "mcts_trader" / "training" / "safe_checkpoint.py"
    dst.write_text(src.replace(old, new), encoding="utf-8")
    spec = importlib.util.spec_from_file_location(f"_mutant_sc_{name}", str(dst))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SC_PATH = os.path.join(REPO, "mcts_trader", "training", "safe_checkpoint.py")


def _mid_run_manifest_tree(tmp_path):
    """4 scored snapshots in `runs/r1`, `covers: runs/r1` already in place. Returns the run dir.

    ep10 is the WORST-scoring one, so with `--keep-best 3` it is the deletion candidate — which
    is what makes "a `protect:` line dropped for ep10 after the cache warmed" the exact shape
    that decides whether the deleting call re-scans.
    """
    run_dir = tmp_path / "runs" / "r1"
    rows = []
    for name, score in (("snapshot_ep10", 0.1), ("snapshot_ep20", 5.0),
                        ("snapshot_ep30", 6.0), ("snapshot_ep40", 7.0)):
        _mk_ckpt(str(run_dir), name)
        rows.append({"path": os.path.realpath(str(run_dir / name)), "score": score,
                     "metric": "sharpe", "epoch": int(name.split("ep")[1])})
    with open(str(run_dir / "scores.jsonl"), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    (tmp_path / "experiments" / "e2").mkdir(parents=True)
    (tmp_path / "experiments" / "e2" / "PROTECTED.txt").write_text("covers: runs/r1\n")
    return run_dir


def test_a_deleting_call_at_the_default_ttl_honours_a_manifest_dropped_mid_run(tmp_path,
                                                                               monkeypatch):
    """BLOCKER (cycle 2), THE DANGEROUS DIRECTION — the one the TTL test above stepped over.

    No `ttl_seconds=0.05`, no `sleep`: the DEFAULT 300 s TTL, `apply=True`, and a `protect:`
    manifest dropped immediately after the cache warmed (the mid-run manifest drop SERVER.md
    documents as normal workflow, and the case the TTL was introduced for). Measured before the
    fix: the offline tool printed `keeping 4` with ep10 protected while the in-run pruner
    printed `DELETE runs/r1/snapshot_ep10` and deleted it, in the same second, with
    `stale=False` and `had_failure=False` and no warning. `stale` only fires when a scan FAILS;
    a scan that SUCCEEDED 299 s ago against a different tree read as authoritative to an
    irreversible deleter.
    """
    sc._reset_protection_cache_for_test()
    monkeypatch.chdir(tmp_path)
    run_dir = _mid_run_manifest_tree(tmp_path)

    # Warm the cache exactly as the first save of a training run does.
    sc.prune_run_dir_by_score(str(run_dir), under_floor=False, keep_best=3, apply=False)

    # ... and NOW the operator drops the protecting manifest.
    (tmp_path / "experiments" / "e1").mkdir(parents=True)
    (tmp_path / "experiments" / "e1" / "PROTECTED.txt").write_text(
        "protect: runs/r1/snapshot_ep10\n")

    msgs, had_failure = sc.prune_run_dir_by_score(str(run_dir), under_floor=False, keep_best=3,
                                                  apply=True)
    assert os.path.isdir(str(run_dir / "snapshot_ep10")), (
        "a manifest-protected checkpoint was DELETED inside the TTL window: " + "\n".join(msgs))
    assert not any("DELETE" in m for m in msgs), msgs
    assert not had_failure, msgs
    assert not sc.protection_cache_is_stale()

    # The offline tool, same tree, same instant, must agree -- that is the claim this fix makes.
    r = _run_gc(["--roots", "runs", "--keep-best", "3"], cwd=str(tmp_path))
    assert "snapshot_ep10" not in r.stdout.split("would delete")[-1] or "would delete" \
        not in r.stdout, r.stdout
    sc._reset_protection_cache_for_test()


def test_red_when_the_deleting_call_reads_the_ttl_cache_at_its_real_drop_site(tmp_path,
                                                                              monkeypatch):
    """MUTATION at the real drop site: give the deleting call the CACHED protect set again and
    the shipped BLOCKER comes straight back — the protected checkpoint is deleted."""
    monkeypatch.chdir(tmp_path)
    run_dir = _mid_run_manifest_tree(tmp_path)

    mutant = _mutant_sc(
        tmp_path, "cached_protect_for_deleter",
        "    named, protect, missing = get_protection(roots, scan_dirs, ttl_seconds, "
        "force_rescan=apply)",
        "    named, protect, missing = get_protection(roots, scan_dirs, ttl_seconds, "
        "force_rescan=False)")

    # TASK_881E/D70 escape hatch: 'e1'/'r1' is fixture convenience for the TTL-cache mutation
    # under test, not a real protection relationship.
    mutant.prune_run_dir_by_score(str(run_dir), under_floor=False, keep_best=3, apply=False,
                                  authority_binding=False)
    (tmp_path / "experiments" / "e1").mkdir(parents=True)
    (tmp_path / "experiments" / "e1" / "PROTECTED.txt").write_text(
        "protect: runs/r1/snapshot_ep10\n")
    msgs, _ = mutant.prune_run_dir_by_score(str(run_dir), under_floor=False, keep_best=3,
                                            apply=True, authority_binding=False)
    assert not os.path.isdir(str(run_dir / "snapshot_ep10")), (
        "the mutation did not reproduce the BLOCKER: " + "\n".join(msgs))
    assert any("DELETE" in m for m in msgs), msgs
    assert not mutant.protection_cache_is_stale(), "the BLOCKER was fail-OPEN: stale was False"


def test_a_failed_fresh_scan_on_a_deleting_call_refuses_instead_of_using_the_cache(tmp_path,
                                                                                   monkeypatch):
    """The other half of the fix: forcing a re-scan must not become a way to lose protection.
    A scan that FAILS keeps the last-known-good set AND marks it stale, and a deleting call
    REFUSES on stale -- it never falls back to the cache it just declined to trust."""
    sc._reset_protection_cache_for_test()
    monkeypatch.chdir(tmp_path)
    run_dir = _mid_run_manifest_tree(tmp_path)
    sc.prune_run_dir_by_score(str(run_dir), under_floor=False, keep_best=3, apply=False)

    # A malformed line anywhere under `experiments/` makes the whole scan INCOMPLETE.
    (tmp_path / "experiments" / "e3").mkdir(parents=True)
    (tmp_path / "experiments" / "e3" / "PROTECTED.txt").write_text("protect: a b\n")

    msgs, had_failure = sc.prune_run_dir_by_score(str(run_dir), under_floor=False, keep_best=3,
                                                  apply=True)
    assert sc.protection_cache_is_stale(), msgs
    assert any("STALE" in m and "REFUSING" in m for m in msgs), msgs
    assert not had_failure, msgs
    for n in ("snapshot_ep10", "snapshot_ep20", "snapshot_ep30", "snapshot_ep40"):
        assert os.path.isdir(str(run_dir / n)), f"{n} deleted under a stale cache: {msgs}"
    sc._reset_protection_cache_for_test()


def _mutant_gc(tmp_path, name, old, new):
    """A COPY of `tools/ckpt_gc.py` with one exact substring replaced. Returned as a PATH — the
    tool is standalone/stdlib-only by design, so a mutant is just another script to run."""
    src = open(GC_PATH, encoding="utf-8").read()
    assert src.count(old) == 1, f"anchor not unique ({src.count(old)}): {old!r}"
    p = tmp_path / f"ckpt_gc_{name}.py"
    p.write_text(src.replace(old, new), encoding="utf-8")
    return str(p)


def _run_script(script, args, cwd, timeout=30):
    try:
        return subprocess.run([sys.executable, script] + args, cwd=cwd, capture_output=True,
                              text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        return subprocess.CompletedProcess(e.cmd, 124, e.stdout or "", e.stderr or "")


# ------------------------------------------------------------------ MAJOR: ONE --protect-file
# reader. `main()` used to carry a second, inline one that dropped `load_protect_file`'s
# non-regular-file `lstat` gate.

def _one_run_dir(tmp_path):
    run_dir = tmp_path / "runs" / "my_run"
    _mk_ckpt(str(run_dir), "checkpoint_100")
    (tmp_path / "experiments").mkdir()
    return run_dir


def test_major_fifo_protect_file_refuses_on_the_DELETE_path_instead_of_hanging(tmp_path):
    """THE MAJOR, measured: `--verify-manifests` exited 2 on a FIFO `--protect-file` while the
    `--apply` delete path exited 124 — it HUNG FOREVER in `open()`. An orchestrator `--apply`
    that never returns and never reports is worse than one that raises."""
    _one_run_dir(tmp_path)
    os.mkfifo(str(tmp_path / "p.fifo"))
    apply_r = _run_gc(["--roots", "runs", "--experiments-dir", "experiments",
                       "--protect-file", "p.fifo", "--apply", D64_OVERRIDE,
                       "--min-free-gb", "1e18"],
                      cwd=str(tmp_path))
    verify_r = _run_gc(["--roots", "runs", "--experiments-dir", "experiments",
                        "--protect-file", "p.fifo", "--verify-manifests"], cwd=str(tmp_path))
    assert apply_r.returncode == 2, f"{apply_r.returncode}: {apply_r.stdout}{apply_r.stderr}"
    assert verify_r.returncode == 2, verify_r.stdout + verify_r.stderr
    assert "fifo/socket/device" in apply_r.stderr, apply_r.stderr


def test_red_when_main_gets_its_own_inline_protect_file_reader_back(tmp_path):
    """MUTATION at the real drop site: put the second reader back and the delete path hangs
    again (exit 124 under the timeout), while `--verify-manifests` still exits 2."""
    _one_run_dir(tmp_path)
    os.mkfifo(str(tmp_path / "p.fifo"))
    mutant = _mutant_gc(
        tmp_path, "inline_protect_reader",
        "    protect_file_paths, protect_file_error = load_protect_file(a.protect_file)",
        "    protect_file_paths, protect_file_error = set(), None\n"
        "    if a.protect_file:\n"
        "        if os.path.islink(a.protect_file):\n"
        "            protect_file_error = f'{a.protect_file}: SYMLINK'\n"
        "        else:\n"
        "            try:\n"
        "                with open(a.protect_file) as fh:\n"
        "                    protect_file_paths = {os.path.realpath(l.strip()) for l in fh\n"
        "                                         if l.strip() and not l.startswith('#')}\n"
        "            except OSError as e:\n"
        "                protect_file_error = f'{a.protect_file}: {e}'")
    m = _run_script(mutant, ["--roots", "runs", "--experiments-dir", "experiments",
                             "--protect-file", "p.fifo", "--apply", D64_OVERRIDE,
                             "--min-free-gb", "1e18"],
                    cwd=str(tmp_path), timeout=8)
    assert m.returncode == 124, f"the mutation did not reproduce the hang: {m.returncode}"
    r = _run_gc(["--roots", "runs", "--experiments-dir", "experiments",
                 "--protect-file", "p.fifo", "--apply", D64_OVERRIDE, "--min-free-gb", "1e18"],
                cwd=str(tmp_path))
    assert r.returncode == 2, r.stdout + r.stderr


def test_the_two_readers_agree_on_an_indented_comment_on_both_paths(tmp_path):
    """The secondary divergence: `not l.startswith("#")` (raw) on the delete path vs
    `.strip().startswith("#")` in `load_protect_file`. An indented comment was ingested as a
    bogus path by the deleter and skipped by verify."""
    _one_run_dir(tmp_path)
    (tmp_path / "p.txt").write_text("runs/my_run\n  # indented comment, not a path\n")
    d = _run_gc(["--roots", "runs", "--experiments-dir", "experiments", "--protect-file",
                 "p.txt", "--min-free-gb", "1e18"], cwd=str(tmp_path))
    v = _run_gc(["--roots", "runs", "--experiments-dir", "experiments", "--protect-file",
                 "p.txt", "--verify-manifests"], cwd=str(tmp_path))
    assert "1 entry read" in d.stderr, d.stderr            # delete path (stderr)
    assert "1 entrie(s) read" in v.stdout, v.stdout        # --verify-manifests (HEAD's wording)
    assert "indented comment" not in (d.stdout + d.stderr), d.stdout + d.stderr


@pytest.mark.parametrize("shape", ["missing", "dir", "livelink", "danglink", "blank"])
def test_every_unusable_protect_file_shape_refuses_apply_identically(tmp_path, shape):
    """The whole class on the DELETE path, through the one reader."""
    _one_run_dir(tmp_path)
    if shape == "missing":
        arg = "no_such_file.txt"
    elif shape == "dir":
        arg = "experiments"
    elif shape == "blank":
        arg = ""
    else:
        (tmp_path / "outside").mkdir()
        real = tmp_path / "outside" / "p.txt"
        real.write_text("runs/my_run\n")
        os.symlink(str(real), str(tmp_path / "p.link"))
        if shape == "danglink":
            os.unlink(str(real))
        arg = "p.link"
    r = _run_gc(["--roots", "runs", "--experiments-dir", "experiments", "--protect-file", arg,
                 "--apply", D64_OVERRIDE, "--min-free-gb", "1e18"], cwd=str(tmp_path))
    assert r.returncode == 2, f"{shape}: {r.returncode} {r.stdout} {r.stderr}"
    assert os.path.isdir(str(tmp_path / "runs" / "my_run" / "checkpoint_100")), shape


# --------------------------------------------- TASK_881D gate (vi): the corpus appears in EVERY
# report, including an --apply log on a tree where nothing was a deletion candidate.

def test_gate_vi_granting_entries_appear_in_the_delete_path_report(tmp_path):
    """A `PROTECTED <path>` line is emitted only when a path was ALREADY a deletion candidate,
    so on a healthy tree the granting half of the corpus appeared in NO delete-path report."""
    run_dir = tmp_path / "runs" / "my_run"
    _mk_ckpt(str(run_dir), "checkpoint_100")
    (tmp_path / "experiments" / "e1").mkdir(parents=True)
    (tmp_path / "experiments" / "e1" / "PROTECTED.txt").write_text(
        "covers: runs/my_run\nprotect: runs/my_run/checkpoint_100\n")
    r = _run_gc(["--roots", "runs", "--experiments-dir", "experiments", "--apply", D64_OVERRIDE,
                 "--min-free-gb", "1e18"], cwd=str(tmp_path))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "1 protect, 1 covers, 0 dangling" in r.stderr, r.stderr
    assert "checkpoint_100  <- protect: runs/my_run/checkpoint_100" in r.stderr, r.stderr
    assert "experiments/e1/PROTECTED.txt:2" in r.stderr, r.stderr
    assert os.path.isdir(str(run_dir / "checkpoint_100"))


# ============================================================================================
# D63(a) — the WRITTEN, ADDITIVE scope amendment that unparks this row.
#
# (A) MAJOR: `--verify-manifests` can never exit 0 on the real tree, because this row's own
#     corpus carries acknowledged dangling entries (scratch disk unmounted). `--known-dangling
#     PATH...` lets an operator ACKNOWLEDGE a specific dangling condition without weakening the
#     check for anything else: the condition still prints in full, a NEW dangling entry still
#     refuses, a malformed manifest still refuses (acknowledgement covers DANGLING only), and an
#     acknowledgement that matches nothing prints a loud STALE ACKNOWLEDGEMENT.
# (B) MINOR: a manifest under a walk-pruned dirname (`.git`/`__pycache__`) is a SILENT
#     withdrawal — `load_manifests` now peeks into a pruned dirname (never trusting what it finds
#     for protect/covers/dangling) and reports it as `MANIFEST NOT SCANNED`, without turning it
#     into a refusal (the D53 `--experiments-dir experiments experiments/__pycache__` collapse
#     shape, and the "ancestor alone does not reach it" control, both stay exit 0).
# ============================================================================================

def test_known_dangling_acknowledges_a_manifest_entry_and_still_prints_it(tmp_path):
    """The condition PRINTS IN FULL — including the citation and the target path — but no longer
    sets the exit code, matching D63(a)'s "prints in full; only stops setting ok = False"."""
    exp = tmp_path / "experiments" / "999_x"
    exp.mkdir(parents=True)
    (exp / "PROTECTED.txt").write_text("protect: runs/does_not_exist/snapshot_ep10\n")
    (tmp_path / "runs").mkdir()

    unacked = _run_gc(["--roots", "runs", "--experiments-dir", "experiments",
                       "--verify-manifests"], cwd=str(tmp_path))
    assert unacked.returncode == 2, unacked.stdout + unacked.stderr

    target = os.path.realpath(str(tmp_path / "runs" / "does_not_exist" / "snapshot_ep10"))
    acked = _run_gc(["--roots", "runs", "--experiments-dir", "experiments",
                     "--known-dangling", "runs/does_not_exist/snapshot_ep10",
                     "--verify-manifests"], cwd=str(tmp_path))
    assert acked.returncode == 0, acked.stdout + acked.stderr
    assert "DANGLING ENTRY (ACKNOWLEDGED via --known-dangling)" in acked.stderr, acked.stderr
    assert target in acked.stderr, acked.stderr
    assert "does_not_exist/snapshot_ep10" in acked.stderr, acked.stderr


def test_known_dangling_never_silences_a_NEW_dangling_entry_beside_it(tmp_path):
    """Acknowledging ONE dangling path must not blanket-silence a SECOND, unrelated one."""
    exp = tmp_path / "experiments" / "999_x"
    exp.mkdir(parents=True)
    (exp / "PROTECTED.txt").write_text(
        "protect: runs/does_not_exist/snapshot_ep10\n"
        "protect: runs/another_ghost/snapshot_ep20\n")
    (tmp_path / "runs").mkdir()
    r = _run_gc(["--roots", "runs", "--experiments-dir", "experiments",
                "--known-dangling", "runs/does_not_exist/snapshot_ep10",
                "--verify-manifests"], cwd=str(tmp_path))
    assert r.returncode == 2, r.stdout + r.stderr
    assert "DANGLING ENTRY (ACKNOWLEDGED via --known-dangling)" in r.stderr, r.stderr
    assert "another_ghost/snapshot_ep20" in r.stderr, r.stderr
    assert "VERIFY-MANIFESTS REFUSAL: DANGLING ENTRY" in r.stderr, r.stderr


def test_known_dangling_never_silences_a_malformed_manifest_at_the_acknowledged_path(tmp_path):
    """D63(a): "acknowledgement covers DANGLING only, never malformation" — a MALFORMED manifest
    living right beside (or naming) an acknowledged path must still refuse."""
    exp = tmp_path / "experiments" / "999_x"
    exp.mkdir(parents=True)
    (exp / "PROTECTED.txt").write_text(
        "protect: runs/does_not_exist/snapshot_ep10\nprotect: a b\n")   # 2nd line is MALFORMED
    (tmp_path / "runs").mkdir()
    r = _run_gc(["--roots", "runs", "--experiments-dir", "experiments",
                "--known-dangling", "runs/does_not_exist/snapshot_ep10",
                "--verify-manifests"], cwd=str(tmp_path))
    assert r.returncode == 2, r.stdout + r.stderr
    assert "malformed manifest content" in r.stderr, r.stderr


def test_known_dangling_covers_a_dangling_root_symlink_too(tmp_path):
    """The SECOND acknowledgeable site: `_root_error`, for a `--roots` entry whose symlink target
    does not resolve (the real `jax_ckpt_fast` shape) — not just a manifest entry."""
    (tmp_path / "experiments").mkdir()
    os.symlink("/no/such/scratchpad", str(tmp_path / "jax_ckpt_fast"))
    unacked = _run_gc(["--roots", "jax_ckpt_fast", "--experiments-dir", "experiments",
                       "--verify-manifests"], cwd=str(tmp_path))
    assert unacked.returncode == 2, unacked.stdout + unacked.stderr
    assert "BROKEN SYMLINK" in unacked.stderr, unacked.stderr

    acked = _run_gc(["--roots", "jax_ckpt_fast", "--experiments-dir", "experiments",
                     "--known-dangling", "jax_ckpt_fast", "--verify-manifests"],
                    cwd=str(tmp_path))
    assert acked.returncode == 0, acked.stdout + acked.stderr
    assert "DANGLING (ACKNOWLEDGED via --known-dangling)" in acked.stderr, acked.stderr
    assert "BROKEN SYMLINK" in acked.stderr, acked.stderr


def test_red_when_the_root_error_ack_is_removed_at_its_real_drop_site(tmp_path):
    """MUTATION at `_root_error`'s OWN real drop site (the second acknowledgeable object,
    distinct from the manifest-dangling loop above): force the acknowledgement check to always
    report False. `--known-dangling jax_ckpt_fast` can no longer silence the broken-symlink
    root, even though the flag is given the exact matching path."""
    (tmp_path / "experiments").mkdir()
    os.symlink("/no/such/scratchpad", str(tmp_path / "jax_ckpt_fast"))

    mutant = _mutant_gc(
        tmp_path, "root_error_ack_disabled",
        '        if isinstance(exc, DanglingScanLink) and _ack.covers(filename):',
        '        if False and isinstance(exc, DanglingScanLink) and _ack.covers(filename):')
    m = _run_script(mutant, ["--roots", "jax_ckpt_fast", "--experiments-dir", "experiments",
                            "--known-dangling", "jax_ckpt_fast", "--verify-manifests"],
                    cwd=str(tmp_path))
    assert m.returncode == 2, f"mutation did not reproduce the disabled root ack: {m.stdout}{m.stderr}"
    assert "DANGLING (ACKNOWLEDGED" not in m.stderr, m.stderr

    r = _run_gc(["--roots", "jax_ckpt_fast", "--experiments-dir", "experiments",
                "--known-dangling", "jax_ckpt_fast", "--verify-manifests"], cwd=str(tmp_path))
    assert r.returncode == 0, f"the real, un-mutated tool must still accept it: {r.stdout}{r.stderr}"


def test_known_dangling_never_silences_a_wrong_shape_root(tmp_path):
    """A root that is the WRONG SHAPE (a regular file, not a dangling symlink) is a DIFFERENT
    fact than "known dangling" and must never be silenced by this flag, even if its literal path
    string happens to be named."""
    (tmp_path / "experiments").mkdir()
    (tmp_path / "not_a_dir").write_text("x")
    r = _run_gc(["--roots", "not_a_dir", "--experiments-dir", "experiments",
                "--known-dangling", "not_a_dir", "--verify-manifests"], cwd=str(tmp_path))
    assert r.returncode == 2, r.stdout + r.stderr
    assert "NOT A DIRECTORY" in r.stderr, r.stderr


def test_stale_known_dangling_acknowledgement_is_reported_loudly(tmp_path):
    """An acknowledgement that matches nothing this run — because the corpus is clean, or
    because the operator mistyped it — is never silently accepted."""
    (tmp_path / "runs").mkdir()
    (tmp_path / "experiments").mkdir()
    r = _run_gc(["--roots", "runs", "--experiments-dir", "experiments",
                "--known-dangling", "runs/nothing_is_actually_dangling_here",
                "--verify-manifests"], cwd=str(tmp_path))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "STALE ACKNOWLEDGEMENT" in r.stderr, r.stderr
    assert "runs/nothing_is_actually_dangling_here" in r.stderr, r.stderr


def test_red_when_the_known_dangling_ack_is_removed_at_its_real_drop_site(tmp_path):
    """MUTATION at the real drop site: `_ack.covers(d)` forced to always report False. Even with
    `--known-dangling` given the exact path, the acknowledgement can no longer be granted."""
    exp = tmp_path / "experiments" / "999_x"
    exp.mkdir(parents=True)
    (exp / "PROTECTED.txt").write_text("protect: runs/does_not_exist/snapshot_ep10\n")
    (tmp_path / "runs").mkdir()

    mutant = _mutant_gc(
        tmp_path, "known_dangling_ack_disabled",
        "        for d in sorted(manifests.dangling):\n"
        "            acknowledged = _ack.covers(d)",
        "        for d in sorted(manifests.dangling):\n"
        "            acknowledged = False and _ack.covers(d)")
    m = _run_script(mutant, ["--roots", "runs", "--experiments-dir", "experiments",
                            "--known-dangling", "runs/does_not_exist/snapshot_ep10",
                            "--verify-manifests"], cwd=str(tmp_path))
    assert m.returncode == 2, f"mutation did not reproduce the disabled ack: {m.stdout}{m.stderr}"

    r = _run_gc(["--roots", "runs", "--experiments-dir", "experiments",
                "--known-dangling", "runs/does_not_exist/snapshot_ep10",
                "--verify-manifests"], cwd=str(tmp_path))
    assert r.returncode == 0, f"the real, un-mutated tool must still accept it: {r.stdout}{r.stderr}"


def test_manifest_under_a_pruned_dirname_is_reported_not_refused(tmp_path):
    """A SPEC.md under `experiments/__pycache__`, reached only through its ancestor `experiments`
    (never named directly), is a SILENT withdrawal at HEAD's own walk semantics — this row makes
    it a NAMED, loud stderr report instead, without turning it into a refusal."""
    (tmp_path / "experiments" / "__pycache__").mkdir(parents=True)
    (tmp_path / "runs" / "a").mkdir(parents=True)
    (tmp_path / "experiments" / "__pycache__" / "SPEC.md").write_text("protected:\n- runs/a\n")
    r = _run_gc(["--roots", "runs", "--experiments-dir", "experiments", "--verify-manifests"],
               cwd=str(tmp_path))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "0 manifest file(s) read" in r.stdout, r.stdout
    assert "0 protect" in r.stdout, r.stdout
    assert "MANIFEST NOT SCANNED" in r.stderr, r.stderr
    assert "__pycache__" in r.stderr, r.stderr
    assert "1 unscanned under a walk-pruned dirname" in r.stdout, r.stdout


def test_manifest_under_a_pruned_dirname_reached_explicitly_is_not_double_reported(tmp_path):
    """D53's collapse shape, re-proven under this row's new report: naming the pruned dirname
    itself as a SECOND, explicit --experiments-dir value scans it for real, and the new report
    must not ALSO claim it was unscanned."""
    (tmp_path / "experiments" / "__pycache__").mkdir(parents=True)
    (tmp_path / "runs" / "a").mkdir(parents=True)
    (tmp_path / "experiments" / "__pycache__" / "SPEC.md").write_text("protected:\n- runs/a\n")
    r = _run_gc(["--roots", "runs", "--experiments-dir", "experiments",
                "experiments/__pycache__", "--verify-manifests"], cwd=str(tmp_path))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "1 manifest file(s) read" in r.stdout, r.stdout
    assert "1 protect" in r.stdout, r.stdout
    assert "MANIFEST NOT SCANNED" not in r.stderr, r.stderr
    assert "0 unscanned under a walk-pruned dirname" in r.stdout, r.stdout


def test_red_when_the_pruned_dirname_peek_is_removed_at_its_real_drop_site(tmp_path):
    """MUTATION at the real drop site: the peek into a walk-pruned dirname disabled. The silent
    withdrawal goes back to being silent — no report, no stat — reproducing the pre-D63(a) gap."""
    (tmp_path / "experiments" / "__pycache__").mkdir(parents=True)
    (tmp_path / "runs" / "a").mkdir(parents=True)
    (tmp_path / "experiments" / "__pycache__" / "SPEC.md").write_text("protected:\n- runs/a\n")

    mutant = _mutant_gc(
        tmp_path, "pruned_peek_disabled",
        '            for fn in names:\n'
        '                if fn not in _WALK_PRUNED_DIRNAMES:\n'
        '                    continue',
        '            for fn in names:\n'
        '                if True:\n'
        '                    continue')
    m = _run_script(mutant, ["--roots", "runs", "--experiments-dir", "experiments",
                            "--verify-manifests"], cwd=str(tmp_path))
    assert m.returncode == 0, m.stdout + m.stderr
    assert "MANIFEST NOT SCANNED" not in m.stderr, m.stderr  # the reproduced silent withdrawal

    r = _run_gc(["--roots", "runs", "--experiments-dir", "experiments", "--verify-manifests"],
               cwd=str(tmp_path))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "MANIFEST NOT SCANNED" in r.stderr, r.stderr


# ============================================================================================
# THE GUARDS 881D's RE-SCOPE LEFT UN-PINNED (regression judge, cycle 2).
#
# `test_ingestion_surface_is_byte_identical_to_head` (881I suite) pins the manifest-INGESTION
# surface. `CANON_FUNCTIONS` — the set 881D is authorized to rewrite — was demoted there from an
# assertion to a COMMENT, which is correct for the ones this row really did rewrite (`collect`,
# `plan`, `_load_scores`, `_is_ckpt_dir`, `_epoch_of`) and WRONG for the ones it did not touch at
# all: they simply lost their pin, and were mutation-proved to survive mutation with nothing
# catching them. Two of them, `_safe_target`'s symlink check and its outside-roots refusal, are
# the ONLY per-path fences left in the delete loop besides `_is_protected` — an unpinned fence in
# front of an irreversible deleter is the exact shape this row exists to remove.
#
# Also pinned: five ingestion CONSTANTS `INGESTION_NAMES` omits. They carry the refusal REASONS
# the ingestion surface prints, and `_WALK_PRUNED_DIRNAMES` is consulted by
# `_dedupe_nested_scan_dirs` to decide whether collapsing a scan dir can drop a manifest.
#
# This list is ADDITIVE to 881I's — it does not touch, widen or re-scope `INGESTION_NAMES`, and
# it asserts nothing about the functions 881D is sanctioned to rewrite.
# ============================================================================================

UNTOUCHED_GUARD_NAMES = (
    "_safe_target",            # symlink check + outside-roots refusal: the delete loop's fences
    "_du",                     # the size a DELETE line reports and `reclaimed` sums
    "free_gb",                 # the disk floor that turns unscored ckpts recency-prunable
    "_MANIFEST_SHAPE_WHY",
    "_DANGLING_LINK_WHY",
    "_LIVE_LINK_NON_DIR_WHY",
    "_WALK_PRUNED_DIRNAMES",
    "_BARE_LOG_GLOB_PROBE_DIR",
)


def _head_gc_source():
    r = subprocess.run(["git", "show", "HEAD:tools/ckpt_gc.py"], cwd=REPO,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout


def _top_level_entity(src, name):
    """The exact text of the top-level `def`/`class`/assignment named `name`, AST-located."""
    lines = src.splitlines(keepends=True)
    for node in ast.parse(src).body:
        nm = getattr(node, "name", None)
        if nm is None and isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == name:
                    nm = name
        if nm == name:
            return "".join(lines[node.lineno - 1:node.end_lineno])
    raise AssertionError(f"{name!r} not found as a top-level def/class/assignment")


@pytest.mark.parametrize("name", UNTOUCHED_GUARD_NAMES)
def test_untouched_guard_is_byte_identical_to_head(name):
    assert _top_level_entity(open(GC_PATH).read(), name) == \
        _top_level_entity(_head_gc_source(), name), \
        f"{name!r} is a guard 881D does not rewrite and it is no longer byte-identical to HEAD"


@pytest.mark.parametrize("name,old,new", [
    ("_safe_target", '    if os.path.islink(path):\n        return False, "is a symlink"',
     '    if False:\n        return False, "is a symlink"'),
    ("_safe_target", '        return False, "outside configured roots"',
     '        return True, "ok"'),
    ("_du", "                total += os.path.getsize(fp)",
     "                total += 0"),
    ("free_gb", "    return st.f_bavail * st.f_frsize / 1e9",
     "    return 1e18"),
    ("_WALK_PRUNED_DIRNAMES", '_WALK_PRUNED_DIRNAMES = (".git", "__pycache__")',
     '_WALK_PRUNED_DIRNAMES = (".git",)'),
])
def test_the_guard_pin_goes_red_when_the_guard_is_mutated(tmp_path, name, old, new):
    """MUTATION, at each guard's real drop site: the pin above must FAIL. Run against a COPY —
    the real tool is never edited. Proves the pin is a check and not a tautology."""
    src = open(GC_PATH).read()
    assert src.count(old) == 1, f"anchor not unique ({src.count(old)}): {old!r}"
    mutated = src.replace(old, new)
    assert _top_level_entity(mutated, name) != _top_level_entity(_head_gc_source(), name), \
        f"mutating {name!r} at its real drop site did not move the pinned text"
