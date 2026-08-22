"""TASK_881G — the manifest grammar: written down, then accepted (881D's unblocker).

881I/881S shipped a REFUSE-path that five lens-runs could not break, but the grammar the tool
actually ACCEPTS existed nowhere an author would look — the canonical MD032 form
(``protected:`` / blank line / list) refused, and 0 of 56 real manifest files ever exercised the
accept path at all. This row (D53) writes the grammar down (see ``tools/ckpt_gc.py``'s module
docstring, "MANIFEST GRAMMAR") and WIDENS the accept path by exactly what that doc now promises —
the dangerous direction, so every newly-accepted shape here carries its own mutation.

Covered:
  (gate ii/v) the MD032 form now parses; a mutation that disables the new branch proves the
              acceptance is real, not incidental
  (gate ii)   the `---`/`--` thematic-break ingestion NIT closed (silently-mis-ingested items ->
              refused); a mutation that disables the guard proves it was a real silent hole
  (gate iii)  one parser, one grammar: a shared item-level corpus accepted/refused identically by
              `parse_protected_txt` and `parse_spec_manifest`
  (scope 5)   the D53 asymmetry: an explicitly-passed nonexistent path now refuses on
              `--experiments-dir`, matching `--protect-file`'s existing behaviour; the DEFAULT
              (flag omitted) staying a silent no-op is a separate, unchanged control
  (scope 4)   nested/duplicate `--experiments-dir` values no longer double-count; `#`-comments in
              `--protect-file` are recognised however they are indented

Cycle-2 additions (the escalated fix pass), each pinning a defect the first cut of this row
SHIPPED — all three in the WITHDRAW/weaken direction the row's own gate calls dangerous:
  (BLOCKER)   the scan-dir collapse ran on `realpath` BEFORE `load_manifests`, ERASING 881I's
              symlinked-scan-root refusal (exit 2 -> exit 0). Nothing covered TWO scan dirs where
              one is a symlink; that case is now here, in both orders and with a link INSIDE the
              tree, plus its mutation
  (MAJOR)     the collapse's stated invariant ("nothing under a collapsed dir goes un-scanned")
              was FALSE — `_walk_scan_dirs` prunes `.git`/`__pycache__`, so a dropped dir behind a
              pruned component was silently NOT scanned and its protection silently withdrawn
  (MAJOR)     the newly-accepted MD032 shape emitted a span SHORT by the blank gap, so
              `strip_spans` cut mid-item — the parse-and-span-disagree class D53 was re-cut over,
              arriving through the new accept path. Round-trip assertions now exist
  (gate i)    the six doc defects an isolated doc-only reader hit (46 cases, 3 SURPRISE, 3
              UNDERSPECIFIED): each amended claim is pinned by a test here so the doc cannot drift
              back to being wrong
"""
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CKPT_GC = os.path.join(REPO, "tools", "ckpt_gc.py")

sys.path.insert(0, os.path.join(REPO, "tools"))
import ckpt_gc as gc  # noqa: E402


def _cli(*args, cwd, timeout=20):
    return subprocess.run([sys.executable, CKPT_GC, *args], cwd=cwd,
                          capture_output=True, text=True, timeout=timeout)


def _cli_script(script, *args, cwd, timeout=20):
    return subprocess.run([sys.executable, script, *args], cwd=cwd,
                          capture_output=True, text=True, timeout=timeout)


def _mutant(tmp_path, name, old, new):
    """Write a copy of the REAL `tools/ckpt_gc.py` with `old` replaced by `new`, and return its
    path so it can be RUN as a CLI. See test_ckpt_881i_manifest_ingestion.py's `_mutant` for the
    full rationale (uniqueness-asserted, source-level, never monkeypatched)."""
    src = open(CKPT_GC).read()
    assert src.count(old) == 1, (
        f"mutation anchor for {name!r} matched {src.count(old)} times, expected exactly 1 — the "
        f"guard moved or was reworded; update this anchor:\n{old!r}")
    src = src.replace(old, new)
    d = tmp_path / "_mutants"
    d.mkdir(exist_ok=True)
    p = d / f"ckpt_gc_{name}.py"
    p.write_text(src)
    return str(p)


def _spec_tree(tmp_path, spec_body, *, extra_run_dirs=()):
    """A scan tree with real `runs/a` and `runs/b` (so a granted entry is never dangling) and one
    `experiments/999_z/SPEC.md` carrying `spec_body`."""
    root = tmp_path / "tree"
    (root / "experiments" / "999_z").mkdir(parents=True)
    (root / "runs" / "a").mkdir(parents=True)
    (root / "runs" / "b").mkdir(parents=True)
    for d in extra_run_dirs:
        (root / d).mkdir(parents=True, exist_ok=True)
    (root / "experiments" / "999_z" / "SPEC.md").write_text(spec_body)
    return root


def _protected_txt_tree(tmp_path, body):
    """The PROTECTED.txt sibling of `_spec_tree`, over the SAME `runs/a` + `runs/b`."""
    root = tmp_path / "tree"
    (root / "experiments" / "999_z").mkdir(parents=True)
    (root / "runs" / "a").mkdir(parents=True)
    (root / "runs" / "b").mkdir(parents=True)
    (root / "experiments" / "999_z" / "PROTECTED.txt").write_text(body)
    return root


# ============================================================================================
# gate (v) / scope item 2 — the canonical MD032 form now PARSES
# ============================================================================================

def test_md032_shape_now_accepts(tmp_path):
    """`protected:` / blank line / list — the FIRST shape an author who runs a markdown linter
    will type, and the exact shape 881D's corpus will use. It must grant both protections, not
    refuse."""
    root = _spec_tree(tmp_path, "protected:\n\n- runs/a\n- runs/b\n")
    r = _cli("--roots", "runs", "--experiments-dir", "experiments", "--verify-manifests",
             cwd=str(root))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "2 protect" in r.stdout
    assert "runs/a" in r.stdout and "runs/b" in r.stdout


def test_md032_shape_with_several_blank_lines_also_accepts(tmp_path):
    """More than one blank line before the first item is the same fence, just wider."""
    root = _spec_tree(tmp_path, "protected:\n\n\n- runs/a\n- runs/b\n")
    r = _cli("--roots", "runs", "--experiments-dir", "experiments", "--verify-manifests",
             cwd=str(root))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "2 protect" in r.stdout


def test_control_a_real_split_is_still_refused_not_swallowed_by_the_md032_fix(tmp_path):
    """CONTROL: once an item has already been consumed, a blank-line gap followed by MORE items
    is still a SPLIT refusal — the MD032 acceptance is gated on `not block` and must never reach
    this shape."""
    root = _spec_tree(tmp_path, "protected:\n- runs/a\n\n- runs/b\n")
    r = _cli("--roots", "runs", "--experiments-dir", "experiments", "--verify-manifests",
             cwd=str(root))
    assert r.returncode == 2, r.stdout + r.stderr
    assert "SPLIT" in r.stderr, r.stderr
    assert "0 protect" in r.stdout


def test_control_a_genuinely_empty_block_still_refuses(tmp_path):
    """CONTROL: `protected:` followed only by blank lines then EOF/prose (nothing ever resumes
    the list) is still the EMPTY-block refusal, unchanged."""
    root = _spec_tree(tmp_path, "protected:\n\nsome prose, no list here\n")
    r = _cli("--roots", "runs", "--experiments-dir", "experiments", "--verify-manifests",
             cwd=str(root))
    assert r.returncode == 2, r.stdout + r.stderr
    assert "is EMPTY" in r.stderr, r.stderr


def test_red_when_the_md032_acceptance_is_removed_at_its_real_drop_site(tmp_path):
    """Disable the REAL new branch (`not block and ...`) at its real drop site and prove the
    MD032 shape goes back to being refused — the acceptance is not incidental to some OTHER
    guard."""
    root = _spec_tree(tmp_path, "protected:\n\n- runs/a\n- runs/b\n")
    mutant = _mutant(
        tmp_path, "no_md032_acceptance",
        "                if (not block and j < len(item_lines)",
        "                if (False and j < len(item_lines)")

    m = _cli_script(mutant, "--roots", "runs", "--experiments-dir", "experiments",
                    "--verify-manifests", cwd=str(root))
    assert m.returncode == 2, m.stdout + m.stderr
    assert "is EMPTY" in m.stderr, m.stderr        # the pre-881G refusal, reproduced
    assert "0 protect" in m.stdout

    r = _cli("--roots", "runs", "--experiments-dir", "experiments", "--verify-manifests",
             cwd=str(root))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "2 protect" in r.stdout


# ============================================================================================
# gate (ii) — the `---`/`--` thematic-break ingestion NIT
# ============================================================================================

def test_thematic_break_alone_refuses_as_empty_not_as_a_bogus_dangling_entry(tmp_path):
    """`---` right after the header matches `_SPEC_BULLET_RE`/`_SPEC_ITEM_RE` (bullet `-` +
    one-token "path" `--`) and, pre-881G, silently became an item naming the literal path `--`.
    It must now refuse as an EMPTY block, never mention a dangling `--` entry."""
    root = _spec_tree(tmp_path, "protected:\n---\n")
    r = _cli("--roots", "runs", "--experiments-dir", "experiments", "--verify-manifests",
             cwd=str(root))
    assert r.returncode == 2, r.stdout + r.stderr
    assert "is EMPTY" in r.stderr, r.stderr
    assert "DANGLING ENTRY" not in r.stderr, r.stderr


def test_thematic_break_after_items_with_no_gap_refuses_as_interrupted(tmp_path):
    """`---` immediately after a consumed item (no blank line) is an INTERRUPTED list, not a
    second item named `--`."""
    root = _spec_tree(tmp_path, "protected:\n- runs/a\n---\n- runs/b\n")
    r = _cli("--roots", "runs", "--experiments-dir", "experiments", "--verify-manifests",
             cwd=str(root))
    assert r.returncode == 2, r.stdout + r.stderr
    assert "INTERRUPTED" in r.stderr, r.stderr
    assert "DANGLING ENTRY" not in r.stderr, r.stderr


def test_control_a_real_path_starting_with_dashes_is_unaffected(tmp_path):
    """The thematic-break guard is scoped to a line that is ENTIRELY dashes — a real (if odd)
    path that merely STARTS with dashes must still be accepted normally."""
    root = _spec_tree(tmp_path, "protected:\n- runs/a\n", extra_run_dirs=("runs/a",))
    (root / "runs" / "--weird").mkdir(parents=True, exist_ok=True)
    (root / "experiments" / "999_z" / "SPEC.md").write_text("protected:\n- runs/--weird\n")
    r = _cli("--roots", "runs", "--experiments-dir", "experiments", "--verify-manifests",
             cwd=str(root))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "1 protect" in r.stdout
    assert "runs/--weird" in r.stdout


def test_red_when_the_thematic_break_guard_is_removed_at_its_real_drop_site(tmp_path):
    """Disable the REAL guard and prove `---` silently grants a protection (to a directory that
    happens to be named `--`) instead of refusing — the exact pre-881G hole, reproduced from
    scratch."""
    root = _spec_tree(tmp_path, "protected:\n---\n")
    (root / "--").mkdir()   # so the mis-ingested entry is NOT dangling, and the hole is silent

    mutant = _mutant(
        tmp_path, "no_thematic_break_guard",
        "            if bullet is not None and item is not None and _is_thematic_break(line):",
        "            if False:")

    m = _cli_script(mutant, "--roots", "runs", "--experiments-dir", "experiments",
                    "--verify-manifests", cwd=str(root))
    assert m.returncode == 0, m.stdout + m.stderr      # SILENT ACCEPT — the reproduced hole
    assert "1 protect" in m.stdout
    assert "-- " in m.stdout or "-- <-" in m.stdout or "/--  <-" in m.stdout, m.stdout

    r = _cli("--roots", "runs", "--experiments-dir", "experiments", "--verify-manifests",
             cwd=str(root))
    assert r.returncode == 2, r.stdout + r.stderr        # the shipped code refuses instead
    assert "is EMPTY" in r.stderr, r.stderr


# ============================================================================================
# gate (iii) — one parser, one grammar: a shared item-level corpus
# ============================================================================================

def _protected_txt_of(shape_items):
    return "".join(f"protect: {p}\n" for p in shape_items) if isinstance(shape_items, list) \
        else shape_items


# Each entry: (name, protected_txt_body, spec_md_body, expect_accept, expect_n_entries)
# `spec_md_body` always includes the `protected:` header; `protected_txt_body` never does (its
# grammar has no header/block concept). Paths are shared across both renderings of one shape.
_SHARED_ITEM_SHAPES = [
    ("single_item",
     "protect: runs/a\n",
     "protected:\n- runs/a\n",
     True, 1),
    ("two_items",
     "protect: runs/a\nprotect: runs/b\n",
     "protected:\n- runs/a\n- runs/b\n",
     True, 2),
    ("md032_gap_before_first_item",
     "protect: runs/a\nprotect: runs/b\n",       # PROTECTED.txt has no block/gap concept at all
     "protected:\n\n- runs/a\n- runs/b\n",
     True, 2),
    ("multi_token_line_refuses_both",
     "protect: runs/a runs/b\n",
     "protected:\n- runs/a runs/b\n",
     False, 0),
    ("trailing_comment_on_item_refuses_both",
     "protect: runs/a # note\n",
     "protected:\n- runs/a # note\n",
     False, 0),
    ("thematic_break_only_refuses_both",
     "---\n",                                     # PROTECTED.txt: not a directive -> malformed
     "protected:\n---\n",
     False, 0),
]


@pytest.mark.parametrize("name,protected_txt_body,spec_md_body,expect_accept,expect_n",
                         _SHARED_ITEM_SHAPES, ids=[s[0] for s in _SHARED_ITEM_SHAPES])
def test_both_parsers_agree_on_the_shared_item_corpus(name, protected_txt_body, spec_md_body,
                                                      expect_accept, expect_n):
    p_entries, _p_spans, p_bad = gc.parse_protected_txt(protected_txt_body, "PROTECTED.txt")
    s_entries, _s_spans, s_bad = gc.parse_spec_manifest(spec_md_body, "SPEC.md")

    p_accept = not p_bad
    s_accept = not s_bad
    assert p_accept == expect_accept, (name, "PROTECTED.txt", protected_txt_body, p_bad)
    assert s_accept == expect_accept, (name, "SPEC.md", spec_md_body, s_bad)
    if expect_accept:
        assert len(p_entries) == expect_n, (name, p_entries)
        assert len(s_entries) == expect_n, (name, s_entries)
        assert sorted(e.path for e in p_entries) == sorted(e.path for e in s_entries)


@pytest.mark.parametrize("name,protected_txt_body,spec_md_body,expect_accept,expect_n",
                         _SHARED_ITEM_SHAPES, ids=[s[0] for s in _SHARED_ITEM_SHAPES])
def test_both_ingestion_paths_agree_end_to_end_via_the_real_cli(tmp_path, name,
                                                                protected_txt_body, spec_md_body,
                                                                expect_accept, expect_n):
    """The same corpus, driven through the REAL `--verify-manifests` CLI on each basename, over
    the SAME run dirs — not just the two parser functions in isolation."""
    p_root = _protected_txt_tree(tmp_path / "p", protected_txt_body)
    s_root = _spec_tree(tmp_path / "s", spec_md_body)

    pr = _cli("--roots", "runs", "--experiments-dir", "experiments", "--verify-manifests",
              cwd=str(p_root))
    sr = _cli("--roots", "runs", "--experiments-dir", "experiments", "--verify-manifests",
              cwd=str(s_root))
    assert (pr.returncode == 0) == expect_accept, (name, "PROTECTED.txt", pr.stdout + pr.stderr)
    assert (sr.returncode == 0) == expect_accept, (name, "SPEC.md", sr.stdout + sr.stderr)
    if expect_accept:
        assert f"{expect_n} protect" in pr.stdout, pr.stdout
        assert f"{expect_n} protect" in sr.stdout, sr.stdout


# ============================================================================================
# scope item 5 (D53) — the asymmetry: explicit nonexistent path refuses on BOTH flags
# ============================================================================================

def test_explicit_nonexistent_experiments_dir_now_refuses(tmp_path):
    """Replaces the deleted `test_absent_scan_root_is_not_a_refusal` /
    `test_control_a_named_but_absent_scan_dir_is_still_not_a_refusal`: an EXPLICITLY-named
    `--experiments-dir` that does not exist now refuses, matching `--protect-file`."""
    (tmp_path / "runs").mkdir()
    r = _cli("--roots", "runs", "--experiments-dir", "no_such_dir", "--verify-manifests",
             cwd=str(tmp_path))
    assert r.returncode != 0, r.stdout + r.stderr
    assert "DOES NOT EXIST" in r.stderr, r.stderr
    assert "no_such_dir" in r.stderr


def test_control_default_omitted_experiments_dir_absent_stays_silent(tmp_path):
    """CONTROL, unchanged residual: the flag OMITTED (using the default `experiments`) which
    happens not to exist is still a silent no-op — "a host with no `experiments/` has nothing to
    say" — because there was never an explicit operator claim to be wrong about."""
    (tmp_path / "runs").mkdir()
    r = _cli("--roots", "runs", "--verify-manifests", cwd=str(tmp_path))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "verify-manifests: OK" in r.stdout


def test_abbreviated_flag_spelling_is_detected_as_explicit_too(tmp_path):
    """The grammar doc's claim: "explicit" is detected from whether argparse consumed a value
    for the option, not from sniffing the literal flag spelling — an abbreviation must refuse
    identically to the full name."""
    (tmp_path / "runs").mkdir()
    r = _cli("--roots", "runs", "--exp", "no_such_dir", "--verify-manifests", cwd=str(tmp_path))
    assert r.returncode != 0, r.stdout + r.stderr
    assert "DOES NOT EXIST" in r.stderr, r.stderr


def test_red_when_the_explicit_nonexistent_guard_is_removed_at_its_real_drop_site(tmp_path):
    (tmp_path / "runs").mkdir()
    mutant = _mutant(
        tmp_path, "no_explicit_nonexistent_guard",
        "            if (_sd.strip() and not os.path.isdir(_sd)\n"
        "                    and classify_entry(_sd).kind == \"unstatable\"):",
        "            if False:")

    m = _cli_script(mutant, "--roots", "runs", "--experiments-dir", "no_such_dir",
                    "--verify-manifests", cwd=str(tmp_path))
    assert m.returncode == 0, m.stdout + m.stderr      # the pre-D53 asymmetry, reproduced

    r = _cli("--roots", "runs", "--experiments-dir", "no_such_dir", "--verify-manifests",
             cwd=str(tmp_path))
    assert r.returncode != 0, r.stdout + r.stderr


# ============================================================================================
# scope item 4 NIT — nested/duplicate --experiments-dir values do not double-count
# ============================================================================================

def test_nested_experiments_dir_values_do_not_double_count(tmp_path):
    root = _spec_tree(tmp_path, "protected:\n- runs/a\n")
    exp = root / "experiments"
    nested = exp / "999_z"   # already exists (created by _spec_tree) and carries the SPEC.md

    solo = _cli("--roots", "runs", "--experiments-dir", str(exp), "--verify-manifests",
               cwd=str(root))
    assert solo.returncode == 0, solo.stdout + solo.stderr

    doubled = _cli("--roots", "runs", "--experiments-dir", str(exp), str(nested),
                   "--verify-manifests", cwd=str(root))
    assert doubled.returncode == 0, doubled.stdout + doubled.stderr
    assert "1 protect" in doubled.stdout, doubled.stdout      # not 2 — the same manifest, once
    # the SAME dirs-walked / manifest-files-read counts as the solo invocation, not doubled
    for line in solo.stdout.splitlines():
        if line.startswith("[ckpt_gc] manifest scan:"):
            solo_scan_line = line
    for line in doubled.stdout.splitlines():
        if line.startswith("[ckpt_gc] manifest scan:"):
            doubled_scan_line = line
    assert "1 manifest file(s) read" in solo_scan_line
    assert "1 manifest file(s) read" in doubled_scan_line
    assert "collapsed into an already-scanned directory" in doubled.stdout


def test_duplicate_identical_experiments_dir_values_do_not_double_count(tmp_path):
    root = _spec_tree(tmp_path, "protected:\n- runs/a\n")
    exp = root / "experiments"
    r = _cli("--roots", "runs", "--experiments-dir", str(exp), str(exp), "--verify-manifests",
             cwd=str(root))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "1 protect" in r.stdout
    assert "1 manifest file(s) read" in r.stdout


def test_red_when_the_nested_scan_dir_dedup_is_removed_at_its_real_drop_site(tmp_path):
    root = _spec_tree(tmp_path, "protected:\n- runs/a\n")
    exp = root / "experiments"
    nested = exp / "999_z"

    mutant = _mutant(
        tmp_path, "no_nested_scan_dir_dedup",
        "    scan_dirs, _dropped_nested = _dedupe_nested_scan_dirs(scan_dirs)",
        "    scan_dirs, _dropped_nested = scan_dirs, []")

    m = _cli_script(mutant, "--roots", "runs", "--experiments-dir", str(exp), str(nested),
                    "--verify-manifests", cwd=str(root))
    assert m.returncode == 0, m.stdout + m.stderr
    for line in m.stdout.splitlines():
        if line.startswith("[ckpt_gc] manifest scan:"):
            assert "2 manifest file(s) read" in line, line   # the reproduced double-count

    r = _cli("--roots", "runs", "--experiments-dir", str(exp), str(nested),
             "--verify-manifests", cwd=str(root))
    assert r.returncode == 0, r.stdout + r.stderr
    for line in r.stdout.splitlines():
        if line.startswith("[ckpt_gc] manifest scan:"):
            assert "1 manifest file(s) read" in line, line


# ============================================================================================
# scope item 4 NIT — `load_protect_file` turning an indented `#` into an entry
# ============================================================================================

def test_indented_comment_in_protect_file_is_not_ingested_as_a_path(tmp_path):
    root = _spec_tree(tmp_path, _TWO_ITEM_LIST := "protected:\n- runs/a\n")
    pf = root / "PROTECTED_LIST.txt"
    pf.write_text("runs/a\n  # this is a comment, not a path\n")
    r = _cli("--roots", "runs", "--protect-file", str(pf), "--experiments-dir", "experiments",
             "--verify-manifests", cwd=str(root))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "1 entrie(s) read" in r.stdout, r.stdout    # not 2 -- the comment is not a path


def test_control_a_column_zero_comment_in_protect_file_was_already_fine(tmp_path):
    root = _spec_tree(tmp_path, "protected:\n- runs/a\n")
    pf = root / "PROTECTED_LIST.txt"
    pf.write_text("runs/a\n# this is a comment, not a path\n")
    r = _cli("--roots", "runs", "--protect-file", str(pf), "--experiments-dir", "experiments",
             "--verify-manifests", cwd=str(root))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "1 entrie(s) read" in r.stdout, r.stdout


def test_red_when_the_indented_comment_fix_is_reverted_at_its_real_drop_site(tmp_path):
    root = _spec_tree(tmp_path, "protected:\n- runs/a\n")
    pf = root / "PROTECTED_LIST.txt"
    pf.write_text("runs/a\n  # this is a comment, not a path\n")

    mutant = _mutant(
        tmp_path, "no_indented_comment_fix",
        '            return ({os.path.realpath(l.strip()) for l in fh\n'
        '                     if l.strip() and not l.strip().startswith("#")}, None)',
        '            return ({os.path.realpath(l.strip()) for l in fh\n'
        '                     if l.strip() and not l.startswith("#")}, None)')

    m = _cli_script(mutant, "--roots", "runs", "--protect-file", str(pf), "--experiments-dir",
                    "experiments", "--verify-manifests", cwd=str(root))
    assert m.returncode == 0, m.stdout + m.stderr
    assert "2 entrie(s) read" in m.stdout, m.stdout    # the reproduced bogus entry

    r = _cli("--roots", "runs", "--protect-file", str(pf), "--experiments-dir", "experiments",
             "--verify-manifests", cwd=str(root))
    assert "1 entrie(s) read" in r.stdout, r.stdout


# ============================================================================================
# CYCLE-2 BLOCKER — the collapse must not erase 881I's symlinked-scan-root REFUSAL
#
# `_dedupe_nested_scan_dirs` runs on `realpath`, BEFORE `load_manifests`, and the first cut
# collapsed a symlinked scan dir into the real one it resolves to. That erased the very input
# `_walk_scan_dirs` needs to raise "manifest reached only by crossing a SYMLINKED DIRECTORY"
# (`_root_via_link`): a shipped 881I refusal, weakened at exit 0 by a COUNTING fix. Nothing
# covered a TWO-scan-dir invocation where one of them is a link — 881I's own control
# (`test_live_symlinked_experiment_dir_still_refuses_with_its_own_message`) names a single scan
# dir, which is why it stayed green through the defect.
# ============================================================================================

def _symlinked_root_tree(tmp_path):
    root = _spec_tree(tmp_path, "protected:\n- runs/a\n")
    (root / "experiments_link").symlink_to("experiments")
    return root


def test_symlinked_scan_root_beside_the_real_one_still_refuses(tmp_path):
    """`--experiments-dir experiments experiments_link`: the link resolves to the real dir, so the
    collapse WOULD drop it — and with it 881I's refusal. It must survive, at exit 2."""
    root = _symlinked_root_tree(tmp_path)
    r = _cli("--roots", "runs", "--experiments-dir", "experiments", "experiments_link",
             "--verify-manifests", cwd=str(root))
    assert r.returncode == 2, r.stdout + r.stderr
    assert "SYMLINKED DIRECTORY" in r.stderr, r.stderr
    assert "collapsed into an already-scanned directory" not in r.stdout, r.stdout


def test_symlinked_scan_root_first_also_still_refuses(tmp_path):
    """The other order: the LINK is named first, so it would be the SURVIVOR and the real dir the
    dropped duplicate. A symlinked dir must not absorb another dir's scan either — both are kept,
    and the refusal still fires."""
    root = _symlinked_root_tree(tmp_path)
    r = _cli("--roots", "runs", "--experiments-dir", "experiments_link", "experiments",
             "--verify-manifests", cwd=str(root))
    assert r.returncode == 2, r.stdout + r.stderr
    assert "SYMLINKED DIRECTORY" in r.stderr, r.stderr
    assert "collapsed into an already-scanned directory" not in r.stdout, r.stdout
    # the REAL dir was still scanned in full — the refusal is additional, not a replacement
    assert "1 protect" in r.stdout, r.stdout


def test_symlink_INSIDE_the_tree_as_a_second_scan_dir_still_refuses(tmp_path):
    """The nested variant of the same shape: the second scan dir is a link to a subdirectory of
    the first. Its realpath is NESTED inside the first, so the collapse would drop it — but the
    surviving walk reaches that subdir by its REAL name and never marks it `via_symlink`, so the
    refusal would vanish rather than merely move."""
    root = _spec_tree(tmp_path, "protected:\n- runs/a\n")
    (root / "experiments" / "link_z").symlink_to("999_z")
    r = _cli("--roots", "runs", "--experiments-dir", "experiments", "experiments/link_z",
             "--verify-manifests", cwd=str(root))
    assert r.returncode == 2, r.stdout + r.stderr
    assert "SYMLINKED DIRECTORY" in r.stderr, r.stderr


def test_control_a_single_symlinked_scan_dir_is_unaffected(tmp_path):
    """CONTROL (881I's own shape, restated here): one scan dir, a link — refuses exactly as it
    did before this row. The collapse is not involved at all with a single dir."""
    root = _symlinked_root_tree(tmp_path)
    r = _cli("--roots", "runs", "--experiments-dir", "experiments_link", "--verify-manifests",
             cwd=str(root))
    assert r.returncode == 2, r.stdout + r.stderr
    assert "SYMLINKED DIRECTORY" in r.stderr, r.stderr


def test_red_when_the_symlink_exclusion_is_removed_at_its_real_drop_site(tmp_path):
    """Let symlinked scan dirs back into the collapse candidate set and the 881I refusal is
    ERASED at exit 0 — the shipped BLOCKER, reproduced from scratch."""
    root = _symlinked_root_tree(tmp_path)
    mutant = _mutant(
        tmp_path, "collapse_ignores_symlinks",
        "    real = [(idx, rp) for idx, (ap, rp) in enumerate(entries) if ap == rp]",
        "    real = [(idx, rp) for idx, (ap, rp) in enumerate(entries)]")

    m = _cli_script(mutant, "--roots", "runs", "--experiments-dir", "experiments",
                    "experiments_link", "--verify-manifests", cwd=str(root))
    assert m.returncode == 0, m.stdout + m.stderr          # the reproduced suppression
    assert "SYMLINKED DIRECTORY" not in m.stderr, m.stderr

    r = _cli("--roots", "runs", "--experiments-dir", "experiments", "experiments_link",
             "--verify-manifests", cwd=str(root))
    assert r.returncode == 2, r.stdout + r.stderr
    assert "SYMLINKED DIRECTORY" in r.stderr, r.stderr


# ============================================================================================
# CYCLE-2 MAJOR — a collapsed dir must be PROVABLY REACHED by the surviving walk
#
# `_walk_scan_dirs` prunes `.git` and `__pycache__` from its descent, so path containment is not
# reachability. The first cut collapsed on containment alone: an explicitly-named
# `experiments/__pycache__` scan dir was dropped into `experiments`, whose walk never descends
# into it — 1 manifest read / 1 protect became 0 read / 0 protect, at exit 0. That is a SILENTLY
# WITHDRAWN protection (deletable checkpoint under 881D), not a double count.
# ============================================================================================

def _pruned_component_tree(tmp_path, component):
    root = tmp_path / "tree"
    (root / "experiments" / component).mkdir(parents=True)
    (root / "runs" / "a").mkdir(parents=True)
    (root / "experiments" / component / "SPEC.md").write_text("protected:\n- runs/a\n")
    return root


@pytest.mark.parametrize("component", ["__pycache__", ".git"])
def test_scan_dir_behind_a_pruned_component_is_not_collapsed_away(tmp_path, component):
    """Both names `_walk_scan_dirs` refuses to descend into. Naming one EXPLICITLY as a scan dir
    is the operator asking for it by hand, and it must still be scanned."""
    root = _pruned_component_tree(tmp_path, component)
    r = _cli("--roots", "runs", "--experiments-dir", "experiments", f"experiments/{component}",
             "--verify-manifests", cwd=str(root))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "1 manifest file(s) read" in r.stdout, r.stdout
    assert "1 protect" in r.stdout, r.stdout
    assert "collapsed into an already-scanned directory" not in r.stdout, r.stdout


def test_control_the_ancestor_alone_really_does_not_reach_it(tmp_path):
    """The premise of the test above, proven rather than assumed: with ONLY `experiments` named,
    the manifest under `__pycache__` is genuinely invisible (that is the walk's own prune rule,
    unchanged by this row) — which is exactly why collapsing the explicit request loses it."""
    root = _pruned_component_tree(tmp_path, "__pycache__")
    r = _cli("--roots", "runs", "--experiments-dir", "experiments", "--verify-manifests",
             cwd=str(root))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "0 manifest file(s) read" in r.stdout, r.stdout
    assert "0 protect" in r.stdout, r.stdout


def test_red_when_the_reachability_condition_is_removed_at_its_real_drop_site(tmp_path):
    """Collapse on containment alone again and the protection silently disappears at exit 0 —
    the shipped MAJOR, reproduced."""
    root = _pruned_component_tree(tmp_path, "__pycache__")
    mutant = _mutant(
        tmp_path, "collapse_without_reachability",
        "            if not _walk_reaches(other_rp, rp):",
        "            if False:")

    m = _cli_script(mutant, "--roots", "runs", "--experiments-dir", "experiments",
                    "experiments/__pycache__", "--verify-manifests", cwd=str(root))
    assert m.returncode == 0, m.stdout + m.stderr
    assert "0 manifest file(s) read" in m.stdout, m.stdout     # the reproduced withdrawal
    assert "0 protect" in m.stdout, m.stdout

    r = _cli("--roots", "runs", "--experiments-dir", "experiments", "experiments/__pycache__",
             "--verify-manifests", cwd=str(root))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "1 manifest file(s) read" in r.stdout, r.stdout
    assert "1 protect" in r.stdout, r.stdout


def test_the_dedup_prune_list_mirrors_the_real_walk():
    """DRIFT GUARD. `_WALK_PRUNED_DIRNAMES` is a MIRROR of the descent filter literal inside
    `_walk_scan_dirs` (left byte-identical to HEAD on purpose). If that literal ever gains or
    loses a name, the collapse's reachability test silently over-reaches again — so the two are
    compared here rather than trusted to stay in step."""
    src = open(CKPT_GC).read()
    anchor = '                if d in (".git", "__pycache__"):'
    assert src.count(anchor) == 1, "the walk's descent filter moved or was reworded"
    assert gc._WALK_PRUNED_DIRNAMES == (".git", "__pycache__"), gc._WALK_PRUNED_DIRNAMES


def test_walk_reaches_is_about_the_walk_not_the_path():
    """The unit statement, so the distinction survives independently of any tree fixture."""
    assert gc._walk_reaches("/a/b", "/a/b") is True
    assert gc._walk_reaches("/a/b", "/a/b/c") is True
    assert gc._walk_reaches("/a/b", "/a/c") is False
    assert gc._walk_reaches("/a/b", "/a/bc") is False          # prefix, not a path component
    assert gc._walk_reaches("/a/b", "/a/b/__pycache__") is False
    assert gc._walk_reaches("/a/b", "/a/b/.git/x") is False
    assert gc._walk_reaches("/a/b/__pycache__", "/a/b/__pycache__/x") is True   # rooted THERE


# ============================================================================================
# CYCLE-2 MAJOR — the MD032 span must cover the gap it skipped (parse <-> strip round-trip)
#
# `strip_spans`' contract is "remove exactly what THIS parse consumed". The gap-skip advanced the
# item index without advancing the span's right edge, so the span was SHORT by the gap and its
# edge landed inside the last item line: `strip_spans` cut mid-item (`'/b\n'`) or left a whole
# item behind. The 881G suite shipped without a single span assertion (`_p_spans` / `_s_spans`
# discarded), which is why this reached a review instead of a test.
# ============================================================================================

_MD032_ROUNDTRIP_SHAPES = [
    ("no_gap",          "protected:\n- runs/a\n- runs/b\n"),
    ("one_blank",       "protected:\n\n- runs/a\n- runs/b\n"),
    ("three_blanks",    "protected:\n\n\n\n- runs/a\n- runs/b\n"),
    ("nine_blanks",     "protected:\n" + "\n" * 9 + "- runs/a\n- runs/b\n"),
    ("ws_only_gap",     "protected:\n   \n\t\n- runs/a\n- runs/b\n"),
]


@pytest.mark.parametrize("name,body", _MD032_ROUNDTRIP_SHAPES,
                         ids=[s[0] for s in _MD032_ROUNDTRIP_SHAPES])
def test_md032_parse_then_strip_removes_the_block_and_nothing_else(name, body):
    """The block alone: a strip must consume the WHOLE block, leaving the empty string — never a
    surviving item and never a mangled fragment of one."""
    entries, spans, bad = gc.parse_spec_manifest(body, "SPEC.md")
    assert not bad, (name, bad)
    assert [e.path for e in entries] == ["runs/a", "runs/b"], (name, entries)
    assert gc.strip_spans(body, spans) == "", (name, repr(gc.strip_spans(body, spans)))


@pytest.mark.parametrize("name,body", _MD032_ROUNDTRIP_SHAPES,
                         ids=[s[0] for s in _MD032_ROUNDTRIP_SHAPES])
def test_md032_parse_then_strip_leaves_the_prose_tail_intact(name, body):
    """The same shapes with prose after them: the strip removes the block and leaves the prose
    byte-for-byte, with no fragment of any item surviving into it."""
    text = body + "\nprose tail\n"
    entries, spans, bad = gc.parse_spec_manifest(text, "SPEC.md")
    assert not bad, (name, bad)
    out = gc.strip_spans(text, spans)
    assert out == "\nprose tail\n", (name, repr(out))
    assert "runs/a" not in out and "runs/b" not in out, (name, repr(out))
    assert "-" not in out, (name, repr(out))


def test_md032_span_ends_exactly_at_the_end_of_the_last_item_line():
    """The span's right edge, stated as a number rather than inferred from a strip: it must land
    ON a line boundary, never inside a line."""
    body = "protected:\n\n\n- runs/a\n- runs/b\n\nprose\n"
    _entries, spans, bad = gc.parse_spec_manifest(body, "SPEC.md")
    assert not bad, bad
    assert len(spans) == 1, spans
    start, end = spans[0]
    assert start == 0
    assert end == body.index("- runs/b\n") + len("- runs/b\n"), (end, spans)
    assert body[end - 1] == "\n", repr(body[end - 5:end])


def test_red_when_the_md032_span_gap_accounting_is_removed_at_its_real_drop_site(tmp_path):
    """Drop the gap out of the span again and the strip goes back to cutting mid-item — the
    shipped MAJOR, reproduced through the REAL module (imported from the mutant source, since
    this defect lives entirely inside the parse and never reaches the CLI)."""
    import importlib.util

    mutant = _mutant(
        tmp_path, "no_md032_span_gap",
        "                    end += sum(len(_gap_line) for _gap_line in item_lines[i:j])\n",
        "")
    spec = importlib.util.spec_from_file_location("ckpt_gc_mut_span", mutant)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    body = "protected:\n\n\n\n- runs/a\n- runs/b\n"
    m_entries, m_spans, m_bad = mod.parse_spec_manifest(body, "SPEC.md")
    assert not m_bad and len(m_entries) == 2          # parses the same...
    assert mod.strip_spans(body, m_spans) == "/b\n"   # ...and strips a MANGLED FRAGMENT

    entries, spans, bad = gc.parse_spec_manifest(body, "SPEC.md")
    assert not bad and len(entries) == 2
    assert gc.strip_spans(body, spans) == ""          # the shipped code consumes the whole block


def test_control_non_md032_spans_are_unchanged_by_the_gap_fix():
    """CONTROL: the shapes with NO gap must produce byte-identical spans to before the fix — the
    gap accounting is reached only through the MD032 branch."""
    body = "protected:\n- runs/a\n- runs/b\n\nprose\n"
    _e, spans, bad = gc.parse_spec_manifest(body, "SPEC.md")
    assert not bad
    assert spans == [(0, len("protected:\n- runs/a\n- runs/b\n"))], spans


# ============================================================================================
# CYCLE-2 NITs — the fail-closed guard must not fail OPEN, and the scan line must count what the
# OPERATOR named
# ============================================================================================

def test_experiments_dir_explicit_is_read_fail_closed(tmp_path):
    """`getattr(a, "experiments_dir_explicit", False)` silently replayed the pre-D53 no-op for any
    caller that did not set the attribute — a fail-closed guard defaulting to the behaviour it
    removes. A plain attribute access raises instead."""
    import argparse as _ap
    ns = _ap.Namespace(protect_file=None, experiments_dir=["no_such_dir"], roots=["runs"],
                       score_regex=None, log_glob="*.log")
    assert not hasattr(ns, "experiments_dir_explicit")
    with pytest.raises(AttributeError):
        gc._run_manifest_check(ns)


def test_scan_line_counts_the_scan_dirs_the_operator_named(tmp_path):
    """The collapse rewrites `scan_dirs` before `load_manifests` fills `stats`, so "requested"
    used to be read off the COLLAPSED list: an operator who named two dirs was told "1 of 1".
    It now reads "1 of 2", with the collapse line accounting for the difference."""
    root = _spec_tree(tmp_path, "protected:\n- runs/a\n")
    exp = root / "experiments"
    r = _cli("--roots", "runs", "--experiments-dir", str(exp), str(exp / "999_z"),
             "--verify-manifests", cwd=str(root))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "1 of 2 scan dir(s) visited" in r.stdout, r.stdout
    assert "1 nested/duplicate scan dir(s) collapsed" in r.stdout, r.stdout


# ============================================================================================
# GATE (i) — the six doc defects, each amended claim pinned
#
# An isolated agent authored 46 cases FROM THE DOC ALONE and another ran them through the real
# CLI: 40 agreed, 3 surprised, 3 were underspecified. Every divergence was a DOC defect (the tool
# is self-consistent), so the doc was amended — and each amendment is pinned here, because a
# grammar doc that is being mirrored into 881D as a corpus precondition must not be able to drift
# back into being wrong without a test failing.
# ============================================================================================

def test_doc1_a_dangling_entry_exits_2(tmp_path):
    """AMENDMENT 1: "NOT malformed / does not poison the rest of the file" reads as ACCEPT. It is
    about intra-file poisoning only — the CLI-level verdict is a REFUSAL at exit 2."""
    root = _spec_tree(tmp_path, "protected:\n- runs/a\n- runs/does_not_exist\n")
    r = _cli("--roots", "runs", "--experiments-dir", "experiments", "--verify-manifests",
             cwd=str(root))
    assert r.returncode == 2, r.stdout + r.stderr
    assert "DANGLING ENTRY" in r.stderr, r.stderr
    assert "1 protect" in r.stdout and "1 dangling" in r.stdout, r.stdout   # the other entry parsed


def test_doc2_a_fenced_example_closed_tight_refuses_the_whole_file(tmp_path):
    """AMENDMENT 2: "a well-formed header inside a fence can only ever ADD protection" is FALSE
    for the most natural shape — the closing fence line, immediately after the last item, is a
    non-item line inside a started list and WITHDRAWS the whole file."""
    root = _spec_tree(tmp_path, "prose\n\n```\nprotected:\n- runs/a\n```\n\nmore prose\n")
    r = _cli("--roots", "runs", "--experiments-dir", "experiments", "--verify-manifests",
             cwd=str(root))
    assert r.returncode == 2, r.stdout + r.stderr
    assert "INTERRUPTED" in r.stderr, r.stderr
    assert "0 protect" in r.stdout, r.stdout


def test_doc2_a_blank_line_before_the_closing_fence_accepts(tmp_path):
    """AMENDMENT 2, the other half: the documented remedy actually works, and the entries inside
    the fenced example are REAL entries."""
    root = _spec_tree(tmp_path, "prose\n\n```\nprotected:\n- runs/a\n\n```\n\nmore prose\n")
    r = _cli("--roots", "runs", "--experiments-dir", "experiments", "--verify-manifests",
             cwd=str(root))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "1 protect" in r.stdout, r.stdout


def test_doc3_a_non_item_first_line_refuses_as_EMPTY_not_INTERRUPTED(tmp_path):
    """AMENDMENT 3: the EMPTY/INTERRUPTED collision, decided. The INTERRUPTED branch needs >=1
    consumed item, so a non-item FIRST line refuses as EMPTY even with items below it."""
    root = _spec_tree(tmp_path, "protected:\nhello\n- runs/a\n")
    r = _cli("--roots", "runs", "--experiments-dir", "experiments", "--verify-manifests",
             cwd=str(root))
    assert r.returncode == 2, r.stdout + r.stderr
    assert "is EMPTY" in r.stderr, r.stderr
    assert "INTERRUPTED" not in r.stderr, r.stderr


def test_doc3_control_a_multi_token_first_item_keeps_its_own_message(tmp_path):
    """AMENDMENT 3's stated exception: an attempted item naming two tokens refuses wherever it
    sits — as the FIRST line here — under its OWN distinct MORE-THAN-ONE-TOKEN message, never as
    EMPTY, because "names nothing" would be false about it."""
    root = _spec_tree(tmp_path, "protected:\n- runs/a runs/b\n")
    r = _cli("--roots", "runs", "--experiments-dir", "experiments", "--verify-manifests",
             cwd=str(root))
    assert r.returncode == 2, r.stdout + r.stderr
    assert "MORE THAN ONE token" in r.stderr, r.stderr
    assert "is EMPTY" not in r.stderr, r.stderr


_ITEM_MICRO_GRAMMAR = [
    ("leading_indentation", "protected:\n  - runs/a\n", "protect:   runs/a\n"),
    ("no_space_after_dash", "protected:\n-runs/a\n", "protect:runs/a\n"),
    ("trailing_whitespace", "protected:\n- runs/a   \n", "protect: runs/a   \n"),
    ("indented_directive",  "protected:\n\t- runs/a\n", "  protect: runs/a\n"),
]


@pytest.mark.parametrize("name,spec_body,protected_body", _ITEM_MICRO_GRAMMAR,
                         ids=[s[0] for s in _ITEM_MICRO_GRAMMAR])
def test_doc4_item_micro_grammar_is_accepted_on_both_paths(name, spec_body, protected_body):
    """AMENDMENT 4: measured, all ACCEPTED, none stated before. Asserted on BOTH parsers, since
    the doc now claims the two share this micro-grammar (gate iii)."""
    s_entries, _s, s_bad = gc.parse_spec_manifest(spec_body, "SPEC.md")
    p_entries, _p, p_bad = gc.parse_protected_txt(protected_body, "PROTECTED.txt")
    assert not s_bad, (name, s_bad)
    assert not p_bad, (name, p_bad)
    assert [e.path for e in s_entries] == ["runs/a"], (name, s_entries)
    assert [e.path for e in p_entries] == ["runs/a"], (name, p_entries)


def test_doc4_control_the_header_asymmetry_is_real(tmp_path):
    """AMENDMENT 4's stated ASYMMETRY: the same indentation that is fine on an ITEM is a NEAR-MISS
    refusal on the HEADER."""
    root = _spec_tree(tmp_path, "  protected:\n  - runs/a\n")
    r = _cli("--roots", "runs", "--experiments-dir", "experiments", "--verify-manifests",
             cwd=str(root))
    assert r.returncode == 2, r.stdout + r.stderr
    assert "NEARLY spells" in r.stderr, r.stderr


_WHITESPACE_BLANK_CONTEXTS = [
    # (name, body, expect_accept, expect_n) -- the SAME three contexts the doc names
    ("md032_gap",       "protected:\n   \n- runs/a\n", True, 1),
    ("legitimate_end",  "protected:\n- runs/a\n \t \nprose\n", True, 1),
    ("split_gap",       "protected:\n- runs/a\n   \n- runs/b\n", False, 0),
]


@pytest.mark.parametrize("name,body,expect_accept,expect_n", _WHITESPACE_BLANK_CONTEXTS,
                         ids=[s[0] for s in _WHITESPACE_BLANK_CONTEXTS])
def test_doc5_whitespace_only_line_counts_as_blank_in_all_three_contexts(name, body,
                                                                        expect_accept, expect_n):
    """AMENDMENT 5: a whitespace-only line behaves identically to an empty one in the MD032 gap,
    the legitimate end, and the SPLIT gap — including refusing where an empty line would."""
    entries, _spans, bad = gc.parse_spec_manifest(body, "SPEC.md")
    assert (not bad) == expect_accept, (name, bad)
    assert len(entries) == expect_n, (name, entries)
    empty_line_twin = body.replace("   \n", "\n").replace(" \t \n", "\n")
    t_entries, _t, t_bad = gc.parse_spec_manifest(empty_line_twin, "SPEC.md")
    assert (not t_bad) == expect_accept, (name, t_bad)
    assert len(t_entries) == len(entries), (name, t_entries, entries)


def test_doc6_entry_paths_resolve_against_the_process_cwd(tmp_path):
    """AMENDMENT 6: entries realpath against the CWD, NOT against the manifest's own directory —
    never stated, and it made a doc-only verifier's whole first pass falsely refuse. Same tree,
    same manifest, two CWDs: one grants, the other DANGLES."""
    root = _spec_tree(tmp_path, "protected:\n- runs/a\n")
    (root / "sub").mkdir()

    at_root = _cli("--roots", "runs", "--experiments-dir", "experiments", "--verify-manifests",
                   cwd=str(root))
    assert at_root.returncode == 0, at_root.stdout + at_root.stderr
    assert "1 protect, 0 covers, 0 dangling" in at_root.stdout, at_root.stdout

    from_sub = _cli("--roots", "../runs", "--experiments-dir", "../experiments",
                    "--verify-manifests", cwd=str(root / "sub"))
    assert from_sub.returncode == 2, from_sub.stdout + from_sub.stderr
    assert "DANGLING ENTRY" in from_sub.stderr, from_sub.stderr
    assert "0 protect, 0 covers, 1 dangling" in from_sub.stdout, from_sub.stdout


def test_the_grammar_doc_is_mirrored_in_both_locations():
    """Scope item 1: the doc lives in TWO places and they must not drift. Byte-identity is
    impossible (one is an rst-flavoured module docstring, the other a markdown task file), so the
    load-bearing CLAIMS are compared instead — every amendment this cycle added, on both sides."""
    module_doc = open(CKPT_GC).read().split('"""')[1]
    task_doc = open(os.path.join(REPO, "tasks", "TASK_881D_PROTECTION_DELETER.md")).read()
    for claim in ("EXIT 2, never 0",
                  "RESOLVE",
                  "MICRO-GRAMMAR IS DELIBERATELY FORGIVING",
                  "IS A BLANK LINE, uniformly",
                  "EMPTY vs INTERRUPTED",
                  "FENCES ARE NOT INVISIBLE, THEY ARE UNSEEN",
                  "NOT TRUE OF THE DELETE PATH",
                  "MAY NEVER CHANGE A VERDICT"):
        assert claim.upper() in module_doc.upper(), f"missing from ckpt_gc.py docstring: {claim}"
        assert claim.upper() in task_doc.upper(), f"missing from TASK_881D: {claim}"
