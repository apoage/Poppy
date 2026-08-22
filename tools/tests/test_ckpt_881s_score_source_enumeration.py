"""TASK_881S — score-source enumeration in `ckpt_gc --verify-manifests` (D50).

`_find_score_dirs` enumerated a directory only when it held a `SCORE_BASENAMES` file
(`scores.jsonl` / `checkpoints_index.json`), so a run scored PURELY through
`--score-regex`/`--log-glob` — the module docstring's own THIRD fallback source — was never
handed to `_load_scores_strict`, and that reader's log-file shape guard
(`_score_source_shape_reason`, in its `score_regex` branch) was dead code for exactly that input
class: `--verify-manifests --score-regex ... --log-glob <FIFO>` with no `scores.jsonl` anywhere
reported "OK" at exit 0, while the identical flags on the real (non-`--verify-manifests`) path —
which has no shape guard at all — hang forever on `open()`.

Fix, in `tools/ckpt_gc.py`:
  * `_find_log_glob_score_dirs` — a NEW, separate enumeration (over the same `--roots`) that finds
    every directory with a `--log-glob` match, independently of `SCORE_BASENAMES`, and hands each
    one to the SAME `_load_scores_strict` the co-located-score path already used. `_find_score_dirs`
    itself is untouched (gate (iv): byte-identical to the 881I-shipped tree).
  * `_run_manifest_check` wires that enumeration in as a second, additive pass over the score dirs.
  * `_bare_log_glob_matches` / `_load_bare_log_glob_scores_strict` (ESCALATED CYCLE, BLOCKER) —
    canon's reader globs TWO terms, `os.path.join(run_dir, log_glob)` AND a bare, CWD-relative
    `log_glob`. The first pass mirrored only the first, so a source reachable only through the
    second was still never enumerated and still returned a false OK (a real checkpoint dir under
    `--roots` plus a FIFO `./topfile.log`). The bare term is now read ONCE PER INVOCATION through
    `_load_scores_strict` itself, and `--help` names the source in the same change.
  * `_find_log_glob_score_dirs`'s RESOLVED-PATH COLLISION dedup (CYCLE 3, MAJOR-B) — one physical
    file must yield one candidate and one report however the pattern is spelled. Cycle 2 shipped
    an `os.path.isabs(log_glob)` guard, which is a syntactic pattern-match on ONE spelling of the
    real invariant (DIRECTORY-INDEPENDENCE: the pattern resolves to the same physical file
    whichever `dirpath` it is joined to) and misses the others. Reproduced on this tree: 5 sibling
    run dirs, no `scores.jsonl`, one shared FIFO one level above them, `--log-glob '../shared.log'`
    -> "5 additional score dir(s) found, 5 examined" and FIVE `VERIFY-MANIFESTS REFUSAL` lines for
    ONE physical file, because `runs/rN/../shared.log` is not absolute. The guard is now the
    invariant itself: every glob match is normalised with `os.path.realpath` and a directory is
    enumerated only if it contributes a resolved path no earlier candidate covers, with the claim
    set seeded from the bare CWD term and from the `already_found` (co-located-score) dirs so the
    dedup holds ACROSS the three passes. The absolute case keeps its previous output byte for byte
    — it is now a consequence of the invariant instead of a special case beside it.
  * `_walk_error` (inside `load_manifests`) no longer double-reports a dangling MANIFEST-basename
    symlink: `_walk_scan_dirs`'s generic `DanglingScanLink` is suppressed for a `MANIFEST_BASENAMES`
    name, because `manifest_shape_refusal` a few lines later reports the identical `fp` more
    specifically. One real defect, one report (scope item 3).

Covered here (gate references from TASK_881S):
  (i)   the log-glob source, reachable independently of SCORE_BASENAMES, on every shape:
        FIFO (representative of the fifo/socket/device class, matching the rest of this codebase's
        convention — see test_ckpt_881i's MANIFEST_SHAPE_TABLE), symlinked (to a regular file: read;
        to a directory: refuse), dangling; each with and without a co-located `scores.jsonl`; plus
        the literal D50 repro (an ABSOLUTE `--log-glob` path pointed straight at a FIFO) and the
        control that `--log-glob` alone (no `--score-regex`) stays inert.
  (ii)/(iii) two mutations at the real drop sites: the log-glob enumeration call in
        `_run_manifest_check`, and the dedup guard in `_walk_error`. Each mutant reproduces the
        original bug (false OK / doubled problem count) the way the real tool no longer does.

Every subprocess test RUNS the documented CLI invocation, never imports `ckpt_gc.main()` —
same discipline as TASK_881I's suite (gate (iv): "run, not imported").
"""
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CKPT_GC = os.path.join(REPO, "tools", "ckpt_gc.py")

sys.path.insert(0, os.path.join(REPO, "tools"))
import ckpt_gc as gc  # noqa: E402

_REGEX = r"Epoch (?P<epoch>\d+) reward (?P<score>[-\w.]+)"


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


def _base_tree(tmp_path):
    (tmp_path / "runs" / "r1").mkdir(parents=True)
    (tmp_path / "experiments").mkdir()
    return tmp_path


def _wide_tree(tmp_path):
    """`_base_tree` with enough directories under `--roots` that a per-DIRECTORY report and a
    per-DEFECT report are distinguishable by COUNT (5 walked dirs: runs, r1, r1/snapshot_ep1, r2,
    r2/snapshot_ep1). The escalated cycle's MAJOR was invisible to a one-run tree."""
    (tmp_path / "runs" / "r1" / "snapshot_ep1").mkdir(parents=True)
    (tmp_path / "runs" / "r2" / "snapshot_ep1").mkdir(parents=True)
    (tmp_path / "experiments").mkdir()
    return tmp_path


def _refusals(r):
    """How many DISTINCT problems the run reported, as an alert keying on the tool's output would
    count them: one `VERIFY-MANIFESTS REFUSAL` line per problem."""
    return r.stderr.count("VERIFY-MANIFESTS REFUSAL")


# ============================================================================================
# gate (i) — the log-glob source, reachable independently of SCORE_BASENAMES, every shape
# ============================================================================================

def test_d50_literal_repro_absolute_fifo_log_glob_refuses_not_falsely_ok(tmp_path):
    """THE reproduction as stated in D50: `--log-glob` given an ABSOLUTE path straight at a FIFO,
    no `scores.jsonl`/`checkpoints_index.json` anywhere. Pre-fix this printed "verify-manifests:
    OK" at exit 0; it must now refuse, and it must not hang doing it.

    The COUNT assertions are load-bearing, not decoration (escalated cycle, MAJOR): the first
    version of this test asserted only "refuses / exit 2", and passed while ONE FIFO produced ONE
    refusal PER WALKED DIRECTORY — `os.path.join(dirpath, log_glob)` discards `dirpath` for an
    absolute pattern, so every directory looked like it held the match. Substring-presence alone
    cannot see a problem count inflating with the size of the tree."""
    tree = _wide_tree(tmp_path)
    fifo = tmp_path / "a_fifo_log"
    os.mkfifo(fifo)

    r = _cli("--roots", "runs", "--experiments-dir", "experiments", "--score-regex", _REGEX,
             "--log-glob", str(fifo), "--verify-manifests", cwd=str(tree), timeout=15)
    assert r.returncode == 2, r.stdout + r.stderr
    assert "BLOCKS FOREVER" in r.stderr
    assert "verify-manifests: REFUSED" in r.stdout

    # ONE defect -> ONE report, on a tree with FIVE walked directories (pre-fix: five).
    assert "5 dir(s) walked" in r.stdout, r.stdout
    assert _refusals(r) == 1, r.stderr
    # ...and ONE unreadable record inside that report, not one per glob term.
    assert "1 unreadable record(s)" in r.stderr, r.stderr
    # An absolute pattern is not a per-run source: it must be reported as the single global
    # candidate, never as N "additional score dir(s)".
    assert "0 additional score dir(s) found, 0 examined" in r.stdout, r.stdout
    assert "1 file(s) matched, 1 read(s)" in r.stdout, r.stdout


def test_log_glob_fifo_relative_pattern_no_scores_jsonl_refuses_not_hangs(tmp_path):
    """The common spelling: `--log-glob '*.log'` (relative, per-run-dir), the matched file itself
    is a FIFO, and NO co-located `scores.jsonl` exists anywhere under `--roots`."""
    tree = _wide_tree(tmp_path)
    os.mkfifo(tree / "runs" / "r1" / "train.log")

    r = _cli("--roots", "runs", "--experiments-dir", "experiments", "--score-regex", _REGEX,
             "--log-glob", "*.log", "--verify-manifests", cwd=str(tree), timeout=15)
    assert r.returncode == 2, r.stdout + r.stderr
    assert "BLOCKS FOREVER" in r.stderr
    assert "train.log" in r.stderr
    # One FIFO in ONE of five walked directories: exactly one refusal, and exactly one directory
    # enumerated by the relative-pattern pass (the count that says WHERE the source was found).
    assert "5 dir(s) walked" in r.stdout, r.stdout
    assert _refusals(r) == 1, r.stderr
    assert "1 additional score dir(s) found, 1 examined" in r.stdout, r.stdout


def test_log_glob_fifo_with_co_located_scores_jsonl_still_refuses(tmp_path):
    """The SAME fifo-named-log shape, but this run dir ALSO has a real `scores.jsonl` (so
    `_find_score_dirs` already reports it via SCORE_BASENAMES). The pre-existing co-located path
    must not mask the log-glob refusal — the two enumerations examine the SAME dir once each
    reaches it, and either one finding the FIFO is enough to refuse."""
    tree = _base_tree(tmp_path)
    (tree / "runs" / "r1" / "scores.jsonl").write_text('{"epoch": 1, "score": 0.5}\n')
    os.mkfifo(tree / "runs" / "r1" / "train.log")

    r = _cli("--roots", "runs", "--experiments-dir", "experiments", "--score-regex", _REGEX,
             "--log-glob", "*.log", "--verify-manifests", cwd=str(tree), timeout=15)
    assert r.returncode == 2, r.stdout + r.stderr
    assert "BLOCKS FOREVER" in r.stderr


def test_log_glob_dangling_symlink_no_scores_jsonl_refuses(tmp_path):
    """A dangling `*.log` symlink, no co-located `scores.jsonl`: must refuse with the dangling
    diagnosis, not be silently treated as "this run is unscored"."""
    tree = _base_tree(tmp_path)
    (tree / "runs" / "r1" / "train.log").symlink_to(tree / "runs" / "r1" / "no_such_target")

    r = _cli("--roots", "runs", "--experiments-dir", "experiments", "--score-regex", _REGEX,
             "--log-glob", "*.log", "--verify-manifests", cwd=str(tree), timeout=15)
    assert r.returncode == 2, r.stdout + r.stderr
    assert "target does not resolve" in r.stderr


def test_log_glob_dangling_symlink_with_scores_jsonl_still_refuses(tmp_path):
    """Same dangling `*.log` symlink, but a co-located, VALID `scores.jsonl` this time — the
    log-glob refusal must not be masked by the primary source being clean."""
    tree = _base_tree(tmp_path)
    (tree / "runs" / "r1" / "scores.jsonl").write_text('{"epoch": 1, "score": 0.5}\n')
    (tree / "runs" / "r1" / "train.log").symlink_to(tree / "runs" / "r1" / "no_such_target")

    r = _cli("--roots", "runs", "--experiments-dir", "experiments", "--score-regex", _REGEX,
             "--log-glob", "*.log", "--verify-manifests", cwd=str(tree), timeout=15)
    assert r.returncode == 2, r.stdout + r.stderr
    assert "target does not resolve" in r.stderr


def test_log_glob_live_symlink_to_directory_no_scores_jsonl_refuses(tmp_path):
    """A LIVE symlink -> DIRECTORY named `*.log`: readable by nothing, must refuse as a directory,
    not be silently skipped the way the pre-881I `scores.jsonl` blind spot was."""
    tree = _base_tree(tmp_path)
    (tree / "sneaky").mkdir()
    (tree / "runs" / "r1" / "train.log").symlink_to(tree / "sneaky")

    r = _cli("--roots", "runs", "--experiments-dir", "experiments", "--score-regex", _REGEX,
             "--log-glob", "*.log", "--verify-manifests", cwd=str(tree), timeout=15)
    assert r.returncode == 2, r.stdout + r.stderr
    assert "score file is a DIRECTORY" in r.stderr


def test_control_log_glob_live_symlink_to_regular_file_is_still_read(tmp_path):
    """CONTROL / canon parity: a log reached through a symlink to a REGULAR file stays readable —
    a score/log source cannot grant coverage, so this row must not tighten that shared policy."""
    tree = _base_tree(tmp_path)
    (tree / "real.log").write_text("Epoch 5 reward 0.75\n")
    (tree / "runs" / "r1" / "train.log").symlink_to(tree / "real.log")

    r = _cli("--roots", "runs", "--experiments-dir", "experiments", "--score-regex", _REGEX,
             "--log-glob", "*.log", "--verify-manifests", cwd=str(tree), timeout=15)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "verify-manifests: OK" in r.stdout


def test_control_log_glob_without_score_regex_stays_inert(tmp_path):
    """`--log-glob` alone (no `--score-regex`) is inert in canon too (see the module docstring's
    scoring order) — a FIFO sitting under a matching name must not be examined, let alone hung on,
    when the flag that activates the branch was never given."""
    tree = _base_tree(tmp_path)
    os.mkfifo(tree / "runs" / "r1" / "train.log")

    r = _cli("--roots", "runs", "--experiments-dir", "experiments",
             "--log-glob", "*.log", "--verify-manifests", cwd=str(tree), timeout=15)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "verify-manifests: OK" in r.stdout


# ============================================================================================
# gate (i)/scope item 3, CYCLE 3 (MAJOR-B) — DIRECTORY-INDEPENDENCE, detected not spelled
#
# The invariant: ONE PHYSICAL FILE -> ONE CANDIDATE -> ONE REPORT, however the pattern is spelled.
# Cycle 2 implemented it as `os.path.isabs(log_glob)`, which is a syntactic pattern-match on one
# spelling of the property; `../shared.log` and a symlink both defeat it while being just as
# directory-independent. `_find_log_glob_score_dirs` now normalises every glob match with
# `os.path.realpath` and enumerates a directory only when it contributes a resolved path no
# earlier candidate already covers.
#
# The other half of the invariant is equally load-bearing and is pinned here too: two GENUINELY
# DISTINCT files must still be reported separately. A dedup that folded two real defects into one
# would satisfy every count assertion below and be strictly worse than the bug it replaced.
# ============================================================================================

def _n_run_tree(tmp_path, n=5):
    """`n` sibling run dirs under `runs/`, none of them score-bearing, plus `experiments/`."""
    for i in range(1, n + 1):
        (tmp_path / "runs" / f"r{i}").mkdir(parents=True)
    (tmp_path / "experiments").mkdir()
    return tmp_path


def test_relative_dotdot_shared_log_across_n_dirs_is_reported_once(tmp_path):
    """MAJOR-B, the reviewer's reproduction verbatim: 5 sibling run dirs with no `scores.jsonl`,
    ONE shared FIFO one level above them, `--log-glob '../shared.log'`.

    `runs/r1/../shared.log` … `runs/r5/../shared.log` are FIVE spellings of ONE physical file, and
    none of them is absolute — so the `os.path.isabs` guard never fired and the tool printed
    "5 additional score dir(s) found, 5 examined" with FIVE refusal lines. One real defect must
    not inflate a problem count an alert might key on (scope item 3), so: one candidate, one line,
    and still exit 2."""
    tree = _n_run_tree(tmp_path, 5)
    os.mkfifo(tree / "runs" / "shared.log")

    r = _cli("--roots", "runs", "--experiments-dir", "experiments", "--score-regex", _REGEX,
             "--log-glob", "../shared.log", "--verify-manifests", cwd=str(tree), timeout=15)
    assert r.returncode == 2, r.stdout + r.stderr
    assert "BLOCKS FOREVER" in r.stderr
    assert "6 dir(s) walked" in r.stdout, r.stdout          # runs + r1..r5
    assert _refusals(r) == 1, r.stderr
    assert "1 unreadable record(s)" in r.stderr, r.stderr
    assert "1 additional score dir(s) found, 1 examined" in r.stdout, r.stdout
    # Deterministic winner: the walk sorts its directory names, so the single report is always the
    # first colliding dir, never an `os.scandir`-order artefact.
    assert "score read INCOMPLETE for runs/r1:" in r.stderr, r.stderr


def test_two_spellings_that_meet_at_a_symlink_are_reported_once(tmp_path):
    """The same invariant reached through a SYMLINK instead of `..`: `runs/r1/train.log` is a real
    FIFO and `runs/r2/train.log` is a symlink to it, matched by the ordinary per-run-dir pattern
    `*.log`. Two directories, two syntactically unrelated matches, ONE physical file — so one
    report. Neither the `isabs` guard nor a purely lexical `os.path.normpath` can see this; only
    resolving the path can."""
    tree = _n_run_tree(tmp_path, 2)
    os.mkfifo(tree / "runs" / "r1" / "train.log")
    (tree / "runs" / "r2" / "train.log").symlink_to("../r1/train.log")

    r = _cli("--roots", "runs", "--experiments-dir", "experiments", "--score-regex", _REGEX,
             "--log-glob", "*.log", "--verify-manifests", cwd=str(tree), timeout=15)
    assert r.returncode == 2, r.stdout + r.stderr
    assert "BLOCKS FOREVER" in r.stderr
    assert _refusals(r) == 1, r.stderr
    assert "1 additional score dir(s) found, 1 examined" in r.stdout, r.stdout
    assert "score read INCOMPLETE for runs/r1:" in r.stderr, r.stderr


def test_plain_relative_pattern_still_reports_each_distinct_file_separately(tmp_path):
    """THE ANTI-OVER-DEDUP CONTROL. Two run dirs, each with its OWN FIFO `train.log`: two
    genuinely distinct physical files, two real defects, and they must stay two reports. A dedup
    keyed on the PATTERN (or on "the log-glob pass already found something") would collapse them
    and hide a defect — strictly worse than the inflation MAJOR-B is about."""
    tree = _n_run_tree(tmp_path, 3)
    os.mkfifo(tree / "runs" / "r1" / "train.log")
    os.mkfifo(tree / "runs" / "r2" / "train.log")

    r = _cli("--roots", "runs", "--experiments-dir", "experiments", "--score-regex", _REGEX,
             "--log-glob", "*.log", "--verify-manifests", cwd=str(tree), timeout=15)
    assert r.returncode == 2, r.stdout + r.stderr
    assert _refusals(r) == 2, r.stderr
    assert "2 additional score dir(s) found, 2 examined" in r.stdout, r.stdout
    assert "score read INCOMPLETE for runs/r1:" in r.stderr, r.stderr
    assert "score read INCOMPLETE for runs/r2:" in r.stderr, r.stderr


def test_shared_log_also_seen_by_a_co_located_score_dir_is_reported_once(tmp_path):
    """The dedup holds ACROSS passes, not merely within the log-glob walk. `runs/r1` carries a
    valid `scores.jsonl`, so the co-located pass reaches it FIRST and — because
    `_load_scores_strict` globs `os.path.join(run_dir, log_glob)` — already reports the shared
    `../shared.log` FIFO. `runs/r2` resolves the identical file; without seeding the claim set
    from `already_found` it would report it a second time."""
    tree = _n_run_tree(tmp_path, 2)
    (tree / "runs" / "r1" / "scores.jsonl").write_text('{"epoch": 1, "score": 0.5}\n')
    os.mkfifo(tree / "runs" / "shared.log")

    r = _cli("--roots", "runs", "--experiments-dir", "experiments", "--score-regex", _REGEX,
             "--log-glob", "../shared.log", "--verify-manifests", cwd=str(tree), timeout=15)
    assert r.returncode == 2, r.stdout + r.stderr
    assert _refusals(r) == 1, r.stderr
    assert "score read INCOMPLETE for runs/r1:" in r.stderr, r.stderr
    assert "0 additional score dir(s) found, 0 examined" in r.stdout, r.stdout


def test_control_shared_clean_log_across_n_dirs_stays_green(tmp_path):
    """CONTROL: the dedup must not manufacture a refusal, and must not swallow a clean read. The
    same 5-dir `../shared.log` shape with a WELL-FORMED log: exit 0, one candidate examined."""
    tree = _n_run_tree(tmp_path, 5)
    (tree / "runs" / "shared.log").write_text("Epoch 5 reward 0.75\n")

    r = _cli("--roots", "runs", "--experiments-dir", "experiments", "--score-regex", _REGEX,
             "--log-glob", "../shared.log", "--verify-manifests", cwd=str(tree), timeout=15)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "verify-manifests: OK" in r.stdout
    assert "1 additional score dir(s) found, 1 examined" in r.stdout, r.stdout


# ============================================================================================
# gate (i), ESCALATED CYCLE — canon's SECOND glob term: the BARE, CWD-relative one
#
# `_load_scores_strict` resolves the `--score-regex` source from TWO globs per run dir:
#     _glob.glob(os.path.join(run_dir, log_glob)) + _glob.glob(log_glob)
# The first pass of this row mirrored only the first term, so a score source reachable ONLY
# through the second — bare, relative to CWD, independent of `--roots` — was still never
# enumerated, `_load_scores_strict` was still never called on it, and `--verify-manifests` still
# returned a false OK on it. D50 one layer further down. These tests pin the second term.
# ============================================================================================

def _cwd_only_tree(tmp_path):
    """A tree where the ONLY thing a bare `--log-glob topfile.log` can match is at CWD: the run
    dirs are real checkpoint dirs, and none of them contains a file of that name."""
    (tmp_path / "runs" / "r1" / "snapshot_ep1").mkdir(parents=True)
    (tmp_path / "experiments").mkdir()
    return tmp_path


def test_bare_cwd_log_glob_fifo_is_examined_not_falsely_ok(tmp_path):
    """THE escalated-cycle reproduction: a real checkpoint dir under `--roots`, and a FIFO
    `./topfile.log` at CWD that the bare glob term — and ONLY the bare glob term — resolves.

    Before this fix: "0 additional score dir(s) found, 0 examined" / "verify-manifests: OK" /
    exit 0, while the identical flags on the real (non-`--verify-manifests`) path hang forever on
    `open()`. The `--help` promise ("REFUSE on anything malformed, symlinked, torn, or dangling")
    was not kept for this input class."""
    tree = _cwd_only_tree(tmp_path)
    os.mkfifo(tree / "topfile.log")

    r = _cli("--roots", "runs", "--experiments-dir", "experiments", "--score-regex", _REGEX,
             "--log-glob", "topfile.log", "--verify-manifests", cwd=str(tree), timeout=15)
    assert r.returncode == 2, r.stdout + r.stderr
    assert "BLOCKS FOREVER" in r.stderr
    assert "verify-manifests: REFUSED" in r.stdout
    # Named as the SOURCE it is, not as the /dev/null probe dir the reader was isolated with.
    assert "the bare `--log-glob topfile.log` term" in r.stderr, r.stderr
    # Once per invocation, never once per directory (3 dirs are walked here).
    assert "3 dir(s) walked" in r.stdout, r.stdout
    assert _refusals(r) == 1, r.stderr
    assert "1 file(s) matched, 1 read(s)" in r.stdout, r.stdout


def test_bare_cwd_log_glob_dangling_symlink_refuses(tmp_path):
    """Shape coverage for the bare term, not just the FIFO: a dangling `./topfile.log` symlink
    must refuse with the dangling diagnosis rather than resolve to "nothing is scored"."""
    tree = _cwd_only_tree(tmp_path)
    (tree / "topfile.log").symlink_to(tree / "no_such_target")

    r = _cli("--roots", "runs", "--experiments-dir", "experiments", "--score-regex", _REGEX,
             "--log-glob", "topfile.log", "--verify-manifests", cwd=str(tree), timeout=15)
    assert r.returncode == 2, r.stdout + r.stderr
    assert "target does not resolve" in r.stderr
    assert _refusals(r) == 1, r.stderr


def test_bare_cwd_log_glob_symlink_to_directory_refuses(tmp_path):
    """A LIVE symlink -> DIRECTORY resolved by the bare term: refused as a directory."""
    tree = _cwd_only_tree(tmp_path)
    (tree / "sneaky").mkdir()
    (tree / "topfile.log").symlink_to(tree / "sneaky")

    r = _cli("--roots", "runs", "--experiments-dir", "experiments", "--score-regex", _REGEX,
             "--log-glob", "topfile.log", "--verify-manifests", cwd=str(tree), timeout=15)
    assert r.returncode == 2, r.stdout + r.stderr
    assert "score file is a DIRECTORY" in r.stderr
    assert _refusals(r) == 1, r.stderr


def test_bare_cwd_log_glob_torn_record_refuses(tmp_path):
    """Not only the SHAPE guard: the bare term goes through `_load_scores_strict` itself, so the
    `--score-regex` strictness rule (a matched line whose groups do not coerce is a torn record,
    not a non-match) covers it too. This is what "mirror the real reader" has to mean — a
    hand-rolled shape-only copy would pass every test above and still miss this."""
    tree = _cwd_only_tree(tmp_path)
    (tree / "topfile.log").write_text("Epoch 5 reward 0.75\nEpoch 6 reward TORN\n")

    r = _cli("--roots", "runs", "--experiments-dir", "experiments", "--score-regex", _REGEX,
             "--log-glob", "topfile.log", "--verify-manifests", cwd=str(tree), timeout=15)
    assert r.returncode == 2, r.stdout + r.stderr
    assert "--score-regex matched but its epoch/score groups did not coerce" in r.stderr
    assert "topfile.log:2" in r.stderr
    assert _refusals(r) == 1, r.stderr


def test_control_bare_cwd_log_glob_clean_file_is_read_and_exits_zero(tmp_path):
    """CONTROL: the new probe must not manufacture refusals. A well-formed CWD log is read, once,
    and the run stays green — with the count line proving the source really was examined (a probe
    that silently matched nothing would also print "OK")."""
    tree = _cwd_only_tree(tmp_path)
    (tree / "topfile.log").write_text("Epoch 5 reward 0.75\n")

    r = _cli("--roots", "runs", "--experiments-dir", "experiments", "--score-regex", _REGEX,
             "--log-glob", "topfile.log", "--verify-manifests", cwd=str(tree), timeout=15)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "verify-manifests: OK" in r.stdout
    assert "1 file(s) matched, 1 read(s)" in r.stdout, r.stdout


def test_control_bare_cwd_log_glob_without_score_regex_stays_inert(tmp_path):
    """`--log-glob` alone is inert in canon; the bare-term probe must inherit that exactly, or a
    FIFO at CWD would hang a mode that was never asked to read logs at all."""
    tree = _cwd_only_tree(tmp_path)
    os.mkfifo(tree / "topfile.log")

    r = _cli("--roots", "runs", "--experiments-dir", "experiments",
             "--log-glob", "topfile.log", "--verify-manifests", cwd=str(tree), timeout=15)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "verify-manifests: OK" in r.stdout
    assert "0 file(s) matched, 0 read(s)" in r.stdout, r.stdout


def test_bare_cwd_log_glob_already_reported_by_a_run_dir_is_not_reported_twice(tmp_path):
    """Scope item 3 applied to this row's OWN addition. `_load_scores_strict` globs the bare term
    for EVERY run dir it is called on, so when a co-located `scores.jsonl` exists the same bad CWD
    log is seen by the run-dir pass AND by the new probe. Measured at the 881I ship: 1 refusal;
    with the probe added naively: 2. One real defect, one report — and still exit 2."""
    tree = _cwd_only_tree(tmp_path)
    (tree / "runs" / "r1" / "scores.jsonl").write_text('{"epoch": 1, "score": 0.5}\n')
    os.mkfifo(tree / "topfile.log")

    r = _cli("--roots", "runs", "--experiments-dir", "experiments", "--score-regex", _REGEX,
             "--log-glob", "topfile.log", "--verify-manifests", cwd=str(tree), timeout=15)
    assert r.returncode == 2, r.stdout + r.stderr
    assert "BLOCKS FOREVER" in r.stderr
    assert _refusals(r) == 1, r.stderr
    # The probe still EXAMINED it — suppressing a duplicate line is not skipping the check.
    assert "1 file(s) matched, 1 read(s)" in r.stdout, r.stdout


def test_red_when_the_bare_probe_duplicate_suppression_is_removed_at_its_real_drop_site(tmp_path):
    """MUTATION: report the bare term's errors unconditionally instead of only the fresh ones. One
    FIFO then produces TWO refusals on a tree that also has a co-located `scores.jsonl` — the
    inflation this row is required not to introduce."""
    mutant = _mutant(
        tmp_path, "no_bare_probe_duplicate_suppression",
        "            fresh = [x for x in e.errors if x not in score_errors_seen]\n",
        "            fresh = list(e.errors)\n")

    tree = _cwd_only_tree(tmp_path / "tree")
    (tree / "runs" / "r1" / "scores.jsonl").write_text('{"epoch": 1, "score": 0.5}\n')
    os.mkfifo(tree / "topfile.log")

    m = _cli_script(mutant, "--roots", "runs", "--experiments-dir", "experiments",
                    "--score-regex", _REGEX, "--log-glob", "topfile.log", "--verify-manifests",
                    cwd=str(tree), timeout=15)
    assert m.returncode == 2, m.stdout + m.stderr
    assert _refusals(m) == 2, m.stderr

    r = _cli("--roots", "runs", "--experiments-dir", "experiments", "--score-regex", _REGEX,
             "--log-glob", "topfile.log", "--verify-manifests", cwd=str(tree), timeout=15)
    assert r.returncode == 2, r.stdout + r.stderr
    assert _refusals(r) == 1, r.stderr


def test_suppression_never_downgrades_the_verdict_when_the_run_dir_pass_is_clean(tmp_path):
    """The suppression is keyed on an error string ALREADY PRINTED, never on "a run dir exists":
    a clean co-located `scores.jsonl` reports nothing, so the bare term's own refusal is printed
    in full and the run still exits 2. Without this, the dedup would have re-created exactly the
    dependency on the primary source that scope item 2 removes from the tests."""
    tree = _cwd_only_tree(tmp_path)
    (tree / "runs" / "r1" / "scores.jsonl").write_text('{"epoch": 1, "score": 0.5}\n')
    # matches ONLY at CWD (no `other.log` under any run dir), so the run-dir pass reads it as part
    # of its own bare term... which is the case above. Use a pattern the run-dir pass cannot see
    # at all: an absolute path, which never enters `_find_score_dirs`'s glob under a run dir.
    other = tmp_path / "elsewhere"
    other.mkdir()
    os.mkfifo(other / "abs.log")

    r = _cli("--roots", "runs", "--experiments-dir", "experiments", "--score-regex", _REGEX,
             "--log-glob", str(other / "abs.log"), "--verify-manifests", cwd=str(tree), timeout=15)
    assert r.returncode == 2, r.stdout + r.stderr
    assert "verify-manifests: REFUSED" in r.stdout
    assert "BLOCKS FOREVER" in r.stderr


def test_the_probe_dir_contributes_no_score_source_of_its_own(tmp_path):
    """The bare term is isolated by handing `_load_scores_strict` a `run_dir` that provably holds
    no co-located score file (`os.devnull`). Pinned in-process, because the whole argument for
    that spelling is that the two co-located branches CANNOT contribute from it: if they ever
    could, this row would have invented a brand-new false-refusal source outside `--roots`."""
    assert gc._BARE_LOG_GLOB_PROBE_DIR == os.devnull
    for basename in sorted(gc.SCORE_BASENAMES):
        p = os.path.join(gc._BARE_LOG_GLOB_PROBE_DIR, basename)
        assert not os.path.exists(p), p
        assert gc._score_source_shape_reason(p) is None, p

    cwd = os.getcwd()
    tree = _cwd_only_tree(tmp_path)
    (tree / "topfile.log").write_text("Epoch 5 reward 0.75\n")
    os.chdir(str(tree))
    try:
        assert gc._load_bare_log_glob_scores_strict(_REGEX, "topfile.log") == {5: 0.75}
        assert gc._bare_log_glob_matches(_REGEX, "topfile.log") == ["topfile.log"]
        assert gc._bare_log_glob_matches(None, "topfile.log") == []
    finally:
        os.chdir(cwd)


def test_help_names_the_log_glob_score_source_it_now_verifies():
    """`--help` must not promise more than the code keeps, and must not stay silent about a source
    the code now checks: the text names the `--score-regex`/`--log-glob` log AND both places it is
    matched from. This is the text half of the BLOCKER — help and behaviour move together."""
    r = subprocess.run([sys.executable, CKPT_GC, "--help"], cwd=REPO,
                       capture_output=True, text=True, timeout=20)
    assert r.returncode == 0, r.stdout + r.stderr
    help_text = " ".join(r.stdout.split())
    assert "--score-regex/--log-glob log" in help_text, help_text
    assert "relative to CWD" in help_text, help_text


# ============================================================================================
# gate (ii)/(iii) — mutations at the REAL drop sites
# ============================================================================================

def test_red_when_the_log_glob_enumeration_is_removed_at_its_real_drop_site(tmp_path):
    """MUTATION: disable `_find_log_glob_score_dirs`'s call site in `_run_manifest_check`. This is
    D50 exactly: the mutant falsely reports OK on a tree whose only score source is a FIFO log and
    no `scores.jsonl` anywhere — precisely the dependency scope item 2 closes in the test suite,
    reproduced here at the source level instead of by omission in a fixture."""
    mutant = _mutant(
        tmp_path, "no_log_glob_enumeration",
        "    log_glob_extra = _find_log_glob_score_dirs(list(a.roots), a.score_regex, a.log_glob,\n"
        "                                               score_dirs_seen)\n",
        "    log_glob_extra = []\n")

    tree = _base_tree(tmp_path / "tree")
    os.mkfifo(tree / "runs" / "r1" / "train.log")

    m = _cli_script(mutant, "--roots", "runs", "--experiments-dir", "experiments",
                    "--score-regex", _REGEX, "--log-glob", "*.log", "--verify-manifests",
                    cwd=str(tree), timeout=15)
    assert m.returncode == 0, m.stdout + m.stderr
    assert "verify-manifests: OK" in m.stdout

    r = _cli("--roots", "runs", "--experiments-dir", "experiments", "--score-regex", _REGEX,
             "--log-glob", "*.log", "--verify-manifests", cwd=str(tree), timeout=15)
    assert r.returncode == 2, r.stdout + r.stderr


def test_red_when_the_bare_log_glob_probe_is_removed_at_its_real_drop_site(tmp_path):
    """MUTATION: disable the once-per-invocation bare-term probe in `_run_manifest_check`. The
    mutant is the escalated cycle's BLOCKER verbatim — a FIFO at CWD that only canon's second glob
    term resolves is never handed to `_load_scores_strict`, and the tool reports OK at exit 0."""
    mutant = _mutant(
        tmp_path, "no_bare_log_glob_probe",
        "    bare_matches = _bare_log_glob_matches(a.score_regex, a.log_glob)\n",
        "    bare_matches = []\n")

    tree = _cwd_only_tree(tmp_path / "tree")
    os.mkfifo(tree / "topfile.log")

    m = _cli_script(mutant, "--roots", "runs", "--experiments-dir", "experiments",
                    "--score-regex", _REGEX, "--log-glob", "topfile.log", "--verify-manifests",
                    cwd=str(tree), timeout=15)
    assert m.returncode == 0, m.stdout + m.stderr
    assert "verify-manifests: OK" in m.stdout

    r = _cli("--roots", "runs", "--experiments-dir", "experiments", "--score-regex", _REGEX,
             "--log-glob", "topfile.log", "--verify-manifests", cwd=str(tree), timeout=15)
    assert r.returncode == 2, r.stdout + r.stderr


def test_red_when_the_resolved_path_collision_dedup_is_removed_at_its_real_drop_site(tmp_path):
    """MUTATION (MAJOR-B's real drop site): enumerate a directory whenever it has ANY match,
    instead of only when it contributes a resolved path no earlier candidate covers. The mutant is
    MAJOR-B verbatim — one shared FIFO reached through `../shared.log` from five sibling run dirs
    becomes FIVE refusals and "5 additional score dir(s) found", while still exiting 2 (which is
    exactly why an exit-code-and-substring test could not see it)."""
    mutant = _mutant(
        tmp_path, "no_resolved_path_collision_dedup",
        "            if resolved - claimed:\n",
        "            if resolved:  # MUTANT: no resolved-path collision dedup\n")

    tree = _n_run_tree(tmp_path / "tree", 5)
    os.mkfifo(tree / "runs" / "shared.log")

    m = _cli_script(mutant, "--roots", "runs", "--experiments-dir", "experiments",
                    "--score-regex", _REGEX, "--log-glob", "../shared.log", "--verify-manifests",
                    cwd=str(tree), timeout=15)
    assert m.returncode == 2, m.stdout + m.stderr
    assert _refusals(m) == 5, m.stderr
    assert "5 additional score dir(s) found, 5 examined" in m.stdout, m.stdout

    r = _cli("--roots", "runs", "--experiments-dir", "experiments", "--score-regex", _REGEX,
             "--log-glob", "../shared.log", "--verify-manifests", cwd=str(tree), timeout=15)
    assert r.returncode == 2, r.stdout + r.stderr
    assert _refusals(r) == 1, r.stderr
    assert "1 additional score dir(s) found, 1 examined" in r.stdout, r.stdout


def test_red_when_the_bare_term_claim_seed_is_removed_at_its_real_drop_site(tmp_path):
    """MUTATION: start the claim set EMPTY instead of seeding it with what the once-per-invocation
    bare-term probe already covers. The absolute pattern then stops being routed to that probe and
    is re-enumerated as a per-run "additional score dir": the refusal loses its source-naming
    wording and doubles its RECORD count (the reader's two glob terms resolve to the same file),
    which is the 881I-era inflation this row must not reintroduce."""
    mutant = _mutant(
        tmp_path, "no_bare_term_claim_seed",
        "    claimed: set[str] = {os.path.realpath(m)\n"
        "                         for m in _bare_log_glob_matches(score_regex, log_glob)}\n",
        "    claimed: set[str] = set()  # MUTANT: bare-term matches not pre-claimed\n")

    tree = _wide_tree(tmp_path / "tree")
    fifo = tree / "a_fifo_log"
    os.mkfifo(fifo)

    m = _cli_script(mutant, "--roots", "runs", "--experiments-dir", "experiments",
                    "--score-regex", _REGEX, "--log-glob", str(fifo), "--verify-manifests",
                    cwd=str(tree), timeout=15)
    assert m.returncode == 2, m.stdout + m.stderr
    assert "1 additional score dir(s) found, 1 examined" in m.stdout, m.stdout
    assert "2 unreadable record(s)" in m.stderr, m.stderr
    assert "the bare `--log-glob" not in m.stderr, m.stderr

    r = _cli("--roots", "runs", "--experiments-dir", "experiments", "--score-regex", _REGEX,
             "--log-glob", str(fifo), "--verify-manifests", cwd=str(tree), timeout=15)
    assert r.returncode == 2, r.stdout + r.stderr
    assert "0 additional score dir(s) found, 0 examined" in r.stdout, r.stdout
    assert "1 unreadable record(s)" in r.stderr, r.stderr
    assert "the bare `--log-glob" in r.stderr, r.stderr


def test_red_when_the_already_found_claim_seed_is_removed_at_its_real_drop_site(tmp_path):
    """MUTATION: stop pre-claiming the matches of directories the CO-LOCATED pass already read.
    The shared `../shared.log` FIFO is then reported once by `runs/r1` (which has a `scores.jsonl`,
    so the co-located pass reaches it) and a second time by `runs/r2` from the log-glob walk — one
    physical file, two reports, across passes instead of within one."""
    mutant = _mutant(
        tmp_path, "no_already_found_claim_seed",
        "    for sd in sorted(already_found):\n        claimed |= _resolved(sd)\n",
        "    # MUTANT: already-read dirs contribute no claims\n")

    tree = _n_run_tree(tmp_path / "tree", 2)
    (tree / "runs" / "r1" / "scores.jsonl").write_text('{"epoch": 1, "score": 0.5}\n')
    os.mkfifo(tree / "runs" / "shared.log")

    m = _cli_script(mutant, "--roots", "runs", "--experiments-dir", "experiments",
                    "--score-regex", _REGEX, "--log-glob", "../shared.log", "--verify-manifests",
                    cwd=str(tree), timeout=15)
    assert m.returncode == 2, m.stdout + m.stderr
    assert _refusals(m) == 2, m.stderr

    r = _cli("--roots", "runs", "--experiments-dir", "experiments", "--score-regex", _REGEX,
             "--log-glob", "../shared.log", "--verify-manifests", cwd=str(tree), timeout=15)
    assert r.returncode == 2, r.stdout + r.stderr
    assert _refusals(r) == 1, r.stderr


# ============================================================================================
# MAJOR-A — PRE-EXISTING, DOCUMENTED HERE, DELIBERATELY NOT FIXED
# ============================================================================================

def test_known_limitation_shared_absolute_log_is_reported_once_per_co_located_score_dir(tmp_path):
    """KNOWN LIMITATION, PRE-EXISTING — this test PINS today's behaviour, it does not endorse it.

    Shape: N run dirs that EACH carry a valid `scores.jsonl` (so `_find_score_dirs` enumerates all
    N through the SCORE_BASENAMES gate), plus ONE shared absolute `--log-glob` FIFO. The co-located
    loop in `_run_manifest_check` hands each of the N dirs to `_load_scores_strict`, and that
    reader globs `os.path.join(run_dir, log_glob)` (which discards `run_dir` for an absolute
    pattern) PLUS the bare `log_glob` — so ONE physical FIFO produces N refusal lines, each
    internally doubled to "2 unreadable record(s)".

    Why it is not closed here (TASK_881S cycle 3, reviewer triage MAJOR-A):
      * NOT a regression from this row — it reproduces identically at `bf1f8d5`, the 881I-shipped
        tree, with the same N lines and the same doubled record count (measured with N=3);
      * closing it requires editing pre-existing lines inside `_load_scores_strict` and/or the
        co-located loop, which TASK_881S gate (iv) — `tools/ckpt_gc.py` must remain a PURE
        INSERTION vs `bf1f8d5`, zero removed or rewritten lines — forbids.
    This row's own dedup (`_find_log_glob_score_dirs`'s resolved-path collision check) is deliberate
    ABOUT ITS OWN PASS: it stops the log-glob walk from ADDING to this count, and the assertion on
    "0 additional score dir(s) found" below is what proves that separation. The next row that is
    allowed to touch the reader inherits this precise reproduction instead of a memory of one.
    """
    n = 3
    tree = _n_run_tree(tmp_path, n)
    for i in range(1, n + 1):
        (tree / "runs" / f"r{i}" / "scores.jsonl").write_text('{"epoch": 1, "score": 0.5}\n')
    fifo = tree / "shared_abs.log"
    os.mkfifo(fifo)

    r = _cli("--roots", "runs", "--experiments-dir", "experiments", "--score-regex", _REGEX,
             "--log-glob", str(fifo), "--verify-manifests", cwd=str(tree), timeout=15)
    assert r.returncode == 2, r.stdout + r.stderr
    # TODAY'S (pre-existing, unwanted) COUNTS — N lines for one file, each doubled internally.
    assert _refusals(r) == n, r.stderr
    assert r.stderr.count("2 unreadable record(s)") == n, r.stderr
    for i in range(1, n + 1):
        assert f"score read INCOMPLETE for runs/r{i}:" in r.stderr, r.stderr
    # ...and the separation that IS this row's: the log-glob pass adds nothing on top of them.
    assert "0 additional score dir(s) found, 0 examined" in r.stdout, r.stdout


def test_known_limitation_major_a_reproduces_identically_at_the_881i_shipped_tree(tmp_path):
    """The "pre-existing, not a regression" half of the claim above, MEASURED rather than asserted:
    the exact same tree is run against `git show bf1f8d5:tools/ckpt_gc.py` — the 881I-shipped
    source this row must remain a pure insertion on top of — and produces the identical counts.

    If a future row closes MAJOR-A, this test and the one above are the pair to update together:
    this one records the starting point, that one records the current output."""
    src = subprocess.run(["git", "show", "bf1f8d5:tools/ckpt_gc.py"], cwd=REPO,
                         capture_output=True, text=True)
    assert src.returncode == 0, src.stderr
    shipped = tmp_path / "_baseline_881i_ckpt_gc.py"
    shipped.write_text(src.stdout)

    n = 3
    tree = _n_run_tree(tmp_path / "tree", n)
    for i in range(1, n + 1):
        (tree / "runs" / f"r{i}" / "scores.jsonl").write_text('{"epoch": 1, "score": 0.5}\n')
    fifo = tree / "shared_abs.log"
    os.mkfifo(fifo)

    b = _cli_script(str(shipped), "--roots", "runs", "--experiments-dir", "experiments",
                    "--score-regex", _REGEX, "--log-glob", str(fifo), "--verify-manifests",
                    cwd=str(tree), timeout=15)
    assert b.returncode == 2, b.stdout + b.stderr
    assert _refusals(b) == n, b.stderr
    assert b.stderr.count("2 unreadable record(s)") == n, b.stderr


def test_red_when_the_bare_probe_record_dedup_is_removed_at_its_real_drop_site(tmp_path):
    """MUTATION: stop collapsing duplicate error strings in the bare-term refusal. For an ABSOLUTE
    `--log-glob` the reader's two glob terms resolve to the identical path, so it appends the
    identical message twice and one FIFO is reported as "2 unreadable record(s)" — the same
    inflation as the MAJOR, one level down in the record count instead of the line count."""
    mutant = _mutant(
        tmp_path, "no_bare_probe_record_dedup",
        "            list(dict.fromkeys(e.errors))) from None",
        "            list(e.errors)) from None")

    tree = _wide_tree(tmp_path / "tree")
    fifo = tree / "a_fifo_log"
    os.mkfifo(fifo)

    m = _cli_script(mutant, "--roots", "runs", "--experiments-dir", "experiments",
                    "--score-regex", _REGEX, "--log-glob", str(fifo), "--verify-manifests",
                    cwd=str(tree), timeout=15)
    assert m.returncode == 2, m.stdout + m.stderr
    assert "2 unreadable record(s)" in m.stderr, m.stderr

    r = _cli("--roots", "runs", "--experiments-dir", "experiments", "--score-regex", _REGEX,
             "--log-glob", str(fifo), "--verify-manifests", cwd=str(tree), timeout=15)
    assert r.returncode == 2, r.stdout + r.stderr
    assert "1 unreadable record(s)" in r.stderr, r.stderr


def test_red_when_the_dangling_manifest_dedup_guard_is_removed_at_its_real_drop_site(tmp_path):
    """MUTATION: put `_walk_error` back to reporting every `DanglingScanLink` unconditionally. A
    dangling manifest-basename symlink is then counted TWICE — the original scope-item-3 defect."""
    mutant = _mutant(
        tmp_path, "no_dangling_manifest_dedup",
        '        if (isinstance(exc, DanglingScanLink)\n'
        '                and os.path.basename(getattr(exc, "filename", "")) in MANIFEST_BASENAMES):\n'
        '            return\n',
        "")

    tree = tmp_path / "tree"
    (tree / "experiments" / "999_x").mkdir(parents=True)
    (tree / "runs").mkdir()
    (tree / "experiments" / "999_x" / "PROTECTED.txt").symlink_to(tree / "no_such_target")

    m = _cli_script(mutant, "--roots", "runs", "--experiments-dir", "experiments",
                    "--verify-manifests", cwd=str(tree), timeout=15)
    assert m.returncode == 2, m.stdout + m.stderr
    assert "2 problem(s)" in m.stderr, m.stderr

    r = _cli("--roots", "runs", "--experiments-dir", "experiments", "--verify-manifests",
             cwd=str(tree), timeout=15)
    assert r.returncode == 2, r.stdout + r.stderr
    assert "1 problem(s)" in r.stderr, r.stderr


def test_dangling_manifest_symlink_is_reported_exactly_once(tmp_path):
    """Scope item 3, stated directly: one real defect (a dangling `PROTECTED.txt` symlink) must
    produce exactly one line of diagnosis, not two — a count an alert might key on must not be
    inflated by a defect that two independent detectors both happen to see."""
    tree = _base_tree(tmp_path)
    (tree / "experiments" / "999_x").mkdir(parents=True)
    (tree / "experiments" / "999_x" / "PROTECTED.txt").symlink_to(tree / "no_such_target")

    r = _cli("--roots", "runs", "--experiments-dir", "experiments", "--verify-manifests",
             cwd=str(tree), timeout=15)
    assert r.returncode == 2, r.stdout + r.stderr
    assert "1 problem(s)" in r.stderr, r.stderr
    # The walk-side generic diagnosis ("BROKEN SYMLINK ... an unmounted volume is the realistic
    # cause") is the one being deduped away; only `manifest_shape_refusal`'s more specific
    # "target does not resolve" wording should remain. (`_run_manifest_check` itself prints each
    # `errors` entry TWICE by construction — once inlined in the exception summary, once in the
    # per-line dump — so a raw substring COUNT over stderr is not the right check for "how many
    # DISTINCT problems were found"; the in-process `errors` list is.)
    assert "an unmounted volume is the realistic cause" not in r.stderr, r.stderr

    cwd = os.getcwd()
    os.chdir(str(tree))
    try:
        with pytest.raises(gc.ManifestScanIncomplete) as exc_info:
            gc.load_manifests(["experiments"])
        assert len(exc_info.value.errors) == 1, exc_info.value.errors
    finally:
        os.chdir(cwd)


def test_control_dedup_does_not_hide_an_unrelated_dangling_link(tmp_path):
    """The dedup is scoped to MANIFEST_BASENAMES names ONLY: a dangling symlink elsewhere in the
    scan tree (not named `PROTECTED.txt`/`SPEC.md`) has no second detector and must still be
    reported — proving the fix does not overreach into suppressing a real, singly-detected defect."""
    tree = _base_tree(tmp_path)
    (tree / "experiments" / "999_x").mkdir(parents=True)
    (tree / "experiments" / "999_x" / "some_other_link").symlink_to(tree / "no_such_target")

    r = _cli("--roots", "runs", "--experiments-dir", "experiments", "--verify-manifests",
             cwd=str(tree), timeout=15)
    assert r.returncode == 2, r.stdout + r.stderr
    assert "some_other_link" in r.stderr
    assert "BROKEN SYMLINK" in r.stderr


# ============================================================================================
# gate (iv) — byte-identity of the untouched enumeration, `_find_score_dirs`
# ============================================================================================

def _head_source():
    r = subprocess.run(["git", "show", "HEAD:tools/ckpt_gc.py"], cwd=REPO,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout


def _func_source(src, name):
    lines = src.splitlines(keepends=True)
    start = next(i for i, l in enumerate(lines) if l.startswith(f"def {name}("))
    end = start + 1
    while end < len(lines) and (lines[end].startswith((" ", "\t", "\n"))):
        end += 1
    return "".join(lines[start:end]).rstrip() + "\n"


def test_find_score_dirs_is_byte_identical_to_the_881i_shipped_tree():
    """The enumeration change is `_find_log_glob_score_dirs` (new) plus its call sites in
    `_run_manifest_check` — `_find_score_dirs` itself, the function the D50 defect lives in, is
    NOT modified at all (TASK_881S gate (iv)). A run scored via the log-glob source reaches
    `_load_scores_strict` through a SEPARATE enumeration, never by changing this one's gate."""
    head, now = _head_source(), open(CKPT_GC).read()
    assert _func_source(now, "_find_score_dirs") == _func_source(head, "_find_score_dirs")
