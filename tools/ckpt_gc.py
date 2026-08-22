#!/usr/bin/env python3
"""Score-based checkpoint GC + disk-floor guard (operator requirement, 2026-08-05).

WHY THIS EXISTS AND WHY IT IS NOT A TRAINING-LOOP FEATURE
--------------------------------------------------------
Long runs write ~89 MB per snapshot and `snapshot_ep{N}` dirs are never overwritten, while the
main save path prunes by RECENCY (flax ``keep=3``) — which after a late-run collapse keeps the
broken checkpoints and evicts the good ones. If a training host fills its filesystem, the
casualty is not just the run: ``trading_data.db`` lives on the same filesystem, and re-shipping
it is impossible on a ~5 Mbps uplink. So pruning has to exist as an EXTERNAL, always-on
mechanism that works on runs already in flight and on runs launched by anything (factory,
planner, hand) — independent of whether a given trainer was taught to prune itself.

POLICY
------
Per run directory: keep the TOP-K by score, plus ALWAYS the newest checkpoint (resume) and the
best-scored one. Scores come from, in order: ``checkpoints_index.json``, ``scores.jsonl``
(``{"epoch": int, "score": float}`` per line), or a run log matched by ``--score-regex``.
Checkpoints with NO score are never pruned on score — they are only pruned by recency when free
space is under the floor, and every such deletion says so in the log. Missing data never causes
a deletion decision (a checkpoint you cannot evaluate is a checkpoint you keep).

SAFETY (all enforced, not documented-and-hoped)
-----------------------------------------------
* dry-run by default; ``--apply`` required to delete anything
* only paths UNDER a configured root, whose basename matches a checkpoint pattern, are eligible
* never deletes: newest per run · best-scored per run · anything listed in ``--protect-file`` ·
  any path containing a ``.db``/``.git``/``trading_data`` component · symlinks (never followed)
* deletion is move-to-``.gc-trash-<pid>`` then rmtree, so a partial delete cannot leave a
  half-checkpoint that looks loadable
* every action is logged with score + reason + bytes

MANIFEST GRAMMAR (TASK_881G, D53) — what a ``PROTECTED.txt`` and a SPEC.md ``protected:``/
``covers:`` block ACCEPT and REFUSE, written down because before this row the accepted grammar
existed nowhere an author would look (0 of 56 real manifest files under this repo ever exercised
it). If a shape is not written here, it is not accepted. The mirrored copy of this section lives
in ``tasks/TASK_881D_PROTECTION_DELETER.md``'s corpus precondition; the implementation this doc
describes is ``parse_protected_txt`` / ``parse_spec_manifest`` / ``load_manifests`` /
``load_protect_file``, below.

MANIFEST BASENAMES: exactly ``PROTECTED.txt`` and ``SPEC.md`` (``MANIFEST_BASENAMES``). Either
name, anywhere under a scanned directory, is read as a manifest. Both share the SAME per-line
ITEM grammar (below, "one path, one line, `$`-anchored"); PROTECTED.txt additionally supports a
plain ``#`` comment line and SPEC.md does not — SPEC.md is Markdown, where a leading ``#`` is
heading TEXT, and that divergence is deliberate, not a bug (see the SPEC.md bullets below).

WHOLE-FILE PRECONDITIONS, identical for both basenames, checked before either parser ever runs:

* the basename must be a real, regular, non-symlink file at a real, tracked path. A symlink (to
  a file OR to a directory), a real directory, a fifo/socket/device, or a manifest reached only
  by crossing a symlinked ANCESTOR directory (including a symlinked scan root) is REFUSED, never
  read — ten distinct messages by shape, see ``manifest_shape_refusal``.
* the file must decode as UTF-8 with no byte-order mark and no NUL byte. A BOM (UTF-8/16/32,
  either endianness), a raw NUL byte, or an invalid UTF-8 sequence is REFUSED, never parsed — see
  ``_read_manifest_text``.
* ANY malformed line, anywhere in the file, WITHDRAWS THE WHOLE FILE's directives — not just the
  offending line. A file that is 90% well-formed and 10% malformed protects NOTHING; the
  malformed line is reported (never silently dropped), but nothing in that file is trusted.
* a well-formed entry naming a path that does not exist on disk is NOT malformed — it is
  DANGLING (``ManifestSet.dangling``): it protects nothing and is reported on its own, but it
  does not poison the rest of the file the way a malformed LINE does. "NOT malformed" is a
  statement about INTRA-FILE poisoning ONLY — the file's other entries still parse and still
  grant — and it is NOT an acceptance: a dangling entry makes ``--verify-manifests`` print
  ``DANGLING ENTRY`` and EXIT 2, never 0, exactly like a refusal (see ``_run_manifest_check``).
  Both statements are true at once, and reading only the first one is how an author concludes a
  typo'd path will pass the check.
* ENTRY PATHS ARE RESOLVED AGAINST THE PROCESS'S CWD, NOT against the manifest's own directory.
  Every entry — in either basename, in ``--protect-file``, relative or absolute — goes through
  ``os.path.realpath(<the text as written>)``, which is relative to wherever the tool was
  INVOKED from. So ``- runs/a`` inside ``experiments/999_z/SPEC.md`` names ``$CWD/runs/a``, NOT
  ``experiments/999_z/runs/a``; from another CWD the same manifest names something else, and
  usually something DANGLING. Manifests are authored for, and this tool is run from, the repo
  root.

PROTECTED.txt GRAMMAR — one directive per line:

* ``protect: <path>`` or ``covers: <path>`` — exactly one directive keyword, a colon, and a
  single-token path, optional surrounding whitespace, nothing else on the line. ACCEPTED::

      protect: runs/some_run/snapshot_ep10
      covers: runs/other_run

* WHITESPACE AROUND THE DIRECTIVE IS FREE, and this is the same micro-grammar the SPEC.md item
  form uses (below): the line is STRIPPED before it is matched, so leading indentation
  (``  protect: runs/a``) and trailing whitespace are ACCEPTED, and the space after the colon is
  optional (``protect:runs/a`` is ACCEPTED). What is NOT free is the number of tokens after the
  colon — exactly one.
* a blank line — ACCEPTED, produces no entry. A line of nothing but SPACES or TABS counts as
  blank here exactly as an empty one does (the test is on the STRIPPED line).
* a line whose STRIPPED form starts with ``#`` (any leading indentation before the ``#`` is fine
  — the test is on the stripped line, not the raw one) — ACCEPTED as a comment, produces no
  entry.
* anything else is MALFORMED (whole file withdrawn) — including: ``protect:``/``covers:``
  followed by TWO tokens (``protect: a b``, the unanchored subset-grant shape D53 closed on both
  formats); a bare ``protect:`` with no path; an unrecognized directive word; or plain prose.

SPEC.md BLOCK GRAMMAR — a ``protected:`` or ``covers:`` block:

* the header is column-0, lower-case, exact: ``protected:`` or ``covers:``, optionally followed
  on the SAME line by nothing but whitespace up to end-of-line or end-of-file (a torn/truncated
  final line with no trailing newline is still recognized as a header).
* immediately after the header — with EITHER zero blank lines OR one-or-more blank lines before
  the first item — a list of items, one path per line, each written ``- <path>`` (a single
  ``-``, optional whitespace, exactly one non-whitespace token, nothing else on the line). BOTH
  of the following are ACCEPTED and equivalent (the second is the canonical MD032 form — a list
  must be surrounded by blank lines — and is the FIRST shape an author who runs a markdown
  linter will type; accepting it is what D53 required)::

      protected:
      - runs/some_run/snapshot_ep10
      - runs/other_run

      protected:

      - runs/some_run/snapshot_ep10
      - runs/other_run

* THE ITEM MICRO-GRAMMAR IS DELIBERATELY FORGIVING, in deliberate ASYMMETRY with the strictly
  column-0, lower-case, exact HEADER above it. An item line is matched STRIPPED-EQUIVALENTLY, so
  all three of these are ACCEPTED and name exactly the path they appear to:

  - LEADING INDENTATION before the dash (``  - runs/a``) — Markdown nests lists, and an author
    writing a nested list is not making the mistake an indented HEADER makes (an indented header
    is a NEAR-MISS refusal, see the last bullet of this section);
  - NO whitespace between the dash and the path (``-runs/a``);
  - TRAILING whitespace after the path (``- runs/a   ``).

  The asymmetry is on purpose: an indented/odd HEADER can silently withdraw a whole declared
  block, so it refuses; an odd but unambiguous ITEM inside an already-matched block cannot go
  anywhere but into that block, so it is read. What an item may NEVER do is name a second token
  (see the INTERRUPTED bullet below) — that is the one thing whitespace cannot buy.
* A LINE OF NOTHING BUT SPACES OR TABS IS A BLANK LINE, uniformly, in ALL THREE places where
  this section says "blank": the MD032 gap between a header and its first item, the SPLIT gap
  after at least one item, and the blank line that legitimately ENDS a list. There is no shape
  where a whitespace-only line behaves differently from an empty one.
* once at least one item has been consumed, a blank line followed by more PROSE, a heading, or
  end-of-file legitimately ENDS the block (the every-day case: block, blank line, prose resumes).
  ACCEPTED.
* once at least one item has been consumed, a blank line followed by ANOTHER ``- <path>`` item —
  the list resuming after a gap — is REFUSED (a hole inside a list is indistinguishable from a
  torn write that lost the items in between). This is the only place a leading-vs-mid-list blank
  line matters: BEFORE the first item, a gap is the MD032 fence above; AFTER at least one item, a
  gap is a SPLIT.
* a header carrying inline content (``protected: runs/a``) — REFUSED, whole file withdrawn.
* a header with ZERO items (nothing recognizable as an item follows before the block ends) —
  REFUSED as EMPTY, whole file withdrawn — indistinguishable from a truncated write that lost its
  items.
* WHICH REFUSAL WINS WHEN BOTH COULD APPLY (EMPTY vs INTERRUPTED): the INTERRUPTED diagnosis
  requires at least ONE already-consumed item to have been interrupted. So a non-item, non-blank
  line that is the FIRST content after the header — ``protected:`` / ``hello`` / ``- runs/a`` —
  refuses as EMPTY ("it names nothing"), not as INTERRUPTED, even though a later well-formed item
  follows it. Both are exit-2 whole-file withdrawals with the same effect; only the message
  differs. The ONE exception is an attempted item naming more than one token (``- a b``), which
  refuses wherever it sits, first line or not, under its OWN distinct message ("list item at line
  N carries MORE THAN ONE token") — because that line unambiguously IS an attempted item and
  "names nothing" would be a false thing to say about it.
* TWO OR MORE headers for the SAME key in one file (two ``protected:`` blocks, or two
  ``covers:`` blocks) — REFUSED (duplicate), whole file withdrawn. ``protected:`` and ``covers:``
  are independent keys — one of each in the same file is fine.
* an item line naming MORE THAN ONE token (``- runs/a runs/b``, or ``- runs/a # note`` — a
  TRAILING COMMENT on an item is this same shape) — REFUSED (the INTERRUPTED-list class, under
  its own distinct message: "list item at line N carries MORE THAN ONE token"), whole file
  withdrawn. Put each path on its own line; trailing comments on an item are not supported.
* any OTHER non-blank, non-item line inside a started list — a stray word, a ``*`` bullet, a bare
  ``-`` with no path, or a ``#`` line — is NOT a comment here (unlike PROTECTED.txt: a leading
  ``#`` is Markdown heading TEXT, and "supporting" it would mean a strip erasing real document
  content) — REFUSED as INTERRUPTED, whole file withdrawn. Put comments and prose after a blank
  line that ends the list.
* a line that is ENTIRELY two-or-more ``-`` characters (a markdown THEMATIC BREAK / horizontal
  rule, e.g. ``---``) is NEVER treated as a list item, even though its first character reads as
  a bullet marker — it falls through to whichever of the two bullets above applies (INTERRUPTED
  if items were already consumed with no gap before it, EMPTY if nothing was consumed yet).
  RESIDUAL, NAMED HERE, NOT CLOSED: a thematic break on the far side of a blank line AFTER at
  least one item has already been consumed still reads as "resumes the list" and refuses as
  SPLIT rather than as a clean terminator — the DIRECTION (refuse; never silently ingest a bogus
  ``--`` entry) is unchanged, only the wording of that one sub-case is imprecise.
* a header that misses column-0/lower-case/exact spelling by ANYTHING — indentation, a
  capitalized ``PROTECTED:``, a leading ``>``/``*``/``-``, or a byte-order mark — is a NEAR-MISS:
  REFUSED (never a silent skip), whole file withdrawn, EVEN when the same file also carries a
  well-formed block elsewhere, and EVEN when the near-miss text sits inside a fenced code block.
  Both of those are ACCEPTED-DELIBERATELY as consequences of the SAME rule, not oversights: this
  parser does not try to detect Markdown fences at all (a WELL-FORMED header inside a fence is
  ALSO read as a real block — see ``parse_spec_manifest``'s own docstring), and a
  malformed/near-miss line anywhere in a file poisons that whole file's trust regardless of what
  else the file contains.
* FENCES ARE NOT INVISIBLE, THEY ARE UNSEEN — and the difference bites in the most natural shape
  an author writes. Because no fence is detected, the CLOSING BACKTICK-FENCE LINE of a fenced
  EXAMPLE is just another line, and if it comes IMMEDIATELY after the last item it is a non-item
  line inside a started list: the whole file is REFUSED as INTERRUPTED. So "a header inside a
  fence can only ADD protection" is FALSE as written — it can also WITHDRAW the entire file.
  REFUSED::

      ```
      protected:
      - runs/a
      ```

  ACCEPTED (one blank line TERMINATES the list before the closing fence; the block inside the
  example parses as a real block and its entries are real)::

      ```
      protected:
      - runs/a

      ```

  The practical rule for a document that must both illustrate and be ingested: leave a blank line
  after the last item of any example block, and expect every path you write in an example to be
  ingested as a real entry (a nonexistent one lands in DANGLING and exits 2).

``--protect-file`` GRAMMAR — AS READ BY ``--verify-manifests`` (``load_protect_file``), which is
the ONLY reader this section describes; the legacy delete path reads the same file differently and
is stated at the end of this paragraph. A flat list — NOT the PROTECTED.txt directive grammar; no
``protect:``/``covers:`` prefix: one path per line, realpath'd. A blank (stripped) line, or a
line whose STRIPPED form starts with ``#`` (leading indentation before the ``#`` does not exempt
it from being a comment — an 881G fix; the shipped behaviour tested only the RAW line's first
character, so an indented ``  # comment`` line was realpath'd and ingested as a bogus protected
path), is skipped. Anything else is one path, trusted verbatim (no malformed-line concept here —
this file has no directive syntax to get wrong). A missing, symlinked, or non-regular-file
(directory, fifo/socket/device) ``--protect-file``, WHEN THE FLAG IS GIVEN, is REFUSED — true
since before this row and unchanged by it.

NOT TRUE OF THE DELETE PATH (dry-run / ``--apply``), stated because this section is 881D's corpus
precondition and a doc that over-promised here would become a false spec: ``main()``'s deletion
loop does NOT call ``load_protect_file``. It reads the file inline, guarded by ``os.path.exists``:
every non-blank line is realpath'd, ``#`` is NOT a comment there, and the four shapes
``--verify-manifests`` refuses do NOT all fail the same way here. A MISSING ``--protect-file``, or
one that is a DANGLING SYMLINK (``os.path.exists`` follows a symlink and reports False once its
target is gone), is silently treated as EMPTY at exit 0 rather than refused — the
named-but-unusable input this row's whole discipline exists to refuse. A DIRECTORY is not caught
at all: ``open()`` raises ``IsADirectoryError``, uncaught, so the process CRASHES with a traceback
at exit 1 — a crash, not "empty at exit 0". A LIVE SYMLINK to a real file is not refused either:
``os.path.exists`` and ``open()`` both follow it transparently, so the target IS read and its
entries ARE honoured — an unrefused ACCEPT, the opposite failure mode from the missing/dangling
case. A FIFO ``--protect-file`` is worse than either: ``open()`` on a reader-less FIFO in read mode
BLOCKS FOREVER, the same hang ``load_protect_file``'s own docstring warns about three screens
above. That inline read is canon (bit-identical to HEAD by this row's own gate, so 881G does not
change it); UNIFYING the two readers, i.e. routing the deleter through ``load_protect_file`` so a
missing protect file cannot silently un-protect anything before a deletion, is TASK_881D's, and
until then the safe operating rule is: run ``--verify-manifests`` FIRST and act on its exit code,
because the delete path will not repeat its refusals.

``--experiments-dir`` SEMANTICS:

* the flag OMITTED uses the default ``["experiments"]``; if that default does not exist, it is
  a silent no-op (a host with no ``experiments/`` at all has nothing to say) — unchanged
  residual.
* the flag EXPLICITLY GIVEN a value that does not exist is now REFUSED (D53: this was the
  documented asymmetry with ``--protect-file``, which has always refused an explicit nonexistent
  path). "Explicitly given" is detected from whether argparse actually consumed a value for this
  flag, not from comparing the resolved value against the default string, so an abbreviated
  spelling (e.g. ``--exp``) is detected identically to the literal flag name.
* TWO OR MORE ``--experiments-dir`` values where one resolves (realpath) to the same directory as
  another, or to a directory NESTED inside another, are collapsed to the outer/first one before
  scanning — never a refusal: a duplicate or nested request is an operator redundancy, not an
  operator error, and walking the same real subtree twice would double-count every stat
  ``load_manifests`` reports (dirs walked, manifest files read) for the SAME manifests — the
  "nested scan dirs double-counted" NIT this row closes. The collapse is REPORTED on stdout
  ("N nested/duplicate scan dir(s) collapsed …"), and the scan line keeps counting the dirs the
  OPERATOR named ("1 of 2 scan dir(s) visited"), never the post-collapse list.
* A COLLAPSE IS A COUNTING FIX AND MAY NEVER CHANGE A VERDICT, so it happens ONLY when both of
  these hold — otherwise BOTH dirs are kept and the duplicate stat is simply tolerated:

  - NEITHER dir is reached by crossing a SYMLINK (``realpath != abspath``). A symlinked scan dir
    is not a redundancy: it is the input that produces 881I's "manifest reached only by crossing
    a SYMLINKED DIRECTORY" REFUSAL, and collapsing it on realpath before the scan ERASED that
    refusal (measured: ``--experiments-dir experiments experiments_link`` exit 2 -> exit 0). A
    symlinked scan dir is therefore never dropped, and never absorbs another dir's scan either.
  - THE SURVIVING DIR'S WALK PROVABLY REACHES the dropped one. Being underneath it is not enough:
    the walk does not descend into ``.git`` or ``__pycache__``, so ``--experiments-dir
    experiments experiments/__pycache__`` really does need both dirs (measured: collapsing it
    silently dropped a manifest, "1 protect" -> "0 protect", at exit 0 — a WITHDRAWN protection,
    the dangerous direction, not a double count).

  With those two conditions, the surviving dir's walk is PROVEN to reach the dropped directory
  against the walk's own name-based prune list (:data:`_WALK_PRUNED_DIRNAMES`, checked by
  :func:`_walk_reaches`) — not against every possible I/O outcome. Reachability cannot be proven
  past an I/O error: a non-listable intermediate directory (measured — an ancestor directory at
  mode 111, so the surviving walk's own ``os.walk`` cannot list it) still stops the surviving walk
  cold, and the collapse still drops the nested dir it can no longer reach, "1 protect" ->
  "0 protect" on the very same host. That is NOT a silent withdrawal, though: the
  ``PermissionError`` raises :class:`ManifestScanIncomplete` and the whole run REFUSES at exit 2
  ("Permission denied" printed, on both the collapsed and the uncollapsed invocation), so no
  verdict ever changes and the fail-closed direction holds — but "nothing under a collapsed
  directory goes un-scanned" is not literally true; only "nothing goes un-scanned without the
  scan itself loudly refusing" is.

KNOWN RESIDUAL, NAMED AND NOT CLOSED BY THIS ROW (fail-safe in direction — a diagnostic
imprecision, never a silent mis-accept): line-number DRIFT between the two parsers on a manifest
that contains an exotic Unicode line separator (``\\v``, ``\\f``, ``\\x1c``-``\\x1e``, ``\\x85``,
U+2028, U+2029). PROTECTED.txt's line counter is ``str.splitlines()`` (which breaks on all of
those); SPEC.md's is a plain ``\\n`` count. A manifest using one of those bytes could get a
different line number reported by each parser for the "same" position. Not reachable on the real
corpus (0 of 56 files use any of these bytes) and never changes an accept/refuse decision, only
the number in a message.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import stat
import sys
import time
from typing import NamedTuple

CKPT_PATTERNS = (re.compile(r"^snapshot_ep\d+$"), re.compile(r"^ckpt(_.*)?$"),
                 re.compile(r"^checkpoint_\d+$"))
FORBIDDEN_PARTS = (".git", "trading_data", "fixtures", "backups")


def _is_ckpt_dir(path: str) -> bool:
    base = os.path.basename(path.rstrip("/"))
    if any(p.match(base) for p in CKPT_PATTERNS):
        return True
    # a flax checkpoint dir: contains checkpoint_* files
    try:
        return any(n.startswith("checkpoint_") for n in os.listdir(path))
    except OSError:
        return False


def _safe_target(path: str, roots: list[str]) -> tuple[bool, str]:
    rp = os.path.realpath(path)
    if os.path.islink(path):
        return False, "is a symlink"
    parts = rp.split(os.sep)
    for bad in FORBIDDEN_PARTS:
        if any(bad in p for p in parts):
            return False, f"path contains forbidden component '{bad}'"
    if not any(rp.startswith(os.path.realpath(r) + os.sep) for r in roots):
        return False, "outside configured roots"
    if not os.path.isdir(rp):
        return False, "not a directory"
    return True, "ok"


def _du(path: str) -> int:
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path, followlinks=False):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def _epoch_of(path: str) -> int | None:
    m = re.search(r"(?:snapshot_ep|checkpoint_)(\d+)", os.path.basename(path.rstrip("/")))
    return int(m.group(1)) if m else None


def _load_scores(run_dir: str, score_regex: str | None, log_glob: str) -> dict[int, float]:
    """epoch -> score, from the most authoritative source available."""
    idx = os.path.join(run_dir, "checkpoints_index.json")
    if os.path.exists(idx):
        try:
            rows = json.load(open(idx))
            out = {}
            for r in rows if isinstance(rows, list) else rows.get("checkpoints", []):
                if r.get("epoch") is not None and r.get("score") is not None:
                    out[int(r["epoch"])] = float(r["score"])
            if out:
                return out
        except Exception as e:
            print(f"  [warn] unreadable {idx}: {e} — treating run as unscored", file=sys.stderr)
    sj = os.path.join(run_dir, "scores.jsonl")
    if os.path.exists(sj):
        out = {}
        for line in open(sj):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                out[int(d["epoch"])] = float(d["score"])
            except Exception:
                continue
        if out:
            return out
    if score_regex:
        import glob as _glob
        pat = re.compile(score_regex)
        out = {}
        for lf in _glob.glob(os.path.join(run_dir, log_glob)) + _glob.glob(log_glob):
            try:
                for line in open(lf, errors="replace"):
                    m = pat.search(line)
                    if m:
                        try:
                            out[int(m.group("epoch"))] = float(m.group("score"))
                        except Exception:
                            continue
            except OSError:
                continue
        if out:
            return out
    return {}


def free_gb(path: str) -> float:
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 1e9


def collect(roots: list[str]) -> dict[str, list[str]]:
    """run_dir -> [checkpoint dirs]"""
    runs: dict[str, list[str]] = {}
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, _files in os.walk(root, followlinks=False):
            keep_walking = []
            for d in dirnames:
                full = os.path.join(dirpath, d)
                if os.path.islink(full):
                    continue
                if _is_ckpt_dir(full):
                    runs.setdefault(dirpath, []).append(full)
                else:
                    keep_walking.append(d)
            dirnames[:] = keep_walking
    return runs


def plan(runs, scores_of, keep_best, keep_latest, under_floor, keep_unscored):
    """-> (deletions[(path, bytes, reason)], kept_summary[str])"""
    deletions, summary = [], []
    for run_dir, ckpts in sorted(runs.items()):
        scores = scores_of(run_dir)
        by_mtime = sorted(ckpts, key=lambda p: os.path.getmtime(p), reverse=True)
        newest = set(by_mtime[:max(1, keep_latest)])
        scored = [(p, scores.get(_epoch_of(p))) for p in ckpts]
        have = [(p, s) for p, s in scored if s is not None]
        none = [p for p, s in scored if s is None]
        keep = set(newest)
        if have:
            best = sorted(have, key=lambda t: t[1], reverse=True)
            keep |= {p for p, _ in best[:max(1, keep_best)]}
            for p, s in best[max(1, keep_best):]:
                if p in keep:
                    continue
                deletions.append((p, _du(p), f"score {s:.6g} outside top-{keep_best}"))
        if none:
            # unscored: never pruned on score. Only recency-pruned when disk is under the floor.
            keep |= set(sorted(none, key=os.path.getmtime, reverse=True)[:keep_unscored])
            if under_floor:
                for p in sorted(none, key=os.path.getmtime, reverse=True)[keep_unscored:]:
                    if p not in keep:
                        deletions.append((p, _du(p), "UNSCORED, recency-pruned (disk under floor)"))
        summary.append(f"{run_dir}: {len(ckpts)} ckpts, {len(have)} scored, keeping {len(keep)}")
    return deletions, summary


# ======================================================================================
# TASK_881I — MANIFEST INGESTION ONLY (D45 capacity re-cut of the halted TASK_881)
# ======================================================================================
# This section PARSES, CLASSIFIES and REFUSES. It deletes nothing, and it is never called
# from `collect()` / `plan()` / the `--apply` delete loop above, which stay byte-identical to
# what shipped before this row (canon parity). Everything below is reached only through
# `--verify-manifests`, a separate, additive CLI mode that returns before any checkpoint is
# ever discovered.
#
# TASK_881 (the halted work, tag `stash-881-halted-d45`) welded this ingestion to an explicit
# protection MODEL (which run dirs are "covered", and therefore prunable) and a DELETER, and
# died on its 3rd cycle inside the ingestion half: `_parse_spec_block` read only the FIRST
# `protected:`/`covers:` block in a SPEC.md while a second, independently-looping
# `_strip_spec_manifest_blocks` erased EVERY such block's text — so a checkpoint named only in
# a second block lost both its protection and its citation, deleted at exit 0. D45 re-cut the
# row so the fix (one function owns parse AND strip) ships on its own, before the protection
# model and deleter are rebuilt on top of it (TASK_881D).
#
# The protection model, the deleter, `plan()`, `root_resolution()`, and the TTL re-scan are
# TASK_881D's and are NOT touched or reimplemented here — this section does not decide what
# survives, only what a manifest or score file actually SAYS, and whether that document can be
# trusted at all.

MANIFEST_BASENAMES = ("PROTECTED.txt", "SPEC.md")
# The score-bearing basenames `_load_scores_strict` will try to open, named once so the enumeration
# in `_find_score_dirs` cannot drift from the files actually read.
SCORE_BASENAMES = ("scores.jsonl", "checkpoints_index.json")
_MANIFEST_LINE_RE = re.compile(r"^(protect|covers):\s*(\S+)\s*$")
# The ONLY line shape a SPEC.md `protected:` / `covers:` block may contain besides its header.
# Named once because `parse_spec_manifest` now has to ask "is this line an item?" in TWO places —
# once to consume a line, once to decide whether a blank line ENDS the list or merely SPLITS it —
# and two spellings of the same question drifting apart is precisely this row's failure mode
# (cycle 3 died of a parser and a stripper disagreeing about what counts as manifest content).
_SPEC_ITEM_RE = re.compile(r"^\s*-\s*(\S+)\s*$")
# `\s*$` IS THE FIX, not decoration. Unanchored (`^\s*-\s*(\S+)`), this regex consumed only the
# FIRST token of `- runs/keep_a runs/keep_b` and produced ONE entry — a SUBSET grant at exit 0,
# "verify-manifests: OK", with `runs/keep_b` protected by nothing and cited by nothing. Worse, the
# span computed for the block covered the WHOLE line, so `strip_spans` erased `runs/keep_b`'s only
# mention from the text as well: a strip removing more than the parse consumed, which is verbatim
# the defect D45 re-cut this row over. `_MANIFEST_LINE_RE` above has always been `$`-anchored and
# REFUSES the identical shape in a PROTECTED.txt; the two parsers disagreed and the anchored one
# was right. A line naming more than one path now REFUSES through the same INTERRUPTED/malformed
# channel as every other un-consumable line, so a subset grant is impossible at the LINE level as
# well as at the block level.
_SPEC_BULLET_RE = re.compile(r"^\s*-\s*(\S+)")
# "This line is TRYING to be a list item" — deliberately the OLD, unanchored grammar, and kept
# separate rather than merged into the anchored one. It is used for two decisions that must stay
# LOOSE while the entry grammar goes strict:
#   * whether a line inside a started list is an attempted item (so `- a b` reaches the
#     multi-path refusal instead of falling through to the generic non-item branch), and
#   * whether the far side of a blank line RESUMES the list (so `protected:\n- a\n\n- b c\n` is
#     still refused as SPLIT; matching only the ANCHORED grammar there would have quietly ended
#     the block at the hole and granted `a` alone — re-opening the subset hole one layer up).
# Best-effort, DIAGNOSTIC ONLY: used to name "the run dir a malformed line names" in a refusal
# message. It never creates a protect/covers entry — a malformed line is never trusted.
_PATHLIKE_HINT_RE = re.compile(r"([./A-Za-z0-9_-]+/[./A-Za-z0-9_-]+)")

_THEMATIC_BREAK_RE = re.compile(r"^-{2,}$")
# A markdown THEMATIC BREAK (horizontal rule), e.g. ``---``/``----``. It superficially matches
# `_SPEC_BULLET_RE` (the leading `-` reads as the bullet marker, the remaining dashes as a
# one-token "path"), and without this guard `---` silently ingested as an item naming the
# literal path `--` — almost always landing in `ManifestSet.dangling` (fail-safe in EFFECT, since
# a path made of bare dashes essentially never exists) but never a parse anyone reviewed. 881G
# ingestion NIT (D53): a line that IS ENTIRELY two-or-more dashes is never a list item, at either
# of `_is_thematic_break`'s two call sites in `parse_spec_manifest`.


def _is_thematic_break(line: str) -> bool:
    """True for a line that is a markdown thematic break, never a ``- <path>`` item — see
    :data:`_THEMATIC_BREAK_RE`."""
    return bool(_THEMATIC_BREAK_RE.match(line.strip()))


class ManifestEntry(NamedTuple):
    kind: str          # "protect" | "covers"
    path: str          # raw path text, exactly as written (not yet realpath'd)
    source: str        # manifest file this entry came from
    lineno: int         # 1-based line number (PROTECTED.txt) or block-header line (SPEC.md)


class MalformedManifestLine(NamedTuple):
    source: str
    lineno: int
    raw: str
    hint_path: str | None   # best-effort path-like token in the bad line, diagnostic ONLY


def parse_protected_txt(text: str, source: str
                        ) -> tuple[list[ManifestEntry], list[tuple[int, int]],
                                   list[MalformedManifestLine]]:
    """Single pass over a PROTECTED.txt: the read that PRODUCES entries is the same read that
    computes the character SPANS a strip may remove (scope item 1) — there is no second
    function re-deriving "what counts as manifest content" that can drift from this one.

    Recognized: ``protect: <path>`` / ``covers: <path>``, one directive per line; blank lines
    and ``#`` comments are consumed (spanned, so a strip removes them too) but produce no entry.
    Any OTHER non-blank line is MALFORMED: it is reported, NOT spanned (so a strip can never
    make a malformed line disappear without a trace) and NEVER turned into an entry. TASK_881
    required the directive, not just a path, so a mis-formatted line cannot accidentally protect
    nothing while looking like it protects something; this row goes one step further (scope item
    2, D45 planner note 2/3's spirit) and refuses to treat it as reported-and-ignored — see
    ``load_manifests``, which turns any non-empty ``malformed`` into a refusal of the WHOLE file,
    not a per-line skip.
    """
    entries: list[ManifestEntry] = []
    spans: list[tuple[int, int]] = []
    malformed: list[MalformedManifestLine] = []
    pos = 0
    for lineno, raw in enumerate(text.splitlines(keepends=True), 1):
        start, end = pos, pos + len(raw)
        pos = end
        line = raw.strip()
        if not line or line.startswith("#"):
            spans.append((start, end))
            continue
        m = _MANIFEST_LINE_RE.match(line)
        if m:
            entries.append(ManifestEntry(m.group(1), m.group(2), source, lineno))
            spans.append((start, end))
            continue
        hint = _PATHLIKE_HINT_RE.search(line)
        malformed.append(MalformedManifestLine(source, lineno, raw.rstrip("\n"),
                                                hint.group(1) if hint else None))
    return entries, spans, malformed


def parse_spec_manifest(text: str, source: str
                        ) -> tuple[list[ManifestEntry], list[tuple[int, int]],
                                   list[MalformedManifestLine]]:
    """Single pass over a SPEC.md's ``protected:`` / ``covers:`` list blocks — the exact fix
    D45 re-cut this row over.

    Cycle-3's ``_parse_spec_block`` read entries with one function (``re.search``, first match
    only per key) and ``_strip_spec_manifest_blocks`` computed what to remove with a SECOND,
    independently-looping function (``while True: re.search(...)``) that kept going past the
    first match. A SPEC.md with two ``protected:`` blocks therefore had its second block's
    entries silently dropped by the parser while the stripper erased BOTH blocks' text from
    what the advisory prose scanner would see — a checkpoint named only in the second block was
    invisible to protection AND its citation removed before any prose scan ever ran. Deleted at
    exit 0.

    THE FIX: one walk finds every ``^protected:\\n`` / ``^covers:\\n`` header for each key.
    Exactly one header per key is the well-formed case: its list lines become entries, and its
    exact text (header + list lines) becomes the span a strip may remove — entries and spans
    come from the SAME loop, so a strip can never remove more than this parse actually consumed
    (scope item 1, gate ii). TWO OR MORE headers for the same key is a REFUSAL (scope item 3):
    NO entries are produced for that key and NOTHING is spanned for it, so the ambiguity can
    never be silently resolved by preferring one block over the other.

    THE HEADER MATCH IS NEWLINE-OPTIONAL, and that matters. An earlier version required a
    literal newline after the key (``^protected:[ \\t]*\\n``), so a file whose FINAL line is the
    header and which lacks a trailing newline — a torn or truncated write, the same class the
    ``scores.jsonl`` guard already treats as realistic, not a corruption exotic — matched
    NOTHING: no entries, no malformed report, exit 0 saying OK. ``parse_protected_txt``'s
    ``splitlines(keepends=True)`` has always been robust to a missing final newline; this parser
    now is too, so the header is at least SEEN. Two further shapes are then refusals rather than
    silent zero-coverage, per this row's own contract ("a line that is neither consumed nor
    refusable is a refusal, never a silent skip"):

    * a header line carrying INLINE content (``protected: runs/a``) — it reads as granting
      coverage and grants none;
    * a well-formed header with ZERO list items — a block that names nothing is indistinguishable
      from a block whose items were lost to a truncated write, so it WITHDRAWS rather than
      quietly granting nothing.

    THE SAME REASONING, EXTENDED TO A *PARTIALLY* CONSUMED BLOCK (the cycle-2 fix). The item loop
    used to ``break`` on any line that was not ``- <path>``, so a mid-block ``#`` comment, a ``*``
    bullet, a bare ``-``, a stray word, or a blank line with more items after it ENDED the list
    silently: a 2-item block with one comment between the items granted ONE protection, at exit 0,
    "verify-manifests: OK", with no diagnostic of any kind. A comment line cost a protection. The
    empty-block guard's own argument covers this case exactly — a partially consumed block is as
    indistinguishable from a truncated write as an empty one — so the loop now REFUSES the whole
    block instead of ending it:

    * a non-blank, non-item line after at least one consumed item ⇒ refusal (nothing parsed);
    * a blank line whose far side RESUMES the list with another ``- <path>`` item ⇒ refusal. A
      blank line followed by prose, a heading, or EOF is still the block's legitimate terminator,
      which is what keeps a normal SPEC.md (block, blank line, prose) parsing.

    THE SAME REASONING AT THE *LINE* LEVEL (the cycle-3 fix). ``_SPEC_ITEM_RE`` was UNANCHORED, so
    ``- runs/keep_a runs/keep_b`` matched, produced ONE entry for the first token, and spanned the
    WHOLE line: a SUBSET grant at exit 0, ``verify-manifests: OK``, with the second path protected
    by nothing — and its only mention in the document removed by :func:`strip_spans`, so the strip
    DID remove more than the parse consumed. The identical shape in a ``PROTECTED.txt`` has always
    been refused (``_MANIFEST_LINE_RE`` is ``$``-anchored), so the two parsers disagreed and the
    anchored one was right. An attempted item naming more than one token is now a refusal too.

    AND AT THE *HEADER* LEVEL (also cycle-3). ``^{key}:`` is column-0, lower-case and exact, so a
    header that misses it by one character — indented under a Markdown bullet, capitalized, behind
    a ``>``, or preceded by a BOM — matched nothing and fell through ``if not headers: continue``:
    the WHOLE declared block vanished at exit 0, with the row's own scan counts ("1 manifest
    file(s) read / 0 protect") making the false green read like a considered answer. A deliberately
    LOOSE near-header pattern now refuses that case; see the drop site for the argument and for the
    corpus measurement that shows it refuses nothing that exists today.

    ``#`` IS DELIBERATELY NOT A SUPPORTED COMMENT in a SPEC.md block, unlike in
    ``parse_protected_txt``: SPEC.md is Markdown, where a leading ``#`` is heading TEXT, and
    "supporting" it would mean SPANNING it — letting :func:`strip_spans` delete a heading, i.e.
    re-creating the strip-removes-more-than-the-parse-consumed defect this row exists to close.
    The refusal message says so. See the inline comment at the drop site for the full argument.

    ACCEPTED, NOT A DEFECT (judged 2026-08-19, do not "fix"): a ``protected:`` / ``covers:``
    header written inside SPEC.md PROSE, or inside a fenced code block, parses as a real block.
    That is fail-SAFE in the only direction that matters here — the ENTRIES it produces can only
    ADD, i.e. make the (future, TASK_881D) deleter MORE conservative — and a bogus path so
    ingested does not pass quietly: it lands in ``ManifestSet.dangling`` and hits
    ``load_manifests``'s loud dangling-entry refusal at exit 2. Teaching this parser to skip
    fences would create the opposite, unsafe failure: a real header ignored because something
    upstream looked like a fence.

    CORRECTED (881G gate (i), from a reader who followed the doc): "can only ADD" is true of the
    ENTRIES and FALSE of the FILE. Fences are not detected, so a closing backtick-fence line
    sitting IMMEDIATELY after the example's last item is a non-item line inside a started list and
    REFUSES THE WHOLE FILE as INTERRUPTED — the natural way to write a fenced example withdraws
    everything the file grants. One blank line before the closing fence terminates the list and
    the file parses. The module docstring's grammar section states this shape with both examples.
    """
    entries: list[ManifestEntry] = []
    spans: list[tuple[int, int]] = []
    malformed: list[MalformedManifestLine] = []
    for key in ("protected", "covers"):
        kind = "protect" if key == "protected" else "covers"
        # `(?:\n|\Z)`, not `\n`: a header on an unterminated final line must still be SEEN.
        headers = list(re.finditer(rf"^{key}:(?P<rest>[^\n]*)(?:\n|\Z)", text, re.MULTILINE))
        # NEAR-MISS HEADER DETECTION — the last silent `continue` in this ingestion layer.
        #
        # The strict header above is column-0, lower-case and exact. A header that misses it by
        # ONE CHARACTER therefore produced `headers == []` and fell straight through the
        # `if not headers: continue` below: no entries, no malformed report, "0 protect",
        # "verify-manifests: OK", exit 0 — and the counts added by this row make the false green
        # MORE convincing, because "1 manifest file(s) read / 0 protect" reads as "that file grants
        # nothing" when the file plainly declares a block. Measured shapes, all silent before this
        # guard: an INDENTED block (`- Checkpoints:` / `  protected:` — the natural Markdown nesting
        # an author reaches for), `PROTECTED:` in capitals, and a header preceded by a BOM. The
        # blast radius is bigger than any per-line defect: the WHOLE declared block vanishes.
        #
        # It also contradicted this parser's own contract — "a line that is neither consumed nor
        # refusable is a refusal, never a silent skip" — which until now was enforced only INSIDE
        # an already-matched block, and falsified the row's accepted residual that a
        # `protected:`/`covers:` line in SPEC.md PROSE is a REFUSAL: a near-miss header was not a
        # refusal, it was silence.
        #
        # The detector is deliberately LOOSER than the grammar (leading whitespace, `>` blockquote
        # markers, `*`/`-` bullets, any case, space before the colon) and it is compared per LINE
        # against the strict headers actually matched, so a near-miss is caught even in a file that
        # ALSO carries a well-formed block. Verified free on the real corpus: all 56
        # PROTECTED.txt/SPEC.md files under `experiments/` + `runs/` produce ZERO matches, so this
        # cannot refuse anything that exists today. Fail-safe direction: it only ever WITHDRAWS.
        near = list(re.finditer(rf"(?im)^[\s>*-]*{key}\s*:", text))
        strict_lines = {text.count("\n", 0, h.start()) + 1 for h in headers}
        # By END position (just past the colon), never by start: `[\s>*-]*` may begin a match on a
        # preceding BLANK line, so a well-formed header's own loose match can start one line early.
        near_bad = [m for m in near
                    if text.count("\n", 0, m.end()) + 1 not in strict_lines]
        if near_bad:
            m = near_bad[0]
            near_lineno = text.count("\n", 0, m.end()) + 1
            # The WHOLE offending line, unstripped and repr'd: what makes this shape dangerous is
            # that the difference is invisible (leading spaces, a capital, a BOM), so a message that
            # stripped it would print something indistinguishable from a valid header. Sliced on
            # "\n" boundaries, the same way `near_lineno` was counted — `str.splitlines()` also
            # splits \x0b/\x0c/\x85/U+2028-9 and would drift from the number reported here.
            _ls = text.rfind("\n", 0, m.end()) + 1
            _le = text.find("\n", m.end())
            raw_line = text[_ls:_le if _le != -1 else len(text)]
            malformed.append(MalformedManifestLine(
                source, near_lineno,
                f"line {near_lineno} NEARLY spells a '{key}:' manifest header "
                f"({raw_line!r}) but does not match it — a manifest header must be "
                f"written at column 0, in lower case, exactly '{key}:'. An indented, capitalized, "
                f"bullet-prefixed, blockquoted or BOM-preceded spelling reads to its author as a "
                f"granted block and grants NOTHING, so it is REFUSED rather than skipped in "
                f"silence; none parsed.", None))
            continue
        if not headers:
            continue
        if len(headers) > 1:
            lines = [text.count("\n", 0, h.start()) + 1 for h in headers]
            malformed.append(MalformedManifestLine(
                source, lines[0],
                f"duplicate '{key}:' block ({len(headers)} occurrences at lines {lines}) — "
                f"refused, none parsed", None))
            continue
        h = headers[0]
        start, end = h.start(), h.end()
        block_lineno = text.count("\n", 0, start) + 1
        inline = h.group("rest").strip()
        if inline:
            hint = _PATHLIKE_HINT_RE.search(inline)
            malformed.append(MalformedManifestLine(
                source, block_lineno,
                f"'{key}:' header carries INLINE content {inline!r} — a manifest block header "
                f"must stand alone above a '- <path>' list; refused, none parsed",
                hint.group(1) if hint else None))
            continue
        block: list[ManifestEntry] = []
        item_lines = text[h.end():].splitlines(keepends=True)
        # Consumed by INDEX, not by a bare `for`: deciding whether a blank line ENDS this list or
        # SPLITS it requires looking PAST that line (see the blank-line branch below), which a
        # `for line in ...` loop cannot do.
        interrupted: MalformedManifestLine | None = None
        i = 0
        while i < len(item_lines):
            line = item_lines[i]
            item_lineno = block_lineno + 1 + i
            if not line.strip():
                # A blank line is this block's legitimate TERMINATOR — a SPEC.md's prose resumes
                # after one, and `test_roundtrip_spec_md_single_block` pins that. It is NOT a
                # terminator when the list RESUMES after the gap: items separated from their
                # header by a hole are exactly what a torn or badly re-merged write looks like,
                # which is the SAME indistinguishability the empty-block guard below refuses. So
                # look past the run of blanks: a `- <path>` line on the far side SPLITS one list
                # and is refused; anything else (prose, a heading, EOF) genuinely ends the block.
                j = i
                while j < len(item_lines) and not item_lines[j].strip():
                    j += 1
                # `_SPEC_BULLET_RE`, not the anchored `_SPEC_ITEM_RE`: "does the list RESUME?" has
                # to stay the LOOSE question. Anchoring it would make `protected:\n- a\n\n- b c\n`
                # end the block cleanly at the hole and grant `a` alone — the subset hole moved one
                # layer up rather than closed.
                # `and block`: the SPLIT diagnosis only makes sense when there is a list ABOVE the
                # hole to be split FROM. With nothing consumed yet — `protected:\n\n- item`, which
                # is the shape markdownlint MD032 REQUIRES (a list surrounded by blank lines) — the
                # SPLIT message told the author to "close the gap", i.e. straight into a lint
                # violation, about a gap that is not between two halves of anything. Falling through
                # to the EMPTY-block guard instead keeps the refusal EXACTLY as it was (this shape
                # was refused before and is refused now, same exit code) and only makes the wording
                # true: under this grammar the block names nothing. WHETHER the MD032 form should
                # PARSE is a grammar/SPEC decision that 881D's corpus depends on, and it is the
                # planner's call — deliberately NOT decided here.
                #
                # DECIDED (881G, D53): the MD032 form PARSES. When `block` is still empty, a
                # leading blank-line run is not a hole in an existing list — there is no list yet
                # to split — it is the mandatory list-fence markdownlint MD032 requires. If the
                # far side resumes with a genuine `- <path>` item (never a thematic break —
                # `_is_thematic_break` excludes `---`/`--`/etc., which would otherwise silently
                # ingest as an item naming the trailing dashes), skip the gap and keep consuming;
                # otherwise (prose, a heading, or EOF) this is genuinely an empty block, exactly
                # as before. Gated on `not block` so it can never reach — and never changes — the
                # SPLIT check immediately below, which keeps refusing a gap once the list has
                # actually started.
                if (not block and j < len(item_lines)
                        and _SPEC_BULLET_RE.match(item_lines[j])
                        and not _is_thematic_break(item_lines[j])):
                    # THE SKIPPED GAP IS PART OF THE SPAN. `end` is the running right edge of
                    # "what this parse consumed", and every OTHER consumed line advances it (see
                    # `end += len(line)` at the bottom of this loop). Jumping `i` to `j` without
                    # advancing `end` left the span SHORT by exactly the gap's byte length, so its
                    # right edge landed INSIDE the last item line and `strip_spans` cut mid-item:
                    # `protected:\n\n\n\n- runs/a\n- runs/b\n` stripped to the fragment `/b\n`,
                    # and with a wider gap the last item SURVIVED the strip entirely. That
                    # falsifies `strip_spans`' own contract ("can only remove what that parse
                    # actually consumed" — it removed something else) on the exact shape 881D's
                    # corpus will use, and it is the same parse-and-span-disagree class D53 was
                    # re-cut over, arriving through the new accept path instead of the old one.
                    # The blank run is consumed BY this block (it is the MD032 list fence, not
                    # free text between two blocks), so it belongs inside the span: a strip
                    # removes the header, its fence, and its items, and nothing else.
                    end += sum(len(_gap_line) for _gap_line in item_lines[i:j])
                    i = j
                    continue
                if j < len(item_lines) and block and _SPEC_BULLET_RE.match(item_lines[j]):
                    interrupted = MalformedManifestLine(
                        source, item_lineno,
                        f"'{key}:' list is SPLIT by a blank line at line {item_lineno} and then "
                        f"RESUMES with a '- <path>' item at line {block_lineno + 1 + j} — a hole "
                        f"inside a list is indistinguishable from a truncated write that lost the "
                        f"items in between, and ending the list at the hole would silently drop "
                        f"every item below it; refused, none parsed. Close the gap, or move the "
                        f"later items into the same unbroken list.", None)
                break
            bullet = _SPEC_BULLET_RE.match(line)
            item = _SPEC_ITEM_RE.match(line) if bullet else None
            if bullet is not None and item is not None and _is_thematic_break(line):
                # 881G ingestion NIT (D53): `---`/`--`/… matches BOTH `_SPEC_BULLET_RE` and
                # `_SPEC_ITEM_RE` (the trailing dashes read as a single "token"), so without this
                # it silently became an item naming the literal path `--`. Neither `bullet` nor
                # `item` is trusted for this shape — it falls through to the SAME
                # interruption/empty-block handling as any other non-item line, immediately
                # below (local to this iteration only; the blank-line resume checks above guard
                # against the same shape independently, at their own call sites).
                bullet = item = None
            if bullet is not None and item is None:
                # AN ATTEMPTED ITEM THAT NAMES MORE THAN ONE TOKEN — the line-level twin of every
                # block-level refusal above. `- runs/keep_a runs/keep_b` matched the old unanchored
                # grammar, produced ONE entry for `runs/keep_a`, and spanned the WHOLE line: a
                # subset grant at exit 0 whose dropped path lost its protection AND its citation
                # (`strip_spans` erased the text that named it). Refusing here is not a new policy,
                # it is `_MANIFEST_LINE_RE`'s existing one — a PROTECTED.txt line of the identical
                # shape has always been refused as malformed — finally applied on both sides.
                # NOT gated on `block`: unlike the generic non-item branch below, this line is
                # unambiguously an attempted item whatever its position, so the EMPTY-block guard's
                # "it names nothing" message would be factually wrong about it (it names two).
                rest = line.strip()
                hint = _PATHLIKE_HINT_RE.search(rest)
                interrupted = MalformedManifestLine(
                    source, item_lineno,
                    f"'{key}:' list item at line {item_lineno} carries MORE THAN ONE token "
                    f"({rest!r}) — a manifest item names exactly ONE path per line. Consuming the "
                    f"first token and ignoring the rest would GRANT A SUBSET while spanning (and "
                    f"so erasing) the whole line, leaving the other path(s) protected by nothing "
                    f"and cited by nothing; refused, none parsed. Put each path on its own "
                    f"'- <path>' line, and note that trailing comments are not supported here.",
                    hint.group(1) if hint else None)
                break
            if not item:
                # A NON-BLANK line that is not a `- <path>` item, inside a list that has already
                # produced items. `break`ing here ended the list SILENTLY, so any interloper — a
                # `#` line, a `*` bullet, a stray word, a bare `-` with no path — cost every item
                # below it its protection at exit 0, "verify-manifests: OK", with no diagnostic of
                # any kind. That is the same failure class this parser's empty-block guard already
                # reasons about: a PARTIALLY consumed block is as indistinguishable from a
                # truncated write as an EMPTY one is, so it WITHDRAWS rather than quietly granting
                # a subset.
                #
                # DESIGN DECISION — `#` IS NOT A COMMENT HERE, deliberately, unlike in
                # `parse_protected_txt`. PROTECTED.txt is a plain-text directive file whose own
                # grammar defines `#` as a comment; SPEC.md is MARKDOWN, where a leading `#` is a
                # HEADING, i.e. real document content. Giving it the `parse_protected_txt`
                # treatment means SPANNING it — and `spans` is the channel `strip_spans` deletes
                # through, so a "supported comment" would let a strip erase a heading. That would
                # be a fresh instance of the exact defect D45 re-cut this row over (a strip
                # removing more than the parse legitimately consumed), traded for the convenience
                # of an in-block comment. Refusing costs a SPEC author one blank line before their
                # prose and says so in the message; spanning costs a silent edit to their
                # document. So this parser spans NOTHING but a header and its `- <path>` items,
                # and every other line inside a started list is a refusal.
                if block:
                    hint = _PATHLIKE_HINT_RE.search(line)
                    interrupted = MalformedManifestLine(
                        source, item_lineno,
                        f"'{key}:' list is INTERRUPTED at line {item_lineno} by "
                        f"{line.strip()!r}, which is not a '- <path>' item — ending the list here "
                        f"would silently drop every item BELOW it, indistinguishable from a "
                        f"truncated write; refused, none parsed. NOTE: '#' is not a comment "
                        f"inside a SPEC.md manifest block (SPEC.md is Markdown, where '#' is "
                        f"heading TEXT) — put comments and prose after a blank line that ends "
                        f"the list.",
                        hint.group(1) if hint else None)
                # else: nothing consumed yet, so this is a header that names nothing — the
                # EMPTY-block guard below owns that case and its message is the accurate one.
                break
            block.append(ManifestEntry(kind, item.group(1), source, block_lineno))
            end += len(line)
            i += 1
        if interrupted is not None:
            malformed.append(interrupted)
            continue
        if not block:
            malformed.append(MalformedManifestLine(
                source, block_lineno,
                f"'{key}:' block is EMPTY — it names nothing, which is indistinguishable from a "
                f"torn/truncated write that lost its '- <path>' items; refused, none parsed",
                None))
            continue
        entries.extend(block)
        spans.append((start, end))
    return entries, spans, malformed


def strip_spans(text: str, spans: list[tuple[int, int]]) -> str:
    """Remove exactly the character ranges ``spans`` names, nothing else.

    ``spans`` must come from the SAME parse call that produced them (``parse_protected_txt`` /
    ``parse_spec_manifest``) — by construction this can only remove what that parse actually
    consumed as manifest content (gate ii: stripping anything MORE is impossible, because
    ``spans`` is the only channel span information travels through; there is no independent
    second pass that could disagree with the parse about what counts as manifest content).
    """
    out = []
    pos = 0
    for start, end in sorted(spans):
        if start < pos:
            continue  # overlapping/out-of-order spans (should not occur) are never double-cut
        out.append(text[pos:start])
        pos = end
    out.append(text[pos:])
    return "".join(out)


# --- entry SHAPE classification: the ROOT both link defects share -----------------------------
# EVERY security-relevant decision in this section is made from an `lstat`-based classification
# computed HERE, and NEVER from how `os.walk` bucketed an entry into `dirnames` / `filenames`.
# `os.walk` splits entries with `DirEntry.is_dir()`, which FOLLOWS symlinks, so THE SAME NAME
# lands in a different bucket depending on what its target happens to be — and two of this row's
# cycles died on opposite sides of exactly that one mistake:
#
#   shape of `experiments/<id>/PROTECTED.txt`  os.walk bucket  what a link-following split got wrong
#   ----------------------------------------  --------------  -------------------------------------
#   regular file                              filenames       (the intended case: read it)
#   symlink -> regular file                   filenames       correctly refused
#   symlink -> DIRECTORY, live                DIRNAMES        CYCLE-3 BLOCKER: `load_manifests`
#                                                             tested `fn in MANIFEST_BASENAMES`
#                                                             over `filenames` ONLY, so the name
#                                                             was never evaluated at all — silent
#                                                             exit 0, output byte-identical to a
#                                                             tree with NO manifest, i.e. an
#                                                             operator's protection intent
#                                                             dropped with zero signal
#   symlink -> nonexistent, or a link cycle   filenames       CYCLE-1 BLOCKER, the mirror image:
#                                                             symlink checks ran over `dirnames`
#                                                             ONLY, so this was invisible too
#   real DIRECTORY                            dirnames        walked into as a plain subdir, name
#                                                             never evaluated: silent exit 0
#   fifo / socket / device                    filenames       `open()` on a reader-less FIFO
#                                                             BLOCKS FOREVER: the tool HUNG
#                                                             (measured pre-fix: exit 124 under
#                                                             `timeout 15`) instead of refusing
#   symlink -> symlink -> anything            per FINAL target the chain has to be resolved, not
#                                                             just its first hop
#
# The rule this section therefore enforces once, for the whole shape space rather than per shape:
# manifest-basename identity is tested BEFORE and INDEPENDENTLY of the directory-vs-file split,
# over EVERY entry `os.walk` returned (both lists, concatenated), and a manifest basename that is
# not a real, regular, non-symlink file REFUSES — whatever its target is, and whichever list
# `os.walk` happened to put it in.


class EntryShape(NamedTuple):
    """What a path REALLY is: one ``lstat`` (never follows) plus, for a symlink, one following
    ``stat`` to learn the type at the END of the chain."""

    path: str
    is_link: bool
    kind: str    # "file" | "dir" | "other" | "unstatable" | (links only) "dangling"

    @property
    def is_regular_file(self) -> bool:
        """The ONLY shape a manifest may have: a real, regular, non-symlink file."""
        return not self.is_link and self.kind == "file"


def _kind_of_mode(mode: int) -> str:
    if stat.S_ISDIR(mode):
        return "dir"
    if stat.S_ISREG(mode):
        return "file"
    return "other"


def classify_entry(path: str) -> EntryShape:
    """Classify ``path`` from its own ``lstat``, independently of any caller's bucketing.

    ``kind`` describes the FINAL target for a symlink (``os.stat`` follows the whole chain, so
    ``symlink -> symlink -> dir`` classifies as ``("dir", is_link=True)``) and the entry itself
    otherwise. A link whose chain cannot be resolved — broken target, or a cycle, which fails with
    ``ELOOP`` — is ``"dangling"``. ``"unstatable"`` means even the ``lstat`` failed: the entry
    vanished mid-scan, or its parent became unreadable. That is reported, never assumed empty.
    """
    try:
        st = os.lstat(path)
    except OSError:
        return EntryShape(path, False, "unstatable")
    if not stat.S_ISLNK(st.st_mode):
        return EntryShape(path, False, _kind_of_mode(st.st_mode))
    try:
        target_st = os.stat(path)        # follows the FULL chain; ENOENT/ELOOP -> dangling
    except OSError:
        return EntryShape(path, True, "dangling")
    return EntryShape(path, True, _kind_of_mode(target_st.st_mode))


_MANIFEST_SHAPE_WHY = ("a manifest is the input that can grant coverage to an irreversible "
                       "deleter, so it must be a real, regular, non-symlink file at a real, "
                       "tracked path")


def manifest_shape_refusal(path: str) -> str | None:
    """``None`` iff ``path`` may be opened and read as a manifest; otherwise the refusal message,
    naming the shape actually found.

    Called on EVERY entry whose basename is in ``MANIFEST_BASENAMES``, from EITHER of ``os.walk``'s
    lists (see the shape table above) — that is what makes this a fix for the shape SPACE instead
    of a patch for one shape. The symlink-to-FILE wording is preserved verbatim from the guard that
    already shipped, so an existing refusal's message does not move; the symlink-to-DIRECTORY
    wording is deliberately DISTINCT from ``load_manifests``'s "reached only by crossing a
    SYMLINKED DIRECTORY" and from ``DanglingScanLink``'s "BROKEN SYMLINK", because conflating three
    different diagnoses is how the previous cycles lost track of which shape was covered.
    """
    sh = classify_entry(path)
    if sh.is_regular_file:
        return None
    if sh.is_link:
        if sh.kind == "file":
            return (f"{path}: manifest is a SYMLINK — REFUSED, not read (its target may be "
                    f"untracked or outside the tree). Replace it with the real file.")
        if sh.kind == "dir":
            return (f"{path}: manifest basename is a SYMLINK TO A DIRECTORY — REFUSED, not read. "
                    f"`os.walk` files this shape under `dirnames` (`is_dir()` follows the link), "
                    f"so a filenames-only manifest check never evaluated the name at all and the "
                    f"protection intent it declares was dropped at exit 0; {_MANIFEST_SHAPE_WHY}.")
        if sh.kind == "dangling":
            return (f"{path}: manifest is a SYMLINK whose target does not resolve (broken target, "
                    f"or a link cycle) — REFUSED, not read; {_MANIFEST_SHAPE_WHY}.")
        return (f"{path}: manifest is a SYMLINK to a non-regular file — REFUSED, never opened "
                f"(opening a reader-less FIFO BLOCKS FOREVER); {_MANIFEST_SHAPE_WHY}.")
    if sh.kind == "dir":
        return (f"{path}: manifest basename is a real DIRECTORY, not a file — REFUSED. A directory "
                f"named {os.path.basename(path)!r} is not a manifest, and `os.walk` would "
                f"otherwise walk into it as an ordinary subdir without ever evaluating the name; "
                f"{_MANIFEST_SHAPE_WHY}.")
    if sh.kind == "unstatable":
        return (f"{path}: manifest basename could not be lstat'd (it vanished mid-scan, or its "
                f"parent became unreadable) — REFUSED rather than assumed empty; "
                f"{_MANIFEST_SHAPE_WHY}.")
    return (f"{path}: manifest basename is not a regular file (fifo/socket/device) — REFUSED, "
            f"never opened: `open()` on a reader-less FIFO BLOCKS FOREVER, which HUNG this tool "
            f"instead of refusing; {_MANIFEST_SHAPE_WHY}.")


def _score_source_shape_reason(path: str) -> str | None:
    """``None`` iff ``path`` is safe to OPEN as a score file; otherwise the reason (no path
    prefix — callers add it).

    Weaker than :func:`manifest_shape_refusal` ON PURPOSE, and the difference is the whole
    argument: a score file cannot grant coverage, and canon's ``_load_scores`` happily reads one
    through a symlink, so a live symlink to a regular file stays readable here (canon parity).
    What is refused is what cannot be READ: a directory (``open`` would raise anyway, but say so
    in the same vocabulary as everything else), a dangling link (``os.path.exists`` follows, so
    the old test skipped it in silence), and above all a FIFO — ``open()`` on a reader-less FIFO
    BLOCKS FOREVER, so the shape has to be rejected BEFORE the open, not around it.
    """
    sh = classify_entry(path)
    if sh.kind in ("file", "unstatable"):
        # "unstatable" here means "nothing (readable) is there" — not a score source at all, and
        # never opened. Absence of a score file is normal (see the module docstring: missing data
        # never causes a deletion decision), so it is not an error.
        return None
    if sh.kind == "dangling":
        return "score file is a SYMLINK whose target does not resolve — unreadable, refused"
    if sh.kind == "dir":
        return "score file is a DIRECTORY, not a file — unreadable, refused"
    return ("score file is not a regular file (fifo/socket/device) — refused, never opened "
            "(`open()` on a reader-less FIFO BLOCKS FOREVER)")


class UnusableScoreSource(OSError):
    """A ``--roots`` entry named like a score file whose SHAPE makes it unreadable.

    Same value-not-thrown pattern as :class:`DanglingScanLink`, for the same reason: it travels
    the ``on_error`` channel so ``--verify-manifests`` refuses instead of silently treating the
    run dir as unscored.
    """

    def __init__(self, path: str, reason: str):
        OSError.__init__(self)
        self.filename = path
        self._msg = reason

    def __str__(self) -> str:
        return self._msg


# --- dangling-symlink refusal -----------------------------------------------------------------
# A symlink whose target does not resolve is the single most dangerous shape for this tool,
# because it is INVISIBLE to every symlink check that inspects `os.walk`'s `dirnames`:
# `os.walk` classifies entries with `DirEntry.is_dir()`, which FOLLOWS the link, so when the
# target's stat fails the entry is reported as a FILE. A dangling `experiments/<id>` directory
# link therefore never appears in `dirnames` at all, and a manifest that lived under it vanishes
# with zero signal. This is not hypothetical on this repo: `jax_ckpt_fast` / `jax_ckpt_easy`
# point at `/mnt/scratch/...`, which is not the current mount point, so both are
# dangling right now. A dangling link WITHDRAWS coverage — it is a REFUSAL, never a silent skip.

_DANGLING_LINK_WHY = ("its target does not resolve, so anything that lived under it — including "
                      "a manifest that grants coverage — is INVISIBLE to this scan (an unmounted "
                      "volume is the realistic cause). Fix or remove the link; a dangling link "
                      "WITHDRAWS coverage rather than silently granting none")


class DanglingScanLink(OSError):
    """A symlink under (or AT) a scan root whose target does not resolve.

    Constructed as a value, never thrown: it is handed to the walk's ``onerror`` channel so it
    lands in ``load_manifests``'s ``errors`` list and refuses the whole scan, exactly like an
    unreadable manifest does. ``__str__`` is overridden because ``OSError.__new__`` would
    otherwise read the two constructor arguments as ``(errno, strerror)`` and render this as
    ``[Errno None] None``.
    """

    def __init__(self, path: str, what: str = "BROKEN SYMLINK"):
        OSError.__init__(self)
        self.filename = path
        self._msg = f"{what} — REFUSED: {_DANGLING_LINK_WHY}"

    def __str__(self) -> str:
        return self._msg


def _is_dangling_link(path: str) -> bool:
    """True for a symlink that does not resolve to anything (broken target, or a link cycle,
    whose stat fails with ELOOP). Deliberately NOT true for a live symlink: those are already
    handled by the ``via_symlink`` refusal in ``_walk_scan_dirs``, and this fix must not weaken or
    duplicate that path.

    Expressed through :func:`classify_entry` so there is exactly ONE lstat-based classifier in this
    file: "dangling" is a shape in the table, not a separate ad-hoc test that could drift from it.
    """
    sh = classify_entry(path)
    return sh.is_link and sh.kind == "dangling"


def _unresolvable_link_on_path(path: str) -> str | None:
    """The symlink component that makes ``path`` unusable as a directory, or ``None``.

    Distinguishes "a plain directory that simply is not there" (fine — a host that has no
    ``experiments/`` is not a refusal) from "a directory made unreachable by a symlink that does
    not resolve to a directory" (a refusal). Walks up from ``path`` and stops at the first
    component that IS a real directory, so an existing symlinked ANCESTOR (a legitimately
    relocated repo) never false-positives.
    """
    p = os.path.abspath(path)
    while True:
        if os.path.islink(p) and not os.path.isdir(p):
            return p
        parent = os.path.dirname(p)
        if parent == p or os.path.isdir(p):
            return None
        p = parent


_LIVE_LINK_NON_DIR_WHY = (
    "it is a SYMLINK whose target EXISTS but is not a directory, so nothing can be scanned under "
    "it and whatever the intended directory would have granted is WITHDRAWN. This is NOT the "
    "broken-link case and is deliberately worded so it cannot be mistaken for one: the target is "
    "present, so there is no missing mount to restore — the link points at the wrong KIND of "
    "thing, and the fix is to repoint it")


class UnusableScanLink(OSError):
    """A symlink that makes a scan dir / checkpoint root unusable AS A DIRECTORY while its target
    nevertheless RESOLVES (``link -> regular file``, ``link -> fifo``, ``link -> link -> file``).

    Split out of :class:`DanglingScanLink` because that class's message was FACTUALLY FALSE for
    this shape: ``_unresolvable_link_on_path`` returns any symlink that is not a directory, so a
    LIVE link to a regular file refused (correctly) with "BROKEN SYMLINK … its target does not
    resolve … an unmounted volume is the realistic cause". On a workstation where a genuinely
    dangling ``jax_ckpt_fast`` is an accepted residual, that sends the operator to chase a mount
    that is fine. The REFUSAL is right and unchanged; only the diagnosis moves.

    Value, never thrown — same ``on_error`` channel discipline as :class:`DanglingScanLink`.
    """

    _FOUND = {"file": "a REGULAR FILE", "dir": "a DIRECTORY", "other": "a fifo/socket/device",
              "unstatable": "not statable"}

    def __init__(self, path: str, what: str, kind: str):
        OSError.__init__(self)
        self.filename = path
        found = self._FOUND.get(kind, f"a {kind}")
        self._msg = (f"{what} — REFUSED: {_LIVE_LINK_NON_DIR_WHY}. Its target resolves to "
                     f"{found}.")

    def __str__(self) -> str:
        return self._msg


def _unusable_scan_link_error(bad_link: str, dangling_what: str, live_what: str) -> OSError:
    """Pick the ACCURATE diagnosis for a symlink :func:`_unresolvable_link_on_path` flagged.

    That finder answers "is this path unusable as a directory because of a symlink?", which is
    True for a dangling link AND for a live link to a non-directory — two different facts that had
    one message between them. One ``lstat``-based :func:`classify_entry` (never a second ad-hoc
    test) decides which.
    """
    kind = classify_entry(bad_link).kind
    if kind == "dangling":
        return DanglingScanLink(bad_link, dangling_what)
    return UnusableScanLink(bad_link, live_what, kind)


_WALK_PRUNED_DIRNAMES = (".git", "__pycache__")
# THE DESCENT FILTER, NAMED. `_walk_scan_dirs` below does NOT descend into a subdirectory with one
# of these names (`if d in (".git", "__pycache__")`, left byte-identical to HEAD deliberately —
# this row does not rewrite it), so a directory whose path from a scan root crosses one of these
# names is NOT reached by that root's walk, even though it lies underneath it. That is invisible to
# anything reasoning about the tree by path prefix alone, and `_dedupe_nested_scan_dirs` is exactly
# such a reasoner: it may collapse an operator-named scan dir into an ancestor ONLY when the
# ancestor's walk really reaches it. Consulted there through this name, and pinned against drift by
# `test_the_dedup_prune_list_mirrors_the_real_walk` in the 881G suite — if the literal below ever
# gains or loses an entry, that test fails rather than the collapse silently over-reaching.


def _walk_scan_dirs(scan_dir: str, on_error):
    """``os.walk`` that never silently omits a symlinked subdirectory, and marks a manifest
    reached only by crossing a symlink (including a symlinked SCAN ROOT itself).

    Plain ``os.walk(sd, followlinks=False)`` neither descends a symlinked subdirectory NOR
    calls ``onerror`` for it — a symlinked ``experiments/<id>`` would make that experiment's
    entire manifest invisible with ZERO signal, which for coverage that gates an irreversible
    deleter reads as "nothing to protect". So symlinked directories ARE followed here, but each
    real directory is walked at most once (keyed by ``realpath``), which is what makes a
    self-referential symlink cycle terminate instead of looping forever.

    Yields ``(dirpath, names, via_symlink)``, where ``names`` is EVERY entry ``os.walk`` listed in
    that directory — ``dirnames`` and ``filenames`` CONCATENATED, deliberately handed over
    UN-SPLIT. That is the root fix: the caller re-classifies each name with ``lstat``
    (:func:`classify_entry`) instead of inheriting a split that was computed by following symlinks.
    Yielding only ``filenames`` is what made a live ``PROTECTED.txt -> some/dir`` link invisible to
    every manifest check, and yielding only ``dirnames`` to the symlink check is what made a
    dangling one invisible; the two blockers were the same mistake, so neither list is trusted now.

    ``via_symlink`` is True when this directory
    was only reachable by crossing a symlink — including when the SCAN ROOT ITSELF resolves
    elsewhere (``--experiments-dir`` pointed directly at a symlink): ``os.walk`` always lists
    the top path it is handed even when that path is a link, so treating only DISCOVERED
    symlinks as ``via_symlink`` would leave rooting the scan at the symlink itself as a bypass
    of the whole check. ``load_manifests`` refuses to READ any manifest found this way — a
    manifest is the authoritative signal that can grant coverage, and it must live at a real,
    tracked path.

    A DANGLING symlink is reported through ``on_error`` instead: it cannot be followed, and
    ``os.walk`` does not even offer it in ``dirnames`` (see ``_is_dangling_link``), so the only
    honest thing to say about it is that this scan is INCOMPLETE.
    """
    visited: set[str] = set()
    _root_via_link = os.path.realpath(scan_dir) != os.path.abspath(scan_dir)
    queue: list[tuple[str, bool]] = [(scan_dir, _root_via_link)]
    while queue:
        top, via_link = queue.pop(0)
        top_rp = os.path.realpath(top)
        if top_rp in visited:
            continue
        for dirpath, dirnames, filenames in os.walk(top, followlinks=False, onerror=on_error):
            visited.add(os.path.realpath(dirpath))
            # EVERY entry, captured BEFORE any directory-vs-file decision is acted on. Whatever
            # else happens below, the caller sees every name in this directory.
            names = list(dirnames) + list(filenames)
            keep = []
            for d in dirnames:
                if d in (".git", "__pycache__"):
                    continue
                full = os.path.join(dirpath, d)
                # `classify_entry`, not `os.path.islink` + trust-the-bucket: the DESCENT decision
                # is made from lstat and the resolved target type, so a link is followed (and
                # MARKED) only when it really leads to a directory, and never merely because
                # `os.walk` said `is_dir()`.
                shape = classify_entry(full)
                if shape.is_link:
                    if shape.kind == "dir":
                        queue.append((full, True))
                    # a link in `dirnames` that does NOT resolve to a directory is not descended;
                    # it is still reported below (dangling) and, if it carries a manifest basename,
                    # refused by `manifest_shape_refusal` in `load_manifests`.
                    continue
                keep.append(d)
            dirnames[:] = keep
            # os.walk mis-files a dangling DIRECTORY link into `filenames` (is_dir() follows the
            # link and its stat fails), so the dirnames loop above can never see one. Scanning
            # `names` rather than `filenames` here means this no longer DEPENDS on that mis-filing
            # being in one particular direction.
            for fn in names:
                full = os.path.join(dirpath, fn)
                if _is_dangling_link(full):
                    on_error(DanglingScanLink(full))
            yield dirpath, names, via_link


class ManifestSet(NamedTuple):
    protect: set[str]                  # realpath'd, EXISTING protect entries
    covers: set[str]                   # realpath'd, EXISTING covers entries
    dangling: set[str]                 # realpath'd entries (protect OR covers) that DO NOT EXIST
    citations: dict[str, list[str]]    # realpath -> ["<kind>: <raw> <- <source>:<line>", ...]


class ManifestScanIncomplete(OSError):
    """A manifest under the scan dirs could not be trusted IN FULL — a symlinked manifest file
    or directory, an unreadable file, a malformed PROTECTED.txt line, or a duplicate SPEC.md
    block. The manifest set (``.partial``) is a SUBSET of the real one: never read as "this run
    dir really has no manifest" without surfacing why.
    """

    def __init__(self, errors: list[str], partial: "ManifestSet"):
        self.errors = list(errors)
        self.partial = partial
        head = "; ".join(self.errors[:3]) + (" …" if len(self.errors) > 3 else "")
        super().__init__(f"manifest scan INCOMPLETE: {len(self.errors)} problem(s) under the "
                         f"scan dirs ({head}) — the manifest set is a SUBSET of the real one")


_MANIFEST_ENCODING_WHY = (
    "a manifest is parsed as UTF-8 text, and a byte-order mark or an undecodable byte means the "
    "bytes on disk are not the encoding this parser reads — so every directive in the file is "
    "mis-read. A BOM is INVISIBLE in `cat` and in every editor, which is what makes this shape "
    "dangerous rather than merely wrong: a column-0 `protected:` header becomes "
    "`\\ufeffprotected:`, matches nothing, and the block it introduces silently grants nothing")

# The BOMs, LONGEST PREFIX FIRST — `\xff\xfe\x00\x00` (UTF-32LE) starts with `\xff\xfe` (UTF-16LE),
# so testing the short one first would mis-name the encoding in the refusal.
_MANIFEST_BOMS = ((b"\x00\x00\xfe\xff", "UTF-32BE"), (b"\xff\xfe\x00\x00", "UTF-32LE"),
                  (b"\xef\xbb\xbf", "UTF-8"), (b"\xfe\xff", "UTF-16BE"), (b"\xff\xfe", "UTF-16LE"))


def _read_manifest_text(path: str) -> tuple[str | None, str | None]:
    """``(text, None)`` if ``path`` reads as UTF-8 manifest text; ``(None, refusal)`` otherwise.

    Read as BYTES and decoded STRICTLY, deliberately replacing ``open(path, errors="replace")``.
    ``errors="replace"`` CANNOT FAIL, and that was the hole: a UTF-16 manifest decoded to
    ``\\ufffd\\ufffdp\\x00r\\x00o\\x00…``, matched no header, and reported "1 manifest file(s) read
    / 0 protect / verify-manifests: OK" at exit 0 — byte-identical to a SPEC.md that genuinely
    grants nothing, on a file that declares a block. Same for a UTF-8 BOM, which is invisible on
    inspection. Both are now refusals, in the same fail-safe direction as every other shape check in
    this section: an input this tool cannot read must never be indistinguishable from an input that
    says nothing. Verified free on the real corpus (0 of 56 manifest files carries a BOM, a NUL
    byte, or an invalid UTF-8 sequence), so nothing that exists today starts refusing.
    """
    try:
        with open(path, "rb") as fh:
            data = fh.read()
    except OSError as e:
        return None, f"{path}: {e}"
    for bom, name in _MANIFEST_BOMS:
        if data.startswith(bom):
            return None, (f"{path}: manifest begins with a {name} BYTE-ORDER MARK — REFUSED, not "
                          f"parsed; {_MANIFEST_ENCODING_WHY}. Re-save it as UTF-8 with no BOM.")
    if b"\x00" in data:
        # UTF-16/32 WITHOUT a BOM: the NUL padding bytes are each valid UTF-8, so a strict decode
        # succeeds and only this test can tell the file apart from text.
        return None, (f"{path}: manifest contains a NUL byte — REFUSED, not parsed (it is binary, "
                      f"or UTF-16/UTF-32 with no byte-order mark); {_MANIFEST_ENCODING_WHY}.")
    try:
        return data.decode("utf-8"), None
    except UnicodeDecodeError as e:
        return None, (f"{path}: manifest is not valid UTF-8 ({e}) — REFUSED, not parsed; "
                      f"{_MANIFEST_ENCODING_WHY}.")


def load_manifests(scan_dirs: list[str], stats: dict | None = None) -> ManifestSet:
    """Scan ``scan_dirs`` (default: ``experiments``) for ``PROTECTED.txt`` / ``SPEC.md``
    manifests and CLASSIFY every directive they contain.

    Raises :class:`ManifestScanIncomplete` when ANY manifest could not be fully trusted:

    * a manifest BASENAME that is anything other than a real, regular, non-symlink file — a
      symlink to a file, a symlink to a DIRECTORY, a symlink to a dangling target or through a
      cycle, a real directory, a fifo/socket/device — is REFUSED, not read, whichever of
      ``os.walk``'s lists it appeared in (see the shape table above ``classify_entry`` and
      ``manifest_shape_refusal``); likewise a manifest reached only by crossing a symlinked
      DIRECTORY, including a symlinked scan root (see ``_walk_scan_dirs``). A manifest is the input
      that can grant coverage to an irreversible deleter, and its target may be untracked or
      outside the tree entirely;
    * an OS-level read error (permissions, race);
    * ANY malformed line / duplicate SPEC block in the file (scope items 1-3) — the WHOLE
      file's directives are withdrawn, not just the offending line. A partially-trustworthy
      manifest is not trustworthy: a malformed line is reported (never silently dropped) and it
      poisons every directive in that file, because a manifest author who mis-formatted one
      line has demonstrated the file was not reviewed the way the protection model requires.

    A well-formed entry naming a path that DOES NOT EXIST is not an incompleteness (the manifest
    was read in full and says something coherent) — it is a DANGLING entry, classified into
    ``ManifestSet.dangling`` rather than ``.protect``/``.covers``, and reported immediately
    (D45 planner note 3): it protects nothing, and withdraws coverage from any run dir it names
    or lies under, in the fail-safe direction, never silently un-refused by a typo.

    ``stats`` (optional, an out-param) is filled with how much was actually LOOKED AT:
    ``scan_dirs_visited`` / ``dirs_walked`` / ``manifest_files_read`` / ``manifest_files_refused``.
    Without it the caller's happy-path output ("0 protect, 0 covers, 0 dangling") cannot
    distinguish "56 manifests read, none of them grants coverage" — the real state of this repo —
    from "the scan read nothing at all", which are the same three lines but opposite situations.
    It is filled even when this function goes on to RAISE, so a refusal reports its scope too.
    """
    protect: set[str] = set()
    covers: set[str] = set()
    dangling: set[str] = set()
    citations: dict[str, list[str]] = {}
    errors: list[str] = []
    n_scan_dirs = n_dirs_walked = n_read = n_refused = 0

    def _walk_error(exc: OSError) -> None:
        # TASK_881S scope item 3 (D50): a dangling symlink named like a manifest basename is ALSO
        # caught, more specifically, a few lines below in this same walk, by
        # `manifest_shape_refusal` on the identical `fp` (same underlying `classify_entry` state —
        # both are `lstat`-derived, so they can never disagree). Reporting it here TOO would double
        # one real defect into two problem-count entries, exactly what an alert keying on that count
        # must not see. Every OTHER dangling link (not a manifest basename) has no other detector
        # in this file and is still reported here, unchanged.
        if (isinstance(exc, DanglingScanLink)
                and os.path.basename(getattr(exc, "filename", "")) in MANIFEST_BASENAMES):
            return
        errors.append(f"{getattr(exc, 'filename', '?')}: {exc}")

    def _classify(entry: ManifestEntry) -> None:
        rp = os.path.realpath(entry.path)
        bucket = protect if entry.kind == "protect" else covers
        exists = os.path.exists(rp)
        (bucket if exists else dangling).add(rp)
        citations.setdefault(rp, []).append(f"{entry.kind}: {entry.path} <- "
                                            f"{entry.source}:{entry.lineno}")
        if not exists:
            print(f"[ckpt_gc] MANIFEST ENTRY NAMES NOTHING: {entry.source}:{entry.lineno}: "
                  f"'{entry.kind}: {entry.path}' resolves to {rp}, which does not exist — it "
                  f"protects nothing, and withdraws coverage from any run dir it names or lies "
                  f"under (fail-safe direction). Fix the path or delete the line.",
                  file=sys.stderr)

    for sd in scan_dirs:
        if not os.path.isdir(sd):
            # A scan dir that is simply absent is not a refusal (a host with no `experiments/`
            # has nothing to say). A scan dir made unusable by a symlink that does not resolve
            # IS: `os.path.isdir` follows links, so `--experiments-dir` pointed at a dangling
            # link (or at a path underneath one) would otherwise be skipped here in exactly the
            # same silent way `os.walk` skips one it finds mid-tree.
            bad_link = _unresolvable_link_on_path(sd)
            if bad_link is not None:
                exc = _unusable_scan_link_error(
                    bad_link, "SCAN DIR UNUSABLE (BROKEN SYMLINK)",
                    "SCAN DIR UNUSABLE (SYMLINK TO A NON-DIRECTORY)")
                where = "" if os.path.abspath(sd) == bad_link else f" [bad component: {bad_link}]"
                errors.append(f"{sd}: {exc}{where}")
                continue
            # ...and a scan dir that EXISTS but is the WRONG SHAPE is a third case, distinct from
            # both: `--experiments-dir` pointed at a REGULAR FILE (or a fifo/socket/device) has no
            # unresolvable symlink component, so `_unresolvable_link_on_path` returns None and this
            # used to fall through to a silent `continue` — output byte-identical to a clean scan,
            # exit 0. That is a WITHDRAWAL of whatever the real directory would have granted, and
            # every other input in this tool shape-checks instead (manifests: ten shapes;
            # `--protect-file`: missing/symlink/dir/fifo; score files: torn/dir/fifo; `--roots`:
            # dangling link). Only NON-symlink shapes are judged here: a symlink is already the
            # branch above's business, and reaching into it would blur two diagnoses the way the
            # earlier cycles did. An ABSENT path ("unstatable") stays a silent no-op — a host with
            # no `experiments/` has nothing to say, which is a stated control of this row
            # (test_absent_scan_root_is_not_a_refusal); "not there" and "there but not a
            # directory" are different claims and get different answers.
            sd_shape = classify_entry(sd)
            if not sd_shape.is_link and sd_shape.kind in ("file", "other"):
                found = "a REGULAR FILE" if sd_shape.kind == "file" else "a fifo/socket/device"
                errors.append(
                    f"{sd}: --experiments-dir is NOT A DIRECTORY ({found}) — REFUSED, nothing "
                    f"scanned under it. A named-but-unscannable scan dir must not be "
                    f"indistinguishable from a clean scan: it would report zero manifests at "
                    f"exit 0 while WITHDRAWING whatever coverage the intended directory grants.")
            continue
        n_scan_dirs += 1
        for dirpath, names, via_symlink in _walk_scan_dirs(sd, _walk_error):
            n_dirs_walked += 1
            # `names` is dirnames+filenames UN-SPLIT (see `_walk_scan_dirs`), and the basename test
            # below therefore runs BEFORE any directory-vs-file classification — the ordering IS
            # the fix. A name in MANIFEST_BASENAMES reaches a refusal path whenever it is not a
            # real, regular, non-symlink file, no matter which list `os.walk` put it in and no
            # matter what its target is.
            for fn in names:
                if fn not in MANIFEST_BASENAMES:
                    continue
                fp = os.path.join(dirpath, fn)
                if via_symlink:
                    n_refused += 1
                    errors.append(
                        f"{fp}: manifest reached only by crossing a SYMLINKED DIRECTORY — "
                        f"REFUSED, not read (a manifest grants coverage to an irreversible "
                        f"deleter and must live at a real, tracked path)")
                    continue
                shape_refusal = manifest_shape_refusal(fp)
                if shape_refusal is not None:
                    n_refused += 1
                    errors.append(shape_refusal)
                    continue
                text, read_refusal = _read_manifest_text(fp)
                if read_refusal is not None:
                    n_refused += 1
                    errors.append(read_refusal)
                    continue
                n_read += 1
                ents, _spans, bad = (parse_protected_txt(text, fp) if fn == "PROTECTED.txt"
                                     else parse_spec_manifest(text, fp))
                if bad:
                    for b in bad:
                        named = f" (names {b.hint_path!r})" if b.hint_path else ""
                        errors.append(f"{b.source}:{b.lineno}: malformed manifest content "
                                      f"{b.raw!r}{named} — refused; none of this file's "
                                      f"directives are trusted")
                    continue  # scope item 2: a malformed line WITHDRAWS the whole file
                for e in ents:
                    _classify(e)

    if stats is not None:
        # Written BEFORE the raise below, deliberately: a refusal has to be able to say how much of
        # the tree it actually got through, or the operator cannot tell a refusal that saw
        # everything from one that stopped at the first scan dir.
        stats.update(scan_dirs_requested=len(scan_dirs), scan_dirs_visited=n_scan_dirs,
                     dirs_walked=n_dirs_walked, manifest_files_read=n_read,
                     manifest_files_refused=n_refused)
    result = ManifestSet(protect, covers, dangling, citations)
    if errors:
        raise ManifestScanIncomplete(errors, result)
    return result


def load_protect_file(path: str | None) -> tuple[set[str], str | None]:
    """Read ``--protect-file`` into a set of realpath'd entries -> ``(paths, error)``.

    A missing, symlinked, or directory ``--protect-file`` is a REFUSAL (``error`` is not
    ``None``), never silently read as an empty set — a named-but-unusable protect file must not
    be indistinguishable from "nothing to protect". A symlink gets the same treatment
    ``load_manifests`` gives a symlinked manifest, for the same reason: it is explicit, tracked
    input that can grant coverage, and its target may not be.
    """
    if path is None:
        return set(), None
    if os.path.islink(path):
        return set(), (f"{path}: --protect-file is a SYMLINK — refused, not read (its target "
                       f"may be untracked or outside the tree)")
    # Same lstat-first discipline as the manifest side: a shape that cannot be READ is rejected
    # BEFORE the `open()` below. A DIRECTORY is left to `IsADirectoryError` (its message is already
    # exact), but a fifo/socket must never reach `open()` at all — a reader-less FIFO blocks forever
    # and the operator sees a hung cron job instead of a refusal.
    if classify_entry(path).kind == "other":
        return set(), (f"{path}: --protect-file is not a regular file (fifo/socket/device) — "
                       f"refused, never opened (`open()` on a reader-less FIFO BLOCKS FOREVER)")
    try:
        with open(path) as fh:
            # 881G ingestion NIT (D53): `not l.strip().startswith("#")`, not `not
            # l.startswith("#")` — the RAW-line check let an INDENTED comment (`  # comment`)
            # through, because the string that starts with `#` is the STRIPPED one; the raw one
            # starts with whitespace. That mismatch realpath'd the comment text itself into a
            # bogus protected-path entry. Blank lines already strip first for the same check.
            return ({os.path.realpath(l.strip()) for l in fh
                     if l.strip() and not l.strip().startswith("#")}, None)
    except IsADirectoryError:
        return set(), f"{path}: --protect-file is a DIRECTORY, not a file — refused, not read"
    except OSError as e:
        return set(), f"{path}: --protect-file could not be read ({e})"


class ScoreReadIncomplete(OSError):
    """A score source (``checkpoints_index.json`` / ``scores.jsonl``) could not be read IN
    FULL, so any resolved scores are a SUBSET. ``scores.jsonl`` is appended by potentially
    concurrent writers with no file locking, so a torn row is a realistic outcome, not a
    corruption exotic — this is the "torn ``scores.jsonl``" class in gate (i).
    """

    def __init__(self, run_dir: str, errors: list[str]):
        self.run_dir = run_dir
        self.errors = list(errors)
        head = "; ".join(self.errors[:3]) + (" …" if len(self.errors) > 3 else "")
        super().__init__(f"score read INCOMPLETE for {run_dir}: {len(self.errors)} unreadable "
                         f"record(s) ({head}) — the resolved scores are a SUBSET")


def _load_scores_strict(run_dir: str, score_regex: str | None, log_glob: str) -> dict[int, float]:
    """Ingestion-only, STRICT sibling of ``_load_scores`` above: same three sources
    (``checkpoints_index.json``, ``scores.jsonl``, and a log matched by ``--score-regex``), but a
    torn or unparseable record RAISES :class:`ScoreReadIncomplete` instead of being silently
    skipped — INCLUDING in the ``--score-regex`` branch, which until now kept canon's
    ``except Exception: continue`` and so contradicted this sentence (see the comment at that
    branch). A score source whose SHAPE makes it unreadable (directory, dangling link, fifo) is
    refused BEFORE it is opened, via :func:`_score_source_shape_reason`.

    Deliberately a SEPARATE function, not a modification of ``_load_scores``: the canon
    (``plan()`` / ``--apply``) invariant is bit-identical behaviour, and ``_load_scores``'s
    permissive skip-and-continue is exactly what the live deletion path still does. This
    stricter reader is reached ONLY from ``--verify-manifests`` (see ``_run_manifest_check``);
    wiring its refusal into an actual keep/delete decision is TASK_881D's, not this row's.
    """
    errors: list[str] = []
    out: dict[int, float] = {}
    idx = os.path.join(run_dir, "checkpoints_index.json")
    idx_shape = _score_source_shape_reason(idx)
    if idx_shape is not None:
        errors.append(f"{idx}: {idx_shape}")
    elif os.path.exists(idx):
        try:
            with open(idx) as fh:
                rows = json.load(fh)
            for r in rows if isinstance(rows, list) else rows.get("checkpoints", []):
                if r.get("epoch") is not None and r.get("score") is not None:
                    out[int(r["epoch"])] = float(r["score"])
        except Exception as e:
            errors.append(f"{idx}: {e}")
    sj = os.path.join(run_dir, "scores.jsonl")
    sj_shape = _score_source_shape_reason(sj)
    if sj_shape is not None:
        errors.append(f"{sj}: {sj_shape}")
    elif os.path.exists(sj):
        try:
            with open(sj) as fh:
                for lineno, line in enumerate(fh, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        out[int(d["epoch"])] = float(d["score"])
                    except Exception as e:
                        errors.append(f"{sj}:{lineno}: unparseable score row ({e})")
        except (OSError, ValueError) as e:
            errors.append(f"{sj}: {e}")
    if score_regex:
        import glob as _glob
        pat = re.compile(score_regex)
        for lf in _glob.glob(os.path.join(run_dir, log_glob)) + _glob.glob(log_glob):
            lf_shape = _score_source_shape_reason(lf)
            if lf_shape is not None:
                errors.append(f"{lf}: {lf_shape}")
                continue
            try:
                with open(lf, errors="replace") as fh:
                    for lineno, line in enumerate(fh, 1):
                        m = pat.search(line)
                        if not m:
                            continue
                        try:
                            out[int(m.group("epoch"))] = float(m.group("score"))
                        except (IndexError, TypeError, ValueError, re.error) as e:
                            # THE DOCSTRING'S PROMISE, NOW KEPT. This branch used to be
                            # `except Exception: continue` — a matched record whose `epoch`/`score`
                            # groups did not coerce was silently DROPPED, exactly like canon's
                            # permissive `_load_scores`, while the docstring above claimed this
                            # reader RAISES on precisely that. A regex that matches a line and then
                            # yields no usable epoch/score is either a torn log line or a
                            # misconfigured `--score-regex`; both are "the resolved scores are a
                            # SUBSET", which is what `ScoreReadIncomplete` exists to say. Safe to
                            # make strict because this reader is reachable ONLY from
                            # `_run_manifest_check` (`--verify-manifests`), never from canon.
                            errors.append(f"{lf}:{lineno}: --score-regex matched but its "
                                          f"epoch/score groups did not coerce ({e}): "
                                          f"{line.strip()!r}")
            except OSError as e:
                errors.append(f"{lf}: {e}")
    if errors:
        raise ScoreReadIncomplete(run_dir, errors)
    return out


class UnusableCheckpointRoot(OSError):
    """A ``--roots`` value that cannot be enumerated for score files AT ALL: an EMPTY/blank string,
    or an existing path that is not a directory.

    The score half of ``--verify-manifests`` had only the dangling-link branch below, so every
    OTHER wrong shape of ``--roots`` fell through the bare ``if not os.path.isdir(root): continue``
    and the whole score check verified NOTHING at exit 0 — measured on a tree whose score file was
    provably torn: ``--roots runs`` refused ("score read INCOMPLETE"), ``--roots afile`` and
    ``--roots ''`` both reported "verify-manifests: OK". A typo'd or unset root returned a full
    green, and unlike the manifest side there was not even a weak signal, because no count of score
    dirs was printed: "OK with zero score dirs examined" was byte-identical to "every score file
    clean". This is the same class ``load_manifests`` already refuses for ``--experiments-dir``;
    covering one enumeration of two is what left the hole.

    An EMPTY value is called out separately because ``--roots "$CKPT_ROOTS"`` with the variable
    unset is the canonical cron/CI spelling of it, and `os.path.isdir("")` is False with no symlink
    component and no ``lstat``-able shape to report.

    Value, never thrown: it travels the ``on_error`` channel like every other refusal here. Scoped
    to the ingestion path — canon's ``main()`` still filters ``--roots`` with its own
    ``[r for r in a.roots if os.path.isdir(r)]`` line, untouched, so ``--apply`` behaviour is
    bit-identical.
    """

    def __init__(self, path: str, reason: str):
        OSError.__init__(self)
        self.filename = path
        self._msg = reason

    def __str__(self) -> str:
        return self._msg


def _find_score_dirs(roots: list[str], on_error=None, stats: dict | None = None) -> list[str]:
    """Directories under ``roots`` that carry a score-bearing file. Ingestion-only enumeration:
    it does NOT classify anything as a checkpoint or a run (that is ``collect()``'s /
    TASK_881D's job) — it only says where there is a score FILE to try to read, so
    ``--verify-manifests`` can exercise the torn-score-file guard without any of the
    discovery/keep/delete machinery above.

    ``on_error`` (optional) is called with a :class:`DanglingScanLink` for every symlink under
    ``roots`` whose target does not resolve, including a ROOT that is itself such a link. This
    is the live shape on this repo: the default root ``jax_ckpt_fast`` points into
    ``/mnt/scratch``, which is not the current mount point, so the whole root would
    otherwise be dropped by the ``isdir`` test below without a word. Every OTHER unusable shape of
    a root — blank/empty string, a regular file, a fifo, a live symlink to either — is refused
    through the same channel as :class:`UnusableCheckpointRoot`; see that class for the measurement.

    ``stats`` (optional, an out-param) mirrors ``load_manifests``'s: ``roots_requested`` /
    ``roots_visited`` / ``dirs_walked`` / ``score_dirs_found``. Without it "verify-manifests: OK"
    with ZERO score dirs examined is byte-identical to "every score file is clean".
    """
    found = []
    n_roots = n_dirs = 0
    for root in roots:
        if not os.path.isdir(root):
            if on_error is not None:
                bad_link = _unresolvable_link_on_path(root)
                if bad_link is not None:
                    on_error(_unusable_scan_link_error(
                        bad_link, "CHECKPOINT ROOT UNUSABLE (BROKEN SYMLINK)",
                        "CHECKPOINT ROOT UNUSABLE (SYMLINK TO A NON-DIRECTORY)"))
                elif not root.strip():
                    on_error(UnusableCheckpointRoot(
                        root, f"--roots was given an EMPTY/BLANK value ({root!r}) — REFUSED, no "
                              f"score file enumerated under it. This is `--roots \"$VAR\"` with "
                              f"VAR unset, and it must not be indistinguishable from a scan in "
                              f"which every score file was read and found clean."))
                else:
                    # Only NON-symlink shapes here: a symlink is the branch above's business, and
                    # reaching into it would blur two diagnoses (the same conflation the earlier
                    # cycles died of). An ABSENT root stays a silent no-op, matching the scan-dir
                    # side: "not there" and "there but not a directory" are different claims.
                    root_shape = classify_entry(root)
                    if not root_shape.is_link and root_shape.kind in ("file", "other"):
                        found_what = ("a REGULAR FILE" if root_shape.kind == "file"
                                      else "a fifo/socket/device")
                        on_error(UnusableCheckpointRoot(
                            root, f"--roots entry is NOT A DIRECTORY ({found_what}) — REFUSED, no "
                                  f"score file enumerated under it. A named-but-unenumerable root "
                                  f"must not be indistinguishable from a clean score scan: it "
                                  f"verifies NOTHING while reporting success."))
            continue
        n_roots += 1
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            n_dirs += 1
            # Same root discipline as `_walk_scan_dirs`: capture EVERY entry first, then classify
            # each one with lstat. The old `"scores.jsonl" in filenames` test inherited `os.walk`'s
            # link-following split, so a LIVE symlink -> DIRECTORY named `scores.jsonl` landed in
            # `dirnames` and the run dir silently stopped being score-bearing — the same blind spot
            # as the manifest blocker, one layer down.
            names = list(dirnames) + list(filenames)
            dirnames[:] = [d for d in dirnames
                           if not classify_entry(os.path.join(dirpath, d)).is_link]
            has_score = False
            for fn in names:
                # a dangling DIRECTORY link is mis-filed as a file by os.walk, so the dirnames
                # filter above can never see one; scanning `names` no longer depends on which way
                # the mis-filing goes.
                full = os.path.join(dirpath, fn)
                if on_error is not None and _is_dangling_link(full):
                    on_error(DanglingScanLink(full))
                if fn not in SCORE_BASENAMES:
                    continue
                reason = _score_source_shape_reason(full)
                if reason is not None:
                    # Refused, not silently un-found: a name that says "scores live here" and
                    # cannot be read as a score file must not resolve to "this run is unscored".
                    if on_error is not None:
                        on_error(UnusableScoreSource(full, reason))
                    continue
                has_score = True
            if has_score:
                found.append(dirpath)
    if stats is not None:
        stats.update(roots_requested=len(roots), roots_visited=n_roots, dirs_walked=n_dirs,
                     score_dirs_found=len(found))
    return sorted(found)


# ======================================================================================
# TASK_881S — SCORE-SOURCE ENUMERATION (D50, seeded by TASK_881I's 3-cycle halt)
# ======================================================================================
# `_find_score_dirs` above enumerates a directory only when it holds a `SCORE_BASENAMES` file
# (`scores.jsonl` / `checkpoints_index.json`). The module docstring names a THIRD score source,
# consulted in the SAME order — a run log matched by `--score-regex`/`--log-glob` — and that
# source was never enumerated at all: a run scored PURELY through it was never handed to
# `_load_scores_strict`, so that reader's log-file shape guard (`_score_source_shape_reason`, in
# its `score_regex` branch) was dead code for exactly that input class. Measured (D50):
# `--verify-manifests --score-regex ... --log-glob <FIFO>` with no `scores.jsonl` anywhere printed
# "verify-manifests: OK" at exit 0, while the identical flags on the real (non-`--verify-manifests`)
# path — which has no shape guard at all — hang forever on `open()`. So the pre-flight guarantee
# `--help` advertises for `--verify-manifests` did not hold for this input class.
#
# `_find_log_glob_score_dirs` below is a SEPARATE enumeration, not a modification of
# `_find_score_dirs` (kept byte-identical — TASK_881S gate (iv)): it only decides WHERE to look,
# handing every directory it finds to the SAME `_load_scores_strict` the co-located-score path
# already uses, so the shape check itself is never duplicated or re-derived here.

def _find_log_glob_score_dirs(roots: list[str], score_regex: str | None, log_glob: str,
                              already_found) -> list[str]:
    """Directories under ``roots`` that carry a ``--log-glob`` match, found INDEPENDENTLY of
    :func:`_find_score_dirs`'s ``SCORE_BASENAMES`` gate (D50; see the section banner above).

    This function never opens anything and never classifies a shape: it only decides WHICH
    directories to hand to :func:`_load_scores_strict`, which re-globs the identical pattern and
    does its own, already-correct shape check (FIFO/socket/directory/dangling-symlink — the same
    guard the co-located-score path already exercises) before ever calling ``open()``. Deciding
    the shape a second time here would risk a second, differently-worded refusal for the same
    file — the failure mode scope item 3 fixes elsewhere for manifests, not reintroduced here.

    ``already_found`` is the caller's own :func:`_find_score_dirs` result over these same
    ``roots``: skipped here so a directory that has BOTH a real score file and a matching log is
    not handed to ``_load_scores_strict`` twice, which would print any refusal twice. Root-shape
    refusals (a dangling/blank/non-directory ``--roots`` entry) are likewise not re-reported here
    — the caller's ``_find_score_dirs`` call, over the identical ``roots`` with ``on_error``
    wired, is already authoritative for root shape.

    Inert (returns ``[]``) when ``score_regex`` is not given: canon's own log-glob branch is dead
    without it too (see the module docstring), so there is nothing this input class could hide.

    ONE PHYSICAL FILE YIELDS ONE CANDIDATE, however the pattern is spelled (TASK_881S scope item
    3). The property that matters is not "the pattern is absolute" — it is DIRECTORY-INDEPENDENCE:
    a pattern that resolves to the SAME physical file no matter which ``dirpath`` it is joined to.
    An absolute pattern has that property because ``os.path.join(dirpath, log_glob)`` DISCARDS
    ``dirpath``; a relative ``..``-traversal pattern has it too, by arithmetic instead of by
    syntax (``runs/r1/../shared.log`` and ``runs/r2/../shared.log`` are the same file), and so
    does any pair of spellings that meet at a symlink. An ``os.path.isabs`` test — which is what
    stood here — is a pattern-match on ONE spelling of the property and misses the others:
    measured on a 5-sibling-run tree with a single shared FIFO one level up and
    ``--log-glob '../shared.log'``, it enumerated FIVE "additional score dir(s)" and printed FIVE
    ``VERIFY-MANIFESTS REFUSAL`` lines for ONE file — a problem count an alert keys on, inflated
    5x by one defect and growing with the size of the tree.

    So the property is DETECTED, never spelled: every glob match is normalised with
    ``os.path.realpath`` and a directory is enumerated only when it contributes a resolved path no
    earlier candidate already covers. Directory-independent patterns collapse to one candidate by
    construction, and genuinely distinct files (the ordinary ``--log-glob '*.log'`` case, one real
    log per run dir) stay distinct — two real defects are never folded into one.

    The claim set is SEEDED before the walk with everything already covered elsewhere in the same
    invocation, so the dedup holds ACROSS passes and not merely within this one:

    * the bare, CWD-relative glob term (:func:`_bare_log_glob_matches`), which
      :func:`_load_bare_log_glob_scores_strict` reads exactly once per invocation. This is what
      keeps an absolute pattern reported by that once-per-invocation checker, with its
      source-naming message, exactly as before — the absolute case is now a CONSEQUENCE of the
      invariant rather than a special case beside it; and
    * every ``already_found`` directory's own matches — the caller has already handed those dirs
      to ``_load_scores_strict``, which globs the identical term for them, so a file shared with a
      co-located-score dir is already reported and must not be reported again from here.
    """
    if not score_regex:
        return []
    import glob as _glob

    def _resolved(dirpath: str) -> set[str]:
        # Enumeration only — `realpath` never opens anything, and a shape that cannot be opened
        # (fifo/dangling/directory) still normalises fine; the SHAPE verdict stays
        # `_load_scores_strict`'s, exactly as the paragraph above promises.
        return {os.path.realpath(m) for m in _glob.glob(os.path.join(dirpath, log_glob))}

    claimed: set[str] = {os.path.realpath(m)
                         for m in _bare_log_glob_matches(score_regex, log_glob)}
    for sd in sorted(already_found):
        claimed |= _resolved(sd)
    found = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, _filenames in os.walk(root, followlinks=False):
            # Same non-descent-through-symlinks discipline as `_find_score_dirs`: a directory
            # reached only by crossing a symlink is not walked into looking for logs either.
            # SORTED so that which of several colliding directories wins the single report is
            # deterministic, rather than an artefact of `os.scandir` order.
            dirnames[:] = sorted(d for d in dirnames
                                 if not classify_entry(os.path.join(dirpath, d)).is_link)
            if dirpath in already_found:
                continue
            resolved = _resolved(dirpath)
            if resolved - claimed:
                found.append(dirpath)
            claimed |= resolved
    return sorted(set(found))


# --- canon's SECOND log-glob term: the bare, CWD-relative one ---------------------------------
# `_load_scores_strict` (like canon's `_load_scores`) globs TWO terms for the `--score-regex`
# source, per run dir:
#
#     _glob.glob(os.path.join(run_dir, log_glob)) + _glob.glob(log_glob)
#                └── term 1: under the run dir ──┘   └─ term 2: BARE, relative to CWD ─┘
#
# Term 2 does not depend on `run_dir` at all: it is a single, GLOBAL score source, resolved once
# against the process's CWD. `_find_log_glob_score_dirs` above mirrors only term 1, so a score
# source reachable ONLY through term 2 was still never enumerated and `_load_scores_strict` was
# never called on it — D50 one layer further down, and the same false green: a tree of
# `runs/r1/snapshot_ep1/` plus a FIFO `./topfile.log`, run with `--log-glob topfile.log`, printed
# "0 additional score dir(s) found, 0 examined" / "verify-manifests: OK" / exit 0, while the same
# flags on the real (non-`--verify-manifests`) path hang forever on `open()`.
#
# So term 2 is probed here ONCE PER INVOCATION — not once per directory, which is what would
# multiply one bad file into one refusal per walked dir.

# A `run_dir` that provably holds NO co-located score file, used to isolate term 2 from term 1.
# `os.devnull` is a character device, so `os.path.join(os.devnull, "scores.jsonl")` and
# `os.path.join(os.devnull, "checkpoints_index.json")` can never exist (`lstat` -> ENOTDIR ->
# `classify_entry` "unstatable" -> `_score_source_shape_reason` None -> `os.path.exists` False ->
# the branch is skipped), and `glob("/dev/null/<relative pattern>")` is empty for the same reason.
# Passing it as `run_dir` therefore leaves the bare term as the ONLY source that can contribute.
_BARE_LOG_GLOB_PROBE_DIR = os.devnull


def _bare_log_glob_matches(score_regex: str | None, log_glob: str) -> list[str]:
    """What canon's bare, CWD-relative ``_glob.glob(log_glob)`` term resolves to, right now.

    Enumeration only — nothing is opened or classified here (same division of labour as
    :func:`_find_log_glob_score_dirs`). Exists so ``--verify-manifests`` can PRINT how many files
    this source contributed: "0 file(s) matched" and "every matched file read clean" must not be
    the same line, which is the reporting discipline the rest of this mode already follows.

    Inert (``[]``) without ``score_regex``, exactly like canon's branch and like
    :func:`_find_log_glob_score_dirs`.
    """
    if not score_regex:
        return []
    import glob as _glob
    return sorted(_glob.glob(log_glob))


def _load_bare_log_glob_scores_strict(score_regex: str | None, log_glob: str) -> dict[int, float]:
    """Read the bare, CWD-relative ``--log-glob`` term through :func:`_load_scores_strict` itself,
    ONCE per invocation, so every shape guard and every strictness rule that reader already
    enforces covers this input class too.

    The reader is CALLED, never re-implemented: a hand-rolled second copy of the FIFO/dangling/
    directory shape check and of the ``--score-regex`` coercion check could silently drift from
    the one that actually runs on the real path, and "the verifier's mirror disagrees with the
    reader" is precisely the class of hole this row exists to close. ``_load_scores_strict`` is
    left byte-identical (TASK_881S gate (iv)); the isolation is done by the ``run_dir`` argument
    — see :data:`_BARE_LOG_GLOB_PROBE_DIR`.

    Two deliberate differences from a raw call, both in the REPORTING, never in what is checked:

    * the refusal names the SOURCE ("the bare `--log-glob …` term, matched relative to CWD …")
      instead of ``/dev/null``, which would be an unreadable diagnosis for an operator; and
    * duplicate error strings are collapsed. For an ABSOLUTE ``log_glob`` the reader's two glob
      terms resolve to the identical path, so it appends the identical message twice and reports
      "2 unreadable record(s)" for ONE file. That doubling is pre-existing behaviour of the reader
      (unchanged since the 881I ship, and reachable there through any co-located score dir), so it
      is not fixed HERE — but this row must not carry it into the count it is itself responsible
      for.
    """
    try:
        return _load_scores_strict(_BARE_LOG_GLOB_PROBE_DIR, score_regex, log_glob)
    except ScoreReadIncomplete as e:
        raise ScoreReadIncomplete(
            f"the bare `--log-glob {log_glob}` term (matched relative to CWD {os.getcwd()}, "
            f"independently of --roots — canon's reader globs it for every run)",
            list(dict.fromkeys(e.errors))) from None


def _walk_reaches(ancestor_rp: str, rp: str) -> bool:
    """Would a ``_walk_scan_dirs`` rooted at ``ancestor_rp`` actually REACH ``rp``?

    Path containment is NOT the same question, and that gap is what makes this function exist: the
    walk does not descend into :data:`_WALK_PRUNED_DIRNAMES`, so ``experiments/__pycache__`` lies
    UNDER ``experiments`` and is never visited from it. Both arguments must already be realpaths.
    """
    if rp == ancestor_rp:
        return True
    if not rp.startswith(ancestor_rp + os.sep):
        return False
    return not any(part in _WALK_PRUNED_DIRNAMES
                   for part in rp[len(ancestor_rp) + 1:].split(os.sep))


def _dedupe_nested_scan_dirs(scan_dirs: list[str]) -> tuple[list[str], list[str]]:
    """Collapse ``scan_dirs`` so a duplicate or NESTED ``--experiments-dir`` request walks each
    real subtree exactly once, instead of double-counting it (881G ingestion NIT, D53): two
    requested dirs that resolve (``os.path.realpath``) to the SAME directory, or where one
    resolves to a directory INSIDE another, name the same manifests twice — every stat
    :func:`load_manifests` reports (dirs walked, manifest files read) would otherwise double for
    no real defect, the same class of inflated count TASK_881S closed on the score-source side.

    Returns ``(kept, dropped)`` in the ORIGINAL order. A dropped dir is dropped ONLY when the
    surviving dir's walk is PROVEN to reach it against the walk's own name-based prune list
    (:func:`_walk_reaches`) — not against every possible I/O outcome, so "nothing goes un-scanned"
    holds only up to the first I/O error the walk hits (a non-listable intermediate directory
    still drops the nested dir's coverage; see the module docstring's "MANIFEST GRAMMAR" section
    for the measured case and why it is a loud refusal, not a silent one). That is
    ACCEPTED-DELIBERATELY, never a refusal, because a redundant request is an operator REDUNDANCY,
    not the operator ERROR a nonexistent path is (see the D53 refusal in
    :func:`_run_manifest_check`, which this function's output is never fed through — only EXISTING
    directories participate here; an absent one is that other check's business).

    TWO CONDITIONS GATE EVERY COLLAPSE, because an unconditional one is not a de-duplication, it is
    a SILENT WITHDRAWAL OF A PROTECTION — the dangerous direction this row's own gate names, and
    both holes were shipped by the first cut of this function:

    * ONLY REAL PATHS PARTICIPATE, as a candidate to drop AND as an ancestor to keep. A scan dir
      whose ``realpath`` differs from its ``abspath`` is reached by crossing a SYMLINK, and that is
      not a redundancy — it is the input ``_walk_scan_dirs``' ``_root_via_link`` turns into 881I's
      "manifest reached only by crossing a SYMLINKED DIRECTORY" REFUSAL. Collapsing it on realpath
      BEFORE :func:`load_manifests` ever runs ERASED that refusal: ``--experiments-dir experiments
      experiments_link`` (and the same shape with a link to a subdirectory INSIDE the tree) refused
      at exit 2 before this row and passed at exit 0 after it. A shipped refusal must not be
      weakened by a counting fix; a symlinked scan dir is now never collapsed and never absorbs
      another dir's scan.
    * THE SURVIVOR'S WALK MUST REACH THE DROPPED DIR (:func:`_walk_reaches`). Path containment
      alone is not reachability — ``_walk_scan_dirs`` prunes :data:`_WALK_PRUNED_DIRNAMES` from its
      descent, so ``--experiments-dir experiments experiments/__pycache__`` walked the second dir
      explicitly before this row (1 manifest read, 1 protect) and, once collapsed into
      ``experiments``, read NOTHING and printed "0 protect" at exit 0. Under 881D that silently
      makes a manifest-protected checkpoint deletable, which is the WITHDRAWAL direction, not the
      double-count this optimisation exists to fix. When the walk does not reach it, BOTH dirs are
      kept: a duplicated stat is a cosmetic defect, a dropped protection is not.
    """
    entries = [(os.path.abspath(d), os.path.realpath(d))
               for d in scan_dirs if os.path.isdir(d)]
    # Index -> realpath, for the REAL (non-symlinked) entries only; see the docstring.
    real = [(idx, rp) for idx, (ap, rp) in enumerate(entries) if ap == rp]
    redundant_idx: set[int] = set()
    for idx, rp in real:
        for other_idx, other_rp in real:
            if other_idx == idx:
                continue
            if rp == other_rp and other_idx > idx:
                continue    # an exact duplicate is absorbed by its FIRST occurrence, not a later
            if rp != other_rp and not rp.startswith(other_rp + os.sep):
                continue    # unrelated subtree
            if not _walk_reaches(other_rp, rp):
                continue    # nested BY PATH but not reached BY THE WALK -> keep both
            redundant_idx.add(idx)
            break
    kept: list[str] = []
    dropped: list[str] = []
    ex_idx = 0
    for d in scan_dirs:
        if os.path.isdir(d):
            (dropped if ex_idx in redundant_idx else kept).append(d)
            ex_idx += 1
        else:
            kept.append(d)
    return kept, dropped


def _run_manifest_check(a) -> int:
    """TASK_881I's CLI entry point (``--verify-manifests``): PARSE every manifest, protect-file,
    and score file this tool would otherwise consult, CLASSIFY what each one says, and REFUSE
    loudly on anything malformed — without discovering a single checkpoint, planning a single
    deletion, or resolving a single checkpoint root (``collect()`` / ``plan()`` /
    ``root_resolution()`` are never called from here).

    Exits 0 only if every manifest, every protect-file line, and every score file found was
    fully readable AND no manifest entry is dangling. Exits 2 (never 0) on any refusal — D45
    planner note 3: the halted work's dangling-entry refusal was silently exit-0, indistinguishable
    from a clean run to a cron caller. This function is the fix for exactly that gap; it does
    not touch, and is never called from, the ``--apply`` delete loop above.
    """
    ok = True

    protect_paths, protect_err = load_protect_file(a.protect_file)
    if protect_err:
        print(f"[ckpt_gc] VERIFY-MANIFESTS REFUSAL: {protect_err}", file=sys.stderr)
        ok = False
    else:
        print(f"[ckpt_gc] --protect-file: {len(protect_paths)} entrie(s) read")

    # No `or []` fallback: that idiom read as "tolerate an empty list" while actually being a
    # no-op on one, which is how a bare `--experiments-dir` came to scan nothing at exit 0. The
    # degenerate cases are refused where they can be refused with a real diagnostic — argparse, via
    # `nargs="+"` for a flag with NO values and `type=_scan_dir_arg` for a BLANK value (see
    # `main()`). So by the time this runs the list is non-empty AND no element is blank.
    # (The earlier wording claimed only the first of those, and was therefore true of the LIST and
    # false of its CONTENT: `--experiments-dir ''` reached here as `[""]` and scanned nothing at
    # exit 0 — verbatim the failure `nargs="+"` was written to prevent, via the spelling a cron job
    # actually produces.)
    scan_dirs = list(a.experiments_dir)
    # D53 (881G): an EXPLICITLY-passed nonexistent `--experiments-dir` now REFUSES, closing the
    # asymmetry with `--protect-file` (which has always refused an explicit nonexistent path,
    # see `load_protect_file` above) — same operator error, same outcome now on both flags.
    # `test_absent_scan_root_is_not_a_refusal` exercised exactly this shape at exit 0 and is
    # deleted; its replacement lives in test_ckpt_881g_manifest_grammar.py. The DEFAULT (flag
    # omitted) staying a silent no-op is UNCHANGED: a host with no `experiments/` at all still
    # has nothing to say — only an operator-NAMED nonexistent path is a refusal.
    # `a.experiments_dir_explicit`, NOT `getattr(a, ..., False)`: a fail-closed guard must not
    # DEFAULT to the behaviour it was written to remove. The getattr spelling made a namespace
    # missing this attribute (a hand-built one in a test, a future second call site) silently
    # replay the pre-D53 no-op — an explicitly-named nonexistent scan dir back to exit 0 — with no
    # signal at all. A plain attribute access raises AttributeError instead: loud, at the call
    # site, on the caller that forgot to set it. `main()` sets it unconditionally, immediately
    # after `parse_args`, so the only way to reach this without it is a caller that is wrong.
    if a.experiments_dir_explicit:
        for _sd in scan_dirs:
            # `_sd.strip()`: a BLANK value is a different shape with its own dedicated,
            # already-loud guard (`_scan_dir_arg`, at parse time) — this check must not race it
            # or double-diagnose the same input under a second message.
            if (_sd.strip() and not os.path.isdir(_sd)
                    and classify_entry(_sd).kind == "unstatable"):
                print(f"[ckpt_gc] VERIFY-MANIFESTS REFUSAL: {_sd}: --experiments-dir was "
                      f"EXPLICITLY given a path that DOES NOT EXIST — refused. A named scan dir "
                      f"that names nothing must not be indistinguishable from a clean scan "
                      f"reporting zero manifests at exit 0.", file=sys.stderr)
                ok = False
    # How many scan dirs the OPERATOR named, captured BEFORE the collapse rewrites the list. The
    # scan line below reports "<visited> of <requested>", and reading `requested` off the collapsed
    # list turned an operator's two named dirs into "1 of 1 scan dir(s) visited" — a count that
    # silently agrees with itself and can no longer be checked against the command line. It now
    # reads "1 of 2", with the collapse line immediately above it accounting for the difference.
    _n_scan_dirs_requested = len(scan_dirs)
    scan_dirs, _dropped_nested = _dedupe_nested_scan_dirs(scan_dirs)
    if _dropped_nested:
        print(f"[ckpt_gc] --experiments-dir: {len(_dropped_nested)} nested/duplicate scan dir(s) "
              f"collapsed into an already-scanned directory (each real subtree is walked exactly "
              f"once): {', '.join(sorted(_dropped_nested))}")
    scan_stats: dict = {}
    try:
        manifests = load_manifests(scan_dirs, stats=scan_stats)
    except ManifestScanIncomplete as e:
        print(f"[ckpt_gc] VERIFY-MANIFESTS REFUSAL: {e}", file=sys.stderr)
        for line in e.errors:
            print(f"[ckpt_gc]   {line}", file=sys.stderr)
        manifests = e.partial
        ok = False

    # HOW MUCH WAS LOOKED AT, printed before WHAT WAS FOUND. "0 protect, 0 covers, 0 dangling" is
    # the true state of this repo today (56 manifest basenames exist; none carries a
    # `protected:`/`covers:` block) and it is ALSO what an unscanned or empty tree prints — three
    # identical lines for opposite situations. The counts separate them, and they are the cheapest
    # possible check on a scan that silently narrowed.
    print(f"[ckpt_gc] manifest scan: {scan_stats.get('scan_dirs_visited', 0)} of "
          f"{_n_scan_dirs_requested} scan dir(s) visited, "
          f"{scan_stats.get('dirs_walked', 0)} dir(s) walked, "
          f"{scan_stats.get('manifest_files_read', 0)} manifest file(s) read, "
          f"{scan_stats.get('manifest_files_refused', 0)} refused before read")
    print(f"[ckpt_gc] manifests: {len(manifests.protect)} protect, {len(manifests.covers)} "
          f"covers, {len(manifests.dangling)} dangling")
    # This row exists to make manifest state operator-visible, so the classification is LISTED on
    # every path, not only when something is dangling: a bare count cannot tell an operator
    # whether the line they wrote is the line the tool read. One line per classified path, with
    # the citation that produced it.
    for kind, paths in (("protect", manifests.protect), ("covers", manifests.covers)):
        for p in sorted(paths):
            cites = ", ".join(manifests.citations.get(p, [])) or "(no citation)"
            print(f"[ckpt_gc]   {kind:<7} {p}  <- {cites}")
    if manifests.dangling:
        ok = False
        for d in sorted(manifests.dangling):
            for c in manifests.citations.get(d, []):
                print(f"[ckpt_gc] VERIFY-MANIFESTS REFUSAL: DANGLING ENTRY — {c} -> {d} does "
                      f"not exist", file=sys.stderr)

    def _root_error(exc: OSError) -> None:
        nonlocal ok
        print(f"[ckpt_gc] VERIFY-MANIFESTS REFUSAL: "
              f"{getattr(exc, 'filename', '?')}: {exc}", file=sys.stderr)
        ok = False

    # NOTE: `a.roots` is passed WHOLE, not pre-filtered by `os.path.isdir` — that filter is
    # exactly what silently drops a root which is a dangling symlink (`jax_ckpt_fast` today).
    # `_find_score_dirs` applies the same filter internally, but reports FIRST — the dangling link,
    # and (cycle 3) every other unusable shape of a root: blank/empty, a regular file, a fifo, a
    # live link to any of those. Before that, only the dangling case was reported and the score
    # half verified NOTHING at exit 0 for every other wrong `--roots`.
    score_stats: dict = {}
    n_score_read = 0
    score_dirs_seen: set[str] = set()
    # Every score-source error string ALREADY printed, so the bare-`--log-glob` probe at the end of
    # this function does not report a file a run-dir read has just reported. `_load_scores_strict`
    # globs the bare term for EVERY run dir it is called on, so with a co-located `scores.jsonl`
    # present the same bad CWD log is seen by both passes; without this, adding the probe would
    # have turned one refusal into two on that tree (measured: 1 at the 881I ship -> 2). Scope item
    # 3, applied to this row's OWN addition.
    score_errors_seen: set[str] = set()
    for sd in _find_score_dirs(list(a.roots), on_error=_root_error, stats=score_stats):
        score_dirs_seen.add(sd)
        try:
            _load_scores_strict(sd, a.score_regex, a.log_glob)
        except ScoreReadIncomplete as e:
            score_errors_seen.update(e.errors)
            print(f"[ckpt_gc] VERIFY-MANIFESTS REFUSAL: {e}", file=sys.stderr)
            ok = False
        n_score_read += 1

    # The score half's equivalent of the manifest scan line above, and for the same reason: without
    # it, "verify-manifests: OK" after examining ZERO score dirs (a typo'd, blank or wrong-shape
    # `--roots`) is byte-identical to "every score file was read and is clean".
    print(f"[ckpt_gc] score scan: {score_stats.get('roots_visited', 0)} of "
          f"{score_stats.get('roots_requested', len(a.roots))} checkpoint root(s) visited, "
          f"{score_stats.get('dirs_walked', 0)} dir(s) walked, "
          f"{score_stats.get('score_dirs_found', 0)} score dir(s) found, "
          f"{n_score_read} examined")

    # TASK_881S (D50): the THIRD score source — `--score-regex`/`--log-glob` — enumerated
    # independently of the `SCORE_BASENAMES` gate above, so a run scored purely through it reaches
    # the same `_load_scores_strict` and its shape guard is no longer dead code for this input
    # class. See `_find_log_glob_score_dirs`.
    log_glob_extra = _find_log_glob_score_dirs(list(a.roots), a.score_regex, a.log_glob,
                                               score_dirs_seen)
    for sd in log_glob_extra:
        try:
            _load_scores_strict(sd, a.score_regex, a.log_glob)
        except ScoreReadIncomplete as e:
            score_errors_seen.update(e.errors)
            print(f"[ckpt_gc] VERIFY-MANIFESTS REFUSAL: {e}", file=sys.stderr)
            ok = False
        n_score_read += 1
    print(f"[ckpt_gc] score scan (log-glob fallback, independent of SCORE_BASENAMES): "
          f"{len(log_glob_extra)} additional score dir(s) found, "
          f"{len(log_glob_extra)} examined")

    # TASK_881S, escalated cycle: canon's reader globs TWO terms for this source, and the pass
    # above mirrors only the first (under each run dir). The second is BARE — relative to CWD,
    # independent of `--roots` and of every directory walked — so it is checked here, exactly
    # once per invocation. Without this, a score source reachable only through that term was
    # never handed to `_load_scores_strict` at all and its shape guard stayed dead code for the
    # input class D50 is about. See `_load_bare_log_glob_scores_strict`.
    bare_matches = _bare_log_glob_matches(a.score_regex, a.log_glob)
    if bare_matches:
        try:
            _load_bare_log_glob_scores_strict(a.score_regex, a.log_glob)
        except ScoreReadIncomplete as e:
            # Only what no pass above has already said, and NEVER a downgrade of the verdict: the
            # term is examined unconditionally (that is the whole fix), and if every one of its
            # errors was already reported then `ok` is already False from that report. Suppressing
            # a duplicate LINE is not the same as making the check conditional on a run dir
            # existing — with no score dir at all, `score_errors_seen` is empty and this refuses on
            # its own, which is exactly the D50 tree.
            fresh = [x for x in e.errors if x not in score_errors_seen]
            if fresh:
                print(f"[ckpt_gc] VERIFY-MANIFESTS REFUSAL: "
                      f"{ScoreReadIncomplete(e.run_dir, fresh)}", file=sys.stderr)
                ok = False
    print(f"[ckpt_gc] score scan (bare --log-glob term, relative to CWD, independent of --roots): "
          f"{len(bare_matches)} file(s) matched, "
          f"{1 if bare_matches else 0} read(s) — once per invocation, never once per directory")

    print(f"[ckpt_gc] verify-manifests: {'OK' if ok else 'REFUSED'}")
    return 0 if ok else 2


def _scan_dir_arg(value: str) -> str:
    """argparse ``type`` for ``--experiments-dir``: reject a BLANK value at PARSE time.

    ``nargs="+"`` closed the BARE flag (`--experiments-dir` with no values). It did not close the
    flag given an EMPTY STRING, which is the SAME defect through the spelling a cron job actually
    produces: ``--experiments-dir "$EXPDIR"`` with ``EXPDIR`` unset. All three spellings —
    ``--experiments-dir ''``, ``--experiments-dir=``, ``--experiments-dir ' '`` — scanned nothing
    and printed "0 of 1 scan dir(s) visited … 0 manifest file(s) read" / "verify-manifests: OK" /
    exit 0 on a tree carrying two real protections. The `nargs` comment below asserted the list is
    non-empty by construction, which was true of the LIST and false of its CONTENT.

    It is a provable inconsistency inside one invocation, too: ``--protect-file ''`` in the same
    run IS refused ("could not be read ([Errno 2] … '')"). Empty-string flags failing OPEN is a
    documented incident class in this repo, so the empty case is refused where it can be refused
    with a real diagnostic — argparse — rather than deep inside the scan.

    An ABSENT-but-named directory is still NOT a refusal (`test_absent_scan_root_is_not_a_refusal`
    passes a real name): this rejects a value that names nothing at all, not a value that names
    something which is not there. Scoped to ``--experiments-dir``, a flag this row added and which
    canon never reads, so ``--apply`` behaviour cannot move; canon's ``--roots`` keeps its own
    parsing untouched (its blank case is refused in ``_find_score_dirs``, on the ingestion path
    only).
    """
    if not value.strip():
        raise argparse.ArgumentTypeError(
            f"--experiments-dir was given a BLANK value ({value!r}) — refused. A blank scan dir "
            f"scans nothing while reporting success, which WITHDRAWS whatever coverage the "
            f"intended directory grants. This is `--experiments-dir \"$VAR\"` with VAR unset: "
            f"omit the flag entirely to use the default ['experiments'].")
    return value


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roots", nargs="+", default=["jax_ckpt_fast", "runs"])
    ap.add_argument("--keep-best", type=int, default=3, help="top-K by score to keep per run")
    ap.add_argument("--keep-latest", type=int, default=1, help="newest N always kept (resume)")
    ap.add_argument("--keep-unscored", type=int, default=2,
                    help="newest N unscored kept even under the disk floor")
    ap.add_argument("--min-free-gb", type=float, default=30.0,
                    help="below this, unscored checkpoints become recency-prunable")
    ap.add_argument("--critical-free-gb", type=float, default=12.0,
                    help="below this, print a CRITICAL line (the DB shares this filesystem)")
    ap.add_argument("--score-regex", default=None,
                    help=r"e.g. 'Epoch (?P<epoch>\d+).*reward_ma20[= ](?P<score>[-\d.]+)'")
    ap.add_argument("--log-glob", default="*.log")
    ap.add_argument("--protect-file", default=None, help="file of paths never to delete")
    # NAMING CONSTRAINT (canon parity, not taste): argparse resolves unambiguous long-option
    # PREFIXES, so adding an option can silently change what an existing abbreviation means. The
    # first draft of this row called the mode `--check-manifests`, which made `--c` — previously
    # an unambiguous alias for `--critical-free-gb` — an "ambiguous option" error. Both flags
    # below therefore start with a letter no pre-881I long option uses ('e', 'v'), so every
    # abbreviation that resolved at HEAD still resolves to the same option. Enforced by
    # test_ckpt_881i_manifest_ingestion.py::test_head_option_abbreviations_still_resolve, which
    # checks the whole class rather than this one instance — consult it before renaming or adding
    # any flag here.
    # `nargs="+"`, NOT `nargs="*"`, and that is a correctness choice rather than a style one.
    # Under `nargs="*"`, a BARE `--experiments-dir` (flag present, no values) parses to `[]` — not
    # to the default — so `--verify-manifests --experiments-dir` scanned ZERO directories, not even
    # `experiments/`, and printed "0 protect, 0 covers, 0 dangling" / "verify-manifests: OK" /
    # exit 0 on a tree carrying real protections. A degenerate flag silently WITHDREW REAL
    # COVERAGE while reporting success, which is what this whole section exists to prevent (cf.
    # `_DANGLING_LINK_WHY`: "WITHDRAWS coverage rather than silently granting none"). `nargs="+"`
    # keeps the two cases distinct where the distinction belongs, in argparse itself: flag OMITTED
    # -> `["experiments"]` (the default, unchanged); flag given with values -> those values
    # (unchanged); flag given EMPTY -> refused at parse time, "expected at least one argument",
    # exit 2. Nothing else about the flag moves, so the abbreviation class pinned by
    # test_head_option_abbreviations_still_resolve is untouched (`nargs` is not part of a name).
    # `type=_scan_dir_arg` closes the other half: `nargs="+"` refuses the flag given NO values, and
    # the type refuses a value that is BLANK. See `_scan_dir_arg` — the comment at `scan_dirs` in
    # `_run_manifest_check` used to claim the list was non-empty by construction, which was true of
    # the list and FALSE of its content.
    ap.add_argument("--experiments-dir", nargs="+", type=_scan_dir_arg, default=["experiments"],
                    help="TASK_881I: dirs scanned for PROTECTED.txt/SPEC.md manifests by "
                         "--verify-manifests (at least one value required if the flag is given; "
                         "omit it entirely for the default). Ingestion only — no effect on "
                         "--apply.")
    ap.add_argument("--verify-manifests", action="store_true",
                    help="TASK_881I: PARSE and CLASSIFY every manifest (--experiments-dir), "
                         "--protect-file, and score file (--roots) this tool would otherwise "
                         "consult, and REFUSE (exit 2) on anything malformed, symlinked, torn, "
                         "unreadable-by-shape (TASK_881S: the score sources include the "
                         "--score-regex/--log-glob log, matched BOTH under each --roots run dir "
                         "and — as this tool's own reader does — relative to CWD), "
                         "or dangling. Deletes and discovers NOTHING — returns before --apply's "
                         "checkpoint discovery ever runs. Mutually exclusive in effect with "
                         "--apply (this mode never reaches it).")
    ap.add_argument("--apply", action="store_true", help="actually delete (default: dry run)")
    a = ap.parse_args()

    # D53 (881G): detect whether `--experiments-dir` was EXPLICITLY given, without touching the
    # frozen `ap.add_argument("--experiments-dir", ..., default=["experiments"], ...)` line above
    # (verbatim-matched by test_red_when_experiments_dir_nargs_is_reverted_at_its_real_drop_site).
    # A second, throwaway parser that declares ONLY this one option resolves the SAME argparse
    # abbreviation rules against the SAME argv, so an abbreviated spelling (e.g. `--exp foo`) is
    # detected identically to the full flag name — sniffing `sys.argv` for a literal substring
    # would miss that.
    # `nargs="*"`, no `type=`, deliberately looser than the real flag's own `nargs="+"` +
    # `type=_scan_dir_arg`: this sentinel exists ONLY to answer "was the flag present at all", so
    # it must never itself raise on a shape (a bare flag, a blank value) the REAL parser's own
    # guards are responsible for diagnosing — those guards already ran, above, on `ap`, before
    # this sentinel is ever reached, and this line must not race them with a second opinion.
    _edp = argparse.ArgumentParser(add_help=False)
    _edp.add_argument("--experiments-dir", nargs="*", default=None)
    _edns, _ = _edp.parse_known_args()
    a.experiments_dir_explicit = _edns.experiments_dir is not None

    if a.verify_manifests:
        return _run_manifest_check(a)

    roots = [r for r in a.roots if os.path.isdir(r)]
    if not roots:
        print(f"[ckpt_gc] no existing roots among {a.roots} — nothing to do")
        return 0
    protect = set()
    if a.protect_file and os.path.exists(a.protect_file):
        protect = {os.path.realpath(l.strip()) for l in open(a.protect_file) if l.strip()}

    fg = free_gb(roots[0])
    under = fg < a.min_free_gb
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(f"[ckpt_gc {stamp}] free={fg:.1f}GB floor={a.min_free_gb}GB "
          f"under_floor={under} mode={'APPLY' if a.apply else 'dry-run'}")
    if fg < a.critical_free_gb:
        print(f"[ckpt_gc] CRITICAL: free {fg:.1f}GB < {a.critical_free_gb}GB — "
              f"trading_data.db shares this filesystem; stop writing before it fills")

    runs = collect(roots)
    deletions, summary = plan(runs, lambda rd: _load_scores(rd, a.score_regex, a.log_glob),
                              a.keep_best, a.keep_latest, under, a.keep_unscored)
    for s in summary:
        print("  " + s)

    reclaimed = 0
    for path, size, reason in deletions:
        ok, why = _safe_target(path, roots)
        if not ok:
            print(f"  REFUSE {path}: {why}")
            continue
        if os.path.realpath(path) in protect:
            print(f"  PROTECTED {path}: listed in protect-file")
            continue
        print(f"  {'DELETE' if a.apply else 'would delete'} {path} "
              f"({size/1e6:.0f} MB) — {reason}")
        if a.apply:
            trash = f"{path}.gc-trash-{os.getpid()}"
            try:
                os.rename(path, trash)      # atomic: a half-deleted ckpt can never look loadable
                shutil.rmtree(trash)
                reclaimed += size
            except OSError as e:
                print(f"  ERROR deleting {path}: {e}")
    print(f"[ckpt_gc] {'reclaimed' if a.apply else 'would reclaim'} {reclaimed/1e9:.2f} GB "
          f"across {len(deletions)} checkpoints; free now {free_gb(roots[0]):.1f}GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
