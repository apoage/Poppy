# TASK_881D — the checkpoint protection model + deleter, on 881I's settled ingestion

**Status**: READY 2026-08-17 (planner; D45 re-cut). **Amended 2026-08-19 (D50):** the ingestion
881D consumes is settled by 881I's D50 re-cut — but the SCORE-SOURCE side is NOT settled
(TASK_881S owns it) and this row must not be told otherwise. **NEW PRECONDITION (D50): there is
no live manifest corpus at all** — 0 of 56 SPEC.md/PROTECTED.txt files carries a
`protected:`/`covers:` block, so the protection model currently protects NOTHING and its first
real exercise would be whatever an operator writes after it ships. **Scope item 0 below authors
the corpus.** Also carried: the tool is `tools/ckpt_gc.py`, NOT `tools/ckpt_archive.py`.
**Sequenced AFTER TASK_881I (shipped run 37) — AND BLOCKED UNTIL TASK_881G SHIPS (D53,
2026-08-20):** the corpus this row's scope item 0 authors must be written against a grammar that
exists in writing; today the MD032 form refuses and the accepted grammar is documented nowhere.
**Size: M. Risk: HIGH in consequence** — this row owns the `--apply` path of a tool that deletes
checkpoints. The D32/D34 `--apply` freeze STANDS until this ships AND 880's D35 gate opens.
**Resume from the tree state after 881I ships** (the `stash-881-halted-d45` tag is superseded;
run 36's durable copies are `lore/handoff_2026-08-19_factory_run_36_881I_cycle3.patch` for the
ingestion and the D45 stash for the protection/deleter halves, which six opus judges across three
cycles verified — take them as settled; re-verify, don't rebuild).

## What is already verified in the stash (from D45 — re-run each kill from scratch, do not take on report)

- All six prior kills DEAD: symlinked manifest FILE/DIR refusals; torn `scores.jsonl` per-run-dir
  refusal with peak survival; the cycle-2 killer's full four-step sequence (healthy
  `--write-index` 42.0 → tear → `--apply` refuses AND index still reads 42.0 → follow the
  refusal's own remediation → re-run under the floor → **peak survives**); missing/symlinked/
  directory `--protect-file` → exit 2.
- `write_run_index` atomicity under SIGKILL mid-`json.dump`; degrade-not-raise on read-only dirs.
- `root_resolution()` closes the D38-c3 asymmetry — survivor sets BYTE-IDENTICAL under `--apply`
  on both deleters with a simulated mounted scratch disk.
- `plan()`'s 4th return value honoured by both production callers.
- MAJOR E's dangling-entry refusal correctly scoped (3 unresolvable `828_stage3_train` entries
  refuse 0 run dirs; all 7 checkpoints across `runs/832_e5` + `runs/861_{ref,imprint}` survive).

## Scope

0. **PRECONDITION — author the real `PROTECTED.txt` corpus** (D50): the three
   `experiments/828_stage3_train` entries D45 names (scratch disk-unmounted, they stay loudly named
   in every report), at minimum, over the run dirs the deleter can actually see. The protection
   model's acceptance below is then exercised against a LIVE corpus, not rebuilt inputs alone —
   and the `--apply`-freeze conversation happens with eyes open about what is actually protected.

   **The grammar to author it against (TASK_881G, D53 — SHIPPED; mirrored verbatim from
   `tools/ckpt_gc.py`'s module docstring, "MANIFEST GRAMMAR" section):**

   MANIFEST BASENAMES: exactly `PROTECTED.txt` and `SPEC.md` (`MANIFEST_BASENAMES`). Either name,
   anywhere under a scanned directory, is read as a manifest. Both share the SAME per-line ITEM
   grammar (below, "one path, one line, `$`-anchored"); PROTECTED.txt additionally supports a
   plain `#` comment line and SPEC.md does not — SPEC.md is Markdown, where a leading `#` is
   heading TEXT, and that divergence is deliberate, not a bug.

   WHOLE-FILE PRECONDITIONS, identical for both basenames, checked before either parser ever runs:
   - the basename must be a real, regular, non-symlink file at a real, tracked path. A symlink
     (to a file OR to a directory), a real directory, a fifo/socket/device, or a manifest reached
     only by crossing a symlinked ANCESTOR directory (including a symlinked scan root) is
     REFUSED, never read.
   - the file must decode as UTF-8 with no byte-order mark and no NUL byte.
   - ANY malformed line, anywhere in the file, WITHDRAWS THE WHOLE FILE's directives — not just
     the offending line.
   - a well-formed entry naming a path that does not exist on disk is NOT malformed — it is
     DANGLING: it protects nothing and is reported on its own, but does not poison the rest of
     the file the way a malformed LINE does. "NOT malformed" is about INTRA-FILE poisoning ONLY
     (the file's other entries still parse and still grant) and is NOT an acceptance: a dangling
     entry makes `--verify-manifests` print `DANGLING ENTRY` and EXIT 2, never 0, exactly like a
     refusal. Both are true at once; reading only the first is how an author concludes a typo'd
     path will pass the check.
   - ENTRY PATHS RESOLVE AGAINST THE PROCESS'S CWD, NOT against the manifest's own directory.
     Every entry — either basename, `--protect-file`, relative or absolute — goes through
     `os.path.realpath(<the text as written>)`, relative to wherever the tool was INVOKED. So
     `- runs/a` inside `experiments/999_z/SPEC.md` names `$CWD/runs/a`, NOT
     `experiments/999_z/runs/a`; from another CWD the same manifest names something else, and
     usually something DANGLING. Author the corpus for, and run the tool from, the repo root.

   PROTECTED.txt GRAMMAR — one directive per line:
   - `protect: <path>` or `covers: <path>` — exactly one directive keyword, a colon, and a
     single-token path, optional surrounding whitespace, nothing else on the line. ACCEPTED:
     ```
     protect: runs/some_run/snapshot_ep10
     covers: runs/other_run
     ```
   - WHITESPACE AROUND THE DIRECTIVE IS FREE — the same micro-grammar the SPEC.md item form uses
     (below). The line is STRIPPED before it is matched, so leading indentation
     (`  protect: runs/a`) and trailing whitespace are ACCEPTED, and the space after the colon is
     optional (`protect:runs/a` is ACCEPTED). What is NOT free is the number of tokens after the
     colon — exactly one.
   - a blank line — ACCEPTED, produces no entry. A line of nothing but SPACES or TABS counts as
     blank here exactly as an empty one does (the test is on the STRIPPED line).
   - a line whose STRIPPED form starts with `#` (indentation before the `#` is fine) — ACCEPTED
     as a comment, produces no entry.
   - anything else is MALFORMED (whole file withdrawn) — including two tokens after the
     directive (`protect: a b`), a bare `protect:` with no path, an unrecognized directive word,
     or plain prose.

   SPEC.md BLOCK GRAMMAR — a `protected:` or `covers:` block:
   - the header is column-0, lower-case, exact: `protected:` or `covers:`, optionally followed on
     the SAME line by nothing but whitespace to end-of-line/file (a torn final line with no
     trailing newline still counts as a header).
   - immediately after the header — with EITHER zero blank lines OR one-or-more blank lines
     before the first item — a list of items, one path per line, each written `- <path>` (a
     single `-`, optional whitespace, exactly one non-whitespace token, nothing else on the
     line). BOTH of the following are ACCEPTED and equivalent (the second is the canonical MD032
     form — a list surrounded by blank lines — and is the FIRST shape an author who runs a
     markdown linter will type; accepting it is what D53 required):
     ```
     protected:
     - runs/some_run/snapshot_ep10
     - runs/other_run
     ```
     ```
     protected:

     - runs/some_run/snapshot_ep10
     - runs/other_run
     ```
   - THE ITEM MICRO-GRAMMAR IS DELIBERATELY FORGIVING, in deliberate ASYMMETRY with the strictly
     column-0, lower-case, exact HEADER above it. An item line is matched STRIPPED-EQUIVALENTLY,
     so all three of these are ACCEPTED and name exactly the path they appear to: LEADING
     INDENTATION before the dash (`  - runs/a`, the Markdown nested-list shape); NO whitespace
     between the dash and the path (`-runs/a`); TRAILING whitespace after the path
     (`- runs/a   `). The asymmetry is on purpose: an odd HEADER can silently withdraw a whole
     declared block, so it refuses; an odd but unambiguous ITEM inside an already-matched block
     can only go into that block, so it is read. What an item may NEVER do is name a second token
     (see INTERRUPTED below) — the one thing whitespace cannot buy.
   - A LINE OF NOTHING BUT SPACES OR TABS IS A BLANK LINE, uniformly, in ALL THREE places this
     grammar says "blank": the MD032 gap between a header and its first item, the SPLIT gap after
     at least one item, and the blank line that legitimately ENDS a list. There is no shape where
     a whitespace-only line behaves differently from an empty one.
   - once at least one item has been consumed, a blank line followed by more PROSE, a heading, or
     end-of-file legitimately ENDS the block. ACCEPTED.
   - once at least one item has been consumed, a blank line followed by ANOTHER `- <path>` item —
     the list resuming after a gap — is REFUSED (SPLIT). This is the only place a leading-vs-mid-
     list blank line matters: BEFORE the first item, a gap is the MD032 fence above; AFTER at
     least one item, a gap is a SPLIT.
   - a header carrying inline content (`protected: runs/a`) — REFUSED, whole file withdrawn.
   - a header with ZERO items — REFUSED as EMPTY, whole file withdrawn.
   - WHICH REFUSAL WINS WHEN BOTH COULD APPLY (EMPTY vs INTERRUPTED): the INTERRUPTED diagnosis
     requires at least ONE already-consumed item to have been interrupted, so a non-item,
     non-blank line that is the FIRST content after the header (`protected:` / `hello` /
     `- runs/a`) refuses as EMPTY ("it names nothing"), not as INTERRUPTED, even though a
     well-formed item follows it. Both are exit-2 whole-file withdrawals; only the message
     differs. The ONE exception is an attempted item naming more than one token (`- a b`), which
     refuses wherever it sits under its OWN distinct message ("list item at line N carries MORE
     THAN ONE token"), because that line unambiguously IS an attempted item and "names nothing"
     would be false about it.
   - TWO OR MORE headers for the SAME key in one file — REFUSED (duplicate), whole file
     withdrawn. `protected:` and `covers:` are independent keys — one of each is fine.
   - an item line naming MORE THAN ONE token (`- runs/a runs/b`, or `- runs/a # note` — a
     TRAILING COMMENT on an item is this same shape) — REFUSED (the INTERRUPTED-list class, under
     its own distinct message: "list item at line N carries MORE THAN ONE token"), whole file
     withdrawn.
   - any OTHER non-blank, non-item line inside a started list (a stray word, a `*` bullet, a bare
     `-`, or a `#` line — NOT a comment here, unlike PROTECTED.txt) — REFUSED as INTERRUPTED,
     whole file withdrawn.
   - a line that is ENTIRELY two-or-more `-` characters (a markdown THEMATIC BREAK, e.g. `---`)
     is NEVER a list item, even though its first character reads as a bullet marker — it falls
     through to whichever of the two bullets above applies.
   - a header that misses column-0/lower-case/exact spelling by ANYTHING (indentation, a
     capitalized `PROTECTED:`, a leading `>`/`*`/`-`, a byte-order mark) is a NEAR-MISS: REFUSED
     (never silent), whole file withdrawn, EVEN beside a well-formed block elsewhere and EVEN
     inside a fenced code block — both accepted-deliberately for the same reason (a well-formed
     header inside a fence is ALSO read as a real block; this parser does not try to detect
     fences at all).
   - FENCES ARE NOT INVISIBLE, THEY ARE UNSEEN — and the difference bites in the most natural
     shape an author writes. Because no fence is detected, the CLOSING BACKTICK-FENCE LINE of a
     fenced EXAMPLE is just another line, and if it comes IMMEDIATELY after the last item it is a
     non-item line inside a started list: the whole file is REFUSED as INTERRUPTED. So "a header
     inside a fence can only ADD protection" is FALSE as written — it can also WITHDRAW the
     entire file. REFUSED:

     ````
     ```
     protected:
     - runs/a
     ```
     ````

     ACCEPTED (one blank line TERMINATES the list before the closing fence; the block inside the
     example parses as a real block and its entries are real):

     ````
     ```
     protected:
     - runs/a

     ```
     ````

     The practical rule for a document that must both illustrate and be ingested: leave a blank
     line after the last item of any example block, and expect every path written in an example
     to be ingested as a real entry (a nonexistent one lands in DANGLING and exits 2).

   `--protect-file` GRAMMAR — AS READ BY `--verify-manifests` (`load_protect_file`), the ONLY
   reader this grammar describes (a flat list — no `protect:`/`covers:` prefix): one path per
   line, realpath'd. A blank (stripped) line, or a line whose STRIPPED form starts with `#`
   (indentation before the `#` does not exempt it), is skipped; anything else is one path,
   trusted verbatim. A missing, symlinked, or non-regular-file `--protect-file`, WHEN THE FLAG IS
   GIVEN, is REFUSED.

   NOT TRUE OF THE DELETE PATH (dry-run / `--apply`), stated because a doc that over-promised
   here would become this row's false precondition: `main()`'s deletion loop does NOT call
   `load_protect_file`. It reads the file inline, guarded by `os.path.exists` — every non-blank
   line is realpath'd, `#` is NOT a comment there, and the four shapes `--verify-manifests`
   refuses do NOT all fail the same way here. A MISSING `--protect-file`, or one that is a
   DANGLING SYMLINK (`os.path.exists` reports False once a symlink's target is gone), is silently
   treated as EMPTY at exit 0 rather than refused. A DIRECTORY is not caught at all: `open()`
   raises `IsADirectoryError`, uncaught, so the process CRASHES with a traceback at exit 1 — a
   crash, not "empty at exit 0". A LIVE SYMLINK to a real file is not refused either: it is READ
   AND HONOURED, an unrefused ACCEPT, the opposite failure mode from the missing/dangling case. A
   FIFO `--protect-file` is worse than either: `open()` on a reader-less FIFO in read mode BLOCKS
   FOREVER, the same hang `load_protect_file`'s own docstring warns about. That inline read is
   canon and 881G did not change it; UNIFYING the two readers (routing the deleter through
   `load_protect_file`, so a missing protect file cannot silently un-protect anything before a
   deletion) IS THIS ROW'S WORK. Until then: run `--verify-manifests` FIRST and act on its exit
   code, because the delete path will not repeat its refusals.

   `--experiments-dir` SEMANTICS: the flag OMITTED and absent-by-default is a silent no-op; the
   flag EXPLICITLY GIVEN a nonexistent value now REFUSES (D53 — matches `--protect-file`); two
   values that resolve to the same or a NESTED directory are collapsed to one before scanning
   (never a refusal — a redundant request is not an operator error). The collapse is REPORTED on
   stdout and the scan line still counts the dirs the OPERATOR named ("1 of 2 scan dir(s)
   visited"). A COLLAPSE IS A COUNTING FIX AND MAY NEVER CHANGE A VERDICT, so it happens ONLY
   when (a) NEITHER dir is reached by crossing a SYMLINK (`realpath != abspath` — a symlinked
   scan dir is the input that produces the "manifest reached only by crossing a SYMLINKED
   DIRECTORY" refusal, and collapsing it on realpath ERASED that refusal), and (b) the SURVIVING
   dir's walk PROVABLY REACHES the dropped one (being underneath it is not enough: the walk does
   not descend into `.git` or `__pycache__`, so collapsing `experiments experiments/__pycache__`
   silently dropped a manifest, "1 protect" -> "0 protect", at exit 0). Otherwise BOTH dirs are
   kept and the duplicate stat is tolerated. Even together, those two conditions only prove the
   surviving walk REACHES the dropped directory against its own name-based prune list — not
   against every I/O outcome: a non-listable intermediate directory (measured — an ancestor at
   mode 111) still stops the walk and still drops the nested dir it can no longer reach, "1
   protect" -> "0 protect" on the same host. That is NOT a silent withdrawal, though: it raises a
   `ManifestScanIncomplete` and the whole run REFUSES at exit 2 ("Permission denied" printed on
   both the collapsed and uncollapsed invocation), so no verdict ever changes — the fail-closed
   direction holds — but "nothing under a collapsed directory goes un-scanned" is not literally
   true; only "nothing goes un-scanned without the scan itself loudly refusing" is.

1. Resume the protection model + deleter from the stash; re-run every kill above against the
   resumed tree.
2. Consume 881I's shipped ingestion as a settled input (do not re-open it; do not assume the
   score-source side is settled — that is 881S).
3. The score reader resolving by `path`/`step` (D35's original item), the `_is_protected`
   descendant rules, the TTL re-scan, the failed-`rmtree` audit row, and the `followlinks=False`
   BLOCKER — all as previously specced in 881, unchanged.
4. The disputed pre-existing `_order_newest_first` — needs a third independent measurement
   (carried from 881).
5. The documented side effect from D45 planner note 4: the dangling-advisory-doc refusal also
   disables the IN-RUN pruner process-wide via the stale cache — independently judged CORRECT,
   but its blast radius is global by assertion rather than necessity, and the docstring omits the
   in-run pruner effect. Fix the docstring; scope the refusal's blast radius if cheap, document
   it if not.

## Acceptance (gates)

- (i) Every kill from the verified list reproduced on the final tree, independently rebuilt.
- (ii) A `--apply` dry-run on the REAL tree keeps all 7 load-bearing checkpoints
  (`runs/832_e5` + `runs/861_{ref,imprint}`), printed and diffed against the pre-881 survivor set.
- (iii) 881I's malformed-input corpus still refuses identically when driven through 881D's
  callers (integration, not just unit).
- (iv) Byte-identity of every non-`--apply` code path vs HEAD.
- (v) Standing clauses: guards go red when removed at the real drop site; real CLI invocations,
  no stubbed internals; mutations independently rebuilt.
- (vi) **The corpus of scope item 0 exists, is tracked-or-deliberately-untracked with the reason
  recorded (D48's class: cited-but-untracked evidence is a named structural pattern now), and
  appears in every report the tool emits.**

## Out of scope

Manifest ingestion semantics (881I's, settled). Score-source enumeration (881S). TASK_880 (gated
by D35 on THIS row shipping + real score files the new reader resolves). Lifting the `--apply`
freeze (operator's, post-ship).
