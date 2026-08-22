> **STATUS (2026-08-22): SHIPPED as `task881s-score-source-enumeration` (05254a2).**
> Gates (i)–(iv) PASS. D50 is closed: `--verify-manifests … --score-regex … --log-glob <FIFO>` went
> from `verify-manifests: OK` **exit 0** to **exit 2, one refusal, no hang**, while the real path
> still hangs at EXIT=124 — so the guard covers the genuine danger, not a rehearsal of it.
> `tools/tests`: **460 passed, 1 skipped, 0 failed** (orchestrator-run first-hand).
>
> **Took 3 cycles, and the reason is worth keeping.** The row's own dedup requirement (scope item 3)
> was **pattern-matched twice before it was implemented**. Cycle 1 mirrored only the first of canon's
> two glob terms, so D50 reproduced one layer down through the bare CWD-relative term. Cycle 2
> guarded on `os.path.isabs` — a spelling test standing in for the real invariant — and a reviewer
> broke it with a relative `../shared.log`, which is directory-independent by traversal without being
> absolute (5 refusals for one physical file). Cycle 3 implemented the invariant itself:
> **resolved-path collision**. The lesson generalises past this row — a guard written as a syntactic
> test for one spelling of a property is not a guard on that property.
> **Also: the cycle-1 repro test asserted substring presence and exit code but never a COUNT**, which
> is exactly why a per-directory inflation shipped under a passing test. Count assertions were added.
>
> **Gate (iv):** pure insertion — 249 added, **0** removed or rewritten vs the 881I-shipped tree
> (`bf1f8d5`), independently verified by two judges. The delete path stays unreachable and the
> `--apply` freeze is untouched.
>
> **MAJOR-A — pre-existing, deliberately NOT closed here, and now reproducible on demand.** With N
> score dirs that each carry a valid `scores.jsonl` plus one shared absolute `--log-glob`, the tool
> prints N refusal lines for one physical file. It reproduces identically at `bf1f8d5`, so it is not
> a regression from this row, and closing it requires editing pre-existing `_load_scores_strict`
> lines that gate (iv) forbids. Two tests ship with it: one pinning today's counts, one running the
> same tree against `git show bf1f8d5:tools/ckpt_gc.py` so "pre-existing" is **measured, not
> asserted**. Both go red if it is ever fixed. **Candidate for 881D or a row permitted to touch the
> reader.**
>
> **Remainders — 3 MINORs, all count-inflation, all fail-closed (exit codes correct, nothing
> suppressed). The 3-cycle halt rule was reached, so these are named rather than fixed:**
> (1) a dangling `--log-glob` symlink is now reported **twice** where HEAD reported once — the same
> duplicate-refusal class this row closes for `MANIFEST_BASENAMES`, moving the other way on the score
> side; (2) `realpath` folds two DISTINCT dangling symlinks sharing a missing target into one
> candidate and does NOT fold hardlinks to a single inode, so the "one physical file, one candidate"
> docstring over-claims one way and under-claims the other — keying on `(st_dev, st_ino)` closes it;
> (3) `--log-glob ''` matches every walked directory and emits one refusal each (several hundred
> lines on this repo) — `--experiments-dir` refuses a blank at parse time, `--log-glob` has no
> equivalent.
>
> **Note for 881G/881D:** `--verify-manifests` now opens and line-reads every `*.log` under `--roots`
> (448 files on this repo, previously 0). Widened IO surface, and the pre-existing uncoercible-record
> echo is unclipped.

# TASK_881S — score-source enumeration in `ckpt_gc --verify-manifests`

**Status**: READY 2026-08-19 (planner; seeded by D50 — the layer run 36's three 881I cycles never
scoped, where the MAJOR that halted the row lives).
**Size: XS/S. Risk: MEDIUM** — same tool, same refusal discipline as 881I.
**Sequenced AFTER TASK_881I ships** (same mode of the same tool; do not race it).

## The defect (D50, reproduced)

`_find_score_dirs` enumerates a directory only if it contains `SCORE_BASENAMES`
(`scores.jsonl` / `checkpoints_index.json`), so a run scored PURELY via
`--score-regex`/`--log-glob` — the module's own documented THIRD fallback source — is never
handed to `_load_scores_strict`, and that reader's log-file shape guard is dead code for exactly
that input class. Reproduced: `--verify-manifests … --score-regex … --log-glob <FIFO>` with no
`scores.jsonl` anywhere → `verify-manifests: OK`, **exit 0**; the same flags on the real path →
**EXIT=124, hangs forever**. So `--verify-manifests` is not the pre-flight guarantee its own
`--help` advertises. Not live-dangerous today (no caller outside tests) and touches no delete
path — but it must not ship labelled "closed".

## Scope

1. **Enumerate what you claim to consult.** Every score source `--verify-manifests` claims to
   consult must be ENUMERATED by the same code path that consults it: `--log-glob`/`--score-regex`
   reachable independently of the `SCORE_BASENAMES` gate, with the same shape guards (no hang on
   a FIFO log, refusal on symlinked/torn inputs) as the co-located-score path.
2. **The test rule that hid it:** a test for a fallback source must NOT construct the primary
   source alongside it — rewrite the three `--score-regex` tests so the regex branch runs with no
   `scores.jsonl` present (the suite's own comment currently documents the dependency instead of
   closing it).
3. **Duplicate-refusal dedup:** a dangling manifest symlink is currently reported twice (once by
   `_walk_scan_dirs`'s `DanglingScanLink`, once by `manifest_shape_refusal`) — one real defect
   must not inflate a problem count an alert might key on.

## Acceptance (gates)

- (i) The D50 reproduction now refuses (exit 2) or safely completes on every constructed shape —
  FIFO/socket/symlinked/dangling log inputs, with and without a co-located `scores.jsonl`.
- (ii) The fallback tests pass with the primary source ABSENT; a mutation that re-adds the
  dependency is caught.
- (iii) Every guard goes red when removed at its real drop site; mutations independently rebuilt;
  suites green twice back-to-back.
- (iv) Byte-identity of every code path outside the enumeration change vs the 881I-shipped tree.

## Out of scope

Manifest ingestion (881I's, settled). The protection model and deleter (881D). The `--help`
promise beyond what the code can keep (if the guarantee can't be made true, the help text is
narrowed in the same change — never the reverse).
