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
