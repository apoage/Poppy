# TASK_881D — the checkpoint protection model + deleter, on 881I's settled ingestion

**Status**: READY 2026-08-17 (planner; D45 re-cut). **Sequenced AFTER TASK_881I** — it consumes
881I's ingestion as a settled input and does not re-open it.
**Size: M. Risk: HIGH in consequence** — this row owns the `--apply` path of a tool that deletes
checkpoints. The D32/D34 `--apply` freeze STANDS until this ships AND 880's D35 gate opens.
**Resume from tag `stash-881-halted-d45`** — the protection-model and deleter halves, which six
opus judges across three cycles verified (take them as settled; re-verify, don't rebuild).

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

1. Resume the protection model + deleter from the stash; re-run every kill above against the
   resumed tree.
2. Replace the halted 881's ingestion calls with 881I's single-function contract.
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

## Out of scope

Manifest ingestion semantics (881I's, settled). TASK_880 (gated by D35 on THIS row shipping +
real score files the new reader resolves). Lifting the `--apply` freeze (operator's, post-ship).
