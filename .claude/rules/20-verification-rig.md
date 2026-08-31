---
description: Mechanics of the HEAD-control worktree A/B rig and the traps that make a clean worktree lie to you. Applies when running suites, byte-identity gates, or eval sweeps.
paths:
  - "experiments/**"
  - "fixtures/**"
  - "tools/**"
  - "tests/**"
---

# Verification rig — the A/B worktree and its traps

The prohibition ("never mutate HEAD in the live tree") is in
`.claude/rules/00-shared-tree-concurrency.md`. This file is the **mechanics**.

```bash
git worktree add --detach <scratch> HEAD
```

## Two traps that make a clean worktree lie to you

1. **`fixtures/trading_fixture.db` and `runs/` are gitignored**, so they are **absent** from a
   fresh worktree. Symlink them in
   (`ln -s /home/lukas/Trader/<path> <worktree>/<path>`) or you will chase a phantom
   `teacher_floor` / byte-identity diff. `runs/` may already exist as a tracked directory — check
   per-subdirectory, not just the top level.

2. **Some suites cannot run from a worktree at all.**
   `experiments/861_search_teacher_imprint/smoke_e2e.py` and anything going through `imprint.py`'s
   provenance gate compare the *db path string* recorded in the collection's provenance against
   the one this run resolved. A symlinked fixture resolves to
   `<worktree>/fixtures/trading_fixture.db`, so the gate correctly **refuses**. That is not a
   defect — do **not** "fix" it. Settle the question a cheaper way: diff the generating constant
   between the two trees instead of running the suite.

## Read the suite's own exit line

When running a suite in the background, read the **suite's** exit line, not the harness
notification's exit code. Piping through `tail` makes the notification report the *pipeline's*
status. An 861 run reported "exit code 0" while the suite itself printed `861_SMOKE_EXIT: 1`.

## Where suites run

Test suites, eval sweeps, and verification runs belong on `gpu-server`, on GPU, wrapped — not on
workstation CPU. Two reasons, and the first gets forgotten: **workstation CPU tests are slow**,
and while they grind, three RTX 3090s sit idle. The second is the crash class: unwrapped jobs on
the display box are unbounded and unattributable. See
`.claude/rules/10-compute-allocation.md` for the wrapper.

**"It was only the fixture DB" is not an exemption.** The rule is about *where* and *how much*,
not *which database*.

Do not fan out parallel subagents that each run suites or sweeps — load multiplies invisibly
because every agent looks small on its own. Factory run 40 put 8 unwrapped agents on the
workstation and drove load average to 17.6 while `gpu-server` idled. Cap concurrency deliberately.

## Instrument discipline

Every result names its instrument and its framework epoch — a number without an instrument is not
a result. See the `experiments/LEDGER.md` preamble (TASK_882 instrument discipline, and the
E-a…E-f epoch list) before citing any verdict as "closed".
