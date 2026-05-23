# TASK_772 — Modularity Restructure (Coding-Agent Effectiveness)

**Status**: ✓ ALL 5 SUBSTEPS SHIPPED — 2026-05-23.

## Implementation log

Net effect: 5 splits, 2737 LOC moved out of the three largest files.

| Commit | Substep | File-1 delta | File-2 |
|---|---|---|---|
| `55404a0` | 772.1 | `train_phase2.py` 3140→2831 (-309) | `train_phase2_cli.py` NEW 347 |
| `0ffe336` | 772.2 | `train_easy.py` 3541→3383 (-158) | `train_easy_loop.py` NEW 201 |
| `04d366d` | 772.3 | `diff_teacher/data_loader.py` 1027→550 (-477) | `diff_teacher/rainbow_loader.py` NEW 507 |
| `b9060d7` | 772.4 | `mcts_trader/jax_agent/trainer.py` 1932→1634 (-298) | `mcts_trader/jax_agent/trainer_losses.py` NEW 341 |
| `65e73e0` | 772.5 | `train_easy.py` 3383→2381 (-1002) | `mcts_trader/data/segment_pool.py` NEW 1048 |

Updated file-size totals post-restructure (top 5 by LOC):
1. `train_easy.py` 2381 (was 3541, -33%) — still 🟡 cohesive (`OnTheFlyGenerator` 1255 LOC dominates; deferred Split 1.2 = HIGH risk, needs CI smoke gate)
2. `tools/orchestration/e-dojo_v1.py` 2199 (untouched — splits 3.1/3.2 deferred)
3. `train_phase2.py` 1969 (was 3140) post-771/772 — still 🟡 cohesive (`make_stage_b_train_step` 538-LOC JIT closure dominates; deferred Split 2.2 = HIGH risk)
4. `train_jax_fast.py` 1620 (untouched — cohesive `main()`, no clean split)
5. `mcts_trader/jax_agent/trainer.py` 1634 (was 1932) — still 🟡 (`_loss_fn` 495 LOC + checkpoint mixin candidates; deferred Split 4.2)

## Operator decisions deferred

- **Deploy** — all 5 commits sit on workstation `master`, untested on server. Deploy after operator review.
- **Continue with Split 2.2** (`make_stage_b_train_step` extract) — HIGH risk + needs a 50-step pre/post smoke for bit-identity. Not done this task; recommend a follow-up TASK after 763/769 results to avoid mid-flight risk.
- **Continue with Splits 1.2 / 3.1 / 3.2 / 9.1 / 10.1** — medium-effort, marginal value vs the 5 already shipped. Defer unless an agent-effectiveness issue manifests.
**Source**: `tasks/CODEBASE_MODULARITY_AUDIT_2026-05-23.md`, top-5 priority splits.
**Goal**: trim oversized files so sub-agents can target focused modules. **Bit-identical behavior required** at every step — this is a refactor, not a redesign.

## Protocol (per substep)

1. **Edit**: move designated symbols to new file; update imports in caller(s).
2. **Test**:
   - Syntax smoke: `python -c "import <new_module>"`  
   - Caller-import smoke: `python -c "import <caller_module>"` (or `python <script>.py --help` for entrypoints).
   - For MEDIUM-risk splits: additional class-construct smoke (instantiate moved class with safe args).
3. **Commit** ONLY on green.
4. **Halt** on any failure; revert; report.

No deploy this task — pure local refactor. Deploy after the chain closes (operator decides).

## Substeps (ordered low-risk → medium-risk)

| # | Split | Risk | LOC moved | Status |
|---|---|---|---:|---|
| 772.1 | `parse_args` out of `train_phase2.py` → `train_phase2_cli.py` | LOW | 332 | ✓ SHIPPED (train_phase2.py 3140→2831; tests: import/--help/argparse-parse all green) |
| 772.2 | `train_epoch`/`validate`/`_launch_benchmark`/`_check_benchmark_passed` out of `train_easy.py` → `train_easy_loop.py` | LOW | 158 | ✓ SHIPPED (train_easy.py 3541→3383; tests: import + --help + 4-symbol direct import all green) |
| 772.3 | `RainbowTableLoader` out of `diff_teacher/data_loader.py` → `diff_teacher/rainbow_loader.py` | LOW | 477 | ✓ SHIPPED (data_loader.py 1027→550; re-export shim preserves `from .data_loader import RainbowTableLoader`; tests: direct + re-export + module-resolution all green) |
| 772.4 | 4 BC/grad helpers + 2 config dataclasses + 4 schedule helpers out of `mcts_trader/jax_agent/trainer.py` → `mcts_trader/jax_agent/trainer_losses.py` | MEDIUM | 298 | ✓ SHIPPED (trainer.py 1932→1634; re-export shim preserves all 10 symbol paths; tests: import + re-export + schedule-helper values all green) |
| 772.5 | `SegmentPool` + 3 state-channel helpers out of `train_easy.py` → `mcts_trader/data/segment_pool.py` | MEDIUM | 1002 | ✓ SHIPPED (train_easy.py 3383→2381 — biggest single-file delta; 6 tests green incl. cross-file `train_phase2 --help`; JIT decorator moved with the function, not left orphan) |

## Excluded from this task

- Split 2.2 (`make_stage_b_train_step` 538-LOC JIT closure) — HIGH risk, needs CI smoke gate first; defer.
- Split 10.1 (`SyntheticMarket` 15 levels) — MEDIUM risk, wide blast radius; defer.
- All "no-action" recommendations from audit §2.6, §2.8 (`train_jax_fast.py`, `jax_encoder.py`) — cohesive, leave alone.

## Failure protocol

If any substep fails the import or argparse smoke:
1. `git checkout -- <affected files>` to revert local edits
2. Document the failure in this file under "Substep N — BLOCKED: <reason>"
3. STOP — don't attempt downstream substeps that depend on the failed move

## Implementation log

(empty — populated as substeps complete)
