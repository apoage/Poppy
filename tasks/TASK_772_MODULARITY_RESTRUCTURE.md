# TASK_772 — Modularity Restructure (Coding-Agent Effectiveness)

**Status**: IN PROGRESS — 2026-05-23.
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
| 772.3 | `RainbowTableLoader` out of `diff_teacher/data_loader.py` → `diff_teacher/rainbow_loader.py` | LOW | ~595 | PENDING |
| 772.4 | `compute_large_margin_loss`/`compute_q_filtered_bc_loss`/`_compute_per_head_grad_norms`/`_clip_gradients_per_head`/`compute_annealed_lambda`/`compute_bc_weight`/`compute_scheduled_sampling_prob`/`compute_neural_scale` out of `mcts_trader/jax_agent/trainer.py` → `mcts_trader/jax_agent/trainer_losses.py` | MEDIUM | ~150 | PENDING |
| 772.5 | `SegmentPool` out of `train_easy.py` → `mcts_trader/data/segment_pool.py` | MEDIUM | ~924 | PENDING |

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
