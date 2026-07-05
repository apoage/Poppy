# Phase 2 Implementation — TASK_716a

## Context

Read these files to understand the project:

1. `CLAUDE.md` — project structure, naming conventions, efficiency rules
2. `tasks/TASK_716_PHASE2_MCTS_DYNAMICS_TRAINING.md` — full Phase 2 spec with subtask breakdown
3. `easy_config.py` — Phase 1 config (you'll extend this)
4. `easy_trainer.py` — Phase 1 trainer (reference for model creation pattern)
5. `train_easy.py` — Phase 1 training (contains SegmentPool to reuse, DO NOT modify)
6. `mcts_trader/jax_agent/muzero_model.py` — MuZeroNetworks model (DO NOT modify)

## Current State

- Phase 1 training complete at ~E8900, checkpoint at `jax_ckpt_easy_A/branch_A_historical/phase1_best.pkl`
- Model: 49.6M params, state_dim=128, action_space=320, large preset
- Dynamics layers exist in checkpoint but are ALL ZEROS (blank slate for Phase 2)
- Encoder + TRM reasoning fully trained (0% sparse)

## Task

Implement **TASK_716a** (subtask section in the spec). Create `train_phase2.py` and `phase2_config.py` — skeleton only, no training logic.

Read the task file for exact deliverables, validation criteria, and gate.
