# TASK_727 Smoke Test Plan — 2026-04-19

## Pre-execution state
- GPU 0 (CMP 90HX, 10GB): 13MB used, idle.
- GPU 1 (RTX 3090, 24GB): 706MB used — any other training is NOT currently on this GPU (or very light).
- Note: nvidia-smi index 1 = RTX 3090 = JAX id 0 via CUDA order.

## Planned command (smoke only, no training)
```
CUDA_VISIBLE_DEVICES=1 python train_phase2.py \
  --checkpoint jax_ckpt_easy_phase2_724c/phase2_stageB_e300.pkl \
  --stage B --test-init
```
Uses `--test-init` which runs dynamics init + single recurrent_inference, then exits.
Does NOT spin up SegmentPool or training loop. Should finish in <30s.

## What this validates
- `initialize_missing_aux_heads` runs cleanly on pre-727 checkpoint
- DirectionHead params get initialized
- MuZeroNetworks apply() works with direction_head in tree
- No shape mismatches

## What it does NOT validate
- Full train step with future_cum_returns (needs SegmentPool)
- JIT compile of new loss (needs batch)
- Direction loss gradient correctness

## Rollback if crash
Files modified:
- mcts_trader/jax_agent/muzero_model.py
- phase2_config.py
- train_phase2.py
Git: all changes uncommitted on master. `git checkout -- <file>` reverts.
