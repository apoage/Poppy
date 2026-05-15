#!/usr/bin/env bash
# Smoke test wrapper for TASK_749 Titans-MAG.
# Run from repo root.

set -euo pipefail

cd "$(dirname "$0")/../.."

mkdir -p runs/000_smoke/titans_smoke

CUDA_VISIBLE_DEVICES=0 python train_phase2.py \
  --stage B --epochs 10 --steps-per-epoch 50 --batch-size 64 \
  --periods-file config/foxglove_periods.json \
  --n-slots 4 --max-runs-per-group 1 \
  --factored --factored-temperature 0.1 \
  --no-reset-value-head --value-target-mode nstep --td-discount 0.95 \
  --value-weight 0.5 --bc-weight-min 0.05 --bc-max-trade-weight 4.0 \
  --direction-weight 0.5 --encoder-lr 3e-5 \
  --encoder-version 3 --embed-dim 64 --encoder-heads 8 \
  --checkpoint jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl \
  --rank-loss mse --rank-weight 1.0 --aux-price-weight 0.1 \
  --titans-memory --titans-variant mag --titans-memory-depth 2 \
  --titans-persistent-tokens 4 --reset-titans-memory \
  --output-dir runs/000_smoke/titans_smoke \
  --name 000_titans_smoke 2>&1 | tee runs/000_smoke/titans_smoke/train.log
