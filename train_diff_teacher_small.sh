#!/bin/bash
# Small Diff Teacher - fits in 12GB alongside synthetic training
# Adversarial market generator for curriculum

echo "=== Diff Teacher Small Model Training ==="
echo "Target: Basic adversarial generator"
echo "Window: 100 steps (fits in 12GB VRAM)"
echo ""

python -m diff_teacher.train \
    --rainbow \
    --db-path trading_data.db \
    --schedule-path training_schedule.json \
    \
    --window-size 100 \
    --latent-dim 64 \
    --encoder-layers 2 \
    --flow-layers 3 \
    \
    --batch-size 16 \
    --windows-per-epoch 10000 \
    --batches-per-period 50 \
    \
    --encoder-epochs 20 \
    --generator-epochs 40 \
    \
    --checkpoint-dir checkpoints/diff_teacher_small \
    --checkpoint-every 5 \
    --log-every 50 \
    \
    "$@"
