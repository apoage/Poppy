#!/bin/bash
# Production Training Commands
# Phase 1: Large-Scale Pre-Training
# 64 assets, 1440 steps (24h episodes), 10,000 epochs

echo "======================================================================"
echo "MCTS Trading Agent - Phase 1 Pre-Training"
echo "======================================================================"
echo ""
echo "Configuration:"
echo "  Assets: 64 (random, excluding forex)"
echo "  Date Range: 2024-01-01 to 2025-12-12 (~2 years)"
echo "  Episode Length: 1440 steps (24 hours)"
echo "  Epochs: 10,000"
echo "  Min Coverage: 75%"
echo "  Estimated Time: ~4 hours"
echo ""
echo "======================================================================"
echo ""

# Phase 1: Pre-Training
python train_manager.py \
  --random-subset 64 \
  --exclude-forex \
  --date-start 2024-01-01 \
  --date-end 2025-12-12 \
  --min-candles 262800 \
  --min-coverage 0.75 \
  --steps 1440 \
  --epochs 10000 \
  --no-validate

echo ""
echo "======================================================================"
echo "Phase 1 Training Complete!"
echo "======================================================================"
echo ""
echo "Next Steps:"
echo "  1. Check logs: tail logs/train_manager.log"
echo "  2. Verify checkpoints: ls -lh jax_ckpt_fast/"
echo "  3. Run Phase 2: bash START_TRAINING_PHASE2.sh"
echo ""
