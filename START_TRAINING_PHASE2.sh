#!/bin/bash
# Production Training Commands
# Phase 2: Fine-Tuning on Recent Data
# 64 assets, 200 steps (3.3h episodes), 2,000 epochs

echo "======================================================================"
echo "MCTS Trading Agent - Phase 2 Fine-Tuning"
echo "======================================================================"
echo ""
echo "Configuration:"
echo "  Assets: 64 (random, excluding forex)"
echo "  Date Range: 2025-10-01 to 2025-12-12 (recent 2.5 months)"
echo "  Episode Length: 200 steps (3.3 hours)"
echo "  Epochs: 2,000"
echo "  Min Coverage: 90%"
echo "  Estimated Time: ~50 minutes"
echo ""
echo "======================================================================"
echo ""

# Phase 2: Fine-Tuning
python train_manager.py \
  --random-subset 64 \
  --exclude-forex \
  --date-start 2025-10-01 \
  --date-end 2025-12-12 \
  --min-coverage 0.9 \
  --steps 200 \
  --epochs 2000 \
  --no-validate

echo ""
echo "======================================================================"
echo "Phase 2 Training Complete!"
echo "======================================================================"
echo ""
echo "Next Steps:"
echo "  1. Check logs: tail logs/train_manager.log"
echo "  2. Verify checkpoints: ls -lh jax_ckpt_fast/"
echo "  3. Validate on paper trading"
echo ""
