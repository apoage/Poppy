#!/bin/bash
# Test 2-instance parallel training on single GPU

# Enable dynamic GPU memory allocation (share GPU memory between processes)
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_FORCE_GPU_ALLOW_GROWTH=true

echo "=========================================="
echo "LAUNCHING 2 PARALLEL TRAINING INSTANCES"
echo "=========================================="
echo ""
echo "Instance 1: name=test_dual_1, checkpoint=jax_ckpt_fast/test_dual_1"
echo "Instance 2: name=test_dual_2, checkpoint=jax_ckpt_fast/test_dual_2"
echo ""
echo "Monitor GPU: nvidia-smi -l 1"
echo "Monitor logs: tail -f logs/test_dual_*.log"
echo ""

# Launch instance 1
python train_jax_fast.py \
  --config config/temp_schedule_p3_run1.json \
  --epochs 20 \
  --steps 1440 \
  --mode MARGIN \
  --fee 0 \
  --slippage 0 \
  --sims 32 \
  --noise 0.5 \
  --noise-frac 0.5 \
  --batch-size 512 \
  --name test_dual_1 \
  > logs/test_dual_1.log 2>&1 &

INSTANCE1_PID=$!
echo "✅ Instance 1 launched (PID: $INSTANCE1_PID)"

# Stagger launch by 5 seconds to avoid initialization collision
sleep 5

# Launch instance 2 with different seed
# Note: We can't easily change seed via CLI, but different checkpoint dir helps
python train_jax_fast.py \
  --config config/temp_schedule_p3_run1.json \
  --epochs 20 \
  --steps 1440 \
  --mode MARGIN \
  --fee 0 \
  --slippage 0 \
  --sims 32 \
  --noise 0.5 \
  --noise-frac 0.5 \
  --batch-size 512 \
  --name test_dual_2 \
  > logs/test_dual_2.log 2>&1 &

INSTANCE2_PID=$!
echo "✅ Instance 2 launched (PID: $INSTANCE2_PID)"

echo ""
echo "Both instances running in background"
echo "  Instance 1 PID: $INSTANCE1_PID"
echo "  Instance 2 PID: $INSTANCE2_PID"
echo ""
echo "Monitor with:"
echo "  nvidia-smi -l 1              # GPU usage"
echo "  tail -f logs/test_dual_*.log # Training logs"
echo "  ps aux | grep train_jax      # Process status"
echo ""
echo "To stop:"
echo "  kill $INSTANCE1_PID $INSTANCE2_PID"
echo ""

# Wait for both to complete
wait $INSTANCE1_PID
echo "✅ Instance 1 completed"

wait $INSTANCE2_PID
echo "✅ Instance 2 completed"

echo ""
echo "=========================================="
echo "DUAL INSTANCE TEST COMPLETE"
echo "=========================================="
