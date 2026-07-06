#!/bin/bash
# Test 2-instance parallel training with 15M param model

# Enable dynamic GPU memory allocation
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_FORCE_GPU_ALLOW_GROWTH=true

echo "=========================================="
echo "DUAL INSTANCE TEST - 15M PARAM MODEL"
echo "=========================================="
echo ""
echo "Model: state_dim=96 (~15M params)"
echo "Instance 1: jax_ckpt_fast/test_15M_dual_1"
echo "Instance 2: jax_ckpt_fast/test_15M_dual_2"
echo ""
echo "Monitor: nvidia-smi -l 1"
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
  --state-dim 96 \
  --name test_15M_dual_1 \
  > logs/test_15M_dual_1.log 2>&1 &

INSTANCE1_PID=$!
echo "✅ Instance 1 launched (PID: $INSTANCE1_PID)"

# Stagger by 5 seconds
sleep 5

# Launch instance 2
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
  --state-dim 96 \
  --name test_15M_dual_2 \
  > logs/test_15M_dual_2.log 2>&1 &

INSTANCE2_PID=$!
echo "✅ Instance 2 launched (PID: $INSTANCE2_PID)"

echo ""
echo "Monitoring:"
echo "  GPU:  nvidia-smi -l 1"
echo "  Logs: tail -f logs/test_15M_dual_*.log"
echo ""
echo "Stop: kill $INSTANCE1_PID $INSTANCE2_PID"
echo ""

# Wait for completion
wait $INSTANCE1_PID
echo "✅ Instance 1 completed"

wait $INSTANCE2_PID
echo "✅ Instance 2 completed"

echo ""
echo "=========================================="
echo "TEST COMPLETE - Check VRAM & GPU usage"
echo "=========================================="
