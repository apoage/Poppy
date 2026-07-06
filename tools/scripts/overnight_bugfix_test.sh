#!/bin/bash
# ============================================================
# OVERNIGHT BUG FIX VALIDATION TEST SUITE
# ============================================================
# Total budget: 7 hours (420 minutes)
# Created: 2026-02-01
#
# Tests Phase 1 + Phase 2 bug fixes:
#   - BC masked mean, PER weights, PopArt fix
#   - Consistency loss, MCTS normalization
#   - Demo priority scaling, stratified sampling
#   - Mixed replay, gradient blocking, burn sign fix
#   - Weight decay, reduced grad clip
# ============================================================

set -e  # Exit on error

# Configuration
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/overnight_${TIMESTAMP}"
RESULTS_FILE="${LOG_DIR}/summary.txt"

mkdir -p "$LOG_DIR"

echo "============================================================" | tee "$RESULTS_FILE"
echo "OVERNIGHT BUG FIX TEST SUITE - Started $(date)" | tee -a "$RESULTS_FILE"
echo "Log directory: $LOG_DIR" | tee -a "$RESULTS_FILE"
echo "============================================================" | tee -a "$RESULTS_FILE"

# Helper function to run a test
run_test() {
    local name=$1
    local duration=$2
    shift 2
    local args="$@"

    echo "" | tee -a "$RESULTS_FILE"
    echo "------------------------------------------------------------" | tee -a "$RESULTS_FILE"
    echo "TEST: $name (~${duration} min)" | tee -a "$RESULTS_FILE"
    echo "Started: $(date)" | tee -a "$RESULTS_FILE"
    echo "Args: $args" | tee -a "$RESULTS_FILE"
    echo "------------------------------------------------------------" | tee -a "$RESULTS_FILE"

    local start_time=$(date +%s)

    # Run training, capture output
    python train_jax_fast.py $args \
        2>&1 | tee "${LOG_DIR}/${name}.log"

    local exit_code=${PIPESTATUS[0]}
    local end_time=$(date +%s)
    local elapsed=$(( (end_time - start_time) / 60 ))

    if [ $exit_code -eq 0 ]; then
        echo "RESULT: SUCCESS (${elapsed} min)" | tee -a "$RESULTS_FILE"
    else
        echo "RESULT: FAILED (exit code $exit_code, ${elapsed} min)" | tee -a "$RESULTS_FILE"
    fi

    # Copy training CSV if exists
    if [ -f "logs/training_${name}.csv" ]; then
        cp "logs/training_${name}.csv" "${LOG_DIR}/"
    fi

    echo "Finished: $(date)" | tee -a "$RESULTS_FILE"
}

# ============================================================
# TEST 1: Smoke Test (15 min)
# Quick sanity check that training doesn't crash
# ============================================================
run_test "1_smoke_test" 15 \
    --name "bugfix_1_smoke_${TIMESTAMP}" \
    --synthetic-level 0 \
    --epochs 30 \
    --steps 100 \
    --parallel-envs 4 \
    --jax-env \
    --batch-size 256 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --global-grad-clip 5.0

# ============================================================
# TEST 2: Imprint Test (45 min)
# Test BC imprinting with all fixes
# ============================================================
run_test "2_imprint_test" 45 \
    --name "bugfix_2_imprint_${TIMESTAMP}" \
    --synthetic-level 0 \
    --epochs 60 \
    --steps 200 \
    --parallel-envs 4 \
    --jax-env \
    --batch-size 512 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --global-grad-clip 5.0 \
    --imprint

# ============================================================
# TEST 3: Full Features (60 min)
# Test with prediction loss + burning money
# ============================================================
run_test "3_full_features" 60 \
    --name "bugfix_3_full_${TIMESTAMP}" \
    --synthetic-level 0 \
    --epochs 80 \
    --steps 200 \
    --parallel-envs 4 \
    --jax-env \
    --batch-size 512 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --global-grad-clip 5.0 \
    --use-prediction-loss \
    --burning-money \
    --imprint

# ============================================================
# TEST 4: From Checkpoint (90 min)
# Continue from batch8 checkpoint with fixes
# ============================================================
if [ -d "jax_ckpt_fast/popy_v3e_v1_historical_batch8" ]; then
    run_test "4_checkpoint_resume" 90 \
        --load-checkpoint jax_ckpt_fast/popy_v3e_v1_historical_batch8 \
        --name "bugfix_4_ckpt_${TIMESTAMP}" \
        --synthetic-level 0 \
        --epochs 100 \
        --steps 400 \
        --parallel-envs 4 \
        --jax-env \
        --weight-decay 1e-4 \
        --global-grad-clip 5.0 \
        --imprint
else
    echo "SKIP: Test 4 - Checkpoint not found" | tee -a "$RESULTS_FILE"
fi

# ============================================================
# TEST 5: Extended Training (120 min)
# Longer run to see if metrics improve over time
# ============================================================
run_test "5_extended" 120 \
    --name "bugfix_5_extended_${TIMESTAMP}" \
    --synthetic-level 0 \
    --epochs 150 \
    --steps 300 \
    --parallel-envs 4 \
    --jax-env \
    --batch-size 512 \
    --lr 1e-4 \
    --lr-schedule cosine \
    --lr-min 1e-5 \
    --weight-decay 1e-4 \
    --global-grad-clip 5.0 \
    --use-prediction-loss \
    --imprint

# ============================================================
# TEST 6: Aggressive LR (45 min)
# Test with higher learning rate
# ============================================================
run_test "6_aggressive_lr" 45 \
    --name "bugfix_6_aggr_${TIMESTAMP}" \
    --synthetic-level 0 \
    --epochs 60 \
    --steps 200 \
    --parallel-envs 4 \
    --jax-env \
    --batch-size 512 \
    --lr 3e-4 \
    --weight-decay 1e-4 \
    --global-grad-clip 5.0 \
    --imprint

# ============================================================
# SUMMARY
# ============================================================
echo "" | tee -a "$RESULTS_FILE"
echo "============================================================" | tee -a "$RESULTS_FILE"
echo "ALL TESTS COMPLETE - $(date)" | tee -a "$RESULTS_FILE"
echo "============================================================" | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"
echo "Results saved to: $LOG_DIR" | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"
echo "To review results:" | tee -a "$RESULTS_FILE"
echo "  cat ${RESULTS_FILE}" | tee -a "$RESULTS_FILE"
echo "  ls -la ${LOG_DIR}/" | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"
echo "Key metrics to check in each CSV:" | tee -a "$RESULTS_FILE"
echo "  - bc_loss: Should be >0.01 (not ~1e-05)" | tee -a "$RESULTS_FILE"
echo "  - policy_entropy: Should stay 3-5 (not collapse)" | tee -a "$RESULTS_FILE"
echo "  - reward_ma20: Should trend upward" | tee -a "$RESULTS_FILE"
echo "  - alpha: Should improve over epochs" | tee -a "$RESULTS_FILE"
echo "============================================================" | tee -a "$RESULTS_FILE"

# Generate quick comparison report
echo "" | tee -a "$RESULTS_FILE"
echo "QUICK METRICS COMPARISON:" | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"

for csv in ${LOG_DIR}/*.csv; do
    if [ -f "$csv" ]; then
        name=$(basename "$csv" .csv)
        echo "--- $name ---" | tee -a "$RESULTS_FILE"
        # Get last 5 rows of key metrics
        tail -5 "$csv" | cut -d',' -f1,2,3,12,24 | head -6 | tee -a "$RESULTS_FILE"
        echo "" | tee -a "$RESULTS_FILE"
    fi
done

echo "Done! Total runtime: $SECONDS seconds ($(( SECONDS / 60 )) minutes)"
