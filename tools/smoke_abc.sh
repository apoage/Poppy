#!/bin/bash
# Smoke A/B/C test — validate regression findings
#
# GPU mapping (nvidia-smi index):
#   0 = CMP 90HX  (10 GB)
#   1 = RTX 3090  (24 GB)
#
# Strategy: B runs alone on 90HX the whole time.
#           A then C run sequentially on 3090 (can't share — JAX pre-allocates).
#
# A (3090, sequential): Control (hidden=1024, AR=True)
# B (90HX, parallel):   hidden=2048, AR=True
# C (3090, sequential): hidden=2048, AR=False
#
# 25 epochs × 100 steps × batch 128 = 320K samples each
#
# Usage: ./tools/smoke_abc.sh

set +e  # Don't exit on error — we handle failures ourselves

# Clean up children on script exit/interrupt
cleanup() {
    echo ""
    echo "[$(date +%H:%M:%S)] Cleaning up..."
    pkill -f "train_easy.py.*smoke_" 2>/dev/null
}
trap cleanup EXIT INT TERM

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

source trader_env/bin/activate

# Kill any existing training
pkill -f "train_easy.py.*smoke_" 2>/dev/null || true
sleep 2

# Clean previous smoke test outputs
rm -rf jax_ckpt_easy/smoke_A_control jax_ckpt_easy/smoke_B_hidden2048 jax_ckpt_easy/smoke_C_hidden2048_noAR
rm -f logs/smoke_A_control*.csv logs/smoke_B_hidden2048*.csv logs/smoke_C_hidden2048_noAR*.csv

echo "============================================"
echo "  SMOKE A/B/C TEST — Regression Validation"
echo "============================================"
echo ""
echo "  A (3090, first):  Control (hidden=1024, AR=True)"
echo "  B (90HX, parallel): hidden=2048, AR=True"
echo "  C (3090, second): hidden=2048, AR=False"
echo ""
echo "  25 epochs, batch=128, 100 steps/epoch"
echo "  B runs in parallel with A, then C takes A's GPU slot"
echo "============================================"
echo ""

# Common args
COMMON="--no-latent-cache --buffer-size 16 --n-envs 4 --samples-per-episode 100"

# B: hidden=2048 + AR on 90HX (runs the entire time)
echo "[$(date +%H:%M:%S)] Starting B (hidden=2048, AR=True) on 90HX..."
CUDA_VISIBLE_DEVICES=0 python train_easy.py \
    --config config/smoke_B_hidden2048.json \
    $COMMON \
    > logs/smoke_B.log 2>&1 &
PID_B=$!

sleep 3

# A: Control on 3090 (runs first)
echo "[$(date +%H:%M:%S)] Starting A (control) on 3090..."
CUDA_VISIBLE_DEVICES=1 python train_easy.py \
    --config config/smoke_A_control.json \
    $COMMON \
    > logs/smoke_A.log 2>&1 &
PID_A=$!

echo ""
echo "PIDs: A=$PID_A (3090)  B=$PID_B (90HX)"
echo "B runs in background. A runs first, then C takes its place."
echo ""
echo "Monitor:"
echo "  tail -f logs/smoke_A.log"
echo "  tail -f logs/smoke_B.log"
echo ""

# Wait for A to finish
echo "Waiting for A to finish..."
wait $PID_A
EXIT_A=$?
echo "[$(date +%H:%M:%S)] A finished (exit=$EXIT_A)"

sleep 3  # Let GPU memory clear

# C: hidden=2048 + no AR on 3090 (takes A's slot)
echo "[$(date +%H:%M:%S)] Starting C (hidden=2048, noAR) on 3090..."
CUDA_VISIBLE_DEVICES=1 python train_easy.py \
    --config config/smoke_C_hidden2048_noAR.json \
    $COMMON \
    > logs/smoke_C.log 2>&1 &
PID_C=$!

echo "PID: C=$PID_C (3090)"
echo "Monitor: tail -f logs/smoke_C.log"
echo ""

# Wait for both B and C
echo "Waiting for B and C to finish..."
wait $PID_B
EXIT_B=$?
echo "[$(date +%H:%M:%S)] B finished (exit=$EXIT_B)"

wait $PID_C
EXIT_C=$?
echo "[$(date +%H:%M:%S)] C finished (exit=$EXIT_C)"

echo ""
echo "============================================"
echo "  ALL DONE — Check CSVs for results"
echo "============================================"

# Quick summary if csvs exist
python tools/smoke_compare.py 2>/dev/null || echo "(Run tools/smoke_compare.py manually)"
