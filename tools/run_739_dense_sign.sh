#!/bin/bash
# TASK_739: Dense sign-of-forward-return oracle (Tier A1).
# Single variant: TASK_736 baseline + --oracle-mode dense_sign.
#
# Usage:
#   bash tools/run_739_dense_sign.sh
#
# Reference spec:     tasks/TASK_739_DENSE_SIGN_ORACLE.md
# Reference baseline: TASK_736 — Alpha +3.87% AVG, Bal.Acc 26.5%, beats-random 6/6.

set -eu
set -o pipefail

# --- Environment ---
cd "$(dirname "$0")/.."                    # repo root
source trader_env/bin/activate
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7

PHASE1_CKPT=jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl
EPOCHS=200
OUT_DIR=jax_ckpt_easy_phase2_739_dense_sign
SUMMARY=task_739_result.md
TAG=dense_sign
SIGN_EPS=7.5e-3   # Calibrated 2026-04-24 via tools/calibrate_739_epsilon.py: gives ~13% trade density across 3 Foxglove periods (mean std(r_fwd)≈9.2e-3). ε=1e-4 per spec default gave 68% trade density — pathological.

# --- Sanity checks ---
if [[ ! -f "${PHASE1_CKPT}" ]]; then
  echo "ERROR: Phase 1 checkpoint missing: ${PHASE1_CKPT}"
  exit 2
fi
if systemctl is-active --quiet reth; then
[redacted: host service detail]
  exit 2
fi

mkdir -p "${OUT_DIR}"

# --- Shared arguments (identical to TASK_736 except --oracle-mode + --sign-epsilon) ---
COMMON_ARGS=(
  --checkpoint "${PHASE1_CKPT}"
  --stage B --epochs "${EPOCHS}"
  --periods-file config/foxglove_periods.json
  --n-slots 17 --max-runs-per-group 10
  --factored --factored-temperature 0.1
  --no-reset-value-head --value-target-mode nstep --td-discount 0.95
  --value-weight 0.5
  --bc-weight-min 0.05 --bc-max-trade-weight 4.0
  --direction-weight 0.5 --encoder-lr 3e-5
  --encoder-version 3 --embed-dim 64 --encoder-heads 8
  --rank-loss mse --rank-weight 1.0
  --aux-price-weight 0.1
  --oracle-mode dense_sign --sign-epsilon "${SIGN_EPS}"
)

# --- Summary header ---
{
  echo "# TASK_739 Result: Dense Sign-of-Forward-Return Oracle"
  echo ""
  echo "Reference: TASK_736 Alpha +3.87% AVG / Bal.Acc 26.5% / 6/6 beats-random"
  echo "Config: TASK_736 baseline + --oracle-mode dense_sign --sign-epsilon ${SIGN_EPS}"
  echo ""
  echo "| Tag | Bal.Acc AVG | Alpha AVG | ModelPnL AVG | Beats random | Dir.Acc (final) | Policy loss (final) |"
  echo "|---|---|---|---|---|---|---|"
} > "${SUMMARY}"

echo ""
echo "=================================================================================="
echo "[${TAG}] $(date '+%Y-%m-%d %H:%M:%S') — Training (dense_sign, ε=${SIGN_EPS})"
echo "=================================================================================="

# Idempotency: re-runs overwrite phase2_stageB_*.pkl in OUT_DIR.
python train_phase2.py "${COMMON_ARGS[@]}" \
  --output-dir "${OUT_DIR}" \
  > "${OUT_DIR}/stage_b_nohup.log" 2>&1 || {
    echo "ERROR: [${TAG}] training failed — see ${OUT_DIR}/stage_b_nohup.log"
    echo "| ${TAG} | TRAIN_FAILED | — | — | — | — | — |" >> "${SUMMARY}"
    exit 1
  }

echo "[${TAG}] $(date '+%Y-%m-%d %H:%M:%S') — Training done, benchmarking."

python tools/benchmark_easy.py \
  --checkpoint "${OUT_DIR}/phase2_stageB_best.pkl" \
  --preset medium \
  --json "${OUT_DIR}/benchmark.json" \
  > "${OUT_DIR}/benchmark.log" 2>&1 || {
    echo "ERROR: [${TAG}] benchmark failed — see ${OUT_DIR}/benchmark.log"
    echo "| ${TAG} | BENCH_FAILED | — | — | — | — | — |" >> "${SUMMARY}"
    exit 1
  }

# --- Parse results (mirrors tools/run_737_ablations.sh) ---
bal_acc=$(grep -E "^AVERAGE" "${OUT_DIR}/benchmark.log" | awk '{print $2}')
alpha=$(grep -E "^AVERAGE" "${OUT_DIR}/benchmark.log" | awk '{print $7}')
pnl=$(grep -E "^AVERAGE" "${OUT_DIR}/benchmark.log" | awk '{print $5}')
beats=$(grep -E "Beats random:.*[0-9]+/6" "${OUT_DIR}/benchmark.log" | grep -oE "[0-9]+/6" | head -1)
dir_acc=$(grep -E "^[0-9-]+ [0-9:]+.*- INFO - E +[0-9]+:" "${OUT_DIR}/stage_b_nohup.log" | tail -1 | grep -oE "acc=0\.[0-9]+" | cut -d= -f2)
policy_loss=$(grep -E "^[0-9-]+ [0-9:]+.*- INFO - E +[0-9]+:" "${OUT_DIR}/stage_b_nohup.log" | tail -1 | grep -oE "policy=[0-9]+\.[0-9]+" | cut -d= -f2)

echo "[${TAG}] Bal.Acc=${bal_acc}, Alpha=${alpha}, PnL=${pnl}, beats=${beats}, dir_acc=${dir_acc}, policy_loss=${policy_loss}"
echo "| ${TAG} | ${bal_acc} | ${alpha} | ${pnl} | ${beats} | ${dir_acc} | ${policy_loss} |" >> "${SUMMARY}"

echo ""
echo "=================================================================================="
echo "TASK_739 complete. Summary written to ${SUMMARY}:"
echo "=================================================================================="
cat "${SUMMARY}"
echo ""
echo "Checkpoint + logs: ${OUT_DIR}/"
