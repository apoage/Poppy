#!/bin/bash
# A1+B1 stacked test (per lore/notes_2026-04-24.md §67):
#   A1 = dense_sign oracle (ε=7.5e-3, calibrated)
#   B1 = drop BC entirely from Stage B (--bc-weight-min 0.0, --bc-max-trade-weight 0.0)
#
# Decision gate (TASK_733 framework):
#   dir_acc > 0.62 → label/loss reformulation still alive; ship path.
#   dir_acc ≤ 0.58 → architecture is dominant bottleneck → escalate to TASK_741.

set -eu
set -o pipefail

cd "$(dirname "$0")/.."
source trader_env/bin/activate
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7

PHASE1_CKPT=jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl
EPOCHS=200
OUT_DIR=jax_ckpt_easy_phase2_a1b1
SUMMARY=task_a1b1_result.md
TAG=a1b1
SIGN_EPS=7.5e-3

if [[ ! -f "${PHASE1_CKPT}" ]]; then
  echo "ERROR: Phase 1 checkpoint missing: ${PHASE1_CKPT}"
  exit 2
fi
if systemctl is-active --quiet reth; then
  echo "ERROR: reth is still running — stop it first."
  exit 2
fi

mkdir -p "${OUT_DIR}"

COMMON_ARGS=(
  --checkpoint "${PHASE1_CKPT}"
  --stage B --epochs "${EPOCHS}"
  --periods-file config/foxglove_periods.json
  --n-slots 17 --max-runs-per-group 10
  --factored --factored-temperature 0.1
  --no-reset-value-head --value-target-mode nstep --td-discount 0.95
  --value-weight 0.5
  --bc-weight-min 0.0 --bc-max-trade-weight 0.0
  --direction-weight 0.5 --encoder-lr 3e-5
  --encoder-version 3 --embed-dim 64 --encoder-heads 8
  --rank-loss mse --rank-weight 1.0
  --aux-price-weight 0.1
  --oracle-mode dense_sign --sign-epsilon "${SIGN_EPS}"
)

{
  echo "# A1+B1 Stacked Result: dense_sign oracle + no BC"
  echo ""
  echo "Reference: TASK_736 Alpha +3.87% / Bal.Acc 26.5% / 6/6"
  echo "Config: TASK_736 baseline + dense_sign(ε=${SIGN_EPS}) + bc-weight-min=0"
  echo ""
  echo "| Tag | Bal.Acc AVG | Alpha AVG | ModelPnL AVG | Beats random | Dir.Acc (final) | Policy loss (final) |"
  echo "|---|---|---|---|---|---|---|"
} > "${SUMMARY}"

echo "=================================================================================="
echo "[${TAG}] $(date '+%Y-%m-%d %H:%M:%S') — Training (A1+B1)"
echo "=================================================================================="

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
echo "A1+B1 complete. Summary at ${SUMMARY}:"
echo "=================================================================================="
cat "${SUMMARY}"
echo ""
echo "Decision gate: dir_acc > 0.62 → label-side wins; ≤ 0.58 → escalate to TASK_741."
