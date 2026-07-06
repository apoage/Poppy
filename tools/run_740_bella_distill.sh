#!/bin/bash
# TASK_740: Bella distillation prediction head (Path β-v2).
# Single variant: TASK_736 baseline + --bella-distill-head (weight 0.1).
#
# Prerequisites (enforced below):
#   1. config/bella_preds_manifest.json exists
#   2. Every shard listed in the manifest exists on disk
#   3. reth (archive node) is not running
#   4. Phase 1 warmstart checkpoint present
#   5. Bella generator checkpoint present
#
# Usage:
#   bash tools/run_740_bella_distill.sh
#
# Reference spec:     tasks/TASK_740_BELLA_DISTILL_HEAD.md
# Reference baseline: TASK_736 — Alpha +3.87% AVG, Bal.Acc 26.5%, beats-random 6/6.

set -eu
set -o pipefail

# --- Environment ---
cd "$(dirname "$0")/.."                    # repo root
source trader_env/bin/activate
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7

PHASE1_CKPT=jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl
BELLA_CKPT=checkpoints/diff_teacher/generator_best/generator
MANIFEST=config/bella_preds_manifest.json
EPOCHS=200
OUT_DIR=jax_ckpt_easy_phase2_740_bella_distill
SUMMARY=task_740_result.md
TAG=bella_distill
BELLA_W=0.1

# --- Preconditions ---
if [[ ! -f "${PHASE1_CKPT}" ]]; then
  echo "ERROR: Phase 1 checkpoint missing: ${PHASE1_CKPT}"
  exit 2
fi
if [[ ! -d "${BELLA_CKPT}" ]]; then
  echo "ERROR: Bella checkpoint missing: ${BELLA_CKPT}"
  exit 2
fi
if [[ ! -f "${MANIFEST}" ]]; then
  echo "ERROR: Bella manifest missing: ${MANIFEST}"
  echo "       Run tools/precompute_bella_predictions.py first."
  exit 2
fi
missing_shards=$(python - <<PY
import json, sys
from pathlib import Path
m = json.load(open("${MANIFEST}"))
missing = []
for tag, entry in m.get("shards", {}).items():
    p = Path(entry["path"])
    if not p.is_absolute():
        p = Path.cwd() / p
    if not p.exists():
        missing.append(str(p))
if missing:
    for x in missing[:5]:
        print(x)
    if len(missing) > 5:
        print(f"... +{len(missing) - 5} more")
    sys.exit(1)
PY
) || {
  echo "ERROR: Bella shards listed in manifest are missing:"
  echo "${missing_shards}"
  exit 2
}
if systemctl is-active --quiet reth; then
[redacted: host service detail]
  exit 2
fi

mkdir -p "${OUT_DIR}"

# --- Shared arguments (identical to TASK_736 except --bella-distill-head) ---
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
  --bella-distill-head --bella-distill-weight "${BELLA_W}"
  --bella-preds-manifest "${MANIFEST}"
)

# --- Summary header ---
{
  echo "# TASK_740 Result: Bella Distillation Prediction Head (β-v2)"
  echo ""
  echo "Reference: TASK_736 Alpha +3.87% AVG / Bal.Acc 26.5% / 6/6 beats-random"
  echo "Config: TASK_736 baseline + --bella-distill-head --bella-distill-weight ${BELLA_W}"
  echo ""
  echo "| Tag | Bal.Acc AVG | Alpha AVG | ModelPnL AVG | Beats random | Dir.Acc (final) | Bella loss (final) |"
  echo "|---|---|---|---|---|---|---|"
} > "${SUMMARY}"

echo ""
echo "=================================================================================="
echo "[${TAG}] $(date '+%Y-%m-%d %H:%M:%S') — Training (bella_distill, w=${BELLA_W})"
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

# --- Parse results (mirrors tools/run_739_dense_sign.sh) ---
bal_acc=$(grep -E "^AVERAGE" "${OUT_DIR}/benchmark.log" | awk '{print $2}')
alpha=$(grep -E "^AVERAGE" "${OUT_DIR}/benchmark.log" | awk '{print $7}')
pnl=$(grep -E "^AVERAGE" "${OUT_DIR}/benchmark.log" | awk '{print $5}')
beats=$(grep -E "Beats random:.*[0-9]+/6" "${OUT_DIR}/benchmark.log" | grep -oE "[0-9]+/6" | head -1)
dir_acc=$(grep -E "^[0-9-]+ [0-9:]+.*- INFO - E +[0-9]+:" "${OUT_DIR}/stage_b_nohup.log" | tail -1 | grep -oE "acc=0\.[0-9]+" | cut -d= -f2)
bella_loss=$(grep -E "^[0-9-]+ [0-9:]+.*- INFO - E +[0-9]+:" "${OUT_DIR}/stage_b_nohup.log" | tail -1 | grep -oE "bella=[0-9]+\.[0-9]+" | cut -d= -f2)

echo "[${TAG}] Bal.Acc=${bal_acc}, Alpha=${alpha}, PnL=${pnl}, beats=${beats}, dir_acc=${dir_acc}, bella_loss=${bella_loss}"
echo "| ${TAG} | ${bal_acc} | ${alpha} | ${pnl} | ${beats} | ${dir_acc} | ${bella_loss} |" >> "${SUMMARY}"

echo ""
echo "=================================================================================="
echo "TASK_740 complete. Summary written to ${SUMMARY}:"
echo "=================================================================================="
cat "${SUMMARY}"
echo ""
echo "Checkpoint + logs: ${OUT_DIR}/"
