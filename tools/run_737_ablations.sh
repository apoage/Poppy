#!/bin/bash
# TASK_737: Ablation suite on TASK_736 baseline.
# Runs 3 variants sequentially, benchmarks each, emits a summary at the end.
#
# Usage:
#   bash tools/run_737_ablations.sh
#
# Reference: tasks/TASK_737_ABLATION_SUITE.md
# Reference baseline: TASK_736 — Alpha +3.87% AVG, Bal.Acc 26.5%, beats-random 6/6.

set -eu
set -o pipefail

# --- Environment ---
cd "$(dirname "$0")/.."                    # repo root
source trader_env/bin/activate
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7

PHASE1_CKPT=jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl
PHASE1_CKPT_27CH=jax_ckpt_easy_phase1_v3/easy_medium/phase1_best.pkl
EPOCHS=200
SUMMARY=ablation_737_summary.md

# Sanity: check checkpoints exist and reth is stopped
if [[ ! -f "${PHASE1_CKPT}" ]]; then
  echo "ERROR: Phase 1 checkpoint missing: ${PHASE1_CKPT}"
  exit 2
fi
if [[ ! -f "${PHASE1_CKPT_27CH}" ]]; then
  echo "ERROR: 27-ch Phase 1 checkpoint missing (needed for legacy_27ch run): ${PHASE1_CKPT_27CH}"
  exit 2
fi
if systemctl is-active --quiet reth; then
[redacted: host service detail]
  exit 2
fi

# --- Shared arguments (identical to TASK_736 except the ablation flag and checkpoint) ---
# NB: --checkpoint is set per-variant so legacy_27ch can swap in the 27-ch Phase 1.
COMMON_ARGS=(
  --stage B --epochs "${EPOCHS}"
  --periods-file config/foxglove_periods.json
  --n-slots 17 --max-runs-per-group 10
  --factored --factored-temperature 0.1
  --no-reset-value-head --value-target-mode nstep --td-discount 0.95
  --value-weight 0.5
  --bc-weight-min 0.05 --bc-max-trade-weight 4.0
  --direction-weight 0.5 --encoder-lr 3e-5
  --encoder-version 3 --embed-dim 64 --encoder-heads 8
)

# --- Summary header ---
echo "# TASK_737 Ablation Summary"                                    > "${SUMMARY}"
echo ""                                                              >> "${SUMMARY}"
echo "Reference: TASK_736 Alpha +3.87% AVG / Bal.Acc 26.5% / 6/6 beats-random" >> "${SUMMARY}"
echo ""                                                              >> "${SUMMARY}"
echo "| Tag | Bal.Acc AVG | Alpha AVG | ModelPnL AVG | Beats random | Dir.Acc (final) | Policy loss (final) |" >> "${SUMMARY}"
echo "|---|---|---|---|---|---|---|"                                 >> "${SUMMARY}"

run_variant() {
  local tag="$1"
  shift
  local out_dir="jax_ckpt_easy_phase2_737_${tag}"
  local extra_args=("$@")

  mkdir -p "${out_dir}"
  echo ""
  echo "=================================================================================="
  echo "[${tag}] $(date '+%Y-%m-%d %H:%M:%S') — Training with extras: ${extra_args[*]}"
  echo "=================================================================================="

  # Training — COMMON_ARGS provides --checkpoint by default; extras may override.
  python train_phase2.py "${COMMON_ARGS[@]}" "${extra_args[@]}" \
    --output-dir "${out_dir}" \
    > "${out_dir}/stage_b_nohup.log" 2>&1 || {
      echo "ERROR: [${tag}] training failed — see ${out_dir}/stage_b_nohup.log"
      echo "| ${tag} | TRAIN_FAILED | — | — | — | — | — |" >> "${SUMMARY}"
      return 1
    }

  echo "[${tag}] $(date '+%Y-%m-%d %H:%M:%S') — Training done, benchmarking."

  # Benchmark
  python tools/benchmark_easy.py \
    --checkpoint "${out_dir}/phase2_stageB_best.pkl" \
    --preset medium \
    --json "${out_dir}/benchmark.json" \
    > "${out_dir}/benchmark.log" 2>&1 || {
      echo "ERROR: [${tag}] benchmark failed — see ${out_dir}/benchmark.log"
      echo "| ${tag} | BENCH_FAILED | — | — | — | — | — |" >> "${SUMMARY}"
      return 1
    }

  # Parse results for summary
  local bal_acc alpha pnl beats dir_acc policy_loss
  bal_acc=$(grep -E "^AVERAGE" "${out_dir}/benchmark.log" | awk '{print $2}')
  alpha=$(grep -E "^AVERAGE" "${out_dir}/benchmark.log" | awk '{print $7}')
  pnl=$(grep -E "^AVERAGE" "${out_dir}/benchmark.log" | awk '{print $5}')
  beats=$(grep -cE "^[a-z_]+ +[0-9]+\.[0-9]%.*-[0-9]+\." "${out_dir}/benchmark.log" 2>/dev/null || echo "?")
  # Easier beats-random: count "PASS" lines in success criteria section
  beats=$(grep -E "Beats random:.*[0-9]+/6" "${out_dir}/benchmark.log" | grep -oE "[0-9]+/6" | head -1)
  # Final epoch dir_acc and policy_loss
  dir_acc=$(grep -E "^[0-9-]+ [0-9:]+.*- INFO - E +[0-9]+:" "${out_dir}/stage_b_nohup.log" | tail -1 | grep -oE "acc=0\.[0-9]+" | cut -d= -f2)
  policy_loss=$(grep -E "^[0-9-]+ [0-9:]+.*- INFO - E +[0-9]+:" "${out_dir}/stage_b_nohup.log" | tail -1 | grep -oE "policy=[0-9]+\.[0-9]+" | cut -d= -f2)

  echo "[${tag}] Bal.Acc=${bal_acc}, Alpha=${alpha}, PnL=${pnl}, beats=${beats}, dir_acc=${dir_acc}, policy_loss=${policy_loss}"
  echo "| ${tag} | ${bal_acc} | ${alpha} | ${pnl} | ${beats} | ${dir_acc} | ${policy_loss} |" >> "${SUMMARY}"
}

# =====================================================================
# Run 1: no_aux — ablate TASK_735 aux price head
# =====================================================================
run_variant "no_aux" \
  --checkpoint "${PHASE1_CKPT}" \
  --rank-loss mse --rank-weight 1.0 \
  --no-aux-price-head

# =====================================================================
# Run 2: no_rank — ablate TASK_730 rank head
# =====================================================================
run_variant "no_rank" \
  --checkpoint "${PHASE1_CKPT}" \
  --no-rank-head \
  --aux-price-weight 0.1

# =====================================================================
# Run 3: minimal — ablate both aux heads (value-weight alone)
# =====================================================================
run_variant "minimal" \
  --checkpoint "${PHASE1_CKPT}" \
  --no-rank-head \
  --no-aux-price-head

# =====================================================================
# Run 4: legacy_27ch — ablate TASK_732 cross-sectional channels
# Warmstart from 27-ch V3 Phase 1 checkpoint; --legacy-27ch disables
# channels 28-32 in the encoder. Tests whether cross-sectional inputs
# contribute on top of the TASK_736 baseline once ExIt is working.
# =====================================================================
run_variant "legacy_27ch" \
  --checkpoint "${PHASE1_CKPT_27CH}" \
  --legacy-27ch \
  --rank-loss mse --rank-weight 1.0 \
  --aux-price-weight 0.1

# --- Done ---
echo ""
echo "=================================================================================="
echo "All 4 ablations complete. Summary written to ${SUMMARY}:"
echo "=================================================================================="
cat "${SUMMARY}"
echo ""
echo "Checkpoints + logs:"
ls -d jax_ckpt_easy_phase2_737_*/
