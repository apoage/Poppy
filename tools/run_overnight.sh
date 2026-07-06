#!/bin/bash
# Overnight runner: TASK_739 (dense sign oracle) → TASK_737 ablation suite.
#
# Rationale: launch TASK_739 first. It is the cheapest new code path (~2.2 hr)
# and a new oracle integration — catches wiring failures within the first ~15
# minutes of pool build. If it OOMs or crashes, we still have ~8 hr left in the
# window to attempt TASK_737 independently. TASK_737 is the slower (~8.5 hr)
# four-variant ablation matrix and it does not depend on TASK_739.
#
# Total wall-clock: ~10.5 hr. Safe to launch at 19:00–21:00 for overnight.
#
# Usage:
#   bash tools/run_overnight.sh > overnight_$(date +%F).log 2>&1 &
#
# Preconditions (enforced below + in the sub-scripts):
#   - reth + lighthouse stopped
#   - Both P1 checkpoints present (32-ch xs32 for 739/most of 737; 27-ch v3 for 737 legacy_27ch)
#   - ~25 GB free disk (checkpoints + logs across 5 training runs)

set -eu
set -o pipefail

cd "$(dirname "$0")/.."

START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
START_EPOCH=$(date +%s)

echo "=================================================================================="
echo "Overnight runner — TASK_739 → TASK_737"
echo "Started: ${START_TIME}"
echo "=================================================================================="

# --- Preconditions (fail fast, before touching either sub-script) ---
if systemctl is-active --quiet reth; then
[redacted: host service detail]
  exit 2
fi
if systemctl is-active --quiet lighthouse; then
[redacted: host service detail]
  exit 2
fi
if [[ ! -f jax_ckpt_easy_phase1_xs32/easy_medium/phase1_best.pkl ]]; then
  echo "ERROR: 32-ch Phase 1 checkpoint missing (required for 739 and 737 runs 1–3)."
  exit 2
fi
if [[ ! -f jax_ckpt_easy_phase1_v3/easy_medium/phase1_best.pkl ]]; then
  echo "ERROR: 27-ch V3 Phase 1 checkpoint missing (required for 737 legacy_27ch)."
  exit 2
fi
if [[ ! -x tools/run_739_dense_sign.sh ]]; then
  echo "ERROR: tools/run_739_dense_sign.sh missing or not executable."
  exit 2
fi
if [[ ! -x tools/run_737_ablations.sh ]]; then
  echo "ERROR: tools/run_737_ablations.sh missing or not executable."
  exit 2
fi

# --- Phase 1/2: TASK_739 (fastest, ~2.2 hr) ---
echo ""
echo "=================================================================================="
echo "Phase 1/2: TASK_739 — dense sign oracle (Tier A1)  @  $(date '+%H:%M:%S')"
echo "=================================================================================="
PHASE_739_START=$(date +%s)
if bash tools/run_739_dense_sign.sh; then
  PHASE_739_STATUS="OK"
else
  PHASE_739_STATUS="FAILED"
  echo "WARNING: TASK_739 failed. Continuing with TASK_737 suite anyway — it is independent."
fi
PHASE_739_DURATION=$(( $(date +%s) - PHASE_739_START ))

# Brief pause so the OS releases VRAM before the next JAX process allocates.
sleep 30

# --- Phase 2/2: TASK_737 suite (~8.5 hr, 4 variants) ---
echo ""
echo "=================================================================================="
echo "Phase 2/2: TASK_737 — ablation suite (4 runs)  @  $(date '+%H:%M:%S')"
echo "=================================================================================="
PHASE_737_START=$(date +%s)
if bash tools/run_737_ablations.sh; then
  PHASE_737_STATUS="OK"
else
  PHASE_737_STATUS="FAILED"
fi
PHASE_737_DURATION=$(( $(date +%s) - PHASE_737_START ))

# --- Summary ---
END_TIME=$(date '+%Y-%m-%d %H:%M:%S')
TOTAL_DURATION=$(( $(date +%s) - START_EPOCH ))

fmt_duration() {
  local s=$1
  printf '%dh%02dm' $(( s / 3600 )) $(( (s % 3600) / 60 ))
}

echo ""
echo "=================================================================================="
echo "Overnight runner complete"
echo "=================================================================================="
echo "Started:       ${START_TIME}"
echo "Finished:      ${END_TIME}"
echo "Total:         $(fmt_duration ${TOTAL_DURATION})"
echo ""
echo "TASK_739:      ${PHASE_739_STATUS}  ($(fmt_duration ${PHASE_739_DURATION}))  → task_739_result.md"
echo "TASK_737:      ${PHASE_737_STATUS}  ($(fmt_duration ${PHASE_737_DURATION}))  → ablation_737_summary.md"
echo ""
echo "Checkpoints + logs:"
ls -d jax_ckpt_easy_phase2_739_* jax_ckpt_easy_phase2_737_* 2>/dev/null | sed 's/^/  /'
echo ""
if [[ "${PHASE_739_STATUS}" == "OK" && "${PHASE_737_STATUS}" == "OK" ]]; then
  exit 0
else
  echo "NOTE: One or both phases failed — review logs above."
  exit 1
fi
