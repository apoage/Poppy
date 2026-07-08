#!/usr/bin/env bash
# 831-EXT — R1 robustness battery (see SPEC.md). Sequential cells inside one tmux.
# Usage (on gpu-server, inside tmux):  bash experiments/831_ext/run_battery.sh [DB] [GPU]
set -u
cd "$(dirname "$0")/../.."
source trader_env/bin/activate
DB="${1:-$HOME/Trader/trading_data.db}"
GPU="${2:-2}"    # verified-idle GPU for the K-extension cells (never GPU 0; GPU 1 has ComfyUI)
OUT=runs/831_ext; mkdir -p "$OUT"
PROBE="experiments/831_search_probe/search_probe.py"
S10=0,1,2,3,4,5,6,7,8,9
S5=0,1,2,3,4

cell() {  # cell <name> <cpu|gpu> <probe args...>
  local name="$1" mode="$2"; shift 2
  echo "[831x] $name start $(date +%F' '%T)"
  if [[ "$mode" == gpu ]]; then
    JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=true \
    tools/safe_run.sh --name "831x_$name" --mem 24G --gpu "$GPU" --gpu-frac 0.7 -- \
      python -u "$PROBE" --db "$DB" --discount 0.997 --proposal-bias 3.0 --max-decisions -1 \
      --tag "ext-$name" --out "$OUT/$name.json" "$@" > "$OUT/$name.log" 2>&1
  else
    tools/safe_run.sh --name "831x_$name" --mem 24G --cpu -- \
      python -u "$PROBE" --db "$DB" --discount 0.997 --proposal-bias 3.0 --max-decisions -1 \
      --tag "ext-$name" --out "$OUT/$name.json" "$@" > "$OUT/$name.log" 2>&1
  fi
  echo "[831x] $name rc=$? $(date +%F' '%T)"
}

# A — carried-scheme robustness: full grid, 10 seeds, full window
cell seeds10 cpu --start 2021-01-01 --end 2025-09-01 --seeds $S10 --value-rollout-steps 3
# B — per-regime: dithered-book, all K, 10 seeds (per-window corridors come free)
cell bear22 cpu --start 2022-01-01 --end 2023-01-01 --schemes dithered-book --seeds $S10 --value-rollout-steps 3
cell bull23 cpu --start 2023-01-01 --end 2024-01-01 --schemes dithered-book --seeds $S10 --value-rollout-steps 3
cell y2024  cpu --start 2024-01-01 --end 2025-01-01 --schemes dithered-book --seeds $S10 --value-rollout-steps 3
cell y2025  cpu --start 2025-01-01 --end 2025-09-01 --schemes dithered-book --seeds $S10 --value-rollout-steps 3
# C — value-horizon requirement: dithered|K16, R sweep (R=3 = the official run, not re-run)
for R in 1 2 6 12; do
  cell "r$R" cpu --start 2021-01-01 --end 2025-09-01 --schemes dithered-book --k-grid 16 --seeds $S5 --value-rollout-steps "$R"
done
# D — K extension (GPU): does K>32 keep paying?
cell k64  gpu --start 2021-01-01 --end 2025-09-01 --schemes dithered-book --k-grid 64  --seeds $S5 --value-rollout-steps 3
cell k128 gpu --start 2021-01-01 --end 2025-09-01 --schemes dithered-book --k-grid 128 --seeds $S5 --value-rollout-steps 3

date > "$OUT/BATTERY_DONE"
echo "[831x] battery complete $(date +%F' '%T)"
