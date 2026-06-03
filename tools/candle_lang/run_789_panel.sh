#!/usr/bin/env bash
# run_789_panel.sh — self-advancing orchestrator for the TASK_789 candle-language
# PANEL (branch C). Modeled on the project's tmux + exit-gated-cell idiom: each cell
# is gated on the prior cell's exit status; a failing prereq aborts the panel.
#
# Members (design §6):
#   base-alphabet self-test  (shared prereq sanity gate — CPU)
#   L1 BPE  / L2 ABBA / L3 SAX+grammar / L5 matrix-profile   (CPU symbolic members)
#   L4 conv-VQ reconstruction tokenizer  (GPU member, runs LAST)
#
# HARD rules honored here:
#   * Every cell is wrapped in tools/safe_run.sh (cgroup RAM cap; CPU vs GPU placement).
#   * GPU cells NEVER blind `--gpu 1,2,3`: we pick ONE idle GPU verified via
#     `nvidia-smi --query-gpu=memory.used` ~0 immediately before launch (blocking #1).
#     If no GPU is idle (or nvidia-smi is missing), the L4 cell is SKIPPED, not forced.
#   * safe_run owns MEM_FRACTION via --gpu-frac; we never export PREALLOCATE.
#   * DB integrity: a one-time PRAGMA quick_check warns (non-fatal) before any cell.
#
# This is a build-on-the-workstation orchestrator. The CPU members are safe here.
# The GPU member is gated on a verified-idle GPU; on the single display-driving
# workstation 3090 that gate will (correctly) refuse and skip L4 — run it on the
# server where GPU 1/2/3 exist. Override the GPU search set with CANDLE_GPUS.
#
# Usage:
#   tools/candle_lang/run_789_panel.sh [--db PATH] [--out-dir DIR]
#                                      [--max-windows-per-asset N] [--mem 64G]
#                                      [--gpu-frac 0.7] [--skip-l4] [--no-cpu-overlap]
#
# Env overrides:
#   CANDLE_DB        DB path (config.db_path() default if --db unset)
#   CANDLE_GPUS      space/comma list of GPU indices to probe for idleness (default "1 2 3")
#   GPU_IDLE_MB      "idle" threshold in MiB of used VRAM (default 512)
set -euo pipefail

# ---- repo root: this script lives at <root>/tools/candle_lang/ ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

SAFE_RUN="$REPO_ROOT/tools/safe_run.sh"
PY="${CANDLE_PY:-python}"

# ---- defaults ----
DB=""                       # empty -> let each member fall back to config.db_path()
OUT_DIR="$REPO_ROOT/runs/789_token_c0"
MAX_WPA="20000"             # max-windows-per-asset (RAM/time bound for CPU members)
MEM="64G"                   # cgroup RAM cap per cell
GPU_FRAC="0.7"              # VRAM fraction for the L4 GPU cell
SKIP_L4=0
CPU_OVERLAP=1               # L1/L2/L3/L5 may run concurrently (they're independent)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --db)  DB="$2"; shift 2;;
    --out-dir) OUT_DIR="$2"; shift 2;;
    --max-windows-per-asset) MAX_WPA="$2"; shift 2;;
    --mem) MEM="$2"; shift 2;;
    --gpu-frac) GPU_FRAC="$2"; shift 2;;
    --skip-l4) SKIP_L4=1; shift;;
    --no-cpu-overlap) CPU_OVERLAP=0; shift;;
    -h|--help) sed -n '2,40p' "$0"; exit 0;;
    *) echo "[789] unknown arg: $1" >&2; exit 2;;
  esac
done

mkdir -p "$OUT_DIR"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

# DB to use for PRAGMA + member --db. Resolve via the package's own db_path() if unset.
if [[ -z "$DB" ]]; then
  DB="${CANDLE_DB:-$($PY -c 'from tools.candle_lang import config as C; print(C.db_path())' 2>/dev/null || echo "$REPO_ROOT/trading_data.db")}"
fi

echo "============================================================"
echo "[789] candle-language PANEL orchestrator"
echo "[789] repo      : $REPO_ROOT"
echo "[789] db        : $DB"
echo "[789] out-dir   : $OUT_DIR"
echo "[789] max-w/asset: $MAX_WPA   mem-cap: $MEM   gpu-frac: $GPU_FRAC"
echo "[789] cpu-overlap: $CPU_OVERLAP   skip-l4: $SKIP_L4"
echo "============================================================"

# A member --db flag only if we have a concrete DB path (else members use config.db_path()).
DB_FLAG=()
[[ -n "$DB" ]] && DB_FLAG=(--db "$DB")

# ------------------------------------------------------------------
# Preconditions: one-time DB PRAGMA quick_check (WARN, non-fatal).
# ------------------------------------------------------------------
precheck_db() {
  if [[ ! -f "$DB" ]]; then
    echo "[789][WARN] DB not found at '$DB' — members will fall back to config.db_path()." >&2
    return 0
  fi
  echo "[789] PRAGMA quick_check on $DB ..."
  local res
  res="$($PY - "$DB" <<'PYEOF' 2>/dev/null || true
import sqlite3, sys
db = sys.argv[1]
try:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    row = con.execute("PRAGMA quick_check").fetchone()
    con.close()
    print(row[0] if row else "no-result")
except Exception as e:
    print(f"error: {e}")
PYEOF
)"
  if [[ "$res" == "ok" ]]; then
    echo "[789] DB quick_check: ok"
  else
    echo "[789][WARN] DB quick_check NOT ok: '$res' (continuing — non-fatal)." >&2
  fi
}

# ------------------------------------------------------------------
# GPU idleness: pick ONE idle GPU. NEVER blind --gpu 1,2,3.
# Echoes the chosen index on stdout; returns non-zero if none idle.
# ------------------------------------------------------------------
pick_idle_gpu() {
  command -v nvidia-smi >/dev/null 2>&1 || { echo "[789] nvidia-smi missing" >&2; return 1; }
  local threshold="${GPU_IDLE_MB:-512}"
  local candidates="${CANDLE_GPUS:-1 2 3}"
  candidates="${candidates//,/ }"
  # Snapshot used-MiB per index once.
  local -a used
  mapfile -t used < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
  local g
  for g in $candidates; do
    if [[ -z "${used[$g]:-}" ]]; then
      echo "[789] GPU $g: no nvidia-smi reading (skip)" >&2
      continue
    fi
    local mb="${used[$g]//[[:space:]]/}"
    if [[ "$mb" =~ ^[0-9]+$ ]] && (( mb <= threshold )); then
      echo "[789] GPU $g idle (${mb} MiB used <= ${threshold}) -> selected" >&2
      echo "$g"
      return 0
    fi
    echo "[789] GPU $g busy (${mb} MiB used > ${threshold})" >&2
  done
  return 1
}

# ------------------------------------------------------------------
# Cell runner: wraps a CPU member in safe_run, tees to a log, returns its exit.
# ------------------------------------------------------------------
run_cpu_cell() {  # name out_basename module [extra args...]
  local name="$1"; local out_base="$2"; local module="$3"; shift 3
  local log="$LOG_DIR/${name}.log"
  echo "[789] >>> CPU cell $name -> $OUT_DIR/$out_base  (log: $log)"
  "$SAFE_RUN" --name "789_${name}" --cpu --mem "$MEM" -- \
    "$PY" -m "$module" "${DB_FLAG[@]}" \
      --out "$OUT_DIR/$out_base" \
      --max-windows-per-asset "$MAX_WPA" \
      "$@" 2>&1 | tee "$log"
  local rc="${PIPESTATUS[0]}"
  echo "[789] <<< CPU cell $name exit=$rc"
  return "$rc"
}

# ==================================================================
# Cell 0 — preconditions
# ==================================================================
precheck_db

# ==================================================================
# Cell 1 — base-alphabet self-test (shared prereq sanity GATE, CPU).
# Each symbolic member fits its own base alphabet via base_alphabet.fit_alphabet;
# there is no shared on-disk table to produce. This cell proves the shared symbol
# machinery is healthy (id range, determinism) before we spend time on members.
# Gating: if this fails, the symbol substrate is broken -> abort the whole panel.
# ==================================================================
echo "[789] >>> Cell 1: base-alphabet self-test (shared prereq)"
if ! "$SAFE_RUN" --name 789_base_alphabet --cpu --mem "$MEM" -- \
      "$PY" -m tools.candle_lang.base_alphabet 2>&1 | tee "$LOG_DIR/base_alphabet.log"; then
  echo "[789][FATAL] base-alphabet self-test failed — aborting panel." >&2
  exit 1
fi
echo "[789] base-alphabet self-test ok."

# ==================================================================
# Cell 2 — CPU symbolic members: L1 / L2 / L3 / L5.
# Independent; may overlap (each has its own safe_run RAM cap). The panel waits
# for ALL of them before the L4 GPU cell. Each member writes its own RESULT.md.
# ==================================================================
declare -a CPU_RC_NAMES
declare -a CPU_RC_PIDS

launch_cpu() {  # name out_basename module [extra args...]
  if (( CPU_OVERLAP )); then
    run_cpu_cell "$@" &
    CPU_RC_PIDS+=("$!")
    CPU_RC_NAMES+=("$1")
  else
    if run_cpu_cell "$@"; then
      echo "[789] $1 OK"
    else
      echo "[789][WARN] $1 FAILED (rc!=0) — continuing with remaining members." >&2
    fi
  fi
}

echo "[789] >>> Cell 2: CPU symbolic members (overlap=$CPU_OVERLAP)"
launch_cpu L1_bpe            L1_BPE_RESULT.md   tools.candle_lang.symbolic.l1_bpe
launch_cpu L2_abba          l2_abba_RESULT.md  tools.candle_lang.symbolic.l2_abba
launch_cpu L3_sax_grammar   L3_RESULT.md       tools.candle_lang.symbolic.l3_sax_grammar
launch_cpu L5_matrix_profile L5_RESULT.md      tools.candle_lang.symbolic.l5_matrix_profile

CPU_FAILED=0
if (( CPU_OVERLAP )); then
  # Join all overlapped CPU members; record failures but don't abort (independent members).
  for i in "${!CPU_RC_PIDS[@]}"; do
    pid="${CPU_RC_PIDS[$i]}"; nm="${CPU_RC_NAMES[$i]}"
    if wait "$pid"; then
      echo "[789] $nm OK"
    else
      echo "[789][WARN] $nm FAILED (rc!=0)." >&2
      CPU_FAILED=$((CPU_FAILED+1))
    fi
  done
fi
echo "[789] CPU symbolic members done (failures: ${CPU_FAILED:-0})."

# ==================================================================
# Cell 3 — L4 conv-VQ GPU member, LAST, on a VERIFIED-IDLE single GPU.
# Gated on Cell 2 completion (we are past the wait above). If no GPU is idle,
# we SKIP rather than forcing a busy/blind device.
# ==================================================================
if (( SKIP_L4 )); then
  echo "[789] L4 skipped by --skip-l4."
else
  echo "[789] >>> Cell 3: L4 GPU member — probing for an idle GPU"
  if GPU_IDX="$(pick_idle_gpu)"; then
    echo "[789] launching L4 on GPU $GPU_IDX (gpu-frac $GPU_FRAC)"
    L4_LOG="$LOG_DIR/L4.log"
    if "$SAFE_RUN" --name 789_L4 --mem "$MEM" --gpu "$GPU_IDX" --gpu-frac "$GPU_FRAC" -- \
        "$PY" -m tools.candle_lang.train_l4 "${DB_FLAG[@]}" \
          --out "$OUT_DIR/L4_RESULT.md" \
          --max-windows-per-asset "$MAX_WPA" \
          --gpu-frac "$GPU_FRAC" 2>&1 | tee "$L4_LOG"; then
      echo "[789] L4 OK"
    else
      echo "[789][WARN] L4 FAILED (rc=${PIPESTATUS[0]:-?})." >&2
    fi
  else
    echo "[789][SKIP] No idle GPU among '${CANDLE_GPUS:-1 2 3}' (or nvidia-smi unavailable)." >&2
    echo "[789][SKIP] NOT forcing a blind --gpu. Run L4 on the server where a GPU is idle:" >&2
    echo "           CANDLE_GPUS=<idle> tools/candle_lang/run_789_panel.sh --skip-l4=0" >&2
  fi
fi

echo "============================================================"
echo "[789] PANEL complete. Results in: $OUT_DIR"
ls -1 "$OUT_DIR"/*.md 2>/dev/null | sed 's/^/[789]   /' || true
echo "============================================================"
