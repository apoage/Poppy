#!/usr/bin/env bash
# safe_run.sh — run a job inside a HARD resource envelope so a runaway can never
# OOM/crash the host. Wraps the command in a systemd-run --user cgroup scope with a
# RAM cap (cgroup-OOM kills ONLY this job at MemoryMax), and sets GPU/CPU placement.
#
# Why: 2026-05-31 a heavy JAX job on the workstation's display GPU hard-locked the box.
# Lesson: bound every job's RAM (cgroup), and keep GPU compute OFF the workstation's
# display-driving 3090 (use --cpu here, or run on the server). Works on workstation AND
# gpu-server (both cgroup v2 + systemd --user).
#
# Usage:
#   tools/safe_run.sh --name <tag> --mem 16G [--swap 0] [--cpu | --gpu N [--gpu-frac 0.7]] -- <cmd...>
#
# Examples:
#   # CPU-only, hard 16G RAM cap (no swap thrash) — safe on the workstation:
#   tools/safe_run.sh --name c0_gate --mem 16G --cpu -- python tools/tokenizer_c0/run_gate_numpy.py
#   # Server GPU cell, 40G RAM cap, GPU 1 at 0.7 VRAM:
#   tools/safe_run.sh --name 787_s7 --mem 40G --gpu 1 --gpu-frac 0.7 -- python -u train_phase2.py ...
set -euo pipefail

NAME="safe_run"; MEM="16G"; SWAP="0"; MODE="gpu"; GPU="0"; GPUFRAC="0.7"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) NAME="$2"; shift 2;;
    --mem)  MEM="$2"; shift 2;;
    --swap) SWAP="$2"; shift 2;;
    --cpu)  MODE="cpu"; shift;;
    --gpu)  MODE="gpu"; GPU="$2"; shift 2;;
    --gpu-frac) GPUFRAC="$2"; shift 2;;
    --) shift; break;;
    *) echo "safe_run: unknown arg $1" >&2; exit 2;;
  esac
done
[[ $# -eq 0 ]] && { echo "safe_run: no command after --" >&2; exit 2; }

# MemoryHigh = soft throttle a touch below the hard cap so we throttle before the kill.
HIGH="$MEM"
ENVV=()
if [[ "$MODE" == "cpu" ]]; then
  ENVV=(JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=)
  echo "[safe_run] $NAME | CPU-only | RAM cap $MEM (swap $SWAP)"
else
  ENVV=(CUDA_VISIBLE_DEVICES="$GPU" XLA_PYTHON_CLIENT_MEM_FRACTION="$GPUFRAC")
  echo "[safe_run] $NAME | GPU $GPU (VRAM frac $GPUFRAC) | RAM cap $MEM (swap $SWAP)"
fi

# --scope runs in the foreground (so tee/logging and orchestrator gating work).
# MemorySwapMax=0 by default → bound RAM hits MemoryMax and the cgroup OOM-killer
# reaps ONLY this scope, instead of thrashing 128G swap and freezing the host.
exec systemd-run --user --scope --quiet \
  --unit="${NAME}_$$" \
  -p MemoryMax="$MEM" -p MemoryHigh="$HIGH" -p MemorySwapMax="$SWAP" \
  env "${ENVV[@]}" "$@"
