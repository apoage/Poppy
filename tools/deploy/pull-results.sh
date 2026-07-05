#!/usr/bin/env bash
# Pull training results from gpu-server to workstation.
#
# Server-resident, workstation-pulled paths:
#   runs/         — per-experiment outputs (checkpoints, notes, logs)
#   logs/         — global training logs
#   wandb/        — wandb runs (if used)
#
# These paths are gitignored on both sides — pure rsync transport.
#
# Usage:
#   tools/deploy/pull-results.sh             # pull runs/ and logs/
#   tools/deploy/pull-results.sh --dry-run
#   tools/deploy/pull-results.sh runs/2026-05-10_foo   # specific subdir

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

SRV="gpu-server"
SRV_PATH="Trader"
DRY=""
TARGETS=()

for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY="--dry-run" ;;
    -*) echo "unknown flag: $arg"; exit 2 ;;
    *)  TARGETS+=("$arg") ;;
  esac
done

if [[ ${#TARGETS[@]} -eq 0 ]]; then
  TARGETS=(runs logs)
fi

for path in "${TARGETS[@]}"; do
  echo "==> pull $path"
  mkdir -p "$REPO_ROOT/$(dirname "$path")"
  rsync -av $DRY --partial --inplace --progress \
    "$SRV:$SRV_PATH/$path/" \
    "$REPO_ROOT/$path/"
done

echo "==> done"
