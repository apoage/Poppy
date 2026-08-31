#!/usr/bin/env bash
# TASK_899's consumer side: sync the VPS's accumulated Kraken staging into the repo.
# Run on the WORKSTATION whenever it is awake (no schedule needed — the VPS accumulates).
# Usage: tools/pull_from_apothecary.sh [ssh-alias-or-target]   (default: apothecary)
set -euo pipefail
TARGET="${1:-apothecary}"
DEST="/home/lukas/Trader/runs/858_backfill/api_staging"
mkdir -p "$DEST"
rsync -a --ignore-existing "${TARGET}:kraken_pull/staging/" "$DEST/"
rsync -a --ignore-existing --include='DUMP_FOUND_*' --exclude='*' "${TARGET}:kraken_pull/" "$DEST/"
echo "synced to $DEST — if a DUMP_FOUND_* file appeared, grab the zip ids for retry_download.sh"
