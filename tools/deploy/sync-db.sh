#!/usr/bin/env bash
# tools/deploy/sync-db.sh — workstation → gpu-server DB sync, integrity-gated.
#
# Closes the 2026-05-12 silent-corruption hole: previously the DB moved via
# zstd → rsync with a sha256 only of the COMPRESSED file. zstd faithfully
# compresses a corrupt SQLite DB and the sha256 matches — page-level
# corruption propagated to the server undetected. The fix is a source-side
# PRAGMA integrity_check, gating the entire sync.
#
# Safety: this script does NOT live-swap the DB on the server. It rsyncs the
# .zst + .sha256; the operator decompresses + re-verifies MANUALLY when
# training is idle (see the printed command at the end). The server's live
# trading_data.db must not be overwritten while train_phase2.py / train_easy.py
# are reading it.
#
# Usage:
#   tools/deploy/sync-db.sh                 # full: integrity check → compress → rsync
#   tools/deploy/sync-db.sh --check-only    # workstation integrity check only
#   tools/deploy/sync-db.sh --skip-compress # use existing .zst (re-sync only)
#   tools/deploy/sync-db.sh --dry-run       # show what would happen
#
# Env overrides:
#   SRV_PORT (22) SRV_USER (operator) SRV_HOST (<redacted-ip>) SRV_PATH (Trader)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

DB="trading_data.db"
ZST="trading_data.db.zst"
SHA="trading_data.db.zst.sha256"

CHECK_ONLY=""
SKIP_COMPRESS=""
DRY=""
for arg in "$@"; do
  case "$arg" in
    --check-only)    CHECK_ONLY=1 ;;
    --skip-compress) SKIP_COMPRESS=1 ;;
    --dry-run)       DRY=1 ;;
    -h|--help)       sed -n '2,21p' "$0"; exit 0 ;;
    -*)              echo "unknown flag: $arg" >&2; exit 2 ;;
  esac
done

# ---- 1. Source-side integrity check (THE gap closed) ---------------------
[[ -f "$DB" ]] || { echo "❌ $DB not found at $(pwd)" >&2; exit 1; }
SIZE="$(du -h "$DB" | cut -f1)"
echo "==> PRAGMA integrity_check on $DB ($SIZE)..."
RESULT="$(sqlite3 "$DB" 'PRAGMA integrity_check' 2>&1)"
if [[ "$RESULT" != "ok" ]]; then
  echo "❌ integrity_check FAILED — refusing to sync:" >&2
  echo "$RESULT" | head -20 >&2
  exit 1
fi
echo "    integrity_check: ok"

[[ -n "$CHECK_ONLY" ]] && exit 0

# ---- 2. Compress (skip if .zst is fresh) ---------------------------------
if [[ -z "$SKIP_COMPRESS" ]]; then
  if [[ ! -f "$ZST" ]] || [[ "$DB" -nt "$ZST" ]]; then
    echo "==> zstd -19 → $ZST"
    [[ -n "$DRY" ]] || zstd -f --keep -19 -o "$ZST" "$DB"
  else
    echo "    $ZST newer than $DB — skipping compress (use --skip-compress to suppress this message)"
  fi
fi
[[ -f "$ZST" ]] || { echo "❌ $ZST missing after compress step" >&2; exit 1; }

# ---- 3. Transit checksum (catches rsync corruption only — NOT source-side
#         SQLite corruption, which is what step 1 is for) -----------------
echo "==> sha256sum → $SHA"
[[ -n "$DRY" ]] || sha256sum "$ZST" > "$SHA"

# ---- 4. Rsync to server --------------------------------------------------
SRV_PORT="${SRV_PORT:-22}"
SRV_USER="${SRV_USER:-operator}"
SRV_HOST="${SRV_HOST:-<redacted-ip>}"
SRV_PATH="${SRV_PATH:-Trader}"

echo "==> rsync $ZST $SHA → $SRV_USER@$SRV_HOST:$SRV_PATH/"
if [[ -n "$DRY" ]]; then
  echo "    (dry-run: skipped)"
else
  rsync -avP --partial -e "ssh -p $SRV_PORT" \
    "$ZST" "$SHA" \
    "$SRV_USER@$SRV_HOST:$SRV_PATH/"
fi

cat <<EOF

==> Workstation sync done. Server-side live-swap is MANUAL — perform it ONLY
    when training is idle (no train_phase2.py / train_easy.py reading the DB):

  ssh -p $SRV_PORT $SRV_USER@$SRV_HOST '
    cd $SRV_PATH &&
    sha256sum -c $SHA &&
    unzstd -k -f $ZST &&
    sqlite3 $DB "PRAGMA integrity_check"
  '

The server-side PRAGMA integrity_check is the second gate — it re-verifies the
SQLite-level integrity after decompress, closing the 2026-05-12 hole on the
server side too. Both gates must return "ok" before the new DB is in use.
EOF
