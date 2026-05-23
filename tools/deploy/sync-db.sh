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
# .zst + .sha256; the operator decompresses to a trading_data.db.NEW staging
# file, re-verifies integrity_check on the staged file, then atomically renames
# into place when training is idle (the printed snippet at the end uses this
# atomic-rename flow). The server's live trading_data.db is never overwritten
# in-place — a failed staged integrity_check leaves the live DB untouched.
#
# Usage:
#   tools/deploy/sync-db.sh                 # full: integrity check → compress → rsync
#   tools/deploy/sync-db.sh --check-only    # workstation integrity check only
#   tools/deploy/sync-db.sh --skip-compress # use existing .zst (re-sync only)
#   tools/deploy/sync-db.sh --force-stale   # allow --skip-compress with .zst older than .db
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
FORCE_STALE=""
DRY=""
for arg in "$@"; do
  case "$arg" in
    --check-only)    CHECK_ONLY=1 ;;
    --skip-compress) SKIP_COMPRESS=1 ;;
    --force-stale)   FORCE_STALE=1 ;;
    --dry-run)       DRY=1 ;;
    -h|--help)       sed -n '2,24p' "$0"; exit 0 ;;
    -*)              echo "unknown flag: $arg" >&2; exit 2 ;;
  esac
done

# ---- 0. Carry-over cleanup -----------------------------------------------
# A prior crash mid-compress may have left a partial .zst.tmp on disk. Remove
# it before this run's compress so the atomic .tmp → .zst rename below is the
# only path that can produce the final $ZST.
if [[ -f "$ZST.tmp" ]]; then
  echo "==> Removing stale $ZST.tmp from a prior interrupted compress"
  [[ -n "$DRY" ]] || rm -f "$ZST.tmp"
fi

# ---- 1. Source-side integrity check (THE gap closed) ---------------------
[[ -f "$DB" ]] || { echo "❌ $DB not found at $(pwd)" >&2; exit 1; }
SIZE="$(du -h "$DB" | cut -f1)"

# di/F4: WAL checkpoint before integrity_check + compress. No-op on
# non-WAL DBs. Bites silently if the DB is ever in WAL mode (recent commits
# live in trading_data.db-wal until checkpointed; compressing the bare .db
# loses them).
echo "==> WAL checkpoint (no-op if not in WAL mode)..."
[[ -n "$DRY" ]] || sqlite3 "$DB" 'PRAGMA wal_checkpoint(TRUNCATE);' >/dev/null

echo "==> PRAGMA integrity_check on $DB ($SIZE)..."
# di/F1: capture rc explicitly so sqlite3 itself crashing (SIGSEGV, OOM-kill,
# locked DB, missing binary) prints a diagnostic instead of a bare set -e exit.
RC=0
RESULT="$(sqlite3 "$DB" 'PRAGMA integrity_check' 2>&1)" || RC=$?
if [[ $RC -ne 0 ]]; then
  echo "❌ sqlite3 itself failed (exit=$RC) — could not run integrity_check" >&2
  echo "$RESULT" | head -20 >&2
  exit 1
fi
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
    echo "==> zstd -19 → $ZST (via $ZST.tmp + atomic rename)"
    # di/F3: compress to .zst.tmp, atomic mv on success. If zstd is ^C'd /
    # OOM-killed mid-write, the partial .zst.tmp is left but the canonical
    # $ZST is unchanged — next run's mtime check stays honest (DB still newer
    # than the old .zst, recompress triggers) and the stale .tmp is cleaned
    # up by the carry-over block above.
    if [[ -z "$DRY" ]]; then
      zstd -f --keep -19 -o "$ZST.tmp" "$DB"
      mv "$ZST.tmp" "$ZST"
    fi
  else
    echo "    $ZST newer than $DB — skipping compress (use --skip-compress to suppress this message)"
  fi
else
  # di/F5: --skip-compress with a .zst older than the .db silently ships
  # stale data. Refuse unless --force-stale is also passed.
  if [[ -f "$ZST" ]] && [[ "$DB" -nt "$ZST" ]]; then
    DB_MTIME="$(date -r "$DB" '+%F %T')"
    ZST_MTIME="$(date -r "$ZST" '+%F %T')"
    echo "⚠️  --skip-compress requested but $ZST is OLDER than $DB" >&2
    echo "    $DB  mtime: $DB_MTIME" >&2
    echo "    $ZST mtime: $ZST_MTIME" >&2
    if [[ -z "$FORCE_STALE" ]]; then
      echo "❌ Refusing to ship stale .zst. Re-run without --skip-compress, or pass --force-stale to override." >&2
      exit 1
    fi
    echo "    --force-stale set — proceeding with stale .zst." >&2
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

==> Workstation sync done. Server-side swap is MANUAL and atomic — perform it
    ONLY when training is idle (no train_phase2.py / train_easy.py reading the
    DB). The snippet below decompresses to trading_data.db.NEW, re-verifies the
    staged file, then renames into place. A failed re-verify leaves the live
    trading_data.db untouched:

  ssh -p $SRV_PORT $SRV_USER@$SRV_HOST '
    cd $SRV_PATH &&
    sha256sum -c $SHA &&
    unzstd -k -f $ZST -o $DB.NEW &&
    sqlite3 $DB.NEW "PRAGMA integrity_check" | tee /tmp/srv_integrity.out | grep -qx "ok" || { rm -f $DB.NEW; echo "❌ staged integrity_check failed"; cat /tmp/srv_integrity.out; exit 1; } &&
    echo "OK to swap — when training idle: mv $DB.NEW $DB"
  '

The server-side PRAGMA integrity_check is the second gate — it re-verifies the
SQLite-level integrity on the .NEW staging file before any rename, closing the
2026-05-12 hole on the server side too. Both gates must return "ok" before the
operator runs the final mv. The mv is atomic on the same filesystem (rename(2)).
EOF
