#!/usr/bin/env bash
# Deploy workstation working tree to gpu-server.
#
# Two transports:
#   1. git push training branch  — code, agent context, experiment specs
#   2. rsync push                 — Bella checkpoints (large, not in git)
#
# Reads working tree, NOT HEAD — uncommitted master state ships fine.
#
# Usage:
#   tools/deploy/deploy.sh              # full deploy (git + checkpoints)
#   tools/deploy/deploy.sh --dry-run    # show what would happen
#   tools/deploy/deploy.sh --code-only  # skip Bella checkpoint rsync
#   tools/deploy/deploy.sh --memory     # also sync ~/.claude project memory

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

WORKTREE="$REPO_ROOT/.git-training-worktree"
EXCLUDE_FILE="$REPO_ROOT/tools/deploy/.training-exclude"
SRV="gpu-server"
SRV_PATH="Trader"   # relative to operator home
DRY=""
CODE_ONLY=0
SYNC_MEMORY=0

for arg in "$@"; do
  case "$arg" in
    --dry-run)   DRY="--dry-run" ;;
    --code-only) CODE_ONLY=1 ;;
    --memory)    SYNC_MEMORY=1 ;;
    *) echo "unknown arg: $arg"; exit 2 ;;
  esac
done

echo "==> repo:      $REPO_ROOT"
echo "==> worktree:  $WORKTREE"
echo "==> server:    $SRV:$SRV_PATH"
[[ -n "$DRY" ]] && echo "==> DRY RUN"

# 1. Ensure git remote `srv` points at gpu-server (idempotent).
if ! git remote get-url srv >/dev/null 2>&1; then
  echo "==> adding git remote srv → ssh://gpu-server/~/Trader"
  git remote add srv "ssh://$SRV/~/$SRV_PATH"
fi
SRV_REMOTE="srv"

# 2. Ensure training-branch worktree exists
if [[ ! -d "$WORKTREE" ]]; then
  echo "==> creating training-branch worktree (first run)"
  git worktree add -B training "$WORKTREE" 2>/dev/null || \
    git worktree add "$WORKTREE" training
fi

# 2. Stage working tree → worktree via rsync, honoring excludes.
#    --delete prunes files removed from workstation. .git is excluded so the
#    worktree's git symlink stays intact.
echo "==> staging working tree → training worktree"
# -v on dry-run for inspection; quiet during real runs (use --stats for summary)
STAGE_FLAGS="-a"
[[ -n "$DRY" ]] && STAGE_FLAGS="-av"
rsync $STAGE_FLAGS $DRY --delete --stats \
  --exclude='.git' \
  --exclude-from="$EXCLUDE_FILE" \
  "$REPO_ROOT/" "$WORKTREE/"

if [[ -n "$DRY" ]]; then
  echo "==> dry run: skipping commit + push"
else
  # 3. Commit (allow-empty so even no-change deploys advance the branch).
  cd "$WORKTREE"
  git add -A
  if git diff --cached --quiet; then
    echo "==> no changes since last deploy — committing empty marker"
    git commit --allow-empty -m "deploy $(date -Iseconds)"
  else
    git commit -m "deploy $(date -Iseconds)"
  fi

  # 4. Push training branch to server.
  echo "==> git push $SRV_REMOTE training"
  git push "$SRV_REMOTE" training
  cd "$REPO_ROOT"
fi

# 5. rsync Bella checkpoints (large, not in git).
if [[ "$CODE_ONLY" -eq 0 ]]; then
  if [[ -d "$REPO_ROOT/checkpoints/diff_teacher" ]]; then
    echo "==> rsync Bella checkpoints"
    # Server's rsync 3.1.3 has no --mkpath; pre-create the target directory.
    [[ -z "$DRY" ]] && ssh "$SRV" "mkdir -p $SRV_PATH/checkpoints/diff_teacher"
    rsync -av $DRY --partial --inplace --progress \
      "$REPO_ROOT/checkpoints/diff_teacher/" \
      "$SRV:$SRV_PATH/checkpoints/diff_teacher/"
  else
    echo "==> WARN: no checkpoints/diff_teacher/ on workstation, skipping"
  fi
fi

# 6. Optional: sync Claude project memory to server's claude project dir.
if [[ "$SYNC_MEMORY" -eq 1 ]]; then
  LOCAL_MEM="$HOME/.claude/projects/-home-lukas-Trader"
  REMOTE_MEM=".claude/projects/-home-operator-Trader"
  if [[ -d "$LOCAL_MEM" ]]; then
    echo "==> rsync Claude project memory → server"
    ssh "$SRV" "mkdir -p $REMOTE_MEM"
    rsync -av $DRY \
      --exclude='tool-results/' \
      --exclude='*.jsonl' \
      "$LOCAL_MEM/" "$SRV:$REMOTE_MEM/"
  else
    echo "==> WARN: no local claude project dir at $LOCAL_MEM"
  fi
fi

echo "==> done"
