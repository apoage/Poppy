#!/usr/bin/env bash
# One-time server bootstrap. Run from workstation:
#   tools/deploy/bootstrap-server.sh
#
# Performs on gpu-server:
#   - git init in ~/Trader (if not already)
#   - configures receive.denyCurrentBranch=updateInstead so git push training works
#   - creates initial training branch checkout
#   - sanity checks (python, GPU)
#
# Idempotent: safe to re-run. Does NOT build the venv (do that after first deploy
# so requirements.lock.txt is on server).

set -euo pipefail

SRV="gpu-server"
SRV_PATH="Trader"

echo "==> server-side state check"
ssh "$SRV" "cd $SRV_PATH 2>/dev/null && pwd && ls -d .git 2>/dev/null && echo 'git OK' || echo 'no git yet'"

echo "==> initializing git on server (idempotent)"
ssh "$SRV" bash -s <<'REMOTE'
set -euo pipefail
cd ~/Trader

if [[ ! -d .git ]]; then
  echo "  -> git init"
  git init -q
  # Older git doesn't support `init -b`; set HEAD manually so the first push
  # to `training` lands on the current branch (required for updateInstead).
  git symbolic-ref HEAD refs/heads/training
  git config user.email "gpu-server@local"
  git config user.name  "gpu-server"
fi

# Always (re)set this — it's the key to push-into-working-tree.
git config receive.denyCurrentBranch updateInstead

# Allow the deploy push to land on training even if we have other branches.
# The first push from workstation will populate the working tree.
echo "  -> server git config done"
git config --get receive.denyCurrentBranch
REMOTE

echo "==> checking server tooling"
ssh "$SRV" bash -s <<'REMOTE'
echo "  -> python3: $(which python3) ($(python3 --version 2>&1))"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
else
  echo "  -> no nvidia-smi"
fi
REMOTE

echo "==> bootstrap done. Next: tools/deploy/deploy.sh from workstation."
