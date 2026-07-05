#!/bin/bash
# Reset e-dojo training state for fresh start
# Usage: ./tools/orchestration/reset_edojo.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRADER_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=== E-DOJO RESET ===${NC}"
echo ""

# Check for running processes
RUNNING=$(pgrep -f "e-dojo_v1.py" || true)
if [ -n "$RUNNING" ]; then
    echo -e "${RED}WARNING: e-dojo is running (PID: $RUNNING)${NC}"
    echo "Stop it first with: pkill -f e-dojo_v1.py"
    read -p "Kill it now? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        pkill -f "e-dojo_v1.py" || true
        pkill -f "train_easy.py" || true
        sleep 2
        echo "Killed."
    else
        echo "Aborting."
        exit 1
    fi
fi

# Checkpoint locations
CKPT_BASE="/mnt/scratch/trader_checkpoints"
LOGS_DIR="$TRADER_DIR/logs"
EDOJO_LOGS="$LOGS_DIR/e-dojo"

echo "Checkpoint dir: $CKPT_BASE"
echo "Logs dir: $LOGS_DIR"
echo ""

# Show current sizes
echo -e "${YELLOW}Current disk usage (scratch disk):${NC}"
du -sh "$CKPT_BASE/jax_ckpt_easy_A" 2>/dev/null || echo "  Branch A: (not found)"
du -sh "$CKPT_BASE/jax_ckpt_easy_B" 2>/dev/null || echo "  Branch B: (not found)"
du -sh "$CKPT_BASE/jax_ckpt_easy_merged" 2>/dev/null || echo "  Merged: (not found)"
echo -e "${YELLOW}Current disk usage (local):${NC}"
du -sh "$TRADER_DIR/jax_ckpt_easy_A" 2>/dev/null || echo "  Branch A: (not found)"
du -sh "$TRADER_DIR/jax_ckpt_easy_B" 2>/dev/null || echo "  Branch B: (not found)"
du -sh "$TRADER_DIR/jax_ckpt_easy_merged" 2>/dev/null || echo "  Merged: (not found)"
du -sh "$LOGS_DIR" 2>/dev/null || echo "  Logs: (not found)"
echo ""

read -p "Delete all checkpoints and logs? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborting."
    exit 0
fi

# 1. Delete checkpoints (scratch disk)
echo -e "${YELLOW}Deleting scratch disk checkpoints...${NC}"
rm -rf "$CKPT_BASE/jax_ckpt_easy_A/branch_A_bella"/*.pkl 2>/dev/null || true
rm -rf "$CKPT_BASE/jax_ckpt_easy_B/branch_B_synth"/*.pkl 2>/dev/null || true
rm -rf "$CKPT_BASE/jax_ckpt_easy_B/branch_B_bella_inv"/*.pkl 2>/dev/null || true
rm -rf "$CKPT_BASE/jax_ckpt_easy_merged"/*.pkl 2>/dev/null || true

# Also clean local project checkpoints (if any)
echo -e "${YELLOW}Deleting local checkpoints...${NC}"
rm -rf "$TRADER_DIR/jax_ckpt_easy_A" 2>/dev/null || true
rm -rf "$TRADER_DIR/jax_ckpt_easy_B" 2>/dev/null || true
rm -rf "$TRADER_DIR/jax_ckpt_easy_merged" 2>/dev/null || true

# Recreate directories
mkdir -p "$CKPT_BASE/jax_ckpt_easy_A/branch_A_bella"
mkdir -p "$CKPT_BASE/jax_ckpt_easy_B/branch_B_synth"
mkdir -p "$CKPT_BASE/jax_ckpt_easy_B/branch_B_bella_inv"
mkdir -p "$CKPT_BASE/jax_ckpt_easy_merged"
echo "  Checkpoints deleted."

# 2. Reset dojo state
echo -e "${YELLOW}Resetting dojo state...${NC}"
mkdir -p "$EDOJO_LOGS"
cat > "$EDOJO_LOGS/dojo_state.json" << 'EOF'
{
  "total_epochs": 0,
  "current_level": 0,
  "steps_since_merge": 0,
  "current_merge_interval": 8000,
  "merge_count": 0,
  "epoch_at_last_merge": 0,
  "last_cosine": 0.0,
  "last_barrier": 0.0,
  "last_agreement_a": 0.0,
  "last_agreement_b": 0.0
}
EOF
echo "  dojo_state.json reset."

# 3. Reset merge history
echo "[]" > "$EDOJO_LOGS/merge_history.json"
echo "  merge_history.json reset."

# 4. Delete logs
echo -e "${YELLOW}Deleting logs...${NC}"
rm -f "$LOGS_DIR"/*.csv 2>/dev/null || true
rm -f "$EDOJO_LOGS"/*.log 2>/dev/null || true
echo "  Logs deleted."

# Summary
echo ""
echo -e "${GREEN}=== RESET COMPLETE ===${NC}"
echo ""
du -sh "$CKPT_BASE" 2>/dev/null
echo ""
echo "To start fresh training:"
echo "  python tools/orchestration/e-dojo_v1.py --run --preset large --epochs 2500 --dual-bella"
