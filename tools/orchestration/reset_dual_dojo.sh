#!/bin/bash
# Reset dual dojo training state for fresh start
# Usage: ./tools/orchestration/reset_dual_dojo.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRADER_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}=== DUAL DOJO RESET ===${NC}"
echo ""

# Check for running processes
RUNNING=$(pgrep -f "dual_dojo.py" || true)
if [ -n "$RUNNING" ]; then
    echo -e "${RED}WARNING: dual_dojo is running (PID: $RUNNING)${NC}"
    read -p "Kill it now? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        pkill -f "dual_dojo.py" || true
        # Kill training branches (match branch names)
        pkill -f "branch_A_3090" || true
        pkill -f "branch_B_90hx" || true
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
DUAL_LOGS="$LOGS_DIR/dual-dojo"

echo "Checkpoint dir: $CKPT_BASE"
echo "Logs dir: $LOGS_DIR"
echo ""

# Show current sizes
echo -e "${YELLOW}Current disk usage:${NC}"
du -sh "$CKPT_BASE/jax_ckpt_dual_A" 2>/dev/null || echo "  Branch A: (not found)"
du -sh "$CKPT_BASE/jax_ckpt_dual_B" 2>/dev/null || echo "  Branch B: (not found)"
du -sh "$CKPT_BASE/jax_ckpt_dual_merged" 2>/dev/null || echo "  Merged: (not found)"
echo ""

read -p "Delete all dual dojo checkpoints and logs? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborting."
    exit 0
fi

# Delete checkpoints
echo -e "${YELLOW}Deleting dual dojo checkpoints...${NC}"
rm -rf "$CKPT_BASE/jax_ckpt_dual_A" 2>/dev/null || true
rm -rf "$CKPT_BASE/jax_ckpt_dual_B" 2>/dev/null || true
rm -rf "$CKPT_BASE/jax_ckpt_dual_merged" 2>/dev/null || true

# Recreate directories
mkdir -p "$CKPT_BASE/jax_ckpt_dual_A"
mkdir -p "$CKPT_BASE/jax_ckpt_dual_B"
mkdir -p "$CKPT_BASE/jax_ckpt_dual_merged"
echo "  Checkpoints deleted."

# Reset state
echo -e "${YELLOW}Resetting dual dojo state...${NC}"
mkdir -p "$DUAL_LOGS"
cat > "$DUAL_LOGS/dual_dojo_state.json" << 'EOF'
{
  "total_epochs_a": 0,
  "total_epochs_b": 0,
  "epochs_since_merge": 0,
  "current_merge_interval": 30,
  "merge_count": 0,
  "epoch_at_last_merge": 0,
  "last_cosine": 0.0,
  "last_agreement_a": 0.0,
  "last_agreement_b": 0.0
}
EOF
echo "  dual_dojo_state.json reset."

# Reset merge history
echo "[]" > "$DUAL_LOGS/merge_history.json"
echo "  merge_history.json reset."

# Delete logs
echo -e "${YELLOW}Deleting dual dojo logs...${NC}"
rm -f "$LOGS_DIR"/branch_A_3090_*.csv 2>/dev/null || true
rm -f "$LOGS_DIR"/branch_B_90hx_*.csv 2>/dev/null || true
rm -f "$DUAL_LOGS"/*.log 2>/dev/null || true
echo "  Logs deleted."

# Summary
echo ""
echo -e "${GREEN}=== RESET COMPLETE ===${NC}"
echo ""
echo "To start dual dojo training:"
echo "  python tools/orchestration/dual_dojo.py --run --checkpoint <path_to_checkpoint.pkl>"
