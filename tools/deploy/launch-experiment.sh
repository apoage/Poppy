#!/usr/bin/env bash
# Launch one experiment on a single GPU, in a detached tmux session.
# Run on SERVER (not workstation). Workstation operator: ssh in to run, or have
# the server agent invoke this.
#
# Usage:
#   tools/deploy/launch-experiment.sh <exp_id> <gpu_index> [-- <python_cmd...>]
#
# Examples:
#   tools/deploy/launch-experiment.sh 000_smoke 0
#   tools/deploy/launch-experiment.sh 739_dense_sign 1 -- python train_easy.py --spec ...
#
# Behavior:
#   - Pins the run to GPU <gpu_index> via CUDA_VISIBLE_DEVICES
#   - Creates tmux session named exp_<exp_id>
#   - Output lands in runs/<exp_id>/train.log
#   - If no python cmd given, reads experiments/<exp_id>/SPEC.md and refuses to
#     guess — operator must specify the command (the agent does this after
#     reading the SPEC).

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <exp_id> <gpu_index> [-- <python_cmd...>]" >&2
  exit 2
fi

EXP_ID="$1"
GPU="$2"
shift 2
[[ "${1:-}" == "--" ]] && shift

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SPEC="$REPO_ROOT/experiments/$EXP_ID/SPEC.md"
RUN_DIR="$REPO_ROOT/runs/$EXP_ID"
SESSION="exp_$EXP_ID"

if [[ ! -f "$SPEC" ]]; then
  echo "error: no SPEC at $SPEC" >&2
  exit 1
fi

if [[ $# -eq 0 ]]; then
  echo "error: no command provided. Read $SPEC and pass the command after --" >&2
  exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "error: tmux session $SESSION already exists. Attach with: tmux attach -t $SESSION" >&2
  exit 1
fi

mkdir -p "$RUN_DIR"
echo "==> launching $EXP_ID on GPU $GPU in tmux session $SESSION"
echo "==> log: $RUN_DIR/train.log"

# Build the full command. tee captures both stdout and stderr to the log.
CMD="cd $REPO_ROOT && source trader_env/bin/activate && \
  CUDA_VISIBLE_DEVICES=$GPU $* 2>&1 | tee -a $RUN_DIR/train.log"

tmux new-session -d -s "$SESSION" "bash -c \"$CMD\""

echo "==> launched. Monitor with:"
echo "    tmux attach -t $SESSION       # interactive"
echo "    tail -f $RUN_DIR/train.log    # follow log"
