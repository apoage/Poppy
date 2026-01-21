#!/bin/bash
# Cleanup script for TASK_065 Refactoring
# ONLY RUN THIS AFTER FULL SYSTEM VALIDATION

echo "Cleaning up backup files..."
rm mcts_trader/env/vectorbt_env.py.bak

echo "Cleanup complete."
rm mcts_trader/jax_agent/trainer.py.bak
rm -rf tools/_old/
