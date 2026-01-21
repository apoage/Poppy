#!/bin/bash
# Cleanup script for TASK_065 Refactoring
# ONLY RUN THIS AFTER FULL SYSTEM VALIDATION

echo "Cleaning up backup files..."
rm mcts_trader/env/vectorbt_env.py.bak

echo "Cleanup complete."
rm mcts_trader/jax_agent/trainer.py.bak
rm -rf tools/_old/
rm mcts_trader/jax_agent/exploration.py.bak
rm mcts_trader/jax_agent/sac_entropy.py.bak
rm mcts_trader/jax_agent/exploration.py
rm mcts_trader/jax_agent/exploration.py.bak
rm mcts_trader/jax_agent/sac_entropy.py.bak
rm mcts_trader/jax_agent/exploration.py.old
rm train_jax_fast.py.bak
