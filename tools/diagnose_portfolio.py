#!/usr/bin/env python3
"""Diagnose why rewards are 0.00"""

from train_jax_fast import FastTradingEnv
import numpy as np

print("Testing portfolio behavior...\n")

env = FastTradingEnv(n_assets=64, lookback=60)
obs, _ = env.reset()

print(f"Initial state:")
print(f"  Portfolio value: ${env.portfolio.state.total_value:.2f}")
print(f"  Cash: ${env.portfolio.state.cash:.2f}")
print(f"  Positions: {len(env.portfolio.state.positions)}")
print()

print("Executing 10 random actions:")
for i in range(10):
    action = np.random.randint(0, env.action_space.n)
    asset_idx = action // 3
    action_type = action % 3
    action_name = ["Sell", "Hold", "Buy"][action_type]

    prev_value = env.portfolio.state.total_value
    obs, reward, term, trunc, _ = env.step(action)
    new_value = env.portfolio.state.total_value

    print(f"  Step {i+1}: asset={asset_idx}, action={action_name}, "
          f"value: ${prev_value:.2f} -> ${new_value:.2f}, reward={reward:+.4f}")

print()
print(f"Final state:")
print(f"  Portfolio value: ${env.portfolio.state.total_value:.2f}")
print(f"  Cash: ${env.portfolio.state.cash:.2f}")
print(f"  Positions: {len(env.portfolio.state.positions)}")

if env.portfolio.state.total_value == 10000.0:
    print("\n❌ Portfolio value unchanged - trades not executing!")
else:
    print("\n✅ Portfolio value changed - trades are working!")
