#!/usr/bin/env python3
"""Quick test to verify episodes don't terminate early"""

from train_jax_fast import FastTradingEnv
import numpy as np

print("Testing episode length...")

env = FastTradingEnv(n_assets=64, lookback=60)

# Test 5 episodes
for ep in range(5):
    obs, info = env.reset()
    steps = 0

    for step in range(200):
        action = np.random.randint(0, env.action_space.n)
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1

        if terminated or truncated:
            break

    print(f"Episode {ep+1}: {steps} steps, terminated={terminated}, truncated={truncated}")

print("\n✅ Test complete!")
print("Expected: Most episodes should reach 200 steps (truncated=True)")
print("If episodes end early with terminated=True, there's still an issue.")
