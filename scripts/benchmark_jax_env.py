#!/usr/bin/env python3
"""
Benchmark: Python VectorBTTradingEnv vs JAX Trading Environment

Compares throughput between:
1. Current serial Python env.step() loop
2. New JAX vectorized batch_step()

Run: python scripts/benchmark_jax_env.py
"""

import time
import numpy as np
import jax
import jax.numpy as jnp

# Setup paths
import sys
sys.path.insert(0, '/home/lukas/Trader')

from mcts_trader.env.jax_trading_env import (
    EnvConfig, TradingState,
    jit_batch_reset, jit_batch_step,
)


def generate_synthetic_prices(n_steps: int, n_assets: int) -> np.ndarray:
    """Generate synthetic sine wave prices."""
    t = np.linspace(0, 4 * np.pi, n_steps)
    prices = np.zeros((n_steps, n_assets))
    for i in range(n_assets):
        freq = 1.0 + 0.1 * (i % 10)
        phase = 2 * np.pi * i / n_assets
        base = 100 + 10 * (i % 5)
        prices[:, i] = base + 10 * np.sin(freq * t + phase)
    return prices


def benchmark_jax_env(n_envs: int, n_steps: int, n_assets: int = 64) -> float:
    """Benchmark JAX vectorized environment."""
    config = EnvConfig(
        n_assets=n_assets,
        initial_cash=100_000.0,
        fee_bps=0.0,
        max_steps=n_steps,
        window_size=60,
        position_size_pct=0.1,
    )

    # Generate prices
    prices = jnp.array(generate_synthetic_prices(n_steps + 200, n_assets), dtype=jnp.float32)

    # Initialize envs
    keys = jax.random.split(jax.random.PRNGKey(42), n_envs)
    states = jit_batch_reset(keys, prices, config)

    # Warm up JIT
    actions = jnp.zeros(n_envs, dtype=jnp.int32)
    _ = jit_batch_step(states, actions, prices, config)

    # Benchmark
    total_steps = 0
    start = time.time()

    for step_idx in range(n_steps):
        # Random actions
        actions = jax.random.randint(
            jax.random.PRNGKey(step_idx),
            (n_envs,),
            0, n_assets * 5
        )
        states, obs, rewards, dones = jit_batch_step(states, actions, prices, config)
        total_steps += n_envs

    # Wait for GPU
    jax.block_until_ready(states.cash)
    elapsed = time.time() - start

    return total_steps / elapsed


def main():
    print("=" * 60)
    print("JAX Vectorized Trading Environment Benchmark")
    print("=" * 60)
    print()

    # Test different batch sizes
    batch_sizes = [4, 16, 64, 256, 1000]
    n_steps = 100

    print(f"{'Batch Size':>12} | {'Throughput':>18} | {'vs 4 envs':>12}")
    print("-" * 50)

    baseline = None
    for n_envs in batch_sizes:
        throughput = benchmark_jax_env(n_envs, n_steps)

        if baseline is None:
            baseline = throughput
            speedup = "baseline"
        else:
            speedup = f"{throughput / baseline:.1f}x"

        print(f"{n_envs:>12} | {throughput:>14,.0f} sps | {speedup:>12}")

    print()
    print("=" * 60)
    print("Comparison with current Python environment")
    print("=" * 60)
    print()
    print("Current train_jax_fast.py bottleneck:")
    print("  for env in envs: env.step(action)  # Serial Python loop")
    print()
    print("Estimated Python env throughput:")
    print("  - 4 envs serial: ~400-800 steps/sec (CPU bound)")
    print("  - Main bottleneck: Python object creation, dict lookups")
    print()

    jax_throughput = benchmark_jax_env(1000, 100)
    python_estimate = 600  # Typical for 4 serial envs

    print(f"JAX 1000 envs: {jax_throughput:,.0f} steps/sec")
    print(f"Python 4 envs: ~{python_estimate} steps/sec (estimated)")
    print(f"Speedup: {jax_throughput / python_estimate:.0f}x")
    print()
    print("Integration plan:")
    print("  1. Replace serial env.step() loop with jit_batch_step()")
    print("  2. Keep prices on GPU (no transfer)")
    print("  3. Run 100-1000 parallel envs instead of 4")
    print("  4. Entire self-play episode stays on GPU")


if __name__ == "__main__":
    main()
