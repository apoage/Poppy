"""
TASK_768 train_easy.py path smoke: exercise the same call path as
train_easy.py:1786 — jit_vmap_encode_temporal(prices, window_starts, config)
on a randomized 4D OHLCV tensor.

Verifies the wrapper still works correctly when invoked through the
batched + vmap'd JIT entry point. This is the hot path during training.
"""
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

sys.path.insert(0, "/home/lukas/Trader")

import jax
import jax.numpy as jnp
import numpy as np

from mcts_trader.env.jax_encoder import (
    EncoderConfig,
    jit_vmap_encode_temporal,
    jit_batch_encode_temporal,
)


def main():
    cfg = EncoderConfig(n_assets=64, window_size=64, n_channels=32)
    rng = np.random.default_rng(7)
    # Shape (N=2 episodes, n_steps=200, n_assets=64, OHLCV=5)
    base = 100.0
    arr = np.zeros((2, 200, 64, 5), dtype=np.float32)
    for ep in range(2):
        for a in range(64):
            log_walk = np.cumsum(rng.normal(0, 0.01, size=200))
            close = base * np.exp(log_walk)
            arr[ep, :, a, 0] = close * (1 - 0.0005)  # open
            arr[ep, :, a, 1] = close * 1.001         # high
            arr[ep, :, a, 2] = close * 0.999         # low
            arr[ep, :, a, 3] = close                  # close
            arr[ep, :, a, 4] = rng.uniform(0.5, 2.0, size=200) * 1000  # volume

    prices = jnp.asarray(arr)
    window_starts = jnp.arange(0, 100, 20)  # (5,)

    print(f"Input prices shape: {prices.shape}")
    print(f"Window starts shape: {window_starts.shape}")

    out = jit_vmap_encode_temporal(prices, window_starts, cfg)
    out_np = jax.device_get(out)
    print(f"jit_vmap_encode_temporal output:")
    print(f"  shape: {out_np.shape}  (expected: (N, L, 32, 64, 64))")
    print(f"  dtype: {out_np.dtype}")
    print(f"  NaN: {bool(np.isnan(out_np).any())}")
    print(f"  Inf: {bool(np.isinf(out_np).any())}")
    print(f"  min: {out_np.min():.4f}, max: {out_np.max():.4f}")

    # Also test the single-episode batched form
    single = jit_batch_encode_temporal(prices[0], window_starts, cfg)
    single_np = jax.device_get(single)
    print()
    print(f"jit_batch_encode_temporal output:")
    print(f"  shape: {single_np.shape}  (expected: (L, 32, 64, 64))")
    print(f"  NaN: {bool(np.isnan(single_np).any())}")

    # Confirm jit_vmap result matches per-episode jit_batch result
    per_ep = jnp.stack([
        jit_batch_encode_temporal(prices[i], window_starts, cfg)
        for i in range(prices.shape[0])
    ])
    per_ep_np = jax.device_get(per_ep)
    print()
    print(f"vmap vs per-episode loop: array_equal={np.array_equal(out_np, per_ep_np)}")

    print()
    print("TRAIN-EASY PATH SMOKE: OK")


if __name__ == "__main__":
    main()
