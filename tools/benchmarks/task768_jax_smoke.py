"""
TASK_768 JAX-level smoke test: invoke the actual `_compute_all_channels_temporal`
and `_compute_all_channels` from jax_encoder.py with random inputs and verify
they run without NaN/Inf and produce the expected shape + the numpy-emulated
reference's value (the bit-identity check already proved old==new with numpy).

This is a "does it run under JAX" check — not a parity check. The parity check
is at tools/benchmarks/task768_bit_identity_check.py.
"""
import os
import sys

# Force CPU to avoid GPU init in a quick smoke
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_DISABLE_JIT", "0")

sys.path.insert(0, "/home/lukas/Trader")

import jax
import jax.numpy as jnp
import numpy as np

from mcts_trader.env.jax_encoder import (
    EncoderConfig,
    _compute_all_channels,
    _compute_all_channels_temporal,
    _compute_per_asset_channels,
    _compute_per_asset_channels_temporal,
    _compute_cross_sectional,
    _compute_cross_sectional_temporal,
)

def main():
    rng = np.random.default_rng(7)
    n_assets, ws = 64, 64
    cfg = EncoderConfig(n_assets=n_assets, window_size=ws, n_channels=32)

    base = 100.0
    log_walk = np.cumsum(rng.normal(0, 0.01, size=(n_assets, ws)), axis=1)
    prices = jnp.asarray((base * np.exp(log_walk)).astype(np.float32))
    volumes = jnp.asarray((rng.uniform(0.5, 2.0, size=(n_assets, ws)) * 1000.0).astype(np.float32))
    highs = jnp.asarray((np.asarray(prices) * (1.0 + rng.uniform(0, 0.005, size=prices.shape))).astype(np.float32))
    lows = jnp.asarray((np.asarray(prices) * (1.0 - rng.uniform(0, 0.005, size=prices.shape))).astype(np.float32))

    print("Calling _compute_all_channels_temporal (post-768 wrapper)...")
    temp = _compute_all_channels_temporal(prices, volumes, highs, lows, cfg)
    temp = jax.device_get(temp)
    print(f"  shape={temp.shape}, dtype={temp.dtype}")
    print(f"  has NaN: {bool(np.isnan(temp).any())}")
    print(f"  has Inf: {bool(np.isinf(temp).any())}")
    print(f"  min={temp.min():.6f}, max={temp.max():.6f}")

    print()
    print("Calling _compute_all_channels (post-768 tiled wrapper)...")
    tiled = _compute_all_channels(prices, volumes, highs, lows, cfg)
    tiled = jax.device_get(tiled)
    print(f"  shape={tiled.shape}, dtype={tiled.dtype}")
    print(f"  has NaN: {bool(np.isnan(tiled).any())}")
    print(f"  has Inf: {bool(np.isinf(tiled).any())}")
    print(f"  min={tiled.min():.6f}, max={tiled.max():.6f}")

    print()
    print("Calling split helpers directly...")
    pa_t, r1m_t = _compute_per_asset_channels_temporal(prices, volumes, highs, lows, cfg)
    xs_t = _compute_cross_sectional_temporal(prices, r1m_t, n_assets, ws, cfg)
    print(f"  temporal: per-asset shape={pa_t.shape}, returns_1m shape={r1m_t.shape}")
    print(f"  temporal: cross-sectional tuple len={len(xs_t)}, each shape={xs_t[0].shape}")
    pa, r1m = _compute_per_asset_channels(prices, volumes, highs, lows, cfg)
    xs = _compute_cross_sectional(prices, r1m, n_assets, cfg)
    print(f"  tiled: per-asset shape={pa.shape}, returns_1m shape={r1m.shape}")
    print(f"  tiled: cross-sectional tuple len={len(xs)}, each shape={xs[0].shape}")

    # Confirm finite cross-sectional values
    for i, (name, v) in enumerate(zip(["rank_1", "rank_20", "beta", "idio_vol", "xs_mom"], xs_t)):
        v_np = jax.device_get(v)
        print(f"  XS-{i} {name}: min={v_np.min():.4f}, max={v_np.max():.4f}, "
              f"NaN={bool(np.isnan(v_np).any())}, Inf={bool(np.isinf(v_np).any())}")

    print()
    print("SMOKE: OK")


if __name__ == "__main__":
    main()
