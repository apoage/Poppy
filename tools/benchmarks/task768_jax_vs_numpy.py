"""
TASK_768 JAX-vs-numpy parity check: verify the actual JAX
`_compute_all_channels{,_temporal}` (post-768) produce a tensor
that matches the numpy-emulated pre-768 reference bit-identically.

Combined with `task768_bit_identity_check.py` (numpy_old vs numpy_new),
this closes the loop:
    numpy_old == numpy_new (proves the split is exact)
    AND jax_new == numpy_old (proves the JAX implementation matches)
    ⇒ jax_new bit-identical to pre-768 monolithic JAX.

Note: JAX float32 ops have the same numeric semantics as numpy float32
for these operations (add/sub/mul/div/std/mean/argsort/clip/where), so
np.array_equal is the appropriate gate.
"""
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

sys.path.insert(0, "/home/lukas/Trader")
sys.path.insert(0, "/home/lukas/Trader/tools/benchmarks")

import jax
import jax.numpy as jnp
import numpy as np

from mcts_trader.env.jax_encoder import (
    EncoderConfig as JaxEncoderConfig,
    _compute_all_channels,
    _compute_all_channels_temporal,
)
from task768_bit_identity_check import (
    EncoderConfig as NpEncoderConfig,
    old_monolithic,
    old_monolithic_tiled,
)


def run():
    overall_ok = True
    for seed_label, seed_v, sigma, vol_min, vol_max in [
        ("seed=7", 7, 0.01, 0.5, 2.0),
        ("seed=42", 42, 0.02, 0.1, 5.0),
    ]:
        rng = np.random.default_rng(seed_v)
        n_assets, ws = 64, 64
        base = 100.0
        log_walk = np.cumsum(rng.normal(0, sigma, size=(n_assets, ws)), axis=1)
        prices_np = (base * np.exp(log_walk)).astype(np.float32)
        volumes_np = (rng.uniform(vol_min, vol_max, size=(n_assets, ws)) * 1000.0).astype(np.float32)
        highs_np = (prices_np * (1.0 + rng.uniform(0, 0.005, size=prices_np.shape))).astype(np.float32)
        lows_np = (prices_np * (1.0 - rng.uniform(0, 0.005, size=prices_np.shape))).astype(np.float32)

        prices = jnp.asarray(prices_np)
        volumes = jnp.asarray(volumes_np)
        highs = jnp.asarray(highs_np)
        lows = jnp.asarray(lows_np)

        np_cfg = NpEncoderConfig(n_assets=n_assets, window_size=ws, n_channels=32)
        jx_cfg = JaxEncoderConfig(n_assets=n_assets, window_size=ws, n_channels=32)

        # Reference: pre-768 numpy monolithic
        ref_t = old_monolithic(prices_np, volumes_np, highs_np, lows_np, np_cfg)
        ref_tl = old_monolithic_tiled(prices_np, volumes_np, highs_np, lows_np, np_cfg)

        # Subject: post-768 JAX split-wrapper
        jx_t = jax.device_get(_compute_all_channels_temporal(prices, volumes, highs, lows, jx_cfg))
        jx_tl = jax.device_get(_compute_all_channels(prices, volumes, highs, lows, jx_cfg))

        # Compare
        eq_t = np.array_equal(ref_t, jx_t)
        diff_t = float(np.max(np.abs(ref_t - jx_t)))
        eq_tl = np.array_equal(ref_tl, jx_tl)
        diff_tl = float(np.max(np.abs(ref_tl - jx_tl)))

        print(f"[{seed_label}] JAX TEMPORAL vs numpy reference: array_equal={eq_t}, max abs diff={diff_t}")
        print(f"[{seed_label}] JAX TILED    vs numpy reference: array_equal={eq_tl}, max abs diff={diff_tl}")
        if not (eq_t and eq_tl):
            overall_ok = False
            # Per-channel breakdown
            for c in range(jx_cfg.n_channels):
                d_t = float(np.max(np.abs(ref_t[c] - jx_t[c])))
                d_tl = float(np.max(np.abs(ref_tl[c] - jx_tl[c])))
                if d_t > 0 or d_tl > 0:
                    print(f"    ch{c}: TEMPORAL diff={d_t}, TILED diff={d_tl}")

    print()
    print("JAX <-> NUMPY GATE:", "PASS" if overall_ok else "FAIL")
    return overall_ok


if __name__ == "__main__":
    ok = run()
    raise SystemExit(0 if ok else 1)
