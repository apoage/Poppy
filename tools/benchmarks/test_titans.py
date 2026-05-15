"""Standalone smoke test for TitansMemoryBlock and GatedTemporalMemory.

Run from repo root:
    CUDA_VISIBLE_DEVICES=0 python tools/benchmarks/test_titans.py
"""
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.3')

import sys

import jax
import jax.numpy as jnp
import flax.linen as nn

# Ensure repo root on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from mcts_trader.jax_agent.titans_memory import TitansMemoryBlock, GatedTemporalMemory


def test_titans_block_forward():
    N, T, D = 4, 8, 16
    block = TitansMemoryBlock(embed_dim=D, memory_depth=2, persistent_tokens=2)
    x = jnp.ones((N, T, D), dtype=jnp.float32) * 0.1
    rng = jax.random.PRNGKey(0)
    params = block.init(rng, x)
    print("TitansMemoryBlock initialized OK")
    flat, _ = jax.tree_util.tree_flatten_with_path(params)
    print("Param tree (top entries):")
    for k, v in flat[:10]:
        print(f"  {k}: shape={v.shape}")

    y, diag = block.apply(params, x)
    print(f"Forward: y.shape={y.shape} (expected ({N}, {T}, {D}))")
    for dk in ('theta_mean', 'eta_mean', 'alpha_mean', 'mem_norm', 'surprise_norm'):
        print(f"  diag[{dk}] = {float(diag[dk]):.6f}")
    assert y.shape == (N, T, D), f"Bad shape: {y.shape}"

    # Backward test
    def outer_loss(params, x):
        y, _ = block.apply(params, x)
        return jnp.mean(y ** 2)

    loss_val = outer_loss(params, x)
    print(f"Outer loss: {float(loss_val):.6f}")

    grads = jax.grad(outer_loss)(params, x)
    flat_g, _ = jax.tree_util.tree_flatten_with_path(grads)
    n_finite = sum(int(jnp.all(jnp.isfinite(v))) for _, v in flat_g)
    n_total = len(flat_g)
    print(f"Gradients: {n_finite}/{n_total} finite")
    assert n_finite == n_total, "Some gradients non-finite!"


def test_mag_combiner():
    N, T, D = 4, 8, 16
    mag = GatedTemporalMemory(embed_dim=D, memory_depth=2, persistent_tokens=2,
                              gate_init=-10.0)
    x_seq = jnp.ones((N, T, D), dtype=jnp.float32) * 0.1
    mingru_out = jnp.full((N, D), 0.5, dtype=jnp.float32)
    rng = jax.random.PRNGKey(1)
    params = mag.init(rng, x_seq, mingru_out)
    print("\nGatedTemporalMemory initialized OK")

    out, diag = mag.apply(params, x_seq, mingru_out)
    print(f"MAG out.shape={out.shape}, gate_value={float(diag['gate_value']):.6f}")
    err = jnp.max(jnp.abs(out - mingru_out))
    print(f"max(|out - mingru_out|) = {float(err):.6e} (expect ~0 at init)")
    assert err < 1e-3, f"MAG not byte-identical at init: err={err}"

    def outer_loss(params, x_seq, mingru_out):
        out, _ = mag.apply(params, x_seq, mingru_out)
        return jnp.mean(out ** 2)

    grads = jax.grad(outer_loss)(params, x_seq, mingru_out)
    flat_g, _ = jax.tree_util.tree_flatten_with_path(grads)
    n_finite = sum(int(jnp.all(jnp.isfinite(v))) for _, v in flat_g)
    n_total = len(flat_g)
    print(f"MAG gradients: {n_finite}/{n_total} finite")
    assert n_finite == n_total


def test_jit_and_remat():
    """Same shapes that the encoder will see: N = B*A, T=64, D=64."""
    N, T, D = 32, 64, 64  # smaller B*A for fast smoke
    mag = GatedTemporalMemory(embed_dim=D, memory_depth=2, persistent_tokens=4)
    x_seq = jnp.ones((N, T, D), dtype=jnp.float32) * 0.05
    mingru_out = jnp.zeros((N, D), dtype=jnp.float32)
    rng = jax.random.PRNGKey(2)
    params = mag.init(rng, x_seq, mingru_out)

    # JIT compile a forward+backward
    @jax.jit
    def step(params, x_seq, mingru_out):
        def loss_fn(params):
            out, _ = mag.apply(params, x_seq, mingru_out)
            return jnp.mean(out ** 2)
        loss, grads = jax.value_and_grad(loss_fn)(params)
        return loss, grads

    loss, grads = step(params, x_seq, mingru_out)
    print(f"\nJIT compile + train step OK: loss={float(loss):.6f}")
    flat_g, _ = jax.tree_util.tree_flatten_with_path(grads)
    n_finite = sum(int(jnp.all(jnp.isfinite(v))) for _, v in flat_g)
    print(f"  Gradients: {n_finite}/{len(flat_g)} finite")


if __name__ == "__main__":
    test_titans_block_forward()
    test_mag_combiner()
    test_jit_and_remat()
    print("\nAll Titans memory tests passed.")
