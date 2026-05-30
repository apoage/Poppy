"""TASK_786 smoke: chunked encoder forward is numerically identical to legacy.

Builds two BridgeTrainers with the SAME seed (→ identical params), one with
`oracle_distill_encode_chunk=0` (legacy single-shot) and one chunked, then
compares `evaluate_batch` losses on the same synthetic bridge batch. Also runs
a real `train_step` to confirm grads flow through the chunked+remat path.

Run: python tools/smoke_786_encode_chunk.py
"""
import numpy as np
import jax
import jax.numpy as jnp

from easy_config import EasyConfig
from bridge_trainer import create_bridge_trainer


def make_config(encode_chunk):
    c = EasyConfig()
    c.H_cycles, c.L_cycles = 2, 3      # small/fast
    c.epochs, c.steps_per_epoch, c.warmup_steps = 1, 1, 1
    c.oracle_distill_unroll_steps = 2
    c.oracle_distill_encode_chunk = encode_chunk
    return c


def make_batch(B, K, C, W, N, A, seed=0):
    rng = np.random.default_rng(seed)
    T = K + 1
    vd = rng.random((B, T, A)).astype(np.float32)
    vd /= vd.reshape(B, T, A // 5, 5).sum(-1, keepdims=True).repeat(5, -1).reshape(B, T, A)
    return {
        'observations': jnp.asarray(rng.standard_normal((B, T, C, W, N)), jnp.float32),
        'actions': jnp.asarray(rng.integers(0, A, (B, K)), jnp.int32),
        'rewards': jnp.asarray(rng.standard_normal((B, K)), jnp.float32),
        'visit_distributions': jnp.asarray(vd),
        'root_value': jnp.asarray(rng.standard_normal((B,)), jnp.float32),
        'dones': jnp.zeros((B, K), jnp.bool_),
    }


def main():
    B, K = 8, 2
    legacy = create_bridge_trainer(make_config(0), seed=123)
    chunked = create_bridge_trainer(make_config(4), seed=123)   # BT=24, chunk=4 → 6 chunks

    C = legacy.config.n_channels
    N = legacy.config.n_assets
    W = legacy.config.window_size
    A = legacy.config.action_space_size
    batch = make_batch(B, K, C, W, N, A)

    m_legacy = legacy.evaluate_batch(batch)
    m_chunk = chunked.evaluate_batch(batch)

    print("metric            legacy        chunked       |Δ|")
    worst = 0.0
    for k in sorted(m_legacy):
        a, b = m_legacy[k], m_chunk[k]
        d = abs(a - b)
        worst = max(worst, d / (abs(a) + 1e-9))
        print(f"  {k:18s} {a:12.6f} {b:12.6f} {d:.2e}")
    print(f"\nworst relative Δ = {worst:.2e}")
    assert worst < 1e-4, "chunked encode diverged from legacy!"

    # grads must flow through chunked+remat path
    before = float(jax.tree_util.tree_leaves(chunked.params)[0].sum())
    tm = chunked.train_step(batch)
    after = float(jax.tree_util.tree_leaves(chunked.params)[0].sum())
    print(f"\ntrain_step loss={tm['total_loss']:.4f} grad_norm={tm['grad_norm']:.4f}")
    assert np.isfinite(tm['total_loss']), "non-finite loss"
    assert before != after, "params did not update — grad did not flow"
    print("\nPASS: chunked encode is equivalent + trainable.")


if __name__ == '__main__':
    main()
