# TASK_136b: Batch Encode All Episodes in Single JAX Call

**Date**: 2026-02-02
**Status**: READY FOR IMPLEMENTATION (fallback to 136a)
**Priority**: MEDIUM
**Consensus**: FEASIBLE - Simpler than 136a but less speedup

---

## Executive Summary

| Metric | Current | With Fix |
|--------|---------|----------|
| JAX calls per batch | 15-30 | **1** |
| Encoding time | ~200ms | ~50ms |
| Speedup | - | **4-10×** |
| Memory | +0 MB | +134 MB temp |

**Use case**: Fallback if 136a (pre-encoding) proves problematic.

---

## Key Discovery

**Location**: `jax_encoder.py:491-516`

```python
@partial(jax.jit, static_argnames=['config', 'per_env'])
def jit_batch_encode_simple(prices, window_starts, config, per_env=False):
    if per_env:
        # Each sample has its own price data - PERFECT FOR BATCHING
        return jax.vmap(
            lambda p, ws: encode_observation_simple(p, ws, config)
        )(prices, window_starts)
```

This API already supports batched encoding!

---

## Implementation

### Option A: Pad and Batch (Recommended)

```python
def _encode_observations_batched(self, entries: np.ndarray) -> np.ndarray:
    """Single JAX call encoding using jit_batch_encode_simple."""
    from mcts_trader.env.jax_encoder import jit_batch_encode_simple, EncoderConfig

    batch_size = len(entries)
    episode_ids = entries['episode_id'].astype(np.int32)
    steps = entries['step'].astype(np.int32)

    # Find max episode length
    unique_eps = np.unique(episode_ids)
    max_T = max(self._episode_prices[int(ep)].shape[0] for ep in unique_eps)

    # Build padded price array (batch_size, max_T, n_assets)
    padded_prices = np.zeros((batch_size, max_T, self.n_assets), dtype=np.float32)
    for i in range(batch_size):
        ep_prices = self._episode_prices[int(episode_ids[i])]
        padded_prices[i, :len(ep_prices), :] = ep_prices

    # Compute window_starts
    window_starts = jnp.array(steps - self.window_size)

    # Single JAX call - no Python loop!
    config = EncoderConfig(n_assets=self.n_assets, window_size=self.window_size)
    observations = jit_batch_encode_simple(
        jnp.array(padded_prices),
        window_starts,
        config,
        per_env=True
    )

    return np.array(observations)
```

### Option B: Index-based (from Gemini)

```python
@partial(jax.jit, static_argnames=['config'])
def jit_encode_from_bank(price_bank, ep_indices, window_starts, config):
    """Encode using pre-built price bank."""
    def _encode(ep_idx, start):
        return encode_observation_simple(price_bank[ep_idx], start, config)
    return jax.vmap(_encode)(ep_indices, window_starts)
```

---

## Memory Analysis

```
Temporary allocation:
- padded_prices: 1024 × 512 × 64 × 4 = 134 MB
- output: 1024 × 27 × 64 × 64 × 4 = 452 MB
- Total: ~600 MB temporary (freed after call)
```

---

## Comparison with 136a

| Aspect | 136a (Pre-encode) | 136b (Batch call) |
|--------|-------------------|-------------------|
| sample_batch time | ~5ms | ~50ms |
| Memory | +345 MB permanent | +134 MB temporary |
| Complexity | Medium | Low |
| Speedup | **25×** | **4-10×** |

**Verdict**: 136a is better, but 136b is simpler if you don't want to change the storage format.

---

## Files to Modify

1. `mcts_trader/jax_agent/expert_buffer.py` - Replace `_encode_observations_vectorized`

---

## Agent Consensus

| Agent | Verdict | Notes |
|-------|---------|-------|
| Claude | FEASIBLE | ~50ms vs 200ms current |
| Gemini | FEASIBLE | 10× speedup, standard pattern |
| K | FEASIBLE | "5-15× speedup" |

**IMPLEMENT IF 136a proves problematic.**
