# TASK_136b-cl: Batch Encode All Episodes in Single JAX Call

**Date**: 2026-02-02
**Status**: INVESTIGATION COMPLETE
**Priority**: MEDIUM (less impactful than 136a pre-encoding)

---

## Executive Summary

**Verdict: FEASIBLE but less optimal than TASK_136a**

- Single JAX call IS possible using existing `jit_batch_encode_simple(per_env=True)`
- Requires padding episodes to same length + duplicating prices per sample
- Temporary memory: ~134 MB for 1024-sample batch
- Expected speedup: **3-5x** (vs 25x from 136a)
- **Recommendation**: Implement 136a first, use 136b as fallback

---

## Current Bottleneck Analysis

**Location**: `expert_buffer.py:506-553`

```python
for ep_id in unique_episodes:           # ~30 iterations
    close_prices = jnp.array(...)       # H→D transfer (~3ms)
    encoded = jax.vmap(...)(window_starts)  # GPU kernel (~1ms)
    observations[batch_indices] = np.array(encoded)  # D→H BLOCKS (~2ms)
```

**Per iteration**: ~6ms (3ms H→D + 1ms kernel + 2ms D→H sync)
**30 episodes**: ~180ms + Python overhead = **~200ms total**

---

## Key Discovery: Existing API Supports This

**Location**: `jax_encoder.py:491-516`

```python
@partial(jax.jit, static_argnames=['config', 'per_env'])
def jit_batch_encode_simple(prices, window_starts, config, per_env=False):
    if per_env:
        # Each sample has its own price data
        return jax.vmap(
            lambda p, ws: encode_observation_simple(p, ws, config)
        )(prices, window_starts)
```

**API when `per_env=True`:**
- `prices`: (batch_size, T, n_assets) - unique prices per sample
- `window_starts`: (batch_size,) - one per sample
- Returns: (batch_size, 27, n_assets, n_assets)

---

## Solution Design

### Step 1: Find max episode length

```python
# Get all episode lengths
episode_ids = entries['episode_id']
unique_eps = np.unique(episode_ids)
max_T = max(self._episode_prices[ep].shape[0] for ep in unique_eps)
```

### Step 2: Build padded price array

```python
batch_size = len(entries)
n_assets = self.n_assets

# Pre-allocate (batch_size, max_T, n_assets)
padded_prices = np.zeros((batch_size, max_T, n_assets), dtype=np.float32)

# Fill each sample's prices (with padding)
for i, ep_id in enumerate(episode_ids):
    ep_prices = self._episode_prices[ep_id]  # (T_i, n_assets)
    T_i = ep_prices.shape[0]
    padded_prices[i, :T_i, :] = ep_prices
    # Padding with zeros is fine - window_start ensures we only use valid region
```

### Step 3: Compute window_starts

```python
steps = entries['step'].astype(np.int32)
window_starts = jnp.array(steps - self.window_size)
```

### Step 4: Single JAX call

```python
from mcts_trader.env.jax_encoder import jit_batch_encode_simple, EncoderConfig

config = EncoderConfig(n_assets=n_assets, window_size=self.window_size, n_channels=27)

# Single GPU call - no Python loop!
observations = jit_batch_encode_simple(
    jnp.array(padded_prices),  # (batch, max_T, n_assets)
    window_starts,              # (batch,)
    config,
    per_env=True
)

return np.array(observations)  # Single D→H transfer
```

---

## Memory Analysis

### Temporary Memory (during encoding)

| Component | Calculation | Size |
|-----------|-------------|------|
| Padded prices | 1024 × 512 × 64 × 4 | **134 MB** |
| Output observations | 1024 × 27 × 64 × 64 × 4 | **452 MB** |
| **Total temporary** | | **~600 MB** |

This is acceptable - it's transient GPU memory freed after the call.

### Comparison to Current

Current approach also allocates 452 MB output, plus ~5-15 MB per episode transfer.
Net memory increase: ~134 MB (padded prices array).

---

## Performance Estimate

### Current (Python loop, 30 episodes)

| Phase | Time |
|-------|------|
| 30× H→D (prices) | 30 × 3ms = 90ms |
| 30× GPU kernels | 30 × 1ms = 30ms |
| 30× D→H (encoded) | 30 × 2ms = 60ms |
| Python loop overhead | ~20ms |
| **Total** | **~200ms** |

### Proposed (single call)

| Phase | Time |
|-------|------|
| 1× H→D (padded prices 134MB) | ~10ms |
| 1× GPU kernel (large vmap) | ~5ms |
| 1× D→H (output 452MB) | ~35ms |
| **Total** | **~50ms** |

**Speedup: ~4x**

---

## Comparison: 136a vs 136b

| Aspect | 136a (Pre-encode) | 136b (Batch call) |
|--------|-------------------|-------------------|
| sample_batch time | ~2ms (just tile) | ~50ms (encode) |
| Memory | +345 MB permanent | +134 MB temporary |
| Complexity | Medium (new storage) | Low (refactor loop) |
| Speedup | **~25x** | **~4x** |
| When encoding | At add_demo time | At sample_batch time |

**Recommendation**: 136a is superior for training throughput. 136b is a simpler fallback.

---

## Implementation Code

```python
def _encode_observations_vectorized(self, entries: np.ndarray) -> np.ndarray:
    """Single-call batch encoding using jit_batch_encode_simple."""
    import jax.numpy as jnp
    from mcts_trader.env.jax_encoder import jit_batch_encode_simple, EncoderConfig

    batch_size = len(entries)
    n_assets = self.n_assets

    # Get episode info
    episode_ids = entries['episode_id'].astype(np.int32)
    steps = entries['step'].astype(np.int32)

    # Validate steps
    valid_mask = steps >= self.window_size
    if not np.all(valid_mask):
        # Handle invalid steps (same as current)
        pass

    # Find max episode length
    unique_eps = np.unique(episode_ids)
    max_T = max(self._episode_prices[int(ep)].shape[0] for ep in unique_eps)

    # Build padded price array (batch_size, max_T, n_assets)
    padded_prices = np.zeros((batch_size, max_T, n_assets), dtype=np.float32)
    for i in range(batch_size):
        ep_prices = self._episode_prices[int(episode_ids[i])]
        T_i = ep_prices.shape[0]
        padded_prices[i, :T_i, :] = ep_prices

    # Compute window_starts
    window_starts = jnp.array(steps - self.window_size)

    # Single JAX call
    config = EncoderConfig(
        n_assets=n_assets,
        window_size=self.window_size,
        n_channels=27,
    )

    observations = jit_batch_encode_simple(
        jnp.array(padded_prices),
        window_starts,
        config,
        per_env=True
    )

    return np.array(observations)
```

---

## Risks and Mitigations

### Risk 1: JIT recompilation on different max_T
- JAX may recompile if max_T changes between batches
- **Mitigation**: Pad to fixed MAX_EPISODE_LEN (e.g., 512)

### Risk 2: Memory spikes
- 600 MB temporary allocation
- **Mitigation**: Acceptable for modern GPUs (8+ GB)

### Risk 3: Padding inefficiency
- Short episodes waste memory
- **Mitigation**: Most episodes are similar length (~400 steps)

---

## Conclusion

**136b is a valid optimization but 136a is better.**

- 136b: 4x speedup, low implementation effort
- 136a: 25x speedup, medium implementation effort

**Implement 136a first. Use 136b if 136a proves problematic.**
