# TASK_136a-cl: Pre-encode Observations Investigation

**Date**: 2026-02-02
**Status**: INVESTIGATION COMPLETE - READY FOR IMPLEMENTATION
**Priority**: HIGH

---

## Executive Summary

**Verdict: YES, pre-encoding is feasible with compressed storage.**

| Option | Memory (50k demos) | Feasible? |
|--------|-------------------|-----------|
| Full (27, 64, 64) | 22.1 GB | NO |
| Compressed (27, 64) | 345 MB | YES (64x reduction) |

---

## Finding 1: ALL Channels Are Tiled (100% Redundancy)

**Location**: `mcts_trader/env/jax_encoder.py:140-292`

Every channel creates output via tiling a (n_assets,) vector to (n_assets, n_assets):

```python
# Channel 0: Close price
output = output.at[0].set(jnp.tile(norm_prices[:, -1:], (1, n)))

# Channel 4: 1-min return
output = output.at[4].set(jnp.tile(returns_1m[:, -1:], (1, n)))

# ... all 27 channels follow this pattern
```

**Conclusion**: Output (27, 64, 64) has 64x redundancy. Store (27, 64), tile at sample time.

---

## Finding 2: Memory Math

```
Full:       27 × 64 × 64 × 4 bytes = 442 KB/demo × 50k = 22.1 GB
Compressed: 27 × 64 × 4 bytes      = 6.9 KB/demo × 50k = 345 MB
```

---

## Finding 3: JAX Available in Callback

**Location**: `train_jax_fast.py:615-617`

```python
optimal_path = compute_optimal_path(
    jnp.array(log_returns), k=4, method='greedy', transaction_cost=0.0
)
```

JAX is already used in the callback. Encoder can run there.

---

## Finding 4: Current Bottleneck

**Location**: `expert_buffer.py:506-553`

```python
for ep_id in unique_episodes:          # Python loop - SLOW
    close_prices = jnp.array(...)      # Host→Device
    encoded = jax.vmap(...)(window_starts)  # GPU
    observations[batch_indices] = np.array(encoded)  # Device→Host BLOCKS
```

Each BC batch triggers multiple GPU calls + sync points.

---

## Implementation Plan

### Step 1: Add compressed encoder to jax_encoder.py

```python
def encode_observation_compressed(prices, window_start, config):
    """Returns (27, n_assets) instead of (27, n, n)."""
    # Extract pre-tile values from existing computation
    # Return stacked column vectors
```

### Step 2: Add storage to UnifiedIndexBuffer

```python
def __init__(self, ...):
    self._compressed_obs = np.zeros(
        (demo_capacity, 27, n_assets), dtype=np.float32
    )  # 345 MB
```

### Step 3: Encode at add_demo time

```python
def add_demo(self, ..., compressed_obs):
    self._compressed_obs[self.demo_count] = compressed_obs
```

### Step 4: Tile at sample_batch time

```python
def sample_batch(self, ...):
    compressed = self._compressed_obs[indices]  # (batch, 27, 64)
    observations = np.tile(
        compressed[:, :, :, np.newaxis], (1, 1, 1, self.n_assets)
    )  # (batch, 27, 64, 64) - FAST
```

### Step 5: Batch-encode in callback

```python
# In on_episode_complete_callback:
valid_steps = range(window_size, n_steps)
window_starts = jnp.array([s - window_size for s in valid_steps])
all_obs = jax.vmap(compressed_encoder)(window_starts)  # Single GPU call

for i, step in enumerate(valid_steps):
    expert_buffer.add_demo(..., compressed_obs=all_obs[i])
```

---

## Expected Speedup

| Metric | Current | After |
|--------|---------|-------|
| sample_batch time | ~50ms | ~2ms |
| GPU utilization | 5-17% | 60-80% |
| Encoding | Per sample_batch | Once at add_demo |

**~25x speedup for BC batch preparation**

---

## Files to Modify

1. `mcts_trader/env/jax_encoder.py` - Add `encode_observation_compressed()`
2. `mcts_trader/jax_agent/expert_buffer.py` - Add storage, modify add_demo/sample_batch
3. `train_jax_fast.py` - Batch-encode in callback
