# TASK_136a: Pre-encode Observations at Demo Time

**Date**: 2026-02-02
**Status**: IMPLEMENTED
**Priority**: HIGH (after 136d)
**Consensus**: ALL AGENTS AGREE - Feasible with compression

---

## Executive Summary

| Metric | Full Storage | Compressed Storage |
|--------|--------------|-------------------|
| Shape | (27, 64, 64) | (27, 64) |
| Per demo | 442 KB | 6.9 KB |
| 50k demos | 22.1 GB | **345 MB** |
| Speedup | - | **20-25×** |

**Key Discovery**: ALL 27 encoder channels use `jnp.tile` - 100% redundancy. Store (27, 64), tile at sample time.

---

## Why Compression Works

**Location**: `mcts_trader/env/jax_encoder.py:140-292`

```python
# EVERY channel does this:
output = output.at[0].set(jnp.tile(norm_prices[:, -1:], (1, n)))   # Close
output = output.at[4].set(jnp.tile(returns_1m[:, -1:], (1, n)))    # 1m return
output = output.at[5].set(jnp.tile(returns_5m[:, None], (1, n)))   # 5m return
# ... all 27 channels tile (n,) → (n, n)
```

**Result**: (27, 64, 64) has 64× redundancy. Store first column only.

---

## Implementation Plan

### Step 1: Add compressed encoder to `jax_encoder.py`

```python
def encode_observation_compressed(prices, window_start, config):
    """Return (27, n_assets) instead of (27, n, n)."""
    # Compute raw values before tiling
    # Return stacked column vectors
    ...

def decompress_observation(compressed):
    """Tile (27, n) back to (27, n, n)."""
    n = compressed.shape[1]
    return jnp.tile(compressed[..., None], (1, 1, n))
```

### Step 2: Add storage to `expert_buffer.py`

```python
class UnifiedIndexBuffer:
    def __init__(self, ...):
        # TASK_136a: Pre-encoded observation storage
        self._compressed_obs = np.zeros(
            (demo_capacity, 27, n_assets), dtype=np.float32
        )  # 345 MB for 50k demos
```

### Step 3: Batch-encode in callback

```python
# In on_episode_complete_callback:
valid_steps = np.arange(window_size, n_steps)
window_starts = jnp.array(valid_steps - window_size)

# Single GPU call for entire episode
all_obs = jax.vmap(compressed_encoder)(window_starts)  # (n_steps, 27, 64)

for i, step in enumerate(valid_steps):
    expert_buffer.add_demo(..., compressed_obs=all_obs[i])
```

### Step 4: Tile at sample_batch time

```python
def sample_batch(self, ...):
    compressed = self._compressed_obs[indices]  # (batch, 27, 64)
    observations = np.tile(
        compressed[:, :, :, np.newaxis], (1, 1, 1, self.n_assets)
    )  # (batch, 27, 64, 64) - FAST
```

---

## Expected Impact

| Metric | Before | After |
|--------|--------|-------|
| sample_batch time | ~200ms | ~5ms |
| GPU utilization | 5-17% | 60-80% |
| Memory | +0 MB | +345 MB |

---

## Files to Modify

1. `mcts_trader/env/jax_encoder.py` - Add `encode_observation_compressed()`
2. `mcts_trader/jax_agent/expert_buffer.py` - Add storage, modify add_demo/sample_batch
3. `train_jax_fast.py` - Batch-encode in callback

---

## Risks

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Compression artifacts | Very Low | All channels are truly tiled |
| Memory pressure | Low | 345 MB is manageable |
| Implementation bugs | Medium | Keep fallback path |

---

## Agent Consensus

| Agent | Verdict | Notes |
|-------|---------|-------|
| Claude | FEASIBLE | 25× speedup, 345 MB RAM |
| Gemini | FEASIBLE | "Strictly superior" to 136b |
| K | FEASIBLE | Detailed implementation plan |

**IMPLEMENT AFTER 136d (frequency reduction).**
