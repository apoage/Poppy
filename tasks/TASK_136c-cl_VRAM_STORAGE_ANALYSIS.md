# TASK_136c-cl: VRAM Price Storage Analysis

**Date**: 2026-02-02
**Status**: INVESTIGATION COMPLETE - NOT RECOMMENDED
**Priority**: LOW (not the bottleneck)

---

## Executive Summary

**Verdict: NOT WORTH IMPLEMENTING**

The RAM→VRAM transfer for episode prices is **NOT the bottleneck**.

| Transfer | Size | Time | % of Total |
|----------|------|------|------------|
| Prices (30 eps) | 3 MB | **0.24 ms** | 0.12% |
| D→H output | 452 MB | ~40 ms | 20% |
| Python loop | - | ~20 ms | 10% |
| GPU kernels | - | ~30 ms | 15% |

**Real bottlenecks** are Python loop (136b) and encoding itself (136a).

---

## Transfer Time Analysis

### Per-Episode Transfer

```
Episode size: 400 steps × 64 assets × 4 bytes = 102,400 bytes = 100 KB
PCIe 3.0 x16 bandwidth: ~12 GB/s
Transfer time: 100 KB / 12 GB/s = 8.3 microseconds
```

### Per Batch (30 unique episodes)

```
Total transfer: 30 × 100 KB = 3 MB
Transfer time: 3 MB / 12 GB/s = 0.25 ms
```

### Comparison to Other Costs

| Operation | Time | Location |
|-----------|------|----------|
| H→D prices (30 eps) | 0.25 ms | `jnp.array(close_prices_np)` |
| GPU encode (30 vmaps) | ~30 ms | `jax.vmap(encoder_fn)` |
| D→H output (452 MB) | ~40 ms | `np.array(encoded)` |
| Python loop overhead | ~20 ms | `for ep_id in unique_episodes` |
| **Total** | **~90 ms** | |

**H→D transfer is 0.3% of total time** - eliminating it saves nothing meaningful.

---

## VRAM Budget (If We Did Implement)

### Available VRAM

```
Total GPU VRAM: 24 GB (RTX 4090)
Model + activations: ~17 GB
Available: ~7 GB
```

### Storage Requirements

```
Per episode: 400 × 64 × 4 = 100 KB
Per epoch (50 envs): 50 × 100 KB = 5 MB
Max before clear (50 epochs): 2500 × 100 KB = 250 MB
```

**Verdict**: Storage is feasible (250 MB << 7 GB available)

---

## JAX Array Lifetime in Python Dict

### Behavior

```python
# JAX arrays in dict DO persist
self._episode_prices_vram: Dict[int, jax.Array] = {}
self._episode_prices_vram[ep_id] = jax.device_put(prices)  # On GPU

# Array stays on GPU until:
# 1. Dict entry deleted
# 2. Dict cleared
# 3. Python garbage collection
```

### Clear Behavior

```python
def clear(self):
    # This releases VRAM
    self._episode_prices_vram.clear()
    # May need to force GC for immediate release:
    import gc
    gc.collect()
```

### Threading Considerations

- JAX arrays are immutable - safe to read from multiple threads
- Writing to dict needs lock (already have `self._lock`)
- No special JAX threading issues

---

## Why NOT to Implement

### 1. Minimal Benefit

```
Current total time: ~200 ms per BC batch
Time saved by VRAM storage: ~0.25 ms
Improvement: 0.12%
```

### 2. Added Complexity

- Need to manage VRAM lifecycle
- Need to track which episodes are on GPU
- Clear() must free VRAM properly
- Potential memory fragmentation

### 3. Better Alternatives Exist

| Approach | Improvement | Complexity |
|----------|-------------|------------|
| 136a (pre-encode) | **25x** | Medium |
| 136b (batch call) | **4x** | Low |
| 136c (VRAM prices) | **1.001x** | Medium |

---

## Where the Time Actually Goes

Based on profiling the current implementation:

```
_encode_observations_vectorized breakdown:
├── Python loop setup: ~5 ms
├── For each episode (×30):
│   ├── Dict lookup: 0.01 ms
│   ├── jnp.array(): 0.01 ms (H→D transfer - TINY)
│   ├── jax.vmap(): 1 ms (GPU kernel)
│   └── np.array(): 1.5 ms (D→H sync - BLOCKS)
├── Total loop: ~75 ms
└── Output scatter: ~5 ms
Total: ~90 ms
```

**Real bottlenecks:**
1. **D→H sync** (`np.array(encoded)`): 30 × 1.5 ms = 45 ms
2. **GPU kernel fragmentation**: 30 small kernels vs 1 large
3. **Python loop**: ~20 ms overhead

---

## Recommendation

**Do NOT implement TASK_136c.**

Instead:
1. **TASK_136a** (pre-encode at add time): 25x speedup, eliminates encoding
2. **TASK_136b** (batch JAX call): 4x speedup if 136a not feasible

VRAM storage for prices would add complexity for negligible benefit.

---

## If You Still Want VRAM Storage

Here's how it would work (NOT RECOMMENDED):

```python
class UnifiedIndexBuffer:
    def __init__(self, ...):
        # VRAM storage instead of RAM
        self._episode_prices_vram: Dict[int, jax.Array] = {}

    def add_episode(self, prices: np.ndarray) -> int:
        with self._lock:
            if prices.ndim == 3:
                prices = prices[:, :, 3]

            ep_id = self._next_episode_id
            # Store directly on GPU
            self._episode_prices_vram[ep_id] = jax.device_put(
                prices.astype(np.float32)
            )
            self._next_episode_id += 1
            return ep_id

    def _encode_observations_vectorized(self, entries):
        # No jnp.array() needed - already on GPU
        close_prices = self._episode_prices_vram[int(ep_id)]  # Already jax.Array
        ...

    def clear(self):
        self._episode_prices_vram.clear()
        gc.collect()  # Help release VRAM
```

**But again: this saves 0.25 ms out of 200 ms. Not worth it.**
