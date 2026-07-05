# TASK_135: Eliminate Price Cache - Direct Price Storage in Expert Buffer

**Date**: 2026-02-02
**Status**: IMPLEMENTED ✓
**Priority**: HIGH - saves 12GB disk, eliminates VRAM→RAM→disk copy overhead

## Problem

Current BC/Imprint flow has redundant data copying:
```
jax_prices (VRAM) → copy slice → price_cache (disk 12GB) → expert_buffer refs episode_id → BC reads from disk
```

This is wasteful because:
1. Prices already exist as numpy snapshot in callback (`_full_prices_ref`)
2. Copying to disk adds latency
3. 12GB disk files created per run (pre-allocated for 10k episodes × 1k steps)
4. Cache must be cleared on data regeneration anyway
5. Actual data per epoch is only ~5-25MB, not 12GB

## Key Discovery

`result['_full_prices_ref']` is already a **safe numpy snapshot**:

```python
# self_play.py line 728
cpu_prices = np.array(prices)  # JAX array → numpy COPY

# self_play.py line 869
result['_full_prices_ref'] = cpu_prices  # Passed to callback
```

This snapshot is:
- A **copy** (not a reference to jax_prices)
- On **CPU** (numpy array)
- **Safe from reassignment** when jax_prices changes

## Validated: All Three Use Cases

| Use Case | Source | Shape | Path to Callback | Valid? |
|----------|--------|-------|------------------|--------|
| **Synthetic** | `jax_prices` | (n_steps, n_assets) | `np.array(prices)` → `_full_prices_ref` | ✓ |
| **Bella** | `generate_bella_prices_on_device()` | (seq_len, n_assets) | `np.array(prices)` → `_full_prices_ref` | ✓ |
| **Historical** | `jax_prices` (from DB) | (n_steps, n_assets) | `np.array(prices)` → `_full_prices_ref` | ✓ |

**All three follow same validated path:**
```
prices_for_play (JAX array, GPU)
    ↓
run_self_play_jax(prices_for_play, ...)
    ↓
line 728: cpu_prices = np.array(prices)  ← COPY to CPU
    ↓
line 869: result['_full_prices_ref'] = cpu_prices
    ↓
on_episode_complete callback receives numpy snapshot ✓
```

## Proposed Solution (Simplified)

Instead of storing episode_id references, **store prices directly in expert_buffer entries**:

```
OLD: expert_buffer stores (episode_id, step) → later looks up prices from disk cache
NEW: expert_buffer stores (prices_slice, step) → prices already embedded, no lookup
```

### Data Flow After Fix

```
Episode completes:
  - callback receives result['_full_prices_ref'] (numpy, ~5MB)
  - callback receives result['prices'] (episode slice)
  - expert_buffer.add_demo(prices=episode_prices, step=step, ...)

BC training:
  - expert_buffer.sample_batch()
  - Prices already in demo entry, no external lookup
  - Reconstruct observation directly from stored prices
```

### Memory Cost (Acceptable)

Per epoch with 50 parallel envs:
```
50 envs × 400 steps × 64 assets × 4 bytes = 5.1 MB
```

Even storing 1000 demos:
```
1000 demos × 400 steps × 64 assets × 4 bytes = 102 MB
```

vs current: **12.8 GB** pre-allocated on disk

## When to Clear Expert Buffer

| Event | Synthetic | Bella | Historical |
|-------|-----------|-------|------------|
| Level switch (weighted) | ✓ Clear | N/A | N/A |
| Data regenerate interval | ✓ Clear | N/A | N/A |
| Noise schedule regenerate | ✓ Clear | N/A | N/A |
| Bella generates (every epoch!) | N/A | ✓ Clear | N/A |
| Period switch (--periods-file) | N/A | N/A | ✓ Clear |

**Critical: Bella mode regenerates every epoch**, so expert_buffer must clear every epoch.

## Implementation Steps

1. [x] Modify `expert_buffer.add_demo()` to accept prices directly
2. [x] Store prices slice in demo entry (not episode_id reference)
3. [x] Update `_encode_observations_vectorized()` to use stored prices
4. [x] Remove `price_cache` dependency from expert_buffer
5. [x] Remove `price_cache` initialization from train_jax_fast.py
6. [x] Add `expert_buffer.clear()` calls at data regeneration points:
   - [x] Weighted level switch
   - [x] Noise schedule regeneration
   - [x] Regenerate interval
   - [x] Bella generation (every epoch)
   - [x] Period switch in historical mode
7. [ ] Test with synthetic data
8. [ ] Test with historical data
9. [ ] Test with Bella data
10. [ ] Delete or deprecate price_cache.py (optional - can keep for reference)

## Files to Modify

```
mcts_trader/jax_agent/expert_buffer.py  - Store prices directly, remove price_cache dep
train_jax_fast.py                        - Remove price_cache init, add clear() calls
mcts_trader/jax_agent/price_cache.py    - Delete or deprecate
```

## Code Locations for Clear Calls

```python
# train_jax_fast.py - Add expert_buffer.clear() at these locations:

# Line 821: Weighted level switch
jax_env_states, jax_prices, jax_env_config = pregenerated_levels[new_level]
if expert_buffer: expert_buffer.clear()  # ADD THIS

# Line 792: Noise schedule regeneration
jax_env_states, jax_prices, jax_env_config = setup_jax_environment(config, seed=epoch * 1000)
if expert_buffer: expert_buffer.clear()  # ADD THIS

# Line 841: Regenerate interval
jax_env_states, jax_prices, jax_env_config = setup_jax_environment(config, seed=epoch * 1000)
if expert_buffer: expert_buffer.clear()  # ADD THIS

# Line 883: Bella generation (inside epoch loop - clears every epoch!)
prices_for_play = generate_bella_prices_on_device(...)
if expert_buffer: expert_buffer.clear()  # ADD THIS
```

## Benefits

| Metric | Before | After |
|--------|--------|-------|
| Disk usage | 12GB per run | 0 |
| Memory per epoch | 12GB pre-allocated | ~5-25MB actual |
| Data copy path | VRAM→RAM→disk→RAM | VRAM→RAM (once) |
| Latency | Disk I/O on sample | Direct memory access |
| Complexity | price_cache + expert_buffer | expert_buffer only |

## Rollback

If issues arise, price_cache can be restored - it's a separate module with no dependencies on this change.

## Edge Cases Verified

1. **jax_prices reassignment** - Safe because callback receives numpy copy, not reference
2. **Async regeneration** - Safe because callback fires inside run_self_play_jax, before reassignment
3. **JIT cache** - Line 835 clears JIT cache after level switch (already in place)
4. **GPU memory** - No change; prices copied to CPU same as before
5. **Array lifetime** - Numpy copy owned by expert_buffer entry, no dangling references
