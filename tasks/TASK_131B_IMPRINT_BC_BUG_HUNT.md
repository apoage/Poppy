# TASK_131b: BUG HUNT - Imprint/BC System Information Loss & Garbage Feeding

**Status**: COMPLETE - 3 CRITICAL BUGS FOUND
**Investigator**: KIMI
**Date**: 2026-02-01

---

## Executive Summary

Found **3 CRITICAL BUGS** causing the Imprint/BC system to feed garbage or ineffective information:

| Bug | Severity | Issue | Impact |
|-----|----------|-------|--------|
| Bug 1 | **CRITICAL** | Wrong `window_start` passed to encoder | All demo observations are garbage |
| Bug 2 | **HIGH** | Threshold-based action selection | Demos teach mostly HOLD actions |
| Bug 3 | **MEDIUM** | Small demo batch size | BC loss impact diluted |

---

## Bug 1: CRITICAL - Wrong Window Start in Observation Encoding

**File**: `mcts_trader/jax_agent/expert_buffer.py:458-465`

### Problem
The `step` value stored in the buffer represents the timestep where the action was taken. However, `encode_observation_simple` expects `window_start` (the beginning of the lookback window), not the current step.

### Current Code (BROKEN)
```python
# Line 458: steps is the step index where action was taken
steps = entries['step'][mask].astype(np.int32)

# Line 463-465: Passing step directly as window_start (WRONG!)
encoded = jax.vmap(
    lambda s: encoder_fn(close_prices, s)
)(jnp.array(steps))
```

### Root Cause
- `step` = 100 means "action taken at timestep 100"
- `window_start` should be `step - window_size` = 40 (for 60-step window)
- By passing `step` directly, encoder slices `prices[100:160]` instead of `prices[40:100]`
- The observation encoded is from the FUTURE relative to when the action was taken

### Impact
- **ALL demo observations are garbage** - encoded from wrong timesteps
- BC loss computed on observations that don't match the actions
- Model learns noise instead of useful patterns
- bc_loss ≈ 0.1 appears negligible because gradients are random

### Fix
```python
# In expert_buffer.py _encode_observations_vectorized method:

# Line 458 - Add window_start calculation:
steps = entries['step'][mask].astype(np.int32)
window_starts = jnp.maximum(steps - self.window_size, 0)  # FIX: Calculate window start

# Line 463-465 - Use window_starts instead of steps:
encoded = jax.vmap(
    lambda ws: encoder_fn(close_prices, ws)
)(window_starts)
```

---

## Bug 2: HIGH - Threshold-Based Action Selection Produces All HOLD

**File**: `train_jax_fast.py:662-670`

### Problem
The optimal action computation uses a fixed threshold of 0.001 for returns:

```python
if best_return > 0.001:      # 0.1% return threshold
    optimal_action = best_asset * 5 + 2  # Buy
elif best_return < -0.001:   # -0.1% return threshold  
    optimal_action = best_asset * 5 + 0  # Sell
else:
    optimal_action = best_asset * 5 + 1  # Hold
```

### Root Cause
- Log returns of 0.001 = 0.1% - very small move in minute-bar data
- In normal market conditions, most minute-to-minute returns are < 0.1%
- Results in mostly HOLD actions being stored as "optimal"
- The oracle selected these assets as top-k by absolute return, but we convert to HOLD

### Impact
- Demos teach agent to HOLD rather than trade
- BC loss converges quickly (HOLD is easy to learn)
- No actual trading strategy is imprinted
- action_agreement appears high but meaningless (always predicting HOLD)

### Fix
Remove threshold - if asset is in top-k selected by oracle, use directional signal:
```python
# In train_jax_fast.py on_episode_complete_callback:

# Lines 662-670 - Replace with:
if best_return > 0:
    optimal_action = best_asset * 5 + 2  # Buy
else:
    optimal_action = best_asset * 5 + 0  # Sell
# Never HOLD - oracle selected this asset for a reason
```

---

## Bug 3: MEDIUM - Demo Batch Size May Be Too Small

**File**: `mcts_trader/jax_agent/trainer.py:275, 1673-1676`

### Problem
```python
expert_batch_size: int = 128   # Total samples from expert buffer
demo_ratio: float = 0.125       # 12.5% should be demos

# With batch_size=128 and demo_ratio=0.125:
# n_demo = 128 * 0.125 = 16 demos
# n_agent = 128 - 16 = 112 agent samples
```

If `demo_count < 16` (e.g., at startup), the buffer reduces demo count further.

### Impact
- At early training, very few demos may be sampled
- BC loss impact is diluted by small batch size
- 128 total samples may be too small for stable BC learning

### Fix
Option A: Increase batch size
```python
expert_batch_size: int = 512   # Was 128
```

Option B: Use pure demo batches
```python
expert_batch = expert_buffer.sample_batch(
    batch_size=self.expert_bc_config.expert_batch_size,
    demo_ratio=1.0,  # ALL from demos, no agent mixing
)
```

---

## Additional Findings

### Observation Shape Mismatch Check
**File**: `mcts_trader/jax_agent/expert_buffer.py:424`

Current code pre-allocates observations with shape `(batch_size, 27, n_assets, n_assets)`. With `n_assets=64`, this is `(batch_size, 27, 64, 64)`.

This matches the expected observation shape from `jax_encoder.py`. **No issue here** - shape is correct.

### Action Space Verification
**File**: `train_jax_fast.py:663-670`

Action encoding uses `asset_idx * 5 + action_type` where:
- 0 = Sell
- 1 = Hold  
- 2 = Buy
- 3 = LimitSell
- 4 = LimitBuy

This matches the main training action space (64 assets × 5 actions = 320). **No issue here**.

---

## Recommended Fix Priority

### P0 (Critical) - Implement Immediately
1. **Fix Bug 1** (window_start calculation) - This is the root cause of garbage observations

### P1 (High) - Implement Today  
2. **Fix Bug 2** (action threshold) - Remove 0.001 threshold for diverse actions

### P2 (Medium) - Monitor & Adjust
3. **Increase expert_batch_size** from 128 to 256 or 512
4. **Add comprehensive logging** to verify fixes

---

## Verification Steps After Fix

### 1. Verify Observation Encoding
Add debug print in `_encode_observations_vectorized`:
```python
print(f"[EXPERT ENCODE] step={steps[0]}, window_start={window_starts[0]}, "
      f"price_range=[{window_starts[0]}:{window_starts[0]+self.window_size}], "
      f"episode_len={len(close_prices)}")
```

Expected: `window_start = step - 60` (for 60-step window)

### 2. Verify Action Distribution
Add action counting in callback:
```python
action_counts = {'buy': 0, 'sell': 0, 'hold': 0}
# ... in loop ...
action_type = optimal_action % 5
if action_type == 2: action_counts['buy'] += 1
elif action_type == 0: action_counts['sell'] += 1
else: action_counts['hold'] += 1

print(f"[CALLBACK] Optimal action distribution: {action_counts}")
```

Expected: Roughly equal Buy/Sell, minimal Hold

### 3. Verify BC Loss Impact
After fixes, expected metrics:
- `bc_loss`: Should increase to 0.5-2.0 range (not stuck at 0.1)
- `expert_action_agreement`: Should start low (~0.1-0.3) and increase over time
- `expert_batch_size`: Should be stable at configured value

---

## Code Changes Required

### Change 1: expert_buffer.py (Line 458-465)
```python
# BEFORE:
steps = entries['step'][mask].astype(np.int32)
encoded = jax.vmap(
    lambda s: encoder_fn(close_prices, s)
)(jnp.array(steps))

# AFTER:
steps = entries['step'][mask].astype(np.int32)
window_starts = jnp.maximum(steps - self.window_size, 0)
encoded = jax.vmap(
    lambda ws: encoder_fn(close_prices, ws)
)(window_starts)
```

### Change 2: train_jax_fast.py (Lines 662-670)
```python
# BEFORE:
if best_return > 0.001:
    optimal_action = best_asset * 5 + 2
elif best_return < -0.001:
    optimal_action = best_asset * 5 + 0
else:
    optimal_action = best_asset * 5 + 1

# AFTER:
if best_return > 0:
    optimal_action = best_asset * 5 + 2  # Buy
else:
    optimal_action = best_asset * 5 + 0  # Sell
```

### Change 3: trainer.py (Line 275)
```python
# BEFORE:
expert_batch_size: int = 128

# AFTER:
expert_batch_size: int = 256  # Or 512
```

---

## Summary

The Imprint/BC system has been feeding garbage observations due to a critical window_start calculation bug. This explains:
- bc_loss ≈ 0.1 (negligible)
- action_agreement not improving
- Model not learning from demonstrations

**Fixing Bug 1 (window_start) is the highest priority** - it will make the BC loss meaningful.

**Fixing Bug 2 (action threshold) is second priority** - it will make demos teach actual trading.

After these fixes, the BC system should start functioning as designed.
