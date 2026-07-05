# TASK_132e: Window Edge Future Leakage Fix

**Auditor**: Gemini CLI (TASK_132c)
**Date**: 2026-02-01
**Status**: FIXED

## Problem

When storing demos for steps < window_size, the observation encoding leaks future data.

**Example:**
- step=30, window_size=60
- window_start = max(30 - 60, 0) = 0
- Encoder slices prices[0:60]
- At decision time t=30, agent sees prices[31:59] = **29 steps of future**

**Location**: `mcts_trader/jax_agent/expert_buffer.py` + `train_jax_fast.py`

## Solution

**Option B (Selected)**: Only store demos where `step >= window_size`

- Simple implementation
- Sacrifices first 60 steps per episode (acceptable)
- No padding complexity
- Synthetic training adjusted to generate longer episodes (buffer for window fill)

## Changes Required

### 1. expert_buffer.py - Skip early steps in add_demo or encoding
Filter out steps < window_size when encoding observations.

### 2. train_jax_fast.py - Skip early steps in callback
Only call `expert_buffer.add_demo()` for steps >= window_size.

### 3. Synthetic environment - Ensure episodes long enough
Episodes should be at least `window_size + min_useful_steps` to have data after window fill.

## Impact

- Lose first 60 steps of demo data per episode
- ~600 step episodes → ~540 usable steps (10% loss, acceptable)
- Eliminates future leakage completely

## Fix Applied

### train_jax_fast.py (line ~648)
```python
# TASK_132e: Skip steps < window_size to prevent future leakage
window_size = expert_buffer.window_size if expert_buffer else 60
for step in range(n_steps):
    if step < window_size:
        continue
    # ... rest of loop
```

### expert_buffer.py (_encode_observations_vectorized)
```python
# TASK_132e: Filter out steps < window_size (defensive check)
valid_step_mask = steps >= self.window_size
if not np.all(valid_step_mask):
    # Filter to valid steps only, warn if significant
    steps = steps[valid_step_mask]
    batch_indices = batch_indices[valid_step_mask]

# No clamping needed now - all steps guaranteed >= window_size
window_starts = jnp.array(steps) - self.window_size
```
