# TASK_031: PopArt Denormalization for Logging

## Status: IMPLEMENTED (2026-01-04)

### Implementation Summary
1. Added `denormalize_value()` helper function to `self_play.py`
2. Added `popart_state` parameter to all three self_play functions:
   - `run_self_play()`
   - `run_self_play_batched()`
   - `run_self_play_jax()`
3. MCTS root values are now denormalized before logging (in original scale)
4. Added `popart_mean` and `popart_std` columns to CSV logging
5. Updated `train_jax_fast.py` to pass `trainer.popart.state` to self_play calls

### Files Modified
- `mcts_trader/training/self_play.py` - Added denormalization logic
- `train_jax_fast.py` - Pass popart_state, add CSV columns

## Problem Statement

MCTS Root Value vs Actual MC Returns shows NO CORRELATION (cloud pattern) despite 1000+ epochs of training. Value MSE ~0.8 suggests learning is happening, but diagnostic plots are useless.

**Root Cause**: PopArt normalizes training targets but logged MCTS values are never denormalized.

| Metric | Scale | Range |
|--------|-------|-------|
| MCTS Root Value | Normalized | -0.8 to +0.8 |
| Actual MC Returns | Original | -4% to +0.75% |

Comparing normalized predictions to original-scale returns = meaningless comparison.

## Current Flow (Broken)

```
Training:
  MC Returns (original) → PopArt.normalize() → Network learns normalized targets

Logging:
  Network output → _categorical_to_scalar() → CSV (normalized, wrong scale!)
  MC Returns → CSV (original scale)

Result: No correlation visible because different scales
```

## Solution

### Option A: Denormalize at Logging (Recommended)

Pass PopArt state to self_play and denormalize MCTS root values:

```python
# self_play.py - run_self_play()
def run_self_play(..., popart_state=None):
    ...
    # After getting MCTS root value
    if 'root_value' in stats:
        value_scalar = stats['root_value'][0]
        # Denormalize if PopArt is active
        if popart_state is not None:
            value_scalar = value_scalar * popart_state.std + popart_state.mean
        mcts_root_values.append(float(value_scalar))
```

### Option B: Log Both + PopArt Stats

Log raw and let post-processing handle it:

```python
# CSV columns to add:
popart_mean, popart_std, mcts_root_value_raw, mcts_root_value_denorm
```

### Option C: Normalize MC Returns in CSV

Alternative: normalize MC returns for comparison instead of denormalizing predictions.

## Implementation Plan

### Phase 1: Pass PopArt State to Self-Play

```python
# train_jax_fast.py - modify self_play calls

# Get PopArt state if enabled
popart_state = None
if trainer.use_popart and trainer.popart is not None:
    popart_state = trainer.popart.state

# Pass to self_play
results = run_self_play_jax(
    ...,
    popart_state=popart_state,  # NEW
)
```

### Phase 2: Denormalize in Self-Play

```python
# mcts_trader/training/self_play.py

from mcts_trader.jax_agent.popart import PopArtState, denormalize_values

def run_self_play(..., popart_state: Optional[PopArtState] = None):
    ...
    # In the step loop, after getting MCTS value:
    if 'root_value' in stats:
        value_scalar = stats['root_value'][0]
        if popart_state is not None:
            # Denormalize to original scale
            std = jnp.sqrt(popart_state.var + 1e-8)
            value_scalar = value_scalar * std + popart_state.mean
        mcts_root_values.append(float(value_scalar))
```

### Phase 3: Add PopArt Stats to CSV

```python
# train_jax_fast.py - CSV header
popart_mean, popart_std

# CSV row
popart_mean = trainer.popart.mean if trainer.use_popart else 0.0
popart_std = trainer.popart.std if trainer.use_popart else 1.0
```

## Files to Modify

| File | Changes |
|------|---------|
| `mcts_trader/training/self_play.py` | Add `popart_state` param, denormalize values |
| `train_jax_fast.py` | Pass popart_state to self_play, add CSV columns |
| `mcts_trader/jax_agent/popart.py` | Export `PopArtState` type (if needed) |

## Expected Results

After fix:
- MCTS Root Value range: -4% to +1% (matches MC returns scale)
- Scatter plot should show correlation if model is learning
- Value MSE can be properly interpreted

## Validation

```python
# Quick check: denormalized values should match MC return scale
assert abs(mcts_root_value_denorm.mean() - mc_returns.mean()) < 1.0
assert abs(mcts_root_value_denorm.std() - mc_returns.std()) < 0.5
```

## Notes

- This is a LOGGING fix, not a training fix
- MCTS action selection works correctly (relative ordering preserved)
- The correlation should exist - if not visible after this fix, there's a deeper issue
- Consider adding `--log-popart-stats` flag to control verbosity

## Priority

**HIGH** - Without this fix, value function diagnostics are useless. Can't tell if model is learning values correctly.
