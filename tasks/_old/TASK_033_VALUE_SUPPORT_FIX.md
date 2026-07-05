# TASK_033: Value Support Scale Fix for PopArt

## Status: IMPLEMENTED (2026-01-04)

## Problem Statement

The model was "blind steering" - achieving low value MSE (~0.8) but not actually learning value differences. Root cause analysis revealed:

### The Information Bottleneck

| Component | Before Fix | After Fix |
|-----------|-----------|-----------|
| Value Support Range | [-750, +750] | [-10, +10] |
| Number of Bins | 751 | 201 |
| Bin Width | 2.0 | 0.1 |
| Information Utilization | 0.5% | 31.6% |

### Why This Happened

1. Value support was designed for **raw MC returns** (symlog over 200 steps ≈ ±700)
2. PopArt normalizes targets to **mean=0, std=1** (range ≈ ±3)
3. Normalized targets only mapped to bins 373-377 (5 bins out of 751!)
4. Network learned to predict bin 375 (value=0) for everything → MSE ≈ 0.8
5. No actual value learning occurred

### Evidence

```python
# Before fix: Random logits
predicted_values: range=[-50, +50], mean=-3.0, std=19.5
normalized_targets: range=[-3, +3], mean=0, std=1.0
MSE = 392  # HUGE mismatch!

# Network converges to center bin (value=0) → MSE ≈ 1.0
# This is a local minimum, not actual learning!

# After fix: Random logits
predicted_values: range=[-1, +1], mean=0.07, std=0.52
normalized_targets: range=[-3, +3], mean=0, std=1.0
MSE = 1.2  # Same scale, network can learn!
```

## Solution

Changed value support in `mcts_trader/jax_agent/__init__.py`:

```python
# Before
VALUE_SUPPORT_MIN = -750
VALUE_SUPPORT_MAX = 750
VALUE_SUPPORT_SIZE = 751  # 2-unit resolution

# After
VALUE_SUPPORT_MIN = -10
VALUE_SUPPORT_MAX = 10
VALUE_SUPPORT_SIZE = 401  # 0.05 resolution, more bins than actions (320)
```

**Key insight:** 401 bins > 320 actions ensures value head can distinguish all action values.

## Files Modified

| File | Change |
|------|--------|
| `mcts_trader/jax_agent/__init__.py` | Updated VALUE_SUPPORT_* constants |
| `mcts_trader/jax_agent/trm_flax.py` | Updated comments (2 places) |
| `mcts_trader/jax_agent/muzero_model.py` | Updated comment |

## Breaking Changes

**EXISTING CHECKPOINTS ARE INCOMPATIBLE!**

The model architecture changed from 751-bin to 401-bin value heads. You must:
1. Start training from scratch (no `--load-checkpoint`)
2. Delete old checkpoints in `jax_ckpt_fast/`

## Expected Improvement

After this fix:
- Network can express value differences at 0.1 resolution
- Gradient signal properly flows for value learning
- MCTS root values should correlate with actual MC returns
- Value MSE should decrease during training (not just reach ~0.8 plateau)

## Related Tasks

- TASK_031: PopArt denormalization for logging (enables proper value visualization)
- TASK_021: PopArt implementation (introduced the normalization)

## Notes

- If PopArt is disabled, the support range may need to be increased
- The 201-bin count provides 0.1 resolution, which is fine for normalized values
- Covers ±10 sigma (effectively all practical cases)
