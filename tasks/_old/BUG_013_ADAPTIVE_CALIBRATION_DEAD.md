# BUG #13: Value Scale/Shift Never Updated Adaptively

**Severity:** ~~MEDIUM~~ NOT A BUG
**File:** `train_jax_fast.py:2055-2088`
**Status:** INVESTIGATED - FEATURE IS IMPLEMENTED

## Original Concern
The concern was that `--adaptive-calibration` flag is advertised but never implemented.

## Investigation Result: NOT A BUG

The feature IS implemented in train_jax_fast.py:2055-2088:

```python
if adaptive_calibration:
    calibration_values.append(avg_value)
    calibration_rewards.append(score)

    # Recompute scale every interval after warmup
    if epoch >= calibration_start_epoch and (epoch + 1) % calibration_interval == 0:
        if len(calibration_values) >= 25:
            vals = np.array(calibration_values)
            rews = np.array(calibration_rewards)

            # Compute optimal scale: match std of predictions to std of rewards
            val_std = np.std(vals)
            rew_std = np.std(rews)

            if val_std > 0.01:  # Avoid division by zero
                optimal_scale = rew_std / val_std
                new_scale = 0.7 * search.value_scale + 0.3 * optimal_scale
                new_scale = np.clip(new_scale, 0.5, 4.0)
                search.value_scale = float(new_scale)  # ← UPDATES MCTS SEARCHER

                # Also compute optimal shift
                search.value_shift = float(...)  # ← UPDATES MCTS SEARCHER
```

The feature:
1. Collects predicted values and actual rewards
2. Every 25 epochs after epoch 50, recalculates optimal scale/shift
3. Uses exponential smoothing (0.7/0.3) to avoid sudden jumps
4. Clamps scale to [0.5, 4.0] for stability
5. Actually updates `search.value_scale` and `search.value_shift`

## Investigation Checklist
- [x] Check if --adaptive-calibration is parsed - Yes, line 993
- [x] Determine if feature was implemented - Yes, lines 2055-2088
- [x] Verify it actually updates MCTS values - Yes, lines 2080 and 2086

## Conclusion

No fix needed. The feature is fully implemented and working.
