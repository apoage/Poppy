# BUG #11: Reward Double Symlog Transformation

**Severity:** ~~MEDIUM~~ NOT A BUG
**File:** `mcts_trader/jax_agent/trainer.py:592`
**Status:** INVESTIGATED - WORKING AS INTENDED

## Original Concern
The concern was that rewards are symlog'd in the environment, then `_scalar_to_categorical` applies another transformation, causing "double symlog".

## Investigation Result: NOT A BUG

There is NO double transformation. The two steps are different operations:

### Transformation Flow

1. **Environment** (vectorbt_env.py:1032):
   ```python
   reward = symlog(pct_return, threshold=1.0)
   ```
   - Converts percentage return → symlog value
   - Example: 20% return → symlog(20) ≈ 3.0
   - Result range: approximately [-5, 5] for typical trades

2. **Training** (trainer.py:592):
   ```python
   self._scalar_to_categorical(rewards[:, t - 1])
   ```
   - Takes the symlog'd value (e.g., 3.0)
   - Clips to [-750, 750] (no effect for typical values)
   - Maps to bin index (e.g., bin 377 out of 751)

### Why This Is Correct

- **Symlog** (step 1): Compresses the magnitude scale: `sign(x) * log1p(|x|)`
- **Categorical binning** (step 2): Discretizes for categorical cross-entropy loss

These are different transformations serving different purposes:
- Symlog: Handles reward scale variance (large vs small returns)
- Binning: Enables categorical loss (MuZero-style distributional learning)

### Efficiency Note

Per-step rewards (~[-5, 5]) use only a small portion of the [-750, 750] bin range (designed for MC returns). This is suboptimal but not incorrect. A potential optimization would be to use a smaller support for reward prediction.

## Investigation Checklist
- [x] Trace reward transformation in vectorbt_env.py - Line 1032, symlog applied once
- [x] Check if rewards in batch are symlog'd or raw - Symlog'd (correct)
- [x] Verify what _scalar_to_categorical expects - Any scalar, clips to support range
- [x] Determine if symlog should be applied once or not at all - Once is correct

## Conclusion

No fix needed. The reward pipeline correctly:
1. Applies symlog ONCE in the environment
2. Maps symlog'd values to categorical bins for loss computation
