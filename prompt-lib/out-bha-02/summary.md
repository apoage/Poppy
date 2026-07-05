# Executive Summary: Trade Action Learning Regression

## Problem

Between commits `06c44ab` (TASK_692/693) and `77cb089` (TASK_695), agree_trade collapsed from **~15% to ~7%**. The model became a "turtle" — predicting HOLD everywhere instead of learning trade actions.

## Root Cause

The primary regression is **BUG-2: Confidence Channel Misalignment** in commit `a86c206`.

### The Bug

The confidence channel (channel 33) provides the model with feedback about its own confidence from the previous timestep. During model-state rollout, the confidence is fed back into the next observation. However, the implementation emits `prev_conf` (initialized to zeros) instead of `model_confidence` (the actual computed confidence):

```python
# train_easy.py:302 (WRONG)
return (new_pos, new_pnl, new_steps, new_h_fast, new_h_slow, model_confidence), (state_ch, prev_conf)
```

**Impact**:
- Step 0: Model sees `conf = 0` for all assets
- Step 1+: Model sees lagged confidence (from step w-1, not current)
- During inference: Model sees correct `prev_confidence` from its own prediction

The model cannot learn to use the confidence signal because it receives zeros/lagged values during training. With corrupted state channel feedback, the safest action (HOLD) is reinforced.

## Secondary Contributing Factors

### 1. BUG-1: Returns Alignment Inconsistency
- Single-episode path and batched path use different slicing for returns
- Can cause PnL channel (ch28) to be incorrect
- Fix: Ensure consistent `log_returns[ws - 1:T]` slicing

### 2. BUG-7: Trade-Biased Sampling Deactivation
- Trade indices filtered by `max_offset` are often empty
- Reduces trade examples by ~30%
- Fix: Add fallback when `valid_trades` is empty

### 3. Latent Cache + Model-State Rollout Incompatibility
- 80% of cached episodes use stale parameters
- State channels don't match model's current decisions
- Fix: Disable latent cache with `autoregressive_state=True`

## Recommended Action Plan

1. **FIX BUG-2 IMMEDIATELY** (train_easy.py:302):
   ```python
   # Change:
   return (new_pos, new_pnl, new_steps, new_h_fast, new_h_slow, model_confidence), (state_ch, model_confidence)
   ```

2. **FIX BUG-1** (train_easy.py:577):
   ```python
   # Change:
   returns_aligned = log_returns[ws - 1:T]
   ```

3. **FIX LATENT CACHE** (train_easy.py:1553):
   ```python
   # Add before latent cache enablement:
   if autoregressive_state:
       args.no_latent_cache = True  # Incompatible with model-state rollout
   ```

4. **ADD DIAGNOSTICS** (train_easy.py:1144):
   Log trade sample coverage to verify BUG-7 hypothesis

## Expected Outcome

After fixing BUG-2, the model should see correct confidence channels during training, enabling it to:
- Use confidence as a meaningful signal for uncertainty
- Learn trade actions properly
- Recover `agree_trade` to ~15% by epoch 80

Additional fixes (BUG-1, latent cache, trade sampling) will provide further stability and ensure consistent training behavior.

## Files Modified

- `train_easy.py`: Lines 302, 577, 1553
- Diagnostic logging to be added at line 1144

## Testing Recommendation

1. Apply BUG-2 fix and run training for 10 epochs
2. Monitor `agree_trade` — should increase from ~7% baseline
3. Apply remaining fixes and verify continued improvement
4. Run full training to epoch 80 and confirm `agree_trade ~15%`
