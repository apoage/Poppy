# BUG #6: PopArt Weight Correction Applied Incorrectly

**Severity:** ~~CRITICAL~~ NOT A BUG
**File:** `mcts_trader/jax_agent/trainer.py:788-806`
**Status:** INVESTIGATED - WORKING AS INTENDED

## Original Concern
The concern was that PopArt weight correction is applied before gradient update, and then "discarded" when gradients modify the weights.

## Investigation Result: NOT A BUG

The implementation is **correct** per the PopArt paper. Here's why:

### How PopArt Works

1. **Before batch**: Model outputs normalized values using current stats (μ_old, σ_old)
2. **Update stats**: (μ_old, σ_old) → (μ_new, σ_new)
3. **Correct weights**: W_new = W_old * (σ_old/σ_new), b_new = ...
4. **Train**: Compute gradients on corrected_params, apply gradients

### Why This Is Correct

The correction ensures **continuity of denormalized outputs** when stats change:
- Before correction: denorm(v_old(h)) = v_old(h) * σ_old + μ_old
- After correction: denorm(v_new(h)) = v_new(h) * σ_new + μ_new = same value!

The gradient update then modifies the **corrected weights** to better fit new targets:
```
final_weights = corrected_params - lr * gradients(corrected_params)
```

This is intentional - the correction provides a smooth starting point, then gradients adapt to new targets. The correction is NOT meant to persist unchanged.

### Verification

Looking at `_train_step`:
```python
# Line 703: gradients computed w.r.t. state.params (corrected params)
(loss, metrics), grads = grad_fn(state.params, batch, rng, ...)

# Line 706: gradients applied TO state.params (corrected params)
new_state = state.apply_gradients(grads=grads)
```

The correction IS preserved as the base for gradient descent. This matches the PopArt paper's algorithm.

## Conclusion

No fix needed. The PopArt implementation correctly:
1. Updates statistics
2. Corrects weights to preserve denormalized outputs
3. Trains on normalized targets starting from corrected weights
