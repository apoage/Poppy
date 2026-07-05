# Bug: Double Loss Normalization

## Priority: HIGH
**Status:** FALSE POSITIVE

## Analysis (2026-01-01)

Not a bug. The code correctly normalizes over two **different** dimensions:
1. `.mean()` averages each step's loss over the **batch** dimension
2. `/= unroll_steps` averages the accumulated losses over **time** dimension

This is standard practice: `loss = mean_over_batch(sum_over_time(...)) / T`
**Component:** `mcts_trader/jax_agent/trainer.py`

## Problem Description

The training loss is normalized twice:
1. First by `jnp.mean()` over the batch
2. Then divided by `n_steps` again

This causes loss to be much smaller than intended, requiring larger learning rates.

### Location

```python
# trainer.py - loss computation
def compute_loss(batch):
    value_loss = jnp.mean(value_losses)      # Already normalized by batch
    policy_loss = jnp.mean(policy_losses)    # Already normalized by batch

    # Double normalization:
    total_loss = (value_loss + policy_loss) / n_steps  # Divides AGAIN!
```

### Impact

- Loss values ~100x smaller than expected
- Gradients proportionally smaller
- Learning rate must be ~100x larger to compensate
- Confusing loss values in logs (look artificially small)

### Example

```python
# If batch_size=256, n_steps=400:
# True loss = 5.0
# After jnp.mean(): 5.0 / 256 ≈ 0.0195 (correct normalization)
# After /n_steps: 0.0195 / 400 ≈ 4.88e-5 (way too small!)
```

## Solution

Remove the second normalization:

```python
# trainer.py
def compute_loss(batch):
    value_loss = jnp.mean(value_losses)
    policy_loss = jnp.mean(policy_losses)

    # NO second division
    total_loss = value_loss + policy_loss
    return total_loss
```

Or if per-step normalization is intended, don't use jnp.mean:

```python
# If normalizing per step:
value_loss = jnp.sum(value_losses) / n_steps  # Sum then normalize
# NOT: jnp.mean(value_losses) / n_steps
```

## Files to Modify

| File | Change |
|------|--------|
| `trainer.py` | Remove double normalization |

## Testing

1. Print raw loss values before and after fix
2. Verify gradient magnitudes are reasonable
3. May need to adjust learning rate after fix (likely reduce by ~100x)

## Warning

After fixing this, learning rate may need adjustment. Current LR may be tuned to compensate for small gradients.
