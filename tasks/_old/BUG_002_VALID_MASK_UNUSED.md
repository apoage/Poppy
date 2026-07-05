# BUG #2: Valid Mask Computed But Ignored

**Severity:** HIGH
**File:** `mcts_trader/jax_agent/trainer.py:504`
**Status:** FIXED (2026-01-04)

## Code
```python
# Validity mask for handling buffer wraparound (BUG_006 fix)
valid_mask = batch.get('valid', jnp.ones(observations.shape[0], dtype=jnp.bool_))
num_valid = jnp.maximum(jnp.sum(valid_mask), 1.0)  # Avoid div by zero
...
muzero_loss = ( ... )
# No usage of valid_mask in loss calculation
```

## Issue
The valid_mask is calculated but ignored. The losses (value_loss, policy_loss, etc.) are computed using `.mean()`, which averages over the entire batch, including invalid samples (which contain zeros or garbage data).

## Impact
The model trains on invalid/garbage data from the replay buffer boundaries, introducing noise and potentially biasing the model towards zero outputs (if garbage is zeros).

## Expected Fix
Mask the losses before averaging: `loss = (loss * valid_mask).sum() / num_valid`.

## Investigation Checklist
- [ ] Verify valid_mask is correctly computed in replay buffer
- [ ] Identify all loss computations that need masking
- [ ] Ensure per-sample losses are computed before masking
- [ ] Test that masked loss produces valid gradients
