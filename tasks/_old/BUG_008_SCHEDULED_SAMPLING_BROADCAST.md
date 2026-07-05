# BUG #8: Scheduled Sampling Probability Broadcasting Failure

**Severity:** HIGH
**File:** `mcts_trader/jax_agent/muzero_model.py:349`
**Status:** FIXED

## Code
```python
use_teacher = jax.random.uniform(sample_rng) < scheduled_sampling_prob
# ...
current_log_prices = jnp.where(
    use_teacher[:, None] if use_teacher.ndim == 1 else use_teacher,
    teacher_prices_k,
    current_log_prices
)
```

## Issue
`jax.random.uniform(sample_rng)` returns a scalar (shape `()`), so `use_teacher` is a scalar boolean. The check `use_teacher.ndim == 1` is ALWAYS FALSE, so it falls through to `use_teacher` directly. Then `use_teacher[:, None]` would fail if it somehow were 1D, but since it's a scalar, the comparison broadcasts incorrectly.

## Impact
Scheduled sampling applies uniformly across all batch elements (same random draw for entire batch) rather than per-element randomization, reducing stochasticity in teacher forcing.

## Expected Fix
Use `jax.random.uniform(sample_rng, (batch,)) < scheduled_sampling_prob` to get per-batch-element decisions.

## Investigation Checklist
- [x] Verify the batch size is available at this point - Yes, at line 331
- [x] Confirm scalar vs per-element behavior - Confirmed scalar was used
- [x] Test with explicit per-element random draws - Fixed

## Fix Applied
Changed from scalar to per-batch-element random:
```python
# Before (scalar - same decision for all batch elements):
use_teacher = jax.random.uniform(sample_rng) < scheduled_sampling_prob

# After (per-element - independent decisions):
use_teacher = jax.random.uniform(sample_rng, (batch,)) < scheduled_sampling_prob
```

Now `use_teacher` has shape `(batch,)`, making `use_teacher.ndim == 1` true, so the broadcast logic `use_teacher[:, None]` works correctly.
