# BUG #17: RNG Key Split Inefficiency in Loop

**Severity:** LOW
**File:** `mcts_trader/jax_agent/muzero_model.py:344`
**Status:** TO INVESTIGATE

## Code
```python
for k in range(num_steps):
    rng, noise_rng, sample_rng = jax.random.split(rng, 3)
```

## Issue
Inside a Python loop that will be traced by JAX, this creates a chain of split operations. Each iteration depends on the previous, preventing parallelization of the noise generation.

## Impact
Slower compilation and execution of multi-step prediction.

## Expected Fix
Pre-split all RNG keys outside the loop:
```python
all_rngs = jax.random.split(rng, num_steps * 2)
# Then index into all_rngs[k*2] and all_rngs[k*2+1] in loop
```

## Investigation Checklist
- [ ] Measure actual performance impact
- [ ] Verify pre-splitting maintains correct RNG semantics
- [ ] Implement and benchmark fix
