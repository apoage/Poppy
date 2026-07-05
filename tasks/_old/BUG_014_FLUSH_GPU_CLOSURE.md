# BUG #14: flush_gpu_buffers Captures Mutable Closure Variables

**Severity:** MEDIUM
**File:** `train_jax_fast.py:536-567`
**Status:** TO INVESTIGATE

## Code
```python
def flush_gpu_buffers(batch_start, gpu_obs, gpu_actions, gpu_rewards,
                      gpu_values, gpu_policies, gpu_dones, active_states):
    """Transfer accumulated GPU buffers to CPU storage."""
    # ...
    all_observations[:, batch_start:end_step] = obs_np
    # ...
```

## Issue
`flush_gpu_buffers` is a nested function that captures `all_observations`, `all_actions`, etc. from the enclosing scope. These are mutable numpy arrays. In JAX, this pattern can cause issues with closures over mutable state, especially if the function is traced/jitted later.

Currently it works because the function isn't jitted, but it's fragile design.

## Impact
Potential data corruption if code is refactored to JIT this function.

## Expected Fix
Pass all storage arrays as explicit parameters instead of relying on closure capture.

## Investigation Checklist
- [ ] Verify function is never JIT-compiled
- [ ] Consider refactoring for explicit parameters
- [ ] Add comments warning against JIT-ing
