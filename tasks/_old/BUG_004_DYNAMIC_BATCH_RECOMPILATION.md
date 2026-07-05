# BUG #4: Dynamic Batch Size Causes Recompilation

**Severity:** MEDIUM
**File:** `train_jax_fast.py:255`
**Status:** FIXED (2026-01-04)

## Code
```python
# Stack observations from active envs only
active_indices = [i for i in range(n_envs) if active[i]]
obs_batch = []
...
obs_batch = jnp.array(np.stack(obs_batch), dtype=jnp.float32)

# Batched MCTS - single GPU call for all envs!
actions_jax, policies_jax, stats = search_fn(params, step_rng, obs_batch, mask_batch)
```

## Issue
The batch size for search_fn (JIT-compiled) changes dynamically as environments finish episodes (active list shrinks). JAX JIT compiles a separate kernel for each unique input shape.

## Impact
Severe performance degradation. As episodes finish, the system triggers expensive XLA recompilations for every new batch size (e.g., 255, 254, 253...).

## Expected Fix
Pad the batch to n_envs (fixed size) and use a mask to ignore inactive environments, or use run_self_play_jax which handles this correctly.

## Investigation Checklist
- [ ] Verify run_self_play_batched uses dynamic batch sizes
- [ ] Check if run_self_play_jax already handles this correctly
- [ ] Determine which function is actually used in training
- [ ] Decide: fix run_self_play_batched or deprecate it
