# Bug: Double JIT in MCTS Search

## Priority: HIGH
**Status:** FALSE POSITIVE + CLEANUP DONE

## Analysis (2026-01-01)

No double JIT exists - only one `jax.jit(self._search)` at line 56.

However, found and removed **dead code**: A vmap call at line 157 was being immediately
overwritten by a more efficient batch inference. Removed the dead vmap code.
**Component:** `mcts_trader/jax_agent/mctx_search.py`

## Problem Description

The MCTS search functions have `@jax.jit` decorators but are called inside `run_mcts_search()` which is also JIT-compiled. This causes:
1. Redundant compilation overhead
2. Potential issues with nested JIT behavior
3. Memory fragmentation from multiple compilation units

### Locations

```python
# mctx_search.py

@jax.jit  # First JIT
def get_root_output(model, params, root_obs, ...):
    ...

@jax.jit  # Second JIT
def run_mcts_search(model, params, obs, ...):
    root_output = get_root_output(...)  # Calling JIT from JIT
    ...
```

### Impact

- Slower first-run compilation
- Inefficient memory usage
- Potential XLA fusion prevented
- Harder to debug/profile

## Solution

### Option A: Remove inner JIT decorators

```python
# mctx_search.py

# NO @jax.jit - will be traced when called from run_mcts_search
def get_root_output(model, params, root_obs, ...):
    ...

@jax.jit  # Single JIT at top level
def run_mcts_search(model, params, obs, ...):
    root_output = get_root_output(...)  # Inlined by JIT
    ...
```

### Option B: Use static_argnums correctly

If functions need separate caching:

```python
@partial(jax.jit, static_argnums=(0,))
def run_mcts_search(model, params, obs, ...):
    ...
```

## Files to Modify

| File | Change |
|------|--------|
| `mctx_search.py` | Remove redundant @jax.jit decorators on inner functions |

## Testing

1. Run MCTS search and measure compilation time
2. Profile memory usage
3. Verify search still produces correct results

## Notes

JIT compilation is transitive - inner functions get compiled as part of outer JIT.
