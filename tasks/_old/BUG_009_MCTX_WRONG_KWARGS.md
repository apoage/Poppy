# Bug: Wrong Keyword Arguments to mctx

## Priority: HIGH
**Status:** FALSE POSITIVE

## Verification (2026-01-01)

Checked mctx API signature. All parameters used are valid:
- `gumbel_scale` - valid (default 1.0)
- `max_num_considered_actions` - valid (default 16)
- `qtransform` - valid keyword argument
- `max_depth` - valid (optional)

The mctx calls in mctx_search.py are correct.
**Component:** `mcts_trader/jax_agent/mctx_search.py`

## Problem Description

The mctx library calls use incorrect or non-existent keyword arguments. The mctx API may have changed or arguments are misspelled.

### Location

```python
# mctx_search.py - gumbel_muzero_policy call
policy_output = mctx.gumbel_muzero_policy(
    params=...,
    root=...,
    recurrent_fn=...,
    num_simulations=num_simulations,
    max_depth=max_depth,          # May not exist
    gumbel_scale=gumbel_scale,    # Check API
    ...
)
```

### Potential Issues

1. `max_depth` - mctx uses different parameter name
2. `gumbel_scale` - may be under different namespace
3. `qtransform` - API changed in recent versions

### Impact

- May crash at runtime
- May silently ignore parameters
- MCTS may not behave as intended

## Solution

### Step 1: Check mctx version and API

```python
import mctx
print(mctx.__version__)
help(mctx.gumbel_muzero_policy)
```

### Step 2: Fix parameter names

```python
# mctx_search.py - use correct API
from mctx import gumbel_muzero_policy, RecurrentFnOutput

policy_output = gumbel_muzero_policy(
    params=params,
    rng_key=rng_key,
    root=RootFnOutput(...),
    recurrent_fn=recurrent_fn,
    num_simulations=num_simulations,
    # Check these against current mctx docs
)
```

## Files to Modify

| File | Change |
|------|--------|
| `mctx_search.py` | Update to match current mctx API |

## Testing

1. Run mctx import and API inspection
2. Run MCTS search and verify no warnings
3. Check all parameters are applied

## Reference

- mctx GitHub: https://github.com/google-deepmind/mctx
- Check version in requirements.txt
