# Bug: Sigma Broadcasting Error in Physics Core

## Priority: MEDIUM
**Status:** FALSE POSITIVE

## Verification (2026-01-01)

Line 259: `sigma = sigma_base[None, :] * jnp.ones((batch_size, 1))`

Broadcasting is correct:
- `sigma_base[None, :]` shape: (1, n_assets)
- `jnp.ones((batch_size, 1))` shape: (batch_size, 1)
- Result via broadcast: (batch_size, n_assets) - correct!
**Component:** `mcts_trader/jax_agent/prediction/physics_core.py`

## Problem Description

The volatility parameter `sigma` has incorrect broadcasting when applied to noise. Depending on implementation, this could cause shape mismatches or incorrect noise scaling per asset.

### Potential Issue

```python
# physics_core.py
noise = sigma * jax.random.normal(key, shape=(n_assets,))
# If sigma is scalar but should be per-asset, broadcasting is wrong
# If sigma is (n_assets,) but noise is (batch, n_assets), broadcasting fails
```

### Impact

- Volatility may not scale correctly per asset
- Shape errors in batched operations
- Incorrect physics simulation noise

## Solution

Ensure consistent shapes:

```python
# physics_core.py
def apply_noise(self, sigma, key, shape):
    # sigma should be (n_assets,) or broadcastable
    sigma = jnp.atleast_1d(sigma)
    noise = jax.random.normal(key, shape=shape)
    # Broadcast sigma to match noise shape
    sigma = jnp.broadcast_to(sigma, shape)
    return sigma * noise
```

## Files to Modify

| File | Change |
|------|--------|
| `physics_core.py` | Fix sigma broadcasting |

## Testing

1. Run with batch_size > 1 and n_assets > 1
2. Verify no shape errors
3. Check noise statistics per asset
