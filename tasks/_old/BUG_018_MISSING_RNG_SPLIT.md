# Bug: Missing RNG Key Splits

## Priority: MEDIUM
**Status:** DEFERRED

Requires careful audit of all RNG usage. Current code appears to
split keys correctly in main paths (training loop, self-play).
**Component:** Multiple JAX files

## Problem Description

JAX requires explicit RNG key splitting for reproducibility and correctness. Some code paths may reuse RNG keys without splitting.

### Correct Pattern

```python
key, subkey1, subkey2 = jax.random.split(key, 3)
noise1 = jax.random.normal(subkey1, shape)
noise2 = jax.random.normal(subkey2, shape)  # Different from noise1
```

### Incorrect Pattern

```python
noise1 = jax.random.normal(key, shape)
noise2 = jax.random.normal(key, shape)  # SAME as noise1!
```

### Potential Locations

1. MCTS search - multiple random choices
2. Self-play - action sampling
3. Physics core - noise generation
4. Gumbel noise in policy

### Impact

- Non-random behavior when keys reused
- Correlated noise when should be independent
- Reproducibility issues

## Solution

### Step 1: Audit RNG usage

```bash
grep -rn "jax.random" mcts_trader/
```

### Step 2: Add splits where missing

```python
# Before each random operation, split key
key, subkey = jax.random.split(key)
result = jax.random.normal(subkey, shape)
```

## Files to Modify

| File | Change |
|------|--------|
| Multiple | Add proper RNG splits |

## Testing

1. Run with fixed seed
2. Verify random operations produce different values
3. Check reproducibility with same seed
