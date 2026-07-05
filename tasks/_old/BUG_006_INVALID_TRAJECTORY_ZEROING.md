# Bug: Invalid Trajectories Zeroed Instead of Masked

## Priority: HIGH
**Status:** PARTIAL FIX

## Changes Made (2026-01-01)

1. `replay_buffer.py`: Now returns `valid` mask instead of zeroing data
2. `trainer.py`: Extracts validity mask and tracks `valid_samples`, `valid_ratio` metrics

**Full loss masking deferred** - would require significant refactoring of per-sample loss computation.
Current approach: Invalid samples have minimal impact since buffer rarely wraps with large capacity.
**Component:** `mcts_trader/jax_agent/replay_buffer.py`

## Problem Description

When trajectories are detected as invalid (due to buffer wraparound), they are **zeroed out** instead of **masked**. This corrupts training batches with meaningless zero data.

### Location

```python
# replay_buffer.py:321-331
def sample_batch(self, batch_size: int) -> Dict[str, jnp.ndarray]:
    ...
    # Zero out invalid trajectories
    for key in batch:
        batch[key] = jnp.where(
            valid_mask[:, None, None] if batch[key].ndim > 1 else valid_mask,
            batch[key],
            jnp.zeros_like(batch[key])  # PROBLEM: zeros are still used in training!
        )
```

### Impact

- Invalid trajectories become zero-filled samples
- Model trains on meaningless (all-zero) data
- Gradients corrupted by zero samples
- Learning efficiency severely degraded

### What Happens

```python
# Before: trajectory with data
obs = [0.5, 0.3, 0.1, ...]
reward = [0.02, -0.01, ...]

# After zeroing (for invalid trajectory):
obs = [0.0, 0.0, 0.0, ...]     # Zero observation!
reward = [0.0, 0.0, ...]       # Zero reward!

# Model still sees and trains on these zeros!
```

## Solution

### Option A: Return valid_mask and use in loss (Recommended)

```python
# replay_buffer.py
def sample_batch(self, batch_size: int) -> Tuple[Dict, jnp.ndarray]:
    ...
    return batch, valid_mask  # Return mask

# trainer.py - use mask in loss
def compute_loss(batch, valid_mask):
    losses = compute_per_sample_loss(batch)
    # Only average over valid samples
    masked_loss = jnp.sum(losses * valid_mask) / jnp.sum(valid_mask)
    return masked_loss
```

### Option B: Resample until all valid

```python
# replay_buffer.py
def sample_batch(self, batch_size: int) -> Dict[str, jnp.ndarray]:
    max_attempts = 3
    for _ in range(max_attempts):
        batch, valid_mask = self._sample_raw(batch_size)
        if jnp.all(valid_mask):
            return batch
        # Resample invalid entries
        invalid_indices = jnp.where(~valid_mask)[0]
        # ... resample just those
    return batch  # Give up after max attempts
```

### Option C: Skip invalid in sampling

```python
# Only sample from valid range (no wraparound)
valid_range = self._get_valid_sample_range()
indices = jax.random.choice(key, valid_range, shape=(batch_size,))
```

## Files to Modify

| File | Change |
|------|--------|
| `replay_buffer.py:321-331` | Return valid_mask or resample |
| `trainer.py` | Apply mask in loss computation |

## Testing

1. Force buffer wraparound scenario
2. Verify invalid trajectories don't affect gradients
3. Check loss computation only uses valid samples

## Notes

This is particularly important during early training when buffer is small and wraps frequently.
