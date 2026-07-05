# Bug: Replay Buffer Potential Integer Overflow

## Priority: MEDIUM
**Status:** DEFERRED

Extremely unlikely to occur. Would require billions of trajectories.
Python ints don't overflow, and JAX uses int64 internally.
**Component:** `mcts_trader/jax_agent/replay_buffer.py`

## Problem Description

The replay buffer generation tracking uses integers that could overflow during very long training runs.

### Location

```python
# replay_buffer.py
self.generation = 0  # Increments each trajectory

# After billions of trajectories:
self.generation += 1  # Could overflow int32
```

### Impact

- Generation validation breaks after overflow
- Buffer corruption detection fails
- Very long runs (days/weeks) could hit this

## Solution

```python
# Use int64 or modular arithmetic
self.generation = np.int64(0)

# Or wrap around explicitly
MAX_GENERATION = 2**31 - 1
self.generation = (self.generation + 1) % MAX_GENERATION
```

## Files to Modify

| File | Change |
|------|--------|
| `replay_buffer.py` | Use int64 or modular generation |

## Testing

1. Simulate high generation numbers
2. Verify validation still works
3. No overflow errors

## Notes

Low urgency - only affects extremely long runs.
