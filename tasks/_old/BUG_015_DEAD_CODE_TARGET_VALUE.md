# Bug: Dead Code - Unused Target Value Variables

## Priority: LOW
**Status:** DEFERRED

Low priority cleanup task. No functional impact.
**Component:** Multiple files

## Problem Description

Various files contain unused variables, dead code paths, or orphaned functions. These add confusion and may indicate incomplete implementations.

### Potential Locations

1. **trainer.py**: Unused target network updates
2. **replay_buffer.py**: Commented-out priority sampling
3. **train_jax_fast.py**: Unused imports or variables

## Solution

### Step 1: Static analysis

```bash
# Find unused imports
python -m pyflakes train_jax_fast.py
python -m pyflakes mcts_trader/jax_agent/trainer.py

# Find unused variables
pylint --disable=all --enable=W0612 train_jax_fast.py
```

### Step 2: Remove dead code

Only remove code that is:
1. Definitely unused (not just conditionally unused)
2. Not part of a planned feature
3. Not test/debug code that should stay

## Files to Audit

| File | What to Check |
|------|---------------|
| `trainer.py` | Target network code |
| `replay_buffer.py` | Priority sampling |
| `train_jax_fast.py` | Unused imports |
| `muzero_model.py` | Unused methods |

## Notes

Low priority - dead code doesn't affect correctness, just maintainability.
