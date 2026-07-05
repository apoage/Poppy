# BUG #18: VectorBT Env Uses Global np.random.seed

**Severity:** LOW
**File:** `mcts_trader/env/vectorbt_env.py:342-364`
**Status:** FIXED

## Code
```python
if seed is not None:
    np.random.seed(seed)
self.episode_start_idx = np.random.randint(self.lookback, max_start)
```

## Issue
Using global numpy random state (`np.random.seed`) is not reproducible when running multiple environments in parallel. Each env shares the same global state.

## Impact
Non-reproducible episode starts in parallel training.

## Expected Fix
Use per-environment `np.random.Generator` instance:
```python
self.rng = np.random.default_rng(seed)
self.episode_start_idx = self.rng.integers(self.lookback, max_start)
```

## Investigation Checklist
- [x] Check all uses of np.random in the env - Found np.random.seed and np.random.randint
- [x] Convert to per-instance Generator - Added self.rng = np.random.default_rng()
- [x] Ensure seed is propagated correctly - Seed creates new Generator instance

## Fix Applied
1. Added `self.rng = np.random.default_rng(None)` in `__init__` (line 342-343)
2. Changed reset() to create new Generator when seed provided (line 362-363)
3. Changed `np.random.randint()` to `self.rng.integers()` (line 364)

```python
# Before:
if seed is not None:
    np.random.seed(seed)  # GLOBAL state
self.episode_start_idx = np.random.randint(self.lookback, max_start)

# After:
if seed is not None:
    self.rng = np.random.default_rng(seed)  # PER-INSTANCE
self.episode_start_idx = self.rng.integers(self.lookback, max_start)
```
