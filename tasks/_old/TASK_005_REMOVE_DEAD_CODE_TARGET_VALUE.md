# TASK 005: Remove Dead Code `_compute_target_value` Method

**Severity**: LOW
**Location**: `mcts_trader/jax_agent/trainer.py:140-175`
**Status**: DONE (2025-12-29)
**Estimated Complexity**: Very Low

---

## Problem Description

The `_compute_target_value` method in the trainer is **NEVER CALLED**. It computes Monte Carlo returns, but the actual MC returns are computed in `run_self_play()` in `train_jax_fast.py` and passed directly as `values` in the trajectory.

### Dead Code (`trainer.py:140-175`):
```python
def _compute_target_value(
    self,
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    dones: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute Monte Carlo returns (full episode, no bootstrapping).

    CRITICAL FIX: Using bootstrapped values creates circular dependency
    when value head is stuck. Monte Carlo uses only actual rewards.
    ...
    """
    # Implementation that is never used
```

### Where MC Returns Are Actually Computed (`train_jax_fast.py:130-141`):
```python
# CRITICAL FIX: Compute Monte Carlo returns
rewards_arr = np.array(rewards)
n_steps = len(rewards_arr)
mc_returns = np.zeros(n_steps)
discount = 0.997

running_return = 0.0
for t in range(n_steps - 1, -1, -1):
    running_return = rewards_arr[t] + discount * running_return
    mc_returns[t] = running_return

return {
    ...
    'values': mc_returns,  # Used directly as targets
    ...
}
```

### Where Values Are Used in Training (`trainer.py:258`):
```python
target_values = batch['values']  # Pre-computed MC returns from self-play
```

The `_compute_target_value` method is NEVER called - `target_values` comes directly from the batch.

---

## Impact of Dead Code

1. **Code confusion**: Future developers may think it's used
2. **Maintenance burden**: Code must be kept in sync with actual implementation
3. **Documentation mismatch**: Comments suggest it's critical but it's unused

---

## Proposed Solution

### Option A: Remove Dead Code (Recommended)

Simply delete the unused method:

```python
# DELETE lines 140-175 in trainer.py
```

### Option B: Keep as Utility (Alternative)

If there's value in having the JAX-based implementation for future use:
```python
def _compute_target_value(self, ...):
    """
    Compute Monte Carlo returns.

    NOTE: This method is NOT used in the current training pipeline.
    MC returns are computed in run_self_play() instead. This method
    is kept for potential future use with on-policy training.
    """
```

---

## Implementation Plan

### Step 1: Verify Method is Unused (5 min)

```bash
# Search for any calls to _compute_target_value
grep -rn "_compute_target_value" mcts_trader/
grep -rn "_compute_target_value" *.py
```

Expected output: Only the definition, no calls.

### Step 2: Remove Dead Code (2 min)

Delete lines 140-175 in `trainer.py`.

### Step 3: Verify Training Still Works (10 min)

```bash
python train_jax_fast.py --synthetic-level 0 --epochs 10 --name test_no_dead_code
```

### Step 4: Remove Debug Reference (2 min)

There's a debug call that references this method (`trainer.py:483-488`):
```python
# Compute MC targets to check variance
mc_targets = self._compute_target_value(
    batch['rewards'],
    batch['values'],
    batch['dones']
)
```

This debug logging also needs to be updated or removed since it calls the dead code.

---

## Acceptance Criteria

- [ ] `_compute_target_value` method removed OR clearly marked as unused
- [ ] All references to the method removed
- [ ] Training runs successfully after removal
- [ ] No grep results for `_compute_target_value` calls

---

## Files to Modify

1. `mcts_trader/jax_agent/trainer.py` - Remove method (lines 140-175)
2. `mcts_trader/jax_agent/trainer.py` - Remove debug reference (lines 483-488)

---

## Before/After

### Before (`trainer.py`):
```python
def _compute_target_value(self, rewards, values, dones):
    """..."""
    # 35 lines of unused code

# ... later in train() method ...
mc_targets = self._compute_target_value(...)  # Debug call
```

### After:
```python
# Method removed entirely

# Debug logging simplified:
print(f"  [Train Iter {iteration + 1}] Batch rewards: min={...} max={...}")
```

---

## Risk Assessment

**Risk**: None. The method is definitively unused.

**Verification**:
```bash
# Double-check there are no dynamic calls
python -c "
import ast
import inspect
from mcts_trader.jax_agent.trainer import MuZeroTrainer

# Get source and parse AST
source = inspect.getsource(MuZeroTrainer)
tree = ast.parse(source)

# Find all method calls
calls = [node.func.attr for node in ast.walk(tree)
         if isinstance(node, ast.Call)
         and isinstance(node.func, ast.Attribute)]

print('Method calls found:', set(calls))
print('_compute_target_value called:', '_compute_target_value' in calls)
"
```
