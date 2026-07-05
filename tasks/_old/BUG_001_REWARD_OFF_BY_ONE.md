# BUG #1: Reward Target Off-by-One Error

**Severity:** CRITICAL
**File:** `mcts_trader/jax_agent/trainer.py:539`
**Status:** FIXED (2026-01-04)

## Code
```python
reward_loss += optax.softmax_cross_entropy(
    recurrent_output['reward'],
    self._scalar_to_categorical(rewards[:, t])
).mean()
```

## Issue
Off-by-one error in reward targets. The loop iterates t from 1 to unroll_steps. The dynamics function takes `actions[:, t-1]` (action a_{t-1}). The corresponding immediate reward r_{t-1} (received after a_{t-1}) is stored at `rewards[:, t-1]` (assuming `rewards` array starts with r_0). The code uses `rewards[:, t]`, which is r_t (the reward for the next action).

## Impact
The model learns to predict the future reward r_{t+1} given action a_t, rather than the immediate reward r_t. This breaks causality and makes reward prediction impossible/random, severely degrading performance.

## Expected Fix
Change `rewards[:, t]` to `rewards[:, t-1]`.

## Investigation Checklist
- [ ] Verify the loop structure and action indexing
- [ ] Trace the data flow: action -> dynamics -> reward
- [ ] Confirm rewards array semantics (what index holds what reward)
- [ ] Check if fix breaks anything else
