# BUG #15: Target Network Exists But Never Used

**Severity:** LOW
**File:** `mcts_trader/jax_agent/trainer.py:683-694`
**Status:** TO INVESTIGATE

## Code
```python
# Update target network periodically
new_step_count = state.step_count + 1
should_update_target = (new_step_count % self.target_update_interval) == 0

new_target_params = jax.lax.cond(
    should_update_target,
    lambda: new_state.params,  # True branch: copy current params
    lambda: state.target_params # False branch: keep old target params
)
```

## Issue
Target params exist but are NEVER USED anywhere in loss computation! The loss function always uses `params` (current params), never `state.target_params`. The target network update is dead code.

## Impact
No target network benefit for value learning stability. Higher variance in value targets.

## Expected Fix
Use `target_params` for value targets in TD-style learning, or remove the unused code.

## Investigation Checklist
- [ ] Check if target network was intended for future use
- [ ] Determine if TD-style learning would benefit from target net
- [ ] Either implement usage or remove dead code
