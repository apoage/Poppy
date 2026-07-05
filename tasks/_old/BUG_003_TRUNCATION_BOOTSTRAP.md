# BUG #3: Truncated Episodes Not Bootstrapped

**Severity:** HIGH
**File:** `train_jax_fast.py:165`
**Status:** FIXED (2026-01-04)

## Code
```python
next_obs_dict, reward, terminated, truncated, info = env.step(action_int)
done = terminated or truncated
...
if done:
    break
# ...
# Compute returns from end to start
running_return = 0.0
for t in range(n_steps - 1, -1, -1):
    running_return = rewards_arr[t] + discount * running_return
    mc_returns[t] = running_return
```

## Issue
Incorrect handling of truncated episodes (time limit reached). The code treats truncation as a terminal state with value 0 (V(S_T)=0). In reality, the agent continues to exist, so the return should be bootstrapped from the value of the final state (V(S_T)).

## Impact
The agent learns to undervalue states near the end of the episode, effectively viewing survival as equal to bankruptcy. This distorts the value function.

## Expected Fix
If `truncated` is True, run the model on `next_obs_dict` to estimate V(S_T) and initialize `running_return` with this value instead of 0.0.

## Investigation Checklist
- [ ] Verify terminated vs truncated semantics in environment
- [ ] Track whether episode ended by termination or truncation
- [ ] Get value estimate for final state if truncated
- [ ] Initialize running_return with bootstrap value
- [ ] Apply fix to both run_self_play and run_self_play_batched
