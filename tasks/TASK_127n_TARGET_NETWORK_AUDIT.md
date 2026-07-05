# TASK_127n: Target Network Synchronization Audit

**Status**: AUDITED (Critical Issue Found)
**Priority**: HIGH
**Created**: 2026-01-31

## Scope

Audited target network implementation:
- `mcts_trader/jax_agent/trainer.py` - Target params definition and update
- `mcts_trader/training/self_play.py` - Value bootstrapping
- `mcts_trader/jax_agent/mctx_search.py` - MCTS rollouts
- `train_jax_fast.py` - Parameter passing to self-play

## Architecture Overview

```
Target Network Flow (EXPECTED):
┌─────────────────────────────────────────────────────────────────┐
│ Online Network (params)                                         │
│  - Updated every gradient step                                  │
│  - Used for: policy output, gradient computation                │
└─────────────────────────────────────────────────────────────────┘
                              ↓ (periodic copy)
┌─────────────────────────────────────────────────────────────────┐
│ Target Network (target_params)                                  │
│  - Updated every N steps (hard copy)                            │
│  - SHOULD be used for: value bootstrapping, TD targets          │
└─────────────────────────────────────────────────────────────────┘

ACTUAL IMPLEMENTATION:
┌─────────────────────────────────────────────────────────────────┐
│ target_params exists and is updated...                          │
│ BUT IS NEVER USED ANYWHERE!                                     │
│ All inference uses online params exclusively.                   │
└─────────────────────────────────────────────────────────────────┘
```

## Audit Results

### 1. Target Network Update Mechanism ✅ IMPLEMENTED

**Location**: `trainer.py:1508-1518`

```python
# RESTORED: Update target network periodically (Double Q-learning stability)
should_update_target = (new_step_count % self.target_update_interval) == 0
new_target_params = jax.lax.cond(
    should_update_target,
    lambda: new_state.params,      # True: copy current params to target
    lambda: state.target_params    # False: keep old target params
)

new_state = new_state.replace(
    target_params=new_target_params,
    step_count=new_step_count
)
```

**Configuration**:
- Update interval: 100 steps (default)
- Update type: HARD copy (not soft/Polyak)
- Location: Inside JIT-compiled train_step

**Assessment**:
- ✅ Target params are updated correctly
- ✅ Hard copy every 100 steps
- ⚠️ No soft update option (Polyak averaging)
- ❌ **TARGET PARAMS ARE NEVER USED**

### 2. Target Network Usage ❌ CRITICAL BUG

**Known Issue**: Previously documented as BUG_015

```
tasks/_old/BUG_015_TARGET_NETWORK_UNUSED.md:21:
"Target params exist but are NEVER USED anywhere in loss computation!"
```

**Verification**:

| Component | Uses target_params? | Actually Uses |
|-----------|---------------------|---------------|
| Value bootstrapping | ❌ NO | `trainer.state.params` |
| TD target computation | ❌ NO | `params` (current) |
| MCTS rollouts | ❌ NO | `params` (current) |
| Expert BC Q-filter | ❌ NO | `params` (current) |
| Loss computation | ❌ NO | `params` (current) |

### 3. Value Bootstrapping ❌ USES ONLINE PARAMS

**Location**: `self_play.py:260-264`

```python
# Run inference to get V(s_T)
output = model_apply_fn({'params': params}, final_obs_jax, mode='initial_inference')
support = get_value_support()
value_probs = jax.nn.softmax(output['value'][0])
bootstrap_value = float(jnp.sum(value_probs * support))
```

**Caller** (`train_jax_fast.py:918`):
```python
trajectories, scores = run_self_play_jax(
    ...,
    trainer.state.params,  # ONLINE params, not target_params!
    ...
)
```

**Assessment**:
- ❌ Bootstrap uses online params
- ❌ No option to pass target_params
- ⚠️ Moving target problem: bootstrap value changes with every gradient step

### 4. MCTS Rollouts ❌ USES ONLINE PARAMS

**Location**: `mctx_search.py:164-168, 199`

```python
root_output = self.model_apply_fn(
    {'params': params},  # Online params passed directly
    observation,
    method='initial_inference'
)
...
policy_output = mctx.gumbel_muzero_policy(
    params=params,  # Same online params for tree search
    ...
)
```

**Assessment**:
- ❌ MCTS root value uses online params
- ❌ MCTS dynamics rollouts use online params
- ⚠️ Value estimates change during training

### 5. Expert BC Loss ❌ USES ONLINE PARAMS

**Location**: `trainer.py:1387-1391`

```python
# Run initial inference on expert observations
expert_output = self.model.apply(
    {'params': params},  # Online params, not target_params!
    expert_obs,
    mode='initial_inference'
)
```

**Assessment**:
- ❌ Q-filter comparison uses online params
- ❌ No target network for stable Q estimates

## Critical Issues

### Issue 1: Target Network Is Dead Code ❌ CRITICAL

**Severity**: HIGH

**Problem**: Target network machinery exists but provides zero benefit:
```python
# trainer.py:251 - Defined
target_params: flax.core.FrozenDict = None

# trainer.py:591 - Initialized
target_params=params,

# trainer.py:1509-1517 - Updated
should_update_target = (new_step_count % self.target_update_interval) == 0
new_target_params = jax.lax.cond(...)

# NEVER USED IN ANY INFERENCE OR LOSS COMPUTATION
```

**Impact**:
- Wasted memory storing duplicate params
- Wasted compute updating params every 100 steps
- No stability benefit from Double Q-learning
- Moving target problem in value learning

### Issue 2: Bootstrap Value Uses Online Network ⚠️ HIGH

**Severity**: HIGH

**Problem**: When episode truncates, bootstrap value V(s_T) is computed with online params:
```python
output = model_apply_fn({'params': params}, final_obs_jax, ...)
bootstrap_value = expected_value(output['value'])
```

**Impact**:
- Bootstrap targets shift with every gradient update
- Higher variance in return estimates
- Potential instability in value learning

### Issue 3: No Soft Update Option ⚠️ MEDIUM

**Severity**: MEDIUM

**Problem**: Only hard copy implemented, no Polyak averaging:
```python
# Current: hard copy every 100 steps
new_target_params = jax.lax.cond(
    should_update_target,
    lambda: new_state.params,  # Full copy
    lambda: state.target_params
)

# Missing: soft update (Polyak)
# target = tau * online + (1 - tau) * target
```

**Impact**:
- Abrupt target network changes every 100 steps
- No smooth interpolation option

## Soft Update vs Hard Update

| Method | Formula | Pros | Cons |
|--------|---------|------|------|
| Hard (current) | `target = online` every N steps | Simple, no hyperparameter | Abrupt changes |
| Soft (Polyak) | `target = τ*online + (1-τ)*target` | Smooth, stable | Extra hyperparameter τ |

Standard τ values: 0.001 - 0.01 (update every step)

## Summary

| Aspect | Status | Notes |
|--------|--------|-------|
| Target params defined | ✅ Yes | In TrainState |
| Target params initialized | ✅ Yes | Same as online |
| Target params updated | ✅ Yes | Hard copy every 100 steps |
| **Target params USED** | ❌ NO | **NEVER USED ANYWHERE** |
| Bootstrap uses target | ❌ No | Uses online params |
| MCTS uses target | ❌ No | Uses online params |
| Loss uses target | ❌ No | Uses online params |
| Soft update option | ❌ No | Only hard copy |

## Recommendations

### Option A: Implement Target Network Usage (Recommended)

1. **Pass target_params to self-play for bootstrapping**:
```python
# train_jax_fast.py
trajectories, scores = run_self_play_jax(
    ...,
    trainer.state.target_params,  # Use target for stable bootstrap
    ...
)
```

2. **Use target for value targets in training**:
```python
# trainer.py - compute TD targets with target network
with jax.lax.stop_gradient():
    target_output = self.model.apply({'params': target_params}, next_obs)
    target_value = self._logits_to_scalar(target_output['value'])
```

3. **Add soft update option**:
```python
# Polyak averaging
tau = 0.005
new_target_params = jax.tree_util.tree_map(
    lambda t, o: tau * o + (1 - tau) * t,
    state.target_params,
    new_state.params
)
```

### Option B: Remove Dead Code

If target network is intentionally unused (MuZero doesn't strictly require it since value targets come from MC/GAE returns, not TD):

1. Remove `target_params` from TrainState
2. Remove update logic
3. Document why target network is not needed

### Option C: Keep for Future Use

Document that target_params exists for future Double Q-learning implementation but is currently unused.

## MuZero-Specific Considerations

In standard MuZero:
- Value targets come from MCTS search (root value) or actual returns
- NOT from TD-style bootstrapping with target network
- Target network is less critical than in DQN/SAC

However, this codebase uses:
- MC/GAE returns as value targets (correct)
- Bootstrap value at truncation (would benefit from target network)
- MCTS root values for action selection (less critical)

**Recommendation**: At minimum, use target network for bootstrap value computation to reduce variance in truncated episodes.

## Related Tasks

- BUG_015: Target Network Unused (original documentation)
- TASK_127e: PopArt Normalization Audit
- TASK_127b: MuZero Dynamics Audit
