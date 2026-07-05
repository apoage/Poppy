# TASK_127k: MCTS Search Integration Audit

**Status**: AUDITED (Working, Some Observations)
**Priority**: Medium
**Created**: 2026-01-31

## Scope

Audited MCTS integration:
- `mcts_trader/jax_agent/mctx_search.py` - MCTS wrapper using Google's mctx
- `mcts_trader/training/self_play.py` - Episode collection with MCTS
- `mcts_trader/jax_agent/trainer.py` - MCTS targets in training

## Architecture Overview

```
MCTS Search Flow:
┌────────────────────────────────────────────────────────────────────┐
│ MuZeroMCTXSearch (mctx_search.py)                                  │
│  Uses Google's mctx library (PUCT/Gumbel internally implemented)   │
└────────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────────┐
│ Self-Play (self_play.py)                                           │
│  - Run MCTS at each step                                           │
│  - Collect: observations, actions, MCTS policies, rewards          │
│  - Compute MC/GAE returns as VALUE targets (not MCTS values!)      │
└────────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────────┐
│ Training (trainer.py)                                              │
│  - Policy loss: CrossEntropy(model_policy, MCTS_policy)            │
│  - Value loss: MSE(model_value, MC/GAE_returns)                    │
│  - Reward loss: CrossEntropy(model_reward, actual_reward)          │
└────────────────────────────────────────────────────────────────────┘
```

## Audit Results

### 1. UCB/PUCT Formula ✅ DELEGATED TO MCTX

**Location**: `mctx_search.py:195-221`

```python
if self.use_gumbel:
    # Gumbel MuZero (recommended)
    policy_output = mctx.gumbel_muzero_policy(
        ...
        qtransform=mctx.qtransform_by_parent_and_siblings,
        ...
    )
else:
    # Standard MuZero with PUCT
    policy_output = mctx.muzero_policy(...)
```

**Assessment**:
- ✅ Uses Google's mctx library (well-tested, production quality)
- ✅ Gumbel MuZero is default (recommended over standard PUCT)
- ✅ `qtransform_by_parent_and_siblings` for improved Q-value normalization
- ℹ️ UCB/PUCT formula is internal to mctx, not custom code

**mctx PUCT formula** (from mctx source):
```
UCB(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
```

### 2. Visit Count Normalization ✅ HANDLED BY MCTX

**Location**: `mctx_search.py:224-225`

```python
actions = policy_output.action
policy_probs = policy_output.action_weights  # Normalized visit counts
```

**Assessment**:
- ✅ `action_weights` from mctx are already normalized visit counts
- ✅ No additional normalization needed

**Verification** from diagnostics code (`mctx_search.py:305-312`):
```python
visit_counts = policy_output.action_weights
visit_sum = jnp.sum(visit_counts, axis=-1, keepdims=True)
probs = visit_counts / (visit_sum + 1e-8)  # Already ~1.0
```

### 3. Value Backpropagation ✅ HANDLED BY MCTX

**Location**: `mctx_search.py:89-119`

```python
def _recurrent_fn(self, params, rng, action, embedding):
    output = self.model_apply_fn(...)
    reward_scalar = self._categorical_to_scalar(output['reward']) / 100.0
    value_scalar = self._categorical_to_scalar(output['value'])

    return mctx.RecurrentFnOutput(
        reward=reward_scalar,
        discount=jnp.full_like(reward_scalar, self.discount),
        prior_logits=output['policy_logits'],
        value=value_scalar
    ), output['latent_state']
```

**Assessment**:
- ✅ Value backpropagation is internal to mctx
- ✅ Model provides: reward, value, policy logits, next state
- ✅ mctx handles tree structure and backprop

### 4. Improved Policy from Visit Counts ✅ CORRECT

**Self-play collection** (`self_play.py:114, 219`):
```python
action_jax, policy_jax, stats = search_fn(params, step_rng, obs_jax, mask_jax)
...
policy_probs.append(np.array(policy_jax[0]))  # MCTS visit distribution
```

**Training target** (`trainer.py:1046, 1099-1103`):
```python
target_policies = batch['policy_probs']  # MCTS visit distributions

# Policy loss: match MCTS improved policy
per_sample_policy = optax.softmax_cross_entropy(
    initial_output['policy_logits'],
    target_policies[:, 0]
)
```

**Assessment**:
- ✅ MCTS visit counts used as policy targets
- ✅ Network trained to match MCTS-improved policy
- ✅ This is standard MuZero training procedure

### 5. Temperature Application ⚠️ COMPLEX FLOW

**Temperature in MCTS** (`mctx_search.py:217`):
```python
# Standard MuZero only (not Gumbel)
policy_output = mctx.muzero_policy(
    ...
    temperature=self.temperature,  # Default 1.0
    ...
)
```

**Temperature for Gumbel** (`mctx_search.py:206`):
```python
gumbel_scale=self.gumbel_scale,  # Default 1.0
```

**Assessment**:
- ✅ Gumbel MuZero uses `gumbel_scale` instead of temperature
- ⚠️ Temperature only affects action SELECTION, not visit count distribution
- ⚠️ `action_weights` are raw normalized visit counts (no temperature softmax)

### 6. Value Targets: MC/GAE Returns, NOT MCTS Values ✅ CORRECT

**Location**: `self_play.py:268-277, 296`

```python
# Compute returns (MC or GAE) - NOT from MCTS
returns = compute_returns(
    rewards=rewards_arr,
    values=values_arr,  # Model predictions for GAE baseline
    dones=dones_arr,
    bootstrap_value=bootstrap_value,
    discount=discount,
    use_gae=use_gae,
    gae_lambda=gae_lambda,
)
...
result = {
    ...
    'values': returns,  # MC or GAE returns, NOT MCTS root values
    ...
}
```

**Assessment**:
- ✅ Value targets are MC/GAE returns from actual rewards
- ✅ NOT using MCTS root values as targets (which would be circular)
- ✅ MCTS root values stored separately for diagnostics (`mcts_root_values`)

### 7. Does MCTS Actually Improve Policy? ⚠️ DEPENDS ON SIMULATIONS

**Evidence for improvement**:
1. MCTS performs lookahead search using learned dynamics model
2. Visit counts concentrate on higher-value actions
3. With sufficient simulations, MCTS should outperform raw policy

**Potential issues**:
1. **Low simulation count** (default 32): May not explore enough
2. **Inaccurate dynamics model**: Garbage in, garbage out
3. **Value estimation errors**: Compound over search depth

**Diagnostic metrics available**:
```python
# mctx_search.py:296-323
diagnostics = {
    'root_value_mean': ...,
    'root_value_std': ...,
    'visit_entropy_mean': ...,  # Higher = more exploration
    'calibration_mse': ...,     # Value vs actual reward error
}
```

## Issues Found

### Issue 1: ε-Greedy Applied to Training Target, Not Action ⚠️

**Location**: `mctx_search.py:227-235`

```python
# TASK_037: Apply ε-greedy mixing to training targets
# Action selection uses pure MCTS, but training target gets exploration mixed in
if self.epsilon_greedy > 0:
    policy_probs = apply_epsilon_greedy(
        policy_probs,
        epsilon=self.epsilon_greedy,
        action_mask=action_mask,
    )
```

**Observation**:
- ε-greedy is applied to the POLICY TARGET, not action selection
- This injects noise into what the network learns to imitate
- Intended to prevent MCTS→policy→MCTS sharpening feedback loop

**Assessment**: This is intentional design (TASK_037), but could hurt final performance if ε stays high.

### Issue 2: Reward Scaling in MCTS ⚠️

**Location**: `mctx_search.py:108-110`

```python
# TASK_034: Rewards are trained with 100× scaling for better bin resolution
# Descale here to match actual environment reward scale
reward_scalar = self._categorical_to_scalar(output['reward']) / 100.0
```

**Observation**:
- Reward model trained with 100x scaling
- Must descale in MCTS for correct value backpropagation
- If scaling is wrong, MCTS values will be incorrect

**Assessment**: Scaling looks correct, but verify reward magnitudes in practice.

### Issue 3: No Explicit MCTS Policy Improvement Verification ⚠️

**Problem**: No metrics comparing raw policy vs MCTS policy quality.

**Missing metrics**:
- KL divergence between raw policy and MCTS policy
- Win rate: MCTS-selected action vs raw policy argmax
- Value improvement: MCTS root value vs raw value prediction

**Recommendation**: Add comparison metrics to verify MCTS is actually helping.

### Issue 4: Confidence Thresholding Modifies Both Action AND Policy ⚠️

**Location**: `self_play.py:169-182`

```python
if max_prob < confidence_threshold:
    # Force HOLD action
    action_int = hold_action
    # Create synthetic confident HOLD policy
    synthetic_policy = np.ones_like(probs) * (0.1 / (len(probs) - 1))
    synthetic_policy[hold_action] = 0.9
    policy_jax = jnp.array([synthetic_policy])  # Override MCTS policy!
```

**Observation**:
- When MCTS is uncertain, action is forced to HOLD
- Policy target is also replaced with synthetic "confident HOLD"
- This teaches the network to be confident about HOLD when MCTS is uncertain

**Assessment**: Could cause policy to be overly conservative. Consider if this is desired behavior.

## Summary

| Aspect | Status | Notes |
|--------|--------|-------|
| UCB/PUCT formula | ✅ Correct | Delegated to mctx library |
| Visit count normalization | ✅ Correct | Handled by mctx |
| Value backpropagation | ✅ Correct | Handled by mctx |
| Policy targets from MCTS | ✅ Correct | Visit distributions |
| Value targets | ✅ Correct | MC/GAE returns (not MCTS) |
| Temperature handling | ⚠️ Complex | Different for Gumbel vs PUCT |
| ε-greedy on targets | ⚠️ Intentional | May hurt final performance |
| Reward scaling | ⚠️ Check | Must match training scale |
| Improvement verification | ❌ Missing | Add comparison metrics |

## Recommendations

1. **Add MCTS improvement metrics**:
   ```python
   metrics['mcts_kl_from_raw'] = KL(mcts_policy || raw_policy)
   metrics['mcts_action_diff'] = (mcts_action != raw_argmax).mean()
   ```

2. **Verify reward scaling** matches between training and MCTS inference

3. **Consider decaying ε-greedy** in training targets over time

4. **Monitor visit entropy** - too low suggests MCTS not exploring enough

5. **Compare MCTS root values to actual returns** - large errors indicate dynamics model issues

## mctx Library Reference

The mctx library handles:
- PUCT selection: `UCB(s,a) = Q(s,a) + c * P(s,a) * sqrt(N(s)) / (1 + N(s,a))`
- Gumbel MuZero: Sequential halving with Gumbel noise for action selection
- Value backpropagation: Discounted returns through tree
- Q-value normalization: `qtransform_by_parent_and_siblings` scales by parent visit count

## Related Tasks

- TASK_037: ε-greedy MCTS mixing
- TASK_034: Reward scaling
- TASK_127b: MuZero Dynamics Audit
- TASK_127g: Entropy and Policy Collapse Audit
