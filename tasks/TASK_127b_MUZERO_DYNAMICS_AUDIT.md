# TASK_127b: MuZero Dynamics Model Audit

**Status**: AUDITED (Minor Issues Found)
**Priority**: Medium
**Created**: 2026-01-31

## Scope

Audited MuZero components:
- `mcts_trader/jax_agent/muzero_model.py` - Dynamics, representation, prediction networks
- `mcts_trader/jax_agent/trainer.py` - Loss computation, K-step unrolling
- `mcts_trader/jax_agent/trm_flax.py` - TRM backbone (encoder, reasoning)

## Architecture Overview

```
MuZero Architecture:
┌─────────────────────────────────────────────────────────────────┐
│ Representation: observation → latent_state                       │
│   TRMEncoder (CNN) → Dense projection → state_dim                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Prediction: latent_state → (policy_logits, value)                │
│   TRMReasoning (H_cycles × L_cycles recursive refinement)        │
│   → policy head (Dense → n_actions)                              │
│   → value head (Dense → support_size categorical)                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Dynamics: (latent_state, action) → (next_latent_state, reward)   │
│   action_embedding + concat → Dense → LN → GELU → Dense          │
│   Output split: [0:state_dim] = next_state, [state_dim:] = reward│
│   LayerNorm on next_state to prevent drift                       │
└─────────────────────────────────────────────────────────────────┘
```

## Audit Results

### 1. Dynamics Model Architecture ✅ CORRECT

**Location**: `muzero_model.py:138-178`

```python
def dynamics(self, latent_state, action):
    action_emb = self.action_embedding(action).astype(self.dtype)
    x = jnp.concatenate([latent_state, action_emb], axis=-1)
    x = self.dynamics_dense1(x)      # state_dim * 8 hidden
    x = self.dynamics_ln(x)
    x = nn.gelu(x)
    out = self.dynamics_dense2(x)    # state_dim + support_size output
    next_state = out[:, :self.state_dim]
    reward = out[:, self.state_dim:]
    next_state = self.dynamics_state_norm(next_state)  # Prevent drift
    return next_state, reward
```

**Assessment**:
- ✅ Action embedding correctly concatenated with state
- ✅ Two-layer MLP with LayerNorm and GELU (standard)
- ✅ Output correctly split into next_state and reward
- ✅ LayerNorm on next_state prevents representation drift during MCTS unrolling
- ✅ Mixed precision handled correctly (cast to dtype, back to float32 for outputs)

### 2. Reward Prediction Head ✅ CORRECT

**Location**: `muzero_model.py:66, 169`

```python
# Output dimension
self.dynamics_dense2 = nn.Dense(self.state_dim + self.support_size, ...)

# Output split
reward = out[:, self.state_dim:]  # (batch, support_size) categorical
```

**Assessment**:
- ✅ Reward is categorical distribution over `support_size` bins (MuZero paper compliant)
- ✅ `support_size = 401` bins over [-10, 10] range with PopArt normalization
- ✅ Cross-entropy loss used for reward prediction (trainer.py:1155-1159)

### 3. Representation Network ✅ NO GRADIENT BLOCKERS

**Location**: `trm_flax.py:9-48`

```python
class TRMEncoder(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=conv1_features, ...)(x)
        x = nn.gelu(x)
        x = nn.max_pool(x, ...)
        x = nn.Conv(features=conv2_features, ...)(x)
        x = nn.gelu(x)
        x = nn.avg_pool(x, ...)
        x = x.reshape((x.shape[0], -1))
        x = nn.LayerNorm(...)(x)
        state = nn.Dense(features=self.state_dim, ...)(x)
        return state
```

**Assessment**:
- ✅ No `stop_gradient` or `jax.lax.stop_gradient` calls anywhere in codebase
- ✅ Continuous gradient path from observation to latent_state
- ✅ All operations are differentiable

### 4. K-Step Unrolling ✅ CORRECT

**Location**: `trainer.py:1079-1172`

```python
# Initial inference
initial_output = self.model.apply({'params': params}, observations[:, 0], mode='initial_inference')
latent_state = initial_output['latent_state']

# Compute initial losses (step 0)
value_loss = self._hybrid_value_loss(initial_output['value'], target_values[:, 0])
policy_loss = cross_entropy(initial_output['policy_logits'], target_policies[:, 0])

# Unroll dynamics for steps 1 to unroll_steps
for t in range(1, unroll_steps):
    recurrent_output = self.model.apply({'params': params}, latent_state, actions[:, t-1], mode='recurrent_inference')
    latent_state = recurrent_output['latent_state']  # Chain for next step

    value_loss += self._hybrid_value_loss(recurrent_output['value'], target_values[:, t])
    policy_loss += cross_entropy(recurrent_output['policy_logits'], target_policies[:, t])
    reward_loss += cross_entropy(recurrent_output['reward'], scalar_to_categorical(rewards[:, t-1]))
```

**Assessment**:
- ✅ Latent state correctly chained through unroll steps
- ✅ Value/policy losses accumulated for all steps
- ✅ Reward loss correctly uses `t-1` index (BUG_001 fix documented)
- ✅ Losses averaged over unroll steps at the end
- ✅ Valid mask applied for buffer wraparound (BUG_002 fix)

### 5. Target Computation ✅ CORRECT

Targets come from replay buffer (self_play.py):
- ✅ Value targets are n-step returns (MC or GAE)
- ✅ Policy targets are MCTS search distributions
- ✅ Reward targets are actual environment rewards

### 6. Gradient Flow Analysis ⚠️ POTENTIAL CONCERN

**Question**: Do dynamics gradients flow back to representation?

**Finding**: Gradients DO flow to representation through:
1. **Direct path**: `value_loss_0` and `policy_loss_0` from `initial_output` use `representation()`
2. **Indirect path**: `latent_state` in recurrent_inference comes from `initial_output['latent_state']`

**However**:
⚠️ The dynamics `next_state` output is normalized by `dynamics_state_norm`. If latent representation drifts far from dynamics output distribution, LayerNorm could mask gradient signal.

## Issues Found

### Issue 1: No Representation-Dynamics Consistency Loss

**Severity**: Low

Standard MuZero relies on implicit learning. Explicit consistency could help:
```python
consistency_loss = MSE(dynamics(h_t, a_t), representation(o_{t+1}))
```

### Issue 2: No Gradient Magnitude Monitoring

**Severity**: Low

No metrics for per-component gradient norms to detect if any network isn't receiving gradients.

## Summary

| Component | Status | Notes |
|-----------|--------|-------|
| Dynamics architecture | ✅ Correct | Standard MuZero design |
| Reward head | ✅ Correct | Categorical over 401 bins |
| Representation network | ✅ No blockers | Gradients flow through |
| K-step unrolling | ✅ Correct | Properly chains states |
| Target computation | ✅ Correct | MC/GAE returns, MCTS policies |
| Gradient flow | ⚠️ Works | No consistency loss |

## Recommendations

1. **Optional consistency loss** for representation-dynamics alignment
2. **Add gradient monitoring** per component for diagnostics
3. **Verify reward scale** fits within value support range [-10, 10]

## Related Tasks

- TASK_126a: Reward Spike Investigation
- TASK_126e: Numerical Stability Audit
