# TASK_127k1: MuZero Dynamics Model Audit

## Executive Summary

This audit examines the MuZero dynamics, representation, and prediction components for architectural correctness and gradient flow. **Several design choices deviate from standard MuZero practice**, which may affect learning stability and world model quality.

**Risk Level: MEDIUM-HIGH** - No immediate crashes, but training efficiency may be compromised.

---

## Architecture Overview

```
MuZeroNetworks:
├── representation() → TRMEncoder (CNN → state_dim)
├── dynamics() → action_embedding + MLP (state+action → next_state, reward)
├── prediction() → TRMReasoning (H_cycles × L_cycles recursive refinement)
└── prediction head (optional, Phase 2)
```

---

## Critical Findings

### 🔴 CRITICAL: Missing Latent Consistency Loss

**Standard MuZero:** The dynamics model is trained with a **consistency loss** between:
- `dynamics(s_t, a_t)` → predicted `s_{t+1}`
- `representation(o_{t+1})` → encoded `s_{t+1}` from actual next observation

**Current Implementation:** No consistency loss exists.

**File:** `mcts_trader/jax_agent/trainer.py` (lines 1119-1164)

The unroll loop only trains:
- Value targets (from MCTS)
- Policy targets (from MCTS)
- Reward targets (from environment)

**Missing:**
```python
# Standard MuZero consistency loss (NOT IMPLEMENTED)
next_obs_encoded = self.model.representation(observations[:, t])
next_obs_from_dynamics = recurrent_output['latent_state']
consistency_loss = MSE(next_obs_encoded, next_obs_from_dynamics)
```

**Impact:**
- Dynamics model can drift arbitrarily from encoder
- Predicted latent states may not be interpretable by prediction head
- Compounding errors in MCTS unrolling
- **Training instability** - value/policy heads see different latent distributions during search vs training

**Evidence from code:**
```python
# Line 1121-1126: Only value/policy/reward losses
recurrent_output = self.model.apply(
    {'params': params},
    latent_state,  # From previous dynamics step
    actions[:, t - 1],
    mode='recurrent_inference'
)
# No comparison to encoding of next observation!
```

**Fix:**
```python
# Add after line 1129 (unroll loop)
if 'observations' in batch and t < unroll_steps:
    # Encode actual next observation
    next_obs_encoded = self.model.apply(
        {'params': params},
        observations[:, t],
        mode='representation'
    )
    # Consistency loss
    consistency_loss = jnp.mean(
        (latent_state - next_obs_encoded) ** 2
    )
    total_loss += consistency_weight * consistency_loss
```

---

### 🟡 HIGH: No Gradient Blocking Between Dynamics and Representation

**Standard MuZero:** Gradients from dynamics are typically **stopped** from flowing to the representation encoder during the unroll loop. This prevents the encoder from being corrupted by potentially noisy dynamics gradients.

**Current Implementation:** 
- No `jax.lax.stop_gradient()` calls found in the codebase
- Gradients from dynamics unrolling flow all the way back to:
  1. The initial observation encoding
  2. All dynamics parameters
  3. The prediction head

**File:** `mcts_trader/jax_agent/muzero_model.py` (lines 235-243)

```python
def recurrent_inference(self, latent_state, action):
    next_latent_state, reward = self.dynamics(latent_state, action)
    policy_logits, value = self.prediction(next_latent_state)
    # No stop_gradient on next_latent_state!
```

**Impact:**
- Encoder may overfit to dynamics rather than observations
- Unstable gradients during long unrolls
- Potential degenerate solutions where encoder+dynamics form an "autoencoder"

**Fix:**
```python
def recurrent_inference(self, latent_state, action):
    next_latent_state, reward = self.dynamics(latent_state, action)
    # Stop gradient for consistency loss target
    next_latent_state_sg = jax.lax.stop_gradient(next_latent_state)
    policy_logits, value = self.prediction(next_latent_state_sg)
    # ...
```

---

### 🟡 MEDIUM: Reward Head Predicts Categorical Distribution Without Value Support Alignment

**Issue:** The reward head outputs `support_size` logits (categorical), but the reward scaling and value support configuration may not be properly aligned.

**File:** `mcts_trader/jax_agent/muzero_model.py` (lines 166-169)

```python
# Dynamics output
out = self.dynamics_dense2(x)
next_state = out[:, :self.state_dim]
reward = out[:, self.state_dim:]  # shape: (batch, support_size=401)
```

**File:** `mcts_trader/jax_agent/trainer.py` (lines 1154-1158)

```python
# Reward targets
scaled_rewards = rewards[:, t - 1] * self.reward_scale  # Scale by 100
per_sample_reward = optax.softmax_cross_entropy(
    recurrent_output['reward'],
    self._scalar_to_categorical(scaled_rewards)
)
```

**Potential Issue:** 
- Value support is `[-10, 10]` with 401 bins (for PopArt-normalized values)
- Rewards are symlog-scaled: `symlog(log_return * 100)`
- Typical rewards: `[-5, +5]` after symlog
- Scaled by 100: `[-500, +500]` - **FAR outside value support!**

**Evidence:**
```python
# From __init__.py
VALUE_SUPPORT_MIN = -10
VALUE_SUPPORT_MAX = 10
```

With `REWARD_SCALE = 100.0` and symlog rewards of ±5:
```
scaled_reward = 5 * 100 = 500  # WAY outside [-10, 10]!
```

This means **rewards are being clipped to the support range**, losing information.

**Fix:**
```python
# Either reduce reward_scale or expand value support
REWARD_SCALE = 10.0  # Instead of 100.0
# Or
VALUE_SUPPORT_MIN = -1000
VALUE_SUPPORT_MAX = 1000  # For reward head only
```

---

### 🟡 MEDIUM: Dynamics State Normalization May Hide Problems

**File:** `mcts_trader/jax_agent/muzero_model.py` (lines 171-172)

```python
# Normalize next_state to prevent drift over MCTS unroll steps
next_state = self.dynamics_state_norm(next_state)
```

**Concern:** LayerNorm on dynamics output prevents the latent state from drifting, but:
1. **May mask poor dynamics quality** - normalizes away magnitude errors
2. **Reduces expressiveness** - all latent states have similar norm
3. **May hurt MCTS** - values depend on state magnitude via prediction head

**Standard Practice:**
- Some MuZero implementations use normalization
- Others rely on proper consistency loss to prevent drift
- With consistency loss, normalization may be unnecessary

**Recommendation:** 
- Remove or make optional once consistency loss is added
- Monitor `latent_norm` metric in logs

---

### 🟢 LOW: K-Step Unroll Implementation is Correct

**Verified:** The K-step unrolling in `_train_step` correctly:
1. Starts from initial observation encoding
2. Iteratively applies dynamics for `t=1..unroll_steps`
3. Computes losses at each step
4. Uses correct indexing for reward targets (`t-1`)

**File:** `mcts_trader/jax_agent/trainer.py` (lines 1119-1164)

```python
# Correct: reward at t-1 corresponds to action leading to state t
scaled_rewards = rewards[:, t - 1] * self.reward_scale
```

---

### 🟢 LOW: Action Embedding Architecture is Standard

**Verified:** The dynamics model properly:
1. Embeds discrete actions
2. Concatenates with latent state
3. Passes through MLP
4. Outputs next state and reward

**File:** `mcts_trader/jax_agent/muzero_model.py` (lines 155-176)

---

## Architecture Comparison: Standard vs Current

| Component | Standard MuZero | Current Implementation | Issue |
|-----------|----------------|----------------------|-------|
| **Representation** | CNN → latent | TRMEncoder (CNN) → state_dim | ✅ OK |
| **Dynamics** | (s,a) → s', r | (s,a) → s', r | ✅ OK |
| **Prediction** | (s) → policy, value | TRMReasoning (recursive) → policy, value | Different but valid |
| **Consistency Loss** | **REQUIRED** | **MISSING** | 🔴 Critical |
| **Gradient Blocking** | Stop grad in unroll | None | 🟡 Issue |
| **Reward Support** | Aligned with value | May be misaligned | 🟡 Check |
| **State Norm** | Optional | Always on | 🟡 May mask issues |

---

## Recommendations

### Priority 1: Add Consistency Loss (CRITICAL)

**New file:** `mcts_trader/jax_agent/consistency.py`

```python
"""Latent consistency loss for MuZero."""
import jax.numpy as jnp

def compute_consistency_loss(
    predicted_states: jnp.ndarray,
    encoded_states: jnp.ndarray,
    loss_type: str = 'mse'
) -> jnp.ndarray:
    """
    Compute consistency between dynamics predictions and encoder outputs.
    
    Args:
        predicted_states: (batch, state_dim) from dynamics model
        encoded_states: (batch, state_dim) from representation model
        loss_type: 'mse', 'cosine', or 'huber'
    
    Returns:
        Scalar consistency loss
    """
    if loss_type == 'mse':
        return jnp.mean((predicted_states - encoded_states) ** 2)
    elif loss_type == 'cosine':
        # Normalize then MSE (focus on direction, not magnitude)
        pred_norm = predicted_states / (jnp.linalg.norm(predicted_states, axis=-1, keepdims=True) + 1e-8)
        enc_norm = encoded_states / (jnp.linalg.norm(encoded_states, axis=-1, keepdims=True) + 1e-8)
        return jnp.mean((pred_norm - enc_norm) ** 2)
    elif loss_type == 'huber':
        diff = jnp.abs(predicted_states - encoded_states)
        quadratic = 0.5 * diff ** 2
        linear = diff - 0.5
        return jnp.mean(jnp.where(diff <= 1.0, quadratic, linear))
```

**Integration in trainer.py:**
```python
# In _train_step, inside unroll loop
if self.use_consistency_loss and t < unroll_steps:
    # Encode next observation
    next_encoded = self.model.representation(observations[:, t])
    
    # Consistency loss
    consist_loss = compute_consistency_loss(
        latent_state,  # From dynamics
        next_encoded,  # From encoder
        loss_type='cosine'  # Recommended
    )
    total_loss += self.consistency_weight * consist_loss
    metrics['consistency_loss'] = consist_loss
```

### Priority 2: Add Gradient Blocking

**File:** `mcts_trader/jax_agent/muzero_model.py`

```python
def recurrent_inference(self, latent_state, action, stop_grad=False):
    next_latent_state, reward = self.dynamics(latent_state, action)
    
    if stop_grad:
        # Prevent dynamics gradients from affecting encoder
        next_latent_state = jax.lax.stop_gradient(next_latent_state)
    
    policy_logits, value = self.prediction(next_latent_state)
    return {...}
```

### Priority 3: Audit Reward Scale vs Value Support

**Action:** Log histograms of:
- Raw rewards from environment
- Symlog rewards
- Scaled rewards (`* REWARD_SCALE`)
- Distribution over value support bins

Verify that >95% of rewards fall within `[-10, +10]` after scaling.

If not:
- Either reduce `REWARD_SCALE` from 100 to 10
- Or expand value support for reward head

### Priority 4: Make State Normalization Optional

**File:** `mcts_trader/jax_agent/muzero_model.py`

```python
@nn.compact
def dynamics(self, latent_state, action, normalize_state=True):
    ...
    if normalize_state:
        next_state = self.dynamics_state_norm(next_state)
    return next_state, reward
```

---

## Testing Plan

1. **Consistency Loss Sanity Check:**
   ```python
   # Before training
   obs = batch['observations']
   s0 = model.representation(obs[:, 0])
   s1_pred = model.dynamics(s0, actions[:, 0])
   s1_actual = model.representation(obs[:, 1])
   initial_consistency = MSE(s1_pred, s1_actual)
   print(f"Initial consistency loss: {initial_consistency}")
   # Should decrease with training
   ```

2. **Gradient Flow Test:**
   ```python
   # Check if dynamics gradients flow to encoder
   loss = compute_loss(...)
   grads = jax.grad(loss)(params)
   
   # Should be NON-ZERO for dynamics
   assert jnp.any(grads['dynamics_dense1'] != 0)
   
   # Should be ZERO for encoder if stop_grad enabled
   # (or very small if not)
   print(f"Encoder grad norm: {jnp.linalg.norm(grads['trm']}")
   ```

3. **Reward Support Test:**
   ```python
   # Check clipping
   scaled_rewards = rewards * REWARD_SCALE
   clipped = jnp.clip(scaled_rewards, VALUE_SUPPORT_MIN, VALUE_SUPPORT_MAX)
   clipping_rate = jnp.mean(scaled_rewards != clipped)
   print(f"Reward clipping rate: {clipping_rate}")
   assert clipping_rate < 0.05, "Too much reward clipping!"
   ```

---

## Acceptance Criteria

- [ ] Consistency loss implemented and added to total loss
- [ ] Gradient blocking in recurrent_inference (configurable)
- [ ] Reward scale validated against value support (or support expanded)
- [ ] State normalization made optional
- [ ] Metrics added: `consistency_loss`, `latent_norm_dynamics`, `latent_norm_encoded`
- [ ] Training shows decreasing consistency loss over time
- [ ] No degradation in value/policy performance

---

## References

- MuZero paper (Schrittwieser et al., 2020): https://arxiv.org/abs/1911.08265
- "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model"
- Standard MuZero implementation: https://github.com/deepmind/mctx
- `mcts_trader/jax_agent/muzero_model.py` - Model architecture
- `mcts_trader/jax_agent/trainer.py` - Training loop
- `mcts_trader/jax_agent/flow_matching.py` - Velocity prediction (related work)
