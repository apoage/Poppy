# TASK_127g: Entropy and Policy Collapse Audit

**Status**: AUDITED (Working, Minor Improvements Possible)
**Priority**: Medium
**Created**: 2026-01-31

## Scope

Audited entropy and policy collapse mechanisms:
- `mcts_trader/jax_agent/trainer.py` - Entropy computation and adaptive coefficient
- `mcts_trader/jax_agent/trm_flax.py` - Policy head architecture
- `mcts_trader/jax_agent/mctx_search.py` - MCTS temperature and exploration
- `mcts_trader/jax_agent/exploration/sac_entropy.py` - SAC-style adaptive entropy

## Entropy Computation

### 1. How Entropy is Computed ✅ CORRECT

**Location**: `trainer.py:1105-1108, 1161-1164`

```python
# Compute policy entropy for regularization (prevent collapse)
policy_probs = jax.nn.softmax(initial_output['policy_logits'], axis=-1)
per_sample_entropy = -jnp.sum(policy_probs * jnp.log(policy_probs + 1e-10), axis=-1)
policy_entropy = jnp.sum(per_sample_entropy * valid_mask_float) / num_valid
```

**Formula**: Standard categorical entropy
```
H(π) = -Σ p(a) * log(p(a))
```

**Assessment**:
- ✅ Uses categorical entropy (correct for discrete actions)
- ✅ Computed per sample, then averaged with valid mask
- ✅ Accumulated over unroll steps, then averaged

### 2. Is log(0) Protected? ✅ YES

**Location**: `trainer.py:1107, 1163`

```python
per_sample_entropy = -jnp.sum(policy_probs * jnp.log(policy_probs + 1e-10), axis=-1)
```

**Assessment**:
- ✅ Uses `log(p + 1e-10)` to prevent log(0) = -inf
- ✅ Consistent across all entropy computations in trainer

**Other log protections in codebase**:
| Location | Protection |
|----------|------------|
| `trainer.py:1107, 1163` | `+ 1e-10` |
| `sac_entropy.py:166` | Uses `log_softmax` (numerically stable) |
| `path_solver.py:484` | `+ 1e-10` |

### 3. Entropy Coefficient Scheduling

**Current Implementation**: Static coefficient with adaptive bonus

```python
# trainer.py:377
entropy_coef: float = 0.01,

# trainer.py:1174-1189 - Adaptive bonus
adaptive_bonus = jnp.where(
    policy_entropy < self.min_entropy,
    entropy_coef * 5.0 * (self.min_entropy - policy_entropy),
    0.0
)

# Loss formulation
muzero_loss = (
    ...
    - entropy_coef * policy_entropy  # Maximize entropy
    + adaptive_bonus  # Extra penalty when too low
)
```

**Assessment**:
- ✅ Base coefficient (0.01) SUBTRACTED from loss → maximize entropy
- ✅ Adaptive bonus kicks in when entropy < min_entropy (default 3.0)
- ✅ 5x stronger push when entropy drops below threshold
- ⚠️ No scheduled decay of entropy coefficient over training

## Policy Collapse Prevention Mechanisms

### 1. Entropy Floor ✅ IMPLEMENTED

**Location**: `trainer.py:1177-1181`

```python
# Target entropy ~4.0 (near-uniform over 192 actions = ln(192) ≈ 5.26)
adaptive_bonus = jnp.where(
    policy_entropy < self.min_entropy,  # Default 3.0
    entropy_coef * 5.0 * (self.min_entropy - policy_entropy),
    0.0
)
```

**Entropy scale reference**:
| Actions | Max Entropy | 50% | Min Threshold |
|---------|-------------|-----|---------------|
| 192 | ln(192) = 5.26 | 2.63 | 3.0 (default) |
| 320 | ln(320) = 5.77 | 2.89 | 3.0 |

**Assessment**:
- ✅ Floor at 3.0 nats (configurable via `--min-entropy`)
- ✅ Non-linear penalty: stronger push as entropy drops further
- ⚠️ Reactive mechanism (triggers AFTER collapse starts)

### 2. SAC-Style Adaptive Entropy (Alternative) 📋 IMPLEMENTED BUT NOT DEFAULT

**Location**: `exploration/sac_entropy.py`

```python
def update_alpha(state, optimizer, policy_entropy):
    """
    Soft Actor-Critic style entropy coefficient update.

    α* = argmin_α E[α * (H(π) - H_target)]

    If H(π) < H_target: increase α (more exploration)
    If H(π) > H_target: decrease α (less exploration)
    """
    alpha_loss = state.log_alpha * (policy_entropy - state.target_entropy)
    ...
```

**Assessment**:
- ✅ Proactive: maintains target entropy throughout training
- ✅ Smooth: gradual adjustment
- ⚠️ Not currently used in default training loop
- 📋 Available for future integration

### 3. MCTS Exploration Mechanisms ✅ IMPLEMENTED

**Location**: `mctx_search.py:217-220`

```python
# Standard MuZero with PUCT
policy_output = mctx.muzero_policy(
    ...
    temperature=self.temperature,          # Default 1.0
    dirichlet_alpha=self.dirichlet_alpha,  # Root exploration
    dirichlet_fraction=self.exploration_fraction,
    ...
)
```

**Additional**: ε-greedy mixing for training targets
```python
# trainer.py:230-235
if self.epsilon_greedy > 0:
    policy_probs = apply_epsilon_greedy(
        policy_probs,
        epsilon=self.epsilon_greedy,
        action_mask=action_mask,
    )
```

**Assessment**:
- ✅ Dirichlet noise at root for exploration
- ✅ Temperature-scaled action selection
- ✅ ε-greedy mixing breaks MCTS→policy→MCTS sharpening feedback loop

## Policy Head Architecture

### Logits Computation ✅ NO ISSUES

**Location**: `trm_flax.py:192`

```python
# After H×L reasoning cycles
policy_logits = output  # Raw logits, no temperature scaling
```

**Architecture**:
```
latent + output → Dense(hidden) → LN → GELU → Dense(n_actions) → policy_logits
     ↑ (residual)                                                      ↓
     └──────────────────────────────────────────────────────────────────┘
```

**Assessment**:
- ✅ Raw logits output (no temperature in model)
- ✅ Residual connection for stable training
- ✅ Cast to float32 for numerical stability before softmax
- ⚠️ No logit clipping (could have extreme values)

### Temperature Scaling ⚠️ NOT IN MODEL

**Current state**: Temperature applied in MCTS, not in model

**Temperature usage across codebase**:
| Component | Temperature | Purpose |
|-----------|-------------|---------|
| MCTS action selection | 1.0 | Action sampling |
| ATC loss | 0.1 | Contrastive learning |
| Cycle ATC | 0.1 | Cycle contrastive |
| Imprint loss | configurable | Optimal action softmax |
| **Policy head** | **NONE** | Raw logits |

**Recommendation**: Consider adding optional temperature scaling to policy logits before softmax in training to control sharpness.

### Action Masking ✅ HANDLED IN MCTS

**Location**: `mctx_search.py:176-181, 207`

```python
if action_mask is not None:
    # Set invalid actions to very negative logit
    min_logit = jnp.finfo(root.prior_logits.dtype).min
    root_logits = jnp.where(action_mask.astype(bool), root.prior_logits, min_logit)
    ...
    invalid_actions=~action_mask.astype(bool)
```

**Assessment**:
- ✅ Invalid actions masked with minimum logit value
- ✅ MCTS respects action mask
- ✅ Model outputs unmasked logits (masking is environment-specific)

## Issues Found

### Issue 1: No Entropy Coefficient Decay ⚠️

**Severity**: Low

**Problem**: Entropy coefficient is static (0.01) throughout training.

**Impact**:
- Early training needs high exploration
- Late training should allow more exploitation
- Current adaptive bonus only prevents collapse, doesn't reduce exploration

**Recommendation**: Add optional entropy coefficient schedule:
```python
# Example: decay from 0.05 to 0.001 over training
entropy_coef = initial * (final/initial) ** (step/total_steps)
```

### Issue 2: Reactive vs Proactive Entropy Control ⚠️

**Severity**: Medium

**Problem**: Current adaptive bonus is REACTIVE - only kicks in after entropy drops below threshold.

```python
# Only triggers when already too low
adaptive_bonus = jnp.where(
    policy_entropy < self.min_entropy,  # Already collapsed!
    ...
)
```

**Impact**: Policy may already be partially collapsed before correction kicks in.

**Recommendation**: Use SAC-style proactive control that maintains target entropy continuously.

### Issue 3: No Logit Magnitude Monitoring ⚠️

**Severity**: Low

**Problem**: Policy logits could become very large, causing:
- Numerical instability in softmax
- Overconfident policy (near-deterministic)

**Current state**: No clipping or monitoring of logit magnitudes.

**Recommendation**: Add metrics:
```python
metrics['policy_logit_max'] = jnp.max(jnp.abs(policy_logits))
metrics['policy_logit_std'] = jnp.std(policy_logits)
```

### Issue 4: Entropy Computed on All Actions, Not Valid Only ⚠️

**Severity**: Low

**Problem**: Entropy is computed over ALL 192/320 actions, including potentially invalid ones.

```python
policy_probs = jax.nn.softmax(initial_output['policy_logits'], axis=-1)
per_sample_entropy = -jnp.sum(policy_probs * jnp.log(policy_probs + 1e-10), axis=-1)
```

If action mask exists, invalid actions still contribute to entropy computation.

**Impact**: Entropy metric may not reflect true policy diversity over valid actions.

**Recommendation**: Optionally compute entropy only over valid actions for more accurate metric.

## Collapse Risk Assessment

| Risk Factor | Mitigation | Status |
|-------------|------------|--------|
| Cross-entropy loss too strong | Entropy bonus | ✅ Mitigated |
| MCTS sharpening feedback | ε-greedy mixing | ✅ Mitigated |
| Low exploration in MCTS | Dirichlet noise | ✅ Mitigated |
| Extreme logit values | None | ⚠️ Not mitigated |
| Entropy decay below threshold | Adaptive bonus | ✅ Mitigated |
| Gradual exploration reduction | No schedule | ⚠️ Not controlled |

## Summary

| Aspect | Status | Notes |
|--------|--------|-------|
| Entropy formula | ✅ Correct | Categorical, -Σp*log(p) |
| log(0) protection | ✅ Protected | Uses +1e-10 |
| Entropy floor | ✅ Implemented | Adaptive bonus at min_entropy |
| Exploration decay | ⚠️ Missing | No scheduled reduction |
| Logit clipping | ⚠️ Missing | Could help stability |
| Action masking | ✅ Correct | In MCTS, not model |
| SAC-style entropy | 📋 Available | Not default |

## Recommendations

1. **Add entropy coefficient scheduling** - decay over training for exploration→exploitation
2. **Consider SAC-style proactive control** - maintains target entropy continuously
3. **Monitor logit magnitudes** - add metrics for max/std of policy logits
4. **Optional: Compute masked entropy** - only over valid actions for accurate metric

## Related Tasks

- TASK_040: SAC-style Adaptive Entropy (implemented but not default)
- TASK_037: ε-greedy MCTS mixing
- TASK_127b: MuZero Dynamics Audit
