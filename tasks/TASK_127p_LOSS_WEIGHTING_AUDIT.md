# TASK_127p: Loss Function Weighting Audit

**Status**: AUDITED (Issues Found)
**Priority**: Medium
**Created**: 2026-01-31

## Scope

Audited loss function composition:
- `mcts_trader/jax_agent/trainer.py` - Loss computation and weights
- `mcts_trader/training/config.py` - Default hyperparameters
- `mcts_trader/training/setup.py` - Actual values passed to trainer

## Loss Architecture Overview

```
total_loss Composition:
┌─────────────────────────────────────────────────────────────────┐
│ MuZero Base Loss (muzero_loss)                                  │
│  = value_loss_weight * value_loss     (default: 2.0)            │
│  + policy_loss_weight * policy_loss   (default: 1.0)            │
│  + reward_loss_weight * reward_loss   (default: 1.0)            │
│  - entropy_coef * policy_entropy      (default: 0.01)           │
│  + adaptive_bonus (when entropy < 3.0)                          │
└─────────────────────────────────────────────────────────────────┘
                              +
┌─────────────────────────────────────────────────────────────────┐
│ Optional Auxiliary Losses (conditional)                         │
│  + lambda_pred * prediction_loss      (0.01 → 0.2 annealed)     │
│  + physics_reg_weight * physics_reg   (default: 0.01)           │
│  + atc_weight * atc_loss              (default: 0.1)            │
│  + cycle_balance_weight * cycle_loss  (default: 0.1)            │
│  + cycle_atc_weight * cycle_atc       (default: 0.05)           │
│  + bc_weight * bc_loss                (0.1 → 0.01 decayed)      │
│  + amax_regret_weight * regret_loss   (default: 0.0)            │
│  + amax_imprint_weight * imprint_loss (default: 0.0)            │
└─────────────────────────────────────────────────────────────────┘
```

## Loss Weights Summary

### Core MuZero Losses

| Loss | Weight | Default | Actual | Source |
|------|--------|---------|--------|--------|
| `value_loss` | `value_loss_weight` | 1.0 | **2.0** | setup.py:511 |
| `policy_loss` | `policy_loss_weight` | 1.0 | 1.0 | trainer.py:376 |
| `reward_loss` | `reward_loss_weight` | 1.0 | 1.0 | trainer.py:375 |
| `policy_entropy` | `-entropy_coef` | 0.01 | 0.01 | config.py:152 |

### Auxiliary Losses (Optional)

| Loss | Weight | Default | Annealing |
|------|--------|---------|-----------|
| `prediction_loss` | `lambda_pred` | 0.01 → 0.2 | Linear over 10k steps |
| `physics_reg` | `physics_reg_weight` | 0.01 | None |
| `atc_loss` | `atc_weight` | 0.1 | None |
| `cycle_balance` | `cycle_balance_weight` | 0.1 | None |
| `cycle_atc` | `cycle_atc_weight` | 0.05 | None |
| `bc_loss` | `bc_weight` | 0.1 → 0.01 | Decay over 50k steps |
| `regret_loss` | `amax_regret_weight` | 0.0 | None |
| `imprint_loss` | `amax_imprint_weight` | 0.0 | None |

## Loss Scale Analysis

### Expected Loss Magnitudes

| Loss Type | Formula | Typical Scale | Notes |
|-----------|---------|---------------|-------|
| **Value MSE** | `(pred - target)²` | 0.1 - 10.0 | Depends on PopArt normalization |
| **Value Categorical** | `CE(logits, target_dist)` | 1.0 - 6.0 | max = ln(401) ≈ 6 |
| **Policy CE** | `CE(logits, mcts_policy)` | 0.5 - 5.0 | max = ln(320) ≈ 5.77 |
| **Reward CE** | `CE(logits, scaled_reward)` | 1.0 - 6.0 | max = ln(401) ≈ 6 |
| **Entropy** | `-Σ p*log(p)` | 0.0 - 5.26 | max = ln(192) |

### Hybrid Value Loss Breakdown

**Location**: `trainer.py:665-733`

```python
total_loss = mse_weight * mse_loss + (1.0 - mse_weight) * cat_loss
# Default: 80% MSE, 20% categorical
```

| Component | Weight | Typical Range |
|-----------|--------|---------------|
| MSE/Huber | 0.8 | 0.1 - 10.0 |
| Categorical | 0.2 | 1.0 - 6.0 |

## Issues Found

### Issue 1: Value Loss Weight Hardcoded to 2.0 ⚠️

**Severity**: Medium

**Location**: `setup.py:510-511`

```python
# C1 fix: Match original value_loss_weight from .bak (was 2.0, trainer default is 1.0)
value_loss_weight=2.0,
```

**Problem**: Value loss weight is hardcoded in setup.py, overriding the trainer's default. This means:
- Value loss contributes 2x relative to policy/reward losses
- No CLI flag to change this
- Inconsistent with trainer's documented defaults

**Impact**: Value learning may dominate policy learning, especially early in training when value errors are large.

**Recommendation**: Either:
1. Add `--value-loss-weight` CLI argument
2. Or verify 2.0 is intentional and document why

### Issue 2: No Loss Scaling/Normalization ⚠️

**Severity**: Medium

**Problem**: Different losses have different natural scales:

```
Typical magnitudes (rough estimates):
- Value loss (MSE): ~1.0-5.0 (with PopArt)
- Policy loss (CE): ~1.0-3.0
- Reward loss (CE): ~2.0-4.0
- Entropy: ~3.0-5.0

With weights applied:
- value_loss * 2.0 = ~2.0-10.0
- policy_loss * 1.0 = ~1.0-3.0
- reward_loss * 1.0 = ~2.0-4.0
- entropy * 0.01 = ~0.03-0.05
```

**Impact**:
- Value loss likely dominates total loss
- Entropy bonus is very small relative to other losses
- Gradient magnitudes will be imbalanced

**Recommendation**: Consider loss normalization or adaptive weighting.

### Issue 3: Entropy Coefficient Too Small? ⚠️

**Severity**: Low

**Location**: `config.py:152, trainer.py:1188`

```python
entropy_coef: float = 0.01
...
- entropy_coef * policy_entropy  # Maximize entropy
```

**Problem**: With entropy ~3-5 and coefficient 0.01, entropy bonus contributes only 0.03-0.05 to the loss, while value loss alone can be 2-10.

**Assessment**:
- Entropy bonus may be too weak to prevent collapse
- Relies heavily on adaptive_bonus (5x multiplier when entropy < 3.0)
- Reactive rather than proactive exploration maintenance

**Recommendation**: Consider increasing default entropy_coef to 0.05-0.1 for more consistent exploration.

### Issue 4: Prediction Loss Annealing May Cause Late-Training Instability ⚠️

**Severity**: Low

**Location**: `trainer.py:282-286, 315-318`

```python
weight_initial: float = 0.01      # Start low
weight_final: float = 0.2         # Anneal to this
anneal_steps: int = 10000         # Steps to reach final weight
```

**Problem**: Prediction loss weight increases 20x during training (0.01 → 0.2).

**Impact**:
- Loss landscape changes significantly during training
- Late training may become dominated by prediction loss
- Optimizer momentum from early training may be suboptimal for late loss balance

**Recommendation**: Monitor prediction_loss vs muzero_loss ratio throughout training.

### Issue 5: BC Weight Decay Direction Seems Correct ✅

**Location**: `trainer.py:262-264, 321-324`

```python
bc_weight_initial: float = 0.1     # Start with strong imitation
bc_weight_final: float = 0.01      # Decay as agent improves
bc_weight_decay_steps: int = 50000
```

**Assessment**: BC loss decays 10x over training (0.1 → 0.01), which is correct:
- Strong imitation early when agent is learning
- Weak imitation late to allow policy specialization

### Issue 6: Multiple Losses Disabled by Default ✅

Several auxiliary losses are disabled (weight = 0) by default:
- `amax_regret_weight: 0.0`
- `amax_imprint_weight: 0.0`
- `atc_loss: False` (boolean)
- `cycle_balance: False` (boolean)

**Assessment**: Good - keeps loss function simple by default. Features can be enabled when needed.

## Loss Composition Trace

### Default Configuration (No Auxiliary Losses)

```python
total_loss = muzero_loss
         = 2.0 * value_loss + 1.0 * policy_loss + 1.0 * reward_loss
           - 0.01 * entropy + adaptive_bonus

# Approximate magnitudes:
#   2.0 * 3.0 = 6.0  (value)
# + 1.0 * 2.0 = 2.0  (policy)
# + 1.0 * 3.0 = 3.0  (reward)
# - 0.01 * 4.0 = -0.04 (entropy)
# ≈ 11.0 total

# Value loss contribution: ~55%
# Policy loss contribution: ~18%
# Reward loss contribution: ~27%
# Entropy contribution: <1%
```

### With Prediction Loss Enabled (Late Training)

```python
total_loss = muzero_loss + 0.2 * prediction_loss + 0.01 * physics_reg

# If prediction_loss ≈ 5.0:
# 11.0 + 1.0 + 0.05 ≈ 12.0

# Prediction loss contribution: ~8%
```

### With Full Auxiliary Suite

```python
total_loss = muzero_loss
           + lambda_pred * prediction_loss
           + physics_reg_weight * physics_reg
           + atc_weight * atc_loss
           + cycle_balance_weight * cycle_loss
           + bc_weight * bc_loss
           + regret_weight * regret_loss
           + imprint_weight * imprint_loss

# Potential total: 11 + 1 + 0.05 + 0.5 + 0.5 + 0.1 + 0.1 + 0.1 ≈ 13.4
```

## Gradient Balance Analysis

### Per-Head Gradient Clipping (Optional)

**Location**: `trainer.py:400-404`

```python
per_head_clipping: bool = False,
clip_grad_value: float = 1.0,
clip_grad_policy: float = 1.0,
clip_grad_dynamics: float = 1.0,
clip_grad_encoder: float = 1.0,
```

**Assessment**: Per-head clipping is disabled by default. When enabled, all heads clip at 1.0 (equal treatment).

**Recommendation**: If loss imbalance is causing gradient imbalance, consider enabling per-head clipping with adjusted thresholds:
- Value head: higher clip (e.g., 2.0) since value_loss_weight = 2.0
- Policy head: standard (1.0)
- Dynamics head: standard (1.0)

## Summary

| Aspect | Status | Notes |
|--------|--------|-------|
| Core loss weights | ⚠️ Imbalanced | Value 2x others |
| Entropy coefficient | ⚠️ Low | 0.01 may be too small |
| Loss scaling | ⚠️ None | No normalization |
| Prediction annealing | ⚠️ Check | 20x weight increase |
| BC weight decay | ✅ Correct | Decays over training |
| Auxiliary losses | ✅ Clean | Disabled by default |
| Per-head clipping | ✅ Available | Disabled by default |

## Recommendations

1. **Add CLI flags for loss weights**:
   - `--value-loss-weight` (currently hardcoded 2.0)
   - `--policy-loss-weight`
   - `--reward-loss-weight`

2. **Consider loss normalization**:
   ```python
   # Option: Normalize losses to similar scale before weighting
   normalized_value = value_loss / jax.lax.stop_gradient(value_loss + 1e-8)
   ```

3. **Monitor loss ratios in DS24**:
   - Add `value_loss_ratio = value_loss / total_loss`
   - Add `policy_loss_ratio = policy_loss / total_loss`
   - Track how ratios change during training

4. **Consider higher entropy coefficient**:
   - Default 0.01 may be too weak
   - Try 0.05 or adaptive entropy (SAC-style, TASK_040)

5. **Verify value_loss_weight=2.0 is intentional**:
   - Document rationale for 2x weight
   - Or reduce to 1.0 for balanced training

## Related Tasks

- TASK_127g: Entropy and Policy Collapse Audit
- TASK_040: SAC-style Adaptive Entropy
- TASK_066: Physics Configuration
- TASK_125: Causal Expert Imprinting
