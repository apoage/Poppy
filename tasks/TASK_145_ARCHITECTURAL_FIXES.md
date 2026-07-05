# TASK_145: Critical Architectural Fixes

**Status:** COMPLETE - Ready for Testing

**Created:** 2026-02-10

---

## Implementation Summary (2026-02-10)

### Changes Made

| Fix | File | Change |
|-----|------|--------|
| Multi-discrete loss | `easy_trainer.py:145-156` | Reshape to (B,64,5), per-asset CE |
| Action-weighted loss | `easy_trainer.py:157-165` | 5x weight for trades vs HOLD |
| Success threshold | `easy_config.py:68` | 0.85 → 0.995 |
| Oracle k | `easy_config.py:46-48` | k=1 (value head must match execution) |
| Hold ratio metric | `easy_trainer.py` | Use per-asset action type |
| Label alignment | Verified | Correct (obs[i] → action[i]) |

---

## Adversarial Review Findings (Post-Implementation)

### Issue 1: Value Head Logic Bomb (FIXED)
**Problem**: With `oracle_k=10`, value head learns 10-asset portfolio returns. But model executes 1 trade/step.

**Impact**: MCTS sees value=+2% but can only achieve +0.2% → tree collapse.

**Fix**: Reverted `oracle_k=1`. Value head now predicts realizable single-trade returns.

### Issue 2: Majority HOLD Gradient Washout (FIXED)
**Problem**: With k=1, oracle is 98.4% HOLD. Model learns to "lazy hold" for 90% agreement.

**Math**: 100% HOLD accuracy + 0% trade accuracy = 98% agreement. Trade signals are only 2% of gradient.

**Fix**: Action-weighted loss. Trades weighted 5x:
```python
is_trade = targets_reshaped[:, :, 1] < 0.5  # NOT HOLD
loss_weights = jnp.where(is_trade, 5.0, 1.0)
policy_loss = jnp.sum(per_asset_ce * loss_weights) / jnp.sum(loss_weights)
```

### Issue 3: Softmax Tie-Breaker (ACKNOWLEDGED)
**Problem**: Inference uses `argmax(320)` which picks lowest asset index on ties.

**Status**: Deferred. With k=1, there's only 1 trade signal per step. No ties expected.

**Future Fix**: For k>1, use alpha-scaled logits in `path_solver.py`:
```python
logits = expected_return * 100  # 2% move → logit 2.0
```

---

## Summary

Two fundamental architectural bugs prevent meaningful learning. The model achieves 88% "agreement" in 1 epoch by gaming broken metrics, not by learning to trade.

---

## BUG 1: Softmax Competition (Policy Head)

### The Problem

The policy head outputs 320 logits (64 assets × 5 actions) and applies a **single softmax** over all 320 dimensions.

```python
# BROKEN: All 64 assets compete for probability mass
policy_probs = jax.nn.softmax(policy_logits, axis=-1)  # (batch, 320)
```

### Why This Breaks Training

1. **Probability Competition**: If Asset 7 should BUY (logit=5.0) and Asset 0 should HOLD (logit=0.0):
   - BUY gets ~70% probability
   - HOLD gets ~0.5% probability
   - Cross-entropy **punishes** the model for being confident in HOLD

2. **Impossible Representation**: The model cannot express "HOLD all 64 assets" because softmax forces probability to sum to 1.0 across ALL 320 actions.

3. **Loss Trap**: The model learns to spray probability across random actions to avoid the HOLD penalty, creating noisy gradients.

### The Math

With single softmax over 320 dims:
```
P(HOLD asset 0) = exp(0) / (exp(5) + 63*exp(0) + 256*exp(-inf))
                = 1 / (148.4 + 63)
                = 0.47%
```

The model is punished for correctly predicting HOLD!

### The Fix

Reshape to (batch, 64, 5) and apply **independent softmax per asset**:

```python
# CORRECT: Each asset gets independent 5-way classification
policy_reshaped = policy_logits.reshape(-1, 64, 5)
policy_probs = jax.nn.softmax(policy_reshaped, axis=-1)  # (batch, 64, 5)

# Per-asset cross-entropy loss
per_asset_loss = optax.softmax_cross_entropy(policy_reshaped, targets_reshaped)
loss = jnp.mean(per_asset_loss)
```

---

## BUG 2: Value Head is Blind

### The Evidence

From training logs:
```
val_value_mse: 23.5357
value_scale: 10.0
```

### The Math

```
MSE = 23.5
RMSE = sqrt(23.5) = 4.85 (in scaled space)
Error = 4.85 / value_scale = 4.85 / 10 = ±48.5% (in return space)
```

The value head predicts returns with **±48% error**. It thinks every state is either a crash or a moon.

### Why This Happens

1. **Underfitting**: Value head has insufficient capacity or wrong architecture
2. **Target Scale**: Even with value_scale=10, targets may be too compressed
3. **Gradient Competition**: Value loss competes with policy loss for encoder gradients

### The Fix

1. **Increase value head capacity**: More hidden units, deeper MLP
2. **Separate learning rate**: Value head needs slower, more stable updates
3. **Gradient isolation**: Stop gradients from value head to encoder (or use separate encoder)

---

## Impact on Training

### Before Fix (Current State)
- Model achieves 88% agreement in 1 epoch
- Actually just learning to spam HOLD (98.4% of oracle is HOLD with k=1)
- Value head is random noise
- "Success" threshold trivially met

### After Fix (Expected)
- Per-asset agreement requires learning actual trade patterns
- Value head can distinguish good/bad states
- Training will be harder but meaningful

---

## Implementation Plan

### Phase 1: Policy Head Fix (CRITICAL)

Files to modify:
- `easy_trainer.py`: Reshape logits, per-asset cross-entropy
- `mcts_trader/jax_agent/trainer.py`: Same fix for main trainer

Changes:
```python
# In loss computation:
n_assets = 64
n_actions_per_asset = 5

# Reshape for per-asset softmax
policy_reshaped = policy_logits.reshape(-1, n_assets, n_actions_per_asset)
targets_reshaped = targets.reshape(-1, n_assets, n_actions_per_asset)

# Per-asset cross-entropy (independent softmax per asset)
per_asset_ce = optax.softmax_cross_entropy(policy_reshaped, targets_reshaped)
policy_loss = jnp.mean(per_asset_ce)
```

### Phase 2: Value Head Fix

Files to modify:
- `mcts_trader/jax_agent/trm_flax.py`: Increase value head capacity
- `easy_config.py`: Add value_lr_scale parameter

Changes:
- Double value head hidden dim (currently `state_dim * 4`)
- Add separate optimizer for value head with lower LR
- Consider gradient clipping for value updates

---

## Verification

After fix, expect:
1. **Agreement drops initially**: Model can no longer game single softmax
2. **Slower convergence**: Actually learning patterns, not shortcuts
3. **Value MSE decreases**: < 1.0 target (±10% error in return space)
4. **Hold ratio matches**: pred_hold_ratio ≈ oracle_hold_ratio

---

## Root Cause Analysis

Why wasn't this caught earlier?

1. **Metric mismatch**: Per-asset agreement metric was added (TASK_143) but loss was still global softmax
2. **Easy oracle**: With k=1, oracle is 98.4% HOLD - trivially matched by HOLD-everything policy
3. **Fast "success"**: 85% threshold met in epoch 0, masking the real problem

The model wasn't learning to trade. It was learning to exploit a broken loss function.
