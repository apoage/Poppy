# TASK_127: Training Pipeline Critical Issues Audit

**Date**: 2026-01-31
**Status**: Identified
**Priority**: HIGH

## Executive Summary

Comprehensive audit of Poppy training pipeline revealed **12 critical issues** that may explain "fundamentally broken" training behavior. Issues span gradient flow, loss computation, sampling bias, and configuration gaps.

---

## CRITICAL Issues (Training Breaking)

### 1. BC Confidence Masking Breaks Gradients
**File**: `trainer.py:1420-1422`
**Issue**: BC loss multiplied by confidence then averaged with `jnp.mean()`, but samples with `confidence=0` are included in denominator, diluting gradients.
```python
margin_loss = margin_loss * expert_confidence  # Zeros kill gradients
bc_loss_total = bc_weight * jnp.mean(margin_loss)  # Diluted by zero samples
```
**Fix**: Use masked mean: `jnp.sum(loss * mask) / jnp.sum(mask)`

### 2. PER Importance Weights Computed But Never Used
**File**: `expert_buffer.py:358-381` → `trainer.py`
**Issue**: Importance sampling weights are computed correctly but **never applied to loss**. All loss computations use uniform `jnp.mean()`.
**Impact**: PER is pure overhead with zero benefit. Training is biased toward high-priority samples without correction.
**Fix**: Apply `loss * batch['weights']` in trainer

### 3. PopArt Weight Correction Outside JIT
**File**: `trainer.py:1597-1623`, `popart.py:130-157`
**Issue**: Weight correction happens in Python loop (every 10 iterations), outside JIT-compiled `_train_step`. Gradients computed on corrected params don't inform the correction.
**Impact**: Value head learns against moving targets; gradients and corrections are desynchronized.
**Fix**: Move weight correction inside JIT or use proper value head architecture

### 4. Visit Counts Not Normalized Before Cross-Entropy
**File**: `self_play.py:219`, `trainer.py:1099-1103`
**Issue**: MCTS returns raw visit counts, passed directly to `softmax_cross_entropy` which expects probability distributions.
```python
policy_probs.append(np.array(policy_jax[0]))  # Raw counts!
per_sample_policy = optax.softmax_cross_entropy(logits, target_policies[:, 0])
```
**Impact**: Gradient magnitudes scale with visit count magnitudes, not probability errors.
**Fix**: Normalize: `policy_jax[0] / (jnp.sum(policy_jax[0]) + 1e-8)`

### 5. No MuZero Consistency Loss
**File**: `trainer.py:1079-1173`
**Issue**: Standard MuZero includes `||h(s') - dynamics(h(s), a)||²` to ensure dynamics outputs stay in representation space. **This is completely missing.**
**Impact**: Dynamics can learn transitions that don't align with encoder, causing 5-step unroll drift.
**Fix**: Add consistency auxiliary task in unrolling loop

---

## HIGH Issues (Training Degraded)

### 6. Demo vs Agent Priority Scale Mismatch
**File**: `expert_buffer.py:160-162, 199-201`
**Issue**: Demos use `confidence + 0.001` (range ~0.9), agents use `|td_error| + 1e-6` (range ~1-10). Agents have 10-100x more priority.
**Impact**: High-quality demos are undersampled despite being in "protected" zone.
**Fix**: Use multiplicative boost or normalize priority scales

### 7. Alpha Return Type Mismatch
**File**: `self_play.py:799-841`
**Issue**: `final_return` is cumulative log return, `market_return` is arithmetic mean of simple returns. These are not comparable.
**Impact**: Alpha metrics are systematically inflated, especially in volatile markets.
**Fix**: Convert both to same return space (log or simple)

### 8. Burning Money Sign Error in Bearish Markets
**File**: `jax_trading_env.py:1234`
**Issue**: When `market_return < 0`, the burn penalty becomes a reward due to symlog of negative value.
**Impact**: Agent learns wrong signal - rewarded for inaction in downturns.
**Fix**: Use linear penalty or ensure symlog handles negative correctly

### 9. Weight Decay Disabled (Not Configured)
**File**: `config.py`, `trainer.py:553-556`
**Issue**: `weight_decay` parameter exists in trainer but is **never passed from config**. Defaults to 0.
**Impact**: No L2 regularization, weights unbounded, potential overfitting.
**Fix**: Add `weight_decay: float = 1e-4` to TrainingConfig and pass to trainer

### 10. Learning Rate Schedule Constant by Default
**File**: `config.py:148`
**Issue**: `lr_schedule: str = "constant"` - no annealing happens unless explicitly changed.
**Impact**: May get stuck in local minima, no fine-tuning phase.
**Fix**: Change default to `"cosine"` or document need to enable

---

## MEDIUM Issues (Suboptimal)

### 11. jnp.where Evaluates Both Branches
**File**: `trainer.py:704, 927`
**Issue**: Huber loss uses `jnp.where(cond, quadratic, linear)` - JAX evaluates BOTH branches.
**Impact**: Conflicting gradient signals from both quadratic and linear terms.
**Fix**: Use smooth Huber approximation or `lax.cond` where possible

### 12. Adaptive Entropy Bonus Discontinuity
**File**: `trainer.py:1177-1181`
**Issue**: Entropy bonus jumps from 0 to 5x at threshold, creating gradient discontinuity.
```python
adaptive_bonus = jnp.where(entropy < min_entropy, entropy_coef * 5.0 * gap, 0.0)
```
**Impact**: Training instability when entropy oscillates around threshold.
**Fix**: Use smooth function like `tanh(k * (min_entropy - entropy))`

---

## Summary Table

| # | Issue | Severity | File | Impact |
|---|-------|----------|------|--------|
| 1 | BC confidence masking | CRITICAL | trainer.py:1420 | Gradients diluted |
| 2 | PER weights unused | CRITICAL | expert_buffer.py | Biased sampling |
| 3 | PopArt outside JIT | CRITICAL | trainer.py:1597 | Value learning broken |
| 4 | Visit counts raw | CRITICAL | self_play.py:219 | Policy loss wrong |
| 5 | No consistency loss | CRITICAL | trainer.py:1079 | Dynamics drift |
| 6 | Priority scale mismatch | HIGH | expert_buffer.py:160 | Demos undersampled |
| 7 | Alpha return mismatch | HIGH | self_play.py:799 | Metrics inflated |
| 8 | Burn sign error | HIGH | jax_trading_env.py:1234 | Wrong reward signal |
| 9 | Weight decay disabled | HIGH | config.py | Overfitting risk |
| 10 | LR constant default | HIGH | config.py:148 | No annealing |
| 11 | jnp.where both branches | MEDIUM | trainer.py:704 | Gradient conflict |
| 12 | Entropy discontinuity | MEDIUM | trainer.py:1177 | Oscillation |

---

## Recommended Fix Priority

**Phase 1 (Immediate - Stop Bleeding)**
1. Fix BC confidence masking (5 min)
2. Normalize visit counts before storing (1 line)
3. Enable weight decay in config (2 min)

**Phase 2 (Short-term - Core Fixes)**
4. Apply PER importance weights in loss
5. Fix PopArt timing (move into JIT or redesign)
6. Add consistency loss to MuZero unrolling

**Phase 3 (Medium-term - Polish)**
7. Fix priority scale for demos
8. Unify return type for alpha computation
9. Fix burn sign error
10. Change LR schedule default to cosine

---

## Verification Metrics

After fixes, monitor:
- `bc_loss` should be > 0.01 (not ~1e-05)
- `expert_action_agreement` should rise above 0.3% (random baseline)
- `value_loss` should decrease steadily (not oscillate)
- `policy_entropy` should stay in 3.0-5.0 range (not collapse)
- Alpha should correlate with optimal_alpha (not be systematically inflated)

---

## Notes

These issues compound. A training run with all 12 issues active would:
1. Learn from wrong BC targets (masking)
2. With biased sampling (PER unused)
3. Against moving value targets (PopArt)
4. With wrong policy gradients (raw visit counts)
5. And drifting dynamics (no consistency)
6. Without regularization (no weight decay)
7. At constant learning rate (no schedule)

This explains why previous runs showed "flat MA50 reward" and "constant MA30 alpha" without meaningful learning signal.
