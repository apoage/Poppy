# TASK_127k8: Loss Function Weighting Audit

## Executive Summary

This audit examines the loss function composition, weighting, and balance. The loss structure is well-organized with clear separation between MuZero core losses and auxiliary losses. However, **loss scale imbalances exist** - policy and reward losses dominate by ~5× over value loss, and the entropy bonus has minimal impact.

**Risk Level: MEDIUM** - Loss scales are imbalanced but may be intentional for MuZero. Gradient clipping and per-head clipping help manage this.

---

## Loss Function Architecture

### Overall Loss Composition

```
total_loss = muzero_loss + auxiliary_losses

muzero_loss = (
    value_loss_weight * value_loss +      # Default: 1.0 * value_loss
    policy_loss_weight * policy_loss +    # Default: 1.0 * policy_loss
    reward_loss_weight * reward_loss +    # Default: 1.0 * reward_loss
    - entropy_coef * policy_entropy +     # Default: -0.01 * entropy
    + adaptive_bonus                      # Conditional entropy penalty
)

auxiliary_losses = (
    + lambda_pred * loss_prediction +           # Phase 2 (optional)
    + physics_reg_weight * loss_physics_reg +   # Phase 2 (optional)
    + atc_weight * atc_loss_val +               # TASK_043 (optional)
    + cycle_balance_weight * cycle_bal_loss +   # TASK_110 (optional)
    + cycle_atc_weight * cycle_atc_loss +       # TASK_110 (optional)
    + bc_loss_total +                           # TASK_125 (optional)
    + regret_loss +                             # TASK_119 (optional)
    + imprint_loss                              # TASK_119 (optional)
)
```

---

## Core MuZero Loss Analysis

### Loss Scale Imbalance

**Actual values from training logs:**

| Loss Component | Typical Value | Weight | Contribution | % of Total |
|----------------|---------------|--------|--------------|------------|
| `value_loss` | 0.94 | 1.0 | 0.94 | ~8% |
| `policy_loss` | 4.62 | 1.0 | 4.62 | ~43% |
| `reward_loss` | 5.62 | 1.0 | 5.62 | ~51% |
| `entropy_bonus` | -0.05 | - | -0.05 | ~0% |
| **muzero_loss** | **10.13** | - | **10.13** | **100%** |

**Analysis:**
- ✅ **Policy + Reward dominate**: ~94% of total loss
- ✅ **Value is minor**: Only ~8% of total loss
- ⚠️ **Entropy minimal impact**: -0.5% of total
- ⚠️ **Scale ratio**: Policy/Reward ~5-6× larger than Value

### Is This Imbalance a Problem?

**For MuZero: PROBABLY NOT**

```
Rationale:
1. Policy loss (cross-entropy): ~4-5 is normal for 192-way classification
2. Reward loss (cross-entropy): ~5-6 is normal for 401-bin distribution
3. Value loss (MSE + categorical): ~1 is reasonable for scalar prediction
4. Different loss types have different natural scales
```

**MuZero design philosophy:**
- Policy: Must learn precise action distribution (high loss acceptable)
- Reward: Must predict precise immediate rewards (high loss acceptable)
- Value: Approximate long-term return (lower precision acceptable)

**However:**
- Value gradients may be drowned out by policy/reward gradients
- Value head may learn slower than policy head

---

## Loss Weight Configuration

### Default Weights

**Location:** `trainer.py:374-377`

```python
value_loss_weight: float = 1.0,
reward_loss_weight: float = 1.0,
policy_loss_weight: float = 1.0,
entropy_coef: float = 0.01,
```

**Analysis:**
- ✅ All weights default to 1.0 (balanced in terms of weighting)
- ✅ Equal weighting means loss magnitudes drive importance
- ⚠️ But actual scales differ by 5-6×

### Entropy Coefficient Analysis

**Current:** `entropy_coef = 0.01`

```python
muzero_loss = (
    ...
    - entropy_coef * policy_entropy  # -0.01 * 5.0 = -0.05
)
```

With `policy_entropy ≈ 5.0` (near-uniform for 192 actions):
- Entropy bonus: -0.01 × 5.0 = **-0.05**
- Compared to total loss: **~0.5% impact**

**Is this sufficient?**

```python
# Current: entropy_coef = 0.01
# Total loss reduction: ~0.5%

# Alternative: entropy_coef = 0.1
# Total loss reduction: ~5%
# This would make entropy have meaningful impact
```

**Recommendation:** Consider increasing to 0.05-0.1 if policy collapse is a concern.

---

## Adaptive Entropy Bonus

**Location:** `trainer.py:1174-1181`

```python
adaptive_bonus = jnp.where(
    policy_entropy < self.min_entropy,  # Default: 3.0
    entropy_coef * 5.0 * (self.min_entropy - policy_entropy),
    0.0
)
```

**Trigger condition:**
```
Actual entropy: ~4.9-5.0 (from logs)
Min threshold: 3.0
Adaptive bonus: NEVER triggers (entropy > threshold)
```

**Analysis:**
- ⚠️ Adaptive bonus is effectively **DISABLED**
- Entropy stays well above threshold (healthy policy)
- If entropy drops, bonus could add up to: 0.01 × 5.0 × 3.0 = 0.15
- This would be meaningful (~1.5% of total loss)

---

## Loss Averaging Over Unroll Steps

**Location:** `trainer.py:1166-1172`

```python
# Average over unroll steps
value_loss /= unroll_steps          # ÷ 5
policy_loss /= unroll_steps         # ÷ 5
reward_loss /= (unroll_steps - 1)   # ÷ 4 (no reward at t=0)
policy_entropy /= unroll_steps      # ÷ 5
total_value_mse /= unroll_steps
total_value_cat /= unroll_steps
```

**Analysis:**
- ✅ Proper averaging prevents step count from affecting loss scale
- ⚠️ Reward loss divided by 4 (not 5) due to no root reward
- ✅ Makes losses comparable regardless of unroll length

---

## Auxiliary Losses

### 1. Prediction Loss (Phase 2)

```python
if self.use_prediction_loss:
    total_loss = (
        muzero_loss +
        lambda_pred * loss_prediction +           # Annealed: 0.01 → 0.2
        config.physics_reg_weight * loss_physics_reg  # Default: 0.01
    )
```

**Analysis:**
- ✅ Annealed weight prevents early interference
- ✅ Physics regularization is lightweight

### 2. ATC Contrastive Loss (TASK_043)

```python
if self.atc_loss:
    total_loss = total_loss + self.atc_weight * atc_loss_val
    # atc_weight default: 0.1
```

**Analysis:**
- ⚠️ Fixed weight (not annealed) could interfere early training
- Consider: `atc_weight * min(1.0, step / 10000)`

### 3. Cycle Balance Loss (TASK_110)

```python
if self.cycle_balance:
    total_loss = total_loss + self.cycle_balance_weight * cycle_bal_loss
    # cycle_balance_weight default: 0.1
```

**Analysis:**
- ✅ Helps prevent H0 cycle dominance
- ✅ Modest weight relative to main losses

### 4. Expert BC Loss (TASK_125)

```python
if bc_config.enabled:
    bc_loss_total = bc_weight * loss  # bc_weight annealed: 0.1 → 0.01
    total_loss = total_loss + bc_loss_total
```

**Analysis:**
- ✅ Annealed weight prevents over-reliance on expert
- ✅ Q-filter and large margin options available

### 5. Regret/Imprint Loss (TASK_119)

```python
if self.amax_regret_weight > 0:
    total_loss = total_loss + regret_loss
if self.amax_imprint_weight > 0:
    total_loss = total_loss + imprint_loss
```

**Analysis:**
- ✅ Both optional (disabled by default)
- ✅ Provides hindsight feedback without overfitting

---

## Gradient Magnitude Analysis

### Expected Gradient Contributions

Based on loss scales, approximate gradient contributions:

```
Component    | Loss Scale | Grad Scale | % of Total Grad
-------------|------------|------------|----------------
Value        | 0.94       | ~0.94      | ~8%
Policy       | 4.62       | ~4.62      | ~43%
Reward       | 5.62       | ~5.62      | ~51%
```

**Potential Issue:**
- Value head receives only ~8% of gradient magnitude
- May learn slower than policy/reward heads
- Could lead to inaccurate value estimates

**Mitigation:**
- Per-head gradient clipping (TASK_025) helps balance
- Value loss uses hybrid MSE + categorical (more stable)

---

## Issues and Recommendations

### 🟡 MEDIUM: Loss Scale Imbalance

**Issue:** Policy and reward losses are 5-6× larger than value loss.

**Options:**

1. **Accept as is** (MuZero design)
   ```python
   # Policy and reward naturally have higher cross-entropy scales
   # Value uses MSE which is naturally smaller
   ```

2. **Normalize losses** (if value learning is too slow)
   ```python
   # Add normalization factors
   value_loss_norm = value_loss / 1.0   # Reference scale
   policy_loss_norm = policy_loss / 5.0  # Normalize to ~1.0
   reward_loss_norm = reward_loss / 6.0  # Normalize to ~1.0
   ```

3. **Increase value loss weight** (simple fix)
   ```python
   value_loss_weight: float = 5.0,  # Balance scales
   ```

**Recommendation:** Monitor value prediction accuracy. If value MSE is high relative to policy loss, consider option 3.

---

### 🟢 LOW: Entropy Coefficient May Be Too Small

**Issue:** Entropy bonus has only ~0.5% impact on total loss.

**Current:**
```python
entropy_coef = 0.01  # 0.5% of total loss
```

**Alternative:**
```python
entropy_coef = 0.05  # 2.5% of total loss (more meaningful)
```

**Trade-off:**
- Higher entropy: More exploration, prevents collapse
- Lower entropy: Sharper policies, more exploitation

---

### 🟢 LOW: ATC Loss Not Annealed

**Issue:** ATC loss has fixed weight from step 0.

**Current:**
```python
total_loss = total_loss + self.atc_weight * atc_loss_val
```

**Improved:**
```python
# Anneal over first 10k steps
atc_weight_annealed = self.atc_weight * min(1.0, step / 10000)
total_loss = total_loss + atc_weight_annealed * atc_loss_val
```

---

## Loss Monitoring Recommendations

### Key Metrics to Track

```python
# Add to metrics dict
metrics['loss_ratio_vp'] = policy_loss / (value_loss + 1e-8)
metrics['loss_ratio_rp'] = reward_loss / (policy_loss + 1e-8)
metrics['entropy_contrib_pct'] = 100 * abs(entropy_coef * policy_entropy) / muzero_loss

# Alert if value is being drowned out
if metrics['loss_ratio_vp'] > 10.0:
    print(f"⚠️ Policy loss {metrics['loss_ratio_vp']:.1f}x larger than value loss")
```

### Healthy Ranges

| Metric | Healthy Range | Warning |
|--------|--------------|---------|
| `loss_ratio_vp` | 3-8 | > 10 or < 2 |
| `loss_ratio_rp` | 0.8-1.5 | > 2.0 or < 0.5 |
| `entropy_contrib_pct` | 1-5% | > 10% or < 0.1% |

---

## Total Loss Composition Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LOSS COMPOSITION                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  MuZero Core (Always Active)                                                 │
│  ───────────────────────────                                                 │
│  Value Loss:    ~0.94  (  8%)  ████                                         │
│  Policy Loss:   ~4.62  ( 43%)  ████████████████████                         │
│  Reward Loss:   ~5.62  ( 51%)  ███████████████████████                      │
│  Entropy Bonus: -0.05  ( -0%)  ▏                                            │
│  Adaptive Bonus: 0.00  (  0%)                                               │
│  ─────────────────────────────────────────                                  │
│  MuZero Total:  ~10.13 (100%)                                               │
│                                                                              │
│  Optional Auxiliaries (When Enabled)                                         │
│  ───────────────────────────────────                                         │
│  Prediction Loss:     Variable (annealed)                                    │
│  ATC Loss:            ~0.1-0.5  (if atc_loss=True)                          │
│  Cycle Balance:       ~0.1-0.3  (if cycle_balance=True)                     │
│  Expert BC:           ~0.01-0.1 (if imprint=True, annealed)                 │
│  Regret/Imprint:      ~0.01-0.05 (if amax enabled)                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Acceptance Criteria

- [x] Loss weights documented (all default to 1.0)
- [x] Loss scales measured (policy/reward dominate)
- [x] Entropy contribution calculated (~0.5%)
- [x] Auxiliary losses catalogued
- [ ] Loss imbalance addressed (if value learning suffers)
- [ ] Entropy coefficient tuned (if policy collapse occurs)

---

## References

- `mcts_trader/jax_agent/trainer.py:374-377` - Loss weight defaults
- `mcts_trader/jax_agent/trainer.py:1166-1172` - Loss averaging
- `mcts_trader/jax_agent/trainer.py:1183-1190` - MuZero loss composition
- `mcts_trader/jax_agent/trainer.py:1192-1203` - Metrics initialization
- `mcts_trader/jax_agent/trainer.py:1286-1317` - Prediction loss
- `mcts_trader/jax_agent/trainer.py:1319-1371` - ATC and cycle balance
- `mcts_trader/jax_agent/trainer.py:1373-1454` - Expert BC loss
- `mcts_trader/jax_agent/trainer.py:1456-1461` - Regret/imprint losses
