# TASK_149: Adaptive Class Balancing (Auto-Signal Booster)

**Status:** COMPLETE
**Completed:** 2026-02-10

**Objective:** Automatically balance the loss gradients so that the "Trade" signal and "Hold" signal have equal influence on the model's weights, preventing the "Majority HOLD" signal from drowning out the ideal path demonstrations.

## 1. The Root Cause (Gradient Drowning)
Currently, with `oracle_k = 1` and `trade_loss_weight = 5.0`, the gradient mass is severely skewed:
- **Trade Weight:** 1 asset * 5.0 = 5.0
- **Hold Weight:** 63 assets * 1.0 = 63.0
- **Signal Ratio:** ~7% Trade / 93% Hold.

The model is effectively being taught to ignore the trades because they are mathematically "unimportant" compared to the massive volume of HOLD labels.

## 2. Technical Implementation Fix

### A. Dynamic Weight Calculation
Modify the `loss_fn` in `easy_trainer.py` to calculate an adaptive weight per batch.

```python
# Inside easy_trainer.py -> _train_step_impl -> loss_fn

# 1. Identify trades vs holds
# Action type 1 is HOLD. targets_reshaped is (batch, 64, 5)
is_trade = targets_reshaped[:, :, 1] < 0.5
is_hold = targets_reshaped[:, :, 1] >= 0.5

# 2. Count occurrences in current batch
n_trades = jnp.maximum(jnp.sum(is_trade), 1.0)
n_holds = jnp.maximum(jnp.sum(is_hold), 1.0)

# 3. Calculate adaptive multiplier (e.g., 63.0 / 1.0 = 63.0)
# This ensures total_trade_weight == total_hold_weight
adaptive_trade_weight = n_holds / n_trades

# 4. Apply weights to per-asset cross entropy
loss_weights = jnp.where(is_trade, adaptive_trade_weight, 1.0)
policy_loss = jnp.sum(per_asset_ce * loss_weights) / jnp.sum(loss_weights)
```

## 3. Why this works
- **Zero Configuration:** No need to manually tune `trade_loss_weight` if you change `oracle_k`.
- **Force Convergence:** The model can no longer minimize loss by "getting lucky" on HOLDs. It is forced to solve the trade identification problem to achieve a low loss.
- **Honest Gradients:** Every update step now provides exactly 50% "How to Trade" knowledge and 50% "How to Wait" knowledge.

## 4. Expected Impact
- **Initial Loss:** Will jump to a higher starting point (more honest representation of task difficulty).
- **Metric Movement:** `agree_trade` should start climbing much faster.
- **Stability:** Removes the "Lazy Holder" local optimum entirely.

---

## Implementation Notes (2026-02-10)

### Files Modified
- `easy_trainer.py:157-170`: Replaced fixed `trade_loss_weight` with adaptive per-batch calculation
- `easy_config.py:44-46`: Deprecated `trade_loss_weight` field (kept for checkpoint compatibility)

### New Metric
Added `adaptive_trade_weight` to training metrics for monitoring. Typical values:
- With oracle_k=1: ~63.0 (63 holds per 1 trade)
- With oracle_k=5: ~12.0 (59 holds per 5 trades)

### Comparison to TASK_147
| Approach | Weight | Handles Variance | Auto-tunes |
|----------|--------|------------------|------------|
| TASK_147 | Fixed 60.0 | No | No |
| TASK_149 | n_holds/n_trades | Yes | Yes |

TASK_149 is strictly better: exact 50/50 balance per batch, auto-adjusts to any oracle_k.
