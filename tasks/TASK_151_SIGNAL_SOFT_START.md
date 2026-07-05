# TASK_151: Signal Soft-Start (Warmup Phase)

**Status:** COMPLETE
**Completed:** 2026-02-10

**Objective:** Implement a linear ramp for trade signal weights to prevent gradient explosion and weight fracturing during the initial "random weight" phase of training.

## 1. The Problem (Signal Shock)
With TASK_149, the trade signal mass jumps from ~7% to 50%. On a fresh model with random weights, this creates massive, noisy gradients that can lead to:
- Numerical instability (NaNs).
- Destructive weight updates that "break" the encoder early on.
- Immediate divergence between branches before they learn anything.

## 2. Technical Implementation Fix

### A. Linear Weight Ramping
Modify `easy_trainer.py` to smoothly increase the `adaptive_trade_weight` over the first 50 epochs.

```python
# easy_trainer.py - Inside loss_fn

# 1. Calculate Target Weight (Mass Equality)
target_trade_weight = n_holds / n_trades

# 2. Calculate Warmup Progress (0.0 -> 1.0)
# Note: current_epoch must be passed from train_easy.py
warmup_epochs = getattr(self.config, 'weight_warmup_epochs', 50)
progress = jnp.clip(current_epoch / warmup_epochs, 0.0, 1.0)

# 3. Smoothly Ramp Weight
# Starts at 1.0 (unweighted) and reaches full adaptive weight at end of warmup
active_trade_weight = 1.0 + (target_trade_weight - 1.0) * progress

# 4. Apply Balanced Policy Loss
loss_weights = jnp.where(is_trade, active_trade_weight, 1.0)
policy_loss = jnp.sum(per_asset_ce * loss_weights) / jnp.sum(loss_weights)
```

### B. Configuration Changes
Add the warmup duration to the config.

```python
# easy_config.py
weight_warmup_epochs: int = 50  # Number of epochs to reach full signal mass
```

### C. Training Loop Update
Ensure the current epoch is passed into the trainer's step function.

```python
# train_easy.py - In train_epoch()
# metrics = trainer.train_step(batch, epoch=epoch)
```

## 3. Why this works
- **Gradient Stability:** On Day 1 (random weights), the model behaves like a standard unweighted learner.
- **Eased Hardening:** As the model learns basic market physics, the "Trade" signal becomes louder, forcing it to specialize without the "Signal Shock."
- **Branch Alignment:** Slower weight ramping reduces the speed of divergence, making the first "Basin Synchronization" merge much more likely to succeed.

## 4. Expected Impact
- **Initial Training:** Agreement will start lower (~20% random per-asset) and climb slower but more smoothly.
- **Merge Similarity:** Cosine similarity at the first merge (Epoch 30) should be significantly higher (> 0.80) than the previous attempt (0.36).

---

## Implementation Notes (2026-02-10)

### Files Modified
- `easy_config.py:47-48`: Added `weight_warmup_epochs: int = 50`
- `easy_trainer.py:165-170`: Warmup ramping logic
- `train_easy.py:745-752`: Pass warmup_progress in batch

### Warmup Formula
```python
warmup_progress = min(epoch / warmup_epochs, 1.0)
active_weight = 1.0 + (target_weight - 1.0) * warmup_progress
```

| Epoch | Progress | Weight (if target=63) |
|-------|----------|----------------------|
| 0 | 0.0 | 1.0 (unweighted) |
| 25 | 0.5 | 32.0 |
| 50+ | 1.0 | 63.0 (full) |

### New Metrics
- `warmup_progress`: 0.0 → 1.0 over warmup epochs
- `adaptive_trade_weight`: Now includes ramping (not just raw n_holds/n_trades)
