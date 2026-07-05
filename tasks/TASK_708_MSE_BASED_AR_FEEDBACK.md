# TASK_708: MSE-Based Autoregressive Blend Feedback

**Status**: DONE
**Priority**: 1 (model stuck at 0% trade rate with AR=1.0 and value MSE ~1160)
**Depends on**: TASK_707 (done — period cycling active)

---

## Problem

With AR blend at 100%, the model feeds its own value predictions back as state input. But the value head is catastrophically miscalibrated (benchmark MSE ~1160, target <0.1). This creates a vicious cycle:

1. Bad value predictions → bad state channels fed to model
2. Bad state input → model can't learn to trade → HOLD everything
3. 0% trade rate → no learning signal from actual trades

Training agreement is ~82% but the model is pure HOLD. Benchmark shows 0.0% trade rate across all 6 periods.

---

## Design

**Reactive MSE→AR feedback**: Use EMA-smoothed validation MSE to dynamically set `ar_max_blend`. When value head is uncalibrated, give more oracle state guidance. As it calibrates, let the model earn self-prediction rights.

### Mapping (log-scale interpolation)

| Val MSE (EMA) | AR blend | Meaning |
|---------------|----------|---------|
| ≤ 0.1 | 1.0 | Value head calibrated — full self-play |
| 1.0 | 0.6 | Getting there — mostly self-play |
| 10.0 | 0.2 | Uncalibrated — heavy oracle guidance |
| ≥ 10.0 | 0.2 (floor) | Floor — always some self-play for learning |

```python
# Log-space linear interpolation
log_good = log10(0.1)   # -1.0
log_bad = log10(10.0)   # 1.0
log_mse = log10(clamp(mse_ema, 1e-6, ...))
t = (log_bad - log_mse) / (log_bad - log_good)  # 1.0=good, 0.0=bad
ar_blend = AR_FLOOR + (1.0 - AR_FLOOR) * clamp(t, 0, 1)
```

### EMA Smoothing

Val MSE oscillates wildly (0.05 → 11.0 → 0.05) due to period cycling — some periods have price gaps that spike MSE. Use exponential moving average:

```python
mse_ema = alpha * val_mse + (1 - alpha) * mse_ema  # alpha = 0.1 (~10-epoch window)
```

Only update EMA with finite MSE values (skip NaN/inf from bad periods).

### NaN Protection

Skip MSE→AR update when `val_mse` is NaN or inf (already handled by `math.isfinite()` check).

---

## Implementation

### In `train_easy.py` epoch loop

**Before the training loop** (after checkpoint loading):
```python
import math
mse_ema = 1.0  # Start assuming bad calibration
mse_ema_alpha = 0.1
AR_FLOOR = 0.2
AR_MSE_GOOD = 0.1
AR_MSE_BAD = 10.0
```

**After validation** (after `val_metrics` is computed):
```python
val_mse = val_metrics.get('value_mse', None)
if val_mse is not None and autoregressive_state and math.isfinite(val_mse):
    mse_ema = mse_ema_alpha * val_mse + (1 - mse_ema_alpha) * mse_ema
    log_good = math.log10(AR_MSE_GOOD)
    log_bad = math.log10(AR_MSE_BAD)
    log_mse = math.log10(max(mse_ema, 1e-6))
    t = (log_bad - log_mse) / (log_bad - log_good)
    new_ar = AR_FLOOR + (1.0 - AR_FLOOR) * max(0.0, min(1.0, t))
    config.ar_max_blend = new_ar
```

**Logging**: Add AR blend and MSE EMA to epoch log line and CSV:
```
Epoch 5100 | Loss: 1.92 | Train: 82% | Val: 81% | MSE: 0.05 | AR: 60% | Time: 210s
```

### CSV columns to add

- `ar_blend`: Current AR max blend value
- `mse_ema`: Smoothed MSE used for AR calculation
- `period_label`: Which period this epoch trained on (for regime analysis)

---

## Files

| File | Change |
|------|--------|
| `train_easy.py` | EMA state init, MSE→AR update after validation, log AR in epoch line + CSV |

---

## Acceptance Criteria

- [ ] AR blend dynamically adjusts based on EMA of val MSE
- [ ] MSE ≤ 0.1 → AR = 1.0 (full self-play)
- [ ] MSE ≥ 10.0 → AR = 0.2 (heavy oracle guidance)
- [ ] EMA smoothing prevents epoch-to-epoch AR whiplash
- [ ] NaN/inf MSE values are skipped (no AR update)
- [ ] AR blend logged in epoch output and CSV
- [ ] No JIT recompilation — `ar_max_blend` is read per-episode, not a JIT static arg
- [ ] Model starts trading (trade rate > 0%) as value head calibrates

---

## Expected Behavior

With current MSE oscillating between 0.05 and 11.0:
- EMA will stabilize around ~1-5 range initially
- AR will drop to ~0.4-0.6, giving model 40-60% oracle state guidance
- Value head gets cleaner signal → MSE drops → AR rises
- Eventually MSE < 0.1 → full AR → model self-sustaining

This breaks the vicious cycle: oracle guidance → better value head → better state predictions → model starts making trade decisions instead of pure HOLD.
