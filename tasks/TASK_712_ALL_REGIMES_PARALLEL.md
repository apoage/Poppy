# TASK_712: All Regimes as Parallel VEnvs

**Status**: DONE
**Priority**: 1 (value MSE oscillation is blocking benchmark progress)
**Depends on**: TASK_710 (SegmentPool)

---

## Problem

Value MSE oscillates wildly per regime (0.05↔11.0) because the model only sees 6 of 18 date ranges at a time. When a slot rotates to a new regime, the value distribution shifts and MSE spikes. The MSE EMA (TASK_708) smooths this, but the value head never gets stable gradients across all regimes simultaneously.

This is the root cause of benchmark MSE ~1160: the value head is calibrated to whichever 6 regimes are active, then miscalibrated on the other 12.

## Solution

**All 18 date ranges active simultaneously.** Every training batch contains windows from all market regimes — bull, bear, crash, sideways, etc. The value head sees the full return distribution every step.

### Changes to SegmentPool

1. **`n_slots=18`** — one slot per date range, all active at once
   - RAM: 18 × ~300MB = ~5.4GB (12% of 45GB — fine)
   - Every round-robin cycle of 18 `get_window()` calls covers all regimes

2. **`max_runs_per_group=3`** — cap asset diversity per date range
   - 270→48 segments total (early ranges have 1 run, recent have up to 3)
   - Keeps some asset diversity without 5.6x slowdown
   - Each slot exhausts its 1-3 runs then the group is done

3. **Val pool: `n_slots=6, max_runs_per_group=1`**
   - 6 regimes active for validation (subset of 18)
   - 1 run per range — val doesn't need asset diversity

### Changes to main()

- When `--periods-file` is used:
  - Train pool: `n_slots=18, max_runs_per_group=3`
  - Val pool: `n_slots=6, max_runs_per_group=1` (reversed order for different regime subset)

### No changes needed

- `_refill_buffer`: already uses `segment_pool.get_window()` round-robin
- `train_epoch`: pool handles rotation internally
- `_generate_batch_episodes`: regime mixing happens naturally through buffer
- Checkpoints: no model changes, fully compatible

---

## Numbers

| Metric | Before (TASK_710) | After (TASK_712) |
|--------|-------------------|------------------|
| Active regimes | 6 of 18 | **18 of 18** |
| Segments | 270 | **48** |
| Total windows | ~285k | ~50k |
| RAM (train pool) | ~1.9GB | **~5.4GB** |
| Time to exhaust | ~8.7 days | **~37 hours** |
| Value MSE per batch | Single-regime | **All-regime mix** |

## Expected Impact

- Value MSE stabilization: no more 0.05↔11.0 oscillation per epoch
- TASK_708 AR feedback should settle higher (MSE EMA converges to true average)
- Benchmark MSE should drop (value head sees all regimes during training)
- 5.6x faster to complete one data pass
