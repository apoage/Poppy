# TASK_709: Fix Benchmark Value MSE — Horizon Mismatch + Scale Bug

**Status**: TODO
**Priority**: 1 (benchmark MSE is meaningless — reads ~1160, real value is unknown)
**Depends on**: TASK_708 (done — AR feedback needs correct MSE signal long-term)

---

## Problem

Benchmark value MSE is frozen at ~1160 across all checkpoints (E4399→E5099) while training val MSE is 0.05-11.0. Two bugs compound:

### Bug 1: Horizon Mismatch (critical)

Training computes remaining PnL over **episode_length (~400 steps)**:
```python
# train_easy.py:716
remaining_pnl = cumulative_pnl[-1] - cumulative_pnl  # episode is ~400 steps
```

Benchmark computes remaining PnL over **full 10,000-step trajectory**:
```python
# benchmark_easy.py:688
oracle_remaining = oracle_cum_pnl[-1] - oracle_cum_pnl[action_return_indices]
```

At step 0, benchmark remaining PnL = total oracle PnL (~2.65 for bull). At step 9900, remaining PnL ≈ 0. The value head was trained on 400-step horizons where remaining PnL is tiny (0.01-0.5). Asking it to predict 10K-step remaining PnL is meaningless — guaranteed huge MSE.

### Bug 2: Scale Mismatch (already fixed mid-session)

Training descales model output before MSE:
```python
# easy_trainer.py:264
value_pred = self._value_to_scalar(value_logits) / self.config.value_scale  # descale
value_mse = jnp.mean((value_pred - batch['returns']) ** 2)
```

Benchmark was scaling oracle up instead:
```python
# benchmark_easy.py:689 (OLD — already fixed)
value_mse = float(np.mean((model_values - oracle_remaining * config.value_scale) ** 2))
```

**Fix applied**: Changed to `model_values / config.value_scale - oracle_remaining`. This is correct but the horizon mismatch still dominates.

---

## Design

### Fix: Windowed Remaining PnL

Benchmark should compute remaining PnL over the same horizon the value head was trained on:

```python
ep_len = getattr(config, 'episode_length', 400)
# For each window i: remaining = cum_pnl[i + ep_len] - cum_pnl[i]
# Cap at trajectory end for windows near the end
end_indices = np.minimum(action_return_indices + ep_len, len(oracle_cum_pnl) - 1)
oracle_remaining = oracle_cum_pnl[end_indices] - oracle_cum_pnl[action_return_indices]
```

This gives a fair comparison: model predicts remaining PnL over next ~400 steps, benchmark measures against oracle remaining PnL over the same ~400 steps.

### Edge Case: Tail Windows

Windows within `ep_len` of trajectory end will have shorter horizons. Two options:
- **A) Cap at end** (simpler): `min(i + ep_len, T-1)` — tail windows have smaller remaining PnL naturally
- **B) Exclude tail**: Only compute MSE for windows where `i + ep_len < T` — cleaner but fewer samples

Recommend A — the bias is small and it preserves all windows.

---

## Implementation

### In `tools/benchmark_easy.py`

**Replace** (around line 686-690):
```python
# OLD
oracle_remaining = oracle_cum_pnl[-1] - oracle_cum_pnl[action_return_indices]
value_mse = float(np.mean((model_values / config.value_scale - oracle_remaining) ** 2))

# NEW
ep_len = getattr(config, 'episode_length', 400)
end_indices = np.minimum(action_return_indices + ep_len, len(oracle_cum_pnl) - 1)
oracle_remaining = oracle_cum_pnl[end_indices] - oracle_cum_pnl[action_return_indices]
value_mse = float(np.mean((model_values / config.value_scale - oracle_remaining) ** 2))
```

---

## Files

| File | Change |
|------|--------|
| `tools/benchmark_easy.py` | Windowed remaining PnL (ep_len horizon), scale fix already applied |

---

## Acceptance Criteria

- [ ] Benchmark value MSE uses windowed remaining PnL (episode_length horizon, not full trajectory)
- [ ] Model values descaled before comparison (already done)
- [ ] Benchmark MSE is comparable in scale to training val MSE (0.x - 10.x range, not 1000+)
- [ ] Tail windows handled (capped at trajectory end)
- [ ] Existing benchmark JSON format unchanged (just value_mse becomes meaningful)

---

## Expected Impact

- Benchmark MSE drops from ~1160 to something in the 0.x-10.x range (comparable to training)
- Can now track value head calibration across checkpoints meaningfully
- TASK_708's MSE→AR feedback becomes more informative when benchmark confirms training MSE trends
- Enables future work: if benchmark windowed MSE is high but training MSE is low → regime overfitting confirmed quantitatively
