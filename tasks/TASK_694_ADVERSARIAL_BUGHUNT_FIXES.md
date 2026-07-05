# TASK_694: Fix Bugs from Adversarial Bug Hunt

**Status**: TODO
**Priority**: 1 (HIGH — PnL return misalignment corrupts training signal)
**Depends on**: None (all fixes are independent)

## Background

3 initial adversarial reviews + 6 parallel follow-up agents ran against the Easy Mode
training pipeline (`prompt-lib/adversarial_bughunt_easy_mode.md`).
Reports in `prompt-lib/out_ad0[1-3]/`.

Arbiter validation cross-referenced all findings against actual code. After dedup
and validation: **9 confirmed bugs**, **3 confirmed benchmark-only metric bugs**,
**1 latent issue**, **1 dormant issue**.

---

## TRAINING-AFFECTING BUGS

### BUG-1: PnL Return Timestep Misalignment (HIGH)

**Found by**: R1 initial hunt (reconfirmed by 2 follow-up agents)
**Class**: Train-Inference Distribution Mismatch

`_build_model_state_trajectory` (train_easy.py:279) computes PnL from the wrong return:

```python
ret = returns[idx]   # idx = 0, 1, ..., n_valid-1
```

Window `idx` covers prices `[idx, idx+1, ..., idx+63]`. The action applies at `idx+63`.
PnL should use `returns[idx+63]`, not `returns[idx]`.

Benchmark correctly uses `returns[63 + i]` (benchmark_easy.py:410,432,544).

Both code paths affected:
- Single-episode: `_generate_episode_data()` line 572 passes full `log_returns`
- Batch: `_generate_batch_episodes()` line 784 passes full `log_returns_i`

**Fix** — Align returns before passing to rollout:
```python
# train_easy.py:570 (single-episode path)
aligned_returns = log_returns[self.config.window_size - 1:]
state_channels, confidence = self._model_state_rollout(
    observations, aligned_returns, ...
)

# train_easy.py:784 (batch path)
ws = self.config.window_size
log_returns_i = log_returns[i, ws - 1:valid_lengths[i] - 1]
```

---

### BUG-2: Confidence Channel 1-Step Temporal Lag (MEDIUM)

**Found by**: R1 (lag analysis) + R3 (zero-input framing)

Two sub-issues in confidence channel (ch33):

**A — Zero input during rollout**: `_build_model_state_trajectory` (line 242) feeds zero
confidence. Inference (benchmark_easy.py:506) feeds previous step's confidence.

**B — Temporal lag in assembly**: Training window w gets confidence from step w's output
(line 624). Inference window i gets confidence from step i-1.

**Fix**: Carry prev_confidence through scan + shift by 1 in assembly.

---

### BUG-4: TIES Merge Trims 80% Instead of 20% (HIGH)

**Found by**: EXT-2 (bugs++ agents)
**Validated**: CONFIRMED

`tools/orchestration/merge_utils.py:176-178`:
```python
threshold = jnp.percentile(jnp.abs(d), (1 - trim_fraction) * 100)  # 80th percentile
return jnp.abs(d) >= threshold  # Keeps only top 20%
```

With `trim_fraction=0.2`, TIES paper says trim bottom 20% (keep top 80%). Code does
the inverse — keeps only top 20%, zeros 80% of learned updates.

**Fix** — One line:
```python
threshold = jnp.percentile(jnp.abs(d), trim_fraction * 100)  # 20th percentile
```

---

### BUG-5: Val Generator Never Gets Model Params Updated (HIGH)

**Found by**: EXT-4 (bugs++ agents)
**Validated**: CONFIRMED

`val_generator.set_model_params()` is called once at startup (line 1531) but NEVER
during the training loop. Training generator gets fresh params each epoch, but val
generator uses epoch-0 model for state rollout throughout training.

Validation metrics become increasingly misleading as model diverges from epoch-0 params.

**Fix** — Add before validation call (around line 1581):
```python
val_generator.set_model_params(trainer.model, trainer.params)
val_generator._current_epoch = epoch
val_metrics = validate(trainer, val_generator, config, n_batches=20)
```

---

### BUG-6: best_agreement Resets to 0 After Merge (MEDIUM)

**Found by**: DLOSS-21 (data_loss agents)
**Validated**: CONFIRMED

Merged checkpoints have `metrics = {'source': 'merged', 'method': ...}` — no
`oracle_agreement` key. On load (train_easy.py:1541):
```python
best_agreement = checkpoint['metrics'].get('oracle_agreement', 0.0)
```
Returns 0.0 → immediately overwrites `phase1_best.pkl` with potentially worse weights.

**Fix** — Use branch agreements from merged checkpoint:
```python
best_agreement = max(
    checkpoint['metrics'].get('oracle_agreement', 0.0),
    checkpoint['metrics'].get('a_agreement', 0.0),
    checkpoint['metrics'].get('b_agreement', 0.0),
)
```

---

### BUG-7: Trade-Biased Sampling Boundary Pile-Up (MEDIUM)

**Found by**: EXT-11 (bugs++ agents)
**Validated**: CONFIRMED

`train_easy.py:1126`:
```python
idx = self.rng.choice(trade_indices)
offset = int(np.clip(idx, 0, max_offset))
```

Trade indices near episode end (idx > max_offset) all clamp to `max_offset`, creating
systematic oversampling of the last valid window. With momentum-following oracles,
most trade indices are near episode end → 30% of biased samples pile up at same offset.

**Fix** — Filter before sampling:
```python
valid_trade = trade_indices[trade_indices <= max_offset]
if len(valid_trade) > 0:
    offset = int(self.rng.choice(valid_trade))
```

---

## BENCHMARK-ONLY BUGS (don't affect training)

### BUG-3: Benchmark Value MSE Wrong Reference (LOW)

**Found by**: R1 initial hunt

benchmark_easy.py:636-639 compares model values against cumulative PnL. Model trained
on remaining PnL. Opposite monotonicity — metric is meaningless.

**Fix**: `oracle_remaining = oracle_cum_pnl[-1] - oracle_cum_pnl[action_return_indices]`

---

### BUG-8: Benchmark P&L Simulation Ignores Limit Orders (MEDIUM)

**Found by**: EXT-9 (bugs++ agents)
**Validated**: CONFIRMED

`benchmark_easy.py:240-257` (simulate_model_pnl_detailed) only checks Buy(2) and
Sell(0). Ignores LimitBuy(4) and LimitSell(3). State tracking (line 530-531) correctly
handles all 5 actions. If model outputs limit orders, P&L diverges from state.

**Fix**: Add `or actions[t,i] == 4` to Buy branch, `or actions[t,i] == 3` to Sell.

---

### BUG-9: Oracle vs Model P&L Normalization Mismatch (MEDIUM)

**Found by**: EXT-10 (bugs++ agents)
**Validated**: CONFIRMED

Oracle P&L divides by `k_count` (per-position). Model P&L is raw sum. A model holding
4 positions has ~4x raw P&L vs oracle's normalized P&L for identical exposure.
`pnl_capture_ratio` can exceed 1.0 even when model captures less per position.

**Fix**: Normalize both consistently (both raw or both per-position).

---

### BUG-10: Benchmark Timezone Shift (HIGH)

**Found by**: DLOSS-15 (data_loss agents)
**Validated**: CONFIRMED

`mcts_trader/data/history.py:55,67`:
```python
datetime.strptime(start_date, '%Y-%m-%d').timestamp()
```
Naive datetime → uses local timezone (CET, UTC+1). Kraken timestamps are UTC.
Every benchmark period shifted 1 hour earlier.

**Fix**: `.replace(tzinfo=timezone.utc).timestamp()`

---

### BUG-11: Alpha Metric Compares Log vs Simple Returns (MEDIUM)

**Found by**: EXT-14 (bugs++ agents)
**Validated**: PARTIALLY CORRECT

`model_pnl` is cumulative sum of log returns × positions. `buy_hold_return` is simple
return `(end/start - 1)`. `model_alpha = model_pnl - buy_hold_return` is
apples-to-oranges.

---

## LATENT / DORMANT ISSUES

### LATENT: Buffer Param IDs Desync (NO-CRASH, LOW)

**Found by**: R1 initial hunt + reconfirmed by follow-up agents

`_partial_refresh_buffer()` (line 1071-1073) deletes from `_buffer_param_ids` but not
from buffer. Causes index misalignment. Doesn't crash because `_refill_buffer` only
runs when ALL entries are exhausted (keep_indices = []).

**Fix** (optional): Set sentinel `-1` instead of `del`.

---

### DORMANT: V3 Encoder Broadcasts Future Info (CRITICAL if V3 enabled)

**Found by**: EXT-1 (bugs++ agents)
**Validated**: CONFIRMED architecture issue, but **`encoder_version=2` (V2) is active**.

Channels 5-26 in temporal encoder compute endpoint scalars broadcast across all 64
timesteps. Channels 0-4 use full-window min/max normalization. When V3 minGRU processes
these sequentially, position t=0 sees information from t=63.

V2 (current) flattens temporal dim — no causal violation. Only matters if V3 is enabled.

**Fix needed before enabling V3**: Feed only channels 0-4 to minGRU, inject 5-26 after
scan. Use expanding-window normalization for channels 0-4.

---

## DISMISSED FINDINGS

| Finding | Source | Why Dismissed |
|---------|--------|---------------|
| LR warmup restarts after merge (EXT-3/DLOSS-20) | bugs++, data_loss | **FALSE POSITIVE** — `rebuild_optimizer_state` correctly preserves `adam_state.count` and `schedule_state.count` |
| Padded zeros → -inf returns (EXT-6/DLOSS-5) | bugs++, data_loss | **FALSE POSITIVE** — `valid_lengths` masking removes padding before any downstream use |
| heat_slow permanently saturated (DLOSS-6) | data_loss | **FALSE POSITIVE** — drops to 0.99 after 1 no-trade step (reviewer's math wrong: 0.99^1 = 0.99 < 1.0) |
| Zero-return → SHORT (EXT-5/EXT-7) | bugs++ | **CONFIRMED but HARMLESS** — exact zero returns virtually never occur in continuous price data |
| Batch assembly drops samples (ad02-BUG-2) | out_ad02 | **POSSIBLE but SAFE** — `[:batch_size]` is a safety clip; logic produces exact batch_size |
| Confidence oracle-vs-model (R2 BUG-1) | out_ad02 initial | Only applies to disabled fallback path (`autoregressive_state=False`) |
| Stale latent cache (EXT-13) | bugs++ | Dormant — e-dojo uses `--no-latent-cache` |
| Value bin truncation (ad02-BUG-1/EXT-8) | out_ad02, bugs++ | CONFIRMED but max error 0.025 in [-10,10] support — negligible |
| Volume ratio constant 0.2 (EXT-15/DLOSS-10) | bugs++, data_loss | Design issue — `mean(vol) / mean(vol) = 1.0`. Wasted channel, not a bug. |
| Encoder clip ranges (DLOSS-7,8) | data_loss | Design choices for normalization ranges, not correctness bugs |
| Bella shorter episodes (DLOSS-2) | data_loss | Design issue — different sources have different lengths |

## Completion Criteria

**Training fixes (do before next run):**
- [ ] BUG-1: PnL return alignment (both code paths)
- [ ] BUG-2: Confidence carry through scan + shift in assembly
- [ ] BUG-4: TIES merge percentile fix (one line)
- [ ] BUG-5: Val generator `set_model_params` in training loop
- [ ] BUG-6: best_agreement from merged checkpoint
- [ ] BUG-7: Trade-biased sampling filter before sample

**Benchmark fixes (before next benchmark):**
- [ ] BUG-3: Benchmark value_mse uses remaining PnL
- [ ] BUG-8: P&L simulation handles limit orders
- [ ] BUG-9: Normalize oracle/model P&L consistently
- [ ] BUG-10: UTC timestamps in history.py
- [ ] BUG-11: Alpha metric consistent return type

**Optional:**
- [ ] LATENT: Buffer param_ids sentinel
- [ ] DORMANT: V3 encoder fix (before enabling V3)

---

*Created: 2026-02-22*
*Updated: 2026-02-22 (added bugs++/data_loss agent findings)*
*Source: 3 initial reviews + 6 follow-up agents (prompt-lib/out_ad0[1-3]) + arbiter validation*
