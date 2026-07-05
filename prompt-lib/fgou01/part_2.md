# Part 2: Oracle Math Consistency

## Status: NEEDS_FIX

## Summary

There is a confirmed normalization mismatch between the training oracle and both benchmark PnL simulators. The training oracle divides step PnL by `k_count` (the number of selected assets), producing per-position-average returns. The benchmark oracle and model simulators sum raw returns without normalization, producing aggregate returns that scale with position count. This makes benchmark PnL numbers uninterpretable and inflates them by a factor of ~k.

Additionally, a secondary mismatch exists: the benchmark **value MSE** calculation compares model value predictions (trained against normalized `remaining_pnl` from `greedy_lookahead_path`) against `oracle_cum_pnl` that is already normalized -- so the value MSE is actually internally consistent, but the *context* in the benchmark report is misleading because the P&L column next to it is unnormalized.

## Findings

### 1. Training oracle normalizes by k_count (CONFIRMED)

`mcts_trader/jax_agent/path_solver.py:130-131`:
```python
k_count = jnp.maximum(jnp.sum(new_mask), 1.0)  # Number of selected assets
step_pnl = jnp.sum(jnp.abs(step_returns) * new_mask) / k_count
```

This means the training oracle treats the portfolio as equal-weighted: if k=4 assets are selected, each contributes 1/4 of the total PnL. The `cumulative_pnl` and `total_alpha` returned by `greedy_lookahead_path` are on this normalized scale.

The same normalization is present in the other two path solver methods:
- `soft_topk_path` at line 461: `step_pnl = jnp.sum(abs_returns * selection) / k`
- `sinkhorn_topk_path` at line 581: `step_pnl = jnp.sum(abs_returns * selection) / k`

### 2. Benchmark oracle does NOT normalize (CONFIRMED)

`tools/benchmark_oracle_vs_model.py:399-403`:
```python
k_count = max(np.sum(mask), 1)

# BUG-9 fix: Use raw sum (no /k_count) to match model P&L normalization
step_pnl = np.sum(np.abs(returns[t]) * mask)
```

The `k_count` is computed but deliberately not used. The comment says "BUG-9 fix" -- but this was a misguided fix. The intent was to "match model P&L normalization," but the model P&L is also unnormalized, so both are wrong. The real fix should have been to normalize both, not to denormalize both.

### 3. Benchmark model does NOT normalize (CONFIRMED)

`tools/benchmark_oracle_vs_model.py:468`:
```python
step_pnl += np.sum(position * returns[t])
```

Raw sum of `position * returns` across all positioned assets. If the model holds n positions, PnL scales linearly with n. No normalization by active position count.

### 4. `benchmark_easy.py` inherits the same bugs

`tools/benchmark_easy.py:52-53` imports `simulate_oracle_pnl` and `simulate_model_pnl` from `benchmark_oracle_vs_model.py`, so the same unnormalized calculations propagate to that benchmark too.

`tools/benchmark_easy.py:260-305` has `simulate_model_pnl_detailed` which also does not normalize:
```python
step_pnl += np.sum(position * returns[t])  # line 301
```

`tools/benchmark_easy.py:312-349` has `_simulate_pnl_vectorized` for random baseline, also unnormalized:
```python
total_pnl += np.sum(position * returns[t]) - n_trades * transaction_cost  # line 347
```

### 5. Value MSE is internally consistent but contextually misleading

`tools/benchmark_easy.py:696`:
```python
oracle_remaining = oracle_cum_pnl[-1] - oracle_cum_pnl[action_return_indices]
value_mse = float(np.mean((model_values - oracle_remaining * config.value_scale) ** 2))
```

`oracle_cum_pnl` comes from `greedy_lookahead_path` (line 461-470), which IS normalized by k_count. The model was trained against this same normalized cumulative PnL (via `remaining_pnl` in `train_easy.py:690`). So value MSE is comparing the right things -- but it appears next to P&L numbers that are on a completely different scale, making the report confusing.

### 6. What is k_count?

`k_count = jnp.maximum(jnp.sum(new_mask), 1.0)` at `path_solver.py:130`.

This is the number of assets selected in the binary mask `new_mask` for that timestep. It ranges from 0 to k (clamped to minimum 1.0 to avoid division by zero). With the `min_alpha_threshold` feature (TASK_141), the oracle can select fewer than k assets when no asset has sufficient expected return, so `k_count` varies per timestep. Without threshold, `k_count == k` at every step.

### 7. Should benchmark normalize or should training NOT normalize?

**Benchmark should normalize to match training.** Rationale:

- The `/k_count` normalization in training is semantically correct: it represents per-position average return, which is what an equal-weighted portfolio achieves. Without it, holding more positions would always look better (leverage effect), which is the "4x leverage bug" that TASK_133 originally fixed.
- The value head is trained against normalized PnL. If we denormalize training, we break value prediction and need to retrain.
- The model PnL also needs normalization to be comparable: `position * returns / n_active` gives per-position average return, matching the oracle's scale.
- Without normalization, benchmark reports numbers like "+8,128% oracle" which are meaningless.

### 8. The fundamental semantic difference

The oracle uses `abs(returns) * mask` (captures magnitude regardless of direction). The model uses `position * returns` (captures directional PnL: long profits from positive returns, short from negative). These are fundamentally different PnL calculations:

- Oracle: always positive (it captures the absolute move)
- Model: can be positive or negative (depends on whether position direction matches return sign)

Even after normalizing both, they measure different things. The oracle is an upper bound on capturable alpha (perfect direction + perfect timing). The model PnL is the actual achieved PnL. Comparing them as "capture ratio" is valid only if you understand the oracle is unreachable (it uses hindsight).

## Bugs Found

- **BUG-2.1**: Benchmark oracle PnL is not normalized by k_count, but training oracle IS. Severity: **HIGH** (makes benchmark PnL numbers meaningless; inflated by factor of ~k). File: `tools/benchmark_oracle_vs_model.py:403`.

- **BUG-2.2**: Benchmark model PnL is not normalized by active position count. Severity: **HIGH** (model PnL scales with number of positions held, not comparable to oracle). File: `tools/benchmark_oracle_vs_model.py:468`.

- **BUG-2.3**: `benchmark_easy.py:simulate_model_pnl_detailed` (line 301) and `_simulate_pnl_vectorized` (line 347) also lack normalization. Severity: **HIGH** (same bug propagated to the main benchmarking tool). Files: `tools/benchmark_easy.py:301`, `tools/benchmark_easy.py:347`.

- **BUG-2.4**: The "BUG-9 fix" comment at `benchmark_oracle_vs_model.py:402` is misleading. It claims removing `/k_count` was intentional to match model normalization, but the correct fix should have been to add normalization to the model PnL instead. Severity: **LOW** (documentation/comment issue, but could lead future developers astray).

## Recommended Fixes

### Fix BUG-2.1: Normalize benchmark oracle PnL

File: `tools/benchmark_oracle_vs_model.py`, function `simulate_oracle_pnl`, line 399-403:

```python
# Before (BUG):
k_count = max(np.sum(mask), 1)
# BUG-9 fix: Use raw sum (no /k_count) to match model P&L normalization
step_pnl = np.sum(np.abs(returns[t]) * mask)

# After (FIXED - match training oracle path_solver.py:131):
k_count = max(np.sum(mask), 1)
step_pnl = np.sum(np.abs(returns[t]) * mask) / k_count
```

### Fix BUG-2.2: Normalize benchmark model PnL

File: `tools/benchmark_oracle_vs_model.py`, function `simulate_model_pnl`, line 468:

```python
# Before (BUG):
step_pnl += np.sum(position * returns[t])

# After (FIXED - normalize by number of active positions):
n_active = max(np.sum(np.abs(position) > 0.5), 1)
step_pnl += np.sum(position * returns[t]) / n_active
```

### Fix BUG-2.3a: Normalize benchmark_easy model PnL detailed

File: `tools/benchmark_easy.py`, function `simulate_model_pnl_detailed`, line 301:

```python
# Before (BUG):
step_pnl += np.sum(position * returns[t])

# After (FIXED):
n_active = max(np.sum(np.abs(position) > 0.5), 1)
step_pnl += np.sum(position * returns[t]) / n_active
```

### Fix BUG-2.3b: Normalize benchmark_easy random baseline

File: `tools/benchmark_easy.py`, function `_simulate_pnl_vectorized`, line 347:

```python
# Before (BUG):
total_pnl += np.sum(position * returns[t]) - n_trades * transaction_cost

# After (FIXED):
n_active = max(np.sum(np.abs(position) > 0.5), 1)
total_pnl += np.sum(position * returns[t]) / n_active - n_trades * transaction_cost
```

### Fix BUG-2.4: Update misleading comment

File: `tools/benchmark_oracle_vs_model.py`, line 402:

```python
# Before:
# BUG-9 fix: Use raw sum (no /k_count) to match model P&L normalization

# After (once BUG-2.1 is applied):
# Normalize by k_count to match training oracle (path_solver.py:131)
```

## Impact Assessment

- **Training correctness**: NOT affected. The training oracle in `path_solver.py` is correct and has been correct since TASK_133. Value targets derived from `cumulative_pnl` (normalized) are correct. No retraining needed.
- **Benchmark reporting**: SEVERELY affected. All benchmark PnL numbers, alpha calculations, and capture ratios are inflated/distorted. After fixing, expect oracle PnL to drop by ~k (e.g., from 8,128% to ~2,032% if k=4, or even lower if k=1).
- **Value MSE**: Indirectly affected. The value MSE itself is computed correctly (both sides use normalized PnL), but the benchmark report context makes it look wrong because the P&L column next to it is on a different scale.

## Cross-reference

This issue is already documented in `tasks/TASK_700_BENCHMARK_PNL_NORMALIZATION.md` (status: OPEN). The analysis here confirms the TASK_700 diagnosis is correct and provides the additional finding about `benchmark_easy.py` propagation.
