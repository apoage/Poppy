# TASK_700: Fix Benchmark PnL Normalization

## Status: DONE (patched in TASK_701)
## Priority: High
## Created: 2026-03-03

## Problem

Benchmark PnL numbers are inflated and uninterpretable. Current results show:
- Oracle: +8,128% average over 7-day periods (impossible)
- Model: -962% average (10x capital loss)
- Random: -4,234% average
- Value MSE: 420.1 (target < 0.1)

These numbers are meaningless because PnL is raw-summed across all assets without normalization.

## Root Cause

**Training oracle** (`mcts_trader/jax_agent/path_solver.py:131`):
```python
step_pnl = jnp.sum(jnp.abs(step_returns) * new_mask) / k_count  # NORMALIZED by k
```

**Benchmark oracle** (`tools/benchmark_oracle_vs_model.py:403`):
```python
step_pnl = np.sum(np.abs(returns[t]) * mask)  # NOT normalized — BUG-9 removed /k
```

**Benchmark model** (`tools/benchmark_oracle_vs_model.py:468`):
```python
step_pnl += np.sum(position * returns[t])  # Raw sum across ALL positioned assets
```

The benchmark sums log returns across all 64 assets × 10,000 timesteps without dividing by the number of active positions. The training oracle divides by `k_count`. The mismatch means:

1. Benchmark PnL is ~k× inflated for oracle, ~n_positions× for model
2. Value MSE compares model predictions (trained against normalized oracle) to unnormalized benchmark oracle PnL
3. Alpha = model_pnl - buy_hold compares different scales

## Fix

### File: `tools/benchmark_oracle_vs_model.py`

**`simulate_oracle_pnl` (line 403):**
```python
# Before (BUG):
step_pnl = np.sum(np.abs(returns[t]) * mask)

# After (FIXED):
k_count = max(np.sum(mask), 1)
step_pnl = np.sum(np.abs(returns[t]) * mask) / k_count
```

**`simulate_model_pnl` (line 468):**
```python
# Before (BUG):
step_pnl += np.sum(position * returns[t])

# After (FIXED):
n_active = max(np.sum(np.abs(position) > 0.5), 1)
step_pnl += np.sum(position * returns[t]) / n_active
```

### File: `tools/benchmark_easy.py`

**`simulate_model_pnl_detailed`** — same fix as above, normalize by active positions.

**`_simulate_pnl_vectorized`** (random baseline) — same normalization.

**Value MSE (line 697):**
```python
# Before — oracle_cum_pnl is from path_solver (already /k normalized)
# This may already be correct IF the benchmark uses the same oracle output
# Verify: oracle_cum_pnl source = greedy_lookahead_path → already normalized
# The value_mse might be correct but appears wrong because the model was
# trained against normalized values and the benchmark report context is unnormalized
```

### Expected Results After Fix

With proper per-position normalization, a 7-day period should show:
- Oracle: ~1-5% (k=4, trading top movers)
- Model: -2% to +2% (currently random-level)
- Random: ~-0.5% (transaction costs drain)
- Value MSE: should drop dramatically (predictions were trained on normalized scale)

## Validation

1. Run benchmark before and after fix on same checkpoint
2. Oracle PnL should be in single-digit % range for 7-day periods
3. Value MSE should drop from 420 to a reasonable range
4. Relative ordering (oracle > model > random) should be preserved
5. TASK_666 success criteria may need threshold recalibration

## Notes

- The "BUG-9 fix" comment at line 402 says "Use raw sum (no /k_count) to match model P&L normalization" — but model P&L also doesn't normalize, so both are wrong
- This doesn't affect training quality — only benchmark reporting
- Alpha calculation also benefits since model_pnl and buy_hold will be on same scale
