# TASK_126d: Linear vs Log Return Scale Mismatch

**Status**: DIAGNOSED (Fix Required)
**Priority**: High
**Created**: 2026-01-31

## Problem Statement

Two metrics that should be directly comparable use **different return scales**:
- `market_return`: computed as **linear percentage returns**
- `optimal_alpha`: computed from **log returns**

This causes `optimal_alpha < market_return` even when the oracle outperforms, because log returns are systematically smaller than linear returns for positive moves.

## Evidence

### 1. market_return: LINEAR Returns

**Location**: `mcts_trader/training/self_play.py:786-787`
```python
asset_returns = (end_prices - start_prices) / (start_prices + 1e-8)
market_return = float(np.mean(asset_returns)) * 100
```

Formula: `(P_end - P_start) / P_start × 100`

### 2. optimal_alpha: LOG Returns

**Location**: `mcts_trader/jax_agent/trainer.py:1214-1226`
```python
# log_price_seq is already in LOG space
returns_for_amax = log_price_seq[:, 1:, :] - log_price_seq[:, :-1, :]
optimal_path = compute_optimal_path(returns_for_amax[0], ...)
optimal_alpha_scaled = optimal_path.total_alpha * 100.0
```

Formula: `log(P_t) - log(P_{t-1}) = log(P_t / P_{t-1})`

### 3. Other Metrics: LINEAR Returns

| Metric | Location | Formula |
|--------|----------|---------|
| `agent_return` | `vectorbt_env.py:1183` | `(new_value / initial_value - 1) × 100` (LINEAR) |
| `alpha` | `self_play.py:811` | `agent_return - market_return` (LINEAR) |
| `cumulative_agent_return` | `vectorbt_env.py:1183` | `(new_value / initial_value - 1) × 100` (LINEAR) |

## Numerical Impact

The discrepancy between linear and log returns:

| Actual Move | Linear Return | Log Return | Difference |
|-------------|---------------|------------|------------|
| +8%         | 8.00%         | 7.70%      | -0.30%     |
| +20%        | 20.00%        | 18.23%     | -1.77%     |
| +50%        | 50.00%        | 40.55%     | -9.45%     |
| +100%       | 100.00%       | 69.31%     | -30.69%    |
| -8%         | -8.00%        | -8.34%     | -0.34%     |
| -20%        | -20.00%       | -22.31%    | -2.31%     |
| -50%        | -50.00%       | -69.31%    | -19.31%    |

**Key insight**: For positive returns, log < linear. This makes `optimal_alpha` appear **smaller** than the equivalent linear return.

## Bug Demonstration

Consider an episode where:
- Market goes up 10% (buy-and-hold)
- Oracle captures exactly the market return

**Expected comparison**:
- `market_return` = 10.0 (linear)
- `optimal_alpha` = 10.0 (should match if oracle equals market)

**Actual comparison**:
- `market_return` = 10.0 (linear: `(1.10 - 1.0) / 1.0 × 100`)
- `optimal_alpha` = 9.53 (log: `ln(1.10) × 100`)

**Result**: Oracle appears to underperform by 0.47% even though it matched market!

## Compound Effect Over Time

For per-step log returns summed over 400 steps:
- Each step's log return is slightly smaller than its linear equivalent
- The gap compounds: `sum(log_returns) < sum(linear_returns)`
- Total discrepancy can be 5-15% over a full episode

## Related Metrics Consistency Check

| Metric | Return Type | Consistent? |
|--------|-------------|-------------|
| `market_return` | LINEAR | ✓ Base |
| `agent_return` | LINEAR | ✓ Matches |
| `alpha` | LINEAR | ✓ Matches |
| `optimal_alpha` | **LOG** | ✗ **MISMATCH** |
| `alpha_capture` | Mixed | ✗ Compares log to linear |

## Root Cause

The `optimal_alpha` computation uses `log_price_seq` which is already in log space:
```python
# trainer.py:1207
log_price_seq = batch.get('log_price_sequence')
```

When computing returns from log prices, we get log returns:
```python
# trainer.py:1214
returns_for_amax = log_price_seq[:, 1:, :] - log_price_seq[:, :-1, :]
```

## Proposed Fixes

### Option A: Convert optimal_alpha to Linear Scale (Recommended)

**Location**: `mcts_trader/jax_agent/trainer.py:1214-1226`

```python
# BEFORE:
returns_for_amax = log_price_seq[:, 1:, :] - log_price_seq[:, :-1, :]
optimal_alpha_scaled = optimal_path.total_alpha * 100.0

# AFTER:
# Convert log prices back to prices, then compute linear returns
price_seq = jnp.exp(log_price_seq)
returns_for_amax = (price_seq[:, 1:, :] - price_seq[:, :-1, :]) / (price_seq[:, :-1, :] + 1e-8)
optimal_alpha_scaled = optimal_path.total_alpha * 100.0
```

### Option B: Convert market_return to Log Scale

**Location**: `mcts_trader/training/self_play.py:786-787`

```python
# BEFORE:
asset_returns = (end_prices - start_prices) / (start_prices + 1e-8)

# AFTER:
asset_returns = np.log(end_prices / (start_prices + 1e-8))
```

### Recommendation

**Use Option A** because:
1. Linear returns are more intuitive for reporting
2. Most other metrics already use linear returns
3. Avoids changing the self_play output format

## Additional Considerations

### Path Solver Already Uses Log Returns

The `compute_optimal_path` function in `path_solver.py` is designed for log returns (they're additive). If we pass linear returns, we need to verify the math still works.

For linear returns:
- Cannot simply sum returns (non-additive)
- Would need: `prod(1 + r_i) - 1` instead of `sum(r_i)`

**Better approach**: Keep log returns for internal computation, convert to linear only for the final metric:

```python
# Compute optimal path using log returns (mathematically correct)
returns_for_amax = log_price_seq[:, 1:, :] - log_price_seq[:, :-1, :]
optimal_path = compute_optimal_path(returns_for_amax[0], ...)

# Convert final log return to linear for comparison
# optimal_path.total_alpha is sum of log returns ≈ log(final/initial)
optimal_alpha_linear = (jnp.exp(optimal_path.total_alpha) - 1.0) * 100.0
```

This preserves the mathematical correctness of the path solver while providing comparable output.

## Verification Steps

After fix:
1. For a flat market (0% return), both metrics should be ~0%
2. For a +10% market move with oracle matching market, both should be ~10%
3. For any episode, `optimal_alpha >= market_return` (oracle should never underperform)

## Related Tasks

- TASK_126a: Reward Spike Investigation
- TASK_126c: Path Solver Direction Bug (contributes to optimal_alpha issues)
- TASK_119: Optimal Path Solver (original implementation)
