# TASK_126i: Return Scale Mismatch Fix

## Problem Statement

Two metrics that should be comparable use **different return scales**, causing incorrect comparisons and misleading performance measurements.

**Example discrepancy from logs:**
```
market_return=2.36 optimal_alpha=3.15
```

For an 8% price move:
- Linear return: 8.0%
- Log return: ln(1.08) = 7.7%
- Difference: 0.3% (4% relative error)

For larger moves, this error compounds significantly.

---

## Root Cause Analysis

### Three Different Return Calculations

| Metric | Location | Return Type | Formula |
|--------|----------|-------------|---------|
| `market_return` | `self_play.py:787` | **Linear** | `(end - start) / start * 100` |
| `agent_return` | `jax_trading_env.py:1267` | **Linear** | `(value / initial - 1) * 100` |
| `optimal_alpha` | `trainer.py:1214-1226` | **Log** | `Σ(log_returns) * 100` |

### The Mismatch

```python
# 1. market_return (LINEAR)
# File: mcts_trader/training/self_play.py:786-787
asset_returns = (end_prices - start_prices) / (start_prices + 1e-8)
market_return = float(np.mean(asset_returns)) * 100  # LINEAR %

# 2. optimal_alpha (LOG)
# File: mcts_trader/jax_agent/trainer.py:1214-1226
returns_for_amax = log_price_seq[:, 1:, :] - log_price_seq[:, :-1, :]  # LOG returns
optimal_path = compute_optimal_path(returns_for_amax[0], ...)
optimal_alpha_scaled = optimal_path.total_alpha * 100.0  # LOG %

# 3. alpha calculation (MIXED!)
# File: mcts_trader/training/self_play.py:811
alpha = final_return - market_return  # LINEAR - LINEAR ✓

# 4. alpha_capture (MIXED - WRONG!)
# File: mcts_trader/jax_agent/trainer.py:1234
alpha_capture = agent_pnl / jnp.maximum(jnp.abs(optimal_alpha_scaled), 1e-6)  
# agent_pnl is SYMLED-scaled log returns, optimal_alpha is raw log returns
```

### Why This Causes Problems

#### Problem 1: Underestimation of Oracle Performance

When `optimal_alpha` uses log returns while `market_return` uses linear:

```python
# Market goes from 100 → 108 (+8% linear)
market_return = 8.0  # linear

# Oracle using log returns
optimal_alpha = ln(108/100) * 100 = 7.7  # log

# Compare: 7.7 < 8.0 → "Oracle underperformed market" ❌ WRONG!
```

#### Problem 2: Inconsistent Alpha Comparisons

```python
# alpha = agent_return - market_return (both linear) ✓
alpha = 5.0 - 2.36  # = 2.64 (correct linear alpha)

# But optimal_alpha is log-based
optimal_alpha = 3.15  # log-based

# Cannot compare: 2.64 vs 3.15 are different scales!
```

#### Problem 3: Alpha Capture Calculation is Broken

```python
# File: trainer.py:1229-1234
agent_pnl = jnp.sum(rewards[0])  # Sum of symlog(log_return * 100)
optimal_alpha_scaled = optimal_path.total_alpha * 100.0  # Sum of log_returns * 100

# These are NOT comparable!
# - agent_pnl: symlog-scaled, per-step
# - optimal_alpha: raw log returns
# Both multiplied by 100 but different transformations

alpha_capture = agent_pnl / jnp.abs(optimal_alpha_scaled)  # MEANINGLESS!
```

### Scale Difference Examples

| Price Move | Linear Return | Log Return | Error |
|------------|---------------|------------|-------|
| +1% | 1.00% | 0.995% | 0.05% |
| +5% | 5.00% | 4.88% | 0.12% |
| +8% | 8.00% | 7.70% | 0.30% |
| +20% | 20.00% | 18.23% | 1.77% |
| +50% | 50.00% | 40.55% | 9.45% |
| -10% | -10.00% | -10.54% | 0.54% |
| -20% | -20.00% | -22.31% | 2.31% |

**For volatile crypto assets with ±20% moves, the error exceeds 2%!**

---

## The Fix

### Option 1: Standardize on Log Returns (Recommended)

**Rationale:**
- Log returns are time-additive: `log(P2/P0) = log(P2/P1) + log(P1/P0)`
- Log returns are symmetric: ±10% have equal magnitude
- Agent rewards already use log returns (symlog of log returns)
- Better for compounding over multiple steps

**Changes needed:**

```python
# 1. Fix market_return in self_play.py:786-787
# OLD:
asset_returns = (end_prices - start_prices) / (start_prices + 1e-8)
market_return = float(np.mean(asset_returns)) * 100

# NEW:
log_asset_returns = jnp.log(end_prices / (start_prices + 1e-8))
market_return = float(jnp.mean(log_asset_returns)) * 100
```

```python
# 2. Fix agent_return in jax_trading_env.py:1267
# Already correct - uses linear, but comment should note this is intentional
# OR convert to log:
new_cumulative_return = jnp.log(new_portfolio_value / state.initial_portfolio_value) * 100.0
```

```python
# 3. optimal_alpha is already log-based ✓
```

### Option 2: Standardize on Linear Returns

**Rationale:**
- More intuitive for traders ("8% gain" vs "7.7% log return")
- Matches traditional financial reporting
- Simpler mental math

**Changes needed:**

```python
# 1. Fix optimal_alpha calculation in trainer.py:1214
# OLD:
returns_for_amax = log_price_seq[:, 1:, :] - log_price_seq[:, :-1, :]

# NEW: Convert log returns to linear
log_returns = log_price_seq[:, 1:, :] - log_price_seq[:, :-1, :]
linear_returns = jnp.exp(log_returns) - 1.0
returns_for_amax = linear_returns
```

### Option 3: Keep Both But Label Clearly (Documentation Fix)

If both scales are needed for different purposes, clearly label them:

```python
metrics['market_return_linear'] = market_return_linear
metrics['market_return_log'] = market_return_log
metrics['optimal_alpha_log'] = optimal_alpha_scaled
metrics['optimal_alpha_linear'] = (jnp.exp(optimal_path.total_alpha) - 1) * 100
```

---

## Recommended Implementation

### Phase 1: Fix market_return to Use Log Returns

**File:** `mcts_trader/training/self_play.py`
**Lines:** 784-787

```python
# Compute market return (buy & hold baseline) using LOG RETURNS
start_idx = int(jax_env_states.episode_start_idx[i])
end_idx = start_idx + n_steps_i
if end_idx < len(cpu_prices) and n_steps_i > 0:
    start_prices = cpu_prices[start_idx]
    end_prices = cpu_prices[end_idx - 1]
    # LOG RETURN: ln(P_end / P_start)
    log_asset_returns = np.log(end_prices / (start_prices + 1e-8))
    market_return = float(np.mean(log_asset_returns)) * 100
else:
    market_return = 0.0
```

### Phase 2: Fix agent_return to Use Log Returns

**File:** `mcts_trader/env/jax_trading_env.py`
**Line:** 1267

```python
# LOG RETURN version (consistent with optimal_alpha)
new_cumulative_return = jnp.log(
    new_portfolio_value / state.initial_portfolio_value
) * 100.0
```

### Phase 3: Verify optimal_alpha Scale

**File:** `mcts_trader/jax_agent/trainer.py`
**Lines:** 1220-1226

The comment says "Agent reward: symlog(log_return * 100) per step", which is correct. But verify:

```python
# Agent reward computation (env):
# log_return = jnp.log(safe_new / safe_prev) * 100.0
# reward = symlog(log_return)

# So agent_pnl = sum(symlog(log_returns * 100))

# Optimal path uses raw log returns:
# optimal_path.total_alpha = sum(log_returns)
# optimal_alpha_scaled = total_alpha * 100

# For fair comparison, we should compare:
# - agent_pnl: sum of symlog(log_returns * 100)
# - optimal_pnl: should be sum of symlog(optimal_log_returns * 100)

# Currently we compare symlog-scaled agent vs raw log optimal - WRONG!
```

### Phase 4: Fix Alpha Capture Calculation

**File:** `mcts_trader/jax_agent/trainer.py`
**Lines:** 1228-1235

```python
# Compute comparable PnL values
# Agent: already in symlog space
agent_pnl = jnp.sum(rewards[0])

# Optimal: apply same symlog transformation
from mcts_trader.utils.transforms import symlog
optimal_log_returns = returns_for_amax[0]  # Already selected optimal path
optimal_pnl = jnp.sum(symlog(optimal_log_returns * 100.0))

# Now they are comparable!
metrics['optimal_alpha'] = optimal_pnl  # Symlog-scaled for fair comparison
metrics['optimal_alpha_raw'] = optimal_path.total_alpha * 100.0  # Raw log

# Correct alpha capture
alpha_capture = agent_pnl / jnp.maximum(jnp.abs(optimal_pnl), 1e-6)
```

---

## Verification Tests

### Test 1: Oracle Always Beats Market

```python
import jax.numpy as jnp
from mcts_trader.jax_agent.path_solver import greedy_lookahead_path

# Create simple upward market
prices = jnp.array([100.0, 108.0])  # +8%
log_returns = jnp.log(prices[1:] / prices[:-1])

# Oracle with k=1 (can hold best asset)
optimal = greedy_lookahead_path(log_returns[None, :], k=1)

# Market return (log)
market_return = jnp.mean(log_returns) * 100  # 7.7%

# Oracle should beat or equal market
assert optimal.total_alpha * 100 >= market_return - 1e-6
```

### Test 2: Same Scale Comparison

```python
# Both should use same return type
def test_scale_consistency():
    # Simulate episode
    prices = jnp.linspace(100, 120, 100)  # 20% up
    
    # Log return calculation
    log_returns = jnp.log(prices[1:] / prices[:-1])
    log_market_return = jnp.sum(log_returns) * 100
    
    # Linear return calculation  
    linear_return = (prices[-1] - prices[0]) / prices[0] * 100
    
    # These should be close for small moves
    # For 20% move: 20% vs 18.23% - NOT close!
    print(f"Linear: {linear_return:.2f}%")
    print(f"Log: {log_market_return:.2f}%")
    print(f"Error: {abs(linear_return - log_market_return):.2f}%")
```

### Test 3: Round-Trip Consistency

```python
# Price goes 100 → 110 → 100 (round trip)
prices = jnp.array([100.0, 110.0, 100.0])

# Linear: +10% then -9.09% = +0.91% (wrong - should be 0%)
linear_returns = [(110-100)/100, (100-110)/110]
linear_total = sum(linear_returns) * 100  # 0.91%

# Log: +9.53% then -9.53% = 0% (correct!)
log_returns = jnp.log(jnp.array([110/100, 100/110]))
log_total = jnp.sum(log_returns) * 100  # 0%

# Log returns are correct for round-trip
```

---

## Impact Assessment

### Current State (Broken)
- `market_return` uses linear
- `agent_return` uses linear  
- `optimal_alpha` uses log
- `alpha = agent_return - market_return` ✓ (linear - linear)
- `alpha_capture = agent_pnl / optimal_alpha` ✗ (symlog vs log)

### After Fix
- `market_return` uses log
- `agent_return` uses log
- `optimal_alpha` uses log (already correct)
- `alpha = agent_return - market_return` ✓ (log - log)
- `alpha_capture = agent_pnl / optimal_pnl` ✓ (symlog vs symlog)

### Metrics Affected
1. `optimal_alpha` - Currently underestimated vs market
2. `alpha_capture` - Currently meaningless (comparing incompatible scales)
3. `regret` - Based on wrong alpha_capture
4. Training decisions based on regret may be wrong

---

## Acceptance Criteria

- [ ] `market_return` uses log returns (not linear)
- [ ] `agent_return` uses log returns (not linear)
- [ ] `optimal_alpha` documented as log returns
- [ ] `alpha_capture` compares symlog-scaled agent vs symlog-scaled optimal
- [ ] Test: Oracle always achieves `optimal_alpha >= market_return` (for positive markets)
- [ ] Test: Round-trip price movement gives 0% return (log returns)
- [ ] All existing tests pass with updated expected values

---

## Related Tasks

- TASK_119: Optimal hindsight path (path_solver)
- TASK_121: Fixed observation window
- TASK_126h: Path solver PnL bug fix (directional mask issue)

---

## References

- `mcts_trader/training/self_play.py:784-787` - market_return calculation
- `mcts_trader/jax_agent/trainer.py:1214-1235` - optimal_alpha and alpha_capture
- `mcts_trader/env/jax_trading_env.py:1267` - agent_return calculation
- `mcts_trader/utils/transforms.py` - symlog implementation
