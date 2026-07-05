# TASK_126a: Catastrophic Reward Spike Investigation

**Status**: DIAGNOSED (Fixes Pending)
**Priority**: High
**Created**: 2026-01-31

## Problem Statement

During training, epoch 27 showed a catastrophic reward drop from ~-2 to -30.99:
```
epoch=27 reward=-30.991934 alpha=-4.624 market_return=2.36 optimal_alpha=3.15
```

This represents a **15x worse** performance than normal episodes.

## Investigation Summary

### Files Analyzed
| File | Purpose |
|------|---------|
| `mcts_trader/jax_agent/trainer.py` | Reward aggregation, loss computation |
| `mcts_trader/env/jax_trading_env.py` | Environment reward computation |
| `mcts_trader/jax_agent/differential_reward.py` | Reward shaping |
| `mcts_trader/training/self_play.py` | Episode collection |
| `mcts_trader/jax_agent/returns.py` | MC/GAE return computation |
| `mcts_trader/utils/transforms.py` | symlog transformation |

### Key Finding: The -30.99 is Monte Carlo Return, Not Single-Step

The `reward=-30.991934` is the **discounted cumulative reward** over ~400 steps, computed via:
```python
# returns.py:113
running_return = rewards[t] + discount * running_return
```

For MC return = -30.99 with discount=0.997:
- Implied average per-step reward: ~-0.15
- Implied portfolio loss: **~60% over the episode**

## Root Cause Analysis

### 1. Reward Computation Flow

**Location**: `jax_trading_env.py:1196-1203`
```python
safe_prev = jnp.maximum(state.prev_portfolio_value, 1e-8)
safe_new = jnp.maximum(new_portfolio_value, 1e-8)
log_return = jnp.log(safe_new / safe_prev) * 100.0  # <-- NOT CLIPPED
reward = jnp.where(is_liquidated, symlog(-50.0), symlog(log_return))
```

**Issue**: `log_return` is unbounded before symlog transformation.

### 2. Identified Vulnerabilities

#### A. Missing log_return Clipping (HIGH RISK)
```python
# jax_trading_env.py:1202
log_return = jnp.log(safe_new / safe_prev) * 100.0
# No clipping between computation and symlog!
```

When `safe_prev = 1e-8` (bankruptcy floor) and `safe_new = 100000`:
- `log(10^13) * 100 = 2994` (extreme positive)

When portfolio crashes:
- Repeated moderate losses compound to catastrophic MC return

#### B. Short Position Unlimited Downside
```python
# jax_trading_env.py:220-226
short_value = 2.0 * pos_size * pos_entry - pos_size * current_prices
```
No cap on short exposure relative to portfolio value.

#### C. Post-Bankruptcy Reward Explosion
```python
# jax_trading_env.py:386
new_cash = jnp.where(is_bankrupt, 0.0, new_cash)
```
After bankruptcy, `prev_portfolio_value` can hit the `1e-8` floor, causing subsequent log returns to explode.

#### D. No NaN/Inf Guard on Final Reward
```python
# jax_trading_env.py:1203
reward = jnp.where(is_liquidated, symlog(-50.0), symlog(log_return))
# No jnp.isfinite() check!
```

#### E. Unbounded MC Return Accumulation
```python
# returns.py:113
running_return = rewards[t] + discount * running_return
# No clipping on accumulation
```

### 3. Metrics Analysis

| Metric | Value | Interpretation |
|--------|-------|----------------|
| `market_return` | +2.36% | Market was UP |
| `alpha` | -4.624% | Agent underperformed by 4.6% |
| `optimal_alpha` | +3.15% | Good strategy could gain 3.15% |
| `reward` | -30.99 | Catastrophic cumulative loss |

**Scenario**: Agent likely held **short positions** during a market rally. With the market up +2.36%, leveraged shorts would experience compounding losses.

### 4. Comparison: JAX vs VectorBT Envs

The VectorBT environment has better guards:
```python
# vectorbt_env.py:1155-1160
if prev_value > 1e-8 and new_value > 1e-8:
    log_return = np.log(new_value / prev_value) * 100.0
else:
    log_return = (new_value - prev_value) / (abs(prev_value) + 1e-8) * 100.0
```

The JAX env lacks this conditional fallback.

## Division-by-Zero Audit

### Properly Guarded (Good)
| Location | Guard |
|----------|-------|
| `jax_trading_env.py:744` | `jnp.maximum(buy_exec_price, 1e-10)` |
| `jax_trading_env.py:769` | `jnp.maximum(new_long_size, 1e-10)` |
| `jax_trading_env.py:870` | `jnp.maximum(price, 1e-10)` |
| `jax_trading_env.py:1200-1201` | `jnp.maximum(..., 1e-8)` |

### Unguarded Risks (Problematic)
| Location | Issue |
|----------|-------|
| `jax_trading_env.py:1202` | `log_return` unbounded before symlog |
| `returns.py:113` | `running_return` accumulates without bounds |

## Proposed Fixes

### Fix 1: Clip log_return before symlog
**File**: `mcts_trader/env/jax_trading_env.py:1202`
```python
# BEFORE:
log_return = jnp.log(safe_new / safe_prev) * 100.0

# AFTER:
log_return = jnp.log(safe_new / safe_prev) * 100.0
log_return = jnp.clip(log_return, -100.0, 100.0)  # Cap at +/-100% per step
```

### Fix 2: Add NaN/Inf fallback on reward
**File**: `mcts_trader/env/jax_trading_env.py:1203`
```python
# BEFORE:
reward = jnp.where(is_liquidated, symlog(-50.0), symlog(log_return))

# AFTER:
reward = jnp.where(is_liquidated, symlog(-50.0), symlog(log_return))
reward = jnp.where(jnp.isfinite(reward), reward, symlog(-50.0))  # NaN/Inf fallback
```

### Fix 3: Bound MC return accumulation
**File**: `mcts_trader/jax_agent/returns.py:113`
```python
# BEFORE:
running_return = rewards[t] + discount * running_return

# AFTER:
running_return = rewards[t] + discount * running_return
running_return = np.clip(running_return, -50.0, 50.0)  # Prevent runaway
```

### Fix 4: Cap short exposure (optional, more invasive)
**File**: `mcts_trader/env/jax_trading_env.py` (new logic in short processing)
```python
# Limit short exposure to 50% of portfolio value
max_short_value = portfolio_value * 0.5
shares_to_sell = jnp.minimum(shares_to_sell, max_short_value / jnp.maximum(price, 1e-10))
```

## Verification Steps

To confirm root cause before applying fixes:
1. Add logging to capture per-step rewards during affected epochs
2. Check if `is_bankrupt` or `is_liquidated` triggered
3. Log `prev_portfolio_value` min/max per epoch
4. Track short position sizes during market rallies

## Related Tasks
- TASK_118: Numerical Stability Audit (sigma clipping fixes)
- TASK_115c: BUGHUNT Numerical Stability
- TASK_122: Money Printing Bugs

## Notes

The `jnp.where` JAX pitfall (evaluates both branches) is NOT the primary cause here, but could contribute to gradient issues if either branch produces NaN during backprop.

The symlog transformation correctly compresses extreme values, but the issue is that moderately negative per-step rewards (-0.15) compound over 400 steps to produce the -30.99 MC return. The fix should focus on preventing the conditions that cause consistent moderate losses (short position risk) rather than just clipping the final result.
