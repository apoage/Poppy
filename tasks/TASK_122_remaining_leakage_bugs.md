# TASK_122: Remaining Future Leakage Bugs

**Status:** SUPERSEDED - See `TASK_122_money_printing_bugs.md`
**Date:** 2026-01-29
**Depends On:** TASK_121 (60-step offset fix applied)
**Symptom:** Alpha still 150-640% after TASK_121 fix (should be ~0%)

> **NOTE:** All bugs identified here have been fixed. See `TASK_122_money_printing_bugs.md` for consolidated status.

---

## Current State

### TASK_121 Fix Applied
- Removed `+ config.window_size` from `step()` line 1118
- Removed `+ config.window_size` from `get_info()` line 1622
- JAX cache cleared

### Test Results (Post-Fix)
```
Ep 1 | α: +639.42% (Mkt: +5.2%)  <- Still way too high!
Ep 2 | α: +193.21% (Mkt: +0.6%)
Ep 3 | α: +347.96% (Mkt: +3.6%)
Ep 4 | α: +150.05% (Mkt: +1.6%)
```

**Conclusion:** There is at least one MORE bug causing future leakage or money printing.

---

## Verified Components (No Bug Found)

| Component | File:Line | Index Used | Status |
|-----------|-----------|------------|--------|
| Observation window | `jax_trading_env.py:1352` | `ep_start + step - 59` to `ep_start + step` | ✓ Correct |
| Trading prices | `jax_trading_env.py:1118` | `ep_start + step` | ✓ Fixed |
| Burning money | `jax_trading_env.py:1198` | `ep_start + step` | ✓ Consistent |
| Action mask | `self_play.py:632-635` | Uses `prices[price_idx]` where `price_idx = ep_start + step` | ✓ Correct |
| Encoder window | `jax_encoder.py:104` | `dynamic_slice(prices, (window_start, 0), (ws, n))` | ✓ Correct |
| Market return calc | `self_play.py:761-768` | `prices[ep_start]` to `prices[ep_start + n_steps - 1]` | ✓ Correct |

---

## Remaining Investigation Areas

### Area 1: Portfolio Value Calculation
**File:** `mcts_trader/env/jax_trading_env.py`
**Function:** `compute_portfolio_value()`

```python
def compute_portfolio_value(state, current_prices):
    long_value = sum(pos_size * current_prices)  # Uses current_prices
    short_value = sum(pos_size * (2 * pos_entry - current_prices))
    return cash + long_value + short_value
```

**Check:** Is `current_prices` passed correctly from step()?

### Area 2: Position Entry Recording
**File:** `mcts_trader/env/jax_trading_env.py`
**Function:** `_execute_buy_jax()`, `_execute_sell_jax()`

When agent opens position, `pos_entry` is recorded. Verify it uses:
- `current_prices[asset_idx]` where current_prices = `prices[ep_start + step]`
- NOT a future price

### Area 3: Reward Calculation
**File:** `mcts_trader/env/jax_trading_env.py:1178-1185`

```python
log_return = jnp.log(safe_new / safe_prev) * 100.0
```

**Check:**
- `safe_prev` = previous portfolio value (at step T-1)
- `safe_new` = new portfolio value (at step T)
- Both should use consistent price indices

### Area 4: Observation Features Leaking Future
**File:** `mcts_trader/env/jax_encoder.py`

Check if any feature computation accidentally uses prices beyond the window:
```python
returns_5m = (prices[:, -1] - prices[:, -5]) / (prices[:, -5] + 1e-8)
momentum_60 = (prices[:, -1] - prices[:, 0]) / (price_range + 1e-8)
```

All should use `prices` which is the sliced window, not the full price array.

### Area 5: Cumulative Return Calculation
**File:** `mcts_trader/env/jax_trading_env.py:1249`

```python
new_cumulative_return = (new_portfolio_value / state.initial_portfolio_value - 1.0) * 100.0
```

**Check:** Is `initial_portfolio_value` set correctly at reset?

### Area 6: Historical Data Specific Path
**File:** `mcts_trader/training/env_factory.py:263-367`

The `create_jax_env_historical()` function loads prices from database. Check if there's any offset issue specific to this path.

---

## Debug Strategy

### Step 1: Add Index Verification Print
In `self_play.py` after line 768, add:
```python
if i == 0:
    print(f"[DEBUG] ep_start={start_idx}, trade_range=[{start_idx}, {end_idx-1}], "
          f"mkt={market_return:.1f}%, agent={final_return:.1f}%")
```

### Step 2: Verify Position Entry Prices
In `_execute_buy_jax()`, log the entry price:
```python
jax.debug.print("BUY asset={a} at price={p}", a=asset_idx, p=price)
```

### Step 3: Trace Portfolio Value
At start and end of episode, print:
- `initial_portfolio_value`
- `final_portfolio_value`
- `cumulative_return`

### Step 4: Check for Price Array Mutation
Verify that the price array is not being modified during the episode.

---

## Potential Root Causes (Hypotheses)

### Hypothesis A: Same-Bar Execution Advantage
- Observation includes price at time T
- Trading executes at price T (same price)
- Agent can "see" the price it will trade at

**Why this might cause alpha:** If there's any signal in the observation that correlates with T→T+1 movement, the agent benefits.

**Test:** Change observation to end at T-1, trading at T.

### Hypothesis B: Position PnL in Observation Leaks Info
- Channel 3 of observation shows position PnL
- PnL = (current_price - pos_entry) / pos_entry
- If pos_entry was recorded wrong (future price), PnL would leak info

**Test:** Disable position channels in observation.

### Hypothesis C: Short Position Valuation Bug
```python
short_value = pos_size * (2 * pos_entry - current_prices)
```

If `pos_entry` > `current_prices` (shorted at higher price), this gives positive value. But what if indices are misaligned?

### Hypothesis D: Replay from Wrong Episode Start
The alpha calculation uses `jax_env_states.episode_start_idx[i]` - verify this matches the actual trading range.

---

## Files to Investigate

| Priority | File | Reason |
|----------|------|--------|
| HIGH | `jax_trading_env.py` | Trading execution, position entry |
| HIGH | `self_play.py` | Observation construction, alpha calc |
| MEDIUM | `jax_encoder.py` | Feature computation |
| MEDIUM | `env_factory.py` | Historical data loading |
| LOW | `vectorbt_env.py` | Reference implementation comparison |

---

## Next Steps

1. Add debug prints to verify actual indices at runtime
2. Compare agent's portfolio value trajectory vs expected
3. Check if disabling certain observation channels reduces alpha
4. Test with synthetic data (--synthetic-level 0) to isolate from historical data issues
