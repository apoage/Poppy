# BUG #9: Position-Size Division by Zero in execute_buy

**Severity:** HIGH
**File:** `mcts_trader/env/jax_trading_env.py:507, 530, 627`
**Status:** FIXED

## Code
```python
min_trade_value = 100.0
can_afford = (state.cash > min_trade_value) & (shares_to_buy > 0)

# New long size (if opening or adding)
new_long_size = jnp.where(has_position & is_long,
                           current_size + shares_to_buy,
                           shares_to_buy)
new_long_entry = jnp.where(has_position & is_long,
                            (current_size * current_entry + shares_to_buy * buy_exec_price) / new_long_size,
                            buy_exec_price)
```

## Issue
When adding to existing long position (`adding_long = has_position & is_long & can_afford`), the division by `new_long_size` can produce NaN if `new_long_size` is 0 (e.g., if `shares_to_buy` computes to 0 and `current_size` is also 0 due to floating point edge cases).

## Impact
NaN propagation through portfolio value calculations, corrupting rewards and training.

## Expected Fix
Add `jnp.maximum(new_long_size, 1e-10)` in denominator.

## Investigation Checklist
- [x] Verify the edge case can actually occur - Yes, division by zero possible in jnp.where branches
- [x] Check for similar division issues in execute_sell - Found at line 627
- [x] Add safeguards to all division operations

## Fixes Applied
Three defensive guards added:
1. `execute_buy` line 508: `shares_to_buy = actual_cost / jnp.maximum(buy_exec_price, 1e-10)`
2. `execute_buy` line 532: `safe_divisor = jnp.maximum(new_long_size, 1e-10)` for weighted average entry
3. `execute_sell` line 627: `shares_to_sell = actual_short_cost / jnp.maximum(price, 1e-10)`
