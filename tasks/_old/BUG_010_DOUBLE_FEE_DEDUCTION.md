# Bug: Potential Double Fee Deduction

## Priority: HIGH
**Status:** FALSE POSITIVE

## Analysis (2026-01-01)

No double fee. The buy logic works correctly:
1. `order_value` = total spend (includes fee)
2. `net_value = order_value - fee` (used only to calculate shares)
3. `shares = net_value / price` (shares we can afford)
4. `cash -= order_value` (single deduction covers shares + fee)

The fee is calculated once and embedded in `order_value`. The `net_value`
calculation is just for determining how many shares to buy.
**Component:** `mcts_trader/env/vectorbt_env.py`

## Problem Description

Trading fees may be deducted twice in some execution paths:
1. Once in the order execution logic
2. Again in the portfolio update logic

### Potential Locations

```python
# vectorbt_env.py - _execute_buy()
def _execute_buy(self, asset_idx, price):
    fee = trade_value * self.fee_rate
    net_cost = trade_value + fee
    self.portfolio.cash -= net_cost  # Fee deducted here
    ...
    # Check if fee also in portfolio.update()

# vectorbt_env.py - step()
def step(self, action):
    ...
    # Check for secondary fee deduction
```

### Impact

- Trading costs double what they should be
- Profitability artificially reduced
- Agent may learn to avoid trading entirely
- Fee-based curriculum becomes too harsh

## Solution

### Step 1: Audit all fee deduction points

Search for all places where fees are applied:
```bash
grep -n "fee" vectorbt_env.py
```

### Step 2: Ensure single deduction point

```python
# Define single fee application point
def _calculate_trade_cost(self, trade_value: float) -> float:
    """Calculate total cost including fee. ONLY place fees are applied."""
    return trade_value * (1 + self.fee_rate)

# Use consistently
def _execute_buy(self, asset_idx, price):
    gross_cost = shares * price
    total_cost = self._calculate_trade_cost(gross_cost)
    self.portfolio.cash -= total_cost
    # NO additional fee deduction anywhere else
```

### Step 3: Add fee tracking

```python
# Track fees for verification
self.total_fees_paid = 0.0

def _apply_fee(self, trade_value: float) -> float:
    fee = trade_value * self.fee_rate
    self.total_fees_paid += fee
    return fee
```

## Files to Modify

| File | Change |
|------|--------|
| `vectorbt_env.py` | Audit and consolidate fee deduction |

## Testing

1. Execute known trades with known fees
2. Verify total fees match expected: `n_trades * avg_trade_value * fee_rate`
3. Check portfolio PnL calculation

## Verification

```python
# Test with simple scenario
env = VectorBTTradingEnv(fee_rate=0.0026)  # 26 bps
initial_cash = env.portfolio.cash

# Buy then sell same asset
env.step(action=2)  # Buy asset 0
buy_price = ...
env.step(action=0)  # Sell asset 0
sell_price = ...

# Expected fee
expected_fee = buy_price * 0.0026 + sell_price * 0.0026
actual_fee = initial_cash - env.portfolio.cash - PnL

assert abs(actual_fee - expected_fee) < 0.01
```
