# TASK_050: Fix Limit Order Fill Mechanics

## Status: COMPLETED (2026-01-23)

**Implementation**: `--probabilistic-fills` CLI flag
- Located in: `cli.py`, `config.py`, `setup.py`, `env_factory.py`, `jax_trading_env.py`
- Adds realistic stochastic fill behavior for limit orders

## Problems

### 1. Fill at Limit Price (Not Market)

**Current behavior:**
```python
# Agent places buy limit at $98 when market is $100
# Market drops to $90
# Agent gets filled at $98 (their limit price)
# In reality: would fill at $90 (better price!)
```

**Impact:** Agent learns strategy that gives FREE ALPHA in simulation.

### 2. Marketable Limits Get Maker Fee

**Current behavior:**
```python
# Agent places buy limit at $110 when market is $100
# Order fills instantly at $100
# Charged maker_fee instead of taker_fee
```

**Impact:** Fee arbitrage - instant execution at lower cost.

## Solution

### Fix 1: Fill at Market Price

```python
# jax_trading_env.py - execute_limit_buy

def execute_limit_buy(state, asset_idx, current_price):
    """Execute limit buy at the BETTER of limit or market price."""

    # OLD: fill at limit price
    # fill_price = state.order_price[asset_idx]

    # NEW: fill at market (better for buyer)
    fill_price = jnp.minimum(state.order_price[asset_idx], current_price)

    # ... rest of execution logic
```

```python
# For sells, fill at market (better for seller = higher)
def execute_limit_sell(state, asset_idx, current_price):
    fill_price = jnp.maximum(state.order_price[asset_idx], current_price)
```

### Fix 2: Marketable Orders = Taker Fee

```python
def place_limit_order(state, asset_idx, side, limit_price, current_price, config):
    """Place limit order, detecting marketable orders."""

    is_marketable_buy = (side == 'buy') and (limit_price >= current_price)
    is_marketable_sell = (side == 'sell') and (limit_price <= current_price)

    if is_marketable_buy or is_marketable_sell:
        # Treat as market order - execute immediately with taker fee
        fee = config.taker_fee_bps  # Higher fee
        return execute_market_order(state, asset_idx, side, current_price, fee)
    else:
        # True limit order - queue with maker fee on fill
        return queue_limit_order(state, asset_idx, side, limit_price)
```

## Implementation

### Files to Modify

1. **`mcts_trader/jax_agent/jax_trading_env.py`**
   - `execute_limit_buy()` ~line 580
   - `execute_limit_sell()` ~line 620
   - `_handle_limit_buy_action()` ~line 870
   - `_handle_limit_sell_action()` ~line 920

2. **`mcts_trader/env/vectorbt_env.py`**
   - `_fill_limit_order()` ~line 850
   - `_process_limit_orders()` ~line 787

### Code Changes

#### jax_trading_env.py

```python
# Line ~473 (fill_limit_orders)
# OLD:
buy_fill_price = state.order_price
sell_fill_price = state.order_price

# NEW:
buy_fill_price = jnp.minimum(state.order_price, current_prices)  # Better for buyer
sell_fill_price = jnp.maximum(state.order_price, current_prices)  # Better for seller
```

```python
# Line ~870 (_handle_limit_buy_action)
# Add marketable check:
is_marketable = limit_price >= current_price
fee_bps = jnp.where(is_marketable, config.taker_fee_bps, config.maker_fee_bps)

# If marketable, execute immediately instead of queuing
state = jnp.where(
    is_marketable,
    execute_market_order(state, asset_idx, 'buy', current_price, fee_bps),
    queue_limit_order(state, asset_idx, 'buy', limit_price)
)
```

#### vectorbt_env.py

```python
# Line ~850 (_fill_limit_order)
def _fill_limit_order(self, asset_idx: int, order: LimitOrder, market_price: float):
    """Fill limit order at market price, not limit price."""

    if order.side == 'buy':
        # Fill at better of limit or market (for buyer, lower is better)
        fill_price = min(order.limit_price, market_price)
    else:
        # Fill at better of limit or market (for seller, higher is better)
        fill_price = max(order.limit_price, market_price)

    # Execute with fill_price...
```

## Testing

```python
def test_limit_buy_fills_at_market():
    """Limit buy should fill at market price when better."""
    env = TradingEnv()
    env.reset()

    # Place limit buy at $98, market at $100
    env.place_limit_order(asset=0, side='buy', price=98)

    # Market drops to $90
    env.set_price(asset=0, price=90)
    env.step()  # Process orders

    # Should have filled at $90, not $98
    assert env.last_fill_price == 90

def test_marketable_limit_gets_taker_fee():
    """Limit orders at/through market should pay taker fee."""
    env = TradingEnv()
    env.reset()

    # Place limit buy at $110, market at $100 (marketable!)
    env.place_limit_order(asset=0, side='buy', price=110)

    # Should execute immediately with taker fee
    assert env.last_fee_type == 'taker'
```

## Migration

No migration needed - this is a bugfix. Existing checkpoints work fine.

## Priority

**CRITICAL** - This is the #1 sim-to-real gap issue. Agent strategies that exploit this will fail catastrophically in production.

## Related

- TASK_047: Probabilistic fill model (addresses queue/depth)
- This task: Fill price mechanics (addresses pricing)

Both should be implemented together for realistic limit orders.
