# TASK_047: Realistic Limit Order Fill Model

## Status: COMPLETED (2026-01-23)

**Implementation**: `--probabilistic-fills` CLI flag
- Located in: `cli.py`, `config.py`, `vectorbt_env.py`, `jax_trading_env.py`
- Implements stochastic fill probability based on price distance
- See also TASK_050 for related mechanics

## Problem

Current limit order execution is unrealistic:
- Orders fill instantly when price crosses limit price
- No consideration of market depth or queue position
- Agent can exploit by placing orders barely in-the-money
- Poor sim-to-real transfer for limit order strategies

## Solution

Implement **probabilistic fill model** based on distance from market price.

### Core Logic

```python
def calculate_fill_probability(
    limit_price: float,
    market_price: float,
    side: str,  # 'buy' or 'sell'
    volatility: float,  # Recent price volatility (e.g., std of returns)
    order_age: int = 0,  # Steps since order placed
) -> float:
    """
    Probability that a limit order fills this step.

    Intuition:
    - Far from market = aggressive, takes liquidity, fills fast
    - Close to market = passive, provides liquidity, may not fill
    - Older orders gain priority (time-in-queue bonus)
    """
    # Calculate distance as fraction of price
    distance = abs(limit_price - market_price) / market_price

    # Normalize by volatility (1.0 = one volatility unit away)
    norm_distance = distance / max(volatility, 1e-6)

    # Check if order is marketable (would fill instantly in real market)
    if side == 'buy' and limit_price >= market_price:
        return 1.0  # Buy at or above market = instant fill
    if side == 'sell' and limit_price <= market_price:
        return 1.0  # Sell at or below market = instant fill

    # Base probability from distance
    if norm_distance >= 2.0:
        base_prob = 0.95  # Far from market, likely fills on volatility
    elif norm_distance >= 1.0:
        base_prob = 0.50  # Moderate distance
    elif norm_distance >= 0.5:
        base_prob = 0.20  # Close to market
    else:
        base_prob = 0.10  # Very close, queue is long

    # Age bonus: older orders more likely to fill (queue priority)
    # +5% per step, capped at +30%
    age_bonus = min(0.30, order_age * 0.05)

    # Final probability, capped at 0.95 (never guaranteed except marketable)
    return min(0.95, base_prob + age_bonus)
```

### Fill Decision

```python
def should_fill_order(order, market_price, volatility, rng) -> bool:
    """Probabilistic fill decision."""
    fill_prob = calculate_fill_probability(
        limit_price=order.limit_price,
        market_price=market_price,
        side=order.side,
        volatility=volatility,
        order_age=order.age,
    )
    return rng.random() < fill_prob
```

## Implementation

### Files to Modify

1. **`mcts_trader/env/vectorbt_env.py`**
   - Add `volatility` tracking (rolling std of returns)
   - Add `age` field to LimitOrder dataclass
   - Modify `_process_limit_orders()` to use probabilistic fills

2. **`mcts_trader/jax_agent/jax_trading_env.py`**
   - Same changes for JAX environment
   - Use JAX-compatible random for fill decisions

### LimitOrder Dataclass Update

```python
@dataclass
class LimitOrder:
    asset_idx: int
    side: str  # 'buy' or 'sell'
    limit_price: float
    quantity: float = 1.0
    ttl: int = 100  # Time-to-live in steps
    age: int = 0    # NEW: Steps since placed
```

### Volatility Tracking

```python
def _update_volatility(self):
    """Update rolling volatility estimate."""
    if len(self.price_history) >= 20:
        returns = np.diff(np.log(self.price_history[-21:]))
        self.volatility = np.std(returns)
    else:
        self.volatility = 0.01  # Default 1% volatility
```

### Modified Fill Logic

```python
def _process_limit_orders(self):
    """Process pending limit orders with probabilistic fills."""
    filled_orders = []
    remaining_orders = []

    for order in self.pending_orders:
        order.age += 1  # Increment age
        order.ttl -= 1  # Decrement TTL

        if order.ttl <= 0:
            # Expired
            continue

        market_price = self.current_prices[order.asset_idx]

        if self._should_fill_order(order, market_price):
            filled_orders.append(order)
            self._execute_fill(order, market_price)
        else:
            remaining_orders.append(order)

    self.pending_orders = remaining_orders
    return filled_orders
```

## Testing

### Unit Tests

```python
def test_marketable_orders_fill_instantly():
    """Buy above market or sell below market = instant fill."""
    assert calculate_fill_probability(101, 100, 'buy', 0.01) == 1.0
    assert calculate_fill_probability(99, 100, 'sell', 0.01) == 1.0

def test_far_orders_high_probability():
    """Orders far from market have high fill probability."""
    prob = calculate_fill_probability(95, 100, 'buy', 0.02)  # 2.5 vol units
    assert prob >= 0.90

def test_close_orders_low_probability():
    """Orders close to market have low fill probability."""
    prob = calculate_fill_probability(99.5, 100, 'buy', 0.02)  # 0.25 vol units
    assert prob <= 0.15

def test_age_increases_probability():
    """Older orders more likely to fill."""
    prob_new = calculate_fill_probability(98, 100, 'buy', 0.02, order_age=0)
    prob_old = calculate_fill_probability(98, 100, 'buy', 0.02, order_age=5)
    assert prob_old > prob_new
```

### Integration Test

Run training with new fill model:
```bash
python train_jax_fast.py \
    --synthetic-level 0 \
    --epochs 100 \
    --name test_probabilistic_fills
```

Monitor:
- Limit order fill rate (should decrease)
- Limit order action frequency (agent may adapt strategy)
- Average order age at fill (should be > 1 step for close orders)

## Expected Behavior Changes

| Order Type | Old Behavior | New Behavior |
|------------|--------------|--------------|
| Buy 2% below market | Instant fill on any dip | ~95% per step |
| Buy 1% below market | Instant fill on any dip | ~50% per step |
| Buy 0.5% below market | Instant fill on any dip | ~20% per step |
| Buy 0.1% below market | Instant fill on any dip | ~10% per step |
| Buy at market price | Instant fill | Instant fill |

## Configuration

Add CLI args for tuning:

```python
parser.add_argument('--limit-fill-model', choices=['instant', 'probabilistic'],
                   default='probabilistic', help='Limit order fill model')
parser.add_argument('--limit-fill-vol-window', type=int, default=20,
                   help='Window for volatility calculation')
```

## Priority

**MEDIUM** - Improves realism, prevents limit order exploitation, better sim-to-real transfer.

## Dependencies

None - can be implemented independently.

## Risks

- Agent may stop using limit orders entirely (monitor action distribution)
- May need to tune probability thresholds for curriculum levels
- Synthetic data volatility may differ from real market
