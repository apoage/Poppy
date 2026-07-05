# TASK_052: Randomize Limit Order Fill Priority

## Status: PENDING
## Priority: MEDIUM

## Problem

JAX env uses `jnp.cumsum` on asset indices for fill priority when cash is limited.
Asset 0 always fills before asset 63.

```python
cumulative_buy_cost = jnp.cumsum(buy_cost_if_opening)
can_afford_cumulative = cumulative_buy_cost <= available_cash_for_buys
```

**Impact:** Agent may favor lower-indexed assets because they're "guaranteed" to fill first.

## Solution

Shuffle asset indices before cumsum:

```python
def fill_limit_orders(state, current_prices, rng_key):
    # Randomize priority
    n_assets = state.order_active.shape[0]
    priority_order = jax.random.permutation(rng_key, n_assets)

    # Apply priority to cost calculation
    shuffled_costs = buy_cost_if_opening[priority_order]
    cumulative_cost = jnp.cumsum(shuffled_costs)
    can_afford = cumulative_cost <= available_cash

    # Unshuffle result
    can_afford_original = can_afford[jnp.argsort(priority_order)]
```

## Files

- `mcts_trader/jax_agent/jax_trading_env.py:385` (`fill_limit_orders`)

## Testing

```python
def test_fill_priority_random():
    """No asset should have systematic fill priority."""
    fill_counts = {i: 0 for i in range(64)}

    for _ in range(1000):
        # Create scenario where only 1 order can fill
        filled = env.fill_with_limited_cash()
        fill_counts[filled] += 1

    # Distribution should be roughly uniform
    assert max(fill_counts.values()) < 2 * min(fill_counts.values())
```
