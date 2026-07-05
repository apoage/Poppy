# TASK_054: Position Limits Include Pending Orders

## Status: COMPLETED (2026-01-23)

**Implementation**: `--max-positions` CLI flag
- Located in: `cli.py`, `config.py`, `vectorbt_env.py`, `jax_trading_env.py`
- Usage: `--max-positions 16` limits simultaneous positions
## Priority: MEDIUM

## Problem

With lazy execution, agent can bypass position limits:
- Open 16 positions (max_positions limit)
- Place 48 more limit orders (no cash reserved)
- Effective exposure to 64 assets through order "options"

## Solution

```python
def check_position_limit(state, config):
    n_positions = jnp.sum(state.pos_active)
    n_pending = jnp.sum(state.order_active)
    total_exposure = n_positions + n_pending

    return total_exposure < config.max_positions
```

## Files

- `jax_trading_env.py:1231-1233`
- `vectorbt_env.py` (similar check)

## Alternative

Instead of counting orders, limit total order VALUE:

```python
pending_value = jnp.sum(state.order_active * state.order_price * state.order_quantity)
max_pending_value = 0.5 * state.portfolio_value  # Max 50% in pending orders
```
