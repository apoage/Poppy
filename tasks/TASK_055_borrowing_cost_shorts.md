# TASK_055: Borrowing Cost for Short Positions

## Status: COMPLETED (2026-01-23)

**Implementation**: `--borrow-rate` CLI flag
- Located in: `cli.py`, `config.py`, `setup.py`, `vectorbt_env.py`, `jax_trading_env.py`
- Usage: `--borrow-rate 0.05` for 5% annual borrow rate on shorts
## Priority: MEDIUM

## Problem

Cross-asset hedging is free - no cost to short positions.
Agent can hedge perfectly without realistic borrowing costs.

## Solution

```python
# Per-step borrowing cost for short positions
borrow_rate_annual = 0.05  # 5% annual
borrow_rate_step = borrow_rate_annual / (252 * 400)  # Per step

short_value = jnp.sum(jnp.minimum(state.positions, 0) * current_prices)
borrow_cost = abs(short_value) * borrow_rate_step

reward -= borrow_cost
```

## CLI Arguments

```python
parser.add_argument('--borrow-rate', type=float, default=0.05,
                   help='Annual borrowing rate for short positions')
```

## Files

- `jax_trading_env.py` (reward calculation)
- `vectorbt_env.py` (reward calculation)

## Note

Only applies in margin mode where shorts are allowed.
