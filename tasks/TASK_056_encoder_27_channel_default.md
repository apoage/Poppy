# TASK_056: Use 27-Channel Encoder by Default

## Status: COMPLETED (2026-01-23)

**Implementation**: 27-channel encoder is now the default
- Located in: `mcts_trader/encoder/state_fast.py` (line 31)
- Observation shape: (27, n_assets, n_assets)
## Priority: LOW

## Problem

JAX env simple encoder (5 channels) omits volume and high/low details.
Agent blind to volatility regimes and order imbalance.

## Solution

Make 27-channel encoder the default:

```python
# jax_trading_env.py
parser.add_argument('--encoder-channels', type=int, default=27,
                   help='Number of observation channels (5=simple, 27=full)')
```

## 27-Channel Breakdown

```
Channels 0-4:   OHLCV normalized
Channels 5-9:   Returns (1, 5, 10, 20, 60 step)
Channels 10-14: Volatility features
Channels 15-19: Volume features
Channels 20-24: Technical indicators
Channels 25-26: Position/PnL info
```

## Files

- `jax_trading_env.py:630` (encoder selection)
- `mcts_trader/encoder/state.py` (full encoder)

## Note

May need to verify full encoder works with JAX env - currently untested combination.
