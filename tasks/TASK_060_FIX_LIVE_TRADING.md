# TASK_060: Fix Live Trading Script

## Status: TODO

## Priority: HIGH (blocks live deployment)

## Problem

`real_trade_jax.py` imports a missing module:

```python
from mcts_trader.agent.real_portfolio import RealPortfolioManager
```

But `mcts_trader/agent/real_portfolio.py` does not exist.

## Current State

```
mcts_trader/agent/
├── __init__.py
├── improved_model.py
├── trm.py
└── wrapper.py
# NO real_portfolio.py!
```

## Root Cause

File was likely deleted during cleanup or never created. The import suggests a `RealPortfolioManager` class that handles:
- Live position tracking
- Order execution via Kraken API
- Cash/margin management
- Risk controls (50% reserve, panic button)

## Required Functionality

Based on `real_trade_jax.py` usage, `RealPortfolioManager` needs:

1. **Position Tracking**
   - Track open positions across assets
   - Sync with Kraken account state
   - Handle partial fills

2. **Order Execution**
   - Market orders (immediate)
   - Limit orders (with timeout)
   - Order cancellation

3. **Risk Management**
   - 50% cash reserve (configurable)
   - Max position size limits
   - Drawdown hard stop
   - Panic button (close all)

4. **State Sync**
   - Periodic Kraken balance query
   - Handle disconnects gracefully
   - Reconcile local vs exchange state

## Implementation Options

### Option A: Restore from Git History
```bash
git log --all --full-history -- "**/real_portfolio.py"
git checkout <commit> -- mcts_trader/agent/real_portfolio.py
```

### Option B: Create New Implementation
Use existing components:
- `mcts_trader/data/kraken_client.py` - API wrapper
- `config/real_trade_config.py` - Configuration
- `mcts_trader/env/vectorbt_env.py` - Reference for position logic

### Option C: Adapt VectorBT Portfolio
Extract `Portfolio` class from `vectorbt_env.py` and add:
- Kraken API calls instead of simulation
- Real balance queries
- Order submission

## Acceptance Criteria

- [ ] `real_trade_jax.py` imports without error
- [ ] DRY_RUN=True mode logs orders without executing
- [ ] DRY_RUN=False mode executes real orders on Kraken
- [ ] 50% reserve is enforced
- [ ] Panic button closes all positions
- [ ] Handles API disconnects gracefully

## Dependencies

- `config/real_trade_config.py` (exists, updated 2026-01-15)
- `mcts_trader/data/kraken_client.py` (exists)
- Kraken API credentials in environment

## Notes

This was flagged by Gemini during Operation Reality Check (2026-01-15).
Live trading is blocked until this is resolved.

---

*Created: 2026-01-15*
*Source: Gemini code review session*
