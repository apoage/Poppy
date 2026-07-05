# Bug: Action Decoding Mismatch (3 vs 5 Actions)

## Priority: CRITICAL
**Status:** DONE

## Fix Applied (2026-01-01)

Changed `market_ode.py:77-82`:
- Added `ACTIONS_PER_ASSET = 5` constant
- Changed `action // 3` to `action // ACTIONS_PER_ASSET`
- Changed `action % 3` to `action % ACTIONS_PER_ASSET`
- Updated impact_sign to handle LimitSell (3) and LimitBuy (4) actions
**Component:** `mcts_trader/jax_agent/prediction/market_ode.py`

## Problem Description

The market ODE predictor uses 3-action decoding while the environment uses 5 actions per asset. This causes incorrect action interpretation for limit orders.

### Location

```python
# market_ode.py:77-78
def _decode_action(self, action_idx: int, n_assets: int) -> Tuple[int, int]:
    asset_idx = action_idx // 3   # WRONG: should be // 5
    action_type = action_idx % 3  # WRONG: should be % 5
```

### Environment Action Space (vectorbt_env.py)

```python
# 5 actions per asset:
# action % 5 == 0: Sell (market)
# action % 5 == 1: Hold
# action % 5 == 2: Buy (market)
# action % 5 == 3: Limit Sell
# action % 5 == 4: Limit Buy
```

### Impact

- Prediction model receives wrong action signals
- Limit order actions (3, 4) are incorrectly decoded
- Physics simulation uses wrong trading signals
- Prediction loss learns incorrect dynamics

### Example of Mismatch

```python
action = 13  # Asset 2, Limit Sell (13 = 2*5 + 3)

# Current (WRONG):
asset_idx = 13 // 3 = 4   # Wrong asset!
action_type = 13 % 3 = 1  # Interpreted as Hold!

# Correct:
asset_idx = 13 // 5 = 2   # Correct asset
action_type = 13 % 5 = 3  # Limit Sell
```

## Solution

```python
# market_ode.py:77-78
def _decode_action(self, action_idx: int, n_assets: int) -> Tuple[int, int]:
    asset_idx = action_idx // 5   # Fixed: 5 actions per asset
    action_type = action_idx % 5  # Fixed: 5 action types
    return asset_idx, action_type
```

Also update any constants:

```python
# Add at top of file
ACTIONS_PER_ASSET = 5  # Sell, Hold, Buy, LimitSell, LimitBuy
```

## Files to Modify

| File | Change |
|------|--------|
| `market_ode.py:77-78` | Change `// 3` to `// 5` and `% 3` to `% 5` |

## Testing

1. Verify action decoding for all 5 action types
2. Check limit order actions (3, 4) decode correctly
3. Run prediction model and verify action signals

## Verification

```python
# Test action decoding
for action in range(320):  # 64 assets * 5 actions
    asset = action // 5
    atype = action % 5
    assert asset == action // 5
    assert atype in [0, 1, 2, 3, 4]
    print(f"Action {action}: Asset {asset}, Type {atype}")
```
