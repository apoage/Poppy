# Bug: Level 4 Rotation Generator Orphaned

## Priority: MEDIUM
**Status:** FALSE POSITIVE

## Verification (2026-01-01)

Level 4 is properly registered at line 60:
`4: self._level_4_portfolio_rotation`

The `_level_4_portfolio_rotation` method exists at line 263.
**Component:** `mcts_trader/data/synthetic_market.py`

## Problem Description

The Level 4 "rotation" generator exists but may not be properly connected in the training pipeline. Need to verify it's accessible via `--synthetic-level 4`.

### Context

Level 4 is the most advanced synthetic curriculum:
- Sector rotation patterns
- Multi-asset correlation changes
- Timing challenges

### Potential Issues

1. Generator not registered in level map
2. Generator produces wrong shape
3. Missing from curriculum progression

## Solution

### Step 1: Verify registration

```python
# synthetic_market.py
LEVEL_GENERATORS = {
    0: generate_sine_waves,
    1: generate_lagged_pairs,
    2: generate_dip_crashes,
    3: generate_noisy_trends,
    4: generate_rotation,  # Verify this exists
}
```

### Step 2: Test generation

```python
from mcts_trader.data.synthetic_market import generate_synthetic_market

data = generate_synthetic_market(level=4, n_assets=64, n_steps=1440)
assert data.shape == (1440, 64)
```

## Files to Modify

| File | Change |
|------|--------|
| `synthetic_market.py` | Verify Level 4 registration and output |

## Testing

1. Run `python train_jax_fast.py --synthetic-level 4 --epochs 1`
2. Verify training starts without errors
3. Check generated data has rotation patterns
