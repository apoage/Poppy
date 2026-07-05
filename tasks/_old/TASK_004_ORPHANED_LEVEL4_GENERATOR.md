# TASK 004: Fix Orphaned Level 4 Generator (Portfolio Rotation)

**Severity**: LOW
**Location**: `mcts_trader/data/synthetic_market.py:257-300`, `dojo_dynamic_curriculum.py:221-246`
**Status**: DONE (2025-12-29)
**Estimated Complexity**: Low

---

## Problem Description

The synthetic market generator has a **Level 4: Portfolio Rotation** implementation, but it is NOT included in the curriculum definition. This is either:

1. **Dead code** that should be removed
2. **Incomplete feature** that should be added to curriculum
3. **Advanced level** intentionally held back for later

### Level 4 Implementation (`synthetic_market.py:257-300`):
```python
def _level_4_portfolio_rotation(self) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Level 4: Rotating leadership.
    """
    SECTOR_SIZE = 8
    N_SECTORS = self.n_assets // SECTOR_SIZE
    ROTATION_PERIOD = 50
    PUMP_MAGNITUDE = 0.03
    # ... implementation exists and is complete
```

### Curriculum Definition (`dojo_dynamic_curriculum.py:221-246`):
```python
def create_dynamic_curriculum() -> List[Dict]:
    return [
        {'level': 0, 'generator': 'sine', ...},
        {'level': 1, 'generator': 'lagged', ...},
        {'level': 2, 'generator': 'dip', ...},
        {'level': 3, 'generator': 'noisy_trends', ...},
        # Level 4 is MISSING!
    ]
```

---

## Root Cause Analysis

The generator routes to Level 4 correctly:
```python
generators = {
    0: self._level_0_sine_waves,
    1: self._level_1_lagged_pairs,
    2: self._level_2_dip_crashes,
    3: self._level_3_noisy_trends,
    4: self._level_4_portfolio_rotation  # Exists in routing
}
```

But the curriculum manager never requests Level 4, making it unreachable through normal training flow.

---

## Proposed Solution

### Option A: Add Level 4 to Curriculum (Recommended)

This is the most valuable option since Portfolio Rotation is a sophisticated test:
- Tests sector/cluster awareness
- Tests timing of rotations
- Tests multi-asset coordination

```python
# dojo_dynamic_curriculum.py

def create_dynamic_curriculum() -> List[Dict]:
    return [
        {'level': 0, 'generator': 'sine', 'tactics': ['basic'], ...},
        {'level': 1, 'generator': 'lagged', 'tactics': ['basic'], ...},
        {'level': 2, 'generator': 'dip', 'tactics': ['basic'], ...},
        {'level': 3, 'generator': 'noisy_trends', 'tactics': ['basic', 'morphing'], ...},
        {
            'level': 4,
            'generator': 'rotation',
            'tactics': ['basic', 'morphing', 'sector_tracking'],
            'description': 'Portfolio rotation - progressive difficulty'
        },
    ]
```

### Option B: Remove Dead Code

If Level 4 is not intended for use, remove it:
```python
# synthetic_market.py - Remove lines 257-300
# Remove from generators dict
```

### Option C: Mark as Experimental

Add comment indicating it's reserved for future:
```python
def _level_4_portfolio_rotation(self):
    """
    Level 4: Rotating leadership.

    NOTE: This level is experimental and not included in the default
    curriculum. Use --synthetic-level 4 directly for testing.
    """
```

---

## Implementation Plan (Option A)

### Step 1: Add Level 4 to Curriculum (5 min)

```python
# dojo_dynamic_curriculum.py

def create_dynamic_curriculum() -> List[Dict]:
    return [
        # ... existing levels 0-3 ...
        {
            'level': 4,
            'generator': 'rotation',
            'tactics': ['basic', 'morphing'],
            'description': 'Portfolio rotation - sector leadership cycles'
        },
    ]
```

### Step 2: Verify Generator Works (10 min)

```bash
# Test Level 4 directly
python train_jax_fast.py --synthetic-level 4 --epochs 10 --name test_level4

# Check output for errors
# Verify price data looks reasonable
```

### Step 3: Add Level 4 Specific Thresholds (Optional, 10 min)

Level 4 may need different mastery thresholds due to complexity:

```python
# Consider adjusting difficulty_phases for Level 4
# Or add level-specific threshold adjustments
```

### Step 4: Update Documentation (5 min)

```markdown
# In CLAUDE.md or README

## Synthetic Market Levels

| Level | Name | Description |
|-------|------|-------------|
| 0 | Sine Waves | Deterministic oscillations |
| 1 | Lagged Pairs | Leader-follower correlation |
| 2 | Dip Crashes | V-shaped flash crashes |
| 3 | Noisy Trends | Random walk with drift |
| 4 | Portfolio Rotation | Sector leadership rotation |  <- NEW
```

---

## Acceptance Criteria

- [ ] Level 4 accessible via curriculum OR explicitly marked as experimental
- [ ] If added to curriculum: training runs successfully on Level 4
- [ ] Generator produces valid OHLC data (no NaN, no negative prices)
- [ ] Documentation updated

---

## Files to Modify

1. `dojo_dynamic_curriculum.py` - Add Level 4 to curriculum list
2. `CLAUDE.md` or documentation - Update level descriptions

---

## Testing

```bash
# Test 1: Direct access
python train_jax_fast.py --synthetic-level 4 --epochs 5

# Test 2: Via Dojo (after adding to curriculum)
python dojo_dynamic_curriculum.py --level 4

# Test 3: Verify data quality
python -c "
from mcts_trader.data.synthetic_market import SyntheticMarket
sm = SyntheticMarket(level=4, n_assets=64, n_steps=1000)
print('Level 4 data generated successfully')
print(f'Assets: {len(sm.data)}')
print(f'Steps: {sm.n_steps}')
"
```

---

## Level 4 Behavior Summary

Portfolio Rotation creates:
- 8 sectors of 8 assets each (64 total)
- Every 50 steps, leadership rotates to next sector
- Leading sector gets +3% "pump" factor
- Tests agent's ability to detect and follow sector rotation

This is a valuable test for:
- Multi-asset awareness
- Timing/momentum strategies
- Sector correlation understanding
