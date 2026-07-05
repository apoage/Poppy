# TASK 007: Add Randomization to Level 0 Sine Wave Generator

**Severity**: LOW
**Location**: `mcts_trader/data/synthetic_market.py:124-156`
**Status**: DONE (2025-12-29)
**Estimated Complexity**: Low

---

## Problem Description

Level 0 synthetic market uses **fully deterministic** phase offsets, meaning every episode generates **identical** price trajectories. This limits exploration diversity and may cause overfitting to specific patterns.

### Current Implementation (`synthetic_market.py:137`):
```python
def _level_0_sine_waves(self) -> Dict[str, Dict[str, np.ndarray]]:
    for i in range(self.n_assets):
        period = self.rng.randint(20, 60)      # Random period (OK)
        amplitude = self.rng.uniform(0.08, 0.15)  # Random amplitude (OK)

        # ISSUE: Deterministic phase based on asset index!
        phase = (i * np.pi / 4) % (2 * np.pi)  # Always same for asset i

        # This creates identical trajectories every reset
```

### Comment Explaining Original Intent (line 135-136):
```python
# FIX: Use deterministic phase distribution instead of random
# This ensures mix of rising/falling assets at any given time
```

The original intent was valid (ensure diversity of directions), but the fix went too far by removing ALL randomness.

---

## Impact

1. **Same episode every time**: Agent sees identical price patterns
2. **Overfitting risk**: May memorize specific trajectory instead of learning general patterns
3. **No exploration variety**: Cannot discover alternative strategies
4. **Misleading early success**: Perfect performance on Level 0 may not generalize

---

## Proposed Solution

### Option A: Add Random Phase Offset (Recommended)

Keep the deterministic base phase (for diversity) but add a random offset:

```python
def _level_0_sine_waves(self) -> Dict[str, Dict[str, np.ndarray]]:
    # Random offset applied to ALL assets (shifts entire market)
    global_phase_offset = self.rng.uniform(0, 2 * np.pi)

    for i in range(self.n_assets):
        period = self.rng.randint(20, 60)
        amplitude = self.rng.uniform(0.08, 0.15)

        # Deterministic base phase (ensures direction diversity)
        base_phase = (i * np.pi / 4) % (2 * np.pi)

        # Add random offset (creates episode variety)
        phase = base_phase + global_phase_offset

        # Per-asset random offset for additional variety (optional)
        # phase += self.rng.uniform(-np.pi/8, np.pi/8)
```

### Option B: Fully Random Phases (Simple)

Return to original random behavior:

```python
phase = self.rng.uniform(0, 2 * np.pi)  # Random per asset
```

Risk: Some episodes may have all assets in same direction.

### Option C: Stratified Random Phases (Balanced)

Ensure each "quadrant" has equal representation:

```python
# Divide assets into 4 groups, each gets different base phase
quadrant = i % 4
base_phase = quadrant * np.pi / 2  # 0, π/2, π, 3π/2
# Add small random perturbation
phase = base_phase + self.rng.uniform(-np.pi/4, np.pi/4)
```

---

## Implementation Plan (Option A)

### Step 1: Add Global Phase Offset (5 min)

```python
# synthetic_market.py

def _level_0_sine_waves(self) -> Dict[str, Dict[str, np.ndarray]]:
    data = {}

    # NEW: Random global offset for episode variety
    global_phase_offset = self.rng.uniform(0, 2 * np.pi)

    for i in range(self.n_assets):
        period = self.rng.randint(20, 60)
        amplitude = self.rng.uniform(0.08, 0.15)

        # Deterministic base (direction diversity) + random offset (episode variety)
        base_phase = (i * np.pi / 4) % (2 * np.pi)
        phase = base_phase + global_phase_offset

        # ... rest unchanged
```

### Step 2: Verify Episodes Are Different (5 min)

```python
# Quick test
from mcts_trader.data.synthetic_market import SyntheticMarket

sm = SyntheticMarket(level=0, n_assets=4, n_steps=100)
prices_1 = sm.data['SYN_00']['close'].copy()

sm.reset()  # Or create new instance
sm2 = SyntheticMarket(level=0, n_assets=4, n_steps=100)
prices_2 = sm2.data['SYN_00']['close']

# Should be DIFFERENT now
print(f"Episodes identical: {np.allclose(prices_1, prices_2)}")
```

### Step 3: Add Per-Asset Jitter (Optional, 5 min)

For more variety, add small per-asset random offset:

```python
per_asset_jitter = self.rng.uniform(-np.pi/8, np.pi/8)
phase = base_phase + global_phase_offset + per_asset_jitter
```

---

## Acceptance Criteria

- [ ] Level 0 generates different episodes on each reset
- [ ] Direction diversity is maintained (mix of rising/falling)
- [ ] Training still converges on Level 0
- [ ] No performance regression

---

## Files to Modify

1. `mcts_trader/data/synthetic_market.py` - Add random offset (lines 124-156)

---

## Testing

```bash
# Test 1: Visual inspection
python -c "
from mcts_trader.data.synthetic_market import SyntheticMarket
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for idx, ax in enumerate(axes.flat):
    sm = SyntheticMarket(level=0, n_assets=4, n_steps=200, seed=None)
    for i in range(4):
        ax.plot(sm.data[f'SYN_{i:02d}']['close'], label=f'Asset {i}')
    ax.set_title(f'Episode {idx+1}')
    ax.legend()
plt.tight_layout()
plt.savefig('level0_episodes.png')
print('Saved level0_episodes.png')
"

# Verify: 4 plots should show DIFFERENT patterns

# Test 2: Training convergence
python train_jax_fast.py --synthetic-level 0 --epochs 50 --name test_random_phase
# Should still converge (reward increasing)
```

---

## Risk Assessment

**Risk**: Adding too much randomness could make Level 0 harder, affecting curriculum progression.

**Mitigation**:
- Use small global offset (maintains overall structure)
- Keep deterministic base phases (ensures direction diversity)
- Test curriculum completion before/after change

---

## Alternative Consideration

If full determinism was intentional for **debugging/reproducibility**, consider:

```python
class SyntheticMarket:
    def __init__(self, ..., deterministic: bool = False):
        self.deterministic = deterministic

    def _level_0_sine_waves(self):
        if self.deterministic:
            # Old behavior for debugging
            phase = (i * np.pi / 4) % (2 * np.pi)
        else:
            # New behavior for training
            phase = base_phase + global_offset
```

This allows both use cases.
