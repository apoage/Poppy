# TASK_126h: Path Solver PnL Bug Fix

## Problem Statement

The hindsight-optimal oracle's `optimal_alpha` is sometimes **LESS than `market_return`** (buy-and-hold), which is mathematically impossible - the oracle knows the future and should always beat or match the market.

**Example from logs:**
```
epoch=27 reward=-30.991934 alpha=-4.624 market_return=2.36 optimal_alpha=3.15
```

Here `optimal_alpha=3.15` but `market_return=2.36`, meaning the oracle only captured 3.15% vs the market's 2.36%. The oracle should have captured **at least** `market_return` by simply going long all assets.

---

## Root Cause Analysis

### The Bug: Signs Are Ignored in PnL Calculation

**Location:** `mcts_trader/jax_agent/path_solver.py`

#### Bug in `greedy_lookahead_path()` (lines 92-96)

```python
# Compute actual PnL for this step (sign-aware)
# For simplicity, assume we go long on positive returns, short on negative
step_returns = returns[t]
signs = jnp.sign(step_returns)  # <-- COMPUTED but NEVER USED!
step_pnl = jnp.sum(jnp.abs(step_returns) * new_mask)  # <-- ALWAYS POSITIVE!
```

**The problem:**
1. `signs` is computed but **never applied** to the PnL calculation
2. `jnp.abs(step_returns)` is always positive
3. The mask `new_mask` selects which assets to trade, but **not which direction**

This means the oracle assumes it always makes money on selected assets, even when it picks the wrong direction!

### Same Bug in All Three Methods

| Method | Location | Bug |
|--------|----------|-----|
| `greedy_lookahead_path()` | Line 96 | `jnp.sum(jnp.abs(step_returns) * new_mask)` |
| `soft_topk_path()` | Line 193 | `jnp.sum(abs_returns * selection) / k` |
| `sinkhorn_topk_path()` | Line 312 | `jnp.sum(abs_returns * selection) / k` |

All three methods use `abs_returns` for PnL calculation, ignoring the critical sign information.

---

## Why This Causes `optimal_alpha < market_return`

### Correct Oracle Behavior

The oracle should:
1. **Go LONG** on assets with **positive** returns
2. **Go SHORT** on assets with **negative** returns

**Correct PnL formula:**
```
For each asset i:
  If selected[i] and returns[i] > 0: PnL += returns[i]  (long gains)
  If selected[i] and returns[i] < 0: PnL += -returns[i] (short gains)
  If not selected: PnL += 0
```

Mathematically: `PnL = jnp.sum(jnp.abs(returns) * mask)` is correct **ONLY IF** the mask encodes direction (positive=long, negative=short).

But currently:
- `new_mask` is binary (0 or 1) - it only says WHICH assets, not WHICH DIRECTION
- The code assumes: "I selected this asset, so I must have picked the right direction"
- Reality: The selection is based on `cumulative_abs_returns` which loses sign information

### Example of the Bug

```python
# Two assets, one time step
returns = jnp.array([0.05, -0.03])  # Asset 0: +5%, Asset 1: -3%

# Oracle selects top-k by |return| → selects both (k=2)
new_mask = jnp.array([1.0, 1.0])

# BUGGY: Always positive
current_pnl = jnp.sum(jnp.abs(returns) * new_mask)  # = 0.05 + 0.03 = 0.08 (8%)

# CORRECT: Need to track direction
# If long asset 0 (+5%) and short asset 1 (-3% → +3%)
correct_pnl = 0.05 + 0.03  # = 0.08 (8%) ✓

# But what if selection was wrong?
# If long both: 0.05 + (-0.03) = 0.02 (2%) - WORSE than market average!
```

The current code assumes perfect direction selection, but the selection criteria (`cumulative_abs_returns`) doesn't encode direction preference.

---

## The Fix

### Option 1: Track Direction in Selection (Recommended)

Modify the mask to encode direction (positive=long, negative=short):

```python
# In greedy_lookahead_path():
# After selecting top-k by |return|, determine direction per asset
top_k_indices = jax.lax.top_k(cumulative_abs_returns, k)[1]

# Create directional mask: +1 for long, -1 for short, 0 for not selected
directional_mask = jnp.zeros(n_assets)
for idx in top_k_indices:
    # Go long if expected return is positive, short if negative
    direction = jnp.sign(jnp.mean(future_returns[:, idx]))
    directional_mask = directional_mask.at[idx].set(direction)

# Compute PnL: sum of |returns| * direction_mask * selection
step_pnl = jnp.sum(step_returns * directional_mask)
```

### Option 2: Simpler Fix (Immediate)

For the greedy method, fix the PnL calculation to use actual returns with sign:

```python
# Line 96 in greedy_lookahead_path()
# OLD (buggy):
step_pnl = jnp.sum(jnp.abs(step_returns) * new_mask)

# NEW (correct):
# The mask should have been directional, but it's binary
# For now, compute PnL as if we picked the optimal direction
directional_pnl = jnp.where(
    new_mask > 0,
    jnp.abs(step_returns),  # We "magically" picked the right direction
    0.0
)
step_pnl = jnp.sum(directional_pnl)
```

Wait - that's what the code already does! The real issue is:

### The Real Fix: Selection Must Consider Direction

The fundamental issue is that `new_mask` is computed from `cumulative_abs_returns` which is always positive. The selection process needs to:

1. **Score assets by potential PnL** (not just |return|)
2. **Include direction in the selection mask**

**Corrected greedy_lookahead_path() step function:**

```python
def step(prev_mask, t):
    """Single greedy step with lookahead (FIXED)."""
    future_returns = jax.lax.dynamic_slice(
        returns_padded, (t, 0), (lookahead, n_assets)
    )
    
    # Compute expected PnL for each asset (direction-aware)
    # For each asset, if we go long: PnL = returns
    # If we go short: PnL = -returns
    # We choose direction based on sign of expected return
    expected_returns = jnp.mean(future_returns, axis=0)  # (n_assets,)
    
    # Best possible PnL for each asset (by choosing optimal direction)
    best_case_pnl = jnp.abs(expected_returns)
    
    # Select top-k by best_case_pnl
    top_k_indices = jax.lax.top_k(best_case_pnl, k)[1]
    
    # Create directional mask: +1 for long, -1 for short
    directional_mask = jnp.zeros(n_assets)
    for i, idx in enumerate(top_k_indices):
        direction = jnp.sign(expected_returns[idx])
        # If expected return is 0, default to long
        direction = jnp.where(direction == 0, 1.0, direction)
        directional_mask = directional_mask.at[idx].set(direction)
    
    # Compute actual PnL using CURRENT step returns and DIRECTIONAL mask
    step_returns = returns[t]
    step_pnl = jnp.sum(step_returns * directional_mask)
    
    # Transaction cost based on position changes
    changes = jnp.sum(jnp.abs(directional_mask - prev_mask))
    step_cost = changes * transaction_cost
    
    return directional_mask, (directional_mask, top_k_indices, step_pnl - step_cost, changes)
```

### Fix for soft_topk_path() and sinkhorn_topk_path()

```python
# In soft_topk_path() step function (line 181):
def step(prev_selection, returns_t):
    """Single step with transition penalty (FIXED)."""
    # Use signed returns directly (not abs)
    # Selection scores based on |returns| but PnL uses actual returns
    abs_returns = jnp.abs(returns_t)
    
    # Penalize switching
    continuity_bonus = transition_weight * prev_selection
    adjusted_scores = abs_returns + continuity_bonus
    
    # Soft top-k selection
    selection = soft_topk(adjusted_scores, k, temperature)
    
    # Determine direction for selected assets
    # Use sign of returns: positive=long, negative=short
    directions = jnp.sign(returns_t)
    directions = jnp.where(directions == 0, 1.0, directions)  # Default to long
    
    # PnL = returns * selection * direction
    # Selection acts as weight, direction determines sign
    step_pnl = jnp.sum(returns_t * selection * directions)
    
    # Count changes (soft)
    changes = jnp.sum(jnp.abs(selection - prev_selection))
    
    return selection, (selection, step_pnl, changes)
```

---

## Verification

After the fix, the oracle should:
1. Always have `optimal_alpha >= market_return`
2. Achieve exactly `market_return` when k = n_assets (holding all assets long)
3. Achieve higher than `market_return` when k < n_assets (selective long/short)

### Test Case

```python
import jax.numpy as jnp
from mcts_trader.jax_agent.path_solver import greedy_lookahead_path

# Simple test: 2 assets, 3 steps
# Asset 0: +5%, +3%, -2%
# Asset 1: -3%, +4%, +1%
returns = jnp.array([
    [0.05, -0.03],
    [0.03, 0.04],
    [-0.02, 0.01]
])

# Market return (equal-weight buy-and-hold)
market_return = jnp.mean(jnp.sum(returns, axis=0))  # = 0.08 (8%)

# Oracle with k=2 (should match or beat market)
optimal = greedy_lookahead_path(returns, k=2, lookahead=1)
print(f"Market return: {market_return:.4f}")
print(f"Optimal alpha: {optimal.total_alpha:.4f}")
assert optimal.total_alpha >= market_return - 1e-6, "Oracle should beat market!"
```

---

## Acceptance Criteria

- [ ] `greedy_lookahead_path()` uses directional mask for PnL calculation
- [ ] `soft_topk_path()` uses directional selection for PnL calculation  
- [ ] `sinkhorn_topk_path()` uses directional selection for PnL calculation
- [ ] Oracle always achieves `optimal_alpha >= market_return` (within numerical precision)
- [ ] Test case added: oracle vs market return comparison
- [ ] Test case added: oracle with k=n_assets equals market return
- [ ] Test case added: oracle with perfect foresight always makes positive PnL in up markets

---

## Related Code

- `mcts_trader/jax_agent/path_solver.py` - Main solver implementations
- `mcts_trader/jax_agent/trainer.py:1220-1245` - Where optimal_alpha is computed and logged
- `mcts_trader/training/self_play.py:807-811` - Where market_return is computed

---

## Notes

The comment in line 93 says "assume we go long on positive returns, short on negative" but this assumption was never actually implemented in the code. The `signs` variable was computed and then silently ignored, creating a false sense of correctness.

This bug likely caused:
- Underestimation of true optimal performance
- Incorrect regret calculations (regret = optimal - agent)
- Misleading training signals (agent appears closer to optimal than it really is)
