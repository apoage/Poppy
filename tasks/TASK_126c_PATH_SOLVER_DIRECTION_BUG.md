# TASK_126c: Path Solver Direction-Aware Transaction Cost Bug

**Status**: DIAGNOSED (Fix Ready)
**Priority**: High
**Created**: 2026-01-31

## Problem Statement

The oracle's `optimal_alpha` is sometimes LESS than `market_return` (buy-and-hold), which is mathematically impossible since the oracle knows the future.

**Bug Location**: `mcts_trader/jax_agent/path_solver.py:75-102`

## Root Cause

### Issue 1: `signs` Computed But Not Used

```python
# Lines 92-96
step_returns = returns[t]
signs = jnp.sign(step_returns)  # <-- COMPUTED
step_pnl = jnp.sum(jnp.abs(step_returns) * new_mask)  # <-- signs NOT USED!
```

The comment says "sign-aware" but the code doesn't actually use the `signs` variable.

### Issue 2: Binary Mask Doesn't Track Direction

```python
# Line 90: mask is binary (0 or 1), not directional (-1, 0, +1)
new_mask = jnp.zeros(n_assets).at[top_k_indices].set(1.0)
```

The mask only tracks WHICH assets are selected, not WHETHER the position is long or short.

### Issue 3: Transaction Costs Miss Direction Flips

```python
# Lines 99-100: only counts asset add/remove, NOT long↔short flips!
changes = jnp.sum(jnp.abs(new_mask - prev_mask))
step_cost = changes * transaction_cost
```

**Critical Bug**: When an asset stays selected but needs to flip direction (long→short or short→long), the current code counts **zero** transaction cost!

## Bug Demonstration

| Step | Asset A Return | Correct Action | Current Mask | Computed Cost | Actual Cost |
|------|----------------|----------------|--------------|---------------|-------------|
| t    | +5%            | LONG           | 1            | 1 (enter)     | 1           |
| t+1  | -3%            | SHORT (flip!)  | 1            | **0 (BUG!)**  | **2**       |

The oracle needs to flip from long→short, which requires:
- Exit long position (1 transaction)
- Enter short position (1 transaction)
- **Total: 2 transactions**

But current code sees `|1 - 1| = 0` changes because the binary mask didn't change.

## Impact

1. **Understated transaction costs** → Oracle appears more profitable than it should be
2. **But paradoxically**: The oracle can underperform buy-and-hold because the PnL calculation may also be affected by the direction mismatch
3. **Inconsistent comparisons**: Other methods (`soft_topk_path`, `sinkhorn_topk_path`) also have this issue and additionally divide by `k` (normalization inconsistency)

## Correct Oracle Behavior

The oracle should:
1. Pick top-k assets by `|return|` (highest magnitude)
2. Go **LONG** if `return > 0`
3. Go **SHORT** if `return < 0`
4. Capture `|return|` for each selected asset
5. Pay transaction costs for ALL position changes including direction flips

## Proposed Fix

### Change 1: Use Signed Mask

```python
# BEFORE (line 90):
new_mask = jnp.zeros(n_assets).at[top_k_indices].set(1.0)

# AFTER:
selection_mask = jnp.zeros(n_assets).at[top_k_indices].set(1.0)
signs = jnp.sign(step_returns)
signs = jnp.where(signs == 0, 1.0, signs)  # Default to long for zero returns
new_signed_mask = selection_mask * signs  # -1 short, 0 none, +1 long
```

### Change 2: Direction-Aware PnL

```python
# BEFORE (line 96):
step_pnl = jnp.sum(jnp.abs(step_returns) * new_mask)

# AFTER:
step_pnl = jnp.sum(step_returns * new_signed_mask)
# This equals sum(|returns|) when directions are optimal
```

### Change 3: Direction-Aware Transaction Costs

```python
# BEFORE (line 99):
changes = jnp.sum(jnp.abs(new_mask - prev_mask))

# AFTER:
changes = jnp.sum(jnp.abs(new_signed_mask - prev_signed_mask))
```

**Transaction cost examples with signed mask:**
| Transition | Calculation | Cost |
|------------|-------------|------|
| Enter long | \|+1 - 0\| | 1 |
| Enter short | \|-1 - 0\| | 1 |
| Exit long | \|0 - (+1)\| | 1 |
| Exit short | \|0 - (-1)\| | 1 |
| Long→Short | \|-1 - (+1)\| | **2** |
| Short→Long | \|+1 - (-1)\| | **2** |
| Hold long | \|+1 - (+1)\| | 0 |
| Hold short | \|-1 - (-1)\| | 0 |

### Change 4: Update Carry and Return

```python
# BEFORE:
def step(prev_mask, t):
    ...
    return new_mask, (new_mask, top_k_indices, step_pnl - step_cost, changes)

# Initialize with zeros (no positions)
init_mask = jnp.zeros(n_assets)

# AFTER:
def step(prev_signed_mask, t):
    ...
    return new_signed_mask, (selection_mask, top_k_indices, step_pnl - step_cost, changes)

# Initialize with zeros (no positions)
init_signed_mask = jnp.zeros(n_assets)
```

## Full Fixed Function

```python
def step(prev_signed_mask, t):
    """Single greedy step with lookahead."""
    # Get future returns for lookahead window (safe due to padding)
    future_returns = jax.lax.dynamic_slice(
        returns_padded, (t, 0), (lookahead, n_assets)
    )

    # Score = sum of |returns| over lookahead (greedy for magnitude)
    cumulative_abs_returns = jnp.sum(jnp.abs(future_returns), axis=0)

    # Get top-k by cumulative absolute returns
    top_k_indices = jax.lax.top_k(cumulative_abs_returns, k)[1]

    # Create binary selection mask
    selection_mask = jnp.zeros(n_assets).at[top_k_indices].set(1.0)

    # Compute SIGNED mask: +1 for long, -1 for short
    step_returns = returns[t]
    signs = jnp.sign(step_returns)
    signs = jnp.where(signs == 0, 1.0, signs)  # Default to long for zero
    new_signed_mask = selection_mask * signs

    # PnL = sum(return * signed_position) = sum(|return|) for correct directions
    step_pnl = jnp.sum(step_returns * new_signed_mask)

    # Transaction cost - correctly counts direction flips!
    changes = jnp.sum(jnp.abs(new_signed_mask - prev_signed_mask))
    step_cost = changes * transaction_cost

    return new_signed_mask, (selection_mask, top_k_indices, step_pnl - step_cost, changes)
```

## Additional Issues Found

### Normalization Inconsistency

| Method | PnL Calculation | Normalized? |
|--------|-----------------|-------------|
| `greedy_lookahead_path` | `jnp.sum(...)` | **NO** |
| `soft_topk_path` | `jnp.sum(...) / k` | YES |
| `sinkhorn_topk_path` | `jnp.sum(...) / k` | YES |

This inconsistency can cause comparison issues. Consider normalizing all methods the same way.

### Same Bug in Other Methods

Both `soft_topk_path` (line 193) and `sinkhorn_topk_path` (line 312) have similar issues:
```python
step_pnl = jnp.sum(abs_returns * selection) / k
```

They use `abs_returns` but don't track signed positions for transaction costs.

## Verification Steps

After applying fix:
1. Oracle `optimal_alpha` should **always** >= `market_return` for any price series
2. Transaction costs should be higher when assets require frequent direction flips
3. Run regression test comparing oracle vs buy-and-hold on synthetic data

## Related Tasks

- TASK_119: Optimal Path Solver (original implementation)
- TASK_126a: Reward Spike Investigation (related numerical issues)
