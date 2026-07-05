# TASK_129: Path Solver Static Shape Mismatch Fix

## Issue Analysis
The `greedy_lookahead_path` solver was recently optimized to use a **static internal scan length** (`max_len=512`) to prevent JIT recompilation issues when sequence length `T` varies (the "dynamic T" bug).

**The Problem:**
While the internal scan is now efficient, the function currently returns the **full padded result** (length 512).
The downstream consumer, `trainer.py`'s `optimal_path_to_policy` (used for Imprinting), expects the optimal path to match the input `returns` array (length `T`, typically 400).

**The Crash:**
```python
# mcts_trader/jax_agent/trainer.py
optimal_policy_targets = optimal_path_to_policy(
    optimal_path.selections,  # Shape (512, 64) from solver
    returns_for_amax[0],      # Shape (400, 64) from batch
    ...
)

# mcts_trader/jax_agent/path_solver.py
def optimal_path_to_policy(selections, returns, ...):
    buy_mask = selections * (returns > 0)  # CRASH: Shape mismatch (512 vs 400)
```

## Criticality
- **High**: This will crash training whenever `--imprint` is enabled (which is the default for stable training).
- **Performance**: The fix preserves the performance benefit of the static scan while restoring API compatibility.

## Proposed Changes

### 1. `mcts_trader/jax_agent/path_solver.py`
Modify `greedy_lookahead_path` to slice the results back to `T` before returning.

```python
def greedy_lookahead_path(..., max_len=512):
    # ... existing static scan logic ...
    
    # OLD: Return full padded arrays
    # return OptimalPath(selections=all_masks, ...)
    
    # NEW: Slice back to T
    selections = all_masks[:T]
    indices = all_indices[:T]
    cumulative_pnl = cumulative_pnl[:T]
    
    return OptimalPath(
        selections=selections,
        indices=indices,
        cumulative_pnl=cumulative_pnl,
        total_alpha=total_alpha,
        trade_count=trade_count,
    )
```

## Verification
- **Input:** `returns` of shape `(400, 64)`
- **Internal:** Pads to `(512, 64)`, scans 512 steps (static graph).
- **Output:** Slices back to `(400, 64)`.
- **Result:** Downstream functions receive correct shapes. JIT recompilation is avoided because `T` is usually constant (400) during training. If `T` varies, only the cheap slicing operation changes, not the heavy scan loop.

## Next Steps
Apply the fix to `mcts_trader/jax_agent/path_solver.py`.
