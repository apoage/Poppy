# Part 2: Oracle Math Consistency

## Status: FAIL

## Findings
- **`path_solver.py:130-131`**: Training oracle computes `step_pnl = jnp.sum(jnp.abs(step_returns) * new_mask) / k_count` - divides by k_count (number of selected assets)
- **`benchmark_oracle_vs_model.py:403`**: Benchmark oracle computes `step_pnl = np.sum(np.abs(returns[t]) * mask)` - NO division by k_count
- **`benchmark_oracle_vs_model.py:468`**: Benchmark model computes `step_pnl += np.sum(position * returns[t])` - NO normalization by number of active positions
- **Comment at line 402**: "BUG-9 fix: Use raw sum (no /k_count) to match model P&L normalization" - This indicates a deliberate decision was made, but it creates INCONSISTENCY with training

## Bugs Found

### BUG-2.1: PnL normalization mismatch between training and benchmark
**Severity**: HIGH
**Location**: `mcts_trader/jax_agent/path_solver.py:130-131` vs `tools/benchmark_oracle_vs_model.py:403`
**Description**: Training divides by `k_count` (normalized per-asset PnL), benchmark does not (total absolute PnL). This creates a fundamental inconsistency in how oracle performance is measured.

**Financial interpretation**:
- With `/k_count`: Oracle PnL represents average PnL per selected asset (per-asset average)
- Without `/k_count`: Oracle PnL represents total portfolio PnL (absolute return)

**Impact**:
1. Training optimizes for average per-asset performance, but benchmark evaluates total portfolio performance
2. More aggressive k values (selecting more assets) will have inflated benchmark PnL without `/k_count`, potentially masking poor per-asset selection
3. Model's PnL calculation (`position * returns`) also has no normalization, creating a third distinct metric

### BUG-2.2: Model PnL also unnormalized
**Severity**: MEDIUM
**Location**: `tools/benchmark_oracle_vs_model.py:468`
**Description**: Model PnL is `np.sum(position * returns[t])` - sums across all assets without any normalization. This differs from both training oracle (divided by k_count) and benchmark oracle (no division).

**Impact**: Model PnL scales with the number of positions held, making comparison between different model behaviors (conservative vs aggressive) less meaningful.

## Questions to Answer

1. **What is `k_count` in training?**
   - It's `jnp.sum(new_mask)` - the number of assets selected by the oracle at that timestep
   - With `k=4`, typically 4 assets are selected (can be fewer if alpha threshold filters them out)

2. **Which is the correct financial interpretation?**
   - The per-asset average (`/k_count`) represents the quality of EACH selection independently
   - The total PnL (no division) represents portfolio performance including position sizing
   - For a trading strategy that selects k assets equally-weighted, the per-asset average is more informative about selection quality

3. **Should benchmark normalize by k_count?**
   - YES, for consistency with training. Otherwise, the model is trained to optimize a different objective than what's measured at evaluation time.
   - Alternatively, training could REMOVE the `/k_count` if total PnL is the intended objective.

## Recommended Fixes

1. **Align benchmark oracle with training oracle** (preferred):
```python
def simulate_oracle_pnl(
    selections: np.ndarray,
    returns: np.ndarray,
    transaction_cost: float = 0.001,
) -> Tuple[float, int]:
    """Oracle-style P&L: capture abs(return) for selected assets.

    FIXED: Normalize by k_count to match training oracle (path_solver.py:130-131)
    """
    total_pnl = 0.0
    trade_count = 0

    T, n_assets = returns.shape
    prev_mask = np.zeros(n_assets)

    for t in range(T):
        mask = selections[t]
        k_count = max(np.sum(mask), 1)

        # P&L from abs returns, normalized by k_count to match training
        step_pnl = np.sum(np.abs(returns[t]) * mask) / k_count

        # Transaction costs
        changes = np.sum(np.abs(mask - prev_mask) > 0.5)
        step_pnl -= changes * transaction_cost
        trade_count += int(changes)

        total_pnl += step_pnl
        prev_mask = mask

    return total_pnl, trade_count
```

2. **Consider normalizing model PnL by number of active positions**:
```python
# In simulate_model_pnl()
for t in range(T):
    # ... existing position update code ...

    # Count active positions (non-zero position)
    n_active = max(np.sum(np.abs(position) > 0.5), 1)

    # P&L from position AFTER action, normalized by n_active
    step_pnl = np.sum(position * returns[t]) / n_active
    total_pnl += step_pnl
```

3. **Document the decision**: Clearly state in comments whether the objective is per-asset quality or total portfolio PnL, and ensure all metrics are aligned.
