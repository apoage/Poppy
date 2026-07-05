# Foxglove Data Pipeline Audit Summary

## Critical Bugs (Must Fix Before Training)

### BUG-1.1: Zero columns in historical data
**Severity**: CRITICAL
**Location**: `mcts_trader/data/history.py:574-616`

The `get_window()` method returns all 64 asset columns without detecting or handling zero-valued columns. When sampling from early time periods (pre-2020), many assets didn't exist yet, so entire columns remain zero. The forward-fill logic only fills gaps AFTER the first valid data point.

**Impact**: The model learns to predict HOLD on zero-valued assets, which artificially inflates training agreement. This explains the observed 100% → 65% train/validation drop.

**Fix Required**: Add zero-column detection and cyclic duplication in `get_window()` to replace zero columns with duplicated non-zero assets.

### BUG-2.1: PnL normalization mismatch between training and benchmark
**Severity**: HIGH
**Location**: `mcts_trader/jax_agent/path_solver.py:130-131` vs `tools/benchmark_oracle_vs_model.py:403`

Training oracle computes PnL as `sum(abs(returns) * mask) / k_count` (normalized per-asset), but benchmark oracle computes `sum(abs(returns) * mask)` (total portfolio PnL, no normalization).

**Impact**: The model is trained to optimize per-asset performance but evaluated on total portfolio PnL. This creates a fundamental objective mismatch.

**Fix Required**: Align benchmark oracle PnL calculation with training (add `/k_count` normalization) OR remove normalization from training.

## Important Issues (Fix Soon)

### BUG-1.2: Dead code tile logic
**Severity**: HIGH
**Location**: `train_easy.py:1233-1237`

The cyclic tile "safety net" will never execute because `get_window()` always returns exactly `n_assets` columns. This code should be removed after implementing the proper fix in `get_window()`.

### BUG-1.3: No asset availability mask
**Severity**: HIGH
**Location**: `mcts_trader/data/history.py`

HistoricalMarket does not track or expose which assets were available at each timestamp. This information would be useful for downstream processing and debugging.

**Fix Recommended**: Add `self.asset_availability` tracking during RAM preloading.

### BUG-2.2: Model PnL also unnormalized
**Severity**: MEDIUM
**Location**: `tools/benchmark_oracle_vs_model.py:468`

Model PnL calculation (`sum(position * returns)`) has no normalization, creating a third distinct PnL metric that differs from both training and benchmark oracle.

**Fix Recommended**: Consider normalizing model PnL by number of active positions for consistency.

## Minor Issues (Nice to Have)

1. **Rejection rate tracking**: Add metrics to track actual rejection rate during training (expected ~1%).

2. **AR blend verification logging**: Add explicit logging to confirm AR episodes are using historical prices.

3. **Explicit shape validation**: Add assertion in `get_window()` to verify output shape matches expectations.

4. **Force source flag for testing**: Add ability to force specific data source (historical/Bella/synthetic) for debugging.

## Verified Correct

1. **AR blend with historical data**: AR rollout correctly receives the same historical price windows used for oracle computation.

2. **Benchmark period exclusion**: JSON parsing, UTC timezone handling, overlap detection formula, and rejection sampling are all correct.

3. **Data source selection chain**: Historical data correctly takes priority when set. No mixing of data sources in same batch.

4. **Price window shape and content**: Window size (489 steps), shape `(T, 64)`, forward-fill logic, and raw USD price units are all correct.

## Summary Table

| Part | Status | Critical Bugs | Important Issues | Verified Correct |
|------|--------|---------------|------------------|-----------------|
| Part 1: Zero-Asset Handling | FAIL | 1 | 2 | - |
| Part 2: Oracle Math Consistency | FAIL | 1 | 1 | - |
| Part 3: AR Blend with Historical | PASS | 0 | 0 | 3 items |
| Part 4: Benchmark Period Exclusion | PASS | 0 | 0 | All items |
| Part 5: Data Source Selection | PASS | 0 | 0 | All items |
| Part 6: Price Window Validation | PASS | 0 | 0 | All items |

## Recommended Fix Priority

1. **HIGH PRIORITY**: Fix BUG-1.1 (zero columns) - this directly explains the train/validation regression
2. **HIGH PRIORITY**: Fix BUG-2.1 (PnL normalization mismatch) - ensures training/benchmark consistency
3. **MEDIUM PRIORITY**: Remove dead code (BUG-1.2) after implementing BUG-1.1 fix
4. **LOW PRIORITY**: Add asset availability tracking (BUG-1.3) and model PnL normalization (BUG-2.2)
