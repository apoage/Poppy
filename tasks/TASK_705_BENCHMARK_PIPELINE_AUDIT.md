# TASK_705: Benchmark Pipeline Audit — Asset Mismatch, PnL Scale, Data Loading

**Status**: DONE
**Priority**: 1 (benchmark is useless until fixed — can't trust any eval results)
**Depends on**: TASK_703 (OHLCV pipeline — done)
**Blocks**: All training decisions that rely on benchmark metrics

---

## Problem

Benchmark results are meaningless. Evidence from 2026-03-10 run:

```
bull:       Bal.Acc=38.6%  OraclePnL=+6961%  ModelPnL=-3460%  RandomPnL=-4233%
bear:       Bal.Acc=38.7%  OraclePnL=+11502% ModelPnL=-5639%  RandomPnL=-4302%
boring_week: Bal.Acc=40.3%  OraclePnL=+6727%  ModelPnL=-2594%  RandomPnL=-4227%
```

Oracle PnL of +12000% in a 7-day period is physically impossible. Model PnL worse than random. Three root causes found:

---

## Bug 1: Asset Mismatch — Benchmark Evaluates on Wrong Assets

### Current behavior

`_get_schedule_assets()` (benchmark_easy.py:128-157) collects the **union** of `available_assets` from all training periods within ±30 days of the benchmark window. For the bull period (2024-02-24 to 2024-03-02), this returns **509 assets**.

All 509 are loaded into `HistoricalMarket`. Then `load_benchmark_prices()` truncates to the first 64 (line 204: `ohlcv[:, :n_assets, :]`). Which 64? Whatever order `HistoricalMarket._load_specific_assets()` returns them — likely by `asset_id` or insertion order. **Not a curated subset.**

### Training behavior

Orchestrator uses `training_schedule.json` which has pre-computed `training_runs` — each is a curated 64-asset subset with 30-50% overlap between runs. The model only ever sees these specific subsets during training.

### Impact

Model is evaluated on 64 assets it may never have trained on. The asset composition determines correlation structure, volatility profile, and which patterns the encoder captures. Evaluating on random assets is like testing a model trained on English with a Chinese test set.

### Fix

`benchmark_periods.json` should specify explicit 64-asset lists per period. Or: load the closest `training_runs` entry's assets for each benchmark period and use those exact assets.

**Option A** (preferred): Add `benchmark_assets: [64 symbols]` to each period in `benchmark_periods.json`. Pre-compute once using the same asset selection logic as `training_schedule.json`.

**Option B**: In `_get_schedule_assets()`, instead of collecting the union, find the single closest training_run and return its exact 64 assets.

**Files**: `tools/benchmark_easy.py` (lines 128-157), `config/benchmark_periods.json`

---

## Bug 2: PnL Not Normalized — Numbers Are 64× Inflated

### Current behavior

All three PnL simulations sum returns across all 64 assets without normalization:

```python
# simulate_model_pnl_detailed, line 304:
step_pnl += np.sum(position * returns[t])  # Raw sum across 64 assets

# simulate_oracle_pnl (benchmark_oracle_vs_model.py, line 402):
step_pnl = np.sum(np.abs(returns[t]) * mask)  # Raw sum across selected assets
```

With 64 assets × 10,000 timesteps of log returns, the raw cumulative sum inflates to thousands of percent.

### Expected behavior

PnL should represent the return of an **equal-weight portfolio**. Each asset gets 1/k weight (where k = number of active positions). Or at minimum, divide by n_assets to get per-asset average.

### The oracle PnL is also wrong

`simulate_oracle_pnl` uses `np.abs(returns[t])` — it takes the absolute value of returns, meaning it captures profit from both up AND down moves. This is correct for the oracle (it knows which direction to bet), but the magnitude is inflated because it sums across all selected assets.

For a 7-day period with BTC +23%: oracle selecting 4 assets per step × 10K steps × ~0.1% per-step return = ~4000% cumulative. Dividing by 64 assets gives ~62% — much more realistic.

### Fix

Normalize all PnL by dividing by `n_assets` (or by number of active positions per step for more accuracy):

```python
# Option A: Simple — divide final result
total_pnl /= n_assets

# Option B: Per-step equal-weight (more correct)
n_active = max(np.sum(np.abs(position) > 0.5), 1)
step_pnl = np.sum(position * returns[t]) / n_active
```

**Files**: `tools/benchmark_easy.py` (lines 263-308, 355-377), oracle PnL in benchmark_oracle_vs_model.py or wherever `simulate_oracle_pnl` lives

---

## Bug 3: Data Loading — 509 Assets Loaded, 445 Discarded

### Current behavior

Benchmark loads ALL assets from adjacent training periods (509), creates a full `HistoricalMarket` with RAM cache for all of them, forward-fills 4.4M price gaps, then discards 88% of them.

```
INFO: Loaded 579,941 candles into RAM (97.2 MB) in 1.3s
INFO: Forward-filling price gaps...
INFO: Filled 4,463,702 price gaps via forward-fill
WARNING: 4300 zero/negative prices, filling gaps...
```

### Training behavior

Training loads exactly 64 specific assets into `HistoricalMarket` via `--assets-file`. No wasted loading. No millions of forward-fills.

### Fix

Pass exactly 64 assets to `HistoricalMarket` in `load_benchmark_prices()`. This is a direct consequence of fixing Bug 1 — once we have explicit benchmark asset lists, we pass those directly.

**Impact**: ~10× faster benchmark loading, no wasted RAM, no misleading "4300 zero prices" warnings.

**Files**: `tools/benchmark_easy.py` (lines 160-230)

---

## Bug 4 (Potential): Oracle vs Training Oracle Alignment

### Concern

The benchmark oracle uses `greedy_lookahead_path` with the same parameters as training (k=4, lookahead=20). But verify:

1. Is `transaction_cost` the same in benchmark and training config?
2. Is `min_alpha_threshold` the same?
3. Does the benchmark oracle use the SAME function (from `path_solver.py`) or a different implementation?
4. Does multipass oracle vs single-pass affect comparison?

### Check

Compare `run_period_eval` oracle call (benchmark_easy.py:464-471) with `_generate_batch_episodes` oracle call (train_easy.py:871-916). Both should use identical parameters.

**Files**: `tools/benchmark_easy.py` (lines 464-471), `train_easy.py` (lines 871-916)

---

## Implementation Plan

### Phase A: Fix Asset Selection (Bug 1 + Bug 3)

1. Pre-compute 64-asset benchmark lists for each of the 6 periods
2. Add `benchmark_assets` field to `benchmark_periods.json`
3. Update `load_benchmark_prices()` to use these directly (no more `_get_schedule_assets` union)
4. Pass exact 64 assets as `specific_symbols` to `HistoricalMarket`

### Phase B: Fix PnL Normalization (Bug 2)

1. Decide normalization strategy: per-asset average or equal-weight active positions
2. Update `simulate_model_pnl_detailed()` — divide by n_assets
3. Update `simulate_oracle_pnl()` — divide by n_assets (or by k for oracle)
4. Update `evaluate_random_policy()` / `_simulate_pnl_vectorized()` — same normalization
5. Update alpha computation if needed
6. Verify Sharpe ratio computation uses normalized step returns

### Phase C: Oracle Parameter Alignment (Bug 4)

1. Audit oracle parameters match between benchmark and training
2. Fix any discrepancies

### Phase D: Validation

1. Re-run benchmark after fixes
2. Oracle PnL should be in realistic range (single-digit % for 7-day periods)
3. Model PnL and random PnL should be comparable in magnitude
4. Asset loading should show exactly 64 assets, minimal forward-fill gaps

---

## Acceptance Criteria

- [ ] Benchmark loads exactly 64 pre-specified assets per period (not 509)
- [ ] Forward-fill gaps < 100K (not 4.4M)
- [ ] Zero "zero/negative prices" warnings (assets are known-good)
- [ ] Oracle PnL in realistic range: single-digit % for 7-day periods
- [ ] Model PnL, random PnL, and oracle PnL comparable in magnitude
- [ ] PnL normalized per-asset (divide by n_assets or active positions)
- [ ] Oracle parameters identical to training config
- [ ] Benchmark runs in <60s (not 595s)
- [ ] `benchmark_periods.json` has explicit `benchmark_assets` per period

---

## Estimated Scope

| Component | Effort | Risk |
|-----------|--------|------|
| Asset list pre-computation | Small | Low |
| benchmark_periods.json update | Small | Low |
| load_benchmark_prices fix | Small | Low |
| PnL normalization (3 functions) | Small | Low (math change) |
| Oracle param audit | Tiny | Low |
| Validation run | Small | Low |

**Total**: ~1 session
