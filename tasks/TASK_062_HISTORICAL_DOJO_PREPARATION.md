# TASK_062: Historical Dojo Preparation

## Status: IN_PROGRESS

## Objective

Prepare training infrastructure for historical market data after synthetic curriculum completion.

## Prerequisites

- [ ] Poppy v2 synthetic curriculum complete (L0-L14) - currently running L14
- [ ] Database validation passed
- [ ] Benchmark periods excluded
- [ ] Data integrity verified

## Training Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Parallel Envs | 15 | Optimal speed for historical data |
| Days per Epoch | 15 | 15 days × 1440 min = 21,600 steps |
| Epochs per Run | 60 | 60 × 15 days = 900 days = ~2.5 years |
| Benchmark Frequency | Every 60 epochs | After each ~2.5 year block |

## Rainbow Table Approach

Uses `tools/map_training_data.py` to generate training schedule:
- Scans database for periods with ≥64 concurrent assets
- Generates multiple 64-asset combinations per period (30-50% overlap)
- Each combination = separate training run
- **Total: ~80 years of trainable data** through asset rotation

## Phase 1: Database Validation

### 1.1 Run Validation Tools

```bash
python tools/validate_database.py
python tools/validate_fx_rates.py
```

### 1.2 Check Coverage

- [ ] Date range: earliest to latest data
- [ ] Asset coverage: active + delisted pairs
- [ ] FX rate completeness
- [ ] No zero prices
- [ ] No fixed/stuck prices

### 1.3 Identify Valid Segments

For delisted pairs, identify valid trading segments:
```sql
SELECT asset_id, MIN(timestamp) as start, MAX(timestamp) as end,
       COUNT(*) as candles
FROM ohlcv
GROUP BY asset_id
```

## Phase 2: Data Integrity Checks

### 2.1 Zero Price Detection

```python
# Find any zero or near-zero prices
SELECT asset_id, timestamp, close
FROM ohlcv
WHERE close < 0.0001 OR close IS NULL
```

### 2.2 Stuck Price Detection

```python
# Find periods where price doesn't change (stuck feed)
# Flag if same price for > 60 consecutive minutes
```

### 2.3 FX Rate Validation

```python
# Ensure FX rates exist for all timestamps
# No gaps > 1 hour
# No extreme jumps (> 5% in 1 minute)
```

## Phase 3: Benchmark Period Exclusion

### Current Benchmark Periods (from config/benchmark_periods.json)

| Period | Start | End | Type |
|--------|-------|-----|------|
| Bull | 2024-02-24 | 2024-03-02 | ETF Rally |
| Bear | 2022-06-10 | 2022-06-17 | Luna Crash |
| Sideways | 2023-07-01 | 2023-07-08 | Consolidation |

### Additional Benchmark Periods (TASK_061)

| Period | Start | End | Type |
|--------|-------|-----|------|
| FTX Crash | 2022-11-06 | 2022-11-13 | Panic |
| ETF Approval | 2024-01-08 | 2024-01-15 | Rally |
| Boring Week | 2023-08-14 | 2023-08-21 | Low volatility (summer doldrums) |

**Boring Week**: Mid-August 2023 was exceptionally quiet - BTC traded in ~2% range
for entire week. Tests model behavior during low-opportunity periods.

### Exclusion Implementation

Modify data loader to skip benchmark date ranges during training.

## Phase 4: Training Loop Validation

### 4.1 Data Serving Checks

- [ ] Prices correctly normalized
- [ ] No NaN/Inf values reaching model
- [ ] FX conversion applied correctly
- [ ] Multi-asset alignment (all assets have data for each timestamp)

### 4.2 Edge Cases

**Delisted Pairs**: No continuity needed between epochs. Each epoch picks valid assets
for that date range. No runtime masking required - rainbow table pre-filters.

**Data Gaps**: Handled by `map_training_data.py` which only returns periods with
continuous data. Gaps between epochs are fine (no state carried over).

### 4.3 Smoke Test

Run 5 epochs with historical data before full training:
```bash
python train_jax_fast.py \
    --db-path trading_data.db \
    --start-date 2023-01-01 \
    --end-date 2023-01-15 \
    --epochs 5 \
    --parallel-envs 15 \
    --steps 1440 \
    --name historical_smoke_test
```

Verify:
- No crashes
- Reasonable rewards (not all zeros)
- Actions distributed (not all HOLD)
- No NaN losses

## Phase 5: Historical Dojo Config

### 5.1 Create Dojo v3 (Historical)

```python
# tools/orchestration/dojo_v3_historical.py

HISTORICAL_CONFIG = {
    'parallel_envs': 15,
    'steps': 1440,  # 24 hours of 1-min candles
    'epochs_per_segment': 60,
    'benchmark_every': 60,

    # Date segments (oldest to newest, excluding benchmarks)
    'segments': [
        {'start': '2021-01-01', 'end': '2021-06-30'},
        {'start': '2021-07-01', 'end': '2021-12-31'},
        # ... more segments
    ],

    # Exclude benchmark periods
    'exclude_periods': [
        # From config/benchmark_periods.json
    ],
}
```

### 5.2 Delisted Pair Handling

Option A: Include delisted pairs only in their valid date range
Option B: Mask delisted pairs entirely (simpler but less data)

**Decision**: Option A - more data is better for generalization

## Phase 6: Checkpoint Strategy

### 6.1 Save Points (Level-based naming)

| Checkpoint | After | Description |
|------------|-------|-------------|
| `poppy_v2_L14_complete` | L14 graduation | Synthetic curriculum complete |
| `poppy_v2_H01` | 60 epochs | First historical block (~2.5 years) |
| `poppy_v2_H02` | 120 epochs | Second historical block |
| `poppy_v2_H03` | 180 epochs | Third historical block |
| ... | ... | Continue as needed |
| `poppy_v2_H_complete` | Final | All historical data trained |

### 6.2 Benchmark After Each Block

Run full benchmark suite (Bull/Bear/Sideways + new periods) after each 60-epoch block.

## Validation Checklist

- [ ] `validate_database.py` passes
- [ ] `validate_fx_rates.py` passes
- [ ] All benchmark periods identified and excluded
- [ ] Delisted pairs mapped to valid segments
- [ ] Smoke test completes without errors
- [ ] Historical dojo config created
- [ ] Checkpoint strategy defined

## Files to Create/Modify

| File | Action |
|------|--------|
| `tools/orchestration/dojo_v3_historical.py` | Create |
| `config/historical_training.json` | Create |
| `config/benchmark_periods.json` | Update with more periods |
| `mcts_trader/data/history.py` | Modify to exclude benchmarks |

## Acceptance Criteria

- [ ] Database validated, no critical issues
- [ ] Benchmark exclusion implemented
- [ ] Smoke test passes
- [ ] Historical dojo ready to launch
- [ ] ~2.5+ years of training data available
