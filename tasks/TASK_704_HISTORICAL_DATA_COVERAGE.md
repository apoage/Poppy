# TASK_704: Historical Data Coverage — Sequential Scan + Asset Rotation

**Status**: DONE
**Priority**: 2
**Depends on**: TASK_698 (Foxglove Historical Training)
**Blocks**: Training quality on historical data

---

## Problem

Two coverage gaps in historical training:

### 1. Window Sampling: Only 18% of timestamps seen per period

`get_window_random()` samples uniformly at random. With 258K timestamps and ~51K windows sampled per merge cycle, expected coverage is ~18%. Most price data is never seen before the period rotates.

**Fix**: Sequential scan with stride. Walk through the data systematically, guaranteeing full coverage before wrapping. Add small jitter on subsequent passes to avoid exact repetition.

### 2. Asset Rotation: Only one 64-asset subset per period

`HistoricalScheduleManager` ignores `training_runs` in `training_schedule.json` — it reads `available_assets` and creates a single padded-to-64 run per period. But **the schedule already has multiple `training_runs` per period** (e.g. 16 runs with different 64-asset subsets for a 347-asset period). This is the same pattern Bella's `RainbowTableLoader` (`diff_teacher/data_loader.py:438`) uses successfully.

**Fix**: Flatten `training_runs` into individual runs instead of creating one run from `available_assets`.

---

## Reference: Bella's RainbowTableLoader

`diff_teacher/data_loader.py:438` — solves both coverage problems:
- Pre-computed `training_runs` per period with different asset subsets (30-50% overlap)
- Cumulative window index for weighted sampling across all runs
- Period cache with LRU (5 periods in RAM)
- Full OHLCV `(n_timestamps, n_assets, 5)`

The `training_schedule.json` structure:
```json
{
  "periods": [{
    "start": 1633062660,
    "end": 1648616280,
    "available_assets": [...347 assets...],
    "training_runs": [
      {"assets": [...64 assets...], "run_idx": 0},
      {"assets": [...64 assets...], "run_idx": 1},
      ...16 runs total
    ]
  }]
}
```

---

## Implementation Plan

### Phase A: Sequential Window Scanning

**File**: `mcts_trader/data/history.py`

Replace random sampling in `get_window_random()` with sequential scan:
- Stride = `length // 2` (50% overlap between consecutive windows)
- Track `_scan_cursor` position, advance by stride each call
- On wrap (cursor > max_start): reset to 0, increment `_scan_pass`
- On subsequent passes: add small random jitter (±stride/4) to avoid exact repetition
- Zero-copy numpy view preserved

Coverage improvement: 18% → ~100% per period visit.

### Phase B: Flatten training_runs in HistoricalScheduleManager

**File**: `tools/orchestration/e-dojo_v1.py`

Current code (broken — lines 288-304):
```python
# Takes available_assets, pads to 64, creates ONE run per period
assets = period.get('available_assets', ...)
padded = list(assets)
while len(padded) < self.N_ASSETS: ...
self.runs.append({'assets': padded, ...})
```

Fix: Use `training_runs` when available:
```python
training_runs = period.get('training_runs', [])
if training_runs:
    for run in training_runs:
        run_assets = run['assets'][:self.N_ASSETS]
        # Pad if needed (shouldn't be, runs are pre-padded to 64)
        ...
        self.runs.append({'assets': run_assets, ...})
else:
    # Fallback: old behavior for periods without training_runs
    assets = period.get('available_assets', ...)
    ...
```

This gives 16 runs per period instead of 1. The cursor advances through all runs, seeing different asset subsets each time.

### Phase C: Coverage Logging

**File**: `mcts_trader/data/history.py`

Log coverage stats when `_scan_cursor` wraps:
```
[HISTORICAL] Pass 1 complete: 258409 positions scanned, stride=244, 1059 windows
```

---

## Coverage Math

Current (random sampling + one asset subset):
- 258K timestamps, 51K windows sampled → 18% coverage
- 347 assets, 64 used → 18% asset coverage
- Combined: ~3.3% of total data per period

After fix (sequential scan + training_runs flattened):
- Sequential scan: ~100% timestamp coverage per run
- 16 runs per period: 16 × 64 assets = full coverage with 30-50% overlap
- Total runs: 18 periods × ~10 runs/period ≈ 180 runs before cursor wraps

---

## Validation

- [ ] `get_window_random()` sequential scan covers all positions before wrapping
- [ ] Coverage log printed when pass completes
- [ ] `HistoricalScheduleManager` creates multiple runs per period from `training_runs`
- [ ] No regression in training speed (sequential scan is still zero-copy view)
- [ ] Agreement doesn't drop (sequential shouldn't hurt, may help)

---

## Estimated Scope

| Component | Effort | Risk |
|-----------|--------|------|
| Sequential scan in `get_window_random()` | Small | Low |
| Flatten `training_runs` in schedule manager | Small | Low (data already exists) |
| Coverage logging | Tiny | None |

**Total**: ~30 min
