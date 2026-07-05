# TASK_105: Dojo V3 Period Wraparound & Defensive Loading

## Problem
Dojo v3 historical training hangs when it reaches periods with dates beyond available database data.

**Observed behavior** (2026-01-19):
```
2026-01-19 13:54:04,067 - [DOJO-V3] - BATCH 5: Periods 17-17
2026-01-19 13:54:04,068 - [DOJO-V3] -   P0: 2025-02-25 → 2025-11-30 (278 days)
<hangs forever>
```

**Root cause**:
- `training_schedule.json` contains speculative future periods (e.g., late 2025).
- Database only has data up to ~early 2025.
- The data loader waits indefinitely for non-existent data instead of failing fast.
- The current training loop is linear and terminates (or hangs) at the end of the schedule, rather than wrapping around.

## Solution Plan

### 1. Defensive Data Loading (Fail Fast)
The data loader must not hang. If a requested date range is outside the database bounds, it should raise an immediate error.
- **Action**: Update `mcts_trader/data/history.py` (specifically `_preload_to_ram`) to check `start_date > db_max_date` and raise `ValueError`.

### 2. Startup Schedule Validation
Dojo should filter the schedule before training starts.
- **Action**: In `DojoV3Historical.__init__`, query `SELECT MAX(timestamp) FROM ohlcv`.
- **Action**: Create `self.valid_periods` by filtering the raw schedule: keep only periods where `start_date < db_max_date`.

### 3. Two Training Modes

**Mode A: One-shot** (`--stratified`, default)
- Visit all valid periods exactly once in schedule order
- Preserves prebaked distribution from `training_schedule.json`
- Stops when all periods exhausted
- Use case: Initial 80-year training pass

**Mode B: Infinite loop** (`--stratified --loop`)
- Cycle through periods forever until Ctrl+C
- Wraps around to period 0 after visiting all periods
- Use case: Future continuous training

**Logic**:
```python
valid_periods = [p for p in all_periods if _has_data(p)]
cursor = 0
loop_mode = args.loop  # --loop flag

while True:
    batch_periods = []

    for _ in range(periods_per_batch):
        if cursor >= len(valid_periods):
            if loop_mode:
                cursor = 0  # Wraparound
            else:
                break  # One-shot: stop
        batch_periods.append(valid_periods[cursor])
        cursor += 1

    if not batch_periods:
        break  # All periods exhausted (one-shot mode)

    # Generate periods.json and train...
```

**Key**: No randomization in either mode - schedule order preserved to avoid overfitting specific periods.

## Implementation Checklist

- [x] **Data Loader**: Add bounds check to `HistoricalMarket` in `mcts_trader/data/history.py`.
- [x] **Dojo Class**: Add `_get_db_max_date()` and `_filter_valid_periods()` to `tools/orchestration/dojo_v3_historical.py`.
- [x] **CLI Flag**: Add `--loop` argument for infinite mode (default: one-shot).
- [x] **Training Loop**: Refactor `run_training_stratified` to support both modes.
- [ ] **Testing**:
  - One-shot: Verify training stops after all valid periods visited
  - Loop: Verify Batch N+1 wraps to Period 0 after exhausting schedule

## Files to Modify
- `tools/orchestration/dojo_v3_historical.py`
- `mcts_trader/data/history.py`