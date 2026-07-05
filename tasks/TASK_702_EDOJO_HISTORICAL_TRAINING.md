# TASK_702: e-dojo Historical Data Training (Proper Implementation)

**Status**: DONE
**Priority**: 1 (next major milestone)
**Depends on**: TASK_698 (historical infra), TASK_701 (data pipeline patches)

---

## Motivation

TASK_698 added basic historical data support to e-dojo (`--branch-a-historical`) but it's a naive
first cut: Branch A trains on a single period, the cursor advances once per merge, and there's no
stratified cycling, no schedule-aware progression, no benchmarking between blocks. Meanwhile dojo_v3
has a mature historical training system (stratified mode, period batching, block checkpoints, benchmarks)
but it targets `train_jax_fast.py` (Poppy), not `train_easy.py` (Foxglove).

**Goal**: Give e-dojo a proper historical training mode equivalent to what dojo_v3 provides for Poppy,
adapted to Foxglove's dual-branch merge architecture.

---

## Current State

### What dojo_v3 does well (features to port)

1. **Stratified cycling**: Batches of 4 periods, 15 epochs each → prevents catastrophic forgetting
2. **Rainbow table consumption**: Reads `config/training_schedule.json` with 14 periods, 171 runs
3. **Period filtering**: Skips periods beyond DB max date (`_filter_valid_periods`)
4. **Run flattening**: Each period has multiple training_runs (64-asset combos) → iterates all
5. **Block checkpoints**: Named H01, H02, H03... with benchmarks between blocks
6. **Benchmark integration**: Runs eval on bull/bear/sideways/ftx_crash/etf_approval/boring_week
7. **Hot config overrides**: `config/dojo_override.json` for tuning mid-training
8. **Loop mode**: `--loop` for infinite cycling through all periods

### What e-dojo already has (TASK_698)

1. `--branch-a-historical` flag → Branch A uses historical data
2. `--db-path`, `--historical-schedule` CLI flags
3. `period_cursor` in DojoState (crash recovery)
4. Period cursor advances by 1 on each merge restart
5. Branch B runs Bella as stabilizer (prevents forgetting)

### What's missing

1. **Only advances 1 period per merge** — slow progression through 14 periods × N runs
2. **No run-level iteration** — uses period-level cursor only, ignores training_runs
3. **No block checkpoints** — no named snapshots (H01, H02, ...)
4. **No benchmarking** — no eval between blocks
5. **No stratified batching** — trains on 1 period at a time (not 4)
6. **No epoch-per-period control** — no knob for how long to train each period
7. **No loop/one-shot mode** — no explicit control over cycling
8. **Branch B doesn't adapt** — always Bella, never cycles periods itself
9. **No schedule validation** — no DB max date filtering

---

## Design

### Architecture: Historical Epoch Manager

New class `HistoricalScheduleManager` inside `e-dojo_v1.py` that manages period progression.

```
HistoricalScheduleManager
├── schedule: training_schedule.json (loaded)
├── valid_runs: flattened, DB-date-filtered runs
├── cursor: current position in valid_runs
├── block_count: H01, H02, ...
├── epochs_in_current_block: counter
│
├── get_next_period() → (start_ts, end_ts, assets)
├── advance_cursor()
├── should_checkpoint_block() → bool
└── get_block_name() → "H01"
```

### Period Progression Strategy

The dual-branch architecture changes how we consume the schedule vs dojo_v3:

- **dojo_v3**: Single process, trains epochs_per_period on each period sequentially
- **e-dojo**: Two branches running in parallel, merge every ~20-50 epochs

Proposed approach:

```
Branch A: Historical (cycles through periods)
Branch B: Bella (stabilizer — same as current)

Period cycling:
- Each merge restart → advance to next run in valid_runs
- Every 4 runs → that's one "block" (H01, H02, ...)
- After each block → run benchmark
- After all runs → loop back to run 0 (if --loop)
```

The merge interval (~20-50 epochs) already provides a natural switching point. Each merge:
1. Branches merge → merged_latest.pkl
2. Advance period cursor to next run
3. Restart Branch A with new period bounds
4. If block completed → save named checkpoint + benchmark

### Branch A Period Assignment

On each restart, Branch A gets:
```python
run = valid_runs[cursor % len(valid_runs)]
period_start_ts = run['start_ts']
period_end_ts = run['end_ts']
# Assets handled by HistoricalMarket (full DB loaded, period bounds restrict sampling window)
```

### Block Checkpoint Strategy

```
runs_per_block = 4  (configurable, matches dojo_v3 periods_per_batch)

After runs_per_block merges with period advancement:
  1. Copy merged_latest.pkl → jax_ckpt_easy_merged/foxglove_H01.pkl
  2. Run benchmark suite (benchmark_easy.py or benchmark_oracle_vs_model.py)
  3. Log results to logs/benchmarks/foxglove_H{N}_benchmark.csv
  4. Increment block counter
```

### Benchmark Integration

After each block, run benchmarks on all `config/benchmark_periods.json` periods:
```bash
python tools/benchmark_easy.py \
    --checkpoint jax_ckpt_easy_merged/foxglove_H01.pkl \
    --preset large \
    --periods config/benchmark_periods.json
```

Parse output → extract balanced accuracy, PnL, alpha → save CSV.

### State Persistence (crash recovery)

Extend `DojoState` with:
```python
# Existing (TASK_698):
period_cursor: int = 0

# New:
historical_block_count: int = 0
runs_since_last_block: int = 0
historical_mode: bool = False  # Flag for state restore
```

On restart, `HistoricalScheduleManager` resumes from `period_cursor` — same as current but now
with run-level granularity.

---

## Implementation Plan

### Phase A: HistoricalScheduleManager class

File: `tools/orchestration/e-dojo_v1.py`

1. Add `HistoricalScheduleManager` class:
   - Constructor: loads schedule JSON, filters periods by DB max date (port from dojo_v3)
   - `_flatten_runs()`: port from dojo_v3 — flattens periods into individual runs with assets
   - `get_current_run()`: returns current run dict (start_ts, end_ts, assets)
   - `advance()`: increments cursor, handles wraparound
   - `block_completed()`: returns True every `runs_per_block` advances
   - `get_block_name()`: returns "H{N:02d}"

### Phase B: Wire into EasyDojo merge cycle

File: `tools/orchestration/e-dojo_v1.py`

1. EasyDojo constructor: if historical mode, create `HistoricalScheduleManager`
2. After merge in `check_and_merge()`:
   - Call `schedule_manager.advance()`
   - If `schedule_manager.block_completed()` → save block checkpoint + run benchmark
3. Update `_save_state()`/`_load_state()` with new state fields

### Phase C: Update TrainingLauncher restart logic

File: `tools/orchestration/e-dojo_v1.py`

1. `restart_both_with_checkpoint()`: use `schedule_manager.get_current_run()` for period bounds
   (replaces current manual cursor advance)
2. Clean up TASK_698 inline schedule reading (now handled by HistoricalScheduleManager)

### Phase D: Block checkpoint + benchmark

File: `tools/orchestration/e-dojo_v1.py`

1. Add `_save_block_checkpoint(block_name)`: copy merged_latest.pkl → named block
2. Add `_run_block_benchmark(checkpoint_path, block_name)`:
   - Shell out to `benchmark_easy.py` (or `benchmark_oracle_vs_model.py`)
   - Parse results → save CSV to `logs/benchmarks/`
3. Integrate into merge cycle (after block detection)

### Phase E: CLI flags

File: `tools/orchestration/e-dojo_v1.py`

New/updated flags:
```
--runs-per-block INT    # Runs per block before benchmark (default: 4)
--loop                  # Infinite cycling through all runs
--max-blocks INT        # Stop after N blocks
--start-block INT       # Resume from block N (default: 1)
```

### Phase F: Validation

1. Syntax check: `python -m py_compile tools/orchestration/e-dojo_v1.py`
2. Dry run: launch with `--branch-a-historical` and verify:
   - Schedule loads, periods filter correctly
   - Period cursor advances per merge
   - Block checkpoint created every N merges
   - Benchmark runs after block
   - State persists across restart
3. Monitor first 2 blocks (~8 merges) for correctness

---

## Configuration Defaults

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `runs_per_block` | 4 | Match dojo_v3 `periods_per_batch` |
| merge interval | 20 epochs (~8000 steps) | Current e-dojo default |
| Epochs per period | ~20 (determined by merge interval) | Natural from merge cycle |
| Block = | 4 runs × ~20 epochs = ~80 epochs | Reasonable benchmark frequency |
| Benchmark periods | bull, bear, sideways, ftx_crash, etf_approval, boring_week | Standard suite |

---

## What NOT to change

- Branch B stays as Bella stabilizer (current design is correct)
- Merge logic stays unchanged (Fisher/FedAvg/TIES selection)
- AR blend advancement stays unchanged
- Level advancement stays unchanged (only relevant for synthetic)
- train_easy.py stays unchanged (TASK_698 already wired historical support)
- HistoricalMarket stays unchanged (data layer is solid after TASK_701)

---

## Usage

```bash
# Historical training with benchmarks every 4 runs
python tools/orchestration/e-dojo_v1.py --run --preset large --epochs 5000 \
    --branch-a-historical --db-path trading_data.db \
    --historical-schedule config/training_schedule.json \
    --runs-per-block 4

# Same but infinite loop
python tools/orchestration/e-dojo_v1.py --run --preset large --epochs 5000 \
    --branch-a-historical --db-path trading_data.db \
    --loop --max-blocks 50

# Resume from block 5
python tools/orchestration/e-dojo_v1.py --run --preset large --epochs 5000 \
    --branch-a-historical --db-path trading_data.db \
    --start-block 5 --loop
```
