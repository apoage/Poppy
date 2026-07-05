# TASK_698: Foxglove Historical Data Training

**Status**: DONE
**Priority**: 1 (next major milestone after Phase 1 AR convergence)
**Depends on**: Phase 1 training completion (AR ~100%, balanced agreement >= 75%)
**Blocks**: Real trading readiness

---

## Motivation

Foxglove currently trains exclusively on synthetic data (Bella + SyntheticMarket L0-L14). The model achieves 74%+ balanced agreement on synthetic data but **0% trades on real market data** — it has directional signal but never commits on real distributions.

**Goal**: Train on historical market data using the proven dojo_v3/v4 orchestrator pattern, with proper benchmark evaluation after each merge.

---

## Architecture

### Training Layout

```
e-dojo (orchestrator)
├── Branch A (RTX 3090): Historical — train_easy.py --db-path ... --start-date ... --end-date ...
├── Branch B (CMP 90HX): Bella — train_easy.py --bella ...
├── Merge: merge branch checkpoints
├── Benchmark: benchmark_easy.py --checkpoint merged.pkl --json results.json
├── Advance: orchestrator picks next period from schedule
└── Repeat: reload training → train → merge → benchmark → advance → ...
```

- **Branch A** (RTX 3090): Historical market data from `trading_data.db`, one period per training block
- **Branch B** (CMP 90HX): Bella synthetic to prevent catastrophic forgetting
- **After each merge**: Full benchmark via `benchmark_easy.py` (TASK_682)
- **Period cycling**: Orchestrator advances to next period after each block, not inside `train_easy.py`

### Design Principles

1. **Subprocess pattern** (dojo_v3/v4): orchestrator manages lifecycle, `train_easy.py` just trains on what it's given
2. **One period per subprocess**: no mid-training period switching — avoids RAM cache reload cost
3. **Benchmark is the decision metric**: `benchmark_easy.py` output determines training quality
4. **No data leakage**: training periods and benchmark periods are strictly separated

---

## Data Leakage Analysis

| Vector | Risk | Status |
|--------|------|--------|
| Benchmark periods in training data | **Critical** | SAFE — `training_schedule.json` and `benchmark_periods.json` are disjoint sets. Orchestrator must verify no overlap at startup. |
| Oracle lookahead | None | **Intentional** — expert iteration requires perfect-hindsight oracle. Not leakage. |
| Forward-fill in HistoricalMarket | None | SAFE — fills gaps with last KNOWN price (backward-looking only). |
| Encoder indicators (RSI, etc.) | None | SAFE — computed on the price window itself, no future data. |
| Window boundary bleed | **Medium** | Must ensure oracle lookahead doesn't extend past `end_date`. Window sampling: `max_start = n_timestamps - total_window_length` where `total_window_length = episode_length + window_size + oracle_lookahead + 5 = 489` timesteps. |
| AR blend model state | None | SAFE — model only sees its own past predictions. |
| Cross-period contamination | **Low** | SAFE — each subprocess loads exactly one period. No state carries between periods except model weights (which is intentional learning). |

**Startup validation** (orchestrator must check):
```python
# Verify no benchmark period overlaps any training period
for bench in benchmark_periods:
    for train in training_periods:
        assert not (bench['start'] < train['end'] and bench['end'] > train['start']), \
            f"DATA LEAK: benchmark {bench['name']} overlaps training period {train['start_date']}"
```

---

## Implementation Plan

### Phase A: CLI Flags in train_easy.py

**Files**: `train_easy.py`

Add flags matching `train_jax_fast.py`:
```
--db-path PATH          # SQLite database path (enables historical mode)
--start-date DATE       # Period start (YYYY-MM-DD)
--end-date DATE         # Period end
--assets-file PATH      # JSON list of asset symbols (pre-padded to 64 by orchestrator)
--no-checkpoint         # Disable checkpoint saving (for eval runs)
```

No `--periods-file` — period cycling is the orchestrator's job, not train_easy.py's. Each subprocess trains on exactly one period. This avoids:
- RAM cache reload cost mid-training
- Complexity of `switch_period()` (which doesn't exist)
- Subtle bugs from changing data distribution mid-epoch

### Phase B: Historical Branch in OnTheFlyGenerator

**Files**: `train_easy.py`, `mcts_trader/data/history.py`

Historical data plugs into `OnTheFlyGenerator._refill_buffer()` as a 4th data source alongside Bella, GPU-synth, and CPU-synth.

**In `__init__`**: Accept `historical_market` parameter:
```python
def __init__(self, ..., historical_market=None):
    self.historical_market = historical_market
```

**In `_refill_buffer()`**: Add historical branch before the existing Bella/synth branches:
```python
if self.historical_market is not None:
    # Sample random windows from RAM cache — close prices only (T, n_assets)
    prices_list = []
    levels = []
    for _ in range(n_to_refill):
        prices = self.historical_market.get_window_random(self._synth_n_steps)
        prices_list.append(prices)  # (total_steps, n_assets) — close prices
        levels.append(-2)  # -2 = historical (vs -1 = Bella)
elif self.bella_generator is not None:
    ...  # existing Bella path
```

The rest of the pipeline is unchanged — `_generate_batch_episodes()` takes `prices_list` as `(T, n_assets)` close prices, runs oracle + encoder, fills buffer. Historical prices flow through exactly the same path as synthetic.

**Shape compatibility**: All existing generators (Bella, GPU-synth, CPU-synth) return `(T, n_assets)` close prices. `_generate_batch_episodes()` pads these into `(N, T, n_assets)`. The encoder fakes OHLCV internally (`volumes_T = jnp.ones_like(prices_T)`, highs=lows=close). Historical data must match this shape — return close prices only from `cache_data[:, :, 3]`.

> **Future work**: True OHLCV integration (feeding real volume/high/low to the encoder) would require refactoring `_generate_batch_episodes()`, `encode_observation_temporal()`, and all generators. This is a separate task — not needed for the distribution gap fix.

**In `HistoricalMarket`**: Add two methods:

```python
def get_window_random(self, length: int) -> np.ndarray:
    """Sample a random contiguous window of close prices from RAM cache.

    Args:
        length: Number of timesteps (must include episode_length + window_size + oracle_lookahead + margin)

    Returns:
        (length, n_assets) close prices as float32 — matches synthetic generator output shape
    """
    max_start = len(self.valid_timestamps) - length
    assert max_start > 0, f"Period too short: {len(self.valid_timestamps)} < {length}"
    start_idx = np.random.randint(0, max_start)
    return self.cache_data[start_idx:start_idx + length, :, 3]  # close prices only, zero-copy view

def get_window(self, start_idx: int, length: int) -> np.ndarray:
    """Get a specific contiguous window of close prices from RAM cache.

    Returns:
        (length, n_assets) close prices as float32
    """
    end_idx = min(start_idx + length, len(self.valid_timestamps))
    return self.cache_data[start_idx:end_idx, :, 3]
```

**Key efficiency points**:
- `cache_data[start:end, :, 3]` is a numpy view — **zero copy**, no allocation
- Returns `(T, n_assets)` close prices — identical shape to Bella/synthetic generators
- RAM cache loaded once at startup (~166MB for 90-day period, 64 assets)
- No DB queries during training
- No disk I/O in the training loop

### Phase C: Asset Padding

**Files**: Orchestrator only (not train_easy.py)

Asset padding to 64 is handled by the orchestrator before launching subprocess, matching dojo_v3:
```python
N_ASSETS = 64
if len(assets) < N_ASSETS:
    while len(assets) < N_ASSETS:
        assets = assets + assets[:N_ASSETS - len(assets)]
    assets = assets[:N_ASSETS]
```

Orchestrator writes the padded list to `--assets-file`. `train_easy.py` and `HistoricalMarket` just use what they're given.

### Phase D: Benchmark After Each Merge

**Files**: `tools/orchestration/e-dojo_v1.py`

After each merge in the e-dojo loop:

1. Merge branch checkpoints (existing logic)
2. Run `benchmark_easy.py` on merged checkpoint:
   ```bash
   python tools/benchmark_easy.py \
     --checkpoint logs/e-dojo/merged_latest.pkl \
     --db trading_data.db \
     --json logs/benchmarks/foxglove_merge_{N}.json
   ```
3. Parse JSON output, append to `logs/benchmarks/foxglove_benchmark.csv`:
   ```
   timestamp, merge_num, period_trained, avg_balanced_acc, avg_alpha, avg_trade_rate, avg_sharpe, positive_alpha_count, beats_random_count
   ```
4. Log summary to stdout for monitoring
5. Resume training from merged checkpoint

`benchmark_easy.py` (TASK_682) provides everything:
- 6 benchmark periods from `config/benchmark_periods.json`
- Per-period: balanced_accuracy, model_pnl, alpha, sharpe, max_drawdown, trade_rate, value_mse
- TASK_666 success criteria evaluation
- Structured JSON with summary

**Benchmark runs on RTX 3090** (Branch A GPU) since both branches are paused during merge. Takes ~2-5 min for all 6 periods.

### Phase E: Extend e-dojo for Historical Mode

**Files**: `tools/orchestration/e-dojo_v1.py`

New flags:
```bash
python tools/orchestration/e-dojo_v1.py --run --preset large \
  --branch-a-historical \
  --db-path trading_data.db \
  --schedule training_schedule.json \
  --dual-gpu \
  --epochs 2500
```

Orchestrator responsibilities:
1. **Startup**: Load `training_schedule.json`, filter valid periods (dojo_v3 `_filter_valid_periods`), flatten runs with asset padding (dojo_v3 `_flatten_runs`)
2. **Overlap check**: Verify no training period overlaps any benchmark period (hard assert)
3. **Period cursor**: Track which run is current. Advance after each training block.
4. **Branch A command**: Build `train_easy.py` command with `--db-path`, `--start-date`, `--end-date`, `--assets-file` for current period
5. **After merge**: Run benchmark, log results, advance period cursor
6. **Loop/wrap**: When all periods exhausted, wrap to start (loop mode) or stop

Branch A subprocess command:
```python
assets_file = write_assets_json(current_run['assets'])  # pre-padded to 64
cmd = [
    'python', 'train_easy.py',
    '--db-path', self.db_path,
    '--start-date', current_run['start_date'],
    '--end-date', current_run['end_date'],
    '--assets-file', assets_file,
    '--epochs', str(epochs_per_block),
    '--load-checkpoint', merged_checkpoint,
    '--preset', 'large',
    '--lr', '5e-5',  # Lower LR for historical fine-tuning
]
```

Branch B stays unchanged — Bella on CMP 90HX.

**Hot config overrides**: Reuse `config/dojo_override.json` (same as dojo_v3).

**Block naming**: F01, F02, ... (Foxglove historical series).

---

## Efficiency Considerations

| Concern | Solution |
|---------|----------|
| RAM cache reload per period | One period per subprocess. Cache loads once at startup (~166MB for 90 days, 64 assets). No reload during training. |
| Window sampling | `cache_data[start:end]` is numpy view — zero copy, zero allocation. |
| Disk I/O in training loop | None. All data in RAM after startup. Matches CLAUDE.md anti-pattern rules. |
| Oracle computation | Same batched oracle as synthetic — `_generate_batch_episodes()` unchanged. |
| Subprocess overhead (JIT warmup) | ~30s per restart. Acceptable given blocks are 50+ epochs (~15-30 min each). |
| Benchmark cost | ~2-5 min per merge. Runs during natural pause (both branches idle). No GPU waste. |
| Period too short for episode | Orchestrator validates: period must have >= 489 timestamps (8.15 hours). Skip shorter periods. |

---

## Reference: Reusable Patterns

| Pattern | Source | Action |
|---------|--------|--------|
| Rainbow table schedule | `training_schedule.json` via `map_training_data.py` | Reuse as-is |
| Period filtering (DB max date) | `dojo_v3._filter_valid_periods()` | Copy |
| Asset cyclic padding to 64 | `dojo_v3._flatten_runs()` | Copy |
| Hot config overrides | `config/dojo_override.json` | Reuse same file |
| Benchmark evaluation | `tools/benchmark_easy.py` (TASK_682) | Call with `--json`, parse output |
| Benchmark periods | `config/benchmark_periods.json` | Used by benchmark_easy.py |

---

## Critical Design Decisions

### 1. One Period Per Subprocess (not periods-file cycling)
`HistoricalMarket` loads data via SQL query + RAM cache build at init. Switching periods means reloading — expensive and complex. Instead, the orchestrator launches a new subprocess per period. This matches dojo_v3's proven pattern and avoids:
- `switch_period()` method that doesn't exist
- RAM cache thrashing
- Subtle state leaks between periods

### 2. Historical as New Branch in _refill_buffer()
OnTheFlyGenerator already has multiple data source branches (Bella, GPU-synth, CPU-synth, secondary-mix). Historical becomes another branch. Same downstream pipeline (oracle, encoder, buffer). No new generator class needed.

### 3. Close Prices Only — (T, n_assets) Shape
All existing generators return `(T, n_assets)` close prices. `_generate_batch_episodes()` expects this shape. The encoder fakes OHLCV internally (`volumes = ones_like(prices)`, highs=lows=close). Historical must match: `cache_data[:, :, 3]` extracts close prices as `(T, n_assets)`. True OHLCV integration (feeding real volume/high/low) is a separate future task requiring encoder + generator refactoring.

### 4. Oracle on Historical Data
`greedy_lookahead_path` works identically — perfect hindsight regardless of source. This is intentional expert iteration, not data leakage.

### 5. Benchmark as Final Arbiter
`benchmark_easy.py` evaluates:
- **Balanced accuracy** — agree_hold + agree_trade
- **Alpha** — excess return over market
- **Trade density** — model actually trades (not all-HOLD)
- **Sharpe ratio** — risk-adjusted returns
- **Max drawdown** — tail risk
- **Value MSE** — value prediction quality
These together determine real-trading readiness.

### 6. Distribution Shift Management
- **Dual-branch e-dojo**: Historical on 3090, Bella on 90HX — prevents catastrophic forgetting
- **Lower LR**: 5e-5 for historical vs 3e-4 for synthetic
- **Benchmark CSV**: time series tracks if metrics degrade across merges

---

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `train_easy.py` | Modify | Add `--db-path`, `--start-date`, `--end-date`, `--assets-file`, `--no-checkpoint` flags |
| `train_easy.py` (OnTheFlyGenerator) | Modify | Add `historical_market` param, historical branch in `_refill_buffer()` |
| `mcts_trader/data/history.py` | Modify | Add `get_window_random(length)` and `get_window(start_idx, length)` — return close prices `(T, n_assets)` |
| `tools/orchestration/e-dojo_v1.py` | Modify | Add `--branch-a-historical`, `--db-path`, `--schedule`, period management, benchmark-after-merge, overlap validation |

---

## Validation & Acceptance Criteria

- [ ] `train_easy.py --db-path trading_data.db --start-date 2023-01-01 --end-date 2023-03-01 --preset large --epochs 5` runs without crash
- [ ] Oracle produces reasonable targets (not all-HOLD, not all-TRADE) — log agree_trade > 0
- [ ] Encoder handles historical close prices without NaN/Inf
- [ ] No benchmark period appears in any training period (startup assertion)
- [ ] `benchmark_easy.py --checkpoint merged.pkl --json out.json` produces valid results after historical training
- [ ] e-dojo runs with `--branch-a-historical` (3090) + Bella (90HX), advances periods
- [ ] Benchmark runs automatically after each merge, saves JSON + appends CSV
- [ ] Benchmark CSV shows progression (agree_trade trending up over merges)
- [ ] agree_trade > 0% on benchmark after 50 epochs of historical training

---

## Estimated Scope

| Component | Effort | Risk |
|-----------|--------|------|
| CLI flags in train_easy.py | Small | Low |
| Historical branch in _refill_buffer() | Small | Low (same pattern as Bella branch) |
| HistoricalMarket.get_window_random() | Small | Low (~10 lines, array slicing) |
| e-dojo historical mode + benchmark | Medium | Medium (orchestrator state: period cursor, overlap check) |

**Total**: ~1 session

---

## Launch Command (Target)

```bash
# Full pipeline: historical on 3090, Bella on 90HX, benchmark after each merge
python tools/orchestration/e-dojo_v1.py --run --preset large \
  --branch-a-historical \
  --db-path trading_data.db \
  --schedule training_schedule.json \
  --dual-gpu \
  --epochs 2500
```
