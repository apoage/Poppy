# TASK_707: Period Cycling for Foxglove (Stratified Historical Training)

**Status**: DONE
**Priority**: 0 (CRITICAL — model is overfitting to one period, benchmark degrading every 100 epochs)
**Depends on**: TASK_705 (done), TASK_706 (done)

---

## Problem

Foxglove trains on a single historical period forever. `HistoricalScheduleManager.advance()` is only called after merges (dual-branch mode) — in single-branch mode the cursor never moves. Result: 790+ epochs on the same 180-day crypto winter (2018-10-17 → 2019-04-15, 61 assets). Training agreement climbs to 94% while benchmark PnL goes from -75% → -388%.

Dojo v3 already solves this with `--periods-file` + stratified cycling in `train_jax_fast.py`. We need the same for Foxglove (`train_easy.py`).

---

## Design

Adapt dojo v3's stratified logic: load multiple periods upfront, cycle through them epoch-by-epoch inside `train_easy.py`. No subprocess restarts, no orchestrator complexity.

### How dojo v3 does it (reference: `dojo_v3_historical.py:586-752`)

1. Orchestrator batches N runs from `training_schedule.json`
2. Writes them to a temp JSON (`--periods-file`)
3. `train_jax_fast.py` loads all periods into memory at startup
4. Each epoch picks next period (round-robin)
5. After batch exhausted, orchestrator writes next batch, launches again

### What Foxglove needs

Same pattern but simpler — Foxglove's `OnTheFlyGenerator` already manages a `HistoricalMarket` that serves random windows. We just need to **swap which HistoricalMarket is active** each epoch.

---

## Implementation

### Phase 1: `--periods-file` in train_easy.py

**New CLI flag:**
```
--periods-file PATH    JSON file with multiple periods to cycle through
```

**JSON format** (same as dojo v3):
```json
{
  "periods": [
    {"start_date": "2018-10-17", "end_date": "2019-04-15", "assets": ["BTC/USD", "ETH/USD", ...]},
    {"start_date": "2019-04-15", "end_date": "2019-10-12", "assets": ["BTC/USD", "ETH/USD", ...]},
    {"start_date": "2019-10-12", "end_date": "2020-04-09", "assets": ["BTC/USD", "LTC/USD", ...]},
    ...
  ]
}
```

**Startup** (`main()`, around line 1897):
- If `--periods-file` given, load all periods
- Create one `HistoricalMarket` per period (each caches its data in RAM)
- RAM budget: ~12 MB per period × 171 periods = ~2 GB total (fits easily)
- Pass the list of markets to `OnTheFlyGenerator`

**Epoch cycling** (`OnTheFlyGenerator._refill_buffer`, around line 1191):
- Add `current_period_idx` to generator state
- Each epoch (or each buffer refill cycle), advance to next period: `idx = (idx + 1) % len(markets)`
- Swap `self.historical_market` to `markets[idx]`
- Log period switch: `"[PERIOD] Switching to P{idx}: {start} → {end} ({n_assets} assets)"`

**Validation generator**: Same cycling, but offset by half to avoid train/val seeing same period simultaneously.

### Phase 2: e-dojo generates the periods file

**In `e-dojo_v1.py`** (`single_mode and args.branch_a_historical` path, around line where single-branch is launched):

1. Load full `training_schedule.json` via `HistoricalScheduleManager`
2. Flatten all 171 runs into periods JSON (same as `dojo_v3._flatten_runs`)
3. Write to `/tmp/foxglove_periods.json`
4. Pass `--periods-file /tmp/foxglove_periods.json` to train_easy.py
5. Remove `--start-date`, `--end-date`, `--assets-file` (replaced by periods-file)

This replaces the current single-period launch. No more `schedule_manager.advance()` needed — cycling happens inside train_easy.py.

### Phase 3: Cleanup

- Remove `HistoricalScheduleManager` usage from e-dojo single-branch path (no longer needed for period tracking)
- Keep `--start-date`/`--end-date`/`--assets-file` working for single-period debugging
- `--periods-file` and `--start-date` are mutually exclusive

---

## Memory Considerations

Loading all 171 periods into RAM at once:
- Each period: ~12 MB (166K candles × 64 assets × 5 channels × 4 bytes)
- 171 periods: ~2 GB
- System has 64 GB RAM → fine

Alternative if RAM is tight: load only N periods at a time (e.g., 20), swap lazily. But 2 GB is nothing — load all.

**IMPORTANT**: Each `HistoricalMarket` opens a DB connection during init. Close connections after caching to RAM — only the numpy cache is needed for training.

---

## Files

| File | Change |
|------|--------|
| `train_easy.py` | Add `--periods-file` flag, multi-market loading, epoch cycling in OnTheFlyGenerator |
| `tools/orchestration/e-dojo_v1.py` | Generate periods JSON from schedule, pass `--periods-file` instead of single period |
| `mcts_trader/data/history.py` | Maybe: add `close_db()` method to release connection after cache loaded |

---

## Acceptance Criteria

- [ ] `--periods-file` loads all periods into RAM at startup
- [ ] Training cycles through periods every epoch (round-robin)
- [ ] Period switch logged with date range and asset count
- [ ] Benchmark (TASK_706) still runs every 100 epochs on separate GPU
- [ ] Training covers full 2018-2025 range, not stuck on one period
- [ ] `--start-date`/`--end-date` still work for single-period mode
- [ ] No subprocess restarts — all cycling is internal
- [ ] RAM usage reasonable (~2 GB for all periods)

---

## Launch Command (after implementation)

```bash
python tools/orchestration/e-dojo_v1.py \
  --run --single --preset large --epochs 10000 \
  --branch-a-historical \
  --benchmark-every 100 --benchmark-gpu 1
```

Same command as before — e-dojo internally generates the periods file and passes it. User experience unchanged.
