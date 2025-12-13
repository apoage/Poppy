# Gemini Implementation Checklist

## Files to Create

- [ ] `tools/map_training_data.py` - Database mapper (main tool)
- [ ] `tools/run_scheduled_training.py` - Training automation script
- [ ] `training_schedule.json` - Generated output (run mapper to create)

## Files to Modify

- [ ] `train_manager.py` - Add `--training-schedule`, `--period-id`, `--run-id` arguments
- [ ] `mcts_trader/data/history.py` - Support `date_start_ts` and `date_end_ts` parameters

## Implementation Steps

### Step 1: Create Mapper Tool
- [ ] Create `tools/map_training_data.py`
- [ ] Implement `find_initial_threshold()`
- [ ] Implement `map_valid_periods()`
- [ ] Implement `generate_asset_combinations()`
- [ ] Implement `build_training_schedule()`

### Step 2: Test Mapper
- [ ] Run: `python tools/map_training_data.py`
- [ ] Verify `training_schedule.json` created
- [ ] Check schedule has ≥5 periods
- [ ] Check total epochs > 50,000
- [ ] Verify overlap is 30-50%

### Step 3: Update train_manager.py
- [ ] Add new CLI arguments
- [ ] Add schedule loading logic
- [ ] Test with: `--training-schedule training_schedule.json --period-id 1 --run-id 1 --dry-run`

### Step 4: Update HistoricalMarket
- [ ] Modify `__init__` to accept timestamp parameters
- [ ] Filter valid_timestamps by range

### Step 5: Create Automation Script
- [ ] Create `tools/run_scheduled_training.py`
- [ ] Test running first 2 periods

### Step 6: Full Test
- [ ] Run single training session (5 epochs)
- [ ] Verify no "out of data" errors
- [ ] Verify correct asset count
- [ ] Check checkpoints saved

## Expected Results

After completing all steps:

```bash
# Generate schedule
python tools/map_training_data.py

# Output should show:
# ✓ Found 8 valid periods
# Total epochs available: 87,450
```

```bash
# Test training
python train_manager.py \
  --training-schedule training_schedule.json \
  --period-id 1 \
  --run-id 1 \
  --steps 1440 \
  --epochs 5 \
  --no-validate

# Should complete 5 epochs without running out of data
```

## Verification

- [ ] `training_schedule.json` exists and is valid JSON
- [ ] Schedule summary shows ≥50,000 total epochs
- [ ] Each period has 5-20 training runs
- [ ] Consecutive runs have 30-50% overlap
- [ ] Forex pairs excluded from all asset lists
- [ ] Training with schedule works (no crashes)
- [ ] No "No more valid timestamps" errors

## Common Issues

**Issue**: "Never found 64 concurrent assets"
**Fix**: Lower min_assets to 48 or 32 in `find_initial_threshold()`

**Issue**: "No periods found"
**Fix**: Lower min_duration_days to 14 days

**Issue**: Overlap always 0%
**Fix**: Check `generate_asset_combinations()` logic - should reuse assets from previous run

**Issue**: Training still runs out of data
**Fix**: Ensure `HistoricalMarket` uses `date_start_ts` and `date_end_ts` to filter timestamps

---

Good luck! Follow the spec in `FOR_GEMINI_TRAINING_MAPPER.md` step by step.
