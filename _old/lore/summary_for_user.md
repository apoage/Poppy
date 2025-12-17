# Summary: Training Pipeline Enhancements

## What I Did

### 1. Updated train_manager.py ✅
Added comprehensive asset selection features:

**New Arguments**:
- `--random-subset N` - Select N random assets from database
- `--exclude-forex` - Filter out forex (fiat/fiat) pairs like EUR/USD
- `--date-start / --date-end` - Filter data by time window
- `--min-candles N` - Require minimum data per asset
- `--min-coverage PCT` - Set minimum asset overlap requirement

**New Functions**:
- `is_forex_pair()` - Detects and filters fiat/fiat trading pairs
- Enhanced SQL queries with date filtering and candle count requirements

### 2. Created Documentation 📝

**lore/diary.md**:
- Detailed explanation of two-phase training strategy
- Technical implementation notes
- Monitoring checklist
- Data quality analysis

**lore/gemini_instructions.md**:
- Step-by-step implementation guide for Gemini
- File-by-file changes needed
- Testing procedures
- Troubleshooting guide

**TRAINING_COMMANDS.md**:
- Quick reference for training commands
- Phase 1 and Phase 2 examples
- Troubleshooting tips
- Data range reference

### 3. Tested Implementation ✅
Dry-run test successful:
```
✅ Selected 32 random assets from 333 candidates
✅ Excluded 8 forex pairs
✅ Config written with min_coverage
✅ No errors
```

---

## What Gemini Needs to Do

The `train_manager.py` changes are complete, but the integration isn't finished. Gemini needs to:

1. **Update train_jax_fast.py** - Read `min_coverage` from config and calculate `min_required_assets`
2. **Update FastTradingEnv** - Accept and pass `min_required_assets` parameter
3. **Verify HistoricalMarket** - Ensure it uses the parameter correctly
4. **Test the pipeline** - Run a 1-epoch test to verify everything works

See `lore/gemini_instructions.md` for complete details.

---

## Your Two-Phase Training Strategy

### Phase 1: Pre-Training (2-4 days)
```bash
python train_manager.py \
  --random-subset 64 \
  --exclude-forex \
  --date-start 2024-03-30 \
  --date-end 2025-03-30 \
  --min-candles 262800 \
  --min-coverage 0.85 \
  --steps 1440 \
  --epochs 10000 \
  --no-validate
```

**What this does**:
- Trains on 1 year of diverse market data
- Uses 24-hour episodes (1440 steps)
- Random 64 crypto assets (no forex)
- Requires 85% asset overlap
- Teaches general trading principles

### Phase 2: Fine-Tuning (0.5-1 day)
```bash
python train_manager.py \
  --random-subset 64 \
  --exclude-forex \
  --date-start 2025-03-01 \
  --date-end 2025-03-30 \
  --min-coverage 0.95 \
  --steps 200 \
  --epochs 2000 \
  --no-validate
```

**What this does**:
- Adapts to recent market conditions (Q1 2025)
- Uses 3.3-hour episodes (200 steps)
- Higher overlap requirement (95%)
- Fewer epochs (fine-tuning, not from scratch)

---

## Why This Strategy Works

### Forex Exclusion
- Forex pairs (EUR/USD, GBP/USD) behave differently from crypto
- Driven by central banks, not market sentiment
- Database may have incomplete forex data
- You want a pure crypto strategy

### Long Episodes (1440 steps)
- 1440 minutes = 24 hours = full daily cycle
- Learns Asia/Europe/US trading sessions
- Captures overnight momentum
- Better long-term planning

### Random Asset Selection
- Prevents overfitting to specific coins
- Learns general patterns, not coin-specific quirks
- More robust to new assets

### Two-Phase Approach
- Phase 1: Broad, diverse learning (foundation)
- Phase 2: Adapt to current market (fine-tuning)
- Best of both: robustness + relevance

---

## Data Quality Notes

Your database:
- **Size**: 20GB
- **Coverage**: Ends around March 30, 2025
- **Best period**: 2018-2024 (asset-rich)
- **Recommended window**: Last 1 year (March 2024 - March 2025)
  - This gives 90% valid pairs
  - Includes recent market regime

---

## Next Steps

1. **Give Gemini the instructions**:
   - Share `lore/gemini_instructions.md`
   - Ask them to complete Tasks 1-4
   - Verify with test commands

2. **After Gemini completes**:
   - Test with quick 1-epoch run
   - If successful, start Phase 1 training
   - Monitor progress in `logs/train_manager.log`

3. **After Phase 1 completes**:
   - Save checkpoint
   - Run Phase 2 fine-tuning
   - Validate on paper trading

---

## Testing Before Full Training

### Dry-run (instant)
```bash
python train_manager.py \
  --random-subset 64 \
  --exclude-forex \
  --date-start 2024-03-30 \
  --date-end 2025-03-30 \
  --min-coverage 0.85 \
  --no-validate \
  --dry-run
```
Verify asset selection and config generation.

### Quick test (1 minute)
```bash
python train_manager.py \
  --random-subset 32 \
  --exclude-forex \
  --date-start 2024-12-01 \
  --date-end 2024-12-31 \
  --min-coverage 0.8 \
  --steps 200 \
  --epochs 1 \
  --no-validate
```
Verify full pipeline works end-to-end.

---

## Current Status

✅ **Complete**:
- train_manager.py enhanced with all requested features
- Forex filtering implemented
- Date range selection working
- Documentation created
- Dry-run test successful

⏳ **Pending** (for Gemini):
- Update train_jax_fast.py to read min_coverage
- Update FastTradingEnv to accept parameter
- Verify HistoricalMarket integration
- End-to-end testing

🎯 **Goal**:
- Phase 1: Robust pre-trained model
- Phase 2: Fine-tuned for recent market
- Ready for paper trading validation

---

## Files Created/Modified

**Modified**:
- `train_manager.py` - Enhanced asset selection

**Created**:
- `lore/diary.md` - Training diary and strategy docs
- `lore/gemini_instructions.md` - Step-by-step implementation guide
- `lore/summary_for_user.md` - This summary
- `TRAINING_COMMANDS.md` - Quick reference commands
- `config/current_training.json` - Generated config (from dry-run)

---

## Questions?

See the documentation files for details:
- Strategy explanation → `lore/diary.md`
- Implementation steps → `lore/gemini_instructions.md`
- Quick commands → `TRAINING_COMMANDS.md`
- This summary → `lore/summary_for_user.md`
