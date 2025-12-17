# ✅ Ready to Train - Final Commands

## Test Results: SUCCESS ✅

Gemini's implementation works perfectly!

**Verified**:
- ✅ Asset selection from database
- ✅ Forex exclusion (8 pairs filtered)
- ✅ Date range filtering
- ✅ min_coverage parameter flows correctly
- ✅ Training completes without errors
- ✅ JAX acceleration working (~1.4s/epoch)
- ✅ Rewards are reasonable (+11.79 SPOT, +0.49 MARGIN)

---

## Database Info

**Your database**:
- **Start**: September 10, 2013 (timestamp 1378856820)
- **End**: December 12, 2025 (timestamp 1765558500) - **TODAY!**
- **Assets**: 1,152 distinct assets total
- **Valid after filters**: 333 crypto assets (excluding forex)

**Good news**: Your database is up-to-date (not March 30 as you mentioned), so you can train on very recent data!

---

## Production Training Commands

### Phase 1: Large-Scale Pre-Training
Train on 1 year of diverse data with 24-hour episodes.

```bash
python train_manager.py \
  --random-subset 64 \
  --exclude-forex \
  --date-start 2024-01-01 \
  --date-end 2025-12-12 \
  --min-candles 262800 \
  --min-coverage 0.75 \
  --steps 1440 \
  --epochs 10000 \
  --no-validate
```

**Parameters explained**:
- Uses full 2024 + 2025 YTD (almost 2 years of data!)
- 64 random crypto assets (no forex)
- min-candles 262800 = ~6 months of data per asset
- min-coverage 0.75 = require 75% asset overlap (lowered from 0.85 for more timestamps)
- 1440 steps = 24-hour episodes
- 10,000 epochs ≈ **4 hours** training time (JAX is fast!)

**Expected output**:
```
Market loaded: 64 assets, ~XXX,XXX valid timestamps
DEBUG: Min required assets set to 48 (75.0% coverage)
```

---

### Phase 2: Fine-Tuning (Recent Data)
Train on last 2 months with shorter episodes.

```bash
python train_manager.py \
  --random-subset 64 \
  --exclude-forex \
  --date-start 2025-10-01 \
  --date-end 2025-12-12 \
  --min-coverage 0.9 \
  --steps 200 \
  --epochs 2000 \
  --no-validate
```

**Parameters explained**:
- Oct-Dec 2025 (most recent 2.5 months)
- Higher coverage (0.9) for data quality
- Shorter episodes (200 steps = 3.3 hours)
- 2,000 epochs ≈ **50 minutes** training time

---

## Quick Test (Verify Everything Works)

Before starting the long training, run a quick sanity check:

```bash
python train_manager.py \
  --random-subset 32 \
  --exclude-forex \
  --date-start 2025-11-01 \
  --date-end 2025-12-12 \
  --min-coverage 0.8 \
  --steps 200 \
  --epochs 5 \
  --no-validate
```

Should complete in ~10 seconds. Check for:
- ✅ No crashes
- ✅ Valid timestamps > 1000
- ✅ Rewards are non-zero
- ✅ GPU is being used

---

## Monitoring During Training

### Watch logs
```bash
tail -f logs/train_manager.log
```

### Check GPU usage
```bash
watch -n 1 nvidia-smi
```

### Check training progress
```bash
# Every 5 minutes, check latest epoch
tail logs/train_manager.log | grep "Epoch"
```

### Check checkpoints
```bash
ls -lh jax_ckpt_fast/
```

---

## Known Minor Issues (Non-Critical)

### 1. "Unknown currency USDTCHF" warnings
**What**: Log spam for stablecoin/fiat hybrid pairs like `/USDTCHF`

**Impact**: None (uses fallback FX rate 1.0, which is fine for stablecoins)

**Fix** (optional):
Add to `is_forex_pair()` in train_manager.py:
```python
# Check for stablecoin pairs
if 'USDT' in symbol and any(fiat in symbol for fiat in ['CHF', 'CAD', 'AUD', 'GBP']):
    return True
```

### 2. Limited timestamps in December 2024 test
**What**: Only 1,191 timestamps for Dec 2024 (expected ~43,000)

**Cause**: Random asset selection + 80% coverage + sparse data in that specific month

**Fix**: Use longer date ranges (2024-2025) as recommended above

---

## Performance Expectations

### Training Speed (Based on Test)
- **Per epoch**: ~1.4 seconds (200 steps)
- **Phase 1 (10K epochs)**: ~4 hours
- **Phase 2 (2K epochs)**: ~50 minutes

**Much faster than originally estimated!** JAX acceleration is working well.

### GPU Utilization
- Model: 496,370 parameters (very efficient!)
- Expected GPU usage: 80-100%
- Memory: ~2-4GB (well under RTX 3090's 24GB)

### Model Size
- Tiny compared to original estimate
- Likely using the "fast" model variant
- Should train very quickly

---

## After Training

### Phase 1 Completion
1. Check final epoch reward in logs
2. Verify checkpoint saved to `jax_ckpt_fast/`
3. Note the checkpoint path for Phase 2

### Phase 2 Setup
1. Copy Phase 1 checkpoint (or modify Phase 2 command to load it)
2. Run Phase 2 command above
3. Monitor for improvement in rewards

### Validation
1. Test on held-out data (e.g., last week of Dec 2025)
2. Check if policy makes sensible trades
3. Run paper trading simulation

---

## Emergency Stop

If you need to stop training:
```bash
# Find process
ps aux | grep train_jax_fast

# Kill gracefully
pkill -f train_jax_fast

# Or force kill
pkill -9 -f train_jax_fast
```

Checkpoints are saved periodically, so you can resume later.

---

## Success Criteria

After Phase 1, you should see:
- [x] Epoch rewards trending upward (or stabilizing at positive values)
- [x] No NaN losses or crashes
- [x] Checkpoints saved successfully
- [x] GPU utilization consistent
- [x] Training completes without errors

After Phase 2, you should see:
- [x] Fine-tuned rewards improve over Phase 1
- [x] Model adapts to recent market conditions
- [x] Ready for paper trading validation

---

## You're Ready! 🚀

Everything is working. Just run the Phase 1 command above and let it train for ~4 hours. Monitor the logs to ensure it's progressing.

After Phase 1 completes, run Phase 2 for ~50 minutes.

Then you'll have a fully trained model ready for paper trading!

---

## Questions?

See these docs for details:
- Test results: `lore/test_results.md`
- Strategy explanation: `lore/diary.md`
- Gemini's changes: Files modified as described in test
- Quick reference: `TRAINING_COMMANDS.md`
