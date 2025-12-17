# Training Commands

## 🌈 Rainbow Table Scheduled Training (Recommended)

This method uses the pre-computed `training_schedule.json` to train on 170+ permutations of data, maximizing data usage (30k+ epochs available).

**Start Full Training:**
```bash
python tools/run_scheduled_training.py
```

**Run Test (First 2 runs only):**
```bash
python tools/run_scheduled_training.py --limit 2
```

**Regenerate Schedule:**
```bash
python tools/map_training_data.py
```

---

## Legacy / Manual Training

### 1. Train Fast (JAX + Fast Encoder)
**Optimized for speed.** Uses `FastMultiAssetStateEncoder` (CPU/Numba) and `JAXReplayBuffer`.


## Phase 1: Large-Scale Pre-Training
Train on 1 year of diverse market data with 24-hour episodes.

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

**Parameters explained**:
- `--random-subset 64`: Select 64 random crypto assets
- `--exclude-forex`: Skip fiat/fiat pairs (EUR/USD, etc.)
- `--date-start/end`: 1-year window ending at data cutoff
- `--min-candles 262800`: ~6 months of minute data (180 days × 1440 min/day)
- `--min-coverage 0.85`: Require 85% asset overlap for valid timestamps
- `--steps 1440`: 24-hour episodes (1440 minutes)
- `--epochs 10000`: ~83 hours of training @ 30sec/epoch
- `--no-validate`: Skip Kraken API check (historical data)

**Expected runtime**: 2-4 days on RTX 3090

---

## Phase 2: Fine-Tuning on Recent Data
Adapt to Q1 2025 market conditions with shorter episodes.

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

**Parameters explained**:
- Same assets (or resample)
- 1-month recent window
- Higher coverage (95% = stricter overlap)
- Shorter episodes (200 steps = 3.3 hours)
- Fewer epochs (fine-tuning, not scratch)

**Expected runtime**: 0.5-1 day on RTX 3090

---

## Dry Run Test (No Training)
Test asset selection and config generation without actually training:

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

Check output and `config/current_training.json`.

---

## Quick Test (1 Epoch)
Verify pipeline works end-to-end:

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

Should complete in ~1 minute. Check for errors.

---

## Custom Asset Selection Examples

### Select 128 assets from 2018-2024 (asset-rich period)
```bash
python train_manager.py \
  --random-subset 128 \
  --exclude-forex \
  --date-start 2018-01-01 \
  --date-end 2024-12-31 \
  --min-candles 525600 \
  --min-coverage 0.7 \
  --steps 1440 \
  --epochs 5000 \
  --no-validate
```

### Select specific pairs from CSV (no randomness)
```bash
python train_manager.py \
  --pairs datasets/top64_coverage.csv \
  --min-coverage 0.9 \
  --steps 1440 \
  --epochs 10000 \
  --no-validate
```

### Auto-select top volume pairs from Kraken (live)
```bash
python train_manager.py \
  --auto-select volume \
  --count 64 \
  --steps 200 \
  --epochs 5000
```
(Note: This queries Kraken API, not database)

---

## Monitoring During Training

### Check training progress
```bash
tail -f logs/train_manager.log
```

### Check JAX training logs
```bash
tail -f jax_ckpt_fast/training.log
```

### Monitor GPU usage
```bash
watch -n 1 nvidia-smi
```

### Check current config
```bash
cat config/current_training.json | jq
```

### Count valid timestamps for your date range
```bash
sqlite3 trading_data.db "SELECT COUNT(*) FROM ohlcv WHERE timestamp >= 1711756800 AND timestamp <= 1743379200"
```

---

## Troubleshooting

### "No valid timestamps found"
**Problem**: Asset selection resulted in no overlapping data.

**Solutions**:
1. Reduce `--min-coverage` (try 0.5-0.7)
2. Expand date range
3. Reduce `--min-candles`
4. Remove `--exclude-forex` temporarily

### "Only found X assets, requested Y"
**Problem**: Database doesn't have enough assets meeting criteria.

**Solutions**:
1. Reduce `--random-subset` count
2. Lower `--min-candles` threshold
3. Expand `--date-start` and `--date-end`
4. Check database content:
```bash
sqlite3 trading_data.db "SELECT symbol, COUNT(*) FROM assets a JOIN ohlcv o ON a.id=o.asset_id GROUP BY a.id ORDER BY COUNT(*) DESC LIMIT 20"
```

### Training crashes with OOM (Out of Memory)
**Problem**: GPU memory exhausted.

**Solutions**:
1. Reduce batch size in `train_jax_fast.py`
2. Reduce `--random-subset` (fewer assets = less memory)
3. Monitor with `nvidia-smi` to confirm
4. Check for memory leaks in logs

### Training is very slow
**Problem**: Episodes take too long.

**Solutions**:
1. Reduce `--steps` (1440 → 200 for faster iteration)
2. Reduce MCTS simulations in `trading_config_optimized.py`
3. Verify GPU is being used (`nvidia-smi` should show ~90% util)
4. Check database is on SSD, not HDD

---

## Data Range Reference

**Database Coverage**:
- Data ends: ~March 30, 2025 (timestamp 1743454140)
- 1 year before: March 30, 2024 (timestamp 1711756800)
- 6 months before: September 30, 2024 (timestamp 1727740800)
- 30 days before: March 1, 2025 (timestamp 1740787200)

**Recommended Training Windows**:
- Pre-training: 2024-03-30 to 2025-03-30 (1 year, 90% assets available)
- Fine-tuning: 2025-03-01 to 2025-03-30 (30 days, recent market)
- Historical: 2018-01-01 to 2024-12-31 (asset-rich era)

**Candle Count Estimates**:
- 1 day = 1,440 candles (minutes)
- 1 week = 10,080 candles
- 1 month = ~43,200 candles
- 6 months = ~262,800 candles
- 1 year = ~525,600 candles

Use `--min-candles` to ensure assets have sufficient history.

---

## Advanced: Dataset Rotation (Prevent Overfitting)

To rotate datasets during training (not yet implemented):

```python
# In train_jax_fast.py, add periodic resampling
if epoch % 1000 == 0:
    logging.info("Rotating asset dataset...")
    # Resample new random assets
    # Reload environment with new assets
```

This would require modifying `train_jax_fast.py` to support environment reloading.

---

## Success Checklist

✅ Dry-run completes without errors
✅ `config/current_training.json` contains `min_coverage`
✅ Training prints "DEBUG: Min required assets set to X"
✅ Market loads with >10,000 valid timestamps
✅ No NaN losses or crashes
✅ GPU utilization >80%
✅ Checkpoints saved to `jax_ckpt_fast/`
✅ Reward improves over epochs

---

For detailed implementation instructions, see `lore/gemini_instructions.md`.
For strategy explanation, see `lore/diary.md`.
