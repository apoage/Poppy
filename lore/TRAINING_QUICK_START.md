# Training Quick Start Guide

## Phase 1: Pre-Training (START HERE)

### Option 1: Use Script (Recommended)
```bash
bash START_TRAINING.sh
```

### Option 2: Direct Command
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

### What This Does
- **Selects 64 random crypto assets** (excludes forex pairs like EUR/USD)
- **Date range**: Jan 1, 2024 → Dec 12, 2025 (~2 years of data)
- **Episode length**: 1440 steps = 24 hours of trading on 1-minute candles
- **Training**: 10,000 epochs ≈ **4 hours** on RTX 3090
- **Coverage**: Requires 75% asset overlap (48 of 64 assets) per timestamp

### Monitor Training
```bash
# Watch logs in real-time
tail -f logs/train_manager.log

# Watch GPU usage
watch -n 1 nvidia-smi

# Check progress
grep "Epoch" logs/train_manager.log | tail -20
```

### What to Expect

**Startup (first 30 seconds)**:
```
Selected 64 random assets from 333 candidates
Market loaded: 64 assets, XXX,XXX valid timestamps
DEBUG: Min required assets set to 48 (75.0% coverage)
[1/4] Initializing Environment...
✅ Environment ready
[2/4] Initializing JAX model...
✅ Model ready (496,370 params)
```

**Training Progress**:
```
Epoch    1 | Reward:   +X.XX | Steps: 1440 | Self-play:   1.4s
Epoch   10 | Reward:  +XX.XX | Steps: 1440 | Self-play:   1.4s
...
Epoch 10000 | Reward: +XXX.XX | Steps: 1440 | Self-play:   1.4s
```

**Completion**:
```
Experiment spot_baseline_v1 COMPLETED in XXX.X minutes
Experiment margin_standard_v1 COMPLETED in XXX.X minutes
All experiments finished.
```

### Success Criteria

✅ No crashes or errors
✅ Rewards generally positive and/or improving
✅ Logs are clean (no USDTCHF spam)
✅ GPU utilization 80-100%
✅ Checkpoints saved to `jax_ckpt_fast/`

---

## Phase 2: Fine-Tuning (After Phase 1)

### Option 1: Use Script
```bash
bash START_TRAINING_PHASE2.sh
```

### Option 2: Direct Command
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

### What This Does
- **Same 64 assets** OR new random selection
- **Recent data**: Oct 1 → Dec 12, 2025 (last 2.5 months)
- **Shorter episodes**: 200 steps = 3.3 hours
- **Quick tuning**: 2,000 epochs ≈ **50 minutes**
- **Higher coverage**: 90% (58 of 64 assets) for data quality

---

## Troubleshooting

### "No valid timestamps found"
**Solution**: Lower `--min-coverage` to 0.6 or expand date range

### Training crashes
**Solution**: Check `logs/train_manager.log` for errors. Common issues:
- OOM: Reduce batch size in `train_jax_fast.py`
- Database locked: Close other processes accessing `trading_data.db`

### Training very slow
**Solution**:
- Verify GPU is being used: `nvidia-smi` should show activity
- Check database on SSD, not HDD
- Reduce `--steps` for faster iteration

### Logs still showing USDTCHF spam
**Solution**: Verify `mcts_trader/agent/portfolio.py` was saved correctly with the fix

---

## After Training

### Check Results
```bash
# View final rewards
grep "Epoch.*10000" logs/train_manager.log

# Check checkpoints
ls -lh jax_ckpt_fast/

# View full training summary
tail -100 logs/train_manager.log
```

### Validate Model
```bash
# TODO: Run paper trading validation
# python paper_trade_jax.py --checkpoint jax_ckpt_fast/latest.pkl
```

---

## Configuration Parameters Explained

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `--random-subset 64` | 64 assets | Select 64 random crypto pairs |
| `--exclude-forex` | flag | Skip EUR/USD, GBP/USD, etc. |
| `--date-start` | 2024-01-01 | Start of training data window |
| `--date-end` | 2025-12-12 | End of training data window |
| `--min-candles` | 262800 | Min ~6 months of data per asset |
| `--min-coverage` | 0.75 | Require 75% asset overlap |
| `--steps` | 1440 | Episode length (24 hours) |
| `--epochs` | 10000 | Number of training epochs |
| `--no-validate` | flag | Skip Kraken API validation |

---

## Expected Timeline

| Phase | Duration | Output |
|-------|----------|--------|
| Phase 1 Pre-Training | ~4 hours | Robust general trading model |
| Phase 2 Fine-Tuning | ~50 minutes | Adapted to recent market |
| **Total** | **~5 hours** | Production-ready model |

Much faster than originally estimated (2-4 days) thanks to JAX optimization! 🚀

---

## Next Steps After Completion

1. ✅ Verify checkpoints exist
2. ✅ Review final epoch rewards
3. ⏸️ Run paper trading validation
4. ⏸️ Test on held-out data
5. ⏸️ Deploy to live paper trading

Good luck! 🎯
