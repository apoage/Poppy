# Session Notes: 2026-01-03

## Git Storage Fix

**Problem**: Main SSD running out of space (81% usage, ~27GB .git/objects)

**Solution**: Moved git objects to external storage via symlink
```bash
rsync -av .git/objects/ /mnt/storage/Trader_versionning/git/objects/
rm -rf .git/objects
ln -s /mnt/storage/Trader_versionning/git/objects .git/objects
```

**Result**: Freed ~28GB on main SSD (81% → 68% usage)

---

## JAX Environment Scaling Analysis

### Benchmark Results

| Envs | Self-play | Samples/sec | Notes |
|------|-----------|-------------|-------|
| 4    | 2.5s      | 320         | GPU underutilized |
| 10   | 3.0s      | 667         | Good scaling |
| 20   | 3.5s      | 1143        | Sweet spot |
| 50   | 6.3s      | 1587        | **Peak throughput** |
| 100  | 25s       | 800         | Bandwidth bottleneck |

### Root Cause

Per-step GPU→CPU transfer overhead:
- 100 envs × 200 steps × 442KB obs = **8.8GB per epoch**
- Current: 200 small transfers (massive overhead)
- Optimal: 1 bulk transfer (or batched)
- Result: 14x slower than theoretical

### Hardware Limits (Measured)

```
RAM internal:      5-10 GB/s (sequential write: 9.8 GB/s, read+write: 4.7 GB/s)
PCIe H2D:          5.57 GB/s (RAM → VRAM)
PCIe D2H:          5.22 GB/s (VRAM → RAM) ← bottleneck
VRAM internal:     770 GB/s (fast!)
VRAM capacity:     24 GB (RTX 3090)
```

### Optimization Path

| Phase | Change | Expected Improvement |
|-------|--------|---------------------|
| 1 | fp16 storage | 40% (halves bandwidth) |
| 2 | Batched transfers | 68% |
| 3 | Streaming trajectories | 76% |
| 4 | GPU synthetic generation | 80%+ |

Target: 100 envs @ ~5s (from 25s)

---

## Historical Data Training Strategy

### Critical Constraint

**Cross-asset correlation must be preserved.**

All 64 assets need to be time-synchronized for correlation learning in latent space.
Random time sampling breaks this → each sample sees different correlation structure.

### Solution: Sequential Day Batching

```
Epoch 1: 30 envs see days 1-30 (each env = different sequential day)
Epoch 2: 30 envs see days 31-60
...
```

Benefits:
- Correlation preserved (sequential, not random)
- 30 days of market exposure per epoch
- 80 years / 30 days = ~970 epochs unique data

### Data Math

```
Historical data:     ~80 years continuous (29,200 days)
Data augmentation:   4x (original, inverted, reversed, both)
Effective dataset:   ~116,800 days
At 30 days/epoch:    ~3,893 epochs of unique data
```

### Warmup Strategy

**Problem**: JIT warmup + buffer filling wastes valuable historical data.

**Solution**: Warmup on synthetic, then switch to historical:

```bash
# Phase 1: Warmup (synthetic)
python train_jax_fast.py --synthetic-level 0 --epochs 10 --parallel-envs 30 --jax-env --name warmup

# Phase 2: Historical training
python train_jax_fast.py --db-path trading_data.db --load-checkpoint jax_ckpt_fast/warmup \
    --clear-buffer --sequential-days --parallel-envs 30 --name historical
```

### Recommended Historical Training Config

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `--parallel-envs` | 30 | 30 days/epoch = 1 month coverage |
| `--steps` | 1440 | 1 day of 1-min candles |
| `--sequential-days` | True | Preserve cross-asset correlation |
| `--fp16-storage` | True | Reduce bandwidth (19GB → 9.5GB) |

---

## Reward/Value Handling

Confirmed: Per-sample targets preserved (not averaged across days)

- Day 1 (+10% opportunity) → high value targets
- Day 15 (-5% crash) → low value targets
- Model learns to distinguish different market regimes
- Loss `mean()` is over batch (standard SGD), targets stay per-sample

**Important signals NOT smoothed out.**

---

## Tasks Created/Updated

- **TASK_027**: JAX-vectorized environment - COMPLETE (committed: 59581e9)
- **TASK_028**: Streaming data pipeline - Updated with sequential-days strategy
- **TASK_029**: 100+ environment scaling - NEW

---

## Commits

```
59581e9 TASK_027: JAX-vectorized trading environment (Phases 0-4)
```

---

---

## Hyperparameter Tournament Results (2026-01-01)

33 runs total across 4 rounds.

### Round A: Prediction Variants
| Variation | Avg Reward | Notes |
|-----------|------------|-------|
| no_pred   | -6.18 | Baseline |
| physics   | -4.48 | Physics-only |
| **hybrid** | **-4.38** | Physics + neural (winner) |
| tree      | -6.55 | Tree-based |

### Round B: Value Loss Variants
| Variation | Avg Reward | Notes |
|-----------|------------|-------|
| baseline  | -4.21 | Standard MSE |
| huber     | -5.19 | Worse than MSE |
| **popart** | **-2.96** | **Best overall single change** |

### Round C: LR Warmup
| Variation | Avg Reward | Notes |
|-----------|------------|-------|
| no_warmup | -5.54 | Baseline |
| warmup_fast | -5.23 | Slight improvement |
| **warmup_slow** | **-3.83** | +30% vs no_warmup |

### Round D: Combined Best
| Variation | Avg Reward | Notes |
|-----------|------------|-------|
| combined (hybrid+popart+warmup_slow) | -5.60 | **Worse than individuals!** |

### Key Insights

1. **PopArt is the biggest win**: -2.96 vs baseline -4.21 (30% improvement)
2. **Optimizations don't stack**: Combined (-5.60) worse than popart alone (-2.96)
3. **Slow warmup helps**: -3.83 vs -5.54 when used alone
4. **Hybrid prediction marginal**: -4.38 vs physics -4.48

### Recommended Config for Full Curriculum

Based on results, **popart alone** is the safest choice:
```bash
python train_jax_fast.py \
    --synthetic-level 0 \
    --use-popart \
    --jax-env \
    --parallel-envs 30 \
    --epochs 500 \
    --name curriculum_L0_popart
```

Alternative: Test popart + warmup_slow (without hybrid) to see if those two stack better.

---

## Open Questions

1. Implement `--sequential-days` flag for historical data loader
2. Test fp16 precision impact on training stability
3. Profile exact overhead of per-step transfers vs batched
4. GPU synthetic generation for higher curriculum levels (not just sine)
5. Why does combined config perform worse than individuals? Interaction effects?
6. Test popart + warmup_slow without hybrid prediction
