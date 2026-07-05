# Dojo Quick Start Guide

**TL;DR**: Modular curriculum training system for trading agent

---

## Installation (Already Done)

All code is ready. No installation needed.

---

## Usage

### 1. Test Single Level

```bash
# Train on Level 0 (sine waves - basic trend detection)
python dojo_manager.py --level 0
```

**What happens**:
- Generates synthetic sine wave data (64 assets, 10000 steps)
- Trains agent for 500 epochs
- Saves checkpoint to `dojo_checkpoints/`
- Logs to `logs/dojo_training.log`

**Expected time**: ~10-20 minutes (RTX 3090)

### 2. Full Curriculum

```bash
# Train Levels 0 → 1 → 2 → 3 sequentially
python dojo_manager.py
```

**What happens**:
- Level 0: Sine waves (500 epochs)
- Level 1: Lagged pairs (1000 epochs)
- Level 2: Dip crashes (1500 epochs)
- Level 3: Noisy trends + swaps (2000 epochs)
- Total: ~5000 epochs across 4 levels

**Expected time**: ~2-3 hours (RTX 3090)

### 3. Resume Training

```bash
# Resume from last completed level
python dojo_manager.py --resume
```

**What happens**:
- Reads `dojo_checkpoints/curriculum_progress.json`
- Continues from next level

### 4. Custom Curriculum

Create `config/my_curriculum.json`:
```json
[
  {
    "level": 0,
    "generator": "sine",
    "tactics": ["basic"],
    "epochs": 100,
    "n_assets": 32,
    "n_steps": 5000
  }
]
```

Run:
```bash
python dojo_manager.py --curriculum config/my_curriculum.json
```

---

## Monitoring

### Logs

```bash
# Dojo orchestration log
tail -f logs/dojo_training.log

# Training output (for current level)
tail -f logs/training_fast_dojo_level_*.log
```

### Checkpoints

```bash
# Training checkpoints (per level)
ls -lh jax_ckpt_fast/dojo_level_*

# Curriculum progress
cat dojo_checkpoints/curriculum_progress.json
```

### Metrics

```bash
# Last run metrics
cat last_run_metrics.json
```

---

## Testing

### Test Swap Mechanics

```bash
python tools/test_portfolio_swaps.py
```

**Expected output**:
```
Testing Swap Mechanics...
Buying Asset 0...
Holdings after buy: 4950.0000 shares of Asset 0
Swapping Asset 0 -> Asset 1...
Holdings after swap:
  Asset 0: 0.0000
  Asset 1: 4941.8400
✅ Swap mechanics working
```

### Visualize Synthetic Data

```bash
python tools/visualize_synthetic.py --level 0
```

**Generates**: `synthetic_level_0_preview.png`

---

## Curriculum Levels

| Level | Generator | Tactics | Epochs | Description |
|-------|-----------|---------|--------|-------------|
| 0 | Sine waves | Basic | 500 | Trend detection |
| 1 | Lagged pairs | Basic | 1000 | Correlation detection |
| 2 | Dip crashes | Basic | 1500 | Patience/timing |
| 3 | Noisy trends | Basic + Swaps | 2000 | Fee optimization |

---

## Architecture

```
dojo_manager.py
├── Curriculum config (JSON)
├── Synthetic generator (mcts_trader/data/synthetic_market.py)
├── Tactics (mcts_trader/tactics/basic_trading.py, portfolio_morphing.py)
└── Training loop (train_jax_fast.py)
```

**Key insight**: Tactics are modular - can be composed:
- Level 0-2: `tactics: ["basic"]`
- Level 3+: `tactics: ["basic", "morphing"]`

---

## Troubleshooting

### "No valid pairs found"
- Not applicable (synthetic data, no Kraken validation)

### "CUDA out of memory"
- Reduce `batch_size` in curriculum config
- Or reduce `n_assets` (default 64)

### "Swap mechanics failing"
- Check logs for Position attribute errors
- Should be using `.size` not `.shares` (fixed)

### "Training not converging"
- Check `logs/training_fast_dojo_level_*.log` for NaN losses
- Ensure synthetic data is valid (no NaNs/Infs)

---

## Next Steps

After validating Levels 0-3:
1. Add Levels 4-22 (see `lore/MATHEMATICAL_TRADING_TACTICS_CURRICULUM.md`)
2. Add advanced tactics (pairs trading, market making)
3. Real-data curriculum (transition from synthetic to real Kraken data)

---

## Files Created

```
dojo_manager.py                              # Curriculum orchestrator
mcts_trader/tactics/__init__.py
mcts_trader/tactics/basic_trading.py         # Modular tactic
mcts_trader/tactics/portfolio_morphing.py    # Modular tactic
lore/DOJO_MODULAR_ARCHITECTURE.md            # Architecture design
DOJO_REFACTOR_STATUS.md                      # Detailed status
REFACTOR_SUMMARY.md                          # Summary
QUICK_START_DOJO.md                          # This file
```

---

## Example Session

```bash
# 1. Test single level
python dojo_manager.py --level 0

# Check logs
tail -f logs/dojo_training.log

# Expected output:
# [DOJO] - Starting curriculum training (1 levels)
# [DOJO] - LEVEL 0: sine
# [DOJO] - Tactics: ['basic']
# [DOJO] - Creating synthetic generator: sine (Level 0)
# [DOJO] - Training args: {'synthetic_level': 0, ...}
# [DOJO] - Running command: python train_jax_fast.py --synthetic-level 0 --epochs 500 ...
# ... (training output)
# [DOJO] - Level 0 COMPLETED in 12.3 minutes

# 2. Check results
cat last_run_metrics.json

# 3. If successful, run full curriculum
python dojo_manager.py
```

---

**Status**: ✅ Ready for testing
**First command**: `python dojo_manager.py --level 0`
