# Quick Start: Apply GPU Optimizations

## Current Problem
- GPU: 18% utilization (only 741MB/24GB VRAM used)
- CPU: 81% (bottlenecked)
- Training extremely slow

## Root Cause
1. State encoder running on CPU (heavy NumPy operations)
2. 100 MCTS simulations per step (CPU-bound tree search)
3. Only 4 parallel collectors (not enough to saturate GPU)

## Quick Fix (5 minutes, 10x speedup)

### Step 1: Stop Current Training
```bash
kill 26421
# Or check current PID:
cat current_training_pid.txt
kill $(cat current_training_pid.txt)
```

### Step 2: Edit Config File
File: `mcts_trader/config/trading_config.py`

Change these two lines:
```python
# Line 9: Increase from 4 to 16
collector_env_num=16,

# Line 22: Reduce from 100 to 50
num_simulations=50,
```

### Step 3: Edit Environment File
File: `mcts_trader/env/trading_env.py`

Change line 97:
```python
# BEFORE:
self.encoder = MultiAssetStateEncoder(n_assets=n_assets, lookback=lookback, device='cpu')

# AFTER:
self.encoder = MultiAssetStateEncoder(n_assets=n_assets, lookback=lookback, device='cuda')
```

### Step 4: Restart Training
```bash
python main.py > training_console.log 2>&1 &
echo $! > current_training_pid.txt
```

### Step 5: Monitor GPU (should be 60-80% now)
```bash
watch -n 1 nvidia-smi
```

## Expected Results

**BEFORE:**
- GPU: 18% utilization
- VRAM: 741MB
- Speed: ~10-20 steps/minute

**AFTER:**
- GPU: 60-80% utilization
- VRAM: 4-6GB
- Speed: 200-400 steps/minute (10-20x faster)
- First batch (512): ~2-3 minutes (vs 30+ minutes)

## If You Want Even More Speed (20x speedup)

See `agents/claude/optimizations.patch` for advanced optimizations:
- Switch to FastStateEncoder (fully GPU-accelerated)
- Enable mixed precision (TF32/BF16)
- Additional 2x speedup

## Files Changed
1. `mcts_trader/config/trading_config.py` (2 lines)
2. `mcts_trader/env/trading_env.py` (1 line)

## Detailed Analysis
See `agents/claude/bottleneck_analysis.md` for complete technical analysis.
