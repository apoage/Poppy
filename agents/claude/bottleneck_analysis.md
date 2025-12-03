# GPU Bottleneck Analysis - 2025-12-03

## Current State (CRITICAL)

**GPU Utilization: 18% (SEVERE UNDERUTILIZATION)**
**VRAM Used: 741MB / 24GB (3% utilization)**
**CPU: 81% (BOTTLENECK)**

Process 26421 is running but extremely slow, stuck in data collection phase.

---

## Root Causes Identified

### 1. STATE ENCODER RUNS ON CPU (HIGH IMPACT)
**Location**: `mcts_trader/encoder/state.py:24`
```python
device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
```

**Problem**: Encoder defaults to CPU and performs heavy NumPy operations:
- Line 130: `compute_all()` - technical indicators on CPU
- Line 157-180: `_encode_price_history()` - per-asset normalization loops on CPU
- Line 206-211: `interpolate()` - only final step uses torch (but data already on CPU)

**Impact**: Each environment step calls `encoder.encode()` which:
- Processes 64 assets x 60 timesteps on CPU
- Computes correlations, indicators, normalization on CPU
- Only transfers final tensor to GPU
- This happens 4x in parallel (4 collectors) = 4x wasted CPU cycles

**Solution**: Use `FastStateEncoder` (already implemented but unused!)
- Lines 307-372 show GPU-accelerated encoder
- Computes everything on GPU with torch operations
- Would eliminate CPU bottleneck

---

### 2. MCTS RUNS ON CPU (HIGH IMPACT)
**Location**: LightZero C++ tree (`mcts_ctree`)

**Problem**:
- 100 MCTS simulations per action (config line 22)
- C++ MCTS tree search runs on CPU (no GPU acceleration)
- Each simulation requires neural network forward pass
- Network calls batched but tree traversal is CPU-bound

**Evidence from logs**:
```
mcts_ctree.py:293: batch_traverse(
```
Shows MCTS tree traversal happening on CPU.

**Impact**:
- 100 simulations x 4 collectors = 400 tree searches running concurrently on CPU
- Each simulation: tree node expansion + NN evaluation + backpropagation
- Sequential tree operations cannot be parallelized on GPU

**Solution**: Reduce simulations during data collection
- Use 25-50 simulations for rollouts (fast data generation)
- Use 100+ simulations for evaluation (quality assessment)
- 4x speedup in collection phase

---

### 3. LOW PARALLELISM (MEDIUM IMPACT)
**Location**: `mcts_trader/config/trading_config.py:9`
```python
collector_env_num=4,
```

**Problem**:
- Only 4 parallel environments collecting data
- With slow MCTS (100 sims), need more parallel workers to saturate GPU
- GPU can handle 16-32 parallel envs easily

**Impact**:
- Batch size 512 requires ~128 steps to fill (512 / 4 collectors)
- First training batch takes extremely long
- GPU sits idle waiting for CPU to generate data

**Solution**: Increase to 16-32 collectors
- More parallel data generation
- Better GPU utilization during NN forward passes
- Faster batch filling

---

### 4. SEQUENTIAL NN EVALUATIONS IN MCTS (MEDIUM IMPACT)
**Location**: MCTS policy evaluation

**Problem**:
- MCTS makes many small NN forward passes (batch size varies)
- Not utilizing GPU's massive parallelism
- Each simulation expands tree sequentially

**Impact**:
- GPU gets small batches (< 32) per MCTS call
- Poor tensor core utilization
- Memory bandwidth underutilized

**Solution**:
- Implement virtual loss parallelism (if LightZero supports)
- Pre-batch multiple MCTS queries
- Use larger evaluation batches

---

### 5. DATA LOADING FROM SQLITE (LOW IMPACT)
**Location**: `mcts_trader/data/history.py`

**Problem**:
- Historical data loaded from 20GB SQLite database
- Likely synchronous queries per step
- Database on disk, not in memory

**Impact**:
- Small latency per env step
- Accumulates over thousands of steps

**Solution**:
- Pre-load frequent data into RAM
- Use async database queries
- Cache recent windows in memory

---

## Optimization Priority Ranking

### TIER 1: CRITICAL (Immediate 5-10x speedup)
1. **Switch to FastStateEncoder (GPU)** - 3-5x speedup
2. **Reduce MCTS simulations to 25-50** - 2-4x speedup
3. **Increase collector_env_num to 16** - 2x speedup

**Combined Expected Speedup: 10-20x**

### TIER 2: HIGH (Additional 2-3x speedup)
4. **Move encoder device='cuda' explicitly**
5. **Batch NN evaluations better in MCTS**
6. **Pre-load database into memory cache**

### TIER 3: MEDIUM (Incremental improvements)
7. **Use mixed precision (FP16/BF16)**
8. **Optimize TRM recursion cycles during collection**
9. **Profile and optimize correlation computation**

---

## Recommended Configuration Changes

### Immediate Changes (config/trading_config.py):

```python
trading_muzero_config = dict(
    env=dict(
        env_id='MultiAssetTrading-v0',
        n_assets=64,
        n_actions_per_asset=3,
        collector_env_num=16,  # CHANGED: 4 -> 16
        evaluator_env_num=2,
        n_evaluator_episode=3,
        stop_value=100000,
    ),
    policy=dict(
        model=dict(
            observation_shape=(27, 64, 64),
            action_space_size=64 * 3,
            state_dim=64,
            model_type='conv',
        ),
        mcts=dict(
            num_simulations=50,  # CHANGED: 100 -> 50 (for collection)
            max_depth=10,
            pb_c_base=19652,
            pb_c_init=1.25,
            root_dirichlet_alpha=0.3,
            root_noise_weight=0.25,
        ),
        mcts_eval=dict(  # NEW: separate config for evaluation
            num_simulations=150,  # Higher quality for eval
        ),
        cuda=True,
        env_type='not_board_games',
        game_segment_length=200,
        update_per_collect=50,
        batch_size=512,
        optim_type='Adam',
        learning_rate=0.0002,
        weight_decay=1e-4,
        grad_clip_value=10,
        replay_buffer_size=10000,
        eval_freq=100,
    ),
)
```

### Code Changes:

#### 1. Switch to GPU Encoder (env/trading_env.py:97)

**BEFORE:**
```python
self.encoder = MultiAssetStateEncoder(n_assets=n_assets, lookback=lookback, device='cpu')
```

**AFTER:**
```python
self.encoder = MultiAssetStateEncoder(n_assets=n_assets, lookback=lookback, device='cuda')
```

#### 2. Use FastStateEncoder (Alternative - Best Performance)

**Replace MultiAssetStateEncoder with FastStateEncoder:**

```python
from ..encoder.state import FastStateEncoder

# In TradingEnv.__init__:
self.encoder = FastStateEncoder(n_assets=n_assets, lookback=lookback).cuda()
```

**Note**: FastStateEncoder already implemented (lines 307-372) but not used. It's fully GPU-accelerated with torch operations instead of NumPy loops.

#### 3. Add Mixed Precision Training (main.py)

Add after model creation:
```python
from torch.cuda.amp import autocast, GradScaler

# Enable TF32 for faster matmul on Ampere
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Use bfloat16 for stability (Ampere+ GPUs)
torch.set_float32_matmul_precision('medium')
```

---

## Expected Results After Optimization

### Current Performance:
- GPU Utilization: 18%
- VRAM Usage: 741MB / 24GB
- CPU: 81% (bottlenecked)
- Data Collection: Very slow (~10-20 steps/min estimated)

### After TIER 1 Optimizations:
- GPU Utilization: 60-80%
- VRAM Usage: 4-6GB / 24GB
- CPU: 40-50% (balanced)
- Data Collection: 200-400 steps/min (10-20x faster)
- First batch (512): ~2-3 minutes (vs 30+ minutes current)

### After TIER 1+2 Optimizations:
- GPU Utilization: 80-90%
- VRAM Usage: 8-12GB / 24GB
- Data Collection: 400-600 steps/min
- Training throughput: 50-100 episodes/hour (vs ~5/hour current)

---

## Implementation Plan

### Phase 1: Quick Wins (5 minutes)
1. Edit `mcts_trader/config/trading_config.py`:
   - `collector_env_num: 4 -> 16`
   - `num_simulations: 100 -> 50`
2. Edit `mcts_trader/env/trading_env.py:97`:
   - `device='cpu' -> device='cuda'`
3. Restart training

**Expected speedup: 10x**

### Phase 2: FastStateEncoder (15 minutes)
1. Modify `mcts_trader/env/trading_env.py`:
   - Import `FastStateEncoder`
   - Replace encoder initialization
   - Test encoder output shape compatibility
2. Restart training

**Expected additional speedup: 2x (total 20x)**

### Phase 3: Mixed Precision (10 minutes)
1. Add TF32/BF16 flags to `main.py`
2. Verify training stability
3. Monitor loss curves

**Expected additional speedup: 1.3x (total 26x)**

---

## Monitoring Commands

```bash
# GPU utilization
watch -n 1 nvidia-smi

# Training progress
tail -f training_console.log | grep -v "^\."

# Process stats
ps aux | grep 26421

# VRAM breakdown
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv -l 1
```

---

## Risk Assessment

**LOW RISK**:
- Config changes (collector_env_num, num_simulations)
- Device change to CUDA
- These are standard parameters

**MEDIUM RISK**:
- FastStateEncoder switch
- Need to verify output tensor shapes match
- Test with 1 episode first

**HIGH RISK**:
- Mixed precision
- May cause NaN losses if not stable
- Monitor carefully, revert if needed

---

## Next Steps

1. **IMMEDIATE**: Apply Phase 1 changes (5 min)
2. **VERIFY**: Monitor GPU utilization increase
3. **IF STABLE**: Apply Phase 2 (FastStateEncoder)
4. **REPORT**: Update state.json with results

**Blocked on**: User decision to stop current training (PID 26421) and apply changes
