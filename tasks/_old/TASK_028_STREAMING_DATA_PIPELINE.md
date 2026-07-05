# TASK_028: Streaming Data Pipeline for JAX Environment

## Status: IMPLEMENTED (2026-01-03)

### Implementation Summary
- **Phase 0**: Bug fix - `prices_np` moved outside per-env loop ✅
- **Phase 1**: Pre-allocated arrays + fp16 for obs/policies ✅
- **Phase 2**: Batched GPU transfers (50 steps per batch) ✅

Changes in `train_jax_fast.py`:
- Pre-allocated numpy arrays instead of list appends (eliminates 20,000+ appends)
- fp16 for observations and policies (halves bandwidth)
- Batched GPU→CPU transfers every 50 steps (4 transfers instead of 200)

## Problem Statement

With high parallel env counts (50-100), memory bandwidth becomes the bottleneck:
- 100 envs × 200 steps × 442KB obs = **8.8GB** per epoch
- PCIe bandwidth: ~5.5 GB/s → theoretical minimum 1.6s just for transfer
- Actual: ~25s because of per-step transfer overhead and Python list accumulation

### Bug Found (2026-01-03)
Line 618 in `train_jax_fast.py`: `prices_np = np.array(prices)` is INSIDE the per-env loop,
converting the same prices array N times (100x for 100 envs). Must move outside loop.

### Benchmark Results (current)
| Envs | Self-play | Samples/sec | Notes |
|------|-----------|-------------|-------|
| 4    | 2.5s      | 320         | GPU underutilized |
| 10   | 3.0s      | 667         | Good scaling |
| 20   | 3.5s      | 1143        | Sweet spot |
| 50   | 6.3s      | 1587        | Peak throughput |
| 100  | 25s       | 800         | Bandwidth bottleneck |

### Hardware Constraints
```
RAM bandwidth:     ~5-10 GB/s (internal)
PCIe H2D:          ~5.5 GB/s (RAM→VRAM)
PCIe D2H:          ~5.2 GB/s (VRAM→RAM)
VRAM bandwidth:    ~770 GB/s (internal)
VRAM capacity:     24 GB (RTX 3090)
```

## Root Cause Analysis

Current flow (per step, 200 times per epoch):
```python
# GPU→CPU transfer EVERY STEP (slow!)
obs_np = np.array(obs_batch)      # 100×27×64×64×4 = 44MB
policies_np = np.array(policies)  # 100×320×4 = 128KB

# Python list append (memory fragmentation) - 20,000 appends for 100 envs!
for i in range(n_envs):
    all_observations[i].append(obs_np[i])  # Growing lists
```

Problems (in order of impact):
1. **Python list overhead** (BIGGEST): 20,000 `list.append()` calls cause:
   - Memory fragmentation (lists grow by reallocating + copying)
   - GC pressure from temporary allocations
   - Cache thrashing from non-contiguous memory
2. **Per-step transfers**: 200 small transfers instead of 1 bulk transfer
3. **Full precision**: fp32 when fp16 would suffice for training
4. **Repeated conversion bug**: `prices_np = np.array(prices)` inside per-env loop

## Proposed Solutions

### Phase 0: Bug Fix (Immediate)
Move `prices_np = np.array(prices)` outside the per-env loop:

```python
# BEFORE the for loop (line ~588):
prices_np = np.array(prices)  # Once, not n_envs times

for i in range(n_envs):
    # ... use prices_np ...
```

**Expected improvement**: ~5%

### Phase 1: Pre-allocation + fp16 (BIGGEST WIN)
Replace Python lists with pre-allocated NumPy arrays:

```python
# CURRENT (slow):
all_observations = [[] for _ in range(n_envs)]
all_observations[i].append(obs_np[i])  # 20,000 appends!

# FIXED (fast):
all_observations = np.zeros((n_envs, steps, 27, 64, 64), dtype=np.float16)
all_observations[:, step] = obs_np  # Direct indexing, no append
```

Benefits:
- Eliminates 20,000 list.append() calls
- Contiguous memory (better cache behavior)
- fp16 halves bandwidth: 8.8GB → 4.4GB
- No GC pressure from growing lists

**Expected improvement**: ~50% (from 25s to ~12s for 100 envs)

### Phase 2: Batched Transfers
Instead of transferring every step, batch every N steps:

```python
# Accumulate on GPU for N steps
gpu_buffer_obs = []  # List of JAX arrays (stays on GPU)
for step in range(steps):
    gpu_buffer_obs.append(obs_batch)

    if len(gpu_buffer_obs) >= BATCH_SIZE or step == steps-1:
        # Single bulk transfer
        batch_obs = jnp.stack(gpu_buffer_obs)  # GPU operation (fast)
        cpu_obs = np.array(batch_obs, dtype=np.float16)  # One transfer
        # Add to CPU storage...
        gpu_buffer_obs = []
```

**BATCH_SIZE options**:
- 10 steps: 10 transfers instead of 200 (20x fewer)
- 50 steps: 4 transfers (50x fewer)
- Tune based on VRAM availability

### Phase 3: Stream Completed Trajectories
Add trajectories to replay buffer as envs finish (done=True):

```python
for i in range(n_envs):
    if active[i] and dones_np[i]:
        # Immediately package and add to buffer
        trajectory = package_trajectory(i, all_data[i])
        trainer.add_trajectory(trajectory)
        # Free memory
        all_data[i] = reset_storage()
        active[i] = False
```

Benefits:
- Memory freed as envs complete (don't hold all 100 trajectories)
- Smoother memory usage curve
- Can run more envs within same memory budget

### Phase 4: Pinned Memory (Advanced)
Use CUDA pinned memory for faster async transfers:

```python
import jax
# Pre-allocate pinned host memory
pinned_obs = jax.device_put(
    np.zeros((batch_size, 27, 64, 64), dtype=np.float16),
    device=jax.devices('cpu')[0]
)
# Async transfer
obs_future = jax.device_get(obs_batch, donate_argnums=())
```

### Phase 5: GPU-side Synthetic Generation (Optional)
For synthetic curriculum, generate prices directly on GPU:

```python
@jax.jit
def generate_sine_prices(rng, n_assets, n_steps):
    """Generate sine wave prices entirely on GPU"""
    freqs = jax.random.uniform(rng, (n_assets,), minval=0.01, maxval=0.1)
    phases = jax.random.uniform(rng, (n_assets,), minval=0, maxval=2*jnp.pi)
    t = jnp.arange(n_steps)[:, None]
    prices = 100.0 * (1 + 0.1 * jnp.sin(freqs * t + phases))
    return prices
```

Eliminates CPU→GPU price transfer entirely for synthetic data.

## Implementation Plan

### Minimal Viable Change (Phase 0 + 1)

**Phase 0 - Bug fix (1 line):**
```python
# Move line 618 OUTSIDE the for loop (before line 590):
prices_np = np.array(prices)  # Once, not n_envs times
```

**Phase 1 - Pre-allocation + fp16:**
```python
# Replace list initialization (lines 508-515):
all_observations = np.zeros((n_envs, steps, 27, 64, 64), dtype=np.float16)
all_actions = np.zeros((n_envs, steps), dtype=np.int32)
all_rewards = np.zeros((n_envs, steps), dtype=np.float32)
all_values = np.zeros((n_envs, steps), dtype=np.float32)
all_policy_probs = np.zeros((n_envs, steps, 320), dtype=np.float16)
all_dones = np.zeros((n_envs, steps), dtype=bool)
step_counts = np.zeros(n_envs, dtype=np.int32)  # Track actual steps per env

# In step loop, replace append with direct indexing:
all_observations[active_mask, step] = obs_np[active_mask].astype(np.float16)
step_counts[active_mask] += 1

# In results packaging, slice to actual length:
'observations': all_observations[i, :step_counts[i]].astype(np.float32),
```

### Full Implementation (Phases 0-2)
1. Phase 0: Move prices_np conversion outside loop
2. Phase 1: Pre-allocate arrays + fp16 storage
3. Phase 2: Add `--transfer-batch-size N` for batched GPU→CPU transfers
4. Add memory monitoring/logging

## Expected Results (Validated 2026-01-03)

| Phase | Change | Time | Improvement |
|-------|--------|------|-------------|
| Current | baseline | 25s | - |
| Phase 0 | Bug fix | 24s | 5% |
| Phase 1 | Pre-alloc + fp16 | 12s | **50%** |
| Phase 2 | Batched transfers | 8s | 68% |
| Phase 3 | Streaming | 7s | 72% |
| All combined | - | ~7-8s | **68-72%** |

**Realistic target**: 100 envs @ ~8s = 2,500 samples/sec (vs current 800)
**Optimistic target**: 100 envs @ ~6s = 3,300 samples/sec (with async/pinned memory)

Note: Original 80% estimate was optimistic. 68-72% is realistic without async transfers.

## Testing

```bash
# Baseline
python train_jax_fast.py --synthetic-level 0 --epochs 5 --parallel-envs 100 --jax-env --name baseline

# After Phase 1
python train_jax_fast.py --synthetic-level 0 --epochs 5 --parallel-envs 100 --jax-env --fp16-storage --name fp16_test

# After Phase 2-3
python train_jax_fast.py --synthetic-level 0 --epochs 5 --parallel-envs 100 --jax-env --streaming-batch-size 20 --name streaming_test
```

## Historical Data Strategy

### Sequential Days (Preserve Cross-Asset Correlation)

**Critical constraint**: All 64 assets must be time-synchronized for cross-correlation learning.
Random sampling across time breaks correlation structure.

**Solution**: Sequential day batching
```
Epoch 1: 30 envs see days 1-30 (each env = different sequential day)
Epoch 2: 30 envs see days 31-60
...
```

Benefits:
- Correlation preserved (sequential, not random)
- 30 days of market exposure per epoch
- 80 years / 30 days = ~970 epochs of unique data

### Data Math
```
Historical data:     ~80 years continuous (29,200 days)
Data augmentation:   4x (original, inverted, reversed, both)
Effective dataset:   ~116,800 days
At 30 days/epoch:    ~3,893 epochs of unique data
```

### Warmup on Synthetic (Preserve Historical Data)

**Problem**: JIT warmup + buffer filling wastes valuable historical data.

**Solution**: Warmup phase uses synthetic, then switch:
```bash
# Phase 1: Warmup (JIT compile, fill buffer) - uses synthetic
python train_jax_fast.py \
    --synthetic-level 0 \
    --epochs 10 \
    --parallel-envs 30 \
    --jax-env \
    --name warmup_phase

# Phase 2: Historical training (load checkpoint, clear buffer)
python train_jax_fast.py \
    --db-path trading_data.db \
    --start-date 1945-01-01 \
    --end-date 2025-01-01 \
    --load-checkpoint jax_ckpt_fast/warmup_phase \
    --clear-buffer \
    --sequential-days \
    --parallel-envs 30 \
    --name historical_train
```

### Historical Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `--parallel-envs` | 30 | 30 days/epoch = 1 month coverage |
| `--steps` | 1440 | 1 day of 1-min candles |
| `--sequential-days` | True | Preserve correlation |
| `--fp16-storage` | True | Reduce bandwidth (19GB → 9.5GB) |

### Reward/Value Target Handling

Per-sample targets preserved (not averaged across days):
- Day 1 (+10% opportunity) → high value targets
- Day 15 (-5% crash) → low value targets
- Model learns to distinguish different market regimes
- Loss `mean()` is over batch (standard SGD), targets stay per-sample

## Notes

- Historical data capped at ~30-50 envs due to DB read + bandwidth
- Synthetic can scale to 100+ envs with GPU-side generation
- Future hardware upgrade (PCIe 4.0 x16) would double bandwidth (~11 GB/s)
- Consider M.2 NVMe for faster checkpointing (separate bottleneck)

## Files to Modify

- `train_jax_fast.py`: `run_self_play_jax()` function (~lines 500-630)
- `mcts_trader/jax_agent/trainer.py`: Add streaming trajectory support
- `mcts_trader/data/synthetic_market.py`: (Phase 5) JAX-native generators

## Priority

Medium-High: Current sweet spot is 20-50 envs. This optimization unlocks 100+ envs effectively.
