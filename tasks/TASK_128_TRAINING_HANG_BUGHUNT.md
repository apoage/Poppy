# TASK_128: Critical Training Hang Bug

**Priority**: CRITICAL
**Status**: Active Investigation
**Impact**: 90% chance of hang on any training run >60 epochs - blocks all long-term training

## Problem Statement

Training hangs indefinitely at random epochs (typically 60-90+). The process doesn't crash - it just stops making progress. GPU utilization drops to 0% while memory stays allocated (17GB+). This makes production training impossible.

## Symptoms

```
Training:  81%|████████▏ | 65/80 [19:11<04:29, 17.97s/ep]
[... no further output, process alive but frozen ...]

GPU State when hung:
- GPU Util: 0%
- GPU Memory: 17GB (high, not released)
- CPU: 0%
- Process: Running (not crashed)
```

## Reproduction

**Conditions that trigger hang:**
- Epochs > 60 (sometimes 90+)
- More likely with `--use-prediction-loss`
- May happen with basic config too
- Synthetic data (level 0)
- JAX env enabled

**Minimal reproduction command:**
```bash
python train_jax_fast.py \
  --name hang_repro \
  --synthetic-level 0 \
  --epochs 100 \
  --steps 200 \
  --parallel-envs 4 \
  --jax-env \
  --batch-size 512 \
  --use-prediction-loss \
  --imprint
```

**Time to reproduce:** 30-60 minutes (hang occurs around epoch 60-80)

## Known Facts

1. **Last successful output before hang:**
   - Training metrics printed
   - Checkpoint saved successfully
   - Progress bar updated
   - Then silence

2. **Memory pattern:**
   - Starts at ~8GB VRAM
   - Grows to 17GB+ before hang
   - Memory NOT released after hang

3. **Configurations tested:**
   - Hangs WITH prediction loss (more frequent)
   - May hang WITHOUT prediction loss (less frequent)
   - Hangs with imprinting enabled
   - Basic config (small model, 512 batch) still hangs

4. **What does NOT cause hang:**
   - First 50 epochs always work
   - Self-play alone works
   - Training step alone works (early epochs)

## Suspect Areas

### 1. JAX/XLA Compilation Cache Overflow
**Files:** All JIT-compiled functions
**Theory:** After many epochs, XLA cache grows and causes compilation deadlock
**Check:** Monitor `JAX_COMPILATION_CACHE_DIR` size, add `jax.clear_caches()` periodically

### 2. Memory Fragmentation / OOM
**Files:** `trainer.py`, `replay_buffer.py`, `expert_buffer.py`
**Theory:** Gradual memory leak or fragmentation causes silent OOM
**Check:** Add memory logging, profile with `jax.profiler`

### 3. jax.lax.scan Hangs
**Files:**
- `mcts_trader/jax_agent/prediction/market_ode.py:443` - multi-step prediction
- `mcts_trader/training/self_play.py` - episode generation
- `mcts_trader/jax_agent/mctx_search.py` - MCTS search

**Theory:** Scan accumulates state incorrectly, numerical instability causes infinite loop
**Check:** Add `jax.debug.print` inside scan body, check for NaN/Inf

### 4. Device-to-Host Transfer Deadlock
**Files:** `self_play.py` - `np.array(jax_array)` calls
**Theory:** GPU kernel completes but host transfer hangs
**Check:** Add timeout wrapper around transfers, use `jax.block_until_ready()` explicitly

### 5. Expert Buffer Threading Lock
**Files:** `expert_buffer.py` - `threading.Lock()`
**Theory:** Lock acquired but never released due to exception
**Check:** Add lock timeout, log lock acquisition/release

### 6. Checkpoint Interaction
**Files:** `train_jax_fast.py` - orbax checkpoint saving
**Theory:** Checkpoint save interferes with next epoch's compilation
**Check:** Disable checkpointing and see if hang still occurs

### 7. PopArt Update Outside JIT
**Files:** `trainer.py:1616-1642`
**Theory:** PopArt weight correction (every 10 steps) causes state desync
**Check:** Disable PopArt and see if hang still occurs

### 8. Prediction Loss Specific
**Files:**
- `trainer.py:1316-1346` - prediction loss computation
- `prediction/market_ode.py` - NeuralMarketODE
- `prediction/student_t.py` - hybrid_prediction_loss

**Theory:** Prediction loss path has numerical instability or infinite loop
**Check:** Add timing around `_compute_prediction_loss`, check intermediate values

## Diagnostic Approach

### Phase 1: Isolate Component (30 min)
Run tests with different features disabled:
```bash
# Test A: No prediction loss
python train_jax_fast.py --epochs 100 --steps 200 --jax-env --imprint

# Test B: No imprinting
python train_jax_fast.py --epochs 100 --steps 200 --jax-env --use-prediction-loss

# Test C: No PopArt
python train_jax_fast.py --epochs 100 --steps 200 --jax-env --no-popart

# Test D: No checkpointing (modify script)
# Test E: Smaller batch (256 instead of 512)
```

### Phase 2: Add Timing Instrumentation
Add to `train_jax_fast.py` main loop:
```python
import time

for epoch in range(epochs):
    t0 = time.time()
    print(f"[TIMING] Epoch {epoch} START")

    t1 = time.time()
    trajectories = run_self_play_jax(...)
    print(f"[TIMING] Self-play: {time.time()-t1:.2f}s")

    t2 = time.time()
    # Add to replay buffer
    print(f"[TIMING] Buffer add: {time.time()-t2:.2f}s")

    t3 = time.time()
    metrics = trainer.train_step(...)
    print(f"[TIMING] Train step: {time.time()-t3:.2f}s")

    t4 = time.time()
    # Checkpoint
    print(f"[TIMING] Checkpoint: {time.time()-t4:.2f}s")

    print(f"[TIMING] Epoch {epoch} TOTAL: {time.time()-t0:.2f}s")
    sys.stdout.flush()  # Force output
```

### Phase 3: Memory Profiling
```python
import jax

# Add at start of each epoch
print(f"[MEM] JAX memory: {jax.devices()[0].memory_stats()}")

# Or use nvidia-smi logging
# nvidia-smi --query-gpu=memory.used --format=csv -l 5 > gpu_mem.log &
```

### Phase 4: Numerical Checks
Add inside training step:
```python
# Check for NaN/Inf in gradients
grad_has_nan = jax.tree_util.tree_reduce(
    lambda a, b: a | b,
    jax.tree_util.tree_map(lambda x: jnp.any(jnp.isnan(x) | jnp.isinf(x)), grads)
)
if grad_has_nan:
    print(f"[WARN] NaN/Inf in gradients at step {step}")
```

## Files to Investigate

```
Primary (most likely):
├── train_jax_fast.py           # Main loop, checkpoint logic
├── mcts_trader/jax_agent/trainer.py    # Training step, prediction loss
├── mcts_trader/training/self_play.py   # Episode generation, device transfer
└── mcts_trader/jax_agent/expert_buffer.py  # Threading lock

Secondary:
├── mcts_trader/jax_agent/prediction/market_ode.py  # jax.lax.scan
├── mcts_trader/jax_agent/mctx_search.py  # MCTS with scan
├── mcts_trader/jax_agent/popart.py       # Weight correction
└── mcts_trader/jax_agent/replay_buffer.py  # Memory management
```

## Success Criteria

- [ ] Identify exact function/line where hang occurs
- [ ] Understand root cause (memory? deadlock? numerical?)
- [ ] Implement fix
- [ ] Verify training runs 200+ epochs without hang
- [ ] Document prevention measures

## Previous Analysis

From `TASK_127r_hang_at_epoch_90.md`:
- Suspected JAX/XLA synchronization deadlock
- Happens during device-to-host transfer
- Not a Python logic bug per se
- May be triggered by specific data patterns

## Questions to Answer

1. Does it hang at same epoch count or same wall-clock time?
2. Does it hang with `--parallel-envs 1`?
3. Does it hang on CPU-only (`JAX_PLATFORMS=cpu`)?
4. Does memory grow linearly or spike before hang?
5. Is there a specific batch/trajectory that triggers it?
