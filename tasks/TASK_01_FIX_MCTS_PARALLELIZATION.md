# TASK 01: Fix MCTS Parallelization

⚠️ **STATUS UPDATE (2025-12-17)**: Worker pool code **already exists** at `mcts_trader/jax_agent/mcts_worker_pool.py` (173 lines, modified Dec 16 23:58). Needs verification if it's actually active during training.

**New Objective**: Verify implementation and measure performance improvement.

---

## Original Objective
Fix MCTS to use all CPU cores instead of just one. Currently experiencing CPU bottleneck with single-threaded tree traversal.

## Current State Analysis

**Problem Location**: `mcts_trader/jax_agent/mctx_search.py`

**Issue**: MCTS tree operations are sequential, only using 1 CPU core while GPU sits idle.

**Current Performance**:
- Batch size: 512
- MCTS sims: 32
- CPU utilization: 1 core at 90%, 15 cores idle
- GPU utilization: 60% (starved by CPU)

## Root Cause

The mctx library runs batched MCTS, but JAX/XLA compilation doesn't parallelize tree traversal across CPU cores by default.

**Code Location**: `mcts_trader/jax_agent/mctx_search.py:120-220`

```python
def _search(self, params, rng, observation, action_mask):
    # This runs on CPU sequentially
    # Even though batch=512, JAX doesn't auto-parallelize

    policy_output = mctx.gumbel_muzero_policy(
        params=params,
        rng_key=rng,
        root=root,
        recurrent_fn=recurrent_fn,
        num_simulations=self.num_simulations,  # 32
        # ... tree ops happen HERE (CPU, sequential)
    )
```

## Solution: Multi-Process MCTS

### Strategy
Split batch across CPU workers, each runs MCTS in parallel processes.

**Architecture**:
```
Main Process (GPU)
    ↓ distribute batch
Worker 1 (CPU cores 0-3):   MCTS on samples 0-127
Worker 2 (CPU cores 4-7):   MCTS on samples 128-255
Worker 3 (CPU cores 8-11):  MCTS on samples 256-383
Worker 4 (CPU cores 12-15): MCTS on samples 384-511
    ↓ gather results
Main Process (GPU): Train on full batch
```

## Implementation Steps

### Step 1: Create MCTS Worker Pool

**File**: `mcts_trader/jax_agent/mcts_worker_pool.py` (NEW)

```python
import multiprocessing as mp
import jax
import jax.numpy as jnp
from typing import List, Tuple
import numpy as np

class MCTSWorker:
    """Single worker process that runs MCTS searches"""

    def __init__(self, worker_id: int, search_fn, model_apply_fn):
        self.worker_id = worker_id
        self.search_fn = search_fn
        self.model_apply_fn = model_apply_fn

    def run_batch(self, params, rng, observations, action_masks):
        """
        Run MCTS on a sub-batch.

        Args:
            params: Model parameters
            rng: Random key
            observations: (batch_subset, 27, 64, 64)
            action_masks: (batch_subset, 384)

        Returns:
            actions, policies, stats
        """
        # Call search function (already JIT compiled)
        actions, policies, stats = self.search_fn(
            params, rng, observations, action_masks
        )

        return actions, policies, stats


class MCTSWorkerPool:
    """Pool of MCTS workers for parallel search"""

    def __init__(self, search_fn, model_apply_fn, n_workers: int = 4):
        """
        Args:
            search_fn: JIT-compiled search function
            model_apply_fn: Model apply function
            n_workers: Number of parallel workers (default: 4 for 16 cores)
        """
        self.n_workers = n_workers
        self.search_fn = search_fn
        self.model_apply_fn = model_apply_fn

        # Create worker pool
        mp.set_start_method('spawn', force=True)
        self.pool = mp.Pool(processes=n_workers)

        print(f"✅ MCTS Worker Pool initialized with {n_workers} workers")

    def parallel_search(
        self,
        params,
        rng,
        observations,
        action_masks
    ) -> Tuple[jnp.ndarray, jnp.ndarray, dict]:
        """
        Run MCTS in parallel across workers.

        Args:
            observations: (512, 27, 64, 64)
            action_masks: (512, 384)

        Returns:
            actions: (512,)
            policies: (512, 384)
            stats: dict
        """
        batch_size = observations.shape[0]
        chunk_size = batch_size // self.n_workers

        # Split batch into chunks
        obs_chunks = []
        mask_chunks = []
        rng_chunks = []

        for i in range(self.n_workers):
            start = i * chunk_size
            end = start + chunk_size if i < self.n_workers - 1 else batch_size

            obs_chunks.append(observations[start:end])
            mask_chunks.append(action_masks[start:end])

            # Split RNG key for each worker
            rng, worker_rng = jax.random.split(rng)
            rng_chunks.append(worker_rng)

        # Distribute to workers
        tasks = [
            (params, rng_chunks[i], obs_chunks[i], mask_chunks[i])
            for i in range(self.n_workers)
        ]

        # Run in parallel
        results = self.pool.starmap(self._worker_fn, tasks)

        # Gather results
        all_actions = []
        all_policies = []

        for actions, policies, _ in results:
            all_actions.append(actions)
            all_policies.append(policies)

        # Concatenate
        actions = jnp.concatenate(all_actions, axis=0)
        policies = jnp.concatenate(all_policies, axis=0)

        return actions, policies, {}

    def _worker_fn(self, params, rng, obs, masks):
        """Worker function (runs in subprocess)"""
        return self.search_fn(params, rng, obs, masks)

    def close(self):
        """Shutdown worker pool"""
        self.pool.close()
        self.pool.join()
```

### Step 2: Integrate into Training Loop

**File**: `train_jax_fast.py`

**Modify**:

```python
# Line ~550 (after creating search function)

# OLD:
search = create_mctx_search(
    model,
    num_simulations=args.sims,
    use_gumbel=True,
    dirichlet_alpha=args.noise,
    exploration_fraction=args.noise_frac
)

# NEW:
from mcts_trader.jax_agent.mcts_worker_pool import MCTSWorkerPool

base_search = create_mctx_search(
    model,
    num_simulations=args.sims,
    use_gumbel=True,
    dirichlet_alpha=args.noise,
    exploration_fraction=args.noise_frac
)

# Wrap in worker pool for parallelization
search = MCTSWorkerPool(
    search_fn=base_search,
    model_apply_fn=model.apply,
    n_workers=4  # 4 workers × 4 cores = 16 cores utilized
)

print(f"✅ Parallel MCTS enabled (4 workers, batch split: 512 → 4×128)")
```

### Step 3: Update run_self_play to use pool

**File**: `train_jax_fast.py:340-411`

**Modify**:

```python
def run_self_play(env, search_fn, model_apply_fn, params, rng_key, steps=200, confidence_threshold=0.0):
    """Self-play with parallel MCTS"""
    # ... existing code ...

    for step in range(steps):
        rng, step_rng = jax.random.split(rng)

        obs_jax = jnp.array(obs_array, dtype=jnp.float32)
        mask_jax = jnp.array(mask_array, dtype=jnp.int8)

        # Call search (now parallelized if using MCTSWorkerPool)
        action_jax, policy_jax, stats = search_fn(params, step_rng, obs_jax, mask_jax)

        # ... rest unchanged ...
```

**Note**: No code change needed here - the interface stays the same. Worker pool handles parallelization internally.

## Testing

### Test 1: Verify Parallelization

**Command**:
```bash
# Terminal 1: Start training
python train_jax_fast.py --synthetic-level 0 --epochs 10 --batch-size 512 --sims 32

# Terminal 2: Monitor CPU usage
htop
# Expected: All 16 cores at 70-90% (not just 1 core)
```

### Test 2: Measure Speedup

**Benchmark script**: `tests/benchmark_mcts_parallelization.py` (NEW)

```python
import time
import jax
import jax.numpy as jnp
from train_jax_fast import FastTradingEnv, create_mctx_search, MuZeroNetworks
from mcts_trader.jax_agent.mcts_worker_pool import MCTSWorkerPool

def benchmark():
    # Setup
    env = FastTradingEnv(n_assets=64)
    model = MuZeroNetworks(state_dim=64, action_space_size=192)
    rng = jax.random.PRNGKey(42)

    # Create dummy batch
    batch_size = 512
    obs = jnp.ones((batch_size, 27, 64, 64))
    masks = jnp.ones((batch_size, 192))

    # Test 1: Serial (current)
    search_serial = create_mctx_search(model, num_simulations=32)

    start = time.time()
    _ = search_serial(params, rng, obs, masks)
    serial_time = time.time() - start

    print(f"Serial MCTS (1 core):  {serial_time:.2f}s")

    # Test 2: Parallel (new)
    search_parallel = MCTSWorkerPool(search_serial, model.apply, n_workers=4)

    start = time.time()
    _ = search_parallel.parallel_search(params, rng, obs, masks)
    parallel_time = time.time() - start

    print(f"Parallel MCTS (4 workers): {parallel_time:.2f}s")
    print(f"Speedup: {serial_time / parallel_time:.2f}x")

    search_parallel.close()

if __name__ == "__main__":
    benchmark()
```

**Run**:
```bash
python tests/benchmark_mcts_parallelization.py
```

**Expected Output**:
```
Serial MCTS (1 core):  8.50s
Parallel MCTS (4 workers): 2.30s
Speedup: 3.7x
```

**Pass Criteria**: Speedup > 3.0x

### Test 3: Validate Correctness

**Script**: `tests/test_parallel_mcts_correctness.py` (NEW)

```python
import jax
import jax.numpy as jnp
import numpy as np

def test_same_results():
    """Verify parallel and serial MCTS give same results (with same RNG seed)"""
    # Setup
    rng = jax.random.PRNGKey(42)
    obs = jnp.ones((512, 27, 64, 64))
    masks = jnp.ones((512, 192))

    # Serial
    actions_serial, policies_serial, _ = search_serial(params, rng, obs, masks)

    # Parallel (same RNG seed)
    actions_parallel, policies_parallel, _ = search_parallel.parallel_search(params, rng, obs, masks)

    # Compare
    assert jnp.allclose(actions_serial, actions_parallel), "Actions differ!"
    assert jnp.allclose(policies_serial, policies_parallel, atol=1e-4), "Policies differ!"

    print("✅ Parallel MCTS produces identical results")

if __name__ == "__main__":
    test_same_results()
```

## Performance Expectations

**Before (Serial)**:
- CPU: 1 core at 90%, 15 cores idle
- GPU: 60% util (starved)
- Time per epoch: ~16s

**After (Parallel)**:
- CPU: 16 cores at 70-80%
- GPU: 85-90% util (fed properly)
- Time per epoch: ~5s (3x faster)

## Acceptance Criteria

✅ **PASS if ALL conditions met**:
1. `htop` shows all 16 CPU cores utilized (>50% each)
2. Speedup > 3.0x on benchmark
3. Training still converges (sanity check on Level 0)
4. GPU utilization increases from 60% → 85%+

## Rollback Plan

If parallel MCTS causes issues:

```python
# train_jax_fast.py
# Simply don't wrap in MCTSWorkerPool
USE_PARALLEL_MCTS = False  # Set to False to disable

if USE_PARALLEL_MCTS:
    search = MCTSWorkerPool(base_search, model.apply, n_workers=4)
else:
    search = base_search  # Original serial version
```

## Known Issues / Limitations

**Issue 1: Multiprocessing Overhead**

For very small batches (<128), parallelization overhead might hurt performance.

**Solution**: Only use parallel when batch ≥ 256:
```python
if batch_size >= 256:
    search = MCTSWorkerPool(...)
else:
    search = base_search  # Serial for small batches
```

**Issue 2: JAX GPU Memory Sharing**

Each worker process might try to allocate GPU memory.

**Solution**: Force workers to use CPU only:
```python
# In MCTSWorker.__init__
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Hide GPU from workers
```

## Integration Checklist

- [ ] Create `mcts_trader/jax_agent/mcts_worker_pool.py`
- [ ] Modify `train_jax_fast.py` to use worker pool
- [ ] Create benchmark script
- [ ] Create correctness test
- [ ] Run tests, verify 3x speedup
- [ ] Commit with message: "Fix MCTS parallelization - 3x speedup"

## Next Task After Completion

Once this is working → **TASK_02_SYNTHETIC_MARKET_GENERATORS.md**
