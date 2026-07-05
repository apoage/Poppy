# Parallelization Strategies for Training Speedup

**Goal**: Maximize training throughput on RTX 3090 (24GB VRAM, 16 CPU cores)

**Current**: Batch 512, 1.21× speedup with 4 workers

---

## Strategy 1: Optimize MCTS Worker Pool (Quick Win)

### Test Larger Batches
**Current**: Batch 256 → 1.21× speedup
**Hypothesis**: IPC overhead dominates at small batch sizes

**Test**:
```bash
python tools/benchmark_parallelization.py --test batch --workers 4
```

**Expected**:
- Batch 512: 1.5-2.0× speedup
- Batch 1024: 2.0-3.0× speedup
- Batch 2048: 3.0-4.0× speedup

**If speedup scales**: Increase training batch size, problem solved
**If speedup plateaus**: IPC not the issue, need different strategy

---

### Test More Workers
**Current**: 4 workers
**Hardware**: 16 CPU cores available

**Test**:
```bash
python tools/benchmark_parallelization.py --test workers --batch-size 512
```

**Expected**:
- 8 workers: Better CPU utilization
- 16 workers: Maximum CPU utilization
- Diminishing returns after ~8-12 workers (Amdahl's law)

**Risk**: Too many workers → context switching overhead

---

## Strategy 2: Multi-Instance Training (Parallel Epochs)

### Concept
Run 2-4 independent training processes in parallel on same GPU.

**Advantages**:
- No code changes needed (just launch multiple processes)
- Each process uses different data/checkpoints
- VRAM permits it (24GB total, ~1.25GB per instance)
- CPU cores underutilized (16 cores, only 1-4 used per instance)

**Disadvantages**:
- No single model (need to merge/select best)
- VRAM fragmentation risk
- Need careful process management

### VRAM Analysis

**Per Training Instance** (~1.25 GB):
- Model (7M params): ~50 MB
- Activations (batch 512): ~200 MB
- Replay Buffer: ~1000 MB
- Optimizer State: ~50 MB

**Available**: 24 GB total
**Used by OS/JAX**: ~2 GB
**Available for training**: ~22 GB

**Parallel Instances Possible**: ~17 instances (theoretical)
**Practical Limit**: 4-6 instances (leave headroom for peaks)

### Implementation

**Option A**: Manual Launch (Simple)
```bash
# Terminal 1
CUDA_VISIBLE_DEVICES=0 python train_jax_fast.py --config cfg1.json --epochs 1000 &

# Terminal 2
CUDA_VISIBLE_DEVICES=0 python train_jax_fast.py --config cfg2.json --epochs 1000 &

# Terminal 3
CUDA_VISIBLE_DEVICES=0 python train_jax_fast.py --config cfg3.json --epochs 1000 &

# Terminal 4
CUDA_VISIBLE_DEVICES=0 python train_jax_fast.py --config cfg4.json --epochs 1000 &
```

**Each uses**:
- Different checkpoint dir (jax_ckpt_fast/run1, run2, run3, run4)
- Different random seed
- Same assets (or different for diversity)
- ~1.25 GB VRAM
- 4 CPU cores (if worker pool active)

**Total**: 4 instances × 1.25GB = 5GB VRAM (leaves 17GB free!)

---

**Option B**: Orchestrated (Better)

Create `tools/parallel_training.py`:
```python
#!/usr/bin/env python3
"""
Launch multiple parallel training instances on same GPU.
"""

import subprocess
import time
import os

def launch_parallel_training(n_instances: int = 4, epochs_per: int = 1000):
    """Launch N parallel training instances"""

    processes = []

    for i in range(n_instances):
        # Create config for this instance
        config = {
            'n_assets': 64,
            'asset_symbols': None,
            'min_coverage': 0.75,
            'checkpoint_dir': f'jax_ckpt_fast/parallel_run{i+1}',
            'random_seed': 42 + i,  # Different seed per instance
        }

        config_file = f'config/parallel_instance_{i+1}.json'
        with open(config_file, 'w') as f:
            json.dump(config, f)

        # Launch process
        cmd = [
            'python', 'train_jax_fast.py',
            '--config', config_file,
            '--epochs', str(epochs_per),
        ]

        print(f"Launching instance {i+1}/{n_instances}...")
        proc = subprocess.Popen(cmd, env={**os.environ, 'CUDA_VISIBLE_DEVICES': '0'})
        processes.append(proc)

        time.sleep(5)  # Stagger launches to avoid VRAM spike

    print(f"\n{n_instances} training instances launched!")
    print("Monitor with: nvidia-smi -l 1")

    # Wait for all to complete
    for i, proc in enumerate(processes):
        proc.wait()
        print(f"Instance {i+1} completed with code {proc.returncode}")
```

**Usage**:
```bash
python tools/parallel_training.py --instances 4 --epochs 1000
```

**Expected**:
- 4× effective training throughput (4 runs simultaneously)
- ~5GB VRAM used (well within limits)
- All 16 CPU cores utilized (4 cores per instance)
- GPU shared across instances (JAX handles this)

**Post-Training**: Select best checkpoint from 4 runs or ensemble

---

## Strategy 3: Multi-Step Rollouts (Advanced)

### Concept
Instead of: `step → MCTS → update → step → MCTS → update`
Do: `step → step → step → MCTS batch → update`

**Advantages**:
- Batch multiple environment steps
- Amortize MCTS cost over multiple steps
- Better GPU utilization

**Disadvantages**:
- **Temporal leakage risk**: Future steps see past actions
- Requires careful state management
- More complex implementation

### Leakage Mitigation

**Option**: Multi-agent rollout (no leakage)
```python
# Roll out N independent environments in parallel
envs = [create_env() for _ in range(N)]

# Each env at different time point (no correlation)
for env in envs:
    obs = env.step()  # Different market conditions

# MCTS batch on all observations
actions = mcts_search(all_obs)  # GPU efficient

# Apply actions independently
for env, action in zip(envs, actions):
    env.step(action)
```

**No leakage**: Each environment is independent trajectory

---

## Strategy 4: Hybrid CPU-GPU MCTS (Research Direction)

### Concept
- **GPU**: Policy/value network inference (batch)
- **CPU**: Tree traversal (parallel workers)
- **Hybrid**: Batch GPU calls across many trees

**Implementation**: Already partially done (worker pool)

**Optimization**: Profile and optimize batching
```python
# Instead of: 1 worker → 1 GPU call
# Do: Accumulate requests → batch GPU call → distribute results

# Potential 10-100× GPU efficiency
```

---

## Recommendation: Test Priority

### Phase 1: Quick Verification (30 minutes)
```bash
# Test if larger batches help
python tools/benchmark_parallelization.py --test batch --workers 4

# Test if more workers help
python tools/benchmark_parallelization.py --test workers --batch-size 512

# Check VRAM headroom
python tools/benchmark_parallelization.py --test vram
```

**Decision point**:
- If speedup scales with batch/workers → Optimize and use
- If speedup plateaus → Move to Strategy 2

---

### Phase 2: Multi-Instance Training (1 hour setup)
```bash
# Create parallel training script
# Launch 4 instances
python tools/parallel_training.py --instances 4 --epochs 100

# Monitor
nvidia-smi -l 1
htop
```

**Expected**: 4× throughput, ~5GB VRAM, all cores busy

**If successful**: Use for all future training (massive win)

---

### Phase 3: Hybrid Optimization (if needed)
- Profile worker pool batching
- Optimize GPU call accumulation
- Test multi-step rollouts

---

## Quick Win Estimate

**Current**: ~225 epochs/hour (from docs)

**Strategy 1 Optimized** (batch 1024, 8 workers):
- Conservative: 450 epochs/hour (2× speedup)
- Optimistic: 675 epochs/hour (3× speedup)

**Strategy 2 Multi-Instance** (4 parallel):
- Conservative: 900 epochs/hour (4× speedup)
- Optimistic: 1800 epochs/hour (8× speedup, if instances don't interfere)

**Combined** (optimized + multi-instance):
- Conservative: 1800 epochs/hour (8× speedup)
- Optimistic: 2700 epochs/hour (12× speedup)

---

## Action Plan

1. **Run benchmark** (30 min):
   ```bash
   python tools/benchmark_parallelization.py --test all
   ```

2. **Analyze results**, decide:
   - Batch size to use
   - Worker count to use
   - Whether multi-instance is viable

3. **Implement chosen strategy** (1-2 hours)

4. **Verify on real training** (1 hour)

5. **Document findings** in TASK_01 report

---

**Let's start with the benchmark to get hard numbers!**
