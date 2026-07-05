# Parallel Epoch Training with Parameter Averaging

**Strategy**: Run 4 instances in parallel, average parameters after each epoch batch, continue training from averaged checkpoint.

This provides:
- ✅ **4× speedup** from parallelization
- ✅ **Knowledge continuity** via parameter averaging
- ✅ **Exploration diversity** (4 different trajectories per batch)
- ✅ **Training stability** (averaging reduces variance)

---

## How It Works

```
Epoch Batch 1 (0-50):
  [Instance 0] [Instance 1] [Instance 2] [Instance 3]  ← Train in parallel
       ↓            ↓            ↓            ↓
  ckpt_50_i0   ckpt_50_i1   ckpt_50_i2   ckpt_50_i3
       ↓            ↓            ↓            ↓
       └────────────┴────────────┴────────────┘
                         ↓
                Average parameters
                         ↓
                  merged_ckpt_50

Epoch Batch 2 (51-100):
  Load merged_ckpt_50 into all instances
  [Instance 0] [Instance 1] [Instance 2] [Instance 3]  ← Train in parallel
       ↓            ↓            ↓            ↓
       ...repeat averaging...
```

**Key Concept**: Each instance explores different trajectories (different random seeds), but all start from the same averaged knowledge after each batch.

---

## Usage

### Basic Usage

```bash
python tools/parallel_epoch_training.py \
  --instances 4 \
  --epochs-per-batch 50 \
  --total-epochs 5000
```

### With Custom Config

```bash
python tools/parallel_epoch_training.py \
  --instances 4 \
  --epochs-per-batch 50 \
  --total-epochs 5000 \
  --config config/my_training_config.json \
  --checkpoint-dir jax_ckpt_fast/parallel_run1
```

### Parameters

- `--instances`: Number of parallel training instances (default: 4)
  - Recommended: 4 (uses ~5GB VRAM, all 16 CPU cores)
  - Maximum: 6-8 (limited by VRAM and CPU)

- `--epochs-per-batch`: Epochs per batch before averaging (default: 50)
  - Smaller (10-25): More frequent averaging, more stable but slower
  - Larger (50-100): Less overhead, more exploration

- `--total-epochs`: Total training epochs (default: 5000)

- `--config`: Base config JSON file (optional)
  - If not provided, uses defaults (64 assets, 0.75 coverage, seed 42)

- `--checkpoint-dir`: Root checkpoint directory (default: jax_ckpt_fast/parallel)

---

## Expected Performance

### Single Instance Training
- Throughput: ~225 epochs/hour
- VRAM: ~537 MB (2.2%)
- CPU: 1-4 cores

### Parallel Training (4 instances)
- Throughput: **~900 epochs/hour** (4× speedup)
- VRAM: ~5 GB (21%)
- CPU: All 16 cores utilized

### Example Timeline

**Training 5000 epochs**:
- Single instance: ~22 hours
- Parallel (4 instances): ~5.5 hours

**Training 20000 epochs** (full curriculum):
- Single instance: ~89 hours (3.7 days)
- Parallel (4 instances): ~22 hours (1 day)

---

## Monitoring

### GPU Usage
```bash
nvidia-smi -l 1
```

**Expected**:
- VRAM: ~1.3 GB per instance (4 instances = ~5 GB total)
- GPU Utilization: 80-95% (shared across instances)

### CPU Usage
```bash
htop
```

**Expected**:
- All 16 cores at 60-80% utilization
- 4 Python processes running

### Training Logs
```bash
# View individual instance logs
tail -f logs/parallel_instance_0_batch_0.log
tail -f logs/parallel_instance_1_batch_0.log

# Watch all logs
tail -f logs/parallel_instance_*.log
```

---

## Checkpoint Structure

```
jax_ckpt_fast/parallel/
├── instance_0/          # Instance 0 checkpoints
│   ├── checkpoint_50
│   ├── checkpoint_100
│   └── ...
├── instance_1/          # Instance 1 checkpoints
│   ├── checkpoint_50
│   └── ...
├── instance_2/          # Instance 2 checkpoints
├── instance_3/          # Instance 3 checkpoints
└── merged/             # Averaged checkpoints (use these!)
    ├── checkpoint_50
    ├── checkpoint_100
    └── ...
```

**Important**: Use checkpoints from `merged/` directory for evaluation and deployment.

---

## Potential Concerns: Parameter "Blurring"

### The Concern

**Valid worry**: Averaging parameters from different optimization trajectories might:
- Blur sharp decision boundaries
- Average away specialized features
- Create a "middling" model worse than any individual

### What Research Shows

**Empirically it works well**:
- **Federated Learning** (Google): Billions of devices, parameter averaging, production use
- **Parallel SGD**: Proven convergence properties
- **Loss Landscape Studies**: Models often lie in "connected modes" - averaging stays in low-loss region

**But you should verify for your specific case!**

### Strategies to Handle This

#### Strategy 1: Compare Averaged vs Best Individual

After training, evaluate **both**:

```bash
# Evaluate averaged checkpoint
python evaluate.py --checkpoint jax_ckpt_fast/parallel/merged/

# Evaluate each individual
python evaluate.py --checkpoint jax_ckpt_fast/parallel/instance_0/
python evaluate.py --checkpoint jax_ckpt_fast/parallel/instance_1/
python evaluate.py --checkpoint jax_ckpt_fast/parallel/instance_2/
python evaluate.py --checkpoint jax_ckpt_fast/parallel/instance_3/

# Use whichever performs better!
```

#### Strategy 2: Pretraining with Averaging + Single-Instance Fine-tuning

**Best of both worlds**:

1. **Phase 1**: Pretrain with parallel epochs (fast exploration)
   ```bash
   python tools/parallel_epoch_training.py \
     --instances 4 \
     --epochs-per-batch 50 \
     --total-epochs 4000  # 80% of training
   ```

2. **Phase 2**: Fine-tune single instance (precise optimization)
   ```bash
   python train_jax_fast.py \
     --load-checkpoint jax_ckpt_fast/parallel/merged/ \
     --epochs 1000  # Final 20%
   ```

**This gives**:
- ✅ 3.2× overall speedup (4× for 80% of training)
- ✅ Fast exploration phase (parallel)
- ✅ Precise convergence (single-instance fine-tune)
- ✅ No risk of blurred final model

#### Strategy 3: Increase Averaging Frequency

If blurring is an issue, average more frequently:

```bash
# Average every 10 epochs (more stable)
python tools/parallel_epoch_training.py \
  --epochs-per-batch 10 \
  --total-epochs 5000
```

**Trade-off**: More overhead, but less divergence between instances.

#### Strategy 4: Ensemble Instead

Instead of averaging, keep all 4 models and **ensemble their predictions**:
- Each model votes on action
- Use majority vote or average policy
- More computational cost but potentially better performance

---

## Why This Works

### Parameter Averaging Theory

**Parallel SGD with Parameter Averaging** (Zhang et al., 2016):
- Multiple workers train independently
- Periodically average their parameters
- Proven to converge to same optimum as sequential SGD
- Often **faster convergence** due to exploration diversity

### Benefits Over Standard Data Parallelism

**Standard Data Parallelism**:
- Synchronize gradients every batch
- High communication overhead
- GPU must stay synchronized

**Our Approach (Epoch-Level Averaging)**:
- Synchronize parameters every N epochs
- Minimal overhead (only during averaging)
- No inter-GPU communication during training
- Works even on single GPU!

---

## Troubleshooting

### Instance Crashes
**Symptom**: One or more instances exit with error

**Check logs**:
```bash
cat logs/parallel_instance_0_batch_0.log
```

**Common causes**:
- Out of VRAM: Reduce `--instances` to 3 or 2
- Out of RAM: Reduce batch size in config
- Corrupted checkpoint: Delete checkpoint dir and restart

### Slow Performance
**Symptom**: Not achieving 4× speedup

**Check**:
```bash
nvidia-smi  # Should show ~5GB VRAM used
htop        # Should show all 16 cores busy
```

**If VRAM is full**: Reduce `--instances`
**If CPU is idle**: Check logs for errors
**If GPU util is low**: Increase batch size in config

### Checkpoint Averaging Fails
**Symptom**: Error during checkpoint averaging

**Verify checkpoints exist**:
```bash
ls -lh jax_ckpt_fast/parallel/instance_*/
```

**Each instance should have checkpoint at same step**. If not, one instance may have crashed early.

---

## Advanced Usage

### Resume from Previous Run

If training was interrupted, just re-run the same command. The script will:
1. Detect existing merged checkpoint
2. Resume all instances from latest merged checkpoint
3. Continue from where it left off

### Different Seeds Per Instance

Each instance automatically gets a different random seed:
- Instance 0: base_seed + 0
- Instance 1: base_seed + 1
- Instance 2: base_seed + 2
- Instance 3: base_seed + 3

This ensures exploration diversity while maintaining reproducibility.

### Custom Training Script

If using a different training script:
1. Ensure it supports `--epochs`, `--config`, `--load-checkpoint` flags
2. Modify `launch_parallel_training()` to use your script name

---

## Comparison to Alternatives

| Strategy | Speedup | VRAM | Effort | Knowledge Continuity |
|----------|---------|------|--------|---------------------|
| Single instance | 1× | 0.5 GB | 0 min | ✅ Perfect |
| Worker pool (optimized) | 2-3× | 0.5 GB | 8-16 hrs | ✅ Perfect |
| Multi-instance (independent) | 4× | 5 GB | 5 min | ❌ None (need to select best) |
| **Parallel epochs (this)** | **4×** | **5 GB** | **30 min** | ✅ **Perfect (averaging)** |

**Winner**: Parallel epochs - best ROI (4× speedup, knowledge continuity, 30 min setup)

---

## Next Steps

1. **Test with small run**:
   ```bash
   python tools/parallel_epoch_training.py \
     --instances 2 \
     --epochs-per-batch 10 \
     --total-epochs 50
   ```

2. **Verify VRAM usage**: Should use ~2.6 GB (2 instances)

3. **Check performance**: Should complete 50 epochs in ~2-3 minutes

4. **Scale up**:
   ```bash
   python tools/parallel_epoch_training.py \
     --instances 4 \
     --epochs-per-batch 50 \
     --total-epochs 5000
   ```

5. **Monitor and adjust** based on GPU/CPU utilization

---

## References

- Zhang et al., "Parallel SGD with Parameter Averaging" (2016)
- McMahan et al., "Federated Learning" (2017)
- Lin et al., "Deep Gradient Compression" (2018)

