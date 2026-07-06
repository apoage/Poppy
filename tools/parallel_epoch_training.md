# parallel_epoch_training.py

**Parallel Epoch Training**

## Purpose

Run training with epoch-level parallelization for faster data collection.

## Usage

```bash
python tools/parallel_epoch_training.py
```

## Function

1. Spawns multiple environment instances
2. Collects experience in parallel
3. Aggregates into shared replay buffer
4. Trains on batched updates

## Parallelization Strategy

```
[Env 0] → Experience → ┐
[Env 1] → Experience → ├→ Replay Buffer → Trainer
[Env 2] → Experience → ┤
[Env 3] → Experience → ┘
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_workers` | 4 | Parallel environments |
| `epochs` | 100 | Training epochs |
| `batch_size` | 256 | Training batch size |

## Performance

| Workers | Throughput | GPU Util |
|---------|------------|----------|
| 1 | 1x | 40% |
| 2 | 1.8x | 60% |
| 4 | 3.2x | 85% |

## Dependencies

- multiprocessing

## Related Files

- `benchmark_parallelization.py` - Benchmark strategies
- `train_jax_fast.py` - Main training script
