# benchmark_parallelization.py

**Parallelization Benchmark**

## Purpose

Benchmark different parallelization strategies for training throughput optimization.

## Usage

```bash
python tools/benchmark_parallelization.py
```

## Strategies Tested

1. **Single-threaded**: Baseline
2. **Multi-process environments**: Parallel self-play
3. **Vectorized environments**: Batched stepping
4. **Async data collection**: Overlapped compute

## Metrics

| Metric | Description |
|--------|-------------|
| Episodes/sec | Self-play throughput |
| Steps/sec | Environment stepping rate |
| GPU utilization | Compute efficiency |
| Memory usage | RAM/VRAM consumption |

## Example Output

```
Parallelization Benchmark
=========================

Strategy: Single-threaded
  Episodes/sec: 2.5
  GPU util: 35%

Strategy: 4x Multi-process
  Episodes/sec: 8.2
  GPU util: 78%
  Speedup: 3.3x

Strategy: Vectorized (batch=4)
  Episodes/sec: 9.5
  GPU util: 85%
  Speedup: 3.8x
```

## Recommendations

- **RTX 3090**: 4 workers, vectorized
- **RTX 4090**: 6 workers, vectorized
- **CPU only**: 2 workers, multi-process

## Dependencies

- multiprocessing
- time

## Related Files

- `parallel_epoch_training.py` - Apply parallelization
- `profile_training.py` - Profile performance
