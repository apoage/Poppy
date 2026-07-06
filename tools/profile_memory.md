# profile_memory.py

**Memory Profiler**

## Purpose

Detailed memory profiling for training to identify memory leaks and optimize usage.

## Usage

```bash
python tools/profile_memory.py
```

## Function

1. Tracks RAM allocation over time
2. Monitors GPU VRAM usage
3. Identifies memory growth patterns
4. Detects potential leaks

## Metrics

| Metric | Description |
|--------|-------------|
| Peak RAM | Maximum RAM usage |
| Peak VRAM | Maximum GPU memory |
| Growth rate | Memory increase per epoch |
| Allocations | Object allocation count |

## Example Output

```
Memory Profile
==============
Peak RAM: 14.2 GB
Peak VRAM: 18.5 GB
Growth: +50MB/100 epochs (stable)

Top allocations:
  Replay buffer: 8.2 GB
  Model params: 0.5 GB
  JAX cache: 2.1 GB
```

## Dependencies

- psutil
- nvidia-ml-py (pynvml)

## Related Files

- `profile_training.py` - Full profiling
