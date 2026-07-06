# profile_training.py

**Training Performance Profiler**

## Purpose

Profile training process in real-time to identify CPU/GPU/RAM bottlenecks.

## Usage

```bash
# Profile specific process
python tools/profile_training.py --pid 12345

# Profile with custom interval
python tools/profile_training.py --interval 1.0
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--pid` | None | Target process ID |
| `--interval` | 0.5 | Sampling interval (seconds) |

## Metrics Collected

### GPU Metrics
- GPU utilization %
- Memory utilization %
- Memory used/total MB

### CPU Metrics
- Per-core utilization %
- Total utilization %
- Active cores count

### Process Metrics
- Process memory (RSS, VMS)
- Child process stats
- Thread count

## Example Output

```
[14:32:15] GPU 0: 85% util, 18.2GB/24GB mem
[14:32:15] CPU: 45% total, 8/12 cores active
[14:32:15] Process: 12.3GB RAM, 4 children
```

## Bottleneck Detection

**GPU bottleneck**: GPU util > 90%, CPU < 50%
**CPU bottleneck**: CPU > 80%, GPU < 50%
**Memory bottleneck**: RAM near limit, swap activity

## Dependencies

- psutil
- nvidia-smi (for GPU stats)

## Related Files

- `profile_memory.py` - Detailed memory profiling
- `benchmark_parallelization.py` - Parallel performance
