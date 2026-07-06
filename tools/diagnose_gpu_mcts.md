# diagnose_gpu_mcts.py

**GPU MCTS Diagnostics**

## Purpose

Diagnose GPU-specific issues with MCTS search.

## Usage

```bash
python tools/diagnose_gpu_mcts.py
```

## Checks

1. CUDA device availability
2. JAX XLA compilation
3. MCTS kernel execution
4. Memory allocation patterns

## Common Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| No GPU | Falls back to CPU | Install jax[cuda12] |
| OOM | Crash during search | Reduce batch size |
| Slow JIT | Long first search | Normal, cached after |
| NaN values | Inf in outputs | Check input scaling |

## Dependencies

- jax
- mctx

## Related Files

- `diagnose_jax_training.py` - Full JAX diagnostics
- `test_mctx_jax.py` - MCTS benchmark
