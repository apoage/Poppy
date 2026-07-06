# test_mctx_jax.py

**MCTS Search Performance Test**

## Purpose

Test MCTS (Monte Carlo Tree Search) performance on GPU using the mctx library.

## Usage

```bash
python tools/test_mctx_jax.py
```

## Function

1. Creates MuZero model with 64 state dim, 192 actions
2. Initializes parameters
3. JIT-compiles MCTS search (50 simulations)
4. Runs warmup compilation
5. Benchmarks 100 MCTS searches

## Output Metrics

- **Compilation time**: JIT compilation duration
- **Average search time**: Per-search latency
- **Throughput**: Searches per second

## Example Output

```
🚀 Testing JAX/MCTX MuZero Search on GPU...
Initializing model...
JIT-compiling MCTS search...
✅ Compilation time: 12.34s

Running benchmark (100 steps)...
✅ Benchmark completed:
   Average time: 15.2ms/search
   Throughput: 65.8 searches/sec
```

## Expected Performance

| Hardware | Compilation | Per-Search |
|----------|-------------|------------|
| RTX 3090 | 10-15s | 10-20ms |
| RTX 4090 | 8-12s | 5-15ms |
| CPU only | 20-30s | 200-500ms |

## Dependencies

- mcts_trader.jax_agent.muzero_model
- mcts_trader.jax_agent.mctx_search
- mctx

## Related Files

- `diagnose_jax_training.py` - Full diagnostics
- `test_trm_jax.py` - Test TRM specifically
