# TASK_140: Parallel Episode Generation for train_easy.py

## Problem

Current `train_easy.py` generates episodes **sequentially**:
- One episode at a time in `_generate_episode_data()`
- Buffer refill is slow: 32 episodes × ~50ms = 1.6s stall
- GPU sits idle during price generation and oracle computation

## Goal

Add `--n-envs` flag to parallelize episode generation using JAX vmap, matching train_jax_fast.py's approach.

## Current Flow (Sequential)

```
for i in range(N_BUFFER):
    prices = generate_prices()        # CPU, sequential
    oracle = compute_oracle(prices)   # GPU, sequential
    obs = encode(prices)              # GPU, sequential
    buffer.append(...)
```

**Time**: O(N_BUFFER × episode_time)

## Target Flow (Parallel)

```
prices_batch = generate_prices_batch(n_envs)     # CPU, batched
oracle_batch = vmap(compute_oracle)(prices_batch) # GPU, parallel
obs_batch = vmap(encode)(prices_batch)            # GPU, parallel
buffer.extend(...)
```

**Time**: O(N_BUFFER / n_envs × batch_time)

## Implementation Plan

### 1. Batch Price Generation

Modify `SyntheticMarket` or create wrapper to generate N price matrices at once:

```python
def generate_prices_batch(self, n_envs: int, seeds: list[int]) -> jnp.ndarray:
    """Generate (n_envs, T, n_assets) price tensor."""
    prices_list = []
    for seed in seeds:
        self.regenerate(new_seed=seed)
        prices = self._extract_prices()
        prices_list.append(prices)
    return jnp.stack(prices_list, axis=0)
```

### 2. Vectorized Oracle Computation

The `greedy_lookahead_path` is already JIT-compiled. Wrap with vmap:

```python
@partial(jax.vmap, in_axes=(0, None, None, None, None))
def batched_greedy_lookahead_path(returns, k, lookahead, transaction_cost, max_len):
    return greedy_lookahead_path(returns, k, lookahead, transaction_cost, max_len)
```

### 3. Vectorized Encoding

The encoder already supports batching. Ensure it handles (n_envs, T, n_assets):

```python
# Already batched: jit_batch_encode_compressed handles (B, T, n_assets)
```

### 4. Updated OnTheFlyGenerator

```python
class OnTheFlyGenerator:
    def __init__(self, config, seed, bella_generator=None, n_envs=1):
        self.n_envs = n_envs
        # ...

    def _generate_episode_batch(self, levels: list[int], seeds: list[int]) -> list[dict]:
        """Generate n_envs episodes in parallel."""
        # 1. Generate prices batch (CPU)
        prices_batch = self._generate_prices_batch(levels, seeds)

        # 2. Compute oracle batch (GPU, vmapped)
        log_returns_batch = jnp.log(prices_batch[:, 1:] / jnp.maximum(prices_batch[:, :-1], 1e-8))
        oracle_batch = self._batched_oracle(log_returns_batch)

        # 3. Encode observations batch (GPU)
        obs_batch = self._batched_encode(prices_batch)

        # 4. Return list of episode dicts
        return [...]
```

### 5. CLI Flag

```python
parser.add_argument('--n-envs', type=int, default=1,
                    help='Parallel episodes per buffer refill (default: 1)')
```

## Files to Modify

| File | Changes |
|------|---------|
| `train_easy.py` | Add `--n-envs`, modify `OnTheFlyGenerator` |
| `mcts_trader/jax_agent/path_solver.py` | Add `batched_greedy_lookahead_path` |

## Expected Speedup

| n_envs | Buffer Refill Time | Speedup |
|--------|-------------------|---------|
| 1 | ~1.6s | 1x |
| 4 | ~0.5s | 3x |
| 8 | ~0.3s | 5x |
| 16 | ~0.2s | 8x |

*Diminishing returns due to memory bandwidth and kernel launch overhead*

## VRAM Impact

Each parallel episode adds:
- Prices: (T, 64) × 4 bytes = ~100KB
- Oracle: (T, 64) × 4 bytes = ~100KB
- Observations: (T, 27, 64) × 4 bytes = ~6MB

For n_envs=8: ~50MB additional VRAM (negligible)

## Success Criteria

- [ ] `--n-envs 8` runs without OOM
- [ ] Buffer refill time reduced by 4x+ with n_envs=8
- [ ] Training accuracy unchanged (same oracle, just parallelized)
- [ ] Works with both synthetic and Bella data sources

## Testing

```bash
# Baseline (sequential)
python train_easy.py --preset small --epochs 5 --levels 0 --n-envs 1

# Parallel (8 envs)
python train_easy.py --preset small --epochs 5 --levels 0 --n-envs 8

# Compare buffer refill times in logs
```
