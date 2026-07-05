# TASK_128b: Training Hang Root Cause Analysis

**Status**: ANALYSIS COMPLETE
**Priority**: CRITICAL
**Created**: 2026-02-01
**Parent**: TASK_128

## Executive Summary

Training hangs at epoch 60-90. **Memory is NOT the cause** - training runs fine at 17GB/93% GPU util. The hang is likely triggered by:
1. A specific numerical condition (NaN/Inf propagation)
2. XLA recompilation stall on a specific shape
3. Deadlock in device-to-host transfer
4. Specific data pattern triggering infinite loop in scan

## Root Cause Identification

### Primary Cause: Dynamic Shape JIT Recompilation

**Location**: Two call sites for `compute_optimal_path`:

1. **self_play.py:814** - Called per-environment during trajectory packaging:
```python
optimal_path = compute_optimal_path(
    jnp.array(log_returns), k=4, method='greedy', transaction_cost=0.0
)
```

2. **train_jax_fast.py:628** - Called per-episode in imprint callback:
```python
optimal_path = compute_optimal_path(
    jnp.array(log_returns), k=4, method='greedy', transaction_cost=0.0
)
```

**Problem**: The function `greedy_lookahead_path` in `path_solver.py:33-126` is decorated with:
```python
@partial(jax.jit, static_argnames=['k', 'lookahead'])
```

The sequence length `T` is NOT static, meaning:
- Every unique episode length triggers a new JIT compilation
- With 4 parallel envs × 2 call sites = 8 compilations per epoch (worst case)
- Episode lengths vary (truncation, early termination)
- Over 60-90 epochs: potentially 500+ unique compilations

### REVISED: Sudden Trigger, Not Gradual Exhaustion

**New Evidence** (from live monitoring):
- Training runs fine at 17GB / 93% GPU utilization
- Memory consumption is stable, not growing unboundedly
- Hang is SUDDEN, not gradual degradation

**Revised Hypotheses**:

1. **NaN/Inf Propagation**: Specific batch causes NaN in loss → blocks gradient computation
2. **Device-to-Host Deadlock**: `np.array(jax_array)` blocks because underlying computation never finishes
3. **XLA Recompilation During Async**: New shape triggers compilation while previous op still running
4. **Numerical Condition in lax.scan**: A step function returns NaN → scan behavior undefined
5. **Specific Data Pattern**: Certain price sequence triggers edge case in path solver

### Contributing Factors

#### Factor 1: Double compute_optimal_path Calls

With `--imprint` enabled, `compute_optimal_path` is called TWICE per environment:
1. In `run_self_play_jax` (line 814) - for `episode_optimal_alpha` metric
2. In `on_episode_complete_callback` (line 628) - for imprinting

This doubles the recompilation pressure.

#### Factor 2: lax.scan Inside JIT

**Location**: `path_solver.py:109-111`
```python
_, (all_masks, all_indices, step_pnls, step_changes) = jax.lax.scan(
    step, init_mask, jnp.arange(T)
)
```

The scan length `T` is dynamic, requiring XLA to compile different unrolled versions.

#### Factor 3: Prediction Loss ODE Rollout

**Location**: `prediction/market_ode.py:443-447`
```python
_, trajectory = jax.lax.scan(
    scan_fn,
    (log_prices, log_returns, moving_avg, init_variance),
    (rngs, action_seq),
)
```

With `--use-prediction-loss`, this adds another scan-based operation per training step.

#### Factor 4: Expert Buffer Threading

**Location**: `expert_buffer.py:118`
```python
self._lock = threading.Lock()
```

While the lock uses context managers correctly, the combination of:
- Python threading lock
- JAX async operations
- Callback execution during JAX scan

Could create subtle timing issues, though this is less likely than compilation cache.

## Call Flow Analysis

```
Epoch N with --imprint --use-prediction-loss:

1. run_self_play_jax()
   ├── jax.lax.scan(episode_step, ..., length=200)  # GPU fused
   ├── Device-to-host transfer (cpu_* arrays)
   ├── For each env (4 envs):
   │   ├── compute_optimal_path(T_i steps)  # JIT if new T_i
   │   └── on_episode_complete_callback()
   │       └── compute_optimal_path(T_i steps)  # JIT if new T_i (DUPLICATE!)
   └── Return trajectories

2. trainer.train(50 iterations)
   ├── For each iteration:
   │   ├── sample_batch()
   │   ├── _loss_fn()
   │   │   ├── MuZero K-step unroll
   │   │   └── _compute_prediction_loss()
   │   │       └── model.predict_multi_step()
   │   │           └── market_ode.rollout() → lax.scan  # Fixed K, OK
   │   └── apply_gradients()
   └── Return metrics

Memory grows with each new T_i seen.
After 60-90 epochs, cache exhausts available GPU memory.
```

## Evidence Summary

| Observation | Explanation |
|-------------|-------------|
| Hangs at 60-90 epochs | Specific condition triggered |
| GPU memory 17GB during NORMAL training | **Memory is NOT the issue** |
| GPU util 93% before hang | Training runs fine at high memory |
| GPU util 0% WHEN hung | Something blocks execution |
| More frequent with --imprint | compute_optimal_path or callback issue |
| More frequent with --use-prediction-loss | ODE scan or numerical issue |
| No crash, just freeze | Deadlock or infinite wait, not OOM |

**Key Insight**: Training runs happily at 17GB/93% GPU. The hang is NOT gradual memory exhaustion - it's a sudden block triggered by specific conditions.

## Proposed Fixes

### Fix 1: Pad to Fixed Sequence Length (Recommended)

Modify `compute_optimal_path` to pad inputs to a fixed maximum length:

```python
# path_solver.py
MAX_EPISODE_LENGTH = 256  # Or max(config.steps)

@partial(jax.jit, static_argnames=['k', 'lookahead', 'max_len'])
def greedy_lookahead_path(
    returns: jnp.ndarray,
    k: int = 4,
    lookahead: int = 5,
    transaction_cost: float = 0.001,
    max_len: int = MAX_EPISODE_LENGTH,  # NEW: static max length
) -> OptimalPath:
    T, n_assets = returns.shape

    # Pad to fixed length for consistent JIT shape
    padded_returns = jnp.zeros((max_len, n_assets))
    padded_returns = padded_returns.at[:T].set(returns)

    # ... rest of function uses max_len for scan ...
    # Slice results back to T at the end
```

**Benefit**: Single JIT compilation regardless of episode length.

### Fix 2: Remove Duplicate compute_optimal_path

The callback already receives `episode_optimal_alpha` from self_play. Remove the redundant computation:

```python
# train_jax_fast.py - on_episode_complete_callback
# REMOVE this block (lines 627-630):
# optimal_path = compute_optimal_path(...)

# The result dict already has episode_optimal_alpha computed in self_play
# Use that instead of recomputing
```

**Benefit**: Halves the number of compute_optimal_path calls.

### Fix 3: Periodic Cache Clearing

Add explicit cache clearing every N epochs:

```python
# train_jax_fast.py - in main loop
if epoch > 0 and epoch % 20 == 0:
    jax.clear_caches()
    print(f"[CACHE] Cleared JAX caches at epoch {epoch}")
```

**Benefit**: Prevents unbounded cache growth.

### Fix 4: Use numpy Instead of JAX for Path Solver

Since `compute_optimal_path` is called after device-to-host transfer (on CPU data), use numpy:

```python
# path_solver.py - add numpy version
def compute_optimal_path_numpy(returns_np, k=4, method='greedy', **kwargs):
    """CPU-only path solver - no JIT, no recompilation."""
    # Implement with numpy operations
    # No jax.lax.scan, no JIT
```

**Benefit**: Zero JIT overhead, runs on CPU alongside callback.

### Fix 5: Disable Imprinting Path Computation

If imprinting doesn't need the full optimal path (just actions), simplify:

```python
# train_jax_fast.py - callback
# Instead of compute_optimal_path, use simpler heuristic:
# - Take action based on sign of next return (already known)
# - No need for full k-step lookahead path
```

## Next Steps: Catch the Trigger

Since memory is NOT the issue, we need to catch WHAT triggers the hang:

### 1. Add NaN/Inf Detection
```python
# In trainer.py after gradient computation
grad_has_nan = jax.tree_util.tree_reduce(
    lambda a, b: a | b,
    jax.tree_util.tree_map(lambda x: jnp.any(jnp.isnan(x) | jnp.isinf(x)), grads)
)
jax.debug.callback(lambda x: print(f"[NAN-CHECK] Grads have NaN: {x}") if x else None, grad_has_nan)
```

### 2. Add Timeout Wrappers
```python
# Wrap device-to-host transfers with timeout
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("Device transfer took too long")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(60)  # 60 second timeout
try:
    cpu_array = np.array(jax_array)
finally:
    signal.alarm(0)
```

### 3. Log Last Successful Operations
```python
# Add to main loop - save state before each operation
with open('/tmp/hang_debug.txt', 'w') as f:
    f.write(f"epoch={epoch} phase=self_play\n")
    f.flush()
# ... do self_play ...
with open('/tmp/hang_debug.txt', 'w') as f:
    f.write(f"epoch={epoch} phase=train\n")
    f.flush()
```

### 4. Use jax.debug.print Inside Scans
```python
# In path_solver.py scan body
def step(prev_mask, t):
    jax.debug.print("path_solver step {t}", t=t)
    # ... rest of step
```

## Verification Steps

1. **Catch the trigger**: Run with NaN detection and file logging
2. **Check /tmp/hang_debug.txt**: See which phase hung
3. **Correlate with epoch**: Does it happen at specific epoch or random?

## Quick Test Commands

```bash
# Test A: Without imprinting (halves compute_optimal_path calls)
python train_jax_fast.py --epochs 100 --jax-env --use-prediction-loss
# Expected: May still hang but later (epoch 120+)

# Test B: Without prediction loss (removes ODE scans)
python train_jax_fast.py --epochs 100 --jax-env --imprint
# Expected: May still hang due to path solver

# Test C: With periodic cache clear (add to code first)
python train_jax_fast.py --epochs 100 --jax-env --use-prediction-loss --imprint
# Expected: Should not hang if cache is the issue

# Test D: Single env (reduces shape variation)
python train_jax_fast.py --epochs 100 --jax-env --parallel-envs 1
# Expected: Fewer unique shapes, may hang later
```

## Files to Modify

| File | Change | Priority |
|------|--------|----------|
| `path_solver.py` | Add fixed-length padding | HIGH |
| `train_jax_fast.py:628` | Remove duplicate compute_optimal_path | HIGH |
| `train_jax_fast.py` main loop | Add periodic jax.clear_caches() | MEDIUM |
| `self_play.py:814` | Use numpy path solver | MEDIUM |

## Related Tasks

- TASK_128: Training Hang Bughunt (parent)
- TASK_127r: Hang at Epoch 90 (previous analysis)
- TASK_119: Optimal Hindsight Path (introduced compute_optimal_path)
- TASK_125: Causal Expert Imprinting (introduced callback)
