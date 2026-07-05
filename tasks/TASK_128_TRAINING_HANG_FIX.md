# TASK_128: Training Hang Bug Fix

## Root Cause Analysis

**Primary Suspect**: JAX operations inside threading.Lock

In `expert_buffer.py:sample_batch()`:
```python
with self._lock:
    # ... sampling logic ...
    observations = self._encode_observations_vectorized(entries)  # <-- JAX called while holding lock!
```

The `_encode_observations_vectorized` function calls `jax.vmap()` and JIT-compiled encoders while holding `self._lock`. JAX may spawn its own threads or interact with the GIL in ways that conflict with the lock.

## The Fix

### Fix 1: Release Lock Before JAX Operations (CRITICAL)

Modify `sample_batch()` to release the lock before calling JAX:

```python
def sample_batch(self, batch_size=1024, demo_ratio=0.125):
    # ... validation ...
    
    with self._lock:
        # Do all the sampling logic inside lock
        n_demo = min(int(batch_size * demo_ratio), self.demo_count)
        n_agent = batch_size - n_demo
        # ... etc ...
        
        # Get entries and compute weights while holding lock
        entries = self.indices[indices].copy()  # COPY to release lock safely
        weights = compute_weights()  # Do this inside lock
        
    # RELEASE LOCK before JAX operations
    observations = self._encode_observations_vectorized(entries)
    
    return {...}
```

### Fix 2: Add Hang Detection

Add timeouts to detect where the hang occurs:

```python
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

# Set 30-second timeout for self-play
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(30)
try:
    trajectories = run_self_play_jax(...)
finally:
    signal.alarm(0)
```

### Fix 3: JAX Cache Management

Clear JAX caches periodically to prevent memory buildup:

```python
if epoch % 10 == 0:
    jax.clear_caches()
```

## Implementation

See the actual code changes in:
- `mcts_trader/jax_agent/expert_buffer.py` - Release lock before JAX
- `train_jax_fast.py` - Add more debug instrumentation

## Testing

Run the reproduction command:
```bash
python train_jax_fast.py \
  --name hang_fix_test \
  --synthetic-level 0 \
  --epochs 100 \
  --steps 200 \
  --parallel-envs 4 \
  --jax-env \
  --batch-size 512 \
  --use-prediction-loss \
  --imprint
```

Should complete 100 epochs without hanging.
