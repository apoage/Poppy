# TASK_128: Training Hang Bug Hunt - Final Report

## Root Cause Analysis
The training hang at random epochs (60+) was caused by a **Reentrant Lock Deadlock** in `mcts_trader/jax_agent/price_cache.py`.

### The Culprit
- **Function:** `PriceCache.add_episode()`
- **Logic:** `add_episode` acquires `self._lock` to write data, then checks if it's time to flush (`self._flush_counter >= 100`). If true, it calls `self.flush()`.
- **The Deadlock:** `self.flush()` also attempts to acquire `self._lock`.
- **The Bug:** `self._lock` was initialized as a standard `threading.Lock()`, which is **not reentrant** in Python. When the same thread attempted to acquire the lock a second time, it blocked indefinitely.

### Why it was "Random"
- The hang only occurred when the `_flush_counter` hit exactly 100.
- With 4 parallel environments, each epoch adds 4 episodes.
- The deadlock triggered every 25 epochs (100 / 4), but appeared random due to the starting counter and occasional early returns in the callback (e.g., when alpha was negative).

### Verification
- **GPU Stats:** 0% GPU utilization and stable 17GB VRAM confirmed the hang was in Python (CPU-side), not a JAX kernel crash or a memory leak.
- **Trace logs:** Granular debug prints pinpointed the hang inside `price_cache.add_episode` right before the oracle path calculation.

## Implemented Fix

### 1. `price_cache.py`: Reentrant Lock
Changed `threading.Lock()` to `threading.RLock()` in the `PriceCache` constructor.
- **Result:** `RLock` allows the owner thread to acquire the lock multiple times without blocking, perfectly handling the `add_episode -> flush` call pattern.

## Conclusion
The system is now stable for 100+ epochs. The "Recompilation Bomb" and "Thread Contention" hypotheses were ruled out by the definitive trace identifying the reentrant lock failure.
