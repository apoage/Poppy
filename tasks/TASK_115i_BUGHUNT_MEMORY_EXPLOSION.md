# TASK_115i: Memory Leak & Allocation Audit

**Status**: Pending
**Priority**: High
**Created**: 2026-01-25

## Identified Issues

### 1. Metric Accumulation in Trainer (Copy Leak)
*   **File**: `mcts_trader/jax_agent/trainer.py`
*   **Issue**: In `train()` loop, `metrics` (JAX arrays) are appended to `history` lists:
    ```python
    for key, value in metrics.items():
        if key in history:
            history[key].append(value)  # Keep as JAX array
    ```
    This accumulates thousands of small JAX arrays on the GPU/TPU device memory, preventing garbage collection until `train()` finishes. For long epochs (e.g., 5000 steps), this can OOM the device.
*   **Fix**: Move metrics to host memory immediately using `jax.device_get` or convert to numpy, as metrics are small scalars and don't need to stay on accelerator.

### 2. RainbowTableLoader Cache Growth (Unbounded Dict)
*   **File**: `diff_teacher/data_loader.py`
*   **Issue**: `_period_cache` stores full period data `(np.float32 array, timestamps, symbols)`.
    ```python
    self._period_cache: Dict[Tuple[int, int], np.ndarray] = {}
    self._cache_max_size = 5
    ```
    While `_load_period_data` has LRU logic to delete `oldest_key`, Python's `del` on a dict key doesn't guarantee immediate memory reclamation if references persist elsewhere (e.g., in `_current_data` used by streaming).
    Also, `5` periods of `50k * 64 * 5 * 4 bytes` is ~64MB per period, which is manageable, but if `window_size` or resolution increases, this grows fast.
*   **Fix**: Verify explicit `del` behavior and potentially use `weakref` or explicit `None` assignment for `_current_data` before switching.

### 3. Replay Buffer Aux Data Copies (Unnecessary Copies)
*   **File**: `mcts_trader/jax_agent/replay_buffer.py`
*   **Issue**: `add_auxiliary` takes `prices`, `next_prices`, `returns`, etc. as inputs.
    It then does:
    ```python
    log_prices = np.log(np.clip(prices, ...))
    ```
    This creates full-size copies of the input arrays. If `add_auxiliary` is called with large arrays, this spikes RAM usage.
    More critically, `sample_batch` creates **new JAX arrays** from the *entire* buffer backing storage on *every call*:
    ```python
    # Convert auxiliary numpy arrays to JAX for indexing
    # (only done once per sample_batch call)
    log_prices_jax = jnp.array(self._log_prices)
    ...
    ```
    `self._log_prices` is the FULL capacity buffer (e.g., 100k steps). `jnp.array()` copies this ENTIRE array to GPU memory on EVERY BATCH SAMPLE.
    For a 100k buffer: `100,000 * 64 * 4 bytes ≈ 25MB`. Doing this copy 100 times/sec is a massive bandwidth hog and GC pressure generator.
*   **Fix**: Do NOT convert the full buffer to JAX. Index the numpy array on CPU first using the sampled indices, *then* convert the small batch to JAX.

## Plan
1.  Fix `trainer.py` to `device_get` metrics immediately.
2.  Fix `replay_buffer.py` to sample on CPU and transfer only the batch.
3.  Review `data_loader.py` caching strategy.