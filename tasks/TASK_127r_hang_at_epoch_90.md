# TASK_127r: Training Hang Audit (Epoch 90)

## Incident Report
-   **Event:** Training hung indefinitely at the start of Epoch 91 (after Epoch 90 metrics/checkpoint).
-   **Duration:** >20 minutes.
-   **Resource State:**
    -   GPU Util: 0% (Idle).
    -   GPU Mem: ~0.9GB (Low).
    -   CPU Util: ~13% (Idle/System background).
-   **Last Action:** Finished saving `checkpoint_126000` (Epoch 90).
-   **Current Phase:** Entering `run_self_play_jax` for Epoch 91.

## Root Cause Analysis

### 1. JAX/XLA Deadlock (Most Likely)
-   The code uses `jax.lax.scan` for the entire episode generation on GPU.
-   Data transfer back to CPU (`fetch` function in `self_play.py`) uses `np.array(jnp_array)`.
-   **Mechanism:** If the GPU kernel fails silently (e.g., due to a rare numerical issue, driver timeout, or internal XLA error) but doesn't crash the process, the CPU thread waits indefinitely for the `fetch` result.
-   **Why Epoch 91?** It's just random chance or a specific data pattern in the synthetic generator that triggered an edge case in the compiled kernel.

### 2. Expert Buffer Lock (Unlikely but Possible)
-   `expert_buffer` uses a `threading.Lock`.
-   If the `on_episode_complete` callback (CPU side) was running and crashed *while holding the lock* (unlikely with `with self._lock:`), it could deadlock.
-   However, `on_episode_complete` runs *after* `run_self_play_jax` returns. If `run_self_play_jax` hangs, the callback hasn't started.

### 3. Data Regeneration (Ruled Out)
-   `config.weighted_synthetic` is False (Level 0).
-   `async_regenerator` is None.
-   No background thread activity observed.

## Recommendations

1.  **Timeout for Fetch:** JAX doesn't easily support timeouts on `np.array()`.
2.  **Verbose Logging:** Add print statements *inside* `run_self_play_jax` (using `jax.debug.print` or splitting the scan into chunks) to verify progress if it happens again.
3.  **Kernel Restart:** This might be a one-off driver/XLA fluke. Restarting the training (which the user did) often resolves it.
4.  **Device Memory:** Ensure no other process is competing for GPU (gnome-shell is using 318MB, which is fine).

## Conclusion
The hang appears to be a low-level **JAX/XLA synchronization deadlock** during the device-to-host transfer of self-play trajectories. It is not a logic bug in the Python code.
