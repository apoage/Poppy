# TASK_136c: Keep Episode Prices in VRAM (Investigation)

## 1. Current Storage Analysis

In `expert_buffer.py`:
```python
def add_episode(self, prices: np.ndarray) -> int:
    with self._lock:
        ep_id = self._next_episode_id
        # Stores copy in RAM (numpy)
        self._episode_prices[ep_id] = prices.astype(np.float32).copy()
        return ep_id
```

In `_encode_observations_vectorized`:
```python
# Fetches from RAM
close_prices_np = self._episode_prices.get(int(ep_id))
# Implicit transfer to VRAM when passed to JAX JIT function
close_prices = jnp.array(close_prices_np)
```

## 2. VRAM Budget Analysis

*   **Current Usage**: ~17GB (during heavy training).
*   **Total Capacity**: 24GB.
*   **Headroom**: ~7GB.
*   **Episode Size**: 400 steps × 64 assets × 4 bytes ≈ 100 KB.
*   **Storage Cost**:
    *   1 Epoch (50 envs): 5 MB
    *   100 Epochs (accumulated): 500 MB
*   **Conclusion**: Storing thousands of episodes in VRAM is **trivial** relative to the available headroom.

## 3. Bottleneck Analysis

Is the transfer the problem?

*   **Transfer Size**: 100 KB per episode.
*   **PCIe 4.0 Bandwidth**: ~25 GB/s.
*   **Transfer Time**: 100 KB / 25 GB/s ≈ **4 microseconds (0.004ms)**.
*   **Launch Latency**: JAX async dispatch overhead ≈ **10-50 microseconds**.

**Verdict:** The data transfer time itself is negligible. The issue is **overhead latency**.
Calling `jnp.array()` inside a Python loop 50 times triggers 50 separate dispatch events and 50 separate transfers.

**Comparison:**
*   **Python Loop (Current)**: 50 × (Python Overhead + Dispatch + Transfer) ≈ 50 × 1ms ≈ 50ms.
*   **VRAM Storage**: 50 × (Python Overhead + Dispatch) ≈ 50 × 0.5ms ≈ 25ms. (Still slow due to loop).
*   **Batching (136b)**: 1 × (Overhead + Dispatch + 50x Transfer) ≈ 2ms.

## 4. Findings

1.  **VRAM storage is safe**: JAX arrays persist in Python dicts as long as the reference is held. They live on the device.
2.  **Not the primary bottleneck**: Moving storage to VRAM saves the *transfer time* (which is tiny) but does NOT solve the *dispatch overhead* of the Python loop.
3.  **Synergy**: VRAM storage effectively eliminates the transfer cost, but without **Batching (136b)** or **Pre-encoding (136a)**, the CPU-side loop overhead remains dominant.

## 5. Recommendation

Do **NOT** implement VRAM storage as a standalone fix. It yields minimal gain because the loop overhead dominates.

Instead:
1.  Prioritize **TASK_136a (Pre-encoding)**: Eliminates the loop AND the compute.
2.  If 136a is skipped, combine VRAM storage with **TASK_136b (Batching)**.

**Verdict for 136c**: **Low Impact**. The bottleneck is the Python control flow, not the memory bandwidth.
