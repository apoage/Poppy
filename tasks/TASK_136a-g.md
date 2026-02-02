# TASK_136a: Pre-encode Observations Investigation

## 1. Memory Analysis

The current observation shape is `(27, n_assets, n_assets)`. For `n_assets=64`, this is `(27, 64, 64)`.
However, analysis of `mcts_trader/env/jax_encoder.py` reveals that **all 27 channels are tiled across the last dimension**. The encoder computes a `(27, 64)` feature vector and broadcasts it to `(27, 64, 64)`.

**Compression Opportunity:**
We can store the `(27, 64)` untiled features and restore the full shape at sample time via broadcasting.

**Cost Calculation (50,000 demos):**
*   **Full `(27, 64, 64)`**:
    $50,000 \times 27 \times 64 \times 64 \times 4 \text{ bytes} \approx \textbf{22.1 GB}$ (Infeasible)
*   **Compressed `(27, 64)`**:
    $50,000 \times 27 \times 64 \times 4 \text{ bytes} \approx \textbf{345.6 MB}$ (Feasible)
*   **Compressed `(27, 64)` float16**:
    $345.6 \text{ MB} / 2 \approx \textbf{172.8 MB}$ (Highly Feasible)

**Conclusion:** Pre-encoding is feasible **only if** we store the compressed `(27, 64)` format.

## 2. Implementation Feasibility

### A. Encoder Context
The `on_episode_complete_callback` in `train_jax_fast.py` runs in the main process loop. It has full access to JAX and the GPU. It currently calls `expert_buffer.add_episode(prices)` and then iterates to add demos.

### B. Batch Encoding
Instead of encoding one step at a time (which would be slow), we can encode the **entire episode** in one JAX call before adding samples.
*   Input: `prices` `(T, 64)`
*   Operation: `batch_encode_simple` (modified to return compressed)
*   Output: `encoded_episode` `(T, 27, 64)`

### C. Agent Buffer
The `UnifiedIndexBuffer` also supports `add_agent` (though currently unused in the main loop). To maintain consistency, the buffer infrastructure should support compressed storage for both demo and agent slots, or we only implement this for the demo zone if agent samples are not being used for BC. Assuming we want a unified solution, we allocate a backing store for the full capacity.

Total capacity (550k samples) compressed:
$550,000 \times 27 \times 64 \times 4 \text{ bytes} \approx \textbf{3.8 GB}$.
This fits easily in system RAM (CPU) and potentially in VRAM on a 24GB card, though RAM is safer to leave VRAM for training.

## 3. Implementation Plan

### Step 1: Modify `jax_encoder.py`
Add `encode_observation_compressed` functions that return `(27, n_assets)` instead of `(27, n_assets, n_assets)`.
*   Create `_compute_all_channels_compressed` that removes `jnp.tile` calls.
*   Expose `jit_batch_encode_compressed`.

### Step 2: Modify `expert_buffer.py`
*   **Storage**: Add `self.compressed_obs = np.zeros((total_capacity, 27, n_assets), dtype=np.float32)` (CPU RAM).
*   **Add Demo**: Update `add_demo` to accept `encoded_obs` argument.
    ```python
    def add_demo(..., encoded_obs: np.ndarray = None):
        ...
        self.compressed_obs[idx] = encoded_obs
    ```
*   **Sampling**: Rewrite `sample_batch`.
    *   Remove `_encode_observations_vectorized` (and the loop over episodes).
    *   Fetch compressed obs: `batch_obs_compressed = self.compressed_obs[indices]` (Host -> Device transfer).
    *   Expand on GPU: Use a JIT function `expand_observations(batch_obs_compressed)` which performs `jnp.tile(x[:, :, :, None], (1, 1, 1, n_assets))`.

### Step 3: Update `train_jax_fast.py`
Update `on_episode_complete_callback`:
1.  Compute `window_starts` for all valid steps.
2.  Call `jit_batch_encode_compressed(prices, window_starts)` to get `(N, 27, 64)` features.
3.  Convert to numpy (CPU).
4.  Pass the specific step's encoded vector to `expert_buffer.add_demo`.

## 4. Expected Impact
*   **Throughput**: Eliminates Python loops and redundant PCI transfers during BC sampling. BC batch preparation becomes a simple array slice + transfer.
*   **GPU Utilization**: Should increase from ~10% to >50% during BC-heavy phases.
*   **Memory**: Adds ~3.8GB RAM usage (manageable). VRAM usage remains low as storage is on CPU.

## 5. Alternative (Lazy Tiling)
If 3.8GB RAM is too much (unlikely), we can stick to index-based storage but optimize `_encode_observations_vectorized` to use the compressed encoder + tile on-the-fly.
*   Current: `encode -> (27, 64, 64) -> transfer` (High bandwidth)
*   Optimized Lazy: `encode_compressed -> (27, 64) -> transfer -> tile` (Low bandwidth)
*   This improves transfer speeds but still requires the gather/scatter logic. **Pre-encoding is strictly superior** for training speed as it removes the encoding compute step entirely from the critical path.
