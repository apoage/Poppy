# TASK_136b: Batch Encode Across All Episodes (Investigation)

## 1. Current Implementation Analysis

The function `_encode_observations_vectorized` in `expert_buffer.py` iterates over unique `episode_ids` in the batch. For each episode:
1.  Retrieves `episode_prices` from `self._episode_prices` (dict lookup).
2.  Filters indices belonging to this episode.
3.  Calculates `window_starts`.
4.  Calls `encoder_fn` (JIT-compiled).
5.  Stores result in `observations` array.

**Inefficiencies:**
*   **Python Loop:** Iterates N times (where N is number of unique episodes in batch).
*   **Small Transfers:** Transfers `close_prices` (T, 64) for each episode individually.
*   **Synchronization:** Blocking transfers back to host inside the loop.

## 2. Proposed Single-Call Approach

The idea is to construct a single batch of inputs and call JAX once.

**Target Function:**
`mcts_trader/env/jax_encoder.py:jit_batch_encode_simple(prices, window_starts, config, per_env=True)`

**Signature:**
*   `prices`: `(batch_size, n_steps, n_assets)` (if `per_env=True`)
    *   Wait, this signature assumes *one price array per sample*. That's extremely wasteful (copying `(400, 64)` prices for *every* sample).
    *   Actually, `per_env=True` vmaps over the first dimension: `lambda p, ws: encode(p, ws)`.
    *   So if we have `batch_size` samples, we'd need to provide `batch_size` price arrays.

**The "Gather" Problem:**
We have `batch_size` samples (e.g., 1024). They come from ~20-50 unique episodes.
To use `per_env=True` directly, we would need to construct a `prices` tensor of shape `(1024, Max_T, 64)`.
*   Memory: 1024 * 400 * 64 * 4 bytes ≈ 100 MB. (Feasible per batch).
*   Construction: We'd need to copy the episode price data 1024 times in Python/Numpy. This **copying cost** might outweigh the JAX loop savings.

**Better Approach: Indexing inside JAX?**
If we could pass:
1.  `all_unique_prices`: `(n_unique_eps, Max_T, 64)`
2.  `sample_episode_indices`: `(batch_size,)` -> pointing to which episode
3.  `sample_window_starts`: `(batch_size,)`

Then JAX could do: `encode(all_unique_prices[sample_episode_indices], sample_window_starts)`.

**Feasibility of Indexing:**
*   JAX supports `take` / indexing.
*   We'd need to pad all unique episodes to the same `Max_T`.
*   Constructing `all_unique_prices` still requires gathering from the dictionary `self._episode_prices` in Python.

## 3. Comparison: Loop vs. Batch Gathering

| Feature | Current (Python Loop) | Proposed (Batch Gather) |
| :--- | :--- | :--- |
| **Python Overhead** | High (Loop over ~50 eps) | Medium (Gather ~50 arrays + stack) |
| **Host->Device** | ~50 small transfers | 1 medium transfer (~100MB) |
| **GPU Kernel** | ~50 small kernels | 1 large kernel |
| **Device->Host** | ~50 blocking transfers | 1 blocking transfer |
| **Memory** | Low | Higher (padding + duplication) |

## 4. The "Single Vmap" Design

1.  **Identify Unique Episodes:** Get `unique_ep_ids` from the batch.
2.  **Gather Prices:**
    *   Find `max_len` among these episodes.
    *   Allocate `padded_prices` shape `(n_unique, max_len, n_assets)`.
    *   Copy data from `self._episode_prices` to `padded_prices` (Python/Numpy).
3.  **Map Samples:**
    *   Create `map_ep_id_to_index`.
    *   Create `batch_ep_indices` `(batch_size,)` mapping each sample to its row in `padded_prices`.
4.  **JAX Call:**
    *   Function: `encode_from_padded_bank(padded_prices, batch_ep_indices, window_starts, config)`
    *   Implementation:
        ```python
        def encode_from_bank(prices_bank, ep_idx, start):
            # Select relevant price array
            my_prices = prices_bank[ep_idx] # (Max_T, n_assets)
            # Encode
            return encode_observation_simple(my_prices, start, config)

        # Batch it
        batch_encode = jax.vmap(encode_from_bank)
        ```

## 5. Constraints & Edge Cases

*   **Memory Cost:** `padded_prices` for 50 episodes × 500 steps × 64 assets ≈ 6.4 MB. **Negligible.**
*   **Padding:** Need to ensure padding doesn't affect calculation (it shouldn't, as `dynamic_slice` respects `window_starts`).
*   **Variable Lengths:** If one episode is 400 steps and another is 1000, we pay for 1000 * 50. Acceptable.

## 6. Implementation Plan

**Step 1: Define New JAX Encoder**
Add to `jax_encoder.py`:
```python
@partial(jax.jit, static_argnames=['config'])
def jit_encode_from_bank(price_bank, ep_indices, window_starts, config):
    def _encode(ep_idx, start):
        return encode_observation_simple(price_bank[ep_idx], start, config)
    return jax.vmap(_encode)(ep_indices, window_starts)
```

**Step 2: Update `expert_buffer.py`**
Modify `_encode_observations_vectorized`:
1.  Find unique episodes in batch.
2.  Construct `padded_price_bank` (numpy).
3.  Construct `ep_indices` array mapping samples to bank rows.
4.  Call `jit_encode_from_bank` once.
5.  Return result.

**Comparison to Pre-Encoding (TASK_136a):**
*   **Pre-encoding (136a):** Zero encoding cost at train time. Higher RAM usage (3.8GB). Simpler training loop.
*   **Batch-Encoding (136b):** Encoding cost happens at train time (but batched). Low RAM usage. More complex buffer logic.

**Recommendation:**
Since 3.8GB RAM is available, **Pre-encoding (TASK_136a)** is superior for pure speed. It removes the compute entirely. However, **Batch-Encoding (136b)** is a robust fallback if memory becomes tight or if we want to support on-the-fly augmentation of raw prices later.

Given the goal is *maximum speed* on a 3090, **136a (Pre-encoding)** wins. But 136b is a valid optimization of the current architecture.

## 7. Benchmark Estimate
*   **Current:** ~50 iterations × (0.5ms transfer + 0.5ms kernel + 0.5ms sync) ≈ **75ms** (plus Python overhead).
*   **Proposed:** 1 iteration × (2ms transfer + 2ms kernel + 2ms sync) ≈ **6ms**.
*   **Speedup:** ~10x reduction in encoding latency.

## 8. Conclusion
Feasible and efficient. Standard JAX `vmap` combined with an indexing gather pattern works perfectly here.
