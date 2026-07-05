# TASK_152: Untiling the Encoder (Temporal-Spatial Features)

**Status:** COMPLETE
**Completed:** 2026-02-10

**Objective:** Transform the observation tensor from a redundant "tiled" 2D format into a proper 3D format (Channels, Time, Assets) to allow the Transformer model to actually learn temporal patterns.

## 1. The Root Cause (Tiled Feature Blindness)
Currently, `jax_encoder.py` uses `jnp.tile` to create a `(27, 64, 64)` tensor. 
- **The Issue:** Every row in a channel is identical. There is zero information across the "width" of the tensor.
- **The Impact:** The Transformer's attention mechanism is looking at a static wall of identical columns. It cannot see how Asset A changed over the last 60 minutes relative to its current price; it only sees the final pre-calculated 27 indicators.

## 2. Technical Implementation Fix

### A. Redefine Output Shape
Change the encoder to return a tensor of shape `(27, window_size, n_assets)`. 
- **Example:** `(27, 60, 64)`
- This gives the model 60 steps of history for all 27 features across all 64 assets.

### B. Update `_compute_all_channels` in `jax_encoder.py`
Remove all `jnp.tile` calls and preserve the time dimension.

```python
# OLD:
# output = output.at[4].set(jnp.tile(returns_1m[:, -1:], (1, n)))

# NEW:
# returns_1m is (n_assets, window_size). Just assign it.
output = output.at[4].set(returns_1m) # Now shape is (27, window_size, n_assets)
```

### C. Update Model Architecture (`muzero_model.py`)
Ensure the TRM encoder layer correctly interprets the new input shape.
- Instead of treating the input as a 64x64 image, it should treat it as 64 assets with 60 steps of 27 features each.

---

## 3. The "Ideal Path" Upgrade (Lookahead)
To complement the better features, we must give the model a more "predictable" target.

**File:** `easy_config.py`
```python
oracle_lookahead: int = 20  # Increase from 5 to 20 steps
```
- **Why:** 5 steps is micro-noise. 20 steps is a "regime." It's much easier for a 60-step history to predict a 20-step trend than a 5-step bounce.

---

## 4. Expected Impact
- **Attention Alignment:** The Transformer will finally have "rows" (time) and "columns" (assets) that mean something different. Attention can now find "Lead-Lag" relationships.
- **Agreement:** Once the model can see the *movement* over 60 steps, it can correlate it with the *trend* over 20 steps. You should see `agree_trade` finally break above 5%.
- **VRAM:** This change actually **reduces** memory usage per observation since `60 < 64`.

---

**Next Step:** This is a major change to the "Eyes" of your agent. After implementation, the model will likely need a few hundred epochs to re-learn its representations, but they will be high-fidelity representations this time.

---

## Implementation Notes (2026-02-10)

### Changes Made

**1. easy_config.py**
- `window_size: 64` (was 60) - matches n_assets for model compatibility
- `oracle_lookahead: 20` (was 5) - longer horizon for regime prediction

**2. jax_encoder.py**
- Added `_compute_all_channels_temporal()` - computes real temporal features
- Added `encode_observation_temporal()` - main temporal encoder
- Added `jit_batch_encode_temporal()` - batched version
- Added `jit_vmap_encode_temporal()` - parallel batch version

**3. train_easy.py**
- Replaced `jit_batch_encode_compressed` with `jit_batch_encode_temporal`
- Replaced `jit_vmap_encode_compressed` with `jit_vmap_encode_temporal`
- Removed tiling step in `enable_latent_cache`

**4. easy_trainer.py**
- Removed `jit_batch_tile_compressed` calls in `train_step` and `evaluate_batch`
- Observations now passed directly to model (already full format)

### New Tensor Layout

| Dimension | Old (Tiled) | New (Temporal) |
|-----------|-------------|----------------|
| Shape | (27, 64, 64) | (27, 64, 64) |
| Rows | 64 copies of same data | 64 time steps |
| Cols | 64 assets | 64 assets |
| Ch 0-4 | Latest OHLC + 1m return | Full time series |
| Ch 5-26 | Scalar indicators | Broadcast across time |

### What The Model Now Sees

**Before:** Static wall of identical columns - CNN convolutions couldn't learn temporal patterns
**After:** Real price dynamics over 64 time steps - CNN can learn trends, reversals, momentum

### Expected Impact
- Model must re-learn representations (expect initial performance drop)
- Once adapted, should capture lead-lag relationships
- `agree_trade` should climb above 5% as model learns temporal patterns
