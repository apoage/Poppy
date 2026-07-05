# Part 6: Price Window Shape & Content Validation

## Status: NEEDS_FIX

## Findings

### 1. Config values confirm T = 489

- `easy_config.py:102` — `window_size = 64`
- `easy_config.py:102` — `episode_length = 400`
- `easy_config.py:66` — `oracle_lookahead = 20`
- Formula: `400 + 64 + 20 + 5 = 489`

### 2. `_synth_n_steps` is correctly set to 489

- `train_easy.py:547` — `self._synth_n_steps = config.episode_length + config.window_size + config.oracle_lookahead + 5` = 489
- `train_easy.py:580` — SyntheticMarket also uses the same formula: `n_steps=self.config.episode_length + self.config.window_size + self.config.oracle_lookahead + 5`
- `train_easy.py:1198` — Historical path uses `seq_len = self._synth_n_steps  # 489`
- `train_easy.py:1300` — GPU synthetic path uses `self._synth_n_steps` directly

All generation paths produce `(489, 64)` price windows. PASS.

### 3. `get_window()` shape guarantee

- `history.py:574-616` — `get_window(start_ts, length)` returns `(length, n_assets)` close prices.
- Called with `length = seq_len = 489`, so returns `(489, n_assets)`.
- `n_assets` = number of columns in `cache_data`, which equals `len(self.asset_ids)`.
- `cache_data` shape is `(n_rows, n_cols, 5)` where `n_cols = len(self.asset_ids)`.
- The extraction `cache_data[indices, :, 3]` returns `(489, n_cols)`.
- `train_easy.py:1235-1237` has a safety net: if `window.shape[1] < self.config.n_assets`, it tiles cyclically to pad to 64 columns. This handles the case where the HistoricalMarket was loaded with fewer than 64 assets.

Shape guarantee: PASS. The window is always `(489, 64)` after the padding safety net.

### 4. Forward-fill via `searchsorted(side='right') - 1` gap handling

- `history.py:602-603`:
  ```python
  indices = np.searchsorted(self.valid_timestamps, target_ts, side='right') - 1
  indices = np.clip(indices, 0, len(self.valid_timestamps) - 1)
  ```

**Analysis of edge case: target timestamp before first valid timestamp.**

- `target_ts[i]` is built as `snapped_start + i * 60`.
- `snapped_start` is obtained by snapping `start_ts` to the nearest valid timestamp via `searchsorted(side='left')` (line 594), clamped to `len(valid_timestamps) - 1`.
- So `snapped_start = valid_timestamps[snap_idx]` where `snap_idx >= 0`.
- For `i=0`: `target_ts[0] = snapped_start = valid_timestamps[snap_idx]`.
- `searchsorted(valid_timestamps, valid_timestamps[snap_idx], side='right')` returns `snap_idx + 1` (or more if duplicates).
- So `index = snap_idx + 1 - 1 = snap_idx`. Correct.

**Edge case: large gaps where target timestamps land before valid_timestamps[0].**
This cannot happen because `snapped_start >= valid_timestamps[0]` by construction. The first target timestamp is always `>= valid_timestamps[0]`, so `searchsorted(side='right') - 1 >= 0`. The `np.clip(indices, 0, ...)` is a safety fallback that would fire only if something went wrong with the timestamp arithmetic.

**Edge case: target timestamp lands in a gap.** `searchsorted(side='right') - 1` correctly returns the index of the most recent valid timestamp at-or-before the target. This is correct forward-fill semantics.

Forward-fill logic: PASS (with one caveat, see BUG-6.1 below).

### 5. All-zero columns at the start of `cache_data`

- `history.py:368-382` — Forward-fill during `_preload_to_ram()`:
  ```python
  for c_idx in range(n_cols):
      last_valid = None
      for r_idx in range(n_rows):
          if self.cache_data[r_idx, c_idx, 3] > 0:  # Has close price
              last_valid = self.cache_data[r_idx, c_idx].copy()
          elif last_valid is not None:
              self.cache_data[r_idx, c_idx] = last_valid
  ```
  If an asset has NO data before the start of a window, `last_valid` stays `None`, and those rows remain all-zeros. The forward-fill only works forward from the first non-zero entry.

- **Impact on `get_window()`**: If `get_window()` pulls a window that starts before an asset has any data, that column will be all-zeros for those leading timesteps. Since `get_window()` extracts `cache_data[indices, :, 3]`, those entries will be `0.0`.

- **Downstream impact on encoder**: The encoder computes `log_returns = log(prices[1:] / max(prices[:-1], 1e-8))`. If `prices[t, asset]` transitions from `0.0` to a positive value (say `100.0`): `log(100.0 / 1e-8)` = `log(1e10)` = `23.0`. This is an extreme spike. The encoder clips returns to `[-10, 10]`, but `23.0 > 10.0` so it gets clipped. More importantly, the price normalization `(prices - price_min) / price_range` would have `price_min = 0.0` and `price_range = 100.0`, making the normalized series distorted.

### 6. Historical prices are in raw USD (not normalized)

- `history.py:363` — `cache_data[r_idx, c_idx] = [o, h, l, c, v]` stores raw OHLCV from the database.
- `history.py:616` — `get_window()` returns `cache_data[indices, :, 3].copy()` = raw close prices.
- The encoder normalizes internally via `(prices - price_min) / price_range` (per-asset min-max within the window).
- Synthetic prices also start at raw values (e.g., `100.0 * np.exp(cumsum(returns))`).
- Bella prices are converted to raw USD via `100.0 * jnp.exp(log_prices)` at `setup.py:674`.

Raw USD prices: PASS. All sources provide raw prices; the encoder normalizes internally.

## Bugs Found

- **BUG-6.1**: All-zero columns for late-listing assets in historical windows. Severity: **LOW-MEDIUM**. File: `history.py:368-382`.

  When `_preload_to_ram()` forward-fills, assets with no data before row 0 of the cache retain all-zeros until their first real candle. If `get_window()` returns a window that overlaps this leading-zero region for a given asset, that column will have zeros transitioning to real prices, causing:
  1. An extreme return spike (`log(price / 1e-8)`) at the transition point.
  2. Distorted min-max normalization (min=0 instead of actual price floor).

  However, this is partially mitigated because:
  - The `_build_valid_timestamps_for_assets()` with `min_required_assets=10` (default) only keeps timestamps where at least 10 of the 64 assets have data. This prevents windows from being placed in completely sparse regions.
  - The `reset()` method avoids the first 30 days of data (buffer), giving most assets time to have at least one candle.
  - Late-listing assets (e.g., new coins added mid-dataset) will still have leading zeros in their columns until their first trade.

  **Not a training-breaking bug** but does inject occasional noise. The encoder's return clipping (`[-10, 10]`) limits the damage, and the 0-to-price transition only happens once per affected asset per window.

- **BUG-6.2**: Bella `seq_length` may not match `_synth_n_steps`. Severity: **MEDIUM**. File: `mcts_trader/training/setup.py:598,651`.

  Bella's `GeneratorConfig.seq_length` defaults to `400` (or is loaded from the Bella checkpoint config). The `generate_bella_prices_on_device()` function returns `(n_envs, seq_length, n_assets)` where `seq_length` is the Bella model's native output length. If `seq_length != 489`, the price window will be the wrong shape.

  In `_refill_buffer()` at `train_easy.py:1282-1293`, Bella prices are appended to `prices_list` without a length check. The `_generate_batch_episodes()` pads to `max_len`, but if Bella's `seq_length` is `400` instead of `489`, the episode will be too short: `max_len = 400`, then `n_valid = (400 - 1) - 64 = 335`, meaning only 335 windows instead of 424. The oracle also gets only 335 timesteps of lookahead instead of the full range.

  **This is a shape mismatch if Bella is used.** In practice, the code handles it gracefully (no crash, just shorter episodes), but it means Bella episodes have less data than synthetic/historical ones, and the oracle has reduced lookahead near the end.

## Recommended Fixes

### Fix for BUG-6.1: Back-fill leading zeros

Add a backward-fill pass after the forward-fill in `_preload_to_ram()` to handle assets with no data at the start:

```python
# history.py, after the forward-fill loop (line 382):

# --- Backward-fill leading gaps ---
# Assets listed late in the dataset have zeros at the start.
# Fill them with the FIRST known price (backward-fill).
back_fills = 0
for c_idx in range(n_cols):
    first_valid = None
    for r_idx in range(n_rows):
        if self.cache_data[r_idx, c_idx, 3] > 0:
            first_valid = self.cache_data[r_idx, c_idx].copy()
            break
    if first_valid is not None:
        for r_idx in range(n_rows):
            if self.cache_data[r_idx, c_idx, 3] > 0:
                break
            self.cache_data[r_idx, c_idx] = first_valid
            back_fills += 1
logging.info(f"Filled {back_fills:,} leading price gaps via backward-fill")
```

This ensures no column ever has zeros (assuming the asset has at least one candle in the entire dataset). Assets with zero candles would still be all-zero, but those should not be in the top-64.

### Fix for BUG-6.2: Validate Bella output length

In `train_easy.py`, after Bella generation, truncate or error if shape mismatches:

```python
# train_easy.py, after line 1292 (Bella path in _refill_buffer):
for i in range(n_to_refill):
    p = np.array(prices_batch[i])
    if p.shape[0] < self._synth_n_steps:
        # Pad with last price (constant extrapolation)
        pad_len = self._synth_n_steps - p.shape[0]
        pad = np.tile(p[-1:], (pad_len, 1))
        p = np.concatenate([p, pad], axis=0)
    elif p.shape[0] > self._synth_n_steps:
        p = p[:self._synth_n_steps]
    prices_list.append(p)
    levels.append(-1)
```

This ensures all price windows fed to the pipeline are exactly `(489, 64)` regardless of the Bella model's native sequence length.
