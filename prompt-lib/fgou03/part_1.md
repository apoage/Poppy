# Part 1: Zero-Asset Handling

## Status: FAIL

## Findings
- `HistoricalMarket._try_load_cache` (`mcts_trader/data/history.py`) simply loads the `cache_data` from the `.npz` file, which is constructed by finding the top 64 assets by total volume across the entire DB.
- If an asset from this top-64 list didn't exist in early periods (e.g., 2018), its entire column in `cache_data` is initialized to and remains `0.0`.
- `HistoricalMarket.get_window()` directly slices `self.cache_data[indices, :, 3]`, which returns all 64 columns, including the all-zero columns.
- In `train_easy.py`, `_generate_prices_historical()` has a safety net: `if window.shape[1] < self.config.n_assets:`. However, because `get_window()` *always* returns exactly 64 columns, `window.shape[1]` is always 64. The safety net never triggers, making it dead code.
- As a result, early historical periods contain up to 33 columns of all-zeros, allowing the model to easily achieve artificially high validation agreement by predicting HOLD on these assets.

## Bugs Found
- **BUG-1.1**: Dead code for cyclic padding in `train_easy.py` line ~1235. The shape check `window.shape[1] < n_assets` is always False.
- **BUG-1.2**: Lack of dynamic zero-column detection. The pipeline expects the cache to drop zero-columns, but the cache is a fixed `(T, 64)` matrix.

## Recommended Fixes
- Add dynamic zero-column detection and cyclic padding directly inside `_generate_prices_historical()` in `train_easy.py`. For each window:
```python
            window = market.get_window(start_ts, seq_len)  # (T, n_assets)
            
            # Detect active columns (any price > 0)
            active_mask = (window > 0).any(axis=0)
            n_active = active_mask.sum()
            
            if n_active < self.config.n_assets:
                if n_active == 0:
                    raise ValueError("Sampled window has zero active assets!")
                # Extract only active columns
                active_cols = window[:, active_mask]
                # Cyclically pad back to n_assets
                pad_indices = [i % n_active for i in range(self.config.n_assets)]
                window = active_cols[:, pad_indices]
```
- We don't need to store an asset availability mask in the npz cache; doing it per window dynamically is fast enough and safer.
- The 64 assets loaded are the top 64 by total volume. When an asset didn't exist in 2018, it is zero-filled. The above fix correctly handles this.