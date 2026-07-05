# Part 1: Zero-Asset Handling (CRITICAL)

## Status: FAIL

## Findings
- **`history.py:616`**: `get_window()` returns `self.cache_data[indices, :, 3]` which always returns all `n_assets` columns (64), including zero columns for assets that didn't exist at that timestamp. No zero-detection or cyclic duplication happens here.
- **`history.py:_preload_to_ram()` (lines 319-383)**: Loads data from DB into RAM matrix. Does forward-fill for gaps but does NOT perform cyclic duplication for partial asset coverage. The RAM cache always has `(n_rows, n_assets, 5)` shape where all assets are represented - zeros for missing data.
- **`train_easy.py:1233-1237`**: Contains "safety net" cyclic tile code but it's DEAD CODE. The check `if window.shape[1] < self.config.n_assets:` will NEVER be true because `get_window()` always returns exactly `n_assets` columns.
- **`setup.py:408-413`**: Reference implementation correctly does cyclic padding when `prices.shape[1] < N_ASSETS`, but this logic was NOT ported to Foxglove.

## Bugs Found

### BUG-1.1: Zero columns in historical data
**Severity**: CRITICAL
**Location**: `mcts_trader/data/history.py:574-616`
**Description**: `get_window()` returns a matrix with all 64 asset columns. When sampling from early time periods (pre-2020), many assets didn't exist yet. These columns are filled with zeros during RAM preloading (`_preload_to_ram()` line 368 initializes matrix with zeros). The forward-fill logic only fills gaps AFTER the first valid data point, so entire columns can remain zero if an asset had no data at the start of the period.

**Impact**: The model learns to predict HOLD (action 1) on zero-valued assets, which inflates training agreement since HOLD is the correct action when there's no meaningful price movement. This explains the 100% → 65% train/validation drop.

### BUG-1.2: Dead code tile logic
**Severity**: HIGH
**Location**: `train_easy.py:1233-1237`
**Description**: The cyclic tile safety net will never execute because `get_window()` always returns exactly `n_assets` columns.

### BUG-1.3: No asset availability mask
**Severity**: HIGH
**Location**: `mcts_trader/data/history.py`
**Description**: The HistoricalMarket class does not track or expose which assets were actually available at each timestamp. Without this information, downstream code cannot intelligently handle zero columns.

## Recommended Fixes

1. **Add zero-column detection in `get_window()`**:
```python
def get_window(self, start_ts: int, length: int) -> np.ndarray:
    # ... existing code ...

    prices = self.cache_data[indices, :, 3].copy()  # (length, n_assets)

    # Detect zero columns (entire column is all-zeros or near-zeros)
    zero_mask = np.all(prices < 1e-10, axis=0)  # (n_assets,)
    n_zero = np.sum(zero_mask)

    if n_zero > 0:
        logging.warning(f"get_window: {n_zero}/{self.n_assets} zero columns detected, applying cyclic duplication")

        # Get indices of non-zero columns
        nonzero_indices = np.where(~zero_mask)[0]
        n_nonzero = len(nonzero_indices)

        # Cyclically duplicate non-zero assets to fill all n_assets
        pad_indices = [nonzero_indices[i % n_nonzero] for i in range(self.n_assets)]
        prices = prices[:, pad_indices]

    return prices
```

2. **Remove dead code** at `train_easy.py:1233-1237` since fix #1 above handles the issue at the source.

3. **Add asset availability tracking** to HistoricalMarket:
```python
# In _preload_to_ram(), track availability per timestamp
self.asset_availability = np.zeros((n_rows, n_assets), dtype=bool)
for r_idx in range(n_rows):
    for c_idx in range(n_cols):
        self.asset_availability[r_idx, c_idx] = (self.cache_data[r_idx, c_idx, 3] > 0)

# Expose this data
def get_asset_availability(self, start_ts: int, length: int) -> np.ndarray:
    # Similar to get_window() but returns availability mask
    ...
```

4. **Ensure npz cache stores availability mask** so zero-detection works even when loading from cache.
