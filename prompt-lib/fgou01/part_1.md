# Part 1: Zero-Asset Handling

## Status: FAIL

## Data Flow Summary

The Foxglove historical data pipeline flows as follows:

1. **HistoricalMarket init** (`mcts_trader/data/history.py:16-141`)
   - `_load_top_assets(64)` selects top 64 assets by total data count across ALL time (line 238-265)
   - `_preload_to_ram()` creates `cache_data` of shape `(T_total, 64, 5)` initialized to zeros (line 334)
   - Forward-fill (lines 372-382) fills gaps only from the first non-zero row per column
   - Assets that did not exist at early timestamps remain all-zeros in those rows

2. **`get_window(start_ts, length)`** (`history.py:574-616`)
   - Returns `cache_data[indices, :, 3].copy()` -- shape `(length, 64)` (line 616)
   - Includes zero columns for assets that had no data in that timestamp range
   - No zero-column detection or cyclic padding

3. **`_generate_prices_historical()`** (`train_easy.py:1190-1242`)
   - Calls `get_window()` which returns `(489, 64)`
   - Safety net at line 1235: `if window.shape[1] < self.config.n_assets` -- **NEVER fires** because `cache_data` always has 64 columns regardless of how many contain actual data
   - Zero columns pass through to the oracle and encoder

4. **Downstream damage:**
   - `log_returns = jnp.log(prices[1:] / jnp.maximum(prices[:-1], 1e-8))` (line 641)
     - For zero-column assets: `log(0 / 1e-8) = log(0) = -inf`
     - Or with forward-filled zeros: `log(0/0)` via `maximum(0, 1e-8)` = `log(0/1e-8) = log(0) = -inf`
   - Encoder `_compute_all_channels_temporal` (jax_encoder.py:300-437):
     - `price_range = price_max - price_min` = 0 for zero columns (line 329-330)
     - Guard `price_range = where(range < 1e-8, 1.0, range)` (line 330) prevents div-by-zero
     - But `norm_prices = (0 - 0) / 1.0 = 0` everywhere -- model sees a "flat at zero" asset
     - `returns_1m = (0 - 0) / (0 + 1e-8) = 0` -- no signal
   - Oracle treats zero-column assets as having 0% returns and never selects them -- wasted action space

## Bugs Found

### BUG-1.1: `get_window()` returns zero columns for non-existent assets (CRITICAL)

**File:** `mcts_trader/data/history.py:616`
**Severity:** HIGH

`get_window()` returns raw `cache_data` slices without checking for zero columns. When a window is sampled from an early period (e.g., 2018) where only 30 of the 64 selected assets existed, up to 34 columns will be all-zeros. The training schedule confirms: the first period (2018-10) has only 61 available assets, and many of those 64 top-by-count assets didn't exist until later.

The zero columns propagate to:
- Log returns computation (produces `-inf` or `NaN`)
- Encoder (produces degenerate "flat at zero" features)
- Oracle (wastes action slots on zero-return assets)
- Policy targets (model learns to predict actions for non-existent assets)

### BUG-1.2: Tile safety net in `_generate_prices_historical()` is dead code (MEDIUM)

**File:** `train_easy.py:1235-1237`
**Severity:** MEDIUM

```python
if window.shape[1] < self.config.n_assets:
    reps = (self.config.n_assets // window.shape[1]) + 1
    window = np.tile(window, (1, reps))[:, :self.config.n_assets]
```

This check tests `window.shape[1] < 64`, but `get_window()` always returns `(length, 64)` because the cache matrix is pre-allocated to `n_assets=64` columns. The condition is never True. The comment says "dead code ... safety net for partial loads" but it is also dead for the actual zero-column problem. The real danger is not fewer columns -- it is 64 columns where some are zeros.

### BUG-1.3: `_try_load_cache()` does not store/restore asset availability mask (LOW)

**File:** `mcts_trader/data/history.py:143-179`
**Severity:** LOW

The npz cache stores `cache_data` and `asset_ids` but not per-timestamp asset availability. After loading from cache, there is no way to know which assets have real data at which timestamps without scanning for zero columns. This means any downstream fix must scan the data itself rather than relying on metadata.

### BUG-1.4: `benchmark_easy.py` correctly handles underflow but not zero-within-64 (LOW)

**File:** `tools/benchmark_easy.py:197-224`
**Severity:** LOW (benchmarks only, not training)

`load_benchmark_prices()` has cyclic duplication for `actual_assets < n_assets` (line 198-201) and forward/backward fill for zero prices (lines 212-220). However, when the schedule provides 64+ asset symbols but some have no data in that period, the cache matrix will have 64 columns with zeros. The forward/backward fill handles within-column gaps but acknowledges on line 224 that "entire-column gaps" can still exist, only printing a warning without remediation.

## Answers to Audit Questions

### Q1: Where should zero-column detection + cyclic duplication happen?

**Both `get_window()` and `_generate_prices_historical()`.** The defensive fix should be in `_generate_prices_historical()` because that is the consumer that knows the invariant (64 non-zero asset columns). However, `get_window()` is the better long-term location because it is used by both training and benchmarks. A pragmatic approach:

- **`get_window()`**: Add an optional `pad_zeros=True` parameter that detects zero columns and cyclically replaces them with live columns. This fixes all callers at once.
- **`_generate_prices_historical()`**: Replace the dead `shape[1] < n_assets` check with a zero-column detection + cyclic duplication that catches the actual problem. This is the immediate fix.

### Q2: Should the npz cache store an asset availability mask?

**Yes, strongly recommended.** A per-timestamp availability mask (or at minimum, per-asset "first valid timestamp" array) would enable:
- Fast zero-column detection without scanning data
- Intelligent period selection (avoid periods with too few live assets)
- Weighted oracle that knows which assets are real vs duplicated

Minimal addition to `_save_cache`:
```python
# Per-asset first valid timestamp (n_assets,)
first_valid = np.zeros(n_cols, dtype=np.int64)
for c in range(n_cols):
    nonzero = np.nonzero(self.cache_data[:, c, 3])[0]
    first_valid[c] = self.valid_timestamps[nonzero[0]] if len(nonzero) > 0 else 0
save_dict['asset_first_valid_ts'] = first_valid
```

### Q3: How are the 64 chosen? What happens when an asset didn't exist in 2018?

**Selection:** `_load_top_assets(64)` (history.py:238-265) queries `SELECT a.id, a.symbol, COUNT(o.timestamp) as cnt FROM assets a JOIN ohlcv o ... ORDER BY cnt DESC LIMIT 84` and picks the top 64 non-delisted assets. This means assets are ranked by total data count across all time -- favoring assets like BTC/USD and ETH/USD that have data from 2014+.

**What happens for early periods:** Assets that didn't exist in 2018 (e.g., SOL/USD listed 2020, AVAX/USD listed 2020) will have zero columns in the cache matrix for all 2018 timestamps. The forward-fill only fills from the first non-zero row, so there is nothing to fill from. These zero columns pass through `get_window()` unchanged.

The training schedule confirms the issue: the first period (Oct 2018 - Apr 2019) has only 61 available assets, yet the model always expects 64. Even worse, the full-DB loading mode (`skip_validation=True`, no date filter) creates a matrix with 64 asset columns, some of which are zero for the first ~2 years of data.

## Recommended Fixes

### Fix 1: Zero-column cyclic padding in `_generate_prices_historical()` (immediate)

Replace the dead safety net in `train_easy.py:1233-1237`:

```python
# OLD (dead code -- shape[1] is always 64):
if window.shape[1] < self.config.n_assets:
    reps = (self.config.n_assets // window.shape[1]) + 1
    window = np.tile(window, (1, reps))[:, :self.config.n_assets]

# NEW: Detect zero columns and cyclically replace with live ones
live_mask = np.any(window > 0, axis=0)  # (n_assets,) bool
n_live = int(np.sum(live_mask))
if n_live < window.shape[1] and n_live > 0:
    live_indices = np.where(live_mask)[0]
    pad_indices = np.array([live_indices[i % n_live] for i in range(window.shape[1])])
    window = window[:, pad_indices]
elif n_live == 0:
    logger.warning("get_window returned all-zero prices, skipping episode")
    continue  # skip this episode in the for loop
```

### Fix 2: Zero-column handling in `get_window()` (robust, long-term)

Add to `mcts_trader/data/history.py:get_window()` before the return:

```python
# Detect and replace zero columns with cyclic duplication
result = self.cache_data[indices, :, 3].copy()
live_mask = np.any(result > 0, axis=0)
n_live = int(np.sum(live_mask))
if n_live < result.shape[1] and n_live > 0:
    live_indices = np.where(live_mask)[0]
    pad_indices = np.array([live_indices[i % n_live] for i in range(result.shape[1])])
    result = result[:, pad_indices]
return result
```

### Fix 3: Add asset availability metadata to cache (enhancement)

In `_preload_to_ram()`, after forward-fill, compute and store per-asset first-valid timestamp:

```python
self.asset_first_valid_ts = np.zeros(n_cols, dtype=np.int64)
for c in range(n_cols):
    nonzero_rows = np.nonzero(self.cache_data[:, c, 3])[0]
    if len(nonzero_rows) > 0:
        self.asset_first_valid_ts[c] = self.valid_timestamps[nonzero_rows[0]]
```

Include in `_save_cache()` / `_try_load_cache()`.

### Fix 4: Fix `benchmark_easy.py` entire-column gap handling

Replace the warning on line 224 with cyclic duplication:

```python
still_zero = np.sum(prices <= 0)
if still_zero > 0:
    # Entire columns are zero -- cyclic duplicate from live columns
    live_cols = np.any(prices > 0, axis=0)
    n_live = int(np.sum(live_cols))
    if n_live > 0:
        live_idx = np.where(live_cols)[0]
        pad_idx = np.array([live_idx[i % n_live] for i in range(prices.shape[1])])
        prices = prices[:, pad_idx]
        symbols = [symbols[i] for i in pad_idx]
        print(f"  Fixed: cyclic-duplicated {n_live} live assets to fill {prices.shape[1]} slots")
    else:
        raise ValueError(f"All {prices.shape[1]} asset columns are zero -- no valid data in period")
```
