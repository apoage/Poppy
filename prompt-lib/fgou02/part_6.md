# Part 6: Price Window Shape & Content Validation

## Status: PASS (with minor edge case concern)

## Findings

### Window Size Verification
- **`easy_config.py:101-102`**: `window_size: int = 64` and `episode_length: int = 400`
- **`train_easy.py:1198`**: `seq_len = self._synth_n_steps`
- **`train_easy.py:417`**: `self._synth_n_steps = episode_length + window_size + oracle_lookahead + 5`
  - With default config: 400 + 64 + 20 + 5 = **489** ✓
- **`train_easy.py:1231`**: `window = market.get_window(start_ts, seq_len)` → `get_window()` receives length=489

### Shape Validation
- **`history.py:587`**: `get_window()` returns `(length, n_assets)` where `length` is the requested seq_len.
- **`history.py:588-591`**: Fallback returns `np.zeros((length, n_a), dtype=np.float32)` - correct shape.

### Forward-Fill Behavior
- **`history.py:594-603`**: Uses `searchsorted` with forward-fill logic:
  ```python
  snap_idx = np.searchsorted(self.valid_timestamps, start_ts, side='left')
  target_ts = snapped_start + np.arange(length, dtype=np.int64) * 60
  indices = np.searchsorted(self.valid_timestamps, target_ts, side='right') - 1
  indices = np.clip(indices, 0, len(self.valid_timestamps) - 1)
  ```
  This correctly maps each 1-minute step to the most recent valid timestamp.
- **`history.py:606-614`**: Logs gap statistics on first call, providing visibility into data quality.

### Edge Cases
- **`history.py:588-591`**: Handles edge case where `cache_data is None or len(valid_timestamps) == 0` by returning zeros.
- **`history.py:616`**: Returns `self.cache_data[indices, :, 3].copy()` - includes `.copy()` to prevent modification of RAM cache.

### Price Units
- **`history.py:363`**: `self.cache_data[r_idx, c_idx] = [o, h, l, c, v]` - stores raw close prices from DB
- **`history.py:616`**: Returns channel 3 (close prices) directly from cache - raw USD values, no normalization.

## Bugs Found

None found in the window generation logic itself. However, see Part 1 findings regarding zero columns that may be present in these windows.

## Edge Case Concern

### Potential Issue at Data Boundaries
**`history.py:594-595`**: The snapping logic may not handle the very start of the data correctly:
```python
snap_idx = np.searchsorted(self.valid_timestamps, start_ts, side='left')
snap_idx = min(snap_idx, len(self.valid_timestamps) - 1)
```

If `start_ts` is before the first valid timestamp, `snap_idx = 0`, which is correct.
If `start_ts` is after the last valid timestamp minus `length*60`, the window may not have enough data.

However, `_generate_prices_historical()` at lines 1211-1216 has bounds checking:
```python
max_start_idx = range_max - seq_len
if max_start_idx <= range_min:
    raise ValueError(
        f"Historical period too short: {range_max - range_min} timestamps "
        f"< {seq_len} required"
    )
```

This ensures there's enough data before calling `get_window()`.

## Recommended Fixes

None required for window shape/content. The core logic is correct.

### Note on Zero Columns
While the window shape and content generation logic is correct, the issue of zero columns (covered in Part 1) means that some price windows will have all-zero columns. These need to be fixed via cyclic duplication as described in Part 1's recommended fixes.

### Optional: Add explicit shape validation
Add a check in `get_window()` to ensure output shape matches expectations:
```python
result = self.cache_data[indices, :, 3].copy()
assert result.shape == (length, self.n_assets), \
    f"get_window returned unexpected shape {result.shape}, expected ({length}, {self.n_assets})"
return result
```
