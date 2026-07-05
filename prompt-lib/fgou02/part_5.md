# Part 5: Data Source Selection Chain

## Status: PASS

## Findings
- **`train_easy.py:1274-1276`**: `_refill_buffer()` checks `if self.historical_market is not None` FIRST. Historical data is correctly prioritized when set.
- **`train_easy.py:1278-1281`**: If historical market is NOT set but `_should_use_secondary()` returns True (for 90/10 experience replay mixing), secondary data (Bella or Synthetic) is used.
- **`train_easy.py:1282-1293`**: If historical is not set and secondary is not being used, Bella (if available) is the primary source.
- **`train_easy.py:1294-1330`**: If neither historical nor Bella are available, CPU/GPU synthetic is used as fallback.
- **`train_easy.py:1124-1131`**: `_should_use_secondary()` implements the 90/10 mixing ratio based on `self.mix_ratio` and a counter.

## Bugs Found

None found. The data source selection chain is correct.

## Source Priority Chain (when historical_market is set)

```
_refill_buffer() loop:
├─ if self.historical_market is not None:
│   └─ Use _generate_prices_historical() [HISTORICAL] ✓
├─ elif _should_use_secondary() and (secondary_bella or bella with secondary_levels):
│   └─ Use _generate_secondary_prices() [SECONDARY: Bella/Synth]
├─ elif self.bella_generator is not None:
│   └─ Use Bella generator [BELLA PRIMARY]
└─ else:
    └─ Use CPU/GPU synthetic [SYNTHETIC FALLBACK]
```

**Observation**: When `historical_market` is set, it takes 100% priority. There is NO path where historical data gets mixed with Bella or synthetic data in the same batch.

## Secondary/Primary Logic

The secondary source logic (for 90/10 experience replay) is controlled by:
- `self.mix_ratio` (e.g., 0.1 for 10% secondary)
- `self._mix_counter` which increments each refill
- Every `1/mix_ratio` refills use secondary (e.g., every 10th for 10%)

**Important**: When `historical_market` is set, this entire secondary logic is bypassed because the first check (`if self.historical_market is not None`) returns True.

## Branch B (Bella stabilizer)

For Branch B where Bella is the primary source (no historical_market), the logic works as:
1. `self.historical_market is None` → skip historical
2. Check `_should_use_secondary()` → 10% of refills use synthetic secondary
3. `self.bella_generator is not None` → 90% of refills use Bella primary
4. Else → fallback to synthetic (shouldn't happen if Bella is configured)

This is correct behavior: Bella stabilizer needs some synthetic diversity to prevent overfitting.

## Recommended Fixes

None required. The data source selection chain is correctly implemented.

Optional: Add a flag to explicitly disable historical data for testing purposes:
```python
def _refill_buffer(self, force_source=None):
    """Refill VRAM buffer. force_source can force 'historical', 'bella', or 'synthetic'."""
    if force_source == 'synthetic':
        # Force synthetic even if historical is set
        ...
    elif force_source == 'bella':
        # Force Bella even if historical is set
        ...
    elif self.historical_market is not None:
        # Normal: historical takes priority
        ...
```
