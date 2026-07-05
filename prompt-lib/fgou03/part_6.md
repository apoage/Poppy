# Part 6: Price Window Shape & Content Validation

## Status: PASS

## Findings
- `self._synth_n_steps` is computed correctly as `episode_length + window_size + oracle_lookahead + 5 = 400 + 64 + 20 + 5 = 489`.
- `get_window()` returns the exact shape requested `(489, 64)`.
- The rigid 1-minute grid mapping using `indices = np.searchsorted(..., side='right') - 1` is mathematically sound for forward-filling.
  - If a timestamp is exactly matched (e.g., the first `target_ts`), `searchsorted(side='right')` returns index `i+1`, so `- 1` correctly maps to `i`.
  - If there is a gap (missing trades), `searchsorted(side='right')` returns the index of the next available timestamp `i+1`, so `- 1` safely falls back to index `i` (the most recent valid timestamp), effectively freezing the price.
- `np.clip` safeguards against any out-of-bounds indices at the tail.
- Raw close prices are fetched from channel `3` of the OHLCV array, which is correct for the encoder.

## Bugs Found
- None. The time alignment and shape mechanics are correct.

## Recommended Fixes
- None. The logic handles gaps deterministically and aligns with the fixed model dimensionality.