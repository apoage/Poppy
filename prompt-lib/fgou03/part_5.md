# Part 5: Data Source Selection Chain

## Status: PASS

## Findings
- In `_refill_buffer()` (`train_easy.py`), the source selection chain is implemented as an `if/elif` block.
- The first condition is `if self.historical_market is not None:`. If true, it calls `_generate_prices_historical(n_to_refill)` and immediately moves on.
- The `elif self._should_use_secondary()` and `elif self.bella_generator is not None:` blocks are bypassed entirely when `historical_market` is populated.
- Therefore, there is no mixing of historical data with Bella or synthetic data in the same batch.
- For Branch B (the Bella stabilizer), if `historical_market` is not set but `bella_generator` is, the `elif` chain correctly defers to Bella as the primary source before falling back to GPU synthetic data. 

## Bugs Found
- None. The `if/elif` mutual exclusion enforces strict data source isolation at the batch level.

## Recommended Fixes
- None required. Data source priorities are solid.