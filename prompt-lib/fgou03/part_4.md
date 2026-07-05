# Part 4: Benchmark Period Exclusion

## Status: NEEDS_FIX

## Findings
- `_load_benchmark_periods` correctly parses `YYYY-MM-DD` strings into UNIX timestamps representing the start and end of the exclusion periods.
- `_overlaps_benchmark` uses standard interval overlap logic (`start_ts < bp_end and end_ts > bp_start`), which is correct.
- However, the rejection sampling mechanism in `_generate_prices_historical` has a critical flaw. It loops 20 times to find a non-overlapping window. If all 20 attempts fail, the `else:` block of the `for` loop executes, logging a warning and proceeding to use the *last overlapping sample*.
- If a specific `_hist_period_bounds` perfectly aligns with a benchmark period (which could happen if training periods are strictly partitioned), rejection sampling will fail 20 times on *every episode*, and the benchmark data will leak entirely into the training set.

## Bugs Found
- **BUG-4.1**: Rejection sampling fallback silently accepts overlapping windows if 20 attempts fail. This can leak benchmark data, particularly if the current period bounds heavily overlap with a benchmark period.

## Recommended Fixes
- Change the fallback behavior in `_generate_prices_historical()` to raise an exception or skip generating the episode, rather than using an overlapping window:
```python
        for _ in range(n_episodes):
            valid_sample_found = False
            for attempt in range(20):
                idx = self.rng.integers(range_min, max_start_idx)
                start_ts = int(market.valid_timestamps[idx])
                end_ts = start_ts + window_seconds
                if not self._overlaps_benchmark(start_ts, end_ts):
                    valid_sample_found = True
                    break
            
            if not valid_sample_found:
                raise RuntimeError(f"Failed to find non-overlapping historical window after 20 attempts in range {range_min}-{range_max}.")
```