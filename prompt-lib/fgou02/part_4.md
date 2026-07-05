# Part 4: Benchmark Period Exclusion

## Status: PASS (with edge case note)

## Findings
- **`train_easy.py:1165-1181`**: `_load_benchmark_periods()` correctly loads `config/benchmark_periods.json` and parses dates with UTC timezone using `datetime.strptime(...).replace(tzinfo=timezone.utc).timestamp()`.
- **`config/benchmark_periods.json`**: Contains 6 benchmark periods (7 days each = 42 days total) with correct start/end dates.
- **`train_easy.py:1183-1188`**: `_overlaps_benchmark()` correctly implements overlap detection using the standard formula: `if start_ts < bp_end and end_ts > bp_start: return True`.
- **`train_easy.py:1221-1229`**: Rejection sampling with 20 attempts is implemented. If all 20 attempts overlap, a warning is logged and the last (overlapping) sample is used.
- **Rejection rate calculation**: With 42 days excluded out of ~4380 days (12 years), the theoretical rejection rate is ~0.96%, which is well within the 20-attempt tolerance.

## Bugs Found

None found. The benchmark period exclusion logic is correct.

## Edge Case Note

**`train_easy.py:1222-1229`**: The rejection sampling loop uses `else` clause after `for attempt in range(20)`, which means "if the loop completed without breaking" (i.e., all 20 attempts overlapped). In this case, a warning is logged and the code continues with the LAST attempted window (which overlaps a benchmark period).

```python
for _ in range(n_episodes):
    for attempt in range(20):
        idx = self.rng.integers(range_min, max_start_idx)
        start_ts = int(market.valid_timestamps[idx])
        end_ts = start_ts + window_seconds
        if not self._overlaps_benchmark(start_ts, end_ts):
            break
    else:
        # This runs if NO break occurred (all 20 overlapped)
        logger.warning("TASK_698: All 20 rejection attempts hit benchmark, using last")
```

**Impact**: Extremely low probability (~0.96%^20 ≈ 8.2×10^-48) of all 20 attempts overlapping, so this fallback path essentially never executes in practice.

## Verification

Let's verify the overlap detection formula:
- Two intervals [a1, b1] and [a2, b2] overlap if and only if: `a1 < b2 AND b1 > a2`
- The code uses: `if start_ts < bp_end and end_ts > bp_start: return True`
- This is the correct formula.

The JSON parsing uses UTC timezone, matching HistoricalMarket's use of UTC (see `history.py:46-73`). This ensures consistent timestamp comparison.

## Recommended Fixes

None required. The benchmark period exclusion is correctly implemented.

Optional enhancement: Add a metric to track the actual rejection rate during training to confirm it matches the expected ~1%:
```python
# Add to class
self._rejection_stats = {'attempts': 0, 'overlaps': 0}

# In _generate_prices_historical()
for _ in range(n_episodes):
    for attempt in range(20):
        idx = self.rng.integers(range_min, max_start_idx)
        start_ts = int(market.valid_timestamps[idx])
        end_ts = start_ts + window_seconds
        self._rejection_stats['attempts'] += 1
        if not self._overlaps_benchmark(start_ts, end_ts):
            break
        self._rejection_stats['overlaps'] += 1
    # ... rest of code
```
