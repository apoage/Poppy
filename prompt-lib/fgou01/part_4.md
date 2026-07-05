# Part 4: Benchmark Period Exclusion

## Status: NEEDS_FIX

## Findings

### 1. Benchmark period definitions (config/benchmark_periods.json)
- 6 periods defined, all 7 days each, totaling 42 days.
- Date range spans 2022-06-10 to 2024-03-02 across bull, bear, sideways, panic, rally, and low-volatility regimes.
- Each period uses `start_date` and `end_date` in `YYYY-MM-DD` format. No ambiguity.

### 2. Date parsing (`_load_benchmark_periods`, train_easy.py:1165-1181)
- **CORRECT**: Parses `%Y-%m-%d` format, applies `timezone.utc`, converts to Unix timestamp via `int(datetime.timestamp())`.
- Returns list of `(start, end)` tuples as integer Unix timestamps (seconds).
- Gracefully returns `[]` if config file is missing (silent degradation -- acceptable since benchmark eval would also be impossible without the file).

### 3. Overlap detection (`_overlaps_benchmark`, train_easy.py:1183-1188)
- **CORRECT**: Uses standard interval overlap formula: `start_ts < bp_end and end_ts > bp_start`.
- This correctly detects any overlap: partial start overlap, partial end overlap, full containment, and full enclosure.
- `start_ts` is the sampled window start; `end_ts = start_ts + seq_len * 60` (train_easy.py:1199,1225).

### 4. Timestamp compatibility
- `valid_timestamps` in HistoricalMarket are Unix timestamps in seconds (`np.int64`), sourced from the SQLite database (mcts_trader/data/history.py:292,317).
- `_load_benchmark_periods` produces Unix timestamps in seconds via `datetime.timestamp()`.
- **COMPATIBLE**: Both use the same unit and epoch.

### 5. Rejection sampling (train_easy.py:1221-1229)
- Uses a `for/else` construct: 20 attempts to find a non-overlapping window.
- On success (`not _overlaps_benchmark`), breaks out of the loop.
- **BUG**: On failure (all 20 attempts overlap), the `else` block only logs a warning, then execution falls through to `market.get_window(start_ts, seq_len)` at line 1231 using the **last overlapping sample**. Benchmark-contaminated data leaks into training.

### 6. Rejection rate analysis
- **Full database (~4380 days)**: Each benchmark period creates an exclusion zone of ~7.34 days (7 days + 0.34 day window width). Total exclusion: 6 x 7.34 = ~44 days. Rejection probability per attempt: 44/4380 = ~1.0%. Probability all 20 attempts fail: 0.01^20 = 10^-40. **Negligible**.
- **180-day schedule period (worst case, 2 benchmarks overlap)**: Exclusion ~14.7 days / 180 days = ~8.2%. All 20 fail: 0.082^20 = ~10^-22. **Still negligible**.
- **Conclusion**: The 20-attempt limit is more than sufficient. The fallback bug is extremely unlikely to trigger in practice, but it remains a correctness defect.

### 7. Lazy initialization (train_easy.py:1201-1202)
- `_excluded_periods` is loaded once via `hasattr` check on first call to `_generate_prices_historical`.
- Acceptable pattern -- the config file is static during training runs.

### 8. Scope of exclusion
- Benchmark exclusion applies **only** to the historical data path (`_generate_prices_historical`).
- Synthetic (CPU/GPU) and Bella paths generate procedural data with no correspondence to real market dates. Exclusion is correctly not applied to those paths.

### 9. Schedule generation excludes benchmarks at period boundaries
- `tools/map_training_data.py` splits training schedule periods at benchmark boundaries (line 226-228, 263).
- This means schedule periods fed to `set_historical_period` should not overlap benchmark windows.
- Double protection: even if a schedule period boundary is close to a benchmark, the per-sample `_overlaps_benchmark` check in `_generate_prices_historical` catches any window that would extend into the benchmark.

## Bugs Found

- **BUG-4.1**: Rejection sampling fallback uses contaminated data (Severity: LOW, train_easy.py:1229). When all 20 rejection attempts overlap benchmark periods, the code logs a warning but proceeds with the last (overlapping) sample. This violates the benchmark exclusion invariant. Practically unreachable under current configurations (probability < 10^-22), but a correctness defect that should be fixed for defense-in-depth.

## Recommended Fixes

### Fix BUG-4.1: Skip episode instead of using contaminated data

Replace lines 1221-1231 in `train_easy.py`:

```python
        for _ in range(n_episodes):
            for attempt in range(20):
                idx = self.rng.integers(range_min, max_start_idx)
                start_ts = int(market.valid_timestamps[idx])
                end_ts = start_ts + window_seconds
                if not self._overlaps_benchmark(start_ts, end_ts):
                    break
            else:
                logger.warning("TASK_698: All 20 rejection attempts hit benchmark, skipping episode")
                continue  # skip this episode entirely

            window = market.get_window(start_ts, seq_len)  # (T, n_assets) rigid grid
```

Note: This means `prices_list` may have fewer than `n_episodes` entries. The caller (`_refill_buffer` at line 1275) would get a shorter list, resulting in a slightly smaller buffer refill on that iteration. The buffer refill loop (`while len(self.buffer) < self.N_BUFFER` at line 1270) will call `_generate_prices_historical` again on the next iteration to fill the gap. This is safe and self-correcting.

Alternative (if exact episode count is required): replace the contaminated sample with a synthetic fallback:

```python
            else:
                logger.warning("TASK_698: All 20 rejection attempts hit benchmark, using synthetic fallback")
                level = int(self.rng.choice(self.config.synthetic_levels))
                seed = int(self.rng.integers(0, 2**31))
                _, window = self._generate_prices_cpu(level, seed)
                prices_list.append(window)
                levels.append(level)
                continue
```
