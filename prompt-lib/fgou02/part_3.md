# Part 3: AR (Autoregressive) Blend with Historical Data

## Status: PASS (with minor concerns)

## Findings
- **`train_easy.py:1274-1276`**: When `historical_market` is set, `_refill_buffer()` correctly calls `_generate_prices_historical(n_to_refill)`. Historical data is used as the primary source.
- **`train_easy.py:1231`**: Historical price windows are obtained via `market.get_window(start_ts, seq_len)` which returns `(T, n_assets)` price data.
- **`train_easy.py:948-962`**: AR blend is correctly computed based on `ar_warmup_epochs`, `ar_blend_epochs`, and `ar_max_blend`. The blend formula correctly ramps from 0 to `ar_max_blend`.
- **`train_easy.py:966-971`**: When `ar_blend > 0` and model params are available, AR episodes are selected via coin flip (`rng.random() < ar_blend`).
- **`train_easy.py:976-1026`**: Batched AR rollout correctly receives `all_observations` which comes from `jit_batch_encode_temporal(prices, ...)` where `prices` are the historical price windows. The AR rollout uses the SAME historical prices that were used for oracle computation.
- **`e-dojo_v1.py:194`**: `ar_max_blend` is stored in dojo state and can be adjusted adaptively.
- **`e-dojo_v1.py:219`**: `ar_max_blend` is passed to branch launch config.
- **`e-dojo_v1.py:994`**: When launching branches, `ar_cap = self._dojo_state.ar_max_blend` is used to override config.

## Bugs Found

None found. The AR blend logic with historical data appears correct.

## Minor Concerns

1. **AR cap timing**: The comment at line 799 in the audit prompt mentions "Known bug: it was loaded after, so first launch used stale value." However, the current code shows `self.ar_max_blend` being initialized at line 194 BEFORE any branch launch, so this bug appears to be fixed.

2. **AR blend for historical data only**: The AR blend logic correctly applies to historical data, but there's no explicit verification that historical data is always used when AR is active. The source selection in `_refill_buffer()` at line 1274 checks `if self.historical_market is not None` first, which is correct.

3. **Model-state price alignment**: The AR rollout (`_build_model_state_trajectory_batched`) uses `market_obs` which is pre-encoded observations. These come from `jit_batch_encode_temporal(prices, window_starts, encoder_config)` where `prices` are the historical windows. The alignment appears correct because both oracle and AR use the same price data.

## Verified Correct

1. **Historical price source for AR**: AR rollout receives the same historical price windows that were used for oracle computation. The data flow is:
   - `_generate_prices_historical()` calls `market.get_window()` → returns historical prices
   - These prices are passed to `_generate_batch_episodes()` which encodes them
   - AR episodes use the pre-encoded observations from the same prices

2. **AR blend scheduling**: The blend correctly transitions from oracle (0% AR) through a ramp period to `ar_max_blend`. The formula at line 960 correctly computes the blended value:
   ```python
   ar_blend = min((self._current_epoch - ar_warmup) / ar_blend_epochs, ar_max_blend)
   ```

3. **AR max blend passing through orchestrator**: The orchestrator correctly reads `ar_max_blend` from dojo state and passes it to branch launch config via `ar_cap` variable (line 994).

## Recommended Fixes

None required. The AR blend implementation with historical data is correct.

Optional enhancement: Add explicit logging to confirm AR episodes are using historical prices when historical_market is set.
