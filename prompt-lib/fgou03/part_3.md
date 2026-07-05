# Part 3: AR (Autoregressive) Blend with Historical Data

## Status: PASS (with minor risk)

## Findings
- In `_refill_buffer` (`train_easy.py`), historical prices are fetched and appended to `prices_list`. These are correctly encoded and stored as observations.
- In `_generate_batch_episodes`, the batched AR rollout (`_build_model_state_trajectory_batched`) receives `market_obs_padded` and `returns_padded`. These are properly sliced from the same historical `all_observations` and `log_returns` arrays that the oracle uses. So the rollout gets the exact same historical data context.
- In `tools/orchestration/e-dojo_v1.py`, `--ar-max-blend` is dynamically passed to branch sub-processes via `cmd.extend(["--ar-max-blend", f"{ar_cap:.2f}"])`.
- The known bug regarding stale AR cap value on the first launch has been fixed. In `e-dojo_v1.py` `main()`, the `dojo_state.json` is explicitly parsed and loaded into `launcher._dojo_state` *before* `launcher.launch_single_branch` or `launcher.launch_dual_branches` is called.

## Bugs Found
- None. The AR architecture correctly abstracts away the data source. As long as historical prices are loaded into the pipeline, the rollout correctly autoregresses over them.

## Recommended Fixes
- Continue to monitor inference speed. Batched AR rollout works cleanly for historical data just as it does for synthetic data.