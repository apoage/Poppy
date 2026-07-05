# Part 3: AR Blend with Historical Data

## Status: PASS

## Findings

### 1. AR rollout correctly receives the same prices as the oracle (historical or synthetic)

In `_generate_batch_episodes()` (`train_easy.py:841`), both the oracle and the AR model-state rollout derive their inputs from the same `prices_jax` array:

- **Line 870**: `prices_jax = jnp.array(prices_padded)` -- constructed from `prices_list` (the caller's price data, regardless of source)
- **Line 873**: `log_returns` computed from `prices_jax` -- fed to oracle (line 899) and AR rollout (line 984)
- **Line 944**: `all_observations = jit_vmap_encode_temporal(prices_jax, ...)` -- market encoding fed to AR rollout (line 1012)

When `_refill_buffer()` takes the historical path (`train_easy.py:1274`), `prices_list` contains historical prices from `_generate_prices_historical()`. These flow through `_generate_batch_episodes()` identically to synthetic/Bella prices. The AR model-state rollout receives the exact same historical price window that the oracle used. **No data mismatch.**

### 2. `_build_model_state_trajectory_batched()` receives correct inputs (`train_easy.py:315`)

The function takes `market_obs` (pre-encoded 27-channel market observations) and `returns` (log returns). Both are sliced from the same `prices_jax` that the oracle used:
- `market_obs_padded[b] = all_observations[idx]` (line 1012) -- encoded from `prices_jax` at line 944
- `returns_padded[b] = returns_list[b]` (line 1013) -- sliced from `log_returns` computed at line 873

The AR scan at line 383 (`market_t = market_obs[:, idx]`) correctly reads the historical market channels at each timestep. The returns used for PnL computation at line 410 (`ret = returns[:, idx]`) are also from the same historical data. **Correct.**

### 3. `--ar-max-blend` correctly passed through for historical branches (`e-dojo_v1.py:993-996`)

In `launch_branch()` (line 953), the AR cap is injected into the subprocess command at lines 993-996:
```python
if hasattr(self, '_dojo_state') and self._dojo_state is not None:
    ar_cap = self._dojo_state.ar_max_blend
    if ar_cap < 1.0:
        cmd.extend(["--ar-max-blend", f"{ar_cap:.2f}"])
```

This code runs for ALL branch types (Bella, synthetic, historical) because it is in `launch_branch()`, not in any source-specific branch. Historical launches at line 1187 call `launch_branch()` and receive the AR cap identically to Bella/synthetic branches. **Correct.**

### 4. AR cap from `dojo_state.json` is loaded BEFORE the first branch launch (known bug is FIXED)

The initialization flow in `main()` (`e-dojo_v1.py:1671+`):

1. **Line 1672**: `launcher = TrainingLauncher()` -- no `_dojo_state` attribute yet
2. **Lines 1701-1705**: Pre-load `launcher._dojo_state` from `dojo_state.json` if file exists (explicit comment: "Pre-load dojo state so AR cap is passed to initial launch")
3. **Lines 1707-1722**: Launch branches -- `launch_branch()` checks `self._dojo_state` at line 993 and finds the pre-loaded state
4. **Line 1735**: `dojo = EasyDojo(config, ...)` created, loads its own `self.state` from same file
5. **Line 1739**: `launcher._dojo_state = dojo.state` -- replaces pre-loaded state with the live dojo state object (for future restarts)

When `dojo_state.json` does not exist (fresh run), `_dojo_state` is never set on the launcher, so `hasattr(self, '_dojo_state')` at line 993 returns False, and no `--ar-max-blend` flag is passed. The training uses the config default of 0.90 (`easy_config.py:96`). This is correct behavior for a fresh start.

**The known bug (loading after first launch) has been fixed.**

### 5. AR blend computation is duplicated but consistent

The AR blend probability calculation appears in two locations:
- `_generate_episode_data()` at lines 697-706 (dead code -- never called)
- `_generate_batch_episodes()` at lines 954-962 (active path)

Both use identical logic:
```python
if not autoregressive_state or self._current_epoch < ar_warmup:
    ar_blend = 0.0
elif ar_blend_epochs > 0 and self._current_epoch < ar_warmup + ar_blend_epochs:
    ar_blend = min((self._current_epoch - ar_warmup) / ar_blend_epochs, ar_max_blend)
else:
    ar_blend = ar_max_blend
```

No inconsistency, but the dead code in `_generate_episode_data` could be cleaned up.

## Bugs Found

None. All four audit points pass:
1. AR rollout receives historical prices correctly
2. `_build_model_state_trajectory_batched` gets the same price window as oracle
3. `--ar-max-blend` is passed for historical branches
4. `dojo_state.json` AR cap is loaded before first launch (known bug was already fixed)

## Recommended Fixes

- **Minor cleanup**: Remove or mark `_generate_episode_data()` (`train_easy.py:636`) as deprecated since it is dead code (never called, superseded by `_generate_batch_episodes`). The duplicated AR blend logic there could drift from the active path without anyone noticing.
