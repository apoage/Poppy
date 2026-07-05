# TASK_120: Lazy Observation Reconstruction (RAM/OOM Fix)

## Problem
The `run_self_play_jax` function currently reconstructs observations for all steps immediately upon fetching data from the GPU.
- **Data Size:** 15 envs * 1440 steps * (27 channels * 64 assets * 64 assets) * 4 bytes ≈ **9.5 GB** per episode batch.
- **Impact:** This causes massive RAM spikes, leading to system OOM kills or severe swapping (kernel CPU time spikes) during training, even if VRAM is stable.

## Solution
Implement **Lazy Observation Reconstruction**.
Instead of storing the full `(T, 27, 64, 64)` observation tensor in the trajectory result, store only the metadata needed to reconstruct it on demand.

### Strategy

1.  **Store Metadata Only:**
    - `window_starts`: (T,) indices into the price array.
    - `_full_prices_ref`: Reference to the full price array for the episode (already in memory).

2.  **Modify `run_self_play_jax`:**
    - Remove the call to `batch_encode_simple` inside the loop.
    - Pass `window_starts` and `_full_prices_ref` in the result dictionary.

3.  **Create Helper `reconstruct_observations`:**
    - A function that takes a compressed trajectory and returns the full observation array.
    - Uses `mcts_trader.env.jax_encoder.batch_encode_simple` on CPU/GPU as needed.

4.  **Update `train_jax_fast.py`:**
    - In the training loop, detect if observations are missing.
    - Call `reconstruct_observations` just before adding to the replay buffer.
    - This keeps the memory footprint low during the self-play phase and only expands it when transferring to the replay buffer (which has its own capacity limits).

## Checklist
- [ ] Refactor `mcts_trader/training/self_play.py`:
    - [ ] Update `run_self_play_jax` to skip encoding.
    - [ ] Implement `reconstruct_observations` helper.
- [ ] Refactor `train_jax_fast.py`:
    - [ ] Import `reconstruct_observations`.
    - [ ] Add reconstruction logic before `replay_buffer.add_trajectory`.
- [ ] Verify RAM usage is stable during historical training.

## Detailed Implementation Plan

### 1. `mcts_trader/training/self_play.py` Changes

**Function `run_self_play_jax`:**
```python
# ... (Fetch small arrays from GPU)
cpu_window_starts = fetch(trajectory_batch['window_starts'])
cpu_prices = np.array(prices)

# SKIP immediate reconstruction (this saves ~10GB RAM per batch)
# observations will be reconstructed lazily by the caller

# ... inside the loop per env ...
result = {
    'window_starts': env_window_starts,  # Store for reconstruction
    'actions': cpu_actions[i, :n_steps_i],
    'rewards': rewards_arr,
    # ... other fields ...
}

# Pass reference to FULL prices array for safe reconstruction later
# This is a lightweight reference, not a copy
if 'prices' not in result:
    result['_full_prices_ref'] = cpu_prices 
```

**New Helper Function `reconstruct_observations`:**
```python
def reconstruct_observations(
    trajectory: Dict,
    n_assets: int = 64,
    window_size: int = 60,
    n_channels: int = 27,
) -> np.ndarray:
    """Lazy reconstruction of observations from compressed trajectory data."""
    from mcts_trader.env.jax_encoder import EncoderConfig, batch_encode_simple
    
    if '_full_prices_ref' in trajectory:
        prices = trajectory['_full_prices_ref']
        window_starts = trajectory['window_starts']
    else:
        raise ValueError("Cannot reconstruct: missing full price reference")
        
    config = EncoderConfig(n_assets=n_assets, window_size=window_size, n_channels=n_channels)
    
    # Reconstruct on CPU (or GPU if JAX is available/fast enough)
    # This reconstructs ONE trajectory at a time (600MB) instead of 15 (9.5GB)
    obs = np.array(batch_encode_simple(
        jnp.array(prices), jnp.array(window_starts), config
    ))
    
    return obs.astype(np.float32)
```

### 2. `train_jax_fast.py` Changes

**Training Loop:**
```python
# ... inside the loop ...
if jax_env_states is not None:
    # ... run self play ...
    trajectories, scores = run_self_play_jax(...)

# Buffer update
for traj in trajectories:
    # Check if we need to reconstruct observations (Lazy Loading)
    if 'observations' not in traj and 'window_starts' in traj:
        from mcts_trader.training.self_play import reconstruct_observations
        # Reconstruct ON DEMAND - avoids holding all envs' obs in RAM simultaneously
        traj['observations'] = reconstruct_observations(
            traj, n_assets=n_assets, window_size=60
        )

    replay_buffer.add_trajectory(...)
```