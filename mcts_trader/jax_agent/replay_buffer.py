import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Tuple, Optional
import chex

class JAXReplayBuffer:
    """
    JAX-based replay buffer with GPU storage.
    More efficient than Python lists for JAX training.

    Extended with auxiliary price data for prediction loss (Phase 2).
    All price data stored in LOG-SPACE for numerical stability.
    """

    def __init__(
        self,
        capacity: int,
        observation_shape: Tuple[int, ...] = (27, 64, 64),
        action_size: int = 192,  # n_assets * 3
        sequence_length: int = 5,
        batch_size: int = 256,
        # NEW: Auxiliary data parameters
        n_assets: int = 64,
        n_features: int = 10,
    ):
        """
        Args:
            capacity: Maximum number of trajectories to store
            observation_shape: Shape of observations
            action_size: Number of possible actions (n_assets * 3)
            sequence_length: Length of unroll sequences
            batch_size: Training batch size
            n_assets: Number of tradeable assets (for prediction model)
            n_features: Number of external features (for prediction model)
        """
        self.capacity = capacity
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.obs_shape = observation_shape
        self.action_size = action_size
        self.n_assets = n_assets
        self.n_features = n_features

        # Pre-allocate buffers (using float32 for observations to prevent Inf overflow)
        self.observations = jnp.zeros((capacity, *observation_shape), dtype=jnp.float32)
        self.actions = jnp.zeros((capacity,), dtype=jnp.int32)
        self.rewards = jnp.zeros((capacity,), dtype=jnp.float32)
        self.values = jnp.zeros((capacity,), dtype=jnp.float32)
        self.policy_probs = jnp.zeros((capacity, action_size), dtype=jnp.float32)
        self.dones = jnp.zeros((capacity,), dtype=jnp.float32)

        # NEW: Auxiliary fields for prediction loss (LOG-SPACE)
        # Use numpy arrays for storage (faster CPU writes), convert to JAX on sample
        self._log_prices = np.zeros((capacity, n_assets), dtype=np.float32)
        self._next_log_prices = np.zeros((capacity, n_assets), dtype=np.float32)
        self._log_returns = np.zeros((capacity, n_assets), dtype=np.float32)
        self._moving_avg = np.zeros((capacity, n_assets), dtype=np.float32)
        self._features = np.zeros((capacity, n_features), dtype=np.float32)
        self._has_auxiliary = np.zeros((capacity,), dtype=np.bool_)

        # Trajectory boundaries
        # We store start index and length for each trajectory
        self.trajectory_starts = jnp.zeros((capacity,), dtype=jnp.int32)
        self.trajectory_lengths = jnp.zeros((capacity,), dtype=jnp.int32)
        # Track when trajectory was added (for wrap-around validation)
        # Use int64 to prevent overflow in extremely long training runs
        self.trajectory_generation = jnp.zeros((capacity,), dtype=jnp.int64)

        # Buffer state (these stay on CPU mostly, but JAX handles them)
        self.position = 0
        self.num_trajectories = 0
        self.total_steps_added = 0  # Track total steps for wrap-around validation

    def add_trajectory(
        self,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        rewards: jnp.ndarray,
        values: jnp.ndarray,
        policy_probs: jnp.ndarray,
        dones: jnp.ndarray
    ):
        """
        Add a complete trajectory to the buffer.
        
        Args:
            observations: (T, 27, 64, 64)
            actions: (T,)
            rewards: (T,)
            values: (T,)
            policy_probs: (T, 192)
            dones: (T,)
        """
        # Ensure inputs are JAX arrays
        observations = jnp.asarray(observations)
        actions = jnp.asarray(actions)
        rewards = jnp.asarray(rewards)
        values = jnp.asarray(values)
        policy_probs = jnp.asarray(policy_probs)
        dones = jnp.asarray(dones)

        traj_length = observations.shape[0]

        # Check if we need to wrap around or if traj is too long
        if traj_length > self.capacity:
            print(f"Warning: Trajectory length {traj_length} > capacity {self.capacity}. Skipping.")
            return

        # Simple circular buffer logic for trajectories is complex because variable lengths.
        # Simplified approach: Store flattened, track indices.
        # BUT for high performance, we often use fixed block sizes or just wrap.
        # Here we implement a simple linear wrap-around for the data arrays.
        
        end_pos = self.position + traj_length
        
        if end_pos <= self.capacity:
            # Linear add
            indices = jnp.arange(self.position, end_pos)
            self.observations = self.observations.at[indices].set(observations)
            self.actions = self.actions.at[indices].set(actions)
            self.rewards = self.rewards.at[indices].set(rewards)
            self.values = self.values.at[indices].set(values)
            self.policy_probs = self.policy_probs.at[indices].set(policy_probs)
            self.dones = self.dones.at[indices].set(dones)
        else:
            # Wrap around
            split = self.capacity - self.position
            # First part
            indices1 = jnp.arange(self.position, self.capacity)
            self.observations = self.observations.at[indices1].set(observations[:split])
            self.actions = self.actions.at[indices1].set(actions[:split])
            self.rewards = self.rewards.at[indices1].set(rewards[:split])
            self.values = self.values.at[indices1].set(values[:split])
            self.policy_probs = self.policy_probs.at[indices1].set(policy_probs[:split])
            self.dones = self.dones.at[indices1].set(dones[:split])
            
            # Second part
            remaining = traj_length - split
            indices2 = jnp.arange(0, remaining)
            self.observations = self.observations.at[indices2].set(observations[split:])
            self.actions = self.actions.at[indices2].set(actions[split:])
            self.rewards = self.rewards.at[indices2].set(rewards[split:])
            self.values = self.values.at[indices2].set(values[split:])
            self.policy_probs = self.policy_probs.at[indices2].set(policy_probs[split:])
            self.dones = self.dones.at[indices2].set(dones[split:])

        # Record trajectory metadata
        # We use num_trajectories to index metadata arrays circularly
        traj_idx = self.num_trajectories % self.capacity
        self.trajectory_starts = self.trajectory_starts.at[traj_idx].set(self.position)
        self.trajectory_lengths = self.trajectory_lengths.at[traj_idx].set(traj_length)
        # Track generation for wrap-around validation
        current_generation = self.total_steps_added // self.capacity
        self.trajectory_generation = self.trajectory_generation.at[traj_idx].set(current_generation)

        # Update position
        self.position = (self.position + traj_length) % self.capacity
        self.num_trajectories += 1
        self.total_steps_added += traj_length

    def add_auxiliary(
        self,
        prices: np.ndarray,
        next_prices: np.ndarray,
        returns: Optional[np.ndarray] = None,
        moving_avg: Optional[np.ndarray] = None,
        features: Optional[np.ndarray] = None,
        indices: Optional[np.ndarray] = None,
    ):
        """
        Add auxiliary data for prediction loss.

        IMPORTANT: Should be called AFTER add_trajectory with the same data indices.
        Uses the trajectory just added (based on position).

        All price data is converted to LOG-SPACE for numerical stability.

        Args:
            prices: (T, n_assets) Raw prices at each step (will be log-transformed)
            next_prices: (T, n_assets) Raw prices at t+1 (will be log-transformed)
            returns: (T, n_assets) Optional log-returns (computed if None)
            moving_avg: (T, n_assets) Moving average of log-prices
            features: (T, n_features) External features
            indices: Optional explicit indices to write to (if not using last trajectory)
        """
        traj_length = prices.shape[0]

        # Determine where to write
        if indices is None:
            # Use the trajectory just added
            # The last add_trajectory wrote from (position - traj_length) to (position - 1)
            start_idx = (self.position - traj_length) % self.capacity
            if start_idx + traj_length <= self.capacity:
                indices = np.arange(start_idx, start_idx + traj_length)
            else:
                # Wrapped around
                split = self.capacity - start_idx
                indices = np.concatenate([
                    np.arange(start_idx, self.capacity),
                    np.arange(0, traj_length - split)
                ])

        # Convert to LOG-SPACE with numerical safety
        log_prices = np.log(np.clip(prices, 1e-8, None))
        log_next_prices = np.log(np.clip(next_prices, 1e-8, None))

        # Compute log-returns if not provided
        if returns is not None:
            log_returns = returns  # Assume already log-returns
        else:
            log_returns = log_next_prices - log_prices

        # Moving average (default to current if not provided)
        if moving_avg is not None:
            log_moving_avg = np.log(np.clip(moving_avg, 1e-8, None)) if np.any(moving_avg > 0) else log_prices
        else:
            log_moving_avg = log_prices.copy()

        # Store in buffer
        for i, idx in enumerate(indices):
            self._log_prices[idx] = log_prices[i]
            self._next_log_prices[idx] = log_next_prices[i]
            self._log_returns[idx] = log_returns[i]
            self._moving_avg[idx] = log_moving_avg[i]
            if features is not None and i < len(features):
                self._features[idx] = features[i]
            self._has_auxiliary[idx] = True

    def sample_batch(self, rng: chex.PRNGKey) -> Dict[str, jnp.ndarray]:
        """
        Sample a batch of unroll sequences with generation-validated wrap-around safety.

        GENERATION VALIDATION (Correct by Design):
        1. Each trajectory stores its generation (total_steps // capacity when written)
        2. Current generation = total_steps_added // capacity
        3. A trajectory's data is DEFINITELY VALID if:
           - Same generation (gen_diff == 0): data hasn't been touched
           - Previous generation (gen_diff == 1) AND write position hasn't reached it yet
        4. A trajectory's data is DEFINITELY INVALID if:
           - gen_diff >= 2: buffer has wrapped past this data

        Args:
            rng: JAX random key

        Returns:
            batch: Dict of JAX arrays
        """
        current_generation = self.total_steps_added // self.capacity

        # Compute the range of valid trajectories
        # Start from most recent and go backwards until we find invalid ones
        # For efficiency, we use a heuristic window but validate with generation

        # Heuristic: Start with capacity // 500 recent trajectories
        heuristic_window = max(2, self.capacity // 500)
        heuristic_window = min(heuristic_window, self.num_trajectories)

        # Sample from recent trajectories (logical indices)
        end_logical = self.num_trajectories
        start_logical = max(0, end_logical - heuristic_window)

        # Sample logical trajectory indices
        rng, sample_rng = jax.random.split(rng)
        logical_traj_indices = jax.random.randint(
            sample_rng, (self.batch_size,), start_logical, end_logical
        )

        # Capture current state for use in vmap
        current_position = self.position

        # Convert auxiliary numpy arrays to JAX for indexing
        # (only done once per sample_batch call)
        log_prices_jax = jnp.array(self._log_prices)
        log_returns_jax = jnp.array(self._log_returns)
        moving_avg_jax = jnp.array(self._moving_avg)
        features_jax = jnp.array(self._features)

        # Helper to get one sequence with generation validation
        def get_sequence(batch_idx, key):
            logical_idx = logical_traj_indices[batch_idx]
            physical_traj_idx = logical_idx % self.capacity

            # Get trajectory metadata
            start_pos = self.trajectory_starts[physical_traj_idx]
            traj_len = self.trajectory_lengths[physical_traj_idx]
            traj_gen = self.trajectory_generation[physical_traj_idx]

            # GENERATION VALIDATION
            gen_diff = current_generation - traj_gen

            # Check if trajectory is valid
            is_same_gen = gen_diff == 0
            is_prev_gen_safe = (gen_diff == 1) & (current_position <= start_pos)
            is_valid = is_same_gen | is_prev_gen_safe

            # Sample position within trajectory
            max_start = jnp.maximum(0, traj_len - self.sequence_length)
            pos = jax.random.randint(key, (), 0, max_start + 1)

            # Calculate physical indices in data arrays with wrap-around
            logical_data_indices = start_pos + pos + jnp.arange(self.sequence_length)
            physical_data_indices = logical_data_indices % self.capacity

            # Always fetch data (indexing is cheap)
            obs = self.observations[physical_data_indices]
            acts = self.actions[physical_data_indices]
            rews = self.rewards[physical_data_indices]
            vals = self.values[physical_data_indices]
            pols = self.policy_probs[physical_data_indices]
            dons = self.dones[physical_data_indices]

            # Fetch auxiliary data (LOG-SPACE)
            log_p = log_prices_jax[physical_data_indices]
            log_r = log_returns_jax[physical_data_indices]
            ma = moving_avg_jax[physical_data_indices]
            feat = features_jax[physical_data_indices[0]]  # Features at first timestep

            # Return data with validity flag (don't zero - let loss function mask instead)
            # This preserves gradients for valid samples only
            return {
                'observations': obs,
                'actions': acts,
                'rewards': rews,
                'values': vals,
                'policy_probs': pols,
                'dones': dons,
                # Auxiliary data for prediction loss
                'log_price_sequence': log_p,
                'log_returns': log_r[0],
                'moving_avg': ma[0],
                'features': feat,
                # Validity mask for loss computation
                'valid': is_valid,
            }

        # Vectorize sampling
        rngs = jax.random.split(rng, self.batch_size)
        batch = jax.vmap(get_sequence)(jnp.arange(self.batch_size), rngs)

        return batch

    def clear(self):
        """
        Reset buffer to empty state, keeping allocated memory.
        Use when transferring to a new curriculum level to avoid
        training on stale experiences from a different market regime.
        """
        self.position = 0
        self.num_trajectories = 0
        self.total_steps_added = 0
        # Reset auxiliary data flags
        self._has_auxiliary[:] = False
        # Note: We don't zero the arrays - just reset the counters
        # The data will be overwritten as new trajectories are added

    def __len__(self) -> int:
        return self.num_trajectories
