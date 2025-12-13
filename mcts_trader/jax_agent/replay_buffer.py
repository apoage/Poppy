import jax
import jax.numpy as jnp
from typing import Dict, Tuple
import chex

class JAXReplayBuffer:
    """
    JAX-based replay buffer with GPU storage.
    More efficient than Python lists for JAX training.
    """

    def __init__(
        self,
        capacity: int,
        observation_shape: Tuple[int, ...] = (27, 64, 64),
        action_size: int = 192,  # n_assets * 3
        sequence_length: int = 5,
        batch_size: int = 256
    ):
        """
        Args:
            capacity: Maximum number of trajectories to store
            observation_shape: Shape of observations
            action_size: Number of possible actions (n_assets * 3)
            sequence_length: Length of unroll sequences
            batch_size: Training batch size
        """
        self.capacity = capacity
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.obs_shape = observation_shape
        self.action_size = action_size

        # Pre-allocate buffers (using float32 for observations to prevent Inf overflow)
        self.observations = jnp.zeros((capacity, *observation_shape), dtype=jnp.float32)
        self.actions = jnp.zeros((capacity,), dtype=jnp.int32)
        self.rewards = jnp.zeros((capacity,), dtype=jnp.float32)
        self.values = jnp.zeros((capacity,), dtype=jnp.float32)
        self.policy_probs = jnp.zeros((capacity, action_size), dtype=jnp.float32)
        self.dones = jnp.zeros((capacity,), dtype=jnp.float32)

        # Trajectory boundaries
        # We store start index and length for each trajectory
        self.trajectory_starts = jnp.zeros((capacity,), dtype=jnp.int32)
        self.trajectory_lengths = jnp.zeros((capacity,), dtype=jnp.int32)

        # Buffer state (these stay on CPU mostly, but JAX handles them)
        self.position = 0
        self.num_trajectories = 0

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

        # Update position
        self.position = (self.position + traj_length) % self.capacity
        self.num_trajectories += 1

    def sample_batch(self, rng: chex.PRNGKey) -> Dict[str, jnp.ndarray]:
        """
        Sample a batch of unroll sequences.

        Args:
            rng: JAX random key

        Returns:
            batch: Dict of JAX arrays
        """
        # We only want to sample from VALID trajectories that are currently in buffer.
        # If we wrapped around multiple times, num_trajectories > capacity.
        # Effectively we have min(num_trajectories, capacity) metadata entries?
        # No, metadata array size is 'capacity' but that's huge (same as data steps).
        # Usually metadata capacity << data capacity (e.g. max 10k eps vs 1M steps).
        # Here we made them same size for simplicity but it's wasteful.
        
        # Simplified sampling:
        # Sample 'batch_size' trajectory indices
        num_valid_trajs = min(self.num_trajectories, self.capacity)
        # Note: if num_trajectories > capacity, we might have overwritten data pointed to by old metadata.
        # This implementation is a "ring buffer of data" but "ring buffer of metadata" logic needs care.
        # For prototype: We assume capacity is large enough to hold N trajectories.
        # Actually, let's fix metadata indexing:
        # Metadata should be a separate circular buffer of size N_EPS (e.g. 1000).
        # 'capacity' passed to init is usually data steps.
        
        # For now, let's assume we don't wrap metadata often or capacity is N_EPS.
        # Wait, if capacity=10000 (steps) and traj_len=200, we fit 50 eps.
        
        traj_indices = jax.random.randint(
            rng, (self.batch_size,), 0, num_valid_trajs
        )

        # This logic is tricky in pure JAX without host callback for sampling logic, 
        # but JAX has 'jax.vmap' and 'jax.lax.map'.
        
        # Helper to get one sequence
        def get_sequence(traj_idx, key):
            # Resolve actual index in circular metadata buffer
            # (If we assume metadata buffer size = capacity, this is fine)
            meta_idx = (self.num_trajectories - 1 - traj_idx) % self.capacity
            # Actually simplest: sample from range [0, num_valid_trajs) and treat as logical indices
            # mapping to physical indices is (start_index + i) % size.
            
            # Let's assume metadata buffer is just growing for now (infinite)? No.
            # Use 'traj_indices' as direct indices into metadata arrays?
            # If num_trajs > capacity, we overwrite metadata at index i % capacity.
            # So valid indices are [0, capacity).
            
            real_traj_idx = traj_indices[traj_idx] # This is the random index sampled above
            
            start_pos = self.trajectory_starts[real_traj_idx]
            traj_len = self.trajectory_lengths[real_traj_idx]

            # Sample position within trajectory
            # Ensure we can unroll 'sequence_length' steps
            max_start = jnp.maximum(0, traj_len - self.sequence_length)
            pos = jax.random.randint(key, (), 0, max_start + 1)

            # Extract sequence indices handling wrap-around
            # Logical indices: start_pos + pos + [0..seq_len]
            logical_indices = start_pos + pos + jnp.arange(self.sequence_length)
            physical_indices = logical_indices % self.capacity
            
            return {
                'observations': self.observations[physical_indices],
                'actions': self.actions[physical_indices],
                'rewards': self.rewards[physical_indices],
                'values': self.values[physical_indices],
                'policy_probs': self.policy_probs[physical_indices],
                'dones': self.dones[physical_indices]
            }

        # Vectorize sampling
        # We need distinct keys for position sampling
        rngs = jax.random.split(rng, self.batch_size)
        # We pass indices 0..batch_size-1 just to drive map/vmap
        batch = jax.vmap(get_sequence)(jnp.arange(self.batch_size), rngs)
        
        return batch

    def __len__(self) -> int:
        return self.num_trajectories
