"""
TASK_125: Memory-Efficient Expert Buffer for Causal Expert Imprinting.
TASK_135: Eliminated disk-based price_cache - stores episode prices in RAM.

Uses index-based storage (32 bytes/sample) instead of full observations (400KB/sample).
Achieves 8000x memory reduction: 220 GB -> 27 MB for indices.
Episode prices stored in RAM (~5-25MB per epoch vs 12GB pre-allocated disk).

Architecture:
- Protected demo zone [0:demo_capacity] - never overwritten
- FIFO agent zone [demo_capacity:total_capacity] - circular buffer
- Segment trees for O(log n) priority sampling (PER)
- Internal episode price storage (RAM, cleared on data regeneration)
- Thread-safe with locking
"""

import numpy as np
from typing import Optional, Dict, Tuple
import threading
import time
from functools import lru_cache


# Compact index entry - stores WHERE to find data, not the data itself
# 32 bytes per sample (vs ~400KB for full observation)
INDEX_DTYPE = np.dtype([
    ('episode_id', np.uint32),       # 4 bytes - which episode in PriceCache
    ('step', np.uint16),             # 2 bytes - which step in episode
    ('action', np.uint16),           # 2 bytes - action taken (0-319)
    ('optimal_action', np.uint16),   # 2 bytes - hindsight-optimal action
    ('reward', np.float16),          # 2 bytes - immediate reward
    ('confidence', np.float16),      # 2 bytes - how optimal (for demos)
    ('episode_return', np.float16),  # 2 bytes - total episode return
    ('regime_tag', np.uint8),        # 1 byte - market regime (0-7)
    ('is_demo', np.uint8),           # 1 byte - from demo buffer?
    ('timestamp', np.uint32),        # 4 bytes - when added (for recency)
    ('priority', np.float32),        # 4 bytes - current priority
    ('td_error', np.float32),        # 4 bytes - last TD error
], align=True)
# Total: 32 bytes per entry (with alignment)


class UnifiedIndexBuffer:
    """
    Memory-efficient replay buffer using index storage.

    Layout:
    [0 ... demo_capacity-1] = Protected demo zone (never overwritten)
    [demo_capacity ... total_capacity-1] = FIFO agent zone

    Memory usage:
    - 50k demos x 32 bytes = 1.6 MB
    - 500k agent x 32 bytes = 16 MB
    - Priority trees = ~9 MB
    - Episode prices = ~5-25 MB per epoch (cleared on data regeneration)
    - Total: ~50 MB typical (vs 12 GB pre-allocated disk!)
    """

    def __init__(
        self,
        n_assets: int = 64,
        demo_capacity: int = 50_000,
        agent_capacity: int = 500_000,
        window_size: int = 60,
        per_alpha: float = 0.6,
        per_beta_init: float = 0.4,
        per_beta_final: float = 1.0,
        per_beta_steps: int = 100_000,
        demo_priority_boost: float = 0.001,
    ):
        """
        Initialize unified buffer with protected demo zone.

        TASK_135: Removed price_cache dependency - stores episode prices in RAM.

        Args:
            n_assets: Number of tradeable assets
            demo_capacity: Max demo samples (protected, never overwritten)
            agent_capacity: Max agent samples (FIFO eviction)
            window_size: Observation window size
            per_alpha: Priority exponent for PER
            per_beta_init: Initial importance sampling exponent
            per_beta_final: Final importance sampling exponent
            per_beta_steps: Steps to anneal beta
            demo_priority_boost: Extra priority for demos (epsilon_D)
        """
        self.n_assets = n_assets
        self.demo_capacity = demo_capacity
        self.agent_capacity = agent_capacity
        self.total_capacity = demo_capacity + agent_capacity
        self.window_size = window_size

        # PER parameters
        self.per_alpha = per_alpha
        self.per_beta = per_beta_init
        self.per_beta_init = per_beta_init
        self.per_beta_final = per_beta_final
        self.per_beta_steps = per_beta_steps
        self.demo_priority_boost = demo_priority_boost
        self.update_count = 0

        # Index storage (tiny!)
        self.indices = np.zeros(self.total_capacity, dtype=INDEX_DTYPE)

        # Counters
        self.demo_count = 0
        self.agent_count = 0
        self.agent_write_idx = demo_capacity  # Start after demo zone

        # Priority trees (CPU-side, efficient sampling)
        # Find next power of 2 >= total_capacity
        tree_capacity = 1
        while tree_capacity < self.total_capacity:
            tree_capacity *= 2
        self.sum_tree = np.zeros(2 * tree_capacity, dtype=np.float64)
        self.min_tree = np.full(2 * tree_capacity, np.inf, dtype=np.float64)
        self.tree_capacity = tree_capacity

        self.max_priority = 1.0

        # Thread safety
        self._lock = threading.Lock()

        # TASK_135: Internal episode price storage (replaces disk-based price_cache)
        # Stores episode prices in RAM, cleared on data regeneration
        # Memory: ~5-25 MB per epoch vs 12 GB pre-allocated disk
        self._episode_prices: Dict[int, np.ndarray] = {}
        self._next_episode_id = 0

        # TASK_136a: Pre-encoded compressed observations
        # Stores (T - window_size, 27, n_assets) per episode instead of full (27, n, n)
        # Memory: 64× compression → 345 MB vs 22 GB for 50k demos
        self._compressed_obs: Dict[int, np.ndarray] = {}  # ep_id -> (steps, 27, n_assets)

    # ==================== EPISODE STORAGE (TASK_135) ====================

    def add_episode(self, prices: np.ndarray) -> int:
        """
        Store episode prices and return episode_id.

        TASK_135: Prices stored in RAM instead of disk-based price_cache.
        Called by on_episode_complete callback with _full_prices_ref.

        Args:
            prices: (T, n_assets) close prices for the episode
                    OR (T, n_assets, n_features) OHLCV - will extract close prices

        Returns:
            episode_id for referencing this episode's prices
        """
        with self._lock:
            # Handle OHLCV format: extract close prices (index 3)
            if prices.ndim == 3:
                prices = prices[:, :, 3]  # (T, n_assets, 5) -> (T, n_assets)

            # Store copy to avoid reference issues
            ep_id = self._next_episode_id
            self._episode_prices[ep_id] = prices.astype(np.float32).copy()
            self._next_episode_id += 1

            return ep_id

    def get_episode_prices(self, episode_id: int) -> Optional[np.ndarray]:
        """Get stored prices for an episode."""
        return self._episode_prices.get(episode_id)

    def preencode_episode(self, episode_id: int) -> bool:
        """
        TASK_136a: Pre-encode all valid steps for an episode.

        Pre-computes compressed observations (27, n_assets) for all steps >= window_size.
        Called after add_episode() to avoid encoding at sample time.

        Args:
            episode_id: Episode ID to pre-encode

        Returns:
            True if successful, False if episode not found
        """
        import jax.numpy as jnp
        from mcts_trader.env.jax_encoder import (
            EncoderConfig, jit_batch_encode_compressed
        )

        prices = self._episode_prices.get(episode_id)
        if prices is None:
            return False

        T = prices.shape[0]
        n_valid_steps = T - self.window_size

        if n_valid_steps <= 0:
            return False

        # Config for encoder
        config = EncoderConfig(
            n_assets=self.n_assets,
            window_size=self.window_size,
            n_channels=27,
        )

        # Window starts for all valid steps: steps [window_size, window_size+1, ..., T-1]
        # window_start = step - window_size
        window_starts = jnp.arange(0, n_valid_steps, dtype=jnp.int32)

        # Batch encode all steps at once - single JAX call!
        compressed = jit_batch_encode_compressed(
            jnp.array(prices),
            window_starts,
            config
        )

        # Store (n_valid_steps, 27, n_assets)
        self._compressed_obs[episode_id] = np.array(compressed)
        return True

    def get_episode_count(self) -> int:
        """Get number of stored episodes."""
        return len(self._episode_prices)

    def get_price_storage_mb(self) -> float:
        """Get memory used by price storage in MB."""
        total_bytes = sum(p.nbytes for p in self._episode_prices.values())
        return total_bytes / 1e6

    def get_compressed_obs_mb(self) -> float:
        """TASK_136a: Get memory used by compressed observations in MB."""
        total_bytes = sum(c.nbytes for c in self._compressed_obs.values())
        return total_bytes / 1e6

    def has_compressed_obs(self, episode_id: int) -> bool:
        """Check if episode has pre-encoded compressed observations."""
        return episode_id in self._compressed_obs

    # ==================== ADDING SAMPLES ====================

    def add_demo(
        self,
        episode_id: int,
        step: int,
        action: int,
        optimal_action: int,
        reward: float,
        confidence: float,
        episode_return: float,
        regime_tag: int = 0,
    ) -> bool:
        """
        Add to protected demo zone.

        Args:
            episode_id: Episode index in PriceCache
            step: Step within episode
            action: Action actually taken
            optimal_action: Hindsight-optimal action
            reward: Immediate reward at this step
            confidence: How "optimal" this demo is (0-1)
            episode_return: Total episode return
            regime_tag: Market regime identifier

        Returns:
            True if added, False if demo zone is full
        """
        with self._lock:
            if self.demo_count >= self.demo_capacity:
                return False

            idx = self.demo_count
            self._set_entry(
                idx, episode_id, step, action, optimal_action,
                reward, confidence, episode_return, regime_tag,
                is_demo=1
            )

            # BUG-007 FIX: Scale demo priority to be competitive with agent TD-errors
            # Old: priority = confidence + 0.001 → ~0.8-1.0 (too low!)
            # New: priority = confidence * max_priority → scales with agent priorities
            # This ensures demos with high confidence compete fairly with high-TD-error agents
            priority = confidence * self.max_priority + self.demo_priority_boost
            self._update_priority(idx, priority)

            self.demo_count += 1
            return True

    def add_agent(
        self,
        episode_id: int,
        step: int,
        action: int,
        reward: float,
        td_error: float = 1.0,
        episode_return: float = 0.0,
        regime_tag: int = 0,
    ):
        """
        Add to FIFO agent zone.

        Args:
            episode_id: Episode index in PriceCache
            step: Step within episode
            action: Action taken
            reward: Immediate reward
            td_error: TD error for priority
            episode_return: Total episode return
            regime_tag: Market regime identifier
        """
        with self._lock:
            idx = self.agent_write_idx

            self._set_entry(
                idx, episode_id, step, action, action,  # optimal = action for agent
                reward, 0.0, episode_return, regime_tag,
                is_demo=0
            )
            self.indices[idx]['td_error'] = td_error

            # Priority from TD error
            priority = np.abs(td_error) + 1e-6
            self._update_priority(idx, priority)

            # FIFO: advance and wrap within agent zone
            self.agent_write_idx += 1
            if self.agent_write_idx >= self.total_capacity:
                self.agent_write_idx = self.demo_capacity  # Wrap to agent zone start

            self.agent_count = min(self.agent_count + 1, self.agent_capacity)

    def _set_entry(
        self, idx: int, episode_id: int, step: int, action: int, optimal_action: int,
        reward: float, confidence: float, episode_return: float, regime_tag: int, is_demo: int
    ):
        """Set index entry fields."""
        entry = self.indices[idx]
        entry['episode_id'] = episode_id
        entry['step'] = step
        entry['action'] = action
        entry['optimal_action'] = optimal_action
        entry['reward'] = reward
        entry['confidence'] = confidence
        entry['episode_return'] = episode_return
        entry['regime_tag'] = regime_tag
        entry['is_demo'] = is_demo
        entry['timestamp'] = self.update_count

    # ==================== PRIORITY TREE OPERATIONS ====================

    def _update_priority(self, idx: int, priority: float):
        """Update priority in sum and min trees."""
        priority_alpha = priority ** self.per_alpha
        self.indices[idx]['priority'] = priority
        self.max_priority = max(self.max_priority, priority)

        # Update sum tree (for proportional sampling)
        tree_idx = idx + self.tree_capacity
        self.sum_tree[tree_idx] = priority_alpha
        while tree_idx > 1:
            tree_idx //= 2
            self.sum_tree[tree_idx] = (
                self.sum_tree[2 * tree_idx] +
                self.sum_tree[2 * tree_idx + 1]
            )

        # Update min tree (for importance sampling normalization)
        tree_idx = idx + self.tree_capacity
        self.min_tree[tree_idx] = priority_alpha
        while tree_idx > 1:
            tree_idx //= 2
            self.min_tree[tree_idx] = min(
                self.min_tree[2 * tree_idx],
                self.min_tree[2 * tree_idx + 1]
            )

    def _sample_proportional(self, n: int, valid_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Sample n indices proportional to priority.

        Args:
            n: Number of samples
            valid_mask: Optional mask of valid indices

        Returns:
            Array of sampled indices
        """
        indices = np.zeros(n, dtype=np.int32)
        total = self.sum_tree[1]  # Root = total sum

        if total <= 0:
            return indices

        segment = total / n

        for i in range(n):
            # Sample uniformly within segment
            s = np.random.uniform(segment * i, segment * (i + 1))

            # Find leaf with this cumsum
            idx = 1
            while idx < self.tree_capacity:
                left = 2 * idx
                if left < len(self.sum_tree) and self.sum_tree[left] > s:
                    idx = left
                else:
                    if left < len(self.sum_tree):
                        s -= self.sum_tree[left]
                    idx = left + 1

            buffer_idx = idx - self.tree_capacity

            # Clamp to valid range
            buffer_idx = np.clip(buffer_idx, 0, self.total_capacity - 1)
            indices[i] = buffer_idx

        return indices

    # ==================== SAMPLING ====================

    def sample_batch(
        self,
        batch_size: int = 1024,
        demo_ratio: float = 0.125,  # 128/1024 from demos
        training_step: Optional[int] = None,  # TASK_132B: Pass actual training step
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Sample mixed batch from demo and agent zones.

        Returns dict with observations reconstructed on-demand from PriceCache.

        Args:
            batch_size: Total batch size
            demo_ratio: Fraction of batch from demo zone
            training_step: Actual training step for beta annealing (if None, uses sample count)

        Returns:
            Dict with observations, actions, weights, etc. or None if insufficient data
        """
        total_available = self.demo_count + self.agent_count
        if total_available == 0:
            return None

        # BUG FIX (TASK_131c): Handle small buffers gracefully
        # Old: return None if total_available < batch_size (blocked BC training early)
        # New: Reduce batch_size to available samples (minimum 16 for stable gradients)
        min_batch = 16
        if total_available < min_batch:
            return None  # Not enough for stable batch
        effective_batch_size = min(batch_size, total_available)

        # Anneal beta over training
        # TASK_132B FIX: Use training_step if provided, else fall back to sample count
        self.update_count += 1
        progress_step = training_step if training_step is not None else self.update_count
        self.per_beta = min(
            self.per_beta_final,
            self.per_beta_init +
            (self.per_beta_final - self.per_beta_init) *
            (progress_step / self.per_beta_steps)
        )

        with self._lock:
            # Determine sample counts (using effective_batch_size for small buffers)
            n_demo = min(int(effective_batch_size * demo_ratio), self.demo_count)
            n_agent = effective_batch_size - n_demo
            n_agent = min(n_agent, self.agent_count)
            n_demo = effective_batch_size - n_agent  # Adjust if not enough agent samples

            indices = []

            # Sample from demo zone
            if n_demo > 0 and self.demo_count > 0:
                demo_indices = self._sample_from_range(n_demo, 0, self.demo_count)
                indices.extend(demo_indices)

            # Sample from agent zone
            if n_agent > 0 and self.agent_count > 0:
                agent_start = self.demo_capacity
                agent_end = self.demo_capacity + self.agent_count
                agent_indices = self._sample_from_range(n_agent, agent_start, agent_end)
                indices.extend(agent_indices)

            indices = np.array(indices, dtype=np.int32)

            if len(indices) == 0:
                return None

            # Get entries
            entries = self.indices[indices]

            # Compute importance sampling weights
            priorities = entries['priority'].astype(np.float64) ** self.per_alpha
            priorities = np.maximum(priorities, 1e-10)

            total_priority = self.sum_tree[1]
            if total_priority <= 0:
                total_priority = 1.0

            probs = priorities / total_priority
            probs = np.maximum(probs, 1e-10)

            # w_i = (N * P(i))^(-beta), normalized by max
            N = total_available
            weights = (N * probs) ** (-self.per_beta)

            # Normalize by max weight
            min_prob = self.min_tree[1] / total_priority
            if min_prob > 0:
                max_weight = (N * min_prob) ** (-self.per_beta)
                weights = weights / max(max_weight, 1e-10)
            else:
                weights = weights / max(weights.max(), 1e-10)

            weights = np.clip(weights, 0.1, 10.0).astype(np.float32)

        # Vectorized observation encoding using JAX encoder (OUTSIDE lock)
        # Strategy: Group by episode, batch-encode per episode, scatter back
        observations = self._encode_observations_vectorized(entries)

        return {
            'observations': observations,                          # (batch, channels, window, assets)
            'actions': entries['action'].astype(np.int32),         # (batch,)
            'optimal_actions': entries['optimal_action'].astype(np.int32),  # (batch,)
            'rewards': entries['reward'].astype(np.float32),
            'confidence': entries['confidence'].astype(np.float32),
            'is_demo': entries['is_demo'].astype(bool),
            'weights': weights,
            'indices': indices,
            'regime_tags': entries['regime_tag'].astype(np.int32),
            'episode_returns': entries['episode_return'].astype(np.float32),
        }

    def _encode_observations_vectorized(self, entries: np.ndarray) -> np.ndarray:
        """
        Vectorized observation encoding using JAX encoder with caching.

        TASK_135: Uses internal episode price storage instead of disk-based price_cache.
        TASK_136a: Uses pre-encoded compressed observations when available (25× faster).

        Args:
            entries: Structured array with episode_id, step fields

        Returns:
            (batch_size, 27, n_assets, n_assets) encoded observations
        """
        import jax
        import jax.numpy as jnp
        from mcts_trader.env.jax_encoder import (
            EncoderConfig, encode_observation_simple, jit_batch_tile_compressed
        )

        batch_size = len(entries)
        n_assets = self.n_assets

        # Pre-allocate output
        observations = np.zeros((batch_size, 27, n_assets, n_assets), dtype=np.float32)

        # Encoder config (hashable for caching)
        config = EncoderConfig(
            n_assets=n_assets,
            window_size=self.window_size,
            n_channels=27,
        )

        # Get unique episodes
        episode_ids = entries['episode_id'].astype(np.int32)
        unique_episodes = np.unique(episode_ids)

        # Collect samples that need tiling vs encoding
        compressed_samples = []  # [(batch_idx, compressed_obs), ...]
        encode_samples = []      # [(batch_idx, ep_id, step), ...]

        for ep_id in unique_episodes:
            mask = episode_ids == ep_id
            batch_indices = np.where(mask)[0]
            if len(batch_indices) == 0:
                continue

            steps = entries['step'][mask].astype(np.int32)

            # TASK_132e: Filter out steps < window_size
            valid_step_mask = steps >= self.window_size
            if not np.all(valid_step_mask):
                steps = steps[valid_step_mask]
                batch_indices = batch_indices[valid_step_mask]
                if len(steps) == 0:
                    continue

            # TASK_136a: Check for pre-encoded compressed observations
            compressed = self._compressed_obs.get(int(ep_id))
            if compressed is not None:
                # Look up compressed obs by step index
                # compressed has shape (n_valid_steps, 27, n_assets)
                # step S maps to index (S - window_size)
                for batch_idx, step in zip(batch_indices, steps):
                    obs_idx = step - self.window_size
                    if 0 <= obs_idx < compressed.shape[0]:
                        compressed_samples.append((batch_idx, compressed[obs_idx]))
                    else:
                        # DEBUG: obs_idx out of range - this shouldn't happen often
                        encode_samples.append((batch_idx, int(ep_id), step))
            else:
                # Fall back to encoding (agent samples or missing compression)
                for batch_idx, step in zip(batch_indices, steps):
                    encode_samples.append((batch_idx, int(ep_id), step))

        # TASK_136a: Batch-tile all compressed samples at once
        if compressed_samples:
            compressed_batch_indices = [s[0] for s in compressed_samples]
            compressed_obs = np.stack([s[1] for s in compressed_samples])  # (N, 27, n_assets)

            # Single JAX call to tile all compressed observations
            tiled = jit_batch_tile_compressed(jnp.array(compressed_obs), n_assets)
            observations[compressed_batch_indices] = np.array(tiled)

        # Fall back to full encoding for samples without compressed obs
        if encode_samples:
            encoder_fn = self._get_jit_encoder(config)

            # Group by episode for efficient encoding
            from collections import defaultdict
            ep_samples = defaultdict(list)
            for batch_idx, ep_id, step in encode_samples:
                ep_samples[ep_id].append((batch_idx, step))

            for ep_id, samples in ep_samples.items():
                close_prices_np = self._episode_prices.get(ep_id)
                if close_prices_np is None:
                    continue

                close_prices = jnp.array(close_prices_np)
                batch_indices = [s[0] for s in samples]
                steps = np.array([s[1] for s in samples])
                window_starts = jnp.array(steps) - self.window_size

                try:
                    encoded = jax.vmap(
                        lambda ws: encoder_fn(close_prices, ws)
                    )(window_starts)
                    observations[batch_indices] = np.array(encoded)
                except Exception as e:
                    print(f"[WARN] Encoding failed for episode {ep_id}: {e}")

        return observations

    @staticmethod
    @lru_cache(maxsize=4)
    def _get_jit_encoder(config):
        """Get JIT-compiled encoder function (cached by config)."""
        import jax
        from mcts_trader.env.jax_encoder import encode_observation_simple

        @jax.jit
        def _encode(prices, window_start):
            return encode_observation_simple(prices, window_start, config)

        return _encode

    def _sample_from_range(self, n: int, start: int, end: int) -> list:
        """Sample n indices from [start, end) range using stratified priority sampling.

        BUG-009 FIX: Use stratified sampling (like segment tree) instead of np.random.choice.
        Stratified sampling divides total priority into n equal segments and samples one
        from each segment, ensuring coverage across all priority levels.
        """
        if end <= start or n <= 0:
            return []

        range_size = end - start

        # Get priorities for this range
        priorities = self.indices[start:end]['priority'].astype(np.float64)
        priorities = np.maximum(priorities, 1e-10) ** self.per_alpha

        total = priorities.sum()
        if total <= 0:
            # Uniform sampling fallback
            range_indices = np.arange(start, end)
            return list(np.random.choice(range_indices, min(n, range_size), replace=False))

        # BUG-009 FIX: Stratified sampling - divide total into n segments
        # This ensures we sample evenly across priority distribution
        cumsum = np.cumsum(priorities)
        segment_size = total / n

        indices = []
        for i in range(n):
            # Random value within this segment
            s = np.random.uniform(segment_size * i, segment_size * (i + 1))
            s = min(s, total - 1e-10)  # Clamp to avoid edge case

            # Binary search for the index with this cumsum
            idx = np.searchsorted(cumsum, s)
            idx = min(idx, range_size - 1)  # Clamp to valid range

            indices.append(start + idx)

        return indices

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities after training step."""
        with self._lock:
            for idx, td_error in zip(indices, td_errors):
                if 0 <= idx < self.total_capacity:
                    # BUG-007 FIX: Use consistent priority formula
                    if self.indices[idx]['is_demo']:
                        # Demos: scale by confidence * max_priority (same as add_demo)
                        confidence = self.indices[idx]['confidence']
                        priority = confidence * self.max_priority + self.demo_priority_boost
                    else:
                        # Agents: use TD-error as before
                        priority = np.abs(td_error) + 1e-6

                    self.indices[idx]['td_error'] = td_error
                    self._update_priority(int(idx), float(priority))

    # ==================== CURRICULUM MANAGEMENT ====================

    def clear(self):
        """Clear all demos, agent experience, and stored prices.

        TASK_135: Also clears internal episode price storage.
        Called on data regeneration (level switch, Bella generation, etc.)
        """
        with self._lock:
            old_demo_count = self.demo_count
            old_compressed_count = len(self._compressed_obs)

            self.demo_count = 0
            self.agent_count = 0
            self.agent_write_idx = self.demo_capacity
            self.sum_tree[:] = 0
            self.min_tree[:] = np.inf
            self.max_priority = 1.0
            self.indices[:] = 0

            # TASK_135: Clear internal episode price storage
            price_mb = self.get_price_storage_mb()
            ep_count = len(self._episode_prices)
            self._episode_prices.clear()
            self._next_episode_id = 0

            # TASK_136a FIX: Also clear compressed observations!
            compressed_mb = self.get_compressed_obs_mb()
            self._compressed_obs.clear()

            print(f"[ExpertBuffer] Cleared {old_demo_count} demos, {ep_count} episodes ({price_mb:.1f} MB prices, {compressed_mb:.1f} MB compressed)")

    def reset_for_level(self, new_level: int, keep_fraction: float = 0.0):
        """
        Reset buffer for new curriculum level.

        Args:
            new_level: The new curriculum level (0-14)
            keep_fraction: Fraction of demos to retain (0.0 = clear all)
        """
        with self._lock:
            if keep_fraction <= 0:
                self.clear()
                return

            if self.demo_count == 0:
                # No demos to keep
                self.agent_count = 0
                self.agent_write_idx = self.demo_capacity
                return

            # Keep top-performing demos based on episode_return
            keep_count = max(1, int(self.demo_count * keep_fraction))

            # Sort by episode_return descending
            demo_entries = self.indices[:self.demo_count]
            returns = demo_entries['episode_return'].astype(np.float32)
            keep_indices = np.argsort(returns)[-keep_count:]
            keep_indices = np.sort(keep_indices)  # Keep in order

            # Compact demos to front
            for new_idx, old_idx in enumerate(keep_indices):
                if new_idx != old_idx:
                    self.indices[new_idx] = self.indices[old_idx]

            # Update priorities for kept demos (BUG-007 FIX: use scaled formula)
            for idx in range(keep_count):
                confidence = float(self.indices[idx]['confidence'])
                priority = confidence * self.max_priority + self.demo_priority_boost
                self._update_priority(idx, priority)

            # Clear remaining demo slots
            for idx in range(keep_count, self.demo_count):
                self._update_priority(idx, 0.0)
                self.indices[idx] = 0

            self.demo_count = keep_count

            # Always clear agent buffer on level change
            self.agent_count = 0
            self.agent_write_idx = self.demo_capacity
            for idx in range(self.demo_capacity, self.total_capacity):
                self._update_priority(idx, 0.0)

            print(f"[ExpertBuffer] Reset for L{new_level}: kept {keep_count} demos, cleared agent buffer")

    # ==================== STATS & METRICS ====================

    def get_stats(self) -> Dict[str, float]:
        """Get buffer statistics for DS24 logging."""
        with self._lock:
            demo_entries = self.indices[:self.demo_count] if self.demo_count > 0 else None

            stats = {
                'expert/demo_count': self.demo_count,
                'expert/demo_capacity': self.demo_capacity,
                'expert/demo_fill_ratio': self.demo_count / self.demo_capacity,
                'expert/agent_count': self.agent_count,
                'expert/agent_capacity': self.agent_capacity,
                'expert/agent_fill_ratio': self.agent_count / self.agent_capacity if self.agent_capacity > 0 else 0,
                'expert/total_samples': self.demo_count + self.agent_count,
                'expert/per_beta': self.per_beta,
                'expert/max_priority': self.max_priority,
                'expert/memory_mb': (
                    self.indices.nbytes / 1e6 +
                    self.sum_tree.nbytes / 1e6 +
                    self.min_tree.nbytes / 1e6
                ),
                # TASK_135: Track episode price storage
                'expert/episode_count': len(self._episode_prices),
                'expert/price_storage_mb': self.get_price_storage_mb(),
            }

            if demo_entries is not None and len(demo_entries) > 0:
                stats['expert/avg_confidence'] = float(demo_entries['confidence'].mean())
                stats['expert/avg_return'] = float(demo_entries['episode_return'].mean())
                returns = demo_entries['episode_return'].astype(np.float32)
                stats['expert/return_p25'] = float(np.percentile(returns, 25))
                stats['expert/return_p50'] = float(np.percentile(returns, 50))
                stats['expert/return_p75'] = float(np.percentile(returns, 75))
            else:
                stats['expert/avg_confidence'] = 0.0
                stats['expert/avg_return'] = 0.0
                stats['expert/return_p25'] = 0.0
                stats['expert/return_p50'] = 0.0
                stats['expert/return_p75'] = 0.0

            return stats

    def compute_acceptance_rate(
        self,
        policy_actions: np.ndarray,
        batch_optimal_actions: np.ndarray,
        batch_is_demo: np.ndarray,
    ) -> float:
        """
        Compute expert acceptance rate: how often policy agrees with expert.

        Args:
            policy_actions: (batch,) argmax of policy logits
            batch_optimal_actions: (batch,) optimal actions from batch
            batch_is_demo: (batch,) whether sample is from demo buffer

        Returns:
            Acceptance rate (0-1) for demo samples
        """
        demo_mask = batch_is_demo.astype(bool)
        if demo_mask.sum() == 0:
            return 0.0

        matches = (policy_actions[demo_mask] == batch_optimal_actions[demo_mask])
        return float(matches.mean())


# Convenience function
def create_expert_buffer(
    n_assets: int = 64,
    demo_capacity: int = 50_000,
    agent_capacity: int = 500_000,
    window_size: int = 60,
) -> UnifiedIndexBuffer:
    """
    Create an expert buffer with default settings.

    TASK_135: Removed price_cache dependency - stores episode prices in RAM.

    Args:
        n_assets: Number of tradeable assets
        demo_capacity: Max demo samples
        agent_capacity: Max agent samples
        window_size: Observation window size

    Returns:
        Configured UnifiedIndexBuffer instance
    """
    return UnifiedIndexBuffer(
        n_assets=n_assets,
        demo_capacity=demo_capacity,
        agent_capacity=agent_capacity,
        window_size=window_size,
    )
