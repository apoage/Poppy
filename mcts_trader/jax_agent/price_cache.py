"""
TASK_125: Memory-mapped Price Cache for Causal Expert Imprinting.

Stores raw price data on disk, accessed via memory-mapping.
Only actively-used pages loaded into RAM by OS.

Memory usage: ~12 GB on disk, minimal RAM footprint.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import threading
import os


class PriceCache:
    """
    Memory-mapped price data for on-demand observation reconstruction.

    Stores raw OHLCV data on disk, accessed like array.
    OS handles paging - only actively-used pages loaded into RAM.

    Usage:
        cache = PriceCache(Path("./price_cache"))
        episode_id = cache.add_episode(prices, start_date=1704067200)
        window = cache.get_observation_window(episode_id, step=100, window_size=60)
    """

    def __init__(
        self,
        cache_path: Path,
        max_episodes: int = 10_000,
        max_steps: int = 1000,
        n_assets: int = 64,
        n_features: int = 5,  # OHLCV
        readonly: bool = False,
    ):
        """
        Initialize price cache.

        Args:
            cache_path: Directory for cache files
            max_episodes: Maximum episodes to store
            max_steps: Maximum steps per episode
            n_assets: Number of tradeable assets
            n_features: Features per asset (5 = OHLCV)
            readonly: If True, open existing cache read-only
        """
        self.cache_path = Path(cache_path)
        self.max_episodes = max_episodes
        self.max_steps = max_steps
        self.n_assets = n_assets
        self.n_features = n_features

        # Create cache directory
        self.cache_path.mkdir(parents=True, exist_ok=True)

        # Shape: (episodes, steps, assets, features)
        self.shape = (max_episodes, max_steps, n_assets, n_features)
        self._bytes_per_episode = max_steps * n_assets * n_features * 4  # float32

        # Memory-map the price data file
        prices_file = self.cache_path / 'prices.dat'
        mode = 'r' if readonly else ('r+' if prices_file.exists() else 'w+')

        self.prices = np.memmap(
            prices_file,
            dtype=np.float32,
            mode=mode,
            shape=self.shape
        )

        # Episode metadata (small, kept in RAM)
        metadata_file = self.cache_path / 'metadata.npz'
        if metadata_file.exists() and readonly:
            data = np.load(metadata_file)
            self.episode_lengths = data['lengths'].copy()
            self.episode_start_dates = data['start_dates'].copy()
            self.episode_regimes = data['regimes'].copy()
            self.next_episode_id = int(data['next_id'])
        else:
            self.episode_lengths = np.zeros(max_episodes, dtype=np.uint16)
            self.episode_start_dates = np.zeros(max_episodes, dtype=np.uint32)
            self.episode_regimes = np.zeros(max_episodes, dtype=np.uint8)
            self.next_episode_id = 0

        # TASK_128 FIX: Use RLock (reentrant lock) to prevent deadlock
        # when add_episode calls flush() while holding the lock
        self._lock = threading.RLock()
        self._flush_counter = 0

    def add_episode(
        self,
        prices: np.ndarray,
        start_date: int = 0,
        regime_tag: int = 0,
    ) -> int:
        """
        Add episode price data to cache.

        Args:
            prices: (T, n_assets, n_features) price data [OHLCV]
            start_date: Unix timestamp of episode start
            regime_tag: Market regime identifier (0-7)

        Returns:
            episode_id for referencing this data
        """
        with self._lock:
            if self.next_episode_id >= self.max_episodes:
                raise RuntimeError(
                    f"Price cache full ({self.max_episodes} episodes). "
                    "Increase max_episodes or clear old data."
                )

            episode_id = self.next_episode_id
            T = min(prices.shape[0], self.max_steps)

            # Validate shape
            if prices.shape[1] != self.n_assets or prices.shape[2] != self.n_features:
                raise ValueError(
                    f"Expected prices shape (T, {self.n_assets}, {self.n_features}), "
                    f"got {prices.shape}"
                )

            # Store price data
            self.prices[episode_id, :T] = prices[:T]
            self.episode_lengths[episode_id] = T
            self.episode_start_dates[episode_id] = start_date
            self.episode_regimes[episode_id] = regime_tag

            self.next_episode_id += 1
            self._flush_counter += 1

            # Flush to disk periodically
            if self._flush_counter >= 100:
                self.flush()
                self._flush_counter = 0

            return episode_id

    def get_observation_window(
        self,
        episode_id: int,
        step: int,
        window_size: int = 60,
    ) -> np.ndarray:
        """
        Reconstruct observation window from price cache.

        IMPORTANT: Window covers [step - window_size, step - 1] to maintain causality.
        Agent at step t should NOT see price at step t.

        Args:
            episode_id: Episode index
            step: Current step (will get window BEFORE this step)
            window_size: Number of historical steps

        Returns:
            (window_size, n_assets, n_features) array
        """
        episode_len = int(self.episode_lengths[episode_id])

        # Causal window: [step - window_size, step - 1]
        # At step t, agent sees prices up to t-1
        end_idx = min(step, episode_len)
        start_idx = max(0, end_idx - window_size)

        window = self.prices[episode_id, start_idx:end_idx].copy()

        # Pad if near episode start
        actual_len = window.shape[0]
        if actual_len < window_size:
            pad_size = window_size - actual_len
            if actual_len > 0:
                # Repeat first observation for padding
                padding = np.tile(window[0:1], (pad_size, 1, 1))
                window = np.concatenate([padding, window], axis=0)
            else:
                # Edge case: step=0, return zeros
                window = np.zeros((window_size, self.n_assets, self.n_features), dtype=np.float32)

        return window

    def get_batch_observations(
        self,
        episode_ids: np.ndarray,
        steps: np.ndarray,
        window_size: int = 60,
    ) -> np.ndarray:
        """
        Vectorized batch observation reconstruction.

        Args:
            episode_ids: (batch_size,) episode indices
            steps: (batch_size,) step indices

        Returns:
            (batch_size, window_size, n_assets, n_features) array
        """
        batch_size = len(episode_ids)
        obs = np.zeros(
            (batch_size, window_size, self.n_assets, self.n_features),
            dtype=np.float32
        )

        for i, (ep_id, step) in enumerate(zip(episode_ids, steps)):
            obs[i] = self.get_observation_window(int(ep_id), int(step), window_size)

        return obs

    def get_returns(
        self,
        episode_id: int,
        start_step: int = 0,
        end_step: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute log returns for an episode (for optimal path computation).

        Args:
            episode_id: Episode index
            start_step: Start step (inclusive)
            end_step: End step (exclusive), None = episode end

        Returns:
            (T-1, n_assets) log returns
        """
        episode_len = int(self.episode_lengths[episode_id])
        if end_step is None:
            end_step = episode_len

        # Get close prices (index 3 in OHLCV)
        close_prices = self.prices[episode_id, start_step:end_step, :, 3]

        # Compute log returns
        log_returns = np.log(close_prices[1:] / np.maximum(close_prices[:-1], 1e-8))

        return log_returns

    def get_episode_info(self, episode_id: int) -> dict:
        """Get metadata for an episode."""
        return {
            'episode_id': episode_id,
            'length': int(self.episode_lengths[episode_id]),
            'start_date': int(self.episode_start_dates[episode_id]),
            'regime': int(self.episode_regimes[episode_id]),
        }

    def get_episode_prices(self, episode_id: int) -> Optional[np.ndarray]:
        """
        Get full price data for an episode.

        Args:
            episode_id: Episode index

        Returns:
            (T, n_assets, n_features) array or None if invalid
        """
        if episode_id < 0 or episode_id >= self.next_episode_id:
            return None

        length = int(self.episode_lengths[episode_id])
        if length == 0:
            return None

        return np.array(self.prices[episode_id, :length])

    def flush(self):
        """Flush memory-mapped data to disk."""
        with self._lock:
            self.prices.flush()

            # Save metadata
            np.savez(
                self.cache_path / 'metadata.npz',
                lengths=self.episode_lengths,
                start_dates=self.episode_start_dates,
                regimes=self.episode_regimes,
                next_id=self.next_episode_id,
            )

    def get_stats(self) -> dict:
        """Get cache statistics."""
        total_bytes = self.next_episode_id * self._bytes_per_episode
        return {
            'num_episodes': self.next_episode_id,
            'max_episodes': self.max_episodes,
            'fill_ratio': self.next_episode_id / self.max_episodes,
            'disk_usage_gb': total_bytes / 1e9,
            'avg_episode_length': float(self.episode_lengths[:self.next_episode_id].mean()) if self.next_episode_id > 0 else 0,
        }

    def clear(self):
        """Clear all cached data."""
        with self._lock:
            self.prices[:] = 0
            self.episode_lengths[:] = 0
            self.episode_start_dates[:] = 0
            self.episode_regimes[:] = 0
            self.next_episode_id = 0
            self.flush()

    def __del__(self):
        """Ensure data is flushed on deletion."""
        try:
            self.flush()
        except Exception:
            pass  # Ignore errors during cleanup


# Convenience function
def create_price_cache(
    cache_dir: str = "./.price_cache",
    max_episodes: int = 10_000,
    n_assets: int = 64,
) -> PriceCache:
    """
    Create a PriceCache with default settings for trading.

    Args:
        cache_dir: Directory for cache files
        max_episodes: Maximum number of episodes
        n_assets: Number of tradeable assets

    Returns:
        Configured PriceCache instance
    """
    return PriceCache(
        cache_path=Path(cache_dir),
        max_episodes=max_episodes,
        max_steps=1000,
        n_assets=n_assets,
        n_features=5,  # OHLCV
    )
