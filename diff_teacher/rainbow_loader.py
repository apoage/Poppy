"""
diff_teacher/rainbow_loader.py — RainbowTableLoader.

Extracted from diff_teacher/data_loader.py (TASK_772.3, 2026-05-23) as a
low-risk modularity split. The class is self-contained (sqlite3 + numpy +
training_schedule.json reading); shares only the `MarketWindow` dataclass
with `MarketDataLoader`, imported back from `.data_loader`.

data_loader.py keeps `MarketDataLoader` + `MarketWindow` and re-exports
`RainbowTableLoader` from this module so the `from .data_loader import
RainbowTableLoader` pattern keeps working.
"""

import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from .data_loader import MarketWindow

logger = logging.getLogger(__name__)


class RainbowTableLoader:
    """
    Data loader using rainbow table (training_schedule.json) for full coverage.

    Instead of requiring all 64 assets at every timestamp, this loader:
    1. Reads the pre-computed training schedule with asset rotations
    2. Builds an index of all possible (period, run, offset) windows
    3. Samples randomly from the full 96-year coverage

    Benefits:
    - 50M+ windows vs ~200 with strict 64-asset requirement
    - Automatic asset rotation (30-50% overlap between periods)
    - Full date coverage: 2018-2025
    - Benchmark periods excluded
    """

    DELISTED_ASSETS = {"EOS/USD", "LUNA/USD", "FTT/USD", "SRM/USD"}

    def __init__(
        self,
        db_path: str = "trading_data.db",
        schedule_path: str = "training_schedule.json",
        window_size: int = 400,
        windows_per_epoch: int = 50000,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize the rainbow table loader.

        Args:
            db_path: Path to SQLite database
            schedule_path: Path to training_schedule.json
            window_size: Steps per training window
            windows_per_epoch: How many windows to sample per epoch
            cache_dir: Optional directory for caching loaded periods
        """
        self.db_path = db_path
        self.schedule_path = schedule_path
        self.window_size = window_size
        self.windows_per_epoch = windows_per_epoch
        self.cache_dir = cache_dir

        # Load schedule
        self._load_schedule()

        # Build window index
        self._build_index()

        # Database connection (lazy loaded)
        self._conn = None

        # Cache for loaded period data
        self._period_cache: Dict[Tuple[int, int], np.ndarray] = {}
        self._cache_max_size = 5  # Keep max 5 periods in memory

        logger.info(
            f"RainbowTableLoader initialized: {self.total_windows:,} windows, "
            f"{len(self.schedule['periods'])} periods, "
            f"{self.schedule['summary']['total_training_years']:.1f} years"
        )

    def _load_schedule(self) -> None:
        """Load training schedule from JSON."""
        import json

        schedule_file = Path(self.schedule_path)
        if not schedule_file.exists():
            raise FileNotFoundError(
                f"Training schedule not found: {self.schedule_path}\n"
                f"Run: python tools/map_training_data.py"
            )

        with open(schedule_file) as f:
            self.schedule = json.load(f)

        logger.info(f"Loaded schedule: {self.schedule['summary']['total_periods']} periods")

    def _build_index(self) -> None:
        """
        Build index of all possible windows.

        Index entry: (period_idx, run_idx, window_offset)
        """
        self.index = []

        for period_idx, period in enumerate(self.schedule['periods']):
            duration_days = period['duration_days']
            steps_in_period = duration_days * 1440  # minutes
            windows_in_period = max(0, steps_in_period - self.window_size)

            for run_idx, run in enumerate(period['training_runs']):
                # Add all possible window offsets for this run
                # We don't add every offset (too many), just mark the range
                self.index.append({
                    'period_idx': period_idx,
                    'run_idx': run_idx,
                    'max_offset': windows_in_period,
                    'start_ts': period['start'],
                    'end_ts': period['end'],
                    'assets': run['assets'],
                })

        # Calculate total windows
        self.total_windows = sum(entry['max_offset'] for entry in self.index)

        # Build cumulative index for efficient sampling
        self._cumsum = np.cumsum([0] + [e['max_offset'] for e in self.index])

        logger.info(f"Built index: {len(self.index)} runs, {self.total_windows:,} total windows")

    def _get_conn(self) -> sqlite3.Connection:
        """Get database connection (lazy load)."""
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        return self._conn

    def _sample_window_location(
        self, rng: np.random.Generator
    ) -> Tuple[int, int, int, List[str]]:
        """
        Sample a random window location.

        Returns:
            (period_idx, run_idx, offset, assets)
        """
        # Sample a global window index
        global_idx = rng.integers(0, self.total_windows)

        # Find which run this falls into
        run_idx = np.searchsorted(self._cumsum, global_idx, side='right') - 1
        entry = self.index[run_idx]

        # Calculate offset within this run
        offset = global_idx - self._cumsum[run_idx]

        return (
            entry['period_idx'],
            entry['run_idx'],
            offset,
            entry['assets'],
        )

    def _load_period_data(
        self, period_idx: int, run_idx: int
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load data for a specific period/run combination.

        Returns:
            (data, timestamps, asset_symbols)
            data shape: (n_timestamps, n_assets, 5)
        """
        cache_key = (period_idx, run_idx)

        # Check cache
        if cache_key in self._period_cache:
            return self._period_cache[cache_key]

        # Load from database
        period = self.schedule['periods'][period_idx]
        run = period['training_runs'][run_idx]
        assets = run['assets']

        conn = self._get_conn()
        cursor = conn.cursor()

        # Get asset IDs
        asset_ids = {}
        for symbol in assets:
            if symbol in self.DELISTED_ASSETS:
                continue
            cursor.execute(
                "SELECT id FROM assets WHERE symbol = ?", (symbol,)
            )
            row = cursor.fetchone()
            if row:
                asset_ids[row[0]] = symbol

        if len(asset_ids) < 32:
            logger.warning(f"Period {period_idx} run {run_idx}: only {len(asset_ids)} assets found")

        # Load timestamps in range
        start_ts = period['start']
        end_ts = period['end']

        cursor.execute(
            "SELECT DISTINCT timestamp FROM ohlcv WHERE timestamp >= ? AND timestamp <= ? ORDER BY timestamp",
            (start_ts, end_ts)
        )
        timestamps = np.array([r[0] for r in cursor.fetchall()], dtype=np.int64)

        if len(timestamps) == 0:
            raise ValueError(f"No data for period {period_idx} ({start_ts} - {end_ts})")

        # Build data matrix
        ts_to_idx = {ts: i for i, ts in enumerate(timestamps)}
        id_to_col = {aid: i for i, aid in enumerate(asset_ids.keys())}

        n_assets = len(asset_ids)
        data = np.zeros((len(timestamps), n_assets, 5), dtype=np.float32)

        # Query OHLCV data
        asset_ids_str = ",".join(map(str, asset_ids.keys()))
        cursor.execute(f"""
            SELECT timestamp, asset_id, open, high, low, close, volume
            FROM ohlcv
            WHERE timestamp >= ? AND timestamp <= ?
            AND asset_id IN ({asset_ids_str})
        """, (start_ts, end_ts))

        for ts, asset_id, o, h, l, c, v in cursor:
            if ts in ts_to_idx and asset_id in id_to_col:
                row_idx = ts_to_idx[ts]
                col_idx = id_to_col[asset_id]
                data[row_idx, col_idx] = [o, h, l, c, v]

        asset_symbols = list(asset_ids.values())

        # Update cache (LRU-style)
        if len(self._period_cache) >= self._cache_max_size:
            # Remove oldest entry
            oldest_key = next(iter(self._period_cache))
            del self._period_cache[oldest_key]

        self._period_cache[cache_key] = (data, timestamps, asset_symbols)

        return data, timestamps, asset_symbols

    def get_batch_array(
        self, batch_size: int, rng: Optional[np.random.Generator] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Get a batch of windows sampled from the rainbow table.

        Args:
            batch_size: Number of windows
            rng: Random generator

        Returns:
            prices: (batch_size, window_size, n_assets, 5)
            timestamps: (batch_size, window_size)
            regimes: List of regime labels
        """
        if rng is None:
            rng = np.random.default_rng()

        prices_list = []
        timestamps_list = []
        regimes = []

        for _ in range(batch_size):
            period_idx, run_idx, offset, assets = self._sample_window_location(rng)

            # Load period data (cached)
            data, ts_arr, symbols = self._load_period_data(period_idx, run_idx)

            # Extract window
            max_start = len(data) - self.window_size
            start_idx = min(offset, max_start) if max_start > 0 else 0
            end_idx = start_idx + self.window_size

            window_data = data[start_idx:end_idx]
            window_ts = ts_arr[start_idx:end_idx]

            # Pad if necessary (shouldn't happen often)
            if window_data.shape[0] < self.window_size:
                pad_size = self.window_size - window_data.shape[0]
                window_data = np.pad(
                    window_data,
                    ((0, pad_size), (0, 0), (0, 0)),
                    mode='edge'
                )
                window_ts = np.pad(window_ts, (0, pad_size), mode='edge')

            # Ensure 64 assets (pad with zeros if fewer)
            if window_data.shape[1] < 64:
                pad_assets = 64 - window_data.shape[1]
                window_data = np.pad(
                    window_data,
                    ((0, 0), (0, pad_assets), (0, 0)),
                    mode='constant',
                    constant_values=0
                )
            elif window_data.shape[1] > 64:
                window_data = window_data[:, :64, :]

            prices_list.append(window_data)
            timestamps_list.append(window_ts)
            regimes.append(self._detect_regime(window_data))

        prices = np.stack(prices_list, axis=0)
        timestamps = np.stack(timestamps_list, axis=0)

        return prices, timestamps, regimes

    def _detect_regime(self, prices: np.ndarray) -> str:
        """Simple regime detection based on returns."""
        close = prices[:, :, 3]  # (window_size, n_assets)

        # Mask zero prices
        valid_mask = close > 0
        if not np.any(valid_mask):
            return "sideways"

        # Calculate returns only for valid prices
        returns = np.diff(close, axis=0) / (close[:-1] + 1e-10)
        returns = np.where(valid_mask[:-1], returns, 0)

        mean_return = np.mean(returns)
        volatility = np.std(returns)

        if mean_return < -0.001 and volatility > 0.01:
            return "crash"
        elif mean_return > 0.001 and volatility < 0.02:
            return "rally"
        elif volatility > 0.02:
            return "volatile"
        else:
            return "sideways"

    def normalize(self, prices: np.ndarray) -> np.ndarray:
        """
        Normalize prices for model input.

        Uses per-window normalization (first valid step = baseline).
        Handles missing assets (zeros) gracefully.
        """
        if prices.ndim == 3:
            prices = prices[np.newaxis]
            squeeze = True
        else:
            squeeze = False

        # Use first timestep as baseline
        baseline = prices[:, 0:1, :, :]

        # Handle zeros (missing assets): use 1.0 as baseline to avoid inf
        baseline = np.where(baseline > 0, baseline, 1.0)

        # Normalize: relative change from baseline
        normalized = np.where(prices > 0, prices / baseline - 1.0, 0.0)

        # Clip extreme values
        normalized = np.clip(normalized, -10.0, 10.0)

        if squeeze:
            normalized = normalized[0]

        return normalized.astype(np.float32)

    def iter_epoch(
        self, batch_size: int = 32, seed: int = 42
    ) -> Iterator[Tuple[np.ndarray, np.ndarray, List[str]]]:
        """
        Iterate over one epoch of sampled windows.

        Yields batches until windows_per_epoch is reached.
        """
        rng = np.random.default_rng(seed)
        n_batches = self.windows_per_epoch // batch_size

        for _ in range(n_batches):
            yield self.get_batch_array(batch_size, rng)

    def get_batch_streaming(
        self, batch_size: int, rng: Optional[np.random.Generator] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Get batch from currently loaded period (much faster).

        Call switch_period() to change the source period.
        """
        if rng is None:
            rng = np.random.default_rng()

        # Ensure we have a current period loaded
        if not hasattr(self, '_current_data') or self._current_data is None:
            self.switch_period(rng)

        data, timestamps, symbols = self._current_data

        # Sample windows from current period
        max_start = len(data) - self.window_size
        if max_start <= 0:
            # Period too short, switch to another
            self.switch_period(rng)
            data, timestamps, symbols = self._current_data
            max_start = len(data) - self.window_size

        prices_list = []
        timestamps_list = []
        regimes = []

        for _ in range(batch_size):
            start_idx = rng.integers(0, max(1, max_start))
            end_idx = start_idx + self.window_size

            window_data = data[start_idx:end_idx]
            window_ts = timestamps[start_idx:end_idx]

            # Pad if necessary
            if window_data.shape[0] < self.window_size:
                pad_size = self.window_size - window_data.shape[0]
                window_data = np.pad(window_data, ((0, pad_size), (0, 0), (0, 0)), mode='edge')
                window_ts = np.pad(window_ts, (0, pad_size), mode='edge')

            # Ensure 64 assets
            if window_data.shape[1] < 64:
                pad_assets = 64 - window_data.shape[1]
                window_data = np.pad(window_data, ((0, 0), (0, pad_assets), (0, 0)), mode='constant')
            elif window_data.shape[1] > 64:
                window_data = window_data[:, :64, :]

            prices_list.append(window_data)
            timestamps_list.append(window_ts)
            regimes.append(self._detect_regime(window_data))

        return np.stack(prices_list), np.stack(timestamps_list), regimes

    def switch_period(self, rng: Optional[np.random.Generator] = None) -> None:
        """Switch to a random period/run for streaming mode."""
        if rng is None:
            rng = np.random.default_rng()

        # Pick random entry
        entry_idx = rng.integers(0, len(self.index))
        entry = self.index[entry_idx]

        logger.info(
            f"Switching to period {entry['period_idx']}, run {entry['run_idx']} "
            f"({entry['max_offset']:,} windows)"
        )

        # Load the period
        self._current_data = self._load_period_data(entry['period_idx'], entry['run_idx'])
        self._current_entry = entry

    def iter_epoch_streaming(
        self,
        batch_size: int = 32,
        batches_per_period: int = 100,
        seed: int = 42,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray, List[str]]]:
        """
        Iterate over one epoch using streaming mode (faster).

        Loads one period at a time, samples many batches from it,
        then switches to a new period.

        Args:
            batch_size: Windows per batch
            batches_per_period: How many batches before switching periods
            seed: Random seed
        """
        rng = np.random.default_rng(seed)
        n_batches = self.windows_per_epoch // batch_size
        batches_yielded = 0

        while batches_yielded < n_batches:
            # Switch to new period
            self.switch_period(rng)

            # Yield batches from this period
            for _ in range(min(batches_per_period, n_batches - batches_yielded)):
                yield self.get_batch_streaming(batch_size, rng)
                batches_yielded += 1

    @property
    def n_windows(self) -> int:
        """Number of windows per epoch (sampled)."""
        return self.windows_per_epoch

    def __len__(self) -> int:
        return self.windows_per_epoch

    def close(self) -> None:
        """Close database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
