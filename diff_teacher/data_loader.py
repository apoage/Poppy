"""
Data Loader for Diffusion Teacher.

Reads market windows from trading_data.db for training the generator.
Completely independent from mcts_trader data loading.
"""

import sqlite3
import logging
import numpy as np
from typing import Iterator, Tuple, Optional, Dict, List
from dataclasses import dataclass
from pathlib import Path
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MarketWindow:
    """A single market window for training."""

    prices: np.ndarray  # Shape: (window_size, n_assets, 5) - OHLCV
    timestamps: np.ndarray  # Shape: (window_size,)
    asset_symbols: List[str]  # Asset names for reference
    start_idx: int  # Index in full dataset
    regime_label: Optional[str] = None  # Auto-detected regime


class MarketDataLoader:
    """
    Loads market windows from SQLite database.

    Designed for training diffusion models - provides random windows
    with optional regime labels for conditioning.

    Warning:
        This class is NOT thread-safe. The internal cache (_period_cache)
        uses a simple dict without locking. Do not share instances between
        threads. For multi-threaded loading, create separate instances per
        thread or use external synchronization.
        (HIGH-015: Thread safety documented)
    """

    DELISTED_ASSETS = {"EOS/USD", "LUNA/USD", "FTT/USD", "SRM/USD"}

    def __init__(
        self,
        db_path: str = "trading_data.db",
        n_assets: int = 64,
        window_size: int = 400,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        cache_path: Optional[str] = None,
        min_required_assets: int = 50,
    ):
        """
        Initialize the data loader.

        Args:
            db_path: Path to SQLite database
            n_assets: Number of assets to load (top by data count)
            window_size: Steps per training window
            start_date: Filter start (YYYY-MM-DD or timestamp)
            end_date: Filter end (YYYY-MM-DD or timestamp)
            cache_path: Optional numpy cache for faster loading
            min_required_assets: Minimum assets required per timestamp
        """
        self.db_path = db_path
        self.n_assets = n_assets
        self.window_size = window_size
        self.min_required_assets = min_required_assets

        # Parse date filters
        self.filter_start_ts, self.filter_end_ts = self._parse_dates(
            start_date, end_date
        )

        # Try loading from cache
        if cache_path and self._try_load_cache(cache_path):
            logger.info("Loaded from cache successfully")
            return

        # Connect to database
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()

        # Load assets and data
        self._load_assets()
        self._load_timestamps()
        self._load_data_matrix()

        # Save cache if path specified
        if cache_path:
            self._save_cache(cache_path)

    def _parse_dates(
        self, start_date: Optional[str], end_date: Optional[str]
    ) -> Tuple[int, int]:
        """Parse date strings to timestamps."""
        from datetime import datetime

        start_ts = 0
        end_ts = 9999999999

        if start_date:
            if isinstance(start_date, int) or str(start_date).isdigit():
                start_ts = int(start_date)
            else:
                try:
                    start_ts = int(
                        datetime.strptime(str(start_date), "%Y-%m-%d").timestamp()
                    )
                except ValueError:
                    logger.warning(f"Could not parse start_date: {start_date}")

        if end_date:
            if isinstance(end_date, int) or str(end_date).isdigit():
                end_ts = int(end_date)
            else:
                try:
                    end_ts = int(
                        datetime.strptime(str(end_date), "%Y-%m-%d").timestamp()
                    )
                except ValueError:
                    logger.warning(f"Could not parse end_date: {end_date}")

        return start_ts, end_ts

    def _load_assets(self) -> None:
        """Load top assets by data count."""
        query = """
            SELECT a.id, a.symbol, COUNT(o.timestamp) as cnt
            FROM assets a
            JOIN ohlcv o ON a.id = o.asset_id
            WHERE a.is_active = 1
            GROUP BY a.id
            ORDER BY cnt DESC
            LIMIT ?
        """
        self.cursor.execute(query, (self.n_assets + 20,))
        rows = self.cursor.fetchall()

        self.assets: Dict[int, str] = {}
        for asset_id, symbol, _ in rows:
            if symbol in self.DELISTED_ASSETS:
                continue
            self.assets[asset_id] = symbol
            if len(self.assets) >= self.n_assets:
                break

        self.asset_ids = list(self.assets.keys())
        self.asset_symbols = [self.assets[aid] for aid in self.asset_ids]
        self.id_to_col = {aid: i for i, aid in enumerate(self.asset_ids)}

        logger.info(f"Loaded {len(self.assets)} assets")

    def _load_timestamps(self) -> None:
        """Load valid timestamps where enough assets have data."""
        asset_ids_str = ",".join(map(str, self.asset_ids))

        query = f"""
            SELECT timestamp
            FROM ohlcv
            WHERE asset_id IN ({asset_ids_str})
            AND timestamp >= ? AND timestamp <= ?
            GROUP BY timestamp
            HAVING COUNT(DISTINCT asset_id) >= ?
            ORDER BY timestamp ASC
        """
        self.cursor.execute(
            query,
            (self.filter_start_ts, self.filter_end_ts, self.min_required_assets),
        )
        rows = self.cursor.fetchall()

        self.timestamps = np.array([r[0] for r in rows], dtype=np.int64)
        self.ts_to_idx = {ts: i for i, ts in enumerate(self.timestamps)}

        logger.info(f"Loaded {len(self.timestamps):,} valid timestamps")

    def _load_data_matrix(self) -> None:
        """Load all OHLCV data into memory."""
        logger.info("Loading data into RAM...")
        t0 = time.time()

        n_timestamps = len(self.timestamps)
        n_assets = len(self.asset_ids)

        # Matrix: (timestamps, assets, 5) for OHLCV
        self.data = np.zeros((n_timestamps, n_assets, 5), dtype=np.float32)

        # Query all data in range
        asset_ids_str = ",".join(map(str, self.asset_ids))
        start_ts = self.timestamps[0]
        end_ts = self.timestamps[-1]

        query = f"""
            SELECT timestamp, asset_id, open, high, low, close, volume
            FROM ohlcv
            WHERE timestamp >= ? AND timestamp <= ?
            AND asset_id IN ({asset_ids_str})
        """
        self.cursor.execute(query, (int(start_ts), int(end_ts)))

        rows_loaded = 0
        for ts, asset_id, open_, high, low, close, volume in self.cursor:
            if ts in self.ts_to_idx and asset_id in self.id_to_col:
                row_idx = self.ts_to_idx[ts]
                col_idx = self.id_to_col[asset_id]
                self.data[row_idx, col_idx] = [open_, high, low, close, volume]
                rows_loaded += 1

        elapsed = time.time() - t0
        logger.info(
            f"Loaded {rows_loaded:,} data points in {elapsed:.1f}s "
            f"({rows_loaded / elapsed / 1e6:.2f}M rows/sec)"
        )

    def _try_load_cache(self, cache_path: str) -> bool:
        """Try to load from numpy cache."""
        if not Path(cache_path).exists():
            return False

        try:
            t0 = time.time()
            cache = np.load(cache_path, allow_pickle=True)

            self.data = cache["data"]
            self.timestamps = cache["timestamps"]
            self.asset_ids = cache["asset_ids"].tolist()
            self.asset_symbols = cache["asset_symbols"].tolist()

            # Rebuild indices
            self.ts_to_idx = {ts: i for i, ts in enumerate(self.timestamps)}
            self.id_to_col = {aid: i for i, aid in enumerate(self.asset_ids)}

            logger.info(f"Loaded cache in {time.time() - t0:.1f}s")
            return True
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return False

    def _save_cache(self, cache_path: str) -> None:
        """Save data to numpy cache."""
        try:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_path,
                data=self.data,
                timestamps=self.timestamps,
                asset_ids=np.array(self.asset_ids),
                asset_symbols=np.array(self.asset_symbols),
            )
            size_mb = Path(cache_path).stat().st_size / (1024**2)
            logger.info(f"Saved cache ({size_mb:.1f} MB)")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    @property
    def n_windows(self) -> int:
        """Number of possible windows in dataset."""
        return max(0, len(self.timestamps) - self.window_size + 1)

    def get_window(self, start_idx: int) -> MarketWindow:
        """Get a specific window by start index."""
        if start_idx < 0 or start_idx >= self.n_windows:
            raise IndexError(f"Window index {start_idx} out of range [0, {self.n_windows})")

        end_idx = start_idx + self.window_size
        prices = self.data[start_idx:end_idx].copy()
        timestamps = self.timestamps[start_idx:end_idx].copy()

        # Detect regime (simple heuristic)
        regime = self._detect_regime(prices)

        return MarketWindow(
            prices=prices,
            timestamps=timestamps,
            asset_symbols=self.asset_symbols,
            start_idx=start_idx,
            regime_label=regime,
        )

    def _detect_regime(self, prices: np.ndarray) -> str:
        """
        Simple regime detection based on returns.

        Args:
            prices: (window_size, n_assets, 5) OHLCV data

        Returns:
            One of: "crash", "rally", "sideways", "volatile"
        """
        # Use close prices
        close = prices[:, :, 3]  # (window_size, n_assets)

        # Calculate returns
        returns = np.diff(close, axis=0) / (close[:-1] + 1e-10)
        mean_return = np.mean(returns)
        volatility = np.std(returns)

        # Thresholds (per-step returns)
        if mean_return < -0.001 and volatility > 0.01:
            return "crash"
        elif mean_return > 0.001 and volatility < 0.02:
            return "rally"
        elif volatility > 0.02:
            return "volatile"
        else:
            return "sideways"

    def random_window(self, rng: Optional[np.random.Generator] = None) -> MarketWindow:
        """Get a random window."""
        if rng is None:
            rng = np.random.default_rng()
        idx = rng.integers(0, self.n_windows)
        return self.get_window(idx)

    def iter_windows(
        self, batch_size: int = 32, shuffle: bool = True, seed: int = 42
    ) -> Iterator[List[MarketWindow]]:
        """
        Iterate over windows in batches.

        Args:
            batch_size: Windows per batch
            shuffle: Shuffle window order
            seed: Random seed for shuffling

        Yields:
            List of MarketWindow objects
        """
        indices = np.arange(self.n_windows)

        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(indices)

        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i : i + batch_size]
            yield [self.get_window(idx) for idx in batch_indices]

    def get_batch_array(
        self, batch_size: int, rng: Optional[np.random.Generator] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Get a batch as numpy arrays (more efficient for JAX).

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

        indices = rng.integers(0, self.n_windows, size=batch_size)

        prices = np.stack(
            [self.data[i : i + self.window_size] for i in indices], axis=0
        )
        timestamps = np.stack(
            [self.timestamps[i : i + self.window_size] for i in indices], axis=0
        )
        regimes = [self._detect_regime(prices[b]) for b in range(batch_size)]

        return prices, timestamps, regimes

    def normalize(self, prices: np.ndarray) -> np.ndarray:
        """
        Normalize prices for model input.

        Uses per-window normalization (first step = baseline).

        Args:
            prices: (batch, window_size, n_assets, 5) or (window_size, n_assets, 5)

        Returns:
            Normalized prices (returns relative to first step)
        """
        if prices.ndim == 3:
            prices = prices[np.newaxis]
            squeeze = True
        else:
            squeeze = False

        # Use first timestep as baseline
        baseline = prices[:, 0:1, :, :]  # (batch, 1, n_assets, 5)

        # Avoid division by zero (especially for volume)
        baseline = np.maximum(baseline, 1e-10)

        # Normalize: relative change from baseline
        normalized = prices / baseline - 1.0

        # Clip extreme values
        normalized = np.clip(normalized, -10.0, 10.0)

        if squeeze:
            normalized = normalized[0]

        return normalized.astype(np.float32)

    def denormalize(
        self, normalized: np.ndarray, baseline: np.ndarray
    ) -> np.ndarray:
        """
        Denormalize model output back to prices.

        Args:
            normalized: Normalized values
            baseline: Original first-step values

        Returns:
            Reconstructed prices
        """
        return (normalized + 1.0) * baseline

    def close(self) -> None:
        """Close database connection."""
        if hasattr(self, "conn"):
            self.conn.close()

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, idx: int) -> MarketWindow:
        return self.get_window(idx)


# =============================================================================
# Rainbow Table Loader - Uses training_schedule.json for full coverage
# =============================================================================


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


# =============================================================================
# Standalone test
# =============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", default="trading_data.db")
    parser.add_argument("--n-assets", type=int, default=64)
    parser.add_argument("--window-size", type=int, default=400)
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--end-date", default="2024-12-31")
    parser.add_argument("--rainbow", action="store_true", help="Test RainbowTableLoader")
    parser.add_argument("--schedule-path", default="training_schedule.json")
    args = parser.parse_args()

    if args.rainbow:
        # Test RainbowTableLoader
        print("=" * 60)
        print("RAINBOW TABLE LOADER TEST")
        print("=" * 60)

        loader = RainbowTableLoader(
            db_path=args.db_path,
            schedule_path=args.schedule_path,
            window_size=args.window_size,
            windows_per_epoch=1000,  # Small for testing
        )

        print(f"\nTotal windows available: {loader.total_windows:,}")
        print(f"Windows per epoch: {loader.windows_per_epoch:,}")
        print(f"Periods: {len(loader.schedule['periods'])}")
        print(f"Training runs: {len(loader.index)}")

        # Test batch loading
        print("\nLoading test batch...")
        t0 = time.time()
        rng = np.random.default_rng(42)
        prices, timestamps, regimes = loader.get_batch_array(batch_size=8, rng=rng)
        elapsed = time.time() - t0

        print(f"  Batch loaded in {elapsed:.2f}s")
        print(f"  Prices shape: {prices.shape}")
        print(f"  Regimes: {regimes}")

        # Check for zeros (missing data)
        zero_pct = (prices == 0).sum() / prices.size * 100
        print(f"  Zero values: {zero_pct:.1f}%")

        # Test normalization
        normalized = loader.normalize(prices)
        print(f"\nNormalized:")
        print(f"  Range: {normalized.min():.3f} to {normalized.max():.3f}")

        # Test multiple batches (check different periods)
        print("\nLoading 5 more batches to test diversity...")
        for i in range(5):
            prices, _, regimes = loader.get_batch_array(batch_size=4, rng=rng)
            print(f"  Batch {i+1}: regimes={regimes}")

        # Speed benchmark
        print("\nSpeed benchmark (100 batches of 32)...")
        t0 = time.time()
        for _ in range(100):
            loader.get_batch_array(batch_size=32, rng=rng)
        elapsed = time.time() - t0
        print(f"  100 batches in {elapsed:.1f}s ({3200/elapsed:.0f} windows/sec)")

        loader.close()
        print("\nRainbow table loader test complete!")

    else:
        # Test original MarketDataLoader
        loader = MarketDataLoader(
            db_path=args.db_path,
            n_assets=args.n_assets,
            window_size=args.window_size,
            start_date=args.start_date,
            end_date=args.end_date,
        )

        print(f"\nDataset: {len(loader)} windows")
        print(f"Assets: {loader.asset_symbols[:5]}...")
        print(f"Time range: {loader.timestamps[0]} - {loader.timestamps[-1]}")

        # Test random window
        window = loader.random_window()
        print(f"\nRandom window:")
        print(f"  Shape: {window.prices.shape}")
        print(f"  Regime: {window.regime_label}")
        print(f"  Price range: {window.prices[:, :, 3].min():.2f} - {window.prices[:, :, 3].max():.2f}")

        # Test batch
        prices, timestamps, regimes = loader.get_batch_array(batch_size=8)
        print(f"\nBatch:")
        print(f"  Prices shape: {prices.shape}")
        print(f"  Regimes: {regimes}")

        # Test normalization
        normalized = loader.normalize(prices)
        print(f"\nNormalized:")
        print(f"  Range: {normalized.min():.3f} to {normalized.max():.3f}")

        loader.close()
