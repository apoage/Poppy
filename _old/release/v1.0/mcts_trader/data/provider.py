"""
Abstract data provider for training and backtesting.

Supports multiple data sources:
- SQLite database (historical data)
- CSV files
- Live exchange data
- Synthetic data generation
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Iterator, Tuple
import numpy as np
import sqlite3
import logging
from datetime import datetime


@dataclass
class MarketSnapshot:
    """Single point-in-time market data for all assets"""
    timestamp: int
    data: Dict[str, Dict]  # {symbol: {open, high, low, close, volume}}

    def __len__(self) -> int:
        return len(self.data)


class DataProvider(ABC):
    """Abstract base class for market data providers"""

    @abstractmethod
    def get_assets(self) -> List[str]:
        """Get list of available asset symbols"""
        pass

    @abstractmethod
    def get_time_range(self) -> Tuple[int, int]:
        """Get (min_timestamp, max_timestamp)"""
        pass

    @abstractmethod
    def get_snapshot(self, timestamp: int) -> MarketSnapshot:
        """Get market data at specific timestamp"""
        pass

    @abstractmethod
    def iter_snapshots(
        self,
        start: Optional[int] = None,
        end: Optional[int] = None,
        step: int = 60
    ) -> Iterator[MarketSnapshot]:
        """Iterate through market snapshots"""
        pass

    @abstractmethod
    def get_valid_timestamps(self, min_assets: int = 10) -> np.ndarray:
        """Get timestamps where enough assets have data"""
        pass


class SQLiteDataProvider(DataProvider):
    """
    Provides market data from SQLite database.

    Optimized for training with pre-computed timestamp filtering.
    """

    def __init__(
        self,
        db_path: str = "trading_data.db",
        n_assets: int = 64,
        min_coverage: int = 10,
        cache_timestamps: bool = True
    ):
        self.db_path = db_path
        self.n_assets = n_assets
        self.min_coverage = min_coverage

        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()

        # Load top assets by data coverage
        self.assets = self._load_top_assets(n_assets)
        self.asset_ids = list(self.assets.keys())
        self.id_to_symbol = self.assets
        self.symbol_to_id = {v: k for k, v in self.assets.items()}

        # Time range
        self.min_time, self.max_time = self._get_time_range()

        # Pre-compute valid timestamps (one-time cost)
        if cache_timestamps:
            logging.info(f"Building timestamp index (min {min_coverage} assets)...")
            self.valid_timestamps = self._build_valid_timestamps()
            logging.info(f"Indexed {len(self.valid_timestamps):,} valid timestamps")
        else:
            self.valid_timestamps = None

    def _load_top_assets(self, n: int) -> Dict[int, str]:
        """Load top assets by data point count"""
        query = """
            SELECT a.id, a.symbol, COUNT(o.timestamp) as cnt
            FROM assets a
            JOIN ohlcv o ON a.id = o.asset_id
            GROUP BY a.id
            ORDER BY cnt DESC
            LIMIT ?
        """
        self.cursor.execute(query, (n,))
        return {r[0]: r[1] for r in self.cursor.fetchall()}

    def _get_time_range(self) -> Tuple[int, int]:
        """Get global time range"""
        self.cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM ohlcv")
        return self.cursor.fetchone()

    def _build_valid_timestamps(self) -> np.ndarray:
        """
        Pre-compute timestamps where >= min_coverage of our assets have data.

        This is the key optimization for training - one-time O(n) cost,
        then O(log n) lookups during training.
        """
        asset_ids_str = ','.join(map(str, self.asset_ids))

        query = f"""
            SELECT timestamp
            FROM ohlcv
            WHERE asset_id IN ({asset_ids_str})
            GROUP BY timestamp
            HAVING COUNT(DISTINCT asset_id) >= ?
            ORDER BY timestamp ASC
        """

        self.cursor.execute(query, (self.min_coverage,))
        rows = self.cursor.fetchall()

        if not rows:
            logging.warning(f"No timestamps with {self.min_coverage}+ assets. Using all.")
            self.cursor.execute("SELECT DISTINCT timestamp FROM ohlcv ORDER BY timestamp")
            rows = self.cursor.fetchall()

        return np.array([r[0] for r in rows], dtype=np.int64)

    def get_assets(self) -> List[str]:
        return list(self.assets.values())

    def get_time_range(self) -> Tuple[int, int]:
        return self.min_time, self.max_time

    def get_snapshot(self, timestamp: int) -> MarketSnapshot:
        """Get market data at timestamp"""
        asset_ids_str = ','.join(map(str, self.asset_ids))

        # Direct string substitution (parameterized queries had issues)
        query = f"""
            SELECT asset_id, open, high, low, close, volume
            FROM ohlcv
            WHERE timestamp = {timestamp} AND asset_id IN ({asset_ids_str})
        """
        self.cursor.execute(query)
        rows = self.cursor.fetchall()

        data = {}
        for r in rows:
            asset_id = r[0]
            if asset_id in self.id_to_symbol:
                data[self.id_to_symbol[asset_id]] = {
                    'open': float(r[1]) if r[1] else 0.0,
                    'high': float(r[2]) if r[2] else 0.0,
                    'low': float(r[3]) if r[3] else 0.0,
                    'close': float(r[4]) if r[4] else 0.0,
                    'volume': float(r[5]) if r[5] else 0.0,
                }

        return MarketSnapshot(timestamp=timestamp, data=data)

    def iter_snapshots(
        self,
        start: Optional[int] = None,
        end: Optional[int] = None,
        step: int = 60
    ) -> Iterator[MarketSnapshot]:
        """Iterate through valid timestamps"""
        if self.valid_timestamps is None:
            raise RuntimeError("Timestamps not cached. Set cache_timestamps=True")

        # Find start/end indices
        ts = self.valid_timestamps

        if start is not None:
            start_idx = np.searchsorted(ts, start, side='left')
        else:
            start_idx = 0

        if end is not None:
            end_idx = np.searchsorted(ts, end, side='right')
        else:
            end_idx = len(ts)

        # Iterate
        for i in range(start_idx, end_idx):
            yield self.get_snapshot(int(ts[i]))

    def get_valid_timestamps(self, min_assets: int = None) -> np.ndarray:
        if self.valid_timestamps is not None:
            return self.valid_timestamps

        min_assets = min_assets or self.min_coverage
        return self._build_valid_timestamps()

    def get_random_start(self, lookahead: int = 300, lookback: int = 60) -> int:
        """
        Get random valid starting timestamp with enough lookahead.

        Args:
            lookahead: Minimum steps available after start
            lookback: Buffer to skip from beginning (for encoder warmup)
        """
        if self.valid_timestamps is None:
            raise RuntimeError("Timestamps not cached")

        ts = self.valid_timestamps

        # Reserve space for lookahead and lookback
        max_idx = len(ts) - lookahead
        min_idx = lookback

        if max_idx <= min_idx:
            return int(ts[min(min_idx, len(ts) - 1)])

        idx = np.random.randint(min_idx, max_idx)
        return int(ts[idx])

    def get_next_timestamp(self, current: int) -> Optional[int]:
        """Get next valid timestamp after current (O(log n))"""
        if self.valid_timestamps is None:
            raise RuntimeError("Timestamps not cached")

        next_possible = current + 60  # 1 minute
        idx = np.searchsorted(self.valid_timestamps, next_possible, side='left')

        if idx < len(self.valid_timestamps):
            return int(self.valid_timestamps[idx])
        return None

    def close(self):
        """Close database connection"""
        self.conn.close()


class EpisodeGenerator:
    """
    Generates training episodes from data provider.

    Handles:
    - Random episode start selection
    - Data iteration with step tracking
    - Episode termination conditions
    """

    def __init__(
        self,
        provider: DataProvider,
        episode_length: int = 200,
        lookback_warmup: int = 60
    ):
        self.provider = provider
        self.episode_length = episode_length
        self.lookback_warmup = lookback_warmup

        self.current_timestamp = None
        self.steps_taken = 0
        self.valid_ts = provider.get_valid_timestamps()

    def reset(self) -> MarketSnapshot:
        """Start new episode from random point"""
        if isinstance(self.provider, SQLiteDataProvider):
            self.current_timestamp = self.provider.get_random_start(
                lookahead=self.episode_length + 50,
                lookback=self.lookback_warmup
            )
        else:
            # Generic fallback
            min_t, max_t = self.provider.get_time_range()
            buffer = (self.episode_length + self.lookback_warmup) * 60
            self.current_timestamp = np.random.randint(min_t + buffer, max_t - buffer)

        self.steps_taken = 0
        return self.provider.get_snapshot(self.current_timestamp)

    def step(self) -> Tuple[MarketSnapshot, bool]:
        """
        Advance to next timestamp.

        Returns:
            (snapshot, done): Market data and whether episode ended
        """
        if isinstance(self.provider, SQLiteDataProvider):
            next_ts = self.provider.get_next_timestamp(self.current_timestamp)
        else:
            next_ts = self.current_timestamp + 60

        self.steps_taken += 1
        done = (
            next_ts is None or
            self.steps_taken >= self.episode_length
        )

        if next_ts:
            self.current_timestamp = next_ts

        snapshot = self.provider.get_snapshot(self.current_timestamp)
        return snapshot, done

    def get_lookback_data(self) -> List[MarketSnapshot]:
        """Get historical data for encoder warmup"""
        if self.valid_ts is None:
            return []

        # Find current index
        idx = np.searchsorted(self.valid_ts, self.current_timestamp, side='left')

        # Get lookback timestamps
        start_idx = max(0, idx - self.lookback_warmup)
        lookback_ts = self.valid_ts[start_idx:idx]

        return [self.provider.get_snapshot(int(ts)) for ts in lookback_ts]
