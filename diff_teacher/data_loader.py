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

# TASK_871: reuse the repo's one canonical UTC-midnight date parse (already the rule
# TASK_869 made canonical everywhere windows are resolved on the read-of-record surface)
# instead of a second local-time-resolving copy.
#
# DISCLOSED COST, measured — this import is NOT free and does NOT leave this loader's
# "completely independent from mcts_trader data loading" boundary intact as literally
# written. `trader.backtest` pulls scipy AND (since TASK_870 routed its DB handle through
# the shared read-only helper) `mcts_trader.utils.db`: importing this module went from
# pulling zero mcts_trader modules to pulling `mcts_trader{,.utils,.utils.db}`, and 0.49s
# -> 0.71s. The reached module is leaf-level (`os` + `sqlite3` only, no JAX — JAX is
# already in this loader's import graph from its own pre-existing deps, unchanged here),
# so the coupling is nominal rather than behavioural. It is recorded instead of hidden:
# if the independence boundary is meant strictly, the fix is a neutral home for the date
# parse (neither `trader.backtest` nor `mcts_trader`), which is a planner row, not this
# filler's scope.
from trader.backtest import _to_epoch

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
        """Parse date strings to timestamps.

        TASK_871: 'YYYY-MM-DD' strings resolve at UTC midnight (`trader.backtest._to_epoch`)
        rather than the machine's local timezone — this window feeds a SQL filter against
        `ohlcv.timestamp`, which is UTC epoch seconds, so a local-time parse silently shifted
        the window by the local UTC offset (the same 869 defect class, off the read-of-record
        surface). Digit-string / int timestamps pass through unchanged, as before.
        """
        start_ts = 0
        end_ts = 9999999999

        if start_date:
            if isinstance(start_date, int) or str(start_date).isdigit():
                start_ts = int(start_date)
            else:
                try:
                    start_ts = _to_epoch(str(start_date))
                except ValueError:
                    logger.warning(f"Could not parse start_date: {start_date}")

        if end_date:
            if isinstance(end_date, int) or str(end_date).isdigit():
                end_ts = int(end_date)
            else:
                try:
                    end_ts = _to_epoch(str(end_date))
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


# TASK_772.3 (2026-05-23): RainbowTableLoader moved to .rainbow_loader.
# Re-export here so `from .data_loader import RainbowTableLoader` keeps working.
from .rainbow_loader import RainbowTableLoader  # noqa: F401


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
