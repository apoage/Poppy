import sqlite3
import logging
import numpy as np
from typing import Dict, Tuple, Optional

class HistoricalMarket:
    """
    Manages historical data replay for training.
    Reads from SQLite database efficiently.
    """
    def __init__(self, db_path: str = "trading_data.db", n_assets: int = 64, min_required_assets: int = 10, 
                 specific_symbols: list = None, start_date: Optional[str] = None, end_date: Optional[str] = None):
        """
        Initialize HistoricalMarket with robust filtering.

        Args:
            db_path: Path to SQLite database
            n_assets: Number of assets to track
            min_required_assets: Minimum assets required at each timestamp (default: 10, relaxed from 64)
            specific_symbols: Optional list of symbols to load specifically (overrides Top N)
            start_date: Optional start date filter (YYYY-MM-DD or unix timestamp)
            end_date: Optional end date filter (YYYY-MM-DD or unix timestamp)
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.n_assets = n_assets
        self.min_required_assets = min_required_assets

        # Date filtering
        from datetime import datetime
        self.filter_start_ts = 0
        self.filter_end_ts = 9999999999

        if start_date:
            if isinstance(start_date, int) or str(start_date).isdigit():
                self.filter_start_ts = int(start_date)
            else:
                try:
                    self.filter_start_ts = int(datetime.strptime(str(start_date), '%Y-%m-%d').timestamp())
                except ValueError:
                     # Try ISO format just in case
                     try:
                         self.filter_start_ts = int(datetime.fromisoformat(str(start_date)).timestamp())
                     except:
                         logging.warning(f"Could not parse start_date: {start_date}, ignoring.")

        if end_date:
            if isinstance(end_date, int) or str(end_date).isdigit():
                self.filter_end_ts = int(end_date)
            else:
                try:
                    self.filter_end_ts = int(datetime.strptime(str(end_date), '%Y-%m-%d').timestamp())
                except ValueError:
                     try:
                         self.filter_end_ts = int(datetime.fromisoformat(str(end_date)).timestamp())
                     except:
                         logging.warning(f"Could not parse end_date: {end_date}, ignoring.")

        logging.info(f"Market Filter: Start={self.filter_start_ts}, End={self.filter_end_ts}")

        if specific_symbols:
            logging.info(f"Loading {len(specific_symbols)} specific assets...")
            self.assets = self._load_specific_assets(specific_symbols)
            # Update n_assets to match what we found
            self.n_assets = len(self.assets)
        else:
            # Load top assets first (by overall data coverage)
            self.assets = self._load_top_assets(n_assets)
            
        self.asset_ids = list(self.assets.keys())
        self.id_to_symbol = self.assets

        if not self.asset_ids:
            # Check if we failed to load specific assets
            if specific_symbols:
                logging.error(f"Could not find any of the {len(specific_symbols)} requested symbols in DB!")
                logging.error(f"Requested sample: {specific_symbols[:5]}")
            raise ValueError("No valid assets found in database")

        # Get time range (clamped by filter)
        self.min_time, self.max_time = self._get_time_range()

        # Pre-compute valid timestamps FOR OUR SPECIFIC ASSETS
        logging.info(f"Building valid timestamp index (min {min_required_assets} of our {self.n_assets} assets)...")
        self.valid_timestamps = self._build_valid_timestamps_for_assets()

        logging.info(f"Market loaded: {len(self.assets)} assets, "
                     f"{len(self.valid_timestamps):,} valid timestamps, "
                     f"Time range: {self.min_time}-{self.max_time}")

        # Ensure current time respects filter
        self.current_time = max(self.min_time, self.filter_start_ts)

    def _load_specific_assets(self, symbols: list) -> Dict[int, str]:
        """Load specific assets by symbol list"""
        # 1. Try exact match
        placeholders = ','.join(['?'] * len(symbols))
        query = f"SELECT id, symbol FROM assets WHERE symbol IN ({placeholders})"
        self.cursor.execute(query, symbols)
        rows = self.cursor.fetchall()
        
        found = {r[0]: r[1] for r in rows}
        found_symbols = set(found.values())
        
        # 2. Check for missing and try variants
        missing = [s for s in symbols if s not in found_symbols]
        if missing:
            # Try variants: XBT/USD <-> XXBTZUSD
            # This is hard to do in SQL IN clause efficiently without massive ORs
            # So we fetch ALL assets map and match in python
            self.cursor.execute("SELECT id, symbol FROM assets")
            all_assets = self.cursor.fetchall()
            
            # Map normalized -> (id, symbol)
            # Normalization: Remove slash, handle XBT/BTC
            norm_map = {}
            for aid, sym in all_assets:
                norm = sym.replace('/', '').replace('BTC', 'XBT')
                norm_map[norm] = (aid, sym)
                # Also try X prefix variant logic if needed, but keeping it simple
            
            for m in missing:
                # Normalize requested symbol
                m_norm = m.replace('/', '').replace('BTC', 'XBT')
                if m_norm in norm_map:
                    aid, real_sym = norm_map[m_norm]
                    found[aid] = real_sym
        
        return found

    def _load_top_assets(self, n: int) -> Dict[int, str]:
        """Load top assets by data count"""
        DELISTED_ASSETS = {'EOS/USD', 'LUNA/USD', 'FTT/USD', 'SRM/USD'}
        
        query = """
            SELECT a.id, a.symbol, COUNT(o.timestamp) as cnt
            FROM assets a
            JOIN ohlcv o ON a.id = o.asset_id
            WHERE a.is_active = 1
            GROUP BY a.id
            ORDER BY cnt DESC
            LIMIT ?
        """
        # We fetch more than N initially to account for filtering
        self.cursor.execute(query, (n + 20,))
        rows = self.cursor.fetchall()
        
        assets = {}
        for r in rows:
            symbol = r[1]
            if symbol in DELISTED_ASSETS:
                logging.info(f"Skipping delisted asset: {symbol}")
                continue
            
            assets[r[0]] = symbol
            if len(assets) >= n:
                break
                
        return assets
    
    def _get_time_range(self) -> Tuple[int, int]:
        """Get intersection of time ranges for selected assets"""
        # Ideally we want the common time range where most assets exist
        # Approximation: Get min/max of the most data-rich asset
        # A better approach: Global min/max of the entire table
        self.cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM ohlcv")
        return self.cursor.fetchone()

    def _build_valid_timestamps_for_assets(self) -> np.ndarray:
        """
        Build valid timestamps for OUR SPECIFIC ASSETS.
        Only includes timestamps where at least min_required_assets of our tracked assets have data.
        Also applies start/end date filters.

        Returns:
            Sorted numpy array of valid timestamps
        """
        asset_ids_str = ','.join(map(str, self.asset_ids))
        query = f"""
            SELECT timestamp
            FROM ohlcv
            WHERE asset_id IN ({asset_ids_str})
            AND timestamp >= ? AND timestamp <= ?
            GROUP BY timestamp
            HAVING COUNT(DISTINCT asset_id) >= ?
            ORDER BY timestamp ASC
        """
        self.cursor.execute(query, (self.filter_start_ts, self.filter_end_ts, self.min_required_assets))
        rows = self.cursor.fetchall()

        if not rows:
            logging.warning(f"No timestamps found with {self.min_required_assets}+ of our assets in range {self.filter_start_ts}-{self.filter_end_ts}.")
            if self.filter_start_ts == 0 and self.filter_end_ts >= 9999999999:
                 # Only fallback if we are NOT filtering strict ranges
                 logging.warning("Falling back to all timestamps.")
                 query_fallback = "SELECT DISTINCT timestamp FROM ohlcv ORDER BY timestamp ASC"
                 self.cursor.execute(query_fallback)
                 rows = self.cursor.fetchall()
            else:
                 # If range filtered returned 0, we really have 0 data
                 return np.array([], dtype=np.int64)

        return np.array([r[0] for r in rows], dtype=np.int64)

    def _build_valid_timestamps_robust(self) -> np.ndarray:
        """
        Pre-compute ALL timestamps that have >= min_required_assets from ANY assets.
        This is more robust than filtering by specific asset IDs.

        Returns:
            Sorted numpy array of valid timestamps
        """
        query = """
            SELECT timestamp
            FROM ohlcv
            GROUP BY timestamp
            HAVING COUNT(DISTINCT asset_id) >= ?
            ORDER BY timestamp ASC
        """
        self.cursor.execute(query, (self.min_required_assets,))
        rows = self.cursor.fetchall()

        if not rows:
            logging.warning(f"No timestamps found with {self.min_required_assets}+ assets. Trying with lower threshold...")
            # Fallback: try with half the requirement
            self.cursor.execute(query, (self.min_required_assets // 2,))
            rows = self.cursor.fetchall()

        return np.array([r[0] for r in rows], dtype=np.int64)

    def _load_assets_from_valid_timestamps(self, n: int) -> Tuple[Dict[int, str], list, Dict[int, str]]:
        """
        Load top N assets based on data coverage within valid timestamps.

        Returns:
            Tuple of (assets dict, asset_ids list, id_to_symbol dict)
        """
        # Get min/max of valid timestamps
        if len(self.valid_timestamps) == 0:
            # Fallback to all timestamps
            ts_min, ts_max = self.min_time, self.max_time
        else:
            ts_min = self.valid_timestamps[0]
            ts_max = self.valid_timestamps[-1]

        logging.info(f"Loading {n} assets from timestamp range {ts_min}-{ts_max}")

        # Find assets with most data points in valid timestamp range
        query = """
            SELECT a.id, a.symbol, COUNT(o.timestamp) as cnt
            FROM assets a
            JOIN ohlcv o ON a.id = o.asset_id
            WHERE o.timestamp >= ? AND o.timestamp <= ?
            GROUP BY a.id
            ORDER BY cnt DESC
            LIMIT ?
        """
        self.cursor.execute(query, (ts_min, ts_max, n))
        rows = self.cursor.fetchall()

        logging.info(f"Found {len(rows)} assets with data in valid timestamp range")

        if len(rows) == 0:
            logging.error("No assets found! Falling back to simple asset load...")
            # Fallback: load any assets
            query_fallback = "SELECT id, symbol FROM assets ORDER BY id LIMIT ?"
            self.cursor.execute(query_fallback, (n,))
            rows = self.cursor.fetchall()

        assets = {r[0]: r[1] for r in rows}
        asset_ids = list(assets.keys())
        id_to_symbol = assets.copy()

        return assets, asset_ids, id_to_symbol

    def _build_valid_timestamps(self) -> np.ndarray:
        """
        Pre-compute all timestamps that have >= min_active_assets coverage.
        This is computed ONCE at init, avoiding slow GROUP BY queries during training.

        Returns:
            Sorted numpy array of valid timestamps
        """
        # Query: Find all timestamps where count of our tracked assets >= threshold
        asset_ids_str = ','.join(map(str, self.asset_ids))
        query = f"""
            SELECT timestamp
            FROM ohlcv
            WHERE asset_id IN ({asset_ids_str})
            GROUP BY timestamp
            HAVING COUNT(DISTINCT asset_id) >= ?
            ORDER BY timestamp ASC
        """
        self.cursor.execute(query, (self.min_active_assets,))
        rows = self.cursor.fetchall()

        if not rows:
            logging.warning(f"No timestamps found with {self.min_active_assets}+ assets. Using all timestamps.")
            # Fallback: use all unique timestamps
            query_fallback = "SELECT DISTINCT timestamp FROM ohlcv ORDER BY timestamp ASC"
            self.cursor.execute(query_fallback)
            rows = self.cursor.fetchall()

        return np.array([r[0] for r in rows], dtype=np.int64)
        
    def reset(self, start_time: Optional[int] = None, lookahead_steps: int = 300) -> Dict[str, Dict]:
        """
        Reset market to a random valid timestamp with enough lookahead.

        Args:
            start_time: Specific timestamp to start at (optional)
            lookahead_steps: Minimum number of steps that must be available after this timestamp
        """
        if start_time:
            self.current_time = start_time
        else:
            # Ensure we have enough lookahead for a full episode
            # Reserve last N timestamps for lookahead (N = lookahead_steps)
            max_start_idx = len(self.valid_timestamps) - lookahead_steps

            if max_start_idx <= 0:
                # Not enough data for lookahead, just use first timestamp
                if len(self.valid_timestamps) > 0:
                    self.current_time = self.valid_timestamps[0]
                else:
                    self.current_time = self.min_time
            else:
                # Pick random index ensuring lookahead
                # Also avoid first 30 days for lookback
                buffer_seconds = 60 * 60 * 24 * 30  # 30 days
                buffer_min_time = self.min_time + buffer_seconds

                # Find first valid index >= buffer_min_time
                min_start_idx = np.searchsorted(self.valid_timestamps, buffer_min_time, side='left')

                # Pick random index between min and max
                if min_start_idx < max_start_idx:
                    idx = np.random.randint(min_start_idx, max_start_idx)
                    self.current_time = self.valid_timestamps[idx]
                else:
                    # Fallback: use min_start_idx if possible
                    if min_start_idx < len(self.valid_timestamps):
                        self.current_time = self.valid_timestamps[min_start_idx]
                    else:
                        self.current_time = self.valid_timestamps[0]

        return self.get_current_data()
    
    def step(self) -> Dict[str, Dict]:
        """Advance to the next valid timestamp (O(log n) binary search)"""

        next_possible_time = self.current_time + 60

        # Binary search for next valid timestamp >= next_possible_time
        idx = np.searchsorted(self.valid_timestamps, next_possible_time, side='left')

        if idx < len(self.valid_timestamps):
            self.current_time = self.valid_timestamps[idx]
        else:
            # No more valid data in the future
            logging.warning(f"No more valid timestamps after {self.current_time}. Ending episode.")
            self.current_time = self.max_time + 60  # Signal end of data

        return self.get_current_data()
                
    def get_current_data(self) -> Dict[str, Dict]:
        """
        Get candle data for all assets at current_time.
        Returns: {symbol: {open, high, low, close, volume}}
        """
        asset_ids_str = ','.join(map(str, self.asset_ids))
        query = f"""
            SELECT asset_id, open, high, low, close, volume
            FROM ohlcv
            WHERE timestamp = {self.current_time} AND asset_id IN ({asset_ids_str})
        """
        # Don't use placeholders - SQLite might have issues
        self.cursor.execute(query)
        rows = self.cursor.fetchall()

        if len(rows) == 0:
            logging.warning(f"Query returned 0 rows at timestamp {self.current_time}. First 5 asset IDs: {self.asset_ids[:5]}")

        if len(rows) > 0 and len(rows) < 5:  # Debug: only log when few assets
            logging.debug(f"get_current_data: {len(rows)} rows at timestamp {self.current_time}")

        data = {}
        for r in rows:
            asset_id = r[0]
            if asset_id in self.id_to_symbol:
                sym = self.id_to_symbol[asset_id]
                data[sym] = {
                    'open': float(r[1]) if r[1] is not None else 0.0,
                    'high': float(r[2]) if r[2] is not None else 0.0,
                    'low': float(r[3]) if r[3] is not None else 0.0,
                    'close': float(r[4]) if r[4] is not None else 0.0,
                    'volume': float(r[5]) if r[5] is not None else 0.0
                }
            else:
                logging.debug(f"Asset ID {asset_id} not in mapping")

        if len(rows) > 0 and len(data) == 0:
            logging.warning(f"Query returned {len(rows)} rows but 0 assets added to data dict!")

        return data
        
    def get_history(self, lookback_minutes: int) -> Dict[str, Dict]:
        """Get historical window for initialization (slow, use sparingly)"""
        start = self.current_time - (lookback_minutes * 60)
        query = f"""
            SELECT asset_id, timestamp, open, high, low, close, volume
            FROM ohlcv
            WHERE timestamp > ? AND timestamp <= ? 
            AND asset_id IN ({','.join(map(str, self.asset_ids))})
            ORDER BY timestamp ASC
        """
        self.cursor.execute(query, (start, self.current_time))
        rows = self.cursor.fetchall()
        
        # Reconstruct
        # This is complex because we need to pivot.
        # For encoder init, we might just loop step() lookback times.
        # Or simpler: just return empty/zeros and let the agent warm up.
        return {} 

    def close(self):
        self.conn.close()
