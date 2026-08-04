import logging
import numpy as np
import os
import time
from typing import Dict, Tuple, Optional

# TASK_870: read-only handle on the canon trading_data.db (see mcts_trader/utils/db.py) — a
# plain read-write sqlite3.connect on this 24GB read-only source of record lets SQLite's
# hot-journal recovery rewrite the file on the first pure-SELECT read.
from mcts_trader.utils.db import _ro_connect

class HistoricalMarket:
    """
    Manages historical data replay for training.
    Reads from SQLite database efficiently.
    """
    def __init__(self, db_path: str = "trading_data.db", n_assets: int = 64, min_required_assets: int = 10,
                 specific_symbols: list = None, start_date: Optional[str] = None, end_date: Optional[str] = None,
                 use_ram_cache: bool = True, skip_validation: bool = False,
                 cache_path: Optional[str] = None):
        """
        Initialize HistoricalMarket with robust filtering and RAM caching.

        Args:
            cache_path: If provided, tries to load from cache first.
                       If cache doesn't exist, saves to it after loading.
                       Use for faster startup (7min -> 5sec for 102M points).
        """
        self._loaded_from_cache = False

        # Try loading from cache first (massive speedup for tournament/repeated runs)
        if cache_path and self._try_load_cache(cache_path):
            self._loaded_from_cache = True
            # Still need DB connection for any dynamic queries
            self.db_path = db_path
            self.conn = _ro_connect(db_path, check_same_thread=False)
            self.cursor = self.conn.cursor()
            self.current_time = max(self.min_time, 0)
            return  # Skip all the heavy loading

        self.db_path = db_path
        self.conn = _ro_connect(db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.n_assets = n_assets
        self.min_required_assets = min_required_assets
        self.use_ram_cache = use_ram_cache
        self.skip_validation = skip_validation  # Skip expensive GROUP BY when date range pre-validated

        # Date filtering
        # BUG-10 fix: Use UTC timezone for all conversions (Kraken data is UTC)
        from datetime import datetime, timezone
        self.filter_start_ts = 0
        self.filter_end_ts = 9999999999

        if start_date:
            if isinstance(start_date, int) or str(start_date).isdigit():
                self.filter_start_ts = int(start_date)
            else:
                try:
                    self.filter_start_ts = int(datetime.strptime(str(start_date), '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp())
                except ValueError:
                     try:
                         self.filter_start_ts = int(datetime.fromisoformat(str(start_date)).replace(tzinfo=timezone.utc).timestamp())
                     except:
                         logging.warning(f"Could not parse start_date: {start_date}, ignoring.")

        if end_date:
            if isinstance(end_date, int) or str(end_date).isdigit():
                self.filter_end_ts = int(end_date)
            else:
                try:
                    self.filter_end_ts = int(datetime.strptime(str(end_date), '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp())
                except ValueError:
                     try:
                         self.filter_end_ts = int(datetime.fromisoformat(str(end_date)).replace(tzinfo=timezone.utc).timestamp())
                     except:
                         logging.warning(f"Could not parse end_date: {end_date}, ignoring.")

        logging.info(f"Market Filter: Start={self.filter_start_ts}, End={self.filter_end_ts}")

        if specific_symbols:
            logging.info(f"Loading {len(specific_symbols)} specific assets...")
            self.assets = self._load_specific_assets(specific_symbols)
            self.n_assets = len(self.assets)
        else:
            self.assets = self._load_top_assets(n_assets)
            
        self.asset_ids = list(self.assets.keys())
        self.id_to_symbol = self.assets
        self.symbol_to_id = {v: k for k, v in self.assets.items()}

        if not self.asset_ids:
            if specific_symbols:
                logging.error(f"Could not find any of the {len(specific_symbols)} requested symbols in DB!")
            raise ValueError("No valid assets found in database")

        self.min_time, self.max_time = self._get_time_range()

        # TASK_105: Defensive bounds check - fail fast if requesting future data
        if self.filter_start_ts > self.max_time:
            from datetime import datetime
            req_start = datetime.fromtimestamp(self.filter_start_ts).strftime('%Y-%m-%d')
            db_max = datetime.fromtimestamp(self.max_time).strftime('%Y-%m-%d')
            raise ValueError(
                f"Requested start_date ({req_start}) is beyond database max ({db_max}). "
                f"No data available for this period."
            )

        logging.info(f"Building valid timestamp index (min {min_required_assets} of our {self.n_assets} assets)...")
        self.valid_timestamps = self._build_valid_timestamps_for_assets()

        logging.info(f"Market loaded: {len(self.assets)} assets, "
                     f"{len(self.valid_timestamps):,} valid timestamps, "
                     f"Time range: {self.min_time}-{self.max_time}")

        # Ensure current time respects filter
        self.current_time = max(self.min_time, self.filter_start_ts)
        
        # RAM Cache
        self.cache_data = None
        self.ts_to_idx = None
        self.id_to_col = None
        
        # FX Cache
        self.fx_matrix = None
        self.fx_symbols = []
        self.fx_sym_to_idx = {}
        
        # Load FX Symbols if table exists
        try:
            cursor = self.conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='fx_rates'")
            if cursor.fetchone():
                cursor = self.conn.execute("SELECT DISTINCT symbol FROM fx_rates ORDER BY symbol")
                self.fx_symbols = [r[0] for r in cursor.fetchall()]
                self.fx_sym_to_idx = {s: i for i, s in enumerate(self.fx_symbols)}
                logging.info(f"Found {len(self.fx_symbols)} FX symbols: {self.fx_symbols}")
        except Exception as e:
            logging.warning(f"Failed to load FX symbols: {e}")
        
        if self.use_ram_cache and len(self.valid_timestamps) > 0:
            self._preload_to_ram()

        # Save cache if path specified and we loaded from DB
        if cache_path and self.cache_data is not None and not self._loaded_from_cache:
            self._save_cache(cache_path)

    def close_db(self):
        """Close DB connection after RAM cache is loaded. Saves file descriptors when many periods loaded."""
        if hasattr(self, 'conn') and self.conn is not None:
            self.conn.close()
            self.conn = None
            self.cursor = None

    def _try_load_cache(self, cache_path: str) -> bool:
        """Try to load from numpy cache file. Returns True if successful."""
        if not os.path.exists(cache_path):
            return False

        try:
            t0 = time.time()
            logging.info(f"Loading from cache: {cache_path}")
            data = np.load(cache_path, allow_pickle=True)

            # Restore state
            self.valid_timestamps = data['valid_timestamps']
            self.cache_data = data['cache_data']
            self.asset_ids = data['asset_ids'].tolist()
            self.assets = dict(zip(self.asset_ids, data['asset_symbols']))
            self.id_to_symbol = self.assets
            self.symbol_to_id = {v: k for k, v in self.assets.items()}
            self.min_time = int(data['min_time'])
            self.max_time = int(data['max_time'])
            self.n_assets = len(self.asset_ids)

            # Rebuild indices
            self.ts_to_idx = {ts: i for i, ts in enumerate(self.valid_timestamps)}
            self.id_to_col = {aid: i for i, aid in enumerate(self.asset_ids)}

            # FX data if present
            if 'fx_matrix' in data and data['fx_matrix'] is not None:
                self.fx_matrix = data['fx_matrix']
                self.fx_symbols = data['fx_symbols'].tolist() if 'fx_symbols' in data else []
                self.fx_sym_to_idx = {s: i for i, s in enumerate(self.fx_symbols)}

            logging.info(f"Loaded from cache in {time.time()-t0:.1f}s: "
                        f"{len(self.assets)} assets, {len(self.valid_timestamps):,} timestamps")
            return True
        except Exception as e:
            logging.warning(f"Failed to load cache {cache_path}: {e}")
            return False

    def _save_cache(self, cache_path: str):
        """Save current state to numpy cache file."""
        try:
            t0 = time.time()
            logging.info(f"Saving cache to: {cache_path}")

            save_dict = {
                'valid_timestamps': self.valid_timestamps,
                'cache_data': self.cache_data,
                'asset_ids': np.array(self.asset_ids),
                'asset_symbols': np.array([self.assets[aid] for aid in self.asset_ids]),
                'min_time': self.min_time,
                'max_time': self.max_time,
            }

            # Include FX data if available
            if self.fx_matrix is not None:
                save_dict['fx_matrix'] = self.fx_matrix
                save_dict['fx_symbols'] = np.array(self.fx_symbols)

            np.savez_compressed(cache_path, **save_dict)
            file_size = os.path.getsize(cache_path) / (1024**2)
            logging.info(f"Cache saved ({file_size:.1f} MB) in {time.time()-t0:.1f}s")
        except Exception as e:
            logging.warning(f"Failed to save cache: {e}")

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
            self.cursor.execute("SELECT id, symbol FROM assets")
            all_assets = self.cursor.fetchall()
            
            norm_map = {}
            for aid, sym in all_assets:
                norm = sym.replace('/', '').replace('BTC', 'XBT')
                norm_map[norm] = (aid, sym)
            
            for m in missing:
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
        self.cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM ohlcv")
        return self.cursor.fetchone()

    def _build_valid_timestamps_for_assets(self) -> np.ndarray:
        """
        Build valid timestamps for OUR SPECIFIC ASSETS.
        If skip_validation=True, just get all unique timestamps in range (much faster).
        """
        asset_ids_str = ','.join(map(str, self.asset_ids))

        if self.skip_validation:
            # Fast path: trust pre-validated date range, just get distinct timestamps
            print(f"      [HistoricalMarket] Fast-loading timestamps (skip_validation=True)...", flush=True)
            query = f"""
                SELECT DISTINCT timestamp
                FROM ohlcv
                WHERE asset_id IN ({asset_ids_str})
                AND timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp ASC
            """
            self.cursor.execute(query, (self.filter_start_ts, self.filter_end_ts))
            rows = self.cursor.fetchall()
            print(f"      [HistoricalMarket] Found {len(rows)} timestamps", flush=True)
            return np.array([r[0] for r in rows], dtype=np.int64)

        # Standard path: expensive GROUP BY with HAVING COUNT validation
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
                 logging.warning("Falling back to all timestamps.")
                 query_fallback = "SELECT DISTINCT timestamp FROM ohlcv ORDER BY timestamp ASC"
                 self.cursor.execute(query_fallback)
                 rows = self.cursor.fetchall()
            else:
                 return np.array([], dtype=np.int64)

        return np.array([r[0] for r in rows], dtype=np.int64)

    def _preload_to_ram(self):
        """Load all relevant OHLCV data into a numpy matrix for O(1) access"""
        logging.info("🚀 Preloading data to RAM...")
        import time
        t0 = time.time()
        
        # Create mapping for assets -> matrix columns
        self.id_to_col = {asset_id: i for i, asset_id in enumerate(self.asset_ids)}
        
        # Create mapping for timestamps -> matrix rows
        self.ts_to_idx = {ts: i for i, ts in enumerate(self.valid_timestamps)}
        
        # Matrix: (Timestamps, Assets, 5) -> Open, High, Low, Close, Volume
        n_rows = len(self.valid_timestamps)
        n_cols = len(self.asset_ids)
        self.cache_data = np.zeros((n_rows, n_cols, 5), dtype=np.float32)
        
        # Query ALL data in range
        # Chunking to avoid massive SQL result sets if needed, but 180 days is fine
        asset_ids_str = ','.join(map(str, self.asset_ids))
        
        # We only want data for our valid timestamps
        # Constructing IN clause for 200k timestamps is bad.
        # Instead, query range and filter in python.
        
        start_ts = self.valid_timestamps[0]
        end_ts = self.valid_timestamps[-1]
        
        query = f"""
            SELECT timestamp, asset_id, open, high, low, close, volume
            FROM ohlcv
            WHERE timestamp >= {start_ts} AND timestamp <= {end_ts}
            AND asset_id IN ({asset_ids_str})
        """
        
        self.cursor.execute(query)
        
        # Iterate cursor to fill matrix
        count = 0
        for row in self.cursor:
            ts, aid, o, h, l, c, v = row
            if ts in self.ts_to_idx and aid in self.id_to_col:
                r_idx = self.ts_to_idx[ts]
                c_idx = self.id_to_col[aid]
                self.cache_data[r_idx, c_idx] = [o, h, l, c, v]
                count += 1
                
        logging.info(f"✅ Loaded {count:,} candles into RAM ({self.cache_data.nbytes / 1024**2:.1f} MB) in {time.time()-t0:.1f}s")

        # --- Forward-fill missing prices ---
        # Assets may have gaps where no trade occurred. Forward-fill with last known price.
        # This prevents positions from "disappearing" when price data is missing.
        logging.info("📈 Forward-filling price gaps...")
        gaps_filled = 0
        for c_idx in range(n_cols):
            last_valid = None
            for r_idx in range(n_rows):
                if self.cache_data[r_idx, c_idx, 3] > 0:  # Has close price
                    last_valid = self.cache_data[r_idx, c_idx].copy()
                elif last_valid is not None:
                    # Fill gap with last known values
                    self.cache_data[r_idx, c_idx] = last_valid
                    gaps_filled += 1
        logging.info(f"✅ Filled {gaps_filled:,} price gaps via forward-fill")

        # --- Load FX Data ---
        if self.fx_symbols:
            t_fx = time.time()
            n_fx_cols = len(self.fx_symbols)
            # Init with 1.0 (default for USD/USD or missing stablecoins, safer than 0 for division)
            self.fx_matrix = np.ones((n_rows, n_fx_cols), dtype=np.float32)
            
            # Query FX
            fx_query = f"""
                SELECT timestamp, symbol, rate
                FROM fx_rates
                WHERE timestamp >= {start_ts} AND timestamp <= {end_ts}
            """
            self.cursor.execute(fx_query)
            
            fx_count = 0
            for row in self.cursor:
                ts, sym, rate = row
                if ts in self.ts_to_idx and sym in self.fx_sym_to_idx:
                    r_idx = self.ts_to_idx[ts]
                    c_idx = self.fx_sym_to_idx[sym]
                    self.fx_matrix[r_idx, c_idx] = rate
                    fx_count += 1
            
            logging.info(f"✅ Loaded {fx_count:,} FX rates into RAM in {time.time()-t_fx:.1f}s")

            # Forward-fill FX gaps (FX trades less frequently than crypto)
            fx_gaps_filled = 0
            for c_idx in range(n_fx_cols):
                last_valid = 1.0  # Default to 1.0 (USD/USD)
                for r_idx in range(n_rows):
                    if self.fx_matrix[r_idx, c_idx] != 1.0:  # Has actual rate (not default)
                        last_valid = self.fx_matrix[r_idx, c_idx]
                    elif self.fx_matrix[r_idx, c_idx] == 1.0 and last_valid != 1.0:
                        self.fx_matrix[r_idx, c_idx] = last_valid
                        fx_gaps_filled += 1
            logging.info(f"✅ Filled {fx_gaps_filled:,} FX rate gaps via forward-fill")

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

    def get_current_fx_rates(self) -> Dict[str, float]:
        """Get FX rates for the current timestamp."""
        if self.use_ram_cache and self.fx_matrix is not None:
            if self.current_time not in self.ts_to_idx:
                return {}
            
            r_idx = self.ts_to_idx[self.current_time]
            rates = self.fx_matrix[r_idx]
            
            return {sym: float(rates[i]) for sym, i in self.fx_sym_to_idx.items()}
        else:
            # Fallback SQL (slow)
            query = f"SELECT symbol, rate FROM fx_rates WHERE timestamp = {self.current_time}"
            self.cursor.execute(query)
            return {r[0]: float(r[1]) for r in self.cursor.fetchall()}

    def get_current_data(self) -> Dict[str, Dict]:
        """
        Get candle data for all assets at current_time.
        """
        if self.use_ram_cache and self.cache_data is not None:
            # RAM Lookup
            if self.current_time not in self.ts_to_idx:
                return {}
                
            r_idx = self.ts_to_idx[self.current_time]
            # Get slice for this timestamp: (n_assets, 5)
            row_data = self.cache_data[r_idx]
            
            data = {}
            for i, asset_id in enumerate(self.asset_ids):
                # Check if data exists (close > 0 or vol >= 0? close is safest)
                # If everything is 0, it means no data for this asset at this time
                vals = row_data[i]
                if vals[3] > 0: # Close > 0
                    sym = self.id_to_symbol[asset_id]
                    data[sym] = {
                        'open': float(vals[0]),
                        'high': float(vals[1]),
                        'low': float(vals[2]),
                        'close': float(vals[3]),
                        'volume': float(vals[4])
                    }
            return data
            
        else:
            # SQL Fallback
            asset_ids_str = ','.join(map(str, self.asset_ids))
            query = f"""
                SELECT asset_id, open, high, low, close, volume
                FROM ohlcv
                WHERE timestamp = {self.current_time} AND asset_id IN ({asset_ids_str})
            """
            self.cursor.execute(query)
            rows = self.cursor.fetchall()

            if len(rows) == 0:
                logging.warning(f"Query returned 0 rows at timestamp {self.current_time}. First 5 asset IDs: {self.asset_ids[:5]}")

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
            return data
        
    def get_window_random(self, length: int) -> np.ndarray:
        """Sample a sequential window of OHLCV prices from RAM cache.

        TASK_703: Returns full OHLCV instead of close-only.
        Walks through data with stride = length//2 (50% overlap), guaranteeing
        full coverage before wrapping. Small jitter on subsequent passes.

        Returns:
            (length, n_assets, 5) OHLCV prices as float32
        """
        assert self.cache_data is not None, "RAM cache not loaded — set use_ram_cache=True"
        max_start = len(self.valid_timestamps) - length
        assert max_start > 0, f"Period too short: {len(self.valid_timestamps)} timestamps < {length} required"

        if not hasattr(self, '_scan_cursor'):
            self._scan_cursor = 0
            self._scan_stride = max(length // 2, 1)
            self._scan_pass = 0

        start_idx = self._scan_cursor
        if self._scan_pass > 0:
            jitter = np.random.randint(-self._scan_stride // 4, self._scan_stride // 4 + 1)
            start_idx = np.clip(start_idx + jitter, 0, max_start)

        self._scan_cursor += self._scan_stride
        if self._scan_cursor > max_start:
            self._scan_pass += 1
            windows_per_pass = max_start // self._scan_stride + 1
            logging.getLogger(__name__).info(
                f"[HISTORICAL] Pass {self._scan_pass} complete: "
                f"{windows_per_pass} windows, stride={self._scan_stride}")
            self._scan_cursor = 0

        start_idx = min(start_idx, max_start)
        return self.cache_data[start_idx:start_idx + length]  # TASK_703: full OHLCV

    def get_window(self, start_idx: int, length: int) -> np.ndarray:
        """Get a specific contiguous window of OHLCV prices from RAM cache.

        TASK_703: Returns full OHLCV instead of close-only.

        Returns:
            (length, n_assets, 5) OHLCV prices as float32
        """
        assert self.cache_data is not None, "RAM cache not loaded — set use_ram_cache=True"
        end_idx = min(start_idx + length, len(self.valid_timestamps))
        return self.cache_data[start_idx:end_idx]  # TASK_703: full OHLCV

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
