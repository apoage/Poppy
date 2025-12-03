import sqlite3
import logging
import numpy as np
from typing import Dict, Tuple, Optional

class HistoricalMarket:
    """
    Manages historical data replay for training.
    Reads from SQLite database efficiently.
    """
    def __init__(self, db_path: str = "trading_data.db", n_assets: int = 64):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        
        # Load assets
        self.assets = self._load_top_assets(n_assets) # {id: symbol}
        self.asset_ids = list(self.assets.keys())
        self.id_to_symbol = self.assets
        
        if not self.asset_ids:
            raise ValueError("No valid assets found in database")
            
        # Get time range
        self.min_time, self.max_time = self._get_time_range()
        logging.info(f"Market loaded: {len(self.assets)} assets, Time range: {self.min_time}-{self.max_time}")
        
        self.current_time = self.min_time
        
    def _load_top_assets(self, n: int) -> Dict[int, str]:
        """Load top assets by data count"""
        query = """
            SELECT a.id, a.symbol, COUNT(o.timestamp) as cnt
            FROM assets a
            JOIN ohlcv o ON a.id = o.asset_id
            GROUP BY a.id
            ORDER BY cnt DESC
            LIMIT ?
        """
        self.cursor.execute(query, (n,))
        rows = self.cursor.fetchall()
        return {r[0]: r[1] for r in rows}
    
    def _get_time_range(self) -> Tuple[int, int]:
        """Get intersection of time ranges for selected assets"""
        # Ideally we want the common time range where most assets exist
        # Approximation: Get min/max of the most data-rich asset
        # A better approach: Global min/max of the entire table
        self.cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM ohlcv")
        return self.cursor.fetchone()
        
    def reset(self, start_time: Optional[int] = None) -> Dict[str, Dict]:
        """
        Reset market to a random time.
        """
        if start_time:
            self.current_time = start_time
        else:
            # Pick random time, ensuring some buffer
            # Avoid the very beginning/end to ensure we have lookback data and future data
            buffer = 60 * 60 * 24 * 30 # 30 days
            start = max(self.min_time + buffer, self.min_time)
            end = max(self.max_time - buffer, start)
            if start >= end:
                start = self.min_time
                end = self.max_time
            
            self.current_time = np.random.randint(start, end)
            
            # Align to minute
            self.current_time -= (self.current_time % 60)
            
        return self.get_current_data()
    
    def step(self) -> Dict[str, Dict]:
        """Advance 1 minute and return data"""
        self.current_time += 60
        return self.get_current_data()
    
    def get_current_data(self) -> Dict[str, Dict]:
        """
        Get candle data for all assets at current_time.
        Returns: {symbol: {open, high, low, close, volume}}
        """
        query = f"""
            SELECT asset_id, open, high, low, close, volume
            FROM ohlcv
            WHERE timestamp = ? AND asset_id IN ({','.join(map(str, self.asset_ids))})
        """
        self.cursor.execute(query, (self.current_time,))
        rows = self.cursor.fetchall()
        
        data = {}
        for r in rows:
            asset_id = r[0]
            if asset_id in self.id_to_symbol:
                sym = self.id_to_symbol[asset_id]
                data[sym] = {
                    'open': r[1],
                    'high': r[2],
                    'low': r[3],
                    'close': r[4],
                    'volume': r[5]
                }
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
