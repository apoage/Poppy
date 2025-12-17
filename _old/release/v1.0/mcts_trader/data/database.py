import aiosqlite
from typing import List, Dict, Optional
import logging

class Database:
    def __init__(self, db_path: str = "trading_data.db"):
        self.db_path = db_path
        self.conn = None
        
    async def connect(self):
        self.conn = await aiosqlite.connect(self.db_path)
        await self._create_tables()
        
    async def _create_tables(self):
        """Create tables if not exist"""
        await self.conn.executescript('''
            CREATE TABLE IF NOT EXISTS assets (
                id INTEGER PRIMARY KEY,
                symbol VARCHAR(20) UNIQUE NOT NULL,
                kraken_symbol VARCHAR(20),
                wsname VARCHAR(20),
                is_active BOOLEAN DEFAULT 1,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS ohlcv (
                asset_id INTEGER,
                timestamp INTEGER NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                trade_count INTEGER,
                PRIMARY KEY (asset_id, timestamp)
            );
            
            CREATE TABLE IF NOT EXISTS fetch_log (
                asset_id INTEGER,
                data_type VARCHAR(20),
                oldest_timestamp INTEGER,
                newest_timestamp INTEGER,
                last_fetch TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (asset_id, data_type)
            );
            
            CREATE INDEX IF NOT EXISTS idx_ohlcv_time ON ohlcv(timestamp);
        ''')
        await self.conn.commit()
    
    async def get_or_create_asset(self, symbol: str, wsname: str = None) -> int:
        """Get asset ID or create if not exists"""
        cursor = await self.conn.execute(
            "SELECT id FROM assets WHERE symbol = ?", (symbol,)
        )
        row = await cursor.fetchone()
        
        if row:
            return row[0]
        
        cursor = await self.conn.execute(
            "INSERT INTO assets (symbol, wsname) VALUES (?, ?)",
            (symbol, wsname)
        )
        await self.conn.commit()
        return cursor.lastrowid
    
    async def insert_ohlcv(self, asset_id: int, candles: List):
        """Insert OHLCV candles (upsert)"""
        # Kraken OHLC: [time, open, high, low, close, vwap, volume, count]
        await self.conn.executemany(
            '''INSERT OR REPLACE INTO ohlcv 
               (asset_id, timestamp, open, high, low, close, volume, trade_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
            [(asset_id, int(c[0]), float(c[1]), float(c[2]), 
              float(c[3]), float(c[4]), float(c[6]), int(c[7])) 
             for c in candles]
        )
        await self.conn.commit()
    
    async def get_ohlcv(
        self, 
        asset_id: int, 
        start_time: int, 
        end_time: int
    ) -> List[Dict]:
        """Get OHLCV data for time range"""
        cursor = await self.conn.execute(
            '''SELECT timestamp, open, high, low, close, volume 
               FROM ohlcv 
               WHERE asset_id = ? AND timestamp BETWEEN ? AND ?
               ORDER BY timestamp''',
            (asset_id, start_time, end_time)
        )
        rows = await cursor.fetchall()
        return [
            {'timestamp': r[0], 'open': r[1], 'high': r[2], 
             'low': r[3], 'close': r[4], 'volume': r[5]}
            for r in rows
        ]
    
    async def get_fetch_log(self, asset_id: int, data_type: str) -> Optional[Dict]:
        cursor = await self.conn.execute(
            '''SELECT oldest_timestamp, newest_timestamp 
               FROM fetch_log WHERE asset_id = ? AND data_type = ?''',
            (asset_id, data_type)
        )
        row = await cursor.fetchone()
        if row:
            return {'oldest_timestamp': row[0], 'newest_timestamp': row[1]}
        return None
    
    async def update_fetch_log(
        self, asset_id: int, data_type: str, 
        oldest: int, newest: int
    ):
        await self.conn.execute(
            '''INSERT OR REPLACE INTO fetch_log 
               (asset_id, data_type, oldest_timestamp, newest_timestamp)
               VALUES (?, ?, 
                       MIN(?, COALESCE((SELECT oldest_timestamp FROM fetch_log 
                                        WHERE asset_id = ? AND data_type = ?), ?)),
                       MAX(?, COALESCE((SELECT newest_timestamp FROM fetch_log 
                                        WHERE asset_id = ? AND data_type = ?), ?)))''',
            (asset_id, data_type, oldest, asset_id, data_type, oldest,
             newest, asset_id, data_type, newest)
        )
        await self.conn.commit()
    
    async def close(self):
        if self.conn:
            await self.conn.close()
