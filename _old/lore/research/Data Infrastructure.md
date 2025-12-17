**PROMPT 2: Data Infrastructure**

```markdown
# Data Infrastructure - Kraken Data Fetcher & Local Cache

## Objective
Build robust data pipeline that fetches historical + real-time data from Kraken, caches locally, respects rate limits, and serves data to the trading agent.

## Requirements

### 1. Database Schema (SQLite for dev, Postgres for prod)

```sql
-- Assets we're tracking
CREATE TABLE assets (
    id INTEGER PRIMARY KEY,
    symbol VARCHAR(20) UNIQUE NOT NULL,  -- e.g., 'BTC/USD'
    kraken_symbol VARCHAR(20) NOT NULL,  -- e.g., 'XXBTZUSD'
    base_currency VARCHAR(10),
    quote_currency VARCHAR(10),
    is_active BOOLEAN DEFAULT TRUE,
    min_order_size DECIMAL(20,10),
    price_decimals INTEGER,
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- OHLCV candles (1-minute base, aggregate for larger timeframes)
CREATE TABLE ohlcv (
    asset_id INTEGER REFERENCES assets(id),
    timestamp BIGINT NOT NULL,           -- Unix timestamp
    open DECIMAL(20,10),
    high DECIMAL(20,10),
    low DECIMAL(20,10),
    close DECIMAL(20,10),
    volume DECIMAL(20,10),
    trade_count INTEGER,
    PRIMARY KEY (asset_id, timestamp)
);

-- Individual trades (for tick-level analysis)
CREATE TABLE trades (
    id BIGINT PRIMARY KEY,
    asset_id INTEGER REFERENCES assets(id),
    timestamp BIGINT NOT NULL,
    price DECIMAL(20,10),
    volume DECIMAL(20,10),
    side VARCHAR(4),                     -- 'buy' or 'sell'
    order_type VARCHAR(10)               -- 'market' or 'limit'
);

-- Track what data we've fetched (for incremental updates)
CREATE TABLE fetch_log (
    asset_id INTEGER REFERENCES assets(id),
    data_type VARCHAR(20),               -- 'ohlcv' or 'trades'
    oldest_timestamp BIGINT,
    newest_timestamp BIGINT,
    last_fetch TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (asset_id, data_type)
);

-- Indexes for fast queries
CREATE INDEX idx_ohlcv_time ON ohlcv(timestamp);
CREATE INDEX idx_trades_time ON trades(timestamp);
CREATE INDEX idx_trades_asset_time ON trades(asset_id, timestamp);
```

### 2. Kraken API Client

```python
"""
kraken_client.py - Rate-limited Kraken API client

Rate Limits (respect these!):
- Public endpoints: 1 req/sec safe, burst up to 15
- Private endpoints: Based on tier (use counter system)
- WebSocket: 150 connections/10min per IP
"""

import asyncio
import aiohttp
import time
from dataclasses import dataclass
from typing import Optional, List, Dict
import logging

@dataclass
class RateLimiter:
    """Token bucket rate limiter"""
    max_tokens: int = 15
    refill_rate: float = 1.0  # tokens per second
    tokens: float = 15.0
    last_refill: float = 0.0
    
    async def acquire(self, tokens: int = 1):
        """Wait until tokens available"""
        while True:
            now = time.time()
            # Refill tokens
            elapsed = now - self.last_refill
            self.tokens = min(self.max_tokens, self.tokens + elapsed * self.refill_rate)
            self.last_refill = now
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return
            
            # Wait for refill
            wait_time = (tokens - self.tokens) / self.refill_rate
            await asyncio.sleep(wait_time)


class KrakenClient:
    BASE_URL = "https://api.kraken.com"
    
    def __init__(self, api_key: str = None, api_secret: str = None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.rate_limiter = RateLimiter()
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, *args):
        await self.session.close()
    
    async def get_tradeable_pairs(self) -> Dict:
        """Get all tradeable asset pairs"""
        await self.rate_limiter.acquire()
        async with self.session.get(f"{self.BASE_URL}/0/public/AssetPairs") as resp:
            data = await resp.json()
            return data.get('result', {})
    
    async def get_ohlc(self, pair: str, interval: int = 1, since: int = None) -> List:
        """
        Get OHLC data
        interval: 1, 5, 15, 30, 60, 240, 1440, 10080, 21600 minutes
        since: Unix timestamp (returns data since this time)
        """
        await self.rate_limiter.acquire()
        params = {'pair': pair, 'interval': interval}
        if since:
            params['since'] = since
            
        async with self.session.get(
            f"{self.BASE_URL}/0/public/OHLC", 
            params=params
        ) as resp:
            data = await resp.json()
            if data.get('error'):
                logging.error(f"OHLC error for {pair}: {data['error']}")
                return []
            result = data.get('result', {})
            # Result has pair data and 'last' timestamp
            pair_data = [v for k, v in result.items() if k != 'last']
            return pair_data[0] if pair_data else []
    
    async def get_trades(self, pair: str, since: int = None) -> tuple:
        """
        Get recent trades (up to 1000)
        Returns: (trades_list, last_timestamp)
        """
        await self.rate_limiter.acquire()
        params = {'pair': pair}
        if since:
            params['since'] = since
            
        async with self.session.get(
            f"{self.BASE_URL}/0/public/Trades",
            params=params
        ) as resp:
            data = await resp.json()
            if data.get('error'):
                logging.error(f"Trades error for {pair}: {data['error']}")
                return [], None
            result = data.get('result', {})
            last = result.pop('last', None)
            trades = list(result.values())[0] if result else []
            return trades, last
    
    async def get_ticker(self, pairs: List[str]) -> Dict:
        """Get current ticker for multiple pairs"""
        await self.rate_limiter.acquire()
        pair_str = ','.join(pairs)
        async with self.session.get(
            f"{self.BASE_URL}/0/public/Ticker",
            params={'pair': pair_str}
        ) as resp:
            data = await resp.json()
            return data.get('result', {})
```

### 3. Data Fetcher with Incremental Updates

```python
"""
data_fetcher.py - Incremental historical data fetcher
"""

import asyncio
from datetime import datetime, timedelta
from typing import List
import logging

class DataFetcher:
    def __init__(self, client: KrakenClient, db: Database):
        self.client = client
        self.db = db
        
    async def fetch_top_volatile_assets(self, n: int = 64) -> List[str]:
        """
        Get top N volatile assets by 24h volume
        Excludes stablecoins (USDT, USDC, DAI, etc.)
        """
        pairs = await self.client.get_tradeable_pairs()
        tickers = await self.client.get_ticker(list(pairs.keys()))
        
        stablecoins = {'USDT', 'USDC', 'DAI', 'TUSD', 'BUSD', 'USDP', 'GUSD'}
        
        # Calculate volatility score: volume * price_range / price
        scored = []
        for pair, info in pairs.items():
            base = info.get('base', '')
            if any(stable in base for stable in stablecoins):
                continue
            if pair not in tickers:
                continue
            
            ticker = tickers[pair]
            try:
                volume = float(ticker['v'][1])  # 24h volume
                high = float(ticker['h'][1])
                low = float(ticker['l'][1])
                price = float(ticker['c'][0])
                
                if price > 0 and volume > 0:
                    volatility = (high - low) / price
                    score = volume * volatility
                    scored.append((pair, score, info['wsname']))
            except (KeyError, ValueError, ZeroDivisionError):
                continue
        
        # Sort by score, return top N
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(p[0], p[2]) for p in scored[:n]]
    
    async def fetch_historical_ohlcv(
        self, 
        pair: str, 
        days_back: int = 30,
        interval: int = 1  # 1-minute candles
    ):
        """
        Fetch historical OHLCV data incrementally
        Respects rate limits, caches progress
        """
        asset_id = await self.db.get_or_create_asset(pair)
        
        # Check what we already have
        fetch_log = await self.db.get_fetch_log(asset_id, 'ohlcv')
        
        end_time = int(datetime.now().timestamp())
        start_time = end_time - (days_back * 24 * 60 * 60)
        
        if fetch_log:
            # Continue from where we left off
            if fetch_log['oldest_timestamp'] <= start_time:
                # Already have this data, just update recent
                since = fetch_log['newest_timestamp']
            else:
                # Need older data
                since = start_time
        else:
            since = start_time
        
        logging.info(f"Fetching OHLCV for {pair} from {datetime.fromtimestamp(since)}")
        
        total_candles = 0
        current_since = since
        
        while current_since < end_time:
            candles = await self.client.get_ohlc(pair, interval, current_since)
            
            if not candles:
                break
            
            # Insert into database
            await self.db.insert_ohlcv(asset_id, candles)
            total_candles += len(candles)
            
            # Update progress
            newest = max(c[0] for c in candles)
            oldest = min(c[0] for c in candles)
            await self.db.update_fetch_log(asset_id, 'ohlcv', oldest, newest)
            
            current_since = newest + 1
            
            # Progress logging
            if total_candles % 5000 == 0:
                logging.info(f"  {pair}: {total_candles} candles fetched")
            
            # Small delay to be nice
            await asyncio.sleep(0.5)
        
        logging.info(f"Completed {pair}: {total_candles} total candles")
        return total_candles
    
    async def fetch_all_assets(self, assets: List[str], days_back: int = 30):
        """Fetch data for all assets sequentially (respects rate limits)"""
        for i, (pair, wsname) in enumerate(assets):
            logging.info(f"[{i+1}/{len(assets)}] Fetching {pair}")
            try:
                await self.fetch_historical_ohlcv(pair, days_back)
            except Exception as e:
                logging.error(f"Error fetching {pair}: {e}")
                await asyncio.sleep(5)  # Back off on error
```

### 4. WebSocket Real-time Feed

```python
"""
websocket_feed.py - Real-time data via Kraken WebSocket V2
"""

import asyncio
import json
import websockets
from typing import Callable, List, Dict
import logging

class KrakenWebSocket:
    WS_URL = "wss://ws.kraken.com/v2"
    
    def __init__(self, on_ticker: Callable, on_ohlc: Callable):
        self.on_ticker = on_ticker
        self.on_ohlc = on_ohlc
        self.ws = None
        self.subscribed_pairs = []
        
    async def connect(self):
        """Establish WebSocket connection"""
        self.ws = await websockets.connect(self.WS_URL)
        logging.info("WebSocket connected")
        
    async def subscribe(self, pairs: List[str], channels: List[str] = ['ticker', 'ohlc']):
        """Subscribe to channels for given pairs"""
        for channel in channels:
            msg = {
                "method": "subscribe",
                "params": {
                    "channel": channel,
                    "symbol": pairs
                }
            }
            if channel == 'ohlc':
                msg["params"]["interval"] = 1  # 1-minute
                
            await self.ws.send(json.dumps(msg))
            logging.info(f"Subscribed to {channel} for {len(pairs)} pairs")
        
        self.subscribed_pairs = pairs
        
    async def listen(self):
        """Main listening loop"""
        async for message in self.ws:
            try:
                data = json.loads(message)
                await self._handle_message(data)
            except json.JSONDecodeError:
                logging.warning(f"Invalid JSON: {message}")
            except Exception as e:
                logging.error(f"Error handling message: {e}")
    
    async def _handle_message(self, data: Dict):
        """Route message to appropriate handler"""
        channel = data.get('channel')
        
        if channel == 'ticker':
            await self.on_ticker(data['data'])
        elif channel == 'ohlc':
            await self.on_ohlc(data['data'])
        elif channel == 'heartbeat':
            pass  # Ignore heartbeats
        elif data.get('method') == 'subscribe':
            if data.get('success'):
                logging.info(f"Subscription confirmed: {data}")
            else:
                logging.error(f"Subscription failed: {data}")
    
    async def close(self):
        if self.ws:
            await self.ws.close()
```

### 5. Database Operations

```python
"""
database.py - SQLite/Postgres database operations
"""

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
```

### 6. Main Data Collection Script

```python
"""
collect_data.py - Main script to run data collection
"""

import asyncio
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

async def main():
    # Initialize
    db = Database("trading_data.db")
    await db.connect()
    
    async with KrakenClient() as client:
        fetcher = DataFetcher(client, db)
        
        # Get top 64 volatile assets
        logging.info("Finding top 64 volatile assets...")
        top_assets = await fetcher.fetch_top_volatile_assets(64)
        
        logging.info(f"Selected assets: {[a[0] for a in top_assets[:10]]}...")
        
        # Fetch historical data (30 days)
        # This will take a while - be patient!
        logging.info("Starting historical data fetch (30 days)...")
        logging.info("This may take several hours - progress saved incrementally")
        
        await fetcher.fetch_all_assets(top_assets, days_back=30)
        
        logging.info("Historical fetch complete!")
        
        # Now start real-time feed
        logging.info("Starting real-time WebSocket feed...")
        
        async def on_ticker(data):
            # Update latest prices in memory/DB
            for item in data:
                symbol = item['symbol']
                price = float(item['last'])
                # Store or process...
        
        async def on_ohlc(data):
            # Insert new candles
            for item in data:
                symbol = item['symbol']
                # Insert into DB...
        
        ws = KrakenWebSocket(on_ticker, on_ohlc)
        await ws.connect()
        await ws.subscribe([a[1] for a in top_assets])  # Use wsnames
        
        await ws.listen()  # Runs forever
    
    await db.close()

if __name__ == "__main__":
    asyncio.run(main())
```

## Usage

```bash
# First run - fetch historical data
python collect_data.py

# Check progress
sqlite3 trading_data.db "SELECT symbol, COUNT(*) FROM ohlcv o JOIN assets a ON o.asset_id=a.id GROUP BY asset_id"

# Resume after interruption (automatic - checks fetch_log)
python collect_data.py
```

## Notes
- First historical fetch takes 4-8 hours for 64 assets × 30 days
- Data is saved incrementally - safe to interrupt
- WebSocket provides real-time updates after historical catch-up
- Rate limiter ensures we stay within Kraken's limits
```

