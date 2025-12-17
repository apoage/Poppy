import asyncio
from datetime import datetime, timedelta
from typing import List
import logging
from .kraken_client import KrakenClient
from .database import Database

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
        
        # Chunk keys to avoid URL length limits/520 errors
        all_pairs = list(pairs.keys())
        chunk_size = 50
        tickers = {}
        
        for i in range(0, len(all_pairs), chunk_size):
            chunk = all_pairs[i:i + chunk_size]
            try:
                chunk_tickers = await self.client.get_ticker(chunk)
                tickers.update(chunk_tickers)
                await asyncio.sleep(0.2) # Small delay between chunks
            except Exception as e:
                logging.error(f"Failed to fetch tickers for chunk {i}: {e}")
                continue
        
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
                # ticker format: {'v': [today, 24h], 'h': [today, 24h], ...}
                volume = float(ticker['v'][1])  # 24h volume
                high = float(ticker['h'][1])
                low = float(ticker['l'][1])
                price = float(ticker['c'][0])
                
                if price > 0 and volume > 0:
                    volatility = (high - low) / price
                    score = volume * volatility
                    scored.append((pair, score, info.get('wsname', pair)))
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
        # We rely on the fetcher being initialized with a DB that handles get_or_create_asset
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
        
        # Safety break
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
