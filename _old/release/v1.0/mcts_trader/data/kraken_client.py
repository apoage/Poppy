import asyncio
import aiohttp
import time
from dataclasses import dataclass
from typing import Optional, List, Dict
import logging

@dataclass
class RateLimiter:
    """Token bucket rate limiter with adaptive backoff"""
    max_tokens: int = 15
    refill_rate: float = 0.5  # tokens per second
    tokens: float = 15.0
    last_refill: float = 0.0
    penalty_delay: float = 0.0
    
    async def acquire(self, tokens: int = 1):
        """Wait until tokens available"""
        while True:
            now = time.time()
            
            # Apply penalty delay if active
            if self.penalty_delay > 0:
                await asyncio.sleep(self.penalty_delay)
                self.penalty_delay = max(0, self.penalty_delay - (now - self.last_refill))
                now = time.time() # Update time after sleep

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

    def penalize(self, seconds: float = 5.0):
        """Add a penalty delay (backoff)"""
        self.penalty_delay = max(self.penalty_delay, seconds)


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
    
    async def _request(self, endpoint: str, params: Dict = None) -> Dict:
        """
        Internal helper to handle requests with rate limit retries
        """
        retries = 0
        max_retries = 10
        backoff = 5.0
        
        while retries < max_retries:
            await self.rate_limiter.acquire()
            
            try:
                async with self.session.get(f"{self.BASE_URL}/{endpoint}", params=params) as resp:
                    if resp.status in [429, 520, 529, 500, 502, 503, 504]:
                        # Penalize the rate limiter globally
                        self.rate_limiter.penalize(backoff)
                        
                        logging.warning(f"Kraken API {resp.status} error. Retrying in {backoff}s...")
                        await asyncio.sleep(backoff)
                        retries += 1
                        backoff = min(backoff * 1.5, 60)
                        continue
                        
                    data = await resp.json()
                    
                    if data.get('error'):
                        errors = data['error']
                        # Check for rate limit errors
                        if any('Too many requests' in e or 'Rate limit exceeded' in e for e in errors):
                            self.rate_limiter.penalize(backoff)
                            logging.warning(f"Rate limit hit ({errors}). Retrying in {backoff}s...")
                            await asyncio.sleep(backoff)
                            retries += 1
                            backoff = min(backoff * 1.5, 60)
                            continue
                        else:
                            return data
                    
                    return data
                    
            except Exception as e:
                logging.warning(f"Request failed: {e}. Retrying in {backoff}s...")
                await asyncio.sleep(backoff)
                retries += 1
                backoff = min(backoff * 1.5, 60)
        
        logging.error(f"Max retries reached for {endpoint}")
        return {'error': ['Max retries reached']}

    async def get_tradeable_pairs(self) -> Dict:
        """Get all tradeable asset pairs"""
        data = await self._request("0/public/AssetPairs")
        return data.get('result', {})
    
    async def get_ohlc(self, pair: str, interval: int = 1, since: int = None) -> List:
        """
        Get OHLC data
        interval: 1, 5, 15, 30, 60, 240, 1440, 10080, 21600 minutes
        since: Unix timestamp (returns data since this time)
        """
        params = {'pair': pair, 'interval': interval}
        if since:
            params['since'] = since
            
        data = await self._request("0/public/OHLC", params=params)
        
        if data.get('error'):
            logging.error(f"OHLC error for {pair}: {data['error']}")
            return []
            
        result = data.get('result', {})
        pair_data = [v for k, v in result.items() if k != 'last']
        return pair_data[0] if pair_data else []
    
    async def get_trades(self, pair: str, since: int = None) -> tuple:
        """
        Get recent trades (up to 1000)
        Returns: (trades_list, last_timestamp)
        """
        params = {'pair': pair}
        if since:
            params['since'] = since
            
        data = await self._request("0/public/Trades", params=params)
        
        if data.get('error'):
            logging.error(f"Trades error for {pair}: {data['error']}")
            return [], None
            
        result = data.get('result', {})
        last = result.pop('last', None)
        trades = list(result.values())[0] if result else []
        return trades, last
    
    async def get_ticker(self, pairs: List[str]) -> Dict:
        """Get current ticker for multiple pairs"""
        pair_str = ','.join(pairs)
        data = await self._request("0/public/Ticker", params={'pair': pair_str})
        return data.get('result', {})
