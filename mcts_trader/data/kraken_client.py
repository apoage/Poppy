import asyncio
import aiohttp
import time
import hashlib
import hmac
import base64
import urllib.parse
import os
from dataclasses import dataclass
from typing import Optional, List, Dict, Union
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
        self.api_key = api_key or os.environ.get('KRAKEN_API_KEY')
        self.api_secret = api_secret or os.environ.get('KRAKEN_API_SECRET')
        self.rate_limiter = RateLimiter()
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, *args):
        if self.session:
            await self.session.close()

    def _get_kraken_signature(self, urlpath: str, data: Dict, secret: str) -> str:
        """Sign request with API secret"""
        postdata = urllib.parse.urlencode(data)
        encoded = (str(data['nonce']) + postdata).encode()
        message = urlpath.encode() + hashlib.sha256(encoded).digest()

        mac = hmac.new(base64.b64decode(secret), message, hashlib.sha512)
        return base64.b64encode(mac.digest()).decode()
    
    async def _request(self, endpoint: str, params: Dict = None) -> Dict:
        """
        Internal helper for PUBLIC requests (GET)
        """
        retries = 0
        max_retries = 10
        backoff = 5.0
        
        while retries < max_retries:
            await self.rate_limiter.acquire()
            
            try:
                async with self.session.get(f"{self.BASE_URL}/{endpoint}", params=params) as resp:
                    if resp.status in [429, 520, 529, 500, 502, 503, 504]:
                        self.rate_limiter.penalize(backoff)
                        logging.warning(f"Kraken API {resp.status} error. Retrying in {backoff}s...")
                        await asyncio.sleep(backoff)
                        retries += 1
                        backoff = min(backoff * 1.5, 60)
                        continue
                        
                    data = await resp.json()
                    
                    if data.get('error'):
                        errors = data['error']
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

    async def _private_request(self, endpoint: str, data: Dict = None) -> Dict:
        """
        Internal helper for PRIVATE requests (POST)
        """
        if not self.api_key or not self.api_secret:
            raise ValueError("API Key and Secret required for private requests")

        retries = 0
        max_retries = 5 # Fewer retries for trading
        backoff = 2.0
        
        # Private endpoints often cost more tokens
        # We'll acquire 2 tokens conservatively
        
        urlpath = f"/{endpoint}"
        
        while retries < max_retries:
            await self.rate_limiter.acquire(tokens=2)
            
            # Nonce must be always increasing
            if data is None:
                data = {}
            data['nonce'] = str(int(time.time() * 1000))
            
            headers = {
                'API-Key': self.api_key,
                'API-Sign': self._get_kraken_signature(urlpath, data, self.api_secret)
            }
            
            try:
                async with self.session.post(f"{self.BASE_URL}{urlpath}", data=data, headers=headers) as resp:
                    if resp.status in [429, 520, 529, 500, 502, 503, 504]:
                        self.rate_limiter.penalize(backoff)
                        logging.warning(f"Kraken API {resp.status} error. Retrying in {backoff}s...")
                        await asyncio.sleep(backoff)
                        retries += 1
                        backoff = min(backoff * 1.5, 60)
                        continue
                        
                    data_resp = await resp.json()
                    
                    if data_resp.get('error'):
                        errors = data_resp['error']
                        if any('Too many requests' in e or 'Rate limit exceeded' in e for e in errors):
                            self.rate_limiter.penalize(backoff)
                            logging.warning(f"Rate limit hit ({errors}). Retrying in {backoff}s...")
                            await asyncio.sleep(backoff)
                            retries += 1
                            backoff = min(backoff * 1.5, 60)
                            continue
                        else:
                            return data_resp
                    
                    return data_resp
            except Exception as e:
                logging.warning(f"Private Request failed: {e}. Retrying in {backoff}s...")
                await asyncio.sleep(backoff)
                retries += 1
        
        logging.error(f"Max retries reached for {endpoint}")
        return {'error': ['Max retries reached']}

    # --- Public Methods ---

    async def get_tradeable_pairs(self) -> Dict:
        """Get all tradeable asset pairs"""
        data = await self._request("0/public/AssetPairs")
        return data.get('result', {})
    
    async def get_ohlc(self, pair: str, interval: int = 1, since: int = None) -> List:
        """Get OHLC data"""
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
    
    async def get_ticker(self, pairs: List[str]) -> Dict:
        """Get current ticker for multiple pairs"""
        pair_str = ','.join(pairs)
        data = await self._request("0/public/Ticker", params={'pair': pair_str})
        return data.get('result', {})

    # --- Private Methods (Trading) ---

    async def get_account_balance(self) -> Dict:
        """Get asset balances"""
        data = await self._private_request("0/private/Balance")
        return data.get('result', {})

    async def get_trade_balance(self, asset: str = "ZUSD") -> Dict:
        """Get trade balance info (equity, margin, etc.)"""
        data = await self._private_request("0/private/TradeBalance", {'asset': asset})
        return data.get('result', {})

    async def get_open_orders(self) -> Dict:
        """Get open orders info"""
        data = await self._private_request("0/private/OpenOrders")
        return data.get('result', {}).get('open', {})

    async def get_open_positions(self) -> Dict:
        """Get open margin positions"""
        data = await self._private_request("0/private/OpenPositions")
        return data.get('result', {})

    async def add_order(self, pair: str, side: str, order_type: str, volume: float, 
                       price: float = None, leverage: str = None, 
                       reduce_only: bool = False) -> Dict:
        """
        Place a new order
        side: 'buy' or 'sell'
        order_type: 'market' or 'limit'
        """
        params = {
            'pair': pair,
            'type': side,
            'ordertype': order_type,
            'volume': str(volume)
        }
        if price:
            params['price'] = str(price)
        if leverage:
            params['leverage'] = str(leverage)
        if reduce_only:
            # Kraken flag for reduce-only is obscure, usually passed in 'flags' or specific calls
            # For simplicity/standard api, we might just rely on correct sizing.
            pass

        return await self._private_request("0/private/AddOrder", params)

    async def cancel_order(self, txid: str) -> Dict:
        """Cancel a specific order"""
        return await self._private_request("0/private/CancelOrder", {'txid': txid})

    async def cancel_all_orders(self) -> Dict:
        """Cancel all open orders"""
        return await self._private_request("0/private/CancelAll")
