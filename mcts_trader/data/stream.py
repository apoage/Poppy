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
