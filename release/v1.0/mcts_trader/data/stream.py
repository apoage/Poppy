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
        """Route message to appropriate handler, adapting V2 format to V1 expected by legacy handlers"""
        channel = data.get('channel')
        msg_type = data.get('type')
        
        if channel == 'ohlc' and msg_type == 'update':
            # Adapt V2 OHLC to V1-like list structure
            # V2: {"data": [{"symbol": "BTC/USD", "open": 1.0, ...}]}
            # V1 Expected: [id, [time, etime, o, h, l, c, vwap, vol, count], name, pair]
            from datetime import datetime
            
            for item in data.get('data', []):
                try:
                    # Parse ISO timestamp to unix - robust handling
                    ts_str = str(item.get('interval_begin', '')).strip()

                    if not ts_str:
                        continue

                    # Handle Z suffix if present
                    if ts_str.endswith('Z'):
                        ts_str = ts_str[:-1]

                    # Handle nanosecond precision (Python only supports microseconds - 6 digits)
                    # Kraken V2 sends 9 digits (nanoseconds)
                    if '.' in ts_str:
                        main_part, frac_part = ts_str.rsplit('.', 1)
                        # Truncate to 6 digits max, pad if needed
                        frac_part = frac_part[:6].ljust(6, '0')
                        ts_str = f"{main_part}.{frac_part}"

                    try:
                        ts = datetime.fromisoformat(ts_str).timestamp()
                    except ValueError:
                        # Fallback: try parsing without fractional seconds
                        if '.' in ts_str:
                            ts_str = ts_str.split('.')[0]
                        ts = datetime.fromisoformat(ts_str).timestamp()
                    
                    # Construct V1-like payload
                    v1_payload = [
                        0, # Channel ID (dummy)
                        [
                            str(int(ts)),      # Time (int string)
                            str(int(ts) + 60), # End Time (int string)
                            str(item.get('open')),
                            str(item.get('high')),
                            str(item.get('low')),
                            str(item.get('close')),
                            str(item.get('vwap')),
                            str(item.get('volume')),
                            item.get('trades')
                        ],
                        'ohlc',
                        item.get('symbol')
                    ]
                    await self.on_ohlc(v1_payload)
                except Exception as e:
                    logging.error(f"Error adapting OHLC message: {e}")
                    
        elif channel == 'ticker':
            await self.on_ticker(data['data'])
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
