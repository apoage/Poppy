import asyncio
import json
import websockets
from typing import Callable, List, Dict
import logging

# Index at which the time field starts in any string Python 3.10's
# datetime.fromisoformat() can parse: a rigid 10-char 'YYYY-MM-DD' date plus exactly
# one separator character. Used by the OHLC timestamp adapter below to locate a
# trailing UTC offset without ever matching the date's own hyphens.
#
# JUDGEMENT (TASK_872 fix round): when a timestamp is too short to have a time field
# (< 12 chars, e.g. a bare 'YYYY-MM-DD'), the offset detach is SKIPPED, not failed
# loud. This is deliberate and is safe by construction, not by heuristic: such a
# string cannot carry an offset that fromisoformat would honour, so the detach would
# be a no-op anyway. The one adjacent-looking case, '2024-07-15+05:30', is NOT a
# date-with-offset -- fromisoformat reads the '+' at index 10 as the separator and
# '05:30' as the TIME (-> 2024-07-15 05:30:00), so detaching there would corrupt a
# string that currently parses. Anything else short and malformed raises in BOTH
# conversion branches and is already dropped loudly by the handler's outer except.
# So there is no input for which skipping produces a silently wrong epoch, which is
# the only failure mode a fail-loud branch here would buy us.
_TIME_FIELD_START = 11


class KrakenWebSocket:
    WS_URL = "wss://ws.kraken.com/v2"
    PING_INTERVAL = 30  # Send ping every 30 seconds
    RECONNECT_DELAY = 5  # Wait 5 seconds before reconnecting

    def __init__(self, on_ticker: Callable, on_ohlc: Callable):
        self.on_ticker = on_ticker
        self.on_ohlc = on_ohlc
        self.ws = None
        self.subscribed_pairs = []
        self.subscribed_channels = []
        self._running = False
        self._reconnect_count = 0

    async def connect(self):
        """Establish WebSocket connection with keepalive"""
        self.ws = await websockets.connect(
            self.WS_URL,
            ping_interval=self.PING_INTERVAL,
            ping_timeout=20,
            close_timeout=10
        )
        self._running = True
        self._reconnect_count = 0
        logging.info("WebSocket connected")
        
    async def subscribe(self, pairs: List[str], channels: List[str] = ['ticker', 'ohlc']):
        """Subscribe to channels for given pairs"""
        # Store for reconnection
        # Use set to avoid duplicates and ensure all previously subscribed pairs are included
        self.subscribed_pairs.extend(pairs)
        self.subscribed_pairs = list(set(self.subscribed_pairs))
        self.subscribed_channels = list(set(self.subscribed_channels + channels))

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

    async def _resubscribe(self):
        """Resubscribe to all channels after reconnect"""
        if not self.subscribed_pairs:
            return

        # Dedupe pairs (already done in subscribe but good to be safe)
        unique_pairs = list(set(self.subscribed_pairs))
        chunk_size = 20

        for channel in self.subscribed_channels:
            for i in range(0, len(unique_pairs), chunk_size):
                chunk = unique_pairs[i:i+chunk_size]
                msg = {
                    "method": "subscribe",
                    "params": {
                        "channel": channel,
                        "symbol": chunk
                    }
                }
                if channel == 'ohlc':
                    msg["params"]["interval"] = 1

                await self.ws.send(json.dumps(msg))
                await asyncio.sleep(0.3)  # Rate limit

        logging.info(f"Resubscribed to {len(unique_pairs)} pairs")
        
    async def listen(self):
        """Main listening loop with auto-reconnect"""
        while self._running:
            try:
                async for message in self.ws:
                    try:
                        data = json.loads(message)
                        await self._handle_message(data)
                    except json.JSONDecodeError:
                        logging.warning(f"Invalid JSON: {message}")
                    except Exception as e:
                        logging.error(f"Error handling message: {e}")

            except websockets.exceptions.ConnectionClosed as e:
                logging.warning(f"WebSocket disconnected: {e}")
                await self._reconnect()

            except Exception as e:
                logging.error(f"WebSocket error: {e}")
                await self._reconnect()

    async def _reconnect(self):
        """Attempt to reconnect with exponential backoff (indefinitely)"""
        if not self._running:
            return

        self._reconnect_count += 1

        delay = self.RECONNECT_DELAY * (2 ** (self._reconnect_count - 1))  # Exponential backoff
        delay = min(delay, 60)  # Cap at 60 seconds

        logging.info(f"Reconnecting in {delay}s (attempt {self._reconnect_count})...")
        await asyncio.sleep(delay)

        try:
            await self.connect()
            await self._resubscribe()
            logging.info("Reconnected successfully")
        except Exception as e:
            logging.error(f"Reconnection failed: {e}. Retrying...")
            # Keep trying indefinitely
            await self._reconnect()
    
    async def _handle_message(self, data: Dict):
        """Route message to appropriate handler, adapting V2 format to V1 expected by legacy handlers"""
        channel = data.get('channel')
        msg_type = data.get('type')
        
        if channel == 'ohlc' and msg_type == 'update':
            # Adapt V2 OHLC to V1-like list structure
            # V2: {"data": [{"symbol": "BTC/USD", "open": 1.0, ...}]}
            # V1 Expected: [id, [time, etime, o, h, l, c, vwap, vol, count], name, pair]
            from datetime import datetime, timezone

            for item in data.get('data', []):
                try:
                    # Parse ISO timestamp to unix - robust handling
                    ts_str = str(item.get('interval_begin', '')).strip()

                    if not ts_str:
                        continue

                    # Handle Z suffix if present. Kraken's interval_begin is documented UTC,
                    # but Python 3.10's fromisoformat() cannot parse a trailing 'Z' at all, so
                    # it must be stripped here -- which leaves fromisoformat() with no offset
                    # info, producing a NAIVE datetime. Naive .timestamp() resolves in the
                    # HOST's local timezone, not UTC (TASK_872: was silently shifting every
                    # live candle by the machine's UTC offset). Below, any naive result is
                    # given tzinfo=utc explicitly before calling .timestamp(); a string that
                    # already carries a real offset is left alone (fromisoformat already
                    # returns it aware).
                    if ts_str.endswith('Z'):
                        ts_str = ts_str[:-1]

                    # Detach any trailing numeric UTC offset (+HH:MM / -HH:MM) BEFORE the
                    # nanosecond truncation below, and re-attach it at parse time. The
                    # truncation slices the fractional FIELD, which for an offset-bearing
                    # string also contains the offset: with >=6 fractional digits it
                    # deleted the offset outright (silently naive, no exception), with
                    # fewer it cut the offset in half (ValueError -> the fallback then
                    # dropped it too). Either way the tzinfo guard below saw a naive
                    # datetime and stamped UTC onto a wall-clock reading that was never
                    # UTC -- a silent epoch error of exactly the offset.
                    #
                    # The scan is POSITIONAL, not separator-matched. Python 3.10's
                    # fromisoformat() only accepts a rigid 10-character 'YYYY-MM-DD'
                    # date ('2024-7-15...' etc. all raise), but it accepts ANY single
                    # character as the date/time separator -- 'T', lowercase 't', a
                    # space and even 'X' or '_' all parse. So the time field always
                    # starts at index 11, and searching for a literal 'T'/' ' would
                    # miss every other separator and silently skip the detach (the
                    # exact defect class this block exists to close). Starting the
                    # backward scan at index 11 also makes it structurally impossible
                    # to match the date's own hyphens at indices 4 and 7, or the
                    # separator at index 10.
                    #
                    # A string shorter than 12 chars yields an empty range and is left
                    # alone deliberately -- see the note below.
                    offset_suffix = ''
                    for _i in range(len(ts_str) - 1, _TIME_FIELD_START - 1, -1):
                        if ts_str[_i] in '+-':
                            offset_suffix = ts_str[_i:]
                            ts_str = ts_str[:_i]
                            break

                    # Handle nanosecond precision (Python only supports microseconds - 6 digits)
                    # Kraken V2 sends 9 digits (nanoseconds)
                    if '.' in ts_str:
                        main_part, frac_part = ts_str.rsplit('.', 1)
                        # Truncate to 6 digits max, pad if needed
                        frac_part = frac_part[:6].ljust(6, '0')
                        ts_str = f"{main_part}.{frac_part}"

                    try:
                        dt = datetime.fromisoformat(ts_str + offset_suffix)
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        ts = dt.timestamp()
                    except ValueError:
                        # Fallback: try parsing without fractional seconds
                        if '.' in ts_str:
                            ts_str = ts_str.split('.')[0]
                        dt = datetime.fromisoformat(ts_str + offset_suffix)
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        ts = dt.timestamp()
                    
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
                # Only log symbol, not full message with deprecation warnings
                result = data.get('result', {})
                symbol = result.get('symbol', 'unknown')
                channel = result.get('channel', 'unknown')
                logging.debug(f"Subscribed to {channel}: {symbol}")
            else:
                logging.error(f"Subscription failed: {data}")
    
    async def close(self):
        """Gracefully close the connection"""
        self._running = False
        if self.ws:
            await self.ws.close()
