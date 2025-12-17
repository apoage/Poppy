#!/usr/bin/env python3
"""
JAX Paper Trading Script
Connects to Kraken Real-Time API via WebSocket, runs the trained MuZero agent, and executes paper trades.
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
import asyncio
import os
import logging
from typing import Dict, List

# JAX Setup
os.environ['JAX_PLATFORM_NAME'] = 'cpu' # Inference on CPU is sufficient and robust

# Project Imports
from mcts_trader.jax_agent.muzero_model import MuZeroNetworks
from mcts_trader.jax_agent.trainer import MuZeroTrainer
from mcts_trader.jax_agent.mctx_search import create_mctx_search
from mcts_trader.data.history import HistoricalMarket
from mcts_trader.encoder.state_fast import FastMultiAssetStateEncoder
from mcts_trader.agent.portfolio import PortfolioManager
from mcts_trader.data.kraken_client import KrakenClient
from mcts_trader.data.stream import KrakenWebSocket
from mcts_trader.data.database import Database
from datetime import datetime

# Configuration
CHECKPOINT_DIR = os.path.abspath("jax_ckpt_fast/checkpoint_49750")
N_ASSETS = 64
LOOKBACK = 60
SIMULATIONS = 32 # Higher than training (16) for better inference
INITIAL_CAPITAL = 10000.0

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('paper_trading.log'),
        logging.StreamHandler()
    ]
)

class PaperTrader:
    def __init__(self):
        self.portfolio = PortfolioManager(INITIAL_CAPITAL)
        self.market_state = {} # {symbol: {close, high, low, volume, open}}
        
        # Database for saving history
        self.db = Database("trading_data.db")
        self.candle_buffer = {} # Pair -> Last Candle Data (for detecting close)
        
        # Load assets from training DB to ensure correct order
        print("Loading asset list from DB...")
        self.history_db = HistoricalMarket(n_assets=N_ASSETS)
        self.asset_map = self.history_db.assets # {id: symbol} (e.g., 370: '/EURGBP')
        self.ordered_ids = list(self.asset_map.keys())
        
        # Clean symbols for Kraken API (remove leading '/')
        self.kraken_symbols = [s.lstrip('/') for s in self.asset_map.values()]
        self.symbol_to_id = {s.lstrip('/'): i for i, s in self.asset_map.items()}
        
        # Map DB ID to Action Index
        self.db_id_to_action_idx = {db_id: i for i, db_id in enumerate(self.ordered_ids)}
        
        self.api_symbol_map = {} # DB Symbol -> REST API Symbol (XXBTZUSD)
        self.ws_symbol_map = {} # DB Symbol -> WS V2 Symbol (BTC/USD)
        self.reverse_api_map = {} # WS Symbol -> DB Symbol
        
        print(f"Tracking {len(self.kraken_symbols)} assets.")

        # Initialize Encoder
        self.encoder = FastMultiAssetStateEncoder(n_assets=N_ASSETS, lookback=LOOKBACK, device='cpu')
        self.encoder.register_assets(self.kraken_symbols) # Use clean symbols

        # Initialize Model
        print("Initializing JAX model...")
        self.model = MuZeroNetworks(state_dim=64, action_space_size=192, support_size=601)
        self.trainer = MuZeroTrainer(model=self.model)
        
        # Create dummy state to load params into
        rng = jax.random.PRNGKey(0)
        dummy_obs = jnp.ones((1, 27, 64, 64))
        self.trainer.init_train_state(rng, dummy_obs)
        
        # Load Checkpoint
        print(f"Loading checkpoint from {CHECKPOINT_DIR}...")
        try:
            from flax.training import checkpoints
            self.trainer.state = checkpoints.restore_checkpoint(
                ckpt_dir=CHECKPOINT_DIR,
                target=self.trainer.state
            )
            print("Checkpoint loaded successfully.")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            raise

        # Inference function (MCTS)
        self.search = create_mctx_search(self.model, num_simulations=SIMULATIONS, use_gumbel=False)

    async def _build_symbol_map(self):
        """Build mapping between DB symbols and Kraken API symbols"""
        print("Fetching tradeable pairs from Kraken to build symbol map...")
        async with KrakenClient() as client:
            pairs = await client.get_tradeable_pairs()
            
            # Create lookup tables
            wsname_to_api = {}
            altname_to_api = {}
            api_to_wsname = {}
            
            for api_name, info in pairs.items():
                if 'wsname' in info:
                    wsname_to_api[info['wsname']] = api_name
                    api_to_wsname[api_name] = info['wsname']
                if 'altname' in info:
                    altname_to_api[info['altname']] = api_name
                    
            # Map our DB symbols
            mapped_count = 0
            for db_symbol in self.kraken_symbols:
                api_symbol = None
                ws_symbol = None
                
                # Try 1: Exact wsname match
                if db_symbol in wsname_to_api:
                    api_symbol = wsname_to_api[db_symbol]
                # Try 2: Altname match
                elif db_symbol.replace('/', '') in altname_to_api:
                    api_symbol = altname_to_api[db_symbol.replace('/', '')]
                # Try 3: Substitution
                if not api_symbol:
                    variants = [
                        db_symbol.replace('BTC', 'XBT'),
                        db_symbol.replace('DOGE', 'XDG'),
                        db_symbol.replace('USD', 'ZUSD'),
                    ]
                    for var in variants:
                        if var in wsname_to_api:
                            api_symbol = wsname_to_api[var]
                            break
                        if var.replace('/', '') in altname_to_api:
                            api_symbol = altname_to_api[var.replace('/', '')]
                            break
                            
                if api_symbol:
                    # Found REST Symbol. Now derive WS Symbol.
                    self.api_symbol_map[db_symbol] = api_symbol
                    
                    # WS V2 Fix: Must use ISO format (BTC/USD, not XBT/USD)
                    base_ws = api_to_wsname.get(api_symbol, db_symbol)
                    ws_symbol = base_ws.replace('XBT', 'BTC').replace('XDG', 'DOGE')
                    
                    self.ws_symbol_map[db_symbol] = ws_symbol
                    self.reverse_api_map[ws_symbol] = db_symbol
                    
                    mapped_count += 1
                else:
                    print(f"Warning: Could not find Kraken API symbol for '{db_symbol}'")
            
            print(f"Successfully mapped {mapped_count}/{len(self.kraken_symbols)} symbols.")

    async def warmup(self):
        """Fetch historical data to fill encoder buffers"""
        print("Warming up encoder with historical data (REST API)...")
        async with KrakenClient() as client:
            now = int(time.time())
            since = now - (LOOKBACK * 60 * 3) # 3 hours buffer to be safe
            
            print(f"Fetching history for {len(self.kraken_symbols)} pairs...")
            history_buffer = {} # timestamp -> {symbol: data}
            
            # Fetch sequentially/batched to respect rate limits
            for i, pair in enumerate(self.kraken_symbols):
                api_pair = self.api_symbol_map.get(pair)
                if not api_pair:
                    continue

                try:
                    # Interval 1 minute
                    ohlc = await client.get_ohlc(api_pair, interval=1, since=since)
                    if not ohlc:
                        continue
                        
                    # ohlc is list of [time, o, h, l, c, vwap, vol, count]
                    for candle in ohlc:
                        ts = int(candle[0])
                        if ts not in history_buffer:
                            history_buffer[ts] = {}
                        
                        history_buffer[ts][pair] = {
                            'open': float(candle[1]),
                            'high': float(candle[2]),
                            'low': float(candle[3]),
                            'close': float(candle[4]),
                            'volume': float(candle[6])
                        }
                    
                    # Simple rate limit handling: sleep every 5 requests
                    if (i + 1) % 5 == 0:
                        print(f"  Fetched {i+1}/{len(self.kraken_symbols)} pairs...")
                        await asyncio.sleep(1.0)
                        
                except Exception as e:
                    print(f"Error fetching {pair} ({api_pair}): {e}")

            # Sort by time and replay
            timestamps = sorted(history_buffer.keys())
            print(f"Replaying {len(timestamps)} minutes of history...")
            
            for ts in timestamps:
                self.encoder.update(history_buffer[ts])
                # Keep latest state updated
                self.market_state.update(history_buffer[ts])
                
            print("Warmup complete.")

    async def handle_ticker(self, data):
        pass # Not used

    async def handle_ohlc(self, msg: List):
        # Msg format: [channelID, [data], channelName, pair]
        # Data: [time, etime, o, h, l, c, vwap, vol, count]
        try:
            if not isinstance(msg, list) or len(msg) < 4:
                return
                
            ws_pair = msg[-1] # e.g. BTC/USD
            data = msg[1]
            
            # Map back to DB symbol
            db_pair = self.reverse_api_map.get(ws_pair)
            if not db_pair:
                return

            # Parse data
            # Use int(float(...)) to be safe against '123.0' strings
            current_ts = int(float(data[0]))
            
            current_data = {
                'timestamp': current_ts,
                'open': float(data[2]),
                'high': float(data[3]),
                'low': float(data[4]),
                'close': float(data[5]),
                'vwap': float(data[6]),
                'volume': float(data[7]),
                'count': int(float(data[8] or 0)) # Handle count safely
            }

            # Check for candle close
            if db_pair in self.candle_buffer:
                prev_data = self.candle_buffer[db_pair]
                if current_ts > prev_data['timestamp']:
                    # Previous candle closed! Save to DB.
                    await self.save_candle(db_pair, prev_data)
            
            # Update buffer
            self.candle_buffer[db_pair] = current_data

            # Update state for trading
            self.market_state[db_pair] = {
                'close': current_data['close'],
                'volume': current_data['volume'],
                'high': current_data['high'],
                'low': current_data['low'],
                'open': current_data['open']
            }
        except Exception as e:
            logging.error(f"WS Parse Error: {e}")

    async def save_candle(self, pair: str, data: Dict):
        """Save closed candle to database"""
        try:
            # clean_pair = pair.lstrip('/')
            # Find Asset ID
            # symbol_to_id stores keys like 'XBT/USD' (from lstrip)
            asset_id = self.symbol_to_id.get(pair)
            if not asset_id:
                 # Try stripping slash?
                 asset_id = self.symbol_to_id.get(pair.lstrip('/'))
                 
            if asset_id:
                # [timestamp, open, high, low, close, vwap, volume, count]
                record = [
                    data['timestamp'],
                    data['open'],
                    data['high'],
                    data['low'],
                    data['close'],
                    data['vwap'],
                    data['volume'],
                    data['count']
                ]
                await self.db.insert_ohlcv(asset_id, [record])
                # logging.info(f"Saved candle for {pair} at {data['timestamp']}")
        except Exception as e:
            logging.error(f"DB Save Error: {e}")

    async def trade_step(self):
        """Execute one trading step based on current market state"""
        timestamp = time.strftime('%H:%M:%S')
        
        # 1. Update encoder with collected data (snapshot of latest state)
        self.encoder.update(self.market_state)
        
        # 2. Encode
        obs = self.encoder.encode()
        # Convert to JAX
        obs_jax = jnp.array(np.expand_dims(obs.numpy(), 0), dtype=jnp.float32)
        
        # 3. MCTS Search
        rng = jax.random.PRNGKey(int(time.time()))
        action_jax, policy, value = self.search(self.trainer.state.params, rng, obs_jax)
        action = int(action_jax[0])
        
        # 4. Decode Action
        asset_idx = action // 3
        action_type = action % 3 # 0=Sell, 1=Hold, 2=Buy
        
        # 5. Execute
        # Map asset_idx back to DB ID and Symbol
        db_id = self.ordered_ids[asset_idx]
        symbol = self.asset_map[db_id]
        clean_symbol = symbol.lstrip('/')
        
        price = self.market_state.get(clean_symbol, {}).get('close', 0)
        
        action_str = ["SELL", "HOLD", "BUY"][action_type]
        
        # Only log if action is not HOLD, or occasionally
        if action_type != 1:
            print(f"\n[{timestamp}] ACTION: {action_str} {clean_symbol} @ {price:.5f}")
            print(f"  Value Est: {value['root_value'][0]:.2f}")
        else:
            print(f"[{timestamp}] HOLD (Value: {value['root_value'][0]:.2f})", end='\r')
        
        # Update Portfolio (Paper)
        actions = np.ones(N_ASSETS, dtype=np.int32)
        actions[asset_idx] = action_type
        sizes = np.full(N_ASSETS, 0.5, dtype=np.float32)
        
        # Prices map for portfolio
        current_prices = {}
        for s, d in self.market_state.items():
             if s in self.symbol_to_id:
                 pid = self.symbol_to_id[s]
                 if pid in self.db_id_to_action_idx:
                     idx = self.db_id_to_action_idx[pid]
                     current_prices[idx] = d['close']
                     
        if current_prices:
            self.portfolio.execute_actions(actions, sizes, current_prices)
            
            # Log Portfolio Stats
            val = self.portfolio.state.total_value
            cash = self.portfolio.state.cash
            n_pos = len(self.portfolio.state.positions)
            
            if action_type != 1:
                print(f"  Portfolio: ${val:.2f} | Cash: ${cash:.2f} | Positions: {n_pos}")

    async def trade_loop(self):
        """Runs the trading logic every 60 seconds"""
        print("\nStarting Trade Loop (60s interval)...")
        while True:
            # Align to minute boundary
            now = time.time()
            next_minute = (int(now) // 60 + 1) * 60
            sleep_time = next_minute - now
            
            # Sleep until the minute closes
            await asyncio.sleep(sleep_time)
            
            # Wait a small buffer (e.g. 1s) for final WS updates to arrive for that minute
            await asyncio.sleep(1.0)
            
            try:
                await self.trade_step()
            except Exception as e:
                logging.error(f"Trade Step Error: {e}")

    async def main_loop(self):
        # 0. Connect DB
        await self.db.connect()

        # 1. Build Symbol Map
        await self._build_symbol_map()
        
        # 2. Warmup
        await self.warmup()
        
        # Connect to WS
        print("Connecting to Kraken WebSocket...")
        ws = KrakenWebSocket(on_ticker=self.handle_ticker, on_ohlc=self.handle_ohlc)
        await ws.connect()
        
        # Subscribe to all pairs using WS symbols
        print(f"Subscribing to {len(self.kraken_symbols)} pairs...")
        
        # Collect valid WS symbols for subscription
        valid_ws_symbols = []
        for s in self.kraken_symbols:
            if s in self.ws_symbol_map:
                valid_ws_symbols.append(self.ws_symbol_map[s])
        
        chunk_size = 20
        for i in range(0, len(valid_ws_symbols), chunk_size):
            chunk = valid_ws_symbols[i:i+chunk_size]
            await ws.subscribe(chunk, channels=['ohlc'])
            await asyncio.sleep(0.5) # Gentle subscribe
            
        print("Listening for live data...")
        
        # Run concurrently
        await asyncio.gather(
            ws.listen(),
            self.trade_loop()
        )

if __name__ == "__main__":
    trader = PaperTrader()
    try:
        asyncio.run(trader.main_loop())
    except KeyboardInterrupt:
        print("\nPaper trading stopped by user.")
    except Exception as e:
        print(f"\nFatal error: {e}")