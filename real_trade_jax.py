#!/usr/bin/env python3
"""
JAX Real Trading Script
Connects to Kraken Real-Time API via WebSocket for Data,
and REST API for Execution.

WARNING: THIS SCRIPT TRADES REAL MONEY.
Ensure you have configured `config/real_trade_config.py` correctly.
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
import asyncio
import os
import sys
import logging
from typing import Dict, List

# JAX Setup
os.environ['JAX_PLATFORM_NAME'] = 'cpu' # Inference on CPU is robust

# Project Imports
from mcts_trader.jax_agent.muzero_model import MuZeroNetworks
from mcts_trader.jax_agent.trainer import MuZeroTrainer
from mcts_trader.jax_agent.mctx_search import create_mctx_search
from mcts_trader.data.history import HistoricalMarket
from mcts_trader.encoder.state_fast import FastMultiAssetStateEncoder

from mcts_trader.data.kraken_client import KrakenClient
from mcts_trader.data.stream import KrakenWebSocket
from mcts_trader.data.database import Database
from mcts_trader.agent.real_portfolio import RealPortfolioManager
from config import real_trade_config as cfg

# Logging Setup
timestamp = time.strftime("%Y%m%d_%H%M%S")
log_file = f"real_trading_{timestamp}.log"

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', log_file)),
        logging.StreamHandler()
    ]
)

class RealTrader:
    def __init__(self):
        self.start_time = time.time()
        self.market_state = {} 
        self.candle_buffer = {}
        
        # 1. API Client
        if not cfg.API_KEY or not cfg.API_SECRET:
            logging.error("API Key/Secret missing! Set KRAKEN_API_KEY env var or config.")
            sys.exit(1)
            
        self.api_client = KrakenClient(cfg.API_KEY, cfg.API_SECRET)
        
        # 2. Database (for Asset List & History)
        self.db = Database("trading_data.db")
        
        logging.info("Loading asset list from DB...")
        self.history_db = HistoricalMarket(n_assets=cfg.N_ASSETS)
        self.asset_map = self.history_db.assets # {id: symbol}
        self.ordered_ids = list(self.asset_map.keys())
        
        # Clean symbols
        self.kraken_symbols = [s.lstrip('/') for s in self.asset_map.values()]
        self.symbol_to_id = {s.lstrip('/'): i for i, s in self.asset_map.items()}
        self.db_id_to_action_idx = {db_id: i for i, db_id in enumerate(self.ordered_ids)}
        
        # Symbol Maps (will be built via API)
        self.api_symbol_map = {} 
        self.ws_symbol_map = {} 
        self.reverse_api_map = {}
        
        # 3. Real Portfolio Manager
        # We need the ID->Symbol map for the portfolio
        portfolio_symbols = {i: self.asset_map[db_id].lstrip('/') for i, db_id in enumerate(self.ordered_ids)}
        
        self.portfolio = RealPortfolioManager(
            api_client=self.api_client,
            symbols=portfolio_symbols,
            mode=cfg.TRADING_MODE,
            leverage=cfg.LEVERAGE,
            market_threshold=cfg.MARKET_ORDER_THRESHOLD,
            limit_timeout=cfg.LIMIT_ORDER_TIMEOUT,
            max_drawdown=cfg.MAX_DRAWDOWN_HARD_STOP,
            dry_run=cfg.DRY_RUN
        )
        
        # 4. Encoder & Model
        self.encoder = FastMultiAssetStateEncoder(n_assets=cfg.N_ASSETS, lookback=cfg.LOOKBACK, device='cpu')
        self.encoder.register_assets(self.kraken_symbols)

        logging.info("Initializing JAX model...")
        self.model = MuZeroNetworks(state_dim=64, action_space_size=192, support_size=601)
        self.trainer = MuZeroTrainer(model=self.model)
        
        # Dummy init
        rng = jax.random.PRNGKey(0)
        dummy_obs = jnp.ones((1, 27, 64, 64))
        self.trainer.init_train_state(rng, dummy_obs)
        
        # Load Checkpoint
        logging.info(f"Loading checkpoint from {cfg.CHECKPOINT_DIR}...")
        try:
            from flax.training import checkpoints
            self.trainer.state = checkpoints.restore_checkpoint(
                ckpt_dir=cfg.CHECKPOINT_DIR,
                target=self.trainer.state
            )
            logging.info("Checkpoint loaded.")
        except Exception as e:
            logging.critical(f"Failed to load checkpoint: {e}")
            sys.exit(1)

        self.search = create_mctx_search(self.model, num_simulations=cfg.SIMULATIONS, use_gumbel=True)

    async def _build_symbol_map(self):
        """Build mapping between DB symbols and Kraken API symbols"""
        logging.info("Fetching tradeable pairs from Kraken...")
        # Assume session is already active from main_loop
        pairs = await self.api_client.get_tradeable_pairs()
        
        wsname_to_api = {}
        altname_to_api = {}
        api_to_wsname = {}
        
        for api_name, info in pairs.items():
            if 'wsname' in info:
                wsname_to_api[info['wsname']] = api_name
                api_to_wsname[api_name] = info['wsname']
            if 'altname' in info:
                altname_to_api[info['altname']] = api_name

        mapped_count = 0
        manual_map = {'MATIC/USD': 'POL/USD', 'MATIC/EUR': 'POL/EUR'}
        DELISTED_PAIRS = {
            'EOS/USD', 'LUNA/USD', 'LUNA2/USD', 'UST/USD', 'FTT/USD', 
            'SRM/USD', 'ANT/USD', 'WAVES/USD', 'OMG/USD', 'KSM/USD'
        }

        for db_symbol in self.kraken_symbols:
            if db_symbol in DELISTED_PAIRS:
                logging.info(f"Skipping delisted symbol: {db_symbol}")
                continue

            api_symbol = None

            # Resolving logic...
            lookup_symbol = manual_map.get(db_symbol, db_symbol)

            # Try direct wsname lookup
            if lookup_symbol in wsname_to_api:
                api_symbol = wsname_to_api[lookup_symbol]
            # Try altname without slash
            elif lookup_symbol.replace('/', '') in altname_to_api:
                api_symbol = altname_to_api[lookup_symbol.replace('/', '')]

            # Enhanced variant generation if not found
            if not api_symbol:
                variants = [
                    lookup_symbol.replace('BTC', 'XBT'),
                    lookup_symbol.replace('DOGE', 'XDG'),
                    lookup_symbol.replace('USD', 'ZUSD'),
                    lookup_symbol.replace('EUR', 'ZEUR'),
                    lookup_symbol.replace('GBP', 'ZGBP'),
                    lookup_symbol.replace('CAD', 'ZCAD'),
                    lookup_symbol.replace('CHF', 'ZCHF'),
                    lookup_symbol.replace('JPY', 'ZJPY'),
                ]

                # Also try combinations (e.g., ETHBTC → ETH/XBT)
                if '/' not in lookup_symbol and len(lookup_symbol) > 3:
                    # Guess: First 3-4 chars = base, rest = quote
                    for split_pos in [3, 4]:
                        if split_pos < len(lookup_symbol):
                            base = lookup_symbol[:split_pos]
                            quote = lookup_symbol[split_pos:]

                            # Try with slash and XBT/Z prefix variants
                            combo_variants = [
                                f"{base}/{quote}",
                                f"{base}/{quote.replace('BTC', 'XBT')}",
                                f"{base.replace('BTC', 'XBT')}/{quote}",
                                f"{base}/{quote.replace('USD', 'ZUSD').replace('EUR', 'ZEUR').replace('GBP', 'ZGBP').replace('CAD', 'ZCAD').replace('CHF', 'ZCHF')}",
                                # Try 'X' prefix logic (XETH/XXBT)
                                f"X{base}/X{quote.replace('BTC', 'XBT')}",
                                f"X{base}X{quote.replace('BTC', 'XBT')}"
                            ]
                            variants.extend(combo_variants)

                for var in variants:
                    if var in wsname_to_api:
                        api_symbol = wsname_to_api[var]
                        break
                    elif var.replace('/', '') in altname_to_api:
                        api_symbol = altname_to_api[var.replace('/', '')]
                        break
                        
            if api_symbol:
                self.api_symbol_map[db_symbol] = api_symbol
                base_ws = api_to_wsname.get(api_symbol, db_symbol)
                # Store normalized symbol for consistent logging
                self.portfolio.symbols[self.db_id_to_action_idx[db_symbol]] = base_ws
                mapped_count += 1
            else:
                logging.warning(f"Could not map symbol: {db_symbol}")
                logging.debug(f"  Tried variants: {variants if 'variants' in locals() else 'None'}")
                # Optional: dump keys to see if we missed it
                # logging.debug(f"  Available wsnames sample: {list(wsname_to_api.keys())[:5]}")
        
        logging.info(f"Mapped {mapped_count}/{len(self.kraken_symbols)} symbols.")
                ws_symbol = base_ws.replace('XBT', 'BTC').replace('XDG', 'DOGE')
                
                self.ws_symbol_map[db_symbol] = ws_symbol
                self.reverse_api_map[ws_symbol] = db_symbol
                mapped_count += 1
            else:
                logging.warning(f"Could not map symbol: {db_symbol}")

        logging.info(f"Mapped {mapped_count}/{len(self.kraken_symbols)} symbols.")

    async def warmup(self):
        """Fetch historical data to fill encoder buffers"""
        logging.info("Warming up encoder with historical data (REST API)...")
        now = int(time.time())
        since = now - (cfg.LOOKBACK * 60 * 3) # 3 hours buffer
        
        logging.info(f"Fetching history for {len(self.kraken_symbols)} pairs...")
        history_buffer = {} # timestamp -> {symbol: data}
        
        # Use the persistent session in api_client if active, or it manages itself?
        # KrakenClient context manager handles session. 
        # But here we are inside main_loop's 'async with self.api_client'.
        # So we can just call methods.
        
        for i, pair in enumerate(self.kraken_symbols):
            api_pair = self.api_symbol_map.get(pair)
            if not api_pair: continue

            try:
                ohlc = await self.api_client.get_ohlc(api_pair, interval=1, since=since)
                if not ohlc: continue
                    
                for candle in ohlc:
                    ts = int(candle[0])
                    if ts not in history_buffer: history_buffer[ts] = {}
                    
                    history_buffer[ts][pair] = {
                        'open': float(candle[1]), 'high': float(candle[2]), 
                        'low': float(candle[3]), 'close': float(candle[4]), 
                        'volume': float(candle[6])
                    }
                
                if (i + 1) % 5 == 0:
                    logging.info(f"  Fetched {i+1}/{len(self.kraken_symbols)} pairs...")
                    await asyncio.sleep(1.0) # Rate limit niceness
                    
            except Exception as e:
                logging.error(f"Warmup fetch error {pair}: {e}")

        timestamps = sorted(history_buffer.keys())
        logging.info(f"Replaying {len(timestamps)} minutes of history...")
        
        for ts in timestamps:
            self.encoder.update(history_buffer[ts])
            self.market_state.update(history_buffer[ts])
            
        logging.info("Warmup complete.")

    async def handle_ohlc(self, msg: List):
        # ... (Same parsing logic as paper_trade_jax.py) ...
        try:
            if not isinstance(msg, list) or len(msg) < 4:
                return
            ws_pair = msg[-1]
            data = msg[1]
            db_pair = self.reverse_api_map.get(ws_pair)
            if not db_pair: return

            current_ts = int(float(data[0]))
            current_data = {
                'timestamp': current_ts,
                'open': float(data[2]), 'high': float(data[3]), 'low': float(data[4]),
                'close': float(data[5]), 'volume': float(data[7])
            }
            
            self.market_state[db_pair] = current_data
        except Exception as e:
            logging.error(f"WS Parse: {e}")

    async def handle_ticker(self, data):
        pass

    async def trade_step(self):
        """Execute one trading step"""
        # 1. Sync Portfolio State (Crucial Step)
        await self.portfolio.sync()
        
        # 2. Update Encoder & Inference
        self.encoder.update(self.market_state)
        obs = self.encoder.encode()
        obs_jax = jnp.array(np.expand_dims(obs.numpy(), 0), dtype=jnp.float32)
        
        # Get Valid Action Mask from Portfolio
        valid_mask = self.portfolio.get_valid_actions().flatten()
        mask_jax = jnp.array(np.expand_dims(valid_mask, 0), dtype=jnp.int8)

        # 3. MCTS
        rng = jax.random.PRNGKey(int(time.time()))
        action_jax, policy, value = self.search(
            self.trainer.state.params, rng, obs_jax, mask_jax
        )
        action = int(action_jax[0])
        
        # 4. Decode
        asset_idx = action // 3
        
        # Extract Confidence
        policy_probs = policy[0]
        confidence = float(policy_probs[action])
        
        # 5. Execute
        # We batch this if we have multiple agents, but here it's one action per step?
        # Wait, MCTS output is ONE action for the whole portfolio?
        # NO, the MuZero model outputs a policy over ALL actions (192).
        # We need to decide: Do we take just the TOP action? Or top K?
        # Paper trader took just the sampled action.
        
        # Let's stick to single action per step for now to be safe.
        actions = {asset_idx: action % 3} # 0=Sell, 1=Hold, 2=Buy
        confidences = {asset_idx: confidence}
        
        # Get Prices
        prices = {}
        for s, d in self.market_state.items():
             if s in self.symbol_to_id:
                 pid = self.symbol_to_id[s]
                 if pid in self.db_id_to_action_idx:
                     idx = self.db_id_to_action_idx[pid]
                     prices[idx] = d['close']

        if prices:
            await self.portfolio.execute_actions(actions, confidences, prices)

    async def _setup_websocket(self) -> KrakenWebSocket:
        logging.info("Connecting WS...")
        ws = KrakenWebSocket(on_ticker=self.handle_ticker, on_ohlc=self.handle_ohlc)
        await ws.connect()
        
        # Subscribe logic...
        valid_ws_symbols = [self.ws_symbol_map[s] for s in self.kraken_symbols if s in self.ws_symbol_map]
        
        chunk_size = 20
        for i in range(0, len(valid_ws_symbols), chunk_size):
            chunk = valid_ws_symbols[i:i+chunk_size]
            await ws.subscribe(chunk, channels=['ohlc'])
            await asyncio.sleep(0.5)
            
        return ws

    async def main_loop(self):
        # Initialize Client Session
        async with self.api_client as client:
            await self.db.connect()
            
            # Setup Maps
            await self._build_symbol_map()
            
            # Initialize Portfolio (Sync)
            await self.portfolio.initialize()
            
            # Warmup... (TODO: Full implementation)
            
            # Supervisory Loop
            while True:
                ws = None
                try:
                    ws = await self._setup_websocket()
                    
                    logging.info("Starting Trade Loop...")
                    
                    # We need a loop for trade_step
                    async def loop_logic():
                        run_seconds = cfg.RUN_HOURS * 3600 if cfg.RUN_HOURS > 0 else float('inf')
                        while True:
                            if time.time() - self.start_time >= run_seconds:
                                break
                            
                            # Align minute
                            now = time.time()
                            next_min = (int(now) // 60 + 1) * 60
                            await asyncio.sleep(next_min - now + 1)
                            
                            await self.trade_step()
                    
                    await asyncio.gather(ws.listen(), loop_logic())
                    
                    # If we get here, time limit reached
                    break

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logging.error(f"Error: {e}. Restarting...")
                    if ws: await ws.close()
                    await asyncio.sleep(5)
                finally:
                    if ws: await ws.close()

    async def shutdown(self):
        logging.info("Shutting down...")
        # Re-enter context to ensure valid session for cleanup
        async with self.api_client:
            await self.portfolio.close_all_positions()


if __name__ == "__main__":
    trader = RealTrader()
    try:
        if cfg.DRY_RUN:
            logging.warning("!!! DRY RUN MODE ACTIVE - NO REAL ORDERS WILL BE SENT !!!")
        else:
            logging.warning("!!! LIVE TRADING MODE - REAL FUNDS AT RISK !!!")
            logging.warning("Press Ctrl+C immediately to abort.")
            time.sleep(5) # Give user a chance to panic cancel
            
        asyncio.run(trader.main_loop())
    except KeyboardInterrupt:
        logging.warning("User Interrupted! Closing positions...")
        asyncio.run(trader.shutdown())
    except Exception as e:
        logging.critical(f"Fatal: {e}")
