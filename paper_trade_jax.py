import jax
import jax.numpy as jnp
import numpy as np
import time
import asyncio
import os
import sys
import logging
import argparse
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
INITIAL_CAPITAL = 100.0  # Small account test
RUN_HOURS = 6  # Auto-close positions and report after N hours (0 = unlimited)

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'paper_trading.log')),
        logging.StreamHandler()
    ]
)

class ActionLogger:
    def __init__(self, filepath=os.path.join('logs', "trade_log.csv"), new_session=True):
        # Create timestamped log for each session to avoid mixing data
        if new_session:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base, ext = os.path.splitext(filepath)
            self.filepath = f"{base}_{timestamp}{ext}"
        else:
            self.filepath = filepath

        # Always write fresh header for new session
        with open(self.filepath, 'w') as f:
            f.write("timestamp,symbol,action,price,size,fee,value_est,policy_conf,portfolio_value,cash\n")
        logging.info(f"Logging trades to: {self.filepath}")
                
    def log(self, timestamp, symbol, action, price, size, fee, value_est, policy_conf, port_val, cash):
        with open(self.filepath, 'a') as f:
            f.write(f"{timestamp},{symbol},{action},{price:.6f},{size:.6f},{fee:.6f},{value_est:.4f},{policy_conf:.4f},{port_val:.2f},{cash:.2f}\n")

class PaperTrader:
    def __init__(self, mode: str = 'SPOT'):
        # Initialize Logger
        self.logger = ActionLogger()
        self.start_time = time.time()
        self.trade_count = 0
        self.signal_count = 0
        self.prices_cache = {}  # Latest prices for position closing
        self.fx_rates_cache = {}  # Latest FX rates for currency conversion
        
        # Initialize Portfolio with Fees & Slippage
        # Kraken fees: ~0.26% Taker, ~0.16% Maker. Using 0.26% (26 bps) to be conservative.
        # For small accounts (<$1000), increase max position % to meet minimums
        max_pos_pct = 0.25 if INITIAL_CAPITAL < 1000 else 0.10
        self.portfolio = PortfolioManager(
            initial_capital=INITIAL_CAPITAL,
            max_position_pct=max_pos_pct,
            trading_fee_bps=26,
            slippage_bps=5,
            mode=mode
        )
        self.market_state = {} # {symbol: {close, high, low, volume, open}}
        
        # Database for saving history
        self.db = Database("trading_data.db")
        self.candle_buffer = {} # Pair -> Last Candle Data (for detecting close)
        
        # Load assets from training DB to ensure correct order
        logging.info("Loading asset list from DB...")
        self.history_db = HistoricalMarket(n_assets=N_ASSETS)
        self.asset_map = self.history_db.assets # {id: symbol} (e.g., 370: '/EURGBP')
        self.ordered_ids = list(self.asset_map.keys())
        
        # Clean symbols for Kraken API (remove leading '/')
        self.kraken_symbols = [s.lstrip('/') for s in self.asset_map.values()]
        self.symbol_to_id = {s.lstrip('/'): i for i, s in self.asset_map.items()}
        
        # Map DB ID to Action Index
        self.db_id_to_action_idx = {db_id: i for i, db_id in enumerate(self.ordered_ids)}
        
        # Register assets with Portfolio for correct logging (Fix ASSET_X)
        idx_to_symbol = {i: self.asset_map[db_id].lstrip('/') for i, db_id in enumerate(self.ordered_ids)}
        self.portfolio.register_assets(idx_to_symbol)
        
        self.api_symbol_map = {} # DB Symbol -> REST API Symbol (XXBTZUSD)
        self.ws_symbol_map = {} # DB Symbol -> WS V2 Symbol (BTC/USD)
        self.reverse_api_map = {} # WS Symbol -> DB Symbol
        
        logging.info(f"Tracking {len(self.kraken_symbols)} assets.")

        # Initialize Encoder
        self.encoder = FastMultiAssetStateEncoder(n_assets=N_ASSETS, lookback=LOOKBACK, device='cpu')
        self.encoder.register_assets(self.kraken_symbols) # Use clean symbols

        # Initialize Model
        logging.info("Initializing JAX model...")
        self.model = MuZeroNetworks(state_dim=64, action_space_size=192, support_size=601)
        self.trainer = MuZeroTrainer(model=self.model)
        
        # Create dummy state to load params into
        rng = jax.random.PRNGKey(0)
        dummy_obs = jnp.ones((1, 27, 64, 64))
        self.trainer.init_train_state(rng, dummy_obs)
        
        # Load Checkpoint
        logging.info(f"Loading checkpoint from {CHECKPOINT_DIR}...")
        try:
            from flax.training import checkpoints
            self.trainer.state = checkpoints.restore_checkpoint(
                ckpt_dir=CHECKPOINT_DIR,
                target=self.trainer.state
            )
            logging.info("Checkpoint loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load checkpoint: {e}")
            raise

        # Inference function (MCTS)
        self.search = create_mctx_search(self.model, num_simulations=SIMULATIONS, use_gumbel=True)

    async def _build_symbol_map(self):
        """Build mapping between DB symbols and Kraken API symbols"""
        logging.info("Fetching tradeable pairs from Kraken to build symbol map...")
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

            # Manual substitutions for renamed/delisted assets
            manual_map = {
                'MATIC/USD': 'POL/USD', # MATIC migrated to POL
                'MATIC/EUR': 'POL/EUR',
            }

            for db_symbol in self.kraken_symbols:
                api_symbol = None
                ws_symbol = None

                # Use substituted symbol for lookup if exists
                lookup_symbol = manual_map.get(db_symbol, db_symbol)

                # Try 1: Exact wsname match
                if lookup_symbol in wsname_to_api:
                    api_symbol = wsname_to_api[lookup_symbol]
                # Try 2: Altname match
                elif lookup_symbol.replace('/', '') in altname_to_api:
                    api_symbol = altname_to_api[lookup_symbol.replace('/', '')]

                # Try 3: Enhanced variant generation
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
                                ]
                                variants.extend(combo_variants)

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
                    logging.warning(f"Could not find Kraken API symbol for '{db_symbol}'")
            
            logging.info(f"Successfully mapped {mapped_count}/{len(self.kraken_symbols)} symbols.")

    async def warmup(self):
        """Fetch historical data to fill encoder buffers"""
        logging.info("Warming up encoder with historical data (REST API)...")
        async with KrakenClient() as client:
            now = int(time.time())
            since = now - (LOOKBACK * 60 * 3) # 3 hours buffer to be safe
            
            logging.info(f"Fetching history for {len(self.kraken_symbols)} pairs...")
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
                        logging.info(f"  Fetched {i+1}/{len(self.kraken_symbols)} pairs...")
                        await asyncio.sleep(1.0)
                        
                except Exception as e:
                    logging.error(f"Error fetching {pair} ({api_pair}): {e}")

            # Sort by time and replay
            timestamps = sorted(history_buffer.keys())
            logging.info(f"Replaying {len(timestamps)} minutes of history...")
            
            for ts in timestamps:
                self.encoder.update(history_buffer[ts])
                # Keep latest state updated
                self.market_state.update(history_buffer[ts])
                
            logging.info("Warmup complete.")

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
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        
        # 1. Update encoder with collected data (snapshot of latest state)
        self.encoder.update(self.market_state)
        
        # Prices map for portfolio
        current_prices = {}
        fx_rates = {}  # symbol -> price for FX conversion
        for s, d in self.market_state.items():
             if s in self.symbol_to_id:
                 pid = self.symbol_to_id[s]
                 if pid in self.db_id_to_action_idx:
                     idx = self.db_id_to_action_idx[pid]
                     current_prices[idx] = d['close']
             # Also store raw symbol->price for FX lookups
             fx_rates[s] = d['close']

        # Cache prices for close_all_positions
        self.prices_cache = current_prices.copy()
        self.fx_rates_cache = fx_rates.copy()

        # CRITICAL: Update portfolio with FX rates and current prices AT THE START of trade_step
        # This ensures get_valid_actions and subsequent portfolio ops have fresh state
        self.portfolio.fx_rates = fx_rates
        self.portfolio.update_prices(current_prices, self.fx_rates_cache)
        
        # 2. Encode
        obs = self.encoder.encode()
        # Convert to JAX
        obs_jax = jnp.array(np.expand_dims(obs.numpy(), 0), dtype=jnp.float32)
        
        # 2b. Get Valid Action Mask (now uses fresh prices/state)
        # Shape: (N_ASSETS, 3) -> Flatten to (N_ASSETS * 3,)
        valid_mask = self.portfolio.get_valid_actions().flatten()
        mask_jax = jnp.array(np.expand_dims(valid_mask, 0), dtype=jnp.int8)
        
        # 3. MCTS Search
        rng = jax.random.PRNGKey(int(time.time()))
        
        # Pass observation and mask as separate arguments
        action_jax, policy, value = self.search(
            self.trainer.state.params, 
            rng, 
            obs_jax, 
            mask_jax
        )
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
        value_est = float(value['root_value'][0])
        
        # Extract policy confidence for chosen action
        policy_probs = policy[0] # (192,)
        policy_conf = float(policy_probs[action])
        
        # Update Portfolio (Paper)
        actions = np.ones(N_ASSETS, dtype=np.int32)
        actions[asset_idx] = action_type
        sizes = np.full(N_ASSETS, 0.5, dtype=np.float32)
        
        # Get portfolio state AFTER update_prices but BEFORE executing orders
        val = self.portfolio.state.total_value
        cash = self.portfolio.state.cash
        n_pos = len(self.portfolio.state.positions)

        # First check SL/TP triggers before new trades
        sl_tp_orders = []
        if current_prices:
            sl_tp_orders = self.portfolio.check_stop_loss_take_profit(current_prices, self.fx_rates_cache)
            for order in sl_tp_orders:
                reason = order.get('reason', 'unknown')
                pnl = order.get('pnl', 0)
                logging.info(f"*** {reason.upper()} TRIGGERED *** {order['type']} {order['symbol']} | PnL: ${pnl:+.2f}")
                self.logger.log(timestamp, order['symbol'], order['type'], order['price'], order['size'],
                               order['price'] * order['size'] * self.portfolio.trading_fee_rate,
                               value_est, "POLICY_TRIGGER", val, cash) # Use latest val/cash
        else:
            logging.warning("No current prices available for SL/TP checks or execution.")

        executed_orders = []
        if current_prices:
            executed_orders = self.portfolio.execute_actions(actions, sizes, current_prices, self.fx_rates_cache)

        # Log Portfolio Stats
        
        # LOGGING
        # If order executed, log it details.
        # If executed_orders list is not empty, we have a trade.
        
        if executed_orders:
            for order in executed_orders:
                # Order keys: asset_idx, symbol, side, size, price, type
                o_symbol = order.get('symbol', clean_symbol)
                o_type = order.get('type', action_str)
                o_price = order.get('price', price)
                o_size = order.get('size', 0)
                o_cost = o_size * o_price
                o_fee = o_cost * self.portfolio.trading_fee_rate
                
                logging.info(f"EXECUTION: {o_type} {o_symbol} | Size: {o_size:.6f} | Price: {o_price:.5f} | Fee: ${o_fee:.4f}")
                self.logger.log(timestamp, o_symbol, o_type, o_price, o_size, o_fee, value_est, policy_conf, val, cash)
        else:
            # Even if no trade executed (Hold, or failed limits), log the INTENT if it wasn't just 'Hold'
            if action_type != 1:  # Not Hold
                # Check if this signal reinforces an existing position - trail TP if so
                if asset_idx in current_prices:
                    tp_updated = self.portfolio.reinforce_position(asset_idx, action_type, current_prices[asset_idx])
                    if tp_updated:
                        pos = self.portfolio.state.positions.get(asset_idx)
                        if pos:
                            logging.info(f"TRAILING TP: {action_str} {clean_symbol} @ {price:.5f} | New TP: {pos.take_profit:.5f}")
                            self.logger.log(timestamp, clean_symbol, f"TRAIL_TP_{action_str}", price, 0, 0, value_est, policy_conf, val, cash)
                    else:
                        logging.info(f"SIGNAL: {action_str} {clean_symbol} @ {price:.5f} (Not Executed/Limits?)")
                        self.logger.log(timestamp, clean_symbol, f"SIGNAL_{action_str}", price, 0, 0, value_est, policy_conf, val, cash)
                else:
                    logging.info(f"SIGNAL: {action_str} {clean_symbol} @ {price:.5f} (No Price)")
                    self.logger.log(timestamp, clean_symbol, f"SIGNAL_{action_str}", price, 0, 0, value_est, policy_conf, val, cash)
            else:
                logging.info(f"HOLD (Value: {value_est:.2f})")
        
        if action_type != 1 or executed_orders:
             logging.info(f"  Portfolio: ${val:.2f} | Cash: ${cash:.2f} | Positions: {n_pos}")
             if n_pos > 0:
                 logging.info("  Open Positions:")
                 for pid, pos in self.portfolio.state.positions.items():
                     logging.info(f"    - {pos.symbol:<10} {pos.side.upper():<5} | Size: {pos.size:.4f} | Entry: {pos.entry_price:.4f} | PnL: {pos.pnl_percent:+.2f}%")

        # Track counts
        if sl_tp_orders:
            self.trade_count += len(sl_tp_orders)
        if executed_orders:
            self.trade_count += len(executed_orders)
        elif action_type != 1:
            self.signal_count += 1

    def close_all_positions(self):
        """Close all open positions at current prices with proper FX conversion"""
        from mcts_trader.agent.portfolio import get_quote_to_usd_rate

        logging.info("=" * 60)
        logging.info("CLOSING ALL POSITIONS")
        logging.info("=" * 60)

        closed = []
        for idx, pos in list(self.portfolio.state.positions.items()):
            price = self.prices_cache.get(idx, pos.current_price)

            # Get FX rate for this position's quote currency
            fx_rate = get_quote_to_usd_rate(pos.quote_currency, self.fx_rates_cache)

            if pos.side == 'long':
                # Sell to close long: receive proceeds minus slippage/fee
                exec_price = price * (1 - self.portfolio.slippage_rate)
                value = pos.size * exec_price
                value_usd = value * fx_rate
                fee_usd = value_usd * self.portfolio.trading_fee_rate
                self.portfolio.state.cash += (value_usd - fee_usd)
                pnl = (exec_price - pos.entry_price) * pos.size * fx_rate - fee_usd
            else:
                # Buy to close short: pay current price plus slippage/fee
                exec_price = price * (1 + self.portfolio.slippage_rate)
                value = pos.size * exec_price
                value_usd = value * fx_rate
                fee_usd = value_usd * self.portfolio.trading_fee_rate
                # Short close: we pay to buy back, cash decreases
                self.portfolio.state.cash -= (value_usd + fee_usd)
                pnl = (pos.entry_price - exec_price) * pos.size * fx_rate - fee_usd

            closed.append({
                'symbol': pos.symbol,
                'side': pos.side,
                'size': pos.size,
                'entry': pos.entry_price,
                'exit': price,
                'pnl': pnl,
                'fee': fee_usd
            })
            logging.info(f"  Closed {pos.side.upper()} {pos.symbol}: {pos.size:.6f} @ {price:.4f} | PnL: ${pnl:+.2f}")

        self.portfolio.state.positions.clear()
        self.portfolio.state.total_value = self.portfolio.state.cash
        return closed

    def print_report(self):
        """Print final trading session report"""
        elapsed = time.time() - self.start_time
        hours = elapsed / 3600

        final_value = self.portfolio.state.total_value
        total_return = (final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        hourly_return = total_return / hours if hours > 0 else 0

        logging.info("=" * 60)
        logging.info("PAPER TRADING SESSION REPORT")
        logging.info("=" * 60)
        logging.info(f"Duration: {hours:.2f} hours ({elapsed/60:.0f} minutes)")
        logging.info(f"Initial Capital: ${INITIAL_CAPITAL:.2f}")
        logging.info(f"Final Value: ${final_value:.2f}")
        logging.info(f"Total Return: {total_return:+.2f}%")
        logging.info(f"Hourly Return: {hourly_return:+.2f}%/hr")
        logging.info(f"Annualized (approx): {hourly_return * 24 * 365:+.1f}%")
        logging.info("-" * 60)
        logging.info(f"Trades Executed: {self.trade_count}")
        logging.info(f"Signals Blocked: {self.signal_count} (min size / limits)")
        logging.info(f"Max Drawdown: {self.portfolio.state.drawdown*100:.2f}%")
        logging.info("=" * 60)

    async def trade_loop(self):
        """Runs the trading logic every 60 seconds"""
        run_seconds = RUN_HOURS * 3600 if RUN_HOURS > 0 else float('inf')
        logging.info(f"Starting Trade Loop (60s interval, {RUN_HOURS}h limit)...")

        while True:
            # Check time limit
            if time.time() - self.start_time >= run_seconds:
                logging.info(f"*** Time limit reached ({RUN_HOURS} hours) ***")
                break

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

        # Time's up - close positions and report
        self.close_all_positions()
        self.print_report()

    async def _setup_websocket(self) -> KrakenWebSocket:
        logging.info("Connecting to Kraken WebSocket...")
        ws = KrakenWebSocket(on_ticker=self.handle_ticker, on_ohlc=self.handle_ohlc)
        await ws.connect()
        
        logging.info(f"Subscribing to {len(self.kraken_symbols)} pairs...")
        
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
            
        logging.info("Listening for live data...")
        return ws

    async def main_loop(self):
        # 0. Connect DB
        await self.db.connect()

        # 1. Build Symbol Map
        MAX_API_RETRIES = 5
        RETRY_DELAY = 10  # seconds

        for attempt in range(1, MAX_API_RETRIES + 1):
            try:
                await self._build_symbol_map()
                logging.info("Kraken API symbol map built successfully.")
                break  # Exit loop if successful
            except Exception as e:
                logging.error(f"Failed to build Kraken API symbol map (attempt {attempt}/{MAX_API_RETRIES}): {e}")
                if attempt < MAX_API_RETRIES:
                    logging.info(f"Retrying in {RETRY_DELAY} seconds...")
                    await asyncio.sleep(RETRY_DELAY)
                else:
                    logging.critical(f"Could not reach Kraken API after {MAX_API_RETRIES} attempts. Exiting.")
                    sys.exit(1)
        
        # 2. Warmup
        await self.warmup()
        
        # Main supervisory loop for resilience
        while True:
            ws = None 
            try:
                # 3. Connect to WS and subscribe
                ws = await self._setup_websocket()
                
                logging.info("Starting main concurrent tasks (WebSocket listener and trade loop)...")
                
                # Run concurrently
                await asyncio.gather(
                    ws.listen(),
                    self.trade_loop()
                )
                
                # If gather completes without exception, check if trade_loop finished naturally
                run_seconds = RUN_HOURS * 3600 if RUN_HOURS > 0 else float('inf')
                if time.time() - self.start_time >= run_seconds:
                    logging.info("Trade loop finished due to time limit. Exiting supervisory loop.")
                    break # Exit the supervisory loop

            except asyncio.CancelledError:
                logging.info("Main tasks cancelled. Shutting down gracefully.")
                break # Exit the supervisory loop if explicitly cancelled (e.g. by KeyboardInterrupt)
            except Exception as e:
                logging.error(f"Main concurrent tasks failed with unexpected error: {e}. Re-initializing and restarting in 5 seconds...")
                # Continue the loop to attempt restart
            finally:
                if ws:
                    await ws.close()
                await asyncio.sleep(5) # Small delay before attempting to restart

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='JAX MuZero Paper Trader')
    parser.add_argument('--margin', action='store_true', help='Enable Margin Trading (Shorts)')
    args = parser.parse_args()
    
    mode = 'MARGIN' if args.margin else 'SPOT'
    logging.info(f"Starting Paper Trader in {mode} mode...")
    
    trader = PaperTrader(mode=mode)
    try:
        logging.info("Press Ctrl+C to stop and liquidate positions.")
        asyncio.run(trader.main_loop())
    except KeyboardInterrupt:
        logging.info("\nPaper trading stopped by user (Ctrl+C). Closing all positions...")
        trader.close_all_positions()
        try:
            trader.print_report()
        except Exception as report_e:
            logging.error(f"Error generating final report: {report_e}")
        time.sleep(1) # Give logging a moment to flush
        sys.exit(0)
    except Exception as e:
        logging.critical(f"\nFatal error: {e}")
        # Attempt to close positions on fatal error too? 
        # Might be risky if state is corrupted, but arguably safer than leaving open positions.
        # Let's keep it safe and just exit for now on fatal error.
        sys.exit(1)