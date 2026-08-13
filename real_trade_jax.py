#!/usr/bin/env python3
"""
JAX Real Trading Script
Connects to Kraken Real-Time API via WebSocket for Data,
and REST API for Execution.

WARNING: THIS SCRIPT TRADES REAL MONEY.
Ensure you have configured `config/real_trade_config.py` correctly.

------------------------------------------------------------------------------
STATUS (2026-07-05): BROKEN / LEGACY. This is the old MuZero live-trader path.
It depends on mcts_trader.agent.real_portfolio.RealPortfolioManager, which is
not present in this working tree (see the guarded import below) and is not
being resurrected. Live trading (gate 4 of the deployment ladder) is now
planned on the `trader/` package instead — see the paper-trading loop at
`runs/820_paper` and `tasks/TASK_820_SHIP_TREND_CTA.md` for the current plan.
Do not use this file for live/paper trading; kept only for reference/history.
------------------------------------------------------------------------------
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
from mcts_trader.jax_agent import VALUE_SUPPORT_SIZE
from mcts_trader.training.config import TrainingConfig
from mcts_trader.training.setup import setup_model_and_trainer
from mcts_trader.jax_agent.mctx_search import create_mctx_search
from mcts_trader.data.history import HistoricalMarket
from mcts_trader.encoder.state_fast import FastMultiAssetStateEncoder

from mcts_trader.data.kraken_client import KrakenClient
from mcts_trader.data.stream import KrakenWebSocket
from mcts_trader.data.database import Database
from config import real_trade_config as cfg
# TASK_888 (D38): the ONE checkpoint resolver + fail-closed loader, shared with
# eval_vector.py / train_jax_fast.py / experiments/857_vhat_distill/distill.py.
from mcts_trader.jax_agent import checkpoint_io

# real_portfolio was never committed to this tree (see module docstring above).
# Guard the import so the module can at least be imported/inspected; the
# entrypoint below fails fast with a clear message instead of a NameError/
# ImportError traceback if anyone actually tries to run this legacy path.
try:
    from mcts_trader.agent.real_portfolio import RealPortfolioManager
    _REAL_PORTFOLIO_AVAILABLE = True
except ImportError:
    RealPortfolioManager = None
    _REAL_PORTFOLIO_AVAILABLE = False

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


def load_checkpoint_or_die(trainer, checkpoint_dir):
    """TASK_888 (D38 consequence order item 1 — the highest-consequence site in the queue):
    the ONE fail-closed checkpoint load for the live-trading path. Fixes two bugs that were,
    until this task, the SAME bug wearing two faces:

    (1) The pre-888 call here — ``self.trainer.load_checkpoint(cfg.CHECKPOINT_DIR)`` — DISCARDS
        its return value. ``trainer.load_checkpoint`` (trainer.py:1824) ends ``return
        restored_state``; it never assigns ``self.state`` itself (only ``init_train_state``
        does that, trainer.py:380). So this process restored a checkpoint from disk, threw it
        away, and traded the fresh ``PRNGKey(0)`` init from ``init_train_state`` on EVERY run,
        not just a bad one — ``trade_step`` reads ``self.trainer.state.params``, which was
        never reassigned to the loaded state.
    (2) Even with the return value captured, flax's ``restore_checkpoint`` silently returns
        ``target`` UNCHANGED on a nonexistent/unloadable checkpoint dir (D37) — never a raise,
        never ``None`` — so a bad ``cfg.CHECKPOINT_DIR`` would ALSO have logged "Checkpoint
        loaded successfully" while trading random-init weights.

    ``checkpoint_io.load_checkpoint_into`` closes both: it MUTATES ``trainer.state`` itself (the
    caller cannot forget to assign it) and REFUSES (raises) on anything that is not a real,
    complete load, resolved via the SAME function every other checkpoint guard in this program
    uses. No ``allow_fresh_init`` escape hatch is offered here, deliberately, unlike
    ``distill.py --allow-fresh-init``: this is the live-trading site, and there is no legitimate
    reason to ever trade random-init weights. Exits the process (matching this call's pre-888
    contract of ``try/except`` + ``sys.exit(1)``) rather than raising, so the fix cannot be
    silently bypassed by a future caller that forgets to catch a raised exception.

    The log line CLIPS its echo of the path (``checkpoint_io.clip_for_refusal``): the refusal
    message already embeds it once, so re-embedding it raw here doubled an operator-supplied
    string that is bounded only by ``MAX_ARG_STRLEN`` — measured, a ~130,000-char input produced
    a 390,559-char log line. What is refused does not change; only how it is spelled."""
    try:
        evidence = checkpoint_io.load_checkpoint_into(trainer, checkpoint_dir)
    except Exception as e:
        logging.critical(f"REFUSING to trade: checkpoint failed to load from "
                         f"{checkpoint_io.clip_for_refusal(checkpoint_dir)}: "
                         f"{checkpoint_io.clip_for_refusal(e, limit=4000, quote=False)}")
        sys.exit(1)
    logging.info(f"Checkpoint loaded: resolved={evidence['resolved_path']} "
                 f"step={evidence['step']} "
                 f"fingerprint={evidence['param_fingerprint'][:12]}...")
    return evidence


def build_model_and_trainer(n_assets=None, state_dim=None, use_prediction_head=None,
                            use_popart=None, use_vector_action=None, sims=None, seed=0):
    """THE model/trainer construction for this file — built through the CANONICAL
    ``TrainingConfig`` -> ``setup_model_and_trainer`` path, the same one train_jax_fast.py,
    eval_vector.py, distill.py, tools/paper_trade_vectorbt.py and the backfill use.

    TASK_888 (fix pass, the BLOCKER at this row's own site). This used to hand-build
    ``MuZeroNetworks(...)``/``MuZeroTrainer(...)`` inline in ``RealTrader.__init__``, and so
    silently drifted from every other load site: it never passed ``use_vector_action``, which
    defaults False (``muzero_model.py``), so the model it built had NO
    ``vector_action_embedding`` at all — this repo's ACTIVE BUILD. ``restore_checkpoint``
    finishes with ``from_state_dict(target, restored)``, which returns a pytree CONGRUENT TO THE
    TARGET, so a checkpoint leaf the target does not have is DROPPED WITHOUT A WORD. Measured on
    a real 64-asset checkpoint (``runs/861_ref/ckpt_ref_s0``, own fingerprint 83b8c96126fa...):
    202 of its 204 leaves restored into that target, losing
    ``vector_action_embedding.vec_act_proj.{kernel,bias}`` — the entire vector-action policy
    projection — while ``checkpoint_io.load_checkpoint_into`` reported ``loaded=True`` with a
    stable, well-formed but WRONG fingerprint (4c7918f1bf42...). Re-reproduced at tiny size in
    ``test_real_trade_checkpoint_888.py`` (253 of 255 leaves, 7a2ced92552c... vs the
    checkpoint's 315c86889deb...). The object-identity detector correctly sees a REAL load and
    the shape check can only compare leaves both trees have, so ``load_checkpoint_or_die`` would
    have logged a perfectly formatted TASK_888 evidence line over an untrained policy head.

    No load-time check can close that class (see ``loaded_param_shape_defect``'s docstring): the
    only countermeasure is that the TARGET IS THE CANONICAL ARCHITECTURE, which is what this
    function makes true here. ``use_vector_action`` defaults to False — the scalar architecture
    this file has always built, so the canon path is unchanged — and is overridable only via the
    explicit keyword argument (this module's own tests build the vector-action side
    deliberately). Binding that default to what a CHECKPOINT was actually saved with (so a wrong
    answer REFUSES instead of silently dropping/randomly-initialising
    ``vector_action_embedding``) is TASK_889's architecture-binding layer, not this row's:
    ``trainer_metadata.json`` does not record it yet, and ``config/real_trade_config.py`` carries
    no such knob.

    SCOPE OF THE CONSEQUENCE, stated exactly and no wider: this site is DORMANT today. The
    guarded ``RealPortfolioManager`` import fails in this tree (``_REAL_PORTFOLIO_AVAILABLE`` is
    False), so ``RealTrader()`` is never constructed, and ``cfg.CHECKPOINT_DIR`` names a path
    that does not exist. It was a landmine, not an active break.

    Every argument except ``use_vector_action`` defaults to the corresponding
    ``config/real_trade_config.py`` constant; overrides exist so the construction path can be
    unit-tested at a tiny size WITHOUT instantiating ``RealTrader`` (which needs a Kraken client,
    the live DB and a portfolio manager). Canon-faithful: ``model_size="small"`` is a neutral
    placeholder so the explicit ``state_dim`` always wins (``TrainingConfig.__post_init__`` only
    remaps state_dim from model_size when state_dim is at ITS default AND model_size !=
    "small"), reproducing exactly what the hand-built ``MuZeroNetworks(state_dim=cfg.STATE_DIM,
    ...)`` did."""
    n_assets_val = int(cfg.N_ASSETS if n_assets is None else n_assets)
    # Two config constants the canonical builder DERIVES rather than reads. Silently ignoring a
    # value the operator set in real_trade_config.py is the same class of defect this task
    # exists to close (a config that says one thing while the process does another), so they are
    # asserted instead. ACTION_SPACE_SIZE is only checked when n_assets comes from the config —
    # an explicit override (tests) is not the config speaking.
    if n_assets is None and int(cfg.ACTION_SPACE_SIZE) != n_assets_val * 5:
        raise RuntimeError(
            f"real_trade_config.py: ACTION_SPACE_SIZE={cfg.ACTION_SPACE_SIZE} but the canonical "
            f"model builder derives it as N_ASSETS*5 = {n_assets_val * 5}. Fix the config; "
            f"refusing to build a model whose action space is not the one configured.")
    if int(cfg.SUPPORT_SIZE) != int(VALUE_SUPPORT_SIZE):
        raise RuntimeError(
            f"real_trade_config.py: SUPPORT_SIZE={cfg.SUPPORT_SIZE} but the canonical model "
            f"builder uses the central VALUE_SUPPORT_SIZE={VALUE_SUPPORT_SIZE}. Fix the config; "
            f"refusing to build a value head with a support the config disagrees with.")
    tcfg = TrainingConfig(
        name="real_trade_jax",
        model_size="small",
        state_dim=int(cfg.STATE_DIM if state_dim is None else state_dim),
        use_prediction_loss=bool(cfg.USE_PREDICTION_HEAD if use_prediction_head is None
                                 else use_prediction_head),
        use_popart=bool(cfg.USE_POPART if use_popart is None else use_popart),
        # TASK_888R: no config knob exists for this (config/real_trade_config.py carries none —
        # see this function's docstring). False matches what this file has always built.
        # Binding it to what a checkpoint actually needs is TASK_889's scope, not this row's.
        vector_action=bool(use_vector_action) if use_vector_action is not None else False,
        sims=int(cfg.SIMULATIONS if sims is None else sims),
    )
    model, trainer = setup_model_and_trainer(tcfg, n_assets_val)
    dummy_obs = jnp.ones((1, 27, n_assets_val, n_assets_val))
    trainer.init_train_state(jax.random.PRNGKey(seed), dummy_obs)
    return model, trainer


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
        # Model architecture must match checkpoint (configured in real_trade_config.py).
        # TASK_888: built through the CANONICAL TrainingConfig -> setup_model_and_trainer path
        # (see build_model_and_trainer for the vector_action drift this closes), including the
        # PRNGKey(0) dummy init this block used to do inline.
        self.model, self.trainer = build_model_and_trainer()

        # Load Checkpoint (use orbax for modern checkpoints). TASK_888: fail-closed, and the
        # loaded state is actually installed on the trainer this time (see
        # load_checkpoint_or_die's docstring for the two-bug history this replaces).
        logging.info(f"Loading checkpoint from {cfg.CHECKPOINT_DIR}...")
        load_checkpoint_or_die(self.trainer, cfg.CHECKPOINT_DIR)

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
    if not _REAL_PORTFOLIO_AVAILABLE:
        sys.exit(
            "FATAL: legacy MuZero live-trader path is non-functional: "
            "mcts_trader.agent.real_portfolio was never committed; live "
            "trading (gate 4) is planned on the trader/ package — see "
            "tasks/TASK_820_SHIP_TREND_CTA.md"
        )
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
