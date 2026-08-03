"""
VectorBT-compatible Trading Environment

Uses lightweight position tracking with the same accounting formulas as VectorBT.
This ensures consistency between training and paper trading.

Key benefits over custom PortfolioManager:
- Standard accounting (same as vectorbt.Portfolio)
- No FX handling bugs (positions valued in USD directly)
- Simpler position logic (no complex margin calculations)
- Easy validation against VectorBT backtests

Data Sources (pick one):
1. synthetic_level: Use synthetic market generator (0-3)
2. prices: Direct numpy array of prices (n_steps, n_assets)
3. db_path: Load from SQLite database

Usage:
    # Synthetic data
    env = VectorBTTradingEnv(synthetic_level=0, n_assets=64)

    # Database data
    env = VectorBTTradingEnv(
        db_path='trading_data.db',
        n_assets=64,
        start_date='2024-12-01',
        end_date='2024-12-15',
    )

    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field

from mcts_trader.jax_agent.burning_money import BurningMoneyConfig, compute_burn_numpy
from mcts_trader.utils.transforms import symlog


# =============================================================================
# Hard Liquidation Constants (TASK_ENV_HARD_LIQUIDATION)
# =============================================================================
# Threshold for bankruptcy detection - terminate when portfolio falls below this
# fraction of initial capital. Prevents "death spiral" of negative cash/ghost shares.
LIQUIDATION_THRESHOLD = 0.01  # 1% of initial capital

# Minimum cash required to execute new buy orders. Prevents executing trades
# when functionally bankrupt.
MIN_CASH_THRESHOLD = 10.0  # $10 minimum

# Symlog-equivalent large penalty for bankruptcy (before symlog transformation)
# symlog(10) ≈ 2.4, so -50 raw → symlog ≈ -3.9 (significant negative signal)
LIQUIDATION_PENALTY_RAW = -50.0


@dataclass
class LimitOrder:
    """
    Pending limit order - persists across timesteps until filled or expired.

    Unlike IOC (Immediate-or-Cancel), these orders sit in the book and wait
    for favorable prices. This is the whole point of limit orders:
    - Wait for better execution price
    - Earn maker rebates (lower fees)
    - Express price predictions with time horizon

    Parameters (limit_price, ttl) come from model's limit order prediction head:
    - High certainty prediction → aggressive price, long TTL
    - Low certainty → conservative price, short TTL
    - Can integrate with Neural Market Physics (σ, α predictions)
    """
    asset_idx: int
    side: str  # 'buy' or 'sell'
    size: float  # Order size in shares
    limit_price: float  # From model prediction or default offset
    reserved_cash: float = 0.0  # Cash reserved for buy orders
    created_step: int = 0  # Step when order was created
    ttl: int = 100  # Time-to-live in steps (from model or default)

    def can_fill(self, current_price: float) -> bool:
        """Check if limit order can be filled at current price"""
        if self.side == 'buy':
            # Limit buy fills when price drops to or below limit
            return current_price <= self.limit_price
        else:  # sell
            # Limit sell fills when price rises to or above limit
            return current_price >= self.limit_price

    def is_expired(self, current_step: int) -> bool:
        """Check if order has exceeded its TTL"""
        return (current_step - self.created_step) >= self.ttl


@dataclass
class Position:
    """Simple position tracker - VectorBT compatible"""
    asset_idx: int
    side: str  # 'long' or 'short'
    size: float  # Number of shares (positive)
    entry_price: float  # Average entry price
    stop_loss: float = 0.0  # Absolute price for stop loss (0 = disabled)
    take_profit: float = 0.0  # Absolute price for take profit (0 = disabled)

    def value(self, current_price: float) -> float:
        """Current market value"""
        if self.side == 'long':
            return self.size * current_price
        else:  # short
            # Short value = entry_value - current_value + entry_value
            # = 2 * entry_value - size * current_price
            return 2 * self.size * self.entry_price - self.size * current_price

    def pnl(self, current_price: float) -> float:
        """Unrealized PnL"""
        if self.side == 'long':
            return self.size * (current_price - self.entry_price)
        else:  # short
            return self.size * (self.entry_price - current_price)


@dataclass
class PortfolioState:
    """Portfolio state - VectorBT compatible"""
    cash: float
    positions: Dict[int, Position] = field(default_factory=dict)
    pending_orders: Dict[int, LimitOrder] = field(default_factory=dict)  # asset_idx -> order

    def total_value(self, prices: Dict[int, float]) -> float:
        """Total portfolio value (cash + positions)"""
        position_value = sum(
            pos.value(prices.get(idx, pos.entry_price))
            for idx, pos in self.positions.items()
        )
        return self.cash + position_value

    def total_pnl(self, prices: Dict[int, float]) -> float:
        """Total unrealized PnL"""
        return sum(
            pos.pnl(prices.get(idx, pos.entry_price))
            for idx, pos in self.positions.items()
        )


class VectorBTTradingEnv(gym.Env):
    """
    Gym environment with VectorBT-compatible portfolio accounting.

    Observation: (27, n_assets, n_assets) tensor from FastMultiAssetStateEncoder
    Action: Discrete(n_assets * 3) - each asset has SELL(0), HOLD(1), BUY(2)
    Reward: Percentage portfolio return per step
    """

    def __init__(
        self,
        # Data source (one of these required)
        prices: Optional[np.ndarray] = None,  # (n_steps, n_assets) price array
        synthetic_level: Optional[int] = None,  # Use synthetic generator
        db_path: Optional[str] = None,  # Load from SQLite database
        schedule_path: Optional[str] = None,  # Load from training_schedule.json
        cache_path: Optional[str] = None,  # Numpy cache for fast loading (7min->5sec)

        # Database options
        start_date: Optional[str] = None,  # Date filter (YYYY-MM-DD or timestamp)
        end_date: Optional[str] = None,  # Date filter (YYYY-MM-DD or timestamp)
        specific_symbols: Optional[List[str]] = None,  # Use specific assets (for rainbow table training)
        # TASK_868 (D31 a2): capture the pre-fill 1-min real-print mask on the DB load path
        # (default OFF — canon training loads pay no extra memory). env_factory requests it only
        # when a caller asks for the causal frozen-cell open-gate mask.
        capture_raw_nonzero: bool = False,

        # Portfolio settings
        n_assets: int = 64,
        initial_cash: float = 10000.0,
        fee_rate: float = 0.0026,  # 26 bps per trade (or use trading_fee_bps)
        trading_fee_bps: int = None,  # Alternative: fee in basis points
        slippage_rate: float = 0.0005,  # 5 bps slippage (or use slippage_bps)
        slippage_bps: int = None,  # Alternative: slippage in basis points
        position_size: float = 0.10,  # 10% of portfolio per trade
        max_positions: int = 16,  # Max concurrent positions

        # Episode settings
        max_steps: int = 200,
        lookback: int = 60,

        # Synthetic settings
        synthetic_n_steps: int = 10000,

        # Trading mode
        allow_short: bool = True,  # Enable short selling

        # TP/SL Guardrails (Phase 1)
        tp_sl_enabled: bool = True,
        default_stop_loss_pct: float = 0.05,  # 5% stop loss
        default_take_profit_pct: float = 0.10,  # 10% take profit

        # Limit Orders - Persistent orders with model-driven TTL
        # Orders persist in pending_orders until filled or TTL expires
        # TTL and limit_price come from model predictions (or defaults)
        limit_orders_enabled: bool = True,
        maker_fee_bps: int = 10,  # 10 bps for limit orders (maker)
        limit_offset_pct: float = 0.002,  # 0.2% default offset (when no model prediction)
        default_limit_order_ttl: int = 100,  # Default TTL in steps (when no model prediction)

        # Multi-Action - multiple actions per timestep
        action_points: int = 1,  # Number of actions per timestep (1-4)
        borrow_rate_annual: float = 0.0,  # Annual borrow cost for shorts (e.g., 0.05 = 5%)

        # TASK_051: Burning Money (cash opportunity cost)
        burning_money: bool = False,
        burn_market_linked: bool = True,
        burn_base_rate: float = 0.00005,
        burn_random: bool = True,
        burn_random_prob: float = 0.03,
    ):
        super().__init__()

        self.borrow_rate_annual = borrow_rate_annual
        self.n_assets = n_assets
        self.initial_cash = initial_cash

        # TASK_051: Burning Money config
        self.burn_config = BurningMoneyConfig(
            enabled=burning_money,
            market_linked=burn_market_linked,
            base_decay_rate=burn_base_rate,
            random_burn=burn_random,
            random_burn_prob=burn_random_prob,
        )

        # Support both fee_rate (decimal) and trading_fee_bps (basis points)
        if trading_fee_bps is not None:
            self.fee_rate = trading_fee_bps / 10000.0
            self.trading_fee_bps = trading_fee_bps
        else:
            self.fee_rate = fee_rate
            self.trading_fee_bps = int(fee_rate * 10000)

        # Support both slippage_rate (decimal) and slippage_bps (basis points)
        if slippage_bps is not None:
            self.slippage_rate = slippage_bps / 10000.0
        else:
            self.slippage_rate = slippage_rate

        self.position_size = position_size
        self.max_positions = max_positions
        self.max_steps = max_steps
        self.lookback = lookback
        self.allow_short = allow_short

        # TP/SL Guardrails
        self.tp_sl_enabled = tp_sl_enabled
        self.default_stop_loss_pct = default_stop_loss_pct
        self.default_take_profit_pct = default_take_profit_pct

        # Limit Orders config (persistent orders with model-driven TTL)
        self.limit_orders_enabled = limit_orders_enabled
        self.n_actions_per_asset = 5  # Fixed: [Sell, Hold, Buy, Limit Sell, Limit Buy]
        self.maker_fee_rate = maker_fee_bps / 10000.0  # Convert bps to decimal
        self.limit_offset_pct = limit_offset_pct
        self.default_limit_order_ttl = default_limit_order_ttl

        # Multi-Action config
        self.action_points = action_points

        # Initialize price data
        self.schedule_path = schedule_path
        self.schedule_data = None
        # TASK_868 (D31): disclosures/capture on the DB load paths (any branch that
        # reaches a DB loader). asset_aliases_ = per-slot alias usage; raw_nonzero_ =
        # the pre-fill 1-min real-print mask (only when capture_raw_nonzero).
        self.capture_raw_nonzero = bool(capture_raw_nonzero)
        self.asset_aliases_ = []
        self.raw_nonzero_ = None

        if synthetic_level is not None:
            from mcts_trader.data.synthetic_market import SyntheticMarket
            self.synthetic = SyntheticMarket(
                level=synthetic_level,
                n_assets=n_assets,
                n_steps=synthetic_n_steps
            )
            self.prices = None  # Generated on reset
            self.db_path = None
            # Match SyntheticMarket symbol format: SYN_00, SYN_01, etc.
            self.symbols = [f"SYN_{i:02d}" for i in range(n_assets)]
        elif schedule_path is not None:
            # Load from training schedule (random period/run selection)
            self.synthetic = None
            self.db_path = db_path or 'trading_data.db'
            self.prices, self.symbols = self._load_from_schedule(schedule_path, n_assets)
            self.n_assets = len(self.symbols)
        elif db_path is not None:
            self.synthetic = None
            self.db_path = db_path
            self.specific_symbols = specific_symbols
            # Try loading from cache first (massive speedup: 7min -> 5sec)
            # Note: cache bypass when using specific_symbols (different assets each time),
            # and when capture_raw_nonzero is requested (the cache carries no raw mask).
            if (cache_path and specific_symbols is None and not self.capture_raw_nonzero
                    and self._try_load_cache(cache_path)):
                logging.info(f"Loaded from cache: {cache_path}")
            else:
                self.prices, self.symbols = self._load_database_prices(
                    db_path, n_assets, start_date, end_date, specific_symbols,
                    capture_raw_nonzero=self.capture_raw_nonzero,
                )
                # Save cache for next time (only if not using specific symbols)
                if cache_path and specific_symbols is None:
                    self._save_cache(cache_path)
            self.n_assets = len(self.symbols)  # May be fewer if not enough assets
        elif prices is not None:
            self.synthetic = None
            self.db_path = None
            self.prices = prices
            self.symbols = [f"ASSET_{i}" for i in range(n_assets)]
        else:
            raise ValueError("Must provide 'prices', 'synthetic_level', 'schedule_path', or 'db_path'")

        # Initialize encoder
        from mcts_trader.encoder.state_fast import FastMultiAssetStateEncoder
        self.encoder = FastMultiAssetStateEncoder(n_assets=n_assets, lookback=lookback)
        self.encoder.register_assets(self.symbols)

        # Action and observation spaces
        # Action space: n_assets * 5 actions per asset
        # [0] Market Sell, [1] Hold, [2] Market Buy, [3] Limit Sell, [4] Limit Buy
        action_space_size = n_assets * 5
        self.action_space = spaces.Discrete(action_space_size)
        self.observation_space = spaces.Dict({
            'observation': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(27, n_assets, n_assets),
                dtype=np.float32
            ),
            'action_mask': spaces.Box(
                low=0, high=1,
                shape=(action_space_size,),
                dtype=np.int8
            ),
            'to_play': spaces.Box(low=-1, high=1, shape=(1,), dtype=np.int8)
        })

        # Episode state
        self.portfolio: Optional[PortfolioState] = None
        self.current_step = 0
        # BUG_018 FIX: Per-instance RNG for reproducibility in parallel training
        self.rng = np.random.default_rng(None)  # Will be seeded in reset()
        self.episode_start_idx = 0
        self._current_prices: Dict[int, float] = {}

    def reset(self, seed=None, options=None, start_offset: int = None):
        """
        Reset environment for new episode.

        Args:
            seed: Optional seed for reproducibility
            options: Gymnasium options dict (unused)
            start_offset: If provided, use this as episode start index (TASK_064).
                          Enables smart allocation with distributed tiling.
        """
        super().reset(seed=seed)

        # Generate new synthetic data if using synthetic mode
        if self.synthetic is not None:
            self.synthetic.reset()
            self._generate_price_array()

        # Random start position (leave room for episode)
        max_start = len(self.prices) - self.max_steps - self.lookback - 10
        if max_start <= self.lookback:
            max_start = self.lookback + 1

        # BUG_018 FIX: Use per-instance RNG instead of global np.random.seed
        # This ensures reproducibility when running multiple environments in parallel
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # TASK_064: Smart allocation - use provided start_offset if given
        if start_offset is not None:
            # Clamp to valid range
            self.episode_start_idx = max(self.lookback, min(start_offset, max_start))
        else:
            self.episode_start_idx = self.rng.integers(self.lookback, max_start)
        self.current_step = 0

        # Reset portfolio
        self.portfolio = PortfolioState(cash=self.initial_cash)

        # Reset encoder and warmup
        self.encoder.buffer_idx = 0
        for i in range(self.lookback):
            step_idx = self.episode_start_idx - self.lookback + i
            self._update_encoder(step_idx)

        # Get current prices
        self._update_current_prices()

        # Track initial state for baseline calculations
        self._initial_prices = np.array([self._current_prices.get(i, 1.0) for i in range(self.n_assets)])
        self._initial_portfolio_value = self.initial_cash
        self._cumulative_agent_return = 0.0  # Track total agent return

        # TASK_051: Track previous prices for burning money calculation
        self._prev_step_prices = self._initial_prices.copy()

        return self._get_observation(), {}

    def set_fees(self, fee_bps: int, slippage_bps: int):
        """
        Dynamically update fee and slippage rates.

        Used by AdaptiveFeeController to gradually increase trading costs.

        Args:
            fee_bps: Trading fee in basis points (e.g., 16 = 0.16%)
            slippage_bps: Slippage in basis points (e.g., 5 = 0.05%)
        """
        self.fee_rate = fee_bps / 10000.0
        self.trading_fee_bps = fee_bps
        self.slippage_rate = slippage_bps / 10000.0

    def set_prices(self, prices: np.ndarray):
        """
        Dynamically update prices array (TASK_064: stratified allocation).

        Used to switch between periods without recreating env.
        Requires historical mode (synthetic=None).

        Args:
            prices: New prices array, shape (n_steps, n_assets)
        """
        if self.synthetic is not None:
            raise ValueError("Cannot set_prices in synthetic mode")
        if prices.shape[1] != self.n_assets:
            raise ValueError(f"Prices shape {prices.shape} doesn't match n_assets={self.n_assets}")
        self.prices = prices

    def _generate_price_array(self):
        """Generate price array from synthetic market"""
        n_steps = self.synthetic.n_steps
        self.prices = np.zeros((n_steps, self.n_assets), dtype=np.float32)

        # Reset and step through to collect all prices
        self.synthetic.reset()
        for i in range(n_steps):
            data = self.synthetic.get_current_data()
            for symbol, info in data.items():
                if symbol in self.encoder.symbol_map:
                    asset_idx = self.encoder.symbol_map[symbol]
                    self.prices[i, asset_idx] = info['close']
            self.synthetic.step()

        self.synthetic.reset()  # Reset for actual episode

    def _try_load_cache(self, cache_path: str) -> bool:
        """Try to load prices from numpy cache. Returns True if successful."""
        import os
        import time

        if not os.path.exists(cache_path):
            return False

        try:
            t0 = time.time()
            logging.info(f"Loading from cache: {cache_path}")
            data = np.load(cache_path, allow_pickle=True)

            self.prices = data['prices']
            self.symbols = data['symbols'].tolist()

            logging.info(f"Cache loaded in {time.time()-t0:.1f}s: "
                        f"{self.prices.shape[0]:,} steps, {len(self.symbols)} assets")
            return True
        except Exception as e:
            logging.warning(f"Failed to load cache {cache_path}: {e}")
            return False

    def _save_cache(self, cache_path: str):
        """Save prices to numpy cache for fast future loading."""
        import os
        import time

        try:
            t0 = time.time()
            logging.info(f"Saving cache to: {cache_path}")

            np.savez_compressed(
                cache_path,
                prices=self.prices,
                symbols=np.array(self.symbols)
            )

            file_size = os.path.getsize(cache_path) / (1024**2)
            logging.info(f"Cache saved ({file_size:.1f} MB) in {time.time()-t0:.1f}s")
        except Exception as e:
            logging.warning(f"Failed to save cache: {e}")

    def _load_database_prices(
        self,
        db_path: str,
        n_assets: int,
        start_date: Optional[str],
        end_date: Optional[str],
        specific_symbols: Optional[List[str]] = None,
        capture_raw_nonzero: bool = False,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Load price data from SQLite database.

        Args:
            db_path: Path to SQLite database
            n_assets: Number of assets to load (ignored if specific_symbols provided)
            start_date: Start date filter (YYYY-MM-DD or timestamp)
            end_date: End date filter (YYYY-MM-DD or timestamp)
            specific_symbols: List of specific asset symbols to load (for rainbow table training)
            capture_raw_nonzero: TASK_868 (D31 a2) — also capture the pre-fill 1-min
                real-print mask into ``self.raw_nonzero_`` (default False = canon,
                no extra memory). Only env_factory's eval path requests it.

        Returns:
            prices: (n_steps, n_assets) array of close prices
            symbols: List of asset symbols

        TASK_868 (D31 opt (a1)) — ALIAS UNION. Each requested/selected symbol resolves to
        EVERY ``assets`` row whose normalized form matches (mcts_trader/data/
        asset_aliases.py: ``/USDTEUR`` == ``USDTEUR`` etc.), and the twins' ``ohlcv``
        rows are unioned into the slot's single series — the ingest identity split is
        repaired at load time with NO DB writes. Overlapping timestamps must AGREE
        (rel <= 1e-12) or the load REFUSES (AliasDisagreementError names both asset_ids
        + the first disagreeing timestamp); agreeing duplicates dedupe. The coverage
        RANKING (no-symbols branch) is likewise grouped by normalized key, so a split
        series ranks on its whole history (D31: "the split slots' coverage ranking is
        itself distorted today"). Per-slot alias usage is disclosed on
        ``self.asset_aliases_``. On a DB whose duplicate spellings carry no rows in the
        window (e.g. the verification fixture) this is numerically bit-identical to the
        pre-868 loader.
        """
        import sqlite3
        from datetime import datetime, timezone
        import logging
        from mcts_trader.data.asset_aliases import (
            alias_agreement_gate, ranked_minute_universe, resolve_requested_symbols)

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Parse date filters
        filter_start_ts = 0
        filter_end_ts = 9999999999

        # TASK_869 (D31 (a3) scope 1): dates resolve at UTC midnight, NOT local time — the
        # window extent no longer depends on the machine's timezone (pre-fix a CEST
        # workstation and a UTC server read different windows for the same --start/--end;
        # measured 308-row-class offset on the OOS window). Same rule history.py always
        # used; inclusivity semantics unchanged (end = midnight of the end date).
        if start_date:
            if isinstance(start_date, (int, float)):
                filter_start_ts = int(start_date)
            elif str(start_date).isdigit():
                filter_start_ts = int(start_date)
            else:
                try:
                    filter_start_ts = int(datetime.strptime(str(start_date), '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp())
                except ValueError:
                    try:
                        # FIX (run-25 panel): an offset-bearing ISO input keeps its offset
                        # (astimezone to the correct instant); only NAIVE inputs are labeled UTC.
                        _dt = datetime.fromisoformat(str(start_date))
                        _dt = _dt.astimezone(timezone.utc) if _dt.tzinfo else _dt.replace(tzinfo=timezone.utc)
                        filter_start_ts = int(_dt.timestamp())
                    except:
                        logging.warning(f"Could not parse start_date: {start_date}")

        if end_date:
            if isinstance(end_date, (int, float)):
                filter_end_ts = int(end_date)
            elif str(end_date).isdigit():
                filter_end_ts = int(end_date)
            else:
                try:
                    filter_end_ts = int(datetime.strptime(str(end_date), '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp())
                except ValueError:
                    try:
                        # FIX (run-25 panel): an offset-bearing ISO input keeps its offset
                        # (astimezone to the correct instant); only NAIVE inputs are labeled UTC.
                        _dt = datetime.fromisoformat(str(end_date))
                        _dt = _dt.astimezone(timezone.utc) if _dt.tzinfo else _dt.replace(tzinfo=timezone.utc)
                        filter_end_ts = int(_dt.timestamp())
                    except:
                        logging.warning(f"Could not parse end_date: {end_date}")

        logging.info(f"Loading database: {db_path}")
        logging.info(f"Date filter: {filter_start_ts} to {filter_end_ts}")

        if specific_symbols:
            # Use specific symbols (rainbow table training mode)
            logging.info(f"Using specific symbols: {specific_symbols[:5]}... ({len(specific_symbols)} total)")
            # TASK_868 (a1): resolve each requested symbol to its whole ALIAS GROUP (every
            # assets row whose normalized form matches). An unknown spelling keeps the canon
            # warn-and-skip behaviour; a KNOWN twin group unions below.
            all_asset_rows = cursor.execute("SELECT id, symbol FROM assets").fetchall()
            resolution, unknown = resolve_requested_symbols(all_asset_rows, specific_symbols)
            for sym in unknown:
                logging.warning(f"Symbol {sym} not found in database, skipping")
            if not resolution:
                raise ValueError(f"No assets found matching symbols: {specific_symbols[:5]}...")
            # Preserve order from specific_symbols; one column per requested symbol.
            id_groups = [r["asset_ids"] for r in resolution]
            symbols = [r["symbol"] for r in resolution]
        else:
            # TASK_868 (a1): the coverage ranking is GROUPED by normalized symbol — per-id
            # counts summed across twin rows (the canon query ranked each twin separately,
            # which is the distortion D31 names). Representative symbol = the in-window
            # max-count member's DB spelling (deterministic).
            ranked = ranked_minute_universe(cursor, filter_start_ts, filter_end_ts, n_assets)
            if not ranked:
                raise ValueError(f"No data found in {db_path} for date range")
            id_groups = [g["asset_ids"] for g in ranked]
            symbols = [g["symbol"] for g in ranked]

        # TASK_868 (a1): FAIL-LOUD agreement gate, once per load — every alias pair's
        # shared-timestamp closes in the window must agree (rel <= 1e-12) or the load
        # REFUSES (AliasDisagreementError names both asset_ids + the first disagreeing
        # timestamp). Agreeing duplicate timestamps dedupe in the union below.
        gate_disclosure = alias_agreement_gate(
            conn, "ohlcv", id_groups, filter_start_ts, filter_end_ts)
        # Per-id in-window row counts (the D31 disclosure rider: per-slot asset_ids +
        # per-id row counts).
        all_ids = sorted({aid for g in id_groups for aid in g})
        ph_ids = ','.join('?' * len(all_ids))
        per_id_counts = {int(r[0]): int(r[1]) for r in cursor.execute(
            f"""SELECT asset_id, COUNT(*) FROM ohlcv
                WHERE asset_id IN ({ph_ids}) AND timestamp >= ? AND timestamp <= ?
                GROUP BY asset_id""",
            (*all_ids, filter_start_ts, filter_end_ts)).fetchall()}
        gate_by_ids = {tuple(d["asset_ids"]): d["overlaps"] for d in gate_disclosure}
        self.asset_aliases_ = [
            {"slot": col, "symbol": sym, "asset_ids": list(g),
             "per_id_counts": {str(aid): per_id_counts.get(aid, 0) for aid in g},
             "aliased": len(g) > 1,
             "overlap_agreement": gate_by_ids.get(tuple(g), [])}
            for col, (sym, g) in enumerate(zip(symbols, id_groups))]

        # Union: every member id maps to its group's column.
        asset_ids = all_ids
        id_to_col = {aid: col for col, g in enumerate(id_groups) for aid in g}

        logging.info(f"Selected {len(symbols)} assets: {symbols[:5]}...")

        # Get all timestamps in range
        placeholders = ','.join('?' * len(asset_ids))
        query = f"""
            SELECT DISTINCT timestamp
            FROM ohlcv
            WHERE asset_id IN ({placeholders})
              AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp
        """
        cursor.execute(query, (*asset_ids, filter_start_ts, filter_end_ts))
        timestamps = [row[0] for row in cursor.fetchall()]

        if len(timestamps) < self.lookback + self.max_steps:
            raise ValueError(f"Not enough data: {len(timestamps)} rows, need {self.lookback + self.max_steps}")

        logging.info(f"Loaded {len(timestamps)} timestamps")

        # Create timestamp to index mapping
        ts_to_idx = {ts: i for i, ts in enumerate(timestamps)}

        # Initialize price array
        n_steps = len(timestamps)
        prices = np.zeros((n_steps, len(symbols)), dtype=np.float32)

        # Load all OHLCV data
        query = f"""
            SELECT asset_id, timestamp, close
            FROM ohlcv
            WHERE asset_id IN ({placeholders})
              AND timestamp >= ? AND timestamp <= ?
        """
        cursor.execute(query, (*asset_ids, filter_start_ts, filter_end_ts))

        rows_loaded = 0
        for asset_id, timestamp, close in cursor.fetchall():
            if timestamp in ts_to_idx and asset_id in id_to_col and close:
                row_idx = ts_to_idx[timestamp]
                col_idx = id_to_col[asset_id]
                # Alias union: duplicate (timestamp, group) rows AGREED at the gate above,
                # so a second write lands the same value — dedupe is numerically free.
                # (A falsy close is ABSENT by every loader's convention; the zero-init
                # array already carries that, so skipping it is numerically identical.)
                prices[row_idx, col_idx] = close
                rows_loaded += 1

        logging.info(f"Loaded {rows_loaded:,} price points into ({n_steps}, {len(symbols)}) array")

        # TASK_868 (D31 a2): the pre-fill 1-min real-print mask, captured ONLY on request
        # (env_factory's eval open-gate path). This is the causal evidence for the
        # frozen-cell force-HOLD mask — a cell is gated iff the slot printed nothing
        # inside its bar, which is knowable from data <= the decision bar alone.
        if capture_raw_nonzero:
            self.raw_nonzero_ = (prices != 0).copy()

        # Forward fill any missing prices
        for col in range(prices.shape[1]):
            mask = prices[:, col] == 0
            if mask.any():
                # Find first non-zero
                first_valid = np.argmax(~mask)
                if first_valid > 0:
                    prices[:first_valid, col] = prices[first_valid, col]
                # Forward fill
                for i in range(1, n_steps):
                    if prices[i, col] == 0:
                        prices[i, col] = prices[i-1, col]

        conn.close()
        return prices, symbols

    def _load_from_schedule(
        self,
        schedule_path: str,
        n_assets: int
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Load price data from a training schedule (random period/run selection).

        The schedule is generated by tools/map_training_data.py and contains
        valid training periods with pre-selected asset combinations.

        Returns:
            prices: (n_steps, n_assets) array of close prices
            symbols: List of asset symbols
        """
        import json
        import logging

        # Load schedule
        with open(schedule_path) as f:
            schedule = json.load(f)

        self.schedule_data = schedule

        # Randomly select a period
        periods = schedule['periods']
        period = np.random.choice(periods)

        # Randomly select a training run from this period
        if period['training_runs']:
            run = np.random.choice(period['training_runs'])
            symbols = run['assets'][:n_assets]  # Limit to n_assets
        else:
            # Fallback: use period's available assets
            symbols = period['available_assets'][:n_assets]

        logging.info(f"Schedule: Selected period {period['start_date'][:10]} to {period['end_date'][:10]}")
        logging.info(f"  Run with {len(symbols)} assets (overlap: {run.get('overlap_with_previous', 0):.0%})")

        # Load prices for these specific assets from database
        start_ts = period['start']
        end_ts = period['end']

        return self._load_database_prices_for_symbols(
            self.db_path, symbols, start_ts, end_ts
        )

    def _load_database_prices_for_symbols(
        self,
        db_path: str,
        symbols: List[str],
        start_ts: int,
        end_ts: int
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Load prices for specific symbols from database.

        Args:
            db_path: Database path
            symbols: List of asset symbols to load
            start_ts: Start timestamp
            end_ts: End timestamp

        Returns:
            prices: (n_steps, n_assets) array
            symbols: List of loaded symbols
        """
        import sqlite3
        import logging
        from mcts_trader.data.asset_aliases import (
            alias_agreement_gate, resolve_requested_symbols)

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # TASK_868 (a1): the SAME alias union as _load_database_prices — each schedule
        # symbol resolves to every assets row whose normalized form matches, the gate
        # refuses disagreeing overlaps, and the twins' rows union into one column.
        id_rows = cursor.execute("SELECT id, symbol FROM assets").fetchall()
        resolution, unknown = resolve_requested_symbols(id_rows, symbols)
        if unknown:
            logging.warning(f"Only found {len(resolution)}/{len(symbols)} assets in database")
        if not resolution:
            raise ValueError("No valid assets found in database for schedule")

        valid_symbols = [r["symbol"] for r in resolution]
        id_groups = [r["asset_ids"] for r in resolution]
        alias_agreement_gate(conn, "ohlcv", id_groups, start_ts, end_ts)
        asset_ids = sorted({aid for g in id_groups for aid in g})
        id_to_col = {aid: col for col, g in enumerate(id_groups) for aid in g}
        self.asset_aliases_ = [
            {"slot": col, "symbol": sym, "asset_ids": list(g), "aliased": len(g) > 1}
            for col, (sym, g) in enumerate(zip(valid_symbols, id_groups))]

        # Get timestamps in range
        id_placeholders = ','.join('?' * len(asset_ids))
        cursor.execute(f"""
            SELECT DISTINCT timestamp
            FROM ohlcv
            WHERE asset_id IN ({id_placeholders})
              AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp
        """, (*asset_ids, start_ts, end_ts))
        timestamps = [row[0] for row in cursor.fetchall()]

        if len(timestamps) < self.lookback + self.max_steps:
            raise ValueError(f"Not enough data: {len(timestamps)} rows, need {self.lookback + self.max_steps}")

        logging.info(f"Loaded {len(timestamps)} timestamps for {len(valid_symbols)} assets")

        # Create timestamp to index mapping
        ts_to_idx = {ts: i for i, ts in enumerate(timestamps)}

        # Initialize price array
        n_steps = len(timestamps)
        prices = np.zeros((n_steps, len(valid_symbols)), dtype=np.float32)

        # Load all OHLCV data
        cursor.execute(f"""
            SELECT asset_id, timestamp, close
            FROM ohlcv
            WHERE asset_id IN ({id_placeholders})
              AND timestamp >= ? AND timestamp <= ?
        """, (*asset_ids, start_ts, end_ts))

        for asset_id, timestamp, close in cursor.fetchall():
            if timestamp in ts_to_idx and asset_id in id_to_col and close:
                row_idx = ts_to_idx[timestamp]
                col_idx = id_to_col[asset_id]
                prices[row_idx, col_idx] = close

        # Forward fill any missing prices
        for col in range(prices.shape[1]):
            mask = prices[:, col] == 0
            if mask.any():
                first_valid = np.argmax(~mask)
                if first_valid > 0:
                    prices[:first_valid, col] = prices[first_valid, col]
                for i in range(1, n_steps):
                    if prices[i, col] == 0:
                        prices[i, col] = prices[i-1, col]

        conn.close()
        return prices, valid_symbols

    def _update_encoder(self, step_idx: int):
        """Feed price data to encoder"""
        market_data = {}
        for asset_idx in range(self.n_assets):
            symbol = self.symbols[asset_idx]
            price = self.prices[step_idx, asset_idx]
            market_data[symbol] = {
                'open': price,
                'high': price * 1.001,  # Approximate
                'low': price * 0.999,
                'close': price,
                'volume': 1000.0,  # Placeholder
            }
        self.encoder.update(market_data)

    def _update_current_prices(self):
        """Update current price dict"""
        step_idx = self.episode_start_idx + self.current_step
        self._current_prices = {
            i: self.prices[step_idx, i]
            for i in range(self.n_assets)
        }

    def _process_limit_orders(self) -> Dict[str, List[int]]:
        """
        Check and process pending limit orders each timestep.

        Orders can:
        1. Fill - if price condition is met
        2. Expire - if TTL exceeded (returns reserved cash)

        Returns:
            Dict with lists of asset indices for 'filled' and 'expired' orders
        """
        result = {'filled': [], 'expired': []}

        if not self.limit_orders_enabled:
            return result

        for asset_idx in list(self.portfolio.pending_orders.keys()):
            order = self.portfolio.pending_orders[asset_idx]
            price = self._current_prices.get(asset_idx, 0)

            if price <= 0:
                continue

            # Check expiration first (TTL exceeded)
            if order.is_expired(self.current_step):
                # LAZY EXECUTION: No cash to refund
                del self.portfolio.pending_orders[asset_idx]
                result['expired'].append(asset_idx)
                continue

            # Check if order can be filled at current price
            if order.can_fill(price):
                # FIX: Execute at limit price, not market close price.
                # Using 'price' (Close) creates "Magic Fill" bias where agent
                # buys at bottom of crash (e.g., Limit $98, Close $90 -> Fill $90).
                # Reality: Fills at $98 as price crosses through it.
                # This is standard conservative assumption for backtesting without tick data.
                exec_price = order.limit_price
                filled = self._fill_limit_order(asset_idx, order, exec_price)
                del self.portfolio.pending_orders[asset_idx]
                if filled:
                    result['filled'].append(asset_idx)
                else:
                    # LAZY EXECUTION: Order cancelled due to insufficient funds
                    result['expired'].append(asset_idx)  # Reuse expired list for cancellations

        return result

    def _fill_limit_order(self, asset_idx: int, order: LimitOrder, fill_price: float) -> bool:
        """
        Fill a limit order at the given price with maker fee.

        LAZY EXECUTION: Cash is checked at fill time, not at order placement.
        If insufficient cash, order is cancelled (returns False).

        Args:
            asset_idx: Asset index
            order: The limit order to fill
            fill_price: Price at which to fill

        Returns:
            True if order was filled, False if cancelled due to insufficient funds
        """
        if order.side == 'buy':
            # Create long position (or close short)
            if asset_idx in self.portfolio.positions:
                pos = self.portfolio.positions[asset_idx]
                if pos.side == 'short':
                    # Close short position (always allowed - we're closing, not opening)
                    proceeds = pos.size * (2 * pos.entry_price - fill_price)
                    fee = abs(proceeds) * self.maker_fee_rate
                    self.portfolio.cash += proceeds - fee
                    del self.portfolio.positions[asset_idx]
                    return True
                else:
                    # Already long, order is stale/invalid
                    return False
            else:
                # LAZY EXECUTION: Check if we have enough cash NOW
                shares = order.size
                cost = shares * fill_price
                fee = cost * self.maker_fee_rate
                total_cost = cost + fee

                if total_cost > self.portfolio.cash:
                    # Insufficient funds - cancel order (no fill)
                    return False

                # Open long position with maker fee
                self.portfolio.cash -= total_cost

                stop_loss = fill_price * (1 - self.default_stop_loss_pct) if self.tp_sl_enabled else 0.0
                take_profit = fill_price * (1 + self.default_take_profit_pct) if self.tp_sl_enabled else 0.0

                self.portfolio.positions[asset_idx] = Position(
                    asset_idx=asset_idx,
                    side='long',
                    size=shares,
                    entry_price=fill_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )
                return True

        else:  # sell
            # Create short position (or close long)
            if asset_idx in self.portfolio.positions:
                pos = self.portfolio.positions[asset_idx]
                if pos.side == 'long':
                    # Close long position (always allowed - we're closing)
                    proceeds = pos.size * fill_price
                    fee = proceeds * self.maker_fee_rate
                    self.portfolio.cash += proceeds - fee
                    del self.portfolio.positions[asset_idx]
                    return True
                else:
                    # Already short, order is stale/invalid
                    return False
            elif self.allow_short:
                # LAZY EXECUTION: Check if we have enough cash NOW for margin
                shares = order.size
                order_value = shares * fill_price
                fee = order_value * self.maker_fee_rate
                total_cost = order_value + fee

                if total_cost > self.portfolio.cash:
                    # Insufficient funds - cancel order (no fill)
                    return False

                stop_loss = fill_price * (1 + self.default_stop_loss_pct) if self.tp_sl_enabled else 0.0
                take_profit = fill_price * (1 - self.default_take_profit_pct) if self.tp_sl_enabled else 0.0

                self.portfolio.cash -= total_cost
                self.portfolio.positions[asset_idx] = Position(
                    asset_idx=asset_idx,
                    side='short',
                    size=shares,
                    entry_price=fill_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )
                return True
            else:
                # Shorting not allowed, order invalid
                return False

    def _check_guardrails(self) -> Dict[str, List[int]]:
        """
        Check TP/SL for all positions, close if triggered.

        Returns:
            Dict with lists of asset indices closed by 'stop_loss' or 'take_profit'
        """
        closed = {'stop_loss': [], 'take_profit': []}

        if not self.tp_sl_enabled:
            return closed

        for asset_idx in list(self.portfolio.positions.keys()):
            pos = self.portfolio.positions[asset_idx]
            price = self._current_prices.get(asset_idx, pos.entry_price)

            triggered = None

            if pos.side == 'long':
                # Stop loss: price drops below threshold
                if pos.stop_loss > 0 and price <= pos.stop_loss:
                    triggered = 'stop_loss'
                # Take profit: price rises above threshold
                elif pos.take_profit > 0 and price >= pos.take_profit:
                    triggered = 'take_profit'

            elif pos.side == 'short':
                # Stop loss: price rises above threshold
                if pos.stop_loss > 0 and price >= pos.stop_loss:
                    triggered = 'stop_loss'
                # Take profit: price drops below threshold
                elif pos.take_profit > 0 and price <= pos.take_profit:
                    triggered = 'take_profit'

            if triggered:
                # Close position
                self._close_position(asset_idx)
                closed[triggered].append(asset_idx)

        return closed

    def _close_position(self, asset_idx: int) -> float:
        """
        Close a position at current market price.

        Returns:
            Net proceeds from closing the position
        """
        if asset_idx not in self.portfolio.positions:
            return 0.0

        pos = self.portfolio.positions[asset_idx]
        price = self._current_prices.get(asset_idx, pos.entry_price)

        if pos.side == 'long':
            # Sell at lower price (slippage)
            exec_price = price * (1 - self.slippage_rate)
            proceeds = pos.size * exec_price
        else:  # short
            # Buy to cover at higher price (slippage)
            exec_price = price * (1 + self.slippage_rate)
            # Short PnL: we sold high (entry), now buy to cover
            proceeds = pos.size * (2 * pos.entry_price - exec_price)

        fee = abs(proceeds) * self.fee_rate
        net_proceeds = proceeds - fee

        self.portfolio.cash += net_proceeds
        del self.portfolio.positions[asset_idx]

        return net_proceeds

    def step(self, action) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute action(s) and advance environment by one timestep.

        Multi-Action Mode (action_points > 1):
            - Accept list of actions: [a1, a2, a3, a4]
            - All actions execute sequentially BEFORE market time advances
            - Portfolio state updates between actions, but prices stay frozen
            - Enables "asset jumping" (sell A, buy B in same timestep)

        Single-Action Mode (action_points == 1):
            - Accept single int action (backward compatible)

        Args:
            action: int or List[int] - action(s) to execute

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Normalize action input
        if isinstance(action, (list, tuple, np.ndarray)):
            actions = list(action)[:self.action_points]  # Limit to action_points
        else:
            # Single action - repeat if multi-action mode
            actions = [action] * self.action_points

        # 1. Process pending limit orders FIRST (fill or expire)
        limit_orders_result = self._process_limit_orders()

        # 2. Check TP/SL guardrails (before agent actions)
        guardrails_triggered = self._check_guardrails()

        # Get previous portfolio value (before any actions)
        prev_value = self.portfolio.total_value(self._current_prices)

        # 3. Execute all actions sequentially (market time frozen)
        actions_executed = []
        for action_idx, act in enumerate(actions):
            # Decode action
            asset_idx = act // 5
            action_type = act % 5

            # Execute action: [0] Market Sell, [1] Hold, [2] Market Buy, [3] Limit Sell, [4] Limit Buy
            if action_type == 0:  # Market Sell
                self._execute_sell(asset_idx)
            elif action_type == 2:  # Market Buy
                self._execute_buy(asset_idx)
            elif action_type == 3:  # Limit Sell
                self._execute_limit_sell(asset_idx)
            elif action_type == 4:  # Limit Buy
                self._execute_limit_buy(asset_idx)
            # action_type == 1 is HOLD, do nothing

            actions_executed.append({
                'action_idx': action_idx,
                'asset': asset_idx,
                'type': action_type,
                'portfolio_value': self.portfolio.total_value(self._current_prices)
            })

        # 4. Advance market time (once per timestep, after all actions)
        self.current_step += 1
        step_idx = self.episode_start_idx + self.current_step

        # Check termination
        terminated = step_idx >= len(self.prices) - 1
        truncated = self.current_step >= self.max_steps

        if not terminated:
            self._update_encoder(step_idx)
            self._update_current_prices()

        # FIX Bug #5: Apply borrow cost for short positions (deducts from cash)
        # This aligns with JAX env and real-world shorting costs
        if self.borrow_rate_annual > 0 and not terminated:
            # 100,800 steps/year (252 trading days * 400 steps) - matches JAX env
            rate_per_step = self.borrow_rate_annual / 100800.0
            total_short_value = sum(
                pos.size * self._current_prices.get(pos.asset_idx, pos.entry_price)
                for pos in self.portfolio.positions.values()
                if pos.side == 'short'
            )
            borrow_cost = total_short_value * rate_per_step
            self.portfolio.cash -= borrow_cost

        # Calculate reward (percentage return for entire action sequence)
        new_value = self.portfolio.total_value(self._current_prices)

        # =================================================================
        # HARD LIQUIDATION CHECK (TASK_ENV_HARD_LIQUIDATION)
        # =================================================================
        # Terminate early if portfolio value falls below threshold.
        # This prevents "death spiral" scenarios:
        # - Negative cash trap (fees push cash negative)
        # - Ghost shares (negative share calculations)
        # - Inverted PnL (negative shares act like opposite position)
        # - NaN explosion (divide by near-zero values)
        liquidation_triggered = False
        liquidation_threshold_value = self.initial_cash * LIQUIDATION_THRESHOLD

        if new_value < liquidation_threshold_value:
            liquidation_triggered = True
            terminated = True  # Force early termination
            # Apply large penalty (raw value, will be symlog'd below)
            pct_return = LIQUIDATION_PENALTY_RAW
            # Log warning (visible in training output)
            print(f"  ⚠️ LIQUIDATION: Portfolio ${new_value:.2f} < threshold ${liquidation_threshold_value:.2f} "
                  f"(step {self.current_step})")
        else:
            # FIX Bug #6: Use Log Returns for geometric consistency
            # Arithmetic returns are biased: +50% then -33% (net zero) yields positive reward sum
            # Log returns are symmetric: log(1.5) + log(0.66) ≈ 0
            # We then apply symlog for range compression (MuZero value support)
            if prev_value > 1e-8 and new_value > 1e-8:
                log_return = np.log(new_value / prev_value) * 100.0
            else:
                # Fallback for edge cases (near-bankruptcy)
                log_return = (new_value - prev_value) / (abs(prev_value) + 1e-8) * 100.0
            pct_return = log_return

        # Use symlog transformation (now on log-returns)
        # This preserves geometric symmetry while compressing large moves
        reward = symlog(pct_return, threshold=1.0)

        # TASK_051: Apply burning money (cash opportunity cost)
        if self.burn_config.enabled:
            current_prices_burn = np.array([self._current_prices.get(i, 1.0) for i in range(self.n_assets)])
            burn = compute_burn_numpy(
                cash=self.portfolio.cash,
                portfolio_value=new_value,
                market_prices=current_prices_burn,
                prev_prices=self._prev_step_prices,
                config=self.burn_config,
                rng=self.rng,
            )
            reward -= burn  # Subtract burn from reward

        # Track cumulative agent return from portfolio value (correct compounding)
        # OLD (WRONG): self._cumulative_agent_return += pct_return  # Additive is incorrect!
        # NEW (CORRECT): Compute from portfolio value like JAX env does
        self._cumulative_agent_return = (new_value / self._initial_portfolio_value - 1.0) * 100.0

        # Calculate market baseline (buy-and-hold return)
        current_prices = np.array([self._current_prices.get(i, 1.0) for i in range(self.n_assets)])
        # Avoid division by zero
        valid_mask = self._initial_prices > 0
        if valid_mask.sum() > 0:
            # Mean return across all assets (equal-weight buy-and-hold)
            price_returns = (current_prices[valid_mask] - self._initial_prices[valid_mask]) / self._initial_prices[valid_mask] * 100
            market_return = float(np.mean(price_returns))
        else:
            market_return = 0.0

        # Alpha = agent return - market return (positive = beating market)
        alpha = self._cumulative_agent_return - market_return

        # Info for debugging
        info = {
            'portfolio_value': new_value,
            'pct_return': pct_return,  # Raw percentage return (before symlog)
            'cash': self.portfolio.cash,
            'n_positions': len(self.portfolio.positions),
            'n_pending_orders': len(self.portfolio.pending_orders),
            'step': self.current_step,
            'guardrails_triggered': guardrails_triggered,  # TP/SL closures
            'limit_orders': limit_orders_result,  # Filled/expired orders
            'actions_executed': actions_executed,  # Multi-action trace
            'liquidation_triggered': liquidation_triggered,  # Hard liquidation flag
            # Market baseline metrics
            'market_return': market_return,  # Buy-and-hold return (%)
            'agent_return': self._cumulative_agent_return,  # Cumulative agent return (%)
            'alpha': alpha,  # Agent - Market (positive = outperforming)
        }

        # Always update prev prices for burn tracking (allows hot config enable)
        self._prev_step_prices = current_prices.copy()

        return self._get_observation(), reward, terminated, truncated, info

    def _execute_buy(self, asset_idx: int):
        """Execute BUY order (close short or open long)"""
        # FIX: Cancel pending limit orders to prevent "Ghost Fills"
        # Market order = "I want this NOW", implies old conditional orders are stale
        # LAZY EXECUTION: No cash to refund since we don't reserve upfront
        if asset_idx in self.portfolio.pending_orders:
            del self.portfolio.pending_orders[asset_idx]

        if asset_idx in self.portfolio.positions:
            pos = self.portfolio.positions[asset_idx]

            if pos.side == 'short':
                # Close short position (buy to cover)
                price = self._current_prices.get(asset_idx, pos.entry_price)
                exec_price = price * (1 + self.slippage_rate)  # Buy at higher price

                # Short PnL: we sold high, now buy low (or vice versa)
                # Return collateral + PnL: entry_value + (entry_price - exec_price) * size
                proceeds = pos.size * (2 * pos.entry_price - exec_price)
                fee = abs(proceeds) * self.fee_rate
                net_proceeds = proceeds - fee

                self.portfolio.cash += net_proceeds
                del self.portfolio.positions[asset_idx]
            else:
                # Already long - do nothing
                pass
            return

        # Skip if already at max positions
        if len(self.portfolio.positions) >= self.max_positions:
            return

        # Cash floor guardrail: prevent buying when effectively bankrupt
        # (TASK_ENV_HARD_LIQUIDATION)
        if self.portfolio.cash < MIN_CASH_THRESHOLD:
            return

        # Get price with slippage (buy at higher price)
        price = self._current_prices.get(asset_idx, 0)
        if price <= 0:
            return

        exec_price = price * (1 + self.slippage_rate)

        # Calculate order size
        portfolio_value = self.portfolio.total_value(self._current_prices)
        order_value = portfolio_value * self.position_size

        # Apply fee
        fee = order_value * self.fee_rate
        net_value = order_value - fee

        # Check if we have enough cash
        if net_value > self.portfolio.cash:
            net_value = self.portfolio.cash * 0.99  # Leave some buffer
            fee = net_value * self.fee_rate / (1 - self.fee_rate)
            order_value = net_value + fee

        # Calculate shares
        shares = net_value / exec_price
        if shares <= 0:
            return

        # Calculate TP/SL prices for long position
        stop_loss = exec_price * (1 - self.default_stop_loss_pct) if self.tp_sl_enabled else 0.0
        take_profit = exec_price * (1 + self.default_take_profit_pct) if self.tp_sl_enabled else 0.0

        # Update portfolio
        self.portfolio.cash -= order_value
        self.portfolio.positions[asset_idx] = Position(
            asset_idx=asset_idx,
            side='long',
            size=shares,
            entry_price=exec_price,
            stop_loss=stop_loss,
            take_profit=take_profit
        )

    def _execute_sell(self, asset_idx: int):
        """Execute SELL order (close long or open short)"""
        # FIX: Cancel pending limit orders to prevent "Ghost Fills"
        # Market order = "I want this NOW", implies old conditional orders are stale
        # LAZY EXECUTION: No cash to refund since we don't reserve upfront
        if asset_idx in self.portfolio.pending_orders:
            del self.portfolio.pending_orders[asset_idx]

        if asset_idx in self.portfolio.positions:
            pos = self.portfolio.positions[asset_idx]

            if pos.side == 'long':
                # Close long position
                price = self._current_prices.get(asset_idx, pos.entry_price)
                exec_price = price * (1 - self.slippage_rate)  # Sell at lower price

                proceeds = pos.size * exec_price
                fee = proceeds * self.fee_rate
                net_proceeds = proceeds - fee

                self.portfolio.cash += net_proceeds
                del self.portfolio.positions[asset_idx]
            else:
                # Already short - do nothing (could add to position later)
                pass
        elif self.allow_short:
            # Open short position
            if len(self.portfolio.positions) >= self.max_positions:
                return

            price = self._current_prices.get(asset_idx, 0)
            if price <= 0:
                return

            # Short at lower price (slippage works against us)
            exec_price = price * (1 - self.slippage_rate)

            # Calculate order size
            portfolio_value = self.portfolio.total_value(self._current_prices)
            order_value = portfolio_value * self.position_size

            # Apply fee upfront
            fee = order_value * self.fee_rate
            net_value = order_value - fee

            if net_value > self.portfolio.cash:
                net_value = self.portfolio.cash * 0.99
                fee = net_value * self.fee_rate / (1 - self.fee_rate)
                order_value = net_value + fee

            # Calculate shares to short
            shares = net_value / exec_price
            if shares <= 0:
                return

            # Calculate TP/SL prices for short position (inverted from long)
            # Stop loss is ABOVE entry (price rising is bad for short)
            # Take profit is BELOW entry (price falling is good for short)
            stop_loss = exec_price * (1 + self.default_stop_loss_pct) if self.tp_sl_enabled else 0.0
            take_profit = exec_price * (1 - self.default_take_profit_pct) if self.tp_sl_enabled else 0.0

            # Reserve cash as collateral (we receive proceeds from short sale)
            # But need collateral for margin - simplify: just deduct order_value
            self.portfolio.cash -= order_value
            self.portfolio.positions[asset_idx] = Position(
                asset_idx=asset_idx,
                side='short',
                size=shares,
                entry_price=exec_price,
                stop_loss=stop_loss,
                take_profit=take_profit
            )

    def _execute_limit_buy(self, asset_idx: int, limit_price: float = None, ttl: int = None):
        """
        Place a limit buy order that persists until filled or TTL expires.

        Unlike IOC (Immediate-or-Cancel), this order sits in pending_orders
        and waits for the price to drop to the limit. Benefits:
        - Better execution price (waits for dip)
        - Maker fee instead of taker fee
        - Express model's price prediction with time horizon

        Args:
            asset_idx: Asset to buy
            limit_price: Target price (from model prediction, or default offset)
            ttl: Time-to-live in steps (from model prediction, or default)
        """
        # Cancel any existing order for this asset
        # LAZY EXECUTION: No cash to refund since we don't reserve upfront
        if asset_idx in self.portfolio.pending_orders:
            del self.portfolio.pending_orders[asset_idx]

        # Skip if already have position or at max combined positions+orders
        if asset_idx in self.portfolio.positions:
            return
        total_orders = len(self.portfolio.positions) + len(self.portfolio.pending_orders)
        if total_orders >= self.max_positions:
            return

        # Cash floor guardrail: prevent limit buying when effectively bankrupt
        # (TASK_ENV_HARD_LIQUIDATION)
        if self.portfolio.cash < MIN_CASH_THRESHOLD:
            return

        price = self._current_prices.get(asset_idx, 0)
        if price <= 0:
            return

        # Use provided limit price or calculate default
        if limit_price is None:
            limit_price = price * (1 - self.limit_offset_pct)

        # Use provided TTL or default
        if ttl is None:
            ttl = self.default_limit_order_ttl

        # Calculate order size (for reference at fill time)
        portfolio_value = self.portfolio.total_value(self._current_prices)
        order_value = portfolio_value * self.position_size

        # LAZY EXECUTION: Don't reserve cash, just record intended order size
        # Cash availability will be checked at fill time
        estimated_fee = order_value * self.maker_fee_rate
        total_cost = order_value + estimated_fee

        # Calculate shares at limit price
        shares = (total_cost - total_cost * self.maker_fee_rate) / limit_price
        if shares <= 0:
            return

        # Create order WITHOUT reserving cash (lazy execution)
        self.portfolio.pending_orders[asset_idx] = LimitOrder(
            asset_idx=asset_idx,
            side='buy',
            size=shares,
            limit_price=limit_price,
            reserved_cash=0.0,  # LAZY: No cash reserved upfront
            created_step=self.current_step,
            ttl=ttl
        )

    def _execute_limit_sell(self, asset_idx: int, limit_price: float = None, ttl: int = None):
        """
        Place a limit sell order that persists until filled or TTL expires.

        Unlike IOC, this order waits for price to rise to the limit. Benefits:
        - Better execution price (waits for spike)
        - Maker fee instead of taker fee
        - Express model's price prediction with time horizon

        Args:
            asset_idx: Asset to sell
            limit_price: Target price (from model prediction, or default offset)
            ttl: Time-to-live in steps (from model prediction, or default)
        """
        # Cancel any existing order for this asset
        # LAZY EXECUTION: No cash to refund since we don't reserve upfront
        if asset_idx in self.portfolio.pending_orders:
            del self.portfolio.pending_orders[asset_idx]

        price = self._current_prices.get(asset_idx, 0)
        if price <= 0:
            return

        # Use provided limit price or calculate default
        if limit_price is None:
            limit_price = price * (1 + self.limit_offset_pct)

        # Use provided TTL or default
        if ttl is None:
            ttl = self.default_limit_order_ttl

        # If we have a long position, create sell order to close it
        if asset_idx in self.portfolio.positions:
            pos = self.portfolio.positions[asset_idx]
            if pos.side == 'long':
                # Create limit sell for existing position
                self.portfolio.pending_orders[asset_idx] = LimitOrder(
                    asset_idx=asset_idx,
                    side='sell',
                    size=pos.size,
                    limit_price=limit_price,
                    reserved_cash=0.0,  # No cash reserved for selling existing position
                    created_step=self.current_step,
                    ttl=ttl
                )
            # If already short, do nothing
            return

        # If no position and short allowed, create order to open short
        if not self.allow_short:
            return

        total_orders = len(self.portfolio.positions) + len(self.portfolio.pending_orders)
        if total_orders >= self.max_positions:
            return

        # Calculate order size for short
        portfolio_value = self.portfolio.total_value(self._current_prices)
        order_value = portfolio_value * self.position_size

        # Check if we have enough cash for margin
        estimated_fee = order_value * self.maker_fee_rate
        total_reserve = order_value + estimated_fee

        if total_reserve > self.portfolio.cash:
            return

        shares = order_value / limit_price
        if shares <= 0:
            return

        # For short sells, we don't reserve cash upfront
        # (Short sale generates cash, margin handled on fill)
        self.portfolio.pending_orders[asset_idx] = LimitOrder(
            asset_idx=asset_idx,
            side='sell',
            size=shares,
            limit_price=limit_price,
            reserved_cash=0.0,
            created_step=self.current_step,
            ttl=ttl
        )

    def _get_observation(self) -> Dict:
        """Get observation dict for MuZero"""
        obs = self.encoder.encode()  # Returns np.ndarray directly

        # Build action mask
        mask = self._get_action_mask()

        return {
            'observation': obs.astype(np.float32),
            'action_mask': mask,
            'to_play': np.array([-1], dtype=np.int8)
        }

    def _get_action_mask(self) -> np.ndarray:
        """
        Generate valid action mask for 5-action space.

        Actions per asset:
        - [0] Market Sell: close long or open short
        - [1] Hold: always valid
        - [2] Market Buy: open long or close short
        - [3] Limit Sell: close long or open short (queued)
        - [4] Limit Buy: open long or close short (queued)
        """
        mask = np.zeros(self.n_assets * 5, dtype=np.int8)
        total_orders = len(self.portfolio.positions) + len(self.portfolio.pending_orders)

        for asset_idx in range(self.n_assets):
            base_idx = asset_idx * 5
            price = self._current_prices.get(asset_idx, 0)
            has_position = asset_idx in self.portfolio.positions
            has_pending = asset_idx in self.portfolio.pending_orders

            # [0] Market Sell - close long or open short
            has_long = has_position and self.portfolio.positions[asset_idx].side == 'long'
            can_open_short = (
                self.allow_short and
                total_orders < self.max_positions and
                price > 0 and
                not has_position
            )
            if has_long or can_open_short:
                mask[base_idx + 0] = 1

            # [1] Hold - always valid
            mask[base_idx + 1] = 1

            # [2] Market Buy - open long or close short
            has_short = has_position and self.portfolio.positions[asset_idx].side == 'short'
            can_open_long = (
                price > 0 and
                total_orders < self.max_positions and
                not has_position and
                self.portfolio.cash > 100
            )
            if can_open_long or has_short:
                mask[base_idx + 2] = 1

            # [3] Limit Sell - close long or open short (queued)
            if has_long or can_open_short:
                mask[base_idx + 3] = 1

            # [4] Limit Buy - open long or close short (queued, no existing pending)
            if (can_open_long or has_short) and not has_pending:
                mask[base_idx + 4] = 1

        return mask

    def get_portfolio_stats(self) -> Dict:
        """Get detailed portfolio statistics for debugging"""
        value = self.portfolio.total_value(self._current_prices)
        pnl = self.portfolio.total_pnl(self._current_prices)

        positions_info = []
        for idx, pos in self.portfolio.positions.items():
            current_price = self._current_prices.get(idx, pos.entry_price)
            positions_info.append({
                'asset': idx,
                'symbol': self.symbols[idx],
                'side': pos.side,
                'size': pos.size,
                'entry': pos.entry_price,
                'current': current_price,
                'pnl': pos.pnl(current_price),
                'value': pos.value(current_price),
            })

        return {
            'total_value': value,
            'cash': self.portfolio.cash,
            'unrealized_pnl': pnl,
            'return_pct': (value / self.initial_cash - 1) * 100,
            'n_positions': len(self.portfolio.positions),
            'positions': positions_info,
        }


# Utility function to validate against VectorBT
def validate_against_vectorbt(env: VectorBTTradingEnv, n_steps: int = 100):
    """
    Run identical trades in both VectorBTTradingEnv and actual VectorBT
    to verify accounting matches.
    """
    try:
        import vectorbt as vbt
        import pandas as pd
    except ImportError:
        print("VectorBT not installed, skipping validation")
        return None

    # Reset environment
    obs, _ = env.reset()

    # Record all trades
    trades = []
    values_env = [env.portfolio.total_value(env._current_prices)]

    np.random.seed(42)
    for step in range(n_steps):
        # Random action (biased towards HOLD)
        action = np.random.choice([0, 1, 1, 1, 2])  # More holds
        asset = np.random.randint(0, env.n_assets)
        full_action = asset * 3 + action

        # Execute in env
        obs, reward, done, truncated, info = env.step(full_action)
        values_env.append(info['portfolio_value'])

        if done or truncated:
            break

    print(f"Environment final value: ${values_env[-1]:,.2f}")
    print(f"Environment return: {(values_env[-1]/values_env[0]-1)*100:.2f}%")

    return values_env


if __name__ == "__main__":
    # Test with synthetic data
    print("Testing VectorBTTradingEnv with synthetic data...")

    env = VectorBTTradingEnv(
        synthetic_level=0,  # Simple sine waves
        n_assets=10,
        initial_cash=10000,
        max_steps=100,
    )

    obs, info = env.reset()
    print(f"Observation shape: {obs['observation'].shape}")
    print(f"Action mask shape: {obs['action_mask'].shape}")
    print(f"Initial portfolio: ${env.portfolio.cash:,.2f}")

    # Run a few steps
    total_reward = 0
    for i in range(50):
        # Random action
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

        if i % 10 == 0:
            print(f"Step {i}: reward={reward:.4f}%, positions={info['n_positions']}, value=${info['portfolio_value']:,.2f}")

        if done or truncated:
            break

    print(f"\nFinal stats:")
    stats = env.get_portfolio_stats()
    print(f"  Total value: ${stats['total_value']:,.2f}")
    print(f"  Return: {stats['return_pct']:.2f}%")
    print(f"  Sum of rewards: {total_reward:.2f}%")
    print(f"  Positions: {stats['n_positions']}")
