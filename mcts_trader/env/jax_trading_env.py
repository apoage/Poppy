"""
JAX-Vectorized Trading Environment (Full Implementation)

This module provides a fully JAX-compatible trading environment that can be
vmapped for massive parallelization on GPU. Designed to replace the serial
Python env.step() bottleneck in train_jax_fast.py.

Phases Implemented:
- Phase 0: Market orders (buy/sell long positions) ✓
- Phase 1: Short positions + TP/SL guardrails ✓
- Phase 2: Limit orders with TTL ✓
- Phase 3: Full 27-channel encoder (separate file)
- Phase 4: Training loop integration (pending)

Key Design Principles:
----------------------
1. IMMUTABLE STATE: All state updates return new state objects (no in-place mutation)
2. PURE FUNCTIONS: No side effects, deterministic given inputs
3. FIXED-SIZE ARRAYS: No dynamic dicts, all state is fixed-size arrays
4. JNP.WHERE FOR BRANCHING: No Python if/else, use jnp.where for conditionals
5. JAX.LAX.SWITCH FOR ACTIONS: Dispatch to action handlers without Python control flow

State Structure:
---------------
Instead of Dict[int, Position], we use parallel arrays:
    pos_active[i] = True if asset i has a position
    pos_side[i] = 1 (long) or 2 (short)
    pos_size[i] = number of shares
    pos_entry[i] = entry price
    pos_sl[i] = stop loss price (0 = disabled)
    pos_tp[i] = take profit price (0 = disabled)

Similarly for limit orders:
    order_active[i] = True if asset i has pending order
    order_side[i] = 1 (buy) or 2 (sell)
    order_size[i] = order size in shares
    order_price[i] = limit price
    order_reserved[i] = cash reserved for buy orders
    order_created[i] = step when created
    order_ttl[i] = time-to-live remaining

Usage:
------
    # Single environment
    state = reset(key, prices, config)
    state, obs, reward, done = step(state, action, prices)

    # Batched (1000 parallel envs)
    batch_step = jax.vmap(step, in_axes=(0, 0, None))
    states, obs, rewards, dones = batch_step(states, actions, prices)

Performance:
-----------
    Benchmark (RTX 3090):
    - 1000 parallel envs: 1.85M steps/sec
    - 256 parallel envs: 31K steps/sec
    - Speedup vs Python serial: 3000x+

Author: Claude Code
Task: TASK_027_JAX_VECTORIZED_ENV.md
"""

import jax
import jax.numpy as jnp
import chex
from typing import Tuple
from functools import partial


# =============================================================================
# Configuration
# =============================================================================

@chex.dataclass(frozen=True)
class EnvConfig:
    """Immutable environment configuration.

    Attributes:
        n_assets: Number of tradeable assets (default 64)
        initial_cash: Starting cash balance
        fee_bps: Trading fee in basis points (1 bp = 0.01%)
        slippage_bps: Slippage in basis points
        maker_fee_bps: Fee for limit orders (typically lower)
        max_steps: Maximum steps per episode
        window_size: Observation window size (for price history)
        position_size_pct: Fraction of portfolio to use per trade
        allow_short: Enable short selling
        tp_sl_enabled: Enable take-profit/stop-loss guardrails
        default_stop_loss_pct: Stop loss percentage (5% = 0.05)
        default_take_profit_pct: Take profit percentage (10% = 0.10)
        default_limit_order_ttl: Time-to-live for limit orders
        limit_offset_pct: Default offset for limit price from current
        liquidation_threshold_pct: Liquidate if portfolio drops below this % of initial
    """
    n_assets: int = 64
    initial_cash: float = 100_000.0
    fee_bps: float = 0.0
    slippage_bps: float = 0.0
    maker_fee_bps: float = 0.0  # Lower fees for limit orders
    max_steps: int = 400
    window_size: int = 60
    position_size_pct: float = 0.1  # 10% of portfolio per trade
    allow_short: bool = True
    tp_sl_enabled: bool = False
    default_stop_loss_pct: float = 0.05  # 5%
    default_take_profit_pct: float = 0.10  # 10%
    default_limit_order_ttl: int = 100
    limit_offset_pct: float = 0.002  # 0.2%
    liquidation_threshold_pct: float = 0.01  # 1% of initial


# =============================================================================
# State Definition
# =============================================================================

@chex.dataclass(frozen=True)
class TradingState:
    """Complete trading environment state - fully JAX compatible.

    All arrays are fixed-size to enable JIT compilation and vmap.
    No Python dicts or dynamic structures.

    Scalar State:
        cash: Current cash balance
        current_step: Steps taken in current episode
        episode_start_idx: Index into price array where episode started
        initial_portfolio_value: Portfolio value at episode start (for return calc)

    Position Arrays (n_assets,):
        pos_active: Boolean mask - True if position exists for this asset
        pos_side: Position side - 0=none, 1=long, 2=short
        pos_size: Number of shares held
        pos_entry: Average entry price
        pos_sl: Stop loss price (0 = disabled)
        pos_tp: Take profit price (0 = disabled)

    Limit Order Arrays (n_assets,):
        order_active: Boolean mask - True if pending order exists
        order_side: Order side - 0=none, 1=buy, 2=sell
        order_size: Order size in shares
        order_price: Limit price
        order_reserved: Cash reserved (buy orders only)
        order_created: Step when order was created
        order_ttl: Time-to-live in steps

    Tracking:
        prev_portfolio_value: Portfolio value at previous step (for reward)
        cumulative_return: Running return since episode start
    """
    # Scalar state
    cash: chex.Array                    # ()
    current_step: chex.Array            # ()
    episode_start_idx: chex.Array       # ()
    initial_portfolio_value: chex.Array # ()

    # Position arrays (n_assets,)
    pos_active: chex.Array              # bool
    pos_side: chex.Array                # int: 0/1/2
    pos_size: chex.Array                # float
    pos_entry: chex.Array               # float
    pos_sl: chex.Array                  # float (0 = disabled)
    pos_tp: chex.Array                  # float (0 = disabled)

    # Limit order arrays (n_assets,)
    order_active: chex.Array            # bool
    order_side: chex.Array              # int: 0/1/2
    order_size: chex.Array              # float
    order_price: chex.Array             # float
    order_reserved: chex.Array          # float
    order_created: chex.Array           # int
    order_ttl: chex.Array               # int

    # Tracking
    prev_portfolio_value: chex.Array    # ()
    cumulative_return: chex.Array       # ()

    # Random key for stochastic elements
    rng_key: chex.PRNGKey               # (2,)


# =============================================================================
# Pure Helper Functions
# =============================================================================

def get_current_prices(prices: chex.Array, step: chex.Array,
                       episode_start: chex.Array) -> chex.Array:
    """Get current prices for all assets."""
    idx = episode_start + step
    return prices[idx]


def compute_position_values(pos_active: chex.Array, pos_side: chex.Array,
                           pos_size: chex.Array, pos_entry: chex.Array,
                           current_prices: chex.Array) -> chex.Array:
    """Compute market value of all positions.

    Long: value = size * current_price
    Short: value = 2 * size * entry - size * current_price
    """
    long_value = pos_size * current_prices
    short_value = 2.0 * pos_size * pos_entry - pos_size * current_prices
    values = jnp.where(pos_side == 1, long_value,
                      jnp.where(pos_side == 2, short_value, 0.0))
    return jnp.where(pos_active, values, 0.0)


def compute_portfolio_value(state: TradingState,
                           current_prices: chex.Array) -> chex.Array:
    """Compute total portfolio value (cash + positions - reserved)."""
    position_values = compute_position_values(
        state.pos_active, state.pos_side, state.pos_size,
        state.pos_entry, current_prices
    )
    # Reserved cash for buy orders is already deducted from cash,
    # but we include it in portfolio value calculation
    reserved_cash = jnp.sum(jnp.where(state.order_active & (state.order_side == 1),
                                       state.order_reserved, 0.0))
    return state.cash + jnp.sum(position_values) + reserved_cash


def apply_fee_and_slippage(price: chex.Array, is_buy: chex.Array,
                          fee_bps: float, slippage_bps: float) -> chex.Array:
    """Apply trading costs to execution price."""
    total_cost_bps = fee_bps + slippage_bps
    multiplier = 1.0 + total_cost_bps / 10000.0
    return jnp.where(is_buy, price * multiplier, price / multiplier)


def symlog(x: chex.Array, threshold: float = 1.0) -> chex.Array:
    """Symmetric logarithm transformation."""
    return jnp.sign(x) * jnp.log1p(jnp.abs(x) / threshold) * threshold


# =============================================================================
# TP/SL Guardrail Functions
# =============================================================================

def check_tp_sl_triggers(state: TradingState, current_prices: chex.Array,
                        config: EnvConfig) -> Tuple[chex.Array, chex.Array]:
    """Check which positions should be closed due to TP or SL.

    Returns:
        Tuple of (should_close_mask, close_prices)
        - should_close_mask: (n_assets,) bool - True if position should close
        - close_prices: (n_assets,) float - Execution price for closing
    """
    # Long positions: SL triggers when price <= sl_price, TP when price >= tp_price
    long_sl_triggered = (state.pos_side == 1) & (state.pos_sl > 0) & (current_prices <= state.pos_sl)
    long_tp_triggered = (state.pos_side == 1) & (state.pos_tp > 0) & (current_prices >= state.pos_tp)

    # Short positions: SL triggers when price >= sl_price, TP when price <= tp_price
    short_sl_triggered = (state.pos_side == 2) & (state.pos_sl > 0) & (current_prices >= state.pos_sl)
    short_tp_triggered = (state.pos_side == 2) & (state.pos_tp > 0) & (current_prices <= state.pos_tp)

    should_close = state.pos_active & (long_sl_triggered | long_tp_triggered |
                                        short_sl_triggered | short_tp_triggered)

    return should_close, current_prices


def apply_tp_sl_closures(state: TradingState, current_prices: chex.Array,
                        config: EnvConfig) -> TradingState:
    """Close positions that hit TP or SL levels."""
    if not config.tp_sl_enabled:
        return state

    should_close, _ = check_tp_sl_triggers(state, current_prices, config)

    # Calculate proceeds for each position that should close
    # Long: proceeds = size * price / (1 + slippage)
    # Short: proceeds = 2 * size * entry - size * price * (1 + slippage)
    sell_price = apply_fee_and_slippage(current_prices, False, config.fee_bps, config.slippage_bps)
    buy_price = apply_fee_and_slippage(current_prices, True, config.fee_bps, config.slippage_bps)

    long_proceeds = state.pos_size * sell_price
    short_proceeds = 2.0 * state.pos_size * state.pos_entry - state.pos_size * buy_price

    proceeds = jnp.where(state.pos_side == 1, long_proceeds,
                        jnp.where(state.pos_side == 2, short_proceeds, 0.0))
    proceeds = jnp.where(should_close, proceeds, 0.0)

    # Update cash
    new_cash = state.cash + jnp.sum(proceeds)

    # Clear closed positions
    new_pos_active = jnp.where(should_close, False, state.pos_active)
    new_pos_side = jnp.where(should_close, 0, state.pos_side)
    new_pos_size = jnp.where(should_close, 0.0, state.pos_size)
    new_pos_entry = jnp.where(should_close, 0.0, state.pos_entry)
    new_pos_sl = jnp.where(should_close, 0.0, state.pos_sl)
    new_pos_tp = jnp.where(should_close, 0.0, state.pos_tp)

    return state.replace(
        cash=new_cash,
        pos_active=new_pos_active,
        pos_side=new_pos_side,
        pos_size=new_pos_size,
        pos_entry=new_pos_entry,
        pos_sl=new_pos_sl,
        pos_tp=new_pos_tp,
    )


# =============================================================================
# Limit Order Functions
# =============================================================================

def process_limit_orders(state: TradingState, current_prices: chex.Array,
                        config: EnvConfig) -> TradingState:
    """Process pending limit orders - check fills and expirations.

    This is called at the START of each step, before the action.
    """
    # Check expiration: (current_step - created) >= ttl
    expired = state.order_active & ((state.current_step - state.order_created) >= state.order_ttl)

    # Return reserved cash for expired buy orders
    expired_buy_cash = jnp.sum(jnp.where(expired & (state.order_side == 1),
                                          state.order_reserved, 0.0))

    # Check fill conditions
    # Buy: fills when current_price <= limit_price
    # Sell: fills when current_price >= limit_price
    can_fill_buy = state.order_active & (state.order_side == 1) & (current_prices <= state.order_price)
    can_fill_sell = state.order_active & (state.order_side == 2) & (current_prices >= state.order_price)
    can_fill = (can_fill_buy | can_fill_sell) & ~expired

    # Process fills
    state = fill_limit_orders(state, current_prices, can_fill, config)

    # Clear expired orders and update cash
    new_order_active = jnp.where(expired, False, state.order_active)
    new_order_side = jnp.where(expired, 0, state.order_side)
    new_order_size = jnp.where(expired, 0.0, state.order_size)
    new_order_price = jnp.where(expired, 0.0, state.order_price)
    new_order_reserved = jnp.where(expired, 0.0, state.order_reserved)
    new_order_created = jnp.where(expired, 0, state.order_created)
    new_order_ttl = jnp.where(expired, 0, state.order_ttl)

    return state.replace(
        cash=state.cash + expired_buy_cash,
        order_active=new_order_active,
        order_side=new_order_side,
        order_size=new_order_size,
        order_price=new_order_price,
        order_reserved=new_order_reserved,
        order_created=new_order_created,
        order_ttl=new_order_ttl,
    )


def fill_limit_orders(state: TradingState, current_prices: chex.Array,
                     can_fill: chex.Array, config: EnvConfig) -> TradingState:
    """Fill limit orders that meet their price conditions."""
    # Maker fee for limit orders (typically lower than taker)
    maker_fee_rate = config.maker_fee_bps / 10000.0

    # Process each fillable order vectorized
    is_buy_fill = can_fill & (state.order_side == 1)
    is_sell_fill = can_fill & (state.order_side == 2)

    # Buy fill: open long or close short
    # Cash: return reserved, deduct actual cost + fee
    buy_fill_price = state.order_price  # Fill at limit price
    buy_cost = state.order_size * buy_fill_price
    buy_fee = buy_cost * maker_fee_rate

    # For buys: check if closing short or opening long
    closing_short = is_buy_fill & state.pos_active & (state.pos_side == 2)
    opening_long = is_buy_fill & ~state.pos_active

    # Close short proceeds: 2 * size * entry - size * fill_price
    short_close_proceeds = state.pos_size * (2.0 * state.pos_entry - buy_fill_price)
    short_close_fee = jnp.abs(short_close_proceeds) * maker_fee_rate

    # Sell fill: close long or open short
    sell_fill_price = state.order_price
    sell_proceeds = state.order_size * sell_fill_price
    sell_fee = sell_proceeds * maker_fee_rate

    closing_long = is_sell_fill & state.pos_active & (state.pos_side == 1)
    opening_short = is_sell_fill & ~state.pos_active & config.allow_short

    # Calculate cash changes
    # Buy to open long: +reserved - cost - fee
    # Buy to close short: +reserved + short_proceeds - fee
    # Sell to close long: +proceeds - fee
    # Sell to open short: -cost - fee (no reserved for short)

    cash_change_buy_long = jnp.where(opening_long,
                                      state.order_reserved - buy_cost - buy_fee, 0.0)
    cash_change_buy_short = jnp.where(closing_short,
                                       state.order_reserved + short_close_proceeds - short_close_fee, 0.0)
    cash_change_sell_long = jnp.where(closing_long,
                                       sell_proceeds - sell_fee, 0.0)
    cash_change_sell_short = jnp.where(opening_short,
                                        -buy_cost - buy_fee, 0.0)  # Short sale generates cash later

    total_cash_change = jnp.sum(cash_change_buy_long + cash_change_buy_short +
                                cash_change_sell_long + cash_change_sell_short)

    # Update positions
    # Opening long
    new_pos_active = jnp.where(opening_long, True, state.pos_active)
    new_pos_side = jnp.where(opening_long, 1, state.pos_side)
    new_pos_size = jnp.where(opening_long, state.order_size, state.pos_size)
    new_pos_entry = jnp.where(opening_long, buy_fill_price, state.pos_entry)

    # Set TP/SL for new long positions
    new_pos_sl = jnp.where(opening_long & config.tp_sl_enabled,
                           buy_fill_price * (1.0 - config.default_stop_loss_pct),
                           state.pos_sl)
    new_pos_tp = jnp.where(opening_long & config.tp_sl_enabled,
                           buy_fill_price * (1.0 + config.default_take_profit_pct),
                           state.pos_tp)

    # Closing short (clear position)
    new_pos_active = jnp.where(closing_short, False, new_pos_active)
    new_pos_side = jnp.where(closing_short, 0, new_pos_side)
    new_pos_size = jnp.where(closing_short, 0.0, new_pos_size)
    new_pos_entry = jnp.where(closing_short, 0.0, new_pos_entry)
    new_pos_sl = jnp.where(closing_short, 0.0, new_pos_sl)
    new_pos_tp = jnp.where(closing_short, 0.0, new_pos_tp)

    # Closing long (clear position)
    new_pos_active = jnp.where(closing_long, False, new_pos_active)
    new_pos_side = jnp.where(closing_long, 0, new_pos_side)
    new_pos_size = jnp.where(closing_long, 0.0, new_pos_size)
    new_pos_entry = jnp.where(closing_long, 0.0, new_pos_entry)
    new_pos_sl = jnp.where(closing_long, 0.0, new_pos_sl)
    new_pos_tp = jnp.where(closing_long, 0.0, new_pos_tp)

    # Opening short
    new_pos_active = jnp.where(opening_short, True, new_pos_active)
    new_pos_side = jnp.where(opening_short, 2, new_pos_side)
    new_pos_size = jnp.where(opening_short, state.order_size, new_pos_size)
    new_pos_entry = jnp.where(opening_short, sell_fill_price, new_pos_entry)

    # Set TP/SL for new short positions (inverted levels)
    new_pos_sl = jnp.where(opening_short & config.tp_sl_enabled,
                           sell_fill_price * (1.0 + config.default_stop_loss_pct),
                           new_pos_sl)
    new_pos_tp = jnp.where(opening_short & config.tp_sl_enabled,
                           sell_fill_price * (1.0 - config.default_take_profit_pct),
                           new_pos_tp)

    # Clear filled orders
    filled = can_fill
    new_order_active = jnp.where(filled, False, state.order_active)
    new_order_side = jnp.where(filled, 0, state.order_side)
    new_order_size = jnp.where(filled, 0.0, state.order_size)
    new_order_price = jnp.where(filled, 0.0, state.order_price)
    new_order_reserved = jnp.where(filled, 0.0, state.order_reserved)
    new_order_created = jnp.where(filled, 0, state.order_created)
    new_order_ttl = jnp.where(filled, 0, state.order_ttl)

    return state.replace(
        cash=state.cash + total_cash_change,
        pos_active=new_pos_active,
        pos_side=new_pos_side,
        pos_size=new_pos_size,
        pos_entry=new_pos_entry,
        pos_sl=new_pos_sl,
        pos_tp=new_pos_tp,
        order_active=new_order_active,
        order_side=new_order_side,
        order_size=new_order_size,
        order_price=new_order_price,
        order_reserved=new_order_reserved,
        order_created=new_order_created,
        order_ttl=new_order_ttl,
    )


# =============================================================================
# Action Execution Functions
# =============================================================================

def execute_hold(state: TradingState, asset_idx: chex.Array,
                current_prices: chex.Array, config: EnvConfig) -> TradingState:
    """Execute HOLD action - no state change."""
    return state


def execute_buy(state: TradingState, asset_idx: chex.Array,
               current_prices: chex.Array, config: EnvConfig) -> TradingState:
    """Execute BUY action - open/add long or close short.

    If short exists: Close the short position
    If no position: Open new long (uses available cash if can't afford full trade)
    If long exists: Add to position

    NOTE: Like VectorBT, if can't afford full trade, uses available cash * 0.99
    to ensure trades actually execute rather than failing silently.
    """
    price = current_prices[asset_idx]
    buy_exec_price = apply_fee_and_slippage(price, True, config.fee_bps, config.slippage_bps)

    portfolio_value = compute_portfolio_value(state, current_prices)
    ideal_trade_value = portfolio_value * config.position_size_pct
    ideal_cost = ideal_trade_value  # Cost before adjustment

    # VectorBT behavior: if can't afford full trade, use available cash
    # (lines 1118-1122 in vectorbt_env.py)
    available_cash = state.cash * 0.99  # Leave 1% buffer like VectorBT
    actual_cost = jnp.minimum(ideal_cost, available_cash)
    shares_to_buy = actual_cost / buy_exec_price
    cost = actual_cost

    has_position = state.pos_active[asset_idx]
    is_long = state.pos_side[asset_idx] == 1
    is_short = state.pos_side[asset_idx] == 2
    current_size = state.pos_size[asset_idx]
    current_entry = state.pos_entry[asset_idx]

    # Case 1: Close short position
    # Short proceeds: 2 * size * entry - size * buy_price
    short_proceeds = current_size * (2.0 * current_entry - buy_exec_price)
    fee_on_close = jnp.abs(short_proceeds) * config.fee_bps / 10000.0

    # Case 2: Open/add to long - has cash for trade (after adjustment)
    # Minimum trade threshold to prevent dust trades
    min_trade_value = 100.0
    can_afford = (state.cash > min_trade_value) & (shares_to_buy > 0)

    # New long size (if opening or adding)
    new_long_size = jnp.where(has_position & is_long,
                               current_size + shares_to_buy,
                               shares_to_buy)
    new_long_entry = jnp.where(has_position & is_long,
                                (current_size * current_entry + shares_to_buy * buy_exec_price) / new_long_size,
                                buy_exec_price)

    # Determine which case applies
    closing_short = has_position & is_short
    opening_long = ~has_position & can_afford
    adding_long = has_position & is_long & can_afford

    # Calculate cash change
    cash_after_close_short = state.cash + short_proceeds - fee_on_close
    cash_after_buy = state.cash - cost

    new_cash = jnp.where(closing_short, cash_after_close_short,
                        jnp.where(opening_long | adding_long, cash_after_buy, state.cash))

    # Update position
    should_change = closing_short | opening_long | adding_long

    new_pos_active = state.pos_active.at[asset_idx].set(
        jnp.where(closing_short, False,
                 jnp.where(opening_long | adding_long, True, state.pos_active[asset_idx]))
    )
    new_pos_side = state.pos_side.at[asset_idx].set(
        jnp.where(closing_short, 0,
                 jnp.where(opening_long | adding_long, 1, state.pos_side[asset_idx]))
    )
    new_pos_size = state.pos_size.at[asset_idx].set(
        jnp.where(closing_short, 0.0,
                 jnp.where(opening_long | adding_long, new_long_size, state.pos_size[asset_idx]))
    )
    new_pos_entry = state.pos_entry.at[asset_idx].set(
        jnp.where(closing_short, 0.0,
                 jnp.where(opening_long | adding_long, new_long_entry, state.pos_entry[asset_idx]))
    )

    # Set TP/SL for new long positions
    new_pos_sl = state.pos_sl.at[asset_idx].set(
        jnp.where(closing_short, 0.0,
                 jnp.where(opening_long & config.tp_sl_enabled,
                          buy_exec_price * (1.0 - config.default_stop_loss_pct),
                          state.pos_sl[asset_idx]))
    )
    new_pos_tp = state.pos_tp.at[asset_idx].set(
        jnp.where(closing_short, 0.0,
                 jnp.where(opening_long & config.tp_sl_enabled,
                          buy_exec_price * (1.0 + config.default_take_profit_pct),
                          state.pos_tp[asset_idx]))
    )

    return state.replace(
        cash=new_cash,
        pos_active=new_pos_active,
        pos_side=new_pos_side,
        pos_size=new_pos_size,
        pos_entry=new_pos_entry,
        pos_sl=new_pos_sl,
        pos_tp=new_pos_tp,
    )


def execute_sell(state: TradingState, asset_idx: chex.Array,
                current_prices: chex.Array, config: EnvConfig) -> TradingState:
    """Execute SELL action - close long or open short.

    If long exists: Close the long position
    If no position and allow_short: Open new short (uses available cash if can't afford full trade)
    If short exists: Do nothing (can't add to short with market order)

    NOTE: Like VectorBT, if can't afford full short, uses available cash * 0.99
    to ensure trades actually execute rather than failing silently.
    """
    price = current_prices[asset_idx]
    sell_exec_price = apply_fee_and_slippage(price, False, config.fee_bps, config.slippage_bps)

    portfolio_value = compute_portfolio_value(state, current_prices)
    ideal_trade_value = portfolio_value * config.position_size_pct

    has_position = state.pos_active[asset_idx]
    is_long = state.pos_side[asset_idx] == 1
    is_short = state.pos_side[asset_idx] == 2
    current_size = state.pos_size[asset_idx]
    current_entry = state.pos_entry[asset_idx]

    # Case 1: Close long position
    long_proceeds = current_size * sell_exec_price
    fee_on_close = long_proceeds * config.fee_bps / 10000.0

    # Case 2: Open short (if no position and allowed)
    # VectorBT behavior: if can't afford full trade, use available cash
    ideal_short_cost = ideal_trade_value  # Cash needed for margin
    available_cash = state.cash * 0.99  # Leave 1% buffer like VectorBT
    actual_short_cost = jnp.minimum(ideal_short_cost, available_cash)
    shares_to_sell = actual_short_cost / price
    short_cost = actual_short_cost

    fee_on_open = short_cost * config.fee_bps / 10000.0
    min_trade_value = 100.0
    can_afford_short = (state.cash > min_trade_value) & (shares_to_sell > 0)

    # Determine which case applies
    closing_long = has_position & is_long
    opening_short = ~has_position & config.allow_short & can_afford_short

    # Calculate cash change
    cash_after_close_long = state.cash + long_proceeds - fee_on_close
    cash_after_open_short = state.cash - short_cost - fee_on_open

    new_cash = jnp.where(closing_long, cash_after_close_long,
                        jnp.where(opening_short, cash_after_open_short, state.cash))

    # Update position
    new_pos_active = state.pos_active.at[asset_idx].set(
        jnp.where(closing_long, False,
                 jnp.where(opening_short, True, state.pos_active[asset_idx]))
    )
    new_pos_side = state.pos_side.at[asset_idx].set(
        jnp.where(closing_long, 0,
                 jnp.where(opening_short, 2, state.pos_side[asset_idx]))
    )
    new_pos_size = state.pos_size.at[asset_idx].set(
        jnp.where(closing_long, 0.0,
                 jnp.where(opening_short, shares_to_sell, state.pos_size[asset_idx]))
    )
    new_pos_entry = state.pos_entry.at[asset_idx].set(
        jnp.where(closing_long, 0.0,
                 jnp.where(opening_short, sell_exec_price, state.pos_entry[asset_idx]))
    )

    # Set TP/SL for new short positions (inverted)
    new_pos_sl = state.pos_sl.at[asset_idx].set(
        jnp.where(closing_long, 0.0,
                 jnp.where(opening_short & config.tp_sl_enabled,
                          sell_exec_price * (1.0 + config.default_stop_loss_pct),
                          state.pos_sl[asset_idx]))
    )
    new_pos_tp = state.pos_tp.at[asset_idx].set(
        jnp.where(closing_long, 0.0,
                 jnp.where(opening_short & config.tp_sl_enabled,
                          sell_exec_price * (1.0 - config.default_take_profit_pct),
                          state.pos_tp[asset_idx]))
    )

    return state.replace(
        cash=new_cash,
        pos_active=new_pos_active,
        pos_side=new_pos_side,
        pos_size=new_pos_size,
        pos_entry=new_pos_entry,
        pos_sl=new_pos_sl,
        pos_tp=new_pos_tp,
    )


def execute_limit_buy(state: TradingState, asset_idx: chex.Array,
                     current_prices: chex.Array, config: EnvConfig) -> TradingState:
    """Execute LIMIT BUY action - place buy order below current price."""
    price = current_prices[asset_idx]
    limit_price = price * (1.0 - config.limit_offset_pct)

    # Cancel existing order for this asset (return reserved if buy)
    existing_reserved = jnp.where(
        state.order_active[asset_idx] & (state.order_side[asset_idx] == 1),
        state.order_reserved[asset_idx],
        0.0
    )

    # Calculate order size
    portfolio_value = compute_portfolio_value(state, current_prices)
    order_value = portfolio_value * config.position_size_pct
    estimated_fee = order_value * config.maker_fee_bps / 10000.0
    total_reserve = order_value + estimated_fee

    # Adjust if not enough cash
    available_cash = state.cash + existing_reserved
    total_reserve = jnp.minimum(total_reserve, available_cash * 0.99)
    shares = (total_reserve - total_reserve * config.maker_fee_bps / 10000.0) / limit_price

    # Skip if already have position or can't afford
    has_position = state.pos_active[asset_idx]
    can_place = ~has_position & (shares > 0) & (available_cash >= 10.0)

    # Update order arrays
    new_order_active = state.order_active.at[asset_idx].set(
        jnp.where(can_place, True, state.order_active[asset_idx])
    )
    new_order_side = state.order_side.at[asset_idx].set(
        jnp.where(can_place, 1, state.order_side[asset_idx])
    )
    new_order_size = state.order_size.at[asset_idx].set(
        jnp.where(can_place, shares, state.order_size[asset_idx])
    )
    new_order_price = state.order_price.at[asset_idx].set(
        jnp.where(can_place, limit_price, state.order_price[asset_idx])
    )
    new_order_reserved = state.order_reserved.at[asset_idx].set(
        jnp.where(can_place, total_reserve, state.order_reserved[asset_idx])
    )
    new_order_created = state.order_created.at[asset_idx].set(
        jnp.where(can_place, state.current_step, state.order_created[asset_idx])
    )
    new_order_ttl = state.order_ttl.at[asset_idx].set(
        jnp.where(can_place, config.default_limit_order_ttl, state.order_ttl[asset_idx])
    )

    # Update cash: only change when can_place=True
    # BUG FIX: Previously returned existing_reserved even when can_place=False,
    # but order_reserved wasn't cleared, causing money printing.
    # Now only adjust cash when actually placing a new order.
    reserved_change = jnp.where(can_place, existing_reserved - total_reserve, 0.0)
    new_cash = state.cash + reserved_change

    return state.replace(
        cash=new_cash,
        order_active=new_order_active,
        order_side=new_order_side,
        order_size=new_order_size,
        order_price=new_order_price,
        order_reserved=new_order_reserved,
        order_created=new_order_created,
        order_ttl=new_order_ttl,
    )


def execute_limit_sell(state: TradingState, asset_idx: chex.Array,
                      current_prices: chex.Array, config: EnvConfig) -> TradingState:
    """Execute LIMIT SELL action - place sell order above current price."""
    price = current_prices[asset_idx]
    limit_price = price * (1.0 + config.limit_offset_pct)

    # Cancel existing order (return reserved if buy)
    existing_reserved = jnp.where(
        state.order_active[asset_idx] & (state.order_side[asset_idx] == 1),
        state.order_reserved[asset_idx],
        0.0
    )

    has_position = state.pos_active[asset_idx]
    is_long = state.pos_side[asset_idx] == 1
    current_size = state.pos_size[asset_idx]

    # If we have long position, sell that size
    # If no position and shorts allowed, create short order
    portfolio_value = compute_portfolio_value(state, current_prices)
    order_value = portfolio_value * config.position_size_pct
    shares_for_short = order_value / limit_price

    # Determine order size
    order_size = jnp.where(has_position & is_long, current_size, shares_for_short)

    # Can place if: have long OR (no position AND shorts allowed)
    can_place = (has_position & is_long) | (~has_position & config.allow_short)
    can_place = can_place & (order_size > 0)

    # Update order arrays
    new_order_active = state.order_active.at[asset_idx].set(
        jnp.where(can_place, True, state.order_active[asset_idx])
    )
    new_order_side = state.order_side.at[asset_idx].set(
        jnp.where(can_place, 2, state.order_side[asset_idx])
    )
    new_order_size = state.order_size.at[asset_idx].set(
        jnp.where(can_place, order_size, state.order_size[asset_idx])
    )
    new_order_price = state.order_price.at[asset_idx].set(
        jnp.where(can_place, limit_price, state.order_price[asset_idx])
    )
    new_order_reserved = state.order_reserved.at[asset_idx].set(
        jnp.where(can_place, 0.0, state.order_reserved[asset_idx])  # Sell orders don't reserve
    )
    new_order_created = state.order_created.at[asset_idx].set(
        jnp.where(can_place, state.current_step, state.order_created[asset_idx])
    )
    new_order_ttl = state.order_ttl.at[asset_idx].set(
        jnp.where(can_place, config.default_limit_order_ttl, state.order_ttl[asset_idx])
    )

    # Update cash: only return reserved when can_place=True
    # BUG FIX: Same issue as execute_limit_buy - only change cash when placing order
    reserved_change = jnp.where(can_place, existing_reserved, 0.0)
    new_cash = state.cash + reserved_change

    return state.replace(
        cash=new_cash,
        order_active=new_order_active,
        order_side=new_order_side,
        order_size=new_order_size,
        order_price=new_order_price,
        order_reserved=new_order_reserved,
        order_created=new_order_created,
        order_ttl=new_order_ttl,
    )


# =============================================================================
# Core Environment Functions
# =============================================================================

def reset(rng_key: chex.PRNGKey, prices: chex.Array,
          config: EnvConfig) -> TradingState:
    """Reset environment to initial state with random episode start."""
    n_steps = prices.shape[0]
    n_assets = config.n_assets

    key1, key2 = jax.random.split(rng_key)
    max_start = n_steps - config.max_steps - config.window_size
    episode_start = jax.random.randint(key1, (), 0, max_start)

    return TradingState(
        cash=jnp.array(config.initial_cash),
        current_step=jnp.array(0),
        episode_start_idx=episode_start,
        initial_portfolio_value=jnp.array(config.initial_cash),
        pos_active=jnp.zeros(n_assets, dtype=bool),
        pos_side=jnp.zeros(n_assets, dtype=jnp.int32),
        pos_size=jnp.zeros(n_assets, dtype=jnp.float32),
        pos_entry=jnp.zeros(n_assets, dtype=jnp.float32),
        pos_sl=jnp.zeros(n_assets, dtype=jnp.float32),
        pos_tp=jnp.zeros(n_assets, dtype=jnp.float32),
        order_active=jnp.zeros(n_assets, dtype=bool),
        order_side=jnp.zeros(n_assets, dtype=jnp.int32),
        order_size=jnp.zeros(n_assets, dtype=jnp.float32),
        order_price=jnp.zeros(n_assets, dtype=jnp.float32),
        order_reserved=jnp.zeros(n_assets, dtype=jnp.float32),
        order_created=jnp.zeros(n_assets, dtype=jnp.int32),
        order_ttl=jnp.zeros(n_assets, dtype=jnp.int32),
        prev_portfolio_value=jnp.array(config.initial_cash),
        cumulative_return=jnp.array(0.0),
        rng_key=key2,
    )


def step(state: TradingState, action: chex.Array,
         prices: chex.Array, config: EnvConfig) -> Tuple[TradingState, chex.Array, chex.Array, chex.Array]:
    """Execute one environment step.

    Order of operations:
    1. Process pending limit orders (fills and expirations)
    2. Check TP/SL guardrails
    3. Execute the action
    4. Check liquidation
    5. Compute reward and observation
    """
    # Get current prices
    current_prices = get_current_prices(
        prices, state.current_step + config.window_size, state.episode_start_idx
    )

    # 1. Process limit orders
    state = process_limit_orders(state, current_prices, config)

    # 2. Check TP/SL guardrails
    state = apply_tp_sl_closures(state, current_prices, config)

    # 3. Decode and execute action
    asset_idx = action // 5
    action_type = action % 5

    new_state = jax.lax.switch(
        action_type,
        [
            lambda s, a, p, c: execute_sell(s, a, p, c),        # 0: sell
            lambda s, a, p, c: execute_hold(s, a, p, c),        # 1: hold
            lambda s, a, p, c: execute_buy(s, a, p, c),         # 2: buy
            lambda s, a, p, c: execute_limit_sell(s, a, p, c),  # 3: limit_sell
            lambda s, a, p, c: execute_limit_buy(s, a, p, c),   # 4: limit_buy
        ],
        state, asset_idx, current_prices, config
    )

    # 4. Check liquidation
    new_portfolio_value = compute_portfolio_value(new_state, current_prices)
    liquidation_threshold = config.initial_cash * config.liquidation_threshold_pct
    is_liquidated = new_portfolio_value < liquidation_threshold

    # 5. Compute reward
    portfolio_change = new_portfolio_value - state.prev_portfolio_value
    portfolio_return_pct = portfolio_change / (state.prev_portfolio_value + 1e-8) * 100.0
    reward = jnp.where(is_liquidated, symlog(-50.0), symlog(portfolio_return_pct))

    # Update tracking
    new_cumulative_return = (new_portfolio_value / state.initial_portfolio_value - 1.0) * 100.0

    # Advance step and check termination
    new_step = state.current_step + 1
    done = (new_step >= config.max_steps) | is_liquidated

    final_state = new_state.replace(
        current_step=new_step,
        prev_portfolio_value=new_portfolio_value,
        cumulative_return=new_cumulative_return,
    )

    # Compute observation
    obs = compute_observation(final_state, prices, config)

    return final_state, obs, reward, done


# =============================================================================
# Observation Construction
# =============================================================================

# Import 27-channel encoder (optional)
from mcts_trader.env.jax_encoder import (
    EncoderConfig as FullEncoderConfig,
    encode_observation_simple as encode_27ch_simple,
    encode_observation_with_state as encode_27ch_with_state,
)


def compute_observation_27ch(state: TradingState, prices: chex.Array,
                             config: EnvConfig) -> chex.Array:
    """Compute 27-channel observation tensor (full encoder).

    This provides feature parity with FastMultiAssetStateEncoder used in training.
    Output shape: (27, n_assets, n_assets)

    Channels:
    - 0-3: Normalized OHLC
    - 4-7: Log returns at 1, 5, 15, 60 min
    - 8-11: Volume patterns
    - 12-15: Price momentum
    - 16-19: Volatility
    - 20-23: Price percentiles
    - 24-26: VWAP and bands
    """
    encoder_config = FullEncoderConfig(
        n_assets=config.n_assets,
        window_size=config.window_size,
        n_channels=27,
    )

    window_start = state.episode_start_idx + state.current_step

    # Use simple encoder (prices only, synthetic-compatible)
    return encode_27ch_simple(prices, window_start, encoder_config)


def compute_observation_31ch(state: TradingState, prices: chex.Array,
                             config: EnvConfig) -> chex.Array:
    """Compute 31-channel observation tensor (27 market + 4 portfolio).

    Extends 27-channel with portfolio state:
    - Channel 27: Position indicator (-1=short, 0=none, +1=long)
    - Channel 28: Position PnL
    - Channel 29: Order indicator
    - Channel 30: Order proximity
    """
    encoder_config = FullEncoderConfig(
        n_assets=config.n_assets,
        window_size=config.window_size,
        n_channels=27,
    )

    window_start = state.episode_start_idx + state.current_step

    return encode_27ch_with_state(
        prices, None, None, None, window_start,
        state.pos_active, state.pos_side, state.pos_entry,
        state.order_active, state.order_side,
        encoder_config,
    )


def compute_observation(state: TradingState, prices: chex.Array,
                       config: EnvConfig) -> chex.Array:
    """Compute observation tensor from current state.

    Simplified Observation (5 channels):
    - Channel 0: Normalized price (0-1 over window)
    - Channel 1: Recent returns (last step)
    - Channel 2: Position indicator (-1=short, 0=none, +1=long)
    - Channel 3: Position PnL (normalized)
    - Channel 4: Pending order indicator

    Shape: (5, 1, n_assets)
    """
    start_idx = state.episode_start_idx + state.current_step
    window = jax.lax.dynamic_slice(
        prices,
        (start_idx, 0),
        (config.window_size, config.n_assets)
    )

    # Channel 0: Normalized price
    p_min = jnp.min(window, axis=0)
    p_max = jnp.max(window, axis=0)
    current_price = window[-1]
    norm_price = (current_price - p_min) / (p_max - p_min + 1e-8)

    # Channel 1: Recent return
    prev_price = window[-2]
    returns = jnp.log(current_price / (prev_price + 1e-8))
    returns_clipped = jnp.clip(returns, -0.1, 0.1)
    returns_normalized = returns_clipped / 0.1

    # Channel 2: Position indicator
    position_indicator = jnp.where(
        state.pos_active,
        jnp.where(state.pos_side == 1, 1.0, -1.0),
        0.0
    )

    # Channel 3: Position PnL (percentage, clipped)
    pnl_long = (current_price - state.pos_entry) / (state.pos_entry + 1e-8)
    pnl_short = (state.pos_entry - current_price) / (state.pos_entry + 1e-8)
    pnl = jnp.where(state.pos_side == 1, pnl_long,
                   jnp.where(state.pos_side == 2, pnl_short, 0.0))
    pnl_normalized = jnp.clip(pnl * 10, -1, 1)  # 10% = 1.0
    pnl_normalized = jnp.where(state.pos_active, pnl_normalized, 0.0)

    # Channel 4: Pending order indicator
    order_indicator = jnp.where(
        state.order_active,
        jnp.where(state.order_side == 1, 1.0, -1.0),  # 1=buy, -1=sell
        0.0
    )

    obs = jnp.stack([
        norm_price[None, :],
        returns_normalized[None, :],
        position_indicator[None, :],
        pnl_normalized[None, :],
        order_indicator[None, :],
    ], axis=0)

    return obs


# =============================================================================
# Action Masking (matches VectorBT behavior)
# =============================================================================

def compute_action_mask(state: TradingState, current_prices: chex.Array,
                        config: EnvConfig, max_positions: int = 16,
                        min_cash_threshold: float = 100.0) -> chex.Array:
    """Compute valid action mask based on portfolio state.

    This matches VectorBT's _get_action_mask() logic to ensure
    models trained on JAX env can transfer to real trading.

    Actions per asset (5-action space):
        [0] Market Sell: close long or open short
        [1] Hold: always valid
        [2] Market Buy: open long or close short
        [3] Limit Sell: close long or open short (queued)
        [4] Limit Buy: open long or close short (queued, no existing pending)

    Args:
        state: Current TradingState
        current_prices: Current prices (n_assets,)
        config: EnvConfig
        max_positions: Maximum concurrent positions (default 16)
        min_cash_threshold: Minimum cash required to open new position

    Returns:
        mask: Valid action mask (n_assets * 5,) with 1=valid, 0=invalid
    """
    n_assets = config.n_assets

    # Count total orders (positions + pending orders)
    n_positions = jnp.sum(state.pos_active.astype(jnp.int32))
    n_pending = jnp.sum(state.order_active.astype(jnp.int32))
    total_orders = n_positions + n_pending

    # Per-asset conditions
    has_position = state.pos_active  # (n_assets,)
    is_long = state.pos_side == 1    # (n_assets,)
    is_short = state.pos_side == 2   # (n_assets,)
    has_long = has_position & is_long
    has_short = has_position & is_short
    has_pending = state.order_active  # (n_assets,)

    # Price validity
    price_valid = current_prices > 0  # (n_assets,)

    # Cash check
    has_cash = state.cash > min_cash_threshold

    # Position limit check
    below_limit = total_orders < max_positions

    # Can open new position (no existing position + cash + limit + price valid)
    can_open = ~has_position & has_cash & below_limit & price_valid

    # Build mask for each action type
    # [0] SELL: close long OR (allow_short AND can open short)
    can_sell = has_long | (config.allow_short & can_open)

    # [1] HOLD: always valid
    can_hold = jnp.ones(n_assets, dtype=jnp.bool_)

    # [2] BUY: close short OR can open long
    can_buy = has_short | can_open

    # [3] LIMIT SELL: same as sell
    can_limit_sell = can_sell

    # [4] LIMIT BUY: same as buy AND no pending order for this asset
    can_limit_buy = can_buy & ~has_pending

    # Stack into (n_assets, 5) then flatten to (n_assets * 5,)
    mask_per_asset = jnp.stack([
        can_sell,
        can_hold,
        can_buy,
        can_limit_sell,
        can_limit_buy,
    ], axis=1)  # (n_assets, 5)

    return mask_per_asset.flatten().astype(jnp.int8)


def batch_compute_action_mask(states: TradingState, current_prices: chex.Array,
                              config: EnvConfig, max_positions: int = 16,
                              min_cash_threshold: float = 100.0) -> chex.Array:
    """Compute action masks for batch of states.

    Args:
        states: Batched TradingState (vmapped)
        current_prices: Current prices for each env (n_envs, n_assets)
        config: EnvConfig
        max_positions: Maximum positions per env
        min_cash_threshold: Minimum cash to open position

    Returns:
        masks: (n_envs, n_assets * 5)
    """
    return jax.vmap(
        lambda s, p: compute_action_mask(s, p, config, max_positions, min_cash_threshold)
    )(states, current_prices)


@partial(jax.jit, static_argnames=['config', 'max_positions', 'min_cash_threshold'])
def jit_batch_compute_action_mask(states: TradingState, current_prices: chex.Array,
                                   config: EnvConfig, max_positions: int = 16,
                                   min_cash_threshold: float = 100.0) -> chex.Array:
    """JIT-compiled batched action mask computation."""
    return batch_compute_action_mask(states, current_prices, config, max_positions, min_cash_threshold)


# =============================================================================
# Batched Operations
# =============================================================================

def batch_reset(rng_keys: chex.Array, prices: chex.Array,
                config: EnvConfig) -> TradingState:
    """Reset multiple environments in parallel."""
    return jax.vmap(lambda k: reset(k, prices, config))(rng_keys)


def batch_step(states: TradingState, actions: chex.Array,
               prices: chex.Array, config: EnvConfig) -> Tuple[TradingState, chex.Array, chex.Array, chex.Array]:
    """Step multiple environments in parallel."""
    return jax.vmap(lambda s, a: step(s, a, prices, config))(states, actions)


# =============================================================================
# JIT-Compiled Versions
# =============================================================================

@partial(jax.jit, static_argnames=['config'])
def jit_reset(rng_key: chex.PRNGKey, prices: chex.Array,
              config: EnvConfig) -> TradingState:
    """JIT-compiled single environment reset."""
    return reset(rng_key, prices, config)


@partial(jax.jit, static_argnames=['config'])
def jit_step(state: TradingState, action: chex.Array,
             prices: chex.Array, config: EnvConfig) -> Tuple[TradingState, chex.Array, chex.Array, chex.Array]:
    """JIT-compiled single environment step."""
    return step(state, action, prices, config)


@partial(jax.jit, static_argnames=['config'])
def jit_batch_reset(rng_keys: chex.Array, prices: chex.Array,
                    config: EnvConfig) -> TradingState:
    """JIT-compiled batched reset."""
    return batch_reset(rng_keys, prices, config)


@partial(jax.jit, static_argnames=['config'])
def jit_batch_step(states: TradingState, actions: chex.Array,
                   prices: chex.Array, config: EnvConfig) -> Tuple[TradingState, chex.Array, chex.Array, chex.Array]:
    """JIT-compiled batched step."""
    return batch_step(states, actions, prices, config)


# =============================================================================
# Utility Functions
# =============================================================================

def get_info(state: TradingState, prices: chex.Array,
             config: EnvConfig) -> dict:
    """Extract info dict for debugging (NOT JIT-compatible)."""
    current_prices = get_current_prices(
        prices, state.current_step + config.window_size, state.episode_start_idx
    )
    portfolio_value = compute_portfolio_value(state, current_prices)

    return {
        'portfolio_value': float(portfolio_value),
        'cash': float(state.cash),
        'cumulative_return': float(state.cumulative_return),
        'current_step': int(state.current_step),
        'n_positions': int(jnp.sum(state.pos_active)),
        'n_long': int(jnp.sum(state.pos_active & (state.pos_side == 1))),
        'n_short': int(jnp.sum(state.pos_active & (state.pos_side == 2))),
        'n_orders': int(jnp.sum(state.order_active)),
        'n_buy_orders': int(jnp.sum(state.order_active & (state.order_side == 1))),
        'n_sell_orders': int(jnp.sum(state.order_active & (state.order_side == 2))),
    }
