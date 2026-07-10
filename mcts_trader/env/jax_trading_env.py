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
from mcts_trader.utils.transforms import symlog


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
    # TASK_047: Probabilistic limit order fills
    probabilistic_fills: bool = False  # Disabled by default for backwards compat
    # TASK_055: Borrowing cost for short positions
    borrow_rate_annual: float = 0.0  # Annual rate (e.g., 0.05 = 5%)
    # TASK_053: Episode length variance (anti-exploit)
    episode_variance: int = 0  # ±N steps variance (0 = fixed length)
    # TASK_051: Burning money (cash opportunity cost)
    burning_money: bool = False
    burn_market_linked: bool = True
    burn_base_rate: float = 0.00005  # 0.005% per step baseline
    # Random burns (wife with debit card extreme mode)
    burn_random: bool = False
    burn_random_prob: float = 0.02  # 2% chance per step
    burn_random_max: float = 500.0  # Max random burn amount
    burn_random_limit: int = 10  # Max burns per episode
    # TASK_818 Tier B: action-attributable realized-PnL-on-close reward (default OFF = canon)
    reward_attributable: bool = False
    # TASK_828 Stage 1: vector-action one-shot portfolio rebalance (default OFF = canon).
    # Gate for step_vector(); the scalar step() path never reads this field, so with it
    # OFF the canon path is bit-identical. Not yet wired through training/search (Stage 2/3).
    vector_action: bool = False
    # TASK_839: budget-constrained vector-step sizing (D7 opt (a); default OFF = canon).
    # Read ONLY inside step_vector(); with it OFF that path is bit-identical (the branch is
    # DCE'd — config is a static jit arg). Bounds aggregate simultaneous OPENING spend to
    # PV * budget_pct, which is <= cash at/near episode start (PV ~= cash) — eliminates the
    # step-0 env-wall bankruptcy (LEDGER 828s3/env-wall). NOT <= cash in general: on an
    # invested book (PV > cash) a breadth of opens can still overdraw (process_margin_call
    # resolves it, as OFF), but per-leg spend is always <= the OFF path so ON is never worse.
    vector_budget_step: bool = False
    vector_budget_pct: float = 0.99


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

    # TASK_053: Episode length variance
    max_episode_steps: chex.Array       # () - actual max steps for this episode

    # TASK_051: Random burns counter (wife with debit card mode)
    random_burns_remaining: chex.Array  # () - burns left this episode


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
# Margin Call / Liquidation Functions
# =============================================================================

def process_margin_call(state: TradingState, current_prices: chex.Array,
                       config: EnvConfig) -> TradingState:
    """Handle margin call when cash goes negative (short losses exceed cash).

    Realistic margin mechanics:
    1. Cash < 0 from short losses → liquidate long positions to cover
    2. Holding longs with no spare cash → perfectly fine (you own assets)
    3. Can't cover even after liquidating all longs → bankruptcy (portfolio wiped)

    This should be called AFTER any operation that might make cash negative
    (primarily short position closures).
    """
    # Only process if cash is negative
    needs_margin_call = state.cash < 0

    # Calculate deficit to cover
    deficit = jnp.abs(jnp.minimum(state.cash, 0.0))

    # Find long positions we can liquidate
    is_long = state.pos_active & (state.pos_side == 1)
    long_values = jnp.where(is_long, state.pos_size * current_prices, 0.0)

    # Apply fees to get net proceeds
    fee_rate = config.fee_bps / 10000.0
    slippage_rate = config.slippage_bps / 10000.0
    net_rate = 1.0 - fee_rate - slippage_rate
    long_proceeds = long_values * net_rate

    # Calculate cumulative proceeds (in asset order)
    cumulative_proceeds = jnp.cumsum(long_proceeds)

    # Determine which longs to liquidate to cover deficit
    # Liquidate all longs where cumulative proceeds before this one < deficit
    # (i.e., we still need more cash)
    prev_cumulative = jnp.concatenate([jnp.array([0.0]), cumulative_proceeds[:-1]])
    should_liquidate = is_long & (prev_cumulative < deficit) & needs_margin_call

    # Calculate actual proceeds from liquidated positions
    liquidation_proceeds = jnp.where(should_liquidate, long_proceeds, 0.0)
    total_liquidation_proceeds = jnp.sum(liquidation_proceeds)

    # New cash after liquidation
    new_cash = jnp.where(needs_margin_call,
                         state.cash + total_liquidation_proceeds,
                         state.cash)

    # Clear liquidated positions
    new_pos_active = jnp.where(should_liquidate, False, state.pos_active)
    new_pos_side = jnp.where(should_liquidate, 0, state.pos_side)
    new_pos_size = jnp.where(should_liquidate, 0.0, state.pos_size)
    new_pos_entry = jnp.where(should_liquidate, 0.0, state.pos_entry)
    new_pos_sl = jnp.where(should_liquidate, 0.0, state.pos_sl)
    new_pos_tp = jnp.where(should_liquidate, 0.0, state.pos_tp)

    # Check for bankruptcy: cash still negative after liquidating all longs
    # In this case, we zero out everything (portfolio wiped)
    total_long_value = jnp.sum(long_proceeds)
    is_bankrupt = needs_margin_call & (deficit > total_long_value)

    # Bankruptcy: wipe remaining positions and set cash to 0
    # (Can't go more negative - this is the floor)
    new_cash = jnp.where(is_bankrupt, 0.0, new_cash)
    new_pos_active = jnp.where(is_bankrupt, False, new_pos_active)
    new_pos_side = jnp.where(is_bankrupt, 0, new_pos_side)
    new_pos_size = jnp.where(is_bankrupt, 0.0, new_pos_size)
    new_pos_entry = jnp.where(is_bankrupt, 0.0, new_pos_entry)
    new_pos_sl = jnp.where(is_bankrupt, 0.0, new_pos_sl)
    new_pos_tp = jnp.where(is_bankrupt, 0.0, new_pos_tp)

    # Also cancel all pending orders on bankruptcy
    new_order_active = jnp.where(is_bankrupt, False, state.order_active)
    new_order_side = jnp.where(is_bankrupt, 0, state.order_side)
    new_order_size = jnp.where(is_bankrupt, 0.0, state.order_size)
    new_order_price = jnp.where(is_bankrupt, 0.0, state.order_price)
    new_order_reserved = jnp.where(is_bankrupt, 0.0, state.order_reserved)

    return state.replace(
        cash=new_cash,
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
    )


# =============================================================================
# Limit Order Functions
# =============================================================================

def process_limit_orders(state: TradingState, current_prices: chex.Array,
                        config: EnvConfig, rng_key: chex.PRNGKey = None) -> TradingState:
    """Process pending limit orders - check fills and expirations.

    LAZY EXECUTION: No reserved cash - just clear expired orders.
    This is called at the START of each step, before the action.

    TASK_047: Probabilistic fills based on distance from market.
    TASK_052: Random fill priority (no asset index bias).
    """
    # Check expiration: (current_step - created) >= ttl
    expired = state.order_active & ((state.current_step - state.order_created) >= state.order_ttl)

    # No reserved cash to return in lazy mode - orders just expire

    # Check fill conditions
    # Buy: fills when current_price <= limit_price
    # Sell: fills when current_price >= limit_price
    can_fill_buy = state.order_active & (state.order_side == 1) & (current_prices <= state.order_price)
    can_fill_sell = state.order_active & (state.order_side == 2) & (current_prices >= state.order_price)
    can_fill_basic = (can_fill_buy | can_fill_sell) & ~expired

    # TASK_047: Probabilistic fill model
    # Orders far from market fill easily; orders close to market may not fill (queue position)
    if rng_key is not None and config.probabilistic_fills:
        key1, key2 = jax.random.split(rng_key)

        # Calculate distance from market as fraction of price
        # For buys: (limit_price - market_price) / market_price (negative when in-the-money)
        # For sells: (market_price - limit_price) / market_price (negative when in-the-money)
        buy_distance = (state.order_price - current_prices) / (current_prices + 1e-8)
        sell_distance = (current_prices - state.order_price) / (current_prices + 1e-8)
        distance = jnp.where(state.order_side == 1, buy_distance, sell_distance)

        # Marketable orders (distance <= 0) always fill
        is_marketable = distance <= 0

        # For non-marketable orders, fill probability based on distance
        # Base probability: starts at 0.1 for orders very close to market
        # Increases with distance (farther = more aggressive = higher fill prob)
        # At 2% distance: ~50% fill prob; at 5% distance: ~95% fill prob
        # Formula: p = min(0.95, 0.1 + distance * 20)
        base_prob = jnp.clip(0.1 + jnp.abs(distance) * 20.0, 0.0, 0.95)

        # Order age bonus: +5% per step, max +30%
        order_age = state.current_step - state.order_created
        age_bonus = jnp.clip(order_age * 0.05, 0.0, 0.30)

        fill_prob = jnp.where(is_marketable, 1.0, jnp.clip(base_prob + age_bonus, 0.0, 0.95))

        # Random fill decision
        random_vals = jax.random.uniform(key1, shape=fill_prob.shape)
        probabilistic_fill = random_vals < fill_prob

        can_fill = can_fill_basic & probabilistic_fill
        rng_for_priority = key2
    else:
        can_fill = can_fill_basic
        rng_for_priority = rng_key

    # Process fills (handles insufficient funds by cancelling)
    state = fill_limit_orders(state, current_prices, can_fill, config, rng_for_priority)

    # Clear expired orders (no cash change in lazy mode)
    new_order_active = jnp.where(expired, False, state.order_active)
    new_order_side = jnp.where(expired, 0, state.order_side)
    new_order_size = jnp.where(expired, 0.0, state.order_size)
    new_order_price = jnp.where(expired, 0.0, state.order_price)
    new_order_reserved = jnp.where(expired, 0.0, state.order_reserved)
    new_order_created = jnp.where(expired, 0, state.order_created)
    new_order_ttl = jnp.where(expired, 0, state.order_ttl)

    return state.replace(
        order_active=new_order_active,
        order_side=new_order_side,
        order_size=new_order_size,
        order_price=new_order_price,
        order_reserved=new_order_reserved,
        order_created=new_order_created,
        order_ttl=new_order_ttl,
    )


def fill_limit_orders(state: TradingState, current_prices: chex.Array,
                     can_fill: chex.Array, config: EnvConfig,
                     rng_key: chex.PRNGKey = None) -> TradingState:
    """Fill limit orders that meet their price conditions.

    LAZY EXECUTION: Checks funds/position at fill time.
    Orders cancelled (soft penalty) if resources unavailable.

    Accounting Safety Guarantees:
    1. Buy orders (opening_long) use CUMULATIVE cash check to prevent overdraw
       - TASK_052: Orders processed in RANDOM order (no asset bias)
       - Sell proceeds (closing_long) counted towards available cash first
    2. Sell orders (closing_long, opening_short) don't consume cash, they generate it
    3. Closing short operations settle margin - may have P&L impact

    Edge Cases:
    - Multiple simultaneous buys: Handled by cumulative check (fixed 2026-01-04)
    - Short closing when deeply underwater: May impact cash if loss > margin
      (rare edge case, requires 100%+ loss which is margin-call territory)
    """
    # Maker fee for limit orders (typically lower than taker)
    maker_fee_rate = config.maker_fee_bps / 10000.0

    # Process each fillable order vectorized
    is_buy_order = can_fill & (state.order_side == 1)
    is_sell_order = can_fill & (state.order_side == 2)

    # Buy fill: open long or close short
    # FIX: Fill at LIMIT price, not market price (Magic Fill bug fix).
    # Old logic: jnp.minimum(order, market) gave unrealistic "better price"
    # e.g., Limit $98, Market $90 -> Fill $90 = free $8 alpha
    # Reality: As price drops from $100 to $90, it crosses $98 first -> fills at $98
    buy_fill_price = state.order_price
    buy_cost = state.order_size * buy_fill_price
    buy_fee = buy_cost * maker_fee_rate
    total_buy_cost = buy_cost + buy_fee

    # LAZY CHECK: For sell orders that close longs, calculate proceeds first
    # These generate cash that can fund buy orders
    closing_long_tentative = is_sell_order & state.pos_active & (state.pos_side == 1)
    # FIX: Fill at LIMIT price, not market price (Magic Fill bug fix).
    # Old logic: jnp.maximum(order, market) gave unrealistic "better price"
    # e.g., Limit $100, Market $110 -> Fill $110 = free $10 alpha
    # Reality: As price rises from $95 to $110, it crosses $100 first -> fills at $100
    sell_actual_price = state.order_price
    sell_proceeds = state.order_size * sell_actual_price
    sell_fee = sell_proceeds * maker_fee_rate
    sell_proceeds_net = jnp.where(closing_long_tentative, sell_proceeds - sell_fee, 0.0)
    total_sell_proceeds = jnp.sum(sell_proceeds_net)

    # Available cash for buys = current cash + proceeds from closing longs
    available_cash_for_buys = state.cash + total_sell_proceeds

    # For buy orders opening long, use CUMULATIVE check to prevent overdraw
    wants_to_open_long = is_buy_order & ~state.pos_active
    buy_cost_if_opening = jnp.where(wants_to_open_long, total_buy_cost, 0.0)

    # TASK_052: Randomize fill priority to prevent asset index bias
    # Without this, asset 0 always fills before asset 63 when cash is limited
    n_assets = buy_cost_if_opening.shape[0]
    if rng_key is not None:
        # Shuffle costs, do cumsum, then unshuffle result
        priority_order = jax.random.permutation(rng_key, n_assets)
        inverse_order = jnp.argsort(priority_order)
        shuffled_costs = buy_cost_if_opening[priority_order]
        cumulative_shuffled = jnp.cumsum(shuffled_costs)
        can_afford_shuffled = cumulative_shuffled <= available_cash_for_buys
        can_afford_cumulative = can_afford_shuffled[inverse_order]
    else:
        # Fallback: deterministic order (for backwards compatibility)
        cumulative_buy_cost = jnp.cumsum(buy_cost_if_opening)
        can_afford_cumulative = cumulative_buy_cost <= available_cash_for_buys

    # For closing short, always allow - margin call handles negative cash
    # This is realistic: you can't refuse to close a losing short, you owe the money
    # The margin call system will liquidate longs or bankrupt if needed
    closing_short = is_buy_order & state.pos_active & (state.pos_side == 2)

    # Final execution flags
    opening_long = wants_to_open_long & can_afford_cumulative

    # Cancel buy orders that can't be funded (opening_long only - shorts always close)
    buy_cancelled = is_buy_order & ~state.pos_active & ~opening_long

    # Close short proceeds: 2 * size * entry - size * fill_price
    short_close_proceeds = state.pos_size * (2.0 * state.pos_entry - buy_fill_price)
    short_close_fee = jnp.abs(short_close_proceeds) * maker_fee_rate

    # Sell fill: close long or open short
    # (sell_proceeds and sell_fee already calculated above for closing_long_tentative)
    # sell_actual_price is now just order_price (Magic Fill fix)
    sell_fill_price = sell_actual_price

    # Use closing_long_tentative as the final closing_long (same condition)
    closing_long = closing_long_tentative

    # TASK_124 FIX: Short opening via limit order needs margin deduction!
    # Previously: opening_short didn't deduct cash = FREE MONEY BUG
    # Short margin = position_size * entry_price (same as market order)
    short_margin_cost = state.order_size * sell_fill_price
    short_margin_fee = short_margin_cost * maker_fee_rate
    total_short_margin = short_margin_cost + short_margin_fee

    # Check if can afford short margin (like buy orders)
    # Must have cash AFTER close_long proceeds but BEFORE buy costs
    available_for_short = state.cash + total_sell_proceeds  # Like buy orders
    wants_to_open_short = is_sell_order & ~state.pos_active & config.allow_short
    can_afford_short = total_short_margin <= available_for_short
    opening_short = wants_to_open_short & can_afford_short

    # Cancel sell orders if position no longer exists (was market sold) OR can't afford
    sell_cancelled = is_sell_order & ~state.pos_active & ~config.allow_short
    sell_cancelled = sell_cancelled | (wants_to_open_short & ~can_afford_short)

    # Calculate cash changes (no reserved cash in lazy mode)
    # Buy to open long: -cost - fee
    # Buy to close short: +short_proceeds - fee
    # Sell to close long: +proceeds - fee
    # Sell to open short: -margin - fee (TASK_124 FIX: was 0.0!)

    cash_change_buy_long = jnp.where(opening_long, -total_buy_cost, 0.0)
    cash_change_buy_short = jnp.where(closing_short,
                                       short_close_proceeds - short_close_fee, 0.0)
    cash_change_sell_long = jnp.where(closing_long,
                                       sell_proceeds - sell_fee, 0.0)
    cash_change_sell_short = jnp.where(opening_short, -total_short_margin, 0.0)  # TASK_124 FIX

    total_cash_change = jnp.sum(cash_change_buy_long + cash_change_buy_short +
                                cash_change_sell_long + cash_change_sell_short)

    # Track which orders actually filled vs cancelled
    actually_filled = (opening_long | closing_short | closing_long | opening_short)
    cancelled = buy_cancelled | sell_cancelled

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

    # Clear filled AND cancelled orders (both should be removed)
    to_clear = actually_filled | cancelled
    new_order_active = jnp.where(to_clear, False, state.order_active)
    new_order_side = jnp.where(to_clear, 0, state.order_side)
    new_order_size = jnp.where(to_clear, 0.0, state.order_size)
    new_order_price = jnp.where(to_clear, 0.0, state.order_price)
    new_order_reserved = jnp.where(to_clear, 0.0, state.order_reserved)
    new_order_created = jnp.where(to_clear, 0, state.order_created)
    new_order_ttl = jnp.where(to_clear, 0, state.order_ttl)

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
    # BUG_009 FIX: Guard against division by zero (price should never be 0, but defensive)
    shares_to_buy = actual_cost / jnp.maximum(buy_exec_price, 1e-10)
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
    # BUG_009 FIX: Guard against division by zero in weighted average entry price
    # Even if the result isn't used, NaN in computation can affect gradients/JIT
    safe_divisor = jnp.maximum(new_long_size, 1e-10)
    new_long_entry = jnp.where(has_position & is_long,
                                (current_size * current_entry + shares_to_buy * buy_exec_price) / safe_divisor,
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

    # FIX: Cancel pending limit orders to prevent "Ghost Fills"
    # Market order = "I want this NOW", old conditional orders are stale
    # JAX env uses lazy execution (no reserved cash), so just deactivate
    new_order_active = state.order_active.at[asset_idx].set(False)

    return state.replace(
        cash=new_cash,
        pos_active=new_pos_active,
        pos_side=new_pos_side,
        pos_size=new_pos_size,
        pos_entry=new_pos_entry,
        pos_sl=new_pos_sl,
        pos_tp=new_pos_tp,
        order_active=new_order_active,
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
    # BUG_009 FIX: Guard against division by zero (price should never be 0, but defensive)
    shares_to_sell = actual_short_cost / jnp.maximum(price, 1e-10)
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

    # FIX: Cancel pending limit orders to prevent "Ghost Fills"
    # Market order = "I want this NOW", old conditional orders are stale
    # JAX env uses lazy execution (no reserved cash), so just deactivate
    new_order_active = state.order_active.at[asset_idx].set(False)

    return state.replace(
        cash=new_cash,
        pos_active=new_pos_active,
        pos_side=new_pos_side,
        pos_size=new_pos_size,
        pos_entry=new_pos_entry,
        pos_sl=new_pos_sl,
        pos_tp=new_pos_tp,
        order_active=new_order_active,
    )


def execute_limit_buy(state: TradingState, asset_idx: chex.Array,
                     current_prices: chex.Array, config: EnvConfig) -> TradingState:
    """Execute LIMIT BUY action - place buy order below current price.

    LAZY EXECUTION: Does NOT reserve cash. Funds checked at fill time.
    If insufficient funds when price triggers, order is cancelled (soft penalty).
    """
    price = current_prices[asset_idx]
    limit_price = price * (1.0 - config.limit_offset_pct)

    # Calculate intended order size (based on current portfolio, but no reservation)
    portfolio_value = compute_portfolio_value(state, current_prices)
    order_value = portfolio_value * config.position_size_pct
    shares = order_value / limit_price

    # Can place if no existing position on this asset
    # (multiple limit orders allowed - no ~has_pending check)
    has_position = state.pos_active[asset_idx]
    can_place = ~has_position & (shares > 0)

    # Update order arrays - just record intent, no cash movement
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
    # No reservation - set to 0
    new_order_reserved = state.order_reserved.at[asset_idx].set(
        jnp.where(can_place, 0.0, state.order_reserved[asset_idx])
    )
    new_order_created = state.order_created.at[asset_idx].set(
        jnp.where(can_place, state.current_step, state.order_created[asset_idx])
    )
    new_order_ttl = state.order_ttl.at[asset_idx].set(
        jnp.where(can_place, config.default_limit_order_ttl, state.order_ttl[asset_idx])
    )

    # NO cash change - lazy execution
    return state.replace(
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
    """Execute LIMIT SELL action - place sell order above current price.

    LAZY EXECUTION: Does NOT block shares. Position checked at fill time.
    If position no longer exists when price triggers, order is cancelled (soft penalty).
    Does NOT cancel existing limit orders - multiple orders allowed.
    """
    price = current_prices[asset_idx]
    limit_price = price * (1.0 + config.limit_offset_pct)

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

    # Update order arrays - just record intent, no position blocking
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
        jnp.where(can_place, 0.0, state.order_reserved[asset_idx])
    )
    new_order_created = state.order_created.at[asset_idx].set(
        jnp.where(can_place, state.current_step, state.order_created[asset_idx])
    )
    new_order_ttl = state.order_ttl.at[asset_idx].set(
        jnp.where(can_place, config.default_limit_order_ttl, state.order_ttl[asset_idx])
    )

    # NO cash/position changes - lazy execution
    return state.replace(
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

    key1, key2, key3 = jax.random.split(rng_key, 3)

    # TASK_053: Variable episode length
    if config.episode_variance > 0:
        # Random episode length: base ± variance
        length_offset = jax.random.randint(key3, (), -config.episode_variance, config.episode_variance + 1)
        actual_max_steps = jnp.maximum(config.max_steps + length_offset, 100)  # Min 100 steps
    else:
        actual_max_steps = jnp.array(config.max_steps)

    # TASK_121: Fix observation window to be LOOKBACK (not forward)
    # TASK_124: min_start now needs window_size (not window_size-1) because
    # observation window ends at t-1 to prevent same-step leak
    # At step 0: start_idx = episode_start - window_size, needs >= 0
    # max_start: need room for max_steps of trading after episode_start
    min_start = config.window_size
    max_start = n_steps - actual_max_steps - 1
    # TASK_122: Ensure at least 10 valid starting positions for diversity
    max_start = jnp.maximum(max_start, min_start + 10)
    episode_start = jax.random.randint(key1, (), min_start, max_start)

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
        max_episode_steps=actual_max_steps,  # TASK_053
        random_burns_remaining=jnp.array(config.burn_random_limit),  # TASK_051
    )


def step(state: TradingState, action: chex.Array,
         prices: chex.Array, config: EnvConfig) -> Tuple[TradingState, chex.Array, chex.Array, chex.Array]:
    """Execute one environment step.

    Order of operations:
    1. Process pending limit orders (fills and expirations)
    1b. Process margin calls (short losses from limit fills)
    2. Check TP/SL guardrails
    3. Execute the action
    4. Process margin calls (short losses from action)
    5. Check liquidation threshold
    6. Compute reward and observation

    Margin call mechanics:
    - Cash < 0 from short losses → liquidate longs to cover
    - Holding longs with $0 cash → perfectly fine
    - Can't cover → bankruptcy (portfolio wiped, cash = 0)
    """
    # Get current prices
    # FIX: Remove +window_size offset that caused 60-step future leakage
    # Trading must happen at same time as observation ends (current_step)
    current_prices = get_current_prices(
        prices, state.current_step, state.episode_start_idx
    )

    # Split RNG for stochastic operations (TASK_047, TASK_052)
    rng_key, limit_order_key = jax.random.split(state.rng_key)

    # 1. Process limit orders (with probabilistic fills and random priority)
    state = process_limit_orders(state, current_prices, config, limit_order_key)
    # Update RNG key in state
    state = state.replace(rng_key=rng_key)
    # Margin call after limit orders (limit buy might close short at loss)
    state = process_margin_call(state, current_prices, config)

    # 2. Check TP/SL guardrails
    state = apply_tp_sl_closures(state, current_prices, config)

    # 3. Decode and execute action
    asset_idx = action // 5
    action_type = action % 5

    # TASK_818 Tier B: snapshot pre-action position state for realized-PnL attribution.
    # Pure name-bindings to existing tracers — emit ZERO XLA ops, so they are harmless
    # even when the flag is OFF (they only feed the guarded branch below, which is not
    # traced when reward_attributable=False → dead, absent from the jaxpr).
    prev_active_attr = state.pos_active   # (n_assets,) bool   — post limit/tp_sl/margin, pre-action
    prev_side_attr   = state.pos_side     # (n_assets,) int 0/1/2
    prev_entry_attr  = state.pos_entry    # (n_assets,) float

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

    # 4. Process margin calls (if cash went negative from short losses)
    # This liquidates longs to cover, or bankrupts if can't cover
    new_state = process_margin_call(new_state, current_prices, config)

    # 5. Check liquidation
    new_portfolio_value = compute_portfolio_value(new_state, current_prices)
    liquidation_threshold = config.initial_cash * config.liquidation_threshold_pct
    is_liquidated = new_portfolio_value < liquidation_threshold

    # TASK_055: Compute borrowing cost for short positions
    # Annual rate converted to per-step rate (assuming ~400 steps = 1 day trading)
    # 252 trading days * 400 steps/day = 100,800 steps/year
    if config.borrow_rate_annual > 0:
        borrow_rate_step = config.borrow_rate_annual / 100800.0
        # Calculate value of short positions
        is_short = new_state.pos_active & (new_state.pos_side == 2)
        short_values = jnp.where(is_short, new_state.pos_size * current_prices, 0.0)
        total_short_value = jnp.sum(short_values)
        borrow_cost = total_short_value * borrow_rate_step
    else:
        borrow_cost = 0.0

    # FIX Bug #4 (Phantom Tax): Actually deduct borrow cost from cash
    # Before: borrow_cost penalized reward but agent kept all money
    # After: borrow_cost deducted from cash (affects portfolio value and future trades)
    new_cash_after_borrow = new_state.cash - borrow_cost
    new_state = new_state.replace(cash=new_cash_after_borrow)
    new_portfolio_value = new_portfolio_value - borrow_cost  # Adjust for reward calc

    # 5. Compute reward using Log Returns for geometric consistency (Bug #6 fix)
    # Arithmetic returns are biased: +50% then -33% (net zero) yields positive reward sum
    # Log returns are symmetric: log(1.5) + log(0.66) ≈ 0
    # We then apply symlog for range compression (MuZero value support)
    safe_prev = jnp.maximum(state.prev_portfolio_value, 1e-8)
    safe_new = jnp.maximum(new_portfolio_value, 1e-8)
    log_return = jnp.log(safe_new / safe_prev) * 100.0
    reward = jnp.where(is_liquidated, symlog(-50.0), symlog(log_return))

    # TASK_818 Tier B (flag --reward-attributable, default OFF → canon bit-identical):
    # credit realized log-PnL on a True->False position flip (a CLOSE) caused by THIS step.
    # Long  realized = log(exit/entry);  Short = mirror = -log(exit/entry) = log(entry/exit).
    # This is the multi-step holding-period return, attributed to the CLOSE action.
    if config.reward_attributable:
        closed     = prev_active_attr & (~new_state.pos_active)          # (n_assets,) bool
        safe_exit  = jnp.maximum(current_prices, 1e-8)                   # close exec ≈ current (== at cost=0)
        safe_entry = jnp.maximum(prev_entry_attr, 1e-8)
        long_log   = jnp.log(safe_exit / safe_entry)                    # log(exit/entry)
        realized   = jnp.where(prev_side_attr == 1, long_log,
                      jnp.where(prev_side_attr == 2, -long_log, 0.0))    # short = mirror
        realized   = jnp.where(closed, realized, 0.0)                   # zero where no close
        realized_sum = jnp.sum(realized)                                # scalar; ~1 close/step
        # Don't stack the realized spike onto the liquidation sentinel (symlog(-50)).
        reward = reward + jnp.where(is_liquidated, 0.0, symlog(realized_sum * 100.0))

    # TASK_051: Burning money (cash opportunity cost) - wife with debit card simulation
    if config.burning_money:
        # Cash ratio (fraction uninvested)
        cash_ratio = new_state.cash / jnp.maximum(new_portfolio_value, 1e-8)
        cash_ratio = jnp.clip(cash_ratio, 0.0, 1.0)

        burn = jnp.array(0.0)

        # Market-linked burn (opportunity cost)
        if config.burn_market_linked:
            # Current step prices vs previous step
            step_idx = state.episode_start_idx + state.current_step
            prev_step_idx = jnp.maximum(step_idx - 1, 0)
            current_prices_step = prices[step_idx]
            prev_prices_step = prices[prev_step_idx]

            # Average market return this step
            returns = (current_prices_step - prev_prices_step) / jnp.maximum(prev_prices_step, 1e-8)
            market_return = jnp.mean(returns)

            # Cash "missed" this return (positive market = penalty for holding cash)
            # BUG-035 FIX: Clamp to non-negative - in bear market, holding cash is correct
            # (no opportunity cost), so we shouldn't reward OR penalize. Previously,
            # negative market_return led to negative burn which became a reward via symlog.
            market_burn = jnp.maximum(market_return * cash_ratio, 0.0)
            burn = burn + market_burn

        # Base decay (constant pressure to deploy capital)
        base_burn = config.burn_base_rate * cash_ratio
        burn = burn + base_burn

        # Subtract burn from reward (apply symlog to keep in same scale)
        reward = reward - symlog(burn * 100.0)  # Scale to match log_return scale

    # TASK_051: Random burns (wife with debit card extreme mode)
    # Split RNG key for random burn decision
    rng_key, burn_key1, burn_key2 = jax.random.split(new_state.rng_key, 3)
    new_state = new_state.replace(rng_key=rng_key)

    if config.burn_random:
        # Roll for random burn
        roll = jax.random.uniform(burn_key1, ())
        burn_amount = jax.random.uniform(burn_key2, (), minval=0.0, maxval=config.burn_random_max)

        # Conditions: probability check AND burns remaining AND enough cash
        should_burn = (roll < config.burn_random_prob) & (new_state.random_burns_remaining > 0)
        actual_burn = jnp.where(should_burn, jnp.minimum(burn_amount, new_state.cash), 0.0)

        # Apply burn: deduct from cash and reward
        new_cash = new_state.cash - actual_burn
        new_state = new_state.replace(cash=new_cash)

        # Decrement counter only if burn happened
        new_burns_remaining = jnp.where(
            should_burn,
            new_state.random_burns_remaining - 1,
            new_state.random_burns_remaining
        )
        new_state = new_state.replace(random_burns_remaining=new_burns_remaining)

        # Add burn penalty to reward (proportional to portfolio, symlog scaled)
        burn_pct = actual_burn / jnp.maximum(new_portfolio_value, 1e-8) * 100.0
        reward = reward - symlog(burn_pct)

        # TASK_126: Update portfolio value to reflect cash burn
        # This ensures cumulative_return and prev_portfolio_value are consistent
        new_portfolio_value = new_portfolio_value - actual_burn

    # Update tracking
    new_cumulative_return = (new_portfolio_value / state.initial_portfolio_value - 1.0) * 100.0

    # Advance step and check termination
    # TASK_053: Use per-episode max_steps for variable length episodes
    new_step = state.current_step + 1
    done = (new_step >= state.max_episode_steps) | is_liquidated

    final_state = new_state.replace(
        current_step=new_step,
        prev_portfolio_value=new_portfolio_value,
        cumulative_return=new_cumulative_return,
    )

    # Compute observation
    obs = compute_observation(final_state, prices, config)

    return final_state, obs, reward, done


# =============================================================================
# Vector-Action Step (TASK_828 Stage 1) — one-shot portfolio rebalance
# =============================================================================

def step_vector(state: TradingState, action_vec: chex.Array,
                prices: chex.Array, config: EnvConfig
                ) -> Tuple[TradingState, chex.Array, chex.Array, chex.Array]:
    """Execute one VECTOR-action step: all assets acted on simultaneously.

    ``action_vec`` is ``(n_assets,)`` with each entry in ``{0..4}`` using the SAME
    per-asset action codes as the scalar path (0=sell, 1=hold, 2=buy, 3=limit_sell,
    4=limit_buy). Where the scalar ``step`` decodes a single ``action`` into one
    ``(asset_idx, action_type)`` and touches one asset, ``step_vector`` applies every
    asset's action_type in one step = a one-shot portfolio rebalance.

    The per-asset fill/fee/margin/PnL math is the SAME as ``execute_sell``/``execute_hold``/
    ``execute_buy``/``execute_limit_sell``/``execute_limit_buy``, just vectorised over all
    assets with ``jnp.where`` selection keyed on ``action_vec`` (no ``lax.switch``/``lax.cond``,
    so this is vmap-safe). Reward is the same whole-portfolio log-return formula as ``step``
    (plus the optional ``--reward-attributable`` realized-PnL-on-close term, which now sums the
    realized PnL of EVERY asset closed by this rebalance).

    INTENDED simultaneous-rebalance divergence from an equivalent SEQUENCE of scalar steps
    (documented, and asserted in tests/test_vector_env.py):
      * All per-asset decisions are sized against ONE snapshot of ``cash`` /
        ``compute_portfolio_value`` taken before any fill (a true simultaneous rebalance),
        whereas sequential scalar steps re-read cash/portfolio after each fill. When several
        buys/short-opens are requested at once their sizes therefore do not "see" each other's
        cash consumption; combined demand can drive ``cash`` negative, which ``process_margin_call``
        then resolves (liquidate longs / bankrupt) exactly as in the scalar path. The two agree
        exactly whenever affordability never binds and portfolio value is invariant across the
        rebalance (e.g. flat prices, ample cash, distinct assets, fee=0) — the equivalence test
        constructs precisely that regime.
      * A scalar SEQUENCE also advances ``current_step`` (and thus the price index / RNG) once
        per asset; ``step_vector`` advances it once for the whole rebalance. The equivalence test
        holds prices flat so the price index is immaterial and compares material state only.
    """
    # Get current prices (identical to scalar step)
    current_prices = get_current_prices(
        prices, state.current_step, state.episode_start_idx
    )

    # Split RNG (mirror scalar step so limit-fill / burn stochasticity is preserved)
    rng_key, limit_order_key = jax.random.split(state.rng_key)

    # 1. Process pending limit orders, 1b. margin, 2. tp/sl — same action-independent preamble
    state = process_limit_orders(state, current_prices, config, limit_order_key)
    state = state.replace(rng_key=rng_key)
    state = process_margin_call(state, current_prices, config)
    state = apply_tp_sl_closures(state, current_prices, config)

    # Snapshot pre-action position state for realized-PnL attribution (same as scalar step).
    prev_active_attr = state.pos_active
    prev_side_attr   = state.pos_side
    prev_entry_attr  = state.pos_entry

    # ---- 3. Vectorised action execution (one-shot rebalance against a single snapshot) ----
    n_assets = config.n_assets
    fee_rate = config.fee_bps / 10000.0

    has_position = state.pos_active
    is_long  = state.pos_side == 1
    is_short = state.pos_side == 2
    cur_size  = state.pos_size
    cur_entry = state.pos_entry

    # ONE snapshot of cash / portfolio value (the simultaneity — see docstring divergence note).
    snap_cash = state.cash
    portfolio_value = compute_portfolio_value(state, current_prices)
    ideal_trade_value = portfolio_value * config.position_size_pct
    available_cash = snap_cash * 0.99  # 1% buffer, mirrors scalar execute_buy/sell
    min_trade_value = 100.0
    has_min_cash = snap_cash > min_trade_value

    a_sell  = action_vec == 0
    a_hold  = action_vec == 1
    a_buy   = action_vec == 2
    a_lsell = action_vec == 3
    a_lbuy  = action_vec == 4

    # TASK_839 (D7 opt (a)): budget-constrained spend sizing for the market legs.
    # OFF (default) => ideal_spend_value IS ideal_trade_value, so the two lines below that
    # consume it (actual_cost / actual_short_cost) are byte-identical to canon and the whole
    # block is DCE'd (config is a static jit arg). ON => each simultaneous OPENING trade sizes
    # at PV * min(position_size_pct, budget_pct / n_open_trades). This bounds AGGREGATE opening
    # spend to PV * budget_pct, which is <= cash at/near episode start (PV ~= cash — the step-0
    # env-wall regime), eliminating the step-0 bankruptcy (LEDGER 828s3/env-wall). It does NOT
    # guarantee <= cash in general: on an already-invested book (PV > cash) a breadth of opens
    # can still overdraw and process_margin_call resolves it exactly as before — but per-leg
    # spend is always <= the OFF path (both clamp each leg at available_cash), so ON is never
    # worse than canon.
    #   * n_trades counts SPEND legs only — opening/adding longs (a_buy, not closing a short)
    #     and opening shorts (a_sell, no position, shorting allowed). Closing trades
    #     (buy-of-short, sell-of-long) are PROCEEDS and are excluded.
    #   * The divisor uses only action + position + the size-insensitive `has_min_cash`, so
    #     there is NO circular dependence on the sizing we are computing.
    #   * LIMIT orders (ls_*/lb_* below) DELIBERATELY keep the original ideal_trade_value:
    #     they reserve 0 cash immediately and fill one-at-a-time via process_limit_orders, so
    #     they are not the simultaneous-overdraft source this fix targets (market-spend-only).
    ideal_spend_value = ideal_trade_value  # OFF: byte-identical to today
    if config.vector_budget_step:
        open_long_mask  = a_buy  & (~(has_position & is_short)) & has_min_cash
        open_short_mask = a_sell & (~has_position) & config.allow_short & has_min_cash
        n_trades = jnp.sum(open_long_mask.astype(jnp.int32) + open_short_mask.astype(jnp.int32))
        pct_eff = jnp.minimum(config.position_size_pct,
                              config.vector_budget_pct / jnp.maximum(n_trades, 1))
        ideal_spend_value = portfolio_value * pct_eff

    # --- BUY branch (mirror execute_buy) ---
    buy_exec_price = apply_fee_and_slippage(current_prices, True, config.fee_bps, config.slippage_bps)
    actual_cost = jnp.minimum(ideal_spend_value, available_cash)  # scalar (TASK_839: spend-sized)
    shares_to_buy = actual_cost / jnp.maximum(buy_exec_price, 1e-10)
    short_proceeds_buy = cur_size * (2.0 * cur_entry - buy_exec_price)
    fee_on_close_short = jnp.abs(short_proceeds_buy) * fee_rate
    can_afford_buy = has_min_cash & (shares_to_buy > 0)
    new_long_size = jnp.where(has_position & is_long, cur_size + shares_to_buy, shares_to_buy)
    safe_divisor = jnp.maximum(new_long_size, 1e-10)
    new_long_entry = jnp.where(has_position & is_long,
                               (cur_size * cur_entry + shares_to_buy * buy_exec_price) / safe_divisor,
                               buy_exec_price)
    closing_short = has_position & is_short
    opening_long  = (~has_position) & can_afford_buy
    adding_long   = has_position & is_long & can_afford_buy
    buy_change    = closing_short | opening_long | adding_long
    buy_cash_delta = jnp.where(closing_short, short_proceeds_buy - fee_on_close_short,
                     jnp.where(opening_long | adding_long, -actual_cost, 0.0))
    buy_pos_active = jnp.where(closing_short, False,
                     jnp.where(opening_long | adding_long, True, has_position))
    buy_pos_side = jnp.where(closing_short, 0,
                   jnp.where(opening_long | adding_long, 1, state.pos_side))
    buy_pos_size = jnp.where(closing_short, 0.0,
                   jnp.where(opening_long | adding_long, new_long_size, cur_size))
    buy_pos_entry = jnp.where(closing_short, 0.0,
                    jnp.where(opening_long | adding_long, new_long_entry, cur_entry))
    buy_pos_sl = jnp.where(closing_short, 0.0,
                 jnp.where(opening_long & config.tp_sl_enabled,
                           buy_exec_price * (1.0 - config.default_stop_loss_pct), state.pos_sl))
    buy_pos_tp = jnp.where(closing_short, 0.0,
                 jnp.where(opening_long & config.tp_sl_enabled,
                           buy_exec_price * (1.0 + config.default_take_profit_pct), state.pos_tp))

    # --- SELL branch (mirror execute_sell) ---
    sell_exec_price = apply_fee_and_slippage(current_prices, False, config.fee_bps, config.slippage_bps)
    long_proceeds = cur_size * sell_exec_price
    fee_on_close_long = long_proceeds * fee_rate
    actual_short_cost = jnp.minimum(ideal_spend_value, available_cash)  # scalar (TASK_839: spend-sized)
    shares_to_sell = actual_short_cost / jnp.maximum(current_prices, 1e-10)
    fee_on_open_short = actual_short_cost * fee_rate
    can_afford_short = has_min_cash & (shares_to_sell > 0)
    closing_long  = has_position & is_long
    opening_short = (~has_position) & config.allow_short & can_afford_short
    sell_cash_delta = jnp.where(closing_long, long_proceeds - fee_on_close_long,
                      jnp.where(opening_short, -actual_short_cost - fee_on_open_short, 0.0))
    sell_pos_active = jnp.where(closing_long, False,
                      jnp.where(opening_short, True, has_position))
    sell_pos_side = jnp.where(closing_long, 0,
                    jnp.where(opening_short, 2, state.pos_side))
    sell_pos_size = jnp.where(closing_long, 0.0,
                    jnp.where(opening_short, shares_to_sell, cur_size))
    sell_pos_entry = jnp.where(closing_long, 0.0,
                     jnp.where(opening_short, sell_exec_price, cur_entry))
    sell_pos_sl = jnp.where(closing_long, 0.0,
                  jnp.where(opening_short & config.tp_sl_enabled,
                            sell_exec_price * (1.0 + config.default_stop_loss_pct), state.pos_sl))
    sell_pos_tp = jnp.where(closing_long, 0.0,
                  jnp.where(opening_short & config.tp_sl_enabled,
                            sell_exec_price * (1.0 - config.default_take_profit_pct), state.pos_tp))

    # --- LIMIT SELL branch (mirror execute_limit_sell) ---
    ls_limit_price = current_prices * (1.0 + config.limit_offset_pct)
    ls_shares_for_short = ideal_trade_value / jnp.maximum(ls_limit_price, 1e-10)
    ls_order_size = jnp.where(has_position & is_long, cur_size, ls_shares_for_short)
    can_place_ls = ((has_position & is_long) | ((~has_position) & config.allow_short)) & (ls_order_size > 0)

    # --- LIMIT BUY branch (mirror execute_limit_buy) ---
    lb_limit_price = current_prices * (1.0 - config.limit_offset_pct)
    lb_shares = ideal_trade_value / jnp.maximum(lb_limit_price, 1e-10)
    can_place_lb = (~has_position) & (lb_shares > 0)

    # --- Combine positions: default = unchanged (hold / limit branches don't move positions) ---
    new_pos_active = jnp.where(a_buy, buy_pos_active,
                     jnp.where(a_sell, sell_pos_active, state.pos_active))
    new_pos_side = jnp.where(a_buy, buy_pos_side,
                   jnp.where(a_sell, sell_pos_side, state.pos_side))
    new_pos_size = jnp.where(a_buy, buy_pos_size,
                   jnp.where(a_sell, sell_pos_size, state.pos_size))
    new_pos_entry = jnp.where(a_buy, buy_pos_entry,
                    jnp.where(a_sell, sell_pos_entry, state.pos_entry))
    new_pos_sl = jnp.where(a_buy, buy_pos_sl,
                 jnp.where(a_sell, sell_pos_sl, state.pos_sl))
    new_pos_tp = jnp.where(a_buy, buy_pos_tp,
                 jnp.where(a_sell, sell_pos_tp, state.pos_tp))

    # --- Combine cash: sum per-asset deltas onto the single snapshot cash ---
    cash_delta = jnp.where(a_buy, buy_cash_delta,
                 jnp.where(a_sell, sell_cash_delta, 0.0))
    new_cash = snap_cash + jnp.sum(cash_delta)

    # --- Combine orders: market (buy/sell) cancels pending order_active (mirrors scalar);
    #     limit_sell/limit_buy place a new order where allowed; hold leaves orders untouched. ---
    market = a_buy | a_sell
    place_ls = a_lsell & can_place_ls
    place_lb = a_lbuy & can_place_lb
    place_any = place_ls | place_lb
    new_order_active = jnp.where(market, False,
                       jnp.where(place_any, True, state.order_active))
    new_order_side = jnp.where(place_ls, 2,
                     jnp.where(place_lb, 1, state.order_side))
    new_order_size = jnp.where(place_ls, ls_order_size,
                     jnp.where(place_lb, lb_shares, state.order_size))
    new_order_price = jnp.where(place_ls, ls_limit_price,
                      jnp.where(place_lb, lb_limit_price, state.order_price))
    new_order_reserved = jnp.where(place_any, 0.0, state.order_reserved)
    new_order_created = jnp.where(place_any, state.current_step, state.order_created)
    new_order_ttl = jnp.where(place_any, config.default_limit_order_ttl, state.order_ttl)

    new_state = state.replace(
        cash=new_cash,
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

    # 4. Margin call (short losses from the rebalance may have driven cash negative)
    new_state = process_margin_call(new_state, current_prices, config)

    # 5. Liquidation check (identical to scalar step)
    new_portfolio_value = compute_portfolio_value(new_state, current_prices)
    liquidation_threshold = config.initial_cash * config.liquidation_threshold_pct
    is_liquidated = new_portfolio_value < liquidation_threshold

    # TASK_055: borrow cost for shorts (identical to scalar step)
    if config.borrow_rate_annual > 0:
        borrow_rate_step = config.borrow_rate_annual / 100800.0
        is_short_new = new_state.pos_active & (new_state.pos_side == 2)
        short_values = jnp.where(is_short_new, new_state.pos_size * current_prices, 0.0)
        total_short_value = jnp.sum(short_values)
        borrow_cost = total_short_value * borrow_rate_step
    else:
        borrow_cost = 0.0

    new_cash_after_borrow = new_state.cash - borrow_cost
    new_state = new_state.replace(cash=new_cash_after_borrow)
    new_portfolio_value = new_portfolio_value - borrow_cost

    # 6. Reward: whole vector-step portfolio log-return (same base formula as scalar step)
    safe_prev = jnp.maximum(state.prev_portfolio_value, 1e-8)
    safe_new = jnp.maximum(new_portfolio_value, 1e-8)
    log_return = jnp.log(safe_new / safe_prev) * 100.0
    reward = jnp.where(is_liquidated, symlog(-50.0), symlog(log_return))

    # Optional --reward-attributable term: realized log-PnL summed over EVERY asset this
    # rebalance closed (True->False flip). Same math as scalar step; naturally multi-close here.
    if config.reward_attributable:
        closed     = prev_active_attr & (~new_state.pos_active)
        safe_exit  = jnp.maximum(current_prices, 1e-8)
        safe_entry = jnp.maximum(prev_entry_attr, 1e-8)
        long_log   = jnp.log(safe_exit / safe_entry)
        realized   = jnp.where(prev_side_attr == 1, long_log,
                     jnp.where(prev_side_attr == 2, -long_log, 0.0))
        realized   = jnp.where(closed, realized, 0.0)
        realized_sum = jnp.sum(realized)
        reward = reward + jnp.where(is_liquidated, 0.0, symlog(realized_sum * 100.0))

    # TASK_051: burning money (identical to scalar step)
    if config.burning_money:
        cash_ratio = new_state.cash / jnp.maximum(new_portfolio_value, 1e-8)
        cash_ratio = jnp.clip(cash_ratio, 0.0, 1.0)
        burn = jnp.array(0.0)
        if config.burn_market_linked:
            step_idx = state.episode_start_idx + state.current_step
            prev_step_idx = jnp.maximum(step_idx - 1, 0)
            current_prices_step = prices[step_idx]
            prev_prices_step = prices[prev_step_idx]
            returns = (current_prices_step - prev_prices_step) / jnp.maximum(prev_prices_step, 1e-8)
            market_return = jnp.mean(returns)
            market_burn = jnp.maximum(market_return * cash_ratio, 0.0)
            burn = burn + market_burn
        base_burn = config.burn_base_rate * cash_ratio
        burn = burn + base_burn
        reward = reward - symlog(burn * 100.0)

    # TASK_051: random burns (identical to scalar step)
    rng_key, burn_key1, burn_key2 = jax.random.split(new_state.rng_key, 3)
    new_state = new_state.replace(rng_key=rng_key)
    if config.burn_random:
        roll = jax.random.uniform(burn_key1, ())
        burn_amount = jax.random.uniform(burn_key2, (), minval=0.0, maxval=config.burn_random_max)
        should_burn = (roll < config.burn_random_prob) & (new_state.random_burns_remaining > 0)
        actual_burn = jnp.where(should_burn, jnp.minimum(burn_amount, new_state.cash), 0.0)
        new_cash = new_state.cash - actual_burn
        new_state = new_state.replace(cash=new_cash)
        new_burns_remaining = jnp.where(should_burn,
                                        new_state.random_burns_remaining - 1,
                                        new_state.random_burns_remaining)
        new_state = new_state.replace(random_burns_remaining=new_burns_remaining)
        burn_pct = actual_burn / jnp.maximum(new_portfolio_value, 1e-8) * 100.0
        reward = reward - symlog(burn_pct)
        new_portfolio_value = new_portfolio_value - actual_burn

    # Update tracking / advance step (identical to scalar step)
    new_cumulative_return = (new_portfolio_value / state.initial_portfolio_value - 1.0) * 100.0
    new_step = state.current_step + 1
    done = (new_step >= state.max_episode_steps) | is_liquidated

    final_state = new_state.replace(
        current_step=new_step,
        prev_portfolio_value=new_portfolio_value,
        cumulative_return=new_cumulative_return,
    )

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

    # TASK_121: Use LOOKBACK window (not forward)
    # TASK_122: Clamp to prevent negative index if episode_start_idx < window_size-1
    # TASK_124: Fix same-step observation leak - window ends at t-1, not t
    # Agent observes [t-window, t-1], executes at t (no look-ahead bias)
    window_start = state.episode_start_idx + state.current_step - config.window_size
    window_start = jnp.maximum(window_start, 0)

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

    # TASK_121: Use LOOKBACK window (not forward)
    # TASK_122: Clamp to prevent negative index
    # TASK_124: Fix same-step observation leak - window ends at t-1, not t
    window_start = state.episode_start_idx + state.current_step - config.window_size
    window_start = jnp.maximum(window_start, 0)

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
    # TASK_121: Use LOOKBACK window (not forward)
    # TASK_122: Clamp to prevent negative index
    # TASK_124: Fix same-step observation leak - window ends at t-1, not t
    # Agent observes [t-window, t-1], executes at t (no look-ahead bias)
    start_idx = state.episode_start_idx + state.current_step - config.window_size
    start_idx = jnp.maximum(start_idx, 0)
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

    # [3] LIMIT SELL: same as sell (lazy execution - multiple orders allowed)
    can_limit_sell = can_sell

    # [4] LIMIT BUY: same as buy (lazy execution - multiple orders allowed)
    # No ~has_pending check - agent can place multiple limit orders
    can_limit_buy = can_buy

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
    """Reset multiple environments in parallel (shared prices)."""
    return jax.vmap(lambda k: reset(k, prices, config))(rng_keys)


def batch_step(states: TradingState, actions: chex.Array,
               prices: chex.Array, config: EnvConfig) -> Tuple[TradingState, chex.Array, chex.Array, chex.Array]:
    """Step multiple environments in parallel (shared prices)."""
    return jax.vmap(lambda s, a: step(s, a, prices, config))(states, actions)


def batch_step_vector(states: TradingState, action_vecs: chex.Array,
                      prices: chex.Array, config: EnvConfig) -> Tuple[TradingState, chex.Array, chex.Array, chex.Array]:
    """Vector-action step across a batch of envs (shared prices). TASK_828 Stage 1.

    Args:
        states: Batched TradingState
        action_vecs: (n_envs, n_assets) per-asset action codes in {0..4}
    """
    return jax.vmap(lambda s, a: step_vector(s, a, prices, config))(states, action_vecs)


# =============================================================================
# Per-Env Unique Prices (each env has independent price data)
# =============================================================================

def batch_reset_unique(rng_keys: chex.Array, prices: chex.Array,
                       config: EnvConfig) -> TradingState:
    """Reset multiple environments with unique prices per env.

    Args:
        rng_keys: (n_envs, 2) array of RNG keys
        prices: (n_envs, n_steps, n_assets) - each env has its own price data
        config: Environment config

    Returns:
        Batched TradingState
    """
    return jax.vmap(lambda k, p: reset(k, p, config))(rng_keys, prices)


def batch_step_unique(states: TradingState, actions: chex.Array,
                      prices: chex.Array, config: EnvConfig) -> Tuple[TradingState, chex.Array, chex.Array, chex.Array]:
    """Step multiple environments with unique prices per env.

    Args:
        states: Batched TradingState
        actions: (n_envs,) actions
        prices: (n_envs, n_steps, n_assets) - each env has its own price data
        config: Environment config

    Returns:
        (new_states, observations, rewards, dones)
    """
    return jax.vmap(lambda s, a, p: step(s, a, p, config))(states, actions, prices)


def batch_step_vector_unique(states: TradingState, action_vecs: chex.Array,
                             prices: chex.Array, config: EnvConfig) -> Tuple[TradingState, chex.Array, chex.Array, chex.Array]:
    """Vector-action step with UNIQUE prices per env. TASK_828 Stage 3.

    Mirrors ``batch_step_unique`` (per-env 3-D prices) but calls ``step_vector``
    so a per-asset action VECTOR is executed against each env's own price data.
    Without this, vector mode on unique-price batches would silently run the
    SCALAR ``step`` on a vector action and corrupt trajectories (DEC-2).

    Args:
        states: Batched TradingState
        action_vecs: (n_envs, n_assets) per-asset action codes in {0..4}
        prices: (n_envs, n_steps, n_assets) — each env has its own price data
        config: Environment config

    Returns:
        (new_states, observations, rewards, dones)
    """
    return jax.vmap(lambda s, a, p: step_vector(s, a, p, config))(states, action_vecs, prices)


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
def jit_step_vector(state: TradingState, action_vec: chex.Array,
                    prices: chex.Array, config: EnvConfig) -> Tuple[TradingState, chex.Array, chex.Array, chex.Array]:
    """JIT-compiled single-env vector-action step. TASK_828 Stage 1."""
    return step_vector(state, action_vec, prices, config)


@partial(jax.jit, static_argnames=['config'])
def jit_batch_step_vector(states: TradingState, action_vecs: chex.Array,
                          prices: chex.Array, config: EnvConfig) -> Tuple[TradingState, chex.Array, chex.Array, chex.Array]:
    """JIT-compiled batched vector-action step. TASK_828 Stage 1."""
    return batch_step_vector(states, action_vecs, prices, config)


@partial(jax.jit, static_argnames=['config'])
def jit_batch_reset(rng_keys: chex.Array, prices: chex.Array,
                    config: EnvConfig) -> TradingState:
    """JIT-compiled batched reset."""
    return batch_reset(rng_keys, prices, config)


@partial(jax.jit, static_argnames=['config'])
def jit_batch_reset_unique(rng_keys: chex.Array, prices: chex.Array,
                           config: EnvConfig) -> TradingState:
    """JIT-compiled batched reset with unique prices per env."""
    return batch_reset_unique(rng_keys, prices, config)


@partial(jax.jit, static_argnames=['config'])
def jit_batch_step_unique(states: TradingState, actions: chex.Array,
                          prices: chex.Array, config: EnvConfig) -> Tuple[TradingState, chex.Array, chex.Array, chex.Array]:
    """JIT-compiled batched step with unique prices per env."""
    return batch_step_unique(states, actions, prices, config)


@partial(jax.jit, static_argnames=['config'])
def jit_batch_step_vector_unique(states: TradingState, action_vecs: chex.Array,
                                 prices: chex.Array, config: EnvConfig) -> Tuple[TradingState, chex.Array, chex.Array, chex.Array]:
    """JIT-compiled batched vector-action step with unique prices per env. TASK_828 Stage 3."""
    return batch_step_vector_unique(states, action_vecs, prices, config)


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
    # FIX: Remove +window_size offset (same fix as step function)
    current_prices = get_current_prices(
        prices, state.current_step, state.episode_start_idx
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
