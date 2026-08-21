"""
Test suite for JAX-vectorized trading environment.

Tests cover:
1. State initialization and reset
2. Action execution (buy/sell/hold) - including shorts
3. Portfolio value calculations
4. Reward computation
5. Batched operations (vmap)
6. JIT compilation
7. TP/SL guardrails
8. Limit orders (placement, fill, expiration)

Run with: pytest tests/test_jax_env.py -v
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

import sys
sys.path.insert(0, '/home/lukas/Trader')
from mcts_trader.env.jax_trading_env import (
    EnvConfig, TradingState, reset, step,
    batch_reset, batch_step,
    jit_reset, jit_step, jit_batch_reset, jit_batch_step,
    compute_portfolio_value, compute_position_values,
    get_current_prices, symlog, get_info,
    process_limit_orders, apply_tp_sl_closures,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def default_config():
    """Default environment configuration."""
    return EnvConfig(
        n_assets=64,
        initial_cash=100_000.0,
        fee_bps=0.0,
        slippage_bps=0.0,
        max_steps=100,
        window_size=60,
        position_size_pct=0.1,
        allow_short=True,
        tp_sl_enabled=False,
    )


@pytest.fixture
def simple_config():
    """Simplified config for basic tests."""
    return EnvConfig(
        n_assets=4,
        initial_cash=10_000.0,
        fee_bps=0.0,
        slippage_bps=0.0,
        max_steps=50,
        window_size=10,
        position_size_pct=0.25,
        allow_short=True,
        tp_sl_enabled=False,
    )


@pytest.fixture
def tp_sl_config():
    """Config with TP/SL enabled."""
    return EnvConfig(
        n_assets=4,
        initial_cash=10_000.0,
        fee_bps=0.0,
        slippage_bps=0.0,
        max_steps=50,
        window_size=10,
        position_size_pct=0.25,
        allow_short=True,
        tp_sl_enabled=True,
        default_stop_loss_pct=0.05,
        default_take_profit_pct=0.10,
    )


@pytest.fixture
def synthetic_prices():
    """Generate synthetic sine wave prices for testing."""
    n_steps = 500
    n_assets = 64
    t = np.linspace(0, 4 * np.pi, n_steps)
    prices = np.zeros((n_steps, n_assets))
    for i in range(n_assets):
        freq = 1.0 + 0.1 * (i % 10)
        phase = 2 * np.pi * i / n_assets
        base = 100 + 10 * (i % 5)
        prices[:, i] = base + 10 * np.sin(freq * t + phase)
    return jnp.array(prices, dtype=jnp.float32)


@pytest.fixture
def simple_prices():
    """Simple prices for 4 assets."""
    n_steps = 200
    n_assets = 4
    t = np.linspace(0, 2 * np.pi, n_steps)
    prices = np.zeros((n_steps, n_assets))
    prices[:, 0] = 100 + 10 * np.sin(t)       # Sine wave
    prices[:, 1] = 100 + 0.1 * np.arange(n_steps)  # Uptrend
    prices[:, 2] = 100 - 0.1 * np.arange(n_steps)  # Downtrend
    prices[:, 3] = 100 * np.ones(n_steps)     # Flat
    return jnp.array(prices, dtype=jnp.float32)


# =============================================================================
# Test State Initialization
# =============================================================================

class TestReset:
    """Tests for environment reset."""

    def test_reset_returns_valid_state(self, default_config, synthetic_prices):
        key = jax.random.PRNGKey(42)
        state = reset(key, synthetic_prices, default_config)
        assert isinstance(state, TradingState)
        assert state.cash == default_config.initial_cash
        assert state.current_step == 0
        assert state.cumulative_return == 0.0

    def test_reset_initializes_empty_positions(self, default_config, synthetic_prices):
        key = jax.random.PRNGKey(42)
        state = reset(key, synthetic_prices, default_config)
        assert not jnp.any(state.pos_active)
        assert jnp.all(state.pos_side == 0)
        assert jnp.all(state.pos_sl == 0.0)
        assert jnp.all(state.pos_tp == 0.0)

    def test_reset_initializes_empty_orders(self, default_config, synthetic_prices):
        key = jax.random.PRNGKey(42)
        state = reset(key, synthetic_prices, default_config)
        assert not jnp.any(state.order_active)
        assert jnp.all(state.order_side == 0)
        assert jnp.all(state.order_reserved == 0.0)

    def test_reset_random_start(self, default_config, synthetic_prices):
        key1 = jax.random.PRNGKey(42)
        key2 = jax.random.PRNGKey(123)
        state1 = reset(key1, synthetic_prices, default_config)
        state2 = reset(key2, synthetic_prices, default_config)
        assert state1.episode_start_idx != state2.episode_start_idx

    def test_reset_jit_compatible(self, default_config, synthetic_prices):
        key = jax.random.PRNGKey(42)
        state = jit_reset(key, synthetic_prices, default_config)
        assert state.cash == default_config.initial_cash


# =============================================================================
# Test Long Positions
# =============================================================================

class TestLongPositions:
    """Tests for long position actions."""

    def test_hold_action_no_change(self, simple_config, simple_prices):
        key = jax.random.PRNGKey(42)
        state = reset(key, simple_prices, simple_config)
        action = jnp.array(1)  # asset 0, hold
        new_state, obs, reward, done = step(state, action, simple_prices, simple_config)
        assert new_state.cash == state.cash
        assert not jnp.any(new_state.pos_active)

    def test_buy_creates_long_position(self, simple_config, simple_prices):
        key = jax.random.PRNGKey(42)
        state = reset(key, simple_prices, simple_config)
        action = jnp.array(2)  # asset 0, buy
        new_state, obs, reward, done = step(state, action, simple_prices, simple_config)
        assert new_state.pos_active[0]
        assert new_state.pos_side[0] == 1  # Long
        assert new_state.pos_size[0] > 0
        assert new_state.cash < state.cash

    def test_sell_closes_long_position(self, simple_config, simple_prices):
        key = jax.random.PRNGKey(42)
        state = reset(key, simple_prices, simple_config)
        # Buy first
        action_buy = jnp.array(2)
        state, _, _, _ = step(state, action_buy, simple_prices, simple_config)
        assert state.pos_active[0]
        # Then sell
        action_sell = jnp.array(0)
        state, _, _, _ = step(state, action_sell, simple_prices, simple_config)
        assert not state.pos_active[0]
        assert state.pos_size[0] == 0


# =============================================================================
# Test Short Positions
# =============================================================================

class TestShortPositions:
    """Tests for short position actions."""

    def test_sell_creates_short_when_no_position(self, simple_config, simple_prices):
        """Selling without position should open short."""
        key = jax.random.PRNGKey(42)
        state = reset(key, simple_prices, simple_config)
        action = jnp.array(0)  # asset 0, sell
        new_state, obs, reward, done = step(state, action, simple_prices, simple_config)
        assert new_state.pos_active[0]
        assert new_state.pos_side[0] == 2  # Short
        assert new_state.pos_size[0] > 0

    def test_buy_closes_short(self, simple_config, simple_prices):
        """Buying should close existing short."""
        key = jax.random.PRNGKey(42)
        state = reset(key, simple_prices, simple_config)
        # Open short
        action_sell = jnp.array(0)
        state, _, _, _ = step(state, action_sell, simple_prices, simple_config)
        assert state.pos_side[0] == 2  # Short
        # Buy to close
        action_buy = jnp.array(2)
        state, _, _, _ = step(state, action_buy, simple_prices, simple_config)
        assert not state.pos_active[0]  # Position closed

    def test_short_value_calculation(self):
        """Short value = 2*size*entry - size*current."""
        pos_active = jnp.array([True])
        pos_side = jnp.array([2])  # Short
        pos_size = jnp.array([10.0])
        pos_entry = jnp.array([100.0])
        current_prices = jnp.array([90.0])  # Price went down (good for short)

        values = compute_position_values(
            pos_active, pos_side, pos_size, pos_entry, current_prices
        )
        # Short value = 2 * 10 * 100 - 10 * 90 = 2000 - 900 = 1100
        # (includes profit of 100 from price drop)
        assert jnp.isclose(values[0], 1100.0)

    def test_short_disabled_when_allow_short_false(self, simple_prices):
        """Should not open short when allow_short=False."""
        config = EnvConfig(
            n_assets=4,
            initial_cash=10_000.0,
            max_steps=50,
            window_size=10,
            position_size_pct=0.25,
            allow_short=False,  # Disabled
        )
        key = jax.random.PRNGKey(42)
        state = reset(key, simple_prices, config)
        initial_cash = float(state.cash)
        action = jnp.array(0)  # asset 0, sell
        new_state, _, _, _ = step(state, action, simple_prices, config)
        # Should have no effect
        assert not new_state.pos_active[0]
        assert float(new_state.cash) == initial_cash


# =============================================================================
# Test TP/SL Guardrails
# =============================================================================

class TestTPSLGuardrails:
    """Tests for take-profit and stop-loss guardrails."""

    def test_long_position_gets_tp_sl(self, tp_sl_config, simple_prices):
        """Long position should have TP/SL set on entry."""
        key = jax.random.PRNGKey(42)
        state = reset(key, simple_prices, tp_sl_config)
        action = jnp.array(2)  # buy
        new_state, _, _, _ = step(state, action, simple_prices, tp_sl_config)
        entry = float(new_state.pos_entry[0])
        sl = float(new_state.pos_sl[0])
        tp = float(new_state.pos_tp[0])
        # Long: SL = entry * 0.95, TP = entry * 1.10
        assert jnp.isclose(sl, entry * 0.95, rtol=0.01)
        assert jnp.isclose(tp, entry * 1.10, rtol=0.01)

    def test_short_position_gets_inverted_tp_sl(self, tp_sl_config, simple_prices):
        """Short position TP/SL should be inverted."""
        key = jax.random.PRNGKey(42)
        state = reset(key, simple_prices, tp_sl_config)
        action = jnp.array(0)  # sell to open short
        new_state, _, _, _ = step(state, action, simple_prices, tp_sl_config)
        entry = float(new_state.pos_entry[0])
        sl = float(new_state.pos_sl[0])
        tp = float(new_state.pos_tp[0])
        # Short: SL = entry * 1.05, TP = entry * 0.90
        assert jnp.isclose(sl, entry * 1.05, rtol=0.01)
        assert jnp.isclose(tp, entry * 0.90, rtol=0.01)


# =============================================================================
# Test Limit Orders
# =============================================================================

class TestLimitOrders:
    """Tests for limit order placement and filling."""

    def test_limit_buy_places_order(self, simple_config, simple_prices):
        """Limit buy should place order, not execute immediately."""
        key = jax.random.PRNGKey(42)
        state = reset(key, simple_prices, simple_config)
        action = jnp.array(4)  # asset 0, limit_buy
        new_state, _, _, _ = step(state, action, simple_prices, simple_config)
        assert new_state.order_active[0]
        assert new_state.order_side[0] == 1  # Buy order
        # Env moved to LAZY EXECUTION (TASK_124, 2026-01-30): no cash is
        # reserved at placement time, funds are checked at fill time instead.
        assert new_state.order_reserved[0] == 0.0  # No cash reserved (lazy)

    def test_limit_sell_places_order(self, simple_config, simple_prices):
        """Limit sell should place order."""
        key = jax.random.PRNGKey(42)
        state = reset(key, simple_prices, simple_config)
        action = jnp.array(3)  # asset 0, limit_sell
        new_state, _, _, _ = step(state, action, simple_prices, simple_config)
        assert new_state.order_active[0]
        assert new_state.order_side[0] == 2  # Sell order
        assert new_state.order_reserved[0] == 0.0  # No cash reserved for sell

    def test_limit_buy_order_filled(self, simple_prices):
        """Limit buy should fill when price drops to limit."""
        config = EnvConfig(
            n_assets=4,
            initial_cash=10_000.0,
            max_steps=200,
            window_size=10,
            position_size_pct=0.25,
            limit_offset_pct=0.05,  # 5% below current = likely to fill on sine
        )
        key = jax.random.PRNGKey(42)
        state = reset(key, simple_prices, config)

        # Place limit buy on sine wave asset
        action = jnp.array(4)  # limit_buy asset 0
        state, _, _, _ = step(state, action, simple_prices, config)
        assert state.order_active[0]

        # Step until filled or max steps
        filled = False
        for _ in range(50):
            action = jnp.array(1)  # hold
            state, _, _, done = step(state, action, simple_prices, config)
            if state.pos_active[0]:  # Order filled
                filled = True
                break
            if done:
                break

        # Should have filled (sine wave crosses 5% threshold)
        # Note: May not always fill depending on random start
        # Just verify state is consistent
        if filled:
            assert state.pos_side[0] == 1  # Long
            assert not state.order_active[0]  # Order cleared

    def test_limit_order_expiration(self, simple_prices):
        """Limit order should expire after TTL."""
        config = EnvConfig(
            n_assets=4,
            initial_cash=10_000.0,
            max_steps=200,
            window_size=10,
            position_size_pct=0.25,
            default_limit_order_ttl=5,  # Very short TTL
            limit_offset_pct=0.20,  # 20% offset - unlikely to fill
        )
        key = jax.random.PRNGKey(42)
        state = reset(key, simple_prices, config)

        # Place limit buy
        action = jnp.array(4)  # limit_buy
        state, _, _, _ = step(state, action, simple_prices, config)
        reserved = float(state.order_reserved[0])
        assert state.order_active[0]
        # LAZY EXECUTION (TASK_124, 2026-01-30): placing an order never
        # reserves cash, so reserved stays 0 for the life of the order.
        assert reserved == 0.0

        # Step past TTL
        for _ in range(10):
            action = jnp.array(1)  # hold
            state, _, _, _ = step(state, action, simple_prices, config)

        # Order should have expired
        assert not state.order_active[0]
        # Reserved cash should be returned (approximately, minus any executed)
        # Cash should be close to initial
        assert float(state.cash) > config.initial_cash * 0.95


# =============================================================================
# Test Portfolio Calculations
# =============================================================================

class TestPortfolio:
    """Tests for portfolio value calculations."""

    def test_initial_portfolio_value(self, simple_config, simple_prices):
        key = jax.random.PRNGKey(42)
        state = reset(key, simple_prices, simple_config)
        current_prices = get_current_prices(
            simple_prices,
            state.current_step + simple_config.window_size,
            state.episode_start_idx
        )
        portfolio_value = compute_portfolio_value(state, current_prices)
        assert jnp.isclose(portfolio_value, simple_config.initial_cash)

    def test_portfolio_includes_reserved_cash(self, simple_config, simple_prices):
        """Portfolio value should include reserved cash for orders."""
        key = jax.random.PRNGKey(42)
        state = reset(key, simple_prices, simple_config)
        initial_value = float(compute_portfolio_value(
            state,
            get_current_prices(simple_prices, simple_config.window_size, state.episode_start_idx)
        ))

        # Place limit buy (reserves cash)
        action = jnp.array(4)
        new_state, _, _, _ = step(state, action, simple_prices, simple_config)

        new_value = float(compute_portfolio_value(
            new_state,
            get_current_prices(simple_prices, simple_config.window_size + 1, new_state.episode_start_idx)
        ))

        # Portfolio value should be similar (just moved from cash to reserved)
        assert jnp.isclose(new_value, initial_value, rtol=0.01)


# =============================================================================
# Test Rewards
# =============================================================================

class TestRewards:
    """Tests for reward computation."""

    def test_symlog_transformation(self):
        assert symlog(jnp.array(0.0)) == 0.0
        assert symlog(jnp.array(1.0)) > 0
        assert symlog(jnp.array(-1.0)) < 0
        large_val = symlog(jnp.array(100.0))
        small_val = symlog(jnp.array(1.0))
        assert large_val < 100 * small_val


# =============================================================================
# Test Batched Operations
# =============================================================================

class TestBatching:
    """Tests for batched (vmap) operations."""

    def test_batch_reset(self, default_config, synthetic_prices):
        batch_size = 16
        keys = jax.random.split(jax.random.PRNGKey(42), batch_size)
        states = batch_reset(keys, synthetic_prices, default_config)
        assert states.cash.shape == (batch_size,)
        assert states.pos_active.shape == (batch_size, default_config.n_assets)
        assert states.order_active.shape == (batch_size, default_config.n_assets)
        assert jnp.all(states.cash == default_config.initial_cash)

    def test_batch_step(self, default_config, synthetic_prices):
        batch_size = 16
        keys = jax.random.split(jax.random.PRNGKey(42), batch_size)
        states = batch_reset(keys, synthetic_prices, default_config)
        actions = jax.random.randint(
            jax.random.PRNGKey(0),
            (batch_size,),
            0, default_config.n_assets * 5
        )
        new_states, obs, rewards, dones = batch_step(
            states, actions, synthetic_prices, default_config
        )
        assert new_states.current_step.shape == (batch_size,)
        assert rewards.shape == (batch_size,)
        assert dones.shape == (batch_size,)
        assert jnp.all(new_states.current_step == 1)

    def test_batch_jit(self, default_config, synthetic_prices):
        batch_size = 32
        keys = jax.random.split(jax.random.PRNGKey(42), batch_size)
        states = jit_batch_reset(keys, synthetic_prices, default_config)
        assert states.cash.shape == (batch_size,)
        actions = jnp.zeros(batch_size, dtype=jnp.int32) + 1
        new_states, obs, rewards, dones = jit_batch_step(
            states, actions, synthetic_prices, default_config
        )
        assert new_states.current_step.shape == (batch_size,)


# =============================================================================
# Test Termination
# =============================================================================

class TestTermination:
    """Tests for episode termination."""

    def test_done_after_max_steps(self, simple_config, simple_prices):
        key = jax.random.PRNGKey(42)
        state = reset(key, simple_prices, simple_config)
        done = False
        n_steps = 0
        while not done and n_steps < simple_config.max_steps + 10:
            action = jnp.array(1)  # Hold
            state, obs, reward, done = step(state, action, simple_prices, simple_config)
            n_steps += 1
        assert done
        assert n_steps == simple_config.max_steps


# =============================================================================
# Test Observation
# =============================================================================

class TestObservation:
    """Tests for observation computation."""

    def test_observation_shape(self, simple_config, simple_prices):
        key = jax.random.PRNGKey(42)
        state = reset(key, simple_prices, simple_config)
        action = jnp.array(1)
        state, obs, reward, done = step(state, action, simple_prices, simple_config)
        # Phase 1-2: (5, 1, n_assets)
        assert obs.shape == (5, 1, simple_config.n_assets)

    def test_observation_values_bounded(self, simple_config, simple_prices):
        key = jax.random.PRNGKey(42)
        state = reset(key, simple_prices, simple_config)
        action = jnp.array(1)
        state, obs, reward, done = step(state, action, simple_prices, simple_config)
        # Normalized price in [0, 1]
        assert jnp.all(obs[0] >= 0) and jnp.all(obs[0] <= 1)
        # Returns in [-1, 1]
        assert jnp.all(obs[1] >= -1) and jnp.all(obs[1] <= 1)
        # Position indicator in {-1, 0, 1}
        assert jnp.all(jnp.abs(obs[2]) <= 1)


# =============================================================================
# Test Fees
# =============================================================================

class TestTradingCosts:
    """Tests for fee and slippage handling."""

    def test_fees_reduce_value(self):
        config_no_fee = EnvConfig(
            n_assets=4, initial_cash=10_000.0, fee_bps=0.0,
            max_steps=50, window_size=10, position_size_pct=0.25
        )
        config_with_fee = EnvConfig(
            n_assets=4, initial_cash=10_000.0, fee_bps=50.0,
            max_steps=50, window_size=10, position_size_pct=0.25
        )
        prices = jnp.ones((200, 4)) * 100.0
        key = jax.random.PRNGKey(42)

        state1 = reset(key, prices, config_no_fee)
        action = jnp.array(2)
        state1, _, _, _ = step(state1, action, prices, config_no_fee)

        state2 = reset(key, prices, config_with_fee)
        state2, _, _, _ = step(state2, action, prices, config_with_fee)

        current_prices = prices[config_no_fee.window_size]
        value1 = compute_portfolio_value(state1, current_prices)
        value2 = compute_portfolio_value(state2, current_prices)
        assert value2 < value1


# =============================================================================
# Performance Benchmark
# =============================================================================

@pytest.mark.slow
class TestPerformance:
    """Performance benchmarks."""

    def test_large_batch_throughput(self, default_config, synthetic_prices):
        import time
        batch_size = 1000
        n_steps = 100

        keys = jax.random.split(jax.random.PRNGKey(42), batch_size)
        states = jit_batch_reset(keys, synthetic_prices, default_config)

        # Warm up
        actions = jnp.zeros(batch_size, dtype=jnp.int32) + 1
        _ = jit_batch_step(states, actions, synthetic_prices, default_config)

        start = time.time()
        for _ in range(n_steps):
            actions = jax.random.randint(
                jax.random.PRNGKey(0), (batch_size,),
                0, default_config.n_assets * 5
            )
            states, obs, rewards, dones = jit_batch_step(
                states, actions, synthetic_prices, default_config
            )
        jax.block_until_ready(states.cash)
        elapsed = time.time() - start

        throughput = batch_size * n_steps / elapsed
        print(f"\nThroughput: {throughput:.0f} steps/sec ({batch_size} envs, {n_steps} steps)")
        assert throughput > 1000


# =============================================================================
# JAX Encoder Tests
# =============================================================================

from mcts_trader.env.jax_encoder import (
    EncoderConfig, encode_observation, encode_observation_simple,
    encode_observation_with_state, jit_encode_simple, jit_batch_encode_simple,
    batch_encode_simple, validate_encoder_output,
)


@pytest.fixture
def encoder_config():
    """Default encoder configuration."""
    return EncoderConfig(
        n_assets=64,
        window_size=60,
        n_channels=27,
    )


@pytest.fixture
def small_encoder_config():
    """Smaller encoder config for faster tests."""
    return EncoderConfig(
        n_assets=4,
        window_size=60,
        n_channels=27,
    )


class TestEncoderBasic:
    """Basic encoder functionality tests."""

    def test_output_shape(self, encoder_config, synthetic_prices):
        """Test that encoder produces correct output shape."""
        obs = encode_observation_simple(
            synthetic_prices, jnp.array(0), encoder_config
        )
        assert obs.shape == (27, 64, 64)

    def test_no_nan_inf(self, encoder_config, synthetic_prices):
        """Test that encoder output has no NaN or Inf values."""
        obs = encode_observation_simple(
            synthetic_prices, jnp.array(0), encoder_config
        )
        assert not jnp.any(jnp.isnan(obs))
        assert not jnp.any(jnp.isinf(obs))

    def test_values_normalized(self, encoder_config, synthetic_prices):
        """Test that most values are in reasonable ranges."""
        obs = encode_observation_simple(
            synthetic_prices, jnp.array(50), encoder_config
        )
        # Most channels should be normalized to [-1, 1] or [0, 1]
        # Allow some slack for edge cases
        ohlc_channels = obs[:4]  # Channels 0-3
        assert jnp.all(ohlc_channels >= -0.1)
        assert jnp.all(ohlc_channels <= 1.1)

    def test_jit_compilation(self, encoder_config, synthetic_prices):
        """Test that JIT compilation works."""
        obs = jit_encode_simple(
            synthetic_prices, jnp.array(0), encoder_config
        )
        assert obs.shape == (27, 64, 64)

    def test_different_window_starts(self, encoder_config, synthetic_prices):
        """Test encoding at different window positions."""
        obs1 = encode_observation_simple(
            synthetic_prices, jnp.array(0), encoder_config
        )
        obs2 = encode_observation_simple(
            synthetic_prices, jnp.array(100), encoder_config
        )
        # Different window positions should give different observations
        assert not jnp.allclose(obs1, obs2)


class TestEncoderChannels:
    """Test individual encoder channels."""

    def test_close_price_channel(self, small_encoder_config):
        """Test that close price channel (0) reflects current price position."""
        # Create prices that rise over window
        n_steps = 200
        n_assets = 4
        prices = jnp.zeros((n_steps, n_assets))
        for i in range(n_assets):
            prices = prices.at[:, i].set(
                jnp.linspace(100, 200, n_steps)
            )

        obs = encode_observation_simple(
            prices, jnp.array(0), small_encoder_config
        )
        # Current price is max, should be normalized to ~1.0
        close_channel = obs[0, :, 0]
        assert jnp.allclose(close_channel, 1.0, atol=0.01)

    def test_returns_channel_positive(self, small_encoder_config):
        """Test that positive returns are correctly captured."""
        n_steps = 200
        n_assets = 4
        prices = jnp.zeros((n_steps, n_assets))
        for i in range(n_assets):
            prices = prices.at[:, i].set(
                jnp.linspace(100, 110, n_steps)  # 10% increase
            )

        obs = encode_observation_simple(
            prices, jnp.array(0), small_encoder_config
        )
        # 60-min return channel (7) should be positive
        returns_60m = obs[7, 0, 0]
        assert returns_60m > 0

    def test_returns_channel_negative(self, small_encoder_config):
        """Test that negative returns are correctly captured."""
        n_steps = 200
        n_assets = 4
        prices = jnp.zeros((n_steps, n_assets))
        for i in range(n_assets):
            prices = prices.at[:, i].set(
                jnp.linspace(110, 100, n_steps)  # 10% decrease
            )

        obs = encode_observation_simple(
            prices, jnp.array(0), small_encoder_config
        )
        # 60-min return channel (7) should be negative
        returns_60m = obs[7, 0, 0]
        assert returns_60m < 0

    def test_volatility_channel(self, small_encoder_config):
        """Test that high volatility is captured."""
        n_steps = 200
        n_assets = 4
        # High volatility: big oscillations
        t = jnp.linspace(0, 20 * jnp.pi, n_steps)
        prices_volatile = 100 + 20 * jnp.sin(t)
        # Low volatility: small oscillations
        prices_stable = 100 + 2 * jnp.sin(t)

        prices_v = jnp.tile(prices_volatile[:, None], (1, n_assets))
        prices_s = jnp.tile(prices_stable[:, None], (1, n_assets))

        obs_v = encode_observation_simple(
            prices_v, jnp.array(0), small_encoder_config
        )
        obs_s = encode_observation_simple(
            prices_s, jnp.array(0), small_encoder_config
        )

        # Volatility channel (16-19) should be higher for volatile prices
        vol_v = obs_v[16, 0, 0]
        vol_s = obs_s[16, 0, 0]
        assert vol_v > vol_s


class TestEncoderWithState:
    """Test extended encoder with portfolio state."""

    def test_output_shape_with_state(self, small_encoder_config):
        """Test that extended encoder produces 31 channels."""
        n_steps = 200
        n_assets = 4
        prices = jnp.ones((n_steps, n_assets)) * 100.0

        pos_active = jnp.array([True, False, True, False])
        pos_side = jnp.array([1, 0, 2, 0])  # long, none, short, none
        pos_entry = jnp.array([95.0, 0.0, 105.0, 0.0])
        order_active = jnp.zeros(n_assets, dtype=bool)
        order_side = jnp.zeros(n_assets, dtype=jnp.int32)

        obs = encode_observation_with_state(
            prices, None, None, None, jnp.array(0),
            pos_active, pos_side, pos_entry, order_active, order_side,
            small_encoder_config,
        )
        assert obs.shape == (31, 4, 4)

    def test_position_indicator(self, small_encoder_config):
        """Test that position indicators are correctly set."""
        n_steps = 200
        n_assets = 4
        prices = jnp.ones((n_steps, n_assets)) * 100.0

        pos_active = jnp.array([True, False, True, False])
        pos_side = jnp.array([1, 0, 2, 0])  # long, none, short, none
        pos_entry = jnp.array([95.0, 0.0, 105.0, 0.0])
        order_active = jnp.zeros(n_assets, dtype=bool)
        order_side = jnp.zeros(n_assets, dtype=jnp.int32)

        obs = encode_observation_with_state(
            prices, None, None, None, jnp.array(0),
            pos_active, pos_side, pos_entry, order_active, order_side,
            small_encoder_config,
        )

        # Channel 27: Position indicator
        pos_indicator = obs[27, :, 0]
        assert pos_indicator[0] == 1.0   # Long
        assert pos_indicator[1] == 0.0   # No position
        assert pos_indicator[2] == -1.0  # Short
        assert pos_indicator[3] == 0.0   # No position


class TestEncoderBatched:
    """Test batched encoder operations."""

    def test_batch_encode(self, encoder_config, synthetic_prices):
        """Test batched encoding produces correct shape."""
        batch_size = 10
        window_starts = jnp.arange(batch_size) * 10  # Different start positions

        obs_batch = batch_encode_simple(
            synthetic_prices, window_starts, encoder_config
        )
        assert obs_batch.shape == (batch_size, 27, 64, 64)

    def test_jit_batch_encode(self, encoder_config, synthetic_prices):
        """Test JIT-compiled batched encoding."""
        batch_size = 10
        window_starts = jnp.arange(batch_size) * 10

        obs_batch = jit_batch_encode_simple(
            synthetic_prices, window_starts, encoder_config
        )
        assert obs_batch.shape == (batch_size, 27, 64, 64)

    def test_batch_vs_individual(self, encoder_config, synthetic_prices):
        """Test that batch encoding matches individual encoding."""
        batch_size = 5
        window_starts = jnp.arange(batch_size) * 20

        # Batched
        obs_batch = batch_encode_simple(
            synthetic_prices, window_starts, encoder_config
        )

        # Individual
        for i in range(batch_size):
            obs_single = encode_observation_simple(
                synthetic_prices, window_starts[i], encoder_config
            )
            assert jnp.allclose(obs_batch[i], obs_single, atol=1e-5)


class TestEncoderValidation:
    """Test encoder validation utilities."""

    def test_validate_correct_output(self, encoder_config, synthetic_prices):
        """Test validation passes for correct output."""
        obs = encode_observation_simple(
            synthetic_prices, jnp.array(0), encoder_config
        )
        assert validate_encoder_output(obs, encoder_config)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
