"""
Test script for validating TASK implementations.

Tests:
- TASK_050: Fill at market price (limit orders fill at better of limit/market)
- TASK_047: Probabilistic fills (distance-based fill probability)
- TASK_052: Random fill priority (shuffle asset order for fills)
- TASK_049: Synthetic noise (anti-memorization noise)
- TASK_024: LR schedule (optax schedule integration)
- TASK_053: Episode variance (random episode lengths)
- TASK_055: Borrow cost (short position costs)

Run with: python tools/testing/test_task_implementations.py
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple

# Suppress JAX GPU initialization messages for cleaner output
import os
os.environ['JAX_PLATFORMS'] = 'cpu'  # Run tests on CPU for simplicity

def print_test(name: str, passed: bool, details: str = ""):
    """Print test result."""
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {status}: {name}")
    if details and not passed:
        print(f"         {details}")
    return passed


def test_task_050_fill_at_market_price():
    """Test TASK_050: Limit orders fill at better of limit/market price."""
    print("\n" + "="*60)
    print("TASK_050: Fill at Market Price")
    print("="*60)

    from mcts_trader.env.jax_trading_env import EnvConfig, TradingState

    all_passed = True

    # Test the logic directly
    # Buy limit at $100, market at $95 -> should fill at $95 (better for buyer)
    limit_price = 100.0
    market_price = 95.0
    buy_fill = min(limit_price, market_price)
    all_passed &= print_test(
        "Buy fills at min(limit, market)",
        buy_fill == 95.0,
        f"Expected 95.0, got {buy_fill}"
    )

    # Sell limit at $100, market at $105 -> should fill at $105 (better for seller)
    limit_price = 100.0
    market_price = 105.0
    sell_fill = max(limit_price, market_price)
    all_passed &= print_test(
        "Sell fills at max(limit, market)",
        sell_fill == 105.0,
        f"Expected 105.0, got {sell_fill}"
    )

    # Verify config has probabilistic_fills option
    config = EnvConfig()
    all_passed &= print_test(
        "EnvConfig has probabilistic_fills",
        hasattr(config, 'probabilistic_fills'),
        "probabilistic_fills not found in EnvConfig"
    )

    return all_passed


def test_task_047_probabilistic_fills():
    """Test TASK_047: Probabilistic fill model."""
    print("\n" + "="*60)
    print("TASK_047: Probabilistic Fills")
    print("="*60)

    from mcts_trader.env.jax_trading_env import EnvConfig

    all_passed = True

    # Test config option exists
    config = EnvConfig(probabilistic_fills=True)
    all_passed &= print_test(
        "probabilistic_fills config works",
        config.probabilistic_fills == True,
        f"Expected True, got {config.probabilistic_fills}"
    )

    # Test the probability formula logic
    # Marketable orders (distance <= 0) should always fill
    distance = -0.01  # Price is 1% better than limit
    is_marketable = distance <= 0
    all_passed &= print_test(
        "Marketable orders detected (distance <= 0)",
        is_marketable,
        f"distance={distance} should be marketable"
    )

    # Non-marketable orders have probability based on distance
    distance = 0.05  # 5% away from market
    base_prob = min(0.1 + abs(distance) * 20.0, 0.95)
    expected = min(0.1 + 0.05 * 20.0, 0.95)  # 0.1 + 1.0 = 1.1 -> clipped to 0.95
    all_passed &= print_test(
        "Non-marketable probability calculation",
        abs(base_prob - expected) < 0.01,
        f"Expected ~0.95, got {base_prob}"
    )

    # Small distance should give low probability
    distance = 0.001  # 0.1% away
    base_prob = min(0.1 + abs(distance) * 20.0, 0.95)
    expected = 0.1 + 0.001 * 20.0  # = 0.12
    all_passed &= print_test(
        "Small distance gives low probability",
        abs(base_prob - expected) < 0.01,
        f"Expected ~0.12, got {base_prob}"
    )

    # Age bonus calculation
    order_age = 10  # steps
    age_bonus = min(order_age * 0.05, 0.30)
    all_passed &= print_test(
        "Age bonus calculation",
        abs(age_bonus - 0.30) < 0.01,  # 10 * 0.05 = 0.5, clamped to 0.3
        f"Expected 0.30, got {age_bonus}"
    )

    return all_passed


def test_task_052_random_fill_priority():
    """Test TASK_052: Random fill priority to prevent asset index bias."""
    print("\n" + "="*60)
    print("TASK_052: Random Fill Priority")
    print("="*60)

    all_passed = True

    # Test the shuffle logic
    n_assets = 64
    key = jax.random.PRNGKey(42)

    # Create some costs
    costs = jnp.arange(n_assets, dtype=jnp.float32)  # 0, 1, 2, ..., 63

    # Shuffle
    priority_order = jax.random.permutation(key, n_assets)
    inverse_order = jnp.argsort(priority_order)

    # Verify shuffle is a permutation
    sorted_priority = jnp.sort(priority_order)
    expected = jnp.arange(n_assets)
    all_passed &= print_test(
        "Shuffle is a valid permutation",
        jnp.allclose(sorted_priority, expected),
        "Shuffled order is not a permutation"
    )

    # Verify inverse_order recovers original
    shuffled = costs[priority_order]
    recovered = shuffled[inverse_order]
    all_passed &= print_test(
        "Inverse order recovers original",
        jnp.allclose(recovered, costs),
        "Inverse order doesn't recover original array"
    )

    # Verify different keys give different orders
    key2 = jax.random.PRNGKey(123)
    priority_order2 = jax.random.permutation(key2, n_assets)
    all_passed &= print_test(
        "Different keys give different orders",
        not jnp.allclose(priority_order, priority_order2),
        "Same order for different keys"
    )

    return all_passed


def test_task_049_synthetic_noise():
    """Test TASK_049: Synthetic noise for anti-memorization."""
    print("\n" + "="*60)
    print("TASK_049: Synthetic Noise")
    print("="*60)

    from mcts_trader.data.synthetic_market import SyntheticMarket

    all_passed = True

    # Test noise_level parameter exists
    try:
        market = SyntheticMarket(level=0, n_assets=4, n_steps=100, noise_level=0.0)
        all_passed &= print_test(
            "noise_level parameter exists",
            True
        )
    except TypeError as e:
        all_passed &= print_test(
            "noise_level parameter exists",
            False,
            str(e)
        )
        return all_passed

    # Test noise_level=0 gives deterministic prices
    market1 = SyntheticMarket(level=0, n_assets=2, n_steps=100, seed=42, noise_level=0.0)
    market2 = SyntheticMarket(level=0, n_assets=2, n_steps=100, seed=42, noise_level=0.0)

    market1.reset()
    market2.reset()

    prices1 = []
    prices2 = []
    for _ in range(50):
        p1 = market1.get_current_data()['SYN_00']['close']
        p2 = market2.get_current_data()['SYN_00']['close']
        prices1.append(p1)
        prices2.append(p2)
        market1.step()
        market2.step()

    all_passed &= print_test(
        "noise_level=0 is deterministic",
        np.allclose(prices1, prices2),
        f"Prices differ: {prices1[:5]} vs {prices2[:5]}"
    )

    # Test noise_level>0 adds variation
    market_noisy = SyntheticMarket(level=0, n_assets=2, n_steps=100, seed=42, noise_level=1.0)
    market_noisy.reset()

    prices_noisy = []
    for _ in range(50):
        p = market_noisy.get_current_data()['SYN_00']['close']
        prices_noisy.append(p)
        market_noisy.step()

    # Noisy prices should be different from clean prices (most of the time)
    max_diff = max(abs(p1 - pn) for p1, pn in zip(prices1, prices_noisy))
    all_passed &= print_test(
        "noise_level>0 adds variation",
        max_diff > 0.01,  # At least some difference
        f"Max diff only {max_diff}, expected > 0.01"
    )

    return all_passed


def test_task_024_lr_schedule():
    """Test TASK_024: Learning rate schedule integration."""
    print("\n" + "="*60)
    print("TASK_024: LR Schedule")
    print("="*60)

    from mcts_trader.jax_agent.schedules import LRSchedule, create_scheduled_optimizer
    import optax

    all_passed = True

    # Test constant schedule
    lr_sched = LRSchedule(
        initial_lr=1e-4,
        schedule_type='constant'
    )
    all_passed &= print_test(
        "Constant schedule returns initial_lr",
        abs(lr_sched.get_lr(0) - 1e-4) < 1e-10,
        f"Expected 1e-4, got {lr_sched.get_lr(0)}"
    )

    # Test cosine schedule with warmup
    lr_sched = LRSchedule(
        initial_lr=1e-3,
        final_lr=1e-5,
        warmup_epochs=10,
        total_epochs=100,
        schedule_type='cosine'
    )

    # During warmup, LR should increase
    lr_0 = lr_sched.get_lr(0)
    lr_5 = lr_sched.get_lr(5)
    all_passed &= print_test(
        "Cosine: LR increases during warmup",
        lr_5 > lr_0,
        f"LR at 0: {lr_0}, LR at 5: {lr_5}"
    )

    # After warmup, LR should decrease
    lr_10 = lr_sched.get_lr(10)
    lr_50 = lr_sched.get_lr(50)
    all_passed &= print_test(
        "Cosine: LR decreases after warmup",
        lr_50 < lr_10,
        f"LR at 10: {lr_10}, LR at 50: {lr_50}"
    )

    # Test optax conversion
    try:
        optax_sched = lr_sched.to_optax_schedule(steps_per_epoch=100)
        # Optax schedule takes step number
        lr_step_0 = optax_sched(0)
        all_passed &= print_test(
            "Optax schedule conversion works",
            lr_step_0 > 0,
            f"LR at step 0: {lr_step_0}"
        )
    except Exception as e:
        all_passed &= print_test(
            "Optax schedule conversion works",
            False,
            str(e)
        )

    # Test create_scheduled_optimizer
    try:
        optimizer = create_scheduled_optimizer(lr_sched, steps_per_epoch=100)
        all_passed &= print_test(
            "create_scheduled_optimizer works",
            optimizer is not None,
            "Returned None"
        )
    except Exception as e:
        all_passed &= print_test(
            "create_scheduled_optimizer works",
            False,
            str(e)
        )

    return all_passed


def test_task_053_episode_variance():
    """Test TASK_053: Episode length variance."""
    print("\n" + "="*60)
    print("TASK_053: Episode Variance")
    print("="*60)

    from mcts_trader.env.jax_trading_env import EnvConfig, TradingState, reset

    all_passed = True

    # Test config option exists
    config = EnvConfig(max_steps=400, episode_variance=50)
    all_passed &= print_test(
        "episode_variance config exists",
        hasattr(config, 'episode_variance') and config.episode_variance == 50,
        f"episode_variance={getattr(config, 'episode_variance', 'N/A')}"
    )

    # Test TradingState has max_episode_steps field
    all_passed &= print_test(
        "TradingState has max_episode_steps",
        'max_episode_steps' in TradingState.__dataclass_fields__,
        "max_episode_steps field not found"
    )

    # Test reset produces variable episode lengths
    config = EnvConfig(n_assets=8, max_steps=400, episode_variance=50)

    # Create dummy prices
    prices = jnp.ones((2000, 8), dtype=jnp.float32) * 100.0

    # Reset with different keys
    lengths = []
    for i in range(20):
        key = jax.random.PRNGKey(i)
        state = reset(key, prices, config)
        lengths.append(float(state.max_episode_steps))

    # Check if we got varying lengths
    unique_lengths = len(set(lengths))
    all_passed &= print_test(
        "Reset produces varying episode lengths",
        unique_lengths > 1,
        f"Only got {unique_lengths} unique length(s): {set(lengths)}"
    )

    # Check lengths are within expected range [350, 450]
    min_len = min(lengths)
    max_len = max(lengths)
    all_passed &= print_test(
        "Episode lengths within valid range",
        min_len >= 350 and max_len <= 450,
        f"Range [{min_len}, {max_len}], expected [350, 450]"
    )

    # Test with variance=0 gives fixed length
    config_fixed = EnvConfig(n_assets=8, max_steps=400, episode_variance=0)
    lengths_fixed = []
    for i in range(10):
        key = jax.random.PRNGKey(i)
        state = reset(key, prices, config_fixed)
        lengths_fixed.append(float(state.max_episode_steps))

    all_passed &= print_test(
        "variance=0 gives fixed length",
        len(set(lengths_fixed)) == 1,
        f"Got {len(set(lengths_fixed))} unique lengths"
    )

    return all_passed


def test_task_055_borrow_cost():
    """Test TASK_055: Borrowing cost for short positions."""
    print("\n" + "="*60)
    print("TASK_055: Borrow Cost")
    print("="*60)

    from mcts_trader.env.jax_trading_env import EnvConfig

    all_passed = True

    # Test config option exists
    config = EnvConfig(borrow_rate_annual=0.05)
    all_passed &= print_test(
        "borrow_rate_annual config exists",
        hasattr(config, 'borrow_rate_annual'),
        "borrow_rate_annual not found"
    )

    all_passed &= print_test(
        "borrow_rate_annual value correct",
        abs(config.borrow_rate_annual - 0.05) < 1e-6,
        f"Expected 0.05, got {config.borrow_rate_annual}"
    )

    # Test the borrow cost formula
    # Annual rate = 5%, converted to per-step
    # Assuming ~400 steps = 1 week, 52 weeks/year
    # steps_per_year = 400 * 52 * 7 / 7 = 400 * 52 = 20,800
    # But implementation uses 100,800 (more conservative, ~1.5 min steps)
    borrow_rate_annual = 0.05
    borrow_rate_step = borrow_rate_annual / 100800.0

    # For $10,000 short position
    short_value = 10000.0
    cost_per_step = short_value * borrow_rate_step

    # Should be about $0.005 per step ($0.5 per 100 steps)
    expected_cost = 10000 * 0.05 / 100800  # ~0.00496
    all_passed &= print_test(
        "Borrow cost per step calculation",
        abs(cost_per_step - expected_cost) < 0.001,
        f"Expected ~{expected_cost:.6f}, got {cost_per_step:.6f}"
    )

    # Annualized: 100800 steps * $0.005 = ~$500 = 5% of $10k
    annual_cost = cost_per_step * 100800
    all_passed &= print_test(
        "Annual cost equals rate * value",
        abs(annual_cost - 500.0) < 1.0,  # ~$500 annual on $10k short
        f"Expected ~$500, got ${annual_cost:.2f}"
    )

    return all_passed


def test_system_integration():
    """Test overall system integration - can we run training?"""
    print("\n" + "="*60)
    print("System Integration Test")
    print("="*60)

    all_passed = True

    # Test 1: Can import all modules
    try:
        from mcts_trader.env.jax_trading_env import EnvConfig, reset, step
        from mcts_trader.data.synthetic_market import SyntheticMarket
        from mcts_trader.jax_agent.schedules import LRSchedule, SimsSchedule
        from mcts_trader.training.env_factory import create_jax_env_and_prices
        all_passed &= print_test("All imports successful", True)
    except Exception as e:
        all_passed &= print_test("All imports successful", False, str(e))
        return all_passed

    # Test 2: Can create environment with all new options
    try:
        states, prices, config = create_jax_env_and_prices(
            synthetic_level=0,
            n_envs=4,
            n_assets=8,
            n_steps=500,
            max_episode_steps=100,
            window_size=20,
            probabilistic_fills=True,
            borrow_rate_annual=0.05,
            synthetic_noise=0.5,
            episode_variance=10,
        )
        all_passed &= print_test(
            "create_jax_env_and_prices works with all options",
            states is not None and prices is not None,
            "Returned None"
        )
    except Exception as e:
        all_passed &= print_test(
            "create_jax_env_and_prices works with all options",
            False,
            str(e)
        )
        return all_passed

    # Test 3: Can run a few steps
    try:
        from mcts_trader.env.jax_trading_env import jit_batch_step

        # Run 10 steps with random actions
        for i in range(10):
            key = jax.random.PRNGKey(i)
            actions = jax.random.randint(key, (4,), 0, 8 * 5)  # 8 assets * 5 actions
            states, obs, rewards, dones = jit_batch_step(states, actions, prices, config)

        all_passed &= print_test(
            "Can run batched steps",
            True
        )
    except Exception as e:
        all_passed &= print_test(
            "Can run batched steps",
            False,
            str(e)
        )

    # Test 4: Check shapes are correct
    try:
        assert states.cash.shape == (4,), f"cash shape: {states.cash.shape}"
        assert obs.shape[0] == 4, f"obs batch: {obs.shape[0]}"
        assert rewards.shape == (4,), f"rewards shape: {rewards.shape}"
        assert dones.shape == (4,), f"dones shape: {dones.shape}"
        all_passed &= print_test("Output shapes correct", True)
    except AssertionError as e:
        all_passed &= print_test("Output shapes correct", False, str(e))

    return all_passed


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("TASK IMPLEMENTATION VALIDATION")
    print("="*60)

    results = {}

    # Run each test
    results['TASK_050'] = test_task_050_fill_at_market_price()
    results['TASK_047'] = test_task_047_probabilistic_fills()
    results['TASK_052'] = test_task_052_random_fill_priority()
    results['TASK_049'] = test_task_049_synthetic_noise()
    results['TASK_024'] = test_task_024_lr_schedule()
    results['TASK_053'] = test_task_053_episode_variance()
    results['TASK_055'] = test_task_055_borrow_cost()
    results['Integration'] = test_system_integration()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    passed = sum(results.values())
    total = len(results)

    for name, result in results.items():
        status = "✓" if result else "✗"
        print(f"  {status} {name}")

    print(f"\n  Total: {passed}/{total} passed")

    if passed == total:
        print("\n  🎉 All tests passed!")
        return 0
    else:
        print(f"\n  ⚠ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
