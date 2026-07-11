"""
Environment factory for MuZero training.

This module provides factory functions for creating trading environments,
both JAX-based (GPU-accelerated) and VectorBT-based (Python).
"""
import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Optional, List, Any


def create_jax_env_and_prices(
    synthetic_level: int,
    n_envs: int,
    n_assets: int = 64,
    n_steps: int = 2000,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
    max_episode_steps: int = 400,
    window_size: int = 60,
    position_size_pct: float = 0.1,
    rng_key: Optional[jnp.ndarray] = None,
    # TASK_047/052: Probabilistic fills and random priority
    probabilistic_fills: bool = False,
    # TASK_055: Borrowing cost for shorts
    borrow_rate_annual: float = 0.0,
    # TASK_049: Synthetic noise for anti-memorization
    synthetic_noise: float = 0.0,
    # TASK_053: Episode length variance
    episode_variance: int = 0,
    # TASK_058: Multi-modal noise
    temporal_jitter: int = 0,
    time_dilation: float = 0.0,
    phase_shift_prob: float = 0.0,
    # TASK_051: Burning money (cash opportunity cost)
    burning_money: bool = False,
    burn_market_linked: bool = True,
    burn_base_rate: float = 0.00005,
    # Random burns (wife with debit card extreme mode)
    burn_random: bool = False,
    burn_random_prob: float = 0.02,
    burn_random_max: float = 500.0,
    burn_random_limit: int = 10,
    # TASK_818 Tier B: realized-PnL-on-close attributable reward (default OFF = canon)
    reward_attributable: bool = False,
    # TASK_821: seed for the SyntheticMarket price generator. None (default) = current
    # behavior (unseeded RandomState → OS entropy). Set only when --seed is given, so
    # two runs with the same --seed generate identical synthetic prices.
    price_seed: Optional[int] = None,
    # TASK_828 Stage 3: vector-action env step (default OFF = canon scalar path).
    vector_action: bool = False,
    # TASK_839: budget-constrained vector-step sizing (default OFF = canon).
    vector_budget_step: bool = False,
    vector_budget_pct: float = 0.99,
    # TASK_840: target-weight rebalance step (default OFF = canon).
    vector_weight_rebalance: bool = False,
    vector_weight_unit: float = -1.0,
    vector_weight_max: float = 4.0,
    vector_rebalance_deadband: float = -1.0,
) -> Tuple[Any, jnp.ndarray, Any]:
    """Create JAX environment states and synthetic prices.

    This factory creates the batched environment states and price data needed
    for fully GPU-accelerated self-play using run_self_play_jax.

    Args:
        synthetic_level: Synthetic curriculum level (0-4)
            0 = sine waves, 1 = lagged, 2 = dip, 3 = trends, 4 = rotation
        n_envs: Number of parallel environments
        n_assets: Number of assets (default 64)
        n_steps: Total price steps to generate (should be > max_episode_steps * 2)
        fee_bps: Trading fee in basis points
        slippage_bps: Slippage in basis points
        max_episode_steps: Max steps per episode
        window_size: Observation window size
        position_size_pct: Position size as fraction of portfolio
        rng_key: JAX RNG key (optional, uses time-based seed if None)

    Returns:
        Tuple of:
            - states: Batched TradingState from jit_batch_reset
            - prices: Price array (n_steps, n_assets) as JAX array
            - config: EnvConfig for the JAX environment
    """
    from mcts_trader.env.jax_trading_env import (
        EnvConfig, jit_batch_reset,
    )
    from mcts_trader.data.synthetic_market import SyntheticMarket

    # TASK_124: Use time-based seed to prevent memorization
    if rng_key is None:
        import time
        rng_key = jax.random.PRNGKey(int(time.time() * 1000) % (2**31))

    # Generate synthetic prices using existing generator
    synthetic = SyntheticMarket(
        level=synthetic_level,
        n_assets=n_assets,
        n_steps=n_steps,
        # TASK_049: Noise for anti-memorization
        noise_level=synthetic_noise,
        # TASK_058: Multi-modal noise
        temporal_jitter=temporal_jitter,
        time_dilation=time_dilation,
        phase_shift_prob=phase_shift_prob,
        # TASK_821: None (default) = unseeded RandomState, bit-identical to before
        seed=price_seed,
    )
    synthetic.reset()

    # Collect prices
    prices_np = np.zeros((n_steps, n_assets), dtype=np.float32)
    for i in range(n_steps):
        data = synthetic.get_current_data()
        for j in range(n_assets):
            symbol = f"SYN_{j:02d}"
            if symbol in data:
                prices_np[i, j] = data[symbol]['close']
        synthetic.step()

    prices = jnp.array(prices_np, dtype=jnp.float32)

    # Create JAX env config
    config = EnvConfig(
        n_assets=n_assets,
        initial_cash=100_000.0,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        max_steps=max_episode_steps,
        window_size=window_size,
        position_size_pct=position_size_pct,
        allow_short=True,
        tp_sl_enabled=False,
        # TASK_047/052: Market realism
        probabilistic_fills=probabilistic_fills,
        # TASK_055: Short position costs
        borrow_rate_annual=borrow_rate_annual,
        # TASK_053: Episode length variance
        episode_variance=episode_variance,
        # TASK_051: Burning money (cash opportunity cost)
        burning_money=burning_money,
        burn_market_linked=burn_market_linked,
        burn_base_rate=burn_base_rate,
        # Random burns
        burn_random=burn_random,
        burn_random_prob=burn_random_prob,
        burn_random_max=burn_random_max,
        burn_random_limit=burn_random_limit,
        # TASK_818 Tier B
        reward_attributable=reward_attributable,
        # TASK_828 Stage 3
        vector_action=vector_action,
        # TASK_839: budget-constrained vector-step sizing (default OFF = canon)
        vector_budget_step=vector_budget_step,
        vector_budget_pct=vector_budget_pct,
        # TASK_840: target-weight rebalance step (default OFF = canon)
        vector_weight_rebalance=vector_weight_rebalance,
        vector_weight_unit=vector_weight_unit,
        vector_weight_max=vector_weight_max,
        vector_rebalance_deadband=vector_rebalance_deadband,
    )

    # Initialize batched environment states
    keys = jax.random.split(rng_key, n_envs)
    states = jit_batch_reset(keys, prices, config)

    return states, prices, config


def create_jax_env_unique_prices(
    synthetic_level: int,
    n_envs: int,
    n_assets: int = 64,
    n_steps: int = 500,  # Per-env steps (episode length + buffer)
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
    max_episode_steps: int = 400,
    window_size: int = 60,
    position_size_pct: float = 0.1,
    rng_key: Optional[jnp.ndarray] = None,
    probabilistic_fills: bool = False,
    borrow_rate_annual: float = 0.0,
    synthetic_noise: float = 0.0,
    episode_variance: int = 0,
    temporal_jitter: int = 0,
    time_dilation: float = 0.0,
    phase_shift_prob: float = 0.0,
    burning_money: bool = False,
    burn_market_linked: bool = True,
    burn_base_rate: float = 0.00005,
    burn_random: bool = False,
    burn_random_prob: float = 0.02,
    burn_random_max: float = 500.0,
    burn_random_limit: int = 10,
    # TASK_818 Tier B: realized-PnL-on-close attributable reward (default OFF = canon)
    reward_attributable: bool = False,
    # TASK_828 Stage 3: vector-action env step (default OFF = canon scalar path).
    vector_action: bool = False,
    # TASK_839: budget-constrained vector-step sizing (default OFF = canon).
    vector_budget_step: bool = False,
    vector_budget_pct: float = 0.99,
    # TASK_840: target-weight rebalance step (default OFF = canon).
    vector_weight_rebalance: bool = False,
    vector_weight_unit: float = -1.0,
    vector_weight_max: float = 4.0,
    vector_rebalance_deadband: float = -1.0,
    n_workers: int = 8,
) -> Tuple[Any, jnp.ndarray, Any]:
    """Create JAX env with UNIQUE prices per env (multithreaded generation).

    Each environment gets its own independently generated price data,
    maximizing diversity for training.

    Args:
        synthetic_level: Synthetic curriculum level (0-14)
        n_envs: Number of parallel environments
        n_assets: Number of assets (default 64)
        n_steps: Steps per env (should be > max_episode_steps + window_size)
        n_workers: Number of threads for parallel generation

    Returns:
        Tuple of:
            - states: Batched TradingState
            - prices: (n_envs, n_steps, n_assets) JAX array - unique per env
            - config: EnvConfig
    """
    from mcts_trader.env.jax_trading_env import (
        EnvConfig, jit_batch_reset_unique,
    )
    from mcts_trader.data.synthetic_market import SyntheticMarket
    from concurrent.futures import ThreadPoolExecutor
    import threading

    # TASK_124: Use time-based seed to prevent memorization
    if rng_key is None:
        import time
        rng_key = jax.random.PRNGKey(int(time.time() * 1000) % (2**31))

    # Thread-safe price generation for one env
    def generate_single_env(env_idx: int, seed: int) -> np.ndarray:
        """Generate price data for a single env."""
        synthetic = SyntheticMarket(
            level=synthetic_level,
            n_assets=n_assets,
            n_steps=n_steps,
            noise_level=synthetic_noise,
            temporal_jitter=temporal_jitter,
            time_dilation=time_dilation,
            phase_shift_prob=phase_shift_prob,
            seed=seed,  # Unique seed per env
        )
        synthetic.reset()

        prices_np = np.zeros((n_steps, n_assets), dtype=np.float32)
        for i in range(n_steps):
            data = synthetic.get_current_data()
            for j in range(n_assets):
                symbol = f"SYN_{j:02d}"
                if symbol in data:
                    prices_np[i, j] = data[symbol]['close']
            synthetic.step()

        return prices_np

    # Generate unique seeds for each env
    seeds = [int(jax.random.randint(jax.random.fold_in(rng_key, i), (), 0, 2**31)) for i in range(n_envs)]

    # Parallel generation
    all_prices = np.zeros((n_envs, n_steps, n_assets), dtype=np.float32)
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(generate_single_env, i, seeds[i]): i for i in range(n_envs)}
        for future in futures:
            env_idx = futures[future]
            all_prices[env_idx] = future.result()

    prices = jnp.array(all_prices, dtype=jnp.float32)

    # Create JAX env config
    config = EnvConfig(
        n_assets=n_assets,
        initial_cash=100_000.0,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        max_steps=max_episode_steps,
        window_size=window_size,
        position_size_pct=position_size_pct,
        allow_short=True,
        tp_sl_enabled=False,
        probabilistic_fills=probabilistic_fills,
        borrow_rate_annual=borrow_rate_annual,
        episode_variance=episode_variance,
        burning_money=burning_money,
        burn_market_linked=burn_market_linked,
        burn_base_rate=burn_base_rate,
        burn_random=burn_random,
        burn_random_prob=burn_random_prob,
        burn_random_max=burn_random_max,
        burn_random_limit=burn_random_limit,
        # TASK_818 Tier B
        reward_attributable=reward_attributable,
        # TASK_828 Stage 3
        vector_action=vector_action,
        # TASK_839: budget-constrained vector-step sizing (default OFF = canon)
        vector_budget_step=vector_budget_step,
        vector_budget_pct=vector_budget_pct,
        # TASK_840: target-weight rebalance step (default OFF = canon)
        vector_weight_rebalance=vector_weight_rebalance,
        vector_weight_unit=vector_weight_unit,
        vector_weight_max=vector_weight_max,
        vector_rebalance_deadband=vector_rebalance_deadband,
    )

    # Initialize batched environment states with unique prices
    keys = jax.random.split(rng_key, n_envs)
    states = jit_batch_reset_unique(keys, prices, config)

    return states, prices, config


def create_jax_env_historical(
    db_path: str,
    start_date: str,
    end_date: str,
    n_envs: int,
    n_assets: int = 64,
    specific_symbols: Optional[List[str]] = None,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
    max_episode_steps: int = 400,
    position_size_pct: float = 0.1,
    rng_key: Optional[jax.random.PRNGKey] = None,
    probabilistic_fills: bool = False,
    borrow_rate_annual: float = 0.0,
    burning_money: bool = False,
    burn_market_linked: bool = True,
    burn_base_rate: float = 0.00005,
    burn_random: bool = False,
    burn_random_prob: float = 0.02,
    burn_random_max: float = 500.0,
    burn_random_limit: int = 10,
    # TASK_818 Tier B: realized-PnL-on-close attributable reward (default OFF = canon)
    reward_attributable: bool = False,
    # TASK_818-freq: resample 1-min closes to K-minute bars (1 = canon 1-min)
    bar_minutes: int = 1,
    # TASK_828 Stage 3: vector-action env step (default OFF = canon scalar path).
    vector_action: bool = False,
    # TASK_839: budget-constrained vector-step sizing (default OFF = canon).
    vector_budget_step: bool = False,
    vector_budget_pct: float = 0.99,
    # TASK_840: target-weight rebalance step (default OFF = canon).
    vector_weight_rebalance: bool = False,
    vector_weight_unit: float = -1.0,
    vector_weight_max: float = 4.0,
    vector_rebalance_deadband: float = -1.0,
) -> Tuple[Any, jnp.ndarray, Any]:
    """Create JAX environment with HISTORICAL prices from database.

    Loads prices from the trading database and creates a fully GPU-accelerated
    JAX environment for historical data training.

    Args:
        db_path: Path to trading database
        start_date: Start date string (YYYY-MM-DD or timestamp)
        end_date: End date string (YYYY-MM-DD or timestamp)
        n_envs: Number of parallel environments
        n_assets: Number of assets (will pad/truncate to match)
        specific_symbols: Optional list of symbols to load
        fee_bps: Trading fee in basis points
        slippage_bps: Slippage in basis points
        max_episode_steps: Maximum steps per episode
        position_size_pct: Position size as fraction of portfolio
        rng_key: JAX RNG key (optional)
        probabilistic_fills: Whether to use probabilistic fill model
        borrow_rate_annual: Annual borrow rate for margin
        burning_money: Enable cash burn mechanic
        burn_*: Burn configuration options

    Returns:
        Tuple of:
            - states: Batched TradingState from jit_batch_reset
            - prices: Price array (n_steps, n_assets) as JAX array
            - config: EnvConfig for the JAX environment
    """
    from mcts_trader.env.jax_trading_env import (
        EnvConfig, jit_batch_reset,
    )
    from mcts_trader.env.vectorbt_env import VectorBTTradingEnv

    # TASK_124: Use time-based seed to prevent memorization
    if rng_key is None:
        import time
        rng_key = jax.random.PRNGKey(int(time.time() * 1000) % (2**31))

    # Load historical prices using VectorBTTradingEnv (handles DB loading)
    temp_env = VectorBTTradingEnv(
        db_path=db_path,
        start_date=start_date,
        end_date=end_date,
        n_assets=n_assets,
        specific_symbols=specific_symbols,
        initial_cash=10000.0,
        max_steps=100,  # Doesn't matter, just for loading
    )
    prices_np = temp_env.prices  # (n_steps, n_loaded_assets)

    # TASK_818-freq: resample 1-min closes to K-minute bars (1 = canon, no-op)
    if bar_minutes > 1:
        prices_np = prices_np[::bar_minutes]

    # Pad to n_assets if needed
    if prices_np.shape[1] < n_assets:
        n_loaded = prices_np.shape[1]
        pad_indices = [i % n_loaded for i in range(n_assets)]
        prices_np = prices_np[:, pad_indices]
    elif prices_np.shape[1] > n_assets:
        prices_np = prices_np[:, :n_assets]

    prices = jnp.array(prices_np, dtype=jnp.float32)

    # Create JAX env config
    config = EnvConfig(
        n_assets=n_assets,
        initial_cash=100_000.0,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        max_steps=max_episode_steps,
        window_size=60,
        position_size_pct=position_size_pct,
        probabilistic_fills=probabilistic_fills,
        borrow_rate_annual=borrow_rate_annual,
        burning_money=burning_money,
        burn_market_linked=burn_market_linked,
        burn_base_rate=burn_base_rate,
        burn_random=burn_random,
        burn_random_prob=burn_random_prob,
        burn_random_max=burn_random_max,
        burn_random_limit=burn_random_limit,
        # TASK_818 Tier B
        reward_attributable=reward_attributable,
        # TASK_828 Stage 3
        vector_action=vector_action,
        # TASK_839: budget-constrained vector-step sizing (default OFF = canon)
        vector_budget_step=vector_budget_step,
        vector_budget_pct=vector_budget_pct,
        # TASK_840: target-weight rebalance step (default OFF = canon)
        vector_weight_rebalance=vector_weight_rebalance,
        vector_weight_unit=vector_weight_unit,
        vector_weight_max=vector_weight_max,
        vector_rebalance_deadband=vector_rebalance_deadband,
    )

    # Initialize batched environment states
    keys = jax.random.split(rng_key, n_envs)
    states = jit_batch_reset(keys, prices, config)

    return states, prices, config


def create_vectorbt_envs(
    n_envs: int,
    synthetic_level: int,
    n_assets: int = 64,
    window_size: int = 64,
    mode: str = 'MARGIN',
    fee_bps: int = 0,
    slippage_bps: int = 0,
    synthetic_steps: int = 2000,
) -> List[Any]:
    """Create multiple VectorBT environments for batched self-play.

    Args:
        n_envs: Number of environments to create
        synthetic_level: Synthetic curriculum level (0-4)
        n_assets: Number of assets
        window_size: Observation window size
        mode: Trading mode ('SPOT' or 'MARGIN')
        fee_bps: Trading fee in basis points
        slippage_bps: Slippage in basis points
        synthetic_steps: Number of synthetic price steps to generate

    Returns:
        List of VectorBTTradingEnv instances
    """
    from mcts_trader.env.vectorbt_env import VectorBTTradingEnv

    envs = []
    for _ in range(n_envs):
        env = VectorBTTradingEnv(
            n_assets=n_assets,
            window_size=window_size,
            mode=mode,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            synthetic_level=synthetic_level,
            synthetic_steps=synthetic_steps,
        )
        envs.append(env)

    return envs


def get_env_config_from_args(args) -> dict:
    """Extract environment configuration from argparse namespace.

    Utility function to extract environment-related arguments from
    the training script's argparse namespace.

    Args:
        args: argparse.Namespace with environment arguments

    Returns:
        Dict with environment configuration
    """
    config = {
        'n_assets': getattr(args, 'n_assets', 64),
        'window_size': getattr(args, 'window_size', 64),
        'mode': getattr(args, 'mode', 'MARGIN'),
        'fee_bps': getattr(args, 'fee', 0),
        'slippage_bps': getattr(args, 'slippage', 0),
        'synthetic_level': getattr(args, 'synthetic_level', None),
        'steps': getattr(args, 'steps', 400),
    }
    return config
