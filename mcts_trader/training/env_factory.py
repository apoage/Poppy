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
        rng_key: JAX RNG key (optional, uses PRNGKey(42) if None)

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

    if rng_key is None:
        rng_key = jax.random.PRNGKey(42)

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
