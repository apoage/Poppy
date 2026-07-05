"""
TASK_051: Burning Money (HOLD Cost Mechanism)

Forces agent to deploy capital by adding opportunity cost to holding cash.

Key insight: Cash burns, positions don't (as much).
- Bull market: Cash "misses" gains (penalty)
- Bear market: Cash "avoids" losses (reward)

This is differential reward applied specifically to uninvested capital.
"""

from dataclasses import dataclass
from typing import Optional
import jax.numpy as jnp
import numpy as np


@dataclass
class BurningMoneyConfig:
    """Configuration for burning money mechanics."""

    # Master enable
    enabled: bool = False

    # Market-linked burn (opportunity cost on cash)
    market_linked: bool = True

    # Base decay (constant drain regardless of market)
    base_decay_rate: float = 0.00005  # 0.005% per step baseline

    # Random component (anti-gaming)
    random_burn: bool = True
    random_burn_prob: float = 0.03    # 3% chance of surprise cost
    random_burn_min: float = 0.0005   # 0.05% minimum surprise
    random_burn_max: float = 0.002    # 0.2% maximum surprise


def compute_burn_jax(
    cash: jnp.ndarray,
    portfolio_value: jnp.ndarray,
    market_prices: jnp.ndarray,
    prev_prices: jnp.ndarray,
    config: BurningMoneyConfig,
    rng_key: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """
    Compute burning money penalty/reward for JAX environment.

    Args:
        cash: Current cash balance
        portfolio_value: Total portfolio value
        market_prices: Current asset prices [n_assets]
        prev_prices: Previous asset prices [n_assets]
        config: BurningMoneyConfig
        rng_key: JAX random key (for random burn)

    Returns:
        burn: Amount to SUBTRACT from reward (positive = penalty, negative = reward)
    """
    if not config.enabled:
        return jnp.array(0.0)

    # Cash ratio (fraction uninvested)
    cash_ratio = cash / jnp.maximum(portfolio_value, 1e-8)
    cash_ratio = jnp.clip(cash_ratio, 0.0, 1.0)

    burn = jnp.array(0.0)

    # 1. Market-linked burn (opportunity cost)
    if config.market_linked:
        # Average market return this step
        returns = (market_prices - prev_prices) / jnp.maximum(prev_prices, 1e-8)
        market_return = jnp.mean(returns)

        # Cash "missed" this return (or avoided loss if negative)
        # Positive market = positive burn (penalty for holding cash)
        # Negative market = negative burn (reward for holding cash)
        market_burn = market_return * cash_ratio
        burn = burn + market_burn

    # 2. Base decay (constant pressure to deploy capital)
    base_burn = config.base_decay_rate * cash_ratio
    burn = burn + base_burn

    # 3. Random burn (surprise costs - anti-gaming)
    # Note: Random burn is harder in pure JAX, may need to be applied in Python wrapper
    # For now, skip in JAX version (or use deterministic approximation)

    return burn


def compute_burn_numpy(
    cash: float,
    portfolio_value: float,
    market_prices: np.ndarray,
    prev_prices: np.ndarray,
    config: BurningMoneyConfig,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    Compute burning money penalty/reward for NumPy environment.

    Args:
        cash: Current cash balance
        portfolio_value: Total portfolio value
        market_prices: Current asset prices [n_assets]
        prev_prices: Previous asset prices [n_assets]
        config: BurningMoneyConfig
        rng: NumPy random generator (for random burn)

    Returns:
        burn: Amount to SUBTRACT from reward (positive = penalty, negative = reward)
    """
    if not config.enabled:
        return 0.0

    # Cash ratio (fraction uninvested)
    cash_ratio = cash / max(portfolio_value, 1e-8)
    cash_ratio = np.clip(cash_ratio, 0.0, 1.0)

    burn = 0.0

    # 1. Market-linked burn (opportunity cost)
    if config.market_linked:
        # Average market return this step
        valid_mask = prev_prices > 1e-8
        if valid_mask.any():
            returns = np.where(
                valid_mask,
                (market_prices - prev_prices) / prev_prices,
                0.0
            )
            market_return = np.mean(returns[valid_mask])
        else:
            market_return = 0.0

        # Cash "missed" this return
        market_burn = market_return * cash_ratio
        burn += market_burn

    # 2. Base decay (constant pressure)
    base_burn = config.base_decay_rate * cash_ratio
    burn += base_burn

    # 3. Random burn (surprise costs)
    if config.random_burn and rng is not None:
        if rng.random() < config.random_burn_prob:
            surprise = rng.uniform(config.random_burn_min, config.random_burn_max)
            burn += surprise * cash_ratio

    return burn


def create_config_from_args(args) -> BurningMoneyConfig:
    """Create BurningMoneyConfig from argparse namespace."""
    return BurningMoneyConfig(
        enabled=getattr(args, 'burning_money', False),
        market_linked=getattr(args, 'burn_market_linked', True),
        base_decay_rate=getattr(args, 'burn_base_rate', 0.00005),
        random_burn=getattr(args, 'burn_random', True),
        random_burn_prob=getattr(args, 'burn_random_prob', 0.03),
        random_burn_min=getattr(args, 'burn_random_min', 0.0005),
        random_burn_max=getattr(args, 'burn_random_max', 0.002),
    )


def add_argparse_args(parser):
    """Add burning money CLI arguments to parser."""
    group = parser.add_argument_group('Burning Money (TASK_051)')

    group.add_argument('--burning-money', action='store_true',
                       help='Enable burning money mechanics (cash opportunity cost)')
    group.add_argument('--burn-market-linked', action='store_true', default=True,
                       help='Cash burns at market return rate')
    group.add_argument('--no-burn-market-linked', action='store_false', dest='burn_market_linked',
                       help='Disable market-linked burn')
    group.add_argument('--burn-base-rate', type=float, default=0.00005,
                       help='Base per-step cash decay rate (default: 0.00005)')
    group.add_argument('--burn-random', action='store_true', default=True,
                       help='Add random burn events')
    group.add_argument('--no-burn-random', action='store_false', dest='burn_random',
                       help='Disable random burn')
    group.add_argument('--burn-random-prob', type=float, default=0.03,
                       help='Probability of random burn event per step (default: 0.03)')
    group.add_argument('--burn-random-min', type=float, default=0.0005,
                       help='Minimum random burn amount (default: 0.0005)')
    group.add_argument('--burn-random-max', type=float, default=0.002,
                       help='Maximum random burn amount (default: 0.002)')

    return parser
