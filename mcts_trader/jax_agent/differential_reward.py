"""
Differential Reward (Opportunity Cost).

TASK_038: Reward relative to market benchmark.

The problem with absolute rewards:
- R(HOLD) = 0 always → HOLD appears "safe"
- R(TRADE) = ΔP - fees → often negative due to fees
- Agent learns: E[R(HOLD)] = 0 > E[R(TRADE)] ≈ -fees

Differential reward solution:
- R = R_agent - R_benchmark
- If market goes up and agent HOLDs: R = 0 - market_return < 0 (penalty!)
- If market goes down and agent HOLDs: R = 0 - market_return > 0 (reward!)
- HOLD now has VARIABLE reward based on market conditions

References:
- Moody & Saffell (2001): Differential Sharpe Ratio
- Gemini Research: "Reward shaping with opportunity cost"
"""

import jax.numpy as jnp
from typing import Tuple, Optional
import chex
from mcts_trader.utils.transforms import symlog


def compute_market_return(
    current_prices: jnp.ndarray,
    prev_prices: jnp.ndarray,
    weights: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """
    Compute market benchmark return.

    Args:
        current_prices: (..., num_assets) current prices
        prev_prices: (..., num_assets) previous prices
        weights: Optional (num_assets,) weights. Default: equal-weighted.

    Returns:
        (...,) market return as percentage
    """
    # Per-asset returns
    price_returns = (current_prices - prev_prices) / jnp.maximum(prev_prices, 1e-10)

    if weights is None:
        # Equal-weighted average
        market_return = jnp.mean(price_returns, axis=-1)
    else:
        # Weighted average
        market_return = jnp.sum(price_returns * weights, axis=-1)

    return market_return


def compute_differential_reward(
    agent_return: jnp.ndarray,
    market_return: jnp.ndarray,
    baseline_weight: float = 1.0,
) -> jnp.ndarray:
    """
    Compute differential (opportunity cost) reward.

    Args:
        agent_return: (...,) agent's portfolio return
        market_return: (...,) market benchmark return
        baseline_weight: How much to weight baseline (0=ignore, 1=full)

    Returns:
        (...,) differential reward
    """
    return agent_return - baseline_weight * market_return


def compute_differential_reward_from_values(
    prev_portfolio_value: jnp.ndarray,
    new_portfolio_value: jnp.ndarray,
    current_prices: jnp.ndarray,
    prev_prices: jnp.ndarray,
    baseline_type: str = 'market',
    baseline_weight: float = 1.0,
) -> Tuple[jnp.ndarray, dict]:
    """
    Compute differential reward from portfolio and price data.

    This is the main function to call from the environment.

    Args:
        prev_portfolio_value: (...,) previous portfolio value
        new_portfolio_value: (...,) new portfolio value
        current_prices: (..., num_assets) current prices
        prev_prices: (..., num_assets) previous prices
        baseline_type: Type of baseline
            - 'market': Equal-weighted market return
            - 'hold': Return from holding current positions (not implemented)
            - 'zero': No baseline (standard reward)
        baseline_weight: Weight for baseline (0-1)

    Returns:
        (differential_reward, metrics_dict)
    """
    # Agent's return
    agent_return = (new_portfolio_value - prev_portfolio_value) / jnp.maximum(
        prev_portfolio_value, 1e-10
    )

    # Market return (baseline)
    if baseline_type == 'market':
        market_return = compute_market_return(current_prices, prev_prices)
    elif baseline_type == 'zero':
        market_return = jnp.zeros_like(agent_return)
    else:
        # Default to market
        market_return = compute_market_return(current_prices, prev_prices)

    # Differential reward
    differential = compute_differential_reward(
        agent_return, market_return, baseline_weight
    )

    metrics = {
        'agent_return': jnp.mean(agent_return),
        'market_return': jnp.mean(market_return),
        'differential_reward': jnp.mean(differential),
        'alpha': jnp.mean(agent_return - market_return),  # Excess return
    }

    return differential, metrics


class DifferentialRewardWrapper:
    """
    Wrapper to add differential reward computation to any environment.

    Usage:
        wrapper = DifferentialRewardWrapper(baseline_type='market')
        raw_reward = env.step(action)
        diff_reward = wrapper.transform(
            raw_reward,
            current_prices=env.current_prices,
            prev_prices=env.prev_prices,
            portfolio_value=env.portfolio_value,
            prev_portfolio_value=env.prev_portfolio_value,
        )
    """

    def __init__(
        self,
        baseline_type: str = 'market',
        baseline_weight: float = 1.0,
        use_differential: bool = True,
    ):
        """
        Args:
            baseline_type: 'market', 'hold', or 'zero'
            baseline_weight: How much to weight baseline
            use_differential: If False, pass through raw reward
        """
        self.baseline_type = baseline_type
        self.baseline_weight = baseline_weight
        self.use_differential = use_differential

        # Running statistics
        self.total_agent_return = 0.0
        self.total_market_return = 0.0
        self.num_steps = 0

    def transform(
        self,
        raw_reward: jnp.ndarray,
        current_prices: jnp.ndarray,
        prev_prices: jnp.ndarray,
        portfolio_value: jnp.ndarray,
        prev_portfolio_value: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, dict]:
        """
        Transform raw reward to differential reward.

        Args:
            raw_reward: Original environment reward (unused, we recompute)
            current_prices: Current asset prices
            prev_prices: Previous asset prices
            portfolio_value: Current portfolio value
            prev_portfolio_value: Previous portfolio value

        Returns:
            (differential_reward, metrics_dict)
        """
        if not self.use_differential:
            return raw_reward, {'raw_reward': jnp.mean(raw_reward)}

        diff_reward, metrics = compute_differential_reward_from_values(
            prev_portfolio_value=prev_portfolio_value,
            new_portfolio_value=portfolio_value,
            current_prices=current_prices,
            prev_prices=prev_prices,
            baseline_type=self.baseline_type,
            baseline_weight=self.baseline_weight,
        )

        # Update running statistics
        self.total_agent_return += float(metrics['agent_return'])
        self.total_market_return += float(metrics['market_return'])
        self.num_steps += 1

        metrics['cumulative_alpha'] = self.total_agent_return - self.total_market_return
        metrics['avg_alpha_per_step'] = metrics['cumulative_alpha'] / max(self.num_steps, 1)

        return diff_reward, metrics

    def reset(self):
        """Reset running statistics."""
        self.total_agent_return = 0.0
        self.total_market_return = 0.0
        self.num_steps = 0

    def get_alpha(self) -> float:
        """Get cumulative excess return (alpha)."""
        return self.total_agent_return - self.total_market_return


def symlog_differential(
    differential_reward: jnp.ndarray,
    scale: float = 100.0,
    threshold: float = 1.0,
) -> jnp.ndarray:
    """
    Apply symlog transformation to differential reward.

    Args:
        differential_reward: Raw differential reward (usually small percentages)
        scale: Scale factor before symlog (100 converts 0.01 to 1.0)
        threshold: Symlog threshold

    Returns:
        Transformed reward
    """
    scaled = differential_reward * scale
    return symlog(scaled, threshold=threshold)


# Example integration with JAX environment

def integrate_differential_reward_jax(
    agent_pnl: jnp.ndarray,          # (batch,) agent's PnL
    prev_portfolio_value: jnp.ndarray,  # (batch,) previous value
    current_prices: jnp.ndarray,     # (batch, num_assets)
    prev_prices: jnp.ndarray,        # (batch, num_assets)
    use_differential: bool = True,
    baseline_weight: float = 1.0,
) -> jnp.ndarray:
    """
    JAX-compatible differential reward computation.

    Can be used inside JIT-compiled functions.

    Args:
        agent_pnl: Agent's profit/loss this step
        prev_portfolio_value: Portfolio value before step
        current_prices: Current asset prices
        prev_prices: Previous asset prices
        use_differential: Whether to apply differential
        baseline_weight: Weight for market baseline

    Returns:
        (batch,) differential reward
    """
    # Agent's percentage return
    agent_return = agent_pnl / jnp.maximum(prev_portfolio_value, 1e-10)

    if not use_differential:
        return agent_return

    # Market return (equal-weighted)
    price_returns = (current_prices - prev_prices) / jnp.maximum(prev_prices, 1e-10)
    market_return = jnp.mean(price_returns, axis=-1)

    # Differential
    differential = agent_return - baseline_weight * market_return

    return differential
