"""
Alpha-relative reward computation.

TASK_118: Fast max-alpha calculation using top-k selection.

Instead of solving NP-hard optimal portfolio problem, approximate with:
    max_alpha ≈ Σ_t top_k(|returns_t|)

This is O(T × n), fully JAX-compatible, and provides consistent
normalization across market regimes.

Note: Uses |returns| so captures both long AND short opportunities.
If asset drops 5%, shorting it captures +5% alpha.
"""

import jax
import jax.numpy as jnp
from typing import Tuple


def compute_max_alpha_topk(
    returns: jnp.ndarray,
    k: int = 4,
) -> jnp.ndarray:
    """
    Compute maximum achievable alpha using top-k selection per step.

    Args:
        returns: (T, n_assets) or (batch, T, n_assets) log returns
        k: Number of top assets to consider per step (position limit proxy)

    Returns:
        Scalar or (batch,) max alpha estimate
    """
    abs_returns = jnp.abs(returns)

    if returns.ndim == 2:
        # (T, n_assets) -> scalar
        top_k_values = jax.lax.top_k(abs_returns, k)[0]  # (T, k)
        return jnp.sum(top_k_values)
    else:
        # (batch, T, n_assets) -> (batch,)
        # vmap over batch dimension
        return jax.vmap(lambda r: compute_max_alpha_topk(r, k))(returns)


def compute_alpha_ratio(
    achieved_pnl: jnp.ndarray,
    returns: jnp.ndarray,
    k: int = 4,
    eps: float = 1e-6,
) -> jnp.ndarray:
    """
    Compute ratio of achieved PnL to max possible alpha.

    Args:
        achieved_pnl: Scalar or (batch,) achieved profit/loss
        returns: (T, n_assets) or (batch, T, n_assets) returns
        k: Position limit for max alpha calculation
        eps: Floor to prevent div/0

    Returns:
        Alpha ratio in [0, 1+] (can exceed 1 if lucky)
    """
    max_alpha = compute_max_alpha_topk(returns, k)
    return achieved_pnl / jnp.maximum(max_alpha, eps)


def alpha_relative_reward(
    raw_reward: jnp.ndarray,
    returns: jnp.ndarray,
    k: int = 4,
    blend: float = 0.5,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Blend raw reward with alpha-relative reward.

    Args:
        raw_reward: Original PnL-based reward
        returns: Price returns for the episode
        k: Position limit
        blend: Weight for alpha-relative (0=pure raw, 1=pure alpha-relative)

    Returns:
        blended_reward: Combined reward signal
        alpha_ratio: For logging/debugging
    """
    alpha_ratio = compute_alpha_ratio(raw_reward, returns, k)

    # Blend: (1-blend)*raw + blend*alpha_relative
    # Scale alpha_ratio to similar magnitude as raw_reward
    raw_scale = jnp.maximum(jnp.abs(raw_reward), 1e-6)
    alpha_reward = alpha_ratio * raw_scale

    blended = (1 - blend) * raw_reward + blend * alpha_reward

    return blended, alpha_ratio


# Convenience: per-step alpha for denser reward signal
def compute_per_step_alpha(
    returns: jnp.ndarray,
    k: int = 4,
) -> jnp.ndarray:
    """
    Compute max alpha available at each timestep.

    Args:
        returns: (T, n_assets) returns
        k: Number of top assets

    Returns:
        (T,) max alpha per step
    """
    abs_returns = jnp.abs(returns)
    top_k_values = jax.lax.top_k(abs_returns, k)[0]  # (T, k)
    return jnp.sum(top_k_values, axis=1)  # (T,)


def compute_cumulative_alpha(
    returns: jnp.ndarray,
    k: int = 4,
) -> jnp.ndarray:
    """
    Compute cumulative max alpha curve.

    Useful for comparing agent's cumulative PnL against theoretical max.

    Args:
        returns: (T, n_assets) returns
        k: Number of top assets

    Returns:
        (T,) cumulative max alpha
    """
    per_step = compute_per_step_alpha(returns, k)
    return jnp.cumsum(per_step)


def compute_optimal_trade_count(
    returns: jnp.ndarray,
    k: int = 4,
    smooth_window: int = 5,
) -> jnp.ndarray:
    """
    Compute optimal number of trades to capture max alpha.

    Uses smoothed returns to avoid counting spurious trades from
    tiny ranking fluctuations. Only counts meaningful regime changes.

    Args:
        returns: (T, n_assets) returns
        k: Number of top assets to hold
        smooth_window: EMA-like smoothing window (higher = fewer trades)

    Returns:
        Scalar: total number of trades needed
    """
    abs_returns = jnp.abs(returns)
    T, n_assets = abs_returns.shape

    # Smooth returns with cumulative sum window (simple moving average proxy)
    # This adds persistence - assets need to be consistently strong to be top-k
    if smooth_window > 1:
        # Cumsum-based rolling sum (JAX-friendly)
        cumsum = jnp.cumsum(abs_returns, axis=0)
        # Padded difference for rolling window
        padded = jnp.concatenate([jnp.zeros((smooth_window, n_assets)), cumsum], axis=0)
        smoothed = (cumsum - padded[:-smooth_window]) / smooth_window
    else:
        smoothed = abs_returns

    # Get top-k indices based on smoothed returns
    top_k_indices = jax.lax.top_k(smoothed, k)[1]  # (T, k)

    # Convert to one-hot for set comparison
    def to_mask(indices):
        return jnp.zeros(n_assets).at[indices].set(1.0)

    masks = jax.vmap(to_mask)(top_k_indices)  # (T, n_assets)

    # Count changes between consecutive steps
    changes = jnp.abs(masks[1:] - masks[:-1])  # (T-1, n_assets)
    trades_per_step = jnp.sum(changes, axis=1)  # (T-1,)

    # Total trades (each entry or exit)
    total_trades = jnp.sum(trades_per_step)

    return total_trades


def compute_optimal_metrics(
    returns: jnp.ndarray,
    k: int = 4,
    smooth_window: int = 5,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute all optimal trading metrics in one pass.

    Args:
        returns: (T, n_assets) returns
        k: Number of top assets
        smooth_window: Smoothing for trade count (reduces spurious trades)

    Returns:
        max_alpha: Total achievable alpha
        trade_count: Optimal number of trades
        alpha_per_trade: Efficiency metric (alpha / trades)
    """
    max_alpha = compute_max_alpha_topk(returns, k)
    trade_count = compute_optimal_trade_count(returns, k, smooth_window)
    alpha_per_trade = max_alpha / jnp.maximum(trade_count, 1.0)

    return max_alpha, trade_count, alpha_per_trade
