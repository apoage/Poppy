"""
Student-t distribution utilities for prediction head.

TASK_116: Fat-tailed output distribution for financial returns.

Financial returns have heavier tails than Gaussian - using Student-t:
- When ν < 30: fat tails (financial-like)
- When ν → ∞: converges to Gaussian
- Learnable ν lets model adapt to data characteristics

Reference: MOIRAI (Salesforce) uses Student-t mixture outputs.
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Optional
import flax.linen as nn


def student_t_nll(
    target: jnp.ndarray,
    mu: jnp.ndarray,
    sigma: jnp.ndarray,
    df: jnp.ndarray,
    eps: float = 1e-6,
) -> jnp.ndarray:
    """
    Compute Student-t negative log-likelihood.

    NLL = log(σ) + (ν+1)/2 * log(1 + z²/ν) + const
    where z = (x - μ) / σ

    Args:
        target: Ground truth values (any shape)
        mu: Predicted mean (same shape as target)
        sigma: Predicted scale (same shape as target), must be > 0
        df: Degrees of freedom (same shape or broadcastable), must be > 0
        eps: Small constant for numerical stability

    Returns:
        NLL per element (same shape as target)
    """
    # Ensure positive sigma and df
    sigma = jnp.maximum(sigma, eps)
    df = jnp.maximum(df, eps)

    # Standardized residual
    z = (target - mu) / sigma

    # Student-t NLL (dropping constants that don't affect gradients)
    # Full: -log(Γ((ν+1)/2)) + log(Γ(ν/2)) + 0.5*log(νπ) + log(σ) + (ν+1)/2 * log(1 + z²/ν)
    # Simplified (for gradient-based learning):
    nll = jnp.log(sigma) + 0.5 * (df + 1) * jnp.log1p(z ** 2 / df)

    return nll


def student_t_loss(
    target: jnp.ndarray,
    mu: jnp.ndarray,
    log_sigma: jnp.ndarray,
    log_df: jnp.ndarray,
    df_min: float = 2.1,
    df_max: float = 100.0,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute Student-t loss with bounded parameters.

    Uses log-parameterization for sigma and df for numerical stability.

    Args:
        target: Ground truth values
        mu: Predicted mean
        log_sigma: Log of predicted scale
        log_df: Log of degrees of freedom (before bounds)
        df_min: Minimum df (>2 for finite variance)
        df_max: Maximum df (large = nearly Gaussian)

    Returns:
        loss: Mean NLL
        sigma: Actual scale values (for logging)
        df: Actual df values (for logging)
    """
    # Convert from log-space with clipping to prevent overflow/underflow
    # exp(10) ≈ 22026, exp(-10) ≈ 4.5e-5 - safe range for sigma
    log_sigma_clipped = jnp.clip(log_sigma, -10.0, 10.0)
    sigma = jnp.exp(log_sigma_clipped)

    # Bound df to [df_min, df_max] using sigmoid
    df_range = df_max - df_min
    df = df_min + df_range * jax.nn.sigmoid(log_df)

    # Compute NLL
    nll = student_t_nll(target, mu, sigma, df)

    return jnp.mean(nll), sigma, df


class StudentTHead(nn.Module):
    """
    Small head to predict Student-t parameters from features.

    Outputs (log_sigma, log_df) to be combined with mean prediction.
    """
    hidden_dim: int = 64
    n_assets: int = 64

    @nn.compact
    def __call__(self, features: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Predict uncertainty parameters.

        Args:
            features: (batch, feature_dim) input features (e.g., latent state)

        Returns:
            log_sigma: (batch, n_assets) log scale
            log_df: (batch, n_assets) log degrees of freedom
        """
        # Small MLP for uncertainty
        h = nn.Dense(self.hidden_dim)(features)
        h = nn.silu(h)
        h = nn.Dense(self.hidden_dim)(h)
        h = nn.silu(h)

        # Output heads
        log_sigma = nn.Dense(self.n_assets, name='sigma_head')(h)
        log_df = nn.Dense(self.n_assets, name='df_head')(h)

        # Initialize sigma near typical volatility (~0.01)
        # log(0.01) ≈ -4.6, so initialize bias near -4
        log_sigma = log_sigma - 4.0

        # Initialize df near 5 (moderately fat tails)
        # Will be bounded to [2.1, 100] in loss function
        log_df = log_df + 1.0  # Start near df ≈ 5

        return log_sigma, log_df


def hybrid_prediction_loss(
    target: jnp.ndarray,
    mu: jnp.ndarray,
    log_sigma: Optional[jnp.ndarray] = None,
    log_df: Optional[jnp.ndarray] = None,
    use_student_t: bool = True,
    student_t_weight: float = 0.5,
    mse_weight: float = 0.5,
) -> Tuple[jnp.ndarray, dict]:
    """
    Hybrid loss combining MSE and Student-t NLL.

    This allows gradual transition from pure MSE to Student-t,
    or using both for stability.

    Args:
        target: Ground truth (batch, n_assets)
        mu: Predicted mean (batch, n_assets)
        log_sigma: Log scale (batch, n_assets), optional
        log_df: Log df (batch, n_assets), optional
        use_student_t: Whether to include Student-t component
        student_t_weight: Weight for Student-t NLL
        mse_weight: Weight for MSE component

    Returns:
        loss: Combined loss
        metrics: Dict with component losses and learned parameters
    """
    metrics = {}

    # MSE component (always computed for baseline comparison)
    mse_loss = jnp.mean((target - mu) ** 2)
    metrics['mse_loss'] = mse_loss

    if use_student_t and log_sigma is not None and log_df is not None:
        # Student-t component
        student_t_nll_val, sigma, df = student_t_loss(target, mu, log_sigma, log_df)
        metrics['student_t_nll'] = student_t_nll_val
        metrics['sigma_mean'] = jnp.mean(sigma)
        metrics['df_mean'] = jnp.mean(df)

        # Combined loss
        total_loss = mse_weight * mse_loss + student_t_weight * student_t_nll_val
    else:
        # Pure MSE fallback
        total_loss = mse_loss

    metrics['total_loss'] = total_loss
    return total_loss, metrics


# Convenience function for gradual adoption
def maybe_student_t_loss(
    target: jnp.ndarray,
    mu: jnp.ndarray,
    uncertainty_params: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
    use_student_t: bool = False,
) -> Tuple[jnp.ndarray, dict]:
    """
    Wrapper for easy integration - uses MSE if Student-t disabled.

    Args:
        target: Ground truth
        mu: Predicted mean
        uncertainty_params: Optional (log_sigma, log_df) tuple
        use_student_t: Flag to enable Student-t (can be CLI arg)

    Returns:
        loss, metrics
    """
    if use_student_t and uncertainty_params is not None:
        log_sigma, log_df = uncertainty_params
        return hybrid_prediction_loss(
            target, mu, log_sigma, log_df,
            use_student_t=True,
            student_t_weight=0.8,
            mse_weight=0.2,
        )
    else:
        mse = jnp.mean((target - mu) ** 2)
        return mse, {'mse_loss': mse, 'total_loss': mse}
