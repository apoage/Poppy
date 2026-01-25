"""Fidelity metrics comparing generated to real data."""

import numpy as np
from typing import Optional
from scipy import stats

from ..constants import OHLCV_CLOSE


def wasserstein_distance(
    generated: np.ndarray,
    real: np.ndarray,
) -> float:
    """
    Compute Wasserstein-1 distance between return distributions.

    Lower is better - indicates generated returns have similar
    distribution to real returns.

    Args:
        generated: Generated price/return data
        real: Real price/return data

    Returns:
        Wasserstein-1 distance
    """
    # Compute returns if prices
    if generated.ndim >= 2:
        gen_returns = np.diff(np.log(np.maximum(generated, 1e-10)), axis=0).flatten()
    else:
        gen_returns = generated.flatten()

    if real.ndim >= 2:
        real_returns = np.diff(np.log(np.maximum(real, 1e-10)), axis=0).flatten()
    else:
        real_returns = real.flatten()

    # Remove NaN
    gen_returns = gen_returns[~np.isnan(gen_returns)]
    real_returns = real_returns[~np.isnan(real_returns)]

    # Compute Wasserstein distance
    return stats.wasserstein_distance(gen_returns, real_returns)


def correlation_matrix_error(
    generated: np.ndarray,
    real: np.ndarray,
) -> float:
    """
    Compute Frobenius norm of correlation matrix difference.

    Lower is better - indicates similar cross-asset dependencies.

    Args:
        generated: Generated data (timesteps, assets, [features])
        real: Real data (timesteps, assets, [features])

    Returns:
        Frobenius norm of |Σ_gen - Σ_real|
    """
    # Extract returns - MED-002 FIX: Use OHLCV_CLOSE (3) not 0
    if generated.ndim == 3:
        gen_returns = np.diff(generated[:, :, OHLCV_CLOSE], axis=0)
    else:
        gen_returns = np.diff(generated, axis=0)

    if real.ndim == 3:
        real_returns = np.diff(real[:, :, OHLCV_CLOSE], axis=0)
    else:
        real_returns = np.diff(real, axis=0)

    # Compute correlation matrices
    gen_corr = np.corrcoef(gen_returns.T)
    real_corr = np.corrcoef(real_returns.T)

    # Handle NaN
    gen_corr = np.nan_to_num(gen_corr, nan=0.0)
    real_corr = np.nan_to_num(real_corr, nan=0.0)

    # Frobenius norm
    diff = gen_corr - real_corr
    frobenius = np.sqrt(np.sum(diff ** 2))

    return frobenius


def distribution_divergence(
    generated: np.ndarray,
    real: np.ndarray,
    n_bins: int = 100,
) -> float:
    """
    Compute KL divergence between return distributions.

    Lower is better. Uses histogram approximation.

    Args:
        generated: Generated data
        real: Real data
        n_bins: Number of histogram bins

    Returns:
        KL divergence (nats)
    """
    # Compute returns
    if generated.ndim >= 2:
        gen_returns = np.diff(np.log(np.maximum(generated, 1e-10)), axis=0).flatten()
    else:
        gen_returns = generated.flatten()

    if real.ndim >= 2:
        real_returns = np.diff(np.log(np.maximum(real, 1e-10)), axis=0).flatten()
    else:
        real_returns = real.flatten()

    # Remove NaN
    gen_returns = gen_returns[~np.isnan(gen_returns)]
    real_returns = real_returns[~np.isnan(real_returns)]

    # Compute histograms with shared bins
    all_returns = np.concatenate([gen_returns, real_returns])
    bins = np.linspace(np.percentile(all_returns, 1), np.percentile(all_returns, 99), n_bins)

    gen_hist, _ = np.histogram(gen_returns, bins=bins, density=True)
    real_hist, _ = np.histogram(real_returns, bins=bins, density=True)

    # Add small epsilon to avoid log(0)
    eps = 1e-10
    gen_hist = gen_hist + eps
    real_hist = real_hist + eps

    # Normalize
    gen_hist = gen_hist / gen_hist.sum()
    real_hist = real_hist / real_hist.sum()

    # KL divergence: sum(p * log(p/q))
    # Guard against extreme ratios
    ratio = np.clip(gen_hist / real_hist, 1e-10, 1e10)
    kl_div = np.sum(gen_hist * np.log(ratio))

    return kl_div


def moment_matching_error(
    generated: np.ndarray,
    real: np.ndarray,
) -> dict:
    """
    Compare statistical moments between generated and real data.

    Args:
        generated: Generated data
        real: Real data

    Returns:
        Dictionary of moment differences
    """
    # Compute returns
    if generated.ndim >= 2:
        gen_returns = np.diff(np.log(np.maximum(generated, 1e-10)), axis=0).flatten()
    else:
        gen_returns = generated.flatten()

    if real.ndim >= 2:
        real_returns = np.diff(np.log(np.maximum(real, 1e-10)), axis=0).flatten()
    else:
        real_returns = real.flatten()

    # Remove NaN
    gen_returns = gen_returns[~np.isnan(gen_returns)]
    real_returns = real_returns[~np.isnan(real_returns)]

    return {
        'mean_diff': np.abs(np.mean(gen_returns) - np.mean(real_returns)),
        'std_diff': np.abs(np.std(gen_returns) - np.std(real_returns)),
        'skew_diff': np.abs(stats.skew(gen_returns) - stats.skew(real_returns)),
        'kurtosis_diff': np.abs(stats.kurtosis(gen_returns) - stats.kurtosis(real_returns)),
    }
