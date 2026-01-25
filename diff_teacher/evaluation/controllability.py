"""Controllability metrics for skeleton-conditioned generation."""

import numpy as np
from typing import Optional, Dict
from dataclasses import dataclass

import sys
sys.path.insert(0, '/home/lukas/Trader')

# Try to import skeleton types, fall back to dict handling
try:
    from diff_teacher.skeleton.types import MarketSkeleton, REGIME_NAMES
    from diff_teacher.constants import OHLCV_CLOSE
except ImportError:
    MarketSkeleton = dict
    REGIME_NAMES = ['crash', 'rally', 'sideways', 'volatile', 'calm']
    OHLCV_CLOSE = 3  # Fallback


@dataclass
class ControllabilityReport:
    """Report of controllability metrics."""
    regime_adherence: float         # Accuracy of regime classification
    volatility_targeting_rmse: float  # RMSE of volatility targeting
    correlation_targeting_error: float  # Frobenius norm of correlation error
    trend_direction_accuracy: float   # Accuracy of trend direction

    def to_dict(self) -> dict:
        return {
            'regime_adherence': self.regime_adherence,
            'volatility_targeting_rmse': self.volatility_targeting_rmse,
            'correlation_targeting_error': self.correlation_targeting_error,
            'trend_direction_accuracy': self.trend_direction_accuracy,
        }


def classify_regime(returns: np.ndarray, window: int = 20) -> np.ndarray:
    """
    Classify regime from returns using simple heuristics.

    Args:
        returns: Return series (timesteps, [assets])
        window: Rolling window

    Returns:
        Array of regime indices per timestep
    """
    if returns.ndim > 1:
        avg_returns = returns.mean(axis=1)
    else:
        avg_returns = returns

    n = len(avg_returns)
    regimes = np.full(n, 2)  # Default: sideways

    # Rolling statistics
    for i in range(window, n):
        window_returns = avg_returns[i - window:i]
        mean_ret = np.mean(window_returns)
        vol = np.std(window_returns)

        # Normalize
        if vol > 0:
            ret_z = mean_ret / vol
        else:
            ret_z = 0

        # Classify
        if ret_z < -1.5 and vol > np.std(avg_returns) * 1.2:
            regimes[i] = 0  # crash
        elif ret_z > 1.5:
            regimes[i] = 1  # rally
        elif vol > np.std(avg_returns) * 1.5:
            regimes[i] = 3  # volatile
        elif vol < np.std(avg_returns) * 0.5:
            regimes[i] = 4  # calm
        else:
            regimes[i] = 2  # sideways

    return regimes


def regime_adherence(
    generated: np.ndarray,
    target_regime: int,
    window: int = 20,
) -> float:
    """
    Measure how well generated data adheres to target regime.

    Args:
        generated: Generated price data
        target_regime: Target regime index (0-4)
        window: Classification window

    Returns:
        Adherence score [0, 1] (fraction of timesteps in target regime)
    """
    # Compute returns
    if generated.ndim >= 2:
        returns = np.diff(np.log(np.maximum(generated, 1e-10)), axis=0)
    else:
        returns = generated

    # Classify
    classified = classify_regime(returns, window)

    # Compute adherence
    adherence = np.mean(classified == target_regime)

    return adherence


def volatility_targeting(
    generated: np.ndarray,
    target_vol_percentile: float,
    reference_vol: Optional[float] = None,
    window: int = 20,
) -> float:
    """
    Measure volatility targeting accuracy.

    Args:
        generated: Generated price data
        target_vol_percentile: Target volatility percentile [0, 1]
        reference_vol: Reference volatility for scaling
        window: Rolling window for realized volatility

    Returns:
        RMSE of volatility targeting
    """
    # Compute returns
    if generated.ndim >= 2:
        returns = np.diff(np.log(np.maximum(generated, 1e-10)), axis=0)
        if returns.ndim > 1:
            returns = returns.mean(axis=-1) if returns.ndim == 2 else returns.mean(axis=(1, 2))
    else:
        returns = generated

    # Compute realized volatility
    realized_vol = np.zeros(len(returns) - window + 1)
    for i in range(len(realized_vol)):
        realized_vol[i] = np.std(returns[i:i + window])

    # Estimate reference vol from data if not provided
    if reference_vol is None:
        reference_vol = np.std(returns)

    # Target volatility
    target_vol = target_vol_percentile * reference_vol * 2  # Scale factor

    # RMSE
    rmse = np.sqrt(np.mean((realized_vol - target_vol) ** 2))

    return rmse


def correlation_targeting(
    generated: np.ndarray,
    target_clusters: np.ndarray,
) -> float:
    """
    Measure correlation structure targeting accuracy.

    Assets in the same cluster should have higher correlation
    than assets in different clusters.

    Args:
        generated: Generated price data (timesteps, assets, [features])
        target_clusters: Target cluster assignments (n_assets,)

    Returns:
        Error metric (lower is better)
    """
    # Compute returns - MED-002 FIX: Use OHLCV_CLOSE (3) not 0
    if generated.ndim == 3:
        returns = np.diff(generated[:, :, OHLCV_CLOSE], axis=0)
    else:
        returns = np.diff(generated, axis=0)

    n_assets = returns.shape[1]

    # Compute correlation matrix
    corr = np.corrcoef(returns.T)
    corr = np.nan_to_num(corr, nan=0.0)

    # Build target correlation structure
    # Same cluster -> should be highly correlated
    # Different cluster -> should have lower correlation
    intra_cluster_corrs = []
    inter_cluster_corrs = []

    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            if target_clusters[i] == target_clusters[j]:
                intra_cluster_corrs.append(corr[i, j])
            else:
                inter_cluster_corrs.append(corr[i, j])

    # Metric: intra-cluster should be higher than inter-cluster
    if len(intra_cluster_corrs) > 0 and len(inter_cluster_corrs) > 0:
        mean_intra = np.mean(intra_cluster_corrs)
        mean_inter = np.mean(inter_cluster_corrs)

        # Error: how much intra exceeds inter (negative = bad)
        # Ideal: intra >> inter, so (intra - inter) should be positive
        # Return negative of this so lower = better
        error = -(mean_intra - mean_inter)
    else:
        error = 0.0

    return error


def trend_direction_accuracy(
    generated: np.ndarray,
    target_direction: np.ndarray,
    window: int = 20,
) -> float:
    """
    Measure trend direction accuracy.

    Args:
        generated: Generated price data
        target_direction: Target direction per asset (-1, 0, +1)
        window: Window for trend estimation

    Returns:
        Accuracy [0, 1]
    """
    # Compute returns - MED-002 FIX: Use OHLCV_CLOSE (3) not 0
    if generated.ndim == 3:
        returns = np.diff(generated[:, :, OHLCV_CLOSE], axis=0)
    elif generated.ndim == 2:
        returns = np.diff(generated, axis=0)
    else:
        returns = generated.reshape(-1, 1)

    n_assets = returns.shape[1]

    # Compute realized direction
    realized_direction = np.zeros(n_assets)
    for i in range(n_assets):
        # Use sign of mean return over full period
        mean_return = np.mean(returns[:, i])
        std_return = np.std(returns[:, i])

        if std_return > 0:
            z = mean_return / std_return
            if z > 0.5:
                realized_direction[i] = 1
            elif z < -0.5:
                realized_direction[i] = -1
            else:
                realized_direction[i] = 0
        else:
            realized_direction[i] = 0

    # Accuracy
    if len(target_direction) != n_assets:
        # Broadcast if needed
        if len(target_direction) == 1:
            target_direction = np.full(n_assets, target_direction[0])
        else:
            target_direction = target_direction[:n_assets]

    accuracy = np.mean(realized_direction == target_direction)

    return accuracy


def compute_controllability(
    generated: np.ndarray,
    skeleton: Optional[MarketSkeleton] = None,
    target_regime: Optional[int] = None,
    target_vol_percentile: Optional[float] = None,
    target_clusters: Optional[np.ndarray] = None,
    target_direction: Optional[np.ndarray] = None,
) -> ControllabilityReport:
    """
    Compute comprehensive controllability report.

    Args:
        generated: Generated price data
        skeleton: MarketSkeleton with targets (alternative to individual args)
        target_regime: Target regime index
        target_vol_percentile: Target volatility percentile
        target_clusters: Target correlation clusters
        target_direction: Target trend direction

    Returns:
        ControllabilityReport
    """
    # Extract targets from skeleton if provided
    if skeleton is not None:
        if hasattr(skeleton, 'regime_sequence') and skeleton.regime_sequence is not None:
            target_regime = int(np.bincount(np.array(skeleton.regime_sequence).astype(int)).argmax())
        if hasattr(skeleton, 'vol_regime') and skeleton.vol_regime is not None:
            target_vol_percentile = float(np.mean(np.array(skeleton.vol_regime)))
        if hasattr(skeleton, 'correlation_clusters') and skeleton.correlation_clusters is not None:
            target_clusters = np.array(skeleton.correlation_clusters)
        if hasattr(skeleton, 'trend_direction') and skeleton.trend_direction is not None:
            td = np.array(skeleton.trend_direction)
            target_direction = td.mean(axis=0) if td.ndim > 1 else td

    # Compute metrics
    regime_adh = regime_adherence(generated, target_regime or 2) if target_regime is not None else np.nan
    vol_rmse = volatility_targeting(generated, target_vol_percentile or 0.5) if target_vol_percentile is not None else np.nan

    if target_clusters is not None:
        corr_err = correlation_targeting(generated, target_clusters)
    else:
        corr_err = np.nan

    if target_direction is not None:
        trend_acc = trend_direction_accuracy(generated, target_direction)
    else:
        trend_acc = np.nan

    return ControllabilityReport(
        regime_adherence=regime_adh,
        volatility_targeting_rmse=vol_rmse,
        correlation_targeting_error=corr_err,
        trend_direction_accuracy=trend_acc,
    )
