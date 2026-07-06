"""Stylized facts evaluation for financial time series."""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from ..constants import OHLCV_CLOSE


@dataclass
class StylizedFactsReport:
    """Report of stylized facts metrics."""
    fat_tail_index: float           # Should be ~3 for financial returns
    volatility_clustering: float    # ACF of squared returns at lag 1
    leverage_effect: float          # Correlation(r_t, sigma^2_{t+1})
    autocorrelation_lag1: float     # ACF of returns at lag 1 (should be ~0)
    autocorrelation_lag5: float
    mean_return: float
    std_return: float
    skewness: float
    kurtosis: float                 # Should be >3 (fat tails)

    def to_dict(self) -> dict:
        return {
            'fat_tail_index': self.fat_tail_index,
            'volatility_clustering': self.volatility_clustering,
            'leverage_effect': self.leverage_effect,
            'autocorrelation_lag1': self.autocorrelation_lag1,
            'autocorrelation_lag5': self.autocorrelation_lag5,
            'mean_return': self.mean_return,
            'std_return': self.std_return,
            'skewness': self.skewness,
            'kurtosis': self.kurtosis,
        }


def compute_returns(prices: np.ndarray) -> np.ndarray:
    """Compute log returns from prices."""
    if prices.ndim == 3:
        # (timesteps, assets, features) - use close price
        # BUG-009 FIX: Use standardized OHLCV_CLOSE constant
        close_idx = min(OHLCV_CLOSE, prices.shape[-1] - 1)
        close = prices[:, :, close_idx]
    else:
        close = prices

    log_prices = np.log(np.maximum(close, 1e-10))
    returns = np.diff(log_prices, axis=0)
    return returns


def fat_tail_index(returns: np.ndarray, quantile: float = 0.95) -> float:
    """
    Estimate tail index using Hill estimator.

    For financial returns, typically α ≈ 3.
    Higher values = lighter tails, lower = heavier.

    Args:
        returns: Return series (flatten if multi-dimensional)
        quantile: Threshold quantile for tail estimation

    Returns:
        Estimated tail index α
    """
    returns_flat = returns.flatten()
    returns_flat = returns_flat[~np.isnan(returns_flat)]

    if len(returns_flat) < 100:
        return np.nan

    # Use absolute returns for tail analysis
    abs_returns = np.abs(returns_flat)
    threshold = np.quantile(abs_returns, quantile)
    tail_values = abs_returns[abs_returns > threshold]

    if len(tail_values) < 10:
        return np.nan

    # Hill estimator: α = n / Σ log(x_i / threshold)
    log_ratios = np.log(tail_values / threshold)
    sum_log_ratios = np.sum(log_ratios)
    if np.abs(sum_log_ratios) < 1e-10:
        return np.nan  # Degenerate case: all tail values equal threshold
    alpha = len(tail_values) / sum_log_ratios

    return alpha


def autocorrelation(x: np.ndarray, lag: int = 1) -> float:
    """Compute autocorrelation at specified lag."""
    x = x.flatten()
    x = x[~np.isnan(x)]

    if len(x) <= lag:
        return np.nan

    x_mean = np.mean(x)
    x_var = np.var(x)

    if x_var < 1e-10:
        return 0.0

    x_centered = x - x_mean
    # LOW-002 FIX: Guard against small denominator
    denominator = max(len(x) * x_var, 1e-10)
    acf = np.sum(x_centered[lag:] * x_centered[:-lag]) / denominator

    return acf


def volatility_clustering(returns: np.ndarray, lags: Tuple[int, ...] = (1, 5, 10, 20)) -> Dict[int, float]:
    """
    Measure volatility clustering via ACF of squared returns.

    Financial returns exhibit volatility clustering: periods of high
    volatility tend to cluster together.

    Args:
        returns: Return series (timesteps,) or (timesteps, assets)
        lags: Lags to compute ACF at

    Returns:
        Dictionary mapping lag -> ACF value
    """
    # BUG-005 FIX: Compute per-asset, then average to avoid boundary bleed
    if returns.ndim == 2:
        n_assets = returns.shape[1]
        result = {lag: [] for lag in lags}
        for asset_idx in range(n_assets):
            asset_returns = returns[:, asset_idx]
            squared_returns = asset_returns ** 2
            for lag in lags:
                acf_val = autocorrelation(squared_returns, lag)
                if not np.isnan(acf_val):
                    result[lag].append(acf_val)
        # Average across assets
        return {lag: np.mean(vals) if vals else np.nan for lag, vals in result.items()}
    else:
        # 1D case - original behavior
        squared_returns = returns.flatten() ** 2
        result = {}
        for lag in lags:
            result[lag] = autocorrelation(squared_returns, lag)
        return result


def leverage_effect(returns: np.ndarray, window: int = 20) -> float:
    """
    Measure leverage effect: correlation between returns and future volatility.

    In equity markets, negative returns tend to increase future volatility
    (leverage effect). Should be negative (typically around -0.3).

    Args:
        returns: Return series (timesteps,) or (timesteps, assets)
        window: Window for volatility estimation

    Returns:
        Correlation coefficient
    """
    # BUG-005 FIX: Compute per-asset, then average to avoid boundary bleed
    if returns.ndim == 2:
        n_assets = returns.shape[1]
        correlations = []
        for asset_idx in range(n_assets):
            asset_returns = returns[:, asset_idx]
            asset_returns = asset_returns[~np.isnan(asset_returns)]

            if len(asset_returns) < window + 10:
                continue

            # Compute rolling volatility for this asset
            volatility = np.zeros(len(asset_returns) - window + 1)
            for i in range(len(volatility)):
                volatility[i] = np.std(asset_returns[i:i + window])

            # Align: return at t with volatility at t+1
            returns_aligned = asset_returns[window - 1:-1]
            vol_aligned = volatility[1:]

            if len(returns_aligned) < 10:
                continue

            # Correlation for this asset
            corr = np.corrcoef(returns_aligned, vol_aligned)[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)

        return np.mean(correlations) if correlations else np.nan
    else:
        # 1D case - original behavior
        returns_flat = returns.flatten()
        returns_flat = returns_flat[~np.isnan(returns_flat)]

        if len(returns_flat) < window + 10:
            return np.nan

        # Compute rolling volatility
        volatility = np.zeros(len(returns_flat) - window + 1)
        for i in range(len(volatility)):
            volatility[i] = np.std(returns_flat[i:i + window])

        # Align: return at t with volatility at t+1
        returns_aligned = returns_flat[window - 1:-1]
        vol_aligned = volatility[1:]

        if len(returns_aligned) < 10:
            return np.nan

        # Correlation
        corr = np.corrcoef(returns_aligned, vol_aligned)[0, 1]

        return corr


def compute_stylized_facts(
    prices: np.ndarray,
    reference: Optional[np.ndarray] = None,
) -> StylizedFactsReport:
    """
    Compute comprehensive stylized facts report.

    Args:
        prices: Price data (timesteps, assets, [features])
        reference: Optional reference data for comparison

    Returns:
        StylizedFactsReport with all metrics
    """
    returns = compute_returns(prices)
    returns_flat = returns.flatten()
    returns_flat = returns_flat[~np.isnan(returns_flat)]

    # Basic statistics
    mean_return = np.mean(returns_flat)
    std_return = np.std(returns_flat)

    # Higher moments - MED-009 FIX: Use 1e-10 threshold to avoid overflow with tiny std
    skewness = np.mean(((returns_flat - mean_return) / std_return) ** 3) if std_return > 1e-10 else 0
    kurtosis = np.mean(((returns_flat - mean_return) / std_return) ** 4) if std_return > 1e-10 else 0

    # Stylized facts
    tail_idx = fat_tail_index(returns)
    vol_cluster = volatility_clustering(returns)
    leverage = leverage_effect(returns)
    acf1 = autocorrelation(returns_flat, 1)
    acf5 = autocorrelation(returns_flat, 5)

    return StylizedFactsReport(
        fat_tail_index=tail_idx,
        volatility_clustering=vol_cluster.get(1, np.nan),
        leverage_effect=leverage,
        autocorrelation_lag1=acf1,
        autocorrelation_lag5=acf5,
        mean_return=mean_return,
        std_return=std_return,
        skewness=skewness,
        kurtosis=kurtosis,
    )


def compare_stylized_facts(
    generated: np.ndarray,
    real: np.ndarray,
) -> Dict[str, float]:
    """
    Compare stylized facts between generated and real data.

    Args:
        generated: Generated price data
        real: Real price data

    Returns:
        Dictionary of metric differences (generated - real)
    """
    gen_facts = compute_stylized_facts(generated)
    real_facts = compute_stylized_facts(real)

    differences = {}
    for key in gen_facts.to_dict():
        gen_val = getattr(gen_facts, key)
        real_val = getattr(real_facts, key)
        if not np.isnan(gen_val) and not np.isnan(real_val):
            differences[f'{key}_diff'] = gen_val - real_val
            differences[f'{key}_gen'] = gen_val
            differences[f'{key}_real'] = real_val

    return differences


# Expected values for well-behaved financial time series
EXPECTED_STYLIZED_FACTS = {
    'fat_tail_index': 3.0,          # Power law exponent ~3
    'volatility_clustering_lag1': 0.2,  # Positive, slowly decaying
    'leverage_effect': -0.3,         # Negative
    'autocorrelation_lag1': 0.0,     # Near zero
    'kurtosis': 6.0,                 # Excess kurtosis (>3)
}
