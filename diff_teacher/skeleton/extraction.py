"""Extract skeleton structure from real market data."""

import jax.numpy as jnp
from jax import Array
from typing import Optional, Tuple
import numpy as np

from .types import MarketSkeleton, Event, REGIME_TO_IDX


def rolling_mean(x: Array, window: int) -> Array:
    """Compute rolling mean along axis 0."""
    kernel = jnp.ones(window) / window
    # Pad and convolve for each column
    if x.ndim == 1:
        padded = jnp.pad(x, (window - 1, 0), mode='edge')
        return jnp.convolve(padded, kernel, mode='valid')
    else:
        result = []
        for i in range(x.shape[1]):
            padded = jnp.pad(x[:, i], (window - 1, 0), mode='edge')
            result.append(jnp.convolve(padded, kernel, mode='valid'))
        return jnp.stack(result, axis=1)


def rolling_std(x: Array, window: int) -> Array:
    """Compute rolling std along axis 0."""
    mean = rolling_mean(x, window)
    mean_sq = rolling_mean(x ** 2, window)
    var = jnp.maximum(mean_sq - mean ** 2, 1e-10)
    return jnp.sqrt(var)


def compute_returns(prices: Array) -> Array:
    """Compute log returns from prices."""
    # prices: (timesteps, assets, features) or (timesteps, assets)
    if prices.ndim == 3:
        # Use close price (index 3) or first feature
        close_idx = min(3, prices.shape[-1] - 1)
        close = prices[:, :, close_idx]
    else:
        close = prices

    log_prices = jnp.log(jnp.maximum(close, 1e-10))
    returns = jnp.diff(log_prices, axis=0)
    return returns


def extract_trend_direction(returns: Array, window: int = 20) -> Array:
    """
    Extract trend direction per asset.

    Returns:
        Array of shape (n_timesteps-1, n_assets) with values in {-1, 0, +1}
    """
    smoothed = rolling_mean(returns, window)
    # Threshold for "sideways" - within 0.5 std of zero
    std = rolling_std(returns, window)
    threshold = 0.5 * std

    direction = jnp.where(
        smoothed > threshold, 1,
        jnp.where(smoothed < -threshold, -1, 0)
    )
    return direction.astype(jnp.int32)


def extract_vol_regime(returns: Array, window: int = 20) -> Array:
    """
    Extract volatility regime (normalized).

    Returns:
        Array of shape (n_timesteps-1,) with values in [0, 1]
    """
    vol = rolling_std(returns, window)
    # Average across assets
    avg_vol = vol.mean(axis=1)
    # Normalize to [0, 1] - guard against zero range
    range_val = jnp.maximum(avg_vol.max() - avg_vol.min(), 1e-10)
    vol_norm = (avg_vol - avg_vol.min()) / range_val
    return vol_norm


def extract_correlation_clusters(returns: Array, n_clusters: int = 5) -> Array:
    """
    Extract correlation-based asset clusters.

    Uses simple k-means on correlation features.
    Returns:
        Array of shape (n_assets,) with cluster IDs
    """
    n_assets = returns.shape[1]

    # Compute correlation matrix
    returns_np = np.array(returns)
    corr = np.corrcoef(returns_np.T)
    corr = np.nan_to_num(corr, nan=0.0)

    # Simple clustering: use correlation rows as features
    # K-means-like assignment based on correlation similarity
    # For simplicity, use hierarchical-ish approach

    # Initialize cluster centers randomly from assets
    rng = np.random.default_rng(42)
    center_indices = rng.choice(n_assets, n_clusters, replace=False)
    centers = corr[center_indices]

    # Assign each asset to nearest center
    for _ in range(10):  # Iterate
        # Compute distances
        distances = np.zeros((n_assets, n_clusters))
        for k in range(n_clusters):
            distances[:, k] = np.sum((corr - centers[k]) ** 2, axis=1)

        # Assign
        labels = distances.argmin(axis=1)

        # Update centers
        new_centers = np.zeros_like(centers)
        for k in range(n_clusters):
            mask = labels == k
            if mask.sum() > 0:
                new_centers[k] = corr[mask].mean(axis=0)
            else:
                new_centers[k] = centers[k]
        centers = new_centers

    return jnp.array(labels)


def extract_alpha_shape(returns: Array) -> Array:
    """
    Extract cumulative alpha opportunity shape.

    This represents the "ideal" cumulative return if trading optimally.
    Normalized to [0, 1].

    Returns:
        Array of shape (n_timesteps-1,)
    """
    # Optimal return at each step: take best asset's absolute return
    optimal_per_step = jnp.abs(returns).max(axis=1)
    cumulative = jnp.cumsum(optimal_per_step)

    # Normalize
    alpha_norm = cumulative / (cumulative[-1] + 1e-10)
    return alpha_norm


def detect_regime_sequence(returns: Array, n_states: int = 5) -> Array:
    """
    Detect regime sequence using simple heuristics.

    Maps to: crash(0), rally(1), sideways(2), volatile(3), calm(4)

    Returns:
        Array of shape (n_timesteps-1,) with regime indices
    """
    n_timesteps = returns.shape[0]

    # Compute features
    avg_return = returns.mean(axis=1)
    volatility = returns.std(axis=1)

    # Rolling features
    window = 20
    rolling_ret = rolling_mean(avg_return, window)
    rolling_vol = rolling_mean(volatility, window)

    # Normalize - guard against zero range
    vol_range = jnp.maximum(rolling_vol.max() - rolling_vol.min(), 1e-10)
    ret_range = jnp.maximum(rolling_ret.max() - rolling_ret.min(), 1e-10)
    vol_percentile = (rolling_vol - rolling_vol.min()) / vol_range
    ret_percentile = (rolling_ret - rolling_ret.min()) / ret_range

    # Classify regimes
    regimes = jnp.zeros(n_timesteps, dtype=jnp.int32)

    # crash: negative returns, high vol
    crash_mask = (ret_percentile < 0.3) & (vol_percentile > 0.6)

    # rally: positive returns, moderate vol
    rally_mask = (ret_percentile > 0.6) & (vol_percentile < 0.7)

    # volatile: high vol, neutral returns
    volatile_mask = (vol_percentile > 0.7) & ~crash_mask

    # calm: low vol
    calm_mask = vol_percentile < 0.3

    # sideways: everything else
    sideways_mask = ~(crash_mask | rally_mask | volatile_mask | calm_mask)

    regimes = jnp.where(crash_mask, 0, regimes)
    regimes = jnp.where(rally_mask, 1, regimes)
    regimes = jnp.where(sideways_mask, 2, regimes)
    regimes = jnp.where(volatile_mask, 3, regimes)
    regimes = jnp.where(calm_mask, 4, regimes)

    return regimes


def detect_events(returns: Array, threshold: float = 3.0) -> list:
    """
    Detect discrete events (outliers, spikes).

    Returns:
        List of Event objects
    """
    events = []

    # Global statistics
    mean = returns.mean()
    std = returns.std()

    # Detect outliers
    avg_returns = returns.mean(axis=1)
    z_scores = (avg_returns - mean) / (std + 1e-10)

    outlier_mask = jnp.abs(z_scores) > threshold
    outlier_indices = jnp.where(outlier_mask)[0]

    for idx in outlier_indices:
        z = float(z_scores[idx])
        events.append(Event(
            time=int(idx),
            event_type='spike' if z > 0 else 'crash',
            magnitude=min(abs(z) / 5.0, 1.0),  # Normalize to [0, 1]
            direction=1 if z > 0 else -1,
        ))

    return events


def extract_skeleton(
    prices: Array,
    window: int = 20,
    n_clusters: int = 5,
    detect_events_threshold: float = 3.0,
) -> MarketSkeleton:
    """
    Extract complete skeleton from price data.

    Args:
        prices: Array of shape (n_timesteps, n_assets, n_features) or (n_timesteps, n_assets)
        window: Rolling window for smoothing
        n_clusters: Number of correlation clusters
        detect_events_threshold: Z-score threshold for event detection

    Returns:
        MarketSkeleton with all components extracted
        Note: n_timesteps in skeleton is original price length, but arrays have length n-1
    """
    # Store original dimensions BEFORE diff reduces by 1
    original_n_timesteps = prices.shape[0]
    n_assets = prices.shape[1]

    returns = compute_returns(prices)

    trend_direction = extract_trend_direction(returns, window)
    vol_regime = extract_vol_regime(returns, window)
    correlation_clusters = extract_correlation_clusters(returns, n_clusters)
    alpha_shape = extract_alpha_shape(returns)
    regime_sequence = detect_regime_sequence(returns)
    events = detect_events(returns, detect_events_threshold)

    return MarketSkeleton(
        trend_direction=trend_direction,
        vol_regime=vol_regime,
        correlation_clusters=correlation_clusters,
        alpha_shape=alpha_shape,
        regime_sequence=regime_sequence,
        events=events,
        n_timesteps=original_n_timesteps,  # Original price length (arrays are n-1)
        n_assets=n_assets,
    )


def extract_skeleton_batch(
    prices_batch: Array,
    window: int = 20,
    n_clusters: int = 5,
) -> list:
    """
    Extract skeletons from batch of price data.

    Args:
        prices_batch: Array of shape (batch, n_timesteps, n_assets, n_features)

    Returns:
        List of MarketSkeleton objects
    """
    batch_size = prices_batch.shape[0]
    skeletons = []

    for i in range(batch_size):
        skeleton = extract_skeleton(prices_batch[i], window, n_clusters)
        skeletons.append(skeleton)

    return skeletons
