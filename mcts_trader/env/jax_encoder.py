"""
JAX-Vectorized 27-Channel State Encoder

Ports the FastMultiAssetStateEncoder to JAX for GPU-accelerated encoding.
Designed to work with the JAX trading environment for end-to-end GPU training.

Output tensor: (27, n_assets, n_assets)
Channels:
- 0-3: Normalized OHLC (latest values, tiled across width)
- 4-7: Log returns at 1, 5, 15, 60 min
- 8-11: Volume patterns (recent vs average ratios)
- 12-15: Price momentum (rate of change)
- 16-19: Volatility (rolling std of returns)
- 20-23: Price percentiles (position in recent range)
- 24-26: VWAP and bands

Key differences from Python version:
- No circular buffer - takes full price array + window indices
- Uses jnp instead of np for GPU acceleration
- JIT-compatible with jax.jit and jax.vmap
- Handles missing OHLCV gracefully (uses close for H/L, ones for volume)

Usage:
    # Single observation
    obs = encode_observation(prices, volumes, highs, lows, window_start, config)

    # Batched (multiple envs)
    batch_encode = jax.vmap(encode_observation, in_axes=(None, None, None, None, 0, None))
    obs_batch = batch_encode(prices, volumes, highs, lows, start_indices, config)

Author: Claude Code
Task: TASK_027_JAX_VECTORIZED_ENV.md Phase 3
"""

import jax
import jax.numpy as jnp
import chex
from functools import partial
from typing import Tuple, Optional


# =============================================================================
# Configuration
# =============================================================================

@chex.dataclass(frozen=True)
class EncoderConfig:
    """Configuration for the 32-channel encoder (TASK_732).

    27 per-asset-independent channels (OHLCV, returns, volatility, percentiles,
    VWAP, momentum) + 5 cross-sectional channels at the end (ranks, beta,
    idiosyncratic vol, xs momentum). Cross-sectional block gives the model
    explicit "where this asset stands vs the other 63" signal, absent pre-732.

    Attributes:
        n_assets: Number of assets (default 64)
        window_size: Lookback window for features (default 60)
        n_channels: Number of output channels (default 32 post-TASK_732)
        return_clip: Max return value before clipping (default 10.0)
        vol_scale: Volatility scaling factor (default 100.0)
        use_cheat_channel: TASK_733 — opt-in diagnostic. Adds channel 32 with
            sign-of-forward-log-return (perfect-information probe). Requires
            the caller to pass `future_prices` to encoder entry points and
            bump `n_channels` to ≥ 33. NEVER ship a checkpoint with this True.
        cheat_lookahead: Steps ahead for forward return (matches oracle_lookahead).
        cheat_epsilon: Dead-zone threshold on |r_fwd|; outside it → ±1, inside → 0.
    """
    n_assets: int = 64
    window_size: int = 60
    n_channels: int = 32
    return_clip: float = 10.0
    vol_scale: float = 100.0
    # TASK_733: cheat-channel diagnostic (opt-in, never for production)
    use_cheat_channel: bool = False
    cheat_lookahead: int = 20
    cheat_epsilon: float = 1e-4


# =============================================================================
# Core Encoding Functions
# =============================================================================

def encode_observation(
    prices: chex.Array,           # (n_steps, n_assets) - close prices
    volumes: Optional[chex.Array], # (n_steps, n_assets) or None
    highs: Optional[chex.Array],   # (n_steps, n_assets) or None
    lows: Optional[chex.Array],    # (n_steps, n_assets) or None
    window_start: chex.Array,      # () - start index of window
    config: EncoderConfig,
) -> chex.Array:
    """Encode market state into 27-channel observation tensor.

    If volumes/highs/lows are None, uses close prices or ones as fallback.
    This allows the encoder to work with synthetic data that only has prices.

    Args:
        prices: Close price array (n_steps, n_assets)
        volumes: Volume array or None (uses ones if None)
        highs: High price array or None (uses close if None)
        lows: Low price array or None (uses close if None)
        window_start: Starting index for the observation window
        config: Encoder configuration

    Returns:
        Observation tensor of shape (27, n_assets, n_assets)

    Raises:
        AssertionError: If window_size < 60 (required for all feature computations)
    """
    # Validate window size - features use 5, 15, 30, 60 step lookbacks
    assert config.window_size >= 60, (
        f"window_size must be >= 60 for all features, got {config.window_size}"
    )

    n = config.n_assets
    ws = config.window_size

    # Extract window
    price_window = jax.lax.dynamic_slice(
        prices, (window_start, 0), (ws, n)
    )  # (ws, n)

    # Handle optional arrays
    vol_window = jax.lax.dynamic_slice(
        volumes, (window_start, 0), (ws, n)
    ) if volumes is not None else jnp.ones((ws, n), dtype=jnp.float32)

    high_window = jax.lax.dynamic_slice(
        highs, (window_start, 0), (ws, n)
    ) if highs is not None else price_window

    low_window = jax.lax.dynamic_slice(
        lows, (window_start, 0), (ws, n)
    ) if lows is not None else price_window

    # Transpose to (n_assets, window_size) for per-asset operations
    prices_T = price_window.T  # (n, ws)
    volumes_T = vol_window.T   # (n, ws)
    highs_T = high_window.T    # (n, ws)
    lows_T = low_window.T      # (n, ws)

    # Compute features
    output = _compute_all_channels(
        prices_T, volumes_T, highs_T, lows_T, config
    )

    return output


def _compute_per_asset_channels(
    prices: chex.Array,   # (n_assets, window_size)
    volumes: chex.Array,  # (n_assets, window_size)
    highs: chex.Array,    # (n_assets, window_size)
    lows: chex.Array,     # (n_assets, window_size)
    config: EncoderConfig,
) -> Tuple[chex.Array, chex.Array]:
    """TASK_768: Universe-independent per-asset channels 0-26 (tiled path).

    Returns:
        output:     (n_channels, n, n) tensor with channels 0-26 populated and
                    channels 27+ left zero (cross-sectional + cheat are filled
                    by the wrapper).
        returns_1m: (n, ws) raw 1-step returns needed by the cross-sectional
                    block. Returning this avoids recomputing it downstream.

    Channel order is preserved bit-identically with the pre-768 monolithic
    `_compute_all_channels` for slots 0-26.
    """
    n = config.n_assets
    ws = config.window_size

    # Initialize output (n_channels, n, n)
    # TASK_689: Parameterized channel count (was hardcoded 27)
    output = jnp.zeros((config.n_channels, n, n), dtype=jnp.float32)

    # =========================================================================
    # Normalization
    # =========================================================================

    # Price normalization (0-1 over window)
    price_min = jnp.min(prices, axis=1, keepdims=True)  # (n, 1)
    price_max = jnp.max(prices, axis=1, keepdims=True)  # (n, 1)
    price_range = price_max - price_min
    price_range = jnp.where(price_range < 1e-8, 1.0, price_range)

    norm_prices = (prices - price_min) / price_range  # (n, ws)
    norm_highs = (highs - price_min) / price_range
    norm_lows = (lows - price_min) / price_range

    # Volume normalization (z-score)
    vol_mean = jnp.mean(volumes, axis=1, keepdims=True)  # (n, 1)
    vol_std = jnp.std(volumes, axis=1, keepdims=True) + 1e-8
    norm_volumes = jnp.clip((volumes - vol_mean) / vol_std, -3, 3) / 3.0  # [-1, 1]

    # =========================================================================
    # Channel 0-3: Latest OHLC (normalized, tiled across width)
    # =========================================================================

    output = output.at[0].set(jnp.tile(norm_prices[:, -1:], (1, n)))      # Close
    output = output.at[1].set(jnp.tile(norm_highs[:, -1:], (1, n)))       # High
    output = output.at[2].set(jnp.tile(norm_lows[:, -1:], (1, n)))        # Low
    output = output.at[3].set(jnp.tile(norm_volumes[:, -1:], (1, n)))     # Volume

    # =========================================================================
    # Channel 4-7: Log returns at 1, 5, 15, 60 min
    # =========================================================================

    # 1-min return: (price[t] - price[t-1]) / price[t-1]
    # TASK_122: Fixed denominator to use previous price, not current
    prev_prices = jnp.concatenate([prices[:, 0:1], prices[:, :-1]], axis=1)
    returns_1m = (prices - prev_prices) / (prev_prices + 1e-8)
    returns_1m = jnp.clip(returns_1m, -config.return_clip, config.return_clip)
    output = output.at[4].set(jnp.tile(returns_1m[:, -1:], (1, n)))

    # 5-min return
    returns_5m = (prices[:, -1] - prices[:, -5]) / (prices[:, -5] + 1e-8)
    returns_5m = jnp.clip(returns_5m, -config.return_clip, config.return_clip)
    output = output.at[5].set(jnp.tile(returns_5m[:, None], (1, n)))

    # 15-min return
    returns_15m = (prices[:, -1] - prices[:, -15]) / (prices[:, -15] + 1e-8)
    returns_15m = jnp.clip(returns_15m, -config.return_clip, config.return_clip)
    output = output.at[6].set(jnp.tile(returns_15m[:, None], (1, n)))

    # 60-min return (full window)
    returns_60m = (prices[:, -1] - prices[:, 0]) / (prices[:, 0] + 1e-8)
    returns_60m = jnp.clip(returns_60m, -config.return_clip, config.return_clip)
    output = output.at[7].set(jnp.tile(returns_60m[:, None], (1, n)))

    # =========================================================================
    # Channel 8-11: Volume patterns (recent vs average)
    # =========================================================================

    vol_ratio_5m = jnp.mean(volumes[:, -5:], axis=1) / (vol_mean[:, 0] + 1e-8)
    vol_ratio_15m = jnp.mean(volumes[:, -15:], axis=1) / (vol_mean[:, 0] + 1e-8)
    vol_ratio_30m = jnp.mean(volumes[:, -30:], axis=1) / (vol_mean[:, 0] + 1e-8)
    vol_ratio_60m = jnp.mean(volumes, axis=1) / (vol_mean[:, 0] + 1e-8)

    output = output.at[8].set(jnp.tile(jnp.clip(vol_ratio_5m[:, None], 0, 5) / 5.0, (1, n)))
    output = output.at[9].set(jnp.tile(jnp.clip(vol_ratio_15m[:, None], 0, 5) / 5.0, (1, n)))
    output = output.at[10].set(jnp.tile(jnp.clip(vol_ratio_30m[:, None], 0, 5) / 5.0, (1, n)))
    output = output.at[11].set(jnp.tile(jnp.clip(vol_ratio_60m[:, None], 0, 5) / 5.0, (1, n)))

    # =========================================================================
    # Channel 12-15: Price momentum (rate of change)
    # =========================================================================

    momentum_5 = (prices[:, -1] - prices[:, -5]) / (price_range[:, 0] + 1e-8)
    momentum_15 = (prices[:, -1] - prices[:, -15]) / (price_range[:, 0] + 1e-8)
    momentum_30 = (prices[:, -1] - prices[:, -30]) / (price_range[:, 0] + 1e-8)
    momentum_60 = (prices[:, -1] - prices[:, 0]) / (price_range[:, 0] + 1e-8)

    output = output.at[12].set(jnp.tile(jnp.clip(momentum_5[:, None], -1, 1), (1, n)))
    output = output.at[13].set(jnp.tile(jnp.clip(momentum_15[:, None], -1, 1), (1, n)))
    output = output.at[14].set(jnp.tile(jnp.clip(momentum_30[:, None], -1, 1), (1, n)))
    output = output.at[15].set(jnp.tile(jnp.clip(momentum_60[:, None], -1, 1), (1, n)))

    # =========================================================================
    # Channel 16-19: Volatility (rolling std of returns)
    # =========================================================================

    vol_5 = jnp.std(returns_1m[:, -5:], axis=1)
    vol_15 = jnp.std(returns_1m[:, -15:], axis=1)
    vol_30 = jnp.std(returns_1m[:, -30:], axis=1)
    vol_60 = jnp.std(returns_1m, axis=1)

    output = output.at[16].set(jnp.tile(jnp.clip(vol_5[:, None] * config.vol_scale, 0, 1), (1, n)))
    output = output.at[17].set(jnp.tile(jnp.clip(vol_15[:, None] * config.vol_scale, 0, 1), (1, n)))
    output = output.at[18].set(jnp.tile(jnp.clip(vol_30[:, None] * config.vol_scale, 0, 1), (1, n)))
    output = output.at[19].set(jnp.tile(jnp.clip(vol_60[:, None] * config.vol_scale, 0, 1), (1, n)))

    # =========================================================================
    # Channel 20-23: Price percentiles (position in recent range)
    # =========================================================================

    def compute_percentile(window_prices: chex.Array, window: int) -> chex.Array:
        """Compute where current price is in recent window range."""
        window_data = window_prices[:, -window:]
        w_min = jnp.min(window_data, axis=1)
        w_max = jnp.max(window_data, axis=1)
        w_range = w_max - w_min
        w_range = jnp.where(w_range < 1e-8, 1.0, w_range)
        return (window_prices[:, -1] - w_min) / w_range

    pct_5 = compute_percentile(prices, 5)
    pct_15 = compute_percentile(prices, 15)
    pct_30 = compute_percentile(prices, 30)
    pct_60 = compute_percentile(prices, 60)

    output = output.at[20].set(jnp.tile(pct_5[:, None], (1, n)))
    output = output.at[21].set(jnp.tile(pct_15[:, None], (1, n)))
    output = output.at[22].set(jnp.tile(pct_30[:, None], (1, n)))
    output = output.at[23].set(jnp.tile(pct_60[:, None], (1, n)))

    # =========================================================================
    # Channel 24-26: VWAP and bands
    # =========================================================================

    # VWAP over last 20 steps
    vwap_window = 20
    vwap_prices = prices[:, -vwap_window:]
    vwap_volumes = volumes[:, -vwap_window:]
    vol_sum = jnp.sum(vwap_volumes, axis=1, keepdims=True) + 1e-8
    vwap = jnp.sum(vwap_prices * vwap_volumes, axis=1) / vol_sum[:, 0]
    vwap_norm = (vwap - price_min[:, 0]) / (price_range[:, 0] + 1e-8)

    # Bands (±1 std from VWAP)
    price_std = jnp.std(vwap_prices, axis=1)
    upper_band = (vwap + price_std - price_min[:, 0]) / (price_range[:, 0] + 1e-8)
    lower_band = (vwap - price_std - price_min[:, 0]) / (price_range[:, 0] + 1e-8)

    output = output.at[24].set(jnp.tile(vwap_norm[:, None], (1, n)))
    output = output.at[25].set(jnp.tile(jnp.clip(upper_band[:, None], 0, 1), (1, n)))
    output = output.at[26].set(jnp.tile(jnp.clip(lower_band[:, None], 0, 1), (1, n)))

    return output, returns_1m


def _compute_cross_sectional(
    prices: chex.Array,      # (n_assets, window_size) — raw (un-normalized)
    returns_1m: chex.Array,  # (n_assets, window_size) — clipped 1-step returns
    n: int,                  # runtime asset-axis size; SPEC: n is a runtime arg
    config: EncoderConfig,
) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
    """TASK_768: Cross-sectional channels 27-31 (tiled path).

    Float32 discipline (SPEC risk #1): all inputs and outputs stay float32.
    Channel order (SPEC risk #2): callers MUST place these at slots 27-31 in
    that exact order: rank_1, rank_20, beta_scaled, idio_vol_scaled, xs_mom.

    Computation is bit-identical to the pre-768 monolithic implementation
    (same formulas, same intermediates, same clipping constants).
    """
    n_f = max(float(n - 1), 1.0)
    lookback_20 = min(20, prices.shape[-1] - 1)

    # Channel 27: rank of current 1-step return
    cur_r1 = returns_1m[:, -1]  # (n,)
    rank_1 = jnp.argsort(jnp.argsort(cur_r1)).astype(jnp.float32) / n_f

    # Channel 28: rank of 20-step (or ws-1 if window shorter) cumulative return
    r20 = (prices[:, -1] - prices[:, -lookback_20]) / (prices[:, -lookback_20] + 1e-8)
    r20 = jnp.clip(r20, -config.return_clip, config.return_clip)
    rank_20 = jnp.argsort(jnp.argsort(r20)).astype(jnp.float32) / n_f

    # Channel 29: rolling beta to equal-weighted universe (full window)
    uni_rets = jnp.mean(returns_1m, axis=0)
    asset_centered = returns_1m - jnp.mean(returns_1m, axis=1, keepdims=True)
    uni_centered = uni_rets - jnp.mean(uni_rets)
    cov = jnp.mean(asset_centered * uni_centered[None, :], axis=1)
    var_uni = jnp.mean(uni_centered ** 2)
    beta_raw = cov / (var_uni + 1e-8)
    beta_scaled = jnp.clip(beta_raw, -3.0, 3.0) / 3.0

    # Channel 30: idiosyncratic vol — std of residuals after removing beta*universe
    residuals = returns_1m - beta_raw[:, None] * uni_rets[None, :]
    idio_vol = jnp.std(residuals, axis=1)
    idio_vol_scaled = jnp.clip(idio_vol * config.vol_scale, 0, 1)

    # Channel 31: cross-sectional momentum — current return minus universe mean
    uni_mean_cur = jnp.mean(cur_r1)
    xs_mom = jnp.clip((cur_r1 - uni_mean_cur) * 100.0, -1.0, 1.0)

    return rank_1, rank_20, beta_scaled, idio_vol_scaled, xs_mom


def _compute_all_channels(
    prices: chex.Array,   # (n_assets, window_size)
    volumes: chex.Array,  # (n_assets, window_size)
    highs: chex.Array,    # (n_assets, window_size)
    lows: chex.Array,     # (n_assets, window_size)
    config: EncoderConfig,
) -> chex.Array:
    """Compute all 27/32 channels of the observation tensor (tiled path).

    TASK_768: Thin wrapper — per-asset slots 0-26 are computed by
    `_compute_per_asset_channels` (universe-independent); cross-sectional
    slots 27-31 are computed by `_compute_cross_sectional` (universe-aware,
    takes `n` as a runtime arg). Output tensor is bit-identical to the
    pre-768 monolithic implementation.

    Channel order invariant (DO NOT CHANGE): `[0-26 per-asset][27-31 cross-sectional]`.
    The base imprint's `temporal_proj.kernel` (trm_encoder_v3.py:141, shape
    (39, 32)) binds each column to a channel index; reordering would force
    a re-imprint.

    All operations are vectorized across assets using JAX primitives.
    """
    n = config.n_assets

    # Per-asset slots 0-26 + raw 1-step returns
    output, returns_1m = _compute_per_asset_channels(
        prices, volumes, highs, lows, config
    )

    # Cross-sectional slots 27-31 (universe-coupled)
    if config.n_channels >= 32:
        # TASK_738: skip in legacy 27-ch mode
        rank_1, rank_20, beta_scaled, idio_vol_scaled, xs_mom = _compute_cross_sectional(
            prices, returns_1m, n, config
        )
        output = output.at[27].set(jnp.tile(rank_1[:, None], (1, n)))
        output = output.at[28].set(jnp.tile(rank_20[:, None], (1, n)))
        output = output.at[29].set(jnp.tile(beta_scaled[:, None], (1, n)))
        output = output.at[30].set(jnp.tile(idio_vol_scaled[:, None], (1, n)))
        output = output.at[31].set(jnp.tile(xs_mom[:, None], (1, n)))

    return output


# =============================================================================
# TASK_152: Temporal Encoder (Untiled - Real Time Series)
# =============================================================================

def _compute_per_asset_channels_temporal(
    prices: chex.Array,   # (n_assets, window_size)
    volumes: chex.Array,  # (n_assets, window_size)
    highs: chex.Array,    # (n_assets, window_size)
    lows: chex.Array,     # (n_assets, window_size)
    config: EncoderConfig,
) -> Tuple[chex.Array, chex.Array]:
    """TASK_768: Universe-independent per-asset channels 0-26 (temporal path).

    Returns:
        output:     (n_channels, ws, n) tensor with channels 0-26 populated and
                    channels 27+ left zero (cross-sectional + cheat are filled
                    by the wrapper).
        returns_1m: (n, ws) raw 1-step returns needed by the cross-sectional
                    block. Returning this avoids recomputing it downstream.

    Channel order is preserved bit-identically with the pre-768 monolithic
    `_compute_all_channels_temporal` for slots 0-26.
    """
    n = config.n_assets
    ws = config.window_size

    # Initialize output (n_channels, ws, n) - TIME × ASSETS
    # TASK_689: Parameterized channel count (was hardcoded 27)
    output = jnp.zeros((config.n_channels, ws, n), dtype=jnp.float32)

    # =========================================================================
    # Normalization (same as original)
    # =========================================================================
    price_min = jnp.min(prices, axis=1, keepdims=True)  # (n, 1)
    price_max = jnp.max(prices, axis=1, keepdims=True)  # (n, 1)
    price_range = price_max - price_min
    price_range = jnp.where(price_range < 1e-8, 1.0, price_range)

    norm_prices = (prices - price_min) / price_range  # (n, ws)
    norm_highs = (highs - price_min) / price_range
    norm_lows = (lows - price_min) / price_range

    vol_mean = jnp.mean(volumes, axis=1, keepdims=True)
    vol_std = jnp.std(volumes, axis=1, keepdims=True) + 1e-8
    norm_volumes = jnp.clip((volumes - vol_mean) / vol_std, -3, 3) / 3.0

    # =========================================================================
    # Channels 0-3: FULL TEMPORAL OHLC (not just latest!)
    # Shape: (n, ws).T = (ws, n) - each row is a time step
    # =========================================================================
    output = output.at[0].set(norm_prices.T)    # Close prices over time
    output = output.at[1].set(norm_highs.T)     # High prices over time
    output = output.at[2].set(norm_lows.T)      # Low prices over time
    output = output.at[3].set(norm_volumes.T)   # Volume over time

    # =========================================================================
    # Channel 4: FULL 1-minute returns time series
    # =========================================================================
    prev_prices = jnp.concatenate([prices[:, 0:1], prices[:, :-1]], axis=1)
    returns_1m = (prices - prev_prices) / (prev_prices + 1e-8)
    returns_1m = jnp.clip(returns_1m, -config.return_clip, config.return_clip)
    output = output.at[4].set(returns_1m.T)     # 1m returns over time

    # =========================================================================
    # Channels 5-26: Scalar indicators (broadcast across time)
    # These are computed at the CURRENT time step, then broadcast
    # =========================================================================

    # Helper to broadcast scalar (n,) to (ws, n)
    def broadcast_scalar(x):
        return jnp.broadcast_to(x[None, :], (ws, n))

    # Channel 5-7: Multi-horizon returns (at current time)
    returns_5m = (prices[:, -1] - prices[:, -5]) / (prices[:, -5] + 1e-8)
    returns_15m = (prices[:, -1] - prices[:, -15]) / (prices[:, -15] + 1e-8)
    returns_64m = (prices[:, -1] - prices[:, 0]) / (prices[:, 0] + 1e-8)  # Full window

    output = output.at[5].set(broadcast_scalar(jnp.clip(returns_5m, -config.return_clip, config.return_clip)))
    output = output.at[6].set(broadcast_scalar(jnp.clip(returns_15m, -config.return_clip, config.return_clip)))
    output = output.at[7].set(broadcast_scalar(jnp.clip(returns_64m, -config.return_clip, config.return_clip)))

    # Channel 8-11: Volume patterns
    vol_ratio_5m = jnp.mean(volumes[:, -5:], axis=1) / (vol_mean[:, 0] + 1e-8)
    vol_ratio_15m = jnp.mean(volumes[:, -15:], axis=1) / (vol_mean[:, 0] + 1e-8)
    vol_ratio_30m = jnp.mean(volumes[:, -30:], axis=1) / (vol_mean[:, 0] + 1e-8)
    vol_ratio_64m = jnp.mean(volumes, axis=1) / (vol_mean[:, 0] + 1e-8)

    output = output.at[8].set(broadcast_scalar(jnp.clip(vol_ratio_5m, 0, 5) / 5.0))
    output = output.at[9].set(broadcast_scalar(jnp.clip(vol_ratio_15m, 0, 5) / 5.0))
    output = output.at[10].set(broadcast_scalar(jnp.clip(vol_ratio_30m, 0, 5) / 5.0))
    output = output.at[11].set(broadcast_scalar(jnp.clip(vol_ratio_64m, 0, 5) / 5.0))

    # Channel 12-15: Momentum
    momentum_5 = (prices[:, -1] - prices[:, -5]) / (price_range[:, 0] + 1e-8)
    momentum_15 = (prices[:, -1] - prices[:, -15]) / (price_range[:, 0] + 1e-8)
    momentum_30 = (prices[:, -1] - prices[:, -30]) / (price_range[:, 0] + 1e-8)
    momentum_64 = (prices[:, -1] - prices[:, 0]) / (price_range[:, 0] + 1e-8)

    output = output.at[12].set(broadcast_scalar(jnp.clip(momentum_5, -1, 1)))
    output = output.at[13].set(broadcast_scalar(jnp.clip(momentum_15, -1, 1)))
    output = output.at[14].set(broadcast_scalar(jnp.clip(momentum_30, -1, 1)))
    output = output.at[15].set(broadcast_scalar(jnp.clip(momentum_64, -1, 1)))

    # Channel 16-19: Volatility
    vol_5 = jnp.std(returns_1m[:, -5:], axis=1)
    vol_15 = jnp.std(returns_1m[:, -15:], axis=1)
    vol_30 = jnp.std(returns_1m[:, -30:], axis=1)
    vol_64 = jnp.std(returns_1m, axis=1)

    output = output.at[16].set(broadcast_scalar(jnp.clip(vol_5 * config.vol_scale, 0, 1)))
    output = output.at[17].set(broadcast_scalar(jnp.clip(vol_15 * config.vol_scale, 0, 1)))
    output = output.at[18].set(broadcast_scalar(jnp.clip(vol_30 * config.vol_scale, 0, 1)))
    output = output.at[19].set(broadcast_scalar(jnp.clip(vol_64 * config.vol_scale, 0, 1)))

    # Channel 20-23: Price percentiles
    def compute_percentile(window_prices, window):
        window_data = window_prices[:, -window:]
        w_min = jnp.min(window_data, axis=1)
        w_max = jnp.max(window_data, axis=1)
        w_range = jnp.where((w_max - w_min) < 1e-8, 1.0, w_max - w_min)
        return (window_prices[:, -1] - w_min) / w_range

    output = output.at[20].set(broadcast_scalar(compute_percentile(prices, 5)))
    output = output.at[21].set(broadcast_scalar(compute_percentile(prices, 15)))
    output = output.at[22].set(broadcast_scalar(compute_percentile(prices, 30)))
    output = output.at[23].set(broadcast_scalar(compute_percentile(prices, 64)))

    # Channel 24-26: VWAP and bands
    vwap_window = 20
    vwap_prices = prices[:, -vwap_window:]
    vwap_volumes = volumes[:, -vwap_window:]
    vol_sum = jnp.sum(vwap_volumes, axis=1, keepdims=True) + 1e-8
    vwap = jnp.sum(vwap_prices * vwap_volumes, axis=1) / vol_sum[:, 0]
    vwap_norm = (vwap - price_min[:, 0]) / (price_range[:, 0] + 1e-8)

    price_std = jnp.std(vwap_prices, axis=1)
    upper_band = (vwap + price_std - price_min[:, 0]) / (price_range[:, 0] + 1e-8)
    lower_band = (vwap - price_std - price_min[:, 0]) / (price_range[:, 0] + 1e-8)

    output = output.at[24].set(broadcast_scalar(vwap_norm))
    output = output.at[25].set(broadcast_scalar(jnp.clip(upper_band, 0, 1)))
    output = output.at[26].set(broadcast_scalar(jnp.clip(lower_band, 0, 1)))

    return output, returns_1m


def _compute_cross_sectional_temporal(
    prices: chex.Array,      # (n_assets, window_size) — raw (un-normalized)
    returns_1m: chex.Array,  # (n_assets, window_size) — clipped 1-step returns
    n: int,                  # runtime asset-axis size; SPEC: n is a runtime arg
    ws: int,                 # runtime window size (used for the lookback_20 bound)
    config: EncoderConfig,
) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
    """TASK_768: Cross-sectional channels 27-31 (temporal path).

    Float32 discipline (SPEC risk #1): all inputs and outputs stay float32.
    `prices` and `returns_1m` come from the per-asset preprocessing function
    which runs on float32 OHLCV slices; nothing downcasts before this point.
    If the wrapper is ever invoked from inside a bf16 jit context, callers
    must cast back to float32 before concat (no such site exists today).

    Channel order (SPEC risk #2): the wrapper places these at slots 27-31 in
    this exact order: rank_1, rank_20, beta_scaled, idio_vol_scaled, xs_mom.

    Computation is bit-identical to the pre-768 monolithic implementation
    (same formulas, same intermediates, same clipping constants).
    """
    n_f = max(float(n - 1), 1.0)
    lookback_20 = min(20, ws - 1)  # safe index for short windows

    # Channel 27: rank of current 1-step return
    cur_r1 = returns_1m[:, -1]  # (n,)
    rank_1 = jnp.argsort(jnp.argsort(cur_r1)).astype(jnp.float32) / n_f

    # Channel 28: rank of 20-step (or ws-1 if window shorter) cumulative return
    r20 = (prices[:, -1] - prices[:, -lookback_20]) / (prices[:, -lookback_20] + 1e-8)
    r20 = jnp.clip(r20, -config.return_clip, config.return_clip)
    rank_20 = jnp.argsort(jnp.argsort(r20)).astype(jnp.float32) / n_f

    # Channel 29: rolling beta to equal-weighted universe (full window)
    uni_rets = jnp.mean(returns_1m, axis=0)                     # (ws,)
    asset_centered = returns_1m - jnp.mean(returns_1m, axis=1, keepdims=True)  # (n, ws)
    uni_centered = uni_rets - jnp.mean(uni_rets)                # (ws,)
    cov = jnp.mean(asset_centered * uni_centered[None, :], axis=1)  # (n,)
    var_uni = jnp.mean(uni_centered ** 2)                       # scalar
    beta_raw = cov / (var_uni + 1e-8)                           # (n,) unclipped beta
    beta_scaled = jnp.clip(beta_raw, -3.0, 3.0) / 3.0           # scale to [-1, 1]

    # Channel 30: idiosyncratic vol — std of residuals after removing beta*universe
    residuals = returns_1m - beta_raw[:, None] * uni_rets[None, :]  # (n, ws)
    idio_vol = jnp.std(residuals, axis=1)                        # (n,)
    idio_vol_scaled = jnp.clip(idio_vol * config.vol_scale, 0, 1)

    # Channel 31: cross-sectional momentum — current return minus universe mean
    uni_mean_cur = jnp.mean(cur_r1)
    xs_mom = jnp.clip((cur_r1 - uni_mean_cur) * 100.0, -1.0, 1.0)

    return rank_1, rank_20, beta_scaled, idio_vol_scaled, xs_mom


def _compute_all_channels_temporal(
    prices: chex.Array,   # (n_assets, window_size)
    volumes: chex.Array,  # (n_assets, window_size)
    highs: chex.Array,    # (n_assets, window_size)
    lows: chex.Array,     # (n_assets, window_size)
    config: EncoderConfig,
    future_close: Optional[chex.Array] = None,  # (n_assets,) close at t+ws+lookahead-1
) -> chex.Array:
    """Compute 32-channel observation with REAL temporal data.

    TASK_152: Unlike _compute_all_channels which tiles scalar features,
    this function preserves temporal information:
    - Channels 0-4:  Full time series (normalized OHLC + 1m returns)
    - Channels 5-26: Per-asset scalar indicators broadcast across time
    - Channels 27-31 (TASK_732): Cross-sectional scalars broadcast across time
      (rank-1step, rank-20step, beta-to-universe, idio-vol, xs-momentum)
    - Channel 32 (TASK_733, opt-in): sign of forward log-return in {−1, 0, +1}
      with dead-zone eps. Cheat input — never in production.

    Output shape: (32, window_size, n_assets) - rows are TIME, cols are ASSETS.

    TASK_768: Thin wrapper — per-asset slots 0-26 are computed by
    `_compute_per_asset_channels_temporal` (universe-independent); cross-sectional
    slots 27-31 are computed by `_compute_cross_sectional_temporal` (universe-aware,
    takes `n` as a runtime arg). Output tensor is bit-identical to the
    pre-768 monolithic implementation. Cheat channel 32 stays in this wrapper
    because it consumes `future_close` from the encoder entry point.

    Channel order invariant (DO NOT CHANGE): `[0-26 per-asset][27-31 cross-sectional]
    [32 cheat-opt-in][33+ agent-state appended downstream]`. The base imprint's
    `temporal_proj.kernel` (trm_encoder_v3.py:141, shape (39, 32)) binds each column
    to a channel index; reordering would force a re-imprint.
    """
    n = config.n_assets
    ws = config.window_size

    # Helper to broadcast scalar (n,) to (ws, n) — also used by the cheat write below.
    def broadcast_scalar(x):
        return jnp.broadcast_to(x[None, :], (ws, n))

    # Per-asset slots 0-26 + raw 1-step returns
    output, returns_1m = _compute_per_asset_channels_temporal(
        prices, volumes, highs, lows, config
    )

    # Cross-sectional slots 27-31 (universe-coupled; n + ws as runtime args)
    if config.n_channels >= 32:
        # TASK_738: legacy 27-ch mode skips cross-sectional channels.
        # Only write them when the output array has room (32+ channels).
        rank_1, rank_20, beta_scaled, idio_vol_scaled, xs_mom = _compute_cross_sectional_temporal(
            prices, returns_1m, n, ws, config
        )
        output = output.at[27].set(broadcast_scalar(rank_1))
        output = output.at[28].set(broadcast_scalar(rank_20))
        output = output.at[29].set(broadcast_scalar(beta_scaled))
        output = output.at[30].set(broadcast_scalar(idio_vol_scaled))
        output = output.at[31].set(broadcast_scalar(xs_mom))

    # =========================================================================
    # Channel 32: CHEAT CHANNEL (TASK_733) — opt-in diagnostic
    # Sign of forward log-return with dead-zone. Pure future information —
    # NEVER enable for production checkpoints; benchmark pipeline refuses them.
    # Computed AFTER other channels so cheat values cannot leak into the
    # window-normalized price_min/price_max used by channels 0-4.
    # =========================================================================
    if config.use_cheat_channel and config.n_channels >= 33:
        # future_close shape (n_assets,). Guarded log via jnp.maximum to match
        # the zero-price-gap handling pattern elsewhere (TASK_705).
        # Padding sentinel: callers pass a zero tensor past end-of-series; a
        # non-positive future close is unambiguously "no data" (real OHLCV
        # prices are always > 0). Treat it as HOLD (sign=0) so the cheat
        # channel never emits a spurious direction from zero-padded tails.
        cur_close = prices[:, -1]  # (n,)
        safe_cur = jnp.maximum(cur_close, 1e-8)
        safe_fut = jnp.maximum(future_close, 1e-8)
        r_fwd = jnp.log(safe_fut / safe_cur)  # (n,)
        eps = config.cheat_epsilon
        raw_sign = jnp.where(
            r_fwd > eps, 1.0,
            jnp.where(r_fwd < -eps, -1.0, 0.0)
        )
        valid_future = future_close > 0.0
        cheat_sign = jnp.where(valid_future, raw_sign, 0.0)  # (n,) in {-1, 0, +1}
        output = output.at[32].set(broadcast_scalar(cheat_sign))

    return output


def encode_observation_temporal(
    prices: chex.Array,        # (n_steps, n_assets, 5) OHLCV or (n_steps, n_assets) legacy
    window_start: chex.Array,  # ()
    config: EncoderConfig,
    future_prices: Optional[chex.Array] = None,  # TASK_733 cheat input
) -> chex.Array:
    """Encode observation using temporal encoder (TASK_152 + TASK_732).

    TASK_703: Accepts full OHLCV (n_steps, n_assets, 5) or legacy close-only (n_steps, n_assets).
    Returns (32, window_size, n_assets) with real time series data
    (27 per-asset channels + 5 cross-sectional channels).

    TASK_733: When `config.use_cheat_channel=True`, the caller MUST pass
    `future_prices` of shape (lookahead, n_assets, 5) or (lookahead, n_assets)
    covering [window_start + window_size, window_start + window_size + lookahead).
    The last row's close drives the sign-of-forward-log-return in channel 32.
    """
    n = config.n_assets
    ws = config.window_size

    # TASK_703: Handle both OHLCV and legacy close-only
    if prices.ndim == 3:
        # Full OHLCV: (n_steps, n_assets, 5) → O=0, H=1, L=2, C=3, V=4
        ohlcv_window = jax.lax.dynamic_slice(prices, (window_start, 0, 0), (ws, n, 5))
        close_T = ohlcv_window[:, :, 3].T   # (n_assets, ws)
        high_T = ohlcv_window[:, :, 1].T
        low_T = ohlcv_window[:, :, 2].T
        volume_T = ohlcv_window[:, :, 4].T
    else:
        # Legacy close-only: (n_steps, n_assets)
        price_window = jax.lax.dynamic_slice(prices, (window_start, 0), (ws, n))
        close_T = price_window.T
        high_T = close_T
        low_T = close_T
        volume_T = jnp.ones_like(close_T)

    # TASK_733: extract cheat close from the last row of the forward window.
    # `future_prices` layout mirrors `prices`: OHLCV-3D or close-only-2D.
    future_close = None
    if config.use_cheat_channel:
        if future_prices is None:
            raise ValueError(
                "EncoderConfig.use_cheat_channel=True but future_prices is None. "
                "Caller must pass a (lookahead, n_assets[, 5]) forward-window "
                "tensor covering [t+window_size, t+window_size+lookahead)."
            )
        if future_prices.ndim == 3:
            future_close = future_prices[-1, :, 3]  # (n_assets,)
        else:
            future_close = future_prices[-1, :]

    return _compute_all_channels_temporal(
        close_T, volume_T, high_T, low_T, config, future_close=future_close
    )


@partial(jax.jit, static_argnames=['config'])
def jit_encode_temporal(
    prices: chex.Array,        # (n_steps, n_assets, 5) OHLCV or (n_steps, n_assets)
    window_start: chex.Array,
    config: EncoderConfig,
    future_prices: Optional[chex.Array] = None,  # TASK_733: (lookahead, n_assets, 5) or (lookahead, n_assets)
) -> chex.Array:
    """JIT-compiled temporal encoder. TASK_703: accepts OHLCV or close-only.

    TASK_733: `future_prices` is traced (dynamic) — not a static arg. When the
    cheat channel is disabled (default), pass None; when enabled, pass a
    (lookahead, n_assets[, 5]) slice covering the forward window.
    """
    return encode_observation_temporal(prices, window_start, config, future_prices)


@partial(jax.jit, static_argnames=['config'])
def jit_batch_encode_temporal(
    prices: chex.Array,        # (n_steps, n_assets, 5) OHLCV or (n_steps, n_assets)
    window_starts: chex.Array, # (batch,)
    config: EncoderConfig,
    future_prices: Optional[chex.Array] = None,  # TASK_733: (batch, lookahead, n_assets[, 5])
) -> chex.Array:
    """JIT-compiled batched temporal encoder. TASK_703: accepts OHLCV or close-only.

    Returns (batch, 27, window_size, n_assets).

    TASK_733: `future_prices` (if supplied) must have a leading batch dim
    matching `window_starts`; each slot covers the forward window for its
    corresponding window start.
    """
    if future_prices is None:
        return jax.vmap(
            lambda ws: encode_observation_temporal(prices, ws, config)
        )(window_starts)
    return jax.vmap(
        lambda ws, fp: encode_observation_temporal(prices, ws, config, fp)
    )(window_starts, future_prices)


@partial(jax.jit, static_argnames=['config'])
def jit_vmap_encode_temporal(
    prices_batch: chex.Array,      # (N, n_steps, n_assets, 5) or (N, n_steps, n_assets)
    window_starts: chex.Array,     # (L,) - shared window starts
    config: EncoderConfig,
    future_prices_batch: Optional[chex.Array] = None,  # TASK_733: (N, L, lookahead, n_assets[, 5])
) -> chex.Array:
    """Encode multiple environments in parallel via vmap (TASK_152 temporal).
    TASK_703: accepts OHLCV (N, n_steps, n_assets, 5) or close-only (N, n_steps, n_assets).

    Returns:
        (N, L, 27, window_size, n_assets) encoded temporal observations

    TASK_733: `future_prices_batch` (if supplied) mirrors `prices_batch`'s
    outer N axis plus the shared L axis across window_starts.
    """
    # Double vmap: outer over episodes (N), inner over windows (L)
    if future_prices_batch is None:
        return jax.vmap(
            lambda p: jit_batch_encode_temporal(p, window_starts, config)
        )(prices_batch)
    return jax.vmap(
        lambda p, fp: jit_batch_encode_temporal(p, window_starts, config, fp)
    )(prices_batch, future_prices_batch)


# =============================================================================
# Extended Observation with Portfolio State
# =============================================================================

def encode_observation_with_state(
    prices: chex.Array,
    volumes: Optional[chex.Array],
    highs: Optional[chex.Array],
    lows: Optional[chex.Array],
    window_start: chex.Array,
    pos_active: chex.Array,      # (n_assets,) bool
    pos_side: chex.Array,        # (n_assets,) int: 0/1/2
    pos_entry: chex.Array,       # (n_assets,) float
    order_active: chex.Array,    # (n_assets,) bool
    order_side: chex.Array,      # (n_assets,) int: 0/1/2
    config: EncoderConfig,
) -> chex.Array:
    """Encode observation with additional portfolio state channels.

    Extends the 27-channel market observation with 4 portfolio channels:
    - Channel 27: Position indicator (-1=short, 0=none, +1=long)
    - Channel 28: Position PnL (normalized)
    - Channel 29: Order indicator (-1=sell, 0=none, +1=buy)
    - Channel 30: Order proximity (how close limit to current price)

    Returns: (31, n_assets, n_assets)
    """
    n = config.n_assets

    # Get base 27-channel encoding
    market_obs = encode_observation(
        prices, volumes, highs, lows, window_start, config
    )  # (27, n, n)

    # Get current price for PnL calculation
    current_price = jax.lax.dynamic_slice(
        prices, (window_start + config.window_size - 1, 0), (1, n)
    )[0]  # (n,)

    # Portfolio channels
    # Channel 27: Position indicator
    position_indicator = jnp.where(
        pos_active,
        jnp.where(pos_side == 1, 1.0, -1.0),
        0.0
    )

    # Channel 28: Position PnL (percentage, clipped to [-1, 1])
    pnl_long = (current_price - pos_entry) / (pos_entry + 1e-8)
    pnl_short = (pos_entry - current_price) / (pos_entry + 1e-8)
    pnl = jnp.where(pos_side == 1, pnl_long,
                   jnp.where(pos_side == 2, pnl_short, 0.0))
    pnl_normalized = jnp.clip(pnl * 10, -1, 1)  # 10% = 1.0
    pnl_normalized = jnp.where(pos_active, pnl_normalized, 0.0)

    # Channel 29: Order indicator
    order_indicator = jnp.where(
        order_active,
        jnp.where(order_side == 1, 1.0, -1.0),
        0.0
    )

    # Channel 30: Placeholder (order proximity could go here)
    order_proximity = jnp.zeros(n, dtype=jnp.float32)

    # Expand and tile portfolio channels
    portfolio_channels = jnp.stack([
        jnp.tile(position_indicator[:, None], (1, n)),
        jnp.tile(pnl_normalized[:, None], (1, n)),
        jnp.tile(order_indicator[:, None], (1, n)),
        jnp.tile(order_proximity[:, None], (1, n)),
    ], axis=0)  # (4, n, n)

    # Concatenate
    full_obs = jnp.concatenate([market_obs, portfolio_channels], axis=0)  # (31, n, n)

    return full_obs


# =============================================================================
# JIT-Compiled Versions
# =============================================================================

@partial(jax.jit, static_argnames=['config'])
def jit_encode_observation(
    prices: chex.Array,
    volumes: Optional[chex.Array],
    highs: Optional[chex.Array],
    lows: Optional[chex.Array],
    window_start: chex.Array,
    config: EncoderConfig,
) -> chex.Array:
    """JIT-compiled observation encoding."""
    return encode_observation(prices, volumes, highs, lows, window_start, config)


@partial(jax.jit, static_argnames=['config'])
def jit_encode_observation_with_state(
    prices: chex.Array,
    volumes: Optional[chex.Array],
    highs: Optional[chex.Array],
    lows: Optional[chex.Array],
    window_start: chex.Array,
    pos_active: chex.Array,
    pos_side: chex.Array,
    pos_entry: chex.Array,
    order_active: chex.Array,
    order_side: chex.Array,
    config: EncoderConfig,
) -> chex.Array:
    """JIT-compiled observation encoding with portfolio state."""
    return encode_observation_with_state(
        prices, volumes, highs, lows, window_start,
        pos_active, pos_side, pos_entry, order_active, order_side,
        config
    )


# =============================================================================
# Batched Operations
# =============================================================================

def batch_encode_observation(
    prices: chex.Array,
    volumes: Optional[chex.Array],
    highs: Optional[chex.Array],
    lows: Optional[chex.Array],
    window_starts: chex.Array,  # (batch_size,)
    config: EncoderConfig,
) -> chex.Array:
    """Encode observations for multiple environments."""
    return jax.vmap(
        lambda ws: encode_observation(prices, volumes, highs, lows, ws, config)
    )(window_starts)


@partial(jax.jit, static_argnames=['config'])
def jit_batch_encode_observation(
    prices: chex.Array,
    volumes: Optional[chex.Array],
    highs: Optional[chex.Array],
    lows: Optional[chex.Array],
    window_starts: chex.Array,
    config: EncoderConfig,
) -> chex.Array:
    """JIT-compiled batched observation encoding."""
    return batch_encode_observation(
        prices, volumes, highs, lows, window_starts, config
    )


# =============================================================================
# Simplified Encoder for Synthetic Data
# =============================================================================

def encode_observation_simple(
    prices: chex.Array,        # (n_steps, n_assets)
    window_start: chex.Array,  # ()
    config: EncoderConfig,
) -> chex.Array:
    """Simplified encoder for synthetic data (prices only).

    Uses prices for OHLC and ones for volume.
    This is the recommended encoder for synthetic curriculum training.
    """
    return encode_observation(
        prices=prices,
        volumes=None,  # Will use ones
        highs=None,    # Will use prices
        lows=None,     # Will use prices
        window_start=window_start,
        config=config,
    )


@partial(jax.jit, static_argnames=['config'])
def jit_encode_simple(
    prices: chex.Array,
    window_start: chex.Array,
    config: EncoderConfig,
) -> chex.Array:
    """JIT-compiled simplified encoder."""
    return encode_observation_simple(prices, window_start, config)


def batch_encode_simple(
    prices: chex.Array,
    window_starts: chex.Array,
    config: EncoderConfig,
) -> chex.Array:
    """Batched simplified encoder."""
    return jax.vmap(
        lambda ws: encode_observation_simple(prices, ws, config)
    )(window_starts)


@partial(jax.jit, static_argnames=['config', 'per_env'])
def jit_batch_encode_simple(
    prices: chex.Array,
    window_starts: chex.Array,
    config: EncoderConfig,
    per_env: bool = False,
) -> chex.Array:
    """JIT-compiled batched simplified encoder.

    Args:
        prices: (n_steps, n_assets) for shared mode, (n_envs, n_steps, n_assets) for per_env mode
        window_starts: (n_envs,) array of starting indices
        config: Encoder configuration
        per_env: If True, prices is (n_envs, n_steps, n_assets) with unique data per env

    Returns:
        (n_envs, n_channels, n_assets, n_assets) encoded observations
    """
    if per_env:
        # Each env has its own price data
        return jax.vmap(
            lambda p, ws: encode_observation_simple(p, ws, config)
        )(prices, window_starts)
    else:
        # Shared prices across all envs
        return batch_encode_simple(prices, window_starts, config)


# =============================================================================
# Validation
# =============================================================================

def validate_encoder_output(obs: chex.Array, config: EncoderConfig) -> bool:
    """Validate encoder output shape and values."""
    expected_shape = (config.n_channels, config.n_assets, config.n_assets)

    if obs.shape != expected_shape:
        print(f"Shape mismatch: {obs.shape} != {expected_shape}")
        return False

    # Check for NaN/Inf
    if jnp.any(jnp.isnan(obs)):
        print("Output contains NaN values")
        return False

    if jnp.any(jnp.isinf(obs)):
        print("Output contains Inf values")
        return False

    return True


# =============================================================================
# TASK_136a: Compressed Encoder (Pre-encode demos without tiling)
# =============================================================================
# All 27 channels use jnp.tile(value, (1, n)) which is 100% redundant.
# Compressed storage: (27, n_assets) instead of (27, n_assets, n_assets)
# Memory savings: 64× compression → 345 MB vs 22.1 GB for 50k demos

def _compute_all_channels_compressed(
    prices: chex.Array,   # (n_assets, window_size)
    volumes: chex.Array,  # (n_assets, window_size)
    highs: chex.Array,    # (n_assets, window_size)
    lows: chex.Array,     # (n_assets, window_size)
    config: EncoderConfig,
) -> chex.Array:
    """Compute all 27 channels WITHOUT tiling - compressed format.

    Returns (27, n_assets) instead of (27, n_assets, n_assets).
    Each channel stores the 1D feature vector that would be tiled.
    """
    n = config.n_assets

    # Initialize output (n_channels, n) - compressed!
    # TASK_689: Parameterized channel count (was hardcoded 27)
    output = jnp.zeros((config.n_channels, n), dtype=jnp.float32)

    # =========================================================================
    # Normalization (same as full encoder)
    # =========================================================================
    price_min = jnp.min(prices, axis=1, keepdims=True)  # (n, 1)
    price_max = jnp.max(prices, axis=1, keepdims=True)  # (n, 1)
    price_range = price_max - price_min
    price_range = jnp.where(price_range < 1e-8, 1.0, price_range)

    norm_prices = (prices - price_min) / price_range  # (n, ws)
    norm_highs = (highs - price_min) / price_range
    norm_lows = (lows - price_min) / price_range

    # Volume normalization (z-score)
    vol_mean = jnp.mean(volumes, axis=1, keepdims=True)  # (n, 1)
    vol_std = jnp.std(volumes, axis=1, keepdims=True) + 1e-8
    norm_volumes = jnp.clip((volumes - vol_mean) / vol_std, -3, 3) / 3.0  # [-1, 1]

    # =========================================================================
    # Channel 0-3: Latest OHLC (normalized) - NO TILING
    # =========================================================================
    output = output.at[0].set(norm_prices[:, -1])    # Close
    output = output.at[1].set(norm_highs[:, -1])     # High
    output = output.at[2].set(norm_lows[:, -1])      # Low
    output = output.at[3].set(norm_volumes[:, -1])   # Volume

    # =========================================================================
    # Channel 4-7: Log returns at 1, 5, 15, 60 min
    # =========================================================================
    prev_prices = jnp.concatenate([prices[:, 0:1], prices[:, :-1]], axis=1)
    returns_1m = (prices - prev_prices) / (prev_prices + 1e-8)
    returns_1m = jnp.clip(returns_1m, -config.return_clip, config.return_clip)
    output = output.at[4].set(returns_1m[:, -1])

    returns_5m = (prices[:, -1] - prices[:, -5]) / (prices[:, -5] + 1e-8)
    returns_5m = jnp.clip(returns_5m, -config.return_clip, config.return_clip)
    output = output.at[5].set(returns_5m)

    returns_15m = (prices[:, -1] - prices[:, -15]) / (prices[:, -15] + 1e-8)
    returns_15m = jnp.clip(returns_15m, -config.return_clip, config.return_clip)
    output = output.at[6].set(returns_15m)

    returns_60m = (prices[:, -1] - prices[:, 0]) / (prices[:, 0] + 1e-8)
    returns_60m = jnp.clip(returns_60m, -config.return_clip, config.return_clip)
    output = output.at[7].set(returns_60m)

    # =========================================================================
    # Channel 8-11: Volume patterns (recent vs average)
    # =========================================================================
    vol_ratio_5m = jnp.mean(volumes[:, -5:], axis=1) / (vol_mean[:, 0] + 1e-8)
    vol_ratio_15m = jnp.mean(volumes[:, -15:], axis=1) / (vol_mean[:, 0] + 1e-8)
    vol_ratio_30m = jnp.mean(volumes[:, -30:], axis=1) / (vol_mean[:, 0] + 1e-8)
    vol_ratio_60m = jnp.mean(volumes, axis=1) / (vol_mean[:, 0] + 1e-8)

    output = output.at[8].set(jnp.clip(vol_ratio_5m, 0, 5) / 5.0)
    output = output.at[9].set(jnp.clip(vol_ratio_15m, 0, 5) / 5.0)
    output = output.at[10].set(jnp.clip(vol_ratio_30m, 0, 5) / 5.0)
    output = output.at[11].set(jnp.clip(vol_ratio_60m, 0, 5) / 5.0)

    # =========================================================================
    # Channel 12-15: Price momentum (rate of change)
    # =========================================================================
    momentum_5 = (prices[:, -1] - prices[:, -5]) / (price_range[:, 0] + 1e-8)
    momentum_15 = (prices[:, -1] - prices[:, -15]) / (price_range[:, 0] + 1e-8)
    momentum_30 = (prices[:, -1] - prices[:, -30]) / (price_range[:, 0] + 1e-8)
    momentum_60 = (prices[:, -1] - prices[:, 0]) / (price_range[:, 0] + 1e-8)

    output = output.at[12].set(jnp.clip(momentum_5, -1, 1))
    output = output.at[13].set(jnp.clip(momentum_15, -1, 1))
    output = output.at[14].set(jnp.clip(momentum_30, -1, 1))
    output = output.at[15].set(jnp.clip(momentum_60, -1, 1))

    # =========================================================================
    # Channel 16-19: Volatility (rolling std of returns)
    # =========================================================================
    vol_5 = jnp.std(returns_1m[:, -5:], axis=1)
    vol_15 = jnp.std(returns_1m[:, -15:], axis=1)
    vol_30 = jnp.std(returns_1m[:, -30:], axis=1)
    vol_60 = jnp.std(returns_1m, axis=1)

    output = output.at[16].set(jnp.clip(vol_5 * config.vol_scale, 0, 1))
    output = output.at[17].set(jnp.clip(vol_15 * config.vol_scale, 0, 1))
    output = output.at[18].set(jnp.clip(vol_30 * config.vol_scale, 0, 1))
    output = output.at[19].set(jnp.clip(vol_60 * config.vol_scale, 0, 1))

    # =========================================================================
    # Channel 20-23: Price percentiles (position in recent range)
    # =========================================================================
    def compute_percentile(window_prices: chex.Array, window: int) -> chex.Array:
        window_data = window_prices[:, -window:]
        w_min = jnp.min(window_data, axis=1)
        w_max = jnp.max(window_data, axis=1)
        w_range = w_max - w_min
        w_range = jnp.where(w_range < 1e-8, 1.0, w_range)
        return (window_prices[:, -1] - w_min) / w_range

    output = output.at[20].set(compute_percentile(prices, 5))
    output = output.at[21].set(compute_percentile(prices, 15))
    output = output.at[22].set(compute_percentile(prices, 30))
    output = output.at[23].set(compute_percentile(prices, 60))

    # =========================================================================
    # Channel 24-26: VWAP and bands
    # =========================================================================
    vwap_window = 20
    vwap_prices = prices[:, -vwap_window:]
    vwap_volumes = volumes[:, -vwap_window:]
    vol_sum = jnp.sum(vwap_volumes, axis=1, keepdims=True) + 1e-8
    vwap = jnp.sum(vwap_prices * vwap_volumes, axis=1) / vol_sum[:, 0]
    vwap_norm = (vwap - price_min[:, 0]) / (price_range[:, 0] + 1e-8)

    price_std = jnp.std(vwap_prices, axis=1)
    upper_band = (vwap + price_std - price_min[:, 0]) / (price_range[:, 0] + 1e-8)
    lower_band = (vwap - price_std - price_min[:, 0]) / (price_range[:, 0] + 1e-8)

    output = output.at[24].set(vwap_norm)
    output = output.at[25].set(jnp.clip(upper_band, 0, 1))
    output = output.at[26].set(jnp.clip(lower_band, 0, 1))

    return output


def encode_observation_compressed(
    prices: chex.Array,        # (n_steps, n_assets)
    window_start: chex.Array,  # ()
    config: EncoderConfig,
) -> chex.Array:
    """Encode to compressed format (27, n_assets) without tiling.

    Use this for pre-encoding demos. At sample time, use tile_compressed_observation()
    to expand to full (27, n_assets, n_assets) format.
    """
    n = config.n_assets
    ws = config.window_size

    # Extract window
    price_window = jax.lax.dynamic_slice(prices, (window_start, 0), (ws, n))

    # Use prices for OHLCV (synthetic data)
    prices_T = price_window.T      # (n, ws)
    volumes_T = jnp.ones_like(prices_T)

    return _compute_all_channels_compressed(prices_T, volumes_T, prices_T, prices_T, config)


@partial(jax.jit, static_argnames=['config'])
def jit_encode_compressed(
    prices: chex.Array,
    window_start: chex.Array,
    config: EncoderConfig,
) -> chex.Array:
    """JIT-compiled compressed encoder."""
    return encode_observation_compressed(prices, window_start, config)


@partial(jax.jit, static_argnames=['config'])
def jit_batch_encode_compressed(
    prices: chex.Array,        # (n_steps, n_assets)
    window_starts: chex.Array, # (batch,)
    config: EncoderConfig,
) -> chex.Array:
    """JIT-compiled batched compressed encoder.

    Args:
        prices: (n_steps, n_assets) price data
        window_starts: (batch,) array of window start indices
        config: Encoder configuration

    Returns:
        (batch, 27, n_assets) compressed observations
    """
    return jax.vmap(
        lambda ws: encode_observation_compressed(prices, ws, config)
    )(window_starts)


@partial(jax.jit, static_argnames=['config'])
def jit_vmap_encode_compressed(
    prices_batch: chex.Array,      # (N, n_steps, n_assets)
    window_starts: chex.Array,     # (L,) - shared window starts
    config: EncoderConfig,
) -> chex.Array:
    """Encode multiple environments in parallel via vmap.

    Args:
        prices_batch: (N, n_steps, n_assets) batched price data
        window_starts: (L,) shared window indices
        config: Encoder configuration

    Returns:
        (N, L, 27, n_assets) encoded observations
    """
    # Double vmap: outer over episodes (N), inner over windows (L)
    return jax.vmap(
        lambda p: jit_batch_encode_compressed(p, window_starts, config)
    )(prices_batch)


def tile_compressed_observation(
    compressed: chex.Array,  # (27, n_assets)
    n_assets: int,
) -> chex.Array:
    """Expand compressed observation to full format.

    Args:
        compressed: (27, n_assets) compressed observation
        n_assets: Number of assets (for tiling)

    Returns:
        (27, n_assets, n_assets) full observation
    """
    # Tile each channel: (n,) → (n, n) by repeating across columns
    return jnp.tile(compressed[:, :, None], (1, 1, n_assets))


@partial(jax.jit, static_argnames=['n_assets'])
def jit_tile_compressed(
    compressed: chex.Array,
    n_assets: int,
) -> chex.Array:
    """JIT-compiled tiling of compressed observations."""
    return tile_compressed_observation(compressed, n_assets)


@partial(jax.jit, static_argnames=['n_assets'])
def jit_batch_tile_compressed(
    compressed_batch: chex.Array,  # (batch, 27, n_assets)
    n_assets: int,
) -> chex.Array:
    """JIT-compiled batched tiling of compressed observations.

    Args:
        compressed_batch: (batch, 27, n_assets) compressed observations
        n_assets: Number of assets (for tiling)

    Returns:
        (batch, 27, n_assets, n_assets) full observations
    """
    return jax.vmap(lambda c: tile_compressed_observation(c, n_assets))(compressed_batch)
