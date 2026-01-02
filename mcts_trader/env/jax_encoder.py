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
    """Configuration for the 27-channel encoder.

    Attributes:
        n_assets: Number of assets (default 64)
        window_size: Lookback window for features (default 60)
        n_channels: Number of output channels (fixed at 27)
        return_clip: Max return value before clipping (default 10.0)
        vol_scale: Volatility scaling factor (default 100.0)
    """
    n_assets: int = 64
    window_size: int = 60
    n_channels: int = 27
    return_clip: float = 10.0
    vol_scale: float = 100.0


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
    """
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


def _compute_all_channels(
    prices: chex.Array,   # (n_assets, window_size)
    volumes: chex.Array,  # (n_assets, window_size)
    highs: chex.Array,    # (n_assets, window_size)
    lows: chex.Array,     # (n_assets, window_size)
    config: EncoderConfig,
) -> chex.Array:
    """Compute all 27 channels of the observation tensor.

    All operations are vectorized across assets using JAX primitives.
    """
    n = config.n_assets
    ws = config.window_size

    # Initialize output (27, n, n)
    output = jnp.zeros((27, n, n), dtype=jnp.float32)

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

    # 1-min return
    returns_1m = jnp.diff(prices, axis=1, prepend=prices[:, 0:1]) / (prices + 1e-8)
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

    return output


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


@partial(jax.jit, static_argnames=['config'])
def jit_batch_encode_simple(
    prices: chex.Array,
    window_starts: chex.Array,
    config: EncoderConfig,
) -> chex.Array:
    """JIT-compiled batched simplified encoder."""
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
