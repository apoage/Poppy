"""
TASK_125: On-Demand Observation Encoder for Causal Expert Imprinting.

JIT-compiled feature computation that transforms raw OHLCV prices
into 27-channel encoded observations on GPU.

Called on-demand after batch sampling from ExpertBuffer.
"""

import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple


class ObservationEncoder:
    """
    Encode raw price windows into model input features.

    Called on-demand after batch sampling.
    Runs on GPU for efficiency via JAX JIT compilation.

    Input: (batch, window, assets, 5) OHLCV
    Output: (batch, 27, window, assets) encoded observation
    """

    def __init__(
        self,
        n_channels: int = 27,
        window_size: int = 60,
        n_assets: int = 64,
    ):
        self.n_channels = n_channels
        self.window_size = window_size
        self.n_assets = n_assets

        # Pre-compile the encoding function
        self._encode_batch_jit = jax.jit(self._encode_batch_impl)

    def encode_batch(self, prices: jnp.ndarray) -> jnp.ndarray:
        """
        Convert raw OHLCV to 27-channel encoded observation.

        Args:
            prices: (batch, window, assets, 5) OHLCV data

        Returns:
            (batch, n_channels, window, assets) encoded observation
        """
        return self._encode_batch_jit(prices)

    def _encode_batch_impl(self, prices: jnp.ndarray) -> jnp.ndarray:
        """
        JIT-compiled implementation of batch encoding.

        Computes 27 channels of technical features from OHLCV.
        """
        # Extract OHLCV
        open_p = prices[..., 0]   # (batch, window, assets)
        high_p = prices[..., 1]
        low_p = prices[..., 2]
        close_p = prices[..., 3]
        volume = prices[..., 4]

        # Avoid division by zero
        eps = 1e-8

        # === Normalize by first close (relative prices) ===
        first_close = close_p[:, 0:1, :]  # (batch, 1, assets)
        first_close = jnp.maximum(first_close, eps)

        norm_open = open_p / first_close
        norm_high = high_p / first_close
        norm_low = low_p / first_close
        norm_close = close_p / first_close

        # === Log returns ===
        log_returns = jnp.log(close_p[:, 1:] / jnp.maximum(close_p[:, :-1], eps))
        log_returns = jnp.concatenate([jnp.zeros_like(log_returns[:, :1]), log_returns], axis=1)

        # === Volume normalization (z-score) ===
        vol_mean = jnp.mean(volume, axis=1, keepdims=True)
        vol_std = jnp.std(volume, axis=1, keepdims=True) + eps
        norm_volume = (volume - vol_mean) / vol_std
        norm_volume = jnp.clip(norm_volume, -5, 5)  # Clip outliers

        # === Technical indicators ===

        # SMA ratios (5, 10, 20 periods)
        sma_5 = self._rolling_mean(close_p, 5)
        sma_10 = self._rolling_mean(close_p, 10)
        sma_20 = self._rolling_mean(close_p, 20)

        sma_ratio_5 = close_p / jnp.maximum(sma_5, eps)
        sma_ratio_10 = close_p / jnp.maximum(sma_10, eps)
        sma_ratio_20 = close_p / jnp.maximum(sma_20, eps)

        # Volatility (rolling std of returns)
        volatility_10 = self._rolling_std(log_returns, 10)
        volatility_20 = self._rolling_std(log_returns, 20)

        # RSI approximation (14 period)
        gains = jnp.maximum(log_returns, 0)
        losses = jnp.maximum(-log_returns, 0)
        avg_gain = self._rolling_mean(gains, 14)
        avg_loss = self._rolling_mean(losses, 14) + eps
        rsi = avg_gain / (avg_gain + avg_loss)

        # High-low range (ATR-like)
        hl_range = (high_p - low_p) / jnp.maximum(close_p, eps)

        # Close-open range (body size)
        co_range = (close_p - open_p) / jnp.maximum(close_p, eps)

        # Upper/lower shadows
        upper_shadow = (high_p - jnp.maximum(close_p, open_p)) / jnp.maximum(close_p, eps)
        lower_shadow = (jnp.minimum(close_p, open_p) - low_p) / jnp.maximum(close_p, eps)

        # === Lagged returns (1, 2, 5, 10, 20 step) ===
        lagged_1 = jnp.roll(log_returns, 1, axis=1)
        lagged_1 = lagged_1.at[:, :1].set(0)

        lagged_2 = jnp.roll(log_returns, 2, axis=1)
        lagged_2 = lagged_2.at[:, :2].set(0)

        lagged_5 = jnp.roll(log_returns, 5, axis=1)
        lagged_5 = lagged_5.at[:, :5].set(0)

        lagged_10 = jnp.roll(log_returns, 10, axis=1)
        lagged_10 = lagged_10.at[:, :10].set(0)

        lagged_20 = jnp.roll(log_returns, 20, axis=1)
        lagged_20 = lagged_20.at[:, :20].set(0)

        # === Cumulative returns (from window start) ===
        cum_returns = jnp.cumsum(log_returns, axis=1)

        # === Cross-asset momentum rank ===
        rank = self._cross_asset_rank(log_returns)

        # === MACD-like (fast - slow) ===
        macd = sma_5 - sma_20
        macd_norm = macd / jnp.maximum(sma_20, eps)

        # === Bollinger-like (distance from SMA in std units) ===
        bb_distance = (close_p - sma_20) / jnp.maximum(volatility_20 * jnp.sqrt(20) * close_p, eps)
        bb_distance = jnp.clip(bb_distance, -3, 3)

        # === Stack channels ===
        # Total: 27 channels
        channels = [
            # Price levels (4)
            norm_open, norm_high, norm_low, norm_close,
            # Returns (1)
            log_returns,
            # Volume (1)
            norm_volume,
            # SMA ratios (3)
            sma_ratio_5, sma_ratio_10, sma_ratio_20,
            # Volatility (2)
            volatility_10, volatility_20,
            # RSI (1)
            rsi,
            # Candlestick features (4)
            hl_range, co_range, upper_shadow, lower_shadow,
            # Lagged returns (5)
            lagged_1, lagged_2, lagged_5, lagged_10, lagged_20,
            # Cumulative (1)
            cum_returns,
            # Cross-asset rank (1)
            rank,
            # Momentum indicators (2)
            macd_norm, bb_distance,
            # Padding to 27 (2)
            jnp.zeros_like(norm_close),
            jnp.zeros_like(norm_close),
        ]

        # Stack: (batch, channels, window, assets)
        encoded = jnp.stack(channels[:self.n_channels], axis=1)

        # Clip extreme values
        encoded = jnp.clip(encoded, -10, 10)

        # Replace NaN/Inf with 0
        encoded = jnp.where(jnp.isfinite(encoded), encoded, 0.0)

        return encoded

    def _rolling_mean(self, x: jnp.ndarray, window: int) -> jnp.ndarray:
        """
        Compute rolling mean along time axis.

        Uses cumsum trick for efficiency.
        """
        # x: (batch, time, assets)
        cumsum = jnp.cumsum(x, axis=1)

        # Shifted cumsum for window subtraction
        shifted = jnp.concatenate([
            jnp.zeros_like(cumsum[:, :window]),
            cumsum[:, :-window]
        ], axis=1)

        rolling_sum = cumsum - shifted
        rolling_mean = rolling_sum / window

        # First (window-1) values use available data
        for i in range(1, window):
            rolling_mean = rolling_mean.at[:, i].set(cumsum[:, i] / (i + 1))

        return rolling_mean

    def _rolling_std(self, x: jnp.ndarray, window: int) -> jnp.ndarray:
        """
        Compute rolling standard deviation.

        Uses rolling mean and rolling mean of squares.
        """
        mean = self._rolling_mean(x, window)
        sq_mean = self._rolling_mean(x ** 2, window)
        var = sq_mean - mean ** 2
        var = jnp.maximum(var, 1e-10)  # Numerical stability
        return jnp.sqrt(var)

    def _cross_asset_rank(self, returns: jnp.ndarray) -> jnp.ndarray:
        """
        Rank assets by return at each timestep.

        Returns percentile rank (0-1) across assets.
        """
        # returns: (batch, time, assets)
        n_assets = returns.shape[-1]

        # Argsort gives indices that would sort the array
        sorted_idx = jnp.argsort(returns, axis=-1)

        # Argsort of argsort gives ranks
        ranks = jnp.argsort(sorted_idx, axis=-1)

        # Normalize to [0, 1]
        return ranks / max(n_assets - 1, 1)


# Convenience functions

def create_observation_encoder(
    n_assets: int = 64,
    window_size: int = 60,
) -> ObservationEncoder:
    """Create an observation encoder with default settings."""
    return ObservationEncoder(
        n_channels=27,
        window_size=window_size,
        n_assets=n_assets,
    )


@partial(jax.jit, static_argnums=(1, 2))
def encode_prices_to_observation(
    prices: jnp.ndarray,
    n_channels: int = 27,
    n_assets: int = 64,
) -> jnp.ndarray:
    """
    Standalone JIT-compiled encoding function.

    Args:
        prices: (batch, window, assets, 5) OHLCV

    Returns:
        (batch, n_channels, window, assets) encoded observation
    """
    encoder = ObservationEncoder(n_channels=n_channels, n_assets=n_assets)
    return encoder._encode_batch_impl(prices)


def encode_batch_numpy(prices: jnp.ndarray, encoder: ObservationEncoder) -> jnp.ndarray:
    """
    Encode prices and return as numpy array.

    Useful for debugging/testing.
    """
    encoded_jax = encoder.encode_batch(jax.device_put(prices))
    return jnp.asarray(encoded_jax)
