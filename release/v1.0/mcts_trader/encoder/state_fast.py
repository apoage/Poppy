"""
Fast MultiAssetStateEncoder - Optimized for speed

Removes expensive operations:
- NO correlation matrices (O(n²) computation removed)
- Minimal technical indicators (only essential ones)
- Vectorized operations
- No device transfers during computation

Speed improvement: ~10-20x faster than full encoder
"""

import torch
import numpy as np
from typing import Dict


class FastMultiAssetStateEncoder:
    """
    Lightweight encoder for 64-asset market state.

    Output tensor: (27, 64, 64)
    Channels:
    - 0-3: Normalized OHLC (4 channels, tiled across 64 width)
    - 4-7: Log returns at 1, 5, 15, 60 min (4 channels)
    - 8-11: Volume patterns (4 channels)
    - 12-15: Price momentum (4 channels)
    - 16-19: Volatility (4 channels)
    - 20-23: Price percentiles (4 channels)
    - 24-26: VWAP, Upper/Lower bands (3 channels)

    Total: 27 channels
    """

    def __init__(
        self,
        n_assets: int = 64,
        lookback: int = 60,
        device: str = 'cpu'  # Always use CPU to avoid PyTorch/JAX conflict
    ):
        self.n_assets = n_assets
        self.lookback = lookback
        self.device = device

        # Asset mapping
        self.asset_map: Dict[int, str] = {}
        self.symbol_map: Dict[str, int] = {}

        # Circular buffers (2x lookback for safety)
        buf_size = lookback * 2
        self.price_buffer = np.zeros((n_assets, buf_size), dtype=np.float32)
        self.volume_buffer = np.zeros((n_assets, buf_size), dtype=np.float32)
        self.high_buffer = np.zeros((n_assets, buf_size), dtype=np.float32)
        self.low_buffer = np.zeros((n_assets, buf_size), dtype=np.float32)
        self.buffer_idx = 0

    def register_assets(self, symbols: list):
        """Register assets with their indices"""
        for i, symbol in enumerate(symbols):
            self.asset_map[i] = symbol
            self.symbol_map[symbol] = i

    def update(self, market_data: Dict[str, Dict]):
        """
        Update buffers with new market data.
        Handles missing data by forward-filling prices and zeroing volume.

        Args:
            market_data: {symbol: {'open': x, 'high': x, 'low': x, 'close': x, 'volume': x}}
        """
        buf_size = self.price_buffer.shape[1]
        idx = self.buffer_idx % buf_size
        prev_idx = (self.buffer_idx - 1) % buf_size
        
        # Iterate over ALL registered assets to ensure completeness
        for symbol, asset_idx in self.symbol_map.items():
            if symbol in market_data:
                data = market_data[symbol]
                close_price = data.get('close', 0.0)
                
                # Check if this is the first valid data point (prev was 0)
                # If so, backfill to prevent 0 -> Price massive return spike
                if self.price_buffer[asset_idx, prev_idx] == 0.0 and close_price > 0:
                    # Backfill the buffer with this first price to simulate stable history
                    # We only need to backfill enough to cover the lookback, but filling all is safe
                    self.price_buffer[asset_idx, :] = close_price
                    self.high_buffer[asset_idx, :] = close_price
                    self.low_buffer[asset_idx, :] = close_price
                    # Don't backfill volume, it really was 0
                
                self.price_buffer[asset_idx, idx] = close_price
                self.volume_buffer[asset_idx, idx] = data.get('volume', 0)
                self.high_buffer[asset_idx, idx] = data.get('high', 0)
                self.low_buffer[asset_idx, idx] = data.get('low', 0)
            else:
                # Missing data! Forward fill from previous step
                if self.buffer_idx > 0:
                    # Use previous close for Price/High/Low
                    prev_price = self.price_buffer[asset_idx, prev_idx]
                    self.price_buffer[asset_idx, idx] = prev_price
                    self.high_buffer[asset_idx, idx] = prev_price
                    self.low_buffer[asset_idx, idx] = prev_price
                    # Volume is 0 when no trade data
                    self.volume_buffer[asset_idx, idx] = 0.0
                else:
                    # Initial step with missing data: set to 0 (handled in encode safely?)
                    self.price_buffer[asset_idx, idx] = 0.0
                    self.volume_buffer[asset_idx, idx] = 0.0
                    self.high_buffer[asset_idx, idx] = 0.0
                    self.low_buffer[asset_idx, idx] = 0.0

        self.buffer_idx += 1

    def _get_recent_data(self, buffer: np.ndarray) -> np.ndarray:
        """Get most recent lookback data from circular buffer"""
        buf_size = buffer.shape[1]

        if self.buffer_idx < self.lookback:
            # Not enough data yet, pad with zeros
            if self.buffer_idx == 0:
                # No data at all, return zeros
                return np.zeros((buffer.shape[0], self.lookback), dtype=np.float32)

            valid_data = buffer[:, :self.buffer_idx]
            pad_len = self.lookback - self.buffer_idx
            return np.pad(valid_data, ((0, 0), (pad_len, 0)), mode='edge')

        # Extract last lookback values (circular buffer logic)
        idx = self.buffer_idx % buf_size
        if idx >= self.lookback:
            return buffer[:, idx - self.lookback:idx]
        else:
            # Wrap around
            part1 = buffer[:, -(self.lookback - idx):]
            part2 = buffer[:, :idx]
            return np.concatenate([part1, part2], axis=1)

    def encode(self) -> torch.Tensor:
        """
        Fast encoding - returns (27, 64, 64) tensor ready for CNN.

        Returns:
            torch.Tensor: Shape (27, 64, 64)
        """
        # Get recent data (64 assets × 60 timesteps)
        prices = self._get_recent_data(self.price_buffer)    # (64, 60)
        volumes = self._get_recent_data(self.volume_buffer)  # (64, 60)
        highs = self._get_recent_data(self.high_buffer)      # (64, 60)
        lows = self._get_recent_data(self.low_buffer)        # (64, 60)

        # Initialize output tensor (27, 64, 64)
        output = np.zeros((27, 64, 64), dtype=np.float32)

        # --- VECTORIZED FEATURE COMPUTATION ---

        # Normalize prices (avoid division by zero)
        price_min = np.min(prices, axis=1, keepdims=True)
        price_max = np.max(prices, axis=1, keepdims=True)
        price_range = price_max - price_min
        price_range[price_range < 1e-8] = 1.0

        norm_prices = (prices - price_min) / price_range  # (64, 60)
        norm_highs = (highs - price_min) / price_range
        norm_lows = (lows - price_min) / price_range

        # Normalize volumes (z-score)
        vol_mean = np.mean(volumes, axis=1, keepdims=True)
        vol_std = np.std(volumes, axis=1, keepdims=True) + 1e-8
        norm_volumes = np.clip((volumes - vol_mean) / vol_std, -3, 3) / 3.0  # [-1, 1]

        # Channel 0-3: Latest OHLC normalized (tile across width)
        output[0, :, :] = np.tile(norm_prices[:, -1:], (1, 64))     # Close
        output[1, :, :] = np.tile(norm_highs[:, -1:], (1, 64))      # High
        output[2, :, :] = np.tile(norm_lows[:, -1:], (1, 64))       # Low
        output[3, :, :] = np.tile(norm_volumes[:, -1:], (1, 64))    # Volume

        # Channel 4-7: Log returns at different timeframes
        # CLIP RETURNS to prevent explosions from 0->Price jumps
        returns_1m = np.diff(prices, axis=1, prepend=prices[:, 0:1]) / (prices + 1e-8)
        returns_1m = np.clip(returns_1m, -10.0, 10.0)
        output[4, :, :] = np.tile(returns_1m[:, -1:], (1, 64))  # 1-min return

        if prices.shape[1] >= 5:
            returns_5m = (prices[:, -1] - prices[:, -5]) / (prices[:, -5] + 1e-8)
            returns_5m = np.clip(returns_5m, -10.0, 10.0)
            output[5, :, :] = np.tile(returns_5m[:, None], (1, 64))

        if prices.shape[1] >= 15:
            returns_15m = (prices[:, -1] - prices[:, -15]) / (prices[:, -15] + 1e-8)
            returns_15m = np.clip(returns_15m, -10.0, 10.0)
            output[6, :, :] = np.tile(returns_15m[:, None], (1, 64))

        if prices.shape[1] >= 60:
            returns_60m = (prices[:, -1] - prices[:, -60]) / (prices[:, -60] + 1e-8)
            returns_60m = np.clip(returns_60m, -10.0, 10.0)
            output[7, :, :] = np.tile(returns_60m[:, None], (1, 64))

        # Channel 8-11: Volume patterns (recent vs average)
        vol_ratio_5m = np.mean(volumes[:, -5:], axis=1) / (vol_mean[:, 0] + 1e-8)
        vol_ratio_15m = np.mean(volumes[:, -15:], axis=1) / (vol_mean[:, 0] + 1e-8)
        vol_ratio_30m = np.mean(volumes[:, -30:], axis=1) / (vol_mean[:, 0] + 1e-8)
        vol_ratio_60m = np.mean(volumes[:, -60:], axis=1) / (vol_mean[:, 0] + 1e-8)

        output[8, :, :] = np.tile(np.clip(vol_ratio_5m[:, None], 0, 5) / 5.0, (1, 64))
        output[9, :, :] = np.tile(np.clip(vol_ratio_15m[:, None], 0, 5) / 5.0, (1, 64))
        output[10, :, :] = np.tile(np.clip(vol_ratio_30m[:, None], 0, 5) / 5.0, (1, 64))
        output[11, :, :] = np.tile(np.clip(vol_ratio_60m[:, None], 0, 5) / 5.0, (1, 64))

        # Channel 12-15: Price momentum (rate of change)
        if prices.shape[1] >= 5:
            momentum_5 = (prices[:, -1] - prices[:, -5]) / (price_range[:, 0] + 1e-8)
            output[12, :, :] = np.tile(np.clip(momentum_5[:, None], -1, 1), (1, 64))

        if prices.shape[1] >= 15:
            momentum_15 = (prices[:, -1] - prices[:, -15]) / (price_range[:, 0] + 1e-8)
            output[13, :, :] = np.tile(np.clip(momentum_15[:, None], -1, 1), (1, 64))

        if prices.shape[1] >= 30:
            momentum_30 = (prices[:, -1] - prices[:, -30]) / (price_range[:, 0] + 1e-8)
            output[14, :, :] = np.tile(np.clip(momentum_30[:, None], -1, 1), (1, 64))

        if prices.shape[1] >= 60:
            momentum_60 = (prices[:, -1] - prices[:, -60]) / (price_range[:, 0] + 1e-8)
            output[15, :, :] = np.tile(np.clip(momentum_60[:, None], -1, 1), (1, 64))

        # Channel 16-19: Volatility (rolling std of returns)
        if prices.shape[1] >= 5:
            vol_5 = np.std(returns_1m[:, -5:], axis=1)
            output[16, :, :] = np.tile(np.clip(vol_5[:, None] * 100, 0, 1), (1, 64))

        if prices.shape[1] >= 15:
            vol_15 = np.std(returns_1m[:, -15:], axis=1)
            output[17, :, :] = np.tile(np.clip(vol_15[:, None] * 100, 0, 1), (1, 64))

        if prices.shape[1] >= 30:
            vol_30 = np.std(returns_1m[:, -30:], axis=1)
            output[18, :, :] = np.tile(np.clip(vol_30[:, None] * 100, 0, 1), (1, 64))

        if prices.shape[1] >= 60:
            vol_60 = np.std(returns_1m[:, -60:], axis=1)
            output[19, :, :] = np.tile(np.clip(vol_60[:, None] * 100, 0, 1), (1, 64))

        # Channel 20-23: Price percentiles (where is current price in recent range?)
        for i, window in enumerate([5, 15, 30, 60]):
            if prices.shape[1] >= window:
                window_prices = prices[:, -window:]
                window_min = np.min(window_prices, axis=1)
                window_max = np.max(window_prices, axis=1)
                window_range = window_max - window_min
                window_range[window_range < 1e-8] = 1.0
                percentile = (prices[:, -1] - window_min) / window_range
                output[20 + i, :, :] = np.tile(percentile[:, None], (1, 64))

        # Channel 24-26: VWAP and bands
        if prices.shape[1] >= 20:
            vwap = np.sum(prices[:, -20:] * volumes[:, -20:], axis=1) / (np.sum(volumes[:, -20:], axis=1) + 1e-8)
            vwap_norm = (vwap - price_min[:, 0]) / (price_range[:, 0] + 1e-8)
            output[24, :, :] = np.tile(vwap_norm[:, None], (1, 64))

            # Upper/lower bands (±1 std)
            price_std = np.std(prices[:, -20:], axis=1)
            upper_band = (vwap + price_std - price_min[:, 0]) / (price_range[:, 0] + 1e-8)
            lower_band = (vwap - price_std - price_min[:, 0]) / (price_range[:, 0] + 1e-8)
            output[25, :, :] = np.tile(np.clip(upper_band[:, None], 0, 1), (1, 64))
            output[26, :, :] = np.tile(np.clip(lower_band[:, None], 0, 1), (1, 64))

        # Convert to PyTorch tensor (on CPU)
        return torch.from_numpy(output).float()
