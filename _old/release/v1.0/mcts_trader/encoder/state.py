import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional
from .correlation import CorrelationEngine
from .indicators import TechnicalIndicators

class MultiAssetStateEncoder:
    """
    Encodes 64-asset market state into tensor for TRM+MCTS.
    
    Output tensor structure:
    - Correlation matrices: (4, 64, 64) - 3 timeframes + lead-lag
    - Per-asset indicators: (n_indicators, 64)
    - Price history encoding: (64, lookback, 3) - OHLC normalized
    
    Total channels for CNN: 4 + n_indicators + 3 = ~25 channels
    """
    
    def __init__(
        self,
        n_assets: int = 64,
        lookback: int = 60,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.n_assets = n_assets
        self.lookback = lookback
        self.device = device
        
        self.correlation_engine = CorrelationEngine(n_assets)
        self.indicators = TechnicalIndicators(n_assets)
        
        # Asset index to symbol mapping
        self.asset_map: Dict[int, str] = {}
        self.symbol_map: Dict[str, int] = {}
        
        # Price history buffer
        self.price_buffer = np.zeros((n_assets, lookback * 2), dtype=np.float32)
        self.volume_buffer = np.zeros((n_assets, lookback * 2), dtype=np.float32)
        self.high_buffer = np.zeros((n_assets, lookback * 2), dtype=np.float32)
        self.low_buffer = np.zeros((n_assets, lookback * 2), dtype=np.float32)
        self.buffer_idx = 0
        
    def register_assets(self, symbols: list):
        """Register assets with their indices"""
        for i, symbol in enumerate(symbols):
            self.asset_map[i] = symbol
            self.symbol_map[symbol] = i
    
    def update(self, market_data: Dict[str, Dict]):
        """
        Update buffers with new market data.
        
        Args:
            market_data: {symbol: {'open': x, 'high': x, 'low': x, 'close': x, 'volume': x}}
        """
        for symbol, data in market_data.items():
            if symbol not in self.symbol_map:
                continue
            
            idx = self.symbol_map[symbol]
            buf_idx = self.buffer_idx % (self.lookback * 2)
            
            self.price_buffer[idx, buf_idx] = data['close']
            self.volume_buffer[idx, buf_idx] = data['volume']
            self.high_buffer[idx, buf_idx] = data['high']
            self.low_buffer[idx, buf_idx] = data['low']
        
        self.buffer_idx += 1
        
        # Update correlation matrices periodically
        if self.buffer_idx % 5 == 0:  # Every 5 minutes
            returns = self._compute_returns()
            self.correlation_engine.update(returns)
    
    def _compute_returns(self) -> np.ndarray:
        """Compute returns from price buffer"""
        prices = self._get_ordered_buffer(self.price_buffer)
        returns = np.diff(prices, axis=1) / (prices[:, :-1] + 1e-8)
        return returns
    
    def _get_ordered_buffer(self, buffer: np.ndarray) -> np.ndarray:
        """Get buffer in chronological order"""
        buf_size = buffer.shape[1]
        if self.buffer_idx < buf_size:
            return buffer[:, :self.buffer_idx]
        
        idx = self.buffer_idx % buf_size
        return np.concatenate([buffer[:, idx:], buffer[:, :idx]], axis=1)
    
    def encode(self) -> Dict[str, torch.Tensor]:
        """
        Encode current state into tensors.
        
        Returns:
            {
                'correlation': (4, 64, 64),
                'indicators': (n_indicators, 64),
                'price_history': (64, lookback, 4),
                'combined': (C, 64, 64) - ready for CNN
            }
        """
        # Get ordered buffers
        full_prices = self._get_ordered_buffer(self.price_buffer)
        full_volumes = self._get_ordered_buffer(self.volume_buffer)
        full_highs = self._get_ordered_buffer(self.high_buffer)
        full_lows = self._get_ordered_buffer(self.low_buffer)
        
        # Check if we have enough data
        current_len = full_prices.shape[1]
        
        if current_len < self.lookback:
            # Pad with the first value
            pad_len = self.lookback - current_len
            
            prices = np.pad(full_prices, ((0,0), (pad_len, 0)), mode='edge')
            volumes = np.pad(full_volumes, ((0,0), (pad_len, 0)), mode='edge')
            highs = np.pad(full_highs, ((0,0), (pad_len, 0)), mode='edge')
            lows = np.pad(full_lows, ((0,0), (pad_len, 0)), mode='edge')
        else:
            prices = full_prices[:, -self.lookback:]
            volumes = full_volumes[:, -self.lookback:]
            highs = full_highs[:, -self.lookback:]
            lows = full_lows[:, -self.lookback:]
        
        # 1. Correlation matrices (4, 64, 64)
        corr_tensor = self.correlation_engine.get_correlation_tensor()
        
        # 2. Technical indicators (n_indicators, 64)
        ind_dict = self.indicators.compute_all(prices, volumes, highs, lows)
        ind_tensor = self.indicators.to_tensor(ind_dict)
        
        # 3. Price history normalized (64, lookback, 4)
        price_history = self._encode_price_history(prices, volumes, highs, lows)
        
        # 4. Combined tensor for CNN
        combined = self._create_combined_tensor(corr_tensor, ind_tensor, price_history)
        
        return {
            'correlation': corr_tensor.to(self.device),
            'indicators': ind_tensor.to(self.device),
            'price_history': price_history.to(self.device),
            'combined': combined.to(self.device)
        }
    
    def _encode_price_history(
        self, 
        prices: np.ndarray,
        volumes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray
    ) -> torch.Tensor:
        """
        Encode price history with per-asset normalization.
        Shape: (64, lookback, 4) - close, volume, high, low (normalized)
        """
        result = np.zeros((self.n_assets, self.lookback, 4), dtype=np.float32)
        
        for i in range(self.n_assets):
            # Normalize price to [0, 1] within lookback
            p = prices[i]
            p_min, p_max = p.min(), p.max()
            if p_max > p_min:
                result[i, :, 0] = (p - p_min) / (p_max - p_min)
            else:
                result[i, :, 0] = 0.5
            
            # Normalize volume (z-score, clipped)
            v = volumes[i]
            v_mean, v_std = v.mean(), v.std()
            if v_std > 0:
                result[i, :, 1] = np.clip((v - v_mean) / v_std, -3, 3) / 6 + 0.5
            else:
                result[i, :, 1] = 0.5
            
            # High/low as distance from close
            if p_max > p_min:
                result[i, :, 2] = (highs[i] - p) / (p_max - p_min + 1e-8)
                result[i, :, 3] = (p - lows[i]) / (p_max - p_min + 1e-8)
        
        return torch.from_numpy(result)
    
    def _create_combined_tensor(
        self,
        corr: torch.Tensor,      # (4, 64, 64)
        indicators: torch.Tensor, # (n_ind, 64)
        history: torch.Tensor     # (64, lookback, 4)
    ) -> torch.Tensor:
        """
        Create combined tensor for CNN input.
        Shape: (C, 64, 64) where C = 4 + n_ind + lookback
        """
        n_ind = indicators.shape[0]
        
        # Start with correlation matrices
        combined = [corr]
        
        # Expand indicators to (n_ind, 64, 64) by repeating
        ind_expanded = indicators.unsqueeze(2).expand(-1, -1, 64)
        combined.append(ind_expanded)
        
        # Transpose history to (4, 64, lookback) and pad/interpolate to (4, 64, 64)
        hist_transposed = history.permute(2, 0, 1)  # (4, 64, lookback)
        if self.lookback != 64:
            # Interpolate to 64
            hist_transposed = torch.nn.functional.interpolate(
                hist_transposed.unsqueeze(0),
                size=(64, 64),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        combined.append(hist_transposed)
        
        return torch.cat(combined, dim=0)
    
    def get_lead_lag_signals(self, threshold: float = 0.3) -> Dict[int, list]:
        """
        Get current lead-lag signals above threshold.
        
        Returns:
            {leader_idx: [(follower_idx, correlation, lag_minutes), ...]}:
        """
        signals = {}
        lead_lag = self.correlation_engine.lead_lag_matrix
        
        for i in range(self.n_assets):
            followers = []
            for j in range(self.n_assets):
                if i != j and lead_lag[i, j] > threshold:
                    # i leads j
                    corr = self.correlation_engine.corr_matrices[60][i, j]
                    followers.append((j, corr, lead_lag[i, j]))
            
            if followers:
                followers.sort(key=lambda x: abs(x[1]), reverse=True)
                signals[i] = followers
        
        return signals

class GPUCorrelationComputer(nn.Module):
    """
    Compute correlation matrix on GPU for speed.
    """
    
    def __init__(self, n_assets: int = 64):
        super().__init__()
        self.n_assets = n_assets
    
    def forward(self, returns: torch.Tensor) -> torch.Tensor:
        """
        Compute correlation matrix.
        
        Args:
            returns: (n_assets, time_steps)
            
        Returns:
            correlation: (n_assets, n_assets)
        """
        # Standardize
        means = returns.mean(dim=1, keepdim=True)
        stds = returns.std(dim=1, keepdim=True).clamp(min=1e-8)
        standardized = (returns - means) / stds
        
        # Correlation = (X @ X.T) / n
        n = returns.shape[1]
        corr = torch.mm(standardized, standardized.t()) / n
        
        return corr
    
    def compute_lagged_correlation(
        self, 
        returns: torch.Tensor, 
        max_lag: int = 10
    ) -> torch.Tensor:
        """
        Compute correlation at different lags.
        
        Returns:
            (2*max_lag+1, n_assets, n_assets) - correlation at each lag
        """
        n_assets, T = returns.shape
        lags = range(-max_lag, max_lag + 1)
        
        results = []
        for lag in lags:
            if lag < 0:
                a = returns[:, -lag:]
                b = returns[:, :lag]
            elif lag > 0:
                a = returns[:, :-lag]
                b = returns[:, lag:]
            else:
                a = b = returns
            
            corr = self.forward(a) if lag == 0 else self._cross_corr(a, b)
            results.append(corr)
        
        return torch.stack(results, dim=0)
    
    def _cross_corr(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Cross-correlation between two return series"""
        a_std = (a - a.mean(dim=1, keepdim=True)) / a.std(dim=1, keepdim=True).clamp(min=1e-8)
        b_std = (b - b.mean(dim=1, keepdim=True)) / b.std(dim=1, keepdim=True).clamp(min=1e-8)
        return torch.mm(a_std, b_std.t()) / a.shape[1]


class FastStateEncoder(nn.Module):
    """
    End-to-end GPU state encoder.
    """
    
    def __init__(self, n_assets: int = 64, lookback: int = 60):
        super().__init__()
        self.n_assets = n_assets
        self.lookback = lookback
        self.corr_computer = GPUCorrelationComputer(n_assets)
        
        # Learnable indicator weights (let model learn what matters)
        self.indicator_importance = nn.Parameter(torch.ones(20))
        
    def forward(
        self,
        prices: torch.Tensor,    # (n_assets, lookback)
        volumes: torch.Tensor,   # (n_assets, lookback)
        highs: torch.Tensor,     # (n_assets, lookback)
        lows: torch.Tensor       # (n_assets, lookback)
    ) -> torch.Tensor:
        """
        Fast encoding on GPU.
        
        Returns:
            state: (C, n_assets, n_assets) ready for CNN
        """
        # Compute returns
        returns = (prices[:, 1:] - prices[:, :-1]) / prices[:, :-1].clamp(min=1e-8)
        
        # Correlation at multiple timeframes
        corr_15 = self.corr_computer(returns[:, -15:])
        corr_60 = self.corr_computer(returns)
        
        # Lead-lag (simplified - use argmax of lagged correlations)
        # Need roughly 30 points for lag check
        if returns.shape[1] >= 30:
             lagged = self.corr_computer.compute_lagged_correlation(returns[:, -30:], max_lag=5)
             lead_lag = lagged.max(dim=0)[0] - lagged.min(dim=0)[0]  # Asymmetry
        else:
             lead_lag = torch.zeros(self.n_assets, self.n_assets, device=prices.device)

        
        # Stack correlation layers
        corr_stack = torch.stack([corr_15, corr_60, lead_lag], dim=0)
        
        # Price features as image (simplified)
        price_norm = (prices - prices.min(dim=1, keepdim=True)[0]) / \
                     (prices.max(dim=1, keepdim=True)[0] - prices.min(dim=1, keepdim=True)[0] + 1e-8)
        
        # Interpolate to 64x64 if needed
        if self.lookback != self.n_assets:
            price_img = torch.nn.functional.interpolate(
                price_norm.unsqueeze(0).unsqueeze(0),
                size=(self.n_assets, self.n_assets),
                mode='bilinear',
                align_corners=False
            ).squeeze()
        else:
            price_img = price_norm
        
        # Combine
        combined = torch.cat([corr_stack, price_img.unsqueeze(0)], dim=0)
        
        return combined
