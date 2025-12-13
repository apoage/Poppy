**PROMPT 3: Multi-Asset State Encoder**

```markdown
# Multi-Asset State Encoder - Correlation Matrix & Signal Processing

## Objective
Encode 64 assets' price/volume data + cross-asset correlations into tensor format suitable for TRM+MCTS agent. Must capture lead-lag relationships.

## Core Components

### 1. Correlation Matrix Computation

```python
"""
correlation.py - Rolling correlation matrix for 64 assets
"""

import numpy as np
import torch
from typing import Dict, List, Tuple
from numba import jit, prange
import pandas as pd

class CorrelationEngine:
    """
    Computes rolling correlations between all asset pairs.
    Key for detecting lead-lag relationships.
    """
    
    def __init__(
        self, 
        n_assets: int = 64,
        lookback_windows: List[int] = [15, 60, 240],  # 15min, 1hr, 4hr
        lag_range: int = 10  # Check correlations with ±10 minute lags
    ):
        self.n_assets = n_assets
        self.windows = lookback_windows
        self.lag_range = lag_range
        
        # Pre-allocate correlation matrices
        self.corr_matrices = {
            w: np.zeros((n_assets, n_assets), dtype=np.float32)
            for w in lookback_windows
        }
        
        # Lead-lag matrix: positive = row leads column
        self.lead_lag_matrix = np.zeros((n_assets, n_assets), dtype=np.float32)
        
    def update(self, returns: np.ndarray):
        """
        Update correlation matrices with new returns data.
        
        Args:
            returns: Shape (n_assets, lookback) - recent returns for each asset
        """
        for window in self.windows:
            if returns.shape[1] >= window:
                window_returns = returns[:, -window:]
                self.corr_matrices[window] = self._compute_correlation(window_returns)
        
        # Update lead-lag relationships
        if returns.shape[1] >= max(self.windows) + self.lag_range:
            self._update_lead_lag(returns)
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def _compute_correlation(returns: np.ndarray) -> np.ndarray:
        """Fast correlation computation with Numba"""
        n = returns.shape[0]
        corr = np.zeros((n, n), dtype=np.float32)
        
        # Standardize
        means = np.zeros(n)
        stds = np.zeros(n)
        for i in range(n):
            means[i] = np.mean(returns[i])
            stds[i] = np.std(returns[i])
            if stds[i] < 1e-8:
                stds[i] = 1e-8
        
        standardized = np.zeros_like(returns)
        for i in range(n):
            standardized[i] = (returns[i] - means[i]) / stds[i]
        
        # Compute correlations
        for i in prange(n):
            for j in range(i, n):
                corr[i, j] = np.mean(standardized[i] * standardized[j])
                corr[j, i] = corr[i, j]
        
        return corr
    
    def _update_lead_lag(self, returns: np.ndarray):
        """
        Compute lead-lag relationships.
        Positive value at [i,j] means asset i leads asset j.
        """
        n = self.n_assets
        window = 60  # Use 1hr for lead-lag
        
        for i in range(n):
            for j in range(i + 1, n):
                best_lag, best_corr = self._find_best_lag(
                    returns[i], returns[j], window
                )
                # Positive lag means i leads j
                self.lead_lag_matrix[i, j] = best_lag * best_corr
                self.lead_lag_matrix[j, i] = -best_lag * best_corr
    
    def _find_best_lag(
        self, 
        series_a: np.ndarray, 
        series_b: np.ndarray,
        window: int
    ) -> Tuple[int, float]:
        """Find lag that maximizes correlation"""
        best_lag = 0
        best_corr = 0
        
        a = series_a[-window:]
        
        for lag in range(-self.lag_range, self.lag_range + 1):
            if lag < 0:
                b = series_b[-window + lag:lag] if lag != 0 else series_b[-window:]
            else:
                b = series_b[-window - lag:-lag] if lag != 0 else series_b[-window:]
            
            if len(a) == len(b) and len(a) > 0:
                corr = np.corrcoef(a, b)[0, 1]
                if abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_lag = lag
        
        return best_lag, best_corr
    
    def get_correlation_tensor(self) -> torch.Tensor:
        """
        Returns stacked correlation matrices as tensor.
        Shape: (n_windows + 1, n_assets, n_assets)
        """
        matrices = [self.corr_matrices[w] for w in sorted(self.windows)]
        matrices.append(self.lead_lag_matrix)
        
        return torch.from_numpy(np.stack(matrices, axis=0))
    
    def get_strongest_correlations(self, asset_idx: int, top_k: int = 5) -> List[Tuple[int, float, float]]:
        """Get most correlated assets for given asset"""
        corrs = self.corr_matrices[60][asset_idx]  # Use 1hr
        leads = self.lead_lag_matrix[asset_idx]
        
        # Sort by absolute correlation
        indices = np.argsort(np.abs(corrs))[::-1]
        
        results = []
        for idx in indices[1:top_k+1]:  # Skip self
            results.append((idx, corrs[idx], leads[idx]))
        
        return results
```

### 2. Technical Indicators (Per Asset)

```python
"""
indicators.py - Technical indicators for each asset
"""

import numpy as np
import torch
from typing import Dict
from numba import jit

class TechnicalIndicators:
    """
    Compute standard technical indicators.
    All indicators normalized to [-1, 1] or [0, 1] range.
    """
    
    def __init__(self, n_assets: int = 64):
        self.n_assets = n_assets
        
    def compute_all(
        self, 
        prices: np.ndarray,      # (n_assets, lookback)
        volumes: np.ndarray,     # (n_assets, lookback)
        highs: np.ndarray,       # (n_assets, lookback)
        lows: np.ndarray         # (n_assets, lookback)
    ) -> Dict[str, np.ndarray]:
        """Compute all indicators for all assets"""
        
        indicators = {}
        
        # Price-based
        indicators['returns'] = self._returns(prices)
        indicators['log_returns'] = self._log_returns(prices)
        indicators['rsi'] = self._rsi(prices, period=14)
        indicators['rsi_fast'] = self._rsi(prices, period=7)
        indicators['macd'] = self._macd(prices)
        indicators['macd_signal'] = self._macd_signal(prices)
        indicators['bollinger_pos'] = self._bollinger_position(prices)
        indicators['momentum'] = self._momentum(prices, period=10)
        indicators['roc'] = self._rate_of_change(prices, period=10)
        
        # Volume-based
        indicators['volume_sma_ratio'] = self._volume_sma_ratio(volumes)
        indicators['obv_slope'] = self._obv_slope(prices, volumes)
        indicators['vwap_dist'] = self._vwap_distance(prices, volumes, highs, lows)
        
        # Volatility
        indicators['atr'] = self._atr(prices, highs, lows)
        indicators['volatility'] = self._volatility(prices)
        
        # Trend
        indicators['sma_cross'] = self._sma_cross(prices)
        indicators['ema_trend'] = self._ema_trend(prices)
        indicators['adx'] = self._adx(prices, highs, lows)
        
        # Support/Resistance
        indicators['dist_to_high'] = self._distance_to_high(prices, highs)
        indicators['dist_to_low'] = self._distance_to_low(prices, lows)
        
        return indicators
    
    @staticmethod
    @jit(nopython=True)
    def _returns(prices: np.ndarray) -> np.ndarray:
        """Simple returns, last value"""
        n_assets = prices.shape[0]
        result = np.zeros(n_assets, dtype=np.float32)
        for i in range(n_assets):
            if prices[i, -2] > 0:
                result[i] = (prices[i, -1] - prices[i, -2]) / prices[i, -2]
        return np.clip(result, -0.1, 0.1) * 10  # Scale to roughly [-1, 1]
    
    @staticmethod
    @jit(nopython=True)
    def _log_returns(prices: np.ndarray) -> np.ndarray:
        n_assets = prices.shape[0]
        result = np.zeros(n_assets, dtype=np.float32)
        for i in range(n_assets):
            if prices[i, -2] > 0 and prices[i, -1] > 0:
                result[i] = np.log(prices[i, -1] / prices[i, -2])
        return np.clip(result, -0.1, 0.1) * 10
    
    @staticmethod
    @jit(nopython=True)
    def _rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
        """RSI normalized to [-1, 1]"""
        n_assets = prices.shape[0]
        result = np.zeros(n_assets, dtype=np.float32)
        
        for i in range(n_assets):
            if prices.shape[1] < period + 1:
                continue
                
            gains = 0.0
            losses = 0.0
            
            for j in range(-period, 0):
                change = prices[i, j] - prices[i, j-1]
                if change > 0:
                    gains += change
                else:
                    losses -= change
            
            if losses > 0:
                rs = gains / losses
                rsi = 100 - (100 / (1 + rs))
            else:
                rsi = 100
            
            result[i] = (rsi - 50) / 50  # Normalize to [-1, 1]
        
        return result
    
    @staticmethod
    def _macd(prices: np.ndarray, fast: int = 12, slow: int = 26) -> np.ndarray:
        """MACD line normalized"""
        n_assets = prices.shape[0]
        result = np.zeros(n_assets, dtype=np.float32)
        
        for i in range(n_assets):
            ema_fast = TechnicalIndicators._ema(prices[i], fast)
            ema_slow = TechnicalIndicators._ema(prices[i], slow)
            macd = ema_fast - ema_slow
            # Normalize by price
            if prices[i, -1] > 0:
                result[i] = macd / prices[i, -1] * 100
        
        return np.clip(result, -1, 1)
    
    @staticmethod
    def _ema(series: np.ndarray, period: int) -> float:
        """Exponential moving average"""
        alpha = 2 / (period + 1)
        ema = series[0]
        for val in series[1:]:
            ema = alpha * val + (1 - alpha) * ema
        return ema
    
    @staticmethod
    def _bollinger_position(prices: np.ndarray, period: int = 20) -> np.ndarray:
        """Position within Bollinger Bands [-1, 1]"""
        n_assets = prices.shape[0]
        result = np.zeros(n_assets, dtype=np.float32)
        
        for i in range(n_assets):
            if prices.shape[1] < period:
                continue
            
            window = prices[i, -period:]
            sma = np.mean(window)
            std = np.std(window)
            
            if std > 0:
                result[i] = (prices[i, -1] - sma) / (2 * std)
        
        return np.clip(result, -1, 1)
    
    @staticmethod
    def _atr(prices: np.ndarray, highs: np.ndarray, lows: np.ndarray, period: int = 14) -> np.ndarray:
        """Average True Range normalized by price"""
        n_assets = prices.shape[0]
        result = np.zeros(n_assets, dtype=np.float32)
        
        for i in range(n_assets):
            if prices.shape[1] < period + 1:
                continue
            
            tr_sum = 0.0
            for j in range(-period, 0):
                high_low = highs[i, j] - lows[i, j]
                high_close = abs(highs[i, j] - prices[i, j-1])
                low_close = abs(lows[i, j] - prices[i, j-1])
                tr_sum += max(high_low, high_close, low_close)
            
            atr = tr_sum / period
            if prices[i, -1] > 0:
                result[i] = atr / prices[i, -1]  # As percentage
        
        return np.clip(result, 0, 0.1) * 10  # Scale to [0, 1]
    
    @staticmethod
    def _volatility(prices: np.ndarray, period: int = 20) -> np.ndarray:
        """Rolling volatility normalized"""
        n_assets = prices.shape[0]
        result = np.zeros(n_assets, dtype=np.float32)
        
        for i in range(n_assets):
            if prices.shape[1] < period + 1:
                continue
            
            returns = np.diff(np.log(prices[i, -period-1:] + 1e-8))
            result[i] = np.std(returns) * np.sqrt(period)
        
        return np.clip(result, 0, 0.5) * 2  # Scale to [0, 1]
    
    # ... Additional indicators (momentum, sma_cross, etc.) follow same pattern
    
    @staticmethod
    def _momentum(prices: np.ndarray, period: int = 10) -> np.ndarray:
        n_assets = prices.shape[0]
        result = np.zeros(n_assets, dtype=np.float32)
        for i in range(n_assets):
            if prices.shape[1] > period and prices[i, -period-1] > 0:
                result[i] = (prices[i, -1] - prices[i, -period-1]) / prices[i, -period-1]
        return np.clip(result, -0.2, 0.2) * 5
    
    @staticmethod
    def _volume_sma_ratio(volumes: np.ndarray, period: int = 20) -> np.ndarray:
        n_assets = volumes.shape[0]
        result = np.zeros(n_assets, dtype=np.float32)
        for i in range(n_assets):
            if volumes.shape[1] >= period:
                sma = np.mean(volumes[i, -period:])
                if sma > 0:
                    result[i] = volumes[i, -1] / sma - 1
        return np.clip(result, -2, 2) / 2
    
    @staticmethod
    def _sma_cross(prices: np.ndarray, fast: int = 10, slow: int = 30) -> np.ndarray:
        """SMA crossover signal"""
        n_assets = prices.shape[0]
        result = np.zeros(n_assets, dtype=np.float32)
        for i in range(n_assets):
            if prices.shape[1] >= slow:
                sma_fast = np.mean(prices[i, -fast:])
                sma_slow = np.mean(prices[i, -slow:])
                if sma_slow > 0:
                    result[i] = (sma_fast - sma_slow) / sma_slow * 10
        return np.clip(result, -1, 1)

    def to_tensor(self, indicators: Dict[str, np.ndarray]) -> torch.Tensor:
        """Stack all indicators into tensor (n_indicators, n_assets)"""
        arrays = [indicators[k] for k in sorted(indicators.keys())]
        return torch.from_numpy(np.stack(arrays, axis=0).astype(np.float32))
```

### 3. Combined State Encoder

```python
"""
state_encoder.py - Combined multi-asset state encoder
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional

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
        device: str = 'cuda'
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
        prices = self._get_ordered_buffer(self.price_buffer)[:, -self.lookback:]
        volumes = self._get_ordered_buffer(self.volume_buffer)[:, -self.lookback:]
        highs = self._get_ordered_buffer(self.high_buffer)[:, -self.lookback:]
        lows = self._get_ordered_buffer(self.low_buffer)[:, -self.lookback:]
        
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
            {leader_idx: [(follower_idx, correlation, lag_minutes), ...]}
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
```

### 4. GPU-Accelerated Batch Encoding

```python
"""
gpu_encoder.py - GPU-accelerated encoding for real-time processing
"""

import torch
import torch.nn as nn

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
        lagged = self.corr_computer.compute_lagged_correlation(returns[:, -30:], max_lag=5)
        lead_lag = lagged.max(dim=0)[0] - lagged.min(dim=0)[0]  # Asymmetry
        
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
                mode='bilinear'
            ).squeeze()
        else:
            price_img = price_norm
        
        # Combine
        combined = torch.cat([corr_stack, price_img.unsqueeze(0)], dim=0)
        
        return combined
```

## Usage Example

```python
# Initialize
encoder = MultiAssetStateEncoder(n_assets=64, lookback=60, device='cuda')
encoder.register_assets(asset_symbols)  # List of 64 symbols

# Update with new data (called every minute)
encoder.update(market_data)

# Get encoded state for agent
state = encoder.encode()

# Feed to TRM+MCTS
policy, value = agent(state['combined'])

# Get actionable signals
signals = encoder.get_lead_lag_signals(threshold=0.3)
for leader, followers in signals.items():
    print(f"{encoder.asset_map[leader]} leads: {[(encoder.asset_map[f[0]], f[1]) for f in followers[:3]]}")
```

## Output Tensor Shapes

| Component | Shape | Description |
|-----------|-------|-------------|
| correlation | (4, 64, 64) | 3 timeframes + lead-lag |
| indicators | (~20, 64) | Technical indicators per asset |
| price_history | (64, 60, 4) | OHLC history normalized |
| combined | (~27, 64, 64) | CNN-ready tensor |
```
