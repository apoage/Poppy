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
    def _macd_signal(prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> np.ndarray:
        # Simplified placeholder: return MACD - Signal (histogram)
        # Proper implementation requires maintaining history of MACD values
        # For now, we return 0 to avoid error, assume MACD is covered by _macd
        return np.zeros(prices.shape[0], dtype=np.float32)

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
    
    @staticmethod
    def _momentum(prices: np.ndarray, period: int = 10) -> np.ndarray:
        n_assets = prices.shape[0]
        result = np.zeros(n_assets, dtype=np.float32)
        for i in range(n_assets):
            if prices.shape[1] > period and prices[i, -period-1] > 0:
                result[i] = (prices[i, -1] - prices[i, -period-1]) / prices[i, -period-1]
        return np.clip(result, -0.2, 0.2) * 5
    
    @staticmethod
    def _rate_of_change(prices: np.ndarray, period: int = 10) -> np.ndarray:
        return TechnicalIndicators._momentum(prices, period)
    
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
    def _obv_slope(prices: np.ndarray, volumes: np.ndarray) -> np.ndarray:
        # Placeholder
        return np.zeros(prices.shape[0], dtype=np.float32)

    @staticmethod
    def _vwap_distance(prices: np.ndarray, volumes: np.ndarray, highs: np.ndarray, lows: np.ndarray) -> np.ndarray:
        # Placeholder
        return np.zeros(prices.shape[0], dtype=np.float32)
    
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

    @staticmethod
    def _ema_trend(prices: np.ndarray) -> np.ndarray:
        # Placeholder
        return np.zeros(prices.shape[0], dtype=np.float32)
        
    @staticmethod
    def _adx(prices: np.ndarray, highs: np.ndarray, lows: np.ndarray) -> np.ndarray:
        # Placeholder
        return np.zeros(prices.shape[0], dtype=np.float32)

    @staticmethod
    def _distance_to_high(prices: np.ndarray, highs: np.ndarray) -> np.ndarray:
        # Placeholder
        return np.zeros(prices.shape[0], dtype=np.float32)

    @staticmethod
    def _distance_to_low(prices: np.ndarray, lows: np.ndarray) -> np.ndarray:
        # Placeholder
        return np.zeros(prices.shape[0], dtype=np.float32)

    def to_tensor(self, indicators: Dict[str, np.ndarray]) -> torch.Tensor:
        """Stack all indicators into tensor (n_indicators, n_assets)"""
        arrays = [indicators[k] for k in sorted(indicators.keys())]
        return torch.from_numpy(np.stack(arrays, axis=0).astype(np.float32))
