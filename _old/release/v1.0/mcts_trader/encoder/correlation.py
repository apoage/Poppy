import numpy as np
import torch
from typing import Dict, List, Tuple
from numba import jit, prange

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
