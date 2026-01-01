import numpy as np
from typing import Dict, Optional
import logging

class SyntheticMarket:
    """
    Generates synthetic OHLCV data for curriculum training.
    Compatible with HistoricalMarket interface.
    """

    def __init__(
        self,
        level: int,
        n_assets: int = 64,
        n_steps: int = 10000,
        seed: Optional[int] = None
    ):
        """
        Args:
            level: Curriculum level (0-3)
            n_assets: Number of synthetic assets
            n_steps: Total steps in episode
            seed: Random seed for reproducibility
        """
        self.level = level
        self.n_assets = n_assets
        self.n_steps = n_steps
        self.current_step = 0

        # Bug Fix 1: Removed global seed. Use local RNG for isolation.
        self.rng = np.random.RandomState(seed)

        logging.info(f"Generating Level {level} synthetic data ({n_assets} assets, {n_steps} steps)...")

        # Generate all data upfront
        self.data = self._generate_level_data()

        # Bug Fix 5: Validation
        self._validate_data()

        # Create asset map (compatible with HistoricalMarket)
        self.assets = {i: f'SYN_{i:02d}' for i in range(n_assets)}
        self.symbol_map = self.assets
        self.symbol_to_id = {v: k for k, v in self.assets.items()}

        # Valid timestamps (all steps are valid)
        self.valid_timestamps = list(range(n_steps))
        self.min_time = 0
        self.max_time = n_steps - 1

        logging.info(f"✅ Synthetic Level {level} data ready")

    def _generate_level_data(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Route to appropriate generator"""
        generators = {
            0: self._level_0_sine_waves,
            1: self._level_1_lagged_pairs,
            2: self._level_2_dip_crashes,
            3: self._level_3_noisy_trends,
            4: self._level_4_portfolio_rotation
        }

        if self.level not in generators:
            raise ValueError(f"Unknown level: {self.level}")

        return generators[self.level]()

    def _validate_data(self):
        """Bug Fix 5: Ensure data integrity (no NaNs, Infs, or negative prices)"""
        for symbol, ohlc in self.data.items():
            for key in ['open', 'high', 'low', 'close']:
                arr = ohlc[key]
                if not np.isfinite(arr).all():
                    raise ValueError(f"Data validation failed: {symbol} {key} contains NaNs or Infs")
                if (arr <= 0).any():
                    raise ValueError(f"Data validation failed: {symbol} {key} contains non-positive values")
            
            # Check OHLC consistency
            if not ((ohlc['low'] <= ohlc['open']) & (ohlc['open'] <= ohlc['high'])).all():
                 raise ValueError(f"Data validation failed: {symbol} Open outside [Low, High]")
            if not ((ohlc['low'] <= ohlc['close']) & (ohlc['close'] <= ohlc['high'])).all():
                 raise ValueError(f"Data validation failed: {symbol} Close outside [Low, High]")

    def reset(self):
        """Reset to beginning of episode"""
        self.current_step = 0
        return self.get_current_data()

    def step(self) -> Dict[str, Dict]:
        """Advance 1 step, return new data"""
        self.current_step += 1

        if self.current_step >= self.n_steps:
            return {}  # Episode done

        return self.get_current_data()

    def get_current_data(self) -> Dict[str, Dict]:
        """Get current step data (compatible with HistoricalMarket)"""
        if self.current_step >= self.n_steps:
            return {}

        current_data = {}

        for i in range(self.n_assets):
            symbol = f'SYN_{i:02d}'
            current_data[symbol] = {
                'timestamp': self.current_step,
                'open': float(self.data[symbol]['open'][self.current_step]),
                'high': float(self.data[symbol]['high'][self.current_step]),
                'low': float(self.data[symbol]['low'][self.current_step]),
                'close': float(self.data[symbol]['close'][self.current_step]),
                'volume': 1000.0
            }

        return current_data

    def get_current_fx_rates(self) -> Dict[str, float]:
        """Mock FX rates (all USD)"""
        return {}

    # ========== Level Generators ==========

    def _level_0_sine_waves(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Level 0: Pure sine waves, zero noise.
        """
        data = {}

        for i in range(self.n_assets):
            # Deterministic parameters per asset if seed is set
            period = self.rng.randint(10, 50)
            amplitude = self.rng.uniform(0.05, 0.20)
            phase = self.rng.uniform(0, 2*np.pi)
            baseline = 100.0

            t = np.arange(self.n_steps)
            close_prices = baseline * (1 + amplitude * np.sin(2*np.pi*t/period + phase))

            # Bug Fix 2: Deterministic OHLC (no random spread)
            # Create a synthetic spread that is purely a function of the price
            spread = close_prices * 0.001 
            
            data[f'SYN_{i:02d}'] = {
                'open': close_prices, # Open = Close of previous roughly, let's just say Open=Close for simplicity or Open=Close-delta
                'high': close_prices + spread,
                'low': close_prices - spread,
                'close': close_prices
            }
            # Note: For pure sine, Open=Close is fine, or Open=Close(t-1). 
            # Simple deterministic structure: Open=Close, High=Close+0.1%, Low=Close-0.1%

        return data

    def _level_1_lagged_pairs(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Level 1: Lagged pairs - test correlation detection.
        """
        data = {}
        LAG = 10 
        n_leaders = self.n_assets // 2
        
        # Leaders
        leaders = {}
        for i in range(n_leaders):
            period = self.rng.randint(15, 40)
            amplitude = self.rng.uniform(0.10, 0.25)
            phase = self.rng.uniform(0, 2*np.pi)
            t = np.arange(self.n_steps)
            close = 100 * (1 + amplitude * np.sin(2*np.pi*t/period + phase))
            leaders[i] = close
            data[f'SYN_{i:02d}'] = self._prices_to_ohlc(close) # Random spread ok here

        # Followers
        for i in range(n_leaders):
            leader_idx = i
            follower_idx = i + n_leaders
            if follower_idx >= self.n_assets: break

            leader_close = leaders[leader_idx]
            follower_close = np.roll(leader_close, LAG)
            follower_close[:LAG] = leader_close[0]

            noise = self.rng.normal(0, 0.01, size=self.n_steps)
            follower_close *= (1 + noise)

            data[f'SYN_{follower_idx:02d}'] = self._prices_to_ohlc(follower_close)

        return data

    def _level_2_dip_crashes(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Level 2: Rare V-shaped crashes.
        """
        data = {}
        BASELINE = 100.0
        CRASH_PROB = 0.02
        CRASH_SIZE = 0.10

        for i in range(self.n_assets):
            prices = np.ones(self.n_steps) * BASELINE
            
            # Vectorized crash generation for speed/cleanliness
            crashes = self.rng.rand(self.n_steps) < CRASH_PROB
            prices[crashes] *= (1 - CRASH_SIZE)

            noise = self.rng.normal(0, 0.005, size=self.n_steps)
            prices *= (1 + noise)

            data[f'SYN_{i:02d}'] = self._prices_to_ohlc(prices)

        return data

    def _level_3_noisy_trends(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Level 3: Random Walk with Drift (Trends).
        Bug Fix 3: clearer math.
        """
        data = {}
        TREND_SWITCH_PROB = 0.05 # 5% chance to switch trend state
        TREND_STRENGTH = 0.005   # 0.5% drift per step
        NOISE_STD = 0.02         # 2% volatility

        for i in range(self.n_assets):
            prices = np.zeros(self.n_steps)
            prices[0] = 100.0
            
            # State: 0=Flat, 1=Up, -1=Down
            trend_state = 0
            
            for t in range(1, self.n_steps):
                # Markov chain for trend state
                if self.rng.rand() < TREND_SWITCH_PROB:
                    # Switch to a random new state
                    trend_state = self.rng.choice([-1, 0, 1])
                
                # Drift component
                drift = trend_state * TREND_STRENGTH
                
                # Noise component
                noise = self.rng.normal(0, NOISE_STD)
                
                # Multiplicative random walk
                change = 1 + drift + noise
                prices[t] = prices[t-1] * change
                
                # Floor at 0.01 to prevent negative/zero
                if prices[t] < 0.01: prices[t] = 0.01

            data[f'SYN_{i:02d}'] = self._prices_to_ohlc(prices)

        return data

    def _level_4_portfolio_rotation(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Level 4: Rotating leadership.
        """
        data = {}
        SECTOR_SIZE = 8
        N_SECTORS = self.n_assets // SECTOR_SIZE
        ROTATION_PERIOD = 50
        PUMP_MAGNITUDE = 0.03

        # Base GBM
        base_prices = {}
        for i in range(self.n_assets):
            prices = np.zeros(self.n_steps)
            prices[0] = 100.0
            
            # Vectorized GBM
            returns = self.rng.normal(0, 0.01, size=self.n_steps)
            # Cumulative product for GBM
            # P_t = P_0 * prod(1+r)
            prices = 100.0 * np.cumprod(1 + returns)
            base_prices[i] = prices

        # Add Pumps
        if N_SECTORS > 0:
            for t in range(self.n_steps):
                leading_sector = (t // ROTATION_PERIOD) % N_SECTORS
                
                # Identify sector assets
                start_idx = leading_sector * SECTOR_SIZE
                end_idx = start_idx + SECTOR_SIZE
                
                # Apply pump to base prices in place? 
                # Better to apply factor during generation, but modifying array is fine.
                # Only apply if t % ROTATION_PERIOD < ROTATION_PERIOD (always true for full period leadership)
                # Task said "During pump", which implies the whole period.
                
                for asset_idx in range(start_idx, min(end_idx, self.n_assets)):
                     base_prices[asset_idx][t] *= (1 + PUMP_MAGNITUDE)

        for i in range(self.n_assets):
            data[f'SYN_{i:02d}'] = self._prices_to_ohlc(base_prices[i])

        return data

    def _prices_to_ohlc(self, close_prices: np.ndarray) -> Dict[str, np.ndarray]:
        """Helper: Convert close prices to OHLC with constraints"""
        spread = 0.002
        
        # Generate raw variations
        u1 = self.rng.uniform(-spread, spread, size=len(close_prices))
        u2 = self.rng.uniform(0, spread*2, size=len(close_prices))
        u3 = self.rng.uniform(0, spread*2, size=len(close_prices))
        
        open_prices = close_prices * (1 + u1)
        # Preliminary High/Low based on Close
        high_prices = close_prices * (1 + u2)
        low_prices = close_prices * (1 - u3)

        # Bug Fix 4: Strict Enveloping
        # High must be max of (Open, Close, proposed High)
        high_prices = np.maximum(high_prices, np.maximum(open_prices, close_prices))
        # Low must be min of (Open, Close, proposed Low)
        low_prices = np.minimum(low_prices, np.minimum(open_prices, close_prices))
        
        return {
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices
        }