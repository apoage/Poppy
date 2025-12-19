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

        if seed is not None:
            np.random.seed(seed)
        
        # Keep internal RNG state to ensure reproducibility if needed elsewhere
        self.rng = np.random.RandomState(seed)

        logging.info(f"Generating Level {level} synthetic data ({n_assets} assets, {n_steps} steps)...")

        # Generate all data upfront
        self.data = self._generate_level_data()

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
            4: self._level_4_portfolio_rotation # Placeholder for future extension
        }

        if self.level not in generators:
            raise ValueError(f"Unknown level: {self.level}")

        return generators[self.level]()

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

        Goal: Test basic buy low / sell high.
        Each asset is a sine wave with random period/amplitude.
        """
        data = {}

        for i in range(self.n_assets):
            # Random parameters
            period = self.rng.randint(10, 50)      # 10-50 steps per cycle
            amplitude = self.rng.uniform(0.05, 0.20)  # 5-20% amplitude
            phase = self.rng.uniform(0, 2*np.pi)   # Random phase
            baseline = 100.0

            # Generate sine wave
            t = np.arange(self.n_steps)
            close_prices = baseline * (1 + amplitude * np.sin(2*np.pi*t/period + phase))

            # OHLC with tiny spread (no randomness in spread for pure sine wave level?) 
            # Original task desc said "no randomness".
            # But _prices_to_ohlc adds spread randomness. Let's keep it simple here as per spec "zero noise".
            # Actually, standardizing on _prices_to_ohlc is better for consistency, but for Level 0 "Pure sine waves",
            # explicit clean OHLC is probably better.
            
            open_prices = close_prices * 0.9995
            high_prices = close_prices * 1.0005
            low_prices = close_prices * 0.9995

            data[f'SYN_{i:02d}'] = {
                'open': open_prices,
                'high': high_prices,
                'low': low_prices,
                'close': close_prices
            }

        return data

    def _level_1_lagged_pairs(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Level 1: Lagged pairs - test correlation detection.

        Goal: Prove correlation matrix encoder works.
        32 "leader" assets (sine waves)
        32 "follower" assets (lagged copies with 1% noise)
        """
        data = {}
        LAG = 10  # 10-step lag

        # Ensure we have enough assets for pairs
        n_leaders = self.n_assets // 2
        
        # Generate leader assets
        leaders = {}
        for i in range(n_leaders):
            period = self.rng.randint(15, 40)
            amplitude = self.rng.uniform(0.10, 0.25)
            phase = self.rng.uniform(0, 2*np.pi)

            t = np.arange(self.n_steps)
            close = 100 * (1 + amplitude * np.sin(2*np.pi*t/period + phase))
            leaders[i] = close
            
            data[f'SYN_{i:02d}'] = self._prices_to_ohlc(close)

        # Create followers (lagged + noise)
        for i in range(n_leaders):
            leader_idx = i
            follower_idx = i + n_leaders
            
            if follower_idx >= self.n_assets:
                break

            # Leader
            leader_close = leaders[leader_idx]

            # Follower (lagged)
            follower_close = np.roll(leader_close, LAG)
            follower_close[:LAG] = leader_close[0]  # Fill gap

            # Add 1% noise
            noise = self.rng.normal(0, 0.01, size=self.n_steps)
            follower_close *= (1 + noise)

            data[f'SYN_{follower_idx:02d}'] = self._prices_to_ohlc(follower_close)

        return data

    def _level_2_dip_crashes(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Level 2: Rare V-shaped crashes - test patience.

        Goal: Learn to wait for opportunities, then act fast.
        Flat baseline with rare -10% crashes that recover in 1 step.
        """
        data = {}
        BASELINE = 100.0
        CRASH_PROB = 0.02  # 2% chance per step
        CRASH_SIZE = 0.10  # 10% drop

        for i in range(self.n_assets):
            prices = np.ones(self.n_steps) * BASELINE

            # Add crashes
            # We iterate to apply crashes. Note: modifying array in place.
            # Using random mask is faster but task specified logic.
            # Let's use a loop for clarity as per spec.
            for t in range(self.n_steps - 1):
                if self.rng.rand() < CRASH_PROB:
                    prices[t] *= (1 - CRASH_SIZE)  # Crash
                    # Instant recovery (next step back to baseline)

            # Add tiny noise (0.5%)
            noise = self.rng.normal(0, 0.005, size=self.n_steps)
            prices *= (1 + noise)

            data[f'SYN_{i:02d}'] = self._prices_to_ohlc(prices)

        return data

    def _level_3_noisy_trends(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Level 3: Noisy random walk with trends - test fee awareness.

        Goal: Only trade during strong trends (overcome 31 bps cost).
        60% of time in trend (±0.5% drift per step)
        40% choppy (pure noise)
        """
        data = {}
        TREND_PROB = 0.60
        TREND_LENGTH = 20
        TREND_DRIFT = 0.005  # 0.5% per step
        NOISE_STD = 0.02     # 2% noise

        for i in range(self.n_assets):
            prices = [100.0]
            in_trend = False
            trend_remaining = 0
            trend_direction = 1

            for t in range(1, self.n_steps):
                # Enter trend?
                if not in_trend and self.rng.rand() < (TREND_PROB / TREND_LENGTH):
                    in_trend = True
                    trend_remaining = TREND_LENGTH
                    trend_direction = self.rng.choice([-1, 1])

                # Drift (if in trend)
                drift = TREND_DRIFT * trend_direction if in_trend else 0

                # Noise
                noise = self.rng.normal(0, NOISE_STD)

                # Update
                new_price = prices[-1] * (1 + drift + noise)
                prices.append(max(new_price, 1.0))  # Floor at $1

                # Decrement trend
                if in_trend:
                    trend_remaining -= 1
                    if trend_remaining <= 0:
                        in_trend = False

            prices = np.array(prices)
            data[f'SYN_{i:02d}'] = self._prices_to_ohlc(prices)

        return data

    def _level_4_portfolio_rotation(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Level 4: Rotating leadership - test portfolio morphing.
        (Placeholder from Extension Task)
        """
        data = {}
        SECTOR_SIZE = 8
        N_SECTORS = self.n_assets // SECTOR_SIZE
        ROTATION_PERIOD = 50
        PUMP_MAGNITUDE = 0.03  # 3% per step during leadership

        # Generate base GBM for all assets
        base_prices = {}
        for i in range(self.n_assets):
            # Flat with small noise
            prices = [100.0]
            for t in range(1, self.n_steps):
                noise = self.rng.normal(0, 0.01)
                prices.append(prices[-1] * (1 + noise))
            base_prices[i] = np.array(prices)

        # Add rotation pumps
        for t in range(self.n_steps):
            # Which sector is leading?
            if N_SECTORS > 0:
                leading_sector = (t // ROTATION_PERIOD) % N_SECTORS

                # Pump that sector
                for i in range(SECTOR_SIZE):
                    asset_idx = leading_sector * SECTOR_SIZE + i
                    if asset_idx < self.n_assets:
                        if t % ROTATION_PERIOD < ROTATION_PERIOD:  # During pump
                            base_prices[asset_idx][t] *= (1 + PUMP_MAGNITUDE)

        # Convert to OHLC
        for i in range(self.n_assets):
            data[f'SYN_{i:02d}'] = self._prices_to_ohlc(base_prices[i])

        return data

    def _prices_to_ohlc(self, close_prices: np.ndarray) -> Dict[str, np.ndarray]:
        """Helper: Convert close prices to OHLC"""
        # Simple OHLC generation (close ± random spread)
        spread = 0.002  # 0.2% spread

        open_prices = close_prices * (1 + self.rng.uniform(-spread, spread, size=len(close_prices)))
        high_prices = close_prices * (1 + self.rng.uniform(0, spread*2, size=len(close_prices)))
        low_prices = close_prices * (1 - self.rng.uniform(0, spread*2, size=len(close_prices)))

        # Ensure high >= close >= low
        high_prices = np.maximum(high_prices, close_prices)
        low_prices = np.minimum(low_prices, close_prices)

        return {
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices
        }
