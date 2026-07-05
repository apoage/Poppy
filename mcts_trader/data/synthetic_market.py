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
        seed: Optional[int] = None,
        # TASK_049: Synthetic noise for anti-memorization
        noise_level: float = 0.0,  # 0.0 = deterministic, 1.0 = full noise
        # TASK_058: Multi-modal noise for adaptive difficulty
        temporal_jitter: int = 0,      # Max steps to shift observations
        time_dilation: float = 0.0,    # 0.0-0.3, max stretch/compress factor
        phase_shift_prob: float = 0.0, # 0.0-0.5, probability of pattern phase shift
    ):
        """
        Args:
            level: Curriculum level (0-14)
            n_assets: Number of synthetic assets
            n_steps: Total steps in episode
            seed: Random seed for reproducibility
            noise_level: Noise intensity (0.0-1.0). Higher = harder to memorize.
                         Recommended warmup: start at 0.0, increase to 0.1+ by L5
            temporal_jitter: Max steps to jitter observations (forces timing robustness)
            time_dilation: Max factor to stretch/compress time (forces speed adaptation)
            phase_shift_prob: Probability of mid-episode phase shift (forces re-detection)
        """
        self.level = level
        self.n_assets = n_assets
        self.n_steps = n_steps
        self.current_step = 0
        self.noise_level = noise_level  # TASK_049
        # TASK_058: Multi-modal noise
        self.temporal_jitter = temporal_jitter
        self.time_dilation = time_dilation
        self.phase_shift_prob = phase_shift_prob

        # Bug Fix 1: Removed global seed. Use local RNG for isolation.
        self.rng = np.random.RandomState(seed)

        logging.debug(f"Generating Level {level} synthetic data ({n_assets} assets, {n_steps} steps)...")

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

        logging.debug(f"✅ Synthetic Level {level} data ready")

    def _generate_level_data(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Route to appropriate generator"""
        generators = {
            # Phase 1: Basic Mechanics (Levels 0-4)
            0: self._level_0_sine_waves,
            1: self._level_1_lagged_pairs,
            2: self._level_2_dip_crashes,
            3: self._level_3_noisy_trends,
            4: self._level_4_portfolio_rotation,
            # Phase 2: Statistical Arbitrage (Levels 5-7)
            5: self._level_5_pairs_trading,
            6: self._level_6_triangle_arbitrage,
            7: self._level_7_index_arbitrage,
            # Phase 3: Momentum & Reversal (Levels 8-10)
            8: self._level_8_momentum,
            9: self._level_9_mean_reversion,
            10: self._level_10_breakout,
            # Phase 4: Market Microstructure (Levels 11-13)
            11: self._level_11_bid_ask_bounce,
            12: self._level_12_order_imbalance,
            13: self._level_13_market_making,
            # Phase 5: Volatility (Level 14)
            14: self._level_14_volatility_clustering,
        }

        if self.level not in generators:
            raise ValueError(f"Unknown level: {self.level}. Available: 0-14")

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

    def regenerate(self, new_seed: Optional[int] = None):
        """
        Regenerate synthetic data with new randomness (anti-memorization).

        Args:
            new_seed: Optional new seed. If None, uses random seed.
        """
        if new_seed is not None:
            self.rng = np.random.RandomState(new_seed)
        else:
            # Use a random seed from current RNG
            self.rng = np.random.RandomState(self.rng.randint(0, 2**31))

        self.data = self._generate_level_data()
        self._validate_data()
        self.current_step = 0
        logging.debug(f"  Regenerated Level {self.level} synthetic data")

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

    # ========== TASK_049/058: Noise Application ==========

    def _apply_noise(self, prices: np.ndarray) -> np.ndarray:
        """Apply stochastic noise to deterministic prices.

        TASK_049: Prevents agent from memorizing exact patterns.

        Components:
        1. Random walk (cumulative drift)
        2. Per-step multiplicative noise
        3. Amplitude variation

        Args:
            prices: Base deterministic price series (n_steps,)

        Returns:
            Noisy price series (n_steps,)
        """
        if self.noise_level <= 0:
            return prices

        n = len(prices)

        # 1. Random walk component (trend drift)
        # Scaled by noise_level: at 1.0, gives ~5% drift over episode
        walk_std = 0.0005 * self.noise_level
        random_walk = np.cumsum(self.rng.normal(0, walk_std, n))

        # 2. Per-step noise (high-frequency)
        # Scaled by noise_level: at 1.0, gives ~0.5% per-step noise
        step_noise = self.rng.normal(0, 0.005 * self.noise_level, n)

        # 3. Amplitude variation (slow modulation)
        # Makes sine amplitude vary over episode
        amp_period = max(100, n // 10)
        amp_variation = 1 + 0.2 * self.noise_level * np.sin(
            2 * np.pi * np.arange(n) / amp_period + self.rng.uniform(0, 2 * np.pi)
        )

        # Combine: multiplicative noise
        noisy_prices = prices * (1 + random_walk + step_noise) * amp_variation

        # Ensure positive prices
        return np.maximum(noisy_prices, prices * 0.5)

    def _apply_temporal_jitter(self, prices: np.ndarray) -> np.ndarray:
        """TASK_058: Apply temporal jitter (observation timing shift).

        Shifts the entire price series by a random amount, simulating
        delayed or advanced observations. Forces model to be robust
        to timing variations.

        Args:
            prices: Price series (n_steps,)

        Returns:
            Jittered prices
        """
        if self.temporal_jitter <= 0:
            return prices

        jitter = self.rng.randint(-self.temporal_jitter, self.temporal_jitter + 1)
        return np.roll(prices, jitter)

    def _apply_time_dilation(self, prices: np.ndarray) -> np.ndarray:
        """TASK_058: Apply time dilation (stretch/compress patterns).

        Resamples the price series to a dilated time axis, making patterns
        appear faster or slower. Forces model to recognize patterns at
        different speeds.

        Args:
            prices: Price series (n_steps,)

        Returns:
            Dilated prices with same length
        """
        if self.time_dilation <= 0:
            return prices

        n = len(prices)

        # Random dilation factor for this series
        dilation = 1.0 + self.rng.uniform(-self.time_dilation, self.time_dilation)

        # Create dilated time axis
        original_time = np.arange(n)
        dilated_time = original_time * dilation

        # Interpolate to original time axis
        dilated_prices = np.interp(original_time, dilated_time, prices)

        return dilated_prices

    def _apply_phase_shift(self, prices: np.ndarray, pattern_period: int = 100) -> np.ndarray:
        """TASK_058: Apply random phase shifts mid-episode.

        With some probability, shifts the pattern phase at random points.
        Forces model to re-detect patterns rather than relying on fixed timing.

        Args:
            prices: Price series (n_steps,)
            pattern_period: Approximate pattern period for shift points

        Returns:
            Phase-shifted prices
        """
        if self.phase_shift_prob <= 0:
            return prices

        n = len(prices)
        result = prices.copy()

        # Determine potential shift points
        n_periods = n // pattern_period
        for i in range(n_periods):
            if self.rng.random() < self.phase_shift_prob:
                shift_point = i * pattern_period + self.rng.randint(0, pattern_period)
                shift_amount = self.rng.randint(pattern_period // 4, pattern_period // 2)

                # Shift remainder of series
                if shift_point + shift_amount < n:
                    result[shift_point:] = np.roll(result[shift_point:], shift_amount)

        return result

    def _apply_all_noise(self, prices: np.ndarray, pattern_period: int = 100) -> np.ndarray:
        """TASK_058: Apply all noise types in sequence.

        Order matters:
        1. Time dilation (changes pattern speed)
        2. Phase shift (changes pattern timing)
        3. Price noise (adds randomness)
        4. Temporal jitter (shifts observation)

        Args:
            prices: Base price series
            pattern_period: Period for phase shift calculation

        Returns:
            Fully noised prices
        """
        result = prices

        # Apply in order of increasing locality
        result = self._apply_time_dilation(result)
        result = self._apply_phase_shift(result, pattern_period)
        result = self._apply_noise(result)
        result = self._apply_temporal_jitter(result)

        return result

    # ========== Level Generators ==========

    def _level_0_sine_waves(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Level 0: Pure sine waves, zero noise.

        Episode variety: Random global phase offset ensures each episode
        starts at different point in the cycle while maintaining
        direction diversity across assets.
        """
        data = {}

        # TASK_007 FIX: Random global offset for episode variety
        # Each episode starts at a different phase, but asset relationships preserved
        global_phase_offset = self.rng.uniform(0, 2 * np.pi)

        for i in range(self.n_assets):
            # Deterministic parameters per asset if seed is set
            period = self.rng.randint(20, 60)  # Longer periods (was 10-50)
            amplitude = self.rng.uniform(0.08, 0.15)  # Moderate amplitude

            # Deterministic base phase (ensures direction diversity across assets)
            # + Random global offset (ensures episode variety)
            base_phase = (i * np.pi / 4) % (2 * np.pi)  # Spread phases evenly
            phase = base_phase + global_phase_offset
            baseline = 100.0

            t = np.arange(self.n_steps)
            close_prices = baseline * (1 + amplitude * np.sin(2*np.pi*t/period + phase))

            # TASK_049: Apply noise to prevent memorization
            close_prices = self._apply_noise(close_prices)

            # Deterministic OHLC spread (purely a function of the price)
            spread = close_prices * 0.001

            data[f'SYN_{i:02d}'] = {
                'open': close_prices,
                'high': close_prices + spread,
                'low': close_prices - spread,
                'close': close_prices
            }

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
            # TASK_049: Apply noise
            close = self._apply_noise(close)
            leaders[i] = close
            data[f'SYN_{i:02d}'] = self._prices_to_ohlc(close)

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
            # TASK_049: Apply noise
            follower_close = self._apply_noise(follower_close)

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
            # TASK_049: Apply additional noise
            prices = self._apply_noise(prices)

            data[f'SYN_{i:02d}'] = self._prices_to_ohlc(prices)

        return data

    def _level_3_noisy_trends(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Level 3: Random Walk with Drift (Trends).
        Bug Fix 3: clearer math.
        TASK_058: Now applies all noise types.
        TASK_134 FIX: Reduced volatility and added mean-reversion to prevent
                      unbounded price drift that causes 500+ prediction loss.
        """
        data = {}
        TREND_SWITCH_PROB = 0.05  # 5% chance to switch trend state
        TREND_STRENGTH = 0.003    # 0.3% drift per step (was 0.5% - too aggressive)
        NOISE_STD = 0.008         # 0.8% volatility (was 2% - way too high)
        MEAN_REVERSION = 0.002    # Pull back toward 100 to prevent unbounded drift
        BASE_PRICE = 100.0

        for i in range(self.n_assets):
            prices = np.zeros(self.n_steps)
            prices[0] = BASE_PRICE

            # State: 0=Flat, 1=Up, -1=Down
            trend_state = 0

            for t in range(1, self.n_steps):
                # Markov chain for trend state
                if self.rng.rand() < TREND_SWITCH_PROB:
                    # Switch to a random new state
                    trend_state = self.rng.choice([-1, 0, 1])

                # Drift component
                drift = trend_state * TREND_STRENGTH

                # TASK_134: Mean reversion to prevent unbounded drift
                # Pulls price back toward BASE_PRICE logarithmically
                log_deviation = np.log(prices[t-1] / BASE_PRICE)
                mean_reversion = -MEAN_REVERSION * log_deviation

                # Noise component
                noise = self.rng.normal(0, NOISE_STD)

                # Multiplicative random walk with mean reversion
                change = 1 + drift + mean_reversion + noise
                prices[t] = prices[t-1] * change

                # Clamp to reasonable range (50-200) to prevent extreme values
                prices[t] = np.clip(prices[t], 50.0, 200.0)

            # TASK_058: Apply all noise types
            prices = self._apply_all_noise(prices, pattern_period=int(1.0 / TREND_SWITCH_PROB))
            data[f'SYN_{i:02d}'] = self._prices_to_ohlc(prices)

        return data

    def _level_4_portfolio_rotation(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Level 4: Rotating sector leadership.

        Tests:
        - Sector/cluster awareness (8 assets per sector)
        - Timing of rotation cycles (every 50 steps)
        - Multi-asset coordination

        Checkpoint compatible: Same tensor shapes as other levels.
        """
        data = {}
        SECTOR_SIZE = 8
        N_SECTORS = self.n_assets // SECTOR_SIZE  # 64/8 = 8 sectors
        ROTATION_PERIOD = 50
        PUMP_MAGNITUDE = 0.002  # 0.2% per step (cumulative over period)

        # Generate base GBM prices for all assets
        base_prices = {}
        for i in range(self.n_assets):
            # TASK_004 FIX: Proper GBM using log-returns
            # P_t = P_0 * exp(cumsum(log_returns))
            # This ensures P_0 = 100.0 exactly (unlike cumprod which starts at 100*(1+r[0]))
            returns = self.rng.normal(0, 0.01, size=self.n_steps)
            log_returns = np.cumsum(returns)
            prices = 100.0 * np.exp(log_returns)
            base_prices[i] = prices.copy()

        # Apply sector rotation pumps
        if N_SECTORS > 0:
            for t in range(self.n_steps):
                leading_sector = (t // ROTATION_PERIOD) % N_SECTORS

                start_idx = leading_sector * SECTOR_SIZE
                end_idx = min(start_idx + SECTOR_SIZE, self.n_assets)

                for asset_idx in range(start_idx, end_idx):
                    base_prices[asset_idx][t] *= (1 + PUMP_MAGNITUDE)

        # Convert to OHLC format (same as other levels)
        for i in range(self.n_assets):
            # TASK_058: Apply all noise types
            prices = self._apply_all_noise(base_prices[i], pattern_period=ROTATION_PERIOD)
            data[f'SYN_{i:02d}'] = self._prices_to_ohlc(prices)

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

    # ========== Advanced Level Generators (Levels 5-14) ==========

    def _generate_gbm(self, drift: float = 0.0, volatility: float = 0.01) -> np.ndarray:
        """Helper: Generate Geometric Brownian Motion"""
        returns = self.rng.normal(drift, volatility, self.n_steps)
        return 100.0 * np.exp(np.cumsum(returns))

    def _level_5_pairs_trading(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Level 5: Cointegrated pairs for statistical arbitrage.

        Structure:
        - 32 pairs (64 assets total)
        - Each pair: Asset_i cointegrated with Asset_{i+32}
        - Spread follows Ornstein-Uhlenbeck process

        Optimal Strategy:
        - Detect spread divergence (> 2σ)
        - Short expensive asset, long cheap asset
        - Exit when spread returns to mean
        """
        data = {}
        N_PAIRS = self.n_assets // 2

        # OU parameters for spread
        THETA = 0.15        # Mean reversion speed
        MU = 0.0            # Long-run mean
        SIGMA_SPREAD = 0.05 # Spread volatility

        for pair_idx in range(N_PAIRS):
            # Common factor for this pair
            common_factor = self._generate_gbm(drift=0.0001, volatility=0.01)

            # Generate OU process for spread
            spread = np.zeros(self.n_steps)
            spread[0] = self.rng.normal(0, SIGMA_SPREAD)

            for t in range(1, self.n_steps):
                dW = self.rng.normal(0, 1)
                dS = -THETA * (spread[t-1] - MU) + SIGMA_SPREAD * dW
                spread[t] = spread[t-1] + dS

            # Create cointegrated pair
            beta = 1.0 + self.rng.uniform(-0.1, 0.1)

            price_a = common_factor * (1 + spread / 2)
            price_b = common_factor * (1 - spread / (2 * beta))

            # Add small idiosyncratic noise
            price_a *= (1 + self.rng.normal(0, 0.002, self.n_steps))
            price_b *= (1 + self.rng.normal(0, 0.002, self.n_steps))

            # Ensure positive prices
            price_a = np.maximum(price_a, 0.01)
            price_b = np.maximum(price_b, 0.01)

            # TASK_058: Apply all noise types (OU reversion period ~1/theta)
            ou_period = int(1.0 / THETA) if THETA > 0 else 100
            price_a = self._apply_all_noise(price_a, pattern_period=ou_period)
            price_b = self._apply_all_noise(price_b, pattern_period=ou_period)

            data[f'SYN_{pair_idx:02d}'] = self._prices_to_ohlc(price_a)
            data[f'SYN_{pair_idx + N_PAIRS:02d}'] = self._prices_to_ohlc(price_b)

        return data

    def _level_6_triangle_arbitrage(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Level 6: Triangle arbitrage opportunities.

        Structure:
        - Assets organized as cross-pairs
        - Periodic mispricing events (5% of steps)
        - Mispricing magnitude: 1-3%

        Optimal Strategy:
        - Detect mispricing in triangular relationship
        - Execute 3-leg arbitrage
        """
        data = {}

        # Generate base currencies (16 assets)
        base_prices = {}
        for i in range(16):
            base_prices[i] = self._generate_gbm(drift=0.0001, volatility=0.015)
            # TASK_058: Apply noise BEFORE converting to OHLC
            noisy_prices = self._apply_all_noise(base_prices[i], pattern_period=20)
            data[f'SYN_{i:02d}'] = self._prices_to_ohlc(noisy_prices)

        # Quote pairs with small noise
        for i in range(16):
            noise = 1 + self.rng.normal(0, 0.002, self.n_steps)
            quote_prices = base_prices[i] * noise
            # TASK_058: Apply noise BEFORE converting to OHLC
            noisy_prices = self._apply_all_noise(quote_prices, pattern_period=20)
            data[f'SYN_{i+16:02d}'] = self._prices_to_ohlc(noisy_prices)

        # Cross pairs with mispricing events
        for i in range(16):
            j = (i + 1) % 16
            fair_rate = base_prices[i] / np.maximum(base_prices[j], 0.01)

            # Add mispricing events
            mispricing = np.ones(self.n_steps)
            for t in range(self.n_steps):
                if self.rng.rand() < 0.05:  # 5% chance
                    direction = self.rng.choice([-1, 1])
                    magnitude = self.rng.uniform(0.01, 0.03)
                    mispricing[t] = 1 + direction * magnitude

            actual_rate = fair_rate * mispricing
            actual_rate = np.maximum(actual_rate, 0.01)
            # TASK_058: Apply noise BEFORE converting to OHLC
            noisy_prices = self._apply_all_noise(actual_rate, pattern_period=20)
            data[f'SYN_{i+32:02d}'] = self._prices_to_ohlc(noisy_prices)

        # Fill remaining assets with neutral GBM
        for i in range(48, self.n_assets):
            prices = self._generate_gbm(drift=0.0, volatility=0.01)
            # TASK_058: Apply all noise types
            prices = self._apply_all_noise(prices, pattern_period=20)
            data[f'SYN_{i:02d}'] = self._prices_to_ohlc(prices)

        return data

    def _level_7_index_arbitrage(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Level 7: Index vs component arbitrage.

        Structure:
        - 56 individual assets (components)
        - 8 "index" assets (each tracking ~7 components)
        - Index lags components by 3-5 steps

        Optimal Strategy:
        - Compute fair index value from components
        - Trade index vs components when mispriced
        """
        data = {}

        N_COMPONENTS = 56
        N_INDICES = 8
        COMPONENTS_PER_INDEX = N_COMPONENTS // N_INDICES
        INDEX_LAG = 4

        # Generate component prices
        component_prices = {}
        for i in range(N_COMPONENTS):
            component_prices[i] = self._generate_gbm(drift=0.0002, volatility=0.012)
            # TASK_058: Apply noise BEFORE converting to OHLC
            noisy_prices = self._apply_all_noise(component_prices[i], pattern_period=INDEX_LAG * 5)
            data[f'SYN_{i:02d}'] = self._prices_to_ohlc(noisy_prices)

        # Generate index prices (lagged basket)
        for idx in range(N_INDICES):
            start_comp = idx * COMPONENTS_PER_INDEX
            end_comp = start_comp + COMPONENTS_PER_INDEX

            # Equal-weighted basket
            basket = np.zeros(self.n_steps)
            for comp_idx in range(start_comp, end_comp):
                basket += component_prices[comp_idx] / COMPONENTS_PER_INDEX

            # Lagged index
            index_price = np.roll(basket, INDEX_LAG)
            index_price[:INDEX_LAG] = basket[:INDEX_LAG]

            # Add premium/discount events
            for t in range(self.n_steps):
                if self.rng.rand() < 0.03:
                    premium = self.rng.uniform(-0.02, 0.02)
                    index_price[t] *= (1 + premium)

            index_price = np.maximum(index_price, 0.01)
            # TASK_058: Apply noise to index
            index_price = self._apply_all_noise(index_price, pattern_period=INDEX_LAG * 5)
            data[f'SYN_{N_COMPONENTS + idx:02d}'] = self._prices_to_ohlc(index_price)

        return data

    def _level_8_momentum(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Level 8: Momentum with trend persistence.

        Structure:
        - Assets follow momentum regime
        - 70% chance trend continues
        - Regime switches every ~20 steps

        Optimal Strategy:
        - Detect trend direction from recent returns
        - Ride winners, cut losers
        """
        data = {}

        LOOKBACK = 20
        PERSISTENCE_PROB = 0.70
        TREND_DRIFT = 0.003
        VOLATILITY = 0.01

        for i in range(self.n_assets):
            prices = np.zeros(self.n_steps)
            prices[0] = 100.0
            trend = self.rng.choice([-1, 0, 1])

            for t in range(1, self.n_steps):
                if t % LOOKBACK == 0:
                    if self.rng.rand() > PERSISTENCE_PROB:
                        trend = self.rng.choice([-1, 0, 1])

                drift = trend * TREND_DRIFT
                noise = self.rng.normal(0, VOLATILITY)
                prices[t] = prices[t-1] * (1 + drift + noise)
                prices[t] = max(prices[t], 0.01)

            # TASK_058: Apply all noise types
            prices = self._apply_all_noise(prices, pattern_period=LOOKBACK)
            data[f'SYN_{i:02d}'] = self._prices_to_ohlc(prices)

        return data

    def _level_9_mean_reversion(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Level 9: Strong mean-reverting prices (OU process).

        Structure:
        - Each asset reverts to VWAP = 100
        - Fast reversion (θ = 0.2)

        Optimal Strategy:
        - Fade extremes (buy low, sell high)
        - Size based on distance from mean
        """
        data = {}

        THETA = 0.20
        MU = 100.0
        SIGMA = 3.0

        for i in range(self.n_assets):
            prices = np.zeros(self.n_steps)
            prices[0] = MU + self.rng.normal(0, SIGMA)

            for t in range(1, self.n_steps):
                dW = self.rng.normal(0, 1)
                dP = -THETA * (prices[t-1] - MU) + SIGMA * 0.1 * dW
                prices[t] = prices[t-1] + dP
                prices[t] = max(prices[t], 0.01)

            # TASK_058: Apply all noise types (OU process mean reversion period ~ 1/θ = 5)
            prices = self._apply_all_noise(prices, pattern_period=5)
            data[f'SYN_{i:02d}'] = self._prices_to_ohlc(prices)

        return data

    def _level_10_breakout(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Level 10: Consolidation → Breakout patterns.

        Structure:
        - Consolidation: 30 steps in tight range
        - Breakout: 20 steps trending ±5%

        Optimal Strategy:
        - Detect consolidation (low volatility)
        - Trade breakout direction
        """
        data = {}

        CONSOLIDATION_PERIOD = 30
        BREAKOUT_PERIOD = 20
        RANGE_SIZE = 0.02
        BREAKOUT_DRIFT = 0.0025

        for i in range(self.n_assets):
            prices = np.zeros(self.n_steps)
            prices[0] = 100.0

            t = 1
            phase = 'consolidation'
            phase_start = 0
            base_price = 100.0
            breakout_direction = 0

            while t < self.n_steps:
                if phase == 'consolidation':
                    noise = self.rng.uniform(-RANGE_SIZE, RANGE_SIZE)
                    prices[t] = base_price * (1 + noise)

                    if t - phase_start >= CONSOLIDATION_PERIOD:
                        phase = 'breakout'
                        phase_start = t
                        breakout_direction = self.rng.choice([-1, 1])

                elif phase == 'breakout':
                    drift = breakout_direction * BREAKOUT_DRIFT
                    noise = self.rng.normal(0, 0.005)
                    prices[t] = prices[t-1] * (1 + drift + noise)

                    if t - phase_start >= BREAKOUT_PERIOD:
                        phase = 'consolidation'
                        phase_start = t
                        base_price = prices[t]

                prices[t] = max(prices[t], 0.01)
                t += 1

            # TASK_058: Apply all noise types
            prices = self._apply_all_noise(prices, pattern_period=CONSOLIDATION_PERIOD + BREAKOUT_PERIOD)
            data[f'SYN_{i:02d}'] = self._prices_to_ohlc(prices)

        return data

    def _level_11_bid_ask_bounce(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Level 11: Bid-ask bounce microstructure.

        Structure:
        - Mid price follows slow random walk
        - Reported price bounces between bid and ask
        - Spread = 0.1% (10 bps)

        Optimal Strategy:
        - Provide liquidity (buy at bid, sell at ask)
        """
        data = {}

        SPREAD_BPS = 10
        SPREAD = SPREAD_BPS / 10000

        for i in range(self.n_assets):
            mid_returns = self.rng.normal(0, 0.001, self.n_steps)
            mid_price = 100.0 * np.exp(np.cumsum(mid_returns))

            order_flow = self.rng.choice([-1, 1], self.n_steps)

            prices = np.where(
                order_flow == 1,
                mid_price * (1 + SPREAD/2),
                mid_price * (1 - SPREAD/2)
            )

            prices = np.maximum(prices, 0.01)
            # TASK_058: Apply all noise types (microstructure - short period)
            prices = self._apply_all_noise(prices, pattern_period=10)
            data[f'SYN_{i:02d}'] = self._prices_to_ohlc(prices)

        return data

    def _level_12_order_imbalance(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Level 12: Order book imbalance signals.

        Structure:
        - Imbalance signal embedded in volume data
        - 80% predictive accuracy when |imbalance| > 0.3

        Note: Volume encodes imbalance
        - Volume > 1200: Positive imbalance (price up)
        - Volume < 800: Negative imbalance (price down)
        """
        data = {}

        SIGNAL_THRESHOLD = 0.3
        ACCURACY = 0.80
        MOVE_SIZE = 0.005

        for i in range(self.n_assets):
            prices = np.zeros(self.n_steps)
            volumes = np.zeros(self.n_steps)
            prices[0] = 100.0
            volumes[0] = 1000.0

            for t in range(1, self.n_steps):
                imbalance = self.rng.uniform(-0.5, 0.5)
                volumes[t] = 1000 + imbalance * 400

                if abs(imbalance) > SIGNAL_THRESHOLD:
                    if self.rng.rand() < ACCURACY:
                        direction = np.sign(imbalance)
                    else:
                        direction = -np.sign(imbalance)
                    drift = direction * MOVE_SIZE
                else:
                    drift = 0

                noise = self.rng.normal(0, 0.002)
                prices[t] = prices[t-1] * (1 + drift + noise)
                prices[t] = max(prices[t], 0.01)

            # TASK_058: Apply all noise types
            prices = self._apply_all_noise(prices, pattern_period=20)
            ohlc = self._prices_to_ohlc(prices)
            ohlc['volume'] = volumes
            data[f'SYN_{i:02d}'] = ohlc

        return data

    def _level_13_market_making(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Level 13: Market making with adverse selection.

        Structure:
        - 95% of time: Small random noise (±0.1%)
        - 5% of time: Informed order (large move ±1%)

        Optimal Strategy:
        - Provide liquidity most of the time
        - Detect and avoid informed flow
        """
        data = {}

        NOISE_SIZE = 0.001
        INFORMED_PROB = 0.05
        INFORMED_SIZE = 0.01

        for i in range(self.n_assets):
            prices = np.zeros(self.n_steps)
            prices[0] = 100.0

            for t in range(1, self.n_steps):
                if self.rng.rand() < INFORMED_PROB:
                    direction = self.rng.choice([-1, 1])
                    move = direction * INFORMED_SIZE
                else:
                    move = self.rng.uniform(-NOISE_SIZE, NOISE_SIZE)

                prices[t] = prices[t-1] * (1 + move)
                prices[t] = max(prices[t], 0.01)

            # TASK_058: Apply all noise types (informed orders rare - longer period)
            prices = self._apply_all_noise(prices, pattern_period=int(1/INFORMED_PROB))
            data[f'SYN_{i:02d}'] = self._prices_to_ohlc(prices)

        return data

    def _level_14_volatility_clustering(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Level 14: GARCH(1,1) volatility dynamics.

        Structure:
        - Volatility is autocorrelated
        - High vol follows high vol

        Optimal Strategy:
        - Detect volatility regime
        - Reduce position size in high vol
        """
        data = {}

        # GARCH(1,1) parameters: σ²[t] = ω + α*ε²[t-1] + β*σ²[t-1]
        # ω (OMEGA) must be tiny - it's the base variance level
        # α + β < 1 for stationarity
        OMEGA = 0.000001  # Base variance (was 0.05 - way too high!)
        ALPHA = 0.10      # ARCH coefficient (reaction to shocks)
        BETA = 0.85       # GARCH coefficient (persistence)

        for i in range(self.n_assets):
            prices = np.zeros(self.n_steps)
            variances = np.zeros(self.n_steps)

            prices[0] = 100.0
            variances[0] = 0.0001

            for t in range(1, self.n_steps):
                prev_return = np.log(prices[t-1] / prices[max(0, t-2)]) if t > 1 else 0
                # Proper GARCH(1,1): σ²[t] = ω + α*ε²[t-1] + β*σ²[t-1]
                variances[t] = OMEGA + ALPHA * prev_return**2 + BETA * variances[t-1]
                variances[t] = np.clip(variances[t], 0.000001, 0.001)  # Cap variance reasonably

                vol = np.sqrt(variances[t])
                epsilon = self.rng.normal(0, 1)
                ret = vol * epsilon

                prices[t] = prices[t-1] * np.exp(ret)
                prices[t] = max(prices[t], 0.01)

            # TASK_058: Apply all noise types (vol clustering - medium period)
            prices = self._apply_all_noise(prices, pattern_period=50)
            data[f'SYN_{i:02d}'] = self._prices_to_ohlc(prices)

        return data