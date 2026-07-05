# Advanced Synthetic Levels Design (Levels 5-14)

**Purpose**: Extend the curriculum beyond basic pattern recognition to teach sophisticated trading tactics.

**Prerequisites**: Model has passed Levels 0-4 (basic mechanics + portfolio rotation)

---

## Overview: Level Progression

| Level | Name | Tactic | Key Skill |
|-------|------|--------|-----------|
| 5 | Pairs Trading | Statistical Arbitrage | Cointegration detection |
| 6 | Triangle Arbitrage | Cross-pair | Mispricing exploitation |
| 7 | Index Arbitrage | Frontrunning | Index vs components |
| 8 | Momentum | Trend following | Ride winners |
| 9 | Mean Reversion | Fade extremes | OU process |
| 10 | Breakout | Range → Trend | Consolidation detection |
| 11 | Bid-Ask Bounce | Spread capture | Microstructure |
| 12 | Order Imbalance | Predict direction | Volume analysis |
| 13 | Market Making | Liquidity provision | Inventory management |
| 14 | Volatility Clustering | GARCH dynamics | Risk sizing |

---

## Level 5: Pairs Trading (Cointegration)

### Mathematical Foundation

Two assets A and B are cointegrated if their spread is mean-reverting:
```
Spread(t) = Price_A(t) - β × Price_B(t)
Spread follows OU process: dS = -θ(S - μ)dt + σdW
```

Trade when spread deviates > 2σ from mean:
- Spread > μ + 2σ: Short A, Long B (expect convergence)
- Spread < μ - 2σ: Long A, Short B

### Generator Implementation

```python
def _level_5_pairs_trading(self) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Level 5: Cointegrated pairs for statistical arbitrage.

    Structure:
    - 32 pairs (64 assets total)
    - Each pair: Asset_i cointegrated with Asset_{i+32}
    - Spread follows Ornstein-Uhlenbeck process
    - Clear entry/exit signals when spread > 2σ

    Optimal Strategy:
    - Detect spread divergence
    - Short expensive asset, long cheap asset
    - Exit when spread returns to mean
    """
    data = {}
    N_PAIRS = self.n_assets // 2

    # OU parameters for spread
    THETA = 0.15        # Mean reversion speed
    MU = 0.0            # Long-run mean
    SIGMA_SPREAD = 0.05 # Spread volatility

    # Underlying common factor
    common_factor = self._generate_gbm(drift=0.0001, volatility=0.01)

    for pair_idx in range(N_PAIRS):
        # Generate OU process for spread
        spread = np.zeros(self.n_steps)
        spread[0] = self.rng.normal(0, SIGMA_SPREAD)

        for t in range(1, self.n_steps):
            dW = self.rng.normal(0, 1) * np.sqrt(1)  # dt=1
            dS = -THETA * (spread[t-1] - MU) + SIGMA_SPREAD * dW
            spread[t] = spread[t-1] + dS

        # Create cointegrated pair
        # Asset A = common_factor + spread/2
        # Asset B = common_factor - spread/2
        # So: Asset_A - Asset_B = spread (cointegrated)

        beta = 1.0 + self.rng.uniform(-0.1, 0.1)  # Slight hedge ratio variation

        price_a = common_factor * (1 + spread / 2)
        price_b = common_factor * (1 - spread / (2 * beta))

        # Add small idiosyncratic noise
        price_a *= (1 + self.rng.normal(0, 0.002, self.n_steps))
        price_b *= (1 + self.rng.normal(0, 0.002, self.n_steps))

        data[f'SYN_{pair_idx:02d}'] = self._prices_to_ohlc(price_a)
        data[f'SYN_{pair_idx + N_PAIRS:02d}'] = self._prices_to_ohlc(price_b)

    return data

def _generate_gbm(self, drift: float = 0.0, volatility: float = 0.01) -> np.ndarray:
    """Helper: Generate GBM process"""
    returns = self.rng.normal(drift, volatility, self.n_steps)
    return 100.0 * np.exp(np.cumsum(returns))
```

### Pass Criteria
- Average reward: > +2.0%
- Trade frequency: 10-30% (selective trading)
- Pair correlation awareness: Model should trade correlated pairs together

### Key Insight
This level tests whether the model can:
1. Detect cointegration relationships
2. Wait for spread divergence (patience)
3. Execute paired trades (multi-asset coordination)

---

## Level 6: Triangle Arbitrage

### Mathematical Foundation

Three assets with cross-pair relationships:
```
BTC/USDT = 50000
ETH/USDT = 2500
BTC/ETH = actual_rate

Fair BTC/ETH = BTC/USDT / ETH/USDT = 20.0

If actual BTC/ETH = 19.5 (BTC cheap vs ETH):
  1. Buy BTC with USDT
  2. Sell BTC for ETH
  3. Sell ETH for USDT
  Profit: ~2.5%
```

### Generator Implementation

```python
def _level_6_triangle_arbitrage(self) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Level 6: Triangle arbitrage opportunities.

    Structure:
    - 64 assets organized as cross-pairs
    - Periodic mispricing events (5% of steps)
    - Mispricing magnitude: 1-3%

    Asset Groups (example with 64 assets):
    - Assets 0-15: Base currencies (like BTC, ETH, etc.)
    - Assets 16-31: Quote 1 pairs
    - Assets 32-47: Quote 2 pairs
    - Assets 48-63: Cross pairs (with occasional mispricing)

    Optimal Strategy:
    - Detect mispricing in triangular relationship
    - Execute 3-leg arbitrage atomically
    """
    data = {}

    # Generate base currencies
    base_prices = {}
    for i in range(16):
        base_prices[i] = self._generate_gbm(drift=0.0001, volatility=0.015)

    # Quote 1 (vs USDT-like stable)
    for i in range(16):
        data[f'SYN_{i:02d}'] = self._prices_to_ohlc(base_prices[i])

    # Quote 2 (vs another stable)
    for i in range(16):
        # Add small noise vs base
        noise = 1 + self.rng.normal(0, 0.002, self.n_steps)
        data[f'SYN_{i+16:02d}'] = self._prices_to_ohlc(base_prices[i] * noise)

    # Cross pairs (with mispricing events)
    for i in range(16):
        j = (i + 1) % 16  # Cross with next asset

        # Fair cross rate
        fair_rate = base_prices[i] / base_prices[j]

        # Add mispricing events
        mispricing = np.ones(self.n_steps)
        for t in range(self.n_steps):
            if self.rng.rand() < 0.05:  # 5% chance
                # Mispricing: 1-3%
                direction = self.rng.choice([-1, 1])
                magnitude = self.rng.uniform(0.01, 0.03)
                mispricing[t] = 1 + direction * magnitude

        actual_rate = fair_rate * mispricing
        data[f'SYN_{i+32:02d}'] = self._prices_to_ohlc(actual_rate)

    # Fill remaining assets with neutral GBM
    for i in range(48, self.n_assets):
        data[f'SYN_{i:02d}'] = self._prices_to_ohlc(
            self._generate_gbm(drift=0.0, volatility=0.01)
        )

    return data
```

### Pass Criteria
- Average reward: > +1.5%
- Must detect at least 50% of mispricing events
- Sharpe ratio: > 2.0 (high-reward, low-variance opportunities)

---

## Level 7: Index Arbitrage

### Mathematical Foundation

Index tracks basket of components:
```
Index_fair = Σ(weight_i × price_i)
Index_actual = observed price

If Index_actual > Index_fair + threshold:
  - Short index, long components

If Index_actual < Index_fair - threshold:
  - Long index, short components
```

### Generator Implementation

```python
def _level_7_index_arbitrage(self) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Level 7: Index vs component arbitrage.

    Structure:
    - 56 individual assets (components)
    - 8 "index" assets (each tracking ~7 components)
    - Index lags components by 3-5 steps
    - Occasional index premium/discount

    Optimal Strategy:
    - Compute fair index value from components
    - Trade index vs components when mispriced
    - Frontrun index rebalancing
    """
    data = {}

    N_COMPONENTS = 56
    N_INDICES = 8
    COMPONENTS_PER_INDEX = N_COMPONENTS // N_INDICES  # 7
    INDEX_LAG = 4  # Steps index lags components

    # Generate component prices
    component_prices = {}
    for i in range(N_COMPONENTS):
        component_prices[i] = self._generate_gbm(drift=0.0002, volatility=0.012)
        data[f'SYN_{i:02d}'] = self._prices_to_ohlc(component_prices[i])

    # Generate index prices (lagged basket + noise)
    for idx in range(N_INDICES):
        # Components in this index
        start_comp = idx * COMPONENTS_PER_INDEX
        end_comp = start_comp + COMPONENTS_PER_INDEX

        # Equal-weighted basket
        basket = np.zeros(self.n_steps)
        for comp_idx in range(start_comp, end_comp):
            basket += component_prices[comp_idx] / COMPONENTS_PER_INDEX

        # Lagged index (index reacts slowly)
        index_price = np.roll(basket, INDEX_LAG)
        index_price[:INDEX_LAG] = basket[:INDEX_LAG]

        # Add premium/discount events
        for t in range(self.n_steps):
            if self.rng.rand() < 0.03:  # 3% chance
                premium = self.rng.uniform(-0.02, 0.02)  # ±2%
                index_price[t] *= (1 + premium)

        data[f'SYN_{N_COMPONENTS + idx:02d}'] = self._prices_to_ohlc(index_price)

    return data
```

### Pass Criteria
- Average reward: > +1.0%
- Must exploit index lag consistently
- Trade frequency correlates with mispricing events

---

## Level 8: Time-Series Momentum

### Mathematical Foundation

Momentum factor:
```
Signal(t) = sign(Return_{t-20:t})  # Past 20-step return
Position = Signal × risk_budget

Persistence: 70% chance trend continues for next 20 steps
```

### Generator Implementation

```python
def _level_8_momentum(self) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Level 8: Momentum with trend persistence.

    Structure:
    - Assets follow momentum regime
    - If up last 20 steps: 70% chance continue up
    - If down last 20 steps: 70% chance continue down
    - Regime switches with 30% probability

    Optimal Strategy:
    - Detect trend direction from recent returns
    - Ride winners, cut losers
    - Size position based on trend strength
    """
    data = {}

    LOOKBACK = 20
    PERSISTENCE_PROB = 0.70
    TREND_DRIFT = 0.003  # 0.3% per step when trending
    VOLATILITY = 0.01

    for i in range(self.n_assets):
        prices = np.zeros(self.n_steps)
        prices[0] = 100.0

        # Initialize trend state randomly
        trend = self.rng.choice([-1, 0, 1])

        for t in range(1, self.n_steps):
            # Every LOOKBACK steps, potentially switch regime
            if t % LOOKBACK == 0:
                if self.rng.rand() > PERSISTENCE_PROB:
                    # Regime switch
                    trend = self.rng.choice([-1, 0, 1])

            # Generate return with trend drift
            drift = trend * TREND_DRIFT
            noise = self.rng.normal(0, VOLATILITY)

            prices[t] = prices[t-1] * (1 + drift + noise)
            prices[t] = max(prices[t], 0.01)  # Floor

        data[f'SYN_{i:02d}'] = self._prices_to_ohlc(prices)

    return data
```

### Pass Criteria
- Average reward: > +1.5%
- Win rate on trend-following trades: > 55%
- Must demonstrate trend detection (not random trading)

---

## Level 9: Mean Reversion

### Mathematical Foundation

Ornstein-Uhlenbeck process around VWAP:
```
dP = -θ(P - μ)dt + σdW

Where:
- θ = mean reversion speed (0.1-0.3)
- μ = long-run mean (VWAP = 100)
- σ = volatility

Trade when: |P - μ| > 2σ
Direction: Toward μ
```

### Generator Implementation

```python
def _level_9_mean_reversion(self) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Level 9: Strong mean-reverting prices.

    Structure:
    - Each asset reverts to VWAP = 100
    - Fast reversion (θ = 0.2)
    - Clear entry signals at ±2σ

    Optimal Strategy:
    - Fade extremes (buy low, sell high)
    - Size based on distance from mean
    - Quick exits as price reverts
    """
    data = {}

    THETA = 0.20      # Mean reversion speed
    MU = 100.0        # Long-run mean
    SIGMA = 3.0       # Volatility (std dev ~3)

    for i in range(self.n_assets):
        prices = np.zeros(self.n_steps)
        # Start at random deviation from mean
        prices[0] = MU + self.rng.normal(0, SIGMA)

        for t in range(1, self.n_steps):
            dW = self.rng.normal(0, 1)
            dP = -THETA * (prices[t-1] - MU) + SIGMA * 0.1 * dW
            prices[t] = prices[t-1] + dP
            prices[t] = max(prices[t], 0.01)  # Floor

        data[f'SYN_{i:02d}'] = self._prices_to_ohlc(prices)

    return data
```

### Pass Criteria
- Average reward: > +2.0%
- Must demonstrate contrarian behavior (buy dips, sell rallies)
- Trade frequency: 15-25%

---

## Level 10: Breakout Trading

### Mathematical Foundation

Price alternates between:
```
Consolidation: Range [μ - r, μ + r] for 30 steps
Breakout: Strong trend (±5%) for 20 steps

Breakout signal: Price crosses range boundary
Direction: Continue in breakout direction
```

### Generator Implementation

```python
def _level_10_breakout(self) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Level 10: Consolidation → Breakout patterns.

    Structure:
    - Consolidation phase: 30 steps in tight range
    - Breakout phase: 20 steps trending ±5%
    - Clear transition signals

    Optimal Strategy:
    - Detect consolidation (low volatility)
    - Wait for breakout confirmation
    - Ride trend, exit on reversal signs
    """
    data = {}

    CONSOLIDATION_PERIOD = 30
    BREAKOUT_PERIOD = 20
    RANGE_SIZE = 0.02      # ±2% range during consolidation
    BREAKOUT_DRIFT = 0.0025  # 0.25% per step during breakout

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
                # Tight range around base
                noise = self.rng.uniform(-RANGE_SIZE, RANGE_SIZE)
                prices[t] = base_price * (1 + noise)

                # Check if consolidation period ended
                if t - phase_start >= CONSOLIDATION_PERIOD:
                    phase = 'breakout'
                    phase_start = t
                    breakout_direction = self.rng.choice([-1, 1])

            elif phase == 'breakout':
                # Strong trend
                drift = breakout_direction * BREAKOUT_DRIFT
                noise = self.rng.normal(0, 0.005)
                prices[t] = prices[t-1] * (1 + drift + noise)

                # Check if breakout period ended
                if t - phase_start >= BREAKOUT_PERIOD:
                    phase = 'consolidation'
                    phase_start = t
                    base_price = prices[t]

            t += 1

        data[f'SYN_{i:02d}'] = self._prices_to_ohlc(prices)

    return data
```

### Pass Criteria
- Average reward: > +1.5%
- Must detect breakout within 3 steps of range break
- Win rate on breakout trades: > 60%

---

## Level 11: Bid-Ask Bounce

### Mathematical Foundation

Microstructure noise from bid-ask spread:
```
Bid = Mid - spread/2
Ask = Mid + spread/2

Last price alternates based on order flow:
- Buy order → Last = Ask
- Sell order → Last = Bid
```

### Generator Implementation

```python
def _level_11_bid_ask_bounce(self) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Level 11: Bid-ask bounce microstructure.

    Structure:
    - Mid price follows slow random walk
    - Reported price bounces between bid and ask
    - Spread = 0.1% (10 bps)

    Optimal Strategy:
    - Identify mean-reversion from bid-ask noise
    - Provide liquidity (buy at bid, sell at ask)
    - Small but consistent profits
    """
    data = {}

    SPREAD_BPS = 10  # 10 basis points = 0.1%
    SPREAD = SPREAD_BPS / 10000

    for i in range(self.n_assets):
        # Slow mid-price random walk
        mid_returns = self.rng.normal(0, 0.001, self.n_steps)
        mid_price = 100.0 * np.exp(np.cumsum(mid_returns))

        # Bid-ask bounce (random order flow)
        order_flow = self.rng.choice([-1, 1], self.n_steps)  # -1=sell, 1=buy

        prices = np.where(
            order_flow == 1,
            mid_price * (1 + SPREAD/2),  # Buy order → Ask price
            mid_price * (1 - SPREAD/2)   # Sell order → Bid price
        )

        data[f'SYN_{i:02d}'] = self._prices_to_ohlc(prices)

    return data
```

### Pass Criteria
- Average reward: > +0.5% (small but consistent)
- Sharpe ratio: > 3.0 (very low variance)
- Trade frequency: 30-50% (active market making)

---

## Level 12: Order Book Imbalance

### Mathematical Foundation

Order imbalance predicts short-term direction:
```
Imbalance = (Bid_volume - Ask_volume) / (Bid_volume + Ask_volume)

If Imbalance > 0.3 → Price likely to rise (80% accuracy)
If Imbalance < -0.3 → Price likely to fall (80% accuracy)
```

### Generator Implementation

```python
def _level_12_order_imbalance(self) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Level 12: Order book imbalance signals.

    Structure:
    - Imbalance signal embedded in volume data
    - 80% predictive accuracy when |imbalance| > 0.3
    - Trade in direction of imbalance

    Note: Volume field encodes imbalance signal
    - Volume > 1200: Positive imbalance (price up)
    - Volume < 800: Negative imbalance (price down)
    - Volume 800-1200: Neutral
    """
    data = {}

    SIGNAL_THRESHOLD = 0.3
    ACCURACY = 0.80
    MOVE_SIZE = 0.005  # 0.5% move when signal fires

    for i in range(self.n_assets):
        prices = np.zeros(self.n_steps)
        volumes = np.zeros(self.n_steps)
        prices[0] = 100.0
        volumes[0] = 1000.0

        for t in range(1, self.n_steps):
            # Generate imbalance signal
            imbalance = self.rng.uniform(-0.5, 0.5)

            # Encode in volume
            volumes[t] = 1000 + imbalance * 400  # Range: [800, 1200]

            # Price follows imbalance with ACCURACY probability
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

        ohlc = self._prices_to_ohlc(prices)
        ohlc['volume'] = volumes
        data[f'SYN_{i:02d}'] = ohlc

    return data
```

### Pass Criteria
- Average reward: > +1.0%
- Must show correlation between volume and trade direction
- Prediction accuracy: > 70%

---

## Level 13: Market Making

### Mathematical Foundation

Avellaneda-Stoikov market making:
```
Quote bid at: mid - spread/2 - inventory_penalty
Quote ask at: mid + spread/2 + inventory_penalty

Where inventory_penalty = γ × inventory × σ²
```

### Generator Implementation

```python
def _level_13_market_making(self) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Level 13: Market making with adverse selection.

    Structure:
    - 95% of time: Small random noise (±0.1%)
    - 5% of time: Informed order (large move ±1%)

    Optimal Strategy:
    - Provide liquidity most of the time
    - Detect and avoid informed flow
    - Manage inventory risk
    """
    data = {}

    NOISE_SIZE = 0.001      # ±0.1% normal moves
    INFORMED_PROB = 0.05    # 5% informed orders
    INFORMED_SIZE = 0.01    # ±1% on informed orders

    for i in range(self.n_assets):
        prices = np.zeros(self.n_steps)
        prices[0] = 100.0

        for t in range(1, self.n_steps):
            if self.rng.rand() < INFORMED_PROB:
                # Informed order (adverse selection)
                direction = self.rng.choice([-1, 1])
                move = direction * INFORMED_SIZE
            else:
                # Normal noise
                move = self.rng.uniform(-NOISE_SIZE, NOISE_SIZE)

            prices[t] = prices[t-1] * (1 + move)

        data[f'SYN_{i:02d}'] = self._prices_to_ohlc(prices)

    return data
```

### Pass Criteria
- Average reward: > +0.3%
- Inventory turnover: High (avoid accumulating position)
- Must avoid large losses on informed orders

---

## Level 14: Volatility Clustering (GARCH)

### Mathematical Foundation

GARCH(1,1) process:
```
Return_t = σ_t × ε_t, where ε_t ~ N(0,1)
σ_t² = α + β × σ_{t-1}² + γ × Return_{t-1}²

Parameters: α=0.05, β=0.85, γ=0.10
```

### Generator Implementation

```python
def _level_14_volatility_clustering(self) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Level 14: GARCH volatility dynamics.

    Structure:
    - Volatility is autocorrelated
    - High vol follows high vol
    - Low vol follows low vol

    Optimal Strategy:
    - Detect volatility regime
    - Reduce position size in high vol
    - Increase size in low vol trends
    """
    data = {}

    ALPHA = 0.05    # Base variance
    BETA = 0.85     # Variance persistence
    GAMMA = 0.10    # Shock impact

    for i in range(self.n_assets):
        prices = np.zeros(self.n_steps)
        variances = np.zeros(self.n_steps)

        prices[0] = 100.0
        variances[0] = 0.0001  # Initial variance (1% vol)

        for t in range(1, self.n_steps):
            # GARCH variance update
            prev_return = np.log(prices[t-1] / prices[max(0, t-2)]) if t > 1 else 0
            variances[t] = ALPHA + BETA * variances[t-1] + GAMMA * prev_return**2

            # Generate return with current volatility
            vol = np.sqrt(variances[t])
            epsilon = self.rng.normal(0, 1)
            ret = vol * epsilon

            prices[t] = prices[t-1] * np.exp(ret)
            prices[t] = max(prices[t], 0.01)

        data[f'SYN_{i:02d}'] = self._prices_to_ohlc(prices)

    return data
```

### Pass Criteria
- Average reward: > +0.5%
- Must show vol-aware position sizing
- Max drawdown < 15%

---

## Implementation Plan

### Phase 1: Statistical Arbitrage (Levels 5-7)
1. Implement `_level_5_pairs_trading()`
2. Implement `_level_6_triangle_arbitrage()`
3. Implement `_level_7_index_arbitrage()`
4. Add to curriculum config

### Phase 2: Momentum & Reversal (Levels 8-10)
1. Implement `_level_8_momentum()`
2. Implement `_level_9_mean_reversion()`
3. Implement `_level_10_breakout()`

### Phase 3: Microstructure (Levels 11-13)
1. Implement `_level_11_bid_ask_bounce()`
2. Implement `_level_12_order_imbalance()`
3. Implement `_level_13_market_making()`

### Phase 4: Volatility (Level 14)
1. Implement `_level_14_volatility_clustering()`

---

## Curriculum Integration

Update `config/pretrain_curriculum.json` to include advanced levels:

```json
{
  "stage": "5.1",
  "level": 5,
  "generator": "pairs_trading",
  "tactics": ["basic", "morphing"],
  "epochs": 300,
  "steps": 600,
  "sims": 32,
  "fee_bps": 10,
  "slippage_bps": 2,
  "pass_threshold": 1.5,
  "description": "Statistical Arbitrage: Cointegrated pairs"
}
```

---

## Success Criteria

A model is considered "fully trained" when it passes:
1. **Level 14** with +0.5% reward AND
2. **Max drawdown < 15%** AND
3. **No level regression** (can still pass Levels 0-13)

This represents mastery of:
- Basic mechanics (Levels 0-4)
- Statistical arbitrage (Levels 5-7)
- Momentum/reversal (Levels 8-10)
- Microstructure (Levels 11-13)
- Volatility dynamics (Level 14)
