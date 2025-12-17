# Synthetic Curriculum Training Design

## Philosophy

**Core Principle**: "If a small TRM cannot solve a noiseless sine wave, it will never solve Bitcoin."

This curriculum establishes a **synthetic pre-training phase** where the model learns fundamental trading mechanics in controlled, interpretable environments before facing real market data. This approach:

1. **Validates Architecture** - Proves the TRM + MCTS + Correlation Matrix encoder actually works
2. **Prevents Turtle Mode** - Shows the model that profitable trading IS possible
3. **Builds Intuition** - Imprints core patterns (lag detection, dip catching, trend following)
4. **Enables Debugging** - If model fails Level 0, the code is broken (not the data)

---

## Phase 1: Mechanics Levels (Synthetic Data)

### Level 0: The Trend Surfer
**Objective**: Verify basic Buy Low / Sell High

**Data Characteristics**:
- 64 assets, each a pure sine wave
- Different periods (5-50 steps) and amplitudes (5-20%)
- Zero noise, zero fees, zero slippage
- Perfect predictability

**Expected Behavior**:
- Agent should learn to Buy at troughs, Sell at peaks
- Target reward: +10% per episode (200 steps)
- **Pass Criteria**: Avg reward > +5% over 100 episodes

**Why This Matters**:
If the agent cannot beat a noiseless sine wave, the neural network is not learning the input-output mapping. This indicates a bug in:
- State encoder (is price data reaching the model?)
- Action execution (are trades actually happening?)
- Reward calculation (is PnL computed correctly?)

---

### Level 1: The Lag Hunter
**Objective**: Verify Correlation Matrix Encoder

**Data Characteristics**:
- 64 assets split into 32 pairs
- Asset B = Asset A shifted by 10 steps (perfect lag)
- Example: BTC is a sine wave, ETH follows BTC with 10-minute delay
- Zero noise initially, then add 1% Gaussian noise

**Expected Behavior**:
- Agent must learn to watch Asset A to predict Asset B
- If BTC peaked 10 minutes ago, ETH will peak NOW
- The correlation matrix should highlight these pairs
- Target reward: +8% per episode

**Pass Criteria**: Avg reward > +4% over 100 episodes

**Why This Matters**:
This is the **critical test for multi-asset architecture**. If the model treats each asset independently (like a single-asset trader), it will fail. Success here proves:
- Correlation matrix encoder is working
- Attention mechanism (if using Transformer) captures cross-asset dependencies
- Model understands temporal relationships

---

### Level 2: The Dip Catcher
**Objective**: Verify Conditional Patience / HOLD mastery

**Data Characteristics**:
- Flat baseline (price = 100)
- Rare random -10% crashes (V-shape recovery in 1 step)
- Crash probability: 2% per step
- Small noise (0.5%)

**Expected Behavior**:
- Agent should HOLD most of the time (flat = no opportunity)
- When crash detected, BUY immediately
- Sell after recovery
- Avoid false alarms (noise-triggered buys)

**Pass Criteria**: Sharpe ratio > 1.5 (reward vs volatility)

**Why This Matters**:
Tests risk management and patience. The agent must learn:
- HOLD is the correct action most of the time (defeats "always trade" behavior)
- Rare opportunities justify action
- Prevents "thrashing" (buying/selling on every tiny move due to fees)

---

### Level 3: The Spread Beater
**Objective**: Verify Fee/Slippage Awareness

**Data Characteristics**:
- Noisy random walk with embedded trends
- Trends: 60% of the time, price drifts +0.5%/step for 20 steps
- Noise: 2% Gaussian
- **High Fees**: 26 bps + 5 bps slippage

**Expected Behavior**:
- Agent must ONLY trade during strong trends
- Must avoid "choppy" periods (noise without trend)
- Position sizing matters (small trend = small position)
- Target reward: +2% per episode (realistic after fees)

**Pass Criteria**: Avg reward > +1% AND trade frequency < 20% (agent is selective)

**Why This Matters**:
This is the **bridge to real markets**. Real trading has:
- High transaction costs (26 bps is realistic for Kraken)
- Noise dominates signal
- Most of the time, the correct action is HOLD

If the agent beats Level 3, it has learned:
1. Trend detection (signal > noise)
2. Cost-benefit analysis (only trade when reward > cost)
3. Risk management (position sizing)

**Failure Mode**: If agent "turtles" here (100% HOLD), it means it's overfitting to cost avoidance. Solution: Temporarily reduce fees to 10 bps, let it learn to trade, then ramp back up.

---

## Phase 2: Augmented Reality (Transition to Real Data)

### 2.1 The Mirror World (Inversion)
**Concept**: Take real historical data `P`, create `P_inv = 1/P`

**Example**:
- BTC pumps 2020-2021: In mirror world, it crashes
- ETH crashes 2022: In mirror world, it pumps

**Benefit**:
- Doubles dataset size instantly
- Forces model to learn that "Shorting a top" = "Buying a bottom" (mathematically identical)
- Prevents long-only bias

**Implementation**:
```python
def mirror_prices(ohlcv_data):
    """Invert all prices"""
    inverted = ohlcv_data.copy()
    for field in ['open', 'high', 'low', 'close']:
        inverted[field] = 1.0 / ohlcv_data[field]
    # Swap high/low because inversion flips them
    inverted['high'], inverted['low'] = inverted['low'], inverted['high']
    return inverted
```

---

### 2.2 The Time Warp (Temporal Scaling)
**Concept**: Randomly speed up or slow down segments

**Example**:
- 5-minute crash in 2020 → Stretch to 30-minute bleed
- 2-hour pump → Compress to 20-minute spike

**Benefit**:
- Prevents memorization of exact candle counts
- Forces model to learn relative patterns (shape) vs absolute timing
- Agent can't learn "Buy after exactly 4 red candles"

**Implementation**:
```python
import scipy.interpolate as interp

def warp_time(ohlcv_data, warp_factor):
    """
    warp_factor > 1: Slow down (more candles)
    warp_factor < 1: Speed up (fewer candles)
    """
    old_time = np.arange(len(ohlcv_data))
    new_time = np.linspace(0, len(ohlcv_data)-1, int(len(ohlcv_data) * warp_factor))

    warped = {}
    for field in ['open', 'high', 'low', 'close', 'volume']:
        f = interp.interp1d(old_time, ohlcv_data[field], kind='linear')
        warped[field] = f(new_time)

    return warped
```

**Usage**: Apply random warp factor 0.5-2.0 to each episode segment.

---

### 2.3 The Noise Injection
**Concept**: Add Gaussian noise to prices

**Benefit**:
- If agent relies on exact price levels (e.g., "Buy BTC at $20,000.00"), this breaks that
- Forces reliance on relative structures (RSI, MACD, correlations)
- Improves generalization

**Implementation**:
```python
def add_noise(ohlcv_data, noise_std=0.01):
    """Add 1% Gaussian noise to prices"""
    noisy = ohlcv_data.copy()
    noise = np.random.normal(0, noise_std, size=len(ohlcv_data))

    for field in ['open', 'high', 'low', 'close']:
        noisy[field] *= (1 + noise)

    return noisy
```

---

## Phase 3: The Gladiator (Dynamic Difficulty on Real Data)

### Adaptive Difficulty Scaler

**Current Implementation** (in `run_scheduled_training.py`):
```python
current_fee = 0
current_slippage = 0

for episode in training:
    reward = agent.train(fee=current_fee, slippage=current_slippage)

    if reward > 5.0:  # Agent is winning
        # Ramp up slippage first (easier to adapt to)
        if current_slippage < 5:
            current_slippage += 1
        # Then ramp up fees (harder)
        elif current_fee < 26:
            current_fee += 2
```

**Proposed Enhancement**:
Add volatility/noise scaling:
```python
volatility_multiplier = 1.0

if reward > 5.0:
    # Also increase noise
    volatility_multiplier += 0.1  # Make market noisier
```

**Why**: As agent masters clean data, we add noise to prevent overfitting to specific historical patterns.

---

## Implementation Architecture

### File Structure
```
mcts_trader/
├── data/
│   ├── synthetic_market.py       # NEW: Generates Levels 0-3
│   ├── augmented_market.py       # NEW: Mirror/Warp/Noise transforms
│   └── history.py                # EXISTING: Real data loader
├── env/
│   └── trading_env_curriculum.py # NEW: Env wrapper for curriculum
└── tools/
    └── run_curriculum.py         # NEW: Curriculum orchestrator
```

### SyntheticMarket Class (New)

```python
class SyntheticMarket:
    """Generates synthetic price data for curriculum training"""

    def __init__(self, level: int, n_assets: int = 64, n_steps: int = 10000):
        self.level = level
        self.n_assets = n_assets
        self.n_steps = n_steps
        self.data = self._generate()

    def _generate(self):
        if self.level == 0:
            return self._generate_sine_waves()
        elif self.level == 1:
            return self._generate_lagged_pairs()
        elif self.level == 2:
            return self._generate_dip_crashes()
        elif self.level == 3:
            return self._generate_noisy_trends()

    def _generate_sine_waves(self):
        """Level 0: Pure sine waves"""
        data = {}
        for i in range(self.n_assets):
            period = np.random.randint(5, 50)
            amplitude = np.random.uniform(0.05, 0.2)
            phase = np.random.uniform(0, 2*np.pi)

            t = np.arange(self.n_steps)
            prices = 100 * (1 + amplitude * np.sin(2*np.pi*t/period + phase))

            data[f'ASSET_{i}'] = {
                'timestamp': t,
                'open': prices,
                'high': prices * 1.001,
                'low': prices * 0.999,
                'close': prices,
                'volume': np.ones_like(prices) * 1000
            }

        return data

    # ... other levels ...
```

### CurriculumEnv Wrapper (New)

```python
class CurriculumTradingEnv(FastTradingEnv):
    """Extends FastTradingEnv to use synthetic data"""

    def __init__(self, level: int, **kwargs):
        self.level = level
        self.synthetic_market = SyntheticMarket(level=level)

        # Override FastTradingEnv's market with synthetic
        super().__init__(**kwargs)
        self.market = self.synthetic_market
```

### Curriculum Orchestrator (New)

```python
# tools/run_curriculum.py

def run_curriculum():
    """Run full synthetic curriculum"""

    for level in range(4):  # Levels 0-3
        print(f"=== Starting Level {level} ===")

        env = CurriculumTradingEnv(level=level)

        # Train until pass criteria
        while True:
            metrics = train_epoch(env)

            if meets_pass_criteria(level, metrics):
                print(f"✅ Level {level} PASSED")
                save_checkpoint(f'level_{level}_complete.pkl')
                break

            if metrics['epochs'] > MAX_EPOCHS_PER_LEVEL:
                print(f"❌ Level {level} FAILED - Could not meet criteria")
                sys.exit(1)

    print("🎓 Synthetic curriculum COMPLETE! Moving to real data...")

    # Phase 2: Augmented real data
    env = AugmentedTradingEnv(
        use_mirror=True,
        use_warp=True,
        use_noise=True
    )
    train(env, epochs=10000)
```

---

## Pass Criteria Summary

| Level | Avg Reward | Trade Frequency | Sharpe | Notes |
|-------|------------|-----------------|--------|-------|
| 0 | > +5% | Any | Any | Pure execution test |
| 1 | > +4% | Any | Any | Correlation test |
| 2 | Any | < 10% | > 1.5 | Patience test |
| 3 | > +1% | < 20% | > 1.0 | Fee awareness |

**Progression Rule**: Must pass current level before advancing. If agent fails a level after 5000 epochs, STOP and debug (architecture is broken, not the curriculum).

---

## Integration with Existing System

### Modify `train_jax_fast.py`

Add `--synthetic-level` argument:
```python
parser.add_argument('--synthetic-level', type=int, default=None,
                   help='Use synthetic data (0-3 for levels, None for real)')

if args.synthetic_level is not None:
    env = CurriculumTradingEnv(level=args.synthetic_level, ...)
else:
    env = FastTradingEnv(...)  # Real data
```

### Modify `run_scheduled_training.py`

Add pre-training phase:
```python
# Phase 0: Synthetic Pre-training
for level in range(4):
    run_training_run(
        synthetic_level=level,
        epochs=1000,
        ...
    )

# Phase 1: Real data with curriculum
for period in schedule['periods']:
    # ... existing logic ...
```

---

## Expected Timeline

| Phase | Duration | Epochs | Outcome |
|-------|----------|--------|---------|
| Level 0 | 1 hour | 500 | Baseline validation |
| Level 1 | 2 hours | 1000 | Correlation mastery |
| Level 2 | 2 hours | 1000 | Patience training |
| Level 3 | 4 hours | 2000 | Fee awareness |
| **Total Synthetic** | **9 hours** | **4500** | **Foundation ready** |
| Augmented Real | 12 hours | 5000 | Generalization |
| Pure Real | ∞ | 50000+ | Production agent |

---

## Validation Tools

### Synthetic Data Visualizer
```python
# tools/visualize_synthetic.py
import matplotlib.pyplot as plt

market = SyntheticMarket(level=1)
data = market.get_asset_data('ASSET_0')

plt.plot(data['close'])
plt.title('Level 1: Lagged Asset Pair')
plt.show()
```

### Curriculum Debugger
```python
# If agent fails Level 0, run this:
def debug_level_0():
    env = CurriculumTradingEnv(level=0)
    obs, _ = env.reset()

    # Manual trading loop
    for step in range(200):
        # Force BUY at troughs
        prices = [env.market.get_asset_data(f'ASSET_{i}')['close'][step]
                  for i in range(64)]
        min_idx = np.argmin(prices)
        action = min_idx * 3 + 2  # BUY

        obs, reward, done, truncated, info = env.step(action)
        print(f"Step {step}: Action={action}, Reward={reward:.2f}")

    print(f"Final Portfolio Value: {env.portfolio.state.total_value}")
```

---

## Next Steps

1. **Implement SyntheticMarket class** (`mcts_trader/data/synthetic_market.py`)
2. **Create curriculum orchestrator** (`tools/run_curriculum.py`)
3. **Add `--synthetic-level` to train_jax_fast.py**
4. **Run Level 0 validation** (should pass in < 1 hour)
5. **If Level 0 fails**: Debug state encoder, action execution, reward calc
6. **If Level 0 passes**: Proceed to Level 1 (correlation test)

---

## Success Metrics

**Before Real Data**:
- All 4 synthetic levels PASSED
- Model demonstrates: Buy Low/Sell High, Lag Detection, Patience, Fee Awareness
- Checkpoint saved: `synthetic_curriculum_complete.pkl`

**After Augmented Reality**:
- Model generalizes to mirror/warped/noisy data
- No overfitting to specific historical patterns

**Production Readiness**:
- Model beats Level 3 criteria on REAL data (reward > +1%, trades < 20%)
- Sharpe ratio > 1.0 on held-out validation set
