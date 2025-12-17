# TASK 02: Synthetic Market Generators (Levels 0-3)

## Objective
Implement synthetic market data generators for curriculum training Levels 0-3. Enable training on mathematically controlled environments before real data.

## Prerequisites
- TASK_01 completed (parallel MCTS working)

## Scope
Create 4 synthetic data generators:
- **Level 0**: Pure sine waves (trend detection)
- **Level 1**: Lagged pairs (correlation detection)
- **Level 2**: Dip crashes (patience/timing)
- **Level 3**: Noisy trends (fee awareness)

## Implementation

### File Structure
```
mcts_trader/data/
├── history.py           # Existing (real data)
├── synthetic_market.py  # NEW (this task)
└── __init__.py
```

### Step 1: Create Base Class

**File**: `mcts_trader/data/synthetic_market.py` (NEW)

```python
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
            3: self._level_3_noisy_trends
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
            period = np.random.randint(10, 50)      # 10-50 steps per cycle
            amplitude = np.random.uniform(0.05, 0.20)  # 5-20% amplitude
            phase = np.random.uniform(0, 2*np.pi)   # Random phase
            baseline = 100.0

            # Generate sine wave
            t = np.arange(self.n_steps)
            close_prices = baseline * (1 + amplitude * np.sin(2*np.pi*t/period + phase))

            # OHLC with tiny spread (no randomness)
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

        # Generate 32 leader assets
        leaders = {}
        for i in range(32):
            period = np.random.randint(15, 40)
            amplitude = np.random.uniform(0.10, 0.25)
            phase = np.random.uniform(0, 2*np.pi)

            t = np.arange(self.n_steps)
            close = 100 * (1 + amplitude * np.sin(2*np.pi*t/period + phase))
            leaders[i] = close

        # Create followers (lagged + noise)
        for i in range(32):
            # Leader
            leader_close = leaders[i]
            data[f'SYN_{i:02d}'] = self._prices_to_ohlc(leader_close)

            # Follower (lagged)
            follower_close = np.roll(leader_close, LAG)
            follower_close[:LAG] = leader_close[0]  # Fill gap

            # Add 1% noise
            noise = np.random.normal(0, 0.01, size=self.n_steps)
            follower_close *= (1 + noise)

            data[f'SYN_{i+32:02d}'] = self._prices_to_ohlc(follower_close)

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
            for t in range(self.n_steps - 1):
                if np.random.rand() < CRASH_PROB:
                    prices[t] *= (1 - CRASH_SIZE)  # Crash
                    # Instant recovery (next step back to baseline)

            # Add tiny noise (0.5%)
            noise = np.random.normal(0, 0.005, size=self.n_steps)
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
                if not in_trend and np.random.rand() < (TREND_PROB / TREND_LENGTH):
                    in_trend = True
                    trend_remaining = TREND_LENGTH
                    trend_direction = np.random.choice([-1, 1])

                # Drift (if in trend)
                drift = TREND_DRIFT * trend_direction if in_trend else 0

                # Noise
                noise = np.random.normal(0, NOISE_STD)

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

    def _prices_to_ohlc(self, close_prices: np.ndarray) -> Dict[str, np.ndarray]:
        """Helper: Convert close prices to OHLC"""
        # Simple OHLC generation (close ± random spread)
        spread = 0.002  # 0.2% spread

        open_prices = close_prices * (1 + np.random.uniform(-spread, spread, size=len(close_prices)))
        high_prices = close_prices * (1 + np.random.uniform(0, spread*2, size=len(close_prices)))
        low_prices = close_prices * (1 - np.random.uniform(0, spread*2, size=len(close_prices)))

        # Ensure high >= close >= low
        high_prices = np.maximum(high_prices, close_prices)
        low_prices = np.minimum(low_prices, close_prices)

        return {
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices
        }
```

### Step 2: Integrate into FastTradingEnv

**File**: `train_jax_fast.py`

**Modify** FastTradingEnv.__init__ to accept synthetic market:

```python
# Around line 32
class FastTradingEnv(gym.Env):
    def __init__(
        self,
        db_path: str = "trading_data.db",
        n_assets: int = 64,
        # ... existing args ...
        synthetic_level: Optional[int] = None,  # NEW
        synthetic_steps: int = 10000             # NEW
    ):
        # ... existing code ...

        # NEW: Use synthetic market if level specified
        if synthetic_level is not None:
            from mcts_trader.data.synthetic_market import SyntheticMarket

            logging.info(f"Using SYNTHETIC market (Level {synthetic_level})")
            self.market = SyntheticMarket(
                level=synthetic_level,
                n_assets=n_assets,
                n_steps=synthetic_steps
            )
            self.padded_symbols = list(self.market.assets.values())
            self.n_assets = n_assets

        else:
            # EXISTING: Use real historical data
            self.market = HistoricalMarket(
                db_path,
                n_assets=len(unique_symbols) if unique_symbols else n_assets,
                # ... rest of existing args ...
            )
```

**Add CLI argument**:

```python
# Around line 415 (argparse section)
parser.add_argument('--synthetic-level', type=int, default=None,
                   help='Use synthetic data (0=sine, 1=lag, 2=dip, 3=trends, None=real)')
parser.add_argument('--synthetic-steps', type=int, default=10000,
                   help='Steps per synthetic episode')
```

**Pass to env**:

```python
# Around line 487 (env creation)
env = FastTradingEnv(
    n_assets=n_assets,
    # ... existing args ...
    synthetic_level=args.synthetic_level,  # NEW
    synthetic_steps=args.synthetic_steps   # NEW
)
```

## Testing

### Test 1: Level 0 Data Generation

**Script**: `tools/visualize_synthetic.py` (NEW)

```python
import matplotlib.pyplot as plt
import numpy as np
from mcts_trader.data.synthetic_market import SyntheticMarket

def visualize_level(level: int):
    market = SyntheticMarket(level=level, n_assets=64, n_steps=1000)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i in range(4):
        symbol = f'SYN_{i:02d}'
        prices = market.data[symbol]['close']

        axes[i].plot(prices[:200], linewidth=1.5)
        axes[i].set_title(f'{symbol} (Level {level})')
        axes[i].set_xlabel('Step')
        axes[i].set_ylabel('Price')
        axes[i].grid(True, alpha=0.3)

    plt.suptitle(f'Synthetic Level {level} Preview', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'synthetic_level_{level}_preview.png', dpi=150)
    print(f"Saved: synthetic_level_{level}_preview.png")

if __name__ == "__main__":
    for level in [0, 1, 2, 3]:
        visualize_level(level)
```

**Run**:
```bash
python tools/visualize_synthetic.py
# Check generated PNGs - verify patterns look correct
```

**Expected**:
- Level 0: Clean sine waves
- Level 1: First 32 assets, then their lagged copies
- Level 2: Flat with occasional sharp dips
- Level 3: Noisy with embedded trends

### Test 2: Integration with FastTradingEnv

**Command**:
```bash
python train_jax_fast.py \
  --synthetic-level 0 \
  --synthetic-steps 1000 \
  --epochs 10 \
  --steps 200 \
  --name test_synthetic_l0
```

**Expected Output**:
```
Using SYNTHETIC market (Level 0)
✅ Synthetic Level 0 data ready
Environment ready (MARGIN mode, 64 assets, 200 steps/episode)
...
Epoch 1 | Reward: +3.45 | ...
```

**Pass Criteria**: Training runs without errors.

### Test 3: Level 0 Learnability

**Full training run**:
```bash
python train_jax_fast.py \
  --synthetic-level 0 \
  --epochs 1000 \
  --steps 200 \
  --batch-size 512 \
  --sims 32 \
  --name level_0_full
```

**Monitor**: `logs/training_fast_level_0_full.log`

**Pass Criteria**:
- Avg reward > +5% within 500 epochs
- Model learns to buy at troughs, sell at peaks

**If fails**: Architecture is broken - STOP and debug before proceeding.

## Acceptance Criteria

✅ **PASS if ALL met**:
1. All 4 levels generate data without errors
2. Visualizations look correct (sine, lag, dip, trends)
3. FastTradingEnv works with `--synthetic-level 0`
4. Level 0 training converges (reward > +5%)

## Debugging

**If Level 0 doesn't converge**:

1. **Check observation encoder**:
```python
# Add debug prints
obs = env._get_observation()
print(f"Obs shape: {obs['observation'].shape}")
print(f"Obs range: [{obs['observation'].min():.2f}, {obs['observation'].max():.2f}]")
```

2. **Check actions are executing**:
```python
# In env.step()
print(f"Action {action}: asset_idx={asset_idx}, type={action_type}")
print(f"Portfolio value: {self.portfolio.state.total_value:.2f}")
```

3. **Manual trading test**:
```python
# Force buy at troughs
env.reset()
for step in range(200):
    prices = [env.market.data[f'SYN_{i:02d}']['close'][step] for i in range(64)]
    min_idx = np.argmin(prices)
    action = min_idx * 3 + 2  # BUY

    obs, reward, done, truncated, info = env.step(action)
    print(f"Step {step}: Bought {min_idx}, Reward: {reward:.2f}")
```

If manual trading is profitable but model isn't learning → problem is in model/training, not env.

## Integration Checklist

- [ ] Create `mcts_trader/data/synthetic_market.py`
- [ ] Add synthetic level support to `train_jax_fast.py`
- [ ] Create visualization script
- [ ] Test all 4 levels generate correctly
- [ ] Run full Level 0 training to convergence
- [ ] Commit: "Add synthetic market generators (Levels 0-3)"

## Next Task

Once Level 0 passes → **TASK_03_SCALE_MODEL_TO_25M.md**
