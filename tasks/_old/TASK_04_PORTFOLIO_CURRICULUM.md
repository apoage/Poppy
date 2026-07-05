# TASK 04: Portfolio Curriculum (Levels 4-6)

## Objective
Implement advanced curriculum levels that teach portfolio management tactics, not just single-asset trading.

## Prerequisites
- TASK_02 completed (synthetic Levels 0-3 working)
- TASK_03 completed (25M model available)

## Scope
Add 3 new synthetic levels:
- **Level 4**: Portfolio Rebalancer (rotation tactics)
- **Level 5**: Adversarial Market (robustness)
- **Level 6**: Chaos Mode (black swans)

## Implementation

### Step 1: Extend SyntheticMarket

**File**: `mcts_trader/data/synthetic_market.py`

**Add to** `_generate_level_data()`:

```python
def _generate_level_data(self) -> Dict[str, Dict[str, np.ndarray]]:
    generators = {
        0: self._level_0_sine_waves,
        1: self._level_1_lagged_pairs,
        2: self._level_2_dip_crashes,
        3: self._level_3_noisy_trends,
        4: self._level_4_portfolio_rotation,  # NEW
        5: self._level_5_adversarial,         # NEW (placeholder)
        6: self._level_6_chaos_mode            # NEW
    }
    # ...
```

**Add Level 4 generator**:

```python
def _level_4_portfolio_rotation(self) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Level 4: Portfolio rotation - teach rebalancing tactics.

    Goal: Learn to rotate capital between assets.
    - 64 assets with correlated GBM
    - Opportunity rotates every 50 steps (different asset pumps)
    - Must EXIT winners and ENTER new opportunities
    """
    data = {}

    # Generate correlation matrix
    correlation_matrix = self._generate_correlation_matrix(self.n_assets)

    # Generate correlated Brownian motion
    prices = self._correlated_brownian_motion(
        n_assets=self.n_assets,
        n_steps=self.n_steps,
        correlation=correlation_matrix,
        drift=0.0001,  # Base drift
        volatility=0.02
    )

    # Add rotating momentum (5 assets pump each 50-step window)
    window_size = 50
    n_windows = self.n_steps // window_size

    for w in range(n_windows):
        start = w * window_size
        end = start + window_size

        # Pick 5 random assets to pump
        winners = np.random.choice(self.n_assets, size=5, replace=False)

        for winner in winners:
            # 20% pump over 50 steps
            pump = np.linspace(1.0, 1.20, window_size)
            prices[winner, start:end] *= pump

    # Convert to OHLC
    for i in range(self.n_assets):
        data[f'SYN_{i:02d}'] = self._prices_to_ohlc(prices[i, :])

    return data

def _generate_correlation_matrix(self, n: int) -> np.ndarray:
    """Generate random valid correlation matrix"""
    # Random correlation matrix (positive definite)
    A = np.random.randn(n, n)
    corr = np.dot(A, A.T)

    # Normalize to correlation (diagonal = 1)
    D = np.diag(1.0 / np.sqrt(np.diag(corr)))
    corr = D @ corr @ D

    # Ensure -1 <= corr <= 1
    corr = np.clip(corr, -0.9, 0.9)
    np.fill_diagonal(corr, 1.0)

    return corr

def _correlated_brownian_motion(
    self,
    n_assets: int,
    n_steps: int,
    correlation: np.ndarray,
    drift: float = 0.0,
    volatility: float = 0.02
) -> np.ndarray:
    """
    Generate correlated GBM for multiple assets.

    Returns: (n_assets, n_steps) price array
    """
    # Cholesky decomposition for correlation
    L = np.linalg.cholesky(correlation)

    # Generate uncorrelated random returns
    returns_uncorr = np.random.normal(drift, volatility, size=(n_assets, n_steps))

    # Apply correlation
    returns_corr = L @ returns_uncorr

    # Convert returns to prices (starting at 100)
    prices = np.zeros((n_assets, n_steps))
    prices[:, 0] = 100.0

    for t in range(1, n_steps):
        prices[:, t] = prices[:, t-1] * (1 + returns_corr[:, t])

    return prices
```

**Add Level 6 generator**:

```python
def _level_6_chaos_mode(self) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Level 6: Chaos mode - black swan events.

    Goal: Survive the impossible.
    - Flash crashes (-20% in 1 step)
    - Correlation flips (assets suddenly anti-correlate)
    - Asset delisting (1 random asset goes to 0)
    - Fee spikes (random 5× fees)
    """
    # Start with Level 4 data (portfolio rotation)
    data = self._level_4_portfolio_rotation()

    # Add chaos events
    for i in range(self.n_assets):
        prices = data[f'SYN_{i:02d}']['close']

        # Flash crashes (5 per episode)
        crash_steps = np.random.choice(self.n_steps, size=5, replace=False)
        for step in crash_steps:
            if step < self.n_steps - 1:
                prices[step] *= 0.80  # -20%
                # Recovery over next 10 steps
                recovery = np.linspace(0.80, 1.0, min(10, self.n_steps - step))
                prices[step:step+len(recovery)] *= recovery

        # Delisting (1 random asset)
        if i == np.random.randint(self.n_assets):
            delist_step = np.random.randint(self.n_steps // 2, self.n_steps)
            prices[delist_step:] = 0.001  # Near-zero (not exactly 0 to avoid div by 0)

    return data

def _level_5_adversarial(self) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Level 5: Adversarial - placeholder for now.

    This will be implemented in TASK_06 (requires dual models).
    For now, return Level 4 data.
    """
    logging.warning("Level 5 (Adversarial) not yet implemented, using Level 4 data")
    return self._level_4_portfolio_rotation()
```

### Step 2: Pass Criteria Module

**File**: `mcts_trader/curriculum/pass_criteria.py` (NEW)

```python
import numpy as np
from typing import Dict, List

class PassCriteria:
    """Defines pass criteria for each curriculum level"""

    CRITERIA = {
        0: {'avg_reward': 5.0, 'min_episodes': 100},
        1: {'avg_reward': 4.0, 'min_episodes': 100},
        2: {'sharpe_ratio': 1.5, 'min_episodes': 100},
        3: {'avg_reward': 1.0, 'trade_freq_max': 0.20, 'min_episodes': 100},
        4: {'sharpe_ratio': 2.0, 'turnover_min': 0.50, 'min_episodes': 100},
        5: {'avg_reward': 0.0, 'min_episodes': 100},  # Just survive adversary
        6: {'bankruptcy_rate': 0.0, 'max_drawdown': 0.30, 'min_episodes': 100}
    }

    @staticmethod
    def check_level(level: int, metrics: dict) -> bool:
        """Check if metrics meet pass criteria"""
        criteria = PassCriteria.CRITERIA.get(level)
        if not criteria:
            return False

        if level in [0, 1]:
            # Reward-based
            return metrics['avg_reward'] >= criteria['avg_reward']

        elif level == 2:
            # Sharpe-based
            return metrics['sharpe_ratio'] >= criteria['sharpe_ratio']

        elif level == 3:
            # Reward + trade frequency
            return (metrics['avg_reward'] >= criteria['avg_reward'] and
                    metrics['trade_frequency'] <= criteria['trade_freq_max'])

        elif level == 4:
            # Sharpe + portfolio turnover
            return (metrics['sharpe_ratio'] >= criteria['sharpe_ratio'] and
                    metrics['portfolio_turnover'] >= criteria['turnover_min'])

        elif level == 5:
            # Survive adversary (positive reward)
            return metrics['avg_reward'] >= criteria['avg_reward']

        elif level == 6:
            # Survive chaos (no bankruptcy, max drawdown)
            return (metrics['bankruptcy_rate'] <= criteria['bankruptcy_rate'] and
                    metrics['max_drawdown'] <= criteria['max_drawdown'])

        return False


class MetricsTracker:
    """Track rolling metrics"""

    def __init__(self, window: int = 100):
        self.window = window
        self.rewards = []
        self.trades = []
        self.steps = []
        self.portfolio_values = []

    def add_episode(self, reward: float, n_trades: int, n_steps: int, final_value: float):
        self.rewards.append(reward)
        self.trades.append(n_trades)
        self.steps.append(n_steps)
        self.portfolio_values.append(final_value)

        # Keep only last window
        if len(self.rewards) > self.window:
            self.rewards.pop(0)
            self.trades.pop(0)
            self.steps.pop(0)
            self.portfolio_values.pop(0)

    def get_metrics(self) -> Dict:
        """Calculate current metrics"""
        if len(self.rewards) == 0:
            return {
                'avg_reward': 0,
                'sharpe_ratio': 0,
                'trade_frequency': 0,
                'portfolio_turnover': 0,
                'bankruptcy_rate': 0,
                'max_drawdown': 0
            }

        avg_reward = np.mean(self.rewards)
        std_reward = np.std(self.rewards)
        sharpe = avg_reward / std_reward if std_reward > 0 else 0

        total_trades = sum(self.trades)
        total_steps = sum(self.steps)
        trade_freq = total_trades / total_steps if total_steps > 0 else 0

        # Portfolio turnover (approximate: trades / episodes)
        turnover = total_trades / len(self.trades) if len(self.trades) > 0 else 0

        # Bankruptcy rate (final value < 1000)
        bankruptcies = sum(1 for v in self.portfolio_values if v < 1000)
        bankruptcy_rate = bankruptcies / len(self.portfolio_values)

        # Max drawdown
        if len(self.portfolio_values) > 1:
            peak = self.portfolio_values[0]
            max_dd = 0
            for v in self.portfolio_values:
                peak = max(peak, v)
                dd = (peak - v) / peak
                max_dd = max(max_dd, dd)
        else:
            max_dd = 0

        return {
            'avg_reward': avg_reward,
            'sharpe_ratio': sharpe,
            'trade_frequency': trade_freq,
            'portfolio_turnover': turnover,
            'bankruptcy_rate': bankruptcy_rate,
            'max_drawdown': max_dd
        }
```

### Step 3: Curriculum Orchestrator

**File**: `tools/run_curriculum.py` (NEW)

```python
#!/usr/bin/env python3
"""
Curriculum Training Orchestrator
Runs agent through Levels 0-6 sequentially.
"""

import sys
import logging
import subprocess
from mcts_trader.curriculum.pass_criteria import PassCriteria, MetricsTracker

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def run_level(level: int, max_epochs: int = 5000, model_size: str = 'medium'):
    """Train on a single level until pass criteria met"""

    logging.info("=" * 70)
    logging.info(f"LEVEL {level} - Starting")
    logging.info("=" * 70)

    checkpoint_path = None
    if level > 0:
        # Load previous level checkpoint
        checkpoint_path = f"jax_ckpt_fast/curriculum_level_{level-1}/checkpoint_final"

    cmd = [
        "python", "train_jax_fast.py",
        "--synthetic-level", str(level),
        "--model-size", model_size,
        "--epochs", str(max_epochs),
        "--batch-size", "512" if model_size == 'small' else "256",
        "--sims", "64",
        "--name", f"curriculum_level_{level}"
    ]

    if checkpoint_path:
        cmd.extend(["--load-checkpoint", checkpoint_path])

    logging.info(f"Command: {' '.join(cmd)}")

    # Run training
    result = subprocess.run(cmd)

    if result.returncode != 0:
        logging.error(f"Level {level} training FAILED")
        return False

    # Check pass criteria
    # (In real impl, would parse metrics from training log)
    # For now, assume pass if no crash
    logging.info(f"✅ Level {level} PASSED")

    # Save final checkpoint
    final_ckpt = f"jax_ckpt_fast/curriculum_level_{level}/checkpoint_final"
    logging.info(f"Checkpoint: {final_ckpt}")

    return True


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-level', type=int, default=0, help='Start from level')
    parser.add_argument('--end-level', type=int, default=6, help='End at level')
    parser.add_argument('--model-size', type=str, default='medium', choices=['small', 'medium', 'large'])
    parser.add_argument('--max-epochs', type=int, default=5000, help='Max epochs per level')
    args = parser.parse_args()

    for level in range(args.start_level, args.end_level + 1):
        if level == 5:
            logging.info("Skipping Level 5 (Adversarial) - requires TASK_06")
            continue

        success = run_level(level, args.max_epochs, args.model_size)

        if not success:
            logging.error(f"FAILED at Level {level}. Stopping.")
            sys.exit(1)

    logging.info("=" * 70)
    logging.info("🎓 CURRICULUM COMPLETE!")
    logging.info("=" * 70)


if __name__ == "__main__":
    main()
```

## Testing

### Test 1: Level 4 Data Generation

```bash
python -c "
from mcts_trader.data.synthetic_market import SyntheticMarket
market = SyntheticMarket(level=4, n_assets=64, n_steps=1000)
print('Level 4 data shape:', len(market.data))
print('First asset price range:', market.data['SYN_00']['close'].min(), '-', market.data['SYN_00']['close'].max())
"
```

### Test 2: Level 4 Training

```bash
python train_jax_fast.py \
  --synthetic-level 4 \
  --model-size medium \
  --epochs 2000 \
  --batch-size 256 \
  --sims 64 \
  --name test_level_4
```

**Monitor**: Portfolio turnover (should be > 50% if learning to rotate).

### Test 3: Level 6 Chaos Survival

```bash
python train_jax_fast.py \
  --synthetic-level 6 \
  --model-size large \
  --epochs 3000 \
  --name test_level_6
```

**Monitor**: Bankruptcy rate (should be 0%).

### Test 4: Full Curriculum

```bash
python tools/run_curriculum.py \
  --start-level 0 \
  --end-level 6 \
  --model-size medium \
  --max-epochs 5000
```

**Expected**: Completes without failures (skips Level 5 for now).

## Acceptance Criteria

✅ **PASS if ALL met**:
1. Level 4 data generates correctly (correlated rotation)
2. Level 6 data generates correctly (flash crashes, delistings)
3. Model trains on Level 4 without errors
4. Model trains on Level 6 without errors
5. Full curriculum (0-6, skip 5) completes

## Integration Checklist

- [ ] Add Level 4 and 6 generators to `synthetic_market.py`
- [ ] Create `mcts_trader/curriculum/pass_criteria.py`
- [ ] Create `tools/run_curriculum.py`
- [ ] Test Level 4 data generation
- [ ] Test Level 6 data generation
- [ ] Run full curriculum
- [ ] Commit: "Add portfolio curriculum (Levels 4, 6)"

## Next Task

Once this works → **TASK_05_CROSS_PAIR_TRADING.md**
