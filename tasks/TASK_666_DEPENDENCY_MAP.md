# TASK_666: Easy Mode Dependency Map

**Date**: 2026-02-04
**Purpose**: Document exactly which functions Easy Mode needs from existing infrastructure

---

## Overview: Data Flow for Easy Mode

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        EASY MODE DATA PIPELINE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐    │
│  │  DATA SOURCE     │     │  ENVIRONMENT     │     │  ORACLE          │    │
│  │                  │     │                  │     │                  │    │
│  │ SyntheticMarket  │────▶│  env_factory.py  │────▶│  path_solver.py  │    │
│  │ SQLiteProvider   │     │  jax_trading_env │     │                  │    │
│  │ PriceCache       │     │                  │     │                  │    │
│  └──────────────────┘     └────────┬─────────┘     └────────┬─────────┘    │
│                                    │                        │              │
│                                    ▼                        ▼              │
│                          ┌──────────────────┐     ┌──────────────────┐    │
│                          │  ENCODING        │     │  TRAINING        │    │
│                          │                  │     │                  │    │
│                          │  jax_encoder.py  │────▶│  easy_trainer.py │    │
│                          │                  │     │  (NEW)           │    │
│                          └──────────────────┘     └──────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. PRICE CACHE (`mcts_trader/jax_agent/price_cache.py`)

### Functions Needed by Easy Mode

| Function | Line | Signature | Purpose |
|----------|------|-----------|---------|
| `PriceCache.__init__` | 30-91 | `(cache_path, max_episodes=10000, max_steps=1000, n_assets=64, n_features=5, readonly=False)` | Initialize memory-mapped storage |
| `add_episode` | 93-141 | `(prices: np.ndarray, start_date=0, regime_tag=0) -> int` | Store episode, returns episode_id |
| `get_observation_window` | 143-184 | `(episode_id, step, window_size=60) -> np.ndarray` | Get causal window (T, n, 5) |
| `get_batch_observations` | 186-211 | `(episode_ids, steps, window_size=60) -> np.ndarray` | Batch retrieval (B, T, n, 5) |
| `get_returns` | 213-240 | `(episode_id, start_step=0, end_step=None) -> np.ndarray` | Log returns for oracle |
| `get_episode_prices` | 251-268 | `(episode_id) -> np.ndarray` | Full price array for episode |
| `flush` | 270-282 | `()` | Persist to disk |
| `create_price_cache` | 314-336 | `(cache_dir, max_episodes, n_assets) -> PriceCache` | Convenience factory |

### Critical Bug Fixes Embedded
- **Line 90**: `threading.RLock()` (TASK_128 fix - prevents deadlock)
- **Lines 165-168**: Causal window `[step - window_size, step - 1]` (agent sees t-1, not t)

### Usage in Easy Mode
```python
# In generate_oracle_data.py
cache = create_price_cache('./oracle_cache', max_episodes=50000)
episode_id = cache.add_episode(prices, start_date=timestamp)
returns = cache.get_returns(episode_id)  # For path_solver
window = cache.get_observation_window(episode_id, step, window_size=60)  # For encoding
```

---

## 2. JAX ENCODER (`mcts_trader/env/jax_encoder.py`)

### Functions Needed by Easy Mode

| Function | Line | Signature | Purpose |
|----------|------|-----------|---------|
| `EncoderConfig` | 46-61 | `dataclass(n_assets=64, window_size=60, n_channels=27)` | Config |
| `encode_observation_simple` | 450-467 | `(prices, window_start, config) -> (27, n, n)` | Simple encoder (prices only) |
| `jit_batch_encode_simple` | 491-516 | `(prices, window_starts, config, per_env=False) -> (B, 27, n, n)` | Batched encoding |
| `encode_observation_compressed` | 688-708 | `(prices, window_start, config) -> (27, n)` | 64x smaller format |
| `jit_batch_encode_compressed` | 721-739 | `(prices, window_starts, config) -> (B, 27, n)` | Batched compressed |
| `tile_compressed_observation` | 742-756 | `(compressed, n_assets) -> (27, n, n)` | Expand at sample time |
| `jit_batch_tile_compressed` | 768-782 | `(compressed_batch, n_assets) -> (B, 27, n, n)` | Batched expansion |

### 27 Channels (for reference)
```
0-3:   Normalized OHLC (latest, tiled)
4-7:   Log returns at 1, 5, 15, 60 steps
8-11:  Volume patterns (recent vs average)
12-15: Price momentum (rate of change)
16-19: Volatility (rolling std of returns)
20-23: Price percentiles (position in range)
24-26: VWAP and bands
```

### Critical Bug Fixes Embedded
- **Line 186-187**: `prev_prices` for return denominator (TASK_122 fix)
- Safe division guards: `(x + 1e-8)` throughout

### Usage in Easy Mode
```python
# In easy_trainer.py - compressed format for storage (64x smaller)
config = EncoderConfig(n_assets=64, window_size=60)
compressed_obs = jit_batch_encode_compressed(prices, window_starts, config)  # (B, 27, 64)

# At sample time - expand for model input
full_obs = jit_batch_tile_compressed(compressed_obs, n_assets=64)  # (B, 27, 64, 64)
```

---

## 3. PATH SOLVER / ORACLE (`mcts_trader/jax_agent/path_solver.py`)

### Functions Needed by Easy Mode

| Function | Line | Signature | Purpose |
|----------|------|-----------|---------|
| `OptimalPath` | 20-26 | `NamedTuple(selections, indices, cumulative_pnl, total_alpha, trade_count)` | Result type |
| `greedy_lookahead_path` | 38-154 | `(returns, k=4, lookahead=5, transaction_cost=0.001, max_len=512) -> OptimalPath` | **Primary oracle** |
| `optimal_path_to_policy` | 428-491 | `(selections, returns, n_actions_per_asset=5, temperature=1.0) -> (T, 320)` | Convert to soft policy targets |
| `compute_imprint_loss` | 494-522 | `(agent_policy, optimal_policy, mask=None) -> scalar` | Soft cross-entropy loss |
| `compute_optimal_path` | 367-392 | `(returns, k=4, method='greedy', **kwargs) -> OptimalPath` | Unified interface |
| `compute_regret` | 399-421 | `(agent_pnl, returns, k=4) -> (regret_curve, optimal_pnl, total_regret)` | Evaluation utility |

### Critical Constants
```python
DEFAULT_MAX_PATH_LEN = 512  # Line 35 - prevents JIT recompilation (TASK_128)
```

### Critical Bug Fixes Embedded
- **Lines 67-74**: Padding to static `max_len` (TASK_128 - prevents XLA stall at epoch 60+)
- **Line 115**: `/ k_count` for equal-weighted portfolio (TASK_133 - was 4x leverage bug)

### Oracle Selection Logic (Lines 99-115)
```python
# At each step:
# 1. Look ahead `lookahead` steps
# 2. Select top-k assets by cumulative |returns|
# 3. Oracle goes long on positive, short on negative
# 4. PnL = sum(|returns| * selection) / k  (equal-weighted)
```

### Usage in Easy Mode
```python
# In generate_oracle_data.py
returns = cache.get_returns(episode_id)  # (T-1, n_assets)
optimal_path = greedy_lookahead_path(returns, k=4, lookahead=5)

# Get soft policy targets for training
policy_targets = optimal_path_to_policy(
    optimal_path.selections,
    returns,
    n_actions_per_asset=5,  # 5 × 64 = 320 actions
    temperature=1.0
)  # (T, 320)

# In easy_trainer.py loss function
loss = compute_imprint_loss(agent_policy, oracle_policy)
```

---

## 4. ENVIRONMENT FACTORY (`mcts_trader/training/env_factory.py`)

### Functions Needed by Easy Mode

| Function | Line | Signature | Purpose |
|----------|------|-----------|---------|
| `create_jax_env_and_prices` | 13-138 | `(synthetic_level, n_envs, n_assets=64, ...) -> (states, prices, config)` | Shared prices |
| `create_jax_env_unique_prices` | 141-264 | `(synthetic_level, n_envs, n_workers=8, ...) -> (states, prices, config)` | Unique per env |
| `create_jax_env_historical` | 267-373 | `(db_path, start_date, end_date, n_envs, ...) -> (states, prices, config)` | From database |

### Key Parameters (all factories)
```python
# Core
synthetic_level: int        # 0-14 curriculum level
n_envs: int                 # Parallel environments
n_assets: int = 64
n_steps: int = 2000         # Price timesteps

# Fees
fee_bps: float = 0.0
slippage_bps: float = 0.0

# Anti-memorization (TASK_049/058)
synthetic_noise: float = 0.0
temporal_jitter: int = 0
time_dilation: float = 0.0
phase_shift_prob: float = 0.0

# Market realism
probabilistic_fills: bool = False
borrow_rate_annual: float = 0.0
burning_money: bool = False
```

### Usage in Easy Mode
```python
# Synthetic curriculum training
states, prices, config = create_jax_env_and_prices(
    synthetic_level=3,
    n_envs=256,
    n_assets=64,
    n_steps=2000,
    synthetic_noise=0.1,  # Anti-memorization
)

# Historical validation
states, prices, config = create_jax_env_historical(
    db_path='trading_data.db',
    start_date='2024-01-01',
    end_date='2024-12-31',
    n_envs=128,
)
```

---

## 5. SYNTHETIC MARKET (`mcts_trader/data/synthetic_market.py`)

### Functions Needed by Easy Mode

| Function | Line | Signature | Purpose |
|----------|------|-----------|---------|
| `SyntheticMarket.__init__` | 11-67 | `(level, n_assets=64, n_steps=10000, seed=None, noise_level=0.0, ...)` | Initialize generator |
| `reset` | 133-136 | `() -> Dict` | Reset to start |
| `step` | 138-145 | `() -> Dict` | Advance one step |
| `get_current_data` | 147-165 | `() -> Dict[str, Dict]` | Get OHLCV at current step |
| `regenerate` | 115-131 | `(new_seed=None)` | New random data (anti-memorization) |

### Curriculum Levels (Lines 69-97)
```
Level 0-4:   Basic Mechanics (sine, lagged pairs, dips, trends, rotation)
Level 5-7:   Statistical Arbitrage (pairs, triangles, index arb)
Level 8-10:  Momentum & Reversal (momentum, mean-reversion, breakout)
Level 11-13: Market Microstructure (bid-ask bounce, order imbalance, market making)
Level 14:    Volatility Clustering (GARCH)
```

### Anti-Memorization Features (TASK_049/058)
```python
noise_level: float = 0.0     # 0-1, random walk + per-step noise
temporal_jitter: int = 0     # Max steps to shift observations
time_dilation: float = 0.0   # 0-0.3, stretch/compress patterns
phase_shift_prob: float = 0.0 # 0-0.5, mid-episode phase shifts
```

### Usage in Easy Mode
```python
# Direct generation for oracle dataset
synthetic = SyntheticMarket(
    level=5,
    n_assets=64,
    n_steps=10000,
    noise_level=0.1,
    seed=42,
)
synthetic.reset()

for step in range(synthetic.n_steps):
    data = synthetic.get_current_data()
    prices[step] = [data[f'SYN_{i:02d}']['close'] for i in range(64)]
    synthetic.step()
```

---

## 6. JAX TRADING ENVIRONMENT (`mcts_trader/env/jax_trading_env.py`)

### Types Needed by Easy Mode

| Type | Line | Purpose |
|------|------|---------|
| `EnvConfig` | 75-125 | Environment configuration dataclass |
| `TradingState` | 132-199 | Full environment state (immutable) |

### Functions Needed by Easy Mode

| Function | Line | Signature | Purpose |
|----------|------|-----------|---------|
| `reset` | ~250 | `(key, prices, config) -> TradingState` | Initialize single env |
| `jit_batch_reset` | ~400 | `(keys, prices, config) -> TradingState` | Batched initialization |
| `step` | ~500 | `(state, action, prices, config) -> (state, obs, reward, done)` | Single step |
| `jit_batch_step` | ~700 | `(states, actions, prices, config) -> (states, obs, rewards, dones)` | Batched step |

### EnvConfig Key Fields (Lines 75-125)
```python
n_assets: int = 64
initial_cash: float = 100_000.0
fee_bps: float = 0.0
slippage_bps: float = 0.0
max_steps: int = 400
window_size: int = 60
position_size_pct: float = 0.1
allow_short: bool = True

# TASK features
probabilistic_fills: bool = False   # TASK_047
borrow_rate_annual: float = 0.0     # TASK_055
episode_variance: int = 0           # TASK_053
burning_money: bool = False         # TASK_051
```

### TradingState Key Fields (Lines 132-199)
```python
# Scalar
cash: Array
current_step: Array
initial_portfolio_value: Array

# Position arrays (n_assets,)
pos_active: Array   # bool
pos_side: Array     # 0/1/2
pos_size: Array
pos_entry: Array

# Limit order arrays (n_assets,)
order_active: Array
order_side: Array
order_price: Array
order_ttl: Array

# Tracking
cumulative_return: Array
rng_key: PRNGKey
```

### Action Space
```
5 actions per asset × 64 assets = 320 total actions
action = asset_idx * 5 + action_type

action_type:
  0 = Sell (close long or open short)
  1 = Hold
  2 = Buy (open long or close short)
  3 = LimitSell
  4 = LimitBuy
```

### Usage in Easy Mode
```python
# Evaluation loop
from mcts_trader.env.jax_trading_env import EnvConfig, jit_batch_reset, jit_batch_step

config = EnvConfig(n_assets=64, max_steps=400)
states = jit_batch_reset(rng_keys, prices, config)

for step in range(config.max_steps):
    # Get model predictions
    obs = encode_observations(states, prices, ...)
    actions = model.predict(obs).argmax(axis=-1)

    # Environment step
    states, obs, rewards, dones = jit_batch_step(states, actions, prices, config)
```

---

## 7. SAFE CHECKPOINT (`mcts_trader/training/safe_checkpoint.py`)

### Functions Needed by Easy Mode

| Function | Line | Signature | Purpose |
|----------|------|-----------|---------|
| `SafeCheckpointManager.__init__` | ~50 | `(checkpoint_dir, metric_name, patience=50, ...)` | Initialize manager |
| `should_save` | ~150 | `(metric: float) -> bool` | Check if improvement |
| `save_checkpoint` | ~200 | `(data: dict, step: int, metric: float)` | Save with metadata |
| `should_rollback` | ~250 | `(metric: float) -> bool` | Check for degradation |
| `load_best_checkpoint` | ~300 | `() -> (data, step)` | Load best model |

### Key Parameters
```python
checkpoint_dir: str       # Where to save
metric_name: str          # 'avg_reward', 'oracle_agreement', etc.
patience: int = 50        # Steps before rollback
rollback_threshold: float = 0.5  # Relative degradation threshold
keep_best_n: int = 3      # Number of checkpoints to keep
rolling_window: int = 20  # For smoothed comparison
```

### Usage in Easy Mode
```python
manager = SafeCheckpointManager(
    checkpoint_dir='jax_ckpt_easy/phase1',
    metric_name='oracle_agreement',
    patience=50,
    keep_best_n=3,
)

for epoch in range(epochs):
    metrics = train_epoch(...)

    if manager.should_save(metrics['oracle_agreement']):
        manager.save_checkpoint(
            {'params': params, 'opt_state': opt_state},
            epoch,
            metrics['oracle_agreement']
        )

    if manager.should_rollback(metrics['oracle_agreement']):
        data, epoch = manager.load_best_checkpoint()
        params = data['params']
```

---

## 8. EVALUATOR (`mcts_trader/training/evaluator.py`)

### Functions Needed by Easy Mode

| Function | Line | Signature | Purpose |
|----------|------|-----------|---------|
| `run_generalization_eval` | ~50 | `(eval_level, search_fn, model_apply_fn, params, ...) -> dict` | Single level eval |
| `run_multi_level_eval` | ~120 | `(levels, search_fn, model_apply_fn, params, ...) -> dict` | Multi-level eval |

### Returned Metrics
```python
{
    'reward': float,        # Episode return (%)
    'alpha': float,         # Outperformance vs buy-and-hold
    'market_return': float, # Buy-and-hold baseline
    'agent_return': float,  # Agent's actual return
    'pct_hold': float,      # % hold actions
    'pct_buy': float,       # % buy actions
    'pct_sell': float,      # % sell actions
}
```

### Usage in Easy Mode
```python
# Evaluate on unseen curriculum levels
eval_metrics = run_generalization_eval(
    eval_level=4,  # Different from training
    search_fn=None,  # No MCTS in Phase 1
    model_apply_fn=model.apply,
    params=params,
    rng_key=rng_key,
    steps=200,
)
print(f"Alpha: {eval_metrics['alpha']:.2%}")
```

---

## 9. UTILITY: TRANSFORMS (`mcts_trader/utils/transforms.py`)

### Functions Needed by Easy Mode

| Function | Line | Signature | Purpose |
|----------|------|-----------|---------|
| `symlog` | ~10 | `(x, threshold=1.0) -> Array` | Compress large values |
| `inverse_symlog` | ~25 | `(y, threshold=1.0) -> Array` | Decompress |

### Formula
```python
# Forward: preserves sign, compresses magnitude
y = sign(x) * log(|x|/threshold + 1) * threshold

# Inverse
x = sign(y) * (exp(|y|/threshold) - 1) * threshold
```

### Usage in Easy Mode
```python
# For value head (if using categorical support)
from mcts_trader.utils.transforms import symlog, inverse_symlog

# Transform returns before discretizing
transformed_returns = symlog(oracle_returns)

# After softmax expectation, invert
predicted_returns = inverse_symlog(value_expectation)
```

---

## Complete Import Map for Easy Mode

```python
# easy_trainer.py imports

# Data & Environment
from mcts_trader.training.env_factory import (
    create_jax_env_and_prices,
    create_jax_env_historical,
)
from mcts_trader.data.synthetic_market import SyntheticMarket

# Encoding
from mcts_trader.env.jax_encoder import (
    EncoderConfig,
    jit_batch_encode_compressed,
    jit_batch_tile_compressed,
)

# Oracle
from mcts_trader.jax_agent.path_solver import (
    greedy_lookahead_path,
    optimal_path_to_policy,
    compute_imprint_loss,
)

# Storage
from mcts_trader.jax_agent.price_cache import (
    PriceCache,
    create_price_cache,
)

# Training Management
from mcts_trader.training.safe_checkpoint import SafeCheckpointManager
from mcts_trader.training.evaluator import run_generalization_eval

# Environment (for evaluation)
from mcts_trader.env.jax_trading_env import (
    EnvConfig,
    jit_batch_reset,
    jit_batch_step,
)

# Utils
from mcts_trader.utils.transforms import symlog, inverse_symlog
```

---

## What Easy Mode Does NOT Need

These components can be ignored/removed in Easy Mode:

| Component | Why Not Needed |
|-----------|----------------|
| `trainer.py` | Replaced by `easy_trainer.py` |
| `popart.py` | Oracle returns are stable |
| `redo.py` | No dormancy issues in supervised |
| `sac_entropy.py` | No exploration needed |
| `atc.py` | No temporal consistency loss |
| `cycle_balance.py` | No TRM cycle balancing |
| `adaptive_*.py` | No curriculum complexity |
| `mctx_search.py` (training) | MCTS only at inference |

---

## Validation Checklist

Before implementing Easy Mode, verify these work:

- [ ] `PriceCache.get_returns()` produces valid log returns
- [ ] `greedy_lookahead_path()` runs without XLA stall
- [ ] `optimal_path_to_policy()` produces (T, 320) shape
- [ ] `jit_batch_encode_compressed()` produces (B, 27, 64)
- [ ] `create_jax_env_and_prices()` returns valid states/prices
- [ ] `SafeCheckpointManager` saves/loads correctly

Run quick test:
```bash
python -c "
from mcts_trader.jax_agent.path_solver import greedy_lookahead_path
import jax.numpy as jnp
returns = jnp.ones((100, 64)) * 0.01
path = greedy_lookahead_path(returns, k=4)
print(f'Total alpha: {path.total_alpha:.4f}')
print(f'Trade count: {path.trade_count}')
"
```

---

## Summary: Lines of Code Being Reused

| Component | File | Lines | Status |
|-----------|------|-------|--------|
| PriceCache | price_cache.py | 336 | REUSE |
| JAX Encoder | jax_encoder.py | 782 | REUSE |
| Path Solver | path_solver.py | 523 | REUSE |
| Env Factory | env_factory.py | 441 | REUSE |
| SyntheticMarket | synthetic_market.py | 1040 | REUSE |
| JAX Env | jax_trading_env.py | 1672 | REUSE |
| SafeCheckpoint | safe_checkpoint.py | 456 | REUSE |
| Evaluator | evaluator.py | 173 | REUSE |
| Transforms | transforms.py | 44 | REUSE |
| **TOTAL** | | **5,467** | |

Easy Mode reuses **5,467 lines** of battle-tested infrastructure.
New code needed: ~400-500 lines.

