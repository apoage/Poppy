# Session Notes - 2026-01-11

## Market Realism Tasks Implementation

Implemented 7 market realism features to make training more realistic and prevent reward hacking.

### TASK_050: Fill at Market Price

**Problem**: Limit orders were filling at the limit price even when market price was better.

**Solution** (jax_trading_env.py:475-485):
```python
# Buy fills at minimum of limit and market (better for buyer)
buy_fill_price = jnp.minimum(state.order_price, current_prices)

# Sell fills at maximum of limit and market (better for seller)
sell_actual_price = jnp.maximum(state.order_price, current_prices)
```

### TASK_047: Probabilistic Fills

**Problem**: Limit orders filled deterministically when price touched limit, unrealistic.

**Solution**: Distance-based probability model (jax_trading_env.py:426-462):
- Marketable orders (distance <= 0): Always fill
- Non-marketable: `base_prob = 0.1 + distance * 20.0` (capped at 0.95)
- Age bonus: `+0.05` per step (capped at 0.30)
- Uses RNG for stochastic fills

**CLI**: `--probabilistic-fills`

### TASK_052: Random Fill Priority

**Problem**: Assets filled in index order, causing bias (lower indices get priority).

**Solution** (jax_trading_env.py:499-513):
```python
priority_order = jax.random.permutation(rng_key, n_assets)
inverse_order = jnp.argsort(priority_order)
shuffled_costs = buy_cost_if_opening[priority_order]
cumulative_shuffled = jnp.cumsum(shuffled_costs)
can_afford_shuffled = cumulative_shuffled <= available_cash
can_afford_cumulative = can_afford_shuffled[inverse_order]
```

### TASK_049: Synthetic Noise

**Problem**: Agent could memorize exact price patterns in synthetic data.

**Solution** (synthetic_market.py:_apply_noise):
- Random walk component: `cumsum(normal(0, 0.0005 * noise_level))`
- Per-step noise: `normal(0, 0.005 * noise_level)`
- Amplitude variation: sinusoidal `1 ± 0.2 * noise_level`

**CLI**: `--synthetic-noise 0.5`

### TASK_053: Episode Variance

**Problem**: Fixed episode length allows agent to exploit timing.

**Solution** (jax_trading_env.py):
- Added `max_episode_steps` to TradingState
- Reset calculates: `actual_max = max_steps + randint(-variance, variance+1)`
- Minimum enforced at 100 steps

**CLI**: `--episode-variance 30` (±30 steps)

### TASK_055: Borrow Cost

**Problem**: Short positions had no holding cost, unrealistic.

**Solution** (jax_trading_env.py:1127-1138):
```python
borrow_rate_step = config.borrow_rate_annual / 100800.0  # ~1.5min steps/year
is_short = new_state.pos_active & (new_state.pos_side == 2)
short_values = jnp.where(is_short, new_state.pos_size * current_prices, 0.0)
borrow_cost = jnp.sum(short_values) * borrow_rate_step
portfolio_change = new_portfolio_value - prev_portfolio_value - borrow_cost
```

**CLI**: `--borrow-rate 0.05` (5% annual)

### TASK_024: LR Schedule

**Problem**: Constant learning rate suboptimal for curriculum training.

**Solution** (schedules.py):
- Added `to_optax_schedule()` method to LRSchedule
- Supports: constant, cosine, linear with warmup
- Integrated into trainer.py optimizer creation

**CLI**: `--lr-schedule cosine --lr-warmup-epochs 20 --lr-min 1e-6`

## Testing & Validation

Created comprehensive test script (tools/testing/test_task_implementations.py):
- 32 unit tests across all 7 tasks
- System integration test
- All tests passing

15-epoch training run verified all features work together:
```bash
python train_jax_fast.py \
    --synthetic-level 0 --epochs 15 \
    --probabilistic-fills --borrow-rate 0.05 \
    --synthetic-noise 0.5 --episode-variance 20 \
    --lr-schedule cosine --lr-warmup-epochs 5
```

## TASK_056: Curriculum Realism Integration

Created task for integrating realism features into curriculum system.

**Design**: Hybrid YAML + auto-scaling approach
- Auto-scaling formulas provide defaults based on level
- YAML files can override specific levels
- CLI arguments always win

**Precedence**: CLI > YAML > Auto-defaults

**Auto-scaling by level**:
| Levels | Noise | Variance | Borrow | Prob Fills |
|--------|-------|----------|--------|------------|
| 0-4    | 0→0.3 | 0→28     | 0      | OFF        |
| 5-7    | 0.3→0.4 | 30→39  | 2-4%   | ON         |
| 8+     | 0.5   | 50       | 5-8%   | ON         |

## TASK_057: Codebase Cleanup (COMPLETED)

Major reorganization of project structure.

### New Structure
```
Trader/
├── train_jax_fast.py         # Main entry (root)
├── real_trade_jax.py         # Live trading (root)
├── config/curriculum/        # YAML configs
├── jax_ckpt_fast/           # All checkpoints (merged dojo_checkpoints)
├── logs/                     # Logs + reports + sweep_results + tournament_results
├── scripts/                  # Shell scripts
└── tools/
    ├── analysis/            # analyze_sweep.py, cross_validate.py
    ├── benchmarks/          # measure_*.py, profile_*.py
    ├── sweeps/              # *_sweep.py, *_tournament.py
    ├── orchestration/       # dojo_*.py, train_manager.py
    ├── debug/               # debug_*.py
    └── testing/             # test_*.py
```

### Files Moved
- 6 sweep/tournament scripts → tools/sweeps/
- 3 orchestration scripts → tools/orchestration/
- 4 benchmark scripts → tools/benchmarks/
- 4 test scripts → tools/testing/
- 2 analysis scripts → tools/analysis/
- 1 debug script → tools/debug/
- 3 shell scripts → scripts/
- reports/, sweep_results/, tournament_results/ → logs/
- dojo_checkpoints/* → jax_ckpt_fast/

### Path Updates
All moved scripts now use:
```python
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))
```

Shell scripts use:
```bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
```

### Deleted
- miner_capture.pcap (old debug artifact)

## Updated Training Command

Full realism training command:
```bash
python train_jax_fast.py \
    --synthetic-level 2 \
    --load-checkpoint jax_ckpt_fast/L1_kitchen_sink_v2 \
    --clear-buffer --reset-optimizer \
    --synthetic-steps 10000 \
    --epochs 500 \
    --batch-size 384 \
    --model-size small \
    --parallel-envs 50 \
    --jax-env --bf16 \
    --use-popart --popart-beta 0.0001 \
    --use-prediction-loss \
    --adaptive-entropy \
    --ds24-level 3 --ds24-intermediates-every 10 \
    --sims 24 --sims-warmup 300 --sims-min 4 \
    --entropy-coef 0.2 \
    --entropy-hold-threshold 0.55 \
    --entropy-boost-low 0.4 --entropy-boost-high 0.6 \
    --prediction-loss-weight 0.5 \
    --physics-warmup-steps 10000 \
    --probabilistic-fills \
    --borrow-rate 0.03 \
    --synthetic-noise 0.3 \
    --episode-variance 30 \
    --lr-schedule cosine \
    --lr-warmup-epochs 20 \
    --lr-min 1e-6 \
    --name L2_full_realism
```

## Files Modified

| File | Change |
|------|--------|
| `mcts_trader/env/jax_trading_env.py` | TASK_047, 050, 052, 053, 055 |
| `mcts_trader/data/synthetic_market.py` | TASK_049 noise |
| `mcts_trader/jax_agent/schedules.py` | TASK_024 optax integration |
| `mcts_trader/jax_agent/trainer.py` | TASK_024 LR schedule params |
| `mcts_trader/training/env_factory.py` | New params pass-through |
| `train_jax_fast.py` | CLI args for all new features |
| `CLAUDE.md` | New folder structure |
| `tools/**/*.py` | PROJECT_ROOT path handling |
| `scripts/*.sh` | PROJECT_ROOT path handling |

## Remaining Tasks

### Ready to Implement
- TASK_056: Curriculum config loader (design complete)

### Deferred
- TASK_022: TD(λ) returns
- TASK_025: Per-head gradient clipping
- TASK_042: ReDo dormant neurons
- TASK_043: ATC representation learning
