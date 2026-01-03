# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

**TRM+MuZero Trading Agent** - A reinforcement learning system for cryptocurrency trading that combines:
- **TRM (Tiny Recursive Models)**: Compact neural architecture (~7-36M params) with recursive reasoning
- **MuZero-style MCTS**: Monte Carlo Tree Search for planning via Google's `mctx` library
- **JAX/Flax**: High-performance GPU-accelerated training
- **VectorBT accounting**: Professional portfolio management with limit orders

**Core Innovation**: Uses TRM's recursive refinement (H×L cycles) instead of deep networks, fully GPU-accelerated MCTS via mctx, and symlog reward transformation for stable learning.

## Current Architecture (JAX-based)

### Key Components

```
mcts_trader/
├── jax_agent/                    # Core JAX/Flax implementation
│   ├── __init__.py              # Value support config (751 bins, [-750, 750])
│   ├── trm_flax.py              # TRM model (Encoder + Recursive Reasoning)
│   ├── muzero_model.py          # MuZero networks (Representation, Dynamics, Prediction)
│   ├── trainer.py               # Training loop with hybrid MSE+categorical loss
│   ├── replay_buffer.py         # JAX replay buffer with generation validation
│   ├── mctx_search.py           # GPU MCTS via mctx (Gumbel MuZero)
│   └── prediction/              # Neural Market Physics (NeuralOGCM-inspired)
│       ├── physics_core.py      # Differentiable market physics (α, β, σ, Γ)
│       ├── neural_corrector.py  # Neural residual corrector (STE)
│       └── market_ode.py        # Combined ODE predictor
│
├── env/
│   └── vectorbt_env.py          # Trading environment (1414 lines)
│                                 # - 5-action space: [Sell, Hold, Buy, LimitSell, LimitBuy]
│                                 # - Symlog rewards (replaces clipping)
│                                 # - Limit order support with TTL
│                                 # - TP/SL guardrails (optional)
│
├── data/
│   ├── synthetic_market.py      # Curriculum generators (Levels 0-4)
│   │                            # 0=sine, 1=lagged, 2=dip, 3=trends, 4=rotation
│   ├── history.py               # Historical data from SQLite
│   └── kraken_client.py         # Kraken API client
│
└── encoder/
    └── state.py                 # Multi-asset state encoder (27×64×64 tensor)

# Root training files
train_jax_fast.py               # Main training script (750+ lines)
dojo_dynamic_curriculum.py      # Curriculum manager with adaptive difficulty
dojo_manager.py                 # Parallel training orchestration
```

### Model Sizes

| Size | state_dim | Params | Use Case |
|------|-----------|--------|----------|
| small | 64 | ~7.3M | Fast iteration, curriculum |
| medium | 96 | ~15M | Balanced |
| large | 160 | ~36M | Maximum capacity |

### Action Space

5 actions per asset × 64 assets = **320 discrete actions**:
- `action % 5 == 0`: Sell (market)
- `action % 5 == 1`: Hold
- `action % 5 == 2`: Buy (market)
- `action % 5 == 3`: Limit Sell
- `action % 5 == 4`: Limit Buy

Formula: `action = asset_idx * 5 + action_type`

### Value Support

Categorical value representation with symlog rewards:
- **Range**: [-750, +750] with 751 bins
- **Resolution**: 2 units per bin
- **Transformation**: Symlog for per-step rewards, MC returns for training targets

## Running the System

### Training (Synthetic Curriculum)

```bash
# Activate environment
source trader_env/bin/activate

# Level 0: Sine waves (easiest)
python train_jax_fast.py \
    --synthetic-level 0 \
    --epochs 500 \
    --steps 400 \
    --batch-size 256 \
    --sims 32 \
    --model-size small \
    --name curriculum_L0

# With adaptive entropy (prevents policy collapse)
python train_jax_fast.py \
    --synthetic-level 0 \
    --epochs 500 \
    --adaptive-entropy \
    --entropy-hold-threshold 0.70 \
    --name curriculum_L0_adaptive

# Resume from checkpoint
python train_jax_fast.py \
    --synthetic-level 1 \
    --load-checkpoint jax_ckpt_fast/curriculum_L0 \
    --clear-buffer \
    --reset-optimizer \
    --name curriculum_L1
```

### Training (Real Data)

```bash
# From database with date range
python train_jax_fast.py \
    --db-path trading_data.db \
    --start-date 2024-12-01 \
    --end-date 2024-12-15 \
    --epochs 1000 \
    --name real_data_run
```

### Key CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--synthetic-level` | None | Synthetic curriculum (0-4) |
| `--model-size` | small | small/medium/large |
| `--sims` | 32 | MCTS simulations |
| `--batch-size` | 256 | Training batch size |
| `--epochs` | 1000 | Training epochs |
| `--steps` | 400 | Steps per episode |
| `--fee` | 0 | Trading fee (bps) |
| `--slippage` | 0 | Slippage (bps) |
| `--adaptive-entropy` | False | Boost entropy when HOLD% high |
| `--clear-buffer` | False | Clear buffer on checkpoint load |
| `--reset-optimizer` | False | Reset optimizer on checkpoint load |
| `--tp-sl-enabled` | False | Enable TP/SL guardrails |

## Key Technical Details

### TRM Architecture (trm_flax.py)

```
Input (27×64×64) → CNN Encoder → state (64/96/160-dim)
                                    ↓
                    ┌─── H_cycles (default 4) ───┐
                    │   ┌─ L_cycles (default 6) ─┐ │
                    │   │ Latent refinement      │ │
                    │   └────────────────────────┘ │
                    │   Output refinement          │
                    └──────────────────────────────┘
                                    ↓
                    Policy logits (320) + Value (751 bins)
```

**Weight Sharing Option** (`share_weights=True`): Paper-compliant shared weights across cycles (smaller model).

### Reward Transformation (symlog)

```python
def symlog(x, threshold=1.0):
    """Symmetric log: preserves sign, compresses magnitude"""
    return np.sign(x) * np.log1p(np.abs(x) / threshold) * threshold
```

Replaces asymmetric clipping that was causing negative bias in MC returns.

### Replay Buffer Generation Validation

The buffer tracks trajectory generations to detect wrap-around corruption:
```python
# Valid if: same generation OR (prev generation AND write position hasn't reached it)
gen_diff = current_generation - traj_gen
is_valid = (gen_diff == 0) | ((gen_diff == 1) & (current_position <= start_pos))
```

### MCTS via mctx

Uses Google's `mctx` library for GPU-accelerated search:
- **Gumbel MuZero** (default): Better exploration via Gumbel noise
- JIT-compiled for maximum performance
- ~32 simulations per decision (configurable)

## Curriculum System

### Synthetic Levels

| Level | Generator | Pattern | Tactics |
|-------|-----------|---------|---------|
| 0 | sine | Pure sine waves | Momentum detection |
| 1 | lagged | Leader-follower pairs | Correlation exploitation |
| 2 | dip | Random V-shaped crashes | Dip buying |
| 3 | noisy_trends | Random walk + drift | Trend following |
| 4 | rotation | Sector rotation cycles | Multi-asset timing |

### Dynamic Difficulty Phases

Each level has 5 difficulty phases (progressive fees/slippage):
1. Phase 1: 0 fee, 0 slip, threshold +2.5%
2. Phase 2: 5 bps fee, threshold +2.0%
3. Phase 3: 10 bps fee + 2 bps slip, threshold +1.5%
4. Phase 4: 16 bps fee + 5 bps slip, threshold +1.0%
5. Phase 5: 26 bps fee + 5 bps slip, threshold +0.5%

Run with: `python dojo_dynamic_curriculum.py --level 0 --model-size small`

### MCTS Simulation Warmup (TODO)

**Problem**: With high simulations (64), MCTS quickly converges to HOLD before value function learns.
The policy doesn't collapse (entropy stays ~4.0), but action distribution becomes HOLD-dominant (80%+).

**Insight**: Fewer sims = more stochastic policy = more exploration. Let value function learn first,
then add deep search to exploit learned values.

**Proposed curriculum** (not yet implemented):
```
--sims-warmup N  # Warm up simulations over N epochs

Epoch 1-N/4:    8 sims   (explore, learn basic patterns)
Epoch N/4-N/2:  16 sims  (refine value function)
Epoch N/2-3N/4: 32 sims  (improve policy)
Epoch 3N/4+:    64 sims  (full planning depth)
```

This is analogous to teaching chess: learn piece values first, then tactics, then deep calculation.

## Environment Details (vectorbt_env.py)

### Position Types
- **Long**: Buy asset, profit from price increase
- **Short** (margin mode): Borrow and sell, profit from price decrease

### Limit Orders
```python
@dataclass
class LimitOrder:
    asset_idx: int
    side: str  # 'buy' or 'sell'
    limit_price: float
    ttl: int = 100  # Time-to-live in steps
```

Orders persist until filled (price condition met) or expired (TTL exceeded).

### TP/SL Guardrails (optional)
- Stop Loss: Auto-close at -5%
- Take Profit: Auto-close at +10%

## Files & Directories

| Path | Purpose |
|------|---------|
| `train_jax_fast.py` | Main training script |
| `dojo_dynamic_curriculum.py` | Curriculum manager |
| `jax_ckpt_fast/` | Checkpoint storage |
| `logs/` | Training logs |
| `lore/` | Design documents, notes |
| `tasks/` | Feature task specifications |
| `trading_data.db` | SQLite database (22GB) |
| `config/` | Training configurations |

## Prediction Model (In Development)

Based on NeuralOGCM paper - physics-informed neural predictor:

```
Physics Core:        G_phys = α·returns - β·(price - MA) + σ·noise + Γ·correlation
Neural Corrector:    G_neural = STE(latent_state)  # Spatio-Temporal Evolution
Combined:            G = G_phys + G_neural
Next Price:          p_{t+1} = p_t + dt · G
```

Task files: `tasks/TASK_008_*.md` through `TASK_012_*.md`

## Development Guidelines

### Before Modifying Core Code
1. Read the relevant lore document in `lore/`
2. Check for related tasks in `tasks/`
3. Understand the JAX/Flax patterns used

### Common Pitfalls
- **Don't use `jax.lax.cond` inside `vmap`** - causes memory explosion. Use `jnp.where` instead.
- **Replay buffer logical vs physical indexing** - trajectories use logical indices, data uses physical.
- **Symlog is bidirectional** - rewards go symlog → MC returns → training targets
- **Model sizes affect batch limits** - large model may OOM with batch > 512

### Performance Tuning
- XLA autotuning level 4 enabled (`.xla_cache/`)
- JAX compilation cache (`.jax_cache/`)
- First compile is slow, subsequent runs use cache

## Key Dependencies

```
jax>=0.4.20
flax>=0.8.0
mctx>=0.0.5
optax>=0.1.7
orbax-checkpoint>=0.4.0
numpy>=1.24.0
gymnasium>=0.29.0
```

## Database Schema

```sql
-- Assets
assets(id, symbol, base_currency, quote_currency, ...)

-- OHLCV candles (1-minute)
ohlcv(asset_id, timestamp, open, high, low, close, volume, ...)

-- FX rates
fx_rates(base_currency, quote_currency, timestamp, rate)
```

## Monitoring Training

```bash
# Watch training log
tail -f logs/training_fast_<experiment_name>.log

# Check GPU utilization
nvidia-smi -l 1

# View recent checkpoints
ls -lt jax_ckpt_fast/
```

## Quick Reference

```bash
# New curriculum run from scratch
python train_jax_fast.py --synthetic-level 0 --epochs 500 --name L0

# Continue to next level
python train_jax_fast.py --synthetic-level 1 --load-checkpoint jax_ckpt_fast/L0 --clear-buffer --reset-optimizer --name L1

# Large model training
python train_jax_fast.py --synthetic-level 0 --model-size large --batch-size 256 --name L0_large

# Real data with fees
python train_jax_fast.py --db-path trading_data.db --fee 16 --slippage 5 --name real_fees
```

## Architecture Principles

### Codebase Organization

New features should be modular and self-contained:

```
mcts_trader/jax_agent/
├── trainer.py           # Main training loop only
├── muzero_model.py      # Model architecture
├── mctx_search.py       # MCTS search
├── replay_buffer.py     # Experience storage
├── schedules.py         # NEW: LR/sims schedules
├── popart.py            # NEW: Value normalization
├── returns.py           # NEW: TD(λ)/GAE computation
└── prediction/          # Market prediction (separate concern)
```

### train_jax_fast.py Role

The main script should be a **router**, not implementation:
- Parse CLI arguments
- Initialize components
- Wire components together
- Run main training loop
- Handle checkpointing

**Move implementation to modules:**
- Schedules → `schedules.py`
- Return computation → `returns.py`
- Value normalization → `popart.py`
- Gradient manipulation → Keep in `trainer.py` (tightly coupled)

### Default Parameters (2026-01-03)

| Parameter | Default | Rationale |
|-----------|---------|-----------|
| `--parallel-envs` | 20-50 | Batched transfers optimize at higher counts |
| `--adaptive-entropy` | True | Prevents HOLD collapse |
| `--value-loss-type` | mse | Huber available for high-variance |
| `--sims` | 32 | Balance exploration/exploitation |
| `--sims-warmup` | 100 | Gradual MCTS depth increase |
| `--sims-min` | 8 | Start with exploration |
| `--use-popart` | True | Stable value normalization |
| `--popart-beta` | 0.0001 | Slow adaptation for curriculum |

### Optimal Env Configurations (2026-01-03)

**Sweet spot: ~20,000 samples/epoch** (avoids system instability at 40k+)

| Use Case | Config | Samples | Est. Self-play |
|----------|--------|---------|----------------|
| Real-time data | `--parallel-envs 15 --steps 1440` | 21,600 | ~50s |
| Synthetic (diversity) | `--parallel-envs 50 --steps 400` | 20,000 | ~25s |
| Synthetic (fast) | `--parallel-envs 100 --steps 200` | 20,000 | ~25s |

**Tuned Hyperparameters (tournament-validated):**
```bash
--use-prediction-loss --prediction-loss-weight 0.2
--use-popart --popart-beta 0.0001
--sims-warmup 100 --sims-min 8
--adaptive-entropy --entropy-hold-threshold 0.70
--entropy-boost-low 0.05 --entropy-boost-high 0.2
--ds24-level 3 --ds24-intermediates-every 5
--pred-head-size large
```

### Training Optimization Tasks

See `tasks/` for implementation plans:
- **TASK_020**: MCTS sims warmup (gradual increase) - COMPLETE
- **TASK_021**: PopArt normalization (curriculum-safe) - COMPLETE
- **TASK_022**: TD(λ) returns (variance reduction)
- **TASK_024**: LR schedule (warmup + cosine)
- **TASK_025**: Per-head gradient clipping
- **TASK_027**: JAX-vectorized environment - COMPLETE
- **TASK_028**: Streaming data pipeline - COMPLETE (pre-alloc + fp16 + batched transfers)
- **TASK_029**: 100+ environment scaling - COMPLETE
