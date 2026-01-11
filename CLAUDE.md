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

Categorical value representation for PopArt-normalized values:
- **Range**: [-10, +10] with 401 bins (TASK_033 fix)
- **Resolution**: 0.05 units per bin
- **Bins > Actions**: 401 bins > 320 actions (ensures value differentiation)
- **PopArt normalization**: Targets normalized to mean=0, std=1
- **Transformation**: Symlog for per-step rewards, PopArt for training targets

**IMPORTANT**: Value support must match PopArt scale. The original [-750, +750] caused 99.5% information loss!

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

### Synthetic Levels (0-14)

**Basic Patterns (0-4)**
| Level | Generator | Pattern | Tactics |
|-------|-----------|---------|---------|
| 0 | sine | Pure sine waves | Momentum detection |
| 1 | lagged | Leader-follower pairs | Correlation exploitation |
| 2 | dip | Random V-shaped crashes | Dip buying |
| 3 | noisy_trends | Random walk + drift | Trend following |
| 4 | rotation | Sector rotation cycles | Multi-asset timing |

**Arbitrage Patterns (5-7)**
| Level | Generator | Pattern | Tactics |
|-------|-----------|---------|---------|
| 5 | pairs_trading | Cointegrated pairs | Spread trading |
| 6 | triangle_arb | 3-asset price loops | Triangular arbitrage |
| 7 | index_arb | Index vs components | Index arbitrage |

**Strategy Patterns (8-10)**
| Level | Generator | Pattern | Tactics |
|-------|-----------|---------|---------|
| 8 | momentum | Trending with noise | Trend following |
| 9 | mean_reversion | Oscillating around mean | Fade extremes |
| 10 | breakout | Range then breakout | Breakout detection |

**Microstructure Patterns (11-13)**
| Level | Generator | Pattern | Tactics |
|-------|-----------|---------|---------|
| 11 | bid_ask_bounce | Spread dynamics | Market microstructure |
| 12 | order_imbalance | Volume pressure | Order flow |
| 13 | market_making | Two-sided liquidity | Spread capture |

**Advanced (14)**
| Level | Generator | Pattern | Tactics |
|-------|-----------|---------|---------|
| 14 | volatility_clustering | GARCH-like vol | Volatility timing |

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

```
Trader/
├── CLAUDE.md                    # Project documentation
├── README.md                    # Project readme
├── requirements.txt             # Dependencies
├── trading_data.db              # SQLite database (22GB)
├── train_jax_fast.py            # Main training entry point
├── real_trade_jax.py            # Live trading entry point
├── trader_env/                  # Virtual environment
│
├── config/
│   └── curriculum/              # Curriculum YAML configs (TASK_056)
│
├── datasets/                    # Data files
│
├── docs/
│   └── AI_bleed research/       # Dual collapse research
│
├── jax_ckpt_fast/               # All checkpoints (unified)
│
├── logs/
│   ├── training_*.csv/log       # Training logs
│   ├── reports/                 # Analysis reports
│   ├── sweep_results/           # Sweep outputs
│   └── tournament_results/      # Tournament outputs
│
├── lore/                        # Design documents, notes
│
├── mcts_trader/                 # Main package
│   ├── data/                    # Data loaders, synthetic market
│   ├── encoder/                 # State encoder
│   ├── env/                     # Trading environments (JAX + VectorBT)
│   ├── jax_agent/               # Model, trainer, MCTS, schedules
│   └── training/                # Env factory, curriculum config
│
├── scripts/                     # Shell scripts
│   ├── run_training.sh          # CUDA wrapper
│   ├── START_TRAINING.sh        # Phase 1 training
│   └── START_TRAINING_PHASE2.sh # Phase 2 fine-tuning
│
├── tasks/                       # Task specifications
│
├── tests/                       # Unit tests (pytest)
│
└── tools/                       # Analysis & utility scripts
    ├── analysis/                # analyze_sweep.py, cross_validate.py
    ├── benchmarks/              # measure_*.py, profile_*.py
    ├── sweeps/                  # *_sweep.py, *_tournament.py
    ├── orchestration/           # dojo_*.py, train_manager.py
    ├── debug/                   # debug_*.py
    └── testing/                 # test_*.py, test_*.sh
```

| Key Paths | Purpose |
|-----------|---------|
| `train_jax_fast.py` | Main training script |
| `tools/sweeps/` | Warmup/stability sweeps and tournaments |
| `tools/orchestration/` | Curriculum manager, dojo scripts |
| `jax_ckpt_fast/` | All checkpoints (including dojo) |
| `logs/` | Training logs and sweep results |
| `config/curriculum/` | Per-level YAML configs |

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
- **Division guards required** - Always use `jnp.maximum(divisor, 1e-10)` in JAX code; `jnp.where` evaluates both branches
- **Don't JIT `flush_gpu_buffers`** - it captures mutable numpy arrays from closure
- **Per-instance RNG for parallel envs** - use `np.random.default_rng()` not `np.random.seed()`

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

**Tuned Hyperparameters (production config 2026-01-06):**
```bash
python train_jax_fast.py \
    --synthetic-level 0 --synthetic-steps 10000 \
    --epochs 500 --batch-size 384 --model-size small \
    --steps 400 --parallel-envs 50 --jax-env --bf16 \
    --use-popart --popart-beta 0.0001 \
    --sims 32 --sims-warmup 100 --sims-min 8 \
    --use-prediction-loss --prediction-loss-weight 0.2 \
    --pred-head-size large \
    --ds24-level 3 --ds24-intermediates-every 5 \
    --adaptive-entropy --entropy-hold-threshold 0.85 \
    --entropy-boost-low 0.05 --entropy-boost-high 0.3 \
    --regenerate-data-every 50 \
    --fee 0 --slippage 0
```

**Best Stability Config (warmup_sweep.py 2026-01-06):**

Sweep tested 12 configurations. Best result: "kitchen_sink" (stability score 0.7354)
- Delayed HOLD>70% until epoch 27 (vs epoch 8 for baseline)
- Entropy never dropped below 3.5 during 50-epoch test
- Max reward 39.96, final reward 32.60

```bash
# Kitchen Sink - Most stable configuration found
--sims-warmup 300 --sims-min 4 --sims-max 24 \
--entropy-coef 0.2 \
--entropy-hold-threshold 0.55 \
--entropy-boost-low 0.4 --entropy-boost-high 0.6 \
--prediction-loss-weight 0.5 \
--physics-warmup-steps 10000
```

| Rank | Config | Score | MaxRwd | HOLD>70% at |
|------|--------|-------|--------|-------------|
| 1 | kitchen_sink | 0.735 | 39.96 | epoch 27 |
| 2 | very_high_entropy_05 | 0.725 | 38.27 | epoch 27 |
| 3 | combined_v3_aggressive | 0.695 | 38.87 | epoch 15 |
| 4 | combined_v2 | 0.690 | 41.19 | epoch 8 |
| 5 | early_adaptive_060 | 0.675 | 38.71 | epoch 8 |

**Key insights from sweep:**
- Lower sims (4-24) + longer warmup (300) delays collapse
- Higher entropy coefficient (0.2) + aggressive boosts (0.4-0.6) help
- Early adaptive trigger (0.55 vs 0.85) catches collapse sooner
- Higher prediction loss weight (0.5) may help prediction head

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

## Dual Collapse Problem (2026-01-06 Research)

### Problem Statement

Training exhibits **coupled policy-prediction collapse**:
1. **Policy collapse**: Entropy bleeds → HOLD dominates (85%+)
2. **Prediction collapse**: After warmup, loss → 0.01 (trivial "no change" solution)

These are coupled via feedback loop:
```
MCTS produces sharp π (HOLD-heavy)
    ↓
Policy trained to match → π_network becomes HOLD-heavy
    ↓
Data distribution shifts to HOLD-only
    ↓
Prediction head sees no dynamics → learns "no change"
    ↓
Value estimates become uninformative
    ↓
MCTS produces even sharper HOLD targets
```

### Root Causes Identified

| Cause | Effect |
|-------|--------|
| **MCTS sharpening** | More sims → sharper policy → faster collapse |
| **HOLD appears safe** | R(HOLD)=0 vs R(TRADE)≈-fees → HOLD bias |
| **Martingale trap** | Predicting "no change" is strong local optimum |
| **Dormant neurons** | HOLD-only data kills network capacity |
| **Representation collapse** | All states → similar embeddings |

### Interventions (Task Files Created)

| Task | Intervention | Target |
|------|--------------|--------|
| **TASK_036** | Warmup configuration sweep | Find stable hyperparams |
| **TASK_037** | ε-greedy MCTS: `π_train = (1-ε)*π_mcts + ε*uniform` | Policy collapse |
| **TASK_038** | Differential reward: `r = r_agent - r_market` | HOLD incentive |
| **TASK_039** | Higher Dirichlet noise (α=0.5, ε=0.5) | MCTS exploration |
| **TASK_040** | SAC-style adaptive entropy (learnable α) | Entropy control |
| **TASK_041** | Flow matching (predict velocity not state) | Prediction collapse |
| **TASK_042** | ReDo (recycle dormant neurons) | Network capacity |
| **TASK_043** | Augmented Temporal Contrast (ATC) | Representation collapse |

### Priority Order

1. **TASK_041** (Flow Matching) - Directly prevents prediction identity solution
2. **TASK_040** (SAC Entropy) - Proactive entropy control vs reactive
3. **TASK_037** (ε-greedy) - Breaks MCTS→policy feedback loop
4. **TASK_038** (Differential Reward) - Removes HOLD=safe incentive
5. **TASK_039** (Dirichlet) - Quick win, synergistic with above
6. **TASK_042** (ReDo) - Addresses symptom (dead neurons)
7. **TASK_043** (ATC) - Representation health

### Implemented Modules (2026-01-06)

All modules are in `mcts_trader/jax_agent/`:

| Module | File | CLI Flags |
|--------|------|-----------|
| ε-greedy MCTS | `exploration.py` | `--epsilon-greedy 0.2` |
| Dirichlet Noise | `exploration.py` | `--dirichlet-aggressive` |
| SAC Entropy | `sac_entropy.py` | `--sac-entropy --sac-target-entropy-ratio 0.5` |
| Flow Matching | `flow_matching.py` | `--flow-matching --flow-velocity-weight 0.8` |
| Differential Reward | `differential_reward.py` | `--differential-reward --differential-weight 1.0` |

**Example usage combining interventions:**
```bash
python train_jax_fast.py \
    --synthetic-level 0 \
    --epsilon-greedy 0.2 \
    --dirichlet-aggressive \
    --sac-entropy \
    --flow-matching \
    --differential-reward
```

### Research Sources

Research documents in `/home/lukas/Trader/docs/AI_bleed research/`:
- `big gemini.txt` - Comprehensive deep research
- `kimi.txt` - Collapse mechanism analysis
- `cl_cli.txt` - Causal diagram
- `compass_artifact_*.md` - Breaking dual collapse

### Known Limitations (2026-01-04 Audit)

Documented issues that are acceptable for now but may need future attention:

| Issue | File | Status | Notes |
|-------|------|--------|-------|
| **MCTS terminal discount** | `mctx_search.py:104` | Accepted | Discount always γ=0.997, no terminal handling. Acceptable for fixed-length episodes (γ^400≈0.30 natural decay). Future: add terminal prediction head to dynamics model. |
| **Target network unused** | `trainer.py:683-694` | Dead code | `target_params` updated but never used in loss. Decision needed: implement TD-style value learning or remove. |
| **RNG split inefficiency** | `muzero_model.py:344` | Optimization | RNG splits inside loop prevent parallelization. Future: pre-split all keys outside loop. |

### Code Quality Notes (2026-01-04 Audit)

**Confirmed Working Correctly** (investigated, not bugs):
- **PopArt weight correction**: Correction before gradient is intentional - provides smooth starting point, then gradients adapt
- **Replay buffer `<=` check**: Correct because `position` is "next write" not "last wrote"
- **Symlog + categorical**: Not double transformation - symlog (compression) and binning (discretization) serve different purposes
- **Adaptive calibration**: Fully implemented at `train_jax_fast.py:2055-2088`

**Fixed Bugs (2026-01-04)**:
- Scheduled sampling scalar → per-batch-element random (`muzero_model.py:349`)
- Division by zero guards in position sizing (`jax_trading_env.py:508,532,627`)
- Per-instance RNG for parallel reproducibility (`vectorbt_env.py:342,362-364`)

## Session Notes (2026-01-06)

### Work Completed

1. **Stability Intervention Modules Created** - All modular, minimal main program growth:
   - `mcts_trader/jax_agent/exploration.py` - ε-greedy MCTS + Dirichlet helpers
   - `mcts_trader/jax_agent/sac_entropy.py` - SAC-style learnable entropy coefficient
   - `mcts_trader/jax_agent/flow_matching.py` - Velocity prediction loss
   - `mcts_trader/jax_agent/differential_reward.py` - Opportunity cost rewards

2. **CLI Flags Added to train_jax_fast.py**:
   ```
   --epsilon-greedy 0.2        # Mix uniform into MCTS policy
   --dirichlet-aggressive      # α=0.5, ε=0.5 (vs default 0.3, 0.25)
   --sac-entropy               # Enable learnable entropy coefficient
   --sac-target-entropy-ratio  # Target as % of max entropy (default: 0.5)
   --flow-matching             # Predict velocity not state
   --flow-velocity-weight      # Weight for velocity loss (default: 0.8)
   --differential-reward       # Subtract market average from rewards
   --differential-weight       # Weight for differential component
   ```

3. **Sweep/Tournament Scripts Created**:
   - `stability_sweep.py` - Tests interventions individually and in combinations
   - `stability_tournament.py` - Multi-run statistical tournament with confidence intervals

   Usage:
   ```bash
   # Quick test (50 epochs, ~2h)
   python stability_sweep.py --mode quick

   # Full sweep (150 epochs, ~8h)
   python stability_sweep.py --mode full

   # Overnight (300 epochs, ~16h)
   python stability_sweep.py --mode overnight

   # Tournament (5 runs per config)
   python stability_tournament.py --mode full --runs-per-config 5
   ```

   Both scripts support:
   - `--parallel-envs N` (default: 50)
   - `--no-jax-env` (use numpy env instead)
   - Mode-based timeouts: quick=30min, full=90min, overnight=180min

4. **Task Files Updated**:
   - TASK_037 (ε-greedy) → IMPLEMENTED
   - TASK_039 (Dirichlet) → IMPLEMENTED
   - TASK_040 (SAC Entropy) → IMPLEMENTED
   - TASK_041 (Flow Matching) → IMPLEMENTED
   - TASK_038 (Differential Reward) → IMPLEMENTED

5. **Bug Fixes**:
   - Fixed syntax error in `stability_sweep.py:322` - missing `)` in `cmd.extend([...])`
   - Fixed syntax error in `stability_tournament.py:207` - same issue

### Next Steps

1. Run overnight stability sweep to find best intervention combinations
2. Integrate winning interventions into main training loop
3. Implement TASK_042 (ReDo) and TASK_043 (ATC) if needed

### Sweep Configurations Being Tested

| Config | Interventions |
|--------|---------------|
| baseline_kitchen_sink | Best warmup config, no new interventions |
| epsilon_010/020/030 | ε-greedy at 10%, 20%, 30% |
| dirichlet_aggressive | α=0.5, ε=0.5 |
| sac_entropy_040/050 | SAC entropy at 40%, 50% target |
| flow_matching_080/100 | Flow matching at 80%, 100% velocity weight |
| differential_reward_050/100 | Differential reward at 50%, 100% |
| combo_exploration | ε-greedy 20% + aggressive Dirichlet |
| combo_entropy_control | ε-greedy 20% + SAC entropy |
| combo_prediction_reward | Flow matching + differential reward |
| kitchen_sink_v2 | All interventions enabled |

## Session Notes (2026-01-11)

### Work Completed

1. **Training Enhancement Tasks Implemented** (commit `222a3ce`):
   - **TASK_022**: TD(λ)/GAE returns - `mcts_trader/jax_agent/returns.py`
   - **TASK_025**: Per-head gradient clipping - in `trainer.py`
   - **TASK_042**: ReDo dormancy framework - `mcts_trader/jax_agent/redo.py`
   - **TASK_043**: ATC contrastive loss - `mcts_trader/jax_agent/atc.py`

2. **New CLI Flags**:
   ```
   # TASK_022: GAE Returns
   --use-gae                    # Enable GAE instead of MC returns
   --gae-lambda 0.95            # GAE lambda parameter

   # TASK_025: Per-head Gradient Clipping
   --per-head-clipping          # Enable per-head clipping
   --clip-grad-value 1.0        # Value head max norm
   --clip-grad-policy 1.0       # Policy head max norm
   --clip-grad-dynamics 1.0     # Dynamics head max norm
   --clip-grad-encoder 1.0      # Encoder max norm

   # TASK_042: ReDo
   --redo                       # Enable dormant neuron recycling
   --redo-interval 10           # Epochs between checks
   --redo-threshold 0.01        # CV threshold for dormancy

   # TASK_043: ATC
   --atc-loss                   # Enable ATC contrastive loss
   --atc-weight 0.1             # Weight for ATC loss
   --atc-k-step 5               # Temporal distance for positives
   --atc-temperature 0.1        # Contrastive temperature
   ```

3. **Partial Implementations**:
   - TASK_042 and TASK_043 require model changes to expose intermediate activations
   - Core loss functions and monitoring frameworks are complete

### Testing Results

- [x] Unit test: GAE returns computation - PASSED
- [x] Unit test: Per-head gradient clipping - PASSED (bug fixed in commit `8f28b4a`)
- [x] Unit test: ReDo dormancy monitor - PASSED
- [x] Unit test: ATC contrastive loss - PASSED
- [x] Integration test: Short training run - PASSED
- [x] Validate kitchen_sink_v2 config - PASSED (10 epochs completed)

**Bug Fixed**: Per-head gradient clipping was returning wrong pytree structure.
Used `jax.tree_util.tree_map_with_path` instead of `unflatten_dict` to preserve exact FrozenDict structure.

### Training Results (L2_full_realism)

Impressive results shared:
- Alpha: +663% at epoch 291 (400% MA50)
- HOLD: 65% (healthy, not collapsed)
- Entropy: 5.26 (healthy)
- Note: This is 400% per 400-minute block, not annual
