# TASK_666: Easy Mode Implementation Plan

**Date**: 2026-02-03
**Status**: PLANNING → IMPLEMENTATION
**Goal**: Working trading bot via Expert Iteration (not full MuZero)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     EASY MODE ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Phase 1: SUPERVISED LEARNING (The Cheat)                       │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐                   │
│  │  Oracle  │───►│ Policy   │───►│  Train   │                   │
│  │(path_    │    │ Targets  │    │  CE+MSE  │                   │
│  │ solver)  │    │          │    │          │                   │
│  └──────────┘    └──────────┘    └──────────┘                   │
│                                                                  │
│  Phase 2: DYNAMICS MODEL (The Dream) - Optional                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐                   │
│  │  Frozen  │───►│ Dynamics │───►│  Train   │                   │
│  │ Encoder  │    │  Model   │    │  MSE     │                   │
│  └──────────┘    └──────────┘    └──────────┘                   │
│                                                                  │
│  Phase 3: INFERENCE MCTS (The Thought)                          │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐                   │
│  │ Policy + │───►│  mctx    │───►│  Action  │                   │
│  │ Dynamics │    │  search  │    │          │                   │
│  └──────────┘    └──────────┘    └──────────┘                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Codebase Reference Map

### Components to REUSE (no modification needed)

| Component | Location | Purpose |
|-----------|----------|---------|
| Oracle | `mcts_trader/jax_agent/path_solver.py:39-154` | `greedy_lookahead_path()` - hindsight optimal |
| Policy targets | `mcts_trader/jax_agent/path_solver.py:429-491` | `optimal_path_to_policy()` - 320-action targets |
| Imprint loss | `mcts_trader/jax_agent/path_solver.py:494-523` | `compute_imprint_loss()` - soft CE |
| TRM Encoder | `mcts_trader/jax_agent/trm_flax.py:219-248` | `TinyRecursiveTradingModel` |
| Observation encoder | `mcts_trader/env/jax_encoder.py:68-100` | `encode_observation()` → (27, n, n) |
| Compressed encoder | `mcts_trader/env/jax_encoder.py` | `jit_batch_encode_compressed()` → (27, n) |
| Expert buffer | `mcts_trader/jax_agent/expert_buffer.py:43-134` | Index-based storage with oracle actions |
| Fast generator | `mcts_trader/jax_agent/fast_generator.py` | Synthetic price generation L0-L14 |
| Benchmark periods | `config/benchmark_periods.json` | 6 test periods (bull, bear, sideways, etc.) |

### Components to SIMPLIFY (create new versions)

| Current | New | Changes |
|---------|-----|---------|
| `trainer.py` (87KB, 2400 lines) | `easy_trainer.py` (~300 lines) | Remove MuZero loss, PopArt, MCTS training |
| `train_jax_fast.py` (complex) | `train_easy.py` (~200 lines) | Just supervised loop |
| DS24 dashboard | DSe dashboard | Simplified metrics for Easy Mode |

### Components to REMOVE from Easy Mode

| Component | Why |
|-----------|-----|
| `popart.py` | Oracle returns are stable, no normalization needed |
| `redo.py` | Dormancy not an issue with supervised learning |
| `sac_entropy.py` | No exploration needed |
| `mctx_search.py` (training) | MCTS only at inference time |
| `adaptive_*.py` | No curriculum complexity |
| `cycle_balance.py` | No TRM cycle balancing needed |
| `atc.py` | No temporal consistency loss |

---

## Phase 0: Support Tools (DSe Dashboard)

### 0.1 Create DSe from DS24

**Source**: `tools/DS24/`
**Target**: `tools/DSe/`

DSe tracks simplified metrics for Easy Mode:

```
Easy Mode Metrics (DSe)
├── Training Progress
│   ├── epoch, step
│   ├── policy_loss (CE on oracle actions)
│   ├── value_loss (MSE on oracle returns)
│   └── total_loss
│
├── Oracle Agreement
│   ├── oracle_agreement_pct (target: >85%)
│   ├── top1_accuracy
│   ├── top5_accuracy
│   └── action_distribution
│
├── Value Prediction
│   ├── value_mse
│   ├── value_correlation
│   └── return_prediction_error
│
└── Evaluation
    ├── eval_reward (on benchmark periods)
    ├── eval_alpha (vs buy-and-hold)
    └── eval_sharpe
```

**Files to create**:
- `tools/DSe/metrics_logger.py` - Simplified from DS24 version
- `tools/DSe/server.py` - Simplified dashboard server
- `tools/DSe/index.html` - New dashboard UI
- `tools/DSe/dashboard.js` - New JS for Easy Mode charts
- `tools/DSe/styles.css` - Copy from DS24
- `tools/DSe/start_dse.sh` - Launch script

### 0.2 DSe Metrics Logger API

```python
# tools/DSe/metrics_logger.py

class DSeMetricsLogger:
    """Simplified logger for Easy Mode training."""

    COLUMNS = [
        # Core training
        'epoch', 'step', 'policy_loss', 'value_loss', 'total_loss',
        'learning_rate', 'grad_norm',

        # Oracle agreement
        'oracle_agreement', 'top1_accuracy', 'top5_accuracy',

        # Value prediction
        'value_mse', 'value_correlation',

        # Evaluation (when run)
        'eval_reward', 'eval_alpha', 'eval_sharpe',
        'eval_period',  # which benchmark period
    ]

    def log_training_step(self, metrics: dict): ...
    def log_evaluation(self, metrics: dict): ...
```

---

## Phase 1: Easy Trainer Implementation

### 1.1 EasyConfig

**File**: `easy_config.py`

```python
@dataclass
class EasyConfig:
    # Model (reuse existing)
    state_dim: int = 64
    n_assets: int = 64
    action_space_size: int = 320  # 5 × 64
    H_cycles: int = 4
    L_cycles: int = 6

    # Training
    epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 1000
    grad_clip: float = 1.0

    # Loss weights
    policy_weight: float = 1.0
    value_weight: float = 0.5

    # Oracle (from path_solver.py)
    oracle_k: int = 4
    oracle_lookahead: int = 5
    transaction_cost: float = 0.001

    # Data
    window_size: int = 60
    episode_length: int = 400

    # Success criteria
    target_oracle_agreement: float = 0.85
```

### 1.2 EasyTrainer

**File**: `easy_trainer.py`

Key simplifications from `trainer.py`:

```python
class EasyTrainer:
    """Supervised-only trainer for Easy Mode."""

    def __init__(self, config: EasyConfig):
        # Create model (reuse MuZeroNetworks but simpler init)
        self.model = MuZeroNetworks(
            state_dim=config.state_dim,
            action_space_size=config.action_space_size,
            H_cycles=config.H_cycles,
            L_cycles=config.L_cycles,
            use_prediction_head=False,  # No dynamics during Phase 1
        )

        # Simple optimizer (no PopArt, no complex scheduling)
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(config.grad_clip),
            optax.adamw(config.learning_rate, weight_decay=config.weight_decay),
        )

    def loss_fn(self, params, batch):
        """
        Simple supervised loss.

        From trainer.py:36-112, we reuse:
        - optax.softmax_cross_entropy_with_integer_labels for policy
        - MSE for value

        We REMOVE:
        - MuZero TD loss (trainer.py ~line 400-500)
        - PopArt normalization (trainer.py ~line 200-250)
        - Dynamics/reward prediction loss
        - ATC loss
        - Cycle balance loss
        """
        obs = batch['observations']           # (B, 27, 64, 64) or compressed
        oracle_actions = batch['actions']     # (B,) int
        oracle_returns = batch['returns']     # (B,) float

        # Forward pass
        latent = self.model.representation(obs)
        policy_logits, value_logits = self.model.prediction(latent)

        # Policy loss: Cross-entropy with oracle actions
        # Reuse from trainer.py:107
        policy_loss = optax.softmax_cross_entropy_with_integer_labels(
            policy_logits, oracle_actions
        ).mean()

        # Value loss: MSE with oracle returns
        # Simplified from trainer.py categorical value handling
        # Convert value logits to scalar via softmax expectation
        value_pred = self._value_logits_to_scalar(value_logits)
        value_loss = jnp.mean((value_pred - oracle_returns) ** 2)

        total_loss = (
            config.policy_weight * policy_loss +
            config.value_weight * value_loss
        )

        # Metrics
        metrics = {
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'total_loss': total_loss,
            'oracle_agreement': self._compute_agreement(policy_logits, oracle_actions),
        }

        return total_loss, metrics
```

### 1.3 Data Pipeline

**Reuse existing components**:

1. **Price generation**: `fast_generator.py` for synthetic L0-L14
2. **Oracle computation**: `path_solver.py:greedy_lookahead_path()`
3. **Policy targets**: `path_solver.py:optimal_path_to_policy()`
4. **Storage**: `expert_buffer.py:UnifiedIndexBuffer`
5. **Encoding**: `jax_encoder.py:jit_batch_encode_compressed()`

**New file**: `generate_oracle_data.py`

```python
"""
Bulk oracle data generation for Easy Mode.

Uses:
- mcts_trader/jax_agent/fast_generator.py for prices
- mcts_trader/jax_agent/path_solver.py for oracle
- mcts_trader/jax_agent/expert_buffer.py for storage
"""

def generate_episode(level: int, steps: int, config):
    """Generate single episode with oracle labels."""
    # 1. Generate prices
    prices = generate_synthetic_prices(level, steps)

    # 2. Compute oracle path
    returns = compute_log_returns(prices)
    optimal_path = greedy_lookahead_path(
        returns, k=config.oracle_k, lookahead=config.oracle_lookahead
    )

    # 3. Convert to policy targets
    policy_targets = optimal_path_to_policy(
        optimal_path.selections, returns
    )

    # 4. Compute returns for value targets
    value_targets = optimal_path.cumulative_pnl

    return {
        'prices': prices,
        'oracle_actions': policy_targets.argmax(axis=-1),  # Hard labels
        'oracle_returns': value_targets,
        'oracle_selections': optimal_path.selections,
    }

def generate_dataset(config, output_path):
    """Generate full oracle dataset."""
    # Mix of synthetic levels and historical data
    episodes = []

    # Synthetic L0-L14 curriculum
    for level in range(15):
        for _ in range(config.episodes_per_level):
            episodes.append(generate_episode(level, config.steps, config))

    # Save to npz
    save_oracle_dataset(episodes, output_path)
```

---

## Phase 2: Training Entry Point

### 2.1 train_easy.py

**File**: `train_easy.py`

```python
"""
Easy Mode Training - Expert Iteration Phase 1

Usage:
    python train_easy.py --epochs 100 --batch-size 256
    python train_easy.py --load-checkpoint jax_ckpt_easy/phase1_best.ckpt

Dashboard:
    ./tools/DSe/start_dse.sh  # Then open localhost:8051
"""

def main():
    config = parse_easy_args()

    # 1. Setup
    trainer = EasyTrainer(config)
    logger = DSeMetricsLogger(f'logs/easy_{config.name}.csv')

    # 2. Load or generate data
    if config.oracle_data_path:
        dataset = load_oracle_dataset(config.oracle_data_path)
    else:
        dataset = generate_oracle_dataset(config)

    # 3. Training loop
    for epoch in range(config.epochs):
        epoch_metrics = {'policy_loss': 0, 'value_loss': 0, 'oracle_agreement': 0}

        for batch in dataset.batches(config.batch_size):
            metrics = trainer.train_step(batch)
            accumulate_metrics(epoch_metrics, metrics)

        # Log to DSe
        logger.log_training_step(epoch_metrics)

        # Evaluate on benchmark periods
        if epoch % config.eval_every == 0:
            eval_metrics = run_easy_eval(trainer, config)
            logger.log_evaluation(eval_metrics)

        # Check success criteria
        if epoch_metrics['oracle_agreement'] > config.target_oracle_agreement:
            print(f"Phase 1 COMPLETE at epoch {epoch}")
            print(f"Oracle agreement: {epoch_metrics['oracle_agreement']:.1%}")
            save_checkpoint(trainer, 'phase1_complete.ckpt')
            break

        # Checkpoint
        if epoch % config.save_every == 0:
            save_checkpoint(trainer, f'phase1_epoch{epoch}.ckpt')

    print("Phase 1 training complete!")
```

---

## Phase 3: Testing Protocol

### 3.1 Unit Tests

**File**: `tests/test_easy_mode.py`

```python
def test_oracle_path_solver():
    """Verify oracle produces valid paths."""
    returns = generate_random_returns(T=100, n=64)
    path = greedy_lookahead_path(returns, k=4)

    assert path.selections.shape == (100, 64)
    assert path.total_alpha > 0  # Should capture some alpha
    assert path.trade_count > 0

def test_easy_trainer_forward():
    """Verify trainer forward pass works."""
    config = EasyConfig(batch_size=4)
    trainer = EasyTrainer(config)

    batch = create_dummy_batch(config)
    loss, metrics = trainer.loss_fn(trainer.params, batch)

    assert loss > 0
    assert 'oracle_agreement' in metrics

def test_easy_trainer_backward():
    """Verify gradients flow correctly."""
    config = EasyConfig(batch_size=4)
    trainer = EasyTrainer(config)

    batch = create_dummy_batch(config)
    grads = jax.grad(lambda p: trainer.loss_fn(p, batch)[0])(trainer.params)

    # Check gradients are non-zero
    grad_norm = optax.global_norm(grads)
    assert grad_norm > 0

def test_oracle_agreement_metric():
    """Verify oracle agreement computation."""
    # Perfect agreement
    logits = jax.nn.one_hot(jnp.arange(10), 320)
    actions = jnp.arange(10)
    assert compute_oracle_agreement(logits, actions) == 1.0

    # Random should be ~1/320
    logits = jax.random.normal(jax.random.PRNGKey(0), (10, 320))
    agreement = compute_oracle_agreement(logits, actions)
    assert agreement < 0.1
```

### 3.2 Integration Tests

```python
def test_full_training_loop():
    """Test complete training loop for a few epochs."""
    config = EasyConfig(epochs=3, batch_size=32)

    # Generate small dataset
    dataset = generate_oracle_dataset(config, n_episodes=10)

    # Train
    trainer = EasyTrainer(config)
    initial_loss = None

    for epoch in range(config.epochs):
        for batch in dataset.batches(config.batch_size):
            _, metrics = trainer.train_step(batch)
            if initial_loss is None:
                initial_loss = metrics['total_loss']

    # Loss should decrease
    final_loss = metrics['total_loss']
    assert final_loss < initial_loss

def test_dse_logging():
    """Test DSe metrics logger."""
    logger = DSeMetricsLogger('/tmp/test_dse.csv')

    logger.log_training_step({
        'epoch': 1,
        'policy_loss': 0.5,
        'oracle_agreement': 0.75,
    })

    # Verify file written
    assert os.path.exists('/tmp/test_dse.csv')
```

---

## Phase 4: Evaluation

### 4.1 Benchmark Evaluation

Reuse `config/benchmark_periods.json` for evaluation:

```python
def run_easy_eval(trainer, config):
    """Evaluate on benchmark periods."""
    results = {}

    for period in BENCHMARK_PERIODS:
        prices = load_benchmark_prices(period['name'])

        # Run inference (no MCTS yet in Phase 1)
        rewards, alpha, sharpe = evaluate_policy(
            trainer.model, trainer.params, prices
        )

        results[period['name']] = {
            'reward': rewards,
            'alpha': alpha,
            'sharpe': sharpe,
        }

    return results
```

### 4.2 Success Criteria Checklist

**Phase 1 Complete When**:
- [ ] Oracle agreement > 85% on training data
- [ ] Oracle agreement > 75% on validation data
- [ ] Value MSE < 0.1
- [ ] Pure intuition (argmax policy) beats random baseline
- [ ] Positive alpha on at least 4/6 benchmark periods

---

## Phase 5: Cleanup Plan

After Easy Mode is working:

### 5.1 Files to KEEP

```
Trader/
├── CLAUDE.md
├── easy_config.py
├── easy_trainer.py
├── train_easy.py
├── trade_easy.py
├── generate_oracle_data.py
│
├── mcts_trader/
│   ├── jax_agent/
│   │   ├── path_solver.py        # Oracle
│   │   ├── muzero_model.py       # Model architecture
│   │   ├── trm_flax.py           # TRM encoder
│   │   ├── expert_buffer.py      # Data storage
│   │   └── mctx_search.py        # For Phase 3 inference
│   └── env/
│       ├── jax_encoder.py        # Observation encoding
│       └── jax_trading_env.py    # Environment
│
├── tools/
│   └── DSe/                      # Easy Mode dashboard
│
├── config/
│   └── benchmark_periods.json    # Evaluation periods
│
├── lore/                         # Session notes (KEEP ALL)
├── tasks/                        # Task specs (KEEP ALL)
└── docs/                         # Research docs (KEEP ALL)
```

### 5.2 Files to ARCHIVE (move to `_archive/`)

```
_archive/
├── train_jax_fast.py
├── real_trade_jax.py
├── tools/DS24/
├── tools/orchestration/
├── mcts_trader/jax_agent/
│   ├── trainer.py
│   ├── popart.py
│   ├── redo.py
│   ├── atc.py
│   ├── cycle_balance.py
│   ├── adaptive_*.py
│   └── exploration/
└── diff_teacher/                 # Bella (keep for Phase 2 option)
```

---

## Implementation Order

```
Week 1: Support Tools
├── Day 1-2: Create DSe dashboard (fork from DS24)
├── Day 3: Create easy_config.py
└── Day 4-5: Create generate_oracle_data.py, test oracle pipeline

Week 2: Easy Trainer
├── Day 1-2: Create easy_trainer.py
├── Day 3: Create train_easy.py
└── Day 4-5: Unit tests, integration tests

Week 3: Training & Evaluation
├── Day 1-3: Train Phase 1 to 85% oracle agreement
├── Day 4: Benchmark evaluation
└── Day 5: Iterate on hyperparameters if needed

Week 4: Phase 3 + Cleanup
├── Day 1-2: Add inference MCTS (trade_easy.py)
├── Day 3: Test MCTS improvement over pure intuition
├── Day 4-5: Codebase cleanup
```

---

## Quick Reference Commands

```bash
# Generate oracle data
python generate_oracle_data.py --levels 0-14 --episodes-per-level 500

# Train Phase 1
python train_easy.py --epochs 100 --batch-size 256 --name phase1_run1

# Monitor training
./tools/DSe/start_dse.sh  # localhost:8051

# Evaluate
python train_easy.py --eval-only --checkpoint jax_ckpt_easy/phase1_complete.ckpt

# Inference with MCTS (Phase 3)
python trade_easy.py --checkpoint phase1.ckpt --sims 50
```
