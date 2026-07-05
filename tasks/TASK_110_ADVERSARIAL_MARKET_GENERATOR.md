# TASK_110: Adversarial Market Generator

## Status: IN_PROGRESS (Phase 1-4 Complete)

**Priority**: HIGH (Core curriculum enhancement)
**Effort**: 3-4 sessions
**Dependencies**: TASK_108 (Diff Teacher), TASK_109 (Validation)

---

## Objective

Build an adversarial market generation system that automatically creates training scenarios targeting the agent's weaknesses. Uses PAIRED-style regret-based curriculum to continuously challenge the TRM+MuZero trader.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ADVERSARIAL TRAINING LOOP                           │
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │   Diff       │───▶│   Agent      │───▶│   Regret     │                  │
│  │  Teacher     │    │  Evaluator   │    │  Calculator  │                  │
│  │ (Generator)  │    │ (TRM+MuZero) │    │              │                  │
│  └──────┬───────┘    └──────────────┘    └──────┬───────┘                  │
│         │                                        │                          │
│         │         ┌──────────────────┐          │                          │
│         └─────────│   Curriculum     │◀─────────┘                          │
│                   │   Controller     │                                      │
│                   └────────┬─────────┘                                      │
│                            │                                                │
│                            ▼                                                │
│                   ┌──────────────────┐                                      │
│                   │  train_jax_fast  │                                      │
│                   │  (Agent Trainer) │                                      │
│                   └──────────────────┘                                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Components

### Phase 1: Foundation (COMPLETE)

| Component | File | Status |
|-----------|------|--------|
| Market Encoder (VAE) | `diff_teacher/encoder.py` | ✅ Done |
| Flow Matching Generator | `diff_teacher/flow_matching.py` | ✅ Done |
| Optimal Solver | `diff_teacher/optimal_solver.py` | ✅ Done |
| Regret Calculator | `diff_teacher/regret_curriculum.py` | ✅ Done |
| Rainbow Data Loader | `diff_teacher/data_loader.py` | ✅ Done |
| Training Script | `diff_teacher/train.py` | ✅ Done |
| Small Model Training | `train_diff_teacher_small.sh` | ⏸️ Needs restart |

### Phase 2: Agent Evaluator ✅ COMPLETE

**File**: `diff_teacher/agent_evaluator.py`

**Purpose**: Run TRM+MuZero agent on generated sequences and compute PnL.

**Interface**:
```python
class AgentEvaluator:
    """Evaluate TRM+MuZero agent on generated market sequences."""

    def __init__(
        self,
        checkpoint_path: str,
        model_size: str = "small",
        n_simulations: int = 24,
    ):
        """
        Load agent from checkpoint.

        Args:
            checkpoint_path: Path to TRM+MuZero checkpoint
            model_size: small/medium/large
            n_simulations: MCTS simulations per step
        """
        pass

    def evaluate(
        self,
        prices: jnp.ndarray,  # (batch, steps, n_assets)
        initial_cash: float = 10000.0,
    ) -> EvaluationResult:
        """
        Run agent on price sequences.

        Returns:
            EvaluationResult with:
            - actual_pnl: Agent's PnL
            - actions: Action sequence
            - positions: Position history
            - metrics: Hold rate, trade count, etc.
        """
        pass

    def evaluate_batch(
        self,
        prices: jnp.ndarray,  # (batch, steps, n_assets)
    ) -> List[EvaluationResult]:
        """Evaluate batch of sequences in parallel."""
        pass
```

**Implementation Details**:
1. Load TRM model and MuZero networks from checkpoint
2. Create JAX trading environment with generated prices
3. Run MCTS search at each step
4. Track positions, trades, and compute final PnL
5. Return structured results for regret calculation

**Key Challenges**:
- Efficient batched evaluation (use JAX vmap)
- Handle variable sequence lengths
- Match environment config (fees, slippage) with training

**Implementation Notes** (2026-01-24):
- `AgentEvaluator` auto-detects architecture from checkpoint `trainer_metadata.json`
- `SimpleStateEncoder` creates (27, n_assets, n_assets) observations from raw prices
- Sequential evaluation loop with MCTS search at each step
- Tracks: positions, trades, PnL, hold rate, max positions

---

### Phase 3: Adversarial Loss ✅ COMPLETE

**File**: `diff_teacher/adversarial_training.py`

**Purpose**: Train generator to create sequences where agent has HIGH regret.

**Loss Function**:
```python
def adversarial_loss(
    generator_params,
    encoder_params,
    agent_evaluator: AgentEvaluator,
    batch_latents: jnp.ndarray,
    regime_indices: jnp.ndarray,
) -> Tuple[float, Dict[str, float]]:
    """
    Adversarial loss: maximize agent regret.

    L_adv = -mean(regret) + λ_realism * L_realism

    Where:
    - regret = (optimal_pnl - actual_pnl) / |optimal_pnl|
    - L_realism = reconstruction loss (keep sequences realistic)
    """
    # Generate sequences
    generated = generator.apply(generator_params, latents, regimes)

    # Decode to prices
    prices = decoder.apply(encoder_params, generated)

    # Evaluate agent
    results = agent_evaluator.evaluate_batch(prices)
    actual_pnls = jnp.array([r.actual_pnl for r in results])

    # Compute optimal
    optimal_pnls = optimal_pnl_jax(prices)

    # Regret (higher = agent struggles more)
    regret = (optimal_pnls - actual_pnls) / (jnp.abs(optimal_pnls) + 1e-6)

    # Adversarial loss: MAXIMIZE regret (minimize negative regret)
    adv_loss = -jnp.mean(regret)

    # Realism constraint (keep sequences plausible)
    realism_loss = compute_realism_loss(prices)

    total_loss = adv_loss + lambda_realism * realism_loss

    return total_loss, {
        'adv_loss': adv_loss,
        'realism_loss': realism_loss,
        'mean_regret': jnp.mean(regret),
        'max_regret': jnp.max(regret),
    }
```

**Realism Constraints**:
```python
def compute_realism_loss(prices: jnp.ndarray) -> float:
    """
    Ensure generated sequences are realistic.

    Checks:
    - Price positivity: prices > 0
    - Volatility bounds: not too extreme
    - Correlation structure: assets correlated like real markets
    - No discontinuities: smooth price paths
    """
    # Positivity
    pos_loss = jnp.mean(jnp.maximum(-prices, 0))

    # Volatility (returns std between 0.001 and 0.1)
    returns = jnp.diff(prices, axis=1) / (prices[:, :-1] + 1e-6)
    vol = jnp.std(returns, axis=1)
    vol_loss = jnp.mean(jnp.maximum(vol - 0.1, 0) + jnp.maximum(0.001 - vol, 0))

    # Smoothness (no jumps > 10%)
    jump_loss = jnp.mean(jnp.maximum(jnp.abs(returns) - 0.1, 0))

    return pos_loss + vol_loss + jump_loss
```

**Training Loop**:
```python
def train_adversarial(
    teacher: DiffusionTeacher,
    agent_evaluator: AgentEvaluator,
    epochs: int = 50,
    batch_size: int = 16,
    lambda_realism: float = 0.1,
):
    """
    Adversarial training loop.

    1. Sample random latents
    2. Generate sequences
    3. Evaluate agent
    4. Compute regret
    5. Update generator to INCREASE regret
    """
    for epoch in range(epochs):
        # Sample latents
        latents = sample_latents(batch_size)

        # Bias towards regimes where agent struggles
        regimes = curriculum.sample_hard_regimes()

        # Compute loss and gradients
        loss, metrics, grads = adversarial_step(
            teacher, agent_evaluator, latents, regimes
        )

        # Update generator
        teacher.generator_state = update_generator(grads)

        # Log metrics
        log(f"Epoch {epoch}: regret={metrics['mean_regret']:.3f}")
```

---

### Phase 4: Curriculum Integration ✅ COMPLETE

**File**: `diff_teacher/curriculum_bridge.py`

**Purpose**: External data generation with minimal trainer changes.

**Architecture** (as implemented):
```
┌─────────────────────────────────────────────────────────────────┐
│                    EXTERNAL CURRICULUM LOOP                      │
│                                                                  │
│   ┌──────────────────┐         ┌─────────────────┐              │
│   │ curriculum_bridge │────────▶│ generated_data/ │              │
│   │ (Data Generator)  │  .npz   │ adversarial/    │              │
│   └──────────────────┘         └────────┬────────┘              │
│                                          │                       │
│                                          │ --adversarial-data-dir│
│                                          ▼                       │
│                              ┌─────────────────────┐             │
│                              │  train_jax_fast.py  │             │
│                              │  (loads & mixes)    │             │
│                              └─────────────────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

**Key Components**:

1. **AdversarialDataGenerator** - Generates .npz files with adversarial sequences
2. **AdversarialDataLoader** - Loads sequences for validation/monitoring
3. **CurriculumMonitor** - Watches training logs, triggers regeneration
4. **Trainer Integration** - Loads and mixes adversarial data at startup

**CLI Usage**:
```bash
# Generate adversarial data (teacher runs separately)
python -m diff_teacher.curriculum_bridge \
    --teacher-checkpoint checkpoints/diff_teacher_small \
    --output-dir generated_data/adversarial \
    --n-batches 10 \
    --generate-only

# Or: Run continuous curriculum loop
python -m diff_teacher.curriculum_bridge \
    --teacher-checkpoint checkpoints/diff_teacher_small \
    --output-dir generated_data/adversarial \
    --monitor-log "logs/training_*.log"
```

**Train with adversarial data**:
```bash
python train_jax_fast.py \
    --synthetic-level 0 \
    --adversarial-data-dir generated_data/adversarial \
    --adversarial-ratio 0.3 \
    --epochs 100
```

**Files Modified** (minimal changes):
- `mcts_trader/training/cli.py` - Added `--adversarial-data-dir`, `--adversarial-ratio`
- `mcts_trader/training/config.py` - Added `adversarial_data_dir`, `adversarial_ratio` fields
- `mcts_trader/training/setup.py` - Added `load_adversarial_prices()`, `mix_adversarial_prices()`
- `train_jax_fast.py` - Import and call mixing functions at startup

**Data Format** (adversarial_batch_NNNNNN.npz):
```python
{
    'prices': np.ndarray,   # (batch, steps, n_assets) float32
    'regimes': np.ndarray,  # (batch,) strings like 'crash', 'rally'
}
```

**Test Results** (2026-01-24):
- Generated 20 sequences across 2 batch files
- Loaded and mixed 10 sequences successfully
- 40% of synthetic prices replaced with adversarial data
- All integration tests pass

---

### Phase 5: Weakness Detection

**File**: `diff_teacher/weakness_detector.py`

**Purpose**: Analyze agent behavior to find exploitable patterns.

**Analysis Methods**:
```python
class WeaknessDetector:
    """Detect agent weaknesses for targeted adversarial generation."""

    def analyze_regime_performance(
        self,
        results: List[EvaluationResult],
    ) -> Dict[str, float]:
        """
        Group results by regime, find weak regimes.

        Returns:
            regime -> mean_regret mapping
        """
        by_regime = defaultdict(list)
        for r in results:
            by_regime[r.regime].append(r.regret)

        return {
            regime: np.mean(regrets)
            for regime, regrets in by_regime.items()
        }

    def analyze_pattern_sensitivity(
        self,
        results: List[EvaluationResult],
    ) -> Dict[str, float]:
        """
        Find price patterns where agent struggles.

        Patterns:
        - Volatility clustering
        - Trend reversals
        - Correlation breakdowns
        - Gap opens
        """
        pass

    def analyze_action_biases(
        self,
        results: List[EvaluationResult],
    ) -> Dict[str, Any]:
        """
        Detect action-level issues.

        Issues:
        - Over-holding (high hold rate)
        - Poor timing (buy high, sell low)
        - Asset concentration
        - Missing arbitrage opportunities
        """
        pass

    def generate_weakness_report(self) -> WeaknessReport:
        """
        Comprehensive weakness analysis.

        Used to condition adversarial generator.
        """
        return WeaknessReport(
            weak_regimes=['crash', 'volatile'],
            weak_patterns=['trend_reversal', 'volatility_spike'],
            action_biases={'hold_rate': 0.72, 'timing_score': 0.3},
            recommendations=[
                'Increase crash regime training',
                'Add more reversal patterns',
            ]
        )
```

---

## CLI Interface

**New training script**: `train_adversarial.py`

```bash
# Phase 1: Train base generator (CURRENT)
./train_diff_teacher_small.sh

# Phase 2: Train adversarial generator
python -m diff_teacher.train_adversarial \
    --teacher-checkpoint checkpoints/diff_teacher_small \
    --agent-checkpoint jax_ckpt_fast/latest \
    --epochs 50 \
    --batch-size 16 \
    --lambda-realism 0.1

# Phase 3: Run adversarial curriculum training
python train_jax_fast.py \
    --adversarial-curriculum checkpoints/diff_teacher_adversarial \
    --adversarial-ratio 0.3 \
    --epochs 100
```

---

## Testing Plan

### Unit Tests

```python
# test_agent_evaluator.py
def test_evaluator_loads_checkpoint():
    evaluator = AgentEvaluator("jax_ckpt_fast/test")
    assert evaluator.model is not None

def test_evaluator_returns_pnl():
    prices = generate_test_prices(100, 64)
    result = evaluator.evaluate(prices)
    assert hasattr(result, 'actual_pnl')
    assert isinstance(result.actual_pnl, float)

def test_evaluator_batched():
    prices = generate_test_prices(8, 100, 64)
    results = evaluator.evaluate_batch(prices)
    assert len(results) == 8

# test_adversarial_loss.py
def test_adversarial_loss_increases_regret():
    # Train generator for a few steps
    initial_regret = compute_mean_regret(teacher, evaluator)
    train_adversarial(teacher, evaluator, epochs=10)
    final_regret = compute_mean_regret(teacher, evaluator)
    assert final_regret > initial_regret  # Generator finds harder cases

def test_realism_constraint():
    prices = teacher.generate(batch_size=32)
    assert (prices > 0).all()  # Positive prices
    returns = compute_returns(prices)
    assert returns.std() < 0.2  # Reasonable volatility
```

### Integration Tests

```python
# test_curriculum_integration.py
def test_mixed_batch_generation():
    curriculum = AdversarialCurriculum(teacher_ckpt, agent_ckpt)
    prices, is_adv = curriculum.generate_training_batch(32, normal_loader)
    assert prices.shape == (32, 100, 64)
    assert is_adv.sum() == int(32 * 0.3)

def test_weakness_detection():
    detector = WeaknessDetector()
    report = detector.generate_weakness_report(results)
    assert len(report.weak_regimes) > 0
```

---

## Success Criteria

1. **Agent Evaluator**
   - [x] Loads TRM+MuZero checkpoint (auto-detects architecture from metadata)
   - [x] Runs inference on generated sequences (MCTS search)
   - [x] Returns accurate PnL (tracks positions, fees, cash)
   - [ ] Batched evaluation < 1s for 32 sequences (currently sequential)

2. **Adversarial Training**
   - [x] Generator learns to increase regret (0.689 → 0.719 in test)
   - [x] Generated sequences remain realistic (realism loss included)
   - [x] Mean regret increases during adversarial training (+0.030 in 2 epochs)

3. **Curriculum Integration**
   - [x] Mixed batches work in train_jax_fast.py (via external .npz files)
   - [x] Data loading and mixing tested successfully
   - [ ] Agent performance improves on adversarial sequences (needs training run)
   - [ ] No memory leaks during extended training (needs testing)

4. **Weakness Detection**
   - [ ] Correctly identifies weak regimes
   - [ ] Detected weaknesses correlate with actual performance

---

## Timeline

| Phase | Description | Effort | Dependencies |
|-------|-------------|--------|--------------|
| 1 | Foundation (Diff Teacher) | ✅ Complete | - |
| 2 | Agent Evaluator | 1 session | Phase 1 |
| 3 | Adversarial Loss | 1 session | Phase 2 |
| 4 | Curriculum Integration | 0.5 session | Phase 3 |
| 5 | Weakness Detection | 0.5 session | Phase 2 |
| 6 | Testing & Tuning | 1 session | All |

---

## Files to Create

| File | Purpose |
|------|---------|
| `diff_teacher/agent_evaluator.py` | Run agent on generated sequences |
| `diff_teacher/adversarial_training.py` | Train generator to maximize regret |
| `diff_teacher/curriculum_bridge.py` | Connect to train_jax_fast.py |
| `diff_teacher/weakness_detector.py` | Analyze agent weaknesses |
| `diff_teacher/train_adversarial.py` | CLI for adversarial training |

---

## Notes

- Start with small model (window=100, latent=64) for fast iteration
- Use 12GB VRAM budget (can run alongside synthetic training)
- Agent checkpoint should be recent (represents current weaknesses)
- Realism constraint prevents degenerate solutions (impossible markets)
- Mix ratio (30% adversarial) prevents forgetting normal scenarios

---

## References

- PAIRED: Adversarial Environment Generation (Dennis et al., 2020)
- Regret-based curriculum: `diff_teacher/regret_curriculum.py`
- TRM+MuZero architecture: `mcts_trader/jax_agent/trm_flax.py`
