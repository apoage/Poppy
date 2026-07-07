# TASK_104: System 3 - Adversarial Recursive Market Generator

## Status: RESEARCH / FUTURE

## Vision

A co-evolutionary training system where a **market generator** (GPU 1) creates increasingly complex market scenarios while a **trading model** (GPU 0) learns to handle them. The generator adapts to keep the model in the "zone of proximal development" (0-5% alpha) - always challenging but learnable.

This is **self-play for markets** - instead of playing against itself (AlphaGo), the model plays against an adaptive environment that learns to challenge it.

## Problem Statement

Current approach limitations:
1. **Synthetic data too simple**: Sine waves, dips, trends don't capture real market complexity
2. **Historical data is finite**: Only ~80 years of crypto data, no counterfactuals
3. **Static curriculum**: Fixed levels don't adapt to model capabilities
4. **Sim-to-real gap**: Model masters toy problems but struggles on real regimes

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  GPU 1: MARKET GENERATOR (World Model)                              │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ Complexity Controller                                          │ │
│  │   - Monitor model alpha (rolling window)                       │ │
│  │   - if alpha > 5%: increase_difficulty()                       │ │
│  │   - if alpha < 0%: decrease_difficulty() or add_hints()        │ │
│  │   - if alpha in [0,5%]: optimal learning zone, add_variety()   │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                              │                                       │
│                              ▼                                       │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ Pattern Composer                                               │ │
│  │   - Combines base patterns into complex scenarios              │ │
│  │   - Learned mixing weights from model feedback                 │ │
│  │   - Can invent novel combinations                              │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                              │                                       │
│                              ▼                                       │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ Pattern Library (Hierarchical)                                 │ │
│  │                                                                │ │
│  │ TIER 1 - Primitives:                                          │ │
│  │   sine, trend, noise, mean_reversion, momentum                │ │
│  │                                                                │ │
│  │ TIER 2 - Composed:                                            │ │
│  │   dip_recovery, double_top, head_shoulders, channel           │ │
│  │                                                                │ │
│  │ TIER 3 - Multi-asset:                                         │ │
│  │   correlation, pairs, sector_rotation, contagion              │ │
│  │                                                                │ │
│  │ TIER 4 - Regime:                                              │ │
│  │   vol_clustering, regime_switch, correlation_breakdown        │ │
│  │                                                                │ │
│  │ TIER 5 - Events:                                              │ │
│  │   flash_crash, short_squeeze, liquidity_crisis, black_swan    │ │
│  │                                                                │ │
│  │ TIER 6 - Adversarial:                                         │ │
│  │   patterns_that_break_this_model (learned)                    │ │
│  │   future_scenarios_never_seen                                 │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                              │                                       │
│                              │ stream market data                    │
│                              ▼                                       │
└──────────────────────────────┼──────────────────────────────────────┘
                               │
         ┌─────────────────────┴─────────────────────┐
         │           Shared Memory / Queue           │
         │  (market states, model performance)       │
         └─────────────────────┬─────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  GPU 0: TRADING MODEL (Current TRM+MuZero System)                   │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ Training Loop                                                  │ │
│  │   - Receives streamed market epochs                           │ │
│  │   - Self-play with MCTS                                       │ │
│  │   - Reports alpha/metrics back to generator                   │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ Forgetting Prevention                                         │ │
│  │   - Replay buffer includes old pattern samples                │ │
│  │   - Periodic evaluation on pattern suite                      │ │
│  │   - Catastrophic forgetting detection                         │ │
│  └────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

## Core Principles

### 1. Zone of Proximal Development
Keep model at the edge of its capabilities:
- Too easy (alpha > 5%) → model memorizes, doesn't generalize
- Too hard (alpha < 0%) → model learns nothing, random policy
- Sweet spot (0-5%) → maximum learning gradient

### 2. No Catastrophic Forgetting
- Maintain pattern replay buffer
- Periodic evaluation on full pattern suite
- If old pattern performance drops, re-introduce it

### 3. Emergent Complexity
Generator doesn't just follow fixed curriculum - it **learns** what challenges the model:
- Track which patterns cause failures
- Generate more of those (adversarial)
- Create novel combinations of failure modes

### 4. Infinite Curriculum
Never "done" - always new challenges:
- Tier 6 patterns are learned adversarially
- Can generate patterns that don't exist in historical data
- Prepares model for future unknown regimes

## Market Generator Architecture

### Option A: GAN-based Generator
```python
class MarketGAN:
    def __init__(self):
        self.generator = PriceSeriesGenerator()  # Generates (T, N_assets) price paths
        self.discriminator = RealFakeClassifier()  # Distinguishes real vs synthetic
        self.difficulty_encoder = DifficultyNet()  # Maps difficulty level to latent

    def generate_epoch(self, difficulty: float, pattern_mix: Dict[str, float]):
        z = self.difficulty_encoder(difficulty, pattern_mix)
        prices = self.generator(z)
        return prices
```

### Option B: Compositional Generator
```python
class CompositionalMarket:
    def __init__(self):
        self.primitives = {
            'sine': SineGenerator(),
            'trend': TrendGenerator(),
            'noise': NoiseGenerator(),
            'jump': JumpGenerator(),
            # ... more primitives
        }
        self.composer = PatternComposer()  # Learned mixing network

    def generate_epoch(self, difficulty: float):
        # Select and weight patterns based on difficulty
        weights = self.composer(difficulty, model_feedback)

        prices = np.zeros((T, N_assets))
        for name, gen in self.primitives.items():
            prices += weights[name] * gen.generate()

        return prices
```

### Option C: Neural ODE Market (Physics-Informed)
```python
class NeuralMarketODE:
    """
    Differentiable market simulation based on:
    - Supply/demand dynamics
    - Order flow imbalance
    - Volatility clustering (GARCH-like)
    - Cross-asset correlation structure
    """
    def __init__(self):
        self.drift = LearnedDrift()      # Market tendency
        self.diffusion = LearnedVol()    # Volatility dynamics
        self.jump = LearnedJump()        # Discontinuities
        self.correlation = LearnedCorr() # Asset relationships

    def simulate(self, initial_state, T, difficulty):
        # Neural ODE integration with learned dynamics
        return odeint(self.dynamics, initial_state, t)
```

## Difficulty Controller

```python
class DifficultyController:
    def __init__(self):
        self.alpha_history = deque(maxlen=100)
        self.current_difficulty = 0.0  # 0 = easiest, 1 = hardest
        self.pattern_weights = default_weights()

    def update(self, epoch_alpha: float):
        self.alpha_history.append(epoch_alpha)
        mean_alpha = np.mean(self.alpha_history)

        if mean_alpha > 0.05:  # Too easy
            self.increase_difficulty()
        elif mean_alpha < 0.0:  # Too hard
            self.decrease_difficulty()
        else:  # Sweet spot
            self.add_variety()

    def increase_difficulty(self):
        options = [
            self.add_noise,
            self.add_regime_changes,
            self.increase_correlation_instability,
            self.add_jumps,
            self.shorten_patterns,  # Less time to react
            self.add_adversarial_patterns,
        ]
        random.choice(options)()
        self.current_difficulty = min(1.0, self.current_difficulty + 0.05)

    def decrease_difficulty(self):
        # Reduce complexity or add "hints" (clearer patterns)
        self.current_difficulty = max(0.0, self.current_difficulty - 0.05)
```

## Pattern Library Details

### Tier 1: Primitives
```python
primitives = {
    'sine': {'period': [10, 1000], 'amplitude': [0.01, 0.10]},
    'trend': {'drift': [-0.001, 0.001], 'duration': [100, 1000]},
    'noise': {'type': ['gaussian', 'student_t', 'levy'], 'scale': [0.001, 0.05]},
    'mean_reversion': {'half_life': [10, 100], 'equilibrium': 'rolling_mean'},
    'momentum': {'lookback': [5, 50], 'strength': [0.1, 0.9]},
}
```

### Tier 2: Composed Patterns
```python
composed = {
    'dip_recovery': combine(trend(-), jump(-), trend(+)),
    'double_top': combine(trend(+), reversal, trend(+), reversal, trend(-)),
    'channel': combine(trend, bounded_oscillation),
    'breakout': combine(channel, sudden_trend),
}
```

### Tier 3: Multi-Asset
```python
multi_asset = {
    'correlation': {'type': ['positive', 'negative', 'time_varying']},
    'pairs': {'spread': 'mean_reverting', 'legs': 'cointegrated'},
    'sector_rotation': {'n_sectors': 4, 'rotation_period': [50, 200]},
    'contagion': {'trigger': 'one_asset_crash', 'spread': 'correlation_based'},
}
```

### Tier 4: Regime
```python
regime = {
    'vol_clustering': {'garch_params': learned},
    'regime_switch': {'n_regimes': [2, 4], 'transition_matrix': learned},
    'correlation_breakdown': {'trigger': 'vol_spike', 'new_corr': 'random'},
}
```

### Tier 5: Events
```python
events = {
    'flash_crash': {'duration': [1, 10], 'magnitude': [-0.10, -0.30], 'recovery': [10, 100]},
    'short_squeeze': {'buildup': [50, 200], 'squeeze': [5, 20], 'magnitude': [0.20, 1.00]},
    'liquidity_crisis': {'spread_widening': 10x, 'slippage_spike': 5x},
    'black_swan': {'probability': 0.001, 'magnitude': 'extreme', 'correlation': 1.0},
}
```

### Tier 6: Adversarial (Learned)
```python
class AdversarialPatternGenerator:
    """
    Learns patterns that specifically break the current model.

    Trained adversarially:
    - Generator tries to minimize model alpha
    - Model tries to maximize alpha
    - Equilibrium = hardest patterns model can still learn
    """
    def __init__(self, model_feedback_dim):
        self.pattern_generator = TransformerGenerator()
        self.failure_predictor = FailurePredictor()  # Predicts what will break model

    def generate_adversarial(self, model_state):
        # Generate pattern predicted to cause model failure
        failure_embedding = self.failure_predictor(model_state)
        adversarial_prices = self.pattern_generator(failure_embedding)
        return adversarial_prices
```

## Implementation Phases

### Phase 1: Infrastructure (2-3 weeks)
- [ ] Multi-GPU communication setup (shared memory / queues)
- [ ] Streaming market data pipeline
- [ ] Performance feedback loop
- [ ] Basic difficulty controller

### Phase 2: Compositional Generator (2-3 weeks)
- [ ] Implement Tier 1-3 primitives
- [ ] Pattern composition framework
- [ ] Difficulty-to-pattern mapping
- [ ] Integration with existing training loop

### Phase 3: Regime & Events (2-3 weeks)
- [ ] Tier 4-5 patterns
- [ ] Regime switching mechanics
- [ ] Event injection system
- [ ] Forgetting prevention (replay buffer)

### Phase 4: Adversarial Learning (3-4 weeks)
- [ ] Adversarial pattern generator
- [ ] Co-training loop
- [ ] Equilibrium detection
- [ ] Novel pattern emergence

### Phase 5: Validation (2 weeks)
- [ ] Benchmark vs pure historical training
- [ ] Transfer to real trading evaluation
- [ ] Ablation studies

## Research Questions

1. **Optimal difficulty curve**: Linear? Exponential? Adaptive?
2. **Pattern interference**: Do some patterns hurt learning of others?
3. **Forgetting dynamics**: How often to replay old patterns?
4. **Adversarial equilibrium**: Does it converge? Multiple equilibria?
5. **Transfer**: Does synthetic adversarial training transfer to real markets?
6. **Emergent patterns**: What novel patterns does the generator discover?

## Success Metrics

1. **Learning efficiency**: Epochs to reach X% alpha on benchmark
2. **Generalization**: Performance on held-out historical periods
3. **Robustness**: Variance of alpha across different regimes
4. **No forgetting**: Maintained performance on early patterns
5. **Real-world transfer**: Paper trading performance

## Hardware Requirements

- GPU 0: RTX 3090 (trading model) - current setup
- GPU 1: Additional GPU for market generator
  - Could be smaller (RTX 3070/3080) for compositional generator
  - Needs to be similar size for full GAN approach
- Shared memory: Fast inter-GPU communication
- Alternative: Time-slice single GPU (slower but works)

## Related Work

- **POET** (Paired Open-Ended Trailblazer): Co-evolution of agents and environments
- **Procgen**: Procedurally generated game levels for RL generalization
- **Domain Randomization**: Vary simulation parameters for sim-to-real transfer
- **GANs for Time Series**: TimeGAN, Stock-GAN, etc.
- **Self-Play**: AlphaGo/AlphaZero paradigm applied to environment

## Dependencies

- Existing: JAX, Flax, mctx (current stack)
- New: Multi-process communication (Ray?), additional GPU support

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Generator mode collapse | Medium | High | Diversity regularization |
| No convergence | Medium | High | Fallback to pure historical |
| Overfits to generator | High | Medium | Regular historical validation |
| Compute cost | High | Medium | Start with compositional (cheaper) |
| Forgetting | High | Medium | Mandatory replay buffer |

## Open Questions for Research

1. Can a learned generator create patterns that don't exist yet but *could* exist?
2. Is adversarial training necessary or does compositional curriculum suffice?
3. What's the minimum generator complexity needed for good transfer?
4. How to validate "future-proofness" without waiting for the future?

---

## Research Prompt

See `tasks/TASK_104_research_prompt.md` for deep research prompt to explore this concept further.
