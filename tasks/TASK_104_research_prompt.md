# TASK_104: Research Prompt - Adversarial Recursive Market Generator

## Research Query

I'm developing an adversarial curriculum learning system for training a cryptocurrency trading agent. The core idea is **co-evolution between a market generator and a trading model**.

## Context

Current system:
- Trading model: TRM (Tiny Recursive Model) + MuZero-style MCTS
- Training: JAX/Flax on RTX 3090
- Current approach: Synthetic curriculum (15 levels) → Historical data fine-tuning
- Problem: Synthetic patterns too simple, model overfits to toy problems, struggles on real market regimes

Observed behavior:
- Model excels on violent crashes (FTX collapse: +6% alpha)
- Model fails on slow bleeds, sideways chop (-1% to -2% alpha)
- Synthetic "dip" pattern matches crashes, but real bears are more complex

## Proposed System: "System 3"

Two-GPU setup where:
1. **GPU 0**: Trading model (existing TRM+MuZero)
2. **GPU 1**: Adaptive market generator

The generator creates market scenarios and adjusts difficulty based on model performance:
- If model alpha > 5%: increase complexity
- If model alpha < 0%: decrease complexity or add hints
- Sweet spot 0-5%: "zone of proximal development"

Key features:
- **Hierarchical pattern library**: Primitives → Composed → Multi-asset → Regime → Events → Adversarial
- **No forgetting**: Periodic replay of mastered patterns
- **Emergent complexity**: Generator learns what breaks the model
- **Infinite curriculum**: Never "done" - always new challenges

## Research Questions

### 1. Generator Architecture
- What's the best architecture for a controllable market generator?
  - GAN-based (learns realistic price distributions)
  - Compositional (combines hand-crafted primitives)
  - Neural ODE (physics-informed market dynamics)
  - Hybrid approaches
- How to make generation controllable by "difficulty" parameter?
- How to ensure generated markets are realistic enough to transfer?

### 2. Difficulty Control
- What should the difficulty metric be? (Alpha alone? Sharpe? Max drawdown?)
- Optimal difficulty curve: Should it be linear, exponential, or fully adaptive?
- How fast should difficulty change? (Risk of instability)
- Multi-dimensional difficulty: Can we independently control vol, trend, correlation, events?

### 3. Pattern Composition
- How to combine primitive patterns without interference?
- Should composition be learned or hand-designed?
- How many primitives are needed for good coverage?
- Can we learn new primitives from historical data?

### 4. Adversarial Dynamics
- How to train generator adversarially without mode collapse?
- What's the right game formulation? (Zero-sum? Cooperative?)
- How to detect and escape bad equilibria?
- Can adversarial training discover genuinely novel failure modes?

### 5. Forgetting Prevention
- How often to replay old patterns?
- Fixed schedule vs. performance-triggered replay?
- Does EWC (Elastic Weight Consolidation) or similar help?
- How to detect forgetting early?

### 6. Transfer to Real Markets
- How to validate that synthetic training transfers?
- What synthetic patterns best predict real-world performance?
- Is there a "sufficient statistics" set of patterns?
- How to measure the sim-to-real gap?

### 7. Implementation
- Multi-GPU communication: Ray? Shared memory? Message passing?
- Streaming architecture: How to pipeline generation and training?
- Checkpointing: How to save/restore generator + model jointly?
- Single-GPU fallback: Time-slicing viable?

## Specific Technical Questions

1. **For GANs**:
   - TimeGAN, Quant-GAN, Stock-GAN - which adapts best to conditional generation?
   - How to condition on difficulty/pattern type?
   - Discriminator architecture for multivariate time series?

2. **For Compositional**:
   - Best primitive set for financial markets?
   - Additive vs multiplicative composition?
   - How to handle different time scales?

3. **For Neural ODE**:
   - How to learn realistic drift/diffusion from data?
   - Jump process modeling (for events)?
   - Correlation structure dynamics?

4. **For Adversarial**:
   - Min-max vs alternating updates?
   - Gradient penalty / regularization for stability?
   - How to encourage diversity in generated patterns?

## Related Work to Explore

1. **Procedural Content Generation for RL**:
   - POET (Paired Open-Ended Trailblazer)
   - Procgen benchmark
   - Automatic curriculum learning

2. **Domain Randomization**:
   - Sim-to-real in robotics
   - Parameter randomization strategies
   - Domain adaptation techniques

3. **GANs for Finance**:
   - Conditional GANs for time series
   - Market simulation papers
   - Synthetic data for backtesting

4. **Self-Play and Co-Evolution**:
   - AlphaGo/AlphaZero
   - OpenAI Five
   - Evolutionary strategies

5. **Continual/Lifelong Learning**:
   - Catastrophic forgetting solutions
   - Elastic Weight Consolidation
   - Progressive neural networks

## Desired Output

Please provide:

1. **Architecture recommendations** for the market generator
2. **Difficulty control algorithm** with pseudocode
3. **Pattern library design** - what primitives and how to compose
4. **Adversarial training protocol** - game formulation, update rules
5. **Forgetting prevention strategy**
6. **Implementation roadmap** - phased approach with milestones
7. **Validation methodology** - how to know it's working
8. **Risk mitigation** - what could go wrong and how to detect/fix

## Constraints

- Must work with existing JAX/Flax training pipeline
- Primary GPU: RTX 3090 24GB (trading model)
- Secondary GPU: TBD (generator) - could be smaller
- Should be able to fallback to single-GPU if needed
- Training should not be slower than 2x current speed

## Current Performance Baseline

On historical data (without System 3):
- FTX crash period: +6% alpha (good)
- Sideways periods: -1% alpha (bad)
- Slow bear markets: -1.5% to -2.5% alpha (bad)
- Overall: Specialist in violent moves, poor at grinding

Goal: Consistent 2-5% alpha across all regimes.
