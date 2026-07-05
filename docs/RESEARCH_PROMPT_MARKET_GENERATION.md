# Research Prompt: Controllable Market Data Generation

## Research Goal

Design and implement a **controllable generative model for financial time series** that can:
1. Generate realistic multi-asset OHLCV price data
2. Accept high-level "skeleton" conditioning for scenario control
3. Scale efficiently to long sequences (1440+ timesteps) and many assets (64+)

---

## Part 1: Skeleton-Conditioned Generation (ControlNet for Markets)

### Core Concept

Adapt the ControlNet paradigm from image generation to financial time series:
- **Skeleton**: Normalized abstract structure (tokens) defining WHAT should happen
- **Generator**: Learns to render skeleton into realistic market data at proper scale

```
Image Domain:        Skeleton → ControlNet → Pixels
Market Domain:       Market Skeleton → Generator → OHLCV Prices
```

### Research Questions

1. **Skeleton Design**
   - What is the minimal skeleton representation that provides useful control?
   - How to encode:
     - Trend direction (per-asset, time-varying)
     - Correlation structure (clusters, lead-lag relationships)
     - Volatility regime sequence
     - Alpha opportunities (where/when profit exists)
     - Market phase transitions (crash → recovery → rally)
   - Should skeleton be discrete tokens or continuous embeddings?
   - How to handle conflicting skeleton components?

2. **Conditioning Mechanism**
   - FiLM (scale/shift) vs cross-attention vs concatenation?
   - Per-layer conditioning vs global conditioning?
   - How much capacity should go to skeleton encoding vs generation?

3. **Training Strategy**
   - Extract skeletons from historical data → train conditioned generator
   - How to ensure generator learns both fidelity (realistic) and controllability (follows skeleton)?
   - Classifier-free guidance for stronger conditioning?
   - Two-stage: pre-train unconditional, fine-tune with skeleton?

4. **Evaluation**
   - Fidelity: Statistical similarity to real markets (vol, autocorrelation, fat tails)
   - Controllability: Does output match skeleton specification?
   - Diversity: Same skeleton → varied but consistent realizations
   - Utility: Does RL agent trained on generated data generalize to real data?

### Key References to Explore

- ControlNet (Zhang et al., 2023) - Conditioning image diffusion
- Classifier-Free Diffusion Guidance (Ho & Salimans, 2022)
- TimeGAN (Yoon et al., 2019) - Time series generation
- Conditional Flow Matching (Lipman et al., 2023)
- Market regime detection literature

---

## Part 2: Mamba / State Space Models for Market Generation

### Why Mamba?

| Model | Memory | Long-range | Speed | Cross-asset |
|-------|--------|------------|-------|-------------|
| CNN (dilated) | O(n) | Limited by kernel | Fast | Implicit |
| Transformer | O(n²) | Global | Slow | Explicit attention |
| **Mamba/SSM** | O(n) | Global | Fast | Implicit |

Mamba offers transformer-like global context with CNN-like efficiency.

### Research Questions

1. **Architecture Adaptation**
   - How to adapt Mamba for flow matching / diffusion?
   - Inject time/regime conditioning: FiLM? AdaLN? Cross-attention?
   - Multi-asset: treat as single long sequence or separate streams with mixing?
   - Bidirectional Mamba for diffusion (process noise → signal)?

2. **Selective State Space**
   - How do input-dependent A, B, C, D matrices help for market data?
   - What does "selectivity" mean for financial patterns?
   - Optimal d_state, d_conv, expansion factor for market generation?

3. **Implementation Challenges**
   - JAX/Flax implementation: pure JAX scan vs custom CUDA kernel
   - Parallel scan (associative scan) for training efficiency
   - Gradient flow through selective scan
   - Memory-efficient backward pass

4. **Hybrid Approaches (Jamba)**
   - Mix Mamba + sparse attention layers
   - When does explicit cross-asset attention help vs pure SSM?
   - Ratio of Mamba to attention layers?

5. **Multi-GPU Training**
   - Data parallelism for 4x GPUs without NVLink
   - Gradient sync over PCIe bandwidth constraints
   - Batch size scaling: 32 total → 8 per GPU?

### Key References to Explore

- Mamba: Linear-Time Sequence Modeling (Gu & Dao, 2023)
- Jamba: Hybrid Mamba-Attention (AI21 Labs, 2024)
- S4: Structured State Spaces for Sequence Modeling (Gu et al., 2022)
- mamba-jax community implementations
- Flow Matching for generative modeling

---

## Part 3: Comparative Benchmark Design

### Experiment Setup

Train on same data, same epochs, compare architectures:

| Model | Params | Architecture |
|-------|--------|--------------|
| CNN Baseline | 93M | Dilated causal conv, 768d/12L |
| CNN Large | 302M | Dilated causal conv, 1536d/16L |
| Mamba | ~100M | SSM blocks with FiLM |
| Hybrid | ~100M | Mamba + sparse attention |
| Transformer | 22M | Full attention (baseline) |

### Metrics

1. **Training Efficiency**
   - Time per epoch
   - GPU memory usage
   - Convergence speed (epochs to target loss)

2. **Generation Quality**
   - Statistical fidelity (ACF, volatility clustering, fat tails)
   - Visual plausibility (price paths look realistic)
   - Cross-asset correlation structure

3. **Conditioning Effectiveness** (for skeleton-conditioned)
   - Trend direction accuracy
   - Regime fidelity (crash looks like crash)
   - Alpha opportunity placement

4. **Downstream Utility**
   - RL agent performance when trained on generated data
   - Generalization to real market data
   - Curriculum learning effectiveness

---

## Implementation Roadmap

### Phase 1: CNN Baseline (TASK_111) ✓
- [x] Implement dilated causal conv generator
- [x] Integrate with diff_teacher training
- [x] Train 93M model on historical data
- [ ] Evaluate generation quality

### Phase 2: Skeleton Extraction & Conditioning (TASK_112)
- [ ] Implement skeleton extraction from real data
- [ ] Design skeleton encoder
- [ ] Add FiLM conditioning to CNN
- [ ] Train skeleton-conditioned generator
- [ ] Build skeleton design interface

### Phase 3: Mamba Implementation (TASK_113)
- [ ] Implement MambaBlock in pure JAX
- [ ] Create MambaFlowGenerator
- [ ] Profile speed/memory vs CNN
- [ ] Optimize with parallel scan
- [ ] Multi-GPU data parallelism

### Phase 4: Benchmark & Select
- [ ] Train all architectures on same data
- [ ] Compare metrics
- [ ] Select winner for production use
- [ ] Document findings

---

## Specific Search Queries

For web/academic research:

1. "Mamba state space model JAX implementation"
2. "ControlNet conditioning mechanism flow matching"
3. "Time series generation diffusion model conditioning"
4. "Selective state space model financial data"
5. "Classifier-free guidance conditional generation"
6. "Multi-asset financial time series generation deep learning"
7. "Jamba hybrid mamba attention architecture"
8. "Flow matching conditional generation tutorial"
9. "Market regime detection clustering"
10. "Correlation structure generation finance"

---

## Notes

- **Hardware constraint**: 4x RTX 3090 without NVLink (PCIe only)
- **Sequence length target**: 1440 timesteps (1 day at 1-min resolution)
- **Asset count**: 64 crypto pairs
- **Features per asset**: 5 (OHLCV normalized)
- **Input shape**: (batch, 1440, 64*5) = (batch, 1440, 320)

The goal is not just generation quality, but **controllable generation** for RL curriculum design.
