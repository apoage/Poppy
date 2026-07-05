# TASK_108: Latent Diffusion Market Generator for RL Curriculum

## Status: IMPLEMENTATION IN PROGRESS

**Priority**: HIGH (Architectural Innovation)
**Effort**: Very High (Multi-week research + implementation)
**Dependencies**: Stable training pipeline (Dojo V4)

---

## Hardware Constraints

| Constraint | Target | Rationale |
|------------|--------|-----------|
| **Inference VRAM** | ≤6 GB (ideal) / ≤10GB (OK) | Fit on single 3090 or CMP 90HX |
| **Generation Speed** | ≥1500 steps/sec | Real-time curriculum |
| **Training VRAM** | ≤24 GB | Single GPU training |

**Available Hardware**:
- RTX 3090 (24GB) - Primary
- CMP 90HX (10GB) - Mining card, no display (backup)
- Option: Second RTX 3090 + NVLink bridge

**Deployment Options**:
- **Option A**: Single 3090 (if generator ≤6GB) - **STRONGLY PREFERRED**
- **Option B**: 3090 + CMP 90HX (if generator ≤10GB) - Fallback
- **Option C**: Dual 3090 + NVLink (if generator >10GB) - Last resort

**Why Single GPU is Critical**:
- Adding second GPU consumes PCIe lanes (x16 → x8/x8 split)
- Bottlenecks NVMe, network, other peripherals
- IPC overhead adds latency to feedback loop
- Simpler code, fewer failure modes

**Design Priority**: Aggressively optimize generator to fit ≤6GB

---

## Problem Statement

Current synthetic market generators (L0-L14) use hand-crafted mathematical models:
- Sine waves, GARCH, Ornstein-Uhlenbeck, etc.
- Limited realism - don't capture true market microstructure
- No learned cross-asset correlations
- Can't generate "similar to period X" scenarios on demand

**Goal**: Train a diffusion model on historical data that can:
1. Generate realistic multi-asset market dynamics (64 assets)
2. Preserve cross-asset correlations and interactions
3. Support curriculum learning by generating specific regimes on demand
4. Enable closed-loop interaction (agent actions affect market)

---

## Proposed Architecture: Adaptive Feedback Loop

### Core Concept: Student-Teacher with Regret Minimization

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    ADAPTIVE CURRICULUM LOOP                              │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    GPU 1 (TEACHER - 6GB)                        │    │
│  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │    │
│  │  │  Generator   │───▶│   Market     │───▶│   Optimal    │      │    │
│  │  │  (Diffusion) │    │   Sequence   │    │   Solver     │      │    │
│  │  │   ~50M params│    │  1500 steps  │    │  (Hindsight) │      │    │
│  │  └──────────────┘    └──────────────┘    └──────┬───────┘      │    │
│  │         ▲                                        │              │    │
│  │         │                                        ▼              │    │
│  │         │                                 ┌──────────────┐      │    │
│  │         │                                 │   Regret     │      │    │
│  │         │                                 │   Score      │      │    │
│  │         │                                 │ (opt - actual)│     │    │
│  │         │                                 └──────┬───────┘      │    │
│  │         │                                        │              │    │
│  │  ┌──────┴───────┐                               │              │    │
│  │  │  Difficulty  │◀──────────────────────────────┘              │    │
│  │  │  Controller  │                                              │    │
│  │  │ (if regret↓  │                                              │    │
│  │  │  → harder)   │                                              │    │
│  │  └──────────────┘                                              │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                      │                                   │
│                                      │ Market Data                       │
│                                      ▼                                   │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    GPU 0 (STUDENT - 24GB)                       │    │
│  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │    │
│  │  │   Market     │───▶│  TRM+MuZero  │───▶│   Actions    │      │    │
│  │  │   Sequence   │    │   (Student)  │    │   Executed   │      │    │
│  │  └──────────────┘    └──────────────┘    └──────┬───────┘      │    │
│  │                                                  │              │    │
│  │                                                  ▼              │    │
│  │                                          ┌──────────────┐      │    │
│  │                                          │   Actual     │──────┼────┼─▶ Back to Teacher
│  │                                          │   PnL        │      │    │
│  │                                          └──────────────┘      │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### Feedback Loop Steps

1. **Generator** creates market sequence (1500 steps, ~1 second)
2. **Optimal Solver** computes best possible PnL (hindsight oracle)
3. **Student (Trader)** executes trades on sequence
4. **Regret Score** = Optimal PnL - Actual PnL
5. **Difficulty Controller** adjusts generation:
   - Low regret → Generate harder scenarios
   - High regret → Generate similar or easier scenarios
6. **Loop continues** - adaptive curriculum

### Why This Works

- **No human-designed curriculum** - difficulty emerges from regret
- **Automatic weakness detection** - high regret scenarios get regenerated
- **Bounded difficulty** - never too hard (would just confuse)
- **Similar to PAIRED/PLR** - proven in procedural RL

---

## Original Static Architecture (Simpler Version)

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  Historical  │───▶│   Encoder    │───▶│   Latent     │      │
│  │  Market Data │    │   (VAE/CNN)  │    │   Space z    │      │
│  │  64 assets   │    │              │    │   (512-dim)  │      │
│  └──────────────┘    └──────────────┘    └──────┬───────┘      │
│                                                  │              │
│                                                  ▼              │
│                                          ┌──────────────┐      │
│                                          │  Diffusion   │      │
│                                          │   Model      │      │
│                                          │  (Denoising) │      │
│                                          └──────┬───────┘      │
│                                                  │              │
│  ┌──────────────┐    ┌──────────────┐           │              │
│  │  Curriculum  │◀───│   Vector     │◀──────────┘              │
│  │   Manager    │    │   Database   │                          │
│  │              │    │  (scenarios) │                          │
│  └──────┬───────┘    └──────────────┘                          │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  Generated   │───▶│   Decoder    │───▶│   Trading    │      │
│  │  Vectors z'  │    │   (optional) │    │    Agent     │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Performance Targets & Feasibility

### Target: 6GB VRAM Inference

| Technique | Memory Savings | Implementation |
|-----------|----------------|----------------|
| **Quantization (INT8)** | 4x reduction | `jax.numpy.int8`, TensorRT |
| **Distillation** | 2-4x smaller model | Train small from large |
| **Flow Matching** | 1-step generation | Replace DDPM (50 steps → 1) |
| **Streaming generation** | Constant memory | Generate step-by-step |

**Model Size Budget (6GB)**:
```
6GB VRAM budget:
- Model weights (INT8): ~2GB → 2B params max
- Activations: ~2GB
- KV cache/buffers: ~2GB

Realistic target: 200-500M params (FP16) or 500M-1B (INT8)
```

### Target: 1500 Steps/Second

| Approach | Steps/Sec | Notes |
|----------|-----------|-------|
| **DDPM (50 steps)** | ~30-100 | Too slow |
| **DDIM (10 steps)** | ~150-500 | Better |
| **Flow Matching (1 step)** | ~1000-2000 | Achievable! |
| **Consistency Models** | ~1000-3000 | Single-step diffusion |
| **Cached/Batched** | 2-5x boost | Amortize overhead |

**Recommendation**: Use **Flow Matching** or **Consistency Distillation**
- Train full DDPM first
- Distill to 1-step model for inference
- Keep DDPM for high-quality scenario generation

### Optimal Solver (Hindsight Oracle)

Computing optimal PnL for a sequence:

```python
def compute_optimal_pnl(prices: Array, n_assets: int = 64) -> float:
    """
    Hindsight optimal: For each asset, compute max profit
    assuming perfect buy-low-sell-high timing.

    This is O(T) per asset using peak-valley detection.
    """
    total_profit = 0
    for asset in range(n_assets):
        asset_prices = prices[:, asset]
        # Find all local minima (buy) and maxima (sell)
        profit = compute_peak_valley_profit(asset_prices)
        total_profit += profit
    return total_profit
```

**Speed**: O(T × N_assets) = O(1500 × 64) = ~100K ops → <1ms

---

## Key Components

| Component | Purpose | Implementation Options |
|-----------|---------|------------------------|
| **Encoder** | Compress 64-asset window → latent vector | 1D-CNN, Transformer, VAE |
| **Diffusion Model** | Generate new market state vectors | DDPM, Latent Diffusion, Flow Matching |
| **Cross-Asset Attention** | Preserve correlations between assets | Transformer self-attention, GNN |
| **Vector Database** | Store/retrieve scenario embeddings | Qdrant, Milvus, FAISS |
| **Curriculum Manager** | Select scenarios based on agent weakness | Prioritized sampling |
| **Decoder** | (Optional) Convert vectors back to prices | Mirror of encoder |

---

## Training Loop Pseudocode

```python
# Phase 1: Train the Market Generator
encoder = TransformerEncoder(input_dim=64_assets, latent_dim=512)
diffusion = LatentDiffusionModel(latent_dim=512)
decoder = TransformerDecoder(latent_dim=512, output_dim=64_assets)

for batch in historical_market_data:
    z = encoder(batch)  # Compress to latent
    loss = diffusion.train_step(z)  # Learn to denoise

# Phase 2: Build Scenario Database
scenario_db = VectorDatabase()
for regime in ["crash", "rally", "sideways", "correlation_breakdown"]:
    z_scenarios = diffusion.generate(condition=regime, n=10000)
    scenario_db.insert(z_scenarios, tags=regime)

# Phase 3: Curriculum Training for Trader
trader = RLAgent(state_dim=512, action_dim=num_actions)

for episode in range(num_episodes):
    # Find trader's weakness
    weakness = curriculum.identify_weakness(trader)

    # Query scenarios that exploit weakness
    hard_scenarios = scenario_db.search(weakness, k=batch_size)

    # Train on hard scenarios (feed vectors directly!)
    for z in hard_scenarios:
        action = trader.act(z)
        z_next = diffusion.step(z, action)  # Market responds
        reward = compute_pnl(z, z_next, action)
        trader.learn(z, action, reward, z_next)
```

---

## Key Design Decisions

1. **Feed vectors directly to trader (skip decoder)** - faster training, cleaner signal
2. **Condition diffusion on market regimes** - enables curriculum control
3. **Closed-loop interaction** - trader actions affect next market state
4. **Cross-attention for 64 assets** - preserves realistic correlations

---

## Research Papers & Projects

### Core References (Researched 2026-01-23)

#### 1. TRADES: Transformer Diffusion for LOB (arXiv 2502.07071)

**Paper**: [TRADES: Generating Realistic Market Simulations with Diffusion Models](https://arxiv.org/abs/2502.07071)
**Code**: [DeepMarket GitHub](https://github.com/LeonardoBerti00/DeepMarket)
**Authors**: Leonardo Berti, Bardh Prenkaj, Paola Velardi
**Status**: Conference publication Aug 2025 (Toronto)

**Key Contributions**:
- Transformer-based Denoising Diffusion Probabilistic Engine for LOB simulation
- Generates realistic order flows as time series conditioned on market state
- Captures temporal AND spatial characteristics of high-frequency market data
- Evaluated on Tesla and Intel order books
- Uses predictive score (MAE) - train on synthetic, test on real

**Architecture**: Transformer + autoregressive time-series diffusion

**Why it matters**: Direct template for our multi-asset market generation. Order-level generation is more realistic than price-level.

---

#### 2. DigMA/DiGA: Diffusion Guided Meta Agent (arXiv 2408.12991)

**Paper**: [Controllable Financial Market Generation with Diffusion Guided Meta Agent](https://arxiv.org/abs/2408.12991)
**Authors**: Yu-Hao Huang, Chang Xu, Yang Liu, Weiqing Liu, Wu-Jun Li, Jiang Bian
**Updated**: January 2026

**Key Contributions**:
- Conditional diffusion model captures market state dynamics
- Meta agent with financial economic priors generates orders
- Two modules: meta controller (fits market states) + meta agent (generates orders)
- **Validates as RL training environment** - trains A2C agents for HFT
- Superior controllability AND generation fidelity

**Why it matters**: Proves diffusion + RL integration works. Shows how to condition on market regimes.

---

#### 3. MarS: Large Market Model (arXiv 2409.07486) - ICLR 2025

**Paper**: [MarS: Financial Market Simulation Engine Powered by Generative Foundation Model](https://arxiv.org/abs/2409.07486)
**Code**: [Microsoft MarS GitHub](https://github.com/microsoft/MarS/)
**Project**: [mars-lmm.github.io](https://mars-lmm.github.io/)
**Authors**: Microsoft Research (Junjie Li, Yang Liu, et al.)

**Key Contributions**:
- **Large Market Model (LMM)** - order-level generative foundation model
- Scales with data size and model complexity (follows scaling laws!)
- Real-time simulation with user-injected interactive orders
- Supports vague target scenario descriptions
- Four applications: forecast, detection, analysis, **agent training**

**Architecture**: Foundation model approach - pre-train large, fine-tune for tasks

**Why it matters**: Shows scaling laws apply to market models. Microsoft-backed, production quality.

---

#### 4. FinRL-Meta: Vectorized Trading Environments (NeurIPS 2022)

**Paper**: [FinRL-Meta: Market Environments and Benchmarks](https://papers.neurips.cc/paper_files/paper/2022/file/0bf54b80686d2c4dc0808c2e98d430f7-Paper-Datasets_and_Benchmarks.pdf)
**Code**: [FinRL-Meta GitHub](https://github.com/AI4Finance-Foundation/FinRL-Meta)

**Key Contributions**:
- Universe of market environments for financial RL
- Vectorized environment for massively parallel simulations
- DataOps pipeline: dynamic datasets → gym-style environments
- Three-layer architecture: data layer, environment layer, agent layer

**Future Plans**: FinRL-Metaverse with 10,000+ agents, GPU-based simulation (Isaac Gym style)

**Why it matters**: Infrastructure patterns for our environment. Vectorized parallel simulation is key.

---

#### 5. FTS-Diffusion: Financial Time Series Diffusion (ICLR 2024)

**Paper**: [FTS-Diffusion](https://github.com/pseudo-Skye/Time-Will-Tell/blob/main/financial%20trading/FTS-Diffusion%20(ICLR%202024).md)

**Key Contributions**:
- Three-prong process: pattern recognition → generation → evolution
- Handles "irregularity" - lack of predictable patterns in financial data
- Scale-invariant pattern detection
- Connects generated segments into complete time series

**Why it matters**: Addresses key challenge of irregular patterns in market data.

---

#### 6. Wavelet-DDPM for Synthetic Financial Data (2024-2025)

**Paper**: [Generation of synthetic financial time series by diffusion models](https://arxiv.org/abs/2410.18897)
**Journal**: Quantitative Finance (2025)

**Key Contributions**:
- Wavelet transformation converts time series → images
- DDPM generates images → inverse wavelet → realistic time series
- Preserves stylized facts: fat tails, volatility clustering, seasonality
- Handles multiple time series (prices, volumes, spreads)

**Why it matters**: Novel encoding approach. Could be alternative to transformer encoder.

---

### Additional 2025 Papers

| Paper | Venue | Key Innovation |
|-------|-------|----------------|
| **CoFinDiff** | arXiv 2025 | Controllable financial diffusion |
| **DHMoE** | AAAI 2025 | Diffusion hierarchical multi-granular expertise |
| **InterDiff** | ECML-PKDD 2025 | Inter-stock correlations via classifier-free guidance |

### Emerging Techniques

1. **GBM-aligned diffusion**: Forward noise injection aligned with Geometric Brownian Motion SDE
2. **Cross-attention conditioning**: Condition on volatility, trend, regime
3. **Scenario generation for risk**: Stress testing, VaR, CVaR estimation

### Research Gaps / Opportunities

1. **Multi-asset correlations**: Most papers focus on single asset or small baskets
2. **Curriculum integration**: No paper explicitly addresses curriculum RL with diffusion
3. **JAX implementation**: Most are PyTorch, opportunity for JAX optimization
4. **Real-time generation**: Speed for on-the-fly curriculum training

### Adaptive Curriculum / Regret-Based Training

| Paper | Key Idea | Relevance |
|-------|----------|-----------|
| **PAIRED** | Protagonist-Antagonist environment design | Regret-based difficulty |
| **PLR** | Prioritized Level Replay | Replay hard scenarios |
| **ACCEL** | Evolved curriculum | Genetic env generation |
| **Robust PLR** | Minimax regret | Theoretical foundation |

**Key Insight from PAIRED**:
> "Generate environments where the protagonist (student) has high regret
> compared to an antagonist that knows the environment"

Our setup: Generator = Antagonist, Trader = Protagonist, Optimal Solver = Oracle

### Fast Diffusion / Single-Step Generation

| Method | Steps | Speed | Quality |
|--------|-------|-------|---------|
| **DDPM** | 50-1000 | Slow | Best |
| **DDIM** | 10-50 | Medium | Good |
| **Flow Matching** | 1-4 | Fast | Good |
| **Consistency Models** | 1-2 | Fast | Good |
| **Rectified Flow** | 1 | Fastest | Good |

**Recommendation**: Train with DDPM, distill to Consistency Model for inference

---

## Implementation Phases

### Phase 0: Research (COMPLETE)
- [x] Deep dive into TRADES architecture
- [x] Understand DigMA/DiGA diffusion-RL integration
- [x] Survey time-series diffusion SOTA
- [x] Identify key architectural choices

### Phase 1: Data Pipeline (COMPLETE)
- [x] Extract windows from trading_data.db (27M candles)
- [x] Design latent representation (what to encode?)
- [x] Build efficient data loader for diffusion training
- **Implementation**: `diff_teacher/data_loader.py` (474 lines)

### Phase 2: Encoder/Decoder (COMPLETE - Code Ready)
- [x] Design encoder architecture (Transformer with cross-asset attention)
- [x] Implement VAE on historical windows
- [ ] Train and validate reconstruction quality
- [ ] Measure latent space properties
- **Implementation**: `diff_teacher/encoder.py` (582 lines)

### Phase 3: Diffusion Model (COMPLETE - Code Ready)
- [x] Implement Flow Matching in JAX
- [ ] Train on encoded latent vectors
- [x] Add regime conditioning (crash, rally, sideways, volatile)
- [ ] Validate generation quality
- **Implementation**: `diff_teacher/flow_matching.py` (593 lines)

### Phase 4: Optimal Solver & Curriculum (COMPLETE - Code Ready)
- [x] Implement hindsight optimal PnL solver
- [x] Implement regret-based curriculum controller
- **Implementation**: `diff_teacher/optimal_solver.py` (412 lines)
- **Implementation**: `diff_teacher/regret_curriculum.py` (443 lines)

### Phase 5: Teacher Integration (COMPLETE - Code Ready)
- [x] Create main teacher class combining all components
- [x] Create training script entry point
- [ ] Train and validate full pipeline
- **Implementation**: `diff_teacher/teacher.py` (594 lines)
- **Implementation**: `diff_teacher/train.py` (524 lines)

### Phase 6: Scenario Database (TODO)
- [ ] Build vector database (FAISS/Qdrant)
- [ ] Generate scenario library (crash, rally, etc.)
- [ ] Implement similarity search

### Phase 7: Curriculum Integration (TODO)
- [ ] Modify trading agent to accept latent input
- [ ] Implement weakness identification
- [ ] Build hard-scenario mining
- [ ] Compare vs current synthetic curriculum

### Phase 8: Closed-Loop (TODO - Advanced)
- [ ] Model market response to agent actions
- [ ] Implement action-conditioned dynamics
- [ ] Validate market impact realism

---

## Technical Considerations

### Latent Space Design

**Option A: Per-timestep encoding**
```
Input: (64 assets, T timesteps, 5 OHLCV) → z: (T, 512)
```
- Pro: Fine-grained control
- Con: Long sequences

**Option B: Window encoding**
```
Input: (64 assets, 400 steps, 5 OHLCV) → z: (512,)
```
- Pro: Compact, fast
- Con: Less temporal detail

**Option C: Hierarchical**
```
Input: (64 assets, 400 steps) → z_local: (40, 128) → z_global: (512,)
```
- Pro: Multi-scale
- Con: Complex

### Cross-Asset Correlations

Critical to preserve:
- Pairs cointegration (BTC/ETH, EUR/USD pairs)
- Sector correlations (DeFi tokens move together)
- Inverse correlations (risk-on vs risk-off)
- Contagion patterns (FTX crash spreading)

**Approach**: Self-attention over asset dimension in encoder

### Regime Conditioning

How to condition diffusion on "regime"?
1. **Classifier-free guidance**: Train with/without labels, interpolate
2. **Cross-attention**: Inject regime embedding
3. **Prefix conditioning**: Prepend regime tokens

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Reconstruction MSE | < 0.01 | Encoder-decoder on held-out data |
| Correlation preservation | > 0.95 Spearman | Cross-asset correlations in generated vs real |
| Regime accuracy | > 80% | Classifier can identify generated regime |
| Agent transfer | +10% reward | Agent trained on diffusion vs hand-crafted |
| Generation speed | > 1000 episodes/sec | Latency for curriculum training |

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Mode collapse | Generated data too homogeneous | Diversity loss, classifier-free guidance |
| Correlation loss | Assets move independently | Explicit correlation loss term |
| Slow generation | Training bottleneck | Latent diffusion, flow matching |
| Distribution shift | Generated != real enough | Discriminator validation |
| Overfitting to history | Memorization | Held-out periods for validation |

---

## Resources Required

- **GPU**: A100 80GB for diffusion training
- **Storage**: ~100GB for scenario database
- **Time**: 2-4 weeks research, 4-8 weeks implementation
- **Dependencies**: JAX, Flax, Orbax, FAISS/Qdrant

---

## Hardware Deployment Options

### Option A: Single RTX 3090 (24GB) - If Generator ≤6GB

```
┌─────────────────────────────────────────────────────────────────┐
│                 SINGLE GPU (RTX 3090 24GB)                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────┐  ┌──────────────────────┐            │
│  │  TRM+MuZero TRAINING │  │  GENERATOR INFERENCE │            │
│  │  ~16-18GB VRAM       │  │  ~4-6GB VRAM         │            │
│  └──────────────────────┘  └──────────────────────┘            │
│                                                                 │
│  Total: ~20-24GB - FITS ON ONE 3090!                           │
│  Pro: No IPC overhead, fastest communication                   │
│  Con: Tight on memory, may need careful scheduling             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Best case**: Generator fits in 6GB → Everything on one 3090

### Option B: RTX 3090 + CMP 90HX (10GB)

```
┌─────────────────────────────────────────────────────────────────┐
│              RTX 3090 (24GB) + CMP 90HX (10GB)                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  GPU 0: RTX 3090 (24GB)               GPU 1: CMP 90HX (10GB)   │
│  ┌─────────────────────────┐          ┌─────────────────────────┐
│  │  TRM+MuZero TRAINING    │◀────────▶│  GENERATOR INFERENCE    │
│  │  - Full model (~36M)    │  PCIe    │  - Larger model OK      │
│  │  - Replay buffer        │          │  - FP16 (no INT8 needed)│
│  │  - MCTS simulations     │          │  - More headroom        │
│  │  VRAM: ~20GB            │          │  VRAM: ~8GB usable      │
│  └─────────────────────────┘          └─────────────────────────┘
│                                                                 │
│  Note: CMP 90HX has no display output (mining card)            │
│  Communication: PCIe (slower than NVLink)                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**CMP 90HX specs**: 10GB HBM2e, ~50 TFLOPs FP32 (similar to 3090)

### Option C: Dual RTX 3090 + NVLink (48GB total)

```
┌─────────────────────────────────────────────────────────────────┐
│              DUAL RTX 3090 + NVLINK (48GB)                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  GPU 0: RTX 3090 (24GB)    NVLink    GPU 1: RTX 3090 (24GB)    │
│  ┌─────────────────────────┐ ════ ┌─────────────────────────┐  │
│  │  TRM+MuZero TRAINING    │      │  GENERATOR INFERENCE    │  │
│  │  - Full model           │ 112  │  - Full precision       │  │
│  │  - Large replay buffer  │ GB/s │  - No quantization      │  │
│  │  - Parallel self-play   │      │  - Larger models OK     │  │
│  │  VRAM: ~20GB            │      │  VRAM: ~20GB available  │  │
│  └─────────────────────────┘      └─────────────────────────┘  │
│                                                                 │
│  NVLink: 112 GB/s bidirectional (vs 32 GB/s PCIe 4.0)         │
│  Can pool memory for 48GB unified (with limitations)           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**NVLink benefits**:
- 3.5x faster than PCIe for GPU-GPU transfer
- Near-zero latency for feedback loop
- Can train larger generator models

### Recommendation by VRAM Target

| Generator Size | Recommended Setup | Notes |
|----------------|-------------------|-------|
| **≤6GB** | Single 3090 | Ideal - no IPC overhead |
| **6-10GB** | 3090 + CMP 90HX | Good use of mining card |
| **>10GB** | Dual 3090 + NVLink | Maximum flexibility |

### Memory Budget Targets

| Component | Target (Aggressive) | Target (Comfortable) |
|-----------|---------------------|----------------------|
| Generator model | 2GB (INT8) | 4GB (FP16) |
| Generator activations | 1GB | 2GB |
| Optimal solver | 0.5GB | 1GB |
| Buffers/overhead | 1GB | 2GB |
| **Total Generator** | **4.5GB** | **9GB** |
| **Fits on** | Single 3090 ✅ | CMP 90HX ✅ |

### Aggressive Optimization Strategies (Target: ≤6GB)

| Technique | Memory Savings | Speed Impact | Complexity |
|-----------|----------------|--------------|------------|
| **INT8 Quantization** | 4x | ~Same | Medium |
| **Flow Matching (1-step)** | ~10x vs DDPM | Faster! | Low |
| **Smaller latent dim** (256 vs 512) | 2x | ~Same | Low |
| **Gradient checkpointing** | 2-3x | 20% slower | Low |
| **Model distillation** | 2-4x smaller | ~Same | High |
| **Streaming generation** | Constant mem | ~Same | Medium |
| **FP16 inference** | 2x | ~Same | Trivial |

**Priority Order**:
1. Flow Matching (biggest win, actually faster)
2. FP16 inference (free)
3. Smaller latent dim if needed
4. INT8 if still tight
5. Distillation as last resort

**Target Architecture**:
```
Generator: ~50M params (FP16) = 100MB weights
           + activations ~500MB
           + buffers ~400MB
           = ~1GB total (very achievable!)
```

With Flow Matching, the generator can be MUCH smaller than DDPM because:
- No iterative denoising (1 step vs 50-1000)
- No noise schedule storage
- Simpler architecture

### Communication Protocol

```python
# Single GPU version (if generator ≤6GB)
class AdaptiveCurriculum:
    """Generator + Trader on same GPU."""

    def __init__(self):
        self.generator = FlowMatchingGenerator()  # ~4GB
        self.trader = TRMMuZeroAgent()            # ~16GB
        self.optimal_solver = HindsightOracle()   # ~0.5GB
        # Total: ~20GB on RTX 3090 (24GB)

    def train_episode(self, difficulty: float):
        # Generate market (GPU, ~1ms)
        prices = self.generator.sample(condition=difficulty)

        # Compute optimal PnL (GPU, <1ms)
        optimal_pnl = self.optimal_solver(prices)

        # Trader executes (GPU, ~3ms)
        actions, actual_pnl = self.trader.execute(prices)

        # Regret-based curriculum update
        regret = optimal_pnl - actual_pnl
        new_difficulty = self.adjust_difficulty(regret)

        # Standard RL update
        self.trader.learn(observations, actions, rewards)

        return regret, new_difficulty

# Multi-GPU version (if generator >6GB)
# Uses CUDA IPC or ZeroMQ for cross-GPU communication
```

### Latency Budget (Target: <10ms per episode)

| Operation | Time | GPU |
|-----------|------|-----|
| Generate sequence | 1-2ms | GPU 1 |
| Optimal solver | <1ms | GPU 1 |
| IPC transfer | 1-2ms | Both |
| Trader execution | 3-5ms | GPU 0 |
| Regret compute | <1ms | GPU 1 |
| **Total** | **~8ms** | - |

---

## Architecture Recommendations (Based on Research)

### Encoder Choice

| Option | Source | Pros | Cons |
|--------|--------|------|------|
| **Transformer** | TRADES, MarS | Best for cross-asset attention | Compute heavy |
| **Wavelet + CNN** | Wavelet-DDPM | Preserves stylized facts | Less flexible |
| **1D-CNN** | FTS-Diffusion | Fast, simple | Limited cross-asset |

**Recommendation**: Transformer encoder with cross-asset self-attention (like TRADES)

### Diffusion Model Choice

| Option | Source | Pros | Cons |
|--------|--------|------|------|
| **DDPM** | TRADES, Wavelet | Well-understood, stable | Slow sampling |
| **Flow Matching** | - | Fast sampling, simpler | Less tested on finance |
| **LMM (Foundation)** | MarS | Scales, flexible | Massive compute needed |

**Recommendation**: Start with DDPM (proven), migrate to Flow Matching for speed

### Conditioning Approach

| Method | How it works | When to use |
|--------|--------------|-------------|
| **Classifier-free guidance** | Train with/without labels | Regime control |
| **Cross-attention** | Inject condition embedding | Fine-grained control |
| **Meta controller** | Separate model for conditions | DigMA approach |

**Recommendation**: Classifier-free guidance for simplicity, like InterDiff paper

### Generation Level

| Level | Examples | Realism | Complexity |
|-------|----------|---------|------------|
| **Price-level** | FTS-Diffusion | Medium | Low |
| **Order-level** | TRADES, MarS, DigMA | High | High |
| **Latent-level** | Our proposed | Medium-High | Medium |

**Recommendation**: Latent-level (encode price data, generate in latent space, decode optional)

### Proposed Final Architecture

```
Historical Data (64 assets × 400 steps × OHLCV)
        ↓
Transformer Encoder (cross-asset attention)
        ↓
Latent Space z (512-dim per timestep, or 512 global)
        ↓
DDPM with Classifier-Free Guidance (regime labels)
        ↓
Generated Latents z'
        ↓
[Option A] Feed directly to TRM (z' as observation)
[Option B] Decode to prices for existing pipeline
```

---

## Related Tasks

- TASK_045: Dojo V2 Curriculum (current synthetic approach)
- TASK_064: Smart Allocation (data distribution)
- TASK_058: Adaptive Noise (curriculum difficulty)

---

## Notes

This is a significant architectural change that could replace or augment the current L0-L14 synthetic curriculum. Start with research phase to validate feasibility before major implementation effort.

Key question: Is the added complexity worth it vs improving hand-crafted generators?

---

## Research Log

### 2026-01-23: Initial Research Complete

**Papers Reviewed:**
1. **TRADES** (arXiv 2502.07071) - Transformer + DDPM for LOB simulation
   - Best template for our architecture
   - Has open-source code (DeepMarket)
   - Order-level generation is more realistic

2. **DigMA/DiGA** (arXiv 2408.12991) - Diffusion guided meta agent
   - **Validates diffusion as RL training environment**
   - Shows A2C training works on generated data
   - Meta controller + meta agent architecture

3. **MarS** (arXiv 2409.07486) - Microsoft's Large Market Model
   - ICLR 2025 accepted
   - Foundation model approach with scaling laws
   - Most production-ready option

4. **FinRL-Meta** - Vectorized environment infrastructure
   - Good patterns for parallel simulation
   - DataOps pipeline design

5. **FTS-Diffusion** (ICLR 2024) - Pattern-based generation
   - Three-prong: recognize → generate → evolve
   - Handles irregular financial patterns

6. **Wavelet-DDPM** - Image-based encoding
   - Novel: time series → wavelet → image → DDPM
   - Preserves stylized facts

**Key Insights:**
- Diffusion models ARE being used for financial RL training (DigMA proves it)
- Order-level generation (TRADES, MarS) is more realistic than price-level
- Scaling laws apply to market models (MarS)
- Cross-asset correlations are under-explored (opportunity for us)
- Most implementations are PyTorch, JAX would be novel

**Recommended Starting Point:**
1. Study DeepMarket code (TRADES implementation)
2. Adapt for 64-asset multi-market setting
3. Add curriculum conditioning
4. Integrate with existing TRM+MuZero agent

**Next Steps:**
- [ ] Clone and study DeepMarket repo
- [ ] Analyze MarS architecture from Microsoft
- [ ] Design latent space for 64-asset encoding
- [ ] Prototype encoder on our historical data

### 2026-01-23: Refined Architecture (Feedback Loop)

**User Requirements:**
- 6GB VRAM inference target (run on secondary GPU)
- 1500 steps/second generation speed
- Full feedback loop: generate → evaluate → adapt

**Key Architecture Decisions:**
1. **Two-GPU setup**: Teacher (6GB) + Student (24GB)
2. **Flow Matching / Consistency Model**: For 1-step generation speed
3. **Regret-based curriculum**: Similar to PAIRED algorithm
4. **Optimal hindsight solver**: Compute max possible PnL

**New Research Areas:**
- PAIRED (Protagonist-Antagonist Induced Regret)
- PLR (Prioritized Level Replay)
- Consistency Models / Flow Matching for fast inference
- INT8 quantization for 6GB target

**Feasibility Assessment:**
- 1500 steps/sec: Achievable with Flow Matching (1-step)
- 6GB VRAM: Achievable with distillation + INT8
- Feedback loop: Standard multi-process Python
- Optimal solver: O(T×N) = trivial compute

**Hardware Options Clarified:**
- If ≤6GB: Single RTX 3090 handles both (best!)
- If ≤10GB: Use CMP 90HX mining card for generator
- If >10GB: Dual 3090 with NVLink bridge (112 GB/s)

**Target**: Aggressive optimization to fit on single 3090

### 2026-01-23: Initial Implementation Complete

**Module Created**: `diff_teacher/` (3821 lines total)

| File | Lines | Description |
|------|-------|-------------|
| `__init__.py` | 37 | Package init with exports |
| `config.py` | 162 | All configuration dataclasses |
| `data_loader.py` | 474 | Database reader with caching |
| `encoder.py` | 582 | Transformer VAE with cross-asset attention |
| `flow_matching.py` | 593 | Flow Matching generator with regime conditioning |
| `optimal_solver.py` | 412 | Hindsight oracle (peak-valley & DP) |
| `regret_curriculum.py` | 443 | PAIRED-style difficulty controller |
| `teacher.py` | 594 | Main orchestration class |
| `train.py` | 524 | CLI training entry point |

**Key Design Decisions:**
1. **Separate module** - Lives in `diff_teacher/`, completely independent from `mcts_trader/`
2. **Shares database** - Reads from same `trading_data.db` but no code dependencies
3. **Flow Matching** - Single-step generation (no DDPM 50-step overhead)
4. **Regime conditioning** - 4 regimes (crash, rally, sideways, volatile) + unconditional
5. **Transformer encoder** - Cross-asset + temporal attention for correlation preservation
6. **JAX-native** - All components implemented in JAX/Flax for GPU efficiency

**Training Command:**
```bash
python -m diff_teacher.train \
    --db-path trading_data.db \
    --n-assets 64 \
    --window-size 400 \
    --encoder-epochs 50 \
    --generator-epochs 100 \
    --batch-size 32
```

**Next Steps:**
1. Train encoder on historical data
2. Train flow matching generator
3. Benchmark generation speed
4. Integrate with trader for curriculum loop

