# TASK_112: Skeleton-Conditioned Market Generation (ControlNet for Markets)

## Status: GROUNDWORK COMPLETE (Phase 2 of Unified Architecture)

## Implementation Progress (2026-01-25)

### Completed: Skeleton Module (`diff_teacher/skeleton/`)

| File | Purpose | Status |
|------|---------|--------|
| `types.py` | MarketSkeleton dataclass, Event, regime constants | DONE |
| `extraction.py` | extract_skeleton(), trend/vol/correlation extractors | DONE |
| `encoder.py` | SkeletonEncoder with FiLM parameter generation | DONE |
| `builder.py` | Fluent SkeletonBuilder API | DONE |

### Key Implementations

**MarketSkeleton dataclass:**
```python
@dataclass
class MarketSkeleton:
    trend_direction: Optional[Array] = None  # {-1, 0, +1}
    vol_regime: Optional[Array] = None       # [0, 1]
    correlation_clusters: Optional[Array] = None
    alpha_shape: Optional[Array] = None
    regime_sequence: Optional[Array] = None
    events: List[Event] = field(default_factory=list)
```

**SkeletonBuilder (fluent API):**
```python
skeleton = SkeletonBuilder() \
    .add_regime('crash', duration=50) \
    .set_volatility_percentile(0.8) \
    .set_correlation_mode('crisis') \
    .build()
```

**SkeletonEncoder (FiLM conditioning):**
- Encodes skeleton components to gamma, beta, h0
- Supports trend, volatility, correlation, alpha shape
- Sinusoidal timestep encoding integrated

### Remaining Work
- [ ] Integrate skeleton conditioning into CNN generator (TASK_111)
- [ ] Train conditioned model
- [ ] Evaluate controllability metrics

---

## Overview

Adapt the **ControlNet paradigm** from image generation to financial time series. The skeleton provides normalized structure tokens; the generator renders them into realistic market data at proper scale.

```
Image:   Canny Edge → ControlNet → Pixels
Market:  Skeleton   → Generator  → OHLCV Prices
```

## Key Insight: Normalized Tokens

Skeleton feeds **structure**, not values:
- Direction vectors, not price levels
- Correlation cluster IDs, not correlation values
- Trend shapes, not actual returns
- Vol regime sequence [0,1], not volatility numbers

Generator learns to:
- Scale to realistic price levels
- Add proper microstructure (wicks, gaps)
- Generate realistic volatility patterns
- Maintain cross-asset dynamics

---

## Three Skeleton Categories

### 1. Macro-Regime Skeleton (Style Constraint)
**What**: Economic weather - VIX level, rate environment, regime label
**Encoding**: Learned embeddings + continuous normalization
**Injection**: FiLM (global scale/shift of features)

```python
macro_skeleton = {
    'regime': 'crash',           # {crash, rally, sideways, volatile, calm}
    'vix_percentile': 0.85,      # [0, 1]
    'rate_regime': 'rising',     # {rising, falling, zero}
}
```

### 2. Structural Skeleton (Trend Constraint)
**What**: Price trajectory shape - moving averages, support/resistance
**Encoding**: Dense continuous signal (like Canny edges)
**Injection**: Initial state + FiLM

```python
structural_skeleton = {
    'trend_direction': Array[n_timesteps, n_assets],  # {-1, 0, +1}
    'alpha_shape': Array[n_timesteps],                 # cumulative opportunity
    'support_levels': List[float],
    'resistance_levels': List[float],
}
```

### 3. Event Skeleton (Sparse Constraint)
**What**: Discrete shocks - earnings, FOMC, flash crashes
**Encoding**: Sparse binary mask (Dirac comb)
**Injection**: Cross-attention (events need precise temporal alignment)

```python
event_skeleton = {
    'events': [
        {'time': 100, 'type': 'earnings', 'magnitude': 0.8},
        {'time': 250, 'type': 'fomc', 'magnitude': 0.3},
    ]
}
```

---

## Conditioning Mechanisms

### Comparison Table

| Mechanism | Best For | Cost | When to Use |
|-----------|----------|------|-------------|
| **FiLM** | Global regime/volatility | O(1) | Macro-regime skeleton |
| **AdaLN** | Timestep + class | O(1) | All conditioning |
| **AdaLN-Zero** | Stable fine-tuning | O(1) | ControlNet branch |
| **Cross-Attention** | Sparse events | O(n·m) | Event skeleton |
| **Initial State** | Persistent context | O(1) | Structural skeleton |

### Recommended Pattern

```python
class SkeletonConditioner(nn.Module):
    d_model: int
    n_regimes: int = 5

    @nn.compact
    def __call__(self, skeleton, t):
        # 1. Addition for scalars (timestep, vol percentile)
        t_emb = sinusoidal_embedding(t, self.d_model)
        vol_emb = nn.Dense(self.d_model)(skeleton.vol_regime[:, None])

        # 2. Learned embeddings for categorical (regime)
        regime_emb = nn.Embed(self.n_regimes, self.d_model)(skeleton.regime_idx)

        # 3. FiLM for sequential (history context)
        combined = t_emb + vol_emb + regime_emb
        gamma, beta = nn.Dense(self.d_model * 2)(combined).split(2, axis=-1)

        # 4. Initial state for Mamba
        h0 = nn.Dense(self.d_model)(combined)

        return gamma, beta, h0
```

### Zero Convolution for ControlNet

Preserve pretrained model dynamics during fine-tuning:

```python
class ZeroConv1D(nn.Module):
    """1x1 conv initialized to zero - gradients still flow!"""
    features: int

    @nn.compact
    def __call__(self, x):
        kernel = self.param('kernel', nn.initializers.zeros,
                           (1, x.shape[-1], self.features))
        bias = self.param('bias', nn.initializers.zeros, (self.features,))
        return lax.conv_general_dilated(x, kernel, (1,), 'SAME') + bias
```

---

## Skeleton Extraction from Real Data

```python
def extract_skeleton(prices: Array, window: int = 20) -> MarketSkeleton:
    """Extract skeleton from historical data for training."""
    returns = jnp.diff(jnp.log(prices), axis=0)

    # Trend direction (per-asset, time-varying)
    trend_direction = jnp.sign(rolling_mean(returns, window))

    # Volatility regime (normalized)
    vol = rolling_std(returns, window)
    vol_regime = vol / vol.max()

    # Correlation clusters (static, from full period)
    corr_matrix = jnp.corrcoef(returns.T)
    correlation_clusters = spectral_clustering(corr_matrix, n_clusters=5)

    # Alpha shape (cumulative optimal return)
    optimal_return = jnp.cumsum(jnp.abs(returns).max(axis=1))
    alpha_shape = optimal_return / optimal_return[-1]

    # Regime sequence (HMM-based)
    regime_sequence = hmm_regime_detection(returns, n_states=5)

    return MarketSkeleton(
        trend_direction=trend_direction,
        vol_regime=vol_regime,
        correlation_clusters=correlation_clusters,
        alpha_shape=alpha_shape,
        regime_sequence=regime_sequence,
    )
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  SKELETON ENCODER                           │
├─────────────────────────────────────────────────────────────┤
│ trend_encoder:   Linear → SiLU → Linear                     │
│ cluster_encoder: Embedding(n_clusters, d_model)             │
│ vol_encoder:     Linear → SiLU → Linear                     │
│ alpha_encoder:   Conv1D (captures shape)                    │
│ combiner:        Concat → Linear → LayerNorm                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              SKELETON-CONDITIONED GENERATOR                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   noise ──┬──→ Input Projection ──→ h                       │
│           │                                                 │
│   skeleton ──→ Skeleton Encoder ──→ cond                    │
│                                                             │
│   for layer in layers:                                      │
│       h = h + FiLM(MambaBlock(h), cond)  # or CNN           │
│       h = LayerNorm(h)                                      │
│                                                             │
│   output = Linear(h)                                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Training Strategy

### Phase 1: Base Generator (TASK_111) - IN PROGRESS
Train unconditional flow matching on historical data:
- Generator learns market "physics" and realistic patterns
- No skeleton yet
- **Current status**: 768d/12L model, loss ~0.5

### Phase 2: Skeleton Extraction
Build extraction pipeline:
- Extract skeletons from all training periods
- Build vocabulary of real skeleton patterns
- Store skeleton alongside price data

### Phase 3: Conditional Training
Fine-tune with skeleton conditioning:
```python
skeleton = extract_skeleton(real_prices)
generated = generator(noise, skeleton_embedding, t)
loss = flow_matching_loss(generated, real_prices)
```

Options:
- **From scratch**: Train conditioned model end-to-end
- **ControlNet-style**: Freeze base, add skeleton branch with zero convs

### Phase 4: Skeleton Designer Interface
```python
skeleton = SkeletonBuilder() \
    .add_regime('sideways', duration=100) \
    .add_regime('crash', duration=50, magnitude=0.3) \
    .add_regime('recovery', duration=150) \
    .set_correlation_clusters([[0,1,2], [3,4,5]]) \
    .set_alpha_opportunities([(50, 0.1), (200, -0.05)]) \
    .build()

# Generate multiple realizations
prices = [generate(skeleton, seed=i) for i in range(10)]
```

---

## Use Cases

### 1. Difficulty Control for RL Curriculum
```python
# Easy: Clear trends, low correlation
easy = {'trend': [+1]*64, 'corr_clusters': range(64)}

# Hard: Noisy, high correlation, brief alpha windows
hard = {'trend': [0]*64, 'corr_clusters': [0]*64, 'alpha_window': 10}
```

### 2. Stress Testing
```python
crash_test = skeleton_with_regime('crash', magnitude=0.5)
whipsaw_test = skeleton_with_regimes(['rally', 'crash'] * 4)
```

### 3. Robustness Testing
```python
# Same skeleton, different noise → agent should perform consistently
for seed in range(100):
    prices = generate(skeleton, seed=seed)
    reward = evaluate_agent(prices)
```

---

## Evaluation Metrics

### Controllability (Primary)
- **Regime adherence**: Classification accuracy on generated data
- **Volatility targeting**: RMSE(target_vol, realized_vol)
- **Correlation matching**: ‖Σ_target - Σ_generated‖_F
- **J-FTSD**: Joint Fréchet Time Series Distance

### Fidelity (Secondary)
- Stylized facts preserved despite conditioning
- Generated data still looks realistic

### Diversity
- Same skeleton → varied but consistent realizations
- Vendi Score for measuring variation

---

## Implementation Steps

### Step 1: Skeleton Data Structure
```python
@dataclass
class MarketSkeleton:
    trend_direction: Array      # [n_timesteps, n_assets] in {-1, 0, +1}
    vol_regime: Array           # [n_timesteps] in [0, 1]
    correlation_clusters: Array # [n_assets] cluster IDs
    alpha_shape: Array          # [n_timesteps] in [0, 1]
    regime_sequence: Array      # [n_segments] regime labels
    events: List[Event]         # sparse events
```

### Step 2: Extraction Pipeline
- [ ] `extract_skeleton()` function
- [ ] HMM regime detection
- [ ] Correlation clustering
- [ ] Event detection (outliers, volume spikes)

### Step 3: Skeleton Encoder Module
- [ ] Per-component encoders
- [ ] Combiner network
- [ ] FiLM parameter generation

### Step 4: Conditional Generator
- [ ] Modify existing CNN/Mamba to accept skeleton
- [ ] Add FiLM layers at each block
- [ ] Optional: ControlNet branch with zero convs

### Step 5: Training Loop
- [ ] Extract skeleton → train with conditioning
- [ ] Validate controllability metrics

---

## Relation to Other Tasks

| Task | Role |
|------|------|
| TASK_111 | Base generator (CNN) - learns market physics |
| TASK_112 | This task - adds controllability via skeleton |
| TASK_113 | Mamba backbone - more efficient for long sequences |

TASK_111 CNN is the **foundation** - it learns market physics.
TASK_112 adds **controllability** - you specify what market to generate.
TASK_113 provides **efficiency** - O(n) complexity for scaling.

---

## Publication Potential

This is **HuggingFace-publishable** research:
- Novel application of ControlNet to finance
- Useful for stress testing, scenario analysis
- Not alpha extraction (skeleton doesn't predict, it controls)
- Clear contribution to research community

---

## References

- ControlNet (Zhang et al., 2023)
- ControlNet-XS (feedback-control rethinking)
- T2I-Adapter (lightweight conditioning)
- FiLM (Perez et al., 2018)
- AdaLN from DiT (Peebles & Xie, 2023)
