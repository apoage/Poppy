# Unified Market Generation Architecture: Skeleton-Mamba-Flow

## Executive Summary

This document synthesizes research on controllable financial time series generation, consolidating insights from multiple architectural explorations into a cohesive blueprint for building a **Skeleton-Conditioned Market Generator**.

**Core Innovation**: Combine selective State Space Models (Mamba) with Flow Matching and ControlNet-style conditioning to create a generator that is:
- **Controllable**: Skeleton inputs dictate market regime, volatility, correlation structure
- **Efficient**: O(n) complexity enables 10K+ timestep sequences
- **Fast**: Flow Matching with ODE solvers enables few-step generation
- **Publishable**: Useful for research community, not a trading alpha leak

```
┌─────────────────────────────────────────────────────────────────┐
│                    SKELETON-MAMBA-FLOW                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Skeleton Input          Noise Input                           │
│   (regime, vol, corr)     (Gaussian)                           │
│         │                      │                                │
│         ▼                      ▼                                │
│   ┌──────────────┐      ┌──────────────┐                       │
│   │   Skeleton   │      │   Noisy      │                       │
│   │   Encoder    │      │   Prices     │                       │
│   └──────┬───────┘      └──────┬───────┘                       │
│          │                     │                                │
│          └──────────┬──────────┘                                │
│                     ▼                                           │
│          ┌──────────────────────┐                              │
│          │  BiMamba Backbone    │  ← FiLM/AdaLN conditioning   │
│          │  (1:7 attention)     │                              │
│          └──────────┬───────────┘                              │
│                     │                                           │
│                     ▼                                           │
│          ┌──────────────────────┐                              │
│          │  Flow Matching Head  │  ← Velocity prediction       │
│          └──────────┬───────────┘                              │
│                     │                                           │
│                     ▼                                           │
│          ┌──────────────────────┐                              │
│          │  Diffrax ODE Solver  │  ← Euler (fast) / Dopri5     │
│          └──────────┬───────────┘                              │
│                     │                                           │
│                     ▼                                           │
│            Generated OHLCV                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 1. The Financial Generation Trilemma

Any market generator must balance three competing objectives:

| Objective | Challenge | Our Solution |
|-----------|-----------|--------------|
| **Fidelity** | Reproduce stylized facts (fat tails, vol clustering) | Mamba's selective memory captures long-range dependencies |
| **Controllability** | Dictate regime, volatility, correlation | Multi-modal skeleton conditioning via FiLM + AdaLN |
| **Scalability** | Handle 10K+ timesteps efficiently | O(n) SSM complexity + parallel associative scan |

---

## 2. Skeleton Types and Encoding

### 2.1 Three Skeleton Categories

| Type | Description | Encoding | Injection Method |
|------|-------------|----------|------------------|
| **Macro-Regime** | Economic weather (VIX, rates, regime label) | Learned embeddings + continuous normalization | FiLM (global scale/shift) |
| **Structural** | Price trajectory shape (trend, support/resistance) | Dense continuous signal | Initial state + FiLM |
| **Event** | Discrete shocks (earnings, FOMC) | Sparse binary mask | Cross-attention |

### 2.2 Skeleton Components (Normalized Tokens)

```python
@dataclass
class MarketSkeleton:
    # Per-asset, time-varying
    trend_direction: Array[n_timesteps, n_assets]  # {-1, 0, +1}

    # Global, time-varying
    vol_regime: Array[n_timesteps]  # [0, 1] normalized

    # Per-asset, static
    correlation_clusters: Array[n_assets]  # cluster IDs

    # Global, time-varying
    alpha_shape: Array[n_timesteps]  # cumulative opportunity curve

    # Sparse events
    events: List[Event]  # {time, type, magnitude}

    # Discrete regime sequence
    regime_sequence: Array[n_segments]  # {crash, rally, sideways, volatile, calm}
```

### 2.3 Skeleton Extraction from Real Data

```python
def extract_skeleton(prices: Array, window: int = 20) -> MarketSkeleton:
    returns = jnp.diff(jnp.log(prices), axis=0)

    return MarketSkeleton(
        trend_direction=jnp.sign(rolling_mean(returns, window)),
        vol_regime=rolling_std(returns, window) / rolling_std(returns, window).max(),
        correlation_clusters=cluster_correlation_matrix(returns),
        alpha_shape=cumulative_optimal_return(prices) / max_return,
        events=detect_outliers(returns, threshold=3.0),
        regime_sequence=hmm_regime_detection(returns, n_states=5),
    )
```

---

## 3. Backbone Architecture: BiMamba with Sparse Attention

### 3.1 Why Mamba over Transformer

| Aspect | Transformer | Mamba | Winner |
|--------|-------------|-------|--------|
| Complexity | O(n²) | O(n) | Mamba |
| Long-range | Global (expensive) | Global (cheap) | Mamba |
| In-context learning | Excellent | Limited | Transformer |
| Memory efficiency | KV cache grows | Constant state | Mamba |

**Solution**: Hybrid with **1:7 attention-to-Mamba ratio** (Jamba pattern)

### 3.2 Selective State Space (The Core Innovation)

Standard SSM uses fixed A, B, C matrices. Mamba makes them **input-dependent**:

```python
# Traditional SSM (fixed filter)
h_t = A @ h_{t-1} + B @ x_t
y_t = C @ h_t

# Mamba (selective - input-dependent)
delta_t = softplus(Linear(x_t))  # Controls update rate
B_t = Linear(x_t)                 # Input gate
C_t = Linear(x_t)                 # Output gate

h_t = exp(delta_t * A) @ h_{t-1} + delta_t * B_t @ x_t
y_t = C_t @ h_t
```

**Financial Interpretation**:
- Large δ → Focus on new input (react to shock)
- Small δ → Rely on memory (trust the trend)
- Like an **adaptive Kalman filter**

### 3.3 JAX Implementation with Parallel Scan

```python
import jax
import jax.numpy as jnp
from jax import lax

def selective_scan_parallel(x, delta, A, B, C):
    """
    Parallel associative scan for Mamba.
    O(log L) depth instead of O(L) sequential.
    """
    batch, seq, d_inner = x.shape
    d_state = A.shape[-1]

    # Discretize
    delta_A = jnp.exp(jnp.einsum('bld,dn->bldn', delta, A))
    delta_B_x = jnp.einsum('bld,bln,bld->bldn', delta, B, x)

    # Binary operator for associative scan
    def binary_op(carry, element):
        A_prev, h_prev = carry
        A_curr, Bx_curr = element
        return (A_curr * A_prev, A_curr * h_prev + Bx_curr)

    # Parallel scan
    _, h = lax.associative_scan(
        binary_op,
        (delta_A.transpose(1,0,2,3), delta_B_x.transpose(1,0,2,3)),
        axis=0
    )
    h = h.transpose(1,0,2,3)

    # Output
    y = jnp.einsum('bldn,bln->bld', h, C)
    return y
```

### 3.4 BiMamba for Diffusion/Flow Matching

Generative models need bidirectional context (not just causal):

```python
class BiMambaBlock(nn.Module):
    d_model: int
    d_state: int = 64  # Mamba-2 uses larger state

    @nn.compact
    def __call__(self, x, cond=None):
        # Forward pass
        h_fwd = MambaBlock(self.d_model, self.d_state)(x, cond)

        # Backward pass (reverse sequence)
        x_rev = jnp.flip(x, axis=1)
        h_bwd = MambaBlock(self.d_model, self.d_state)(x_rev, cond)
        h_bwd = jnp.flip(h_bwd, axis=1)

        # Fuse
        h = nn.Dense(self.d_model)(jnp.concatenate([h_fwd, h_bwd], axis=-1))
        return h
```

### 3.5 Hybrid Architecture (Jamba-style)

```python
class HybridMambaAttention(nn.Module):
    n_layers: int = 24
    d_model: int = 768
    attention_every: int = 8  # 1:7 ratio

    @nn.compact
    def __call__(self, x, cond, t):
        for i in range(self.n_layers):
            # AdaLN conditioning from timestep
            scale, shift, gate = self.adaln_modulation(t, cond)

            if i % self.attention_every == 0 and i > 0:
                # Sparse attention layer (every 8th)
                x = x + gate * SelfAttention(self.d_model)(
                    self.layer_norm(x) * (1 + scale) + shift
                )
            else:
                # Mamba layer (7 out of 8)
                x = x + gate * BiMambaBlock(self.d_model)(
                    self.layer_norm(x) * (1 + scale) + shift,
                    cond
                )
        return x
```

---

## 4. Conditioning Mechanisms

### 4.1 Comparison

| Mechanism | Best For | Cost | Implementation |
|-----------|----------|------|----------------|
| **FiLM** | Global regime/volatility | O(1) | `y = gamma(z) * x + beta(z)` |
| **AdaLN** | Timestep + class | O(1) | `y = (1 + scale(z)) * LN(x) + shift(z)` |
| **AdaLN-Zero** | Stable training | O(1) | Initialize projections to zero |
| **Cross-Attention** | Sparse events, alpha signals | O(n·m) | Q from features, K/V from skeleton |
| **Initial State** | Persistent context | O(1) | `h_0 = Linear(condition)` |

### 4.2 Multi-Modal Conditioning Pipeline

```python
class SkeletonConditioner(nn.Module):
    d_model: int
    n_regimes: int = 5

    @nn.compact
    def __call__(self, skeleton: MarketSkeleton, t: float):
        # Regime embedding (categorical)
        regime_emb = nn.Embed(self.n_regimes, self.d_model)(skeleton.regime_sequence)

        # Volatility encoding (continuous)
        vol_emb = nn.Dense(self.d_model)(skeleton.vol_regime[:, None])
        vol_emb = nn.silu(vol_emb)

        # Correlation structure (via small GNN or MLP on cluster assignments)
        corr_emb = nn.Dense(self.d_model)(
            nn.one_hot(skeleton.correlation_clusters, num_classes=10)
        )

        # Time embedding (sinusoidal)
        t_emb = sinusoidal_embedding(t, self.d_model)

        # Combine
        combined = regime_emb.mean(0) + vol_emb.mean(0) + corr_emb.mean(0) + t_emb

        # Generate FiLM parameters
        film_params = nn.Dense(self.d_model * 2)(combined)
        gamma, beta = jnp.split(film_params, 2, axis=-1)

        # Generate initial state for Mamba
        h0 = nn.Dense(self.d_model)(combined)

        return gamma, beta, h0
```

### 4.3 ControlNet-Style Zero Convolution

For fine-tuning pretrained models without destroying learned dynamics:

```python
class ZeroConv1D(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        # Initialize weights AND biases to zero
        kernel = self.param('kernel', nn.initializers.zeros, (1, x.shape[-1], self.features))
        bias = self.param('bias', nn.initializers.zeros, (self.features,))
        return lax.conv_general_dilated(x, kernel, (1,), 'SAME') + bias
```

---

## 5. Flow Matching (Not Diffusion)

### 5.1 Why Flow Matching over DDPM

| Aspect | DDPM | Flow Matching |
|--------|------|---------------|
| Training target | Noise prediction | Velocity prediction |
| Paths | Curved (stochastic) | Straight (deterministic) |
| Sampling steps | 50-1000 | 1-10 |
| Numerical stability | Score can explode | Velocity is bounded |

### 5.2 Conditional Flow Matching (CFM) Loss

```python
def flow_matching_loss(model, params, x_1, skeleton, key):
    """
    OT-CFM: Optimal Transport Conditional Flow Matching
    """
    batch = x_1.shape[0]

    # Sample noise and time
    x_0 = jax.random.normal(key, x_1.shape)
    t = jax.random.uniform(jax.random.fold_in(key, 1), (batch, 1, 1))

    # Interpolate (straight path)
    x_t = (1 - t) * x_0 + t * x_1

    # Target velocity (direction to data)
    target_v = x_1 - x_0

    # Predict velocity
    pred_v = model.apply(params, x_t, t.squeeze(), skeleton)

    # MSE loss
    return jnp.mean((pred_v - target_v) ** 2)
```

### 5.3 Sampling with Diffrax

```python
import diffrax

def generate(model, params, skeleton, key, n_steps=10):
    """Generate market data via ODE integration."""

    # Initial noise
    x_0 = jax.random.normal(key, (batch, seq_len, n_features))

    # Define vector field
    def vector_field(t, x, args):
        return model.apply(params, x, t, skeleton)

    # Solve ODE from t=0 (noise) to t=1 (data)
    solver = diffrax.Euler()  # Fast for straight paths
    # solver = diffrax.Dopri5()  # Adaptive for complex conditioning

    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(vector_field),
        solver,
        t0=0.0,
        t1=1.0,
        dt0=1.0 / n_steps,
        y0=x_0,
    )

    return solution.ys[-1]
```

---

## 6. Data Pipeline

### 6.1 Preprocessing Protocol

```python
def preprocess_for_generation(prices: Array) -> Tuple[Array, Array]:
    """
    Separate normalized residuals from volatility process.
    Model generates both channels.
    """
    # Log returns
    returns = jnp.diff(jnp.log(prices), axis=0)

    # GARCH-style volatility estimation (simplified)
    vol = exponential_moving_std(returns, alpha=0.94)

    # Normalized residuals
    z = returns / jnp.maximum(vol, 1e-8)

    # Multi-channel output: [residuals, volatility]
    return jnp.stack([z, vol], axis=-1)
```

### 6.2 Augmentation Strategies

| Strategy | Implementation | Rationale |
|----------|---------------|-----------|
| **Time reversal** | `x_aug = x[::-1]` | Patterns in rallies mirror selloffs |
| **Time dilation** | Resample with interpolation | Same pattern at different speeds |
| **Magnitude scaling** | `x_aug = x * scale` | Volatility-adjusted equivalence |
| **Regime oversampling** | Upsample crash/crisis windows | Rare but critical scenarios |

---

## 7. Evaluation Framework

### 7.1 Tier 1: Statistical Fidelity (Stylized Facts)

| Metric | Formula | Target |
|--------|---------|--------|
| **Fat tails** | Hill estimator α | Match real (α ≈ 3) |
| **Vol clustering** | ACF(r²) at lags 1,5,10,20 | Positive slow decay |
| **Leverage effect** | Corr(r_t, σ²_{t+1}) | Negative (≈ -0.3) |
| **Correlation consistency** | ‖Σ_real - Σ_syn‖_F | Minimize |

### 7.2 Tier 2: Controllability

| Metric | Description |
|--------|-------------|
| **J-FTSD** | Joint Fréchet Time Series Distance (conditioning specificity) |
| **Regime adherence** | Classification accuracy on generated data |
| **Volatility targeting** | RMSE(target_vol, realized_vol) |

### 7.3 Tier 3: Downstream Utility

```python
def evaluate_sim_to_real_gap(generator, rl_agent, real_data, n_episodes=1000):
    """
    The ultimate test: does synthetic data improve RL training?
    """
    # Train on synthetic
    synthetic_data = [generator.sample() for _ in range(n_episodes)]
    agent_syn = train_agent(rl_agent, synthetic_data)
    sharpe_syn = evaluate(agent_syn, synthetic_data)

    # Test on real
    sharpe_real = evaluate(agent_syn, real_data)

    # Sim-to-Real Gap
    gap_ratio = sharpe_real / sharpe_syn

    # Interpretation:
    # 0.8-1.0: Faithful simulator
    # < 0.5: Agent exploited simulator artifacts
    # > 1.0: Simulator was harder (domain randomization success)

    return gap_ratio
```

### 7.4 TSTR / TRTS Protocol

- **TSTR** (Train Synthetic, Test Real): If error ≈ TRTR, synthetic has valid signals
- **TRTS** (Train Real, Test Synthetic): If error too low, synthetic is "too easy" (mode collapse)

---

## 8. Implementation Phases

### Phase 1: CNN Baseline (TASK_111) ✅ IN PROGRESS
- [x] Dilated causal conv generator
- [x] Integration with diff_teacher training
- [x] 93M model training (768d/12L)
- [ ] Quality evaluation

### Phase 2: Skeleton Extraction (TASK_112)
- [ ] Implement skeleton extraction functions
- [ ] Build skeleton encoder
- [ ] Add FiLM conditioning to existing CNN
- [ ] Train skeleton-conditioned generator
- [ ] Validate controllability

### Phase 3: Mamba Backbone (TASK_113)
- [ ] Implement MambaBlock in pure JAX
- [ ] BiMamba for bidirectional generation
- [ ] Hybrid architecture (1:7 attention ratio)
- [ ] Profile vs CNN baseline
- [ ] Multi-GPU data parallelism

### Phase 4: Flow Matching + Diffrax
- [ ] Replace noise prediction with velocity prediction
- [ ] Integrate Diffrax ODE solver
- [ ] Benchmark sampling speed
- [ ] Few-step generation (1-10 steps)

### Phase 5: Publication Preparation
- [ ] Clean standalone codebase
- [ ] Comprehensive evaluation suite
- [ ] Paper / technical report
- [ ] HuggingFace model release

---

## 9. HuggingFace Publication Strategy

### What to Publish

| Component | Publish? | Rationale |
|-----------|----------|-----------|
| Generator architecture | ✅ Yes | Novel research contribution |
| Trained weights | ✅ Yes | Enables reproducibility |
| Skeleton extraction code | ✅ Yes | Useful for community |
| Training data | ⚠️ Subset | Can't publish full DB |
| Evaluation suite | ✅ Yes | Establishes benchmarks |
| RL integration | ❌ No | Part of trading system |

### Repository Structure

```
market-flow-generator/
├── README.md
├── src/
│   ├── models/
│   │   ├── mamba_block.py
│   │   ├── flow_matching.py
│   │   └── skeleton_encoder.py
│   ├── data/
│   │   ├── skeleton_extraction.py
│   │   └── preprocessing.py
│   └── evaluation/
│       ├── stylized_facts.py
│       └── controllability.py
├── configs/
├── notebooks/
│   └── demo.ipynb
└── checkpoints/
    └── market_flow_93m/
```

### Positioning

- **Title**: "Market-Flow: Skeleton-Conditioned Financial Time Series Generation with State Space Models"
- **Angle**: ControlNet for markets + Mamba efficiency
- **Not**: Alpha extraction, trading strategy, profit generation
- **Contribution**: Controllable simulation for stress testing, RL training, scenario analysis

---

## 10. References

### Core Papers
- Mamba: Linear-Time Sequence Modeling (Gu & Dao, 2023)
- Mamba-2: State Space Duality (Dao & Gu, 2024)
- Jamba: Hybrid Transformer-Mamba (AI21, 2024)
- ControlNet (Zhang et al., 2023)
- Flow Matching (Lipman et al., 2023)

### Financial Time Series
- TimeGAN (Yoon et al., 2019)
- TSDiff (Kollovieh et al., 2023)
- Stylized Facts of Financial Time Series (Cont, 2001)

### Tools
- Diffrax: JAX differential equation solvers
- mamba-jax: Community JAX implementation
- Flax: Neural network library for JAX
