# Skeleton-conditioned financial market generation: A technical blueprint

**Financial time series generation has advanced dramatically through conditional diffusion and flow matching models, with ControlNet-style conditioning and Mamba state space models emerging as the most promising architectural choices.** This synthesis addresses how to build a skeleton-conditioned market generator that can control regime, volatility, and correlation structure while achieving computational efficiency for long sequences. The key insight: combining selective SSMs with minimal attention layers and FiLM/AdaLN conditioning creates models that are both expressive and tractable for financial applications where sequences routinely exceed 10,000 timesteps.

---

## ControlNet conditioning adapted for temporal sequences

ControlNet's genius lies in its "zero convolution" technique—1×1 convolutions initialized with both weights and biases at zero. Before training begins, these layers output zeros, ensuring **no distortion** to pretrained model weights. Despite zero initialization, gradients flow because ∂L/∂W = x^T · ∂L/∂y ≠ 0 when input x ≠ 0. This allows conditioning signals to gradually "grow" their influence.

The architecture creates a **trainable copy** of the encoder that processes conditioning inputs, with zero convolutions connecting its outputs to the frozen decoder at each resolution level:

```python
# ControlNet connection formula
y_c = F(x; Θ) + Z(F(x + Z(c; Θ_z1); Θ_c); Θ_z2)
# F = frozen model, Z = zero convolutions, c = conditioning signal
```

For time series adaptation, the key architectural change is replacing 2D convolutions with 1D temporal convolutions. The conditioning signal (regime labels, volatility targets, correlation structure) passes through a small encoder network before being injected via zero convolutions. This preserves the pretrained model's learned dynamics while enabling fine-grained control.

**Alternative conditioning approaches offer different trade-offs.** T2I-Adapter achieves similar conditioning with only **77M parameters** versus ControlNet's 1.4B by using lightweight adapter modules that add features directly to encoder outputs. IP-Adapter introduces decoupled cross-attention for conditioning, processing conditioning inputs through separate K/V projections that combine additively with the main attention output. GLIGEN uses gated self-attention with learnable scalars initialized to zero, enabling spatial grounding without modifying base model weights.

### Conditioning mechanism comparison for financial time series

The choice of conditioning mechanism significantly impacts both model capacity and computational cost. Three dominant approaches have emerged for injecting conditioning signals:

**FiLM (Feature-wise Linear Modulation)** applies affine transformations to intermediate features: `FiLM(x) = γ(z) ⊙ x + β(z)`, where γ and β are predicted from conditioning input z via learned MLPs. Best practice places FiLM layers **after normalization** and applies the same modulation across spatial/temporal locations. FiLM excels for global conditions (regime, volatility level) but lacks fine-grained control.

**Cross-attention conditioning** (as in Stable Diffusion) computes Q from features and K/V from conditioning embeddings. This enables **fine-grained alignment** between conditioning sequences and generated content—essential for text-to-image but potentially overkill for simpler financial conditions. Memory scales with batch × heads × sequence_length × conditioning_length.

**Adaptive Layer Normalization (AdaLN)** from DiT integrates conditioning into normalization layers, predicting scale/shift/gate parameters from timestep and class embeddings. The **AdaLN-Zero variant** initializes projection layers to output zeros, making each transformer block start as an identity function for stable training. This approach dominates modern diffusion transformers due to efficiency (constant cost regardless of conditioning complexity).

For heterogeneous financial inputs combining categorical regimes, continuous volatility targets, and sequential history, the recommended pattern is:
- **Addition** for scalar conditions (timestep, volatility percentile)
- **Learned embeddings** for categorical conditions (regime labels)
- **Cross-attention or FiLM** for sequential conditions (historical context)

---

## Extracting market structure for conditioning signals

Effective conditioning requires extracting interpretable structure from financial time series. Four categories of structural features serve as natural conditioning signals.

### Regime detection creates discrete market states

**Hidden Markov Models** remain the gold standard for regime detection, modeling market states as latent variables influencing observable returns. A 2-3 state model captures essential dynamics: low/high volatility or bull/bear/sideways regimes. The emission distributions are typically Gaussian for returns, with transition matrices learning regime persistence and switching probabilities.

```python
from hmmlearn.hmm import GaussianHMM
model = GaussianHMM(n_components=3, covariance_type="full", n_iter=1000)
model.fit(features)  # features: [returns, range, volume_ratio]
states = model.predict(features)  # 0, 1, or 2 per timestep
state_probs = model.predict_proba(features)  # soft assignment for conditioning
```

**Change point detection** via the PELT algorithm identifies structural breaks with O(n) complexity. For financial data, use the "normal" cost function detecting both mean and variance shifts. The ruptures library provides production-ready implementation with configurable penalty parameters controlling detection sensitivity.

**Clustering on rolling statistics** offers a simpler alternative: compute rolling features (mean, volatility, skewness, kurtosis over 20-day windows), standardize, and apply K-means or GMM clustering. This approach is more interpretable and allows custom feature engineering.

### Volatility regimes require multi-scale analysis

GARCH models provide conditional volatility estimates that can be clustered into regimes. More practically, realized volatility at multiple scales (**5, 20, 60-day windows**) creates a multi-resolution feature vector for regime classification. VIX-based thresholds (< 15 low, 15-25 normal, 25-35 elevated, > 35 crisis) offer simple but effective discretization.

Jump detection using bipower variation separates continuous volatility from discrete jumps—critical for capturing flash crashes and extreme events that pure volatility measures miss.

### Correlation structure as conditioning signal

Rolling correlation matrices capture time-varying dependence structure. For conditioning, extract:
- **Average correlation** (scalar): market stress indicator
- **PCA on flattened correlations**: captures correlation regime shifts
- **Minimum spanning tree properties**: network topology metrics like diameter and average path length

Change point detection on PCA-transformed correlation features identifies correlation regime shifts—essential for portfolio applications where correlation breakdown during crises is a key risk factor.

### Creating normalized conditioning vectors

Convert extracted structures into model inputs through:

1. **One-hot encoding** for discrete regimes (3 states → [1,0,0], [0,1,0], [0,0,1])
2. **Learned embeddings** when regime semantics should be discovered during training
3. **Continuous skeleton vectors** combining normalized features: trend strength, volatility percentile, momentum, mean-reversion indicators
4. **Multi-resolution encoding** extracting features at scales [5, 20, 60, 120 days] then concatenating or applying PCA

Temporal alignment is critical: rolling features produce NaN values for initial windows. Use forward-fill or interpolation to align conditioning signals with raw data for training.

---

## Mamba architecture enables linear-complexity generation

### The selective state space innovation

Mamba's core innovation transforms standard SSMs into **selective** (input-dependent) models. The continuous-time SSM equations:

```
x'(t) = A·x(t) + B·u(t)  [state equation]
y(t) = C·x(t)            [output equation]
```

are discretized and made content-aware by computing **Δ, B, and C as functions of the input**:

```python
Δ = softplus(Linear_Δ(x) + dt_bias)  # discretization step
B = Linear_B(x)                       # input matrix (selective)
C = Linear_C(x)                       # output matrix (selective)
```

This selection mechanism allows Mamba to **selectively propagate or forget information** based on current input—unlike prior SSMs that processed all inputs uniformly. Small Δ values emphasize previous context (forget gate), while large Δ values emphasize current input.

### Parallel scan enables efficient training

The recurrence h_t = Ā·h_{t-1} + B̄·x_t appears sequential but exploits associativity for parallel computation. Define binary operator:

```
(a₁, b₁) ⊕ (a₂, b₂) = (a₂·a₁, a₂·b₁ + b₂)
```

This operator is **associative**, enabling parallel scan with O(log L) depth instead of O(L) sequential steps. JAX provides `jax.lax.associative_scan` for direct implementation:

```python
def binary_op(carry, element):
    A_prev, h_prev = carry
    A_curr, Bx_curr = element
    return (A_curr * A_prev, A_curr * h_prev + Bx_curr)

_, h = lax.associative_scan(binary_op, (A_bar, Bx), axis=1)
y = jnp.einsum('bldn,bln->bld', h, C)
```

### JAX implementation considerations

Porting Mamba from CUDA to JAX involves several challenges. The official implementation uses fused CUDA kernels combining discretization, scan, and output computation in single GPU operations with explicit SRAM management. JAX relies on XLA compiler optimization, resulting in **1.5-3× slower performance** without custom kernels.

Key optimizations for JAX:
- **Chunked processing**: Process sequences in 256-token chunks to fit intermediate states in cache
- **Gradient checkpointing**: Use `@jax.checkpoint` to trade compute for memory, achieving O(1) activation memory
- **Pallas kernels**: For production performance, custom Pallas kernels can approach CUDA efficiency (still experimental as of early 2025)

Available JAX implementations include **mamba-jax** (Equinox models with associative scan), **mamba2-jax** (Mamba-2 with CPU/GPU/TPU support), and minimal implementations for educational purposes.

### Mamba-2 advances the architecture

Mamba-2 (2024) introduces the **State Space Duality (SSD)** framework showing that selective SSMs and linear attention are mathematically equivalent through semiseparable matrix decompositions. Key improvements:

- **Larger state size**: N = 64-256 versus Mamba-1's N = 16
- **Chunked algorithm**: 2-8× faster through better hardware utilization
- **Parallel projections**: All data-dependent parameters computed at block start

---

## Hybrid architectures combine SSM efficiency with attention flexibility

### Jamba establishes the hybrid paradigm

AI21's Jamba combines Transformer layers, Mamba layers, and Mixture-of-Experts with a **1:7 attention-to-Mamba ratio**. The critical finding: **never place Transformer blocks at the front**. Initial Mamba layers efficiently compress long sequences into hidden states; sparse Transformer layers placed later perform expensive global attention on processed representations.

This creates computational efficiency: Jamba's KV cache is only **4GB at 256K context** versus Mixtral's 32GB. The model achieves 3× throughput while maintaining quality competitive with pure Transformers.

### Zamba pushes minimal attention further

Zyphra's Zamba series demonstrates that "one attention layer is all you need." Zamba2 uses:
- Mamba2 backbone with **two alternating shared attention blocks** (1:6 ratio)
- **LoRA projectors** on shared MLP blocks for depth-specialization with minimal parameters
- Original embeddings concatenated to attention block inputs (improves information maintenance)

This achieves state-of-the-art performance for sub-3B models with 6× reduction in KV cache versus transformers.

### Design principles for financial applications

| Component | Best For | Limitations |
|-----------|----------|-------------|
| **Mamba/SSM** | Efficient sequence scanning, local pattern compression, streaming inference | Limited in-context learning, format adherence |
| **Attention** | Global context, precise information routing, in-context recall | Quadratic complexity, KV cache growth |

For financial time series generation, a hybrid with **1:6 to 1:8 attention-to-SSM ratio** balances efficiency and expressiveness. Place attention layers in later stages after SSM layers have compressed sequence information.

---

## SSM-based diffusion and flow matching models

### DiS establishes SSM diffusion baseline

Diffusion State Space Models (DiS, Feb 2024) replace U-Net/Transformer backbones with state space architectures. All inputs—time, condition, noisy patches—are treated as **concatenated tokens** processed by bidirectional SSM:

```
[time_embed, class_embed, patch_1, patch_2, ..., patch_N] → Bidirectional SSM → Noise prediction
```

Skip connections between shallow and deep layers follow U-Net design. Computational advantage: O(J) versus O(J²) for self-attention.

### ZigMa solves spatial continuity

Standard raster scanning ignores spatial locality when flattening 2D images to 1D sequences. ZigMa introduces **zigzag scanning patterns** that maintain spatial continuity—critical for images but also relevant for multivariate financial time series where asset proximity matters.

For 3D video/temporal data, ZigMa factorizes into 2D spatial + 1D temporal processing, with some layers using temporal-first scanning for learning motion dynamics.

### Conditioning mechanisms for SSM flow models

Four approaches have emerged for conditioning SSM-based generators:

1. **Token prepending** (DiS): Embed timestep and condition, prepend to sequence, process jointly through SSM
2. **AdaLN modulation**: Timestep generates scale/shift/gate for normalization layers
3. **Initial state conditioning** (DiMSUM): `h_{-1} = Linear(condition_embedding)`—inject condition as initial recurrent state
4. **FiLM modulation**: Feature-wise affine transformation with γ, β predicted from condition

For financial applications, **FiLM combined with initial state conditioning** offers the best balance: FiLM handles global conditions (regime, volatility) while state initialization provides persistent context throughout generation.

---

## Evaluation framework for conditioned financial generation

### Fidelity requires domain-specific metrics

Generic metrics like FID may show good scores while synthetic data fails under trading evaluation. **Stylized facts preservation is non-negotiable**:

- **Fat tails**: Compare tail indices (ξ) from Generalized Pareto Distribution; |ξ_real - ξ_generated| should minimize
- **Volatility clustering**: Match ACF decay of absolute returns at lags 1, 5, 10, 20
- **Leverage effect**: Verify negative cross-correlation between returns and future volatility

**Signature-based metrics** (Sig-W1, Sig-MMD) capture path-level dynamics essential for time series. FFAD (Fréchet Fourier-transform Autoencoder Distance) provides time-series-specific Fréchet distance.

### Controllability measures conditioning adherence

**J-FTSD** (Joint Fréchet Time Series Distance) evaluates whether generated series are specific to conditioning inputs—superior to naive FID extension. Measure:
- Regime classification accuracy on generated data
- Correlation structure matching: |ρ_target - ρ_generated|
- Volatility targeting: RMSE between target and achieved volatility

### Diversity detection prevents mode collapse

**Vendi Score** measures diversity without requiring reference data, using eigenvalue entropy of similarity matrices. **DMD-GEN** applies Dynamic Mode Decomposition with Optimal Transport for mode collapse detection in time series—identifies which dynamic modes collapsed.

For conditioned generation, measure intra-class variation: standard deviation of generated samples given identical conditioning should be non-trivial.

### Downstream utility validates practical value

The ultimate test: does synthetic data improve RL agent training?

- **Sample efficiency**: Steps to reach performance threshold on synthetic vs real
- **RL Premium**: Sharpe_RL - Sharpe_best_baseline
- **Generalization gap**: |Performance_real - Performance_synthetic|
- **TSTR/TRTS**: Train on Synthetic, Test on Real (and vice versa)

Recent work shows **2-8× sample efficiency improvements** with properly generated synthetic data for RL agent training.

---

## Implementation recommendations for JAX

### Architecture selection

For sequences up to 10K timesteps, a **Mamba2-attention hybrid** (1:6 ratio) with FiLM conditioning provides optimal balance. For longer sequences (100K+), pure Mamba2 with initial state conditioning may be necessary for memory constraints.

### Conditioning pipeline

```python
# Multi-modal conditioning for financial generation
class FinancialConditioner:
    def __init__(self, n_regimes, volatility_dim, history_dim):
        self.regime_embed = nn.Embedding(n_regimes, hidden_dim)
        self.vol_mlp = MLP([volatility_dim, hidden_dim, hidden_dim])
        self.history_encoder = MambaEncoder(history_dim, hidden_dim)
        self.film_generator = nn.Linear(hidden_dim * 3, hidden_dim * 2)
    
    def forward(self, regime_idx, vol_target, history):
        r = self.regime_embed(regime_idx)
        v = self.vol_mlp(vol_target)
        h = self.history_encoder(history).mean(dim=1)
        combined = torch.cat([r, v, h], dim=-1)
        gamma, beta = self.film_generator(combined).chunk(2, dim=-1)
        return gamma, beta  # For FiLM modulation
```

### Evaluation pipeline

1. **Fidelity**: Sig-W1, stylized facts (fat tails, volatility clustering, leverage)
2. **Controllability**: J-FTSD, regime adherence, volatility targeting accuracy
3. **Diversity**: Vendi Score, DMD-GEN mode collapse detection
4. **Downstream**: TSTR evaluation, RL agent sample efficiency

### Key implementation pitfalls

- **Forgetting causality**: SSMs are naturally causal; don't accidentally leak future information in conditioning
- **Scale mismatch**: Financial returns have different scales than image pixels; normalize carefully
- **Regime bleeding**: Ensure regime transitions in conditioning are reflected in generated data
- **Evaluation overfitting**: Don't optimize for metrics that don't correlate with downstream utility

The combination of selective state spaces, hybrid attention layers, and multi-modal conditioning creates a powerful framework for controllable financial market generation—enabling RL agents to train on diverse, realistic scenarios before deployment.