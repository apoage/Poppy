# TASK_113: Mamba-Based Market Generation

## Status: GROUNDWORK COMPLETE (Phase 3 of Unified Architecture)

## Implementation Progress (2026-01-25)

### Completed: Mamba Module (`diff_teacher/mamba/`)

| File | Purpose | Status |
|------|---------|--------|
| `ssm.py` | selective_scan_sequential, selective_scan_parallel | DONE |
| `mamba_block.py` | CausalConv1D, MambaBlock with FiLM | DONE |
| `bimamba.py` | BiMambaBlock, SelfAttention | DONE |
| `mamba_flow.py` | HybridMambaFlow, MambaFlowGenerator, flow_matching_loss | DONE |

### Key Implementations

**Parallel Selective Scan (O(log L)):**
```python
def selective_scan_parallel(x, delta, A, B, C):
    """Uses jax.lax.associative_scan for parallel SSM."""
    delta_A = jnp.exp(jnp.einsum('bld,dn->bldn', delta, A))
    delta_B_x = jnp.einsum('bld,bln,bld->bldn', delta, B, x)

    def binary_op(carry, element):
        A_prev, h_prev = carry
        A_curr, Bx_curr = element
        return (A_curr * A_prev, A_curr * h_prev + Bx_curr)

    _, h = lax.associative_scan(binary_op, (delta_A_t, delta_B_x_t), axis=0)
    return jnp.einsum('bldn,bln->bld', h, C)
```

**MambaBlock with FiLM:**
- d_model, d_state=64, d_conv=4, expand=2
- Input-dependent A/B/C (selective)
- Optional FiLM conditioning

**HybridMambaFlow (1:7 Attention Ratio):**
- 24 layers, attention every 8th (not at front)
- BiMamba blocks for generative modeling
- AdaLN modulation for time conditioning

### Remaining Work
- [ ] Benchmark against CNN (TASK_111)
- [ ] Train Mamba variant on historical data
- [ ] Compare speed, memory, quality

---

## Why Mamba

State Space Models (SSMs) like Mamba offer the **best of both worlds**:

```
         Memory    Long-range    Cross-asset    Speed      In-Context
CNN      O(n)      Limited       Implicit       Fast       None
Transformer O(n²)  Global        Explicit       Slow       Excellent
Mamba    O(n)      Global        Implicit       Fast       Limited
```

**Key advantage**: Global context with linear complexity - essential for 10K+ timestep financial sequences.

---

## The Selective State Space Innovation

### Standard SSM (Fixed Filter)
```python
# Time-invariant matrices - processes all inputs uniformly
h_t = A @ h_{t-1} + B @ x_t
y_t = C @ h_t
```

### Mamba (Selective - Input-Dependent)
```python
# Parameters computed from input - content-aware processing
delta_t = softplus(Linear(x_t))  # Controls update rate
B_t = Linear(x_t)                 # Input gate (selective)
C_t = Linear(x_t)                 # Output gate (selective)

h_t = exp(delta_t * A) @ h_{t-1} + delta_t * B_t @ x_t
y_t = C_t @ h_t
```

### Financial Interpretation

The selection mechanism is like an **adaptive Kalman filter**:

| δ (delta) | Behavior | Market Analog |
|-----------|----------|---------------|
| Small | Emphasize memory, ignore input | Trust the trend, filter noise |
| Large | Emphasize input, reset state | React to shock, update belief |

This is exactly what traders do intuitively!

---

## Mamba-2 Advances

Mamba-2 (2024) introduces **State Space Duality (SSD)**:
- SSMs and linear attention are mathematically equivalent
- Chunked algorithm: 2-8x faster through better hardware utilization
- Larger state dimension: N = 64-256 vs Mamba-1's N = 16

---

## JAX Implementation: Parallel Associative Scan

The recurrence `h_t = A_t h_{t-1} + B_t x_t` appears sequential, but the operation is **associative**:

```python
import jax
import jax.numpy as jnp
from jax import lax

def selective_scan_parallel(x, delta, A, B, C):
    """
    Parallel associative scan for Mamba.
    O(log L) depth instead of O(L) sequential.

    Args:
        x: (batch, seq, d_inner) - input
        delta: (batch, seq, d_inner) - timestep
        A: (d_inner, d_state) - state transition (learned, fixed)
        B: (batch, seq, d_state) - input projection (selective)
        C: (batch, seq, d_state) - output projection (selective)
    """
    batch, seq, d_inner = x.shape
    d_state = A.shape[-1]

    # Discretize A using Zero-Order Hold
    delta_A = jnp.exp(jnp.einsum('bld,dn->bldn', delta, A))

    # Discretize B*x
    delta_B_x = jnp.einsum('bld,bln,bld->bldn', delta, B, x)

    # Binary operator for associative scan
    # (a1, b1) ⊕ (a2, b2) = (a2*a1, a2*b1 + b2)
    def binary_op(carry, element):
        A_prev, h_prev = carry
        A_curr, Bx_curr = element
        return (A_curr * A_prev, A_curr * h_prev + Bx_curr)

    # Transpose for scan over sequence dimension
    delta_A_t = delta_A.transpose(1, 0, 2, 3)  # (seq, batch, d_inner, d_state)
    delta_B_x_t = delta_B_x.transpose(1, 0, 2, 3)

    # Parallel scan - O(log L) depth
    _, h = lax.associative_scan(binary_op, (delta_A_t, delta_B_x_t), axis=0)

    # Transpose back
    h = h.transpose(1, 0, 2, 3)  # (batch, seq, d_inner, d_state)

    # Output: y = C @ h
    y = jnp.einsum('bldn,bln->bld', h, C)

    return y
```

### Performance Notes

| Implementation | Speed | Memory |
|----------------|-------|--------|
| CUDA fused kernel (official) | Fastest | Optimal |
| JAX associative_scan | 1.5-3x slower | Good |
| Sequential scan | 10x+ slower | Minimal |

For production, consider:
- **Chunked processing**: 256-token chunks fit in cache
- **Gradient checkpointing**: `@jax.checkpoint` for memory
- **Pallas kernels**: Custom JAX kernels (experimental)

---

## Mamba Block Architecture

```python
import flax.linen as nn

class MambaBlock(nn.Module):
    """Selective State Space Model block."""

    d_model: int
    d_state: int = 64      # SSM state dimension (Mamba-2 uses larger)
    d_conv: int = 4        # Local convolution width
    expand: int = 2        # Expansion factor

    @nn.compact
    def __call__(self, x, cond=None):
        batch, seq, _ = x.shape
        d_inner = self.d_model * self.expand

        # 1. Linear projection (expand) with gating
        x_proj = nn.Dense(d_inner * 2)(x)
        x_main, x_gate = jnp.split(x_proj, 2, axis=-1)

        # 2. Short convolution (local context)
        x_conv = CausalConv1D(d_inner, self.d_conv)(x_main)
        x_conv = nn.silu(x_conv)

        # 3. SSM parameters (input-dependent = selective)
        x_dbl = nn.Dense(self.d_state * 2 + 1)(x_conv)
        dt, B, C = jnp.split(x_dbl, [1, 1 + self.d_state], axis=-1)
        dt = nn.softplus(dt)

        # 4. A parameter (learned, not input-dependent)
        A = -jnp.exp(self.param('A_log',
            nn.initializers.uniform(), (d_inner, self.d_state)))

        # 5. Selective scan
        y = selective_scan_parallel(x_conv, dt, A, B, C)

        # 6. Gate and project
        y = y * nn.silu(x_gate)

        # 7. FiLM conditioning (if provided)
        if cond is not None:
            gamma, beta = cond
            y = y * (1 + gamma) + beta

        return nn.Dense(self.d_model)(y)
```

---

## BiMamba for Generative Models

Diffusion/Flow Matching requires bidirectional context:

```python
class BiMambaBlock(nn.Module):
    """Bidirectional Mamba for diffusion/flow matching."""

    d_model: int
    d_state: int = 64

    @nn.compact
    def __call__(self, x, cond=None):
        # Forward pass (causal)
        h_fwd = MambaBlock(self.d_model, self.d_state)(x, cond)

        # Backward pass (reverse sequence)
        x_rev = jnp.flip(x, axis=1)
        h_bwd = MambaBlock(self.d_model, self.d_state)(x_rev, cond)
        h_bwd = jnp.flip(h_bwd, axis=1)

        # Fuse via concatenation + projection
        h_combined = jnp.concatenate([h_fwd, h_bwd], axis=-1)
        return nn.Dense(self.d_model)(h_combined)
```

---

## Hybrid Architecture: Jamba Pattern

Pure Mamba has limited **in-context learning** (copying exact values). Solution: sparse attention layers.

### Design Principles

| Finding | Implication |
|---------|-------------|
| Attention excels at precise recall | Use for event conditioning |
| Mamba excels at compression | Use for bulk sequence processing |
| Never put attention at front | Let Mamba compress first |
| 1:7 ratio is optimal | 1 attention per 7 Mamba blocks |

```python
class HybridMambaFlow(nn.Module):
    """Hybrid Mamba-Attention for flow matching."""

    n_layers: int = 24
    d_model: int = 768
    d_state: int = 64
    attention_every: int = 8  # 1:7 ratio

    @nn.compact
    def __call__(self, x, t, skeleton=None):
        # Time + skeleton conditioning
        cond = self.build_conditioning(t, skeleton)

        for i in range(self.n_layers):
            # AdaLN modulation
            scale, shift, gate = self.adaln(cond)

            if i % self.attention_every == 0 and i > 0:
                # Sparse attention layer (every 8th, not at start)
                h = self.layer_norm(x) * (1 + scale) + shift
                x = x + gate * SelfAttention(self.d_model)(h)
            else:
                # Mamba layer
                h = self.layer_norm(x) * (1 + scale) + shift
                x = x + gate * BiMambaBlock(self.d_model, self.d_state)(h)

        return nn.Dense(self.input_dim)(x)
```

---

## MambaFlow: Mamba + Flow Matching

```python
class MambaFlowGenerator(nn.Module):
    """Mamba-based flow matching for market generation."""

    d_model: int = 768
    n_layers: int = 24
    d_state: int = 64
    n_regimes: int = 5

    @nn.compact
    def __call__(self, x, t, skeleton=None, training=False):
        batch, seq, input_dim = x.shape

        # Input projection
        h = nn.Dense(self.d_model)(x)

        # Time embedding (sinusoidal + MLP)
        t_emb = sinusoidal_embedding(t, self.d_model // 2)
        t_emb = nn.Dense(self.d_model)(nn.silu(nn.Dense(self.d_model)(t_emb)))

        # Skeleton conditioning (if provided)
        if skeleton is not None:
            skel_emb = SkeletonEncoder(self.d_model)(skeleton)
            cond = t_emb + skel_emb
        else:
            cond = t_emb

        # Hybrid Mamba-Attention backbone
        h = HybridMambaFlow(self.n_layers, self.d_model, self.d_state)(h, cond)

        # Output velocity prediction
        return nn.Dense(input_dim)(h)


def flow_matching_loss(model, params, x_1, skeleton, key):
    """OT-CFM loss for MambaFlow."""
    batch = x_1.shape[0]

    # Sample noise and time
    x_0 = jax.random.normal(key, x_1.shape)
    t = jax.random.uniform(jax.random.fold_in(key, 1), (batch,))

    # Interpolate (straight path)
    t_expand = t[:, None, None]
    x_t = (1 - t_expand) * x_0 + t_expand * x_1

    # Target velocity
    target_v = x_1 - x_0

    # Predict velocity
    pred_v = model.apply(params, x_t, t, skeleton)

    return jnp.mean((pred_v - target_v) ** 2)
```

---

## Sampling with Diffrax

```python
import diffrax

def generate_market_data(model, params, skeleton, key, n_steps=10):
    """Generate via ODE integration."""

    # Initial noise
    x_0 = jax.random.normal(key, (batch, seq_len, n_features))

    # Vector field
    def velocity_field(t, x, args):
        return model.apply(params, x, jnp.array([t]), skeleton)

    # Solve ODE
    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(velocity_field),
        diffrax.Euler(),  # Fast for straight paths
        # diffrax.Dopri5(),  # Adaptive for complex conditioning
        t0=0.0,
        t1=1.0,
        dt0=1.0 / n_steps,
        y0=x_0,
    )

    return solution.ys[-1]
```

---

## Multi-GPU Training (4x without NVLink)

```python
from functools import partial
import jax

# Data parallelism via pmap
@partial(jax.pmap, axis_name='batch')
def train_step(state, batch, key):
    def loss_fn(params):
        x_1, skeleton = batch
        return flow_matching_loss(model, params, x_1, skeleton, key)

    grads = jax.grad(loss_fn)(state.params)

    # Average gradients across devices (PCIe sync)
    grads = jax.lax.pmean(grads, axis_name='batch')

    return state.apply_gradients(grads=grads)

# Replicate state across devices
state = jax.device_put_replicated(state, jax.devices())

# Training loop
for batch in dataloader:
    # Split batch across devices: 32 → 8 per GPU
    batch = jax.tree_map(lambda x: x.reshape(4, -1, *x.shape[1:]), batch)
    state = train_step(state, batch, keys)
```

### PCIe Bandwidth Considerations

- 4x GPUs without NVLink: ~32 GB/s per link (vs 600 GB/s NVLink)
- Gradient sync becomes bottleneck for large models
- Solution: Gradient compression, async updates, or larger batch accumulation

---

## Model Configurations

| Config | d_model | n_layers | d_state | Params | Use Case |
|--------|---------|----------|---------|--------|----------|
| Small | 512 | 12 | 32 | ~50M | Fast iteration |
| Medium | 768 | 24 | 64 | ~150M | Balanced |
| Large | 1024 | 32 | 128 | ~400M | Maximum quality |

---

## Implementation Phases

### Phase 1: Basic MambaBlock
- [ ] Implement MambaBlock in pure JAX
- [ ] Test selective_scan_parallel correctness
- [ ] Profile speed vs sequential implementation
- [ ] Verify gradients flow correctly

### Phase 2: BiMamba + Flow Matching
- [ ] Create BiMambaBlock
- [ ] Integrate with flow matching training
- [ ] Add time conditioning (AdaLN)
- [ ] Test generation quality

### Phase 3: Hybrid Architecture
- [ ] Add sparse attention layers (1:7 ratio)
- [ ] Implement skeleton conditioning
- [ ] Compare vs pure Mamba

### Phase 4: Optimization
- [ ] Parallel scan optimization
- [ ] Gradient checkpointing
- [ ] Mixed precision (bfloat16)
- [ ] Multi-GPU data parallelism

### Phase 5: Benchmark vs CNN
- [ ] Same data, same epochs comparison
- [ ] Metrics: loss, quality, speed, memory
- [ ] Pick winner for production

---

## Comparison Benchmark Plan

Train on same data, measure:

| Metric | CNN (TASK_111) | Mamba (This) | Hybrid |
|--------|----------------|--------------|--------|
| Final loss | ? | ? | ? |
| Training time | ? | ? | ? |
| GPU memory | ~5GB | ~3GB (est) | ~4GB (est) |
| Inference speed | ~640ms | ~200ms (est) | ~300ms (est) |
| Long sequence (10K) | OOM | Works | Works |

---

## Relation to Other Tasks

| Task | Role | Status |
|------|------|--------|
| TASK_111 | CNN baseline | IN PROGRESS |
| TASK_112 | Skeleton conditioning | RESEARCH |
| TASK_113 | Mamba backbone (this) | FUTURE |

The Mamba backbone can be swapped into TASK_112's skeleton-conditioned generator once validated.

---

## Publication Potential

This is **highly publishable**:
- Novel application of Mamba to financial generation
- Flow matching + SSM combination
- Comprehensive benchmark vs transformers/CNNs
- Practical for 10K+ timestep sequences

Positioning: "MambaFlow: Efficient Skeleton-Conditioned Financial Time Series Generation with State Space Models"

---

## Resources

### Papers
- [Mamba](https://arxiv.org/abs/2312.00752) - Gu & Dao 2023
- [Mamba-2](https://arxiv.org/abs/2405.21060) - Dao & Gu 2024
- [Jamba](https://arxiv.org/abs/2403.19887) - AI21 Labs 2024
- [DiS](https://arxiv.org/abs/2402.05608) - Diffusion State Space Models
- [Flow Matching](https://arxiv.org/abs/2210.02747) - Lipman et al. 2023

### Code
- [mamba-ssm](https://github.com/state-spaces/mamba) - Official PyTorch
- [mamba-jax](https://github.com/radarFudan/mamba-jax) - Community JAX port
- [mamba2-jax](https://github.com/vasqu/mamba2-jax) - Mamba-2 JAX
- [Diffrax](https://docs.kidger.site/diffrax/) - JAX ODE solvers

---

## Notes

- "There is no such thing as overkill" - agreed
- Mamba is cool, worth exploring even if CNN works
- SSMs are a natural fit for time series
- If this works well, could be SOTA for market generation
- Financial interpretation of selectivity is compelling
