# TASK 08: Transformer Migration (TRM v2)

**Priority**: HIGH (major architectural upgrade)
**Status**: Design phase
**Complexity**: High (full architecture rewrite)
**Breaks Checkpoints**: YES

---

## Objective

Replace TRM recursive blocks with standard Transformer architecture while:
1. Enabling FP16 Tensor Core compute (8× speedup potential)
2. Reducing VRAM usage via half-precision
3. Maintaining or improving reasoning capabilities
4. Keeping fallback path to TRM if needed

---

## Why Transformer

| Aspect | TRM (Current) | Transformer |
|--------|---------------|-------------|
| Training Speed | Baseline | **2-8× faster** |
| Hardware | CUDA cores (29 TFLOPS) | **Tensor Cores (235 TFLOPS)** |
| VRAM | fp32 (~14GB for 36M) | **fp16 (~7GB for 36M)** |
| Multi-asset correlation | Implicit | **Explicit attention** |
| Research backing | Custom | **Massive ecosystem** |

### TRM Strengths to Preserve

- Iterative refinement ("thinking" through H×L cycles)
- Compute efficiency on reasoning tasks (ARC benchmark competitive)
- Recursive inductive bias

### Transformer Advantages

- Native attention for 64-asset correlation modeling
- Flash Attention (memory-efficient, faster)
- Tensor Core utilization
- Better tooling and debugging resources

---

## Architecture Design

### Current TRM Flow
```
Observation (27, 64, 64)
    ↓
CNN Encoder → state (batch, state_dim)
    ↓
H_cycles × L_cycles MLP refinement → latent
    ↓
Policy/Value heads
```

### Proposed Transformer Flow
```
Observation (27, 64, 64)
    ↓
Patch Embedding OR CNN Encoder → tokens (batch, n_tokens, d_model)
    ↓
N × Transformer Blocks (attention + FFN)
    ↓
Pooling → latent (batch, state_dim)
    ↓
Policy/Value heads
```

### Key Components

```python
class TransformerReasoning(nn.Module):
    """Transformer-based reasoning replacing TRM cycles."""

    d_model: int = 256        # Hidden dimension
    n_heads: int = 8          # Attention heads
    n_layers: int = 6         # Transformer blocks
    ffn_dim: int = 1024       # FFN hidden dimension
    dropout: float = 0.1
    dtype: jnp.dtype = jnp.bfloat16  # Tensor Core precision

    @nn.compact
    def __call__(self, x, train: bool = True):
        # x: (batch, n_tokens, d_model)

        for i in range(self.n_layers):
            # Multi-head self-attention
            attn_out = nn.MultiHeadDotProductAttention(
                num_heads=self.n_heads,
                dtype=self.dtype,
                kernel_init=nn.initializers.xavier_uniform(),
            )(x, x)
            x = nn.LayerNorm(dtype=self.dtype)(x + attn_out)

            # FFN
            ffn_out = nn.Dense(self.ffn_dim, dtype=self.dtype)(x)
            ffn_out = nn.gelu(ffn_out)
            ffn_out = nn.Dense(self.d_model, dtype=self.dtype)(ffn_out)
            x = nn.LayerNorm(dtype=self.dtype)(x + ffn_out)

        # Pool to single latent vector
        latent = jnp.mean(x, axis=1)  # (batch, d_model)
        return latent
```

---

## FP16 Tensor Core Implementation

### Precision Strategy

```python
# Master weights in fp32 (gradient stability)
# Compute in bfloat16 (Tensor Cores)
# Loss in fp32 (numerical stability)

class MixedPrecisionWrapper:
    def __init__(self, model):
        self.model = model
        self.master_params = None  # fp32

    def forward(self, x):
        # Cast inputs to bf16
        x_bf16 = x.astype(jnp.bfloat16)
        # Compute in bf16
        out = self.model.apply(self.params_bf16, x_bf16)
        # Cast output to fp32 for loss
        return out.astype(jnp.float32)

    def update(self, grads):
        # Accumulate gradients in fp32
        # Update master weights
        # Sync bf16 copy
```

### Tensor Core Requirements

1. **Dimensions multiples of 8** (for FP16) or 16 (for INT8)
   - d_model = 256 (divisible by 8)
   - n_heads = 8 (256/8 = 32 per head, divisible by 8)
   - ffn_dim = 1024 (divisible by 8)
   - n_tokens = 64 (assets) or padded

2. **XLA Configuration**
   ```python
   # Enable Tensor Cores in JAX
   jax.config.update("jax_default_matmul_precision", "bfloat16")
   ```

3. **Gradient Scaling** (if fp16 instead of bf16)
   ```python
   # bf16 has larger exponent range, usually doesn't need scaling
   # fp16 may need loss scaling for small gradients
   ```

---

## VRAM Optimization

### Expected Savings

| Component | fp32 | bf16 | Savings |
|-----------|------|------|---------|
| Model weights (36M) | 144 MB | 72 MB | 50% |
| Activations | ~3 GB | ~1.5 GB | 50% |
| Optimizer states | ~288 MB | ~144 MB | 50% |
| **Total (batch=2048)** | **~16 GB** | **~8 GB** | **50%** |

### Additional Optimizations

1. **Flash Attention** (memory-efficient attention)
   ```python
   # Use JAX Flash Attention implementation
   from jax.experimental.pallas.ops import attention
   ```

2. **Gradient Checkpointing** (trade compute for memory)
   ```python
   @jax.checkpoint
   def transformer_block(x):
       ...
   ```

3. **Activation Compression** (future)

---

## Implementation Plan

### Phase 1: Basic Transformer (fp32)
- [ ] Create `transformer_flax.py` with TransformerReasoning
- [ ] Match parameter count to TRM (~36M)
- [ ] Integrate with MuZeroNetworks
- [ ] Verify training works (no regressions)

### Phase 2: Mixed Precision (bf16)
- [ ] Add dtype parameter to all layers
- [ ] Implement master weight / compute weight split
- [ ] Test for inf/nan stability
- [ ] Verify Tensor Core utilization (nvidia-smi)

### Phase 3: Optimizations
- [ ] Flash Attention integration
- [ ] Gradient checkpointing if needed
- [ ] Profile and optimize bottlenecks

### Phase 4: Validation
- [ ] A/B test vs TRM on Level 0-3
- [ ] Compare value-reward correlation
- [ ] Measure actual speedup (target: 2-4×)
- [ ] Verify VRAM reduction (target: 50%)

---

## Fallback Plan

If Transformer underperforms:
1. **Keep TRM as option** - config switch `--model-arch trm|transformer`
2. **Hybrid approach** - CNN encoder + Transformer reasoning
3. **TRM with bf16** - retrofit precision to existing TRM

---

## Metrics to Track

```python
# New CSV columns
'model_arch': 'trm' or 'transformer'
'precision': 'fp32' or 'bf16'
'tensor_core_utilization': percentage from profiler
'peak_vram_gb': max GPU memory used
'tokens_per_second': throughput metric
```

---

## Research References

### Efficient Transformers
- Flash Attention (Dao et al., 2022) - memory-efficient attention
- Flash Attention 2 (Dao, 2023) - faster, better parallelism

### RL + Transformers
- Decision Transformer (Chen et al., 2021) - offline RL
- Gato (Reed et al., 2022) - generalist agent
- RT-2 (Brohan et al., 2023) - vision-language-action

### Mixed Precision
- Mixed Precision Training (Micikevicius et al., 2018) - NVIDIA
- BFloat16 Training (Kalamkar et al., 2019) - Google

### Recursive/Iterative Models
- Universal Transformers (Dehghani et al., 2019) - shared weights, depth via recurrence
- PonderNet (Banino et al., 2021) - adaptive computation

---

## Acceptance Criteria

- [ ] Training speed ≥2× faster than TRM
- [ ] VRAM usage ≤50% of TRM (fp32)
- [ ] Value-reward correlation ≥0.8 (no regression)
- [ ] Tensor Core utilization >50% (nvidia-smi shows fp16 activity)
- [ ] Stable training (no inf/nan for 1000+ epochs)
- [ ] Level 0-3 performance matches or exceeds TRM

---

## Notes

- bf16 preferred over fp16 (larger exponent range, no scaling needed)
- Keep CNN encoder initially (proven), swap to patch embedding later if needed
- 64 assets = 64 tokens = natural fit for attention
- Attention weights interpretable: "BTC attending to ETH correlation"

---

## Known Flaws to Fix in TRM v2

### 1. Value Support Range Saturation

**Problem**: Current support range [-10, +10] with 101 bins is too narrow.

**Observed**:
- Actual rewards reaching +18% to +28% on Level 3
- Value head saturates at bin 100 (value = 10.0)
- Predictions become non-linear at extremes
- Linear calibration (scale/shift) cannot fix saturation

**Evidence**:
```
Value Head: Peak bin=100/101 (10.00) | Peak prob=0.147
Actual rewards: -5 to +28%
Value-Reward graph: linear -5 to +10, then diverges
```

**Fix for TRM v2**:
```python
# Wider support range
support_range = 30  # [-30, +30] instead of [-10, +10]
support_size = 121  # More bins for finer resolution

# Or dynamic support based on observed reward range
support = jnp.linspace(-max_observed * 1.5, max_observed * 1.5, support_size)
```

**Impact**: Better value calibration at extremes, more accurate MCTS planning for high-reward scenarios.
