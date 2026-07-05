# Precision and Performance Notes

**Status**: Documentation (future optimization consideration)
**Last Updated**: December 21, 2025

---

## Current Precision: float32

The entire JAX training pipeline uses **float32** precision:

| Component | Dtype | Location |
|-----------|-------|----------|
| Observations | `jnp.float32` | train_jax_fast.py:625 |
| Actions | `jnp.int32` | Discrete indices only |
| Rewards/Values | `jnp.float32` | replay_buffer.py:37-38 |
| Policy probs | `jnp.float32` | replay_buffer.py:39 |
| Model weights | float32 (default) | Flax Dense/Conv layers |
| Gradients | float32 | No AMP configured |

---

## bfloat16/fp16 History

### Why Disabled

bfloat16 was tested and disabled due to:
- **inf/nan errors** in loss function during early training versions
- Suspected cause: Value head categorical distribution (101 bins) sensitive to precision
- Softmax over wide support range (-10 to +10) may have numerical issues

### Potential Benefits (RTX 3090 Ampere)

| Precision | TFLOPS | Notes |
|-----------|--------|-------|
| FP32 (CUDA cores) | 29.28 | Current mode |
| FP16 (CUDA cores) | 29.28 | Same as FP32 |
| **FP16 Tensor Cores** | **235** | **8× faster!** |

Memory bandwidth: 936 GB/s

**Theoretical speedup**: **8× for compute-bound operations** if Tensor Cores enabled.

### Tensor Core Requirements
- Matrix dimensions must be multiples of 8 (for FP16) or 16 (for INT8)
- Need to use specific JAX/XLA flags or libraries (e.g., cuBLAS with TF32)
- Flax Dense layers may need explicit dtype and padding

### Implementation Notes for Future

```python
# Option 1: Global JAX precision
jax.config.update("jax_default_matmul_precision", "bfloat16")

# Option 2: Per-layer dtype (Flax)
nn.Dense(features=128, dtype=jnp.bfloat16)

# Option 3: Mixed precision training
# Keep loss in fp32, compute in fp16/bf16
def loss_with_precision(params, batch):
    # Cast to bfloat16 for forward pass
    params_bf16 = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), params)
    loss = compute_loss(params_bf16, batch)
    # Cast back to fp32 for gradients
    return loss.astype(jnp.float32)
```

### Known Precision Trouble Spots

1. **Softmax on categorical value head**
   - 101 bins with range [-10, +10]
   - Large logits → softmax overflow risk

2. **Layer normalization**
   - Variance computation sensitive to precision
   - May need `epsilon` adjustment for fp16

3. **Gradient accumulation**
   - Small gradients may underflow in fp16
   - Consider gradient scaling

---

## Performance Optimization Roadmap

### Phase 1: Profile Current Bottlenecks
- [ ] Profile with JAX profiler
- [ ] Identify compute vs memory bound operations
- [ ] Measure actual GPU utilization

### Phase 2: Safe Precision Experiments
- [ ] Try bfloat16 on encoder only (CNN layers)
- [ ] Keep value/policy heads in fp32
- [ ] Monitor for inf/nan during training

### Phase 3: Full Mixed Precision
- [ ] Implement proper loss scaling
- [ ] Cast master weights (fp32) → compute weights (bf16)
- [ ] Gradient scaling for stability

---

## Quick Reference

```python
# Check current precision
import jax.numpy as jnp
print(jnp.ones(1).dtype)  # Should show float32

# Enable bfloat16 globally (EXPERIMENTAL - may cause inf/nan)
# jax.config.update("jax_default_matmul_precision", "bfloat16")
```

---

## Dependencies

Before enabling mixed precision:
- [ ] Value head stable (correlation >0.8)
- [ ] No existing inf/nan in training
- [ ] Checkpointing working (for rollback)
- [ ] Monitoring in place to detect precision issues

---

## Related: Transformer Migration

FP16 Tensor Cores are bundled with Transformer migration (TRM v2).
See `tasks/TASK_08_TRANSFORMER_TRM_V2.md` for full plan.

Expected combined benefits:
- **8× compute speedup** (Tensor Cores)
- **50% VRAM reduction** (bf16 precision)
- **Better parallelization** (attention vs sequential MLP loops)
