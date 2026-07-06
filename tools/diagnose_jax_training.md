# diagnose_jax_training.py

**JAX Training Diagnostics**

## Purpose

Comprehensive diagnostic tool for JAX MuZero training issues. Checks XLA setup, GPU access, dependencies, and model initialization.

## Usage

```bash
python tools/diagnose_jax_training.py
```

## Checks Performed

### 1. JAX Setup
- JAX version and installation
- Available devices (GPU/TPU/CPU)
- GPU computation test

### 2. Dependencies
| Package | Purpose |
|---------|---------|
| `jax` | Core JAX library |
| `flax` | Neural network library |
| `optax` | Optimizers |
| `mctx` | MCTS library |
| `chex` | Testing utilities |
| `orbax` | Checkpointing |

### 3. Model Initialization
- MuZero network creation
- Parameter initialization
- Forward pass test

### 4. Data Quality
- Observation shape validation
- Action mask validation

## Example Output

```
============================================================
1. Checking JAX Setup
============================================================
✅ JAX version: 0.4.25
✅ Available devices: [CudaDevice(id=0)]
✅ GPU computation test passed

============================================================
2. Checking Dependencies
============================================================
✅ jax: Core JAX library
✅ flax: Neural network library
✅ optax: Optimizers
✅ mctx: MCTS library
...
```

## Common Issues

**No GPU detected**:
- Check CUDA installation
- Verify `jax[cuda12]` installed

**Import errors**:
- Run `pip install jax flax optax mctx chex orbax-checkpoint`

## Related Files

- `test_mctx_jax.py` - Test MCTS performance
- `test_trm_jax.py` - Test TRM model
