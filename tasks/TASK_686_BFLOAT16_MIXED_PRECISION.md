# TASK_686: Enable bfloat16 Mixed Precision

**Status**: DONE
**Priority**: 3 (enables TASK_683 to fit in 24GB)
**Source**: Analysis 004 (GLM), validated by codebase exploration

## Problem

Current training uses float32 everywhere. With TASK_683 widening state_dim 64→128, model params grow ~3× and may not fit in 24GB at batch=1024.

## Good News: Infrastructure Already Exists

The TRM/MuZero codebase already has full `dtype`/`param_dtype` support:

- `TRMEncoder` (trm_flax.py:10-49): `dtype`, `param_dtype` fields, casts input
- `TRMReasoning` (trm_flax.py:51-68): `dtype`, `param_dtype` fields, all Dense layers use them
- `MuZeroNetworks` (muzero_model.py:37-39): `dtype`, `param_dtype` fields, passed to TRM
- Output casting (trm_flax.py:208-211): policy/value/latent cast back to float32

**Not wired**: EasyTrainer doesn't pass dtype to MuZeroNetworks. Uses default float32.

## Changes

### 1. `easy_config.py`

```python
compute_dtype: str = "float32"  # "float32" or "bfloat16"
```

### 2. `easy_trainer.py`

```python
import jax.numpy as jnp

compute_dtype = jnp.bfloat16 if config.compute_dtype == "bfloat16" else jnp.float32

self.model = MuZeroNetworks(
    ...,
    dtype=compute_dtype,
    param_dtype=jnp.float32,  # Always store params in float32
)
```

### 3. `mcts_trader/env/jax_encoder.py`

Check that encoder output is float32 (model input). If encoder produces float32, the model's internal dtype handling takes care of the rest.

## Expected Benefit

- ~25% VRAM reduction (float32 params + bfloat16 activations)
- ~1.5-2× speedup on Ampere GPUs (RTX 3090, etc.)
- Enables wider model (TASK_683) to fit at batch=1024

## Risk

**Low**:
- Output casting already implemented (trm_flax.py:208-211)
- Value support [-10, 10] × 401 bins: softmax in float32 (already cast)
- Gradient accumulation via Adam: float32 (optax default)
- Only risk: batch norm statistics if added (not currently used)

## Verification

1. Run 10 epochs with float32 config → record agreement
2. Switch to bfloat16 → run 10 epochs → compare agreement (should be within 0.5%)
3. Monitor for NaN/inf in loss
4. Measure VRAM usage reduction
