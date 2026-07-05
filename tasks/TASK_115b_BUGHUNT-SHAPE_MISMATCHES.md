# TASK_115b: Bug Hunt - Shape Mismatches

## Status: IN PROGRESS
**Date**: 2026-01-25
**Scope**: `diff_teacher/mamba/` and `diff_teacher/skeleton/`

## Identified Issues

### BUG-001: SSM Sequential Scan Broadcasting False Positive
**File**: `diff_teacher/mamba/ssm.py:40`
**Severity**: LOW (False Positive)
**Status**: VERIFIED/CLOSED

```python
# Discretize A: exp(delta * A)
delta_A = jnp.exp(delta_t[:, :, None] * A)  # (batch, d_inner, d_state)
```

**Re-analysis**: 
`delta_t` is `(batch, d_inner)`. `delta_t[:, :, None]` is `(batch, d_inner, 1)`. 
`A` is `(d_inner, d_state)`. 
The broadcasting `(B, D, 1) * (D, N)` results in `(B, D, N)`, where each channel `d` in `A` is scaled by its specific `delta`. This matches Mamba's channel-wise discretization logic.

### BUG-002: MambaBlock FiLM Dimension Mismatch
**File**: `diff_teacher/mamba/mamba_block.py:114`
**Severity**: CRITICAL
**Status**: OPEN

```python
# diff_teacher/mamba/mamba_block.py

d_inner = self.d_model * self.expand  # e.g., 1536
# ...
# y has shape (batch, seq, d_inner)
if cond is not None:
    gamma, beta = cond # (batch, d_model) e.g., 768
    y = y * (1 + gamma) + beta  # ERROR: 1536 vs 768
```

**Problem**: 
The state-space expansion inside `MambaBlock` changes the feature dimension to `d_inner`. The conditioning parameters `gamma/beta` are still in `d_model` space. 
**Fix**: Project `gamma/beta` to `d_inner` or move FiLM outside the expansion.

### BUG-003: SkeletonEncoder Stack Heterogeneity
**File**: `diff_teacher/skeleton/encoder.py:157`
**Severity**: CRITICAL
**Status**: OPEN

```python
# diff_teacher/skeleton/encoder.py

# TrendEncoder(sequence) -> (T, D)
# ClusterEncoder(static) -> (D,)

if embeddings:
    combined = jnp.stack(embeddings, axis=0).mean(axis=0)
```

**Problem**: 
`jnp.stack` will crash if the list contains both sequence `(T, D)` and global `(D,)` embeddings.
**Fix**: Broadcast all embeddings to `(T, D)` before stacking.

### BUG-004: VolRegimeEncoder Squeeze Failure
**File**: `diff_teacher/skeleton/encoder.py:52`
**Severity**: MEDIUM
**Status**: OPEN

```python
# Input (1,) -> Dense(D) -> (D,)
return x.squeeze(0)
```

**Problem**: 
Squeezing an axis of size `D` (where `D > 1`) raises a `ValueError`.
**Fix**: Remove `squeeze(0)`.

### BUG-005: HybridMambaFlow AdaLN Pooling Assumption
**File**: `diff_teacher/mamba/mamba_flow.py:91`
**Severity**: HIGH
**Status**: OPEN

```python
# HybridMambaFlow
scale, shift, gate = adaln(x.mean(axis=1), cond) 
```

**Problem**: 
`x.mean(axis=1)` pools the sequence to a global vector. If `cond` is sequence-based `(B, S, D)` (e.g. from `TrendEncoder`), `AdaLN` will fail or produce incorrect broadcasting results when applied back to `x`.
**Fix**: Support sequence-based conditioning in `AdaLN` without pooling.

### BUG-006: SkeletonConditioner Global Broadcasting
**File**: `diff_teacher/skeleton/encoder.py:236`
**Severity**: HIGH
**Status**: OPEN

```python
# SkeletonConditioner
gamma_s, beta_s, h0_s = skel_encoder(skeleton)
# ...
gamma_s = jnp.broadcast_to(gamma_s, (batch_size, self.d_model))
```

**Problem**: 
If `skel_encoder` returns a sequence `(T, D)`, `broadcast_to` will fail when trying to fit it into `(B, D)`.
**Fix**: Explicitly handle `ndim` of returned embeddings.

## Summary

| Bug ID | Location | Type | Severity |
|--------|----------|------|----------|
| BUG-001 | `mamba/ssm.py` | Shape/Broadcasting | LOW (FP) |
| BUG-002 | `mamba/mamba_block.py` | Dimension Mismatch | CRITICAL |
| BUG-003 | `skeleton/encoder.py` | Stack/Shape | CRITICAL |
| BUG-004 | `skeleton/encoder.py` | Tensor Op | MEDIUM |
| BUG-005 | `mamba/mamba_flow.py` | Conditioning Logic | HIGH |
| BUG-006 | `skeleton/encoder.py` | Broadcasting | HIGH |
