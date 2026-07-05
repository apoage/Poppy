# TASK_115: Bug Hunt - Tensor Shape Mismatches

## Status: IN PROGRESS

## Goal
Systematic review of tensor shape handling across `diff_teacher/mamba/` and `diff_teacher/skeleton/` modules to identify and fix shape mismatches before they cause runtime failures.

## Focus Areas
- vmap/scan axis assumptions
- Batch dimension handling (is batch always first?)
- Broadcasting bugs where shapes accidentally align but semantics are wrong
- einsum subscript errors (bld vs bldn)
- Reshape operations that assume specific dimension ordering

---

## Hunt 1: Mamba + Skeleton Modules (2026-01-25)

### Critical Bugs

#### BUG-001: FiLM Conditioning Shape Mismatch in MambaBlock
**File**: `diff_teacher/mamba/mamba_block.py:117-123`
**Severity**: CRITICAL
**Status**: OPEN

```python
if cond is not None:
    gamma, beta = cond
    if gamma.ndim == 2:  # (batch, d_model)
        gamma = gamma[:, None, :]
        beta = beta[:, None, :]
    y = y * (1 + gamma) + beta  # BUG: y is (batch, seq, d_inner), gamma is (batch, 1, d_model)
```

**Problem**: `y` has shape `(batch, seq, d_inner)` where `d_inner = d_model * expand`. But `gamma/beta` is `(batch, 1, d_model)`. When `expand > 1`, shapes mismatch.

**Fix**: Either:
1. Project gamma/beta to d_inner before applying, OR
2. Move FiLM after output projection (line 126)

---

#### BUG-002: Inherited FiLM Bug in HybridMambaFlow
**File**: `diff_teacher/mamba/mamba_flow.py:116-123`
**Severity**: CRITICAL
**Status**: OPEN (blocked by BUG-001)

Passes `(batch, d_model)` conditioning to BiMambaBlock, inheriting BUG-001.

---

#### BUG-003: Skeleton Encoder Silently Drops 2D Embeddings
**File**: `diff_teacher/skeleton/encoder.py:219-241`
**Severity**: HIGH
**Status**: OPEN

```python
if skeleton.trend_direction is not None:
    trend_emb = TrendEncoder(self.d_model)(skeleton.trend_direction)
    if trend_emb.ndim == 1:
        embeddings.append(trend_emb)
    # If 2D (n_timesteps, d_model), silently dropped!
```

**Problem**: Per-timestep embeddings are silently discarded. Same issue for vol_regime and regime_sequence.

**Fix**: Handle sequence embeddings explicitly or aggregate before adding.

---

#### BUG-004: Skeleton Conditioner Broadcast Failure
**File**: `diff_teacher/skeleton/encoder.py:313-318`
**Severity**: HIGH
**Status**: OPEN

```python
gamma_s = jnp.broadcast_to(gamma_s, (batch_size, self.d_model))
```

**Problem**: If SkeletonEncoder returns `(n_timesteps, d_model)`, broadcast to `(batch_size, d_model)` fails.

**Fix**: Add shape check and handle sequence conditioning path.

---

#### BUG-005: MambaFlowGenerator Skeleton Shape Assumption
**File**: `diff_teacher/mamba/mamba_flow.py:192-195`
**Severity**: HIGH
**Status**: OPEN

```python
if skeleton_cond is not None:
    gamma_s, beta_s, h0_s = skeleton_cond
    cond = cond + gamma_s + h0_s  # Assumes (batch, d_model) or broadcastable
```

**Problem**: Skeleton conditioning could be `(n_timesteps, d_model)` which doesn't broadcast with `(batch, d_model)`.

---

### Medium Bugs

#### BUG-006: SelfAttention Divisibility Assumption
**File**: `diff_teacher/mamba/bimamba.py:108-129`
**Severity**: MEDIUM
**Status**: OPEN

```python
head_dim = self.d_model // self.n_heads
...
out = out.transpose(0, 2, 1, 3).reshape(batch, seq, self.d_model)
```

**Problem**: If `d_model % n_heads != 0`, reshape fails. Example: `d_model=100, n_heads=8` → `head_dim=12`, `8*12=96 != 100`.

**Fix**: Add assertion or use `d_model` that divides evenly.

---

### Low Priority

#### BUG-007: AdaLN Unused Parameter
**File**: `diff_teacher/mamba/mamba_flow.py:35-52`
**Severity**: LOW
**Status**: OPEN

The `x` parameter in `AdaLN.__call__` is accepted but never used. Misleading interface.

---

### Issues (Not Bugs)

#### ISSUE-001: Extraction n_timesteps Off-by-One
**File**: `diff_teacher/skeleton/extraction.py:251,268-269`

`n_timesteps` stored in skeleton is actually `n_original - 1` due to `jnp.diff` in returns computation.

#### ISSUE-002: EventEncoder Not JIT-Compatible
**File**: `diff_teacher/skeleton/encoder.py:170-183`

Python for-loop over dynamic event list causes recompilation for each event count.

---

## Hunt 2: Evaluation, Generation, Flow Matching (2026-01-25)

### High Bugs

#### BUG-008: Streaming skeleton_cond Type Mismatch
**File**: `diff_teacher/generation/streaming.py:182-188`
**Severity**: HIGH
**Status**: OPEN

```python
def euler_step(carry, t_idx):
    x = carry
    t = jnp.full((batch_size,), t_idx * dt)
    v = self.model.apply(
        {'params': params},
        x,
        t,
        skeleton_cond=skeleton_embedding  # BUG: expects tuple (gamma, beta, h0)
    )
```

**Problem**: `MambaFlowGenerator` expects `skeleton_cond` as `Tuple[Array, Array, Array]` (gamma, beta, h0), but `skeleton_embedding` is passed as single array.

**Fix**: Either unpack embedding into tuple, or change API.

---

#### BUG-009: Generator Reshape Without Validation
**File**: `diff_teacher/generation/generator.py:191-197`
**Severity**: HIGH
**Status**: OPEN

```python
generated = generated.reshape(
    n_samples,
    seq_length,
    self.config.n_assets,
    self.config.n_features,
)
```

**Problem**: If model output dim != `n_assets * n_features`, reshape fails with cryptic error. No validation that dimensions match.

**Same bug in**: `streaming.py:89-94` and `streaming.py:150-156`

**Fix**: Add assertion: `assert generated.shape[-1] == n_assets * n_features`

---

#### BUG-010: Controllability target_direction Silent Truncation
**File**: `diff_teacher/evaluation/controllability.py:258-263`
**Severity**: HIGH
**Status**: OPEN

```python
if len(target_direction) != n_assets:
    if len(target_direction) == 1:
        target_direction = np.full(n_assets, target_direction[0])
    else:
        target_direction = target_direction[:n_assets]  # Silent truncation!
```

**Problem**: If `target_direction` is shorter than `n_assets` (but > 1), comparison at line 265 will broadcast incorrectly or fail.

**Fix**: Raise error if length mismatch isn't 1.

---

### Medium Bugs

#### BUG-011: Inconsistent Feature Index Across Modules
**File**: Multiple files
**Severity**: MEDIUM
**Status**: OPEN

```python
# fidelity.py:61-63 - uses feature 0
gen_returns = np.diff(generated[:, :, 0], axis=0)

# stylized_facts.py:39-40 - uses min(3, shape-1)
close_idx = min(3, prices.shape[-1] - 1)
close = prices[:, :, close_idx]

# controllability.py:174-175 - uses feature 0
returns = np.diff(generated[:, :, 0], axis=0)
```

**Problem**: Some modules assume close price at index 0, others at index 3. Data format is OHLCV where close is index 3.

**Fix**: Standardize to index 3 (close) or add feature_idx parameter.

---

#### BUG-012: Generator Python Loop Prevents Efficient JIT
**File**: `diff_teacher/generation/generator.py:220-227`
**Severity**: MEDIUM
**Status**: OPEN

```python
for i in range(n_steps):
    t = jnp.array([i * dt] * n_samples)
    v = self.model.apply({'params': params}, x, t, regime)
    x = x + v * dt
```

**Problem**: Python for-loop inside JIT-compiled function creates multiple XLA traces. Should use `jax.lax.scan` for single-trace execution.

**Same bug in**: `conv_flow_matching.py:372-375`, `flow_matching.py:369-372`

**Fix**: Replace with `jax.lax.scan` (already done correctly in `streaming.py:117-124`).

---

### Low Priority

#### BUG-013: Controllability Returns Axis Confusion
**File**: `diff_teacher/evaluation/controllability.py:132-136`
**Severity**: LOW
**Status**: OPEN

```python
returns = np.diff(np.log(np.maximum(generated, 1e-10)), axis=0)
if returns.ndim > 1:
    returns = returns.mean(axis=-1) if returns.ndim == 2 else returns.mean(axis=(1, 2))
```

**Problem**: Confusing logic - 3D case averages over both seq AND assets (axes 1,2), which loses temporal structure. Probably intentional but undocumented.

---

### Issues (Not Bugs)

#### ISSUE-003: VRAMGenerator Captures self.model in Closure
**File**: `diff_teacher/generation/streaming.py:120`

Model captured in JIT closure works but causes recompilation if model object changes.

---

## Summary Table

| ID | File | Line | Severity | Status |
|----|------|------|----------|--------|
| BUG-001 | mamba_block.py | 117-123 | CRITICAL | OPEN |
| BUG-002 | mamba_flow.py | 116-123 | CRITICAL | OPEN |
| BUG-003 | encoder.py | 219-241 | HIGH | OPEN |
| BUG-004 | encoder.py | 313-318 | HIGH | OPEN |
| BUG-005 | mamba_flow.py | 192-195 | HIGH | OPEN |
| BUG-006 | bimamba.py | 108-129 | MEDIUM | OPEN |
| BUG-007 | mamba_flow.py | 35-52 | LOW | OPEN |
| BUG-008 | streaming.py | 182-188 | HIGH | OPEN |
| BUG-009 | generator.py | 191-197 | HIGH | OPEN |
| BUG-010 | controllability.py | 258-263 | HIGH | OPEN |
| BUG-011 | multiple | - | MEDIUM | OPEN |
| BUG-012 | generator.py | 220-227 | MEDIUM | OPEN |
| BUG-013 | controllability.py | 132-136 | LOW | OPEN |

---

| ID | File | Line | Severity | Status |
|----|------|------|----------|--------|
| BUG-001 | mamba_block.py | 117-123 | CRITICAL | OPEN |
| BUG-002 | mamba_flow.py | 116-123 | CRITICAL | OPEN |
| BUG-003 | encoder.py | 219-241 | HIGH | OPEN |
| BUG-004 | encoder.py | 313-318 | HIGH | OPEN |
| BUG-005 | mamba_flow.py | 192-195 | HIGH | OPEN |
| BUG-006 | bimamba.py | 108-129 | MEDIUM | OPEN |
| BUG-007 | mamba_flow.py | 35-52 | LOW | OPEN |

---

## Notes

- **BUG-001/002**: Will cause immediate runtime failure when `expand > 1` (default is 2)
- **BUG-003/004/005**: Only trigger with per-timestep skeleton conditioning
- **BUG-008**: Will fail when using skeleton-conditioned streaming generation
- **BUG-009**: Silent failure if model output doesn't match expected asset/feature count
- **BUG-011**: Data processed incorrectly - close price at wrong index
- All einsum operations in `ssm.py` verified correct
- Conv and transformer flow matching modules are clean (no shape bugs found)
