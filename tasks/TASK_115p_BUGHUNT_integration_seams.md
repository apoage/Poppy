# Bug Hunt Report: Integration Seam Failures

**Date**: 2026-01-25
**Auditor**: Claude Opus 4.5
**Scope**: Interface incompatibilities between diff_teacher modules

## Summary

| ID | File(s) | Severity | Description |
|----|---------|----------|-------------|
| BUG-101 | mamba/mamba_block.py:117-123 | CRITICAL | FiLM conditioning applied to wrong dimension (d_inner vs d_model) |
| BUG-102 | generation/generator.py:224 | HIGH | skeleton_cond never passed to MambaFlowGenerator |
| BUG-103 | mamba/mamba_flow.py:194-195 | HIGH | beta_s from skeleton conditioning is unpacked but never used |
| BUG-104 | skeleton/encoder.py:259-262 | MEDIUM | h0 returned but MambaBlock doesn't accept initial state |
| BUG-105 | evaluation/*.py | MEDIUM | Inconsistent OHLCV index assumptions across modules |
| BUG-106 | skeleton/encoder.py:170-183 | MEDIUM | EventEncoder Python list iteration breaks JIT |
| BUG-107 | mamba/mamba_flow.py:50 | LOW | AdaLN gate initialization doesn't produce zero-centered values |
| BUG-108 | mamba/bimamba.py:84-89 | LOW | BiMambaBlock passes same cond to both forward/backward |

---

## Details

#### BUG-101: FiLM Conditioning Dimension Mismatch
**File**: `diff_teacher/mamba/mamba_block.py:117-126`
**Severity**: CRITICAL
**Status**: OPEN

```python
# 7. FiLM conditioning (if provided)
if cond is not None:
    gamma, beta = cond
    # Broadcast to sequence if needed
    if gamma.ndim == 2:  # (batch, d_model)
        gamma = gamma[:, None, :]
        beta = beta[:, None, :]
    y = y * (1 + gamma) + beta  # BUG: y is (batch, seq, d_inner)

# 8. Output projection
y = nn.Dense(self.d_model, use_bias=False)(y)
```

**Problem**:
- `y` has shape `(batch, seq, d_inner)` where `d_inner = d_model * expand`
- `gamma`/`beta` have shape `(batch, 1, d_model)` after broadcast
- With `expand=2`, `d_inner = 2 * d_model`, so FiLM is applied to mismatched dimensions
- JAX will broadcast gamma/beta, truncating the conditioning to only affect half of d_inner

**Impact**: Conditioning only affects first `d_model` dimensions of internal representation. The second half of d_inner is modulated by broadcast values, likely causing unstable training or weak conditioning.

**Fix**: Apply FiLM after output projection, or project gamma/beta to d_inner:
```python
# Option A: Move FiLM after projection
y = nn.Dense(self.d_model, use_bias=False)(y)
if cond is not None:
    gamma, beta = cond
    if gamma.ndim == 2:
        gamma = gamma[:, None, :]
        beta = beta[:, None, :]
    y = y * (1 + gamma) + beta
return y

# Option B: Project conditioning to d_inner
if cond is not None:
    gamma, beta = cond
    gamma = nn.Dense(d_inner)(gamma)
    beta = nn.Dense(d_inner)(beta)
    if gamma.ndim == 2:
        gamma = gamma[:, None, :]
        beta = beta[:, None, :]
    y = y * (1 + gamma) + beta
```

---

#### BUG-102: Skeleton Conditioning Never Passed to Model
**File**: `diff_teacher/generation/generator.py:201-228`
**Severity**: HIGH
**Status**: OPEN

```python
def _generate_impl(
    self,
    params: dict,
    key: Array,
    n_samples: int,
    seq_length: int,
    regime: Optional[Array],
    n_steps: int,
) -> Array:
    # ...
    for i in range(n_steps):
        t = jnp.array([i * dt] * n_samples)

        # Predict velocity - skeleton_cond is NEVER passed!
        v = self.model.apply({'params': params}, x, t, regime)

        x = x + v * dt
    return x
```

**Problem**:
- `MarketGenerator.generate()` accepts a `skeleton` parameter (line 146)
- But `_generate_impl()` never receives or passes skeleton conditioning to the model
- `MambaFlowGenerator.__call__` expects `skeleton_cond` parameter (line 154)
- Skeleton-conditioned generation is documented but non-functional

**Impact**: Skeleton conditioning feature is completely broken. All generation is effectively unconditional.

**Fix**: Thread skeleton through the generation pipeline:
```python
def _generate_impl(
    self,
    params: dict,
    key: Array,
    n_samples: int,
    seq_length: int,
    regime: Optional[Array],
    skeleton_cond: Optional[Tuple[Array, Array, Array]],  # Add parameter
    n_steps: int,
) -> Array:
    # ...
    for i in range(n_steps):
        t = jnp.array([i * dt] * n_samples)
        v = self.model.apply({'params': params}, x, t, regime, skeleton_cond)
        x = x + v * dt
    return x
```

---

#### BUG-103: beta_s Never Used in MambaFlowGenerator
**File**: `diff_teacher/mamba/mamba_flow.py:192-195`
**Severity**: HIGH
**Status**: OPEN

```python
# Skeleton conditioning (if provided)
if skeleton_cond is not None:
    gamma_s, beta_s, h0_s = skeleton_cond  # beta_s unpacked
    cond = cond + gamma_s + h0_s           # beta_s NEVER USED
```

**Problem**:
- `SkeletonConditioner` returns `(gamma, beta, h0)` where beta is the FiLM shift parameter
- MambaFlowGenerator unpacks beta_s but discards it
- Only gamma_s and h0_s are added to the conditioning

**Impact**: Skeleton FiLM shift (translation) is lost. Only scale and initial state affect generation.

**Fix**: Use beta_s in the conditioning:
```python
if skeleton_cond is not None:
    gamma_s, beta_s, h0_s = skeleton_cond
    cond = cond + gamma_s + h0_s
    # Pass beta through to layers, or add separate shift channel
```

---

#### BUG-104: h0 Returned but Not Used by MambaBlock
**File**: `diff_teacher/skeleton/encoder.py:259-262` & `mamba/mamba_block.py`
**Severity**: MEDIUM
**Status**: OPEN

```python
# encoder.py returns h0 as "Initial state for Mamba"
h0 = nn.Dense(self.d_model)(combined)
return gamma, beta, h0
```

```python
# mamba_block.py - MambaBlock signature
def __call__(
    self,
    x: Array,
    cond: Optional[Tuple[Array, Array]] = None,  # Only gamma, beta - no h0
) -> Array:
```

**Problem**:
- SkeletonEncoder produces `h0` documented as "Initial state for Mamba"
- MambaBlock doesn't accept initial state parameter
- ssm.py initializes state to zeros (line 52): `h0 = jnp.zeros((batch, d_inner, d_state))`
- The skeleton initial state is propagated through MambaFlowGenerator but never actually used by MambaBlock

**Impact**: Initial state conditioning is non-functional. Wasted computation.

**Fix**: Either remove h0 from encoder or add initial state support to MambaBlock:
```python
# Option: Add h0 to MambaBlock
def __call__(
    self,
    x: Array,
    cond: Optional[Tuple[Array, Array]] = None,
    h0: Optional[Array] = None,  # Initial state
) -> Array:
```

---

#### BUG-105: Inconsistent OHLCV Index Assumptions
**Files**: Multiple evaluation modules
**Severity**: MEDIUM
**Status**: OPEN

| File | Line | Index Used | Assumes |
|------|------|------------|---------|
| `stylized_facts.py` | 39 | `min(3, shape[-1]-1)` | Close at index 3 |
| `fidelity.py` | 62, 67 | `[:, :, 0]` | Open at index 0 |
| `controllability.py` | 175, 231 | `[:, :, 0]` | Open at index 0 |

```python
# stylized_facts.py:39 - Uses Close
close_idx = min(3, prices.shape[-1] - 1)
close = prices[:, :, close_idx]

# fidelity.py:62 - Uses Open (index 0)
gen_returns = np.diff(generated[:, :, 0], axis=0)

# controllability.py:175 - Uses Open (index 0)
returns = np.diff(generated[:, :, 0], axis=0)
```

**Problem**:
- Stylized facts uses Close price (index 3) for return calculations
- Fidelity/controllability use Open price (index 0)
- Return calculations on Open vs Close give different results
- No central OHLCV constant definition

**Impact**: Inconsistent evaluation metrics. Comparing Open-returns to Close-returns.

**Fix**: Standardize on Close price (index 3) or create central constant:
```python
# In diff_teacher/__init__.py or types.py
OHLCV_IDX = {'open': 0, 'high': 1, 'low': 2, 'close': 3, 'volume': 4}
CLOSE_IDX = 3
```

---

#### BUG-106: EventEncoder Python List Iteration Breaks JIT
**File**: `diff_teacher/skeleton/encoder.py:170-183`
**Severity**: MEDIUM
**Status**: OPEN

```python
def __call__(self, events: list, n_timesteps: int) -> Array:
    event_mask = jnp.zeros((n_timesteps, self.d_model))

    if not events:
        return event_mask

    # Python for-loop over list - NOT JIT COMPATIBLE
    for event in events:
        if event.time < n_timesteps:
            type_idx = hash(event.event_type) % self.n_event_types
            # ... operations using Python int event.time
            event_mask = event_mask.at[event.time].set(event_emb)

    return event_mask
```

**Problem**:
- Python `for event in events` iterates over a Python list
- This creates a trace dependency that prevents JIT compilation
- `event.time` is Python int used in `at[].set()` which is dynamic indexing
- `hash(event.event_type)` is Python operation

**Impact**: EventEncoder can't be JIT-compiled. Falls back to eager execution, ~10-100x slower.

**Fix**: Use scatter operations with padded arrays:
```python
def __call__(self, event_times: Array, event_types: Array,
             event_magnitudes: Array, n_timesteps: int) -> Array:
    # Pre-padded arrays allow JIT compilation
    # Use jnp.where for conditional masking
```

---

#### BUG-107: AdaLN Gate Initialization
**File**: `diff_teacher/mamba/mamba_flow.py:46-51`
**Severity**: LOW
**Status**: OPEN

```python
def __call__(self, x: Array, cond: Array) -> Tuple[Array, Array, Array]:
    # Project conditioning to modulation parameters
    params = nn.Dense(self.d_model * 3)(cond)
    scale, shift, gate = jnp.split(params, 3, axis=-1)

    # Gate initialized near zero (AdaLN-Zero style)
    gate = nn.sigmoid(gate)  # Actually gives ~0.5, not ~0
```

**Problem**:
- Comment says "AdaLN-Zero style" which should initialize gate to produce ~0
- But `nn.Dense` default initialization + sigmoid gives output ~0.5
- AdaLN-Zero typically uses special initialization or `gate = gate * 0.0` at init

**Impact**: Initial training doesn't benefit from AdaLN-Zero's stable gradient flow. Minor issue.

**Fix**: Initialize bias to produce near-zero sigmoid output:
```python
# Add initialization that produces sigmoid output near 0
gate_proj = nn.Dense(self.d_model, kernel_init=nn.initializers.zeros,
                     bias_init=nn.initializers.constant(-5.0))
gate = nn.sigmoid(gate_proj(cond))  # sigmoid(-5) ≈ 0.007
```

---

#### BUG-108: BiMambaBlock Same Conditioning for Both Directions
**File**: `diff_teacher/mamba/bimamba.py:38-56`
**Severity**: LOW
**Status**: OPEN

```python
def __call__(self, x: Array, cond: Optional[Tuple[Array, Array]] = None) -> Array:
    # Forward pass
    h_fwd = MambaBlock(...)(x, cond)  # Same cond

    # Backward pass
    x_rev = jnp.flip(x, axis=1)
    h_bwd = MambaBlock(...)(x_rev, cond)  # Same cond again
    h_bwd = jnp.flip(h_bwd, axis=1)
```

**Problem**:
- Forward and backward Mamba blocks receive identical conditioning
- For time-dependent conditioning (e.g., varying gamma along sequence), backward pass should get flipped conditioning

**Impact**: If conditioning varies along sequence dimension, backward pass gets wrong alignment. Minor if conditioning is constant per-batch.

**Fix**: Flip conditioning for backward pass if sequence-varying:
```python
if cond is not None and cond[0].ndim == 3:  # (batch, seq, d)
    cond_bwd = (jnp.flip(cond[0], axis=1), jnp.flip(cond[1], axis=1))
else:
    cond_bwd = cond
h_bwd = MambaBlock(...)(x_rev, cond_bwd)
```

---

## Potential NaN/Inf Edge Cases

#### EDGE-01: Empty Skeleton Handling
**File**: `skeleton/encoder.py:244-247`
```python
if embeddings:
    combined = jnp.stack(embeddings, axis=0).mean(axis=0)
else:
    combined = jnp.zeros(self.d_model)
```
**Status**: OK - handled correctly with zeros fallback.

#### EDGE-02: Zero-Length Regime Sequence
**File**: `skeleton/types.py:96-99`
```python
def get_regime_idx(self) -> int:
    if self.regime_sequence is not None and len(self.regime_sequence) > 0:
        return int(jnp.bincount(self.regime_sequence.astype(jnp.int32)).argmax())
    return 2  # Default: sideways
```
**Status**: OK - handled with default fallback.

#### EDGE-03: Selective Scan Numerical Stability
**File**: `mamba/ssm.py:38, 96`
```python
delta_A = jnp.exp(delta_t[:, :, None] * A)  # exp can overflow
```
**Status**: OK - A is initialized negative (-exp(uniform)), delta is softplus (positive but bounded).

---

## Recommendations

1. **CRITICAL**: Fix BUG-101 immediately - FiLM dimension mismatch affects all Mamba conditioning
2. **HIGH**: Fix BUG-102 to enable skeleton conditioning feature
3. **HIGH**: Fix BUG-103 to use full FiLM (gamma + beta)
4. **MEDIUM**: Create central OHLCV constants (BUG-105)
5. **MEDIUM**: Refactor EventEncoder for JIT compatibility (BUG-106)
