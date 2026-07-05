# TASK_115k: Bug Hunt - diff_teacher/skeleton/ and diff_teacher/generation/

**Status**: OPEN
**Created**: 2026-01-25
**Scope**: API inconsistencies, NaN/Inf edge cases, type mismatches

---

## Summary Table

| Bug ID | File | Line | Severity | Status | Description |
|--------|------|------|----------|--------|-------------|
| BUG-001 | encoder.py | 278 | CRITICAL | OPEN | Division by zero in sinusoidal_embedding when d_model < 2 |
| BUG-002 | builder.py | 173 | CRITICAL | OPEN | Division by near-zero in 'sparse' alpha_shape mode |
| BUG-003 | encoder.py | 165,168 | HIGH | OPEN | Zero-size embedding when d_model=1 |
| BUG-004 | extraction.py | 145 | HIGH | OPEN | Cumulative alpha can produce very large values |
| BUG-005 | extraction.py | 79,170,171 | MEDIUM | OPEN | Normalization can exceed [0,1] bounds |
| BUG-006 | types.py | 67-71 | MEDIUM | OPEN | .tolist() on None raises AttributeError (incorrect None check) |
| BUG-007 | generator.py | 136-138 | MEDIUM | OPEN | Silent None on missing checkpoint |
| BUG-008 | extraction.py | 94-96 | LOW | OPEN | Implicit Array->ndarray conversion |

---

## Detailed Bug Reports

---

#### BUG-001: Division by Zero in Sinusoidal Embedding
**File**: `diff_teacher/skeleton/encoder.py:275-282`
**Severity**: CRITICAL
**Status**: OPEN

```python
def sinusoidal_embedding(self, t: Array, dim: int) -> Array:
    """Sinusoidal positional embedding for timestep."""
    half_dim = dim // 2
    emb = jnp.log(10000) / (half_dim - 1)  # <-- DIVISION BY ZERO if dim < 2
    emb = jnp.exp(jnp.arange(half_dim) * -emb)
    emb = t[:, None] * emb[None, :]
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
    return emb
```

**Problem**:
- `half_dim = dim // 2` evaluates to 0 when `dim < 2`
- `(half_dim - 1)` becomes `-1` when `dim=1` or `0` when `dim=0`
- Division produces `-inf` or `inf`, which propagates through generation
- This affects `SkeletonConditioner` when instantiated with very small `d_model`

**Shape Analysis**:
- Input: `t` shape `(batch,)`, `dim` int
- When `dim=1`: `half_dim=0`, denominator=`-1`, result=`-9.21` (valid but wrong semantics)
- When `dim=0`: `half_dim=0`, denominator=`-1`, `jnp.arange(0)` = empty, concat fails

**Fix**:
```python
half_dim = max(dim // 2, 1)  # Guard against zero
emb = jnp.log(10000) / max(half_dim - 1, 1)  # Guard against div-by-zero
```

---

#### BUG-002: Division by Near-Zero in 'sparse' Alpha Shape
**File**: `diff_teacher/skeleton/builder.py:167-173`
**Severity**: CRITICAL
**Status**: OPEN

```python
elif shape == 'sparse':
    # Few concentrated opportunities
    self._alpha_shape = jnp.where(
        (t > 0.2) & (t < 0.3), t * 3,
        jnp.where((t > 0.7) & (t < 0.8), t * 3, t * 0.1)
    )
    self._alpha_shape = self._alpha_shape / self._alpha_shape[-1]  # <-- PROBLEM
```

**Problem**:
- For most timesteps outside [0.2,0.3] and [0.7,0.8], value = `t * 0.1`
- At `t=1.0` (last element): if outside both ranges, value = `1.0 * 0.1 = 0.1`
- Division by 0.1 is fine, but consider `n_timesteps=10`:
  - `t = linspace(0,1,9)` = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
  - At index 2 (t=0.25): `0.25*3=0.75` (in range)
  - At index 6 (t=0.75): `0.75*3=2.25` (in range)
  - At index 8 (t=1.0): `1.0*0.1=0.1` (outside ranges)
  - Normalization: `[...] / 0.1` produces values > 1.0
- For edge cases where `n_timesteps-1` places the last point exactly at 0.8:
  - `t[-1] = 0.8` -> falls in range -> `0.8*3=2.4`
  - But if `n_timesteps=6`: `t[-1] = 1.0` -> `0.1` -> very small

**Real Issue**: With certain `n_timesteps` values, `self._alpha_shape[-1]` can be as low as `0.1`, causing normalized values up to 10x larger than expected. Not strictly Inf, but breaks [0,1] assumption.

**Fix**:
```python
self._alpha_shape = self._alpha_shape / jnp.maximum(self._alpha_shape[-1], 1e-6)
# Or ensure final value is always the max:
self._alpha_shape = self._alpha_shape / jnp.maximum(self._alpha_shape.max(), 1e-6)
```

---

#### BUG-003: Zero-Size Embedding with Small d_model
**File**: `diff_teacher/skeleton/encoder.py:165,168`
**Severity**: HIGH
**Status**: OPEN

```python
class EventEncoder(nn.Module):
    d_model: int
    n_event_types: int = 10

    @nn.compact
    def __call__(self, events: list, n_timesteps: int) -> Array:
        # ...
        type_embed = nn.Embed(num_embeddings=self.n_event_types, features=self.d_model // 2)  # <-- Line 165
        mag_proj = nn.Dense(self.d_model // 2)  # <-- Line 168
```

**Problem**:
- When `d_model=1`: `d_model // 2 = 0`
- `nn.Embed(..., features=0)` creates zero-width embedding
- `nn.Dense(0)` creates a layer outputting nothing
- `jnp.concatenate([type_emb, mag_emb])` concatenates two empty arrays

**Shape Analysis**:
- With `d_model=1`:
  - `type_emb` shape: `(0,)`
  - `mag_emb` shape: `(0,)`
  - `event_emb` shape: `(0,)` after concat
  - `event_mask` shape: `(n_timesteps, 1)` initialized
  - Assignment `event_mask.at[event.time].set(event_emb)` fails due to shape mismatch

**Fix**:
```python
half_d = max(self.d_model // 2, 1)
type_embed = nn.Embed(num_embeddings=self.n_event_types, features=half_d)
mag_proj = nn.Dense(half_d)
```

---

#### BUG-004: Cumulative Alpha Can Produce Large Values
**File**: `diff_teacher/skeleton/extraction.py:140-146`
**Severity**: HIGH
**Status**: OPEN

```python
def extract_alpha_shape(returns: Array) -> Array:
    # Optimal return at each step: take best asset's absolute return
    optimal_per_step = jnp.abs(returns).max(axis=1)
    cumulative = jnp.cumsum(optimal_per_step)

    # Normalize
    alpha_norm = cumulative / (cumulative[-1] + 1e-10)  # <-- Issue here
    return alpha_norm
```

**Problem**:
- When all returns are exactly zero (flat market): `cumulative[-1] = 0`
- `0 / (0 + 1e-10) = 0` for all elements - this is fine
- But when returns are very small (e.g., 1e-12):
  - `cumulative[-1]` might be `1e-10` or smaller
  - Division can produce values >> 1 due to floating point
- More critically: no output validation ensures `alpha_norm` is in [0,1]

**Shape Analysis**:
- Input: `returns` shape `(n_timesteps-1, n_assets)`
- Output should be `(n_timesteps-1,)` with values in [0, 1]
- Edge case: constant returns produces linear alpha shape (correct)
- Edge case: near-zero returns produces near-zero / 1e-10 = large numbers (WRONG)

**Fix**:
```python
denom = jnp.maximum(cumulative[-1], 1e-10)
alpha_norm = jnp.clip(cumulative / denom, 0.0, 1.0)
```

---

#### BUG-005: Normalization Can Exceed [0,1] Bounds
**File**: `diff_teacher/skeleton/extraction.py:79,170,171`
**Severity**: MEDIUM
**Status**: OPEN

```python
# Line 79 (extract_vol_regime):
vol_norm = (avg_vol - avg_vol.min()) / (avg_vol.max() - avg_vol.min() + 1e-10)

# Lines 170-171 (detect_regime_sequence):
vol_percentile = (rolling_vol - rolling_vol.min()) / (rolling_vol.max() - rolling_vol.min() + 1e-10)
ret_percentile = (rolling_ret - rolling_ret.min()) / (rolling_ret.max() - rolling_ret.min() + 1e-10)
```

**Problem**:
- When `max - min < 1e-10`, the epsilon dominates
- Example: `max=0.5e-10`, `min=0.3e-10`, range=`0.2e-10`
  - Denominator: `0.2e-10 + 1e-10 = 1.2e-10`
  - `(0.5e-10 - 0.3e-10) / 1.2e-10 = 0.2e-10 / 1.2e-10 = 0.167` (looks fine)
  - But: `(0.5e-10 - 0.3e-10) / 1.2e-10` for the max value:
    `(0.5e-10 - 0.3e-10) / 1.2e-10 = 0.167` - not 1.0!
- When range is exactly 0 (constant input):
  - All values = constant, so `x - min = 0` for all
  - Output = `0 / 1e-10 = 0` (fine, but should probably be 0.5)

**Fix**:
```python
range_val = avg_vol.max() - avg_vol.min()
if range_val < 1e-10:
    vol_norm = jnp.full_like(avg_vol, 0.5)  # Constant -> middle value
else:
    vol_norm = (avg_vol - avg_vol.min()) / range_val
```

Or simpler:
```python
vol_norm = jnp.clip(vol_norm, 0.0, 1.0)
```

---

#### BUG-006: to_dict() Fails on Partial Skeleton
**File**: `diff_teacher/skeleton/types.py:64-76`
**Severity**: MEDIUM
**Status**: OPEN

```python
def to_dict(self) -> dict:
    """Convert to dictionary for serialization."""
    return {
        'trend_direction': self.trend_direction.tolist() if self.trend_direction is not None else None,
        'vol_regime': self.vol_regime.tolist() if self.vol_regime is not None else None,
        # ... etc
    }
```

**Problem**:
- Code looks correct at first glance
- BUT: the condition checks `if X is not None`, then calls `.tolist()`
- If X is a JAX array with value that evaluates to boolean False (empty array), condition fails
- Actually, this is NOT a bug - `is not None` is correct for Optional fields

**Correction**: After review, this is NOT a bug. Keeping entry for documentation that this was checked.

**Status**: FALSE POSITIVE - Remove from summary

---

#### BUG-007: Silent None on Missing Checkpoint
**File**: `diff_teacher/generation/generator.py:136-138`
**Severity**: MEDIUM
**Status**: OPEN

```python
# Load params
params = checkpoints.restore_checkpoint(str(checkpoint_path), target=None)
if 'params' in params:
    params = params['params']
```

**Problem**:
- `checkpoints.restore_checkpoint()` returns `None` if checkpoint doesn't exist
- `if 'params' in None` raises `TypeError`
- No validation that checkpoint exists before attempting to load

**Shape Analysis**: N/A (not a shape issue)

**Fix**:
```python
params = checkpoints.restore_checkpoint(str(checkpoint_path), target=None)
if params is None:
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
if 'params' in params:
    params = params['params']
```

---

#### BUG-008: Implicit Array to ndarray Conversion
**File**: `diff_teacher/skeleton/extraction.py:94-96`
**Severity**: LOW
**Status**: OPEN

```python
# Compute correlation matrix
returns_np = np.array(returns)  # Explicit conversion - GOOD
corr = np.corrcoef(returns_np.T)
corr = np.nan_to_num(corr, nan=0.0)
```

**Problem**:
- This is actually well-handled with explicit `np.array()` conversion
- However, the function signature claims `Array` input but uses NumPy internally
- Inconsistent with pure-JAX design of other functions

**Note**: This is a design inconsistency, not a bug. k-means clustering requires NumPy.

**Status**: NOTED - Design decision, not a bug

---

## Type Hint Inconsistencies

### 1. Return Type Mismatch: `extract_skeleton_batch`
**File**: `diff_teacher/skeleton/extraction.py:273-277`

```python
def extract_skeleton_batch(
    prices_batch: Array,
    window: int = 20,
    n_clusters: int = 5,
) -> list:  # Returns List[MarketSkeleton], not generic list
```

**Issue**: Return type should be `List[MarketSkeleton]`, not bare `list`.

### 2. Missing Return Type: SkeletonBuilder Methods
**File**: `diff_teacher/skeleton/builder.py`

Most builder methods return `'SkeletonBuilder'` as string annotation (forward reference), but some lack any return type hint entirely.

### 3. Union vs Optional Confusion
**File**: `diff_teacher/generation/generator.py:147`

```python
regime: Optional[Union[int, str]] = None,
```

This is technically correct but could be simplified to `Union[int, str, None]`.

---

## API Inconsistencies

### 1. Array vs ndarray Return Types

| Function | Input | Output | Issue |
|----------|-------|--------|-------|
| `generator.generate()` | - | `np.ndarray` | Returns NumPy (by design for user-facing API) |
| `generator.generate_on_device()` | - | `Array` | Returns JAX (for device-resident ops) |
| `streaming.generate_batch()` | `Array` | `Tuple[Array, Array]` | Returns JAX (consistent) |
| `extraction.extract_skeleton()` | `Array` | `MarketSkeleton` | Contains JAX Arrays internally |
| `extraction.extract_correlation_clusters()` | `Array` | `Array` | Uses NumPy internally but returns JAX |

**Assessment**: The split is intentional - user-facing APIs return NumPy, internal/device APIs return JAX. This is good design.

### 2. None Handling Patterns

| Function | None Check | Issue |
|----------|------------|-------|
| `SkeletonEncoder.__call__` | `if X is not None` | Correct |
| `SkeletonConditioner.__call__` | `if skeleton is not None and skeleton.has_conditioning()` | Correct |
| `VRAMGenerator.generate_batch` | `regime: Optional[Array]` | Passed through, model handles |
| `builder._build_vol_regime` | Multiple fallback branches | Correct |

**Assessment**: None handling is generally consistent across the codebase.

---

## Recommendations

### Immediate (Critical/High)
1. Fix BUG-001: Add guard in `sinusoidal_embedding` for small `d_model`
2. Fix BUG-002: Guard alpha_shape normalization in builder
3. Fix BUG-003: Guard EventEncoder for small `d_model`
4. Fix BUG-004: Clip cumulative alpha output to [0,1]

### Short-term (Medium)
5. Fix BUG-007: Add checkpoint existence check
6. Add output validation for skeleton extraction functions
7. Standardize type hints across all public functions

### Long-term (Low)
8. Consider adding `@jax.ensure_compile_time_eval` decorators where appropriate
9. Add runtime assertions for d_model >= 2 in constructors
10. Document the Array vs ndarray API boundary explicitly

---

*Last updated: 2026-01-25*
