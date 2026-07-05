# TASK_115j: Bug Hunt - API Contract Violations

**Focus**: `diff_teacher/generation/` and `diff_teacher/skeleton/`
**Date**: 2026-01-25
**Status**: IN_PROGRESS

## Findings

#### BUG-001: Non-Deterministic Event Encoding
**File**: `diff_teacher/skeleton/encoder.py:168`
**Severity**: CRITICAL
**Status**: OPEN

```python
type_idx = hash(event.event_type) % self.n_event_types
```

**Problem**: `hash()` is randomized by default in Python sessions (PYTHONHASHSEED). This means the same event type (e.g., 'fomc') will map to different embeddings in different runs/processes, breaking model reproducibility and inference consistency.
**Fix**: Use a deterministic string-to-index mapping (e.g., a fixed dictionary or consistent hashing).

#### BUG-002: SkeletonEncoder Discards Temporal/Sparse Information
**File**: `diff_teacher/skeleton/encoder.py:230`
**Severity**: CRITICAL
**Status**: OPEN

```python
        # Combine global embeddings
        if embeddings:
            combined = jnp.stack(embeddings, axis=0).mean(axis=0)
```

**Problem**: 
1. `SkeletonEncoder` averages all embeddings into a single global vector `combined`, destroying temporal structure.
2. `EventEncoder` and `AlphaShapeEncoder` are defined but **NEVER INSTANTIATED OR CALLED**.
3. `trend_direction`, `vol_regime` are only used if they produce global embeddings (via checks like `if trend_emb.ndim == 1`). If they produce sequences (which `TrendEncoder` can), those sequences are ignored because `embeddings` list only collects `ndim=1` items (or `stack` would fail).
**Fix**: Properly handle sequence embeddings (e.g., concatenate them or add them to the time embedding in `SkeletonConditioner`) and instantiate/use the missing encoders.

#### BUG-003: Unused Argument `return_sequence`
**File**: `diff_teacher/skeleton/encoder.py:214`
**Severity**: HIGH
**Status**: OPEN

```python
    def __call__(
        self,
        skeleton: MarketSkeleton,
        return_sequence: bool = False,
    ) -> Tuple[Array, Array, Optional[Array]]:
```

**Problem**: The `return_sequence` argument is present in the signature but completely ignored in the body. The function always returns global FiLM parameters derived from the averaged `combined` vector.

#### BUG-004: Hardcoded System Path
**File**: `diff_teacher/generation/generator.py:15`
**Severity**: MEDIUM
**Status**: OPEN

```python
sys.path.insert(0, '/home/lukas/Trader')
```

**Problem**: Absolute path hardcoded. This will break if the project is moved or run on another machine.
**Fix**: Use relative imports or proper package installation.

#### BUG-005: SkeletonBuilder Shape Mismatch
**File**: `diff_teacher/skeleton/builder.py:150`
**Severity**: MEDIUM
**Status**: OPEN

```python
            else:
                # Per-timestep, broadcast to assets
                direction_arr = jnp.array(direction, dtype=jnp.int32)
                self._trend_direction = jnp.broadcast_to(
                    direction_arr[:, None],
                    (len(direction), self._n_assets)
                )
```

**Problem**: When setting `trend_direction` from a list, the code uses `len(direction)` as the time dimension. If this does not match `self._n_timesteps - 1`, it creates a `MarketSkeleton` where components have inconsistent shapes (e.g., `vol_regime` might be length 1439, but `trend_direction` length 100).
**Fix**: Validate `len(direction)` against `self._n_timesteps` or resize/resample.

#### BUG-006: TrendEncoder Value Assumption
**File**: `diff_teacher/skeleton/encoder.py:24`
**Severity**: LOW
**Status**: OPEN

```python
        # Shift to {0, 1, 2} for embedding
        trend_shifted = trend_direction + 1  # {-1,0,1} -> {0,1,2}
```

**Problem**: Assumes `trend_direction` is strictly in `{-1, 0, 1}`. If values are outside this range (e.g. `2`), `nn.Embed` will error or produce garbage (index out of bounds).
**Fix**: `jnp.clip(trend_direction, -1, 1)` or assertion.

#### BUG-007: Zero-Padding Artifacts in Causal Convolutions
**File**: `diff_teacher/conv_flow_matching.py:100`
**Severity**: CRITICAL
**Status**: OPEN

```python
        pad_len = (self.kernel_size - 1) * self.dilation
        x_padded = jnp.pad(x, ((0, 0), (pad_len, 0), (0, 0)), mode='constant')
```

**Problem**: Using `mode='constant'` (zeros) for causal padding in market data generation creates a massive "cold start" artifact. Since prices are typically normalized around 0 (returns) or positive (OHLC), a block of zeros at the start of every sequence creates a non-market discontinuity that the model must waste capacity trying to smooth out.
**Fix**: Use `mode='edge'` or `mode='reflect'` to extend the first known values.

#### BUG-010: Generator/VAE Mode Inconsistency
**File**: `diff_teacher/teacher.py:349`
**Severity**: HIGH
**Status**: OPEN

```python
        # Decode to prices (if encoder is trained)
        if self.encoder_state is not None:
            # For VAE, we need to decode each timestep
            # This is expensive - in practice, might skip decoding
            # and work directly in latent space
            prices = None  # TODO: Implement proper decoding
        else:
            prices = None
```

**Problem**: `DiffusionTeacher.generate` is the primary API for getting data to trade on, but it returns `prices=None`. This makes the "VAE mode" (learning on latents) currently broken for any downstream agent that needs prices (which is all of them).
**Fix**: Implement the decoder call.

#### BUG-011: AdaLN Broadcasting Bug
**File**: `diff_teacher/flow_matching.py:105`
**Severity**: HIGH
**Status**: OPEN

```python
        # Reshape for broadcasting
        for _ in range(x.ndim - 2):
            scale = scale[:, None, ...]
            shift = shift[:, None, ...]
```

**Problem**: `scale` and `shift` are produced by `nn.Dense(dim)(cond)` which has shape `(batch, dim)`. The loop adds dimensions in the *middle*: `(batch, None, dim)`. For a 3D input `(batch, seq, dim)`, this results in `(batch, 1, dim)`. This is correct for `(batch, seq, dim)`, but if `x` has more dimensions (e.g. `(batch, seq, assets, features)`), this logic might fail or be ambiguous depending on how `dim` matches.
**Fix**: Use explicit broadcasting or `jnp.expand_dims`.

#### BUG-012: HybridMambaFlow Internal Feedback Loop
**File**: `diff_teacher/mamba/mamba_flow.py:109`
**Severity**: CRITICAL
**Status**: OPEN

```python
            # Get modulation parameters
            scale, shift, gate = adaln(x.mean(axis=1), cond)  # Pool over sequence
```

**Problem**: The modulation parameters (FiLM) are derived by pooling the *current noisy state* `x`. During flow matching denoising at low $t$ (high noise), the mean of the noise will corrupt the conditioning signal `cond`. This creates a catastrophic feedback loop where the model's own noisy output dictates how it should be conditioned, leading to "mode collapse" or noise-amplification.
**Fix**: Remove `x.mean(axis=1)` from the `adaln` call; use `cond` only.

#### BUG-013: Mamba Selective Scan Memory Blowup
**File**: `diff_teacher/mamba/ssm.py:112`
**Severity**: HIGH
**Status**: OPEN

```python
    delta_A = jnp.exp(jnp.einsum('bsd,dn->bsdn', delta, A))
    delta_B_x = jnp.einsum('bsd,bsn,bsd->bsdn', delta, B, x)
```

**Problem**: The parallel scan pre-computes $A$ and $B$ tensors of shape `(batch, seq, d_inner, d_state)`. With 1440 steps and `d_inner=1536`, `d_state=64`, this consumes ~4.4GB per batch (8). This exceeds the 6GB VRAM limit when coupled with other layers.
**Fix**: Use `jax.lax.scan` with a custom associative operator or reduce `d_state`.

#### BUG-015: Biased Autocorrelation Estimate
**File**: `diff_teacher/evaluation/stylized_facts.py:97`
**Severity**: HIGH
**Status**: OPEN

```python
    acf = np.sum(x_centered[lag:] * x_centered[:-lag]) / (len(x) * x_var)
```

**Problem**: Using `len(x) * x_var` as the denominator is statistically biased for short sequences. The correct denominator for Sample ACF is the sum of squares of the entire centered series.
**Fix**: `np.sum(x_centered[lag:] * x_centered[:-lag]) / np.sum(x_centered**2)`.

#### BUG-016: Evaluation Metric Return Consistency
**File**: `diff_teacher/evaluation/fidelity.py:53`
**Severity**: MEDIUM
**Status**: OPEN

```python
    # Extract returns
    if generated.ndim == 3:
        gen_returns = np.diff(generated[:, :, 0], axis=0)
```

**Problem**: `correlation_matrix_error` uses raw price differences (`np.diff`), while `wasserstein_distance` uses log returns (`np.diff(np.log(...))`). This inconsistency makes "fidelity" reports misleading if price scales differ.
**Fix**: Standardize on log returns for all financial fidelity metrics.

## Summary

| File | Critical | High | Medium | Low | Total |
|------|----------|------|--------|-----|-------|
| `skeleton/encoder.py` | 2 | 1 | 0 | 1 | 4 |
| `generation/generator.py` | 0 | 0 | 1 | 0 | 1 |
| `skeleton/builder.py` | 0 | 0 | 1 | 0 | 1 |
| `conv_flow_matching.py` | 1 | 0 | 0 | 0 | 1 |
| `flow_matching.py` | 0 | 1 | 0 | 0 | 1 |
| `teacher.py` | 0 | 1 | 0 | 0 | 1 |
| `mamba/mamba_flow.py` | 1 | 0 | 0 | 0 | 1 |
| `mamba/ssm.py` | 0 | 1 | 0 | 0 | 1 |
| `evaluation/stylized_facts.py` | 0 | 1 | 0 | 0 | 1 |
| `evaluation/fidelity.py` | 0 | 0 | 1 | 0 | 1 |
| **Total** | **4** | **5** | **4** | **1** | **14** |
