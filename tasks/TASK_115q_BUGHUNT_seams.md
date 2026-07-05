# TASK_115q: Integration Seam Failures Bug Hunt

**Date**: 2026-01-25
**Hunter**: Gemini
**Scope**: `diff_teacher` interfaces (Skeleton, Mamba, Generator, Evaluation)

## Summary of Findings

| ID | Severity | Component | Issue | Status |
|----|----------|-----------|-------|--------|
| BUG-001 | **CRITICAL** | Generator | `MarketGenerator` ignores `skeleton` argument | OPEN |
| BUG-002 | HIGH | Skeleton | `SkeletonEncoder` drops sequence/event data | OPEN |
| BUG-003 | HIGH | Mamba | `AdaLN` shares weights/outputs across all layers | OPEN |
| BUG-004 | MEDIUM | Evaluation | Inconsistent price feature indices (0 vs 3) | OPEN |
| BUG-005 | LOW | Evaluation | Time-series boundary bleed in flattened metrics | OPEN |

## Detailed Reports

### BUG-001: MarketGenerator ignores skeleton argument
**File**: `diff_teacher/generation/generator.py:126-155`
**Severity**: **CRITICAL**
**Status**: OPEN

```python
    def generate(
        self,
        n_samples: int = 1,
        # ...
        skeleton: Optional[MarketSkeleton] = None,  # <--- Argument provided
        # ...
    ) -> np.ndarray:
        # ...
        # Generate
        key = jax.random.PRNGKey(seed)
        generated = self._generate_jit(
            self.params,
            key,
            n_samples,
            seq_length,
            regime,  # <--- Regime is passed
            n_steps,
        ) # <--- Skeleton is NOT passed!
```

**Problem**: The `skeleton` argument passed to `generate()` is completely dropped. The `_generate_jit` (and underlying `_generate_impl`) function signature does not accept a skeleton, and `MambaFlowGenerator` (the model) is called with `skeleton_cond=None` (implicitly, as it's not passed). Users expecting conditional generation will get unconditional results (or simple regime conditioning).

**Fix**: 
1. Update `_generate_impl` to accept `skeleton`.
2. Inside `generate`, if `skeleton` is provided, encode it using `SkeletonConditioner` (or just `SkeletonEncoder`)?
   Wait, `MambaFlowGenerator` takes `skeleton_cond` (gamma, beta, h0).
   `MarketGenerator` needs to run `SkeletonEncoder` (which requires parameters!) or `SkeletonConditioner`.
   The model `MambaFlowGenerator` *contains* the logic to add `skeleton_cond` to embeddings.
   But `SkeletonEncoder` parameters need to be part of `self.params`.
   If `SkeletonEncoder` is not part of `MambaFlowGenerator`, its params are missing.
   `MambaFlowGenerator` does NOT seem to contain `SkeletonEncoder` as a submodule.
   It expects `skeleton_cond` tuple passed in.
   This suggests `SkeletonEncoder` should be run *outside* the JIT or passed in.
   But `SkeletonEncoder` has weights!
   This reveals a deeper architectural gap: Where does `SkeletonEncoder` live? Is it trained end-to-end? If so, it should be a submodule of the top-level model.

### BUG-002: SkeletonEncoder drops sequence/event data
**File**: `diff_teacher/skeleton/encoder.py:175-225`
**Severity**: HIGH
**Status**: OPEN

```python
        # Encode each component if present
        if skeleton.trend_direction is not None:
            trend_emb = TrendEncoder(self.d_model)(skeleton.trend_direction)
            if trend_emb.ndim == 1:
                embeddings.append(trend_emb)
            # For sequence, handle separately below <--- Never handled
```

**Problem**: `SkeletonEncoder` accumulates embeddings into a list `embeddings` ONLY if `ndim == 1` (global embeddings). It has a comment `# For sequence, handle separately below` but no code implements this.
1. `RegimeEncoder` output (if sequence) is dropped.
2. `TrendEncoder` output (if sequence) is dropped.
3. `EventEncoder` is defined in the file but NEVER instantiated or called in `SkeletonEncoder`.
Consequently, the returned `gamma, beta, h0` are purely global (time-invariant), destroying any temporal structure in the skeleton (like "crash then recovery").

**Fix**: Implement the sequence combination logic. Probably need to maintain a separate `sequence_embeddings` list, stack/mean them (if aligned), and add them to the broadcasted global embeddings before generating FiLM parameters.

### BUG-003: AdaLN shares weights/outputs across all layers
**File**: `diff_teacher/mamba/mamba_flow.py:65-71`
**Severity**: HIGH
**Status**: OPEN

```python
        # AdaLN for each layer
        adaln = AdaLN(self.d_model) # <--- Instantiated once

        for i in range(self.n_layers):
            # Get modulation parameters
            scale, shift, gate = adaln(x.mean(axis=1), cond)  # <--- Called repeatedly
```

**Problem**: `AdaLN` projects `cond` -> `scale, shift, gate`. Since the same module instance is used, the `nn.Dense` inside it is shared. Since `cond` is identical for all layers (global conditioning), `adaln(..., cond)` produces the **exact same** `scale, shift, gate` for every layer. This forces every layer to apply the exact same modulation, severely limiting the conditioning's power.

**Fix**: Move `AdaLN` instantiation inside the loop, or use `name=f'adaln_{i}'` to force separate parameters.

### BUG-004: Inconsistent price feature indices
**File**: `diff_teacher/evaluation/fidelity.py:44` vs `diff_teacher/evaluation/stylized_facts.py:38`
**Severity**: MEDIUM
**Status**: OPEN

**Problem**:
- `stylized_facts.py`: `close_idx = min(3, prices.shape[-1] - 1)` (Tries to grab index 3, usually Close)
- `fidelity.py`: `gen_returns = np.diff(generated[:, :, 0], axis=0)` (Grabs index 0, usually Open)

If Open and Close differ (volatility > 0), metrics are calculated on different time series.

**Fix**: Standardize on a helper function `get_close_price(prices)` in `evaluation/utils.py` or similar, defaulting to the config's Close index (usually 3).

### BUG-005: Time-series boundary bleed in flattened metrics
**File**: `diff_teacher/evaluation/stylized_facts.py:80-90`
**Severity**: LOW
**Status**: OPEN

```python
    returns_flat = returns.flatten()
    # ...
    # Compute rolling volatility
    for i in range(len(volatility)):
        volatility[i] = np.std(returns_flat[i:i + window])
```

**Problem**: Flattening a `(Timesteps, Assets)` array concatenates asset time series. Rolling windows (like volatility or autocorrelation) will slide across the jump from `Asset_i[Last]` to `Asset_{i+1}[First]`. This introduces junk data points that are physically meaningless.

**Fix**: Compute metrics per asset and average them, or use `vmap`/masking to respect boundaries.
