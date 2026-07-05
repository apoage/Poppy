# TASK_115r: Adversarial Code Review of Patches

**Date**: 2026-01-25
**Reviewer**: Gemini
**Scope**: `diff_teacher` patch verification

## Executive Summary

Patches have been applied to address the "Integration Seam" bugs (TASK_115q).
**Status**: mixed.
- **Fixed**: Critical wiring in `MarketGenerator` and `AdaLN` parameter sharing.
- **Partially Fixed**: Skeleton sequence encoding (introduced normalization bug).
- **Not Fixed**: Time-series boundary bleeding in evaluation metrics.
- **New Issues**: 3 new logic/consistency bugs identified in the patched code.

## Detailed Findings

### 1. Regression: Sequence Embedding Normalization (BUG-007)
**File**: `diff_teacher/skeleton/encoder.py:209-224`
**Severity**: **HIGH**

```python
            seq_combined = jnp.zeros((n_timesteps, self.d_model))
            for seq_emb in sequence_embeddings:
                # ...
                if seq_len >= n_timesteps:
                    seq_combined = seq_combined + seq_emb[:n_timesteps]
                else:
                    seq_combined = seq_combined.at[:seq_len].add(seq_emb)
            # FLAWED NORMALIZATION
            seq_combined = seq_combined / max(len(sequence_embeddings), 1)
```

**Analysis**: The code sums embeddings element-wise and then divides by the *total count of sequence types*.
If you have `Trend` (length 100) and `Events` (length 100), it works: `(Trend + Event) / 2`.
But if you have `Trend` (length 100) and `Events` (length 0 or short):
- Indices 0-End get `Trend / 2`.
- The `Trend` signal is effectively halved because the `Events` signal was absent/short, but the divisor remained `2` (count of lists).
**Impact**: Conditioning signal magnitude fluctuates wildly depending on how many sequence types are present/active, destabilizing the model.
**Fix**: Normalize by `count_at_timestep` mask, or ensure all inputs are fully padded to `n_timesteps` before averaging.

### 2. Oversight: Evaluation Boundary Bleed (BUG-005)
**File**: `diff_teacher/evaluation/stylized_facts.py:80-90`
**Severity**: LOW
**Status**: **NOT FIXED**

The code still flattens the multi-asset array `(T, Assets)` into `(T*Assets)` before running rolling window metrics (volatility, autocorrelation).
- `volatility[i] = np.std(returns_flat[i:i + window])`
This calculates volatility across the "seam" where Asset A ends and Asset B begins, producing junk data.

### 3. Logic: PureMambaFlowGenerator Broken (BUG-008)
**File**: `diff_teacher/mamba/mamba_flow.py`
**Severity**: MEDIUM

The `PureMambaFlowGenerator` class was NOT updated to accept `skeleton_cond`.
```python
    def __call__(self, x, t, regime=None, training=False): # Missing skeleton_cond
```
If a user sets `architecture="mamba"` (pure), the `MarketGenerator` might instantiate this (depending on import logic, currently `mamba` maps to `MambaFlowGenerator` which is Hybrid, so this is dead code, but dangerous/misleading).

### 4. Logic: Fail-Open Conditional Generation
**File**: `diff_teacher/generation/generator.py:126`
**Severity**: LOW

```python
        if skeleton is not None and skeleton_cond is None:
            warnings.warn(...)
            # Proceeds to call _generate_jit with skeleton_cond=None
```
The generator warns but continues to generate *unconditionally*. This "fail-open" behavior is risky for research pipelines where warnings are ignored. It should likely raise `ValueError`.

### 5. Consistency: Hardcoded Constants (BUG-009)
**File**: `diff_teacher/evaluation/stylized_facts.py` vs `fidelity.py`
**Severity**: TRIVIAL

- `fidelity.py` uses `OHLCV_CLOSE` (correct).
- `stylized_facts.py` uses `min(3, shape[-1]-1)` (hardcoded).
This introduces subtle skew if feature ordering changes.

## Verification of Fixes

| Original Bug | Status | Notes |
|--------------|--------|-------|
| Generator ignores `skeleton` | **FIXED** | Passed through `_generate_impl`. |
| `SkeletonEncoder` drops sequences | **PARTIAL** | Logic added but normalization is flawed (BUG-007). |
| `AdaLN` sharing weights | **FIXED** | Unique `name`s added to instances. |
| Inconsistent Price Index | **FIXED** | `fidelity.py` updated to `OHLCV_CLOSE`. |

## Recommendations

1.  **Refactor `SkeletonEncoder` normalization**: Use a weighted average where weights are masks of valid data at each timestep.
2.  **Fix `stylized_facts.py`**: Iterate over assets to compute metrics, then average, instead of flattening.
3.  **Update `PureMambaFlowGenerator`**: Add `skeleton_cond` argument or remove the class if deprecated.
4.  **Harden `MarketGenerator`**: Raise error if skeleton provided but conditioning params missing.
