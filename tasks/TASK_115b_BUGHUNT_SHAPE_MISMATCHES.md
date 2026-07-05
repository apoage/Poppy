# TASK_115b: Shape Mismatch & Masking Fixes

**Status**: Pending
**Priority**: High
**Created**: 2026-01-25

## Identified Issues

### 1. Incorrect Sigma Estimation in Student-t Loss (Trainer)
*   **File**: `mcts_trader/jax_agent/trainer.py`
*   **Issue**: In `_compute_prediction_loss`, `sigma_per_step` is calculated using `jnp.std(errors, axis=(0, 2))`. This reduction includes invalid samples (padding) in the batch, biasing the sigma estimate (likely making it smaller if padding errors are zero, or larger if random).
*   **Fix**: Compute weighted standard deviation using `valid_mask`, or mask errors before std computation.

### 2. Unmasked Student-t Metrics (Trainer)
*   **File**: `mcts_trader/jax_agent/trainer.py`
*   **Issue**: `student_t_metrics` (`student_t_nll_mean`, `sigma_mean`) use global `jnp.mean`, ignoring `valid_mask`. This pollutes metrics with invalid sample data.
*   **Fix**: Apply `valid_mask` to metric reductions.

### 3. Skeleton Conditioning Broadcasting (Mamba Flow)
*   **File**: `diff_teacher/mamba/mamba_flow.py`
*   **Issue**: `PureMambaFlowGenerator` and `MambaFlowGenerator` blindly add `skeleton_cond` components (`gamma_s + beta_s + h0_s`) to `cond`. If `skeleton_cond` contains sequence embeddings (rank 2) from `SkeletonEncoder(return_sequence=True)`, this addition will fail broadcasting against `cond` (rank 2, `batch, dim`) unless `batch == seq_len`.
*   **Fix**: 
    *   Assert `skeleton_cond` is global (rank 1) in `generate`.
    *   Or handle sequence broadcasting explicitly if sequence conditioning is intended (requires changing `cond` shape handling).

### 4. `TRMEncoder` Input Shape Ambiguity
*   **File**: `mcts_trader/jax_agent/trm_flax.py`
*   **Issue**: `TRMEncoder` uses `nn.Conv` (NHWC default) but input `observations` might be NCHW `(B, 27, 64, 64)` depending on pipeline.
*   **Verify**: Check `train_jax_fast.py` pipeline to confirm observation shape format. If NCHW, `TRMEncoder` needs explicit `NHWC` transpose or config.

## Plan
1.  Fix `sigma` estimation and metrics in `trainer.py` to use `valid_mask`.
2.  Add shape assertions/broadcasting logic in `mamba_flow.py`.
3.  Verify TRM input shapes.
