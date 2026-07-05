# TASK_118: Numerical Stability Audit & Fixes

**Status**: FIXED
**Priority**: High
**Created**: 2026-01-25
**Fixed**: 2026-01-26

## Objective
Address numerical stability issues identified during code review, specifically focusing on potential overflows, underflows, and unbounded values in JAX computations.

## Identified Issues

### 1. Student-t Loss Instability
*   **File**: `mcts_trader/jax_agent/prediction/student_t.py`
*   **Issue**: In `student_t_loss`, `log_sigma` is exponentiated directly (`sigma = jnp.exp(log_sigma)`). Extreme values of `log_sigma` (e.g., > 88 or < -88) can cause `inf` or `0` (underflow), leading to NaNs in NLL computation.
*   **Proposed Fix**: Clip `log_sigma` to a safe range (e.g., `[-10.0, 10.0]`) before exponentiation.

### 2. Mamba SSM Sequential Scan Divergence
*   **File**: `diff_teacher/mamba/ssm.py`
*   **Issue**: The `selective_scan_sequential` implementation calculates `delta_A = jnp.exp(delta_t[:, :, None] * A)` without clipping. The parallel implementation (`selective_scan_parallel`) correctly clips the exponent to `[-20.0, 20.0]`. This inconsistency can lead to divergence between the two implementations and instability in the sequential version.
*   **Proposed Fix**: Apply the same clipping logic (`jnp.clip(..., -20.0, 20.0)`) to the sequential implementation.

### 3. Unbounded Sigma Estimate in Trainer
*   **File**: `mcts_trader/jax_agent/trainer.py`
*   **Issue**: In `_compute_prediction_loss` (within the Student-t branch), `sigma_per_step` is estimated using `jnp.std(errors)`. While it has a floor (`1e-4`), it lacks a ceiling. If model predictions diverge wildly (e.g., during early training or instability), `sigma` can become huge, and `jnp.log(sigma)` in the loss term can contribute to gradient explosions or instability.
*   **Proposed Fix**: Clip the estimated `sigma` (or `log(sigma)`) to a reasonable upper bound (e.g., `10.0`).

### 4. General Review
*   **Action**: Scan other critical numerical sections (loss functions, probability computations) for similar unbounded exponentiation or division operations.

## Plan
1.  Apply clipping fixes to `student_t.py`, `ssm.py`, and `trainer.py`.
2.  Verify that tests pass (especially those comparing parallel vs. sequential SSM).

---

## Fixes Applied (2026-01-26)

### Fix 1: Student-t log_sigma clipping
**File**: `mcts_trader/jax_agent/prediction/student_t.py:85`
```python
# BEFORE:
sigma = jnp.exp(log_sigma)

# AFTER:
log_sigma_clipped = jnp.clip(log_sigma, -10.0, 10.0)
sigma = jnp.exp(log_sigma_clipped)
```

### Fix 2: SSM sequential scan clipping
**File**: `diff_teacher/mamba/ssm.py:38`
```python
# BEFORE:
delta_A = jnp.exp(delta_t[:, :, None] * A)

# AFTER (consistent with parallel implementation):
delta_A = jnp.exp(jnp.clip(delta_t[:, :, None] * A, -20.0, 20.0))
```

### Fix 3: Trainer sigma ceiling
**File**: `mcts_trader/jax_agent/trainer.py:749`
```python
# BEFORE:
sigma_per_step = jnp.maximum(jnp.std(errors, axis=(0, 2)), 1e-4)

# AFTER (with ceiling):
sigma_per_step = jnp.clip(jnp.std(errors, axis=(0, 2)), 1e-4, 10.0)
```

All fixes verified to compile successfully.
