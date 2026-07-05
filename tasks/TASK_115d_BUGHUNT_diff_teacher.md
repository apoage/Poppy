# Bug Hunt Report: diff_teacher/

**Date**: 2026-01-25
**Auditor**: Gemini
**Scope**: Numerical instability (NaN/Inf, div by zero, log/exp overflow)

## Summary

| ID | File | Severity | Description |
|----|------|----------|-------------|
| BUG-001 | diff_teacher/flow_matching.py | HIGH | Division by zero in time embedding scaling for small dimensions |
| BUG-002 | diff_teacher/conv_flow_matching.py | HIGH | Division by zero in time embedding scaling for small dimensions |
| BUG-003 | diff_teacher/flow_matching.py | MEDIUM | Division by zero if num_steps is 0 |
| BUG-004 | diff_teacher/conv_flow_matching.py | MEDIUM | Division by zero if num_steps is 0 |
| BUG-005 | diff_teacher/data_loader.py | LOW | Unsafe division by epsilon-adjusted price |
| BUG-006 | diff_teacher/optimal_solver.py | LOW | Unsafe division by epsilon-adjusted price |
| BUG-007 | diff_teacher/generation/streaming.py | MEDIUM | Division by zero if num_steps is 0 |

---

## Details

#### BUG-001: Time Embedding Division by Zero
**File**: `diff_teacher/flow_matching.py:40-41`
**Severity**: HIGH
**Status**: OPEN

```python
half_dim = self.dim // 2
emb_scale = jnp.log(10000.0) / (half_dim - 1)
```

**Problem**: If `self.dim` is 2 or 3, `half_dim` becomes 1. `half_dim - 1` becomes 0, causing division by zero. While default dimensions are usually larger, this makes the module brittle to configuration changes or small tests.

**Fix**: Ensure denominator is non-zero.
```python
emb_scale = jnp.log(10000.0) / (jnp.maximum(half_dim - 1, 1.0))
```

#### BUG-002: Time Embedding Division by Zero (Conv)
**File**: `diff_teacher/conv_flow_matching.py:47-48`
**Severity**: HIGH
**Status**: OPEN

```python
half_dim = self.dim // 2
emb_scale = jnp.log(10000.0) / (half_dim - 1)
```

**Problem**: Same as BUG-001 but in the convolutional implementation.

**Fix**: 
```python
emb_scale = jnp.log(10000.0) / (jnp.maximum(half_dim - 1, 1.0))
```

#### BUG-003: Sampling Steps Division by Zero
**File**: `diff_teacher/flow_matching.py:367`
**Severity**: MEDIUM
**Status**: OPEN

```python
dt = 1.0 / num_steps
```

**Problem**: `num_steps` is an integer argument. If 0 is passed, this raises a ZeroDivisionError.

**Fix**: Add assertion or guard.
```python
num_steps = max(num_steps, 1)
dt = 1.0 / num_steps
```

#### BUG-004: Sampling Steps Division by Zero (Conv)
**File**: `diff_teacher/conv_flow_matching.py:370`
**Severity**: MEDIUM
**Status**: OPEN

```python
dt = 1.0 / num_steps
```

**Problem**: Same as BUG-003.

**Fix**:
```python
num_steps = max(num_steps, 1)
dt = 1.0 / num_steps
```

#### BUG-005: Unsafe Price Return Calculation
**File**: `diff_teacher/data_loader.py:293` (and others)
**Severity**: LOW
**Status**: OPEN

```python
returns = np.diff(close, axis=0) / (close[:-1] + 1e-10)
```

**Problem**: If `close` is 0 (e.g., missing data padded with 0), `close + 1e-10` is very small (`1e-10`). If the numerator is non-zero (e.g., price jumps from 0 to 100), the result is massive (`1e12`). While often masked out later, the intermediate value can overflow or cause issues.

**Fix**: Use `np.maximum` for safer epsilon usage.
```python
returns = np.diff(close, axis=0) / np.maximum(close[:-1], 1e-10)
```

#### BUG-006: Unsafe Sharpe Calculation
**File**: `diff_teacher/optimal_solver.py:203`
**Severity**: LOW
**Status**: OPEN

```python
returns = np.diff(prices, axis=0) / (prices[:-1] + 1e-10)
```

**Problem**: Same pattern as BUG-005. Can produce Inf/NaN if prices are near 0.

**Fix**:
```python
returns = np.diff(prices, axis=0) / np.maximum(prices[:-1], 1e-10)
```

#### BUG-007: Sampling Steps Division by Zero (Streaming)
**File**: `diff_teacher/generation/streaming.py:115` & `177`
**Severity**: MEDIUM
**Status**: OPEN

```python
dt = 1.0 / n_steps
```

**Problem**: `n_steps` is an integer argument from `StreamConfig`. If 0 is passed, this raises a ZeroDivisionError.

**Fix**:
```python
dt = 1.0 / max(n_steps, 1)
```

---
**Note**: Mamba implementation (`diff_teacher/mamba/`) was audited and found to be numerically stable. `softplus` ensures positive deltas, and `-exp(A_log)` ensures negative A matrix, preventing state explosion.