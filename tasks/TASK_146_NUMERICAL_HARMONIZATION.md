# TASK_146: Numerical Harmonization (Dilution & Scale Fix)

**Status:** COMPLETE

**Completed:** 2026-02-10

**Objective:** Correct the target normalization and value scaling to provide an "honest" and strong learning signal.

---

## Bug Validation Summary

| Bug | Status | Evidence |
|-----|--------|----------|
| 1. Oracle 320-way softmax | **CONFIRMED** | `path_solver.py:607` uses single softmax |
| 2. Value scale overflow | **CONFIRMED** | remaining_pnl up to 9.25 × 10 = 92.5, clipped to 10 |
| 3. MSE unit mismatch | **CONFIRMED** | Comparing [-1,1] to [0,9+] |

---

## Bug 1: Oracle Targets Use 320-Way Softmax

**Location:** `mcts_trader/jax_agent/path_solver.py:607`

**Current (BROKEN):**
```python
policy_targets = jax.nn.softmax(logits, axis=-1)  # 320-way competition
```

**Problem:** Oracle targets have all 64 assets competing for probability mass. Even though we fixed the LOSS to be per-asset, the TARGETS themselves are wrong.

**Fix:**
```python
# Reshape for per-asset softmax
n_assets = logits.shape[-1] // n_actions_per_asset
logits_reshaped = logits.reshape(T, n_assets, n_actions_per_asset)
probs_reshaped = jax.nn.softmax(logits_reshaped, axis=-1)  # 5-way per asset
policy_targets = probs_reshaped.reshape(T, -1)  # Flatten back to 320
```

---

## Bug 2: Value Scale Causes Clipping

**Evidence:**
```
remaining_pnl range: [0, 9.25] (raw)
With value_scale=10: [0, 92.5] (scaled)
Support range: [-10, 10] (categorical bins)
After clipping: everything > 1.0 raw → maps to same bin!
```

**Problem:**
- remaining_pnl is cumulative log return, can reach 9+ over 400 steps
- Multiplying by value_scale=10 puts it way outside [-10, 10] support
- Clipping loses all information about returns > 1.0

**Fix:** Set `value_scale = 1.0` (values already fit in ~[0, 10] range)

```python
# easy_config.py
value_scale: float = 1.0  # Was 10.0, caused overflow clipping
```

---

## Bug 3: MSE Unit Mismatch

**Current (BROKEN):**
```python
value_pred = self._value_to_scalar(value_logits) / self.config.value_scale
value_mse = jnp.mean((value_pred - batch['returns']) ** 2)
```

**Problem:**
- Model predicts in [-10, 10] scaled space
- Dividing by 10 gives [-1, 1]
- Comparing to batch['returns'] which is [0, 9+]
- MSE of ~24 makes sense: (1 - 5)² = 16

**Fix:** With value_scale=1.0, this becomes correct:
- Model predicts in [-10, 10]
- Dividing by 1.0 gives [-10, 10]
- batch['returns'] is [0, 9] - compatible!

---

## Implementation Plan

### Fix 1: Per-Asset Softmax in Oracle (CRITICAL)
**File:** `mcts_trader/jax_agent/path_solver.py`

```python
# Line ~607, in optimal_path_to_policy:
# Replace:
#   policy_targets = jax.nn.softmax(logits, axis=-1)
# With:
n_assets = logits.shape[-1] // n_actions_per_asset
logits_64x5 = logits.reshape(T, n_assets, n_actions_per_asset)
probs_64x5 = jax.nn.softmax(logits_64x5, axis=-1)
policy_targets = probs_64x5.reshape(T, -1)
```

### Fix 2: Value Scale = 1.0
**File:** `easy_config.py`

```python
value_scale: float = 1.0  # TASK_146: Was 10.0, caused clipping
```

### Fix 3: MSE Automatically Fixed
With value_scale=1.0, the MSE calculation becomes correct.

---

## Expected Impact After Fix

| Metric | Before | After |
|--------|--------|-------|
| Policy Loss | ~0.016 (suspiciously low) | ~0.5-1.5 (realistic) |
| Value MSE | ~24 (nonsensical) | < 5.0 (meaningful) |
| Agreement | ~90% (gaming targets) | Lower initially, then climb |

---

## Root Cause

The 320-way softmax in oracle targets meant:
- Even with correct per-asset loss, model learned to match WRONG targets
- Oracle targets had probability spread across all 320 actions
- Model easily matched this "spread" pattern → low loss, fake agreement

---

## Implementation Summary (2026-02-10)

### Changes Made

| Fix | File | Line | Change |
|-----|------|------|--------|
| Per-asset softmax | `path_solver.py` | 607 | Reshape to (T,64,5), softmax axis=-1, reshape back |
| Value scale | `easy_config.py` | 43 | `value_scale: 1.0` (was 10.0) |

### Code Changes

**path_solver.py:607** (optimal_path_to_policy):
```python
# TASK_146: Per-asset softmax (each asset has independent 5-way classification)
# Previously: 320-way softmax forced all assets to compete for probability mass
# Now: Reshape to (T, n_assets, 5), softmax per asset, reshape back
logits_reshaped = logits.reshape(T, n_assets, n_actions_per_asset)
probs_reshaped = jax.nn.softmax(logits_reshaped, axis=-1)  # 5-way per asset
policy_targets = probs_reshaped.reshape(T, n_actions)
```

**easy_config.py**:
```python
value_scale: float = 1.0  # TASK_146: Was 10.0, caused overflow clipping
```

### Verification

Run training and expect:
- Policy loss > 0.1 (was 0.016 - suspiciously low)
- Value MSE < 10.0 (was 24.0 - units mismatch)
- Agreement starts lower but climbs meaningfully
