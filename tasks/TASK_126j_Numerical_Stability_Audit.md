# TASK_126j: Numerical Stability Audit

## Executive Summary

This audit identifies **numerical stability risks** in the JAX trading system. Most issues are mitigated, but several edge cases and inconsistencies could cause NaN propagation, gradient explosions, or incorrect calculations under extreme market conditions.

**Risk Level: MEDIUM** - No critical immediate bugs found, but defensive improvements recommended.

---

## Findings by Category

### 1. ✅ GOOD: Epsilon Guards on Division

Most division operations properly use epsilon guards:

```python
# trainer.py:211 - Gradient clipping (GOOD)
scale = jnp.minimum(1.0, max_norm / (norm + 1e-8))

# trainer.py:766 - Batch normalization (GOOD)
num_valid = jnp.maximum(jnp.sum(valid_mask_float), 1.0)

# trainer.py:1060, 1068 - Alpha ratio computation (GOOD)
alpha_ratio = raw_reward_sum / jnp.maximum(max_alpha, 1e-6)

# jax_trading_env.py:744, 870 - Share calculation (GOOD)
shares_to_buy = actual_cost / jnp.maximum(buy_exec_price, 1e-10)
```

### 2. ⚠️  MODERATE RISK: Missing Epsilon Guards

#### Issue 2.1: `shares` calculation in limit orders

**Location:** `jax_trading_env.py:950`
```python
# MISSING EPSILON:
shares = order_value / limit_price
```

**Risk:** If `limit_price` is 0 (malformed data), this creates NaN.

**Fix:**
```python
shares = order_value / jnp.maximum(limit_price, 1e-10)
```

#### Issue 2.2: `shares_for_short` calculation

**Location:** `jax_trading_env.py:1012`
```python
# MISSING EPSILON:
shares_for_short = order_value / limit_price
```

**Same fix as above.**

#### Issue 2.3: Division in price normalization

**Location:** `jax_trading_env.py:1396`
```python
# MISSING EPSILON:
returns_normalized = returns_clipped / 0.1
```

**Risk:** This is actually safe (0.1 is constant), but pattern is inconsistent.

#### Issue 2.4: Fee calculations without epsilon

**Location:** Multiple places in `jax_trading_env.py`
```python
# Lines 348-349, 525, 756, 862, 873
fee_rate = config.fee_bps / 10000.0
slippage_rate = config.slippage_bps / 10000.0
maker_fee_rate = config.maker_fee_bps / 10000.0
```

**Risk:** These are constant divisions (safe), but if config values are ever dynamic/JAX-traced, could cause issues.

### 3. ✅ GOOD: Log/Exp Protection

#### Issue 3.1: Logarithm inputs are guarded

```python
# jax_trading_env.py:1202 - Reward computation (GOOD)
safe_prev = jnp.maximum(state.prev_portfolio_value, 1e-8)
safe_new = jnp.maximum(new_portfolio_value, 1e-8)
log_return = jnp.log(safe_new / safe_prev) * 100.0

# path_solver.py:172, 291 - Soft top-k (GOOD)
next_scores = scaled + jnp.log(mask)  # mask has minimum 1e-10
P = jnp.exp(jnp.clip(log_P, -20.0, 20.0))  # Clipped before exp
```

#### Issue 3.2: Exp overflow protection

```python
# path_solver.py:291 - Sinkhorn (GOOD)
P = jnp.exp(jnp.clip(log_P, -20.0, 20.0))  # Prevents overflow

# path_solver.py:274 - Log-domain Sinkhorn (GOOD)
log_K = jnp.clip(log_K, -20.0, 20.0)  # Prevents underflow/overflow
```

### 4. ⚠️  MODERATE RISK: Student-t Loss Numerical Issues

**Location:** `trainer.py:904-905`

```python
# Student-t NLL computation
z = errors / sigma  # sigma is clipped to [1e-4, 10.0] ✓
student_t_nll = jnp.log(sigma) + 0.5 * (df + 1) * jnp.log1p(z ** 2 / df)
```

**Analysis:**
- ✅ `sigma` is clipped: `jnp.clip(..., 1e-4, 10.0)` 
- ✅ Uses `jnp.log1p()` for numerical stability
- ⚠️ `z ** 2` can overflow if errors are large (>1e19)
- ⚠️ `df` (degrees of freedom) could theoretically be < 2 (infinite variance)

**Recommended Fix:**
```python
# Clip z to prevent overflow in z**2
z = jnp.clip(errors / sigma, -1e9, 1e9)  # Prevent overflow
# Or use stable formulation
student_t_nll = jnp.log(sigma) + 0.5 * (df + 1) * jnp.log1p((errors/sigma) ** 2 / df)
```

### 5. ⚠️  LOW RISK: Power Operations Without Clipping

**Location:** `trainer.py:934`

```python
# Step weight decay
step_weights = jnp.array([config.step_decay ** i for i in range(k)])
```

**Risk:** If `config.step_decay > 1.0` and `k` is large, this could overflow. Currently `step_decay=0.9` (safe).

**Fix:** Add validation or clipping:
```python
step_decay = jnp.clip(config.step_decay, 0.0, 1.0)
step_weights = jnp.array([step_decay ** i for i in range(k)])
```

### 6. ✅ GOOD: NaN/Inf Detection

**Location:** `trainer.py:1579-1585, 1634`

```python
# Batch validation (GOOD)
if jnp.isnan(v).any() or jnp.isinf(v).any():
    raise ValueError(f"NaNs/Infs detected in training batch key '{k}'!")

# Loss check (GOOD)
if (iteration + 1) % 50 == 0 and jnp.isnan(metrics['total_loss']):
    raise ValueError("Training stopped due to NaN loss.")
```

### 7. ⚠️  LOW RISK: jnp.where Evaluates Both Branches

**JAX Pitfall:** `jnp.where(condition, x, y)` evaluates BOTH `x` and `y` before selecting.

**Safe usages found:**
```python
# jax_trading_env.py:224-226 - Safe (no side effects)
values = jnp.where(pos_side == 1, long_value,
                  jnp.where(pos_side == 2, short_value, 0.0))

# trainer.py:704, 927 - Huber loss (Safe)
per_sample_mse = jnp.where(abs_error <= delta, quadratic, linear)
```

**Potential issue:**
```python
# jax_trading_env.py:1203
reward = jnp.where(is_liquidated, symlog(-50.0), symlog(log_return))
```

Both `symlog(-50.0)` and `symlog(log_return)` are evaluated. If `log_return` is NaN, the output might be NaN even when `is_liquidated=True`.

**Fix:** Use `jax.lax.cond` for true branching (but can't JIT easily) or ensure both branches are safe:
```python
# Ensure log_return is safe even when not used
log_return = jnp.where(is_liquidated, 0.0, log_return)  # Dummy value
reward = jnp.where(is_liquidated, symlog(-50.0), symlog(log_return))
```

### 8. ⚠️  MODERATE RISK: Division by Small Numbers in Path Solver

**Location:** `path_solver.py:193, 312`

```python
# PnL calculation (both methods)
step_pnl = jnp.sum(abs_returns * selection) / k
```

`k` is static argument, so this is safe unless `k=0` (which shouldn't happen).

**Location:** `path_solver.py:179, 309`

```python
# Normalization
return final_selection / jnp.maximum(jnp.sum(final_selection), 1e-6) * k
```

✅ Already has epsilon guard.

### 9. ⚠️  LOW RISK: Log(0) in Cross-Entropy

**Location:** `trainer.py:1103, 1159`

```python
per_sample_entropy = -jnp.sum(policy_probs * jnp.log(policy_probs + 1e-10), axis=-1)
```

✅ Protected with `+ 1e-10` epsilon.

But what if `policy_probs` becomes exactly 0 due to masking?
```python
# If policy_probs = 0, log(0 + 1e-10) = log(1e-10) = -23
# Multiply by 0 = 0, so safe
```

✅ Safe due to multiplication by 0.

### 10. 🔴 HIGH RISK: Inverse Symlog Overflow

**Location:** `mcts_trader/utils/transforms.py:21-31`

```python
def inverse_symlog(y: Array, threshold: float = 1.0) -> Array:
    """x = sign(y) * (exp(|y|/threshold) - 1) * threshold"""
    return jnp.sign(y) * (jnp.expm1(np.abs(y) / threshold)) * threshold
```

**Risk:** If `|y| > 709` (for float32), `exp(|y|)` overflows to `inf`.

**Impact:** Value head outputs could theoretically exceed this range if not properly clipped.

**Fix:** Add clipping before exp:
```python
def inverse_symlog(y: Array, threshold: float = 1.0) -> Array:
    # Clip to prevent overflow: exp(88) ≈ 1e38 (float32 max ~3e38)
    y_clipped = jnp.clip(y, -88.0, 88.0)  # For float32
    return jnp.sign(y) * (jnp.expm1(np.abs(y_clipped) / threshold)) * threshold
```

### 11. ⚠️  MODERATE RISK: Variance Computation with Single Sample

**Location:** `trainer.py:725, 1622`

```python
pred_std = jnp.std(predicted_values)
tgt_std = jnp.std(target_values)
```

If `predicted_values` has only one unique value (e.g., model collapsed), std = 0.
Subsequent division by std could be problematic if not guarded.

**Currently safe** (no division by std found), but worth monitoring.

### 12. ✅ GOOD: Gradient Clipping

**Location:** `trainer.py:158-242, 347`

```python
# Global gradient clipping
optax.clip_by_global_norm(10.0)

# Per-head gradient clipping (TASK_025)
def _clip_gradients_per_head(grads, max_norm_value, ...)
```

✅ Multiple layers of gradient protection.

---

## Summary of Issues

| Severity | Count | Issues |
|----------|-------|--------|
| 🔴 High | 1 | `inverse_symlog` overflow risk |
| 🟡 Medium | 4 | Missing epsilon on `shares` calc, Student-t overflow, `jnp.where` NaN propagation |
| 🟢 Low | 3 | Power overflow, step_decay validation, variance edge case |
| ✅ Good | 15+ | Most divisions, logs, exps properly guarded |

---

## Recommended Fixes (Priority Order)

### 1. 🔴 CRITICAL: Fix `inverse_symlog` Overflow

**File:** `mcts_trader/utils/transforms.py`

```python
def inverse_symlog(y: Array, threshold: float = 1.0) -> Array:
    """
    Inverse symmetric logarithmic transformation.
    Clipped to prevent overflow.
    """
    # exp(88) ≈ 1.6e38, close to float32 max (~3.4e38)
    # Leave margin for (exp - 1) * threshold calculation
    max_input = 88.0 if y.dtype == jnp.float32 else 709.0
    y_clipped = jnp.clip(y, -max_input, max_input)
    
    return jnp.sign(y) * (jnp.expm1(np.abs(y_clipped) / threshold)) * threshold
```

### 2. 🟡 MEDIUM: Add Epsilon to `shares` Calculation

**File:** `mcts_trader/env/jax_trading_env.py`

```python
# Line 950
shares = order_value / jnp.maximum(limit_price, 1e-10)

# Line 1012
shares_for_short = order_value / jnp.maximum(limit_price, 1e-10)
```

### 3. 🟡 MEDIUM: Clip `z` in Student-t Loss

**File:** `mcts_trader/jax_agent/trainer.py:904`

```python
z = jnp.clip(errors / sigma, -1e9, 1e9)  # Prevent overflow in z**2
```

### 4. 🟢 LOW: Add `jnp.where` Safety Comment

**File:** `mcts_trader/env/jax_trading_env.py:1203`

```python
# NOTE: jnp.where evaluates both branches. Ensure log_return is safe.
# If log_return is NaN, could contaminate result even when is_liquidated=True.
reward = jnp.where(is_liquidated, symlog(-50.0), symlog(log_return))
```

Or add explicit safety:
```python
# Ensure both branches are safe
safe_log_return = jnp.where(is_liquidated, 0.0, log_return)  # Dummy when liquidated
reward = jnp.where(is_liquidated, symlog(-50.0), symlog(safe_log_return))
```

### 5. 🟢 LOW: Add `step_decay` Validation

**File:** `mcts_trader/jax_agent/trainer.py:934`

```python
step_decay = jnp.clip(config.step_decay, 0.0, 1.0)
step_weights = jnp.array([step_decay ** i for i in range(k)])
```

---

## Testing Recommendations

1. **NaN/Inf Injection Test:**
   ```python
   # Test that the system handles NaN inputs gracefully
   def test_nan_handling():
       nan_obs = jnp.full((1, 27, 64, 64), jnp.nan)
       # Should not crash, should detect and raise
   ```

2. **Overflow Test:**
   ```python
   # Test extreme values
   large_value = jnp.array([1e38])
   result = inverse_symlog(symlog(large_value))
   # Should handle gracefully without overflow
   ```

3. **Edge Case Test:**
   ```python
   # Test zero portfolio value
   zero_portfolio = jnp.array(0.0)
   # Should not produce NaN in reward computation
   ```

---

## Acceptance Criteria

- [ ] `inverse_symlog` has overflow protection
- [ ] `shares` calculation in limit orders has epsilon guard
- [ ] Student-t loss `z` value is clipped
- [ ] All division operations have epsilon guards (audit complete)
- [ ] NaN/Inf detection tests pass
- [ ] Extreme value tests pass

---

## Related Tasks

- TASK_037-043: Stability modules
- TASK_116: Student-t output distribution
- TASK_126f: Catastrophic reward drop fix
- TASK_126h: Path solver PnL bug fix
- TASK_126i: Return scale mismatch fix
