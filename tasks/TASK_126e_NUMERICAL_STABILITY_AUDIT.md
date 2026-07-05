# TASK_126e: General Numerical Stability Audit

**Status**: AUDITED (Issues Found)
**Priority**: Medium
**Created**: 2026-01-31

## Scope

Audited files:
- `mcts_trader/jax_agent/trainer.py`
- `mcts_trader/jax_agent/path_solver.py`
- `mcts_trader/env/jax_trading_env.py`
- Related files in `mcts_trader/jax_agent/`

## Issues Found

### Category 1: Division Without Epsilon Guards

#### CRITICAL: `exploration/epsilon_greedy.py:39`
```python
mixed_policy = mixed_policy / jnp.sum(mixed_policy, axis=-1, keepdims=True)
```
**Risk**: If `mixed_policy` is all zeros (e.g., invalid action mask), division by zero.
**Fix**: `mixed_policy / jnp.maximum(jnp.sum(...), 1e-10)`

#### CRITICAL: `exploration/dirichlet.py:13`
```python
return gamma_samples / jnp.sum(gamma_samples, axis=-1, keepdims=True)
```
**Risk**: Gamma samples could sum to zero in degenerate cases.
**Fix**: Add epsilon guard.

#### LOW: `__init__.py:71`
```python
return (VALUE_SUPPORT_MAX - VALUE_SUPPORT_MIN) / (VALUE_SUPPORT_SIZE - 1)
```
**Risk**: Only if VALUE_SUPPORT_SIZE = 1 (static, unlikely).
**Status**: Acceptable (static values).

### Category 2: Log of Potentially Zero/Negative Values

#### GUARDED: `jax_trading_env.py:1202`
```python
log_return = jnp.log(safe_new / safe_prev) * 100.0
```
**Status**: Safe - `safe_new` and `safe_prev` are guarded with `jnp.maximum(..., 1e-8)`.

#### GUARDED: `jax_trading_env.py:1394`
```python
returns = jnp.log(current_price / (prev_price + 1e-8))
```
**Status**: Safe - has epsilon guard.

#### GUARDED: `observation_encoder.py:78`
```python
log_returns = jnp.log(close_p[:, 1:] / jnp.maximum(close_p[:, :-1], eps))
```
**Status**: Safe - has jnp.maximum guard.

#### GUARDED: `path_solver.py:172`
```python
mask = jnp.maximum(1.0 - selection, 1e-10)
next_scores = scaled + jnp.log(mask)
```
**Status**: Safe - mask is guarded before log.

#### RISKY: `path_solver.py:484`
```python
log_agent = jnp.log(agent_policy + 1e-10)
```
**Status**: Safe - has epsilon guard.

### Category 3: Exp of Large Values (Overflow Risk)

#### GOOD: `path_solver.py:291`
```python
P = jnp.exp(jnp.clip(log_P, -20.0, 20.0))
```
**Status**: Properly clipped.

#### RISKY: `fast_generator.py:317`
```python
prices = jnp.exp(log_prices)
```
**Risk**: If `log_prices` is large (e.g., > 88), overflow to inf.
**Fix**: `jnp.exp(jnp.clip(log_prices, -20.0, 20.0))`

#### GUARDED: `sac_entropy.py:110, 131, 146`
```python
alpha = jnp.exp(log_alpha)
```
**Status**: `log_alpha` is clipped elsewhere (`clip_min=-10, clip_max=0`).

### Category 4: Sqrt of Potentially Negative Values

#### RISKY: `observation_encoder.py:224`
```python
return jnp.sqrt(var)
```
**Risk**: Variance could be negative due to numerical errors.
**Fix**: `jnp.sqrt(jnp.maximum(var, 0.0))` or `jnp.sqrt(var + 1e-10)`

#### RISKY: `prediction/tree_search.py:457, 754`
```python
uncertainty = jnp.sqrt(total_variance) + jnp.sqrt(avg_asset_variance) + 0.5 * var_spread
```
**Risk**: Variances could be negative.
**Fix**: Add `jnp.maximum(..., 0.0)` guards.

#### GOOD: `redo.py:98`
```python
std = np.sqrt(np.maximum(var, 0))
```
**Status**: Properly guarded.

#### GOOD: `popart.py:68, 106, 126, 151, 326`
```python
std = jnp.sqrt(state.var + 1e-8)
```
**Status**: Properly guarded with epsilon.

### Category 5: jnp.where Branch Evaluation (JAX Pitfall)

**Background**: In JAX, `jnp.where(cond, x, y)` evaluates BOTH `x` and `y` before selecting. If either branch produces NaN/Inf, it can propagate even when that branch isn't selected.

#### RISKY: `jax_trading_env.py:1203`
```python
reward = jnp.where(is_liquidated, symlog(-50.0), symlog(log_return))
```
**Risk**: If `log_return` is NaN (e.g., from 0/0), the NaN propagates even when `is_liquidated=True`.
**Fix**: Guard `log_return` before the where, or use `jnp.nan_to_num()`.

#### EXTENSIVE USE: `jax_trading_env.py` (100+ occurrences)
```python
new_cash = jnp.where(is_bankrupt, 0.0, new_cash)
new_pos_active = jnp.where(opening_long, True, state.pos_active)
# etc.
```
**Status**: Generally safe because both branches are valid values (not computed expressions that could overflow).

#### RISKY PATTERN: Nested where with division
```python
# Example from jax_trading_env.py:783-784
new_cash = jnp.where(closing_short, cash_after_close_short,
                    jnp.where(opening_long | adding_long, cash_after_buy, state.cash))
```
**Status**: Safe if `cash_after_close_short` and `cash_after_buy` are pre-computed and guarded.

### Category 6: NaN Propagation in Losses

#### trainer.py:1103, 1159
```python
per_sample_entropy = -jnp.sum(policy_probs * jnp.log(policy_probs + 1e-10), axis=-1)
```
**Status**: Guarded with `+ 1e-10`.

#### trainer.py:704, 927 (Huber Loss)
```python
per_sample_mse = jnp.where(abs_error <= delta, quadratic, linear)
```
**Status**: Safe - both branches are finite for finite inputs.

### Category 7: Gradient Explosion Points

#### trainer.py:211
```python
scale = jnp.minimum(1.0, max_norm / (norm + 1e-8))
```
**Status**: Used in gradient clipping - properly guarded.

#### trainer.py:554
```python
optax.clip_by_global_norm(10.0),
```
**Status**: Gradient clipping is applied in optimizer chain.

## Summary Table

| File | Issue | Severity | Status |
|------|-------|----------|--------|
| `exploration/epsilon_greedy.py:39` | Division by zero | HIGH | **NEEDS FIX** |
| `exploration/dirichlet.py:13` | Division by zero | HIGH | **NEEDS FIX** |
| `fast_generator.py:317` | Exp overflow | MEDIUM | **NEEDS FIX** |
| `observation_encoder.py:224` | Sqrt of negative | MEDIUM | **NEEDS FIX** |
| `prediction/tree_search.py:457,754` | Sqrt of negative | MEDIUM | **NEEDS FIX** |
| `jax_trading_env.py:1203` | NaN in where branch | MEDIUM | **NEEDS FIX** |
| All other locations | Various | LOW | Properly guarded |

## Recommended Fixes

### Fix 1: epsilon_greedy.py:39
```python
# BEFORE:
mixed_policy = mixed_policy / jnp.sum(mixed_policy, axis=-1, keepdims=True)

# AFTER:
mixed_policy = mixed_policy / jnp.maximum(jnp.sum(mixed_policy, axis=-1, keepdims=True), 1e-10)
```

### Fix 2: dirichlet.py:13
```python
# BEFORE:
return gamma_samples / jnp.sum(gamma_samples, axis=-1, keepdims=True)

# AFTER:
return gamma_samples / jnp.maximum(jnp.sum(gamma_samples, axis=-1, keepdims=True), 1e-10)
```

### Fix 3: fast_generator.py:317
```python
# BEFORE:
prices = jnp.exp(log_prices)

# AFTER:
prices = jnp.exp(jnp.clip(log_prices, -20.0, 20.0))
```

### Fix 4: observation_encoder.py:224
```python
# BEFORE:
return jnp.sqrt(var)

# AFTER:
return jnp.sqrt(jnp.maximum(var, 1e-10))
```

### Fix 5: tree_search.py:457 and 754
```python
# BEFORE:
uncertainty = jnp.sqrt(total_variance) + jnp.sqrt(avg_asset_variance) + 0.5 * var_spread

# AFTER:
uncertainty = jnp.sqrt(jnp.maximum(total_variance, 0.0)) + \
              jnp.sqrt(jnp.maximum(avg_asset_variance, 0.0)) + \
              0.5 * var_spread
```

### Fix 6: jax_trading_env.py:1202-1203
```python
# BEFORE:
log_return = jnp.log(safe_new / safe_prev) * 100.0
reward = jnp.where(is_liquidated, symlog(-50.0), symlog(log_return))

# AFTER:
log_return = jnp.log(safe_new / safe_prev) * 100.0
log_return = jnp.where(jnp.isfinite(log_return), log_return, 0.0)  # Guard before where
reward = jnp.where(is_liquidated, symlog(-50.0), symlog(log_return))
```

## JAX-Specific Best Practices

1. **Always guard before jnp.where**: If either branch involves computation that could produce NaN/Inf, guard the inputs first.

2. **Use jnp.maximum for denominators**: `x / jnp.maximum(y, 1e-8)` instead of `x / (y + 1e-8)` when y should never be negative.

3. **Clip before exp**: `jnp.exp(jnp.clip(x, -20, 20))` prevents overflow.

4. **Guard sqrt inputs**: `jnp.sqrt(jnp.maximum(x, 0))` or `jnp.sqrt(x + 1e-8)`.

5. **Use jnp.nan_to_num sparingly**: It can hide bugs. Better to fix the source.

## Related Tasks

- TASK_118: Numerical Stability Audit (previous audit, focused on different areas)
- TASK_126a: Reward Spike Investigation (related to reward computation)
- TASK_115c: BUGHUNT Numerical Stability
