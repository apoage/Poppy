# TASK_127k3: PopArt Normalization Deep Audit

## Executive Summary

This audit traces the PopArt normalization flow through the training pipeline. **Critical architectural incompatibilities** exist between PopArt's weight correction mechanism and the categorical value head design. PopArt assumes the network outputs unnormalized scalar values, but the current architecture outputs categorical logits over a fixed support range.

**Risk Level: HIGH** - PopArt weight correction may not work as intended with categorical value heads, potentially causing training instability.

---

## PopArt Overview

### Standard PopArt Algorithm

```
Standard PopArt (from DeepMind paper):
1. Network outputs: v(h) = unnormalized scalar value
2. PopArt normalizes targets: y' = (y - μ) / σ
3. Loss computed: L = (v(h) - y')²
4. When (μ, σ) change:
   - Apply weight correction: W_new = W_old × (σ_old/σ_new)
   - This preserves: v_new(h) ≈ v_old(h) for all h
```

### Current Implementation

```
Current Implementation:
1. Network outputs: logits over support [-10, 10] (categorical)
2. _logits_to_scalar: v(h) = Σ softmax(logits) × support
3. PopArt normalizes targets: y' = (y - μ) / σ
4. Loss computed: L = (v(h) - y')²
5. Weight correction applied to final layer
```

**Key Difference:** Standard PopArt uses scalar output heads; current implementation uses categorical distributions.

---

## Critical Findings

### 🔴 CRITICAL: PopArt Weight Correction May Be Incompatible with Categorical Value Heads

**Standard PopArt Weight Correction:**
```python
# When statistics change from (μ, σ) to (μ', σ'):
W_new = W_old × (σ_old / σ_new)
b_new = b_old × (σ_old / σ_new) + (μ_old - μ_new) / σ_new

# This ensures:
# v_new(h) = W_new · h + b_new
#          = (W_old × σ_old/σ') · h + (b_old × σ_old/σ' + (μ_old-μ')/σ')
#          = (W_old · h + b_old) × σ_old/σ' + (μ_old-μ')/σ'
#          ≈ v_old(h)  (when normalized appropriately)
```

**Current Categorical Head:**
```python
# Value head outputs logits over support bins
logits = value_head(hidden_state)  # (batch, 401)
probs = softmax(logits)
value = sum(probs * support)  # Expected value in [-10, 10]
```

**Problem:** 
- PopArt correction applies to the final linear layer weights
- But the output is logits, not scalar values
- Correcting logits by `scale = σ_old/σ_new` doesn't preserve the expected value

**Example:**
```python
# Before correction:
logits = [0, 0, 10, 0, ...]  # Sharp peak at bin 2
probs = softmax(logits) ≈ [0, 0, 0.99, 0, ...]
value = 0×(-10) + 0×(-9.95) + 0.99×(-9.9) + ... ≈ -9.9

# After correction (multiply logits by scale=2.0):
logits_new = [0, 0, 20, 0, ...]
probs_new = softmax(logits_new) ≈ [0, 0, 0.9999..., 0, ...]
value_new ≈ -9.9  # Essentially unchanged!

# PopArt expected the value to scale by σ_old/σ_new
# But categorical distribution is invariant to logit scaling (mostly)
```

**Impact:**
- Weight correction doesn't preserve network outputs as intended
- When PopArt statistics change, the network may need to relearn value scale
- Training instability during curriculum transitions

**Fix Options:**

**Option 1: Remove PopArt or make it update-only (no weight correction)**
```python
def normalize_and_correct(self, targets, params):
    # Update statistics only
    self.state = update_statistics(self.state, targets, self.beta)
    # Normalize targets
    norm_targets = normalize_targets(targets, self.state)
    # Return params unchanged - no weight correction!
    return norm_targets, params
```

**Option 2: Use scalar value head instead of categorical**
```python
# Change value head from categorical to scalar
value = nn.Dense(1)(hidden_state)  # Scalar output
# Then PopArt correction works as intended
```

**Option 3: Apply correction to support bins, not weights**
```python
# Instead of correcting weights, adjust the support range
scale, shift = compute_weight_correction(self.state)
# Adjust support: new_support = (support - shift) / scale
# This is complex and may not work well
```

**Recommended:** Option 1 (disable weight correction) or Option 2 (scalar head).

---

### 🟡 HIGH: PopArt Statistics Update Outside JIT, But Used Inside JIT

**Location:** `trainer.py:1597-1623`

**Flow:**
```python
# Outside JIT (Python land):
if iteration % 10 == 0:
    valid_values = raw_values[valid_mask_flat]
    _, corrected_params = self.popart.normalize_and_correct(
        valid_values, self.state.params
    )
    self.state = self.state.replace(params=corrected_params)

norm_values = self.popart.normalize(raw_values)
batch = {**batch, 'values': norm_values.reshape(batch['values'].shape)}

# Inside JIT (train_step_fn):
loss, metrics = self.train_step_fn(self.state, batch, train_rng, entropy_coef)
```

**Problem:** The PopArt statistics (mean, std) are:
1. Updated outside JIT (Python) every 10 iterations
2. Used to normalize targets outside JIT
3. Normalized targets are passed to JIT-compiled function

This is **correct** for PopArt - the statistics should be treated as constants during gradient computation. However, there's a subtle issue:

**The weight correction happens outside JIT**, which means:
- Parameters are modified before entering JIT
- JIT sees different parameter values on different calls
- This triggers recompilation or causes cache misses

**Evidence of Performance Impact:**
```python
# Line 1598 comment:
"PERF: Only update PopArt stats every 10 iterations to reduce GPU sync overhead"
```

This suggests the PopArt updates are causing performance issues.

**Fix:** 
- Move weight correction inside JIT using `jax.lax.cond`
- Or apply correction as a learned affine transform within the network

---

### 🟡 MEDIUM: Value Head Outputs Fixed Range [-10, 10], But PopArt Normalizes Targets

**Issue:** The value support range is fixed at [-10, 10]:
```python
# From __init__.py
VALUE_SUPPORT_MIN = -10
VALUE_SUPPORT_MAX = 10
VALUE_SUPPORT_SIZE = 401
```

But PopArt normalizes targets to approximately N(0, 1).

**The Problem:**
- PopArt assumes the network learns to predict normalized targets
- Network outputs are constrained to [-10, 10] by support range
- If normalized targets exceed [-10, 10], they get clipped

**Example:**
```python
# Raw targets after curriculum jump: mean=50, std=20
# PopArt normalization: (y - 50) / 20
# Normalized targets: range [-5, +5] (OK)

# Later, raw targets: mean=100, std=50
# PopArt normalization: (y - 100) / 50
# Normalized targets: range [-4, +2] (still OK, but getting close to limits)
```

**Actually this seems OK** for typical value ranges, but if normalized targets approach ±10, clipping will occur.

**Mitigation:** Ensure PopArt converges quickly to keep normalized targets in [-5, 5] range.

---

### 🟢 LOW: Gradient Flow Through Normalization is Correct

**Verified:** The gradient flow through PopArt normalization is correct:

```python
# 1. Model outputs value_logits
initial_output = self.model.apply(...)
value_logits = initial_output['value']

# 2. Convert to scalar
predicted_values = self._logits_to_scalar(value_logits)
#    This is differentiable: sum(softmax(logits) * support)

# 3. Loss computed against normalized targets
per_sample_mse = (predicted_values - target_values) ** 2
#    target_values are already normalized (PopArt applied outside)

# 4. Gradient flows back:
#    loss → predicted_values → softmax → logits → model parameters
```

**No gradient blocking issue** - PopArt statistics are treated as constants (updated outside JIT), so gradients flow correctly through the value head.

---

## PopArt Call Chain

```
┌─────────────────────────────────────────────────────────────────────┐
│ OUTSIDE JIT (trainer.train_epoch)                                    │
│                                                                      │
│  1. Sample batch from replay buffer                                  │
│     batch = replay_buffer.sample_batch(rng)                          │
│                                                                      │
│  2. Flatten values: raw_values = batch['values'].flatten()           │
│                                                                      │
│  3. Every 10 iterations:                                             │
│     valid_values = raw_values[valid_mask]                            │
│     _, corrected_params = popart.normalize_and_correct(              │
│         valid_values, params                                         │
│     )                                                                │
│     # Updates PopArt statistics (mean, var)                          │
│     # Applies weight correction to value head                        │
│     params = corrected_params  ← MODIFIED OUTSIDE JIT!               │
│                                                                      │
│  4. Normalize targets using current stats:                           │
│     norm_values = popart.normalize(raw_values)                       │
│     batch['values'] = norm_values.reshape(...)                       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ INSIDE JIT (trainer.train_step_fn = jax.jit(_train_step))            │
│                                                                      │
│  5. Model forward:                                                   │
│     output = model.apply(params, observations[:, 0])                 │
│     value_logits = output['value']  # (batch, 401)                   │
│                                                                      │
│  6. Convert to scalar:                                               │
│     predicted_values = _logits_to_scalar(value_logits)               │
│     # = sum(softmax(logits) * support, axis=-1)                      │
│                                                                      │
│  7. Compute loss:                                                    │
│     target_values = batch['values'][:, 0]  # Already normalized!     │
│     loss = (predicted_values - target_values)²                       │
│                                                                      │
│  8. Gradient computation:                                            │
│     grads = jax.grad(loss)(params)                                   │
│     # Flows through: value head → dynamics → representation          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Issues Summary

| Issue | Severity | Location | Description |
|-------|----------|----------|-------------|
| Weight correction incompatible with categorical head | 🔴 Critical | `popart.py:160-250` | Logit scaling doesn't preserve expected values |
| Weight correction outside JIT | 🟡 High | `trainer.py:1613-1616` | May cause recompilation |
| Support range limits | 🟡 Medium | `__init__.py:51-53` | Normalized targets may clip |
| Gradient flow | 🟢 OK | `trainer.py:665-733` | Correctly flows through normalization |

---

## Recommendations

### Priority 1: Disable PopArt Weight Correction (Safest Fix)

**File:** `mcts_trader/jax_agent/popart.py:278-308`

```python
def normalize_and_correct(self, targets, params):
    """
    Update statistics and normalize targets.
    
    NOTE: Weight correction disabled for categorical value heads.
    See TASK_127k3 for details.
    """
    # Update statistics
    self.state = update_statistics(self.state, targets, self.beta)
    
    # Normalize targets
    norm_targets = normalize_targets(targets, self.state)
    
    # DISABLED: Weight correction incompatible with categorical value heads
    # The correction formula assumes scalar outputs, but we have categorical distributions
    # corrected_params = apply_weight_correction_flax(params, scale, shift)
    
    return norm_targets, params  # Return params unchanged
```

**Rationale:** 
- Weight correction is the problematic part
- Normalization alone still provides value (stable targets)
- Much simpler than changing architecture to scalar value head

### Priority 2: Monitor Normalized Target Distribution

Add logging to track if normalized targets are hitting support limits:

```python
# In trainer.py after normalization
if self.use_popart:
    norm_values = self.popart.normalize(raw_values)
    
    # Log distribution statistics
    metrics['popart/mean'] = self.popart.mean
    metrics['popart/std'] = self.popart.std
    metrics['popart/target_min'] = jnp.min(norm_values)
    metrics['popart/target_max'] = jnp.max(norm_values)
    
    # Warn if hitting limits
    if jnp.abs(jnp.max(norm_values)) > 8.0:  # Getting close to ±10
        print(f"⚠️  PopArt targets near support limit: max={jnp.max(norm_values):.2f}")
```

### Priority 3: Consider Scalar Value Head (Long-term)

If PopArt weight correction is needed for curriculum learning:

1. Change value head output from `(batch, 401)` to `(batch, 1)`
2. Remove categorical cross-entropy component from value loss
3. PopArt weight correction will work correctly
4. Simpler architecture, possibly better stability

**Trade-off:**
- Categorical: Better uncertainty estimation, distributional RL benefits
- Scalar: Simpler, works correctly with PopArt

---

## Testing Plan

1. **Weight Correction Effectiveness Test:**
   ```python
   # 1. Forward pass with current params
   out1 = model.apply(params, obs)
   value1 = popart.denormalize(_logits_to_scalar(out1['value']))
   
   # 2. Update PopArt statistics (simulate curriculum jump)
   popart.state = update_statistics(popart.state, new_targets)
   
   # 3. Apply weight correction
   scale, shift = compute_weight_correction(popart.state)
   corrected_params = apply_weight_correction_flax(params, scale, shift)
   
   # 4. Forward pass with corrected params
   out2 = model.apply(corrected_params, obs)
   value2 = popart.denormalize(_logits_to_scalar(out2['value']))
   
   # 5. Check if values are preserved
   diff = jnp.abs(value2 - value1)
   assert jnp.mean(diff) < 0.1, f"Weight correction failed! Mean diff: {jnp.mean(diff)}"
   ```

2. **Normalized Target Range Test:**
   ```python
   # During training, monitor normalized target distribution
   norm_targets = popart.normalize(batch['values'])
   
   assert jnp.min(norm_targets) > -10, "Targets clipped at lower bound"
   assert jnp.max(norm_targets) < 10, "Targets clipped at upper bound"
   
   # Should be within [-5, 5] for healthy training
   assert jnp.std(norm_targets) < 3, "Targets too spread out"
   ```

3. **Gradient Flow Test:**
   ```python
   # Verify gradients flow through value head
   loss = compute_loss(params, batch)
   grads = jax.grad(loss)(params)
   
   # Check value head has non-zero gradients
   assert jnp.any(grads['value_head']['kernel'] != 0)
   assert jnp.any(grads['value_head']['bias'] != 0)
   ```

---

## Acceptance Criteria

- [ ] Decision made on PopArt weight correction (disable or fix)
- [ ] If disabled: Add comment explaining why in popart.py
- [ ] Normalized target range monitored in logs
- [ ] No target clipping warnings during normal training
- [ ] Weight correction test passes (if kept) or is removed (if disabled)
- [ ] Training stability verified over curriculum transitions

---

## References

- PopArt paper: https://arxiv.org/abs/1602.07714
- `mcts_trader/jax_agent/popart.py` - PopArt implementation
- `mcts_trader/jax_agent/trainer.py:1597-1623` - PopArt usage in training
- `mcts_trader/jax_agent/__init__.py:51-53` - Value support configuration
