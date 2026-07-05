# TASK_132B: BC/Imprint System Comprehensive Audit

**Date:** 2026-02-01
**Auditor:** Kimi Code CLI
**Status:** FIXED (P0/P1 items)
**Scope:** Complete audit of BC/Imprint information flow with three focus areas:
1. Shape consistency through the pipeline
2. Priority/sampling logic (PER weights, confidence weighting)
3. Metric logging completeness

---

## EXECUTIVE SUMMARY

| Audit Area | Status | Critical Issues | Medium Issues | Minor Issues |
|------------|--------|-----------------|---------------|--------------|
| Shape Consistency | ⚠️ SUSPICIOUS | 0 | 1 | 2 |
| Priority/Sampling Logic | ⚠️ SUSPICIOUS | 0 | 2 | 1 |
| Metric Logging | 🐛 BUGS FOUND | 2 | 2 | 1 |

**RECOMMENDATION:** Fix metric logging bugs immediately (affects training monitoring). Address shape consistency and sampling issues before scaling to larger models.

---

## 1. SHAPE CONSISTENCY AUDIT

### 1.1 Data Flow Pipeline

```
Phase 1: Episode Storage (train_jax_fast.py:560-693)
├── prices: (T, n_assets) or (T, n_assets, 5) - stored in PriceCache
├── optimal_path.indices: (T-1, k) - selected asset indices  
└── optimal_action: scalar - encoded as asset_idx * 5 + action_type

Phase 2: Buffer Sampling (expert_buffer.py:302-409)
├── entries: structured array with episode_id, step fields
├── observations: (batch_size, 27, n_assets, n_assets) - encoded via JAX
├── optimal_actions: (batch_size,) - int32 action indices
├── confidence: (batch_size,) - float32 confidence scores
└── weights: (batch_size,) - float32 PER importance sampling weights

Phase 3: Model Inference (trainer.py:1422-1430)
├── expert_obs: (expert_batch_size, 27, n_assets, n_assets)
├── expert_policy_logits: (expert_batch_size, n_actions) where n_actions = 320
├── expert_value_logits: (expert_batch_size, support_size) where support_size = 401
└── expert_q_values: (expert_batch_size,) - scalar expected values

Phase 4: BC Loss Computation (trainer.py:1468-1492)
├── bc_ce_loss: (expert_batch_size,) - per-sample cross-entropy
├── expert_confidence: (expert_batch_size,) - per-sample confidence
├── per_weights: (expert_batch_size,) - per-sample PER weights
├── combined_weights: (expert_batch_size,) - confidence * per_weights
└── bc_loss_total: scalar - weighted mean loss
```

### 1.2 Shape Verification Results

| Connection | Expected | Actual | Status |
|------------|----------|--------|--------|
| expert_obs → model.apply | (B, 27, 64, 64) | (B, 27, 64, 64) | ✅ OK |
| model → policy_logits | (B, 320) | (B, 320) | ✅ OK |
| policy_logits → softmax | (B, 320) | (B, 320) | ✅ OK |
| expert_actions → one_hot | (B,) | (B,) | ✅ OK |
| expert_actions → cross_entropy | (B,) int | (B,) int | ✅ OK |
| bc_ce_loss → weighted_mean | (B,) float | (B,) float | ✅ OK |
| combined_weights → weight_sum | scalar | scalar | ✅ OK |

### 1.3 ⚠️ ISSUE: Potential Shape Mismatch in Large Margin Loss

**Location:** `trainer.py:35-66`

**Problem:** The `compute_large_margin_loss` function expects `q_values` with shape `(batch, num_actions)` but receives `expert_policy_logits` which ARE the policy logits, not Q-values.

```python
# Line 1448-1452 in trainer.py
margin_loss = compute_large_margin_loss(
    expert_policy_logits,  # These are POLICY LOGITS, not Q-values!
    expert_actions,
    margin=bc_config.large_margin
)
```

**Impact:** The large margin loss computes `jnp.max(q_values + margin_mask)` but if `q_values` are unnormalized logits, the margin constraint may be ineffective or produce unexpected gradients.

**Fix Recommendation:**
```python
# Option 1: Rename for clarity (function works on logits too)
# Just rename the parameter from q_values to logits

# Option 2: If true Q-values are needed, compute them properly:
# expert_q_values = self._logits_to_scalar(expert_value_logits)  # (B,)
# But this gives scalar, not per-action Q-values

# Recommendation: Rename function/parameter for clarity
```

### 1.4 ⚠️ ISSUE: Observation Window Shape for Early Steps

**Location:** `expert_buffer.py:471`

**Problem:** When `step < window_size`, `window_start = step - window_size` becomes negative, so it's clamped to 0. This means early observations have fewer than `window_size` historical steps.

```python
window_starts = jnp.maximum(jnp.array(steps) - self.window_size, 0)
# For step=10, window_size=60: window_start=0, only 10 steps of history
```

**Impact:** Observations for early episode steps have insufficient history, potentially producing lower-quality features for the first `window_size` steps of each episode.

**Fix Recommendation:**
```python
# In train_jax_fast.py:647, filter out early steps:
for step in range(n_steps):
    if step < window_size:  # Skip early steps
        continue
    # ... rest of demo storage
```

### 1.5 Minor Issue: Encoder Output Shape Docstring Mismatch

**Location:** `expert_buffer.py:421-422`

**Problem:** Docstring says output is `(batch_size, 27, n_assets, n_assets)` but encoder actually returns `(batch_size, 27, n_assets, n_assets)` - the docstring comment says `(batch, channels, window, assets)` which is WRONG.

The actual shape is `(batch, 27 channels, n_assets height, n_assets width)` - a square spatial tensor, not `(channels, window, assets)`.

**Fix:** Update docstring to clarify shape semantics.

---

## 2. PRIORITY/SAMPLING LOGIC AUDIT

### 2.1 PER (Prioritized Experience Replay) Implementation

**Location:** `expert_buffer.py:160-165, 541-556`

#### Demo Priority Calculation:
```python
priority = confidence * self.max_priority + self.demo_priority_boost
```

| Component | Value Range | Purpose |
|-----------|-------------|---------|
| confidence | 0.0 - 1.0 | Oracle demo quality |
| max_priority | dynamically updated | Scale with agent TD-errors |
| demo_priority_boost | 0.001 (default) | Epsilon for non-zero priority |

**Status:** ✅ Correct - demos scale with agent priorities, ensuring they remain competitive.

#### Agent Priority Calculation:
```python
priority = np.abs(td_error) + 1e-6
```

**Status:** ✅ Standard PER formula.

### 2.2 Importance Sampling Weights Calculation

**Location:** `expert_buffer.py:369-392`

```python
# Priority to probability
priorities = entries['priority'].astype(np.float64) ** self.per_alpha
probs = priorities / total_priority

# Importance sampling weight
weights = (N * probs) ** (-self.per_beta)
weights = weights / max_weight  # Normalize
weights = np.clip(weights, 0.1, 10.0)  # Stability clipping
```

**Status:** ✅ Correct implementation of PER importance sampling.

### 2.3 ⚠️ ISSUE: Beta Annealing Calculation

**Location:** `expert_buffer.py:331-338`

**Problem:** Beta annealing uses `self.update_count` which increments every `sample_batch()` call, but this may not correspond to training steps if samples are drawn multiple times per epoch.

```python
self.per_beta = min(
    self.per_beta_final,
    self.per_beta_init +
    (self.per_beta_final - self.per_beta_init) *
    (self.update_count / self.per_beta_steps)  # update_count = sample calls, not steps
)
```

**Impact:** If `sample_batch()` is called multiple times per epoch (e.g., for different training iterations), beta anneals faster than expected.

**Fix Recommendation:**
```python
# Pass training_step explicitly:
def sample_batch(self, batch_size: int, demo_ratio: float, training_step: int = None):
    if training_step is not None:
        progress = training_step / self.per_beta_steps
    else:
        progress = self.update_count / self.per_beta_steps
```

### 2.4 ⚠️ ISSUE: Confidence Weighting vs PER Weights Interaction

**Location:** `trainer.py:1457-1487`

**Problem:** When both `weight_by_confidence=True` and PER weights are active, the combined weights are:

```python
combined_weights = expert_confidence * per_weights
```

This means:
- High-confidence demo with low PER weight → low combined weight
- Low-confidence demo with high PER weight → high combined weight

**Question:** Is this the intended behavior? Should confidence override PER or combine multiplicatively?

**Current Behavior:** Multiplicative combination
- Pros: Both factors influence sampling
- Cons: High-confidence samples can be downweighted by PER

**Recommendation:** Consider additive combination or hierarchical weighting:
```python
# Option 1: Hierarchical (confidence as primary, PER as secondary)
combined_weights = expert_confidence * (per_weights ** 0.5)  # Dampen PER effect

# Option 2: Normalize separately then combine
norm_confidence = expert_confidence / expert_confidence.sum()
norm_per_weights = per_weights / per_weights.sum()
combined_weights = 0.7 * norm_confidence + 0.3 * norm_per_weights
```

### 2.5 Minor Issue: Demo Ratio Handling with Small Buffers

**Location:** `expert_buffer.py:342-345`

**Problem:** When buffer has few samples, the effective demo ratio may be skewed:

```python
n_demo = min(int(effective_batch_size * demo_ratio), self.demo_count)
n_agent = effective_batch_size - n_demo
n_agent = min(n_agent, self.agent_count)
n_demo = effective_batch_size - n_agent  # Adjust if not enough agent samples
```

If `agent_count` is very low, `n_demo` increases to compensate, potentially exceeding `demo_ratio`.

**Impact:** Early training may sample more demos than configured ratio.

**Status:** This is probably intentional for stability with small buffers.

---

## 3. METRIC LOGGING COMPLETENESS AUDIT

### 3.1 Metrics Computed in Trainer vs Logged to CSV

| Metric | Computed | Logged | Status |
|--------|----------|--------|--------|
| bc_loss | ✅ trainer.py:1498 | ✅ train_jax_fast.py:1295 | OK |
| bc_weight | ✅ trainer.py:1499 | ✅ train_jax_fast.py:1296 | OK |
| expert_action_agreement | ✅ trainer.py:1492 | ✅ train_jax_fast.py:1297 | 🐛 BUGGY |
| expert_acceptance_rate | ✅ trainer.py:1501 | ✅ train_jax_fast.py:1298 | 🐛 BUGGY |
| expert_batch_size | ✅ trainer.py:1500 | ❌ train_jax_fast.py | MISSING |
| bc_ce_loss | ✅ trainer.py:1482,1487 | ❌ train_jax_fast.py | MISSING |
| bc_margin_loss | ✅ trainer.py:1462,1467 | ❌ train_jax_fast.py | MISSING |

### 3.2 🐛 CRITICAL BUG: Falsy Check in CSV Row Construction

**Location:** `train_jax_fast.py:1295-1298`

**Problem:** The condition `if metrics.get('expert_action_agreement')` evaluates to `False` when the value is `0.0` or `jnp.array(0.0)`!

```python
# BUGGY CODE:
'expert_action_agreement': get_last(metrics.get('expert_action_agreement', [])) 
    if metrics.get('expert_action_agreement') else None,
```

**Impact:**
- When no expert batch is available, metric shows `None` instead of `0.0`
- Cannot distinguish between "BC not active" vs "0% agreement"
- Breaks DS24 logging consistency
- Time series plots will have gaps

**Fix:**
```python
# FIXED CODE:
'expert_action_agreement': get_last(metrics.get('expert_action_agreement', [0])) 
    if 'expert_action_agreement' in metrics else None,
```

Same issue exists for `expert_acceptance_rate` at line 1298.

### 3.3 🐛 BUG: Type Inconsistency in Else Branch

**Location:** `trainer.py:1503-1507`

**Problem:** In the else branch (no expert batch), metrics are set as Python floats, but in the main branch they're JAX arrays:

```python
# Main branch (lines 1498-1501):
metrics['bc_loss'] = bc_loss_total  # JAX array
metrics['expert_action_agreement'] = jnp.mean(expert_agreement)  # JAX array

# Else branch (lines 1503-1507):
metrics['bc_loss'] = 0.0  # Python float!
metrics['expert_action_agreement'] = 0.0  # Python float!
```

**Impact:** Downstream code expecting JAX arrays may fail or behave inconsistently.

**Fix:**
```python
# FIXED CODE:
else:
    metrics['bc_loss'] = jnp.array(0.0)
    metrics['bc_weight'] = jnp.array(0.0)
    metrics['expert_batch_size'] = 0
    metrics['expert_acceptance_rate'] = jnp.array(0.0)
    metrics['expert_action_agreement'] = jnp.array(0.0)
```

### 3.4 Medium Issue: Missing Metrics in CSV Header

**Location:** `train_jax_fast.py:697-723`

**Problem:** The following computed metrics are NOT in the CSV header:
- `expert_batch_size` - useful for debugging buffer issues
- `bc_ce_loss` / `bc_margin_loss` - loss breakdown not logged
- `optimal_alpha` - oracle performance metric

**Impact:** Cannot analyze BC loss components separately or track oracle quality.

**Fix:** Add to `base_columns`:
```python
base_columns = [
    # ... existing columns ...
    'expert_batch_size', 'bc_ce_loss', 'bc_margin_loss', 'optimal_alpha',
]
```

### 3.5 Medium Issue: Metric Name Collision

**Location:** `trainer.py:1492, 1501`

**Problem:** `expert_action_agreement` and `expert_acceptance_rate` are set to the SAME VALUE:

```python
metrics['expert_action_agreement'] = jnp.mean(expert_agreement)  # Line 1492
metrics['expert_acceptance_rate'] = jnp.mean(expert_agreement)   # Line 1501 - SAME!
```

The comment says "Alias for backwards compat" but this wastes a CSV column.

**Recommendation:** Either:
1. Remove one from metrics (keep just `expert_action_agreement`)
2. Or make them different: `acceptance_rate` = demos accepted / total demos attempted

---

## 4. RECOMMENDED FIXES (PRIORITIZED)

### P0: Critical (Fix Immediately)

1. **Fix falsy check in CSV row construction** (`train_jax_fast.py:1297-1298`)
   - Change `if metrics.get('key')` to `if 'key' in metrics`

2. **Fix type inconsistency in else branch** (`trainer.py:1503-1507`)
   - Wrap values in `jnp.array()`

### P1: High (Fix Before Next Training Run)

3. **Add missing metrics to CSV header** (`train_jax_fast.py:720`)
   - Add `expert_batch_size`, `bc_ce_loss`, `bc_margin_loss`

4. **Clarify large margin loss parameter naming** (`trainer.py:1448`)
   - Rename `q_values` to `logits` in function if that's what's being used

### P2: Medium (Fix When Convenient)

5. **Fix beta annealing to use training steps** (`expert_buffer.py:331-338`)
   - Pass `training_step` explicitly to `sample_batch()`

6. **Filter early episode steps from demos** (`train_jax_fast.py:647`)
   - Skip steps < `window_size` to ensure full observation windows

7. **Review confidence vs PER weight interaction** (`trainer.py:1457-1487`)
   - Consider hierarchical weighting instead of multiplicative

### P3: Low (Nice to Have)

8. **Remove duplicate metric** (`trainer.py:1501`)
   - Either `expert_action_agreement` or `expert_acceptance_rate`, not both

9. **Fix docstring** (`expert_buffer.py:421-422`)
   - Correct shape semantics description

---

## 5. PATCH FILE

```diff
# Fix 1: train_jax_fast.py - Falsy check bug
@@ -1294,10 +1294,10 @@
                 'amax_alpha_capture': get_last(metrics.get('alpha_capture', [0])),
                 'amax_optimal_gain': get_last(metrics.get('optimal_gain', [0])),
                 # TASK_125: Causal Expert Imprinting metrics
-                'bc_loss': get_last(metrics.get('bc_loss', [])) if metrics.get('bc_loss') else None,
-                'bc_weight': get_last(metrics.get('bc_weight', [])) if metrics.get('bc_weight') else None,
-                'expert_action_agreement': get_last(metrics.get('expert_action_agreement', [])) if metrics.get('expert_action_agreement') else None,
-                'expert_acceptance_rate': get_last(metrics.get('expert_acceptance_rate', [])) if metrics.get('expert_acceptance_rate') else None,
+                'bc_loss': get_last(metrics.get('bc_loss', [0])) if 'bc_loss' in metrics else None,
+                'bc_weight': get_last(metrics.get('bc_weight', [0])) if 'bc_weight' in metrics else None,
+                'expert_action_agreement': get_last(metrics.get('expert_action_agreement', [0])) if 'expert_action_agreement' in metrics else None,
+                'expert_acceptance_rate': get_last(metrics.get('expert_acceptance_rate', [0])) if 'expert_acceptance_rate' in metrics else None,
                 'expert_demo_count': expert_buffer.demo_count if expert_buffer else 0,

# Fix 2: trainer.py - Type consistency
@@ -1500,11 +1500,11 @@
             metrics['expert_acceptance_rate'] = jnp.mean(expert_agreement)  # Alias for backwards compat
         else:
-            metrics['bc_loss'] = 0.0
-            metrics['bc_weight'] = 0.0
+            metrics['bc_loss'] = jnp.array(0.0)
+            metrics['bc_weight'] = jnp.array(0.0)
             metrics['expert_batch_size'] = 0
-            metrics['expert_acceptance_rate'] = 0.0
-            metrics['expert_action_agreement'] = 0.0
+            metrics['expert_acceptance_rate'] = jnp.array(0.0)
+            metrics['expert_action_agreement'] = jnp.array(0.0)

# Fix 3: train_jax_fast.py - Add missing metrics to header
@@ -718,6 +718,7 @@
         # TASK_125: Causal Expert Imprinting metrics
         'bc_loss', 'bc_weight', 'expert_action_agreement', 'expert_acceptance_rate', 'expert_demo_count',
+        'expert_batch_size', 'bc_ce_loss', 'bc_margin_loss',
         # Generalization eval metrics
         'eval_level', 'eval_reward', 'eval_alpha', 'eval_pct_hold',
```

---

## 6. VERIFICATION CHECKLIST

After applying fixes, verify:

- [ ] CSV contains `0.0` (not `None`) for BC metrics when no expert batch
- [ ] `bc_ce_loss` appears in CSV when using standard BC loss
- [ ] `bc_margin_loss` appears in CSV when using large margin loss
- [ ] `expert_batch_size` is logged correctly
- [ ] All BC metrics have consistent types (JAX arrays in trainer)
- [ ] Training runs without type errors
- [ ] CSV columns align with data rows

---

## APPENDIX: File References

| File | Key Lines | Purpose |
|------|-----------|---------|
| `train_jax_fast.py` | 560-693 | Episode callback, demo storage |
| `train_jax_fast.py` | 697-723 | CSV header definition |
| `train_jax_fast.py` | 1295-1299 | CSV row construction |
| `expert_buffer.py` | 122-168 | Demo storage with priority |
| `expert_buffer.py` | 302-409 | Batch sampling with PER |
| `expert_buffer.py` | 411-485 | Observation encoding |
| `trainer.py` | 256-280 | ExpertBCConfig |
| `trainer.py` | 1408-1508 | BC loss computation |
| `jax_encoder.py` | 450-467 | encode_observation_simple |

---

## FIXES APPLIED

**Date:** 2026-02-01

### P0 Fixes (Critical)

1. **Falsy check bug** - `train_jax_fast.py:1301-1309`
   - Changed `if metrics.get('key')` to `if 'key' in metrics`
   - Now correctly handles 0.0 values

2. **Type inconsistency** - `trainer.py:1502-1507`
   - Wrapped Python floats in `jnp.array()`
   - Else branch now type-consistent with main branch

### P1 Fixes (High)

3. **Missing metrics** - `train_jax_fast.py:727-728` (header) + `:1307-1309` (row)
   - Added `expert_batch_size`, `bc_ce_loss`, `bc_margin_loss`

### Previously Fixed (TASK_132e)

4. **Window edge leakage** (Section 1.4)
   - Skip steps < window_size in demo storage
