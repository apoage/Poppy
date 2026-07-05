# TASK_132d: BC/Imprint Metric Propagation Audit

**Auditor**: Claude Opus 4.5
**Date**: 2026-02-01
**Status**: FIXED

## 1. Missing CSV Column Value (expert_action_agreement)

**Location**: `train_jax_fast.py:720` (header) vs `train_jax_fast.py:1295-1298` (row)

**Finding**:
- CSV header includes `'expert_action_agreement'` at line 720
- CSV row dict does NOT include `'expert_action_agreement'` key
- Row dict includes `'expert_acceptance_rate'` instead (different key!)
- DictWriter uses `extrasaction='ignore'` so mismatch is silent

**Trace**:
```
trainer.py:1491-1492:
    expert_agreement = (policy_actions == expert_actions).astype(jnp.float32)
    metrics['expert_action_agreement'] = jnp.mean(expert_agreement)

trainer.py:1501:
    metrics['expert_acceptance_rate'] = jnp.mean(expert_agreement)  # Alias

train_jax_fast.py:720 (header):
    'bc_loss', 'bc_weight', 'expert_action_agreement', 'expert_demo_count',

train_jax_fast.py:1295-1298 (row):
    'bc_loss': get_last(metrics.get('bc_loss', [])) ...
    'bc_weight': get_last(metrics.get('bc_weight', [])) ...
    'expert_acceptance_rate': expert_buffer.demo_count / ...  # WRONG KEY + WRONG VALUE
    'expert_demo_count': expert_buffer.demo_count ...
```

**Impact**:
- `expert_action_agreement` column in CSV is always empty/None
- Cannot monitor BC convergence (policy matching oracle actions)
- The value logged as `expert_acceptance_rate` is actually a ratio calculation, not the trainer metric

**Recommendation**:
1. Add `'expert_acceptance_rate'` to CSV header (line 720)
2. Add to csv_row:
```python
'expert_action_agreement': get_last(metrics.get('expert_action_agreement', [])) if metrics.get('expert_action_agreement') else None,
'expert_acceptance_rate': get_last(metrics.get('expert_acceptance_rate', [])) if metrics.get('expert_acceptance_rate') else None,
```

**Fix Applied**:
- `train_jax_fast.py:720` - Added both columns to header
- `train_jax_fast.py:1297-1298` - Extract both metrics from trainer metrics dict

---

## 2. Observation Shape Comment Mismatch (Minor - Documentation Only)

**Location**: `trainer.py:1416`

**Finding**: Comment says `(expert_batch_size, 27, 60, 64)` but actual shape is `(expert_batch_size, 27, 64, 64)`

**Trace**:
```
expert_buffer.py:432:
    observations = np.zeros((batch_size, 27, n_assets, n_assets), dtype=np.float32)
    # n_assets = 64, so shape is (batch, 27, 64, 64)

jax_encoder.py:90:
    # Returns: Observation tensor of shape (27, n_assets, n_assets)
```

**Impact**: Documentation confusion only, code works correctly

**Recommendation**: Fix comment to `(expert_batch_size, 27, n_assets, n_assets)`

---

## 3. Audit Prompts for Further Investigation

### Prompt A: Oracle Action Encoding Audit
```
Audit compute_optimal_path() -> action conversion in train_jax_fast.py:648-677
Verify: optimal_path.indices correctly maps to action = asset * 5 + type
Check: Edge cases when selected_indices is empty or returns are exactly 0
```

### Prompt B: Window Start Calculation Audit
```
Audit expert_buffer.py:468-471 window_start calculation
Verify: window_start = step - window_size gives correct lookback
Check: Boundary case when step < window_size (should clamp to 0)
Compare: Self-play encoder call vs expert_buffer encoder call signatures
```

### Prompt C: Priority Sampling Audit
```
Audit expert_buffer.py sample_batch() PER implementation
Verify: Demo priority = confidence * max_priority (not just confidence)
Check: Stratified sampling implementation in _sample_from_range()
Verify: IS weights computed correctly for unbiased gradients
```
