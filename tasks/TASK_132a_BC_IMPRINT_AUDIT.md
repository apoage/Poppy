# TASK_132a: BC/Imprint System Comprehensive Audit

**Status**: Audit Complete
**Date**: 2026-02-01
**Scope**: Information flow, shape consistency, priority logic, metric logging

---

## Executive Summary

Three audits performed on the BC (Behavior Cloning) / Imprint system:
1. **Shape Consistency**: 1 documentation bug found
2. **Priority/Sampling Logic**: All correct
3. **Metric Logging**: 1 BUG found - `expert_action_agreement` not propagated to CSV

---

## Audit 1: Shape Consistency

### Trace: Observation → Encoder → Buffer → Trainer → Model

| Component | Location | Shape | Status |
|-----------|----------|-------|--------|
| Price Cache | `price_cache.py` | `(T, n_assets, 5)` OHLCV | OK |
| Encoder Input | `jax_encoder.py:104-106` | `(T, n_assets)` close prices | OK |
| Encoder Output | `jax_encoder.py:90,150` | `(27, n_assets, n_assets)` = **(27, 64, 64)** | OK |
| Buffer Storage | `expert_buffer.py:432` | `(batch, 27, n_assets, n_assets)` = **(batch, 27, 64, 64)** | OK |
| Trainer Comment | `trainer.py:1416` | **Says (batch, 27, 60, 64)** | **WRONG** |
| Model Input | `muzero_model.py:102,189` | `(batch, 27, 64, 64)` | OK |

### Issue #1: Documentation Bug

**Location**: `mcts_trader/jax_agent/trainer.py:1416`

```python
# Current (WRONG):
expert_obs = expert_batch['observations']  # (expert_batch_size, 27, 60, 64)

# Should be:
expert_obs = expert_batch['observations']  # (expert_batch_size, 27, n_assets, n_assets)
```

**Impact**: Misleading documentation only - code works correctly
**Severity**: Low
**Fix**: Update comment

---

## Audit 2: Priority/Sampling Logic (PER)

### Demo Priority Formula

**Location**: `expert_buffer.py:160-165`

```python
# BUG-007 FIX: Scale demo priority to be competitive with agent TD-errors
priority = confidence * self.max_priority + self.demo_priority_boost
```

| Check | Status |
|-------|--------|
| Demo priority scales with confidence | OK |
| Demo priority competes with agent TD-errors | OK |
| `demo_priority_boost` (epsilon_D = 0.001) applied | OK |

### Agent Priority Formula

**Location**: `expert_buffer.py:203-204`

```python
priority = np.abs(td_error) + 1e-6
```

| Check | Status |
|-------|--------|
| Priority from absolute TD-error | OK |
| Epsilon floor (1e-6) prevents zero priority | OK |

### Importance Sampling Weights

**Location**: `expert_buffer.py:369-392`

```python
# w_i = (N * P(i))^(-beta), normalized by max
weights = (N * probs) ** (-self.per_beta)
weights = weights / max(max_weight, 1e-10)
weights = np.clip(weights, 0.1, 10.0)
```

| Check | Status |
|-------|--------|
| Beta annealing (0.4 → 1.0 over 100k steps) | OK |
| Weights normalized by max for stability | OK |
| Weights clipped to [0.1, 10.0] | OK |

### Stratified Sampling (BUG-009 Fix)

**Location**: `expert_buffer.py:500-539`

| Check | Status |
|-------|--------|
| Divides total priority into n equal segments | OK |
| Samples one from each segment (stratified) | OK |
| Binary search via `np.searchsorted` | OK |

### BC Loss Weight Application

**Location**: `trainer.py:1457-1487`

```python
# Combined weights: confidence * PER weights
combined_weights = expert_confidence * per_weights
weight_sum = jnp.maximum(jnp.sum(combined_weights), 1e-8)
bc_loss_total = bc_weight * jnp.sum(bc_ce_loss * combined_weights) / weight_sum
```

| Check | Status |
|-------|--------|
| Confidence weighting applied | OK |
| PER weights for unbiased gradients | OK |
| Masked mean (not simple mean) | OK |
| Division guarded by epsilon | OK |

**Conclusion**: Priority/sampling logic is correct. No bugs found.

---

## Audit 3: Metric Logging Completeness

### Metrics Computed in `_loss_fn` (trainer.py)

| Metric | Set in _loss_fn | In history dict | In CSV | Status |
|--------|----------------|-----------------|--------|--------|
| `bc_loss` | L1498 | L1618 | L720 | OK |
| `bc_weight` | L1499 | L1619 | L720 | OK |
| `expert_acceptance_rate` | L1501 | L1620 | L720 | OK |
| `expert_action_agreement` | L1492 | **MISSING** | L720 | **BUG** |
| `expert_batch_size` | L1500 | missing | - | OK (not needed in CSV) |
| `bc_margin_loss` | L1462/1467 | missing | - | OK (internal detail) |
| `bc_ce_loss` | L1482/1487 | missing | - | OK (internal detail) |
| `optimal_alpha` | L1271 | missing | - | OK (not in CSV) |
| `optimal_trades` | L1272 | missing | - | OK (not in CSV) |
| `regret` | L1288 | missing | - | OK (not in CSV) |
| `atc_loss` | L1368 | missing | - | OK (not in CSV) |
| `cycle_balance_loss` | L1387 | missing | - | OK (not in CSV) |
| `cycle_atc_loss` | L1400 | missing | - | OK (not in CSV) |

### Issue #2: BUG - `expert_action_agreement` Not Logged

**Symptom**: `expert_action_agreement` column in CSV is always `None`

**Root Cause Chain**:
1. `trainer.py:1492` sets `metrics['expert_action_agreement']` in `_loss_fn`
2. `trainer.py:1592-1621` defines `history` dict - **does NOT include `'expert_action_agreement'`**
3. `trainer.py:1694-1696` only copies metrics to history if key exists in history
4. `trainer.train()` returns `history` as `metrics`
5. `train_jax_fast.py:1297` tries `metrics.get('expert_action_agreement', [])` → returns `[]`
6. Empty list is falsy → returns `None` to CSV

**Location**: `mcts_trader/jax_agent/trainer.py:1620`

**Fix**:
```python
# trainer.py line 1620, add this key to history dict:
'expert_acceptance_rate': [],
'expert_action_agreement': [],  # <-- ADD THIS LINE
```

**Impact**: Key BC learning metric not being logged to CSV
**Severity**: Medium (diagnostic value lost, no training impact)

---

## Audit 4: Information Flow Summary (from initial audit)

### Oracle Action Flow
| Location | Finding | Status |
|----------|---------|--------|
| `train_jax_fast.py:634-636` | `compute_optimal_path()` returns `OptimalPath.indices` | OK |
| `train_jax_fast.py:671-673` | Action encoding: `asset * 5 + type` (0=Sell, 2=Buy) | OK |
| `expert_buffer.py:122-168` | `add_demo()` stores `optimal_action` | OK |

### Observation Encoding Flow
| Location | Finding | Status |
|----------|---------|--------|
| `expert_buffer.py:468-471` | `window_start = step - window_size` | OK |
| `expert_buffer.py:476-480` | vmap over window_starts per episode | OK |

### BC Loss Flow
| Location | Finding | Status |
|----------|---------|--------|
| `trainer.py:1417` | Uses `expert_batch['optimal_actions']` (correct key) | OK |
| `trainer.py:1470-1471` | Cross-entropy with integer labels | OK |
| `trainer.py:1491-1492` | Agreement = `argmax(policy) == optimal_action` | OK |

---

## Fixes Required

### Fix #1: Comment Correction (Low Priority)

**File**: `mcts_trader/jax_agent/trainer.py`
**Line**: 1416

```diff
-            expert_obs = expert_batch['observations']  # (expert_batch_size, 27, 60, 64)
+            expert_obs = expert_batch['observations']  # (expert_batch_size, 27, n_assets, n_assets)
```

### Fix #2: Add Missing History Key (Medium Priority)

**File**: `mcts_trader/jax_agent/trainer.py`
**Line**: 1620

```diff
             # TASK_125: Causal Expert Imprinting metrics
             'bc_loss': [],
             'bc_weight': [],
             'expert_acceptance_rate': [],
+            'expert_action_agreement': [],
         }
```

---

## Verification Commands

After applying fixes, verify with:

```bash
# Run short training with imprinting enabled
python train_jax_fast.py --synthetic-level 0 --epochs 10 --imprint --bc-loss

# Check CSV for non-null expert_action_agreement
head -20 logs/training_*.csv | grep expert_action_agreement
```

---

## Summary

| Audit | Issues Found | Severity | Fixed |
|-------|-------------|----------|-------|
| Shape Consistency | 1 (comment) | Low | Pending |
| Priority/Sampling | 0 | - | N/A |
| Metric Logging | 1 (missing key) | Medium | Pending |
| Information Flow | 0 | - | N/A |

**Total Issues**: 2 (1 low, 1 medium)
