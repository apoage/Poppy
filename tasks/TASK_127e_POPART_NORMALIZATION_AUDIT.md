# TASK_127e: PopArt Normalization Deep Audit

**Status**: AUDITED (Working Correctly, Minor Issues)
**Priority**: Low
**Created**: 2026-01-31

## Scope

Audited PopArt implementation:
- `mcts_trader/jax_agent/popart.py` - Core implementation
- `mcts_trader/jax_agent/trainer.py` - Training usage
- `mcts_trader/training/self_play.py` - Inference usage

## PopArt Overview

```
PopArt Flow:
┌────────────────────────────────────────────────────────────────────┐
│ BEFORE TRAINING STEP (CPU, outside JIT)                            │
│  1. Update running statistics: μ_new, σ_new from batch targets     │
│  2. Compute weight correction: scale = σ_old/σ_new                 │
│  3. Apply correction to value head: W_new = W_old * scale          │
│  4. Normalize targets: (target - μ) / σ                            │
└────────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────────┐
│ TRAINING STEP (GPU, inside JIT)                                    │
│  1. Model predicts normalized value: v_norm = f(obs; params)       │
│  2. Loss = MSE(v_norm, target_norm)                                │
│  3. Gradients flow normally through value head                     │
└────────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────────┐
│ INFERENCE (self_play)                                              │
│  1. Model predicts normalized value: v_norm                        │
│  2. Denormalize for action selection: v_raw = v_norm * σ + μ       │
└────────────────────────────────────────────────────────────────────┘
```

## Audit Results

### 1. normalize() vs unnormalize() Usage ✅ CORRECT

**Normalize (training)**:
```python
# trainer.py:1618
norm_values = self.popart.normalize(raw_values)
batch = {**batch, 'values': norm_values.reshape(...)}
```

**Denormalize (inference)**:
```python
# self_play.py:126
value_denorm = denormalize_value(value_scalar, popart_state)
```

**Assessment**:
- ✅ Targets normalized BEFORE loss computation
- ✅ Predictions denormalized AFTER inference for action selection
- ✅ Consistent usage across codebase

### 2. Weight Correction: Inside or Outside JIT? ✅ CORRECT (Outside)

**Location**: `trainer.py:1609-1616`

```python
if iteration % 10 == 0:
    _, corrected_params = self.popart.normalize_and_correct(
        valid_values, self.state.params
    )
    self.state = self.state.replace(params=corrected_params)
```

**Assessment**:
- ✅ Weight correction happens OUTSIDE JIT (Python-level)
- ✅ Corrected params passed INTO JIT-compiled train_step_fn
- ✅ This is CORRECT - correction must be outside JIT because it uses Python dict operations

**Why outside JIT is correct**:
1. `copy.deepcopy` in `apply_weight_correction_flax` is not JAX-traceable
2. Weight correction is a pre-processing step, not part of the loss
3. Corrected params become the new "starting point" for gradients

### 3. Does Gradient Flow Through Normalization? ✅ YES (Correctly)

**The gradient flow path**:
```
Loss Computation (inside JIT):
  model(obs, params) → value_logits → softmax → expected_value
                                                      ↓
                                              MSE(expected_value, target_norm)
                                                      ↓
                                              ∂Loss/∂params (gradient)
```

**Key insight**:
- `target_norm` is a CONSTANT (no gradient needed through it)
- Model outputs are DIFFERENTIABLE
- Gradients flow: Loss → value_head weights → rest of model

**The normalization does NOT block gradients because**:
1. Statistics update is outside the computation graph
2. Targets are treated as constants in the loss
3. Only model predictions need gradients

### 4. Where Exactly Are Values Normalized/Unnormalized?

| Operation | Location | Inside JIT? |
|-----------|----------|-------------|
| Update statistics | `trainer.py:1613` | ❌ No |
| Compute weight correction | `popart.py:130-157` | ❌ No |
| Apply weight correction | `trainer.py:1616` | ❌ No |
| Normalize targets | `trainer.py:1618` | ❌ No |
| Replace batch values | `trainer.py:1623` | ❌ No |
| Compute loss | `trainer.py:1091-1096` | ✅ Yes |
| Denormalize for inference | `self_play.py:126` | ❌ No |

### 5. Is Loss on Normalized or Raw Values?

**Answer**: Loss is computed on **NORMALIZED** values.

```python
# trainer.py:1045 - uses batch['values'] which are already normalized
target_values = batch['values']  # NORMALIZED by PopArt

# trainer.py:1091-1096 - loss computation
value_loss, value_mse, value_cat = self._hybrid_value_loss(
    initial_output['value'],  # Raw model output (predicts normalized scale)
    target_values[:, 0],      # NORMALIZED targets
    ...
)
```

### 6. When Are PopArt Statistics Updated?

**Location**: `trainer.py:1609`

```python
if iteration % 10 == 0:  # Every 10 iterations
    _, corrected_params = self.popart.normalize_and_correct(...)
```

**Assessment**:
- ⚠️ Statistics only updated every 10 iterations (performance optimization)
- ✅ Normalization still happens every iteration using current stats
- ⚠️ Weight correction also only every 10 iterations

**Potential Issue**: Between updates (iterations 1-9), the model trains with slightly stale statistics. This is a trade-off for performance but could cause small inconsistencies.

### 7. Value Prediction Path Trace

```
TRAINING:
  obs → TRM.encode() → latent → TRM.reasoning() → value_logits
                                                        ↓
                                                  softmax → expected_value_norm
                                                        ↓
                                        MSE(expected_value_norm, target_norm)
                                                        ↓
                                              Gradient flows to all params ✅

INFERENCE:
  obs → TRM.encode() → latent → TRM.reasoning() → value_logits
                                                        ↓
                                                  softmax → expected_value_norm
                                                        ↓
                                          denormalize(expected_value_norm)
                                                        ↓
                                                  value_raw for MCTS
```

## Issues Found

### Issue 1: Statistics Update Frequency ⚠️

**Severity**: Low

**Problem**: Statistics only updated every 10 iterations:
```python
if iteration % 10 == 0:
```

**Impact**:
- Between updates, targets are normalized with slightly stale μ/σ
- Could cause small training inconsistencies
- Trade-off for GPU sync performance

**Recommendation**: Monitor value loss variance across the 10-iteration window.

### Issue 2: Weight Correction Not Applied Every Iteration ⚠️

**Severity**: Low

**Problem**: Weight correction also only every 10 iterations.

**Impact**: For 9 iterations, value head weights are not corrected for the latest statistics. The model trains on normalized targets but outputs predictions from un-corrected weights.

**However**: The EMA beta is very small (0.0001), so statistics change slowly. Over 10 iterations, the drift is minimal.

### Issue 3: Value Support Range Assumption

**Severity**: Low

**Problem**: Code comments mention:
```python
# With PopArt normalization (default):
# - Value targets have mean≈0, std≈1
# - Support range [-10, 10] is sufficient
```

If PopArt fails to normalize properly (e.g., extreme reward spike), values could exceed support range.

### Issue 4: PopArt State Not Passed to All Inference Paths

**Severity**: Low

**Problem**: PopArt state must be explicitly passed to inference functions:
```python
# self_play.py:56
popart_state: Optional[PopArtState] = None,
```

If not passed, values won't be denormalized correctly.

**Status**: Currently handled correctly in main paths, but easy to forget in new code.

## Gradient Flow Verification

**Does PopArt break gradient flow?** ❌ NO

**Proof**:
1. Weight correction is a one-time transform BEFORE training step
2. Normalized targets are constants (like any label)
3. Loss function is standard MSE/cross-entropy with differentiable inputs
4. No `stop_gradient` or `jax.lax.stop_gradient` in PopArt code

**The gradient computation**:
```python
# Simplified
loss = MSE(model(obs, corrected_params), normalized_target)
grad = jax.grad(loss)(corrected_params)  # Works normally!
```

The `corrected_params` are just regular parameters from JAX's perspective - the correction was applied before tracing.

## Summary

| Aspect | Status | Notes |
|--------|--------|-------|
| Normalize/denormalize usage | ✅ Correct | Consistent across codebase |
| Weight correction location | ✅ Correct | Outside JIT as required |
| Gradient flow | ✅ Preserved | No blockers |
| Loss computation | ✅ Correct | On normalized values |
| Statistics update freq | ⚠️ Every 10 iter | Performance trade-off |
| State persistence | ✅ Saved/restored | In checkpoints |

## Recommendations

1. **Monitor value loss variance** across the 10-iteration update window
2. **Add assertion** that PopArt state is not None when use_popart=True in inference
3. **Consider adaptive update frequency** - update more often when statistics change rapidly

## Related Tasks

- TASK_021: PopArt Implementation (original)
- TASK_127b: MuZero Dynamics Audit
- TASK_126a: Reward Spike Investigation (related to value scale issues)
