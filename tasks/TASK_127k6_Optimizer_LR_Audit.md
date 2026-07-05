# TASK_127k6: Optimizer and Learning Rate Audit

## Executive Summary

This audit examines the optimizer configuration, learning rate schedules, and gradient clipping. **Overall configuration is reasonable**, with AdamW optimizer and configurable LR schedules. However, several configuration inconsistencies and potential issues were identified.

**Risk Level: MEDIUM** - Default LR may be high, gradient clipping is very permissive, and per-head clipping is disabled by default.

---

## Optimizer Configuration

### ✅ CORRECT: AdamW Optimizer

**Location:** `mcts_trader/jax_agent/trainer.py:553-556`

```python
self.optimizer = optax.chain(
    optax.clip_by_global_norm(10.0),
    optax.adamw(learning_rate=lr_schedule, weight_decay=weight_decay)
)
```

**Analysis:**
- ✅ **AdamW**: Correct choice for modern deep learning (decoupled weight decay)
- ✅ **Optax chain**: Clean composable optimizer design
- ⚠️ **Adam betas**: Using defaults (β1=0.9, β2=0.999) - reasonable but not tuned

**AdamW vs Adam:**
```python
# Adam: weight_decay is added to gradient (L2 regularization)
# AdamW: weight_decay is subtracted from parameters directly
# AdamW typically performs better with large models
```

---

## Learning Rate Configuration

### Default Learning Rate

| Source | Value | Notes |
|--------|-------|-------|
| `trainer.py:367` | `2e-4` | Trainer default |
| `training/config.py:147` | `1e-4` | Config default |
| `schedules.py:92` | `1e-4` | Schedule default |

**🟡 ISSUE: Inconsistent defaults between trainer and config**

```python
# trainer.py
learning_rate: float = 2e-4,  # Default

# training/config.py  
lr: float = 1e-4  # Different default!
```

**Impact:** Depending on how trainer is instantiated, LR could be 2x different than expected.

**Recommendation:** Sync defaults:
```python
# Both should use same value
learning_rate: float = 1e-4  # Conservative default
```

---

### Learning Rate Schedules

**Location:** `mcts_trader/jax_agent/schedules.py:85-177`

```python
@dataclass
class LRSchedule:
    initial_lr: float = 1e-4
    final_lr: float = 1e-6
    warmup_epochs: int = 10
    total_epochs: int = 1000
    schedule_type: Literal['constant', 'cosine', 'linear'] = 'constant'
```

**Schedule Types:**

| Type | Formula | Use Case |
|------|---------|----------|
| `constant` | `LR = initial_lr` | Stable training, default |
| `cosine` | `final + 0.5*(init-final)*(1+cos(π*progress))` | Smooth decay, good convergence |
| `linear` | `initial + progress*(final-initial)` | Simple decay |

**Warmup Phase:**
```python
if epoch < warmup_epochs:
    return initial_lr * (epoch + 1) / warmup_epochs
# Linear warmup from 0 → initial_lr
```

**Analysis:**
- ✅ All three schedule types properly implemented
- ✅ Warmup phase prevents early instability
- ✅ Proper optax integration
- ⚠️ **Default is 'constant'** - may miss convergence benefits of decay

---

### LR Schedule Integration

**Location:** `trainer.py:538-556`

```python
# TASK_024: Create optimizer with LR schedule
if lr_schedule_type == 'constant':
    lr_schedule = learning_rate
else:
    from .schedules import LRSchedule
    lr_sched = LRSchedule(
        initial_lr=learning_rate,
        final_lr=lr_min,
        warmup_epochs=lr_warmup_epochs,
        total_epochs=total_epochs,
        schedule_type=lr_schedule_type,
    )
    lr_schedule = lr_sched.to_optax_schedule(steps_per_epoch)
```

**Analysis:**
- ✅ Schedule converted to optax-compatible function
- ✅ Step-based scheduling (converts epochs → steps)
- ⚠️ `total_epochs` required even for constant schedule (unused but required param)

---

## Gradient Clipping

### Global Gradient Clipping

**Location:** `trainer.py:554`

```python
self.optimizer = optax.chain(
    optax.clip_by_global_norm(10.0),  # Very permissive!
    optax.adamw(learning_rate=lr_schedule, weight_decay=weight_decay)
)
```

**Analysis:**
- **Clip value: 10.0** - Very high, allows large gradients
- Typical values: 0.5-5.0 for stability
- With AdamW's adaptive learning rates, large gradients may still be problematic

**Comparison:**
```python
# Conservative (stable training)
optax.clip_by_global_norm(1.0)

# Moderate (balanced)
optax.clip_by_global_norm(5.0)

# Current (very permissive)
optax.clip_by_global_norm(10.0)  # Allows extreme gradients
```

**🟡 RECOMMENDATION:** Consider lowering to 5.0 or 1.0 if training is unstable.

---

### Per-Head Gradient Clipping (TASK_025)

**Location:** `trainer.py:399-404, 1492-1500`

```python
# Config
per_head_clipping: bool = False,  # Disabled by default!
clip_grad_value: float = 1.0,
clip_grad_policy: float = 1.0,
clip_grad_dynamics: float = 1.0,
clip_grad_encoder: float = 1.0,

# Usage in training
if self.per_head_clipping:
    grads = _clip_gradients_per_head(
        grads,
        max_norm_value=self.clip_grad_value,      # 1.0
        max_norm_policy=self.clip_grad_policy,    # 1.0
        max_norm_dynamics=self.clip_grad_dynamics,# 1.0
        max_norm_encoder=self.clip_grad_encoder,  # 1.0
    )
```

**Implementation:** `trainer.py:158-242`

```python
def _clip_gradients_per_head(
    grads: flax.core.FrozenDict,
    max_norm_value: float = 1.0,
    max_norm_policy: float = 1.0,
    max_norm_dynamics: float = 1.0,
    max_norm_encoder: float = 1.0,
) -> flax.core.FrozenDict:
    """Clip gradients separately for each network head."""
    # Categorizes gradients by:
    # - value_keys: 'value_head'
    # - policy_keys: 'output_refiner', 'trmreasoning'
    # - dynamics_keys: 'dynamics', 'action_embedding'
    # - encoder_keys: 'encoder', 'cnn'
    # - other_keys: uncategorized (no clipping)
```

**Analysis:**
- ✅ **Excellent feature** - Prevents one head from dominating
- ✅ Sensible defaults (1.0 per head)
- ❌ **Disabled by default** (`per_head_clipping=False`)
- ⚠️ Global clip (10.0) may override per-head benefits

**Recommendation:** Enable per-head clipping by default:
```python
per_head_clipping: bool = True,  # Enable for better multi-task balance
```

---

## Weight Decay

### Configuration

| Source | Value | Notes |
|--------|-------|-------|
| `trainer.py:368` | `1e-4` | Default |
| `training/config.py` | Uses trainer default | No override |

**Analysis:**
- ✅ Standard value for AdamW
- ✅ Applied via `optax.adamw()` (correct decoupled implementation)
- No issues identified

---

## Gradient Norm Monitoring

**Location:** `trainer.py:1522-1527`

```python
# Add gradient norm to metrics
metrics['grad_norm'] = optax.global_norm(grads)

# Per-head gradient norms for DS24 monitoring (Level 4+)
if getattr(self, 'track_grad_norms', False):
    per_head_norms = _compute_per_head_grad_norms(grads)
    metrics.update(per_head_norms)
```

**Per-head norms computed:**
- `grad_norm_value` - Value head
- `grad_norm_policy` - Policy head (TRM reasoning)
- `grad_norm_dynamics` - Dynamics network

**Analysis:**
- ✅ Global norm always tracked
- ✅ Per-head norms available with `track_grad_norms=True`
- ⚠️ `track_grad_norms` default is `False` (performance optimization)

---

## Issues Summary

### 🔴 HIGH: Inconsistent LR Defaults

**Problem:**
- `trainer.py`: 2e-4
- `config.py`: 1e-4

**Fix:**
```python
# In trainer.py
learning_rate: float = 1e-4,  # Sync with config
```

---

### 🟡 MEDIUM: Global Gradient Clip Very Permissive

**Problem:**
- Current: 10.0
- Typical: 1.0-5.0

**Risk:** Large gradients can cause:
- Parameter updates that overshoot
- Training instability
- Value function divergence

**Recommendation:**
```python
# Start with conservative clipping
optax.clip_by_global_norm(5.0)  # Or 1.0 for more stability
```

---

### 🟡 MEDIUM: Per-Head Clipping Disabled

**Problem:**
```python
per_head_clipping: bool = False  # Disabled
```

**Benefits of enabling:**
- Value head (often large grads) won't starve policy head
- Encoder gets stable gradients
- Dynamics learns at appropriate pace

**Recommendation:**
```python
per_head_clipping: bool = True  # Enable by default
```

---

### 🟢 LOW: No Learning Rate Decay by Default

**Problem:**
```python
lr_schedule_type: str = 'constant'  # No decay
```

**Trade-off:**
- Constant: Simpler, good for continual learning
- Cosine: Better final convergence, may need LR tuning

**Recommendation:** For fixed-length training, use cosine:
```python
lr_schedule_type: str = 'cosine'
lr_min: float = 1e-5  # 10% of initial
```

---

## Optimal Configuration Recommendations

### Conservative (Stable Training)

```python
# Optimizer
learning_rate: float = 1e-4
weight_decay: float = 1e-4
lr_schedule_type: str = 'cosine'
lr_min: float = 1e-5
lr_warmup_epochs: int = 10

# Gradient clipping
per_head_clipping: bool = True
clip_grad_value: float = 1.0
clip_grad_policy: float = 1.0
clip_grad_dynamics: float = 1.0
clip_grad_encoder: float = 1.0
# Reduce global clip when per-head is enabled
global_clip: float = 5.0  # Or remove entirely
```

### Aggressive (Faster Learning, Higher Risk)

```python
# Optimizer
learning_rate: float = 3e-4
weight_decay: float = 5e-5
lr_schedule_type: str = 'constant'

# Gradient clipping
per_head_clipping: bool = False
global_clip: float = 10.0  # Current setting
```

---

## Learning Rate Selection Guide

| Scenario | LR | Schedule | Rationale |
|----------|----| ----|------|
| Initial experiments | 1e-4 | constant | Safe baseline |
| Fast convergence | 3e-4 | cosine | Higher LR with decay |
| Fine-tuning | 1e-5 | constant | Don't destroy pretrained weights |
| Large batch (1024+) | 3e-4 | linear | Scale LR with batch size |
| Small batch (64) | 5e-5 | constant | Lower LR for noise |

---

## Validation Checklist

- [x] Optimizer is AdamW (correct)
- [x] Weight decay at 1e-4 (reasonable)
- [ ] LR defaults are consistent (🔴 **FAIL** - 2e-4 vs 1e-4)
- [x] LR schedules implemented correctly
- [ ] Global gradient clip ≤ 5.0 (🔴 **FAIL** - currently 10.0)
- [ ] Per-head clipping enabled (🟡 **WARNING** - disabled)
- [x] Gradient norm monitoring available
- [ ] LR decay used for fixed training (🟡 **WARNING** - constant by default)

---

## Testing Recommendations

1. **Gradient norm histogram:**
   ```python
   # Check if grads are being clipped frequently
   if metrics['grad_norm'] > global_clip * 0.9:
       print(f"Warning: Gradients near clip threshold: {metrics['grad_norm']:.2f}")
   ```

2. **Per-head gradient balance:**
   ```python
   # Check ratio of policy/value grad norms
   ratio = metrics['grad_norm_policy'] / (metrics['grad_norm_value'] + 1e-8)
   if ratio < 0.1:
       print(f"Warning: Policy gradients {ratio:.3f}x smaller than value")
   ```

3. **Learning rate schedule verification:**
   ```python
   # Log LR at each epoch to verify schedule
   print(f"Epoch {epoch}: LR = {current_lr:.2e}")
   ```

---

## References

- `mcts_trader/jax_agent/trainer.py:367-556` - Optimizer setup
- `mcts_trader/jax_agent/schedules.py` - LR schedules
- `mcts_trader/training/config.py:147-157` - Config defaults
- `mcts_trader/jax_agent/trainer.py:158-242` - Per-head gradient clipping
