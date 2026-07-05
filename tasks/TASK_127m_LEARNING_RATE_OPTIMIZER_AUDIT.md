# TASK_127m: Learning Rate and Optimizer Audit

**Status**: AUDITED (Working Correctly, Minor Observations)
**Priority**: Low
**Created**: 2026-01-31

## Scope

Audited optimizer configuration:
- `mcts_trader/jax_agent/trainer.py` - Optimizer creation and gradient clipping
- `mcts_trader/jax_agent/schedules.py` - LR schedule implementations
- `mcts_trader/training/config.py` - Default hyperparameters
- `mcts_trader/training/cli.py` - CLI arguments

## Optimizer Configuration

### 1. Optimizer Type: AdamW

**Location**: `trainer.py:553-556`

```python
self.optimizer = optax.chain(
    optax.clip_by_global_norm(10.0),
    optax.adamw(learning_rate=lr_schedule, weight_decay=weight_decay)
)
```

**Assessment**:
- Optimizer: AdamW (Adam with decoupled weight decay)
- Weight decay: 1e-4 (default, standard for transformers)
- Beta values: Default (0.9, 0.999) - not customized
- Global gradient clipping: 10.0 (applied BEFORE optimizer)

### 2. Default Hyperparameters

| Parameter | Default | Location | Assessment |
|-----------|---------|----------|------------|
| `lr` | 1e-4 | config.py:147 | Conservative, good for stability |
| `lr_min` | 1e-6 | config.py:149 | Reasonable floor |
| `lr_warmup_epochs` | 10 | config.py:150 | Standard warmup |
| `weight_decay` | 1e-4 | trainer.py:368 | Standard for transformers |
| `entropy_coef` | 0.01 | config.py:152 | Low exploration bonus |
| `discount` | 0.997 | config.py:151 | High (long horizon) |

### 3. Learning Rate Schedule Options

**Location**: `schedules.py:86-177`

```python
schedule_type: Literal['constant', 'cosine', 'linear'] = 'constant'
```

**Available schedules**:
1. **Constant**: No decay (default)
2. **Cosine**: Warmup → peak → cosine decay to `lr_min`
3. **Linear**: Warmup → peak → linear decay to `lr_min`

**Warmup implementation** (`schedules.py:104-105`):
```python
if epoch < self.warmup_epochs:
    return self.initial_lr * (epoch + 1) / self.warmup_epochs
```

**Assessment**:
- Warmup is LINEAR ramp from 0 → initial_lr over `warmup_epochs`
- After warmup, schedule follows cosine/linear decay
- optax.warmup_cosine_decay_schedule used for cosine variant

### 4. Gradient Clipping Configuration

**Global Clipping** (`trainer.py:554`):
```python
optax.clip_by_global_norm(10.0)  # Always active
```

**Per-Head Clipping** (TASK_025, optional):
```python
# trainer.py:400-404
per_head_clipping: bool = False,  # Disabled by default
clip_grad_value: float = 1.0,
clip_grad_policy: float = 1.0,
clip_grad_dynamics: float = 1.0,
clip_grad_encoder: float = 1.0,
```

**Per-head clipping implementation** (`trainer.py:180-238`):
- Categorizes gradients by component (value_head, policy, dynamics, encoder)
- Computes separate norms per component
- Applies independent scaling factors

**Assessment**:
- Global clipping at 10.0 is generous (rarely triggered)
- Per-head clipping defaults are conservative (1.0 each)
- Per-head clipping is OFF by default

## Audit Results

### Issue 1: Default LR Schedule is Constant

**Severity**: Low

**Problem**: Default `lr_schedule='constant'` means no LR decay.

```python
# config.py:148
lr_schedule: str = "constant"
```

**Impact**:
- No automatic LR reduction over training
- May oscillate near optimum instead of converging smoothly
- Works fine for short runs, less ideal for long training

**Recommendation**: Consider `--lr-schedule cosine` for long training runs (500+ epochs).

### Issue 2: Global Clip Norm of 10.0 is High

**Severity**: Low

**Problem**: Global gradient clipping at 10.0 rarely activates.

```python
optax.clip_by_global_norm(10.0)
```

**Impact**:
- Only clips when total gradient norm exceeds 10.0
- Most updates will NOT be clipped
- Provides emergency protection but not regularization

**Recommendation**: Consider reducing to 1.0-5.0 for more aggressive regularization, especially if training becomes unstable.

### Issue 3: Warmup Doesn't Reset on Checkpoint Load

**Severity**: Medium

**Problem**: When loading a checkpoint, the LR schedule continues from step 0 in the schedule, not from the checkpoint's step count.

**Evidence**: The optimizer is created fresh in `__init__` with a new schedule:
```python
# trainer.py:540-551
if lr_schedule_type == 'constant':
    lr_schedule = learning_rate
else:
    lr_sched = LRSchedule(...)
    lr_schedule = lr_sched.to_optax_schedule(steps_per_epoch)
```

When loading a checkpoint, if using a non-constant schedule:
- LR schedule restarts from step 0
- Model parameters are at step N
- This causes LR mismatch

**Workaround**: Use `--lr-schedule constant` when resuming, or manually account for warmup.

### Issue 4: AdamW Uses Default Betas

**Severity**: Info

**Observation**: AdamW uses default beta values (0.9, 0.999).

```python
optax.adamw(learning_rate=lr_schedule, weight_decay=weight_decay)
# No b1/b2 specified
```

**Assessment**: Default betas are generally fine. Some transformers benefit from lower b2 (0.98-0.99) for faster adaptation, but this is domain-specific.

### Issue 5: Weight Decay Not Configurable via CLI

**Severity**: Low

**Problem**: `weight_decay` is hardcoded in `trainer.py:368` with no CLI flag.

```python
weight_decay: float = 1e-4,  # No --weight-decay argument
```

**Impact**: Cannot tune weight decay without code changes.

**Recommendation**: Add `--weight-decay` CLI argument.

## Gradient Flow Verification

**Gradient clipping order**:
```
1. Compute gradients (jax.value_and_grad)
2. Optional: Per-head clipping (TASK_025)
3. Global clipping (optax.clip_by_global_norm)
4. AdamW update (momentum, weight decay)
```

**Assessment**:
- Clipping applied BEFORE optimizer statistics update
- This is correct - prevents corrupt momentum from gradient spikes
- AdamW's internal clipping (eps=1e-8) provides additional stability

## Optimizer State Management

### Reset on Curriculum Change

**Location**: `trainer.py:1821-1838`

```python
def reset_optimizer_state(self, state: TrainState) -> TrainState:
    """Reset optimizer state while preserving model parameters."""
    new_opt_state = self.optimizer.init(state.params)
    return state.replace(opt_state=new_opt_state)
```

**Assessment**:
- Resets Adam momentum and variance estimates
- Preserves model weights and target network
- Useful when transferring between curriculum levels
- Called via `--reset-optimizer` CLI flag

### LR Decay on Rollback

**Location**: `config.py:325`

```python
lr_decay_on_rollback: float = 0.5  # Halve LR on checkpoint rollback
```

**Assessment**: When training detects divergence and rolls back to a previous checkpoint, LR is reduced by 50%. Good safety mechanism.

## Summary

| Aspect | Status | Notes |
|--------|--------|-------|
| Optimizer type | AdamW | Standard choice for transformers |
| Weight decay | 1e-4 | Good default, not CLI-tunable |
| LR schedule | Constant (default) | Consider cosine for long runs |
| LR warmup | 10 epochs | Linear ramp, works correctly |
| Global grad clip | 10.0 | High threshold, rarely activates |
| Per-head clipping | Off (optional) | Available via TASK_025 |
| Betas | Default (0.9, 0.999) | Standard, not tunable |
| Checkpoint resume | Schedule restarts | May cause LR mismatch |

## Recommendations

1. **Add `--weight-decay` CLI argument** for tuning regularization
2. **Consider lower global clip norm** (1.0-5.0) for more regularization
3. **Use `--lr-schedule cosine`** for long training runs
4. **Fix schedule on checkpoint resume** - continue from saved step count
5. **Document LR schedule interaction with checkpoints** in README

## Configuration Reference

```bash
# Conservative (default)
python train_jax_fast.py --lr 1e-4 --lr-schedule constant

# Aggressive with decay
python train_jax_fast.py --lr 3e-4 --lr-schedule cosine --lr-warmup-epochs 20

# With per-head clipping
python train_jax_fast.py --per-head-clipping --clip-grad-value 0.5 --clip-grad-policy 1.0

# Reset optimizer on level change
python train_jax_fast.py --load-checkpoint X --reset-optimizer
```

## Related Tasks

- TASK_024: LR Schedule Implementation
- TASK_025: Per-Head Gradient Clipping
- TASK_127g: Entropy and Policy Collapse Audit
