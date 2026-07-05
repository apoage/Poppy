# TASK 024: Learning Rate Schedule

**Severity**: FEATURE
**Priority**: HIGH
**Location**: `mcts_trader/jax_agent/schedules.py`, `train_jax_fast.py`
**Status**: OPEN
**Effort**: Easy (1-2 hours)
**Dependencies**: None

---

## Problem

Current implementation uses constant learning rate throughout training:
```python
optax.adam(learning_rate=1e-4)
```

**Issues**:
1. Same LR for exploration phase and exploitation phase
2. No fine-tuning capability late in training
3. Can't recover from bad early updates

---

## Solution: LR Schedules

### Common Schedules

1. **Cosine Annealing** (Recommended for RL)
```
lr(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * t / T))
```

2. **Warmup + Cosine**
```
Warmup:  lr(t) = lr_max * t / warmup_steps
Decay:   lr(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * (t - warmup) / (T - warmup)))
```

3. **Step Decay**
```
lr(t) = lr_init * decay_rate^(t // step_size)
```

4. **Warmup-Stable-Decay (WSD)** - Good for unknown training length
```
Warmup:  Linear increase to lr_max
Stable:  Constant at lr_max
Decay:   Linear decrease to lr_min
```

---

## Research (2025)

### Why Warmup Works
- [2025 Analysis](https://arxiv.org/html/2406.09405v1): Warmup allows optimizer to collect accurate gradient statistics before large updates
- Particularly important for Adam (variance of adaptive LR is large early on)

### Beyond Cosine
- [Infinite LR Schedules (2025)](https://arxiv.org/html/2503.02844v1): WSD enables training without predetermined step count
- [WSD vs Cosine](https://ozanciga.wordpress.com/2025/10/06/learning-rate-scheduling-warmup-stable-decay-vs-cosine/): WSD matches or beats cosine in many settings

### Optax Support
Optax provides built-in schedules:
- `optax.cosine_decay_schedule()`
- `optax.warmup_cosine_decay_schedule()`
- `optax.linear_schedule()`
- `optax.piecewise_constant_schedule()`

---

## Implementation

### Add to `mcts_trader/jax_agent/schedules.py`

```python
"""Training schedules for learning rate and other hyperparameters."""

import optax
from typing import Optional

def create_lr_schedule(
    schedule_type: str,
    init_lr: float,
    min_lr: float = 1e-6,
    total_steps: int = 100000,
    warmup_steps: int = 1000,
    decay_rate: float = 0.1,
    step_size: int = 30000,
) -> optax.Schedule:
    """Create learning rate schedule.

    Args:
        schedule_type: 'constant', 'cosine', 'warmup_cosine', 'step', 'wsd'
        init_lr: Initial/maximum learning rate
        min_lr: Minimum learning rate
        total_steps: Total training steps
        warmup_steps: Warmup period (for warmup schedules)
        decay_rate: Decay factor (for step schedule)
        step_size: Steps between decays (for step schedule)

    Returns:
        optax.Schedule function
    """
    if schedule_type == 'constant':
        return optax.constant_schedule(init_lr)

    elif schedule_type == 'cosine':
        return optax.cosine_decay_schedule(
            init_value=init_lr,
            decay_steps=total_steps,
            alpha=min_lr / init_lr  # Final value ratio
        )

    elif schedule_type == 'warmup_cosine':
        return optax.warmup_cosine_decay_schedule(
            init_value=min_lr,
            peak_value=init_lr,
            warmup_steps=warmup_steps,
            decay_steps=total_steps - warmup_steps,
            end_value=min_lr
        )

    elif schedule_type == 'step':
        # Step decay: lr * decay_rate every step_size steps
        boundaries = list(range(step_size, total_steps, step_size))
        values = [init_lr * (decay_rate ** i) for i in range(len(boundaries) + 1)]
        return optax.piecewise_constant_schedule(
            init_value=init_lr,
            boundaries_and_scales={b: decay_rate for b in boundaries}
        )

    elif schedule_type == 'wsd':
        # Warmup-Stable-Decay
        stable_end = int(total_steps * 0.8)  # 80% stable, 20% decay
        return optax.join_schedules(
            schedules=[
                optax.linear_schedule(min_lr, init_lr, warmup_steps),
                optax.constant_schedule(init_lr),
                optax.linear_schedule(init_lr, min_lr, total_steps - stable_end),
            ],
            boundaries=[warmup_steps, stable_end]
        )

    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")
```

### Integration in Trainer

```python
# trainer.py

def __init__(self, ..., lr_schedule: Optional[optax.Schedule] = None):
    if lr_schedule is not None:
        self.optimizer = optax.adam(learning_rate=lr_schedule)
    else:
        self.optimizer = optax.adam(learning_rate=learning_rate)
```

### Integration in train_jax_fast.py

```python
# After parsing args
if args.lr_schedule != 'constant':
    from mcts_trader.jax_agent.schedules import create_lr_schedule

    # Estimate total steps
    total_steps = args.epochs * (args.steps // args.batch_size) * args.train_batches

    lr_schedule = create_lr_schedule(
        schedule_type=args.lr_schedule,
        init_lr=args.lr,
        min_lr=args.lr_min,
        total_steps=total_steps,
        warmup_steps=args.lr_warmup_steps,
    )
else:
    lr_schedule = None

trainer = MuZeroTrainer(
    ...,
    learning_rate=args.lr,
    lr_schedule=lr_schedule,
)
```

---

## CLI Arguments

```python
parser.add_argument('--lr-schedule', type=str, default='constant',
    choices=['constant', 'cosine', 'warmup_cosine', 'step', 'wsd'],
    help='Learning rate schedule type')
parser.add_argument('--lr-min', type=float, default=1e-6,
    help='Minimum learning rate for schedules')
parser.add_argument('--lr-warmup-steps', type=int, default=1000,
    help='Warmup steps for warmup schedules')
```

---

## Testing

```bash
# Test cosine schedule
python train_jax_fast.py \
    --synthetic-level 0 \
    --epochs 200 \
    --lr 3e-4 \
    --lr-schedule warmup_cosine \
    --lr-warmup-steps 500 \
    --name test_cosine_lr

# Compare to constant
python train_jax_fast.py \
    --synthetic-level 0 \
    --epochs 200 \
    --lr 3e-4 \
    --name test_constant_lr
```

---

## Files to Create/Modify

| File | Changes |
|------|---------|
| `mcts_trader/jax_agent/schedules.py` | Add `create_lr_schedule()` |
| `mcts_trader/jax_agent/trainer.py` | Accept lr_schedule parameter |
| `train_jax_fast.py` | Add CLI args, create schedule |

---

## Success Criteria

1. Training with warmup_cosine converges stably
2. Final performance >= constant LR baseline
3. Loss curve is smoother (especially late in training)
4. Current LR logged to metrics for debugging
