# TASK 020: MCTS Simulations Warmup Schedule

**Severity**: FEATURE
**Priority**: HIGH
**Location**: `mcts_trader/jax_agent/trainer.py`, `train_jax_fast.py`
**Status**: COMPLETE
**Effort**: Medium (2-3 hours)
**Dependencies**: None

---

## Problem

With high MCTS simulations (32-64), the policy quickly converges to deterministic actions (80%+ HOLD) before the value function has learned meaningful estimates. This creates a chicken-and-egg problem:
- MCTS needs good value estimates to find good actions
- Value function needs diverse experiences to learn
- High sims = deterministic policy = no exploration = poor value learning

**Symptom**: Reward collapses early in training despite reasonable entropy.

---

## Solution: Simulation Warmup

Gradually increase MCTS simulations over training epochs:

```
Epoch 1-N/4:    8 sims   (high exploration, learn basic patterns)
Epoch N/4-N/2:  16 sims  (refine value function)
Epoch N/2-3N/4: 32 sims  (improve policy)
Epoch 3N/4+:    64 sims  (full planning depth)
```

**Analogy**: Teaching chess - learn piece values first, then tactics, then deep calculation.

---

## Research (2025)

### MuZero Variants
- [Muesli](https://arxiv.org/pdf/2104.06159) removes MCTS entirely during training, using policy optimization
- [OptionZero](https://arxiv.org/html/2511.09219) modifies expansion threshold for exploration
- Standard MuZero uses Dirichlet noise at root for exploration

### Key Insight
From MuZero analysis: "The learned model is generally not accurate for policy evaluation early in training" - this supports using fewer sims initially when the model is unreliable.

---

## Implementation

### New CLI Arguments

```python
# train_jax_fast.py
parser.add_argument('--sims-warmup', type=int, default=0,
    help='Epochs to warm up simulations (0=disabled)')
parser.add_argument('--sims-min', type=int, default=8,
    help='Minimum simulations during warmup')
parser.add_argument('--sims-max', type=int, default=None,
    help='Maximum simulations (defaults to --sims)')
```

### New Module: `mcts_trader/jax_agent/schedules.py`

```python
"""Training schedules for curriculum-based hyperparameters."""

from dataclasses import dataclass
from typing import Optional

@dataclass
class SimsSchedule:
    """MCTS simulation count schedule."""
    min_sims: int = 8
    max_sims: int = 64
    warmup_epochs: int = 100
    schedule_type: str = 'linear'  # 'linear', 'step', 'exponential'

    def get_sims(self, epoch: int) -> int:
        """Get simulation count for current epoch."""
        if self.warmup_epochs <= 0:
            return self.max_sims

        progress = min(1.0, epoch / self.warmup_epochs)

        if self.schedule_type == 'linear':
            sims = self.min_sims + progress * (self.max_sims - self.min_sims)
        elif self.schedule_type == 'step':
            # Step schedule: 8 -> 16 -> 32 -> 64
            if progress < 0.25:
                sims = self.min_sims
            elif progress < 0.5:
                sims = 16
            elif progress < 0.75:
                sims = 32
            else:
                sims = self.max_sims
        elif self.schedule_type == 'exponential':
            # Exponential: more time at low sims
            import math
            log_min = math.log2(self.min_sims)
            log_max = math.log2(self.max_sims)
            sims = 2 ** (log_min + progress * (log_max - log_min))
        else:
            sims = self.max_sims

        return int(sims)
```

### Integration in Training Loop

```python
# train_jax_fast.py - in training loop
if args.sims_warmup > 0:
    sims_schedule = SimsSchedule(
        min_sims=args.sims_min,
        max_sims=args.sims or args.sims_max,
        warmup_epochs=args.sims_warmup
    )
    current_sims = sims_schedule.get_sims(epoch)
    trainer.update_simulations(current_sims)  # New method
```

### Trainer Update

```python
# trainer.py
def update_simulations(self, num_simulations: int):
    """Update MCTS simulation count (for warmup schedule)."""
    self.search = create_mctx_search(
        self.model,
        num_simulations=num_simulations,
        use_gumbel=True
    )
```

---

## Testing

```bash
# Test simulation warmup
python train_jax_fast.py \
    --synthetic-level 0 \
    --epochs 200 \
    --sims 64 \
    --sims-warmup 100 \
    --sims-min 8 \
    --name test_sims_warmup

# Compare to no warmup
python train_jax_fast.py \
    --synthetic-level 0 \
    --epochs 200 \
    --sims 64 \
    --name test_no_warmup
```

**Expected**: Warmup version should show:
- Higher action diversity early in training
- Better value calibration (lower value std)
- Faster reward improvement

---

## Files to Modify

| File | Changes |
|------|---------|
| `mcts_trader/jax_agent/schedules.py` | NEW: SimsSchedule class |
| `mcts_trader/jax_agent/trainer.py` | Add `update_simulations()` method |
| `train_jax_fast.py` | Add CLI args, integrate schedule |
| `hyperparam_tournament.py` | Add sims warmup to config |

---

## Success Criteria

1. Action distribution stays diverse (HOLD < 60%) during warmup
2. Value function MSE decreases steadily
3. Final performance >= non-warmup baseline
4. No training instability during sims transitions
