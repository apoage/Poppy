# TASK 021: PopArt Value Normalization

**Severity**: FEATURE
**Priority**: HIGH
**Location**: `mcts_trader/jax_agent/popart.py` (NEW), `trainer.py`
**Status**: COMPLETE
**Effort**: Medium (3-4 hours)
**Dependencies**: None

---

## Problem

Value targets vary dramatically across:
1. **Curriculum levels**: L0 sine waves vs L4 real data have different reward scales
2. **Training progress**: Early random exploration vs late exploitation
3. **Market regimes**: Bull vs bear markets have different return distributions

Current approach uses fixed symlog transformation, but doesn't adapt to running statistics.

**Symptom**: Value function struggles during curriculum transitions; high value std early in training.

---

## Solution: PopArt (Preserving Outputs Precisely while Adaptively Rescaling Targets)

PopArt normalizes value targets using running mean/std, then adjusts the value head weights to preserve outputs when statistics change.

### Key Equations

```
Normalized target: y_norm = (y - μ) / σ
Value prediction: v = W @ h + b  (standard)
PopArt update:
  W_new = W_old * (σ_old / σ_new)
  b_new = (σ_old * b_old + μ_old - μ_new) / σ_new
```

This ensures `v_new(h) ≈ v_old(h)` even when statistics change.

---

## Research (2025)

### DeepMind's PopArt
- [Original Paper](https://arxiv.org/pdf/1602.07714): "Learning values across many orders of magnitude"
- [Multi-task RL](https://ar5iv.labs.arxiv.org/html/1809.04474): Extended to actor-critic for IMPALA
- [DeepMind Blog](https://deepmind.google/discover/blog/preserving-outputs-precisely-while-adaptively-rescaling-targets/)

### Key Results
- PopArt-IMPALA: 110% median normalized score on Atari-57
- Enables single agent to play multiple games with different reward scales
- Eliminates need for reward clipping (which distorts objectives)

---

## Implementation

### New Module: `mcts_trader/jax_agent/popart.py`

```python
"""PopArt: Adaptive value normalization for curriculum learning."""

import jax
import jax.numpy as jnp
from flax import struct
from typing import Tuple

@struct.dataclass
class PopArtState:
    """Running statistics for PopArt normalization."""
    mean: jnp.ndarray  # Running mean of targets
    var: jnp.ndarray   # Running variance of targets
    count: int         # Number of samples seen
    # Store old stats for weight correction
    mean_old: jnp.ndarray
    std_old: jnp.ndarray

def init_popart_state() -> PopArtState:
    """Initialize PopArt state."""
    return PopArtState(
        mean=jnp.zeros(()),
        var=jnp.ones(()),
        count=0,
        mean_old=jnp.zeros(()),
        std_old=jnp.ones(()),
    )

def update_statistics(
    state: PopArtState,
    targets: jnp.ndarray,
    beta: float = 0.0001  # EMA decay for running stats
) -> PopArtState:
    """Update running mean/var with new targets."""
    # Store old statistics
    std_old = jnp.sqrt(state.var + 1e-8)

    # Compute batch statistics
    batch_mean = jnp.mean(targets)
    batch_var = jnp.var(targets)

    # EMA update
    new_mean = (1 - beta) * state.mean + beta * batch_mean
    new_var = (1 - beta) * state.var + beta * batch_var

    return PopArtState(
        mean=new_mean,
        var=new_var,
        count=state.count + len(targets),
        mean_old=state.mean,
        std_old=std_old,
    )

def normalize_targets(
    targets: jnp.ndarray,
    state: PopArtState
) -> jnp.ndarray:
    """Normalize targets using running statistics."""
    std = jnp.sqrt(state.var + 1e-8)
    return (targets - state.mean) / std

def denormalize_values(
    values: jnp.ndarray,
    state: PopArtState
) -> jnp.ndarray:
    """Denormalize predicted values back to original scale."""
    std = jnp.sqrt(state.var + 1e-8)
    return values * std + state.mean

def compute_weight_correction(
    state: PopArtState
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute scale and shift corrections for value head weights.

    Returns:
        scale: Multiply weights by this
        shift: Add to bias after scaling
    """
    std_new = jnp.sqrt(state.var + 1e-8)
    std_old = state.std_old + 1e-8

    scale = std_old / std_new
    shift = (state.mean_old - state.mean) / std_new

    return scale, shift

def apply_weight_correction(
    params: dict,
    scale: jnp.ndarray,
    shift: jnp.ndarray,
    value_head_prefix: str = 'value_head'
) -> dict:
    """Apply PopArt correction to value head weights.

    Modifies: W_new = W_old * scale, b_new = b_old * scale + shift
    """
    def correct_layer(p):
        if 'kernel' in p and 'bias' in p:
            return {
                'kernel': p['kernel'] * scale,
                'bias': p['bias'] * scale + shift,
            }
        return p

    # Apply to all value head layers
    new_params = {}
    for key, value in params.items():
        if value_head_prefix in key:
            new_params[key] = correct_layer(value)
        else:
            new_params[key] = value

    return new_params
```

### Integration in Trainer

```python
# trainer.py

class MuZeroTrainer:
    def __init__(self, ..., use_popart: bool = False):
        self.use_popart = use_popart
        if use_popart:
            from .popart import init_popart_state
            self.popart_state = init_popart_state()

    def _compute_loss(self, params, batch, rng):
        # Get targets
        target_values = batch['values']

        if self.use_popart:
            from .popart import normalize_targets
            target_values = normalize_targets(target_values, self.popart_state)

        # ... rest of loss computation

    def train_step(self, state, batch, rng):
        # Update PopArt statistics BEFORE loss computation
        if self.use_popart:
            from .popart import (
                update_statistics,
                compute_weight_correction,
                apply_weight_correction
            )
            old_state = self.popart_state
            self.popart_state = update_statistics(
                self.popart_state, batch['values']
            )

            # Correct value head weights to preserve outputs
            scale, shift = compute_weight_correction(self.popart_state)
            state = state.replace(
                params=apply_weight_correction(state.params, scale, shift)
            )

        # ... rest of training step
```

---

## CLI Arguments

```python
# train_jax_fast.py
parser.add_argument('--use-popart', action='store_true', default=False,
    help='Enable PopArt value normalization (recommended for curriculum)')
parser.add_argument('--popart-beta', type=float, default=0.0001,
    help='PopArt EMA decay rate (default: 0.0001)')
```

---

## Testing

```bash
# Test PopArt on curriculum transition
python train_jax_fast.py \
    --synthetic-level 0 \
    --epochs 100 \
    --use-popart \
    --name test_popart_L0

# Resume to L1 (where reward scale changes)
python train_jax_fast.py \
    --synthetic-level 1 \
    --load-checkpoint jax_ckpt_fast/test_popart_L0 \
    --use-popart \
    --name test_popart_L1
```

**Expected**: Smoother value loss during curriculum transitions.

---

## Files to Create/Modify

| File | Changes |
|------|---------|
| `mcts_trader/jax_agent/popart.py` | NEW: PopArt implementation |
| `mcts_trader/jax_agent/trainer.py` | Integrate PopArt |
| `train_jax_fast.py` | Add CLI args |

---

## Success Criteria

1. Value loss doesn't spike during curriculum transitions
2. Value predictions stay calibrated across reward scales
3. No performance regression on single-level training
4. Minimal overhead (< 5% slower)
