# TASK 003: Reset Optimizer State on Curriculum Regime Change

**Severity**: MEDIUM
**Location**: `mcts_trader/jax_agent/trainer.py:504-516`
**Status**: DONE (2025-12-29)
**Estimated Complexity**: Medium

---

## Problem Description

When loading a checkpoint for a new curriculum level, the optimizer state (AdamW momentum and variance estimates) is preserved from the previous regime. This can cause:

1. **Momentum mismatch**: Old momentum vectors push weights in directions that were optimal for previous regime but harmful for new one
2. **Variance bias**: EMA of squared gradients reflects old gradient magnitudes, causing incorrect learning rate scaling
3. **Initial instability**: First epochs after regime change may have erratic updates

### Current Code (`trainer.py:504-516`):
```python
def load_checkpoint(self, path: str, step: int = None):
    """Load model checkpoint"""
    from flax.training import checkpoints
    restored_state = checkpoints.restore_checkpoint(
        ckpt_dir=path,
        target=self.state,  # Restores FULL state including opt_state
        step=step
    )
    return restored_state
```

The `TrainState` object includes:
- `params` - Model weights (SHOULD preserve)
- `opt_state` - Optimizer momentum/variance (SHOULD reset on regime change)
- `target_params` - Target network (SHOULD preserve)
- `step_count` - Training step counter

---

## Root Cause Analysis

Flax's `checkpoints.restore_checkpoint` restores the entire `TrainState` including optimizer internals. This is correct for resuming training on SAME data, but wrong for transfer learning to new domains.

### AdamW State Structure:
```python
opt_state = (
    ScaleByAdamState(
        count=step_count,
        mu=momentum_ema,      # First moment (momentum)
        nu=variance_ema       # Second moment (variance)
    ),
    ...
)
```

When regime changes:
- `mu` (momentum) still points toward old optimum
- `nu` (variance) reflects old gradient magnitudes
- Learning rate effectively becomes wrong for new gradients

---

## Proposed Solution

### Option A: Add `--reset-optimizer` CLI Flag (Recommended)

```python
# train_jax_fast.py
parser.add_argument('--reset-optimizer', action='store_true',
                    help='Reset optimizer state when loading checkpoint')

# After loading checkpoint:
if args.reset_optimizer:
    print("🔄 Resetting optimizer state for new regime...")
    state = trainer.reset_optimizer_state(state)
```

### Option B: Gradual Momentum Decay

Instead of hard reset, decay old momentum over first N epochs:
```python
# Soft reset: scale down old momentum
decay_factor = 0.1
new_mu = jax.tree_map(lambda x: x * decay_factor, old_opt_state.mu)
```

### Option C: Learning Rate Warmup (Complementary)

Use learning rate warmup after regime change to reduce impact of stale optimizer state:
```python
if args.load_checkpoint:
    # Linear warmup for first 100 steps
    warmup_schedule = optax.linear_schedule(
        init_value=1e-6,
        end_value=args.learning_rate,
        transition_steps=100
    )
```

---

## Implementation Plan

### Step 1: Add `reset_optimizer_state()` Method (20 min)

```python
# mcts_trader/jax_agent/trainer.py

def reset_optimizer_state(self, state: TrainState) -> TrainState:
    """
    Reset optimizer state while preserving model parameters.

    Use when transferring to new curriculum level to avoid
    momentum/variance mismatch.
    """
    # Re-initialize optimizer with current params
    new_opt_state = self.optimizer.init(state.params)

    # Create new state with fresh optimizer
    return state.replace(
        opt_state=new_opt_state,
        # Optionally reset step_count for learning rate schedules
        # step_count=0
    )
```

### Step 2: Add CLI Flag (10 min)

```python
# train_jax_fast.py
parser.add_argument('--reset-optimizer', action='store_true',
                    help='Reset optimizer state (momentum/variance) when loading checkpoint')

# Usage:
if args.load_checkpoint:
    restored_state = trainer.restore_checkpoint(abs_ckpt_path)
    if restored_state:
        if args.reset_optimizer:
            restored_state = trainer.reset_optimizer_state(restored_state)
            print("🔄 Optimizer state reset for new regime")
        trainer.state = restored_state
```

### Step 3: Update Dojo Manager (10 min)

```python
# dojo_dynamic_curriculum.py

def _train_phase(self, ...):
    cmd = [...]

    if total_epochs_so_far > 0 and os.path.exists(checkpoint_dir):
        cmd.extend(['--load-checkpoint', checkpoint_dir])

        # Reset optimizer on level changes (not phase changes)
        if self._is_level_transition():
            cmd.append('--reset-optimizer')
```

### Step 4: Add Learning Rate Warmup Option (20 min)

```python
# train_jax_fast.py
parser.add_argument('--lr-warmup-steps', type=int, default=0,
                    help='Linear LR warmup steps after checkpoint load')

# In trainer initialization:
if args.lr_warmup_steps > 0:
    schedule = optax.join_schedules(
        schedules=[
            optax.linear_schedule(1e-6, args.learning_rate, args.lr_warmup_steps),
            optax.constant_schedule(args.learning_rate)
        ],
        boundaries=[args.lr_warmup_steps]
    )
```

---

## Acceptance Criteria

- [ ] `reset_optimizer_state()` method implemented in trainer
- [ ] `--reset-optimizer` CLI flag added
- [ ] Dojo manager uses flag on level transitions
- [ ] Optional: Learning rate warmup implemented
- [ ] Training stable after regime change with optimizer reset

---

## Files to Modify

1. `mcts_trader/jax_agent/trainer.py` - Add reset method
2. `train_jax_fast.py` - Add CLI flags
3. `dojo_dynamic_curriculum.py` - Use flags on transitions

---

## Testing

```bash
# Test 1: Verify optimizer reset
python train_jax_fast.py \
    --synthetic-level 1 \
    --load-checkpoint jax_ckpt_fast/level0 \
    --reset-optimizer

# Check logs for:
# "🔄 Optimizer state reset for new regime"

# Test 2: Compare training curves
# A) Without reset (expect instability in first 10-20 epochs)
# B) With reset (expect smooth adaptation)

# Test 3: Verify model weights preserved
# Compare initial predictions before/after reset
# (Should be identical - only optimizer state changes)
```

---

## Risk Assessment

**Risk**: Resetting optimizer loses learned gradient statistics that could help.

**Mitigation**:
- Make reset optional (flag-based)
- Consider soft reset (momentum decay) instead of hard reset
- Use learning rate warmup to reduce initial update magnitude
- Only reset on LEVEL changes, not phase changes within same level

---

## Research Reference

From "An Empirical Study of Example Forgetting during Deep Neural Network Learning" (Toneva et al., 2019):
> When data distribution shifts, momentum-based optimizers can cause "negative transfer"
> where old momentum directions conflict with new gradient signals.

Recommendation: Reset optimizer state when distribution shift is significant (different curriculum level).
