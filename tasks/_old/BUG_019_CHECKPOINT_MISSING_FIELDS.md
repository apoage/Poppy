# Bug: Checkpoint May Miss Required Fields

## Priority: MEDIUM
**Status:** DEFERRED

The Flax checkpointer saves full TrainState which includes all
critical fields (params, opt_state, step_count). Entropy coef
and buffer state could be added as enhancement.
**Component:** `train_jax_fast.py`

## Problem Description

Checkpoint saving/loading may not include all required state, causing incomplete restoration.

### Potentially Missing Fields

1. Optimizer state momentum buffers
2. Replay buffer state
3. Entropy coefficient (if adaptive)
4. RNG state
5. Training metrics history

### Current Checkpoint Structure

```python
# What's saved
ckpt = {
    'params': trainer_state.params,
    'opt_state': trainer_state.opt_state,
    'step': trainer_state.step,
}
```

### What May Be Missing

```python
# Should also save:
ckpt = {
    ...
    'entropy_coef': trainer.entropy_coef,
    'rng_key': current_key,
    'buffer_state': buffer.get_state(),
    'metrics_history': metrics_log,
}
```

### Impact

- Resuming training doesn't match paused state
- Optimizer momentum lost
- Exploration parameters reset
- Buffer contents lost

## Solution

### Complete checkpoint saving

```python
def save_checkpoint(trainer_state, trainer, buffer, key, path):
    ckpt = {
        'params': trainer_state.params,
        'opt_state': trainer_state.opt_state,
        'step': trainer_state.step,
        'entropy_coef': trainer.entropy_coef,
        'rng_key': key,
        'buffer_position': buffer.position,
        'buffer_generation': buffer.generation,
    }
    orbax.save(path, ckpt)
```

### Complete checkpoint loading

```python
def load_checkpoint(path, trainer, buffer):
    ckpt = orbax.restore(path)
    # Restore all fields
    trainer.entropy_coef = ckpt.get('entropy_coef', 0.01)
    buffer.position = ckpt.get('buffer_position', 0)
    # etc.
```

## Files to Modify

| File | Change |
|------|--------|
| `train_jax_fast.py` | Save/restore complete state |

## Testing

1. Train for 100 epochs, save
2. Resume and verify exact same behavior
3. Compare metrics before/after resume
