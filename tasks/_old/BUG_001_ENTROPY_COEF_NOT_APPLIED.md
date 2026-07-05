# Bug: Entropy Coefficient Not Applied in JIT-Compiled Training

## Priority: CRITICAL
**Status:** DONE
**Component:** `train_jax_fast.py`, `mcts_trader/jax_agent/trainer.py`

## Problem Description

The entropy coefficient (`entropy_coef`) is captured at JIT compilation time and never updated during training. When adaptive entropy modifies `self.entropy_coef`, the change has no effect because the JIT-compiled `update_step` function retains the original value.

### Root Cause

```python
# train_jax_fast.py:1143-1145
if args.adaptive_entropy:
    hold_pct = metrics.get('hold_pct', 50) / 100.0
    if hold_pct > args.entropy_hold_threshold:
        trainer.entropy_coef = max(0.05, min(0.3, trainer.entropy_coef * 1.05))
```

This modifies `trainer.entropy_coef` on the Python object, but JAX's JIT has already captured the original value as a static constant.

### Impact

- Adaptive entropy feature is **completely broken**
- Training cannot dynamically adjust exploration
- Agents may collapse to deterministic policies without any correction

## Solution

### Option A: Pass entropy_coef as runtime parameter (Recommended)

Modify `update_step` to accept `entropy_coef` as an explicit parameter:

```python
# trainer.py - update_step signature
@partial(jax.jit, static_argnums=(0,))
def update_step(self, state, batch, entropy_coef):  # Add param
    ...
    entropy_loss = entropy_coef * -jnp.mean(entropy)  # Use param
    ...
```

```python
# train_jax_fast.py - call with current value
metrics = trainer.update_step(trainer_state, batch, trainer.entropy_coef)
```

### Option B: Use non-JIT entropy scaling

Apply entropy bonus outside JIT in the loss aggregation.

## Files to Modify

| File | Change |
|------|--------|
| `trainer.py` | Add `entropy_coef` parameter to `update_step` |
| `train_jax_fast.py` | Pass current entropy_coef value in training loop |

## Testing

1. Run with `--adaptive-entropy`
2. Verify entropy_coef changes are reflected in actual training loss
3. Check that high HOLD% triggers entropy boost effect

## Related Issues

- BUG_002: Entropy bonus logic is also backwards (separate fix)
