# TASK_685: Fix Gradient Over-Suppression

**Status**: DONE
**Priority**: 2
**Source**: Analysis 003 (Claude) — "Gradient Clipping Mismatch"

## Problem

```
Observed grad_norm: 13-16 (every step)
grad_clip threshold: 1.0 (optax.clip_by_global_norm)
Suppression factor: 13-16× every single step
```

The optimizer chain in `easy_trainer.py`:
```python
optax.chain(
    optax.clip_by_global_norm(1.0),    # clips first
    optax.adamw(schedule, weight_decay=1e-4)
)
```

Consequences:
1. **Cross-head gradient competition**: Adaptive trade weighting (24×) amplifies policy trade gradients which dominate global norm. Value head and hold gradients get same suppression factor even though they're not the source.
2. **23% variance** in suppression (13× to 16×) introduces multiplicative noise on entire update.
3. **Value head learns at policy trade's pace**, not its own optimal pace.

## Fix

Change `grad_clip` from 1.0 to 5.0 in `easy_config.py`:

```python
grad_clip: float = 5.0  # Was 1.0 — stop 13-16× over-suppression
```

This reduces suppression from 13-16× to 2.6-3.2× (still protective, much less aggressive).

## Why 5.0?

- Grad norms at 13-16 are healthy (no NaN/explosion risk)
- 5.0 still clips outlier batches (norms >20)
- Adam's second moment tracking partially compensates for consistent clipping, but the 23% variance is the issue
- grad_clip=5.0 → 2.6-3.2× suppression → <25% variance effect

## Risk

**Low** — grad norms at 13-16 are not dangerous. No explosion risk.

If norms spike >30 after increasing clip (unlikely), reduce back to 3.0.

## Changes

### 1. `easy_config.py` (line 41)

```python
grad_clip: float = 5.0  # Was 1.0
```

That's it. Single line change.

## Verification

1. Monitor grad_norm first 20 epochs — should see same 13-16 range
2. Monitor agree_trade — should improve if value head gets better gradient signal
3. If loss becomes unstable (>5.0), reduce clip to 3.0
4. Can be tested with current architecture (no need to wait for TASK_683/684)
