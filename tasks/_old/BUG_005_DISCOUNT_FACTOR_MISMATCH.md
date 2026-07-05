# Bug: Discount Factor Mismatch Between Components

## Priority: HIGH
**Status:** DONE

## Fix Applied (2026-01-01)

1. Changed CLI default from 0.99 to 0.997 for consistency
2. Added `discount=args.discount` to MuZeroTrainer initialization
3. Verified that `run_self_play()` already uses `args.discount` correctly
**Component:** `train_jax_fast.py`, `mcts_trader/jax_agent/trainer.py`

## Problem Description

Different parts of the codebase use different default discount factors:
- `run_self_play()`: Uses `discount=0.997` (hardcoded default)
- `MuZeroTrainer`: Uses `discount=0.99` (class default)
- CLI: Added `--discount` but not wired everywhere

### Locations

```python
# train_jax_fast.py:run_self_play()
def run_self_play(env, model, params, ..., discount=0.997):  # Default 0.997
    ...
    mc_return = compute_mc_returns(rewards, discount)  # Uses 0.997

# trainer.py:MuZeroTrainer.__init__()
class MuZeroTrainer:
    def __init__(self, ..., discount: float = 0.99):  # Default 0.99
        self.discount = discount
```

### Impact

- Self-play generates trajectories with `discount=0.997`
- Trainer may expect `discount=0.99` for loss calculation
- Inconsistent temporal discounting affects value estimation
- Bootstrap targets computed with wrong discount

### The Math Problem

```python
# Self-play computes MC returns with 0.997:
MC_target = r_t + 0.997*r_{t+1} + 0.997^2*r_{t+2} + ...

# Trainer bootstraps with 0.99:
TD_target = r_t + 0.99 * V(s_{t+1})

# These targets have different scales!
```

## Solution

### Step 1: Standardize on one value

Pick one discount factor (recommend 0.99 for trading - higher discount = more myopic):

```python
# train_jax_fast.py - argparse
parser.add_argument('--discount', type=float, default=0.99)
```

### Step 2: Wire consistently

```python
# train_jax_fast.py
trainer = MuZeroTrainer(..., discount=args.discount)
trajectory = run_self_play(..., discount=args.discount)
trajectory = run_self_play_batched(..., discount=args.discount)
```

### Step 3: Remove hardcoded defaults

```python
# Make discount required parameter (no default)
def run_self_play(env, model, params, ..., discount):  # No default
```

## Files to Modify

| File | Change |
|------|--------|
| `train_jax_fast.py` | Wire args.discount to all components |
| `trainer.py` | Ensure discount is used consistently |

## Testing

1. Print discount values at runtime to verify consistency
2. Check MC returns match expected discounting
3. Verify value targets use same discount

## Notes

Higher discount (0.997) = values future rewards more = longer horizon planning
Lower discount (0.99) = more myopic = faster learning but shorter horizon

For trading, 0.99 is reasonable (1440 steps/day, effective horizon ~100 steps).
