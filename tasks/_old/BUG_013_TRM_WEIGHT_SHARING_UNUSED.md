# Bug: TRM Weight Sharing Flag Unused

## Priority: MEDIUM
**Status:** FALSE POSITIVE

## Verification (2026-01-01)

The `share_weights` flag IS implemented and used:
- Line 70: Conditional in setup()
- Line 138: Conditional in latent refinement
- Line 163: Conditional in output refinement

The flag correctly switches between shared and per-cycle weights.
**Component:** `mcts_trader/jax_agent/trm_flax.py`

## Problem Description

The `share_weights` flag exists in the TRM model but the implementation doesn't actually share weights across cycles. Each cycle has independent weights regardless of the flag.

### Location

```python
# trm_flax.py
class TRMReasoning(nn.Module):
    share_weights: bool = False  # Flag exists

    def setup(self):
        # But weights are created per-cycle regardless
        for h in range(self.h_cycles):
            for l in range(self.l_cycles):
                # Each cycle gets own weights
                self.latent_refiner_h{h}_l{l} = nn.Dense(...)
```

### Paper Reference

The TRM paper (Tiny Recursive Models) specifies weight sharing across cycles to reduce parameter count. Without it, model is larger than intended.

### Impact

- Model larger than paper specification
- Higher memory usage
- Different behavior than paper describes
- `share_weights=True` has no effect

## Solution

Implement actual weight sharing:

```python
# trm_flax.py
class TRMReasoning(nn.Module):
    share_weights: bool = False

    def setup(self):
        if self.share_weights:
            # Single set of weights shared across cycles
            self.shared_latent_refiner = nn.Dense(self.state_dim)
            self.shared_output_refiner = nn.Dense(self.state_dim)
        else:
            # Per-cycle weights (current behavior)
            ...

    def __call__(self, x):
        for h in range(self.h_cycles):
            for l in range(self.l_cycles):
                if self.share_weights:
                    x = self.shared_latent_refiner(x)
                else:
                    x = getattr(self, f'latent_refiner_h{h}_l{l}')(x)
```

## Files to Modify

| File | Change |
|------|--------|
| `trm_flax.py` | Implement actual weight sharing when flag is True |

## Testing

1. Count parameters with `share_weights=True` vs `False`
2. Verify shared weights model is smaller
3. Test training with shared weights

## Notes

May want to keep non-shared as default since it's been working. Shared weights is an option for smaller model.
