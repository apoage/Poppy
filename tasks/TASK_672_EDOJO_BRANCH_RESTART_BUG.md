# TASK_672: e-dojo Branch Restart Bug (CRITICAL)

**Status**: FIXED
**Priority**: CRITICAL
**Created**: 2026-02-07
**Fixed**: 2026-02-07

## Root Cause (CONFIRMED)

**Two bugs found:**

1. **Step counting was broken** - counting poll iterations (every 30s) instead of actual training steps.
   Fixed by computing steps from checkpoint epochs: `steps = (current_epoch - epoch_at_merge) × 400`

2. **CLI flag was correct** - e-dojo uses `train_easy.py` which expects `--checkpoint` (not `--load-checkpoint` like `train_jax_fast.py`).

Fixed in `tools/orchestration/e-dojo_v1.py`.

## Problem

Branches are NOT restarting after merge to load merged checkpoint. Evidence:

```
Merge Timeline (cosine similarity DROPPING instead of staying high):
1 fedavg 0.719
2 fedavg 0.503
3 fedavg 0.403
4 fedavg 0.340
5 fedavg 0.299

Agreement also very low:
A Agreement: 2.2%
B Agreement: 16.4%
```

**Expected behavior**: After merge, both branches load merged checkpoint → start from same point → cosine should stay ~1.0 or slowly decrease then recover at next merge.

**Actual behavior**: Cosine drops monotonically → branches never converge → federated learning is broken.

## Root Cause Hypotheses

1. **Launcher not attached** - `self.launcher is None` so restart never called
2. **Restart called but fails silently** - exception swallowed somewhere
3. **Restart happens but checkpoint not actually loaded** - train_jax_fast ignores the checkpoint arg
4. **Restart happens but branches diverge extremely fast** - unlikely given magnitude of drop

## Code Locations

- `tools/orchestration/e-dojo_v1.py:514-517` - restart call after merge
- `tools/orchestration/e-dojo_v1.py:837-889` - `restart_both_with_checkpoint()` implementation
- `tools/orchestration/e-dojo_v1.py:1074-1076` - launcher creation (only when `--run`)

## Investigation Plan

1. Add explicit logging before/after restart call
2. Check if launcher is None at merge time
3. Verify train_jax_fast actually loads the checkpoint passed via --load-checkpoint
4. Check for swallowed exceptions in restart logic
5. Verify PIDs actually change after merge

## Fix Requirements

- Both branches MUST load merged checkpoint after every merge
- Add visible confirmation: `[DOJO] Branch A restarted with merged checkpoint (PID: old→new)`
- Add fallback: if restart fails, STOP training (don't silently continue diverging)

## Testing

After fix, cosine similarity should:
- Jump back up after each merge (branches starting from same point)
- Stay high (0.8+) between merges if training is stable
- Never monotonically decrease across 5+ merges
