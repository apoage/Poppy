# TASK_687: Hard Example Mining for Trade Accuracy

**Status**: DONE
**Priority**: 4 (after architectural fixes)
**Source**: Analysis 004 (GLM) — "Focus on the Margin"

## Problem

96.1% of timestep×asset targets are HOLD, only 3.9% are trades. Current mitigation:
- Adaptive trade weighting: ~24× (helps gradient magnitude, not sample frequency)
- Trade-biased sampling: 15% of windows from trade-heavy regions (TASK_680)

But the model still sees trades only ~4% of the time per batch. The adaptive weight makes trade gradients large but doesn't increase the diversity of trade examples the model sees.

## Proposal

Increase trade example frequency in each batch to 30-50% by:

### Option A: Increase trade_sample_prob (simplest)

```python
# easy_config.py
trade_sample_prob: float = 0.30  # Was 0.15
```

Already implemented in TASK_680. Just a config change.

### Option B: Oracle-aware batch construction (more effective)

After oracle computation, before batch sampling:
1. Identify timestep×asset pairs where oracle says TRADE
2. Ensure each batch contains at least 30% trade windows
3. Implemented as weighted sampling in `OnTheFlyGenerator._sample_batch_from_buffer()`

### Option C: Loss-aware replay (most complex)

Track per-sample loss. Re-sample high-loss windows more frequently. Similar to Prioritized Experience Replay (PER) from DQN literature.

## Recommendation

**Option A first** (single config change), then Option B if insufficient.

Option C is overkill — PER adds complexity and the gains are marginal when the fundamental issue is representation quality (TASK_683).

## Risk

- If too aggressive (>50% trade samples), model may forget hold patterns (catastrophic forgetting)
- Validation must use uniform distribution (not biased) to measure true performance

## Verification

1. trade_sample_prob=0.30: monitor agree_trade and agree_hold
2. If agree_hold drops >5pp while agree_trade rises, reduce to 0.25
3. Can test independently from TASK_683 (uses current architecture)
