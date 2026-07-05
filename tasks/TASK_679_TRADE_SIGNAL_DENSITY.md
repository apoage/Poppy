# TASK_679: Trade Signal Density — Break 52.5% Agreement Plateau

**Status**: In Progress
**Priority**: Critical (blocking all training progress)
**Depends on**: TASK_674, TASK_675, TASK_677 (all applied)

## Problem

Training plateaued at 52.5% balanced agreement after 618 epochs despite all gradient fixes.
CSV analysis reveals the root cause:

| Metric | Value |
|--------|-------|
| oracle_hold_ratio | 98.9% |
| agree_hold | 95.8% |
| **agree_trade** | **9.4%** (below 20% random baseline) |
| adaptive_trade_weight | 64.0 (max, working) |
| best_agreement | 53.4% (peaked, flat for hundreds of epochs) |

TASK_675 fixed gradient **magnitude** (trades get equal gradient mass). But the **density**
problem remains: with k=1 on 64 assets, only ~0.7 trades/step → 1.1% trade density.
The model can't learn trade patterns from ~700 examples per 65,536 positions per batch.

## Phase 1: Lower `min_alpha_threshold` (DONE)

`easy_config.py:58`: 0.001 → 0.0005

```
Old: threshold_per_step = 0.001 × 20 = 0.02 (2% bar) → ~1.1% trades
New: threshold_per_step = 0.0005 × 20 = 0.01 (1% bar) → ~2-3% trades (est.)
```

Still requires 1% absolute cumulative return over 20 steps — real signal, not noise.
Expected: 2-3× more trade examples, enough for model to start distinguishing contexts.

## Phase 2: Trade-Biased Sampling (if needed)

If threshold alone doesn't break the plateau (expected: gets to ~55-60% then stalls),
implement weighted timestep sampling in `OnTheFlyGenerator.generate_batch()`:

```
P(timestep t) ∝ 1.0 + trade_boost × has_trade(t)
```

- **Index phase** (`_refill_buffer`): scan oracle actions per episode, build sparse index
  of timesteps with trade signals. Cost: ~0.1ms per episode.
- **Sampling phase** (`generate_batch`): weighted random offset selection favoring trade
  windows. `trade_boost=5.0` → trade timesteps 6× more likely to be sampled.
- No oracle quality change — same labels, different presentation frequency.
- ~30 LOC, toggleable via config flag.

Combined with Phase 1: ~6-10× more trade gradient signal vs original baseline.

### Interaction with other tasks

- **TASK_678 (GMR)**: Complementary. GMR cleans label quality, sampling increases frequency.
  Different axes of the same problem.
- **TASK_676 (Streaming encoder)**: Independent. Encoder architecture doesn't affect
  sampling strategy.
- **TASK_675 (Gradient balance)**: Stacks. Weight balance × density = both needed.

## Success Criteria

- agree_trade > 30% (currently 9.4%)
- balanced_agreement > 60% (currently 52.5%)
- Improvement visible within first 100 epochs of restarted training

## Testing Requirements

**Test every function and feature immediately after implementation.**

### Test 1: Threshold validation (Phase 1)
```python
# After first epoch with new threshold, verify trade density improved
# Log oracle_hold_ratio — should drop from 98.9% to ~96-97%
metrics = trainer.train_step(batch)
print(f"oracle_hold_ratio: {metrics['oracle_hold_ratio']:.3f}")
assert metrics['oracle_hold_ratio'] < 0.98, \
    f"Hold ratio still {metrics['oracle_hold_ratio']:.3f}, threshold may not be effective"
```

### Test 2: Trade accuracy tracking
```python
# After 50 epochs, agree_trade should be above random (20%)
# If still < 15% after 100 epochs → threshold needs further lowering or Phase 2
assert metrics['agree_trade'] > 0.15, \
    f"agree_trade={metrics['agree_trade']:.3f} still below random after 50 epochs"
```

### Test 3: No regression on hold accuracy
```python
# agree_hold should stay above 90% (was 95.8%)
assert metrics['agree_hold'] > 0.90, \
    f"agree_hold regressed to {metrics['agree_hold']:.3f}"
```

---

## Files Modified

- `easy_config.py` — min_alpha_threshold 0.001 → 0.0005
- `train_easy.py` — (Phase 2: OnTheFlyGenerator sampling, if needed)
