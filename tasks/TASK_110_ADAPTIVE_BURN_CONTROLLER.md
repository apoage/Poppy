# TASK_110: Adaptive Burn Controller

**Status:** COMPLETED
**Created:** 2026-01-24
**Category:** Training / Adaptive Difficulty

## Problem

Model tends to HOLD excessively (75%+ of actions) because:
- Holding is "safe" - no transaction costs
- Burning money at fixed rate isn't enough pressure
- Need escalating consequences for persistent holding

## Solution

Created `AdaptiveBurnController` that monitors hold rate over consecutive epochs and progressively escalates burn pressure when the model refuses to trade.

## Implementation

### New File: `mcts_trader/jax_agent/adaptive_burn.py`

```python
class AdaptiveBurnController:
    """
    Escalates burn pressure when hold rate stays above threshold.

    Args:
        hold_threshold: Hold rate triggering escalation (default: 0.60)
        patience: Consecutive epochs before stage change (default: 5)
        max_stage: Maximum burn stage (default: 4 = hardcore)
    """
```

### Escalation Stages

| Stage | Name | Base Rate | Random Burn | Random % | Liquidation |
|-------|------|-----------|-------------|----------|-------------|
| 0 | base | 0.005%/step | No | 0% | No |
| 1 | market | 0.005%/step | Yes | 3% | No |
| 2 | random | 0.010%/step | Yes | 7% | No |
| 3 | aggressive | 0.020%/step | Yes | 10% | No |
| 4 | hardcore | 0.030%/step | Yes | 15% | 2% |

### Trigger Logic

```
if hold_rate > threshold for patience epochs:
    stage += 1  # Escalate

if hold_rate < threshold for patience epochs:
    stage -= 1  # De-escalate
```

### Per-Episode Impact (400 steps, 75% hold)

| Stage | Base Burn | Random Events | Total Cash Drain |
|-------|-----------|---------------|------------------|
| 0 | ~1.5% | 0 | ~1.5% |
| 1 | ~1.5% | ~12 events | ~3% |
| 2 | ~3% | ~28 events | ~6% |
| 3 | ~6% | ~40 events | ~10% |
| 4 | ~9% | ~60 events + liquidation | ~15%+ |

## CLI Flags

```bash
--adaptive-burn                    # Enable adaptive burn escalation
--burn-hold-threshold 0.60         # Hold % that triggers escalation (default: 60%)
--burn-patience 5                  # Consecutive epochs before stage change (default: 5)
```

## Usage

### Standalone Training

```bash
python train_jax_fast.py \
    --synthetic-level 12 \
    --burning-money \
    --adaptive-burn \
    --burn-hold-threshold 0.60 \
    --burn-patience 5
```

### Dojo V4 Phase 2

Added to `_train_synthetic_adaptive()` in dojo_v4_full_curriculum.py:

```python
# Burning money with ADAPTIVE ESCALATION (Phase 2 hardcore mode)
if burning_money:
    cmd.extend([
        '--burning-money',
        '--adaptive-burn',
        '--burn-hold-threshold', '0.60',
        '--burn-patience', '5',
    ])
```

Now when Phase 2 enables burning_money (based on hold rate > 50%), it automatically gets adaptive escalation.

## Files Modified

| File | Changes |
|------|---------|
| `mcts_trader/jax_agent/adaptive_burn.py` | NEW - AdaptiveBurnController class |
| `mcts_trader/training/cli.py` | +3 CLI flags |
| `mcts_trader/training/config.py` | +3 config fields |
| `train_jax_fast.py` | Import + initialization + update loop |
| `tools/orchestration/dojo_v4_full_curriculum.py` | Phase 2 integration |

## Testing

```bash
# Test escalation
python train_jax_fast.py \
    --synthetic-level 0 \
    --epochs 50 \
    --burning-money \
    --adaptive-burn \
    --burn-patience 3 \
    --name test_adaptive_burn
```

Watch for `[BURN] ESCALATING` messages when hold rate stays high.

## Related Tasks

- TASK_051: Original burning money implementation
- TASK_058: Adaptive noise controller (similar pattern)
- TASK_068: Zombie flag restoration

## Future Enhancements

- [ ] Add actual position liquidation in environment (currently only cash burn)
- [ ] Different thresholds per curriculum level
- [ ] Integration with adaptive fees for combined difficulty scaling
