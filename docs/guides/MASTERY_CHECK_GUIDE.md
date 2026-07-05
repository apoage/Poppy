# Dojo Mastery Check System

**Added**: Dec 19, 2025
**Status**: ✅ Implemented

---

## Overview

The Dojo curriculum now includes **mastery checks** - agents must pass each level before advancing to the next.

### How It Works

1. **Train**: Agent trains for specified epochs (e.g., 500 for Level 0)
2. **Check**: System evaluates recent performance (last 50 epochs average)
3. **Decision**:
   - ✅ **Pass** (reward ≥ threshold) → Advance to next level
   - ❌ **Fail** (reward < threshold) → Add retry epochs and retrain

### Retry Logic

If agent fails mastery check:
- Add `retry_epochs` (250-500) to current checkpoint
- Resume training from last checkpoint
- Re-evaluate after retry epochs
- Max retries: 2 (configurable)

**Example**:
```
Level 0: Train 500 epochs → Avg reward +0.8% < +1.5% threshold
         ❌ FAIL - Retry with +250 epochs
         Train 250 more (750 total) → Avg reward +1.6% ≥ +1.5%
         ✅ PASS - Advance to Level 1
```

---

## Pass Thresholds

| Level | Generator | Threshold | Reasoning |
|-------|-----------|-----------|-----------|
| 0 | Sine waves | **+1.5%** | Deterministic, easy - should profit well |
| 1 | Lagged pairs | **+1.0%** | Correlation trading - moderate |
| 2 | Dip crashes | **+0.5%** | Punishes impatience - harder |
| 3 | Noisy trends | **+0.2%** | Realistic noise - hardest |

**Why decreasing thresholds?**
- Later levels are harder (more realistic)
- +0.2% on noisy data > +1.5% on perfect sine waves
- Goal: Progressive difficulty, not absolute reward

---

## Configuration

### Per-Level Config

```json
{
  "level": 0,
  "generator": "sine",
  "tactics": ["basic"],
  "epochs": 500,
  "pass_threshold": 1.5,     // Required avg reward (%)
  "retry_epochs": 250,       // Extra epochs if failed
  "max_retries": 2,          // Max retry attempts
  "n_assets": 64,
  "n_steps": 10000
}
```

### Retry Budget

**Level 0**:
- Initial: 500 epochs
- Retry 1: +250 epochs (750 total)
- Retry 2: +250 epochs (1000 total)
- **Max budget**: 1000 epochs

**Level 2** (harder):
- Initial: 1500 epochs
- Retry 1: +400 epochs (1900 total)
- Retry 2: +400 epochs (2300 total)
- **Max budget**: 2300 epochs

---

## Mastery Check Algorithm

```python
def _check_level_mastery(level_idx, config, metrics_file):
    # Load metrics
    metrics = json.load(metrics_file)
    episode_rewards = metrics['episode_rewards']

    # Calculate recent average (last 50 epochs)
    recent_avg = mean(episode_rewards[-50:])

    # Get threshold (from config or defaults)
    threshold = config.get('pass_threshold', defaults[level_idx])

    # Pass/fail
    passed = recent_avg >= threshold

    return passed
```

**Metrics used**:
- `episode_rewards`: Full reward history
- `recent_avg_reward`: Last 50 epochs average (key metric)
- `total_epochs`: Total training time

---

## Training Flow

### Without Mastery Checks (Old)
```
Level 0: 500 epochs → Done (regardless of performance)
Level 1: 1000 epochs → Done (regardless of performance)
```

### With Mastery Checks (New)
```
Level 0: 500 epochs → Check → Pass → Level 1
                            → Fail → +250 epochs → Check → Pass → Level 1
                                                        → Fail → +250 epochs → Check → Pass → Level 1
                                                                                   → Fail → ABORT
```

**Progression is gated** - can't skip to Level 2 without passing Level 1.

---

## Example Session

```bash
# Start training
python dojo_manager.py --level 0

# Output:
# [DOJO] Training Level 0 (sine)
# [DOJO] Config: 500 epochs, 64 assets, 10000 steps
# ...
# (500 epochs later)
# ============================================================
# MASTERY CHECK - Level 0
# ============================================================
# Recent Avg Reward (last 50 epochs): +0.77%
# Pass Threshold:                      +1.50%
# Total Epochs Completed:              500
# Result: ❌ FAIL
# ============================================================
#
# [DOJO] ❌ Level 0 did not pass mastery check
# [DOJO] Will retry with 250 additional epochs
# [DOJO] 🔄 RETRY 1/2: Adding 250 more epochs
# [DOJO] Resuming from checkpoint: jax_ckpt_fast/dojo_level_0_sine
# ...
# (250 more epochs)
# ============================================================
# MASTERY CHECK - Level 0
# ============================================================
# Recent Avg Reward (last 50 epochs): +1.68%
# Pass Threshold:                      +1.50%
# Total Epochs Completed:              750
# Result: ✅ PASS
# ============================================================
#
# [DOJO] ✅ Level 0 PASSED in 42.3 minutes (750 total epochs)
```

---

## Monitoring

### Check Current Progress

```bash
# View mastery check results
tail -f logs/dojo_training.log

# Check saved metrics
cat last_run_metrics.json | jq '.recent_avg_reward'
```

### Curriculum Progress

```bash
cat dojo_checkpoints/curriculum_progress.json
```

Example:
```json
{
  "current_level": 0,
  "completed_levels": 1,
  "total_levels": 4,
  "timestamp": 1734665123
}
```

---

## Tuning Thresholds

If levels are **too easy** (always pass on first try):
- Increase `pass_threshold` (e.g., Level 0: 1.5 → 2.0)

If levels are **too hard** (always fail max retries):
- Decrease `pass_threshold` (e.g., Level 2: 0.5 → 0.3)
- OR increase `retry_epochs` (250 → 500)
- OR increase `max_retries` (2 → 3)

### Custom Curriculum

Create `config/custom_curriculum.json`:
```json
[
  {
    "level": 0,
    "pass_threshold": 2.0,     // Harder threshold
    "retry_epochs": 500,        // More retry budget
    "max_retries": 3,           // More attempts
    ...
  }
]
```

Run with:
```bash
python dojo_manager.py --curriculum config/custom_curriculum.json
```

---

## Benefits

**Ensures learning**:
- Can't progress without demonstrating mastery
- Prevents "lucky passes" (single good episode)
- 50-epoch average = statistically significant

**Adaptive training**:
- Smart agents pass quickly (500 epochs)
- Struggling agents get more time (750-1000 epochs)
- Automatic difficulty adjustment

**Prevents wasted training**:
- Don't advance to Level 2 if Level 1 not mastered
- Higher levels assume lower-level skills learned

**Interpretability**:
- Clear pass/fail criteria
- Transparent metrics (logged to console + file)
- Reproducible (seed-based synthetic data)

---

## Technical Details

### Files Modified

**train_jax_fast.py**:
- Added `episode_rewards` to metrics save
- Added `recent_avg_reward` calculation
- Exports full reward history to `last_run_metrics.json`

**dojo_manager.py**:
- Added `_check_level_mastery()` method
- Added retry loop in `_train_level()`
- Added checkpoint resumption for retries
- Updated default curriculum with thresholds

### Metrics Format

`last_run_metrics.json`:
```json
{
  "avg_reward": 0.77,
  "final_loss": 4.7231,
  "episode_rewards": [3.2, -1.5, 0.8, ...],  // Full history
  "recent_avg_reward": 0.77,                 // Last 50 avg
  "total_epochs": 500,
  "experiment_name": "dojo_level_0_sine"
}
```

---

## Summary

**Status**: ✅ Mastery checks implemented
**Default thresholds**: 1.5% → 1.0% → 0.5% → 0.2% (Levels 0-3)
**Retry budget**: 250-500 epochs per retry, 2 max retries
**CLI**: No changes - `python dojo_manager.py` uses mastery checks automatically

**Next run**: Current Level 0 training will check mastery after 500 epochs!
