# Dynamic Curriculum System

**Created**: Dec 20, 2025
**Status**: ✅ Implemented

---

## New Rules

**Old system**: Train fixed epochs → check once → advance/retry
**New system**: Check every 250 epochs → adjust difficulty dynamically

### Progressive Difficulty Phases

Each level has 5 progressive phases:

| Phase | Fee (bps) | Slippage (bps) | Threshold | Difficulty |
|-------|-----------|----------------|-----------|------------|
| **Phase 1** | 0 | 0 | +2.5% | Training wheels (no friction) |
| **Phase 2** | 5 | 0 | +2.0% | Tiny fee (learning costs) |
| **Phase 3** | 10 | 2 | +1.5% | Light costs |
| **Phase 4** | 16 | 5 | +1.0% | Realistic Kraken costs |
| **Phase 5** | 26 | 5 | +0.5% | Full costs (high-fee tier) |

---

## How It Works

### Training Flow

```
Start Level 0
  ↓
Phase 1 (0 bps fee, 0 bps slip)
  ↓
Train 250 epochs
  ↓
Check mastery: avg_reward ≥ +2.5%?
  ├─ YES → Advance to Phase 2 ✅
  └─ NO → Train another 250 epochs (same phase) ❌

Phase 2 (5 bps fee, 0 bps slip)
  ↓
Train 250 epochs
  ↓
Check mastery: avg_reward ≥ +2.0%?
  ├─ YES → Advance to Phase 3 ✅
  └─ NO → Train another 250 epochs (same phase) ❌

... (continue through Phase 5)

Phase 5 PASSED → Level 0 COMPLETE 🎓
```

### Adaptive Difficulty

**If agent is smart**:
- Passes each phase in 250 epochs
- Total: 5 phases × 250 = **1250 epochs** (fast!)

**If agent struggles**:
- Phase 1: 250 epochs → fail → +250 → pass (500 total)
- Phase 2: 250 epochs → fail → +250 → pass (500 total)
- Total: **2500 epochs** (but learns thoroughly)

**Safety limit**: 2000 epochs max per level

---

## Example Session

```bash
python dojo_dynamic_curriculum.py --level 0
```

**Output**:
```
============================================================
DYNAMIC TRAINING - Level 0 (sine)
============================================================

🎯 Phase 1
   Fee: 0 bps | Slippage: 0 bps
   Pass Threshold: +2.5%

🚀 Training: python train_jax_fast.py --synthetic-level 0 --epochs 250 --fee 0 --slippage 0 ...
(250 epochs run)

────────────────────────────────────────────────────────────
MASTERY CHECK
────────────────────────────────────────────────────────────
Recent Avg Reward: +3.12%
Pass Threshold:    +2.50%
Epochs Evaluated:  250
Result:            ✅ PASS
────────────────────────────────────────────────────────────

✅ Phase 1 PASSED!
   Advancing to next difficulty phase...

🎯 Phase 2
   Fee: 5 bps | Slippage: 0 bps
   Pass Threshold: +2.0%

📂 Resuming from checkpoint (after 250 epochs)
🚀 Training: python train_jax_fast.py --synthetic-level 0 --epochs 250 --fee 5 --slippage 0 --load-checkpoint jax_ckpt_fast/dojo_dynamic_L0_sine ...
(250 more epochs)

────────────────────────────────────────────────────────────
MASTERY CHECK
────────────────────────────────────────────────────────────
Recent Avg Reward: +2.35%
Pass Threshold:    +2.00%
Epochs Evaluated:  250
Result:            ✅ PASS
────────────────────────────────────────────────────────────

✅ Phase 2 PASSED!
   Advancing to next difficulty phase...

... (continues through Phase 5)

🎓 LEVEL COMPLETE! All phases mastered!
   Total epochs: 1250
```

---

## Benefits

### 1. Progressive Learning
- Start easy (no fees) → end hard (full costs)
- Agent learns strategy before learning optimization
- Each phase builds on previous

### 2. Adaptive to Agent Speed
- Fast learners: 250 epochs per phase (1250 total)
- Slow learners: 500+ epochs per phase (2500+ total)
- No wasted time if agent masters quickly

### 3. Early Detection of Issues
- Check every 250 epochs (not 500+)
- Catch problems early
- Adjust difficulty before investing too many epochs

### 4. Smooth Difficulty Curve
- 5 small steps instead of 1 big jump
- 0 bps → 5 bps → 10 bps → 16 bps → 26 bps
- Easier to adapt than 0 → 26 bps jump

---

## Comparison: Old vs New

### Old System (Static)
```
Level 0:
- Train 500 epochs with 0 bps fee
- Check once: ≥ +2.0%?
- If fail: +250 epochs (same difficulty)
- If pass: Advance to Level 1 (16 bps fee)

Problem: Big jump from 0 to 16 bps
```

### New System (Dynamic)
```
Level 0:
- Phase 1: Train 250 epochs @ 0 bps → pass? → advance
- Phase 2: Train 250 epochs @ 5 bps → pass? → advance
- Phase 3: Train 250 epochs @ 10 bps → pass? → advance
- Phase 4: Train 250 epochs @ 16 bps → pass? → advance
- Phase 5: Train 250 epochs @ 26 bps → pass? → complete

Benefit: Smooth 5-step progression
```

---

## Configuration

### Difficulty Phases (Customizable)

In `dojo_dynamic_curriculum.py`:
```python
self.difficulty_phases = [
    {'name': 'Phase 1', 'fee': 0, 'slippage': 0, 'threshold': 2.5},
    {'name': 'Phase 2', 'fee': 5, 'slippage': 0, 'threshold': 2.0},
    {'name': 'Phase 3', 'fee': 10, 'slippage': 2, 'threshold': 1.5},
    {'name': 'Phase 4', 'fee': 16, 'slippage': 5, 'threshold': 1.0},
    {'name': 'Phase 5', 'fee': 26, 'slippage': 5, 'threshold': 0.5},
]
```

**Customize**:
- Add more phases (smoother curve)
- Remove phases (faster progression)
- Adjust thresholds (easier/harder)

---

## Usage

### Train Level 0
```bash
python dojo_dynamic_curriculum.py --level 0
```

### Train Level 3 (with swaps)
```bash
python dojo_dynamic_curriculum.py --level 3
```

### Monitor Progress
```bash
# Watch logs
tail -f logs/dojo_dynamic.log

# Check current checkpoint
ls -lh jax_ckpt_fast/dojo_dynamic_L0_sine/

# View metrics
cat last_run_metrics.json | jq '.recent_avg_reward'
```

---

## Advanced: Multi-Level Curriculum

Run all levels sequentially:

```bash
# Level 0 (sine waves)
python dojo_dynamic_curriculum.py --level 0

# Level 1 (lagged pairs)
python dojo_dynamic_curriculum.py --level 1

# Level 2 (dip crashes)
python dojo_dynamic_curriculum.py --level 2

# Level 3 (noisy trends + swaps)
python dojo_dynamic_curriculum.py --level 3
```

**Or create a wrapper script**:
```bash
#!/bin/bash
for level in 0 1 2 3; do
    echo "Training Level $level..."
    python dojo_dynamic_curriculum.py --level $level
    if [ $? -ne 0 ]; then
        echo "Level $level failed!"
        exit 1
    fi
done
echo "All levels complete!"
```

---

## Tuning Guide

### If Phases Too Easy
```python
# Increase thresholds
{'name': 'Phase 1', 'fee': 0, 'slippage': 0, 'threshold': 3.5},  # Was 2.5
```

### If Phases Too Hard
```python
# Decrease thresholds or reduce fees
{'name': 'Phase 2', 'fee': 3, 'slippage': 0, 'threshold': 1.5},  # Was 5 bps, 2.0%
```

### Add Intermediate Phases
```python
self.difficulty_phases = [
    {'name': 'Phase 1', 'fee': 0, 'slippage': 0, 'threshold': 2.5},
    {'name': 'Phase 1.5', 'fee': 2, 'slippage': 0, 'threshold': 2.2},  # NEW
    {'name': 'Phase 2', 'fee': 5, 'slippage': 0, 'threshold': 2.0},
    ...
]
```

---

## Expected Training Times

**Fast agent** (passes each phase in 250 epochs):
- 5 phases × 250 epochs = 1250 epochs
- ~5.5 sec/epoch × 1250 = ~115 minutes (~2 hours)

**Average agent** (needs 375 epochs per phase):
- 5 phases × 375 epochs = 1875 epochs
- ~5.5 sec/epoch × 1875 = ~172 minutes (~3 hours)

**Slow agent** (needs 500 epochs per phase):
- 5 phases × 500 epochs = 2500 epochs (max budget)
- ~5.5 sec/epoch × 2500 = ~229 minutes (~4 hours)

---

## Summary

**New rules**: ✅ Check every 250 epochs → adjust difficulty dynamically

**Phases**: 5 progressive steps (0 bps → 26 bps fees)

**Benefits**:
- Adaptive to agent speed
- Smooth difficulty curve
- Early problem detection

**Usage**: `python dojo_dynamic_curriculum.py --level 0`

**Status**: Ready to test! 🚀
