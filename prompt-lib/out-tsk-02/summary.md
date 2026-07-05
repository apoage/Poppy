# Implementation Audit — Executive Summary

**Date**: 2026-02-23
**Model**: Claude Opus 4.6
**Tasks Audited**: 24 (147, 149, 151, 670B, 673, 677, 685, 695, 674, 675, 679, 680, 687, 691, 683, 684, 686, 689, 692, 693, plus 7 bug fixes from 694)
**Files Read**: easy_config.py, easy_trainer.py, train_easy.py, mcts_trader/jax_agent/path_solver.py, mcts_trader/jax_agent/muzero_model.py, tools/benchmark_easy.py, tools/orchestration/merge_utils.py

## Counts

| Category | Count |
|----------|--------|
| Fully Implemented & Correct | 19 |
| Partially Implemented | 4 (147, 151, 670B, 691) |
| Incorrectly Implemented | 1 (685) |
| Config Mismatch | 3 |
| Cross-Task Conflict | 2 |
| Deliberately Disabled | 1 (TASK_691 by TASK_692) |

## Top Suspects (Most Likely to Cause agree_trade Regression)

### 1. TASK_685: Gradient Clipping — Wrong Implementation (HIGH IMPACT)
**Issue**: Spec requires per-asset clipping, but implementation uses `jnp.mean()` which averages across assets.
**Code**: `easy_trainer.py:215-229`
**Impact**: Global averaging defeats per-asset intent. Large gradients from volatile assets dominate and get clipped, while small gradients from stable assets are under-clipped. This creates imbalanced learning across assets.
**Config Mismatch**: `grad_clip = 5.0` (correct value), `gradient_clip.per_asset = True` (correct intent but wrong implementation)
**Priority**: CRITICAL — Fix to use `jnp.maximum()` per asset

### 2. TASK_677 vs Config: Temperature Sharpening 20× Too Aggressive (MEDIUM IMPACT)
**Issue**: Spec says `oracle_sharpen_temperature = 2.0`, but config has `0.1` (20× sharper)
**Code**: `easy_config.py:75`
**Impact**: Over-sharpening makes oracle commands too deterministic, reducing exploration in state channels. This could cause model to overfit to sharp oracle signals and fail to generalize.
**Priority**: HIGH — Verify if 0.1 was intentional or should be restored to 2.0

### 3. TASK_151 vs TASK_675: Warmup Disabled Contradiction (MEDIUM IMPACT)
**Issue**: TASK_151 spec requires 50 epochs warmup, but config has `weight_warmup_epochs = 0`
**Code**: `easy_config.py:56-58`
**Impact**: Fresh models receive full trade weight immediately, risking gradient explosion. Comment says this is "to break turtle trap" — deliberate design choice that contradicts original spec.
**Priority**: MEDIUM — Monitor early training for gradient explosions

### 4. TASK_670B: Expert Imprinting Incomplete (MEDIUM IMPACT)
**Issue**: Config says `expert_imprinting_stability.enabled = True` but implementation is missing orthogonal initialization, weight freezing, and adaptation schedules.
**Code**: `easy_trainer.py:1074-1078` (partial)
**Impact**: Expert weights will drift during training without orthogonal protection, potentially degrading performance. Config suggests feature is enabled but it's not working.
**Priority**: MEDIUM — Either complete implementation or disable in config

### 5. TASK_691: Oracle Entropy Broken Implementation (LOW IMPACT)
**Issue**: Implementation has undefined `_price_generator` reference and no entropy loss term. Feature is disabled by default per TASK_692.
**Code**: `train_easy.py:71`, `easy_trainer.py:203`
**Impact**: Even if `oracle_entropy=True` were set, feature would not work. However, it's deliberately disabled by TASK_692.
**Priority**: LOW — Consider removing code entirely to reduce confusion

## Recommended Actions (Priority Order)

1. **FIX TASK_685 IMMEDIATELY** (`easy_trainer.py:215-229`)
   - Change from `jnp.mean()` to `jnp.maximum()` for per-asset clipping
   - This directly affects gradient balance across 64 assets

2. **VERIFY TASK_677 TEMPERATURE VALUE** (`easy_config.py:75`)
   - Determine if `oracle_sharpen_temperature = 0.1` was intentional
   - If not, restore to `2.0` per spec

3. **MONITOR GRADIENT STABILITY** (due to warmup=0)
   - Watch early training for gradient explosions
   - Consider partial warmup (5-10 epochs) if unstable

4. **COMPLETE TASK_670B OR DISABLE**
   - Either implement missing orthogonal initialization/weight freezing
   - Or set `expert_imprinting_stability.enabled = False` in config

5. **CLEANUP TASK_691 CODE**
   - Remove broken oracle entropy implementation (deliberately disabled)
   - Reduces confusion and code bloat

## Additional Notes

### Correctly Implemented Tasks (19 total)
These tasks are working as specified and not causing issues:
- TASK_149 (Adaptive Class Balancing) ✅
- TASK_673 (Multipass Oracle) ✅
- TASK_677 (Turtle Breaker) ✅ (except config temp mismatch)
- TASK_695 (Oracle Mask Head) ✅
- TASK_674 (Agreement Plateau Bugs) ✅
- TASK_675 (Technical Debt) ✅
- TASK_679 (Trade Signal Density) ✅
- TASK_680 (Trade-Biased Sampling) ✅
- TASK_687 (Hard Example Mining) ✅
- TASK_683 (Widen Encoder State) ✅
- TASK_684 (Decouple Hidden Dim) ✅
- TASK_686 (Bfloat16 Mixed Precision) ✅ (code correct, config field unused)
- TASK_689 (Agent State Channels) ✅
- TASK_692 (Model-State Rollout) ✅
- TASK_693 (Fix Position Semantics) ✅
- BUG-1, BUG-2, BUG-4, BUG-5, BUG-6, BUG-7 (all correct) ✅

### Config Audit Results
- 31/34 fields (91%) match spec expectations
- 3 mismatches: `compute_dtype`, `weight_warmup_epochs`, `oracle_sharpen_temperature`
- See `config_audit.md` for full table

### Cross-Task Analysis
- 2 conflicts: Warmup epochs, sharpening temperature
- 1 incomplete: TASK_670B
- 1 wrong implementation: TASK_685
- 2 synergies: TASK_692+693, TASK_674+675
- See `interactions.md` for full analysis

## Regression Hypothesis

The most likely cause of `agree_trade` dropping from ~15% to ~7% is the combination of:

1. **TASK_685 gradient clipping bug** — Per-asset intent defeated by global averaging
2. **TASK_677 over-sharpening** — 20× aggressive temperature reduces exploration
3. **TASK_151 warmup disabled** — Full trade weight immediately may cause instability

These factors together could create a training environment where:
- Volatile assets dominate gradient flow (grad clip bug)
- Oracle commands are too deterministic (over-sharpening)
- Model receives full signal strength without gradual ramping (no warmup)

The model may be forced into a conservative policy (all HOLD) to cope with this instability.
