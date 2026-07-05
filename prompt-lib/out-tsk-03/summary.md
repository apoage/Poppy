# Implementation Audit — Summary

**Date**: 2026-02-23
**Model**: Easy Mode (JAX)
**Tasks Audited**: 40+
**Files Read**: `easy_config.py`, `easy_trainer.py`, `train_easy.py` and others.

## Counts
- Fully Implemented & Correct: ~35
- Partially Implemented: 1
- Incorrectly Implemented: 1 (Critical)
- Missing Implementation: 1
- Config Mismatch: 4
- Cross-Task Conflict: 2

## Top Suspects (most likely to cause agree_trade regression)

1. **TASK_692/693 (Target Mismatch)**: **CRITICAL**. The model is being trained with labels that contradict its input state. It learns from its own rollout (Student State) but uses labels derived from the Oracle's rollout (Oracle State). When states diverge, the gradient is wrong. This explains the collapse to ~7% agreement (random noise).
2. **TASK_684 (Capacity Reduction)**: Large model capacity was accidentally cut by 50% (`trm_hidden_dim` 2048 -> 1024) in the config preset.
3. **TASK_151 (Disabled Warmup)**: Soft start is disabled (`warmup_epochs=0`), leading to potential gradient shock from the high trade weights (TASK_149) in early training.
4. **TASK_154 (Missing Data)**: Historic data loading is completely missing from `train_easy.py`. Training is purely synthetic/Bella.

## Recommended Actions

1. **FIX TASK_692/693**: In `train_easy.py`, re-compute `policy_targets` using the *Student's* position trajectory (from the rollout) instead of using the pre-computed Oracle targets.
2. **FIX TASK_684**: Update `easy_config.py` Large preset to set `trm_hidden_dim=2048` (or `None`).
3. **RESTORE TASK_151**: Set `weight_warmup_epochs=50` in `easy_config.py` to prevent early training instability.
4. **ENABLE TASK_673**: Consider setting `oracle_multipass=True` by default (or check why it's off).
5. **IMPLEMENT TASK_154**: Add support for historic data loading in `train_easy.py` if real-world data training is required.
