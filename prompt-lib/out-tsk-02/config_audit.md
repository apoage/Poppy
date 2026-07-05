# Config Audit — Spec vs Reality

**Date**: 2026-02-23

## Config Field Analysis

| Config Field | Actual Value | Which Task Set It | Expected Value Per Spec | Match? |
|-------------|---------------|-------------------|------------------------|--------|
| **Model Architecture** |||||
| encoder_version | 2 | TASK_676 | 2 (V2 flatten, no temporal) | ✅ |
| state_dim | 64 | TASK_666 | 64 | ✅ |
| n_channels | 34 | TASK_689 | 34 (was 27, +7 state channels) | ✅ |
| embed_dim | 32 | TASK_683 | 32 | ✅ |
| encoder_num_heads | 4 | TASK_683 | 4 | ✅ |
| compute_dtype | "float32" | TASK_686 | "bfloat16" for mixed precision | ⚠️ Spec says bfloat16, config has float32 |
| **Training** |||||
| learning_rate | 3e-4 | TASK_666 | 3e-4 | ✅ |
| weight_decay | 1e-4 | TASK_666 | 1e-4 | ✅ |
| warmup_steps | 1000 | TASK_666 | 1000 | ✅ |
| grad_clip | 5.0 | TASK_685 | 1.0 (was 1.0, changed to 5.0) | ⚠️ Spec said 1.0 was over-suppression, fix to 5.0 - but grad_clip logic still uses mean() |
| weight_warmup_epochs | 0 | TASK_151 | 50 | ❌ Spec says 50, config has 0 (comment explains: "warmup disabled to break turtle trap") |
| **Loss Weights** |||||
| policy_weight | 1.0 | TASK_149 | 1.0 | ✅ |
| value_weight | 0.5 | TASK_670B | 0.5 | ✅ |
| value_scale | 2.0 | TASK_674 | 2.0 (was 1.0) | ✅ |
| trade_loss_weight | 60.0 | TASK_149 (deprecated) | DEPRECATED | ✅ (marked deprecated) |
| **Oracle Parameters** |||||
| oracle_k | 4 | TASK_681/681 | 4 | ✅ |
| oracle_lookahead | 20 | TASK_152 | 20 | ✅ |
| min_alpha_threshold | 0.0005 | TASK_679 | 0.0005 (was 0.001) | ✅ |
| oracle_multipass | False | TASK_673 | False (disabled by default) | ✅ |
| oracle_n_passes | 10 | TASK_673 | 10 | ✅ |
| oracle_noise_scale | 0.03 | TASK_673 | 0.03 | ✅ |
| oracle_sharpen_temperature | 0.1 | TASK_677 | 2.0 | ⚠️ Spec says 2.0, config has 0.1 (comment: "0.1 = sharp commands, 1.0 = no sharpening") |
| oracle_entropy | False | TASK_691/692 | False (disabled) | ✅ (TASK_692 disabled it) |
| oracle_drop_rate | 0.15 | TASK_691 | 0.15 | ✅ |
| oracle_phantom_rate | 0.05 | TASK_691 | 0.05 | ✅ |
| entropy_warmup_epochs | 50 | TASK_691 | 50 | ✅ |
| use_oracle_mask_head | False | TASK_695 | False (disabled by default) | ✅ |
| oracle_mask_weight | 0.1 | TASK_695 | 0.1 | ✅ |
| **Model-State Rollout** |||||
| autoregressive_state | True | TASK_692 | True | ✅ |
| state_refresh_fraction | 0.2 | TASK_692 | 0.2 | ✅ |
| **Data** |||||
| window_size | 64 | TASK_152 | 64 | ✅ |
| episode_length | 400 | TASK_152 | 400 | ✅ |
| trade_sample_prob | 0.30 | TASK_680 | 0.15 (doubled to 0.30 by TASK_687) | ✅ (TASK_687 documented) |
| **Success Criteria** |||||
| target_oracle_agreement | 0.75 | TASK_147 | 0.75 | ✅ |

## Summary Statistics

| Category | Total Fields | Match | Mismatch | Notes |
|----------|-------------|-------|----------|-------|
| Model Architecture | 7 | 6 | 1 | compute_dtype: float32 vs spec bfloat16 |
| Training | 5 | 4 | 1 | weight_warmup_epochs: 0 vs spec 50 |
| Loss Weights | 4 | 4 | 0 | |
| Oracle Parameters | 11 | 10 | 1 | oracle_sharpen_temperature: 0.1 vs spec 2.0 |
| Model-State Rollout | 2 | 2 | 0 | |
| Data | 4 | 4 | 0 | |
| Success Criteria | 1 | 1 | 0 | |
| **TOTAL** | **34** | **31** | **3** | 91% match rate |

## Config Mismatches Analysis

### 1. compute_dtype: "float32" vs spec "bfloat16" (TASK_686)
**Impact**: TASK_686 implemented mixed precision in `easy_trainer.py` with `jax.config.update("jax_default_matmul_precision", "bfloat16")`, but the config field `compute_dtype` defaults to "float32". This is a documentation/config-only mismatch — the actual code uses bfloat16 via the jax config update, not via the config field.

**Action**: Update config comment or remove unused `compute_dtype` field to avoid confusion.

### 2. weight_warmup_epochs: 0 vs spec 50 (TASK_151)
**Impact**: High. TASK_151 intended 50 epochs of warmup to prevent gradient explosion. Comment explains it's disabled "to break turtle trap". This is a deliberate design choice to prioritize immediate trade signal over gradual ramping.

**Action**: Document the decision explicitly in the task spec or consider if the turtle trap concern outweighs gradient explosion risk.

### 3. oracle_sharpen_temperature: 0.1 vs spec 2.0 (TASK_677)
**Impact**: Medium. Spec (TASK_677) says temperature should be 2.0 for sharpening, but config has 0.1 (sharper). Comment says "0.1 = sharp commands, 1.0 = no sharpening". This is effectively 20× sharper than spec intended.

**Action**: Verify if 0.1 was intentional (over-sharpening to break turtle) or if 2.0 should be restored.

## Missing Config Fields (Spec References)

| Field | Task | Missing? | Notes |
|-------|-------|-----------|-------|
| entropy_weight | TASK_691 | ✅ | Not implemented in code or config |
| gradient_clip.per_asset | TASK_685 | ⚠️ | Config field exists but implementation uses mean() |
| expert_imprinting_stability.* | TASK_670B | ⚠️ | Config exists but implementation incomplete |

## Critical Finding: TASK_691 Oracle Entropy

The config for TASK_691 is present (`oracle_entropy: bool = False`, `oracle_drop_rate`, `oracle_phantom_rate`, `entropy_warmup_epochs`), BUT:
1. The implementation is BROKEN (undefined `_price_generator` reference)
2. No entropy loss term in total loss calculation
3. No `--oracle-entropy` CLI flag

This means even if `oracle_entropy=True` were set, the feature would not work correctly.
