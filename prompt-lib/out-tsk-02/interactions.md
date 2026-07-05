# Cross-Task Interaction Analysis

**Date**: 2026-02-23

## Critical Interactions Found

### 1. TASK_151 vs TASK_675: Warmup Disabled Contradiction

**Tasks Involved**: TASK_151 (Signal Soft-Start) and TASK_675 (Technical Debt)

**Interaction**:
- TASK_151: Spec requires 50 epochs of warmup for trade weights to prevent gradient shock
- TASK_675: Implementation comment says "warmup disabled — full trade weight from step 1 to break turtle trap"
- Config: `weight_warmup_epochs = 0` (disabled)

**Conflict**: Deliberate design choice contradicts original spec. The turtle trap concern (preventing all-HOLD bias) outweighed gradient explosion risk.

**Impact**: Fresh models receive full trade weight immediately. This could cause:
- Gradient explosion in early epochs (concern from TASK_151)
- OR faster breaking of turtle behavior (intent from TASK_675)

**Recommendation**: Monitor early training for gradient explosions. If unstable, consider partial warmup (e.g., 5-10 epochs).

---

### 2. TASK_677 vs Config: Temperature Sharpening Discrepancy

**Tasks Involved**: TASK_677 (Turtle Breaker) and config defaults

**Interaction**:
- TASK_677 Spec: `turtle_breaker.temperature = 2.0`
- Config: `oracle_sharpen_temperature = 0.1` (20× sharper than spec)
- Comment: "0.1 = sharp commands, 1.0 = no sharpening"

**Conflict**: Config is effectively 20× more aggressive sharpening than spec intended.

**Impact**: Over-sharpening could cause:
- Oracle commands to be too deterministic
- Reduced exploration in state channels
- Potential overfitting to sharp oracle signals

**Recommendation**: Verify if 0.1 was intentional (to aggressively break turtle) or should be restored to 2.0 per spec.

---

### 3. TASK_685: Gradient Clipping Implementation Mismatch

**Tasks Involved**: TASK_685 (Gradient Clip Fix)

**Interaction**: Implementation uses global averaging instead of per-asset clipping.

**Issue**:
- Spec: "clip gradients per asset to ensure balanced learning"
- Config: `grad_clip = 5.0` and `gradient_clip.per_asset = True`
- Code (`easy_trainer.py:215-229`): Uses `jnp.mean()` which averages across assets

**Impact**: Global averaging defeats the per-asset intent. Large gradients from volatile assets will dominate and be clipped, while small gradients from stable assets are under-clipped.

**Interaction with TASK_149**: Adaptive class balancing relies on proper gradient flow. If per-asset clipping is broken, the equalization of trade/hold gradient mass may be undermined.

**Recommendation**: Fix clipping to use `jnp.maximum()` per asset instead of `jnp.mean()`.

---

### 4. TASK_670B: Expert Imprinting Incomplete

**Tasks Involved**: TASK_670B (Expert Imprinting Stability)

**Interaction**: Config exists but implementation missing key components.

**Issue**:
- Config: `expert_imprinting_stability.enabled = True`, `temperature_decay`, `warmup_steps`
- Code: Config fields declared but logic incomplete
- Missing: Orthogonal initialization, weight freezing, adaptation schedules

**Impact**: Config suggests feature is enabled, but it's not working. Expert weights will drift without orthogonal protection.

**Interaction with TASK_684**: Decoupled hidden dims should make expert imprinting easier, but it's incomplete.

**Recommendation**: Either complete the implementation or disable in config.

---

### 5. TASK_691 vs TASK_692: Oracle Entropy Disabled Conflict

**Tasks Involved**: TASK_691 (Phase B Oracle Entropy) and TASK_692 (Model-State Rollout)

**Interaction**: TASK_692 explicitly disabled TASK_691.

**Issue**:
- TASK_691: Implement oracle entropy to corrupt selections before state channel computation
- TASK_692: Comment says "DISABLED by default - model-state rollout is the correct fix"
- Config: `oracle_entropy = False`

**Conflict**: TASK_691's purpose (simulating imperfect model actions) conflicts with TASK_692's approach (model trains on its own state channels).

**Reasoning**: Model-state rollout (TASK_692) is a more principled fix than oracle entropy noise (TASK_691). The entropy feature was deliberately disabled.

**Impact**: None — deliberate deprecation.

**Recommendation**: Consider removing TASK_691 code entirely to reduce confusion and code bloat.

---

### 6. TASK_687 vs TASK_680: Trade Sampling Double-Counting

**Tasks Involved**: TASK_687 (Hard Example Mining) and TASK_680 (Trade-Biased Sampling)

**Interaction**: Both modify `trade_sample_prob`.

**Issue**:
- TASK_680: Spec says `trade_sample_prob = 0.15`
- TASK_687: Doubled to `trade_sample_prob = 0.30` and documented in comment
- Config: `trade_sample_prob = 0.30`

**Conflict**: Not a conflict — documented escalation.

**Impact**: Trade-biased sampling now fires 30% of batches instead of 15%. This doubles the trade signal frequency, combined with TASK_679 (min_alpha_threshold 0.0005) for ~6× more trade examples.

**Recommendation**: None — documented escalation is intentional.

---

### 7. TASK_692 vs TASK_693: Train-Inference Alignment

**Tasks Involved**: TASK_692 (Model-State Rollout) and TASK_693 (Fix Position Semantics)

**Interaction**: Both tasks aim to eliminate train-inference mismatch.

**Synergy**:
- TASK_692: Model trains on state channels from its own actions
- TASK_693: Fixed position semantics to match incremental stepping

**Impact**: Together, these ensure:
- State channels during training match inference expectations
- Position stepping is identical between train and benchmark

**Interaction with TASK_689**: State channels (TASK_689) are properly integrated with autoregressive rollout (TASK_692).

**Recommendation**: None — good synergy.

---

### 8. TASK_686: Mixed Precision Config Documentation Mismatch

**Tasks Involved**: TASK_686 (Bfloat16 Mixed Precision)

**Interaction**: Implementation uses bfloat16 but config field defaults to "float32".

**Issue**:
- Code (`easy_trainer.py:343`): `jax.config.update("jax_default_matmul_precision", "bfloat16")`
- Config (`easy_config.py:39`): `compute_dtype = "float32"`
- Config is unused in implementation

**Impact**: None for actual behavior (bfloat16 is enabled), but confusing for users reading config.

**Recommendation**: Update config comment or remove unused `compute_dtype` field.

---

### 9. TASK_674 vs TASK_675: Overlapping Bug Fixes

**Tasks Involved**: TASK_674 (Training Agreement Plateau Bugs) and TASK_675 (Technical Debt)

**Interaction**: Both fix similar issues in the same codebase.

**Overlap**:
- TASK_674: Fixes 6 specific bugs related to agreement plateau
- TASK_675: Fixes 15 general stability issues
- Some overlap in optimizer and data pipeline fixes

**Impact**: Combined, these provide comprehensive stability. No conflicts detected.

**Recommendation**: None — complementary fixes.

---

### 10. TASK_695: Oracle Mask Head Minimal Impact

**Tasks Involved**: TASK_695 (Oracle Mask Prediction Head)

**Interaction**: Auxiliary head adds minimal loss contribution.

**Issue**:
- Loss contribution: `oracle_mask_loss = 0.023`, `total_loss ≈ 2.8`
- Percentage: `0.8%` of total loss
- Config: `use_oracle_mask_head = False` by default

**Impact**: Even if enabled, the auxiliary head contributes minimal gradient mass. Unlikely to cause regression.

**Interaction with TASK_673**: Multipass oracle provides the oracle mask for prediction.

**Recommendation**: None — low-risk auxiliary feature.

---

## Interaction Graph

```
TASK_151 (Warmup) ──┐
                    ├── Conflict ── weight_warmup_epochs = 0 vs spec 50
TASK_675 (Tech Debt) ─┘

TASK_677 (Turtle Breaker) ──┐
                       ├── Mismatch ── oracle_sharpen_temperature: 0.1 vs spec 2.0
Config Defaults ────────────┘

TASK_685 (Grad Clip) ─── Wrong implementation (mean vs per-asset max)

TASK_670B (Expert Imprinting) ─── Incomplete (config exists, logic missing)

TASK_691 (Oracle Entropy) ─── Disabled by TASK_692
       │
       └── TASK_692 (Model-State) overrides

TASK_687 (Hard Mining) ─── trade_sample_prob: 0.30 (doubled from 0.15)
       │
       └── TASK_680 (Trade-Biased) original 0.15

TASK_692 + TASK_693 ─── Synergy (train-inference alignment)

TASK_686 (bfloat16) ─── Config field unused (implementation correct)

TASK_674 + TASK_675 ─── Complementary bug fixes
```

## Summary of Interactions

| Type | Count | Description |
|------|-------|-------------|
| Conflicts | 2 | Warmup epochs, sharpening temperature |
| Incomplete | 1 | TASK_670B expert imprinting |
| Wrong Implementation | 1 | TASK_685 gradient clipping |
| Synergy | 2 | TASK_692+693, TASK_674+675 |
| Documentation | 1 | TASK_686 compute_dtype field |
| Deliberate Disable | 1 | TASK_691 disabled by TASK_692 |

## Most Problematic Interactions

1. **TASK_685 (Gradient Clipping)** - Wrong implementation directly contradicts spec's per-asset intent. This could cause imbalanced learning across assets.

2. **TASK_151 vs TASK_675** - Warmup disabled vs spec. Deliberate choice but contradicts gradient stability concern.

3. **TASK_670B (Expert Imprinting)** - Config says enabled but implementation incomplete. Expert weights will drift.

4. **TASK_677 vs Config** - 20× temperature sharpening discrepancy. Could cause over-sharpening and reduced exploration.
