# Per-Task Audit Findings

**Date**: 2026-02-23
**Tasks Audited**: 21 tasks + 7 bug fixes

---

## Tier 1: Loss, Targets, and Gradient Signal

### TASK_147: Agreement Metric Semantic Fix

**Spec Status**: COMPLETE
**Implementation Status**: Fully implemented in both training and evaluation paths.

**Completeness**: FULL
**Correctness**: CORRECT

**Issues Found**:
- None. Balanced accuracy formula at `easy_trainer.py:227-245` exactly matches spec. Both `_train_step_impl` and `evaluate_batch` use identical computation.

**Config Check**:
- `target_oracle_agreement = 0.75` -- matches spec: YES

---

### TASK_149: Adaptive Class Balancing

**Spec Status**: COMPLETE
**Implementation Status**: Evolved beyond spec (continuous weighting via TASK_675/677).

**Completeness**: FULL (exceeds spec)
**Correctness**: CORRECT

**Issues Found**:
- Binary `is_trade` replaced by continuous `trade_strength = 1.0 - hold_prob` at `easy_trainer.py:172-173`. Deliberate TASK_675 improvement. With near-one-hot targets (multipass OFF), functionally identical to binary.
- `MAX_TRADE_WEIGHT = 64.0` at line 181. Original spec had no cap. Cap is adequate for k=4.
- `trade_loss_weight = 60.0` deprecated, kept for checkpoint compat.

**Config Check**:
- Adaptive weight `n_holds/n_trades` ≈ 24.7 in current logs -- correct for k=4.

---

### TASK_151: Signal Soft-Start (Warmup)

**Spec Status**: COMPLETE
**Implementation Status**: Code correct but **disabled** by TASK_677.

**Completeness**: FULL
**Correctness**: CORRECT (code matches spec; config overrides)

**Issues Found**:
- `weight_warmup_epochs = 0` at `easy_config.py:58` bypasses warmup.
- `train_easy.py:1309` correctly returns `warmup_progress = 1.0` when warmup disabled.

**Config Check**:
- `weight_warmup_epochs = 0` -- matches TASK_677 intent: YES

---

### TASK_670B: Expert Imprinting Stability

**Spec Status**: COMPLETE
**Implementation Status**: Partially implemented.

**Completeness**: PARTIAL
**Correctness**: CORRECT for what is implemented.

**Issues Found**:
- Fix 1 (no-latent-cache): Implemented in e-dojo. YES.
- Fix 2 (action dist logging): `easy_trainer.py:267-268`. YES.
- Fix 3 (value_scale): `easy_config.py:53`, `value_scale=2.0`. YES.
- **Concern**: `value_scale=2.0` produces sub-bin-resolution for small trades (1% return -> 0.02, bin width 0.05). Spec suggested tuning to 5.0.

**Config Check**:
- `value_scale = 2.0` -- matches spec default: YES

---

### TASK_673: Multipass Oracle

**Spec Status**: REVIEWED
**Implementation Status**: Fully implemented but **NOT ACTIVE** in current training.

**Completeness**: FULL (exceeds spec)
**Correctness**: MOSTLY CORRECT

**Issues Found**:
1. `oracle_multipass=False` (default) and e-dojo does NOT pass `--oracle-multipass`. All multipass/sharpening features dormant.
2. Return signature 4-tuple (spec says 3). Callers handle correctly.
3. Clean oracle PnL for value targets (improvement over spec's averaged PnL).
4. **`prev_selections` 2-way vs 3-way branch discrepancy**: `train_easy.py:532` maps `returns==0` to `-1.0` (short). `path_solver.py:302` maps it to `0.0` (flat). Active in single-pass path. Edge case but incorrect.

**Config Check**:
- `oracle_multipass = False` -- matches default: YES
- `oracle_noise_scale = 0.03` -- spec says 0.1 (tuned, documented): YES

---

### TASK_677: Turtle Breaker (Temperature Sharpening)

**Spec Status**: All 3 items complete.
**Implementation Status**: Sharpening implemented but **NOT ACTIVE** (depends on multipass).

**Completeness**: FULL
**Correctness**: CORRECT

**Issues Found**:
- Sharpening at `path_solver.py:329-335` only runs when `oracle_multipass=True`. Currently OFF.
- Warmup disabled and continuous weighting are both active and correct.

**Config Check**:
- `oracle_sharpen_temperature = 0.1` -- matches spec: YES (but inactive)

---

### TASK_685: Gradient Clip Fix

**Spec Status**: DONE
**Implementation Status**: Fully implemented.

**Completeness**: FULL
**Correctness**: CORRECT

**Issues Found**: None. `grad_clip=5.0` at `easy_config.py:48`. Logs show grad_norm 5-22, suppression 1-4.4x.

**Config Check**:
- `grad_clip = 5.0` -- matches spec: YES

---

### TASK_695: Oracle Mask Prediction Head

**Spec Status**: TODO (not updated after implementation)
**Implementation Status**: Fully implemented and **ACTIVE** in current training via `--oracle-mask-head`.

**Completeness**: FULL
**Correctness**: CORRECT but turtle-trapped

**Issues Found**:
- `oracle_mask_accuracy ≈ 93.7%` = all-zeros baseline (60/64 = 93.75%). Head learns nothing useful.
- Loss contribution: `0.1 * 0.226 = 0.023` (~0.8% of total). Gradient direction: anti-trade (suppress trade features in shared repr).
- E-dojo passes `--oracle-mask-head` making it active despite `use_oracle_mask_head=False` default.

**Config Check**:
- `use_oracle_mask_head = False` (default) -- matches spec: YES
- `oracle_mask_weight = 0.1` -- matches spec: YES

---

## Tier 2: Data Pipeline and Sampling

### TASK_674: Training Agreement Plateau Bugs

**Spec Status**: Bugs 1,2,4,5,6 DONE. Bug 3 unchecked.
**Implementation Status**: All code-level fixes applied.

**Completeness**: FULL
**Correctness**: CORRECT

**Issues Found**:
- Bug 1 (buffer sample counter): `train_easy.py:1160` `+= take_count`. Correct.
- Bug 2 (optimizer warmup reset): `easy_trainer.py:499-510`. Correct.
- Bug 4 (separate val generator): `train_easy.py:1534-1546`. Correct.
- Bug 5 (value_scale): `easy_config.py:53`. Correct.
- Bug 6 (zero-return direction): 3-way where in `path_solver.py:743`, `path_solver.py:302`, `train_easy.py:162-166`. Correct.

**Config Check**:
- `value_scale = 2.0` -- matches spec: YES

---

### TASK_675: Technical Debt & Stability

**Spec Status**: Items #1-#11, #13-#15 DONE. #12 future work.
**Implementation Status**: All marked items verified.

**Completeness**: FULL
**Correctness**: CORRECT

**Issues Found**: None. All 14 implemented items verified correct.

**Config Check**:
- `steps_per_epoch = 200` (base), overridden to 400 by CLI: YES

---

### TASK_679: Trade Signal Density

**Spec Status**: Phase 1 DONE.
**Implementation Status**: Config change applied.

**Completeness**: FULL (Phase 1)
**Correctness**: CORRECT

**Issues Found**:
- `min_alpha_threshold = 0.0005` at `easy_config.py:68`. All 4 oracle call sites pass it correctly.

**Config Check**:
- `min_alpha_threshold = 0.0005` -- matches spec: YES

---

### TASK_680: Trade-Biased Sampling

**Spec Status**: Implementation done, tests unchecked.
**Implementation Status**: Code complete. **Two critical issues.**

**Completeness**: FULL
**Correctness**: **WRONG** -- effective bias far weaker than intended

**Issues Found**:
1. **CRITICAL: Contiguous block sampling dilutes trade bias ~10-20x.** Coin flip at `train_easy.py:1139-1149` fires once per episode visit, choosing start of a `take_count=20` contiguous block. Only 1 of 20 samples is trade-biased. Spec expected per-sample independence giving 30-35% trade windows per batch. Actual: ~3-8%.
2. **Validation uses biased sampling.** `val_generator` receives same config with `trade_sample_prob=0.30`, inflating val agree_trade. Spec explicitly warns against this.
3. **Missing buffer size warning** per ADV-1 when `buffer_size < 64`.

**Impact on Training**:
- Trade bias ~10-20x weaker than designed. Model still sees hold-dominated batches.
- Val metrics inflated -- true uniform agree_trade is even lower than reported 7%.

**Config Check**:
- `trade_sample_prob = 0.30` -- matches TASK_687 update: YES

---

### TASK_687: Hard Example Mining

**Spec Status**: DONE
**Implementation Status**: Config change (0.15 -> 0.30).

**Completeness**: FULL (Option A)
**Correctness**: CORRECT, but effect nullified by TASK_680 dilution

**Issues Found**:
- Config at `easy_config.py:103`: `trade_sample_prob=0.30`. Correct.
- Doubling from 0.15 to 0.30 has negligible effect due to contiguous block dilution.
- Validation bias not fixed (spec warns against it).

**Config Check**:
- `trade_sample_prob = 0.30` -- matches spec: YES

---

### TASK_691: Phase B Oracle Entropy

**Spec Status**: DONE, disabled.
**Implementation Status**: All code present, properly disabled.

**Completeness**: FULL
**Correctness**: CORRECT

**Issues Found**: None. Properly disabled when `autoregressive_state=True`.

**Config Check**:
- `oracle_entropy = False` -- matches TASK_692 supersession: YES

---

## Tier 3: Architecture and State Channels

### TASK_683: Widen Encoder State

**Spec Status**: DONE
**Implementation Status**: Fully implemented.

**Completeness**: FULL
**Correctness**: CORRECT (but benefit nullified by TASK_684)

**Issues Found**:
- All params wired end-to-end: `state_dim=128`, `embed_dim=64`, `encoder_num_heads=8`.
- **Dead config field**: `EasyConfig.hidden_dim` (256/512) is NEVER used by model code. Only `trm_hidden_dim` is wired.
- **embed_dim/encoder_num_heads are dead for V2**: TRM does not pass them to V2 encoder (`trm_flax.py:248`).

**Config Check**:
- `state_dim = 128` (large) -- matches spec: YES
- `embed_dim = 64` (large) -- matches spec: YES (but dead for V2)

---

### TASK_684: Decouple hidden_dim

**Spec Status**: DONE
**Implementation Status**: Fully implemented. **Spec value is wrong.**

**Completeness**: FULL
**Correctness**: **WRONG** -- spec carried over old value

**Issues Found**:
- **CONFIRMED PRIMARY REGRESSION**: `trm_hidden_dim=1024` in `get_large_config()` prevents FFN from growing with `state_dim=128`.
- Before TASK_683/684: `state_dim=64`, FFN default `64*16=1024`.
- After: `state_dim=128`, FFN should be `128*16=2048`, but capped at 1024.
- ~23M parameters removed. Reasoning capacity unchanged despite wider representation.
- TASK_683 spec line 66 says `hidden_dim=1024 # Was state_dim*16=1024` -- this comment is wrong. At new `state_dim=128`, the default would be 2048.
- TASK_684 spec line 55 says "Existing presets unchanged" -- violated by the cap.

**Data flow trace**:
```
get_large_config() -> trm_hidden_dim=1024
  -> EasyTrainer -> hidden_dim=getattr(config, 'trm_hidden_dim', None) = 1024
    -> MuZeroNetworks(hidden_dim=1024)
      -> TRMReasoning(hidden_dim=1024)
        -> trm_flax.py:112: hidden_dim = 1024 (NOT 128*16=2048)
```

**Config Check**:
- `trm_hidden_dim = 1024` (large) -- matches spec but **spec is wrong**: NO

---

### TASK_686: bfloat16 Mixed Precision

**Spec Status**: DONE
**Implementation Status**: Fully implemented.

**Completeness**: FULL
**Correctness**: CORRECT

**Issues Found**: None. Default float32, bfloat16 opt-in. Not active.

**Config Check**:
- `compute_dtype = "float32"` -- matches spec: YES

---

### TASK_689: Agent State Channels

**Spec Status**: Phase A DONE, Phase B deferred, Phase C partial.
**Implementation Status**: Phase A complete, Phase C in benchmark_easy.py.

**Completeness**: PARTIAL (Phase A + partial C)
**Correctness**: CORRECT

**Issues Found**:
- All 6 state channels + 1 confidence = 34 total. Correct.
- Temporal leak fix (t-1 indexing) at `train_easy.py:636`. Correct.
- Benchmark inference at `benchmark_easy.py:528-609` matches training semantics.

**Config Check**:
- `n_channels = 34` -- matches spec: YES

---

### TASK_692: Model-State Rollout

**Spec Status**: DONE
**Implementation Status**: Fully implemented.

**Completeness**: FULL
**Correctness**: CORRECT

**Issues Found**:
- `_build_model_state_trajectory()` at `train_easy.py:173-308`. JIT'd scan with stop_gradient. Correct.
- `set_model_params()` at `train_easy.py:1036-1056`. Increments param_version, partial buffer refresh. Correct.
- FIFO refresh with sentinel at `train_easy.py:1058-1093`. Correct.
- **Turtle feedback loop**: Model predictions → flat state channels → reinforces HOLD → predictions stay flat. This is architecturally correct but creates a bootstrap problem (see interactions.md).

**Config Check**:
- `autoregressive_state = True` -- matches spec: YES
- `state_refresh_fraction = 0.2` -- matches spec: YES

---

### TASK_693: Fix Position Semantics

**Spec Status**: IMPLEMENTED
**Implementation Status**: All 4 bugs fixed.

**Completeness**: FULL
**Correctness**: CORRECT

**Issues Found**:
- Bug 1 (absolute→incremental position): `train_easy.py:263-276`. Verified semantic equivalence with `benchmark_easy.py:586-593` across all 6 position states.
- Bug 2 (confidence always zero): Fixed. `prev_conf` from carry fed into observation.
- Bug 3 (wasted oracle computation): Fixed. `elif` structure prevents redundant oracle calls.
- Bug 4 (dimension mismatch): Fixed. Conditional slicing for model vs oracle rollout.

**Config Check**: No config changes required.

---

## Tier 4: Bug Fixes (TASK_694)

### BUG-1: Returns Alignment

**Correctness**: CORRECT
- Single-episode: `returns_aligned = log_returns[ws - 1:]` → `returns_aligned[idx] = log_returns[idx + 63]`. Correct.
- Batch: `log_returns[i, ws - 1:valid_lengths[i] - 1]`. Consistent alignment. Excess scan steps process padded data but output discarded.

### BUG-2: Confidence Channel

**Correctness**: CORRECT
- 6-element carry init matches unpack order. Emits `prev_conf` not `model_confidence`. Matches inference.

### BUG-4: TIES Merge Trim Direction

**Correctness**: CORRECT
- `threshold = jnp.percentile(|d|, trim_fraction * 100)` → keeps top 80%. Matches TIES paper.

### BUG-5: Val Generator Params

**Correctness**: CORRECT
- `val_generator.set_model_params()` called before each validation when `autoregressive_state=True`.

### BUG-6: best_agreement After Merge

**Correctness**: CORRECT
- `max(oracle_agreement, a_agreement, b_agreement)`. Handles merged checkpoints.

### BUG-7: Trade-Biased Sampling Boundary

**Correctness**: CORRECT
- `valid_trades = trade_indices[trade_indices <= max_offset]`. Only ~3-10% of indices filtered.

### LATENT: Buffer Param IDs

**Correctness**: CORRECT
- Sentinel `-1` instead of `del`. Prevents list length desync. FIFO ordering preserved.
