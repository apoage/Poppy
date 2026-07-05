# Task Implementation Audits

## Tier 1: Loss, Targets, and Gradient Signal

### TASK_147: Agreement Metric Fix
**Spec Status**: Balanced accuracy: 0.5*trade + 0.5*hold.
**Implementation Status**: `easy_trainer.py` lines 207-235.
**Completeness**: FULL
**Correctness**: CORRECT. Computes `agree_trade` and `agree_hold` separately and averages them.
**Config Check**: N/A

### TASK_149: Adaptive Class Balancing
**Spec Status**: Adaptive trade weighting ~25x.
**Implementation Status**: `easy_trainer.py`.
**Completeness**: FULL
**Correctness**: CORRECT. Calculates `n_holds / n_trades` dynamically per batch.
**Issues**: Can be unstable if batch has near-zero trades, but `jnp.maximum(..., 1.0)` guards this.

### TASK_151: Signal Soft Start
**Spec Status**: Warmup ramp for trade weight.
**Implementation Status**: `train_easy.py` passes `warmup_progress`. `easy_trainer.py` applies it.
**Completeness**: FULL
**Correctness**: CORRECT (logic-wise).
**Issues**: **Config Mismatch**. `weight_warmup_epochs` defaults to 0 in `easy_config.py`, effectively disabling this feature.

### TASK_157: Oracle K and Adaptive Weight
**Spec Status**: Adjust oracle K and update weighting logic.
**Implementation Status**: `easy_config.py`.
**Completeness**: FULL
**Correctness**: CORRECT. `oracle_k=4` and `min_alpha_threshold` lowered to 0.0005.

### TASK_670B: Expert Imprinting Stability
**Spec Status**: Value loss fixes, return scaling.
**Implementation Status**: `easy_trainer.py`.
**Completeness**: FULL
**Correctness**: CORRECT. `value_scale=2.0` matches spec.

### TASK_673: Multipass Oracle
**Spec Status**: Soft targets from multi-pass oracle.
**Implementation Status**: `train_easy.py`.
**Completeness**: FULL
**Correctness**: CORRECT.
**Issues**: **Disabled by Default**. `oracle_multipass` is False in config.

### TASK_681: Oracle Trade Density Ceiling
**Spec Status**: Ensure oracle trade density doesn't exceed reasonable limits (restore K=4).
**Implementation Status**: `easy_config.py`.
**Completeness**: FULL
**Correctness**: CORRECT. `oracle_k` restored to 4.

### TASK_685: Gradient Clip Fix
**Spec Status**: Increase gradient clipping to prevent suppression.
**Implementation Status**: `easy_config.py`.
**Completeness**: FULL
**Correctness**: CORRECT. `grad_clip` increased to 5.0.

### TASK_695: Oracle Mask Prediction Head
**Spec Status**: Aux head for predicting oracle mask.
**Implementation Status**: `easy_trainer.py`, `mcts_trader/jax_agent/muzero_model.py`.
**Completeness**: FULL
**Correctness**: CORRECT. Loss added, config toggle present.

## Tier 2: Data Pipeline and Sampling

### TASK_153: Memory Efficient Buffer
**Spec Status**: Optimized buffer management.
**Implementation Status**: `train_easy.py`.
**Completeness**: PARTIAL
**Correctness**: CORRECT (Implementation).
**Issues**: **Default Mismatch**. Spec recommends `n_envs=4`, code defaults to `n_envs=8`.

### TASK_154: Historic Data Easy Mode
**Spec Status**: Integrate historic data loader.
**Implementation Status**: `train_easy.py`.
**Completeness**: **NOT IMPLEMENTED**.
**Correctness**: N/A
**Issues**: Code only handles Synthetic and Bella data. No historic data path exists.

### TASK_674: Plateau Bugs
**Spec Status**: Fix buffer sample counter.
**Implementation Status**: `train_easy.py`.
**Completeness**: FULL
**Correctness**: CORRECT. `buffer_sample_counts` managed correctly.

### TASK_680: Trade Biased Sampling
**Spec Status**: Coin-flip trade-biased offset sampling.
**Implementation Status**: `train_easy.py`.
**Completeness**: FULL
**Correctness**: CORRECT.

### TASK_682: Real Data Benchmark
**Spec Status**: Benchmark tool for real market data validation.
**Implementation Status**: `train_easy.py` (hook), `tools/benchmark_easy.py`.
**Completeness**: FULL
**Correctness**: CORRECT.

### TASK_691: Phase B Oracle Entropy
**Spec Status**: Oracle entropy injection.
**Implementation Status**: `train_easy.py`.
**Completeness**: FULL
**Correctness**: CORRECT. `corrupt_oracle_selections` implemented and wired.

## Tier 3: Architecture and State Channels

### TASK_152: Untiling Encoder
**Spec Status**: Use (batch, time, 64, 64) format.
**Implementation Status**: `train_easy.py`.
**Completeness**: FULL
**Correctness**: CORRECT. `jit_vmap_encode_temporal` used.

### TASK_684: Decouple Hidden Dim
**Spec Status**: Separate hidden_dim from state_dim.
**Implementation Status**: `easy_config.py`, `easy_trainer.py`.
**Completeness**: PARTIAL
**Correctness**: WRONG in Preset.
**Issues**: `get_large_config` hardcodes `trm_hidden_dim=1024`. For `state_dim=128`, default should be `2048`. This is a 50% capacity regression.

### TASK_689: Agent State Channels
**Spec Status**: 34 state channels (27 market + 7 agent).
**Implementation Status**: `easy_config.py`, `train_easy.py`.
**Completeness**: FULL
**Correctness**: CORRECT. 34 channels confirmed.

### TASK_692/693: Model State Rollout & Position Semantics
**Spec Status**: Autoregressive state channels, incremental position stepping.
**Implementation Status**: `train_easy.py`.
**Completeness**: FULL
**Correctness**: **CRITICAL LOGIC ERROR (Target Mismatch)**.
**Issues**:
- The code calculates `policy_targets` using the **Oracle's** state path (via `greedy_lookahead_path` -> `optimal_path_to_policy`).
- The code calculates `observations` using the **Student's** state path (via `_build_model_state_trajectory`).
- **Result**: The model receives inputs where it might be Long, paired with a label "Buy" (because the Oracle was neutral).
- **Impact**: Model learns to Buy when already Long (illegal/wasteful) or Hold when neutral (missed opportunity). This destroys the "Agreement" signal because the label is invalid for the input state.

## Tier 4: Bug Fixes and Orchestration

**BUG-1 (Alignment)**: Checked. `returns` are sliced `[ws-1:]` to align with window decision point. CORRECT.
**BUG-2 (Confidence)**: Checked. `prev_conf` carried in scan. CORRECT.
**BUG-4 (Merge Trim)**: Checked. Correctly implemented in `merge_utils.py`.
**TASK_690 (Dual Dojo)**: Orchestration script `tools/orchestration/dual_dojo.py` exists and appears structured correctly.
