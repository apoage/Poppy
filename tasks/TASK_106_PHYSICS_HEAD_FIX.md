# TASK_106: Physics Head Gradient Fix & Architecture Overhaul

**Status**: COMPLETED ✅ (Integrated into TASK_065 Refactor)
**Completion Date**: 2026-01-21

## Context
Investigation of training logs revealed that the "Physics Head" components (momentum, reversion, etc.) appear static, and prediction loss remains high.
Code analysis identified multiple critical architectural defects preventing the physics engine from learning.

## Root Cause Analysis

### 1. Zero-Gradient Trap (Initial State)
*   **Mechanism:** The physics ODE requires non-zero inputs (e.g., `log_returns`) to compute forces (e.g., `Momentum = α * log_returns`).
*   **Defect:** `predict_multi_step` used the `latent_to_market` decoder to derive the initial state ($t=0$). At start, this decoder outputs near-zero values.
*   **Result:** `Forces ≈ 0` → `∂Forces/∂Params ≈ 0`. Physics parameters received no learning signal.

### 2. Information Bottleneck (The "192 into 64" Problem)
*   **Mechanism:** The Physics Engine requires an initial state of $(LogPrice, LogReturn, MA)$ for all 64 assets ($64 \times 3 = 192$ values).
*   **Defect:** The model's `state_dim` was default **64**. It is mathematically impossible to losslessly compress 192 independent values into 64 dimensions.
*   **Result:** Initial state was always a lossy approximation, causing a "bad start" for every prediction.

### 3. Encoder Destroyed Asset Identity
*   **Mechanism:** The old `TRMEncoder` used standard CNN layers + Pooling.
*   **Defect:** Pooling mixed data from adjacent assets, destroying their unique identity.
*   **Result:** The decoder could not extract precise asset-specific prices from the "global soup" in the latent space.

### 4. Neural Cannibalization
*   **Mechanism:** `G_total = G_phys + G_neural`.
*   **Defect:** The flexible Transformer (`G_neural`) learned faster than the rigid linear Physics (`G_phys`), "eating" all gradients.

---

## Implementation Results

### 1. Initial State Teacher Forcing (Fix for #1 & #2)
*   **Change**: `MuZeroNetworks.predict_multi_step` now accepts ground-truth `teacher_log_returns` and `teacher_moving_avg`.
*   **Logic**: Overrides the $t=0$ state with teacher data during training. This bypasses the dimensional bottleneck and ensures non-zero inputs for gradient flow.

### 2. Decoder Reconstruction Loss (Fix for #2 & #4)
*   **Change**: Added `loss_decoder` to `trainer.py`.
*   **Logic**: MSE between `latent_to_market(latent_state)` and ground truth initial state. Forces the encoder to prioritize price/return features.

### 3. AssetTransformerEncoder V2 (Fix for #3)
*   **Change**: Replaced CNN encoder with a Transformer-based architecture (`trm_encoder_v2.py`).
*   **Logic**: Extracts features PER ASSET and uses Cross-Asset Attention. Preserves intra-epoch asset identity.

### 4. Neural Scale Annealing (Fix for #4)
*   **Change**: Enforced passing of `neural_scale` to `NeuralMarketODE`.
*   **Logic**: Starts at 0 (Physics-only) and anneals to 1. Forces the Physics core to explain the trend before the Neural head helps.

## Verification Results

### 1. Gradient Flow Test
*   **File**: `tests/test_trainer_loss.py`
*   **Metric**: Verified non-zero gradient norms for `alpha` and `beta` parameters.
*   **Result**: **PASS** (norms ~2.4e-6, previously effectively zero).

### 2. Physics Signal Test
*   **File**: `tests/test_physics_core.py`
*   **Metric**: Verified `G_momentum` is positive for up-trends and `G_reversion` is negative for over-valued prices.
*   **Result**: **PASS**.

### 3. Training Loop Integration
*   **Command**: `python train_jax_fast.py --epochs 1 --steps 10 --use-prediction-loss`
*   **Result**: **PASS**. Successful initialization, self-play, and training step execution with new components.