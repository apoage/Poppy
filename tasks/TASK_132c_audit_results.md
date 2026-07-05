# TASK_132c Audit Results: BC/Imprint Information Flow

**Date:** 2026-02-01
**Auditor:** Gemini CLI

## Executive Summary
A comprehensive audit of the Behavior Cloning (BC) and Causal Expert Imprinting system was performed. A **CRITICAL BUG** was identified in the observation encoding pipeline for offline training (`expert_buffer.py`), causing future information leakage. The online environment (`jax_trading_env.py`) is safe. Other audited components (Oracle Flow, Sampling, Metrics, Shape Consistency) are functioning correctly.

---

## 1. Information Flow Audit

### 1.1 Oracle Action Flow
*   **Status:** ✅ **OK**
*   **Location:** `train_jax_fast.py` (lines ~630-660) & `expert_buffer.py`
*   **Verification:**
    *   `compute_optimal_path` uses log returns correctly.
    *   Action integer encoding (`asset_idx * 5 + action_type`) matches environment spec.
    *   `optimal_action` stored as `uint16` in `UnifiedIndexBuffer`.

### 1.2 Observation Encoding Flow (Offline BC)
*   **Status:** 🔴 **CRITICAL BUG (Future Leakage)**
*   **Location:** `mcts_trader/jax_agent/expert_buffer.py:465` (`_encode_observations_vectorized`)
*   **Finding:**
    *   The buffer calculates `window_starts = jnp.maximum(steps - window_size, 0)`.
    *   It passes the raw (unpadded) `episode_prices` to the encoder.
    *   For `step=0` (start of episode), `window_start` clamps to 0.
    *   The encoder `jax.lax.dynamic_slice` grabs `prices[0:60]`.
    *   **Leakage:** At decision time $t=0$, the agent sees prices $P_0 \dots P_{59}$ (i.e., 59 steps into the future).
    *   **Contrast:** The online environment (`jax_trading_env.py`) avoids this by ensuring `episode_start_idx >= window_size`, guaranteeing valid history. The offline buffer stores episodes starting at $t=0$ without pre-history padding.
*   **Impact:** The BC agent learns to cheat by "looking ahead" in the observation window, leading to unrealistically high validation performance that collapses in live/online evaluation.
*   **Fix:** Pad the `close_prices` array with `window_size - 1` copies of $P_0$ (edge padding) before slicing, and remove the `- window_size` offset in `window_starts`.

### 1.3 Sampling Flow
*   **Status:** ✅ **OK**
*   **Location:** `expert_buffer.py` (`sample_batch`)
*   **Verification:**
    *   Correctly packages observations, actions, and weights.
    *   `confidence` correctly passed (though currently static 1.0).
    *   PER weights calculated correctly.

### 1.4 BC Loss Flow
*   **Status:** ✅ **OK**
*   **Location:** `trainer.py` (`_loss_fn`)
*   **Verification:**
    *   Correct extraction of `expert_batch['optimal_actions']`.
    *   Loss computation (Large Margin or Cross-Entropy) is correct.
    *   `expert_action_agreement` metric computed correctly.

### 1.5 Metrics Propagation
*   **Status:** ⚠️ **MINOR ISSUE (Naming Ambiguity)**
*   **Location:** `train_jax_fast.py` vs `trainer.py`
*   **Finding:**
    *   `trainer.py` computes `expert_acceptance_rate` as "Action Agreement %".
    *   `train_jax_fast.py` (DS24 logger) calculates a separate `expert_acceptance_rate` as "Storage Rate" (demos stored / total steps).
    *   The CSV log uses the Trainer's definition (Agreement).
    *   **Impact:** Confusion when interpreting logs, but no functional breakage.

---

## 2. Shape Consistency Audit
*   **Status:** ✅ **OK**
*   **Pipeline:** `expert_buffer.py` -> `trainer.py` -> `muzero_model.py`
*   **Verification:**
    *   Buffer output: `(batch, 27, 60, 64)`? **Correction:** The encoder collapses the window dimension into features.
    *   `jax_encoder` output: `(27, n_assets, n_assets)` (channels, height, width).
    *   Model input: `(batch, 27, 64, 64)`.
    *   **Verdict:** Shapes align correctly. The "window" dimension is consumed by the encoder to produce feature maps.

---

## 3. Priority & Sampling Logic Audit
*   **Status:** ✅ **OK**
*   **Location:** `expert_buffer.py`
*   **Verification:**
    *   **Priority:** `priority = confidence * max_priority + boost`. With `confidence=1.0`, demos compete with the highest-error agent samples.
    *   **Stratified Sampling:** `_sample_from_range` uses stratified sampling (standard for PER), ensuring diverse priority coverage.
    *   **Demo Ratio:** Explicit logic ensures `demo_ratio` (12.5%) of the batch comes from the protected demo zone.

---

## 4. Recommendations
1.  **Fix Observation Leakage:** Modify `expert_buffer.py` to pad episode prices before encoding.
2.  **Clarify Metrics:** Rename the DS24 "storage rate" metric to `expert_storage_rate` to distinguish it from `expert_action_agreement`.
