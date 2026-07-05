# TASK_127: TPU Migration and Bella v2m (Mamba + DEX Data)

**Status:** PLANNING
**Owner:** Lukas
**Date:** 2026-01-30

## 1. Objective
Enable "Bella v2m" (Version 2 Mamba) training on Google Cloud TPU using a sovereign, copyright-free dataset derived from Decentralized Exchanges (DEXs). Goal: 10x reduction in training time (10 days -> 1 day) by utilizing high-bandwidth memory (HBM) and eliminating PCIe bottlenecks for 128-256 parallel environments.

## 2. TPU Migration (Infrastructure)
*   **Goal:** Run `train_jax_fast.py` and Bella generation natively on TPU devices (v4/v5).
*   **Kaggle/Colab Optimization (9-hour Free Sessions):**
    - [ ] **Throughput Target**: Achieve 60,000 - 80,000 steps per environment per session (8x v5e).
    - [ ] **Scaling**: Scale to 128+ venvs (vs 50 on GPU) leveraging HBM bandwidth.
    - [ ] **Persistence**: Script for automatic checkpoint upload to GCS/Kaggle at hour 8.5.
    - [ ] **JAX Cache**: Pre-warm JAX compilation cache to skip 1-2 hours of startup overhead.
*   **Action Items:**
    - [ ] **Conditional Flags**: Update `train_jax_fast.py` to wrap NVIDIA-specific env vars (`XLA_FLAGS`, `TF_GPU_ALLOCATOR`) in `if not tpu_detected:` blocks.
    - [ ] **Platform Agnosticism**: Remove `os.environ['JAX_PLATFORM_NAME'] = 'gpu'` from `diff_teacher/agent_evaluator.py`.
    - [ ] **Pod Support**: Add `jax.distributed.initialize()` support for multi-host TPU Pod training.

## 3. DEX Data Pipeline ("The Harvester")

### 3.1 Data Acquisition Strategy
We need high-fidelity historical data (2020-2025) and real-time streams.
*   **DEX Harvester Node Setup:**
    -   *Strategy:* Use a lightweight execution client (Reth) on a local machine or GCE instance for real-time monitoring.
    -   *Historical:* Pull from BigQuery (primary) to create a static training corpus.
    -   *Real-time:* Subscribe to Uniswap v3 Swap events via WebSocket.

### 3.2 The "Clean" Pipeline (Ideal Approach)
Raw DEX events (`Swap`) are noisy (MEV bots, flash loans). We need a rigorous cleaning process to produce "Mamba-grade" data.

#### Step 1: Raw Extraction (BigQuery/RPC)
Extract `Swap` events from top Uniswap V3 pools (USDC/ETH, WBTC/USDC, etc.).
- **Schema:** `block_number`, `transaction_index`, `log_index`, `sender`, `recipient`, `amount0`, `amount1`, `sqrtPriceX96`, `liquidity`, `tick`.

#### Step 2: "Sanitization" (The Cleaning Layer)
1.  **MEV Filtering (Sandwich Attacks):**
    -   Detect atomic bundles: Large Buy -> Victim Tx -> Large Sell (in same block).
    -   *Action:* Discard the artificial price spikes caused by the attacker. Use the "pre-attack" or "post-attack" equilibrium price.
2.  **Flash Loan Exclusion:**
    -   Detect transactions with massive volume > Pool Liquidity * 50% that revert to original state.
    -   *Action:* Drop these ticks.
3.  **Liquidity-Weighted Pricing:**
    -   Uniswap V3 price is $P = 1.0001^{tick}$.
    -   *Action:* Verify `liquidity > threshold` before accepting a price update. Thin liquidity = manipulation risk.
    -   *HBM Readiness:* Convert ticks to float32 prices and store as serialized NumPy arrays for TPU direct loading.

#### Step 3: Resampling (Temporal Regularization)
Block times are irregular (~12s for ETH). Mamba expects fixed steps.
-   **Method:** Time-weighted Average Price (TWAP) into 1-minute candles.
-   **Missing Data:** If no swaps in a minute, forward-fill previous price (do NOT zero-fill). Mark as `is_stale=1` feature.

#### Step 4: Feature Engineering for Bella v2m
-   **Log-Returns:** $r_t = \ln(P_t) - \ln(P_{t-1})$.
-   **Realized Volatility:** Standard deviation of tick-by-tick returns within the minute.
-   **Liquidity Depth:** Sum of liquidity `L` in active tick range (proxy for market depth).
-   **Gas Price:** Correlation with network congestion (often precedes volatility).

## 4. Bella v2m Architecture
-   **Model:** Mamba (SSM) scaled up.
    -   `d_model`: 1024 (vs 512).
    -   `n_layers`: 24 (vs 12).
    -   `d_state`: 64 (vs 16).
-   **Objective:** "Mamba Art" - Generative pre-training on the clean DEX corpus.
-   **Context:** 4096 tokens (approx 3 days of minute-data).

## 5. Next Actions
1.  Create `tools/dex_harvester/` directory.
2.  Write `extract_bigquery.sql` to fetch raw Swap events.
3.  Implement `clean_dex_data.py` (JAX/Polars based) for Steps 2-3.
4.  Draft `tools/tpu_orchestration/kaggle_resume.sh` to handle the 9-hour session cycle.
5.  Update `train_jax_fast.py` for TPU compatibility (Manual apply).