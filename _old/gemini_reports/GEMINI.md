# Gemini Notes

## Current Understanding (as of 2025-12-16)

*   **Project Goal:** JAX MuZero Trading Agent with TRM architecture for multi-asset cryptocurrency trading.

### Performance Achieved

*   **Self-play per episode:** Improved from 9 minutes to 1.5-2.5 seconds.
*   **Steps per episode:** Improved from 1 to 200.
*   **Database step time:** Improved from 500ms to 0ms (pre-computed RAM cache).
*   **Encoding time:** Improved from ~200ms to ~10ms.
*   **FX Rate Lookup:** Instant (O(1) RAM lookup) with <0.3% error vs Kraken Ticker.

### Resolved Issues

*   **NaN Loss:** Resolved as of 2025-12-09 by using `int32` for saving values in the JAX pipeline.
*   **Paper Trading Results:** Addressed unit confusion and position sizing bugs (2025-12-11).
*   **FX Rate Fallback / "Turtle" Collapse:**
    *   **Status:** **RESOLVED** (2025-12-16).
    *   **Fix:**
        1.  **Database:** Created `fx_rates` table (`tools/build_fx_database.py`) with ~860k records using Weighted Consensus from crypto bridges.
        2.  **Validation:** Verified against Kraken Ticker (`tools/validate_fx_rates.py`) with ~0.24% deviation.
        3.  **Integration:** `HistoricalMarket` preloads `fx_rates` into RAM. `FastTradingEnv` fetches authoritative rates. `PortfolioManager` uses simplified currency keys.
        4.  **Verification:** Smoke test (50 steps) confirmed smooth execution and reward generation.

### Active Optimizations

*   **Training Controls:** Added `--batch-size` (dynamic, default 256) and `--confidence-threshold` (gate-keeping) to `train_jax_fast.py`.
*   **Logging:** Added `B/H/S` (Buy/Hold/Sell) counts to training logs to easily detect "Turtle" behavior.

## TODO / Technical Debt

### HIGH Priority (Breaking/Critical)
- [ ] **Verify Training Stability:** Run a longer training session (e.g., 50-100 epochs) to confirm the agent learns effectively with the new FX rates and does not revert to "Turtle" behavior.
- [ ] **Gym → Gymnasium migration**: Gym is unmaintained since 2022 and doesn't support NumPy 2.0; migration is partly done but Gym is still called in some places.

### MEDIUM Priority (Stability)
- [ ] **Orbax transformations API**: Warning about deprecated API.
- [ ] **tensorflow.io.gfile shim**: Install tensorflow or use alternative for GCS paths.
- [ ] **API abstraction layer**: Continue work on abstracting the API layer beyond Kraken.

### LOW Priority (Quality of Life / Audit)
- [ ] **TPU backend warning**: Suppress or handle gracefully the warning about TPU backend when running on GPU/CPU.
