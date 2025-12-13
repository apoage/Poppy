# Gemini Notes

## Current Understanding (as of 2025-12-09)

*   **Project Goal:** JAX MuZero Trading Agent with TRM architecture for multi-asset cryptocurrency trading.

### Performance Achieved

*   **Self-play per episode:** Improved from 9 minutes to 1.5-2.5 seconds.
*   **Steps per episode:** Improved from 1 to 200.
*   **Database step time:** Improved from 500ms to 0ms (pre-computed).
*   **Encoding time:** Improved from ~200ms to ~10ms.

### Known Issues (v1.0 from DEVELOPMENT_DIARY.md)

3.  **No Correlation Features:** Removed for speed, may impact strategy quality.

### Important Notes & Resolved Issues

*   **NaN Loss:** Resolved as of 2025-12-09 by using `int32` for saving values in the JAX pipeline.
*   **Paper Trading Results:** The previously noted "suspicious" doubling of portfolio value was due to critical bugs related to multi-currency unit confusion and position sizing without FX conversion, which have been fixed as of 2025-12-11. The model's multi-currency FX strategies are now correctly accounted for, showing realistic performance.
*   **WebSocket Stability:** Auto-reconnect with exponential backoff and stop-loss/take-profit system implemented as of 2025-12-11.

### Next Steps (from DEVELOPMENT_DIARY.md & Architectural Gaps)

*   **Future (Architectural Enhancements):**
    1.  Add gated residuals to TRM.
    2.  Add cross-asset attention.
    3.  Implement Differential Sharpe reward.
    4.  Abstract API layer (beyond Kraken).
    5.  Consider adding back lightweight correlations.
    6.  GAF/MTF encoding.
    7.  ReZero reanalysis.
    8.  LR scheduling.
    9.  Position sizing head.

## TODO / Technical Debt

### HIGH Priority (Breaking/Critical)
- [ ] **Gym → Gymnasium migration**: Gym is unmaintained since 2022 and doesn't support NumPy 2.0; migration is partly done but Gym is still called in some places.
- [ ] **Orbax transformations API**: Warning about deprecated API, potential checkpoint loading failure.

### MEDIUM Priority (Stability)
- [ ] **tensorflow.io.gfile shim**: Install tensorflow or use alternative for GCS paths.
- [ ] **API abstraction layer**: Continue work on abstracting the API layer beyond Kraken (e.g., Binance, Coinbase).

### LOW Priority (Quality of Life / Audit)
- [ ] **TPU backend warning**: Suppress or handle gracefully the warning about TPU backend when running on GPU/CPU.
- [ ] **Trading loop time limit review**: Audit the already implemented trading loop time limit for any remaining bugs or areas for improvement.

## TODO:

3.  Create Architectural Plan (`agents/gemini/plan.md`) based on the "Next Steps" and "Architectural Gaps" identified.
