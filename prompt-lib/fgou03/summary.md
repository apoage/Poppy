# Foxglove Data Pipeline Audit Summary

## Critical Bugs (Must Fix Before Training)
1. **BUG-1.1 & BUG-1.2: Zero-Asset Hallucination Leakage**. Historical data contains up to 33 columns of all-zeros for assets that did not exist in early periods. The static shape check `window.shape[1] < n_assets` in `train_easy.py` never triggers because the array is always `(T, 64)`. This causes 100% agreement on zero-assets (trivial HOLDs) and destroys validation generalization. Must add dynamic zero-column detection and cyclic padding inside `_generate_prices_historical()`.

## Important Issues (Fix Soon)
1. **BUG-4.1: Benchmark Rejection Sampling Fallback**. The rejection sampler loops 20 times to avoid overlapping with benchmark periods, but simply accepts an overlapping sample if it hits the maximum attempt limit. It should instead raise an error or re-try to ensure zero benchmark leakage.
2. **BUG-2.1: Oracle vs Model P&L Normalization**. `benchmark_oracle_vs_model.py` computes raw sum of P&L without dividing by the number of active positions. This incorrectly represents unbounded leverage (e.g., 4x leverage for `k=4`) and makes P&L magnitudes structurally uncomparable between the model and the oracle.

## Minor Issues (Nice to Have)
1. None directly observed, though keeping track of AR performance bottlenecks is highly recommended.

## Verified Correct
1. **AR Blend Logic**: The batched AR rollout correctly ingests and autoregresses over historical prices, and orchestrator caps propagate smoothly.
2. **Data Source Isolation**: `_refill_buffer` uses rigorous `if/elif` exclusion, cleanly ensuring historical windows are never silently mixed with synthetic or diffusion windows in the same batch.
3. **Rigid Time-Grid**: The `get_window` mapping effectively transforms sparse historical ticks into the continuous 1-minute discrete sequence that the TRM requires, perfectly managing data gaps through nearest-neighbor forward-filling.