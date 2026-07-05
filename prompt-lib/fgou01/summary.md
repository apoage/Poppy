# Foxglove Data Pipeline Audit Summary

**Auditor**: Claude Opus 4.6 | **Date**: 2026-03-03 | **Output**: `prompt-lib/fgou01/`

## Scorecard

| Part | Title | Status | Bugs |
|------|-------|--------|------|
| 1 | Zero-Asset Handling | **FAIL** | 4 (1 critical, 1 medium, 2 low) |
| 2 | Oracle Math Consistency | **NEEDS_FIX** | 4 (3 high, 1 low) |
| 3 | AR Blend with Historical | **PASS** | 0 |
| 4 | Benchmark Period Exclusion | **NEEDS_FIX** | 1 (low) |
| 5 | Data Source Selection Chain | **NEEDS_FIX** | 1 (low) |
| 6 | Price Window Shape & Content | **NEEDS_FIX** | 2 (1 medium, 1 low-medium) |

**Total: 12 bugs** — 1 critical, 3 high, 2 medium, 6 low

---

## Critical Bugs (Must Fix Before Training)

1. **BUG-1.1** — `get_window()` returns zero columns for non-existent assets (`history.py:616`). Up to 34 of 64 columns are all-zeros in early periods (pre-2020). Zero columns produce `-inf` log returns, degenerate encoder features, and wasted oracle action slots. The tile safety net in `train_easy.py:1235` is dead code (checks column count, not column content). This is the root cause of the 100% → 65% agreement drop.

## Important Issues (Fix Soon)

2. **BUG-2.1** — Benchmark oracle PnL not normalized by `k_count` (`benchmark_oracle_vs_model.py:403`). Training oracle divides by `k_count`; benchmark does not. All benchmark PnL/alpha numbers are inflated by ~k×. A prior "BUG-9 fix" intentionally removed the division, but that was the wrong fix direction.

3. **BUG-2.2** — Benchmark model PnL not normalized by active position count (`benchmark_oracle_vs_model.py:468`). Same unnormalized sum problem on the model side.

4. **BUG-2.3** — `benchmark_easy.py` inherits the same unnormalized PnL from `simulate_model_pnl_detailed` (line 301) and `_simulate_pnl_vectorized` (line 347).

5. **BUG-6.2** — Bella `seq_length` may not match `_synth_n_steps` (489). If Bella outputs 400 steps, episodes are silently shorter with reduced oracle lookahead. (`setup.py:598,651`)

## Minor Issues (Nice to Have)

6. **BUG-1.2** — Dead tile safety net in `_generate_prices_historical()` (`train_easy.py:1235`). Tests `shape[1] < 64` which is always False because `cache_data` pre-allocates 64 columns.

7. **BUG-1.3** — npz cache lacks asset availability metadata (`history.py:143-179`). Forces runtime scanning to detect zero columns.

8. **BUG-1.4** — `benchmark_easy.py:224` warns about entire-column zeros but doesn't fix them.

9. **BUG-2.4** — Misleading "BUG-9 fix" comment (`benchmark_oracle_vs_model.py:402`).

10. **BUG-4.1** — Rejection sampling fallback uses contaminated data (`train_easy.py:1229`). When all 20 attempts overlap benchmark periods, the last (overlapping) sample is used. Probability < 10^-22 but a correctness defect.

11. **BUG-5.1** — `_use_gpu_synth` flag semantically incorrect when historical is set (`train_easy.py:546`). Functionally harmless due to priority chain, but maintenance hazard.

12. **BUG-6.1** — Leading all-zero columns for late-listing assets cause extreme return spikes at zero→price transitions (`history.py:368-382`). Partially mitigated by `min_required_assets` filter and return clipping.

## Verified Correct

1. **AR blend** receives identical historical prices for both oracle and model-state rollout (Part 3)
2. **`--ar-max-blend`** correctly passed through for all branch types, including historical (Part 3)
3. **AR cap pre-loading** bug was already fixed — `dojo_state.json` loaded before first branch launch (Part 3)
4. **Benchmark period date parsing** and overlap detection formula are correct (Part 4)
5. **Data source priority chain** in `_refill_buffer()` correctly isolates historical from Bella/synthetic (Part 5)
6. **Validation generator** correctly excludes historical data (Part 5)
7. **`_synth_n_steps = 489`** consistently used across all generation paths (Part 6)
8. **Forward-fill via `searchsorted`** correctly handles gaps; edge cases are safe (Part 6)
9. **Historical prices** are in raw USD, matching encoder expectations (Part 6)

---

## Recommended Fix Order

| Priority | Bug | Effort | Impact |
|----------|-----|--------|--------|
| 1 | BUG-1.1 (zero columns) | ~30 min | Fixes root cause of agreement drop |
| 2 | BUG-2.1+2.2+2.3 (benchmark normalization) | ~20 min | Makes benchmark numbers meaningful |
| 3 | BUG-6.1 (backward-fill leading zeros) | ~15 min | Eliminates extreme return spikes |
| 4 | BUG-6.2 (Bella seq_length validation) | ~10 min | Ensures consistent episode length |
| 5 | BUG-4.1 (rejection fallback) | ~5 min | Defense-in-depth |
| 6 | BUG-5.1 (gpu_synth flag) | ~2 min | Maintenance cleanup |
| 7 | BUG-1.2+1.3+1.4+2.4 (dead code, comments) | ~15 min | Code hygiene |
