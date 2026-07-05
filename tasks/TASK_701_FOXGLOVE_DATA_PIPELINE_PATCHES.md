# TASK_701: Foxglove Data Pipeline Patches (Audit Consolidation)

## Status: DONE
## Priority: Critical
## Created: 2026-03-03
## Source: 3-way audit — `prompt-lib/fgou01`, `prompt-lib/fgou02`, `prompt-lib/fgou03`

## Overview

Three independent auditors reviewed the Foxglove historical data pipeline end-to-end. This task consolidates their findings into a single patch list, ordered by severity and auditor consensus.

**Auditor agreement key**: 3/3 = unanimous, 2/3 = majority, 1/3 = single auditor (verified plausible)

---

## P0 — CRITICAL (Must fix before training)

### PATCH-1: Zero-column cyclic duplication in historical windows
**Consensus: 3/3** | Files: `mcts_trader/data/history.py`, `train_easy.py`

**Problem**: `get_window()` returns `(T, 64)` where up to 33 columns are all-zeros for assets that didn't exist in early periods (pre-2020). The model learns trivial HOLD predictions on zeros → inflates training agreement to 100%, drops to ~65% on validation. This is the root cause of the train/val regression.

**Why it happens**: `_preload_to_ram()` pre-allocates `cache_data` with shape `(n_rows, 64, 5)`. Forward-fill only works forward from the first non-zero entry per asset. Assets listed after 2020 have no data in 2018 rows → remain zero.

**Dead safety net**: `train_easy.py:1235` checks `window.shape[1] < 64` — always False because cache has 64 columns. The bug is zero *content*, not missing *columns*.

**Fix** — Replace dead safety net in `_generate_prices_historical()` (`train_easy.py:1233-1237`):

```python
# OLD (dead code):
if window.shape[1] < self.config.n_assets:
    reps = (self.config.n_assets // window.shape[1]) + 1
    window = np.tile(window, (1, reps))[:, :self.config.n_assets]

# NEW: Detect zero columns and cyclically replace with live ones
live_mask = np.any(window > 0, axis=0)  # (n_assets,) bool
n_live = int(np.sum(live_mask))
if n_live < window.shape[1] and n_live > 0:
    live_indices = np.where(live_mask)[0]
    pad_indices = np.array([live_indices[i % n_live] for i in range(window.shape[1])])
    window = window[:, pad_indices]
elif n_live == 0:
    logger.warning("get_window returned all-zero prices, skipping episode")
    continue  # skip this episode in the for loop
```

**Optional long-term**: Also add zero-column handling inside `get_window()` itself so all callers (training + benchmark) benefit. But the immediate fix in `_generate_prices_historical()` is sufficient.

---

## P1 — HIGH (Fix soon, affects benchmark reporting)

### PATCH-2: Benchmark PnL normalization
**Consensus: 3/3** | **Already tracked as TASK_700** — do not duplicate work.

Cross-reference: TASK_700 covers the full fix for `simulate_oracle_pnl`, `simulate_model_pnl`, `simulate_model_pnl_detailed`, and `_simulate_pnl_vectorized`. All three auditors confirmed the diagnosis. Training is NOT affected — only benchmark reporting.

Summary of what TASK_700 fixes:
- `benchmark_oracle_vs_model.py:403` — add `/k_count` to oracle PnL
- `benchmark_oracle_vs_model.py:468` — add `/n_active` to model PnL
- `benchmark_easy.py:301` — same for `simulate_model_pnl_detailed`
- `benchmark_easy.py:347` — same for `_simulate_pnl_vectorized`
- `benchmark_oracle_vs_model.py:402` — fix misleading "BUG-9 fix" comment

---

## P2 — MEDIUM (Fix for correctness / robustness)

### PATCH-3: Backward-fill leading zeros in `_preload_to_ram()`
**Consensus: 1/3 (fgou01)** — verified plausible, overlaps with PATCH-1 root cause

**Problem**: Forward-fill in `history.py:368-382` only fills gaps after the first non-zero entry. Assets listed later (e.g., SOL/USD, 2020) have leading zeros. When a window overlaps the zero→price transition, the encoder sees `log(price / 1e-8) = 23.0` (clipped to 10.0) — an extreme return spike.

**Mitigation**: PATCH-1's cyclic duplication removes the zero columns entirely, so the encoder never sees the transition. However, backward-fill is still good defense-in-depth.

**Fix** — Add after the forward-fill loop in `history.py` (`_preload_to_ram()`):

```python
# Backward-fill leading gaps (assets listed late have zeros at start)
back_fills = 0
for c_idx in range(n_cols):
    first_valid = None
    for r_idx in range(n_rows):
        if self.cache_data[r_idx, c_idx, 3] > 0:
            first_valid = self.cache_data[r_idx, c_idx].copy()
            break
    if first_valid is not None:
        for r_idx in range(n_rows):
            if self.cache_data[r_idx, c_idx, 3] > 0:
                break
            self.cache_data[r_idx, c_idx] = first_valid
            back_fills += 1
logger.info(f"Backward-filled {back_fills:,} leading price gaps")
```

### PATCH-4: Rejection sampling fallback uses contaminated data
**Consensus: 2/3 (fgou01, fgou03)** | File: `train_easy.py:1221-1229`

**Problem**: When all 20 rejection attempts overlap benchmark periods, the `for/else` construct falls through and uses the last (overlapping) sample. Benchmark data leaks into training.

**Practical impact**: Probability < 10^-22 under normal conditions. But if a schedule period perfectly aligns with a benchmark, fgou03 notes it could fail on every episode.

**Fix** — Change fallback to skip the episode:

```python
for _ in range(n_episodes):
    for attempt in range(20):
        idx = self.rng.integers(range_min, max_start_idx)
        start_ts = int(market.valid_timestamps[idx])
        end_ts = start_ts + window_seconds
        if not self._overlaps_benchmark(start_ts, end_ts):
            break
    else:
        logger.warning("TASK_701: All 20 rejection attempts hit benchmark, skipping episode")
        continue  # skip this episode — buffer refill loop self-corrects

    window = market.get_window(start_ts, seq_len)
```

### PATCH-5: Validate Bella output length matches `_synth_n_steps`
**Consensus: 1/3 (fgou01)** — verified plausible

**Problem**: Bella's `GeneratorConfig.seq_length` may default to 400, not 489. If mismatched, Bella episodes are silently shorter with reduced oracle lookahead. No crash, but fewer training timesteps per episode.

**Fix** — In `_refill_buffer()` Bella path, after generation:

```python
for i in range(n_to_refill):
    p = np.array(prices_batch[i])
    if p.shape[0] < self._synth_n_steps:
        pad_len = self._synth_n_steps - p.shape[0]
        p = np.concatenate([p, np.tile(p[-1:], (pad_len, 1))], axis=0)
    elif p.shape[0] > self._synth_n_steps:
        p = p[:self._synth_n_steps]
    prices_list.append(p)
    levels.append(-1)
```

---

## P3 — LOW (Cleanup / defense-in-depth)

### PATCH-6: Fix `_use_gpu_synth` flag semantics
**Consensus: 1/3 (fgou01)** | File: `train_easy.py:546`

**Problem**: `self._use_gpu_synth = (bella_generator is None)` is True even when `historical_market` is set. Functionally harmless (priority chain prevents wrong path), but semantically wrong and a maintenance hazard.

**Fix**:
```python
self._use_gpu_synth = (bella_generator is None and historical_market is None)
```

### PATCH-7: Add asset availability metadata to npz cache
**Consensus: 2/3 (fgou01, fgou02)** | File: `mcts_trader/data/history.py`

**Problem**: npz cache stores `cache_data` and `asset_ids` but not per-asset first-valid timestamps. Downstream code must scan data to detect zero columns.

**Fix**: After forward/backward-fill in `_preload_to_ram()`, compute and store:
```python
self.asset_first_valid_ts = np.zeros(n_cols, dtype=np.int64)
for c in range(n_cols):
    nonzero = np.nonzero(self.cache_data[:, c, 3])[0]
    if len(nonzero) > 0:
        self.asset_first_valid_ts[c] = self.valid_timestamps[nonzero[0]]
```
Include in `_save_cache()` / `_try_load_cache()`.

### PATCH-8: Fix benchmark_easy.py entire-column zero handling
**Consensus: 1/3 (fgou01)** | File: `tools/benchmark_easy.py:224`

**Problem**: Warns about entire-column zeros but doesn't fix them. Should apply same cyclic duplication as PATCH-1.

### PATCH-9: Remove dead `_generate_prices_bella()` method
**Consensus: 1/3 (fgou01)** | File: `train_easy.py:1114-1122`

Never called — superseded by inline Bella generation in `_refill_buffer()`. Dead code cleanup.

### PATCH-10: Remove dead `_generate_episode_data()` method
**Consensus: 1/3 (fgou01)** | File: `train_easy.py:636`

Never called — superseded by `_generate_batch_episodes()`. Contains duplicated AR blend logic that could drift.

---

## Verified Correct (No Action Needed)

All three auditors confirmed these are bug-free:

| Area | Verdict | Notes |
|------|---------|-------|
| AR blend with historical data | **PASS 3/3** | Same prices flow to oracle and AR rollout |
| `--ar-max-blend` passing through e-dojo | **PASS 3/3** | AR cap injected in `launch_branch()` common path |
| AR cap stale-load bug | **FIXED 3/3** | Pre-loaded before first launch at `e-dojo_v1.py:1701-1705` |
| Benchmark date parsing + overlap formula | **PASS 3/3** | Correct UTC conversion, standard interval overlap |
| Data source priority chain (`if/elif`) | **PASS 3/3** | Historical → Secondary → Bella → GPU synth → CPU synth |
| No cross-source batch mixing | **PASS 3/3** | Each episode tagged with source; no leakage |
| Validation excludes historical | **PASS 3/3** | Intentional — tests generalization |
| `_synth_n_steps = 489` | **PASS 3/3** | Consistent across all generation paths |
| Forward-fill via `searchsorted` | **PASS 3/3** | Handles gaps correctly; edge cases safe |
| Historical prices in raw USD | **PASS 3/3** | Encoder normalizes internally |

---

## Patch Execution Order

```
1. PATCH-1  (zero-column cyclic duplication)     — unblocks training
2. PATCH-3  (backward-fill leading zeros)         — defense-in-depth for PATCH-1
3. PATCH-4  (rejection sampling fallback)         — correctness fix
4. TASK_700 (benchmark PnL normalization)         — separate task, parallel
5. PATCH-5  (Bella seq_length validation)         — if using Bella
6. PATCH-6..10 (cleanup)                          — batch together
```
