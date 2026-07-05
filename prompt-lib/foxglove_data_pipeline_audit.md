# Foxglove Data Pipeline Audit: Historical Integration & Math Validation

> **OUTPUT_DIR**: `{{OUTPUT_DIR}}`
> Replace `{{OUTPUT_DIR}}` with a unique folder path before use (e.g. `prompt-lib/out-fda01`).
> Each reviewer gets their own folder. Do NOT read other folders under `prompt-lib/out-*`.

You are auditing the data pipeline for Foxglove (easy-mode training). The system recently added **historical market data** (real Kraken OHLCV) as a data source alongside Bella (diffusion-generated) and synthetic (GPU sine waves). There are confirmed bugs. Your job is to trace every data path end-to-end and find all math/logic errors.

## Why This Exists

Training showed **100% train agreement dropping to ~65% validation** within 3 epochs on historical data. Root cause: 33 out of 64 asset columns are all-zeros in early time periods (pre-2020). The model learns to predict HOLD on zeros (trivial) which inflates agreement. The zero-asset duplication that exists in `setup.py` (dojo_v4) was NOT correctly ported to the Foxglove pipeline.

---

## Audit Scope

### Part 1: Zero-Asset Handling (CRITICAL)

**The Rule**: There must NEVER be zero-valued asset columns in training data. When fewer than 64 assets have data, active assets must be **cyclically duplicated** to fill all 64 slots.

**Reference implementation** (correct): `mcts_trader/training/setup.py:408-413`
```python
if prices.shape[1] < N_ASSETS:
    n_loaded = prices.shape[1]
    pad_indices = [i % n_loaded for i in range(N_ASSETS)]
    prices = prices[:, pad_indices]
```

**Audit these files for correct zero-asset handling:**

| File | What to Check | Known Issue |
|------|---------------|-------------|
| `mcts_trader/data/history.py` | `get_window()` returns `cache_data[indices, :, 3]` — does it handle zero columns? | Returns all 64 cols including zeros |
| `mcts_trader/data/history.py` | `_try_load_cache()` — does npz loading detect/handle partial asset coverage? | Loads raw cache, no zero detection |
| `train_easy.py` | `_generate_prices_historical()` lines ~1220-1242 — tile safety net at 1235 | Dead code: shape[1] always == 64 |
| `tools/benchmark_easy.py` | `load_benchmark_prices()` lines ~197-205 | Has tile logic, but relies on actual_assets < n_assets |

**Questions to answer:**
1. Where should zero-column detection + cyclic duplication happen? In `get_window()` itself? In `_generate_prices_historical()`? Both?
2. Should the npz cache store the asset availability mask so we know which columns are real vs zero at any timestamp?
3. The DB has ~700 assets total but only 64 are loaded. How are the 64 chosen? Are they the "top 64 by volume"? What happens when an asset in the top-64 list didn't exist in 2018?

### Part 2: Oracle Math Consistency

**The Rule**: Oracle PnL calculation must be identical between training and benchmarking. Currently it is NOT.

| Component | PnL Formula | File:Line |
|-----------|-------------|-----------|
| Training oracle | `step_pnl = sum(abs(returns) * mask) / k_count` | `path_solver.py:130-131` |
| Benchmark oracle | `step_pnl = sum(abs(returns) * mask)` (NO `/k_count`) | `benchmark_oracle_vs_model.py` (via `simulate_oracle_pnl`) |
| Benchmark model | `step_pnl = sum(position * returns)` (NO `/n_active`) | `benchmark_oracle_vs_model.py` (via `simulate_model_pnl`) |

**Questions to answer:**
1. Read `tools/benchmark_oracle_vs_model.py` — find `simulate_oracle_pnl` and `simulate_model_pnl`. Confirm the normalization mismatch.
2. What is `k_count` in training? It's the number of selected assets (sum of the binary mask). Without `/k_count`, oracle PnL scales with the number of assets selected — more selections = higher PnL regardless of quality.
3. Should benchmark normalize by k_count to match training, or should training NOT normalize? Which is the correct financial interpretation?
4. Does the model PnL also need normalization? The model outputs positions for ALL 64 assets — should it divide by the number of non-zero positions?

### Part 3: AR (Autoregressive) Blend with Historical Data

**The Rule**: AR blend controls what fraction of episodes use model-predicted state vs oracle state. At AR=0.75, 75% of episodes should use model state rollout.

**Audit these:**
1. `train_easy.py` — `_refill_buffer()` around line 1253: When `historical_market` is set, does AR rollout still work? The model-state trajectory needs prices to encode — does it get the historical prices or synthetic?
2. `train_easy.py` — `_build_model_state_trajectory_batched()`: This builds the model's own state trajectory. Does it correctly receive the same historical price window that the oracle used?
3. `tools/orchestration/e-dojo_v1.py` — `launch_branch()`: Is `--ar-max-blend` correctly passed through for historical branches?
4. Does the AR cap from `dojo_state.json` get loaded BEFORE the first branch launch? (Known bug: it was loaded after, so first launch used stale value.)

### Part 4: Benchmark Period Exclusion

**The Rule**: 6 benchmark periods (7 days each, 42 days total) in `config/benchmark_periods.json` must NEVER appear in training data.

**Audit:**
1. `train_easy.py` — `_load_benchmark_periods()` and `_overlaps_benchmark()`: Do they correctly parse the JSON dates and convert to timestamps?
2. Is the exclusion check using the right timestamp comparison? `start_ts < bp_end and end_ts > bp_start` (overlap detection).
3. Does the rejection sampling (20 attempts max) have a fallback? What happens if all 20 attempts overlap? Does it use the last (overlapping) sample?
4. With 42 days excluded out of ~4380 days of data (12 years), rejection rate should be <1%. Verify this is true.

### Part 5: Data Source Selection Chain

**The Rule**: When `historical_market` is set, it should be the primary data source. Other sources (Bella, synthetic) should not be mixed in.

**Audit `_refill_buffer()` in `train_easy.py`:**
1. Read the source selection chain. Is historical always chosen when `self.historical_market is not None`?
2. Is there any path where historical data gets mixed with Bella data in the same batch?
3. For Branch B (Bella stabilizer): When historical_market is NOT set but Bella is, does Bella take priority over synthetic?
4. Is `use_secondary` logic interacting with historical in unexpected ways?

### Part 6: Price Window Shape & Content Validation

**The Rule**: Every price window fed to the encoder must be shape `(T, 64)` where T = `episode_length + window_size + oracle_lookahead + 5` = 489 timesteps, and NO column should be all-zeros.

**Audit:**
1. What is `self._synth_n_steps` set to? Is it 489?
2. Does `get_window()` always return exactly `(489, 64)`? What happens at the edges of available data?
3. `get_window()` uses forward-fill via `searchsorted(side='right') - 1`. Does this correctly handle gaps? What if the first timestamp has no data?
4. Are historical prices in the correct unit (raw USD, not normalized)? The encoder expects raw close prices.

---

## Output Format

For each Part (1-6), write findings to `{{OUTPUT_DIR}}/part_N.md` with:

```markdown
# Part N: [Title]

## Status: [PASS | FAIL | NEEDS_FIX]

## Findings
- [Finding 1 with file:line reference]
- [Finding 2...]

## Bugs Found
- **BUG-N.1**: [Description, severity, file:line]
- **BUG-N.2**: ...

## Recommended Fixes
- [Fix 1 with code snippet]
- [Fix 2...]
```

Then write a summary to `{{OUTPUT_DIR}}/summary.md`:

```markdown
# Foxglove Data Pipeline Audit Summary

## Critical Bugs (Must Fix Before Training)
1. ...

## Important Issues (Fix Soon)
1. ...

## Minor Issues (Nice to Have)
1. ...

## Verified Correct
1. ...
```

---

## Key Files to Read

| Priority | File | What to Check |
|----------|------|---------------|
| 1 | `mcts_trader/data/history.py` | `get_window()`, `_try_load_cache()`, constructor |
| 1 | `train_easy.py` | `_generate_prices_historical()`, `_refill_buffer()`, `_build_model_state_trajectory_batched()`, `main()` historical setup |
| 1 | `mcts_trader/training/setup.py` | Lines 408-413: reference cyclic padding |
| 2 | `tools/benchmark_easy.py` | `load_benchmark_prices()`, benchmark runner |
| 2 | `tools/benchmark_oracle_vs_model.py` | `simulate_oracle_pnl`, `simulate_model_pnl` |
| 2 | `mcts_trader/jax_agent/path_solver.py` | Lines 130-131: oracle PnL formula |
| 3 | `tools/orchestration/e-dojo_v1.py` | `launch_branch()`, `launch_dual_branches()`, AR cap handling |
| 3 | `config/benchmark_periods.json` | Period definitions |
| 3 | `easy_config.py` | `episode_length`, `window_size`, `oracle_lookahead` |
