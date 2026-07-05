# TASK_692: Fix State Channels — Model's Own State, Not Oracle's

**Status**: DONE — implemented, deployed, training active
**Priority**: 0 (CRITICAL — was fundamentally broken)
**Depends on**: TASK_689 Phase A (DONE), TASK_693 position semantics (DONE)
**Supersedes**: TASK_691 oracle entropy (disabled, code kept)

## Problem

State channels are auxiliary context inputs that give the model information about its trading state (positions, PnL, heat, duration, capacity). During training they were computed from the **oracle's** trajectory. At inference they are computed from the **model's** trajectory. The distributions are fundamentally different.

**Was (broken):**
```
Training:   oracle_selections → build_state_channels() → model input (ch 27-33)
Inference:  model_actions → build_state_channels() → model input (ch 27-33)
```

The model only trained on the **diagonal** of the (state, target) space — where state matches oracle. At inference, every model mistake pushes it **off-diagonal** into states it has zero training signal for.

**Pre-fix evidence:**
- Training (oracle state): 83.6% balanced accuracy
- Benchmark (model's own state): **33.8%** — below random (50%)
- Old model without state channels: 58.6% on same benchmark
- Adding oracle-derived state channels made benchmark accuracy **worse**

## What Was Implemented

### 1. `_build_model_state_trajectory()` — `train_easy.py:174-305`

JIT'd `lax.scan` autoregressive rollout. Each timestep:
- Builds state channels from carry (position, pnl, steps, h_fast, h_slow)
- Tiles state across window, concatenates with market obs → (34, ws, n_assets)
- Runs `model.apply()` with `stop_gradient` → policy logits
- Extracts action via `argmax`, computes model confidence from softmax
- TASK_693: Incremental position stepping (Buy=step toward +1, Sell=step toward -1) matching `benchmark_easy.py` inference semantics
- Updates state carry (PnL, heat, duration, capacity)

Returns `(T, 6, n_assets)` state trajectory + `(T, n_assets)` confidence.

### 2. `set_model_params()` — `train_easy.py:1016-1030`

Generator-trainer coupling. Called at epoch start (`train_epoch_on_the_fly()` line 1272). Stores params + model apply fn, increments param version counter, triggers partial buffer refresh.

### 3. Partial FIFO buffer refresh — `train_easy.py:1038-1073`

DAgger-style refresh. On each `set_model_params()` call:
- Finds oldest episodes by param version (FIFO ordering)
- Marks `state_refresh_fraction` (default 20%) of buffer as exhausted
- Exhausted episodes regenerated on next `generate_batch()` call with fresh model params

### 4. Generator integration — `train_easy.py:552-581`

In `_generate_episode_data()`:
```python
use_model_rollout = (use_state_channels and autoregressive_state
                     and self._model_params is not None)
if use_model_rollout:
    # Encode market obs (27ch), then roll model through episode
    state_channels, confidence = self._model_state_rollout(
        observations, log_returns, self._model_params, self._model_apply_fn, ...
    )
```
Falls back to oracle state (TASK_691 path) when `_model_params` is None (first epoch before trainer provides params).

Same pattern in `_generate_batch_episodes()` (line 781+).

### 5. Config — `easy_config.py:84-86`

```python
autoregressive_state: bool = True       # Use model-state rollout instead of oracle state
state_refresh_fraction: float = 0.2     # Fraction of buffer to refresh per epoch (20%)
```

TASK_691 config kept but disabled: `oracle_entropy: bool = False` (line 79).

### 6. No scheduled sampling

No oracle warmup. Model starts with random state (noise in channels 27-33), learns market channels first, state channels become informative as model improves. Implicit ramp — no forgetting risk.

## Files Modified

| File | Lines | Change |
|------|-------|--------|
| `train_easy.py:174-305` | 130 | `_build_model_state_trajectory()` — JIT'd lax.scan rollout |
| `train_easy.py:375-377` | 3 | Generator init: `_model_params`, `_model_state_rollout` |
| `train_easy.py:552-581` | 30 | `_generate_episode_data()` model rollout path |
| `train_easy.py:781-788` | 8 | `_generate_batch_episodes()` model rollout path |
| `train_easy.py:1016-1073` | 57 | `set_model_params()` + `_partial_refresh_buffer()` |
| `train_easy.py:1271-1273` | 3 | Epoch start: call `set_model_params()` |
| `train_easy.py:1530-1531` | 2 | Val: call `set_model_params()` on val_generator |
| `easy_config.py:84-86` | 3 | `autoregressive_state`, `state_refresh_fraction` |
| `easy_config.py:79` | 1 | `oracle_entropy: False` (disabled) |

## Completion Criteria

- [x] State channels computed from model's own actions (not oracle)
- [x] Generator has `set_model_params()` for inference rollout
- [x] Rollout loop JIT-compiled via `lax.scan`
- [x] Partial FIFO buffer refresh working (20% per epoch default)
- [x] TASK_691 oracle entropy disabled
- [ ] Real-data benchmark > 50% (above random) — **pending benchmark run**
- [ ] Train-inference gap < 10 points — **pending benchmark run**

## Review Provenance

Distilled from 3 independent model reviews (2026-02-22):
- **Claude (adversarial)**: Performance math (36% overhead), Option A (no warmup) endorsed, Option B (heavier entropy) withdrawn, forgetting analogy retracted
- **GLM (engineering)**: Cold-start analysis, progressive FIFO refresh design, JIT recompilation risk, DAgger framing
- **Gemini (architectural)**: Off-diagonal coverage framing (key insight), Ghost Target analysis (resolved — oracle is market-optimal, mismatch is learning signal), episode-level sampling, limit-order target distribution flag

### Implementation note from Gemini (check during impl)
If oracle only outputs positions {-1,0,+1} but policy head has 5 actions including LimitBuy/LimitSell, verify target distribution spreads probability across equivalent actions. Model shouldn't be penalized for using limit orders when oracle used market orders.

---

*Implemented: 2026-02-22*
*Training active with autoregressive state*
