# Encoder Direction Audit — Find Why the Model Can't Distinguish BUY from SELL

You are an expert ML engineer debugging a trading model. Your job is to trace how market data flows from raw OHLCV prices into model observations, and find where **directional information (trend/sign of returns) gets destroyed**.

---

## The Bug

The model's response to oracle BUY signals and oracle SELL signals is statistically identical:

```
Bull period:
  When oracle BUYs → model: SELL 2.6%, HOLD 93.6%, BUY 2.9%
  When oracle SELLs → model: SELL 2.5%, HOLD 94.1%, BUY 2.6%

Bear period:
  When oracle BUYs → model: SELL 4.4%, HOLD 89.7%, BUY 0.4%
  When oracle SELLs → model: SELL 4.4%, HOLD 89.5%, BUY 0.4%
```

The model literally outputs the same action distribution regardless of whether prices are going up or down. This is too specific to be a capacity or training issue — it points to direction being lost in the observation encoding.

The model is profitable in crash/bear markets (learned volatility patterns, short bias) but fails in bull/rally (-24% alpha). It beats random in all 6 benchmark periods. It's not a bad model — it just can't see direction.

---

## What to Investigate

### 1. The Encoder Pipeline (PRIMARY SUSPECT)

Trace the full path from raw prices to model input:

**Files to read (in order):**
- `mcts_trader/env/jax_encoder.py` — the core encoder. Look for `jit_batch_encode_temporal`, `EncoderConfig`, and any normalization functions.
- `train_phase2.py` — `AsyncBatchPipeline._encode_jax()` (around line 525) — how the encoder is called during training.
- `tools/benchmark_easy.py` — `run_period_eval()` (around line 405) — how the encoder is called during benchmark inference.

**What to look for:**
- **Per-window normalization**: Any operation that centers or scales data within the window (z-score, min-max, mean-subtraction). This is the most likely culprit. If `obs = (prices - mean(prices_in_window)) / std(prices_in_window)`, then an uptrend window and a downtrend window with equal volatility produce IDENTICAL observations.
- **Absolute value or squaring**: Any `abs()`, `**2`, or operation that removes the sign of returns.
- **Return computation without level**: If the encoder only passes returns (price[t] - price[t-1]) without any price level information, the model sees local changes but not the global trend direction. Returns alone can encode direction IF they're not centered, but they lose "how far up we are."
- **Symmetric features**: Features that are the same for up and down movements (e.g., realized volatility, ATR, absolute volume).

### 2. The Observation Tensor Structure

The observation shape is `(B, n_channels, window_size, n_assets)` where:
- `n_channels` = 27 (market) + 7 (state channels, if enabled) = 34
- `window_size` = 64 (time steps)
- `n_assets` = 64

Read the encoder to understand what each of the 27 market channels contains. Document them. Check if ANY channel preserves the sign of returns or the direction of the trend.

### 3. The TRM Encoder (AssetTransformerEncoder)

**File**: `mcts_trader/jax_agent/trm_flax.py` — look for `AssetTransformerEncoder` class.

The TRM encoder has cross-asset attention. Check if it does any internal normalization (LayerNorm on the time dimension, batch normalization) that could erase direction after the observation encoding.

### 4. State Channels (If Enabled)

**File**: `train_phase2.py`, `AsyncBatchPipeline._prepare_numpy()` (around line 493-505) and `train_easy.py` for `build_state_channels`.

The state channels include position, PnL, hold duration, heat, capacity, confidence. Check if any of these carry directional information. The model config has `n_channels=34` which means 7 state channels are enabled. These channels reflect the model's OWN state (not market direction), so they likely don't help with direction either.

---

## What I Expect You to Find

One of these scenarios:

**Scenario A: Per-window z-score normalization**
The encoder does something like:
```python
# This kills direction:
obs = (prices - prices.mean(axis=time_dim)) / prices.std(axis=time_dim)
```
An uptrend from 100→110 and a downtrend from 100→90 both become zero-centered with similar std.

**Fix**: Add an un-normalized channel that preserves direction. Options:
- Window return: `sign(close[-1] / close[0] - 1)` as a scalar channel tiled across time
- Cumulative return from window start: `close[t] / close[0] - 1` (preserves the slope)
- Raw log-price level (or log-price relative to some reference)

**Scenario B: Returns-only encoding without level**
The encoder computes log-returns `log(close[t] / close[t-1])` and that's the main price channel. Returns DO encode direction step-by-step, but over a 64-step window, the cumulative direction might be hard to extract from individual return steps, especially after cross-asset attention mixes the signals.

**Fix**: Add a cumulative return channel that makes the trend explicit.

**Scenario C: The encoding preserves direction, but the TRM can't use it**
The observations contain directional information, but the TRM's reasoning (MLP loops, not temporal) can't aggregate sequential direction into a trend signal.

**Fix**: This is harder — the architecture may need temporal awareness. But first verify this by checking: if you print the observation tensor for a bull window vs a bear window, are they visibly different?

---

## Deliverables

1. **Document all 27 market channels** — what each one is, how it's computed, whether it preserves direction.
2. **Identify the specific normalization** that kills direction (file, function, line number).
3. **Propose a minimal fix** — the smallest change that adds direction back to the observation. Preferably adding a channel, not changing existing ones (to preserve checkpoint compatibility).
4. **If direction IS preserved in observations**, prove it by showing that the observation tensors for a bull window and a bear window are measurably different (e.g., different channel means). Then the problem is downstream (TRM can't extract direction from the encoding).

---

## System Context (for understanding the codebase)

- **Architecture**: TRM (Tiny Recursive Model) — encoder (cross-asset attention) → reasoning (MLP loops, H=6, L=8) → policy/value heads
- **Encoder**: `AssetTransformerEncoder` in `trm_flax.py` — takes (B, n_channels, window_size, n_assets) → (B, state_dim=128) latent
- **Action space**: 5 × 64 assets = 320 (0=Sell, 1=Hold, 2=Buy, 3=LimitSell, 4=LimitBuy)
- **Oracle**: greedy_lookahead_path — selects k=4 assets per step based on lookahead returns
- **Training**: Phase 2 Stage B — factored per-asset search + BC regularizer
- **Config**: `n_channels=34` (27 market + 7 state), `window_size=64`, `n_assets=64`
- Observations are encoded by `jit_batch_encode_temporal()` in `mcts_trader/env/jax_encoder.py`
- The encoder is FROZEN during Phase 2 training (trained in Phase 1 on oracle BC targets)
- JAX/Flax codebase, all computations in JAX
