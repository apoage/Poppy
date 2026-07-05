# TASK_703: True OHLCV Pipeline (End Fake Channels)

**Status**: DONE
**Priority**: 2 (after TASK_698 historical training is working)
**Depends on**: TASK_698 (historical data integration proves value)
**Blocks**: Real trading accuracy, Bella fidelity, distribution consistency

---

## Problem

The entire Foxglove pipeline currently fakes OHLCV data:

**Encoder** (`jax_encoder.py:456-459`):
```python
prices_T = price_window.T          # close prices only
volumes_T = jnp.ones_like(prices_T)  # FAKE: volumes are all 1.0
# highs = close, lows = close       # FAKE: passed as prices_T twice
return _compute_all_channels_temporal(prices_T, volumes_T, prices_T, prices_T, config)
```

**Impact**: 8 of 27 encoder channels (30%) carry no useful signal:
- Channel 1 (High): **Fake** — identical to close
- Channel 2 (Low): **Fake** — identical to close
- Channel 3 (Volume): **Fake** — all ones → normalizes to 0.0
- Channels 8-11 (Volume ratios): **Fake** — all constant (1.0/1.0)
- Channels 24-26 (VWAP + bands): **Degraded** — VWAP without volume = simple price mean

**The irony**: Bella already generates full OHLCV `(batch, seq, 64, 5)` on GPU. The output is **thrown away** at `setup.py:657`:
```python
close_prices = generated[:, :, :, 3]  # ← discards 80% of Bella's output
```

All three data sources have or can produce OHLCV — the bottleneck is entirely in the consumer pipeline.

---

## Root Cause: The Discard Chain

```
Bella generates (batch, seq, 64, 5) full OHLCV on GPU
  └→ setup.py:657 — extracts close only: generated[:, :, :, 3]
     └→ train_easy.py — _refill_buffer gets (T, n_assets)
        └→ _generate_batch_episodes pads to (N, T, n_assets)
           └→ encoder receives (T, n_assets), fakes OHLCV internally
              └→ 8 channels wasted

Historical has (T, n_assets, 5) in RAM cache
  └→ TASK_698 slices cache_data[:, :, 3] — close only
     └→ same pipeline → same waste

Benchmark loads (T, n_assets, 5) from DB
  └→ benchmark_easy.py:191 — cache_data[:, :, 3]
     └→ same encoder → same fake channels
```

**Fix**: Stop discarding. Pass OHLCV through the entire chain.

---

## What Stays the Same

- **Oracle** (`greedy_lookahead_path`) — operates on close price log returns. No change.
- **Model architecture** — input is encoder output `(n_channels, window_size, n_assets)`. Unchanged.
- **AR blend** — operates on encoded observations. No change.
- **Encoder output shape** — still `(27, ws, n_assets)`. Same channels, just real data instead of fakes.
- **Checkpoints** — fully compatible. Weights depend on encoder output shape, not input shape.
- **VRAM** — Bella already computes all 5 channels on GPU. No additional VRAM cost. Pipeline just stops discarding them.

---

## Implementation Plan

### Phase A: Encoder Accepts OHLCV

**Files**: `mcts_trader/env/jax_encoder.py`

Change `encode_observation_temporal` to accept both shapes:
```python
def encode_observation_temporal(
    prices: chex.Array,        # (n_steps, n_assets, 5) OHLCV or (n_steps, n_assets) legacy
    window_start: chex.Array,
    config: EncoderConfig,
) -> chex.Array:
    n = config.n_assets
    ws = config.window_size

    if prices.ndim == 3:
        # Full OHLCV: (n_steps, n_assets, 5) → O=0, H=1, L=2, C=3, V=4
        price_window = jax.lax.dynamic_slice(prices, (window_start, 0, 0), (ws, n, 5))
        close_T = price_window[:, :, 3].T
        high_T = price_window[:, :, 1].T
        low_T = price_window[:, :, 2].T
        volume_T = price_window[:, :, 4].T
    else:
        # Legacy close-only
        price_window = jax.lax.dynamic_slice(prices, (window_start, 0), (ws, n))
        close_T = price_window.T
        high_T = close_T
        low_T = close_T
        volume_T = jnp.ones_like(close_T)

    return _compute_all_channels_temporal(close_T, volume_T, high_T, low_T, config)
```

Update all entry points: `jit_encode_temporal`, `jit_batch_encode_temporal`, `jit_vmap_encode_temporal`, compressed variant.

**JIT note**: `ndim` check in traced code may cause issues. Alternative: use `static_argnames=['ohlcv_mode']` flag on EncoderConfig to select branch at trace time. Avoids tracing both paths.

### Phase B: Stop Discarding Bella's OHLCV

**Files**: `mcts_trader/training/setup.py`

This is the critical fix. `generate_bella_prices_on_device()` currently:
1. Generates `(n_envs, seq_len, n_assets, 5)` full OHLCV ✓
2. Reshapes correctly ✓
3. **Discards 4 channels** at line 657 ✗
4. Converts close to price levels via cumsum ✓ (but only on close)

Fix: apply the volatility scaling + price conversion to ALL channels:

```python
# Current (BROKEN):
close_prices = generated[:, :, :, 3]  # DISCARDS 80%
step_returns = close_prices * volatility_scale
log_prices = jnp.cumsum(step_returns, axis=1)
prices = 100.0 * jnp.exp(log_prices)

# Fixed: convert all OHLCV channels to price levels
volatility_scale = 0.015
ohlcv_returns = generated * volatility_scale  # (n_envs, seq, n_assets, 5)

# Close prices (channel 3) — cumulative returns → price levels
close_log = jnp.cumsum(ohlcv_returns[:, :, :, 3], axis=1)
close_prices = 100.0 * jnp.exp(close_log)

# Open (channel 0) — shift close by 1 step
open_prices = jnp.concatenate([close_prices[:, :1], close_prices[:, :-1]], axis=1)

# High/Low (channels 1, 2) — Bella generates realistic H/L offsets from close
# Scale relative to close: high >= close, low <= close
high_offset = jnp.abs(ohlcv_returns[:, :, :, 1]) * volatility_scale
low_offset = jnp.abs(ohlcv_returns[:, :, :, 2]) * volatility_scale
high_prices = close_prices * (1.0 + high_offset)
low_prices = close_prices * (1.0 - low_offset)

# Volume (channel 4) — Bella generates volume patterns
# Convert from z-scored to realistic volume levels
vol_raw = generated[:, :, :, 4]  # z-scored volume from Bella
volume = jnp.exp(vol_raw * 0.5 + 10.0)  # log-normal scale
volume = jnp.maximum(volume, 1.0)

# Stack OHLCV
ohlcv = jnp.stack([open_prices, high_prices, low_prices, close_prices, volume], axis=-1)
ohlcv = jnp.maximum(ohlcv, 1.0)  # ensure positive

if n_envs == 1:
    return ohlcv[0]   # (seq_len, n_assets, 5)
else:
    return ohlcv       # (n_envs, seq_len, n_assets, 5)
```

**Important**: Bella's flow matching model was trained on normalized OHLCV from the database. The H/L/V channels contain learned market structure — they're NOT random. The conversion above must respect what Bella learned:
- Bella outputs are z-scored/standardized returns, not raw prices
- Each OHLCV channel was independently normalized during training
- The conversion must map each channel back to realistic price/volume space
- Verify Bella's normalization scheme in `diff_teacher/data_loader.py` to get the inverse transform right

### Phase C: Training Pipeline — _generate_batch_episodes()

**Files**: `train_easy.py`

Update padding and oracle to handle OHLCV:
```python
if prices_list[0].ndim == 3:
    # OHLCV: (N, T, n_assets, 5)
    n_features = prices_list[0].shape[2] if prices_list[0].ndim == 2 else 5
    prices_padded = np.zeros((n_batch, max_len, n_assets, 5), dtype=np.float32)
    for i, p in enumerate(prices_list):
        prices_padded[i, :p.shape[0]] = p
    prices_jax = jnp.array(prices_padded)
    # Oracle uses close prices only
    close_prices = prices_jax[:, :, :, 3]
    log_returns = jnp.log(close_prices[:, 1:] / jnp.maximum(close_prices[:, :-1], 1e-8))
    # Pass full OHLCV to encoder
else:
    # Legacy close-only: (N, T, n_assets)
    ...existing code...
```

Update `_refill_buffer()` to pass OHLCV through (no more shape reduction).

### Phase D: Historical Source — Full OHLCV Passthrough

**Files**: `mcts_trader/data/history.py`

Change `get_window_random()` (added in TASK_698) to return full OHLCV:
```python
def get_window_random(self, length: int) -> np.ndarray:
    """Returns (length, n_assets, 5) OHLCV from RAM cache. Zero-copy view."""
    max_start = len(self.valid_timestamps) - length
    start_idx = np.random.randint(0, max_start)
    return self.cache_data[start_idx:start_idx + length]  # full OHLCV
```

Also update TASK_698's `_refill_buffer()` historical branch to not slice `[:, :, 3]`.

### Phase E: Benchmark — Full OHLCV

**Files**: `tools/benchmark_easy.py`

Change `load_benchmark_prices()` to return OHLCV:
```python
# Line 191 — current:
prices = market.cache_data[:, :, 3]  # close only

# New:
ohlcv = market.cache_data  # (T, n_assets, 5) full OHLCV
```

Update asset padding to handle 3D:
```python
if actual_assets < n_assets:
    repeats = (n_assets // actual_assets) + 1
    ohlcv = np.tile(ohlcv, (1, repeats, 1))[:, :n_assets, :]
```

Update inference loop to pass OHLCV to encoder. Oracle still uses `ohlcv[:, :, 3]` for close prices.

### Phase F: SyntheticMarket — Synthetic OHLCV

**Files**: `mcts_trader/env/synth_market.py`

SyntheticMarket generates close prices only. Wrap output with synthetic HLV:
```python
def close_to_ohlcv(close_prices: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Convert close prices to synthetic OHLCV. Shape: (T, n_assets) → (T, n_assets, 5)."""
    T, n = close_prices.shape
    ohlcv = np.zeros((T, n, 5), dtype=np.float32)

    ohlcv[0, :, 0] = close_prices[0]
    ohlcv[1:, :, 0] = close_prices[:-1]     # Open = prev close

    returns = np.diff(np.log(np.maximum(close_prices, 1e-8)), axis=0)
    vol = np.maximum(np.std(returns, axis=0, keepdims=True), 1e-6)
    ohlcv[:, :, 1] = close_prices * (1 + np.abs(rng.standard_normal((T, n))) * vol * 0.5)  # High
    ohlcv[:, :, 2] = close_prices * (1 - np.abs(rng.standard_normal((T, n))) * vol * 0.5)  # Low
    ohlcv[:, :, 3] = close_prices            # Close
    ohlcv[:, :, 4] = np.exp(rng.standard_normal((T, n)) * 0.5 + 10)  # Volume

    return ohlcv
```

This gives the model varied (not constant) H/L/V channels even on synthetic data. The patterns won't match real markets, but the model learns to use these channels and gets real signal from historical/Bella sources.

---

## Bella OHLCV: What to Verify

Bella was trained on real OHLCV — the flow matching model learned real volume dynamics, wick patterns, and H/L ranges. But the conversion from Bella's output space back to price space (Phase B) must be validated:

1. **Check normalization**: Read `diff_teacher/data_loader.py` — how was OHLCV normalized before training? Each channel independently? Returns-based? Z-scored?
2. **Inverse transform**: The conversion in `generate_bella_prices_on_device()` must invert Bella's training normalization correctly for ALL 5 channels
3. **Sanity checks** after conversion:
   - `high >= close >= low` for every timestep (OHLCV constraint)
   - `high >= open` and `low <= open`
   - Volume > 0 always
   - No NaN/Inf
4. **Distribution match**: Compare Bella-generated OHLCV statistics against real DB data — means, stds, H-L ranges, volume distributions

---

## Data Leakage Check

| Vector | Status |
|--------|--------|
| High/Low contain future info? | NO — per-candle (1-min). H/L are within that minute. |
| Volume as signal? | SAFE — concurrent with price, not forward-looking. |
| Synthetic HLV patterns? | LOW RISK — noise-based, won't teach spurious patterns. |
| Bella OHLCV distribution mismatch? | MONITOR — Bella trained on real OHLCV, so its output should match. Synthetic HLV won't match but that's acceptable noise. |

---

## Distribution Consistency

After this task, all three data sources produce `(T, n_assets, 5)` OHLCV:

| Source | H/L Quality | Volume Quality | Notes |
|--------|-------------|----------------|-------|
| **Historical** | Real | Real | Ground truth from DB |
| **Bella** | Learned (real structure) | Learned (real patterns) | Trained on real OHLCV, generates all 5 channels |
| **Synthetic** | Noise-based | Noise-based | Provides variance, not structure. Acceptable for curriculum levels. |

The model sees consistent `(T, n_assets, 5)` shape from all sources. Historical and Bella provide real signal in all channels. Synthetic provides noise in H/L/V but this is fine — the model learns from the better sources.

---

## Migration Strategy

**Backward compatibility during transition**:
1. Encoder auto-detects `ndim == 3` (OHLCV) vs `ndim == 2` (close-only)
2. All generators can be upgraded independently
3. Checkpoints fully compatible — encoder output shape unchanged

**Rollout order**:
1. **Phase A** (encoder) — enables OHLCV, still works with close-only
2. **Phase B** (Bella fix) — stop discarding, highest immediate value (already computed on GPU)
3. **Phase C** (training pipeline) — glue to pass OHLCV through
4. **Phase D** (historical) — real OHLCV from DB cache
5. **Phase E** (benchmark) — accurate evaluation
6. **Phase F** (synthetic) — lowest priority, noise-based HLV

---

## Validation & Acceptance Criteria

- [ ] Encoder produces valid `(27, ws, n_assets)` output for both `(T, n_assets)` and `(T, n_assets, 5)` inputs
- [ ] Channels 1-3 (H, L, V) show actual variance when fed real OHLCV (not constant like today)
- [ ] Volume channels (8-11) show meaningful ratios with real volume data
- [ ] VWAP channels (24-26) differ from simple price mean when real volume is present
- [ ] Bella OHLCV satisfies constraints: `high >= max(open, close)`, `low <= min(open, close)`, `volume > 0`
- [ ] Bella OHLCV statistics match real DB data within 2 standard deviations
- [ ] Historical training with OHLCV produces equal or better benchmark metrics vs close-only
- [ ] Benchmark with real OHLCV shows differentiated channel activations
- [ ] No NaN/Inf from zero volumes (guard with `jnp.maximum(vol, 1e-8)`)
- [ ] Existing close-only path still works (backward compat until fully migrated)
- [ ] No JIT recompilation storm (stable compilation with OHLCV shape)

---

## Estimated Scope

| Component | Effort | Risk |
|-----------|--------|------|
| Encoder OHLCV support (Phase A) | Medium | Medium (JIT shape handling) |
| Bella OHLCV passthrough (Phase B) | Medium | Medium (inverse normalization must be correct) |
| Training pipeline (Phase C) | Small | Low |
| Historical OHLCV (Phase D) | Small | Low (remove `[:, :, 3]` slice) |
| Benchmark OHLCV (Phase E) | Small | Low |
| Synthetic OHLCV wrapper (Phase F) | Small | Low |

**Total**: ~1 session. Critical path is Phase A (encoder) + Phase B (Bella inverse transform).

---

## Impact

- **8 dead channels → 8 live channels** — 30% more encoder capacity utilized
- **Bella output fully used** — stop computing and discarding 80% of generated data
- **Volume patterns** — spikes before breakouts, confirmation of trends
- **High/Low (wicks)** — rejection patterns, intra-bar volatility, support/resistance
- **Real VWAP** — institutional reference price with volume weighting
- **Training/inference distribution match** — real trading sees real OHLCV, training should too
- **Benchmark accuracy** — evaluate on what the model will actually see in production
