# TASK_693: Fix Position Semantics in Model-State Rollout

**Status**: IMPLEMENTED
**Priority**: 0 (CRITICAL — TASK_692 rollout uses wrong position logic, recreating the train-inference mismatch it was supposed to fix)
**Depends on**: TASK_692 (model-state rollout — already implemented)
**Blocks**: All training with `autoregressive_state=True`

## Problem

TASK_692 introduced `_build_model_state_trajectory()` with autoregressive rollout, correctly fixing the oracle-state dependency. However, the position update function `action_type_to_position()` uses **absolute** position mapping, while the benchmark (and all inference code) uses **incremental** stepping. This creates the same class of train-inference mismatch that TASK_692 was designed to eliminate.

### Bug 1: Absolute vs Incremental Position (CRITICAL)

**Training rollout** (`train_easy.py:270`):
```python
new_pos = action_type_to_position(action_types)
# Buy → always +1, Sell → always -1, Hold → always 0
```

**Benchmark inference** (`benchmark_easy.py:529-537`):
```python
# Buy: short(-1)→flat(0), flat(0)→long(+1), long(+1)→long(+1) [no-op]
# Sell: long(+1)→flat(0), flat(0)→short(-1), short(-1)→short(-1) [no-op]
```

**Concrete example — model is short (pos=-1), decides Buy (action 2):**

| | Training rollout | Benchmark inference |
|---|---|---|
| new_pos | **+1** (absolute jump) | **0** (incremental step) |
| Position change | Flip: -1 → +1 | Close short: -1 → 0 |
| PnL accounting | Earns on +1 next step | Flat, earns nothing |
| trade_strength | 2.0 (traded + flipped) | 1.0 (traded only) |
| heat update | Double spike | Single spike |

This affects every state channel downstream: position, PnL, hold duration, heat, capacity. The model trains on state trajectories that are physically impossible under the incremental position rules used at inference.

**Second example — model is long (pos=+1), decides Buy (action 2):**

| | Training rollout | Benchmark inference |
|---|---|---|
| new_pos | +1 | +1 |
| traded | 0 (no change) | 0 (no change) |

This case happens to agree, but only because pos already equals the absolute target. The disagreement occurs on all cross-position transitions (short→buy, long→sell, flat→hold-already-flat).

**Third example — model is long (pos=+1), decides Sell (action 0):**

| | Training rollout | Benchmark inference |
|---|---|---|
| new_pos | **-1** (absolute jump) | **0** (incremental step) |
| Position change | Flip: +1 → -1 | Close long: +1 → 0 |
| flip detected | Yes | No |
| trade_strength | 2.0 | 1.0 |

### Bug 2: Confidence Channel Always Zero (MEDIUM)

`_build_model_state_trajectory()` line 252: `conf_tiled = jnp.zeros(...)` — channel 33 is hardcoded zero during training.

Benchmark (`benchmark_easy.py:524-525`): confidence = `np.max(model_softmax, axis=-1)` — channel 33 carries the model's own softmax confidence.

The model can learn to ignore a zero channel, but it's an unnecessary mismatch.

### Bug 3: Wasted Oracle State Computation (LOW)

In `_generate_episode_data()`, when `use_model_rollout=True`:
- Lines 547-563: oracle state channels are computed (`build_state_channels(target_position, ...)`)
- Lines 580-597: model state channels are computed, **overwriting** the oracle result

The oracle computation at 558-563 is pure waste (~200ms per episode). Not a correctness issue.

## Root Cause

`action_type_to_position()` (`train_easy.py:173-192`) treats the 5-action space as absolute position targets:
```
0=Sell → -1, 1=Hold → 0, 2=Buy → +1, 3=LimitSell → -1, 4=LimitBuy → +1
```

But the actual trading semantics (used in `benchmark_easy.py`, `simulate_model_pnl_detailed()`, and `real_trade_jax.py`) treat actions as **directional steps** relative to current position:
```
Buy:  short(-1)→flat(0), flat(0)→long(+1), long(+1)→no-op
Sell: long(+1)→flat(0), flat(0)→short(-1), short(-1)→no-op
Hold: no position change
```

## Fix

### 1. Replace `action_type_to_position()` with incremental stepping in `scan_step`

Delete `action_type_to_position()` (lines 173-192). Replace line 270 in `scan_step` with:

```python
# Incremental position stepping (matches benchmark/inference semantics)
buy = (action_types == 2) | (action_types == 4)   # Buy or LimitBuy
sell = (action_types == 0) | (action_types == 3)   # Sell or LimitSell

# Buy: short→flat, flat→long (move toward +1, capped)
# Sell: long→flat, flat→short (move toward -1, capped)
new_pos = jnp.where(
    buy,
    jnp.minimum(pos + 1.0, 1.0),   # step toward long, cap at +1
    jnp.where(
        sell,
        jnp.maximum(pos - 1.0, -1.0),  # step toward short, cap at -1
        pos  # Hold: no change
    )
)
```

**Why `jnp.minimum(pos + 1, 1)` works:**
- pos=-1, Buy: min(-1+1, 1) = min(0, 1) = 0 (short→flat)
- pos=0, Buy: min(0+1, 1) = min(1, 1) = 1 (flat→long)
- pos=+1, Buy: min(1+1, 1) = min(2, 1) = 1 (long→long, no-op)
- pos=+1, Sell: max(1-1, -1) = max(0, -1) = 0 (long→flat)
- pos=0, Sell: max(0-1, -1) = max(-1, -1) = -1 (flat→short)
- pos=-1, Sell: max(-1-1, -1) = max(-2, -1) = -1 (short→short, no-op)

This is pure `jnp.where` — no conditionals, vmap-safe, JIT-safe, same semantics as benchmark.

### 2. Add model confidence to rollout state

In `scan_step`, after the model forward pass (line 263), compute confidence from the model's own softmax:

```python
# Confidence from model's own softmax (matches benchmark behavior)
softmax_probs = jax.nn.softmax(policy_reshaped, axis=-1)  # (n_assets, 5)
model_confidence = jnp.max(softmax_probs, axis=-1)  # (n_assets,)
```

Then at line 252, replace the zeros:
```python
conf_tiled = jnp.tile(model_confidence[None, None, :], (1, ws, 1))  # (1, ws, n_assets)
```

And return `model_confidence` as part of the scan output so it can be used for the confidence channel in episode construction.

**Updated scan return**: Return `(state_ch, model_confidence)` instead of just `state_ch`. Update the caller to unpack both.

### 3. Skip oracle state computation when model rollout is active

In `_generate_episode_data()`, move the oracle state channel block (lines 547-563) inside the `else` branch. The `if use_state_channels:` guard at line 547 should be restructured:

```python
# Before (wasteful):
if use_state_channels:
    # compute oracle state (ALWAYS, even if overwritten)
    ...
autoregressive_state = ...
if use_model_rollout:
    # compute model state (overwrites oracle)
    ...
else:
    # compute oracle state AGAIN
    ...

# After (clean):
if use_model_rollout:
    # compute model state only
    ...
elif use_state_channels:
    # compute oracle state only
    ...
```

## Files to Modify

| File | Change |
|------|--------|
| `train_easy.py` | (1) Delete `action_type_to_position()` function. (2) Replace absolute position logic in `scan_step` (line 270) with incremental `jnp.minimum/maximum` stepping. (3) Add model confidence computation in `scan_step`. (4) Update `_build_model_state_trajectory` return to include confidence. (5) Restructure `_generate_episode_data()` to skip oracle state when model rollout is active. |

## Verification

After fix, run a trajectory comparison test:
```python
# Generate one episode with model rollout
# Extract position trajectory from training rollout
# Run same market data through benchmark inference loop
# Assert position trajectories are identical step-by-step
```

If positions match at every timestep, the train-inference mismatch is eliminated.

## Expected Impact

| Metric | Before fix (broken rollout) | After fix |
|--------|---------------------------|-----------|
| Position logic | Absolute (flips possible) | Incremental (step-by-step) |
| Train-inference state match | NO (different trajectories) | YES (identical trajectories) |
| Confidence channel (ch 33) | Always zero | Model's softmax max |
| Wasted oracle computation | ~200ms/episode | Eliminated |

## Additional Bug Found During Implementation: Dimension Mismatch (Bug 4)

The TASK_692 implementation had a fourth bug: the state channel slicing used oracle-style
return-timestep indexing (`state_channels[start_idx-1:end_idx-1]` where `start_idx=ws-1=63`)
on model rollout output that is indexed by WINDOW (0..n_valid-1).

For ws=64 and T=400: oracle state has 399 entries (indexed 0..398), so `state_channels[62:398]`
works. But model rollout has n_valid=335 entries (indexed 0..334), so `state_channels[62:398]`
goes out of bounds at index 335+.

**Fix**: Conditional slicing. Model rollout uses `state_channels[:n_windows]`
(scan output[w] = state before window w). Oracle uses `state_channels[start_idx-1:end_idx-1]`.
Applied to both `_generate_episode_data` and `_generate_batch_episodes`.

## Completion Criteria

- [x] `action_type_to_position()` deleted
- [x] `scan_step` uses incremental position stepping matching benchmark
- [x] Model confidence populates channel 33 during rollout
- [x] Oracle state not computed when model rollout is active
- [x] State channel slicing corrected for window-indexed model rollout output
- [ ] Position trajectories identical between training rollout and benchmark inference on same inputs
