# TASK_121: Future Leakage Bug Report

**Status:** FIXED (see also TASK_122 for additional bugs)
**Date:** 2026-01-29
**Symptom:** 1492-2100% alpha on epoch 1 (should be ~0% for random model)

---

## Executive Summary

Critical future leakage bug in JAX trading environment where trading execution happens at prices 60 steps ahead of the observation window. The TASK_121 fix corrected the observation to use LOOKBACK windows, but failed to update the trading price index to match.

---

## Bug #1: 60-Step Forward Trading Offset (ROOT CAUSE)

**File:** `mcts_trader/env/jax_trading_env.py`
**Lines:** 1116-1120 (step function), 1619-1623 (get_info function)

### The Problem

```python
# BUGGY CODE (before fix):
current_prices = get_current_prices(
    prices, state.current_step + config.window_size, state.episode_start_idx
)
```

This computes: `idx = episode_start + current_step + 60`

### Index Trace at step=0, episode_start=59, window_size=60:

| Component | Index Calculation | Result | Status |
|-----------|-------------------|--------|--------|
| **Observation start** | `59 + 0 - 60 + 1` | `0` | Correct |
| **Observation end** | `0 + 60 - 1` | `59` | Correct |
| **Trading (BUGGY)** | `59 + 0 + 60` | `119` | **60 steps ahead!** |
| **Trading (FIXED)** | `59 + 0` | `59` | Aligned |

### Leak Mechanism

1. **Observation shows:** `prices[0:60]` (indices 0-59)
2. **Trading executes at:** `prices[119]` (60 steps in future!)
3. **pos_entry records:** Future price ($105 at step 119)
4. **Next observation sees:** Current price ($100 at step 60) AND pos_entry ($105)
5. **Crystal ball effect:** Agent can deduce future price direction from pos_entry vs current_price discrepancy

### Applied Fix

```python
# FIXED CODE:
current_prices = get_current_prices(
    prices, state.current_step, state.episode_start_idx
)
```

**Status:** Applied to lines 1118-1120 and 1622-1624

---

## Bug #2: Inconsistent Price Indices in Burning Money

**File:** `mcts_trader/env/jax_trading_env.py`
**Lines:** 1198-1201

### The Code

```python
# Uses episode_start + current_step (NO +window_size):
step_idx = state.episode_start_idx + state.current_step
prev_step_idx = jnp.maximum(step_idx - 1, 0)
current_prices_step = prices[step_idx]
prev_prices_step = prices[prev_step_idx]
```

### Problem

- Trading (buggy) used: `episode_start + current_step + window_size`
- Burning money used: `episode_start + current_step`
- **60-step mismatch** between trading prices and burn penalty calculation

### Status

After Bug #1 fix, this is now CONSISTENT (both use `episode_start + current_step`).

---

## Bug #3: Alpha Calculation Time Window Mismatch

**File:** `mcts_trader/training/self_play.py`
**Lines:** 761-768

### The Code

```python
start_idx = int(jax_env_states.episode_start_idx[i])
end_idx = start_idx + n_steps_i
start_prices = cpu_prices[start_idx]
end_prices = cpu_prices[end_idx - 1]
asset_returns = (end_prices - start_prices) / (start_prices + 1e-8)
market_return = float(np.mean(asset_returns)) * 100
```

### Problem (Before Bug #1 Fix)

- **Market return:** Calculated from `prices[episode_start]` to `prices[episode_start + n_steps - 1]`
- **Agent trades:** At `prices[episode_start + 60]` to `prices[episode_start + 60 + n_steps - 1]`
- **Different time windows = bogus alpha comparison**

### Status

After Bug #1 fix, agent now trades at `prices[episode_start]` to `prices[episode_start + n_steps - 1]`, which matches market return window.

---

## Bug #4: VectorBT vs JAX Environment Inconsistency

**Files:**
- `mcts_trader/env/vectorbt_env.py:819` — Uses `episode_start_idx + current_step` ✓
- `mcts_trader/env/jax_trading_env.py:1117` — Used `episode_start_idx + current_step + window_size` ✗

### Impact

Models trained on VectorBT env would not transfer correctly to JAX env (and vice versa) due to different price indexing semantics.

### Status

After Bug #1 fix, both environments use consistent indexing.

---

## Verification Steps

### 1. Clear JAX Cache (REQUIRED)
```bash
rm -rf .jax_cache/ .xla_cache/
```

### 2. Add Debug Print (Optional)
In `jax_trading_env.py` step function, uncomment:
```python
jax.debug.print("TRADE_IDX={x} (ep_start={e} + step={s})",
    x=state.episode_start_idx + state.current_step,
    e=state.episode_start_idx,
    s=state.current_step)
```

### 3. Expected Results After Fix
- Epoch 1 alpha: Should be ~0% (±5%) for random model
- Alpha should NOT be consistently positive or negative
- Should vary randomly around 0

---

## Remaining Investigation

If alpha remains high after clearing cache and applying fix:

### Check 1: Observation Window in self_play.py
```python
# Line 616-617 - should match step() indexing
price_idx = state.episode_start_idx + state.current_step
window_starts = price_idx - encoder_config.window_size + 1
```

### Check 2: Position Entry Price Recording
When agent buys, `pos_entry` should be recorded at `prices[episode_start + current_step]`, not a future price.

### Check 3: Portfolio Value Calculation
`compute_portfolio_value()` should use the same `current_prices` as trading execution.

### Check 4: Out-of-Bounds Access
With original bug, at step 399 with episode_start=99:
- Trading index = 99 + 399 + 60 = 558
- If n_steps = 500, this is OUT OF BOUNDS
- JAX clips to last valid index, causing undefined behavior

---

## Files Modified

| File | Lines | Change |
|------|-------|--------|
| `mcts_trader/env/jax_trading_env.py` | 1118-1120 | Removed `+ config.window_size` from step() |
| `mcts_trader/env/jax_trading_env.py` | 1622-1624 | Removed `+ config.window_size` from get_info() |

---

## Timeline

- **TASK_121 Original:** Fixed observation window from FORWARD to LOOKBACK
- **TASK_121 Incomplete:** Did NOT update trading price index to match
- **This Fix:** Aligns trading index with observation window endpoint

---

## Test Command

```bash
python train_jax_fast.py --synthetic-level 0 --epochs 10 --steps 400 --parallel-envs 4
```

**Expected:** Alpha fluctuates around 0% (random noise), NOT consistently 1000%+
