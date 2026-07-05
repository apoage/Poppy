# TASK_121: Fix Observation Window Data Leak (Future Price Visibility)

**Created:** 2026-01-29
**Priority:** CRITICAL
**Status:** FIXED (see also TASK_122 for additional bugs found)

## Problem Summary

The JAX trading environment has a critical data leak: the observation encoder receives a **FORWARD window** of prices instead of a **LOOKBACK window**. This means the agent sees 59 future price steps in every observation, enabling "fortune telling" and unrealistic performance (777% alpha on historical data).

## Root Cause Analysis

### Current (Broken) Behavior

In `mcts_trader/training/self_play.py` line 610:
```python
window_starts = state.episode_start_idx + state.current_step
```

The encoder then extracts `prices[window_start : window_start + 60]`:
- At step `t`, `window_start = episode_start + t`
- Observation contains `prices[episode_start + t : episode_start + t + 60]`
- Trade executes at `prices[episode_start + t]`
- **BUG:** Agent sees prices at t+1, t+2, ... t+59 (59 FUTURE prices!)

### Expected Behavior

At step `t`, the agent should only see historical prices:
- Observation should contain `prices[episode_start + t - 59 : episode_start + t + 1]`
- This is a 60-step LOOKBACK window ending at the current price
- No future prices visible

### Impact

- Model learns to "predict" future by literally seeing it
- 777%+ alpha on historical benchmarks (should be ~10-50% realistic)
- Trained model is useless for real trading (no future data available)
- All checkpoints trained with this bug are compromised

## Files to Modify

### 1. `mcts_trader/training/self_play.py`

**Line 610** - Fix window_starts calculation:
```python
# OLD (sees future):
window_starts = state.episode_start_idx + state.current_step

# NEW (lookback only):
window_starts = state.episode_start_idx + state.current_step  # No change here
# But encoder needs window_size offset - see below
```

Actually, the cleanest fix is to adjust what we pass to the encoder. The encoder expects `window_start` to be the START of the window, so for lookback we need:
```python
window_starts = state.episode_start_idx + state.current_step - window_size + 1
```

But this requires `episode_start_idx >= window_size - 1` to avoid negative indices.

### 2. `mcts_trader/env/jax_trading_env.py`

**Line 1062-1063** - Fix episode_start initialization:
```python
# OLD:
max_start = n_steps - actual_max_steps - config.window_size
episode_start = jax.random.randint(key1, (), 0, jnp.maximum(max_start, 1))

# NEW - ensure room for lookback at start:
min_start = config.window_size - 1  # Need window_size-1 steps of history
max_start = n_steps - actual_max_steps - 1  # -1 because we don't need forward window anymore
episode_start = jax.random.randint(key1, (), min_start, jnp.maximum(max_start, min_start + 1))
```

### 3. `mcts_trader/training/self_play.py` (multiple locations)

All places computing window_starts need the lookback offset:
- Line 610: Main JAX self-play loop
- Possibly other locations using window_starts

## Verification

After fix, run benchmark and expect:
- Alpha should drop to realistic levels (10-50% range)
- Model should NOT outperform on historical data it hasn't trained on
- Synthetic data performance may also change (patterns still learnable but not via cheating)

## Testing

```bash
# Quick sanity check - run benchmark on bull period
python train_jax_fast.py \
    --db-path trading_data.db \
    --start-date 2024-02-24 --end-date 2024-03-02 \
    --epochs 7 --steps 1440 --parallel-envs 4 \
    --load-checkpoint jax_ckpt_fast/popy_V3c_v1_synthetic_complete \
    --no-checkpoint --jax-env --bf16 --sims 24

# If alpha is still 700%+, bug not fixed
# If alpha drops to <100%, fix is working
```

## Notes

- All existing checkpoints (popy_v3e, etc.) were trained with this leak and should be retrained
- The OAI (Optimal Action Imprint) was initially suspected but is NOT the cause
- This bug has been present since the JAX environment was introduced

## Additional Fix: Optimal Alpha Units (trainer.py)

The `optimal_alpha` metric was in raw log return space (tiny values like 0.05) while
`agent_pnl` was in symlog(return*100) space. These aren't comparable.

**Fix:** Scale optimal_alpha by 100 to match the agent's *100 scaling:
```python
optimal_alpha_scaled = optimal_path.total_alpha * 100.0
```

This makes alpha_capture ratios meaningful (should be ~0.1-1.0 range instead of wildly off).
