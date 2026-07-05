# TASK_126f: Catastrophic Reward Drop Fix

## Problem Statement

During training, epoch 27 showed a catastrophic reward drop from ~-2 to **-30.99**, indicating severe portfolio collapse or a reward computation bug.

**Data point from logs:**
```
epoch=27 reward=-30.991934 alpha=-4.624 market_return=2.36 optimal_alpha=3.15
```

The agent underperformed the market by 4.6% (alpha=-4.624) while the market gained 2.36%, yet the total episode reward was -30.99 - an order of magnitude worse than typical.

---

## Root Cause Analysis

### 1. Reward Computation Chain

**Primary location:** `mcts_trader/env/jax_trading_env.py` (lines 1196-1264)

```python
# Per-step reward computation (line 1202)
safe_prev = jnp.maximum(state.prev_portfolio_value, 1e-8)
safe_new = jnp.maximum(new_portfolio_value, 1e-8)
log_return = jnp.log(safe_new / safe_prev) * 100.0
reward = jnp.where(is_liquidated, symlog(-50.0), symlog(log_return))
```

**The math behind -30.99:**
- With 200-400 steps per episode, accumulated small negative returns compound
- A 95% portfolio drop → `log(0.05) * 100 ≈ -300` (raw)
- After symlog: `symlog(-300) ≈ -5.7` per step
- With discounting and accumulation → **~-30.99 total episode reward**

### 2. Missing Safeguards Identified

#### A. No Reward Clipping in Environment

**Issue:** Rewards are computed but NOT clipped to the value support range `[-10, 10]` before being returned.

```python
# Current code (jax_trading_env.py:1203):
reward = jnp.where(is_liquidated, symlog(-50.0), symlog(log_return))
# No clipping to VALUE_SUPPORT_RANGE!
```

While `_scalar_to_categorical()` in `trainer.py:623` clips during training, raw episode rewards can exceed the support range, causing:
- Value head training instability
- MCTS search divergence (root values outside expected range)
- Gradient explosions in reward prediction head

#### B. No NaN/Inf Protection in Environment

**Issue:** NaN/Inf checks exist in `trainer.py:1579-1585` (batch validation), but there's **no protection in the environment** before rewards are accumulated.

**Risk scenario:**
- Division by near-zero portfolio values
- `log(0)` or `log(negative)` from numerical errors
- Propagated through 200+ steps before detection

#### C. Weak Division-by-Zero Protection

| Location | Current Protection | Risk |
|----------|-------------------|------|
| `jax_trading_env.py:1200` | `jnp.maximum(prev_portfolio_value, 1e-8)` | 1e-8 floor still allows ratios of 10^8 |
| `differential_reward.py:45` | `jnp.maximum(prev_prices, 1e-10)` | No upper bound check |
| `trainer.py:1150` | None | `scaled_rewards = reward * 100.0` can explode |

### 3. Market Conditions That Trigger Catastrophic Rewards

| Condition | Mechanism | Reward Impact |
|-----------|-----------|---------------|
| **Hard Liquidation** | `is_liquidated=True` triggers `symlog(-50.0)` | -3.9 per liquidation event |
| **Short Squeeze** | Cash goes negative, margin calls fail | Bankruptcy → portfolio wipe |
| **Accumulated Small Losses** | 200 steps × -0.15 per step | -30 total |
| **Burning Money Mode** (TASK_051) | Cash opportunity cost penalties | Additional subtraction from reward |
| **Extreme Leverage** | Large position sizes amplify losses | Non-linear reward degradation |

### 4. Value Support Mismatch

**Configuration:** `mcts_trader/jax_agent/__init__.py:51-53`
```python
VALUE_SUPPORT_MIN = -10
VALUE_SUPPORT_MAX = 10
VALUE_SUPPORT_SIZE = 401  # 0.05 resolution
```

**Problem:** The environment can produce rewards outside this range:
- Liquidation penalty: `symlog(-50.0) ≈ -3.9` ✓ (within range)
- 99% portfolio drop: `symlog(-460) ≈ -6.1` ✓ (within range)
- **But accumulated over many steps with discount 0.997:** MC returns can exceed ±10

---

## Recommended Fixes

### Phase 1: Immediate Hotfix (Critical)

#### 1.1 Add Per-Step Reward Clipping

**File:** `mcts_trader/env/jax_trading_env.py`
**Location:** Line 1203

```python
from mcts_trader.jax_agent import VALUE_SUPPORT_MIN, VALUE_SUPPORT_MAX

# Clip per-step rewards to value support range
raw_reward = jnp.where(is_liquidated, symlog(-50.0), symlog(log_return))
reward = jnp.clip(raw_reward, VALUE_SUPPORT_MIN, VALUE_SUPPORT_MAX)
```

**Rationale:** Prevents extreme per-step rewards from destabilizing training, even if the episode sum exceeds the range.

#### 1.2 Add NaN/Inf Detection in Environment

**File:** `mcts_trader/env/jax_trading_env.py`
**Location:** After line 1203

```python
# Defensive: Check for NaN/Inf in reward (indicates numerical instability)
is_invalid = jnp.isnan(reward) | jnp.isinf(reward)
reward = jnp.where(is_invalid, jnp.array(0.0), reward)
```

#### 1.3 Strengthen Division-by-Zero Protection

**File:** `mcts_trader/env/jax_trading_env.py`
**Location:** Lines 1200-1201

```python
# Use a more reasonable floor that prevents extreme ratios
# If portfolio drops below $1, consider it effectively zero
safe_prev = jnp.maximum(state.prev_portfolio_value, 1.0)
safe_new = jnp.maximum(new_portfolio_value, 1e-8)
```

### Phase 2: Training Stability (High Priority)

#### 2.1 Add Reward Scaling Safeguards in Trainer

**File:** `mcts_trader/jax_agent/trainer.py`
**Location:** Line 1150

```python
# Before scaling, clip to prevent overflow
scaled_rewards = jnp.clip(rewards[:, t - 1], VALUE_SUPPORT_MIN, VALUE_SUPPORT_MAX) * self.reward_scale
```

#### 2.2 Add Episode-Level Reward Diagnostics

**File:** `mcts_trader/training/self_play.py`
**Location:** In `run_self_play_jax()` after episode completion

```python
# Log warning for extreme episode rewards (indicates instability)
if abs(scores[i]) > 20.0:
    print(f"⚠️  EXTREME REWARD: Episode score {scores[i]:.2f} (possible liquidation or bug)")
```

### Phase 3: Architecture Improvements (Medium Priority)

#### 3.1 Consider Reward Normalization

Implement PopArt-style normalization for rewards (not just values):
- Track running mean/std of rewards
- Normalize before feeding to value head
- Denormalize for interpretation

#### 3.2 Liquidation Penalty Review

Current: `symlog(-50.0) ≈ -3.9`

Consider if this penalty is too severe relative to typical rewards (~0.01-0.1 per step). A single liquidation wipes out ~39-390 steps of good performance.

Alternative:
```python
# Proportional penalty based on portfolio loss
liquidation_penalty = symlog((state.initial_portfolio_value - portfolio_value) / state.initial_portfolio_value * 100)
reward = jnp.where(is_liquidated, liquidation_penalty, symlog(log_return))
```

#### 3.3 Add Gradient Clipping for Reward Head

**File:** `mcts_trader/jax_agent/trainer.py`
**Location:** In `_clip_gradients_per_head()`

Ensure reward head gradients are clipped separately to prevent reward prediction from dominating training.

---

## Testing Plan

### Unit Tests

1. **Test extreme portfolio drops:**
   ```python
   # Portfolio drops 99%
   prev_value = 100000.0
   new_value = 1000.0
   # Should produce clipped reward, not NaN/Inf
   ```

2. **Test liquidation boundary:**
   ```python
   # Portfolio at exact liquidation threshold
   portfolio_value = config.initial_cash * config.liquidation_threshold_pct
   # Should handle edge case without crash
   ```

3. **Test reward clipping:**
   ```python
   # Verify rewards are within VALUE_SUPPORT range
   assert VALUE_SUPPORT_MIN <= reward <= VALUE_SUPPORT_MAX
   ```

### Integration Tests

1. **Run 50-epoch training** with extreme market conditions (high volatility)
2. **Monitor for:**
   - Rewards outside [-10, 10]
   - NaN/Inf in gradients
   - Sudden reward spikes (indicating clipping failures)

---

## Acceptance Criteria

- [ ] Per-step rewards are clipped to `[-10, 10]` in environment
- [ ] NaN/Inf detection added to environment step function
- [ ] Division-by-zero protection strengthened (floor at $1.0 instead of $1e-8)
- [ ] Episode-level diagnostics log extreme rewards (>20.0)
- [ ] No catastrophic reward drops (< -25) in 100 epochs of training
- [ ] All existing tests pass

---

## Related Tasks

- TASK_051: Burning Money (cash opportunity cost)
- TASK_066: Physics configuration (reward_scale parameter)
- TASK_118: Alpha-relative reward normalization
- TASK_119: Optimal hindsight path (regret penalty)

---

## References

- `mcts_trader/env/jax_trading_env.py` - Environment reward computation
- `mcts_trader/jax_agent/trainer.py` - Training loop and reward scaling
- `mcts_trader/jax_agent/__init__.py` - Value support configuration
- `mcts_trader/utils/transforms.py` - Symlog implementation
