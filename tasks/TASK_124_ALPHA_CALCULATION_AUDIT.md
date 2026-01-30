# TASK_124: Alpha Calculation Audit - Still Showing 645% on Epoch 1

**Date:** 2026-01-29
**Status:** OPEN
**Priority:** CRITICAL
**Symptom:** After fixing ALL known future leakage bugs, alpha STILL shows 645% on epoch 1

## Context

Multiple rounds of bug fixes applied (TASK_121, TASK_122, TASK_123):
- Observation window: LOOKBACK (not forward) ✅
- Trading price: No +window_size offset ✅
- Env reset between epochs: Added jit_batch_reset ✅
- Differential reward: LAGGED (prepend) ✅
- Encoder denominator: prev_prices ✅
- Window start clamping: jnp.maximum(0) ✅
- JIT cache clearing: Added ✅

**User Note:** "all trouble began by implementing amax feature" (TASK_119 Optimal Action Imprint)

---

## CRITICAL FINDING: AMAX Imprint Loss Leaks Future

### The Leak Chain

1. **self_play.py:796-798** - Stores FULL episode prices:
   ```python
   result['prices'] = cpu_prices[start_idx:end_idx]  # ALL prices in episode
   result['next_prices'] = cpu_prices[start_idx + 1:next_end]  # FUTURE relative to each step!
   ```

2. **replay_buffer.py:329** - Stores as `log_price_sequence`

3. **trainer.py:1086** - Computes returns from FULL sequence:
   ```python
   returns_for_amax = log_price_seq[:, 1:, :] - log_price_seq[:, :-1, :]
   ```
   At training timestep t, this includes returns from t+1, t+2, ... (FUTURE!)

4. **trainer.py:1088** - Optimal path uses FUTURE:
   ```python
   optimal_path = compute_optimal_path(returns_for_amax[0])
   ```

5. **trainer.py:1128-1142** - Policy trained toward FUTURE-aware targets:
   ```python
   optimal_policy_targets = optimal_path_to_policy(optimal_path.selections, returns_for_amax[0])
   imprint_loss = compute_imprint_loss(agent_policy, optimal_policy_targets)
   ```

### Why This Causes 645% Alpha

Even on epoch 1 with random weights:
- The `imprint_loss` gradient updates push policy toward future-optimal actions
- After just 50 training iterations, policy learns to "predict" future
- This isn't prediction - it's MEMORIZATION of hindsight-optimal actions
- The gradient from imprint_loss is DIRECTLY leaking future info into the policy!

### Immediate Test

```bash
# Test WITHOUT amax to confirm this is the cause:
python train_jax_fast.py --synthetic-level 0 --epochs 5 --parallel-envs 4 \
    --no-checkpoint --jax-env --steps 400
# If alpha drops to ~0%, AMAX is confirmed as the leak
```

### Fix Options

**Option A: Disable AMAX for now**
Set `--amax-imprint-weight 0` or `--amax-method none`

**Option B: Only use PAST returns for optimal path**
The imprint should use returns that were KNOWN at each step, not the full future sequence.

---

## Alpha Calculation Trace

### Where alpha comes from:

```python
# train_jax_fast.py:871-872
alpha = trajectory.get('alpha', 0.0)
market_return = trajectory.get('market_return', 0.0)

# self_play.py:758, 764-767, 791
final_return = float(final_states.cumulative_return[i])  # From JAX env
start_prices = cpu_prices[start_idx]
end_prices = cpu_prices[end_idx - 1]
asset_returns = (end_prices - start_prices) / (start_prices + 1e-8)
market_return = float(np.mean(asset_returns)) * 100
alpha = final_return - market_return

# jax_trading_env.py:1251
new_cumulative_return = (new_portfolio_value / state.initial_portfolio_value - 1.0) * 100.0
```

### Key Variables to Audit:

| Variable | Source | Expected | Verify |
|----------|--------|----------|--------|
| `final_return` | `final_states.cumulative_return[i]` | Realistic % | ? |
| `market_return` | `(end_prices - start_prices) / start_prices * 100` | ~5% for bull period | Shows +5.2% ✓ |
| `new_portfolio_value` | `compute_portfolio_value(new_state, current_prices)` | Close to initial | ? |
| `current_prices` | `get_current_prices(prices, step, episode_start)` | `prices[episode_start + step]` | ✓ |
| `start_idx` | `jax_env_states.episode_start_idx[i]` | >= window_size - 1 | ? |
| `end_idx` | `start_idx + n_steps_i` | Valid index | ? |

---

## Bug Hunt Areas

### Area 1: AMAX / OAI Feature (TASK_119)
**User suspects this caused the regression**

Files to audit:
- `mcts_trader/jax_agent/trainer.py:1075-1147` - amax computation
- `mcts_trader/jax_agent/path_solver.py` - optimal path calculation

Questions:
1. Does `optimal_path_to_policy()` leak future info into training?
2. Does `returns_for_amax` use correct price indices?
3. Is `imprint_loss` computed with future-aware targets?

```python
# trainer.py:1086 - Check this carefully
returns_for_amax = log_price_seq[:, 1:, :] - log_price_seq[:, :-1, :]
# Is log_price_seq using correct lookback window?
```

### Area 2: Position Entry Price Recording
When agent buys, `pos_entry` should record price at action time.

Check in `_execute_buy_jax()` and `_execute_sell_jax()`:
```python
# jax_trading_env.py - verify pos_entry = current_prices[asset]
# NOT a future price
```

### Area 3: Portfolio Value Formula
```python
# jax_trading_env.py:229-244
def compute_portfolio_value(state, current_prices):
    # Long: pos_size * current_prices
    # Short: pos_size * (2 * pos_entry - current_prices)
    # Verify formulas are correct
```

### Area 4: Price Index Consistency
All these should use SAME index `episode_start + current_step`:
- Observation window END
- Trading execution
- Portfolio valuation
- Reward calculation

### Area 5: Log Price Sequence for AMAX
```python
# Where is log_price_seq constructed?
# Does it use lookback or forward window?
# Is it passed to trainer correctly?
```

### Area 6: Replay Buffer Price Storage
```python
# train_jax_fast.py - how prices are stored for replay
# self_play.py:794-803 - prices and next_prices collection
# Are replay buffer prices correct?
```

---

## Specific Checks for Bug Hunters

### Check 1: Print actual indices
Add debug in self_play.py:
```python
if i == 0:
    print(f"[DEBUG] episode_start={start_idx}, n_steps={n_steps_i}")
    print(f"[DEBUG] end_idx={end_idx}, prices_shape={cpu_prices.shape}")
    print(f"[DEBUG] start_price={start_prices[0]:.2f}, end_price={end_prices[0]:.2f}")
    print(f"[DEBUG] final_return={final_return:.2f}%, market_return={market_return:.2f}%")
```

### Check 2: Portfolio value sanity
Add debug in jax_trading_env.py step():
```python
jax.debug.print("STEP {s}: portfolio={p:.2f}, initial={i:.2f}, cum_ret={r:.2f}%",
    s=state.current_step, p=new_portfolio_value,
    i=state.initial_portfolio_value, r=new_cumulative_return)
```

### Check 3: Verify env reset is working
```python
# After jit_batch_reset, print episode_start_idx
print(f"[DEBUG] After reset: episode_starts={jax_env_states.episode_start_idx[:4]}")
```

### Check 4: Verify observation window
```python
# In compute_observation, print window bounds
jax.debug.print("OBS: window [{w}, {e}] for step {s}",
    w=window_start, e=window_start+config.window_size, s=state.current_step)
```

---

## Hypothesis: Instant Execution Artifact

Even with all bugs fixed, 645% alpha could be from "instant execution":
- Agent observes Close[t]
- Agent executes at Price[t] (same price!)
- On trending data, knowing execution price = perfect timing

**Test:** Run on random noise data (--synthetic-level 0 with high noise)
- If alpha drops to ~0%, it's instant execution artifact
- If alpha stays high, there's still a bug

---

## Files to Audit (Priority Order)

1. **HIGH:** `mcts_trader/jax_agent/trainer.py` - amax/imprint feature
2. **HIGH:** `mcts_trader/jax_agent/path_solver.py` - optimal path
3. **HIGH:** `mcts_trader/training/self_play.py` - alpha calculation, price collection
4. **MEDIUM:** `mcts_trader/env/jax_trading_env.py` - portfolio value, position entry
5. **MEDIUM:** `train_jax_fast.py` - replay buffer, price passing
6. **LOW:** `mcts_trader/jax_agent/replay_buffer.py` - price storage

---

## Test Commands

```bash
# Clear all caches first
rm -rf ~/.cache/jax/ .jax_cache/ .xla_cache/

# Test 1: Without amax feature
python train_jax_fast.py --synthetic-level 0 --epochs 5 --parallel-envs 4 \
    --no-checkpoint --jax-env --steps 400
# Expected: alpha ~0% if bugs fixed, feature not involved

# Test 2: With amax feature disabled explicitly
python train_jax_fast.py --synthetic-level 0 --epochs 5 --parallel-envs 4 \
    --no-checkpoint --jax-env --steps 400 --amax-method none
# Compare with Test 1

# Test 3: Historical data
python train_jax_fast.py --db-path trading_data.db \
    --start-date 2024-02-24 --end-date 2024-03-02 \
    --epochs 5 --steps 1440 --parallel-envs 4 \
    --no-checkpoint --jax-env
# Expected: alpha should be realistic (not 600%+)
```

---

## BUGS IDENTIFIED AND FIXED

**Date:** 2026-01-30

### Bug 1: Static Seed (TASK_124) - ✅ FIXED

**File:** `mcts_trader/training/setup.py:254`
```python
rng_key=jax.random.PRNGKey(42)  # WAS STATIC SEED!
```

**Fix Applied:** Dynamic epoch-based seed in `setup_jax_environment(config, seed=epoch*1000)`
- `setup.py`: Added `seed` parameter
- `train_jax_fast.py`: All calls pass epoch-based seeds
- `env_factory.py`: Changed fallback from `PRNGKey(42)` to time-based

### Bug 2: AMAX Future-Aware Training - ✅ CONFIRMED

**File:** `mcts_trader/jax_agent/trainer.py:1075-1147`

The AMAX feature causes training-time data leakage:
1. `returns_for_amax = log_price_seq[:, 1:, :] - log_price_seq[:, :-1, :]` - computes ALL returns
2. `optimal_path = compute_optimal_path(returns_for_amax[0])` - uses FUTURE returns
3. `imprint_loss = compute_imprint_loss(policy, optimal_targets)` - trains toward future-aware targets

**Status:** Controlled by flags (default: DISABLED)
```python
amax_method=None          # Skip optimal path computation
amax_imprint_weight=0.0   # Skip imprint loss
amax_regret_weight=0.0    # Skip regret penalty
```

Guard at line 1078: `if self.amax_method is not None:` prevents execution when disabled.

### Bug 3: Same-Step Observation Leak - ✅ FIXED

**File:** `mcts_trader/env/jax_trading_env.py:1304-1358`

The agent observed `prices[t]` before deciding to execute at `prices[t]`:
```python
# OLD (buggy):
start_idx = episode_start + current_step - window_size + 1
# Window: [t-59, t] → agent sees prices[t]!

# NEW (fixed):
start_idx = episode_start + current_step - window_size
# Window: [t-60, t-1] → agent only sees up to prices[t-1]
```

**Fix Applied:**
- `compute_observation_27ch()`: Line 1306 - removed `+ 1`
- `compute_observation_31ch()`: Line 1331 - removed `+ 1`
- `compute_observation()`: Line 1357 - removed `+ 1`
- `reset()`: Line 1065 - `min_start = window_size` (was `window_size - 1`)

**Why This Caused 645% Alpha:**
On synthetic sine waves, knowing `Close[t]` before trading at `Close[t]` allows perfect momentum exploitation. Even a "random" model with lucky weights on return features captures the entire trend.

### All Bugs Summary

| Bug | Root Cause | Status |
|-----|------------|--------|
| Static Seed | `PRNGKey(42)` → memorization | ✅ FIXED |
| AMAX Leak | Future-aware training targets | ✅ CONTROLLED (disabled by default) |
| Same-Step Obs (env) | Agent sees execution price in jax_trading_env.py | ✅ FIXED |
| Same-Step Obs (self-play) | Agent sees execution price in self_play.py | ✅ FIXED |
| Model Init Seed | `PRNGKey(42)` for model weights | ✅ FIXED |
| **Limit Short Free Money** | No margin deducted on limit short fills | ✅ FIXED |

### Bug 6: Limit Order Short - FREE MONEY BUG (2026-01-30)

**File:** `mcts_trader/env/jax_trading_env.py:615`
```python
# OLD (buggy):
cash_change_sell_short = jnp.where(opening_short, 0.0, 0.0)  # No cash deducted!
```

When opening SHORT via limit sell fill:
- Position created with value `size * entry_price`
- NO margin deducted from cash
- This is FREE MONEY - portfolio increased without cost!

**Fix:** Deduct `short_margin_cost + fee` from cash when opening shorts via limit fills.

**Impact:** Random model alpha went from ~76% to ~0% (expected for random actions).

---

## Bug Hunt Prompt for External Models (v2)

```
TASK: Find the FUNDAMENTAL bug causing 1753% alpha on epoch 1
      Even with AMAX feature DISABLED, alpha is still impossible.

CONTEXT:
- JAX trading RL system showing 1753% alpha vs 5.2% market return
- This is with a RANDOM model (epoch 1, no learning yet)
- AMAX/imprint_loss is DISABLED (weights = 0)
- All known future leakage bugs already fixed:
  * Observation window: LOOKBACK (not forward) ✅
  * Trading price: episode_start + step (no offset) ✅
  * Env reset: jit_batch_reset after each epoch ✅
  * Differential reward: LAGGED (prepend) ✅
  * Encoder denominator: prev_prices ✅
  * Window start clamping: jnp.maximum(0) ✅

THE BUG IS NOT IN:
- imprint_loss (disabled)
- observation window (fixed)
- differential reward (fixed to lagged)

ALPHA CALCULATION CHAIN:
1. self_play.py:758 - final_return = final_states.cumulative_return[i]
2. jax_trading_env.py:1251 - cumulative_return = (portfolio_value / initial - 1) * 100
3. jax_trading_env.py:1156 - portfolio_value = compute_portfolio_value(state, current_prices)
4. jax_trading_env.py:229-240 - portfolio = cash + position_values + reserved
5. jax_trading_env.py:222-226 - position_values = pos_size * current_prices (for longs)

FOCUS ON THESE EXACT CODE PATHS:
1. get_current_prices() - Does idx = episode_start + step correctly?
2. execute_buy() - Does pos_entry = current_prices[asset_idx]?
3. compute_portfolio_value() - Is the formula correct?
4. reset() - Is prev_portfolio_value initialized correctly?
5. step() - Is current_prices passed consistently everywhere?

SPECIFIC QUESTIONS:
- At step t, what price index is used for trading? (Should be episode_start + t)
- At step t, what price index is used for portfolio valuation? (Should be episode_start + t)
- Is there ANY code path where prices[episode_start + t + N] with N > 0 is accessed?

KEY FILES (read in this order):
1. mcts_trader/env/jax_trading_env.py - MOST IMPORTANT
   - get_current_prices(): line 207-211
   - compute_portfolio_value(): line 229-240
   - execute_buy(): line 707-820
   - step(): line 1090-1267
   - reset(): line 1046-1095

2. mcts_trader/training/self_play.py
   - Alpha calculation: line 758-791
   - Price collection: line 794-803

3. mcts_trader/training/setup.py
   - Static seed issue: line 254

DEBUG APPROACH:
1. Trace a single step: prices used at each stage
2. Verify idx = episode_start + current_step everywhere
3. Check if pos_entry ever gets a future price
4. Check if portfolio_value ever uses future prices

OUTPUT FORMAT:
For each bug found:
- File:Line
- Exact code snippet
- Index trace showing the leak
- Why it causes 1753% alpha inflation
- Suggested fix with code
```

---

## Notes

- Market return shows realistic +5.2% (not bugged)
- Alpha of 645% means agent_return = 650% in ~400 steps
- This is ~1.6% per step compounding, which is absurd
- Either portfolio value is wrong, or something leaks future
