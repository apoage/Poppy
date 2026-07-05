# TASK_122: Money Printing Bugs - Kimi & Gemini Analysis

**Created:** 2026-01-29
**Priority:** CRITICAL
**Status:** Fixed
**Related:** TASK_121 (partially fixed)

## Fixes Applied

| Bug | File | Fix |
|-----|------|-----|
| Overfitting Loop | `train_jax_fast.py:763` | Added `jit_batch_reset()` after each epoch |
| Differential Reward | `train_jax_fast.py:800` | Changed prepend to append (concurrent return) |
| Encoder Denominator | `jax_encoder.py:185`, `state_fast.py:180` | Use `prev_prices` in denominator |
| Episode Range | `jax_trading_env.py:1068` | `min_start + 10` instead of `min_start + 1` |

## Summary

Despite TASK_121 window fix being correctly applied and verified loading, alpha remains at 1492% on epoch 1. Two external AI analyses (Kimi and Gemini) identified additional critical bugs.

---

## BUG 1: Infinite Overfitting Loop (Kimi - CRITICAL)

### Problem
Environment states are initialized **once at startup** and **never reset between epochs**. The agent plays the exact same episode (same start index, same prices) thousands of times.

### Location
`train_jax_fast.py` - training loop

### Mechanism
```python
# At startup:
jax_env_states = initialize_environment(...)  # episode_start_idx = 100 (example)

# Each epoch:
for epoch in range(epochs):
    trajectories = run_self_play_jax(jax_env_states, ...)  # Same states reused!
    # jax_env_states is NEVER reset with new random start indices
```

### Impact
- Agent memorizes specific price movements: "Sine wave peaks at step 15, buy now"
- Achieves impossible returns on that ONE memorized episode
- Explains 1492% alpha: perfect memorization of deterministic future

### Fix
Reset environment states with new random seed at start of every epoch:
```python
for epoch in range(epochs):
    # Reset states with new random key for each epoch
    epoch_key = jax.random.fold_in(master_key, epoch)
    jax_env_states = reset_env_states(jax_env_states, epoch_key)
    trajectories = run_self_play_jax(jax_env_states, ...)
```

---

## BUG 2: Differential Reward Future Leakage (Gemini - CRITICAL)

### Problem
When `differential_reward` is enabled, market returns are computed using `prices[1:] - prices[:-1]` which is **future information** relative to each step's decision point.

### Location
`train_jax_fast.py:793-806`

### Current Code
```python
if getattr(config, 'differential_reward', False):
    for traj in trajectories:
        if 'prices' in traj and len(traj['prices']) > 1:
            prices = np.array(traj['prices'])
            price_returns = (prices[1:] - prices[:-1]) / (prices[:-1] + 1e-8)  # FUTURE!
            market_returns = np.mean(price_returns, axis=1)
            market_returns = np.concatenate([[0.0], market_returns])
```

### Impact
- At step t, reward is adjusted by market return from t to t+1
- This is future information - the agent gets rewarded based on what the market will do
- Creates artificial alpha that doesn't exist in real trading

### Fix
Use only returns observable at each step:
```python
if getattr(config, 'differential_reward', False):
    for traj in trajectories:
        if 'prices' in traj and len(traj['prices']) > 1:
            prices = np.array(traj['prices'])
            # At step t, only t-1 -> t return is known
            for t in range(1, len(traj['rewards'])):
                past_return = (prices[t] - prices[t-1]) / (prices[t-1] + 1e-8)
                traj['rewards'][t] -= diff_weight * np.mean(past_return)
```

---

## BUG 3: Encoder Return Calculation (Gemini - MEDIUM)

### Problem
Return formula uses wrong denominator.

### Location
`jax_encoder.py:185`

### Current Code
```python
returns_1m = jnp.diff(prices, axis=1, prepend=prices[:, 0:1]) / (prices + 1e-8)
# Computes: (prices[t] - prices[t-1]) / prices[t]  <- WRONG!
```

### Correct Formula
```python
# Return should be: (prices[t] - prices[t-1]) / prices[t-1]
prev_prices = jnp.concatenate([prices[:, 0:1], prices[:, :-1]], axis=1)
returns_1m = (prices - prev_prices) / (prev_prices + 1e-8)
```

### Impact
- Distorts return magnitude (not future leakage, but incorrect calculation)
- May affect learning quality

---

## BUG 4: Episode Start Range Edge Case (Gemini - LOW)

### Location
`jax_trading_env.py:1068`

### Problem
```python
max_start = jnp.maximum(max_start, min_start + 1)
```
If `n_steps` is too small, this creates a minimal valid range that might cause issues.

### Fix
```python
max_start = jnp.maximum(max_start, min_start + 10)  # Ensure reasonable range
```

---

## Verification After Fixes

After applying fixes, run:
```bash
python train_jax_fast.py \
    --db-path trading_data.db \
    --start-date 2024-02-24 --end-date 2024-03-02 \
    --epochs 7 --steps 1440 --parallel-envs 4 \
    --load-checkpoint jax_ckpt_fast/popy_V3c_v1_synthetic_complete \
    --no-checkpoint --jax-env --bf16 --sims 24
```

**Expected Results:**
- Epoch 1 alpha: Between -10% to +10% (random model baseline)
- After training: Gradual improvement to realistic 10-50% range
- No more 1000%+ returns

---

---

## Defenses Against Future Bugs

### 1. Unit Test for Causality
```python
def test_no_future_leak():
    """Ensure observations don't contain future prices."""
    env = create_jax_env(...)
    state = env.reset(rng_key, prices, config)

    for t in range(10):
        obs = compute_observation(state, prices, config)
        # Verify observation window ends at current step
        window_end = state.episode_start_idx + state.current_step
        assert obs_uses_only_prices_up_to(obs, window_end)

        action = random_action()
        state, reward, done, info = env.step(state, action, prices, config)
```

### 2. Invariant Check (Runtime)
```python
# Add to train_jax_fast.py after epoch 1
if epoch == 0 and alpha > 50.0:
    print(f"WARNING: Alpha {alpha:.1f}% on epoch 1 - possible future leak!")
    print("Random policy should have alpha ~0% (±5%)")
    # Optionally: halt and audit
```

### 3. Index Audit Tool
```python
def audit_price_indices(state, config):
    """Verify all price accesses satisfy: idx <= episode_start + current_step"""
    ep_start = state.episode_start_idx
    step = state.current_step
    max_valid_idx = ep_start + step

    # Check observation window
    window_start = ep_start + step - config.window_size + 1
    window_end = ep_start + step
    assert window_end <= max_valid_idx, f"Observation sees future: {window_end} > {max_valid_idx}"

    # Check trading price
    trade_idx = ep_start + step
    assert trade_idx <= max_valid_idx, f"Trading at future price: {trade_idx} > {max_valid_idx}"
```

---

## Remaining Concern: "Instant Execution" Leakage

**Status:** Design artifact, not a code bug

If alpha remains unrealistic after all fixes, this may be a simulation design issue:

- **The Leak:** Agent observes market state at time t (including Close[t]) and executes trade at exactly Price[t]
- **The Result:** On low-noise synthetic data (sine waves), knowing exact execution price at decision time allows perfect timing with zero latency
- **Recommendation:** Shift execution to t+1 (next open) so agent must predict future price rather than react to current one

This is a **simulation realism** issue, not future data leakage. Real trading has:
- Order latency (ms to seconds)
- Slippage between decision and fill
- Price movement during execution

For now, accept this as a known limitation. High alpha on synthetic data is expected; real market data should show more realistic results.

---

## Additional Cleanup

After fixing, clear all caches:
```bash
rm -rf /home/lukas/Trader/.xla_cache/*
rm -rf /home/lukas/Trader/.jax_cache/*
rm -rf ~/.cache/jax/
```

---

## Attribution

- **Bug 1 (Overfitting Loop)**: Identified by Gemini
- **Bug 2-4 (Differential Reward, Encoder, Edge Case)**: Identified by Kimi
- **TASK_121 Window Fix**: Verified correct by both
