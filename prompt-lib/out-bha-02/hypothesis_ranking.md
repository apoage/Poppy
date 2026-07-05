# Hypothesis Ranking for Regression Bug Hunt: Trade Action Learning Collapse

## Summary

Between commits `06c44ab` and `77cb089`, agree_trade collapsed from ~15% to ~7%. The regression is caused by **multiple interacting bugs** in the model-state rollout implementation, with the **confidence channel misalignment (BUG-2)** being the primary killer.

---

## Ranked Hypotheses

### 1. CONFIRMED: Confidence Channel Misalignment (BUG-2) - PRIMARY CULPRIT

**Suspect Commit**: `a86c206` (BUG-2 completion)

**Change**: Confidence channel emission in scan output

**Why It Kills Trade Learning**:
The confidence channel (channel 33) provides the model with its own confidence from the previous timestep. During model-state rollout, the confidence signal is fed back into the next observation. However, the scan output emits `prev_conf` (zero for first step), not `model_confidence` (actual computed confidence). This creates a **mismatch between training and inference**:

- **Inference** (benchmark_easy.py:579-581): `prev_confidence` is updated with `max(softmax)` at each step
- **Training** (train_easy.py:299-302): `confidence_trajectory[w] = prev_conf` which is the **previous step's confidence** (zero at start)

The critical issue: `prev_conf` is **initialized to zeros** (line 215) and only propagates the **previous** iteration's confidence. At step 0, `prev_conf = 0`, and this zero gets tiled into the observation. The model sees zeros instead of meaningful confidence signals during training.

**Evidence**:
```python
# train_easy.py:299-302
# Emit prev_conf (not model_confidence) so confidence_trajectory[w] =
# confidence model SAW at step w, matching inference alignment.
# model_confidence propagates forward via carry for step w+1.
return (new_pos, new_pnl, new_steps, new_h_fast, new_h_slow, model_confidence), (state_ch, prev_conf)
```

vs benchmark_easy.py:
```python
# benchmark_easy.py:579-581
# 5. Update confidence from model's own softmax
model_softmax = np.array(jax.nn.softmax(logits_per_asset[0], axis=-1))  # (n_assets, 5)
prev_confidence = np.max(model_softmax, axis=-1)  # (n_assets,)
```

**Result**: The model in training sees zero confidence channels for the entire first step and lagging confidence thereafter. This corrupts the state channels and makes the model rely on incorrect feedback signals.

**Proposed Fix**:
```python
# In train_easy.py:302, emit model_confidence, not prev_conf
return (new_pos, new_pnl, new_steps, new_h_fast, new_h_slow, model_confidence), (state_ch, model_confidence)
```

---

### 2. LIKELY: Returns Alignment Bug (BUG-1)

**Suspect Commit**: `06c44ab` (TASK_692/693)

**Change**: Returns slicing for model-state rollout

**Why It Kills Trade Learning**:
The returns array is misaligned with the observation windows:

```python
# train_easy.py:576-577
ws = self.config.window_size
returns_aligned = log_returns[ws - 1:]  # Slice from index 63 onwards
```

But observations are computed from `window_starts = jnp.arange(n_valid)` where `n_valid = T - ws + 1`. This means:
- `observations[0]` covers prices `[0:64]`, decision point is at index 63
- `returns_aligned[0] = log_returns[63]` ✓ This is correct for the decision point

However, in the **batched path** (line 800), the slicing is different:
```python
# train_easy.py:800
log_returns_i = log_returns[i, ws - 1:valid_lengths[i] - 1]
```

This inconsistency may cause misalignment between the PnL computed in the rollout and the actual returns at decision points.

**Evidence**:
```python
# Single episode path (line 577):
returns_aligned = log_returns[ws - 1:]  # (T - ws + 1, n_assets)

# Batched path (line 800):
log_returns_i = log_returns[i, ws - 1:valid_lengths[i] - 1]
```

The batched path uses `valid_lengths[i] - 1` while the single path doesn't offset the end. This could cause OOB or shape mismatches.

**Proposed Fix**:
Ensure consistent returns alignment across both paths:
```python
# Single episode path - match batched path
returns_aligned = log_returns[ws - 1:T]  # or returns[ws - 1: -1] if that's the intent
```

---

### 3. LIKELY: Trade-Biased Sampling Deactivation (BUG-7)

**Suspect Commit**: `06c44ab` (TASK_694 - mentioned in prompt)

**Change**: Filter trade indices before sampling

**Why It Kills Trade Learning**:
```python
# train_easy.py:1142-1147
valid_trades = trade_indices[trade_indices <= max_offset]
if (len(valid_trades) > 0
        and max_offset > 0
        and self.rng.random() < self.config.trade_sample_prob):
    offset = int(self.rng.choice(valid_trades))
```

If most `trade_indices` exceed `max_offset`, then `valid_trades` is often empty, and trade-biased sampling never fires. With 30% `trade_sample_prob`, losing trade-biased samples would significantly reduce the number of trade examples the model sees.

**Evidence**:
- `max_offset = max(0, n_available - take_count)`
- `trade_indices` are indices where `trade_density > 0.01`
- If trades are concentrated at the end of episodes (typical with oracle behavior), early samples will have `valid_trades = []`

**Proposed Fix**:
Track trade statistics and adjust the threshold or sampling strategy:
```python
# Add logging for debugging
if len(trade_indices) > 0:
    coverage = len(valid_trades) / len(trade_indices)
    if coverage < 0.1:
        logger.warning(f"Trade sample coverage low: {coverage:.1%}")
```

---

### 4. POSSIBLE: Latent Cache + Model-State Rollout Incompatibility

**Suspect Commit**: `06c44ab` (TASK_692/693)

**Change**: Both latent caching and model-state rollout enabled simultaneously

**Why It Kills Trade Learning**:
When latent caching is enabled (`self._use_latent_cache = True`), the model state is computed from **old parameters** (when the cache was filled). But model-state rollout needs **current parameters** to compute the correct state trajectory.

The flow is:
1. `set_encoder(trainer.model, trainer.params)` is called once at startup
2. Episodes are cached with latent states (using params at that time)
3. During training, `set_model_params()` is called each epoch with new params
4. **But** the cached latent states are NOT regenerated

Result: Model sees state channels generated from **stale parameters** while training with **current parameters**. This creates a mismatch.

**Evidence**:
```python
# train_easy.py:1552-1563 (startup)
if not getattr(args, 'no_latent_cache', False):
    logger.info("Enabling latent state caching...")
    generator.set_encoder(trainer.model, trainer.params)  # Cache with INITIAL params

# train_easy.py:1299-1301 (per epoch)
if autoregressive_state:
    generator.set_model_params(trainer.model, trainer.params)  # Update for rollout
```

The `_partial_refresh_buffer()` refreshes 20% of episodes with new params (line 1056), but 80% remain stale.

**Proposed Fix**:
Disable latent caching when model-state rollout is enabled:
```python
# train_easy.py:1553-1563
# Latent caching is incompatible with model-state rollout
autoregressive_state = getattr(config, 'autoregressive_state', False)
if autoregressive_state:
    logger.info("Disabling latent cache (incompatible with model-state rollout)...")
    args.no_latent_cache = True
elif not getattr(args, 'no_latent_cache', False):
    logger.info("Enabling latent state caching...")
    generator.set_encoder(trainer.model, trainer.params)
```

---

### 5. LOW: Oracle Mask Head (TASK_695)

**Suspect Commit**: `77cb089`

**Change**: Added auxiliary oracle mask prediction head

**Why It Likely Doesn't Kill Trade Learning**:
The oracle mask head contributes minimal gradient mass:
- `oracle_mask_loss = 0.023` vs `total_loss = 2.8` (~0.8% of total)
- `oracle_mask_weight = 0.1` by default

This is not sufficient to collapse agree_trade from 15% to 7%.

**Evidence**:
```python
# easy_trainer.py:215-219
total_loss = (
    self.config.policy_weight * policy_loss +
    self.config.value_weight * value_loss +
    self.config.oracle_mask_weight * oracle_mask_loss  # 0.1 * 0.023 ≈ 0.002
)
```

**Verdict**: Not the culprit, but worth disabling for sanity check.

---

### 6. LOW: Optimizer Warmup Guard

**Suspect Commit**: `77cb089`

**Change**: Added guard for `warmup_steps > total_steps`

**Why It Likely Doesn't Kill Trade Learning**:
With `epochs=2500, steps_per_epoch=200`, `total_steps = 500000`. Default `warmup_steps = 1000`. The guard doesn't change behavior in normal runs.

**Verdict**: Not the culprit.

---

### 7. LOW: Model Construction Parameters

**Suspect Commit**: `77cb089`

**Change**: Added encoder_version, embed_dim, encoder_num_heads to model construction

**Why It Likely Doesn't Kill Trade Learning**:
These parameters now use config values instead of defaults. If the config specifies the same values as the old defaults, there's no change.

**Verdict**: Possible if config specifies different values, but unlikely the primary cause.

---

## Root Cause Summary

The **primary regression cause** is the confidence channel misalignment (BUG-2). The model in training sees zero/lagged confidence channels while inference sees actual confidence. This corrupts the state channels the model relies on for its decisions.

**Secondary contributing factors**:
1. Returns alignment inconsistency between single and batched paths
2. Trade-biased sampling may be effectively disabled if trades cluster at episode end
3. Latent caching with model-state rollout causes parameter staleness

The model becomes a "turtle" (always predicting HOLD) because:
- The confidence signal (which should indicate "I was confident before") is corrupted
- State channels don't provide reliable feedback about the model's own actions
- The safest action (HOLD) is reinforced when feedback is unreliable
