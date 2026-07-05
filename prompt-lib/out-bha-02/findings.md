# Regression Findings: Trade Action Learning Collapse

---

### REGRESSION-1: Confidence Channel Misalignment (BUG-2)

**Suspect Commit**: `a86c206` BUG-2 completion: emit prev_conf in scan output for training alignment

**Change**: Confidence channel emission in `_build_model_state_trajectory` scan output

**Confidence**: CONFIRMED

**Why It Kills Trade Learning**:

The confidence channel (channel 33) provides the model with its own confidence from the previous timestep. During model-state rollout, the confidence signal is fed back into the next observation. However, the scan output emits `prev_conf` instead of `model_confidence`, creating a critical **training-inference mismatch**:

**Training** (train_easy.py:299-302):
```python
# Emit prev_conf (not model_confidence) so confidence_trajectory[w] =
# confidence model SAW at step w, matching inference alignment.
# model_confidence propagates forward via carry for step w+1.
return (new_pos, new_pnl, new_steps, new_h_fast, new_h_slow, model_confidence), (state_ch, prev_conf)
```

**Inference** (benchmark_easy.py:579-581):
```python
# 5. Update confidence from model's own softmax
model_softmax = np.array(jax.nn.softmax(logits_per_asset[0], axis=-1))  # (n_assets, 5)
prev_confidence = np.max(model_softmax, axis=-1)  # (n_assets,)
```

**The Bug**: `prev_conf` is initialized to `jnp.zeros(n_assets)` (line 215) and represents the **confidence from the previous iteration**. At the first scan step (idx=0):
- `prev_conf = zeros` (initial value)
- `conf_tiled = zeros` tiled across window (line 242)
- Observation at step 0 contains zero confidence

The model sees **zeros** instead of meaningful confidence signals. Even though `model_confidence` is computed from the actual softmax (line 261), it is stored in the carry for the NEXT iteration, while the current observation gets `prev_conf` (zeros).

**Evidence**:
- `agree_trade` dropped from ~15% to ~7% — model is not learning trade actions
- `agree_hold` stayed at ~94% — model learned to hold (safest action when feedback is corrupted)
- Confidence channel is supposed to help the model understand its own uncertainty, but it's providing zeros

**Impact**:
- Step 0: Model sees `conf = 0` for all assets
- Step 1+: Model sees `conf` from step `w-1` (lagged by one step)
- During inference: Model sees `conf` from its own previous prediction

The model in training cannot learn to use its confidence signal because it's always seeing zeros/lagged values.

**Proposed Fix**:
```python
# In train_easy.py, line 302, emit model_confidence instead of prev_conf:
# BEFORE:
return (new_pos, new_pnl, new_steps, new_h_fast, new_h_slow, model_confidence), (state_ch, prev_conf)

# AFTER:
return (new_pos, new_pnl, new_steps, new_h_fast, new_h_slow, model_confidence), (state_ch, model_confidence)
```

This ensures that `confidence_trajectory[w] = model_confidence` (the confidence the model just computed), which matches inference semantics where `prev_confidence` is updated with the current model's confidence.

---

### REGRESSION-2: Returns Alignment Inconsistency (BUG-1)

**Suspect Commit**: `06c44ab` TASK_692/693: Model-state rollout + fix 4 bugs in position/slicing logic

**Change**: Returns slicing for model-state rollout (single vs batched paths)

**Confidence**: LIKELY

**Why It Kills Trade Learning**:

The returns array alignment differs between the single-episode and batched paths:

**Single episode path** (train_easy.py:576-577):
```python
ws = self.config.window_size
returns_aligned = log_returns[ws - 1:]  # Shape: (T - ws + 1, n_assets)
```

**Batched path** (train_easy.py:800):
```python
log_returns_i = log_returns[i, ws - 1:valid_lengths[i] - 1]  # Different slicing
```

The single-episode path doesn't offset the end of the slice, while the batched path uses `valid_lengths[i] - 1`. This inconsistency can cause:
1. Shape mismatches between returns and observations
2. PnL calculation errors in the state channel computation
3. Incorrect value targets for the model

**Evidence**:
```python
# Observations shape: (n_valid, 27, ws, n_assets) where n_valid = T - ws + 1
# Single path returns: (T - ws + 1, n_assets) — matches
# Batched path returns: (valid_lengths[i] - 1 - (ws - 1), n_assets) — may differ
```

**Impact**:
If returns are misaligned, the PnL channel (ch28) in the state channels is wrong. The model receives incorrect feedback about the profitability of its actions.

**Proposed Fix**:
```python
# In train_easy.py, line 577, match the batched path:
# BEFORE:
returns_aligned = log_returns[ws - 1:]

# AFTER:
returns_aligned = log_returns[ws - 1:T]  # Explicitly match T dimension
```

---

### REGRESSION-3: Trade-Biased Sampling Deactivation (BUG-7)

**Suspect Commit**: `06c44ab` (mentioned in TASK_694)

**Change**: Filter trade indices before sampling

**Confidence**: LIKELY

**Why It Kills Trade Learning**:

```python
# train_easy.py:1142-1147
valid_trades = trade_indices[trade_indices <= max_offset]
if (len(valid_trades) > 0
        and max_offset > 0
        and self.rng.random() < self.config.trade_sample_prob):
    offset = int(self.rng.choice(valid_trades))
```

If `trade_indices` contains indices that exceed `max_offset` (which is typical if trades cluster at the end of episodes), then `valid_trades` is often empty, and the coin-flip for trade-biased sampling never fires.

With `trade_sample_prob = 0.30`, losing trade-biased samples means:
- 30% of batches SHOULD sample from trade-heavy timesteps
- If `valid_trades` is empty, these samples fall back to uniform sampling
- The model sees ~30% fewer trade examples than intended

**Evidence**:
- `trade_indices` are computed as `np.where(trade_density > 0.01)[0]` (line 989)
- `trade_density` is the complement of hold probability (1 - mean of action 1)
- Oracle tends to cluster trades around profitable regimes, often at episode end
- `max_offset = max(0, n_available - take_count)` — decreases as episode is sampled

**Impact**:
Reduced trade example frequency directly explains `agree_trade` dropping from 15% to 7%.

**Proposed Fix**:
Add diagnostic logging and adjust sampling strategy:
```python
# In train_easy.py, after line 1144:
if self.rng.random() < self.config.trade_sample_prob:
    if len(valid_trades) > 0:
        offset = int(self.rng.choice(valid_trades))
    else:
        # Fallback: if no valid trades, sample uniformly but log
        logger.debug(f"Trade sample: {len(trade_indices)} trades, 0 valid (max_offset={max_offset})")
        offset = self.rng.integers(0, max_offset + 1) if max_offset > 0 else 0
```

---

### REGRESSION-4: Latent Cache + Model-State Rollout Incompatibility

**Suspect Commit**: `06c44ab` TASK_692/693

**Change**: Both latent caching and model-state rollout enabled simultaneously

**Confidence**: LIKELY

**Why It Kills Trade Learning**:

When latent caching is enabled, latent states are pre-computed once at startup:
```python
# train_easy.py:1554-1556
if not getattr(args, 'no_latent_cache', False):
    logger.info("Enabling latent state caching...")
    generator.set_encoder(trainer.model, trainer.params)  # Cache with INITIAL params
```

But model-state rollout requires current parameters to compute accurate state trajectories:
```python
# train_easy.py:1299-1301
if autoregressive_state:
    generator.set_model_params(trainer.model, trainer.params)  # Update for rollout
```

The problem: Latent states are cached with **initial parameters** and never regenerated (except 20% via `_partial_refresh_buffer`). The model trains with current parameters but sees state channels computed from old parameters.

**Evidence**:
```python
# train_easy.py:1058-1093
def _partial_refresh_buffer(self):
    """Refresh a fraction of buffer episodes with new params."""
    n_refresh = max(1, int(n_total * 0.2))  # Only 20%
    # Mark entries for regeneration
    for idx in indices_to_refresh:
        self.buffer_sample_counts[idx] = self.SAMPLES_PER_EPISODE
```

80% of the buffer uses stale parameters throughout the epoch.

**Impact**:
State channels (pos, pnl, heat, etc.) don't match the model's current decision-making. The model receives inconsistent feedback and learns to ignore state channels, falling back to HOLD.

**Proposed Fix**:
Disable latent caching when model-state rollout is enabled:
```python
# In train_easy.py:1553-1563
autoregressive_state = getattr(config, 'autoregressive_state', False)
if autoregressive_state:
    # Latent caching is incompatible with model-state rollout
    logger.info("Disabling latent cache (incompatible with model-state rollout)...")
    args.no_latent_cache = True
elif not getattr(args, 'no_latent_cache', False):
    logger.info("Enabling latent state caching...")
    generator.set_encoder(trainer.model, trainer.params)
```

---

### REGRESSION-5: Oracle Mask Head (TASK_695) - UNLIKELY CULPRIT

**Suspect Commit**: `77cb089` TASK_695: Auxiliary oracle mask prediction head

**Change**: Added auxiliary oracle mask prediction head

**Confidence**: LOW (Unlikely)

**Why It Likely Doesn't Kill Trade Learning**:

The oracle mask head contributes minimal gradient mass:
- `oracle_mask_loss = 0.023` from epoch 77 metrics
- `total_loss ≈ 2.8`
- `oracle_mask_loss / total_loss ≈ 0.008` (0.8%)
- Default `oracle_mask_weight = 0.1`

This tiny loss contribution cannot explain a 53% drop in `agree_trade`.

**Evidence**:
```python
# easy_trainer.py:215-219
total_loss = (
    self.config.policy_weight * policy_loss +
    self.config.value_weight * value_loss +
    self.config.oracle_mask_weight * oracle_mask_loss  # 0.1 * 0.023 ≈ 0.0023
)
```

**Verdict**: Not the primary culprit, but can be disabled for sanity check using `--no-oracle-mask-head`.

---

### REGRESSION-6: Optimizer Warmup Guard - NOT A CULPRIT

**Suspect Commit**: `77cb089`

**Change**: Added guard for `warmup_steps > total_steps`

**Confidence**: NOT A CULPRIT

**Why It Doesn't Kill Trade Learning**:

With `epochs=2500, steps_per_epoch=200`, `total_steps = 500000`. Default `warmup_steps = 1000`. The guard `effective_warmup = min(config.warmup_steps, max(total_steps - 1, 0))` doesn't change behavior in normal runs.

---

### REGRESSION-7: Model Construction Parameters - NOT A CULPRIT (UNLESS CONFIG CHANGED)

**Suspect Commit**: `77cb089`

**Change**: Added encoder_version, embed_dim, encoder_num_heads to model construction

**Confidence**: NOT A CULPRIT (unless config specifies different values)

**Why It Likely Doesn't Kill Trade Learning**:

These parameters now use config values instead of defaults. If the config specifies the same values as the old defaults, there's no change.

**Evidence**:
```python
# easy_trainer.py:54-57
encoder_version=getattr(config, 'encoder_version', 2),
hidden_dim=getattr(config, 'trm_hidden_dim', None),
embed_dim=getattr(config, 'embed_dim', 32),
encoder_num_heads=getattr(config, 'encoder_num_heads', 4),
```

**Verdict**: Worth verifying config values match expected defaults, but unlikely the primary cause.

---

## Recommended Fix Order

1. **FIX BUG-2 FIRST** (train_easy.py:302): Change `prev_conf` to `model_confidence` in scan output
2. **FIX BUG-1 SECOND** (train_easy.py:577): Ensure consistent returns alignment
3. **FIX LATENT CACHE** (train_easy.py:1553): Disable latent cache with model-state rollout
4. **ADD DIAGNOSTICS** (train_easy.py:1144): Log trade sample coverage

After applying these fixes, expect `agree_trade` to recover to ~15% by epoch 80.
