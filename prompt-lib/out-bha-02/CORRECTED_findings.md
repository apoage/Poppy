# CORRECTED: Trade Action Learning Regression

## Cross-Validation Summary

After thorough analysis and cross-validation, my initial findings were **INCORRECT** about BUG-1 and BUG-2.

---

## TRUE ROOT CAUSE: Original Confidence Channel Bug (06c44ab)

### The Actual Bug

Commit `06c44ab` (TASK_692/693) introduced model-state rollout with a **critical bug**:

```python
# train_easy.py line ~241 (original commit 06c44ab)
conf_tiled = jnp.zeros((1, ws, n_assets))
```

The confidence channel (channel 33) was **ALWAYS ZEROS** during the entire autoregressive rollout.

### Impact

From epochs 50-77 (approximately), the model was trained with **zero confidence channels**:
- At every step, `conf_tiled = zeros`
- The model learned that confidence channel provides **no useful information**
- The model learned to **ignore the confidence channel** entirely
- This corrupted the training signal for trade actions

### Evidence

The regression `agree_trade` dropped from ~15% to ~7% starting from epoch 50. This aligns perfectly with when commit 06c44ab was introduced.

### Timeline

| Epoch | Git State | Confidence Channel | Model Behavior |
|--------|-----------|-------------------|----------------|
| 0-49 | Pre-06c44ab | N/A (27 channels, no conf) | Working (agree ~15%) |
| 50-77 | Post-06c44ab, pre-a86c206 | **ZEROS** | Model learns conf = useless |
| 78+ | Post-a86c206 | Prev_conf (lagged) | Model still ignores conf |

### The "BUG-2" Fix (a86c206)

Commit a86c206 changed the code from:
```python
conf_tiled = jnp.zeros((1, ws, n_assets))
```
to:
```python
conf_tiled = jnp.tile(prev_conf[None, None, :], (1, ws, 1))
```

This fix **aligns training with inference**, but comes **too late**. By epoch 78, the model had already spent ~28 epochs learning that confidence is useless.

### Why Model Still Fails (Epochs 78-90)

Even after the fix, the model continues to fail because:

1. **Learned Bias**: The model learned `confidence_channel → ignore`
2. **Entropy Injection**: Commit 06c44ab also added `oracle_entropy` mode (TASK_691) which injects noise into oracle selections
3. **Confidence Lag**: Even with the fix, confidence is lagged by one step (`prev_conf` from carry instead of `model_confidence`)
4. **Parameter Mismatch**: Latent cache uses stale parameters with model-state rollout

The combination of these factors causes the model to remain "turtled" (predicting HOLD everywhere).

---

## Secondary Contributing Factors

### 1. Latent Cache + Model-State Rollout Incompatibility

**Confirmed Issue**: When `autoregressive_state=True` and latent caching is enabled:
- Latent states are cached at startup with **initial parameters**
- During training, `set_model_params()` is called each epoch with **new parameters**
- `_partial_refresh_buffer()` only refreshes **20%** of episodes
- **80%** of buffer uses stale parameters throughout epoch

The model trains with **current parameters** but sees state channels generated from **old parameters**.

**Impact**: Creates mismatch between model's decision-making and the state it receives.

**Fix**: Disable latent cache when model-state rollout is enabled:
```python
# In train_easy.py, line 1553:
autoregressive_state = getattr(config, 'autoregressive_state', False)
if autoregressive_state:
    args.no_latent_cache = True  # Latent cache incompatible with model-state rollout
```

### 2. Oracle Entropy (TASK_691)

Commit 06c44ab added oracle entropy injection:
```python
p_drop, q_phantom = self._get_entropy_schedule()
if p_drop > 0 or q_phantom > 0:
    noisy_selections = corrupt_oracle_selections(
        oracle_selections, entropy_key, p_drop, q_phantom
    )
    target_position = compute_target_position(noisy_selections, log_returns)
```

When model-state rollout uses oracle-state (fallback), the entropy injection corrupts the target positions, providing noisy training signal.

### 3. Confidence Channel Timing

Even with the BUG-2 fix, the confidence channel is **lagged by one step**:
- At step 0: `prev_conf = 0` (from init)
- At step 1: `prev_conf = model_confidence from step 0`
- At step 2: `prev_conf = model_confidence from step 1`

The observation at step `w` uses `prev_conf` (from step `w-1`), which matches inference behavior where `prev_confidence` comes from the previous iteration.

However, this means the first step always sees zeros, and there's always a 1-step lag between model computing confidence and seeing it.

---

## Recommended Fixes

### 1. IMMEDIATE: Disable Latent Cache with Model-State Rollout

```python
# train_easy.py, around line 1553:
# Before enabling latent cache, check for autoregressive_state incompatibility
autoregressive_state = getattr(config, 'autoregressive_state', False)
if autoregressive_state:
    logger.warning("Disabling latent cache (incompatible with model-state rollout)...")
    args.no_latent_cache = True
elif not getattr(args, 'no_latent_cache', False):
    logger.info("Enabling latent state caching...")
    generator.set_encoder(trainer.model, trainer.params)
    val_generator.set_encoder(trainer.model, trainer.params)
```

### 2. REGENERATE BUFFER ON CHECKPOINT LOAD

When loading a checkpoint, clear the buffer:
```python
# train_easy.py, in main() after loading checkpoint:
if args.checkpoint:
    checkpoint = trainer.load_checkpoint(ckpt_path)
    start_epoch = checkpoint['epoch'] + 1
    # Clear buffer to avoid stale state channels from old parameters
    generator.buffer = []
    generator.buffer_sample_counts = []
    generator._buffer_param_ids = []
    logger.info("Buffer cleared after checkpoint load — regenerating with fresh params")
```

### 3. (Optional) Reduce Confidence Lag

To eliminate the 1-step confidence lag, use the current step's confidence instead of previous:

```python
# train_easy.py, line 242 - experimental:
# Use model_confidence directly instead of prev_conf (eliminates 1-step lag)
# This changes semantics but may help with convergence
conf_tiled = jnp.tile(model_confidence[None, None, :], (1, ws, 1))
```

This changes semantics but provides more immediate feedback to the model.

---

## Expected Outcome

After applying fix #1 (disable latent cache) and #2 (regenerate buffer on load):

- Model will see **consistent state channels** from **current parameters**
- Training-inference mismatch will be eliminated
- `agree_trade` should gradually recover from ~7% toward ~15% over ~50 epochs

Full recovery may require:
- Starting from a clean checkpoint (before epoch 50)
- Or retraining from scratch with the fixes applied
