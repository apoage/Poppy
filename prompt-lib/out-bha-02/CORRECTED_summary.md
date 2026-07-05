# CORRECTED: Trade Action Learning Regression — Executive Summary

## Problem

Between commits `06c44ab` and `77cb089`, `agree_trade` collapsed from **~15% to ~7%**. The model became a "turtle" — predicting HOLD everywhere instead of learning trade actions.

## Root Cause (CORRECTED)

The primary regression is **Original Confidence Channel Bug in commit 06c44ab**, NOT the BUG-2 fix.

### The True Bug

Commit `06c44ab` (TASK_692/693) introduced model-state rollout with a **critical bug**:

```python
# train_easy.py line ~241 (original commit 06c44ab)
conf_tiled = jnp.zeros((1, ws, n_assets))
```

The confidence channel (channel 33) was **ALWAYS ZEROS** during the entire autoregressive rollout.

### Impact

From epochs 50-77 (before a86c206 fix), the model was trained with **zero confidence channels**:
- At every step, `conf_tiled = zeros`
- The model learned that confidence channel provides **no useful information**
- The model learned to **ignore confidence channel** entirely
- This corrupted the training signal for trade actions

The regression `agree_trade` dropped from ~15% to ~7% starting from epoch 50 — aligns perfectly with when commit 06c44ab was introduced.

### Timeline

| Epoch | Git State | Confidence Channel | Model Behavior |
|--------|-------------|-------------------|----------------|
| 0-49 | Pre-06c44ab | N/A (27 channels, no conf) | Working (~15% agree_trade) |
| 50-77 | Post-06c44ab, pre-a86c206 | **ZEROS** | Model learns conf = useless |
| 78+ | Post-a86c206 | Prev_conf (lagged) | Model still ignores conf |

### Why Model Still Fails (Epochs 78-90)

Even after a86c206 "fix", the model continues to fail because:

1. **Learned Bias**: Model learned `confidence_channel → ignore` during 28 epochs
2. **Latent Cache Staleness**: 80% of cached episodes use old parameters
3. **Confidence Lag**: Confidence is still lagged by one step (prev_conf from carry vs current)

The combination causes the model to remain "turtled" (predicting HOLD everywhere).

---

## Contributing Factors

### 1. Latent Cache + Model-State Rollout Incompatibility

**Confirmed Issue**:

When `autoregressive_state=True` and latent caching is enabled:
- Latent states are cached at startup with **initial parameters**
- During training, `set_model_params()` is called each epoch with **new parameters**
- `_partial_refresh_buffer()` only refreshes **20%** of episodes
- **80%** of buffer uses stale parameters throughout epoch

The model trains with **current parameters** but sees state channels generated from **old parameters**.

**Impact**: Creates mismatch between model's decision-making and the state it receives.

**Fix**: Disable latent cache when model-state rollout is enabled.

### 2. Oracle Entropy (TASK_691)

Commit 06c44ab also added oracle entropy injection which may add noise to training signal.

---

## Recommended Fixes (Priority Order)

### 1. IMMEDIATE: Disable Latent Cache with Model-State Rollout

```python
# train_easy.py, around line 1553:
autoregressive_state = getattr(config, 'autoregressive_state', False)
if autoregressive_state:
    logger.warning("Disabling latent cache (incompatible with model-state rollout)...")
    args.no_latent_cache = True
```

### 2. REGENERATE BUFFER ON CHECKPOINT LOAD

When loading a checkpoint, clear buffer to avoid stale state:

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

To eliminate 1-step confidence lag, use current step's confidence instead of previous:

```python
# train_easy.py, line 242 - experimental:
conf_tiled = jnp.tile(model_confidence[None, None, :], (1, ws, 1))
```

This changes semantics but provides more immediate feedback.

---

## Expected Outcome

After applying fixes:
- Model will see **consistent state channels** from **current parameters**
- Training-inference mismatch will be eliminated
- `agree_trade` should gradually recover from ~7% toward ~15% over ~50 epochs

Full recovery may require starting from a clean checkpoint before epoch 50, or retraining from scratch.

## Files to Modify

- `train_easy.py`: Lines ~1553 (latent cache guard), ~1570 (buffer clear on load)
- Diagnostic logging to verify buffer staleness reduction

## Key Takeaway

The regression was NOT caused by the BUG-2 fix (a86c206), but by the **original bug** in commit 06c44ab that used `conf_tiled = jnp.zeros(...)`. The model learned to ignore confidence channel during ~28 epochs, and even after fixing the zeros bug, it couldn't unlearn that bias.
