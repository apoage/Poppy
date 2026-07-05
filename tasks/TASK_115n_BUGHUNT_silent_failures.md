# Bug Hunt Report: Silent Failures and Numerical Instabilities

**Date**: 2026-01-25
**Hunter**: Gemini (Audit Mode)
**Focus**: Shape mismatches, silent NaNs, masked failures, and incorrect broadcasting.

## Critical Findings

### BUG-001: PopArt Normalization Polluted by Invalid Replay Buffer Samples
**File**: `mcts_trader/jax_agent/trainer.py:1175`
**Severity**: **HIGH**
**Status**: OPEN

```python
            # PopArt: normalize targets and correct value head weights (TASK_021)
            # PERF: Only update PopArt stats every 10 iterations to reduce GPU sync overhead
            if self.use_popart and self.popart is not None:
                raw_values = batch['values'].flatten()  # <--- INCLUDES INVALID SAMPLES

                # Update statistics less frequently (amortizes sync cost)
                if iteration % 10 == 0:
                    norm_values, corrected_params = self.popart.normalize_and_correct(
                        raw_values, self.state.params
                    )
```

**Problem**: 
The `replay_buffer.sample_batch` returns a batch that may contain invalid/stale/zero-padded samples (marked by `batch['valid']`). 
`PopArtNormalizer` receives `raw_values` which includes these garbage values. 
It computes `mean` and `std` over the entire `raw_values` array, effectively biasing the value normalization towards 0 (or whatever stale data is in the buffer).
This is catastrophic for value learning, especially when the buffer wraps around or during early training.

**Fix**:
1. Pass `batch['valid']` mask to `normalize_and_correct`.
2. Update `mcts_trader/jax_agent/popart.py` to accept a mask and compute weighted mean/variance.

---

### BUG-002: VAE Loss Numerical Instability
**File**: `diff_teacher/encoder.py:493`
**Severity**: **MEDIUM**
**Status**: OPEN

```python
    # KL divergence: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
    kl_loss = -0.5 * jnp.mean(1 + log_var - mu**2 - jnp.exp(log_var))
```

**Problem**:
`log_var` is the output of a Linear layer and is unbounded. If `log_var` > 88 (float32), `jnp.exp(log_var)` overflows to Infinity.
This results in `kl_loss` being `-inf` (or `NaN` if combined with other terms).
This silently kills training or causes gradients to explode/vanish.

**Fix**:
Clamp `log_var` before exponentiation or use `softplus` for variance parameterization.
```python
jnp.exp(jnp.clip(log_var, max=20.0))
```

---

### BUG-005: Prediction Loss Trains on Garbage Data
**File**: `mcts_trader/jax_agent/trainer.py:721`
**Severity**: **HIGH**
**Status**: OPEN

```python
        # Total prediction loss combines decoder reconstruction + trajectory prediction
        loss_prediction = jnp.mean(weighted_loss) + loss_decoder
```

**Problem**:
`_compute_prediction_loss` calculates the mean over the **entire batch**, completely ignoring `batch['valid']`.
The `weighted_loss` and `loss_decoder` terms include contributions from invalid, stale, or zero-padded trajectories in the replay buffer.
This causes the physics/prediction head (and the shared encoder) to learn from garbage data, likely destabilizing representation learning.

**Fix**:
1. Pass `valid_mask` to `_compute_prediction_loss`.
2. Apply mask to `weighted_loss` and `loss_decoder` before averaging.
   ```python
   loss_prediction = jnp.sum(weighted_loss * valid_mask) / jnp.sum(valid_mask)
   ```

---

### BUG-003: Potential NaN in Differential Reward Metrics
**File**: `mcts_trader/jax_agent/differential_reward.py:123`
**Severity**: LOW
**Status**: OPEN

```python
    metrics = {
        'agent_return': jnp.mean(agent_return),
        'market_return': jnp.mean(market_return),
        # ...
    }
```

**Problem**:
If `agent_return` has shape `(0,)` (empty batch), `jnp.mean` returns `NaN`.
While rare in fixed-size batch training, this function is general-purpose and might be called with empty arrays in some edge cases (e.g. evaluation).

**Fix**:
Add a check for empty arrays or use `jnp.mean` with a safe fallback.

---

### BUG-004: Unsafe Reduction in Market ODE Prediction
**File**: `mcts_trader/jax_agent/prediction/market_ode.py:522`
**Severity**: LOW
**Status**: OPEN

```python
L522: step_mse = jnp.mean((pred_steps - target_steps) ** 2, axis=-1)  # (B, K)
L527: prediction_loss = jnp.mean(weighted_mse)
```

**Problem**:
`jnp.mean(weighted_mse)` reduces over ALL dimensions (Batch and K steps). 
If specific steps are padded/masked (e.g. end of episode), this inclusion distorts the loss.
It assumes `weighted_mse` is already masked or valid.

**Fix**:
Explicitly handle masks if padding is expected in `pred_steps`.

---

---

## Hunt Section: 2026-01-25 (Claude Opus 4.5)

### Category 1: Incorrect Axis in Reduction Operations

#### BUG-006: Flow Matching Batch Normalization Collapses Batch Dimension
**File**: `mcts_trader/jax_agent/flow_matching.py:196-197`
**Severity**: CRITICAL
**Status**: OPEN

```python
batch_mean = jnp.mean(velocity, axis=0)
batch_std = jnp.std(velocity, axis=0) + eps
```

**Problem**: `velocity` shape is `(batch, sequence, latent_dim)`. Using `axis=0` reduces batch dimension → produces `(sequence, latent_dim)`. Per-batch normalization collapses, different samples interfere with each other.

**Fix**: Use proper per-sample or running statistics normalization.

---

#### BUG-007: Encoder Global Pooling Destroys Structure
**File**: `diff_teacher/encoder.py:254, 297`
**Severity**: HIGH
**Status**: OPEN

```python
x = jnp.mean(x, axis=(1, 2))  # (B, H)
pooled = jnp.mean(h, axis=(1, 2))
```

**Problem**: Input shape `(B, T, A, H)`. Global average pooling with `axis=(1, 2)` removes ALL temporal and asset structure. Generated sequences lack coherent asset-specific temporal patterns.

**Fix**: Use attention-based or hierarchical pooling that preserves structure.

---

#### BUG-008: Diversity Loss Measures Norm Variance Not Representation Diversity
**File**: `diff_teacher/adversarial_training.py:159, 162`
**Severity**: MEDIUM
**Status**: OPEN

```python
mean_latent = jnp.mean(flat, axis=0, keepdims=True)
variance = jnp.mean(jnp.sum((flat - mean_latent) ** 2, axis=1))
```

**Problem**: Computes "variance of L2 norms" across batch, not diversity in representation space. Mode collapse not properly detected.

**Fix**: Use covariance across batch or mutual information metrics.

---

### Category 2: Regime Index Out of Bounds

#### BUG-009: Volatility Array OOB in Skeleton Builder
**File**: `diff_teacher/skeleton/builder.py:244-245`
**Severity**: CRITICAL
**Status**: OPEN

```python
vol_map = jnp.array([0.9, 0.4, 0.3, 0.8, 0.2])  # crash, rally, sideways, volatile, calm
return vol_map[regime_seq]
```

**Problem**: `vol_map` has 5 elements (indices 0-4). If `regime_seq` contains values >= 5, JAX indexing silently wraps: `5 % 5 = 0`. Generator receives corrupted volatility conditioning without any error.

**Fix**: Add bounds check: `regime_seq = jnp.clip(regime_seq, 0, 4)`

---

#### BUG-010: Embedding Layer Wraps OOB Indices
**File**: `diff_teacher/skeleton/encoder.py:133-134`
**Severity**: HIGH
**Status**: OPEN

```python
embed = nn.Embed(num_embeddings=self.n_regimes, features=self.d_model)
embedded = embed(regime_sequence)
```

**Problem**: `nn.Embed(num_embeddings=5)` expects indices [0, 4]. OOB indices wrap: regime 5 → embedding 0. Different regimes produce identical embeddings.

**Fix**: Clamp before embedding: `regime_sequence = jnp.clip(regime_sequence, 0, self.n_regimes-1)`

---

#### BUG-011: Dict Default Silences Invalid Regime Names
**File**: `diff_teacher/flow_matching.py:510`
**Severity**: MEDIUM
**Status**: OPEN

```python
def regime_to_idx(regime: str) -> int:
    return REGIME_TO_IDX.get(regime, 4)  # Silent default
```

**Problem**: Any typo in regime name silently returns regime 4 (calm). Training proceeds with wrong regime.

**Fix**: Raise instead of default: `if regime not in REGIME_TO_IDX: raise ValueError(...)`

---

### Category 3: Skeleton Fields Used Without None Check

#### BUG-012: Shape Mismatch in SkeletonEncoder Broadcast
**File**: `diff_teacher/skeleton/encoder.py:244-257, 312-318`
**Severity**: CRITICAL
**Status**: OPEN

```python
# SkeletonEncoder returns 1D
combined = jnp.stack(embeddings, axis=0).mean(axis=0)  # (d_model,)
# SkeletonConditioner broadcasts wrong
gamma_s = jnp.broadcast_to(gamma_s, (batch_size, self.d_model))
```

**Problem**: Broadcast repeats same `(d_model,)` values across batch, broadcasting each scalar d_model times instead of batch times. Creates wrong batch conditioning.

**Fix**: Validate shapes, use proper batch expansion.

---

#### BUG-013: JAX Array len() in Traced Context
**File**: `diff_teacher/skeleton/types.py:94-99`
**Severity**: CRITICAL
**Status**: OPEN

```python
def get_regime_idx(self) -> int:
    if self.regime_sequence is not None and len(self.regime_sequence) > 0:
        return int(jnp.bincount(self.regime_sequence.astype(jnp.int32)).argmax())
```

**Problem**: `len()` on JAX Array raises `TypeError` in JIT contexts. Fails at trace time.

**Fix**: Use `self.regime_sequence.shape[0]` instead of `len()`.

---

#### BUG-014: Negative Event Time Wraps to Wrong Position
**File**: `diff_teacher/skeleton/encoder.py:170-183`
**Severity**: HIGH
**Status**: OPEN

```python
for event in events:
    if event.time < n_timesteps:
        event_mask = event_mask.at[event.time].set(event_emb)
```

**Problem**: Only checks `event.time < n_timesteps`, not `event.time >= 0`. Negative indices wrap and place events at wrong positions silently.

**Fix**: Add check: `if 0 <= event.time < n_timesteps:`

---

### Category 4: Off-by-One in Sequence Indexing

#### BUG-015: Skeleton n_timesteps Stored as n-1
**File**: `diff_teacher/skeleton/extraction.py:34-46, 250-270`
**Severity**: CRITICAL
**Status**: OPEN

```python
returns = jnp.diff(log_prices, axis=0)  # (n-1) from (n)
n_timesteps = returns.shape[0]  # This is n-1, not n!
```

**Problem**: When `extract_skeleton()` is called with prices of shape `(n, n_assets)`, it computes returns with shape `(n-1, n_assets)`. Then stores `n_timesteps = n-1`. Creates mismatch with generation.

**Fix**: Store original `n_timesteps` before diff, or document this clearly.

---

#### BUG-016: Encoder Window Under-read on Small Windows
**File**: `mcts_trader/env/jax_encoder.py:182, 187, 244, 251-254`
**Severity**: CRITICAL
**Status**: OPEN

```python
returns_5m = (prices[:, -1] - prices[:, -5]) / (prices[:, -5] + 1e-8)
returns_15m = (prices[:, -1] - prices[:, -15]) / (prices[:, -15] + 1e-8)
# ...
pct_60 = compute_percentile(prices, 60)
```

**Problem**: Encoder assumes window contains enough history for all lookback periods (5, 15, 30, 60 steps). If `window_size < 60`, indexing returns fewer elements silently, corrupting calculations.

**Fix**: Add bounds check: `assert window_size >= 60` or handle smaller windows.

---

### Category 5: NaN/Inf Edge Cases

#### BUG-017: Flow Matching Division by Zero Running Std
**File**: `mcts_trader/jax_agent/flow_matching.py:208`
**Severity**: CRITICAL
**Status**: OPEN

```python
normalized = (velocity - running_mean) / running_std
```

**Problem**: `running_std` can be zero on first call if velocity has uniform values. NaN propagates through training.

**Fix**: Add epsilon before use: `/ (running_std + eps)`

---

#### BUG-018: KL Divergence Division by Zero
**File**: `diff_teacher/evaluation/fidelity.py:136`
**Severity**: CRITICAL
**Status**: OPEN

```python
kl_div = np.sum(gen_hist * np.log(gen_hist / real_hist))
```

**Problem**: After epsilon addition, `real_hist[i] = 0` in bins where generated data has probability still causes Inf.

**Fix**: `np.log(np.maximum(gen_hist / real_hist, 1e-10))`

---

#### BUG-019: Hill Estimator Zero Sum
**File**: `diff_teacher/evaluation/stylized_facts.py:77-79`
**Severity**: CRITICAL
**Status**: OPEN

```python
log_ratios = np.log(tail_values / threshold)
alpha = len(tail_values) / np.sum(log_ratios)
```

**Problem**: `np.sum(log_ratios)` can be zero if all tail_values = threshold. Division by zero.

**Fix**: `/ np.maximum(np.sum(log_ratios), 1e-8)`

---

#### BUG-020: SSM Exponential Overflow
**File**: `diff_teacher/mamba/ssm.py:38, 96`
**Severity**: HIGH
**Status**: OPEN

```python
delta_A = jnp.exp(delta_t[:, :, None] * A)
delta_A = jnp.exp(jnp.einsum('bsd,dn->bsdn', delta, A))
```

**Problem**: If `delta * A` has large positive values, exp overflows → Inf. NaN cascade in state updates.

**Fix**: Clamp: `jnp.exp(jnp.clip(..., -20, 20))`

---

#### BUG-021: Log of Unchecked Base Price
**File**: `mcts_trader/jax_agent/fast_generator.py:313`
**Severity**: HIGH
**Status**: OPEN

```python
log_base = jnp.log(self.base_price)
```

**Problem**: If `self.base_price <= 0`, produces `-Inf`. Corrupts all downstream prices.

**Fix**: Add validation at init: `assert self.base_price > 0`

---

#### BUG-022: Temperature Division Unbounded
**File**: `mcts_trader/jax_agent/atc.py:55, 60, 63`
**Severity**: MEDIUM
**Status**: OPEN

```python
neg_sim = jnp.dot(...) / temperature
```

**Problem**: If temperature → 0, produces Inf in logits.

**Fix**: Bounds: `temperature = jnp.maximum(temperature, 0.01)`

---

### Category 6: Swapped Arguments (Currently Correct but Fragile)

#### BUG-023: Weight Index Assignment Confusion
**File**: `mcts_trader/jax_agent/trainer.py:468-481`
**Severity**: HIGH (Refactoring Risk)
**Status**: OPEN

```python
weight_upper = indices - lower_idx
weight_lower = 1.0 - weight_upper
targets = targets.at[jnp.arange(batch_size), lower_idx].set(weight_lower)
targets = targets.at[jnp.arange(batch_size), upper_idx].add(weight_upper)
```

**Problem**: Variable naming is counter-intuitive. Code is currently CORRECT but easy to swap during refactoring, producing wrong categorical distributions silently.

**Fix**: Rename to `weight_at_lower_bin` / `weight_at_upper_bin` for clarity.

---

#### BUG-024: Tensor Reshape Order Fragile
**File**: `mcts_trader/jax_agent/muzero_model.py:265-273`
**Severity**: HIGH (Refactoring Risk)
**Status**: OPEN

```python
decoded = self.latent_to_market(latent_state)  # (batch, n_assets*3)
decoded = decoded.reshape(batch, 3, self.n_assets)
```

**Problem**: Reshape to `(batch, 3, n_assets)` vs `(batch, n_assets, 3)` both compile but produce different results. Wrong order breaks dictionary unpacking silently.

**Fix**: Add shape assertion after reshape.

---

## Summary Table (All Findings)

| Bug ID | File | Severity | Category | Status |
|--------|------|----------|----------|--------|
| BUG-001 | trainer.py:1175 | HIGH | PopArt mask | OPEN |
| BUG-002 | encoder.py:493 | MEDIUM | VAE KL overflow | OPEN |
| BUG-003 | differential_reward.py:123 | LOW | Empty array | OPEN |
| BUG-004 | market_ode.py:522 | LOW | Mask reduction | OPEN |
| BUG-005 | trainer.py:721 | HIGH | Prediction mask | OPEN |
| BUG-006 | flow_matching.py:196 | CRITICAL | Axis collapse | OPEN |
| BUG-007 | encoder.py:254 | HIGH | Pooling structure | OPEN |
| BUG-008 | adversarial_training.py:159 | MEDIUM | Diversity metric | OPEN |
| BUG-009 | skeleton/builder.py:245 | CRITICAL | Regime OOB | OPEN |
| BUG-010 | skeleton/encoder.py:134 | HIGH | Embedding OOB | OPEN |
| BUG-011 | flow_matching.py:510 | MEDIUM | Dict default | OPEN |
| BUG-012 | skeleton/encoder.py:312 | CRITICAL | Broadcast shape | OPEN |
| BUG-013 | skeleton/types.py:96 | CRITICAL | JAX len() | OPEN |
| BUG-014 | skeleton/encoder.py:170 | HIGH | Negative index | OPEN |
| BUG-015 | skeleton/extraction.py:250 | CRITICAL | n_timesteps off-by-one | OPEN |
| BUG-016 | jax_encoder.py:182 | CRITICAL | Window under-read | OPEN |
| BUG-017 | flow_matching.py:208 | CRITICAL | Zero std division | OPEN |
| BUG-018 | fidelity.py:136 | CRITICAL | KL div zero | OPEN |
| BUG-019 | stylized_facts.py:77 | CRITICAL | Hill zero sum | OPEN |
| BUG-020 | mamba/ssm.py:96 | HIGH | Exp overflow | OPEN |
| BUG-021 | fast_generator.py:313 | HIGH | Log non-positive | OPEN |
| BUG-022 | atc.py:55 | MEDIUM | Temperature zero | OPEN |
| BUG-023 | trainer.py:479 | HIGH | Weight naming | OPEN |
| BUG-024 | muzero_model.py:267 | HIGH | Reshape order | OPEN |

## Action Plan

1.  **Fix BUG-001 (PopArt)**: Modify `popart.py` and `trainer.py` to use `batch['valid']`.
2.  **Fix BUG-005 (Prediction)**: Pass `valid_mask` to `_compute_prediction_loss` and use it.
3.  **Fix BUG-002 (VAE)**: Clamp `log_var` to [-20, 20].
4.  **Fix BUG-006, BUG-017**: Fix flow matching normalization axis and add epsilon guard.
5.  **Fix BUG-009, BUG-010**: Add regime index bounds checking/clamping.
6.  **Fix BUG-013**: Replace `len()` with `.shape[0]` in skeleton types.
7.  **Fix BUG-015, BUG-016**: Fix n_timesteps calculation and window size validation.
8.  **Fix BUG-018, BUG-019**: Add epsilon guards in evaluation metrics.
9.  **Fix BUG-020**: Clamp SSM exponentials.