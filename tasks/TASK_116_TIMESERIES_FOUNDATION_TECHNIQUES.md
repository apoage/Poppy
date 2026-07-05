# TASK_116: Time Series Foundation Model Techniques

## Status: IN PROGRESS (Student-t IMPLEMENTED)

## Goal

Integrate techniques from Chronos (Amazon) and MOIRAI (Salesforce) into Poppy's prediction head and potentially Bella's architecture.

---

## Quick Wins (Implement First)

### 1. Student-t Output Distribution
**Priority: HIGH** - Direct improvement to prediction head

Current: Gaussian (μ, σ) output assumes thin tails
Upgrade: Student-t (μ, σ, ν) with learnable degrees of freedom

```python
# Current
pred_mu, pred_sigma = physics_head(state)
loss = gaussian_nll(target, pred_mu, pred_sigma)

# Upgrade
pred_mu, pred_sigma, pred_df = physics_head(state)
loss = student_t_nll(target, pred_mu, pred_sigma, pred_df)
# df < 30 = fat tails, df → ∞ = Gaussian
```

**Why**: Financial returns have fat tails. Student-t models this natively without symlog hacks.

**Files to modify**:
- `mcts_trader/jax_agent/prediction/physics_core.py`
- `mcts_trader/jax_agent/trainer.py`

---

### 2. Variate Encoding
**Priority: HIGH** - Better multi-asset modeling

Current: 64 assets processed with implicit position
Upgrade: Explicit variate ID encoding (like MOIRAI)

```python
# Add learnable variate embeddings
variate_embed = nn.Embed(n_assets, d_model)
x = x + variate_embed(asset_ids)  # Before attention
```

**Why**: Lets attention distinguish BTC from ETH explicitly, not just by position.

**Files to modify**:
- `mcts_trader/jax_agent/trm_flax.py` (encoder)
- `mcts_trader/encoder/state.py`

---

### 3. RoPE Embeddings
**Priority: MEDIUM** - Modern position encoding

Current: Learned absolute position embeddings (or none)
Upgrade: Rotary Position Embeddings (relative, extrapolates)

```python
def apply_rope(q, k, positions):
    """Rotate queries and keys by position-dependent angles."""
    freqs = 1.0 / (10000 ** (jnp.arange(0, d, 2) / d))
    angles = positions[:, None] * freqs[None, :]
    cos, sin = jnp.cos(angles), jnp.sin(angles)
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot
```

**Why**: Better generalization to different sequence lengths, relative positions matter more than absolute.

**Files to modify**:
- `mcts_trader/jax_agent/trm_flax.py`

---

## Optional Modules (Enable via Flag)

### 4. Price Tokenization (Chronos-style)
**Priority: LOW** - Experimental

Convert continuous prices to discrete tokens:
```python
def tokenize_prices(prices, n_bins=4096):
    scaled = prices / jnp.abs(prices).mean()
    bins = jnp.linspace(-10, 10, n_bins)
    tokens = jnp.digitize(scaled, bins)
    return tokens
```

**Flag**: `--tokenize-prices --price-vocab-size 4096`

**Why**: Language model techniques might help. Worth testing.

---

### 5. Multi-Patch Encoding
**Priority: MEDIUM** - For different timeframes

Different patch sizes for different frequencies:
```python
patches = {
    'micro': patch_1d(prices, patch_size=4),   # Tick-level
    'meso': patch_1d(prices, patch_size=16),   # Minute-level
    'macro': patch_1d(prices, patch_size=64),  # Hour-level
}
# Concatenate and process together
```

**Flag**: `--multi-scale-patches`

**Why**: Market has structure at multiple timescales.

---

### 6. Any-Variate Attention (MOIRAI-style)
**Priority: MEDIUM** - Flatten then encode

Instead of (batch, time, assets, features):
```python
# Flatten assets into sequence
x_flat = x.reshape(batch, time * assets, features)
# Add variate encoding
variate_ids = jnp.repeat(jnp.arange(assets), time)
x_flat = x_flat + variate_embed(variate_ids)
# Single attention over everything
```

**Flag**: `--any-variate-attention`

**Why**: Cross-asset attention without separate heads.

---

### 7. SwiGLU Activation
**Priority: LOW** - Easy upgrade

Replace GELU/ReLU in FFN:
```python
def swiglu(x, W1, W2, W3):
    return (nn.silu(x @ W1) * (x @ W2)) @ W3
```

**Flag**: `--use-swiglu`

**Why**: Slightly better than GELU in transformers.

---

### 8. Mixture Distribution Output
**Priority: MEDIUM** - For Bella

Output mixture of distributions for generation:
```python
# Mixture of Student-t
weights = softmax(mixture_logits)  # K components
mus, sigmas, dfs = per_component_params
# Sample from mixture
```

**Flag**: `--mixture-output --n-mixture-components 3`

**Why**: Captures multimodal uncertainty (crash OR rally, not average).

---

## Implementation Plan

### Phase 1: Quick Wins
- [x] Student-t output distribution (2026-01-26)
- [ ] Variate encoding
- [ ] RoPE embeddings
- [x] Add CLI flags (Student-t: --use-student-t)

### Phase 2: Optional Modules
- [ ] Price tokenization (experimental)
- [ ] Multi-patch encoding
- [ ] Any-variate attention
- [ ] SwiGLU activation

### Phase 3: Bella Integration
- [ ] Mixture distribution output
- [ ] Evaluate on generation quality

---

## CLI Flags to Add

```bash
# Quick wins (default on after validation)
--student-t-output          # Use Student-t instead of Gaussian
--variate-encoding          # Add asset ID embeddings
--use-rope                  # Rotary position embeddings

# Optional modules
--tokenize-prices           # Chronos-style tokenization
--price-vocab-size 4096     # Vocabulary size for tokenization
--multi-scale-patches       # Multiple patch sizes
--any-variate-attention     # Flatten assets into sequence
--use-swiglu                # SwiGLU activation in FFN
--mixture-output            # Mixture distribution output
--n-mixture-components 3    # Number of mixture components
```

---

## References

- [Chronos Paper](https://arxiv.org/abs/2403.07815) - Tokenization, T5 architecture
- [MOIRAI Paper](https://arxiv.org/abs/2402.02592) - Any-variate, multi-patch, mixture output
- [Chronos-Bolt](https://arxiv.org/abs/2410.08378) - Patching for speed
- [RoPE Paper](https://arxiv.org/abs/2104.09864) - Rotary embeddings
- [SwiGLU Paper](https://arxiv.org/abs/2002.05202) - GLU variants

---

## Notes

- Student-t is the highest impact, lowest effort change
- Variate encoding is almost free and helps interpretability
- Tokenization is risky - could lose precision, needs careful testing
- These are **representation learning** techniques, applicable to both forecasting and generation

---

## Completed Implementations

### Student-t Output Distribution (2026-01-26)

**Files created/modified**:
- `mcts_trader/jax_agent/prediction/student_t.py` (NEW)
- `mcts_trader/jax_agent/prediction/__init__.py`
- `mcts_trader/jax_agent/trainer.py`
- `mcts_trader/training/config.py`
- `mcts_trader/training/setup.py`
- `mcts_trader/training/cli.py`

**CLI Usage**:
```bash
python train_jax_fast.py --use-prediction-loss --use-student-t
```

**Technical Details**:
- Uses adaptive sigma based on error statistics (no learnable params needed initially)
- Fixed df from config midpoint: (df_min + df_max) / 2
- Hybrid loss: `mse_weight * MSE + student_t_weight * NLL`
- Default weights: MSE=0.2, Student-t=0.8

**Module functions available**:
- `student_t_nll(target, mu, sigma, df)` - Raw NLL computation
- `student_t_loss(target, mu, log_sigma, log_df)` - Bounded params version
- `StudentTHead(hidden_dim, n_assets)` - Flax module for learnable uncertainty
- `hybrid_prediction_loss()` - Combined MSE + Student-t
- `maybe_student_t_loss()` - Easy integration wrapper
