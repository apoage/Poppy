# TASK_683: Widen Encoder and State Dimensions

**Status**: DONE
**Priority**: 1 (highest)
**Consensus**: All 4 analyses agree this is the primary bottleneck
**Depends on**: TASK_684 (hidden_dim decoupling)

## Problem

The V3 encoder compresses (27 channels × 64 timesteps × 64 assets) = 110,592 values into a 64-dim state vector. This is a 1,728:1 compression ratio. The model has exactly **1 latent dimension per asset**.

- Encoder params: 0.14M (1% of total model)
- Reasoning params: 14M (96% of model) — operating on impoverished 64-dim input
- agree_trade stuck at 43% because specific trade signals (which asset, which direction) are lost in compression
- agree_hold at 78% because "don't trade" is a safe global default learnable from compressed state

Oracle ambiguity analysis confirms theoretical ceiling is 97.1% — the bottleneck is model, not data.

## Changes

### 1. `mcts_trader/jax_agent/trm_encoder_v3.py`

No code changes needed — `embed_dim` and `state_dim` are already class parameters. Just needs them passed from above.

### 2. `mcts_trader/jax_agent/trm_flax.py` (line 233)

Pass `embed_dim` and `num_heads` to `StreamingAssetEncoder`:

```python
# Before:
self.encoder = StreamingAssetEncoder(state_dim=self.state_dim, dtype=self.dtype, param_dtype=self.param_dtype)

# After:
self.encoder = StreamingAssetEncoder(
    state_dim=self.state_dim,
    embed_dim=self.embed_dim,
    num_heads=self.num_heads,
    dtype=self.dtype,
    param_dtype=self.param_dtype,
)
```

Add `embed_dim` and `num_heads` as class params to `TinyRecursiveTradingModel`.

### 3. `mcts_trader/jax_agent/muzero_model.py`

Add `embed_dim` param to `MuZeroNetworks`, pass to TRM.

### 4. `easy_config.py`

Add `embed_dim: int = 32` field. Update presets:
- small: embed_dim=32, state_dim=64
- medium: embed_dim=32, state_dim=64
- large: embed_dim=64, state_dim=128

### 5. `easy_trainer.py`

Pass `embed_dim` to `MuZeroNetworks`.

## Target Config (combined with TASK_684)

```python
# "large_v2" preset
embed_dim: int = 64         # Was 32 — 2x temporal capacity
state_dim: int = 128        # Was 64 — 2 dims per asset
hidden_dim: int = 1024      # Was state_dim*16=1024 — decoupled (TASK_684)
num_heads: int = 8          # Keep at 8 (128/8=16 dim/head)
```

## Expected Impact

- Trade accuracy: 43% → 55-60% (+12-17pp)
- Balanced accuracy: 61% → 70-75% (+9-14pp)
- Param increase: 14.6M → ~30-50M (mostly in reasoning Dense layers due to wider state)
- VRAM: May need batch_size 1024→512

## Verification

1. Fresh training run from scratch (cannot resume — architecture changed)
2. Monitor first 50 epochs for improvement trajectory
3. Compare train/val gap (should still be <1%)
4. If VRAM OOM at batch=1024, reduce to 512
