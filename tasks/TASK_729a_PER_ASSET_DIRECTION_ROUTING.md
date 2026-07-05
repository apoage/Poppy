# TASK_729a: Per-Asset Direction Routing — Break the Global-Latent Bottleneck

**Status**: READY
**Priority**: CRITICAL — all post-726 work confirmed dir_acc ≈ 0.59 is the true ceiling
**Depends on**: TASK_728c (encoder unfreeze, direction refiner)
**Scope**: MVP only. One encoder change, one head rewrite. No architecture tour.
**Budget**: ~1 hr implementation, 1 overnight training run.

---

## The One Hypothesis

Five sequential fixes (726→727→728→728b→728c) all moved their target metric and all hit the same wall: Bal.Acc ~34%, bounded by `dir_acc ≈ 0.59`. The common bottleneck is the encoder's final global projection:

```
per-asset features (B, 64, embed_dim=64)
        │
        │  global pooling + Dense → (B, 128)        ← information loss here
        ▼
global latent (B, 128)
        │
        ├── policy head  → (B, 320)
        ├── value head   → (B, 401)
        └── direction head → (B, 64)   ← per-asset logits extracted from global latent
```

The per-asset features already exist inside the encoder with shape `(B, 64, embed_dim)` — they *are* per-asset by construction. The direction head is reading them *after* they've been mashed through a 64→128 global projection. That projection is what's capping `dir_acc`.

**Fix**: give `DirectionHead` direct access to per-asset features. Policy head stays unchanged (still reads global latent) — refiner (TASK_728) already turns improved direction_logits into per-asset policy bias.

---

## Implementation (Minimum Viable)

### Change 1 — Expose per-asset features from the encoder

File: `mcts_trader/jax_agent/trm_encoder_v2.py` (or whichever V2 file is live).

Find where the encoder computes per-asset features before the final global projection. Store them as an attribute or return them as a second output.

Path A (cleanest, if you control the encoder call): return a tuple `(global_latent, per_asset_features)` where `per_asset_features.shape == (B, n_assets, embed_dim)`.

Path B (least invasive, if encoder is called from many places): have `TinyRecursiveTradingModel.encode()` call the encoder and *also* stash the pre-pool features via a small wrapper. Return both from `representation()`.

Pick whichever minimises diff. Target: a tensor of shape `(B, 64, D)` where `D = embed_dim` (currently 64) reaches `MuZeroNetworks.initial_inference`.

### Change 2 — Rewrite `DirectionHead` to consume per-asset features

File: `mcts_trader/jax_agent/muzero_model.py`.

```python
class DirectionHead(nn.Module):
    """TASK_729a: Per-asset features in, per-asset logits out. No cross-asset mixing."""
    n_assets: int = 64
    hidden_dim: int = 128  # Smaller than 256 — per-asset head doesn't need width
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, per_asset_feats: jnp.ndarray) -> jnp.ndarray:
        # per_asset_feats: (B, n_assets, D)
        x = nn.Dense(features=self.hidden_dim, dtype=self.dtype,
                     param_dtype=self.param_dtype, name='dir_1')(per_asset_feats)
        x = nn.gelu(x)
        x = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype,
                         param_dtype=self.param_dtype, name='dir_ln')(x)
        # One scalar per asset
        logits = nn.Dense(features=1, dtype=self.dtype,
                          param_dtype=self.param_dtype, name='dir_out')(x)
        return logits.squeeze(-1).astype(jnp.float32)  # (B, n_assets)
```

Key shift: no cross-asset attention inside the head. Each asset's logit is a pure function of its own feature vector. If the encoder's per-asset feature already carries enough direction information, this is all we need.

### Change 3 — Wire per-asset features into `initial_inference` and `prediction_only`

Where direction_head is called, pass `per_asset_feats` instead of `latent_state`. Everything else (policy head, policy_direction_refiner, value head) stays unchanged.

### Change 4 — Handle checkpoint compatibility

Old checkpoints have `direction_head` params shaped for `(128 → 256 → 64)`. New head is `(D → 128 → 1)` applied per asset. These are incompatible — `initialize_missing_aux_heads` currently only fills *missing* subtrees, not reshape mismatches.

Simplest fix in `initialize_missing_aux_heads`: if `direction_head` shape in ckpt doesn't match fresh init, **replace it wholesale** with the fresh init. Log a warning. `policy_direction_refiner` weights are still compatible (same shape) and we keep them.

```python
# In initialize_missing_aux_heads, after model.init:
for k, v in fresh.items():
    if k in existing_keys and k == 'direction_head':
        # Shape-check and replace on mismatch
        try:
            jax.tree_util.tree_map(lambda a, b: a.shape == b.shape, existing[k], v)
        except Exception:
            logger.warning(f"direction_head shape mismatch — replacing with fresh init")
            new_params[k] = v
```

Actually simpler: just always fresh-init `direction_head` when `use_direction_head=True` and a CLI flag `--reset-direction-head` is set. Deterministic, no tree_map gymnastics.

### Change 5 — CLI

```python
parser.add_argument('--reset-direction-head', action='store_true',
                    help='Fresh-init direction_head params (use when head arch changed)')
```

---

## Training Command

Warmstart from **728c's latest `e250.pkl`** (or `e200.pkl` — whichever has the trained encoder + trained refiner). The encoder has been tuning for 200+ epochs; don't throw that away.

```bash
mkdir -p jax_ckpt_easy_phase2_729a && nohup python train_phase2.py \
  --checkpoint jax_ckpt_easy_phase2_728c/phase2_stageB_e250.pkl \
  --stage B --epochs 200 \
  --periods-file config/foxglove_periods.json --n-slots 17 --max-runs-per-group 10 \
  --factored --factored-temperature 0.1 \
  --no-reset-value-head --value-target-mode nstep --td-discount 0.95 \
  --value-weight 0.0 --bc-weight-min 0.05 --bc-max-trade-weight 4.0 \
  --direction-weight 0.5 --encoder-lr 3e-5 \
  --reset-direction-head \
  --output-dir jax_ckpt_easy_phase2_729a \
  > jax_ckpt_easy_phase2_729a/stage_b_nohup.log 2>&1 &
```

Note: 200 epochs, not 300 — we will know by E100 if this broke the ceiling or not. No point grinding past a plateau.

---

## Diagnostic Gates — Binary Outcome

### E50
- `dir_acc` > 0.63 → **encoder feature quality is sufficient**, bottleneck was routing. Keep training.
- `dir_acc` < 0.60 → **per-asset features are also capacity-limited**, not just the global projection. This is the terminal signal — no more loss-tweak runs will help; the encoder architecture itself needs widening (TASK_729b) or replacement.

### E100 / benchmark
- `dir_acc` > 0.70 AND Bal.Acc > 42% → TASK_729a is the answer. Productize.
- `dir_acc` > 0.65 AND Bal.Acc > 38% → working but marginal. Consider stacking (widen head, raise direction weight).
- `dir_acc` still ≈ 0.59 → definitive: encoder is the ceiling. Stop iterating on losses/heads.

---

## Why This Is MVP and Not More

Deliberately excluded:
- Per-asset policy head routing (would require re-architecting TRM reasoning — big)
- Encoder widening (try routing first; widening is next if routing isn't enough)
- Cross-asset attention variants (refactoring rabbit hole)
- Value head changes (value_weight=0 in current runs, not the problem)

One change, one head, one training run. Decisive diagnostic.

---

## Files Modified

| File | Change |
|------|--------|
| `mcts_trader/jax_agent/trm_encoder_v2.py` | Expose per-asset features alongside global latent |
| `mcts_trader/jax_agent/trm_flax.py` | Thread `per_asset_feats` through `encode()` |
| `mcts_trader/jax_agent/muzero_model.py` | `DirectionHead` consumes per-asset features |
| `train_phase2.py` | `--reset-direction-head` CLI flag + shape-mismatch handling |

No Phase2Config change (direction head hidden dim can be adjusted via existing `direction_hidden_dim` field if needed).

---

## Cost Ceiling

If E50 `dir_acc` is still stuck at 0.59, STOP. That result alone settles the question — the encoder architecture is the ceiling and no amount of head/loss engineering will save it. Further runs only burn electricity. Kill the training immediately, document the result, and the next decision is about whether to widen the encoder (TASK_729b) or accept the current baseline and move to deployment.

Gate discipline: don't let the 200-epoch budget run if the E50 signal is flat.
