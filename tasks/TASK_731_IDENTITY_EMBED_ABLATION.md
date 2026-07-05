# TASK_731: `asset_identity_embed` Ablation

**Status**: CONTINGENT — runs in parallel with or after TASK_730 if V3 hits ceiling.
**Priority**: Cheapest test in the pivot stack. <1 hour implementation, ~2-3 hr verdict.
**Rationale**: `research/729c_ceiling_alternatives.md` §2; also flagged in `lore/notes_2026-04-19.md:437,457` but never executed.

---

## Hypothesis

The `asset_identity_embed` parameter in both V2 (`trm_encoder_v2.py:68`) and V3 (`trm_encoder_v3.py:129-133`) adds a learned per-asset positional embedding before cross-asset attention. This embedding gives the model a fixed "which asset am I looking at" signal.

**Suspicion**: under sparse per-asset oracle labels (93.75% HOLD), the encoder may learn to route direction-sensitive behavior through the identity embed itself (constant per asset) rather than extracting direction from content features. This would explain:
- dir_acc plateaus at ~0.57 across V2, V2+wider, V3 (all use identity embed)
- 0.5% noise collapse to HOLD (noise cannot perturb the constant identity feature, but can perturb signal features — so the policy falls back on the identity bias, which says "trade none")

---

## The Change

Add a boolean flag `use_identity_embed: bool = True` to both encoders. When False, skip the identity add step.

### V2 (`trm_encoder_v2.py`)
```python
class AssetTransformerEncoder(nn.Module):
    ...
    use_identity_embed: bool = True

    @nn.compact
    def __call__(self, x, deterministic=True, return_per_asset=False):
        ...
        if self.use_identity_embed:
            x_assets = x_assets + identity_embed(...)
        ...
```

### V3 (`trm_encoder_v3.py::_post_recurrence`)
```python
if self.use_identity_embed:
    pos_embed = self.param('asset_identity_embed', ...)
    h = h + pos_embed[:, :n_assets, :]
```

### Config / CLI
```python
# easy_config.py + phase2_config.py
use_identity_embed: bool = True
```

```python
# train_easy.py + train_phase2.py
parser.add_argument('--no-identity-embed', action='store_true',
                    help='Ablate asset_identity_embed (TASK_731 diagnostic)')
if args.no_identity_embed:
    config.use_identity_embed = False
```

---

## Test Plan

### Phase A — Phase-1-Lite without identity embed
Fresh encoder (shape change from removing identity param), so must retrain Phase 1. Use the 20-epoch Phase-1-Lite protocol from 729b.

```bash
mkdir -p jax_ckpt_easy_phase1_no_ident && CUDA_VISIBLE_DEVICES=0 nohup python train_easy.py \
  --periods-file config/foxglove_periods.json --epochs 20 \
  --encoder-version 3 --embed-dim 64 --encoder-heads 8 \
  --no-identity-embed \
  --output-dir jax_ckpt_easy_phase1_no_ident \
  > jax_ckpt_easy_phase1_no_ident/phase1_nohup.log 2>&1 &
```

Gate: oracle trade-agreement ≥ 50% by E20 (looser than V3's 70% — we expect some performance cost from removing a useful feature).
If ≥ 70%: identity embed was redundant, ablation is free win.
If 50-70%: identity embed was doing something useful; depending on Phase 2 results, may need to re-add.
If < 50%: identity embed was critical; roll back, document, move on to alt #3.

### Phase B — Stage B from no-identity Phase 1
Same as 729b Phase 2 command, swap checkpoint and add `--no-identity-embed`.

Gate at E50:
- `dir_acc > V3-baseline + 0.03` → ablation helped. Strong signal. Continue.
- `dir_acc ≈ V3-baseline` → ablation is neutral; identity embed was not the shortcut.
- `dir_acc < V3-baseline - 0.03` → ablation hurt. Roll back. This is informative — means model was using identity for useful specialization, not a shortcut.

---

## Why This Is Worth ~3 Hours

It's a **one-line change** that tests a specific, falsifiable hypothesis about why direction accuracy caps at 0.57. Most expensive outcome is 3 hours of Foxglove + Phase 2 E50 with a null result.

If the ablation DOES help, it's proof that the identity-embed-as-shortcut pathway is real, and the pivot to TASK_730 (rank oracle) becomes even more motivated — rank oracle has no per-asset constant bias.

---

## Files Modified

| File | Change |
|------|--------|
| `mcts_trader/jax_agent/trm_encoder_v2.py` | `use_identity_embed: bool = True` + `if self.use_identity_embed` gate |
| `mcts_trader/jax_agent/trm_encoder_v3.py` | Same, on `_post_recurrence` |
| `mcts_trader/jax_agent/trm_flax.py` | Pass `use_identity_embed=getattr(config, ...)` to encoder setup |
| `easy_config.py` / `phase2_config.py` | `use_identity_embed: bool = True` |
| `train_easy.py` / `train_phase2.py` | `--no-identity-embed` flag |

---

## Combined with TASK_730

If rank oracle (730) fixes direction but benchmark Bal.Acc is still sub-40%, stack the ablation (731) on top. Fresh Phase 1 retrain required only once — run 730+731 simultaneously, not sequentially.
