# TASK_729b: Wider Encoder — Test Whether Per-Asset Feature Capacity Is The Bottleneck

**Status**: READY
**Priority**: CRITICAL — TASK_729a proved per-asset routing alone doesn't move dir_acc (173 epochs, +0.5pp)
**Depends on**: TASK_729a (per-asset routing wired; kept for reuse)
**Scope**: One encoder width change + required retrain path. Binary test at E50.
**Budget**: 2 hrs electricity for the gate check (E50). Full run 8–12 hrs if gate passes.

---

## What 729a Proved, And Why 729b Follows

TASK_729a bypassed the global 64→128 projection. Dir-acc still capped at 0.57. That means the **per-asset features themselves** (32-dim per asset, post-attention, pre-projection) don't carry enough direction information.

Suspicion: the `asset_embedding` Dense at the start of the encoder compresses `(window × channels) = 60 × 27 = 1620` signals down to `embed_dim = 32`. That's ~50:1 compression per asset. Directional signal likely gets squashed here, not at the later global projection.

**Hypothesis**: widening `embed_dim` from 32 to 64 or 128 raises the per-asset feature ceiling. Everything downstream (cross-asset attention, per-asset DirectionHead, PerAssetPolicyHead) benefits from richer per-asset features.

If widening doesn't move dir_acc past 0.63 at E50 → ceiling is not encoder width. It's something fundamental about the encoder family (feed-forward temporal compression) or the input channels themselves, and no further 729* runs will help.

---

## The One Change

File: `mcts_trader/jax_agent/trm_encoder_v2.py` — no structural changes, just a default param bump.

But: **widening breaks shape compatibility with every existing checkpoint**. Every encoder param scales with `embed_dim`:
- `asset_embedding/kernel`: `(window*channels, embed_dim)` → shape changes
- `asset_identity_embed`: `(1, n_assets, embed_dim)` → shape changes
- Cross-asset attention internal projections → shape changes
- `global_state_projection/kernel`: `(n_assets*embed_dim, state_dim)` → shape changes

**Consequence**: new encoder params must be fresh-init. We cannot warmstart from 728c / 729a. This is the cost of testing the hypothesis.

Two mitigation strategies, in order of cheapness:

### Strategy A (fastest, recommended as FIRST test) — Phase-1-Lite Imprint Then Phase 2

1. Fresh-train encoder via 20-epoch Foxglove Phase 1 imprint with wider embed_dim. Enough to reach basic oracle-match accuracy. Do NOT run the full curriculum — we just need a non-random encoder.
2. Roll straight to Phase 2 Stage B with all TASK_727/728/729a wiring.
3. Gate check at Phase-2 E50. If dir_acc > 0.63 → continue. If not → ceiling is NOT encoder width.

Phase 1 lite cost: ~2–4 hrs (vs 24+ hrs for full Foxglove curriculum). Phase 2 gate at E50: another 2 hrs. Total budget to verdict: ~4–6 hrs electricity.

### Strategy B (more expensive, only if Strategy A says "keep widening") — Full Phase 1 Retrain

Only do this if Strategy A's E50 gate PASSES and you want to extract maximum signal. Not a diagnostic run — a productization run. 12–24 hrs of full Foxglove Phase 1, then 200 epochs Phase 2.

---

## Implementation — Strategy A

### Change 1 — Encoder width config

`phase2_config.py` — the `embed_dim` field is inherited from `EasyConfig`. Confirm default value, update for wider run:

```python
# In EasyConfig or Phase2Config override:
embed_dim: int = 64            # was 32 — widen per-asset feature capacity
encoder_num_heads: int = 8     # was 4 — scale heads with embed_dim (64/8 = 8 head dim)
```

Verify that `TinyRecursiveTradingModel` passes `embed_dim` / `encoder_num_heads` through to the V2 encoder. Already does per `trm_flax.py`. No code change required there.

### Change 2 — CLI flags for the widening (optional, for reproducibility)

`train_phase2.py`:
```python
parser.add_argument('--embed-dim', type=int, default=None,
                    help='Per-asset embedding width (V2 encoder)')
parser.add_argument('--encoder-heads', type=int, default=None,
                    help='Cross-asset attention head count')
```

```python
if args.embed_dim is not None:
    config.embed_dim = args.embed_dim
if args.encoder_heads is not None:
    config.encoder_num_heads = args.encoder_heads
```

Do the same in `train_easy.py` / `easy_trainer.py` if those already don't support embed_dim override.

### Change 3 — Phase-1 Lite run

Launch via the existing Foxglove training path (`train_easy.py` + `e-dojo_v1.py`). This project already has imprint orchestrators. Use a minimal config:

```bash
python train_easy.py \
  --periods-file config/foxglove_periods.json \
  --epochs 20 \
  --embed-dim 64 \
  --encoder-heads 8 \
  --output-dir jax_ckpt_easy_phase1_wider \
  > jax_ckpt_easy_phase1_wider/phase1_nohup.log 2>&1 &
```

(Adjust flag names if `train_easy.py` uses different names — check its CLI first.)

What "enough" looks like for Phase-1 Lite: oracle trade-agreement > 60%. Check the logs. We don't need 75% — we just need a trained non-random encoder to seed Phase 2.

### Change 4 — Phase 2 from fresh Phase-1 encoder

Once `jax_ckpt_easy_phase1_wider` has a viable checkpoint:

```bash
mkdir -p jax_ckpt_easy_phase2_729b && nohup python train_phase2.py \
  --checkpoint jax_ckpt_easy_phase1_wider/easy_final.pkl \
  --stage B --epochs 200 \
  --periods-file config/foxglove_periods.json --n-slots 17 --max-runs-per-group 10 \
  --factored --factored-temperature 0.1 \
  --no-reset-value-head --value-target-mode nstep --td-discount 0.95 \
  --value-weight 0.0 --bc-weight-min 0.05 --bc-max-trade-weight 4.0 \
  --direction-weight 0.5 --encoder-lr 3e-5 \
  --embed-dim 64 --encoder-heads 8 \
  --output-dir jax_ckpt_easy_phase2_729b \
  > jax_ckpt_easy_phase2_729b/stage_b_nohup.log 2>&1 &
```

Note: DO NOT pass `--reset-direction-head` — new Phase 1 checkpoint won't have direction_head at all; `initialize_missing_aux_heads` handles that case already.

---

## Diagnostic Gates

### Phase 1 Lite gate
- Oracle trade-agreement > 60% by E20. If not → encoder can't imprint even basic patterns with this width; something wrong with config, not width itself. Debug before pushing to Phase 2.

### Phase 2 E20
- `dir_acc` should exceed **0.58**. This is ~2pp better than 729a's 0.565 at E21. If similar to 729a → widening hasn't changed early dynamics.

### Phase 2 E50 (THE GATE)
- `dir_acc` > **0.63** → widening works. Continue to E200, benchmark.
- `dir_acc` 0.60–0.62 → marginal. Widen further (`embed_dim 128`)? Not worth the electricity budget. Stop here and benchmark current checkpoint.
- `dir_acc` < **0.60** → **KILL.** Ceiling is not encoder width. It's the encoder family or the input channels. 729* is exhausted.

### Phase 2 E100 / benchmark
- Bal.Acc > 42% AND dir_acc > 0.65 → real breakthrough. Deploy this.
- Bal.Acc ≤ 38% even with dir_acc > 0.63 → the policy head can't make use of stronger direction; something else is broken. Fallback: raise `direction_weight` to 1.0 and re-run E50 gate.

---

## Failure Modes

| Symptom | Meaning | Action |
|---|---|---|
| Phase 1 Lite loss exploding | embed_dim mismatched somewhere in pipeline | Verify `embed_dim` reaches V2 encoder; check `asset_embedding` Dense output shape |
| Phase 2 OOM | Wider encoder uses more VRAM | Halve `stage_b_batch_size` via `--batch-size` |
| Dynamics consistency divergent | State_dim unchanged but latent space is different now | Let it run 20 epochs; dynamics re-learns. If still bad at E50, increase consistency weight |
| dir_acc stuck at 0.57 | Per-asset feature capacity was NOT the bottleneck | Stop. Consider TASK_729c (V3 temporal encoder) or accept baseline |

---

## Cost Discipline

Phase 1 Lite run MUST be kept to ≤4 hrs. If Foxglove orchestrator is going longer, cap it manually. We're testing a hypothesis, not training for production.

Phase 2 E50 gate is a hard kill line. Do not let the run go past E50 without verifying dir_acc > 0.60. No "let me see where it plateaus" excuses — we already proved that plateau pattern.

If 729b fails at the E50 gate, the result is definitive: encoder architecture needs to change (TASK_729c), or the project has hit its baseline and should ship 728c. Either way, stop iterating on width.

---

## Files Modified

| File | Change |
|------|--------|
| `phase2_config.py` (and `easy_config.py` if embed_dim lives there) | Confirm/update `embed_dim=64`, `encoder_num_heads=8` |
| `train_phase2.py` | Add `--embed-dim` and `--encoder-heads` CLI overrides |
| `train_easy.py` | Same CLI flags (if not already present) |

No new modules. No architectural changes. Just a width bump + its consequences (retrain).
