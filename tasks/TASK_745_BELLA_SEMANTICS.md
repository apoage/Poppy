# TASK_745 — Bella Distillation Asset-Order Semantics

**Status**: INVESTIGATED 2026-05-14. Decision pending on fix option A/B/C.
**Blocks**: TASK_740 (Bella forward-rollout distillation training).
**Owner**: Workstation operator decides design; implementation TBD.

## Finding (verdict C — mixed semantics)

Bella's output columns are **DB-ID-bound** but the trainer expects **rotation-specific asset order**. No remapping happens. Per-asset distillation MSE is column-mismatched → noise gradient.

See `~/.claude/.../memory/project_bella_column_mismatch.md` for full
file:line citations.

## What the agent found (summary)

1. **Bella training asset universe is FIXED** (`diff_teacher/data_loader.py:142-154`): asset order = `list(dict.keys())` from SQL = database ID order. Not canonical, not symbol-sorted.
2. **Bella output column k = DB ID k**, semantically (the k-th asset Bella saw during its own training).
3. **Trainer rotation per period** (`foxglove_periods.json`): 270 entries = 18 date ranges × ~15 asset rotations. Each has its own `assets` list in its own order.
4. **HistoricalMarket silently drops order info** (`mcts_trader/data/history.py:214-222`): `found = {r[0]: r[1] for r in rows}` populates a dict from SQL row order, not the input symbol list order.
5. **Precompute does cyclic pad/truncate** (`tools/precompute_bella_predictions.py:134-138`) — no per-asset remap to align with the period's asset list.
6. **Trainer loss is direct per-column MSE** (`train_phase2.py:1105`) — no permutation safety net.
7. **Manifest tag is date-range-only** on both sides, so 270 rotations collapse to 18 unique shards.

## What's broken

- Per-asset Bella distillation: `model[i]` is compared against `bella_shard[i]` where `i` indexes *different assets* on each side.
- Existing 18 shards on server (`~/Trader/data/bella_preds/`, 60 MB): semantically wrong; recommend `rm` before any fix re-run.

## Fix options (operator decides)

### Option A — Per-rotation column remap

Modify `_process_period` in `tools/precompute_bella_predictions.py`:

```python
# After Bella generates preds_bella of shape (T, 64, n_horizons):
period_assets = period.get('assets', [])   # rotation-specific order
bella_assets = bella_canonical_asset_list  # need to extract from Bella ckpt

# Build remap: rotation index i → Bella index, or NaN if asset not in Bella's universe
remap = jnp.array([bella_assets.index(a) if a in bella_assets else -1
                   for a in period_assets])

# Reindex Bella output to rotation order
preds_remapped = np.where(
    remap[None, :, None] >= 0,
    preds_bella[:, remap, :],   # column gather
    np.nan                       # asset not in Bella → NaN, masked by trainer
)
```

Also change the manifest tag to include the rotation hash:
```python
period_tag = f"{start}__{end}__{hashlib.sha256(','.join(sorted(period_assets)).encode()).hexdigest()[:8]}"
```

Trainer side: matching tag construction in `train_easy.py:774`.

**Cost**: ~50 lines patch + re-run precompute for 270 rotations. At stride=64 K=1 with ~5 min per unique rotation → estimated ~17h wall (one full day).

**Pros**: per-asset signal becomes valid; trainer's NaN mask handles assets Bella never saw.

**Cons**: largest implementation; precompute time is non-trivial.

### Option B — Force canonical Bella universe

Drop rotation entirely. Trainer uses Bella's fixed 64-asset universe for all periods. Simple, but throws out the asset-rotation regime design that everything else depends on.

**Not recommended** unless the rotation design is independently abandoned.

### Option C — Aggregate-return regime prior

Replace per-asset distillation MSE with **mean-across-assets** MSE:

```python
# In train_phase2.py near line 1105:
model_mean = jnp.mean(model_per_asset_log_return, axis=-1)
bella_mean = jnp.mean(bella_preds_full[w], axis=-1)
loss = jnp.mean((model_mean - bella_mean) ** 2)
```

Bella becomes a regime-shape prior, not a per-asset teacher. Existing shards (without proper column mapping) are still usable since the mean across columns is order-invariant.

**Cost**: ~10 lines trainer patch + delete the existing 18 shards (precompute is fine for this).

**Pros**: matches what Bella actually IS (unconditional regime sampler); cheapest fix; existing shards still usable after re-run with the same script (mean is permutation-invariant).

**Cons**: weaker signal — carries regime-mean info but not per-asset specificity. May not break the 60-step input ceiling that TASK_740 was targeting in the first place.

## Recommendation (workstation operator's call)

- If Bella retrain is on the medium-term roadmap (Tier-D), defer TASK_740 entirely and revisit after retrain. Retrain could use canonical asset symbols (symbol → integer hash) avoiding the DB-ID drift.
- If TASK_740 needs to ship before retrain: **Option C is the cheapest acceptable fix**; gets a real signal test with low risk, can escalate to Option A if the regime prior helps.
- Option A is the "right answer" if we want Bella as a serious per-asset teacher.

## DO NOT

- Run TASK_740 training with current shards. Loss gradient is noise; could actively hurt the model.
- Trust the existing 18 shards. Recommend `rm ~/Trader/data/bella_preds/*.npz && rm ~/Trader/config/bella_preds_manifest.json` after the operator picks a fix.

## Related

- `tasks/TASK_744_BELLA_PRECOMPUTE_FIX.md` (already completed: path fix + stride flag)
- `tasks/TASK_740_BELLA_DISTILL_HEAD.md` (the original training task; needs amendment)
- `experiments/740_bella_distill/SPEC.md` (current SPEC — references the broken design)
- `memory/project_bella_column_mismatch.md` (this finding)
