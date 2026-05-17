# TASK_745 spot-check — Bella column-remap verification

## Top-line verdict: PASS

The Option A per-rotation column remap shipped in `tools/precompute_bella_predictions.py`
is exact end-to-end. Bella v2 follow-up tasks are unblocked from the column-mismatch
gate.

## Method

- Rotation: period 0 = `2018-10-17 → 2019-04-15`, rotation_hash `a7185523`
- Shard: `data/bella_preds/bella_preds_2018-10-17__2019-04-15__a7185523.npz`
  (rsync'd from server; manifest also pulled to `/tmp/745_spotcheck/`)
- Timestep: window w=0 (first stride window)
- Cell traced: trainer column 17 = `ETC/EUR`
- Cell traced (NaN path): trainer column 10 = `DASH/EUR` (not in Bella's universe)

## Evidence

1. **Manifest tag wiring** (`tools/precompute_bella_predictions.py:60-67, 182-188` vs
   `train_easy.py:782-786`): both ends construct the rotation hash from
   `sha256(",".join(sorted(period_assets)))[:8]`. Hash for period 0 = `a7185523` matches
   the shard filename and the manifest key.
2. **Bella canonical asset list, independent re-derivation from `trading_data.db`** (same
   SQL as `diff_teacher/data_loader.py:131-157`): list of 64 symbols matches the
   shard-embedded `bella_canonical_assets` exactly.
3. **Trainer asset order, independent re-derivation via `HistoricalMarket(specific_symbols=p['assets'], ...)`**: 61 symbols match `trainer_asset_order` embedded in the shard exactly.
4. **Remap, independently constructed** from the two lists above using the loop at
   `tools/precompute_bella_predictions.py:238-243`: matches the shard's `remap` array
   bit-for-bit (64/64 entries). 33 valid, 31 NaN-masked.
5. **Bella forward pass independently re-run** (1 window, K=4, regime=1, seed=1777085765
   = `salt(tag) + 0`, 5.3 s wall on workstation): manually-remapped output matches
   shard `preds[0]` byte-for-byte. `max abs diff = 0.0` over all 132 valid cells; NaN
   mask matches exactly. (Bella's RNG is `jax.random.fold_in`-based and the seed salt is
   tag-derived, so byte-equality is the expected outcome — and we got it.)
6. **Cell-trace for trainer col 17**: `shard preds[0, h=10, 17] = 1.811140e-3`,
   reproduced `bella_out[h=10, 51] = 1.811140e-3`, `remap[17] = 51`,
   `bella_canonical[51] = ETC/EUR == trainer_asset_order[17]`. Symbols agree.
7. **Cell-trace for NaN path (col 10)**: trainer = `DASH/EUR`, `remap[10] = -1`, shard
   value is `nan`, my value is `nan`. Trainer's `jnp.isfinite(tgt_bella)` mask
   (`train_phase2.py:1183`) drops it cleanly.

## Per-rotation column-remap function trace

- **Hash construction**: `_rotation_hash` in `tools/precompute_bella_predictions.py:60-67`
  (precompute) ↔ same logic inline at `train_easy.py:782-785` (trainer). MATCHES.
- **Canonical-asset derivation**: `_load_bella_canonical_assets` at
  `tools/precompute_bella_predictions.py:70-102`. Run once per precompute invocation
  (line 375). Replicates `diff_teacher/data_loader.py::_load_assets` SQL exactly.
- **Remap construction**: `tools/precompute_bella_predictions.py:228-243`. Called once
  per period inside `_process_period` (line 191).
- **Remap application**: `tools/precompute_bella_predictions.py:277-288`. Per-window
  fancy-gather + NaN mask, applied before write at line 295.
- **Persistence**: `remap`, `trainer_asset_order`, `bella_canonical_assets`,
  `rotation_hash` all saved into the NPZ at `tools/precompute_bella_predictions.py:295-307`.
- **Consumer**: `train_easy.py:_load_bella_shard` (lines 752-818). Tag lookup at line
  786-789. No further remap on load — correct, because precompute did it.
- **Loss site**: `train_phase2.py:1180-1196`. Per-asset MSE; sample-wise NaN-finite
  mask handles unknown assets cleanly.

## Implication for Bella v2 tasks (753/754/755)

**Bella v2 tasks are unblocked, proceed.** The TASK_745 caveat in `runs/740_bella_distill/result.md`
is removed: the −1.92pp Alpha regression seen in 740 is NOT attributable to scrambled
gradient. It is attributable to "per-asset Bella distillation, correctly wired,
hurts Alpha by ~2pp." 753/754/755 may proceed without re-precomputing.

Note: 51% of trainer cells are NaN-masked (33 valid / 64 columns under cyclic pad → only
33 unique assets carry signal in this rotation; aggregate is 31/64 unknown per
manifest entry, varying ~31–36 across rotations). This is structural — many trainer
rotations include EUR/CAD/GBP/CHF crosses Bella never saw. The trainer's NaN mask
handles this correctly per the loss site. If 753/754 want denser signal, expanding
Bella's training universe is the lever, not the remap.

## Artifacts

- Reproduction script + logs: `/tmp/745_spotcheck/bella_repro2.log`
- Shard used: `/tmp/745_spotcheck/bella_preds_2018-10-17__2019-04-15__a7185523.npz`
- Manifest copy: `/tmp/745_spotcheck/bella_preds_manifest.json`
