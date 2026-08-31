# bella column mismatch

> Bella's output columns are bound to DB-ID order; trainer's columns are rotation-specific (from foxglove_periods.json per-period assets list). No remapping anywhere → per-asset distillation MSE is column-mismatched, gradients are noise. TASK_740 hypothesis cannot be tested cleanly until fixed.

*Migrated from local agent memory 2026-08-31 (`project_bella_column_mismatch.md`) so subagents and the server agent
can read it — see `lore/adr/0008`. Content verbatim; only frontmatter and wiki-links changed.*
TASK_740 (Bella forward-rollout distillation) is **architecturally broken as designed** until per-asset column semantics are aligned. Discovered 2026-05-14 via TASK_745 investigation.

## Root cause (file:line)

- Bella's training data: `diff_teacher/data_loader.py:142-154` — `asset_ids = list(self.assets.keys())` where `self.assets` is dict from SQL. **Asset order = database ID order**, NOT canonical symbol/rotation order.
- Bella's output: shape `(K, T, 64, 5)`, column k = "Bella's training asset k" = DB ID k.
- Trainer's rotation: `foxglove_periods.json` has 270 entries (18 unique date ranges × ~15 rotations). Each entry has its own `assets` list. Trainer column i = rotation's asset i.
- HistoricalMarket loads symbols from SQL with `found = {r[0]: r[1] for r in rows}` (`mcts_trader/data/history.py:214-222`) → dict order = SQL row order = DB ID order, **not** the input symbol list order. Asset-list-order info silently dropped.
- Precompute's `_window_forward_log_returns` pads/truncates cyclically (`tools/precompute_bella_predictions.py:134-138`) without per-asset remap.
- Trainer loss: `train_phase2.py:1105` does direct MSE per-column, no permutation.
- Manifest key: `<start>__<end>` only on both sides (`tools/precompute_bella_predictions.py:152` and `train_easy.py:774`) → 270 rotations collapse to 18 unique shards even before column issue.

## Impact

- The 18 existing shards on server (`~/Trader/data/bella_preds/`, 60 MB total, generated 2026-05-14 07:59-09:12) are **semantically unreliable** — recommend `rm` before any new precompute work.
- Per-asset distillation MSE on these shards = noise gradient, may actively hurt training.
- TASK_740 SPEC in `experiments/740_bella_distill/SPEC.md` as written **cannot yield meaningful signal**.

## Fix options

A. **Precompute applies per-rotation column remap** — modify `_process_period` to map Bella's DB-ID-ordered output → rotation's asset order. Bella columns for assets not in rotation → NaN. Re-run precompute for all 270 unique rotations. ~17h compute at stride=64 K=1.

B. **Force trainer to canonical Bella universe** — drop asset rotation; trainer uses Bella's fixed 64. Simplest but kills the asset-rotation regime design.

C. **Abandon per-asset distillation** — use Bella aggregate return (mean across columns) as regime-shape prior. Cleaner semantically given Bella is an unconditional generator. Weaker signal but defensible.

## How to apply

- **Don't run TASK_740 with current shards or current loss formulation.** Set `bella_distill_loss_weight=0` or skip the head entirely.
- Before re-attempting, pick a fix option. Until then TASK_740 stays BLOCKED in `experiments/QUEUE.md`.
- The other Tier-2 experiments (741 Mint running, 739 dense_sign ready, 734 flow_matching ready, 738 legacy_27ch pending ckpt) are unaffected — no Bella dependency.

See `lore/notes_2026-05-13.md` §104 + `tasks/TASK_745_BELLA_SEMANTICS.md` for the full investigation.
