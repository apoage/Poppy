# TASK_876 — Score-based checkpoint retention (operator requirement; PREREQUISITE for the long base-training program)

**Status**: READY 2026-08-05 (operator: "checkpointing needs to be pruned based on checkpoint
score .. that is necessary"). **Size: S. Risk: low-medium** (touches the training save path; must
never delete a resumable checkpoint). Surface: `mcts_trader/training/safe_checkpoint.py`,
`mcts_trader/jax_agent/trainer.py::save_checkpoint`, `train_jax_fast.py` (save/snapshot sites).

## Why (measured 2026-08-05)

Long multi-day runs are about to start on `gpu-server`, which has **136 GB free**. Today nothing
prunes by score:

- **Snapshots are unbounded BY DESIGN**: `train_jax_fast.py:~1782` creates
  `snapshot_ep{epoch}` dirs — the code comment says "periodic named snapshots that don't get
  overwritten". Each is **~89 MB** (measured: `jax_ckpt_fast/e5exit2_pq_s*`). A 2000-epoch run
  with `snapshot_every=25` ⇒ 80 × 89 MB ≈ **7 GB per seed**, ~21 GB for a 3-seed run — and
  nothing ever removes the bad ones.
- **The main save prunes by RECENCY, not score**: `trainer.save_checkpoint` calls flax
  `checkpoints.save_checkpoint(..., keep=3, overwrite=True)` — keep-3-most-recent. A late-run
  collapse (the program has seen several) therefore evicts the good checkpoints and keeps the
  broken ones.
- `SafeCheckpointManager` already tracks `keep_best_n` + `best_metric` and has 2 deletion sites,
  but it is **opt-in** (`config.safe_checkpoints` defaults False) and sits on a separate path
  from both the default save and the snapshots. Verify whether its losers are actually reclaimed
  from disk or only dropped from its in-memory list — report either way.

## Scope

1. **One retention policy, applied to every save path** (default save, snapshots, safe-manager):
   keep the **top-K by score** plus **always the latest resumable checkpoint** (and the first, if
   cheap, for reference) — K configurable, sane default 3. Recency-only pruning is removed as the
   effective policy.
2. **Score provenance — never prune on a metric the run didn't measure.** The score is the run's
   designated metric, recorded per checkpoint: a periodic in-env eval score where the run
   produces one, else the rolling train reward (what `SafeCheckpointManager` uses today). Record
   WHICH metric, its value, and the epoch in the checkpoint's metadata. **If no score is
   available for a checkpoint, it is NOT prunable** (fail-safe = keep, never delete on missing
   data — the 866 no-fail-open rule).
3. **Auditable index**: `checkpoints_index.json` in the run dir — one row per checkpoint (path,
   epoch, metric name, score, kept/pruned, reason, bytes). Every deletion is logged with its
   score and why it lost. Pruning decisions must be reconstructable after the fact.
4. **Never delete these**: the latest checkpoint (resume), the current best, anything referenced
   by a `runs/*/…` artifact under `experiments/` (a checkpoint some ledger row points at is not
   garbage). Deletion is atomic (move-then-remove, never a partial dir left behind).
5. **Disk floor**: a configurable minimum-free-space guard — below the floor, refuse to write a
   new snapshot (loud, with the number) rather than filling the disk; log projected volume at run
   start (`epochs / snapshot_every × size`) so a run that cannot fit says so on launch.
6. **Offline GC mode**: `--gc-checkpoints <dir>` applies the same policy to an existing directory
   (dry-run by default, `--apply` to act) — usable on the current leftovers
   (`jax_ckpt_fast/821_planner_daily` 267 MB, three `e5exit2_pq_s*` at 89 MB each).

## Acceptance (gates)

- (i) Simulated run with scripted scores: top-K by score survive, losers are gone FROM DISK
  (bytes reclaimed, shown), latest + best always present, index matches reality.
- (ii) Collapse scenario: scores rise then fall off a cliff → the good early checkpoints survive
  and the late bad ones are pruned (this is the case flax `keep=3` gets backwards — demonstrate
  the old behavior does the wrong thing and the new one does not).
- (iii) Missing-score checkpoint is never deleted; referenced-by-artifact checkpoint is never
  deleted (both demonstrated).
- (iv) Resume works after a prune (load latest and load best, both succeed).
- (v) Disk-floor refusal fires with the free-space number named; dry-run GC lists without
  deleting, `--apply` reclaims and logs.
- (vi) Selftest green; no numerics change on any committed baseline (this is storage policy only).

## Out of scope

Choosing the training recipe or the score metric's definition for the long run (planner, in the
run's pre-registration); the WF battery currently running (15 ckpts ≈ 1.3 GB, no pruning needed);
remote/off-host archival of checkpoints.
