# TASK_876 — Emit checkpoint SCORES so the shipped retention mechanism can act on them

> **RESCOPED 2026-08-06 by the D32 ruling — READY again as the NON-DESTRUCTIVE half only.**
> The row splits: deletion POLICY moves to [TASK_881](TASK_881_CHECKPOINT_PROTECTION_MODEL.md)
> (the protection model), the remote archive tool's discovery defect moves to
> [TASK_880](TASK_880_CKPT_ARCHIVE_DISCOVERY_FIX.md). What stays here, resuming FROM THE STASH:
>
> 1. **`scores.jsonl` emission at every save** — the original scope 0, and the thing the program
>    actually needs (a 42-checkpoint sweep found 0 scored; under the operator's ruling an
>    unscored checkpoint is garbage, so an unscored long run produces garbage by construction).
> 2. **The three mechanical fixes that need no ruling**: `_is_protected` honouring DESCENDANT
>    entries; a TTL re-scan instead of a launch snapshot; a failed `rmtree` recorded as a FAILED
>    prune in the audit index instead of a successful one.
> 3. **Revert `flax_keep = 1_000_000` to the pre-876 bound and set `overwrite=False`.** This is
>    the finding that mattered most in the run-26 report — `flax_keep = 1_000_000` plus the
>    pre-existing `overwrite=True` is a protection-blind mass delete on any resume at a lower
>    step, and pre-876 that cost at most 2 dirs. The task had enlarged its own blast radius.
> 4. **`--ckpt-retention` defaults to `recency` — the destructive path stays OFF** (invariant 1).
>    Scope 1's demand that recency stop being the effective policy is met by the OFFLINE tool
>    being the effective policy (`tools/ckpt_gc.py`, planner-shipped, operator-run), not by
>    flipping an in-run default ahead of its protection model. The default flips in TASK_881,
>    gate (vi), with the collapse case re-proven rather than assumed.
>
> Nothing in this half deletes anything new, so none of it depends on D32's unsettled question.
> The factory's disclosure-instead-of-silent-choice on the default was correct, and the "S"
> sizing was the planner's error, not the spec's content — emission and deletion policy should
> have been separate rows at authoring time.
>
> **Prior status (2026-08-06): STOPPED, NOT SHIPPED — built and gate-green, halted on the 3-cycle rule
> with a new human wall, D32 (the protection model).** Code preserved in `git stash`
> ("TASK_876 … STOPPED on halt rule (protection model, D32)"), re-apply verified clean on HEAD;
> nothing committed, nothing tagged. 1,208 insertions across 6 files — `tools/ckpt_gc.py`,
> `mcts_trader/training/{safe_checkpoint,config,cli}.py`, `mcts_trader/jax_agent/trainer.py`,
> `train_jax_fast.py`. **The row was sized S; it is not — that mis-sizing is the finding to carry
> forward, not a criticism of the spec's content.**
>
> **What is PROVEN (independent flow-verifier, real flax saves, not unit stubs).** All six
> acceptance gates (i)–(vi) PASS. The collapse case — the task's entire reason for existing —
> demonstrated side-by-side on one curve rising to +9 then crashing to −9.5: the OLD flax `keep=3`
> ends holding `checkpoint_7000/8000/9000`, the three MOST collapsed; the new policy holds
> `checkpoint_3000/4000/5000` (the good pre-collapse peak) plus the newest as the resume point.
> Resume-after-prune works through the real load path for both latest and best. Disk stays bounded
> under the floor with unscored checkpoints (peak 21.5 MB vs an unbounded 43 MB and climbing), the
> resume point present at every step. Three real layouts all work end-to-end. Audit history is
> reconstructable across interleaved in-run/offline prune waves. Idempotent. selftest 12/12;
> fixture `b0f8de8b…` and `trading_data.db` byte-untouched throughout.
>
> **Answering the spec's own open question:** `SafeCheckpointManager` DOES reclaim its losers from
> disk (`shutil.rmtree`, `safe_checkpoint.py`), not merely drop them from its in-memory list —
> verified by direct exercise. One deletion site, not the two the task text estimated.
>
> **Three BLOCKERs found and closed across the cycles** (each reproduced, each re-verified):
> the epoch-key fallback lent one checkpoint's score to a different UNSCORED one and deleted it
> (flax names dirs by `step_count`, not epoch — `scores.get(500)` against `{10,20,30}`); a
> fabricated `0.0` score on an epoch-0 crash out-ranked real negative scores and evicted a real
> checkpoint; and the disk floor was measured once on `roots[0]` although `jax_ckpt_fast`
> (`/dev/sdd1`) and `trading_data.db` (`/dev/nvme0n1p2`) are on DIFFERENT filesystems here — so it
> deleted from a 664 GB-free disk because another was low, and could never fire for the DB whose
> protection is its stated purpose.
>
> **Why it stopped — the protection model, now D32.** Scope 4 says anything referenced by an
> `experiments/` artifact is never garbage, and that has no mechanical definition in this repo:
> `experiments/828_stage3_train/SPEC.md:211` cites runs **by name** ("Battery e5exit2_pq_s{0,1,2}"),
> never by path, and `experiments/LEDGER.md` holds zero `jax_ckpt_fast/` references — so 4 of 7 run
> dirs get zero protection. A reference naming a path INSIDE a checkpoint protects the inner path
> but not the checkpoint (ancestors-only walk), and the delete then destroys the protected path. The
> protect set is a launch snapshot while `SERVER.md` documents mid-run SPEC drops as normal, so the
> in-run deleter and the offline GC disagree about what is protected. **And `flax_keep = 1_000_000`
> with the pre-existing `overwrite=True` is a protection-blind mass delete: on a resume at a lower
> step flax removes every checkpoint above it, consulting no never-delete set (verified on flax
> 0.10.7). Pre-876 that cost at most 2 dirs under `keep=3` — this task enlarged its own blast
> radius, and it is the finding that argues against starting the long run on the current code.**
> Smaller and open: the safe-manager is a 4th deletion site outside the policy (default OFF, already
> bounded by its own `keep_best_n`), and a failed `rmtree` leaks bytes forever under
> `*.gc-trash-<pid>` while the audit index records a normal successful prune.
>
> **Operator call carried in D32 alongside the model choice:** `ckpt_retention` currently defaults
> to `'score'` — new destructive behavior default-ON, contradicting invariant 1 ("flag-gated,
> default OFF") while Scope 1 explicitly demands recency-only be removed as the effective policy.
> Three judges flagged the conflict independently; the factory left it default-ON and disclosed it
> rather than silently choosing. `--ckpt-retention recency --ckpt-min-free-gb 0` is the full
> pre-876 restore (recency alone does NOT restore it — the floor refusal still applies).
>
> **Out of scope, but load-bearing and urgent (planner row recommended before the srv program):**
> `tools/ckpt_archive.py` carries the SAME discovery defect on the tool that **deletes on the
> server** — its embedded `REMOTE_PROBE` (lines 41–45) misidentifies run dirs exactly as
> `ckpt_gc.py` did, omits `checkpoint_\d+` / `checkpoint_epoch_\d+` / `checkpoint_golden`, and reads
> scores epoch-keyed only, so an entire scored run can be judged uninteresting and removed from
> `gpu-server`. Excluded here because the task file puts remote/off-host archival out of scope.

> **SCOPE CORRECTED 2026-08-05 (operator: "this is off factory — there must be a mechanism that
> will prune checkpoints or all are gone and we are syncing whole database").** The MECHANISM is
> not a factory row and is no longer pending: the planner shipped it directly, because runs were
> already writing checkpoints and a full disk on `gpu-server` puts `trading_data.db` — 26 GB,
> unshippable on a ~5 Mbps uplink — at risk.
> **Shipped by the planner (this repo, tested):**
> * `tools/ckpt_gc.py` — score-based GC + disk-floor guard. Keeps top-K by score + always the
>   newest (resume) + the best; unscored are NEVER score-pruned (only recency-pruned under the
>   disk floor, and the log says so); refuses paths outside its roots or containing
>   `trading_data`/`.git`/`fixtures`/`backups`; deletes move-then-rmtree so a half-deleted
>   checkpoint can never look loadable; dry-run by default. Six acceptance gates pass, including
>   **the collapse case** (scores rise then crash: flax `keep=3` keeps the broken checkpoints,
>   this keeps the good ones) and "newest survives even when it scores worst".
> * `tools/ckpt_archive.py` — the lifecycle the operator asked for: score the server's
>   checkpoints, DOWNLOAD the interesting ones to the workstation (~440 GB free), verify the local
>   copy, and only then clear the server (136 GB free, disposable compute). Never uploads.
>   Manifest written before any deletion.
> **What remains for the factory is the missing INPUT, not the policy:** nothing in this repo
> writes a per-checkpoint score, so a 30 GB / 42-checkpoint sweep of the server on 2026-08-05
> found **0 scored** — the GC's fail-safe then (correctly) protects everything, which makes it
> unable to choose. Close that gap:

**Status**: READY 2026-08-05 (operator requirement; scope corrected — mechanism shipped, this row
is now the score-emission half). **Size: S. Risk: low-medium** (touches the training save path;
must never delete a resumable checkpoint). Surface: `mcts_trader/training/safe_checkpoint.py`,
`mcts_trader/jax_agent/trainer.py::save_checkpoint`, `train_jax_fast.py` (save/snapshot sites).

## Scope 0 (NEW, and the reason this row still exists)

Every checkpoint-writing run must emit a score the shipped tools can read: append
`{"epoch": <int>, "score": <float>, "metric": "<name>"}` to `<run_dir>/scores.jsonl` at each
save, and/or write `checkpoints_index.json` in the same shape. The metric is the run's designated
one (periodic in-env eval where it exists, else the rolling train reward `SafeCheckpointManager`
already tracks) and its NAME is recorded — never prune against a metric the run didn't measure.
Retrofit is not required for the 42 legacy checkpoints (already archived to the workstation), but
`--gc-checkpoints`-style cleanup of new runs must work end-to-end with real emitted scores.

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
