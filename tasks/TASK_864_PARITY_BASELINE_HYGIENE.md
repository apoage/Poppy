# TASK_864 — Tracked parity baseline must not be rewritable by a default invocation (FILLER)

**Status**: READY 2026-07-28 (seeded from the factory run-20 wrap's standing-risk flag).
**Size: XS/S. Risk: low.** SAME-FILE constraint: touches `experiments/832_linear_imprint/` —
run AFTER TASK_863 ships, never in parallel with it.

## Why (factory run 20 finding, reduced-not-closed)

`experiments/832_linear_imprint/results_fixture_smoke.json` is a TRACKED parity baseline that the
module's own default invocation rewrites in place, and it is stale by ~74 commits (drift
traceable to `c2f1ba5`/TASK_846). During run 20 an empty-string flag bug actually clobbered it
(caught, restored). The live risk: someone runs the module, sees a dirty tracked file, commits
the "refresh" — and a real behavior change gets laundered into the baseline without anyone
deciding the values were allowed to move.

## Scope

1. **Never write tracked paths by default**: the default invocation writes its outputs to an
   untracked location (`runs/` or the module's output dir convention). Overwriting the tracked
   baseline requires an explicit `--write-parity-baseline`, which first prints a per-field diff
   summary vs the existing file and refuses if the file is git-dirty.
2. **Reconcile the standing drift, deliberately**: compare current-HEAD fixture output vs the
   stale tracked baseline; attribute each moved field to its causing commit (starting point:
   `c2f1ba5`/846 fee-headroom). If every delta is attributable to shipped intentional changes →
   refresh the baseline in a dedicated commit whose message carries the attribution table. If ANY
   delta is unattributable → STOP, do not refresh, seed a DECISIONS row with the unexplained
   fields.

## Acceptance (gates)

- (i) Default invocation leaves `git status` clean (demonstrated before/after).
- (ii) Explicit-flag path works and its refusals (dirty file) fire.
- (iii) Baseline either refreshed WITH the per-field attribution table in the commit message, or
  a STOP verdict with the unexplained fields named — no silent refresh, no partial pass.
- (iv) Whatever consumes the baseline for parity checks still passes against the refreshed file
  (or the STOP verdict documents that it was checked against the stale one).

## Out of scope

The frame-pin work (TASK_863 owns that surface first); changing what the parity check compares;
any other module's baselines (if the same pattern exists elsewhere, report — one-line note in the
wrap — do not fix here).
