# TASK_864 — Tracked parity baseline must not be rewritable by a default invocation (FILLER)

> **STATUS (2026-07-29): SHIPPED as `task864-parity-baseline-hygiene`.** All four acceptance gates
> PASS. (i) default invocation leaves `git status` clean — writes a gitignored
> `results_fixture_smoke_local.json`; (ii) the explicit `--write-parity-baseline` path works and its
> refusals fire (dirty file 0.9s fail-fast before training, `--out` conflict, suffixed experiment,
> non-fixture db, **plus recipe and validity gates added in the fix round**); (iii) baseline
> **REFRESHED** with the per-field attribution table in a dedicated commit — zero unattributable
> fields; (iv) the sole consumer verified, and the refreshed file reproduces **bit-identically** from
> its documented recipe (`md5 16736523175d25659405d568608a2a07`).
>
> **THIS TASK FILE'S PREMISE WAS A MISDIAGNOSIS — corrected below.** The "~74 commits stale / 479
> differing leaves" figure compared the baseline against a **flagless default** run, but the
> committed artifact is the **weight-ON** deliverable (`--vector-weight-rebalance`), already
> documented at `SPEC.md:205-211`. **~461 of ~468 leaves were the flag, not drift.** Real drift:
> ~7 fields + one conditional block, all from `c2f1ba5`/TASK_846, matching the ratified **D16**
> ruling verbatim (`teacher_weight_net21 −4.687 → −3.111`).
>
> **Fail-closed judge found 1 BLOCKER + 1 MAJOR + 1 MINOR + 1 NIT, all fixed in one round:**
> the promote flag never checked WHICH experiment produced the artifact (the flagless weight-OFF
> default promoted onto the weight-ON baseline — 462/545 leaves, exit 0 — with **no downstream
> detector**, contradicting the module's own `imprint.py:104` invariant); promotion did not require
> the run to have passed its own validity gate (`--iters 1 --no-smoke-assert` installed an untrained
> artifact as the oracle); the copy was non-atomic and read a shared path. Now recipe-pinned
> (expected values re-derived from the module's own argparse defaults), validity-gated independent of
> `--smoke-assert`, and atomic (`.tmp` + `os.replace`).
>
> **FOLLOW-UP, and it explains the whole incident: `CANON_832_PINS` is a weight-OFF oracle guarding
> a weight-ON baseline.** The 861 smoke pins the flagless default; the committed artifact is the
> weight-ON one, so NOTHING in the repo pins the weight-ON numbers. The file was not unguarded by
> accident — it was guarded by an oracle for a different configuration, which is exactly how 846
> moved it silently while every gate stayed green. Add a weight-ON pin, or make the pins read the
> tracked baseline.
>
> Other remainders (module-wide self-clobbering incl. `--quick` downgrading 861's baseline to seed 0;
> `861/smoke_e2e.py:428/459` passing against arbitrary content; `passed` false→true being a
> substantive verdict change whose binding read is the scheduled real-panel run) are in CHANGELOG
> 2026-07-29 and `.dark-factory/verdicts/`.

**Status**: SHIPPED 2026-07-29 (was READY 2026-07-28, seeded from the factory run-20 wrap's standing-risk flag).
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
