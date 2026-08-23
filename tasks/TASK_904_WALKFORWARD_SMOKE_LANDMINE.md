# TASK_904 — walkforward-lane hardening: the smoke overwrite landmine and the WF_SUMMARY fake-null

**Status**: READY 2026-08-23 (planner; seeded from factory run 40's flagged remainders — the
ruling the run asked for, made planner-side without a DECISIONS row because the direction is a
defect fix, not a values call).
**Size: S. Risk: LOW today, HIGH the moment it arms** — see below.

## Why this row exists

Two defects in the 837/875 walkforward lane, both found by run 40's provenance audit:

1. **`walkforward.py smoke` overwrites tracked real-fold definitions.** `run_smoke`
   (`experiments/837_walkforward/walkforward.py:1080-1147`) calls
   `materialize(FIXTURE_DB, REAL_FOLDS, "real", real_root, …)` with no guard, no dry-run, and no
   out-root override — running smoke from the repo root overwrites **8 tracked files under
   `folds/real/`** with fixture-derived content. Today the overwrite is content-idempotent (only
   `generated` stamps move — those files were themselves fixture-materialized on 2026-07-08), so
   this is a **latent landmine, not present damage**. It arms the moment the `SPEC.md:60-72`
   scheduled real-db re-materialization is committed: any later `smoke` then silently reverts
   the real folds to fixture content, prints `SMOKE PASS`, and exits 0 — while job manifests
   still reference `trading_data.db`.
2. **`runs/875_wf/WF_SUMMARY.json` ships as `[]` — a documented silent fake-null.** The
   orchestrator read `sharpe`/`teacher.floor_inenv`; the schema keys are
   `sharpe_full`/`tiers["21"]["teacher_floor_inenv"]` (`NOTE_summary_key_mismatch.md`). The file
   named like the summary is not the source of the numbers — the chain is the raw per-seed JSONs
   + `grade_correct.py`, which is intact and (post-903) verified to reproduce the LEDGER row
   exactly. A reader must not be able to consume an empty summary at exit 0.

## Scope

1. **Smoke must not write tracked real-fold content.** Out-root the real rails to a temp dir and
   diff against expectations instead of overwriting (run 40's option (a)). **And** make
   `materialize` refuse to write a directory whose content it did not generate (option (c)) —
   the fail-closed belt that matters once the real-db re-materialization lands; the smoke fix
   alone leaves every other caller unguarded. If (a)+(c) proves awkward together, the named
   fallback is gating the materialize step behind an explicit `--materialize-real` flag.
2. **Kill the fake-null.** `WF_SUMMARY.json` is either generated correctly from the schema's
   real keys, or removed/absent with readers pointed at `grade_correct.py` — and the smoke/gate
   treats an empty summary as a failure, never a pass.

## Acceptance (gates)

- (i) `walkforward.py smoke` runs green with **zero writes to tracked paths** (content-hash A/B
  on `folds/real/` before/after, or strace).
- (ii) A decoy real-materialization (tracked-looking content the tool did not generate) followed
  by `smoke` leaves the decoy content untouched AND refuses/exits non-zero.
- (iii) An empty `WF_SUMMARY.json` fails loudly; the generated-or-removed resolution is stated
  in the artifact.
- (iv) The 875 grading chain (raw per-seed JSONs + `grade_correct.py`) is untouched and the
  903-verified numbers do not move.

## Out of scope

The `SPEC.md:60-72` real-db re-materialization itself (a separate scheduled row — this row only
ensures the landmine is defused BEFORE that can land). Re-grading 875. The 875 evidence
durability question — handled planner-side (evidence mirrored to the workstation 2026-08-23).
