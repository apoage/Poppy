> **STATUS (2026-08-24, factory run 41): HALTED on the 3-cycle rule. NOT SHIPPED. See D61.**
> Work preserved in `lore/handoff_2026-08-24_factory_run_41_904_cycle3.patch` (1838 lines,
> including a 32-test suite); `experiments/837_walkforward/` reverted to HEAD.
>
> **Both ORIGINAL defects were solved. The row died on the BELT it added, not on the landmine.**
> Cycle 1's sentinel-string gate was proven broken in both directions (forgeable AND it locked out
> the `SPEC.md:62-64` re-materialization). Cycle 2's redesign — the operator's
> `--overwrite-existing` flag as the gate, provenance demoted to disclosure, fixture-over-real
> refused even with the flag — is **sound and twice re-verified**. Cycle 3 closed four more holes
> and then failed its own re-verify on a demonstrated BLOCKER.
>
> **The BLOCKER:** the new corridor refusal **mutates the tracked fold tree** while exiting 1 and
> printing "Nothing was written." The write at `walkforward.py:1185` is INSIDE the per-fold loop,
> before a later fold's refusal at `:1174-1182` (reproduced on a copy: tree hash `8a44fea7…` →
> `952677ec…`, exit 1). It contradicts the refusal text, the design note at `:1093`, and
> `_expect_refusal` at `:1629-1632` — which names this exact shape as "not fail-closed". The
> smoke's own battery cannot catch it: it perturbs `SMOKE_FOLDS[0]`, so the refusal always fires
> on the first iteration. **A per-fold write-then-refuse loop cannot be fail-closed; the corridor
> writer needs a two-phase design** (compute and decide every fold, then write).
>
> Two MAJORs stand with it: `_prior_db_kind` catches a manifest that LIES but not one that OMITS
> (adding an empty `_provenance: {}` disables the one refusal no flag may override — fixture
> content written over a real-provenance tree); and `_assert_summary_writable` cannot fire on the
> worst summary of all, because `any([])` is False, so a ZERO-fold rollout skips the guard,
> overwrites a real summary, and emits a LEDGER row reading "over 0/0 folds … Scheduled-run
> numbers" — the exact fake-null class this row exists to kill.
>
> **HEAD still carries the original latent landmine** (smoke overwrites 15 tracked real-fold files
> — the task file's "8" is wrong, confirmed by `git ls-files`; content-idempotent today, arms when
> the `SPEC.md:60-72` real re-materialization lands). That was preferred deliberately over
> shipping a demonstrated mutation-on-refusal that HEAD does not have.
>
> **Scope 2 IS done and verified, and survives without a commit.** The `WF_SUMMARY.json` fake-null
> is dead: regenerated from `grade_correct.py` (untouched, md5 `e87b080d…`) via `runpy`, atomic
> write, refusing empty and partial results — `folds_positive=4, folds_above_own_floor=1`, matching
> the LEDGER `875/wf-generalization` row, reproduced independently by three agents. It lives under
> gitignored `runs/875_wf/` on the workstation, so no commit is possible or needed.
>
> **Nothing was lost on the real tree**: `folds/real` tree hash `1b31624a2da9984620567afc30df4cbf`
> before and after every cycle, and gate (iv)'s 875 numbers never moved.
>
> **Resume per D61** — the factory recommends re-cutting the row in two (904A = smoke out-rooting +
> materialize gate + unified guarded reader, all verified sound; 904B = the corridor/aggregate
> writers, two-phase). The row failed because it grew to cover four distinct writers with different
> natural gates, not because the work is wrong.

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
