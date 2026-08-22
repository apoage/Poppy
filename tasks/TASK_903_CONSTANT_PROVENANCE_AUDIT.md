> **STATUS (2026-08-22, factory run 40): SHIPPED as `task903-constant-provenance`. 8 constants
> enumerated, 9 gates, all classifications independently re-derived by an adversarial judge and held.
> NO VERDICT CHANGED — and both load-bearing verdicts now rest on numbers that were actually
> re-derived. Verified on gpu-server: 9/9 gates, exit 0.**
>
> **875's numbers REPRODUCE EXACTLY — the headline result.** The audit first classified them
> PROVENANCE-UNRECOVERABLE-FROM-WORKSTATION. That was true of the workstation and stated as though it
> were true of the numbers: the artifacts were simply on the other host (`gpu-server:runs/875_wf/`,
> 1.4 GB). Re-derived read-only via `grade_correct.py` — **no re-run of the 8h57m battery** — all five
> folds' medians and floors, all 15 seeds, f5's terminal PVs (59,860/83,721/67,056), zero bankruptcies,
> full step counts, and `folds graded: 5/5 | above own floor: 1` match the LEDGER row exactly.
> **"No OOS edge" rests on numbers that reproduce.** Superseded classification kept visible per D39.
>
> **The file named like the summary is empty.** `runs/875_wf/WF_SUMMARY.json` ships as `[]` — a
> documented silent fake-null (`NOTE_summary_key_mismatch.md`: the orchestrator read
> `sharpe`/`teacher.floor_inenv`; the schema keys are `sharpe_full`/`tiers["21"]["teacher_floor_inenv"]`).
> The provenance chain is the raw per-seed JSONs + `grade_correct.py`, and it is intact.
>
> **892-C4 — a "REQUIRED" precondition the real leg never satisfied.** The 2000-trial
> `calibration_real_leg.json` exists but was never wired in; `report.py:1082` embeds the old
> 200/400-trial table in every shipped report. **Non-material, checked not asserted:** recomputed
> across every calibrated field for all three seeds — no sign change, no zero-crossing, no band change.
> **ANTI-EDGE holds; D47 untouched.**
>
> **A skipped gate can no longer certify a finding it did not make.** Exit contract: `0` = every gate
> ran and passed; `1` = a check failed; `2` = nothing failed but a gate skipped fail-closed. **A skip
> never exits 0**, and the summary is built only from gates that ran. Verified both ways.
> Also removed: a "self-cleaning" `git checkout --` that silently destroyed unrelated uncommitted work
> at exit 0 — replaced by a throwaway worktree with a fail-closed pre-check.
>
> **Classifications:** REPRODUCES — 892-C1, 892-C3, 892-C6, 875-C1. DRIFTS-with-named-cause
> (both non-material) — 892-C4, 837-C1. PROVENANCE-UNRECOVERABLE (non-material, disclosed) —
> 892-C2, 892-C5.
>
> **REMAINDERS → planner.** (1) `walkforward.py smoke` overwrites 8 tracked `folds/real/*` files with
> fixture-derived content, unguarded — content-idempotent TODAY (only `generated` stamps move), so a
> latent landmine that arms the moment the `SPEC.md:60-72` real-db re-materialization is committed.
> (2) None of `runs/875_wf/`, `grade_correct.py`, `orch_wf875.sh` has ever been committed: a
> load-bearing verdict's evidence lives on one host, gitignored — D48's cited-but-untracked class.
> (3) `report.py:220-250`'s two MEASURED-MARGIN prose values are transcribed an order too small
> (non-material — no code path reads the prose); 892's own lane.

# TASK_903 — the constant-provenance question, pointed at 892's calibration table and 875's walk-forward numbers

**Status**: READY 2026-08-22 (planner; seeded by D59's ruling, answering the run-39 factory's
question "should the same question be pointed at other lanes?" — YES).
**Size: S/M. Risk: LOW to run, potentially HIGH in consequence** — a drift found here can put a
closed verdict back on the table; the row STOPS and escalates in that case (see gate iv).

## Why this row exists

Run 39's transferable result was not any fix but one question: **is each shipped constant the
exact output of the command its own comment advertises?** Asked of TASK_897's instrument, it
failed five of seven times — including a "within MC noise, never bit-identically" claim on a
bit-deterministic path, "fresh random draws each run" over deterministically-seeded strata, one
unrecoverable provenance, and one bar widened 3pp→6pp to fit its result. Nothing suggests that
module was unusual; it is simply the one that got asked. Two shipped lanes carry constants that
have never been asked:

- **892's calibration table and shipped constants.** The 892 instrument received no independent
  provenance pass; its real-leg verdict (ANTI-EDGE, LEDGER 892/action-ev) closed the self-play
  line and fed D47's pivot. The verdict was read off calibrated CIs — if the calibration
  constants do not correspond to their advertised commands, the CI claim is unverified.
- **875's walk-forward numbers.** The pathfinder audit (2026-08-19) re-derived 875's VERDICT and
  found it SOLID — but it never asked the per-constant provenance question. 875's numbers
  underpin "the losses are real" and the imprint lane's closure as deployable.

## Scope

1. **Enumerate** every shipped constant in the 892 and 875 instruments/artifacts whose comment
   or artifact text advertises a generating command, seeds, or a reproduction claim.
2. **Ask the question of each:** re-run the advertised command; compare EXACTLY where the path
   is deterministic, or against the stated noise model where one genuinely exists (and verify
   the noise model is real — a deterministic path has no MC noise to disclose).
3. **Classify** each constant: REPRODUCES / DRIFTS-with-named-cause / PROVENANCE-UNRECOVERABLE.
   Re-pin or re-disclose per D39 — superseded records kept visible, never silently replaced.
4. **Sensitivity, not re-grading.** For every drift found: state which artifact claim and which
   verdict conjunct it touches, and whether the drift is material to that claim. The row does
   NOT re-grade 892 or 875 — material drift escalates to the planner as a DECISIONS row with the
   evidence attached.

## Acceptance (gates)

- (i) The enumeration is complete: a test or manifest lists every constant asked, so "we asked"
   is itself checkable (the 881G gate-(i) doc-only-author shape).
- (ii) Every classification is demonstrated by a re-run, with the command and result in the
   artifact.
- (iii) Any re-pinned constant carries its superseded value visibly (D39).
- (iv) Every material drift — one that changes the reading of a verdict the LEDGER carries —
   surfaces as a seeded DECISIONS row before the audit task closes. Absence of drift is also
   the deliverable: it converts "never asked" into "asked, holds".

## Out of scope

Re-running the 892 or 875 real legs. Changing any estimator or calibration METHOD (that is
TASK_901's lane for dependence intervals). Touching the 809 lineage. Verdict changes of any
kind — this row measures provenance; rulings change verdicts.
