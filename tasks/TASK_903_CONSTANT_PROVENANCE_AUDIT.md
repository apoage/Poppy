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
