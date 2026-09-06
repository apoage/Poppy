# Factory run 55 handoff — TASK_919 phase A (2026-09-06)

Lane: implementer. Armed at session start, tree carrying the D82-fenced TASK_914 candidate
(intentional, untouched throughout — `capture.py` numstat 71/32 identical at start and end,
`test_streaming_reader.py` untracked throughout).

## What shipped

`experiments/919_execution_turnover/` — the H x theta x accounting rebalance grid, plus the
instruments the row grew while being verified. Zero edits to `trader/` across all four passes
(`git log -1 -- trader/` still `18d42d3`). Fixture `b0f8de8b...` never regenerated (checked at
every pass boundary). Suite 7 -> 16 checks; `python -m trader selftest` 17/17 throughout.

## Cycle history — 3 cycles + a D76 closing pass

| cycle | model | outcome |
|---|---|---|
| 1 | sonnet | FAILED — 2 BLOCKER (hold-shares not share-carrying; RESULTS table disagreed with its own artifact on 26/48 rows, up to +102%), 3 MAJOR |
| 2 | opus | FAILED — BLOCKERs fixed and verified, but the panel found the leverage confound the pass itself called unresolvable, plus a vacuous prose gate |
| 3 | opus | FAILED on framing — no BLOCKER, but 4 MAJOR: the leverage headline is construction-dependent and reverses under the other construction |
| closing | opus | D76 exception — reframing + transcription of judge-collected evidence ONLY. No new solution attempt, no new evidence. |

**D76 DEVIATION, DISCLOSED HERE AND IN THE VERDICT, NOT IN A FOOTNOTE.** The closing pass exceeds
`halt_after_failures = 3`. It qualifies because it (a) replaces framing THIS RUN proved defective
(cycle 3's own panel refuted cycle 3's headline), (b) only transcribes evidence judges had already
collected and confirmed — the constrained-book table, the gate-quantity arm, the exposure figures —
with no new solution attempt and no new evidence-gathering of its own, and (c) is disclosed in the
verdict itself. Precedent: run 48 cycle 4 (un-shipping its own wrong SATA-cable culprit); run 49 is
the compliant 3-cycle precedent.

## The two results

**1. The pre-registered band question is REFUSED FOR WANT OF POWER.** SE(annualized gate Sharpe) =
0.447-0.457, H-invariant (n*H ~ 1704 constant exactly cancels sqrt(365/H) — a correctly designed
invariance). Two independent derivations agree to 0.0023. MDE@80% = 1.152 raw (5.8x the +/-0.20
band) / 1.815 Bonferroni (9.1x). Measured cell correlation (min 0.550, median 0.910) does not
rescue it: the most favourable non-degenerate pair (rho 0.9837) still needs 1.6x the band.
**0/48 cells clear +0.20 at alpha=0.05/48 on all three windows; 46/48 UNDECIDED.** ~85x sample
needed. Same disposition class as run 54's TASK_917 P1. This conclusion is PRIOR TO and INDEPENDENT
OF every code defect found — no re-implementation rescues it.

**2. The hold-shares advantage is UNRESOLVED, and the construction-dependence IS the finding.**
See the LEDGER rows. Two de-levering constructions disagree in sign; neither clears zero.

**3. The one execution-shaped survivor: theta.** Of the candidate cell's +0.6523 uplift over the
frozen cell, **+0.3263 (50%) is the pure theta hysteresis effect** — canon target-weight accounting,
surviving de-levering entirely. Does not clear the family's Bonferroni bar; recorded as the most
promising lead, not a result. Cycle 3 had mislabelled this as leverage.

## Corrections to the record (all measured, see LEDGER)

1. The inherited analytic flip model's FORMULA reproduces none of its own quoted numbers except by
   coincidence at H=2, and the quantity was mis-denominated (turnover per day, not flip probability).
   Correct model: `(2/pi)*arccos((L-H)/L)/H`.
2. Hysteresis at theta=0.5 removes **73.1%** of flip turnover, not the recorded "40-50%"
   (theta=0.25 -> 59.8%, theta=1.0 -> 83.4%). theta is not the cheap free lunch the record implies.
3. TASK_919's spec quoted the hold-shares BEAR22 anchor BACKWARDS (+1.218 is the constant-dollar
   reference, +0.225 the fixed-shares floor). Confirmed against `830_corridor/SPEC.md:95-98` and
   LEDGER `830/E0` by two independent lenses.
4. The hold-shares attribution split does not clear its own noise floor (share-carry -0.0328,
   CI [-0.6065,+0.5132]); DIRECTION kept, SPLIT downgraded to UNRESOLVED.

## Data provenance (verified twice this run, read-only)

`ohlcv_daily` is BYTE-IDENTICAL between `fixtures/trading_fixture.db` and `trading_data.db`:
sha256 `90d67f40eb2aee31...`, 499,430 rows, 411 assets, 2013-10-06 -> 2026-07-06, both sides.
**So the daily grid produces PRODUCTION numbers, not fixture proxies.** The minute table is NOT at
parity (fixture 11 assets / 3 months vs source 1,130 assets / 254,904,234 rows), which is exactly
why the hour-of-day arm is mechanism-only with a scheduled real leg — not an omission.

## Process findings — SEVEN guards that could not reach their own claims

This run's dominant lesson, and direct evidence for **D83 candidate (i)**:
1. `test_g2_no_trader_edits` imported `trader` and returned a string asserting no edits.
2. A code comment cited a regression test in `test_919_grid.py` that was never written.
3. `hold-shares` named a mechanism it did not implement (a compounding swap).
4. RESULTS.md's headline table was disconnected from its own artifact (26/48 rows, up to +102%).
5. The prose gate fenced only inside its markers; a judge planted a poison one line outside and
   `--check` returned exit 0. (Measured reach after the fix: **50.9%**; 2-dp literals **1%**.)
6. The attribution split was asserted with directional confidence and no error bars BY THE SAME
   DOCUMENT that refused the band question for want of power — an internal double standard.
7. **The orchestrator's own drafted LEDGER row** said "entirely gross-exposure scaling / dispositions
   WITHDRAWN" — caught by an estimator lens's PRE-COMMITTED condition ("upgrade to BLOCKER if the
   verdict says refuted rather than unresolved"). The lens anticipated the failure mode before it
   happened.
**Five of the seven were caught only by an adversarial lens. None was caught by its author.**

Second finding, for **D83 candidate (ii)** (artifact+revision provenance per number): three of the
four record corrections above were provenance failures — numbers inherited from `tactics_gaps.md`
and the task spec without recomputation, two of which were simply wrong and one backwards.

Third: a lens correctly treated the orchestrator's mid-flight SendMessage updates as UNVERIFIED
input arriving over a tool channel and re-derived every claim from code before using it — and
caught one of the orchestrator's own claims (MINOR-O1) as partly wrong in the process. That is the
right handling and it is what made the panel work.

## Open for the operator
- **D85** (seeded this run) — what happens to the execution/turnover line. Three options; factory
  recommends (a) FOLD AND CLOSE + promote the MDE bound to a spec-time gate.
- **D82, D83, D84, D81(a)/(b), D80(ii)/(iii), D43, RO02** — all still OPEN, untouched by this run.
- Follow-up rows NOT taken (would have been new solution attempts): add borrow to
  `renormalised_to_gross1`; key the prose whitelist by (value, context) and refuse sub-3-decimal
  matches; emit an `hour_of_day` result JSON so its figures leave the whitelist; the
  gross-renormalised hold-shares arm as a first-class scored variant.
