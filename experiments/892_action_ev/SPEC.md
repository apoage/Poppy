# 892/action-ev — does ANY policy on this recipe have positive-EV actions?

**PRE-REGISTRATION. Written 2026-08-16 by the planner before any instrument exists; hash to be
recorded in the task file's launch note before the real leg runs. No training anywhere in this
experiment — eval-only on existing, fingerprint-stamped checkpoints.**

**AMENDMENT 1 — 2026-08-19, a deliberate PLANNER act (D49's condition — never edited by the row
that deviated), new body hash recorded in `tasks/TASK_892_ACTION_EV_AUDIT.md`.** Carries three
sources, all landed BEFORE the real leg runs: (i) **D49(a), operator-ruled 2026-08-19** — the
delivered event-study estimator is ACCEPTED for the cost0 PRIMARY read (provably cost-neutral
there); the fee16/slip5 sign-persistence check is DEMOTED to a reported-but-non-binding
diagnostic; and D49's interval-method escalation is adopted (measured pre-launch: block-bootstrap
coverage plateaus ~88% under realistic dependence and never reaches nominal — the pre-registered
block=21 EQUALS the dependence length, so the remedy is the interval method, not more data);
(ii) **D51's re-cut** — the report's banding/exclusion layer lands together with this amendment;
(iii) **four operator-adopted pathfinder amendments** (`docs/pathfinder_audit_2026-08-19.md` §D).
Inline changes flagged [A1].

## Why this question, and why now

`887/collapse-confirm` relocated the program's live question. The hold collapse is ADAPTIVE:
at zero costs, every mechanism that makes the policy act more makes it lose more (median final-50
return A −2.86% · E −9.11% · D −13.84%, monotone in activity). Underneath the collapse, the
distill FAIL (`879/rung3`), the teacher-quality FAIL (`861/search-teacher-imprint`), and the
walk-forward FAIL (`875`, 1/5) sits one unmeasured possibility: **the policies' non-HOLD actions
may simply carry no edge** — in which case every negative so far is one finding, and the pivot is
to reward/data/edge structure, not to more policy machinery.

That possibility is directly measurable without training anything. This experiment measures it.

## Question

Conditioned on a trained policy taking a **non-HOLD** action at a decision, is the realized
forward return different from what a **matched-activity random policy** would have realized at
comparable states?

## Subjects

- **Primary:** the 3 clean-semantics reference checkpoints `runs/861_ref/ckpt_ref_s{0,1,2}`
  (post-D31 recipe, param_fingerprint retro-stamped `8123a6b`, loaded through the 888R refusing
  resolver). Operator: 848 policy-argmax, weight env (`--vector-weight-rebalance`), teacher floor
  in run.
- **[A1] Second subject class (never pooled with the imprint primaries):** one 887 A-arm
  (collapsed) and one 887 D-arm (un-collapsed) ep300 snapshot from `runs/887_confirm` — the
  from-scratch self-play policies 887's claim was actually about. These directly test "the
  collapsed policy's rare actions" (A) and "the un-collapsed policy's frequent actions" (D) —
  the exact quantities 887's interpretation rests on.
- **Secondary (budget permitting, reported separately, never pooled with primary):** the
  +8.02-class search operator (visit-argmax, sims 32, dithered-book K16, true-env leaf) on
  `ckpt_ref_s1` — does SEARCH add per-action EV over argmax? **[A1] Label: the true-env leaf is
  an R-step LOOKAHEAD — this leg is a CAPACITY CEILING, not a deployable comparison.**
- **Contrast (reported separately):** `runs/832_e5/ckpt_imprint_full` (old-era anchor).
  **[A1] This leg inherits the unresolved 882 epoch confound (post-hoc reconstructed frame,
  mismatched window) — it is EXCLUDED from the D47 evidence base, not merely reported
  separately.**

## Design

1. **Policy walks, two legs [A1].** (a) **BINDING leg:** replay each subject over the OOS window
   **2025-10-05→2026-03-31** (the 861 scorecard's `oos_full` window, 116 decisions, 62/64 slots —
   the 861 artifacts prove it runnable). The D47 verdict is read from THIS leg: an in-window EDGE
   is near-foregone (the ref ckpts are 97.9%-agreement clones of a teacher whose in-window floor
   is positive) and nearly information-free. (b) **SANITY leg:** the original window
   2021-01-01..2025-10-04 (the 882 Leg-A window, 1673 decisions) — in-window NULL/ANTI-EDGE would
   be devastating and is worth having, but only OOS EDGE can re-open the line. Both legs at two
   cost tiers: **cost0** (fee0/slip0 — the 887 instrument; separates edge from fees) and
   **fee16/slip5** (honest costs — [A1] per D49(a) this tier is a REPORTED DIAGNOSTIC, never a
   band condition, until an env-accounting estimator exists). Record every decision: taken action
   vector, position deltas, **net exposure [A1]**, and the realized forward H=21 net return
   attributable to the decision, **plus the forward H=21 MARKET return of the held assets over the
   same horizon [A1]** — so a directional artifact (long book in a rally) is separable from
   timing. **[A1] Estimator disclosure (D49(a) condition):** the delivered estimator is an
   event-study estimator (counterfactual hold of the entry direction for exactly H decisions,
   entry-side cost only), NOT the env-accounting/PV-exact walk replay this section originally
   froze; every artifact must disclose the delivered estimator, this deviation, and the bias
   DIRECTION (costs understated ⇒ sign-persistence flattered, asymmetrically by turnover).
2. **Null walks.** Same window, same costs, same env: R ≥ 20 seeded walks of a **matched-activity
   random policy** — per decision, each asset's action drawn from the corresponding subject
   policy's own marginal per-asset action distribution (measured on that subject's walk), so the
   null has the SAME activity rate, long/short mix, and average book width, but no state
   conditioning. A scripted operator (`--decision-rule random-matched`, seeded, path-independent)
   implements it; seed sweep makes the null a distribution, not a walk.
3. **Comparison.** Per-decision realized forward returns of policy non-HOLD decisions vs the
   null's, matched within regime bucket (the DEC-2 regime windows) so a bull-heavy policy book is
   not credited for beta. Block-bootstrap CI over decisions (block = 21, the horizon), 10k
   resamples.

## Metric and pre-registered bands

Primary quantity: **Δ = mean realized forward H=21 net return after policy non-HOLD decisions −
the same quantity over the null's decisions**, at cost0, pooled over the 3 ref seeds (per-seed
reported; pooling only if directions agree), **on the BINDING (OOS) leg [A1]**.

**[A1] Co-primary: position-CHANGE-only Δ.** The primary subjects' action streams are ~90%
non-HOLD cells that are mostly deadband re-trims of a standing book; the headline Δ therefore
mostly measures book quality (≈ the teacher's signal), not decision quality. Report separately:
**entries/exits** (sign change or 0↔nonzero) vs **maintenance re-trims**. The entry/exit Δ is
the decision-quality read; the re-trim Δ is reported for completeness.

**[A1] Interval method — pre-registered calibration requirement (D49's escalation).** The
originally pre-registered block bootstrap (block=21, 10k resamples) is measured UNDER-CALIBRATED
for this data: consecutive forward windows overlap by up to 20, so the block length EQUALS the
dependence length, and measured coverage under realistic dependence plateaus ~88% and never
reaches nominal at any attainable n. The remedy is the interval method, not more data.
Pre-registered requirement: **before the real leg runs, the instrument's two-scenario
calibration harness must certify the chosen interval method at ≥ 93% two-sided coverage under
the realistic-dependence scenario at EACH reported stratum's block count** (pooled, per-regime,
entry/exit, both leg sizes — the OOS leg's 116 decisions yield ~6 blocks at block 21, the
hardest case). Candidate remedies, in order: (i) block length > dependence length where n
permits; (ii) subsampled non-overlapping event windows at stride H; (iii) a calibrated-width
correction. The chosen method, its measured coverage per stratum, and any stratum that could
NOT be calibrated (reported with the insufficient-n flag, never a bare CI) are disclosed in the
artifact. Any guard in the banding layer is a **tolerance relative to the data's own scale,
never an exact-equality or bit-identity test, and its test constructs the NEAR case** (D51's
transferable rule).

- **EDGE** — lower 95% CI of Δ > 0 at cost0 **on the OOS (binding) leg [A1]**, AND Δ keeps its
  sign at fee16/slip5 **[A1] — per D49(a) the fee16 leg is a REPORTED DIAGNOSTIC ONLY: its sign
  is disclosed alongside the disclosed estimator bias, but it neither makes nor breaks the
  band**. ⇒ the recipe
  has an edge worth shaping; the pivot targets mechanisms that protect it (and 887's collapse is
  re-read as the policy failing to FIND its own edge, not correctly refusing one).
- **ANTI-EDGE** — upper 95% CI of Δ < 0 at cost0 **on the OOS leg [A1]**. ⇒ 887's read confirmed at decision level: the
  actions themselves destroy value; the pivot is to reward/data/edge structure and this becomes
  the recorded structural finding the next hypothesis is built on.
- **NULL** — CI spans 0 **on the OOS leg [A1]**. ⇒ actions are noise; same pivot as ANTI-EDGE, weaker form. Also a
  finding: it bounds any future policy improvement to finding edge that is not currently expressed.

**All three bands are publishable outcomes. There is no fourth, silent outcome: any failure of
the instrument itself (null activity mismatch > 0.5pp, PV conservation violation, < 30 non-HOLD
decisions per seed) is an INSTRUMENT FAIL, reported as such, not folded into a band.**

Thin-n disclosure, pre-registered and **[A1] CORRECTED** (the original text undercounted by ~2
orders of magnitude): the ref ckpts' own artifacts show ~89.6% non-HOLD action cells
(51,354+44,607 of 107,072 in `runs/861_ref/results_s0.json`'s `policy_action_hist`) — the book
re-marks almost every slot daily, so non-HOLD decisions number ~95k/seed pre-blocking, not "low
hundreds". The binding constraint is the BLOCK structure: ~80 blocks of 21 on the sanity leg,
**~6 blocks on the 116-decision OOS binding leg** — the honest number to pre-disclose, and the
reason the [A1] calibration requirement above is binding per stratum. Within-regime matching
thins further; any stratum that cannot be calibrated is flagged, never bare-CI'd.

## Cost

Instrument build = factory (fixture smoke + the [A1] calibration harness). Real leg = one
scheduled run (policy walks: 3 primaries + 2 × 887 subjects + secondary + contrast × 2 legs × 2
tiers ≈ 28 walks, plus R ≥ 20 null walks per subject×leg×tier cell). **Operator standing
guidance 2026-08-19: schedule to keep the server GPUs fed; no CPU-heavy work on the
workstation** — the real leg runs on gpu-server's idle GPUs. No training. Verdict → LEDGER
`892/action-ev` → D47 (the pivot ruling; D42–D46 were taken by factory runs 34/35; this program
burns decision numbers fast).

## Explicitly NOT in this experiment

No new training run, no rung-1 re-spec, no epsilon schedule. `manifest_884_battery.txt` stays
unlaunched regardless of outcome: EDGE re-specs rung 1 from scratch around the found edge;
ANTI-EDGE/NULL closes the self-play line pending a new structural hypothesis.
