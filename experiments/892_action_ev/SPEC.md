# 892/action-ev — does ANY policy on this recipe have positive-EV actions?

**PRE-REGISTRATION. Written 2026-08-16 by the planner before any instrument exists; hash to be
recorded in the task file's launch note before the real leg runs. No training anywhere in this
experiment — eval-only on existing, fingerprint-stamped checkpoints.**

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
- **Secondary (budget permitting, reported separately, never pooled with primary):** the
  +8.02-class search operator (visit-argmax, sims 32, dithered-book K16, true-env leaf) on
  `ckpt_ref_s1` — does SEARCH add per-action EV over argmax?
- **Contrast (reported separately):** `runs/832_e5/ckpt_imprint_full` (old-era anchor).

## Design

1. **Policy walks.** Replay each subject over the eval window 2021-01-01..2025-10-04 (the 882
   Leg-A window, 1673 decisions) at two cost tiers: **cost0** (fee0/slip0 — the 887 instrument;
   separates edge from fees) and **fee16/slip5** (honest costs). Record every decision: taken
   action vector, position deltas, and the realized forward H=21 net return attributable to the
   decision (the env's own accounting; PV conservation exact).
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
reported; pooling only if directions agree).

- **EDGE** — lower 95% CI of Δ > 0 at cost0, AND Δ keeps its sign at fee16/slip5. ⇒ the recipe
  has an edge worth shaping; the pivot targets mechanisms that protect it (and 887's collapse is
  re-read as the policy failing to FIND its own edge, not correctly refusing one).
- **ANTI-EDGE** — upper 95% CI of Δ < 0 at cost0. ⇒ 887's read confirmed at decision level: the
  actions themselves destroy value; the pivot is to reward/data/edge structure and this becomes
  the recorded structural finding the next hypothesis is built on.
- **NULL** — CI spans 0. ⇒ actions are noise; same pivot as ANTI-EDGE, weaker form. Also a
  finding: it bounds any future policy improvement to finding edge that is not currently expressed.

**All three bands are publishable outcomes. There is no fourth, silent outcome: any failure of
the instrument itself (null activity mismatch > 0.5pp, PV conservation violation, < 30 non-HOLD
decisions per seed) is an INSTRUMENT FAIL, reported as such, not folded into a band.**

Thin-n disclosure, pre-registered: ref-seed books are ~26L/31S on 64 assets, so non-HOLD
decisions number in the low hundreds per seed; the CI will be honest about it, and the
matched-n comparison is within-regime, which thins further. That is the price of measuring on
policies that actually trade.

## Cost

Instrument build = factory (fixture smoke only). Real leg = one scheduled workstation run
(3 subjects × 2 tiers + ~60 null walks ≈ 84 lightweight evals; est. 2–4 h CPU/single-GPU).
No server battery, no training. Verdict → LEDGER `892/action-ev` → D44 (the pivot ruling;
D42/D43 were taken by factory run 34's 891-halt and 857-smoke-assert decisions).

## Explicitly NOT in this experiment

No new training run, no rung-1 re-spec, no epsilon schedule. `manifest_884_battery.txt` stays
unlaunched regardless of outcome: EDGE re-specs rung 1 from scratch around the found edge;
ANTI-EDGE/NULL closes the self-play line pending a new structural hypothesis.
