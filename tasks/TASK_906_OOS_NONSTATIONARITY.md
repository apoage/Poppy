# TASK_906 — the 809/897 per-decision series carries EXCESS DISPERSION across its OOS window

> **STATUS (2026-08-25, factory run 42): NOT SHIPPED — VERIFICATION FOUND TWO BLOCKERS. Work is
> preserved in the tree; do not read the measurements below as settled.** An earlier revision of
> this banner said "SHIPPED" and said scope 4's causes were excluded. Both claims were withdrawn
> by the verification panel and the banner is corrected here rather than edited quietly.
>
> **BLOCKER 1 — the withdrawn banner overclaimed scope 4.** It read "alias/identity, forward-fill
> staleness, and universe turnover EXCLUDED by measurement". The row's OWN artifact says otherwise:
> `experiments/906_oos_nonstationarity/GATE4_CANDIDATE_CAUSES.json` records
> `alias_4a.conclusion = "twin id 1144 HAS rows in this window that the exact-spelling read misses
> -> NOT EXCLUDED, needs the blast-radius sizing TASK_897/D56 already did for its own 64-asset
> universe, redone for this 7-asset one"`. The `experiments/LEDGER.md` row is more honest than the
> banner was ("EXCLUDED IN EFFECT, not vacuously"; "the crude rule this row pre-registered reads
> NOT-EXCLUDED on its own") — the overstatement was confined to the most-read artifact, which is
> the worst place for it. **An unexcluded cause is not an excluded cause** (gate (ii)).
>
> **BLOCKER 2 — forward-fill (4b) was never actually tested: a temporal-misalignment wiring bug.**
> `experiments/906_oos_nonstationarity/measure_causes.py:139-158` correlates `chunk_sharpes_h1`
> (**26** chunks of 1000 REBALANCE DECISIONS, each spanning much of the 5-month window) against
> `chunk_mean_staleness` (**217** chunks of 1000 RAW MINUTE BARS, each ~16.7 h) by truncating to
> `n_common = min(26, 217) = 26` instead of aligning them. So Sharpe-chunk 25 (end of the window)
> is compared against raw minutes 25,000-26,000 (day ~17 of ~150). The reported `|r| = 0.15` vs the
> pre-registered `|r| >= 0.4` bar is therefore a correlation between unrelated calendar spans and
> excludes nothing. Independently reproduced: naive truncation of the raw arrays returns
> `0.15243937660446572` against the script's `0.15243937660446574` — identical to 14 significant
> figures, which is what proves this is the code that ran. `PREREGISTRATION.md:56-58` had specified
> per-DECISION chunking; the 26-vs-217 length mismatch was the tell. **Fix direction:** rebuild
> `chunk_mean_staleness` on the actual rebalance-decision boundaries (or resample the 217-chunk
> series onto the 26 decision-chunk calendar spans) before correlating. The unaffected aggregate
> `overall_frac_bars_stale_gt0 = 5.78%` is correctly computed and still stands.
>
> **Gate (i) is PARTIALLY MET, not met.** `independent_construction.py` imports none of TASK_901's
> modules but runs on TASK_901's own `REAL_SERIES_H{1,5,20}.npy` output arrays. That is a different
> STATISTIC on the SAME CONSTRUCTED SERIES: it cleanly rules out an artifact of 901's
> bootstrap/chi-square machinery, but by construction it CANNOT detect an artifact baked into the
> series itself (alias handling, forward-fill, universe choice, k=2 concentration) — which is
> exactly what scope 4 exists to rule out. `PREREGISTRATION.md:16-23` discloses this narrowing
> honestly; the withdrawn banner did not.
>
> **What DOES stand, pending the fix pass.** The excess dispersion reproduces through an unrelated
> statistic (classical ANOVA+Levene, two outcome-blind segmentations whose pre-registration ordering
> was verified against file mtimes), and is SHARPENED to a VARIANCE effect: Levene rejects flat
> groups at every one of 6 (H, segmentation) cells while ANOVA never rejects the mean
> (p 0.21-1.00) — directionless heteroskedasticity, NOT decay, regime-switch, or a datable level
> shift. Carry the open caveat that Levene's extreme p-values (to 4.3e-274) are read at face value
> on a series this program's own records call anti-persistent (full-series VR 0.273 at q=2048), with
> no dependence-robust check run; the effect size is large (group std ranging 2.8-5.7x within a
> cell) so it is unlikely to flip, but it is an unclosed gap. Causes (c) frozen-cell mask (EXCLUDED
> BY CONSTRUCTION — `price_valid` verified absent from this lineage's loader) and (d) universe
> turnover (EXCLUDED — and cleared as non-tautological: `probe.py:199-211` ranks on TRAIN-window
> bar count, so 100% test-window presence is a real empirical result) both survived adversarial
> checking. A fifth, un-named risk is flagged and unmeasured: portfolio concentration from the
> 7-asset/k=2 demonstration construction.
>
> **Scope 2's convention IS delivered and is NOT withdrawn:** the "NON-STATIONARITY DISCLOSURE"
> doctrine paragraph in `experiments/LEDGER.md`, above its table — where verdicts are read, not
> only here. It was judged careful and correctly hedged.
>
> **Scope 3 affected-verdict read (unchanged, but note the weakest leg).** TASK_897 srv leg #2
> window-(a) cells AFFECTED (same span measured here); window-(b) NOT-EXCLUDED but unmeasured (no
> local per-decision series for that span; building one is a srv-scale job, planner-schedulable).
> TASK_892/D47 ANTI-EDGE NOT AFFECTED — reason is a concrete data-length fact (116 decisions, too
> short to chunk). TASK_875/wf-generalization headline NOT AFFECTED, "OOS seed spread" secondary
> claim QUALIFIED — but this reason is a plausibility argument, not a measurement, and should be
> cited as reasoning rather than as settled.
>
> **Planner-owned, deliberately not done here:** no `docs/pathway.md` write (out of the implementer
> lane; the convention's home was judged to be the ledger). Detail and all candidate-cause
> measurements: `experiments/LEDGER.md` row `906/oos-dispersion`.

**Status**: READY 2026-08-24 (seeded by factory run 41 from TASK_901's gate-(iii) re-scope — a
measurement that arrived as a by-product of testing something else, and outgrew the row that found
it).
**Size: M. Risk: MEDIUM–HIGH in consequence** — this bears on *every whole-window Sharpe claim on
this series*, not on one instrument.

## The measurement

TASK_901 built the real per-decision series through the 806/808/809 lineage to test a dependence
probe. The probe question is settled elsewhere; **this** fell out of it, measured independently by
an opus judge and then reproduced by the implementer at all three horizons:

| | H=1 | H=5 | H=20 |
|---|---|---|---|
| between-chunk sd vs within-chunk SE | 0.09491 vs 0.03538 | — | — |
| variance ratio | **7.19** | 6.37 | 6.73 |
| χ² (25 df) / p | 196.5 / **1.4e-28** | / 1.6e-20 | / 2.1e-33 |

**CORRECTED 2026-08-24, before anyone acts on this row.** The first draft of this file (`7dd8d26`)
said the chunk Sharpes "drift monotonically" and quoted slope −0.00386/chunk. **A re-verify measured
that the trend is NOT significant and the drift claim must be struck.** Refit of the same data:

| H | slope | SE | t | p | R² | Spearman ρ |
|---|---|---|---|---|---|---|
| 1 | −0.00386 | 0.00241 | −1.61 | **0.122** | 0.097 | −0.29 |
| 5 | −0.00274 | 0.00244 | −1.12 | 0.273 | 0.052 | −0.25 |
| 20 | −0.00118 | 0.00264 | −0.45 | 0.660 | 0.009 | −0.17 |

At H=20 the second-half mean (−0.221) is *higher* than the first-half (−0.245) despite the quoted
negative slope, and the "last three chunks are lowest" illustration was a selected 3-vs-3 extract —
chunks 9–12 are as low, chunks 20–21 as high. **Detrending removes essentially none of the excess
dispersion** (variance ratio 7.19 → 6.77 at H=1). So the finding is **excess dispersion, full stop**:
the pseudo-truth is not a constant. It is NOT an established decaying edge, and the two have very
different implications for what 897's whole-window numbers mean.

Supporting facts that rule out the obvious alternative explanation (serial dependence):

- **The bootstrap SE is FLAT in block length**: 0.033644 (block 2) → 0.035912 (block 60) — +6.7%
  across a **30× block change** (H=5 +1.8%, H=20 +1.0%). So serial dependence is not what the
  chunk-to-chunk spread is made of.
- **The series is ANTI-persistent**: chunk VR declines 0.982 → 0.882 at q=64; full-series VR
  **0.273 at q=2048**. A block bootstrap is, if anything, conservative here.

**Conclusion: the population per-decision Sharpe is not constant across the OOS window.** The
spread between chunks is roughly 7× what within-chunk sampling can produce — **and that is the whole
claim.** Whether it trends, switches regime, or varies without direction is exactly what scope item 1
is for; this row must not assume the answer, and its first draft did.

## Why this matters beyond TASK_901

Every verdict this program reads off this series is a **whole-window** statistic — one Sharpe, one
PnL, one interval, over 2024-11 → 2026-03. If the population Sharpe VARIES within that window — in
any pattern, trending or not — then:

1. A whole-window point estimate is an average over a changing quantity, not an estimate of a fixed
   one. It can be negative while a sub-window is positive, and vice versa.
2. **Any interval built on an i.i.d.-or-block-resampling assumption understates uncertainty** — the
   between-chunk variance is 7× the within-chunk variance and no block length inside the current
   `MAX_BLOCK_CAP=60` can represent it. This is a different failure from D49/D55's block-length
   problem and is not fixed by the same remedy.
3. It reframes what TASK_901 was asked to do. D57 parked 897's `CI lo > 0` conjunct pending a
   certified dependence instrument. **If the dominant uncertainty is a moving population parameter
   rather than serial dependence, a range-based block rule is answering the wrong question** — see
   TASK_901's own gate-(iii) finding that containment 0.577–0.654 reflects the dispersion, not
   estimator quality.

## Scope

1. **Characterise the variation — do not assume it is decay.** Trend, regime switch, level shift
   at a datable point, or directionless heteroskedasticity are all live, and the measured trend is
   NOT significant (p = 0.12 / 0.27 / 0.66). Fit and report; state the segmentation BEFORE looking at
   outcomes. **If a trend is claimed, it needs a test that clears its own bar** — the first draft of
   this row asserted one that does not, which is the same defect class TASK_902 item 2 exists for.
2. **Decide what a whole-window claim means on a drifting series** — and write the answer down as a
   convention: report sub-window statistics, or an explicitly time-varying estimate, or state the
   whole-window number as a historical average with its drift disclosed. This is the deliverable.
3. **Establish which existing verdicts are affected.** At minimum re-read, under the chosen
   convention: 897's srv leg (whenever it lands), 892/D47's ANTI-EDGE, and 875's walk-forward
   result. **Re-READ, do not re-run** — this row is an interpretation pass, not a compute pass.
4. **Check whether the dispersion is an artifact of the construction** before treating it as a
   market fact: the alias/identity work (D56, 868/877), the forward-fill of stale prices, the
   frozen-cell mask, and universe turnover across the window are all candidate causes. A *varying
   instrument* would look identical to a *varying market* in this measurement. **This is the gate
   that matters most** — every downstream consequence above assumes the variation is real.

## Acceptance (gates)

- (i) The excess dispersion is reproduced independently of TASK_901's code path (different
  construction, same conclusion — or a demonstration that it was an artifact of that path).
- (ii) Scope item 4 is answered with measurements, and names which candidate causes were excluded
  and how.
- (iii) The convention from scope 2 is written where verdicts are read, not only in this task file.
- (iv) The affected-verdict list from scope 3 is explicit — including any verdict that turns out
  NOT to be affected, and why.

## Out of scope

Re-running any srv leg or GPU battery. Changing TASK_901's instrument conclusions (settled in that
row). The `MAX_BLOCK_CAP=60` ceiling question — related, and TASK_901's other open question, but a
distinct problem: that one is about representing *dependence*, this one is about a *moving
population parameter*. Asserting a TREND without a test that clears its own bar — the first draft's
error, corrected above.

## Provenance

`experiments/901_range_based_dependence_probe/` (`stationarity_diagnostics`, `paired_check.py`,
`REAL_SERIES_H{1,5,20}.npy`), factory run 41. The judge's independent measurement and the
implementer's reproduction agree to the digits quoted above.
