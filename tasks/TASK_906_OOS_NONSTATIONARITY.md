# TASK_906 — the 809/897 per-decision series is measurably NON-STATIONARY across its OOS window

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

Chunk Sharpes drift monotonically (slope −0.00386/chunk; last three −0.380, −0.393, −0.425).
Supporting facts that rule out the obvious alternative explanation:

- **The bootstrap SE is FLAT in block length**: 0.033644 (block 2) → 0.035912 (block 60) — +6.7%
  across a **30× block change** (H=5 +1.8%, H=20 +1.0%). So serial dependence is not what the
  chunk-to-chunk spread is made of.
- **The series is ANTI-persistent**: chunk VR declines 0.982 → 0.882 at q=64; full-series VR
  **0.273 at q=2048**. A block bootstrap is, if anything, conservative here.

**Conclusion: the population per-decision Sharpe is not constant across the OOS window.** The
spread between chunks is roughly 7× what within-chunk sampling can produce.

## Why this matters beyond TASK_901

Every verdict this program reads off this series is a **whole-window** statistic — one Sharpe, one
PnL, one interval, over 2024-11 → 2026-03. If the population Sharpe drifts within that window, then:

1. A whole-window point estimate is an average over a changing quantity, not an estimate of a fixed
   one. It can be negative while a sub-window is positive, and vice versa.
2. **Any interval built on an i.i.d.-or-block-resampling assumption understates uncertainty** — the
   between-chunk variance is 7× the within-chunk variance and no block length inside the current
   `MAX_BLOCK_CAP=60` can represent it. This is a different failure from D49/D55's block-length
   problem and is not fixed by the same remedy.
3. It reframes what TASK_901 was asked to do. D57 parked 897's `CI lo > 0` conjunct pending a
   certified dependence instrument. **If the dominant uncertainty is drift rather than dependence,
   a range-based block rule is answering the wrong question** — see TASK_901's own gate-(iii)
   finding that containment 0.577–0.654 measures drift, not estimator quality.

## Scope

1. **Characterise the drift.** Is it monotone decay, regime change, or a level shift at a
   datable point? Fit and report, do not assert. Use disjoint sub-windows; state the segmentation
   BEFORE looking at outcomes.
2. **Decide what a whole-window claim means on a drifting series** — and write the answer down as a
   convention: report sub-window statistics, or an explicitly time-varying estimate, or state the
   whole-window number as a historical average with its drift disclosed. This is the deliverable.
3. **Establish which existing verdicts are affected.** At minimum re-read, under the chosen
   convention: 897's srv leg (whenever it lands), 892/D47's ANTI-EDGE, and 875's walk-forward
   result. **Re-READ, do not re-run** — this row is an interpretation pass, not a compute pass.
4. **Check whether the drift is an artifact of the construction** before treating it as a market
   fact: the alias/identity work (D56, 868/877), the forward-fill of stale prices, the frozen-cell
   mask, and universe turnover across the window are all candidate causes. A drifting *instrument*
   would look identical to a drifting *market* in this measurement.

## Acceptance (gates)

- (i) The drift is reproduced independently of TASK_901's code path (different construction, same
  conclusion — or a demonstration that it was an artifact of that path).
- (ii) Scope item 4 is answered with measurements, and names which candidate causes were excluded
  and how.
- (iii) The convention from scope 2 is written where verdicts are read, not only in this task file.
- (iv) The affected-verdict list from scope 3 is explicit — including any verdict that turns out
  NOT to be affected, and why.

## Out of scope

Re-running any srv leg or GPU battery. Changing TASK_901's instrument conclusions (settled in that
row). The `MAX_BLOCK_CAP=60` ceiling question — related, and TASK_901's other open question, but a
distinct problem: that one is about representing *dependence*, this one is about a *moving
population parameter*.

## Provenance

`experiments/901_range_based_dependence_probe/` (`stationarity_diagnostics`, `paired_check.py`,
`REAL_SERIES_H{1,5,20}.npy`), factory run 41. The judge's independent measurement and the
implementer's reproduction agree to the digits quoted above.
