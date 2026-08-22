# TASK_901 — the range-based dependence probe (upgrades every block-bootstrap CI on dependent data)

**Status**: READY 2026-08-22 (planner; seeded by D57's option (b), after the τ guard was measured
inert out-of-family on the real minute data).
**Size: S/M. Risk: MEDIUM** — this is a new measurement instrument, and its own certification is
the trap: it MUST be validated out-of-family, or it repeats the τ guard's exact failure (valid
only inside the family it was calibrated on).

## Why this row exists

A block bootstrap needs block ≳ correlation **range**; τ (integrated autocorrelation time) mixes
range with amplitude — measured: a persistent component worth 4% of variance (`0.2·AR1(φ=.99)`)
reads τ=1, certifies 100%, covers 0.805; at φ=.999 coverage is 0.376. On 897's real per-decision
series every cell reads τ = 1–2 against block 60, so a τ-guard certifies everything. 897's
`CI lo > 0` conjunct is UNAVAILABLE until this probe exists (D57(a)); when this row ships, 897's
verdict is re-read from the same artifacts with no srv re-run. The probe is also the general
instrument any future per-decision CI on dependent data needs (892's daily-bar calibration was
keyed to block=21=range by luck of horizon; nothing else in the program is).

## Scope

1. **A range estimator with a stated claim.** Candidates, in order: (i) first lag where the ACF
   enters the white-noise band (with the band's own multiple-comparisons correction stated);
   (ii) a variance-ratio estimate at a grid of lags; (iii) spectral density at zero. Pick by
   measured coverage, not by appeal — and say in the artifact WHAT the estimator assumes.
2. **Out-of-family validation harness.** The probe must be certified against dependence
   structures OUTSIDE the family used to build it: triangular ACF, AR(1) mixes at φ ∈
   {.9,.99,.999}, the 4%-variance persistent component that broke τ, seasonal/regime-switching
   dependence, AND the real minute per-decision series (the 7 majors, 2024-11→2025-04, H ∈
   {1,5,20}) with an empirical containment check (the 97-disjoint-chunks construction the D57
   halt used — it contained the grand Sharpe in 0.639 of chunks while τ certified 100%).
3. **The certified quantity feeds a block rule:** given n_dec and the estimated range, the
   instrument returns a block length and a calibrated-width factor (or refuses as insufficient-n),
   with the whole chain's coverage re-measured at the certified pairings.

## Acceptance (gates)

- (i) Certified coverage ≥ 93% on every validation family, INCLUDING the families the estimator
  was not tuned on — measured, with the table in the artifact.
- (ii) The 4%-variance persistent component (τ's killer) reads range ≫ 1 and either calibrates
  correctly or refuses.
- (iii) On the real minute series, the empirical containment check is run and reported — the
  probe's certification must not exceed the measured containment.
- (iv) When this row ships: 897's `CI lo > 0` conjunct is re-read from the EXISTING srv-leg
  artifacts (same traces, new interval) — recorded as a follow-up read, not a new run.

## Out of scope

Re-running any srv leg. Touching 892's daily-bar calibration (it is certified for its own
statistic and block regime — D55's reuse warning). Editing the 809 lineage.
