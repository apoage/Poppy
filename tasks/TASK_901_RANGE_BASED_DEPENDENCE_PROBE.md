> **STATUS (2026-08-24, factory run 41): SHIPPED as `task901-range-dependence-probe` (`5c9932b`) —
> a MEASURED NEGATIVE. Gates (i)–(iii). Gate (iv) DEFERRED and unfabricated** (`gate_iv_hook.py`
> raises `Gate4NotYetRunnable`; 897's srv-leg artifact does not exist yet, and no 897 number appears
> anywhere in this row).
>
> **Headline: no candidate in {`acf_band`, `variance_ratio`, `spectral_zero`} is certified as a
> general range instrument at ≥93% out-of-family.** That survived three implementation cycles and
> three independent verifications **unchanged**.
>
> **What kept changing were the REASONS — and that is this row's real deliverable.** Cycle 1
> condemned `variance_ratio` by a one-step read-off rule its own docstring did not describe (the
> false plateau reproduces on the POPULATION VR curve, with zero sampling noise). Cycle 2 fixed that
> and condemned it again, by a tail rule with a **fixed** 0.10 tolerance against a tail whose
> sampling sd reaches **0.319** — a population-flat curve cannot pass it, and an SE-aware tolerance
> finds a plateau in **100%** of the same draws. Cycle 3 stopped attributing and started separating.
> Each version implied a different next experiment: *abandon the approach* → *abandon it differently*
> → **the question is open**. Only the last is what was measured.
>
> **Per candidate.** `acf_band` and `spectral_zero` fail conclusively (13–44σ on the 4%-variance
> persistent component that broke τ). **`variance_ratio` OVER-CERTIFIES at a low rate** — 0.185
> pooled over four independent draws on persistent φ=.999 — and cycle 2's "does not over-certify
> anywhere" is **WITHDRAWN**. Its high refusal rate is **NOT separated** from the read-off rule and
> the block ceiling; that is stated as UNKNOWN rather than resolved from a counterfactual.
>
> **TWO INSTRUMENT DEFECTS SURFACED, DISCLOSED, NOT FIXED** — both fixes would be knobs turned after
> seeing a number, so they are the planner's:
> 1. **`MAX_BLOCK_CAP=60` does most of the "refusing."** On the persistent component the estimator
>    returns a finite plateau in 16.4% of draws and **163 of 164 are killed by the cap** — a ceiling
>    that would refuse **τ** too. Gate (ii)'s "pass" is therefore part ceiling, not all estimator.
> 2. **One calibration stratum of 27** (`block=3, n_dec=3000`) fails at **0.925 vs 0.930** on a
>    200-trial batch (z = −0.27 — a coin flip) while suppressing up to **32%** of some cells'
>    certifications. Found by the implementer, and it **overturned a judge's explicit "verified
>    clean"**; reproduced bit-identically from a cleared cache.
>
> **Gate (iii) cannot discriminate between estimators, and its finding is EXCESS DISPERSION, not
> drift.** Between-chunk variance ≈7× within-chunk (p = 1e-20…1e-33 at all three horizons) while the
> bootstrap SE is **flat in block length** (+6.7% across a 30× change) and the series is
> **anti-persistent** (VR 0.273 at q=2048). The trend does NOT clear its own bar (slope p = 0.122 /
> 0.273 / 0.660), so "monotone" was struck. **Seeded as `TASK_906`** (corrected in `d7b0d3f` after
> the drift claim failed re-verification). Containment 0.577–0.654 is **withdrawn as evidence against
> any estimator, including retroactively against τ** — τ's synthetic 0.805/0.376 is a different
> measurement and stands.
>
> **Consequence for TASK_897:** its `CI lo > 0` conjunct is **NOT YET ESTABLISHED** — not
> "structurally unavailable". That overstatement was the orchestrator's and was corrected in
> `2eaa844`.
>
> **Open questions handed forward:** (1) `acf_band` reads range 18 and *still* under-covers, pointing
> at the `MAX_BLOCK_CAP` ceiling and the matched-dependence calibration convention rather than the
> range read; (2) whether an SE-aware `variance_ratio` read-off is usable is **UNKNOWN** — the row
> declines to assert it either way from a counterfactual; (3) `RESULTS.json` does not record the
> calibration table it consumed (`calib_table.py` reconstructs it; `run_validation.py` should persist
> it).
>
> _Read `GATE_II_VERDICT["variance_ratio"]` with the certify-rate column. Its gate-(ii) "pass" is a
> REFUSAL, and the implementer flagged that shape as inherently misreadable even after stating it
> three ways._

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
