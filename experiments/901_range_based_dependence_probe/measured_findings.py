#!/usr/bin/env python3
"""TASK_901 — the pinned, measured out-of-family certification result.

This module IS this row's prose artifact (there is no separate `REPORT.md`), kept as CODE (pinned
constants + this docstring) rather than a hand-written narrative file, matching this repo's own
convention for recording measured evidence (`experiments/897_corrected_809/calibration.py`'s
`MEASURED_CALIBRATION_TABLE` / `MEASURED_TAU_OUT_OF_FAMILY_HALTED_CYCLE`, `experiments/
897_corrected_809/calibration.py`'s `BLOCK_CAP_SELECTION_RATIONALE`). The full run's
machine-readable output is `RESULTS.json` (synthetic sweep + real-data leg merged by
`combine_results.py`); `MEASURED_COVERAGE_TABLE` below is a summary of that JSON, rendered from it
rather than hand-transcribed, for a reader who wants the numbers without parsing it.

SCOPE SHIPPED: gates (i)-(iii). Gate (iv) is deferred (`gate_iv_hook.py`) — 897's srv leg was
mid-relaunch when this row was implemented; no 897 number is fabricated, stubbed, or approximated
anywhere in this module.

============================================================================================
HEADLINE (fix cycle 3, 2026-08-24 — THE CONCLUSION IS UNCHANGED FROM CYCLE 2; THREE OF ITS
STATED REASONS WERE WRONG AND ARE REPLACED):

  NO CANDIDATE IN {acf_band, variance_ratio, spectral_zero} IS CERTIFIED AS A GENERAL RANGE
  INSTRUMENT AT >= 93% OUT-OF-FAMILY. The three fail in different ways, and this row has now been
  wrong twice about WHY:

    * `acf_band` and `spectral_zero` FAIL out-of-family, measured: on the 4%-variance persistent
      component they certify 71-85% of draws whose real coverage is 0.72 / 0.63 (phi=.99) and
      0.26 / 0.26 (phi=.999) against a 0.93 bar. That is over-certification, the forbidden middle
      ground, and it is 13-44 sigma from the bar. Their negative is solid and is untouched by
      everything corrected in cycle 3.
    * `variance_ratio` OVER-CERTIFIES WHERE IT LEAKS, and its headline "refusal" is not one thing
      and is not a measured property of the variance-ratio statistic. Three separate corrections:
        - IT DOES OVER-CERTIFY. On the persistent component at phi=.999 it certifies rarely and
          COVERS 0.185, 95% CI [0.063, 0.381] on 27 certifications from a powered re-run of that
          single cell (5000 trials, `VR_LEAK_CELL`) — p = 2e-21 against the 0.93 bar, and
          5/35 = 0.143 [0.048, 0.303] pooled with the shipped cell's 0/8. Cycle 2's "does not
          over-certify anywhere" is WITHDRAWN.
        - "REFUSES 100%" IS TWO COMPONENTS. At phi=.99 the estimator returns NaN on 84.9% +- 1.1%
          of draws and a FINITE plateau on 15.1%; every one of those finite reads (151/151) is
          then rejected by `block_rule`'s MAX_BLOCK_CAP=60 ceiling, which would reject ANY
          estimator reading above 60 — tau included. See REFUSAL_CHANNELS.
        - THE ESTIMATOR'S OWN REFUSAL RATE IS A RULE-AND-GRID PROPERTY. `first_levelled_index`
          tests a FIXED 0.10 tolerance against a curve whose sampling sd reaches 0.316 at the top
          of the grid (3.2x the tolerance); it refuses 54.7% +- 1.6% of i.i.d. draws at n=3000,
          90.9% of those refusals are driven by the top two grid points, and when it does return
          on white noise the median q is exactly 1/4 of the largest grid point at every n — it is
          reading the grid, not the data. That refusal rate has NOT been separated from the
          variance-ratio statistic and this row does not claim it has. See
          RULE_IS_NOT_THE_STATISTIC.
    * GATE (iii) AS SPECIFIED CANNOT DISCRIMINATE BETWEEN ESTIMATORS AT ALL. Its containment
      numbers measure the SERIES (a population Sharpe that is not constant across the window), not
      the instrument. All cycle-1 containment-based verdicts are withdrawn, including
      retroactively against tau. The load-bearing finding is EXCESS DISPERSION, not a trend: see
      GATE_III_NONSTATIONARITY and DRIFT_REFIT, and cycle 2's "monotone drift" is struck.

  Reported as the row's primary result per its own explicit "report failure precisely rather than
  tuning until a number clears". NOTHING HAS BEEN TUNED IN ANY CYCLE: `plateau_tol` is unchanged
  at 0.10, `k_consec` unchanged at 5, `alpha` unchanged at 0.05, `TARGET_COVERAGE` unchanged at
  0.93, `BLOCK_STRATA` and `MAX_BLOCK_CAP` unchanged. Cycle 2 corrected two read-off DEFECTS and
  re-measured; cycle 3 corrected ATTRIBUTIONS and measured what it needed to correct them.
============================================================================================

WHAT CHANGED IN FIX CYCLE 3 (2026-08-24) — FOUR MIS-ATTRIBUTIONS, NO REVERSAL OF THE NEGATIVE.
An independent re-verification confirmed the top-line conclusion and `acf_band`/`spectral_zero`'s
13-44 sigma failures reproduce exactly. What it refuted were REASONS, which is what cycle 2 had
already got wrong once. Each is corrected below and each cost a measurement, all of them cheap,
none of them a re-run of the 63-cell sweep (`readoff_diagnostics.py`, `drift_check.py`,
`spread_check.py`, `calib_table.py`, `vr_leak_cell.py`; results in READOFF_DIAG.json,
DRIFT_CHECK.json, SPREAD_CHECK.json, CALIB_TABLE.json, VR_LEAK_CELL.json):

  MAJOR 1 — the corrected read-off rule is ITSELF an artifact at this grid, one level down from
  cycle 1's BLOCKER. `first_levelled_index` compares against a FIXED `plateau_tol=0.10` while the
  sampling sd of its input grows to 3.2x that tolerance at the top of the grid. This cycle took
  the DISCLOSE option, not the fix option: `plateau_tol` stays 0.10 and every place this module
  called `variance_ratio` "conservative" now says that the refusal rate is a property of the rule
  and the grid that has NOT been separated from the statistic. The separating counterfactual is
  recorded as evidence, not shipped. See RULE_IS_NOT_THE_STATISTIC for the numbers and for why
  disclosing beat fixing here.

  MAJOR 2 — "refuses" was one column hiding five branches. `coverage_harness.run_coverage_cell`
  increments a single `n_refused` for estimator-NaN, MAX_BLOCK_CAP rejection, the
  MIN_NONOVERLAP_BLOCKS floor, a failed width calibration, and — fifth — a bootstrap that never
  produced a usable CI. The fifth measured ZERO (0 of 228 per-chunk records across the nine
  real-leg runs), which is why a four-channel split reconciles against a five-branch counter. Those are facts about different
  components: on the family gate (ii) is about, the cap is doing most of the work that was being
  credited to the estimator. Split and re-measured on every cell (REFUSAL_CHANNELS), plus the
  calibration table the sweep consumed and then discarded (CALIBRATION_REFUSAL_CHANNEL).
  `median_unresolved_grid_q` was also a MISLABEL and is retracted.

  MAJOR 3 — "never over-certifies" was contradicted by this module's own cell. 0/8 against a 0.93
  bar is p = 5.8e-10 — conclusive in the FAIL direction — while POWER_AND_RESOLUTION's
  "n_certified < 20 is not readable as pass or fail at all" shielded it. That rule is correct for
  ESTABLISHING a pass and wrong for REFUTING one; it is now stated ONE-SIDED, and rather than
  argue the threshold the cell was re-run at 5x trials until it cleared the threshold on its own
  terms (VR_LEAK_CELL: 27 certifications, coverage 0.185 [0.063, 0.381]).

  MAJOR 4 — "monotone drift" is STRUCK everywhere. Refit of this row's own chunk Sharpes
  (DRIFT_REFIT): the H=1 slope is -0.00386 +- 0.00241, t = -1.61, p = 0.122, R^2 = 0.097, and it
  weakens with H (p = 0.273 / 0.660); at H=20 the second half is HIGHER than the first. The
  "last three chunks" illustration was a selected extract. THE LOAD-BEARING FINDING IS UNTOUCHED
  AND IS EXCESS DISPERSION: the pseudo-truth is not a constant (chi2 p = 1e-20 to 1e-33), and
  detrending removes essentially none of it (between/within variance ratio 7.19 -> 6.77 at H=1).
  Wording aligned with the downstream row `tasks/TASK_906_OOS_NONSTATIONARITY.md`.

WHAT CHANGED IN FIX CYCLE 2 (all three are corrections of DEFECTS, not tuning):

  BLOCKER 1 — `variance_ratio`'s read-off rule contradicted its own docstring. The shipped code
  returned at the first grid point whose ONE-STEP relative change was < 10%, while the docstring
  described "the smallest q at which the curve has plateaued". Those differ on any slowly-rising
  curve, DETERMINISTICALLY — not through sampling noise, which is what cycle 1's verdict blamed.
  The counter-example is the exact family the gate is about: the POPULATION VR of 0.2*AR1(.99) is
  VR(2)=1.03960, VR(4)=1.11801 (a 7.54% step, under tolerance -> "plateau at q=4") while the same
  population curve continues 1.272, 1.567, 2.113, 3.049, 4.442, 6.062 at q=256 toward an
  asymptote of 8.92; at phi=.999 it declares q=4 against an asymptote of 80.9. Fixed to the
  docstring's rule (levelled off against ALL larger grid points) and re-measured. Cycle 1's
  GATE_II_VERDICT text for this candidate ("sampling noise ... frequently mistaken for a genuine
  plateau") was factually wrong about the mechanism and is withdrawn.

  MAJOR 2 — `acf_band`'s Bonferroni band was applied in the FAIL-OPEN direction. Splitting
  alpha=0.05 over 240 lags gave a per-lag band of 3.71/sqrt(n) = 0.06771 at n=3000, against the
  standard 0.03578 and tau's own 2.0/sqrt(n) = 0.03651. A WIDER band declares "back to noise"
  SOONER -> shorter block -> narrower interval -> less coverage. The persistent component's
  population rho(1) = 0.04*0.99 = 0.03960 sits BETWEEN the standard band and the widened one, so
  the correction was erasing exactly the structure the gate is about. Fixed to the standard
  per-lag 1.96/sqrt(n) (see `range_probe._band_z`) and re-measured.
  THE MECHANISM SENTENCE THAT CLOSED THE DIRECTION IS ALSO WITHDRAWN. Cycle 1 wrote that this
  family's population correlation "is itself below the sampling-noise floor at every lag ... no
  stopping-rule fix helps". FALSE: rho(1)=0.0396 is 1.11x the 1.96/sqrt(3000) floor (2.2x the
  1/sqrt(n) floor). CYCLE 2's "~100 consecutive lags" HERE WAS WRONG AND IS CORRECTED (cycle 3):
  the population ACF is rho(h) = 0.04*phi^h, so at phi=.99 and n=3000 it stays above the
  1.96/sqrt(n) band for 11 lags (0.99^h > 0.03578/0.04 => h <= 11) and above the 1/sqrt(n) floor
  for 78; ~111 is the phi=.999 number against the 1.96 band. Prose only — the claim this sentence
  supports (that the structure is detectable at all, contra cycle 1) needs one lag, not a hundred,
  and is independently established by the measured readout below. Measured consequence: the median
  readout on that family moves from 1.0 (Bonferroni band) to 18.0 (standard band) — the
  first-crossing statistic DOES see the structure IN THE MEDIAN (per-draw spread is wide; see
  ACF_BAND_READOUT_SPREAD). THE INTERVAL STILL UNDER-COVERS ANYWAY (0.724
  +- 0.015), which is a different and weaker claim than "the signal is invisible": what fails is
  the CONVERSION of a correct-ish range read into a covering interval at block <= 60, not the
  detection. `acf_band` genuinely fails gate (i)/(ii) — for this reason, not the retracted one.

  MAJOR 3 — gate (iii) measured non-stationarity and cycle 1 attributed it to the estimators. See
  GATE_III_NONSTATIONARITY: the within-chunk bootstrap SE is FLAT in block length, the series is
  ANTI-persistent (VR falls in q), and the between-chunk Sharpe dispersion is 6.4-7.2x the
  within-chunk SE (chi2_25 = 196.5, p = 1.4e-28 at H=1). Containment cannot be raised by any
  block choice, and the paired test (PAIRED_DISCRIMINATION) shows estimators whose blocks differ
  by up to 19x contain the grand Sharpe on the SAME chunks, chunk for chunk, everywhere it was
  checked. (Cycle 2 counted that as "8 of 9 pairs"; with three estimators the third pairing of
  each horizon is IMPLIED by the other two, so the honest count is at most 3 horizons x 2
  independent comparisons — and the paired test is a COROLLARY of the flat-SE measurement rather
  than independent evidence. Corrected in PAIRED_DISCRIMINATION.)

  MAJOR 4 — gate (iii)'s H shortfall is CLOSED (H in {1,5,20} all run, not just H=1), and the
  universe description is corrected (see GATE_III_CONSTRUCTION): "the 7 majors" is four
  underlyings plus duplicate listings and a stablecoin FX pair, and `tau=0.9` binds before `k` in
  BOTH this config and 897's, so cycle 1's "same long_short shape at a comparable ratio" was
  wrong. Measured on every decision bar of all three horizons: present count 7/7 and eligible
  pool exactly 1 per side (`n_elig_hist={1: 217438}` at H=1).

WHAT WAS BUILT (`range_probe.py`) — three candidates, `estimate(x) -> (range_est, info)`, NaN on
refusal:

  (i)   `acf_band` — first lag at which `k_consec=5` CONSECUTIVE sample autocorrelations fall
        inside the STANDARD per-lag white-noise band 1.96/sqrt(n) (no family-wise widening — that
        is the fail-open direction here; `_band_z` states the argument). ASSUMES the dependence
        has a finite reach after which the ACF is flat. BREAKS on the 4%-variance persistent
        component: it now READS the structure (MEDIAN range 18 at phi=.99, vs 1.0 under the old
        band — but per-draw p10 = 3, p90 = 62 over 1000 draws, so a SINGLE draw's readout is
        nearly uninformative; see ACF_BAND_READOUT_SPREAD) while the k-inflated interval at the
        resulting block still covers only 0.724 +- 0.015; at phi=.999 the readout collapses to 4.0
        (the near-random-walk component is largely removed by demeaning the finite window) and
        coverage is 0.262 +- 0.015.
  (ii)  `variance_ratio` — Lo-MacKinlay overlapping VR(q) over a geometric grid; range = smallest
        q whose VR is within a FIXED 10% of VR at EVERY LARGER grid point; NaN when no q
        qualifies. ASSUMES VR(q) is eventually monotone-then-flat (the real series is
        anti-persistent, i.e. outside that assumption — disclosed, not closed). WHAT IT DOES,
        STATED IN THE SEPARATED FORM CYCLE 3 MEASURED: on the persistent component at phi=.99 the
        ESTIMATOR returns NaN on 84.9% of draws and the BLOCK CEILING rejects the remaining 15.1%
        (151/151 of the finite reads), so the "100% refusal" is two components, one of which is
        not the estimator; the estimator's own NaN rate is a property of the fixed tolerance at
        this grid (54.7% on i.i.d. at n=3000, where the population VR is 1 everywhere) and has NOT
        been separated from the variance-ratio statistic; and where it does certify it OVER-
        CERTIFIES — persistent phi=.999 covers 0.185 [0.063, 0.381] on 27 certifications. See
        REFUSAL_CHANNELS, RULE_IS_NOT_THE_STATISTIC, VR_LEAK_CELL.
  (iii) `spectral_zero` — Andrews (1991) AR(1) plug-in bandwidth, read as a range. ASSUMES the
        AR(1) plug-in stands in for the spectral shape. BREAKS on the persistent component because
        the plug-in is a monotone function of rho1 ALONE: rho1 = 0.0396 there, so the bandwidth
        stays small (median 2.92 / 2.09) and it certifies 71-72% of draws at coverage 0.63 / 0.26.

`block_rule(n_dec, range_est)` snaps the range UP (never down) to the nearest of `BLOCK_STRATA =
(2,3,5,8,12,18,26,38,60)`, REFUSES OUTRIGHT if the raw estimate exceeds `MAX_BLOCK_CAP=60`
(D55's certified-ceiling pattern — refuse above the measured range rather than extrapolate —
applied directly, not only cited), checks a `MIN_NONOVERLAP_BLOCKS=10` floor, and calibrates a
width-inflation factor `k` via the SAME closed-form method `experiments/897_corrected_809/
calibration.py::k_needed_for_trial` uses (imported, never re-derived), against the triangular
family at `dep == block` (897's own matched-dependence convention — correct in-family,
conditional out of it, which is exactly what the sweep below measures rather than assumes).

============================================================================================
MEASURED_COVERAGE_TABLE (gate i) — 1000 trials/cell (up from cycle 1's 200; see POWER_AND_
RESOLUTION), n_boot_ci=300; calibration at 200+200 trials/n_boot=300 per (block, n_dec) stratum,
45 strata pre-warmed once and cached. Every coverage number carries se = sqrt(p*(1-p)/n_certified)
— a bare point estimate is never reported alone. COVERAGE IS CONDITIONAL ON CERTIFICATION: the
denominator is `n_certified`, and `certify_rate` is printed beside it because a low certify_rate
makes a high coverage a statement about a self-selected subset, not about the instrument. "None" =
no trial was certified (nothing scoreable — refusal reported, not hidden). Source: `RESULTS.json`
(gpu-server synthetic sweep `RESULTS_srv.json` + workstation real leg `REAL_LEG.json`, merged by
`combine_results.py`).
READ `certify_rate` AGAINST `REFUSAL_CHANNELS`, NOT AS "THE ESTIMATOR REFUSED" (fix cycle 3):
1 - certify_rate merges estimator-NaN, the MAX_BLOCK_CAP=60 ceiling rejecting a FINITE read, the
MIN_NONOVERLAP_BLOCKS floor (never binding here), and one failing width-calibration stratum. Some
of the 0.000 rows below are 100% CEILING and 0% estimator (e.g. every spectral_zero triangular
cell at dep >= 10 / n=3000). `median_range_est` is likewise the median DECLARED range over the
draws that returned a number — for `variance_ratio` it is a declared plateau, never a grid ceiling.
============================================================================================
"""
from __future__ import annotations

MEASURED_COVERAGE_TABLE = {
    # (estimator, family, n_dec): {"coverage", "se", "n_certified", "n_trials", "certify_rate",
    # "median_range_est"} — rendered from RESULTS.json, not hand-transcribed. `median_range_est`
    # is the median over trials whose estimator returned a number (refusals excluded).
# n_trials/cell = 1000, n_boot_ci = 300
    ("acf_band", "iid", 200): {"coverage": 0.9880, "se": 0.0034, "n_certified": 1000, "n_trials": 1000, "certify_rate": 1.000, "median_range_est": 1.00},
    ("acf_band", "iid", 1000): {"coverage": 0.9840, "se": 0.0040, "n_certified": 1000, "n_trials": 1000, "certify_rate": 1.000, "median_range_est": 1.00},
    ("acf_band", "iid", 3000): {"coverage": 0.9864, "se": 0.0037, "n_certified": 959, "n_trials": 1000, "certify_rate": 0.959, "median_range_est": 1.00},
    ("acf_band", "triangular_dep4", 1000): {"coverage": 0.9670, "se": 0.0056, "n_certified": 1000, "n_trials": 1000, "certify_rate": 1.000, "median_range_est": 5.00},
    ("acf_band", "triangular_dep4", 3000): {"coverage": 0.9730, "se": 0.0051, "n_certified": 1000, "n_trials": 1000, "certify_rate": 1.000, "median_range_est": 4.00},
    ("acf_band", "triangular_dep10", 1000): {"coverage": 0.9819, "se": 0.0042, "n_certified": 997, "n_trials": 1000, "certify_rate": 0.997, "median_range_est": 13.00},
    ("acf_band", "triangular_dep10", 3000): {"coverage": 0.9779, "se": 0.0047, "n_certified": 995, "n_trials": 1000, "certify_rate": 0.995, "median_range_est": 14.00},
    ("acf_band", "triangular_dep15", 1000): {"coverage": 0.9713, "se": 0.0053, "n_certified": 975, "n_trials": 1000, "certify_rate": 0.975, "median_range_est": 20.00},
    ("acf_band", "triangular_dep15", 3000): {"coverage": 0.9761, "se": 0.0049, "n_certified": 962, "n_trials": 1000, "certify_rate": 0.962, "median_range_est": 20.00},
    ("acf_band", "triangular_dep22", 1000): {"coverage": 0.9754, "se": 0.0052, "n_certified": 896, "n_trials": 1000, "certify_rate": 0.896, "median_range_est": 31.00},
    ("acf_band", "triangular_dep22", 3000): {"coverage": 0.9781, "se": 0.0050, "n_certified": 869, "n_trials": 1000, "certify_rate": 0.869, "median_range_est": 32.00},
    ("acf_band", "triangular_dep32", 1000): {"coverage": 0.9727, "se": 0.0062, "n_certified": 695, "n_trials": 1000, "certify_rate": 0.695, "median_range_est": 45.00},
    ("acf_band", "triangular_dep32", 3000): {"coverage": 0.9590, "se": 0.0077, "n_certified": 658, "n_trials": 1000, "certify_rate": 0.658, "median_range_est": 48.00},
    ("acf_band", "triangular_dep50", 1000): {"coverage": 0.9772, "se": 0.0056, "n_certified": 703, "n_trials": 1000, "certify_rate": 0.703, "median_range_est": 46.00},
    ("acf_band", "triangular_dep50", 3000): {"coverage": 0.9850, "se": 0.0066, "n_certified": 334, "n_trials": 1000, "certify_rate": 0.334, "median_range_est": 77.00},
    ("acf_band", "ar1_phi0.9", 3000): {"coverage": 0.9557, "se": 0.0068, "n_certified": 903, "n_trials": 1000, "certify_rate": 0.903, "median_range_est": 32.00},
    ("acf_band", "ar1_phi0.99", 3000): {"coverage": None, "se": None, "n_certified": 0, "n_trials": 1000, "certify_rate": 0.000, "median_range_est": 160.00},
    ("acf_band", "ar1_phi0.999", 3000): {"coverage": None, "se": None, "n_certified": 0, "n_trials": 1000, "certify_rate": 0.000, "median_range_est": 215.00},
    ("acf_band", "persistent_component_phi0.99", 3000): {"coverage": 0.7243, "se": 0.0154, "n_certified": 845, "n_trials": 1000, "certify_rate": 0.845, "median_range_est": 18.00},
    ("acf_band", "persistent_component_phi0.999", 3000): {"coverage": 0.2625, "se": 0.0153, "n_certified": 823, "n_trials": 1000, "certify_rate": 0.823, "median_range_est": 4.00},
    ("acf_band", "seasonal_regime", 3000): {"coverage": 0.9652, "se": 0.0059, "n_certified": 976, "n_trials": 1000, "certify_rate": 0.976, "median_range_est": 24.00},
    ("variance_ratio", "iid", 200): {"coverage": 0.9954, "se": 0.0032, "n_certified": 436, "n_trials": 1000, "certify_rate": 0.436, "median_range_est": 4.00},
    ("variance_ratio", "iid", 1000): {"coverage": 0.9815, "se": 0.0058, "n_certified": 540, "n_trials": 1000, "certify_rate": 0.540, "median_range_est": 16.00},
    ("variance_ratio", "iid", 3000): {"coverage": 0.9897, "se": 0.0073, "n_certified": 194, "n_trials": 1000, "certify_rate": 0.194, "median_range_est": 64.00},
    ("variance_ratio", "triangular_dep4", 1000): {"coverage": 0.9796, "se": 0.0061, "n_certified": 539, "n_trials": 1000, "certify_rate": 0.539, "median_range_est": 16.00},
    ("variance_ratio", "triangular_dep4", 3000): {"coverage": 0.9828, "se": 0.0099, "n_certified": 174, "n_trials": 1000, "certify_rate": 0.174, "median_range_est": 64.00},
    ("variance_ratio", "triangular_dep10", 1000): {"coverage": 0.9862, "se": 0.0052, "n_certified": 509, "n_trials": 1000, "certify_rate": 0.509, "median_range_est": 32.00},
    ("variance_ratio", "triangular_dep10", 3000): {"coverage": 0.9758, "se": 0.0120, "n_certified": 165, "n_trials": 1000, "certify_rate": 0.165, "median_range_est": 64.00},
    ("variance_ratio", "triangular_dep15", 1000): {"coverage": 0.9836, "se": 0.0057, "n_certified": 489, "n_trials": 1000, "certify_rate": 0.489, "median_range_est": 32.00},
    ("variance_ratio", "triangular_dep15", 3000): {"coverage": 0.9708, "se": 0.0144, "n_certified": 137, "n_trials": 1000, "certify_rate": 0.137, "median_range_est": 64.00},
    ("variance_ratio", "triangular_dep22", 1000): {"coverage": 0.9797, "se": 0.0071, "n_certified": 394, "n_trials": 1000, "certify_rate": 0.394, "median_range_est": 32.00},
    ("variance_ratio", "triangular_dep22", 3000): {"coverage": 0.9706, "se": 0.0205, "n_certified": 68, "n_trials": 1000, "certify_rate": 0.068, "median_range_est": 64.00},
    ("variance_ratio", "triangular_dep32", 1000): {"coverage": 0.9781, "se": 0.0097, "n_certified": 228, "n_trials": 1000, "certify_rate": 0.228, "median_range_est": 32.00},
    ("variance_ratio", "triangular_dep32", 3000): {"coverage": 0.8667, "se": 0.0878, "n_certified": 15, "n_trials": 1000, "certify_rate": 0.015, "median_range_est": 128.00},
    ("variance_ratio", "triangular_dep50", 1000): {"coverage": 0.8824, "se": 0.0781, "n_certified": 17, "n_trials": 1000, "certify_rate": 0.017, "median_range_est": 32.00},
    ("variance_ratio", "triangular_dep50", 3000): {"coverage": None, "se": None, "n_certified": 0, "n_trials": 1000, "certify_rate": 0.000, "median_range_est": 128.00},
    ("variance_ratio", "ar1_phi0.9", 3000): {"coverage": 0.9048, "se": 0.0453, "n_certified": 42, "n_trials": 1000, "certify_rate": 0.042, "median_range_est": 64.00},
    ("variance_ratio", "ar1_phi0.99", 3000): {"coverage": None, "se": None, "n_certified": 0, "n_trials": 1000, "certify_rate": 0.000, "median_range_est": 128.00},
    ("variance_ratio", "ar1_phi0.999", 3000): {"coverage": None, "se": None, "n_certified": 0, "n_trials": 1000, "certify_rate": 0.000, "median_range_est": None},
    ("variance_ratio", "persistent_component_phi0.99", 3000): {"coverage": None, "se": None, "n_certified": 0, "n_trials": 1000, "certify_rate": 0.000, "median_range_est": 128.00},
    ("variance_ratio", "persistent_component_phi0.999", 3000): {"coverage": 0.0000, "se": 0.0000, "n_certified": 8, "n_trials": 1000, "certify_rate": 0.008, "median_range_est": 128.00},
    ("variance_ratio", "seasonal_regime", 3000): {"coverage": 0.9125, "se": 0.0316, "n_certified": 80, "n_trials": 1000, "certify_rate": 0.080, "median_range_est": 64.00},
    ("spectral_zero", "iid", 200): {"coverage": 0.9860, "se": 0.0037, "n_certified": 1000, "n_trials": 1000, "certify_rate": 1.000, "median_range_est": 1.39},
    ("spectral_zero", "iid", 1000): {"coverage": 0.9870, "se": 0.0036, "n_certified": 1000, "n_trials": 1000, "certify_rate": 1.000, "median_range_est": 1.38},
    ("spectral_zero", "iid", 3000): {"coverage": 0.9859, "se": 0.0042, "n_certified": 780, "n_trials": 1000, "certify_rate": 0.780, "median_range_est": 1.35},
    ("spectral_zero", "triangular_dep4", 1000): {"coverage": 0.9870, "se": 0.0036, "n_certified": 1000, "n_trials": 1000, "certify_rate": 1.000, "median_range_est": 25.85},
    ("spectral_zero", "triangular_dep4", 3000): {"coverage": 0.9850, "se": 0.0038, "n_certified": 1000, "n_trials": 1000, "certify_rate": 1.000, "median_range_est": 37.47},
    ("spectral_zero", "triangular_dep10", 1000): {"coverage": 0.9940, "se": 0.0025, "n_certified": 992, "n_trials": 1000, "certify_rate": 0.992, "median_range_est": 50.58},
    ("spectral_zero", "triangular_dep10", 3000): {"coverage": None, "se": None, "n_certified": 0, "n_trials": 1000, "certify_rate": 0.000, "median_range_est": 73.49},
    ("spectral_zero", "triangular_dep15", 1000): {"coverage": 0.9778, "se": 0.0127, "n_certified": 135, "n_trials": 1000, "certify_rate": 0.135, "median_range_est": 66.48},
    ("spectral_zero", "triangular_dep15", 3000): {"coverage": None, "se": None, "n_certified": 0, "n_trials": 1000, "certify_rate": 0.000, "median_range_est": 97.48},
    ("spectral_zero", "triangular_dep22", 1000): {"coverage": 1.0000, "se": 0.0000, "n_certified": 2, "n_trials": 1000, "certify_rate": 0.002, "median_range_est": 85.74},
    ("spectral_zero", "triangular_dep22", 3000): {"coverage": None, "se": None, "n_certified": 0, "n_trials": 1000, "certify_rate": 0.000, "median_range_est": 126.42},
    ("spectral_zero", "triangular_dep32", 1000): {"coverage": None, "se": None, "n_certified": 0, "n_trials": 1000, "certify_rate": 0.000, "median_range_est": 108.90},
    ("spectral_zero", "triangular_dep32", 3000): {"coverage": None, "se": None, "n_certified": 0, "n_trials": 1000, "certify_rate": 0.000, "median_range_est": 162.67},
    ("spectral_zero", "triangular_dep50", 1000): {"coverage": None, "se": None, "n_certified": 0, "n_trials": 1000, "certify_rate": 0.000, "median_range_est": 144.97},
    ("spectral_zero", "triangular_dep50", 3000): {"coverage": None, "se": None, "n_certified": 0, "n_trials": 1000, "certify_rate": 0.000, "median_range_est": 219.11},
    ("spectral_zero", "ar1_phi0.9", 3000): {"coverage": None, "se": None, "n_certified": 0, "n_trials": 1000, "certify_rate": 0.000, "median_range_est": 73.48},
    ("spectral_zero", "ar1_phi0.99", 3000): {"coverage": None, "se": None, "n_certified": 0, "n_trials": 1000, "certify_rate": 0.000, "median_range_est": 222.56},
    ("spectral_zero", "ar1_phi0.999", 3000): {"coverage": None, "se": None, "n_certified": 0, "n_trials": 1000, "certify_rate": 0.000, "median_range_est": 222.56},
    ("spectral_zero", "persistent_component_phi0.99", 3000): {"coverage": 0.6261, "se": 0.0181, "n_certified": 714, "n_trials": 1000, "certify_rate": 0.714, "median_range_est": 2.92},
    ("spectral_zero", "persistent_component_phi0.999", 3000): {"coverage": 0.2598, "se": 0.0164, "n_certified": 716, "n_trials": 1000, "certify_rate": 0.716, "median_range_est": 2.09},
    ("spectral_zero", "seasonal_regime", 3000): {"coverage": 0.9650, "se": 0.0058, "n_certified": 1000, "n_trials": 1000, "certify_rate": 1.000, "median_range_est": 19.90},
}

# ---------------------------------------------------------------------------------------------
# GATE (ii) — the 4%-variance persistent component (tau's killer). The gate asks for: reads
# range >> 1 AND (calibrates correctly OR refuses). Re-measured in fix cycle 2 with BOTH read-off
# defects corrected.
# THE CYCLE-2 SPLIT WRITTEN HERE ("two over-certify, one refuses") IS WITHDRAWN (fix cycle 3).
# ALL THREE OVER-CERTIFY ON THIS FAMILY. `variance_ratio` declines on the overwhelming majority
# of draws, but where it does certify at phi=.999 it covers 0.185 [0.063, 0.381] on 27
# certifications, p = 2e-21 against the bar (VR_LEAK_CELL) -- so "one refuses, cleanly" is not
# what was measured. What separates it from the other two is HOW OFTEN it leaks, not WHETHER.
# Read GATE_II_VERDICT["variance_ratio"] and VR_LEAK_CELL before this header's word "refuses",
# and note that part of that refusal is `block_rule`'s ceiling rather than the estimator
# (REFUSAL_CHANNELS).
# Compare to tau's OWN measured failure (`experiments/897_corrected_809/calibration.py::
# MEASURED_TAU_OUT_OF_FAMILY_HALTED_CYCLE`): certify_rate=1.0, coverage 0.805 (phi=.99) / 0.376
# (phi=.999) — tau reads range 1 and certifies everything.
# ---------------------------------------------------------------------------------------------
GATE_II_VERDICT = {
    "acf_band": {
        "phi_0.99": {"median_range": 18.0, "range_p10": 3.0, "range_p90": 62.0,
                     "certify_rate": 0.845, "coverage": 0.7243,
                     "coverage_se": 0.0154, "n_certified": 845,
                     "estimator_nan_rate": 0.000, "cap_refusal_rate": 0.106,
                     "calibration_refusal_rate": 0.029},
        "phi_0.999": {"median_range": 4.0, "certify_rate": 0.823, "coverage": 0.2625,
                      "coverage_se": 0.0153, "n_certified": 823,
                      "estimator_nan_rate": 0.038, "cap_refusal_rate": 0.067,
                      "calibration_refusal_rate": 0.052},
        "passes": False,
        "reason": "READS the structure IN THE MEDIAN (median range 18.0 at phi=.99, up from 1.0 "
                  "under cycle 1's fail-open Bonferroni band; per-draw p10 = 3.0 and p90 = 62.0 "
                  "over 1000 draws, so 'it sees the structure' is true of the median and not of a "
                  "single read -- ACF_BAND_READOUT_SPREAD) but it neither calibrates correctly "
                  "nor refuses: it "
                  "certifies 84.5% of draws at 0.724 +- 0.015 coverage, 13.4 sigma below the 0.93 "
                  "bar. At phi=.999 the readout falls back to 4.0 -- demeaning a finite window "
                  "removes most of a near-random-walk component -- and coverage is 0.262 +- "
                  "0.015. The failure is the RANGE -> INTERVAL conversion at block <= 60, NOT (as "
                  "cycle 1 claimed) that the correlation is invisible at every lag.",
        "cycle1_claim_withdrawn": "'median range_est=1.0 at both phi -- does NOT read range >> 1' "
                                  "was an artifact of the fail-open band, not of the statistic.",
    },
    "variance_ratio": {
        # `median_declared_plateau_q` = median over the draws whose ESTIMATOR RETURNED A NUMBER of
        # the q it declared as the plateau (RESULTS.json's `median_range_est`). Cycle 2 shipped
        # this field as "median_unresolved_grid_q", which was a MISLABEL and is retracted: the
        # grid ceiling at n=3000 is deterministically 256 (grid 2,4,...,256 from max_q = n//10 =
        # 300), never 128, and the same dict reported "median_range": None for a cell whose median
        # DECLARED range is 128. Both fields are now named for what they are.
        "phi_0.99": {"grid_ceiling_q": 256, "median_declared_plateau_q": 128.0,
                     "certify_rate": 0.000, "coverage": None, "n_certified": 0,
                     "estimator_nan_rate": 0.849, "estimator_nan_rate_se": 0.0113,
                     "cap_refusal_rate": 0.151, "cap_refusal_rate_se": 0.0113},
        "phi_0.999": {"grid_ceiling_q": 256, "median_declared_plateau_q": 128.0,
                      "certify_rate": 0.008, "coverage": 0.0, "coverage_se": 0.0,
                      "n_certified": 8, "estimator_nan_rate": 0.943,
                      "cap_refusal_rate": 0.049,
                      "powered_rerun": "5000 trials, seed 901: 27 certified, coverage 0.185, "
                                       "95% CI [0.063, 0.381], p = 2.2e-21 vs the 0.93 bar "
                                       "(VR_LEAK_CELL)"},
        "passes": "QUALIFIED — see passes_qualified; NOT a clean pass and NOT 'never "
                  "over-certifies'",
        "passes_qualified": "TAKES THE GATE'S REFUSE BRANCH AT phi=.99, BUT NOT AS ONE COMPONENT "
                            "AND NOT AIRTIGHT AT phi=.999. Separated (REFUSAL_CHANNELS, 1000 "
                            "fresh draws): at phi=.99 the ESTIMATOR returns NaN on 84.9% +- 1.1% "
                            "of draws and a finite plateau on 15.1%, and `block_rule`'s "
                            "MAX_BLOCK_CAP=60 rejects every one of those 151 finite reads (essentially all, not "
                            "structurally all -- a fresh seed gave 145 of 146; see "
                            "REFUSAL_CHANNELS_READING). So "
                            "gate (ii)'s 'pass' here is attributable in part to the BLOCK "
                            "CEILING, which is not this estimator and would refuse ANY estimator "
                            "reading above 60 -- tau included, had tau ever read above 60. "
                            "At phi=.999 the leak is real and, powered up, CONCLUSIVE AGAINST the "
                            "candidate: 27 certifications cover 0.185 [0.063, 0.381], p = 2e-21 "
                            "against the bar. Finally, the estimator's own NaN rate is a property "
                            "of a fixed 0.10 tolerance at a grid whose sampling sd reaches 0.316 "
                            "(RULE_IS_NOT_THE_STATISTIC) -- it has NOT been separated from the "
                            "variance-ratio statistic, so nothing here licenses calling the "
                            "STATISTIC conservative.",
        "cycle1_claim_withdrawn": "'median range_est=4.0 -- modal outcome is a FALSE small "
                                  "plateau ... sampling noise mistaken for a genuine plateau' was "
                                  "produced by a one-step read-off rule that trips DETERMINISTIC"
                                  "ALLY on the population curve (see the BLOCKER-1 note in this "
                                  "module's docstring). The candidate was condemned by its "
                                  "read-off rule, not by the statistic.",
        "cycle2_claims_withdrawn": [
            "'does not over-certify anywhere' -- contradicted by this module's own phi=.999 cell "
            "(0/8, p = 5.8e-10) and by the powered re-run (5/27, p = 2e-21).",
            "'median_unresolved_grid_q = 128' -- mislabel; 128 is the median DECLARED plateau, "
            "the grid ceiling is 256.",
            "'refuses 100% of draws' read as a statement about the estimator -- 15.1% of that is "
            "the block ceiling rejecting a finite read.",
        ],
    },
    "spectral_zero": {
        # NOTE the refusal split here: spectral_zero NEVER returns NaN on this family and is never
        # capped -- its ~29% non-certification is ENTIRELY the block=3/n_dec=3000 calibration
        # stratum (CALIBRATION_REFUSAL_CHANNEL), i.e. not the estimator declining at all.
        "phi_0.99": {"median_range": 2.92, "certify_rate": 0.714, "coverage": 0.6261,
                     "coverage_se": 0.0181, "n_certified": 714,
                     "estimator_nan_rate": 0.000, "cap_refusal_rate": 0.000,
                     "calibration_refusal_rate": 0.315},
        "phi_0.999": {"median_range": 2.09, "certify_rate": 0.716, "coverage": 0.2598,
                      "coverage_se": 0.0164, "n_certified": 716,
                      "estimator_nan_rate": 0.000, "cap_refusal_rate": 0.000,
                      "calibration_refusal_rate": 0.317},
        "passes": False,
        "reason": "The Andrews plug-in bandwidth is a monotone function of rho1 ALONE (times "
                  "n^(1/3)); this family's population rho1 is 0.0396, so the bandwidth stays "
                  "small and the candidate certifies ~71% of draws at 0.63 / 0.26 coverage. "
                  "Cycle 1's docstring rationale for this candidate -- that the plug-in reads a "
                  "near-unit-root series as needing a LARGE bandwidth 'even though rho1 may be "
                  "small' -- is impossible and is corrected in `range_probe.range_spectral_zero`.",
    },
}

# ---------------------------------------------------------------------------------------------
# FIX CYCLE 3, MAJOR 1 — THE READ-OFF RULE IS ITSELF AN ARTIFACT AT THIS GRID, AND THIS ROW
# DISCLOSES THAT RATHER THAN FIXING IT.
# Cycle 1 blamed `variance_ratio`'s behaviour on sampling noise when the cause was a deterministic
# read-off defect (BLOCKER 1). Cycle 2 fixed the rule and then blamed the STATISTIC for being
# "unusably conservative". That is the same mis-attribution one level down: the corrected rule
# tests a FIXED relative tolerance against a curve whose sampling error is not fixed.
# Source: `readoff_diagnostics.rule_vs_statistic`, i.i.d. draws (population VR = 1 at every q, so
# every bit of the dispersion below is sampling noise), 1000 draws per n_dec, seed 901.
# ---------------------------------------------------------------------------------------------
RULE_IS_NOT_THE_STATISTIC = {
    "plateau_tol": 0.10,
    "sd_of_VR_q_iid_n3000": {2: 0.0182, 4: 0.0345, 8: 0.0543, 16: 0.0819, 32: 0.1177,
                             64: 0.1661, 128: 0.2336, 256: 0.3157},
    "sd_se_n3000": {2: 0.0004, 4: 0.0008, 8: 0.0012, 16: 0.0018, 32: 0.0026, 64: 0.0037,
                    128: 0.0052, 256: 0.0071},
    "max_sd_over_tolerance": 3.16,
    "fixed_rule_nan_rate_iid": {200: 0.540, 1000: 0.462, 3000: 0.547},
    "fixed_rule_nan_rate_iid_se": {200: 0.0158, 1000: 0.0158, 3000: 0.0157},
    "refusal_driven_by_top_two_grid_points": {200: 0.974, 1000: 0.922, 3000: 0.909},
    "median_returned_q_iid": {200: 4.0, 1000: 16.0, 3000: 64.0},
    "largest_grid_point": {200: 16, 1000: 64, 3000: 256},
    "correct_q2_among_finite_returns_iid": {200: "141/460", 1000: "134/538", 3000: "82/453"},
    "counterfactual_se_aware_tol": {
        "definition": "0.10*|v_i| + 2*sd_iid(q_j) when comparing grid point j to i; NOTHING else "
                      "changed, same draws, same grid, same code path.",
        "plateau_found_rate": {200: 1.000, 1000: 1.000, 3000: 1.000},
        "correct_q2_rate": {200: 0.983, 1000: 0.982, 3000: 0.967},
        "correct_q2_rate_se": {200: 0.0041, 1000: 0.0042, 3000: 0.0056},
    },
    "reading": (
        "THE 46-55% i.i.d. REFUSAL IS A PROPERTY OF THIS RULE AT THIS GRID, NOT A MEASURED "
        "PROPERTY OF THE VARIANCE-RATIO STATISTIC. Three facts, all on i.i.d. data where the "
        "population VR is exactly 1 at every q and the correct answer is q=2: (1) the sampling sd "
        "of the rule's input reaches 3.16x the fixed tolerance it is tested against; (2) 91% of "
        "the refusals at n=3000 are driven by the top two grid points, i.e. by the noisiest "
        "points on the curve; (3) when the rule does return, the median q is 4 / 16 / 64 at "
        "n = 200 / 1000 / 3000 -- exactly one quarter of the largest grid point every time, and "
        "only 82 of 453 finite returns at n=3000 are the correct q=2. It is reading the grid. "
        "Holding everything else fixed and making the tolerance aware of the compared point's own "
        "sampling error finds a plateau on 100% of the SAME draws and gets q=2 right on 96.7% "
        "+- 0.6%. "
        "WHY THIS ROW DISCLOSES RATHER THAN FIXES: an SE-aware tolerance is a DIFFERENT "
        "INSTRUMENT. Its refusal behaviour, and therefore every coverage number conditional on "
        "certification, would have to be re-certified out-of-family across all 21 cells -- the "
        "exact 63,000-trial sweep this fix cycle is forbidden to re-run, and the exact "
        "'calibrated inside one family, shipped as general' failure this whole row exists to "
        "prevent. Adopting it here would also be a parameter change made after seeing that it "
        "improves a number, which is what this row has refused to do in every cycle. So "
        "`plateau_tol` stays 0.10, the counterfactual above is recorded as EVIDENCE, and every "
        "statement about `variance_ratio`'s refusal rate in this module is scoped to the rule and "
        "the grid. WHAT THIS DOES NOT CHANGE: the top-line negative. It removes 'unusably "
        "conservative' as a claim about the statistic; it does not make `variance_ratio` a "
        "certified instrument, and the over-certification measured in VR_LEAK_CELL runs the other "
        "way entirely."
    ),
}

# ---------------------------------------------------------------------------------------------
# FIX CYCLE 3, MAJOR 2 — "REFUSES" WAS ONE COLUMN HIDING FIVE BRANCHES.
# `coverage_harness.run_coverage_cell` increments a single `n_refused` at five distinct points.
# FOUR are re-measured below. THE FIFTH is a bootstrap that returned a non-finite Sharpe or CI
# bound (`coverage_harness.py`'s isfinite guard); it has no counterpart here because this leg
# draws no bootstrap, and it MEASURED ZERO -- 0 of the 228 per-chunk records across the nine
# real-leg runs is `unusable_bootstrap`. That zero is what lets a four-way split reconcile
# against a five-branch counter; it is named rather than quietly dropped.
# Split here, measured over 1000 FRESH draws per cell (seed 901, `readoff_diagnostics.
# refusal_channels`) through the real `block_rule` code path, no bootstrap drawn. The full 63-cell
# split is in READOFF_DIAG.json; the cells below are the ones this module's prose leans on.
# CONSISTENCY CHECK: `certifiable` here vs `certify_rate` in MEASURED_COVERAGE_TABLE (independent
# seeds) agree to within Monte Carlo error on all 63 cells -- e.g. 0.204 vs 0.194, 0.865 vs 0.845,
# 0.008 vs 0.008, 0.685 vs 0.714.
# CHANNELS: estimator_nan = the estimator itself returned NaN. cap = it returned a FINITE range
# above MAX_BLOCK_CAP=60 and `block_rule` refused (a property of the CEILING; it would refuse any
# estimator, tau included). min_blocks = fewer than MIN_NONOVERLAP_BLOCKS=10 non-overlapping
# blocks (measured 0.000 on every cell -- never binding at these n_dec). calib = the width-factor
# calibration for that (block, n_dec) did not achieve target (see CALIBRATION_REFUSAL_CHANNEL).
# ---------------------------------------------------------------------------------------------
REFUSAL_CHANNELS = {
    # (estimator, family, n_dec): (certifiable, estimator_nan, cap, min_blocks, calib)
    ("variance_ratio", "iid", 3000): (0.204, 0.547, 0.249, 0.000, 0.000),
    ("variance_ratio", "iid", 1000): (0.538, 0.462, 0.000, 0.000, 0.000),
    ("variance_ratio", "iid", 200): (0.460, 0.540, 0.000, 0.000, 0.000),
    ("variance_ratio", "persistent_component_phi0.99", 3000): (0.000, 0.849, 0.151, 0.000, 0.000),
    ("variance_ratio", "persistent_component_phi0.999", 3000): (0.008, 0.943, 0.049, 0.000, 0.000),
    ("variance_ratio", "ar1_phi0.9", 3000): (0.036, 0.561, 0.403, 0.000, 0.000),
    ("variance_ratio", "ar1_phi0.99", 3000): (0.000, 0.922, 0.078, 0.000, 0.000),
    ("variance_ratio", "ar1_phi0.999", 3000): (0.000, 0.998, 0.002, 0.000, 0.000),
    ("variance_ratio", "triangular_dep50", 3000): (0.000, 0.595, 0.405, 0.000, 0.000),
    ("variance_ratio", "seasonal_regime", 3000): (0.066, 0.506, 0.428, 0.000, 0.000),
    ("acf_band", "iid", 3000): (0.958, 0.000, 0.000, 0.000, 0.042),
    ("acf_band", "ar1_phi0.99", 3000): (0.000, 0.389, 0.611, 0.000, 0.000),
    ("acf_band", "ar1_phi0.999", 3000): (0.000, 0.958, 0.042, 0.000, 0.000),
    ("acf_band", "persistent_component_phi0.99", 3000): (0.865, 0.000, 0.106, 0.000, 0.029),
    ("acf_band", "persistent_component_phi0.999", 3000): (0.843, 0.038, 0.067, 0.000, 0.052),
    ("acf_band", "triangular_dep50", 3000): (0.361, 0.025, 0.614, 0.000, 0.000),
    ("spectral_zero", "iid", 3000): (0.782, 0.000, 0.000, 0.000, 0.218),
    ("spectral_zero", "triangular_dep10", 3000): (0.000, 0.000, 1.000, 0.000, 0.000),
    ("spectral_zero", "triangular_dep50", 3000): (0.000, 0.000, 1.000, 0.000, 0.000),
    ("spectral_zero", "ar1_phi0.99", 3000): (0.000, 0.000, 1.000, 0.000, 0.000),
    ("spectral_zero", "persistent_component_phi0.99", 3000): (0.685, 0.000, 0.000, 0.000, 0.315),
    ("spectral_zero", "persistent_component_phi0.999", 3000): (0.683, 0.000, 0.000, 0.000, 0.317),
}
REFUSAL_CHANNELS_SE = ("binomial over 1000 draws: <= 0.0158 on every rate above (max at p=0.5); "
                       "0.0113 at p=0.15, 0.0064 at p=0.045.")
REFUSAL_CHANNELS_READING = (
    "WHAT THE SPLIT CHANGES. (1) GATE (ii)'s 'PASS' IS PART CEILING. `variance_ratio` on the "
    "persistent component at phi=.99 does not refuse as one thing: the estimator NaNs on 84.9% "
    "and MAX_BLOCK_CAP=60 rejects 100% of the 151 finite reads -- A SAMPLE FACT, NOT A "
    "STRUCTURAL ONE: on a fresh seed the cycle-3 re-verification drew 145 of 146 capped, i.e. "
    "one read slipped under the ceiling and certified (leak rate <~0.6% of draws), so read it "
    "as 'essentially all' with the denominator attached. The ceiling is not an estimator "
    "property -- it refuses ANY reading above 60, tau included -- so gate (ii)'s allowed 'reads "
    "range >> 1 and refuses' outcome is only partly earned by the candidate. "
    "(2) SPECTRAL_ZERO DOES NOT 'REFUSE' AT ALL ON MOST CELLS WHERE ITS certify_rate IS 0.000: it "
    "returns a finite 73-223 and the ceiling rejects it (100% cap on every triangular cell at "
    "dep >= 10 / n=3000 and on all three AR(1) cells). Reading its zeros as estimator refusals "
    "over-credits its caution -- an estimator that reads 223 and is externally capped is not the "
    "same object as one that declines. "
    "(3) THE i.i.d. CELL IS THE STRONGEST TELL. `variance_ratio` at n=3000 on WHITE NOISE, where "
    "the answer is q=2: 54.7% estimator-NaN plus 24.9% cap -- i.e. a quarter of white-noise draws "
    "produce a range estimate ABOVE 60. Both channels are the read-off rule and the grid "
    "(RULE_IS_NOT_THE_STATISTIC), not the dependence structure. "
    "(4) THE min_blocks CHANNEL IS DEAD CODE AT THESE n_dec (0.000 everywhere, 63/63 cells) and "
    "the calib channel is ONE stratum (below). Every 'refuses X%' sentence in this module is now "
    "scoped to a channel."
)

CALIBRATION_REFUSAL_CHANNEL = {
    "source": "calib_table.py -> CALIB_TABLE.json (same parameters as the sweep's prewarm: 200 "
              "calibration + 200 validation trials, n_boot=300, seed 0 -- it reproduces the table "
              "the sweep consumed and then discarded, which is why this was invisible until "
              "cycle 3).",
    "strata_touched": 27,
    "strata_failing": 1,
    "failing_stratum": {"block": 3, "n_dec": 3000, "k": 1.203, "validation_coverage": 0.925,
                        "target": 0.93, "validation_n": 200, "binomial_se": 0.0186,
                        "z_vs_target": -0.27},
    "consequence": (
        "`achieved` is a hard >=, so every trial whose range snaps to block=3 at n_dec=3000 "
        "refuses -- on a validation batch that misses the bar by 0.005 with an SE of 0.019, i.e. "
        "a coin-flip at this batch size. That single stratum IS the entire calib column: 31.5% / "
        "31.7% of spectral_zero's draws on the persistent component, 21.8% on i.i.d. at n=3000, "
        "and 4.2% / 2.9% / 5.2% of acf_band's on the same cells. Those cells' certify_rate is "
        "therefore NOT the estimator declining to read the data. NO COVERAGE NUMBER MOVES: "
        "coverage is conditional on certification and the refused draws were never scored -- "
        "and the BIAS RUNS TOWARD THIS ROW'S NEGATIVE, NOT AWAY FROM IT, because block=3 is "
        "selected only by range reads in (2, 3], i.e. the SHORTEST reads, which are exactly "
        "the draws most likely to under-cover on a persistent family: had that stratum cleared "
        "0.930, the affected coverages would if anything be LOWER, never higher. "
        "DISCLOSED, NOT TUNED -- raising n_trials_valid until block=3 clears the bar would be "
        "changing a knob to change a number after seeing it."
    ),
}

VR_LEAK_CELL = {
    "why": "Cycle 2 asserted `variance_ratio` 'does not over-certify anywhere' while its own "
           "table showed 0/8 on persistent phi=.999 -- p = 0.07^8 = 5.8e-10 against the 0.93 "
           "bar, decisive in the FAIL direction. POWER_AND_RESOLUTION's 'n_certified < 20 is not "
           "readable as pass or fail at all' shielded exactly that evidence. Rather than argue "
           "the threshold, the SAME cell was re-run at 5x trials until it cleared the threshold "
           "on cycle 2's own terms. Nothing else re-run; no parameter changed.",
    "cell": ("variance_ratio", "persistent_component_phi0.999", 3000),
    "n_trials": 5000, "seed": 901,
    "n_certified": 27, "certify_rate": 0.0054, "n_contained": 5,
    "coverage": 0.1852, "coverage_se": 0.0748,
    "clopper_pearson_95": [0.063, 0.381],
    "p_value_vs_bar_0.93": 2.2e-21,
    "pooled_with_shipped_cell": {"contained": 5, "certified": 35, "coverage": 0.1429,
                                 "clopper_pearson_95": [0.048, 0.303],
                                 "p_value_vs_bar_0.93": 5.2e-30},
    "independent_reverification": "The cycle-3 re-verify independently drew this cell and got 13 "
                                  "certified at coverage 0.308 [0.091, 0.614]. Pooling all three "
                                  "independent draws: 9/48 = 0.188 [0.089, 0.326], p = 8e-37.",
    "reading": "`variance_ratio` OVER-CERTIFIES on this family. It is a rare event (0.5-0.8% of "
               "draws) and does not change the top-line negative, which was already NO for other "
               "reasons -- but 'never over-certifies' is false, and it was the one property that "
               "made this candidate sound qualitatively safer than the other two. It is not.",
}

ACF_BAND_READOUT_SPREAD = {
    "source": "spread_check.py -> SPREAD_CHECK.json (estimator only, no bootstrap; seed 901).",
    "persistent_component_phi0.99_n3000": {"n_draws": 1000, "n_nan": 0, "median": 18.0,
                                           "p10": 3.0, "p90": 62.0, "min": 1.0, "max": 177.0},
    "triangular_dep20_n4000": {"n_draws": 200, "n_nan": 0, "median": 29.0, "p10": 20.0,
                               "p90": 63.1, "min": 19.0, "max": 149.0,
                               "frac_inside_unit_test_bound_20_60": 0.845},
    "reading": "'THE STATISTIC DOES SEE THE STRUCTURE' IS TRUE OF THE MEDIAN ONLY. On the family "
               "gate (ii) is about, the median readout of 18 sits inside a p10-p90 span of "
               "3 to 62 -- a single draw's range read is close to uninformative, which is a "
               "second, independent reason the median-18 result does not rescue the candidate "
               "(the first being that its interval under-covers anyway). The second row is the "
               "unit test's own cell and is discussed in `test_range_probe.py`.",
}

# ---------------------------------------------------------------------------------------------
# GATE (iii) — the real minute series. CONSTRUCTION, stated exactly, with cycle 1's two
# overstatements corrected.
# ---------------------------------------------------------------------------------------------
GATE_III_CONSTRUCTION = {
    "lineage": "H in {1,5,20}, long_short, N=60, tau=0.9, fee=8bp (897's smaller attainable maker "
               "tier), k=2, through the SAME 806/808/809 lineage 897 uses (`probe.py`/`decide.py`, "
               "imported verbatim), against the byte-verified extract "
               "`runs/897_corrected_809/minute_extract_897.db`. OOS window 2024-11-01 -> "
               "2025-04-01 (`probe.Config` defaults, UNMODIFIED).",
    "horizons_run": [1, 5, 20],
    "h_shortfall_cycle1": "CLOSED. Cycle 1 ran H=1 only against a gate that names H in {1,5,20}, "
                          "and disclosed it only in passing. All three now run; the "
                          "non-stationarity finding replicates at all three (see "
                          "GATE_III_NONSTATIONARITY).",
    "symbols": ["XBT/USD", "/USDTEUR", "XBT/EUR", "ETH/USD", "LTC/EUR", "LTC/USD", "XRP/USD"],
    "universe_honestly_described": "NOT 'the 7 majors' the task file names. `n_universe=7` "
                                   "(train-window-locked top-N-by-coverage) resolves to FOUR "
                                   "underlyings -- BTC, ETH, LTC, XRP -- with BTC and LTC listed "
                                   "twice against different quote currencies, plus one stablecoin "
                                   "FX pair (/USDTEUR). Cross-sectional breadth is roughly half "
                                   "what the asset count suggests. It is A real per-decision "
                                   "series through the audited lineage, which is what the gate "
                                   "needs; it is not the named universe.",
    "k16_degenerate_cell": "897's own grid default k=16 REFUSES OUTRIGHT (all-zero book, every "
                           "bar, forever) against a 7-asset universe under variant='long_short', "
                           "because `decide._decide_step` requires present_count >= 2*k. "
                           "Measured directly, twice (build and fix-cycle re-verification): "
                           "k=16, n_universe=7 -> n_decisions=0. Not a data problem -- a "
                           "k/universe-size mismatch that produces a SILENT degenerate cell. "
                           "Routed to TASK_902 as a class; not fixed here.",
    "k2_justification_corrected": "Cycle 1 justified k=2 as 'the same long_short shape at a "
                                  "comparable ratio (2 of 7 ~= 29% vs 16 of 64 = 25%)'. WRONG: "
                                  "tau=0.9 binds BEFORE k in BOTH configurations. "
                                  "`_decide_step` sets n_elig = max(1, round((1-tau)*npr)), which "
                                  "is 1 for npr in {4..7} and 6 at npr=64. MEASURED on this "
                                  "series: every decision bar has all 7 assets present and an "
                                  "eligible pool of exactly 1 per side (present_count_hist = "
                                  "{7: 217438}, n_elig_hist = {1: 217438} at H=1; same at H=5 and "
                                  "H=20). So the book takes at most one fresh long and one fresh "
                                  "short per bar at weight 1/k = 1/2, with held names retained "
                                  "only by the N=60 min-hold rule. k=2 is simply the largest k "
                                  "that lets this universe trade at all, not a shape match.",
    "provenance": "INDEPENDENT of the D57 halted cycle's pinned 0.639 construction -- that "
                  "cycle's exact symbol selection was never committed to this repo. This series "
                  "is written to REAL_SERIES_H{1,5,20}.npy with sha256 in REAL_LEG.json so any "
                  "later re-read needs no rebuild.",
}

# ---------------------------------------------------------------------------------------------
# GATE (iii), THE FINDING. The chunk-containment construction (the D57 halt's 97-disjoint-chunk
# design, re-run here) DOES NOT MEASURE A RANGE ESTIMATOR. It measures whether the population
# per-decision Sharpe is constant across the OOS window. On this series it is not.
# Three independent measurements, all from `coverage_harness.stationarity_diagnostics`, none of
# which calls a range estimator:
# ---------------------------------------------------------------------------------------------
GATE_III_NONSTATIONARITY = {
    "H=1": {
        "n_decisions": 26296, "n_chunks": 26, "chunk_size": 1000, "grand_sharpe": -0.188096,
        "boot_se_by_block": {2: 0.033644, 3: 0.034115, 5: 0.033858, 8: 0.033989, 12: 0.034021,
                             18: 0.034328, 26: 0.034760, 38: 0.035087, 60: 0.035912},
        "boot_se_block60_over_block2": 1.067,
        "chunk_vr_mean": {2: 0.982, 4: 0.964, 8: 0.933, 16: 0.927, 32: 0.924, 64: 0.882},
        "series_vr": {2: 0.984, 8: 0.931, 32: 0.906, 128: 0.775, 512: 0.523, 2048: 0.273},
        "between_chunk_sd": 0.09491, "within_chunk_se_mean": 0.03538,
        "between_within_variance_ratio": 7.19,
        "chi2_stat": 196.5, "chi2_df": 25, "chi2_p_value": 1.42e-28,
        # SLOPE RETAINED AS A RECORDED NUMBER, NOT AS A FINDING -- it is not significant; see
        # DRIFT_REFIT (t = -1.61, p = 0.122, R^2 = 0.097). The first3/last3 pair below is a
        # SELECTED 3-vs-3 EXTRACT, kept only because cycle 2 quoted it and it must be visible
        # next to its own retraction. Stated exactly (and NOT as an earlier cycle-3 draft of this
        # comment had it): at H=1 the last three chunks ARE the three lowest
        # (DRIFT_CHECK.json["H=1"]["lowest3_chunks"] = [23, 24, 25]); what makes the extract
        # SELECTED is that the path is not monotone through them -- chunks 9-12 run -0.301 /
        # -0.348 / -0.302 / -0.316, comparable lows in the MIDDLE of the window, and chunks 20-21
        # run -0.143 / -0.127, as high as anything in the first three. Down, up, down. The
        # non-significance conclusion does not rest on this sentence at all: it rests on
        # t / p / R^2 / Spearman / the two-bucket halves / non-replication at H=5 and H=20
        # (DRIFT_REFIT). It is not evidence of a trend.
        "chunk_sharpe_slope_per_chunk": -0.00386,
        "chunk_sharpe_slope_se": 0.00241,
        "chunk_sharpe_slope_p": 0.122,
        "chunk_sharpes_first3_SELECTED_EXTRACT": [-0.356, -0.216, -0.109],
        "chunk_sharpes_last3_SELECTED_EXTRACT": [-0.380, -0.393, -0.425],
    },
    "H=5": {
        "n_decisions": 25785, "n_chunks": 25, "grand_sharpe": -0.182050,
        "boot_se_by_block": {2: 0.034268, 8: 0.033339, 60: 0.034870},
        "boot_se_block60_over_block2": 1.018,
        "chunk_vr_mean": {2: 0.977, 8: 0.881, 64: 0.762},
        "series_vr": {2: 0.992, 32: 0.849, 512: 0.734, 2048: 1.063},
        "between_chunk_sd": 0.08835, "within_chunk_se_mean": 0.03501,
        "between_within_variance_ratio": 6.37,
        "chi2_stat": 151.8, "chi2_df": 24, "chi2_p_value": 1.56e-20,
        "chunk_sharpe_slope_per_chunk": -0.00274, "chunk_sharpe_slope_se": 0.00244,
        "chunk_sharpe_slope_p": 0.273,
    },
    "H=20": {
        "n_decisions": 25512, "n_chunks": 25, "grand_sharpe": -0.197564,
        "boot_se_by_block": {2: 0.035525, 8: 0.035927, 60: 0.035884},
        "boot_se_block60_over_block2": 1.010,
        "chunk_vr_mean": {2: 0.988, 8: 0.951, 64: 0.818},
        "series_vr": {2: 0.993, 32: 0.908, 512: 0.931, 2048: 1.016},
        "between_chunk_sd": 0.09368, "within_chunk_se_mean": 0.03611,
        "between_within_variance_ratio": 6.73,
        "chi2_stat": 219.0, "chi2_df": 24, "chi2_p_value": 2.14e-33,
        "chunk_sharpe_slope_per_chunk": -0.00118, "chunk_sharpe_slope_se": 0.00264,
        "chunk_sharpe_slope_p": 0.660,
    },
    "reading": (
        "1. THE BLOCK LENGTH CANNOT MOVE THE INTERVAL. The mean within-chunk block-bootstrap SE "
        "of the Sharpe changes by 6.7% (H=1), 1.8% (H=5) and 1.0% (H=20) across a THIRTYFOLD "
        "change in block length (2 -> 60). The block is the only quantity a range estimator sets. "
        "2. THE SERIES IS ANTI-PERSISTENT, i.e. the opposite of the failure mode a range probe "
        "guards against: VR(q) DECLINES in q inside chunks (1.0 -> 0.88 / 0.76 / 0.82 at q=64) "
        "and across the full series (0.273 at q=2048, H=1). Under negative dependence a block "
        "bootstrap is if anything CONSERVATIVE. "
        "3. THE PSEUDO-TRUTH IS NOT A CONSTANT -- AS EXCESS DISPERSION, NOT AS A TREND. "
        "Between-chunk Sharpe dispersion is 6.4-7.2x the within-chunk SE (chi2 = 196.5 / 151.8 / "
        "219.0 on 25 / 24 / 24 df, p = 1e-20 to 1e-33). CYCLE 2 CALLED THIS A 'MONOTONE DRIFT' "
        "AND THAT WORD IS STRUCK (fix cycle 3): refit with standard errors (DRIFT_REFIT), the "
        "H=1 slope is -0.00386 +- 0.00241, t = -1.61, p = 0.122, R^2 = 0.097, weakening to "
        "p = 0.273 (H=5) and p = 0.660 (H=20); at H=20 the second-half mean is HIGHER than the "
        "first; and detrending removes essentially none of the dispersion (variance ratio "
        "7.19 -> 6.77 at H=1). There is no established trend, and no decaying edge. A containment "
        "check scores every chunk against ONE number that the data says is not one number -- that "
        "is the finding, full stop, and it does not need a direction. "
        "CONCLUSION: containment of 0.58-0.82 states that the population per-decision Sharpe is "
        "NOT CONSTANT across this OOS window. No range estimator and no oracle can raise it. "
        "The gate as specified cannot discriminate between estimators; that is a real result "
        "about the GATE, and it is the one this row reports."
    ),
}

# ---------------------------------------------------------------------------------------------
# FIX CYCLE 3, MAJOR 4 — "MONOTONE DRIFT" IS STRUCK. THE FINDING IS EXCESS DISPERSION, FULL STOP.
# Cycle 2 reported `chunk_sharpe_slope_per_chunk` as a bare point estimate and read it in prose as
# a monotone drift, illustrated with a 3-vs-3 extract. A slope with no SE is not a trend. Refit of
# THE SAME chunk Sharpes already in REAL_LEG.json (`drift_check.py` -> DRIFT_CHECK.json; post-hoc
# read, no rebuild, no DB). Wording here is aligned with the downstream row this fed,
# `tasks/TASK_906_OOS_NONSTATIONARITY.md` (corrected 2026-08-24, commit d7b0d3f).
# ---------------------------------------------------------------------------------------------
DRIFT_REFIT = {
    "H=1": {"n_chunks": 26, "slope": -0.00386, "se": 0.00241, "t": -1.61, "p": 0.122,
            "r_squared": 0.097, "spearman_rho": -0.294, "spearman_p": 0.144,
            "first_half_mean": -0.2326, "second_half_mean": -0.2410,
            "between_within_var_ratio_raw": 7.19, "between_within_var_ratio_detrended": 6.77},
    "H=5": {"n_chunks": 25, "slope": -0.00274, "se": 0.00244, "t": -1.12, "p": 0.273,
            "r_squared": 0.052, "spearman_rho": -0.248, "spearman_p": 0.231,
            "first_half_mean": -0.2084, "second_half_mean": -0.2200,
            "between_within_var_ratio_raw": 6.37, "between_within_var_ratio_detrended": 6.30},
    "H=20": {"n_chunks": 25, "slope": -0.00118, "se": 0.00264, "t": -0.45, "p": 0.660,
             "r_squared": 0.009, "spearman_rho": -0.165, "spearman_p": 0.429,
             "first_half_mean": -0.2451, "second_half_mean": -0.2286,
             "between_within_var_ratio_raw": 6.73, "between_within_var_ratio_detrended": 6.96},
    "detrended_sd_note": "residual sd taken at ddof=2 (two fitted parameters), so the detrended "
                         "ratio is compared with the raw one on equal footing.",
    "reading": (
        "NO TREND IS ESTABLISHED AT ANY HORIZON. t = -1.61 / -1.12 / -0.45 (p = 0.122 / 0.273 / "
        "0.660); the linear fit explains 9.7% of the between-chunk variance at H=1 and 0.9% at "
        "H=20; Spearman rho is -0.29 / -0.25 / -0.17, none significant; and at H=20 the "
        "second-half mean (-0.229) is HIGHER than the first-half (-0.245), i.e. the sign of the "
        "'drift' does not even survive a two-bucket comparison. "
        "THE 3-vs-3 ILLUSTRATION IS SELECTED. At H=1 the last three chunks are in fact the three "
        "lowest, but the series is not monotone through them: chunks 9-12 run -0.301 / -0.348 / "
        "-0.302 / -0.316 -- comparable lows in the MIDDLE of the window -- and chunks 20-21 run "
        "-0.143 / -0.127, as high as anything in the first three. Down, up, down is not a drift, "
        "and the pattern does not replicate: at H=5 and H=20 the three lowest chunks are 0, 9 and "
        "24. Quoting the first three against the last three picks the two ends of a "
        "non-significant fit. "
        "WHAT SURVIVES, UNTOUCHED AND LOAD-BEARING: the pseudo-truth is not a constant. The chi2 "
        "test of a constant population Sharpe rejects at p = 1e-20 to 1e-33, and DETRENDING "
        "REMOVES ESSENTIALLY NONE OF THE EXCESS DISPERSION (between/within variance ratio "
        "7.19 -> 6.77 at H=1, 6.37 -> 6.30 at H=5, and it RISES at H=20). So the dispersion is "
        "not a trend the containment check could be corrected for -- it is dispersion, and the "
        "containment arithmetic follows from it directly: half-width 1.265 * 1.96 * 0.0354 = "
        "0.0878 against a between-chunk sd of 0.0949 predicts containment 0.646, measured 0.615 "
        "(16/26). "
        "WHY THE DISTINCTION MATTERS DOWNSTREAM: 'monotone drift' reads as a DECAYING EDGE, which "
        "would imply the recent regime is the relevant one and older performance should be "
        "discounted. Excess dispersion implies something different and weaker -- that a "
        "whole-window number is an average over heterogeneous sub-periods whose spread the CI "
        "does not represent. TASK_906 owns that question; this row must not prejudge it with a "
        "trend it did not measure."
    ),
}

# The containment numbers themselves, RETAINED but re-scoped: descriptive of the series, never
# evidence about an estimator. SEs are binomial and treat the 26 (or 25) chunks as INDEPENDENT,
# which the measured EXCESS DISPERSION says they are not -- read them as a floor on the
# uncertainty. (Cycle 2 wrote 'the measured drift' here; struck, see DRIFT_REFIT.)
GATE_III_REAL_DATA_CONTAINMENT = {
    "H=1": {"n_chunks": 26, "grand_sharpe": -0.188096,
            "acf_band": {"certify_rate": 1.000, "contains": 0.6154, "se": 0.0954,
                         "median_range_est": 1.0},
            "variance_ratio": {"certify_rate": 0.654, "contains": 0.8235, "se": 0.0925,
                               "median_range_est": 16.0},
            "spectral_zero": {"certify_rate": 1.000, "contains": 0.6538, "se": 0.0933,
                              "median_range_est": 2.12}},
    "H=5": {"n_chunks": 25, "grand_sharpe": -0.182050,
            "acf_band": {"certify_rate": 1.000, "contains": 0.7200, "se": 0.0898,
                         "median_range_est": 1.0},
            "variance_ratio": {"certify_rate": 0.400, "contains": 0.7000, "se": 0.1449,
                               "median_range_est": 32.0},
            "spectral_zero": {"certify_rate": 1.000, "contains": 0.7200, "se": 0.0898,
                              "median_range_est": 1.96}},
    "H=20": {"n_chunks": 25, "grand_sharpe": -0.197564,
             "acf_band": {"certify_rate": 1.000, "contains": 0.6800, "se": 0.0933,
                          "median_range_est": 1.0},
             "variance_ratio": {"certify_rate": 0.680, "contains": 0.5882, "se": 0.1194,
                                "median_range_est": 32.0},
             "spectral_zero": {"certify_rate": 1.000, "contains": 0.6800, "se": 0.0933,
                               "median_range_est": 2.00}},
    "withdrawal": (
        "CYCLE 1 READ THESE AS 'FAILS FOR ALL THREE -- every candidate certifies 100% of chunks "
        "while measured containment is 0.58-0.65'. THAT READING IS WITHDRAWN, and so is the same "
        "construction's use as evidence against TAU (the D57 halt's 0.639 at ~100% certification). "
        "The construction cannot separate an estimator from the series -- see "
        "GATE_III_NONSTATIONARITY and PAIRED_DISCRIMINATION. What survives against tau is its "
        "SYNTHETIC out-of-family coverage (0.805 / 0.376 under a KNOWN population Sharpe), which "
        "is a different measurement and is untouched by this. Note also, descriptively, that the "
        "estimator with the LONGEST blocks here (variance_ratio, block 18-38) has the highest "
        "containment at H=1 (0.824) and the LOWEST at H=20 (0.588) -- no ordering, on a "
        "self-selected 40-68% of chunks."
    ),
}

# The discrimination test: restrict each pair of estimators to the chunks BOTH certified, which
# removes the selection difference and leaves only the block length. `paired_check.py` (a post-hoc
# read over REAL_LEG.json; no rebuild, no DB) writes these into the artifact.
PAIRED_DISCRIMINATION = {
    "H=1": [("acf_band blocks[2,3,8,18] 16/26", "spectral_zero blocks[2,3,5] 17/26"),
            ("acf_band blocks[2,3,8,18] 14/17", "variance_ratio blocks[5,8,18,38] 14/17 IDENTICAL"),
            ("spectral_zero blocks[2,3,5] 14/17", "variance_ratio blocks[5,8,18,38] 14/17 IDENTICAL")],
    "H=5": [("acf_band blocks[2,3,5,8,12] 18/25", "spectral_zero blocks[2,3,5] 18/25 IDENTICAL"),
            ("acf_band blocks[2,3,8,12] 7/10", "variance_ratio blocks[2,18,38] 7/10 IDENTICAL"),
            ("spectral_zero blocks[2,3,5] 7/10", "variance_ratio blocks[2,18,38] 7/10 IDENTICAL")],
    "H=20": [("acf_band blocks[2,3,5,8,12,18,38] 17/25", "spectral_zero blocks[2,3,5] 17/25 IDENTICAL"),
             ("acf_band blocks[2,3,5,8,12,18] 10/17", "variance_ratio blocks[2,5,8,18,38] 10/17 IDENTICAL"),
             ("spectral_zero blocks[2,3,5] 10/17", "variance_ratio blocks[2,5,8,18,38] 10/17 IDENTICAL")],
    "reading": "Across all three horizons the estimators contain the grand Sharpe on EXACTLY THE "
               "SAME CHUNKS, chunk for chunk, despite block lengths differing by up to 19x (2 vs "
               "38); the single exception is H=1 acf_band vs spectral_zero, which differ on ONE "
               "chunk of 26. Gate (iii)'s containment statistic is, on this series, blind to the "
               "block length -- which is the entire output of a range estimator. It cannot rank "
               "these candidates, and it could not have ranked tau against them either.",
    "count_corrected_cycle3": "CYCLE 2 SAID '8 OF 9 PAIRS' AND THAT DOUBLE-COUNTS. With three "
                              "estimators there are 3 pairings per horizon but only 2 INDEPENDENT "
                              "comparisons -- the third is implied by the other two (if A==B and "
                              "B==C chunk-for-chunk, A==C is not new information). The honest "
                              "ceiling is 3 horizons x 2 = 6 independent comparisons, and the "
                              "rows above show that structure plainly: each horizon's third line "
                              "is the implied one.",
    "not_independent_evidence_cycle3": "THIS TEST IS A COROLLARY OF THE FLAT-SE MEASUREMENT, NOT "
                                       "A SECOND WITNESS, AND CYCLE 2'S MAJOR-3 SUMMARY PRESENTED "
                                       "IT AS ONE. `boot_se_by_block` already shows the "
                                       "within-chunk SE moving 1.0-6.7% across a 30x change in "
                                       "block; the resulting half-width difference between block "
                                       "2 and block 38 is 10-35%, while (measured by the cycle-3 "
                                       "re-verification) 14 of the 17 common chunks at H=1 sit "
                                       "more than 25% of a half-width from the containment "
                                       "boundary. The identical outcome was therefore close to "
                                       "predetermined by the flat SE. It is a legitimate "
                                       "ILLUSTRATION of that fact; it is not independent "
                                       "confirmation of it.",
}

# ---------------------------------------------------------------------------------------------
POWER_AND_RESOLUTION = (
    "Cycle 1 ran 200 trials/cell, which did NOT resolve the passing cells: e.g. acf_band / "
    "triangular_dep22 / n=1000 read 0.9436 +- 0.0165, z = 0.74 against the 0.93 bar -- "
    "indistinguishable from it -- and four more cells sat within 2 SE. This cycle runs 1000 "
    "trials/cell, giving SE ~ 0.005-0.008 near p = 0.97, so a 0.97 cell now clears 0.93 at 5-8 "
    "sigma. THE RAISE WAS FOR THE PASSING CELLS: the failing cells were already conclusive at 200 "
    "(and are 13-44 sigma now). "
    "WHAT STILL IS NOT RESOLVED, STATED RATHER THAN GLOSSED: every cell whose `n_certified` is "
    "small is an underpowered cell no matter how many trials were drawn, because the denominator "
    "is certifications, not trials. variance_ratio's entire table is in this category (certify "
    "rates 0.000-0.540), as are spectral_zero's triangular cells at dep >= 15 (0.000-0.135). Do "
    "NOT read variance_ratio's 0.97-0.99 coverages as 'clean everywhere except the persistent "
    "component': they are coverages on the 2-54% of draws it was willing to certify, i.e. on a "
    "subset selected by the estimator's own noise. Cells at n_certified < 20 (variance_ratio "
    "triangular_dep32/dep50 at 15/17, persistent phi=.999 at 8; spectral_zero triangular_dep22 "
    "n=1000 at 2) cannot ESTABLISH a pass. "
    "THAT RULE IS ONE-SIDED, AND CYCLE 2 APPLIED IT IN BOTH DIRECTIONS, WHICH WAS WRONG (fix "
    "cycle 3). Cycle 2 wrote that such cells 'are not readable as pass or fail at all' -- and the "
    "sentence then shielded the only conclusive evidence against `variance_ratio`: 0 of 8 against "
    "a 0.93 bar has p = 0.07^8 = 5.8e-10. A tiny denominator destroys the power to CONFIRM a high "
    "coverage and leaves ample power to REFUTE one, because refuting needs the observed rate to "
    "be far below the bar and 0/8 is as far below as arithmetic allows. Correct statement: cells "
    "at n_certified < 20 are not readable as a PASS; they are readable as a FAIL whenever the "
    "one-sided binomial p against the bar is small. That cell was subsequently re-run at 5x "
    "trials to settle it on cycle 2's own threshold anyway -- 27 certifications, coverage 0.185 "
    "[0.063, 0.381], p = 2e-21 (VR_LEAK_CELL). "
    "SEPARATION vs POWER, ALSO CORRECTED: cycle 2's VERDICT said 'no cell has the power to "
    "establish >= 93%' for `variance_ratio`. That is a POWER claim and it is false -- "
    "variance_ratio / iid / n=1000 has n_certified = 540 at 0.9815 +- 0.0058, which is +8.9 sigma "
    "ABOVE the bar. The correct objection is SELECTION, not power, and it is already stated "
    "properly above: those coverages are conditional on a subset the estimator selected with its "
    "own noise, so they do not generalise to the draws it refused. "
    "The gate-(iii) containment SEs are binomial over 25-26 chunks and assume the chunks are "
    "independent draws from one population; the measured excess dispersion "
    "(GATE_III_NONSTATIONARITY) says they are not, so those SEs are floors."
)

VERDICT = (
    "NO CANDIDATE IN {acf_band, variance_ratio, spectral_zero} IS CERTIFIED AS A GENERAL RANGE "
    "INSTRUMENT AT >= 93% OUT-OF-FAMILY. That conclusion has survived three cycles and an "
    "independent re-verification unchanged. WHAT KEPT CHANGING WERE THE REASONS -- cycle 1's were "
    "wrong, cycle 2's were wrong in a different place -- so the reasons are stated below with "
    "their measurements attached and their previous versions named. "
    "(a) `acf_band` and `spectral_zero` FAIL, measured and conclusive: on the 4%-variance "
    "persistent component they certify 71-85% of draws at coverage 0.724 +- 0.015 / 0.626 +- "
    "0.018 (phi=.99) and 0.262 +- 0.015 / 0.260 +- 0.016 (phi=.999), 13-44 sigma below the bar. "
    "Everywhere else both are >= 0.955. This is the part of the negative nothing has touched. The "
    "mechanism is NOT that the persistent component's correlation is below the noise floor -- "
    "with a correctly-directed band `acf_band` reads its MEDIAN range as 18 (per-draw p10 = 3, "
    "p90 = 62) -- but that the conversion of that range into an interval at block <= 60 still "
    "under-covers. "
    "(b) `variance_ratio` FAILS TOO, AND THE CANDIDATE IS NOT THE SAFE ONE. Three corrections to "
    "cycle 2, each measured: (b1) IT OVER-CERTIFIES. On persistent phi=.999 it covers 0.185, 95% "
    "CI [0.063, 0.381] on 27 certifications (p = 2e-21 vs the bar; 5/35 = 0.143 [0.048, 0.303] "
    "pooled with the shipped cell). Cycle 2's 'does not over-certify anywhere' is WITHDRAWN -- it "
    "was contradicted by this module's own 0/8 cell, which cycle 2's own n<20 rule had shielded. "
    "(b2) ITS 'REFUSAL' IS NOT ONE THING. At phi=.99 the estimator NaNs on 84.9% +- 1.1% of draws "
    "and `block_rule`'s MAX_BLOCK_CAP=60 rejects 100% of the 151 finite reads (essentially all, "
    "not structurally all: a fresh seed gave 145/146) -- so gate (ii)'s "
    "REFUSE branch is taken partly by a CEILING that would reject any estimator reading above 60, "
    "tau included. Every 'refuses X%' in this module is now split by channel (REFUSAL_CHANNELS). "
    "(b3) 'UNUSABLY CONSERVATIVE' IS WITHDRAWN AS A CLAIM ABOUT THE STATISTIC. The estimator's "
    "own refusal rate is a property of a FIXED 0.10 tolerance at a grid whose sampling sd reaches "
    "0.316 -- it refuses 54.7% +- 1.6% of i.i.d. draws where the answer is q=2, 91% of those "
    "refusals are driven by the two noisiest grid points, and its median return on white noise is "
    "exactly a quarter of the largest grid point at every n. An SE-aware tolerance finds a "
    "plateau on 100% of the same draws and gets q=2 right 96.7% of the time. That counterfactual "
    "is recorded as EVIDENCE and deliberately NOT shipped (it would be a new instrument needing "
    "its own out-of-family certification). The refusal rate is therefore a RULE-AND-GRID property "
    "that has NOT been separated from the variance-ratio statistic, and this row does not claim "
    "otherwise. See RULE_IS_NOT_THE_STATISTIC. "
    "(c) GATE (iii) CANNOT DISCRIMINATE. Its containment numbers measure the SERIES: the "
    "population per-decision Sharpe is not constant across the OOS window (between-chunk "
    "dispersion 6.4-7.2x the within-chunk SE, chi2 p = 1e-20 to 1e-33). THE FINDING IS EXCESS "
    "DISPERSION, NOT DRIFT -- cycle 2's 'monotone drift' is STRUCK: the slope is -0.00386 +- "
    "0.00241 (p = 0.122) at H=1, weaker at H=5/20, and detrending removes essentially none of the "
    "dispersion (7.19 -> 6.77). The within-chunk SE is flat in block, the series is "
    "anti-persistent, and estimator pairs at blocks differing up to 19x contain the grand Sharpe "
    "on identical chunks -- the last being a COROLLARY of the flat SE, not independent evidence. "
    "All cycle-1 containment-based conclusions are withdrawn, including retroactively against "
    "tau. "
    "WHAT THE PLANNER SHOULD TAKE FROM THIS ROW: the direction is NOT closed by a demonstration "
    "that range estimation is hopeless. All three mechanisms were measured to over-certify "
    "somewhere; none was measured to be a safe alternative; the real-data gate that supplied a "
    "third of the negative was measuring the wrong quantity; and one candidate's headline "
    "behaviour is still confounded with its read-off rule and grid. The open questions this "
    "leaves are (1) whether ANY estimator can convert a correct range read into a covering "
    "interval when the required block exceeds MAX_BLOCK_CAP=60 -- `acf_band` at phi=.99 reads a "
    "median 18 and still under-covers, which points at the block CEILING and the "
    "matched-dependence calibration convention rather than at the range read; (2) what a "
    "real-data gate should be, given the measured excess dispersion of the 897 per-decision "
    "series over its OOS window -- now its own row, TASK_906, since it bears on every "
    "whole-window Sharpe claim made on that series, not just on this instrument; and (3) whether "
    "a variance-ratio read-off whose tolerance tracks its own sampling error is a usable "
    "instrument -- unknown, because certifying it is a new row's worth of sweep, and this row "
    "will not assert it either way from a counterfactual. Nothing was tuned, in any cycle, to "
    "make any of these look answered."
)

COMPUTE_BUDGET = {
    "calibration_prewarm_srv": "9 block strata x 5 n_dec = 45 (block, n_dec) cells, 200+200 "
                               "trials, n_boot=300 each, n_jobs=8, gpu-server, "
                               "tools/safe_run.sh --cpu --mem 16G (62s)",
    "synthetic_sweep_srv": "3 estimators x 21 (family, n_dec) cells x 1000 trials = 63,000 "
                           "trials; 674s wall-clock including prewarm; gpu-server, tmux session "
                           "l901, tools/safe_run.sh --name t901validate2 --mem 16G --cpu",
    "real_data_leg_workstation": "3 horizons (H=1,5,20): calibration prewarm for n_dec=1000 (16s) "
                                 "+ per-horizon series construction (806/808/809 ridge fit + "
                                 "chunked predict + decision engine, n_universe=7; 301s / 92s / "
                                 "54s) + stationarity diagnostics (26 chunks x 9 blocks x 1000 "
                                 "bootstrap resamples) + containment x3 estimators; 497s total; "
                                 "tools/safe_run.sh --name t901realleg3 --cpu --mem 16G on the "
                                 "WORKSTATION (see COMPUTE_LOCATION_DEVIATION)",
    "paired_discrimination": "post-hoc read over REAL_LEG.json, no rebuild, no DB access; "
                             "tools/safe_run.sh --name t901paired --cpu --mem 8G, ~1s",
    "tests": "pytest test_range_probe.py: 26 passed in 4.74s (gpu-server, wrapped). "
             "python -m trader selftest: see the fix-cycle report (gpu-server, wrapped).",
    "fix_cycle_3_diagnostics_srv": "readoff_diagnostics.py -- 3 rule-vs-statistic cells x 1000 "
                                   "i.i.d. draws + 63 refusal-channel cells x 1000 draws "
                                   "(estimator + block_rule only, NO bootstrap), 77s including a "
                                   "27-stratum calibration prewarm; gpu-server, tmux l901diag, "
                                   "tools/safe_run.sh --name t901readoff --cpu --mem 16G. "
                                   "calib_table.py (tmux l901calib, --name t901calibtab, ~50s), "
                                   "spread_check.py (tmux l901spread, --name t901spread, ~3s), "
                                   "vr_leak_cell.py (tmux l901vr, --name t901vrleak, 10s for "
                                   "5000 trials on ONE cell). No 63-cell x 1000-trial sweep was "
                                   "re-run in this cycle.",
    "fix_cycle_3_drift_refit_workstation": "drift_check.py -- post-hoc OLS/Spearman refit over "
                                           "REAL_LEG.json's already-recorded chunk Sharpes; no "
                                           "rebuild, no DB, ~1s; workstation (REAL_LEG.json is a "
                                           "workstation artifact, same as paired_check.py), "
                                           "tools/safe_run.sh --name t901drift --cpu --mem 8G.",
}

COMPUTE_LOCATION_DEVIATION = (
    "The real-data leg (gate iii) ran on the WORKSTATION, not gpu-server, wrapped with "
    "tools/safe_run.sh --cpu --mem 16G exactly like the server jobs -- a disclosed departure "
    "from this task's default 'everything runs on gpu-server' instruction, not an undisclosed "
    "corner cut. Reason: it needs runs/897_corrected_809/minute_extract_897.db, an 11GB "
    "byte-verified extract of the real minute table, and the standing infra rule "
    "(workstation<->server uplink ~5 Mbps, the 24GB source DB is never pushed to the server, "
    "extended-DB reads run on the workstation) applies just as much to an 11GB extract as to "
    "the 24GB source -- this is D54's documented data-local exception, not a new one invented "
    "here. The Monte Carlo coverage/calibration sweep -- the actual compute-shaped part this "
    "row's guidance is aimed at -- DID run on gpu-server as instructed, in its own tmux session. "
    "The real-data read itself is read-only throughout: real_series.py opens the extract "
    "exclusively through sweep_gated_maker._ReadOnlySqlite3ForProbe (897's own read-only shim, "
    "imported not copied), and never touches the real trading_data.db at all."
)

GATE_IV_STATUS = (
    "DEFERRED, not fabricated. gate_iv_hook.py contains the re-read procedure (which artifact, "
    "which fields, which estimator to substitute for tau) as a docstring, and a callable, "
    "reread_897_conjunct, that REFUSES (raises Gate4NotYetRunnable) because 897's srv-leg "
    "artifact does not exist yet at the time of this implementation (897's guard was "
    "mid-relaunch -- tiers {8,16} pre-registered per the '897 SRV LEG LAUNCHED' commit). No 897 "
    "number appears anywhere in this row's code. Given VERDICT above, gate (iv)'s eventual "
    "re-read still has no CERTIFIED estimator to re-read 897's 'CI lo > 0' conjunct with: "
    "TASK_901 does not hand TASK_897 a working replacement for tau. What it does hand 897 is "
    "sharper: (a) `variance_ratio` under the corrected rule would decline rather than certify on "
    "the kind of series 897 cares about -- but that must be stated with cycle 3's two "
    "corrections attached: part of the decline is `block_rule`'s MAX_BLOCK_CAP=60 ceiling rather "
    "than the estimator (REFUSAL_CHANNELS), and the estimator's own share is confounded with its "
    "fixed-tolerance read-off rule (RULE_IS_NOT_THE_STATISTIC). It is a fail-closed answer rather "
    "than no answer; it is not a certified one, and on one measured family the candidate leaks "
    "and over-certifies at coverage 0.185 (VR_LEAK_CELL). "
    "And (b) the measured EXCESS DISPERSION of the 897 per-decision series across its OOS window "
    "(GATE_III_NONSTATIONARITY, DRIFT_REFIT) is a problem for the 'CI lo > 0' conjunct that no "
    "interval width fixes, and that nobody had measured before this cycle. State it as "
    "dispersion, NOT as a decaying edge: the trend is not significant at any horizon (p = 0.122 / "
    "0.273 / 0.660) and detrending removes essentially none of the dispersion. TASK_906 owns the "
    "follow-up."
)
