#!/usr/bin/env python3
"""
TASK_901 — the range-based dependence probe.

WHY THIS EXISTS
----------------
A block bootstrap needs `block >= dependence RANGE` (how far out the autocorrelation genuinely
extends). `experiments/897_corrected_809/honest_estimator.py::dependence_length` measures tau
(the integrated autocorrelation time) as a stand-in for range, and tau equals range ONLY inside
the exact triangular-ACF family it was built against (`calibration.simulate_overlapping`) — where
tau = dep exactly. Out of that family it CONFLATES range with amplitude: a persistent component
worth only 4% of variance (0.2 * AR(1), phi=0.99 — population tau 8.6) reads tau=1, "certifies"
100% of draws, and the ACHIEVED coverage is only 0.805; at phi=0.999 coverage falls to 0.376,
below the 0.530 that raised D55 in the first place. On the real per-decision series every cell
reads tau=1-2 against block=60, so the tau guard is measured INERT exactly where 897 needs it.
See DECISIONS D55/D57 and `honest_estimator.dependence_length`'s own docstring for the full
history.

ONE PIECE OF THAT HISTORY IS WITHDRAWN BY THIS ROW, INCLUDING WHERE IT CUT AGAINST TAU. The D57
halt also cited an empirical containment check on the real data (97 disjoint 1000-decision chunks
containing the grand Sharpe only 0.639 of the time while tau certified ~100%). This row re-ran
that CONSTRUCTION on its own real series and measured what it actually responds to
(`coverage_harness.stationarity_diagnostics`, `measured_findings.GATE_III_NONSTATIONARITY`): the
within-chunk bootstrap SE is FLAT in block length, so the block — the only quantity a range
estimator sets — cannot move the containment number; and the chunk Sharpes' between-chunk
dispersion is many times the within-chunk SE, i.e. the "grand Sharpe" the check scores against is
not a single population value across the window. A containment fraction from that construction
measures NON-STATIONARITY, not an estimator's quality, and is not evidence for or against tau,
these three candidates, or an oracle. TAU'S SYNTHETIC OUT-OF-FAMILY FAILURE (the 0.805 / 0.376
coverages above) IS A DIFFERENT MEASUREMENT AND STANDS UNAFFECTED — it is a true-null Monte Carlo
with a known population Sharpe, which is exactly what the real-data check cannot be.

THIS MODULE MUST NOT REPEAT THAT FAILURE. Its own certification is the deliverable — see
`coverage_harness.py` for the out-of-family Monte Carlo validation and `measured_findings.py`
for the measured table. Nothing here is presumed certified until measured.

THREE CANDIDATES (task's own order; picked by MEASURED coverage, never by appeal — see
`coverage_harness.py` and `measured_findings.MEASURED_COVERAGE_TABLE`):

  (i)   `range_acf_band` — first lag where the sample ACF enters the STANDARD per-lag white-noise
        band (+-1.96/sqrt(n)), confirmed by `k_consec` consecutive in-band lags; no family-wise
        widening, because widening the band shortens the block and is the fail-open direction
        here (see `_band_z`). This is the same "first-crossing" idea `dependence_length` uses,
        but reports the CROSSING LAG itself (the range) rather than `1 + 2*sum|rho|` (tau, which
        additionally weights by amplitude at every intervening lag — the mechanism that conflates
        range with amplitude).
  (ii)  `range_variance_ratio` — a Lo-MacKinlay-style overlapping variance ratio VR(q) at a
        geometric grid of q, reading off the smallest q whose VR is within tolerance of VR at
        EVERY LARGER grid point (a levelled-off curve, not a single small step). If no such q
        exists within the available grid (the dependence's reach is not resolvable at this
        sample size), REFUSES rather than reporting a number the data cannot support.
  (iii) `range_spectral_zero` — an Andrews (1991) AR(1) plug-in automatic bandwidth for a
        Bartlett-kernel long-run-variance (spectral-density-at-zero) estimator, read directly as
        a range in decision units. REFUSES when the selected bandwidth exceeds what the sample
        size can support.

Every estimator shares the signature `estimate(x, **kwargs) -> (range_est: float, info: dict)`,
returning `float('nan')` for range_est on refusal, with `info["reason"]` naming why.

`block_rule` turns a certified range into a block length and a calibrated width-inflation factor
`k` (or a refusal), by REUSING (never re-implementing) the real bootstrap
(`honest_estimator.block_bootstrap_sharpe_ci`) and the real closed-form width-calibration method
(`calibration.k_needed_for_trial`) from `experiments/897_corrected_809/` — imported, not edited,
not copy-pasted.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import norm

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[2]
_897 = _REPO / "experiments" / "897_corrected_809"
if str(_897) not in sys.path:
    sys.path.insert(0, str(_897))

import honest_estimator as hest      # noqa: E402  (READ-ONLY import — never edited, never copied)
import calibration as cal897         # noqa: E402  (READ-ONLY import — reuses k_needed_for_trial,
                                      # simulate_overlapping; never re-derived)

TARGET_COVERAGE = 0.93                 # same bar 892/897 certify to.
MAX_BLOCK_CAP = int(hest.MAX_BLOCK_DEFAULT)   # 60 — imported so this module can never drift from
                                               # the estimator whose bootstrap it feeds.
MIN_NONOVERLAP_BLOCKS = 10              # refuse a block choice that would leave the bootstrap
                                         # fewer non-overlapping blocks than this (897's own
                                         # block_for guarantees >= 20; 10 is a disclosed, slightly
                                         # looser floor for THIS module's own block choice, which
                                         # is range-driven rather than n_dec-driven).

# A continuous range estimate is snapped UP to the nearest of these strata (never down — rounding
# a block length DOWN from the estimated range is the dangerous direction) before calibration is
# looked up. This bounds the number of DISTINCT (block, n_dec) calibrations the coverage harness
# must pay for to len(BLOCK_STRATA) x len(n_dec grid), instead of one fresh ~minutes-long
# calibrate-then-validate pass per trial's raw (often unique) integer range estimate.
BLOCK_STRATA: Tuple[int, ...] = (2, 3, 5, 8, 12, 18, 26, 38, MAX_BLOCK_CAP)


def snap_block(range_est: float) -> int:
    r = max(2, min(MAX_BLOCK_CAP, int(math.ceil(range_est))))
    for s in BLOCK_STRATA:
        if s >= r:
            return int(s)
    return MAX_BLOCK_CAP


def _acf(xc: np.ndarray, denom: float, h: int) -> float:
    return float(np.dot(xc[h:], xc[:-h]) / denom)


def _band_z(alpha: float) -> float:
    """PER-LAG two-sided z critical value for the white-noise band. `norm.ppf` (scipy) — no
    hand-rolled inverse-CDF. alpha=0.05 -> 1.95996, i.e. the standard +-1.96/sqrt(n) ACF band.

    NO MULTIPLE-COMPARISONS WIDENING IS APPLIED, AND THAT IS DELIBERATE — an earlier revision of
    this module split a family-wise alpha over every probed lag (Bonferroni), which widened the
    band to z=3.71 (alpha=0.05 over 240 lags) and was FAIL-OPEN for this module's purpose. The
    direction argument, which that revision had inverted:

      * The estimator's output feeds a BLOCK LENGTH. A WIDER band => the ACF is judged "back to
        noise" SOONER => a SHORTER block => a NARROWER interval => LESS coverage. Wide band =
        unsafe direction.
      * What multiple testing on white noise actually inflates here is FALSE LATE crossings (some
        lag exceeds the band by chance, resetting the run), which lengthens the block — the
        CONSERVATIVE direction — and `k_consec` already controls the early-stop risk that a
        family-wise correction was nominally being bought for.
      * Concretely: the 4%-variance persistent component's POPULATION rho(1) = 0.04*0.99 =
        0.03960 at n=3000 sits BETWEEN the standard band (1.96/sqrt(n) = 0.03578) and the
        Bonferroni band (3.71/sqrt(n) = 0.06771) — so the widened band declared real, measurable
        structure to be noise at lag 1. Measured over 1000 trials at n=3000, everything else
        identical: the MEDIAN readout on that family moves 1.0 (Bonferroni band) -> 18.0
        (standard band). Median only — the per-draw spread is p10 = 3.0, p90 = 62.0 over 1000
        draws (`measured_findings.ACF_BAND_READOUT_SPREAD`), so "the statistic sees the
        structure" is a statement about the distribution's centre, not about any one read. The
        interval STILL under-covers there (0.724 +- 0.015, see
        `measured_findings.GATE_II_VERDICT`) — fixing the direction of this band fixes the
        DETECTION, not the candidate.
        Population arithmetic for the same family, since an earlier revision of
        `measured_findings` got the lag count wrong: rho(h) = 0.04*phi^h, so at phi=0.99, n=3000
        it exceeds 1.96/sqrt(n) for 11 lags and 1/sqrt(n) for 78; the ~111-lag figure belongs to
        phi=0.999.

    This also puts the band within rounding distance of `honest_estimator`'s own truncation rule
    (`DEPENDENCE_ACF_N_SE = 2.0`, i.e. 2.0/sqrt(n) = 0.03651), so this module and the tau it is
    meant to replace read the ACF against comparable, not wildly different, noise floors."""
    return float(norm.ppf(1.0 - float(alpha) / 2.0))


def range_acf_band(x: np.ndarray, max_lag: Optional[int] = None, alpha: float = 0.05,
                   k_consec: int = 5, min_n: int = 40) -> Tuple[float, Dict]:
    """CANDIDATE (i). Estimate the dependence range as the first lag `m` such that `k_consec`
    CONSECUTIVE sample autocorrelations all fall inside a white-noise band.

    THE STATED CLAIM: the band is the standard per-lag two-sided white-noise band,
    `z(alpha/2)/sqrt(n)` with alpha=0.05 (z=1.95996) — at a lag where the population
    autocorrelation is truly zero, a single lag falls outside it with probability alpha. NO
    family-wise (Bonferroni) widening is applied; see `_band_z` for why widening is the FAIL-OPEN
    direction for a statistic that feeds a block length. THE `k_consec` REQUIREMENT is the
    control this module actually needs: a SINGLE lag falling inside band by sampling noise
    (which happens routinely even where the population correlation is real but small, e.g. the
    low-amplitude persistent-component case) would make a first-crossing rule stop immediately —
    exactly `dependence_length`'s failure mode (median tau=1.0 on the 4%-variance persistent
    component). Requiring `k_consec` in a row shrinks the chance of a premature stop driven by
    chance multiplicatively (each additional required in-band lag is itself a fresh draw), which
    is what lets this candidate keep scanning through a genuinely long, low-amplitude tail
    instead of declaring "entered" at the first favorable draw.

    WHAT THIS ASSUMES: the dependence has a genuine FINITE reach after which the ACF is
    (statistically) flat at zero — true by construction for the triangular family and false, in
    the limit, for a literal AR(1) (whose ACF never hits exactly zero). In the AR(1)/persistent
    cases this candidate reads out the lag at which the decayed correlation becomes
    indistinguishable from noise AT THIS SAMPLE SIZE, not the population's asymptotic tail — a
    finite-sample notion of range, not the true infinite-support one. See
    `coverage_harness.py`'s measured table for whether that approximation is good enough to
    calibrate a block bootstrap.

    Returns NaN (refuse) when: fewer than `min_n` finite points, a degenerate (zero-variance)
    series, or the run never completes within `max_lag` (dependence unresolved at this n)."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n < min_n:
        return float("nan"), {"reason": f"n={n} < min_n={min_n}"}
    xc = x - x.mean()
    denom = float(np.dot(xc, xc))
    if not np.isfinite(denom) or denom <= 0.0:
        return float("nan"), {"reason": "degenerate (zero-variance) series"}
    if max_lag is None:
        max_lag = int(min(n // 2, 4 * MAX_BLOCK_CAP))
    max_lag = int(max_lag)
    if max_lag < k_consec:
        return float("nan"), {"reason": f"max_lag={max_lag} < k_consec={k_consec}"}
    band = _band_z(alpha) / math.sqrt(n)
    run = 0
    rhos = []
    for h in range(1, max_lag + 1):
        rho = _acf(xc, denom, h)
        if not np.isfinite(rho):
            return float("nan"), {"reason": f"non-finite rho at lag {h}"}
        rhos.append(rho)
        if abs(rho) < band:
            run += 1
            if run >= k_consec:
                m = h - k_consec + 1
                return float(max(1, m)), {"band": band, "max_lag": max_lag, "crossing_lag": m,
                                          "k_consec": k_consec}
        else:
            run = 0
    return float("nan"), {"reason": f"never entered a {k_consec}-run band within max_lag="
                                    f"{max_lag}", "band": band, "last_rhos": rhos[-k_consec:]}


def variance_ratio(x: np.ndarray, q: int) -> float:
    """Lo-MacKinlay-style OVERLAPPING variance ratio: Var(sum of q consecutive obs) /
    (q * Var(1 obs)). VR(q) -> 1 for i.i.d.; VR(q) -> the long-run-variance ratio (same
    asymptotic quantity tau targets) as q grows past the dependence's reach. Biased at finite q
    (no small-sample df correction applied) — disclosed, not hidden; the estimator below reads
    off the SHAPE of the curve, not any one point's exact value."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = int(x.size)
    q = int(q)
    if q < 1 or q >= n:
        return float("nan")
    mu = x.mean()
    var1 = float(np.sum((x - mu) ** 2)) / max(1, n - 1)
    if var1 <= 0.0:
        return float("nan")
    csum = np.concatenate([[0.0], np.cumsum(x - mu)])
    qsum = csum[q:] - csum[:-q]
    m = n - q + 1
    varq = float(np.sum(qsum ** 2)) / (q * m)
    return varq / var1


def first_levelled_index(vr: Sequence[float], plateau_tol: float = 0.10) -> Optional[int]:
    """THE READ-OFF RULE, isolated so it can be tested against a KNOWN curve rather than only
    through sampling noise (see `test_range_probe.py`, which feeds it the exact POPULATION VR
    curve of the 4%-variance persistent component).

    Returns the smallest index `i` such that every LATER point on the curve is within
    `plateau_tol` (relative to `vr[i]`) of `vr[i]` — "the curve has levelled off from here on".
    Returns None when no index qualifies. The LAST index is never returnable: nothing above it
    could confirm it, and treating "ran out of grid" as "plateau" is the unsafe direction (it
    would certify a short block for an unresolved curve).

    THE TOLERANCE IS FIXED WHILE ITS INPUT'S SAMPLING ERROR IS NOT, AND THAT IS A KNOWN, MEASURED,
    UNRESOLVED PROPERTY OF THIS RULE — read this before reading any refusal rate produced by it.
    `plateau_tol` is a constant 0.10 applied at every grid point, but the sampling sd of VR(q)
    grows with q: measured on i.i.d. draws at n=3000 (population VR = 1 at every q, so ALL of the
    dispersion is sampling noise), 1000 draws, sd by grid point q = 2,4,...,256 is
    0.018, 0.034, 0.054, 0.082, 0.118, 0.166, 0.234, 0.316 (each +- <=0.007) — i.e. the noise at
    the tail of the grid is 3.2x the tolerance the rule tests against. Consequences, same draws:
      * the rule refuses on 54.7% +- 1.6% of i.i.d. draws at n=3000, and 90.9% of those refusals
        are driven by the top two grid points (the point furthest from vr[0] in the rule's own
        relative units is q=256 on 455/547 and q=128 on 42/547);
      * when it DOES return on white noise it is largely reading the GRID rather than the data:
        the median returned q is 4 / 16 / 64 at n = 200 / 1000 / 3000 — exactly one quarter of the
        largest grid point in every case — and only 82 of 453 finite returns are the correct q=2.
    THEREFORE: a refusal rate measured through this rule is a property of THE RULE AT THIS GRID
    that has NOT been separated from the variance-ratio statistic itself, and no text in this row
    may call the statistic "conservative" without saying so. The separating experiment is recorded
    (as a COUNTERFACTUAL, deliberately NOT shipped) in `readoff_diagnostics.rule_vs_statistic`:
    holding everything else fixed and making the tolerance aware of the compared point's own
    sampling error (`0.10*|v_i| + 2*sd_iid(q_j)`) finds a plateau on 100.0% of the SAME i.i.d.
    draws and returns the correct q=2 on 96.7% +- 0.6% of them (98.3% / 98.2% at n=200 / 1000).
    Fix cycle 3 chose NOT to adopt that tolerance — see `measured_findings.
    RULE_IS_NOT_THE_STATISTIC` for why (in short: it would be a NEW instrument needing its own
    out-of-family certification, which this row has no budget left to run, and shipping it
    uncertified is exactly the failure mode this row exists to prevent). `plateau_tol` therefore
    remains 0.10, unchanged since the row was built, and the refusal rate remains DISCLOSED as
    rule-and-grid rather than attributed to the statistic."""
    v = [float(t) for t in vr]
    for i in range(0, len(v) - 1):
        base = max(abs(v[i]), 1e-9)
        if all(abs(v[j] - v[i]) / base < plateau_tol for j in range(i + 1, len(v))):
            return i
    return None


def range_variance_ratio(x: np.ndarray, q_grid=None, plateau_tol: float = 0.10,
                         min_n: int = 60) -> Tuple[float, Dict]:
    """CANDIDATE (ii). Compute VR(q) over a geometric grid of q and read the range as the
    SMALLEST q at which the curve has LEVELLED OFF — meaning VR at that q is within
    `plateau_tol` (relative) of VR at EVERY LARGER q on the grid, not merely of its immediate
    predecessor. If no grid point survives that test (the curve is still moving at the largest q
    the sample supports), the dependence's reach is not resolvable at this sample size — REFUSE
    rather than report a number the data cannot support (gate ii's explicit allowed outcome for
    "range >> 1").

    THE ONE-STEP RULE THIS REPLACES WAS A MEASUREMENT DEFECT, NOT A TUNING CHOICE. Through the
    fix-pass of 2026-08-23 this function returned at the FIRST grid point whose relative change
    from the PREVIOUS point was below `plateau_tol` — which trips on any slowly-rising curve and
    is not what this docstring ever claimed. It is deterministic, not a sampling artifact: the
    POPULATION VR of the 4%-variance persistent component at phi=0.99 is VR(2)=1.03960,
    VR(4)=1.11801 — a one-step relative change of 0.0754 < 0.10, so the one-step rule "plateaus"
    at q=4 while the SAME population curve goes on to 1.272, 1.567, 2.113, 3.049, 4.442, 6.062
    (q=256) against an asymptote of 8.92. At phi=0.999 it declares q=4 against an asymptote of
    80.9. The candidate was therefore condemned by its read-off rule rather than by the
    statistic, and the pre-fix verdict blaming "sampling noise mistaken for a plateau" was wrong
    about the mechanism. `plateau_tol` is UNCHANGED at 0.10 across that correction — nothing here
    was tuned until a number cleared.

    WHAT THIS ASSUMES: the VR curve is eventually monotone-non-decreasing-then-flat in q for the
    dependence structures this module is asked to certify. Genuinely non-monotone dependence
    (e.g. strong negative-then-positive ACF) would defeat the plateau read — not tested by any
    synthetic family in `coverage_harness.py`'s validation set, and the REAL per-decision series
    measured in gate (iii) is anti-persistent (VR DECLINING in q), i.e. exactly the untested
    shape. Disclosed as a gap, not closed.

    ALSO ASSUMED, AND THE REASON THE LARGEST GRID POINT IS NOT RETURNABLE: a "plateau" at the last
    grid point has nothing above it to confirm it. Returning it would convert "the curve ran out
    of grid" into a certified short block — the unsafe direction — so it is a REFUSAL, with the
    grid and the curve reported in `info` (`unconfirmed_plateau_q`) so a reader can still see how
    far the curve was still climbing.

    WHAT "REFUSES" MEANS HERE, AND WHAT IT DOES NOT. This function's refusal (NaN) is ONE of the
    channels that produce a refused trial in `coverage_harness.run_coverage_cell`; `block_rule`'s
    `MAX_BLOCK_CAP=60` ceiling is a SEPARATE one, and on the family gate (ii) is about the two are
    not close to equal. Measured, 1000 fresh draws, persistent component phi=0.99 / weight=0.2 /
    n=3000: this function returns NaN on 84.9% +- 1.1% of draws and a FINITE plateau on 15.1% —
    and every one of those finite reads (151/151 in that sample; a fresh seed gave 145/146, so
    read it as essentially all rather than structurally all) is then rejected by the block
    ceiling, not by this rule. "variance_ratio refuses 100% of draws there" is therefore two different components
    refusing, and the second one would reject ANY estimator that read above 60. See
    `measured_findings.REFUSAL_CHANNELS` for the split on every cell. The refusal rate this
    function itself contributes is also a property of the fixed tolerance at this grid (see
    `first_levelled_index`), not a measured property of the variance-ratio statistic."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n < min_n:
        return float("nan"), {"reason": f"n={n} < min_n={min_n}"}
    if q_grid is None:
        max_q = max(2, n // 10)
        q_grid = []
        q = 2
        while q <= max_q:
            q_grid.append(q)
            q *= 2
        if len(q_grid) < 2:
            return float("nan"), {"reason": f"n={n} too small for a >=2-point VR grid"}
    vr = [variance_ratio(x, q) for q in q_grid]
    if any(not np.isfinite(v) for v in vr):
        return float("nan"), {"reason": "non-finite VR at some grid point", "q_grid": q_grid,
                              "vr": vr}
    i = first_levelled_index(vr, plateau_tol)
    if i is not None:
        return float(q_grid[i]), {"q_grid": q_grid, "vr": vr, "plateau_index": i,
                                  "plateau_tol": plateau_tol}
    return float("nan"), {"reason": f"VR never levels off against all larger grid points "
                                    f"(largest q={q_grid[-1]}, n={n}) — range exceeds what this "
                                    f"sample can resolve",
                          "q_grid": q_grid, "vr": vr, "unconfirmed_plateau_q": q_grid[-1],
                          "plateau_tol": plateau_tol}


def range_spectral_zero(x: np.ndarray, min_n: int = 40,
                        max_bandwidth_frac: float = 0.25) -> Tuple[float, Dict]:
    """CANDIDATE (iii). Andrews (1991) AR(1) plug-in automatic bandwidth for a Bartlett-kernel
    long-run-variance (spectral-density-at-zero) estimator:

        bw = 1.1447 * (a1_hat * n) ** (1/3),   a1_hat = 4*rho1^2 / (1 - rho1^2)^2

    read DIRECTLY as a range estimate in decision units (the bandwidth IS the reach the kernel
    gives non-negligible weight to). The bandwidth is a MONOTONE FUNCTION OF `rho1` ALONE (times
    n^(1/3)): it grows without bound as |rho1| -> 1 and shrinks to ~1.1447*n^(1/3)*small as
    rho1 -> 0.

    THAT MONOTONICITY IS THE CANDIDATE'S WHOLE WEAKNESS AND AN EARLIER REVISION OF THIS DOCSTRING
    HAD IT BACKWARDS. It claimed the plug-in reads "a near-unit-root series as needing a LARGE
    bandwidth even though rho1 may be small in absolute terms" — impossible: the formula sees
    nothing but rho1, so a SMALL rho1 always gives a SMALL bandwidth, whatever the series'
    persistence actually is. That is precisely why this candidate fails on the 4%-variance
    persistent component, whose population rho1 is 0.0396 (near-unit-root persistence carried at
    tiny amplitude): measured median bandwidth 2.92 at phi=0.99, 2.01 at phi=0.999.

    WHAT THIS ASSUMES: the AR(1) plug-in is a reasonable proxy for the spectral shape — exact for
    a true AR(1), approximate otherwise (a standard, widely used approximation in the HAC
    literature, not a claim this module derives fresh). REFUSES (NaN, per this module's
    refusal contract) when the selected bandwidth exceeds `max_bandwidth_frac * n` — a bandwidth
    that large is not supportable at this sample size (gate ii's "insufficient-n" escape hatch);
    the offending bandwidth is reported in `info["bandwidth"]`, never as the return value."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n < min_n:
        return float("nan"), {"reason": f"n={n} < min_n={min_n}"}
    xc = x - x.mean()
    denom = float(np.dot(xc[:-1], xc[:-1]))
    if not np.isfinite(denom) or denom <= 0.0:
        return float("nan"), {"reason": "degenerate (zero-variance) series"}
    rho1 = float(np.dot(xc[1:], xc[:-1]) / denom)
    rho1 = float(np.clip(rho1, -0.98, 0.98))
    a1 = (4.0 * rho1 ** 2) / max(1e-12, (1.0 - rho1 ** 2) ** 2)
    bw = 1.1447 * (a1 * n) ** (1.0 / 3.0)
    bw = max(1.0, bw)
    if bw > max_bandwidth_frac * n:
        # NaN, not the bandwidth: `info["refuse"]` is advisory, but every caller in this module
        # (and `block_rule` itself) keys off the RETURN VALUE. Returning a finite number with a
        # refusal flag let a refusal silently become a certification for any caller that
        # estimated the range on a sub-window and then applied `block_rule` at a larger n_dec
        # (where the same bandwidth no longer trips the ceiling). Unreachable in this row's own
        # sweep -- closed anyway, because the contract at the module docstring says NaN.
        return float("nan"), {"bandwidth": bw,
                    "reason": f"selected bandwidth {bw:.1f} exceeds "
                              f"{max_bandwidth_frac:.2f}*n={max_bandwidth_frac * n:.1f} — not "
                              "supportable at this sample size", "rho1": rho1, "a1": a1,
                    "refuse": True}
    return float(bw), {"rho1": rho1, "a1": a1}


ESTIMATORS = {
    "acf_band": range_acf_band,
    "variance_ratio": range_variance_ratio,
    "spectral_zero": range_spectral_zero,
}


@dataclass
class BlockRuleResult:
    block: Optional[int]
    k: Optional[float]
    refused: bool
    reason: str = ""
    range_est: float = float("nan")
    calibration: Optional[Dict] = None


_K_CACHE: Dict[Tuple[int, int, int], Dict] = {}


def calibrate_k(block: int, n_dec: int, target: float = TARGET_COVERAGE,
                n_trials_calib: int = 200, n_trials_valid: int = 200, n_boot: int = 300,
                seed: int = 0, k_cap: float = 8.0, calibration_margin: float = 0.03) -> Optional[Dict]:
    """Calibrate a width-inflation factor `k` for THIS module's chosen `block`, at THIS `n_dec`,
    via the SAME closed-form method `experiments/897_corrected_809/calibration.py` certified for
    its own (n_dec-derived) block choice — `cal897.k_needed_for_trial` is imported and reused
    verbatim, never re-derived. What differs from 897's `calibrate_block` is the stratification
    key: 897 derives block FROM n_dec (`n_dec // 20`, capped); this module's block comes from a
    RANGE ESTIMATE, so the calibration here is parameterized by `block` directly.

    Calibration draws use `cal897.simulate_overlapping(n_dec, dep=block, ...)` — i.e. dep==block,
    the SAME "matched-dependence" convention 897's own table uses (disclosed there as the EASIEST
    point of the dependence sweep, not a worst case — the same caveat applies here: this `k` is
    conditional on the certified range being an accurate read of the data's real dependence, which
    is exactly what `coverage_harness.py` measures out-of-family).

    Calibrate-then-validate on an INDEPENDENT batch, same discipline as 892/897: the percentile is
    taken at `target + calibration_margin` on the calibration batch and re-measured with NO margin
    on a fresh validation batch before being reported as achieving `target`.

    THIS IS A FOURTH REFUSAL CHANNEL, AND FIX CYCLE 3 MEASURED HOW MUCH OF THE SWEEP IT MOVED. The
    prewarmed table the sweep consumed was never written to `RESULTS.json`, so this was invisible;
    `calib_table.py` dumps it (`CALIB_TABLE.json`, identical parameters, so it reproduces the
    table the sweep used). EXACTLY ONE of the 27 (block, n_dec) strata the sweep touched fails:
    block=3, n_dec=3000, validation-batch coverage 0.925 against target 0.930 — a miss of 0.005 on
    a 200-trial batch whose binomial SE is 0.019, i.e. statistically indistinguishable from the
    bar, but `achieved` is a hard >= so every trial that lands on block=3 at n_dec=3000 refuses.
    That single stratum is the ENTIRE `calibration_refusal` column in
    `measured_findings.REFUSAL_CHANNELS`, and it is not small: it is 31.5% / 31.7% of
    spectral_zero's draws on the persistent component (phi=0.99 / 0.999), 21.8% on i.i.d. at
    n=3000, and 4.2% / 2.9% / 5.2% of acf_band's on those same cells. Those cells' `certify_rate`
    is therefore NOT a statement about the estimator declining to read the data. It does not touch
    any COVERAGE number (coverage is conditional on certification, and the refused draws were
    never scored), and it is disclosed rather than tuned away — raising `n_trials_valid` until
    block=3 passes would be tuning a threshold to change a number, which this row does not do."""
    key = (int(block), int(n_dec), int(seed))
    if key in _K_CACHE:
        return _K_CACHE[key]
    rng = np.random.default_rng(seed)
    calib_pctl = min(0.999, target + calibration_margin)
    ks = []
    for _ in range(int(n_trials_calib)):
        r = cal897.simulate_overlapping(n_dec, block, rng)
        sh = hest.sharpe(r, 1.0)
        lo, hi = hest.block_bootstrap_sharpe_ci(r, 1.0, block, n_boot,
                                                seed=int(rng.integers(0, 2 ** 31 - 1)))
        if not (np.isfinite(sh) and np.isfinite(lo) and np.isfinite(hi)):
            continue
        ks.append(cal897.k_needed_for_trial(sh, lo, hi))
    if not ks:
        out = {"block": block, "n_dec": n_dec, "k": None, "achieved": False,
              "reason": "no calibration trial produced a usable CI at this (block, n_dec)"}
        _K_CACHE[key] = out
        return out
    k_uncapped = float(np.percentile(ks, 100.0 * calib_pctl, method="higher"))
    if k_uncapped > k_cap:
        out = {"block": block, "n_dec": n_dec, "k": None, "k_uncapped": k_uncapped,
              "achieved": False,
              "reason": f"calibration percentile needed k={k_uncapped:.2f} > k_cap={k_cap} — "
                        "refused as 'not a calibration anymore'"}
        _K_CACHE[key] = out
        return out
    k = k_uncapped
    hits, usable = 0, 0
    for _ in range(int(n_trials_valid)):
        r = cal897.simulate_overlapping(n_dec, block, rng)
        sh = hest.sharpe(r, 1.0)
        lo, hi = hest.block_bootstrap_sharpe_ci(r, 1.0, block, n_boot,
                                                seed=int(rng.integers(0, 2 ** 31 - 1)))
        if not (np.isfinite(sh) and np.isfinite(lo) and np.isfinite(hi)):
            continue
        usable += 1
        clo, chi = sh - k * (sh - lo), sh + k * (hi - sh)
        hits += int(clo <= 0.0 <= chi)
    cov = (hits / usable) if usable else None
    out = {"block": int(block), "n_dec": int(n_dec), "k": float(k), "k_uncapped": k_uncapped,
          "validation_coverage": cov, "validation_n": usable,
          "achieved": bool(cov is not None and cov >= target)}
    if not out["achieved"]:
        out["reason"] = (f"validation-batch coverage {cov!r} < target {target} at "
                         f"block={block}, n_dec={n_dec}, k={k:.3f}")
    else:
        out["reason"] = ""
    _K_CACHE[key] = out
    return out


def _calibrate_one(args) -> Tuple[Tuple[int, int, int], Dict]:
    """Top-level (picklable) worker for `prewarm_calibration_table`'s process pool."""
    block, n_dec, kwargs = args
    return (int(block), int(n_dec), int(kwargs.get("seed", 0))), calibrate_k(block, n_dec, **kwargs)


def prewarm_calibration_table(blocks: Sequence[int] = BLOCK_STRATA,
                              n_decs: Sequence[int] = (), n_jobs: int = 1,
                              **calib_kwargs) -> Dict[Tuple[int, int, int], Dict]:
    """Populate `_K_CACHE` for every (block, n_dec) pair in `blocks` x `n_decs` BEFORE running a
    coverage sweep, so the sweep's many trials only ever hit the cache (a fresh calibration is a
    calibrate-then-validate pass costing seconds; paying it once per DISTINCT (block, n_dec) pair
    rather than once per trial is what keeps `coverage_harness.py` affordable). `n_jobs > 1` uses
    a process pool (CPU-only, changes wall clock only) — each worker computes independently and
    results are merged into THIS process's `_K_CACHE` on return (a child process's cache updates
    would otherwise be invisible to the parent)."""
    jobs = [(b, nd, dict(calib_kwargs)) for b in blocks for nd in n_decs]
    if n_jobs <= 1 or len(jobs) <= 1:
        results = {}
        for b, nd, kw in jobs:
            key, val = _calibrate_one((b, nd, kw))
            results[key] = val
        _K_CACHE.update(results)
        return results
    from concurrent.futures import ProcessPoolExecutor
    results = {}
    with ProcessPoolExecutor(max_workers=int(n_jobs)) as ex:
        for key, val in ex.map(_calibrate_one, jobs):
            results[key] = val
    _K_CACHE.update(results)
    return results


def block_rule(n_dec: int, range_est: float, target: float = TARGET_COVERAGE,
              **calib_kwargs) -> BlockRuleResult:
    """The certified quantity -> a block length + calibrated width factor, or a refusal.

    Refuses when: the range estimate itself is non-finite (the estimator already refused);
    the resulting block would leave fewer than `MIN_NONOVERLAP_BLOCKS` non-overlapping blocks in
    `n_dec` decisions; or the width-factor calibration itself does not achieve `target` coverage
    (`calibrate_k`, above)."""
    n_dec = int(n_dec)
    if not np.isfinite(range_est) or range_est < 1.0:
        return BlockRuleResult(None, None, True,
                               "range estimator refused or returned a non-finite/sub-1 range",
                               range_est=range_est)
    if math.ceil(range_est) > MAX_BLOCK_CAP:
        # D55's certified-ceiling pattern, applied here rather than only cited: a range estimate
        # above what this module's calibration table can reach must REFUSE, never be silently
        # capped and calibrated as if the cap were the truth. Capping silently is exactly the
        # extrapolation this repo's own doctrine forbids (gate (ii)'s "reads range >> 1 and
        # EITHER calibrates correctly OR refuses" — a range this large is the REFUSE branch).
        #
        # THIS BRANCH IS NOT A DETAIL: IT IS A LOAD-BEARING PART OF EVERY "REFUSES" NUMBER THIS
        # ROW REPORTS, AND IT IS A PROPERTY OF THE CEILING, NOT OF ANY ESTIMATOR. Measured over
        # 1000 fresh draws per cell (`measured_findings.REFUSAL_CHANNELS`): it accounts for
        # 100% of spectral_zero's refusals on every triangular cell at dep >= 15 and on all three
        # pure AR(1) cells (the estimator itself never returned NaN there — it returned 73-223 and
        # the ceiling rejected it), 61.1% of acf_band's on ar1 phi=0.99, 31.3% of
        # variance_ratio's refusals on i.i.d. at n=3000 (249 of 796 refused draws; that cell is
        # the one where the two denominators differ, because it certifies 20.4% — as a share of
        # DRAWS the cap is 24.9%), and 15.1% of draws on the persistent component at phi=0.99 —
        # where it is 100% of the draws the estimator did resolve. Any reading of the
        # form "estimator X refuses on family Y" must say which share is this ceiling, because
        # this ceiling would refuse ANY estimator reading above 60, tau included.
        return BlockRuleResult(MAX_BLOCK_CAP, None, True,
                               f"range estimate {range_est:.1f} exceeds this module's "
                               f"calibratable ceiling MAX_BLOCK_CAP={MAX_BLOCK_CAP} — refusing "
                               "rather than extrapolating (D55 house form)",
                               range_est=range_est)
    block = snap_block(range_est)
    n_blocks = n_dec // block
    if n_blocks < MIN_NONOVERLAP_BLOCKS:
        return BlockRuleResult(block, None, True,
                               f"only {n_blocks} non-overlapping blocks at block={block} "
                               f"(n_dec={n_dec}) — need >= {MIN_NONOVERLAP_BLOCKS}",
                               range_est=range_est)
    calib = calibrate_k(block, n_dec, target=target, **calib_kwargs)
    if not calib.get("achieved"):
        return BlockRuleResult(block, None, True,
                               calib.get("reason", "width-factor calibration did not achieve "
                                                    "target coverage"),
                               range_est=range_est, calibration=calib)
    return BlockRuleResult(block, float(calib["k"]), False, "", range_est=range_est,
                           calibration=calib)
