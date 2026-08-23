#!/usr/bin/env python3
"""TASK_901 — Monte Carlo coverage measurement harness (the certification itself).

Every cell below runs a TRUE NULL (population Sharpe exactly 0 for every synthetic family, by
`families.py`'s construction): generate a series, run ONE candidate range estimator, feed the
result through `range_probe.block_rule` to get a block length + calibrated width factor (or a
refusal), and — for non-refused trials — check whether the k-inflated bootstrap interval contains
0. Coverage is reported CONDITIONAL ON CERTIFICATION (hits / non-refused trials), alongside the
refusal rate, because refusal is a first-class, disclosed outcome here (D57's whole point): a
family that mostly refuses is not silently folded into a coverage number that hides it.

Every reported coverage carries its own Monte Carlo standard error,
`se = sqrt(p*(1-p)/n_certified)` — a coverage claim with no error bar cannot support a 93%
threshold, which is why every printed/JSON cell carries `se` beside `coverage`.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[2]
_897 = _REPO / "experiments" / "897_corrected_809"
if str(_897) not in sys.path:
    sys.path.insert(0, str(_897))

sys.path.insert(0, str(_THIS.parent))
import honest_estimator as hest   # noqa: E402  (READ-ONLY import)
import range_probe as rp          # noqa: E402
import families as fam            # noqa: E402


def mc_se(p: Optional[float], n: int) -> Optional[float]:
    """Binomial Monte Carlo standard error of a coverage fraction."""
    if p is None or n <= 0:
        return None
    return float(math.sqrt(max(0.0, p * (1.0 - p)) / n))


def run_coverage_cell(estimator_name: str, family_name: str, n_dec: int, family_kwargs: Dict,
                      n_trials: int, seed: int, n_boot_ci: int = 300,
                      calib_kwargs: Optional[Dict] = None) -> Dict:
    """One (estimator, family, n_dec, family_kwargs) cell: `n_trials` independent true-null
    draws, each independently estimating range, deriving a block rule, and (if not refused)
    checking containment of 0 in the k-inflated bootstrap interval.

    `n_refused` MERGES FIVE DIFFERENT EVENTS AND MUST NOT BE READ AS "THE ESTIMATOR REFUSED"
    (fix cycle 3). It is incremented when the estimator returns NaN, when `block_rule` rejects a
    FINITE range above MAX_BLOCK_CAP=60, when the MIN_NONOVERLAP_BLOCKS floor binds, when the
    width calibration for that (block, n_dec) did not achieve target — and, FIFTH, at the
    `not (isfinite(sh) and isfinite(lo) and isfinite(hi))` guard below, where a bootstrap that
    never produced a usable CI is counted as refused rather than scored either way (897's rule).
    That fifth branch MEASURED ZERO: no chunk in any of the nine real-leg runs recorded
    `unusable_bootstrap` (228 per-chunk records, 196 certified / 32 refused, 0 unusable), and it
    is the reason `readoff_diagnostics.refusal_channels` — which draws no bootstrap and therefore
    has no counterpart for it — reconciles a FOUR-way split against this FIVE-branch counter to
    within Monte Carlo error on all 63 cells. Named here so the enumeration is complete rather
    than convenient. On the family gate (ii) is about, the cap channel carries 100% of the
    draws `variance_ratio` actually resolved, and on most `spectral_zero` cells whose
    `certify_rate` reads 0.000 the estimator never refused at all (it returned 73-223 and the
    ceiling rejected it). The split is NOT reconstructible from this function's output; it is
    measured separately by `readoff_diagnostics.refusal_channels` (READOFF_DIAG.json) and pinned
    in `measured_findings.REFUSAL_CHANNELS`. This function is left as it is — the sweep it
    produced is not being re-run, and changing its return shape now would invalidate the
    already-measured RESULTS.json rather than add information."""
    estimator = rp.ESTIMATORS[estimator_name]
    generator: Callable = fam.FAMILIES[family_name]
    calib_kwargs = calib_kwargs or {}
    rng = np.random.default_rng(seed)
    hits = 0
    certified = 0
    refused = 0
    ranges = []
    t0 = time.time()
    for _ in range(int(n_trials)):
        x = generator(n_dec, rng=rng, **family_kwargs)
        range_est, _info = estimator(x)
        ranges.append(range_est)
        rule = rp.block_rule(n_dec, range_est, **calib_kwargs)
        if rule.refused:
            refused += 1
            continue
        sh = hest.sharpe(x, 1.0)
        lo, hi = hest.block_bootstrap_sharpe_ci(x, 1.0, rule.block, n_boot_ci,
                                                seed=int(rng.integers(0, 2 ** 31 - 1)))
        if not (np.isfinite(sh) and np.isfinite(lo) and np.isfinite(hi)):
            refused += 1  # a bootstrap that never ran cannot be scored either way (897's rule)
            continue
        certified += 1
        clo, chi = sh - rule.k * (sh - lo), sh + rule.k * (hi - sh)
        hits += int(clo <= 0.0 <= chi)
    coverage = (hits / certified) if certified else None
    finite_ranges = [r for r in ranges if np.isfinite(r)]
    return {
        "estimator": estimator_name, "family": family_name, "n_dec": int(n_dec),
        "family_kwargs": family_kwargs, "n_trials": int(n_trials),
        "n_certified": certified, "n_refused": refused,
        "certify_rate": certified / n_trials if n_trials else None,
        "coverage": coverage, "coverage_se": mc_se(coverage, certified),
        "median_range_est": float(np.median(finite_ranges)) if finite_ranges else None,
        "runtime_s": time.time() - t0,
    }


def real_data_chunk_containment(reb_ret: np.ndarray, estimator_name: str, chunk_size: int = 1000,
                                n_boot_ci: int = 300, seed: int = 0,
                                calib_kwargs: Optional[Dict] = None) -> Dict:
    """THE D57 HALT'S CONSTRUCTION, re-run here against this module's OWN estimator instead of
    tau: split the real per-decision series into disjoint `chunk_size`-decision chunks, compute
    the GRAND (whole-series) Sharpe as the pseudo-truth, then for each chunk independently
    estimate range -> block rule -> k-inflated CI and check whether it contains the grand Sharpe.
    READ THIS BEFORE READING THE NUMBER — WHAT IT ACTUALLY MEASURES (fix cycle 2, 2026-08-24).
    This construction assumes the population per-decision Sharpe is CONSTANT across the window, so
    that the whole-series Sharpe is a legitimate pseudo-truth for every chunk. On this row's real
    series that assumption is MEASURED FALSE (`stationarity_diagnostics` below: between-chunk
    Sharpe dispersion is 6.4-7.2x the within-chunk bootstrap SE, chi2_25 ~ 200, p < 1e-20). THE
    FINDING IS EXCESS DISPERSION, NOT A TREND — an earlier revision of this docstring said
    "monotone drift across the window" and that is STRUCK (fix cycle 3): refit with standard
    errors the slope is -0.00386 +- 0.00241 at H=1 (p = 0.122), weaker at H=5/20, and detrending
    removes essentially none of the dispersion (`measured_findings.DRIFT_REFIT`). Under a
    population Sharpe that varies across the window for ANY reason, containment is bounded by that
    variation, NOT by the block length — and the within-chunk bootstrap SE is measured FLAT in
    block (1-7% from block 2 to block 60), so no range estimator, and no oracle range estimator
    either, can move this number materially. THEREFORE:
      * `contains_grand_sharpe_fraction` is reported as a measurement of the SERIES (its
        non-stationarity), NOT as evidence for or against any range estimator. An earlier
        revision of this row read it as an indictment of the estimators (and the D57 halted cycle
        used the same construction to indict tau); that reading is withdrawn — see
        `measured_findings.GATE_III_NONSTATIONARITY`.
      * `certify_rate` — the fraction of chunks the estimator was willing to certify (not refuse)
        — is reported beside it as a plain descriptive number. There is NO well-formed pass/fail
        comparison between the two (an earlier docstring here promised a "certify_rate among
        CONTAINING chunks" criterion checked by `run_validation.py`; no such check ever existed
        there, and the criterion as worded is not well formed — a certification rate and a
        containment fraction are not on the same scale). Removed rather than back-filled.
    A chunk that refuses is EXCLUDED from `contains_grand_sharpe_fraction`'s denominator (a
    refused chunk was never certified in the first place, so it cannot be scored as "contained"
    or "missed") and reported separately as `n_refused`. The reported SE is the plain binomial
    one, which treats the chunks as INDEPENDENT draws — the measured excess dispersion says they
    are not identically distributed, so that SE is a floor on the true uncertainty, not an honest
    interval.
    """
    reb_ret = np.asarray(reb_ret, dtype=np.float64)
    reb_ret = reb_ret[np.isfinite(reb_ret)]
    n = reb_ret.size
    grand_sharpe = hest.sharpe(reb_ret, 1.0)
    n_chunks = n // chunk_size
    calib_kwargs = calib_kwargs or {}
    estimator = rp.ESTIMATORS[estimator_name]
    rng = np.random.default_rng(seed)
    hits, certified, refused = 0, 0, 0
    ranges = []
    per_chunk = []
    for c in range(n_chunks):
        chunk = reb_ret[c * chunk_size:(c + 1) * chunk_size]
        range_est, _info = estimator(chunk)
        ranges.append(range_est)
        rule = rp.block_rule(chunk_size, range_est, **calib_kwargs)
        if rule.refused:
            refused += 1
            per_chunk.append({"chunk": c, "status": "refused", "reason": rule.reason})
            continue
        sh = hest.sharpe(chunk, 1.0)
        lo, hi = hest.block_bootstrap_sharpe_ci(chunk, 1.0, rule.block, n_boot_ci,
                                                seed=int(rng.integers(0, 2 ** 31 - 1)))
        if not (np.isfinite(sh) and np.isfinite(lo) and np.isfinite(hi)):
            refused += 1
            per_chunk.append({"chunk": c, "status": "unusable_bootstrap"})
            continue
        certified += 1
        clo, chi = sh - rule.k * (sh - lo), sh + rule.k * (hi - sh)
        contained = bool(clo <= grand_sharpe <= chi)
        hits += int(contained)
        per_chunk.append({"chunk": c, "status": "certified", "block": rule.block, "k": rule.k,
                          "contains_grand_sharpe": contained})
    finite_ranges = [r for r in ranges if np.isfinite(r)]
    contains_frac = (hits / certified) if certified else None
    return {
        "n_dec_total": int(n), "grand_sharpe": float(grand_sharpe) if np.isfinite(grand_sharpe)
        else None,
        "n_chunks": int(n_chunks), "chunk_size_decisions": int(chunk_size),
        "n_certified": certified, "n_refused": refused,
        "certify_rate": certified / n_chunks if n_chunks else None,
        "contains_grand_sharpe_fraction": contains_frac,
        "contains_grand_sharpe_se": mc_se(contains_frac, certified),
        "median_range_est": float(np.median(finite_ranges)) if finite_ranges else None,
        "estimator": estimator_name, "per_chunk": per_chunk,
        "se_note": "binomial SE, treats chunks as independent draws from ONE population; "
                   "the measured excess dispersion (stationarity_diagnostics) says they are "
                   "not — this is a floor.",
        "reading": "a property of the SERIES (non-stationarity), not evidence about the "
                   "estimator — see stationarity_diagnostics and the docstring above.",
    }


def paired_containment_on_common_chunks(cc_a: Dict, cc_b: Dict) -> Dict:
    """THE DISCRIMINATION TEST FOR GATE (iii). Two estimators' chunk-containment runs over the
    SAME series contain two confounded differences: which chunks each was willing to certify, and
    what block length each chose. Restricting to the chunks BOTH certified removes the selection
    difference and leaves only the block length — so if containment is IDENTICAL on the common
    chunks while the blocks differ by an order of magnitude, the construction is measuring
    something the block length does not control, and cannot rank estimators.

    Takes two `real_data_chunk_containment` results; returns the common-chunk counts and the
    distinct blocks each estimator used there."""
    a = {c["chunk"]: c for c in cc_a["per_chunk"] if c["status"] == "certified"}
    b = {c["chunk"]: c for c in cc_b["per_chunk"] if c["status"] == "certified"}
    common = sorted(set(a) & set(b))
    return {
        "estimator_a": cc_a["estimator"], "estimator_b": cc_b["estimator"],
        "n_common_chunks": len(common),
        "a_contains": sum(bool(a[c]["contains_grand_sharpe"]) for c in common),
        "b_contains": sum(bool(b[c]["contains_grand_sharpe"]) for c in common),
        "a_blocks": sorted({int(a[c]["block"]) for c in common}),
        "b_blocks": sorted({int(b[c]["block"]) for c in common}),
        "identical_per_chunk": [c for c in common
                                if bool(a[c]["contains_grand_sharpe"])
                                != bool(b[c]["contains_grand_sharpe"])] == [],
    }


_Z95 = 1.959963984540054   # two-sided 95% normal quantile — the percentile pair below IS 2.5/97.5


def _boot_se(x: np.ndarray, block: int, n_boot: int, seed: int) -> float:
    """Standard error of the annualized Sharpe from THE SAME circular block bootstrap the rest of
    this row uses (`honest_estimator.block_bootstrap_sharpe_ci`, imported — never a second,
    hand-rolled resampler), read off its 2.5/97.5 percentile pair as (hi-lo)/(2*z_.975). A
    normal-shape assumption on the bootstrap distribution, disclosed: it is used ONLY as a scale
    for comparing WITHIN-chunk uncertainty against BETWEEN-chunk dispersion, never to build an
    interval anyone reads."""
    lo, hi = hest.block_bootstrap_sharpe_ci(x, 1.0, int(block), int(n_boot), seed=int(seed))
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return float("nan")
    return float((hi - lo) / (2.0 * _Z95))


def stationarity_diagnostics(reb_ret: np.ndarray, chunk_size: int = 1000,
                             blocks: Sequence[int] = (2, 3, 5, 8, 12, 18, 26, 38, 60),
                             chunk_vr_q: Sequence[int] = (2, 4, 8, 16, 32, 64),
                             series_vr_q: Sequence[int] = (2, 8, 32, 128, 512, 2048),
                             ref_block: int = 60, n_boot: int = 1000, seed: int = 0) -> Dict:
    """GATE (iii), RE-SCOPED (fix cycle 2). Asks what the chunk-containment construction can
    actually discriminate, and answers it with three measurements that need no estimator at all:

      1. `boot_se_by_block` — the mean within-chunk block-bootstrap SE of the Sharpe, as a
         function of the block length, over the SAME block strata `range_probe.BLOCK_STRATA`
         offers. If this is FLAT in block, then the block length — the ONLY thing a range
         estimator controls — cannot move the interval, and containment cannot be evidence about
         a range estimator: a perfect oracle range estimator would produce the same number.
      2. `chunk_vr_mean` / `series_vr` — the variance ratio inside chunks and across the whole
         series. VR RISING in q is positive dependence (the case where too-short a block would
         under-cover, i.e. the case this row was built for). VR FALLING below 1 is
         ANTI-persistence, under which a block bootstrap is if anything CONSERVATIVE — the
         opposite of the failure mode being alleged.
      3. `chunk_sharpes` + `between_within_variance_ratio` + `chi2` — is the population
         per-decision Sharpe even CONSTANT across the window? Under a constant population Sharpe
         and honest within-chunk SEs, the chunk Sharpes should scatter about the grand Sharpe with
         dispersion ~ the within-chunk SE. If between-chunk dispersion is many times the
         within-chunk SE, the pseudo-truth the containment check scores against is not a single
         number, and no interval-width fix can raise containment.

    Every number here is descriptive of the SERIES. Nothing in this function calls a range
    estimator, by design.
    """
    reb_ret = np.asarray(reb_ret, dtype=np.float64)
    reb_ret = reb_ret[np.isfinite(reb_ret)]
    n = int(reb_ret.size)
    n_chunks = n // int(chunk_size)
    chunks = [reb_ret[c * chunk_size:(c + 1) * chunk_size] for c in range(n_chunks)]
    grand = float(hest.sharpe(reb_ret, 1.0))
    chunk_sharpes = [float(hest.sharpe(c, 1.0)) for c in chunks]

    boot_se_by_block = {}
    for b in blocks:
        ses = [_boot_se(c, b, n_boot, seed + 1000 * int(b) + i) for i, c in enumerate(chunks)]
        ses = [s for s in ses if np.isfinite(s)]
        boot_se_by_block[int(b)] = float(np.mean(ses)) if ses else None

    ref_ses = [_boot_se(c, ref_block, n_boot, seed + 7_000_000 + i) for i, c in enumerate(chunks)]
    ref_ses_f = np.array([s for s in ref_ses if np.isfinite(s)], dtype=np.float64)
    sh = np.array(chunk_sharpes, dtype=np.float64)
    between_sd = float(np.std(sh, ddof=1)) if sh.size > 1 else float("nan")
    within_se = float(np.mean(ref_ses_f)) if ref_ses_f.size else float("nan")
    ratio = float((between_sd / within_se) ** 2) if within_se and np.isfinite(within_se) else None
    chi2_stat = float(np.sum(((sh - sh.mean()) / np.array(ref_ses, dtype=np.float64)) ** 2))
    df = int(sh.size - 1)
    try:
        from scipy.stats import chi2 as _chi2
        p_value = float(_chi2.sf(chi2_stat, df))
    except Exception:                                   # scipy is a hard dep elsewhere; be safe
        p_value = None

    idx = np.arange(sh.size, dtype=np.float64)
    slope = float(np.polyfit(idx, sh, 1)[0]) if sh.size > 2 else None

    return {
        "n_dec_total": n, "n_chunks": n_chunks, "chunk_size_decisions": int(chunk_size),
        "grand_sharpe": grand, "chunk_sharpes": chunk_sharpes,
        "boot_se_by_block": boot_se_by_block, "n_boot": int(n_boot), "ref_block": int(ref_block),
        "boot_se_flat_ratio": (
            float(boot_se_by_block[int(blocks[-1])] / boot_se_by_block[int(blocks[0])])
            if boot_se_by_block.get(int(blocks[0])) else None),
        "chunk_vr_mean": {int(q): float(np.mean([rp.variance_ratio(c, int(q)) for c in chunks]))
                          for q in chunk_vr_q},
        "series_vr": {int(q): float(rp.variance_ratio(reb_ret, int(q))) for q in series_vr_q
                      if int(q) < n},
        "between_chunk_sd": between_sd, "within_chunk_se_mean": within_se,
        "between_within_variance_ratio": ratio,
        "chi2_stat": chi2_stat, "chi2_df": df, "chi2_p_value": p_value,
        "chunk_sharpe_slope_per_chunk": slope,
    }
