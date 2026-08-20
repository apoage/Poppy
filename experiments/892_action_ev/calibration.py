#!/usr/bin/env python3
"""TASK_892 D51 re-cut — Amendment 1's pre-registered calibration harness (SPEC.md "Interval
method"). D49's escalation measured the pre-registered block bootstrap (block=21, 10k resamples)
UNDER-CALIBRATED under realistic overlapping dependence: coverage plateaus ~88% and NO SAMPLE
SIZE closes it, because the block LENGTH equals the dependence LENGTH (consecutive forward-return
windows overlap by up to H-1=20). Amendment 1 pre-registers a remedy ladder, in order:
  (i)   block length > dependence length, where n permits
  (ii)  subsampled non-overlapping event windows at stride H
  (iii) a calibrated-width correction
On the OOS binding leg (116 decisions, ~5-6 full blocks of 21), (i) is arithmetically impossible
(you cannot have both block > 21 and >= a useful block count from 116 decisions) and (ii) leaves
~5 independent observations — too few for ANY interval method to certify 93% coverage without an
enormous width. This module implements (iii): measure, by Monte Carlo through the REAL
``report.block_bootstrap_delta_ci`` code path (never a re-implementation), the WIDTH-INFLATION
FACTOR ``k`` needed at each observed block count so that the interval
    [delta - k*(delta-ci_lo_raw), delta + k*(ci_hi_raw-delta)]
achieves >= TARGET_COVERAGE two-sided coverage under a TRUE NULL (independent sides, delta=0),
in BOTH a i.i.d. scenario and the OVERLAPPING-dependence scenario disclosures.py already uses
(each observation a rolling sum of H iid innovations — the autocorrelation structure real
overlapping-horizon forward returns have). Amendment 1 requires certification against the
REALISTIC-DEPENDENCE (overlapping) scenario specifically; the i.i.d. number is carried alongside
for disclosure, exactly as the pre-existing NULL_COVERAGE_TABLE/_OVERLAPPING pair already does.

WHY A CLOSED-FORM PERCENTILE OF ``k_needed``, NOT A GRID SEARCH: for a fixed trial (delta, ci_lo,
ci_hi from ONE real bootstrap run), the inflated interval's containment of 0 is MONOTONIC in k (a
wider interval can only gain containment, never lose it) — so the coverage-at-k curve is exactly
the empirical CDF of the per-trial "k needed to just barely contain 0", and the SMALLEST k that
achieves a target coverage over N trials is the TARGET-th percentile of that per-trial quantity.
This computes exactly the same answer as a grid/bisection search would, without the search.

HONESTY DISCIPLINE (the CRITICAL HONESTY CONSTRAINT the task brief names): the calibrated k is
found on a CALIBRATION batch of trials and then RE-MEASURED on an INDEPENDENT VALIDATION batch
(never the same draws) before being reported as "achieves coverage C" — reporting the in-sample
percentile as its own validation would be circular. If the validation measurement comes back
below TARGET_COVERAGE (possible: percentile estimation noise, or a block count outside anything
this harness can honestly calibrate), the stratum is NOT bandable at that k and must be reported
as such — never silently rounded up.

A calibrated k is only as good as the scenario it was calibrated against. THE OVERLAPPING
SCENARIO IS A SYNTHETIC MODEL (i.i.d. innovations summed over a rolling window) — real forward
returns additionally carry fat tails, heteroskedasticity, cross-sectional correlation and regime
structure that this model does not. Every artifact that reports a calibrated k must disclose this
inherited-scenario caveat; it is NOT eliminated by calibrating, only relocated from "coverage is
~88%" to "coverage is nominal UNDER THIS DEPENDENCE MODEL" (see
disclosures.CALIBRATED_WIDTH_DISCLOSURE).
"""
import os
import sys

import numpy as np

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
if EXP_DIR not in sys.path:
    sys.path.insert(0, EXP_DIR)

import report as report_mod                                   # noqa: E402  (path set above)

TARGET_COVERAGE = 0.93          # SPEC.md [A1]: ">= 93% two-sided coverage"
DEFAULT_BLOCK = 21              # SPEC.md's pre-registered, unchanged block length


def simulate_iid(n_blocks, block, rng):
    """Scenario IID — each observation i.i.d. N(0,1); the optimistic bound (no dependence)."""
    return rng.normal(0.0, 1.0, size=int(n_blocks) * int(block))


def simulate_overlapping(n_blocks, block, rng):
    """Scenario OVERLAPPING — each observation is a rolling sum of `block` i.i.d. innovations, so
    consecutive observations share up to block-1 of them: the autocorrelation structure real
    overlapping-horizon forward returns HAVE (consecutive decisions' forward windows overlap by
    up to H-1). Vectorized rolling sum via cumsum (identical math to disclosures.py's original
    per-element-loop construction, just faster)."""
    n_blocks, block = int(n_blocks), int(block)
    n = n_blocks * block
    innov = rng.normal(0.0, 1.0, size=n + block - 1)
    csum = np.concatenate([[0.0], np.cumsum(innov)])
    return csum[block:block + n] - csum[0:n]


SCENARIOS = {"iid": simulate_iid, "overlapping": simulate_overlapping}


def k_needed_for_trial(delta, ci_lo, ci_hi):
    """The smallest k >= 0 such that 0 lies in the k-inflated interval
    [delta - k*(delta-ci_lo), delta + k*(ci_hi-delta)] (widened around the POINT ESTIMATE, not
    around 0, so a nonzero true delta is not itself treated as evidence against widening).

    NOTE ON A FIXED BUG: an earlier version of this function special-cased "the k=1 (raw,
    unwidened) interval already contains 0" as k_needed=0. That is WRONG — it conflated "no
    widening beyond k=1" with "no widening AT ALL", when the two half-widths shrink at DIFFERENT
    rates as k moves away from 1, so the exact minimal k can (and typically does) sit strictly
    between 0 and 1, not at 0, even when the raw interval already contains 0. The correct
    closed form depends only on the SIGN of delta (containment is monotonic in k on the relevant
    side regardless of where lo/hi start out):
      * delta == 0: the point estimate already sits at 0; k_needed = 0.
      * delta > 0: only the LOWER bound can ever cross 0 (delta + k*d_hi > 0 for any k >= 0
        since d_hi >= 0), so k_needed = delta / d_lo where d_lo = delta - ci_lo (inf if
        d_lo <= 0 -- the lower percentile already sits AT OR ABOVE delta, a one-sided
        degeneracy no widening fixes).
      * delta < 0: symmetric, k_needed = -delta / d_hi where d_hi = ci_hi - delta (inf if
        d_hi <= 0).
    Verified against a brute-force scan over k for representative (delta, lo, hi) triples."""
    delta, lo, hi = float(delta), float(ci_lo), float(ci_hi)
    if delta == 0.0:
        return 0.0
    if delta > 0.0:
        d_lo = delta - lo
        if d_lo <= 0.0:
            return np.inf
        return delta / d_lo
    d_hi = hi - delta
    if d_hi <= 0.0:
        return np.inf
    return -delta / d_hi


def gather_k_needed(n_blocks, block, scenario, n_trials, n_boot, rng):
    """Run n_trials independent TRUE-NULL trials (two independent sides, so the population delta
    is exactly 0) through the REAL ``report.block_bootstrap_delta_ci`` — never a re-implementation
    — and return the array of per-trial k_needed. Trials whose raw CI comes back
    ``insufficient_n`` (below the hard degeneracy floor) are skipped: no k can rescue a bootstrap
    that never ran."""
    sim = SCENARIOS[scenario]
    ks = []
    for _ in range(int(n_trials)):
        s = sim(n_blocks, block, rng)
        n = sim(n_blocks, block, rng)
        ci = report_mod.block_bootstrap_delta_ci(
            s, n, block=block, n_boot=n_boot, seed=int(rng.integers(0, 2**31 - 1)))
        if ci.get("insufficient_n"):
            continue
        ks.append(k_needed_for_trial(ci["delta"], ci["ci_lo_raw"], ci["ci_hi_raw"]))
    return np.asarray(ks, dtype=np.float64)


def measured_coverage_at_k(n_blocks, block, k, scenario, n_trials, n_boot, rng):
    """Independent re-measurement (VALIDATION, not the batch `k` was chosen from): fraction of
    fresh trials whose k-inflated interval contains 0."""
    ks = gather_k_needed(n_blocks, block, scenario, n_trials, n_boot, rng)
    if ks.size == 0:
        return None, 0
    return float(np.mean(ks <= float(k))), int(ks.size)


def calibrate_block_count(n_blocks, block=DEFAULT_BLOCK, target=TARGET_COVERAGE,
                          n_trials_calib=300, n_trials_valid=300, n_boot=1000, seed=0,
                          k_cap=8.0, calibration_margin=0.03):
    """Full calibrate-then-validate pass for ONE block count, in BOTH scenarios.

    Returns a dict: chosen k (the max of the two scenarios' calibration-batch target percentiles,
    so the interval clears the harder scenario without silently under-covering the easier one),
    and INDEPENDENTLY MEASURED validation coverage in each scenario at that k. ``achieved`` is
    True only if the OVERLAPPING validation coverage (Amendment 1's certification scenario) meets
    ``target``; a k that requires more than ``k_cap`` is refused outright (an interval widened
    past that point is not being 'calibrated', it is being inflated until it can't fail, which
    stops being an honest interval) and reported UNACHIEVED with the k it would have taken.

    ``calibration_margin`` — the calibration-batch percentile is taken at
    ``target + calibration_margin`` (capped at 0.999), NOT at ``target`` itself, while the
    ACCEPTANCE test (``achieved``) still checks the independent validation batch against the
    real ``target`` (0.93) with NO margin. This is a standard finite-sample safety buffer, not a
    thumb on the scale: measured directly (a first pass at margin=0, 400+400 trials, seed 89251)
    put five of eleven tested block counts at validation coverage 0.90-0.925 — ONE side of a coin
    flip away from the 0.93 line, because a percentile estimated from ~400 calibration draws and
    then re-evaluated on an INDEPENDENT 400-draw validation batch regresses toward the true
    quantile with a Monte-Carlo SE of order 1.3-2pp at p~0.93; asking the calibration batch to
    clear target EXACTLY, then grading pass/fail on a fresh batch at that same exact line, is a
    coin flip by construction, not a measurement of whether k=target-percentile is adequate. The
    margin absorbs that regression; the validation criterion is NOT relaxed."""
    rng = np.random.default_rng(seed)
    calib_percentile = min(0.999, target + calibration_margin)
    calib_k = {}
    for scenario in ("iid", "overlapping"):
        ks = gather_k_needed(n_blocks, block, scenario, n_trials_calib, n_boot, rng)
        # method="higher": SELECTS an existing order statistic rather than linearly
        # interpolating between two neighbors. A handful of trials can legitimately have
        # k_needed = inf (a one-sided degenerate raw CI no finite widening fixes — see
        # k_needed_for_trial); the default "linear" method's interpolation does
        # inf-involving subtraction near such a boundary and silently returns NaN
        # (reproduced: numpy's own lerp step warns "invalid value encountered in subtract").
        # "higher" never interpolates, so it returns a real (possibly inf) percentile instead.
        calib_k[scenario] = (float(np.percentile(ks, 100.0 * calib_percentile, method="higher"))
                             if ks.size else None)
    valid_ks = [v for v in calib_k.values() if v is not None]
    if not valid_ks:
        return {"n_blocks": int(n_blocks), "block": int(block), "k": None, "achieved": False,
               "reason": "no calibration trial produced a bandable raw CI at this block count"}
    k = max(valid_ks)
    uncapped_k = k
    capped = k > k_cap
    if capped:
        k = float(k_cap)
    out = {"n_blocks": int(n_blocks), "block": int(block), "k": float(k),
          "k_uncapped": float(uncapped_k), "capped_at_k_cap": bool(capped),
          "calibration_percentile_k": calib_k}
    for scenario in ("iid", "overlapping"):
        cov, n_valid = measured_coverage_at_k(n_blocks, block, k, scenario,
                                              n_trials_valid, n_boot, rng)
        out[f"validation_coverage_{scenario}"] = cov
        out[f"validation_n_{scenario}"] = n_valid
    out["achieved"] = (not capped) and (out["validation_coverage_overlapping"] is not None
                                        and out["validation_coverage_overlapping"] >= target)
    if capped:
        out["reason"] = (f"the overlapping-scenario calibration percentile needed k="
                         f"{uncapped_k:.2f} > k_cap={k_cap} — refused as 'not a calibration "
                         f"anymore', reported UNACHIEVED rather than applied")
    elif not out["achieved"]:
        out["reason"] = (f"validation-batch overlapping coverage "
                         f"{out['validation_coverage_overlapping']!r} < target {target} despite "
                         f"a calibration-batch percentile of k={k:.3f} — the calibration did not "
                         f"reproduce out-of-sample at this block count")
    else:
        out["reason"] = None
    return out


def build_calibration_table(block_counts, block=DEFAULT_BLOCK, target=TARGET_COVERAGE,
                            n_trials_calib=300, n_trials_valid=300, n_boot=1000, seed=0,
                            k_cap=8.0, calibration_margin=0.03):
    """The reproducible harness entry point: one calibrate_block_count call per block count,
    returned as a dict keyed by block count. This IS the two-scenario calibration harness
    SPEC.md [A1] pre-registers; ``python calibration.py`` regenerates it end to end and prints the
    table this module's constants in disclosures.py were measured from."""
    table = {}
    for i, nb in enumerate(sorted(set(int(b) for b in block_counts))):
        table[nb] = calibrate_block_count(nb, block=block, target=target,
                                          n_trials_calib=n_trials_calib,
                                          n_trials_valid=n_trials_valid, n_boot=n_boot,
                                          seed=seed + 1000 * i, k_cap=k_cap,
                                          calibration_margin=calibration_margin)
    return table


DISCLOSURE_FIELDS = ("k", "achieved", "validation_coverage_overlapping",
                    "validation_coverage_iid", "validation_n_overlapping",
                    "validation_n_iid", "reason")


def trim_table_for_disclosure(table):
    """The EXACT fields ``disclosures.CALIBRATION_TABLE`` embeds — one definition, so the
    embedded constant is traceably a PROJECTION of this module's own output, not a hand-copy
    that could drift from what ``calibrated_k_for`` actually consumes (it reads only ``k``/
    ``achieved``; the coverage/reason fields are carried for the artifact's own disclosure)."""
    return {int(nb): {f: e.get(f) for f in DISCLOSURE_FIELDS} for nb, e in table.items()}


def calibrated_k_for(n_blocks, table):
    """NEAREST-BELOW floor-lookup — same convention and same fixed bug as
    disclosures.calibrated_k_for (the one report.py actually calls; this copy exists for
    external/standalone use of this module and must not drift from it). Among every table entry
    whose block count is <= n_blocks AND whose calibration was ACHIEVED, take the entry with the
    LARGEST block count (nearest below), not the largest k — k DECREASES with n_blocks in every
    measured table, so "take the max k" silently collapses to "always use the smallest
    calibrated block count", discarding the more precise larger-n measurement. Returns
    ``(k, achieved, source_n_blocks)`` or ``(None, False, None)`` if no achieved entry at or below
    n_blocks exists."""
    candidates = [(nb, e) for nb, e in table.items() if nb <= n_blocks and e.get("achieved")]
    if not candidates:
        return None, False, None
    nb_src, entry = max(candidates, key=lambda kv: kv[0])
    return float(entry["k"]), True, int(nb_src)


# Block counts spanning the strata Amendment 1 names: the OOS binding leg (~5-6 blocks at
# 116 decisions), per-regime / entry-exit / re-trim splits (thinner still, hence the low end),
# and the sanity leg's pooled ~79 blocks (1673 decisions) plus headroom above it.
DEFAULT_BLOCK_COUNTS = (2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 65, 80, 100)


def main():
    import argparse
    import json
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-trials-calib", type=int, default=300)
    ap.add_argument("--n-trials-valid", type=int, default=300)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--block-counts", type=int, nargs="+", default=list(DEFAULT_BLOCK_COUNTS))
    ap.add_argument("--calibration-margin", type=float, default=0.03, dest="calibration_margin")
    ap.add_argument("--out", default=None, help="write the full table as JSON")
    args = ap.parse_args()
    table = build_calibration_table(args.block_counts, n_trials_calib=args.n_trials_calib,
                                    n_trials_valid=args.n_trials_valid, n_boot=args.n_boot,
                                    seed=args.seed, calibration_margin=args.calibration_margin)
    for nb in sorted(table):
        e = table[nb]
        print(f"blocks={nb:4d} k={e['k']!r} achieved={e['achieved']} "
             f"cov_overlap={e.get('validation_coverage_overlapping')!r} "
             f"cov_iid={e.get('validation_coverage_iid')!r} reason={e.get('reason')}")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(table, fh, indent=2, default=float)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
