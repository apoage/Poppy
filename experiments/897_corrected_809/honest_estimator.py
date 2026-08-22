#!/usr/bin/env python3
"""
TASK_897 — the HONEST per-decision, non-overlapping-hold Sharpe estimator.

WHY THIS FILE EXISTS (the "809 trap")
--------------------------------------
TASK_809's `decide.py` computed Sharpe by taking the BAR-LEVEL net return series and
annualizing it as `mean/std * sqrt(minutes_per_year)` (via inherited `backtest._sharpe(net, 1)`)
— i.e. it treated every 1-minute bar as an INDEPENDENT sample. But the decision rule holds a
position for many bars at a time (turnover ~0.087 on the reported-best cell implies an average
hold of ~1/0.087 ≈ 11.5 bars), so consecutive bar returns inside one hold are the SAME bet,
not independent draws. Annualizing by bar-count manufactures independence that is not there and
inflates the Sharpe by roughly sqrt(mean_hold_bars). That is the retracted "+5.08" (2026-06-25,
`lore/notes_2026-06-25.md`; banner in `runs/809_decision_rule_rescue/result.md`).

The fix: aggregate bar-level net returns into PER-DECISION (holding-period) returns first,
count actual decisions (rebalance events) in the window, and annualize by REALIZED
DECISIONS-PER-YEAR, never by bar-count. `sharpe_decision` on `HonestMetrics` below is the ONLY
gate-facing number this module produces; `sharpe_per_bar_DIAGNOSTIC_DO_NOT_GATE_ON` is kept
purely as a visible contrast against the retracted methodology — it must NEVER be read as a
verdict input (see `test_honest_estimator.py`'s gate-(i) mutation test).

No DB, no probe.py/decide.py import — this module is pure numpy so it is trivially unit-testable
and hand-checkable in isolation.

CONFIDENCE INTERVALS ARE **UNCALIBRATED** HERE — see `CI_UNCALIBRATED_DISCLOSURE` below and the
`ci_lo_UNCALIBRATED`/`ci_hi_UNCALIBRATED` field names on `HonestMetrics`. The pre-registered
band's "CI lo > 0" criterion therefore CANNOT be read off this module's interval. One layer up,
`calibration.py` certifies a width-inflation factor per real `(block, n_dec)` pairing through
THIS module's own `block_bootstrap_sharpe_ci`, and `sweep_gated_maker.apply_ci_calibration`
publishes the widened result as `ci_lo_ADVISORY`/`ci_hi_ADVISORY`, NaN wherever the pairing is
uncertified.

TASK_897 D57 (2026-08-22) — `ci_lo_ADVISORY`/`ci_hi_ADVISORY` ARE **NEVER BAND-READABLE ON THIS
LEG EITHER**, certified or not. The calibration's own certifying condition (this cell's measured
per-decision dependence <= the block its CI was bootstrapped at) is checked via
`dependence_length` below, a LINEAR-ACF statistic (the integrated autocorrelation time, tau) that
equals the dependence's RANGE — what a block bootstrap actually needs — only inside the exact
triangular-ACF family the calibration was measured against, and demonstrably UNDER-READS outside
it (see `dependence_length`'s own docstring for the measured numbers). On THIS row's real
per-decision data the guard is EXPECTED NOT TO BITE (every measured series reads tau=1-2 against
block 60), so its near-universal pass here is NOT evidence the dependence condition holds — it is
the guard measured INERT exactly where it will be used. The pre-registered band's third
criterion, "CI lo > 0", is therefore **UNAVAILABLE-because-uncalibratable** on this leg regardless
of which pair a reader looks at. `ci_lo_ADVISORY`/`ci_hi_ADVISORY` are recorded anyway — never
discarded — because **TASK_901** (a range-based dependence probe: first-lag-in-band /
variance-ratio / spectral-density-at-zero, validated out-of-family) re-reads this conjunct
retroactively FROM THE SAME ARTIFACTS.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

SECONDS_PER_YEAR = 365.0 * 24.0 * 3600.0  # 31,536,000 — non-leap convention, matches 808/809's
                                            # MINUTES_PER_YEAR = 365*24*60 at bar_sec=60.

# A Sharpe/CI computed from fewer than this many REAL decisions is not a measurement. Two is the
# hard floor for a sample standard deviation to exist at all; anything below returns NaN rather
# than a number a band could be read off. See `test_fewer_than_two_real_decisions_is_nan`.
MIN_DECISIONS_FOR_SHARPE = 2

# TASK_897 D55(a) — the block-length cap `compute_honest_metrics` selects at (previously a bare
# 20, making dependence longer than 20 decisions structurally uncapturable at ANY n, per D49's
# re-measured finding below). Raised to 60 by measurement, not taste — see
# `calibration.BLOCK_CAP_SELECTION_RATIONALE` for the two reproducible sweeps
# (`measure_block_cap_candidates` / `measure_dependence_mismatch`) behind this number. Owned
# HERE (not in `calibration.py`) so `calibration.py` can import it without creating a circular
# import (`calibration.py` already imports this module to run the REAL bootstrap it calibrates).
MAX_BLOCK_DEFAULT = 60

# --- REALIZED-DEPENDENCE MEASUREMENT (fix cycle 3, MAJOR 1(b)) -------------------------------
# The calibration certifies `k` at dep == block ONLY (see `calibration.CERTIFICATION_IS_
# CONDITIONAL_ON_DEPENDENCE`). Nothing in the pipeline previously MEASURED the real per-decision
# dependence of the 809 strategy, so the certification rested on an assumption the data was never
# asked about. These constants and `dependence_length` below give the srv leg the ability to
# REFUSE: each cell's own realized per-decision return series is measured, and where its
# dependence exceeds the block the cell's CI was bootstrapped at, `sweep_gated_maker.
# apply_ci_calibration` withholds even the advisory pair (`ci_advisory=False`, qualifier
# stands) — and per D57, an advisory=True cell is STILL never band-readable on this leg (the
# guard behind the flag is measured inert out-of-family; see `dependence_length`'s docstring).
DEPENDENCE_ACF_N_SE = 2.0            # white-noise band = n_se/sqrt(n): the truncation rule.
DEPENDENCE_ACF_MAX_LAG_FACTOR = 4    # probe lags out to 4x the largest representable block.
MIN_DECISIONS_FOR_DEPENDENCE = 20    # below this an ACF is not an estimate; refuse instead.

CI_UNCALIBRATED_DISCLOSURE = (
    "THE RAW INTERVAL PRODUCED BY THIS MODULE — the `ci_lo_UNCALIBRATED`/`ci_hi_UNCALIBRATED` "
    "fields, and `block_bootstrap_sharpe_ci`'s return value — IS **UNCALIBRATED**, and remains "
    "so. It is a plain circular block bootstrap percentile pair whose nominal 95% has never been "
    "shown to be its real two-sided coverage for this statistic (an annualized SHARPE RATIO of a "
    "per-decision return series). "
    "IT IS NOT A HYPOTHETICAL WORRY — IT WAS MEASURED. `ci_coverage_probe.py` runs THIS function "
    "under a true null (population Sharpe exactly 0) and found two-sided coverage BELOW the "
    "nominal 0.95 in every cell tested (2026-08-22, 400 trials x 400 resamples, AT THE THEN-"
    "PREVAILING BLOCK CAP OF 20): i.i.d. returns gave 0.905 / 0.902 / 0.920 / 0.935 / 0.948 at "
    "40 / 100 / 200 / 500 / 1000 decisions, and under overlapping dependence of length 20 it "
    "collapsed to 0.530 / 0.865 / 0.868 at 100 / 500 / 1000 decisions. DECISIONS D49 "
    "(2026-08-19) records the same class of failure on this repo's OTHER block bootstrap: where "
    "the block length sits at the dependence length, coverage plateaus near 88% against a 93% "
    "target and NO SAMPLE SIZE closes the gap — the remedy has to be the block length or the "
    "interval method, never more data. "
    "WHAT HAS CHANGED SINCE (D55(a), 2026-08-22 — the CURRENT mechanism, stated here because "
    "this sentence is where a reader checks the calibration's legitimacy): "
    f"(1) the block-length cap is now `MAX_BLOCK_DEFAULT` = {MAX_BLOCK_DEFAULT} decisions, not "
    "20 — `compute_honest_metrics` selects "
    f"`max(min_block, min({MAX_BLOCK_DEFAULT}, n_dec // 20))`, so dependence up to "
    f"{MAX_BLOCK_DEFAULT} decisions is now structurally representable where 20 was the ceiling "
    "before, and the pre-raise numbers quoted above no longer describe the block lengths this "
    "module selects. (2) `experiments/897_corrected_809/calibration.py` ran TASK_892's "
    "calibrate-then-validate METHOD (never its table — 892 calibrates a DELTA-OF-MEANS at a "
    "FIXED block 21 keyed by block COUNT; importing k across that statistic change AND that "
    "block-regime change would be exactly the unvalidated estimator reuse that produced the "
    "retracted '809 trap') against THIS code path, and certified a width-inflation factor `k` "
    "per REAL `(block, n_dec)` pairing, validated on an INDEPENDENT batch and re-measured at "
    "BORROWED (non-stratum) pairings. (3) `sweep_gated_maker.apply_ci_calibration` applies that "
    "`k` around the POINT ESTIMATE and publishes the result as the SEPARATE `ci_lo_ADVISORY`/"
    "`ci_hi_ADVISORY` fields with `ci_advisory=True`. "
    "(4) **THAT CERTIFICATION IS CONDITIONAL ON MEASURED DEPENDENCE, AND THE MEASUREMENT ITSELF "
    "IS ONLY VALID INSIDE ONE FAMILY.** Every `k` in the table was measured with the simulated "
    "dependence set EQUAL to the block length (dep == block). That is NOT a worst case — it is "
    "the EASIEST point of the calibration's own dependence sweep, and earlier revisions of "
    "`calibration.py` wrongly called it 'the matched-dependence worst case'. Held fixed against "
    "dependence 8x its block, the SHIPPED k covers 0.52-0.55 (measured; "
    "`calibration.MEASURED_SHIPPED_K_AT_DEPENDENCE_MISMATCH`) — the same magnitude as the 0.530 "
    "that raised D55. Since `block` is chosen from the SAMPLE SIZE "
    f"(`n_dec // 20`, capped at {MAX_BLOCK_DEFAULT}) and never from the data, 'dependence == "
    "block' would be self-fulfilling if merely assumed. `dependence_length` below ATTEMPTS to "
    "measure each cell's realized per-decision dependence (reported as `measured_dependence_len`, "
    "beside `block_used`) via tau, the integrated autocorrelation time — a LINEAR-ACF statistic "
    "that equals the dependence's RANGE (what a block bootstrap actually needs) ONLY inside the "
    "exact triangular-ACF family the calibration was measured against, and DEMONSTRABLY "
    "UNDER-READS out of family (TASK_897 D57, 2026-08-22): a synthetic series carrying a "
    "persistent component worth only 4% of variance (0.2 * AR(1), phi=0.99 — population tau 8.6, "
    "genuinely under block 60) reads tau=1, certifies 100% of draws, and covers 0.805; at "
    "phi=0.999 coverage is 0.376, BELOW the 0.530 that raised D55 "
    "(`calibration.MEASURED_TAU_OUT_OF_FAMILY_HALTED_CYCLE`). ON THIS ROW'S REAL PER-DECISION "
    "DATA THE GUARD IS EXPECTED NOT TO BITE: every measured series (7 majors, OOS 2024-11 -> "
    "2025-04, H in {1,5,20}, long-only and long-short) reads tau=1-2 against block=60 "
    "(`calibration.MEASURED_REAL_DATA_TAU_INERT_HALTED_CYCLE`, real-data, provenance-stamped, NOT "
    "re-run in this loop) — so a ~100% certification rate on THIS leg's real cells is NOT "
    "evidence the dep<=block condition holds, it is the guard measured INERT exactly where it "
    "will be used; an empirical containment check on the real long/short H=1 series (97 disjoint "
    "1000-decision chunks) contained the grand Sharpe only 0.639 of the time while the guard "
    "certified ~100%. `sweep_gated_maker.apply_ci_calibration` still REFUSES to certify any cell "
    "whose measured dependence exceeds its own block OR cannot be measured at all — fail-closed, "
    "worth keeping, with the qualifier standing for a refused cell. **TASK_901** (a range-based "
    "dependence probe — first-lag-in-band / variance-ratio / spectral-density-at-zero, validated "
    "out-of-family) is what would re-earn this conjunct, retroactively, FROM THE SAME ARTIFACTS. "
    "CONSEQUENCE, STATED PLAINLY: the TASK_897 pre-registered band's third criterion, "
    "'CI lo > 0', is **UNAVAILABLE-because-uncalibratable** on this leg — not off "
    "`ci_lo_UNCALIBRATED`, ever, and not off `ci_lo_ADVISORY` either, REGARDLESS of whether "
    "`ci_advisory` reads True: the guard behind that flag is measured inert on this row's real "
    "data, so a True does not establish the condition it claims to check. `ci_lo_ADVISORY`/"
    "`ci_hi_ADVISORY` are NEVER band-readable on this leg; they are retained, never discarded, so "
    "TASK_901 can re-read them. The raw pair is retained as a spread diagnostic only, so the "
    "advisory and uncalibrated numbers can always be compared side by side."
)


def decision_bars(turn_bar: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Indices of the bars that are REAL decisions: the realized post-fill weight actually
    changed (turn_bar > eps). This is the single definition of "a decision" in this module —
    `segment_by_decision`, `n_decisions` and `mean_hold_bars` all derive from it, so they cannot
    drift apart. A zero-turnover bar is never a decision, including the pre-first-trade stretch."""
    turn_bar = np.asarray(turn_bar, dtype=np.float64)
    return np.where(turn_bar > eps)[0]


def segment_by_decision(net_bar: np.ndarray, turn_bar: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Aggregate BAR-LEVEL net returns into PER-DECISION (holding-period) returns.

    A "decision" is any bar where the realized (post-fill) portfolio weight actually changed
    (turn_bar > eps). Each decision opens a new holding SEGMENT that runs from that bar until
    (but not including) the next decision bar; a segment's return is the SUM of the bar-level
    net returns inside it — the book is static across the hold, so summing per-bar net accrual
    equals the segment's total realized return under the linear per-bar accrual convention
    used upstream (`decide.py`'s `_run_decision`/`_fixed`).

    THE LEADING [0, first_decision) STRETCH IS **NOT** A DECISION AND IS NOT COUNTED.
    Before the first trade there is no book: that stretch is zero-turnover and zero-return, and
    the spec's requirement is annualization "by REALIZED decisions-per-year". A flat pre-trade
    interval is not a realized decision, so counting it would (a) inflate `n_decisions` and hence
    `sqrt(decisions_per_year)` — the SAME inflation direction as the retracted 809 trap this
    module exists to remove — and (b) inject a synthetic 0.0 sample into the per-decision series,
    which defeats `sharpe()`'s own `x.size < 2 -> nan` degeneracy guard and can manufacture a
    finite Sharpe with a zero-width CI out of a SINGLE real bet. (Worked counter-example, from
    the TASK_897 fix-pass review: 2000 bars, ONE real decision at bar 500 with turn=0.2 and a
    constant net of 5e-4 over bars 500..1199 — counting the leading stretch gave n_decisions=2,
    sharpe_decision=+22.93 and CI [+22.93,+22.93], which trivially satisfies a "CI lo > 0"
    criterion off one bet. Not counting it gives n_decisions=1 -> NaN, which is the truth.)

    NOTE — a previous revision of this docstring had the direction backwards, claiming that
    DROPPING the leading segment "biases decisions_per_year upward". It does not: `n_decisions`
    is the NUMERATOR of `decisions_per_year`, so keeping the extra segment is what biases it
    UPWARD. Corrected here.

    Returns an array of per-decision returns; length == the number of decision bars in the
    window (exactly; never +1). An all-flat window returns an EMPTY array — zero decisions.
    """
    net_bar = np.asarray(net_bar, dtype=np.float64)
    turn_bar = np.asarray(turn_bar, dtype=np.float64)
    n = net_bar.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    trade_idx = [int(i) for i in decision_bars(turn_bar, eps)]
    if not trade_idx:
        return np.zeros(0, dtype=np.float64)
    bounds = trade_idx + [n]
    return np.array(
        [float(np.nansum(net_bar[bounds[j]:bounds[j + 1]])) for j in range(len(bounds) - 1)],
        dtype=np.float64,
    )


def decisions_per_year(n_decisions: int, n_bars: int, bar_sec: float) -> float:
    """Annualize by the REALIZED DECISION RATE — never by bar count.

    This is the load-bearing anti-"809-trap" guard. The numerator is REAL decisions (trade bars,
    per `decision_bars`), the denominator is the window's calendar span in years computed from
    n_bars*bar_sec — NOT the bar count itself.

    The denominator is deliberately the FULL window span, including any flat pre-first-trade or
    post-last-exit stretch, not just the traded span: the strategy was in the evaluation window
    for those bars and chose not to trade, so that calendar time is real. Using only the traded
    span would raise decisions/year and hence the annualized Sharpe — the trap direction. Full
    span is the conservative choice.

    A mutation that made this return
    `n_bars / window_years` (bars-per-year) instead of `n_decisions / window_years`
    reintroduces exactly the retracted per-bar annualization; `test_honest_estimator.py`'s
    `test_gate_i_mutation_is_caught` demonstrates that mutation going RED.
    """
    if n_bars <= 0 or bar_sec <= 0:
        return float("nan")
    window_years = (n_bars * bar_sec) / SECONDS_PER_YEAR
    if window_years <= 0:
        return float("nan")
    return n_decisions / window_years


def sharpe(x: np.ndarray, periods_per_year: float) -> float:
    """Annualized Sharpe = mean/std * sqrt(periods_per_year), guarded against degenerate input.
    Caller controls what `periods_per_year` means (decisions/year for the honest number, or
    bars/year for the diagnostic-only contrast) — this function itself is annualization-agnostic.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 2 or not np.isfinite(periods_per_year) or periods_per_year <= 0:
        return float("nan")
    sd = float(x.std(ddof=0))
    if sd < 1e-15:
        return float("nan")
    return float(x.mean() / sd * math.sqrt(periods_per_year))


def cumulative_pnl_and_maxdd(net_bar: np.ndarray) -> Tuple[float, float]:
    """Cumulative net PnL via product compounding, and max drawdown of the equity curve.
    Bar-level compounding is annualization-independent (no per-bar-vs-per-decision choice here)."""
    r = np.asarray(net_bar, dtype=np.float64)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float("nan"), float("nan")
    equity = np.cumprod(1.0 + r)
    pnl = float(equity[-1] - 1.0)
    running_max = np.maximum.accumulate(equity)
    dd = equity / running_max - 1.0
    return pnl, float(np.min(dd))


def block_bootstrap_sharpe_ci(
    reb_ret: np.ndarray, ppy: float, block: int, n_boot: int, seed: int
) -> Tuple[float, float]:
    """Circular block bootstrap 95% *NOMINAL* CI on the annualized Sharpe of a (per-decision)
    return series. `ppy` must already be the honest decisions-per-year — this function does not
    know or care where its input series came from, only how to resample it.

    **UNCALIBRATED — THE 95% IS NOMINAL, NOT MEASURED.** Its true two-sided coverage has never
    been validated for this statistic at these block lengths; see `CI_UNCALIBRATED_DISCLOSURE`
    (and D49 for the measured ~88% plateau on this repo's other block bootstrap). The
    pre-registered TASK_897 band's "CI lo > 0" criterion cannot be read off this output. Callers
    must surface that qualifier wherever they surface the interval — hence the
    `ci_lo_UNCALIBRATED`/`ci_hi_UNCALIBRATED` field names on `HonestMetrics`."""
    r = np.asarray(reb_ret, dtype=np.float64)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 2 or not np.isfinite(ppy) or ppy <= 0:
        return float("nan"), float("nan")
    block = max(2, int(block))
    rng = np.random.RandomState(seed)
    n_blocks = int(math.ceil(n / block))
    factor = math.sqrt(ppy)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.randint(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n] % n
        sample = r[idx]
        sd = sample.std(ddof=0)
        boots[b] = (sample.mean() / sd * factor) if sd > 1e-15 else np.nan
    boots = boots[np.isfinite(boots)]
    if boots.size < 10:
        return float("nan"), float("nan")
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def dependence_length(
    x: np.ndarray,
    max_lag: int = None,
    n_se: float = DEPENDENCE_ACF_N_SE,
) -> float:
    """ATTEMPT TO MEASURE the realized dependence length of a per-decision return series — the
    quantity the whole calibration is conditional on and which nothing in this pipeline
    previously measured. Read the OUT-OF-FAMILY paragraph below before trusting this on real
    data: it is a LINEAR-ACF statistic, valid as a stand-in for dependence RANGE only inside the
    one family it was calibrated against.

    The estimate is the INTEGRATED AUTOCORRELATION TIME,
        tau = 1 + 2 * sum_{h=1..M} |rho(h)|,
    with the sum truncated at the first lag whose sample autocorrelation falls inside the
    white-noise band `n_se / sqrt(n)` (an initial-positive-sequence style truncation; a raw
    first-crossing lag is far noisier, and a fixed window would import an assumption of its own).

    WHY tau MATCHES `block` INSIDE THE CALIBRATED FAMILY, AND ONLY THERE. Under the exact
    dependence model the calibration is certified against (`calibration.simulate_overlapping`: a
    rolling sum of `dep` i.i.d. innovations) the population autocorrelation is triangular,
    rho(h) = 1 - h/dep for h < dep and 0 beyond, so
        tau = 1 + 2 * sum_{h=1..dep-1} (1 - h/dep) = 1 + 2 * (dep-1)/2 = dep
    exactly. Inside THAT family tau reads out, in the SAME units, the `dep` a certified cell's
    `k` was measured at — which is `block` — so `tau <= block` there reads "this series'
    dependence is no longer than the dependence the shipped k was certified at". For an i.i.d.
    series tau -> 1.

    OUT OF FAMILY, THIS BREAKS DOWN (TASK_897 D57, 2026-08-22) — NARROWED HERE BECAUSE AN EARLIER
    REVISION GENERALISED THE IN-FAMILY EQUALITY ABOVE TO EVERY SERIES, WHICH IS FALSE. A block
    bootstrap's real requirement is `block >= dependence RANGE` (how far out the autocorrelation
    extends); tau CONFLATES that range with AMPLITUDE (how large the autocorrelation is at each
    lag), so a low-amplitude but long-range persistent component under-reads tau badly: a
    synthetic series carrying a persistent component worth only 4% of variance
    (`calibration.simulate_persistent_component`, weight=0.2, phi=0.99 — population tau 8.6,
    genuinely under block 60) reads tau=1, certifies 100% of draws, and the achieved coverage is
    only 0.805; at phi=0.999 coverage falls to 0.376, BELOW the 0.530 that raised D55 in the first
    place (`calibration.MEASURED_TAU_OUT_OF_FAMILY_HALTED_CYCLE`; direction reproduced at reduced
    budget by `calibration.measure_tau_out_of_family_mismatch`). ON THIS ROW'S REAL PER-DECISION
    DATA THE GUARD IS THEREFORE EXPECTED NOT TO BITE: every measured real series (7 majors, OOS
    2024-11 -> 2025-04, H in {1,5,20}, long-only and long-short) reads tau=1-2 against block=60
    (`calibration.MEASURED_REAL_DATA_TAU_INERT_HALTED_CYCLE` — real-data, provenance-stamped,
    NOT re-run in this loop per the project's real-data-validation-is-scheduled invariant), so a
    ~100% certification rate on this leg's real cells is NOT evidence `tau <= block` genuinely
    holds — it is the guard measured INERT exactly where it will be used. An empirical
    containment check on the real long/short H=1 series (97 disjoint 1000-decision chunks)
    contained the grand Sharpe only 0.639 of the time while the guard certified ~100%.
    **TASK_901** (a range-based dependence probe — first-lag-in-band / variance-ratio /
    spectral-density-at-zero, validated out-of-family) is the row that replaces this comparison,
    re-reading this conjunct retroactively FROM THE SAME ARTIFACTS.

    FAIL-CLOSED. Returns NaN — which every consumer must treat as REFUSE, never as "fine" —
    whenever the dependence cannot be measured: fewer than `MIN_DECISIONS_FOR_DEPENDENCE`
    finite points, a degenerate (zero-variance) series, or an autocorrelation function that never
    returns to the white-noise band inside `max_lag` (dependence longer than this series can
    resolve is unknown dependence, and unknown dependence must not certify). This fail-closed
    behaviour is unaffected by the out-of-family caveat above — it is what happens when tau CAN
    be computed but should not be trusted as a range measurement that governs it.

    Otherwise returns `ceil(tau)` as a float, floored at 1.0.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n < MIN_DECISIONS_FOR_DEPENDENCE:
        return float("nan")
    xc = x - x.mean()
    denom = float(np.dot(xc, xc))
    if not np.isfinite(denom) or denom <= 0.0:
        return float("nan")           # constant series: no dependence structure is measurable
    if max_lag is None:
        max_lag = int(min(n // 2, DEPENDENCE_ACF_MAX_LAG_FACTOR * MAX_BLOCK_DEFAULT))
    max_lag = int(max_lag)
    if max_lag < 1:
        return float("nan")
    band = float(n_se) / math.sqrt(n)
    acc = 0.0
    for h in range(1, max_lag + 1):
        rho = float(np.dot(xc[h:], xc[:-h])) / denom
        if not np.isfinite(rho):
            return float("nan")
        if abs(rho) < band:
            tau = 1.0 + 2.0 * acc
            return float(max(1.0, math.ceil(tau)))
        acc += abs(rho)
    return float("nan")               # never returned to the band: dependence is UNRESOLVED


@dataclass
class HonestMetrics:
    n_bars: int
    n_decisions: int
    decisions_per_year: float
    mean_hold_bars: float
    turnover_mean: float
    sharpe_decision: float                              # <-- the ONLY gate-facing number.
    ci_lo_UNCALIBRATED: float                            # nominal-95% block-bootstrap bounds whose
    ci_hi_UNCALIBRATED: float                            # true coverage is UNVALIDATED — the band's
                                                          # "CI lo > 0" cannot be read off them.
                                                          # See CI_UNCALIBRATED_DISCLOSURE / D49.
    pnl: float
    maxdd: float
    sharpe_per_bar_DIAGNOSTIC_DO_NOT_GATE_ON: float      # the retracted 809-trap number; kept
                                                          # only as a visible contrast, never a
                                                          # verdict input.
    block_used: int = -1                                  # the block length the CI above was
                                                          # ACTUALLY bootstrapped at (-1 = no CI
                                                          # was computed). The calibration is
                                                          # certified at dep == THIS number.
    measured_dependence_len: float = float("nan")         # `dependence_length(reb_ret)` — the
                                                          # REALIZED dependence of this cell's own
                                                          # per-decision series. NaN = unmeasurable
                                                          # = REFUSE (fix cycle 3, MAJOR 1(b)).


def compute_honest_metrics(
    gross_bar: np.ndarray,
    turn_bar: np.ndarray,
    cost_per_side: float,
    bar_sec: float,
    seed: int = 0,
    n_boot: int = 2000,
    min_block: int = 2,
    max_block: int = MAX_BLOCK_DEFAULT,
) -> HonestMetrics:
    """The single entry point cells/cells-sweeps should call. Computes net = gross - turn*cost,
    then the HONEST per-decision Sharpe (never per-bar), an UNCALIBRATED CI (see
    `CI_UNCALIBRATED_DISCLOSURE` — its coverage is unvalidated and the band's "CI lo > 0"
    criterion cannot be read off it), PnL/maxDD, and (for contrast only) the retracted
    per-bar-annualized number."""
    gross_bar = np.asarray(gross_bar, dtype=np.float64)
    turn_bar = np.asarray(turn_bar, dtype=np.float64)
    net_bar = gross_bar - turn_bar * cost_per_side
    n_bars = int(net_bar.shape[0])

    reb_ret = segment_by_decision(net_bar, turn_bar)
    n_dec = int(reb_ret.shape[0])
    ppy = decisions_per_year(n_dec, n_bars, bar_sec)
    # Mean hold = mean segment length. Segments span [first decision bar, end of window), so the
    # covered span is n_bars - first_decision_bar — NOT n_bars (which would silently credit the
    # untraded lead-in to the holds).
    dbars = decision_bars(turn_bar)
    mean_hold = ((n_bars - int(dbars[0])) / n_dec) if n_dec > 0 and dbars.size else float("nan")

    # Fewer than MIN_DECISIONS_FOR_SHARPE REAL decisions is not a measurement: report NaN rather
    # than a number a band could be read off. (`sharpe`/`block_bootstrap_sharpe_ci` each carry
    # their own size<2 guard; this is the explicit, single, self-documenting statement of the
    # rule at the entry point, and it is what the leading-segment fix relies on.)
    if n_dec < MIN_DECISIONS_FOR_SHARPE:
        sh_dec = float("nan")
        lo = hi = float("nan")
        block = -1
    else:
        sh_dec = sharpe(reb_ret, ppy)
        block = max(min_block, min(max_block, n_dec // 20))
        lo, hi = block_bootstrap_sharpe_ci(reb_ret, ppy, block, n_boot, seed)

    # The measurement the certification is conditional on (MAJOR 1(b)): this cell's OWN realized
    # per-decision dependence, in the same units as `block`. NaN when unmeasurable — consumers
    # must refuse on NaN, never certify.
    dep_len = dependence_length(reb_ret)

    pnl, maxdd = cumulative_pnl_and_maxdd(net_bar)
    turnover_mean = float(np.mean(turn_bar)) if n_bars else float("nan")

    bars_per_year = (SECONDS_PER_YEAR / bar_sec) if bar_sec > 0 else float("nan")
    sh_bar_diagnostic = sharpe(net_bar, bars_per_year)  # the retracted 809 number; diagnostic only

    return HonestMetrics(
        n_bars=n_bars, n_decisions=n_dec, decisions_per_year=ppy, mean_hold_bars=mean_hold,
        turnover_mean=turnover_mean, sharpe_decision=sh_dec, ci_lo_UNCALIBRATED=lo,
        ci_hi_UNCALIBRATED=hi, pnl=pnl,
        maxdd=maxdd, sharpe_per_bar_DIAGNOSTIC_DO_NOT_GATE_ON=sh_bar_diagnostic,
        block_used=int(block), measured_dependence_len=float(dep_len),
    )
