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
band's "CI lo > 0" criterion therefore CANNOT be read off this module's interval.
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

CI_UNCALIBRATED_DISCLOSURE = (
    "THE CONFIDENCE INTERVAL PRODUCED BY THIS MODULE IS **UNCALIBRATED**. "
    "`block_bootstrap_sharpe_ci` is a plain circular block bootstrap whose two-sided coverage "
    "has NEVER been validated for this statistic (an annualized SHARPE RATIO of a per-decision "
    "return series) at the block lengths this module selects. DECISIONS D49 (2026-08-19) records "
    "the directly relevant negative result on this repo's OTHER block bootstrap: where the block "
    "length sits at the dependence length, two-sided coverage PLATEAUS near 88% against a 93% "
    "target and NO SAMPLE SIZE closes the gap — the remedy has to be the block length or the "
    "interval method, never more data. TASK_892 shipped a calibrated-width remedy for THAT "
    "estimator (`experiments/892_action_ev/calibration.py` + `disclosures.CALIBRATION_TABLE`, "
    "validated to 0.940-0.975 coverage), but that table does NOT transfer to this module: it was "
    "measured through `report.block_bootstrap_delta_ci` (a DELTA-OF-MEANS between two sides, not "
    "a Sharpe ratio), at a FIXED block length of 21 keyed only by block COUNT, for containment of "
    "the value 0 under a true null of delta==0. This module bootstraps a different statistic "
    "(mean/sd, a ratio whose bootstrap distribution is driven by the variance estimate) through a "
    "different code path at block lengths 2..20. Importing k from that table would be an "
    "unvalidated transfer across statistic AND block-length regime — exactly the class of "
    "unchecked estimator reuse that produced the retracted '809 trap' in the first place. "
    "AND IT IS NOT A HYPOTHETICAL WORRY — IT IS MEASURED. `ci_coverage_probe.py` runs THIS "
    "function under a true null (population Sharpe exactly 0) and finds two-sided coverage BELOW "
    "the nominal 0.95 in every cell tested (2026-08-22, 400 trials x 400 resamples): i.i.d. "
    "returns give 0.905 / 0.902 / 0.920 / 0.935 / 0.948 at 40 / 100 / 200 / 500 / 1000 decisions, "
    "and under overlapping dependence of length 20 it collapses to 0.530 / 0.865 / 0.868 at "
    "100 / 500 / 1000 decisions. The collapse is structural: `compute_honest_metrics` CAPS the "
    "block length at 20, so dependence longer than 20 decisions cannot be captured at any n — "
    "D49's 'the remedy is the block length or the interval method, never more data', measured "
    "again on this code path. A borrowed k (892's table gives ~2.0 at 5 blocks) would not be a "
    "calibration of this shortfall; it would be a false calibration label over an unmeasured one. "
    "CONSEQUENCE, STATED PLAINLY: the TASK_897 pre-registered band's third criterion, 'CI lo > 0', "
    "CANNOT BE READ OFF THIS INTERVAL. The interval is reported as a spread diagnostic only. "
    "Calibrating it (re-running 892's Monte-Carlo calibrate-then-validate harness against THIS "
    "code path and THIS statistic, in both the i.i.d. and overlapping-dependence scenarios) is a "
    "planner-scheduled prerequisite before any CELL CLEARS read, not an implementation detail."
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


def compute_honest_metrics(
    gross_bar: np.ndarray,
    turn_bar: np.ndarray,
    cost_per_side: float,
    bar_sec: float,
    seed: int = 0,
    n_boot: int = 2000,
    min_block: int = 2,
    max_block: int = 20,
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
    else:
        sh_dec = sharpe(reb_ret, ppy)
        block = max(min_block, min(max_block, n_dec // 20))
        lo, hi = block_bootstrap_sharpe_ci(reb_ret, ppy, block, n_boot, seed)

    pnl, maxdd = cumulative_pnl_and_maxdd(net_bar)
    turnover_mean = float(np.mean(turn_bar)) if n_bars else float("nan")

    bars_per_year = (SECONDS_PER_YEAR / bar_sec) if bar_sec > 0 else float("nan")
    sh_bar_diagnostic = sharpe(net_bar, bars_per_year)  # the retracted 809 number; diagnostic only

    return HonestMetrics(
        n_bars=n_bars, n_decisions=n_dec, decisions_per_year=ppy, mean_hold_bars=mean_hold,
        turnover_mean=turnover_mean, sharpe_decision=sh_dec, ci_lo_UNCALIBRATED=lo,
        ci_hi_UNCALIBRATED=hi, pnl=pnl,
        maxdd=maxdd, sharpe_per_bar_DIAGNOSTIC_DO_NOT_GATE_ON=sh_bar_diagnostic,
    )
