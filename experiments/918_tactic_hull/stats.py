#!/usr/bin/env python3
"""TASK_918 statistics discipline (D81(c) denominator + D85 spec-time MDE).

Both functions here are pure math on the pre-registered `n_decisions_monthly` (the number of
calendar months in the STEP-0 window, ~30 -- the D81(c)-mandated denominator for statistical
POWER claims; the underlying Sharpe/turnover arithmetic itself still runs on the native daily
H-strided grid via `scorer.score_arm`/`trader.backtest`. Nothing here touches the scorer.
"""
import math


def se_annual_null(n_months):
    """SE of an ANNUALIZED Sharpe estimated from `n_months` roughly-independent monthly
    observations under a null of zero true Sharpe (iid monthly returns): the raw per-period
    Sharpe estimator has SE ~= 1/sqrt(n) under the null (Lo 2002-style asymptotics with SR=0);
    annualizing a monthly Sharpe multiplies it by sqrt(12), and that same factor propagates to
    its SE under linear/delta approximation -- SE_annual(n_months) = sqrt(12/n_months)."""
    return math.sqrt(12.0 / max(n_months, 1))


def _expected_max_normal(N):
    """E[max of N iid standard normals], N>=1 (Gumbel-tail approximation for N>=2; exact 0 for
    N=1). z(N) ~= sqrt(2*ln N) - (ln(ln N) + ln(4*pi)) / (2*sqrt(2*ln N))."""
    if N <= 1:
        return 0.0
    a = math.sqrt(2.0 * math.log(N))
    if N == 2:
        return a  # the second-order correction is undefined at ln(ln 2) < 0 in a way that
                  # only matters for N=2; the leading term alone is the standard approximation
                  # here (error < 0.1 at N=2).
    return a - (math.log(math.log(N)) + math.log(4.0 * math.pi)) / (2.0 * a)


def sr0(N, n_months):
    """SR0(N): expected max realized (annualized) Sharpe of N ZERO-MEAN null candidates at
    this window's n_decisions_monthly -- the hindsight-selection-inflation reference every
    'best cell' number must be reported beside (`docs/pathway.md`'s loops-panel standing rule,
    cited in the task file). z(N) from `_expected_max_normal`, SE from `se_annual_null`.
    NOTE (disclosed, not silently assumed): this treats the N candidates as independent; the
    library's arms are NOT independent (shared market/trend exposure), so this is an UPPER
    bound on N's *effective* independent count and therefore likely OVERSTATES the true
    hindsight inflation for a correlated library -- conservative in the direction that makes a
    KILL harder to dismiss as 'just noise', not easier."""
    return _expected_max_normal(N) * se_annual_null(n_months)


def mde95(n_months):
    """D85 minimum detectable Sharpe DIFFERENCE at 95% confidence between two Sharpe estimates
    measured over the SAME window (treated as independent -- conservative: real correlation
    between two candidates scored on overlapping data would only shrink this further, so this
    is an upper bound on the true MDE, i.e. can only make a PASS claim look more
    underpowered, never less). SE_diff = sqrt(2)*SE_annual_null(n_months);
    MDE_95 = 1.96 * SE_diff."""
    se_diff = math.sqrt(2.0) * se_annual_null(n_months)
    return 1.96 * se_diff


def bootstrap_se_from_ci(ci_lo, ci_hi):
    """Cross-check SE implied by an ALREADY-COMPUTED 95% block-bootstrap CI (this window's own
    real data, not the analytic null): SE ~= (ci_hi - ci_lo) / (2*1.96). Reported beside the
    analytic `se_annual_null` so the MDE statement is not resting on one formula alone."""
    return (ci_hi - ci_lo) / (2.0 * 1.96)
