#!/usr/bin/env python3
"""TASK_901 — synthetic dependence families for out-of-family validation.

`simulate_triangular` is the family `dependence_length`/tau was BUILT and calibrated against
(`experiments/897_corrected_809/calibration.py::simulate_overlapping`, imported verbatim, never
re-implemented). `range_probe.calibrate_k` also draws from this family (dep == block, the
matched-dependence convention). Every OTHER generator here is deliberately OUTSIDE that family —
the whole point of this row is that a range estimator built/calibrated against one dependence
shape must be re-measured, not assumed, on shapes it never saw.

Every generator returns a 1-D float64 array of length `n` with POPULATION MEAN (hence population
Sharpe) exactly 0 — so a "true null" containment check (does the CI contain 0?) is well posed for
all of them, mirroring 897's `ci_coverage_probe.py` / `calibration.py` discipline. The REAL
minute-data leg (`real_series.py`) does NOT have a known population mean and uses a different,
internal-consistency check instead (`coverage_harness.py`'s chunk-containment construction).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve()
_REPO = _THIS.parents[2]
_897 = _REPO / "experiments" / "897_corrected_809"
if str(_897) not in sys.path:
    sys.path.insert(0, str(_897))

import calibration as cal897  # noqa: E402  (READ-ONLY import — simulate_overlapping,
                               # simulate_persistent_component reused verbatim, never copied)


def simulate_iid(n: int, rng: np.random.Generator, **_) -> np.ndarray:
    """Population range 1 (no dependence at all) — the optimistic bound every candidate must
    clear before anything harder is worth measuring."""
    return rng.normal(0.0, 1.0, size=int(n))


def simulate_triangular(n: int, dep: int, rng: np.random.Generator, **_) -> np.ndarray:
    """THE CALIBRATION FAMILY — a rolling sum of `dep` i.i.d. innovations, population ACF exactly
    triangular (rho(h) = 1 - h/dep for h < dep, 0 beyond), so the true dependence RANGE is exactly
    `dep`. Thin wrapper over `cal897.simulate_overlapping` (never re-implemented).
    `coverage_harness.py` validates this module on `dep` values NOT used during calibration
    (held-out points), never only the calibration draws themselves."""
    return cal897.simulate_overlapping(int(n), int(dep), rng)


def simulate_ar1(n: int, phi: float, rng: np.random.Generator, **_) -> np.ndarray:
    """OUT-OF-FAMILY: a pure (unmixed) AR(1) at persistence `phi` — population ACF rho(h) = phi^h,
    NEVER hits exactly zero (unlike the triangular family's finite support), so the true "range"
    is a matter of definition (population tau = (1+phi)/(1-phi), finite but growing without bound
    as phi -> 1). Implemented via `cal897.simulate_persistent_component(n, phi, weight=1.0, ...)`
    — weight=1.0 zeroes the mixed-in i.i.d. component (`sqrt(1 - weight**2) == 0`), yielding a
    pure AR(1) through the SAME real code path used for the mixed 4%-variance case below (never a
    second, hand-rolled AR(1) generator)."""
    return cal897.simulate_persistent_component(int(n), float(phi), weight=1.0, rng=rng)


def simulate_persistent_component(n: int, phi: float, rng: np.random.Generator,
                                  weight: float = 0.2, **_) -> np.ndarray:
    """OUT-OF-FAMILY, THE ONE THAT KILLED TAU: `weight**2` of variance (0.2 -> 4%) is a slow AR(1)
    at persistence `phi`, the rest i.i.d. noise — genuinely persistent arbitrarily far out (high
    RANGE) but low AMPLITUDE at every individual lag, which is exactly the shape a raw ACF-band
    crossing under-reads. Thin wrapper over `cal897.simulate_persistent_component` (never
    re-implemented). Gate (ii) of TASK_901 requires this family specifically."""
    return cal897.simulate_persistent_component(int(n), float(phi), weight=float(weight), rng=rng)


def simulate_seasonal_regime(n: int, period: int, phi_low: float, phi_high: float,
                             rng: np.random.Generator, **_) -> np.ndarray:
    """OUT-OF-FAMILY: seasonal/regime-switching dependence — an AR(1) whose persistence
    ALTERNATES between `phi_low` (weak dependence) and `phi_high` (strong dependence) every
    `period/2` decisions (a square-wave regime schedule), rather than holding one dependence
    shape for the whole series. Neither the triangular family (finite-support ACF) nor a single
    AR(1) (constant persistence) represents this: a range estimator that reads out ONE number for
    the whole series is necessarily an average over regimes it was never shown as a single
    family. Population mean 0 throughout (zero-mean AR(1) driven by zero-mean innovations in both
    regimes), so the true-null containment check is well posed exactly as for the other families.

    NOT a generic regime-switching MODEL from the literature — a purpose-built stress case for
    this probe's out-of-family requirement, disclosed as such rather than presented as a standard
    reference generator."""
    n = int(n)
    period = max(2, int(period))
    innov_sd_low = math.sqrt(max(1e-12, 1.0 - float(phi_low) ** 2))
    innov_sd_high = math.sqrt(max(1e-12, 1.0 - float(phi_high) ** 2))
    x = np.empty(n, dtype=np.float64)
    x[0] = rng.normal(0.0, 1.0)
    half = max(1, period // 2)
    for t in range(1, n):
        regime_high = (t // half) % 2 == 1
        phi = float(phi_high) if regime_high else float(phi_low)
        sd = innov_sd_high if regime_high else innov_sd_low
        x[t] = phi * x[t - 1] + rng.normal(0.0, sd)
    return x


FAMILIES = {
    "iid": simulate_iid,
    "triangular": simulate_triangular,
    "ar1": simulate_ar1,
    "persistent_component": simulate_persistent_component,
    "seasonal_regime": simulate_seasonal_regime,
}
