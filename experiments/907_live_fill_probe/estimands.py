"""TASK_907 — the two pre-registered estimands (`PREREGISTRATION.md`), computed from a
`ShadowPairResult` (see `shadow_engine.py`). Pure functions over already-gap-filtered samples;
this module does not itself decide what's usable — `shadow_engine.run_shadow` already dropped
any markout that would straddle a recorded gap (gate iv).

NON-FINITE SAMPLES (the second guard, after `shadow_engine`'s ingestion guard)
-----------------------------------------------------------------------------
A non-finite sample must never enter a mean, a median, a decile, or a tier comparison. It
survives every `>=` comparison as False, propagates silently through `sum()`, and leaves
`sorted()` with a corrupted rank ordering that still renders plausible-looking finite deciles.
That is the precise harm this row must not commit: a number presented as if it had been
measured.

So: non-finite samples are FILTERED, and the count is REPORTED — `n_used` and
`n_dropped_nonfinite` are separate fields carried all the way into the artifact. A silent drop
would be the same defect one layer down. When every sample at a horizon is non-finite the
horizon is still emitted, with `n_used=0`, `mean_bp=None`, `median_bp=None`, no deciles, and
every tier comparison `None` — i.e. NOT-EVALUABLE. A check that did not execute a comparison is
not a passed check, and it is certainly not a "BELOW measured" verdict.

MID SOURCE (the third guard, run-44)
------------------------------------
`compute_estimand1` computes the pre-registered estimand from **book-mid samples only**. The
shadow engine's trade-print fallback is a different quantity (a print is struck at the
aggressor's side of the spread, not at the mid) and its bias is not mean-zero where it matters:
after a fill going badly, aggressor flow is one-sided with the move, so the print sits on the far
side of the mid and measured adverse selection comes out too HIGH. That is the direction of this
row's own preferred conclusion, so it is the one direction a contaminant may not push. The
fallback-backed samples are computed separately by `compute_estimand1_trade_fallback` and
rendered as a labeled SECONDARY distribution, never pooled into the headline.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reference_fill_model import MODEL_ADVERSE_SELECTION_BP_TIERS  # noqa: E402
from shadow_engine import (  # noqa: E402
    HORIZONS_S, MID_SOURCE_BOOK, MID_SOURCE_TRADE, MarkoutSample, ShadowPairResult,
)

DECILES = (10, 20, 30, 40, 50, 60, 70, 80, 90)


def _percentile(sorted_vals: List[float], pct: float) -> float:
    """Linear-interpolation percentile (no numpy dependency for this small a computation).

    Refuses a non-finite input rather than interpolating through one: with a NaN left
    mid-sequence by `sorted()`, the neighbouring ranks silently become meaningless while still
    producing finite-looking output. Callers filter first; this is the backstop.
    """
    if not sorted_vals:
        raise ValueError("percentile of an empty sample")
    if not all(math.isfinite(v) for v in sorted_vals):
        raise ValueError(
            "percentile of a sample containing non-finite values — refusing: a NaN corrupts "
            "the rank ordering and the neighbouring deciles would render as plausible numbers "
            "derived from it")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = (pct / 100.0) * (len(sorted_vals) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = rank - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


@dataclass
class HorizonAdverseSelection:
    horizon: str
    #: Total markout samples seen at this horizon, before the finiteness filter.
    n_samples: int
    #: Samples that actually backed the numbers below.
    n_used: int
    #: Samples excluded for being non-finite. Disclosed, never silent.
    n_dropped_nonfinite: int
    mean_bp: Optional[float]
    median_bp: Optional[float]
    deciles_bp: Dict[int, float]
    #: `True`/`False` only when a real comparison ran; `None` means NOT-EVALUABLE (no finite
    #: median existed to compare against). Never collapse `None` into `False` — that is how a
    #: `tier >= nan` returning False became a confident "BELOW measured" verdict.
    model_tiers_meet_or_exceed_median: Dict[float, Optional[bool]]
    #: Staleness of the backing mid observations at the horizon (p50/p95, seconds). Disclosed so
    #: a reader can check that a minute-scale number rests on minute-scale inputs.
    mid_age_p50_s: Optional[float] = None
    mid_age_p95_s: Optional[float] = None

    @property
    def is_evaluable(self) -> bool:
        return self.median_bp is not None


@dataclass
class Estimand1Result:
    pair: str
    by_horizon: Dict[str, HorizonAdverseSelection] = field(default_factory=dict)
    #: Which mid source(s) back this result. The PRIMARY, pre-registered result is book-mid only;
    #: the trade-print fallback gets its own `Estimand1Result` so the two can never be read as
    #: one distribution.
    mid_source: str = MID_SOURCE_BOOK
    #: How many of this pair's markout samples were EXCLUDED from this result because they came
    #: from the other mid source. Disclosed for the same reason every other drop count is.
    n_excluded_other_source: int = 0

    @property
    def total_dropped_nonfinite(self) -> int:
        return sum(h.n_dropped_nonfinite for h in self.by_horizon.values())

    @property
    def total_used(self) -> int:
        return sum(h.n_used for h in self.by_horizon.values())


@dataclass
class Estimand2Result:
    pair: str
    n_posted: int
    n_touch_fills: int
    n_trade_cross_fills: int
    touch_fill_rate: Optional[float]
    trade_cross_fill_rate: Optional[float]
    n_markouts_dropped_nonfinite: int = 0
    n_markouts_dropped_gap: int = 0
    n_markouts_dropped_outside_span: int = 0
    n_markouts_dropped_no_mid: int = 0
    n_markouts_from_book_mid: int = 0
    n_markouts_from_trade_fallback: int = 0

    @property
    def trade_fallback_share(self) -> Optional[float]:
        total = self.n_markouts_from_book_mid + self.n_markouts_from_trade_fallback
        return None if total == 0 else self.n_markouts_from_trade_fallback / total


def _compute_estimand1(result: ShadowPairResult, mid_source: str) -> Estimand1Result:
    """Estimand 1 over the samples backed by ONE mid source. Never pools sources."""
    kept = [m for m in result.markouts if m.mid_source == mid_source]
    by_horizon: Dict[str, HorizonAdverseSelection] = {}
    for horizon in HORIZONS_S:
        # B3 (run-45): ALL FOUR pre-registered horizons (`HORIZONS_S`) appear here, ALWAYS — a
        # horizon that drew zero samples is a RESULT about the instrument (this probe's window
        # was too short for the 1h markout to ever form), not the absence of a row, and it used
        # to be silently omitted (`if not raw: continue`). Zero-sample and all-non-finite are the
        # SAME reported shape (n_used=0, nothing evaluable, no verdict) — they differ only in
        # `n_samples`, so one branch below covers both; no separate early `continue`.
        samples = [m for m in kept if m.horizon == horizon]
        raw = [m.adverse_selection_bp for m in samples]
        finite = [m for m in samples if math.isfinite(m.adverse_selection_bp)]
        vals = sorted(m.adverse_selection_bp for m in finite)
        ages = sorted(m.mid_age_s for m in finite if math.isfinite(m.mid_age_s))
        n_dropped = len(raw) - len(vals)
        if not vals:
            # Either zero samples at all, or every sample at this horizon was non-finite. Emit
            # the horizon anyway so the artifact SAYS so, with nothing evaluable and no verdict.
            by_horizon[horizon] = HorizonAdverseSelection(
                horizon=horizon, n_samples=len(raw), n_used=0, n_dropped_nonfinite=n_dropped,
                mean_bp=None, median_bp=None, deciles_bp={},
                model_tiers_meet_or_exceed_median={
                    tier: None for tier in MODEL_ADVERSE_SELECTION_BP_TIERS
                },
            )
            continue
        median = _percentile(vals, 50)
        by_horizon[horizon] = HorizonAdverseSelection(
            horizon=horizon,
            n_samples=len(raw),
            n_used=len(vals),
            n_dropped_nonfinite=n_dropped,
            mean_bp=sum(vals) / len(vals),
            median_bp=median,
            deciles_bp={d: _percentile(vals, d) for d in DECILES},
            model_tiers_meet_or_exceed_median={
                tier: bool(tier >= median) for tier in MODEL_ADVERSE_SELECTION_BP_TIERS
            },
            mid_age_p50_s=_percentile(ages, 50) if ages else None,
            mid_age_p95_s=_percentile(ages, 95) if ages else None,
        )
    return Estimand1Result(pair=result.pair, by_horizon=by_horizon, mid_source=mid_source,
                            n_excluded_other_source=len(result.markouts) - len(kept))


def compute_estimand1(result: ShadowPairResult) -> Estimand1Result:
    """The PRIMARY, pre-registered estimand 1: book-mid samples only.

    The trade-print fallback is deliberately NOT pooled in here — see the module docstring for
    why its bias direction makes pooling unsafe rather than merely imprecise.
    """
    return _compute_estimand1(result, MID_SOURCE_BOOK)


def compute_estimand1_trade_fallback(result: ShadowPairResult) -> Estimand1Result:
    """The SECONDARY distribution: samples whose as-of "mid" was a trade print. Reported apart,
    with the bias direction stated, so its share is visible instead of silently blended."""
    return _compute_estimand1(result, MID_SOURCE_TRADE)


def compute_estimand2(result: ShadowPairResult) -> Estimand2Result:
    return Estimand2Result(
        pair=result.pair,
        n_posted=result.n_posted,
        n_touch_fills=result.n_touch_fills,
        n_trade_cross_fills=result.n_trade_cross_fills,
        touch_fill_rate=result.touch_fill_rate,
        trade_cross_fill_rate=result.trade_cross_fill_rate,
        n_markouts_dropped_nonfinite=result.n_markouts_dropped_nonfinite,
        n_markouts_dropped_gap=result.n_markouts_dropped_gap,
        n_markouts_dropped_outside_span=result.n_markouts_dropped_outside_span,
        n_markouts_dropped_no_mid=result.n_markouts_dropped_no_mid,
        n_markouts_from_book_mid=result.n_markouts_from_book_mid,
        n_markouts_from_trade_fallback=result.n_markouts_from_trade_fallback,
    )
