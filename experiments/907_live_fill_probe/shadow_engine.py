"""TASK_907 phase 1 — the shadow engine.

Replays a capture (`capture.py`) into 1-minute bars per pair. At each bar close it posts a
*virtual* buy limit and a *virtual* sell limit at the close price, marks each filled per the
declared reference rule (`reference_fill_model.py`), and records the subsequent markouts for
every touch-fill. No orders, no keys, no money — this consumes a capture file on disk and
produces in-memory records only.

Gate (iv), enforced at markout granularity: a markout is computed only if the window between
the fill and the markout horizon is free of any recorded capture gap (`capture.covers`). A
markout that would straddle a gap is DROPPED, not computed across the hole as though nothing
happened there.

Non-finite values, enforced at INGESTION: a price or timestamp that is not a finite number
never becomes a bar, a mid, or a markout. Prices/timestamps are validated as a capture-format
violation (`CaptureFormatError`) because a NaN in a capture is corruption, not data; a markout
that nonetheless comes out non-finite is DROPPED AND COUNTED
(`ShadowPairResult.n_markouts_dropped_nonfinite`), never silently excluded — a silent drop is
the same defect one layer down from a silently admitted NaN.

THE MID SOURCE IS PART OF THE MEASUREMENT (run-44)
--------------------------------------------------
`MidSeries.mid_at` prefers top-of-book mid and falls back to the last trade print. **A trade
print is not a mid**: it is struck at the aggressor's side of the spread, so substituting one
for a mid mixes two different quantities into one reported adverse-selection distribution. Worse,
the substitution is not mean-zero where it matters: in the moments after a fill that is going
badly, aggressor flow is one-sided in the direction of the move, so the print sits on the far
side of the mid and the measured adverse selection is biased UPWARD (pessimistic). That is the
direction of this row's own preferred conclusion ("even the optimistic bound meets the model's
5 bp, so the model stands calibrated-pessimistic"), which is exactly the direction a contaminant
must never be allowed to push.

So every `MarkoutSample` carries `mid_source` and `mid_age_s`, the two source populations are
counted separately, and `estimands.compute_estimand1` computes the pre-registered estimand from
**book-mid samples only**. Trade-print-backed samples are reported as a clearly labeled
SECONDARY distribution with the bias direction stated. This is not an amendment to the estimand:
`PREREGISTRATION.md` says "mid price at t+h", and it never authorized a print as a mid — the
fallback was an implementation liberty, and confining it is conforming to the pre-registration,
not editing it.
"""
from __future__ import annotations

import bisect
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture import (  # noqa: E402
    DEFAULT_GAP_DETECT_THRESHOLD_S, CaptureFormatError, CaptureMetadata, covers, read_capture,
    _iter_lines_tolerant, _require_finite, _require_price,
)
from reference_fill_model import VirtualLimit, touch_fills, trade_crosses  # noqa: E402

HORIZONS_S: Dict[str, float] = {"1m": 60.0, "5m": 300.0, "20m": 1200.0, "1h": 3600.0}
BAR_SECONDS = 60.0

#: An as-of mid older than this is treated as MISSING, not current.
#:
#: Tightened 300s -> one bar interval (run-44). 300s was the per-sample form of the aggregate
#: defect: a mid observed five minutes before its horizon was being presented as the mid AT that
#: horizon, backing a markout whose shortest pre-registered horizon is 60s. One bar interval is
#: the per-sample counterpart of `capture.DEFAULT_COVERAGE_WINDOW_S`: the aggregate check bounds
#: how much of the CAPTURE may lack minute resolution, this bounds how stale any INDIVIDUAL
#: number's input may be. Exceeding it drops the candidate into `n_markouts_dropped_no_mid`,
#: which the artifact reports.
MAX_MID_STALENESS_S = BAR_SECONDS

#: Mid-source tags. `BOOK_MID` is the pre-registered quantity; `TRADE_PRINT` is the fallback,
#: excluded from the primary estimand (see the module docstring).
MID_SOURCE_BOOK = "book_mid"
MID_SOURCE_TRADE = "trade_print"


@dataclass
class _Bar:
    start_ts: float
    close: float
    high: float
    low: float
    high_ts: float
    low_ts: float
    trade_prices: List[float] = field(default_factory=list)
    trade_ts: List[float] = field(default_factory=list)


@dataclass(frozen=True)
class MidQuote:
    """An as-of mid WITH its provenance. `mid_at` returns this rather than a bare float so no
    caller can consume a mid without also being handed what it was derived from."""
    mid: float
    source: str      # MID_SOURCE_BOOK | MID_SOURCE_TRADE
    age_s: float     # how stale the underlying observation was at the as-of time


@dataclass
class MarkoutSample:
    pair: str
    side: int
    fill_price: float
    fill_ts: float
    horizon: str
    adverse_selection_bp: float
    #: Which quantity backed this sample. Defaults to the pre-registered one; `run_shadow` always
    #: sets it explicitly (pinned by `test_shadow_engine.py`).
    mid_source: str = MID_SOURCE_BOOK
    #: Staleness of the backing observation at the markout horizon, in seconds.
    mid_age_s: float = 0.0


@dataclass
class ShadowPairResult:
    pair: str
    n_bars_with_data: int = 0
    n_posted: int = 0
    n_skipped_no_next_bar: int = 0
    n_touch_fills: int = 0
    n_trade_cross_fills: int = 0
    markouts: List[MarkoutSample] = field(default_factory=list)
    #: Markout candidates whose as-of mid or computed return was non-finite. Counted here and
    #: reported in the artifact — never silently dropped.
    n_markouts_dropped_nonfinite: int = 0
    #: Markout candidates dropped because the horizon window straddled a recorded gap.
    n_markouts_dropped_gap: int = 0
    #: Markout candidates dropped because the horizon fell outside the capture's own span
    #: (usually: the capture simply ends before t+1h). Counted separately from a real gap —
    #: "we stopped recording" and "the wire went dark" are different facts about the data.
    n_markouts_dropped_outside_span: int = 0
    #: Markout candidates dropped because no as-of mid was available at the horizon.
    n_markouts_dropped_no_mid: int = 0
    #: Mid-source split of the markouts actually recorded. A trade print is NOT a mid (see the
    #: module docstring); the two are counted apart so the artifact can say how much of the
    #: distribution rests on each, and the primary estimand uses only the book-mid ones.
    n_markouts_from_book_mid: int = 0
    n_markouts_from_trade_fallback: int = 0

    @property
    def trade_fallback_share(self) -> Optional[float]:
        total = self.n_markouts_from_book_mid + self.n_markouts_from_trade_fallback
        return None if total == 0 else self.n_markouts_from_trade_fallback / total

    @property
    def touch_fill_rate(self) -> Optional[float]:
        return None if self.n_posted == 0 else self.n_touch_fills / self.n_posted

    @property
    def trade_cross_fill_rate(self) -> Optional[float]:
        return None if self.n_posted == 0 else self.n_trade_cross_fills / self.n_posted


class MidSeries:
    """As-of mid-price lookup: prefers book top-of-book mid, falls back to last trade print."""

    def __init__(self):
        self._book_ts: List[float] = []
        self._book_mid: List[float] = []
        self._trade_ts: List[float] = []
        self._trade_px: List[float] = []

    def add_book(self, ts: float, bid: float, ask: float) -> None:
        """Ingestion guard: a non-finite/non-positive quote never becomes a mid."""
        ts = _require_finite(ts, "book ts", "MidSeries.add_book")
        bid = _require_price(bid, "bid", "MidSeries.add_book")
        ask = _require_price(ask, "ask", "MidSeries.add_book")
        self._book_ts.append(ts)
        self._book_mid.append((bid + ask) / 2.0)

    def add_trade(self, ts: float, price: float) -> None:
        """Ingestion guard: a non-finite/non-positive print never becomes a mid.

        Both values are validated BEFORE either list is appended to: these are parallel arrays
        indexed together by `mid_at`, so a raise between the two appends would leave them
        permanently desynchronized and every later lookup off by one (or IndexError).
        """
        ts = _require_finite(ts, "trade ts", "MidSeries.add_trade")
        price = _require_price(price, "price", "MidSeries.add_trade")
        self._trade_ts.append(ts)
        self._trade_px.append(price)

    def finalize(self) -> None:
        # inputs may arrive interleaved across pairs' shared iteration; caller adds in ts order
        pass

    def mid_at(self, t: float) -> Optional[MidQuote]:
        """The as-of mid at `t`, TAGGED with what it came from.

        Book top-of-book mid is preferred; the last trade print is a fallback. Returning a
        `MidQuote` rather than a bare float is deliberate: the fallback used to be documented
        only in a code comment, so a trade print entered the flagship estimand indistinguishably
        from a real mid and the artifact's reader had no way to know.
        """
        if not math.isfinite(t):
            return None  # a non-finite as-of time resolves to "no mid", never to a stale one
        for source, ts_list, px_list in (
                (MID_SOURCE_BOOK, self._book_ts, self._book_mid),
                (MID_SOURCE_TRADE, self._trade_ts, self._trade_px)):
            i = bisect.bisect_right(ts_list, t) - 1
            if i >= 0 and (t - ts_list[i]) <= MAX_MID_STALENESS_S:
                return MidQuote(mid=px_list[i], source=source, age_s=t - ts_list[i])
        return None


def _build_bars_and_mids(path: str) -> Tuple[Dict[str, Dict[float, _Bar]], Dict[str, MidSeries]]:
    bars: Dict[str, Dict[float, _Bar]] = {}
    mids: Dict[str, MidSeries] = {}
    for obj in _iter_lines_tolerant(path):
        t = obj.get("type")
        pair = obj.get("pair")
        if t == "trade" and pair:
            where = f"{path}: trade record for {pair!r}"
            ts = _require_finite(obj["ts"], "ts", where)
            price = _require_price(obj["price"], "price", where)
            bar_start = (ts // BAR_SECONDS) * BAR_SECONDS
            pair_bars = bars.setdefault(pair, {})
            bar = pair_bars.get(bar_start)
            if bar is None:
                bar = _Bar(bar_start, price, price, price, ts, ts)
                pair_bars[bar_start] = bar
            bar.trade_prices.append(price)
            bar.trade_ts.append(ts)
            bar.close = price  # last trade seen so far in this bar (stream arrives in order)
            if price > bar.high:
                bar.high, bar.high_ts = price, ts
            if price < bar.low:
                bar.low, bar.low_ts = price, ts
            mids.setdefault(pair, MidSeries()).add_trade(ts, price)
        elif t == "book" and pair:
            mids.setdefault(pair, MidSeries()).add_book(obj["ts"], obj["bid"], obj["ask"])
    return bars, mids


def run_shadow(path: str,
               gap_detect_threshold_s: float = DEFAULT_GAP_DETECT_THRESHOLD_S,
               pairs: Optional[List[str]] = None) -> Tuple[CaptureMetadata, Dict[str, ShadowPairResult]]:
    """The phase-1 entry point: replay one capture file, return its reconciled metadata plus
    one `ShadowPairResult` per pair found. Does NOT itself refuse on incomplete metadata —
    that gate lives in `render.py`, applied per pair at render time, so a partially-usable
    capture can still report the pairs it can back.
    """
    meta = read_capture(path, gap_detect_threshold_s=gap_detect_threshold_s)
    bars_by_pair, mids_by_pair = _build_bars_and_mids(path)

    results: Dict[str, ShadowPairResult] = {}
    for pair, pair_bars in bars_by_pair.items():
        if pairs is not None and pair not in pairs:
            continue
        result = ShadowPairResult(pair=pair)
        mid_series = mids_by_pair.get(pair, MidSeries())
        starts = sorted(pair_bars.keys())
        result.n_bars_with_data = len(starts)

        for i, start in enumerate(starts[:-1]):
            bar = pair_bars[start]
            next_bar = pair_bars[starts[i + 1]]
            if starts[i + 1] != start + BAR_SECONDS:
                # the next bucket isn't the ADJACENT minute (a hole in trade data for this
                # pair) — cannot determine a touch outcome across a bar we have no data for.
                result.n_skipped_no_next_bar += 1
                continue

            post_ts = start + BAR_SECONDS
            for side in (1, -1):
                # The limit price is the divisor of every markout below. A zero/non-finite
                # close must refuse through this module's CaptureFormatError path rather than
                # surfacing as a raw ZeroDivisionError traceback from an arithmetic line.
                _require_price(bar.close, "bar close (the markout divisor)",
                                f"{path}: bar starting {start}")
                limit = VirtualLimit(side=side, price=bar.close, post_ts=post_ts)
                result.n_posted += 1
                touched = touch_fills(limit, next_bar.high, next_bar.low)
                crossed = trade_crosses(limit, next_bar.trade_prices)
                if crossed:
                    result.n_trade_cross_fills += 1
                if not touched:
                    continue
                result.n_touch_fills += 1
                fill_ts = next_bar.low_ts if side == 1 else next_bar.high_ts
                for horizon, h_s in HORIZONS_S.items():
                    target_ts = fill_ts + h_s
                    if not covers(meta, fill_ts, target_ts):
                        # gate (iv): drop, never compute across the hole. Attribute the reason
                        # precisely — a horizon running past the end of the recording is not
                        # the same fact as the wire going dark mid-capture.
                        if meta.end_ts is None or target_ts > meta.end_ts:
                            result.n_markouts_dropped_outside_span += 1
                        else:
                            result.n_markouts_dropped_gap += 1
                        continue
                    quote = mid_series.mid_at(target_ts)
                    if quote is None:
                        result.n_markouts_dropped_no_mid += 1
                        continue
                    # ONE non-finiteness check, not two (run-44 MINOR). There used to be a
                    # separate `math.isfinite(mid_then)` guard here as well; each of the two was
                    # individually removable with the suite still green, because guard 1 was a
                    # strict superset of guard 2 given `limit.price` is independently validated
                    # finite and strictly positive. Two mutually-masking checks are a coverage
                    # hole a future "de-duplicating" refactor would ship silently, so they are
                    # FOLDED into this single asserted-covered check on the result: a non-finite
                    # `quote.mid` (inf or nan) cannot produce a finite `signed_return` when the
                    # divisor is finite and positive, so this one check catches both reach paths.
                    # Pinned by
                    # `test_shadow_engine.py::test_the_single_nonfinite_guard_covers_both_reach_paths`.
                    signed_return = side * (quote.mid - limit.price) / limit.price * 1e4
                    if not math.isfinite(signed_return):
                        # Drop AND COUNT — a non-finite sample must never enter a mean, median,
                        # decile or tier comparison, and it must never vanish without a number
                        # saying so.
                        result.n_markouts_dropped_nonfinite += 1
                        continue
                    if quote.source == MID_SOURCE_BOOK:
                        result.n_markouts_from_book_mid += 1
                    else:
                        result.n_markouts_from_trade_fallback += 1
                    result.markouts.append(MarkoutSample(
                        pair=pair, side=side, fill_price=limit.price, fill_ts=fill_ts,
                        horizon=horizon, adverse_selection_bp=-signed_return,
                        mid_source=quote.source, mid_age_s=quote.age_s,
                    ))
        results[pair] = result
    return meta, results
