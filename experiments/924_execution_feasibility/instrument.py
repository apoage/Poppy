"""TASK_924 — execution-feasibility instrument over the CLOSED 907 capture.

D81 (binding, named in the task file): this module IMPORTS ONLY from
`experiments/907_live_fill_probe/capture.py`. Nothing under that tree is ever written to or
executed as a mutation target. See `SPEC.md` for the full definitions; this module is the
implementation of that spec, not a second source of definitions.

No DB access. No network. No exchange API. Read-only over the capture JSONL file named on the
command line / passed to `run_instrument`.
"""
from __future__ import annotations

import bisect
import hashlib
import json
import math
import os
import sys
from array import array
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

# --- D81: the ONLY import from the 907 tree. `sys.path` must already point at a staged copy
# of `907_live_fill_probe/` (see `run_full_pass.py` / test setup) — this module never assumes
# a particular repo layout so it can run against the workstation-HEAD copy staged on
# gpu-server, per the task's server-side scratch instructions.
from capture import (  # noqa: E402
    BAR_INTERVAL_S,
    STALE_BOOK_MIN_RUN_S,
    CaptureMetadata,
    read_capture,
)

HOUR_S = 3600.0


# --------------------------------------------------------------------------------------------
# small utilities
# --------------------------------------------------------------------------------------------

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_fingerprint(path: str) -> Dict[str, object]:
    st = os.stat(path)
    return {"path": path, "size": st.st_size, "mtime": st.st_mtime,
            "sha256": sha256_file(path)}


def _pctile(sorted_vals: List[float], pct: float) -> Optional[float]:
    """Linear-interpolation percentile over an already-sorted, all-finite list. Mirrors
    907's own `_pctile` (capture.py) but is a separate implementation — this module does not
    import that private helper, to keep the two code paths independent for G1's cross-check."""
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = (pct / 100.0) * (len(sorted_vals) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = rank - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def _median_iqr(values) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    vals = sorted(values)
    return _pctile(vals, 50), _pctile(vals, 25), _pctile(vals, 75)


# --------------------------------------------------------------------------------------------
# gap-span helpers — REUSE 907's own recorded gap marks; never re-derive gap detection.
# --------------------------------------------------------------------------------------------

class GapSpans:
    """Sorted, disjoint (lo, hi) gap spans, read directly off `CaptureMetadata.merged_gap_spans`
    (907's own coalesced gap ledger). Provides accounted-duration and accounted-segment
    queries used everywhere below. This class contains NO gap-detection logic of its own — it
    only consumes spans 907 already computed."""

    def __init__(self, spans: List[Tuple[float, float]]):
        spans = sorted(spans)
        self.lo = array("d", (s[0] for s in spans))
        self.hi = array("d", (s[1] for s in spans))

    def overlap(self, t0: float, t1: float) -> float:
        if t1 <= t0:
            return 0.0
        i = bisect.bisect_left(self.hi, t0)
        total = 0.0
        n = len(self.lo)
        while i < n and self.lo[i] < t1:
            total += max(0.0, min(self.hi[i], t1) - max(self.lo[i], t0))
            i += 1
        return total

    def accounted_duration(self, t0: float, t1: float) -> float:
        if t1 <= t0:
            return 0.0
        return max(0.0, (t1 - t0) - self.overlap(t0, t1))

    def accounted_segments(self, t0: float, t1: float):
        """Yield (a, b) sub-intervals of [t0, t1) that are NOT inside any gap span."""
        if t1 <= t0:
            return
        i = bisect.bisect_left(self.hi, t0)
        cur = t0
        n = len(self.lo)
        while i < n and self.lo[i] < t1:
            g0, g1 = self.lo[i], self.hi[i]
            if g0 > cur:
                yield (cur, min(g0, t1))
            cur = max(cur, g1)
            if cur >= t1:
                return
            i += 1
        if cur < t1:
            yield (cur, t1)

    def total_accounted(self, span_lo: float, span_hi: float) -> float:
        return self.accounted_duration(span_lo, span_hi)


# --------------------------------------------------------------------------------------------
# independent run tracker — G1(ii) / item 2 / item 4. A SEPARATE implementation from
# capture.py's `_note_book_top` / `_close_book_run`, sharing only the (bid, ask) run rule and
# the reused gap marks, so it can actually catch a bug either side might carry.
# --------------------------------------------------------------------------------------------

@dataclass
class _RunState:
    top: Optional[Tuple[float, float]] = None
    start: Optional[float] = None
    last: Optional[float] = None
    n: int = 0


class RunTracker:
    """Streaming consecutive-identical-(bid, ask) run tracker, independent of capture.py.

    Two readings of the SAME run stream are kept side by side (TASK_924 cycle 2, POST-HOC —
    SPEC.md "ERRATA 2"):

      * SHIPPED DEFINITION (gap-blind; `stale_s`, `run_durations_s`): a run's duration is
        `last - start` regardless of recorded gaps inside it. This reproduces 907's shipped
        `stale_book_s` (`capture.py:2433-2436`, clipped to the span only) and is what G1 checks.
      * GAP-CLIPPED (`stale_s_gap_clipped`, `run_pieces_gap_clipped_s`,
        `stale_pieces_gap_clipped_s`): only when `gaps` is given. Run rule at a gap: a run that
        straddles a recorded gap is SPLIT at the gap — the recorder was not observing inside it,
        and the price seen after the gap is a new observation, not proof the quote survived.
        The run's `[start, last]` interval is intersected with the accounted segments
        (`GapSpans.accounted_segments`, clipped to `[span_lo, span_hi]`); each resulting
        observed piece is its own run for thresholding, i.e. the `> min_run_s` rule applies to
        each piece separately. The stale numerator is the sum of pieces longer than
        `min_run_s`. `stale_s_gap_intersected_unsplit` (reconciliation only) keeps the shipped
        per-run threshold but drops gap seconds — the quantity the cycle-1 judge measured.
    """

    def __init__(self, min_run_s: float = STALE_BOOK_MIN_RUN_S,
                 gaps: Optional["GapSpans"] = None,
                 span: Optional[Tuple[float, float]] = None):
        self.min_run_s = min_run_s
        self.gaps = gaps
        self.span = span
        self._state: Dict[str, _RunState] = {}
        # shipped definition (gap-blind) — G1
        self.stale_s: Dict[str, float] = {}
        self.run_durations_s: Dict[str, List[float]] = {}
        self.stale_run_durations_s: Dict[str, List[float]] = {}
        # gap-clipped (924's own reading)
        self.stale_s_gap_clipped: Dict[str, float] = {}
        self.stale_s_gap_intersected_unsplit: Dict[str, float] = {}
        self.run_pieces_gap_clipped_s: Dict[str, List[float]] = {}
        self.stale_pieces_gap_clipped_s: Dict[str, List[float]] = {}
        self.n_stale_runs_straddling_gap: Dict[str, int] = {}

    def observe(self, pair: str, ts: float, bid: float, ask: float) -> None:
        top = (bid, ask)
        st = self._state.get(pair)
        if st is not None and st.top == top:
            st.last = ts
            st.n += 1
            return
        self._close(pair)
        self._state[pair] = _RunState(top=top, start=ts, last=ts, n=1)

    def _pieces(self, t0: float, t1: float) -> List[float]:
        if self.span is not None:
            t0 = max(t0, self.span[0])
            t1 = min(t1, self.span[1])
        if t1 <= t0:
            return []
        return [b - a for a, b in self.gaps.accounted_segments(t0, t1) if b > a]

    def _close(self, pair: str) -> None:
        st = self._state.get(pair)
        if st is None or st.n <= 1:
            return
        duration = st.last - st.start
        self.run_durations_s.setdefault(pair, []).append(duration)
        if duration > self.min_run_s:
            self.stale_s[pair] = self.stale_s.get(pair, 0.0) + duration
            self.stale_run_durations_s.setdefault(pair, []).append(duration)
        if self.gaps is None:
            return
        pieces = self._pieces(st.start, st.last)
        observed = sum(pieces)
        if duration > self.min_run_s:
            self.stale_s_gap_intersected_unsplit[pair] = (
                self.stale_s_gap_intersected_unsplit.get(pair, 0.0) + observed)
            if len(pieces) != 1 or observed < duration:
                self.n_stale_runs_straddling_gap[pair] = (
                    self.n_stale_runs_straddling_gap.get(pair, 0) + 1)
        run_pieces = self.run_pieces_gap_clipped_s.setdefault(pair, [])
        for d in pieces:
            run_pieces.append(d)
            if d > self.min_run_s:
                self.stale_s_gap_clipped[pair] = self.stale_s_gap_clipped.get(pair, 0.0) + d
                self.stale_pieces_gap_clipped_s.setdefault(pair, []).append(d)

    def close_all(self) -> None:
        for pair in list(self._state.keys()):
            self._close(pair)
        self._state.clear()


def time_weighted_quantiles(durations: List[float],
                            pcts=(10, 25, 50, 75, 90)) -> Dict[str, Optional[float]]:
    """Length-biased (time-weighted) distribution of run durations: the duration of the run a
    uniformly random instant INSIDE a run falls in. Each run is weighted by its own duration.
    Quantile rule (stated so tests can hand-compute it): sort ascending, W = sum(d); the
    p-quantile is the first d_k whose cumulative weight sum_{j<=k} d_j >= p/100 * W.
    Mean = sum(d^2) / sum(d). Zero-length runs carry zero weight."""
    ds = sorted(d for d in durations if d > 0)
    out: Dict[str, Optional[float]] = {f"p{p}_s": None for p in pcts}
    out["mean_s"] = None
    out["total_run_s"] = float(sum(ds))
    if not ds:
        return out
    total = sum(ds)
    targets = sorted((p / 100.0 * total, p) for p in pcts)
    cum = 0.0
    ti = 0
    for d in ds:
        cum += d
        while ti < len(targets) and cum >= targets[ti][0]:
            out[f"p{targets[ti][1]}_s"] = d
            ti += 1
        if ti == len(targets):
            break
    for _, p in targets[ti:]:  # float-rounding guard: remaining quantiles take the max
        out[f"p{p}_s"] = ds[-1]
    out["mean_s"] = sum(d * d for d in ds) / total
    return out


def count_weighted_quantiles(durations: List[float]) -> Dict[str, Optional[float]]:
    ds = sorted(durations)
    return {
        "median_s": _pctile(ds, 50), "p75_s": _pctile(ds, 75), "p90_s": _pctile(ds, 90),
        "p95_s": _pctile(ds, 95), "max_s": ds[-1] if ds else None,
        "mean_s": (sum(ds) / len(ds)) if ds else None, "n_runs": len(ds),
    }


def streak_stats(durations: List[float]) -> Dict[str, Optional[float]]:
    ds = sorted(durations)
    return {"n": len(ds), "median_s": _pctile(ds, 50), "p95_s": _pctile(ds, 95),
            "max_s": ds[-1] if ds else None, "total_s": float(sum(ds))}


# --------------------------------------------------------------------------------------------
# per-pair accumulators for items 1/3/5/6/7
# --------------------------------------------------------------------------------------------

@dataclass
class PairAccumulator:
    pair: str
    n_book_rows: int = 0
    n_trade_rows: int = 0
    n_two_sided_rows: int = 0
    n_one_sided_rows: int = 0
    bid_sz_two_sided: List[float] = field(default_factory=list)
    ask_sz_two_sided: List[float] = field(default_factory=list)
    # item 1: EVENT-weighted rel-half-spread (bp), bucketed by hour-of-day (UTC 0..23)
    spread_by_hod: Dict[int, List[float]] = field(default_factory=dict)
    # item 5 (binding, time-weighted): per elapsed-hour bucket k
    bucket_two_sided_s: Dict[int, float] = field(default_factory=dict)
    bucket_weighted_spread_s: Dict[int, float] = field(default_factory=dict)  # sum(spread*dt)
    # item 5 (non-binding, event-weighted): per elapsed-hour bucket k, two_sided rows only
    bucket_event_spread_sum: Dict[int, float] = field(default_factory=dict)
    bucket_event_spread_n: Dict[int, int] = field(default_factory=dict)
    # forward-fill state
    _last_ts: Optional[float] = None
    _last_two_sided: bool = False
    _last_spread_bp: float = 0.0

    def _add_bucket_weight(self, a: float, b: float, spread_bp: float, two_sided: bool,
                            start_ts: float) -> None:
        if b <= a:
            return
        bucket = int((a - start_ts) // HOUR_S)
        cur = a
        while cur < b:
            bucket_end = start_ts + (bucket + 1) * HOUR_S
            seg_end = min(b, bucket_end)
            dur = seg_end - cur
            if dur > 0 and two_sided:
                self.bucket_two_sided_s[bucket] = self.bucket_two_sided_s.get(bucket, 0.0) + dur
                self.bucket_weighted_spread_s[bucket] = (
                    self.bucket_weighted_spread_s.get(bucket, 0.0) + spread_bp * dur)
            cur = seg_end
            bucket += 1

    def observe_book(self, ts: float, bid: float, ask: float, bid_sz: float, ask_sz: float,
                      start_ts: float, gaps: GapSpans) -> None:
        self.n_book_rows += 1
        two_sided = bid_sz > 0.0 and ask_sz > 0.0
        if two_sided:
            self.n_two_sided_rows += 1
            self.bid_sz_two_sided.append(bid_sz)
            self.ask_sz_two_sided.append(ask_sz)
        else:
            self.n_one_sided_rows += 1
        mid = (bid + ask) / 2.0
        spread_bp = (ask - bid) / 2.0 / mid * 1e4 if mid > 0 else 0.0

        hod = datetime.fromtimestamp(ts, tz=timezone.utc).hour
        self.spread_by_hod.setdefault(hod, []).append(spread_bp)

        if two_sided:
            bucket = int((ts - start_ts) // HOUR_S)
            self.bucket_event_spread_sum[bucket] = (
                self.bucket_event_spread_sum.get(bucket, 0.0) + spread_bp)
            self.bucket_event_spread_n[bucket] = self.bucket_event_spread_n.get(bucket, 0) + 1

        if self._last_ts is not None and ts > self._last_ts:
            for a, b in gaps.accounted_segments(self._last_ts, ts):
                self._add_bucket_weight(a, b, self._last_spread_bp, self._last_two_sided,
                                         start_ts)
        self._last_ts = ts
        self._last_two_sided = two_sided
        self._last_spread_bp = spread_bp

    def observe_trade(self) -> None:
        self.n_trade_rows += 1


# --------------------------------------------------------------------------------------------
# result container
# --------------------------------------------------------------------------------------------

@dataclass
class FeasibilityResult:
    capture_path: str
    start_ts: float
    end_ts: float
    n_gaps: int
    accounted_s_global: float
    g1: Dict[str, object]
    per_pair: Dict[str, Dict[str, object]]


def run_instrument(path: str) -> FeasibilityResult:
    """Two full streaming passes over `path`:
      1. `read_capture(path)` — the SHIPPED 907 reader (unmodified, imported). Produces the
         reconciled gap ledger, span, and per-pair `stale_book_s`/`accounted_s` (G1(i)).
      2. This module's own independent pass — `RunTracker` (G1(ii)) + `PairAccumulator`
         (items 1/3/5/6/7), reusing ONLY the gap spans and span bounds from pass 1's `meta`.
    """
    meta: CaptureMetadata = read_capture(path)
    if meta.start_ts is None or meta.end_ts is None:
        raise ValueError(f"{path}: capture has no start_ts/end_ts — cannot proceed")

    gap_spans_list = list(meta.merged_gap_spans)
    gaps = GapSpans(gap_spans_list)
    accounted_s_global = gaps.total_accounted(meta.start_ts, meta.end_ts)

    tracker = RunTracker(gaps=gaps, span=(meta.start_ts, meta.end_ts))
    accs: Dict[str, PairAccumulator] = {}

    with open(path, "r") as fh:
        for raw in fh:
            raw = raw.rstrip("\n")
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                # Tolerate ONLY a truncated trailing line, exactly as `_iter_lines_tolerant`
                # does; this instrument's own pass mirrors that tolerance rather than
                # reimplementing lookahead logic, since a non-trailing corrupt line is a real
                # format violation the shipped reader (pass 1, already run) would already
                # have raised on.
                continue
            t = obj.get("type")
            if t == "book":
                pair = obj.get("pair")
                if not isinstance(pair, str) or not pair:
                    continue
                ts = obj.get("ts")
                bid = obj.get("bid")
                ask = obj.get("ask")
                bid_sz = obj.get("bid_sz")
                ask_sz = obj.get("ask_sz")
                if not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                           for v in (ts, bid, ask, bid_sz, ask_sz)):
                    continue
                ts = float(ts); bid = float(bid); ask = float(ask)
                bid_sz = float(bid_sz); ask_sz = float(ask_sz)
                if not (math.isfinite(ts) and math.isfinite(bid) and math.isfinite(ask)
                        and math.isfinite(bid_sz) and math.isfinite(ask_sz)):
                    continue
                tracker.observe(pair, ts, bid, ask)
                acc = accs.setdefault(pair, PairAccumulator(pair=pair))
                acc.observe_book(ts, bid, ask, bid_sz, ask_sz, meta.start_ts, gaps)
            elif t == "trade":
                pair = obj.get("pair")
                if not isinstance(pair, str) or not pair:
                    continue
                acc = accs.setdefault(pair, PairAccumulator(pair=pair))
                acc.observe_trade()
    tracker.close_all()

    # G1: shipped vs independent, per pair present in both.
    g1_pairs = {}
    for pair, cov in meta.coverage_by_pair.items():
        shipped_stale = cov.stale_book_s
        shipped_accounted = cov.accounted_s
        indep_stale = tracker.stale_s.get(pair, 0.0)
        indep_accounted = accounted_s_global  # global, identical across pairs by construction
        shipped_frac = shipped_stale / shipped_accounted if shipped_accounted > 0 else None
        indep_frac = indep_stale / indep_accounted if indep_accounted > 0 else None
        g1_pairs[pair] = {
            "shipped_stale_s": shipped_stale, "shipped_accounted_s": shipped_accounted,
            "shipped_fraction": shipped_frac,
            "indep_stale_s": indep_stale, "indep_accounted_s": indep_accounted,
            "indep_fraction": indep_frac,
        }

    per_pair = {}
    for pair, acc in accs.items():
        cov = meta.coverage_by_pair.get(pair)
        accounted_s = cov.accounted_s if cov is not None else accounted_s_global
        two_sided_s = sum(acc.bucket_two_sided_s.values())
        two_sided_fraction = two_sided_s / accounted_s if accounted_s > 0 else None

        all_spreads = [v for vs in acc.spread_by_hod.values() for v in vs]
        median_spread, q25, q75 = _median_iqr(all_spreads)

        hod_table = {}
        for hod in sorted(acc.spread_by_hod):
            m, lo, hi = _median_iqr(acc.spread_by_hod[hod])
            hod_table[hod] = {"n": len(acc.spread_by_hod[hod]), "median_bp": m,
                               "q25_bp": lo, "q75_bp": hi}

        # AS(1h), both forms, per bucket + a pooled summary.
        as1h_buckets = {}
        n_buckets_total = 0
        for bucket in sorted(set(acc.bucket_two_sided_s) | set(acc.bucket_event_spread_n)):
            n_buckets_total += 1
            tw_s = acc.bucket_two_sided_s.get(bucket, 0.0)
            tw_weighted = acc.bucket_weighted_spread_s.get(bucket, 0.0)
            tw_mean = tw_weighted / tw_s if tw_s > 0 else None
            ev_n = acc.bucket_event_spread_n.get(bucket, 0)
            ev_sum = acc.bucket_event_spread_sum.get(bucket, 0.0)
            ev_mean = ev_sum / ev_n if ev_n > 0 else None
            as1h_buckets[bucket] = {
                "time_weighted_bp": tw_mean, "covered_seconds": tw_s,
                "covered_fraction_of_hour": tw_s / HOUR_S,
                "event_weighted_bp": ev_mean, "n_two_sided_rows": ev_n,
            }
        tw_vals = [v["time_weighted_bp"] for v in as1h_buckets.values()
                   if v["time_weighted_bp"] is not None]
        ev_vals = [v["event_weighted_bp"] for v in as1h_buckets.values()
                   if v["event_weighted_bp"] is not None]
        pooled_tw = (sum(v["time_weighted_bp"] * v["covered_seconds"]
                          for v in as1h_buckets.values() if v["time_weighted_bp"] is not None)
                     / two_sided_s) if two_sided_s > 0 else None
        pooled_ev = (sum(acc.bucket_event_spread_sum.values())
                     / sum(acc.bucket_event_spread_n.values())
                     if sum(acc.bucket_event_spread_n.values()) > 0 else None)

        durations = tracker.run_durations_s.get(pair, [])
        # PRE-REGISTERED (SPEC 2.7) count-weighted, gap-blind survival — kept unchanged.
        durations_sorted = sorted(durations)
        survival = {
            "median_s": _pctile(durations_sorted, 50),
            "p75_s": _pctile(durations_sorted, 75),
            "p90_s": _pctile(durations_sorted, 90),
            "p95_s": _pctile(durations_sorted, 95),
            "max_s": durations_sorted[-1] if durations_sorted else None,
            "n_runs": len(durations_sorted),
        }
        pieces = tracker.run_pieces_gap_clipped_s.get(pair, [])
        # POST-HOC (SPEC ERRATA 2): maker-relevant survival is TIME-weighted (length-biased),
        # gap-clipped. Count-weighted kept as secondary, labelled.
        survival_post_hoc = {
            "time_weighted_gap_clipped": time_weighted_quantiles(pieces),
            "time_weighted_shipped_gap_blind": time_weighted_quantiles(durations),
            "count_weighted_gap_clipped_SECONDARY": count_weighted_quantiles(pieces),
        }

        stale_shipped = tracker.stale_s.get(pair, 0.0)
        stale_clipped = tracker.stale_s_gap_clipped.get(pair, 0.0)
        stale_unsplit = tracker.stale_s_gap_intersected_unsplit.get(pair, 0.0)
        frozen_mid = {
            "denominator_accounted_s": accounted_s,
            "frozen_mid_shipped_definition": (stale_shipped / accounted_s
                                              if accounted_s > 0 else None),
            "stale_s_shipped_definition": stale_shipped,
            "frozen_mid_gap_clipped": (stale_clipped / accounted_s
                                       if accounted_s > 0 else None),
            "stale_s_gap_clipped": stale_clipped,
            "frozen_mid_gap_intersected_unsplit_RECONCILIATION": (
                stale_unsplit / accounted_s if accounted_s > 0 else None),
            "stale_s_gap_intersected_unsplit_RECONCILIATION": stale_unsplit,
            "stale_s_inside_recorded_gaps_shipped_minus_unsplit": stale_shipped - stale_unsplit,
            "n_stale_runs_shipped_definition": len(
                tracker.stale_run_durations_s.get(pair, [])),
            "n_stale_runs_straddling_gap": tracker.n_stale_runs_straddling_gap.get(pair, 0),
            "streaks_shipped_definition": streak_stats(
                tracker.stale_run_durations_s.get(pair, [])),
            "streaks_gap_clipped": streak_stats(
                tracker.stale_pieces_gap_clipped_s.get(pair, [])),
            "run_rule_at_gap": ("run straddling a recorded gap is split at the gap; the "
                                 f"> {tracker.min_run_s:g}s threshold applies to each observed "
                                 "piece"),
        }

        quote_updates_per_active_minute = (
            acc.n_book_rows / (accounted_s / 60.0) if accounted_s > 0 else None)

        per_pair[pair] = {
            "n_book_rows": acc.n_book_rows, "n_trade_rows": acc.n_trade_rows,
            "n_two_sided_rows": acc.n_two_sided_rows, "n_one_sided_rows": acc.n_one_sided_rows,
            "row_two_sided_fraction": (acc.n_two_sided_rows / acc.n_book_rows
                                        if acc.n_book_rows > 0 else None),
            "depth_bid_sz_median": _pctile(sorted(acc.bid_sz_two_sided), 50),
            "depth_ask_sz_median": _pctile(sorted(acc.ask_sz_two_sided), 50),
            "depth_bid_sz_p95": _pctile(sorted(acc.bid_sz_two_sided), 95),
            "depth_ask_sz_p95": _pctile(sorted(acc.ask_sz_two_sided), 95),
            "accounted_s": accounted_s,
            "two_sided_time_weighted_fraction": two_sided_fraction,
            "two_sided_seconds": two_sided_s,
            "median_rel_half_spread_bp": median_spread,
            "q25_rel_half_spread_bp": q25, "q75_rel_half_spread_bp": q75,
            "rel_spread_by_hour_of_day": hod_table,
            "as1h_by_bucket": as1h_buckets,
            "as1h_time_weighted_pooled_bp": pooled_tw,
            "as1h_event_weighted_pooled_bp": pooled_ev,
            "as1h_tw_ev_pooled_difference_bp": (
                (pooled_tw - pooled_ev) if (pooled_tw is not None and pooled_ev is not None)
                else None),
            "quote_updates_per_active_minute": quote_updates_per_active_minute,
            "quote_survival": survival,
            "quote_survival_post_hoc": survival_post_hoc,
            "frozen_mid_post_hoc": frozen_mid,
            "n_hour_buckets": n_buckets_total,
        }

    return FeasibilityResult(
        capture_path=path, start_ts=meta.start_ts, end_ts=meta.end_ts,
        n_gaps=len(meta.gaps), accounted_s_global=accounted_s_global,
        g1=g1_pairs, per_pair=per_pair,
    )
