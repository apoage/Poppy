"""TASK_907 phase 1 — the durable, append-only, gap-honest capture format.

Independent of the 806/809/897 lineage: nothing here is imported from that code, and nothing
here touches `trading_data.db`, credentials, or any live order path.

Format
------
An append-only JSONL file. Every line is one JSON object with a `"type"` field:

  * ``meta_start``  — written once, first line: capture_id, pairs, subscribed channels, the
    declared gap-accounting clock basis, recorder start time.
  * ``book``        — top-of-book snapshot: pair, ts, bid, ask, bid_sz, ask_sz.
  * ``trade``       — a print: pair, ts, price, volume, side, and (informational only)
    ``venue_ts``.
  * ``heartbeat``   — liveness marker from the recorder's own clock (not the venue's).
  * ``counters``    — the recorder's own health counters (JSON-decode failures, schema
    mismatches, unrecognized frames). Cumulative; the reader takes the last one. Written so
    "we were connected but could not parse a thing" is VISIBLE in the artifact instead of
    living only in a process that has since exited.
  * ``gap``         — a RECORDED capture gap: start_ts, end_ts, duration_s, reason. Never
    silently interpolated (the 893 lesson) — every disconnect, sequence discontinuity,
    heartbeat timeout, clock jump, or sustained schema drift becomes one of these before
    capture resumes.
  * ``meta_end``    — written once on clean shutdown only. Its ABSENCE is not itself an error
    (a kill mid-capture is expected and handled by resumption), but its PRESENCE is what
    ``closed_cleanly`` reports.

THE CLOCK BASIS (load-bearing, decided once, here)
--------------------------------------------------
Every timestamp used for gap accounting, bar bucketing and markout timing — i.e. every ``ts``
field on every record, including trades — is **LOCAL RECEIPT TIME from the recorder's own
clock** (``clock_basis == "local_receipt"``). The venue's own trade timestamp is preserved as
an informational ``venue_ts`` field and is NEVER used in gap arithmetic.

Rationale: a gap is a statement about *what the recorder observed*, and during an outage the
recorder has no venue clock at all — it only knows when it last held a frame and when it came
back. Mixing bases makes the ledger unsound: a burst of trades carrying skewed or stale venue
timestamps would let successive gap records describe overlapping spans on two different
clocks, and no amount of interval arithmetic can repair a ledger whose entries are denominated
differently. One clock, chosen for the one thing the ledger is about.

Crash safety: every line is flushed and fsynced individually, so a kill mid-write can only ever
truncate the LAST line, never corrupt an earlier one. The reader tolerates and reports a
truncated trailing line rather than raising. Resumption always writes a ``gap`` record spanning
"last known event time" -> "process restart time" BEFORE any new data line, because a resumed
recorder genuinely does not know what happened on the wire while it was down.

Exclusive writer: ``CaptureWriter`` takes an advisory exclusive ``flock`` on the capture file.
Two recorder processes pointed at the same ``--out`` is a realistic operator error over a
multi-day run, and two writers interleaving into one append-only file produce exactly the
overlapping/duplicated gap ledger that gate (iv)'s accounting must never have to reason about.
The second writer refuses to start instead.

HEARTBEATS PROVE THE SOCKET, NOT THE MARKET (load-bearing, run-44)
------------------------------------------------------------------
Gate (iv) has THREE independent defenses, and the distinction between them is the whole point:

1. ``unrecorded_gaps`` — the per-jump connectivity check over ALL event timestamps, heartbeats
   included. Answers "was the socket dark?".
2. ``unrecorded_data_gaps`` — the SAME per-jump check run per pair over **data-bearing events
   only** (``book``/``trade``; heartbeats EXCLUDED). Answers "did we observe the market?". A
   session where heartbeats keep flowing normally — so the recorder's own
   ``heartbeat_timeout_s`` never expires and no gap is ever written — while the book/trade
   subscription silently throttles is invisible to (1) and to ``missing_channels()`` (which only
   fires at literally zero events). It is visible here.
3. ``coverage_by_pair`` — the AGGREGATE, CUMULATIVE check. (1) and (2) are per-jump threshold
   checks: a cadence sitting just *under* the threshold, sustained for the whole capture, passes
   both while never once observing the market at the resolution the estimand claims. Coverage
   bounds the cumulative fraction of accounted time that is not within
   ``coverage_window_s`` of a data-bearing observation.
4. ``pair_reconciliation_gaps`` — the DECLARED-vs-OBSERVED check (run-45). All three checks above
   need the pair to be PRESENT in the data before they can say anything: (1) and (2) need a
   timeline to find a jump in, and (3) needs a coverage record to find degraded. A pair — or one
   declared channel of a pair — that carried **ZERO** data-bearing events for the entire capture
   has none of those. It never enters ``data_ts``, so it gets no coverage record, no cadence row
   and no per-jump timeline, and it is therefore *silently absent from every downstream table*
   while still being named in the artifact's own top-line provenance. Answers "did we observe
   everything we said we subscribed to?".

   The DECLARED side of this check comes from ``meta_start``'s ``pairs``/``channels`` — i.e. the
   recorder's own subscription config, written at capture start — and NEVER from the set of pairs
   observed in the stream. Reconciling the observed set against a "declared" set that was itself
   built by unioning in the observed pairs is circular: it can only ever agree with itself, and
   the real drift it must catch (config says four pairs, capture contains three) is exactly what
   the union hides. ``meta.pairs`` IS that union and is kept only as the provenance label it
   always was; the reconciliation uses ``declared_pairs`` / ``observed_pair_channels``.

The four are ordered from coarse to fine and none subsumes another — (4) is what fires when the
subject of (1)–(3) is missing entirely. A capture must clear all four before any number may be
rendered from it.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, TextIO, Tuple

try:  # pragma: no cover — POSIX only; the probe runs on Linux (workstation + gpu-server)
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

GAP_REASONS = frozenset({
    "disconnect", "seq_discontinuity", "heartbeat_timeout", "clock_jump", "resume",
    "schema_drift",
})

#: The ONLY clock basis this format accepts for gap accounting (see the module docstring).
CLOCK_BASIS = "local_receipt"

#: Record types the reader understands. `format_version` 1 captures have no `counters` record;
#: that is fine — its absence just means the counters are unknown, which the artifact says.
SUPPORTED_FORMAT_VERSIONS = (1, 2)
CURRENT_FORMAT_VERSION = 2

#: Data channels a capture can declare in `meta_start`. A DECLARED channel with zero events is
#: a refusal (see `CaptureMetadata.assert_complete`) — "one channel silently went to zero while
#: the other stayed healthy" must not pass as a complete capture.
DATA_CHANNELS = ("book", "trade")

#: The bar interval the shadow engine buckets on. Declared HERE rather than imported from
#: `shadow_engine` (this module must stay free of downstream imports); the two are pinned equal
#: by `test_coverage_aggregate.py::test_the_two_knobs_are_coupled_to_the_bar_interval`, so they
#: cannot drift apart silently.
BAR_INTERVAL_S = 60.0

#: Data-bearing coverage window. An instant of capture time counts as OBSERVED iff a book/trade
#: event lies within +/- this of it. BAR_INTERVAL_S / 4 = 15s: the shortest pre-registered
#: markout horizon is 60s, so a 15s resolution bounds the timing error on the 1m markout at 25%
#: and on the 5m markout at 5%. Not operator-configurable from the CLI — it is a declared
#: property of the instrument, not a dial (a dial on a fail-loud bound is a bypass).
DEFAULT_COVERAGE_WINDOW_S = BAR_INTERVAL_S / 4.0

#: The declared AGGREGATE bound. A capture may leave at most this fraction of its accounted
#: (non-gap) time un-backed by a data-bearing observation. 2%: the deliverable is a decile
#: report, and a contaminated fraction f can displace a reported decile by at most f in
#: quantile-rank terms, so f = 0.02 means the printed p50 is, worst case, the true p48/p52 —
#: below the resolution at which any decile in this artifact is read.
DEFAULT_MAX_UNCOVERED_FRACTION = 0.02

# A jump between consecutive events larger than this, with no recorded gap spanning it, is
# treated as an UNRECORDED gap (internal-consistency check, gate iv). Kraken's public feed sends
# a heartbeat at least every ~1s when idle and book/trade updates far more often under any real
# liquidity, so a silent multi-minute hole is never a normal quiet market.
#
# CLAMPED TO THE BAR INTERVAL (run-44). This knob and the recorder's `--heartbeat-timeout-s` were
# two independently-configured, unclamped dials with no code coupling them, and the dangerous
# direction is this one being LARGE: at 120s over 60s bars, a jump could swallow an entire
# 1-minute bar — and every markout inside it — while sitting under the threshold. So the reader
# now REFUSES a threshold above one bar interval: any jump big enough to empty a whole bar is
# big enough to flag. The recorder's heartbeat timeout is the other direction (recorder stricter
# than reader = safe), so it is DISCLOSED rather than clamped, and a test pins the recorder's
# default at or below this ceiling so the two cannot drift apart unnoticed.
DEFAULT_GAP_DETECT_THRESHOLD_S = BAR_INTERVAL_S
MAX_GAP_DETECT_THRESHOLD_S = BAR_INTERVAL_S


class CaptureFormatError(ValueError):
    """A capture line or file violates the append-only capture contract."""


def _require_finite(value: Any, what: str, where: str) -> float:
    """Coerce to float and REFUSE anything non-finite.

    Python's `json` module accepts `NaN`/`Infinity` as bare literals by default, so a venue
    error frame, a schema-drift bug, or a hand-edited file can put a non-finite number in a
    capture. Such a value must never reach an arithmetic path: it propagates silently through
    means, poisons `sorted()` rank ordering, and makes every `>=` comparison False — which is
    how a non-finite sample turns into a confidently-worded false claim downstream.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CaptureFormatError(f"{where}: {what} must be a finite number, got {value!r}")
    v = float(value)
    if not math.isfinite(v):
        raise CaptureFormatError(
            f"{where}: {what} is non-finite ({value!r}) — refusing. A non-finite value cannot "
            f"back a measurement, and silently admitting one produces a number that looks "
            f"measured but was not.")
    return v


def _require_price(value: Any, what: str, where: str) -> float:
    """A price must be finite AND strictly positive (a zero price also divides by zero)."""
    v = _require_finite(value, what, where)
    if v <= 0.0:
        raise CaptureFormatError(
            f"{where}: {what} must be strictly positive, got {v!r} — a non-positive price is "
            f"not a market observation (and is the divisor in the markout computation).")
    return v


def _require_size(value: Any, what: str, where: str) -> float:
    v = _require_finite(value, what, where)
    if v < 0.0:
        raise CaptureFormatError(f"{where}: {what} must be >= 0, got {v!r}")
    return v


@dataclass(frozen=True)
class GapRecord:
    start_ts: float
    end_ts: float
    reason: str
    detail: str = ""

    def __post_init__(self) -> None:
        if self.reason not in GAP_REASONS:
            raise CaptureFormatError(f"unknown gap reason {self.reason!r} (must be one of {sorted(GAP_REASONS)})")
        # NOTE: the finiteness check must come FIRST — `end_ts < start_ts` is False for NaN, so
        # a NaN-endpoint gap would otherwise sail straight through the duration check and then
        # contribute NaN to every coverage sum that touches it.
        _require_finite(self.start_ts, "gap start_ts", "gap record")
        _require_finite(self.end_ts, "gap end_ts", "gap record")
        if self.end_ts < self.start_ts:
            raise CaptureFormatError(f"gap end_ts {self.end_ts} precedes start_ts {self.start_ts}")

    @property
    def duration_s(self) -> float:
        return self.end_ts - self.start_ts

    def to_line(self) -> Dict[str, Any]:
        return {
            "type": "gap", "start_ts": self.start_ts, "end_ts": self.end_ts,
            "duration_s": self.duration_s, "reason": self.reason, "detail": self.detail,
        }


class CaptureWriter:
    """Append-only writer. One process, one capture file. Resumable.

    Holds an exclusive advisory `flock` for its whole lifetime: a second writer against the
    same path refuses to start rather than interleaving into the same ledger.

    `channels` declares which data channels this capture is SUBSCRIBED to. When given, the
    reader requires each of them to carry at least one event (see
    `CaptureMetadata.assert_complete`). Synthetic/replay fixtures may omit it, in which case
    the reader falls back to "at least one data event of any kind".
    """

    def __init__(self, path: str, pairs: List[str], capture_id: str,
                 resume_gap_reason: str = "resume",
                 channels: Optional[List[str]] = None,
                 heartbeat_timeout_s: Optional[float] = None,
                 book_emission_policy: Optional[str] = None,
                 book_keepalive_s: Optional[float] = None):
        self.path = path
        self.channels = list(channels) if channels is not None else None
        self._is_resume = os.path.exists(path) and os.path.getsize(path) > 0
        self._fh: TextIO = open(path, "a", buffering=1)
        self._lock_held = False
        self._acquire_exclusive_lock()
        if not self._is_resume:
            meta: Dict[str, Any] = {
                "type": "meta_start", "capture_id": capture_id, "pairs": list(pairs),
                "format_version": CURRENT_FORMAT_VERSION, "clock_basis": CLOCK_BASIS,
            }
            if self.channels is not None:
                meta["channels"] = list(self.channels)
            if heartbeat_timeout_s is not None:
                # Declared so the artifact can print the recorder's liveness knob next to the
                # reader's gap-detect knob. The two used to be independently configured with no
                # code coupling them and nothing stating either one in the output.
                meta["heartbeat_timeout_s"] = _require_finite(
                    heartbeat_timeout_s, "heartbeat_timeout_s", f"{path}: meta_start")
            if book_emission_policy is not None:
                # ADDENDUM A (run 45): the book channel is no longer sampled verbatim — it is
                # emitted on change-or-keepalive. An undeclared emission policy would be a
                # fourth unreconciled declaration (A.4(4)). These two keys are READ BACK by
                # `read_capture` (`declared_book_emission_policy` / `declared_book_keepalive_s`)
                # and printed next to the cadence table by `render._render_coverage_block` — a
                # declaration written but never read is the very failure A.4(4) names, so the
                # write end and the read end land together.
                meta["book_emission_policy"] = str(book_emission_policy)
            if book_keepalive_s is not None:
                meta["book_keepalive_s"] = _require_finite(
                    book_keepalive_s, "book_keepalive_s", f"{path}: meta_start")
            self._write(meta)
        else:
            last_ts = _last_event_ts(path)
            now = _now()
            if last_ts is not None and now > last_ts:
                self.write_gap(GapRecord(last_ts, now, resume_gap_reason,
                                          detail="recorder restarted; wire state during the "
                                                 "outage is unknown"))

    def _acquire_exclusive_lock(self) -> None:
        if fcntl is None:  # pragma: no cover — non-POSIX fallback
            return
        try:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self._fh.close()
            raise CaptureFormatError(
                f"{self.path}: another process already holds the exclusive capture lock — "
                f"refusing to start a second writer on the same capture. Two recorders "
                f"interleaving into one append-only file produce an overlapping, duplicated "
                f"gap ledger, and a gap ledger that cannot be reasoned about cannot back a "
                f"number. Use a distinct --out path per recorder. ({exc})") from exc
        self._lock_held = True

    def _write(self, obj: Dict[str, Any]) -> None:
        self._fh.write(json.dumps(obj, separators=(",", ":"), allow_nan=False) + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def write_book(self, pair: str, ts: float, bid: float, ask: float,
                    bid_sz: float, ask_sz: float) -> None:
        where = f"{self.path}: book record for {pair!r}"
        self._write({"type": "book", "pair": pair,
                      "ts": _require_finite(ts, "ts", where),
                      "bid": _require_price(bid, "bid", where),
                      "ask": _require_price(ask, "ask", where),
                      "bid_sz": _require_size(bid_sz, "bid_sz", where),
                      "ask_sz": _require_size(ask_sz, "ask_sz", where)})

    def write_trade(self, pair: str, ts: float, price: float, volume: float, side: str,
                     venue_ts: Optional[float] = None) -> None:
        """`ts` is LOCAL RECEIPT TIME (the gap-accounting clock). `venue_ts`, when supplied, is
        the venue's own timestamp, recorded for the record and used by nothing.
        """
        where = f"{self.path}: trade record for {pair!r}"
        obj: Dict[str, Any] = {
            "type": "trade", "pair": pair,
            "ts": _require_finite(ts, "ts", where),
            "price": _require_price(price, "price", where),
            "volume": _require_size(volume, "volume", where), "side": side,
        }
        if venue_ts is not None:
            obj["venue_ts"] = _require_finite(venue_ts, "venue_ts", where)
        self._write(obj)

    def write_heartbeat(self, ts: float) -> None:
        self._write({"type": "heartbeat",
                      "ts": _require_finite(ts, "ts", f"{self.path}: heartbeat record")})

    def write_counters(self, ts: float, json_decode_errors: int, schema_mismatch: int,
                        unrecognized_frames: int) -> None:
        """Persist the recorder's health counters into the capture itself (cumulative)."""
        self._write({"type": "counters",
                      "ts": _require_finite(ts, "ts", f"{self.path}: counters record"),
                      "json_decode_errors": int(json_decode_errors),
                      "schema_mismatch": int(schema_mismatch),
                      "unrecognized_frames": int(unrecognized_frames)})

    def write_gap(self, gap: GapRecord) -> None:
        self._write(gap.to_line())

    def close(self, clean: bool = True) -> None:
        if clean:
            self._write({"type": "meta_end", "ts": _now()})
        if self._lock_held and fcntl is not None:
            try:
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            except OSError:  # pragma: no cover — closing the fd releases it regardless
                pass
            self._lock_held = False
        self._fh.close()


def _now() -> float:
    import time
    return time.time()


_EVENT_TYPES_FOR_LAST_TS = ("book", "trade", "heartbeat", "gap")


def _last_event_ts(path: str) -> Optional[float]:
    """Best-effort scan for the last well-formed event's ts, tolerating a truncated tail.

    Explicit `is not None` checks, not truthiness: a legitimate ts of 0.0 is falsy, and the old
    `obj.get("ts") or obj.get("end_ts")` silently skipped it.
    """
    last_ts = None
    for obj in _iter_lines_tolerant(path):
        if obj.get("type") not in _EVENT_TYPES_FOR_LAST_TS:
            continue
        ts = obj.get("ts")
        if ts is None:
            ts = obj.get("end_ts")
        if ts is None:
            continue
        if isinstance(ts, (int, float)) and not isinstance(ts, bool) and math.isfinite(float(ts)):
            last_ts = float(ts)
    return last_ts


def _iter_lines_tolerant(path: str) -> Iterator[Dict[str, Any]]:
    """Yield parsed JSON lines, silently dropping ONLY a truncated/corrupt trailing line.

    A non-trailing corrupt line is a real format violation and raises; the guarantee this
    format offers is "a kill mid-write cannot corrupt already-captured data", not "any
    corruption anywhere is tolerated".
    """
    with open(path, "r") as fh:
        lines = fh.readlines()
    for i, line in enumerate(lines):
        line = line.rstrip("\n")
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                continue  # truncated trailing write from a kill — tolerated, not corruption
            raise CaptureFormatError(f"{path}: corrupt non-trailing line {i} is not valid JSON")


@dataclass(frozen=True)
class ChannelCadence:
    """The REALIZED inter-event interval of one (pair, channel), so a reader can see a 119s
    cadence for what it is. A number derived from a 2-minute-resolution capture must not be
    presentable as a minute-scale measurement, and the only way a reader can check that claim is
    if the realized cadence is printed next to the numbers.
    """
    pair: str
    channel: str
    n_events: int
    median_interval_s: Optional[float]
    p95_interval_s: Optional[float]
    max_interval_s: Optional[float]


@dataclass(frozen=True)
class PairCoverage:
    """Aggregate DATA-BEARING coverage for one pair over the capture's whole data span.

    `accounted_s` is the time the capture CLAIMS to have observed: its data span minus the union
    of its recorded gaps. Recorded gaps are honest, disclosed absence and markouts across them
    are already dropped by `covers()`, so they are excluded from the denominator rather than
    counted against it. What this measures is the UNDISCLOSED remainder: accounted time that no
    data-bearing event came within `window_s` of.

    Computed over the capture's GLOBAL data span, not the pair's own — otherwise a pair that goes
    dead on day 2 of a 10-day capture would be judged over day 1 alone and pass.
    """
    pair: str
    n_data_events: int
    window_s: float
    span_s: float
    recorded_gap_s: float
    accounted_s: float
    covered_s: float

    @property
    def uncovered_s(self) -> float:
        return max(0.0, self.accounted_s - self.covered_s)

    @property
    def uncovered_fraction(self) -> float:
        # A zero-length (or fully gap-covered) span has nothing to be uncovered: the capture
        # makes no coverage claim at all, and `assert_complete`'s other checks own that case.
        if self.accounted_s <= 0.0:
            return 0.0
        return min(1.0, self.uncovered_s / self.accounted_s)


@dataclass
class CaptureMetadata:
    """The reconciled picture of one capture: what it recorded plus what the reader itself
    independently observed in the raw event stream. Gate (iv) lives here.
    """
    path: str
    #: The UNION of declared-and-observed pairs. This is the provenance LABEL it has always been
    #: and consumers (`render`) read it as such — it is deliberately NOT the declared side of the
    #: pair reconciliation, because a union cannot detect its own drift (see the module docstring,
    #: defense 4). Use `declared_pairs` / `observed_pairs` for anything that must distinguish them.
    pairs: List[str] = field(default_factory=list)
    n_book: int = 0
    n_trade: int = 0
    n_heartbeat: int = 0
    gaps: List[GapRecord] = field(default_factory=list)
    start_ts: Optional[float] = None
    end_ts: Optional[float] = None
    closed_cleanly: bool = False
    had_meta_start: bool = False
    truncated_tail: bool = False
    unrecorded_gaps: List[GapRecord] = field(default_factory=list)
    # --- declared contract, read back from meta_start ---
    channels: Optional[List[str]] = None
    clock_basis: Optional[str] = None
    format_version: int = 1
    #: The DECLARED pair set: `meta_start`'s `pairs`, which the recorder writes straight from its
    #: subscription config (`RecorderConfig.pairs` — the same list that goes out as `symbol` on
    #: both subscribe frames; `test_declared_pairs.py` pins that coupling). Populated from
    #: `meta_start` ONLY — never unioned with observed pairs, or defense 4 becomes circular.
    #: `None` means `meta_start` carried no `pairs` field at all, which is a REFUSAL, not a
    #: fallback: `CaptureWriter` writes the field unconditionally at every format version.
    declared_pairs: Optional[List[str]] = None
    # --- recorder health counters, read back from the last `counters` record ---
    counters_present: bool = False
    n_json_decode_errors: int = 0
    n_schema_mismatch: int = 0
    n_unrecognized_frames: int = 0
    # --- gap-ledger hygiene, observed by the reader (normalized, then disclosed) ---
    n_overlapping_gap_records: int = 0
    n_duplicate_gap_records: int = 0
    n_gaps_outside_event_span: int = 0
    merged_gap_spans: List[Tuple[float, float]] = field(default_factory=list)
    # --- the two per-jump knobs and the aggregate bound, carried so the artifact can state them
    gap_detect_threshold_s: float = DEFAULT_GAP_DETECT_THRESHOLD_S
    coverage_window_s: float = DEFAULT_COVERAGE_WINDOW_S
    max_uncovered_fraction: float = DEFAULT_MAX_UNCOVERED_FRACTION
    #: The recorder's own liveness knob, as declared in `meta_start` (format v2+). Disclosed, not
    #: clamped: the recorder being STRICTER than the reader is the safe direction, and the unsafe
    #: direction (a loose reader threshold) is clamped at `MAX_GAP_DETECT_THRESHOLD_S` instead.
    declared_heartbeat_timeout_s: Optional[float] = None
    #: The recorder's book EMISSION policy and its rate-limit floor, as declared in `meta_start`
    #: (A.4(4)). These exist so the artifact can state the cadence floor its book numbers were
    #: measured under: without them a healthy book's cadence table reads `median=5.0 max=5.0`
    #: and the reader cannot tell whether 5.0s is the market's rate or the recorder's rate limit.
    #: A declaration nobody reads is exactly the unreconciled-declaration failure A.4(4) exists
    #: to prevent, so these are READ here and RENDERED (`render._render_coverage_block`), and a
    #: floor at or above `gap_detect_threshold_s` is REFUSED in `read_capture`.
    declared_book_emission_policy: Optional[str] = None
    declared_book_keepalive_s: Optional[float] = None
    # --- defense 2 + 3: data-bearing (heartbeat-EXCLUDED) gap and coverage accounting ---
    data_start_ts: Optional[float] = None
    data_end_ts: Optional[float] = None
    unrecorded_data_gaps: List[GapRecord] = field(default_factory=list)
    coverage_by_pair: Dict[str, PairCoverage] = field(default_factory=dict)
    cadence: List[ChannelCadence] = field(default_factory=list)
    #: The OBSERVED grid: pair -> the data channels that actually carried >= 1 event for that pair.
    #: The single source of truth for "what did we observe" (`observed_pairs` is a view onto it, so
    #: the two cannot be made to disagree). A declared cell missing from this grid is defense 4.
    observed_pair_channels: Dict[str, List[str]] = field(default_factory=dict)
    #: False on hand-constructed metadata that never went through `read_capture`. Gate (iv)
    #: fail-CLOSED: a capture with data events but no coverage accounting is refused rather than
    #: waved through, so the aggregate check cannot be bypassed by constructing the object.
    coverage_evaluated: bool = False

    def missing_channels(self) -> List[str]:
        """DECLARED data channels that carried zero events.

        `n_book + n_trade > 0` is not enough: one channel going silently to zero while the
        other stays healthy is exactly the schema-drift failure mode this probe is most exposed
        to (Kraken v2's live schema is unverified), and the SUM hides it completely.
        """
        if not self.channels:
            return []
        counts = {"book": self.n_book, "trade": self.n_trade}
        return [ch for ch in self.channels if counts.get(ch, 0) == 0]

    @property
    def observed_pairs(self) -> List[str]:
        """Pairs that carried at least one data-bearing (book/trade) event.

        A VIEW onto `observed_pair_channels`, not a second field: two independently-set fields
        could be made to disagree by a hand-crafted metadata object, and "pair-level says observed,
        channel-level says empty" is precisely the shape defense 4 exists to refuse.
        """
        return sorted(p for p, chans in self.observed_pair_channels.items() if chans)

    def declared_pair_set_missing(self) -> bool:
        """True iff this capture declares no pair set at all (absent or empty `meta_start.pairs`).

        Fail-CLOSED, and deliberately NOT modelled on the `channels=None` fallback: `CaptureWriter`
        emits `channels` only when its caller supplies them, so a capture without them is a shape
        the writer really produces — whereas `pairs` is written unconditionally at every format
        version, so a capture without it is a shape this recorder cannot emit. Falling back to
        "declared := observed" for it would BE the bypass: a hand-crafted or drifted capture could
        declare nothing and thereby reconcile perfectly against whatever it happened to contain.
        """
        return not self.declared_pairs

    def pair_reconciliation_gaps(self) -> Dict[str, List[str]]:
        """Defense 4: reconcile the DECLARED subscription against what was actually observed.

        Returns a dict of failure-kind -> offending items, every value empty when the capture
        reconciles. All five directions are failures, because each one is a way for the artifact's
        provenance and its tables to describe different captures:

          * ``declared_without_data``    — declared pair, ZERO book/trade events all capture. The
            escaping case: no timeline for defenses 1-2, no coverage record for defense 3.
          * ``declared_without_coverage`` — declared pair with no entry in `coverage_by_pair`.
            Unreachable from `read_capture` today (it builds one entry per observed pair), but
            `assert_complete` must verify that the coverage map COVERS the pairs it was supposed to
            check rather than trusting its shape.
          * ``declared_channel_without_data`` — declared (pair, channel) cell with zero events, as
            ``pair:channel``. `missing_channels()` is a CAPTURE-WIDE sum, so "BTC has book+trade,
            ETH has trades only" leaves it blind; per-pair coverage pools book+trade, so a dense
            trade stream hides it there too. The cell simply has no cadence row and no presence
            anywhere — absence again, one granularity down.
          * ``observed_undeclared`` / ``coverage_undeclared`` — data (or a coverage record) for a
            pair the capture never declared. Unattributable in the other direction: a number backed
            by a pair no subscription asked for is not a number this instrument measured.

        When `channels` is undeclared the per-cell check is skipped — the same disclosed fallback
        the rest of this module uses for `channels=None`, and `render` prints
        ``channels=undeclared`` so the reader is told which check was available.
        """
        declared = set(self.declared_pairs or [])
        observed = set(self.observed_pairs)
        gaps: Dict[str, List[str]] = {
            "declared_without_data": sorted(declared - observed),
            "declared_without_coverage": sorted(p for p in declared
                                                if p not in self.coverage_by_pair),
            "declared_channel_without_data": [],
            "observed_undeclared": sorted(observed - declared),
            "coverage_undeclared": sorted(p for p in self.coverage_by_pair if p not in declared),
        }
        if self.channels:
            gaps["declared_channel_without_data"] = sorted(
                f"{pair}:{channel}"
                for pair in declared & observed
                for channel in self.channels
                if channel not in set(self.observed_pair_channels.get(pair, ())))
        return gaps

    def degraded_coverage_pairs(self) -> List[str]:
        """Pairs whose CUMULATIVE un-observed fraction of accounted time exceeds the declared
        bound — the aggregate defense.

        The per-jump checks are threshold checks on the LARGEST single hole; they say nothing
        about a degradation spread thinly across the whole capture. A capture whose book/trade
        events arrive every 59s for ten days clears every per-jump check at a 60s threshold while
        never once resolving the minute grid its estimands are denominated in.
        """
        return [pair for pair, cov in sorted(self.coverage_by_pair.items())
                if cov.uncovered_fraction > self.max_uncovered_fraction]

    def is_complete(self) -> bool:
        """Metadata is 'complete' iff there is a meta_start, every DECLARED data channel has at
        least one event (or, for captures declaring none, at least one data event of any kind),
        the reader's own scan found no gap that isn't already RECORDED — over all events AND
        again over data-bearing events alone — no pair's cumulative un-observed fraction exceeds
        the declared bound, and the DECLARED subscription reconciles cell-for-cell with what was
        observed.

        The boolean twin of `assert_complete`: every condition below appears there too, in the same
        order and via the same helper methods, so no input can make one True while the other
        raises. `test_declared_pairs.py::test_is_complete_and_assert_complete_never_disagree`
        pins that.
        """
        return (self.had_meta_start and (self.n_book + self.n_trade) > 0
                and not self.missing_channels()
                and not self.unrecorded_gaps
                and not self.unrecorded_data_gaps
                and self.coverage_evaluated
                and not self.declared_pair_set_missing()
                and not any(self.pair_reconciliation_gaps().values())
                and not self.degraded_coverage_pairs())

    def assert_complete(self) -> None:
        if self.had_meta_start is False:
            raise CaptureFormatError(f"{self.path}: no meta_start record — refusing to treat "
                                      f"this as a capture at all")
        if (self.n_book + self.n_trade) == 0:
            raise CaptureFormatError(f"{self.path}: capture has zero data events")
        missing = self.missing_channels()
        if missing:
            raise CaptureFormatError(
                f"{self.path}: GATE (iv) REFUSAL — subscribed channel(s) {missing} recorded "
                f"ZERO events (book={self.n_book}, trade={self.n_trade}). A capture that was "
                f"connected but recorded nothing on a channel it subscribed to is schema drift "
                f"or a dead subscription, not a quiet market; the SUM of channels looking "
                f"healthy is exactly how that hides. Counters seen: "
                f"json_decode_errors={self.n_json_decode_errors}, "
                f"schema_mismatch={self.n_schema_mismatch}, "
                f"unrecognized_frames={self.n_unrecognized_frames}.")
        if self.unrecorded_gaps:
            spans = ", ".join(f"[{g.start_ts:.0f}, {g.end_ts:.0f}] ({g.duration_s:.1f}s)"
                               for g in self.unrecorded_gaps)
            raise CaptureFormatError(
                f"{self.path}: GATE (iv) REFUSAL — the raw event stream contains a timestamp "
                f"discontinuity with NO matching recorded gap: {spans}. A gap invisible to "
                f"downstream analysis is worse than a crash (the 893 lesson); this capture "
                f"cannot back any number until the discontinuity is either explained by a "
                f"recorded gap or the capture is re-cut around it.")
        if self.unrecorded_data_gaps:
            spans = ", ".join(f"[{g.start_ts:.0f}, {g.end_ts:.0f}] ({g.duration_s:.1f}s)"
                               for g in self.unrecorded_data_gaps[:8])
            more = ("" if len(self.unrecorded_data_gaps) <= 8
                    else f" (+{len(self.unrecorded_data_gaps) - 8} more)")
            raise CaptureFormatError(
                f"{self.path}: GATE (iv) REFUSAL — {len(self.unrecorded_data_gaps)} "
                f"DATA-BEARING discontinuit(y/ies) with no recorded gap covering them: "
                f"{spans}{more}. Heartbeats are excluded from this check on purpose: a heartbeat "
                f"proves the socket is alive, not that the market was observed. A session whose "
                f"heartbeats keep flowing normally — so the recorder's own heartbeat_timeout_s "
                f"never expires and no gap is ever written — while the book/trade subscription "
                f"silently throttles passes the all-event check above and is caught here. "
                f"Threshold: {self.gap_detect_threshold_s}s; declared recorder "
                f"heartbeat_timeout_s: {self.declared_heartbeat_timeout_s}.")
        if not self.coverage_evaluated:
            raise CaptureFormatError(
                f"{self.path}: GATE (iv) REFUSAL — data-bearing coverage was never evaluated for "
                f"this metadata (it did not come from `read_capture`). Fail-closed: the "
                f"aggregate coverage check must not be bypassable by constructing the object.")
        if self.declared_pair_set_missing():
            raise CaptureFormatError(
                f"{self.path}: GATE (iv) REFUSAL — this capture DECLARES NO PAIR SET at all "
                f"(meta_start `pairs` is "
                f"{'absent' if self.declared_pairs is None else 'empty'}; format_version "
                f"{self.format_version}). `CaptureWriter` writes `pairs` into `meta_start` "
                f"unconditionally at every format version, so this shape was never produced by "
                f"this recorder. Fail-CLOSED rather than falling back to \"declared := observed\": "
                f"that fallback IS the bypass — it would let a capture declare nothing and thereby "
                f"reconcile perfectly against whatever it happened to contain, which is a "
                f"completeness claim about nothing. (Contrast `channels`, which the writer emits "
                f"only when its caller supplies them, and the `counters` record, whose absence "
                f"means the counts are UNKNOWN rather than zero — for those, absence is a shape "
                f"this writer really produces; for `pairs` it is not.) Observed pairs: "
                f"{self.observed_pairs}. Remedy: re-run the recorder, which declares its "
                f"subscription config into the capture — do NOT hand-edit a `pairs` field in to "
                f"match what the file already contains.")
        recon = self.pair_reconciliation_gaps()
        if any(recon.values()):
            detail = "; ".join(f"{kind}={items}" for kind, items in sorted(recon.items()) if items)
            raise CaptureFormatError(
                f"{self.path}: GATE (iv) REFUSAL — DECLARED/OBSERVED pair reconciliation failed. "
                f"Declared (from meta_start `pairs`, i.e. the recorder's subscription config): "
                f"{self.declared_pairs}. Observed (>=1 book/trade event): {self.observed_pairs}. "
                f"Declared channels: {self.channels if self.channels else 'undeclared'}; observed "
                f"per pair: "
                f"{ {p: self.observed_pair_channels[p] for p in sorted(self.observed_pair_channels)} }. "
                f"Pairs with a coverage record: {sorted(self.coverage_by_pair)}. Unreconciled: "
                f"{detail}. A pair (or one declared channel of a pair) that carried ZERO "
                f"data-bearing events for the whole capture defeats every other defense by having "
                f"no subject: the per-jump checks need a timeline to find a jump in and the "
                f"aggregate check needs a coverage record to find degraded, so a subscription that "
                f"was dead from t=0 is named in this artifact's own provenance and then absent from "
                f"every table below it. Silent absence presented as completeness is worse than a "
                f"crash (the 893 lesson) — a crash is attributable, a missing row reads as "
                f"'nothing to report'. Data or coverage for an UNDECLARED pair is the same failure "
                f"mirrored: a number backed by a pair no subscription asked for is not a number "
                f"this instrument measured. Remedy: re-cut the capture to a window where every "
                f"declared subscription delivered, or re-run the recorder against a subscription "
                f"that does. NOT 'drop the pair from the declaration' and NOT 'widen the bound' — "
                f"either one launders the absence into a completeness claim, which is the whole "
                f"thing gate (iv) exists to prevent.")
        degraded = self.degraded_coverage_pairs()
        if degraded:
            detail = "; ".join(
                f"{p}: {self.coverage_by_pair[p].uncovered_fraction * 100:.2f}% of "
                f"{self.coverage_by_pair[p].accounted_s:.0f}s accounted time un-observed "
                f"({self.coverage_by_pair[p].n_data_events} data events)"
                for p in degraded)
            worst = max((self.coverage_by_pair[p] for p in degraded),
                         key=lambda c: c.uncovered_fraction)
            raise CaptureFormatError(
                f"{self.path}: GATE (iv) REFUSAL — AGGREGATE data-bearing coverage below the "
                f"declared bound for pair(s) {degraded}: {detail}. Bound: at most "
                f"{self.max_uncovered_fraction * 100:.2f}% of accounted time may sit further "
                f"than {self.coverage_window_s:g}s from a book/trade observation. This is the "
                f"CUMULATIVE check, and it is not the per-jump one: a cadence sitting just under "
                f"the {self.gap_detect_threshold_s:g}s per-jump threshold, sustained, clears "
                f"every jump check while never once resolving the 1-minute grid these estimands "
                f"are denominated in — a capture that never observed the market faster than once "
                f"per {worst.span_s / max(1, worst.n_data_events):.0f}s on average must not "
                f"render as fully covered. Heartbeats are excluded: they prove the socket, not "
                f"the market. Re-cut the capture around the degraded stretch, or re-run the "
                f"recorder — do not widen the bound.")


def read_capture(path: str,
                  gap_detect_threshold_s: float = DEFAULT_GAP_DETECT_THRESHOLD_S,
                  coverage_window_s: float = DEFAULT_COVERAGE_WINDOW_S,
                  max_uncovered_fraction: float = DEFAULT_MAX_UNCOVERED_FRACTION,
                  ) -> CaptureMetadata:
    """Parse a capture file end to end and reconcile recorded gaps against the raw stream."""
    if not (math.isfinite(gap_detect_threshold_s) and gap_detect_threshold_s > 0.0):
        raise CaptureFormatError(
            f"gap_detect_threshold_s must be a finite positive number, got "
            f"{gap_detect_threshold_s!r}")
    if gap_detect_threshold_s > MAX_GAP_DETECT_THRESHOLD_S:
        raise CaptureFormatError(
            f"gap_detect_threshold_s={gap_detect_threshold_s} exceeds the bar interval "
            f"({BAR_INTERVAL_S}s) this probe's estimands are denominated in — refusing. A "
            f"per-jump threshold wider than one bar lets a hole swallow an entire 1-minute bar, "
            f"and every markout inside it, without ever being flagged. The two gap knobs are "
            f"coupled here on purpose: the reader's independent check may not be looser than "
            f"the resolution of the thing being measured. Max: {MAX_GAP_DETECT_THRESHOLD_S}s.")
    if not (math.isfinite(coverage_window_s) and coverage_window_s > 0.0):
        raise CaptureFormatError(
            f"coverage_window_s must be a finite positive number, got {coverage_window_s!r}")
    if coverage_window_s > BAR_INTERVAL_S:
        raise CaptureFormatError(
            f"coverage_window_s={coverage_window_s} exceeds the bar interval ({BAR_INTERVAL_S}s) "
            f"— refusing: a coverage window wider than one bar would certify a capture that "
            f"never resolved the bar grid as fully observed.")
    if not (math.isfinite(max_uncovered_fraction) and 0.0 <= max_uncovered_fraction < 1.0):
        raise CaptureFormatError(
            f"max_uncovered_fraction must be in [0, 1), got {max_uncovered_fraction!r}")

    meta = CaptureMetadata(path=path, gap_detect_threshold_s=gap_detect_threshold_s,
                            coverage_window_s=coverage_window_s,
                            max_uncovered_fraction=max_uncovered_fraction)
    all_ts: List[float] = []
    #: Data-bearing timestamps ONLY (book/trade), keyed pair -> channel. Heartbeats never enter
    #: this structure: that exclusion is the fix for the run-44 BLOCKER.
    data_ts: Dict[str, Dict[str, List[float]]] = {}
    recorded_gaps: List[GapRecord] = []
    lines = list(_iter_lines_tolerant(path))
    if lines and _file_has_truncated_tail(path):
        meta.truncated_tail = True

    pairs_seen: set = set()

    def _note_data_event(channel: str, pair: Any, ts: float, where: str) -> None:
        if not isinstance(pair, str) or not pair:
            raise CaptureFormatError(
                f"{where}: {channel} record has no usable `pair` ({pair!r}) — refusing. A "
                f"data-bearing event that belongs to no pair contributes to the channel counts "
                f"and to the event span while appearing in NO pair's coverage accounting, which "
                f"is exactly the kind of unattributable event gate (iv) cannot reason about.")
        data_ts.setdefault(pair, {}).setdefault(channel, []).append(ts)
        pairs_seen.add(pair)
    for obj in lines:
        t = obj.get("type")
        if t == "meta_start":
            meta.had_meta_start = True
            declared = obj.get("pairs")
            if declared is not None:
                # Shape-check before use. `pairs` is the DECLARED side of defense 4, and a bare
                # string here would be iterated character-by-character into a declared set of
                # single letters that no observed pair can ever match — turning a malformed
                # declaration into a reconciliation failure with an unreadable cause. Refuse at
                # the source instead.
                if (not isinstance(declared, list)
                        or not all(isinstance(p, str) and p for p in declared)):
                    raise CaptureFormatError(
                        f"{path}: meta_start `pairs` must be a list of non-empty strings, got "
                        f"{declared!r} — refusing. This field is the declared side of gate (iv)'s "
                        f"pair reconciliation; a declaration that cannot be read cannot be "
                        f"reconciled, and guessing at its meaning is how a completeness claim "
                        f"gets made about a set nobody stated.")
                meta.declared_pairs = sorted(set(declared))
                pairs_seen.update(declared)
            meta.channels = list(obj["channels"]) if obj.get("channels") is not None else None
            meta.clock_basis = obj.get("clock_basis")
            hb = obj.get("heartbeat_timeout_s")
            if hb is not None:
                meta.declared_heartbeat_timeout_s = _require_finite(
                    hb, "heartbeat_timeout_s", f"{path}: meta_start")
            policy = obj.get("book_emission_policy")
            if policy is not None:
                if not isinstance(policy, str) or not policy:
                    raise CaptureFormatError(
                        f"{path}: meta_start `book_emission_policy` must be a non-empty string, "
                        f"got {policy!r} — refusing rather than reporting a cadence floor whose "
                        f"meaning cannot be stated.")
                meta.declared_book_emission_policy = policy
            keepalive = obj.get("book_keepalive_s")
            if keepalive is not None:
                keepalive = _require_finite(keepalive, "book_keepalive_s",
                                             f"{path}: meta_start")
                if keepalive < 0.0:
                    raise CaptureFormatError(
                        f"{path}: meta_start declares a negative book_keepalive_s "
                        f"({keepalive}) — a rate-limit floor cannot be negative.")
                if keepalive >= gap_detect_threshold_s:
                    # A.4(4): the emission floor MUST stay well under the reader's gap-detect
                    # resolution. At or above it, the recorder's own rate limit can space book
                    # events further apart than the threshold at which this reader starts
                    # calling a silence a gap — so "no gap was recorded" would be evidence of
                    # nothing, and a quiet-but-alive book is indistinguishable from a dark
                    # socket. Refuse the capture rather than report coverage computed under a
                    # floor that defeats the accounting.
                    raise CaptureFormatError(
                        f"{path}: meta_start declares book_keepalive_s={keepalive}s, at or "
                        f"above this reader's gap_detect_threshold_s={gap_detect_threshold_s}s. "
                        f"The recorder's book emission rate limit would then coarsen the record "
                        f"past the resolution gate (iv) reasons at; refusing rather than "
                        f"computing coverage that cannot mean what it says.")
                meta.declared_book_keepalive_s = keepalive
            fv = obj.get("format_version", 1)
            if fv not in SUPPORTED_FORMAT_VERSIONS:
                raise CaptureFormatError(
                    f"{path}: unsupported capture format_version {fv!r} (supported: "
                    f"{list(SUPPORTED_FORMAT_VERSIONS)}) — refusing rather than guessing at "
                    f"the meaning of fields written by a different format.")
            meta.format_version = fv
            if meta.clock_basis is not None and meta.clock_basis != CLOCK_BASIS:
                raise CaptureFormatError(
                    f"{path}: capture declares clock_basis {meta.clock_basis!r}, but this "
                    f"reader does gap accounting on {CLOCK_BASIS!r} only. A gap ledger keyed "
                    f"on a different clock than the events it is compared against is unsound; "
                    f"refusing rather than mixing bases.")
            if meta.channels is not None:
                unknown = [c for c in meta.channels if c not in DATA_CHANNELS]
                if unknown:
                    raise CaptureFormatError(
                        f"{path}: meta_start declares unknown data channel(s) {unknown} "
                        f"(known: {list(DATA_CHANNELS)})")
        elif t == "meta_end":
            meta.closed_cleanly = True
        elif t == "book":
            where = f"{path}: book record"
            meta.n_book += 1
            ts = _require_finite(obj.get("ts"), "ts", where)
            all_ts.append(ts)
            _require_price(obj.get("bid"), "bid", where)
            _require_price(obj.get("ask"), "ask", where)
            _note_data_event("book", obj.get("pair"), ts, where)
        elif t == "trade":
            where = f"{path}: trade record"
            meta.n_trade += 1
            ts = _require_finite(obj.get("ts"), "ts", where)
            all_ts.append(ts)
            _require_price(obj.get("price"), "price", where)
            if obj.get("venue_ts") is not None:
                _require_finite(obj.get("venue_ts"), "venue_ts", where)
            _note_data_event("trade", obj.get("pair"), ts, where)
        elif t == "heartbeat":
            meta.n_heartbeat += 1
            all_ts.append(_require_finite(obj.get("ts"), "ts", f"{path}: heartbeat record"))
        elif t == "counters":
            meta.counters_present = True
            meta.n_json_decode_errors = int(obj.get("json_decode_errors", 0))
            meta.n_schema_mismatch = int(obj.get("schema_mismatch", 0))
            meta.n_unrecognized_frames = int(obj.get("unrecognized_frames", 0))
        elif t == "gap":
            # GapRecord.__post_init__ refuses non-finite endpoints and negative durations.
            g = GapRecord(obj["start_ts"], obj["end_ts"], obj["reason"], obj.get("detail", ""))
            recorded_gaps.append(g)
        else:
            raise CaptureFormatError(f"{path}: unknown record type {t!r}")

    meta.pairs = sorted(p for p in pairs_seen if p)
    # The OBSERVED side of defense 4, built from the data-bearing events alone. `data_ts` only ever
    # gains a (pair, channel) key when an event was actually appended to it, so a declared cell that
    # never carried an event is simply absent here — which is the whole point.
    meta.observed_pair_channels = {pair: sorted(ch for ch, ts in data_ts[pair].items() if ts)
                                    for pair in sorted(data_ts)}
    meta.gaps = sorted(recorded_gaps, key=lambda g: g.start_ts)
    if all_ts:
        meta.start_ts = min(all_ts)
        meta.end_ts = max(all_ts)

    merged, n_overlap, n_dup = _merge_gap_spans(meta.gaps)
    meta.merged_gap_spans = merged
    meta.n_overlapping_gap_records = n_overlap
    meta.n_duplicate_gap_records = n_dup
    if meta.start_ts is not None and meta.end_ts is not None:
        # Disclosed, NOT refused: a gap lying entirely outside the observed event span cannot
        # overlap any inter-event jump, so it can mask nothing. It is legitimate too — a long
        # outage with repeated failed reconnects writes gaps past the last captured event.
        meta.n_gaps_outside_event_span = sum(
            1 for g in meta.gaps if g.end_ts < meta.start_ts or g.start_ts > meta.end_ts)

    all_ts.sort()
    meta.unrecorded_gaps = _find_unrecorded_gaps(all_ts, meta.gaps, gap_detect_threshold_s)

    # --- defense 2: the SAME per-jump check, data-bearing events only, per pair ---------------
    # Heartbeats are deliberately absent from `data_ts`. A degraded session whose heartbeats keep
    # flowing normally passes the check above and is caught here.
    all_data_ts: List[float] = []
    unrecorded_data: List[GapRecord] = []
    for pair in sorted(data_ts):
        pair_ts = sorted(t for chan_ts in data_ts[pair].values() for t in chan_ts)
        all_data_ts.extend(pair_ts)
        for g in _find_unrecorded_gaps(pair_ts, meta.gaps, gap_detect_threshold_s):
            unrecorded_data.append(GapRecord(
                g.start_ts, g.end_ts, g.reason,
                detail=f"pair {pair}: data-bearing (book+trade, heartbeats EXCLUDED) — "
                       f"{g.detail}"))
    meta.unrecorded_data_gaps = unrecorded_data
    all_data_ts.sort()
    if all_data_ts:
        meta.data_start_ts = all_data_ts[0]
        meta.data_end_ts = all_data_ts[-1]

    # --- defense 3: the AGGREGATE, cumulative, data-bearing coverage check ---------------------
    meta.cadence = _cadence_stats(data_ts)
    if meta.data_start_ts is not None and meta.data_end_ts is not None:
        meta.coverage_by_pair = {
            pair: _pair_coverage(pair, data_ts[pair], merged,
                                  meta.data_start_ts, meta.data_end_ts, coverage_window_s)
            for pair in sorted(data_ts)
        }
    meta.coverage_evaluated = True
    return meta


def _file_has_truncated_tail(path: str) -> bool:
    with open(path, "r") as fh:
        lines = fh.readlines()
    if not lines:
        return False
    last = lines[-1].rstrip("\n")
    if not last:
        return False
    try:
        json.loads(last)
        return False
    except json.JSONDecodeError:
        return True


def _merge_gap_spans(gaps: List[GapRecord]) -> Tuple[List[Tuple[float, float]], int, int]:
    """Coalesce recorded gaps into DISJOINT spans (the union), reporting how many records
    overlapped and how many were exact duplicates.

    This is the fix for the gate-(iv) defeat: coverage must be the MEASURE OF THE UNION of the
    recorded gaps, never the sum of their individual overlaps with a jump. Summing lets two
    recorded gaps of 500s each covering [0,500] and [300,800] "cover" 1000s of a 1000s jump,
    reporting a 200s hole as fully explained — while `covers()` still happily returns True for
    a window sitting squarely inside that hole. Exact duplicate records do the same.
    """
    if not gaps:
        return [], 0, 0
    spans = sorted((g.start_ts, g.end_ts) for g in gaps)
    n_duplicates = len(spans) - len(set(spans))
    merged: List[Tuple[float, float]] = []
    cur_start, cur_end = spans[0]
    n_overlaps = 0
    for start, end in spans[1:]:
        if start <= cur_end:
            if start < cur_end:
                n_overlaps += 1
            cur_end = max(cur_end, end)
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = start, end
    merged.append((cur_start, cur_end))
    return merged, n_overlaps, n_duplicates


def _merge_spans(spans: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Coalesce arbitrary (start, end) pairs into the sorted, DISJOINT union. Touching spans
    merge (a zero-length shared instant carries no measure either way)."""
    if not spans:
        return []
    ordered = sorted(spans)
    merged: List[Tuple[float, float]] = []
    cur_start, cur_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= cur_end:
            cur_end = max(cur_end, end)
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = start, end
    merged.append((cur_start, cur_end))
    return merged


def _measure_intersection(a: List[Tuple[float, float]],
                           b: List[Tuple[float, float]]) -> float:
    """Measure of union(a) INTERSECT union(b). Both inputs must already be merged (disjoint,
    sorted) — `_merge_spans` guarantees that. Linear sweep, no double counting."""
    total = 0.0
    i = j = 0
    while i < len(a) and j < len(b):
        lo = max(a[i][0], b[j][0])
        hi = min(a[i][1], b[j][1])
        if hi > lo:
            total += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def _pctile(sorted_vals: List[float], pct: float) -> Optional[float]:
    """Linear-interpolation percentile over an already-sorted, all-finite list."""
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = (pct / 100.0) * (len(sorted_vals) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = rank - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def _cadence_stats(data_ts: Dict[str, Dict[str, List[float]]]) -> List[ChannelCadence]:
    """Realized inter-event intervals per (pair, channel), plus a pooled `book+trade` row.

    Heartbeats never reach here — `data_ts` is built from data-bearing events only.
    """
    out: List[ChannelCadence] = []
    for pair in sorted(data_ts):
        by_chan = data_ts[pair]
        pooled = sorted(t for ts in by_chan.values() for t in ts)
        for channel, ts_list in ([(c, sorted(by_chan[c])) for c in sorted(by_chan)]
                                  + [("book+trade", pooled)]):
            if len(ts_list) < 2:
                out.append(ChannelCadence(pair, channel, len(ts_list), None, None, None))
                continue
            deltas = sorted(ts_list[i] - ts_list[i - 1] for i in range(1, len(ts_list)))
            out.append(ChannelCadence(pair, channel, len(ts_list),
                                       _pctile(deltas, 50), _pctile(deltas, 95), deltas[-1]))
    return out


def _pair_coverage(pair: str, by_channel: Dict[str, List[float]],
                    merged_gaps: List[Tuple[float, float]],
                    span_lo: float, span_hi: float, window_s: float) -> PairCoverage:
    """Aggregate data-bearing coverage for one pair over [span_lo, span_hi].

    An instant counts as OBSERVED iff a book/trade event for this pair lies within +/- window_s
    of it, so `covered` is the measure of the union of those event windows, clipped to the span
    and with the recorded-gap time removed (that time is already out of the denominator).
    """
    ts_list = sorted(t for ts in by_channel.values() for t in ts)
    span = [(span_lo, span_hi)]
    gaps_clipped = _merge_spans(
        [(max(s, span_lo), min(e, span_hi)) for s, e in merged_gaps
         if min(e, span_hi) > max(s, span_lo)])
    span_s = span_hi - span_lo
    recorded_gap_s = _measure_intersection(gaps_clipped, span)
    accounted_s = max(0.0, span_s - recorded_gap_s)
    ev = _merge_spans([(t - window_s, t + window_s) for t in ts_list])
    covered_in_span = _measure_intersection(ev, span)
    covered_in_gaps = _measure_intersection(ev, gaps_clipped)
    covered_s = max(0.0, covered_in_span - covered_in_gaps)
    return PairCoverage(pair=pair, n_data_events=len(ts_list), window_s=window_s,
                         span_s=span_s, recorded_gap_s=recorded_gap_s,
                         accounted_s=accounted_s, covered_s=min(covered_s, accounted_s))


def _find_unrecorded_gaps(sorted_ts: List[float], recorded_gaps: List[GapRecord],
                           threshold_s: float) -> List[GapRecord]:
    """Scan consecutive raw event timestamps for a jump > threshold whose UNCOVERED portion
    (jump duration minus the UNION of the recorded gaps overlapping it) still exceeds the
    threshold. This is the independent check: it does NOT trust the recorder's own gap list, it
    re-derives whether one was needed — but it tolerates a small residual around a recorded
    gap's edges (e.g. the recorder writes its resume gap at restart time, and the first fresh
    event a fraction of a second later is not itself a second, separate outage).

    Coverage is computed over MERGED (disjoint) spans, so overlapping or duplicated gap records
    cannot double-count their way over a real hole.
    """
    merged, _, _ = _merge_gap_spans(recorded_gaps)
    unrecorded = []
    for i in range(1, len(sorted_ts)):
        prev_t, cur_t = sorted_ts[i - 1], sorted_ts[i]
        jump = cur_t - prev_t
        if jump <= threshold_s:
            continue
        covered_duration = sum(
            max(0.0, min(end, cur_t) - max(start, prev_t)) for start, end in merged
        )
        uncovered = jump - covered_duration
        if uncovered > threshold_s:
            unrecorded.append(GapRecord(prev_t, cur_t, "seq_discontinuity",
                                         detail=f"reader-detected: {uncovered:.1f}s of this "
                                                f"{jump:.1f}s jump has no recorded gap "
                                                f"covering it"))
    return unrecorded


def covers(meta: CaptureMetadata, t0: float, t1: float) -> bool:
    """True iff the window [t0, t1] lies inside the capture's span and is free of ANY recorded
    gap (gate iv also applies at the per-markout level, not just at the whole-capture level: a
    markout whose horizon lands inside a recorded gap must not be computed as if the gap didn't
    happen).

    Boundary semantics, pinned by `test_capture_gaps.py`: the window must be fully contained in
    [start_ts, end_ts] (inclusive at both ends), and a gap only disqualifies it on a STRICT
    overlap — a window that merely touches a gap's edge (t1 == gap.start_ts, or
    t0 == gap.end_ts) is clean, because a zero-length shared instant carries no unobserved
    wire time.
    """
    if meta.start_ts is None or meta.end_ts is None:
        return False
    if not (math.isfinite(t0) and math.isfinite(t1)):
        return False
    if t0 < meta.start_ts or t1 > meta.end_ts:
        return False
    return not any(g.start_ts < t1 and g.end_ts > t0 for g in meta.gaps)
