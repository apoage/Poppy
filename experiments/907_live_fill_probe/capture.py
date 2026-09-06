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
import bisect
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
    "schema_drift", "book_checksum_mismatch",
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

#: Cycle 2 (run-45) — the keepalive reconciliation (spec lines 816-821). Addendum A made the
#: recorder emit a book event on top-of-book CHANGE **or** a `book_keepalive_s` timer floor, and
#: declare both keys into `meta_start`. Nothing read them: a two-hour FROZEN book, emitting an
#: identical top of book every 5.0s, reported `uncovered_fraction 0.0`, `accounted_s 7200.0` and a
#: PASSING `assert_complete()` — indistinguishable from a healthy one.
#:
#: The shipped record carries no per-event flag saying "this one was keepalive-driven", and the
#: live capture forbids adding one, so the detection is READER-SIDE and value-based: a run of
#: consecutive book events for one pair whose top-of-book PRICE PAIR `(bid, ask)` is UNCHANGED
#: and which lasts longer than this many seconds is a STALE run, and its repeats do not count as
#: observations of the market. The threshold is the bar interval, for the same reason the
#: gap-detect threshold is: a stretch long enough to empty a whole bar of new book information is
#: long enough to flag, and anything shorter is absorbed by the coverage window anyway.
#:
#: **The key is `(bid, ask)` and NOT `(bid, ask, bid_sz, ask_sz)` (run-49 BLOCKER).** The run key
#: must never be FINER than the quantity it protects. The estimand this staleness term defends is
#: computed from the MID alone (`shadow_engine.py`, `(bid + ask) / 2`); sizes reach no estimand.
#: Keying on sizes too meant any size tick broke the "stale run" while the mid stayed frozen —
#: measured end to end: a price pinned at 100.0/100.2 for 7200s with sizes cycling through seven
#: values reported `stale_book_s 0.0`, `uncovered_fraction 0.0`, an empty `stale_book_pairs`, a
#: PASSING `assert_complete()` and a published `1m median 2.48 bp` table on a mid that never
#: moved. Re-keyed, the same capture reports `stale_book_s 7195.0`. This is the same
#: wrong-denomination shape as the cycle-3 BLOCKER that forced the D73 split, one field over.
#:
#: **WHAT A STALE RUN UNDER THIS KEY DOES AND DOES NOT ASSERT — read before quoting the number.**
#: `recorder._parse_book_frame` emits on `(bid_px, bid_qty, ask_px, ask_qty)` CHANGE or on the
#: keepalive timer, so a repeat of `(bid, ask)` can be either (a) a timer firing with no wire
#: frame behind it, or (b) a real wire frame in which only the SIZES moved. The shipped record
#: carries no per-event flag distinguishing them and the live capture forbids adding one, so this
#: reader cannot tell them apart and counts both. The defensible reading is therefore **"the mid
#: these estimands are built from carried no new information across this stretch"** — NOT "the
#: wire was not observed". Measured on the live 3.586-day capture, where the difference is large:
#: 2,016,348 of BTC/USD's 2,279,026 book events repeat the previous `(bid, ask)`, and 127,071s of
#: 273,233s accounted (46.5%) falls inside a >60s frozen-price run. Interpretation (a) vs (b) for
#: that time is not resolvable in B1 and is not resolved here.
#:
#: **AND WHAT IT DOES NOT COUNT AT ALL (run-49 cycle 3, disclosure gap).** The detector counts one
#: shape: a run of CONSECUTIVE IDENTICAL `(bid, ask)` events. Any change to the price pair —
#: INCLUDING A CHANGE BACK — ends the run. A mid that OSCILLATES over a small set of values, or
#: otherwise moves without ever discovering a new price, is therefore not counted by this measure
#: and is not detected anywhere in ADDENDUM B1. Measured: a top of book alternating between two
#: `(bid, ask)` values for the whole of a 7200s capture reports `stale_book_s 0.0`, an empty
#: `stale_book_pairs()`, `uncovered_fraction 0.0000`, no FROZEN-MID line in the artifact, and a
#: published estimand table (`test_coverage_aggregate.py::
#: test_an_OSCILLATING_dead_mid_reads_stale_book_s_ZERO_and_is_NOT_detected_in_B1`). So
#: `stale_book_s == 0.0` is NOT evidence that the mid was informative; it is evidence only that no
#: LITERAL REPEAT ran longer than this threshold. The denomination-correct instrument for "this
#: mid discovered no new prices" is ADDENDUM B2's book-mid FRESHNESS term. Widening THIS key to
#: catch period-N cycles, bounded-cardinality value sets, or "no net movement over a window" is
#: not the fix — it is the B2 term built in the wrong denomination, and it is the unbounded
#: escaping-input chase that halted TASK_891. Not attempted here, deliberately.
STALE_BOOK_MIN_RUN_S = BAR_INTERVAL_S

#: How far past its DECLARED `book_keepalive_s` floor a book inter-event interval may stretch
#: before it is counted as an overrun (the mirror direction of the same reconciliation). 2x: the
#: floor is a rate LIMIT the recorder promised to respect, and a single missed timer tick is
#: scheduler noise, while 6x (the 30s-hole case) is the recorder's own declaration contradicted.
BOOK_KEEPALIVE_OVERRUN_FACTOR = 2.0

#: How far INSIDE a recorded gap's own HIGH edge a data-bearing event may fall before the ledger
#: is called self-contradictory (run-49). Justified from the MECHANISM, not tuned to a file.
#:
#: Mechanism: `recorder.py` samples the receive clock once per frame and stamps that frame's
#: events with it; `_record_gap` used to sample a SECOND, strictly later clock and write THAT as
#: the gap's `end_ts`. So the very event that TERMINATED an outage was stamped a few microseconds
#: BEFORE the end of its own gap and landed strictly inside it — structurally, on every
#: `clock_jump`, under a perfectly monotonic clock. `_record_gap` now takes the caller's
#: already-sampled `now`, so captures written from here on have ZERO such intrusion. This
#: tolerance exists ONLY for captures already on disk under the old recorder — including the live
#: multi-day capture, which cannot be re-recorded.
#:
#: Magnitude: the intrusion is bounded by the interpreter time between two `time.monotonic()`
#: calls in one code path. Measured over all 10,196,132 lines of the live capture: 584 affected
#: events, min 5.007us / median 12.159us / max **32.902us**, 584/584 on the HIGH edge of an
#: individual `clock_jump` gap. 1ms is ~30x that measured bound and still three orders of
#: magnitude below the smallest shape it must NOT excuse: the merged-span seam (an event sitting
#: on one gap's `end_ts` and the next gap's `start_ts`) sat 2.537s deep at its shallowest, and the
#: reader's own `gap_detect_threshold_s` is 60s. HIGH EDGE ONLY, and deliberately not symmetric: a
#: gap's `start_ts` is `_last_event_ts` (an event's own timestamp, exact) or the same `now`, so no
#: mechanism puts an event just inside the LOW edge, and a tolerance there would excuse a shape
#: nothing is known to produce. Do NOT widen this to make a file pass — the seam shape is excluded
#: by testing containment against INDIVIDUAL gaps rather than merged spans, not by this number.
GAP_EDGE_CLOCK_SAMPLE_TOLERANCE_S = 1e-3


class CaptureFormatError(ValueError):
    """A capture line or file violates the append-only capture contract."""


def _fmt_uncovered_fraction(value: Optional[float]) -> str:
    """Render `PairCoverage.uncovered_fraction` — including its NO-CLAIM sentinel (cycle 2).

    The sentinel must never render as a percentage: `0.000%` is the instrument's strongest
    coverage claim, and `None` means it made none at all.
    """
    return "NO CLAIM (accounted_s=0)" if value is None else f"{value * 100:.3f}%"


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

        def _declaration_fields(record_type: str) -> Dict[str, Any]:
            """The subscription-config fields common to `meta_start` and `meta_resume` (B1,
            run-45): every writer session declares its OWN config, not just the first. Before
            this, the resume branch below wrote only a gap record and silently DISCARDED
            `pairs`/`channels`/`heartbeat_timeout_s` — so a config change on restart (e.g. a pair
            added to the subscription) was invisible to the reader, and the declared side of
            gate (iv) stayed pinned to session 1's config forever.
            """
            fields: Dict[str, Any] = {"type": record_type, "capture_id": capture_id,
                                       "pairs": list(pairs)}
            if self.channels is not None:
                fields["channels"] = list(self.channels)
            if heartbeat_timeout_s is not None:
                # Declared so the artifact can print the recorder's liveness knob next to the
                # reader's gap-detect knob. The two used to be independently configured with no
                # code coupling them and nothing stating either one in the output.
                fields["heartbeat_timeout_s"] = _require_finite(
                    heartbeat_timeout_s, "heartbeat_timeout_s", f"{path}: {record_type}")
            if book_emission_policy is not None:
                # ADDENDUM A (run 45): the book channel is no longer sampled verbatim — it is
                # emitted on change-or-keepalive. An undeclared emission policy would be a
                # fourth unreconciled declaration (A.4(4)). These two keys are READ BACK by
                # `read_capture` (`declared_book_emission_policy` / `declared_book_keepalive_s`)
                # and printed next to the cadence table by `render._render_coverage_block` — a
                # declaration written but never read is the very failure A.4(4) names, so the
                # write end and the read end land together.
                fields["book_emission_policy"] = str(book_emission_policy)
            if book_keepalive_s is not None:
                fields["book_keepalive_s"] = _require_finite(
                    book_keepalive_s, "book_keepalive_s", f"{path}: {record_type}")
            return fields

        if not self._is_resume:
            meta = _declaration_fields("meta_start")
            meta["format_version"] = CURRENT_FORMAT_VERSION
            meta["clock_basis"] = CLOCK_BASIS
            self._write(meta)
        else:
            # B1 (run-45): declare THIS session's config into the capture BEFORE the resume gap,
            # so a reader sees "session 2 declared X" as a fact about the capture rather than
            # losing it the moment the process exits. `ts` is informational provenance only —
            # `meta_resume` is not in `_EVENT_TYPES_FOR_LAST_TS`, so it never enters gap
            # arithmetic, exactly like `meta_start`.
            now = _now()
            resume_meta = _declaration_fields("meta_resume")
            resume_meta["ts"] = now
            self._write(resume_meta)
            last_ts = _last_event_ts(path)
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


_NO_PENDING = object()  #: sentinel — "the buffered line failed to parse", distinct from any obj


def _iter_lines_tolerant(path: str,
                          truncated_tail_out: Optional[List[bool]] = None,
                          ) -> Iterator[Dict[str, Any]]:
    """Yield parsed JSON lines, streaming the file in ONE forward pass, silently dropping ONLY a
    truncated/corrupt trailing line.

    A non-trailing corrupt line is a real format violation and raises; the guarantee this
    format offers is "a kill mid-write cannot corrupt already-captured data", not "any
    corruption anywhere is tolerated".

    TASK_914: this used to `fh.readlines()` the whole file before yielding anything — one of the
    three full-file materializations behind the reader's O(file) RSS. A streaming pass has no
    `len(lines)`, so "is this the last physical line" (the condition tolerance is keyed to) is
    decided with a one-line lookahead instead: each line's parse outcome is buffered and only
    resolved (yielded, or raised as non-trailing corruption) once the NEXT physical line has been
    read — at which point it is certain the buffered one was not last. Blank lines are inert at
    every position (never yielded, never raise, never trigger tolerance) exactly as before; only
    a non-blank line's "lastness" matters.

    `truncated_tail_out`, if given, is a single-element list used purely as an out-parameter: it
    is set to `True` iff the file's own last physical line is non-blank and fails to parse —
    folding what `_file_has_truncated_tail` used to get from its OWN second `readlines()` pass
    into this same single pass (T2). Left `False` (its initial value) otherwise.
    """
    if truncated_tail_out is not None:
        truncated_tail_out[0] = False
    have_pending = False
    pending_is_blank = False
    pending_obj: Any = _NO_PENDING
    pending_line_no = -1
    with open(path, "r") as fh:
        for line_no, raw in enumerate(fh):
            if have_pending:
                # A further physical line exists, so the buffered one is confirmed non-trailing.
                if pending_is_blank:
                    pass
                elif pending_obj is not _NO_PENDING:
                    yield pending_obj
                else:
                    raise CaptureFormatError(
                        f"{path}: corrupt non-trailing line {pending_line_no} is not valid JSON")
            stripped = raw.rstrip("\n")
            pending_is_blank = not stripped
            pending_obj = _NO_PENDING
            if not pending_is_blank:
                try:
                    pending_obj = json.loads(stripped)
                except json.JSONDecodeError:
                    pending_obj = _NO_PENDING
            pending_line_no = line_no
            have_pending = True
    if have_pending:
        # EOF: resolve the true last physical line.
        if pending_is_blank:
            pass
        elif pending_obj is not _NO_PENDING:
            yield pending_obj
        elif truncated_tail_out is not None:
            truncated_tail_out[0] = True  # truncated trailing write — tolerated, not corruption


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
    """Aggregate DATA-BEARING coverage for one pair over the capture's whole CAPTURE span.

    `accounted_s` is the time the capture CLAIMS to have observed: its span minus the union of
    its recorded gaps. Recorded gaps are honest, disclosed absence and markouts across them are
    already dropped by `covers()`, so they are excluded from the denominator rather than counted
    against it. What this measures is the UNDISCLOSED remainder: accounted time that no
    data-bearing event came within `window_s` of.

    `span_s` (and the `recorded_gap_s`/`accounted_s`/`covered_s` derived from it) is denominated
    over the capture's whole CAPTURE span — `[start_ts, end_ts]`, ALL events including heartbeats
    — not the pair's own data-bearing span (B2/run-45). Heartbeats prove the socket was up, which
    is exactly what makes their span the right denominator for "did we observe the market?": a
    pair whose data-bearing events stop 2 hours into a 2-hour-plus-tail capture while heartbeats
    keep flowing must have that dead tail counted AS un-observed time, not excluded from the
    denominator along with the events that would have proven it dead. Denominating over the
    pair's OWN data span was the run-45 BLOCKER: a 7200s capture carrying 130.5s of data reported
    `accounted_s=130, span_s=130` — the denominator collapsed together with the numerator, so
    `uncovered_fraction` read 0.000% against a true 98.188% of the capture un-observed. Also
    computed over the capture's GLOBAL span rather than the pair's own for the reason already
    documented here — otherwise a pair that goes dead on day 2 of a 10-day capture would be
    judged over day 1 alone and pass.
    """
    pair: str
    n_data_events: int
    window_s: float
    span_s: float
    recorded_gap_s: float
    accounted_s: float
    covered_s: float
    #: Seconds of this pair's accounted time during which the recorder's `book` channel emitted a
    #: BYTE-IDENTICAL top of book on its declared keepalive timer (cycle 2, run-45 — the keepalive
    #: MAJOR, spec lines 816-821). Those emissions are excluded from `covered_s`: a keepalive
    #: proves the recorder's own timer, not that the market was observed. Zero whenever the
    #: capture declares no `book_emission_policy`/`book_keepalive_s` — there is then no declared
    #: quantity to reconcile the observed cadence against, which is the D67 precondition.
    stale_book_s: float = 0.0
    #: The longest CONTIGUOUS observed run inside accounted time, seconds. Used only by the
    #: refusal's remedy text (cycle 3, item 2): it is the measurable difference between "there is
    #: a healthy stretch of this capture a re-cut could keep" and "there is not", and the old
    #: single remedy ("re-cut around the degraded stretch") was vacuous for the second case.
    longest_observed_run_s: float = 0.0

    @property
    def uncovered_s(self) -> float:
        return max(0.0, self.accounted_s - self.covered_s)

    @property
    def uncovered_fraction(self) -> Optional[float]:
        """The un-observed fraction of accounted time — or `None`, "NO CLAIM MADE", fail-CLOSED.

        Spec B2 item 3 is unconditional: this branch is fail-CLOSED and `0.0` is the one value
        that must never be returned here. `0.0` is indistinguishable from a measured "0% of this
        pair's accounted time was un-observed", which is the single strongest coverage claim the
        instrument can make — and `accounted_s <= 0` is precisely the state in which it has
        measured nothing at all. Cycle 1 left this returning `0.0` under a comment arguing the
        enumeration of reachable cases was safe; the enumeration missed a third case (a recorded
        gap that SWALLOWS REAL EVENTS — 1442 data events inside a 7200s gap gave `accounted_s=0`,
        `uncovered_fraction=0.0`, `degraded_coverage_pairs()==[]` and a PASSING `assert_complete`),
        which is now refused outright by `CaptureMetadata.events_inside_recorded_gaps`.

        A SENTINEL rather than `1.0`: the requirement is that this branch cannot BACK a `0.0`, not
        that it must claim total absence. Returning `1.0` manufactures a refusal for captures with
        nothing hidden — a genuinely zero-length span, or a span whose absence is fully and
        honestly disclosed by the gap ledger — which contradicts gate (iv)'s own purpose (refuse
        what is UNDISCLOSED, not whatever makes the arithmetic degenerate). `None` is neither: it
        is not a percentage, it cannot be compared against the bound, and every consumer must
        handle it explicitly. `render` prints it as `NO CLAIM`, `degraded_coverage_pairs` cannot
        grade it as within-bound, and `no_claim_coverage_pairs` names it in the artifact.
        """
        if self.accounted_s <= 0.0:
            return None
        return min(1.0, self.uncovered_s / self.accounted_s)


@dataclass(frozen=True)
class SessionDeclaration:
    """One writer session's declared subscription config (B1, run-45).

    Written once per `CaptureWriter` process: `kind="start"` for the file's first session
    (`meta_start`), `kind="resume"` for every later session that reopened the same file
    (`meta_resume`). Before B1, a resumed session's `pairs`/`channels`/`heartbeat_timeout_s` were
    silently DISCARDED — `capture.py`'s resume branch wrote only a gap record — so a config
    change on restart (e.g. subscribing to a newly-added pair) was invisible to the reader: the
    declared side of gate (iv) stayed pinned to session 1's config forever, and the new pair's
    total absence read as "never declared" rather than "declared and dead", which is a WEAKER
    refusal than the one that shape deserves (or, at the boundary, no refusal at all).
    """
    kind: str  # "start" | "resume"
    pairs: List[str]
    channels: Optional[List[str]]
    heartbeat_timeout_s: Optional[float]


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
    #: B1 (run-45): one entry per writer session that declared a config (`meta_start` plus every
    #: `meta_resume`), in file order. `declared_pairs`/`channels` are the UNION across every
    #: session here — reconciling only the FIRST session's declaration is exactly the defect a
    #: pair added on restart escaped through. A single-session capture (the historical shape;
    #: still real, still produced by this writer) has exactly one entry with `kind="start"`.
    sessions: List[SessionDeclaration] = field(default_factory=list)
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
    #: Cycle 2 (run-45): data-bearing events whose timestamp falls STRICTLY INSIDE a recorded gap
    #: span — a self-contradictory ledger, and the third `accounted_s <= 0` case the cycle-1
    #: enumeration missed. Human-readable descriptors, capped; the full count is
    #: `n_events_inside_recorded_gaps`. Strictly inside on purpose: boundary contact is the
    #: normal, healthy shape (a gap starts at `_last_event_ts` and `_close_dark_window` closes at
    #: the first event after the outage, so both endpoints ARE event timestamps) and only
    #: containment is the contradiction.
    #:
    #: Run-49 corrections, both measured on the live capture rather than reasoned about:
    #:   * tested against INDIVIDUAL gap records, never `merged_gap_spans` — two gaps that merely
    #:     TOUCH (a `disconnect` dark window ending at the first event after reconnecting, and the
    #:     `clock_jump` starting at that same event) union into a span whose seam reads as
    #:     containment though neither gap contains anything. 14 events of the live capture's 598
    #:     were exactly this and nothing else.
    #:   * the high edge carries `GAP_EDGE_CLOCK_SAMPLE_TOLERANCE_S`, because the OLD recorder
    #:     sampled a second, later clock for a gap's `end_ts` than the one its own frame's events
    #:     were stamped with — so a `clock_jump`'s terminating event landed microseconds inside
    #:     its own gap, structurally. 584 events of the live capture's 598 were exactly this. The
    #:     recorder no longer does it (`_record_gap(..., now=...)`); the tolerance is for captures
    #:     already on disk.
    events_inside_recorded_gaps: List[str] = field(default_factory=list)
    n_events_inside_recorded_gaps: int = 0
    #: Cycle 2 (run-45), the MIRROR of the staleness check over the same Addendum-A declaration:
    #: book inter-event intervals that EXCEED the declared `book_keepalive_s` floor by more than
    #: `BOOK_KEEPALIVE_OVERRUN_FACTOR`, after removing recorded-gap time. The recorder DECLARED it
    #: would emit at least every `book_keepalive_s`; an interval well past that contradicts the
    #: declaration, and at a 5s floor against the reader's 60s gap-detect threshold a 30s book
    #: silence is invisible to every other check here. DISCLOSED, deliberately not refused — a
    #: scheduler hiccup must not retro-refuse a multi-day capture that cannot be repeated.
    #: The scan is ANCHORED TO THE CAPTURE SPAN (run-49), exactly as `_find_unrecorded_gaps` is:
    #: scanning strictly BETWEEN consecutive book events made the worst case — a book that emits
    #: on time and then simply stops, or starts late — invisible, because no consecutive pair of
    #: book events spans a hole at an edge of the timeline.
    book_keepalive_overruns: Dict[str, int] = field(default_factory=dict)
    book_keepalive_max_interval_s: Dict[str, float] = field(default_factory=dict)

    def no_data_events(self) -> bool:
        """True iff this capture carries no data-bearing event at all (in-gate defect #3's last
        un-routed field, run-49).

        The twins DID still diverge here. `is_complete()` asked `(n_book + n_trade) > 0` and
        `assert_complete()` asked `(n_book + n_trade) == 0`, which are NOT complements: for a
        NEGATIVE sum both are False, so `is_complete()` read incomplete while `assert_complete()`
        sailed past its own guard — the unsafe direction, and exactly the shape in-gate defect #3
        (the `had_meta_start is False` identity test) was raised about. `read_capture` only ever
        increments these counters so it is unreachable from a real file, but `capture.py` CLAIMS
        (see `is_complete`) that every condition appears in both twins "via the same helper
        methods", and that claim was false for this one field. Defect #3's instruction was
        "either route it or drop the claim"; this routes it. `<= 0` on purpose, not `== 0`: a
        negative count is nonsense, and nonsense must land on the refusing side.
        """
        return (self.n_book + self.n_trade) <= 0

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

    def declared_channels_explicitly_empty(self) -> bool:
        """True iff `channels` was declared as an EXPLICIT empty list (defect #4, run-45).

        `channels=None` ("undeclared") is a legitimate, disclosed fallback that SOFTENS the
        per-cell check — synthetic/replay/v1 fixtures produce it, and `CaptureWriter` only emits
        the field when its caller supplies one. `channels=[]` is a different shape: a capture
        that explicitly declares it subscribed to NO data channel at all, which cannot describe a
        capture that has data — the identical shape `declared_pairs=[]` is refused for one field
        over, and `missing_channels()`'s `if not self.channels: return []` treated the two
        identically (silently disabling both per-capture channel checks), which is the
        inconsistency this method exists to name and `assert_complete` to refuse.
        """
        return self.channels is not None and len(self.channels) == 0

    def disclosed_empty_trade_cells(self) -> List[str]:
        """Pairs whose declared `trade` channel carried ZERO events while `book` proves the pair
        is alive (in-gate MAJOR #1, run-45) — DISCLOSED, never refused.

        The over-refusal this closes: a legitimately quiet pair (a live book, no prints in the
        window) used to be indistinguishable from a dead subscription, and `capture.py`'s
        per-cell check refused the WHOLE render over it — discarding every number from every
        OTHER, healthy pair in the same capture. The asymmetry is deliberate: missing `book` on
        an observed pair still REFUSES via `pair_reconciliation_gaps`'s
        `declared_channel_without_data` (an actively-printing pair must have a book, so its
        absence is a broken subscription, not a market fact) — only the missing-`trade`-with-
        book-alive direction is downgraded to disclosure.
        """
        if not self.channels or "trade" not in self.channels:
            return []
        declared = set(self.declared_pairs or [])
        observed = set(self.observed_pairs)
        out = []
        for pair in sorted(declared & observed):
            chans = set(self.observed_pair_channels.get(pair, ()))
            if "book" in chans and "trade" not in chans:
                out.append(pair)
        return out

    def pair_reconciliation_gaps(self) -> Dict[str, List[str]]:
        """Defense 4: reconcile the DECLARED subscription against what was actually observed.

        Returns a dict of failure-kind -> offending items, every value empty when the capture
        reconciles. All six directions are failures, because each one is a way for the artifact's
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
            anywhere — absence again, one granularity down. ASYMMETRIC (in-gate MAJOR #1,
            run-45): a missing `trade` cell on a pair whose `book` is present is NOT counted here
            — see `disclosed_empty_trade_cells` — because no prints in the window is a market
            fact an alive book proves, not a broken subscription. A missing `book` cell (or a
            missing `trade` cell where `book` is ALSO absent — which cannot happen once the pair
            is `declared & observed`, since that requires at least one channel) still refuses.
          * ``observed_undeclared`` / ``coverage_undeclared`` — data (or a coverage record) for a
            pair the capture never declared. Unattributable in the other direction: a number backed
            by a pair no subscription asked for is not a number this instrument measured.
          * ``observed_channel_undeclared`` — data on a channel the capture never declared at all
            (in-gate MAJOR #5, run-45), as ``pair:channel``. `channels=["trade"]` on a capture
            full of `book` events used to reconcile perfectly (this key did not exist), so every
            primary estimand number rested on a channel the capture never declared — the mirror
            of `declared_channel_without_data` one level up, over CHANNELS rather than pairs.

        When `channels` is undeclared the per-cell checks are skipped — the same disclosed
        fallback the rest of this module uses for `channels=None`, and `render` prints
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
            "observed_channel_undeclared": [],
        }
        if self.channels:
            declared_channel_set = set(self.channels)
            cell_gaps = []
            for pair in sorted(declared & observed):
                observed_chans = set(self.observed_pair_channels.get(pair, ()))
                for channel in self.channels:
                    if channel in observed_chans:
                        continue
                    if channel == "trade" and "book" in observed_chans:
                        continue  # disclosed, not gated — see `disclosed_empty_trade_cells`
                    cell_gaps.append(f"{pair}:{channel}")
            gaps["declared_channel_without_data"] = sorted(cell_gaps)
            gaps["observed_channel_undeclared"] = sorted(
                f"{pair}:{channel}"
                for pair in sorted(observed)
                for channel in self.observed_pair_channels.get(pair, ())
                if channel not in declared_channel_set)
        return gaps

    def degraded_coverage_pairs(self) -> List[str]:
        """Pairs whose CUMULATIVE un-observed fraction of accounted time exceeds the declared
        bound — the aggregate defense, and a WHOLE-CAPTURE refusal.

        The per-jump checks are threshold checks on the LARGEST single hole; they say nothing
        about a degradation spread thinly across the whole capture. A capture whose book/trade
        events arrive every 59s for ten days clears every per-jump check at a 60s threshold while
        never once resolving the minute grid its estimands are denominated in.

        Denominated in `uncovered_fraction`, which already treats a keepalive-frozen book run as
        un-observed past its first event (`STALE_BOOK_MIN_RUN_S`, `_pair_coverage`'s
        `_is_stale_repeat`).

        **THE EXACT REACH OF THIS CHECK OVER STALE BOOKS, stated narrowly because the wider claim
        this docstring used to make is measurably false (run-49).** `uncovered_fraction` is
        denominated over the UNION of a pair's `book` AND `trade` timestamps. Stale book time
        therefore breaches the bound only when nothing ELSE is observing that pair over the same
        stretch. Concretely:

          * a pair whose book freezes AND whose trades stop — caught here, whole-capture refusal;
          * a pair whose book freezes or goes silent while its TRADES KEEP FLOWING — **NOT caught,
            here or anywhere in B1.** Measured: a book frozen for 7195s of a 7200s capture with
            trades every 30s yields `uncovered_fraction = 0.00132` against a 2% bound, an empty
            `degraded_coverage_pairs()`, `rc=0`, and a published estimand table built on a mid
            that never moved. The trade stream keeps the union dense, so the aggregate check
            cannot reach it — a check whose denominator does not collapse with the thing it
            measures, and therefore does not fire.

        Both stale-book shapes this reader can MEASURE (frozen-price and book-silent) fall in the
        second bullet the moment the pair is also printing. A third shape is not measured at all:
        an OSCILLATING mid, which breaks every literal `(bid, ask)` repeat run and so contributes
        zero `stale_book_s` — see `STALE_BOOK_MIN_RUN_S`. "Not in `degraded_coverage_pairs()`" is
        thus two steps away from "the mid was informative", not one.

        Closing all of it needs a BOOK-MID-ONLY freshness term,
        denominated independently of trade coverage — which is exactly the D73 diagnosis of why
        the cycle-3 attempt was wrong, and is **ADDENDUM B2 scope, not implemented here**. Nothing
        in B1 should be read as evidence that a stale book backed by live prints has been checked.
        The per-pair EXCLUSION that would let a multi-pair capture render around such a pair is
        B2's too; B1's interim behaviour, within the reach stated above, is the pre-cycle-3
        whole-capture refusal.
        """
        return [pair for pair, cov in sorted(self.coverage_by_pair.items())
                if cov.uncovered_fraction is not None
                and cov.uncovered_fraction > self.max_uncovered_fraction]

    def no_claim_coverage_pairs(self) -> List[str]:
        """Pairs whose `uncovered_fraction` is the NO-CLAIM sentinel (cycle 2, run-45).

        `accounted_s <= 0` means this pair's coverage arithmetic has no denominator, so it has
        made no coverage claim at all. That is NOT the same as "0% un-observed" (the value cycle 1
        returned) and it is NOT the same as a bound violation either — the two reachable honest
        shapes are a genuinely zero-length capture span and a span whose absence the gap ledger
        discloses in full. Named here so the artifact prints `NO CLAIM` instead of a percentage,
        and so `degraded_coverage_pairs` cannot silently grade an absent claim as within-bound.
        The dishonest shape that used to hide in this branch — a gap swallowing real events — is
        refused separately by `events_inside_recorded_gaps`.
        """
        return [pair for pair, cov in sorted(self.coverage_by_pair.items())
                if cov.uncovered_fraction is None]

    def stale_book_pairs(self) -> List[str]:
        """Pairs with ANY keepalive-driven STALE book time (cycle 2, run-45). Disclosure only, at
        any magnitude: a pair loses exactly its stale seconds out of `covered_s` and nothing more
        — those seconds are already folded into `uncovered_fraction`.

        B1 ships NO per-pair exclusion for this, and — stated narrowly, per run-49 — NO reliable
        whole-capture refusal either. The aggregate check (`degraded_coverage_pairs`) can only see
        stale book time that pushes the pair's UNION (book+trade) coverage past the bound, so it
        fires only when the pair's TRADES have stopped too. A pair whose book is frozen or silent
        while its prints keep flowing is disclosed HERE and passes every gate; catching it needs
        the book-mid-only freshness term that is ADDENDUM B2 scope. Read this list as a
        MEASUREMENT of stale book time, never as evidence that staleness has been checked and
        cleared for the pairs absent from it.

        The run key is `(bid, ask)` — the price pair the estimands' mid is built from — and NOT
        the sizes: see `STALE_BOOK_MIN_RUN_S`.

        **AND THE MEASURE ITSELF REACHES ONLY LITERAL REPEATS.** "Stale" here means a run of
        CONSECUTIVE IDENTICAL `(bid, ask)` events; any change to the price pair, including a change
        back, ends the run. A mid that OSCILLATES over a small set of values — or otherwise moves
        without discovering a new price — is not counted by this measure and is not detected
        anywhere in B1: measured, a top of book alternating between two values for a whole 7200s
        capture reports `stale_book_s 0.0` and an empty list here. A pair's absence from this list,
        and a `stale_book_s` of 0.0, therefore say only "no literal repeat ran long enough", never
        "this mid was informative". The instrument for the latter is B2's book-mid freshness term;
        see `STALE_BOOK_MIN_RUN_S` for why widening this key is not that term."""
        return [pair for pair, cov in sorted(self.coverage_by_pair.items())
                if cov.stale_book_s > 0.0]

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
        return (bool(self.had_meta_start) and not self.no_data_events()
                and not self.missing_channels()
                and not self.declared_channels_explicitly_empty()
                and not self.unrecorded_gaps
                and not self.unrecorded_data_gaps
                and not self.events_inside_recorded_gaps
                and self.coverage_evaluated
                and not self.declared_pair_set_missing()
                and not any(self.pair_reconciliation_gaps().values())
                and not self.degraded_coverage_pairs())

    def assert_complete(self) -> None:
        # `not self.had_meta_start`, NOT `is False` (in-gate defect #3, run-45): `had_meta_start`
        # is typed `bool` but nothing stops a hand-built `CaptureMetadata` from carrying an
        # int-like falsy value (e.g. `0`). `0 is False` is `False` in Python — the two are equal
        # but not identical — so an IDENTITY check here let `is_complete()` (which uses `and`,
        # i.e. truthiness) read as incomplete while this raising twin sailed past the check
        # entirely. `bool(...)` above and `not self.had_meta_start` below are the same falsiness
        # test, so the two can no longer diverge on this field.
        if not self.had_meta_start:
            raise CaptureFormatError(f"{self.path}: no meta_start record — refusing to treat "
                                      f"this as a capture at all")
        # `no_data_events()`, NOT a second inline `== 0` (run-49): `> 0` and `== 0` are not
        # complements over the integers, so a negative sum made this twin PASS while
        # `is_complete()` read incomplete. One helper, both call sites.
        if self.no_data_events():
            raise CaptureFormatError(
                f"{self.path}: capture has zero data events "
                f"(book={self.n_book}, trade={self.n_trade})")
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
        if self.declared_channels_explicitly_empty():
            raise CaptureFormatError(
                f"{self.path}: GATE (iv) REFUSAL — this capture DECLARES AN EXPLICITLY EMPTY "
                f"CHANNEL SET (`meta_start`/`meta_resume` `channels` is `[]`, not absent). The "
                f"identical shape as `declared_pairs=[]` (refused above) one field over: a "
                f"capture that explicitly subscribed to nothing cannot describe a capture that "
                f"has data. Contrast `channels` being ABSENT (`None`), which is the legitimate, "
                f"disclosed 'undeclared' fallback synthetic/replay/v1 fixtures produce and which "
                f"only softens the per-cell check — an explicit empty list is a different, "
                f"nonsensical declaration and is refused rather than silently treated the same "
                f"as undeclared.")
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
            # B2/B-ii (run-45): state the TRUE un-observed fraction of the capture span here too,
            # not just the raw gap spans — this check (defense 2, per-jump, anchored to the
            # capture span since B2 item 1) is now often what catches a pair whose data-bearing
            # events stop long before the capture ends, and a reader must not have to cross-
            # reference the aggregate table to learn the magnitude of what this refusal is about.
            cov_detail = "; ".join(
                f"{p}: {_fmt_uncovered_fraction(cov.uncovered_fraction)} of {cov.span_s:.0f}s "
                f"capture span un-observed (accounted_s={cov.accounted_s:.0f}, "
                f"n_data_events={cov.n_data_events})"
                for p, cov in sorted(self.coverage_by_pair.items())) or "(no coverage record)"
            raise CaptureFormatError(
                f"{self.path}: GATE (iv) REFUSAL — {len(self.unrecorded_data_gaps)} "
                f"DATA-BEARING discontinuit(y/ies) with no recorded gap covering them: "
                f"{spans}{more}. Heartbeats are excluded from this check on purpose: a heartbeat "
                f"proves the socket is alive, not that the market was observed. A session whose "
                f"heartbeats keep flowing normally — so the recorder's own heartbeat_timeout_s "
                f"never expires and no gap is ever written — while the book/trade subscription "
                f"silently throttles passes the all-event check above and is caught here. "
                f"Threshold: {self.gap_detect_threshold_s}s; declared recorder "
                f"heartbeat_timeout_s: {self.declared_heartbeat_timeout_s}. Aggregate data-bearing "
                f"coverage over the capture span, for reference — the TRUE un-observed fraction, "
                f"not the 0.000% a data-span-denominated statistic would report: {cov_detail}.")
        if self.events_inside_recorded_gaps:
            shown = "; ".join(self.events_inside_recorded_gaps[:8])
            more = ("" if self.n_events_inside_recorded_gaps <= 8
                    else f" (+{self.n_events_inside_recorded_gaps - 8} more)")
            raise CaptureFormatError(
                f"{self.path}: GATE (iv) REFUSAL — SELF-CONTRADICTORY GAP LEDGER: "
                f"{self.n_events_inside_recorded_gaps} data-bearing event(s) carry a timestamp "
                f"lying STRICTLY INSIDE a recorded gap span: {shown}{more}. A recorded gap is a "
                f"claim that the wire was NOT observed across that span; a book/trade event "
                f"inside it is a claim that it WAS. Both cannot be true, and the direction this "
                f"resolves in silently is the dangerous one: recorded-gap time is removed from "
                f"the coverage denominator, so a gap that swallows its own events drives "
                f"`accounted_s` to 0 — which used to read as `uncovered_fraction = 0.000%`, an "
                f"empty `degraded_coverage_pairs()`, and a PASSING completeness check over a "
                f"capture whose own table said `{self.n_book + self.n_trade} data events | "
                f"whole span recorded as gap | 0 accounted`. `n_gaps_outside_event_span` only "
                f"checks the opposite direction and cannot see this. Containment is tested "
                f"against INDIVIDUAL gap records (never the merged spans: two gaps that merely "
                f"touch at one event's timestamp union into a span whose seam is not a gap "
                f"anybody recorded) with the high edge pulled in by "
                f"{GAP_EDGE_CLOCK_SAMPLE_TOLERANCE_S * 1e6:.0f}us for the recorder's own "
                f"two-clock-sample slack, so what remains is a real contradiction. Diagnoses, "
                f"most likely first: (1) a recorder writing a gap's `end_ts` from a clock sampled "
                f"LATER than the one its own events are stamped with — this recorder's "
                f"`clock_jump` path did exactly that before run-49 and put its terminating event "
                f"microseconds inside its own gap (bounded, and now excluded by the tolerance "
                f"above; anything deeper than that bound is NOT this); (2) a hand-built or "
                f"externally-merged gap ledger; (3) a BACKWARD step of the local clock (this "
                f"format's `clock_basis` is {CLOCK_BASIS!r}, and only FORWARD jumps are detected, "
                f"as `clock_jump` gaps). Remedy: this is a defect in the gap LEDGER, not in the "
                f"data — re-derive the ledger (or re-cut the capture to the span the gaps "
                f"actually describe). Do NOT delete the events to make the ledger true, and do "
                f"NOT delete the gap to make the events true; either one picks a side of a "
                f"contradiction the recorder never resolved. Do NOT widen "
                f"`GAP_EDGE_CLOCK_SAMPLE_TOLERANCE_S` to make a file pass — it is bounded by a "
                f"mechanism, not by a file.")
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
            # in-gate defect #6 (run-45): the remedy text used to be one paragraph for every
            # failure kind, and it was unachievable for a "zero events for the WHOLE capture"
            # kind — re-cutting to a NARROWER window can only REDUCE a print count, so it cannot
            # manufacture events that were never received, and "re-run against a subscription
            # that delivers" IS dropping the pair from the declaration, which the same paragraph
            # forbade in its next breath. Split per failure kind instead.
            dead_kinds = ("declared_without_data", "declared_without_coverage",
                          "declared_channel_without_data")
            mirror_kinds = ("observed_undeclared", "coverage_undeclared",
                            "observed_channel_undeclared")
            remedies = []
            if any(recon.get(k) for k in dead_kinds):
                remedies.append(
                    "for a DEAD subscription (zero events for a declared pair/channel for the "
                    "WHOLE capture): a narrower re-cut cannot manufacture events that were never "
                    "received, so it is not an available remedy here — re-run the recorder "
                    "against a subscription verified to deliver this pair/channel and record a "
                    "FRESH capture; do not hand-edit THIS capture's declaration to match what it "
                    "already contains, and do not widen the bound to admit it")
            if any(recon.get(k) for k in mirror_kinds):
                remedies.append(
                    "for data or a coverage record attributed to an UNDECLARED pair/channel: "
                    "the subscription config used for the NEXT recording must be corrected to "
                    "include what was actually received, and that new capture's own declaration "
                    "will then be the one reconciled against it; this capture's declaration must "
                    "not be edited after the fact to match what it happened to record")
            remedy = " Remedy split by failure kind — " + "; ".join(remedies) + "."
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
                f"this instrument measured. NOT 'drop the pair from the declaration' and NOT "
                f"'widen the bound' — either one launders the absence into a completeness claim, "
                f"which is the whole thing gate (iv) exists to prevent.{remedy}")
        degraded = self.degraded_coverage_pairs()
        if degraded:
            detail = "; ".join(
                f"{p}: "
                f"{_fmt_uncovered_fraction(self.coverage_by_pair[p].uncovered_fraction)}"
                f" of {self.coverage_by_pair[p].accounted_s:.0f}s accounted time un-observed "
                f"({self.coverage_by_pair[p].n_data_events} data events"
                + (f"; of which {self.coverage_by_pair[p].stale_book_s:.0f}s was STALE "
                   f"keepalive-driven book — a top of book whose (bid, ask) never moved across "
                   f"runs longer than {STALE_BOOK_MIN_RUN_S:g}s, so the mid these estimands are "
                   f"built from carried no new information there (this reader cannot tell a "
                   f"keepalive-timer repeat from a size-only wire update; see "
                   f"`STALE_BOOK_MIN_RUN_S`), and is already "
                   f"counted as UN-OBSERVED above. This pair reached the bound because its UNION "
                   f"of book AND trade observations did; stale book time alone reaches it only "
                   f"when the pair's trades have stopped too. A per-pair EXCLUSION, and any check "
                   f"of a stale book MASKED BY LIVE TRADE FLOW, are ADDENDUM B2 scope and NOT "
                   f"implemented here"
                   if self.coverage_by_pair[p].stale_book_s > 0.0 else "")
                + ")"
                for p in degraded)
            # Item 2 (cycle 3): the remedy is split by FAILURE KIND, as in-gate defect #6 required
            # for the thin-cell case, and every branch names an action achievable on data ALREADY
            # RECORDED. "Re-cut the capture around the degraded stretch" is vacuous when the whole
            # capture IS the stretch — for a pair with no observed run long enough to carry even
            # one bar of the minute grid these estimands are denominated in, there is no narrower
            # window to cut to and saying so is the honest remedy.
            recuttable = [p for p in degraded
                          if self.coverage_by_pair[p].longest_observed_run_s >= BAR_INTERVAL_S]
            unrecuttable = [p for p in degraded if p not in recuttable]
            remedies = []
            if recuttable:
                runs = ", ".join(
                    f"{p} (longest contiguous observed run "
                    f"{self.coverage_by_pair[p].longest_observed_run_s:.0f}s)"
                    for p in recuttable)
                remedies.append(
                    f"for pair(s) that DO still have a healthy stretch inside this file — {runs} "
                    f"— re-cut THIS capture to a window contained in that stretch and re-run the "
                    f"reader over the cut; that is achievable on data already recorded, and the "
                    f"cut window must still be longer than the longest pre-registered horizon or "
                    f"the horizons it cannot form become zero-sample rows")
            if unrecuttable:
                remedies.append(
                    f"for pair(s) with NO observed run as long as one {BAR_INTERVAL_S:g}s bar — "
                    f"{unrecuttable} — re-cutting is VACUOUS: there is no narrower window of this "
                    f"file in which the pair is covered, so no reader-side action on this capture "
                    f"can produce a number for it. The achievable actions are to record a FRESH "
                    f"capture against a subscription verified to deliver this pair, and, if the "
                    f"other pairs in this file are wanted meanwhile, to re-cut the file to exclude "
                    f"this pair's records entirely so its absence is a stated non-subscription "
                    f"rather than a silent hole. Neither one is 'widen the bound'")
            remedy = " Remedy split by failure kind — " + "; ".join(remedies) + "."
            worst = max((self.coverage_by_pair[p] for p in degraded),
                         key=lambda c: c.uncovered_fraction or 0.0)
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
                f"the market. KEEPALIVE-STALE book time is COUNTED AS UN-OBSERVED by this check, "
                f"deliberately — but note what that does and does NOT reach: this check is "
                f"denominated over the UNION of each pair's book AND trade observations, so stale "
                f"book time can breach the bound only where the pair's trades have stopped as "
                f"well. A book frozen or silent while its prints keep flowing does NOT reach this "
                f"bound (measured: 7195s of a 7200s capture stale, trades every 30s, "
                f"uncovered_fraction 0.13% against a 2.00% bound) and is NOT caught anywhere in "
                f"ADDENDUM B1 — that needs the book-mid-only freshness term, which is ADDENDUM B2 "
                f"scope, as is any per-pair exclusion. Do NOT widen the bound.{remedy}")


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
    # TASK_914: stream the file in ONE forward pass rather than materializing every line (as a
    # raw string, then again as a parsed dict) before this loop starts — that was the largest
    # single term in the reader's RSS. `_iter_lines_tolerant` now does its own one-line lookahead
    # (see its docstring) and reports whether the file's own last physical line was a tolerated
    # truncation via `tail_status`, folding in what used to be `_file_has_truncated_tail`'s
    # separate second `readlines()` pass. `got_any_line` mirrors the old `lines` (the fully
    # materialized parsed list) being non-empty — the truncated-tail flag only takes effect if at
    # least one line was actually parsed, exactly as before (an all-corrupt-tolerated-tail file
    # produced no `lines` and so never called `_file_has_truncated_tail` at all).
    tail_status: List[bool] = [False]
    got_any_line = False

    pairs_seen: set = set()
    # B1 (run-45): the declared side of gate (iv) is the UNION of every writer session's
    # declaration (`meta_start` plus every `meta_resume`), not session 1's alone — see the
    # `SessionDeclaration`/`sessions` docstrings. `*_field_seen` distinguishes "no session ever
    # declared this field" (stays `None`, the fail-closed sentinel) from "a session declared it
    # as empty" (a real empty set/list, itself still a completeness failure elsewhere).
    declared_pairs_field_seen = False
    declared_pairs_accum: set = set()
    channels_field_seen = False
    channels_accum: set = set()

    def _validate_declared_pairs(declared: Any, record_type: str) -> List[str]:
        # Shape-check before use. `pairs` is the DECLARED side of defense 4, and a bare
        # string here would be iterated character-by-character into a declared set of
        # single letters that no observed pair can ever match — turning a malformed
        # declaration into a reconciliation failure with an unreadable cause. Refuse at
        # the source instead.
        if (not isinstance(declared, list)
                or not all(isinstance(p, str) and p for p in declared)):
            raise CaptureFormatError(
                f"{path}: {record_type} `pairs` must be a list of non-empty strings, got "
                f"{declared!r} — refusing. This field is the declared side of gate (iv)'s "
                f"pair reconciliation; a declaration that cannot be read cannot be "
                f"reconciled, and guessing at its meaning is how a completeness claim "
                f"gets made about a set nobody stated.")
        return list(declared)

    def _validate_declared_channels(channels: Any, record_type: str) -> List[str]:
        unknown = [c for c in channels if c not in DATA_CHANNELS]
        if unknown:
            raise CaptureFormatError(
                f"{path}: {record_type} declares unknown data channel(s) {unknown} "
                f"(known: {list(DATA_CHANNELS)})")
        return list(channels)

    #: Cycle 2 (run-45), the keepalive reconciliation — STREAMING so a multi-day capture never
    #: holds one tuple per book event in memory: per pair, the current run's top-of-book value and
    #: its first/last timestamps. A run ends the moment the value changes; runs longer than
    #: `STALE_BOOK_MIN_RUN_S` are recorded in `stale_runs` and their repeats are excluded from
    #: coverage. See `STALE_BOOK_MIN_RUN_S` for why this is value-based and reader-side.
    book_run: Dict[str, Dict[str, Any]] = {}
    stale_runs: Dict[str, List[Tuple[float, float]]] = {}

    def _close_book_run(pair: str) -> None:
        run = book_run.get(pair)
        if run is None:
            return
        duration = run["last"] - run["start"]
        if run["n"] > 1 and duration > STALE_BOOK_MIN_RUN_S:
            stale_runs.setdefault(pair, []).append((run["start"], run["last"]))

    def _note_book_top(pair: str, ts: float, top: Tuple[Any, Any]) -> None:
        # `top` is the PRICE pair `(bid, ask)` only — see `STALE_BOOK_MIN_RUN_S`. Adding sizes
        # here makes the run key finer than the mid the estimands consume, which is how a frozen
        # mid with jittering sizes read as a fully-observed book (run-49 BLOCKER).
        run = book_run.get(pair)
        if run is not None and run["top"] == top:
            run["last"] = max(run["last"], ts)
            run["n"] += 1
            return
        _close_book_run(pair)
        book_run[pair] = {"top": top, "start": ts, "last": ts, "n": 1}

    def _note_data_event(channel: str, pair: Any, ts: float, where: str) -> None:
        if not isinstance(pair, str) or not pair:
            raise CaptureFormatError(
                f"{where}: {channel} record has no usable `pair` ({pair!r}) — refusing. A "
                f"data-bearing event that belongs to no pair contributes to the channel counts "
                f"and to the event span while appearing in NO pair's coverage accounting, which "
                f"is exactly the kind of unattributable event gate (iv) cannot reason about.")
        data_ts.setdefault(pair, {}).setdefault(channel, []).append(ts)
        pairs_seen.add(pair)
    for obj in _iter_lines_tolerant(path, truncated_tail_out=tail_status):
        got_any_line = True
        t = obj.get("type")
        if t == "meta_start":
            meta.had_meta_start = True
            declared = obj.get("pairs")
            session_pairs: List[str] = []
            if declared is not None:
                session_pairs = _validate_declared_pairs(declared, "meta_start")
                declared_pairs_field_seen = True
                declared_pairs_accum |= set(session_pairs)
                pairs_seen.update(session_pairs)
            session_channels: Optional[List[str]] = None
            if obj.get("channels") is not None:
                session_channels = _validate_declared_channels(obj["channels"], "meta_start")
                channels_field_seen = True
                channels_accum |= set(session_channels)
            # `meta.channels` is set from session 1 alone for the reader's OWN per-capture
            # semantics that predate B1 (`missing_channels()`, the per-cell reconciliation) —
            # B1's union lives in `declared_pairs`/the post-loop `channels` union below; this
            # assignment is superseded there but kept here so a single-session capture (still
            # the common shape) never depends on the post-loop step existing.
            meta.channels = session_channels
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
            meta.sessions.append(SessionDeclaration(
                kind="start", pairs=sorted(session_pairs), channels=session_channels,
                heartbeat_timeout_s=meta.declared_heartbeat_timeout_s))
        elif t == "meta_resume":
            # B1 (run-45): a LATER writer session's own declared config — see
            # `SessionDeclaration`. Unioned into `declared_pairs`/`channels` below (post-loop),
            # never allowed to REPLACE session 1's declaration (that would just move which
            # single session's config gets presented as the whole capture's).
            if not meta.had_meta_start:
                raise CaptureFormatError(
                    f"{path}: meta_resume record with no preceding meta_start — a resumption "
                    f"declaration is meaningless without a first session to resume from.")
            declared = obj.get("pairs")
            session_pairs = []
            if declared is not None:
                session_pairs = _validate_declared_pairs(declared, "meta_resume")
                declared_pairs_field_seen = True
                declared_pairs_accum |= set(session_pairs)
                pairs_seen.update(session_pairs)
            session_channels = None
            if obj.get("channels") is not None:
                session_channels = _validate_declared_channels(obj["channels"], "meta_resume")
                channels_field_seen = True
                channels_accum |= set(session_channels)
            resume_hb = obj.get("heartbeat_timeout_s")
            if resume_hb is not None:
                resume_hb = _require_finite(resume_hb, "heartbeat_timeout_s", f"{path}: meta_resume")
            meta.sessions.append(SessionDeclaration(
                kind="resume", pairs=sorted(session_pairs), channels=session_channels,
                heartbeat_timeout_s=resume_hb))
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
            _note_book_top(obj["pair"], ts, (obj.get("bid"), obj.get("ask")))
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

    meta.truncated_tail = got_any_line and tail_status[0]

    # B1 (run-45): `declared_pairs`/`channels` are the UNION across every writer session that
    # declared one — `meta_start` plus every `meta_resume` — never session 1's alone. A
    # single-session capture (still the common, real shape) reduces to exactly the old
    # behaviour: one session in the union is the same as no union at all.
    meta.declared_pairs = sorted(declared_pairs_accum) if declared_pairs_field_seen else None
    if channels_field_seen:
        meta.channels = sorted(channels_accum)

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
    # B2 item 1 (run-45): anchor the per-pair scan to the CAPTURE span (all events, heartbeats
    # included) — heartbeats prove the socket was up, which is exactly what makes their span the
    # right bound for "did we observe the market?". Without this, a pair whose data-bearing
    # events stop long before the capture's own end (or start long after it) has no interior jump
    # for the loop to find — the edge hole was invisible.
    capture_span = ((meta.start_ts, meta.end_ts)
                     if meta.start_ts is not None and meta.end_ts is not None else None)
    all_data_ts: List[float] = []
    unrecorded_data: List[GapRecord] = []
    for pair in sorted(data_ts):
        pair_ts = sorted(t for chan_ts in data_ts[pair].values() for t in chan_ts)
        all_data_ts.extend(pair_ts)
        for g in _find_unrecorded_gaps(pair_ts, meta.gaps, gap_detect_threshold_s,
                                        span=capture_span):
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
    # B2 item 2 (run-45): denominate over the CAPTURE span (`capture_span`, all events including
    # heartbeats), NOT the data span (`meta.data_start_ts`/`meta.data_end_ts`) — the data span is
    # exactly the quantity that collapses together with the numerator when book/trade goes dark,
    # which is how `accounted_s=130, span_s=130` (both denominator AND numerator collapsed to the
    # 130.5s of data) read as `uncovered_fraction=0.000%` against a true 98.188%. The data span is
    # still computed above (`meta.data_start_ts`/`data_end_ts`) and reconciled against the capture
    # span for disclosure (see `render._render_coverage_block`).
    # --- cycle 2: the KEEPALIVE reconciliation (spec lines 816-821) ---------------------------
    # D67 in one line: reconcile the DECLARED emission policy against the OBSERVED cadence. The
    # precondition is a declaration to reconcile against — a capture that declares no
    # `book_emission_policy`/`book_keepalive_s` (every synthetic/replay/v1 fixture, and every
    # pre-Addendum-A capture) makes no keepalive claim, so there is nothing to contradict and
    # nothing fires. Both directions of the reconciliation are computed here:
    #   * STALE   — the recorder emitted on its timer while the top of book never moved. GATED,
    #               via `covered_s` below: those repeats are not observations of the market.
    #   * OVERRUN — the recorder emitted SLOWER than the floor it declared. DISCLOSED only.
    for pair in list(book_run):
        _close_book_run(pair)
    declares_keepalive = (meta.declared_book_emission_policy is not None
                          and meta.declared_book_keepalive_s is not None)
    if not declares_keepalive:
        stale_runs = {}
    else:
        floor = meta.declared_book_keepalive_s or 0.0
        overrun_threshold = floor * BOOK_KEEPALIVE_OVERRUN_FACTOR
        for pair in sorted(data_ts):
            book_list = sorted(data_ts[pair].get("book", ()))
            if not book_list:
                continue
            # Run-49 MAJOR — the SAME edge hole B2 item 1 already fixed once, in this function's
            # sibling. `range(1, len(book_list))` looks strictly BETWEEN consecutive book events,
            # so a book that emits every 5s for the first half of a capture and then goes silent
            # for the second half produced `book_keepalive_overruns: {}` — an 1800s silence
            # against a declared 5.0s floor, 360x the violation, invisible, because no
            # consecutive pair of book events spans it. Leading silence (capture start to the
            # first book event) was equally invisible. Anchor the scan to the CAPTURE span
            # exactly as `_find_unrecorded_gaps(..., span=capture_span)` is anchored: the
            # leading and trailing intervals are candidate overruns like any other. Recorded-gap
            # time is still subtracted below, so a disclosed outage at either edge does not
            # count. A pair with ZERO book events is skipped rather than reported as one
            # span-long overrun: it has no first book event to anchor to, and a declared `book`
            # cell that carried nothing is already a REFUSAL one level up
            # (`declared_channel_without_data`), not a cadence disclosure.
            scan = list(book_list)
            if capture_span is not None:
                span_lo, span_hi = capture_span
                if span_lo < scan[0]:
                    scan = [span_lo] + scan
                if span_hi > scan[-1]:
                    scan = scan + [span_hi]
            n_over, worst = 0, 0.0
            for i in range(1, len(scan)):
                prev_t, cur_t = scan[i - 1], scan[i]
                interval = cur_t - prev_t
                if interval <= overrun_threshold:
                    continue
                gap_covered = sum(max(0.0, min(e, cur_t) - max(s, prev_t)) for s, e in merged)
                if (interval - gap_covered) <= overrun_threshold:
                    continue  # a DISCLOSED outage, not the recorder ignoring its own floor
                n_over += 1
                worst = max(worst, interval)
            if n_over:
                meta.book_keepalive_overruns[pair] = n_over
                meta.book_keepalive_max_interval_s[pair] = worst

    # --- cycle 2: the gap ledger must not contradict the events it is filed against ------------
    # A recorded gap claims the wire was NOT observed across its span; a data-bearing event
    # strictly inside it claims it WAS. `n_gaps_outside_event_span` only ever checked the opposite
    # direction, so a gap that SWALLOWS its own events drove `accounted_s` to 0 and read as
    # `uncovered_fraction = 0.000%` with a PASSING completeness check.
    #
    # Run-49 (BLOCKER): containment is tested against INDIVIDUAL gap records, NOT against
    # `merged`. `_merge_gap_spans` unions two gaps that merely TOUCH — a `_close_dark_window`
    # `disconnect` gap ending AT the first event after reconnecting, and the `clock_jump` gap
    # starting AT that same event — and the seam between them then reads as containment. Both
    # contacts are LEGAL (each gap ends/starts at an event's own timestamp); their union is not a
    # gap anybody recorded. Measured on the live capture: 14 of 598 flagged events sat in NO
    # individual gap at all, 2.537s-4.396s deep inside a merged span. The merged span stays right
    # for coverage ARITHMETIC (measure of the union, never the sum) — it is only this containment
    # TEST that is wrong to run against it.
    if meta.gaps:
        inside: List[str] = []
        n_inside = 0
        # One half-open contradiction interval per INDIVIDUAL gap, its HIGH edge pulled in by
        # `GAP_EDGE_CLOCK_SAMPLE_TOLERANCE_S` (see that constant for the mechanism and the
        # measured bound). Coalesced only on STRICT overlap — never on mere contact, which is the
        # whole point above — so the result is sorted and disjoint and a binary search over its
        # start points finds the only interval that can contain `t`: O(log m) per event rather
        # than O(m), which matters at multi-day capture sizes (10.2M events x 1,688 gaps).
        contradiction: List[Tuple[float, float]] = []
        for lo, hi in sorted((g.start_ts, g.end_ts - GAP_EDGE_CLOCK_SAMPLE_TOLERANCE_S)
                              for g in meta.gaps):
            if hi <= lo:
                continue  # a gap no longer than the tolerance has no interior left to contradict
            if contradiction and lo < contradiction[-1][1]:
                contradiction[-1] = (contradiction[-1][0], max(contradiction[-1][1], hi))
            else:
                contradiction.append((lo, hi))
        gap_starts = [lo for lo, _ in contradiction]
        for pair in sorted(data_ts):
            for channel in sorted(data_ts[pair]):
                for t in data_ts[pair][channel]:
                    i = bisect.bisect_right(gap_starts, t) - 1
                    if i < 0:
                        continue
                    g_lo, g_hi = contradiction[i]
                    if g_lo < t < g_hi:
                        n_inside += 1
                        if len(inside) < 8:
                            # Name the INDIVIDUAL record, not the coalesced interval — the
                            # coalesced one is an artifact of this scan and would send a reader
                            # looking for a gap the ledger does not contain. Linear over
                            # `meta.gaps`, but bounded to the 8 samples actually printed.
                            src = next((g for g in meta.gaps
                                         if g.start_ts < t < g.end_ts), None)
                            where = (f"[{src.start_ts:.6f}, {src.end_ts:.6f}] "
                                      f"({src.reason})" if src is not None
                                      else f"[{g_lo:.6f}, {g_hi:.6f}]")
                            inside.append(f"{pair}:{channel} at {t:.6f} inside recorded gap "
                                           f"{where}")
        meta.events_inside_recorded_gaps = inside
        meta.n_events_inside_recorded_gaps = n_inside

    # --- defense 3: the AGGREGATE, cumulative, data-bearing coverage check (continued) ---------
    if capture_span is not None and data_ts:
        span_lo, span_hi = capture_span
        meta.coverage_by_pair = {
            pair: _pair_coverage(pair, data_ts[pair], merged, span_lo, span_hi, coverage_window_s,
                                  stale_spans=_merge_spans(stale_runs.get(pair, [])))
            for pair in sorted(data_ts)
        }
    meta.coverage_evaluated = True
    return meta


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


def _subtract_spans(a: List[Tuple[float, float]],
                     b: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """union(a) MINUS union(b), as a sorted disjoint span list. Both inputs must already be
    merged. Used to cut recorded-gap time out of the observed runs before measuring the LONGEST
    one — a run that is only contiguous because a disclosed outage sits inside it is not a
    contiguous stretch a re-cut could keep."""
    out: List[Tuple[float, float]] = []
    j = 0
    for lo, hi in a:
        cur = lo
        k = j
        while k < len(b) and b[k][0] < hi:
            g_lo, g_hi = b[k]
            if g_hi > cur:
                if g_lo > cur:
                    out.append((cur, min(g_lo, hi)))
                cur = max(cur, g_hi)
                if cur >= hi:
                    break
            k += 1
        if cur < hi:
            out.append((cur, hi))
        while j < len(b) and b[j][1] <= hi:
            j += 1
    return [(lo, hi) for lo, hi in out if hi > lo]


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
                    span_lo: float, span_hi: float, window_s: float,
                    stale_spans: Optional[List[Tuple[float, float]]] = None) -> PairCoverage:
    """Aggregate data-bearing coverage for one pair over [span_lo, span_hi].

    An instant counts as OBSERVED iff a book/trade event for this pair lies within +/- window_s
    of it, so `covered` is the measure of the union of those event windows, clipped to the span
    and with the recorded-gap time removed (that time is already out of the denominator).

    `stale_spans` (cycle 2, run-45) are this pair's KEEPALIVE-STALE book runs: stretches over
    which the recorder kept emitting book events on its declared `book_keepalive_s` timer while
    the top of book never moved. A book event strictly inside such a run is the recorder's timer
    firing, not an observation of the market, so it is dropped from the coverage event list — but
    the run's FIRST event is kept (that one really did carry the value), and TRADE events inside
    the run are kept untouched (a print is an independent observation of the market and does not
    depend on the book channel being fresh). What survives is the honest picture: a frozen book
    covers only the window around the moment it froze.
    """
    stale = _merge_spans(list(stale_spans or []))
    stale_starts = [lo for lo, _ in stale]

    def _is_stale_repeat(t: float) -> bool:
        i = bisect.bisect_right(stale_starts, t) - 1
        if i < 0:
            return False
        lo, hi = stale[i]
        return lo < t <= hi  # the run's FIRST event (t == lo) is a real observation

    book_ts = [t for t in by_channel.get("book", ()) if not _is_stale_repeat(t)]
    other_ts = [t for ch, ts in by_channel.items() if ch != "book" for t in ts]
    ts_list = sorted(t for ts in by_channel.values() for t in ts)
    observing_ts = sorted(book_ts + other_ts)
    span = [(span_lo, span_hi)]
    gaps_clipped = _merge_spans(
        [(max(s, span_lo), min(e, span_hi)) for s, e in merged_gaps
         if min(e, span_hi) > max(s, span_lo)])
    span_s = span_hi - span_lo
    recorded_gap_s = _measure_intersection(gaps_clipped, span)
    accounted_s = max(0.0, span_s - recorded_gap_s)

    def _covered(event_ts: List[float]) -> Tuple[float, List[Tuple[float, float]]]:
        ev = _merge_spans([(t - window_s, t + window_s) for t in event_ts])
        covered_in_span = _measure_intersection(ev, span)
        covered_in_gaps = _measure_intersection(ev, gaps_clipped)
        return max(0.0, covered_in_span - covered_in_gaps), ev

    covered_s, covered_spans = _covered(observing_ts)
    stale_clipped = _merge_spans(
        [(max(s, span_lo), min(e, span_hi)) for s, e in stale
         if min(e, span_hi) > max(s, span_lo)])
    stale_book_s = _measure_intersection(stale_clipped, span)
    # Longest contiguous OBSERVED run inside accounted time — the remedy-text discriminator.
    observed_runs = [(max(lo, span_lo), min(hi, span_hi)) for lo, hi in covered_spans
                     if min(hi, span_hi) > max(lo, span_lo)]
    observed_minus_gaps = _subtract_spans(observed_runs, gaps_clipped)
    longest_observed_run_s = max((hi - lo for lo, hi in observed_minus_gaps), default=0.0)
    return PairCoverage(pair=pair, n_data_events=len(ts_list), window_s=window_s,
                         span_s=span_s, recorded_gap_s=recorded_gap_s,
                         accounted_s=accounted_s, covered_s=min(covered_s, accounted_s),
                         stale_book_s=stale_book_s,
                         longest_observed_run_s=longest_observed_run_s)


def _find_unrecorded_gaps(sorted_ts: List[float], recorded_gaps: List[GapRecord],
                           threshold_s: float,
                           span: Optional[Tuple[float, float]] = None) -> List[GapRecord]:
    """Scan consecutive raw event timestamps for a jump > threshold whose UNCOVERED portion
    (jump duration minus the UNION of the recorded gaps overlapping it) still exceeds the
    threshold. This is the independent check: it does NOT trust the recorder's own gap list, it
    re-derives whether one was needed — but it tolerates a small residual around a recorded
    gap's edges (e.g. the recorder writes its resume gap at restart time, and the first fresh
    event a fraction of a second later is not itself a second, separate outage).

    Coverage is computed over MERGED (disjoint) spans, so overlapping or duplicated gap records
    cannot double-count their way over a real hole.

    `span`, when given, is a known BOUND `(span_lo, span_hi)` for `sorted_ts` (B2 item 1,
    run-45): the loop below only ever looks strictly BETWEEN consecutive elements of `sorted_ts`,
    so a hole at an EDGE of the timeline — before the first element or after the last — has no
    consecutive pair spanning it and is invisible without this. An interior hole was already
    caught (cycles 2/3); leading, trailing and both-edge holes were not. Passing the capture's
    all-event span (heartbeats included) as `span` for a per-pair DATA-BEARING scan is exactly
    how the run-45 BLOCKER (a pair whose data stops long before the capture ends, heartbeats
    still flowing) is now caught: the trailing interval becomes a candidate jump like any other.
    """
    merged, _, _ = _merge_gap_spans(recorded_gaps)
    ts = list(sorted_ts)
    if span is not None:
        span_lo, span_hi = span
        if not ts:
            ts = [span_lo, span_hi]
        else:
            if span_lo < ts[0]:
                ts = [span_lo] + ts
            if span_hi > ts[-1]:
                ts = ts + [span_hi]
    unrecorded = []
    for i in range(1, len(ts)):
        prev_t, cur_t = ts[i - 1], ts[i]
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
