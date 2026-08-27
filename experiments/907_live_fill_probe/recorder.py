"""TASK_907 phase 1 — the Kraken PUBLIC websocket recorder (book top + trades).

No credentials anywhere in this file. The public feed needs none, and this module never
imports, reads, or references the Kraken account credential env vars or the authenticated
REST client's signed-request machinery — see `test_credential_free.py`, which greps this file
(by name, deliberately not spelled out in this docstring) for zero occurrences.

Designed to run long-lived in `tmux` under `tools/safe_run.sh` on gpu-server (the orchestrator
launches it; this task does not). Fail-loud on gaps (the 893 lesson): every disconnect,
heartbeat timeout, sustained schema drift, or reader-detected discontinuity is written to the
capture as a `GapRecord` BEFORE new data resumes — never silently interpolated.

The reconnect loop is dependency-injectable (`connect_fn`) so its gap-recording and backoff
behavior is unit-testable without a live socket — see `test_recorder_reconnect.py`.

CLOCK BASIS (see `capture.py`'s module docstring for the full rationale)
-----------------------------------------------------------------------
Every `ts` this recorder writes — including on trades — is LOCAL RECEIPT TIME from
`self._clock()`. The venue's own trade timestamp is written alongside as `venue_ts` and is used
by nothing. Gap accounting therefore compares like with like. Before this, trades carried
venue time while book/heartbeat/gap records carried local time, so a burst of trades with
skewed or stale venue timestamps could make successive `_record_gap()` calls write overlapping
spans denominated on two different clocks.

SCHEMA DRIFT IS LOUD (the highest-risk failure mode for the actual launch)
-------------------------------------------------------------------------
Kraken v2's live schema is UNVERIFIED against the real feed by this row, so a field rename or
shape change is the EXPECTED failure, not a hypothetical one. A valid-JSON frame with an
unexpected shape used to yield zero events and be indistinguishable from innocuous venue
silence. Now:

  * schema mismatches are counted SEPARATELY from JSON-decode failures and from wholly
    unrecognized frames, and all three counters are PERSISTED into the capture (`counters`
    records) so the artifact can surface them;
  * DETECTION RULE for "connected but recording nothing": if
    `SCHEMA_DRIFT_FRAME_THRESHOLD` (200) frames arrive back-to-back without a single one
    yielding an event, AND at least one of them was a schema mismatch or an unrecognized
    frame, the recorder writes a `schema_drift` GAP, logs at ERROR, and forces a reconnect.
    A healthy connection cannot trip this: Kraken's public feed heartbeats at least every ~1s
    and every heartbeat frame IS an event, so the counter resets continuously. A genuinely
    quiet market still heartbeats; only traffic we cannot parse looks like this. Because the
    condition writes a real gap record, gate (iv) then refuses to let any number cross that
    span — "connected but recording nothing" can no longer look like a quiet market.
  * a DECLARED channel that records zero events over the whole capture is a gate-(iv) refusal
    (`capture.CaptureMetadata.assert_complete`) — the old `n_book + n_trade > 0` sum passed
    happily while one channel was silently dead.

THE BOOK CHANNEL IS MAINTAINED, NOT SAMPLED (ADDENDUM A, run 45, on measured wire evidence)
---------------------------------------------------------------------------------------------
A 45s / 5,737-frame live capture (`wire_fixture_kraken_v2_2026-08-27.jsonl` preserves the first
600) proved two defects in the original `bids[0]`/`asks[0]` sampling:

  1. Kraken v2's `book` `update` rows are one-sided INCREMENTAL DELTAS at arbitrary price
     levels, not a top-of-book snapshot — 5,661 of 5,665 rows have an empty `bids` or `asks`
     list, so the old "reject if either side is empty" rule discarded 99.93% of the channel as
     `schema_mismatch`. Worse, the 4 rows it DID accept could carry a delta several levels deep
     as if it were the top (measured: a 2505.76 delta arrived while the true best bid was
     2505.81).
  2. Every book row carries a venue-computed CRC32 `checksum` over the reconstructed top-10
     book (asks ascending then bids descending, each level's price+qty with the decimal point
     and leading zeros stripped, concatenated) — a free declared-vs-observed check on every
     single frame, proven to reconstruct 5,665/5,665 against the fixture.

So the book channel is now a per-pair, per-connection MAINTAINED state machine: a `snapshot`
replaces the book; an `update` applies deltas (qty 0 deletes); the state is trimmed to
`book_depth` after every application; the checksum is verified on every row; a mismatch drops
that book, blocks further emission for the pair until a fresh snapshot, and forces a full
reconnect (book state is per-connection, so this is the only fail-closed action available). The
wire arrives at ~129 frames/s for 2 pairs; what is EMITTED is bounded by `book_keepalive_s`
(change-or-keepalive), declared into `meta_start` alongside the policy name so the artifact
states the cadence floor its numbers were measured under.

Checksum wire fidelity requires decoding frames with `json.loads(text, parse_float=str,
parse_int=str)` — the checksum is defined over the wire's STRING forms, and `2.0 != "2.00000000"`
once decimals are lost to a Python float round-trip.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import sys
import time
import zlib
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture import (  # noqa: E402
    DEFAULT_GAP_DETECT_THRESHOLD_S, CaptureWriter, GapRecord,
)

logger = logging.getLogger("907_live_fill_probe.recorder")

WS_URL = "wss://ws.kraken.com/v2"

#: The data channels this recorder subscribes to, declared into the capture's `meta_start` so
#: the reader can refuse a capture where one of them recorded nothing.
SUBSCRIBED_CHANNELS = ("book", "trade")

#: Frames that Kraken's v2 public feed legitimately sends which carry no market data. These are
#: recognized-but-eventless and are NOT schema mismatches.
NON_DATA_CHANNELS = frozenset({"status", "heartbeat"})

#: See the module docstring: the "connected but recording nothing" detection threshold.
SCHEMA_DRIFT_FRAME_THRESHOLD = 200

#: How often (in frames) to re-persist the health counters when they have changed.
COUNTERS_FLUSH_EVERY_N_FRAMES = 1000

#: The venue's book checksum is DEFINED over exactly the top 10 levels per side, independent of
#: the subscribed `book_depth` (A.3/A.4(3)) — a wider local `book_depth` still checksums top 10.
BOOK_CHECKSUM_DEPTH = 10

#: The name of the emission policy declared into `meta_start` (A.4(4)) — change-or-keepalive,
#: never interpolated. Kept as a string constant (not just the number) so the artifact states
#: WHAT the cadence floor means, not only its value.
BOOK_EMISSION_POLICY = "change_or_keepalive"

#: The single source of the change-or-keepalive rate-limit floor (A.4(4)). `RecorderConfig` and
#: `_parse_frame`'s convenience default both read THIS — they used to carry two independent
#: literal `5.0`s with nothing coupling them, so a change to one would have silently left tests
#: measuring a different cadence floor than the recorder actually runs.
DEFAULT_BOOK_KEEPALIVE_S = 5.0


class _GapAlreadyRecorded(ConnectionError):
    """Raised internally by `_consume` when it has ALREADY written the gap record for the
    condition forcing reconnect (heartbeat timeout, schema drift, book checksum mismatch) —
    signals `run()`'s except-block not to write a second, redundant gap for the same event.
    """


@dataclass
class RecorderConfig:
    pairs: List[str]
    out_path: str
    capture_id: str
    heartbeat_timeout_s: float = 30.0
    backoff_base_s: float = 2.0
    backoff_max_s: float = 60.0
    book_depth: int = 10
    #: Emit a book event for a pair when its top of book differs from the last EMITTED one, or
    #: when this many seconds have elapsed since that pair's last emission — see A.4(4). MUST
    #: stay well under the reader's `gap_detect_threshold_s` (default 60s), or a quiet-but-alive
    #: book reads as a dark socket; pinned by
    #: `test_book_normalization.py::test_book_keepalive_is_well_under_the_gap_detect_threshold`.
    book_keepalive_s: float = DEFAULT_BOOK_KEEPALIVE_S
    schema_drift_frame_threshold: int = SCHEMA_DRIFT_FRAME_THRESHOLD

    def __post_init__(self) -> None:
        # The checksum is defined over the top 10 levels (A.3); a shallower subscription could
        # never be checksum-verified, and the failure mode ("skip the checksum quietly") is
        # exactly what this whole addendum exists to rule out. Refuse loudly at construction
        # rather than silently disabling verification for a too-shallow config.
        if self.book_depth < BOOK_CHECKSUM_DEPTH:
            raise ValueError(
                f"book_depth={self.book_depth} is below {BOOK_CHECKSUM_DEPTH} — the venue's "
                f"checksum is defined over the top {BOOK_CHECKSUM_DEPTH} levels per side, so a "
                f"shallower subscription could never be checksum-verified. Refusing rather than "
                f"silently skipping verification.")
        # The keepalive was DECLARED into the artifact but never validated, so
        # `book_keepalive_s=90.0` constructed happily and declared a cadence floor coarser than
        # the reader's own 60s gap-detect resolution — the artifact would then state a floor
        # under which "no gap was recorded" means nothing. The reader refuses such a capture
        # (`capture.read_capture`); refuse it at construction too, so it is never written.
        if not (math.isfinite(self.book_keepalive_s) and self.book_keepalive_s >= 0.0):
            raise ValueError(
                f"book_keepalive_s={self.book_keepalive_s!r} must be a finite, non-negative "
                f"number of seconds (0 disables the rate limit — every checksum-clean row is "
                f"emitted).")
        if self.book_keepalive_s >= DEFAULT_GAP_DETECT_THRESHOLD_S:
            raise ValueError(
                f"book_keepalive_s={self.book_keepalive_s} is at or above the reader's "
                f"gap_detect_threshold_s ({DEFAULT_GAP_DETECT_THRESHOLD_S}s): the emission rate "
                f"limit would coarsen the record past the resolution gate (iv) reasons at, so a "
                f"quiet-but-alive book would be indistinguishable from a dark socket.")


@dataclass
class ParsedFrame:
    """The outcome of normalizing one wire frame.

    `schema_mismatches` counts rows/frames on a RECOGNIZED data channel whose shape we could
    not read — the schema-drift signal. `unrecognized_frames` counts frames whose channel we do
    not know at all. Both are distinct from a raw JSON decode failure, which never gets here.
    """
    events: List[Dict[str, Any]] = field(default_factory=list)
    schema_mismatches: int = 0
    unrecognized_frames: int = 0
    #: `(pair, detail)` for every book row whose venue checksum did NOT reconcile this frame —
    #: a desynced book must never emit again until a fresh snapshot (A.4(2)). Each entry has
    #: already been counted into `schema_mismatches`; this list carries the reconnect-forcing
    #: detail `_consume` needs to write the gap record.
    book_desyncs: List[Tuple[str, str]] = field(default_factory=list)
    #: The pair of every book row that was a schema mismatch AND whose `symbol` we could read —
    #: the input to `LiveRecorder`'s PER-PAIR drift detector. The global
    #: `_frames_since_last_event` counter resets on ANY pair's event, so one healthy pair masked
    #: another that had gone 100% unreadable (e.g. lost its snapshot after a reconnect and never
    #: got another): no gap, no escalation, and nothing anywhere gates on `n_schema_mismatch`.
    book_row_mismatch_pairs: List[str] = field(default_factory=list)

    @property
    def is_silent(self) -> bool:
        return not self.events


def _subscribe_messages(cfg: RecorderConfig) -> List[Dict[str, Any]]:
    return [
        {"method": "subscribe", "params": {"channel": "book", "symbol": list(cfg.pairs),
                                            "depth": cfg.book_depth}},
        {"method": "subscribe", "params": {"channel": "trade", "symbol": list(cfg.pairs)}},
    ]


def _finite_float(value: Any) -> Optional[float]:
    """Coerce a wire value to a finite float, or None if it is missing/garbage/non-finite.

    `json` accepts bare `NaN`/`Infinity` literals, so a venue error frame can hand us one. A
    non-finite price is a SCHEMA MISMATCH, not a tick: admitting it would let a NaN reach the
    markout arithmetic, where it survives every comparison and turns into a false claim.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


@dataclass
class _PairBookState:
    """One pair's locally-maintained book, per A.4(1). `bids`/`asks` map the wire's STRING price
    to the wire's STRING qty — the checksum (A.3) is defined over those exact string forms, so a
    round-trip through `float` would silently break it (`2.0` != `"2.00000000"`). Sort order for
    "top of book" and for the checksum is always computed fresh from `float(price)`, never
    trusted from wire/insertion order.

    Per-connection: `LiveRecorder._consume` creates a fresh `{}` of these at the start of every
    connection, so a reconnect discards all book state and waits for new snapshots (A.4(1)).
    """
    bids: "OrderedDict[str, str]" = field(default_factory=OrderedDict)
    asks: "OrderedDict[str, str]" = field(default_factory=OrderedDict)
    has_snapshot: bool = False
    #: The last EMITTED top of book (bid_px, bid_qty, ask_px, ask_qty), for the change side of
    #: the change-or-keepalive emission policy (A.4(4)).
    last_emitted: Optional[Tuple[float, float, float, float]] = None
    last_emitted_ts: Optional[float] = None


def _strip_numeric_wire_string(s: str) -> str:
    """The venue's checksum recipe (A.3): remove the decimal point, then strip leading zeros.

    `"0.35040556"` -> `"35040556"`; `"2505.81"` -> `"250581"`. A value that strips to nothing
    (e.g. `"0"` or `"0.00000000"`) keeps a single `"0"` — an empty string would silently drop a
    level's contribution to the checksum rather than contributing its true (zero) digits.

    That last branch is UNREACHABLE from a maintained book and is kept as a guard, not as live
    behaviour: a level whose qty is 0 is DELETED rather than stored (`_apply_book_deltas`) and a
    level whose price parses to <= 0 is refused (`_parse_book_levels`), so nothing that strips
    to empty ever reaches `_book_view`. It stays because the failure it prevents (a level
    contributing zero characters, shifting every subsequent digit) is silent.
    """
    s = s.replace(".", "").lstrip("0")
    return s if s else "0"


@dataclass(frozen=True)
class _BookView:
    """The SINGLE derivation of everything read off a maintained book: the venue checksum AND
    the top of book, computed together from ONE ordered, trimmed view of the state.

    THIS SHAPE IS THE FIX FOR THE RUN-45 BLOCKER. Before it, the checksum reconciled the
    maintained STATE while the emitted event was a SEPARATE derivation from that same state.
    A defect confined to the emission — e.g. reading a one-sided delta row as "the top of book"
    and consulting the maintained book only for the untouched side — therefore left every
    checksum green BY CONSTRUCTION (it never touched the state machine) and put a wrong price
    into the artifact with nothing reconciling it. Measured: that mutation corrupts 482 of the
    fixture's 590 emitted tops, including BTC/USD 79989.2 -> 79974.8 (~18 bp).

    `top` is now literally the head of the very lists that were hashed, produced by the same
    call. There is no second derivation left that could disagree with the one the venue
    reconciles — the defect is not merely tested against, it is unavailable.
    """
    checksum: int
    #: `(bid_px, bid_qty, ask_px, ask_qty)`, or `None` if either side of the top-10 view is
    #: currently empty (a legitimately thin book — not an error; simply nothing to emit).
    top: Optional[Tuple[float, float, float, float]]


def _book_view(state: _PairBookState) -> _BookView:
    """Order and trim a maintained book ONCE, then read both the venue's CRC32 checksum (A.3)
    and the top of book off that one structure.

    Checksum recipe: asks ascending (best first) then bids descending (best first), each side
    TOP-10-trimmed, `price+qty` wire strings concatenated with no separator, decimal points and
    leading zeros stripped, hashed with CRC32. Always trims to `BOOK_CHECKSUM_DEPTH` (10)
    regardless of the configured `book_depth` — the venue's checksum is defined over exactly the
    top 10 (A.3/A.4(3)), not the subscription depth. The top of book is the first element of
    each of those same two lists, so it is covered by the reconciliation, not adjacent to it.
    """
    asks_top = sorted(state.asks.items(), key=lambda kv: float(kv[0]))[:BOOK_CHECKSUM_DEPTH]
    bids_top = sorted(state.bids.items(), key=lambda kv: float(kv[0]),
                       reverse=True)[:BOOK_CHECKSUM_DEPTH]
    parts = [_strip_numeric_wire_string(p) + _strip_numeric_wire_string(q)
             for p, q in asks_top] + [_strip_numeric_wire_string(p) + _strip_numeric_wire_string(q)
                                       for p, q in bids_top]
    checksum = zlib.crc32("".join(parts).encode("ascii"))
    top: Optional[Tuple[float, float, float, float]] = None
    if asks_top and bids_top:
        top = (float(bids_top[0][0]), float(bids_top[0][1]),
               float(asks_top[0][0]), float(asks_top[0][1]))
    return _BookView(checksum, top)


def _parse_book_levels(raw: Any) -> Optional[List[Tuple[str, str]]]:
    """Validate and extract `(price_str, qty_str)` pairs from a wire `bids`/`asks` list.

    Returns `None` (the row is unreadable) if the list is malformed, a level is not an object,
    or a price/qty is missing, non-finite, or non-positive (price) / negative (qty) — the same
    standard `_finite_float` already applies to top-of-book and trade prices. `None` is
    deliberately NOT "empty list": an empty `bids`/`asks` list is normal (Kraken v2 updates are
    frequently one-sided deltas — A.1) and must be treated as "zero levels changed on this
    side", not as a parse failure.
    """
    if not isinstance(raw, list):
        return None
    out: List[Tuple[str, str]] = []
    for lvl in raw:
        if not isinstance(lvl, dict):
            return None
        price_raw, qty_raw = lvl.get("price"), lvl.get("qty")
        price_f = _finite_float(price_raw)
        qty_f = _finite_float(qty_raw)
        if price_f is None or price_f <= 0.0 or qty_f is None or qty_f < 0.0:
            return None
        out.append((str(price_raw), str(qty_raw)))
    return out


def _apply_book_deltas(book: "OrderedDict[str, str]", levels: List[Tuple[str, str]]) -> None:
    """Apply incremental deltas to one side of a maintained book (A.4(1)/A.3): `qty == 0`
    deletes the price level, otherwise the level is set to the given qty."""
    for price_str, qty_str in levels:
        if float(qty_str) == 0.0:
            book.pop(price_str, None)
        else:
            book[price_str] = qty_str


def _trim_book_side(book: "OrderedDict[str, str]", depth: int, *, best_high: bool) -> None:
    """Trim one side of a maintained book down to `depth` levels (A.4(1)'s storage discipline —
    proven against the fixture: keeping levels the venue has stopped updating once they fall out
    of its own top-N view is what desyncs the checksum, since we would then be holding a price
    level whose true current quantity we no longer receive updates for)."""
    items = sorted(book.items(), key=lambda kv: float(kv[0]), reverse=best_high)[:depth]
    book.clear()
    book.update(items)


def _parse_wire_checksum(raw: Any) -> Optional[int]:
    """`raw` is the wire's `checksum` field — an unsigned CRC32, so no `except` is needed to
    read it: it is either already a Python `int` (a hand-built test frame) or a digit-only
    string (the real wire, decoded with `parse_int=str` per A.3). Anything else is unreadable.
    """
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str) and raw.isdigit():
        return int(raw)
    return None


def _parse_book_frame(msg: Dict[str, Any], now: float,
                       book_states: Dict[str, _PairBookState],
                       book_depth: int, book_keepalive_s: float
                       ) -> Tuple[List[Dict[str, Any]], int, List[Tuple[str, str]], List[str]]:
    """Normalize one `book` channel frame against the per-pair maintained state (A.4).

    Returns `(events, schema_mismatches, book_desyncs, mismatch_pairs)`. `book_desyncs` entries
    are `(pair, detail)` for any row whose venue checksum did not reconcile — `_consume` turns
    each of those into a recorded gap and forces a reconnect (A.4(2)); a desynced book never
    emits again until a fresh snapshot arrives on the new connection. `mismatch_pairs` names the
    pair of every mismatched row whose `symbol` was readable, feeding `_consume`'s PER-PAIR drift
    detector (a per-pair failure is invisible to a counter that any other pair's event resets).
    """
    events: List[Dict[str, Any]] = []
    schema_mismatches = 0
    book_desyncs: List[Tuple[str, str]] = []
    mismatch_pairs: List[str] = []

    msg_type = msg.get("type")
    if msg_type not in ("snapshot", "update"):
        # A book frame with no recognizable snapshot/update tag is a frame-shape defect, exactly
        # like the pre-addendum "not isinstance(rows, list) or not rows" bail — one mismatch for
        # the whole frame, not a claim about any individual row we cannot even interpret.
        return events, 1, book_desyncs, mismatch_pairs

    rows = msg.get("data")
    if not isinstance(rows, list) or not rows:
        return events, 1, book_desyncs, mismatch_pairs

    for row in rows:
        if not isinstance(row, dict):
            schema_mismatches += 1
            continue
        pair = row.get("symbol")
        if not pair or not isinstance(pair, str):
            schema_mismatches += 1
            continue
        bid_levels = _parse_book_levels(row.get("bids"))
        ask_levels = _parse_book_levels(row.get("asks"))
        checksum = _parse_wire_checksum(row.get("checksum"))
        if bid_levels is None or ask_levels is None or checksum is None:
            schema_mismatches += 1
            mismatch_pairs.append(pair)
            continue

        if msg_type == "snapshot":
            state = _PairBookState(has_snapshot=True)
            state.bids = OrderedDict(bid_levels)
            state.asks = OrderedDict(ask_levels)
            _trim_book_side(state.bids, book_depth, best_high=True)
            _trim_book_side(state.asks, book_depth, best_high=False)
            book_states[pair] = state
        else:  # update
            state = book_states.get(pair)
            if state is None or not state.has_snapshot:
                # A.4(1): an update for a pair with no prior snapshot is a mismatch, never
                # applied — never fabricate a book from a partial delta stream.
                schema_mismatches += 1
                mismatch_pairs.append(pair)
                continue
            _apply_book_deltas(state.bids, bid_levels)
            _apply_book_deltas(state.asks, ask_levels)
            _trim_book_side(state.bids, book_depth, best_high=True)
            _trim_book_side(state.asks, book_depth, best_high=False)

        # A.4(2): checksum verified on EVERY book row, snapshot or update. A mismatch means our
        # book has desynced and every mid derived from it from here on would be fabricated.
        # ONE derivation (`_book_view`) yields both the checksum and the top of book, so what
        # gets emitted below is the head of exactly the structure the venue just reconciled —
        # see `_BookView` for why a second derivation here was the run-45 BLOCKER.
        view = _book_view(state)
        if view.checksum != checksum:
            schema_mismatches += 1
            book_desyncs.append(
                (pair,
                 f"{msg_type} row: computed checksum {view.checksum}, venue declared {checksum}"))
            book_states.pop(pair, None)
            mismatch_pairs.append(pair)
            continue

        top = view.top
        if top is None:
            continue  # one side legitimately empty after this row — nothing to emit, not drift
        should_emit = (
            state.last_emitted is None
            or top != state.last_emitted
            or (state.last_emitted_ts is not None
                and (now - state.last_emitted_ts) >= book_keepalive_s)
        )
        if should_emit:
            bid_px, bid_qty, ask_px, ask_qty = top
            events.append({"kind": "book", "ts": now, "pair": pair, "bid": bid_px, "ask": ask_px,
                            "bid_sz": bid_qty, "ask_sz": ask_qty})
            state.last_emitted = top
            state.last_emitted_ts = now

    return events, schema_mismatches, book_desyncs, mismatch_pairs


def _parse_frame(msg: Dict[str, Any], now: float,
                  book_states: Optional[Dict[str, _PairBookState]] = None,
                  book_depth: int = BOOK_CHECKSUM_DEPTH,
                  book_keepalive_s: float = DEFAULT_BOOK_KEEPALIVE_S) -> ParsedFrame:
    """Normalize one Kraken v2 public-feed frame into zero or more capture events.

    Defensive by design: an unrecognized or malformed frame is counted and skipped rather than
    crashing the long-lived recorder — but it is never silently counted as a healthy tick
    either. A frame on a KNOWN data channel that we cannot read is a `schema_mismatch`; a frame
    on a channel we do not know at all is an `unrecognized_frame`. Silence on the wire and
    "we are receiving frames we cannot parse" are now different, countable things.

    All event `ts` values are LOCAL RECEIPT TIME (`now`). Trades additionally carry the venue's
    own timestamp as `venue_ts`, informational only.

    `book_states` carries the per-connection maintained book (A.4(1)) and is MUTATED in place —
    callers that care about book continuity across frames (i.e. `LiveRecorder._consume`) must
    pass the SAME dict on every call; omitting it (the default) gives every call a fresh, empty
    state, which is only correct for callers that do not exercise multi-frame book continuity.
    The trade and heartbeat paths are unaffected by any of this and remain pure.
    """
    out = ParsedFrame()
    channel = msg.get("channel")

    if channel == "heartbeat":
        out.events.append({"kind": "heartbeat", "ts": now})
        return out

    if channel == "book":
        # ADDENDUM A: the book channel is MAINTAINED, not sampled — see the module docstring and
        # `_parse_book_frame`. `book_states` is per-connection (owned by the caller); a fresh
        # `{}` here (test callers that omit it) means "no snapshot yet for anything", which is
        # the correct, safe starting state.
        states = book_states if book_states is not None else {}
        events, mismatches, desyncs, mismatch_pairs = _parse_book_frame(
            msg, now, states, book_depth, book_keepalive_s)
        out.events.extend(events)
        out.schema_mismatches += mismatches
        out.book_desyncs.extend(desyncs)
        out.book_row_mismatch_pairs.extend(mismatch_pairs)
        return out

    if channel == "trade":
        rows = msg.get("data")
        if not isinstance(rows, list) or not rows:
            out.schema_mismatches += 1
            return out
        for row in rows:
            if not isinstance(row, dict):
                out.schema_mismatches += 1
                continue
            pair = row.get("symbol")
            price = _finite_float(row.get("price"))
            if not pair or price is None or price <= 0.0:
                out.schema_mismatches += 1
                continue
            volume = _finite_float(row.get("qty", 0.0))
            out.events.append({
                "kind": "trade", "ts": now, "pair": pair, "price": price,
                "volume": max(0.0, volume if volume is not None else 0.0),
                "side": row.get("side", "unknown"),
                "venue_ts": _parse_venue_ts(row.get("timestamp"), now),
            })
        return out

    if channel in NON_DATA_CHANNELS:
        return out  # recognized, legitimately eventless (status/subscription state)
    if channel is None and ("method" in msg or "result" in msg or "error" in msg):
        return out  # a method response / subscribe-ack — recognized, legitimately eventless

    out.unrecognized_frames += 1
    return out


def _parse_venue_ts(raw: Optional[str], fallback: float) -> float:
    """Parse the venue's own timestamp. INFORMATIONAL ONLY — never used in gap accounting, bar
    bucketing, or markout timing (see the module docstring's clock-basis note).
    """
    if not raw:
        return fallback
    try:
        import datetime
        s = raw.replace("Z", "+00:00")
        return datetime.datetime.fromisoformat(s).timestamp()
    except (ValueError, TypeError, AttributeError):
        return fallback


class LiveRecorder:
    """The reconnect/gap-recording state machine. `connect_fn` is injected so tests can drive
    it without a real socket: `connect_fn(url) -> async context manager` yielding an object
    with `async def send(str)` and `async def recv() -> str` (the `websockets` client protocol
    shape), or raising to simulate a failed connection attempt.
    """

    def __init__(self, cfg: RecorderConfig, connect_fn: Callable[[str], Any],
                 writer: Optional[CaptureWriter] = None, clock: Callable[[], float] = time.time,
                 sleep_fn: Optional[Callable[[float], Any]] = None):
        self.cfg = cfg
        self._connect_fn = connect_fn
        self._writer = writer or CaptureWriter(
            cfg.out_path, cfg.pairs, cfg.capture_id,
            channels=list(SUBSCRIBED_CHANNELS),
            # Declared into the capture so the reader can DISCLOSE this knob next to its own
            # gap-detect threshold. The two are independently configured; a capture that does not
            # say what its recorder's liveness tolerance was leaves the reader unable to state
            # what "no gap was recorded" is actually evidence of.
            heartbeat_timeout_s=cfg.heartbeat_timeout_s,
            # A.4(4): an undeclared emission policy would be a fourth unreconciled declaration —
            # the artifact must state the cadence floor its book numbers were measured under.
            book_emission_policy=BOOK_EMISSION_POLICY,
            book_keepalive_s=cfg.book_keepalive_s)
        self._clock = clock
        self._sleep = sleep_fn or asyncio.sleep
        self._last_event_ts: Optional[float] = None
        self._stop = False
        self.attempts = 0
        # Three DISTINCT health counters — collapsing them is how schema drift hid before.
        self.n_json_decode_errors = 0
        self.n_schema_mismatch = 0
        self.n_unrecognized_frames = 0
        self._frames_since_last_event = 0
        self._frames_since_counters_flush = 0
        self._counters_dirty = False
        #: Per-pair maintained book state (A.4(1)) — reset at the start of every connection in
        #: `_consume`, so a reconnect discards it and waits for fresh snapshots.
        self._book_states: Dict[str, _PairBookState] = {}
        #: Consecutive unreadable book rows PER PAIR (reset by that pair's next book event).
        #: `_frames_since_last_event` is global and resets on ANY pair's event, so one healthy
        #: pair kept the global detector permanently armed-and-reset while another pair was
        #: 100% unreadable — no gap, no escalation, nothing gating on the counter.
        self._book_mismatch_streak: Dict[str, int] = {}
        #: Set when a reconnect-forcing gap is written; closed by the FIRST event after the
        #: reconnect. The recorded gap itself is zero-length by construction (it ends at `now`,
        #: which is also `_last_event_ts`), so without this the backoff + reconnect + resubscribe
        #: + snapshot-wait window carried NO gap record and the reader's coverage accounting
        #: counted it as observed.
        self._dark_since: Optional[float] = None
        self._dark_reason: str = ""

    @property
    def unparsed(self) -> int:
        """Back-compat alias: raw JSON-decode failures ONLY. Schema drift is NOT counted here
        (that was the defect — a schema-mismatched frame never incremented anything).
        """
        return self.n_json_decode_errors

    def stop(self) -> None:
        self._stop = True

    async def run(self, max_iterations: Optional[int] = None) -> None:
        """Run the reconnect loop. `max_iterations` is test-only (bounds an otherwise
        unbounded loop); production callers omit it and run until `stop()` / process kill.
        """
        iterations = 0
        backoff = self.cfg.backoff_base_s
        while not self._stop:
            self.attempts += 1
            try:
                async with self._connect_fn(WS_URL) as ws:
                    for sub in _subscribe_messages(self.cfg):
                        await ws.send(json.dumps(sub))
                    backoff = self.cfg.backoff_base_s
                    await self._consume(ws)
            except _GapAlreadyRecorded as exc:
                logger.warning("907 recorder: reconnecting after %s; backing off %.1fs",
                                exc, backoff)
                await self._sleep(backoff)
                backoff = min(backoff * 2, self.cfg.backoff_max_s)
            except Exception as exc:  # noqa: BLE001 — any transport failure triggers a gap
                self._record_gap("disconnect", f"{type(exc).__name__}: {exc}",
                                  dark_until_next_event=True)
                logger.warning("907 recorder: connection attempt %d failed (%s); backing off "
                                "%.1fs before retry — NOT a silent healthy tick", self.attempts,
                                exc, backoff)
                await self._sleep(backoff)
                backoff = min(backoff * 2, self.cfg.backoff_max_s)
            iterations += 1
            if max_iterations is not None and iterations >= max_iterations:
                break

    async def _consume(self, ws: Any) -> None:
        # A.4(1): book state is per-CONNECTION. `_consume` is entered once per successful
        # connection (see `run()`), so resetting here — rather than in `__init__` — is what
        # makes a reconnect discard stale book state and wait for fresh snapshots.
        self._book_states = {}
        self._book_mismatch_streak = {}
        while not self._stop:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=self.cfg.heartbeat_timeout_s)
            except asyncio.TimeoutError:
                self._record_gap("heartbeat_timeout",
                                  f"no frame for {self.cfg.heartbeat_timeout_s}s",
                                  dark_until_next_event=True)
                raise _GapAlreadyRecorded("heartbeat timeout — forcing reconnect")
            now = self._clock()
            prev = self._last_event_ts
            if prev is not None and (now - prev) > self.cfg.heartbeat_timeout_s:
                self._record_gap("clock_jump", f"local clock advanced {now - prev:.1f}s "
                                                f"between consecutive recv() calls")
            self._frames_since_last_event += 1
            self._frames_since_counters_flush += 1
            try:
                # A.3: the book checksum is defined over the wire's STRING forms — a bare
                # `json.loads(raw)` round-trips prices/qtys through Python `float` and silently
                # breaks it (`2.0` != `"2.00000000"`). `_finite_float` already accepts strings,
                # so the trade/heartbeat paths are unaffected (A.4(3)/A.5(A-iii)).
                msg = json.loads(raw, parse_float=str, parse_int=str)
            except json.JSONDecodeError:
                self.n_json_decode_errors += 1
                self._counters_dirty = True
                self._check_schema_drift()
                continue
            if not isinstance(msg, dict):
                self.n_unrecognized_frames += 1
                self._counters_dirty = True
                self._check_schema_drift()
                continue

            parsed = _parse_frame(msg, now, self._book_states, self.cfg.book_depth,
                                   self.cfg.book_keepalive_s)
            if parsed.schema_mismatches:
                self.n_schema_mismatch += parsed.schema_mismatches
                self._counters_dirty = True
                logger.warning("907 recorder: %d schema-mismatched row(s) on channel %r — "
                                "Kraken v2's live schema is UNVERIFIED by this row; this is the "
                                "expected drift failure, not venue silence (cumulative: %d)",
                                parsed.schema_mismatches, msg.get("channel"),
                                self.n_schema_mismatch)
            if parsed.unrecognized_frames:
                self.n_unrecognized_frames += parsed.unrecognized_frames
                self._counters_dirty = True

            for mismatched_pair in parsed.book_row_mismatch_pairs:
                self._book_mismatch_streak[mismatched_pair] = (
                    self._book_mismatch_streak.get(mismatched_pair, 0) + 1)

            if parsed.events:
                self._frames_since_last_event = 0
                for ev in parsed.events:
                    if ev["kind"] == "book":
                        self._book_mismatch_streak[ev["pair"]] = 0
                    self._emit(ev)
            elif not parsed.book_desyncs:
                self._check_schema_drift()

            if parsed.book_desyncs:
                # A.4(2): a checksum mismatch means our book has desynced — drop it, emit
                # nothing further for it, and force a reconnect so a fresh snapshot arrives.
                # Book state is per-connection (A.4(1)), so a full reconnect is the only
                # fail-closed action available; any OTHER pair's valid events from this same
                # frame were already emitted above.
                for pair, detail in parsed.book_desyncs:
                    logger.error("907 recorder: BOOK CHECKSUM MISMATCH for %s — %s; dropping "
                                 "book state and forcing reconnect", pair, detail)
                combined = "; ".join(f"{pair}: {detail}" for pair, detail in parsed.book_desyncs)
                self._record_gap("schema_drift", f"book checksum mismatch — {combined}",
                                  dark_until_next_event=True)
                raise _GapAlreadyRecorded("book checksum mismatch — forcing reconnect")

            # AFTER the desync block on purpose: a desync is the more specific diagnosis and
            # forces the same reconnect, so it must be the one that gets to write the gap when a
            # row is both a desync and the row that tips a pair's streak over the threshold.
            self._check_per_pair_book_drift()

            if (self._counters_dirty
                    and self._frames_since_counters_flush >= COUNTERS_FLUSH_EVERY_N_FRAMES):
                self._flush_counters(now)

    def _check_schema_drift(self) -> None:
        """The "connected but recording nothing" detector — see the module docstring.

        Trips only when frames ARE arriving (so this is not venue silence, which the heartbeat
        timeout already owns) and none of them produced an event for a sustained run, with at
        least one of them positively unparseable. Writes a real `schema_drift` gap so gate (iv)
        refuses to carry a number across the span, then forces a reconnect.
        """
        if self._frames_since_last_event < self.cfg.schema_drift_frame_threshold:
            return
        if (self.n_schema_mismatch + self.n_unrecognized_frames + self.n_json_decode_errors) == 0:
            return
        detail = (f"{self._frames_since_last_event} consecutive frames yielded ZERO events "
                  f"(schema_mismatch={self.n_schema_mismatch}, "
                  f"unrecognized={self.n_unrecognized_frames}, "
                  f"json_decode_errors={self.n_json_decode_errors}) — connected but recording "
                  f"nothing; treating as a capture gap, not a quiet market")
        logger.error("907 recorder: SCHEMA DRIFT — %s", detail)
        self._record_gap("schema_drift", detail, dark_until_next_event=True)
        self._frames_since_last_event = 0
        raise _GapAlreadyRecorded("schema drift — forcing reconnect")

    def _check_per_pair_book_drift(self) -> None:
        """The PER-PAIR half of the "connected but recording nothing" detector.

        `_check_schema_drift` counts frames since ANY event, so it cannot fire while a single
        pair stays healthy. The escaping scenario it misses: pair B loses its book after a
        reconnect and never receives another snapshot, so 100% of B's rows are `schema_mismatch`
        forever while pair A keeps the global counter pinned at zero. No gap is written, nothing
        escalates, and no gate anywhere reads `n_schema_mismatch` — B simply stops existing in
        the artifact, which is precisely the "a declared thing recorded nothing" failure the rest
        of this row refuses. A per-pair streak turns it into a recorded gap and a reconnect.
        """
        stuck = [(pair, n) for pair, n in self._book_mismatch_streak.items()
                 if n >= self.cfg.schema_drift_frame_threshold]
        if not stuck:
            return
        detail = ("per-pair book drift: " + ", ".join(
            f"{pair} produced {n} consecutive unreadable book rows with no book event in between"
            for pair, n in sorted(stuck))
            + " — this pair is connected but recording nothing (a global frame counter cannot "
              "see it, because the other pairs' events keep resetting it); treating as a capture "
              "gap, not a quiet market")
        logger.error("907 recorder: PER-PAIR BOOK DRIFT — %s", detail)
        self._record_gap("schema_drift", detail, dark_until_next_event=True)
        self._book_mismatch_streak = {}
        raise _GapAlreadyRecorded("per-pair book drift — forcing reconnect")

    def _emit(self, ev: Dict[str, Any]) -> None:
        self._close_dark_window(ev["ts"])
        self._last_event_ts = ev["ts"]
        kind = ev["kind"]
        if kind == "heartbeat":
            self._writer.write_heartbeat(ev["ts"])
        elif kind == "book":
            self._writer.write_book(ev["pair"], ev["ts"], ev["bid"], ev["ask"],
                                     ev["bid_sz"], ev["ask_sz"])
        elif kind == "trade":
            self._writer.write_trade(ev["pair"], ev["ts"], ev["price"], ev["volume"],
                                      ev["side"], venue_ts=ev.get("venue_ts"))

    def _flush_counters(self, ts: float) -> None:
        self._writer.write_counters(ts, self.n_json_decode_errors, self.n_schema_mismatch,
                                     self.n_unrecognized_frames)
        self._counters_dirty = False
        self._frames_since_counters_flush = 0

    def _close_dark_window(self, ts: float) -> None:
        """Close the outage a reconnect-forcing gap opened, at the FIRST event after it.

        The gap `_record_gap` writes for a forced reconnect is zero-length by construction: any
        other pair's events from the same frame were emitted first, so `_last_event_ts == now`
        and the record spans `[now, now]`. The DARK TIME that follows — backoff (2s doubling to
        60s) + reconnect + resubscribe + waiting for a fresh snapshot — carried no gap record at
        all, and the reader only flags an unrecorded jump above its 60s threshold, so an outage
        under a minute was silently counted as observed time in the coverage accounting.

        Recorded as `disconnect` (the reason set is the capture format's, and "we were not
        connected across this span" is true of every path that arms this — heartbeat timeout,
        schema drift and checksum desync all tear the socket down); the ORIGINATING reason is
        named in `detail`, and its own zero-length record is still in the ledger immediately
        before this one.

        Deliberately NOT flushed at `close()`: a window that never closed has no event after it,
        so it lies outside the capture's event span and contributes nothing to coverage —
        writing it would add a gap past the last event rather than explain time inside the span.
        """
        since = self._dark_since
        if since is None:
            return
        self._dark_since = None
        if ts <= since:
            return
        self._writer.write_gap(GapRecord(
            since, ts, "disconnect",
            detail=f"dark window: no frame observed between the recorded "
                   f"{self._dark_reason!r} gap and the first event after reconnecting"))

    def _record_gap(self, reason: str, detail: str, *,
                     dark_until_next_event: bool = False) -> None:
        now = self._clock()
        start = self._last_event_ts if self._last_event_ts is not None else now
        if now <= start:
            now = start  # zero-length gap still gets recorded — never dropped for being brief
        self._writer.write_gap(GapRecord(start, now, reason, detail=detail))
        if dark_until_next_event:
            # This gap forces a reconnect: everything from here until the first event on the new
            # connection is unobserved. See `_close_dark_window`.
            self._dark_since = now
            self._dark_reason = reason
        self._last_event_ts = now
        # Persist counters alongside every gap: a gap is exactly the moment a reader will want
        # to know whether we were silent or merely unable to parse what we were being sent.
        self._flush_counters(now)

    def close(self, clean: bool = True) -> None:
        self._flush_counters(self._clock())
        self._writer.close(clean=clean)


def build_argparser():
    import argparse
    p = argparse.ArgumentParser(
        # allow_abbrev=False: a two-character prefix must never stand in for a flag on a
        # safety-relevant entry point (the sibling 881D row shipped exactly that defect).
        allow_abbrev=False,
        description="TASK_907 phase 1 — long-lived Kraken PUBLIC-feed recorder. Launch under "
                    "tmux + tools/safe_run.sh on gpu-server (orchestrator-launched, not run "
                    "from here).")
    p.add_argument("--pairs", required=True,
                    help="comma-separated Kraken pair symbols, e.g. BTC/USD,ETH/USD")
    p.add_argument("--out", required=True,
                    help="capture file path under runs/907_live_fill_probe/ (untracked)")
    p.add_argument("--capture-id", default=None)
    p.add_argument("--heartbeat-timeout-s", type=float, default=30.0)
    return p


def main(argv: Optional[List[str]] = None) -> None:  # pragma: no cover — live entry point
    import uuid
    args = build_argparser().parse_args(argv)
    try:
        import websockets
    except ImportError as exc:
        raise SystemExit(f"the 'websockets' package is required to run the live recorder: {exc}")

    cfg = RecorderConfig(
        pairs=[p.strip() for p in args.pairs.split(",") if p.strip()],
        out_path=args.out,
        capture_id=args.capture_id or f"907-{uuid.uuid4().hex[:12]}",
        heartbeat_timeout_s=args.heartbeat_timeout_s,
    )

    def connect_fn(url: str):
        return websockets.connect(url, ping_interval=20, ping_timeout=15, close_timeout=10)

    recorder = LiveRecorder(cfg, connect_fn)
    try:
        asyncio.run(recorder.run())
    finally:
        recorder.close(clean=True)


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    main()
