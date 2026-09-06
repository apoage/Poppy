"""TASK_907 — the recorder's reconnect/gap-recording state machine, driven WITHOUT a real
socket (dependency-injected `connect_fn`), per the task brief: prove the code works on a
synthetic/replayed harness, not a live multi-day run.
"""
import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture import CaptureFormatError, read_capture  # noqa: E402
from recorder import (  # noqa: E402
    SUBSCRIBE_ACK_MISMATCH_TERMINAL_ATTEMPTS, SUBSCRIBED_CHANNELS, WS_URL, LiveRecorder,
    RecorderConfig, SubscriptionConfigMismatch, _parse_book_frame, _parse_frame,
    _subscribe_messages,
)
from phase2_rails import Phase2RefusalError, RailViolation  # noqa: E402

#: CRC32 of the synthetic one-level-per-side book used throughout this file (bids 100.0/1.0,
#: asks 100.2/2.0), DERIVED BY HAND from A.3's recipe and pinned as a literal: asks
#: `"100.2"`/`"2.0"` -> `"1002"` + `"20"`, bids `"100.0"`/`"1.0"` -> `"1000"` + `"10"`,
#: concatenated asks-then-bids -> `"100220100010"`, `zlib.crc32(b"100220100010") == 2169449095`.
#:
#: It used to be computed by calling `_book_checksum` itself, which gave the assertion below ZERO
#: discriminating power: swapping the asks/bids concatenation order inside `_book_checksum` left
#: this test GREEN, because it only ever asserted that the function agrees with itself.
#: LIMITATION, stated: a one-level literal pins the side order and the strip rule and nothing
#: more — depth trimming, `qty == 0` delete semantics and leading-zero handling are pinned only
#: by the real-wire fixture tests in `test_book_normalization.py`.
SYNTHETIC_BOOK_CHECKSUM = 2169449095


class _AsyncCM:
    def __init__(self, ws=None, exc=None):
        self._ws, self._exc = ws, exc

    async def __aenter__(self):
        if self._exc is not None:
            raise self._exc
        return self._ws

    async def __aexit__(self, *a):
        return False


class FakeWS:
    def __init__(self, recv_items):
        self._items = list(recv_items)
        self.sent = []

    async def send(self, msg):
        self.sent.append(msg)

    async def recv(self):
        if not self._items:
            raise RuntimeError("fake connection exhausted")
        item = self._items.pop(0)
        if isinstance(item, BaseException):
            raise item
        if item == "__HANG__":
            await asyncio.sleep(999)
        return item


async def _fast_sleep(_seconds):
    return None


def _trade_frame(price, ts_iso="2024-01-01T00:00:00Z"):
    return json.dumps({"channel": "trade", "data": [
        {"symbol": "BTC/USD", "price": price, "qty": 1.0, "side": "buy", "timestamp": ts_iso}]})


def _heartbeat_frame():
    return json.dumps({"channel": "heartbeat"})


def _subscribe_ack_frame(channel, symbol):
    return json.dumps({"method": "subscribe", "result": {"channel": channel, "symbol": symbol}})


def test_connect_failure_then_success_records_a_disconnect_gap_and_captures_data(tmp_path):
    out = str(tmp_path / "cap.jsonl")
    attempts = []

    def connect_fn(url):
        attempts.append(url)
        if len(attempts) == 1:
            return _AsyncCM(exc=ConnectionError("simulated connect failure"))
        ws = FakeWS([_trade_frame(100.0), RuntimeError("simulated mid-stream drop")])
        return _AsyncCM(ws=ws)

    cfg = RecorderConfig(pairs=["BTC/USD"], out_path=out, capture_id="t1",
                          heartbeat_timeout_s=5.0, backoff_base_s=0.001, backoff_max_s=0.01)
    rec = LiveRecorder(cfg, connect_fn, sleep_fn=_fast_sleep)
    asyncio.run(rec.run(max_iterations=2))
    rec.close(clean=False)

    assert len(attempts) == 2  # one failed attempt, one that connected
    meta = read_capture(out)
    assert meta.n_trade == 1
    assert len(meta.gaps) >= 1
    assert all(g.reason == "disconnect" for g in meta.gaps)
    assert meta.gaps[0].duration_s >= 0  # zero-length gaps are still recorded, never dropped


def test_heartbeat_timeout_forces_reconnect_with_exactly_one_gap(tmp_path):
    out = str(tmp_path / "cap.jsonl")
    calls = {"n": 0}

    def connect_fn(url):
        calls["n"] += 1
        ws = FakeWS(["__HANG__"])  # never responds within the heartbeat window
        return _AsyncCM(ws=ws)

    cfg = RecorderConfig(pairs=["BTC/USD"], out_path=out, capture_id="t2",
                          heartbeat_timeout_s=0.02, backoff_base_s=0.001, backoff_max_s=0.01)
    rec = LiveRecorder(cfg, connect_fn, sleep_fn=_fast_sleep)
    asyncio.run(rec.run(max_iterations=1))
    rec.close(clean=False)

    meta = read_capture(out)
    assert len(meta.gaps) == 1  # NOT double-recorded (heartbeat_timeout, not also a disconnect)
    assert meta.gaps[0].reason == "heartbeat_timeout"


def test_reconnect_loop_never_silently_treats_a_failure_as_healthy(tmp_path):
    """Every failed attempt logs/records something — the loop cannot spin invisibly."""
    out = str(tmp_path / "cap.jsonl")

    def always_fails(url):
        return _AsyncCM(exc=ConnectionError("always down"))

    cfg = RecorderConfig(pairs=["BTC/USD"], out_path=out, capture_id="t3",
                          heartbeat_timeout_s=5.0, backoff_base_s=0.001, backoff_max_s=0.01)
    rec = LiveRecorder(cfg, always_fails, sleep_fn=_fast_sleep)
    asyncio.run(rec.run(max_iterations=5))
    rec.close(clean=False)

    assert rec.attempts == 5
    meta = read_capture(out)
    assert len(meta.gaps) == 5  # one recorded per failed attempt — nothing is swallowed


def test_parse_frame_book_and_trade_and_heartbeat():
    # ADDENDUM A (run 45, on measured wire evidence): this test previously fed `_parse_frame` a
    # book row with no `type` and no `checksum` and asserted it emitted a `bids[0]`/`asks[0]`
    # "top of book" sample directly. That is the OLD, WRONG behaviour A.1/A.2 measured against
    # the live feed: Kraken v2 `book` `update` rows are one-sided incremental deltas at
    # arbitrary depth, not a top-of-book snapshot, and `bids[0]` is not reliably the best bid.
    # The book channel is now a maintained per-pair state machine verified against the venue's
    # own CRC32 checksum (A.3/A.4) — a `snapshot` frame with a checksum computed the same way
    # the production code verifies it is the correct replacement fixture.
    # The checksum here is a HAND-DERIVED LITERAL (see SYNTHETIC_BOOK_CHECKSUM) — computing it
    # with `_book_checksum` made this assertion circular and blind to a swapped side order.
    book_msg = {"channel": "book", "type": "snapshot", "data": [{
        "symbol": "BTC/USD",
        "bids": [{"price": "100.0", "qty": "1.0"}],
        "asks": [{"price": "100.2", "qty": "2.0"}],
        "checksum": SYNTHETIC_BOOK_CHECKSUM,
        "timestamp": "2026-08-27T00:00:00Z",
    }]}
    parsed = _parse_frame(book_msg, now=42.0, book_states={})
    assert parsed.events == [{"kind": "book", "ts": 42.0, "pair": "BTC/USD", "bid": 100.0,
                               "ask": 100.2, "bid_sz": 1.0, "ask_sz": 2.0}]
    assert parsed.schema_mismatches == 0 and parsed.unrecognized_frames == 0
    assert parsed.book_desyncs == []

    trade_msg = json.loads(_trade_frame(101.5))
    parsed = _parse_frame(trade_msg, now=42.0)
    assert len(parsed.events) == 1
    ev = parsed.events[0]
    assert ev["kind"] == "trade" and ev["pair"] == "BTC/USD" and ev["price"] == 101.5
    # THE CLOCK BASIS: `ts` is local receipt time; the venue's timestamp is informational only.
    assert ev["ts"] == 42.0
    assert ev["venue_ts"] != 42.0  # the 2024-01-01 venue stamp, kept but not used for anything

    parsed = _parse_frame({"channel": "heartbeat"}, now=7.0)
    assert parsed.events == [{"kind": "heartbeat", "ts": 7.0}]

    # a recognized-but-eventless status frame is NOT a schema mismatch and must not raise
    parsed = _parse_frame({"channel": "status", "data": []}, now=1.0)
    assert parsed.events == [] and parsed.schema_mismatches == 0
    assert parsed.unrecognized_frames == 0


# --- MAJOR (run-43): schema-mismatched frames must be COUNTED, PERSISTED and made LOUD ---
#
# The defect: a valid-JSON frame with an unexpected shape for a recognized channel returned []
# and was classified as an innocuous ack/status frame — indistinguishable from real venue
# silence. `unparsed` incremented ONLY on json.JSONDecodeError, never on schema drift, and was
# never written to the capture, logged, or surfaced in the artifact. Kraken v2's live schema is
# UNVERIFIED by this row, so drift is the EXPECTED failure mode, not a hypothetical one.

def _drifted_book_frame():
    """A `book` frame whose rows use a RENAMED price field — exactly what a v2 field rename
    looks like on the wire. Valid JSON, recognized channel, well-formed at the FRAME level
    (`type` and `checksum` present and readable), unreadable at the ROW level.

    ADDENDUM A fix pass: this fixture used to omit `type`, so under `_parse_book_frame` it was
    rejected as a WHOLE-FRAME shape defect before any row was looked at — it never reached the
    per-row validation it was written to exercise, and every per-row check had zero coverage
    (relaxing `_parse_book_levels` left the suite fully green). It now carries a valid `type` and
    a readable `checksum`, so the ONLY thing wrong with it is the renamed price field, and the
    per-row price validation is what rejects it. See
    `test_a_book_frame_with_no_snapshot_or_update_tag_is_rejected_whole` for the other layer.
    """
    return json.dumps({"channel": "book", "type": "update", "data": [
        {"symbol": "BTC/USD", "bids": [{"px": 100.0, "qty": 1.0}],
         "asks": [{"px": 100.2, "qty": 1.0}], "checksum": 1}]})


# The synthetic book every per-row fixture below is a single deliberate mutation of. Its checksum
# is the hand-derived literal, and it is CORRECT for this book — so a per-row check that stops
# rejecting its defect lets the row through to a book that then reconciles, which is the only way
# these tests can tell "rejected at parse" from "rejected later by the checksum".
_GOOD_BOOK_ROW = {
    "symbol": "BTC/USD",
    "bids": [{"price": "100.0", "qty": "1.0"}],
    "asks": [{"price": "100.2", "qty": "2.0"}],
    "checksum": SYNTHETIC_BOOK_CHECKSUM,
}


def _snapshot_frame(row):
    return {"channel": "book", "type": "snapshot", "data": [row]}


class _Missing:
    """Sentinel: `_row_with(bids=_MISSING)` DELETES the key rather than setting it to None."""


_MISSING = _Missing()


def _row_with(**overrides):
    row = dict(_GOOD_BOOK_ROW)
    for k, v in overrides.items():
        if isinstance(v, _Missing):
            row.pop(k, None)
        else:
            row[k] = v
    return row


def test_the_good_row_this_files_defect_fixtures_mutate_is_actually_accepted():
    """Non-vacuity for everything below: the unmutated row parses, reconciles and EMITS. Without
    this, a per-row test suite that rejects everything would look identical to one that works."""
    states = {}
    events, mismatches, desyncs, _ = _parse_book_frame(_snapshot_frame(dict(_GOOD_BOOK_ROW)),
                                                     1.0, states, 10, 0.0)
    assert mismatches == 0 and desyncs == []
    assert events == [{"kind": "book", "ts": 1.0, "pair": "BTC/USD", "bid": 100.0, "ask": 100.2,
                        "bid_sz": 1.0, "ask_sz": 2.0}]
    assert "BTC/USD" in states


@pytest.mark.parametrize("defect,row", [
    ("renamed price field", _row_with(bids=[{"px": "100.0", "qty": "1.0"}])),
    ("missing qty", _row_with(bids=[{"price": "100.0"}])),
    ("missing price", _row_with(asks=[{"qty": "1.0"}])),
    ("non-finite price", _row_with(bids=[{"price": float("nan"), "qty": "1.0"}])),
    ("non-finite qty", _row_with(bids=[{"price": "100.0", "qty": float("inf")}])),
    ("non-positive price", _row_with(bids=[{"price": "0.0", "qty": "1.0"}])),
    ("negative price", _row_with(bids=[{"price": "-100.0", "qty": "1.0"}])),
    ("negative qty", _row_with(bids=[{"price": "100.0", "qty": "-1.0"}])),
    ("boolean price", _row_with(bids=[{"price": True, "qty": "1.0"}])),
    ("level is not an object", _row_with(bids=["100.0"])),
    ("side missing entirely", _row_with(bids=_MISSING)),
    ("side is not a list", _row_with(bids="100.0")),
    ("symbol missing", _row_with(symbol=_MISSING)),
    ("symbol is not a string", _row_with(symbol=123)),
    ("symbol is empty", _row_with(symbol="")),
    ("checksum missing", _row_with(checksum=_MISSING)),
    ("checksum unreadable", _row_with(checksum="not-a-number")),
    ("checksum is a bool", _row_with(checksum=True)),
])
def test_each_per_row_book_defect_is_rejected_before_it_can_touch_the_book(defect, row):
    """MAJOR (run-45 regression lens): every per-row validation must be REACHABLE and
    load-bearing. Each row here is well-formed at the frame level and differs from
    `_GOOD_BOOK_ROW` (which is accepted, above) by exactly one defect, so the per-row loop is
    entered and this one check is the only thing that can reject it.

    The assertions are deliberately stronger than a mismatch count: `desyncs == []` and an empty
    `book_states` pin that the row was refused AT PARSE, before it could be applied to a book. A
    relaxed check that lets the row through would produce either an emitted event or a checksum
    DESYNC (which drops the book and forces a reconnect) — both of which this test catches, and
    neither of which a bare `n_schema_mismatch == 1` would."""
    states = {}
    events, mismatches, desyncs, _ = _parse_book_frame(_snapshot_frame(row), 1.0, states, 10, 0.0)
    assert events == [], f"{defect}: an unreadable row must never emit"
    assert mismatches == 1, f"{defect}: must count exactly one schema mismatch"
    assert desyncs == [], (
        f"{defect}: must be rejected at PARSE, not applied and then caught by the checksum")
    assert states == {}, f"{defect}: an unreadable row must never reach the maintained book"


def test_a_row_that_is_not_an_object_at_all_is_rejected_per_row():
    states = {}
    events, mismatches, desyncs, _ = _parse_book_frame(_snapshot_frame("not-an-object"),
                                                     1.0, states, 10, 0.0)
    assert (events, mismatches, desyncs) == ([], 1, [])
    assert states == {}


def test_a_book_frame_with_no_snapshot_or_update_tag_is_rejected_whole():
    """The OTHER layer, pinned so the two cannot be confused: a book frame with no
    `snapshot`/`update` tag is a whole-FRAME defect and never reaches the per-row loop. Two
    perfectly good rows here yield ONE mismatch and ZERO events — which is precisely why the
    old `type`-less fixtures gave the per-row validations no coverage at all."""
    events, mismatches, desyncs, _ = _parse_book_frame(
        {"channel": "book", "data": [dict(_GOOD_BOOK_ROW), dict(_GOOD_BOOK_ROW)]},
        1.0, {}, 10, 0.0)
    assert (events, mismatches, desyncs) == ([], 1, [])


def _run_one_connection(tmp_path, frames, threshold=200, name="cap.jsonl"):
    out = str(tmp_path / name)

    def connect_fn(url):
        return _AsyncCM(ws=FakeWS(list(frames)))

    cfg = RecorderConfig(pairs=["BTC/USD"], out_path=out, capture_id="t",
                          heartbeat_timeout_s=5.0, backoff_base_s=0.001, backoff_max_s=0.01,
                          schema_drift_frame_threshold=threshold)
    rec = LiveRecorder(cfg, connect_fn, sleep_fn=_fast_sleep)
    asyncio.run(rec.run(max_iterations=1))
    rec.close(clean=False)
    return rec, out


def test_schema_mismatch_is_counted_separately_from_json_decode_failure(tmp_path):
    rec, out = _run_one_connection(tmp_path, [
        _drifted_book_frame(),          # schema mismatch: recognized channel, unreadable rows
        "{not json at all",             # a raw JSON decode failure
        _trade_frame(100.0),            # one healthy event so the capture has data
    ])
    assert rec.n_schema_mismatch == 1, "schema drift must increment its OWN counter"
    assert rec.n_json_decode_errors == 1
    assert rec.unparsed == 1, "the back-compat alias stays JSON-decode-only"
    assert rec.n_unrecognized_frames == 0


def test_unrecognized_channel_is_counted_separately_too(tmp_path):
    rec, _ = _run_one_connection(tmp_path, [
        json.dumps({"channel": "some_channel_we_do_not_know", "data": [{"x": 1}]}),
        _trade_frame(100.0),
    ])
    assert rec.n_unrecognized_frames == 1
    assert rec.n_schema_mismatch == 0
    assert rec.n_json_decode_errors == 0


def test_recognized_eventless_frames_are_not_counted_as_drift(tmp_path):
    """Non-vacuity in the other direction: legitimate status/ack traffic must stay silent."""
    rec, _ = _run_one_connection(tmp_path, [
        json.dumps({"channel": "status", "data": [{"system": "online"}]}),
        json.dumps({"method": "subscribe", "success": True, "result": {"channel": "book"}}),
        _trade_frame(100.0),
    ])
    assert rec.n_schema_mismatch == 0
    assert rec.n_unrecognized_frames == 0
    assert rec.n_json_decode_errors == 0


def test_counters_are_persisted_into_the_capture_and_read_back(tmp_path):
    """They must not live only in a process that has since exited."""
    rec, out = _run_one_connection(tmp_path, [
        _drifted_book_frame(), _drifted_book_frame(), "{bad", _trade_frame(100.0),
    ])
    meta = read_capture(out)
    assert meta.counters_present, "the capture must carry the recorder's health counters"
    assert meta.n_schema_mismatch == 2
    assert meta.n_json_decode_errors == 1


def test_sustained_schema_drift_writes_a_gap_and_forces_reconnect(tmp_path):
    """THE DETECTION RULE for "connected but recording nothing": N consecutive frames yielding
    zero events, with at least one positively unparseable, becomes a `schema_drift` GAP — so
    gate (iv) then refuses to carry any number across the span. It must NOT look like a quiet
    market."""
    frames = [_drifted_book_frame()] * 12 + [_trade_frame(100.0)]
    rec, out = _run_one_connection(tmp_path, frames, threshold=10)

    meta = read_capture(out)
    drift_gaps = [g for g in meta.gaps if g.reason == "schema_drift"]
    assert len(drift_gaps) == 1, "sustained drift must become a RECORDED gap, not silence"
    assert "connected but recording nothing" in drift_gaps[0].detail
    assert "schema_mismatch=10" in drift_gaps[0].detail
    assert rec.n_schema_mismatch >= 10


# --- MINOR (run-45): the PER-PAIR escape from the global drift detector ---
#
# `_frames_since_last_event` resets on ANY pair's event, so with two pairs subscribed the global
# detector cannot fire while one of them is healthy. The escaping scenario: pair B loses its book
# after a reconnect and never gets another snapshot, so 100% of B's rows are `schema_mismatch`
# forever — no gap, no escalation, and nothing anywhere gates on `n_schema_mismatch`. B just
# stops existing in the artifact.

def _unsnapshotted_update(pair):
    """A well-formed `update` for a pair with no snapshot — the exact shape a pair stuck without
    a book produces on every single frame."""
    return json.dumps({"channel": "book", "type": "update", "data": [
        {"symbol": pair, "bids": [{"price": "100.0", "qty": "1.0"}], "asks": [],
         "checksum": 12345}]})


def test_one_pair_stuck_without_a_book_is_detected_even_while_another_pair_is_healthy(tmp_path):
    frames = []
    for _ in range(12):
        frames.append(json.dumps({"channel": "heartbeat"}))   # keeps the GLOBAL counter at zero
        frames.append(_unsnapshotted_update("ETH/USD"))
    frames.append(_trade_frame(100.0))
    rec, out = _run_one_connection(tmp_path, frames, threshold=10)

    meta = read_capture(out)
    drift = [g for g in meta.gaps if g.reason == "schema_drift"]
    assert len(drift) == 1, (
        "a pair that is 100% unreadable must become a RECORDED gap even though heartbeats and "
        "the other pair keep the global frame counter pinned at zero")
    assert "per-pair book drift" in drift[0].detail
    assert "ETH/USD" in drift[0].detail
    # and the global detector demonstrably did NOT fire — this is the per-pair path, not that one
    assert "consecutive frames yielded ZERO events" not in drift[0].detail


def test_a_pair_whose_book_keeps_recovering_never_trips_the_per_pair_detector(tmp_path):
    """Non-vacuity: intermittent unreadable rows are not "stuck". The streak must reset on that
    pair's next successful book event, or the detector would fire on healthy noisy traffic."""
    good = json.dumps({"channel": "book", "type": "snapshot", "data": [dict(_GOOD_BOOK_ROW)]})
    bad = json.dumps({"channel": "book", "type": "update",
                       "data": [_row_with(checksum="not-a-number")]})
    frames = []
    for _ in range(12):
        frames.extend([good, bad])
    rec, out = _run_one_connection(tmp_path, frames, threshold=10)

    assert rec.n_schema_mismatch == 12, "the unreadable rows must still be COUNTED"
    meta = read_capture(out)
    assert [g for g in meta.gaps if g.reason == "schema_drift"] == []
    assert meta.n_book == 12


# --- MINOR (run-45): the dark window after a forced reconnect ---

def test_the_outage_after_a_forced_reconnect_is_recorded_as_a_gap(tmp_path):
    """The gap written for a forced reconnect is zero-length by construction (it ends at `now`,
    which is also `_last_event_ts`). The actual dark time — backoff, reconnect, resubscribe,
    waiting for a fresh snapshot — carried NO record, and the reader only flags an unrecorded
    jump above its 60s threshold, so a sub-minute outage was counted as observed time."""
    out = str(tmp_path / "cap.jsonl")
    ticks = iter([100.0, 101.0, 102.0, 103.0, 104.0])
    last = [104.0]

    def clock():
        try:
            last[0] = next(ticks)
        except StopIteration:
            pass
        return last[0]

    attempts = []

    def connect_fn(url):
        attempts.append(url)
        if len(attempts) == 1:
            return _AsyncCM(exc=ConnectionError("simulated connect failure"))
        return _AsyncCM(ws=FakeWS([_trade_frame(100.0)]))

    cfg = RecorderConfig(pairs=["BTC/USD"], out_path=out, capture_id="dark",
                          heartbeat_timeout_s=500.0, backoff_base_s=0.001, backoff_max_s=0.01)
    rec = LiveRecorder(cfg, connect_fn, clock=clock, sleep_fn=_fast_sleep)
    asyncio.run(rec.run(max_iterations=2))
    rec.close(clean=False)

    meta = read_capture(out)
    dark = [g for g in meta.gaps if "dark window" in g.detail]
    assert len(dark) == 1, "the outage between the forced reconnect and the next event must be recorded"
    assert dark[0].duration_s > 0, "a zero-length record is exactly what this fixes"
    assert dark[0].reason == "disconnect"
    assert "'disconnect'" in dark[0].detail, "the originating reason must be named"
    # the ledger stays clean: adjacent, never overlapping, never duplicated
    assert meta.n_overlapping_gap_records == 0
    assert meta.n_duplicate_gap_records == 0
    # and the dark span is now covered by a recorded gap rather than silently counted as observed
    assert any(s <= dark[0].start_ts and dark[0].end_ts <= e for s, e in meta.merged_gap_spans)


def test_no_dark_window_is_invented_when_the_capture_simply_ends(tmp_path):
    """A window that never closed has no event after it, so it lies OUTSIDE the capture's event
    span and explains no time inside it. Writing one at `close()` would append a gap past the
    last event — a claim about time the capture does not cover."""
    out = str(tmp_path / "cap.jsonl")

    def always_fails(url):
        return _AsyncCM(exc=ConnectionError("always down"))

    cfg = RecorderConfig(pairs=["BTC/USD"], out_path=out, capture_id="t",
                          heartbeat_timeout_s=5.0, backoff_base_s=0.001, backoff_max_s=0.01)
    rec = LiveRecorder(cfg, always_fails, sleep_fn=_fast_sleep)
    asyncio.run(rec.run(max_iterations=3))
    rec.close(clean=False)

    meta = read_capture(out)
    assert len(meta.gaps) == 3, "still exactly one gap per failed attempt — none added, none lost"
    assert [g for g in meta.gaps if "dark window" in g.detail] == []


def test_a_healthy_heartbeating_connection_never_trips_the_drift_detector(tmp_path):
    """Non-vacuity: heartbeats ARE events, so a genuinely quiet market resets the counter
    continuously and cannot be mistaken for drift."""
    frames = []
    for _ in range(50):
        frames.append(json.dumps({"channel": "heartbeat"}))
        frames.append(json.dumps({"channel": "status", "data": [{"system": "online"}]}))
    frames.append(_trade_frame(100.0))
    rec, out = _run_one_connection(tmp_path, frames, threshold=10)

    meta = read_capture(out)
    assert [g for g in meta.gaps if g.reason == "schema_drift"] == []
    assert rec.n_schema_mismatch == 0


def test_drift_detector_does_not_fire_without_any_unparseable_frame(tmp_path):
    """Eventless-but-recognized traffic alone is not drift — only frames we positively could not
    read count toward the condition."""
    frames = [json.dumps({"channel": "status", "data": [{"system": "online"}]})] * 30
    frames.append(_trade_frame(100.0))
    rec, out = _run_one_connection(tmp_path, frames, threshold=5)
    meta = read_capture(out)
    assert [g for g in meta.gaps if g.reason == "schema_drift"] == []


def test_nonfinite_prices_on_the_wire_are_schema_mismatches_not_ticks(tmp_path):
    """json accepts bare NaN literals, so a venue error frame can hand us one. It must be
    counted as drift and never written as a tick."""
    rec, out = _run_one_connection(tmp_path, [
        # a valid `type` and a readable `checksum`, so the NaN price is the ONLY defect and the
        # PER-ROW finiteness check is what has to catch it (without `type` this frame was
        # rejected at the frame shape and the check below was never exercised)
        '{"channel":"book","type":"snapshot","data":[{"symbol":"BTC/USD",'
        '"bids":[{"price":NaN,"qty":1.0}],"asks":[{"price":NaN,"qty":1.0}],"checksum":1}]}',
        '{"channel":"trade","data":[{"symbol":"BTC/USD","price":NaN,"qty":1.0,"side":"buy"}]}',
        '{"channel":"trade","data":[{"symbol":"BTC/USD","price":0.0,"qty":1.0,"side":"buy"}]}',
        _trade_frame(100.0),
    ])
    assert rec.n_schema_mismatch == 3
    meta = read_capture(out)   # must parse cleanly: no NaN was ever persisted
    assert meta.n_book == 0
    assert meta.n_trade == 1


def test_empty_data_list_on_a_data_channel_is_drift(tmp_path):
    rec, _ = _run_one_connection(tmp_path, [
        # `type` present so the EMPTY-ROWS check is what rejects this, not the missing-tag check
        json.dumps({"channel": "book", "type": "update", "data": []}),
        json.dumps({"channel": "trade"}),  # missing `data` entirely
        _trade_frame(100.0),
    ])
    assert rec.n_schema_mismatch == 2


# --- the recorder declares its channels, so a dead channel is a refusal ---

def test_recorder_declares_its_subscribed_channels_in_the_capture(tmp_path):
    rec, out = _run_one_connection(tmp_path, [_trade_frame(100.0)])
    meta = read_capture(out)
    assert meta.channels == list(SUBSCRIBED_CHANNELS)
    assert meta.clock_basis == "local_receipt"
    # trade arrived but book NEVER did -> a declared channel recorded zero events -> refusal
    assert meta.missing_channels() == ["book"]
    with pytest.raises(CaptureFormatError, match="recorded ZERO events"):
        meta.assert_complete()


def test_the_recorder_writes_local_receipt_time_not_venue_time(tmp_path):
    """THE CLOCK BASIS: one clock for gap accounting. A trade's venue timestamp is preserved as
    `venue_ts` but `ts` — the field every gap comparison and bar bucket uses — is local."""
    out = str(tmp_path / "cap.jsonl")
    fixed_local = 1_700_000_000.0

    def connect_fn(url):
        # a 2024 venue stamp, wildly different from the local clock below
        return _AsyncCM(ws=FakeWS([_trade_frame(100.0, ts_iso="2024-01-01T00:00:00Z")]))

    cfg = RecorderConfig(pairs=["BTC/USD"], out_path=out, capture_id="t",
                          heartbeat_timeout_s=5.0, backoff_base_s=0.001, backoff_max_s=0.01)
    rec = LiveRecorder(cfg, connect_fn, clock=lambda: fixed_local, sleep_fn=_fast_sleep)
    asyncio.run(rec.run(max_iterations=1))
    rec.close(clean=False)

    lines = [json.loads(l) for l in open(out) if l.strip()]
    trades = [o for o in lines if o.get("type") == "trade"]
    assert len(trades) == 1
    assert trades[0]["ts"] == fixed_local, "gap-accounting ts must be LOCAL receipt time"
    assert trades[0]["venue_ts"] != fixed_local, "the venue stamp is preserved, separately"
    assert abs(trades[0]["venue_ts"] - 1_704_067_200.0) < 86_400  # ~2024-01-01


def test_skewed_venue_timestamps_cannot_produce_overlapping_gap_records(tmp_path):
    """The clock-basis bug's consequence: with trades carrying venue time and gaps carrying
    local time, a burst of stale venue stamps made successive gaps overlap. Now every gap
    endpoint and every event ts come from the same clock, so the ledger stays ordered."""
    out = str(tmp_path / "cap.jsonl")
    ticks = iter([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0])
    last = [109.0]

    def clock():
        try:
            v = next(ticks)
            last[0] = v
            return v
        except StopIteration:
            return last[0]

    def connect_fn(url):
        # stale venue stamps far in the PAST, then a mid-stream drop forcing a gap
        return _AsyncCM(ws=FakeWS([
            _trade_frame(100.0, ts_iso="1999-01-01T00:00:00Z"),
            _trade_frame(101.0, ts_iso="1999-01-01T00:00:00Z"),
            RuntimeError("drop"),
        ]))

    cfg = RecorderConfig(pairs=["BTC/USD"], out_path=out, capture_id="t",
                          heartbeat_timeout_s=500.0, backoff_base_s=0.001, backoff_max_s=0.01)
    rec = LiveRecorder(cfg, connect_fn, clock=clock, sleep_fn=_fast_sleep)
    asyncio.run(rec.run(max_iterations=2))
    rec.close(clean=False)

    meta = read_capture(out)
    # no gap starts before the capture's own first event, and the ledger needs no normalizing
    assert meta.n_overlapping_gap_records == 0
    assert meta.n_duplicate_gap_records == 0
    for g in meta.gaps:
        assert g.start_ts >= meta.start_ts, "a gap keyed on a foreign clock would precede this"


def test_book_rows_missing_required_fields_are_drift(tmp_path):
    """Covers the missing-field shape of drift (a v2 rename that drops `bids`/`asks` entirely,
    or a row that is not an object at all) — distinct from a renamed PRICE field.

    Every book frame here carries a valid `type` and a checksum that is CORRECT for the book the
    row describes. That is deliberate and load-bearing: without `type` these frames were rejected
    at the frame shape and the per-row checks they name were never reached, and with a WRONG
    checksum a relaxed per-row check would still be caught later by the checksum. As written, a
    per-row check that stops rejecting its defect lets the row through to a book that reconciles
    cleanly — so the mismatch count drops (or, for the missing symbol, a book event is written),
    and this test goes red.
    """
    rec, out = _run_one_connection(tmp_path, [
        # `bids` missing entirely; checksum 2906427990 == crc32(b"1010") is correct for the
        # asks-only book this row WOULD produce if the missing side were tolerated as empty
        json.dumps({"channel": "book", "type": "snapshot", "data": [
            {"symbol": "BTC/USD", "asks": [{"price": "1.0", "qty": "1.0"}],
             "checksum": 2906427990}]}),
        # no `symbol`; everything else is valid and the checksum is correct for this book, so if
        # the symbol check were removed the row would be applied and would EMIT
        json.dumps({"channel": "book", "type": "snapshot",
                     "data": [_row_with(symbol=_MISSING)]}),
        json.dumps({"channel": "book", "type": "snapshot", "data": ["not-an-object"]}),
        json.dumps({"channel": "trade", "data": [{"symbol": "BTC/USD"}]}),  # no price
        _trade_frame(100.0),
    ])
    assert rec.n_schema_mismatch == 4
    meta = read_capture(out)
    assert meta.n_book == 0, "no book event may be written from a row we could not read"
    assert [g for g in meta.gaps if "checksum mismatch" in g.detail] == [], (
        "these are PARSE rejections — none of them may reach the book and desync it")


# --- D68/TASK_910 scope 1: the no-order-path invariant, pinned as mechanism not prose --------
#
# Run 44's demonstrated hole: `test_declared_pairs.py`'s coupling test filters
# `sent` to `frame.get("method") == "subscribe"` BEFORE counting, so
# `len(subscribed) == 2` places no constraint on the TOTAL frames sent — appending a real
# order-transmission frame to the send loop, or switching `WS_URL` to Kraken's authenticated
# endpoint, both left every existing test GREEN. These tests count ALL frames sent (unfiltered)
# and pin the connect target to the literal public `WS_URL`.

def test_only_the_two_public_subscribe_frames_are_ever_sent(tmp_path):
    """Counts EVERY frame the socket sees via `send()`, unfiltered — not just the ones whose
    `method` happens to equal `"subscribe"`. A future edit that appends any additional frame
    (an unsubscribe, an authenticated call, anything) to the send loop must fail this test."""
    out = str(tmp_path / "cap.jsonl")
    created_ws = []

    def connect_fn(url):
        ws = FakeWS(["__HANG__"])
        created_ws.append(ws)
        return _AsyncCM(ws=ws)

    cfg = RecorderConfig(pairs=["BTC/USD", "ETH/USD"], out_path=out, capture_id="noorder1",
                          heartbeat_timeout_s=0.02, backoff_base_s=0.001, backoff_max_s=0.01)
    rec = LiveRecorder(cfg, connect_fn, sleep_fn=_fast_sleep)
    asyncio.run(rec.run(max_iterations=1))
    rec.close(clean=False)

    assert len(created_ws) == 1
    sent = created_ws[0].sent
    expected = [json.dumps(m) for m in _subscribe_messages(cfg)]
    assert len(expected) == 2
    assert sent == expected, (
        "the socket must send EXACTLY the two public subscribe frames and nothing else, in "
        f"order — got {sent!r}")


def test_connect_target_is_always_the_pinned_public_ws_url(tmp_path):
    """Pins the literal `WS_URL` module constant AND asserts the recorder never hands
    `connect_fn` anything else — across a failed attempt, a successful one, and a
    heartbeat-timeout-forced reconnect."""
    assert WS_URL == "wss://ws.kraken.com/v2"
    out = str(tmp_path / "cap.jsonl")
    connect_urls = []
    attempts = {"n": 0}

    def connect_fn(url):
        connect_urls.append(url)
        attempts["n"] += 1
        if attempts["n"] == 1:
            return _AsyncCM(exc=ConnectionError("simulated connect failure"))
        if attempts["n"] == 2:
            ws = FakeWS(["__HANG__"])  # forces a heartbeat-timeout reconnect
            return _AsyncCM(ws=ws)
        ws = FakeWS([_trade_frame(100.0)])
        return _AsyncCM(ws=ws)

    cfg = RecorderConfig(pairs=["BTC/USD"], out_path=out, capture_id="urlpin",
                          heartbeat_timeout_s=0.02, backoff_base_s=0.001, backoff_max_s=0.01)
    rec = LiveRecorder(cfg, connect_fn, sleep_fn=_fast_sleep)
    asyncio.run(rec.run(max_iterations=3))
    rec.close(clean=False)

    assert connect_urls, "connect_fn was never called"
    assert all(u == WS_URL for u in connect_urls), (
        f"connect_fn must be called with the pinned public WS_URL every time — got {connect_urls!r}")


# --- D68/TASK_910 scope 2: refusal classes re-raise ABOVE the broad transport catch -----------

def test_refusal_classes_propagate_through_the_broad_catch_instead_of_retrying(tmp_path):
    """`NotImplementedError`/`Phase2RefusalError`/`RailViolation` must escape `run()`'s broad
    `except Exception` untouched — never turned into a "disconnect" gap + backoff + retry like
    an ordinary transport failure would be. 902 measured that NARROWING the `except Exception`
    clause itself is illusory here; the fix re-raises the refusal classes ABOVE the broad catch
    instead, which stays exactly as broad as before.

    If any of these classes were instead swallowed and retried, `run()` would exhaust
    `max_iterations` and return normally without ever raising — `pytest.raises` below would then
    fail with "DID NOT RAISE", which is exactly the RED signal this test is built to produce
    when the fix is absent.
    """
    for exc_cls in (NotImplementedError, Phase2RefusalError, RailViolation):
        out = str(tmp_path / f"cap_{exc_cls.__name__}.jsonl")

        def connect_fn(url, _cls=exc_cls):
            ws = FakeWS([_cls("simulated refusal propagating from inside _consume")])
            return _AsyncCM(ws=ws)

        cfg = RecorderConfig(pairs=["BTC/USD"], out_path=out, capture_id="refusal",
                              heartbeat_timeout_s=5.0, backoff_base_s=0.001, backoff_max_s=0.01)
        rec = LiveRecorder(cfg, connect_fn, sleep_fn=_fast_sleep)
        with pytest.raises(exc_cls):
            asyncio.run(rec.run(max_iterations=5))
        rec.close(clean=False)


# --- in-gate defect #2 (run-45) / "terminal reconnect" — the venue's own subscribe-ack symbol
# --- is reconciled against `cfg.pairs` at CONNECTION START (after the first heartbeat cycle),
# --- not days later at `capture.py` read time. A NON-TRANSIENT mismatch (the same offending
# --- symbol recurring across `SUBSCRIBE_ACK_MISMATCH_TERMINAL_ATTEMPTS` independent connections)
# --- must STOP the recorder with a TERMINAL signal rather than reconnect-storming the venue
# --- forever with an identical config producing an identical mismatch every time.

def test_a_transient_single_reconnect_ack_mismatch_forces_reconnect_but_does_not_stop(tmp_path):
    """One connection whose ack names a symbol `cfg.pairs` never declared reconciles at the
    first heartbeat, forces a reconnect (a real `schema_drift` gap, in-run, at connection START —
    not the old read-time-only reconciliation) — but does NOT raise `SubscriptionConfigMismatch`
    on its own: a single occurrence could be a raced/partial ack set, so the recorder keeps going.
    """
    out = str(tmp_path / "cap.jsonl")
    attempts = []

    def connect_fn(url):
        attempts.append(url)
        if len(attempts) == 1:
            ws = FakeWS([_subscribe_ack_frame("book", "XBT/USD"), _heartbeat_frame()])
        else:
            ws = FakeWS([_trade_frame(100.0)])  # exhausts -> RuntimeError -> ordinary gap+stop
        return _AsyncCM(ws=ws)

    cfg = RecorderConfig(pairs=["BTC/USD"], out_path=out, capture_id="t-ack",
                          heartbeat_timeout_s=5.0, backoff_base_s=0.001, backoff_max_s=0.01)
    rec = LiveRecorder(cfg, connect_fn, sleep_fn=_fast_sleep)
    asyncio.run(rec.run(max_iterations=2))
    rec.close(clean=False)

    assert len(attempts) == 2, "the mismatch forced exactly one reconnect, not a process exit"
    meta = read_capture(out)
    assert any(g.reason == "schema_drift" and "subscribe ACK named symbol" in g.detail
               and "TERMINAL" not in g.detail for g in meta.gaps), (
        "the in-run reconciliation wrote its own gap, distinct from an old read-time-only check")


def test_the_same_ack_mismatch_recurring_across_connections_is_TERMINAL_and_stops_the_recorder(
        tmp_path):
    """Kraken's legacy `XBT/USD` spelling against a config declaring `BTC/USD`: the SAME
    misconfiguration reconciles false on every single reconnect, because re-subscribing with an
    unchanged config gets back the unchanged venue spelling every time. After
    `SUBSCRIBE_ACK_MISMATCH_TERMINAL_ATTEMPTS` consecutive connections all showing the identical
    offending symbol, the recorder must STOP (raise `SubscriptionConfigMismatch`) rather than
    reconnect forever — and the TERMINAL outcome must be written into the capture BEFORE the
    raise, so a reader who opens the file (even mid-storm) sees why, without waiting days for a
    read-time-only reconciliation that this defect's whole point is to pre-empt.
    """
    out = str(tmp_path / "cap.jsonl")
    attempts = []

    def connect_fn(url):
        attempts.append(url)
        ws = FakeWS([_subscribe_ack_frame("book", "XBT/USD"), _heartbeat_frame(), "__HANG__"])
        return _AsyncCM(ws=ws)

    cfg = RecorderConfig(pairs=["BTC/USD"], out_path=out, capture_id="t-terminal",
                          heartbeat_timeout_s=5.0, backoff_base_s=0.001, backoff_max_s=0.01)
    rec = LiveRecorder(cfg, connect_fn, sleep_fn=_fast_sleep)
    with pytest.raises(SubscriptionConfigMismatch, match="TERMINAL"):
        asyncio.run(rec.run(max_iterations=SUBSCRIBE_ACK_MISMATCH_TERMINAL_ATTEMPTS + 5))
    rec.close(clean=False)

    assert len(attempts) == SUBSCRIBE_ACK_MISMATCH_TERMINAL_ATTEMPTS, (
        "stopped exactly at the terminal threshold — not one attempt earlier, not a storm past it")
    meta = read_capture(out)
    terminal_gaps = [g for g in meta.gaps if g.reason == "schema_drift" and "TERMINAL" in g.detail]
    assert len(terminal_gaps) == 1
    assert "XBT/USD" in terminal_gaps[0].detail
    assert str(SUBSCRIBE_ACK_MISMATCH_TERMINAL_ATTEMPTS) in terminal_gaps[0].detail


# --- run-49: the `clock_jump` path had NO test at all, and that is where the false premise grew --
# Nothing in this suite drove `recorder.py`'s clock-jump branch, so nothing ever observed that the
# gap it writes ends LATER than the event of the very frame that terminated the jump. The reader
# then flagged that event as "strictly inside a recorded gap" and refused the whole capture, and
# the refusal's own text asserted the shape "is not one this recorder can produce under a monotonic
# clock" — false, and produced 584 times in the 3.586-day live capture.

class _MicroClock:
    """`time.monotonic()` at microsecond resolution: every READ advances by `tick`.

    The point of the tick is that it makes a SECOND clock sample inside one code path observably
    later than the first — which is exactly the defect. A test clock that returns a constant
    cannot see this, and a wall clock's tick is not reproducible.
    """

    def __init__(self, start=1_000_000.0, tick=12.159e-6):
        self.t = start
        self.tick = tick
        self.pending_jump = 0.0
        self.reads = 0

    def __call__(self):
        self.reads += 1
        self.t += self.tick + self.pending_jump
        self.pending_jump = 0.0
        return self.t


class _JumpingWS(FakeWS):
    """Arms a forward clock step just before the recorder reads the clock for frame `jump_before`."""

    def __init__(self, recv_items, clock, jump_before, jump_s):
        super().__init__(recv_items)
        self._clock, self._jump_before, self._jump_s = clock, jump_before, jump_s
        self._n = 0

    async def recv(self):
        self._n += 1
        if self._n == self._jump_before:
            self._clock.pending_jump = self._jump_s
        return await super().recv()


def test_a_clock_jump_gap_ends_exactly_at_the_event_that_terminated_it(tmp_path):
    """RED before the fix, by 12.159us.

    `_consume` samples the receive clock once per frame and stamps that frame's events with it.
    `_record_gap` used to sample a SECOND, strictly later clock and write THAT as the gap's
    `end_ts`, so the trade below — delivered by the very frame whose late arrival WAS the jump —
    was stamped microseconds BEFORE the end of its own gap and landed strictly inside it. Under a
    perfectly monotonic clock. Structurally, on every jump.

    The assertion is on the raw ledger, deliberately not on the reader's
    `n_events_inside_recorded_gaps`: the reader carries `GAP_EDGE_CLOCK_SAMPLE_TOLERANCE_S` for
    captures already written by the old behaviour, and would mask this.
    """
    out = str(tmp_path / "cap.jsonl")
    clock = _MicroClock()

    def connect_fn(url):
        return _AsyncCM(ws=_JumpingWS([_trade_frame(100.0), _trade_frame(101.0),
                                        RuntimeError("simulated mid-stream drop")],
                                       clock, jump_before=2, jump_s=10.0))

    cfg = RecorderConfig(pairs=["BTC/USD"], out_path=out, capture_id="jump",
                          heartbeat_timeout_s=5.0, backoff_base_s=0.001, backoff_max_s=0.01)
    rec = LiveRecorder(cfg, connect_fn, clock=clock, sleep_fn=_fast_sleep)
    asyncio.run(rec.run(max_iterations=1))
    rec.close(clean=False)

    rows = [json.loads(line) for line in open(out) if line.strip()]
    trades = [r for r in rows if r["type"] == "trade"]
    gaps = [r for r in rows if r["type"] == "gap"]
    # TASK_913 write-hardening §5: the recorder's live detector now WRITES `emission_silence`
    # (an honest rename — the mechanism measured here always was "no event processed for N
    # seconds", never an NTP-style clock correction). `clock_jump` stays in `GAP_REASONS` only
    # for reading OLD/in-flight captures; this test drives the recorder's actual write path.
    jumps = [g for g in gaps if g["reason"] == "emission_silence"]
    assert len(trades) == 2, "both frames were recorded — the jump must not swallow an event"
    assert len(jumps) == 1, "the clock jump was detected exactly once"
    jump = jumps[0]

    assert jump["start_ts"] == trades[0]["ts"], (
        "a gap starts at the last event before it — that endpoint contact is the healthy shape")
    assert jump["end_ts"] == trades[1]["ts"], (
        "and it must END at the event that terminated it, not at a later clock sample: "
        f"end_ts - event_ts = {jump['end_ts'] - trades[1]['ts']:.9f}s")

    # the ledger and the events it is filed against do not contradict each other, at zero
    # tolerance — the property the reader's tolerance exists to fake for OLD captures.
    for ev in rows:
        if ev["type"] not in ("book", "trade"):
            continue
        for g in gaps:
            assert not (g["start_ts"] < ev["ts"] < g["end_ts"]), (
                f"{ev['type']} at {ev['ts']!r} lies strictly inside recorded gap "
                f"[{g['start_ts']!r}, {g['end_ts']!r}] ({g['reason']})")

    meta = read_capture(out)
    assert meta.n_events_inside_recorded_gaps == 0
