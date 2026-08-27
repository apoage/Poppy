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
    SUBSCRIBED_CHANNELS, LiveRecorder, RecorderConfig, _parse_book_frame, _parse_frame,
)

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
