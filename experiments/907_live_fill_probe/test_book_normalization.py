"""TASK_907 ADDENDUM A (run 45) — the book-channel schema fix, tested against the REAL Kraken
v2 wire.

`wire_fixture_kraken_v2_2026-08-27.jsonl` is 600 real frames off the live public feed
(sha256 pinned below — this file is READ-ONLY and never regenerated). A.1/A.2 measured, on this
exact capture, that Kraken v2 `book` `update` rows are one-sided incremental deltas at arbitrary
depth (not a top-of-book sample) and that the old `bids[0]`/`asks[0]` sampling both rejected
99.93% of the channel AND, on the 4 rows it accepted, recorded a wrong top-of-book. A.3 proved
the venue's own CRC32 `checksum` reconstructs 5,665/5,665 against this fixture when the book is
locally MAINTAINED (snapshot replaces, update applies deltas, trimmed to depth 10) and decoded
with `parse_float=str, parse_int=str` (the checksum is over the wire's STRING forms).

This file discharges A.5's acceptance criteria as permanent tests:
  (A-i)  every book row in the fixture reconciles against the venue checksum, and EVERY emitted
         top-of-book — all 590 rows, not one hand-picked frame — matches an INDEPENDENTLY
         re-derived value.
  (A-ii) the mutation targets (see the run report for the actual apply/run/revert results).
  (A-iii) the trade path is unaffected by the `parse_float=str` decode change.
  (A-iv) covered by the pre-existing `test_credential_free.py`, which still passes unmodified.

THE RUN-45 BLOCKER, AND WHY THE COVERAGE BELOW IS SHAPED THIS WAY
-----------------------------------------------------------------
The first cut of this file checked the emitted top of book at exactly ONE frame (300), which is
an empty-bids row whose single ask delta IS the best ask — the one row shape on which reading a
delta as "the top" is invisible. A faithful re-application of the A.2 defect (take the one-sided
delta as the top, consult the maintained book only for the untouched side) therefore left the
entire suite green while corrupting 482 of the fixture's 590 emitted tops. The checksum tests
could not catch it: they reconcile the maintained STATE, and the defect lived in a SEPARATE
derivation downstream of it. Two things changed in response:

  * `recorder._book_view` now yields the checksum AND the top of book from one ordered structure,
    so the emitted value IS the head of what the venue reconciles — there is no second derivation
    left to disagree;
  * `test_every_emitted_top_of_book_matches_an_independent_rederivation` checks every emitted
    event across the whole fixture against a from-scratch second implementation.
"""
import asyncio
import hashlib
import json
import sys
from collections import OrderedDict
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture import CaptureFormatError, CaptureWriter, DEFAULT_GAP_DETECT_THRESHOLD_S  # noqa: E402
from recorder import (  # noqa: E402
    BOOK_EMISSION_POLICY, SUBSCRIBED_CHANNELS, LiveRecorder, RecorderConfig, _PairBookState,
    _book_view, _parse_book_frame, _parse_frame,
)

FIXTURE_PATH = Path(__file__).resolve().parent / "wire_fixture_kraken_v2_2026-08-27.jsonl"
FIXTURE_SHA256 = "378fa3dc8650df23882954f828cd55be2bbf4cfab63ddec28ed53151592cb831"

#: The venue's OWN declared checksum for fixture frame 7 — the ETH/USD depth-10 `snapshot`. A
#: hardcoded expectation taken from the real wire: nothing in this repo computed it, so it pins
#: the CRC recipe (side order, strip rule, depth) against an external authority rather than
#: against itself. Swapping the asks/bids concatenation order fails this.
FIXTURE_FIRST_SNAPSHOT_IDX = 7
FIXTURE_FIRST_SNAPSHOT_PAIR = "ETH/USD"
FIXTURE_FIRST_SNAPSHOT_CHECKSUM = 3898900712

#: CRC32 of the synthetic one-level-per-side book used by the fail-closed tests below, derived
#: BY HAND from A.3's recipe rather than by asking the code under test what it thinks:
#: asks `100.2`/`2.0` -> `"1002"` + `"20"`, bids `100.0`/`1.0` -> `"1000"` + `"10"`, concatenated
#: asks-then-bids -> `"100220100010"`, and `zlib.crc32(b"100220100010") == 2169449095`.
#: LIMITATION, stated: this literal pins the recipe for a ONE-level book only. Depth trimming,
#: leading-zero handling and delete semantics are pinned by the real-wire fixture tests above —
#: a synthetic unit test of this size cannot reach them.
SYNTHETIC_BOOK_CHECKSUM = 2169449095


def _fixture_lines():
    with open(FIXTURE_PATH, "r") as f:
        return [line for line in f if line.strip()]


def _replay_book_channel(lines, book_depth=10, book_keepalive_s=5.0):
    """Replay every `book`-channel frame through `_parse_book_frame`, decoding each line exactly
    as `LiveRecorder._consume` does (A.3's load-bearing `parse_float=str, parse_int=str`).
    Returns `(emitted, events_by_pair, total_schema_mismatches, all_desyncs, n_book_rows)`, where
    `emitted` is the flat `(frame_idx, event)` list in wire order.
    """
    book_states = {}
    emitted = []
    events_by_pair = {}
    total_mismatches = 0
    all_desyncs = []
    n_book_rows = 0
    for idx, line in enumerate(lines):
        msg = json.loads(line, parse_float=str, parse_int=str)
        if msg.get("channel") != "book":
            continue
        n_book_rows += len(msg.get("data") or [])
        events, mismatches, desyncs, _ = _parse_book_frame(msg, float(idx), book_states,
                                                          book_depth, book_keepalive_s)
        total_mismatches += mismatches
        all_desyncs.extend(desyncs)
        for ev in events:
            emitted.append((idx, ev))
            events_by_pair.setdefault(ev["pair"], []).append((idx, ev))
    return emitted, events_by_pair, total_mismatches, all_desyncs, n_book_rows


def _independently_rederived_tops(lines, depth=10):
    """A SECOND, FROM-SCRATCH implementation of A.3's snapshot + delta + trim algorithm that
    calls NOTHING from `recorder.py`, written in a deliberately different style (plain dicts,
    `max`/`min` over keys rather than a sorted slice) so that a defect cannot be shared with the
    production implementation by construction.

    Returns the ordered list of `(frame_idx, pair, bid, bid_sz, ask, ask_sz)` a correct recorder
    MUST emit for this fixture with the rate limit disabled (`book_keepalive_s=0.0`): every book
    row here is checksum-clean (asserted separately), so exactly the rows leaving both sides of
    the book populated emit, in wire order.

    THIS is what the run-45 BLOCKER needed and did not have: the checksum reconciles the book
    STATE, and until `_book_view` collapsed the two, the emitted top of book was a separate
    derivation that nothing compared against anything.
    """
    books = {}
    expected = []
    for idx, line in enumerate(lines):
        msg = json.loads(line, parse_float=str, parse_int=str)
        if msg.get("channel") != "book":
            continue
        kind = msg["type"]
        for row in msg["data"]:
            pair = row["symbol"]
            if kind == "snapshot":
                books[pair] = {"bids": {}, "asks": {}}
            book = books.get(pair)
            if book is None:
                continue  # an update with no snapshot yet — never applied, never emitted
            for side in ("bids", "asks"):
                for lvl in row.get(side) or []:
                    if float(lvl["qty"]) == 0.0:
                        book[side].pop(lvl["price"], None)
                    else:
                        book[side][lvl["price"]] = lvl["qty"]
                keep = sorted(book[side], key=float, reverse=(side == "bids"))[:depth]
                book[side] = {p: book[side][p] for p in keep}
            if not book["bids"] or not book["asks"]:
                continue
            best_bid = max(book["bids"], key=float)
            best_ask = min(book["asks"], key=float)
            expected.append((idx, pair, float(best_bid), float(book["bids"][best_bid]),
                              float(best_ask), float(book["asks"][best_ask])))
    return expected


# --- fixture integrity ------------------------------------------------------------------------

def test_fixture_is_unmodified():
    """The fixture is READ-ONLY (600 real frames). A changed hash invalidates every
    hand-checked value below — catch that before it produces a silently wrong "pass"."""
    digest = hashlib.sha256(FIXTURE_PATH.read_bytes()).hexdigest()
    assert digest == FIXTURE_SHA256


# --- A.5(A-i): every book row reconciles; a hand-checked frame is correct ----------------------

def test_every_book_row_in_the_fixture_reconciles_against_the_venue_checksum():
    """The re-smoke's pass condition (A.4(5)): `schema_mismatches == 0` on the book channel."""
    lines = _fixture_lines()
    _, _, mismatches, desyncs, n_book_rows = _replay_book_channel(lines)
    # A.1's measured composition: 2 snapshot rows + 588 update rows = 590 book rows total.
    assert n_book_rows == 590, "fixture composition drifted from the measured A.1 count"
    assert mismatches == 0, (
        f"{mismatches} book row(s) failed to reconcile against the venue checksum — every row "
        f"in this fixture is known (A.3) to reconstruct cleanly")
    assert desyncs == [], "no row in this fixture is known to desync; any entry here is a bug"


def test_the_venues_own_declared_checksum_is_reproduced_from_a_real_snapshot():
    """A HARDCODED expectation taken from the real wire (not one this code computed): fixture
    frame 7 is the ETH/USD depth-10 snapshot, and the venue declared checksum 3898900712 over
    it. Reproducing that literal pins the CRC recipe — side order, decimal/leading-zero
    stripping, depth — against an external authority. A circular test that builds its expected
    value by calling `_book_view` cannot: swapping the asks/bids concatenation order leaves such
    a test green, and leaves this one RED."""
    lines = _fixture_lines()
    msg = json.loads(lines[FIXTURE_FIRST_SNAPSHOT_IDX], parse_float=str, parse_int=str)
    row = msg["data"][0]
    assert msg["channel"] == "book" and msg["type"] == "snapshot"
    assert row["symbol"] == FIXTURE_FIRST_SNAPSHOT_PAIR
    assert int(row["checksum"]) == FIXTURE_FIRST_SNAPSHOT_CHECKSUM, (
        "the fixture no longer declares the pinned venue checksum — the literal below would "
        "then be measuring nothing")

    state = _PairBookState(has_snapshot=True)
    state.bids = OrderedDict((lvl["price"], lvl["qty"]) for lvl in row["bids"])
    state.asks = OrderedDict((lvl["price"], lvl["qty"]) for lvl in row["asks"])
    assert _book_view(state).checksum == FIXTURE_FIRST_SNAPSHOT_CHECKSUM


def test_every_emitted_top_of_book_matches_an_independent_rederivation():
    """A.5(A-i), THE RUN-45 BLOCKER'S GUARD: not one hand-picked frame — EVERY emitted event
    across the whole fixture, each checked against `_independently_rederived_tops`, a from-scratch
    second implementation that touches no recorder code.

    The single-frame version of this check was satisfiable by a recorder that reads the
    one-sided delta row as "the top of book" and consults the maintained book only for the
    untouched side, because frame 300 is an empty-bids row whose lone ask delta IS the best ask.
    That mutation corrupts 482 of the 590 emitted values (e.g. frame 11 BTC/USD 79989.2 ->
    79974.8, ~18 bp) and this test fails on the first of them."""
    lines = _fixture_lines()
    emitted, _, mismatches, desyncs, _ = _replay_book_channel(lines, book_keepalive_s=0.0)
    assert mismatches == 0 and desyncs == []

    got = [(idx, ev["pair"], ev["bid"], ev["bid_sz"], ev["ask"], ev["ask_sz"])
           for idx, ev in emitted]
    expected = _independently_rederived_tops(lines)
    # Non-vacuity: this must be checking essentially the whole channel, on both pairs, not a
    # handful of rows that happen to survive some earlier filter.
    assert len(got) > 500, f"only {len(got)} emitted events — this check has gone vacuous"
    assert {p for _, p, *_ in got} == {"BTC/USD", "ETH/USD"}
    assert len(got) == len(expected)
    for a, b in zip(got, expected):
        assert a == b, (
            f"emitted top of book disagrees with the independent re-derivation at frame {a[0]}: "
            f"recorder {a[1:]} vs re-derived {b[1:]}")


def test_hand_checked_top_of_book_at_frame_300():
    """The hand-computed value, kept as a literal anchor under the whole-fixture check above:
    replayed by hand through frame 300, best bid ('79989.2', '0.39994057'), best ask
    ('79989.3', '9.10118370'). `book_keepalive_s=0.0` here makes every checksum-clean row emit,
    so frame 300's event is deterministically present. On its own this assertion is NOT
    sufficient — frame 300 is an empty-bids row, the one shape on which delta-as-top is
    invisible — which is exactly why the test above exists."""
    lines = _fixture_lines()
    _, events_by_pair, mismatches, desyncs, _ = _replay_book_channel(lines, book_keepalive_s=0.0)
    assert mismatches == 0 and desyncs == []
    btc_events = dict(events_by_pair["BTC/USD"])
    assert 300 in btc_events, "frame 300 did not produce a BTC/USD book event"
    ev = btc_events[300]
    assert ev["bid"] == 79989.2
    assert ev["bid_sz"] == 0.39994057
    assert ev["ask"] == 79989.3
    assert ev["ask_sz"] == 9.1011837


# --- A.4(1)/A.4(2): the state machine's fail-closed behaviour ----------------------------------

def test_update_with_no_prior_snapshot_is_a_mismatch_never_applied():
    book_states = {}
    update = {"channel": "book", "type": "update", "data": [{
        "symbol": "BTC/USD", "bids": [{"price": "100.0", "qty": "1.0"}], "asks": [],
        "checksum": 12345}]}
    events, mismatches, desyncs, _ = _parse_book_frame(update, 1.0, book_states, 10, 0.0)
    assert events == [], "an update with no prior snapshot must never emit"
    assert mismatches == 1
    assert desyncs == [], "this is 'no snapshot yet', a distinct reason from checksum desync"
    assert book_states == {}


def test_a_checksum_mismatch_drops_the_book_and_blocks_further_emission():
    book_states = {}
    good_checksum = SYNTHETIC_BOOK_CHECKSUM  # hand-derived literal, never self-computed
    snapshot = {"channel": "book", "type": "snapshot", "data": [{
        "symbol": "BTC/USD", "bids": [{"price": "100.0", "qty": "1.0"}],
        "asks": [{"price": "100.2", "qty": "2.0"}], "checksum": good_checksum}]}
    events, mismatches, desyncs, _ = _parse_book_frame(snapshot, 1.0, book_states, 10, 0.0)
    assert len(events) == 1 and mismatches == 0 and desyncs == []
    assert "BTC/USD" in book_states

    # A declared checksum that does not match what applying this update actually produces.
    bad_update = {"channel": "book", "type": "update", "data": [{
        "symbol": "BTC/USD", "bids": [{"price": "100.1", "qty": "0.5"}], "asks": [],
        "checksum": 1}]}
    events, mismatches, desyncs, _ = _parse_book_frame(bad_update, 2.0, book_states, 10, 0.0)
    assert events == [], "a desynced book must never emit a fabricated top-of-book"
    assert mismatches == 1
    assert len(desyncs) == 1 and desyncs[0][0] == "BTC/USD"
    assert "BTC/USD" not in book_states, "the pair's book must be DROPPED on desync (A.4(2))"

    # No fresh snapshot has arrived: a further update for the SAME pair, even with a checksum
    # that would have been valid against the pre-desync state, must still refuse — the pair
    # stays dark until a real snapshot, not until "a checksum happens to match again".
    another_update = {"channel": "book", "type": "update", "data": [{
        "symbol": "BTC/USD", "bids": [{"price": "100.0", "qty": "1.0"}], "asks": [],
        "checksum": good_checksum}]}
    events, mismatches, desyncs, _ = _parse_book_frame(another_update, 3.0, book_states, 10, 0.0)
    assert events == [], "no snapshot has arrived since the desync — must not emit"
    assert mismatches == 1
    assert "BTC/USD" not in book_states


def test_desync_forces_a_full_reconnect_via_the_live_recorder(tmp_path):
    """End-to-end: a checksum mismatch on the wire must surface as a recorded gap and force
    `LiveRecorder` to reconnect — book state is per-connection (A.4(1)), so this is the only
    fail-closed action available once one pair's book has desynced."""
    from test_recorder_reconnect import _AsyncCM, FakeWS, _fast_sleep  # noqa: E402

    out = str(tmp_path / "cap.jsonl")
    good_checksum = SYNTHETIC_BOOK_CHECKSUM  # hand-derived literal, never self-computed
    snapshot = json.dumps({"channel": "book", "type": "snapshot", "data": [{
        "symbol": "BTC/USD", "bids": [{"price": "100.0", "qty": "1.0"}],
        "asks": [{"price": "100.2", "qty": "2.0"}], "checksum": good_checksum}]})
    bad_update = json.dumps({"channel": "book", "type": "update", "data": [{
        "symbol": "BTC/USD", "bids": [{"price": "100.1", "qty": "0.5"}], "asks": [],
        "checksum": 1}]})

    def connect_fn(url):
        return _AsyncCM(ws=FakeWS([snapshot, bad_update]))

    cfg = RecorderConfig(pairs=["BTC/USD"], out_path=out, capture_id="desync",
                          heartbeat_timeout_s=5.0, backoff_base_s=0.001, backoff_max_s=0.01)
    rec = LiveRecorder(cfg, connect_fn, sleep_fn=_fast_sleep)
    asyncio.run(rec.run(max_iterations=1))
    rec.close(clean=False)

    assert rec.n_schema_mismatch >= 1
    lines = [json.loads(l) for l in open(out) if l.strip()]
    gaps = [o for o in lines if o.get("type") == "gap"]
    assert len(gaps) == 1
    assert "book checksum mismatch" in gaps[0]["detail"]
    books = [o for o in lines if o.get("type") == "book"]
    assert len(books) == 1, "only the good snapshot's book event, nothing from the bad update"


# --- A.4(3): book_depth < 10 refused loudly at construction -------------------------------------

def test_book_depth_below_ten_is_refused_at_construction():
    with pytest.raises(ValueError, match="checksum"):
        RecorderConfig(pairs=["BTC/USD"], out_path="x", capture_id="y", book_depth=9)
    RecorderConfig(pairs=["BTC/USD"], out_path="x", capture_id="y", book_depth=10)  # boundary OK


# --- A.4(4): emission policy + keepalive are bounded, pinned, and declared ---------------------

def test_book_keepalive_is_well_under_the_gap_detect_threshold():
    """In the spirit of the existing coverage-window < gap-detect-threshold pin
    (`test_coverage_aggregate.py::test_the_two_knobs_are_coupled_to_the_bar_interval`): if
    `book_keepalive_s` ever approached `gap_detect_threshold_s` (default 60s), a quiet-but-alive
    book would read as a dark socket and gate (iv) would misfire."""
    default_keepalive = RecorderConfig(pairs=[], out_path="", capture_id="").book_keepalive_s
    assert default_keepalive < DEFAULT_GAP_DETECT_THRESHOLD_S
    assert default_keepalive <= DEFAULT_GAP_DETECT_THRESHOLD_S / 5.0, (
        "book_keepalive_s must sit WELL under the gap-detect threshold, not merely under it")


def test_recorder_declares_book_emission_policy_and_keepalive_into_meta_start(tmp_path):
    """A.4(4): an undeclared emission policy would be a fourth unreconciled declaration — the
    artifact must be able to state the cadence floor its book numbers were measured under."""
    out = str(tmp_path / "cap.jsonl")

    def connect_fn(url):  # pragma: no cover — never invoked; construction only
        raise AssertionError("connect_fn should not be called")

    cfg = RecorderConfig(pairs=["BTC/USD"], out_path=out, capture_id="capB",
                          book_keepalive_s=3.5)
    rec = LiveRecorder(cfg, connect_fn)
    rec.close(clean=False)

    with open(out) as f:
        first = json.loads(f.readline())
    assert first["type"] == "meta_start"
    assert first["book_emission_policy"] == BOOK_EMISSION_POLICY
    assert first["book_keepalive_s"] == 3.5


def test_capture_writer_writes_the_book_fields_only_when_given(tmp_path):
    """Additive-only: a caller that never passes the new kwargs gets a `meta_start` with no
    trace of them — no behavior change for any other writer of this format."""
    out = str(tmp_path / "cap.jsonl")
    writer = CaptureWriter(out, ["BTC/USD"], "capC", channels=list(SUBSCRIBED_CHANNELS))
    writer.close(clean=True)
    with open(out) as f:
        first = json.loads(f.readline())
    assert "book_emission_policy" not in first
    assert "book_keepalive_s" not in first


# --- A.4(4), the READ end: a declaration nobody reads is not a declaration --------------------
#
# The first cut wrote both keys into `meta_start` and stopped there: `CaptureMetadata` had no
# field for them, `read_capture` never looked, and `render.py` had zero hits. A.4(4)'s stated
# purpose — "so the artifact states the cadence floor the numbers were measured under" — was
# therefore unmet, and a healthy book's cadence table read `median=5.0 max=5.0` with no way for
# a reader to tell the market's rate from the recorder's rate limit.

def _capture_with_declared_keepalive(tmp_path, keepalive, name="cap.jsonl", policy=None):
    from capture import read_capture  # noqa: E402 — local import keeps the module header tidy

    out = str(tmp_path / name)
    w = CaptureWriter(out, ["BTC/USD"], "capD", channels=list(SUBSCRIBED_CHANNELS),
                       heartbeat_timeout_s=30.0,
                       book_emission_policy=policy if policy is not None else BOOK_EMISSION_POLICY,
                       book_keepalive_s=keepalive)
    w.write_trade("BTC/USD", 10.0, 100.0, 1.0, "buy")
    w.write_book("BTC/USD", 10.0, 100.0, 100.2, 1.0, 1.0)
    w.write_book("BTC/USD", 15.0, 100.1, 100.3, 1.0, 1.0)
    w.close(clean=True)
    return read_capture(out)


def test_the_reader_reads_the_declared_emission_policy_and_keepalive_back(tmp_path):
    meta = _capture_with_declared_keepalive(tmp_path, 5.0)
    assert meta.declared_book_emission_policy == BOOK_EMISSION_POLICY
    assert meta.declared_book_keepalive_s == 5.0


def test_the_artifact_states_the_cadence_floor_next_to_the_cadence_table(tmp_path):
    from render import render_report  # noqa: E402

    meta = _capture_with_declared_keepalive(tmp_path, 5.0)
    report = render_report([meta], {}, {})
    assert f"`{BOOK_EMISSION_POLICY}`" in report
    assert "cadence FLOOR (book_keepalive_s) = 5s" in report
    assert "the RECORDER's rate limit, not a measurement of the market" in report
    # and the floor is stated in the same block as the cadence numbers it qualifies, immediately
    # ahead of the table (the earlier mention of the phrase is the gate-(iii) preamble's index)
    assert report.index("cadence FLOOR") < report.index("**realized data-bearing cadence**")
    assert report.index("**realized data-bearing cadence**") - report.index("cadence FLOOR") < 400


def test_an_undeclared_cadence_floor_is_reported_as_undeclared_not_assumed(tmp_path):
    """Non-vacuity in the other direction: a capture written by something that declares nothing
    must SAY so, not silently render as if the floor were known."""
    from capture import read_capture  # noqa: E402
    from render import render_report  # noqa: E402

    out = str(tmp_path / "undeclared.jsonl")
    w = CaptureWriter(out, ["BTC/USD"], "capE", channels=list(SUBSCRIBED_CHANNELS))
    w.write_trade("BTC/USD", 10.0, 100.0, 1.0, "buy")
    w.write_book("BTC/USD", 10.0, 100.0, 100.2, 1.0, 1.0)
    w.close(clean=True)
    meta = read_capture(out)
    assert meta.declared_book_keepalive_s is None
    assert "cadence FLOOR (book_keepalive_s) = UNDECLARED" in render_report([meta], {}, {})


def test_the_reader_refuses_a_keepalive_at_or_above_its_gap_detect_threshold(tmp_path):
    """The floor must stay under the resolution gate (iv) reasons at. At or above it, the
    recorder's own rate limit can space book events further apart than the threshold at which
    this reader starts calling a silence a gap — "no gap was recorded" would then be evidence of
    nothing. Hand-written `meta_start` because `RecorderConfig` now refuses to construct such a
    recorder at all (the two ends refuse independently)."""
    out = tmp_path / "coarse.jsonl"
    with open(out, "w") as fh:
        fh.write(json.dumps({"type": "meta_start", "capture_id": "c", "pairs": ["BTC/USD"],
                              "format_version": 2, "clock_basis": "local_receipt",
                              "channels": ["book", "trade"],
                              "book_emission_policy": BOOK_EMISSION_POLICY,
                              "book_keepalive_s": DEFAULT_GAP_DETECT_THRESHOLD_S}) + "\n")
        fh.write(json.dumps({"type": "book", "pair": "BTC/USD", "ts": 1.0, "bid": 100.0,
                              "ask": 100.2, "bid_sz": 1.0, "ask_sz": 1.0}) + "\n")
    from capture import read_capture  # noqa: E402
    with pytest.raises(CaptureFormatError, match="gap_detect_threshold_s"):
        read_capture(str(out))
    # just under the threshold is accepted — the refusal is a boundary, not a blanket ban
    meta = _capture_with_declared_keepalive(tmp_path, DEFAULT_GAP_DETECT_THRESHOLD_S - 1.0,
                                             name="justunder.jsonl")
    assert meta.declared_book_keepalive_s == DEFAULT_GAP_DETECT_THRESHOLD_S - 1.0


def test_recorder_config_refuses_a_keepalive_that_would_outrun_the_gap_threshold():
    """`__post_init__` validated `book_depth` but not `book_keepalive_s`, so `book_keepalive_s=90.0`
    constructed happily and DECLARED a cadence floor coarser than the reader's own resolution."""
    with pytest.raises(ValueError, match="gap_detect_threshold_s"):
        RecorderConfig(pairs=["BTC/USD"], out_path="x", capture_id="y", book_keepalive_s=90.0)
    with pytest.raises(ValueError, match="non-negative"):
        RecorderConfig(pairs=["BTC/USD"], out_path="x", capture_id="y", book_keepalive_s=-1.0)
    with pytest.raises(ValueError, match="non-negative"):
        RecorderConfig(pairs=["BTC/USD"], out_path="x", capture_id="y",
                        book_keepalive_s=float("nan"))
    # the boundary and the disabled case both still construct
    RecorderConfig(pairs=["BTC/USD"], out_path="x", capture_id="y",
                    book_keepalive_s=DEFAULT_GAP_DETECT_THRESHOLD_S - 0.001)
    RecorderConfig(pairs=["BTC/USD"], out_path="x", capture_id="y", book_keepalive_s=0.0)


# --- A.5(A-iii): the trade path is unaffected by parse_float=str/parse_int=str -----------------

def test_trade_path_is_byte_unchanged_under_string_wire_values():
    """`_finite_float` already accepts strings; prove the trade parsing path produces an
    identical event whether values arrive as Python floats (the pre-addendum test style) or as
    strings (what `json.loads(text, parse_float=str, parse_int=str)` actually hands it)."""
    trade_float = {"channel": "trade", "data": [
        {"symbol": "BTC/USD", "price": 100.5, "qty": 2.0, "side": "buy"}]}
    trade_str = {"channel": "trade", "data": [
        {"symbol": "BTC/USD", "price": "100.5", "qty": "2.0", "side": "buy"}]}
    parsed_float = _parse_frame(trade_float, now=1.0)
    parsed_str = _parse_frame(trade_str, now=1.0)
    assert parsed_float.schema_mismatches == parsed_str.schema_mismatches == 0
    ev_f, ev_s = dict(parsed_float.events[0]), dict(parsed_str.events[0])
    ev_f.pop("venue_ts"), ev_s.pop("venue_ts")
    assert ev_f == ev_s == {"kind": "trade", "ts": 1.0, "pair": "BTC/USD", "price": 100.5,
                             "volume": 2.0, "side": "buy"}


def test_fixture_replay_through_the_live_recorder_yields_zero_schema_mismatches(tmp_path):
    """The full pipeline, exercising `LiveRecorder._consume`'s OWN decode call (not just direct
    calls into `_parse_book_frame`) — this is what actually proves the `parse_float=str,
    parse_int=str` decode (A.3) is wired into the production entry point, not only available to
    tests that call the parser functions directly. A.4(5)'s re-smoke pass condition."""
    from test_recorder_reconnect import _AsyncCM, FakeWS, _fast_sleep  # noqa: E402

    out = str(tmp_path / "cap.jsonl")
    lines = [line.rstrip("\n") for line in _fixture_lines()]

    def connect_fn(url):
        return _AsyncCM(ws=FakeWS(lines))

    cfg = RecorderConfig(pairs=["BTC/USD", "ETH/USD"], out_path=out, capture_id="e2e",
                          heartbeat_timeout_s=5.0, backoff_base_s=0.001, backoff_max_s=0.01)
    rec = LiveRecorder(cfg, connect_fn, sleep_fn=_fast_sleep)
    asyncio.run(rec.run(max_iterations=1))
    rec.close(clean=False)

    assert rec.n_schema_mismatch == 0
    assert rec.n_json_decode_errors == 0
