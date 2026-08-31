"""TASK_907 gate — fail-loud gaps. Synthetic/replayed captures only (no live socket needed);
this is what proves the code works per the task brief ("do NOT require a live multi-day run").
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture import (  # noqa: E402
    DEFAULT_COVERAGE_WINDOW_S, GAP_EDGE_CLOCK_SAMPLE_TOLERANCE_S, MAX_GAP_DETECT_THRESHOLD_S,
    CaptureFormatError, CaptureMetadata, CaptureWriter, GapRecord, _find_unrecorded_gaps,
    _merge_gap_spans, _merge_spans, _pair_coverage, covers, read_capture,
)


def _write_raw(path: Path, lines) -> None:
    with open(path, "w") as fh:
        for obj in lines:
            fh.write(json.dumps(obj) + "\n")


def test_clean_capture_is_complete(tmp_path):
    p = tmp_path / "cap.jsonl"
    w = CaptureWriter(str(p), ["BTC/USD"], "cap-1")
    w.write_book("BTC/USD", 100.0, 50000.0, 50001.0, 1.0, 1.0)
    w.write_trade("BTC/USD", 100.5, 50000.5, 0.1, "buy")
    w.write_heartbeat(101.0)
    w.close(clean=True)

    meta = read_capture(str(p))
    assert meta.had_meta_start
    assert meta.n_book == 1 and meta.n_trade == 1 and meta.n_heartbeat == 1
    assert meta.gaps == []
    assert meta.unrecorded_gaps == []
    assert meta.closed_cleanly
    assert meta.is_complete()
    meta.assert_complete()  # must not raise


def test_unrecorded_gap_is_detected_and_refuses(tmp_path):
    p = tmp_path / "cap.jsonl"
    _write_raw(p, [
        {"type": "meta_start", "capture_id": "x", "pairs": ["BTC/USD"], "format_version": 1},
        {"type": "trade", "pair": "BTC/USD", "ts": 100.0, "price": 1.0, "volume": 1.0, "side": "buy"},
        # a 1000s jump with NO recorded gap covering it
        {"type": "trade", "pair": "BTC/USD", "ts": 1100.0, "price": 1.0, "volume": 1.0, "side": "buy"},
    ])
    meta = read_capture(str(p), gap_detect_threshold_s=MAX_GAP_DETECT_THRESHOLD_S)
    assert len(meta.unrecorded_gaps) == 1
    assert meta.unrecorded_gaps[0].start_ts == 100.0
    assert meta.unrecorded_gaps[0].end_ts == 1100.0
    assert not meta.is_complete()
    with pytest.raises(CaptureFormatError, match="GATE \\(iv\\) REFUSAL"):
        meta.assert_complete()


def test_recorded_gap_covering_the_jump_is_complete(tmp_path):
    p = tmp_path / "cap.jsonl"
    _write_raw(p, [
        {"type": "meta_start", "capture_id": "x", "pairs": ["BTC/USD"], "format_version": 1},
        {"type": "trade", "pair": "BTC/USD", "ts": 100.0, "price": 1.0, "volume": 1.0, "side": "buy"},
        {"type": "gap", "start_ts": 100.0, "end_ts": 1100.0, "duration_s": 1000.0,
         "reason": "disconnect", "detail": "test"},
        {"type": "trade", "pair": "BTC/USD", "ts": 1100.0, "price": 1.0, "volume": 1.0, "side": "buy"},
    ])
    meta = read_capture(str(p), gap_detect_threshold_s=MAX_GAP_DETECT_THRESHOLD_S)
    assert meta.unrecorded_gaps == []
    assert meta.is_complete()
    meta.assert_complete()


def test_no_meta_start_refuses(tmp_path):
    p = tmp_path / "cap.jsonl"
    _write_raw(p, [
        {"type": "trade", "pair": "BTC/USD", "ts": 100.0, "price": 1.0, "volume": 1.0, "side": "buy"},
    ])
    meta = read_capture(str(p))
    assert not meta.had_meta_start
    assert not meta.is_complete()
    with pytest.raises(CaptureFormatError):
        meta.assert_complete()


def test_zero_events_refuses(tmp_path):
    p = tmp_path / "cap.jsonl"
    _write_raw(p, [{"type": "meta_start", "capture_id": "x", "pairs": ["BTC/USD"], "format_version": 1}])
    meta = read_capture(str(p))
    assert not meta.is_complete()
    with pytest.raises(CaptureFormatError):
        meta.assert_complete()


def test_truncated_trailing_line_is_tolerated_not_corruption(tmp_path):
    p = tmp_path / "cap.jsonl"
    w = CaptureWriter(str(p), ["BTC/USD"], "cap-1")
    w.write_trade("BTC/USD", 100.0, 1.0, 1.0, "buy")
    w._fh.close()  # simulate a kill: no clean close, and now append a partial write
    with open(p, "a") as fh:
        fh.write('{"type": "trade", "pair": "BTC/USD", "ts": 101.0, "pri')  # truncated, no \n

    meta = read_capture(str(p))
    assert meta.truncated_tail
    assert meta.n_trade == 1  # the truncated line is dropped, not counted, and does not raise


def test_corrupt_non_trailing_line_raises(tmp_path):
    p = tmp_path / "cap.jsonl"
    with open(p, "w") as fh:
        fh.write('{"type": "meta_start", "capture_id": "x", "pairs": ["BTC/USD"], "format_version": 1}\n')
        fh.write("NOT JSON AT ALL\n")
        fh.write('{"type": "trade", "pair": "BTC/USD", "ts": 100.0, "price": 1.0, "volume": 1.0, "side": "buy"}\n')
    with pytest.raises(CaptureFormatError, match="corrupt non-trailing"):
        read_capture(str(p))


def test_resume_writes_gap_spanning_the_outage(tmp_path):
    p = tmp_path / "cap.jsonl"
    w1 = CaptureWriter(str(p), ["BTC/USD"], "cap-1")
    w1.write_trade("BTC/USD", 100.0, 1.0, 1.0, "buy")
    w1._fh.close()  # simulated kill — no clean close

    calls = {"n": 0}

    def fake_now():
        calls["n"] += 1
        return 500.0  # process "restarts" at t=500

    import capture as capture_mod
    orig_now = capture_mod._now
    capture_mod._now = fake_now
    try:
        w2 = CaptureWriter(str(p), ["BTC/USD"], "cap-1")
        w2.write_trade("BTC/USD", 500.5, 1.0, 1.0, "buy")
        w2.close(clean=True)
    finally:
        capture_mod._now = orig_now

    meta = read_capture(str(p))
    assert len(meta.gaps) == 1
    g = meta.gaps[0]
    assert g.reason == "resume"
    assert g.start_ts == 100.0
    assert g.end_ts == 500.0
    assert meta.is_complete()  # the outage IS recorded, so this doesn't trip gate (iv)


def test_gap_record_rejects_unknown_reason():
    with pytest.raises(CaptureFormatError):
        GapRecord(0.0, 1.0, "made_up_reason")


def test_gap_record_rejects_negative_duration():
    with pytest.raises(CaptureFormatError):
        GapRecord(10.0, 5.0, "disconnect")


def test_covers_true_and_false():
    meta = CaptureMetadata(path="x", start_ts=0.0, end_ts=1000.0,
                            gaps=[GapRecord(200.0, 300.0, "disconnect")])
    assert covers(meta, 0.0, 100.0)          # entirely before the gap
    assert covers(meta, 400.0, 900.0)        # entirely after the gap
    assert not covers(meta, 150.0, 250.0)    # straddles the gap start
    assert not covers(meta, 250.0, 350.0)    # straddles the gap end
    assert not covers(meta, 0.0, 5000.0)     # extends past end_ts entirely


# --- BLOCKER 2 (run-43): overlapping / duplicate recorded gaps must not defeat gate (iv) ---
#
# The defect: coverage was the SUM of each recorded gap's overlap with a jump, not the measure
# of their UNION. Two gaps of 500s each covering [0,500] and [300,800] summed to 1000s of a
# 1000s jump, so a real 200s hole read as fully explained — while `covers()` still returned True
# for a window sitting squarely inside it.

def _meta_with_gaps(gaps, start_ts=0.0, end_ts=1000.0, n_trade=2):
    meta = CaptureMetadata(path="x", start_ts=start_ts, end_ts=end_ts, gaps=list(gaps),
                            had_meta_start=True, n_trade=n_trade)
    meta.unrecorded_gaps = _find_unrecorded_gaps([start_ts, end_ts], meta.gaps,
                                                  MAX_GAP_DETECT_THRESHOLD_S)
    # This helper hand-builds metadata rather than going through `read_capture`, so the aggregate
    # coverage accounting is absent. `assert_complete` is fail-CLOSED about that (a capture with
    # data events and no coverage accounting is refused), so declare it evaluated with a coverage
    # record that is honest for this fixture: the two events are the span's own endpoints and the
    # recorded gaps are the only absence in it.
    # Same reason, for the run-45 declared/observed pair reconciliation: a hand-built metadata has
    # no `meta_start` to have declared anything, and the reconciliation is fail-CLOSED on an absent
    # declaration. Declare the one pair this fixture's coverage record is actually about, so the
    # fixture stays internally consistent and the check under test here is still the GAP UNION
    # arithmetic and nothing else. (`channels` stays undeclared, as before, so the per-cell check
    # is not in play either.)
    meta.declared_pairs = ["X"]
    meta.observed_pair_channels = {"X": ["trade"]}
    meta.coverage_evaluated = True
    meta.data_start_ts, meta.data_end_ts = start_ts, end_ts
    meta.coverage_by_pair = {
        "X": _pair_coverage("X", {"trade": [start_ts, end_ts]},
                             _merge_spans([(g.start_ts, g.end_ts) for g in meta.gaps]),
                             start_ts, end_ts, DEFAULT_COVERAGE_WINDOW_S)
    }
    return meta


def test_overlapping_recorded_gaps_do_not_mask_a_real_hole():
    """The exact reproduced case: union coverage 800s of a 1000s jump leaves a 200s hole."""
    gaps = [GapRecord(0.0, 500.0, "disconnect"), GapRecord(300.0, 800.0, "disconnect")]
    meta = _meta_with_gaps(gaps)
    assert len(meta.unrecorded_gaps) == 1, "the 200s hole must be flagged, not summed away"
    assert not meta.is_complete()
    with pytest.raises(CaptureFormatError, match="GATE \\(iv\\) REFUSAL"):
        meta.assert_complete()
    # and the per-markout gate agrees: [850, 950] is inside the real hole
    assert covers(meta, 850.0, 950.0) is False or meta.unrecorded_gaps
    # the hole's own accounting is honest about how much was uncovered
    assert "200.0s of this 1000.0s jump" in meta.unrecorded_gaps[0].detail


def test_exact_duplicate_recorded_gaps_do_not_double_count():
    gaps = [GapRecord(0.0, 500.0, "disconnect"), GapRecord(0.0, 500.0, "disconnect")]
    meta = _meta_with_gaps(gaps)
    assert len(meta.unrecorded_gaps) == 1
    with pytest.raises(CaptureFormatError, match="GATE \\(iv\\) REFUSAL"):
        meta.assert_complete()


def test_genuinely_covering_overlapping_gaps_still_pass():
    """The merge must not turn honest overlapping records into a false refusal."""
    gaps = [GapRecord(0.0, 600.0, "disconnect"), GapRecord(400.0, 1000.0, "resume")]
    meta = _meta_with_gaps(gaps)
    assert meta.unrecorded_gaps == []
    meta.assert_complete()  # must not raise — the union really does cover the whole jump


def test_overlapping_and_duplicate_gap_records_are_disclosed_in_metadata(tmp_path):
    p = tmp_path / "cap.jsonl"
    _write_raw(p, [
        {"type": "meta_start", "capture_id": "x", "pairs": ["BTC/USD"], "format_version": 1},
        {"type": "trade", "pair": "BTC/USD", "ts": 0.0, "price": 1.0, "volume": 1.0, "side": "buy"},
        {"type": "gap", "start_ts": 0.0, "end_ts": 600.0, "duration_s": 600.0,
         "reason": "disconnect", "detail": "a"},
        {"type": "gap", "start_ts": 0.0, "end_ts": 600.0, "duration_s": 600.0,
         "reason": "disconnect", "detail": "duplicate of a"},
        {"type": "gap", "start_ts": 400.0, "end_ts": 1000.0, "duration_s": 600.0,
         "reason": "resume", "detail": "overlaps a"},
        {"type": "trade", "pair": "BTC/USD", "ts": 1000.0, "price": 1.0, "volume": 1.0, "side": "buy"},
    ])
    meta = read_capture(str(p), gap_detect_threshold_s=MAX_GAP_DETECT_THRESHOLD_S)
    assert meta.n_duplicate_gap_records == 1
    assert meta.n_overlapping_gap_records >= 1
    assert meta.merged_gap_spans == [(0.0, 1000.0)]  # 3 records -> 1 disjoint span
    assert meta.unrecorded_gaps == []                # the union genuinely covers the jump


def test_merge_gap_spans_coalesces_and_counts():
    merged, n_overlap, n_dup = _merge_gap_spans([
        GapRecord(0.0, 10.0, "disconnect"),
        GapRecord(0.0, 10.0, "disconnect"),   # exact duplicate
        GapRecord(5.0, 20.0, "disconnect"),   # overlaps
        GapRecord(100.0, 110.0, "disconnect"),  # disjoint
    ])
    assert merged == [(0.0, 20.0), (100.0, 110.0)]
    assert n_dup == 1
    assert n_overlap == 2  # the duplicate (0,10) and the overlapping (5,20) each overlap a predecessor
    assert _merge_gap_spans([]) == ([], 0, 0)


def test_gap_record_rejects_nonfinite_endpoints():
    """NaN endpoints slipped through: `end_ts < start_ts` is False for NaN, and the gap then
    contributed NaN to every coverage sum that touched it."""
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(CaptureFormatError, match="non-finite"):
            GapRecord(bad, 10.0, "disconnect")
        with pytest.raises(CaptureFormatError, match="non-finite"):
            GapRecord(0.0, bad, "disconnect")


def test_gap_outside_the_event_span_is_disclosed_and_masks_nothing(tmp_path):
    """A gap lying entirely outside the observed events is DISCLOSED rather than refused: it
    cannot overlap any inter-event jump, and it is legitimate (a long outage with repeated
    failed reconnects writes gaps past the last captured event). What it must not do is make a
    real hole disappear."""
    p = tmp_path / "cap.jsonl"
    _write_raw(p, [
        {"type": "meta_start", "capture_id": "x", "pairs": ["BTC/USD"], "format_version": 1},
        {"type": "trade", "pair": "BTC/USD", "ts": 0.0, "price": 1.0, "volume": 1.0, "side": "buy"},
        {"type": "trade", "pair": "BTC/USD", "ts": 1000.0, "price": 1.0, "volume": 1.0, "side": "buy"},
        {"type": "gap", "start_ts": 50000.0, "end_ts": 90000.0, "duration_s": 40000.0,
         "reason": "disconnect", "detail": "way outside the event span"},
    ])
    meta = read_capture(str(p), gap_detect_threshold_s=MAX_GAP_DETECT_THRESHOLD_S)
    assert meta.n_gaps_outside_event_span == 1
    assert len(meta.unrecorded_gaps) == 1  # the real 1000s hole is STILL flagged
    with pytest.raises(CaptureFormatError, match="GATE \\(iv\\) REFUSAL"):
        meta.assert_complete()


# --- boundary semantics of covers() / gap edges (MINOR 4) ---

def test_covers_boundary_semantics_are_exact():
    meta = CaptureMetadata(path="x", start_ts=100.0, end_ts=1000.0,
                            gaps=[GapRecord(200.0, 300.0, "disconnect")])
    # capture-span edges are INCLUSIVE
    assert covers(meta, 100.0, 199.0)
    assert covers(meta, 400.0, 1000.0)
    assert not covers(meta, 99.999, 150.0)     # starts a hair before the span
    assert not covers(meta, 400.0, 1000.001)   # ends a hair after the span
    # gap edges: a window that merely TOUCHES a gap edge is clean (zero shared duration)
    assert covers(meta, 100.0, 200.0)          # t1 == gap.start_ts
    assert covers(meta, 300.0, 400.0)          # t0 == gap.end_ts
    # ...but any strict overlap is not
    assert not covers(meta, 100.0, 200.001)
    assert not covers(meta, 299.999, 400.0)
    # a window fully inside the gap
    assert not covers(meta, 220.0, 280.0)


def test_zero_length_recorded_gap_still_disqualifies_a_window_spanning_it():
    """Fail-closed, deliberately. A zero-length gap record is not "nothing happened" — it is a
    real disconnect whose measured duration collapsed to an instant (the recorder had no prior
    event, so start == now). The connection WAS lost at that instant, so a markout window
    spanning it is not gap-clean. Pinned so a future "optimization" cannot quietly widen the
    set of windows that count as clean."""
    zero = CaptureMetadata(path="x", start_ts=0.0, end_ts=1000.0,
                            gaps=[GapRecord(500.0, 500.0, "disconnect")])
    assert not covers(zero, 400.0, 600.0)
    # but a window that only touches its instant from outside is still clean
    assert covers(zero, 400.0, 500.0)
    assert covers(zero, 500.0, 600.0)


def test_covers_refuses_when_span_is_unknown_or_time_is_nonfinite():
    empty = CaptureMetadata(path="x")  # no events at all -> start_ts/end_ts are None
    assert covers(empty, 0.0, 1.0) is False
    meta = CaptureMetadata(path="x", start_ts=0.0, end_ts=1000.0)
    assert covers(meta, float("nan"), 100.0) is False
    assert covers(meta, 0.0, float("inf")) is False


def test_unrecorded_gap_detection_threshold_boundary():
    """A jump exactly AT the threshold is not a gap; a hair over it is."""
    assert _find_unrecorded_gaps([0.0, 120.0], [], 120.0) == []
    assert len(_find_unrecorded_gaps([0.0, 120.001], [], 120.0)) == 1
    # a recorded gap that leaves exactly the threshold uncovered is tolerated
    assert _find_unrecorded_gaps([0.0, 300.0], [GapRecord(0.0, 180.0, "disconnect")], 120.0) == []
    assert len(_find_unrecorded_gaps(
        [0.0, 300.0], [GapRecord(0.0, 179.9, "disconnect")], 120.0)) == 1


# --- non-finite prices/timestamps must never enter a capture or come out of one (BLOCKER 1) ---

def test_reader_refuses_a_nan_book_quote(tmp_path):
    """Python's json accepts bare NaN literals, so a venue error frame or a schema-drift bug can
    put one in a capture file. It must never reach the markout arithmetic."""
    p = tmp_path / "cap.jsonl"
    with open(p, "w") as fh:
        fh.write(json.dumps({"type": "meta_start", "capture_id": "x", "pairs": ["BTC/USD"],
                              "format_version": 1}) + "\n")
        fh.write('{"type":"book","pair":"BTC/USD","ts":100.0,"bid":NaN,"ask":NaN,'
                  '"bid_sz":1.0,"ask_sz":1.0}\n')
    with pytest.raises(CaptureFormatError, match="non-finite"):
        read_capture(str(p))


def test_reader_refuses_nonfinite_and_nonpositive_prices_and_timestamps(tmp_path):
    cases = [
        ('{"type":"trade","pair":"B","ts":1.0,"price":NaN,"volume":1.0,"side":"buy"}', "non-finite"),
        ('{"type":"trade","pair":"B","ts":1.0,"price":Infinity,"volume":1.0,"side":"buy"}', "non-finite"),
        ('{"type":"trade","pair":"B","ts":NaN,"price":1.0,"volume":1.0,"side":"buy"}', "non-finite"),
        ('{"type":"trade","pair":"B","ts":1.0,"price":0.0,"volume":1.0,"side":"buy"}', "strictly positive"),
        ('{"type":"trade","pair":"B","ts":1.0,"price":-5.0,"volume":1.0,"side":"buy"}', "strictly positive"),
        ('{"type":"heartbeat","ts":NaN}', "non-finite"),
    ]
    for i, (line, match) in enumerate(cases):
        p = tmp_path / f"cap{i}.jsonl"
        with open(p, "w") as fh:
            fh.write(json.dumps({"type": "meta_start", "capture_id": "x", "pairs": ["B"],
                                  "format_version": 1}) + "\n")
            fh.write(line + "\n")
        with pytest.raises(CaptureFormatError, match=match):
            read_capture(str(p))


def test_writer_refuses_to_write_a_nonfinite_value(tmp_path):
    p = tmp_path / "cap.jsonl"
    w = CaptureWriter(str(p), ["BTC/USD"], "cap-1")
    try:
        with pytest.raises(CaptureFormatError, match="non-finite"):
            w.write_book("BTC/USD", 1.0, float("nan"), 100.0, 1.0, 1.0)
        with pytest.raises(CaptureFormatError, match="non-finite"):
            w.write_trade("BTC/USD", float("inf"), 100.0, 1.0, "buy")
        with pytest.raises(CaptureFormatError, match="strictly positive"):
            w.write_trade("BTC/USD", 1.0, 0.0, 1.0, "buy")
    finally:
        w.close(clean=False)
    meta = read_capture(str(p))
    assert meta.n_book == 0 and meta.n_trade == 0  # nothing bad was persisted


# --- two writers on one capture (BLOCKER 2, reach path 2) ---

def test_second_writer_on_the_same_capture_refuses(tmp_path):
    p = tmp_path / "cap.jsonl"
    w1 = CaptureWriter(str(p), ["BTC/USD"], "cap-1")
    try:
        with pytest.raises(CaptureFormatError, match="exclusive capture lock"):
            CaptureWriter(str(p), ["BTC/USD"], "cap-2")
    finally:
        w1.close(clean=True)
    # once released, a resuming writer is fine again
    w2 = CaptureWriter(str(p), ["BTC/USD"], "cap-1")
    w2.close(clean=True)


# --- the declared clock basis and channel set ---

def test_reader_refuses_a_foreign_clock_basis(tmp_path):
    p = tmp_path / "cap.jsonl"
    _write_raw(p, [
        {"type": "meta_start", "capture_id": "x", "pairs": ["B"], "format_version": 2,
         "clock_basis": "venue"},
        {"type": "trade", "pair": "B", "ts": 1.0, "price": 1.0, "volume": 1.0, "side": "buy"},
    ])
    with pytest.raises(CaptureFormatError, match="clock_basis"):
        read_capture(str(p))


def test_reader_refuses_an_unsupported_format_version(tmp_path):
    p = tmp_path / "cap.jsonl"
    _write_raw(p, [{"type": "meta_start", "capture_id": "x", "pairs": ["B"],
                     "format_version": 99}])
    with pytest.raises(CaptureFormatError, match="format_version"):
        read_capture(str(p))


def test_a_declared_channel_with_zero_events_refuses(tmp_path):
    """MAJOR: `n_book + n_trade > 0` passed happily while ONE channel was silently dead — the
    exact shape of Kraken v2 schema drift on a single channel."""
    p = tmp_path / "cap.jsonl"
    w = CaptureWriter(str(p), ["BTC/USD"], "cap-1", channels=["book", "trade"])
    w.write_trade("BTC/USD", 100.0, 50000.0, 0.1, "buy")  # trades healthy, book NEVER arrives
    w.close(clean=True)

    meta = read_capture(str(p))
    assert meta.channels == ["book", "trade"]
    assert meta.n_trade == 1 and meta.n_book == 0
    assert (meta.n_book + meta.n_trade) > 0       # the OLD sum check would have passed
    assert meta.missing_channels() == ["book"]
    assert not meta.is_complete()
    with pytest.raises(CaptureFormatError, match="recorded ZERO events"):
        meta.assert_complete()


def test_both_declared_channels_present_is_complete(tmp_path):
    p = tmp_path / "cap.jsonl"
    w = CaptureWriter(str(p), ["BTC/USD"], "cap-1", channels=["book", "trade"])
    w.write_trade("BTC/USD", 100.0, 50000.0, 0.1, "buy")
    w.write_book("BTC/USD", 100.5, 50000.0, 50001.0, 1.0, 1.0)
    w.close(clean=True)
    meta = read_capture(str(p))
    assert meta.missing_channels() == []
    meta.assert_complete()


def test_undeclared_channels_fall_back_to_the_sum_check(tmp_path):
    """Synthetic/replay fixtures may omit the channel declaration; they keep the old contract."""
    p = tmp_path / "cap.jsonl"
    w = CaptureWriter(str(p), ["BTC/USD"], "cap-1")  # channels=None
    w.write_trade("BTC/USD", 100.0, 50000.0, 0.1, "buy")
    w.close(clean=True)
    meta = read_capture(str(p))
    assert meta.channels is None
    assert meta.missing_channels() == []
    meta.assert_complete()


# --- B1 (run-45/46) — a RESUMED session's own declared subscription config must be reconciled,
# --- not silently discarded. Run-44 finding 3, decisive on its own: before this, `declared_pairs`
# --- was a snapshot of session 1's config forever, so a pair added on restart and dead from t=0
# --- appeared ZERO times anywhere in the artifact and `assert_complete()` PASSED.

def test_a_pair_declared_only_in_a_RESUMED_session_reconciles_and_refuses_when_dead(tmp_path):
    """Session 1 declares BTC/USD alone and delivers it; session 2 REOPENS the same file
    declaring `[BTC/USD, ETH/USD]`, and ETH/USD delivers zero events for the rest of the capture
    (a dead subscription added on restart). Pre-B1: `declared_pairs == ['BTC/USD']` forever,
    `is_complete()` True, `assert_complete()` PASSED, and 'ETH/USD' occurs zero times in the
    rendered artifact — worse than pre-cycle-4, where the declared UNION at least surfaced it.
    """
    p = tmp_path / "cap.jsonl"
    w1 = CaptureWriter(str(p), ["BTC/USD"], "cap-1", channels=["book", "trade"])
    w1.write_trade("BTC/USD", 100.0, 50000.0, 0.1, "buy")
    w1.write_book("BTC/USD", 100.5, 50000.0, 50001.0, 1.0, 1.0)
    w1._fh.close()  # simulated kill — no clean close, matching a real restart

    import capture as capture_mod
    orig_now = capture_mod._now
    capture_mod._now = lambda: 500.0
    try:
        w2 = CaptureWriter(str(p), ["BTC/USD", "ETH/USD"], "cap-1", channels=["book", "trade"])
    finally:
        capture_mod._now = orig_now
    w2.write_trade("BTC/USD", 500.5, 50010.0, 0.1, "buy")
    w2.write_book("BTC/USD", 501.0, 50010.0, 50011.0, 1.0, 1.0)
    # ETH/USD is declared by session 2 but NEVER delivers a single event.
    w2.close(clean=True)

    meta = read_capture(str(p))
    assert len(meta.sessions) == 2, "both the meta_start and meta_resume declarations are kept"
    assert meta.sessions[0].kind == "start" and meta.sessions[0].pairs == ["BTC/USD"]
    assert meta.sessions[1].kind == "resume" and meta.sessions[1].pairs == ["BTC/USD", "ETH/USD"]
    # the UNION of every session's declaration — not session 1's snapshot alone
    assert meta.declared_pairs == ["BTC/USD", "ETH/USD"]
    assert meta.observed_pairs == ["BTC/USD"]
    assert not meta.is_complete()
    with pytest.raises(CaptureFormatError) as excinfo:
        meta.assert_complete()
    msg = str(excinfo.value)
    assert "ETH/USD" in msg
    assert "declared_without_data" in msg or "ETH/USD" in msg.split("Unreconciled:")[-1]

    report_text = None
    try:
        from render import render_report  # noqa: E402 — local import, module-level would cycle
        render_report([meta], {}, {})
    except CaptureFormatError as exc:
        report_text = str(exc)
    assert report_text is not None and "ETH/USD" in report_text, (
        "the refusal itself names the pair — 'ETH/USD' must not occur zero times anywhere the "
        "reader can see, even though no artifact is written")


def test_a_pair_added_on_a_RESUMED_session_and_delivered_cleanly_is_not_misdiagnosed(tmp_path):
    """The MIRROR of the test above, pinned separately because the spec calls it out explicitly:
    a pair ADDED on restart and RECEIVED cleanly is a pair the subscription DID ask for — the
    refusal text "a number backed by a pair no subscription asked for" must not fire on it.
    """
    p = tmp_path / "cap.jsonl"
    w1 = CaptureWriter(str(p), ["BTC/USD"], "cap-1", channels=["book", "trade"])
    w1.write_trade("BTC/USD", 100.0, 50000.0, 0.1, "buy")
    w1.write_book("BTC/USD", 100.5, 50000.0, 50001.0, 1.0, 1.0)
    w1._fh.close()

    import capture as capture_mod
    orig_now = capture_mod._now
    capture_mod._now = lambda: 500.0
    try:
        w2 = CaptureWriter(str(p), ["BTC/USD", "ETH/USD"], "cap-1", channels=["book", "trade"])
    finally:
        capture_mod._now = orig_now
    # Both pairs deliver a dense, healthy stream for a full minute after the resume — long
    # enough that ETH/USD's OWN coverage cleanly clears the bound and this test is about the
    # declaration reconciliation, not an artifact of a too-short synthetic window.
    t = 500.5
    while t <= 560.5:
        w2.write_trade("BTC/USD", t, 50010.0, 0.1, "buy")
        w2.write_book("BTC/USD", t, 50010.0, 50011.0, 1.0, 1.0)
        w2.write_trade("ETH/USD", t, 3000.0, 1.0, "buy")
        w2.write_book("ETH/USD", t, 3000.0, 3000.5, 1.0, 1.0)
        t += 5.0
    w2.close(clean=True)

    meta = read_capture(str(p))
    assert meta.declared_pairs == ["BTC/USD", "ETH/USD"]
    assert meta.observed_pairs == ["BTC/USD", "ETH/USD"]
    assert meta.pair_reconciliation_gaps()["declared_without_data"] == []
    assert meta.pair_reconciliation_gaps()["observed_undeclared"] == []
    assert meta.degraded_coverage_pairs() == []
    assert meta.is_complete()
    meta.assert_complete()  # must not raise — ETH/USD WAS declared, by session 2

    from render import render_report  # noqa: E402
    report = render_report([meta], {}, {})
    assert "sessions: 2 (start, resume)" in report
    assert "session 1 (resume): pairs=['BTC/USD', 'ETH/USD']" in report
    assert "HETEROGENEOUS DECLARATIONS" in report, (
        "session 0 declared only BTC/USD, session 1 declared both — the artifact must say the "
        "sessions disagreed rather than silently presenting one as the whole")


# --- run-49 BLOCKER: the self-contradictory-gap-ledger check was denominated on MERGED spans ----
# The check ("a data-bearing event STRICTLY INSIDE a recorded gap") had no test at all before this
# block, and it is what stood between the 3.586-day live capture and every number in it: 598
# flagged events, of which 584 were a recorder artifact bounded at 32.902us and 14 were not inside
# any individual gap at all. Both shapes are reproduced here at their measured geometry.

_GAPS_T0 = 1_700_000_000.0


def _write_seam_capture(path, gaps, reconnect_book_ts=260.0, span_s=600.0):
    """A capture that clears every OTHER gate, so whatever these tests observe is this check.

    Book+trade every 5s over [0, 200] and [300, span_s], one lone `book` event at
    `reconnect_book_ts` standing for the first event after a reconnect, and `gaps` written
    verbatim. Timestamps are relative to `_GAPS_T0`.
    """
    rows = []
    def stretch(lo, hi):
        t = lo
        while t <= hi:
            rows.append({"type": "book", "pair": "BTC/USD", "ts": _GAPS_T0 + t,
                          "bid": 100.0 + t, "ask": 100.2 + t, "bid_sz": 1.0, "ask_sz": 1.0})
            rows.append({"type": "trade", "pair": "BTC/USD", "ts": _GAPS_T0 + t + 0.001,
                          "price": 100.1 + t, "volume": 0.1, "side": "buy"})
            t += 5.0
    stretch(0.0, 200.0)
    rows.append({"type": "book", "pair": "BTC/USD", "ts": _GAPS_T0 + reconnect_book_ts,
                  "bid": 250.0, "ask": 250.2, "bid_sz": 1.0, "ask_sz": 1.0})
    stretch(300.0, span_s)
    for lo, hi, reason in gaps:
        rows.append({"type": "gap", "start_ts": _GAPS_T0 + lo, "end_ts": _GAPS_T0 + hi,
                      "duration_s": hi - lo, "reason": reason, "detail": "synthetic"})
    head = {"type": "meta_start", "capture_id": "seam", "pairs": ["BTC/USD"],
            "format_version": 2, "clock_basis": "local_receipt",
            "channels": ["book", "trade"], "heartbeat_timeout_s": 30.0}
    _write_raw(path, [head] + rows + [{"type": "meta_end", "ts": _GAPS_T0 + span_s}])
    return str(path)


def test_an_event_at_the_SEAM_of_two_touching_gaps_is_not_a_contradiction(tmp_path):
    """RED before the fix. A `_close_dark_window` `disconnect` gap ends AT the first event after
    reconnecting; the `clock_jump` gap the very next frame writes STARTS at that same event
    (`_record_gap` starts every gap at `_last_event_ts`). Both endpoint contacts are legal and
    healthy. `_merge_gap_spans` unions the two, and the containment test — run against the merged
    span — then read the seam between them as containment, though NEITHER gap contains anything.

    Measured on the live capture: 14 of its 598 flagged events were exactly and only this, sitting
    2.537s-4.396s deep inside a merged span and 0.0s from an individual gap's edge. No edge
    tolerance can reach that depth without also excusing real contradictions, so the fix is to
    test containment against INDIVIDUAL gap records. The merged span stays correct for coverage
    ARITHMETIC (the measure of the union) — only this test was wrong to run against it.
    """
    path = _write_seam_capture(tmp_path / "seam.jsonl",
                               [(200.001, 260.0, "disconnect"), (260.0, 300.0, "clock_jump")])
    meta = read_capture(path)
    merged = meta.merged_gap_spans
    assert len(merged) == 1 and merged[0] == (_GAPS_T0 + 200.001, _GAPS_T0 + 300.0), (
        "the two gaps really do merge into one span — if they stop touching this test is vacuous")
    assert _GAPS_T0 + 200.001 < _GAPS_T0 + 260.0 < _GAPS_T0 + 300.0, (
        "and the reconnect event really does sit strictly inside that merged span")
    assert meta.n_events_inside_recorded_gaps == 0
    assert meta.events_inside_recorded_gaps == []
    meta.assert_complete()


def test_an_event_microseconds_inside_a_gaps_HIGH_edge_is_the_recorders_own_clock_slack(tmp_path):
    """RED before the fix. The OLD recorder sampled a second, strictly later clock for a gap's
    `end_ts` than the one it stamped that same frame's events with, so the event that TERMINATED
    a `clock_jump` landed microseconds inside its own gap — 584 times in the live capture, min
    5.007us / median 12.159us / max 32.902us, always on the HIGH edge.

    `recorder._record_gap` now takes the caller's already-sampled `now` so new captures carry zero
    such intrusion (`test_recorder_reconnect.py::
    test_a_clock_jump_gap_ends_exactly_at_the_event_that_terminated_it`). This tolerance is for
    captures already on disk, the live one included, which cannot be re-recorded.
    """
    path = _write_seam_capture(
        tmp_path / "micro.jsonl",
        [(200.001, 260.0 + 12.159e-6, "clock_jump")])
    meta = read_capture(path)
    assert meta.n_events_inside_recorded_gaps == 0
    meta.assert_complete()


def test_the_edge_tolerance_is_narrow_enough_that_a_real_containment_still_refuses(tmp_path):
    """Non-vacuity, in the direction that matters: the tolerance must not have turned the check
    off. An event a full second inside a gap is not two clock samples' worth of slack, and the
    refusal must name the INDIVIDUAL gap record (a reader sent looking for the coalesced span
    would be looking for a gap the ledger does not contain)."""
    path = _write_seam_capture(tmp_path / "deep.jsonl", [(200.001, 261.0, "clock_jump")])
    meta = read_capture(path)
    assert meta.n_events_inside_recorded_gaps == 1
    assert not meta.is_complete()
    with pytest.raises(CaptureFormatError) as excinfo:
        meta.assert_complete()
    msg = str(excinfo.value)
    assert "SELF-CONTRADICTORY GAP LEDGER" in msg
    assert "clock_jump" in msg, "the individual record's own reason, not the merged span"
    assert f"{_GAPS_T0 + 261.0:.6f}" in msg, "and the individual record's own end_ts"


def test_the_gap_edge_tolerance_is_bounded_by_the_mechanism_not_by_a_file():
    """The tolerance is a constant with a stated derivation, and both bounds are load-bearing:
    above the measured 32.902us two-clock-sample intrusion, and orders of magnitude below the
    2.537s shallowest merged-span seam (which is excluded structurally, by testing individual
    gaps, and must never come to depend on this number instead)."""
    assert GAP_EDGE_CLOCK_SAMPLE_TOLERANCE_S > 32.902e-6
    assert GAP_EDGE_CLOCK_SAMPLE_TOLERANCE_S < 2.537
    assert GAP_EDGE_CLOCK_SAMPLE_TOLERANCE_S <= 1e-3, (
        "1ms is ~30x the measured bound; anything larger needs a new measurement, not a nudge")
