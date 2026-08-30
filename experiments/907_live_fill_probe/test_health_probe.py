"""TASK_912 — capture health probe tests. Fixture captures only (synthesized JSONL, no live
capture, no ssh, no `capture.py` import — the probe is stdlib-only and this file exercises
`analyze_capture` directly, exactly the scenarios TASK_912 Scope 5 names).
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from health_probe import analyze_capture, main, run_once, run_watch  # noqa: E402


def _write_raw(path: Path, lines) -> None:
    with open(path, "w") as fh:
        for obj in lines:
            fh.write(json.dumps(obj) + "\n")


def _healthy_lines(pairs=("BTC/USD", "ETH/USD"), declared=None, with_counters=True,
                   format_version=2):
    """A capture whose DECLARED pair set and whose data-bearing pairs are set INDEPENDENTLY —
    the shape `test_declared_pairs.py` exists to defend, restated here for the probe."""
    meta = {"type": "meta_start", "capture_id": "cap-912", "format_version": format_version,
            "clock_basis": "local_receipt"}
    if declared is not ...:
        meta["pairs"] = list(pairs) if declared is None else declared
    lines = [meta]
    for pair in pairs:
        for ts in (1770.0, 1780.0, 1790.0, 1795.0):
            lines.append({"type": "book", "pair": pair, "ts": ts, "bid": 100.0, "ask": 100.1,
                          "bid_sz": 1.0, "ask_sz": 1.0})
            lines.append({"type": "trade", "pair": pair, "ts": ts + 0.5, "price": 100.05,
                          "volume": 0.1, "side": "buy"})
    lines.append({"type": "heartbeat", "ts": 1799.0})
    if with_counters:
        lines.append({"type": "counters", "ts": 1799.0, "json_decode_errors": 0,
                      "schema_mismatch": 0, "unrecognized_frames": 0})
    return lines


def test_healthy_capture_passes(tmp_path):
    """All pairs' book+trade fresh, no schema mismatches -> PASS."""
    p = tmp_path / "cap.jsonl"
    lines = [
        {"type": "meta_start", "capture_id": "cap-912-healthy",
         "pairs": ["BTC/USD", "ETH/USD"], "format_version": 2, "clock_basis": "local_receipt"},
    ]
    for pair in ("BTC/USD", "ETH/USD"):
        for ts in (1770.0, 1780.0, 1790.0, 1795.0):
            lines.append({"type": "book", "pair": pair, "ts": ts, "bid": 100.0, "ask": 100.1,
                          "bid_sz": 1.0, "ask_sz": 1.0})
            lines.append({"type": "trade", "pair": pair, "ts": ts + 0.5, "price": 100.05,
                          "volume": 0.1, "side": "buy"})
    lines.append({"type": "heartbeat", "ts": 1799.0})
    lines.append({"type": "counters", "ts": 1799.0, "json_decode_errors": 0,
                  "schema_mismatch": 0, "unrecognized_frames": 0})
    _write_raw(p, lines)

    record = analyze_capture(str(p), wall_now=1800.0)
    assert record["capture_id"] == "cap-912-healthy"
    assert record["schema_mismatch_count"] == 0
    assert record["verdict"]["status"] == "PASS"
    for pair in ("BTC/USD", "ETH/USD"):
        rec = record["per_pair"][pair]
        assert rec["n_book"] == 4 and rec["n_trade"] == 4
        assert rec["last_book_age_s"] == pytest.approx(5.0)
        assert rec["last_trade_age_s"] == pytest.approx(4.5)


def test_split_feed_book_death_fails_with_split_feed_reason(tmp_path):
    """Run-46's measured failure, the row's whole point: ONE pair's book feed dies (last book
    event at ts=1000) while trades on that SAME pair keep flowing, and the other pair stays
    fully healthy. The capture's newest event of any type stays fresh (<60s old) throughout, so
    this must FAIL with the split-feed reason — distinct from a stale-everything recorder-dead
    capture, and the reason string must say so.
    """
    p = tmp_path / "cap.jsonl"
    lines = [
        {"type": "meta_start", "capture_id": "cap-912-splitfeed",
         "pairs": ["BTC/USD", "ETH/USD"], "format_version": 2, "clock_basis": "local_receipt"},
    ]
    # BTC/USD: book dies at ts=1000 (last book event); trades on BTC/USD keep flowing right up
    # to t~=1795 — the realistic Kraken failure (a book subscription dying while trades continue).
    for ts in (900.0, 950.0, 1000.0):
        lines.append({"type": "book", "pair": "BTC/USD", "ts": ts, "bid": 100.0, "ask": 100.1,
                      "bid_sz": 1.0, "ask_sz": 1.0})
    for ts in (1000.5, 1400.0, 1700.0, 1795.0):
        lines.append({"type": "trade", "pair": "BTC/USD", "ts": ts, "price": 100.0,
                      "volume": 0.1, "side": "buy"})
    # ETH/USD: fully healthy, both channels fresh.
    for ts in (1780.0, 1790.0, 1795.0):
        lines.append({"type": "book", "pair": "ETH/USD", "ts": ts, "bid": 200.0, "ask": 200.2,
                      "bid_sz": 1.0, "ask_sz": 1.0})
        lines.append({"type": "trade", "pair": "ETH/USD", "ts": ts + 0.1, "price": 200.1,
                      "volume": 0.1, "side": "sell"})
    lines.append({"type": "heartbeat", "ts": 1798.0})
    _write_raw(p, lines)

    record = analyze_capture(str(p), wall_now=1800.0)
    btc = record["per_pair"]["BTC/USD"]
    assert btc["last_book_age_s"] == pytest.approx(800.0)   # 1800 - 1000: stale (> 300s)
    assert btc["last_trade_age_s"] == pytest.approx(5.0)    # trades kept flowing on this pair
    assert record["per_pair"]["ETH/USD"]["last_book_age_s"] == pytest.approx(5.0)  # healthy
    assert record["verdict"]["status"] == "FAIL"
    reasons = " ".join(record["verdict"]["reasons"])
    assert "split-feed" in reasons
    assert "BTC/USD" in reasons
    assert "recorder-dead" not in reasons


def test_recorder_dead_capture_fails_with_recorder_dead_reason(tmp_path):
    """Everything (both pairs, all channels) goes stale together -> FAIL, but with the
    recorder-dead reason, not split-feed. Distinguishing the two is the whole point of the rule
    ordering in TASK_912 Scope 3.
    """
    p = tmp_path / "cap.jsonl"
    lines = [
        {"type": "meta_start", "capture_id": "cap-912-dead",
         "pairs": ["BTC/USD", "ETH/USD"], "format_version": 2, "clock_basis": "local_receipt"},
    ]
    for pair in ("BTC/USD", "ETH/USD"):
        for ts in (900.0, 950.0, 1000.0):
            lines.append({"type": "book", "pair": pair, "ts": ts, "bid": 100.0, "ask": 100.1,
                          "bid_sz": 1.0, "ask_sz": 1.0})
            lines.append({"type": "trade", "pair": pair, "ts": ts + 0.2, "price": 100.0,
                          "volume": 0.1, "side": "buy"})
    lines.append({"type": "heartbeat", "ts": 1000.5})
    _write_raw(p, lines)

    # wall_now is ~4000s past the last event of ANY type -> nothing in this capture is fresh.
    record = analyze_capture(str(p), wall_now=5000.0)
    assert record["verdict"]["status"] == "FAIL"
    reasons = " ".join(record["verdict"]["reasons"])
    assert "recorder-dead" in reasons
    assert "split-feed" not in reasons


def test_torn_trailing_line_is_tolerated_not_a_schema_mismatch(tmp_path):
    """A copy taken mid-append truncates only the LAST line (capture.py's fsync-per-line
    crash-safety guarantee). That must be tolerated silently — must NOT inflate
    schema_mismatch_count, must NOT be counted in events_total, and must NOT crash the probe.
    """
    p = tmp_path / "cap.jsonl"
    lines = [
        {"type": "meta_start", "capture_id": "cap-912-torn", "pairs": ["BTC/USD"],
         "format_version": 2, "clock_basis": "local_receipt"},
        {"type": "book", "pair": "BTC/USD", "ts": 1795.0, "bid": 100.0, "ask": 100.1,
         "bid_sz": 1.0, "ask_sz": 1.0},
        {"type": "trade", "pair": "BTC/USD", "ts": 1795.5, "price": 100.0, "volume": 0.1,
         "side": "buy"},
        {"type": "counters", "ts": 1796.0, "json_decode_errors": 0, "schema_mismatch": 0,
         "unrecognized_frames": 0},
    ]
    with open(p, "w") as fh:
        for obj in lines:
            fh.write(json.dumps(obj) + "\n")
        fh.write('{"type": "trade", "pair": "BTC/USD", "ts": 1799.0, "pri')  # torn, no newline

    record = analyze_capture(str(p), wall_now=1800.0)
    assert record["truncated_tail"] is True
    assert record["events_total"] == 4           # torn line dropped, not counted
    assert record["schema_mismatch_count"] == 0   # no false-positive mismatch from the torn line
    assert record["per_pair"]["BTC/USD"]["n_trade"] == 1
    assert record["verdict"]["status"] == "PASS"


def test_schema_mismatch_count_is_fail_full_stop(tmp_path):
    """A nonzero schema_mismatch_count (from the capture's own cumulative counters record) is a
    hard FAIL regardless of how fresh everything else is."""
    p = tmp_path / "cap.jsonl"
    lines = [
        {"type": "meta_start", "capture_id": "cap-912-mismatch", "pairs": ["BTC/USD"],
         "format_version": 2, "clock_basis": "local_receipt"},
        {"type": "book", "pair": "BTC/USD", "ts": 1795.0, "bid": 100.0, "ask": 100.1,
         "bid_sz": 1.0, "ask_sz": 1.0},
        {"type": "trade", "pair": "BTC/USD", "ts": 1795.5, "price": 100.0, "volume": 0.1,
         "side": "buy"},
        {"type": "counters", "ts": 1796.0, "json_decode_errors": 0, "schema_mismatch": 3,
         "unrecognized_frames": 0},
    ]
    _write_raw(p, lines)

    record = analyze_capture(str(p), wall_now=1800.0)
    assert record["schema_mismatch_count"] == 3
    assert record["verdict"]["status"] == "FAIL"
    assert any("schema_mismatch_count=3" in r for r in record["verdict"]["reasons"])


def test_capture_id_absent_is_none_not_guessed(tmp_path):
    """No meta_start capture_id present -> capture_id is None, never guessed."""
    p = tmp_path / "cap.jsonl"
    lines = [
        {"type": "meta_start", "pairs": ["BTC/USD"], "format_version": 1},
        {"type": "trade", "pair": "BTC/USD", "ts": 1795.0, "price": 100.0, "volume": 0.1,
         "side": "buy"},
    ]
    _write_raw(p, lines)
    record = analyze_capture(str(p), wall_now=1800.0)
    assert record["capture_id"] is None


# ============================================================================================
# Run-47 fix pass. Each test below is a REGRESSION test for a defect two adversarial lenses
# found in the first cut of this probe; each was verified RED against the pre-fix module and
# green after. The family is the same one `test_declared_pairs.py` documents at length:
# silent absence presented as completeness.
# ============================================================================================

# --- BLOCKER: a declared-but-never-observed pair was invisible and yielded PASS -------------

def test_a_declared_pair_with_zero_events_is_visible_and_fails(tmp_path):
    """THE BLOCKER, reproduced exactly as the orchestrator hit it: `meta_start.pairs` declares
    three pairs, BTC and ETH are fresh and healthy, SOL/USD has ZERO events anywhere. Pre-fix,
    `per_pair` was populated LAZILY from observed records only, so SOL/USD was absent from
    `per_pair`, absent from every reason string, and could not FAIL — verdict PASS.

    This is `test_declared_pairs.py::test_a_declared_pair_with_zero_data_events_is_refused` at
    probe granularity, and it must have the same answer: an absent age is not a young age.
    """
    p = tmp_path / "cap.jsonl"
    _write_raw(p, _healthy_lines(pairs=("BTC/USD", "ETH/USD"),
                                 declared=["BTC/USD", "ETH/USD", "SOL/USD"]))

    record = analyze_capture(str(p), wall_now=1800.0)

    # the pair is VISIBLE — it has a row of its own, with zeros and unknown ages, not absence
    assert sorted(record["per_pair"]) == ["BTC/USD", "ETH/USD", "SOL/USD"]
    sol = record["per_pair"]["SOL/USD"]
    assert sol["n_book"] == 0 and sol["n_trade"] == 0
    assert sol["last_book_age_s"] is None and sol["last_trade_age_s"] is None
    assert record["declared_pairs"] == ["BTC/USD", "ETH/USD", "SOL/USD"]

    # and it FAILS, named, with the split-feed reason: the rest of the capture is demonstrably
    # alive (newest event 1s old), so this is one dead subscription, not a dead recorder.
    assert record["verdict"]["status"] == "FAIL"
    reasons = " ".join(record["verdict"]["reasons"])
    assert "SOL/USD" in reasons
    assert "split-feed" in reasons
    assert "recorder-dead" not in reasons


def test_the_declared_seed_is_a_union_not_a_replacement(tmp_path):
    """Seeding from the declared set must not hide a pair that was OBSERVED but never DECLARED
    (`test_declared_pairs.py::test_the_union_is_blind_where_the_declared_set_is_not` — the two
    sets are independent and the probe reports the union of both)."""
    p = tmp_path / "cap.jsonl"
    _write_raw(p, _healthy_lines(pairs=("BTC/USD", "ETH/USD"), declared=["BTC/USD", "SOL/USD"]))

    record = analyze_capture(str(p), wall_now=1800.0)
    assert sorted(record["per_pair"]) == ["BTC/USD", "ETH/USD", "SOL/USD"]
    assert record["per_pair"]["ETH/USD"]["n_book"] == 4      # observed-but-undeclared: kept
    assert record["verdict"]["status"] == "FAIL"             # SOL/USD declared, never observed


def test_a_capture_declaring_no_pair_set_is_refused_not_passed(tmp_path):
    """Fail-CLOSED on a missing declaration, for the same reason `capture.py` refuses it: with
    no declared set, "declared := observed" reconciles perfectly against whatever the file
    happens to contain, so the split-feed detector — the whole reason this row exists — would
    be a completeness claim about nothing. `CaptureWriter` writes `pairs` unconditionally at
    every format version, so this shape was never produced by this recorder."""
    p = tmp_path / "cap.jsonl"
    _write_raw(p, _healthy_lines(declared=...))   # meta_start with no `pairs` field at all

    record = analyze_capture(str(p), wall_now=1800.0)
    assert record["declared_pairs"] is None
    assert record["verdict"]["status"] == "FAIL"
    assert any("DECLARES NO PAIR SET" in r for r in record["verdict"]["reasons"])


def test_a_malformed_pairs_declaration_is_refused_not_iterated(tmp_path):
    """A bare string would be iterated character-by-character into a declared set of single
    letters that no observed pair can ever match (`test_declared_pairs.py` closes the same hole
    in the reader). Refuse the shape; never seed per_pair from it."""
    for bad in ("BTC/USD", [""], [1], [None], {"BTC/USD": 1}):
        p = tmp_path / f"cap_{abs(hash(str(bad)))}.jsonl"
        _write_raw(p, _healthy_lines(declared=bad))
        record = analyze_capture(str(p), wall_now=1800.0)
        assert record["verdict"]["status"] == "FAIL", bad
        assert any("MALFORMED PAIR SET" in r for r in record["verdict"]["reasons"]), bad
        assert "B" not in record["per_pair"], "the declaration must never be iterated per-char"


def test_corrupt_ts_and_pair_fields_are_counted_not_silently_dropped(tmp_path):
    """Pre-fix, a record with a non-numeric `ts` or a missing/non-string `pair` was dropped from
    per-pair accounting while still counting toward `events_total` — the count said the data was
    there and the per-pair table said it was not, with nothing reconciling the two."""
    p = tmp_path / "cap.jsonl"
    lines = _healthy_lines(pairs=("BTC/USD",))
    lines.append({"type": "book", "pair": "BTC/USD", "ts": "not-a-number", "bid": 1.0,
                  "ask": 1.1, "bid_sz": 1.0, "ask_sz": 1.0})
    lines.append({"type": "trade", "ts": 1796.0, "price": 1.0, "volume": 0.1, "side": "buy"})
    lines.append({"type": "heartbeat", "ts": None})
    _write_raw(p, lines)

    record = analyze_capture(str(p), wall_now=1800.0)
    assert record["malformed_record_count"] == 3
    assert record["malformed_records"] == {"book:ts": 1, "trade:pair": 1, "heartbeat:ts": 1}
    assert any("malformed" in r for r in record["verdict"]["reasons"])
    assert record["verdict"]["status"] == "WARN"
    assert record["per_pair"]["BTC/USD"]["n_book"] == 4, "the corrupt book event is NOT counted"


# --- MAJOR: gap_histogram was blind to any outage short of all-channels-at-once -------------

def test_gap_histogram_reads_the_captures_own_gap_records(tmp_path):
    """THE SECOND MAJOR, reproduced. A genuine 150s single-pair outage, written by the recorder
    as a native `gap` record, with heartbeats every 5s throughout. Pre-fix the histogram merged
    every pair and every event type into ONE timeline and took deltas, so the live heartbeat
    kept the merged timeline dense and the reading was {gt_5s: 0, ..., max_gap_s: 4.0} — the
    outage was structurally invisible. The capture's own gap ledger is the authoritative
    instrument (capture.py: gaps are never silently interpolated)."""
    p = tmp_path / "cap.jsonl"
    lines = [{"type": "meta_start", "capture_id": "cap-912-gap", "pairs": ["BTC/USD"],
              "format_version": 2, "clock_basis": "local_receipt"}]
    ts = 1000.0
    while ts <= 1795.0:                       # heartbeats every 5s: the merged timeline is dense
        lines.append({"type": "heartbeat", "ts": ts})
        ts += 5.0
    for t in (1000.0, 1795.0):                # BTC book either side of the outage
        lines.append({"type": "book", "pair": "BTC/USD", "ts": t, "bid": 100.0, "ask": 100.1,
                      "bid_sz": 1.0, "ask_sz": 1.0})
        lines.append({"type": "trade", "pair": "BTC/USD", "ts": t, "price": 100.0,
                      "volume": 0.1, "side": "buy"})
    lines.append({"type": "gap", "start_ts": 1200.0, "end_ts": 1350.0, "duration_s": 150.0,
                  "reason": "disconnect", "detail": "socket closed"})
    lines.append({"type": "counters", "ts": 1799.0, "json_decode_errors": 0,
                  "schema_mismatch": 0, "unrecognized_frames": 0})
    _write_raw(p, lines)

    gh = analyze_capture(str(p), wall_now=1800.0)["gap_histogram"]
    assert gh["gt_5s"] == 1
    assert gh["gt_60s"] == 1
    assert gh["gt_600s"] == 0
    assert gh["max_gap_s"] == pytest.approx(150.0)
    assert gh["max_gap_wall_ts"] == pytest.approx(1350.0)   # spec 2: max gap WITH its timestamp
    assert gh["max_gap_reason"] == "disconnect"
    assert gh["n_gap_records"] == 1


def test_gap_buckets_are_cumulative_and_strict(tmp_path):
    """Bucket edges, pinned against the live capture's shape: 1092 native gap records of which
    431 exceed 5s and 362 exceed 60s — i.e. cumulative (a >60s gap counts in >5s too) and
    STRICTLY greater than the edge."""
    p = tmp_path / "cap.jsonl"
    lines = _healthy_lines(pairs=("BTC/USD",))
    for start, dur, reason in [(1100.0, 5.0, "clock_jump"),      # == 5s: not counted
                               (1200.0, 5.5, "disconnect"),      # >5s only
                               (1300.0, 60.0, "clock_jump"),     # == 60s: >5s only
                               (1400.0, 90.0, "heartbeat_timeout")]:  # >5s and >60s
        lines.append({"type": "gap", "start_ts": start, "end_ts": start + dur,
                      "duration_s": dur, "reason": reason})
    _write_raw(p, lines)

    gh = analyze_capture(str(p), wall_now=1800.0)["gap_histogram"]
    assert (gh["gt_5s"], gh["gt_60s"], gh["gt_600s"]) == (3, 1, 0)
    assert gh["max_gap_s"] == pytest.approx(90.0)
    assert gh["max_gap_reason"] == "heartbeat_timeout"


def test_a_native_gap_over_600s_warns(tmp_path):
    """The WARN rule (TASK_912 Scope 3) now reads the same authoritative instrument."""
    p = tmp_path / "cap.jsonl"
    lines = _healthy_lines(pairs=("BTC/USD",))
    lines.append({"type": "gap", "start_ts": 1000.0, "end_ts": 1700.0, "duration_s": 700.0,
                  "reason": "disconnect"})
    _write_raw(p, lines)

    record = analyze_capture(str(p), wall_now=1800.0)
    assert record["verdict"]["status"] == "WARN"
    assert any("700.0s" in r and "gap" in r for r in record["verdict"]["reasons"])


def test_a_gap_record_with_an_unusable_duration_is_counted_malformed(tmp_path):
    """Fail-closed on the instrument itself: a gap record the probe cannot measure must be
    visible, not dropped into the same silence the merged timeline used to provide."""
    p = tmp_path / "cap.jsonl"
    lines = _healthy_lines(pairs=("BTC/USD",))
    lines.append({"type": "gap", "start_ts": None, "end_ts": None, "reason": "disconnect"})
    _write_raw(p, lines)

    record = analyze_capture(str(p), wall_now=1800.0)
    assert record["malformed_records"] == {"gap:duration_s": 1}
    assert record["gap_histogram"]["n_gap_records"] == 0
    assert record["verdict"]["status"] == "WARN"


# --- MAJOR: an absent counters record read as verified-clean --------------------------------

def test_absent_counters_is_unknown_not_verified_clean(tmp_path):
    """THE THIRD FINDING. Pre-fix, `if schema_mismatch_count:` treated `None` (no counters
    record ever seen) as falsy and emitted the SAME all-clear reason string as a verified
    `schema_mismatch=0`. "We cannot know, and the mechanism that would tell us may itself be
    dead" must never be reported identically to "verified clean"."""
    p = tmp_path / "cap.jsonl"
    _write_raw(p, _healthy_lines(with_counters=False, format_version=2))

    record = analyze_capture(str(p), wall_now=1800.0)
    assert record["schema_mismatch_count"] is None
    assert record["counters_present"] is False
    assert record["format_version"] == 2
    assert record["verdict"]["status"] == "WARN"
    reasons = record["verdict"]["reasons"]
    assert any("counters record ABSENT" in r for r in reasons)
    assert not any("all checks passed" in r for r in reasons)

    # non-vacuity: the SAME capture with a counters record is PASS with the all-clear string
    q = tmp_path / "cap_ok.jsonl"
    _write_raw(q, _healthy_lines(with_counters=True, format_version=2))
    ok = analyze_capture(str(q), wall_now=1800.0)
    assert ok["counters_present"] is True and ok["verdict"]["status"] == "PASS"
    assert any("all checks passed" in r for r in ok["verdict"]["reasons"])
    assert ok["verdict"]["reasons"] != record["verdict"]["reasons"]


def test_format_version_1_absent_counters_is_tolerated_but_never_all_clear(tmp_path):
    """`capture.py`: format_version 1 has no counters record, so its absence there is a shape
    the writer really produces — tolerated (no WARN). It still must not claim the counters were
    verified: unknown is unknown at every format version."""
    p = tmp_path / "cap.jsonl"
    _write_raw(p, _healthy_lines(with_counters=False, format_version=1))

    record = analyze_capture(str(p), wall_now=1800.0)
    assert record["verdict"]["status"] == "PASS"
    reasons = record["verdict"]["reasons"]
    assert any("format_version 1" in r and "not verified" in r for r in reasons)
    assert not any("all checks passed" in r for r in reasons)


# --- MINOR: a log write hiccup used to kill the unattended --watch loop ---------------------

def test_watch_survives_a_log_write_failure(tmp_path, capsys):
    """The append was OUTSIDE the per-iteration try/except, so a transient write failure (disk
    full, permission hiccup) terminated the whole unattended monitoring loop — the run-46
    failure mode this row exists to prevent, one level up. Simulated here by making the log PATH
    a directory, so `open(..., "a")` raises IsADirectoryError."""
    cap = tmp_path / "cap.jsonl"
    _write_raw(cap, _healthy_lines())
    log_dir = tmp_path / "health"
    log_dir.mkdir()
    (log_dir / "cap.health.jsonl").mkdir()      # the log path is now un-openable

    log_path = run_watch(str(cap), None, 0.0, log_dir=str(log_dir), iterations=2)

    assert log_path == log_dir / "cap.health.jsonl"
    err = capsys.readouterr().err
    assert err.count("HEALTH LOG WRITE FAILED") == 2, "loud on every failed write, and it ran on"


# ============================================================================================
# Run-48 fix pass. Three defects, each verified RED against the pre-fix module and green after.
# ============================================================================================

# --- BLOCKER: `schema_mismatch` was the module's ONLY unvalidated field ---------------------

def _counters_lines(value, present=True):
    """A healthy single-pair capture whose counters record carries `value` as schema_mismatch."""
    lines = _healthy_lines(pairs=("BTC/USD",), with_counters=False, format_version=2)
    counters = {"type": "counters", "ts": 1799.0, "json_decode_errors": 0,
                "unrecognized_frames": 0}
    if present:
        counters["schema_mismatch"] = value
    lines.append(counters)
    return lines


def test_a_fractional_schema_mismatch_is_never_truncated_into_a_verified_zero(tmp_path):
    """THE BLOCKER, reproduced exactly as the orchestrator hit it. Pre-fix the field went through
    a bare `int(...)`, so `schema_mismatch: 0.9` truncated to 0 and the verdict was PASS with the
    reason string literally asserting `schema_mismatch verified 0`. A mismatch DID occur and the
    probe claimed it verified zero — the "absence read as health" inversion that Scope 3's
    "FAIL, full stop" rule exists to prevent. Unreadable must read as UNKNOWN, never as clean.
    """
    p = tmp_path / "cap.jsonl"
    _write_raw(p, _counters_lines(0.9))

    record = analyze_capture(str(p), wall_now=1800.0)

    assert record["schema_mismatch_count"] is None, "truncating 0.9 to a verified 0 is the bug"
    assert record["malformed_records"].get("counters:schema_mismatch") == 1
    assert record["verdict"]["status"] == "WARN"
    reasons = record["verdict"]["reasons"]
    assert any("schema_mismatch" in r and "UNREADABLE" in r for r in reasons)
    assert not any("verified 0" in r for r in reasons)
    assert not any("all checks passed" in r for r in reasons)


@pytest.mark.parametrize("bad", ["corrupt", None, True, False, [1], {"n": 1}, -1, 0.9, -2.0,
                                 float("nan"), float("inf")])
def test_an_unreadable_schema_mismatch_never_crashes_and_never_reads_clean(tmp_path, bad):
    """The second facet: `"corrupt"` raised ValueError and `null` raised TypeError, both
    propagating straight out of `analyze_capture` and killing the probe. Every other field in
    this module degrades to "malformed" instead; this one crashed or lied. `bool` is rejected
    too (True would have counted as a mismatch of 1, False as a verified zero)."""
    p = tmp_path / f"cap_{abs(hash(repr(bad)))}.jsonl"
    _write_raw(p, _counters_lines(bad))

    record = analyze_capture(str(p), wall_now=1800.0)   # pre-fix: raises for several of these

    assert record["schema_mismatch_count"] is None, bad
    assert record["malformed_records"].get("counters:schema_mismatch") == 1, bad
    assert record["verdict"]["status"] == "WARN", bad
    assert any("UNREADABLE" in r for r in record["verdict"]["reasons"]), bad
    assert not any("verified 0" in r for r in record["verdict"]["reasons"]), bad


def test_a_counters_record_missing_the_field_is_unknown_not_a_verified_zero(tmp_path):
    """`CaptureWriter.write_counters` writes `schema_mismatch` unconditionally, so a counters
    record without it is not a shape this recorder produces. Pre-fix `.get(..., 0)` supplied a
    verified zero for it anyway. Same answer as every other absence in this module: UNKNOWN."""
    p = tmp_path / "cap.jsonl"
    _write_raw(p, _counters_lines(None, present=False))

    record = analyze_capture(str(p), wall_now=1800.0)
    assert record["schema_mismatch_count"] is None
    assert record["counters_present"] is True, "the record WAS there — only its field was not"
    assert record["verdict"]["status"] == "WARN"
    assert any("UNREADABLE" in r for r in record["verdict"]["reasons"])


def test_the_unreadable_reason_is_distinct_from_the_absent_counters_reason(tmp_path):
    """Three different facts, three different strings: verified 0, counters record ABSENT, and
    counters record present but its value UNREADABLE. Reusing a string collapses them."""
    unreadable = analyze_capture(str(_written(tmp_path, "u.jsonl", _counters_lines("corrupt"))),
                                 wall_now=1800.0)["verdict"]["reasons"]
    absent = analyze_capture(
        str(_written(tmp_path, "a.jsonl",
                     _healthy_lines(pairs=("BTC/USD",), with_counters=False))),
        wall_now=1800.0)["verdict"]["reasons"]
    clean = analyze_capture(str(_written(tmp_path, "c.jsonl", _counters_lines(0))),
                            wall_now=1800.0)["verdict"]["reasons"]

    assert unreadable != absent != clean and unreadable != clean
    assert any("counters record ABSENT" in r for r in absent)
    assert not any("counters record ABSENT" in r for r in unreadable)
    assert any("all checks passed" in r and "verified 0" in r for r in clean)


@pytest.mark.parametrize("good,expected", [(3, 3), (2.0, 2), (1, 1)])
def test_a_genuine_positive_schema_mismatch_is_still_fail_full_stop(tmp_path, good, expected):
    """The guard must not soften the invariant it protects: an int-shaped positive value (a float
    that is an exact integer included) is still Addendum A's FAIL, full stop."""
    p = tmp_path / f"cap_{expected}.jsonl"
    _write_raw(p, _counters_lines(good))

    record = analyze_capture(str(p), wall_now=1800.0)
    assert record["schema_mismatch_count"] == expected
    assert record["verdict"]["status"] == "FAIL"
    assert any(f"schema_mismatch_count={expected}" in r for r in record["verdict"]["reasons"])


def _written(tmp_path, name, lines):
    p = tmp_path / name
    _write_raw(p, lines)
    return p


# --- MAJOR: one-shot mode had no fail-closed floor ------------------------------------------

def _corrupt_capture(tmp_path):
    """A corrupt NON-trailing line — real corruption per capture.py's fsync-per-line guarantee,
    so `_iter_lines_tolerant` raises CaptureParseError rather than tolerating it."""
    p = tmp_path / "cap.jsonl"
    with open(p, "w") as fh:
        fh.write(json.dumps({"type": "meta_start", "capture_id": "cap-912-corrupt",
                             "pairs": ["BTC/USD"], "format_version": 2}) + "\n")
        fh.write("{not json at all\n")
        fh.write(json.dumps({"type": "heartbeat", "ts": 1799.0}) + "\n")
    return p


def test_one_shot_emits_a_structured_fail_instead_of_a_bare_traceback(tmp_path, capsys):
    """THE SECOND FINDING. `run_watch` wraps `_analyze_via_copy` in try/except and logs a FAIL
    record; `run_once` had no such floor, so the SAME malformed input that degrades gracefully
    under `--watch` killed the process with a bare traceback — no JSON on stdout, no `--out`
    file. Acceptance criterion (i) is the one-shot path, so it needs the same floor: loud AND
    structured, never silently swallowed."""
    p = _corrupt_capture(tmp_path)
    out = tmp_path / "result.json"

    record = run_once(str(p), None, str(out))      # pre-fix: CaptureParseError escapes

    assert record["verdict"]["status"] == "FAIL"
    assert any("probe error" in r for r in record["verdict"]["reasons"])
    assert "corrupt non-trailing line" in record["error"]

    cap = capsys.readouterr()
    assert json.loads(cap.out.strip())["verdict"]["status"] == "FAIL", "structured, on stdout"
    assert "PROBE ERROR" in cap.err, "and loud on stderr — not silently swallowed"

    assert out.exists(), "--out must still be written when the analysis fails"
    assert json.loads(out.read_text())["verdict"]["status"] == "FAIL"


def test_one_shot_exits_non_zero_when_the_analysis_fails(tmp_path):
    """A monitoring tool that exits 0 on its own failure is worse than one that crashes."""
    p = _corrupt_capture(tmp_path)
    assert main(["--capture", str(p)]) != 0          # pre-fix: the exception escapes main()

    healthy = _written(tmp_path, "ok.jsonl", _healthy_lines(pairs=("BTC/USD",)))
    assert main(["--capture", str(healthy)]) == 0    # unchanged for a capture that analyses


# --- MAJOR: verdict.reasons truncated at the first FAIL-tier condition -----------------------

def test_warn_tier_reasons_are_reported_alongside_a_fail(tmp_path):
    """THE THIRD FINDING, reproduced. The gap/trade-staleness WARN block was gated behind
    `if status != "FAIL":` while the malformed-records check right below it was not — an
    inconsistency inside `_compute_verdict`. A capture with schema_mismatch=2 (FAIL) that ALSO
    has a native 900s gap and a pair whose trades stopped 900s ago reported ONLY the
    schema-mismatch reason; both other facts sat in `gap_histogram`/`per_pair` with nothing in
    `reasons` pointing at them. Only the STATUS obeys FAIL-dominates-WARN; the reasons list is a
    complete account of what the probe saw."""
    p = tmp_path / "cap.jsonl"
    lines = [
        {"type": "meta_start", "capture_id": "cap-912-multi", "pairs": ["BTC/USD"],
         "format_version": 2, "clock_basis": "local_receipt"},
        {"type": "book", "pair": "BTC/USD", "ts": 1795.0, "bid": 100.0, "ask": 100.1,
         "bid_sz": 1.0, "ask_sz": 1.0},                       # book fresh: not a split feed
        {"type": "trade", "pair": "BTC/USD", "ts": 900.0, "price": 100.0, "volume": 0.1,
         "side": "buy"},                                      # last_trade_age_s = 900 > 600
        {"type": "gap", "start_ts": 0.0, "end_ts": 900.0, "duration_s": 900.0,
         "reason": "disconnect"},                             # max gap 900 > 600, gt_600s = 1
        {"type": "counters", "ts": 1799.0, "json_decode_errors": 0, "schema_mismatch": 2,
         "unrecognized_frames": 0},                           # FAIL, full stop
    ]
    _write_raw(p, lines)

    record = analyze_capture(str(p), wall_now=1800.0)

    # the two WARN-tier facts are really present in the data, not just in the reasons
    assert record["gap_histogram"]["gt_600s"] == 1
    assert record["per_pair"]["BTC/USD"]["last_trade_age_s"] == pytest.approx(900.0)

    assert record["verdict"]["status"] == "FAIL", "FAIL still dominates WARN for the STATUS"
    reasons = record["verdict"]["reasons"]
    assert any("schema_mismatch_count=2" in r for r in reasons)
    assert any("900.0s" in r and "gap" in r for r in reasons), "the >600s gap must be reported"
    assert any("last_trade_age_s > 600" in r for r in reasons), "the stale trades too"
