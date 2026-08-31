"""TASK_907 gate (iv), run-44 BLOCKER — AGGREGATE data-bearing coverage.

The defect these tests pin: `_find_unrecorded_gaps` was a PER-JUMP threshold check with no
cumulative protection, and it was fed `all_ts` — which includes `heartbeat` events. So two
distinct degradations sailed through:

  1. A capture whose book+trade events are spaced just UNDER the threshold for its entire
     duration. Every jump passes; the capture never once observed the market at the resolution
     its estimands are denominated in. `meta.is_complete()` returned True and `render_report`
     produced a full artifact reading `recorded_gaps=0`.
  2. A session where heartbeats keep flowing normally — resetting the recorder's own
     `heartbeat_timeout_s`, so no gap is ever written — while the book/trade subscription
     silently throttles to a fraction of expected density (not to literal zero, so
     `missing_channels()` never fires either).

Three defenses now exist and NONE subsumes another; each test below says which one it exercises.
Every fixture here is built with the real production modules — no monkeypatching, no bypass.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture import (  # noqa: E402
    BAR_INTERVAL_S, DEFAULT_COVERAGE_WINDOW_S, DEFAULT_GAP_DETECT_THRESHOLD_S,
    DEFAULT_MAX_UNCOVERED_FRACTION, MAX_GAP_DETECT_THRESHOLD_S, CaptureFormatError,
    CaptureMetadata, CaptureWriter, _measure_intersection, _merge_spans, read_capture,
)
from estimands import compute_estimand1, compute_estimand2  # noqa: E402
from recorder import BOOK_EMISSION_POLICY, RecorderConfig, SUBSCRIBED_CHANNELS  # noqa: E402
from render import render_report  # noqa: E402
from shadow_engine import BAR_SECONDS, run_shadow  # noqa: E402
import cli  # noqa: E402

T0 = 1_700_000_000.0
TEN_DAYS = 10 * 86400.0
TWO_DAYS = 2 * 86400.0

# The coverage statistic is a FRACTION, so it is scale-free: a 59s cadence leaves the same share
# of time unobserved over two days as over two hundred. The reported BLOCKER case is reproduced at
# its literal ten days; the tests that only need the mechanism use shorter spans so this suite
# stays fast enough to run on every change.


def _write_capture(path, pairs_cadence, span_s=TEN_DAYS, heartbeat_every_s=30.0,
                    gaps=(), t0=T0, channels=("book", "trade"), heartbeat_timeout_s=30.0,
                    counters=None):
    """Write a synthetic capture: `pairs_cadence` maps pair -> data-bearing cadence in seconds
    (or pair -> list of (from_s, to_s, cadence_s) stretches). Heartbeats flow independently at
    `heartbeat_every_s` — the point of several tests below is that they keep flowing.
    """
    rows = []
    for pair, cadence in pairs_cadence.items():
        stretches = cadence if isinstance(cadence, list) else [(0.0, span_s, cadence)]
        for lo, hi, step in stretches:
            t = lo
            while t <= hi:
                rows.append((t0 + t, "book", pair))
                rows.append((t0 + t + 0.001, "trade", pair))
                t += step
    t = 0.0
    while t <= span_s:
        rows.append((t0 + t, "heartbeat", None))
        t += heartbeat_every_s
    for g in gaps:
        rows.append((t0 + g[0], "gap", g))
    rows.sort(key=lambda r: (r[0], r[1]))

    with open(path, "w") as fh:
        meta = {"type": "meta_start", "capture_id": "cov", "pairs": sorted(pairs_cadence),
                "format_version": 2, "clock_basis": "local_receipt",
                "channels": list(channels)}
        if heartbeat_timeout_s is not None:
            meta["heartbeat_timeout_s"] = heartbeat_timeout_s
        fh.write(json.dumps(meta) + "\n")
        for ts, kind, extra in rows:
            if kind == "book":
                fh.write(json.dumps({"type": "book", "pair": extra, "ts": ts, "bid": 50000.0,
                                      "ask": 50001.0, "bid_sz": 1.0, "ask_sz": 1.0}) + "\n")
            elif kind == "trade":
                fh.write(json.dumps({"type": "trade", "pair": extra, "ts": ts, "price": 50000.5,
                                      "volume": 0.1, "side": "buy"}) + "\n")
            elif kind == "heartbeat":
                fh.write(json.dumps({"type": "heartbeat", "ts": ts}) + "\n")
            else:
                lo, hi, reason = extra
                fh.write(json.dumps({"type": "gap", "start_ts": t0 + lo, "end_ts": t0 + hi,
                                      "duration_s": hi - lo, "reason": reason,
                                      "detail": "synthetic"}) + "\n")
        if counters is not None:
            fh.write(json.dumps({"type": "counters", "ts": t0 + span_s,
                                  "json_decode_errors": counters.get("json_decode_errors", 0),
                                  "schema_mismatch": counters.get("schema_mismatch", 0),
                                  "unrecognized_frames": counters.get(
                                      "unrecognized_frames", 0)}) + "\n")
        fh.write(json.dumps({"type": "meta_end", "ts": t0 + span_s}) + "\n")
    return str(path)


# --- the reported BLOCKER, verbatim -------------------------------------------------------------

def test_119s_cadence_for_ten_days_refuses_and_renders_nothing(tmp_path):
    """THE REPORTED BLOCKER. Before the fix this exact capture gave `is_complete() -> True`, a
    full rendered artifact (242 posted, 121 touch-fills, 484 markout samples, "recorded_gaps=0")
    and no disclosure of the degraded cadence anywhere."""
    path = _write_capture(tmp_path / "deg119.jsonl", {"BTC/USD": 119.0})
    meta = read_capture(path)

    # the OLD defense sees nothing: heartbeats keep flowing, so no all-event jump exceeds the
    # threshold. This assertion IS the heartbeat-vs-data-bearing separation.
    assert meta.unrecorded_gaps == [], (
        "the all-event (heartbeat-inclusive) check is expected to be blind here — that blindness "
        "is the defect, and if this ever starts firing the test below stops isolating anything")
    assert meta.missing_channels() == []          # both channels have events, so this is blind too

    # the NEW defenses both fire
    assert meta.unrecorded_data_gaps, "data-bearing jumps of 119s must be flagged"
    assert meta.degraded_coverage_pairs() == ["BTC/USD"]
    assert not meta.is_complete()
    with pytest.raises(CaptureFormatError, match="GATE \\(iv\\) REFUSAL"):
        meta.assert_complete()

    # and no number reaches an artifact
    meta2, results = run_shadow(path)
    e1 = {p: compute_estimand1(r) for p, r in results.items()}
    e2 = {p: compute_estimand2(r) for p, r in results.items()}
    assert results["BTC/USD"].n_touch_fills > 0, "the engine really did produce fills to render"
    with pytest.raises(CaptureFormatError, match="GATE \\(iv\\) REFUSAL"):
        render_report([meta2], e1, e2)


def test_the_119s_case_also_refuses_via_the_aggregate_check_alone(tmp_path):
    """Isolation: with the per-jump checks relaxed past 119s, the AGGREGATE check still refuses.
    (The threshold is clamped for real use; here it is raised only to switch defenses 1 and 2 off
    so defense 3 is provably the one doing the work.)"""
    path = _write_capture(tmp_path / "deg119b.jsonl", {"BTC/USD": 119.0})
    meta = read_capture(path, gap_detect_threshold_s=MAX_GAP_DETECT_THRESHOLD_S)
    meta.unrecorded_data_gaps = []   # switch defense 2 off
    meta.unrecorded_gaps = []        # switch defense 1 off
    assert meta.degraded_coverage_pairs() == ["BTC/USD"]
    with pytest.raises(CaptureFormatError, match="AGGREGATE data-bearing coverage"):
        meta.assert_complete()


# --- the isolating case: sub-threshold cadence, caught by NOTHING but the aggregate check -------

def test_59s_cadence_clears_every_per_jump_check_and_only_the_aggregate_check_refuses(tmp_path):
    """The non-vacuity proof for defense 3, with no mutation at all.

    59s < the 60s per-jump threshold, so defenses 1 AND 2 are genuinely clean — asserted, not
    assumed. A capture that never observed the market faster than once per minute must not back
    minute-scale markouts, and only the cumulative check can say so.
    """
    path = _write_capture(tmp_path / "deg59.jsonl", {"BTC/USD": 59.0}, span_s=TWO_DAYS)
    meta = read_capture(path)

    assert meta.unrecorded_gaps == [], "defense 1 is clean at 59s"
    assert meta.unrecorded_data_gaps == [], "defense 2 is clean at 59s — 59 < 60"
    assert meta.missing_channels() == []

    cov = meta.coverage_by_pair["BTC/USD"]
    # window is +/-15s around each observation, so a 59s cadence leaves ~half the time unobserved
    assert cov.uncovered_fraction > 0.4, cov
    assert meta.degraded_coverage_pairs() == ["BTC/USD"]
    assert not meta.is_complete()
    with pytest.raises(CaptureFormatError, match="AGGREGATE data-bearing coverage"):
        meta.assert_complete()


def test_the_aggregate_check_is_what_refuses_the_59s_capture(tmp_path):
    """Mutation proof, built in: the SAME fixture, with only the declared bound relaxed to 1.0,
    passes. So the refusal above is attributable to the bound and to nothing else — and this is
    also the demonstration that the pre-fix code (no bound at all) let it through."""
    path = _write_capture(tmp_path / "deg59b.jsonl", {"BTC/USD": 59.0}, span_s=TWO_DAYS)
    meta = read_capture(path, max_uncovered_fraction=0.999)
    assert meta.degraded_coverage_pairs() == []
    assert meta.is_complete()
    meta.assert_complete()   # must not raise — nothing but the bound changed


# --- the realistic form: heartbeats healthy, one pair's subscription throttles ------------------

def test_a_throttling_subscription_beside_a_healthy_one_is_caught_per_pair(tmp_path):
    """The realistic degradation named in the report: heartbeats keep flowing normally, both
    channels keep carrying events (so `missing_channels()` is blind), and BTC stays healthy —
    while ETH's density silently collapses part-way through. Pooling would hide it; the check is
    per-pair and denominated over the capture's GLOBAL data span, so a pair that degrades on day
    two is not judged over day one alone."""
    path = _write_capture(tmp_path / "throttle.jsonl", {
        "BTC/USD": 10.0,
        "ETH/USD": [(0.0, 86400.0, 10.0), (86400.0, TWO_DAYS, 59.0)],
    }, span_s=TWO_DAYS)
    meta = read_capture(path)

    assert meta.unrecorded_gaps == []
    assert meta.unrecorded_data_gaps == [], "59s stretches clear the per-jump check"
    assert meta.missing_channels() == []
    assert meta.coverage_by_pair["BTC/USD"].uncovered_fraction == pytest.approx(0.0, abs=1e-6)
    assert meta.degraded_coverage_pairs() == ["ETH/USD"], (
        "the healthy pair must not be dragged down, and the degraded one must not be averaged "
        "away by it")
    with pytest.raises(CaptureFormatError, match="ETH/USD"):
        meta.assert_complete()


# --- non-vacuity in the other direction: a healthy capture must still pass ----------------------

def test_a_healthy_dense_capture_clears_all_three_defenses_and_renders(tmp_path):
    """If this fails, the new gate is simply refusing everything and proves nothing."""
    path = _write_capture(tmp_path / "healthy.jsonl", {"BTC/USD": 1.0}, span_s=1800.0)
    meta = read_capture(path)
    assert meta.unrecorded_gaps == []
    assert meta.unrecorded_data_gaps == []
    assert meta.coverage_by_pair["BTC/USD"].uncovered_fraction == pytest.approx(0.0, abs=1e-6)
    assert meta.is_complete()
    meta.assert_complete()

    meta2, results = run_shadow(path)
    e1 = {p: compute_estimand1(r) for p, r in results.items()}
    e2 = {p: compute_estimand2(r) for p, r in results.items()}
    report = render_report([meta2], e1, e2)
    assert "data-bearing coverage (gate iv, AGGREGATE)" in report


def test_recorded_gaps_leave_the_denominator_rather_than_counting_against_coverage(tmp_path):
    """A RECORDED gap is honest, disclosed absence and markouts across it are already dropped by
    `covers()`, so it must not be double-punished as un-observed time. A dense capture with a
    recorded 2-hour outage is still complete."""
    # FIXTURE CORRECTED (cycle 2, run-45): the gap used to start at 3600.0 while `_write_capture`
    # emits this stretch's last TRADE at 3600.001 — i.e. the ledger claimed the outage began
    # BEFORE the last event it recorded, which is a (small) self-contradiction the
    # `events_inside_recorded_gaps` check correctly refuses. 3600.001 is the shape the production
    # recorder actually writes: `_record_gap` starts every gap at `_last_event_ts`.
    #
    # RUN-49 CORRECTION TO THAT NOTE, because it is where a false premise was baked in. The note
    # above reasons about the gap's START only, and the cycle-2 author generalised from it to
    # "every gap this recorder writes ends AT an event, so containment is a shape it cannot
    # produce" — which went into a shipped refusal string and, three cycles later, made the
    # 3.586-day live capture unanalysable. The START endpoint is exact (`_last_event_ts`). The
    # END endpoint was NOT: `_record_gap` sampled a SECOND, later clock, so on the `clock_jump`
    # path the event that terminated the outage landed microseconds INSIDE its own gap. Fixed at
    # the recorder (`_record_gap(..., now=...)`) and tolerated at the reader for captures already
    # written (`GAP_EDGE_CLOCK_SAMPLE_TOLERANCE_S`). Moving a fixture is the right call only when
    # the fixture is the thing that is wrong; check BOTH endpoints before concluding that.
    path = _write_capture(tmp_path / "gapped.jsonl", {
        "BTC/USD": [(0.0, 3600.0, 1.0), (10800.0, 14400.0, 1.0)],
    }, span_s=14400.0, gaps=[(3600.001, 10800.0, "disconnect")])
    meta = read_capture(path)
    cov = meta.coverage_by_pair["BTC/USD"]
    assert cov.recorded_gap_s == pytest.approx(7200.0, abs=1.0)
    assert cov.accounted_s == pytest.approx(7200.0, abs=2.0)
    assert cov.uncovered_fraction < DEFAULT_MAX_UNCOVERED_FRACTION
    assert meta.unrecorded_data_gaps == [], "the outage IS recorded, so it is not a hole"
    meta.assert_complete()


# --- fail-closed: the aggregate check must not be bypassable ------------------------------------

def test_metadata_that_never_had_coverage_evaluated_is_refused(tmp_path):
    """Fail-CLOSED. A hand-constructed `CaptureMetadata` with data events but no coverage
    accounting must not be waved through — that would be a one-line bypass of the whole check."""
    meta = CaptureMetadata(path="hand-built", had_meta_start=True, n_trade=5, n_book=5,
                            start_ts=0.0, end_ts=100.0)
    assert not meta.coverage_evaluated
    assert not meta.is_complete()
    with pytest.raises(CaptureFormatError, match="coverage was never evaluated"):
        meta.assert_complete()


def test_the_coverage_bound_is_not_reachable_from_the_command_line():
    """The bound is a declared property of the instrument, not an operator dial. A flag that can
    widen a fail-loud bound is a bypass with extra steps."""
    flags = {a.dest for a in cli.build_argparser()._actions}
    assert "coverage_window_s" not in flags
    assert "max_uncovered_fraction" not in flags
    assert "gap_detect_threshold_s" in flags   # this one exists, and is clamped (below)


# --- the two knobs, coupled ---------------------------------------------------------------------

def test_a_gap_detect_threshold_wider_than_the_bar_interval_is_refused(tmp_path):
    """The knob coupling. The dangerous direction is a LOOSE reader threshold: at 120s over 60s
    bars a jump could swallow an entire bar, and every markout inside it, unflagged."""
    path = _write_capture(tmp_path / "k.jsonl", {"BTC/USD": 1.0}, span_s=600.0)
    with pytest.raises(CaptureFormatError, match="exceeds the bar interval"):
        read_capture(path, gap_detect_threshold_s=120.0)
    with pytest.raises(CaptureFormatError, match="exceeds the bar interval"):
        read_capture(path, gap_detect_threshold_s=MAX_GAP_DETECT_THRESHOLD_S + 0.001)
    read_capture(path, gap_detect_threshold_s=MAX_GAP_DETECT_THRESHOLD_S)   # at the ceiling: ok

    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(CaptureFormatError, match="finite positive"):
            read_capture(path, gap_detect_threshold_s=bad)
    with pytest.raises(CaptureFormatError, match="exceeds the bar interval"):
        read_capture(path, coverage_window_s=BAR_INTERVAL_S + 1.0)
    for bad in (1.0, 1.5, -0.1, float("nan")):
        with pytest.raises(CaptureFormatError, match="max_uncovered_fraction"):
            read_capture(path, max_uncovered_fraction=bad)


def test_the_two_knobs_are_coupled_to_the_bar_interval():
    """`capture` may not import `shadow_engine`, so the bar interval is declared in both. Pin
    them equal here: if a future change moves the bar interval, the clamp moves with it instead
    of silently going slack."""
    assert BAR_INTERVAL_S == BAR_SECONDS
    assert DEFAULT_GAP_DETECT_THRESHOLD_S == MAX_GAP_DETECT_THRESHOLD_S == BAR_INTERVAL_S
    assert DEFAULT_COVERAGE_WINDOW_S < BAR_INTERVAL_S, (
        "the coverage window must be TIGHTER than the per-jump threshold, or a cadence sitting "
        "just under the threshold would be fully 'covered' and the aggregate check would be "
        "structurally unable to fire")
    # cross-check against the recorder's independently-configured liveness knob
    assert RecorderConfig(pairs=[], out_path="", capture_id="").heartbeat_timeout_s \
        <= MAX_GAP_DETECT_THRESHOLD_S, (
        "the recorder must promise liveness at least as tightly as the reader checks for it")


def test_the_recorders_heartbeat_timeout_is_declared_and_disclosed(tmp_path):
    path = _write_capture(tmp_path / "hb.jsonl", {"BTC/USD": 1.0}, span_s=600.0,
                           heartbeat_timeout_s=17.0)
    meta = read_capture(path)
    assert meta.declared_heartbeat_timeout_s == 17.0
    report = render_report([meta], {}, {})
    assert "heartbeat_timeout_s = 17s" in report

    path2 = _write_capture(tmp_path / "hb2.jsonl", {"BTC/USD": 1.0}, span_s=600.0,
                            heartbeat_timeout_s=None)
    meta2 = read_capture(path2)
    assert meta2.declared_heartbeat_timeout_s is None
    assert "heartbeat_timeout_s = undeclared" in render_report([meta2], {}, {})


# --- the realized cadence must be ON THE PAGE ---------------------------------------------------

def test_the_realized_data_bearing_cadence_is_disclosed_in_the_artifact(tmp_path):
    """A number derived from a coarse capture must not be presentable as a minute-scale
    measurement. The only way a reader can check that is if the realized cadence is printed."""
    path = _write_capture(tmp_path / "cad.jsonl", {"BTC/USD": 5.0}, span_s=3600.0)
    meta = read_capture(path)
    report = render_report([meta], {}, {})
    assert "realized data-bearing cadence" in report
    assert "| BTC/USD | book |" in report
    assert "| BTC/USD | book+trade |" in report
    assert "heartbeats are EXCLUDED" in report
    # the printed median really is the 5s cadence, not an assumption
    book = [c for c in meta.cadence if c.channel == "book"][0]
    assert book.median_interval_s == pytest.approx(5.0, abs=1e-6)
    assert "5.000s" in report


def test_heartbeats_never_enter_the_cadence_or_coverage_accounting(tmp_path):
    """The heart of the BLOCKER, asserted directly: a flood of heartbeats must not improve either
    statistic by one second. A heartbeat proves the socket is alive, not that the market was
    observed."""
    sparse = _write_capture(tmp_path / "a.jsonl", {"BTC/USD": 59.0}, span_s=7200.0,
                             heartbeat_every_s=1.0)
    sparse_few = _write_capture(tmp_path / "b.jsonl", {"BTC/USD": 59.0}, span_s=7200.0,
                                 heartbeat_every_s=30.0)
    a, b = read_capture(sparse), read_capture(sparse_few)
    assert a.n_heartbeat > 20 * b.n_heartbeat, "the fixtures really do differ in heartbeat rate"
    assert (a.coverage_by_pair["BTC/USD"].covered_s
            == pytest.approx(b.coverage_by_pair["BTC/USD"].covered_s))
    assert [c.median_interval_s for c in a.cadence] == [c.median_interval_s for c in b.cadence]
    for m in (a, b):
        assert m.degraded_coverage_pairs() == ["BTC/USD"]


# --- a data-bearing event belonging to no pair is unattributable --------------------------------

def test_a_book_or_trade_record_with_no_pair_is_refused(tmp_path):
    """Such an event increments the channel counts and the event span while appearing in NO
    pair's coverage accounting — an unattributable contributor to gate (iv)'s arithmetic."""
    for line in (
        '{"type":"book","ts":1.0,"bid":1.0,"ask":2.0,"bid_sz":1.0,"ask_sz":1.0}',
        '{"type":"book","pair":"","ts":1.0,"bid":1.0,"ask":2.0,"bid_sz":1.0,"ask_sz":1.0}',
        '{"type":"trade","ts":1.0,"price":1.0,"volume":1.0,"side":"buy"}',
    ):
        p = tmp_path / "np.jsonl"
        with open(p, "w") as fh:
            fh.write(json.dumps({"type": "meta_start", "capture_id": "x", "pairs": ["B"],
                                  "format_version": 1}) + "\n")
            fh.write(line + "\n")
        with pytest.raises(CaptureFormatError, match="no usable `pair`"):
            read_capture(str(p))


# --- the span arithmetic underneath ------------------------------------------------------------

def test_merge_spans_and_measure_intersection():
    assert _merge_spans([]) == []
    assert _merge_spans([(0.0, 5.0), (3.0, 8.0), (20.0, 25.0)]) == [(0.0, 8.0), (20.0, 25.0)]
    assert _merge_spans([(0.0, 5.0), (5.0, 9.0)]) == [(0.0, 9.0)]   # touching spans merge
    # union-not-sum: two windows overlapping the same second are counted once
    assert _measure_intersection([(0.0, 10.0)], [(5.0, 20.0)]) == 5.0
    assert _measure_intersection(_merge_spans([(0.0, 6.0), (4.0, 10.0)]),
                                  [(0.0, 100.0)]) == 10.0
    assert _measure_intersection([(0.0, 1.0)], [(2.0, 3.0)]) == 0.0
    assert _measure_intersection([], [(0.0, 1.0)]) == 0.0
    # multi-span sweep, both sides
    assert _measure_intersection([(0.0, 10.0), (20.0, 30.0)],
                                  [(5.0, 25.0)]) == 10.0


# --- the "199 bad frames / 1 good, repeat forever" pattern: where the bound does and does not
# --- reach it. Stated as a test rather than as a claim in prose.

def test_the_199_bad_frames_pattern_is_gated_exactly_when_it_degrades_REALIZED_CADENCE(tmp_path):
    """A stream that yields one parseable event per 200 frames never trips the recorder's
    200-frame drift detector (`_frames_since_last_event` resets on any event). The lens judged
    that acceptable-because-disclosed via the persisted `n_schema_mismatch` counter and the SCHEMA
    DRIFT banner, and identified it as the same underlying weakness as the BLOCKER.

    The honest answer, pinned here: the aggregate check reaches this pattern precisely when the
    pattern degrades the REALIZED data-bearing cadence past the coverage window. It is a
    time-domain bound, not a frame-domain one — which is the right domain, because the estimand
    is denominated in time.
    """
    drift = {"schema_mismatch": 199_000, "unrecognized_frames": 0}

    # (a) fast wire: 1 good event per 200 frames still lands every ~4s. The capture genuinely DOES
    # resolve the minute grid, so the bound does not fire — and the SCHEMA DRIFT banner is what
    # carries the disclosure.
    fast = read_capture(_write_capture(tmp_path / "fast.jsonl", {"BTC/USD": 4.0},
                                       span_s=3600.0, counters=drift))
    assert fast.degraded_coverage_pairs() == []
    fast.assert_complete()
    report = render_report([fast], {}, {})
    assert "SCHEMA DRIFT OBSERVED" in report
    assert "schema_mismatch=199000" in report
    assert "| BTC/USD | book+trade |" in report, (
        "and the realized cadence is on the page next to it, so a reader can see for themselves "
        "at what resolution the surviving 1-in-200 events actually arrived")

    # (b) slow wire: the same 1-in-200 ratio now means one event per ~40s. That IS a degraded
    # minute-scale capture, and the aggregate bound refuses it — so yes, the new check does cover
    # this pattern, in the regime where the pattern actually damages the measurement.
    slow = read_capture(_write_capture(tmp_path / "slow.jsonl", {"BTC/USD": 40.0},
                                       span_s=3600.0, counters=drift))
    assert slow.unrecorded_data_gaps == [], "40s < the 60s per-jump threshold: defense 2 is clean"
    assert slow.degraded_coverage_pairs() == ["BTC/USD"]
    with pytest.raises(CaptureFormatError, match="AGGREGATE data-bearing coverage"):
        slow.assert_complete()


def test_uncovered_fraction_is_no_claim_when_there_is_nothing_to_cover(tmp_path):
    """A zero-length or fully-gap-covered span MAKES NO COVERAGE CLAIM — this test's own words,
    unchanged since it was written. Cycle 2 (run-45) made the value match them: it must not divide
    by zero, must not read as 100% uncovered, and must not read as `0.0` either, which is the
    instrument's STRONGEST coverage claim and the one value spec B2 item 3 forbids in that branch.

    RENAMED from `test_uncovered_fraction_is_zero_when_there_is_nothing_to_cover` — the old name
    asserted the forbidden value in the node ID itself. The FIXTURE is corrected too: it used
    `_write_capture`, which offsets each trade +0.001s from its book event, so its "zero-duration"
    capture actually had `span_s=0.001` and `accounted_s=0.001` — a genuine, fully-covered 0.0
    that never reached the `accounted_s <= 0` branch this test claims to be about. Written by hand
    here so every event really does sit at one instant and the branch is exercised end to end."""
    path = tmp_path / "z.jsonl"
    with open(path, "w") as fh:
        fh.write(json.dumps({"type": "meta_start", "capture_id": "z", "pairs": ["BTC/USD"],
                              "format_version": 2, "clock_basis": "local_receipt",
                              "channels": ["book", "trade"]}) + "\n")
        fh.write(json.dumps({"type": "book", "pair": "BTC/USD", "ts": T0, "bid": 50000.0,
                              "ask": 50001.0, "bid_sz": 1.0, "ask_sz": 1.0}) + "\n")
        fh.write(json.dumps({"type": "trade", "pair": "BTC/USD", "ts": T0, "price": 50000.5,
                              "volume": 0.1, "side": "buy"}) + "\n")
    meta = read_capture(str(path))
    assert meta.coverage_by_pair["BTC/USD"].span_s == 0.0
    assert meta.coverage_by_pair["BTC/USD"].uncovered_fraction is None
    assert meta.no_claim_coverage_pairs() == ["BTC/USD"]
    assert meta.degraded_coverage_pairs() == [], (
        "NO CLAIM is not a bound violation — nothing is hidden in a zero-length span")
    meta.assert_complete()


# --- ADDENDUM B1 (run 46, D73 carve-out) — the cycle-3 per-pair stale-book EXCLUSION does NOT
# --- ship here. Interim behaviour is the PRE-CYCLE-3 whole-capture refusal, fail-CLOSED: a pair
# --- whose keepalive-driven stale book time alone crosses the coverage bound takes the WHOLE
# --- capture down with it, exactly like a pair with a dead subscription. Built through the real
# --- `CaptureWriter`/keepalive machinery (Addendum A), not the synthetic `_write_capture` helper,
# --- because the stale-run detection is value-based over REAL book rows.

def test_a_thin_pairs_frozen_keepalive_book_refuses_the_WHOLE_capture_not_just_itself(tmp_path):
    """The regression target D73 names for B1: cycle 2 correctly stops counting a keepalive-
    frozen top of book as an observation; the carved interim behaviour routes that straight into
    `degraded_coverage_pairs` — a WHOLE-CAPTURE refusal — with NO per-pair exclusion escape hatch.

    BTC/USD is healthy for the entire 3600s (book changes every 5s, trades print every 10s).
    SOL/USD is THIN: its book changes for the first 300s, then FREEZES — the recorder keeps
    emitting an IDENTICAL top of book on its declared 5.0s keepalive timer — for the remaining
    3300s, and it prints NO trades at all (book is its only data-bearing channel, the shape a
    real illiquid pair takes). This is exactly the case cycle 2 (and now this carve) correctly
    reads as under-covered: `stale_book_s` counts nearly the whole frozen stretch, and nothing
    else (no trade prints) backs it.

    Pre-cycle-3 fail-closed means the refusal discards BOTH pairs' numbers, including the
    perfectly healthy BTC/USD — that is the cost of not having B2's per-pair split yet, and it is
    the direction D73 explicitly wants B1 to fail in.

    SCOPE, added run-49 so this test's title is not read wider than its fixture: the refusal fires
    here because SOL/USD prints NO trades, so its stale book time is the whole of its coverage.
    The same freeze WITH a live trade stream does not reach the bound and does not refuse — see
    `test_that_frozen_mid_is_NOT_refused_by_B1_and_every_shipped_string_says_so`. Whole-capture
    refusal on stale books is not a general property of B1, only of this shape.
    """
    out = str(tmp_path / "thin_pair_frozen.jsonl")
    w = CaptureWriter(out, ["BTC/USD", "SOL/USD"], "capThin",
                       channels=list(SUBSCRIBED_CHANNELS), heartbeat_timeout_s=30.0,
                       book_emission_policy=BOOK_EMISSION_POLICY, book_keepalive_s=5.0)
    t, bid = 0.0, 100.0
    while t <= 3600.0:
        w.write_book("BTC/USD", t, bid, bid + 0.2, 1.0, 1.0)
        bid += 0.001
        t += 5.0
    t = 0.0
    while t <= 3600.0:
        w.write_trade("BTC/USD", t, bid, 0.5, "buy")
        t += 10.0
    t, sbid = 0.0, 20.0
    while t <= 300.0:
        w.write_book("SOL/USD", t, sbid, sbid + 0.05, 5.0, 5.0)
        sbid += 0.001
        t += 5.0
    frozen_bid = sbid
    t = 305.0
    while t <= 3600.0:
        w.write_book("SOL/USD", t, frozen_bid, frozen_bid + 0.05, 5.0, 5.0)
        t += 5.0
    hb = 0.0
    while hb <= 3600.0:
        w.write_heartbeat(hb)
        hb += 10.0
    w.close(clean=True)

    meta = read_capture(out)
    assert meta.coverage_by_pair["BTC/USD"].uncovered_fraction == 0.0, (
        "BTC/USD is healthy on its own — its own numbers must not be the reason this refuses")
    sol = meta.coverage_by_pair["SOL/USD"]
    assert sol.stale_book_s > 3000.0, "the frozen run really was detected"
    assert sol.uncovered_fraction > meta.max_uncovered_fraction
    assert meta.degraded_coverage_pairs() == ["SOL/USD"]
    assert not meta.is_complete()
    with pytest.raises(CaptureFormatError) as excinfo:
        meta.assert_complete()
    msg = str(excinfo.value)
    assert "AGGREGATE data-bearing coverage" in msg
    assert "SOL/USD" in msg
    # the deliverable sentence: a reader must not mistake this refusal for per-pair exclusion
    assert "ADDENDUM B2" in msg and "not implemented" in msg.lower()
    assert not hasattr(meta, "stale_book_excluded_pairs")

    # render_report must refuse too — no artifact, not even one that drops SOL/USD and keeps
    # BTC/USD's healthy numbers (that split is exactly the B2-scope behaviour NOT shipped here).
    with pytest.raises(CaptureFormatError):
        render_report([meta], {}, {})


def test_book_dying_while_trades_keep_flowing_is_NOT_caught_by_B1_pending_addendum_b2(tmp_path):
    """Pins a KNOWN GAP, not a feature — read this test as a documented finding, not a pass.

    D73's own diagnosis: the cycle-3 exclusion was denominated in DATA-BEARING coverage (book OR
    trade), while estimand 1 is computed from BOOK MIDS ONLY. That same denomination problem
    exists in the plain, un-excluded `uncovered_fraction` this carve restores — `covered_s` is
    the union of book AND trade timestamps, so a pair whose BOOK channel goes completely silent
    while its TRADE channel keeps printing reads as fully covered, because trades alone satisfy
    the data-bearing coverage check. This is DIFFERENT from — and NOT fixed by — the B1 carve:
    the thin-pair keepalive-freeze case above (`stale_book_s`, a value-based detector over
    REPEATED book rows) does not fire here at all, because the book channel does not repeat
    anything after it dies — it simply stops.

    BTC/USD's book emits changing values for the first 1800s of a 3600s capture, then the book
    channel goes silent for the remaining 1800s (zero further book events, not even keepalives —
    a plausible per-channel subscription failure). Trades keep printing every 10s for the WHOLE
    3600s. Measured: `uncovered_fraction == 0.0`, `degraded_coverage_pairs() == []`,
    `assert_complete()` passes — an estimand-1 table built on this capture would markout against
    a book mid frozen since t=1800 for three quarters of the capture, with the gate reporting
    full coverage throughout.

    This is the PRE-EXISTING hole D73 names ("a BTC-only capture passes at HEAD too") and is
    explicitly ADDENDUM B2 scope: fixing it needs a book-mid-only freshness term
    (`stale_book_s / accounted_s`, from BOOK timestamps alone, independent of whether trades
    carried coverage) — not implemented here. This test exists so a future change to B1 cannot
    silently claim to have closed this without it being noticed as a behaviour change.
    """
    out = str(tmp_path / "book_dies_trades_flow.jsonl")
    w = CaptureWriter(out, ["BTC/USD"], "capDeath", channels=list(SUBSCRIBED_CHANNELS),
                       heartbeat_timeout_s=30.0, book_emission_policy=BOOK_EMISSION_POLICY,
                       book_keepalive_s=5.0)
    t, bid = 0.0, 100.0
    while t <= 1800.0:
        w.write_book("BTC/USD", t, bid, bid + 0.2, 1.0, 1.0)
        bid += 0.001
        t += 5.0
    hb = 0.0
    while hb <= 3600.0:
        w.write_heartbeat(hb)
        hb += 10.0
    t = 0.0
    while t <= 3600.0:
        w.write_trade("BTC/USD", t, bid, 0.5, "buy" if int(t) % 2 == 0 else "sell")
        t += 10.0
    w.close(clean=True)

    meta = read_capture(out)
    cov = meta.coverage_by_pair["BTC/USD"]
    assert cov.uncovered_fraction == 0.0, (
        "documented gap: trade events alone keep data-bearing coverage full even though the "
        "book itself has been dead for half the capture — see the docstring")
    assert cov.stale_book_s == 0.0, (
        "the value-based stale-RUN detector never fires: the book does not repeat a frozen "
        "value, it simply stops emitting anything")
    assert meta.degraded_coverage_pairs() == []
    assert meta.is_complete()
    meta.assert_complete()  # does NOT raise — this is the gap, asserted explicitly

    # run-49: B1 does now SURFACE a signal for this shape, on the other direction of the same
    # Addendum-A declaration — the book stopped emitting against a declared 5.0s floor, and the
    # edge-anchored overrun scan sees the trailing silence (it did not before: the scan looked
    # strictly BETWEEN consecutive book events, and there is no book event after t=1800 to pair
    # with). Still DISCLOSURE ONLY — the gate does not refuse, the estimand table is still
    # published on a mid frozen since t=1800, and closing that is B2's freshness term. This
    # assertion exists so the disclosure cannot be lost while the gap stays open.
    assert meta.book_keepalive_overruns.get("BTC/USD") == 1
    assert meta.book_keepalive_max_interval_s["BTC/USD"] > 1700.0


# --- run-49 BLOCKER: the staleness run key was FINER than the quantity it protects ---------------

def _write_frozen_mid_jittering_sizes(path, span_s=7200.0, book_every_s=5.0,
                                       trade_every_s=30.0):
    """A book whose (bid, ask) never moves for the whole capture while its SIZES tick, plus a
    live trade stream. Built through the real `CaptureWriter`/keepalive machinery, because the
    stale-run detection is value-based over real book rows."""
    w = CaptureWriter(str(path), ["BTC/USD"], "capJitter",
                       channels=list(SUBSCRIBED_CHANNELS), heartbeat_timeout_s=30.0,
                       book_emission_policy=BOOK_EMISSION_POLICY, book_keepalive_s=5.0)
    sizes = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    t, i = 0.0, 0
    while t <= span_s:
        w.write_book("BTC/USD", t, 100.0, 100.2, sizes[i % 7], sizes[(i + 3) % 7])
        i += 1
        t += book_every_s
    t = 0.0
    while t <= span_s:
        w.write_trade("BTC/USD", t, 100.1, 0.5, "buy")
        t += trade_every_s
    hb = 0.0
    while hb <= span_s:
        w.write_heartbeat(hb)
        hb += 10.0
    w.close(clean=True)
    return str(path)


def test_a_frozen_mid_with_jittering_sizes_is_STALE_the_key_is_the_price_pair(tmp_path):
    """RED before the fix: `stale_book_s == 0.0` and `stale_book_pairs() == []`.

    The staleness run key was `(bid, ask, bid_sz, ask_sz)` — strictly FINER than the quantity it
    protects. The estimands consume only the MID (`shadow_engine`, `(bid+ask)/2`); sizes reach no
    number anywhere. So any size tick broke the "stale run" while the mid stayed frozen, and the
    detector built to notice a frozen mid could be switched off by a quantity that is not the mid.
    Measured end to end before the fix: a price pinned at 100.0/100.2 for 7200s with sizes cycling
    through seven values published `1m median 2.48 bp`, deciles -4.00..+9.00 and
    `mid age p50 0.500s` — plausible-looking markouts on a mid that never moved, with
    `assert_complete()` PASSING and `rc=0`.

    This is the same wrong-denomination shape as the cycle-3 BLOCKER that forced the D73 split,
    one field over: a check must never be keyed on something finer than what it guards.
    """
    path = _write_frozen_mid_jittering_sizes(tmp_path / "jitter.jsonl")
    meta = read_capture(path)
    cov = meta.coverage_by_pair["BTC/USD"]
    assert cov.stale_book_s > 7000.0, (
        "the mid was frozen for the whole 7200s capture — the sizes ticking is not an observation "
        "of anything the estimands read")
    assert meta.stale_book_pairs() == ["BTC/USD"]


def test_that_frozen_mid_is_NOT_refused_by_B1_and_every_shipped_string_says_so(tmp_path):
    """The CLAIM half of the same finding, and the one that matters more (D57: a claim in a
    shipped string the code does not back, rendered where a reader checks legitimacy).

    Re-keying makes the DISCLOSURE honest. It does NOT make the gate fire, because
    `uncovered_fraction` is denominated over the UNION of book and trade timestamps and the trade
    stream keeps that union dense: measured `uncovered_fraction == 0.00132` against a 2% bound,
    `degraded_coverage_pairs() == []`, `assert_complete()` PASSING. B1 therefore CANNOT make true
    the sentence it used to ship — "a pair whose stale book time alone crosses the bound refuses
    the WHOLE capture" — for either known shape (this one, and the book-going-silent case pinned
    above). Both need the book-mid-only freshness term D73 assigns to ADDENDUM B2.

    So this test pins the narrowed claim in the place a reader actually reads it.
    """
    path = _write_frozen_mid_jittering_sizes(tmp_path / "jitter2.jsonl")
    meta = read_capture(path)
    cov = meta.coverage_by_pair["BTC/USD"]
    assert cov.uncovered_fraction < meta.max_uncovered_fraction, (
        "documented, deliberately un-closed: live trades keep the union coverage dense")
    assert meta.degraded_coverage_pairs() == []
    assert meta.is_complete()
    meta.assert_complete()   # does NOT raise — this is the gap, asserted explicitly

    meta2, results = run_shadow(path)
    e1 = {p: compute_estimand1(r) for p, r in results.items()}
    e2 = {p: compute_estimand2(r) for p, r in results.items()}
    report = render_report([meta2], e1, e2)
    assert "FROZEN-MID BOOK TIME" in report
    assert "NOT caught by any gate in ADDENDUM B1" in report, (
        "the artifact must say plainly that a stale book masked by live trade flow is not caught "
        "here — the reader is being shown markouts built on that mid")
    assert "PRICE PAIR `(bid, ask)`" in report, (
        "and must say what the staleness key is, since the estimands' mid is what it guards")
    for banned in ("stale time alone pushes it past the declared bound REFUSES THE WHOLE CAPTURE",
                   "stale book time alone crosses the bound refuses the WHOLE capture"):
        assert banned not in report, f"falsified claim still shipped: {banned!r}"


def test_the_aggregate_refusal_states_the_exact_reach_of_its_stale_book_coverage(tmp_path):
    """The mirror side: where the whole-capture refusal DOES fire (stale book AND no trades), its
    text must scope the claim to the union denomination rather than implying stale book time is
    caught generally."""
    out = str(tmp_path / "thin_frozen.jsonl")
    w = CaptureWriter(out, ["BTC/USD", "SOL/USD"], "capThin2",
                       channels=list(SUBSCRIBED_CHANNELS), heartbeat_timeout_s=30.0,
                       book_emission_policy=BOOK_EMISSION_POLICY, book_keepalive_s=5.0)
    t, bid = 0.0, 100.0
    while t <= 3600.0:
        w.write_book("BTC/USD", t, bid, bid + 0.2, 1.0, 1.0)
        bid += 0.001
        t += 5.0
    t = 0.0
    while t <= 3600.0:
        w.write_trade("BTC/USD", t, bid, 0.5, "buy")
        t += 10.0
    t, sbid = 0.0, 20.0
    while t <= 300.0:
        w.write_book("SOL/USD", t, sbid, sbid + 0.05, 5.0, 5.0)
        sbid += 0.001
        t += 5.0
    t = 305.0
    while t <= 3600.0:
        # sizes tick here too — under the OLD key this whole stretch was not "stale" at all
        w.write_book("SOL/USD", t, sbid, sbid + 0.05, 5.0 + (t % 3), 5.0)
        t += 5.0
    hb = 0.0
    while hb <= 3600.0:
        w.write_heartbeat(hb)
        hb += 10.0
    w.close(clean=True)

    meta = read_capture(out)
    assert meta.degraded_coverage_pairs() == ["SOL/USD"]
    with pytest.raises(CaptureFormatError) as excinfo:
        meta.assert_complete()
    msg = str(excinfo.value)
    assert "UNION of each pair's book AND trade observations" in msg
    assert "NOT caught anywhere in ADDENDUM B1" in msg
    assert "refuses the whole capture exactly like a dead subscription" not in msg, (
        "the unconditional version of that claim is false and must not ship")


# --- run-49 cycle 3 MAJOR: an OSCILLATING dead mid is not counted by the stale-RUN measure -------

def _write_oscillating_dead_mid(path, with_trades, span_s=7200.0, book_every_s=5.0,
                                 trade_every_s=30.0):
    """A top of book that ALTERNATES between two `(bid, ask)` values for the whole capture: a mid
    that moves on every single event and discovers no new price. Written through the real
    `CaptureWriter`/keepalive machinery, because the stale-run detection is value-based over real
    book rows. `trade` is DECLARED in both variants; `with_trades=False` simply records none, which
    is the shape the capture-level channel gate exists for."""
    w = CaptureWriter(str(path), ["BTC/USD"], "capOsc", channels=list(SUBSCRIBED_CHANNELS),
                       heartbeat_timeout_s=30.0, book_emission_policy=BOOK_EMISSION_POLICY,
                       book_keepalive_s=5.0)
    tops = [(100.0, 100.2), (100.01, 100.21)]
    t, i = 0.0, 0
    while t <= span_s:
        bid, ask = tops[i % 2]
        w.write_book("BTC/USD", t, bid, ask, 1.0, 1.0)
        i += 1
        t += book_every_s
    if with_trades:
        t = 0.0
        while t <= span_s:
            w.write_trade("BTC/USD", t, 100.1, 0.5, "buy")
            t += trade_every_s
    hb = 0.0
    while hb <= span_s:
        w.write_heartbeat(hb)
        hb += 10.0
    w.close(clean=True)
    return str(path)


def test_an_OSCILLATING_dead_mid_reads_stale_book_s_ZERO_and_is_NOT_detected_in_B1(tmp_path):
    """DELIBERATELY UN-CLOSED, pinned so nobody later reads this as fixed without noticing.

    This test asserts a HOLE, not a feature. The staleness term counts one shape only: a run of
    CONSECUTIVE IDENTICAL `(bid, ask)` events. Any change to the price pair — INCLUDING A CHANGE
    BACK — ends the run, so a mid alternating over a small set of values defeats the measure
    entirely: every run has n=1 and `stale_book_s` is 0.0 for a book that discovered no price in
    7200s. Measured on the live tree, both variants of this fixture:

        with trades: stale 0.0, uncovered 0.0000, degraded [] -> PUBLISHES
        no trades:   stale 0.0, uncovered 0.0000, degraded [] -> REFUSED, on the CHANNEL gate

    Note which gate does the refusing in the second row: `subscribed channel(s) ['trade'] recorded
    ZERO events`, a capture-level check that is independent of staleness. Staleness never had to
    catch it, and did not. The first row is exactly the class B1's own narrowed text already
    disclaims ("A book frozen or silent while its prints keep flowing is NOT caught by any gate in
    ADDENDUM B1"), so this is NOT a falsification of the shipped claim — it is the residue of it.

    The residue is a DISCLOSURE gap: the FROZEN-MID BOOK TIME line does not render at all here, and
    a reader could take that absence as evidence the mid was informative. Cycle 3 closes it in TEXT
    only (`STALE_BOOK_MIN_RUN_S`, `stale_book_pairs`, `render.py`'s FROZEN-MID block), and this
    test pins both halves: the honest behaviour, and the disclosure that now names its own reach.

    **Do NOT "fix" this by widening the run key** to period-N cycles, bounded-cardinality value
    sets, or "no net movement over a window". That is a book-mid-only FRESHNESS term denominated
    independently of repeats — D73 assigns it to ADDENDUM B2 — and chasing escaping input shapes
    inside B1 is the pattern that halted TASK_891 after three cycles.
    """
    # (a) the honest behaviour, with trades flowing: measured, published, and NOT flagged stale
    path = _write_oscillating_dead_mid(tmp_path / "osc_trades.jsonl", with_trades=True)
    meta = read_capture(path)
    cov = meta.coverage_by_pair["BTC/USD"]
    assert cov.stale_book_s == 0.0, (
        "documented gap: the run key is CONSECUTIVE IDENTICAL (bid, ask), and an alternating mid "
        "breaks every run at the first alternation — see the docstring")
    assert meta.stale_book_pairs() == []
    assert cov.uncovered_fraction == 0.0
    assert meta.degraded_coverage_pairs() == []
    assert meta.is_complete()
    meta.assert_complete()   # does NOT raise — this is the gap, asserted explicitly

    meta2, results = run_shadow(path)
    report = render_report([meta2],
                            {p: compute_estimand1(r) for p, r in results.items()},
                            {p: compute_estimand2(r) for p, r in results.items()})
    assert "FROZEN-MID BOOK TIME" not in report, (
        "the disclosure line does not render for this shape AT ALL — that silence is precisely "
        "the gap the cycle-3 text closes, and it must not be mistaken for a clean bill")

    # (b) the half that matters: with the trade channel declared but silent, the capture is still
    # REFUSED — by the channel gate, not by anything staleness-related.
    path2 = _write_oscillating_dead_mid(tmp_path / "osc_no_trades.jsonl", with_trades=False)
    meta_nt = read_capture(path2)
    assert meta_nt.coverage_by_pair["BTC/USD"].stale_book_s == 0.0
    assert meta_nt.degraded_coverage_pairs() == [], (
        "the book is dense every 5s and none of it is counted stale, so the coverage check has "
        "nothing to say here either")
    assert not meta_nt.is_complete()
    with pytest.raises(CaptureFormatError, match="recorded ZERO events"):
        meta_nt.assert_complete()

    # (c) and where the FROZEN-MID line DOES render, it must now state what "stale" reaches, so a
    # `0.0s` reading or an absent pair cannot be read as "the mid was informative".
    frozen = _write_frozen_mid_jittering_sizes(tmp_path / "osc_ref.jsonl")
    fmeta, fresults = run_shadow(frozen)
    freport = render_report([fmeta],
                             {p: compute_estimand1(r) for p, r in fresults.items()},
                             {p: compute_estimand2(r) for p, r in fresults.items()})
    assert "RUN OF CONSECUTIVE IDENTICAL `(bid, ask)` events" in freport, (
        "the line names a mechanism, so it must name the mechanism's exact reach")
    assert "OSCILLATES" in freport and "NOT DETECTED ANYWHERE IN ADDENDUM B1" in freport
    assert "book-mid FRESHNESS term" in freport, (
        "and must point at the denomination-correct instrument (B2), not at a wider run key")


# --- run-49 MAJOR: `book_keepalive_overruns` had the edge hole B2 item 1 already fixed once ------

def _write_book_stretch(path, book_from_s, book_to_s, span_s=3600.0, book_every_s=5.0):
    """Book events only inside [book_from_s, book_to_s]; heartbeats across the whole span, so the
    CAPTURE span is `span_s` while the book channel is silent at one or both edges."""
    w = CaptureWriter(str(path), ["BTC/USD"], "capEdge", channels=["book"],
                       heartbeat_timeout_s=30.0, book_emission_policy=BOOK_EMISSION_POLICY,
                       book_keepalive_s=5.0)
    t, bid = book_from_s, 100.0
    while t <= book_to_s:
        w.write_book("BTC/USD", t, bid, bid + 0.2, 1.0, 1.0)
        bid += 0.001
        t += book_every_s
    hb = 0.0
    while hb <= span_s:
        w.write_heartbeat(hb)
        hb += 10.0
    w.close(clean=True)
    return str(path)


@pytest.mark.parametrize("book_from_s,book_to_s,label", [
    (0.0, 1800.0, "trailing"),   # emits on time, then simply stops for the second half
    (1800.0, 3600.0, "leading"),  # silent for the first half, then emits on time
])
def test_book_silence_at_a_CAPTURE_SPAN_EDGE_is_a_keepalive_overrun(tmp_path, book_from_s,
                                                                     book_to_s, label):
    """RED before the fix: `book_keepalive_overruns == {}` for both shapes.

    The overrun scan was `for i in range(1, len(book_list))` — strictly BETWEEN consecutive book
    events — which is the identical edge hole B2 item 1 already fixed in `_find_unrecorded_gaps`
    one screen away. A book that emits every 5s for 1800s of a 3600s capture and then goes silent
    has no consecutive PAIR of book events spanning the silence, so an 1800s hole against a
    DECLARED 5.0s floor — 360x the violation, the worst case this check exists for — was invisible.
    Leading silence was invisible for the same reason.

    Disclosure-only either way (a scheduler hiccup must not retro-refuse a multi-day capture), but
    a disclosure that cannot see its own worst case is not a disclosure.
    """
    path = _write_book_stretch(tmp_path / f"edge_{label}.jsonl", book_from_s, book_to_s)
    meta = read_capture(path)
    assert meta.declared_book_keepalive_s == 5.0
    assert meta.book_keepalive_overruns.get("BTC/USD"), (
        f"{label} book silence of ~1800s against a declared 5.0s floor must be disclosed")
    assert meta.book_keepalive_max_interval_s["BTC/USD"] > 1700.0


def test_the_edge_anchored_overrun_scan_stays_quiet_on_a_healthy_book(tmp_path):
    """Non-vacuity: anchoring must not manufacture an overrun at every capture's edges. A book
    emitting across the whole capture span has no edge interval to speak of."""
    path = _write_book_stretch(tmp_path / "edge_healthy.jsonl", 0.0, 3600.0)
    meta = read_capture(path)
    assert meta.book_keepalive_overruns == {}
