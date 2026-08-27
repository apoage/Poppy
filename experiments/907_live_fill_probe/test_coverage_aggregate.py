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
    CaptureMetadata, _measure_intersection, _merge_spans, read_capture,
)
from estimands import compute_estimand1, compute_estimand2  # noqa: E402
from recorder import RecorderConfig  # noqa: E402
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
    path = _write_capture(tmp_path / "gapped.jsonl", {
        "BTC/USD": [(0.0, 3600.0, 1.0), (10800.0, 14400.0, 1.0)],
    }, span_s=14400.0, gaps=[(3600.0, 10800.0, "disconnect")])
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


def test_uncovered_fraction_is_zero_when_there_is_nothing_to_cover(tmp_path):
    """A zero-length or fully-gap-covered span makes no coverage claim, so it must not divide by
    zero nor read as 100% uncovered."""
    path = _write_capture(tmp_path / "z.jsonl", {"BTC/USD": [(0.0, 0.0, 1.0)]}, span_s=0.0)
    meta = read_capture(path)
    assert meta.coverage_by_pair["BTC/USD"].uncovered_fraction == 0.0
    assert meta.degraded_coverage_pairs() == []
