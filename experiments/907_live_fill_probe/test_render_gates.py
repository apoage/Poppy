import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture import (  # noqa: E402
    MAX_GAP_DETECT_THRESHOLD_S, CaptureFormatError, CaptureWriter, GapRecord, read_capture,
)
from estimands import (  # noqa: E402
    Estimand1Result, Estimand2Result, compute_estimand1,
)
from render import (  # noqa: E402
    MAX_RENDERED_GAP_LINES, NOT_EVALUABLE, REQUIRED_DISCLOSURE, render_disclosure_box,
    render_report,
)
from shadow_engine import MarkoutSample, ShadowPairResult  # noqa: E402
import cli  # noqa: E402


def _complete_capture(tmp_path):
    p = tmp_path / "cap.jsonl"
    w = CaptureWriter(str(p), ["BTC/USD"], "cap-1")
    w.write_trade("BTC/USD", 10.0, 100.0, 1.0, "buy")
    w.write_book("BTC/USD", 10.0, 100.0, 100.2, 1.0, 1.0)
    w.close(clean=True)
    return read_capture(str(p))


# --- gate (iii): the required disclosure ---

def test_gate_iii_disclosure_box_contains_required_text():
    box = render_disclosure_box()
    assert REQUIRED_DISCLOSURE in box


def test_gate_iii_required_text_names_the_897_robustness_argument():
    # this pins the SPECIFIC quoted number, not just "some disclaimer exists" — a future
    # refactor that waters the claim down to something vaguer should fail this test.
    assert "0/188 clear even with adverse selection set to ZERO" in REQUIRED_DISCLOSURE
    assert "cannot reopen TASK_897's ALL FAIL" in REQUIRED_DISCLOSURE


def test_gate_iii_disclosure_is_the_first_content_in_a_rendered_report(tmp_path):
    meta = _complete_capture(tmp_path)
    report = render_report([meta], {}, {})
    disclosure_idx = report.index(REQUIRED_DISCLOSURE)
    other_markers = ["Capture provenance", "## "]
    for marker in other_markers:
        idx = report.index(marker)
        assert disclosure_idx < idx, f"{marker!r} appears before the required disclosure"


def test_gate_iii_present_even_with_real_estimand_data(tmp_path):
    meta = _complete_capture(tmp_path)
    e1 = Estimand1Result(pair="BTC/USD", by_horizon={})
    e2 = Estimand2Result(pair="BTC/USD", n_posted=0, n_touch_fills=0, n_trade_cross_fills=0,
                          touch_fill_rate=None, trade_cross_fill_rate=None)
    report = render_report([meta], {"BTC/USD": e1}, {"BTC/USD": e2})
    assert REQUIRED_DISCLOSURE in report


# --- gate (iv): fail-closed on incomplete/absent/inconsistent capture metadata ---

def test_gate_iv_refuses_on_no_meta_start(tmp_path):
    p = tmp_path / "bad.jsonl"
    with open(p, "w") as fh:
        fh.write(json.dumps({"type": "trade", "pair": "BTC/USD", "ts": 1.0, "price": 1.0,
                              "volume": 1.0, "side": "buy"}) + "\n")
    meta = read_capture(str(p))
    with pytest.raises(CaptureFormatError):
        render_report([meta], {}, {})


def test_gate_iv_refuses_on_unrecorded_gap(tmp_path):
    p = tmp_path / "bad.jsonl"
    with open(p, "w") as fh:
        fh.write(json.dumps({"type": "meta_start", "capture_id": "x", "pairs": ["BTC/USD"],
                              "format_version": 1}) + "\n")
        fh.write(json.dumps({"type": "trade", "pair": "BTC/USD", "ts": 0.0, "price": 1.0,
                              "volume": 1.0, "side": "buy"}) + "\n")
        fh.write(json.dumps({"type": "trade", "pair": "BTC/USD", "ts": 10000.0, "price": 1.0,
                              "volume": 1.0, "side": "buy"}) + "\n")
    meta = read_capture(str(p), gap_detect_threshold_s=MAX_GAP_DETECT_THRESHOLD_S)
    with pytest.raises(CaptureFormatError, match="GATE \\(iv\\)"):
        render_report([meta], {}, {})


def test_gate_iv_refuses_on_empty_capture_list():
    with pytest.raises(CaptureFormatError):
        render_report([], {}, {})


# --- BLOCKER 1: the artifact must never render a verdict derived from a non-finite value ---

# Matches a non-finite value rendered AS A NUMBER — `| nan |`, `median nanbp`, `-inf` — while
# not tripping on ordinary prose that happens to contain the letters (e.g. "prove-nan-ce").
_RENDERED_NONFINITE = re.compile(
    r"(?:^|[|\s(])[-+]?(?:nan|inf(?:inity)?)(?:bp)?(?=$|[|\s.,;)])", re.IGNORECASE | re.MULTILINE)


def _rendered_nonfinite(text):
    return _RENDERED_NONFINITE.findall(text)


def test_the_nonfinite_detector_used_below_is_not_vacuous():
    """Proves the guard would have CAUGHT the original defect's exact output."""
    original_bug_output = (
        "| 1m | 11 | nan | nan | 1.00 | 2.00 |\n"
        "- 1m: median nanbp — 0bp: BELOW measured, 5bp: BELOW measured\n")
    assert _rendered_nonfinite(original_bug_output), "the detector must catch the real defect"
    # and it does NOT fire on innocent prose containing the same letters
    assert not _rendered_nonfinite("## Capture provenance (gate iv)")
    assert not _rendered_nonfinite("- influence, nanometre, information, infrastructure")


def _e1_all_nan():
    r = ShadowPairResult(pair="BTC/USD")
    for i in range(3):
        r.markouts.append(MarkoutSample(pair="BTC/USD", side=1, fill_price=100.0,
                                         fill_ts=float(i), horizon="1m",
                                         adverse_selection_bp=float("nan")))
    return compute_estimand1(r)


def test_non_evaluable_horizon_renders_not_evaluable_and_never_below_measured(tmp_path):
    """The reproduced defect emitted, verbatim:
    `1m: median nanbp — 0bp: BELOW measured, 5bp: BELOW measured, 10bp: BELOW measured`
    — a confidently-worded FALSE calibration verdict, because `tier >= nan` is always False."""
    meta = _complete_capture(tmp_path)
    report = render_report([meta], {"BTC/USD": _e1_all_nan()}, {})

    assert not _rendered_nonfinite(report), "a non-finite value reached the artifact"
    assert "BELOW measured" not in report, "a verdict was emitted from a non-finite median"
    assert NOT_EVALUABLE in report
    # the disclosure names the rule, so the reader knows a check did not run
    assert "not a passed check" in report
    # and the dropped samples are counted, not silently excluded
    assert "3" in report and "non-finite" in report.lower()


def test_evaluable_horizon_still_renders_a_real_verdict(tmp_path):
    """Non-vacuity guard: the not-evaluable path must not have disabled real verdicts."""
    meta = _complete_capture(tmp_path)
    r = ShadowPairResult(pair="BTC/USD")
    for i, v in enumerate([1.0, 3.0, 5.0]):
        r.markouts.append(MarkoutSample(pair="BTC/USD", side=1, fill_price=100.0,
                                         fill_ts=float(i), horizon="1m",
                                         adverse_selection_bp=v))
    report = render_report([meta], {"BTC/USD": compute_estimand1(r)}, {})
    assert "BELOW measured" in report      # the 0bp tier really is below the 3.0bp median
    assert "meets/exceeds" in report       # the 5/10bp tiers really do meet it
    assert "median 3.00bp" in report


def test_mixed_finite_and_nonfinite_discloses_the_drop_count(tmp_path):
    meta = _complete_capture(tmp_path)
    r = ShadowPairResult(pair="BTC/USD")
    for i, v in enumerate([1.0, float("nan"), 5.0]):
        r.markouts.append(MarkoutSample(pair="BTC/USD", side=1, fill_price=100.0,
                                         fill_ts=float(i), horizon="1m",
                                         adverse_selection_bp=v))
    report = render_report([meta], {"BTC/USD": compute_estimand1(r)}, {})
    assert not _rendered_nonfinite(report)
    assert "1 markout sample(s) were NON-FINITE" in report
    assert "n_used" in report and "n_dropped_nonfinite" in report
    assert "| 1m | 2 | 1 |" in report  # n_used=2, n_dropped_nonfinite=1


def test_estimand2_drop_counters_are_disclosed(tmp_path):
    meta = _complete_capture(tmp_path)
    e2 = Estimand2Result(pair="BTC/USD", n_posted=10, n_touch_fills=4, n_trade_cross_fills=6,
                          touch_fill_rate=0.4, trade_cross_fill_rate=0.6,
                          n_markouts_dropped_nonfinite=7, n_markouts_dropped_gap=5,
                          n_markouts_dropped_no_mid=3, n_markouts_dropped_outside_span=9)
    report = render_report([meta], {}, {"BTC/USD": e2})
    assert "5 straddled a recorded gap" in report
    assert "9 had a horizon past the end of the capture" in report
    assert "3 had no as-of mid" in report
    assert "7 were non-finite" in report


def test_recorder_health_counters_are_surfaced_in_the_artifact(tmp_path):
    """MAJOR: schema drift must be VISIBLE. A capture with no counters record must say the
    counts are UNKNOWN, not imply they were zero."""
    meta = _complete_capture(tmp_path)
    report = render_report([meta], {}, {})
    assert "NOT RECORDED in this capture" in report
    assert "UNKNOWN for it, not zero" in report

    meta.counters_present = True
    meta.n_schema_mismatch = 12
    meta.n_unrecognized_frames = 3
    report = render_report([meta], {}, {})
    assert "schema_mismatch=12" in report
    assert "unrecognized_frames=3" in report
    assert "SCHEMA DRIFT OBSERVED" in report


def test_gap_ledger_normalization_is_disclosed(tmp_path):
    meta = _complete_capture(tmp_path)
    meta.gaps = [GapRecord(0.0, 10.0, "disconnect"), GapRecord(5.0, 20.0, "disconnect")]
    meta.n_overlapping_gap_records = 1
    meta.merged_gap_spans = [(0.0, 20.0)]
    report = render_report([meta], {}, {})
    assert "gap-ledger normalization" in report
    assert "overlapping_records=1" in report
    assert "measure of the UNION" in report


def test_clock_basis_is_stated_in_the_artifact(tmp_path):
    meta = _complete_capture(tmp_path)
    report = render_report([meta], {}, {})
    assert "local receipt time" in report
    assert "clock_basis=local_receipt" in report


def test_gate_iv_cli_refuses_cleanly_and_writes_no_file(tmp_path):
    p = tmp_path / "bad.jsonl"
    with open(p, "w") as fh:
        fh.write(json.dumps({"type": "trade", "pair": "BTC/USD", "ts": 1.0, "price": 1.0,
                              "volume": 1.0, "side": "buy"}) + "\n")  # no meta_start at all
    out_path = tmp_path / "should_not_exist.md"
    rc = cli.main(["--capture", str(p), "--out", str(out_path)])
    assert rc == 1
    assert not out_path.exists()


# --- run-49 MINOR: the artifact for a real capture was unreadable ------------------------------

def test_the_gap_provenance_listing_is_capped_with_its_total_stated(tmp_path):
    """RED before the fix: one line per recorded gap, uncapped.

    The real 3.586-day capture carries 1,688 gap records, and the rendered artifact came to 1,860
    lines of which ~1,700 were this one loop — the disclosures the artifact exists to make (the
    coverage table, the stale-book and keepalive-overrun lines, the estimand rows) buried under a
    raw ledger dump. Capped like every other sample list in this lane's refusals, and the CAP IS
    NOT A DROP: the total, the total seconds and the per-reason breakdown are stated, and every
    gap still counts in full toward coverage and toward the per-markout `covers()` check.
    """
    p = tmp_path / "many_gaps.jsonl"
    w = CaptureWriter(str(p), ["BTC/USD"], "cap-gaps")
    n_gaps = MAX_RENDERED_GAP_LINES * 3
    t = 0.0
    for i in range(n_gaps):
        w.write_book("BTC/USD", t, 100.0 + i, 100.2 + i, 1.0, 1.0)
        w.write_trade("BTC/USD", t + 0.001, 100.1 + i, 0.5, "buy")
        w.write_gap(GapRecord(t + 0.001, t + 1.0, "clock_jump", detail=f"synthetic {i}"))
        t += 2.0
    w.write_book("BTC/USD", t, 500.0, 500.2, 1.0, 1.0)
    w.write_trade("BTC/USD", t + 0.001, 500.1, 0.5, "buy")
    w.close(clean=True)

    meta = read_capture(str(p))
    assert len(meta.gaps) == n_gaps
    report = render_report([meta], {}, {})

    gap_lines = [ln for ln in report.splitlines() if ln.strip().startswith("- gap: ")]
    listed = [ln for ln in gap_lines if "further recorded gap(s) not listed" not in ln]
    assert len(listed) == MAX_RENDERED_GAP_LINES
    assert f"and {n_gaps - MAX_RENDERED_GAP_LINES} further recorded gap(s) not listed" in report
    assert f"recorded gaps: {n_gaps} record(s) totalling" in report, (
        "the count is the claim; the individual lines are only examples of it")
    assert f"clock_jump={n_gaps}" in report, "with the per-reason breakdown, so nothing is hidden"
