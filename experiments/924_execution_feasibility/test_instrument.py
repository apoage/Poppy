"""TASK_924 — regression tests for `instrument.py`, synthetic inputs only.

D81: never writes into `experiments/907_live_fill_probe/`; imports `capture` read-only from
whatever directory `CAPTURE_907_DIR` names (defaulting to the sibling layout this task's
server-side scratch stages: `<repo_root>/907_live_fill_probe/` next to this file's parent, or
`../907_live_fill_probe` relative to this directory). Run on gpu-server, never bare on the
workstation (`.claude/rules/20-verification-rig.md`).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))


def _add_907_path() -> Path:
    env = os.environ.get("CAPTURE_907_DIR")
    candidates = []
    if env:
        candidates.append(Path(env))
    candidates += [
        _HERE.parent / "907_live_fill_probe",   # staged sibling (scratch layout)
        _HERE.parent.parent / "907_live_fill_probe",  # workstation repo layout
    ]
    for c in candidates:
        if (c / "capture.py").is_file():
            sys.path.insert(0, str(c))
            return c
    raise RuntimeError(
        f"could not find capture.py near any of {candidates} — set CAPTURE_907_DIR")


_add_907_path()

import instrument  # noqa: E402
from capture import CaptureWriter, read_capture  # noqa: E402


def _write_capture(path: Path, pairs, lines) -> None:
    """Write a well-formed capture: meta_start + given (type, kwargs) tuples, via the
    SHIPPED `CaptureWriter` (imported, not reimplemented) so the synthetic fixture is itself
    format-valid per 907's own writer.

    Declares `book_emission_policy`/`book_keepalive_s` exactly as the REAL capture's
    `meta_start` does (`"change_or_keepalive"` / `5.0`, verified against
    `capture_907_main.jsonl`'s own header on gpu-server) — the shipped reader's
    `stale_book_s` is `0.0` for ANY capture that omits these two declarations
    (`capture.py:2143-2146`), so a synthetic fixture that leaves them out cannot exercise
    gate G1's cross-check at all (it would silently compare 0.0 against 0.0)."""
    w = CaptureWriter(str(path), pairs, "test-924",
                       book_emission_policy="change_or_keepalive", book_keepalive_s=5.0)
    for kind, kw in lines:
        if kind == "book":
            w.write_book(**kw)
        elif kind == "trade":
            w.write_trade(**kw)
        elif kind == "heartbeat":
            w.write_heartbeat(kw["ts"])
        elif kind == "gap":
            from capture import GapRecord
            w.write_gap(GapRecord(kw["start_ts"], kw["end_ts"], kw["reason"]))
        else:
            raise ValueError(kind)
    w.close(clean=True)


# --------------------------------------------------------------------------------------------
# gap handling
# --------------------------------------------------------------------------------------------

def test_gap_seconds_excluded_from_accounted_and_two_sided(tmp_path):
    p = tmp_path / "cap.jsonl"
    pair = "BTC/USD"
    lines = [
        ("book", dict(pair=pair, ts=0.0, bid=100.0, ask=100.2, bid_sz=1.0, ask_sz=1.0)),
        ("book", dict(pair=pair, ts=10.0, bid=100.0, ask=100.2, bid_sz=1.0, ask_sz=1.0)),
        # 500s recorded gap: [10, 510]
        ("gap", dict(start_ts=10.0, end_ts=510.0, reason="disconnect")),
        ("book", dict(pair=pair, ts=510.0, bid=100.0, ask=100.2, bid_sz=1.0, ask_sz=1.0)),
        ("book", dict(pair=pair, ts=520.0, bid=100.0, ask=100.2, bid_sz=1.0, ask_sz=1.0)),
    ]
    _write_capture(p, [pair], lines)
    result = instrument.run_instrument(str(p))
    span_s = result.end_ts - result.start_ts
    assert span_s == pytest.approx(520.0, abs=1e-6)
    # 500s of the 520s span is recorded gap -> accounted_s_global == 20s
    assert result.accounted_s_global == pytest.approx(20.0, abs=1e-6)
    pp = result.per_pair[pair]
    # all four book events share the SAME (bid, ask) -> the whole 20s accounted time is
    # two-sided (bid_sz/ask_sz > 0 throughout), and none of the 500s gap leaks into it.
    assert pp["two_sided_seconds"] == pytest.approx(20.0, abs=1e-6)
    assert pp["accounted_s"] == pytest.approx(20.0, abs=1e-6)


def test_gap_time_never_attributed_to_forward_filled_state(tmp_path):
    """A book state that changes right before a gap must not have its PRE-gap value bleed
    into the post-gap accounted seconds, and vice versa — the forward-fill must stop dead at
    the gap boundary, not carry through it."""
    p = tmp_path / "cap.jsonl"
    pair = "ETH/USD"
    lines = [
        ("book", dict(pair=pair, ts=0.0, bid=10.0, ask=10.0, bid_sz=1.0, ask_sz=0.0)),  # one-sided
        ("gap", dict(start_ts=0.0, end_ts=1000.0, reason="disconnect")),
        ("book", dict(pair=pair, ts=1000.0, bid=10.0, ask=10.1, bid_sz=1.0, ask_sz=1.0)),
        ("book", dict(pair=pair, ts=1010.0, bid=10.0, ask=10.1, bid_sz=1.0, ask_sz=1.0)),
    ]
    _write_capture(p, [pair], lines)
    result = instrument.run_instrument(str(p))
    pp = result.per_pair[pair]
    # accounted = 10s (1000..1010), all two-sided.
    assert pp["accounted_s"] == pytest.approx(10.0, abs=1e-6)
    assert pp["two_sided_seconds"] == pytest.approx(10.0, abs=1e-6)


# --------------------------------------------------------------------------------------------
# two-sided time-weighting, hand-computed
# --------------------------------------------------------------------------------------------

def test_two_sided_time_weighting_hand_computed(tmp_path):
    p = tmp_path / "cap.jsonl"
    pair = "SOL/USD"
    lines = [
        ("book", dict(pair=pair, ts=0.0, bid=1.0, ask=1.02, bid_sz=1.0, ask_sz=1.0)),   # two-sided for next 100s
        ("book", dict(pair=pair, ts=100.0, bid=1.0, ask=1.02, bid_sz=1.0, ask_sz=0.0)),  # one-sided for next 50s
        ("book", dict(pair=pair, ts=150.0, bid=1.0, ask=1.02, bid_sz=1.0, ask_sz=1.0)),  # two-sided until end
        ("book", dict(pair=pair, ts=200.0, bid=1.0, ask=1.02, bid_sz=1.0, ask_sz=1.0)),
    ]
    _write_capture(p, [pair], lines)
    result = instrument.run_instrument(str(p))
    pp = result.per_pair[pair]
    assert pp["accounted_s"] == pytest.approx(200.0, abs=1e-6)
    # two-sided seconds: [0,100) + [150,200) = 150s
    assert pp["two_sided_seconds"] == pytest.approx(150.0, abs=1e-6)
    assert pp["two_sided_time_weighted_fraction"] == pytest.approx(0.75, abs=1e-9)


# --------------------------------------------------------------------------------------------
# frozen-run detection agrees with the shipped 907 definition (G1's cross-check, instantiated)
# --------------------------------------------------------------------------------------------

def test_frozen_run_matches_shipped_definition_on_planted_run(tmp_path):
    p = tmp_path / "cap.jsonl"
    pair = "BTC/USD"
    # A run of identical (bid, ask) lasting 120s (> STALE_BOOK_MIN_RUN_S == 60s), preceded and
    # followed by a changing price so the run boundaries are unambiguous.
    lines = [
        ("book", dict(pair=pair, ts=0.0, bid=100.0, ask=100.2, bid_sz=1.0, ask_sz=1.0)),
        ("book", dict(pair=pair, ts=1.0, bid=99.0, ask=99.2, bid_sz=1.0, ask_sz=1.0)),
        ("book", dict(pair=pair, ts=1.5, bid=99.0, ask=99.2, bid_sz=1.0, ask_sz=1.0)),
        ("book", dict(pair=pair, ts=121.5, bid=99.0, ask=99.2, bid_sz=1.0, ask_sz=1.0)),
        ("book", dict(pair=pair, ts=122.0, bid=98.0, ask=98.2, bid_sz=1.0, ask_sz=1.0)),
    ]
    _write_capture(p, [pair], lines)

    result = instrument.run_instrument(str(p))
    g1 = result.g1[pair]
    # Run start is the FIRST event carrying the new top (ts=1.0, matching capture.py's own
    # `_note_book_top` semantics), last is the final repeat (ts=121.5) -> 120.5s > 60s -> counted.
    assert g1["indep_stale_s"] == pytest.approx(120.5, abs=1e-6)

    # Cross-check directly against the SHIPPED reader on the identical file (imported, not
    # reimplemented) — this is gate G1's cross-check instantiated as a test.
    shipped_meta = read_capture(str(p))
    shipped_stale = shipped_meta.coverage_by_pair[pair].stale_book_s
    assert shipped_stale == pytest.approx(g1["indep_stale_s"], abs=1e-6)
    assert shipped_stale == pytest.approx(g1["shipped_stale_s"], abs=1e-9)


def test_oscillating_mid_is_NOT_flagged_frozen(tmp_path):
    """Disclosed B1 blind spot (capture.py:217-231), carried forward deliberately: a mid that
    ALTERNATES between two values for the whole file must read stale_s == 0, exactly as 907's
    own `read_capture` does on the identical shape. This test guards the disclosure staying
    true, not a desirable property."""
    p = tmp_path / "cap.jsonl"
    pair = "XRP/USD"
    lines = []
    t = 0.0
    for i in range(400):
        if i % 2 == 0:
            lines.append(("book", dict(pair=pair, ts=t, bid=1.0, ask=1.01,
                                        bid_sz=1.0, ask_sz=1.0)))
        else:
            lines.append(("book", dict(pair=pair, ts=t, bid=1.02, ask=1.03,
                                        bid_sz=1.0, ask_sz=1.0)))
        t += 1.0  # 400s total, alternating every 1s -> no run ever exceeds 60s
    _write_capture(p, [pair], lines)

    result = instrument.run_instrument(str(p))
    g1 = result.g1[pair]
    assert g1["indep_stale_s"] == 0.0
    assert g1["shipped_stale_s"] == 0.0

    shipped_meta = read_capture(str(p))
    assert pair not in shipped_meta.stale_book_pairs()


# --------------------------------------------------------------------------------------------
# item 3 — book-presence row classification (event-row denominated)
# --------------------------------------------------------------------------------------------

def test_row_classification_two_sided_one_sided_print_only(tmp_path):
    p = tmp_path / "cap.jsonl"
    pair = "BTC/USD"
    lines = [
        ("book", dict(pair=pair, ts=0.0, bid=100.0, ask=100.2, bid_sz=1.0, ask_sz=1.0)),  # two-sided
        ("book", dict(pair=pair, ts=1.0, bid=100.0, ask=100.2, bid_sz=0.0, ask_sz=1.0)),  # one-sided
        ("book", dict(pair=pair, ts=2.0, bid=100.0, ask=100.2, bid_sz=1.0, ask_sz=0.0)),  # one-sided
        ("trade", dict(pair=pair, ts=3.0, price=100.1, volume=0.1, side="buy")),          # print-only
    ]
    _write_capture(p, [pair], lines)
    result = instrument.run_instrument(str(p))
    pp = result.per_pair[pair]
    assert pp["n_two_sided_rows"] == 1
    assert pp["n_one_sided_rows"] == 2
    assert pp["n_trade_rows"] == 1
    assert pp["n_book_rows"] == 3


# --------------------------------------------------------------------------------------------
# G2 read-only proof helper
# --------------------------------------------------------------------------------------------

def test_fingerprint_is_stable_across_repeated_reads(tmp_path):
    p = tmp_path / "cap.jsonl"
    _write_capture(p, ["BTC/USD"],
                    [("book", dict(pair="BTC/USD", ts=0.0, bid=1.0, ask=1.01,
                                    bid_sz=1.0, ask_sz=1.0))])
    fp1 = instrument.file_fingerprint(str(p))
    instrument.run_instrument(str(p))
    fp2 = instrument.file_fingerprint(str(p))
    assert fp1 == fp2


# --------------------------------------------------------------------------------------------
# POST-HOC (SPEC ERRATA 2, cycle 2) — G2 fires on a planted write
# --------------------------------------------------------------------------------------------

def _g2_run(tmp_path, monkeypatch, tamper):
    """Drive the REAL gate code (`run_full_pass.main`) with `instrument.run_instrument`
    wrapped so `tamper(path)` runs after the real pass and before the after-fingerprint."""
    import run_full_pass
    p = tmp_path / "cap.jsonl"
    pair = "BTC/USD"
    _write_capture(p, [pair], [
        ("book", dict(pair=pair, ts=float(t), bid=100.0 + (t % 3), ask=100.2 + (t % 3),
                      bid_sz=1.0, ask_sz=1.0)) for t in range(30)])
    spec = tmp_path / "SPEC.md"
    spec.write_text("spec\n")
    out = tmp_path / "out.json"
    real = instrument.run_instrument

    def wrapped(path):
        res = real(path)
        if tamper is not None:
            tamper(path)
        return res

    monkeypatch.setattr(instrument, "run_instrument", wrapped)
    rc = run_full_pass.main(["--capture", str(p), "--spec", str(spec), "--out", str(out),
                              "--label", "g2_test"])
    return rc, json.loads(out.read_text())


def test_g2_control_no_write_is_read_only_ok(tmp_path, monkeypatch):
    rc, out = _g2_run(tmp_path, monkeypatch, None)
    assert out["g2_read_only_ok"] is True
    assert rc == 0


def test_g2_fires_on_planted_append(tmp_path, monkeypatch):
    def append(path):
        with open(path, "ab") as fh:
            fh.write(b"\n")
    rc, out = _g2_run(tmp_path, monkeypatch, append)
    assert out["g2_read_only_ok"] is False
    assert out["g2_after"]["size"] == out["g2_before"]["size"] + 1
    assert rc == 1


def test_g2_fires_on_same_size_write_with_mtime_restored(tmp_path, monkeypatch):
    """Where size+mtime are BLIND: an in-place same-length byte change with the mtime put
    back. Only the sha256 component can catch it — prove it does."""
    def flip(path):
        st = os.stat(path)
        with open(path, "r+b") as fh:
            data = fh.read()
            i = data.index(b"100.2")
            fh.seek(i)
            fh.write(b"100.3")
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))
    rc, out = _g2_run(tmp_path, monkeypatch, flip)
    assert out["g2_after"]["size"] == out["g2_before"]["size"]
    assert out["g2_after"]["mtime"] == out["g2_before"]["mtime"]
    assert out["g2_after"]["sha256"] != out["g2_before"]["sha256"]
    assert out["g2_read_only_ok"] is False
    assert rc == 1


# --------------------------------------------------------------------------------------------
# POST-HOC (SPEC ERRATA 2) — a stale run straddling a recorded gap
# --------------------------------------------------------------------------------------------

def test_stale_run_straddling_gap_is_split_and_gap_seconds_excluded(tmp_path):
    """Run of identical (bid, ask) from ts=1 to ts=330 with a recorded gap [50, 250] inside it.
    Keepalive-style repeats every 5s on both sides of the gap.
      span = [0, 331], accounted = 331 - 200 = 131 s
      SHIPPED definition (gap-blind): one run, 330 - 1 = 329 s > 60 -> stale 329 s
        (329/131 > 1: a "fraction" above 100% — the defect in its plainest form)
      GAP-CLIPPED: split at the gap -> pieces [1, 50] = 49 s (< 60, not stale) and
        [250, 330] = 80 s (> 60, stale) -> stale 80 s
      UNSPLIT reconciliation (gap seconds dropped, shipped per-run threshold): 329 - 200 = 129 s
    """
    p = tmp_path / "cap.jsonl"
    pair = "BTC/USD"
    A = dict(bid=99.0, ask=99.2, bid_sz=1.0, ask_sz=1.0)
    lines = [("book", dict(pair=pair, ts=0.0, bid=100.0, ask=100.2, bid_sz=1.0, ask_sz=1.0))]
    lines += [("book", dict(pair=pair, ts=float(t), **A)) for t in [1] + list(range(5, 51, 5))]
    lines.append(("gap", dict(start_ts=50.0, end_ts=250.0, reason="disconnect")))
    lines += [("book", dict(pair=pair, ts=float(t), **A)) for t in range(250, 331, 5)]
    lines.append(("book", dict(pair=pair, ts=331.0, bid=98.0, ask=98.2, bid_sz=1.0,
                               ask_sz=1.0)))
    _write_capture(p, [pair], lines)

    result = instrument.run_instrument(str(p))
    assert result.accounted_s_global == pytest.approx(131.0, abs=1e-6)
    fm = result.per_pair[pair]["frozen_mid_post_hoc"]

    # shipped definition — equals 907's own reader on the identical file (G1 quantity)
    shipped = read_capture(str(p)).coverage_by_pair[pair].stale_book_s
    assert shipped == pytest.approx(329.0, abs=1e-6)
    assert fm["stale_s_shipped_definition"] == pytest.approx(shipped, abs=1e-9)
    assert result.g1[pair]["indep_stale_s"] == pytest.approx(shipped, abs=1e-9)
    assert fm["frozen_mid_shipped_definition"] == pytest.approx(329.0 / 131.0, abs=1e-9)

    # gap-clipped — gap seconds excluded AND the run split, 49 s piece below threshold
    assert fm["stale_s_gap_clipped"] == pytest.approx(80.0, abs=1e-6)
    assert fm["frozen_mid_gap_clipped"] == pytest.approx(80.0 / 131.0, abs=1e-9)
    assert fm["streaks_gap_clipped"]["n"] == 1
    assert fm["streaks_gap_clipped"]["max_s"] == pytest.approx(80.0, abs=1e-6)
    assert fm["streaks_shipped_definition"]["max_s"] == pytest.approx(329.0, abs=1e-6)
    assert fm["n_stale_runs_straddling_gap"] == 1

    # the disclosed difference: 200 s of gap + the 49 s sub-threshold pre-gap piece
    assert shipped - fm["stale_s_gap_clipped"] == pytest.approx(249.0, abs=1e-6)
    assert fm["stale_s_gap_intersected_unsplit_RECONCILIATION"] == pytest.approx(129.0, abs=1e-6)
    assert fm["stale_s_inside_recorded_gaps_shipped_minus_unsplit"] == pytest.approx(
        200.0, abs=1e-6)

    # survival pieces are gap-split too: pieces 49 and 80 (the 0->1 run is a singleton)
    tw = result.per_pair[pair]["quote_survival_post_hoc"]["time_weighted_gap_clipped"]
    assert tw["total_run_s"] == pytest.approx(129.0, abs=1e-6)


# --------------------------------------------------------------------------------------------
# POST-HOC (SPEC ERRATA 2) — time-weighted vs count-weighted survival, hand-computed
# --------------------------------------------------------------------------------------------

def test_time_weighted_quantiles_hand_computed():
    # durations 1,1,1,1,6: W = 10. Cumulative 1,2,3,4,10.
    #   p10 target 1.0 -> 1 ; p25 target 2.5 -> 1 ; p50 target 5 -> 6 ; p75 7.5 -> 6 ; p90 9 -> 6
    #   mean = (1+1+1+1+36)/10 = 4.0   vs count-weighted median 1, mean 2.0
    tw = instrument.time_weighted_quantiles([1.0, 1.0, 6.0, 1.0, 1.0, 0.0])
    assert tw["p10_s"] == 1.0 and tw["p25_s"] == 1.0
    assert tw["p50_s"] == 6.0 and tw["p75_s"] == 6.0 and tw["p90_s"] == 6.0
    assert tw["mean_s"] == pytest.approx(4.0)
    cw = instrument.count_weighted_quantiles([1.0, 1.0, 6.0, 1.0, 1.0])
    assert cw["median_s"] == 1.0 and cw["mean_s"] == pytest.approx(2.0)


def test_time_vs_count_weighted_survival_end_to_end(tmp_path):
    """Same 1,1,1,1,6 run set, built as a real capture and read through `run_instrument`."""
    p = tmp_path / "cap.jsonl"
    pair = "ETH/USD"
    tops = [(10.0, 10.1), (11.0, 11.1), (12.0, 12.1), (13.0, 13.1), (14.0, 14.1), (15.0, 15.1)]
    times = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 14), (15,)]
    lines = []
    for (b, a), ts in zip(tops, times):
        for t in ts:
            lines.append(("book", dict(pair=pair, ts=float(t), bid=b, ask=a, bid_sz=1.0,
                                       ask_sz=1.0)))
    _write_capture(p, [pair], lines)
    result = instrument.run_instrument(str(p))
    qs = result.per_pair[pair]["quote_survival_post_hoc"]
    tw = qs["time_weighted_gap_clipped"]
    assert tw["p50_s"] == pytest.approx(6.0) and tw["mean_s"] == pytest.approx(4.0)
    cw = qs["count_weighted_gap_clipped_SECONDARY"]
    assert cw["median_s"] == pytest.approx(1.0) and cw["n_runs"] == 5
    # pre-registered count-weighted (gap-blind) survival unchanged in shape
    assert result.per_pair[pair]["quote_survival"]["median_s"] == pytest.approx(1.0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
