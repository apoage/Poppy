import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture import CaptureFormatError, CaptureWriter, GapRecord  # noqa: E402
from shadow_engine import MidSeries, run_shadow  # noqa: E402


def _base_capture(tmp_path, extra=None):
    """bar0 [0,60): trades at ts=10 (100), ts=50 (102) -> close=102, high=102@50, low=100@10
       bar1 [60,120): trades at ts=70 (101), ts=90 (99) -> close=99, high=101@70, low=99@90
       book: ts=90 mid=100.1, ts=145 mid=98.1 (used for the 1m markout of a fill at ts=90)
    """
    p = tmp_path / "cap.jsonl"
    w = CaptureWriter(str(p), ["BTC/USD"], "cap-1")
    w.write_trade("BTC/USD", 10.0, 100.0, 1.0, "buy")
    w.write_trade("BTC/USD", 50.0, 102.0, 1.0, "sell")
    w.write_trade("BTC/USD", 70.0, 101.0, 1.0, "buy")
    w.write_trade("BTC/USD", 90.0, 99.0, 1.0, "sell")
    w.write_book("BTC/USD", 90.0, 100.0, 100.2, 1.0, 1.0)
    w.write_book("BTC/USD", 145.0, 98.0, 98.2, 1.0, 1.0)
    # extends the capture's recorded span past the 1m-horizon target (fill_ts=90 -> t=150) so
    # gate (iv)'s `covers()` check isn't tripped merely by the file ending — the as-of mid at
    # t=150 still resolves to the 145.0 snapshot (98.1), the last one at/before 150.
    w.write_book("BTC/USD", 200.0, 98.0, 98.2, 1.0, 1.0)
    if extra:
        extra(w)
    w.close(clean=True)
    return str(p)


def test_touch_fill_and_markout_sign(tmp_path):
    path = _base_capture(tmp_path)
    meta, results = run_shadow(path)
    r = results["BTC/USD"]

    # bar0 close=102 posted at post_ts=60; next bar (bar1) low=99 <= 102 -> BUY touch-fills;
    # bar1 high=101 < 102 -> SELL does not touch-fill.
    assert r.n_posted == 2
    assert r.n_touch_fills == 1
    assert r.n_trade_cross_fills == 1  # buy crosses (99 and 101 both <= 102); sell doesn't

    ones = [m for m in r.markouts if m.horizon == "1m"]
    assert len(ones) == 1
    m = ones[0]
    assert m.side == 1
    assert m.fill_price == 102.0
    assert m.fill_ts == 90.0  # bar1's low_ts, since side=+1 fills against the low
    expected_signed = 1 * (98.1 - 102.0) / 102.0 * 1e4
    assert m.adverse_selection_bp == pytest.approx(-expected_signed)
    assert m.adverse_selection_bp > 0  # price fell after a buy fill -> adverse, as expected

    # no book/trade data far enough past ts=145 for 5m/20m/1h horizons -> dropped, not faked
    assert not [x for x in r.markouts if x.horizon in ("5m", "20m", "1h")]


def test_markout_dropped_when_it_straddles_a_recorded_gap(tmp_path):
    def add_gap(w):
        w.write_gap(GapRecord(90.0, 200.0, "disconnect", detail="test gap covering the 1m horizon"))
    path = _base_capture(tmp_path, extra=add_gap)
    meta, results = run_shadow(path)
    r = results["BTC/USD"]
    assert r.n_touch_fills == 1
    # the fill at ts=90 with a 1m target of 150 now straddles the recorded [90,200] gap
    assert not [m for m in r.markouts if m.horizon == "1m"]


def test_missing_next_bar_is_skipped_not_treated_as_unfilled(tmp_path):
    p = tmp_path / "cap.jsonl"
    w = CaptureWriter(str(p), ["BTC/USD"], "cap-1")
    w.write_trade("BTC/USD", 10.0, 100.0, 1.0, "buy")   # bar0 [0,60)
    # bar1 [60,120) has NO trades at all — a hole in the per-pair data
    w.write_trade("BTC/USD", 130.0, 100.0, 1.0, "buy")  # bar2 [120,180)
    w.close(clean=True)

    meta, results = run_shadow(str(p))
    r = results["BTC/USD"]
    assert r.n_skipped_no_next_bar == 1
    assert r.n_posted == 0  # bar0's posting was skipped because bar1 has no data to judge it


# --- BLOCKER 1: non-finite / zero prices must refuse at INGESTION, via CaptureFormatError ---

def test_gap_straddling_and_no_mid_drops_are_counted_not_silent(tmp_path):
    """Every dropped markout candidate is attributed to a reason with a count."""
    def add_gap(w):
        w.write_gap(GapRecord(90.0, 200.0, "disconnect", detail="covers the 1m horizon"))
    path = _base_capture(tmp_path, extra=add_gap)
    meta, results = run_shadow(path)
    r = results["BTC/USD"]
    # 1 touch-fill x 4 horizons = 4 candidates, all dropped: 1m straddles the gap, and
    # 5m/20m/1h fall past the capture's recorded span.
    total_dropped = (r.n_markouts_dropped_gap + r.n_markouts_dropped_outside_span
                     + r.n_markouts_dropped_no_mid + r.n_markouts_dropped_nonfinite)
    assert total_dropped == 4
    # 1m (target 150) lands INSIDE the recorded [90,200] gap -> a real gap drop; 5m/20m/1h run
    # past the capture's own end (200) -> attributed separately, because "we stopped recording"
    # is a different fact about the data than "the wire went dark".
    assert r.n_markouts_dropped_gap == 1
    assert r.n_markouts_dropped_outside_span == 3
    assert r.n_markouts_dropped_nonfinite == 0
    assert r.markouts == []


def test_nan_book_quote_in_a_capture_refuses_rather_than_producing_a_markout(tmp_path):
    """The reproduced BLOCKER-1 path: a single book snapshot with NaN bid/ask (Python's json
    accepts bare NaN literals) used to yield mid_then=nan -> adverse_selection_bp=nan -> a full
    row of nans plus a false 'BELOW measured' verdict."""
    p = tmp_path / "cap.jsonl"
    with open(p, "w") as fh:
        fh.write(json.dumps({"type": "meta_start", "capture_id": "x", "pairs": ["BTC/USD"],
                              "format_version": 1}) + "\n")
        fh.write(json.dumps({"type": "trade", "pair": "BTC/USD", "ts": 10.0, "price": 100.0,
                              "volume": 1.0, "side": "buy"}) + "\n")
        fh.write(json.dumps({"type": "trade", "pair": "BTC/USD", "ts": 70.0, "price": 99.0,
                              "volume": 1.0, "side": "sell"}) + "\n")
        fh.write('{"type":"book","pair":"BTC/USD","ts":150.0,"bid":NaN,"ask":NaN,'
                  '"bid_sz":1.0,"ask_sz":1.0}\n')
    with pytest.raises(CaptureFormatError, match="non-finite"):
        run_shadow(str(p))


def test_zero_price_refuses_through_captureformaterror_not_zerodivisionerror(tmp_path):
    """MINOR 1: a `price: 0.0` print is the divisor of the markout. It must refuse through this
    module's declared error type, not surface as a raw ZeroDivisionError traceback."""
    p = tmp_path / "cap.jsonl"
    with open(p, "w") as fh:
        fh.write(json.dumps({"type": "meta_start", "capture_id": "x", "pairs": ["BTC/USD"],
                              "format_version": 1}) + "\n")
        fh.write(json.dumps({"type": "trade", "pair": "BTC/USD", "ts": 10.0, "price": 0.0,
                              "volume": 1.0, "side": "buy"}) + "\n")
        fh.write(json.dumps({"type": "trade", "pair": "BTC/USD", "ts": 70.0, "price": 100.0,
                              "volume": 1.0, "side": "sell"}) + "\n")
    with pytest.raises(CaptureFormatError, match="strictly positive"):
        run_shadow(str(p))


def test_midseries_refuses_nonfinite_and_nonpositive_inputs():
    ms = MidSeries()
    with pytest.raises(CaptureFormatError, match="non-finite"):
        ms.add_book(1.0, float("nan"), 100.0)
    with pytest.raises(CaptureFormatError, match="non-finite"):
        ms.add_book(float("inf"), 100.0, 100.2)
    with pytest.raises(CaptureFormatError, match="strictly positive"):
        ms.add_trade(1.0, 0.0)
    with pytest.raises(CaptureFormatError, match="non-finite"):
        ms.add_trade(1.0, float("inf"))
    assert ms.mid_at(1.0) is None       # nothing bad was admitted
    assert ms.mid_at(float("nan")) is None  # a non-finite as-of time never resolves to a mid


def test_nonfinite_mid_reaching_the_markout_loop_is_dropped_and_counted(tmp_path):
    """Defense-in-depth branch, exercised directly. `MidSeries` refuses non-finite quotes at
    ingestion, so this can only be reached if a FUTURE code path feeds one in — which is exactly
    why the guard exists. Injecting into the internal arrays proves it drops AND counts rather
    than letting a NaN become a markout."""
    from shadow_engine import MidQuote, MidSeries as _MS
    path = _base_capture(tmp_path)

    real_mid_at = _MS.mid_at
    calls = {"n": 0}

    def poisoned_mid_at(self, t):
        calls["n"] += 1
        # simulate a future path yielding a non-finite mid, correctly tagged
        return MidQuote(mid=float("nan"), source="book_mid", age_s=0.0)

    _MS.mid_at = poisoned_mid_at
    try:
        meta, results = run_shadow(path)
    finally:
        _MS.mid_at = real_mid_at

    r = results["BTC/USD"]
    assert calls["n"] > 0, "the poisoned lookup must actually have been consulted"
    assert r.markouts == [], "a non-finite mid must never become a markout"
    assert r.n_markouts_dropped_nonfinite > 0, "and the drop must be COUNTED, not silent"


# --- MINOR (run-44): the mutually-masking guard pair, FOLDED into one ---------------------------

def test_the_single_nonfinite_guard_covers_both_reach_paths(tmp_path):
    """There used to be TWO non-finiteness guards here — one on `mid_then`, one on the computed
    `signed_return`. Removing either ALONE left the suite green (guard 1 was a strict superset of
    guard 2, given `limit.price` is independently validated finite and strictly positive), so
    neither was covered on its own and a future "de-duplicating" refactor would have shipped
    silently. They are now ONE check on the result, and this test asserts that single check
    catches every shape that could reach it: nan, +inf, -inf.

    Named by the comment at the guard site, so deleting this test breaks that reference.
    """
    from shadow_engine import MidQuote, MidSeries as _MS
    path = _base_capture(tmp_path)
    real_mid_at = _MS.mid_at

    for bad in (float("nan"), float("inf"), float("-inf")):
        _MS.mid_at = lambda self, t, _b=bad: MidQuote(mid=_b, source="book_mid", age_s=0.0)
        try:
            _, results = run_shadow(path)
        finally:
            _MS.mid_at = real_mid_at
        r = results["BTC/USD"]
        assert r.markouts == [], f"a mid of {bad} must never become a markout"
        assert r.n_markouts_dropped_nonfinite > 0, f"the {bad} drop must be COUNTED"
        assert r.n_markouts_from_book_mid == 0, (
            "a dropped candidate must not be counted into the mid-source split either — the "
            "split has to reconcile with the samples that actually exist")


def test_run_shadow_always_sets_the_mid_source_explicitly(tmp_path):
    """`MarkoutSample.mid_source` has a default (so the render tests can hand-build samples), so
    pin that the engine never RELIES on it: every sample it emits carries an explicit,
    recognized source, and the counters reconcile with the samples."""
    from shadow_engine import MID_SOURCE_BOOK, MID_SOURCE_TRADE
    path = _base_capture(tmp_path)
    _, results = run_shadow(path)
    r = results["BTC/USD"]
    assert r.markouts, "the fixture must produce at least one sample for this to mean anything"
    for m in r.markouts:
        assert m.mid_source in (MID_SOURCE_BOOK, MID_SOURCE_TRADE)
        assert m.mid_age_s >= 0.0
    assert (r.n_markouts_from_book_mid + r.n_markouts_from_trade_fallback) == len(r.markouts)
