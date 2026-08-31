"""TASK_907 run-44 MAJOR — the mid-source blend must be tagged, counted, and unpooled.

`MidSeries.mid_at` prefers top-of-book mid and falls back to the last trade print. Before this
change the fallback was disclosed ONLY in a code comment: `MarkoutSample` carried no source
field, nothing in `shadow_engine`/`estimands`/`render` wired one, and a trade print therefore
entered the flagship pre-registered estimand indistinguishably from a real mid.

A print is not a mid — it is struck at the aggressor's side of the spread — and its bias is not
mean-zero where it matters: after a fill going badly, aggressor flow is one-sided with the move,
so the print sits on the far side of the mid and measured adverse selection comes out too HIGH.
"Too high" is the direction of this row's own preferred conclusion ("even the OPTIMISTIC bound
meets the model's 5 bp, so the model stands calibrated-pessimistic"), which is precisely why the
fallback is excluded from the primary estimand rather than merely footnoted.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture import CaptureWriter, read_capture  # noqa: E402
from estimands import (  # noqa: E402
    compute_estimand1, compute_estimand1_trade_fallback, compute_estimand2,
)
from render import render_report  # noqa: E402
from shadow_engine import (  # noqa: E402
    BAR_SECONDS, MAX_MID_STALENESS_S, MID_SOURCE_BOOK, MID_SOURCE_TRADE, MarkoutSample,
    MidSeries, ShadowPairResult, run_shadow,
)


def _capture(tmp_path, books=(), name="cap.jsonl"):
    """bar0 [0,60): trades 10@100, 50@102 -> close 102. bar1 [60,120): 70@101, 90@99.
    A BUY limit at 102 touch-fills against bar1's low at ts=90, so the 1m horizon is t=150.
    `books` are added verbatim; a heartbeat at 200 extends the span past the horizon so
    `covers()` is satisfied and the drop reason under test is the mid, not the span."""
    p = tmp_path / name
    w = CaptureWriter(str(p), ["BTC/USD"], "cap-1")
    for ts, px in ((10.0, 100.0), (50.0, 102.0), (70.0, 101.0), (90.0, 99.0), (145.0, 98.0)):
        w.write_trade("BTC/USD", ts, px, 1.0, "sell")
    for ts, bid, ask in books:
        w.write_book("BTC/USD", ts, bid, ask, 1.0, 1.0)
    w.write_heartbeat(200.0)
    w.close(clean=True)
    return str(p)


# --- the engine tags every sample ---------------------------------------------------------------

def test_a_book_backed_markout_is_tagged_book_mid_with_its_age(tmp_path):
    path = _capture(tmp_path, books=[(145.0, 98.0, 98.2)])
    _, results = run_shadow(path)
    r = results["BTC/USD"]
    ones = [m for m in r.markouts if m.horizon == "1m"]
    assert len(ones) == 1
    assert ones[0].mid_source == MID_SOURCE_BOOK
    assert ones[0].mid_age_s == pytest.approx(5.0)   # book at 145, horizon at 150
    assert r.n_markouts_from_book_mid == 1
    assert r.n_markouts_from_trade_fallback == 0
    assert r.trade_fallback_share == 0.0


def test_a_trade_print_backed_markout_is_tagged_as_the_fallback(tmp_path):
    """No book quote anywhere, so the as-of "mid" is the 145.0 trade print. Before the fix this
    sample was indistinguishable from a book-mid sample in every downstream structure."""
    path = _capture(tmp_path, books=[])
    _, results = run_shadow(path)
    r = results["BTC/USD"]
    ones = [m for m in r.markouts if m.horizon == "1m"]
    assert len(ones) == 1
    assert ones[0].mid_source == MID_SOURCE_TRADE
    assert ones[0].mid_age_s == pytest.approx(5.0)
    assert r.n_markouts_from_book_mid == 0
    assert r.n_markouts_from_trade_fallback == 1
    assert r.trade_fallback_share == 1.0


def test_the_book_preference_is_real_and_the_fallback_only_fires_when_book_is_stale(tmp_path):
    """A book quote 62s before the horizon is beyond the staleness cap, so the fresher trade
    print stands in — and is tagged as such rather than presented as a mid."""
    path = _capture(tmp_path, books=[(88.0, 98.0, 98.2)])   # 150 - 88 = 62s > cap
    _, results = run_shadow(path)
    r = results["BTC/USD"]
    assert [m.mid_source for m in r.markouts if m.horizon == "1m"] == [MID_SOURCE_TRADE]


def test_the_mid_staleness_cap_is_one_bar_interval(tmp_path):
    """Tightened from 300s in run-44: a mid observed five minutes before its horizon was being
    presented as the mid AT that horizon, backing a markout whose shortest pre-registered horizon
    is 60s. The per-sample counterpart of the aggregate coverage bound."""
    assert MAX_MID_STALENESS_S == BAR_SECONDS

    ms = MidSeries()
    ms.add_book(100.0, 99.0, 101.0)
    assert ms.mid_at(100.0 + BAR_SECONDS).source == MID_SOURCE_BOOK   # exactly at the cap: used
    assert ms.mid_at(100.0 + BAR_SECONDS + 0.001) is None             # a hair past: missing
    # the same boundary for the fallback series
    ms2 = MidSeries()
    ms2.add_trade(100.0, 99.0)
    assert ms2.mid_at(100.0 + BAR_SECONDS).source == MID_SOURCE_TRADE
    assert ms2.mid_at(100.0 + BAR_SECONDS + 0.001) is None

    # and the drop is COUNTED, not silent: beyond the cap the candidate lands in
    # n_markouts_dropped_no_mid rather than being marked out against a stale observation.
    p = tmp_path / "stale.jsonl"
    w = CaptureWriter(str(p), ["BTC/USD"], "c")
    for ts, px in ((10.0, 100.0), (50.0, 102.0), (70.0, 101.0), (90.0, 99.0)):
        w.write_trade("BTC/USD", ts, px, 1.0, "sell")
    w.write_heartbeat(400.0)     # extends the span past the 5m horizon, but carries no price
    w.close(clean=True)
    _, results = run_shadow(str(p))
    r = results["BTC/USD"]
    assert r.n_touch_fills == 1
    # 1m (target 150): the fill bar's OWN last print at ts=90 is exactly 60s stale, i.e. exactly
    # at the cap — so it is used, and it is tagged as the FALLBACK, which is why excluding the
    # fallback from the primary estimand matters: without that exclusion a book-less pair's 1m
    # markout would be measured against the fill bar's own print and reported as a mid-return.
    ones = [m for m in r.markouts if m.horizon == "1m"]
    assert len(ones) == 1 and ones[0].mid_source == MID_SOURCE_TRADE
    assert ones[0].mid_age_s == pytest.approx(BAR_SECONDS)
    # B3 (run-45): all four pre-registered horizons are always reported now (never omitted), so
    # the primary (book-mid) estimand still shows "1m" — but with n_used=0, since the only 1m
    # sample here is TRADE-sourced and stays out of the primary estimand.
    e1 = compute_estimand1(r)
    assert set(e1.by_horizon) == {"1m", "5m", "20m", "1h"}
    assert e1.by_horizon["1m"].n_used == 0, "and it stays out of the primary estimand"
    # 5m (target 390): the last print is 300s stale, far past the cap -> dropped and counted
    assert [m for m in r.markouts if m.horizon == "5m"] == []
    assert r.n_markouts_dropped_no_mid >= 1


# --- estimand 1 is book-mid only ----------------------------------------------------------------

def _mixed_result():
    r = ShadowPairResult(pair="BTC/USD")
    for v in (1.0, 1.0, 1.0):
        r.markouts.append(MarkoutSample(pair="BTC/USD", side=1, fill_price=100.0, fill_ts=0.0,
                                         horizon="1m", adverse_selection_bp=v,
                                         mid_source=MID_SOURCE_BOOK, mid_age_s=2.0))
    for v in (100.0, 100.0):
        r.markouts.append(MarkoutSample(pair="BTC/USD", side=1, fill_price=100.0, fill_ts=0.0,
                                         horizon="1m", adverse_selection_bp=v,
                                         mid_source=MID_SOURCE_TRADE, mid_age_s=40.0))
    r.n_markouts_from_book_mid = 3
    r.n_markouts_from_trade_fallback = 2
    return r


def test_the_primary_estimand_excludes_the_trade_print_fallback():
    """The two populations must never be pooled. If they were, this median would be 1.0 -> 100.0
    territory purely from the substitution — and in the pessimistic direction."""
    e1 = compute_estimand1(_mixed_result())
    assert e1.mid_source == MID_SOURCE_BOOK
    h = e1.by_horizon["1m"]
    assert h.n_used == 3
    assert h.median_bp == pytest.approx(1.0)
    assert e1.n_excluded_other_source == 2
    assert h.mid_age_p50_s == pytest.approx(2.0)

    e1f = compute_estimand1_trade_fallback(_mixed_result())
    assert e1f.mid_source == MID_SOURCE_TRADE
    hf = e1f.by_horizon["1m"]
    assert hf.n_used == 2
    assert hf.median_bp == pytest.approx(100.0)
    assert e1f.n_excluded_other_source == 3
    assert hf.mid_age_p50_s == pytest.approx(40.0)


def test_estimand2_carries_the_source_split():
    e2 = compute_estimand2(_mixed_result())
    assert e2.n_markouts_from_book_mid == 3
    assert e2.n_markouts_from_trade_fallback == 2
    assert e2.trade_fallback_share == pytest.approx(0.4)


# --- the artifact says all of it ----------------------------------------------------------------

def _complete_meta(tmp_path):
    p = tmp_path / "meta.jsonl"
    w = CaptureWriter(str(p), ["BTC/USD"], "cap-1")
    w.write_trade("BTC/USD", 10.0, 100.0, 1.0, "buy")
    w.write_book("BTC/USD", 10.0, 100.0, 100.2, 1.0, 1.0)
    w.close(clean=True)
    return read_capture(str(p))


def test_the_artifact_states_the_source_split_and_the_fallback_bias_direction(tmp_path):
    meta = _complete_meta(tmp_path)
    r = _mixed_result()
    report = render_report(
        [meta], {"BTC/USD": compute_estimand1(r)}, {"BTC/USD": compute_estimand2(r)},
        estimand1_trade_fallback_by_pair={"BTC/USD": compute_estimand1_trade_fallback(r)})

    # the split, as counts
    assert "3 from BOOK MID" in report
    assert "2 from the TRADE-PRINT fallback" in report
    assert "40.0% fallback share" in report
    # material share -> prominent
    assert "MATERIAL: 40.0% of this pair's markouts rest on a trade print" in report
    # the bias DIRECTION, named
    assert "too HIGH (pessimistic)" in report
    assert "aggressor's side of the spread" in report
    # the primary table is labeled and its median is the book-mid one
    assert "Mid source: `book_mid`" in report
    assert "median 1.00bp" in report
    # the secondary table exists, is labeled as not-pre-registered, and carries its own median
    assert "estimand 1 SECONDARY — trade-print fallback samples" in report
    assert "**not the pre-registered quantity**" in report
    assert "median 100.00bp" in report
    # staleness of the backing observations is on the page
    assert "mid age p50" in report
    assert "2.000s" in report and "40.000s" in report


def test_no_secondary_table_and_no_material_banner_when_the_fallback_is_unused(tmp_path):
    """Non-vacuity: the disclosures above must be driven by the data, not printed unconditionally.
    A pure book-mid run says so and emits neither the secondary table nor the banner."""
    meta = _complete_meta(tmp_path)
    r = ShadowPairResult(pair="BTC/USD", n_markouts_from_book_mid=2)
    for v in (2.0, 4.0):
        r.markouts.append(MarkoutSample(pair="BTC/USD", side=1, fill_price=100.0, fill_ts=0.0,
                                         horizon="1m", adverse_selection_bp=v))
    report = render_report(
        [meta], {"BTC/USD": compute_estimand1(r)}, {"BTC/USD": compute_estimand2(r)},
        estimand1_trade_fallback_by_pair={"BTC/USD": compute_estimand1_trade_fallback(r)})
    assert "estimand 1 SECONDARY" not in report
    assert "MATERIAL:" not in report
    assert "0.0% fallback share" in report
    assert "2 from BOOK MID" in report


def test_a_print_only_capture_leaves_the_primary_estimand_empty(tmp_path):
    """The strongest form of the fix: when EVERY available mid was a trade print, the
    pre-registered estimand has no samples at all and says so — rather than reporting a
    distribution built entirely out of a quantity it never pre-registered."""
    path = _capture(tmp_path, books=[], name="printonly.jsonl")
    _, results = run_shadow(path)
    r = results["BTC/USD"]
    assert r.n_markouts_from_trade_fallback == 1
    # B3 (run-45): all four pre-registered horizons are always reported (never omitted) — the
    # primary estimand's "1m" row is still present, but with n_used=0, since its only sample was
    # a print, not a mid.
    e1_primary = compute_estimand1(r)
    assert set(e1_primary.by_horizon) == {"1m", "5m", "20m", "1h"}
    assert e1_primary.by_horizon["1m"].n_used == 0, (
        "the primary estimand must have NOTHING USABLE — its only sample was a print, not a mid")
    assert compute_estimand1_trade_fallback(r).by_horizon["1m"].n_used == 1


def test_a_dead_book_channel_beside_a_dense_trade_channel_is_caught_by_the_source_split(tmp_path):
    """Why the aggregate coverage bound is gated on book+trade POOLED rather than per channel.

    Consider the one degradation pooling could hide: a pair whose BOOK subscription dies while
    trades keep arriving densely. Coverage stays ~100% (the trades cover the time), and
    `missing_channels()` stays clean (the book did carry events, earlier). A per-channel coverage
    bound would be the obvious answer — but it is not needed, because the mid-source fix catches
    this exact shape from the other side: every markout after the book dies falls back to a trade
    print, so it is tagged, excluded from the primary estimand, counted, and announced by the
    MATERIAL banner. The two fixes interlock; this test is what makes that a fact rather than an
    argument.
    """
    p = tmp_path / "deadbook.jsonl"
    w = CaptureWriter(str(p), ["BTC/USD"], "db", channels=["book", "trade"],
                      heartbeat_timeout_s=30.0)
    for i in range(1800):
        ts = float(i)
        px = 100.0 + (i % 11) * 0.05
        w.write_trade("BTC/USD", ts, px, 0.1, "buy")
        if i < 300:                     # the book channel dies after 300s; trades never do
            w.write_book("BTC/USD", ts + 0.1, px - 0.01, px + 0.01, 1.0, 1.0)
    w.close(clean=True)

    meta = read_capture(str(p))
    assert meta.missing_channels() == [], "the book DID carry events, so this check is blind"
    assert meta.degraded_coverage_pairs() == [], (
        "pooled coverage is blind too — the dense trades cover the time")
    meta.assert_complete()              # gate (iv) passes, correctly: no time went unobserved

    _, results = run_shadow(str(p))
    r = results["BTC/USD"]
    assert r.n_markouts_from_trade_fallback > 0
    assert r.trade_fallback_share > 0.5, (
        "most markouts land after the book died, so most rest on a print, not a mid")
    report = render_report(
        [meta], {"BTC/USD": compute_estimand1(r)}, {"BTC/USD": compute_estimand2(r)},
        estimand1_trade_fallback_by_pair={"BTC/USD": compute_estimand1_trade_fallback(r)})
    assert "MATERIAL:" in report, "the dead book channel must be announced, not inferred"
    assert "estimand 1 SECONDARY" in report
    # and the primary table's sample count is strictly the book-backed subset
    e1 = compute_estimand1(r)
    assert e1.total_used < len(r.markouts)


def test_the_cli_renders_the_mid_source_disclosure_end_to_end(tmp_path):
    """The whole path: recorder format -> reader -> engine -> estimands -> artifact on disk. A
    dense capture (so gate iv's coverage bound is satisfied) must carry the source split into the
    written file, proving `cli.main` wires the new structures through rather than dropping them."""
    import cli
    p = tmp_path / "dense.jsonl"
    w = CaptureWriter(str(p), ["BTC/USD"], "dense", channels=["book", "trade"],
                      heartbeat_timeout_s=30.0)
    px = 100.0
    for i in range(1800):
        ts = float(i)
        px = 100.0 + (i % 7) * 0.05
        w.write_trade("BTC/USD", ts, px, 0.1, "buy")
        w.write_book("BTC/USD", ts + 0.1, px - 0.01, px + 0.01, 1.0, 1.0)
    w.close(clean=True)

    out = tmp_path / "artifact.md"
    assert cli.main(["--capture", str(p), "--out", str(out)]) == 0
    text = out.read_text()
    assert "from BOOK MID (the pre-registered quantity)" in text
    assert "Mid source: `book_mid`" in text
    assert "data-bearing coverage (gate iv, AGGREGATE)" in text
    assert "realized data-bearing cadence" in text
    assert "heartbeat_timeout_s = 30s" in text
