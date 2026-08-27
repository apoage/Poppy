"""TASK_907 — the artifact renderer.

Two hard gates live here, both TESTED properties of this module (not comments a future
refactor can silently drop):

  * Gate (iii): every rendered report's FIRST box states, verbatim, that this probe calibrates
    the maker-fill model and cannot reopen TASK_897's ALL FAIL verdict, quoting the verdict's
    own robustness argument (0/188 cells clear even with adverse selection at zero).
    `test_render_gates.py::test_gate_iii_disclosure_present` asserts the exact substring.
  * Gate (iv): `render_report` refuses — raises `CaptureFormatError`, emits nothing — if ANY
    capture backing the report is absent, incomplete, or internally inconsistent (per
    `capture.CaptureMetadata.assert_complete`). `test_render_gates.py::test_gate_iv_refusal`
    asserts the raise and that no output file is produced by the CLI wrapper in that case.

Two rules about what a rendered number is allowed to claim:

  * A horizon with no finite sample renders `not-evaluable` for its location AND for every
    model-tier comparison. It NEVER renders "BELOW measured": a check that did not execute a
    comparison is not a passed check, and `tier >= nan` is False for reasons that have nothing
    to do with the model being below anything.
  * Every dropped sample is DISCLOSED with a count next to the numbers it was dropped from
    (non-finite, gap-straddling, no-mid), plus the recorder's own health counters. A number
    whose sample set quietly shrank is a number presented as if it had been measured.

And two disclosures added in run-44, both about a number's RESOLUTION rather than its value:

  * The realized DATA-BEARING cadence (median/p95/max inter-event interval per pair per channel,
    heartbeats excluded) and the aggregate coverage fraction, printed next to the numbers. A
    number derived from a 2-minute-resolution capture must not be presentable as a minute-scale
    measurement, and the only way a reader can check that is if the realized cadence is on the
    page. Both gap knobs and the coverage bound are stated with them.
  * The MID SOURCE split. A trade print is not a mid; the primary estimand-1 table is book-mid
    only and the trade-print-backed samples get their own labeled table with the bias direction
    named. Previously the blend was documented only in a code comment the reader never sees.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture import CaptureFormatError, CaptureMetadata  # noqa: E402
from estimands import Estimand1Result, Estimand2Result  # noqa: E402
from reference_fill_model import MODEL_ADVERSE_SELECTION_BP_TIERS  # noqa: E402

#: What a cell renders when no comparison could be executed. Deliberately NOT a number, NOT
#: "0", and NOT a verdict — see the module docstring.
NOT_EVALUABLE = "not-evaluable"

# Gate (iii): this exact text (module-level, not inlined per call site) is what the test pins.
REQUIRED_DISCLOSURE = (
    "**This probe calibrates the maker-fill model; it cannot reopen TASK_897's ALL FAIL "
    "verdict.** The verdict's own robustness argument: 0/576 attainable-tier cells clear, "
    "0/864 at any nonzero fee, and **0/188 clear even with adverse selection set to ZERO**. "
    "No realized markout distribution rescues a band whose nearest cell is -5.045 with the "
    "penalty at zero. Touch-conditional markouts below are the OPTIMISTIC bound on adverse "
    "selection (real fills are selected worse than touches) — if even this bound meets or "
    "exceeds the model's 5 bp assumption, the model stands calibrated-pessimistic."
)


def render_disclosure_box() -> str:
    return "> " + REQUIRED_DISCLOSURE.replace("\n", "\n> ") + "\n"


def _assert_all_captures_complete(captures: List[CaptureMetadata]) -> None:
    if not captures:
        raise CaptureFormatError(
            "GATE (iv) REFUSAL: no capture metadata supplied at all — refusing to render any "
            "number without a capture to back it.")
    for meta in captures:
        meta.assert_complete()  # raises CaptureFormatError with the specific reason


def _render_capture_meta_block(captures: List[CaptureMetadata]) -> str:
    lines = ["## Capture provenance (gate iv)", ""]
    for meta in captures:
        lines.append(f"- `{meta.path}`: pairs={meta.pairs}, "
                      f"declared_pairs={meta.declared_pairs}, "
                      f"observed_pairs={meta.observed_pairs}, "
                      f"channels={meta.channels if meta.channels else 'undeclared'}, "
                      f"clock_basis={meta.clock_basis or 'undeclared'}, "
                      f"events={meta.n_book + meta.n_trade} "
                      f"(book={meta.n_book}, trade={meta.n_trade}), "
                      f"span=[{meta.start_ts:.0f}, {meta.end_ts:.0f}], "
                      f"recorded_gaps={len(meta.gaps)}, closed_cleanly={meta.closed_cleanly}")
        # Recorder health. "Connected but recording nothing" is invisible unless these are said
        # out loud, and Kraken v2's live schema is UNVERIFIED by this row.
        if meta.counters_present:
            lines.append(f"  - recorder health: json_decode_errors="
                          f"{meta.n_json_decode_errors}, "
                          f"schema_mismatch={meta.n_schema_mismatch}, "
                          f"unrecognized_frames={meta.n_unrecognized_frames}")
            if meta.n_schema_mismatch or meta.n_unrecognized_frames:
                lines.append("  - **SCHEMA DRIFT OBSERVED** — frames on a recognized channel "
                              "could not be read, or frames arrived on a channel this recorder "
                              "does not know. Kraken v2's live schema is unverified by this "
                              "row; treat every number below as covering only what was parsed.")
        else:
            lines.append("  - recorder health: NOT RECORDED in this capture (no `counters` "
                          "record — a pre-format-version-2 or synthetic capture). Schema-drift "
                          "and decode-failure counts are UNKNOWN for it, not zero.")
        # Gap-ledger hygiene: coverage is measured over the UNION of recorded gaps, so
        # overlapping/duplicate records cannot double-count over a real hole. Say when the
        # ledger needed normalizing.
        if (meta.n_overlapping_gap_records or meta.n_duplicate_gap_records
                or meta.n_gaps_outside_event_span):
            lines.append(f"  - gap-ledger normalization: overlapping_records="
                          f"{meta.n_overlapping_gap_records}, "
                          f"duplicate_records={meta.n_duplicate_gap_records}, "
                          f"records_outside_event_span={meta.n_gaps_outside_event_span} — "
                          f"{len(meta.gaps)} recorded gap(s) coalesced into "
                          f"{len(meta.merged_gap_spans)} disjoint span(s); coverage is the "
                          f"measure of the UNION, never the sum of per-gap overlaps.")
        if meta.truncated_tail:
            lines.append("  - truncated trailing line (a kill mid-write) — tolerated, "
                          "the partial record is not counted")
        for g in meta.gaps:
            lines.append(f"  - gap: {g.reason} [{g.start_ts:.0f}, {g.end_ts:.0f}] "
                          f"({g.duration_s:.1f}s) — {g.detail}")
        lines.append(_render_coverage_block(meta))
    lines.append("")
    return "\n".join(lines)


def _fmt_s(value: Optional[float]) -> str:
    return NOT_EVALUABLE if value is None else f"{value:.3f}s"


def _render_coverage_block(meta: CaptureMetadata) -> str:
    """Gate (iv)'s AGGREGATE disclosure: the realized data-bearing cadence, the cumulative
    coverage fraction, and both gap knobs — stated, not assumed.

    A capture exceeding the bound never reaches this function (`assert_complete` refuses first),
    so what this prints is the evidence that the numbers below are minute-scale.
    """
    lines = [
        "  - **data-bearing coverage (gate iv, AGGREGATE)** — computed from `book`/`trade` "
        "events ONLY; heartbeats are EXCLUDED because a heartbeat proves the socket is alive, "
        "not that the market was observed:",
        f"    - knobs: per-jump gap-detect threshold = {meta.gap_detect_threshold_s:g}s "
        f"(hard-clamped to the shadow engine's 60s bar interval — a wider threshold could let a "
        f"jump swallow a whole bar unflagged); coverage window = "
        f"{meta.coverage_window_s:g}s; declared bound = at most "
        f"{meta.max_uncovered_fraction * 100:.2f}% of accounted time un-observed; recorder "
        f"heartbeat_timeout_s = "
        + (f"{meta.declared_heartbeat_timeout_s:g}s"
           if meta.declared_heartbeat_timeout_s is not None else "undeclared")
        + ".",
        f"    - reader-detected discontinuities: {len(meta.unrecorded_gaps)} over all events, "
        f"{len(meta.unrecorded_data_gaps)} over data-bearing events alone (both must be 0 for "
        f"this report to exist).",
    ]
    if not meta.coverage_by_pair:
        lines.append("    - per-pair coverage: NOT APPLICABLE (no data-bearing span).")
    else:
        lines.append("")
        lines.append("| pair | data events | span (s) | recorded gap (s) | accounted (s) | "
                      "observed (s) | un-observed | bound |")
        lines.append("|---" * 8 + "|")
        for pair, cov in sorted(meta.coverage_by_pair.items()):
            lines.append(
                f"| {pair} | {cov.n_data_events} | {cov.span_s:.0f} | "
                f"{cov.recorded_gap_s:.0f} | {cov.accounted_s:.0f} | {cov.covered_s:.0f} | "
                f"{cov.uncovered_fraction * 100:.3f}% | "
                f"{meta.max_uncovered_fraction * 100:.2f}% |")
        lines.append("")
    # A.4(4): the cadence table alone cannot say whether a book median of 5.0s is the MARKET's
    # rate or the RECORDER's rate limit — so the declared floor is stated immediately above it,
    # unconditionally (an undeclared floor is itself the thing a reader must be told). This is
    # the read end of the `meta_start` declaration: `read_capture` refuses a floor at or above
    # the gap-detect threshold, so any floor printed here is inside gate (iv)'s resolution.
    lines.append(
        "    - book emission policy declared by the recorder = "
        + (f"`{meta.declared_book_emission_policy}`"
           if meta.declared_book_emission_policy is not None else "UNDECLARED")
        + "; cadence FLOOR (book_keepalive_s) = "
        + (f"{meta.declared_book_keepalive_s:g}s"
           if meta.declared_book_keepalive_s is not None else "UNDECLARED")
        + ". A book cadence at or near this floor is the RECORDER's rate limit, not a "
          "measurement of the market; only intervals above the floor are the market's own rate.")
    if meta.cadence:
        lines.append("  - **realized data-bearing cadence** (inter-event interval; this is the "
                      "resolution at which the market was actually observed — read every number "
                      "below against it):")
        lines.append("")
        lines.append("| pair | channel | n events | median | p95 | max |")
        lines.append("|---" * 6 + "|")
        for c in meta.cadence:
            lines.append(f"| {c.pair} | {c.channel} | {c.n_events} | "
                          f"{_fmt_s(c.median_interval_s)} | {_fmt_s(c.p95_interval_s)} | "
                          f"{_fmt_s(c.max_interval_s)} |")
        lines.append("")
    return "\n".join(lines)


def _fmt_bp(value: Optional[float]) -> str:
    return NOT_EVALUABLE if value is None else f"{value:.2f}"


#: What the artifact says about each mid source, verbatim. The fallback's entry names the BIAS
#: DIRECTION: a trade print is struck at the aggressor's side of the spread, and after a fill
#: that is going badly aggressor flow is one-sided with the move, so the print sits on the far
#: side of the mid and measured adverse selection comes out too HIGH. Too high is the direction
#: of this row's own preferred conclusion ("even the optimistic bound meets the model's 5 bp"),
#: which is why these samples are reported apart instead of pooled.
MID_SOURCE_NOTES = {
    "book_mid": (
        "PRIMARY, as pre-registered: the as-of mid is top-of-book `(bid + ask) / 2`."),
    "trade_print": (
        "SECONDARY — **not the pre-registered quantity**. No book quote was available at the "
        "horizon, so the last trade print stood in for the mid. A print is not a mid: it is "
        "struck at the aggressor's side of the spread. The bias is NOT mean-zero where it "
        "matters — after a fill that is going badly, aggressor flow is one-sided with the move, "
        "so the print sits on the far side of the mid and measured adverse selection comes out "
        "**too HIGH (pessimistic)**. That is the direction of this row's own preferred "
        "conclusion, so these samples are reported apart and are EXCLUDED from the primary "
        "table above; do not pool them."),
}


def _render_estimand1_section(pair: str, e1: Estimand1Result) -> str:
    label = ("estimand 1: adverse selection (side-signed markout, bp)"
             if e1.mid_source == "book_mid"
             else "estimand 1 SECONDARY — trade-print fallback samples (side-signed markout, bp)")
    lines = [f"### {pair} — {label}", "",
             f"Mid source: `{e1.mid_source}` — "
             f"{MID_SOURCE_NOTES.get(e1.mid_source, 'source not described')}",
             f"Samples from the other mid source, excluded from this table: "
             f"{e1.n_excluded_other_source}.", ""]
    if not e1.by_horizon:
        lines.append("_No touch-fills with a gap-clean markout window backed by this mid "
                      "source for this pair._\n")
        return "\n".join(lines)
    lines.append("| horizon | n_used | n_dropped_nonfinite | mid age p50 | mid age p95 | "
                  "mean bp | median bp | " +
                  " | ".join(f"p{d}" for d in (10, 20, 30, 40, 50, 60, 70, 80, 90)) + " |")
    lines.append("|---" * 16 + "|")
    for horizon in ("1m", "5m", "20m", "1h"):
        h = e1.by_horizon.get(horizon)
        if h is None:
            continue
        if h.deciles_bp:
            deciles = " | ".join(f"{h.deciles_bp[d]:.2f}"
                                  for d in (10, 20, 30, 40, 50, 60, 70, 80, 90))
        else:
            deciles = " | ".join([NOT_EVALUABLE] * 9)
        lines.append(f"| {horizon} | {h.n_used} | {h.n_dropped_nonfinite} | "
                      f"{_fmt_s(h.mid_age_p50_s)} | {_fmt_s(h.mid_age_p95_s)} | "
                      f"{_fmt_bp(h.mean_bp)} | {_fmt_bp(h.median_bp)} | {deciles} |")
    lines.append("")
    lines.append("`mid age` is how stale the backing observation was at the markout horizon "
                  "(cap: one bar interval). A 1-minute markout read off a mid observed tens of "
                  "seconds earlier is not a 1-minute measurement, so the staleness is printed "
                  "rather than assumed away.")
    lines.append("")

    total_dropped = e1.total_dropped_nonfinite
    if total_dropped:
        lines.append(f"**{total_dropped} markout sample(s) were NON-FINITE and were excluded "
                      f"from every mean, median, decile and tier comparison above.** They are "
                      f"counted here rather than silently dropped: `n_used` is the sample count "
                      f"that actually backs each row, and no row's numbers are derived from a "
                      f"rank ordering containing a non-finite value.")
        lines.append("")

    lines.append("Model comparison (does the model's flat tier meet/exceed the measured "
                  "median — i.e. stand calibrated-pessimistic?):")
    for horizon in ("1m", "5m", "20m", "1h"):
        h = e1.by_horizon.get(horizon)
        if h is None:
            continue
        if not h.is_evaluable:
            # No finite median exists. A comparison did not run, so no verdict is emitted —
            # `tier >= nan` being False is not evidence that the tier is below anything.
            lines.append(
                f"- {horizon}: median {NOT_EVALUABLE} "
                f"({h.n_samples} sample(s), {h.n_dropped_nonfinite} non-finite, 0 usable) — "
                + ", ".join(f"{tier:g}bp: {NOT_EVALUABLE}"
                             for tier in h.model_tiers_meet_or_exceed_median)
                # Deliberately avoids spelling the verdict token itself: the exact string
                # "BELOW measured" must appear in this artifact ONLY where a real comparison
                # produced it, so that grepping for it stays meaningful.
                + ". No comparison was executed; a check that did not run is not a passed "
                  "check, and it is not a finding that the tier sits below the measured "
                  "median either.")
            continue
        parts = ", ".join(
            f"{tier:g}bp: "
            + (NOT_EVALUABLE if ok is None else ("meets/exceeds" if ok else "BELOW measured"))
            for tier, ok in h.model_tiers_meet_or_exceed_median.items())
        lines.append(f"- {horizon}: median {h.median_bp:.2f}bp (n_used={h.n_used}) — {parts}")
    lines.append("")
    return "\n".join(lines)


def _render_mid_source_split(e2: Estimand2Result) -> str:
    share = e2.trade_fallback_share
    if share is None:
        return ("- mid-source split of the recorded markouts: none recorded, so neither source "
                "backs anything.\n")
    prominent = ""
    if share >= 0.05:
        prominent = (f"  **MATERIAL: {share * 100:.1f}% of this pair's markouts rest on a trade "
                     f"print rather than a mid.** They are excluded from the primary estimand-1 "
                     f"table and reported separately; their bias direction is PESSIMISTIC on "
                     f"adverse selection, i.e. toward this row's own preferred conclusion. Read "
                     f"the primary table as the measurement and the secondary one as a bound on "
                     f"what the missing book coverage could be doing.\n")
    return (f"- mid-source split of the recorded markouts: "
            f"{e2.n_markouts_from_book_mid} from BOOK MID (the pre-registered quantity), "
            f"{e2.n_markouts_from_trade_fallback} from the TRADE-PRINT fallback "
            f"({share * 100:.1f}% fallback share).\n" + prominent)


def _render_estimand2_section(pair: str, e2: Estimand2Result) -> str:
    tr = "n/a (no postings)" if e2.touch_fill_rate is None else f"{e2.touch_fill_rate:.4f}"
    xr = "n/a (no postings)" if e2.trade_cross_fill_rate is None else f"{e2.trade_cross_fill_rate:.4f}"
    return (
        f"### {pair} — estimand 2: fill rate (both proxies are OPTIMISTIC upper bounds; "
        f"queue priority is unresolvable from the public feed)\n\n"
        f"- posted: {e2.n_posted}\n"
        f"- OHLC-touch fills: {e2.n_touch_fills} (rate {tr})\n"
        f"- trade-cross fills: {e2.n_trade_cross_fills} (rate {xr})\n"
        f"- markout candidates dropped: {e2.n_markouts_dropped_gap} straddled a recorded gap "
        f"(gate iv), {e2.n_markouts_dropped_outside_span} had a horizon past the end of the "
        f"capture, {e2.n_markouts_dropped_no_mid} had no as-of mid at the horizon, "
        f"{e2.n_markouts_dropped_nonfinite} were non-finite. Disclosed, not silent — the fill "
        f"counts above are unaffected by these drops, which apply to estimand 1's samples.\n"
        + _render_mid_source_split(e2)
    )


def render_report(captures: List[CaptureMetadata],
                   estimand1_by_pair: Dict[str, Estimand1Result],
                   estimand2_by_pair: Dict[str, Estimand2Result],
                   generated_ts: Optional[float] = None,
                   estimand1_trade_fallback_by_pair: Optional[
                       Dict[str, Estimand1Result]] = None) -> str:
    """Build the full markdown artifact. Raises `CaptureFormatError` (refuses to render
    anything) if any supplied capture is absent, incomplete, or internally inconsistent —
    gate (iv). The disclosure box (gate iii) is unconditionally the first content emitted.
    """
    _assert_all_captures_complete(captures)  # gate (iv) — raises before any number is built

    generated_ts = generated_ts if generated_ts is not None else time.time()
    parts = [
        "# TASK_907 — live fill-model probe (phase 1: shadow)", "",
        render_disclosure_box(), "",
        f"Generated: {generated_ts:.0f} (unix)  \n"
        f"Pre-registration: `PREREGISTRATION.md` (written before this renderer existed; "
        f"amendments 2026-08-27 (a) and (b) add DISCLOSURE fields — "
        f"`n_used`/`n_dropped_nonfinite`, the realized data-bearing cadence, aggregate coverage, "
        f"mid staleness, and the mid-source split — and, per (b.2), confine estimand 1 to "
        f"book-mid samples; the estimand FORMULAS are unchanged)  \n"
        f"Model tiers compared: {MODEL_ADVERSE_SELECTION_BP_TIERS} bp  \n"
        f"Gap-accounting clock basis: local receipt time (single basis; venue timestamps are "
        f"recorded as `venue_ts` and used by nothing)\n", "",
        _render_capture_meta_block(captures),
    ]
    fallback = estimand1_trade_fallback_by_pair or {}
    pairs = sorted(set(estimand1_by_pair) | set(estimand2_by_pair) | set(fallback))
    for pair in pairs:
        e1 = estimand1_by_pair.get(pair)
        e2 = estimand2_by_pair.get(pair)
        parts.append(f"## {pair}\n")
        if e1 is not None:
            parts.append(_render_estimand1_section(pair, e1))
        e1f = fallback.get(pair)
        if e1f is not None and e1f.by_horizon:
            parts.append(_render_estimand1_section(pair, e1f))
        if e2 is not None:
            parts.append(_render_estimand2_section(pair, e2))
    return "\n".join(parts) + "\n"
