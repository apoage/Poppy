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
from capture import (  # noqa: E402
    BOOK_KEEPALIVE_OVERRUN_FACTOR as CAPTURE_BOOK_KEEPALIVE_OVERRUN_FACTOR,
    CaptureFormatError, CaptureMetadata, _fmt_uncovered_fraction,
)
from estimands import Estimand1Result, Estimand2Result  # noqa: E402
from reference_fill_model import MODEL_ADVERSE_SELECTION_BP_TIERS  # noqa: E402

#: What a cell renders when no comparison could be executed. Deliberately NOT a number, NOT
#: "0", and NOT a verdict — see the module docstring.
NOT_EVALUABLE = "not-evaluable"

#: How many individual gap records the provenance block prints before summarizing (run-49).
#: The real 3.586-day capture carries 1,688 of them, and printing every one produced an 1,860-line
#: artifact of which ~1,700 lines were this loop — the disclosures a reader is meant to check
#: buried under a raw ledger dump. Capped like every other sample list in this module's refusals,
#: with the TOTAL and the per-reason breakdown stated so nothing is silently dropped: the count is
#: the claim, the individual lines are only examples of it.
MAX_RENDERED_GAP_LINES = 40

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
        # B1 (run-45): sessions — so an old single-session capture is never presented as
        # indistinguishable from a reconciled multi-session one. `declared_pairs`/`channels` above
        # are already the UNION across every session; this line is what lets a reader see WHICH
        # sessions contributed to that union, and notice when they disagree.
        lines.append(f"  - sessions: {len(meta.sessions)} "
                      f"({', '.join(s.kind for s in meta.sessions) or 'none'})")
        if len(meta.sessions) > 1:
            for i, s in enumerate(meta.sessions):
                lines.append(f"    - session {i} ({s.kind}): pairs={s.pairs}, "
                              f"channels={s.channels if s.channels is not None else 'undeclared'}, "
                              f"heartbeat_timeout_s={s.heartbeat_timeout_s}")
            distinct_pair_decls = {tuple(s.pairs) for s in meta.sessions if s.pairs}
            distinct_channel_decls = {tuple(s.channels) for s in meta.sessions
                                       if s.channels is not None}
            # cycle 2 (run-45), MINOR 2: `heartbeat_timeout_s` is the THIRD field a session
            # declares, and it was excluded from this comparison — so two sessions differing only
            # in the recorder's own liveness knob never tripped the banner, even though that knob
            # is exactly what "no gap was recorded" is evidence of. The raw values were printed
            # above, so no information was lost; the FLAG was.
            distinct_hb_decls = {s.heartbeat_timeout_s for s in meta.sessions
                                  if s.heartbeat_timeout_s is not None}
            if (len(distinct_pair_decls) > 1 or len(distinct_channel_decls) > 1
                    or len(distinct_hb_decls) > 1):
                lines.append("    - **HETEROGENEOUS DECLARATIONS**: this capture's sessions did "
                              "not all declare the same config — the reconciliation above uses "
                              "the UNION of all of them, never the first session presented as "
                              "the whole.")
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
        if meta.gaps:
            # Run-49: capped, with the totals stated. Every gap still counts in full toward the
            # coverage arithmetic above and toward `covers()` per markout — this cap is about
            # what is PRINTED, and the summary line carries the whole population.
            by_reason: Dict[str, int] = {}
            total_gap_s = 0.0
            for g in meta.gaps:
                by_reason[g.reason] = by_reason.get(g.reason, 0) + 1
                total_gap_s += g.duration_s
            reasons = ", ".join(f"{r}={n}" for r, n in sorted(by_reason.items()))
            lines.append(
                f"  - recorded gaps: {len(meta.gaps)} record(s) totalling {total_gap_s:.0f}s "
                f"(by reason: {reasons}); coalesced into {len(meta.merged_gap_spans)} disjoint "
                f"span(s). Every one of them is counted in the coverage table below and in the "
                f"per-markout `covers()` check; the listing that follows is capped at "
                f"{MAX_RENDERED_GAP_LINES} example(s).")
        for g in meta.gaps[:MAX_RENDERED_GAP_LINES]:
            lines.append(f"  - gap: {g.reason} [{g.start_ts:.0f}, {g.end_ts:.0f}] "
                          f"({g.duration_s:.1f}s) — {g.detail}")
        if len(meta.gaps) > MAX_RENDERED_GAP_LINES:
            lines.append(f"  - gap: ... and {len(meta.gaps) - MAX_RENDERED_GAP_LINES} further "
                          f"recorded gap(s) not listed individually (see the totals above; the "
                          f"capture file itself is the full ledger).")
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
    # B2 item 2 (run-45): coverage above is denominated in the CAPTURE span (all events,
    # heartbeats included) — state the DATA-BEARING span next to it so the reconciliation is
    # visible rather than implied: the gap between the two IS the quantity that used to collapse
    # the denominator together with the numerator (the run-45 BLOCKER).
    if meta.start_ts is not None and meta.end_ts is not None:
        capture_span_s = meta.end_ts - meta.start_ts
        if meta.data_start_ts is not None and meta.data_end_ts is not None:
            data_span_s = meta.data_end_ts - meta.data_start_ts
            lines.append(
                f"    - capture span (all events, heartbeats included): {capture_span_s:.1f}s "
                f"[{meta.start_ts:.0f}, {meta.end_ts:.0f}]; data-bearing span (book/trade only): "
                f"{data_span_s:.1f}s [{meta.data_start_ts:.0f}, {meta.data_end_ts:.0f}] — the "
                f"per-pair coverage table above is denominated in the CAPTURE span, reconciled "
                f"against this data span; the difference between the two "
                f"({capture_span_s - data_span_s:.1f}s) is what a data-span-denominated "
                f"statistic would have silently excluded from its own denominator.")
        else:
            lines.append(
                f"    - capture span (all events, heartbeats included): {capture_span_s:.1f}s "
                f"[{meta.start_ts:.0f}, {meta.end_ts:.0f}]; data-bearing span: NONE (zero "
                f"book/trade events).")
    disclosed_trade = meta.disclosed_empty_trade_cells()
    if disclosed_trade:
        lines.append(
            f"    - **disclosed empty trade cell(s)** (in-gate MAJOR #1, run-45): {disclosed_trade} "
            f"— a live `book` proves the pair's subscription and symbol are alive; zero prints in "
            f"the window is a market fact, not a broken subscription, so it is DISCLOSED here "
            f"rather than refusing the whole render (a missing `book` cell on an observed pair "
            f"still refuses — see the provenance section above if this capture did that).")
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
                f"{_fmt_uncovered_fraction(cov.uncovered_fraction)} | "
                f"{meta.max_uncovered_fraction * 100:.2f}% |")
        lines.append("")
        # cycle 2 (run-45): the NO-CLAIM sentinel, said out loud. `accounted_s <= 0` means this
        # pair's coverage arithmetic had no denominator, so it made no claim — which is a
        # different statement from "0% un-observed" (the value this branch used to return) and
        # must never be read as the strongest coverage claim the instrument can make.
        no_claim = meta.no_claim_coverage_pairs()
        if no_claim:
            lines.append(
                f"    - **NO COVERAGE CLAIM** for pair(s) {no_claim}: their accounted time is "
                f"zero (a zero-length capture span, or a span whose absence the gap ledger "
                f"discloses in full), so no un-observed fraction exists to report. Read as "
                f"\"this instrument measured nothing here\", NOT as \"nothing was missed\".")
    # cycle 2 (run-45), the keepalive reconciliation: the declared emission policy checked against
    # the observed cadence, in BOTH directions. Emitted only when there is something to say, so a
    # capture with a healthy book renders byte-identically to before.
    stale_pairs = meta.stale_book_pairs()
    if stale_pairs:
        detail = ", ".join(
            f"{p}: {meta.coverage_by_pair[p].stale_book_s:.0f}s"
            for p in stale_pairs)
        lines.append(
            f"    - **FROZEN-MID BOOK TIME** — {detail}. Over these stretches the recorder kept "
            f"emitting `book` events (its declared policy is change-or-keepalive at a "
            f"`book_keepalive_s`={meta.declared_book_keepalive_s}s floor) while the top-of-book "
            f"PRICE PAIR `(bid, ask)` never moved. (Sizes are deliberately NOT part of that key: "
            f"the estimands below consume only the mid `(bid+ask)/2`, and keying on sizes too "
            f"made a frozen mid with jittering sizes read as a fully-observed book.) Those "
            f"repeats are EXCLUDED from the observed seconds above — only the first event of each "
            f"frozen run counts — and are already folded into the un-observed fraction printed in "
            f"the table. **What this asserts, exactly:** the mid these estimands are built from "
            f"carried no new information across these stretches. It does NOT assert the wire was "
            f"unobserved: the recorder emits a `book` event on a top-of-book change INCLUDING a "
            f"size-only change, as well as on its keepalive timer, and the shipped record carries "
            f"no per-event flag telling the two apart — so a repeat here may be a timer firing "
            f"with no frame behind it OR a real frame in which only the sizes moved, and this "
            f"reader counts both. "
            f"**READ THE REACH OF THIS NARROWLY.** The un-observed fraction is denominated over "
            f"the UNION of each pair's `book` AND `trade` observations, so stale book time can "
            f"push a pair past the declared bound (and refuse the whole capture) ONLY where that "
            f"pair's trades have stopped as well. **A book frozen or silent while its prints keep "
            f"flowing is NOT caught by any gate in ADDENDUM B1** — measured: 7195s of a 7200s "
            f"capture stale with trades every 30s reads 0.13% un-observed against a 2.00% bound "
            f"and publishes markouts on a mid that never moved. Catching that needs a "
            f"book-mid-only freshness term, which is ADDENDUM B2 scope, as is any per-pair "
            f"exclusion. Read this line as a MEASUREMENT of stale book time, never as evidence "
            f"that staleness has been checked and cleared — for these pairs or for the pairs "
            f"absent from this line. "
            f"**THE EXACT REACH OF THE WORD \"STALE\" HERE, because this line names a mechanism "
            f"and a mechanism invites the stronger reading.** It counts one shape and one only: a "
            f"RUN OF CONSECUTIVE IDENTICAL `(bid, ask)` events. Any change to the price pair — "
            f"including a change back — ENDS the run. So a mid that OSCILLATES over a small set "
            f"of values, or otherwise moves without ever discovering a new price, is NOT COUNTED "
            f"BY THIS MEASURE and is NOT DETECTED ANYWHERE IN ADDENDUM B1. Measured: a top of "
            f"book alternating between two `(bid, ask)` values for the whole of a 7200s capture "
            f"reads `stale_book_s = 0.0`, prints NO line here at all, reports 0.00% un-observed, "
            f"and publishes its markouts. A `0.0s` reading, or absence from this line, is "
            f"therefore NOT evidence that the mid was informative — it is only evidence that no "
            f"literal repeat ran long enough to count. The denomination-correct instrument for "
            f"\"this mid discovered no new prices\" is ADDENDUM B2's book-mid FRESHNESS term; "
            f"this run counter is not it, and widening this counter to chase period-N cycles or "
            f"bounded-value sets would be that B2 term built in the wrong denomination.")
    if meta.book_keepalive_overruns:
        detail = ", ".join(
            f"{p}: {n} interval(s), worst {meta.book_keepalive_max_interval_s[p]:.1f}s"
            for p, n in sorted(meta.book_keepalive_overruns.items()))
        lines.append(
            f"    - **BOOK KEEPALIVE OVERRUN** (the mirror direction of the same declaration) — "
            f"{detail}. The recorder DECLARED it would emit a book event at least every "
            f"{meta.declared_book_keepalive_s}s; these intervals exceed that floor by more than "
            f"{CAPTURE_BOOK_KEEPALIVE_OVERRUN_FACTOR:g}x with no recorded gap explaining them. "
            f"DISCLOSED, not refused: the reader's own gap-detect threshold is "
            f"{meta.gap_detect_threshold_s:g}s, so a hole this size is invisible to every other "
            f"check here and a reader must be told rather than have a multi-day capture "
            f"retro-refused over scheduler noise. The scan is anchored to the CAPTURE span, so "
            f"book silence BEFORE the first book event or AFTER the last one counts as an "
            f"interval like any other — that edge case is the worst one this check exists for "
            f"(a book that emits on time and then simply stops), and it used to be invisible "
            f"because no consecutive pair of book events spans it.")
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
            # B3 (run-45): `compute_estimand1` now always emits all four pre-registered horizons,
            # so this should be unreachable in practice — but the render must not be ABLE to drop
            # a pre-registered horizon silently if it ever is missing (e.g. a hand-built
            # `Estimand1Result` in a test). A stated row, not a silent skip.
            lines.append(f"| {horizon} | (MISSING — pre-registered horizon absent from this "
                          f"result; not omitted, not zero) | | | | | | | | | | | | | | |")
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
                  "rather than assumed away. **It is age by TIMESTAMP, not by VALUE** (run-49): "
                  "it measures the time since the last `book` EVENT for this pair, and the "
                  "recorder emits one at least every `book_keepalive_s` whether or not the top "
                  "of book moved — so a mid frozen in VALUE for hours still reads as seconds "
                  "old here. Read this column against the STALE BOOK TIME line in the coverage "
                  "block above, which measures the other thing. A freshness disclosure that "
                  "reads the TRUE mid age is ADDENDUM B2 scope and does not exist yet.")
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
            lines.append(f"- {horizon}: (MISSING — pre-registered horizon absent from this "
                          f"result; not omitted, not zero; no comparison was or could be run)")
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
                       Dict[str, Estimand1Result]] = None,
                   pair_capture_path: Optional[Dict[str, str]] = None,
                   pairs_filter: Optional[List[str]] = None,
                   captures_declared: Optional[List[str]] = None) -> str:
    """Build the full markdown artifact. Raises `CaptureFormatError` (refuses to render
    anything) if any supplied capture is absent, incomplete, or internally inconsistent —
    gate (iv). The disclosure box (gate iii) is unconditionally the first content emitted.

    `pair_capture_path` (B4, run-45): which supplied capture backed each pair's numbers — stated
    so a reader never has to guess which of several provenance-listed captures a given number
    came from. `pairs_filter`/`captures_declared` (B4/B5): the applied `--pairs` filter and the
    raw `--capture` list, stated next to the declared/observed sets and reconciled against how
    many captures actually contributed a pair's numbers.
    """
    _assert_all_captures_complete(captures)  # gate (iv) — raises before any number is built
    fallback = estimand1_trade_fallback_by_pair or {}
    pairs = sorted(set(estimand1_by_pair) | set(estimand2_by_pair) | set(fallback))

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
    ]
    parts.append(_render_capture_meta_block(captures))
    if captures_declared is not None or pairs_filter is not None:
        contributing = sorted(set((pair_capture_path or {}).values()))
        line = (f"**Command-line reconciliation (B4/B5, run-45):** captures declared on the "
                f"command line: {len(captures_declared or [])} {captures_declared or []}; "
                f"captures that actually contributed at least one pair's numbers: "
                f"{len(contributing)} {contributing}.")
        if pairs_filter is not None:
            line += (f" `--pairs` filter applied: {pairs_filter} (reconciled against the "
                     f"declared/observed union of every supplied capture — a pair named here "
                     f"that no capture declared or observed is a refusal, not a silent empty "
                     f"report).")
        else:
            line += " No `--pairs` filter applied (every pair found is reported)."
        parts.append(line + "\n")
    for pair in pairs:
        e1 = estimand1_by_pair.get(pair)
        e2 = estimand2_by_pair.get(pair)
        source = (pair_capture_path or {}).get(pair)
        heading = f"## {pair}" + (f" (source capture: `{source}`)" if source else "") + "\n"
        parts.append(heading)
        if e1 is not None:
            parts.append(_render_estimand1_section(pair, e1))
        e1f = fallback.get(pair)
        # B3 (run-45): `by_horizon` is no longer a usable proxy for "the fallback source was
        # actually used" — `compute_estimand1_trade_fallback` now ALWAYS populates all four
        # pre-registered horizons (n_used=0 when unused), so `by_horizon` truthiness alone would
        # render an empty secondary table on every pure-book-mid capture. `total_used` is the
        # real signal: at least one trade-print-backed sample actually exists.
        if e1f is not None and e1f.total_used > 0:
            parts.append(_render_estimand1_section(pair, e1f))
        if e2 is not None:
            parts.append(_render_estimand2_section(pair, e2))
    return "\n".join(parts) + "\n"
