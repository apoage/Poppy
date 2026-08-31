"""TASK_907 phase 1 — the shadow-artifact CLI: capture file(s) in, `shadow_<ts>.md` out.

`--out` defaults under `runs/907_live_fill_probe/` (gitignored — `runs/*` and `runs/*/*` in
`.gitignore`), NEVER a tracked path. This repo has a named prior defect class of a default
`--out` landing on a tracked artifact; `_default_out_path` is the single place that decision is
made, and `test_cli_out_default.py` pins it untracked.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture import (  # noqa: E402
    DEFAULT_GAP_DETECT_THRESHOLD_S, MAX_GAP_DETECT_THRESHOLD_S, CaptureFormatError, read_capture,
)
from estimands import (  # noqa: E402
    compute_estimand1, compute_estimand1_trade_fallback, compute_estimand2,
)
from render import render_report  # noqa: E402
from shadow_engine import run_shadow  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _default_out_path(now: Optional[float] = None) -> str:
    ts = int(now if now is not None else time.time())
    out_dir = _REPO_ROOT / "runs" / "907_live_fill_probe"
    return str(out_dir / f"shadow_{ts}.md")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        allow_abbrev=False,  # no flag on this probe is reachable by a prefix — see phase2_rails
        description="TASK_907 phase 1 — replay capture file(s) into the shadow-engine "
                    "estimands and render the artifact.")
    p.add_argument("--capture", action="append", required=True,
                    help="path to a capture JSONL file (repeatable for multiple captures)")
    p.add_argument("--pairs", default=None,
                    help="comma-separated subset of pairs to include (default: all pairs "
                         "found in the capture)")
    p.add_argument("--out", default=None,
                    help="output markdown path (default: untracked "
                         "runs/907_live_fill_probe/shadow_<ts>.md)")
    p.add_argument("--gap-detect-threshold-s", type=float,
                    default=DEFAULT_GAP_DETECT_THRESHOLD_S,
                    help=f"per-jump unrecorded-gap threshold, seconds (default "
                         f"{DEFAULT_GAP_DETECT_THRESHOLD_S:g}; values above "
                         f"{MAX_GAP_DETECT_THRESHOLD_S:g} are REFUSED — a threshold wider than "
                         f"the shadow engine's bar interval would let a jump swallow a whole bar "
                         f"unflagged)")
    # Deliberately NOT exposed as flags: `coverage_window_s` and `max_uncovered_fraction`. The
    # aggregate coverage bound is a declared property of the instrument, not an operator dial —
    # a dial on a fail-loud bound is a bypass with extra steps. They are keyword arguments of
    # `capture.read_capture` for tests only.
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    pairs_filter = [p.strip() for p in args.pairs.split(",")] if args.pairs else None
    out_path = args.out or _default_out_path()

    captures = []
    estimand1_by_pair = {}
    estimand1_fallback_by_pair = {}
    estimand2_by_pair = {}
    #: B4 (run-45): which `--capture` backed each pair's numbers. `cli.py` used to key
    #: `estimand1_by_pair` etc. on PAIR ALONE inside the loop over a repeatable `--capture`, so
    #: two captures both carrying the same pair silently overwrote — the LAST `--capture` on the
    #: command line won, order-dependently, under a provenance block that named both files.
    pair_capture_path: Dict[str, str] = {}
    #: B5: the pair set `--pairs` is reconciled against — per capture, DECLARED **∩** OBSERVED
    #: (spec line 628), then unioned across the supplied captures. Cycle 1 used declared ∪
    #: observed; that is a strictly larger set, so a `--pairs` naming a pair that was declared but
    #: never delivered a single event (or observed but never declared) fell through this
    #: reconciliation and was caught only by the generic zero-estimand refusal below — code and
    #: spec disagreeing silently. The intersection is the right operand for an OPERATOR filter:
    #: it is the set of pairs this instrument can actually back a number for, and a filter naming
    #: anything else is a declaration the instrument contradicts.
    reconcilable: set = set()
    #: Kept alongside for the refusal message only — a reader needs to see WHICH half of the
    #: intersection a named pair fell out of.
    declared_all: set = set()
    observed_all: set = set()
    try:
        for capture_path in args.capture:
            meta, results = run_shadow(capture_path,
                                        gap_detect_threshold_s=args.gap_detect_threshold_s,
                                        pairs=pairs_filter)
            captures.append(meta)
            declared_all |= set(meta.declared_pairs or [])
            observed_all |= set(meta.observed_pairs)
            reconcilable |= set(meta.declared_pairs or []) & set(meta.observed_pairs)
            # No per-pair stale-book exclusion here (ADDENDUM B2 scope, not shipped in B1): a
            # pair with keepalive-STALE book time is left in `results` like any other pair, and
            # `render_report`'s `_assert_all_captures_complete` (via `degraded_coverage_pairs`)
            # is the only thing that can stop it. Run-49, stated narrowly because the wider
            # version of this comment was false: that check is denominated over the UNION of a
            # pair's book AND trade observations, so it reaches a stale book only where the
            # pair's trades stopped too. A book frozen or silent while its prints keep flowing
            # passes every gate here and reaches the artifact — see
            # `CaptureMetadata.degraded_coverage_pairs`. Closing that needs B2's book-mid-only
            # freshness term.
            for pair, result in results.items():
                if pair in pair_capture_path and pair_capture_path[pair] != capture_path:
                    raise CaptureFormatError(
                        f"B4 REFUSAL: pair {pair!r} appears in more than one supplied "
                        f"--capture: {pair_capture_path[pair]!r} and {capture_path!r}. Do NOT "
                        f"silently pool: pooling markouts across captures separated by an outage "
                        f"or session boundary is a DIFFERENT estimand than the pre-registered "
                        f"one, and if it is wanted it must be declared, not defaulted into by a "
                        f"last-`--capture`-wins overwrite. Re-run with non-overlapping captures, "
                        f"or pass --pairs to select a disjoint subset per invocation.")
                pair_capture_path[pair] = capture_path
                estimand1_by_pair[pair] = compute_estimand1(result)
                estimand1_fallback_by_pair[pair] = compute_estimand1_trade_fallback(result)
                estimand2_by_pair[pair] = compute_estimand2(result)

        if pairs_filter is not None:
            # B5: `--pairs` is a THIRD operator declaration, reconciled against nothing before —
            # `--pairs SOL/USD` against a capture without it used to yield exit 0 and a
            # gate-passing artifact with zero estimand sections and no line saying a filter was
            # applied.
            unknown = [p for p in pairs_filter if p not in reconcilable]
            if unknown:
                raise CaptureFormatError(
                    f"B5 REFUSAL: --pairs named {unknown}, which no supplied --capture both "
                    f"declared AND observed (declared ∩ observed, unioned across all supplied "
                    f"captures: {sorted(reconcilable)}; declared anywhere: {sorted(declared_all)}; "
                    f"observed anywhere: {sorted(observed_all)}). An operator filter naming a pair "
                    f"the instrument contradicts is a loud refusal, not a quietly empty report — "
                    f"and a pair that was DECLARED but never delivered an event is exactly as "
                    f"unable to back a number as one that was never declared at all.")

        # B5, UNCONDITIONALLY (cycle 2, run-45): "An artifact with zero estimand sections must
        # never exit 0 carrying a gate-passing disclosure box" is a statement about the ARTIFACT,
        # not about the `--pairs` flag. Cycle 1 nested this inside `if pairs_filter is not None:`,
        # so an entirely ordinary capture — a quiet market: dense healthy book, zero prints —
        # rendered rc=0 with the full gate-(iv) disclosure box, the line "captures that actually
        # contributed at least one pair's numbers: 0 []", and no estimand section at all.
        # `shadow_engine.run_shadow` builds its results from TRADE-bar buckets, so a book-only
        # pair never enters them; the filter was never the only way to reach zero sections.
        if not estimand1_by_pair and not estimand2_by_pair:
            raise CaptureFormatError(
                f"B5 REFUSAL: this invocation produced ZERO estimand sections. An artifact with "
                f"zero estimand sections must never exit 0 carrying a gate-passing disclosure "
                f"box — the box would certify the gates a report passed while the report makes "
                f"no measurement at all. Captures supplied: {list(args.capture)}; "
                f"declared ∩ observed across them: {sorted(reconcilable)}; "
                f"--pairs filter: {pairs_filter if pairs_filter is not None else 'none'}. The "
                f"usual causes are a filter that selected nothing and a capture carrying no "
                f"TRADE-bar data for any pair (a book-only capture: the estimands are bucketed "
                f"on trade bars, so a live book with zero prints yields no section).")

        report = render_report(
            captures, estimand1_by_pair, estimand2_by_pair,
            estimand1_trade_fallback_by_pair=estimand1_fallback_by_pair,
            pair_capture_path=pair_capture_path, pairs_filter=pairs_filter,
            captures_declared=list(args.capture))
    except CaptureFormatError as exc:
        sys.stderr.write(f"REFUSED (gate iv): {exc}\n")
        return 1

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(report)
    print(f"wrote {out_file}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
