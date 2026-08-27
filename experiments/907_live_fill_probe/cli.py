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
from typing import List, Optional

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
    try:
        for capture_path in args.capture:
            meta, results = run_shadow(capture_path,
                                        gap_detect_threshold_s=args.gap_detect_threshold_s,
                                        pairs=pairs_filter)
            captures.append(meta)
            for pair, result in results.items():
                estimand1_by_pair[pair] = compute_estimand1(result)
                estimand1_fallback_by_pair[pair] = compute_estimand1_trade_fallback(result)
                estimand2_by_pair[pair] = compute_estimand2(result)
        report = render_report(
            captures, estimand1_by_pair, estimand2_by_pair,
            estimand1_trade_fallback_by_pair=estimand1_fallback_by_pair)
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
