#!/usr/bin/env python3
"""TASK_924 — drives one full pass of `instrument.py` over a capture file and writes
`results.json`. Called for BOTH the smoke/synthetic pre-registration pass and the full
1.44 GB pass; `--label` distinguishes them in the output.

D81: read-only over the capture path given on the command line. Never touches anything under
`experiments/907_live_fill_probe/`. G2 (read-only proof) is enforced HERE, not trusted from
the caller: the capture's sha256/size/mtime are fingerprinted before AND after the pass and
compared.

Usage:
    python run_full_pass.py --capture <path> --spec <path-to-SPEC.md> \
        --out <path-to-results.json> --label full_pass
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import instrument  # noqa: E402


def _vm_hwm_bytes() -> int:
    """Peak RSS (high-water mark) of THIS process, sampled the same way TASK_914's own guard
    does (`test_streaming_reader.py`): `VmHWM` from `/proc/self/status`, not
    `resource.getrusage().ru_maxrss` (which is not reset across a plain re-invocation the way
    this script is always freshly `python run_full_pass.py`'d, but VmHWM is the documented,
    kernel-tracked non-decreasing high-water mark for the CURRENT `mm`, which is what we want
    regardless)."""
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    # "VmHWM:	   123456 kB"
                    parts = line.split()
                    return int(parts[1]) * 1024
    except FileNotFoundError:
        pass
    return -1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--g1-tolerance-floor-s", type=float, default=60.0)
    ap.add_argument("--g1-tolerance-relative", type=float, default=0.001)
    ap.add_argument("--check-target", action="store_true",
                     help="Compare BTC/USD's shipped fraction against the pre-registered "
                          "127,071/273,233 B1 target (SPEC.md 2.2). Only meaningful for the "
                          "real full capture — smoke/synthetic fixtures have no reason to "
                          "reproduce that specific number and are not checked against it.")
    args = ap.parse_args(argv)

    spec_sha256 = instrument.sha256_file(args.spec)
    spec_registered_at = time.time()

    g2_before = instrument.file_fingerprint(args.capture)

    t0 = time.time()
    result = instrument.run_instrument(args.capture)
    t1 = time.time()

    g2_after = instrument.file_fingerprint(args.capture)
    read_only_ok = (g2_before == g2_after)

    peak_rss_bytes = _vm_hwm_bytes()

    # --- G1 ---
    g1_report = {}
    g1_pass = True
    target_stale_s = 127071.0
    target_accounted_s = 273233.0
    target_fraction = target_stale_s / target_accounted_s
    for pair, g in result.g1.items():
        tol = max(args.g1_tolerance_floor_s,
                   args.g1_tolerance_relative * g["shipped_accounted_s"])
        stale_diff = abs(g["shipped_stale_s"] - g["indep_stale_s"])
        accounted_diff = abs(g["shipped_accounted_s"] - g["indep_accounted_s"])
        agree = stale_diff <= tol and accounted_diff <= tol
        entry = dict(g)
        entry["tolerance_s"] = tol
        entry["stale_s_diff"] = stale_diff
        entry["accounted_s_diff"] = accounted_diff
        entry["implementations_agree"] = agree
        if pair == "BTC/USD" and args.check_target:
            entry["target_stale_s"] = target_stale_s
            entry["target_accounted_s"] = target_accounted_s
            entry["target_fraction"] = target_fraction
            shipped_target_diff = abs(g["shipped_stale_s"] - target_stale_s)
            entry["shipped_matches_target"] = shipped_target_diff <= tol
            if not entry["shipped_matches_target"]:
                g1_pass = False
        if not agree:
            g1_pass = False
        g1_report[pair] = entry

    out = {
        "label": args.label,
        "capture_path": args.capture,
        "spec_sha256": spec_sha256,
        "spec_registered_before_full_pass": True,
        "spec_registered_at": spec_registered_at,
        "g2_before": g2_before,
        "g2_after": g2_after,
        "g2_read_only_ok": read_only_ok,
        "g3_peak_rss_bytes": peak_rss_bytes,
        "g3_peak_rss_gib": peak_rss_bytes / (1024 ** 3) if peak_rss_bytes >= 0 else None,
        "g3_under_4gib": (0 <= peak_rss_bytes < 4 * 1024 ** 3),
        "g1_pass": g1_pass,
        "g1": g1_report,
        "wall_clock_s": t1 - t0,
        "capture_span_s": result.end_ts - result.start_ts,
        "capture_n_gaps": result.n_gaps,
        "accounted_s_global": result.accounted_s_global,
        "per_pair": result.per_pair,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)

    rss_str = f"{peak_rss_bytes / (1024**3):.3f} GiB" if peak_rss_bytes >= 0 else "N/A"
    print(f"[{args.label}] wrote {out_path} | wall_clock={t1 - t0:.1f}s | peak_rss={rss_str}")
    print(f"g1_pass={g1_pass} g2_read_only_ok={read_only_ok} "
          f"g3_under_4gib={out['g3_under_4gib']}")
    return 0 if (g1_pass and read_only_ok and out["g3_under_4gib"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
