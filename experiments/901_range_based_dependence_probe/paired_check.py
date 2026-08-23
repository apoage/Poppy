#!/usr/bin/env python3
"""TASK_901 gate (iii), the discrimination test — a POST-HOC read over the already-written
`REAL_LEG.json`, needing no rebuild of the real series and no DB access at all.

For each horizon it restricts the chunk-containment comparison to the chunks that BOTH estimators
certified, which removes the chunk-SELECTION difference between them and leaves only the block
length. Writes the result back into `REAL_LEG.json` under `real_leg.paired_discrimination` (and
into the per-horizon records), so the artifact carries the test rather than a report claiming it.

    trader_env/bin/python experiments/901_range_based_dependence_probe/paired_check.py
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent))

import coverage_harness as ch   # noqa: E402


def main() -> int:
    path = _THIS.parent / "REAL_LEG.json"
    doc = json.loads(path.read_text())
    legs = doc["real_leg"].get("by_horizon") or {"1": {"by_estimator": doc["real_leg"]["by_estimator"]}}
    out = {}
    for H, leg in legs.items():
        est = leg.get("by_estimator") or {}
        pairs = []
        for a, b in itertools.combinations(sorted(est), 2):
            r = ch.paired_containment_on_common_chunks(est[a], est[b])
            pairs.append(r)
            print(f"H={H}: {r['estimator_a']} (blocks {r['a_blocks']}) "
                  f"{r['a_contains']}/{r['n_common_chunks']}  vs  {r['estimator_b']} "
                  f"(blocks {r['b_blocks']}) {r['b_contains']}/{r['n_common_chunks']}"
                  + ("   [IDENTICAL chunk-by-chunk]" if r["identical_per_chunk"] else ""))
        out[H] = pairs
        leg["paired_discrimination"] = pairs
    doc["real_leg"]["paired_discrimination"] = out
    path.write_text(json.dumps(doc, indent=2, default=str))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
