#!/usr/bin/env python3
"""TASK_901 fix cycle 3 — dump the width-factor calibration table the sweep consumed.

`run_validation.py` prewarms `range_probe._K_CACHE` and then throws the table away: only the
coverage rows are written to RESULTS.json. That hid a fact fix cycle 3 needed — WHICH (block,
n_dec) pairs `calibrate_k` refuses to certify at all, and therefore how much of a cell's
"refusal" is the width calibration rather than the range estimator. Same parameters as the sweep
(200+200 trials, n_boot=300, seed 0), so this reproduces the exact table the sweep used.

Reproduce (gpu-server, wrapped, own tmux):
    tools/safe_run.sh --name t901calibtab --cpu --mem 16G -- \
        trader_env/bin/python experiments/901_range_based_dependence_probe/calib_table.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent))

import range_probe as rp        # noqa: E402
import run_validation as rv     # noqa: E402


def main() -> int:
    n_decs = sorted({nd for _f, _kw, nd in rv.CELLS})
    tbl = rp.prewarm_calibration_table(blocks=rp.BLOCK_STRATA, n_decs=n_decs, n_jobs=8,
                                       n_trials_calib=200, n_trials_valid=200, n_boot=300)
    out = {}
    for (block, n_dec, seed), v in sorted(tbl.items()):
        out[f"block={block},n_dec={n_dec}"] = v
        print(f"[901/calib] block={block:3d} n_dec={n_dec:5d} achieved={v.get('achieved')} "
              f"k={v.get('k')} val_cov={v.get('validation_coverage')} "
              f"{v.get('reason','')}", flush=True)
    (_THIS.parent / "CALIB_TABLE.json").write_text(json.dumps(out, indent=2, default=str))
    print("[901/calib] wrote CALIB_TABLE.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
