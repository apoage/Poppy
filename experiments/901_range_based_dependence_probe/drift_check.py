#!/usr/bin/env python3
"""TASK_901 fix cycle 3 — is the between-chunk Sharpe variation a monotone DRIFT, or just
EXCESS DISPERSION?

Fix cycle 2 reported `chunk_sharpe_slope_per_chunk` as a bare point estimate and read it in prose
as "a monotone drift", illustrated with the first three and last three chunks. A slope with no SE
is not evidence of a trend. This module re-reads the SAME chunk Sharpes already recorded in
REAL_LEG.json (post-hoc; no rebuild, no DB, no series reconstruction) and puts the standard errors
on it: OLS slope SE / t / p / R^2, Spearman rho, a first-half vs second-half mean comparison, and
the between/within variance ratio recomputed on DETRENDED chunk Sharpes — the last one answering
the only question that matters for the row's load-bearing finding, namely how much of the excess
dispersion a linear trend would even absorb.

Reproduce (workstation — REAL_LEG.json is a workstation artifact, same as `paired_check.py`):
    tools/safe_run.sh --name t901drift --cpu --mem 8G -- \
        trader_env/bin/python experiments/901_range_based_dependence_probe/drift_check.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

_THIS = Path(__file__).resolve()


def analyse(sharpes, within_se_mean):
    y = np.asarray(sharpes, dtype=np.float64)
    x = np.arange(y.size, dtype=np.float64)
    lr = stats.linregress(x, y)
    rho, rho_p = stats.spearmanr(x, y)
    half = y.size // 2
    resid = y - (lr.intercept + lr.slope * x)
    return {
        "n_chunks": int(y.size),
        "slope_per_chunk": float(lr.slope),
        "slope_se": float(lr.stderr),
        "slope_t": float(lr.slope / lr.stderr),
        "slope_p": float(lr.pvalue),
        "r_squared": float(lr.rvalue ** 2),
        "spearman_rho": float(rho),
        "spearman_p": float(rho_p),
        "first_half_mean": float(y[:half].mean()),
        "second_half_mean": float(y[-half:].mean()),
        "min_chunk_sharpe": float(y.min()), "max_chunk_sharpe": float(y.max()),
        "argmin_chunk": int(y.argmin()), "argmax_chunk": int(y.argmax()),
        "between_sd_raw": float(y.std(ddof=1)),
        # Residual sd uses ddof=2 (two parameters fitted), so the detrended ratio is compared
        # against the raw one on equal footing rather than being flattered by the lost df.
        "between_sd_detrended": float(resid.std(ddof=2)),
        "between_within_var_ratio_raw": float((y.std(ddof=1) / within_se_mean) ** 2),
        "between_within_var_ratio_detrended": float((resid.std(ddof=2) / within_se_mean) ** 2),
        "last3": [round(float(v), 3) for v in y[-3:]],
        "lowest3_chunks": sorted(int(i) for i in np.argsort(y)[:3]),
        "highest3_chunks": sorted(int(i) for i in np.argsort(y)[-3:]),
    }


def main() -> int:
    real = json.loads((_THIS.parent / "REAL_LEG.json").read_text())["real_leg"]
    out = {}
    for h, leg in real["by_horizon"].items():
        diag = leg["stationarity"]
        key = f"H={h}"
        out[key] = analyse(diag["chunk_sharpes"], diag["within_chunk_se_mean"])
        r = out[key]
        print(f"[901/drift] {key}: slope={r['slope_per_chunk']:+.5f} se={r['slope_se']:.5f} "
              f"t={r['slope_t']:+.2f} p={r['slope_p']:.3f} R2={r['r_squared']:.3f} "
              f"spearman={r['spearman_rho']:+.2f} (p={r['spearman_p']:.3f}) "
              f"halves {r['first_half_mean']:+.3f} -> {r['second_half_mean']:+.3f} | "
              f"between/within var ratio {r['between_within_var_ratio_raw']:.2f} raw -> "
              f"{r['between_within_var_ratio_detrended']:.2f} detrended", flush=True)
    (_THIS.parent / "DRIFT_CHECK.json").write_text(json.dumps(out, indent=2))
    print("[901/drift] wrote DRIFT_CHECK.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
