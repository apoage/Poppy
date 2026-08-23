#!/usr/bin/env python3
"""TASK_901 fix cycle 3 — the SPREAD behind two medians the artifact reported bare.

Cycle 2 reported `acf_band`'s readout on the persistent component as "median range 18" and the
`test_range_probe.py::test_range_acf_band_triangular_reads_out_close_to_dep` annotation as
"20-44 over 8 draws". A median over 1000 draws and a range over 8 both hide the distribution that
a reader needs in order to know what a single draw would do. This measures the percentiles.
Estimator-only (no bootstrap, no block rule) and therefore ~a second per cell.

Reproduce (gpu-server, wrapped, own tmux):
    tools/safe_run.sh --name t901spread --cpu --mem 8G -- \
        trader_env/bin/python experiments/901_range_based_dependence_probe/spread_check.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent))

import range_probe as rp   # noqa: E402
import families as fam     # noqa: E402


def spread(label, gen, n, n_draws, seed, **kw):
    rng = np.random.default_rng(seed)
    vals = []
    n_nan = 0
    for _ in range(n_draws):
        r, _i = rp.range_acf_band(gen(n, rng=rng, **kw))
        if np.isfinite(r):
            vals.append(float(r))
        else:
            n_nan += 1
    v = np.array(vals)
    out = {"label": label, "n": int(n), "n_draws": int(n_draws), "seed": int(seed),
           "n_nan": n_nan,
           "median": float(np.median(v)), "p10": float(np.percentile(v, 10)),
           "p90": float(np.percentile(v, 90)), "min": float(v.min()), "max": float(v.max()),
           "frac_in_20_60": float(np.mean((v >= 20) & (v <= 60)))}
    print(f"[901/spread] {label}: n_draws={n_draws} nan={n_nan} median={out['median']:.1f} "
          f"p10={out['p10']:.1f} p90={out['p90']:.1f} min={out['min']:.0f} max={out['max']:.0f} "
          f"frac_in[20,60]={out['frac_in_20_60']:.3f}", flush=True)
    return out


def main() -> int:
    rows = [
        spread("acf_band / persistent_component phi=0.99 w=0.2 / n=3000",
               fam.simulate_persistent_component, 3000, 1000, 901, phi=0.99, weight=0.2),
        spread("acf_band / triangular dep=20 / n=4000 (the unit test's cell)",
               fam.simulate_triangular, 4000, 200, 901, dep=20),
    ]
    (_THIS.parent / "SPREAD_CHECK.json").write_text(json.dumps(rows, indent=2))
    print("[901/spread] wrote SPREAD_CHECK.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
