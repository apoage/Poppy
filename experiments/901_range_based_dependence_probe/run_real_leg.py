#!/usr/bin/env python3
"""TASK_901 gate (iii) — the real-data leg, run on the WORKSTATION rather than gpu-server.

DISCLOSED DEVIATION FROM THE TASK'S "everything runs on gpu-server" INSTRUCTION, justified rather
than a corner cut: this leg needs `runs/897_corrected_809/minute_extract_897.db`, an 11GB
byte-verified extract of the real minute table. The standing infra rule
(`infra_workstation_uplink.md`: "NEVER push the 24GB DB to the server; extended-DB/OOS reads run
on the workstation" — the uplink is ~5 Mbps) applies just as much to an 11GB extract as to the
24GB source; pushing it to gpu-server to satisfy this row's default compute-location instruction
would violate that standing rule for no benefit (the fit itself is a single, bounded,
n_universe=7 ridge regression over ~1 year of 1-minute bars — CPU-light, not the "sweep" class of
job that instruction exists to keep off the workstation). The Monte Carlo coverage/calibration
sweep (`run_validation.py --skip-real`) DOES run on gpu-server, per that instruction, because
that IS the compute-shaped part this row's compute guidance is aimed at. Wrapped in
`tools/safe_run.sh --cpu --mem 16G` exactly like the server jobs.

WHAT THIS LEG NOW RUNS (fix cycle 2, 2026-08-24):
  * ALL THREE HORIZONS THE GATE NAMES — H in {1, 5, 20}. The shipped cycle ran H=1 only and
    disclosed it in passing; that was a scope shortfall against the gate's own wording, and it is
    closed by running the other two rather than by disclosing it again.
  * `stationarity_diagnostics` FIRST, per horizon — because it decides whether the containment
    number that follows means anything about a range estimator at all. See
    `coverage_harness.real_data_chunk_containment`'s docstring.
  * the chunk-containment check per estimator, retained but RE-SCOPED to a property of the
    series, not evidence about the estimators.
The per-decision series itself is written to `REAL_SERIES_H{H}.npy` beside the JSON so any later
re-read (gate iv, TASK_902) can re-analyse it without a 5-minute rebuild and without re-opening
the extract.

Writes its own JSON (`REAL_LEG.json`) so `combine_results.py` can merge it with gpu-server's
synthetic-sweep output without either leg needing the other's compute location.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent))

import range_probe as rp        # noqa: E402
import coverage_harness as ch   # noqa: E402
import real_series              # noqa: E402

N_BOOT_CI = 300
HORIZONS = (1, 5, 20)


def main() -> int:
    t0 = time.time()
    # This leg only needs calibration at n_dec == chunk_size (1000) — a small, self-contained
    # prewarm, independent of the gpu-server sweep's own (separately cached) calibration table.
    print("[901-real] prewarming calibration table for n_dec=1000 ...", flush=True)
    rp.prewarm_calibration_table(blocks=rp.BLOCK_STRATA, n_decs=(1000,), n_jobs=4,
                                 n_trials_calib=200, n_trials_valid=200, n_boot=300)
    print(f"[901-real] calibration ready in {time.time() - t0:.0f}s", flush=True)

    by_horizon = {}
    for H in HORIZONS:
        print(f"[901-real] H={H}: building real per-decision series "
              f"(long_short, n_universe=7)...", flush=True)
        t_real = time.time()
        reb_ret, real_meta = real_series.build_real_series(H=H, variant="long_short")
        print(f"[901-real] H={H}: series built in {time.time() - t_real:.0f}s: "
              f"n_decisions={reb_ret.size} symbols={real_meta['symbols']} "
              f"present_count_hist={real_meta['present_count_hist']} "
              f"n_elig_hist={real_meta['n_elig_hist']}", flush=True)
        npy = _THIS.parent / f"REAL_SERIES_H{H}.npy"
        np.save(npy, reb_ret)
        real_meta["series_npy"] = npy.name
        real_meta["series_sha256"] = hashlib.sha256(reb_ret.tobytes()).hexdigest()

        if reb_ret.size < 2 * 1000:
            print(f"[901-real] H={H}: only {reb_ret.size} decisions — fewer than 2 chunks; "
                  f"diagnostics and containment skipped (reported as such).", flush=True)
            by_horizon[H] = {"meta": real_meta, "stationarity": None, "by_estimator": None,
                             "note": "too few decisions for the 1000-decision chunk construction"}
            continue

        print(f"[901-real] H={H}: stationarity diagnostics (no estimator involved)...", flush=True)
        diag = ch.stationarity_diagnostics(reb_ret, chunk_size=1000, n_boot=1000, seed=5)
        print(f"[901-real] H={H}: grand_sharpe={diag['grand_sharpe']:.6f} "
              f"n_chunks={diag['n_chunks']} boot_se_by_block={diag['boot_se_by_block']} "
              f"between_sd={diag['between_chunk_sd']:.5f} "
              f"within_se={diag['within_chunk_se_mean']:.5f} "
              f"var_ratio={diag['between_within_variance_ratio']:.2f} "
              f"chi2={diag['chi2_stat']:.1f} df={diag['chi2_df']} p={diag['chi2_p_value']:.3g} "
              f"slope/chunk={diag['chunk_sharpe_slope_per_chunk']:.5f}", flush=True)
        print(f"[901-real] H={H}: chunk_vr_mean={diag['chunk_vr_mean']} "
              f"series_vr={diag['series_vr']}", flush=True)

        est = {}
        for estimator in rp.ESTIMATORS:
            cc = ch.real_data_chunk_containment(reb_ret, estimator, chunk_size=1000,
                                                n_boot_ci=N_BOOT_CI, seed=7)
            est[estimator] = cc
            print(f"[901-real] H={H}: {estimator:15s} n_chunks={cc['n_chunks']} "
                  f"certify_rate={cc['certify_rate']} med_range={cc['median_range_est']} "
                  f"contains_grand_sharpe_fraction={cc['contains_grand_sharpe_fraction']} "
                  f"(+/-{cc['contains_grand_sharpe_se']})", flush=True)
        by_horizon[H] = {"meta": real_meta, "stationarity": diag, "by_estimator": est}

    # `real_leg` keeps H=1 at the top level for backward compatibility with the shipped
    # RESULTS.json shape; `by_horizon` carries all three.
    real_leg = {"meta": by_horizon[1]["meta"], "by_estimator": by_horizon[1]["by_estimator"],
                "stationarity": by_horizon[1]["stationarity"], "by_horizon": by_horizon}
    out_path = _THIS.parent / "REAL_LEG.json"
    out_path.write_text(json.dumps({"real_leg": real_leg, "runtime_s": time.time() - t0},
                                   indent=2, default=str))
    print(f"[901-real] wrote {out_path} ({time.time() - t0:.0f}s total)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
