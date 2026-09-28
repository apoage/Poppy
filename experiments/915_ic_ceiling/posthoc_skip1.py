#!/usr/bin/env python3
"""POST-HOC -- NOT PRE-REGISTERED. TASK_915 cycle-2 fix pass (2026-09-28), written AFTER the
official 160-cell run's G-A results were seen. Nothing here amends SPEC.md's gates; see
SPEC.md "POST-HOC ERRATA" for why it exists.

One-day-skip ("skip1") label robustness re-score. Every F1 feature ends at close_t
(`features.build_f1`: lc[t] - lc[t-lb]) and every label starts at close_t
(`targets.forward_log_return`: lc[t+h] - lc[t]). A same-close alignment like this rewards
anything that anti-correlates close_t's idiosyncratic print noise with itself (bid-ask bounce,
stale/last-trade closes): the noise enters the feature with + sign and the label with - sign.
The skip1 label starts at close_{t+1} instead:

    r_skip[t] = r_fwd[t+1]  (= lc[t+1+h] - lc[t+1]),  r_mkt_skip[t] = r_mkt_fwd[t+1]

and every target is rebuilt with the IDENTICAL formula `targets.build_all_targets` uses, with
the t-dated normalisers unchanged (beta[t], sigma_yz[t], sigma_c2c[t] -- the judge's
convention, /var/tmp/915-judge-leak/skipday_test.py). Features, present mask, estimators,
seeds, refit schedule and purge (`grid.fit_cell`, train_hi = test_lo - h) are untouched: this
module adds NO model logic, it only substitutes the target arrays and calls the module's own
`grid.fit_cell` -> `gates.cell_ic_stats` -> `gates.ga_pass` -> `gates.gb_result`.

Purge note: with the skip1 label the last training row (test_lo - h - 1) has a label ending at
close_{test_lo}, i.e. at the first test row's own feature timestamp -- still never inside any
test LABEL window (which starts at close_{test_lo+1}). The purge margin at that one boundary row
per quarterly refit shrinks from h to h-1 relative to the feature/label seam (corrected run-57
closing pass: "leak-free" was stated more strongly than this boundary case supports; bounded to
O(1) rows per refit, it cannot drive the window-wide F1 result).

`shift_targets(td, 0)` reproduces `build_all_targets`'s arrays bit-for-bit (tested), so the
only difference between an official cell and its skip1 twin is the one-row label shift.

Usage (gpu-server, wrapped, see README):
  python posthoc_skip1.py --official-dir /var/tmp/915-fix/official_copy \
      --outdir /var/tmp/915-fix/run_skip1 [--finalize-only]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import grid    # noqa: E402
import gates   # noqa: E402

POSTHOC = True
FS_PRIORITY = {"F2": 0, "F3": 1, "F4": 2, "F1": 3, "ALL": 4}


def shift_targets(td, k):
    """Return a copy of one horizon's `build_all_targets` dict whose forward components start k
    rows later (label window close_{t+k} -> close_{t+k+h}); t-dated normalisers unchanged.
    Formulas are verbatim `targets.build_all_targets` (k=0 reproduces it exactly)."""
    r_fwd = td["r_fwd"]
    r_mkt_fwd = td["r_mkt_fwd"]
    if k == 0:
        r_s, rm_s = r_fwd, r_mkt_fwd
    else:
        r_s = np.full_like(r_fwd, np.nan)
        r_s[:-k] = r_fwd[k:]
        rm_s = np.full_like(r_mkt_fwd, np.nan)
        rm_s[:-k] = r_mkt_fwd[k:]
    beta, sigma_yz, sigma_c2c = td["beta"], td["sigma_yz"], td["sigma_c2c"]
    resid = r_s - beta * rm_s[:, None]
    out = dict(td)
    out.update({
        "y1_primary_resid_yz": resid / np.maximum(sigma_yz, 1e-10),
        "y2_control_resid_c2c": resid / np.maximum(sigma_c2c, 1e-10),
        "y3_control_abs_return": np.abs(r_s),
        "y4_sign_volscaled": r_s / np.maximum(sigma_yz ** 2, 1e-10),
        "r_fwd": r_s, "r_mkt_fwd": rm_s, "resid_fwd": resid,
    })
    return out


def shifted_built(built, k):
    """Shallow copy of grid.build_panel_features_and_targets' output with every horizon's
    targets shifted by k. Features / present mask are the SAME objects (not copied)."""
    b = dict(built)
    b["targets_by_h"] = {h: shift_targets(td, k) for h, td in built["targets_by_h"].items()}
    return b


def score_cell(built_k, day_ts, target_name, h, fs, est, first_test_date="2022-01-01",
               last_date="2026-06-30", min_train_days=grid.MIN_TRAIN_DAYS):
    """The module's own pipeline for one cell on an already-shifted `built`."""
    res = grid.fit_cell(built_k, day_ts, h, target_name, fs, est,
                        first_test_date=first_test_date, last_date=last_date,
                        min_train_days=min_train_days)
    y = built_k["targets_by_h"][h][target_name]
    stats = gates.cell_ic_stats(res["forecast"], y, day_ts, h)
    passed, reason = gates.ga_pass(stats)
    return res, stats, passed, reason


def gb_for_cell(res, td_k, h, day_ts, mean_ic, cost_bp):
    """Mirror of run_grid.run's G-B block (same windows, same N_eff construction, same
    gates.gb_result call) applied to a (possibly shifted) horizon target dict."""
    import gp_fundamental_law as gpl
    lo, hi = gates._window_rows(day_ts, *gates.GA_WINDOW)
    forecast_win = res["forecast"][lo:hi]
    fwd_ret_win = td_k["r_fwd"][lo:hi]
    resid_fwd_win = td_k["resid_fwd"][lo:hi]
    finite_cols = np.isfinite(forecast_win).any(axis=0) & np.isfinite(fwd_ret_win).any(axis=0)
    resid = resid_fwd_win[:, finite_cols]
    resid_ok_rows = np.isfinite(resid).all(axis=1)
    n_eff = np.nan
    sigma_diag_full = np.full(res["forecast"].shape[1], np.nan)
    if resid_ok_rows.sum() >= 30 and finite_cols.sum() >= 5:
        n_eff, sigma_mat, _shr, _corr = gpl.residual_n_eff_and_sigma(resid[resid_ok_rows])
        sigma_diag_full[finite_cols] = np.diag(sigma_mat)
    gb = gates.gb_result(mean_ic, n_eff, h, forecast_win, fwd_ret_win, sigma_diag_full, cost_bp)
    return n_eff, gb


def _compact_stats(stats):
    keep = ("mean_ic", "nw_se", "nw_t", "bonferroni_t_crit", "n_dates_ga_window",
            "n_bar_live_assets", "tau_ic")
    out = {k: stats[k] for k in keep}
    out["by_year_mean_ic"] = {y: d["mean_ic"] for y, d in stats["by_year"].items()}
    return out


def timing_fragile(official_pass, skip1_pass):
    """Official G-A pass that does not survive the one-day skip. None = not tested (only
    official G-A passes are re-scored)."""
    if not official_pass:
        return None
    return bool(not skip1_pass)


def _official_cells(official_dir):
    rows = []
    for f in sorted((Path(official_dir) / "cells").glob("*.json")):
        rows.append(json.loads(f.read_text()))
    return rows


def sync_official(src, dst):
    """READ-ONLY copy of the official run's outputs into our scratch (never writes `src`).
    A cell file is copied only once it parses as JSON (run_grid writes non-atomically)."""
    import shutil
    src, dst = Path(src), Path(dst)
    (dst / "cells").mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted((src / "cells").glob("*.json")):
        tgt = dst / "cells" / f.name
        if tgt.exists():
            continue
        try:
            json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        shutil.copy2(f, tgt)
        n += 1
    done = (src / "DONE").exists()
    if done:   # grid_results.json is written BEFORE DONE by run_grid, so it is complete here
        for name in ("grid_results.json", "mde_table.json", "launch_log.txt"):
            if (src / name).exists():
                shutil.copy2(src / name, dst / name)
        shutil.copy2(src / "DONE", dst / "DONE")
    elif (src / "mde_table.json").exists():
        shutil.copy2(src / "mde_table.json", dst / "mde_table.json")
    return n, done


def _todo(official_dir, outdir):
    official = _official_cells(official_dir)
    passing = [r for r in official if r["ga_pass"]]
    passing.sort(key=lambda r: (FS_PRIORITY[r["feature_set"]], r["estimator"] != "ridge",
                                r["h"], r["target"]))
    todo = [r for r in passing
            if not (Path(outdir) / "skip1_cells" / f"{r['cell_id']}.json").exists()]
    return official, passing, todo


def run(official_dir, outdir, finalize_only=False, cost_bp=None, repro_ridge=True,
        follow_src=None, poll_seconds=60, deadline_epoch=None):
    import loader
    import run_grid
    cost_bp = run_grid.COST_BP_RESTATED_916 if cost_bp is None else cost_bp
    outdir = Path(outdir)
    (outdir / "skip1_cells").mkdir(parents=True, exist_ok=True)
    if follow_src:
        sync_official(follow_src, official_dir)
    official, passing, todo = _todo(official_dir, outdir)
    print(f"[915-skip1] official cells read={len(official)} ga_pass={len(passing)} "
          f"todo={len(todo)}", flush=True)

    built0 = built1 = day_ts = None
    while not finalize_only:
        if todo and built0 is None:
            t0 = time.time()
            panel = loader.load_panel()
            panel = loader.apply_liquidity_filter(panel)
            day_ts = panel["day_ts"]
            built0 = grid.build_panel_features_and_targets(panel)
            built1 = shifted_built(built0, 1)
            print(f"[915-skip1] panel+features built ({time.time()-t0:.0f}s)", flush=True)
        for i, r in enumerate(todo):
            tc0 = time.time()
            cid, h = r["cell_id"], r["h"]
            res, stats, passed, reason = score_cell(built1, day_ts, r["target"], h,
                                                    r["feature_set"], r["estimator"])
            row = {"cell_id": cid, "target": r["target"], "h": h,
                   "feature_set": r["feature_set"], "estimator": r["estimator"],
                   "official": {**_compact_stats(r["ga_stats"]), "ga_pass": r["ga_pass"],
                                "ga_reason": r["ga_reason"]},
                   "skip1": {**_compact_stats(stats), "ga_pass": passed, "ga_reason": reason,
                             "n_quarters_fit": len(res["quarters"])},
                   "timing_fragile": timing_fragile(r["ga_pass"], passed)}
            # Harness reproduction check: the k=0 re-score of a ridge cell through this
            # module must equal the official number (deterministic estimator only).
            if repro_ridge and r["estimator"] == "ridge":
                _, st0, p0, _ = score_cell(built0, day_ts, r["target"], h, r["feature_set"],
                                           "ridge")
                row["repro_k0"] = {"mean_ic": st0["mean_ic"], "nw_t": st0["nw_t"],
                                   "ga_pass": p0,
                                   "abs_diff_mean_ic": abs(st0["mean_ic"]
                                                           - r["ga_stats"]["mean_ic"])}
            if passed:
                n_eff, gb = gb_for_cell(res, built1["targets_by_h"][h], h, day_ts,
                                        stats["mean_ic"], cost_bp)
                row["skip1_gb"] = {"n_eff": n_eff, "gb": gb}
            row["seconds"] = time.time() - tc0
            (outdir / "skip1_cells" / f"{cid}.json").write_text(json.dumps(row, default=float))
            print(f"[915-skip1] {i+1}/{len(todo)} {cid}: official ic={r['ga_stats']['mean_ic']:.4f}"
                  f" t={r['ga_stats']['nw_t']:.2f} -> skip1 ic={stats['mean_ic']:.4f} "
                  f"t={stats['nw_t']:.2f} GA={'PASS' if passed else 'fail'} "
                  f"({row['seconds']:.1f}s)", flush=True)
        if not follow_src:
            break
        _n, done = sync_official(follow_src, official_dir)
        official, passing, todo = _todo(official_dir, outdir)
        if done and not todo:
            break
        if deadline_epoch is not None and time.time() > deadline_epoch:
            print("[915-skip1] deadline reached before official DONE -- finalizing partial",
                  flush=True)
            break
        if not todo:
            time.sleep(poll_seconds)
            _n, done = sync_official(follow_src, official_dir)
            official, passing, todo = _todo(official_dir, outdir)

    return finalize(official_dir, outdir, official, passing)


def finalize(official_dir, outdir, official, passing):
    outdir = Path(outdir)
    rows = []
    missing = []
    for r in passing:
        p = outdir / "skip1_cells" / f"{r['cell_id']}.json"
        if p.exists():
            rows.append(json.loads(p.read_text()))
        else:
            missing.append(r["cell_id"])
    by_fs_h = {}
    for row in rows:
        key = f"{row['feature_set']}__h{row['h']}"
        d = by_fs_h.setdefault(key, {"official_ga_pass": 0, "skip1_ga_pass": 0})
        d["official_ga_pass"] += 1
        d["skip1_ga_pass"] += int(row["skip1"]["ga_pass"])
    by_fs = {}
    for row in rows:
        d = by_fs.setdefault(row["feature_set"], {"official_ga_pass": 0, "skip1_ga_pass": 0})
        d["official_ga_pass"] += 1
        d["skip1_ga_pass"] += int(row["skip1"]["ga_pass"])
    gb_verdicts = {}
    for row in rows:
        if "skip1_gb" in row:
            v = row["skip1_gb"]["gb"]["verdict"]
            gb_verdicts[v] = gb_verdicts.get(v, 0) + 1
    repro = [row["repro_k0"]["abs_diff_mean_ic"] for row in rows if "repro_k0" in row]
    out = {
        "POSTHOC": "NOT PRE-REGISTERED -- one-day-skip label robustness re-score, written after "
                   "the official G-A results were seen. Does not amend SPEC.md gates.",
        "official_dir": str(official_dir),
        "official_done_marker_present": (Path(official_dir) / "DONE").exists(),
        "n_official_cells_read": len(official),
        "n_official_ga_pass": len(passing),
        "n_rescored": len(rows), "missing_cells": missing,
        "complete": (not missing) and (Path(official_dir) / "DONE").exists()
                    and len(official) == len(grid.ALL_CELLS),
        "label_shift_rows": 1,
        "ga_rule": "gates.ga_pass unchanged (mean_ic>=0.02, NW-t>=Bonferroni crit, IC>0 each year)",
        "summary_by_feature_set": by_fs,
        "summary_by_feature_set_and_h": by_fs_h,
        "n_skip1_ga_pass": sum(int(r["skip1"]["ga_pass"]) for r in rows),
        "skip1_gb_verdict_counts": gb_verdicts,
        "repro_k0_ridge_max_abs_diff_mean_ic": max(repro) if repro else None,
        "repro_k0_ridge_n": len(repro),
        "cells": rows,
    }
    (outdir / "posthoc_skip1.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"[915-skip1] finalize: rescored={len(rows)}/{len(passing)} "
          f"skip1_pass={out['n_skip1_ga_pass']} by_fs={json.dumps(by_fs)}", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--official-dir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--finalize-only", action="store_true")
    ap.add_argument("--no-repro", action="store_true")
    ap.add_argument("--follow-src", default=None,
                    help="official run dir to copy (read-only) into --official-dir until DONE")
    ap.add_argument("--deadline-epoch", type=float, default=None)
    a = ap.parse_args()
    run(a.official_dir, a.outdir, finalize_only=a.finalize_only, repro_ridge=not a.no_repro,
        follow_src=a.follow_src, deadline_epoch=a.deadline_epoch)
