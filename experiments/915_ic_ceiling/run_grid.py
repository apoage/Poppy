#!/usr/bin/env python3
"""TASK_915 grid runner. Loads the PIT OHLCV panel (fixture + 858 seam), builds features/
targets once, iterates the 160-cell grid with PER-CELL CHECKPOINTING (a completed cell's JSON
under --outdir/cells/ is skipped on resume -- an interruption is a resume point, not a loss),
evaluates G-A for every cell and G-B only for G-A-passing cells (per tonight's scope: G-C is
NOT built here), and writes a final grid_results.json + a done-marker file.

Usage:
  trader_env/bin/python run_grid.py --outdir /var/tmp/915-run --cells all
  trader_env/bin/python run_grid.py --outdir /var/tmp/915-run --cells 3   # smoke: first N cells
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import loader           # noqa: E402
import grid             # noqa: E402
import gates            # noqa: E402
import stats_utils as su  # noqa: E402
import mde_report        # noqa: E402

COST_BP_RESTATED_916 = 80.0     # base_taker80 tier; borrow charge applied via 916's
                                  # borrow_window_report methodology on the GRADED book (G-C,
                                  # not this row) -- here the GP simulation's linear cost model
                                  # uses the 80bp base-taker figure as the binding rate cost;
                                  # see SPEC.md Sec.'Denominations' for the borrow disclosure.


def _cell_id(t, h, fs, est):
    return f"{t}__h{h}__{fs}__{est}"


def run(outdir, n_cells=None, first_test_date="2022-01-01", last_date="2026-06-30",
        seed=0):
    outdir = Path(outdir)
    (outdir / "cells").mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("[915] loading panel (fixture + 858 seam)...", flush=True)
    panel = loader.load_panel()
    fixture_sha = panel["fixture_sha256"]
    seam = panel["seam_block"]
    print(f"[915] fixture sha256={fixture_sha[:12]}... seam mismatches={seam['mismatches_n']} "
          f"overlap_cells={seam['overlap_cells_total']}", flush=True)
    panel = loader.apply_liquidity_filter(panel)
    print(f"[915] universe after liquidity filter: {len(panel['syms'])} assets, "
          f"{panel['day_ts'].size} days", flush=True)

    print("[915] building present-mask, targets (4 horizons), features (F1-F4)...", flush=True)
    built = grid.build_panel_features_and_targets(panel)

    mde_tbl = mde_report.mde_table(built["present"], panel["day_ts"])
    (outdir / "mde_table.json").write_text(json.dumps(mde_tbl, indent=2))
    print(f"[915] MDE table: {json.dumps(mde_tbl)}", flush=True)

    cells = grid.ALL_CELLS if n_cells is None else grid.ALL_CELLS[:n_cells]
    results = []
    ga_pass_cells = []
    for i, (tname, h, fs, est) in enumerate(cells):
        cid = _cell_id(tname, h, fs, est)
        cell_path = outdir / "cells" / f"{cid}.json"
        if cell_path.exists():
            results.append(json.loads(cell_path.read_text()))
            continue
        tc0 = time.time()
        res = grid.fit_cell(built, panel["day_ts"], h, tname, fs, est,
                             first_test_date=first_test_date, last_date=last_date)
        target_panel = built["targets_by_h"][h][tname]
        stats = gates.cell_ic_stats(res["forecast"], target_panel, panel["day_ts"], h)
        passed, reason = gates.ga_pass(stats)
        dt_cell = time.time() - tc0
        row = {"cell_id": cid, "target": tname, "h": h, "feature_set": fs, "estimator": est,
               "n_quarters_fit": len(res["quarters"]), "ga_stats": stats,
               "ga_pass": passed, "ga_reason": reason, "fit_seconds": dt_cell,
               "columns_used": res["columns"]}
        cell_path.write_text(json.dumps(row))
        results.append(row)
        print(f"[915] cell {i+1}/{len(cells)} {cid}: GA={'PASS' if passed else 'fail'} "
              f"mean_ic={stats['mean_ic']:.4f} nw_t={stats['nw_t']:.2f} "
              f"({dt_cell:.1f}s)", flush=True)
        if passed:
            ga_pass_cells.append((row, res, target_panel))

    print(f"[915] {len(ga_pass_cells)}/{len(cells)} cells cleared G-A. Evaluating G-B...",
          flush=True)
    gb_results = []
    for row, res, target_panel in ga_pass_cells:
        h = row["h"]
        lo, hi = gates._window_rows(panel["day_ts"], *gates.GA_WINDOW)
        forecast_win = res["forecast"][lo:hi]
        fwd_ret_win = built["targets_by_h"][h]["r_fwd"][lo:hi]
        resid_fwd_win = built["targets_by_h"][h]["resid_fwd"][lo:hi]
        finite_cols = np.isfinite(forecast_win).any(axis=0) & np.isfinite(fwd_ret_win).any(axis=0)
        resid = resid_fwd_win[:, finite_cols]   # market-residual returns -> N_eff's own spectrum
        resid_ok_rows = np.isfinite(resid).all(axis=1)
        n_eff = np.nan
        sigma_diag_full = np.full(res["forecast"].shape[1], np.nan)
        if resid_ok_rows.sum() >= 30 and finite_cols.sum() >= 5:
            resid_clean = resid[resid_ok_rows]
            n_eff, sigma_mat, shrinkage, corr = gpl_n_eff_wrapper(resid_clean)
            sigma_diag_full[finite_cols] = np.diag(sigma_mat)
        gb = gates.gb_result(row["ga_stats"]["mean_ic"], n_eff, h, forecast_win, fwd_ret_win,
                              sigma_diag_full, COST_BP_RESTATED_916)
        gb_row = {"cell_id": row["cell_id"], "n_eff": n_eff, "gb": gb}
        gb_results.append(gb_row)
        print(f"[915] G-B {row['cell_id']}: n_eff={n_eff:.2f} verdict={gb['verdict']} "
              f"net_sharpe_bound={gb['binding']}", flush=True)

    sr0 = su.sr0_multiplicity(len(cells))
    summary = {
        "n_cells_run": len(cells), "n_ga_pass": len(ga_pass_cells),
        "sr0_multiplicity_line": sr0, "mde_table": mde_tbl,
        "gb_results": gb_results,
        "fixture_sha256": fixture_sha, "seam_block": seam,
        "loader_sha256": panel["loader_sha256"],
        "cost_bp_restated_916": COST_BP_RESTATED_916,
        "elapsed_seconds": time.time() - t0,
    }
    (outdir / "grid_results.json").write_text(
        json.dumps({"summary": summary, "cells": results}, default=str, indent=2))
    (outdir / "DONE").write_text(f"done at {time.time()}\n")
    print(f"[915] DONE. {len(ga_pass_cells)}/{len(cells)} G-A pass. "
          f"elapsed={summary['elapsed_seconds']:.1f}s", flush=True)
    return summary


def gpl_n_eff_wrapper(residual_clean):
    import gp_fundamental_law as gpl
    return gpl.residual_n_eff_and_sigma(residual_clean)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--cells", default="all", help="'all' or an integer N (first N cells)")
    ap.add_argument("--first-test-date", default="2022-01-01")
    ap.add_argument("--last-date", default="2026-06-30")
    args = ap.parse_args()
    n_cells = None if args.cells == "all" else int(args.cells)
    run(args.outdir, n_cells=n_cells, first_test_date=args.first_test_date,
        last_date=args.last_date)
