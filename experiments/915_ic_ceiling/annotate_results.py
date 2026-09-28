#!/usr/bin/env python3
"""POST-HOC annotation (TASK_915 cycle 2, 2026-09-28). Reads a COPY of the official run's
outputs (never the official dir itself) plus results/posthoc_skip1.json, and writes
results/grid_results_annotated.json. Adds, without changing any official number:

  per G-B entry : borrow_charged=False, provisional_if_pass=True (G-B charges the 80bp
                  base-taker rate on turnover only; 916's borrow is not charged -- SPEC.md Sec.8)
  per cell      : underpowered = the cell horizon's MDE flag at BOTH t conventions
                  (t=3.0, the value the frozen mde_report.py hard-coded; t=3.6047, the
                  Bonferroni t_crit SPEC.md Sec.7 states), and
                  timing_fragile = official G-A pass that fails G-A under the one-day-skip
                  label (posthoc_skip1.py); None where not tested (official G-A fail).

Usage: python annotate_results.py --official-copy /var/tmp/915-fix/official_copy \
          --skip1 results/posthoc_skip1.json --out results/grid_results_annotated.json
"""
import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import stats_utils as su   # noqa: E402

ALPHA = 0.05
N_TESTS = 160     # the official run's value (frozen gates.py/mde_report.py literal)
BAND = 0.02
T_PINNED = 3.0


def mde_both(mde_tbl, alpha=ALPHA, n_tests=N_TESTS, band=BAND):
    """Recompute the pinned MDE table at t=3.0 (as pinned) and at the Bonferroni t_crit, from
    the SAME panel counts / NW-inflation convention (stats_utils.mde_ic, power_t varied)."""
    t_crit = su.bonferroni_t_crit(alpha, n_tests)
    out = {}
    for h, row in mde_tbl.items():
        args = (row["n_bar_live_assets"] - 1, row["n_dates"], alpha, n_tests, row["nw_inflation"])
        m3, _ = su.mde_ic(*args, power_t=T_PINNED)
        mc, _ = su.mde_ic(*args, power_t=t_crit)
        out[str(h)] = {
            "n_dates": row["n_dates"], "n_bar_live_assets": row["n_bar_live_assets"],
            "nw_inflation": row["nw_inflation"],
            "mde_ic_at_t3_pinned": m3, "underpowered_at_t3_pinned": bool(m3 > band),
            "t_crit_bonferroni": t_crit,
            "mde_ic_at_t_crit": mc, "underpowered_at_t_crit": bool(mc > band),
            "flag_flips": bool((m3 > band) != (mc > band)),
            "pinned_table_value_reproduced": abs(m3 - row["mde_ic_at_t3"]) < 1e-12,
        }
    return out


def annotate(official_copy, skip1_path):
    official_copy = Path(official_copy)
    payload = json.loads((official_copy / "grid_results.json").read_text())
    mde = mde_both(payload["summary"]["mde_table"])
    skip1 = json.loads(Path(skip1_path).read_text()) if skip1_path else None
    skip_by_id = {c["cell_id"]: c for c in (skip1["cells"] if skip1 else [])}

    for cell in payload["cells"]:
        m = mde[str(cell["h"])]
        cell["underpowered"] = {"t3.0_pinned": m["underpowered_at_t3_pinned"],
                                "t_crit_3.6047": m["underpowered_at_t_crit"]}
        s = skip_by_id.get(cell["cell_id"])
        if not cell["ga_pass"]:
            cell["timing_fragile"] = None
            cell["timing_fragile_note"] = "not tested (official G-A fail)"
        elif s is None:
            cell["timing_fragile"] = None
            cell["timing_fragile_note"] = "official G-A pass NOT yet re-scored (partial skip1 run)"
        else:
            cell["timing_fragile"] = s["timing_fragile"]
            cell["skip1_mean_ic"] = s["skip1"]["mean_ic"]
            cell["skip1_nw_t"] = s["skip1"]["nw_t"]
            cell["skip1_ga_pass"] = s["skip1"]["ga_pass"]

    for g in payload["summary"]["gb_results"]:
        g["gb"].setdefault("borrow_charged", False)
        g["gb"].setdefault("provisional_if_pass", True)
        g["borrow_charged"] = False
        g["provisional_if_pass"] = True

    payload["annotation"] = {
        "POSTHOC": "annotation added after the official run; official numbers unchanged",
        "source": str(official_copy / "grid_results.json"),
        "mde_both_conventions": mde,
        "skip1_source": str(skip1_path) if skip1_path else None,
        "skip1_complete": skip1["complete"] if skip1 else False,
        "borrow_note": "every G-B verdict is borrow-UNcharged (80bp base-taker on turnover only);"
                       " a PASS is provisional until re-graded with 916 borrow",
    }
    return payload


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--official-copy", required=True)
    ap.add_argument("--skip1", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = annotate(a.official_copy, a.skip1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out["annotation"]["mde_both_conventions"], indent=2))
