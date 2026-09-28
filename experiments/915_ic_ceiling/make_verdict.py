#!/usr/bin/env python3
"""TASK_915 VERDICT.json -- computed from the result files, never hand-written.

Inputs : results/grid_results_annotated.json (official run + annotation)
         results/posthoc_skip1.json          (POST-HOC one-day-skip re-score)
Output : results/VERDICT.json

Pre-registered outcome = official G-A / G-B only (SPEC.md Sec.9, binding G-B column
restated_916). The skip1 re-score is POST-HOC and is reported beside the pre-registered
outcome, never substituted for it. Status-string rule (PROVENANCE, corrected run-57 closing
pass: this file's last save on srv, 22:18:45, is ~48 s AFTER results/posthoc_skip1.json was
written, 22:17:57, and no earlier draft is preserved -- so "fixed before the skip1 table was read"
is NOT established. The rule is post-hoc; it drives only the status string, never the
pre-committed consequence key, which is computed from the official G-B verdicts alone):
  GA_{PASS|FAIL}__GB_{PASS|MARGINAL|FAIL|NOT_EVALUATED}
  + per feature set with >=1 official G-A pass: `<FS>_TIMING_ARTIFACT` if 0 of its passes
    survive skip1, `<FS>_TIMING_FRAGILE` if fewer than half survive, nothing otherwise
  + `SKIP1_SURVIVORS_<n>` and, if any skip1 G-B was run, `SKIP1_GB_<best verdict>`
  + `PARTIAL` if the official run or the skip1 re-score is incomplete.
"""
import argparse
import json
from pathlib import Path

GB_RANK = {"PASS": 2, "MARGINAL": 1, "FAIL": 0}

CONSEQUENCES = {  # task file TASK_915 "Pre-committed consequences (D79)", verbatim
    "GA_FAIL": "G-A fails everywhere -> recommend repealing the OHLC-only constraint (D4); the "
               "next row is a DATA row (funding/basis, on-chain flow, or the news-LLM hybrid), "
               "not a model row. No further policy machinery on daily bars, any architecture, "
               "MuZero included.",
    "GA_PASS_GB_FAIL": "G-A passes / G-B fails -> record the measured IC/N_eff/turnover triple; "
                       "breadth/cost are operator decisions with numbers; no trader build.",
    "GB_PASS": "G-B passes -> G-C (the GP book, both instruments) is the next row "
               "(PROVISIONAL: G-B is borrow-uncharged; re-grade with 916 borrow first).",
}


def _count(cells, key):
    out = {}
    for c in cells:
        k = str(c[key])
        d = out.setdefault(k, {"n_cells": 0, "n_ga_pass": 0})
        d["n_cells"] += 1
        d["n_ga_pass"] += int(c["ga_pass"])
    return out


def build(annotated, skip1):
    cells = annotated["cells"]
    summ = annotated["summary"]
    n_pass = sum(int(c["ga_pass"]) for c in cells)
    by_fs_h = {}
    for c in cells:
        k = f"{c['feature_set']}__h{c['h']}"
        d = by_fs_h.setdefault(k, {"n_cells": 0, "n_ga_pass": 0})
        d["n_cells"] += 1
        d["n_ga_pass"] += int(c["ga_pass"])

    ic_by_id = {c["cell_id"]: c["ga_stats"]["mean_ic"] for c in cells}
    gb_rows = []
    for g in summ["gb_results"]:
        b = g["gb"]["restated_916"]
        gb_rows.append({"cell_id": g["cell_id"], "verdict": g["gb"]["verdict"],
                        "mean_ic": ic_by_id.get(g["cell_id"]), "n_eff": g["n_eff"],
                        "turnover_annualized": b["turnover_annualized"],
                        "net_sharpe_bound_restated_916": b["net_sharpe_bound"],
                        "net_sharpe_bound_legacy_21bp": g["gb"]["legacy_21bp"]["net_sharpe_bound"],
                        "trading_rate_a_star": b["trading_rate_a_star"],
                        "borrow_charged": False})
    gb_counts = {}
    for r in gb_rows:
        gb_counts[r["verdict"]] = gb_counts.get(r["verdict"], 0) + 1
    finite = [r for r in gb_rows if isinstance(r["net_sharpe_bound_restated_916"], (int, float))
              and r["net_sharpe_bound_restated_916"] == r["net_sharpe_bound_restated_916"]]
    best_gb = max(finite, key=lambda r: r["net_sharpe_bound_restated_916"]) if finite else None
    gb_best_verdict = (max(gb_counts, key=lambda v: GB_RANK[v]) if gb_counts
                       else "NOT_EVALUATED")

    ga_tok = "GA_PASS" if n_pass > 0 else "GA_FAIL"
    tokens = [ga_tok, f"GB_{gb_best_verdict}"]

    # POST-HOC skip1
    s_cells = skip1["cells"] if skip1 else []
    fs_surv = {}
    for s in s_cells:
        d = fs_surv.setdefault(s["feature_set"], {"official_ga_pass": 0, "skip1_ga_pass": 0})
        d["official_ga_pass"] += 1
        d["skip1_ga_pass"] += int(s["skip1"]["ga_pass"])
    for fs in ("F1", "F2", "F3", "F4", "ALL"):
        d = fs_surv.get(fs)
        if not d or d["official_ga_pass"] == 0:
            continue
        frac = d["skip1_ga_pass"] / d["official_ga_pass"]
        d["survival_fraction"] = frac
        if d["skip1_ga_pass"] == 0:
            tokens.append(f"{fs}_TIMING_ARTIFACT")
        elif frac < 0.5:
            tokens.append(f"{fs}_TIMING_FRAGILE")
    survivors = [s for s in s_cells if s["skip1"]["ga_pass"]]
    # informational (not part of the status rule): survival by target, and what skip1 says
    # about every official G-B cell that was NOT a FAIL
    tgt_surv = {}
    for s in s_cells:
        d = tgt_surv.setdefault(s["target"], {"official_ga_pass": 0, "skip1_ga_pass": 0})
        d["official_ga_pass"] += 1
        d["skip1_ga_pass"] += int(s["skip1"]["ga_pass"])
    s_by_id = {s["cell_id"]: s for s in s_cells}
    gb_nonfail_skip1 = []
    for r in gb_rows:
        if r["verdict"] == "FAIL":
            continue
        s = s_by_id.get(r["cell_id"])
        gb_nonfail_skip1.append({
            "cell_id": r["cell_id"], "official_gb_verdict": r["verdict"],
            "official_net_sharpe_restated_916": r["net_sharpe_bound_restated_916"],
            "skip1_rescored": s is not None,
            "skip1_mean_ic": s["skip1"]["mean_ic"] if s else None,
            "skip1_nw_t": s["skip1"]["nw_t"] if s else None,
            "skip1_ga_pass": s["skip1"]["ga_pass"] if s else None,
            "timing_fragile": s["timing_fragile"] if s else None})
    tokens.append(f"SKIP1_SURVIVORS_{len(survivors)}")
    s_gb = {}
    for s in survivors:
        if "skip1_gb" in s:
            v = s["skip1_gb"]["gb"]["verdict"]
            s_gb[v] = s_gb.get(v, 0) + 1
    if s_gb:
        tokens.append(f"SKIP1_GB_{max(s_gb, key=lambda v: GB_RANK[v])}")
    official_complete = summ["n_cells_run"] == 160
    skip1_complete = bool(skip1 and skip1.get("complete"))
    if not (official_complete and skip1_complete):
        tokens.append("PARTIAL")
    status = "__".join(tokens)

    if ga_tok == "GA_FAIL":
        cons_key = "GA_FAIL"
    elif gb_best_verdict == "PASS":
        cons_key = "GB_PASS"
    else:
        cons_key = "GA_PASS_GB_FAIL"

    skip1_table = [{"cell_id": s["cell_id"], "feature_set": s["feature_set"], "h": s["h"],
                    "target": s["target"], "estimator": s["estimator"],
                    "official_mean_ic": s["official"]["mean_ic"],
                    "official_nw_t": s["official"]["nw_t"],
                    "skip1_mean_ic": s["skip1"]["mean_ic"], "skip1_nw_t": s["skip1"]["nw_t"],
                    "skip1_by_year": s["skip1"]["by_year_mean_ic"],
                    "skip1_ga_pass": s["skip1"]["ga_pass"],
                    "skip1_gb_verdict": s.get("skip1_gb", {}).get("gb", {}).get("verdict"),
                    "skip1_gb_net_sharpe_restated_916":
                        s.get("skip1_gb", {}).get("gb", {}).get("restated_916", {})
                        .get("net_sharpe_bound")}
                   for s in s_cells]

    return {
        "task": "TASK_915",
        "status": status,
        "status_rule": "see make_verdict.py docstring (fixed before the full skip1 table was read)",
        "preregistered": {
            "spec_sha256_pinned": "d9b7d4050345adee403e41c5918edce0957418f740a3d1739476c0d6b8c2c95e",
            "ga": {"n_cells_run": summ["n_cells_run"], "n_ga_pass": n_pass,
                   "bonferroni_t_crit": cells[0]["ga_stats"]["bonferroni_t_crit"] if cells else None,
                   "by_feature_set": _count(cells, "feature_set"), "by_h": _count(cells, "h"),
                   "by_target": _count(cells, "target"), "by_feature_set_and_h": by_fs_h},
            "gb": {"binding": "restated_916 (80bp base-taker on turnover)",
                   "borrow_charged": False, "provisional_if_pass": True,
                   "verdict_counts": gb_counts, "best_verdict": gb_best_verdict,
                   "best_cell_by_net_sharpe": best_gb,
                   "ic_neff_turnover_triples": gb_rows},
        },
        "posthoc_skip1": {
            "POSTHOC": "NOT PRE-REGISTERED; does not change the pre-registered outcome",
            "complete": skip1_complete,
            "n_rescored": len(s_cells), "n_survivors": len(survivors),
            "survival_by_feature_set": fs_surv,
            "survivor_gb_verdict_counts": s_gb,
            "survival_by_target": tgt_surv,
            "official_gb_nonfail_cells_under_skip1": gb_nonfail_skip1,
            "table": skip1_table,
            "cause_of_timing_sensitivity": "same-close alignment: F1 features end at close_t and "
                "labels start at close_t (bid-ask bounce / stale-close consistent, not "
                "individually established)",
        },
        "precommitted_consequence_key": cons_key,
        "precommitted_consequence": CONSEQUENCES[cons_key],
        "consequence_note": ("best G-B verdict is MARGINAL (>= +0.56, < +0.97): the task's "
                             "consequence list has no MARGINAL branch; mapped to the not-PASS "
                             "branch, flagged for the operator") if gb_best_verdict == "MARGINAL"
                            else None,
        "precommitted_consequence_source": "tasks/TASK_915_IC_CEILING_AND_GP_BOOK.md, "
                                           "'Pre-committed consequences (adopted with the memo -- D79)'",
        "decision_owner": "operator",
        "mde_both_conventions": annotated.get("annotation", {}).get("mde_both_conventions"),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotated", default=str(Path(__file__).resolve().parent / "results"
                                                / "grid_results_annotated.json"))
    ap.add_argument("--skip1", default=str(Path(__file__).resolve().parent / "results"
                                            / "posthoc_skip1.json"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "results"
                                          / "VERDICT.json"))
    a = ap.parse_args()
    ann = json.loads(Path(a.annotated).read_text())
    sk = json.loads(Path(a.skip1).read_text()) if Path(a.skip1).exists() else None
    v = build(ann, sk)
    Path(a.out).write_text(json.dumps(v, indent=2, default=str))
    print(json.dumps({k: v[k] for k in ("status", "precommitted_consequence_key",
                                         "decision_owner")}, indent=2))
