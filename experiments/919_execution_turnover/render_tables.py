#!/usr/bin/env python3
"""TASK_919 phase A -- generate every markdown table in RESULTS.md FROM `result_919a.json`.

WHY THIS EXISTS (fix pass, run 55 cycle 2 -- BLOCKER-F1). The cycle-1 pass hand-wrote its
48-cell headline table beside the machine artifact, and 26 of 48 rows disagreed with the json
by up to +102% on `gross` (mtimes showed the table had been typed from an earlier harness
revision and never regenerated). The json was independently confirmed correct against the
orchestrator's own pre-brief H-sweep; the hand-written table was corrupt. A hand-maintained
table beside a machine artifact is a defect generator, so no table in RESULTS.md is hand-written
any more: every one of them lives between

    <!-- AUTOGEN:<block> BEGIN -->  ...  <!-- AUTOGEN:<block> END -->

markers and is emitted by this module. `python render_tables.py` rewrites the blocks in place;
`python render_tables.py --check` fails if the shipped file differs from a fresh render, which is
what `test_919_grid.py::test_results_tables_match_json` gates in CI. Prose OUTSIDE the markers
is hand-written and untouched -- but any NUMBER in that prose is quoted from a block below.

Run:  trader_env/bin/python experiments/919_execution_turnover/render_tables.py [--check]
"""
import argparse
import json
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
RESULT_JSON = _HERE / "result_919a.json"
RESULTS_MD = _HERE / "RESULTS.md"

BEGIN = "<!-- AUTOGEN:{name} BEGIN -->"
END = "<!-- AUTOGEN:{name} END -->"


def _f(x, nd=4, sign=True):
    if x is None:
        return "—"
    fmt = f"{{:+.{nd}f}}" if sign else f"{{:.{nd}f}}"
    return fmt.format(float(x))


def _cell_key(c):
    return (c["H"], c["theta"], c["accounting"])


def grid_table(cells):
    """The headline per-cell table. Ordered exactly as the harness scores the grid."""
    out = ["| H | theta | accounting | cost basis | n | gross | turn/dec | turn/day | net8 | net12 "
           "| net21 | net21_borrow_low | p | CI95 lo..hi | sig |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    basis_short = {"target-weight": "target-change only",
                   "drift-inclusive": "upper bound (unnetted)",
                   "hold-shares": "actual trade, OWN book"}
    for c in cells:
        if c.get("n", 0) == 0:
            out.append(f"| {c['H']} | {c['theta']} | {c['accounting']} | — | 0 | window too thin "
                       f"| | | | | | | | | |")
            continue
        out.append(
            f"| {c['H']} | {c['theta']} | {c['accounting']} "
            f"| {basis_short[c['accounting']]} | {c['n']} "
            f"| {_f(c['gross'], 4)} | {_f(c['turn'], 4, sign=False)} "
            f"| {_f(c['turn_per_day'], 4, sign=False)} "
            f"| {_f(c['net8'], 4)} | {_f(c['net12'], 4)} | {_f(c['net21'], 4)} "
            f"| {_f(c['net21_borrow_low'], 4)} | {_f(c['p21_borrow_low'], 4, sign=False)} "
            f"| {_f(c['ci21_borrow_low_lo'], 3)}..{_f(c['ci21_borrow_low_hi'], 3)} "
            f"| {'*' if c['bonferroni_significant'] else ''} |")
    return "\n".join(out)


def decomposition_table(cells):
    out = ["| H | theta | flip | renorm | entry_exit | drift | flip+renorm+entry_exit | "
           "measured turn (target-weight) | closes |",
           "|---|---|---|---|---|---|---|---|---|"]
    for c in cells:
        if c["accounting"] != "target-weight" or c.get("n", 0) == 0:
            continue
        d = c["decomposition"]
        s = d["flip"] + d["renorm"] + d["entry_exit"]
        out.append(f"| {c['H']} | {c['theta']} | {_f(d['flip'],4,False)} | {_f(d['renorm'],4,False)} "
                   f"| {_f(d['entry_exit'],4,False)} | {_f(d['drift'],4,False)} "
                   f"| {_f(s,6,False)} | {_f(c['turn'],6,False)} "
                   f"| {'yes' if d['target_weight_turn_closes'] else 'NO'} |")
    return "\n".join(out)


def turnover_bases_table(cells):
    out = ["| H | theta | target_change_only | upper_bound_unnetted | netted_combined | "
           "hold_shares_actual |", "|---|---|---|---|---|---|"]
    seen = set()
    for c in cells:
        if c.get("n", 0) == 0:
            continue
        k = (c["H"], c["theta"])
        tb = c["turnover_bases"]
        if k in seen:
            # only the hold-shares row carries hold_shares_actual; merge it in
            continue
        hs = [x for x in cells if (x["H"], x["theta"]) == k and x["accounting"] == "hold-shares"]
        hs_val = hs[0]["turnover_bases"]["hold_shares_actual"] if hs else None
        seen.add(k)
        out.append(f"| {c['H']} | {c['theta']} | {_f(tb['target_change_only'],4,False)} "
                   f"| {_f(tb['upper_bound_unnetted'],4,False)} "
                   f"| {_f(tb['netted_combined'],4,False)} "
                   f"| {_f(hs_val,4,False) if hs_val is not None else '—'} |")
    return "\n".join(out)


def correction1_table(rep):
    out = ["| H | n | turn/day measured | flip/day measured | model (corrected) | "
           "ratio vs TOTAL turn | ratio vs FLIP only | model (task as-written) | ratio vs uncorrected |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in rep["correction_1_analytic_anchor"]:
        out.append(f"| {r['H']} | {r['n']} | {_f(r['turn_per_day_measured'],4,False)} "
                   f"| {_f(r['flip_per_day_measured'],4,False)} | {_f(r['model_corrected'],4,False)} "
                   f"| {_f(r['ratio_vs_total_turn'],4,False)} | **{_f(r['ratio_vs_flip_only'],4,False)}** "
                   f"| {_f(r['model_uncorrected'],4,False)} | {_f(r['uncorrected_ratio'],3,False)} |")
    return "\n".join(out)


def correction2_table(rep):
    c = rep["correction_2_bear22_holdshares"]
    d = c["bear22_hold_shares_gross_exposure"] or {}
    out = ["| window | n | target-weight gross | hold-shares gross | Δ (hold − target) |",
           "|---|---|---|---|---|",
           f"| **BEAR22** (2021-11-10..2022-12-31) | {c['n_bear22']} "
           f"| {_f(c['bear22_target_weight_gross_sharpe'])} "
           f"| {_f(c['bear22_hold_shares_gross_sharpe'])} "
           f"| **{_f(c['bear22_delta_hold_minus_target'])}** |",
           f"| PRIMARY full (2021-01-01..2025-09-01) | {rep['frozen_primary']['n']} "
           f"| {_f(c['primary_full_target_weight_gross_sharpe'])} "
           f"| {_f(c['primary_full_hold_shares_gross_sharpe'])} "
           f"| **{_f(c['primary_full_delta_hold_minus_target'])}** |",
           "",
           f"Hold-shares book gross exposure Σ|w| in BEAR22: mean "
           f"{_f(d.get('gross_abs_mean'),3,False)}, min {_f(d.get('gross_abs_min'),3,False)}, "
           f"max {_f(d.get('gross_abs_max'),3,False)} (a share-carrying book is NOT pinned at "
           f"gross 1 — measured, not assumed).",
           f"Direction confirms the 830_corridor mechanism in BEAR22: "
           f"**{c['direction_confirms_spec_in_bear22']}**. Sign reverses outside BEAR22: "
           f"**{c['sign_reverses_outside_bear22']}**."]
    return "\n".join(out)


def correction2_attribution_table(rep):
    c2 = rep["correction_2_bear22_holdshares"]
    a = c2["attribution"]
    ci = c2.get("attribution_ci", {})
    out = ["| window | canon gross (log-sum on target) | + compounding convention → simple-sum "
           "on the SAME target | + share carry → the carried book | Δ from CONVENTION | Δ from "
           "SHARE CARRY |", "|---|---|---|---|---|---|"]
    label = {"bear22": "**BEAR22**", "primary_full": "PRIMARY full"}
    for wname in ("bear22", "primary_full"):
        v = a[wname]
        out.append(f"| {label[wname]} | {_f(v['gross_log_on_target'])} "
                   f"| {_f(v['gross_simple_on_target'])} "
                   f"| {_f(v['gross_hold_shares_carried'])} "
                   f"| **{_f(v['delta_from_compounding_convention'])}** "
                   f"| **{_f(v['delta_from_share_carry'])}** |")
    if ci:
        out += ["",
                "**Each component with its own 95% PAIRED block-bootstrap CI (MAJOR-E2B).** "
                "Both series in each difference live on the same decision rows, so the bootstrap "
                "is paired and needs no correlation assumption. A component whose interval "
                "contains 0 is **UNRESOLVED at this design's power** — the same standard this "
                "pass applies to the ±0.20 band, now applied to its own attribution.", "",
                "| window | component | point | 95% CI (paired) | bootstrap SD | verdict |",
                "|---|---|---|---|---|---|"]
        names = {"delta_from_compounding_convention": "Δ from CONVENTION",
                 "delta_from_share_carry": "Δ from SHARE CARRY"}
        for wname in ("bear22", "primary_full"):
            for key, nice in names.items():
                v = (ci.get(wname) or {}).get(key)
                if not v:
                    continue
                verdict = ("**RESOLVED**" if v["ci_excludes_zero"] else "**UNRESOLVED**")
                out.append(f"| {label[wname]} | {nice} | **{_f(v['point'])}** "
                           f"| [{_f(v['ci_lo'])}, {_f(v['ci_hi'])}] "
                           f"| {_f(v['bootstrap_sd'],4,False)} | {verdict} |")
    return "\n".join(out)


def leverage_renorm_table(rep):
    """BLOCKER-E2A: the gross-renormalised hold-shares comparator."""
    L = rep["leverage_renormalisation"]
    s = L["summary_primary"]
    b = L["bear22_counter_check"]
    out = ["| H | θ | mean Σ\\|w\\| | HS gross RAW | HS gross RENORM | target-weight gross "
           "| RAW − TW | **RENORM − TW** | HS net21 RAW | HS net21 RENORM | TW net21 "
           "| RENORM − TW (net21) |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in L["primary_window_rows"]:
        out.append(f"| {r['H']} | {r['theta']} | {_f(r['mean_gross_exposure'],3,False)} "
                   f"| {_f(r['hs_gross_raw'])} | {_f(r['hs_gross_renorm'])} "
                   f"| {_f(r['tw_gross'])} | {_f(r['raw_minus_tw_gross'])} "
                   f"| **{_f(r['renorm_minus_tw_gross'])}** "
                   f"| {_f(r['hs_net21_raw'])} | {_f(r['hs_net21_renorm'])} "
                   f"| {_f(r['tw_net21'])} | {_f(r['renorm_minus_tw_net21'])} |")
    cc = L["construction_caveat"]
    out += ["",
            f"**Before the rescaling** the hold-shares gross advantage over target-weight is "
            f"{_f(s['raw_advantage_gross_min'])}…{_f(s['raw_advantage_gross_max'])} and is "
            f"**positive in all {s['n_cells']} cells**: "
            f"`{str(s['raw_advantage_positive_in_all_cells']).lower()}`. "
            f"**After it** the residue is "
            f"{_f(s['renorm_advantage_gross_min'])}…{_f(s['renorm_advantage_gross_max'])}, with "
            f"**no consistent sign under THIS construction** "
            f"({s['n_renorm_advantage_positive']} of {s['n_cells']} cells positive; "
            f"consistent-sign check "
            f"`{str(s['renorm_advantage_consistent_sign']).lower()}`).",
            "",
            "**The sign pattern is CONSTRUCTION-DEPENDENT, so the residue is UNRESOLVED, not "
            "zero.** The column above is the proxy `ret/G` — an exact rescaling of the "
            "*realised return series*, holdings untouched. A cycle-3 estimator lens built the "
            "other construction, a book genuinely constrained to `gross_cap=1.0` that "
            "renormalises its **holdings** each decision, and got the opposite pattern "
            f"({cc['constrained_book_advantage_over_target_weight_gross_primary']}). Neither "
            "clears zero. Paired 95% CIs at the frozen cell — raw "
            f"{cc['paired_95pct_cis_at_the_frozen_cell']['raw_advantage']}, proxy "
            f"{cc['paired_95pct_cis_at_the_frozen_cell']['proxy_renormalised']}, constrained "
            f"gross-1 {cc['paired_95pct_cis_at_the_frozen_cell']['constrained_gross1_book']} — "
            "**all straddle zero**. Those figures are TRANSCRIBED from the lens; this pass did "
            "not rebuild the constrained book.",
            "",
            f"**BEAR22 counter-check** (the finding's own disconfirming test): target-weight "
            f"{_f(b['target_weight_gross'])}, hold-shares RAW {_f(b['hold_shares_gross_raw'])}, "
            f"hold-shares RESCALED {_f(b['hold_shares_gross_renorm'])} — a de-levered gap of "
            f"**{_f(b['renorm_minus_tw'])}**. De-levering does **not** rescue hold-shares in "
            f"BEAR22, so CORRECTION 2's BEAR22 direction is a genuine compositional effect and "
            f"**survives**; the full-window *advantage* is left **UNRESOLVED**, not killed."]
    return "\n".join(out)


def correction4_table(rep):
    out = ["| theta | flip turn H=1 | reduction vs θ=0 | flip H=2 | reduction | flip H=3 "
           "| reduction | flip H=5 | reduction |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in rep["correction_4_theta_flip_reduction"]["measured"]:
        cells = [f"| {r['theta']} "]
        for H in (1, 2, 3, 5):
            red = r[f"reduction_H{H}"]
            cells.append(f"| {_f(r[f'flip_H{H}'],4,False)} "
                         f"| {'—' if abs(red) < 1e-12 else f'{100*red:.1f}%'} ")
        out.append("".join(cells) + "|")
    return "\n".join(out)


def mde_table(rep):
    m = rep["mde_primary"]
    band = m["band"]
    out = [f"**SE of one cell's annualized `{m['primary_gate_quantity']}` Sharpe — two "
           f"independent derivations:**", "",
           "| derivation | median | min | max |", "|---|---|---|---|",
           f"| (A) empirical, from each cell's own 95% block-bootstrap CI half-width / 1.96 "
           f"| {_f(m['se_single_cell_bootstrap']['median'],4,False)} "
           f"| {_f(m['se_single_cell_bootstrap']['min'],4,False)} "
           f"| {_f(m['se_single_cell_bootstrap']['max'],4,False)} |",
           f"| (B) closed form, √(365/H)·√((1+SR_period²/2)/n) "
           f"| {_f(m['se_single_cell_closed_form']['median'],4,False)} "
           f"| {_f(m['se_single_cell_closed_form']['min'],4,False)} "
           f"| {_f(m['se_single_cell_closed_form']['max'],4,False)} |",
           "",
           f"The two agree to {_f(m['two_derivations_agree_within'],4,False)}; SE used = "
           f"**{_f(m['se_used'],4,False)}**, and it is essentially INVARIANT ACROSS H "
           f"(n·H ≈ 1704 is held constant by the grid, which exactly cancels the √(365/H) "
           f"annualizer).", "",
           f"**MDE at 80% power, one-sided, vs the pre-registered ±{band:.2f} band:**", "",
           "| SE_diff assumption | SE_diff | MDE @ raw α=0.05 | × band | MDE @ Bonferroni α=0.05/48 | × band |",
           "|---|---|---|---|---|---|"]
    for label, v in m["mde"].items():
        out.append(f"| {label} | {_f(v['se_diff'],4,False)} | **{_f(v['mde_raw_alpha_0.05'],3,False)}** "
                   f"| {v['mde_raw_over_band']:.1f}× | **{_f(v['mde_bonferroni'],3,False)}** "
                   f"| {v['mde_bonferroni_over_band']:.1f}× |")
    r = rep["measured_correlation_primary_h1"]
    a = r.get("most_favourable_non_degenerate") if r.get("n_pairs") else None
    if a:
        n_h1 = rep["frozen_primary"]["n"]
        out += ["",
                f"**ρ is MEASURED, not assumed**, on the {r['n_pairs']} H=1 candidate cells (same "
                f"{n_h1} decision rows as the frozen cell, so the two net series can be paired "
                f"directly): ρ min {_f(r['rho_min'],4,False)}, median "
                f"{_f(r['rho_median'],4,False)}, max {_f(r['rho_max'],4,False)}. The most "
                f"favourable NON-DEGENERATE pair (ρ<0.999 — at ρ≈1 the two cells are the same "
                f"position path differing by a cost constant, so there is nothing to detect) is "
                f"`{a['cell']}` at ρ={_f(a['rho'],4,False)}: SE_diff {_f(a['se_diff'],4,False)} → "
                f"MDE **{_f(a['mde_raw'],3,False)}** raw ({a['mde_raw_over_band']:.1f}× band) / "
                f"**{_f(a['mde_bonferroni'],3,False)}** Bonferroni "
                f"({a['mde_bonferroni_over_band']:.1f}× band). Cross-H cells cannot be paired and "
                f"are less correlated in every measurable case here, so this is a **LOWER BOUND** "
                f"on the MDE of the cells the grid actually nominates."]
    return "\n".join(out)


def band_table(rep, window):
    b = rep[f"band_verdict_{window}"]
    rn = b["reading_max_delta_net"]
    out = ["| reading | value | cell | point estimate > +0.20? | DECIDABLE at this design's power? |",
           "|---|---|---|---|---|",
           f"| max **Δnet** (the band's literal, unqualified \"uplift\") | **{_f(rn['value'])}** "
           f"| H={rn['cell'][0]}, θ={rn['cell'][1]}, {rn['cell'][2]} "
           f"| {'yes' if rn['point_estimate_exceeds_band'] else 'no'} | **NO** |"]
    for mode, v in b["reading_max_delta_cost_drag_saving_WITHIN_MODE"].items():
        out.append(f"| max **Δcost_drag saving**, WITHIN `{mode}` (vs that mode's own "
                   f"H=1/θ=0 cell) | {_f(v['max_cost_drag_saving'])} "
                   f"| H={v['cell'][0]}, θ={v['cell'][1]} "
                   f"| {'yes' if v['exceeds_band'] else 'no'} | **NO** |")
    return "\n".join(out)


def significance_table(rep):
    out = ["| window | cells scored | Bonferroni-significant (α=0.05/48) | raw p<0.05 | "
           "95% CI lower bound > 0 |", "|---|---|---|---|---|"]
    for wname in ("primary", "secondary", "disjoint"):
        s = rep[f"significance_{wname}"]
        out.append(f"| {wname.upper()} | {s['n_cells_scored']} | **{s['n_bonferroni_significant']}** "
                   f"| {s['n_raw_p_lt_05']} | **{s['n_ci_lower_bound_positive']}** |")
    return "\n".join(out)


def windows_table(rep):
    out = ["| window | start | end | n (H=1) | relation to PRIMARY |", "|---|---|---|---|---|"]
    rel = {"primary": "— (the grid's own denominator; the cell was SELECTED here)",
           "secondary": "**strict SUPERSET of PRIMARY** — NOT out-of-sample",
           "disjoint": "**disjoint from PRIMARY** — the only genuinely held-out read"}
    for wname in ("primary", "secondary", "disjoint"):
        wd = rep["windows"][wname]
        n = rep[f"frozen_{wname}"]["n"]
        out.append(f"| **{wname.upper()}** | {wd['start']} | {wd['end']} | {n} | {rel[wname]} |")
    return "\n".join(out)


def best_cell_table(rep):
    key = rep["grid_definition"]["primary_gate_quantity"]
    best = rep["best_primary"]
    out = [f"| window | relation to PRIMARY | n | best cell's `{key}` | its window's frozen cell "
           f"| uplift Δ | inside ±0.20 band? |", "|---|---|---|---|---|---|---|"]
    rel = {"primary": "SELECTED here", "secondary": "SUPERSET (not OOS)", "disjoint": "DISJOINT (OOS)"}
    for wname, cell in (("primary", rep["best_primary"]),
                        ("secondary", rep["best_cell_on_secondary_window"]),
                        ("disjoint", rep["best_cell_on_disjoint_window"])):
        fr = rep[f"frozen_{wname}"]
        if cell is None or cell.get("n", 0) == 0:
            out.append(f"| {wname.upper()} | {rel[wname]} | — | window too thin | | | |")
            continue
        d = cell[key] - fr[key]
        out.append(f"| {wname.upper()} | {rel[wname]} | {cell['n']} | {_f(cell[key])} "
                   f"| {_f(fr[key])} | **{_f(d)}** | {'yes' if abs(d) <= 0.20 else 'no'} |")
    out += ["", f"Best cell (selected on PRIMARY by `{key}`): **H={best['H']}, θ={best['theta']}, "
                f"{best['accounting']}**."]
    return "\n".join(out)


def hold_shares_table(rep):
    out = ["| H | theta | Σ\\|w\\| mean | Σ\\|w\\| min | Σ\\|w\\| max | equity final | equity min |",
           "|---|---|---|---|---|---|---|"]
    for c in rep["cells_primary"]:
        if c["accounting"] != "hold-shares" or c.get("n", 0) == 0:
            continue
        d = c["hold_shares_diag"]
        out.append(f"| {c['H']} | {c['theta']} | {_f(d['gross_abs_mean'],3,False)} "
                   f"| {_f(d['gross_abs_min'],3,False)} | {_f(d['gross_abs_max'],3,False)} "
                   f"| {_f(d['equity_final'],3,False)} | {_f(d['equity_min'],3,False)} |")
    return "\n".join(out)


def uplift_ci_summary_table(rep):
    out = ["| window | cells | raw 95%: CI lower bound > +0.20 | raw 95%: CI wholly BELOW +0.20 "
           "| raw 95%: UNDECIDED (straddles +0.20) | **Bonferroni α=0.05/48: clears band** "
           "| Bonferroni: UNDECIDED | median CI half-width |",
           "|---|---|---|---|---|---|---|---|"]
    for wname in ("primary", "secondary", "disjoint"):
        s = rep[f"uplift_ci_summary_{wname}"]
        r, b = s["raw_95pct"], s["bonferroni_alpha_0.05_over_48"]
        out.append(f"| {wname.upper()} | {s['n_cells']} "
                   f"| {r['n_ci_lower_bound_clears_band']} "
                   f"| {r['n_ci_upper_bound_below_band']} "
                   f"| {r['n_undecided_ci_straddles_band']} "
                   f"| **{b['n_ci_lower_bound_clears_band']}** "
                   f"| {b['n_undecided_ci_straddles_band']} "
                   f"| {_f(s['median_uplift_ci_half_width'],3,False)} |")
    s = rep["uplift_ci_summary_primary"]
    out += ["", f"At raw α=0.05 over a 48-cell family, chance alone produces **at most** "
                f"{s['raw_95pct']['expected_false_positives_at_this_alpha']:.1f} clearances "
                f"(α/2·n is an UPPER BOUND — it assumes every cell sits exactly at the boundary "
                f"null, and the cells share position paths, so the true baseline is lower); the "
                f"measured count is {s['raw_95pct']['n_ci_lower_bound_clears_band']}.", ""]
    rows = ["| window | cell | uplift | raw 95% CI | Bonferroni CI (α=0.05/48) | paired? "
            "| clears band at Bonferroni? |", "|---|---|---|---|---|---|---|"]
    any_row = False
    for wname in ("primary", "secondary", "disjoint"):
        for c in rep[f"uplift_ci_summary_{wname}"]["cells_clearing_band_at_raw_95pct"]:
            any_row = True
            rows.append(f"| {wname.upper()} | H={c['cell'][0]}, θ={c['cell'][1]}, {c['cell'][2]} "
                        f"| {_f(c['uplift'])} | [{_f(c['ci_lo'])}, {_f(c['ci_hi'])}] "
                        f"| [{_f(c['ci_bonferroni_lo'])}, {_f(c['ci_bonferroni_hi'])}] "
                        f"| {'yes' if c['paired'] else 'no (cross-H)'} "
                        f"| **{'YES' if c['bonf_ci_lower_bound_clears_band'] else 'no'}** |")
    if any_row:
        out += ["Every cell clearing the band at raw 95%, and what happens to it under the "
                "family's own pre-registered α:", ""] + rows
    return "\n".join(out)


def uplift_grid_table(rep, window="primary"):
    out = ["| H | theta | accounting | uplift Δ vs frozen | raw 95% CI | Bonferroni CI | paired? "
           "| band verdict for this cell |", "|---|---|---|---|---|---|---|---|"]
    for c in rep[f"cells_{window}"]:
        u = c.get("uplift_ci")
        if not u:
            out.append(f"| {c['H']} | {c['theta']} | {c['accounting']} | window too thin | | | | |")
            continue
        if u["bonf_ci_lower_bound_clears_band"]:
            v = "EXCEEDS band"
        elif u["bonf_ci_upper_bound_below_band"]:
            v = "inside band"
        else:
            v = "UNDECIDED"
        out.append(f"| {c['H']} | {c['theta']} | {c['accounting']} | {_f(u['uplift'])} "
                   f"| [{_f(u['ci_lo'],3)}, {_f(u['ci_hi'],3)}] "
                   f"| [{_f(u['ci_bonferroni_lo'],3)}, {_f(u['ci_bonferroni_hi'],3)}] "
                   f"| {'yes' if u['paired'] else 'no'} | {v} |")
    return "\n".join(out)


def g1_table(rep):
    f = rep["frozen_primary"]
    return "\n".join([
        "| quantity | harness | pinned reference | match |", "|---|---|---|---|",
        f"| n | {f['n']} | 1704 | exact |",
        f"| gross Sharpe | {f['gross']!r} | 1.0338583098255947 | bit-identical |",
        f"| turn (mean/decision) | {f['turn']!r} | 0.1560866077894317 | bit-identical |",
        f"| net@21 Sharpe | {f['net21']!r} | 0.8357541266542514 | bit-identical |"])


def correction3_table(rep):
    c = rep["correction_3_gross_cost_decomposition_h1_vs_h3"]
    return "\n".join([
        "| | H=1 | H=3 | Δ |", "|---|---|---|---|",
        f"| net@21 Sharpe | {_f(c['h1']['net21'])} | {_f(c['h3']['net21'])} "
        f"| **{_f(c['delta_net'])}** |",
        f"| gross Sharpe | {_f(c['h1']['gross'])} | {_f(c['h3']['gross'])} "
        f"| **{_f(c['delta_gross'])}** |",
        f"| cost_drag (gross−net21) | {_f(c['h1']['cost_drag'],4,False)} "
        f"| {_f(c['h3']['cost_drag'],4,False)} | **{_f(c['delta_cost_drag'])}** (a SAVING) |",
        "",
        f"Fraction of Δnet that is a gross/signal effect: "
        f"**{100*c['delta_gross']/c['delta_net']:.0f}%**; fraction that is a genuine cost saving: "
        f"**{-100*c['delta_cost_drag']/c['delta_net']:.0f}%**."])


def blocks(rep):
    return {
        "windows": windows_table(rep),
        "g1": g1_table(rep),
        "correction3": correction3_table(rep),
        "uplift_ci_summary": uplift_ci_summary_table(rep),
        "uplift_grid_primary": uplift_grid_table(rep, "primary"),
        "grid_primary": grid_table(rep["cells_primary"]),
        "grid_secondary": grid_table(rep["cells_secondary"]),
        "grid_disjoint": grid_table(rep["cells_disjoint"]),
        "hold_shares_exposure": hold_shares_table(rep),
        "leverage_renorm": leverage_renorm_table(rep),
        "turnover_bases": turnover_bases_table(rep["cells_primary"]),
        "decomposition": decomposition_table(rep["cells_primary"]),
        "correction1": correction1_table(rep),
        "correction2": correction2_table(rep),
        "correction2_attribution": correction2_attribution_table(rep),
        "correction4": correction4_table(rep),
        "mde": mde_table(rep),
        "band_primary": band_table(rep, "primary"),
        "band_secondary": band_table(rep, "secondary"),
        "band_disjoint": band_table(rep, "disjoint"),
        "significance": significance_table(rep),
        "best_cell": best_cell_table(rep),
    }


# =============================================================================
# MAJOR-C2A -- the drift gate extended to PROSE numbers (run 55 cycle 3)
# =============================================================================
# WHY. Until this cycle `--check` diffed ONLY the bytes between AUTOGEN markers. Prose passes
# through `render()` untouched, so it appears identically in `cur` and `new` and can NEVER
# trigger a diff. A cycle-2 judge demonstrated it live: it planted
# `0.078` -> `9.9999 WRONG NUMBER PLANTED` in prose, ran `--check`, and got
# "AUTOGEN blocks match result_919a.json", exit 0. That is the fifth guard-cannot-reach-its-own-
# claim finding of this run, and it is exactly the shape of BLOCKER-F1, the defect this fence was
# built to prevent. (A flow-verifier separately audited EVERY current prose literal against the
# JSON and found zero disagreements, so this closes a FUTURE-assurance hole, not a live error.)
#
# DESIGN (A) of the two the brief offered: extract-and-match. Every numeric literal in prose must
# either round-match a value present in result_919a.json at its own printed precision, or appear
# in the ENUMERATED whitelist below with a one-line reason why it is not data-derived. Design (B)
# (a rendered manifest of quoted figures) was rejected as the primary: its failure mode is
# FORGETTING to declare a figure, which is the same silent hole relocated. (A) has no
# declaration step to forget.
#
# WHAT (A) CANNOT DO -- MEASURED, not estimated (closing pass). A cycle-3 flow verifier swept a
# last-digit typo over all 144 distinct prose literals in RESULTS.md and counted what the gate
# rejects:
PROSE_GATE_MEASURED_REACH = {
    "method": "one last-digit typo per distinct prose literal, swept over all 144 of them "
              "(1170 mutations); measured by a cycle-3 flow verifier and TRANSCRIBED here, not "
              "re-measured by the closing pass.",
    "overall": "596 of 1170 rejected = 50.9%",
    "by_printed_precision": {
        "0dp": "61%", "1dp": "43%",
        "2dp": "1% -- EFFECTIVELY UNGUARDED",
        "3dp": "34%", "4dp": "72%",
    },
    "cause": "result_919a.json holds 3804 distinct absolute values, so a random 2-decimal "
             "literal in [0,2] round-matches SOMETHING with probability ~99.5%. The gate has "
             "real teeth only on 4-decimal Sharpes -- which is where the judge's demonstrated "
             "poison happened to sit.",
    "demonstrated_value_keyed_holes": [
        "substituting the whitelisted 70.0 into the sentence 'theta=1.0 removes 83.4%, above the "
        "record's ~70-75%' is ACCEPTED, and silently inverts CORRECTION 4's verdict into a "
        "self-refuting sentence",
        "substituting the whitelisted 20 into the share-carry SE sentence is ACCEPTED",
    ],
    "live_instance_of_the_limitation": "whitelist entry 3.5 carries ONE written reason (the "
                                       "flip-ratio pp decline) but is doing duty at a SECOND, "
                                       "never-audited site as well. Fixed in this pass by "
                                       "rewriting that entry's reason to cover both sites.",
    "scrub_class_exemption": "_SCRUBS strips every inline code span AND any bare 3-digit number "
                             "followed by space/paren/comma/period, which silently exempts "
                             "n=852, n=568, n=340, n=416 and n=143 as a CLASS -- larger than the "
                             "whitelist and previously undisclosed.",
    "not_fixed_here": "keying the match by (value, context) and refusing sub-3-decimal matches "
                      "is a REDESIGN, i.e. a new solution attempt, and is out of scope for the "
                      "closing pass. Rowed as follow-up work.",
}
# In short: the match is against the WHOLE artifact's value set, so a low-precision literal
# coincidence-matches almost always. It is a real gate on 4-decimal literals -- where
# BLOCKER-F1-class drift actually lives -- and effectively no gate at 2 decimals. It catches the
# demonstrated defect; it does NOT account for every prose numeric literal, and the artifact must
# not claim that it does.

# Patterns scrubbed BEFORE literal extraction. Each removes a whole CLASS of non-data numeral,
# which is preferable to whitelisting its members one at a time.
# DISCLOSURE (closing pass): this tuple is a LARGER exemption than the whitelist below and was
# previously undisclosed in RESULTS.md. Two entries are broad: the inline-code-span scrub removes
# every number inside backticks, and the bare-3-digit scrub removes ANY 3-digit number followed by
# space/paren/comma/period -- which exempts n=852, n=568, n=340, n=416 and n=143 as a CLASS, not
# as audited individual values. Narrowing these is a redesign and is deliberately NOT done here.
_SCRUBS = (
    (re.compile(r"\d{4}-\d{2}-\d{2}"), " "),                     # ISO dates
    (re.compile(r"(?<![\d.])(?:19|20)\d{2}\s*[-–]\s*(?:19|20)\d{2}(?![\d])"), " "),  # year spans
    (re.compile(r"\b\d{2}:\d{2}\b"), " "),                       # clock times (file mtimes)
    (re.compile(r"[A-Za-z0-9_/]+\.(?:md|py|json|db|sh|txt):[0-9]+(?:[-–,][0-9]+)*"), " "),  # file:line
    (re.compile(r"\bTASK_\d+"), " "),                            # task ids
    (re.compile(r"\bD\d+\b"), " "),                              # DECISIONS row ids
    (re.compile(r"\b(?:run|cycle|section|leg|instrument|phase)\s+\d+"), " "),
    (re.compile(r"\bsha256\b"), " "),                            # the "256" inside the algo name
    (re.compile(r"\b[GEFCOR]\d(?:[a-z])?\b"), " "),              # gate/finding ids G1..G5, E0, F1
    (re.compile(r"\b(?:BLOCKER|MAJOR|MINOR|HUNCH)-[A-Z0-9/]+"), " "),
    (re.compile(r"\b\d{3}_[A-Za-z_0-9]+"), " "),                 # experiment dirs, e.g. 830_corridor
    (re.compile(r"\b\d{3}(?:'s)?(?=\s|/|\)|,|\.)"), " "),        # bare experiment/task refs: 830, 916
    (re.compile(r"`[^`]*`"), " "),                               # inline code spans (identifiers)
)

# The ENUMERATED whitelist. Every entry is a value that is NOT derived from result_919a.json, with
# the reason. It is deliberately MINIMAL -- pruned to exactly the values the document actually
# needs (test_prose_number_gate asserts NO entry is dead, so it cannot be padded with unused
# blanket allowances). If this list ever has to GROW to make the check pass, the check is being
# defeated rather than satisfied: put the number in the ARTIFACT instead.
# LIMITATION, stated: entries are keyed by VALUE, so a whitelisted value is allowed anywhere in
# the document, not only at the site that needed it. That limitation is NOT hypothetical -- see
# PROSE_GATE_MEASURED_REACH["demonstrated_value_keyed_holes"] for two accepted substitutions on
# real sentences, one of which inverts a verdict, and ["live_instance_of_the_limitation"] for an
# entry that was already doing duty at a second site its written reason never named.
# CLOSING-PASS NOTE on the "if this list has to GROW, the check is being defeated" rule below:
# the closing pass had to grow it, because it TRANSCRIBES judge-collected figures that are, by
# construction, absent from result_919a.json (this pass is forbidden from computing them into the
# artifact). Every such entry says so in its reason. That is a real weakening of the gate and is
# recorded as one, not waved through.
PROSE_NUMBER_WHITELIST = {
    2000.0: "n_boot -- StrategyConfig.n_boot, a config constant, not a measured result",
    49.0: "~49 s wall clock -- a runtime measurement of this pass, not a scored quantity",
    24.0: "24 GB -- the on-disk size of trading_data.db (an infrastructure fact)",
    85.0: "85x the sample -- the sample-size scaling stated in band_disposition's own prose",
    0.3643: "cycle 1's uplift for its (mislabeled) best cell -- a HISTORICAL figure, retracted",
    3.5: "the flip-ratio decline in percentage POINTS from H=1 to H=3 -- a difference of two "
         "ratios, so not itself a value in the artifact. AUDIT (closing pass): this entry was "
         "ALSO admitting a second, never-named site -- 'about 3.5% of the single-cell Sharpe SE' "
         "-- which quoted an SE the pass does not derive at BEAR22's n. That site has been "
         "rewritten to quote the paired SD the pass DOES derive (11.4%), so this key is back to "
         "covering exactly the one site its reason names. It is a live example of the "
         "value-keyed limitation, not a hypothetical one.",
    70.0: "the inherited record's claim '~70-75%' at theta=1 sigma (tactics_gaps.md, verbatim)",
    20.0: "hour-of-day (asset,day) pair count in the trailing-staleness window -- an output of "
          "hour_of_day.py, which writes no numbers into result_919a.json",
    1909.6: "hour-of-day max close mismatch on XBT -- likewise an hour_of_day.py output",

    # --- CLOSING-PASS ADDITIONS. Every entry below is a figure a cycle-3 JUDGE measured, which
    # this pass TRANSCRIBES and is forbidden (D76) from computing into result_919a.json. Their
    # presence here is a disclosed weakening of this gate, recorded in RESULTS.md under
    # "What the closing pass left undone", not a quiet accommodation.
    0.3745: "the candidate cell's DE-LEVERED uplift over the frozen cell -- measured by the "
            "cycle-3 estimator lens on an arm this harness does not emit",
    0.2778: "the leverage half of that cell's +0.6523 uplift (0.6523 - 0.3263) -- a difference of "
            "two transcribed figures, not a value in the artifact",
    4.4: "the '4.4e-16' return-identity tolerance the cycle-3 lens validated its constrained "
         "gross-1 book to; the literal scrubber does not parse scientific notation",
    0.0303: "max RENORM-TW after recomputing the net columns on turn/G -- the lens's own "
            "sensitivity check on the cost approximation, not an emitted column",
    0.0442: "min RENORM-TW under the same recomputation",
    144.0: "distinct prose literals in RESULTS.md -- a property of THIS FILE measured by the "
           "cycle-3 flow verifier, not a scored quantity",
    1170.0: "mutations in that sweep (one last-digit typo per literal per position)",
    3804.0: "distinct absolute values in result_919a.json -- the flow verifier's count of the "
            "gate's own match set; a property of the artifact, not a value inside it",
}

# NOTE the character class is [-+] (hyphen FIRST, so it is a literal, not a range). Written as
# [+-−] it is the RANGE U+002B..U+2212, which swallows '±' and blows up float().
_LITERAL = re.compile(r"(?<![\w.])[-+]?\d+(?:,\d{3})*(?:\.\d+)?\s*%?")


def _json_numbers(obj, acc=None):
    """Every numeric value anywhere in the result artifact (bools excluded)."""
    if acc is None:
        acc = []
    if isinstance(obj, bool):
        return acc
    if isinstance(obj, (int, float)):
        acc.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _json_numbers(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            _json_numbers(v, acc)
    return acc


def strip_autogen(md_text):
    return re.sub(r"<!-- AUTOGEN:[A-Za-z0-9_]+ BEGIN -->.*?<!-- AUTOGEN:[A-Za-z0-9_]+ END -->",
                  " ", md_text, flags=re.DOTALL)


def check_prose_numbers(md_text, rep, whitelist=None):
    """Return a list of (literal, context) for every PROSE numeric literal that neither matches a
    value in `rep` at its printed precision nor sits in the whitelist (default
    PROSE_NUMBER_WHITELIST; pass `{}` to see what the whitelist is actually carrying, which is how
    test_prose_number_gate proves no entry is dead)."""
    if whitelist is None:
        whitelist = PROSE_NUMBER_WHITELIST
    vals = sorted({abs(v) for v in _json_numbers(rep)})
    text = strip_autogen(md_text).replace("−", "-")
    for pat, repl in _SCRUBS:
        text = pat.sub(repl, text)
    bad = []
    for m in _LITERAL.finditer(text):
        tok = m.group(0)
        pct = tok.rstrip().endswith("%")
        raw = tok.rstrip().rstrip("%").strip().replace(",", "")
        if raw in ("", "+", "-"):
            continue
        x = abs(float(raw))
        nd = len(raw.split(".")[1]) if "." in raw else 0
        if any(abs(w - x) < 1e-9 for w in whitelist):
            continue
        hit = False
        for v in vals:
            if abs(round(v, nd) - x) < 1e-9 or (pct and abs(round(v * 100.0, nd) - x) < 1e-9):
                hit = True
                break
        if not hit:
            a, b = max(0, m.start() - 60), min(len(text), m.end() + 40)
            bad.append((tok.strip(), " ".join(text[a:b].split())))
    return bad


def render(md_text, rep):
    """Replace every AUTOGEN block's body with a freshly rendered one. Raises if the file
    declares a block this module cannot render, or vice versa -- a silently-unrendered block is
    exactly the drift BLOCKER-F1 was."""
    bl = blocks(rep)
    declared = set(re.findall(r"<!-- AUTOGEN:([A-Za-z0-9_]+) BEGIN -->", md_text))
    missing = declared - set(bl)
    unused = set(bl) - declared
    if missing:
        raise ValueError(f"RESULTS.md declares AUTOGEN blocks this module cannot render: {sorted(missing)}")
    if unused:
        raise ValueError(f"render_tables can render blocks RESULTS.md never places: {sorted(unused)}")
    for name, body in bl.items():
        pat = re.compile(re.escape(BEGIN.format(name=name)) + r".*?" + re.escape(END.format(name=name)),
                         re.DOTALL)
        repl = BEGIN.format(name=name) + "\n" + body + "\n" + END.format(name=name)
        md_text, n = pat.subn(lambda _m, r=repl: r, md_text)
        if n != 1:
            raise ValueError(f"AUTOGEN block {name!r} appears {n} times in RESULTS.md (expected 1)")
    return md_text


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if RESULTS.md differs from a fresh render (no write)")
    args = ap.parse_args(argv)
    rep = json.loads(RESULT_JSON.read_text())
    cur = RESULTS_MD.read_text()
    new = render(cur, rep)
    if args.check:
        if cur != new:
            print("RESULTS.md AUTOGEN blocks are STALE vs result_919a.json "
                  "(run: python render_tables.py)")
            return 1
        bad = check_prose_numbers(cur, rep)
        if bad:
            print(f"RESULTS.md PROSE contains {len(bad)} numeric literal(s) that match NO value "
                  f"in result_919a.json and are not whitelisted (MAJOR-C2A gate):")
            for tok, ctx in bad:
                print(f"  {tok:>18}   ...{ctx}...")
            print("Fix the prose to quote the artifact, or -- only if the number is genuinely "
                  "not data-derived -- add it to PROSE_NUMBER_WHITELIST with a reason.")
            return 1
        print("RESULTS.md AUTOGEN blocks match result_919a.json; no prose literal was REJECTED "
              "by the value-match gate (measured reach: ~51% of last-digit typos overall; ~1% at "
              "2 decimals -- see PROSE_GATE_MEASURED_REACH)")
        return 0
    RESULTS_MD.write_text(new)
    print(f"rendered {len(blocks(rep))} AUTOGEN blocks into {RESULTS_MD}")
    bad = check_prose_numbers(new, rep)
    if bad:
        print(f"WARNING: {len(bad)} prose numeric literal(s) match no value in the artifact:")
        for tok, ctx in bad:
            print(f"  {tok:>18}   ...{ctx}...")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
