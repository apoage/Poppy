#!/usr/bin/env python3
"""TASK_923 — the real-panel book run. Refuses to run unless `SPEC.md` on disk hashes to
`EXPECTED_SPEC_SHA256` (G1 SPEC-PIN: the exit family, cost constants, estimand, band, and MDE
computation are frozen in SPEC.md BEFORE this script's first real-panel run).

Binding reading: entry='t1' (918-lesson execution timing), hedge=True (short + beta=1
market hedge), fee=80bp round-trip taker (per leg) + borrow at the ALT-HIGH scenario
(916-verified). Every other combination below is a reported, NON-BINDING sensitivity,
labelled as such. One book-simulation code path (`book.py`) serves all of them plus the
honesty battery (`battery.py`).

Writes `results/book_result.json`. Read-only against the fixture throughout (CPU, seconds).
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import loader              # noqa: E402
import book as bk          # noqa: E402
import battery             # noqa: E402
import age_decay           # noqa: E402

SPEC_PATH = _HERE / "SPEC.md"
RESULTS_DIR = _HERE / "results"
SPEC_POWER_PATH = RESULTS_DIR / "spec_power.json"
OUT_PATH = RESULTS_DIR / "book_result.json"

# Frozen after SPEC.md was written from spec_power.json's numbers (G1 SPEC-PIN). Filled in
# by the implementer once SPEC.md's content is final; this script refuses to run for real
# against a SPEC.md that does not hash to this value.
EXPECTED_SPEC_SHA256 = "8c16e13a60528f6a5f01557adbd5a2a20d9efc32657afc6f81b89b55c7af3063"

N_PLACEBO_DRAWS = 200
PLACEBO_SEED = 923
CAPACITY_WINDOW_DAYS = 5
CAPACITY_ASSUMED_NOTIONAL_USD = 10000.0
CAPACITY_PARTICIPATION_RATE = 0.01


def _json_default(o):
    if isinstance(o, (np.floating, float)) and not np.isfinite(o):
        return None
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serializable: {type(o)}")


def score_cell(mkt_inputs, day_ts, asset_cols, entry_idx, horizon, hedge, fee_bp,
              borrow_annual):
    ev = bk.compute_event_returns(mkt_inputs, day_ts, asset_cols, entry_idx, horizon)
    costed = bk.net_markout_bp(ev, hedge=hedge, fee_bp=fee_bp, borrow_annual=borrow_annual)
    cohort = bk.cohort_week_ids(day_ts, entry_idx)
    stat = bk.clustered_mean_se(costed["net_bp"], cohort)
    ci_lo, ci_hi = bk.ci_bonferroni(stat)
    m = bk.mde(stat["se"])
    return {"ev": ev, "costed": costed, "cohort": cohort, "stat": stat,
           "ci_bonferroni_lo": ci_lo, "ci_bonferroni_hi": ci_hi, "mde_bp": m}


def horizon_verdict(stat, ci_lo, ci_hi, underpowered, band=bk.BAND_BP):
    if underpowered:
        return "UNDERPOWERED"
    if stat["degenerate"] or not np.isfinite(ci_lo):
        return "DEGENERATE"
    if ci_lo > band:
        return "TRADABLE"
    if ci_hi < 0:
        return "COST_NEGATIVE"
    return "NULL"


def main():
    if not SPEC_PATH.exists():
        raise RuntimeError("923 run_book: SPEC.md not found -- write and freeze it first (G1)")
    spec_sha = hashlib.sha256(SPEC_PATH.read_bytes()).hexdigest()
    if EXPECTED_SPEC_SHA256 == "REPLACE_AFTER_FREEZING_SPEC_MD":
        raise RuntimeError(
            "923 run_book: EXPECTED_SPEC_SHA256 is still the placeholder -- freeze SPEC.md, "
            "compute its sha256, and hardcode it here BEFORE the first real-panel run (G1).")
    if spec_sha != EXPECTED_SPEC_SHA256:
        raise RuntimeError(
            f"923 run_book: SPEC.md sha256 mismatch -- on disk {spec_sha}, expected "
            f"{EXPECTED_SPEC_SHA256}. Refusing to run against an unpinned spec (G1).")
    print(f"[923 run_book] SPEC.md sha256 verified: {spec_sha}")

    if not SPEC_POWER_PATH.exists():
        raise RuntimeError("923 run_book: results/spec_power.json missing -- run "
                          "spec_power.py first (its numbers are what SPEC.md quotes).")
    spec_power = json.loads(SPEC_POWER_PATH.read_text())
    age_star = int(spec_power["age_star_days"])

    print(f"[923 run_book] fixture: {loader.FIXTURE}")
    panel = loader.load_panel(loader.FIXTURE, include_dead=True)
    fixture_sha256 = hashlib.sha256(Path(loader.FIXTURE).read_bytes()).hexdigest()
    print(f"[923 run_book] fixture sha256: {fixture_sha256}")
    if fixture_sha256 != spec_power["fixture_sha256"]:
        raise RuntimeError("923 run_book: fixture sha256 differs from the one spec_power.py "
                          "ran against -- refusing (fixture must never change mid-row).")
    mkt_inputs = loader.market_residual_inputs(panel)
    day_ts = panel["day_ts"]

    asset_cols, n_excluded = bk.listing_events(mkt_inputs["first"], mkt_inputs["last"])
    print(f"[923 run_book] {asset_cols.size} candidate listing events "
          f"({n_excluded} excluded, left-censored panel-start)")
    if asset_cols.size == 0:
        raise RuntimeError("923 run_book: ZERO listing events -- degenerate universe, "
                          "refusing to fabricate a book (zero-event guard).")

    exit_labels = [(h, f"h{h}") for h in bk.EXIT_HORIZONS_FIXED]
    exit_labels.append((age_star, f"age_star({age_star}d)"))

    entry_t1 = bk.entry_indices(mkt_inputs["first"], "t1")[asset_cols]
    entry_same = bk.entry_indices(mkt_inputs["first"], "same_close")[asset_cols]

    binding = []
    non_binding_same_close = []
    non_binding_legacy21 = []
    non_binding_borrow_low = []
    non_binding_unhedged = []
    placebo_by_horizon = {}
    loo_by_horizon = {}

    for horizon, label in exit_labels:
        power_row = next(r for r in spec_power["power_by_horizon"] if r["label"] == label)
        underpowered = bool(power_row["underpowered_vs_band"])

        cell = score_cell(mkt_inputs, day_ts, asset_cols, entry_t1, horizon, hedge=True,
                          fee_bp=bk.FEE_BASE_TAKER_BP, borrow_annual=bk.BORROW_ANNUAL_ALT_HIGH)
        verdict = horizon_verdict(cell["stat"], cell["ci_bonferroni_lo"],
                                  cell["ci_bonferroni_hi"], underpowered)
        row = {"horizon_days": int(horizon), "label": label, "entry": "t1", "hedge": True,
              "fee_bp_per_leg": bk.FEE_BASE_TAKER_BP, "borrow_scenario": "alt_high",
              **cell["stat"], "ci_bonferroni_lo": cell["ci_bonferroni_lo"],
              "ci_bonferroni_hi": cell["ci_bonferroni_hi"], "mde_observed_bp": cell["mde_bp"],
              "underpowered_pre_registered": underpowered, "verdict": verdict}
        binding.append(row)
        print(f"[923 run_book] BINDING {label:>16s}: n={row['n_events']:>4d} "
              f"clusters={row['n_clusters']:>4d} mean={row['mean']:+.1f}bp "
              f"CI=[{row['ci_bonferroni_lo']:+.1f},{row['ci_bonferroni_hi']:+.1f}] "
              f"band=+{bk.BAND_BP:.0f}bp verdict={verdict}")

        # ---- honesty battery: placebo + LOO, at BINDING params, per horizon -------------
        placebo = battery.run_placebo(mkt_inputs, day_ts, horizon, N_PLACEBO_DRAWS,
                                      PLACEBO_SEED, fee_bp=bk.FEE_BASE_TAKER_BP,
                                      borrow_annual=bk.BORROW_ANNUAL_ALT_HIGH, hedge=True)
        pc = battery.placebo_clean(row["mean"], placebo)
        placebo_by_horizon[label] = {"summary": {k: v for k, v in placebo.items()
                                                 if k != "draws"}, "check": pc}
        print(f"[923 run_book]   G2 placebo {label}: mean={placebo['mean']:+.1f}bp "
              f"max_abs={placebo['max_abs']:.1f}bp p={pc['p_empirical']:.4f} "
              f"clean={pc['clean']}")

        loo = battery.leave_one_out(cell["costed"]["net_bp"], cell["cohort"])
        loo_by_horizon[label] = loo
        print(f"[923 run_book]   LOO {label}: n={loo['n']} full_mean={loo['full_mean']:+.1f}bp "
              f"max_abs_delta={loo['max_abs_delta']:.2f}bp"
              if not loo["degenerate"] else f"[923 run_book]   LOO {label}: DEGENERATE (n<3)")

        # ---- non-binding sensitivities --------------------------------------------------
        cell_sc = score_cell(mkt_inputs, day_ts, asset_cols, entry_same, horizon, hedge=True,
                             fee_bp=bk.FEE_BASE_TAKER_BP, borrow_annual=bk.BORROW_ANNUAL_ALT_HIGH)
        non_binding_same_close.append({"horizon_days": int(horizon), "label": label,
                                       "entry": "same_close", **cell_sc["stat"],
                                       "ci_bonferroni_lo": cell_sc["ci_bonferroni_lo"],
                                       "ci_bonferroni_hi": cell_sc["ci_bonferroni_hi"]})

        cell_l21 = score_cell(mkt_inputs, day_ts, asset_cols, entry_t1, horizon, hedge=True,
                              fee_bp=bk.FEE_LEGACY_BP, borrow_annual=bk.BORROW_ANNUAL_ALT_HIGH)
        non_binding_legacy21.append({"horizon_days": int(horizon), "label": label,
                                     "fee_bp_per_leg": bk.FEE_LEGACY_BP, **cell_l21["stat"],
                                     "ci_bonferroni_lo": cell_l21["ci_bonferroni_lo"],
                                     "ci_bonferroni_hi": cell_l21["ci_bonferroni_hi"]})

        cell_blow = score_cell(mkt_inputs, day_ts, asset_cols, entry_t1, horizon, hedge=True,
                               fee_bp=bk.FEE_BASE_TAKER_BP,
                               borrow_annual=bk.BORROW_ANNUAL_ALT_LOW)
        non_binding_borrow_low.append({"horizon_days": int(horizon), "label": label,
                                       "borrow_scenario": "alt_low", **cell_blow["stat"],
                                       "ci_bonferroni_lo": cell_blow["ci_bonferroni_lo"],
                                       "ci_bonferroni_hi": cell_blow["ci_bonferroni_hi"]})

        cell_unh = score_cell(mkt_inputs, day_ts, asset_cols, entry_t1, horizon, hedge=False,
                              fee_bp=bk.FEE_BASE_TAKER_BP,
                              borrow_annual=bk.BORROW_ANNUAL_ALT_HIGH)
        non_binding_unhedged.append({"horizon_days": int(horizon), "label": label,
                                     "hedge": False, **cell_unh["stat"],
                                     "ci_bonferroni_lo": cell_unh["ci_bonferroni_lo"],
                                     "ci_bonferroni_hi": cell_unh["ci_bonferroni_hi"]})

    # ---- capacity + by-year, at the binding entry (t1), horizon = shortest exit (5d window) --
    ev5 = bk.compute_event_returns(mkt_inputs, day_ts, asset_cols, entry_t1,
                                   bk.EXIT_HORIZONS_FIXED[0])
    capacity = battery.capacity_disclosure(panel, day_ts, asset_cols, entry_t1, ev5["valid"],
                                           window_days=CAPACITY_WINDOW_DAYS,
                                           assumed_notional_usd=CAPACITY_ASSUMED_NOTIONAL_USD,
                                           participation_rate=CAPACITY_PARTICIPATION_RATE)
    print(f"[923 run_book] capacity: n_events={capacity['n_events']} "
          f"assumed_notional=${CAPACITY_ASSUMED_NOTIONAL_USD:,.0f} "
          f"ceiling_usd={capacity.get('ceiling_usd')}")

    by_year_h21 = battery.by_cohort_year(
        day_ts, entry_t1,
        score_cell(mkt_inputs, day_ts, asset_cols, entry_t1, 21, hedge=True,
                  fee_bp=bk.FEE_BASE_TAKER_BP,
                  borrow_annual=bk.BORROW_ANNUAL_ALT_HIGH)["costed"]["net_bp"])
    print(f"[923 run_book] by-cohort-year (h21, binding): {by_year_h21}")

    # ---- family verdict (G2/G3/G4 folded in) --------------------------------------------
    # G2 is reported over EVERY horizon, not just the (here: empty) "powered" subset --
    # a vacuous all() over an empty filter would silently hide a real placebo failure.
    g2_clean_by_horizon = {lbl: placebo_by_horizon[lbl]["check"]["clean"]
                          for (_, lbl) in exit_labels}
    g2_overall_clean_all_horizons = all(g2_clean_by_horizon.values())
    powered_labels = [lbl for (_, lbl) in exit_labels
                     if not next(r for r in binding if r["label"] == lbl)
                     ["underpowered_pre_registered"]]
    g2_overall_clean_powered_only = (all(g2_clean_by_horizon[lbl] for lbl in powered_labels)
                                     if powered_labels else None)  # None = vacuous, no powered
                                                                    # horizon exists to check
    any_tradable = any(r["verdict"] == "TRADABLE" for r in binding)
    any_underpowered = any(r["verdict"] == "UNDERPOWERED" for r in binding)
    all_underpowered = all(r["verdict"] == "UNDERPOWERED" for r in binding)
    if all_underpowered:
        family_verdict = "UNDERPOWERED"
    elif g2_overall_clean_powered_only is False:
        family_verdict = "NULL (G2 placebo did not cleanly separate at >=1 powered horizon)"
    elif any_tradable:
        family_verdict = "TRADABLE (conditional on shortability at listing, unverified)"
    elif any_underpowered and not any(r["verdict"] == "COST_NEGATIVE" for r in binding):
        family_verdict = "UNDERPOWERED"
    elif all(r["verdict"] == "COST_NEGATIVE" for r in binding if not r["underpowered_pre_registered"]):
        family_verdict = "COST_NEGATIVE"
    else:
        family_verdict = "NULL"

    print(f"\n[923 run_book] G2 clean by horizon (ALL horizons, not just powered ones): "
          f"{g2_clean_by_horizon}")
    print(f"[923 run_book] G2 overall clean (all horizons)={g2_overall_clean_all_horizons}  "
          f"(powered-only, {len(powered_labels)} horizons)="
          f"{g2_overall_clean_powered_only}")
    print(f"\n[923 run_book] FAMILY VERDICT: {family_verdict}")

    out = {"spec_sha256": spec_sha, "fixture_sha256": fixture_sha256,
          "n_candidate_events": int(asset_cols.size), "n_excluded_left_censored": n_excluded,
          "age_star_days": age_star,
          "binding_table": binding,
          "sensitivities": {
              "same_close_entry_non_binding": non_binding_same_close,
              "legacy21_fee_non_binding": non_binding_legacy21,
              "borrow_low_non_binding": non_binding_borrow_low,
              "unhedged_short_non_binding": non_binding_unhedged,
          },
          "honesty_battery": {
              "placebo_by_horizon": placebo_by_horizon,
              "leave_one_out_by_horizon": loo_by_horizon,
              "capacity_disclosure": capacity,
              "by_cohort_year_h21_binding": by_year_h21,
              "g2_clean_by_horizon": g2_clean_by_horizon,
              "g2_overall_clean_all_horizons": g2_overall_clean_all_horizons,
              "g2_overall_clean_powered_only": g2_overall_clean_powered_only,
          },
          "family_verdict": family_verdict,
          "band_bp": bk.BAND_BP,
          "cost_stack": {"fee_base_taker_bp_per_leg": bk.FEE_BASE_TAKER_BP,
                        "fee_legacy_bp_per_leg_NON_BINDING": bk.FEE_LEGACY_BP,
                        "borrow_annual_alt_high": bk.BORROW_ANNUAL_ALT_HIGH,
                        "borrow_annual_alt_low_NON_BINDING": bk.BORROW_ANNUAL_ALT_LOW}}
    RESULTS_DIR.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2, default=_json_default))
    print(f"[923 run_book] wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
