#!/usr/bin/env python3
"""TASK_923 SPEC-time power computation (D85). Computes n_events, event-level SD, n_clusters,
SE, and MDE per pre-registered exit horizon, PLUS the age-decay exit age* (age_decay.py) --
all from PANEL DATA / P3's own instrument, with NO book cost applied and NO book P&L
computed (D85: "this is allowed pre-registration input as long as no book P&L is computed").

Uses the GROSS (pre-cost) market-hedged residual return per event as the SPEC-time dispersion
proxy: subtracting the binding costs (a fee that is a FIXED constant per event, and a borrow
charge whose variance across events is small relative to the residual-return variance because
`hold_days` is pinned by the horizon, not free-floating) does not materially change the SD --
disclosed simplification, in the same spirit as P3's own beta=1 demeaning disclosure.

Writes `results/spec_power.json`. Must run BEFORE `SPEC.md` is frozen (its numbers are quoted
there) and BEFORE `run_book.py`'s first real-panel run (G1 SPEC-PIN).

Fixture: fixtures/trading_fixture.db, read-only. CPU, seconds.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import loader                    # noqa: E402
import book as bk                # noqa: E402
import age_decay                 # noqa: E402

RESULTS_DIR = _HERE / "results"
OUT_PATH = RESULTS_DIR / "spec_power.json"


def _json_default(o):
    if isinstance(o, (np.floating, float)) and not np.isfinite(o):
        return None
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(f"not serializable: {type(o)}")


def main():
    print(f"[923 spec_power] fixture: {loader.FIXTURE}")
    panel = loader.load_panel(loader.FIXTURE, include_dead=True)
    fixture_sha256 = hashlib.sha256(Path(loader.FIXTURE).read_bytes()).hexdigest()
    print(f"[923 spec_power] fixture sha256: {fixture_sha256}")
    mkt_inputs = loader.market_residual_inputs(panel)

    asset_cols, n_excluded = bk.listing_events(mkt_inputs["first"], mkt_inputs["last"])
    print(f"[923 spec_power] {asset_cols.size} candidate listing events "
          f"({n_excluded} excluded as left-censored panel-start rows)")

    age_star, age_profile = age_decay.derive_age_star(panel, loader.p3)
    print(f"[923 spec_power] age-decay exit age* = {age_star} days "
          f"(P3 bucket profile, 1-SE-of-zero rule): "
          + ", ".join(f"{r['age_lo']}-{r['age_hi']}d mean={r['mean_bp']:+.2f}bp "
                      f"se={r['se_bp']:.2f}bp within1se={r['within_1se_of_zero']}"
                      for r in age_profile))

    exit_labels = [(h, f"h{h}") for h in bk.EXIT_HORIZONS_FIXED]
    exit_labels.append((age_star, f"age_star({age_star}d)"))
    entry_idx = bk.entry_indices(mkt_inputs["first"], "t1")[asset_cols]

    power_rows = []
    for horizon, label in exit_labels:
        ev = bk.compute_event_returns(mkt_inputs, panel["day_ts"], asset_cols, entry_idx,
                                      horizon)
        gross_residual_bp = np.where(ev["valid"], -ev["residual"] * 1e4, np.nan)
        cohort = bk.cohort_week_ids(panel["day_ts"], entry_idx)
        stat = bk.clustered_mean_se(gross_residual_bp, cohort)
        m = bk.mde(stat["se"])
        row = {"horizon_days": int(horizon), "label": label,
              "n_events": stat["n_events"], "n_clusters": stat["n_clusters"],
              "gross_residual_mean_bp": stat["mean"], "gross_residual_sd_bp": stat["sd"],
              "se_bp": stat["se"], "mde_bp": m,
              "underpowered_vs_band": (bool(m > bk.BAND_BP) if np.isfinite(m) else True)}
        power_rows.append(row)
        print(f"[923 spec_power] {label:>16s}  n_events={row['n_events']:>4d}  "
              f"n_clusters={row['n_clusters']:>4d}  gross_SD={row['gross_residual_sd_bp']:.1f}bp "
              f"SE={row['se_bp']:.1f}bp  MDE={row['mde_bp']:.1f}bp  "
              f"band={bk.BAND_BP:.0f}bp  underpowered={row['underpowered_vs_band']}")

    out = {"fixture_sha256": fixture_sha256, "n_candidate_events": int(asset_cols.size),
          "n_excluded_left_censored": n_excluded, "age_star_days": age_star,
          "age_bucket_profile": age_profile,
          "exit_family": [{"horizon_days": h, "label": l} for h, l in exit_labels],
          "power_by_horizon": power_rows,
          "band_bp": bk.BAND_BP, "mde_z": bk.Z_MDE_80PWR_5PCT_TWOSIDED,
          "p3_provenance": {
              "artifact": "experiments/917_second_moment_and_event_instruments/p3_result.json",
              "sha256": "17dcf3cc424f81400ec9d210b788f8f5cf2b25b5fe3318efc1a247abb8e1e0c9",
              "commit": "8f411d1"}}
    RESULTS_DIR.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2, default=_json_default))
    print(f"[923 spec_power] wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
