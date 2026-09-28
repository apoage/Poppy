#!/usr/bin/env python3
"""TASK_923 cycle-2 fix -- POST-HOC corrected G2 (see SPEC.md 'POST-HOC ERRATA').

Applies the task file's own G2 wording ("the placebo ... must produce a distribution the real
book's CI excludes") AFTER the data were seen, on NET (binding costs) and GROSS (fee=0,
borrow=0) both. No new book logic: the real cells come from `run_book.score_cell` and the
placebo draws from `battery.run_placebo` with the SAME seed (923) and count (200) run_book.py
used; the gross placebo is the same call with zero costs (same RNG sequence -> same placebo
dates). The as-implemented criterion (`battery.placebo_clean`) is re-computed alongside and
its net summary is checked against `results/book_result.json` (reproduction guard).

Does NOT check SPEC.md's sha256 pin (SPEC.md now carries an appended errata section; its
pre-registered prefix is verified instead -- see PREREG_PREFIX_*). Read-only fixture; CPU.
Writes `results/g2_corrected.json`.
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
import run_book as rb      # noqa: E402

# SPEC.md as pre-registered = its first PREREG_PREFIX_BYTES bytes (the errata is appended
# after them, never interleaved).
PREREG_PREFIX_BYTES = 12138
PREREG_PREFIX_SHA256 = rb.EXPECTED_SPEC_SHA256
OUT_PATH = rb.RESULTS_DIR / "g2_corrected.json"


def main():
    spec_bytes = rb.SPEC_PATH.read_bytes()
    prefix_sha = hashlib.sha256(spec_bytes[:PREREG_PREFIX_BYTES]).hexdigest()
    if prefix_sha != PREREG_PREFIX_SHA256:
        raise RuntimeError(f"g2_corrected: SPEC.md pre-registered prefix sha {prefix_sha} != "
                           f"{PREREG_PREFIX_SHA256} -- pre-registered text was altered")
    spec_power = json.loads(rb.SPEC_POWER_PATH.read_text())
    prior = json.loads(rb.OUT_PATH.read_text())
    age_star = int(spec_power["age_star_days"])

    panel = loader.load_panel(loader.FIXTURE, include_dead=True)
    fixture_sha256 = hashlib.sha256(Path(loader.FIXTURE).read_bytes()).hexdigest()
    if fixture_sha256 != spec_power["fixture_sha256"]:
        raise RuntimeError("g2_corrected: fixture sha256 differs from spec_power's")
    mkt_inputs = loader.market_residual_inputs(panel)
    day_ts = panel["day_ts"]
    asset_cols, _ = bk.listing_events(mkt_inputs["first"], mkt_inputs["last"])
    entry_t1 = bk.entry_indices(mkt_inputs["first"], "t1")[asset_cols]

    exit_labels = [(h, f"h{h}") for h in bk.EXIT_HORIZONS_FIXED]
    exit_labels.append((age_star, f"age_star({age_star}d)"))
    legs = {"net": (bk.FEE_BASE_TAKER_BP, bk.BORROW_ANNUAL_ALT_HIGH),
            "gross": (0.0, 0.0)}

    rows = []
    for horizon, label in exit_labels:
        row = {"label": label, "horizon_days": int(horizon)}
        for leg, (fee, borrow) in legs.items():
            cell = rb.score_cell(mkt_inputs, day_ts, asset_cols, entry_t1, horizon,
                                 hedge=True, fee_bp=fee, borrow_annual=borrow)
            plc = battery.run_placebo(mkt_inputs, day_ts, horizon, rb.N_PLACEBO_DRAWS,
                                      rb.PLACEBO_SEED, fee_bp=fee, borrow_annual=borrow,
                                      hedge=True)
            real_mean = cell["stat"]["mean"]
            corrected = battery.placebo_separation_task_g2(
                real_mean, cell["ci_bonferroni_lo"], cell["ci_bonferroni_hi"], plc["draws"])
            as_impl = battery.placebo_clean(real_mean, plc)
            row[leg] = {"real_mean": real_mean, "real_se": cell["stat"]["se"],
                        "real_ci_bonferroni_lo": cell["ci_bonferroni_lo"],
                        "real_ci_bonferroni_hi": cell["ci_bonferroni_hi"],
                        "corrected_task_g2": corrected,
                        "as_implemented_deviates_from_task_g2": {
                            "placebo_max_abs": plc["max_abs"], **as_impl},
                        "placebo_draws": plc["draws"]}
            if leg == "net":
                ref = prior["honesty_battery"]["placebo_by_horizon"][label]["summary"]
                if not (np.isclose(ref["mean"], plc["mean"], rtol=0, atol=1e-6)
                        and np.isclose(ref["max_abs"], plc["max_abs"], rtol=0, atol=1e-6)):
                    raise RuntimeError(f"g2_corrected: net placebo for {label} does not "
                                       f"reproduce book_result.json ({plc['mean']} vs "
                                       f"{ref['mean']})")
            c = corrected
            print(f"[923 g2c] {label:>14s} {leg:5s} real={real_mean:+8.1f} "
                  f"CI=[{cell['ci_bonferroni_lo']:+8.1f},{cell['ci_bonferroni_hi']:+8.1f}] "
                  f"placebo mean={c['placebo_mean']:+8.1f} "
                  f"q2.5/97.5=[{c['placebo_q_lo']:+8.1f},{c['placebo_q_hi']:+8.1f}] "
                  f"excludes={c['excludes']} dir={c['direction']} "
                  f"p1={c['k_ge']}/{c['n_placebo_finite']} "
                  f"| as-impl clean={as_impl['clean']}")
        rows.append(row)

    out = {"note": "POST-HOC corrected G2 (task wording), applied after data were seen; "
                   "see SPEC.md POST-HOC ERRATA", "fixture_sha256": fixture_sha256,
           "spec_prereg_prefix_sha256": prefix_sha,
           "n_placebo_draws": rb.N_PLACEBO_DRAWS, "placebo_seed": rb.PLACEBO_SEED,
           "by_horizon": rows}
    OUT_PATH.write_text(json.dumps(out, indent=2, default=rb._json_default))
    print(f"[923 g2c] wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
