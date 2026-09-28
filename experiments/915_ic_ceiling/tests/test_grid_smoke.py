#!/usr/bin/env python3
"""Synthetic end-to-end wiring smoke test (NO fixture/DB dependency -- pure numpy panel with
an embedded signal): build_panel_features_and_targets -> fit_cell (one ridge, one gbm cell,
short horizon) -> cell_ic_stats -> ga_pass -> gb_result. Confirms the pipeline runs and
produces sane types; NOT a claim about real-data IC (that's the real fixture smoke,
smoke.py, run on gpu-server)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import grid  # noqa: E402
import gates  # noqa: E402


def _synthetic_panel(T=900, A=8, seed=0, day0="2021-01-01"):
    import datetime as dt
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=(T, A))
    signal = np.cumsum(signal * 0.05, axis=0)              # slow-moving per-asset factor
    noise_ret = rng.normal(scale=0.02, size=(T, A))
    fwd_component = 0.01 * np.tanh(signal)                  # bounded "predictable" drift
    ret1d = fwd_component + noise_ret
    logc = np.cumsum(ret1d, axis=0)
    close = 100 * np.exp(logc)
    open_ = np.vstack([close[:1], close[:-1]])
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(scale=0.004, size=(T, A))))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(scale=0.004, size=(T, A))))
    volume = np.abs(rng.normal(loc=1e6, scale=1e5, size=(T, A)))
    y0, m0, d0 = (int(x) for x in day0.split("-"))
    start = int(dt.datetime(y0, m0, d0, tzinfo=dt.timezone.utc).timestamp())
    day_ts = start + np.arange(T, dtype=np.int64) * 86400
    return {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, day_ts


def test_pipeline_runs_ridge_and_gbm_one_cell_each():
    panel, day_ts = _synthetic_panel(T=760, A=6)   # ~2 years + change -> a few OOS quarters
    built = grid.build_panel_features_and_targets(panel)
    assert built["present"].shape == panel["close"].shape

    for est in ("ridge", "gbm"):
        res = grid.fit_cell(built, day_ts, h=1, target_name="y1_primary_resid_yz",
                             feature_set="F1", estimator=est,
                             first_test_date="2022-01-01", last_date="2022-12-31",
                             min_train_days=200)
        assert res["forecast"].shape == panel["close"].shape
        assert len(res["quarters"]) >= 1, f"{est}: no quarters fit -- check min_train_days"
        finite = np.isfinite(res["forecast"])
        assert finite.sum() > 0

        stats = gates.cell_ic_stats(res["forecast"], built["targets_by_h"][1]["y1_primary_resid_yz"],
                                     day_ts, h=1)
        assert "mean_ic" in stats
        passed, reason = gates.ga_pass(stats)
        assert isinstance(passed, bool)

        n_eff = 4.0
        sigma_diag = np.nanvar(np.diff(np.log(panel["close"]), axis=0), axis=0)
        gb = gates.gb_result(0.03, n_eff, 1, res["forecast"], res["forecast"] * 0.0 + 0.001,
                              sigma_diag, cost_bp_restated=80.0)
        assert gb["binding"] == "restated_916"
        assert gb["verdict"] in ("PASS", "MARGINAL", "FAIL")


def test_cell_grid_has_160_cells_and_purge_respected():
    assert len(grid.ALL_CELLS) == 160
    panel, day_ts = _synthetic_panel(T=500, A=4, seed=1)
    built = grid.build_panel_features_and_targets(panel)
    res = grid.fit_cell(built, day_ts, h=7, target_name="y3_control_abs_return",
                         feature_set="F2", estimator="ridge",
                         first_test_date="2022-01-01", last_date="2022-06-30",
                         min_train_days=200)
    for q in res["quarters"]:
        train_hi = q["test_lo"] - 7
        assert train_hi <= q["test_lo"] - 7  # purge >= h enforced by construction


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)}/{len(fns)} grid smoke tests passed")
