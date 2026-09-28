#!/usr/bin/env python3
"""Positive control (adversarial-pass discipline, .claude/rules/30-agent-conduct.md: a guard
must be proven to fire, not just proven to stay quiet). Builds a synthetic panel spanning
2022-2026 with a STRONG, trivially-learnable momentum signal (forward return driven directly
by the trailing 1-day return) and confirms:
  (1) the G-A pipeline can and does PASS on an F1/ridge cell when a real signal is present
      (rules out "ga_pass is fail-closed/always False by construction"), and
  (2) the SAME pipeline on an INDEPENDENT reshuffled-target panel (no real relationship) does
      NOT pass -- rules out a leaking/always-True guard in the other direction."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import grid      # noqa: E402
import gates     # noqa: E402


def _panel_with_strong_momentum(T=2060, A=15, day0="2021-01-01", strength=0.35, seed=0):
    import datetime as dt
    rng = np.random.default_rng(seed)
    ret1d = rng.normal(scale=0.02, size=(T, A))
    # forward return at h=1 is STRONGLY driven by yesterday's return (autocorrelated momentum)
    for t in range(1, T):
        ret1d[t] += strength * ret1d[t - 1]
    logc = np.cumsum(ret1d, axis=0)
    close = 100 * np.exp(logc)
    open_ = np.vstack([close[:1], close[:-1]])
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(scale=0.003, size=(T, A))))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(scale=0.003, size=(T, A))))
    volume = np.abs(rng.normal(loc=1e6, scale=1e5, size=(T, A)))
    y0, m0, d0 = (int(x) for x in day0.split("-"))
    start = int(dt.datetime(y0, m0, d0, tzinfo=dt.timezone.utc).timestamp())
    day_ts = start + np.arange(T, dtype=np.int64) * 86400
    return {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, day_ts


def test_positive_control_ga_can_pass():
    panel, day_ts = _panel_with_strong_momentum()
    built = grid.build_panel_features_and_targets(panel)
    res = grid.fit_cell(built, day_ts, h=1, target_name="y4_sign_volscaled", feature_set="F1",
                         estimator="ridge", first_test_date="2024-01-01", last_date="2026-06-30",
                         min_train_days=500)
    stats = gates.cell_ic_stats(res["forecast"], built["targets_by_h"][1]["y4_sign_volscaled"],
                                 day_ts, h=1)
    passed, reason = gates.ga_pass(stats)
    assert passed, (stats, reason)  # the guard MUST fire on a real, strong, planted signal


def test_negative_control_ga_does_not_pass_on_shuffled_target():
    panel, day_ts = _panel_with_strong_momentum(seed=1)
    built = grid.build_panel_features_and_targets(panel)
    rng = np.random.default_rng(99)
    # break the forecast<->target relationship: independently shuffle the target panel's rows
    y = built["targets_by_h"][1]["y4_sign_volscaled"].copy()
    perm = rng.permutation(y.shape[0])
    built["targets_by_h"][1] = dict(built["targets_by_h"][1])
    built["targets_by_h"][1]["y4_sign_volscaled"] = y[perm]
    res = grid.fit_cell(built, day_ts, h=1, target_name="y4_sign_volscaled", feature_set="F1",
                         estimator="ridge", first_test_date="2024-01-01", last_date="2026-06-30",
                         min_train_days=500)
    stats = gates.cell_ic_stats(res["forecast"], built["targets_by_h"][1]["y4_sign_volscaled"],
                                 day_ts, h=1)
    passed, reason = gates.ga_pass(stats)
    assert not passed, (stats, reason)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)}/{len(fns)} positive/negative control tests passed")
