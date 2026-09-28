#!/usr/bin/env python3
"""Smoke tests for features.py: every F1-F4 column has the right shape, F4's flatten count
matches level<=3 of a 4-dim path (4 + 16 + 64 = 84 columns), and feature_set_columns filters
correctly (F1 only trend_*, ALL returns everything)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import features as feat  # noqa: E402
import targets as tg  # noqa: E402


def _toy_panel(T=90, A=3, seed=0):
    rng = np.random.default_rng(seed)
    base = 100 * np.exp(np.cumsum(rng.normal(scale=0.02, size=(T, A)), axis=0))
    close = base
    open_ = np.vstack([close[:1], close[:-1]])
    high = np.maximum(open_, close) * 1.01
    low = np.minimum(open_, close) * 0.99
    volume = np.abs(rng.normal(loc=1000, scale=100, size=(T, A)))
    return {"open": open_, "high": high, "low": low, "close": close, "volume": volume}


def test_f1_f2_f3_shapes():
    panel = _toy_panel()
    T, A = panel["close"].shape
    f1 = feat.build_f1(panel["close"])
    assert len(f1) == len(feat.F1_LOOKBACKS)
    for v in f1.values():
        assert v.shape == (T, A)
    f2 = feat.build_f2(panel["open"], panel["high"], panel["low"], panel["close"])
    for v in f2.values():
        assert v.shape == (T, A)
    present = np.ones((T, A), dtype=bool)
    tgt = tg.build_all_targets(panel, present, h=3)
    f3 = feat.build_f3(panel["close"], tgt["beta"], tgt["mkt_ret_1d"],
                        tg.one_day_log_returns(panel["close"]), panel["volume"])
    for v in f3.values():
        assert v.shape == (T, A)


def test_f4_column_count_and_shape():
    panel = _toy_panel(T=60, A=2)
    f4 = feat.build_f4(panel["close"], panel["high"], panel["low"], panel["volume"], window=15)
    assert len(f4) == 4 + 16 + 64
    for v in f4.values():
        assert v.shape == panel["close"].shape


def test_feature_set_columns_filtering():
    names = ["f1_trend_1d", "f2_yz_vol", "f3_beta", "f4_l1_0", "f4_l2_0_1"]
    assert feat.feature_set_columns(names, "F1") == ["f1_trend_1d"]
    assert feat.feature_set_columns(names, "F4") == ["f4_l1_0", "f4_l2_0_1"]
    assert feat.feature_set_columns(names, "ALL") == names


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)}/{len(fns)} features tests passed")
