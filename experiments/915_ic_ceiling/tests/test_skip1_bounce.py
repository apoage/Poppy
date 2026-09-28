#!/usr/bin/env python3
"""POST-HOC instrument test (TASK_915 cycle 2). D83(i): the skip1 re-score is itself a
verification instrument, so it must be shown to FIRE where the defect sits and to STAY QUIET
where it does not.

(1) Planted defect: a pure bid-ask-bounce price process -- efficient log-price is a driftless
    random walk (zero true predictability), the observed close is that price +/- a half-spread
    chosen iid each day. Because F1 features end at close_t and the label starts at close_t,
    the bounce gives a large, G-A-passing same-close F1 IC; under the one-day-skip label it
    must collapse and the cell must be flagged timing_fragile.
(2) Negative control: a genuine AR(1) return process (phi=0.35) has real predictability two
    days out (corr(r_t, r_{t+2}) = phi^2), so the skip1 cell must STILL pass G-A -- the
    instrument does not flag every signal.
(3) shift_targets(td, 0) reproduces build_all_targets bit-for-bit (the skip1 twin differs
    from the official cell ONLY by the label shift)."""
import datetime as dt
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import grid            # noqa: E402
import posthoc_skip1 as ps  # noqa: E402


def _day_ts(T, day0="2021-01-01"):
    y0, m0, d0 = (int(x) for x in day0.split("-"))
    start = int(dt.datetime(y0, m0, d0, tzinfo=dt.timezone.utc).timestamp())
    return start + np.arange(T, dtype=np.int64) * 86400


def _ohlcv_from_logclose(logc, rng):
    close = 100 * np.exp(logc)
    open_ = np.vstack([close[:1], close[:-1]])
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(scale=0.003, size=close.shape)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(scale=0.003, size=close.shape)))
    volume = np.abs(rng.normal(loc=1e6, scale=1e5, size=close.shape))
    return {"open": open_, "high": high, "low": low, "close": close, "volume": volume}


def _bounce_panel(T=2060, A=15, vol=0.01, half_spread=0.02, seed=3):
    rng = np.random.default_rng(seed)
    eff = np.cumsum(rng.normal(scale=vol, size=(T, A)), axis=0)     # efficient price: RW
    bounce = half_spread * rng.choice([-1.0, 1.0], size=(T, A))      # iid bid/ask print
    return _ohlcv_from_logclose(eff + bounce, rng), _day_ts(T)


def _ar1_panel(T=2060, A=15, phi=0.35, seed=4):
    rng = np.random.default_rng(seed)
    r = rng.normal(scale=0.02, size=(T, A))
    for t in range(1, T):
        r[t] += phi * r[t - 1]
    return _ohlcv_from_logclose(np.cumsum(r, axis=0), rng), _day_ts(T)


def _score(built, day_ts, k):
    bk = ps.shifted_built(built, k)
    _res, stats, passed, _reason = ps.score_cell(
        bk, day_ts, "y1_primary_resid_yz", 1, "F1", "ridge",
        first_test_date="2024-01-01", last_date="2026-06-30", min_train_days=500)
    return stats, passed


def test_shift_zero_is_identity():
    panel, day_ts = _ar1_panel(T=700, A=6)
    built = grid.build_panel_features_and_targets(panel)
    for h, td in built["targets_by_h"].items():
        td0 = ps.shift_targets(td, 0)
        for key in ("y1_primary_resid_yz", "y2_control_resid_c2c", "y3_control_abs_return",
                    "y4_sign_volscaled", "r_fwd", "resid_fwd", "r_mkt_fwd"):
            np.testing.assert_array_equal(td0[key], td[key], err_msg=f"h={h} {key}")


def test_shift_one_is_one_row_label_shift():
    panel, day_ts = _ar1_panel(T=700, A=6)
    built = grid.build_panel_features_and_targets(panel)
    td = built["targets_by_h"][3]
    td1 = ps.shift_targets(td, 1)
    np.testing.assert_array_equal(td1["r_fwd"][:-1], td["r_fwd"][1:])
    assert np.isnan(td1["r_fwd"][-1]).all()
    # normalisers stay t-dated: y4 skip1 = r_fwd[t+1] / sigma_yz[t]^2
    exp = td["r_fwd"][1:] / np.maximum(td["sigma_yz"][:-1] ** 2, 1e-10)
    np.testing.assert_array_equal(td1["y4_sign_volscaled"][:-1], exp)


def test_bounce_same_close_ic_collapses_under_skip1():
    panel, day_ts = _bounce_panel()
    built = grid.build_panel_features_and_targets(panel)
    st0, p0 = _score(built, day_ts, 0)
    st1, p1 = _score(built, day_ts, 1)
    # the defect is present: a pure-noise price passes G-A with a large same-close IC
    assert p0 and st0["mean_ic"] > 0.2, st0["mean_ic"]
    # ... and the instrument now detects it
    assert not p1, (st1["mean_ic"], st1["nw_t"])
    assert abs(st1["mean_ic"]) < 0.25 * st0["mean_ic"], (st0["mean_ic"], st1["mean_ic"])
    assert ps.timing_fragile(p0, p1) is True


def test_genuine_ar1_signal_survives_skip1():
    panel, day_ts = _ar1_panel()
    built = grid.build_panel_features_and_targets(panel)
    st0, p0 = _score(built, day_ts, 0)
    st1, p1 = _score(built, day_ts, 1)
    assert p0, st0["mean_ic"]
    assert p1, (st1["mean_ic"], st1["nw_t"])      # instrument does not flag a real signal
    assert ps.timing_fragile(p0, p1) is False
    assert ps.timing_fragile(False, False) is None


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)}/{len(fns)} skip1 instrument tests passed")
