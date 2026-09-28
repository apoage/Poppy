#!/usr/bin/env python3
"""Unit tests for targets.py: forward-return arithmetic, leak-freeness (mutating data after t
must not move any target/intermediate AT or BEFORE t), and market-return construction."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import targets as tg  # noqa: E402


def _toy_ohlcv(T=140, A=4, seed=0):
    rng = np.random.default_rng(seed)
    base = 100 * np.exp(np.cumsum(rng.normal(scale=0.02, size=(T, A)), axis=0))
    close = base
    open_ = np.vstack([close[:1], close[:-1]]) * (1 + rng.normal(scale=0.002, size=(T, A)))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(scale=0.003, size=(T, A))))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(scale=0.003, size=(T, A))))
    return {"open": open_, "high": high, "low": low, "close": close}


def test_forward_log_return_matches_definition():
    panel = _toy_ohlcv()
    close = panel["close"]
    h = 3
    r = tg.forward_log_return(close, h)
    for t in (0, 10, 50):
        want = np.log(close[t + h]) - np.log(close[t])
        assert np.allclose(r[t], want)
    assert np.all(np.isnan(r[-h:]))


def test_leak_free_targets_unaffected_by_future_mutation():
    panel = _toy_ohlcv(T=160)
    present = np.ones_like(panel["close"], dtype=bool)
    h = 3
    out1 = tg.build_all_targets(panel, present, h)
    panel2 = {k: v.copy() for k, v in panel.items()}
    cut = 140
    for k in panel2:
        panel2[k][cut:] = panel2[k][cut:] * 5.0 + 17.0
    out2 = tg.build_all_targets(panel2, present, h)
    # any row t whose own target only needs data <= t + h must be unaffected for t+h < cut
    boundary = cut - h - 1
    for name in tg.TARGET_NAMES:
        a, b = out1[name][:boundary], out2[name][:boundary]
        assert np.allclose(a, b, equal_nan=True), name


def test_market_return_equal_weight_matches_manual_mean():
    panel = _toy_ohlcv(T=50, A=3)
    present = np.ones_like(panel["close"], dtype=bool)
    mkt = tg.equal_weight_market_return(panel["close"], present)
    lc = np.log(panel["close"])
    r1 = np.vstack([np.full((1, 3), np.nan), lc[1:] - lc[:-1]])
    want = np.nanmean(r1, axis=1)
    assert np.allclose(mkt[1:], want[1:])
    assert np.isnan(mkt[0])


def test_beta_recovers_known_loading():
    """Construct assets whose return is EXACTLY beta_i * market + idio-noise with idio-noise
    set to (near) zero for one asset -> rolling_beta should recover beta_i closely."""
    rng = np.random.default_rng(3)
    T, A = 200, 2
    mkt = rng.normal(scale=0.02, size=T)
    mkt[0] = np.nan
    true_beta = np.array([1.5, -0.7])
    r = mkt[:, None] * true_beta[None, :]
    r[0, :] = np.nan
    beta_est = tg.rolling_beta(r, mkt, window=90)
    t = 150
    assert np.allclose(beta_est[t], true_beta, atol=1e-8)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)}/{len(fns)} targets tests passed")
