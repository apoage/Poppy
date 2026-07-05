#!/usr/bin/env python3
"""TASK_820 self-tests — ST1-ST6, ported verbatim from runs/819_single_asset/portfolio.self_tests
(construction validated before trusting any result). Forced onto the numpy backend: these are the
determinism reference, and the planted-panel magnitudes match the 819 RESULT.md values.

  ST1 annualization identity (_sharpe == mean/std*sqrt(525600/period))
  ST2 planted trend: downtrend book shorts the crash (gross+, long-only B&H dies)
  ST3 diversification: N-asset portfolio Sharpe >> mean single-asset (noise averaging)
  ST4 inverse-vol risk parity: quiet asset overweighted, equal risk contribution
  ST5 sub-cost wiggle: random walk -> gross~0, net@21 mean<0, cost-monotone
  ST6 leak-free: wrecking close>t leaves positions<=t byte-identical
"""

import numpy as np

from . import config
from . import backtest as bt
from .strategy import (build_positions, trend_signal, rolling_vol, first_active,
                       present_mask, weights)
from .config import VOL_FLOOR


def self_tests():
    config.set_backend("numpy")   # determinism reference
    COSTS_BP = (0, 3, 8, 21)
    msgs = []

    # 1. annualization identity (inherits backtest._sharpe)
    rng = np.random.RandomState(0)
    x = rng.randn(500) * 0.01 + 0.001
    expect = np.mean(x) / np.std(x) * np.sqrt(525600 / 60.0)
    assert abs(bt._sharpe(x, 60) - expect) < 1e-9
    msgs.append("[ST1] annualization identity OK (_sharpe == mean/std*sqrt(525600/period_min))")

    # 2. planted DOWNTREND panel (the bear judge): TSMOM shorts -> portfolio gross +, long-only
    #    equal-weight B&H dies (-), portfolio beats B&H.
    rng = np.random.RandomState(3)
    A = 12
    drift = np.concatenate([np.full(120, 0.004), np.full(120, -0.004)])
    base = np.cumsum(np.tile(drift, 12))
    logc = (base[:, None] + np.cumsum(rng.randn(base.size, A) * 0.004, axis=0))
    pos, pres, w, warm = build_positions(logc, [20], "equal", window=20)
    res = bt.regime_series(logc, pos, pres, 0, logc.shape[0], H=5, warm=warm)
    g, tu, bh = res
    sh_g, sh_bh = bt._sharpe(g, 5), bt._sharpe(bh, 5)
    logp_down = np.cumsum(np.full(4000, -0.0006) + rng.randn(4000) * 0.004)
    logc_d = logp_down[:, None] + np.cumsum(rng.randn(4000, A) * 0.003, axis=0)
    posd, presd, _, warmd = build_positions(logc_d, [20], "equal", window=20)
    gd, tud, bhd = bt.regime_series(logc_d, posd, presd, 0, logc_d.shape[0], H=10, warm=warmd)
    shd_g, shd_bh = bt._sharpe(gd, 10), bt._sharpe(bhd, 10)
    assert shd_g > 0.5, f"downtrend portfolio gross Sharpe should be + (shorts), got {shd_g}"
    assert shd_bh < 0.0, f"downtrend long-only B&H should die (-), got {shd_bh}"
    assert shd_g > shd_bh, "portfolio TSMOM must beat long-only B&H in a downtrend (bear signature)"
    assert sh_g > 1.0 and sh_g > abs(sh_bh), f"square-wave portfolio should dominate B&H, g={sh_g} bh={sh_bh}"
    msgs.append(f"[ST2] planted trend: downtrend portfolio gross {shd_g:+.2f}>0 shorts the crash vs "
                f"long-only B&H {shd_bh:+.2f}<0; square-wave portfolio {sh_g:+.2f} dominates B&H {sh_bh:+.2f}")

    # 3. DIVERSIFICATION: common trend + INDEPENDENT idiosyncratic noise; portfolio Sharpe >>
    #    mean single-asset Sharpe.
    rng = np.random.RandomState(5)
    A = 24
    common = np.cumsum(np.tile(np.concatenate([np.full(60, 0.003), np.full(60, -0.003)]), 30))
    idio = np.cumsum(rng.randn(common.size, A) * 0.010, axis=0)
    logc = common[:, None] + idio
    sig = trend_signal(logc, [20])
    T = logc.shape[0]
    t = np.arange(40, T - 5, 5)
    fwd = logc[t + 5] - logc[t]
    g_asset = sig[t] * fwd
    single_sh = np.array([bt._sharpe(g_asset[:, i], 5) for i in range(A)])
    port = np.nanmean(g_asset, axis=1)
    port_sh = bt._sharpe(port, 5)
    mean_single = float(np.nanmean(single_sh))
    assert port_sh > 1.3 * mean_single, \
        f"diversification must lift Sharpe: portfolio {port_sh:.2f} vs mean-single {mean_single:.2f}"
    msgs.append(f"[ST3] DIVERSIFICATION: {A}-asset portfolio Sharpe {port_sh:+.2f} >> mean single-asset "
                f"{mean_single:+.2f} (x{port_sh/max(mean_single,1e-9):.2f}) — noise averaging confirmed")

    # 4. inverse-vol RISK PARITY: quiet asset overweighted, equal risk contribution.
    rng = np.random.RandomState(7)
    base = np.cumsum(np.full(2000, 0.0005))
    a_hi = base + np.cumsum(rng.randn(2000) * 0.020)
    a_lo = base + np.cumsum(rng.randn(2000) * 0.004)
    logc = np.column_stack([a_hi, a_lo])
    rv = rolling_vol(logc, 30)
    fa = first_active(logc)
    pres = present_mask(logc, rv, fa, 30, 30)
    w = weights(pres, rv, "invvol")
    t0 = 1500
    risk = w[t0] * rv[t0]
    assert w[t0, 1] > w[t0, 0], f"inverse-vol must overweight the quiet asset, got {w[t0]}"
    assert abs(risk[0] - risk[1]) / max(risk.mean(), 1e-12) < 0.05, \
        f"inverse-vol risk contributions must be ~equal, got {risk}"
    msgs.append(f"[ST4] risk parity: invvol weights {w[t0,0]:.3f}/{w[t0,1]:.3f} (hi/lo vol) -> equal risk "
                f"contrib {risk[0]:.4f}/{risk[1]:.4f}")

    # 5. sub-cost wiggle: random-walk panel -> gross ~0, turnover>0, net@21 mean<0, net<gross,
    #    cost-monotone.
    rng = np.random.RandomState(11)
    logc = np.cumsum(rng.randn(30000, 16) * 0.001, axis=0)
    pos, pres, w, warm = build_positions(logc, [20], "equal", window=20)
    g, tu, bh = bt.regime_series(logc, pos, pres, 0, logc.shape[0], H=10, warm=warm)
    sh_g = bt._sharpe(g, 10)
    sh_n = bt._sharpe(bt.net_series(g, tu, 21), 10)
    assert np.mean(tu) > 0, "random walk should incur turnover"
    assert sh_n < sh_g + 1e-9, "cost must not increase Sharpe"
    assert np.mean(bt.net_series(g, tu, 21)) < 0, "sub-cost wiggle net mean must be < 0"
    shs = [bt._sharpe(bt.net_series(g, tu, c), 10) for c in COSTS_BP]
    assert all(shs[i] >= shs[i + 1] - 1e-9 for i in range(len(shs) - 1)), f"net Sharpe must be monotone in cost {shs}"
    msgs.append(f"[ST5] sub-cost wiggle: random-walk portfolio gross {sh_g:+.2f}~0, net@21 {sh_n:+.2f}<gross, "
                f"net mean<0; cost-monotone {[round(s,2) for s in shs]} OK")

    # 6. LEAK-FREE: positions at decision t use only close<=t.
    rng = np.random.RandomState(13)
    logc = np.cumsum(rng.randn(2000, 8) * 0.01, axis=0)
    pos, pres, w, warm = build_positions(logc, [7, 14, 30], "invvol", window=30)
    t0 = 900
    logc2 = logc.copy()
    logc2[t0 + 1:] += np.cumsum(rng.randn(logc.shape[0] - t0 - 1, 8) * 0.05, axis=0)
    pos2, pres2, w2, _ = build_positions(logc2, [7, 14, 30], "invvol", window=30)
    assert np.array_equal(pos[:t0 + 1], pos2[:t0 + 1], equal_nan=True), "LEAK: future altered past positions"
    lab1 = logc[t0 + 1] - logc[t0]
    lab2 = logc2[t0 + 1] - logc2[t0]
    assert not np.allclose(lab1, lab2), "label must use the future (sanity)"
    msgs.append("[ST6] leak-free: wrecking close>t leaves positions<=t byte-identical; the forward label moves OK")
    return msgs
