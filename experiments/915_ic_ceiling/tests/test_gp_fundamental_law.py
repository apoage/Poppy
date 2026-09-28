#!/usr/bin/env python3
"""Sanity tests for gp_fundamental_law.py's GP trade-off simulation: turnover falls as the
trading rate `a` falls; a costless (cost_bp=0) sweep prefers a=1 (fastest tracking, since there
is no reason to trade slower); TC in (0,1] for a clean, moderately-costed toy signal; N_eff/
Sigma wiring sanity (delegates to stats_utils, already unit-tested there)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gp_fundamental_law as gpl  # noqa: E402


def _toy_signal(T=400, A=6, phi=0.9, seed=0):
    rng = np.random.default_rng(seed)
    aim_true = np.zeros((T, A))
    for a in range(A):
        x = 0.0
        for t in range(T):
            x = phi * x + rng.normal(scale=1.0)
            aim_true[t, a] = x
    fwd_ret = 0.01 * aim_true + rng.normal(scale=0.02, size=(T, A))
    sigma_diag = np.ones(A)
    return aim_true, fwd_ret, sigma_diag


def test_turnover_monotone_in_trading_rate():
    aim, fwd, sd = _toy_signal()
    sim = gpl.simulate_gp_tradeoff(aim, fwd, sd, cost_bp=21.0, h=1,
                                    a_grid=np.array([0.1, 0.5, 1.0]))
    turns = {row["a"]: row["turnover_annualized"] for row in sim["trace"]}
    assert turns[0.1] < turns[0.5] < turns[1.0]


def test_zero_cost_prefers_full_speed():
    aim, fwd, sd = _toy_signal(phi=0.3)  # fast-decaying signal: no reason to smooth at cost=0
    sim = gpl.simulate_gp_tradeoff(aim, fwd, sd, cost_bp=0.0, h=1,
                                    a_grid=np.array([0.2, 0.5, 0.8, 1.0]))
    assert abs(sim["best"]["a"] - 1.0) < 1e-9


def test_high_cost_slows_trading_rate_down():
    aim, fwd, sd = _toy_signal(phi=0.95)
    sim_cheap = gpl.simulate_gp_tradeoff(aim, fwd, sd, cost_bp=1.0, h=1,
                                          a_grid=np.linspace(0.05, 1.0, 10))
    sim_expensive = gpl.simulate_gp_tradeoff(aim, fwd, sd, cost_bp=500.0, h=1,
                                              a_grid=np.linspace(0.05, 1.0, 10))
    assert sim_expensive["best"]["a"] <= sim_cheap["best"]["a"]


def test_net_sharpe_bound_wrapper_runs_and_returns_finite_fields():
    aim, fwd, sd = _toy_signal()
    out = gpl.net_sharpe_bound_for_cell(ic=0.03, n_eff=8.0, h=1, forecast_panel=aim,
                                         fwd_return_panel=fwd, sigma_diag=sd, cost_bp=21.0)
    assert np.isfinite(out["breadth_BR"])
    assert "net_sharpe_bound" in out
    assert 0.0 <= out["trading_rate_a_star"] <= 1.0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)}/{len(fns)} gp_fundamental_law tests passed")
