"""TASK_839 — behavioral unit tests for the ON path of budget-constrained vector sizing.

OFF-path bit-identity vs canon is already covered by the pinned parity gate
(tools/tests/test_vector_step_parity.py) and the intended-divergence test
(tools/tests/test_vector_env.py::TestEquivalence). This file exercises the NEW
``vector_budget_step=True`` behavior directly:

  (a) on a FRESH / all-cash book (PV == cash, the step-0 env-wall regime) a uniform
      all-market vector no longer overdraws / bankrupts — aggregate opening spend is
      PV*budget_pct <= available cash (contrast: OFF overdraws → margin cascade; the
      all-short case genuinely BANKRUPTS under OFF and survives under ON — LEDGER
      828s3/env-wall). NB this <=-cash bound is scoped to PV~=cash; on an invested book
      (PV>cash) ON can still overdraw but is never worse than OFF — locked by (f);
  (b) the n_trades divisor is correct — N simultaneous opening buys each size at
      ~budget_pct/N of PV;
  (c) CLOSING trades are excluded from n_trades — a pure-close step is byte-identical
      ON vs OFF;
  (d) LIMIT orders keep the original ideal_trade_value sizing — they are unaffected by
      the budget flag even when the market legs' budget binds;
  (e) when n_trades <= 1/position_size_pct the min() hits the position_size_pct cap, so
      the ON sizing is a no-op equal to OFF (bit-identical book);
  (f) HONEST PV>cash limitation: on an already-invested book (PV > cash) a breadth of opens
      can STILL overdraw under ON (the <=-cash bound is NOT general) — and when each leg
      saturates available_cash the post-margin book is byte-identical to OFF (ON degrades to
      canon margin behavior, never worse).

CPU-only, never GPU, never trading_data.db. Run:
    JAX_PLATFORMS=cpu pytest tools/tests/test_budget_step.py -v
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")  # never GPU for env-primitive tests

import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

# Repo root on sys.path (PEP-420 namespace pkg) — computed from this file, not hardcoded.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mcts_trader.env.jax_trading_env import (
    EnvConfig, reset, step_vector, get_current_prices, compute_portfolio_value,
)

INITIAL_CASH = 100_000.0


def flat_prices(n_assets, n_steps=200, base=100.0):
    """Constant-in-time, distinct-per-asset prices — value-conserving rebalance regime."""
    per_asset = base + 5.0 * np.arange(n_assets)
    return jnp.asarray(np.tile(per_asset[None, :], (n_steps, 1)), dtype=jnp.float32)


def cfg(n_assets, budget=False, budget_pct=0.99, fee_bps=0.0, slippage_bps=0.0,
        position_size_pct=0.1, allow_short=True, tp_sl=False):
    return EnvConfig(
        n_assets=n_assets, initial_cash=INITIAL_CASH,
        fee_bps=fee_bps, slippage_bps=slippage_bps,
        max_steps=120, window_size=10, position_size_pct=position_size_pct,
        allow_short=allow_short, tp_sl_enabled=tp_sl,
        vector_budget_step=budget, vector_budget_pct=budget_pct,
    )


def with_longs(state, prices, long_idxs, entry=100.0, size=20.0, cash=INITIAL_CASH):
    """Inject pre-existing LONG positions on the given asset indices."""
    na = int(state.pos_active.shape[0])
    pa = np.zeros(na, dtype=bool)
    ps = np.zeros(na, dtype=np.int32)
    pz = np.zeros(na, dtype=np.float32)
    pe = np.zeros(na, dtype=np.float32)
    for i in long_idxs:
        pa[i], ps[i], pz[i], pe[i] = True, 1, size, entry
    s = state.replace(
        cash=jnp.asarray(cash, dtype=jnp.float32),
        pos_active=jnp.asarray(pa), pos_side=jnp.asarray(ps),
        pos_size=jnp.asarray(pz), pos_entry=jnp.asarray(pe),
    )
    cur = get_current_prices(prices, s.current_step, s.episode_start_idx)
    pv = compute_portfolio_value(s, cur)
    return s.replace(prev_portfolio_value=pv, initial_portfolio_value=pv)


def material(state):
    return (
        float(state.cash),
        np.asarray(state.pos_active), np.asarray(state.pos_side),
        np.asarray(state.pos_size), np.asarray(state.pos_entry),
        np.asarray(state.order_active), np.asarray(state.order_side),
        np.asarray(state.order_size), np.asarray(state.order_price),
    )


# =============================================================================
# (a) aggregate spend within budget — no overdraft / no bankruptcy under ON
# =============================================================================

def test_on_all_buy_spend_within_budget():
    """ON, FRESH all-cash book (PV == cash == initial_cash): 32 simultaneous opening buys size
    at PV*0.99/32 each, so Sigma spend = 0.99*PV, and because PV==cash that IS <= available
    cash — all 32 longs OPEN with NO margin liquidation, cash ~= 0.01*PV. (The <=-cash bound
    is specific to this PV~=cash regime; test_on_pv_gt_cash_* covers the invested book.)"""
    na = 32
    c = cfg(na, budget=True)
    prices = flat_prices(na)
    s = reset(jax.random.PRNGKey(0), prices, c)
    out, _, _, done = step_vector(s, jnp.full((na,), 2, dtype=jnp.int32), prices, c)
    assert int(np.asarray(out.pos_active).sum()) == na, "all opening buys should survive (no overdraft)"
    assert bool(np.all(np.asarray(out.pos_side) == 1))
    assert float(out.cash) >= 0.0
    # Sigma spend = 0.99*PV -> residual cash ~= 0.01*PV (fee=0 here).
    assert np.isclose(float(out.cash), INITIAL_CASH * (1.0 - c.vector_budget_pct), rtol=0, atol=5.0), float(out.cash)
    assert not bool(done)


def test_off_all_buy_overdraws_and_liquidates():
    """OFF (canon): 32 buys each at min(0.1*PV, 0.99*cash) => ~3.2x cash demand -> overdraft
    -> margin call liquidates longs. Fewer than 32 longs survive. This is the behavior the
    fix changes (default OFF stays canon)."""
    na = 32
    c = cfg(na, budget=False)
    prices = flat_prices(na)
    s = reset(jax.random.PRNGKey(0), prices, c)
    out, _, _, _ = step_vector(s, jnp.full((na,), 2, dtype=jnp.int32), prices, c)
    assert int(np.asarray(out.pos_active).sum()) < na, "OFF must overdraw -> margin liquidation"


def test_on_all_short_survives_off_bankrupts():
    """The env-wall bankruptcy, exactly: 32 simultaneous OPENING SHORTS with no longs to
    liquidate. OFF -> cash goes ~-2.2x -> margin call finds no longs -> BANKRUPTCY (pv->0).
    ON -> Sigma short-reserve = 0.99*PV <= cash -> all 32 shorts open, no bankruptcy."""
    na = 32
    prices = flat_prices(na)
    # OFF: bankrupt
    c_off = cfg(na, budget=False)
    s = reset(jax.random.PRNGKey(1), prices, c_off)
    out_off, _, _, done_off = step_vector(s, jnp.zeros((na,), dtype=jnp.int32), prices, c_off)
    cur = get_current_prices(prices, s.current_step, s.episode_start_idx)
    pv_off = float(compute_portfolio_value(out_off, cur))
    assert bool(done_off)
    assert pv_off < INITIAL_CASH * c_off.liquidation_threshold_pct, pv_off
    assert int(np.asarray(out_off.pos_active).sum()) == 0
    # ON: survives
    c_on = cfg(na, budget=True)
    s2 = reset(jax.random.PRNGKey(1), prices, c_on)
    out_on, _, _, done_on = step_vector(s2, jnp.zeros((na,), dtype=jnp.int32), prices, c_on)
    assert not bool(done_on)
    assert int(np.asarray(out_on.pos_active).sum()) == na
    assert bool(np.all(np.asarray(out_on.pos_side) == 2))
    assert float(out_on.cash) >= 0.0


# =============================================================================
# (b) n_trades divisor: N opening buys each size at ~budget_pct/N of PV
# =============================================================================

def test_n_trades_divisor_equal_split():
    """10 simultaneous opening buys (rest HOLD): pct_eff = min(0.1, 0.99/10) = 0.099, so each
    long's notional ~= PV*0.099 and residual cash ~= 0.01*PV."""
    na = 32
    n_buy = 10
    c = cfg(na, budget=True)
    prices = flat_prices(na)
    s = reset(jax.random.PRNGKey(2), prices, c)
    av = np.ones(na, dtype=np.int32)          # HOLD
    av[:n_buy] = 2                            # 10 opening buys
    cur = get_current_prices(prices, s.current_step, s.episode_start_idx)
    out, _, _, _ = step_vector(s, jnp.asarray(av), prices, c)
    pct_eff = min(c.position_size_pct, c.vector_budget_pct / n_buy)   # 0.099
    expected_notional = INITIAL_CASH * pct_eff
    cur_np = np.asarray(cur)
    pos_size = np.asarray(out.pos_size)
    for i in range(n_buy):
        notional = pos_size[i] * cur_np[i]
        assert np.isclose(notional, expected_notional, rtol=1e-4, atol=1.0), (i, notional, expected_notional)
    assert int(np.asarray(out.pos_active).sum()) == n_buy
    assert np.isclose(float(out.cash), INITIAL_CASH - n_buy * expected_notional, rtol=0, atol=5.0)


# =============================================================================
# (c) closing trades excluded from n_trades: pure-close step is a no-op vs OFF
# =============================================================================

def test_closing_only_step_is_flag_invariant():
    """A step that ONLY closes positions (sell all pre-existing longs) has ZERO opening spend
    legs -> budget sizing is irrelevant -> byte-identical ON vs OFF."""
    na = 16
    prices = flat_prices(na)
    long_idxs = list(range(10))
    av = jnp.zeros((na,), dtype=jnp.int32)     # SELL everything -> closes the longs
    c_off = cfg(na, budget=False)
    c_on = cfg(na, budget=True)
    s_off = with_longs(reset(jax.random.PRNGKey(3), prices, c_off), prices, long_idxs)
    s_on = with_longs(reset(jax.random.PRNGKey(3), prices, c_on), prices, long_idxs)
    o_off, _, r_off, _ = step_vector(s_off, av, prices, c_off)
    o_on, _, r_on, _ = step_vector(s_on, av, prices, c_on)
    m_off, m_on = material(o_off), material(o_on)
    assert np.isclose(m_off[0], m_on[0], rtol=0, atol=1e-3), (m_off[0], m_on[0])
    for k in range(1, len(m_off)):
        assert np.allclose(m_off[k], m_on[k], rtol=1e-6, atol=1e-3), k
    assert np.isclose(float(r_off), float(r_on), rtol=1e-6, atol=1e-5)


# =============================================================================
# (d) limit orders unaffected by the budget flag (keep ideal_trade_value sizing)
# =============================================================================

def test_limit_orders_keep_ideal_trade_value():
    """Mixed step: 20 market opening buys (budget binds, pct_eff=min(0.1,0.99/20)=0.0495) plus
    12 LIMIT buys. The limit orders must still size on ideal_trade_value = PV*0.1 (order_size =
    PV*0.1/limit_price), NOT the reduced market spend size — and be identical ON vs OFF."""
    na = 32
    n_mkt = 20
    prices = flat_prices(na)
    av = np.full(na, 4, dtype=np.int32)        # limit-buy default
    av[:n_mkt] = 2                             # first 20 are market opening buys
    av_j = jnp.asarray(av)
    c_on = cfg(na, budget=True)
    c_off = cfg(na, budget=False)
    s_on = reset(jax.random.PRNGKey(4), prices, c_on)
    s_off = reset(jax.random.PRNGKey(4), prices, c_off)
    cur = np.asarray(get_current_prices(prices, s_on.current_step, s_on.episode_start_idx))
    o_on, _, _, _ = step_vector(s_on, av_j, prices, c_on)
    o_off, _, _, _ = step_vector(s_off, av_j, prices, c_off)
    lb_price = cur * (1.0 - c_on.limit_offset_pct)
    expected_order_size = (INITIAL_CASH * c_on.position_size_pct) / lb_price   # ideal_trade_value based
    on_size = np.asarray(o_on.order_size)
    for i in range(n_mkt, na):
        assert bool(o_on.order_active[i]) and int(o_on.order_side[i]) == 1
        assert np.isclose(on_size[i], expected_order_size[i], rtol=1e-4, atol=1e-3), (i, on_size[i], expected_order_size[i])
    # limit order arrays identical ON vs OFF (flag never touches the limit legs)
    assert np.allclose(np.asarray(o_on.order_size), np.asarray(o_off.order_size), rtol=1e-6, atol=1e-3)
    assert np.allclose(np.asarray(o_on.order_price), np.asarray(o_off.order_price), rtol=1e-6, atol=1e-3)
    assert np.array_equal(np.asarray(o_on.order_active), np.asarray(o_off.order_active))


# =============================================================================
# (e) budget cap non-binding (few trades) => ON is a no-op equal to OFF
# =============================================================================

def test_few_trades_is_noop_vs_off():
    """5 opening buys, pct=0.1: n_trades=5 <= 1/pct=10, so pct_eff = min(0.1, 0.99/5=0.198) =
    0.1 (the cap) => ideal_spend_value == ideal_trade_value => bit-identical book ON vs OFF."""
    na = 32
    n_buy = 5
    prices = flat_prices(na)
    av = np.ones(na, dtype=np.int32)
    av[:n_buy] = 2
    av_j = jnp.asarray(av)
    c_on = cfg(na, budget=True)
    c_off = cfg(na, budget=False)
    o_on, _, r_on, _ = step_vector(reset(jax.random.PRNGKey(5), prices, c_on), av_j, prices, c_on)
    o_off, _, r_off, _ = step_vector(reset(jax.random.PRNGKey(5), prices, c_off), av_j, prices, c_off)
    m_on, m_off = material(o_on), material(o_off)
    assert np.isclose(m_on[0], m_off[0], rtol=0, atol=1e-4), (m_on[0], m_off[0])
    for k in range(1, len(m_on)):
        assert np.allclose(m_on[k], m_off[k], rtol=1e-7, atol=1e-4), k
    assert np.isclose(float(r_on), float(r_off), rtol=1e-7, atol=1e-6)


# =============================================================================
# (f) HONEST PV>cash limitation: ON can still overdraw on an invested book, and when each
#     leg saturates available_cash it degrades to the EXACT canon (OFF) margin resolution
# =============================================================================

def test_on_pv_gt_cash_overdraws_and_degrades_to_off():
    """The load-bearing honest-limitation lock. On an INVESTED book (PV = 3x cash: a
    pre-existing long worth 2x cash), two simultaneous opening buys with pct=0.9 / budget=0.99:

      * ON per-leg ideal = PV*min(0.9, 0.99/2=0.495) = PV*0.495 = 1.485x cash > available_cash,
        so BOTH ON and OFF clamp each leg to available_cash (0.99x cash) -> Sigma opening spend
        = 1.98x cash > cash => OVERDRAFT (the <=-cash bound is NOT general when PV>cash), which
        process_margin_call resolves by liquidating the pre-existing long.
      * Because both arms clamp identically, the post-margin book is BYTE-IDENTICAL ON vs OFF
        => budget-ON degrades to canon margin behavior on an invested book (never worse).
    """
    na = 4
    prices = flat_prices(na)
    c_on = cfg(na, budget=True, budget_pct=0.99, position_size_pct=0.9)
    c_off = cfg(na, budget=False, position_size_pct=0.9)
    # pre-existing long on asset 0 worth 2x cash (entry == current price 100, size 2000).
    s_on = with_longs(reset(jax.random.PRNGKey(6), prices, c_on), prices, [0],
                      entry=100.0, size=2000.0, cash=INITIAL_CASH)
    s_off = with_longs(reset(jax.random.PRNGKey(6), prices, c_off), prices, [0],
                       entry=100.0, size=2000.0, cash=INITIAL_CASH)
    cur = get_current_prices(prices, s_on.current_step, s_on.episode_start_idx)
    pv_before = float(compute_portfolio_value(s_on, cur))
    assert np.isclose(pv_before, 3.0 * INITIAL_CASH, rtol=1e-4), pv_before   # PV > cash regime
    av = jnp.array([1, 2, 2, 1], dtype=jnp.int32)     # hold injected long, open two new longs
    o_on, _, _, _ = step_vector(s_on, av, prices, c_on)
    o_off, _, _, _ = step_vector(s_off, av, prices, c_off)
    # (a) overdraft occurred under ON: aggregate opening spend (2 * 0.99*cash = 1.98x cash) > cash,
    #     resolved by margin -> the pre-existing long (asset 0) is liquidated.
    assert not bool(o_on.pos_active[0]), "PV>cash: ON still overdraws -> injected long liquidated"
    assert bool(o_on.pos_active[1]) and bool(o_on.pos_active[2]), "the two new opens survive"
    # (b) byte-identical to the OFF (canon) resolution — ON degrades to canon, never worse.
    m_on, m_off = material(o_on), material(o_off)
    assert np.isclose(m_on[0], m_off[0], rtol=0, atol=1e-3), (m_on[0], m_off[0])   # cash
    for k in range(1, len(m_on)):
        assert np.allclose(m_on[k], m_off[k], rtol=1e-6, atol=1e-3), k
    # per-leg ON spend <= OFF spend (here equal — both clamped to available_cash)
    assert np.allclose(np.asarray(o_on.pos_size), np.asarray(o_off.pos_size), rtol=1e-6, atol=1e-3)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
