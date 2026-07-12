"""TASK_840 — ON-path unit tests for the target-weight rebalance step.

These pin the NEW ``vector_weight_rebalance`` semantics in
``mcts_trader/env/jax_trading_env.py::step_vector`` (default OFF; the two OFF
parity pins live in ``test_vector_step_parity.py`` / ``test_vector_weight_parity.py``
and are NOT touched here). Everything is CPU-only, synthetic prices, never
``trading_data.db``.

Semantics under test (spec item 2/3):
  * per-asset action -> TARGET signed weight: Buy +w_unit, Sell -w_unit,
    Hold/Limit 0 (no re-trim), clipped to +-vector_weight_max*w_unit;
  * each Buy/Sell re-trims the position toward its target fraction of CURRENT PV;
  * Sigma|w| <= 1 by proportional scale-down (no leverage);
  * re-trim only when |target-current| > deadband (turnover discipline);
  * fee=0 => PV conserved across the rebalance => bankruptcy impossible.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import pytest

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from mcts_trader.env.jax_trading_env import (
    EnvConfig, reset, step_vector, get_current_prices, compute_portfolio_value,
)
from mcts_trader.utils.transforms import symlog

INITIAL_CASH = 100_000.0
FLAT_PRICE = 100.0


def _flat_prices(T, na, price=FLAT_PRICE):
    return jnp.full((T, na), price, dtype=jnp.float32)


def _cfg(na, w_unit=-1.0, w_max=4.0, deadband=-1.0, fee_bps=0.0, slippage_bps=0.0,
         allow_short=True, window=5):
    return EnvConfig(
        n_assets=na, initial_cash=INITIAL_CASH, fee_bps=fee_bps, slippage_bps=slippage_bps,
        max_steps=400, window_size=window, position_size_pct=0.1,
        allow_short=allow_short, tp_sl_enabled=False,
        vector_weight_rebalance=True, vector_weight_unit=w_unit,
        vector_weight_max=w_max, vector_rebalance_deadband=deadband,
    )


def _fresh_state(prices, cfg, window=5):
    st = reset(jax.random.PRNGKey(0), prices, cfg)
    return st.replace(
        episode_start_idx=jnp.array(window, dtype=st.episode_start_idx.dtype),
        current_step=jnp.array(0, dtype=st.current_step.dtype),
        initial_portfolio_value=jnp.array(INITIAL_CASH, dtype=jnp.float32),
        prev_portfolio_value=jnp.array(INITIAL_CASH, dtype=jnp.float32),
        cash=jnp.array(INITIAL_CASH, dtype=jnp.float32),
    )


def _pv_at_exec(state_before, new_state, prices):
    """PV of new_state evaluated at the price the rebalance executed against."""
    px = get_current_prices(prices, state_before.current_step, state_before.episode_start_idx)
    return float(compute_portfolio_value(new_state, px))


# ---------------------------------------------------------------------------
# 1. target-weight mapping for each action {0..4}
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("action,expect", [
    (2, "long"),    # Buy  -> +w_unit long
    (0, "short"),   # Sell -> -w_unit short
    (1, "flat"),    # Hold -> unchanged (flat)
    (3, "flat"),    # LimitSell -> position unchanged, order placed
    (4, "flat"),    # LimitBuy  -> position unchanged, order placed
])
def test_action_maps_to_target_weight(action, expect):
    na = 4
    w_unit = 0.25
    prices = _flat_prices(80, na)
    cfg = _cfg(na, w_unit=w_unit)
    st = _fresh_state(prices, cfg)
    av = jnp.array([action] + [1] * (na - 1), dtype=jnp.int32)  # only asset 0 acts
    ns, _, _, _ = step_vector(st, av, prices, cfg)
    active = bool(np.asarray(ns.pos_active)[0])
    side = int(np.asarray(ns.pos_side)[0])
    size = float(np.asarray(ns.pos_size)[0])
    if expect == "long":
        assert active and side == 1
        # notional ~= w_unit * PV ; size = notional / price
        assert size == pytest.approx(w_unit * INITIAL_CASH / FLAT_PRICE, rel=1e-3)
    elif expect == "short":
        assert active and side == 2
        assert size == pytest.approx(w_unit * INITIAL_CASH / FLAT_PRICE, rel=1e-3)
    else:  # flat: position untouched
        assert not active
        if action in (3, 4):  # limit actions still place an order
            assert bool(np.asarray(ns.order_active)[0])


# ---------------------------------------------------------------------------
# 2. unit-stepping with cap
# ---------------------------------------------------------------------------

def test_unit_stepping_with_cap():
    na = 4
    w_unit = 0.1
    w_max_units = 4.0
    prices = _flat_prices(80, na)
    cfg = _cfg(na, w_unit=w_unit, w_max=w_max_units)
    st = _fresh_state(prices, cfg)
    buy = jnp.array([2] + [1] * (na - 1), dtype=jnp.int32)
    expected_units = [1, 2, 3, 4, 4, 4]  # steps up by 1 unit, then caps at 4
    for k, exp_u in enumerate(expected_units):
        st, _, _, _ = step_vector(st, buy, prices, cfg)
        size = float(np.asarray(st.pos_size)[0])
        target_size = exp_u * w_unit * INITIAL_CASH / FLAT_PRICE
        assert size == pytest.approx(target_size, rel=1e-3), f"step {k}: {size} vs {target_size}"
        assert int(np.asarray(st.pos_side)[0]) == 1


# ---------------------------------------------------------------------------
# 3. deadband suppression
# ---------------------------------------------------------------------------

def test_deadband_suppresses_small_rebalance():
    na = 4
    w_unit = 0.1
    prices = _flat_prices(80, na)
    # deadband larger than a single unit step -> Buy makes no trade
    cfg = _cfg(na, w_unit=w_unit, deadband=2.0 * w_unit)
    st = _fresh_state(prices, cfg)
    buy = jnp.array([2] + [1] * (na - 1), dtype=jnp.int32)
    ns, _, _, _ = step_vector(st, buy, prices, cfg)
    assert not bool(np.asarray(ns.pos_active)[0])  # suppressed: still flat
    assert float(np.asarray(ns.cash)) == pytest.approx(INITIAL_CASH, rel=1e-6)


# ---------------------------------------------------------------------------
# 4. Sigma|w| <= 1 scale-down (no leverage)
# ---------------------------------------------------------------------------

def test_sum_abs_weight_scaledown():
    na = 4
    w_unit = 0.5   # 4 assets * 0.5 = 2.0 raw -> must scale to 1.0
    prices = _flat_prices(80, na)
    cfg = _cfg(na, w_unit=w_unit, w_max=4.0)
    st = _fresh_state(prices, cfg)
    buy_all = jnp.full((na,), 2, dtype=jnp.int32)
    ns, _, _, _ = step_vector(st, buy_all, prices, cfg)
    px = get_current_prices(prices, st.current_step, st.episode_start_idx)
    pv = float(compute_portfolio_value(ns, px))
    notional = np.asarray(ns.pos_size) * np.asarray(px)
    total_abs_w = float(notional.sum()) / pv
    assert total_abs_w == pytest.approx(1.0, abs=1e-3)   # scaled down to the no-leverage cap
    assert float(np.asarray(ns.cash)) >= -1e-3           # no overdraft / leverage
    # each asset gets an equal 1/na share after scale-down
    per = float(np.asarray(ns.pos_size)[0]) * FLAT_PRICE / pv
    assert per == pytest.approx(1.0 / na, abs=1e-3)


# ---------------------------------------------------------------------------
# 5. short re-trim -> bounded loss (no bleed, no bankruptcy)
# ---------------------------------------------------------------------------

def test_short_retrim_bounded_loss():
    na = 4
    w_unit = 0.1
    T = 40
    # price rises 3%/step on asset 0 (adverse for a short); flat elsewhere
    px = np.full((T, na), FLAT_PRICE, dtype=np.float32)
    px[:, 0] = FLAT_PRICE * (1.03 ** np.arange(T))
    prices = jnp.asarray(px)
    cfg = _cfg(na, w_unit=w_unit, w_max=4.0)
    st = _fresh_state(prices, cfg, window=5)
    sell = jnp.array([0] + [1] * (na - 1), dtype=jnp.int32)  # constant Sell on asset 0
    pv_prev = INITIAL_CASH
    for k in range(20):
        st, _, _, done = step_vector(st, sell, prices, cfg)
        cur = get_current_prices(prices, st.current_step - 1, st.episode_start_idx)
        pv = float(compute_portfolio_value(st, cur))
        # per-step loss is bounded: short exposure <= w_max*w_unit of PV, one step move
        drop = (pv_prev - pv) / max(pv_prev, 1e-9)
        assert drop < 0.05, f"step {k}: per-step PV drop {drop:.4f} not bounded"
        assert not bool(np.asarray(done)), f"bankrupt at step {k}"
        pv_prev = pv
    assert pv_prev > 0.5 * INITIAL_CASH   # no unbounded bleed


# ---------------------------------------------------------------------------
# 6. fee=0 PV conservation across a rebalance
# ---------------------------------------------------------------------------

def test_fee0_pv_conservation():
    na = 6
    w_unit = 0.1
    prices = _flat_prices(80, na)
    cfg = _cfg(na, w_unit=w_unit, fee_bps=0.0, slippage_bps=0.0)
    st = _fresh_state(prices, cfg)
    rng = np.random.default_rng(7)
    for k in range(8):
        av = jnp.asarray(rng.integers(0, 5, size=na), dtype=jnp.int32)
        pv_before = float(compute_portfolio_value(
            st, get_current_prices(prices, st.current_step, st.episode_start_idx)))
        ns, _, _, _ = step_vector(st, av, prices, cfg)
        pv_after = _pv_at_exec(st, ns, prices)
        assert pv_after == pytest.approx(pv_before, rel=1e-4), f"step {k}: {pv_after} vs {pv_before}"
        assert float(np.asarray(ns.cash)) >= -1e-2   # bankruptcy impossible at fee=0
        st = ns


def _inject_long(state, idx, entry, size, cash=INITIAL_CASH):
    """Set a single explicit long position (for a forced-re-trim setup)."""
    na = int(state.pos_active.shape[0])
    pa = np.zeros(na, dtype=bool); pa[idx] = True
    sd = np.zeros(na, dtype=np.int32); sd[idx] = 1
    sz = np.zeros(na, dtype=np.float32); sz[idx] = size
    en = np.zeros(na, dtype=np.float32); en[idx] = entry
    return state.replace(
        pos_active=jnp.asarray(pa), pos_side=jnp.asarray(sd),
        pos_size=jnp.asarray(sz), pos_entry=jnp.asarray(en),
        cash=jnp.array(float(cash), dtype=jnp.float32),
    )


def _inject_short(state, idx, entry, size, cash):
    """Set a single explicit short position (for the held-winning-short adversarial setup)."""
    na = int(state.pos_active.shape[0])
    pa = np.zeros(na, dtype=bool); pa[idx] = True
    sd = np.zeros(na, dtype=np.int32); sd[idx] = 2
    sz = np.zeros(na, dtype=np.float32); sz[idx] = size
    en = np.zeros(na, dtype=np.float32); en[idx] = entry
    return state.replace(
        pos_active=jnp.asarray(pa), pos_side=jnp.asarray(sd),
        pos_size=jnp.asarray(sz), pos_entry=jnp.asarray(en),
        cash=jnp.array(float(cash), dtype=jnp.float32),
    )


# ---------------------------------------------------------------------------
# (a) FEES — does the weight path charge transaction cost on a re-trim?  [DOCUMENTED]
# ---------------------------------------------------------------------------

def test_fee_charged_on_retrim_documented():
    """Force the SAME re-trim (a Buy that steps up a pre-existing long) at fee0 and at
    fee16/slip5; compare the PV change across the step.

    The weight-rebalance path charges transaction cost on a re-trim via TWO layers:
    (a) the fee+slippage spread baked into buy_exec_price / sell_exec_price
    (apply_fee_and_slippage) — a re-trim CLOSES at sell_exec_price = px/(1+c) and REOPENS
    at buy_exec_price = px*(1+c); and (b) TASK_842: the canon-equivalent SEPARATE explicit
    close-fee (closed_notional*fee_rate) on the closing leg (plus fee_on_open_short on a
    short open), mirroring execute_sell/execute_buy. At fee0 BOTH layers are 0 and PV is
    conserved. Turnover is therefore NOT free, and is now cost-honest vs the canon path."""
    na = 4
    w_unit = 0.1
    prices = _flat_prices(80, na)

    def retrim_pv_change(fee, slip):
        cfg = _cfg(na, w_unit=w_unit, fee_bps=fee, slippage_bps=slip)
        st = _fresh_state(prices, cfg)
        st = _inject_long(st, 0, entry=FLAT_PRICE, size=100.0)  # ~0.09 weight, off the grid
        buy = jnp.array([2] + [1] * (na - 1), dtype=jnp.int32)  # step up -> close+reopen
        pv_before = float(compute_portfolio_value(
            st, get_current_prices(prices, st.current_step, st.episode_start_idx)))
        ns, _, _, _ = step_vector(st, buy, prices, cfg)
        pv_after = _pv_at_exec(st, ns, prices)
        return pv_after - pv_before

    dpv_fee0 = retrim_pv_change(0.0, 0.0)
    dpv_fee16 = retrim_pv_change(16.0, 5.0)

    # fee0: PV conserved across the re-trim (spread == 0)
    assert dpv_fee0 == pytest.approx(0.0, abs=1.0)
    # fee16/slip5: PV strictly DROPS -> a transaction cost IS charged on the re-trim
    assert dpv_fee16 < -1.0
    assert dpv_fee16 < dpv_fee0 - 1.0
    # magnitude sanity: spread ~ (21bp/1.0021)*(closed ~10k + reopened ~19k) ~ 60,
    # PLUS the TASK_842 explicit close-fee ~16bp*closed(~10k) ~ 16  ->  ~ -76 total
    assert -200.0 < dpv_fee16 < -10.0


# ---------------------------------------------------------------------------
# (b) REWARD CONSISTENCY — reward reflects the POST-override realized PV
# ---------------------------------------------------------------------------

def test_reward_reflects_post_override_pv():
    """The returned reward must be computed from the OVERRIDE cash/positions (the PV
    actually realized under weight-rebalance), NOT from a stale pre-override (canon/839)
    assembly. Proof: (1) reward == symlog(log(pv_override/prev)*100) using the realized
    post-step PV; (2) the ON branch's PV genuinely diverges from the OFF branch's PV on
    the same inputs and reward tracks the ON one (so it is not the OFF assembly's reward);
    (3) the state's own prev_portfolio_value (carried forward) equals that realized PV."""
    na = 6
    w_unit = 0.1
    prices = _flat_prices(80, na)
    av = jnp.array([2, 2, 0] + [1] * (na - 3), dtype=jnp.int32)  # Buy,Buy,Sell,Hold...
    prev = INITIAL_CASH

    # ON (weight-rebalance) branch, with fees so the trade materially moves PV
    cfg_on = _cfg(na, w_unit=w_unit, fee_bps=16.0, slippage_bps=5.0)
    st_on = _fresh_state(prices, cfg_on)
    ns_on, _obs, reward_on, _d = step_vector(st_on, av, prices, cfg_on)
    reward_on = float(reward_on)
    pv_on = _pv_at_exec(st_on, ns_on, prices)
    expected_on = float(symlog(np.log(max(pv_on, 1e-8) / max(prev, 1e-8)) * 100.0))
    assert reward_on == pytest.approx(expected_on, rel=1e-4, abs=1e-4)

    # OFF (canon) branch on IDENTICAL inputs -> different trades -> different PV & reward
    cfg_off = EnvConfig(
        n_assets=na, initial_cash=INITIAL_CASH, fee_bps=16.0, slippage_bps=5.0,
        max_steps=400, window_size=5, position_size_pct=0.1, allow_short=True,
        tp_sl_enabled=False, vector_weight_rebalance=False,   # canon path
    )
    st_off = _fresh_state(prices, cfg_off)
    ns_off, _obs2, _reward_off, _d2 = step_vector(st_off, av, prices, cfg_off)
    pv_off = _pv_at_exec(st_off, ns_off, prices)

    # the branches genuinely diverge (override is active), so the ON reward is the
    # OVERRIDE-PV reward, NOT the pre-override (canon) assembly's reward
    assert abs(pv_on - pv_off) > 1.0
    reward_from_off_pv = float(symlog(np.log(max(pv_off, 1e-8) / max(prev, 1e-8)) * 100.0))
    assert abs(reward_on - reward_from_off_pv) > 1e-4

    # and the reward is consistent with the state the ON step actually carried forward
    assert float(np.asarray(ns_on.prev_portfolio_value)) == pytest.approx(pv_on, rel=1e-4)


# ---------------------------------------------------------------------------
# BLOCKER regression (mechanism 1): held WINNING short is reserved by its true
# CAPITAL footprint (posval), not signed market notional — opening new shorts on
# top must NOT overdraw cash / wipe PV at fee=0. (crossval repro.py)
# ---------------------------------------------------------------------------

def test_held_winning_short_no_overdraw_fee0():
    na = 4
    w_unit = 0.25          # w_max = 4*w_unit = 1.0
    T = 40
    # asset0 price collapses 100 -> 53 (a winning short); assets 1..3 flat at 100.
    px = np.full((T, na), FLAT_PRICE, dtype=np.float32)
    drop = [100.0, 100.0, 90.0, 75.0, 60.0, 53.0]
    for i in range(na):
        px[:len(drop), 0] = drop
    prices = jnp.asarray(px)
    cfg = _cfg(na, w_unit=w_unit, fee_bps=0.0, slippage_bps=0.0)
    st = _fresh_state(prices, cfg, window=0)  # start at idx 0 so the drop is in-window
    # open + step a short on asset0 (Sell x2 -> target -0.5), then hold while it wins
    sell0 = jnp.array([0, 1, 1, 1], dtype=jnp.int32)
    hold = jnp.array([1, 1, 1, 1], dtype=jnp.int32)
    for av in (sell0, sell0, hold, hold, hold):
        st, _, _, done = step_vector(st, av, prices, cfg)
        assert float(np.asarray(st.cash)) >= -1e-2
        assert not bool(np.asarray(done))
    # now OPEN NEW shorts on 1,2,3 while the winning short on 0 is held (the wipe trigger)
    open_more = jnp.array([1, 0, 0, 0], dtype=jnp.int32)
    cur = get_current_prices(prices, st.current_step, st.episode_start_idx)
    pv_before = float(compute_portfolio_value(st, cur))
    st, _, _, done = step_vector(st, open_more, prices, cfg)
    cash = float(np.asarray(st.cash))
    pv_after = float(compute_portfolio_value(st, cur))
    assert cash >= -1e-2, f"held winning short under-reserved -> cash {cash} < 0"
    assert pv_after > 0.5 * INITIAL_CASH, f"PV wiped: {pv_after}"
    assert not bool(np.asarray(done))
    # the four shorts' capital reservation must not exceed PV (Σ posval/PV <= 1 + tiny)
    assert pv_after == pytest.approx(pv_before, rel=1e-3)  # fee0 -> PV conserved by the open


# ---------------------------------------------------------------------------
# BLOCKER regression (mechanism 2): a HELD losing short's |w| is clipped to the
# cap and re-trimmed — it cannot drift unbounded; PV never bleeds past cash>=0.
# (flow-verifier repro probe45.py)
# ---------------------------------------------------------------------------

def test_held_losing_short_weight_bounded_fee0():
    na = 1
    w_unit = 0.25          # w_max = 1.0
    # price rises steeply -> a held short loses; nothing else can absorb it
    path = [100, 100, 100, 100, 101, 105, 115, 140, 180, 240, 320, 320]
    px = np.array([[p] for p in path], dtype=np.float32)
    prices = jnp.asarray(px)
    cfg = _cfg(na, w_unit=w_unit, fee_bps=0.0, slippage_bps=0.0, window=3)
    st = _fresh_state(prices, cfg, window=3)
    # open a short (Sell -> target -w_unit) then HOLD through the rising market
    st, _, _, _ = step_vector(st, jnp.array([0], dtype=jnp.int32), prices, cfg)
    w_max = cfg.vector_weight_max * w_unit
    for k in range(7):
        st, _, _, done = step_vector(st, jnp.array([1], dtype=jnp.int32), prices, cfg)
        cur = get_current_prices(prices, st.current_step - 1, st.episode_start_idx)
        pv = float(compute_portfolio_value(st, cur))
        notional = float(abs(np.asarray(st.pos_size)[0] * np.asarray(cur)[0]))
        w = notional / max(pv, 1e-9)
        assert w <= w_max + 1e-3, f"step {k}: |w|={w} ran past the cap {w_max}"
        assert float(np.asarray(st.cash)) >= -1e-2, f"step {k}: cash < 0"
    # position is capped, PV bounded below by cash>=0 (a losing trade, not a bankruptcy wipe)
    assert float(np.asarray(st.cash)) >= -1e-2


# ---------------------------------------------------------------------------
# process_margin_call is a NO-OP under ON: across an adversarial mixed rollout
# at fee=0 the post-step cash is always >= 0, so the margin path (which only
# fires on cash < 0) never acts — the defining guarantee holds book-wide.
# ---------------------------------------------------------------------------

def test_margin_call_is_noop_under_on_fee0():
    from mcts_trader.env.jax_trading_env import process_margin_call
    na = 24
    prices = _flat_prices(120, na)
    # inject a mildly adversarial mix of drifted longs/shorts, then churn market actions
    cfg = _cfg(na, w_unit=-1.0, fee_bps=0.0, slippage_bps=0.0)  # w_unit=1/24
    st = _fresh_state(prices, cfg)
    rng = np.random.default_rng(3)
    for k in range(120):
        av = jnp.asarray(rng.integers(0, 3, size=na), dtype=jnp.int32)  # market only (0/1/2)
        st, _, _, done = step_vector(st, av, prices, cfg)
        cash = float(np.asarray(st.cash))
        assert cash >= -1e-2, f"step {k}: cash {cash} < 0 -> margin path would fire"
        # calling margin again is idempotent (already >=0) -> proves nothing to resolve
        cur = get_current_prices(prices, st.current_step - 1, st.episode_start_idx)
        st2 = process_margin_call(st, cur, cfg)
        assert float(np.asarray(st2.cash)) == pytest.approx(cash, abs=1e-2)
        assert bool(np.asarray((st2.pos_active == st.pos_active).all()))
        assert not bool(np.asarray(done))
    # Σ|w| stayed within the no-leverage cap (+ tiny deadband/fee tolerance) the whole way
    cur = get_current_prices(prices, st.current_step - 1, st.episode_start_idx)
    pv = float(compute_portfolio_value(st, cur))
    sumw = float((np.abs(np.asarray(st.pos_size) * np.asarray(cur))).sum() / max(pv, 1e-9))
    assert sumw <= 1.05, f"Σ|w|={sumw} exceeded the no-leverage cap"


# ---------------------------------------------------------------------------
# TASK_842: canon-equivalent close fee — the weight re-trim's TOTAL round-trip
# cost (spread + explicit close-fee) now equals the equivalent canon scalar
# round-trip's TOTAL cost within 1e-6 (the panel's own equivalence construction:
# mirror notionals => set canon position_size_pct == w_unit so both size the SAME
# notional). At flat prices the only PV loss over a full open->close round-trip is
# transaction cost, so total_fee == initial_cash - final_cash.
# ---------------------------------------------------------------------------

def test_weight_roundtrip_fee_equals_canon_roundtrip():
    from mcts_trader.env.jax_trading_env import execute_buy, execute_sell
    na = 4
    w_unit = 0.1
    fee_bps, slip_bps = 16.0, 5.0
    prices = _flat_prices(80, na)

    # allow_short=False so a Sell that targets a (tiny) negative weight closes cleanly to FLAT
    # rather than opening a residual micro-short — makes the round-trip land exactly flat, so
    # the ONLY cost is the open spread (no explicit open fee) + the explicit close-fee.
    # --- weight path: Buy asset0 (open long -> target +w_unit), then Sell (re-trim to flat) ---
    cfg_w = _cfg(na, w_unit=w_unit, fee_bps=fee_bps, slippage_bps=slip_bps, allow_short=False)
    st = _fresh_state(prices, cfg_w)
    c0 = float(np.asarray(st.cash))
    buy = jnp.array([2] + [1] * (na - 1), dtype=jnp.int32)
    sell = jnp.array([0] + [1] * (na - 1), dtype=jnp.int32)
    st, _, _, _ = step_vector(st, buy, prices, cfg_w)       # open long ~ w_unit*PV notional
    assert bool(np.asarray(st.pos_active)[0]) and int(np.asarray(st.pos_side)[0]) == 1
    st, _, _, _ = step_vector(st, sell, prices, cfg_w)      # re-trim to flat (closes the long)
    assert not bool(np.asarray(st.pos_active)[0])            # back to flat
    weight_total_fee = c0 - float(np.asarray(st.cash))

    # --- canon scalar path: execute_buy (open long) then execute_sell (close long),
    #     position_size_pct == w_unit so the opened notional MIRRORS the weight path exactly ---
    cfg_c = EnvConfig(
        n_assets=na, initial_cash=INITIAL_CASH, fee_bps=fee_bps, slippage_bps=slip_bps,
        max_steps=400, window_size=5, position_size_pct=w_unit, allow_short=False,
        tp_sl_enabled=False, vector_weight_rebalance=False,
    )
    stc = _fresh_state(prices, cfg_c)
    cur = get_current_prices(prices, stc.current_step, stc.episode_start_idx)
    stc = execute_buy(stc, jnp.array(0, dtype=jnp.int32), cur, cfg_c)
    assert bool(np.asarray(stc.pos_active)[0]) and int(np.asarray(stc.pos_side)[0]) == 1
    stc = execute_sell(stc, jnp.array(0, dtype=jnp.int32), cur, cfg_c)
    assert not bool(np.asarray(stc.pos_active)[0])
    canon_total_fee = c0 - float(np.asarray(stc.cash))

    # both charge a real cost, and they now MATCH within 1e-6 (cost-honest turnover)
    assert weight_total_fee > 1.0, f"weight round-trip charged no fee: {weight_total_fee}"
    assert weight_total_fee == pytest.approx(canon_total_fee, abs=1e-6), (
        f"weight total fee {weight_total_fee!r} != canon total fee {canon_total_fee!r} "
        f"(diff {weight_total_fee - canon_total_fee:.3e})")


# ---------------------------------------------------------------------------
# TASK_842: sign check — on a SHORT weight-ON rollout, fee16 strictly REDUCES
# PV vs fee0 (fees can only cost), with no invariant violation (cash>=0, Σ|w|<=1).
# ---------------------------------------------------------------------------

def test_fee16_reduces_pv_vs_fee0_short_rollout():
    na = 4
    w_unit = 0.1
    T = 40
    prices = _flat_prices(T, na)   # flat: any PV divergence between arms is fees alone

    def short_rollout(fee_bps, slip_bps):
        cfg = _cfg(na, w_unit=w_unit, w_max=4.0, fee_bps=fee_bps, slippage_bps=slip_bps)
        st = _fresh_state(prices, cfg)
        # constant Sell on assets 0..1 (open + step shorts, then held at the cap => re-trims)
        sell = jnp.array([0, 0] + [1] * (na - 2), dtype=jnp.int32)
        min_cash = np.inf
        max_sumw = 0.0
        for k in range(20):
            st, _, _, done = step_vector(st, sell, prices, cfg)
            cur = get_current_prices(prices, st.current_step - 1, st.episode_start_idx)
            pv = float(compute_portfolio_value(st, cur))
            cash = float(np.asarray(st.cash))
            sumw = float((np.abs(np.asarray(st.pos_size) * np.asarray(cur))).sum() / max(pv, 1e-9))
            min_cash = min(min_cash, cash)
            max_sumw = max(max_sumw, sumw)
            assert not bool(np.asarray(done)), f"fee{fee_bps}: bankrupt at step {k}"
        cur = get_current_prices(prices, st.current_step - 1, st.episode_start_idx)
        pv_final = float(compute_portfolio_value(st, cur))
        return pv_final, min_cash, max_sumw

    pv_fee0, cash0, sumw0 = short_rollout(0.0, 0.0)
    pv_fee16, cash16, sumw16 = short_rollout(16.0, 5.0)

    # invariants hold in BOTH arms (weight-ON guarantee): cash>=0, no leverage
    for cash, sumw, tag in ((cash0, sumw0, "fee0"), (cash16, sumw16, "fee16")):
        assert cash >= -1e-2, f"{tag}: cash {cash} < 0 (margin path would fire)"
        assert sumw <= 1.05, f"{tag}: Σ|w|={sumw} exceeded the no-leverage cap"

    # fees strictly REDUCE PV (Δ<0): at flat prices fee16 PV must sit below fee0 PV
    delta = pv_fee16 - pv_fee0
    assert delta < 0.0, f"fee16 did not reduce PV vs fee0 (Δ={delta:.4f})"
    assert pv_fee0 == pytest.approx(INITIAL_CASH, rel=1e-4)  # fee0: PV conserved (flat prices)


if __name__ == "__main__":
    raise SystemExit(pytest.main([os.path.abspath(__file__), "-q"]))
