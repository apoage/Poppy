"""
TASK_828 Stage 1 — tests for the vector-action env step (`step_vector`).

Covers the three factory-mandated acceptance checks:
  1. Equivalence   — one `step_vector` == the equivalent SEQUENCE of scalar `step`s on the
                     same start state, in the regime where the simultaneous-rebalance
                     fee/margin divergence vanishes (flat prices, ample cash, distinct assets,
                     fee=0). The intended divergence is asserted separately.
  2. Conservation  — cash+positions value is conserved under a simultaneous rebalance
                     (no money created/destroyed beyond fees). Checked on synthetic AND on a
                     real price slice from fixtures/trading_fixture.db.
  3. Canon parity  — a short scalar-path rollout hashes to a pinned value; step_vector /
                     the new config field do NOT perturb the canon scalar path.

CPU-only, never GPU, never trading_data.db. Run:
    JAX_PLATFORMS=cpu pytest tools/tests/test_vector_env.py -v
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")  # never GPU for env-primitive tests

import hashlib
import sqlite3

import pytest
import jax
import jax.numpy as jnp
import numpy as np

import sys
from pathlib import Path

# Robust, machine-independent import shim. `mcts_trader/` is a PEP-420 namespace package
# (no __init__.py), so the import only resolves once the repo root is on sys.path. We add
# it here — computed from THIS file's location, not hardcoded — so the module imports under
# pytest's DEFAULT (prepend) import mode from any cwd/checkout, with no --import-mode flag
# and no dependence on the workstation's absolute path. This only PREPENDS the real repo
# root; it never shadows or masks a genuine ImportError from the target module.
_REPO_ROOT = Path(__file__).resolve().parents[2]  # tools/tests/ -> tools/ -> repo root
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from mcts_trader.env.jax_trading_env import (
    EnvConfig, reset, step, step_vector,
    jit_step_vector, batch_step_vector, jit_batch_step_vector, batch_reset,
    compute_portfolio_value, get_current_prices,
)
from mcts_trader.utils.transforms import symlog

FIXTURE_DB = "/home/lukas/Trader/fixtures/trading_fixture.db"


# =============================================================================
# Helpers
# =============================================================================

def flat_prices(n_assets, n_steps=300, base=100.0):
    """Constant prices across time -> price index is immaterial, portfolio invariant."""
    per_asset = base + 5.0 * np.arange(n_assets)  # distinct levels, flat in time
    return jnp.asarray(np.tile(per_asset[None, :], (n_steps, 1)), dtype=jnp.float32)


def load_fixture_prices(n_assets=8, n_steps=250):
    """Pivot a clean (n_steps, n_assets) close-price panel from the fixture's daily OHLCV."""
    con = sqlite3.connect(FIXTURE_DB)
    cur = con.cursor()
    top = [r[0] for r in cur.execute(
        "SELECT asset_id FROM ohlcv_daily GROUP BY asset_id "
        "ORDER BY COUNT(*) DESC LIMIT ?", (n_assets,)).fetchall()]
    series = {}
    common = None
    for aid in top:
        rows = cur.execute(
            "SELECT timestamp, close FROM ohlcv_daily WHERE asset_id=? "
            "AND close IS NOT NULL AND close>0 ORDER BY timestamp", (aid,)).fetchall()
        d = {ts: c for ts, c in rows}
        series[aid] = d
        common = set(d) if common is None else (common & set(d))
    con.close()
    ts_sorted = sorted(common)[-n_steps:]
    panel = np.array([[series[aid][ts] for aid in top] for ts in ts_sorted], dtype=np.float32)
    return jnp.asarray(panel)


def base_config(n_assets, fee_bps=0.0, slippage_bps=0.0, tp_sl=False, allow_short=True,
                reward_attributable=False):
    return EnvConfig(
        n_assets=n_assets, initial_cash=100_000.0,
        fee_bps=fee_bps, slippage_bps=slippage_bps,
        max_steps=120, window_size=10, position_size_pct=0.1,
        allow_short=allow_short, tp_sl_enabled=tp_sl,
        reward_attributable=reward_attributable,
    )


def state_with_positions(cfg, prices, seed, longs=None, shorts=None, cash=50_000.0):
    """Build a start state carrying pre-existing positions.

    ``longs``/``shorts`` map asset_idx -> (entry_price, size). ``prev_portfolio_value`` and
    ``initial_portfolio_value`` are pinned to the state's mark-to-market value at the current
    (flat) prices so the whole-portfolio log-return term of the very next step is 0 in the
    fee-free / value-conserving regime — isolating the realized-PnL-on-close attribution term.
    """
    longs = longs or {}
    shorts = shorts or {}
    s = reset(jax.random.PRNGKey(seed), prices, cfg)
    na = cfg.n_assets
    pos_active = np.zeros(na, dtype=bool)
    pos_side = np.zeros(na, dtype=np.int32)
    pos_size = np.zeros(na, dtype=np.float32)
    pos_entry = np.zeros(na, dtype=np.float32)
    for i, (entry, size) in longs.items():
        pos_active[i], pos_side[i], pos_size[i], pos_entry[i] = True, 1, size, entry
    for i, (entry, size) in shorts.items():
        pos_active[i], pos_side[i], pos_size[i], pos_entry[i] = True, 2, size, entry
    s = s.replace(
        cash=jnp.asarray(cash, dtype=jnp.float32),
        pos_active=jnp.asarray(pos_active),
        pos_side=jnp.asarray(pos_side),
        pos_size=jnp.asarray(pos_size),
        pos_entry=jnp.asarray(pos_entry),
    )
    cur = get_current_prices(prices, s.current_step, s.episode_start_idx)
    pv = compute_portfolio_value(s, cur)
    return s.replace(prev_portfolio_value=pv, initial_portfolio_value=pv)


def material(state):
    """The state fields that must match; excludes step counter / rng / prev_pv."""
    return (
        float(state.cash),
        np.asarray(state.pos_active),
        np.asarray(state.pos_side),
        np.asarray(state.pos_size),
        np.asarray(state.pos_entry),
        np.asarray(state.order_active),
        np.asarray(state.order_side),
        np.asarray(state.order_size),
    )


# =============================================================================
# 1. Equivalence: vector step == sequence of scalar steps (exact-agreement regime)
# =============================================================================

class TestEquivalence:

    def test_vector_equals_scalar_sequence_flat_ample_cash(self):
        """Flat prices + ample cash + distinct assets + fee=0 => step_vector is bit-equal
        to the equivalent SEQUENCE of scalar step()s (the simultaneous-rebalance divergence
        vanishes because portfolio value is invariant and affordability never binds)."""
        na = 8
        cfg = base_config(na)
        prices = flat_prices(na)

        # Per-asset action_vec: distinct assets get buy / short / limits / hold.
        action_vec = jnp.array([2, 0, 1, 4, 3, 2, 0, 1], dtype=jnp.int32)

        # --- Vector step ---
        sv = reset(jax.random.PRNGKey(3), prices, cfg)
        sv_out, _, _, _ = step_vector(sv, action_vec, prices, cfg)

        # --- Equivalent scalar SEQUENCE: one scalar step per non-hold asset ---
        ss = reset(jax.random.PRNGKey(3), prices, cfg)
        for asset_idx in range(na):
            atype = int(action_vec[asset_idx])
            if atype == 1:  # hold -> skip (no-op, keeps the comparison a rebalance)
                continue
            scalar_action = jnp.array(asset_idx * 5 + atype, dtype=jnp.int32)
            ss, _, _, _ = step(ss, scalar_action, prices, cfg)

        mv, ms = material(sv_out), material(ss)
        assert np.isclose(mv[0], ms[0], rtol=1e-5, atol=1e-3), (mv[0], ms[0])
        assert np.array_equal(mv[1], ms[1])                    # pos_active
        assert np.array_equal(mv[2], ms[2])                    # pos_side
        assert np.allclose(mv[3], ms[3], rtol=1e-5, atol=1e-3) # pos_size
        assert np.allclose(mv[4], ms[4], rtol=1e-5, atol=1e-3) # pos_entry
        assert np.array_equal(mv[5], ms[5])                    # order_active
        assert np.array_equal(mv[6], ms[6])                    # order_side
        assert np.allclose(mv[7], ms[7], rtol=1e-5, atol=1e-3) # order_size

    def test_all_hold_is_noop(self):
        na = 6
        cfg = base_config(na)
        prices = flat_prices(na)
        s = reset(jax.random.PRNGKey(1), prices, cfg)
        out, _, _, _ = step_vector(s, jnp.ones(na, dtype=jnp.int32), prices, cfg)
        assert float(out.cash) == float(s.cash)
        assert not bool(jnp.any(out.pos_active))
        assert not bool(jnp.any(out.order_active))

    def test_intended_divergence_when_affordability_binds(self):
        """DOCUMENTED intended divergence: when every asset is bought at once, each buy is
        sized against the SAME snapshot cash (simultaneity), so combined demand exceeds cash
        and the margin path engages — whereas the scalar SEQUENCE re-reads depleted cash and
        never overspends. Assert the two genuinely diverge (proves the semantics differ, on
        purpose)."""
        na = 32
        cfg = base_config(na)                     # pct=0.1 -> 32 buys demand ~3.2x cash
        prices = flat_prices(na)
        action_vec = jnp.full((na,), 2, dtype=jnp.int32)  # buy every asset simultaneously

        sv = reset(jax.random.PRNGKey(5), prices, cfg)
        sv_out, _, _, _ = step_vector(sv, action_vec, prices, cfg)

        ss = reset(jax.random.PRNGKey(5), prices, cfg)
        for asset_idx in range(na):
            ss, _, _, _ = step(ss, jnp.array(asset_idx * 5 + 2, dtype=jnp.int32), prices, cfg)

        # Sequential never overdraws (cash stays >=0 via per-step affordability);
        # simultaneous drives cash negative -> margin call liquidates -> different book.
        assert not np.allclose(material(sv_out)[3], material(ss)[3]), \
            "expected simultaneous-rebalance divergence but books matched"
        assert float(ss.cash) >= -1.0  # scalar sequence respects cash


# =============================================================================
# 2. Conservation: no money created/destroyed beyond fees
# =============================================================================

class TestConservation:

    def _pv_before_after(self, cfg, prices, action_vec, seed):
        s = reset(jax.random.PRNGKey(seed), prices, cfg)
        cur = get_current_prices(prices, s.current_step, s.episode_start_idx)
        pv_before = float(compute_portfolio_value(s, cur))
        out, _, _, _ = step_vector(s, action_vec, prices, cfg)
        # value the POST-rebalance book at the SAME instant's prices
        pv_after = float(compute_portfolio_value(out, cur))
        return pv_before, pv_after

    def test_conservation_flat_no_fee(self):
        na = 16
        cfg = base_config(na, fee_bps=0.0)
        prices = flat_prices(na)
        rng = np.random.default_rng(0)
        action_vec = jnp.asarray(rng.integers(0, 5, size=na), dtype=jnp.int32)
        before, after = self._pv_before_after(cfg, prices, action_vec, seed=11)
        # fee=slip=0 -> a rebalance at current prices conserves portfolio value exactly
        assert np.isclose(before, after, rtol=1e-5, atol=1e-2), (before, after)
        assert np.isclose(before, cfg.initial_cash, rtol=1e-5)

    def test_conservation_fixture_prices_no_fee(self):
        prices = load_fixture_prices(n_assets=8, n_steps=200)
        na = int(prices.shape[1])
        cfg = base_config(na, fee_bps=0.0)
        rng = np.random.default_rng(2)
        action_vec = jnp.asarray(rng.integers(0, 5, size=na), dtype=jnp.int32)
        before, after = self._pv_before_after(cfg, prices, action_vec, seed=23)
        assert np.isclose(before, after, rtol=1e-4, atol=1e-1), (before, after)

    def test_fees_only_reduce_value(self):
        na = 12
        prices = flat_prices(na)
        rng = np.random.default_rng(7)
        action_vec = jnp.asarray(rng.integers(0, 3, size=na), dtype=jnp.int32)  # market only
        _, after_nofee = self._pv_before_after(base_config(na, fee_bps=0.0), prices, action_vec, 31)
        _, after_fee = self._pv_before_after(base_config(na, fee_bps=25.0, slippage_bps=5.0),
                                             prices, action_vec, 31)
        # fees can only destroy value, never create it
        assert after_fee <= after_nofee + 1e-3


# =============================================================================
# 3. Canon parity: scalar path untouched (pinned hash) + JIT/batch sanity
# =============================================================================

class TestCanonParity:

    # Pinned from a scalar-path rollout on the pre-TASK_828 module (git HEAD) and confirmed
    # bit-identical on the post-change module. Any drift here means the canon path moved.
    PINNED = "328634c396058726cf508a8d0dd3d2cc6be6bd2b2f0b326fc815bfe384caf9c9"

    def test_scalar_rollout_hash_pinned(self):
        cfg = EnvConfig(n_assets=8, initial_cash=100000.0, fee_bps=10.0, slippage_bps=5.0,
                        max_steps=60, window_size=10, position_size_pct=0.1, allow_short=True,
                        tp_sl_enabled=True)
        rng = np.random.default_rng(0)
        T, na = 300, 8
        prices = jnp.asarray(
            100 + np.cumsum(rng.normal(0, 1, size=(T, na)), axis=0).clip(-40, 40)
            + (np.arange(T)[:, None] * 0.02), dtype=jnp.float32)
        state = reset(jax.random.PRNGKey(7), prices, cfg)
        acts = [3, 17, 1, 22, 9, 14, 0, 31, 5, 2, 19, 7, 12, 25, 1, 4, 33, 8, 20, 6]
        outs = []
        for i in range(60):
            state, _, r, _ = step(state, jnp.array(acts[i % len(acts)]), prices, cfg)
            outs.append((float(state.cash), float(r), int(state.current_step),
                         [float(x) for x in state.pos_size],
                         [float(x) for x in state.pos_entry],
                         [int(x) for x in state.pos_side],
                         [int(x) for x in state.order_active]))
        h = hashlib.sha256(repr(outs).encode()).hexdigest()
        assert h == self.PINNED, f"canon scalar path drifted: {h}"


class TestJitAndBatch:

    def test_jit_step_vector_matches_eager(self):
        na = 10
        cfg = base_config(na)
        prices = flat_prices(na)
        rng = np.random.default_rng(4)
        av = jnp.asarray(rng.integers(0, 5, size=na), dtype=jnp.int32)
        s = reset(jax.random.PRNGKey(9), prices, cfg)
        e_out, _, e_r, _ = step_vector(s, av, prices, cfg)
        j_out, _, j_r, _ = jit_step_vector(s, av, prices, cfg)
        jax.block_until_ready(j_out.cash)
        assert np.isclose(float(e_out.cash), float(j_out.cash), rtol=1e-5, atol=1e-2)
        assert np.allclose(np.asarray(e_out.pos_size), np.asarray(j_out.pos_size),
                           rtol=1e-5, atol=1e-2)
        assert np.isclose(float(e_r), float(j_r), rtol=1e-5, atol=1e-4)

    def test_batch_step_vector_shapes(self):
        na = 8
        cfg = base_config(na)
        prices = flat_prices(na)
        n_envs = 16
        keys = jax.random.split(jax.random.PRNGKey(0), n_envs)
        states = batch_reset(keys, prices, cfg)
        rng = np.random.default_rng(6)
        avs = jnp.asarray(rng.integers(0, 5, size=(n_envs, na)), dtype=jnp.int32)
        out, obs, rewards, dones = jit_batch_step_vector(states, avs, prices, cfg)
        jax.block_until_ready(rewards)
        assert out.cash.shape == (n_envs,)
        assert rewards.shape == (n_envs,)
        assert dones.shape == (n_envs,)
        assert out.pos_size.shape == (n_envs, na)


# =============================================================================
# 4. reward_attributable: realized-PnL-on-close term (sign / entry-snapshot / sum)
# =============================================================================

class TestRewardAttributable:
    """Lock the vectorised realized-PnL-on-close term (`--reward-attributable`) against the
    canon scalar path. Regime: fee=0, ample cash, distinct assets, flat-in-TIME prices so the
    scalar SEQUENCE and the single vector step read the same price and the whole-portfolio
    log-return term is 0 — but the pre-existing positions carry entry != current ("moving
    prices" relative to entry) so CLOSING them realizes genuine, non-zero PnL. In this regime
    the vector step's reward must equal the aggregate reward of the equivalent scalar sequence.
    """

    def _scalar_sequence_reward(self, s, action_vec, prices, cfg):
        """Aggregate reward of the scalar SEQUENCE equivalent to one vector rebalance."""
        na = cfg.n_assets
        total = 0.0
        for i in range(na):
            atype = int(action_vec[i])
            if atype == 1:  # hold -> no scalar step (keeps the sequence a pure rebalance)
                continue
            s, _, r, _ = step(s, jnp.array(i * 5 + atype, dtype=jnp.int32), prices, cfg)
            total += float(r)
        return s, total

    def test_single_long_close_reward_matches_scalar(self):
        """Long AND short pre-exist; the LONG closes in profit (entry<current) while the short
        closes at zero PnL (entry==current). Vector reward == scalar aggregate AND is non-zero,
        locking the long sign, the pre-action entry snapshot, and current-price-as-exit."""
        na = 4
        cfg = base_config(na, reward_attributable=True)
        prices = flat_prices(na)                       # P0=100, P1=105, ...
        s0 = state_with_positions(cfg, prices, seed=41,
                                  longs={0: (100.0 / 1.1, 10.0)},   # +log(1.1) on close
                                  shorts={1: (105.0, 10.0)})        # entry==current -> 0
        action_vec = jnp.array([0, 2, 1, 1], dtype=jnp.int32)       # sell long, buy-close short

        sv_out, _, rv, _ = step_vector(s0, action_vec, prices, cfg)
        ss_out, scalar_total = self._scalar_sequence_reward(
            state_with_positions(cfg, prices, seed=41,
                                 longs={0: (100.0 / 1.1, 10.0)}, shorts={1: (105.0, 10.0)}),
            action_vec, prices, cfg)

        assert np.array_equal(material(sv_out)[1], material(ss_out)[1])   # books agree
        assert float(rv) > 0.1, float(rv)                                # genuinely non-zero, +
        assert np.isclose(float(rv), scalar_total, rtol=1e-5, atol=1e-4), (float(rv), scalar_total)
        # Expected magnitude: base term 0 + symlog(log(1.1)*100).
        assert np.isclose(float(rv), float(symlog(np.log(1.1) * 100.0)), atol=1e-4)

    def test_single_short_close_reward_matches_scalar(self):
        """Mirror of the above: the SHORT closes at a loss (shorted below current) while the long
        closes at zero PnL. Vector reward == scalar aggregate AND is non-zero NEGATIVE — locking
        the short mirror sign (-log(exit/entry)); a flipped sign would go positive and mismatch."""
        na = 4
        cfg = base_config(na, reward_attributable=True)
        prices = flat_prices(na)
        kw = dict(longs={1: (105.0, 10.0)}, shorts={0: (100.0 / 1.1, 10.0)})  # short entry<current
        action_vec = jnp.array([2, 0, 1, 1], dtype=jnp.int32)                  # buy-close short, sell long

        sv_out, _, rv, _ = step_vector(state_with_positions(cfg, prices, 42, **kw),
                                       action_vec, prices, cfg)
        ss_out, scalar_total = self._scalar_sequence_reward(
            state_with_positions(cfg, prices, 42, **kw), action_vec, prices, cfg)

        assert np.array_equal(material(sv_out)[1], material(ss_out)[1])
        assert float(rv) < -0.1, float(rv)                               # non-zero, sign is NEGATIVE
        assert np.isclose(float(rv), scalar_total, rtol=1e-5, atol=1e-4), (float(rv), scalar_total)
        assert np.isclose(float(rv), float(symlog(-np.log(1.1) * 100.0)), atol=1e-4)

    def test_multiclose_sum_matches_scalar(self):
        """Both a long and a short close with EQUAL-magnitude, OPPOSITE-sign realized PnL. The
        vectorised term sums BEFORE symlog: symlog(r_long + r_short) = symlog(0) = 0, matching the
        scalar aggregate symlog(r_long)+symlog(r_short)=X-X=0. Locks that step_vector SUMS realized
        PnL over EVERY closed asset (drop the sum, or flip the short mirror, and this diverges)."""
        na = 4
        cfg = base_config(na, reward_attributable=True)
        prices = flat_prices(na)                       # P0=100, P1=105
        kw = dict(longs={0: (100.0 / 1.1, 10.0)},      # +log(1.1)
                  shorts={1: (105.0 / 1.1, 10.0)})     # -log(1.1)  (105/1.1 < 105 -> short at loss)
        action_vec = jnp.array([0, 2, 1, 1], dtype=jnp.int32)

        sv_out, _, rv, _ = step_vector(state_with_positions(cfg, prices, 43, **kw),
                                       action_vec, prices, cfg)
        ss_out, scalar_total = self._scalar_sequence_reward(
            state_with_positions(cfg, prices, 43, **kw), action_vec, prices, cfg)

        # both positions closed
        assert not bool(jnp.any(sv_out.pos_active))
        assert np.array_equal(material(sv_out)[1], material(ss_out)[1])
        # vector term cancels to ~0; scalar aggregate also ~0 (X + (-X))
        assert np.isclose(float(rv), 0.0, atol=1e-4), float(rv)
        assert np.isclose(float(rv), scalar_total, atol=1e-4), (float(rv), scalar_total)


# =============================================================================
# 5. Forced liquidation: symlog(-50) sentinel + suppression of the attribution term
# =============================================================================

class TestLiquidation:

    def test_forced_liquidation_reward_and_suppression(self):
        """Drive step_vector below the liquidation threshold and assert:
          (a) reward is the symlog(-50) sentinel (the `jnp.where(is_liquidated, ...)` base branch);
          (b) the realized-PnL attribution term is SUPPRESSED under liquidation
              (`reward + jnp.where(is_liquidated, 0.0, ...)`), even though a position closes this
              step with large non-zero realized PnL — i.e. the sentinel is not perturbed;
          (c) the book ends valid (no active positions, finite non-negative cash / portfolio).
        A short shorted at 10 while price is 100 is deeply underwater; closing it drives cash
        negative -> margin call -> bankruptcy -> portfolio 0 < threshold(=1% of 100k = 1000)."""
        na = 4
        cfg = base_config(na, reward_attributable=True)   # attribution ON so suppression is exercised
        prices = flat_prices(na)                          # P0=100
        s0 = state_with_positions(cfg, prices, seed=71,
                                  shorts={0: (10.0, 2.0)},  # short @10, now 100 -> value collapses
                                  cash=100.0)

        out, _, reward, done = step_vector(
            s0, jnp.array([2, 1, 1, 1], dtype=jnp.int32), prices, cfg)  # buy-close the doomed short

        cur = get_current_prices(prices, s0.current_step, s0.episode_start_idx)
        pv_after = float(compute_portfolio_value(out, cur))

        # (a) sentinel reward, exact
        assert np.isclose(float(reward), float(symlog(-50.0)), atol=1e-5), float(reward)
        assert bool(done)
        # (b) suppression is load-bearing: had the attribution term NOT been gated off, the reward
        #     would be sentinel + symlog(realized*100) with realized = -log(100/10) != 0.
        realized = -np.log(100.0 / 10.0)                 # short mirror on close
        unsuppressed = float(symlog(-50.0)) + float(symlog(realized * 100.0))
        assert not np.isclose(float(reward), unsuppressed, atol=1e-3), (float(reward), unsuppressed)
        # (c) book valid
        assert not bool(jnp.any(out.pos_active))
        assert np.isfinite(float(out.cash)) and float(out.cash) >= 0.0
        assert np.isfinite(pv_after) and pv_after >= 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
