#!/usr/bin/env python3
"""TASK_839 acceptance probe — budget-constrained vector step (the env-wall fix).

Two PRE-REGISTERED reads (SPEC.md), fixture / true-model only, CPU-only, never
trading_data.db. Cite: LEDGER 828s3/env-wall; DECISIONS D7 (RATIFIED option (a)).

(i) 834-repro SURVIVAL WALK (control + fix). Zero-logit dithered-book proposal
    (hold_bias=0.0 => near-uniform per-asset actions), legality-masked EXACTLY as
    battery self-play (batch_compute_action_mask, self_play.py:661 operator), 4 envs,
    position_size_pct=0.1 (the training / env-wall regime), on the DIVERSIFIED daily
    panel (the H6 breadth regime — the 1-min fixture ohlcv has only 8 unique assets, so
    padding to 64 would fabricate correlated exposure). Under OFF the near-uniform first
    action reproduces the env-wall (bankruptcy at step 0). Under ON the walk must survive
    >=64 steps with ZERO bankruptcies. Reports per-arm bankruptcy step / survivor count
    and, load-bearing, the count of step-level cash OVERDRAFT events (cash<0 BEFORE the
    margin call) — the fix drives that to zero; any residual death is the acknowledged
    margin-exception path, not the simultaneous-overdraft wall.

    A separate cheap check builds the env through the TRAINING factory
    (create_jax_env_historical) with the flag ON/OFF and asserts it propagates to
    EnvConfig — end-to-end proof of the env_factory threading.

(ii) E1-pattern PERFECT-MODEL probe at K16 (mirrors experiments/831_search_probe —
    dithered-book scheme, truncated true-env HOLD-rollout value), ON vs OFF, at the
    net@21 cost tier (fee16/slip5, the 831 metric). Pre-registered read:
    ON >= OFF - 0.10 (budget sizing must not destroy the NARROW-vector regime E1 lives
    in — a tie/small-loss is a PASS, a collapse is a FAIL). In the E1 regime
    position_size_pct = 0.5/A, so budget_pct/n_trades >= 0.99/A > 0.5/A = pct for any
    n_trades <= A: the min() ALWAYS hits the pct cap => the ON sizing is a provable
    no-op (delta ~= 0). We still RUN both arms and report the real measured numbers.
    A FINDING report (838 pattern), not a ship gate.

Run:  JAX_PLATFORMS=cpu python experiments/839_budget_step/acceptance_probe.py --out <path>
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import importlib.util
import json
import sys
import time
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from mcts_trader.env.jax_trading_env import (
    EnvConfig, reset, step_vector, get_current_prices, compute_portfolio_value,
    batch_reset, batch_step_vector, batch_compute_action_mask,
)
from mcts_trader.jax_agent.sampled_search import SampledMuZeroVectorSearch
from mcts_trader.training.env_factory import create_jax_env_historical

FIXTURE_DB = os.path.join(_REPO, "fixtures", "trading_fixture.db")
PROTECTED_DB = os.path.join(_REPO, "trading_data.db")
INITIAL_CASH = 100_000.0
N_ACTIONS_PER_ASSET = 5
FEE_BPS, SLIP_BPS = 16.0, 5.0           # net@21 tier (16+5 bp/side), mirrors 830/831


def _load_corridor():
    """Import TASK_830's corridor module by path (dir name is not an identifier)."""
    path = os.path.join(_REPO, "experiments", "830_corridor", "corridor.py")
    spec = importlib.util.spec_from_file_location("corridor830_839", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# =============================================================================
# factory-threading check — the flag reaches EnvConfig through create_jax_env_historical
# =============================================================================

def factory_threading_check(db, start, end, bar_minutes):
    """End-to-end proof that --vector-budget-step propagates through the TRAINING factory
    (create_jax_env_historical) into EnvConfig. Cheap: 1 env, 8 assets, 1-min fixture."""
    out = {}
    for flag in (False, True):
        _, _, env_cfg = create_jax_env_historical(
            db_path=db, start_date=start, end_date=end, n_envs=1, n_assets=8,
            fee_bps=0.0, slippage_bps=0.0, max_episode_steps=32, bar_minutes=bar_minutes,
            vector_action=True, vector_budget_step=flag, vector_budget_pct=0.97,
            rng_key=jax.random.PRNGKey(0),
        )
        out[f"budget_{flag}"] = {
            "env_vector_budget_step": bool(env_cfg.vector_budget_step),
            "env_vector_budget_pct": float(env_cfg.vector_budget_pct),
        }
    ok = (out["budget_False"]["env_vector_budget_step"] is False and
          out["budget_True"]["env_vector_budget_step"] is True and
          abs(out["budget_True"]["env_vector_budget_pct"] - 0.97) < 1e-6)
    out["propagates"] = bool(ok)
    return out


# =============================================================================
# (i) survival walk — the env-wall control (OFF) + the fix (ON)
# =============================================================================

def survival_walk(close_panel, budget_step, n_envs, n_walk, seed,
                  position_size_pct=0.1, max_positions=16, min_cash_threshold=100.0,
                  window=5):
    """Walk `n_walk` near-uniform (zero-logit dithered-book, hb0.0) steps over `n_envs`
    envs on a fixed-start daily-panel window; return per-env bankruptcy step (-1 =
    survived), survivor count, and the count of step-level cash OVERDRAFT events.

    Actions are legality-masked EXACTLY as battery self-play (batch_compute_action_mask,
    max_positions/min_cash_threshold). The mask is computed on the PRE-step positions, so
    at step 0 (no positions) it does NOT cap the near-uniform opens — the env-wall control
    is preserved under OFF — while on later steps it caps the book at max_positions."""
    T, A = close_panel.shape
    prices = jnp.asarray(close_panel, dtype=jnp.float32)
    cfg = EnvConfig(
        n_assets=A, initial_cash=INITIAL_CASH, fee_bps=0.0, slippage_bps=0.0,
        window_size=window, position_size_pct=position_size_pct,
        allow_short=True, tp_sl_enabled=False, max_steps=n_walk + 5,
        vector_action=True, vector_budget_step=budget_step, vector_budget_pct=0.99,
    )
    keys = jax.random.split(jax.random.PRNGKey(seed), n_envs)
    states = batch_reset(keys, prices, cfg)
    # Fixed start at `window` for determinism + headroom (835 pattern); per-env action
    # randomness (dither+sample) still gives n_envs independent walks on the same window.
    states = states.replace(
        episode_start_idx=jnp.full_like(states.episode_start_idx, window),
        current_step=jnp.zeros_like(states.current_step),
        max_episode_steps=jnp.full_like(states.max_episode_steps, n_walk + 2),
        initial_portfolio_value=jnp.full_like(states.initial_portfolio_value, INITIAL_CASH),
        prev_portfolio_value=jnp.full_like(states.prev_portfolio_value, INITIAL_CASH),
        cash=jnp.full_like(states.cash, INITIAL_CASH),
    )
    sampler = SampledMuZeroVectorSearch(
        model_apply_fn=None, n_assets=A, n_actions_per_asset=N_ACTIONS_PER_ASSET,
        num_samples=16, hold_bias=0.0, dither_scale=1.0, proposal_scheme="dithered-book")
    zero_logits = jnp.zeros((n_envs, A * N_ACTIONS_PER_ASSET), dtype=jnp.float32)

    def _sum_market_spend(st, avs):
        """Aggregate MARKET-open spend per env at this step (buys + opening shorts), used to
        detect a would-be cash overdraft BEFORE process_margin_call cleans it up."""
        cur = get_current_prices(prices, st.current_step, st.episode_start_idx)  # (n_envs, A)
        pv = jax.vmap(compute_portfolio_value)(st, cur)                          # (n_envs,)
        has_pos = st.pos_active
        is_short = st.pos_side == 2
        a_buy, a_sell = (avs == 2), (avs == 0)
        open_long = a_buy & (~(has_pos & is_short))
        open_short = a_sell & (~has_pos)
        n_open = jnp.sum((open_long | open_short).astype(jnp.int32), axis=-1)    # (n_envs,)
        if budget_step:
            pct_eff = jnp.minimum(position_size_pct, 0.99 / jnp.maximum(n_open, 1))
        else:
            pct_eff = jnp.full_like(pv, position_size_pct)
        per_trade = pv * pct_eff
        spend = per_trade * n_open.astype(jnp.float32)
        return spend, st.cash

    @jax.jit
    def walk_step(states, key):
        dither_key, sample_key = jax.random.split(key)
        price_idx = states.episode_start_idx + states.current_step
        current_prices = prices[price_idx]                                       # (n_envs, A)
        mask = batch_compute_action_mask(states, current_prices, cfg,
                                         max_positions, min_cash_threshold)        # (n_envs, A*5)
        proposal = sampler._build_proposal(zero_logits, mask, None, dither_key)    # (n_envs, A, 5)
        avs = jax.random.categorical(sample_key, proposal, axis=-1).astype(jnp.int32)  # (n_envs, A)
        spend, cash = _sum_market_spend(states, avs)
        new_states, _, _, dones = batch_step_vector(states, avs, prices, cfg)
        return new_states, dones, avs, spend, cash

    base = jax.random.PRNGKey(seed + 777)
    bankrupt_step = np.full(n_envs, -1, dtype=np.int64)
    done_ever = np.zeros(n_envs, dtype=bool)
    overdraft_events = 0
    first_open_frac = None
    for t in range(n_walk):
        states, dones, avs, spend, cash = walk_step(states, jax.random.fold_in(base, t))
        d = np.asarray(dones).astype(bool)
        newly = d & (~done_ever)
        bankrupt_step[newly] = t
        # count overdrafts only on still-live envs (a bankrupt env has ~0 cash forever)
        live = ~done_ever
        would_overdraw = (np.asarray(spend) > np.asarray(cash) * 0.99 + 1.0) & live
        overdraft_events += int(would_overdraw.sum())
        done_ever |= d
        if t == 0:
            av0 = np.asarray(avs)
            first_open_frac = float(np.mean((av0 == 2) | (av0 == 0)))
    jax.block_until_ready(states.cash)
    return {
        "budget_step": bool(budget_step),
        "n_envs": int(n_envs), "n_walk": int(n_walk), "n_assets": int(A),
        "position_size_pct": float(position_size_pct), "max_positions": int(max_positions),
        "bankrupt_step_per_env": [int(x) for x in bankrupt_step],
        "n_survived": int((~done_ever).sum()),
        "n_bankrupt": int(done_ever.sum()),
        "step_level_overdraft_events": int(overdraft_events),
        "first_step_open_frac": first_open_frac,
    }


# =============================================================================
# (ii) E1-pattern perfect-model probe — net@21, ON vs OFF (mirror of 831)
# =============================================================================

def e1_rollout(close_panel, K, seed, warm, corridor, fee_bps, slip_bps, position_size_pct,
               discount, value_rollout_steps, max_decisions, proposal_bias, budget_step,
               window=5):
    """Perfect-model K-sample search rollout (dithered-book scheme), 831 pattern, with the
    vector_budget_step toggle threaded into the EnvConfig. Returns per-decision returns."""
    T, A = close_panel.shape
    prices = jnp.asarray(close_panel, dtype=jnp.float32)
    cfg = EnvConfig(
        n_assets=A, initial_cash=INITIAL_CASH, fee_bps=fee_bps, slippage_bps=slip_bps,
        window_size=window, position_size_pct=position_size_pct,
        allow_short=True, tp_sl_enabled=False, max_steps=400, vector_action=True,
        vector_budget_step=budget_step, vector_budget_pct=0.99,
    )
    N = T - 1 - warm - value_rollout_steps
    if max_decisions is not None:
        N = min(N, int(max_decisions))
    if N < 2:
        raise RuntimeError(f"panel too short: N={N}")

    sampler = SampledMuZeroVectorSearch(
        model_apply_fn=None, n_assets=A, n_actions_per_asset=N_ACTIONS_PER_ASSET, num_samples=K)
    state = reset(jax.random.PRNGKey(0), prices, cfg).replace(
        episode_start_idx=jnp.array(warm), current_step=jnp.array(0),
        max_episode_steps=jnp.array(N + value_rollout_steps + 2),
        initial_portfolio_value=jnp.array(INITIAL_CASH),
        prev_portfolio_value=jnp.array(INITIAL_CASH),
    )
    jstep = partial(step_vector, prices=prices, config=cfg)
    hold_vec = jnp.ones((A,), dtype=jnp.int32)

    def hold_step(carry_state, _):
        ns, _, r, _ = jstep(carry_state, hold_vec)
        return ns, r

    def eval_candidate(st, vec):
        ns, _, r0, _ = jstep(st, vec)
        if value_rollout_steps > 0:
            _, rs = jax.lax.scan(hold_step, ns, xs=None, length=value_rollout_steps)
            disc = discount ** jnp.arange(value_rollout_steps)
            V = jnp.sum(disc * rs)
        else:
            V = jnp.array(0.0)
        return r0 + discount * V

    @jax.jit
    def search_step(st, cand):
        return jax.vmap(lambda v: eval_candidate(st, v), in_axes=0)(cand)

    jstep_j = jax.jit(jstep)
    rng = jax.random.PRNGKey(seed)
    pvs = np.empty(N + 1, dtype=np.float64)
    pvs[0] = INITIAL_CASH
    bankrupt_at = -1
    # dithered-book proposal logits: bias the HOLD slot (831 make_policy_logits scheme c)
    logits = np.zeros((A, N_ACTIONS_PER_ASSET), dtype=np.float32)
    logits[:, 1] = proposal_bias
    logits_j = jnp.asarray(logits.reshape(1, A * N_ACTIONS_PER_ASSET))
    for k in range(N):
        rng, srng = jax.random.split(rng)
        cand = sampler._sample_action_vectors(srng, logits_j)[0]      # (K, A) int32
        Q = search_step(state, cand)                                  # (K,)
        best = int(jnp.argmax(Q))
        state, _, _, _ = jstep_j(state, cand[best])
        cur = get_current_prices(prices, state.current_step - 1, state.episode_start_idx)
        pv = float(compute_portfolio_value(state, cur))
        pvs[k + 1] = pv
        if bankrupt_at < 0 and pv <= INITIAL_CASH * cfg.liquidation_threshold_pct:
            bankrupt_at = k
    with np.errstate(divide="ignore", invalid="ignore"):
        rets = pvs[1:] / np.maximum(pvs[:-1], 1e-8) - 1.0
    rets = np.where(np.isfinite(rets), rets, 0.0)
    return {"rets": rets, "n": N, "final_pv": float(pvs[-1]), "bankrupt_at": bankrupt_at}


def e1_probe(close, logc, corridor, K, seed, discount, value_rollout_steps, max_decisions,
             proposal_bias):
    A = close.shape[1]
    tgt, pres, warm = corridor.book_target_signs(logc)
    position_size_pct = 0.5 / A                          # the narrow E1 regime (831 default)
    off = e1_rollout(close, K, seed, warm, corridor, FEE_BPS, SLIP_BPS, position_size_pct,
                     discount, value_rollout_steps, max_decisions, proposal_bias, budget_step=False)
    on = e1_rollout(close, K, seed, warm, corridor, FEE_BPS, SLIP_BPS, position_size_pct,
                    discount, value_rollout_steps, max_decisions, proposal_bias, budget_step=True)
    net_off = float(corridor._sharpe(off["rets"]))
    net_on = float(corridor._sharpe(on["rets"]))
    delta = net_on - net_off
    return {
        "K": int(K), "seed": int(seed), "n_assets": int(A), "n_decisions": int(off["n"]),
        "warm": int(warm), "position_size_pct": float(position_size_pct),
        "discount": float(discount), "value_rollout_steps": int(value_rollout_steps),
        "proposal_bias": float(proposal_bias), "cost_tier_bp": int(FEE_BPS + SLIP_BPS),
        "net21_off": net_off, "net21_on": net_on, "delta_on_minus_off": float(delta),
        "final_pv_off": off["final_pv"], "final_pv_on": on["final_pv"],
        "bankrupt_at_off": off["bankrupt_at"], "bankrupt_at_on": on["bankrupt_at"],
        "read": "ON >= OFF - 0.10",
        "verdict": "PASS" if net_on >= net_off - 0.10 else "FAIL",
    }


# =============================================================================
# driver
# =============================================================================

def _guard_out(path):
    """Refuse to write onto the protected source-of-record DB or the live-trading tree.

    Resolves symlinks with os.path.realpath BEFORE comparing (os.path.abspath does NOT
    resolve symlinks — an innocuously-named symlink targeting trading_data.db would slip past
    an abspath check and the write would follow it, truncating the 24GB source-of-record;
    invariant #2). Blocks the protected DB and anything under ~/trader-live/. Keeps a basename
    check on BOTH the raw and resolved path as belt-and-suspenders."""
    rp = os.path.realpath(path)
    protected_db = os.path.realpath(PROTECTED_DB)
    live_root = os.path.realpath(os.path.expanduser("~/trader-live"))
    if (rp == protected_db
            or "trading_data.db" in os.path.basename(rp)
            or "trading_data.db" in os.path.basename(path)):
        raise SystemExit(f"refusing to write to the protected source-of-record DB "
                         f"(resolved: {rp})")
    if rp == live_root or rp.startswith(live_root + os.sep):
        raise SystemExit(f"refusing to write under the protected live-trading tree "
                         f"(resolved: {rp})")
    return rp


def main():
    p = argparse.ArgumentParser(description="TASK_839 budget-step acceptance probe")
    p.add_argument("--db", default=FIXTURE_DB)
    # daily panel window (831) — shared by (i) survival walk and (ii) E1 probe
    p.add_argument("--start", default="2021-01-01")
    p.add_argument("--end", default="2025-09-01")
    # (i) survival walk
    p.add_argument("--walk-n-envs", type=int, default=4)
    p.add_argument("--walk-steps", type=int, default=70)     # >= 64 required by the read
    p.add_argument("--walk-seed", type=int, default=0)
    # factory-threading check (1-min fixture window; 834 slice)
    p.add_argument("--ft-start", default="2025-09-14")
    p.add_argument("--ft-end", default="2025-12-11")
    p.add_argument("--ft-bar-minutes", type=int, default=60)
    # (ii) E1 probe
    p.add_argument("--e1-k", type=int, default=16)
    p.add_argument("--e1-seed", type=int, default=0)
    p.add_argument("--e1-discount", type=float, default=0.997)
    p.add_argument("--e1-value-rollout-steps", type=int, default=3)
    p.add_argument("--e1-max-decisions", type=int, default=30)
    p.add_argument("--e1-proposal-bias", type=float, default=3.0)
    p.add_argument("--out", default=None)
    p.add_argument("--skip-e1", action="store_true", help="run only (i) survival + threading")
    p.add_argument("--force-fail-gate", action="store_true",
                   help="self-test hook: force the fail-closed gate to FAIL (exit 1) after "
                        "writing the JSON, WITHOUT altering any measurement — verifies the "
                        "nonzero-exit path")
    args = p.parse_args()

    is_fixture = os.path.abspath(args.db) == os.path.abspath(FIXTURE_DB)
    out = args.out
    if out is None:
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_fixture_smoke.json")
    out = _guard_out(out)

    t0 = time.time()
    print("=" * 84)
    print("TASK_839 ACCEPTANCE PROBE — budget-constrained vector step (env-wall fix)")
    print("=" * 84)

    corridor = _load_corridor()
    day_ts, close, logc, syms, info = corridor.load_panel(args.db, args.start, args.end)
    A = close.shape[1]
    print(f"daily panel: {A} assets, {close.shape[0]} rows ({args.start}..{args.end})")

    # ---- factory threading (end-to-end flag propagation) ----
    ft = factory_threading_check(args.db, args.ft_start, args.ft_end, args.ft_bar_minutes)
    print(f"[ft] create_jax_env_historical propagates --vector-budget-step -> "
          f"{'YES' if ft['propagates'] else 'NO'}")

    # ---- (i) survival walk ----
    print(f"[i] survival walk: {args.walk_n_envs} envs x {args.walk_steps} steps, {A} assets, "
          f"pct=0.1, masked (max_pos=16), zero-logit dithered-book hb0.0")
    walk_off = survival_walk(close, False, args.walk_n_envs, args.walk_steps, args.walk_seed)
    walk_on = survival_walk(close, True, args.walk_n_envs, args.walk_steps, args.walk_seed)
    print(f"    OFF: open_frac@0={walk_off['first_step_open_frac']:.3f}  "
          f"bankrupt_step/env={walk_off['bankrupt_step_per_env']}  "
          f"survived={walk_off['n_survived']}/{walk_off['n_envs']}  "
          f"overdrafts={walk_off['step_level_overdraft_events']}")
    print(f"    ON : open_frac@0={walk_on['first_step_open_frac']:.3f}  "
          f"bankrupt_step/env={walk_on['bankrupt_step_per_env']}  "
          f"survived={walk_on['n_survived']}/{walk_on['n_envs']}  "
          f"overdrafts={walk_on['step_level_overdraft_events']}")
    survival_pass = (walk_on["n_bankrupt"] == 0 and walk_on["n_walk"] >= 64)
    control_reproduced = (walk_off["n_bankrupt"] == walk_off["n_envs"] and
                          all(0 <= s <= 1 for s in walk_off["bankrupt_step_per_env"]))
    print(f"    (i) ON survives >=64 steps, zero bankruptcies -> {'PASS' if survival_pass else 'FAIL'}; "
          f"OFF env-wall reproduced (all bankrupt at step 0) -> {'YES' if control_reproduced else 'NO'}; "
          f"ON step-level overdrafts -> {walk_on['step_level_overdraft_events']}")

    # NIT: store db as a repo-relative path so the committed artifact is host-portable.
    try:
        db_rel = os.path.relpath(os.path.realpath(args.db), _REPO)
    except ValueError:
        db_rel = args.db
    db_field = db_rel if not db_rel.startswith("..") else args.db

    result = {
        "task": "TASK_839 budget-constrained vector step (env-wall fix)",
        "mode": "fixture-smoke" if is_fixture else "custom",
        "db": db_field, "window": [args.start, args.end], "n_assets": int(A),
        "d7": "RATIFIED option (a): PV*min(pct, budget_pct/n_open_trades)",
        "ledger": "828s3/env-wall",
        "factory_threading": ft,
        "survival_walk": {
            "off": walk_off, "on": walk_on,
            "on_survives_ge64_zero_bankrupt": bool(survival_pass),
            "off_env_wall_reproduced_step0": bool(control_reproduced),
        },
        "jax_version": jax.__version__,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # ---- (ii) E1 net@21 ON vs OFF ----
    if not args.skip_e1:
        print(f"[ii] E1 perfect-model probe: K{args.e1_k}, dithered-book, net@21 "
              f"(max_dec={args.e1_max_decisions})")
        e1 = e1_probe(close, logc, corridor, args.e1_k, args.e1_seed, args.e1_discount,
                      args.e1_value_rollout_steps, args.e1_max_decisions, args.e1_proposal_bias)
        print(f"    net@21  OFF={e1['net21_off']:+.4f}  ON={e1['net21_on']:+.4f}  "
              f"delta(ON-OFF)={e1['delta_on_minus_off']:+.4f}  -> {e1['verdict']} "
              f"(read: {e1['read']}, pct=0.5/A={e1['position_size_pct']:.5f} => cap non-binding)")
        result["e1_net21"] = e1

    # Fail-closed gate: (i) ON must survive >=64 steps zero-bankruptcy AND (ii) E1 read must
    # pass (skipped => not gating). Evaluated AFTER the JSON is written so the full evidence is
    # always on disk; a nonzero exit then guards against a future ON-path regression silently
    # reporting PASS by exit status.
    e1_ok = args.skip_e1 or (result.get("e1_net21", {}).get("verdict") == "PASS")
    gate_ok = bool(survival_pass and e1_ok) and not args.force_fail_gate
    result["gate_ok"] = gate_ok
    result["gate"] = {
        "survival_pass": bool(survival_pass), "e1_pass": bool(e1_ok),
        "forced_fail_selftest": bool(args.force_fail_gate),
    }
    result["elapsed_sec"] = round(time.time() - t0, 1)

    with open(out, "w") as fh:
        json.dump(result, fh, indent=2, default=float)
    print("-" * 84)
    print(f"[839] wrote {out}  ({result['elapsed_sec']}s)  gate_ok={gate_ok}")
    print("=" * 84)
    if not gate_ok:
        raise SystemExit(1)
    return result


if __name__ == "__main__":
    main()
