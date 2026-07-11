#!/usr/bin/env python3
"""TASK_840 acceptance probe — target-weight rebalance step (the H6 env).

Three PRE-REGISTERED reads (SPEC.md), fixture / true-model only, CPU-only, never
trading_data.db. Cite: LEDGER 835/S4-p1b; DECISIONS D8 (RATIFIED option (c)).

(i)  834-repro SURVIVAL WALK over the FULL fixture daily panel (1600+ steps). Zero-logit
     dithered-book proposal (hold_bias=0.0 => near-uniform per-asset actions), legality-masked
     exactly as battery self-play (batch_compute_action_mask), on the DIVERSIFIED daily panel.
     Three arms:
       canon  (both flags OFF)   — control: bankrupts at step 0 (the env-wall);
       budget (839 ON)           — control: survives the step-0 wall but still dies ~step 104
                                    (sparse-book monotonic bleed);
       weight (840 ON)           — MUST survive the FULL panel with ZERO bankruptcies AND
                                    final_pv > 0 (bounded Σ|w|<=1 allocation => no bleed).

(ii) STICKY-SHORT probe. Constant Sell on 8 assets (Hold on the rest), 500 steps, single env.
     Control (canon) bankrupts (unbounded short bleed). Under weight-ON the short is re-trimmed
     every step to a bounded fraction of PV, so PV must stay > 50% of initial.

(iii) E1-pattern PERFECT-MODEL probe, net@21 (fee16/slip5), weight-ON. FLOOR = the 819 linear
     book replayed through the SAME weight-ON step (corridor.book_target_signs + _to_action_vec),
     so the comparison is apples-to-apples in weight semantics. Pre-registered read:
     net@21(E1) >= FLOOR + 0.436 (the perfect-model search must still express PROFITABLE books
     in the new action semantics). If it cannot, that is a FINDING recorded in the JSON with
     verdict="FINDING" — NOT a silent pass.

Run:  JAX_PLATFORMS=cpu python experiments/840_weight_rebalance/acceptance_probe.py --out <path>
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
E1_EDGE = 0.436                          # pre-registered net@21 edge over the floor


def _load_corridor():
    """Import TASK_830's corridor module by path (dir name is not an identifier)."""
    path = os.path.join(_REPO, "experiments", "830_corridor", "corridor.py")
    spec = importlib.util.spec_from_file_location("corridor830_840", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _env_cfg(A, fee_bps, slip_bps, pct, *, weight_on, budget_on=False, window=5,
             w_unit=-1.0, w_max=4.0, deadband=-1.0, max_steps=400):
    return EnvConfig(
        n_assets=A, initial_cash=INITIAL_CASH, fee_bps=fee_bps, slippage_bps=slip_bps,
        window_size=window, position_size_pct=pct, allow_short=True, tp_sl_enabled=False,
        max_steps=max_steps, vector_action=True,
        vector_budget_step=budget_on, vector_budget_pct=0.99,
        vector_weight_rebalance=weight_on, vector_weight_unit=w_unit,
        vector_weight_max=w_max, vector_rebalance_deadband=deadband,
    )


# =============================================================================
# factory-threading check — the flag reaches EnvConfig through create_jax_env_historical
# =============================================================================

def factory_threading_check(db, start, end, bar_minutes):
    out = {}
    for flag in (False, True):
        _, _, env_cfg = create_jax_env_historical(
            db_path=db, start_date=start, end_date=end, n_envs=1, n_assets=8,
            fee_bps=0.0, slippage_bps=0.0, max_episode_steps=32, bar_minutes=bar_minutes,
            vector_action=True, vector_weight_rebalance=flag, vector_weight_unit=0.05,
            rng_key=jax.random.PRNGKey(0),
        )
        out[f"weight_{flag}"] = {
            "env_vector_weight_rebalance": bool(env_cfg.vector_weight_rebalance),
            "env_vector_weight_unit": float(env_cfg.vector_weight_unit),
        }
    ok = (out["weight_False"]["env_vector_weight_rebalance"] is False and
          out["weight_True"]["env_vector_weight_rebalance"] is True and
          abs(out["weight_True"]["env_vector_weight_unit"] - 0.05) < 1e-6)
    out["propagates"] = bool(ok)
    return out


# =============================================================================
# (i) survival walk — canon / budget / weight arms over the FULL panel
# =============================================================================

def survival_walk(close_panel, mode, n_envs, n_walk, seed,
                  position_size_pct=0.1, max_positions=16, min_cash_threshold=100.0, window=5):
    """Near-uniform (zero-logit dithered-book, hb0.0) walk, legality-masked as battery self-play.
    `mode` in {canon, budget, weight}. Returns per-env bankruptcy step + final PV + survivors."""
    T, A = close_panel.shape
    prices = jnp.asarray(close_panel, dtype=jnp.float32)
    cfg = _env_cfg(A, 0.0, 0.0, position_size_pct,
                   weight_on=(mode == "weight"), budget_on=(mode == "budget"),
                   window=window, max_steps=n_walk + 5)
    keys = jax.random.split(jax.random.PRNGKey(seed), n_envs)
    states = batch_reset(keys, prices, cfg)
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

    @jax.jit
    def walk_step(states, key):
        dither_key, sample_key = jax.random.split(key)
        price_idx = states.episode_start_idx + states.current_step
        current_prices = prices[price_idx]
        mask = batch_compute_action_mask(states, current_prices, cfg,
                                         max_positions, min_cash_threshold)
        proposal = sampler._build_proposal(zero_logits, mask, None, dither_key)
        avs = jax.random.categorical(sample_key, proposal, axis=-1).astype(jnp.int32)
        new_states, _, _, dones = batch_step_vector(states, avs, prices, cfg)
        # POST-step invariant audit (per env): cash and Σ|w| = Σ|size·price|/PV. cfg.fee_bps=0
        # here, so the ON guarantee is HARD (cash>=0, Σ|w|<=1). Measured on LIVE envs only.
        cur2 = prices[new_states.episode_start_idx + new_states.current_step - 1]
        pv2 = jax.vmap(compute_portfolio_value)(new_states, cur2)
        notion = jnp.abs(new_states.pos_size * cur2)
        sumw = jnp.sum(notion, axis=-1) / jnp.maximum(pv2, 1e-9)
        return new_states, dones, avs, new_states.cash, sumw

    base = jax.random.PRNGKey(seed + 777)
    bankrupt_step = np.full(n_envs, -1, dtype=np.int64)
    done_ever = np.zeros(n_envs, dtype=bool)
    first_open_frac = None
    min_cash = np.inf          # min post-step cash over LIVE envs (invariant: >=0 for weight)
    max_sumw = 0.0             # max post-step Σ|w| over LIVE envs (invariant: <=1 for weight)
    for t in range(n_walk):
        states, dones, avs, cash_t, sumw_t = walk_step(states, jax.random.fold_in(base, t))
        d = np.asarray(dones).astype(bool)
        live = ~done_ever      # envs that were alive going INTO this step
        newly = d & (~done_ever)
        bankrupt_step[newly] = t
        cash_np = np.asarray(cash_t)[live]
        sumw_np = np.asarray(sumw_t)[live]
        if cash_np.size:
            min_cash = min(min_cash, float(cash_np.min()))
            max_sumw = max(max_sumw, float(sumw_np.max()))
        done_ever |= d
        if t == 0:
            av0 = np.asarray(avs)
            first_open_frac = float(np.mean((av0 == 2) | (av0 == 0)))
    jax.block_until_ready(states.cash)
    cur = get_current_prices(prices, states.current_step - 1, states.episode_start_idx)
    final_pv = np.asarray(jax.vmap(compute_portfolio_value)(states, cur), dtype=np.float64)
    return {
        "mode": mode, "n_envs": int(n_envs), "n_walk": int(n_walk), "n_assets": int(A),
        "position_size_pct": float(position_size_pct), "max_positions": int(max_positions),
        "bankrupt_step_per_env": [int(x) for x in bankrupt_step],
        "n_survived": int((~done_ever).sum()), "n_bankrupt": int(done_ever.sum()),
        "final_pv_per_env": [float(x) for x in final_pv],
        "min_final_pv": float(final_pv.min()), "first_step_open_frac": first_open_frac,
        "min_step_cash": (None if not np.isfinite(min_cash) else float(min_cash)),
        "max_step_sum_abs_w": float(max_sumw),
    }


# =============================================================================
# (ii) sticky-short probe — constant Sell on 8 assets
# =============================================================================

def sticky_short(close_panel, mode, n_short, n_steps, position_size_pct=0.1, window=5):
    """Constant Sell on the first `n_short` assets (Hold elsewhere), single env, `n_steps`."""
    T, A = close_panel.shape
    n_short = min(n_short, A)
    prices = jnp.asarray(close_panel, dtype=jnp.float32)
    n_steps = min(n_steps, T - window - 2)
    cfg = _env_cfg(A, 0.0, 0.0, position_size_pct,
                   weight_on=(mode == "weight"), budget_on=(mode == "budget"),
                   window=window, max_steps=n_steps + 5)
    state = reset(jax.random.PRNGKey(0), prices, cfg).replace(
        episode_start_idx=jnp.array(window), current_step=jnp.array(0),
        max_episode_steps=jnp.array(n_steps + 2),
        initial_portfolio_value=jnp.array(INITIAL_CASH),
        prev_portfolio_value=jnp.array(INITIAL_CASH), cash=jnp.array(INITIAL_CASH),
    )
    av = np.ones(A, dtype=np.int32)
    av[:n_short] = 0                      # Sell (open/step short)
    av = jnp.asarray(av)
    jstep = jax.jit(partial(step_vector, prices=prices, config=cfg))
    bankrupt_at = -1
    min_pv = INITIAL_CASH
    for k in range(n_steps):
        state, _, _, done = jstep(state, av)
        cur = get_current_prices(prices, state.current_step - 1, state.episode_start_idx)
        pv = float(compute_portfolio_value(state, cur))
        min_pv = min(min_pv, pv)
        if bankrupt_at < 0 and (bool(done) or pv <= INITIAL_CASH * cfg.liquidation_threshold_pct):
            bankrupt_at = k
            break
    jax.block_until_ready(state.cash)
    cur = get_current_prices(prices, state.current_step - 1, state.episode_start_idx)
    final_pv = float(compute_portfolio_value(state, cur))
    return {
        "mode": mode, "n_short": int(n_short), "n_steps": int(n_steps),
        "final_pv": final_pv, "min_pv": float(min_pv),
        "final_pv_frac": float(final_pv / INITIAL_CASH), "bankrupt_at": bankrupt_at,
    }


# =============================================================================
# (iii) E1 perfect-model probe + 819-book FLOOR, both under weight-ON
# =============================================================================

def book_floor(close_panel, logc, corridor, fee_bps, slip_bps, pct, weight_on,
               max_decisions, window=5):
    """Replay the 819 linear book's target signs through step_vector (weight-ON) => the FLOOR."""
    T, A = close_panel.shape
    prices = jnp.asarray(close_panel, dtype=jnp.float32)
    tgt, pres, warm = corridor.book_target_signs(logc)
    cfg = _env_cfg(A, fee_bps, slip_bps, pct, weight_on=weight_on, window=window)
    N = T - 1 - warm
    if max_decisions is not None:
        N = min(N, int(max_decisions))
    state = reset(jax.random.PRNGKey(0), prices, cfg).replace(
        episode_start_idx=jnp.array(warm), current_step=jnp.array(0),
        max_episode_steps=jnp.array(N + 2),
        initial_portfolio_value=jnp.array(INITIAL_CASH),
        prev_portfolio_value=jnp.array(INITIAL_CASH),
    )
    jstep = jax.jit(partial(step_vector, prices=prices, config=cfg))
    pvs = np.empty(N + 1, dtype=np.float64)
    pvs[0] = INITIAL_CASH
    bankrupt_at = -1
    for k in range(N):
        side = np.asarray(state.pos_side)
        act = corridor._to_action_vec(tgt[warm + k], side, A)
        state, _, _, _ = jstep(state, jnp.asarray(act))
        cur = get_current_prices(prices, state.current_step - 1, state.episode_start_idx)
        pv = float(compute_portfolio_value(state, cur))
        pvs[k + 1] = pv
        if bankrupt_at < 0 and pv <= INITIAL_CASH * cfg.liquidation_threshold_pct:
            bankrupt_at = k
    with np.errstate(divide="ignore", invalid="ignore"):
        rets = pvs[1:] / np.maximum(pvs[:-1], 1e-8) - 1.0
    rets = np.where(np.isfinite(rets), rets, 0.0)
    return {"rets": rets, "n": N, "final_pv": float(pvs[-1]), "bankrupt_at": bankrupt_at,
            "net21": float(corridor._sharpe(rets))}


def e1_rollout(close_panel, K, seed, warm, fee_bps, slip_bps, pct, discount,
               value_rollout_steps, max_decisions, proposal_bias, weight_on, window=5):
    """Perfect-model K-sample search rollout (dithered-book), 831 pattern, weight-ON env."""
    T, A = close_panel.shape
    prices = jnp.asarray(close_panel, dtype=jnp.float32)
    cfg = _env_cfg(A, fee_bps, slip_bps, pct, weight_on=weight_on, window=window)
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
    logits = np.zeros((A, N_ACTIONS_PER_ASSET), dtype=np.float32)
    logits[:, 1] = proposal_bias
    logits_j = jnp.asarray(logits.reshape(1, A * N_ACTIONS_PER_ASSET))
    for k in range(N):
        rng, srng = jax.random.split(rng)
        cand = sampler._sample_action_vectors(srng, logits_j)[0]
        Q = search_step(state, cand)
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
    _, _, warm = corridor.book_target_signs(logc)
    pct = 0.5 / A                         # narrow E1 regime (831 default)
    floor = book_floor(close, logc, corridor, FEE_BPS, SLIP_BPS, pct, weight_on=True,
                       max_decisions=max_decisions)
    e1 = e1_rollout(close, K, seed, warm, FEE_BPS, SLIP_BPS, pct, discount,
                    value_rollout_steps, max_decisions, proposal_bias, weight_on=True)
    net_floor = float(floor["net21"])
    net_e1 = float(corridor._sharpe(e1["rets"]))
    edge = net_e1 - net_floor
    passed = net_e1 >= net_floor + E1_EDGE
    return {
        "K": int(K), "seed": int(seed), "n_assets": int(A), "n_decisions": int(e1["n"]),
        "warm": int(warm), "position_size_pct": float(pct), "discount": float(discount),
        "value_rollout_steps": int(value_rollout_steps), "proposal_bias": float(proposal_bias),
        "cost_tier_bp": int(FEE_BPS + SLIP_BPS),
        "net21_floor_819book_weighton": net_floor, "net21_e1_weighton": net_e1,
        "edge_e1_minus_floor": float(edge), "required_edge": E1_EDGE,
        "floor_final_pv": floor["final_pv"], "e1_final_pv": e1["final_pv"],
        "floor_bankrupt_at": floor["bankrupt_at"], "e1_bankrupt_at": e1["bankrupt_at"],
        "read": f"net21(E1) >= floor + {E1_EDGE}",
        "verdict": "PASS" if passed else "FINDING",
        "finding": (None if passed else
                    "E1 perfect-model search did not clear floor+0.436 in weight semantics "
                    "(fixture-smoke; recorded per SPEC, not a silent pass)."),
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
    p = argparse.ArgumentParser(description="TASK_840 weight-rebalance acceptance probe")
    p.add_argument("--db", default=FIXTURE_DB)
    p.add_argument("--start", default="2021-01-01")
    p.add_argument("--end", default="2025-09-01")
    # (i) survival walk — full panel by default
    p.add_argument("--walk-n-envs", type=int, default=2)
    p.add_argument("--walk-steps", type=int, default=0, help="0 => full panel (T-window-2)")
    p.add_argument("--walk-seed", type=int, default=0)
    # factory-threading check (1-min fixture window; 834 slice)
    p.add_argument("--ft-start", default="2025-09-14")
    p.add_argument("--ft-end", default="2025-12-11")
    p.add_argument("--ft-bar-minutes", type=int, default=60)
    # (ii) sticky short
    p.add_argument("--short-assets", type=int, default=8)
    p.add_argument("--short-steps", type=int, default=500)
    # (iii) E1 probe
    p.add_argument("--e1-k", type=int, default=16)
    p.add_argument("--e1-seed", type=int, default=0)
    p.add_argument("--e1-discount", type=float, default=0.997)
    p.add_argument("--e1-value-rollout-steps", type=int, default=3)
    p.add_argument("--e1-max-decisions", type=int, default=60)
    p.add_argument("--e1-proposal-bias", type=float, default=3.0)
    p.add_argument("--out", default=None)
    p.add_argument("--skip-e1", action="store_true")
    p.add_argument("--force-fail-gate", action="store_true",
                   help="self-test hook: force the fail-closed gate to FAIL (exit 1) after "
                        "writing the JSON, WITHOUT altering any measurement")
    args = p.parse_args()

    is_fixture = os.path.abspath(args.db) == os.path.abspath(FIXTURE_DB)
    out = args.out
    if out is None:
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_fixture_smoke.json")
    out = _guard_out(out)

    t0 = time.time()
    print("=" * 84)
    print("TASK_840 ACCEPTANCE PROBE — target-weight rebalance step (the H6 env)")
    print("=" * 84)

    corridor = _load_corridor()
    day_ts, close, logc, syms, info = corridor.load_panel(args.db, args.start, args.end)
    A, T = close.shape[1], close.shape[0]
    print(f"daily panel: {A} assets, {T} rows ({args.start}..{args.end})")

    n_walk = args.walk_steps if args.walk_steps > 0 else (T - 5 - 2)

    ft = factory_threading_check(args.db, args.ft_start, args.ft_end, args.ft_bar_minutes)
    print(f"[ft] create_jax_env_historical propagates --vector-weight-rebalance -> "
          f"{'YES' if ft['propagates'] else 'NO'}")

    # ---- (i) survival walk — three arms ----
    print(f"[i] survival walk: {args.walk_n_envs} envs x {n_walk} steps (FULL panel), {A} assets")
    walk = {}
    for mode in ("canon", "budget", "weight"):
        walk[mode] = survival_walk(close, mode, args.walk_n_envs, n_walk, args.walk_seed)
        w = walk[mode]
        print(f"    {mode:6s}: bankrupt_step/env={w['bankrupt_step_per_env']}  "
              f"survived={w['n_survived']}/{w['n_envs']}  min_final_pv={w['min_final_pv']:.1f}")
    # HARD invariant audit (fee=0 walk): the defining ON guarantee must hold EVERY step, not
    # merely "no bankruptcy at the end" — cash >= 0 (margin path never fires) AND Σ|w| <= 1
    # (no leverage). The panel flagged that the old gate never checked the invariant it claimed.
    W = walk["weight"]
    CASH_TOL, SUMW_TOL = 1e-2, 1.05   # exact >=0 / <=1 at fee0; tiny tol for float + deadband
    inv_cash_ok = (W["min_step_cash"] is not None) and (W["min_step_cash"] >= -CASH_TOL)
    inv_sumw_ok = W["max_step_sum_abs_w"] <= SUMW_TOL
    invariant_pass = inv_cash_ok and inv_sumw_ok
    survival_pass = (W["n_bankrupt"] == 0 and n_walk >= 1600 and W["min_final_pv"] > 0.0
                     and invariant_pass)
    print(f"    (i) weight-ON survives FULL panel (>=1600), zero bankruptcies, final_pv>0 -> "
          f"{'PASS' if survival_pass else 'FAIL'}")
    print(f"        invariant: min_step_cash={W['min_step_cash']}  "
          f"max_step_Σ|w|={W['max_step_sum_abs_w']:.4f}  "
          f"cash>=0 & Σ|w|<=1 (margin no-op) -> {'PASS' if invariant_pass else 'FAIL'}")

    # ---- (ii) sticky short ----
    print(f"[ii] sticky short: constant Sell on {args.short_assets} assets, {args.short_steps} steps")
    short_canon = sticky_short(close, "canon", args.short_assets, args.short_steps)
    short_weight = sticky_short(close, "weight", args.short_assets, args.short_steps)
    print(f"    canon : final_pv_frac={short_canon['final_pv_frac']:.3f}  "
          f"bankrupt_at={short_canon['bankrupt_at']}")
    print(f"    weight: final_pv_frac={short_weight['final_pv_frac']:.3f}  "
          f"bankrupt_at={short_weight['bankrupt_at']}")
    sticky_pass = (short_weight["bankrupt_at"] < 0 and short_weight["final_pv_frac"] > 0.5)
    print(f"    (ii) weight-ON keeps PV>50% for {args.short_steps} steps -> "
          f"{'PASS' if sticky_pass else 'FAIL'}")

    try:
        db_rel = os.path.relpath(os.path.realpath(args.db), _REPO)
    except ValueError:
        db_rel = args.db
    db_field = db_rel if not db_rel.startswith("..") else args.db

    result = {
        "task": "TASK_840 target-weight rebalance step (the H6 env)",
        "mode": "fixture-smoke" if is_fixture else "custom",
        "db": db_field, "window": [args.start, args.end], "n_assets": int(A), "n_rows": int(T),
        "d8": "RATIFIED option (c): per-asset action -> target signed weight, re-trim to CURRENT PV",
        "ledger": "835/S4-p1b",
        "weight_flags": {                       # the four flags the ON run used (self-description)
            "vector_weight_rebalance": True,
            "vector_weight_unit": -1.0,          # sentinel => 1/n_assets
            "vector_weight_max": 4.0,
            "vector_rebalance_deadband": -1.0,   # sentinel => 0.1*w_unit
        },
        "factory_threading": ft,
        "survival_walk": {
            "canon": walk["canon"], "budget": walk["budget"], "weight": walk["weight"],
            "weight_survives_full_panel_zero_bankrupt": bool(survival_pass),
            "invariant_cash_ge0_sumw_le1_margin_noop": bool(invariant_pass),
        },
        "sticky_short": {
            "canon": short_canon, "weight": short_weight,
            "weight_keeps_pv_gt_50pct": bool(sticky_pass),
        },
        "jax_version": jax.__version__,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # ---- (iii) E1 net@21 weight-ON vs floor ----
    if not args.skip_e1:
        print(f"[iii] E1 perfect-model probe: K{args.e1_k}, dithered-book, net@21, weight-ON "
              f"(max_dec={args.e1_max_decisions})")
        e1 = e1_probe(close, logc, corridor, args.e1_k, args.e1_seed, args.e1_discount,
                      args.e1_value_rollout_steps, args.e1_max_decisions, args.e1_proposal_bias)
        print(f"    floor(819 book)={e1['net21_floor_819book_weighton']:+.4f}  "
              f"E1={e1['net21_e1_weighton']:+.4f}  edge={e1['edge_e1_minus_floor']:+.4f}  "
              f"-> {e1['verdict']} (read: {e1['read']})")
        if e1["finding"]:
            print(f"    FINDING: {e1['finding']}")
        result["e1_net21"] = e1

    e1_pass = args.skip_e1 or (result.get("e1_net21", {}).get("verdict") == "PASS")
    # The load-bearing HARD gates are the bankruptcy-impossibility invariants (i) + (ii).
    # (iii) is recorded with its verdict (PASS / FINDING) but does NOT gate the exit code —
    # a below-edge E1 is a recorded FINDING per SPEC, not a silent pass and not a hard fail.
    gate_ok = bool(survival_pass and sticky_pass) and not args.force_fail_gate
    result["gate_ok"] = gate_ok
    result["gate"] = {
        "survival_pass": bool(survival_pass), "sticky_pass": bool(sticky_pass),
        "invariant_pass": bool(invariant_pass), "e1_pass": bool(e1_pass),
        "e1_is_hard_gate": False, "forced_fail_selftest": bool(args.force_fail_gate),
    }
    result["elapsed_sec"] = round(time.time() - t0, 1)

    with open(out, "w") as fh:
        json.dump(result, fh, indent=2, default=float)
    print("-" * 84)
    print(f"[840] wrote {out}  ({result['elapsed_sec']}s)  gate_ok={gate_ok}  "
          f"e1_verdict={result.get('e1_net21', {}).get('verdict', 'skipped')}")
    print("=" * 84)
    if not gate_ok:
        raise SystemExit(1)
    return result


if __name__ == "__main__":
    main()
