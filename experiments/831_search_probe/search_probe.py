#!/usr/bin/env python3
"""TASK_831 / E1+E2 — true-model search-capacity probe (pathway rung R1).

Runs the Stage-2 Sampled-MuZero *sampling* machinery with the ENV SIMULATOR as dynamics and the
TRUE env reward — NO learned nets — to answer H1 (search side): can a PERFECT-model K-sample search
FIND at least the 819 corridor floor? If not, no learned model will, and 828 Stage 3 must not launch.

Design (see SPEC.md — all decisions pre-registered there):
  * REUSE the canon sampler: imports ``SampledMuZeroVectorSearch`` and calls its
    ``_sample_action_vectors`` (per-asset factored draws) + ``_empirical_prior`` (β̂=nₐ/K) UNCHANGED.
  * REIMPLEMENT a minimal true-model search here (zero canon diffs): root planning over the K sampled
    candidate vectors, scoring each by Q = r₀ + γ·V, V = truncated true-env HOLD rollout; pick argmax.
    (Full K-candidate evaluation, STRONGER than mctx sequential-halving → a FAIL is a real capacity
    ceiling. FLAGGED for the panel.) Reimplementation is FORCED because the canon search requires a
    learned ``model_apply_fn`` and this probe passes ``model_apply_fn=None`` (there is no net). The
    canon categorical value/reward decode (401-bin support, ~0.05 resolution over [-10,10]) is also
    lossy for the ~0.1-magnitude symlog rewards used here — a second reason not to reuse it.
  * E2 schemes are injected purely as per-asset ``policy_logits`` fed to the canon sampler:
      - independent   : zeros            (uniform per-asset draws — untrained factored policy)
      - linear-prior  : bias on the 819 book's target action per asset (mixture toward the book)
      - dithered-book : bias on HOLD     (local moves around the held portfolio)

Reuses TASK_830's corridor module (panel loader, book replay = FLOOR, oracle = CEILING, scorecard)
by path so E1 is scored apples-to-apples on the identical rows. CPU / fixture-only in the factory;
the real-panel sweep is a SCHEDULED single-GPU one-shot (invariant #7). Never touches trading_data.db.
"""
import os
# CPU-only, deterministic — keep JAX off the GPU (scheduled-CPU / smoke job).
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import importlib.util
import json
import sys
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

# Run from anywhere: repo root on the path so `trader` / `mcts_trader` import cleanly.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from mcts_trader.env.jax_trading_env import (
    EnvConfig, reset, step_vector, get_current_prices, compute_portfolio_value,
)
from mcts_trader.jax_agent.sampled_search import SampledMuZeroVectorSearch

FIXTURE_DB = os.path.join(_REPO, "fixtures", "trading_fixture.db")
INITIAL_CASH = 100_000.0
FEE_BPS, SLIP_BPS = 16.0, 5.0
GATE_COSTS = (0, 21)          # bp/side cost tiers — gross + net@21, mirroring 830
N_ACTIONS_PER_ASSET = 5
HOLD = 1
SCHEMES = ("independent", "linear-prior", "dithered-book")
K_GRID = (8, 16, 32)


def _load_corridor():
    """Import the TASK_830 corridor module by path (dir name '830_corridor' is not an identifier)."""
    path = os.path.join(_REPO, "experiments", "830_corridor", "corridor.py")
    spec = importlib.util.spec_from_file_location("corridor830", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# =============================================================================
# E2 proposal schemes → per-asset policy_logits (the ONLY thing that differs per scheme)
# =============================================================================

def make_policy_logits(scheme, pos_side, tgt_row, corridor, A, bias):
    """Build (1, A*N_ACTIONS) factored policy logits for one decision under a proposal scheme.

    Layout matches the canon sampler: index = asset*N_ACTIONS + action, reshaped (A, N_ACTIONS);
    each asset row is an independent categorical marginal drawn K times i.i.d.
    """
    logits = np.zeros((A, N_ACTIONS_PER_ASSET), dtype=np.float32)
    if scheme == "independent":
        pass                                              # uniform per-asset draws
    elif scheme == "linear-prior":
        # target action = one greedy step toward the 819 book's target sign (reuses 830 mapping)
        tgt_act = corridor._to_action_vec(tgt_row, pos_side, A)   # (A,) in {0..4}
        logits[np.arange(A), tgt_act] = bias
    elif scheme == "dithered-book":
        logits[:, HOLD] = bias                            # mostly hold, occasional local flip
    else:
        raise ValueError(f"unknown scheme {scheme!r}")
    return jnp.asarray(logits.reshape(1, A * N_ACTIONS_PER_ASSET))


# =============================================================================
# true-model search (reimplemented here; reuses canon sampling + β̂)
# =============================================================================

def build_search_step(prices, cfg, discount, value_rollout_steps):
    """Return a jitted fn: (state, cand_vecs) -> (Q per candidate). cand_vecs: (K, A) int32.

    Q(k) = r₀(step_vector(state, vec_k)) + γ·Σ_{t=1..R} γ^{t-1} r_t  under an all-HOLD rollout from
    the post-action state. This is the perfect-model value MuZero would learn (truncated true return).
    """
    jstep = partial(step_vector, prices=prices, config=cfg)
    A = cfg.n_assets
    hold_vec = jnp.ones((A,), dtype=jnp.int32)

    def hold_step(carry_state, _):
        ns, _, r, _ = jstep(carry_state, hold_vec)
        return ns, r

    def eval_candidate(state, vec):
        ns, _, r0, _ = jstep(state, vec)
        if value_rollout_steps > 0:
            _, rs = jax.lax.scan(hold_step, ns, xs=None, length=value_rollout_steps)
            disc = discount ** jnp.arange(value_rollout_steps)     # γ^0..γ^{R-1}
            V = jnp.sum(disc * rs)
        else:
            V = jnp.array(0.0)
        return r0 + discount * V

    @jax.jit
    def search_step(state, cand_vecs):
        # broadcast the single real state across the K candidates (in_axes: state=None, vec=0)
        return jax.vmap(lambda v: eval_candidate(state, v), in_axes=0)(cand_vecs)

    return search_step


def run_rollout(close_panel, corridor, scheme, K, seed, warm, tgt,
                fee_bps, slip_bps, position_size_pct, discount, value_rollout_steps,
                max_decisions, proposal_bias, window=5):
    """Plan-then-act rollout: at each decision, K-sample search picks an action VECTOR; step the REAL
    env once. Returns per-decision portfolio returns (aligned warm..) + diagnostics."""
    T, A = close_panel.shape
    prices = jnp.asarray(close_panel, dtype=jnp.float32)
    cfg = EnvConfig(
        n_assets=A, initial_cash=INITIAL_CASH, fee_bps=fee_bps, slippage_bps=slip_bps,
        window_size=window, position_size_pct=position_size_pct,
        allow_short=True, tp_sl_enabled=False, max_steps=400, vector_action=True,
    )
    # leave head-room so the truncated value rollout never indexes past the panel end
    N = T - 1 - warm - value_rollout_steps
    if max_decisions is not None:
        N = min(N, int(max_decisions))
    if N < 2:
        raise RuntimeError(f"panel too short: N={N}")

    # canon sampler reused ONLY for its exact sampling + β̂ machinery (model_apply_fn unused here)
    sampler = SampledMuZeroVectorSearch(
        model_apply_fn=None, n_assets=A, n_actions_per_asset=N_ACTIONS_PER_ASSET, num_samples=K)

    state = reset(jax.random.PRNGKey(0), prices, cfg).replace(
        episode_start_idx=jnp.array(warm), current_step=jnp.array(0),
        max_episode_steps=jnp.array(N + value_rollout_steps + 2),
        initial_portfolio_value=jnp.array(INITIAL_CASH),
        prev_portfolio_value=jnp.array(INITIAL_CASH),
    )
    search_step = build_search_step(prices, cfg, discount, value_rollout_steps)
    jstep = jax.jit(partial(step_vector, prices=prices, config=cfg))

    rng = jax.random.PRNGKey(seed)
    pvs = np.empty(N + 1, dtype=np.float64)
    pvs[0] = INITIAL_CASH
    n_long = n_short = 0
    bankrupt_at = -1
    for k in range(N):
        side = np.asarray(state.pos_side)
        logits = make_policy_logits(scheme, side, tgt[warm + k], corridor, A, proposal_bias)
        rng, srng = jax.random.split(rng)
        cand = sampler._sample_action_vectors(srng, logits)[0]     # (K, A) int32
        Q = search_step(state, cand)                               # (K,)
        best = int(jnp.argmax(Q))
        chosen = cand[best]
        state, _, _, _ = jstep(state, chosen)
        cur = get_current_prices(prices, state.current_step - 1, state.episode_start_idx)
        pv = float(compute_portfolio_value(state, cur))
        pvs[k + 1] = pv
        sd = np.asarray(state.pos_side)
        n_long += int((sd == 1).sum()); n_short += int((sd == 2).sum())
        if bankrupt_at < 0 and pv <= INITIAL_CASH * cfg.liquidation_threshold_pct:
            bankrupt_at = k
    with np.errstate(divide="ignore", invalid="ignore"):
        rets = pvs[1:] / np.maximum(pvs[:-1], 1e-8) - 1.0
    rets = np.where(np.isfinite(rets), rets, 0.0)
    return {
        "rets": rets, "n": N, "final_pv": float(pvs[-1]),
        "avg_long": n_long / max(N, 1), "avg_short": n_short / max(N, 1),
        "bankrupt_at": bankrupt_at,
    }


# =============================================================================
# driver
# =============================================================================

def run(db_path, start, end, out_path, tag, args):
    corridor = _load_corridor()
    day_ts, close, logc, syms, info = corridor.load_panel(db_path, start, end)
    A = close.shape[1]
    position_size_pct = args.position_size_pct if args.position_size_pct is not None else 0.5 / A
    tgt, pres, warm = corridor.book_target_signs(logc)

    # decision horizon for the smoke (shared by floor/ceiling so E1 is apples-to-apples)
    T = close.shape[0]
    N = T - 1 - warm - args.value_rollout_steps
    if args.max_decisions is not None:
        N = min(N, int(args.max_decisions))

    # ---- FLOOR + CEILING recomputed on the SAME rows (reuse 830) ----
    floor = {}
    for c in GATE_COSTS:
        fee, slip = (0.0, 0.0) if c == 0 else (FEE_BPS, SLIP_BPS)
        rep = corridor.replay_floor(close, tgt, warm, fee, slip, position_size_pct)
        floor[c] = corridor._sharpe(rep["rets"][:N])          # same rows as the search rollout
    pos_orc, orc_alpha = corridor.oracle_positions(logc, pres)
    # Window-consistent ceiling: score the oracle on the SAME first-N decision rows as the
    # floor/search (decision rows warm..warm+N-1 touch panel rows up to warm+N). Otherwise the
    # reported [floor, ceiling] corridor mixes windows on any capped/smoke run — floor/search on
    # the first N rows, ceiling on the full ~1673-row panel. On the full scheduled run
    # (--max-decisions None) this is a no-op (the whole panel is scored).
    if args.max_decisions is None:
        end_row = logc.shape[0]
    else:
        end_row = min(warm + N + 1, logc.shape[0])
    orc_cell = corridor.backtest.cell_metrics(
        logc[:end_row], pos_orc[:end_row], pres[:end_row], day_ts[:end_row],
        corridor.config.STRATEGY.H_rebal, warm, scfg=corridor.config.STRATEGY)
    ceiling = {c: (orc_cell["reg"]["FULL"]["gross"] if c == 0
                   else orc_cell["reg"]["FULL"].get(f"net{c}")) for c in GATE_COSTS}

    # ---- E2 grid: scheme × K × seed ----
    seeds = [int(s) for s in args.seeds.split(",") if s != ""]
    grid = {}
    for scheme in (args.schemes.split(",") if args.schemes else SCHEMES):
        for K in (args.k_grid or K_GRID):
            per_seed = {c: [] for c in GATE_COSTS}
            diag = None
            for sd in seeds:
                rep = run_rollout(
                    close, corridor, scheme, K, sd, warm, tgt, 0.0, 0.0, position_size_pct,
                    args.discount, args.value_rollout_steps, args.max_decisions, args.proposal_bias)
                # score the SAME realized return series at each cost tier via cost-adjusted replay
                for c in GATE_COSTS:
                    fee, slip = (0.0, 0.0) if c == 0 else (FEE_BPS, SLIP_BPS)
                    if c == 0:
                        per_seed[c].append(corridor._sharpe(rep["rets"]))
                    else:
                        repc = run_rollout(
                            close, corridor, scheme, K, sd, warm, tgt, fee, slip, position_size_pct,
                            args.discount, args.value_rollout_steps, args.max_decisions,
                            args.proposal_bias)
                        per_seed[c].append(corridor._sharpe(repc["rets"]))
                diag = {"final_pv": rep["final_pv"], "avg_long": rep["avg_long"],
                        "avg_short": rep["avg_short"], "bankrupt_at": rep["bankrupt_at"], "n": rep["n"]}
            grid[f"{scheme}|K{K}"] = {
                "scheme": scheme, "K": K, "n_evals_per_decision": K * (1 + args.value_rollout_steps),
                "sharpe_mean": {c: float(np.mean(per_seed[c])) for c in GATE_COSTS},
                "sharpe_seeds": {c: [float(x) for x in per_seed[c]] for c in GATE_COSTS},
                "diag": diag,
            }

    # ---- E1 verdict: any scheme×K reach the floor? (net@21, mean over seeds) ----
    net_floor = floor[21]
    reached = {name: g["sharpe_mean"][21] >= net_floor for name, g in grid.items()}
    e1_pass = any(reached.values())
    # cheapest reacher = min K, then min evals; tiebreak among equal-cost reachers = HIGHER
    # net@21 Sharpe (performance-aware; -sharpe so min() prefers the larger Sharpe), name last
    # purely for determinism.
    reachers = [(g["K"], g["n_evals_per_decision"], -g["sharpe_mean"][21], name)
                for name, g in grid.items() if reached[name]]
    cheapest = min(reachers)[3] if reachers else None

    result = {
        "tag": tag, "db": db_path, "window": [start, end], "n_assets": A, "n_days": int(day_ts.size),
        "warm": warm, "n_decisions": N, "position_size_pct": position_size_pct,
        "seeds": seeds, "discount": args.discount, "value_rollout_steps": args.value_rollout_steps,
        "proposal_bias": args.proposal_bias, "cost_tiers_bp": list(GATE_COSTS),
        "value_heuristic": f"truncated_env_rollout(R={args.value_rollout_steps}, HOLD, disc={args.discount})",
        "floor": {str(c): (float(v) if v is not None and np.isfinite(v) else None) for c, v in floor.items()},
        "ceiling": {str(c): (float(v) if v is not None and np.isfinite(v) else None) for c, v in ceiling.items()},
        "oracle_gross_alpha": float(orc_alpha),
        "grid": grid,
        "e1": {
            "floor_net21": float(net_floor) if np.isfinite(net_floor) else None,
            "reached_floor": reached,
            "verdict": "PASS" if e1_pass else "FAIL",
            "cheapest_reacher": cheapest,
            "smoke": True,
            "note": ("FIXTURE-SMOKE signal only — real E1 verdict PENDING the scheduled single-GPU "
                     "full-panel sweep. Kill logic: E1 FAIL across ALL schemes ⇒ proposal mechanism "
                     "is the defect ⇒ 828 Stage 3 does NOT launch."),
        },
    }
    _print_report(result)
    if out_path:
        with open(out_path, "w") as fh:
            json.dump(result, fh, indent=2, default=float)
        print(f"\n[831] wrote {out_path}")
    return result


def _fmt(x):
    return "   n/a" if x is None or not np.isfinite(x) else f"{x:+.3f}"


def _print_report(r):
    print("=" * 84)
    print(f"E1/E2 SEARCH-CAPACITY PROBE ({r['tag']}) — {r['window'][0]} .. {r['window'][1]}")
    print(f"panel: {r['n_assets']} assets, {r['n_decisions']} decisions, warm={r['warm']}, "
          f"pct={r['position_size_pct']:.5f}, seeds={r['seeds']}")
    print(f"value heuristic: {r['value_heuristic']}")
    fl, ce = r["floor"], r["ceiling"]
    print(f"corridor floor  net@21 {_fmt(fl.get('21'))}  gross {_fmt(fl.get('0'))}")
    print(f"corridor ceiling net@21 {_fmt(ce.get('21'))}  gross {_fmt(ce.get('0'))}  "
          f"(oracle gross alpha {r['oracle_gross_alpha']:+.2f})")
    print("-" * 84)
    print(f"{'scheme|K':>22} | {'net@21':>8} {'gross':>8} | {'>=floor?':>8} {'evals/dec':>10}")
    for name in sorted(r["grid"], key=lambda n: (r["grid"][n]["scheme"], r["grid"][n]["K"])):
        g = r["grid"][name]
        hit = "YES" if r["e1"]["reached_floor"][name] else "no"
        print(f"{name:>22} | {_fmt(g['sharpe_mean'][21]):>8} {_fmt(g['sharpe_mean'][0]):>8} | "
              f"{hit:>8} {g['n_evals_per_decision']:>10}")
    print("-" * 84)
    print(f"E1 VERDICT: {r['e1']['verdict']}  (floor net@21 {_fmt(r['e1']['floor_net21'])}; "
          f"cheapest reacher: {r['e1']['cheapest_reacher']})")
    print(f"  {r['e1']['note']}")
    print("=" * 84)


def main():
    ap = argparse.ArgumentParser(description="E1/E2 true-model search-capacity probe (TASK_831)")
    ap.add_argument("--db", default=FIXTURE_DB)
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default="2025-09-01")
    ap.add_argument("--position-size-pct", type=float, default=None)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--discount", type=float, default=0.997)
    ap.add_argument("--value-rollout-steps", type=int, default=3,
                    help="truncated true-env HOLD rollout depth for the search value heuristic")
    ap.add_argument("--proposal-bias", type=float, default=3.0,
                    help="logit bias toward the book/HOLD target action in E2 schemes b/c")
    ap.add_argument("--max-decisions", type=int, default=100,
                    help="cap decisions for the fixture smoke (None => full panel, scheduled run)")
    ap.add_argument("--schemes", default=None, help="comma list; default = all three")
    ap.add_argument("--k-grid", default=None, help="comma list of K; default = 8,16,32")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.k_grid:
        args.k_grid = tuple(int(x) for x in args.k_grid.split(","))
    if args.max_decisions is not None and args.max_decisions < 0:
        args.max_decisions = None
    is_fixture = os.path.abspath(args.db) == os.path.abspath(FIXTURE_DB)
    tag = args.tag or ("fixture-smoke" if is_fixture else "real-panel")
    out = args.out
    if out is None and is_fixture:
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_fixture_smoke.json")
    run(args.db, args.start, args.end, out, tag, args)


if __name__ == "__main__":
    main()
