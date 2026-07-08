#!/usr/bin/env python3
"""TASK_838 — acceptance probe: per-node resampling ON vs OFF in the CANON vector search.

Runs the CANON ``SampledMuZeroVectorSearch`` (the class TASK_838 modified) end-to-end in
BOTH arms — ``per_node_resampling=False`` (shipped root-reuse) vs ``True`` — at EQUAL K,
depth>1 sims, on the fixture panel, with a TRUE-MODEL apply_fn (adapted 831 pattern; see
SPEC.md for the pre-registered design and the honesty section on what this half proves).

Unlike 831 (which reimplemented a root-planning loop because canon `_search` needs a
learned model_apply_fn), this probe WRAPS a true model behind the exact apply_fn contract
the canon search uses, so the canon `_search` / mctx Gumbel machinery runs VERBATIM:

  * latent_state = the real ``TradingState`` pytree (mctx embeddings are pytrees; the
    canon search treats latents as opaque). Root observation = flat uint32-bitcast
    encoding of the state (exact codec, round-trip asserted at startup).
  * dynamics/reward = ``step_vector`` + the env's own reward.
  * value = truncated true-env all-HOLD rollout (R=3, gamma=0.997 — the 831 heuristic).
  * value/reward are TWO-HOT encoded over the canon 401-bin support so the canon
    categorical decode recovers the scalar near-exactly (no canon decode changes).
  * policy_logits = zeros everywhere (no learned policy); proposal shaping comes from the
    canon dithered-book knobs (hold_bias, dither_scale), identical in both arms.

CPU + fixture only; never trading_data.db; never GPU. Zero canon diffs.
"""
import os
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

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from mcts_trader.env.jax_trading_env import (
    EnvConfig, reset, step_vector, get_current_prices, compute_portfolio_value,
)
from mcts_trader.jax_agent import get_value_support
from mcts_trader.jax_agent.sampled_search import SampledMuZeroVectorSearch

FIXTURE_DB = os.path.join(_REPO, "fixtures", "trading_fixture.db")
INITIAL_CASH = 100_000.0
FEE_BPS, SLIP_BPS = 16.0, 5.0
GATE_COSTS = (0, 21)
N_ACTIONS_PER_ASSET = 5


def _load_corridor():
    """Import the TASK_830 corridor module by path (831 convention)."""
    path = os.path.join(_REPO, "experiments", "830_corridor", "corridor.py")
    spec = importlib.util.spec_from_file_location("corridor830", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# =============================================================================
# Exact state <-> flat-uint32 codec (root observation transport)
# =============================================================================

class StateCodec:
    """Exact round-trip between a TradingState pytree and a (1, L) uint32 array.

    float32/int32 leaves are BITCAST to uint32 (exact bit transport, no value
    arithmetic); bool leaves go through uint32 0/1; uint32 (rng_key) passes as-is.
    """

    def __init__(self, example_state):
        leaves, self.treedef = jax.tree_util.tree_flatten(example_state)
        self.shapes = [l.shape for l in leaves]
        self.dtypes = [l.dtype for l in leaves]
        self.sizes = [int(np.prod(s)) if len(s) else 1 for s in self.shapes]
        self.total = sum(self.sizes)

    def encode(self, state):
        parts = []
        for leaf in jax.tree_util.tree_leaves(state):
            a = leaf.reshape(-1)
            if a.dtype == jnp.bool_:
                parts.append(a.astype(jnp.uint32))
            elif a.dtype == jnp.uint32:
                parts.append(a)
            else:  # float32 / int32 — 32-bit bitcast is exact
                parts.append(jax.lax.bitcast_convert_type(a, jnp.uint32))
        return jnp.concatenate(parts)[None, :]                      # (1, L)

    def decode(self, flat):
        v = flat[0]
        out, off = [], 0
        for shape, dtype, size in zip(self.shapes, self.dtypes, self.sizes):
            seg = v[off:off + size]
            off += size
            if dtype == jnp.bool_:
                out.append((seg != 0).reshape(shape))
            elif dtype == jnp.uint32:
                out.append(seg.reshape(shape))
            else:
                out.append(jax.lax.bitcast_convert_type(seg, dtype).reshape(shape))
        return jax.tree_util.tree_unflatten(self.treedef, out)


def _assert_roundtrip(codec, state):
    rt = codec.decode(codec.encode(state))
    for a, b in zip(jax.tree_util.tree_leaves(state), jax.tree_util.tree_leaves(rt)):
        assert a.dtype == b.dtype and a.shape == b.shape, (a.dtype, b.dtype)
        assert bool(jnp.all(a == b)), "codec round-trip mismatch"


# =============================================================================
# Two-hot encoding over the canon 401-bin support (near-exact linear decode)
# =============================================================================

def two_hot_logits(x):
    """x: (B,) scalar -> (B, 401) logits whose canon categorical decode ~= x."""
    support = get_value_support()                         # (401,) linspace(-10, 10)
    n = support.shape[0]
    step = (support[-1] - support[0]) / (n - 1)
    xc = jnp.clip(x, support[0], support[-1])
    pos = (xc - support[0]) / step                        # float bin position
    lo = jnp.clip(jnp.floor(pos).astype(jnp.int32), 0, n - 1)
    hi = jnp.clip(lo + 1, 0, n - 1)
    w_hi = pos - lo.astype(pos.dtype)
    w_lo = 1.0 - w_hi
    B = x.shape[0]
    probs = jnp.zeros((B, n), dtype=jnp.float32)
    probs = probs.at[jnp.arange(B), lo].add(w_lo)
    probs = probs.at[jnp.arange(B), hi].add(w_hi)
    return jnp.log(jnp.maximum(probs, 1e-30))


# =============================================================================
# True-model apply_fn (canon model_apply_fn contract; see SPEC honesty section)
# =============================================================================

class TrueVectorModel:
    """Perfect-model stand-in satisfying the canon search's model_apply_fn contract.

    B=1 only (the probe plans one real decision at a time, 831 pattern).
    """

    def __init__(self, prices, cfg, codec, discount, value_rollout_steps):
        self.prices = prices
        self.cfg = cfg
        self.codec = codec
        self.discount = discount
        self.R = value_rollout_steps
        A = cfg.n_assets
        self._hold_vec = jnp.ones((A,), dtype=jnp.int32)
        self._n_policy = A * N_ACTIONS_PER_ASSET
        self._jstep = partial(step_vector, prices=prices, config=cfg)

    def _rollout_value(self, state):
        """Truncated true-env all-HOLD rollout value from `state` (scalar)."""
        if self.R <= 0:
            return jnp.zeros(())

        def hold_step(carry_state, _):
            ns, _, r, _ = self._jstep(carry_state, self._hold_vec)
            return ns, r

        _, rs = jax.lax.scan(hold_step, state, xs=None, length=self.R)
        disc = self.discount ** jnp.arange(self.R)
        return jnp.sum(disc * rs)

    def _batch(self, state):
        return jax.tree_util.tree_map(lambda x: x[None], state)

    def _unbatch(self, state_b):
        return jax.tree_util.tree_map(lambda x: x[0], state_b)

    def __call__(self, variables, *args, method=None, mode=None):
        del variables  # no params — the model IS the env
        if method == 'initial_inference':
            (observation,) = args                          # (1, L) uint32
            state = self.codec.decode(observation)
            value = self._rollout_value(state)[None]       # (1,)
            return {
                'latent_state': self._batch(state),
                'policy_logits': jnp.zeros((1, self._n_policy), dtype=jnp.float32),
                'value': two_hot_logits(value),
            }
        if mode == 'recurrent_inference_vector':
            embedding, vec = args                          # state pytree (B=1), (1, A)
            state = self._unbatch(embedding)
            ns, _, r, _ = self._jstep(state, vec[0].astype(jnp.int32))
            value = self._rollout_value(ns)[None]
            return {
                'latent_state': self._batch(ns),
                'policy_logits': jnp.zeros((1, self._n_policy), dtype=jnp.float32),
                'value': two_hot_logits(value),
                'reward': two_hot_logits(r[None]),
            }
        raise ValueError(f"unsupported call: method={method!r} mode={mode!r}")


# =============================================================================
# Rollout: canon search plans, the REAL env steps once per decision
# =============================================================================

def run_rollout(close_panel, per_node, K, sims, seed, warm, fee_bps, slip_bps,
                position_size_pct, discount, value_rollout_steps, max_decisions,
                hold_bias, dither_scale, window=5):
    T, A = close_panel.shape
    prices = jnp.asarray(close_panel, dtype=jnp.float32)
    cfg = EnvConfig(
        n_assets=A, initial_cash=INITIAL_CASH, fee_bps=fee_bps, slippage_bps=slip_bps,
        window_size=window, position_size_pct=position_size_pct,
        allow_short=True, tp_sl_enabled=False, max_steps=2000, vector_action=True,
    )
    # Head-room: deep search paths advance up to `sims` steps beyond the decision
    # row, plus the R-step value rollout (JAX clamps OOB gathers, but stale prices
    # in deep branches near the panel end would silently distort the comparison).
    depth_headroom = sims + value_rollout_steps
    N = T - 1 - warm - depth_headroom
    if max_decisions is not None:
        N = min(N, int(max_decisions))
    if N < 2:
        raise RuntimeError(f"panel too short: N={N}")

    state = reset(jax.random.PRNGKey(0), prices, cfg).replace(
        episode_start_idx=jnp.array(warm), current_step=jnp.array(0),
        max_episode_steps=jnp.array(N + depth_headroom + 2),
        initial_portfolio_value=jnp.array(INITIAL_CASH),
        prev_portfolio_value=jnp.array(INITIAL_CASH),
    )
    codec = StateCodec(state)
    _assert_roundtrip(codec, state)

    model = TrueVectorModel(prices, cfg, codec, discount, value_rollout_steps)
    search = SampledMuZeroVectorSearch(
        model_apply_fn=model, n_assets=A, n_actions_per_asset=N_ACTIONS_PER_ASSET,
        num_samples=K, num_simulations=sims, max_num_considered_actions=K,
        discount=discount, hold_bias=hold_bias, dither_scale=dither_scale,
        proposal_scheme='dithered-book',
        per_node_resampling=per_node,
    )

    @jax.jit
    def decide(flat_state, rng):
        # deterministic=True — carry-note 3 eval selection (visit argmax).
        av, _, _, _ = search.raw_search(
            {}, rng, flat_state, action_mask=None, book=None, deterministic=True)
        return av

    jstep = jax.jit(partial(step_vector, prices=prices, config=cfg))

    rng = jax.random.PRNGKey(seed)
    pvs = np.empty(N + 1, dtype=np.float64)
    pvs[0] = INITIAL_CASH
    n_long = n_short = 0
    for k in range(N):
        rng, srng = jax.random.split(rng)
        av = decide(codec.encode(state), srng)
        state, _, _, _ = jstep(state, av[0].astype(jnp.int32))
        cur = get_current_prices(prices, state.current_step - 1, state.episode_start_idx)
        pvs[k + 1] = float(compute_portfolio_value(state, cur))
        sd = np.asarray(state.pos_side)
        n_long += int((sd == 1).sum())
        n_short += int((sd == 2).sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        rets = pvs[1:] / np.maximum(pvs[:-1], 1e-8) - 1.0
    rets = np.where(np.isfinite(rets), rets, 0.0)
    return {
        "rets": rets, "n": N, "final_pv": float(pvs[-1]),
        "avg_long": n_long / max(N, 1), "avg_short": n_short / max(N, 1),
    }


# =============================================================================
# driver
# =============================================================================

def run(db_path, start, end, out_path, tag, args):
    corridor = _load_corridor()
    day_ts, close, logc, syms, info = corridor.load_panel(db_path, start, end)
    A = close.shape[1]
    position_size_pct = (args.position_size_pct if args.position_size_pct is not None
                         else 0.5 / A)
    tgt, pres, warm = corridor.book_target_signs(logc)

    T = close.shape[0]
    depth_headroom = args.sims + args.value_rollout_steps
    N = T - 1 - warm - depth_headroom
    if args.max_decisions is not None:
        N = min(N, int(args.max_decisions))

    # Corridor floor recomputed on the SAME rows (context, 831 convention).
    floor = {}
    for c in GATE_COSTS:
        fee, slip = (0.0, 0.0) if c == 0 else (FEE_BPS, SLIP_BPS)
        rep = corridor.replay_floor(close, tgt, warm, fee, slip, position_size_pct)
        floor[c] = corridor._sharpe(rep["rets"][:N])

    seeds = [int(s) for s in args.seeds.split(",") if s != ""]
    arms = {}
    for arm_name, per_node in (("off_root_reuse", False), ("on_per_node", True)):
        per_seed = {c: [] for c in GATE_COSTS}
        diag = None
        for sd in seeds:
            for c in GATE_COSTS:
                fee, slip = (0.0, 0.0) if c == 0 else (FEE_BPS, SLIP_BPS)
                rep = run_rollout(
                    close, per_node, args.k, args.sims, sd, warm, fee, slip,
                    position_size_pct, args.discount, args.value_rollout_steps,
                    args.max_decisions, args.hold_bias, args.dither_scale)
                per_seed[c].append(corridor._sharpe(rep["rets"]))
                if c == 0:
                    diag = {"final_pv": rep["final_pv"], "avg_long": rep["avg_long"],
                            "avg_short": rep["avg_short"], "n": rep["n"]}
        arms[arm_name] = {
            "per_node_resampling": per_node, "K": args.k, "sims": args.sims,
            "sharpe_mean": {c: float(np.mean(per_seed[c])) for c in GATE_COSTS},
            "sharpe_seeds": {c: [float(x) for x in per_seed[c]] for c in GATE_COSTS},
            "diag": diag,
        }

    d_net21 = arms["on_per_node"]["sharpe_mean"][21] - arms["off_root_reuse"]["sharpe_mean"][21]
    d_gross = arms["on_per_node"]["sharpe_mean"][0] - arms["off_root_reuse"]["sharpe_mean"][0]
    on_wins = sum(
        1 for a, b in zip(arms["on_per_node"]["sharpe_seeds"][21],
                          arms["off_root_reuse"]["sharpe_seeds"][21]) if a > b)
    read = ("ON >= OFF (mean net@21)" if d_net21 >= 0 else "OFF > ON (mean net@21)")

    result = {
        "tag": tag, "db": db_path, "window": [start, end], "n_assets": A,
        "n_days": int(day_ts.size), "warm": warm, "n_decisions": N,
        "position_size_pct": position_size_pct, "seeds": seeds,
        "K": args.k, "sims": args.sims, "discount": args.discount,
        "value_rollout_steps": args.value_rollout_steps,
        "hold_bias": args.hold_bias, "dither_scale": args.dither_scale,
        "proposal_scheme": "dithered-book", "cost_tiers_bp": list(GATE_COSTS),
        "value_heuristic": (f"truncated_env_rollout(R={args.value_rollout_steps}, HOLD, "
                            f"disc={args.discount}), two-hot 401-bin encode"),
        "floor": {str(c): (float(v) if v is not None and np.isfinite(v) else None)
                  for c, v in floor.items()},
        "arms": arms,
        "acceptance": {
            "delta_net21_on_minus_off": float(d_net21),
            "delta_gross_on_minus_off": float(d_gross),
            "on_seed_wins_net21": f"{on_wins}/{len(seeds)}",
            # The frozen hypothesis (SPEC.md pre-registration) vs what this run showed.
            "preregistered_criterion": "ON >= OFF (mean net@21)",
            "observed_read": read,
            "smoke": True,
            "note": ("FIXTURE-SMOKE, true-model (constant policy) — tests the fresh-"
                     "candidate-diversity-at-depth mechanism only; a TIE here is a "
                     "FINDING (pre-registered), not a failure. Default stays OFF; the "
                     "trained-policy answer is the scheduled Stage-3/4 stream."),
        },
    }
    _print_report(result)
    if out_path:
        with open(out_path, "w") as fh:
            json.dump(result, fh, indent=2, default=float)
        print(f"\n[838] wrote {out_path}")
    return result


def _fmt(x):
    return "   n/a" if x is None or not np.isfinite(x) else f"{x:+.3f}"


def _print_report(r):
    print("=" * 84)
    print(f"838 PER-NODE RESAMPLING ACCEPTANCE ({r['tag']}) — {r['window'][0]} .. {r['window'][1]}")
    print(f"panel: {r['n_assets']} assets, {r['n_decisions']} decisions, warm={r['warm']}, "
          f"K={r['K']}, sims={r['sims']}, seeds={r['seeds']}")
    print(f"value heuristic: {r['value_heuristic']}")
    fl = r["floor"]
    print(f"corridor floor  net@21 {_fmt(fl.get('21'))}  gross {_fmt(fl.get('0'))}")
    print("-" * 84)
    print(f"{'arm':>16} | {'net@21':>8} {'gross':>8} | per-seed net@21")
    for name, g in r["arms"].items():
        seeds21 = ", ".join(_fmt(x) for x in g["sharpe_seeds"][21])
        print(f"{name:>16} | {_fmt(g['sharpe_mean'][21]):>8} {_fmt(g['sharpe_mean'][0]):>8} | [{seeds21}]")
    acc = r["acceptance"]
    print("-" * 84)
    print(f"delta net@21 (ON-OFF): {_fmt(acc['delta_net21_on_minus_off'])}   "
          f"delta gross: {_fmt(acc['delta_gross_on_minus_off'])}   "
          f"ON seed wins: {acc['on_seed_wins_net21']}")
    print(f"PRE-REGISTERED CRITERION: {acc['preregistered_criterion']}")
    print(f"OBSERVED READ: {acc['observed_read']}")
    print(f"  {acc['note']}")
    print("=" * 84)


def main():
    ap = argparse.ArgumentParser(
        description="TASK_838 acceptance: per-node resampling ON vs OFF (canon search, true model)")
    ap.add_argument("--db", default=FIXTURE_DB)
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default="2025-09-01")
    ap.add_argument("--position-size-pct", type=float, default=None)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--k", type=int, default=8, help="K sampled vectors per node (equal both arms)")
    ap.add_argument("--sims", type=int, default=32, help="num_simulations (depth>1 budget)")
    ap.add_argument("--discount", type=float, default=0.997)
    ap.add_argument("--value-rollout-steps", type=int, default=3)
    ap.add_argument("--hold-bias", type=float, default=3.0,
                    help="dithered-book HOLD bias (E2-carried scheme)")
    ap.add_argument("--dither-scale", type=float, default=1.0)
    ap.add_argument("--max-decisions", type=int, default=60,
                    help="cap decisions for the fixture smoke (negative => full panel)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.max_decisions is not None and args.max_decisions < 0:
        args.max_decisions = None
    is_fixture = os.path.abspath(args.db) == os.path.abspath(FIXTURE_DB)
    assert "trading_data.db" not in os.path.abspath(args.db), "never trading_data.db"
    # The canonical evidence file is written ONLY by the canonical invocation:
    # fixture db AND every acceptance-relevant knob at its default. Any deviation
    # (exploratory sweep) must name its own --out explicitly.
    is_canonical = is_fixture and all([
        args.start == "2021-01-01", args.end == "2025-09-01",
        args.position_size_pct is None, args.seeds == "0,1,2",
        args.k == 8, args.sims == 32, args.discount == 0.997,
        args.value_rollout_steps == 3, args.hold_bias == 3.0,
        args.dither_scale == 1.0, args.max_decisions == 60,
    ])
    tag = args.tag or ("fixture-smoke" if is_fixture else "custom")
    out = args.out
    if out is None:
        if is_canonical:
            out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "results_fixture_smoke.json")
        else:
            ap.error("non-canonical args: pass --out explicitly (the default "
                     "results_fixture_smoke.json is reserved for the all-defaults "
                     "fixture run)")
    run(args.db, args.start, args.end, out, tag, args)


if __name__ == "__main__":
    main()
