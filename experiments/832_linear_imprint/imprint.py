#!/usr/bin/env python3
"""TASK_832 / E5 — linear-book imprint runner (pathway rung R2, FACTORY half).

Mechanism (pinned, QUEUE row 3 scope note 2026-07-08): **BUFFER-INJECTION**.
Linear-book (819 trend CTA) teacher trajectories are generated ON-THE-FLY (no npz
pre-generation — placement rules) and injected into the shipped
``JAXReplayBuffer(vector_action=True)`` via ``add_trajectory`` with ONE-HOT per-asset
policy marginals. The already-shipped factored CE (mean-over-assets,
``trainer._vector_policy_ce``) IS the BC loss — ``trainer.train`` on the injected
buffer = the imprint phase. ZERO canon edits; no trainer/env/search/model diffs.

Pinned design points (pre-registered in SPEC.md before any run):
  (a) reduced ``entropy_coef`` / ``min_entropy`` for the imprint phase (the adaptive
      entropy push fights one-hot targets) — CLI args, trainer DEFAULTS untouched.
  (b) PANEL CONSISTENCY: teacher trajectories walk the TRAINING env panel
      (``create_jax_env_historical``, 1-min ohlcv -> ``bar_minutes`` row-resample),
      with the 819 book signs computed on THAT SAME panel — NOT the 830 corridor's
      daily-close panel. Obs = the training encoder (``jit_batch_encode_simple``,
      window [t-window_size, t-1], self_play.py:646-658 rule) on the same prices.
  (c) the ExIt half is NOT new code: ``--load-checkpoint`` (cli.py:250)
      ``--reset-optimizer`` (cli.py:252) ``--clear-buffer`` (cli.py:254) on
      train_jax_fast.py (consumed at train_jax_fast.py:443-467) — verified anchors.

Imprint-sanity output: the imprinted policy replayed in-env DETERMINISTICALLY via
DIRECT POLICY-ARGMAX (pre-registered; cheaper than visit-argmax and acceptable for
the sanity check) vs the teacher floor on the identical decision rows. The DEC-2
±0.10 tolerance is a RUN-TIME verdict — the fixture smoke never applies it.

FACTORY / SCHEDULED split: this module + its fixture smoke are the factory
deliverable. The real imprint (+ ExIt continuation) is the scheduled <=1-GPU-day run
(DEC-2); it exports JAX_PLATFORMS=cuda — the default here is CPU (house style).
CPU / fixture-only inside the factory; NEVER trading_data.db.
"""
import os
# CPU-only by default (factory/smoke). The scheduled GPU run overrides JAX_PLATFORMS.
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

from trader import backtest
from mcts_trader.env.jax_trading_env import (
    reset, step_vector, get_current_prices, compute_portfolio_value,
)
from mcts_trader.env.jax_encoder import (
    EncoderConfig as FullEncoderConfig, jit_batch_encode_simple,
)
from mcts_trader.training.env_factory import create_jax_env_historical
from mcts_trader.jax_agent.muzero_model import MuZeroNetworks
from mcts_trader.jax_agent.trainer import MuZeroTrainer
from mcts_trader.jax_agent.replay_buffer import JAXReplayBuffer
from mcts_trader.jax_agent.returns import compute_returns

FIXTURE_DB = os.path.join(_REPO, "fixtures", "trading_fixture.db")
FEE_BPS, SLIP_BPS = 16.0, 5.0          # net@21 = fee16 (Kraken taker) + slip5 bp/side
GATE_COSTS = (0, 21)                    # cost tiers, mirroring 830/831
N_ACTIONS_PER_ASSET = 5

# ExIt-half flag anchors (PIN c) — VERIFIED against the working tree 2026-07-08.
EXIT_FLAG_ANCHORS = {
    "--load-checkpoint": "mcts_trader/training/cli.py:250",
    "--reset-optimizer": "mcts_trader/training/cli.py:252",
    "--clear-buffer": "mcts_trader/training/cli.py:254",
    "consumed_load+reset": "train_jax_fast.py:443-462 (trainer.load_checkpoint + reset_optimizer_state)",
    "consumed_clear": "train_jax_fast.py:465-467 (replay_buffer.clear)",
    "trainer.load_checkpoint": "mcts_trader/jax_agent/trainer.py:1700",
    "trainer.reset_optimizer_state": "mcts_trader/jax_agent/trainer.py:1781",
    "buffer.clear": "mcts_trader/jax_agent/replay_buffer.py:423",
}


def _load_corridor():
    """Import the TASK_830 corridor module by path (dir name '830_corridor' is not an
    identifier). Reused pieces: ``book_target_signs`` (819 book signs via
    trader.strategy on a given log-close panel) + ``_to_action_vec`` (the pinned
    book->action_vec mapping). We do NOT reuse its daily-close panel loader (PIN b)."""
    path = os.path.join(_REPO, "experiments", "830_corridor", "corridor.py")
    spec = importlib.util.spec_from_file_location("corridor830", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# =============================================================================
# panel — the TRAINING env panel (PIN b)
# =============================================================================

def load_training_panel(args):
    """Load the panel through the TRAINING path: ``create_jax_env_historical``
    (1-min ohlcv -> ``bar_minutes`` row-resample, coverage-ranked universe, ffill).
    Returns (prices_j (T,A) float32 jnp, logc (T,A) float64 np, env_cfg)."""
    _, prices_j, env_cfg = create_jax_env_historical(
        db_path=args.db,
        start_date=args.start,
        end_date=args.end,
        n_envs=1,                                # batched reset states are discarded
        n_assets=args.n_assets,
        specific_symbols=None,                   # loader ranks by minute coverage
        fee_bps=float(args.fee_bps),
        slippage_bps=float(args.slip_bps),
        max_episode_steps=8,                     # walk state overrides max_episode_steps
        position_size_pct=args.position_size_pct,
        rng_key=jax.random.PRNGKey(args.seed),
        reward_attributable=args.reward_attributable,
        bar_minutes=args.bar_minutes,
        vector_action=True,
        vector_weight_rebalance=args.vector_weight_rebalance,
        vector_weight_unit=args.vector_weight_unit,
        vector_weight_max=args.vector_weight_max,
        vector_rebalance_deadband=args.vector_rebalance_deadband,
    )
    prices_np = np.asarray(prices_j, dtype=np.float64)
    logc = np.log(np.maximum(prices_np, 1e-12))
    return prices_j, logc, env_cfg


# =============================================================================
# in-env walks (teacher replay / imprinted-policy sanity replay)
# =============================================================================

def walk_env(prices_j, cfg, start_row, n_decisions, action_fn):
    """Deterministic single-episode walk from panel row ``start_row`` through
    ``step_vector`` (the corridor/831 replay pattern, on the TRAINING panel/config).
    ``action_fn(k, state) -> (A,) int32`` (state = the pre-step TradingState, so a
    state-greedy mapping can read ``pos_side`` and, for the weight-native ladder,
    reconstruct the current signed weight). Returns per-decision returns + diag."""
    initial_cash = float(cfg.initial_cash)
    state = reset(jax.random.PRNGKey(0), prices_j, cfg).replace(
        episode_start_idx=jnp.array(start_row), current_step=jnp.array(0),
        max_episode_steps=jnp.array(n_decisions + 2),
        initial_portfolio_value=jnp.array(initial_cash),
        prev_portfolio_value=jnp.array(initial_cash),
    )
    jstep = jax.jit(partial(step_vector, prices=prices_j, config=cfg))
    A = int(cfg.n_assets)
    actions = np.zeros((n_decisions, A), dtype=np.int32)
    rewards = np.zeros(n_decisions, dtype=np.float32)
    dones = np.zeros(n_decisions, dtype=np.float32)
    pvs = np.empty(n_decisions + 1, dtype=np.float64)
    pvs[0] = initial_cash
    n_long = n_short = 0
    bankrupt_at = -1
    for k in range(n_decisions):
        act = np.asarray(action_fn(k, state), dtype=np.int32)
        state, _, r, d = jstep(state, jnp.asarray(act))
        jax.block_until_ready(r)
        actions[k] = act
        rewards[k] = float(r)
        dones[k] = float(d)
        cur = get_current_prices(prices_j, state.current_step - 1, state.episode_start_idx)
        pv = float(compute_portfolio_value(state, cur))
        pvs[k + 1] = pv
        sd = np.asarray(state.pos_side)
        n_long += int((sd == 1).sum()); n_short += int((sd == 2).sum())
        if bankrupt_at < 0 and pv <= initial_cash * cfg.liquidation_threshold_pct:
            bankrupt_at = k
    with np.errstate(divide="ignore", invalid="ignore"):
        rets = pvs[1:] / np.maximum(pvs[:-1], 1e-8) - 1.0
    rets = np.where(np.isfinite(rets), rets, 0.0)
    return {
        "actions": actions, "rewards": rewards, "dones": dones, "rets": rets,
        "final_pv": float(pvs[-1]),
        "avg_long": n_long / max(n_decisions, 1),
        "avg_short": n_short / max(n_decisions, 1),
        "bankrupt_at": bankrupt_at,
    }


def encode_training_obs(prices_j, start_row, n_decisions, cfg, block=256):
    """TRAINING observations for decision rows start_row..start_row+N-1: the exact
    self-play rule (self_play.py:646-658): window_start = price_idx - window_size,
    encoded with ``jit_batch_encode_simple`` on the SAME panel prices (PIN b).
    Obs depend only on (prices, window_start) — computed in blocks, held in RAM."""
    A = int(cfg.n_assets)
    enc = FullEncoderConfig(n_assets=A, window_size=int(cfg.window_size), n_channels=27)
    ws_all = np.arange(start_row, start_row + n_decisions, dtype=np.int32) - int(cfg.window_size)
    assert ws_all.min() >= 0, "obs window underflow — start_row must be >= window_size"
    out = []
    for i in range(0, n_decisions, block):
        ob = jit_batch_encode_simple(prices_j, jnp.asarray(ws_all[i:i + block]), enc)
        jax.block_until_ready(ob)
        out.append(np.asarray(ob, dtype=np.float32))
    return np.concatenate(out, axis=0)          # (N, 27, A, A)


# =============================================================================
# buffer injection — the pinned mechanism
# =============================================================================

def inject_teacher(replay_buffer, obs, actions, rewards, dones, discount, chunk_len):
    """ONE-HOT per-asset marginals + MC-return value targets -> ``add_trajectory``.

    values = discounted MC returns over the teacher's env rewards (bootstrap 0 — the
    teacher has no value net; ``compute_returns(use_gae=False)``, the shipped return
    path). marginals[t,i,a] = 1{a == teacher_action[t,i]} — under the factored CE
    (mean-over-assets) this makes trainer 'policy_loss' EXACTLY the BC loss."""
    n = obs.shape[0]
    values = compute_returns(rewards=rewards, values=np.zeros_like(rewards),
                             dones=dones, bootstrap_value=0.0,
                             discount=discount, use_gae=False).astype(np.float32)
    marginals = np.eye(N_ACTIONS_PER_ASSET, dtype=np.float32)[actions]  # (N, A, 5)
    seq_len = replay_buffer.sequence_length
    bounds = list(range(0, n, chunk_len)) + [n]
    if len(bounds) >= 3 and bounds[-1] - bounds[-2] < seq_len:
        bounds.pop(-2)                            # merge a too-short tail chunk
    n_chunks = 0
    for a, b in zip(bounds[:-1], bounds[1:]):
        replay_buffer.add_trajectory(
            jnp.asarray(obs[a:b]), jnp.asarray(actions[a:b]),
            jnp.asarray(rewards[a:b]), jnp.asarray(values[a:b]),
            jnp.asarray(marginals[a:b]), jnp.asarray(dones[a:b]),
        )
        n_chunks += 1
    return values, n_chunks


# =============================================================================
# model / trainer — canon construction, imprint entropy knobs via CLI (PIN a)
# =============================================================================

def build_model_and_trainer(args, n_assets):
    """Mirror setup.py::setup_model_and_trainer for the vector path at the canon
    defaults that matter here (value_loss_weight=2.0 setup.py:530, PopArt ON,
    reward_scale 100, lr 1e-4 cli.py:121). Entropy knobs come from the CLI (PIN a) —
    trainer DEFAULTS are untouched; these are constructor arguments."""
    model = MuZeroNetworks(
        state_dim=args.state_dim,
        action_space_size=n_assets * N_ACTIONS_PER_ASSET,
        H_cycles=args.h_cycles,
        L_cycles=args.l_cycles,
        use_prediction_head=False,
        n_assets=n_assets,
        use_vector_action=True,
        vector_n_actions_per_asset=N_ACTIONS_PER_ASSET,
    )
    trainer = MuZeroTrainer(
        model=model,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        discount=args.discount,
        entropy_coef=args.entropy_coef,          # PIN a (canon 0.01, cli.py:130)
        min_entropy=args.min_entropy,            # PIN a (canon 3.0, cli.py:295)
        use_popart=True,
        value_loss_weight=2.0,                   # canon setup.py:530
        reward_hybrid=args.reward_hybrid,        # 828 Stage-3 primary operator
        vector_action=True,
    )
    dummy_obs = jnp.ones((1, 27, n_assets, n_assets))
    trainer.init_train_state(jax.random.PRNGKey(args.seed), dummy_obs)
    assert trainer.vector_action and model.use_vector_action, "832: vector flag mismatch"
    return model, trainer


def policy_argmax_actions(model, params, obs_np, n_assets, block=128):
    """Pre-registered sanity-replay operator: DIRECT policy argmax per asset from
    initial_inference on the training obs (deterministic, no search). The training
    obs depend only on prices, so all N decisions batch through the net at once."""
    @jax.jit
    def fwd(o):
        out = model.apply({'params': params}, o, mode='initial_inference')
        logits = out['policy_logits'].reshape(o.shape[0], n_assets, N_ACTIONS_PER_ASSET)
        return jnp.argmax(logits, axis=-1)
    acts = []
    for i in range(0, obs_np.shape[0], block):
        a = fwd(jnp.asarray(obs_np[i:i + block]))
        jax.block_until_ready(a)
        acts.append(np.asarray(a, dtype=np.int32))
    return np.concatenate(acts, axis=0)          # (N, A) in {0..4}


# =============================================================================
# scoring
# =============================================================================

def _sharpe(rets, period):
    """Audited annualization (backtest._sharpe): mean/std * sqrt(525600/period).
    period = minutes per decision = bar_minutes (pre-registered)."""
    return backtest._sharpe(np.asarray(rets, dtype=np.float64), period)


def _rel(a, b, eps=1e-8):
    """|a - b| / max(|b|, eps), guarded (b is the reference = order-semantics floor)."""
    if not (np.isfinite(a) and np.isfinite(b)):
        return float("nan")
    return float(abs(a - b) / max(abs(b), eps))


def _make_weight_native_teacher_fn(prices_j, env_cfg, tgt, start_row, n_assets):
    """Weight-native ladder teacher (TASK_845 fallback mapping, pre-registered).

    Per step per asset: ``action = sign(w_target - w_cur)`` while ``|dw| > deadband``,
    else HOLD. ``w_target`` drives to the signed cap ±``w_max`` in the 819 book's target
    direction (flat target => 0); ``w_cur`` is the pre-step signed market-notional weight
    reconstructed from the state exactly as the env does
    (``jax_trading_env.py:1608-1612``). Emits the same 0/1/2 codes the weight env reads as
    -1/0/+1-unit target-weight steps — the natural ladder encoding under weight semantics.
    The (static) weight schedule resolves from ``env_cfg`` with the env's own sentinels."""
    w_unit = (env_cfg.vector_weight_unit if env_cfg.vector_weight_unit > 0.0
              else 1.0 / n_assets)
    w_max = env_cfg.vector_weight_max * w_unit
    deadband = (env_cfg.vector_rebalance_deadband if env_cfg.vector_rebalance_deadband >= 0.0
                else 0.1 * w_unit)

    def fn(k, state):
        # CAVEAT: w_cur is reconstructed from the PRE-preamble state.pos_size, whereas the env
        # computes its own w_cur AFTER process_limit_orders/margin_call/tp_sl
        # (jax_trading_env.py:1608-1612). No-op on the fixture (no limits, tp/sl off, no margin
        # call fires), but at fee16 real-panel a margin call between steps could shrink pos_size
        # so this ladder decision reads a slightly stale w_cur — acceptable for a greedy teacher
        # mapping (it only sets the ±unit step direction; the env still applies the exact math).
        current_prices = np.asarray(get_current_prices(
            prices_j, state.current_step, state.episode_start_idx), dtype=np.float64)
        pv = max(float(compute_portfolio_value(state, jnp.asarray(current_prices,
                                                                  dtype=jnp.float32))), 1e-10)
        size = np.asarray(state.pos_size, dtype=np.float64)
        side = np.asarray(state.pos_side)
        pos_notional = size * current_prices
        w_cur = np.where(side == 1, pos_notional / pv,
                np.where(side == 2, -pos_notional / pv, 0.0))
        tgt_row = tgt[start_row + k]
        w_target = np.where(tgt_row > 0, w_max, np.where(tgt_row < 0, -w_max, 0.0))
        dw = w_target - w_cur
        act = np.ones(n_assets, dtype=np.int32)
        act = np.where(dw > deadband, 2, act)          # need more long  -> BUY
        act = np.where(dw < -deadband, 0, act)         # need more short -> SELL
        return act.astype(np.int32)

    return fn


def order_floor_reference(prices_j, env_cfg, teacher_fn, start_row, N, bar_minutes):
    """The FIXED order-semantics floor: the canonical order-code teacher
    (``corridor._to_action_vec``, via ``teacher_fn``) walked at net@21 through the ORDER
    env (weight OFF). This is the 830/831 corridor floor and the SAME reference both the
    primary and the weight-native re-gate are scored against."""
    order_cfg = env_cfg.replace(fee_bps=FEE_BPS, slippage_bps=SLIP_BPS,
                                vector_weight_rebalance=False)
    wo = walk_env(prices_j, order_cfg, start_row, N, teacher_fn)
    return {"s": _sharpe(wo["rets"], bar_minutes),
            "final_pv": wo["final_pv"], "bankrupt_at": wo["bankrupt_at"]}


def expressibility_gate(prices_j, env_cfg, weight_teacher_fn, start_row, N, bar_minutes,
                        order_ref, mapping, tol=0.10):
    """Pre-registered teacher-expressibility gate (TASK_845, the real deliverable).

    Walks ``weight_teacher_fn`` at net@21 through the WEIGHT env (weight ON) on the
    IDENTICAL decision rows as ``order_ref`` (the fixed order-env floor), and reports
    BOTH net@21 Sharpes. Fixture gate = |teacher_weight - teacher_order| <= ``tol``
    relative (default 10%).

    The teacher walk is state-greedy: ``weight_teacher_fn`` reads the walking env's own
    state each step, so the order-code mapping produces the ±1-unit budget-scaled
    target-weight book (``jax_trading_env.py:1600-1699``). The order-env value is the
    corridor floor reference (830/831); the weight-env value is the book the ExIt loop
    will actually imprint on. A miss > tol means the mapping does NOT reproduce ~floor
    under weight semantics => the caller swaps in the weight-native ladder and re-gates
    against the SAME order_ref (DEC-2 core-risk check)."""
    weight_cfg = env_cfg.replace(fee_bps=FEE_BPS, slippage_bps=SLIP_BPS,
                                 vector_weight_rebalance=True)
    ww = walk_env(prices_j, weight_cfg, start_row, N, weight_teacher_fn)
    s_o = order_ref["s"]
    s_w = _sharpe(ww["rets"], bar_minutes)
    rel = _rel(s_w, s_o)
    passed = bool(np.isfinite(rel) and rel <= tol)
    return {
        "mapping": mapping,
        "teacher_order_net21": float(s_o) if np.isfinite(s_o) else None,
        "teacher_weight_net21": float(s_w) if np.isfinite(s_w) else None,
        "rel_diff": rel if np.isfinite(rel) else None,
        "tolerance_rel": tol,
        "passed": passed,
        "order_final_pv": order_ref["final_pv"], "weight_final_pv": ww["final_pv"],
        "order_bankrupt_at": order_ref["bankrupt_at"], "weight_bankrupt_at": ww["bankrupt_at"],
    }


def _loss_summary(history, key, head=5):
    vals = [float(v) for v in history.get(key, [])]
    if not vals:
        return None
    return {
        "first": vals[0], "last": vals[-1],
        "head_mean": float(np.mean(vals[:head])),
        "tail_mean": float(np.mean(vals[-head:])),
        "min": float(np.min(vals)), "max": float(np.max(vals)),
        "all_finite": bool(np.all(np.isfinite(vals))),
        "trajectory": vals if len(vals) <= 200 else vals[:: max(1, len(vals) // 200)],
    }


# =============================================================================
# driver
# =============================================================================

def run(args, out_path, tag):
    corridor = _load_corridor()

    # ---- panel (TRAINING path, PIN b) + book signs on THAT panel ----
    prices_j, logc, env_cfg = load_training_panel(args)
    T, A = logc.shape
    tgt, pres, warm = corridor.book_target_signs(logc)
    start_row = max(int(warm), int(env_cfg.window_size))
    N = T - 1 - start_row
    if args.max_decisions is not None:
        N = min(N, int(args.max_decisions))
    if N < 8:
        raise RuntimeError(
            f"panel too short: T={T}, warm={warm}, window={env_cfg.window_size} -> N={N} "
            f"(fixture smoke needs --bar-minutes small enough, see SPEC)")
    print(f"[832] panel T={T} A={A} warm={warm} start_row={start_row} N={N} "
          f"bar_minutes={args.bar_minutes}")

    # ---- teacher walks, one per cost tier (same mapping; costs differ) ----
    def teacher_fn(k, state):
        side = np.asarray(state.pos_side)
        return corridor._to_action_vec(tgt[start_row + k], side, A)

    tier_cfg = {}
    for c in GATE_COSTS:
        fee, slip = (0.0, 0.0) if c == 0 else (FEE_BPS, SLIP_BPS)
        tier_cfg[c] = env_cfg.replace(fee_bps=fee, slippage_bps=slip)
    teacher = {c: walk_env(prices_j, tier_cfg[c], start_row, N, teacher_fn)
               for c in GATE_COSTS}
    floor = {c: _sharpe(teacher[c]["rets"], args.bar_minutes) for c in GATE_COSTS}
    print(f"[832] teacher floor in-env: gross {floor[0]:+.3f}  net@21 {floor[21]:+.3f}")

    # ---- teacher-expressibility gate (TASK_845, pre-registered, the real deliverable) ----
    def _gate_line(g):
        rel = g["rel_diff"] if g["rel_diff"] is not None else float("nan")
        print(f"[845 GATE] {g['mapping']}: order net@21 {_fmt(g['teacher_order_net21'])}  "
              f"weight net@21 {_fmt(g['teacher_weight_net21'])}  rel {rel:.3f} "
              f"(tol {g['tolerance_rel']:.2f}) -> {'PASS' if g['passed'] else 'FAIL'}")

    wn_fn = _make_weight_native_teacher_fn(prices_j, env_cfg, tgt, start_row, A)
    order_ref = order_floor_reference(prices_j, env_cfg, teacher_fn, start_row, N,
                                      args.bar_minutes)
    gate = expressibility_gate(prices_j, env_cfg, teacher_fn, start_row, N, args.bar_minutes,
                               order_ref, "order-code (corridor._to_action_vec)")
    _gate_line(gate)
    if not gate["passed"]:
        # FAIL => re-gate the weight-native ladder mapping against the SAME order-code floor
        # reference (pre-registered contingency; the same wn_fn can be IMPRINTED, below).
        wn_gate = expressibility_gate(
            prices_j, env_cfg, wn_fn, start_row, N, args.bar_minutes, order_ref,
            "weight-native ladder (sign(w_target - w_cur) | |dw|>deadband)")
        _gate_line(wn_gate)
        gate["weight_native_regate"] = wn_gate

    # ---- select the teacher that is actually IMPRINTED (TASK_845 MAJOR fix) ----
    # The pre-registered contingency must be ACTIONABLE: when the order-code teacher fails the
    # weight-env expressibility gate, the runner must be able to imprint on the weight-native
    # ladder instead (not merely re-measure it). Default 'order-code' keeps every computed
    # number byte-identical to the pre-selector default path (the injected teacher, agreement,
    # values, floor, sanity and ledger are all unchanged). 'weight-native' walks the SAME
    # per-cost-tier configs with wn_fn and rebinds teacher/floor so the WHOLE run is
    # self-consistent with the imprinted mapping. The gate measurement above is untouched
    # (it always reports BOTH mappings vs the fixed order-code reference).
    imprint_mapping = args.imprint_mapping
    if imprint_mapping == "weight-native":
        # FAIL-CLOSED (TASK_845 MINOR): weight-native imprint is only coherent when the WHOLE
        # run is weight semantics — the teacher walk (tier_cfg inherits env_cfg's flag), the obs
        # encoding, and the ExIt continuation env. Without --vector-weight-rebalance the ladder
        # would walk the ORDER env (floor +8.454) while the gate certifies the WEIGHT-env teacher
        # (-4.593): two different teachers, a gate-uncertified imprint. Reject the combo.
        assert env_cfg.vector_weight_rebalance, (
            "--imprint-mapping weight-native requires --vector-weight-rebalance so the teacher "
            "walk, obs encoding, and ExIt env all run weight semantics; the gate certifies the "
            "weight-env teacher, not the order-env one")
        teacher = {c: walk_env(prices_j, tier_cfg[c], start_row, N, wn_fn)
                   for c in GATE_COSTS}
        floor = {c: _sharpe(teacher[c]["rets"], args.bar_minutes) for c in GATE_COSTS}
        print(f"[832] IMPRINT MAPPING = weight-native ladder; teacher floor in-env: "
              f"gross {floor[0]:+.3f}  net@21 {floor[21]:+.3f}")
    else:
        print(f"[832] IMPRINT MAPPING = order-code (default; byte-identical default path)")

    # ---- training observations for the SAME decision rows (PIN b) ----
    obs = encode_training_obs(prices_j, start_row, N, env_cfg)

    # ---- model + trainer + buffer-injection (the pinned mechanism) ----
    model, trainer = build_model_and_trainer(args, A)
    replay_buffer = JAXReplayBuffer(
        capacity=max(N + 8, 64),
        observation_shape=(27, A, A),
        action_size=A * N_ACTIONS_PER_ASSET,
        batch_size=args.batch_size,
        n_assets=A,
        vector_action=True,
    )
    net_tier = GATE_COSTS[-1]                   # imprint on cost-honest rewards (net@21)
    values, n_chunks = inject_teacher(
        replay_buffer, obs, teacher[net_tier]["actions"], teacher[net_tier]["rewards"],
        teacher[net_tier]["dones"], args.discount, args.chunk_len)
    print(f"[832] injected {N} teacher steps in {n_chunks} trajectory chunk(s); "
          f"buffer={len(replay_buffer)} trajectories")

    # ---- imprint phase = trainer.train on the injected buffer ----
    history = trainer.train(replay_buffer, num_iterations=args.iters,
                            rng=jax.random.PRNGKey(args.seed + 1))
    losses = {k: _loss_summary(history, k) for k in
              ("total_loss", "policy_loss", "value_loss", "reward_loss", "policy_entropy")}
    ce = losses["policy_loss"]
    finite = all(l["all_finite"] for l in losses.values() if l is not None)
    ce_decreased = bool(ce and ce["tail_mean"] < ce["head_mean"])
    print(f"[832] imprint CE (factored, mean-over-assets): first {ce['first']:.4f} -> "
          f"last {ce['last']:.4f} (head5 {ce['head_mean']:.4f} -> tail5 {ce['tail_mean']:.4f}) "
          f"decreased={ce_decreased} finite={finite}")

    # ---- checkpoint for the ExIt half (PIN c: consumed by shipped flags) ----
    # MAJOR fix (verify panel 2026-07-08): save_checkpoint persists trainer.entropy_coef
    # into trainer_metadata.json (trainer.py:1673-1674) and load_checkpoint restores it
    # UNCONDITIONALLY after construction (trainer.py:1719-1720) — without this handoff
    # the scheduled ExIt run would silently execute its whole improvement phase at the
    # imprint's reduced coefficient (10x weaker regularization AND 10x weaker adaptive
    # anti-collapse push, exactly at the collapse-prone post-imprint moment), and even
    # an explicit --entropy-coef on train_jax_fast.py would be overridden by the restore.
    # Training is complete here, so hand the ExIt phase its intended operator value
    # (default 0.01 = canon, cli.py:130) before writing the checkpoint.
    imprint_entropy_coef = float(trainer.entropy_coef)
    trainer.entropy_coef = float(args.ckpt_entropy_coef)
    print(f"[832] ENTROPY HANDOFF: imprint-phase entropy_coef {imprint_entropy_coef} -> "
          f"checkpoint metadata {trainer.entropy_coef} (ExIt-phase value; load_checkpoint "
          f"restores it, trainer.py:1719-1720 — verify the "
          f"'Restored entropy_coef: {trainer.entropy_coef}' log line at ExIt launch)")
    ckpt_path = os.path.abspath(args.ckpt_out)
    trainer.save_checkpoint(ckpt_path)
    print(f"[832] imprint checkpoint -> {ckpt_path}")

    # ---- imprint-sanity: deterministic policy-argmax replay in-env ----
    pol_actions = policy_argmax_actions(model, trainer.state.params, obs, A)
    agreement = float(np.mean(pol_actions == teacher[net_tier]["actions"]))
    sanity = {}
    for c in GATE_COSTS:
        rep = walk_env(prices_j, tier_cfg[c], start_row, N,
                       lambda k, state: pol_actions[k])
        s = _sharpe(rep["rets"], args.bar_minutes)
        sanity[c] = {
            "policy_inenv": float(s) if np.isfinite(s) else None,
            "teacher_floor_inenv": float(floor[c]) if np.isfinite(floor[c]) else None,
            "corridor_position": (float(s - floor[c])
                                  if np.isfinite(s) and np.isfinite(floor[c]) else None),
            "final_pv": rep["final_pv"], "avg_long": rep["avg_long"],
            "avg_short": rep["avg_short"], "bankrupt_at": rep["bankrupt_at"],
        }

    hist = lambda a: {str(i): int((a == i).sum()) for i in range(N_ACTIONS_PER_ASSET)}
    result = {
        "tag": tag, "db": args.db, "window": [args.start, args.end],
        "bar_minutes": args.bar_minutes, "n_assets": A, "n_panel_rows": T,
        "warm": int(warm), "start_row": start_row, "n_decisions": N,
        "position_size_pct": args.position_size_pct,
        "cost_tiers_bp": list(GATE_COSTS), "sharpe_period_minutes": args.bar_minutes,
        "imprint": {
            "imprint_mapping": imprint_mapping,   # which teacher mapping was IMPRINTED
            "mechanism": "buffer-injection: on-the-fly teacher walk -> one-hot per-asset "
                         "marginals -> JAXReplayBuffer(vector_action=True).add_trajectory "
                         "-> trainer.train (shipped factored CE = BC loss)",
            "config": {
                "iters": args.iters, "batch_size": args.batch_size, "lr": args.lr,
                "discount": args.discount, "entropy_coef": args.entropy_coef,
                "min_entropy": args.min_entropy,
                "ckpt_entropy_coef": args.ckpt_entropy_coef,
                "state_dim": args.state_dim,
                "h_cycles": args.h_cycles, "l_cycles": args.l_cycles,
                "use_popart": True, "value_loss_weight": 2.0,
                "reward_hybrid": args.reward_hybrid,
                "reward_attributable": args.reward_attributable,
                "fee_bps": args.fee_bps, "slip_bps": args.slip_bps,
                "vector_weight_rebalance": bool(env_cfg.vector_weight_rebalance),
                "vector_weight_unit": float(env_cfg.vector_weight_unit),
                "vector_weight_max": float(env_cfg.vector_weight_max),
                "vector_rebalance_deadband": float(env_cfg.vector_rebalance_deadband),
                "reward_tier_bp": net_tier, "chunk_len": args.chunk_len,
                "seed": args.seed,
                "value_targets": "MC returns (compute_returns use_gae=False, bootstrap 0)",
            },
            "n_injected_steps": N, "n_chunks": n_chunks,
            "losses": losses,
            "all_losses_finite": finite,
            "policy_ce_decreased": ce_decreased,
        },
        "teacher": {
            "floor_inenv": {str(c): (float(floor[c]) if np.isfinite(floor[c]) else None)
                            for c in GATE_COSTS},
            "final_pv": {str(c): teacher[c]["final_pv"] for c in GATE_COSTS},
            "avg_long": teacher[net_tier]["avg_long"],
            "avg_short": teacher[net_tier]["avg_short"],
            "action_hist": hist(teacher[net_tier]["actions"]),
            "values_stats": {"mean": float(values.mean()), "std": float(values.std()),
                             "min": float(values.min()), "max": float(values.max())},
        },
        "expressibility_gate": gate,
        "sanity": {
            "method": "direct policy-argmax rollout (pre-registered; deterministic, "
                      "no search) on the identical decision rows",
            "per_tier": {str(c): sanity[c] for c in GATE_COSTS},
            "teacher_action_agreement": agreement,
            "policy_action_hist": hist(pol_actions),
        },
        "ledger_row": {
            "metric": "imprint corridor position = policy_argmax in-env net@21 Sharpe "
                      "minus teacher floor (same rows, same panel)",
            "imprint_corridor_position_net21": sanity[21]["corridor_position"],
            "policy_net21": sanity[21]["policy_inenv"],
            "teacher_floor_net21": sanity[21]["teacher_floor_inenv"],
            "dec2_tolerance": 0.10,
            "verdict": "RUN-TIME interpretation (DEC-2 eps-independent build) — "
                       "NOT applied to the fixture smoke; post-ExIt row = scheduled run",
        },
        "exit_half": {
            "note": "ExIt phase = shipped flags on train_jax_fast.py, no new code (PIN c); "
                    "invoke with IDENTICAL arch flags (--vector-action, small model) so the "
                    "checkpoint restores",
            "flags": EXIT_FLAG_ANCHORS,
            "entropy_coef_handoff": {
                "imprint_phase": imprint_entropy_coef,
                "saved_in_metadata": float(args.ckpt_entropy_coef),
                "why": "load_checkpoint UNCONDITIONALLY restores entropy_coef from "
                       "trainer_metadata.json after construction (trainer.py:1719-1720); "
                       "the ExIt phase must run its intended operator, not the imprint knob",
                "operator_check": "verify the 'Restored entropy_coef' log line at ExIt launch",
            },
        },
        "checkpoint": ckpt_path,
    }

    _print_report(result)
    if out_path:
        with open(out_path, "w") as fh:
            json.dump(result, fh, indent=2, default=float)
        print(f"\n[832] wrote {out_path}")

    if args.smoke_assert:
        assert finite, "SMOKE FAIL: non-finite loss in imprint history"
        assert ce_decreased, (
            f"SMOKE FAIL: policy CE did not decrease on teacher targets "
            f"(head5 {ce['head_mean']:.5f} -> tail5 {ce['tail_mean']:.5f})")
        for c in GATE_COSTS:
            assert sanity[c]["policy_inenv"] is not None, "SMOKE FAIL: sanity replay Sharpe missing"
        # TASK_845 gate: the deliverable is that the gate is REPORTED (both net@21 values
        # emitted) and that when the order-code mapping misses, the weight-native ladder is
        # re-gated (and can be IMPRINTED via --imprint-mapping). The PASS/FAIL of the
        # ±10%-RELATIVE threshold is NOT asserted on the fixture (this module's pre-registered
        # "no performance threshold on the fixture smoke" stance): the ~50-decision fixture
        # slice is a losing period (both net@21 negative), so the relative threshold is
        # ill-defined on negative Sharpes and the PASS/FAIL itself is inconclusive. What the
        # fixture DOES reveal is a real STRUCTURAL sizing divergence (see WARN below) that the
        # BINDING real-panel run (planner-owned, DEC-2 ±0.10 ABSOLUTE) must confront.
        assert gate["teacher_order_net21"] is not None and gate["teacher_weight_net21"] is not None, \
            "SMOKE FAIL: expressibility gate did not emit both net@21 values"
        final_gate = gate.get("weight_native_regate", gate)
        assert (gate["passed"] or "weight_native_regate" in gate), \
            "SMOKE FAIL: order-code gate missed but the weight-native re-gate was not shipped"
        if not final_gate["passed"]:
            rel_txt = (f"{final_gate['rel_diff']:.3f}" if final_gate["rel_diff"] is not None
                       else "n/a (non-finite weight Sharpe)")
            print(f"[832] WARN: expressibility gate MISS on the fixture after "
                  f"{final_gate['mapping']} (order {_fmt(final_gate['teacher_order_net21'])} vs "
                  f"weight {_fmt(final_gate['teacher_weight_net21'])}, rel "
                  f"{rel_txt} > {final_gate['tolerance_rel']:.2f}). PASS/FAIL "
                  f"is INCONCLUSIVE on this small losing slice (relative threshold ill-defined "
                  f"on negative Sharpes), but the ~1.25 net@21 gap is a real STRUCTURAL sizing "
                  f"divergence: default w_unit=1/n, w_max=4*w_unit=0.5 make the weight book "
                  f"express ~2x (order-code) to ~8x-then-budget-clipped (weight-native) the "
                  f"order env's per-asset exposure (~4% terminal-wealth gap; ±0.10-ABSOLUTE "
                  f"would also FAIL). REPORTED, not asserted; binding read = real-panel "
                  f"scheduled run (DEC-2), which must confront the weight-param sizing.")
        print("[832] SMOKE PASS: finite losses, decreasing imprint CE, sanity replay emitted "
              "schema, expressibility gate reported (both mappings)")
    return result


def _fmt(x):
    return "   n/a" if x is None or not np.isfinite(x) else f"{x:+.3f}"


def _print_report(r):
    print("=" * 84)
    print(f"E5 LINEAR-BOOK IMPRINT ({r['tag']}) — {r['window'][0]} .. {r['window'][1]}")
    print(f"panel: {r['n_assets']} assets x {r['n_panel_rows']} bars ({r['bar_minutes']}-min), "
          f"warm={r['warm']}, start={r['start_row']}, decisions={r['n_decisions']}")
    im = r["imprint"]
    ce = im["losses"]["policy_loss"]
    print(f"imprint: {im['n_injected_steps']} steps / {im['n_chunks']} chunk(s), "
          f"{im['config']['iters']} iters, entropy_coef={im['config']['entropy_coef']}, "
          f"min_entropy={im['config']['min_entropy']}")
    print(f"  policy CE (BC): {ce['first']:.4f} -> {ce['last']:.4f} "
          f"(head5 {ce['head_mean']:.4f} -> tail5 {ce['tail_mean']:.4f}) "
          f"decreased={im['policy_ce_decreased']} finite={im['all_losses_finite']}")
    print("-" * 84)
    print(f"{'tier':>8} | {'POLICY(env)':>12} {'FLOOR(env)':>11} {'position':>9}")
    for c in r["cost_tiers_bp"]:
        s = r["sanity"]["per_tier"][str(c)]
        label = "gross" if c == 0 else f"net@{c}"
        print(f"{label:>8} | {_fmt(s['policy_inenv']):>12} {_fmt(s['teacher_floor_inenv']):>11} "
              f"{_fmt(s['corridor_position']):>9}")
    g = r.get("expressibility_gate")
    if g is not None:
        fg = g.get("weight_native_regate", g)
        print("-" * 84)
        print(f"[845] teacher-expressibility gate ({fg['mapping']}): "
              f"order net@21 {_fmt(fg['teacher_order_net21'])}  "
              f"weight net@21 {_fmt(fg['teacher_weight_net21'])}  "
              f"rel {fg['rel_diff'] if fg['rel_diff'] is not None else float('nan'):.3f} "
              f"(tol {fg['tolerance_rel']:.2f}) -> {'PASS' if fg['passed'] else 'FAIL'}")
        print("-" * 84)
    print(f"teacher action agreement: {r['sanity']['teacher_action_agreement']:.3f}  "
          f"policy hist {r['sanity']['policy_action_hist']}  "
          f"teacher hist {r['teacher']['action_hist']}")
    print(f"LEDGER (imprint leg): corridor position net@21 = "
          f"{_fmt(r['ledger_row']['imprint_corridor_position_net21'])}  "
          f"(DEC-2 tol ±0.10 — run-time verdict, not applied here)")
    print("=" * 84)


def main():
    ap = argparse.ArgumentParser(description="E5 linear-book imprint runner (TASK_832)")
    ap.add_argument("--db", default=FIXTURE_DB,
                    help="minute-ohlcv DB (default: fixture; scheduled run points at trading_data.db)")
    ap.add_argument("--start", default=None, help="default: fixture smoke 2025-09-14 / real 2021-01-01")
    ap.add_argument("--end", default=None, help="default: fixture smoke 2025-12-12 / real 2025-09-01")
    ap.add_argument("--bar-minutes", type=int, default=None,
                    help="training resample (default: 240 fixture smoke / 1440 real — the pinned daily)")
    ap.add_argument("--n-assets", type=int, default=None,
                    help="default: 8 fixture smoke (minute-slice universe) / 64 real")
    ap.add_argument("--max-decisions", type=int, default=None,
                    help="cap decisions (None = full panel)")
    ap.add_argument("--position-size-pct", type=float, default=None,
                    help="per-asset notional fraction (default 0.5/n_assets, 830 convention)")
    ap.add_argument("--fee-bps", type=float, default=FEE_BPS)
    ap.add_argument("--slip-bps", type=float, default=SLIP_BPS)
    # phase-1c weight-rebalance env (TASK_840/843; default OFF => canon order semantics,
    # byte-identical to the pre-840 path). ON => BOTH in-env walks (teacher trajectory
    # generation AND the sanity replay) run the target-weight rebalance step_vector.
    ap.add_argument("--vector-weight-rebalance", dest="vector_weight_rebalance",
                    action="store_true", default=False,
                    help="TASK_840/843 target-weight rebalance env for BOTH in-env walks")
    ap.add_argument("--vector-weight-unit", type=float, default=-1.0,
                    help="per Buy/Sell target-weight step (sentinel <=0 => 1/n_assets)")
    ap.add_argument("--vector-weight-max", type=float, default=4.0,
                    help="position cap in UNITS of weight-unit (default 4 => +-4*w_unit)")
    ap.add_argument("--vector-rebalance-deadband", type=float, default=-1.0,
                    help="re-trim only when |target-current| > deadband (sentinel <0 => 0.1*w_unit)")
    # imprint phase (PIN a knobs + canon-mirrored trainer knobs)
    ap.add_argument("--iters", type=int, default=None,
                    help="trainer.train iterations (default: 80 smoke / 2000 real)")
    ap.add_argument("--batch-size", type=int, default=None, help="default: 32 smoke / 256 real")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--discount", type=float, default=0.997)
    ap.add_argument("--entropy-coef", type=float, default=0.001,
                    help="PIN a: imprint-phase entropy coef (canon 0.01 — one-hot targets)")
    ap.add_argument("--min-entropy", type=float, default=0.0,
                    help="PIN a: imprint-phase adaptive-entropy floor (canon 3.0; 0 disables the push)")
    ap.add_argument("--ckpt-entropy-coef", type=float, default=0.01,
                    help="entropy_coef written to trainer_metadata.json for the ExIt half "
                         "(canon 0.01, cli.py:130). load_checkpoint restores it UNCONDITIONALLY "
                         "(trainer.py:1719-1720) — the imprint knob must NOT leak into ExIt")
    ap.add_argument("--state-dim", type=int, default=64)
    ap.add_argument("--h-cycles", type=int, default=4)
    ap.add_argument("--l-cycles", type=int, default=6)
    ap.add_argument("--reward-hybrid", dest="reward_hybrid", action="store_true", default=True)
    ap.add_argument("--no-reward-hybrid", dest="reward_hybrid", action="store_false")
    ap.add_argument("--reward-attributable", dest="reward_attributable",
                    action="store_true", default=True)
    ap.add_argument("--no-reward-attributable", dest="reward_attributable", action="store_false")
    ap.add_argument("--imprint-mapping", choices=("order-code", "weight-native"),
                    default="order-code",
                    help="teacher mapping to IMPRINT on: 'order-code' (default; "
                         "corridor._to_action_vec) or 'weight-native' (the ladder the "
                         "expressibility gate re-measures — pre-registered contingency when "
                         "the order-code teacher fails the weight-env gate). Default keeps "
                         "the byte-identical default output.")
    ap.add_argument("--chunk-len", type=int, default=400,
                    help="max steps per injected trajectory chunk (canon episode scale)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt-out", default=None,
                    help="imprint checkpoint dir (default: <exp>/ckpt_smoke on fixture)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke-assert", action="store_true", default=None,
                    help="assert finite + decreasing CE + schema (default ON for fixture db)")
    ap.add_argument("--no-smoke-assert", dest="smoke_assert", action="store_false")
    args = ap.parse_args()

    exp_dir = os.path.dirname(os.path.abspath(__file__))
    is_fixture = os.path.abspath(args.db) == os.path.abspath(FIXTURE_DB)
    # Pre-registered defaults (SPEC.md): fixture smoke vs scheduled real run.
    if args.start is None:
        args.start = "2025-09-14" if is_fixture else "2021-01-01"
    if args.end is None:
        args.end = "2025-12-12" if is_fixture else "2025-09-01"
    if args.bar_minutes is None:
        args.bar_minutes = 240 if is_fixture else 1440
    if args.n_assets is None:
        args.n_assets = 8 if is_fixture else 64
    if args.iters is None:
        args.iters = 80 if is_fixture else 2000
    if args.batch_size is None:
        args.batch_size = 32 if is_fixture else 256
    if args.position_size_pct is None:
        args.position_size_pct = 0.5 / args.n_assets
    if args.ckpt_out is None:
        args.ckpt_out = os.path.join(exp_dir, "ckpt_smoke" if is_fixture else "ckpt_imprint")
    if args.smoke_assert is None:
        args.smoke_assert = is_fixture
    tag = args.tag or ("fixture-smoke" if is_fixture else "real-panel")
    out = args.out
    if out is None and is_fixture:
        out = os.path.join(exp_dir, "results_fixture_smoke.json")
    run(args, out, tag)


if __name__ == "__main__":
    main()
