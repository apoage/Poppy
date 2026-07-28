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

TASK_861 (D26) — ``--teacher-vectors <file>`` swaps the TARGET STREAM for the action vectors
an offline SEARCH teacher chose (recorded by ``experiments/835_stage4_eval/eval_vector.py
--collect-chosen-vectors``), leaving the recipe above untouched: the same in-env trajectory
generation, the same one-hot per-asset marginals, the same factored-CE BC loss, optimizer,
iters and sanity replay. A fail-loud gate refuses any collection that (a) differs in db /
window / panel frame / decision rows / EnvConfig fingerprint, (b) was not recorded with the
PINNED +8.024 teacher operator + checkpoint (``TV_OPERATOR_PIN``; escape hatch
``--allow-off-pin-teacher`` stamps ``operator_off_pin: true``), (c) fails its payload sha256 or
carries non-integer action codes, or (d) does not replay in-env to the walk score the collection
recorded for itself. Default OFF = byte-identical to the shipped module (proven by the canon
pins in ``experiments/861_search_teacher_imprint/smoke_e2e.py``).

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
HOLD_ACTION_CODE = 1                    # == eval_vector.HOLD; {0,3}=short {2,4}=long

# ---- TASK_861: offline search-teacher target stream (--teacher-vectors) --------------------
# Tolerance on the payload-vs-header action-composition cross-check (percentage POINTS of the
# aggregate HOLD fraction). The quantity is exact arithmetic on both sides -- 835 computes
# 100*hold_count.sum()/(n_exec*n_assets) and the gate recomputes 100*mean(payload == HOLD) -- so
# this only absorbs an early-terminated-walk edge case, never a real rewrite. The attack this
# catches moved it 76.75% -> 11.50%, i.e. 65 percentage points.
TV_COMPOSITION_TOL_PCT = 0.5
EVAL_PY = os.path.join(_REPO, "experiments", "835_stage4_eval", "eval_vector.py")
TEACHER_VECTORS_SCHEMA = "861/chosen-vectors/v1"
# Falsification floor (acceptance iv), pinned. The gated quantity is a TRAIN-SET FIT against the
# NEVER-SHUFFLED targets — the 860 lesson: a small-n held-out metric inverts per seed at fixture
# scale, so the gate must be high-SNR. NLL reduction vs the best CONSTANT predictor (the target
# marginal's entropy) over n = N x n_assets label cells is that quantity: a legit imprint drives
# it well positive; a shuffled-target imprint fits a DIFFERENT labeling and cannot beat the
# marginal on the original one.
#
# FLOOR PLACEMENT (fix pass FIX 5 — re-measured at n=10 seeds, fixture panel, 80 iters; the
# earlier 0.10 pin was placed off n=4 and claimed a 0.094 legit margin the wider sweep does not
# support). Measured separation, seeds 0-9 (both arms, same collection):
#   legit    +0.3618 +0.1944 +0.2400 +0.3159 +0.2532 +0.1248 +0.2927 +0.3010 +0.2267 +0.3386
#            min +0.1248  max +0.3618  mean +0.2649  sd 0.0717
#   shuffled -0.2134 -0.1160 -0.1609 -0.1701 -0.1477 -0.0799 -0.1569 -0.1551 -0.2265 -0.1842
#            min -0.2265  max -0.0799  mean -0.1611  sd 0.0428
# gap between the clouds 0.2047, Cohen's d 7.2 — the SEPARATION is excellent; only the floor
# placement was wrong. At 0.10 the legit margin is 0.0248 (0.35 sd => ~1% false-FAIL per seed,
# ~4% over a 4-seed sweep). 0.05 sits midway between the legit cloud's floor and the 0.0
# boundary a constant marginal predictor can never beat: 0.0748 (1.04 sd) of headroom on the
# legit side, 0.1299 (3.04 sd) on the sabotage side. A gate that fails legitimately 1-in-25
# sweeps is a worse instrument than one placed where neither arm can reach it.
TEACHER_VECTORS_NLL_REDUCTION_FLOOR = 0.05
# ---- PINNED TEACHER OPERATOR (fix pass FIX 1) ----------------------------------------------
# TASK_861: "Any deviation is a different experiment and must fail loud." The provenance gate
# used to compare db/window/frame/rows/env but NEVER the operator, so a --sims 8 collection, a
# policy-prior/model-leaf collection and a hold-bias-0 collection all imprinted clean with
# `provenance gate PASS` and exit 0. The falsification gate cannot catch this — it only measures
# whether BC memorised whatever labels it was handed (it scored a -7.08 garbage teacher HIGHER
# than the pinned +8.02 one). So the operator block is PINNED here, verbatim from
# runs/832_e5/results_e5inv854_db_pw0_832_e5__ckpt_imprint_full.json (net@21 +8.0236).
TV_OPERATOR_PIN = {
    "mode": "search",
    "proposal_scheme": "dithered-book",
    "vector_k": 16,
    "sims": 32,
    "hold_bias": 3.0,
    "dither_scale": 1.0,
    "proposal_temp": 1.0,
    "proposal_explore_slots": 2,
    "prior_weight": 0.0,
    "leaf_eval": "true-env",
    "leaf_rollout_r": 3,
    "leaf_vhat_tail": False,
    "gumbel_scale": 3.0,
    "deterministic": True,
    "per_node_resampling": False,
    "vector_budget_step": False,
    "vector_budget_pct": 0.99,
    "vector_weight_rebalance": True,
    "vector_weight_unit": -1.0,
    "vector_weight_max": 4.0,
    "vector_rebalance_deadband": -1.0,
    "model_size": "small",
    "h_cycles": 4,
    "l_cycles": 6,
    "discount": 0.997,
}
# The checkpoint the +8.024 read was taken on (compared by normalized path suffix).
TV_CHECKPOINT_PIN = "runs/832_e5/ckpt_imprint_full"
# Relative tolerance on the recorded-vs-replayed walk score (fix pass FIX 2). 835 marks PV via a
# BATCHED jax.vmap(compute_portfolio_value) while this module's walk_env calls it unbatched, so
# the float32 reduction order differs; the measured legitimate residual is 1.9e-06 relative
# (1 float32 ulp). 1e-4 is ~50x headroom — anything beyond it is a DIFFERENT trajectory.
TV_WALK_SCORE_REL_TOL = 1e-4
# EnvConfig fields the provenance gate does NOT require to match, each with its reason. Every
# other field must be EQUAL or the collection is refused. A value of `True` means unconditional;
# a callable(want_env, got_env) -> bool makes the exemption CONDITIONAL on the surrounding config
# (fix pass FIX 4) so the exemption cannot quietly widen into configurations where the field bites.
TV_ENV_EXEMPT = {
    "max_steps": (True,
                  "episode-length knob only: BOTH walkers override max_episode_steps on the "
                  "STATE (835 _finalize_panel / this module's walk_env), so the config value is "
                  "inert for the replay"),
    "reward_attributable": (
        True,
        "TRAJECTORY-inert but NOT consequence-free. It is a reward-only term "
        "(jax_trading_env.py:1854): it never touches the state transition, so the recorded "
        "action vectors replay to the identical PV path and the identical teacher floor. The 835 "
        "harness exposes no flag for it (its panel is always False) while this runner defaults "
        "True, so a hard refusal would block the pre-registered real leg. CONSEQUENCE, recorded "
        "because the WARN fires on 100% of legitimate pairings and a warning that always fires "
        "carries no signal: inject_teacher turns THIS run's reward stream into the MC value "
        "targets, so the student's value/reward heads are fit under a reward definition the "
        "search teacher never optimised. The POLICY targets (the BC objective this task is "
        "about) are unaffected. Pass --no-reward-attributable to align the two definitions."),
    "position_size_pct": (
        lambda want, got: bool(want.get("vector_weight_rebalance")) and
                          bool(got.get("vector_weight_rebalance")),
        "CONDITIONAL on vector_weight_rebalance being true on BOTH sides (refused otherwise). "
        "Under weight-rebalance, step_vector overwrites cash/positions wholesale "
        "(jax_trading_env.py:1791-1798), limit legs are sized w_unit*PV (:1785) reserving 0 "
        "(:1561), so ideal_trade_value = PV*position_size_pct (:1418) is DEAD CODE. Verified "
        "empirically at fixture scale (fix pass): the pinned operator walked at "
        "--position-size-pct 0.1 vs 0.0625 gives 1.000 chosen-vector agreement, 1.000 "
        "realized-book agreement, net@21 +13.89300020243101 BOTH and final_pv 107485.84375 "
        "BOTH, with position_size_pct the ONLY differing env field. The same pair walked with "
        "weight-rebalance OFF diverges (net@21 +17.2151 vs +17.1620, final_pv 109773.20 vs "
        "106052.72) — which is exactly why the exemption is CONDITIONAL. Needed because the "
        "teacher of record records 0.1 while the pinned "
        "imprint manifest (runs/832_e5/jobs_imprint_full.manifest) uses 0.5/64 = 0.0078125 — "
        "without this the two pre-registered real legs cannot run together."),
}

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


_EVAL_MOD = None


def _load_eval_module():
    """Import experiments/835_stage4_eval/eval_vector.py by path (cached) for its READ-ONLY
    loader mirror ``load_row_timestamps`` — the panel's row-timestamp axis reconstructed with
    the training loader's own queries. Reused, not re-derived (835 imports THIS module by the
    same pattern; both imports happen inside functions, so there is no import cycle). ONLY the
    --teacher-vectors provenance gate calls this: the canon path never touches 835."""
    global _EVAL_MOD
    if _EVAL_MOD is None:
        spec = importlib.util.spec_from_file_location("eval_vector835", EVAL_PY)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _EVAL_MOD = mod
    return _EVAL_MOD


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


# =============================================================================
# TASK_861 — offline search-teacher target stream (--teacher-vectors)
# =============================================================================
# D26 ruled: the best teacher ever measured is the true-env-leaf SEARCH (+8.024 net@21,
# LEDGER 832/E5-inversion3) but its leaf is an R-step LOOKAHEAD, so it can never trade live;
# the best DEPLOYABLE recipe is this module's BC imprint (+0.839 in-sample / +0.898 OOS) but
# its teacher was only the ~floor-grade 819 trend book (+0.542). --teacher-vectors marries them:
# the search runs ONCE offline (835 eval_vector.py --collect-chosen-vectors) and the vectors it
# CHOSE become the hard per-step targets of the EXACT shipped recipe below. Everything else —
# weight-native mapping, one-hot marginals, factored CE, optimizer, iters, sanity replay — is
# untouched. This is NOT ExIt2's visit-distribution CE (whose per-asset argmax rounds to 100%
# HOLD, LEDGER 832/E5-exit2): the targets are the same semantic object this module already
# imprints successfully, one weight vector per decision.

def load_teacher_vectors(path):
    """Read + schema-validate a 861 chosen-vectors collection. Fail-loud on anything the
    provenance gate could not otherwise interpret (missing file, wrong schema, absent blocks)."""
    if not str(path).strip():
        raise RuntimeError("--teacher-vectors was given an EMPTY path.")
    if not os.path.exists(path):
        raise RuntimeError(
            f"--teacher-vectors: collection file not found: {path}. Produce it with "
            f"`eval_vector.py --collect-chosen-vectors <path>` on the SAME db/window/panel.")
    try:
        with open(path) as fh:
            doc = json.load(fh)
    except (ValueError, OSError) as e:
        raise RuntimeError(f"--teacher-vectors: cannot read {path} ({type(e).__name__}: {e}).")
    if not isinstance(doc, dict) or doc.get("schema") != TEACHER_VECTORS_SCHEMA:
        raise RuntimeError(
            f"--teacher-vectors: {path} is not a '{TEACHER_VECTORS_SCHEMA}' collection "
            f"(schema={doc.get('schema') if isinstance(doc, dict) else type(doc).__name__!r}). "
            f"Refusing to guess the layout of an unknown target stream.")
    for block in ("provenance", "decisions"):
        if not isinstance(doc.get(block), dict):
            raise RuntimeError(f"--teacher-vectors: {path} has no '{block}' object.")
    # FIX 2 (fix pass) — PAYLOAD INTEGRITY. The gate used to trust the payload and cast it with
    # np.asarray(..., dtype=np.int64), which silently TRUNCATES non-integer codes toward zero:
    # rewriting every code as x-0.5 passed shape+range and produced a teacher worth -0.649 in
    # place of one worth +13.893, with two green gates. Recompute the producer's sha256 over the
    # canonical payload; a mismatch means the stream is not the stream that was recorded.
    want_sha = doc["provenance"].get("payload_sha256")
    if not want_sha:
        raise RuntimeError(
            f"--teacher-vectors: {path} carries no provenance.payload_sha256 — it predates the "
            f"integrity header. Re-collect with the current eval_vector.py "
            f"--collect-chosen-vectors; an unverifiable target stream is not imprintable.")
    ev = _load_eval_module()
    got_sha = ev.payload_digest(doc["decisions"])
    if got_sha != want_sha:
        raise RuntimeError(
            f"--teacher-vectors PAYLOAD DIGEST MISMATCH: {path}\n"
            f"  recorded sha256 {want_sha}\n  recomputed      {got_sha}\n"
            f"  The decisions payload was modified after collection. Refusing to imprint a "
            f"tampered/corrupted target stream.")
    return doc


def gate_teacher_vectors(doc, path, args, tier_cfg, start_row, N, T):
    """FAIL-LOUD provenance gate (task scope 2). The collected target stream may only be
    imprinted when the imprint run reproduces the EXACT panel and env the walk was recorded on:
    db, window, bar/universe frame, decision rows (index AND timestamp), and the whole EnvConfig
    fingerprint minus the documented ``TV_ENV_EXEMPT`` fields. Every mismatch is collected and
    reported together, then the run ABORTS — a silently-misaligned teacher is a wrong-experiment
    result, not a crash. Returns the provenance summary for the artifact.

    Fix pass: the gate ALSO pins the teacher OPERATOR and CHECKPOINT (FIX 1 —
    ``TV_OPERATOR_PIN`` / ``TV_CHECKPOINT_PIN``; TASK_861 line 35 "any deviation is a different
    experiment and must fail loud") and validates the payload's integrality (FIX 2). Only
    ``--allow-off-pin-teacher`` downgrades the pin to a loud, RECORDED deviation."""
    ev = _load_eval_module()
    p = doc["provenance"]
    d = doc["decisions"]
    mism = []

    def chk(name, want, got, hint=""):
        if want != got:
            mism.append(f"    {name}: collection={want!r}  imprint={got!r}"
                        f"{('  -> ' + hint) if hint else ''}")

    # --- db / window / bar / universe size ---
    chk("db", p.get("db_abspath"), os.path.abspath(args.db))
    chk("window.start", (p.get("window") or [None, None])[0], args.start,
        "pass the SAME --start the collection walk used")
    chk("window.end", (p.get("window") or [None, None])[1], args.end,
        "pass the SAME --end the collection walk used")
    chk("bar_minutes", p.get("bar_minutes"), int(args.bar_minutes))
    chk("n_assets", p.get("n_assets"), int(args.n_assets))

    # --- panel FRAME: this runner is minute-sourced only (create_jax_env_historical) ---
    if p.get("panel_source") != "minute":
        mism.append(f"    panel_source: collection={p.get('panel_source')!r}  imprint='minute'"
                    f"  -> this runner builds its panel through create_jax_env_historical (the "
                    f"1-min ohlcv table, PIN b); it cannot reproduce a daily/framed panel")
    if p.get("panel_frame_from_checkpoint") or p.get("panel_frame"):
        mism.append(f"    panel_frame: collection=fixed-frame "
                    f"({p.get('panel_frame_from_checkpoint')!r})  imprint=coverage-ranked  -> "
                    f"the 853 fixed-frame panel has no counterpart here (slot identity would "
                    f"not match)")
    row_ts, loaded_syms = ev.load_row_timestamps(
        args.db, args.start, args.end, None, int(args.n_assets), int(args.bar_minutes))
    chk("loaded_symbols", p.get("loaded_symbols"), list(loaded_syms),
        "the universes differ (slot k IS asset k positionally) — the target vectors would be "
        "applied to the wrong assets")
    chk("padded", p.get("padded"), bool(len(loaded_syms) < int(args.n_assets)))
    chk("n_rows", p.get("n_rows"), int(T))

    # --- decision rows: index AND wall-clock timestamp ---
    chk("start_row", p.get("start_row"), int(start_row),
        "the first imprinted decision row differs (this runner starts at "
        "max(book_warm, window_size); 835 starts at window_size)")
    n_coll = int(p.get("n_decisions_collected") or 0)
    chk("n_decisions", n_coll, int(N))
    if p.get("partial"):
        mism.append("    partial: the collection walk terminated EARLY (env done / bankruptcy) "
                    "-> the stream does not cover the full decision range")
    # NIT (fix pass): the payload's own row_index must agree with the recorded start_row — a
    # +5-shifted index block used to pass because nothing ever read it.
    want_rows = list(range(int(p.get("start_row") or -1),
                           int(p.get("start_row") or -1) + n_coll))
    if [int(r) for r in (d.get("row_index") or [])] != want_rows:
        mism.append(f"    decisions.row_index: does not equal "
                    f"range(provenance.start_row={p.get('start_row')}, +{n_coll}) -> the payload "
                    f"is indexed against different panel rows than the header claims")
    # FIX 6 (fix pass): timestamps_verified used to be hardcoded True even when this comparison
    # was skipped. Run it whenever the loader mirror can address the rows, record what actually
    # happened, and REFUSE when it cannot (a claim we did not check is worse than a refusal).
    ts_checked = False
    if len(row_ts) != T:
        mism.append(f"    loader-mirror row count: eval_vector.load_row_timestamps returned "
                    f"{len(row_ts)} rows but the training panel has T={T} -> the decision "
                    f"timestamps cannot be verified against this panel "
                    f"(835 _finalize_panel raises on the same condition)")
    else:
        ts_checked = True
        want_ts = [int(t) for t in row_ts[start_row:start_row + N]]
        got_ts = [int(t) for t in (d.get("timestamp") or [])]
        if want_ts != got_ts:
            k = next((i for i in range(min(len(want_ts), len(got_ts)))
                      if want_ts[i] != got_ts[i]), 0)
            mism.append(f"    decision timestamps: first divergence at k={k} "
                        f"(collection={got_ts[k] if k < len(got_ts) else None}, "
                        f"imprint={want_ts[k] if k < len(want_ts) else None}) -> the two runs "
                        f"do NOT walk the same rows in time")

    # --- cost tier + full EnvConfig fingerprint ---
    net_tier = GATE_COSTS[-1]
    chk("cost_tier_bp", p.get("cost_tier_bp"), int(net_tier),
        f"this runner imprints the net@{net_tier} trajectory")
    want_env = ev.env_flag_block(tier_cfg)
    got_env = p.get("env_flags") or {}
    exempt_notes = []
    exempt_applied = []
    for k in sorted(set(want_env) | set(got_env)):
        rule = TV_ENV_EXEMPT.get(k)
        if rule is not None:
            cond, why = rule
            active = cond if cond is True else bool(cond(want_env, got_env))
            if not active:
                # FIX 4: a CONDITIONAL exemption whose precondition does not hold is no
                # exemption at all — fall through to the hard comparison.
                chk(f"env_flags.{k}", got_env.get(k), want_env.get(k),
                    f"exemption NOT applicable here: {why}")
                continue
            exempt_applied.append(k)
            if got_env.get(k) != want_env.get(k):
                exempt_notes.append(
                    f"{k}: collection={got_env.get(k)!r} imprint={want_env.get(k)!r} ({why})")
            continue
        chk(f"env_flags.{k}", got_env.get(k), want_env.get(k))

    # --- PINNED TEACHER OPERATOR + CHECKPOINT (fix pass FIX 1) ---
    op = p.get("operator") or {}
    off_pin = []
    for k in sorted(TV_OPERATOR_PIN):
        if k not in op:
            off_pin.append(f"    operator.{k}: MISSING from the collection header "
                           f"(pin={TV_OPERATOR_PIN[k]!r})")
        elif op[k] != TV_OPERATOR_PIN[k]:
            off_pin.append(f"    operator.{k}: collection={op[k]!r}  "
                           f"PIN={TV_OPERATOR_PIN[k]!r}")
    ck = str(p.get("checkpoint") or "")
    ck_norm = os.path.normpath(ck).replace(os.sep, "/")
    if not (ck_norm == TV_CHECKPOINT_PIN or ck_norm.endswith("/" + TV_CHECKPOINT_PIN)):
        off_pin.append(f"    checkpoint: collection={ck!r}  PIN={TV_CHECKPOINT_PIN!r}")
    # FIX 10: do NOT raise here. `mism` (window / frame / env / payload) is populated BELOW, and
    # raising first discarded all of it -- so a collection that was both off-pin AND on the wrong
    # window reported only the off-pin block. That is the worst possible ordering: the hidden
    # failure (re-flag, free) masked the reported one (re-collect, multi-hour). Both are collected
    # and raised together at the single refusal site below.
    off_pin_fatal = bool(off_pin) and not args.allow_off_pin_teacher

    # --- the vectors themselves ---
    # FIX 2c: dtype/integrality FIRST. The old np.asarray(..., dtype=np.int64) silently
    # truncated non-integer codes toward zero (every code written as x-0.5 passed shape+range
    # and swapped a +13.893 teacher for a -0.649 one).
    raw = d.get("chosen_vectors")
    vecs = None
    try:
        fvec = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as e:
        fvec = None
        mism.append(f"    chosen_vectors: not a numeric array ({type(e).__name__}: {e})")
    if fvec is not None and fvec.ndim != 2:
        mism.append(f"    chosen_vectors shape: got {fvec.shape}, expected "
                    f"({n_coll}, {int(args.n_assets)})")
    elif fvec is not None:
        if fvec.shape != (n_coll, int(args.n_assets)):
            mism.append(f"    chosen_vectors shape: got {fvec.shape}, expected "
                        f"({n_coll}, {int(args.n_assets)})")
        elif not np.all(np.isfinite(fvec)):
            mism.append("    chosen_vectors: contains non-finite entries (NaN/Inf)")
        elif not np.array_equal(fvec, np.round(fvec)):
            bad = int(np.argmax(fvec != np.round(fvec)))
            mism.append(f"    chosen_vectors: NON-INTEGER action code at flat index {bad} "
                        f"(value {fvec.ravel()[bad]!r}) -> action codes are integers 0..4; a "
                        f"float payload would be silently truncated toward zero")
        else:
            vecs = fvec.astype(np.int64)
            if vecs.min() < 0 or vecs.max() >= N_ACTIONS_PER_ASSET:
                mism.append(f"    chosen_vectors range: [{vecs.min()}, {vecs.max()}] outside "
                            f"[0, {N_ACTIONS_PER_ASSET - 1}]")

    # --- the recorded walk score must be present (FIX 2a; replayed + asserted by the caller) ---
    if p.get("walk_net21") is None:
        mism.append("    walk_net21: the collection header carries no recorded walk score -> "
                    "the replay-integrity check (recorded vs re-walked teacher floor) cannot "
                    "run. Re-collect with the current eval_vector.py.")

    # --- FIX 8: the payload's own ACTION COMPOSITION must match what the walk recorded ---------
    # The replay check (_tv_replay_check) is a TRAJECTORY check, not a TARGET check. Under the
    # pinned weight-rebalance env the codes {Hold, LimitSell, LimitBuy} are mutually
    # trajectory-inert at most cells: a verification sweep found 315 of 400 fixture cells (78.8%)
    # admit an inert substitution, and rewriting all of them jointly left the PV path,
    # realized_book and net@21 BIT-IDENTICAL — so every gate passed while 78.8% of the BC targets
    # (the actual object of this experiment) had been replaced. The walk already commits to its
    # own hold fraction in the header; recomputing it from the payload closes that hole for free.
    rec_hold = p.get("walk_pct_hold_agg")
    if rec_hold is None:
        mism.append("    walk_pct_hold_agg: the collection header carries no recorded action "
                    "composition -> the payload's targets cannot be cross-checked against the "
                    "walk that produced them (a trajectory-inert target rewrite would pass every "
                    "other gate). Re-collect with the current eval_vector.py.")
    elif vecs is not None:
        payload_hold = 100.0 * float(np.mean(vecs == HOLD_ACTION_CODE))
        if abs(payload_hold - float(rec_hold)) > TV_COMPOSITION_TOL_PCT:
            mism.append(
                f"    action composition: the payload is {payload_hold:.4f}% HOLD but the "
                f"collection walk recorded {float(rec_hold):.4f}% (tol "
                f"{TV_COMPOSITION_TOL_PCT}pp). The targets in this file are NOT the actions that "
                f"walk chose. NB codes {{Hold, LimitSell, LimitBuy}} are largely "
                f"trajectory-inert under weight-rebalance, so a rewrite among them can reproduce "
                f"the recorded score exactly while changing most of the BC labels — which is "
                f"precisely what this check exists to catch.")

    if mism or off_pin_fatal:
        parts = []
        if off_pin_fatal:
            parts.append(
                "OFF-PIN TEACHER REFUSAL (TASK_861 line 35: \"Any deviation is a different "
                "experiment and must fail loud\") — this collection was NOT recorded with the "
                "+8.024 teacher operator of record:\n" + "\n".join(off_pin) +
                "\n  The falsification gate CANNOT catch this — it only measures whether BC "
                "memorised whatever labels it was handed (it scores a garbage teacher just as "
                "high). Re-collect with the pinned operator, or pass --allow-off-pin-teacher to "
                "run a deliberately off-pin experiment (recorded as operator_off_pin: true).")
        if mism:
            parts.append(
                "PROVENANCE REFUSAL (scope 2) — panel/env/payload not recorded on this imprint "
                "run's panel/env:\n" + "\n".join(mism) +
                "\n  Fix the imprint flags (or re-collect on this panel) — imprinting a "
                "misaligned teacher would train on targets from other rows/assets/semantics "
                "and produce a silently wrong result.")
        raise RuntimeError(
            f"--teacher-vectors REFUSAL: the collected target stream\n  {path}\ncannot be used. "
            f"{len(parts)} independent problem(s) found — ALL are reported so the cheap fix is "
            f"never hidden behind the expensive one:\n\n" + "\n\n".join(parts))

    if off_pin:
        print("!" * 84)
        print("!!! [861] OFF-PIN TEACHER (--allow-off-pin-teacher): this collection was NOT "
              "recorded with")
        print("!!! the +8.024 operator of record. The result is a DIFFERENT EXPERIMENT and is "
              "stamped")
        print("!!! operator_off_pin: true in the artifact. Deviating fields:")
        for line in off_pin:
            print(f"!!! {line.strip()}")
        print("!" * 84)
    for note in exempt_notes:
        print(f"[861] WARN (exempt field, NOT a refusal): {note}")
    print(f"[861] provenance gate PASS: {n_coll} decisions, rows {start_row}.."
          f"{start_row + n_coll - 1}, {args.n_assets} slots, net@{net_tier}, "
          f"panel/env fingerprint matched ({len(want_env)} EnvConfig fields, "
          f"{len(exempt_applied)} exempt: {exempt_applied}); operator pin "
          f"{'OFF-PIN (allowed)' if off_pin else 'MATCHED'}; payload sha256 OK; "
          f"timestamps_verified={ts_checked}")
    return {
        "path": os.path.relpath(os.path.abspath(path), _REPO),
        "schema": doc.get("schema"),
        "collection_provenance": p,
        "gate": {
            "passed": True,
            "n_decisions": n_coll,
            "start_row": int(start_row),
            "env_fields_compared": sorted(set(want_env) - set(exempt_applied)),
            "env_fields_exempt": {k: TV_ENV_EXEMPT[k][1] for k in exempt_applied},
            "env_exempt_mismatches": exempt_notes,
            "timestamps_verified": bool(ts_checked),
            "row_index_verified": True,
            "payload_sha256_verified": True,
            "payload_integrality_verified": True,
            # FIX 1: the operator/checkpoint pin verdict, always recorded.
            "operator_pin": dict(TV_OPERATOR_PIN),
            "checkpoint_pin": TV_CHECKPOINT_PIN,
            "operator_off_pin": bool(off_pin),
            "operator_off_pin_fields": [ln.strip().split(":", 1)[0] for ln in off_pin],
            "operator_off_pin_detail": [ln.strip() for ln in off_pin],
            "allow_off_pin_teacher_flag": bool(args.allow_off_pin_teacher),
        },
        # The recorded walk score the caller replays and asserts against (FIX 2b).
        "recorded_walk": {
            "net21": p.get("walk_net21"), "final_pv": p.get("walk_final_pv"),
            "bankrupt_at": p.get("walk_bankrupt_at"),
            "steps_completed": p.get("walk_steps_completed"),
        },
    }


def _tv_replay_check(tv_prov, replayed_net21, shuffled):
    """FIX 2b — assert the replayed in-env teacher floor reproduces the score the COLLECTION
    WALK recorded for itself (``provenance.walk_net21``), within ``TV_WALK_SCORE_REL_TOL``.

    This is the highest-value integrity check in the pass. The two gates that existed before
    could both be green on a stream recorded by a completely different operator or on a payload
    corrupted after the fact — neither compares the stream against any number the producer
    committed to. This does: the same actions, the same rows, the same env, walked by two
    different modules must score the same. The legitimate two-engine residual is 1 float32 ulp
    (1.9e-06 relative, measured), so the tolerance carries ~50x headroom.

    On a ``--shuffle-teacher-vectors`` sabotage run the stream is DELIBERATELY a different one,
    so the check is skipped (loudly) and recorded as skipped."""
    recorded = tv_prov["recorded_walk"]["net21"]
    if shuffled:
        print(f"[861] replay-integrity check SKIPPED on the sabotage leg (the permuted stream "
              f"is a different trajectory by construction): recorded walk net@21 "
              f"{recorded:+.6f}, permuted replay {replayed_net21:+.6f}")
        return {"checked": False, "reason": "shuffled sabotage leg",
                "recorded_walk_net21": recorded,
                "replayed_floor_net21": float(replayed_net21),
                "rel_tolerance": TV_WALK_SCORE_REL_TOL}
    # FIX 9: check BOTH sides for finiteness. Checking only `replayed` left the guard FAIL-OPEN:
    # with a non-finite RECORDED score, `delta`/`rel` are NaN, `rel >= tol` is False, and the
    # load-bearing integrity check printed PASS on `recorded=+nan`.
    if recorded is None or not np.isfinite(float(recorded)):
        raise RuntimeError(
            f"--teacher-vectors REPLAY INTEGRITY FAILURE: the collection header's recorded walk "
            f"score is not a finite number ({recorded!r}), so the replayed floor "
            f"({float(replayed_net21):+.6f}) cannot be checked against anything. Refusing to "
            f"imprint on a stream with no verifiable score.")
    if not np.isfinite(replayed_net21):
        raise RuntimeError(
            f"--teacher-vectors REPLAY INTEGRITY FAILURE: replaying the collected stream in "
            f"this env produced a non-finite net@21 Sharpe ({replayed_net21}) while the "
            f"collection recorded {float(recorded):+.6f}.")
    delta = abs(float(replayed_net21) - float(recorded))
    rel = delta / max(abs(float(recorded)), 1.0)
    if rel >= TV_WALK_SCORE_REL_TOL:
        raise RuntimeError(
            f"--teacher-vectors REPLAY INTEGRITY FAILURE: the collected stream does NOT "
            f"reproduce the score its own collection walk recorded.\n"
            f"  recorded provenance.walk_net21 {recorded:+.9f}\n"
            f"  replayed in-env teacher floor  {float(replayed_net21):+.9f}\n"
            f"  |delta| {delta:.3e}  rel {rel:.3e}  >= tol {TV_WALK_SCORE_REL_TOL:.0e}\n"
            f"  The two-engine float32 residual is ~2e-06 relative; a gap this large means the "
            f"stream, the panel or the env is not the one that was scored. Refusing to imprint.")
    print(f"[861] replay-integrity PASS: recorded walk net@21 {recorded:+.6f} vs replayed "
          f"in-env floor {float(replayed_net21):+.6f} (|d| {delta:.2e}, rel {rel:.2e} < tol "
          f"{TV_WALK_SCORE_REL_TOL:.0e})")
    return {"checked": True, "recorded_walk_net21": recorded,
            "replayed_floor_net21": float(replayed_net21),
            "abs_delta": delta, "rel_delta": rel,
            "rel_tolerance": TV_WALK_SCORE_REL_TOL, "passed": True}


def policy_action_log_marginals(model, params, obs_np, n_assets, block=128):
    """Per-asset action LOG-marginals (N, A, 5) from the same batched ``initial_inference``
    pass ``policy_argmax_actions`` uses (log_softmax over the 5 per-asset actions)."""
    @jax.jit
    def fwd(o):
        out = model.apply({'params': params}, o, mode='initial_inference')
        logits = out['policy_logits'].reshape(o.shape[0], n_assets, N_ACTIONS_PER_ASSET)
        return jax.nn.log_softmax(logits, axis=-1)
    out = []
    for i in range(0, obs_np.shape[0], block):
        lp = fwd(jnp.asarray(obs_np[i:i + block]))
        jax.block_until_ready(lp)
        out.append(np.asarray(lp, dtype=np.float64))
    return np.concatenate(out, axis=0)


def teacher_vectors_fit(logp, targets_original):
    """The falsification gate quantity (acceptance iv), measured on the TRAIN SET against the
    NEVER-SHUFFLED targets. The 860 lesson: a small-n held-out metric inverts per seed at
    fixture scale, so gate on a high-SNR FIT instead — here n = N x n_assets label cells.

    ``nll_reduction`` = 1 - NLL(policy | original targets) / NLL(best CONSTANT predictor), the
    constant predictor being the target action MARGINAL (its NLL is the marginal's entropy).
    A legit imprint (trained on exactly these targets) drives it well positive; a shuffled-target
    imprint fits a DIFFERENT labeling and cannot beat the marginal on the original one — and
    unlike raw agreement, this cannot be won by collapsing onto the modal action."""
    idx = np.asarray(targets_original, dtype=np.int64)
    nll = float(-np.mean(np.take_along_axis(logp, idx[:, :, None], axis=2)))
    cnt = np.bincount(idx.ravel(), minlength=N_ACTIONS_PER_ASSET).astype(np.float64)
    p = cnt / max(cnt.sum(), 1.0)
    nz = p[p > 0]
    nll_marginal = float(-np.sum(nz * np.log(nz)))
    reduction = float((nll_marginal - nll) / max(nll_marginal, 1e-12))
    argmax = logp.argmax(axis=-1).astype(np.int64)
    agree = float(np.mean(argmax == idx))
    modal = float(p.max())
    return {
        "n_label_cells": int(idx.size),
        "nll_vs_original_targets": nll,
        "nll_marginal_baseline": nll_marginal,
        "nll_reduction": reduction,
        "nll_reduction_floor": TEACHER_VECTORS_NLL_REDUCTION_FLOOR,
        "argmax_agreement_vs_original": agree,
        "modal_action_rate": modal,
        "argmax_excess_over_modal": float(agree - modal),
        "target_action_marginal": {str(i): float(p[i]) for i in range(N_ACTIONS_PER_ASSET)},
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

    # ---- TASK_861 target source: the offline search-teacher's chosen vectors ----
    # Loaded + provenance-gated BEFORE any walk so a misaligned stream aborts in seconds.
    tv_doc = tv_prov = tv_replay = None
    tv_actions = tv_actions_original = None
    if args.teacher_vectors is not None:
        tv_doc = load_teacher_vectors(args.teacher_vectors)
        tv_prov = gate_teacher_vectors(tv_doc, args.teacher_vectors, args,
                                       tier_cfg[GATE_COSTS[-1]], start_row, N, T)
        tv_actions_original = np.asarray(
            tv_doc["decisions"]["chosen_vectors"], dtype=np.int32)
        tv_actions = tv_actions_original
        if args.shuffle_teacher_vectors:
            # FALSIFICATION ONLY (acceptance iv): permute the target stream ACROSS decision
            # steps, destroying the obs<->target correspondence while leaving the marginal
            # action distribution untouched. The smoke gate must FAIL on this.
            perm = np.random.RandomState(int(args.seed) + 9001).permutation(N)
            assert not np.array_equal(perm, np.arange(N)), "shuffle drew the identity permutation"
            tv_actions = tv_actions_original[perm]
            print("!" * 84)
            print("!!! [861] --shuffle-teacher-vectors: SABOTAGE RUN (falsification leg). The "
                  "target stream")
            print("!!! is permuted across decision steps; this artifact is NOT a result and the "
                  "smoke gate")
            print(f"!!! MUST fail (seed {args.seed}, permutation of {N} decisions).")
            print("!" * 84)

    # ---- teacher-expressibility gate (TASK_845, pre-registered, the real deliverable) ----
    def _gate_line(g):
        rel = g["rel_diff"] if g["rel_diff"] is not None else float("nan")
        print(f"[845 GATE] {g['mapping']}: order net@21 {_fmt(g['teacher_order_net21'])}  "
              f"weight net@21 {_fmt(g['teacher_weight_net21'])}  rel {rel:.3f} "
              f"(tol {g['tolerance_rel']:.2f}) -> {'PASS' if g['passed'] else 'FAIL'}")

    # FIX 7 (fix pass): under --teacher-vectors the 819-book teacher walks and the 845
    # expressibility gate are pure WASTE — the 819 teacher/floor computed here is discarded and
    # rebound below, and the gate measures mappings this run does not imprint. That is 4-5
    # discarded full-panel walks (real leg: 1673 decisions x 64 assets each). Flag-gated skip;
    # the canon path (flag OFF) runs exactly as before, so the pins stay byte-identical.
    teacher = floor = gate = None
    if tv_actions is None:
        teacher = {c: walk_env(prices_j, tier_cfg[c], start_row, N, teacher_fn)
                   for c in GATE_COSTS}
        floor = {c: _sharpe(teacher[c]["rets"], args.bar_minutes) for c in GATE_COSTS}
        print(f"[832] teacher floor in-env: gross {floor[0]:+.3f}  net@21 {floor[21]:+.3f}")

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
    else:
        print("[861] skipping the 819-book teacher walks, the order-code floor reference and "
              "the 845 expressibility gate: --teacher-vectors imprints the collected stream, "
              "so those 4-5 full-panel walks would be computed and discarded (FIX 7). The 845 "
              "weight-rebalance FAIL-CLOSED assert stays live below.")

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
    if tv_actions is not None:
        # TASK_861: the target stream REPLACES the replay teacher. Everything downstream is the
        # exact shipped recipe — the same walk_env trajectory generation (so rewards/dones/value
        # targets come from THIS panel and THIS cost tier), the same one-hot per-asset marginals,
        # the same factored-CE BC loss, optimizer, iters and sanity replay.
        imprint_mapping = ("search-teacher-vectors[SHUFFLED-SABOTAGE]"
                           if args.shuffle_teacher_vectors else "search-teacher-vectors")
        # FIX 4: --teacher-vectors + --imprint-mapping weight-native is now ACCEPTED (it is in
        # the pinned runs/832_e5/jobs_imprint_full.manifest recipe, so leg 2 must run verbatim);
        # imprint_mapping is only a label from here on. Accepting it must not drop the 845
        # fail-closed guard, so re-assert it on this path.
        if args.imprint_mapping == "weight-native":
            assert env_cfg.vector_weight_rebalance, (
                "--imprint-mapping weight-native requires --vector-weight-rebalance so the "
                "teacher walk, obs encoding, and ExIt env all run weight semantics; the gate "
                "certifies the weight-env teacher, not the order-env one (TASK_845 MINOR, "
                "preserved across the 861 --teacher-vectors path)")
        tv_fn = lambda k, state: tv_actions[k]                          # noqa: E731
        teacher = {c: walk_env(prices_j, tier_cfg[c], start_row, N, tv_fn)
                   for c in GATE_COSTS}
        floor = {c: _sharpe(teacher[c]["rets"], args.bar_minutes) for c in GATE_COSTS}
        print(f"[832] IMPRINT MAPPING = {imprint_mapping} (861 collected chosen vectors); "
              f"teacher floor in-env: gross {floor[0]:+.3f}  net@21 {floor[21]:+.3f}")
        # FIX 2b — REPLAY INTEGRITY: the stream just walked in THIS env must reproduce the
        # score the collection walk recorded for itself. This one comparison independently
        # catches an off-pin operator, a corrupted payload and any future silent drift: a
        # stream that does not reproduce its recorded score is not the stream that was scored.
        # SKIPPED (loudly) only on the sabotage leg, whose whole point is a different stream.
        tv_replay = _tv_replay_check(tv_prov, floor[GATE_COSTS[-1]],
                                     args.shuffle_teacher_vectors)
    elif imprint_mapping == "weight-native":
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
    # TASK_861 acceptance (iv): the train-set fit measured against the NEVER-SHUFFLED targets.
    tv_fit = None
    if tv_actions_original is not None:
        tv_fit = teacher_vectors_fit(
            policy_action_log_marginals(model, trainer.state.params, obs, A),
            tv_actions_original)
        print(f"[861] teacher-vector fit vs the NEVER-SHUFFLED targets "
              f"(n={tv_fit['n_label_cells']} cells): NLL {tv_fit['nll_vs_original_targets']:.4f} "
              f"vs marginal baseline {tv_fit['nll_marginal_baseline']:.4f} -> reduction "
              f"{tv_fit['nll_reduction']:+.4f} (floor {tv_fit['nll_reduction_floor']:+.2f}); "
              f"argmax agreement {tv_fit['argmax_agreement_vs_original']:.3f} vs modal rate "
              f"{tv_fit['modal_action_rate']:.3f}")
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

    # TASK_861: flag-gated ADDITIVE block — absent (and every key above unchanged) when
    # --teacher-vectors is not passed, so the default artifact is byte-identical to shipped 832.
    if tv_prov is not None:
        result["teacher_vectors"] = {
            "source": tv_prov,
            "shuffled_sabotage": bool(args.shuffle_teacher_vectors),
            "shuffle_seed": (int(args.seed) + 9001) if args.shuffle_teacher_vectors else None,
            "recipe": "target stream REPLACES the replay teacher; trajectory generation, "
                      "one-hot per-asset marginals, factored-CE BC loss, optimizer, iters and "
                      "sanity replay are the shipped 832 recipe, unchanged",
            "fit_vs_original_targets": tv_fit,
            # FIX 1 — top-level, loud: this result is a DIFFERENT EXPERIMENT from the pinned one.
            "operator_off_pin": bool(tv_prov["gate"]["operator_off_pin"]),
            "operator_off_pin_fields": tv_prov["gate"]["operator_off_pin_fields"],
            # FIX 4: the --imprint-mapping flag as GIVEN (a label on this path; the 845
            # weight-rebalance fail-closed assert was still enforced for 'weight-native').
            "imprint_mapping_flag": args.imprint_mapping,
            # FIX 2b — the recorded-vs-replayed walk-score integrity check.
            "replay_integrity": tv_replay,
            # FIX 7 — what the flag-gated skip left out of this artifact.
            "skipped_819_book_legs": {
                "expressibility_gate": True, "order_floor_reference": True,
                "book_teacher_walks": True,
                "why": "the 819-book teacher/floor is discarded and rebound by this path and "
                       "the 845 gate measures mappings this run does not imprint; 4-5 discarded "
                       "full-panel walks (FIX 7). The 845 --vector-weight-rebalance "
                       "fail-closed assert is still enforced.",
            },
        }
        if tv_prov["gate"]["operator_off_pin"]:
            result["operator_off_pin"] = True

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
        # FIX 7: under --teacher-vectors the gate is deliberately not computed (its walks are
        # discarded work), so its asserts apply to the canon path only.
        if gate is None:
            assert tv_actions is not None, "861: expressibility gate missing off the tv path"
            print("[832] expressibility-gate asserts SKIPPED: --teacher-vectors does not "
                  "imprint either 819-book mapping (FIX 7)")
        else:
            assert gate["teacher_order_net21"] is not None and gate["teacher_weight_net21"] is not None, \
                "SMOKE FAIL: expressibility gate did not emit both net@21 values"
            assert (gate["passed"] or "weight_native_regate" in gate), \
                "SMOKE FAIL: order-code gate missed but the weight-native re-gate was not shipped"
        final_gate = None if gate is None else gate.get("weight_native_regate", gate)
        if final_gate is not None and not final_gate["passed"]:
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
        # TASK_861 acceptance (iv): the falsification gate. HIGH-SNR train-set fit against the
        # NEVER-SHUFFLED targets (n = N x n_assets label cells) — deliberately NOT a small-n
        # held-out Sharpe, which inverts per seed at fixture scale (the 860 lesson).
        if tv_fit is not None:
            assert tv_fit["n_label_cells"] == N * A, "861: fit measured on the wrong cell count"
            assert tv_fit["nll_reduction"] >= TEACHER_VECTORS_NLL_REDUCTION_FLOOR, (
                f"SMOKE FAIL (861 falsification gate): the imprinted policy does not fit the "
                f"never-shuffled teacher vectors — NLL {tv_fit['nll_vs_original_targets']:.4f} vs "
                f"marginal baseline {tv_fit['nll_marginal_baseline']:.4f} => reduction "
                f"{tv_fit['nll_reduction']:+.4f} < floor "
                f"{TEACHER_VECTORS_NLL_REDUCTION_FLOOR:+.2f}"
                + (" (EXPECTED: this is a --shuffle-teacher-vectors sabotage run)"
                   if args.shuffle_teacher_vectors else ""))
            print(f"[861] FALSIFICATION GATE PASS: NLL reduction "
                  f"{tv_fit['nll_reduction']:+.4f} >= floor "
                  f"{TEACHER_VECTORS_NLL_REDUCTION_FLOOR:+.2f} on "
                  f"{tv_fit['n_label_cells']} label cells")
        print("[832] SMOKE PASS: finite losses, decreasing imprint CE, sanity replay emitted "
              "schema" + (", expressibility gate reported (both mappings)" if gate is not None
                          else ", 861 teacher-vector legs (provenance + operator pin + payload "
                               "digest + replay integrity + falsification)"))
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
    tv = r.get("teacher_vectors")                       # TASK_861 (absent on the canon path)
    if tv is not None:
        f = tv["fit_vs_original_targets"]
        print(f"[861] teacher vectors: {tv['source']['path']} "
              f"({tv['source']['gate']['n_decisions']} decisions, provenance gate PASS"
              f"{', SHUFFLED SABOTAGE' if tv['shuffled_sabotage'] else ''})")
        if tv.get("operator_off_pin"):
            print(f"      *** OFF-PIN TEACHER (a DIFFERENT experiment from the +8.024 operator "
                  f"of record): {tv['operator_off_pin_fields']}")
        ri = tv.get("replay_integrity") or {}
        if ri.get("checked"):
            print(f"      replay integrity: recorded walk net@21 "
                  f"{ri['recorded_walk_net21']:+.6f} == replayed floor "
                  f"{ri['replayed_floor_net21']:+.6f} (rel {ri['rel_delta']:.2e} < "
                  f"{ri['rel_tolerance']:.0e})")
        elif ri:
            print(f"      replay integrity: SKIPPED ({ri.get('reason')})")
        print(f"      fit vs never-shuffled targets: NLL {f['nll_vs_original_targets']:.4f} vs "
              f"marginal {f['nll_marginal_baseline']:.4f} -> reduction {f['nll_reduction']:+.4f} "
              f"(floor {f['nll_reduction_floor']:+.2f}); argmax agree "
              f"{f['argmax_agreement_vs_original']:.3f} vs modal {f['modal_action_rate']:.3f}")
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
    ap.add_argument("--teacher-vectors", default=None, dest="teacher_vectors",
                    help="TASK_861: JSON collection written by `eval_vector.py "
                         "--collect-chosen-vectors` — the per-decision action vectors the "
                         "offline SEARCH teacher chose. Replaces the replay-teacher target "
                         "stream; everything else is the shipped recipe. A fail-loud provenance "
                         "gate refuses any collection whose db / window / panel frame / decision "
                         "rows / EnvConfig fingerprint do not match this run, that was not "
                         "recorded with the PINNED +8.024 operator/checkpoint, whose payload "
                         "sha256 or integrality is broken, or that does not replay to its own "
                         "recorded walk score. Default OFF = shipped 832, byte-identical")
    ap.add_argument("--allow-off-pin-teacher", dest="allow_off_pin_teacher",
                    action="store_true", default=False,
                    help="TASK_861 fix pass (FIX 1) ESCAPE HATCH: accept a collection recorded "
                         "with an operator/checkpoint other than the pinned +8.024 one. The run "
                         "is stamped operator_off_pin: true (plus the deviating field list) in "
                         "the result artifact and banner-warned on stdout — it is a DIFFERENT "
                         "EXPERIMENT, not the pre-registered one. Needed by the fixture smoke, "
                         "which collects under a checkpoint it just minted")
    ap.add_argument("--shuffle-teacher-vectors", dest="shuffle_teacher_vectors",
                    action="store_true", default=False,
                    help="TASK_861 FALSIFICATION ONLY (acceptance iv): permute the loaded target "
                         "stream across decision steps (obs<->target correspondence destroyed, "
                         "action marginal preserved). The smoke gate MUST fail. Requires "
                         "--teacher-vectors; never use for a real run")
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

    # TASK_861 flag coherence, before anything expensive runs.
    # FIX 3 (fix pass): `--teacher-vectors ""` used to fall through the truthiness test and
    # silently train the DEFAULT 819 trend-book teacher with no teacher_vectors block — and,
    # because tv_suffix stayed empty, overwrite the checked-in results_fixture_smoke.json +
    # ckpt_smoke/ parity baseline. Fail closed on a blank path.
    if args.teacher_vectors is not None and not args.teacher_vectors.strip():
        ap.error("--teacher-vectors was given an EMPTY path; pass the collection JSON (an unset "
                 "shell variable would otherwise silently run the DEFAULT 819-book teacher and "
                 "clobber the canon fixture-smoke baseline).")
    if args.shuffle_teacher_vectors and args.teacher_vectors is None:
        ap.error("--shuffle-teacher-vectors requires --teacher-vectors (it permutes that stream)")
    if args.allow_off_pin_teacher and args.teacher_vectors is None:
        ap.error("--allow-off-pin-teacher requires --teacher-vectors (it relaxes that gate)")
    # FIX 4 (fix pass): --teacher-vectors + --imprint-mapping weight-native is ACCEPTED. The
    # pinned recipe (runs/832_e5/jobs_imprint_full.manifest) passes weight-native, so refusing
    # the combination made the pre-registered real leg 2 unrunnable verbatim. Under
    # --teacher-vectors imprint_mapping is only a LABEL (nothing downstream of the teacher
    # selection reads it); the 845 --vector-weight-rebalance fail-closed assert is re-applied
    # on the teacher-vectors path so accepting the flag does not drop a guard.

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
    # TASK_861: a --teacher-vectors run is a DIFFERENT experiment from the canon smoke, so its
    # default outputs are suffixed — a verification run can never clobber the checked-in parity
    # baseline (the 835 fix-pass precedent, eval_vector.py::main).
    tv_suffix = ""
    if args.teacher_vectors is not None:
        tv_suffix = "_teacher_vectors" + ("_shuffled" if args.shuffle_teacher_vectors else "")
    if args.ckpt_out is None:
        args.ckpt_out = os.path.join(
            exp_dir, ("ckpt_smoke" if is_fixture else "ckpt_imprint") + tv_suffix)
    if args.smoke_assert is None:
        args.smoke_assert = is_fixture
    tag = args.tag or (("fixture-smoke" if is_fixture else "real-panel")
                       + tv_suffix.replace("_", "-"))
    out = args.out
    if out is None and is_fixture:
        out = os.path.join(exp_dir, f"results_fixture_smoke{tv_suffix}.json")
    run(args, out, tag)


if __name__ == "__main__":
    main()
