#!/usr/bin/env python3
"""TASK_857 / D23 step 2 — counterfactual value-head imprint → the DEPLOYABLE v̂-leaf.

The proven-good search (+8.02 net@21, LEDGER 832/E5-inversion3) has ZERO learned parts: its
leaf is an R-step TRUE-ENV rollout = lookahead, deployable never. Every net-dependent path so
far dies at one measured number: v̂ ranks candidates at ~+0.03 correlation with the true R-step
return (849's calibration column) because the value head was never trained on counterfactuals.
This module fixes THAT number directly:

  1. Counterfactual dataset generator (offline; REUSES the 851 leaf machinery
     ``SampledMuZeroVectorSearch._true_env_leaf_values`` — not reinvented): walk trajectories
     (teacher walk + true-env-leaf search-greedy walk, BOTH cost tiers) and at every decision
     state emit (state, candidate action-vector, true R=3-step discounted return) for the K
     candidates the search actually generates (dithered-book hb3.0 — the deployment
     distribution) plus the policy-argmax candidate. Held in RAM (bounded instrument dataset,
     ~thousands..67k tuples/pass) — NEVER pre-generated to disk, NO disk I/O in the train loop
     (efficiency invariant: VRAM>RAM>disk).

  2. Value-head imprint (BC-style, mirrors 832's recipe): regress ``v̂(g(s,a))`` on the true
     returns over the counterfactual set. Trains DYNAMICS + VALUE-head gradients ONLY; the
     policy head, reward head, and encoder are FROZEN (optax multi_transform partition; the
     reward-projection columns of dynamics_dense2 receive exactly zero gradient because the
     loss touches only next_state→value). Warm-start from ``runs/832_e5/ckpt_imprint_full``
     (real) / the 832 ``ckpt_smoke`` (fixture). PopArt normalizes the true-return targets
     (spread across the 401-bin support) exactly as the shipped trainer does.

  3. The deployable read is ``eval_vector.py --leaf-eval vhat-distilled`` (TASK_857 scope 3):
     mechanically the canon MODEL leaf (value head + dynamics, no lookahead), graded against
     the +8.02 true-leaf ceiling. This module's calibration + ranking-transfer legs prove the
     distilled head ranks candidates like the true env BEFORE the corridor read is spent.

Acceptance legs computed here (FIXTURE):
  (i)  Calibration: post-imprint v̂-vs-true-return correlation across candidates on HELD-OUT
       states, reported pre vs post. The ≥+0.5 bar is graded on the REAL leg (planner run);
       on the fixture we demonstrate the machinery yields a finite, sensible correlation that
       IMPROVES over the pre-imprint baseline.
  (ii) Ranking transfer: the v̂-leaf (model-leaf) search reproduces the true-env-leaf search's
       candidate CHOICES on the dumped held-out states (849-style trace comparison; % agreement,
       same rng ⇒ same candidate set, only the leaf differs).

FACTORY / SCHEDULED split (house style): this module + its fixture smoke are the factory
deliverable. The REAL distill (warm from ckpt_imprint_full over the read-only trading_data.db
panel) + the corridor read vs floor +0.424 / ceiling +8.02 are a SCHEDULED run — never inside
the factory loop. Default db = the fixture; CPU by default. NEVER touches trading_data.db.
"""
import os
# CPU-only by default (factory/smoke). The scheduled GPU run overrides JAX_PLATFORMS.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import importlib.util
import json
import sys

import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax.traverse_util import flatten_dict, unflatten_dict

# Run from anywhere: repo root on the path so `trader` / `mcts_trader` import cleanly.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from mcts_trader.env.jax_trading_env import (
    reset, batch_step_vector, jit_batch_compute_action_mask,
    get_current_prices, compute_portfolio_value,
)
from mcts_trader.jax_agent import get_value_support
from mcts_trader.jax_agent.sampled_search import create_sampled_vector_search
from mcts_trader.jax_agent.popart import update_statistics

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
FIXTURE_DB = os.path.join(_REPO, "fixtures", "trading_fixture.db")
IMPRINT_PY = os.path.join(_REPO, "experiments", "832_linear_imprint", "imprint.py")
FEE_BPS, SLIP_BPS = 16.0, 5.0
GATE_COSTS = (0, 21)
N_ACTIONS_PER_ASSET = 5
HOLD = 1
# (ii) deployable-ranking fail-closed-safe floor: reject only STRONG anti-correlation / target-
# shuffle sabotage on the deployed-Q axis. Set BELOW the legitimate distill's fixture-scale noise
# band (deployed-Q min ~-0.12 across seeds 0-3 — the frozen reward head gives no reliable lift)
# and WELL ABOVE target-shuffled sabotage (~-0.49, consistent across seeds). NOT a lift/positive-
# validation bar (that verdict is the planner's real-data corridor read). See the (ii) gate.
DEPLOYED_Q_DEGENERACY_FLOOR = -0.25
# TASK_860 co-distill falsification gate (spec acceptance iv): a LEGIT run's trained heads fit the
# TRUE (never-shuffled) targets on the TRAIN set with high pooled correlation; a --shuffle-targets
# run fits a PERMUTATION of true, so its correlation vs TRUE collapses toward 0. Measured seed-sweep
# (seeds 0-3, n=972 train tuples/run, min over the two heads r_model·true-r0 and v̂·true-G):
#   LEGIT    min +0.833  (band [+0.833, +0.943])  -> PASS
#   SHUFFLE  max +0.055  (band [-0.187, +0.055])  -> FAIL
# The floor sits in that wide gap (cushion ~0.39 above sabotage-max, ~0.38 below legit-min). This is
# the high-SNR discriminator the noisy n=12 held-out per-state Q/r_model metrics CANNOT provide
# (those overlap and invert per-seed — reported directions only, never gated). Both heads must clear
# it (shuffle permutes BOTH the value G and reward r0 targets, so both fits collapse).
TRAIN_FIT_FLOOR = 0.45

# The dynamics + value-head parameter subtree — the ONLY leaves that receive gradient.
# Everything else (encoder, TRM reasoning refiners, policy output head, scalar action
# embedding) is FROZEN. Note dynamics_dense2 emits [next_state | reward]; the value-only
# loss puts exactly zero gradient on the reward columns, so the reward head is frozen in
# effect even though the Dense is in the trainable partition (verified in the smoke).
_TRAIN_TOP = frozenset({
    "dynamics_dense1", "dynamics_ln", "dynamics_dense2", "dynamics_state_norm",
    "vector_action_embedding",
})


def load_imprint_module():
    """Import experiments/832_linear_imprint/imprint.py by path (its dir name is not a python
    identifier). Reused verbatim: ``build_model_and_trainer`` (canon arch/trainer construction),
    ``load_training_panel`` (PIN b training panel), ``encode_training_obs``, ``_load_corridor``
    (819 book signs + ``_to_action_vec``), ``_sharpe``, ``policy_argmax_actions``."""
    spec = importlib.util.spec_from_file_location("imprint832", IMPRINT_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# =============================================================================
# freeze partition (optax multi_transform): dynamics + value-head ONLY
# =============================================================================

def _is_trainable(path):
    """path: tuple of param-tree keys. Trainable iff it is a dynamics leaf (top-level) or the
    TRM value head (trm/reasoning/value_head_*)."""
    if path[0] in _TRAIN_TOP:
        return True
    if len(path) >= 3 and path[0] == "trm" and path[1] == "reasoning" \
            and str(path[2]).startswith("value_head"):
        return True
    return False


def make_partition_optimizer(params, lr):
    """optax.multi_transform: adam on the dynamics+value leaves, set_to_zero on the rest.
    Returns (tx, label_tree, n_train_leaves, n_freeze_leaves)."""
    flat = flatten_dict(params)
    labels = {k: ("train" if _is_trainable(k) else "freeze") for k in flat}
    n_train = sum(1 for v in labels.values() if v == "train")
    n_freeze = len(labels) - n_train
    label_tree = unflatten_dict(labels)
    tx = optax.multi_transform(
        {"train": optax.adam(lr), "freeze": optax.set_to_zero()}, label_tree)
    return tx, label_tree, n_train, n_freeze


# =============================================================================
# value forward (bit-consistent with the deployed model leaf) + decode
# =============================================================================

def _make_value_fns(model, discount):
    """Return (vhat_scalar_fn, deployed_q_fn, value_logits_fn). All flow gradient through
    dynamics AND the value head (NO stop_gradient — the canon ``recurrent_inference_vector``
    stop-grads the prediction, which would starve dynamics; the distill deliberately trains
    dynamics-for-value by calling ``dynamics_vector`` + ``prediction`` directly).

      * ``vhat_scalar`` decodes EXACTLY as the search leaf's ``_categorical_to_scalar``
        (softmax·support, value_scale 1, shift 0) — v̂ IN ISOLATION (the spec's literal (i)).
      * ``deployed_q`` = ``r_model + γ·v̂`` — the quantity the DEPLOYED model leaf actually ranks
        candidates by (sampled_search.py's mctx backup Q = reward + γ·value, reward decoded via
        ``_reward_to_scalar`` = softmax·support). This is the HONEST deployment predictor: the
        distill freezes the reward head, so a well-calibrated v̂ can still be dragged off by a
        poor r_model. Calibrating deployed_q (not v̂ alone) is what actually predicts the
        ranking-transfer / gates the corridor read (TASK_857 fix-pass FIX 2)."""
    support = get_value_support()

    def _next_and_reward(params, latent, vec):
        return model.apply({"params": params}, latent, vec, method="dynamics_vector")

    def value_logits(params, latent, vec):
        next_state, _reward = _next_and_reward(params, latent, vec)
        _policy_logits, value = model.apply(
            {"params": params}, next_state, method="prediction")
        return value

    def vhat_scalar(params, latent, vec):
        probs = jax.nn.softmax(value_logits(params, latent, vec), axis=-1)
        return jnp.sum(probs * support, axis=-1)

    def deployed_q(params, latent, vec):
        next_state, reward = _next_and_reward(params, latent, vec)
        _policy_logits, value = model.apply(
            {"params": params}, next_state, method="prediction")
        r = jnp.sum(jax.nn.softmax(reward, axis=-1) * support, axis=-1)   # _reward_to_scalar
        v = jnp.sum(jax.nn.softmax(value, axis=-1) * support, axis=-1)    # _categorical_to_scalar
        return r + discount * v

    # TASK_860 co-distill: the reward-head logits (dynamics_dense2 reward columns) and their
    # scalar decode (softmax·support = the search's ``_reward_to_scalar``). ``reward_logits`` is
    # the CE-training head; ``r_model_scalar`` is the calibration predictor vs the true r0.
    def reward_logits(params, latent, vec):
        _next_state, reward = _next_and_reward(params, latent, vec)
        return reward

    def r_model_scalar(params, latent, vec):
        _next_state, reward = _next_and_reward(params, latent, vec)
        return jnp.sum(jax.nn.softmax(reward, axis=-1) * support, axis=-1)   # _reward_to_scalar

    return (jax.jit(vhat_scalar), jax.jit(deployed_q), jax.jit(value_logits),
            jax.jit(reward_logits), jax.jit(r_model_scalar))


# =============================================================================
# counterfactual dataset generator (offline; reuses the 851 leaf machinery)
# =============================================================================

def _reset_walk(prices_j, cfg, start_row, horizon):
    """A single (B=1) walk state anchored at panel row ``start_row`` (the 832/851 walk pattern),
    batched for the shipped ``batch_step_vector`` / ``_true_env_leaf_values`` APIs."""
    cash = float(cfg.initial_cash)
    s0 = reset(jax.random.PRNGKey(0), prices_j, cfg).replace(
        episode_start_idx=jnp.array(start_row), current_step=jnp.array(0),
        max_episode_steps=jnp.array(horizon + 2),
        initial_portfolio_value=jnp.array(cash),
        prev_portfolio_value=jnp.array(cash),
    )
    return jax.tree_util.tree_map(lambda x: x[None], s0)   # B=1


def collect_pass(model, params, helper, prices_j, cfg, obs_all, tgt, corridor,
                 start_row, N, A, discount, R, policy, base_rng,
                 max_positions, min_cash):
    """One trajectory pass: walk ``policy`` ('teacher' = 819 book, or 'search' = true-env-leaf
    greedy pick over the candidate set) from ``start_row`` for ``N`` decisions. At each decision
    emit the counterfactual tuples for the K search candidates + the policy-argmax candidate.

    Returns dict of RAM arrays:
        dec_obs   (D, 27, A, A) float32 — one obs per visited decision
        cand      (D, M, A)     int32   — M = K+1 candidate vectors per decision
        G         (D, M)        float32 — true R-step discounted return per candidate
        r0        (D, M)        float32 — the r0 component of G (TASK_860 reward-codistill target)
        states    list[pytree]  B=1 TradingState snapshots (for the ranking-transfer search)
    (D <= N; the walk stops early if the env terminates.)
    """
    book = jnp.ones((1, A), dtype=jnp.int32)
    state = _reset_walk(prices_j, cfg, start_row, N + R + 2)
    dec_obs, cand_dump, G_dump, r0_dump, states = [], [], [], [], []
    for k in range(N):
        obs_k = jnp.asarray(obs_all[k])[None]                     # (1,27,A,A)
        root = model.apply({"params": params}, obs_k, method="initial_inference")
        policy_logits = root["policy_logits"]                     # (1, A*5)
        latent = root["latent_state"]                             # (1, dim)
        price_idx = start_row + k
        current_prices = prices_j[price_idx][None]                # (1, A)
        mask = jit_batch_compute_action_mask(
            state, current_prices, cfg, max_positions, min_cash)
        rng_k = jax.random.fold_in(base_rng, k)
        # K candidates via the SAME dithered-book proposal machinery the search uses
        # (build_root_candidates mirrors _search's sample path) + the policy-argmax vector.
        cand_K = helper.build_root_candidates(policy_logits, mask, book, rng_k)   # (1,K,A)
        argmax_vec = policy_logits.reshape(1, A, N_ACTIONS_PER_ASSET).argmax(-1).astype(jnp.int32)
        cand = jnp.concatenate([cand_K, argmax_vec[:, None, :]], axis=1)          # (1,M,A)
        # TRUE R-step discounted return per candidate — reuse the shipped 851 leaf helper
        # (tail OFF ⇒ pure truncated true-env rollout). Q_true = r0 + γ·V is the 831/851
        # candidate score the true-env leaf ranks by; that IS the "true R-step return".
        r0, V = helper._true_env_leaf_values(params, state, prices_j, cfg, latent, cand)
        G = r0 + discount * V                                     # (1, M)
        dec_obs.append(np.asarray(obs_all[k], dtype=np.float32))
        cand_dump.append(np.asarray(cand[0], dtype=np.int32))
        G_dump.append(np.asarray(G[0], dtype=np.float32))
        # TASK_860: emit the r0 component as its own target (near-zero extra compute — r0 is
        # already returned by the leaf helper above; only stored, never re-computed).
        r0_dump.append(np.asarray(r0[0], dtype=np.float32))
        states.append(jax.tree_util.tree_map(lambda x: np.asarray(x), state))
        # walk action
        if policy == "teacher":
            side = np.asarray(state.pos_side)[0]
            act = np.asarray(corridor._to_action_vec(tgt[start_row + k], side, A),
                             dtype=np.int32)
        else:                                   # 'search' = true-env-leaf greedy over candidates
            act = np.asarray(cand[0][int(np.argmax(np.asarray(G[0])))], dtype=np.int32)
        state, _, _, done = batch_step_vector(
            state, jnp.asarray(act)[None], prices_j, cfg)
        if bool(np.asarray(done)[0]):
            break
    return {
        "dec_obs": np.asarray(dec_obs, dtype=np.float32),
        "cand": np.asarray(cand_dump, dtype=np.int32),
        "G": np.asarray(G_dump, dtype=np.float32),
        "r0": np.asarray(r0_dump, dtype=np.float32),
        "states": states,
    }


def build_counterfactual_dataset(model, params, prices_j, env_cfg, obs_all, tgt, corridor,
                                  start_row, N, A, args):
    """Teacher + search-greedy walks × both cost tiers → the pooled counterfactual set (RAM).
    Returns a list of per-pass dicts (from ``collect_pass``), tagged by (policy, cost)."""
    helper = create_sampled_vector_search(
        model, n_assets=A, n_actions_per_asset=N_ACTIONS_PER_ASSET,
        num_samples=args.vector_k, num_simulations=args.sims,
        max_num_considered_actions=min(args.vector_k, 16), discount=args.discount,
        gumbel_scale=args.gumbel_scale, hold_bias=args.vector_hold_bias,
        dither_scale=args.vector_dither_scale, proposal_scheme="dithered-book",
        # true-env leaf, tail OFF ⇒ the pure truncated true-env rollout = the counterfactual
        # target. value_scale 1.0 keeps the (unused, tail-OFF) decode inert / not fail-loud.
        leaf_eval="true-env", leaf_rollout_r=args.leaf_rollout_r, leaf_vhat_tail=False)
    base_rng = jax.random.PRNGKey(args.seed + 7)
    passes = []
    ctr = 0
    for policy in ("teacher", "search"):
        for c in GATE_COSTS:
            fee, slip = (0.0, 0.0) if c == 0 else (FEE_BPS, SLIP_BPS)
            cfg_c = env_cfg.replace(fee_bps=fee, slippage_bps=slip)
            rng = jax.random.fold_in(base_rng, ctr)
            ctr += 1
            p = collect_pass(model, params, helper, prices_j, cfg_c, obs_all, tgt, corridor,
                             start_row, N, A, args.discount, args.leaf_rollout_r, policy, rng,
                             args.max_positions, args.min_cash_threshold)
            p["policy"], p["cost"] = policy, c
            passes.append(p)
            print(f"[857] pass {policy}@{c}bp: {p['dec_obs'].shape[0]} decisions, "
                  f"{p['cand'].shape[0] * p['cand'].shape[1] if p['cand'].size else 0} tuples")
    return passes, cfg_c


# =============================================================================
# calibration (the 849 instrument) — the HEADLINE is MEAN PER-STATE Pearson
# =============================================================================
# The 849 calibration column = correlation of the predictor with the true R-step return
# ACROSS CANDIDATES *WITHIN* A STATE, then MEANED over (held-out) states. The +0.5 acceptance
# bar descends from THAT statistic. A POOLED Pearson (all state×candidate pairs at once) is a
# DIFFERENT, inflated number — between-state variance in the return level leaks in and lifts it
# (fixture: post per-state +0.66 vs pooled +0.83). So ``mean_per_state_pearson`` is the primary/
# headline/gated number here; ``pooled_pearson`` is kept only as a clearly-labelled secondary.

def _pearson(x, y):
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    if x.size < 2:
        return float("nan")
    sx, sy = x.std(), y.std()
    if sx < 1e-12 or sy < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def calibration(pred_fn, params, latents, cands, Gs):
    """predictor-vs-true-return calibration across candidates. ``pred_fn(params, latent, vec) ->
    (B,) scalar`` (either v̂-alone or the deployed Q = r_model+γ·v̂). ``latents`` (D,dim),
    ``cands`` (D,M,A), ``Gs`` (D,M). Returns a dict whose PRIMARY key is
    ``mean_per_state_pearson`` (the 849 instrument — corr across the M candidates within each
    state, meaned over states) and whose SECONDARY key is the inflated ``pooled_pearson``."""
    D, M, A = cands.shape
    p_all, g_all, per_state = [], [], []
    for d in range(D):
        lat = jnp.asarray(np.repeat(latents[d][None], M, axis=0))       # (M, dim)
        vec = jnp.asarray(cands[d])                                     # (M, A)
        pv = np.asarray(pred_fn(params, lat, vec), dtype=np.float64)    # (M,)
        g = np.asarray(Gs[d], dtype=np.float64)
        p_all.append(pv); g_all.append(g)
        per_state.append(_pearson(pv, g))
    p_all = np.concatenate(p_all); g_all = np.concatenate(g_all)
    per_state = np.asarray(per_state, dtype=np.float64)
    finite = per_state[np.isfinite(per_state)]
    return {
        "mean_per_state_pearson": float(finite.mean()) if finite.size else float("nan"),  # PRIMARY
        "pooled_pearson": _pearson(p_all, g_all),                                          # secondary
        "n_states": int(D), "n_pairs": int(p_all.size),
        "n_states_finite": int(finite.size),
    }


# =============================================================================
# ranking transfer (ii) — v̂-leaf search vs true-env-leaf search choice agreement
# =============================================================================

def ranking_transfer(model, params, prices_j, env_cfg, held_states, held_obs, A, args):
    """On each held-out dumped state run BOTH search operators with the SAME rng (⇒ identical
    candidate set; only the LEAF differs) and compare the selected action VECTOR. Returns
    (pct_agree, n_states). The v̂ leaf is the canon MODEL leaf (== eval_vector --leaf-eval
    vhat-distilled); the reference is the true-env leaf (R, tail OFF)."""
    common = dict(
        n_assets=A, n_actions_per_asset=N_ACTIONS_PER_ASSET,
        num_samples=args.vector_k, num_simulations=args.sims,
        max_num_considered_actions=min(args.vector_k, 16), discount=args.discount,
        gumbel_scale=args.gumbel_scale, hold_bias=args.vector_hold_bias,
        dither_scale=args.vector_dither_scale, proposal_scheme="dithered-book")
    vhat_search = create_sampled_vector_search(model, leaf_eval="model", **common)
    te_search = create_sampled_vector_search(
        model, leaf_eval="true-env", leaf_rollout_r=args.leaf_rollout_r,
        leaf_vhat_tail=False, **common)
    book = jnp.ones((1, A), dtype=jnp.int32)
    base_rng = jax.random.PRNGKey(args.seed + 99)
    agree = 0
    n = len(held_states)
    for i in range(n):
        st = jax.tree_util.tree_map(lambda x: jnp.asarray(x), held_states[i])
        obs = jnp.asarray(held_obs[i])[None]
        cur = prices_j[int(np.asarray(st.episode_start_idx)[0]
                          + np.asarray(st.current_step)[0])][None]
        mask = jit_batch_compute_action_mask(
            st, cur, env_cfg, args.max_positions, args.min_cash_threshold)
        rng = jax.random.fold_in(base_rng, i)
        av_v, _, _, _ = vhat_search(params, rng, obs, mask, book, True)
        av_t, _, _, _ = te_search(params, rng, obs, mask, book, True,
                                  env_state=st, prices=prices_j, env_config=env_cfg)
        agree += int(np.array_equal(np.asarray(av_v), np.asarray(av_t)))
    return (100.0 * agree / max(n, 1)), n


# =============================================================================
# distill training loop (RAM minibatches; dynamics+value only)
# =============================================================================

def distill_value_head(model, trainer, value_logits_fn, latents, cands, Gs, args,
                       reward_logits_fn=None, r0s=None):
    """BC value regression: v̂(g(s,a)) → the true R-step return over the counterfactual set.
    PopArt-normalizes the targets (spread across the 401-bin support) then trains the
    dynamics+value partition with a two-hot categorical CE (the shipped value-loss form).

    TASK_860 co-distill (``args.codistill_reward``, default OFF = bit-identical to shipped 857):
    when ON, ADD a λ·reward-CE term on the true-r0 targets (``r0s``) using the SAME PopArt/two-hot
    reward-loss form the shipped trainer uses (reward_popart-normalize → ``_scalar_to_categorical``
    → CE on the reward logits). The reward CE supplies gradient to the ``dynamics_dense2`` reward
    columns — the ONE new leg that trains — so they move from frozen-in-effect (zero grad under the
    value-only loss) to trained; policy head + encoder stay frozen by the same partition.
    Returns (distilled_params, history dict)."""
    params0 = trainer.state.params
    codistill = bool(getattr(args, "codistill_reward", False)) and reward_logits_fn is not None
    # flatten to per-tuple arrays (repeat the decision latent across its M candidates)
    D, M, A = cands.shape
    lat_flat = np.repeat(latents, M, axis=0)              # (D*M, dim)
    vec_flat = cands.reshape(D * M, A)                    # (D*M, A)
    G_flat = Gs.reshape(D * M).copy()                     # (D*M,)

    # FALSIFICATION HOOK (TASK_857 fix-pass FIX 3 spot-check; default OFF): permute the targets
    # so the value head trains on a state↔return correspondence that is BROKEN. The head still
    # MOVES (value_head_moved stays True) and CE still drops (it memorizes noise), but the 849
    # held-out per-state calibration cannot improve and ranking-transfer collapses to chance —
    # so the (i)/(ii) smoke gates MUST FAIL. Proves those gates are non-hollow, not tautological.
    if getattr(args, "shuffle_targets", False):
        np.random.RandomState(args.seed + 4242).shuffle(G_flat)
        print("[857] WARN: --shuffle-targets ON — targets permuted (falsification spot-check; "
              "the (i)/(ii) gates must now FAIL)")

    # PopArt: snap statistics to the true-return distribution (beta=1 batch replace), then
    # normalize — targets become ~unit-variance, spread across the support. The shipped leaf
    # decode ignores PopArt (softmax·support — TASK_857 fix-pass FIX 4), so this only shapes the
    # training target; the head learns a MONOTONE map of G ⇒ v̂ still orders candidates like the
    # true env. Re-snapped here (NOT the 832 warm-start stats) so the SAVED PopArt matches the
    # returns the head was regressed on; self-consistent via trainer.save_checkpoint.
    if trainer.use_popart and trainer.popart is not None:
        trainer.popart.state = update_statistics(
            trainer.popart.state, jnp.asarray(G_flat), beta=1.0)
        norm = np.asarray(trainer.popart.normalize(jnp.asarray(G_flat)), dtype=np.float32)
    else:
        mu, sd = float(G_flat.mean()), float(max(G_flat.std(), 1e-6))
        norm = ((G_flat - mu) / sd).astype(np.float32)
    targets = np.asarray(trainer._scalar_to_categorical(jnp.asarray(norm)), dtype=np.float32)

    # TASK_860 reward-codistill targets — the SAME PopArt/two-hot form as the value path above,
    # but on the true-r0 component and the reward head's own PopArt normalizer. Built only when ON
    # (OFF ⇒ this block is skipped and the whole function stays bit-identical to shipped 857).
    rtargets, rlw, rnorm = None, 1.0, None
    if codistill:
        r0_flat = r0s.reshape(D * M).copy()                   # (D*M,)
        rlw = float(getattr(args, "reward_loss_weight", 1.0))
        if getattr(args, "shuffle_targets", False):
            # Falsification: break the state↔r0 correspondence too, so r_model calibration cannot
            # improve and the (ii) reward gate must FAIL alongside the value gate.
            np.random.RandomState(args.seed + 4243).shuffle(r0_flat)
        if trainer.reward_hybrid and trainer.reward_popart is not None:
            trainer.reward_popart.state = update_statistics(
                trainer.reward_popart.state, jnp.asarray(r0_flat), beta=1.0)
            rnorm = np.asarray(trainer.reward_popart.normalize(jnp.asarray(r0_flat)),
                               dtype=np.float32)
        else:
            rmu, rsd = float(r0_flat.mean()), float(max(r0_flat.std(), 1e-6))
            rnorm = ((r0_flat - rmu) / rsd).astype(np.float32)
        rtargets = np.asarray(trainer._scalar_to_categorical(jnp.asarray(rnorm)), dtype=np.float32)

    tx, _label_tree, n_train, n_freeze = make_partition_optimizer(params0, args.lr)
    print(f"[857] freeze partition: {n_train} trainable leaves (dynamics+value), "
          f"{n_freeze} frozen (encoder/reasoning/policy/reward)"
          + (f" | +reward CE (λ={rlw:g}) → dynamics_dense2 reward columns UNFROZEN"
             if codistill else ""))
    opt_state = tx.init(params0)

    if not codistill:
        # ---- shipped 857 value-only path (bit-identical) ----
        def loss_fn(params, lat_b, vec_b, tgt_b):
            logits = value_logits_fn(params, lat_b, vec_b)          # (B, support)
            logp = jax.nn.log_softmax(logits, axis=-1)
            return jnp.mean(-jnp.sum(tgt_b * logp, axis=-1))

        @jax.jit
        def train_step(params, opt_state, lat_b, vec_b, tgt_b):
            loss, grads = jax.value_and_grad(loss_fn)(params, lat_b, vec_b, tgt_b)
            updates, opt_state = tx.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss
    else:
        # ---- TASK_860 co-distill path: value CE + λ·reward CE ----
        def loss_fn(params, lat_b, vec_b, tgt_b, rtgt_b):
            vlogits = value_logits_fn(params, lat_b, vec_b)         # (B, support)
            vlogp = jax.nn.log_softmax(vlogits, axis=-1)
            vloss = jnp.mean(-jnp.sum(tgt_b * vlogp, axis=-1))
            rlogits = reward_logits_fn(params, lat_b, vec_b)        # (B, support)
            rlogp = jax.nn.log_softmax(rlogits, axis=-1)
            rloss = jnp.mean(-jnp.sum(rtgt_b * rlogp, axis=-1))
            return vloss + rlw * rloss, (vloss, rloss)

        @jax.jit
        def train_step(params, opt_state, lat_b, vec_b, tgt_b, rtgt_b):
            (loss, (vloss, rloss)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                params, lat_b, vec_b, tgt_b, rtgt_b)
            updates, opt_state = tx.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, vloss, rloss

    params = params0
    n = lat_flat.shape[0]
    bs = min(args.batch_size, n)
    rng = np.random.RandomState(args.seed + 1)
    losses, rlosses = [], []
    for it in range(args.iters):
        perm = rng.permutation(n)
        ep_loss, ep_rloss = [], []
        for b in range(0, n, bs):
            idx = perm[b:b + bs]
            if not codistill:
                params, opt_state, loss = train_step(
                    params, opt_state,
                    jnp.asarray(lat_flat[idx]), jnp.asarray(vec_flat[idx]),
                    jnp.asarray(targets[idx]))
                ep_loss.append(float(loss))
            else:
                params, opt_state, vloss, rloss = train_step(
                    params, opt_state,
                    jnp.asarray(lat_flat[idx]), jnp.asarray(vec_flat[idx]),
                    jnp.asarray(targets[idx]), jnp.asarray(rtargets[idx]))
                ep_loss.append(float(vloss))     # value CE (consistent with the OFF history labels)
                ep_rloss.append(float(rloss))
        losses.append(float(np.mean(ep_loss)))
        if codistill:
            rlosses.append(float(np.mean(ep_rloss)))
    history = {
        "value_ce_first": losses[0], "value_ce_last": losses[-1],
        "value_ce_head_mean": float(np.mean(losses[:5])),
        "value_ce_tail_mean": float(np.mean(losses[-5:])),
        "all_finite": bool(np.all(np.isfinite(losses))),
        "decreased": bool(losses[-1] < losses[0]),
        "trajectory": losses if len(losses) <= 200 else losses[::max(1, len(losses) // 200)],
        "n_train_leaves": n_train, "n_freeze_leaves": n_freeze,
        "n_tuples": int(n),
        "target_norm_stats": {"mean": float(norm.mean()), "std": float(norm.std()),
                              "min": float(norm.min()), "max": float(norm.max())},
    }
    if codistill:
        history["reward_codistill"] = True
        history["reward_loss_weight"] = rlw
        history["reward_ce_first"] = rlosses[0]
        history["reward_ce_last"] = rlosses[-1]
        history["reward_ce_head_mean"] = float(np.mean(rlosses[:5]))
        history["reward_ce_tail_mean"] = float(np.mean(rlosses[-5:]))
        history["reward_all_finite"] = bool(np.all(np.isfinite(rlosses)))
        history["reward_decreased"] = bool(rlosses[-1] < rlosses[0])
        history["reward_target_norm_stats"] = {
            "mean": float(rnorm.mean()), "std": float(rnorm.std()),
            "min": float(rnorm.min()), "max": float(rnorm.max())}
    return params, history


def _frozen_and_reward_unchanged(model, p0, p1, A):
    """Verify the invariant surface: (a) a frozen leaf (policy output head) is BIT-identical;
    (b) the reward-projection columns of dynamics_dense2 are BIT-identical (reward head frozen
    in effect); (c) a trainable leaf (value_head_3) actually MOVED."""
    f0, f1 = flatten_dict(p0), flatten_dict(p1)
    # (a) a representative frozen policy leaf
    pol_key = next((k for k in f0 if k[0] == "trm" and "output_refiner" in "/".join(map(str, k))
                    and k[-1] == "kernel"), None)
    policy_frozen = pol_key is not None and np.array_equal(
        np.asarray(f0[pol_key]), np.asarray(f1[pol_key]))
    # (b) reward columns of dynamics_dense2 (output units [state_dim:] are the reward head)
    dd2_k = ("dynamics_dense2", "kernel")
    state_dim = int(model.state_dim)
    rw0 = np.asarray(f0[dd2_k])[:, state_dim:]
    rw1 = np.asarray(f1[dd2_k])[:, state_dim:]
    reward_frozen = np.array_equal(rw0, rw1)
    ns_moved = not np.array_equal(np.asarray(f0[dd2_k])[:, :state_dim],
                                  np.asarray(f1[dd2_k])[:, :state_dim])
    # (c) value head moved
    vh_k = ("trm", "reasoning", "value_head_3", "kernel")
    value_moved = not np.array_equal(np.asarray(f0[vh_k]), np.asarray(f1[vh_k]))
    return {
        "policy_head_frozen": bool(policy_frozen),
        "reward_head_frozen": bool(reward_frozen),
        "dynamics_next_state_moved": bool(ns_moved),
        "value_head_moved": bool(value_moved),
    }


# =============================================================================
# driver
# =============================================================================

def run(args, out_path, tag):
    imp = load_imprint_module()
    corridor = imp._load_corridor()

    # ---- panel (TRAINING path, PIN b) + book signs on THAT panel (reuse 832) ----
    prices_j, logc, env_cfg = imp.load_training_panel(args)
    T, A = logc.shape
    tgt, pres, warm = corridor.book_target_signs(logc)
    start_row = max(int(warm), int(env_cfg.window_size))
    # headroom so the R-step true-env leaf never indexes past the panel end.
    N = T - 1 - start_row - args.leaf_rollout_r - 1
    if args.max_decisions is not None:
        N = min(N, int(args.max_decisions))
    if N < 8:
        raise RuntimeError(
            f"panel too short: T={T}, warm={warm}, window={env_cfg.window_size}, "
            f"R={args.leaf_rollout_r} -> N={N} (fixture needs a smaller --bar-minutes)")
    print(f"[857] panel T={T} A={A} warm={warm} start_row={start_row} N={N} "
          f"bar_minutes={args.bar_minutes}")

    # ---- model + trainer (canon arch) + warm start ----
    model, trainer = imp.build_model_and_trainer(args, A)
    warm_start = None
    if args.warm_start and os.path.isdir(args.warm_start):
        # NOTE (PopArt handling — TASK_857 fix-pass FIX 4): load_checkpoint restores the
        # WARM-START checkpoint's PopArt stats (the 832 imprint's, ~mean -8.29). Those are
        # DELIBERATELY re-snapped below (distill_value_head) to the counterfactual-return
        # distribution before training and re-saved with the distilled ckpt, because the value
        # head is regressed on THOSE returns — its saved PopArt must match them, not 832's. This
        # is SAFE for the graded operator regardless: the deployed --leaf-eval vhat-distilled
        # decode is softmax·support (``_categorical_to_scalar``) and NEVER denormalizes through
        # PopArt, so the stored stats are cosmetic for the deployment path.
        loaded = trainer.load_checkpoint(os.path.abspath(args.warm_start))
        if loaded is not None:
            trainer.state = loaded
            warm_start = os.path.relpath(args.warm_start, _REPO)
            print(f"[857] warm-started from {warm_start}")
    if warm_start is None:
        print(f"[857] WARM START ABSENT ({args.warm_start}) — fresh-init net (fixture plumbing "
              f"proof; the real run warms from runs/832_e5/ckpt_imprint_full)")
    params_warm = trainer.state.params

    # ---- training observations for the decision rows (reuse 832 encoder path) ----
    obs_all = imp.encode_training_obs(prices_j, start_row, N, env_cfg)   # (N,27,A,A)

    codistill = bool(getattr(args, "codistill_reward", False))

    # ---- counterfactual dataset (RAM) ----
    vhat_fn, deployed_q_fn, value_logits_fn, reward_logits_fn, r_model_fn = \
        _make_value_fns(model, args.discount)
    passes, _last_cfg = build_counterfactual_dataset(
        model, params_warm, prices_j, env_cfg, obs_all, tgt, corridor,
        start_row, N, A, args)

    # pool + train/held-out split (per-DECISION split so held-out states are unseen by distill)
    def _stack(key):
        arrs = [p[key] for p in passes if p[key].shape[0] > 0]
        return np.concatenate(arrs, axis=0) if arrs else np.zeros((0,), dtype=np.float32)
    all_obs = _stack("dec_obs")           # (D,27,A,A)
    all_cand = _stack("cand")             # (D,M,A)
    all_G = _stack("G")                   # (D,M)
    all_r0 = _stack("r0")                 # (D,M) — TASK_860 reward target
    all_states = [s for p in passes for s in p["states"]]
    D = all_obs.shape[0]
    if D < 4:
        raise RuntimeError(f"too few decisions collected (D={D}); widen the window")
    # deterministic split: every k-th decision held out (>=2 held out)
    hold_stride = max(2, D // max(args.held_out, 1))
    held_mask = np.zeros(D, dtype=bool)
    held_mask[::hold_stride] = True
    if held_mask.sum() < 2:
        held_mask[:2] = True
    train_idx = np.where(~held_mask)[0]
    held_idx = np.where(held_mask)[0]
    print(f"[857] dataset: D={D} decisions, M={all_cand.shape[1]} cands/dec, "
          f"{D * all_cand.shape[1]} tuples; train {train_idx.size} / held-out {held_idx.size}")

    # latents (frozen encoder ⇒ recomputed per params; identical pre/post) for calibration
    def _latents(params, obs_np):
        out = model.apply({"params": params}, jnp.asarray(obs_np), method="initial_inference")
        jax.block_until_ready(out["latent_state"])
        return np.asarray(out["latent_state"], dtype=np.float32)
    lat_warm_all = _latents(params_warm, all_obs)

    # ---- (i) calibration — the 849 instrument = MEAN PER-STATE Pearson on HELD-OUT states ----
    # Two predictors: (a) v̂ ALONE (the spec's literal (i) wording) and (b) the DEPLOYED
    # Q = r_model + γ·v̂ (what the model leaf actually ranks by — the honest deployment
    # predictor). HELD-OUT per-state = the +0.5-bar statistic (graded on the REAL panel by the
    # planner); FULL is secondary context. Pooled inside each is the inflated secondary number.
    def _bundle(pred_fn, params, lat, tgt_arr=all_G):
        return {
            "held_out": calibration(pred_fn, params, lat[held_idx],
                                    all_cand[held_idx], tgt_arr[held_idx]),
            "full": calibration(pred_fn, params, lat, all_cand, tgt_arr),
        }
    cal_vhat_pre = _bundle(vhat_fn, params_warm, lat_warm_all)
    cal_q_pre = _bundle(deployed_q_fn, params_warm, lat_warm_all)
    # TASK_860: the r_model-vs-true-r0 calibration column (the direct measure this task fixes),
    # against the r0 targets (NOT G). Computed only when co-distilling (OFF ⇒ shipped 857).
    cal_r_pre = _bundle(r_model_fn, params_warm, lat_warm_all, tgt_arr=all_r0) if codistill else None
    print(f"[857] calibration PRE  (mean per-state Pearson, HELD-OUT — the +0.5-bar statistic): "
          f"v̂-alone r={cal_vhat_pre['held_out']['mean_per_state_pearson']:+.4f} | "
          f"deployed-Q r={cal_q_pre['held_out']['mean_per_state_pearson']:+.4f}")

    # ---- (2) value-head imprint (dynamics+value only) on TRAIN decisions
    #      + TASK_860: optional reward co-distill on true-r0 targets (--codistill-reward) ----
    distilled, history = distill_value_head(
        model, trainer, value_logits_fn,
        lat_warm_all[train_idx], all_cand[train_idx], all_G[train_idx], args,
        reward_logits_fn=reward_logits_fn, r0s=all_r0[train_idx])
    print(f"[857] distill value CE: {history['value_ce_first']:.4f} -> "
          f"{history['value_ce_last']:.4f} (head5 {history['value_ce_head_mean']:.4f} -> "
          f"tail5 {history['value_ce_tail_mean']:.4f}) decreased={history['decreased']} "
          f"finite={history['all_finite']}")
    invariants = _frozen_and_reward_unchanged(model, params_warm, distilled, A)
    print(f"[857] freeze check: {invariants}")

    # ---- (i) calibration POST-imprint ----
    lat_post_all = _latents(distilled, all_obs)     # encoder frozen ⇒ == lat_warm_all
    cal_vhat_post = _bundle(vhat_fn, distilled, lat_post_all)
    cal_q_post = _bundle(deployed_q_fn, distilled, lat_post_all)
    vhat_held_pre = cal_vhat_pre["held_out"]["mean_per_state_pearson"]
    vhat_held_post = cal_vhat_post["held_out"]["mean_per_state_pearson"]
    q_held_pre = cal_q_pre["held_out"]["mean_per_state_pearson"]
    q_held_post = cal_q_post["held_out"]["mean_per_state_pearson"]
    print(f"[857] calibration POST (mean per-state Pearson, HELD-OUT — the +0.5-bar statistic): "
          f"v̂-alone r={vhat_held_post:+.4f} (from {vhat_held_pre:+.4f}) | "
          f"deployed-Q r={q_held_post:+.4f} (from {q_held_pre:+.4f})  [bar +0.5 on the REAL leg]")
    print(f"[857]   secondary (pooled, inflated): v̂-alone held-out "
          f"{cal_vhat_post['held_out']['pooled_pearson']:+.4f} | full "
          f"{cal_vhat_post['full']['mean_per_state_pearson']:+.4f} per-state")

    # ---- TASK_860 (3): r_model-vs-true-r0 calibration column (the direct measure this fixes) ----
    cal_r_post = r_held_pre = r_held_post = None
    r_train_fit = v_train_fit = None
    if codistill:
        cal_r_post = _bundle(r_model_fn, distilled, lat_post_all, tgt_arr=all_r0)
        r_held_pre = cal_r_pre["held_out"]["mean_per_state_pearson"]
        r_held_post = cal_r_post["held_out"]["mean_per_state_pearson"]
        print(f"[857] REWARD co-distill (TASK_860): r_model vs true-r0 held-out per-state "
              f"r={r_held_post:+.4f} (from {r_held_pre:+.4f}) [frozen baseline ~+0.27 pooled] | "
              f"reward CE {history['reward_ce_first']:.4f} -> {history['reward_ce_last']:.4f} "
              f"decreased={history['reward_decreased']}")
        # TASK_860 FALSIFICATION-ROBUST SIGNAL (fixes the fail-closed blocker: held-out per-state
        # Q / r_model at n=12 are pure noise and INVERT per-seed, so no threshold on them separates
        # legit from --shuffle-targets sabotage). Instead measure the TRAIN-SET pooled correlation
        # of the POST-distill heads against the TRUE (never-shuffled) targets: a legit run fits true
        # r0/G ⇒ high corr; --shuffle-targets fits a PERMUTATION of true ⇒ the head matches the
        # permutation, so corr vs TRUE is ~0. Large-n pooled ⇒ high SNR ⇒ robustly separable.
        def _pooled_pred(pred_fn, params, lat_np, cand_np):
            outs = []
            for d in range(cand_np.shape[0]):
                lat = jnp.asarray(np.repeat(lat_np[d][None], cand_np.shape[1], axis=0))
                outs.append(np.asarray(pred_fn(params, lat, jnp.asarray(cand_np[d])),
                                       dtype=np.float64))
            return np.concatenate(outs)
        r_train_fit = _pearson(
            _pooled_pred(r_model_fn, distilled, lat_post_all[train_idx], all_cand[train_idx]),
            all_r0[train_idx].reshape(-1))
        v_train_fit = _pearson(
            _pooled_pred(vhat_fn, distilled, lat_post_all[train_idx], all_cand[train_idx]),
            all_G[train_idx].reshape(-1))
        print(f"[857] TASK_860 falsification gate — TRAIN-SET fit vs TRUE targets "
              f"(n={int(train_idx.size * all_cand.shape[1])}): r_model·true-r0={r_train_fit:+.4f} | "
              f"v̂·true-G={v_train_fit:+.4f} (legit high; --shuffle-targets ~0)")
    # LOUD, scientifically load-bearing note for the planner: deployed-Q < v̂-alone because the
    # FROZEN 832 reward head is a poor r0 estimator (corr(r_model, true r0) is low), dragging the
    # deployable ranking below the v̂-in-isolation calibration. Clearing a v̂-alone bar does NOT
    # establish the DEPLOYABLE operator ranks like the true env — the reward head may be the next
    # locus (a planner decision; TASK_857 does NOT touch it).
    if np.isfinite(q_held_post) and np.isfinite(vhat_held_post) and q_held_post < vhat_held_post:
        print("*" * 84)
        print(f"*** [857] DEPLOYMENT-PREDICTOR GAP: deployed-Q calibration ({q_held_post:+.4f}) < "
              f"v̂-alone ({vhat_held_post:+.4f}).")
        print("*** The model leaf ranks by Q = r_model + γ·v̂; the FROZEN 832 reward head is a poor")
        print("*** r0 estimator, so a well-calibrated v̂ is dragged off in the DEPLOYED ranking.")
        print("*** The (ii) acceptance gate is deployed-Q (NOT v̂-alone, NOT argmax-%). The reward")
        print("*** head may be the next locus (planner decision — TASK_857 does not touch it).")
        print("*" * 84)

    # ---- (ii) THE FAIL-CLOSED-SAFE DEPLOYABLE-RANKING FLOOR — reject strong degeneracy/sabotage --
    # The argmax-CHOICE ranking-transfer % (below) is NOT discriminative at fixture scale
    # (sims=8/K=9/12 states: shared rng draws + the FROZEN reward head dominate SELECTION, so a
    # good AND a garbage leaf both score ~58% — verified: --shuffle-targets collapses deployed-Q
    # to ~-0.5 yet the % does not move). The instrument that MOVES with the value head is the
    # deployed-Q held-out mean per-state calibration (FIX 2): the DEPLOYED leaf ranks by
    # Q = r_model + γ·v̂ vs the true-env per-candidate return.
    #
    # IMPORTANT (seed-robustness, per the round-3 panel): at fixture scale the LEGITIMATE distill's
    # ABSOLUTE deployed-Q is WITHIN NOISE and seed-dependent — measured across seeds 0-3 it is
    # {+0.115, -0.014, -0.119, +0.491} (the frozen 832 reward head gives NO reliable deployed-Q
    # lift; often the distill even nudges it down vs the warm start). So an absolute `> 0` floor
    # FALSE-FAILS ~half of legitimate re-runs. This gate therefore is NOT a positive validation of
    # deployed-ranking lift (that verdict is the planner's real-data corridor read) — it is a
    # FAIL-CLOSED-SAFE floor that only rejects STRONG anti-correlation / target-shuffle sabotage.
    # The floor sits BELOW the legitimate noise band (min ~-0.12 across seeds) and WELL ABOVE
    # target-shuffled sabotage (~-0.49, consistent across seeds): -0.25 (0.13 under legit-min,
    # 0.24 over sabotage). This is the honest ceiling of what the fixture can check on this axis.
    deployed_ranking_ok = bool(np.isfinite(q_held_post)
                               and q_held_post > DEPLOYED_Q_DEGENERACY_FLOOR)
    print(f"[857] (ii) deployable-ranking floor: deployed-Q held-out per-state r={q_held_post:+.4f} "
          f"> {DEPLOYED_Q_DEGENERACY_FLOOR:+.2f} ? {deployed_ranking_ok} "
          f"(fail-closed-safe non-degeneracy floor; rejects target-shuffle sabotage ~-0.5, NOT a "
          f"lift claim — legit deployed-Q is within-noise/seed-dependent at fixture scale)")

    ckpt_path = os.path.abspath(args.ckpt_out)   # written AFTER asserts pass (FIX B); not yet

    # ---- (ii-diagnostic) argmax-choice ranking transfer (REPORTED ONLY; spec's literal (ii)) ----
    held_states = [all_states[i] for i in held_idx]
    held_obs = [all_obs[i] for i in held_idx]
    pct_agree, n_rank = ranking_transfer(
        model, distilled, prices_j, env_cfg, held_states, held_obs, A, args)
    M = int(all_cand.shape[1])
    chance_pct = 100.0 / M
    print(f"[857] ranking transfer (DIAGNOSTIC, not discriminative at fixture scale): v̂-leaf "
          f"reproduces true-env-leaf choice on {pct_agree:.1f}% of {n_rank} held-out states "
          f"(chance ~{chance_pct:.1f}%; real-leg bar >=70% is the planner's leg)")

    result = {
        "tag": tag, "db": args.db, "window": [args.start, args.end],
        "bar_minutes": args.bar_minutes, "n_assets": A, "n_panel_rows": T,
        "warm": int(warm), "start_row": start_row, "n_decisions": N,
        "warm_start": warm_start,
        "cost_tiers_bp": list(GATE_COSTS),
        "dataset": {
            "mechanism": "true-env-leaf counterfactuals (851 _true_env_leaf_values, tail OFF): "
                         "teacher + true-env-leaf-greedy walks x {gross,net@21}; K dithered-book "
                         "candidates (hb=%.1f) + policy-argmax; G=r0+gamma*V (R=%d)"
                         % (args.vector_hold_bias, args.leaf_rollout_r),
            "n_decisions_total": int(D),
            "candidates_per_decision": int(all_cand.shape[1]),
            "n_tuples_total": int(D * all_cand.shape[1]),
            "n_train_decisions": int(train_idx.size),
            "n_heldout_decisions": int(held_idx.size),
            "passes": [{"policy": p["policy"], "cost": p["cost"],
                        "n_decisions": int(p["dec_obs"].shape[0])} for p in passes],
            "storage": "RAM (numpy); no disk I/O in the train loop (efficiency invariant)",
        },
        "imprint": {
            "recipe": "BC value regression v̂(g(s,a))->true R-step return; dynamics+value "
                      "gradients only, policy/reward/encoder frozen (optax multi_transform); "
                      "PopArt-normalized two-hot targets (shipped value-loss form)",
            "config": {
                "iters": args.iters, "batch_size": args.batch_size, "lr": args.lr,
                "discount": args.discount, "leaf_rollout_r": args.leaf_rollout_r,
                "vector_k": args.vector_k, "sims": args.sims,
                "hold_bias": args.vector_hold_bias, "dither_scale": args.vector_dither_scale,
                "state_dim": args.state_dim, "h_cycles": args.h_cycles,
                "l_cycles": args.l_cycles, "seed": args.seed,
            },
            "history": history,
            "freeze_invariants": invariants,
        },
        "calibration": {
            "instrument": "849 calibration column = MEAN PER-STATE Pearson of the predictor vs "
                          "the true R-step return, across candidates WITHIN a state, meaned over "
                          "HELD-OUT states (mean_per_state_pearson = PRIMARY; pooled_pearson = "
                          "inflated secondary). Two predictors: 'vhat_alone' (the spec's literal "
                          "(i)) and 'deployed_q' = r_model+gamma*v_hat (what the leaf actually "
                          "ranks by — the honest deployment predictor + the one that tracks "
                          "ranking-transfer). Real-leg bar +0.5 on HELD-OUT per-state; graded on "
                          "the REAL panel by the planner.",
            "headline_statistic": "mean_per_state_pearson on HELD-OUT states",
            "bar_real_leg": 0.5,
            "vhat_alone": {
                "held_out": {"pre": cal_vhat_pre["held_out"], "post": cal_vhat_post["held_out"],
                             "per_state_delta": (vhat_held_post - vhat_held_pre)},
                "full": {"pre": cal_vhat_pre["full"], "post": cal_vhat_post["full"]},
            },
            "deployed_q": {
                "note": "r_model + gamma*v_hat — the DEPLOYED model-leaf ranking quantity; this "
                        "(not vhat_alone) is what the (ii) fail-closed-safe floor keys on (see "
                        "deployable_ranking_gate). It does NOT improve here (frozen reward head) and "
                        "is within-noise/seed-dependent at fixture scale — the (ii) floor only "
                        "rejects strong sabotage, it is NOT a lift/positive-validation claim.",
                "held_out": {"pre": cal_q_pre["held_out"], "post": cal_q_post["held_out"],
                             "per_state_delta": (q_held_post - q_held_pre)},
                "full": {"pre": cal_q_pre["full"], "post": cal_q_post["full"]},
            },
            "deployment_predictor_gap": {
                "vhat_alone_held_out_per_state_post": vhat_held_post,
                "deployed_q_held_out_per_state_post": q_held_post,
                "gap": (vhat_held_post - q_held_post),
                "reason": "frozen 832 reward head is a poor r0 estimator; drags the deployable "
                          "ranking below the v̂-in-isolation calibration. Reward head = candidate "
                          "next locus (planner decision; TASK_857 does not touch it).",
            },
            # PRIMARY improvement gate: HELD-OUT mean-per-state delta (v̂-alone = the spec's (i)),
            # both finite AND positive. NOT the train-contaminated pooled number.
            "improved_held_out_per_state": bool(
                np.isfinite(vhat_held_post) and np.isfinite(vhat_held_pre)
                and vhat_held_post > vhat_held_pre),
        },
        # (ii) FAIL-CLOSED-SAFE deployable-ranking floor — reject strong degeneracy/sabotage.
        "deployable_ranking_gate": {
            "instrument": "deployed-Q (r_model+gamma*v_hat) held-out MEAN PER-STATE Pearson vs the "
                          "true-env per-candidate return G — the DEPLOYED leaf's ranking quantity. "
                          "This MOVES with the value head (target-shuffle sabotage collapses it to "
                          "~-0.5), unlike argmax-choice agreement.",
            "gate": f"deployed_q held-out mean-per-state Pearson is finite AND > "
                    f"{DEPLOYED_Q_DEGENERACY_FLOOR:+.2f}. FAIL-CLOSED-SAFE non-degeneracy floor: it "
                    f"rejects STRONG anti-correlation / --shuffle-targets sabotage (~-0.5) but is "
                    f"NOT a positive validation of deployed-ranking lift. The LEGITIMATE distill's "
                    f"absolute deployed-Q is within-noise and seed-dependent at fixture scale "
                    f"({{+0.115,-0.014,-0.119,+0.491}} across seeds 0-3; frozen reward head gives no "
                    f"reliable lift) — so the floor sits below that band and above sabotage. The "
                    f"deployed-ranking VERDICT is the planner's real-data corridor read.",
            "deployed_q_held_out_per_state": q_held_post,
            "floor": DEPLOYED_Q_DEGENERACY_FLOOR,
            "passes_floor": deployed_ranking_ok,
            "seed_note": "absolute deployed-Q is within-noise/seed-dependent at fixture scale; this "
                         "floor is fail-closed-safe (rejects sabotage), NOT a lift claim.",
            "real_leg_bar": ">=0.70 argmax-choice agreement on the real panel (planner's leg)",
        },
        "ranking_transfer": {
            "instrument": "ARGMAX-choice agreement: v̂-leaf (model-leaf == eval_vector --leaf-eval "
                          "vhat-distilled) vs true-env-leaf SELECTED-VECTOR agreement on held-out "
                          "states (same rng => same candidate set; only the leaf differs). "
                          "REPORTED DIAGNOSTIC ONLY — the spec's literal (ii); the >=70% bar is the "
                          "planner's real-data leg. NOT discriminative at fixture scale: shared rng "
                          "draws + the frozen reward head dominate SELECTION, so a good and a "
                          "degenerate leaf both score ~58% (the fail-closed-safe (ii) floor is "
                          "deployable_ranking_gate). chance = 100/M is an informational bound, not "
                          "the acceptance gate.",
            "pct_agree": pct_agree, "n_states": n_rank,
            "chance_pct": chance_pct, "n_candidates": M,
            "discriminative": False,
        },
        "deployable_read": {
            "note": "the graded corridor read is a SCHEDULED planner run: "
                    "eval_vector.py --operator search --leaf-eval vhat-distilled on the real "
                    "panel vs floor +0.424 (first) / +8.02 true-leaf ceiling (honest reference). "
                    "net_dependent=true (the v̂ leaf is the learned component, even at pw0).",
            "eval_command": ("python experiments/835_stage4_eval/eval_vector.py --operator "
                             "search --leaf-eval vhat-distilled --checkpoint <distilled_ckpt>"),
        },
        "checkpoint": ckpt_path,
    }

    # ---- TASK_860: reward co-distill deliverables (added ONLY when --codistill-reward; OFF ⇒
    #      the result dict above is bit-identical to the shipped 857 output) ----
    reward_improved = None
    if codistill:
        reward_improved = bool(np.isfinite(r_held_post) and np.isfinite(r_held_pre)
                               and r_held_post > r_held_pre)
        result["calibration"]["r_model"] = {
            "note": "r_model (softmax·support of the reward logits) vs the TRUE r0 per candidate — "
                    "the direct measure TASK_860 fixes (the frozen 832 reward head baseline is "
                    "~+0.27 pooled). Held-out MEAN PER-STATE Pearson, pre vs post co-distill.",
            "held_out": {"pre": cal_r_pre["held_out"], "post": cal_r_post["held_out"],
                         "per_state_delta": (r_held_post - r_held_pre)},
            "full": {"pre": cal_r_pre["full"], "post": cal_r_post["full"]},
            # MINOR 2: this held-out per-state delta is REPORTED, NOT a gate. It is NON-DISCRIMINATIVE
            # at fixture scale (n=12): --shuffle-targets sabotage often shows a LARGER 'improvement'
            # than a legit run (fits a permutation whose per-state ordering happens to correlate).
            # Do NOT read this flag as validation — the robust signal is train_fit_vs_true below.
            "improved_held_out_per_state": reward_improved,
            "non_discriminative_at_fixture_scale": True,
        }
        result["codistill_reward"] = {
            "enabled": True,
            "reward_loss_weight": float(getattr(args, "reward_loss_weight", 1.0)),
            # MINOR 1: honest loss-form claim. The shipped trainer's reward_hybrid loss is
            # 0.8*Huber + 0.2*CE; here we use the SAME PopArt(reward_popart) normalizer + two-hot
            # TARGET but a pure-CE loss (matching 857's value-path simplification and the spec's
            # "λ·reward CE" wording) — NOT the trainer's Huber/CE blend.
            "recipe": "value CE (as shipped 857) + λ·reward CE on the true-r0 counterfactual "
                      "targets; same PopArt(reward_popart) normalizer + two-hot TARGET as the "
                      "shipped trainer, CE loss (matching 857's value path; NOT the trainer's "
                      "0.8*Huber+0.2*CE reward_hybrid blend). The added reward CE supplies gradient "
                      "to the dynamics_dense2 reward columns — the ONE new leg that trains; policy "
                      "head + encoder stay frozen.",
            "reward_head_moved": bool(not invariants["reward_head_frozen"]),
            # THE hard falsification gate (fail-closed-robust across seeds 0-3): TRAIN-SET pooled
            # correlation of the POST-distill heads vs the TRUE (never-shuffled) targets. Legit fits
            # true ⇒ high; --shuffle-targets fits a permutation ⇒ ~0. Large-n ⇒ high SNR.
            "train_fit_vs_true": {
                "r_model_pooled_corr_true_r0": r_train_fit,
                "vhat_pooled_corr_true_G": v_train_fit,
                "n_train_tuples": int(train_idx.size * all_cand.shape[1]),
                "floor": TRAIN_FIT_FLOOR,
                "passes_gate": bool(np.isfinite(r_train_fit) and np.isfinite(v_train_fit)
                                    and r_train_fit > TRAIN_FIT_FLOOR
                                    and v_train_fit > TRAIN_FIT_FLOOR),
                "note": "the ROBUST falsification gate (spec acceptance iv): legit-pass / "
                        "--shuffle-targets-fail on every seed 0-3. Held-out per-state Q and r_model "
                        "(n=12) are pure noise here and INVERT per-seed — reported directions only.",
            },
            "note_deployed_q": "deployed-Q (r_model+γ·v̂) held-out per-state is a REPORTED direction "
                               "only; at fixture n=12 it is noise and seed-dependent (legit and "
                               "sabotage overlap/invert), so it is NOT the gate. The deployed-ranking "
                               "VERDICT is the planner's scheduled real-panel corridor read.",
        }

    # ---- smoke gates (raise BEFORE any deliverable is written — FIX B) ----
    if args.smoke_assert:
        assert history["all_finite"], "SMOKE FAIL: non-finite distill loss"
        assert history["decreased"], (
            f"SMOKE FAIL: value CE did not decrease "
            f"({history['value_ce_head_mean']:.4f} -> {history['value_ce_tail_mean']:.4f})")
        assert invariants["policy_head_frozen"], "SMOKE FAIL: policy head not frozen"
        if codistill:
            # TASK_860: the reward projection is now the ONE new leg that trains.
            assert not invariants["reward_head_frozen"], (
                "SMOKE FAIL: (codistill) dynamics_dense2 reward columns did not train")
            assert history.get("reward_all_finite", False), \
                "SMOKE FAIL: (codistill) non-finite reward CE"
        else:
            assert invariants["reward_head_frozen"], "SMOKE FAIL: reward head not frozen"
        assert invariants["value_head_moved"], "SMOKE FAIL: value head did not train"
        # held-out per-state calibration must be finite (both predictors) — a plumbing check.
        assert np.isfinite(vhat_held_post) and np.isfinite(q_held_post), \
            "SMOKE FAIL: held-out per-state calibration non-finite (v̂-alone or deployed-Q)"
        if not codistill:
            # ===== shipped 857 value-only gates — UNCHANGED (OFF stays byte-identical) =====
            # (i) v̂-in-isolation is the gated quantity (frozen reward head ⇒ deployed-Q can't lift).
            assert result["calibration"]["improved_held_out_per_state"], (
                f"SMOKE FAIL: (i) held-out per-state calibration (the 849 instrument) did not improve "
                f"(v̂-alone pre {vhat_held_pre:+.4f} -> post {vhat_held_post:+.4f})")
            # (ii) FAIL-CLOSED-SAFE deployable-ranking floor: reject strong degeneracy / --shuffle
            # sabotage on the deployed-Q axis (frozen reward head ⇒ this floor is discriminative HERE).
            assert deployed_ranking_ok, (
                f"SMOKE FAIL: (ii) deployable-ranking floor — deployed-Q (r_model+γ·v̂) held-out "
                f"per-state calibration {q_held_post:+.4f} is non-finite or <= the degeneracy floor "
                f"{DEPLOYED_Q_DEGENERACY_FLOOR:+.2f}; the DEPLOYED leaf is strongly anti-correlated "
                f"with the true-env ranking (target-shuffle-sabotage-like degeneracy)")
            print("[857] SMOKE PASS: finite+decreasing value CE; policy/reward frozen + value "
                  "trained; (i) 849 instrument (HELD-OUT mean per-state Pearson) IMPROVED "
                  f"[v̂-alone {vhat_held_pre:+.4f}->{vhat_held_post:+.4f}]; (ii) deployable-ranking "
                  f"floor deployed-Q held-out r={q_held_post:+.4f} > {DEPLOYED_Q_DEGENERACY_FLOOR:+.2f} "
                  f"(fail-closed-safe, not a lift claim); argmax-% diagnostic {pct_agree:.1f}% (reported)")
        else:
            # ===== TASK_860 co-distill gates — ROBUST at fixture scale (spec acceptance iv) =====
            # THE falsification gate: the trained heads must FIT THE TRUE (never-shuffled) targets on
            # the TRAIN set (large n ⇒ high SNR). Legit ⇒ high corr; --shuffle-targets trains on a
            # PERMUTATION of true ⇒ corr vs true ~0 ⇒ FAILS. This is what separates signal from
            # sabotage on EVERY seed 0-3 — unlike the held-out per-state Q / r_model which at n=12 are
            # pure noise and INVERT per-seed (those are reported directions only, never gated).
            assert np.isfinite(r_train_fit) and np.isfinite(v_train_fit), \
                "SMOKE FAIL: (codistill) train-set fit-vs-true non-finite"
            assert r_train_fit > TRAIN_FIT_FLOOR and v_train_fit > TRAIN_FIT_FLOOR, (
                f"SMOKE FAIL: (codistill falsification gate, spec-iv) the trained heads did NOT fit "
                f"the TRUE targets on the train set (r_model·true-r0={r_train_fit:+.4f}, "
                f"v̂·true-G={v_train_fit:+.4f}; floor {TRAIN_FIT_FLOOR:+.2f}). Either training failed "
                f"or the targets were permuted (--shuffle-targets sabotage).")
            print(f"[857] (reported directions, NOISY at n=12 — NOT gated): deployed-Q held-out "
                  f"{q_held_pre:+.4f}->{q_held_post:+.4f} | r_model·true-r0 held-out "
                  f"{r_held_pre:+.4f}->{r_held_post:+.4f} (real grading = planner's real-panel leg)")
            print(f"[857] SMOKE PASS (codistill/TASK_860): finite+decreasing value+reward CE; policy "
                  f"frozen + value trained + REWARD columns trained; FALSIFICATION GATE (spec-iv) "
                  f"TRAIN-SET fit vs TRUE: r_model·true-r0={r_train_fit:+.4f}, v̂·true-G="
                  f"{v_train_fit:+.4f} (both > {TRAIN_FIT_FLOOR:+.2f}; --shuffle-targets FAILS this)")

    # ---- deliverables — written ONLY after all smoke asserts pass (FIX B). A failing run
    #      raises above and never reaches here; a --shuffle-targets falsification run NEVER
    #      writes the shipped ckpt_smoke / results JSON (deliverable preserved). ----
    if getattr(args, "shuffle_targets", False):
        print("[857] --shuffle-targets falsification run: NOT writing the deliverable "
              "(ckpt_smoke / results JSON preserved).")
    else:
        trainer.state = trainer.state.replace(params=distilled)
        # TASK_876 fix-2 (M2): as in 832's imprint.py — `overwrite=True` keeps this one-shot
        # runner re-runnable under the D32 default (overwrite=False on the training path).
        # `ckpt_path` is a single-purpose, gitignored OUTPUT directory (`ckpt_smoke/` /
        # `ckpt_distilled/`, see this module's .gitignore) holding only the distilled head
        # this invocation produced; nothing resumes from a prior invocation's copy and no
        # newer checkpoint can exist there, so the mass-delete overwrite=False guards against
        # is impossible. Without it the second run of the same command aborts on
        # `Destination .../checkpoint_<step> already exists`.
        trainer.save_checkpoint(ckpt_path, overwrite=True)
        print(f"[857] distilled checkpoint -> {ckpt_path}")
        if out_path:
            with open(out_path, "w") as fh:
                json.dump(result, fh, indent=2, default=float)
            print(f"[857] wrote {out_path}")
    return result


def main():
    ap = argparse.ArgumentParser(description="TASK_857 counterfactual v̂-leaf distill runner")
    ap.add_argument("--db", default=FIXTURE_DB,
                    help="minute-ohlcv DB (default: fixture; scheduled run points at trading_data.db)")
    ap.add_argument("--start", default=None, help="default: fixture 2025-09-14 / real 2021-01-01")
    ap.add_argument("--end", default=None, help="default: fixture 2025-12-12 / real 2025-09-01")
    ap.add_argument("--bar-minutes", type=int, default=None,
                    help="training resample (default: 240 fixture / 1440 real)")
    ap.add_argument("--n-assets", type=int, default=None, help="default: 8 fixture / 64 real")
    ap.add_argument("--max-decisions", type=int, default=None,
                    help="cap decisions PER walk (default: 30 fixture / None real)")
    ap.add_argument("--position-size-pct", type=float, default=None,
                    help="per-asset notional fraction (default 0.5/n_assets, 830 convention)")
    ap.add_argument("--fee-bps", type=float, default=FEE_BPS)
    ap.add_argument("--slip-bps", type=float, default=SLIP_BPS)
    # weight-rebalance env passthrough (mirror 832; default OFF = order semantics, canon)
    ap.add_argument("--vector-weight-rebalance", dest="vector_weight_rebalance",
                    action="store_true", default=False)
    ap.add_argument("--vector-weight-unit", type=float, default=-1.0)
    ap.add_argument("--vector-weight-max", type=float, default=4.0)
    ap.add_argument("--vector-rebalance-deadband", type=float, default=-1.0)
    # distill knobs
    ap.add_argument("--iters", type=int, default=None,
                    help="distill epochs (default: 80 fixture / 2000 real)")
    ap.add_argument("--batch-size", type=int, default=None, help="default: 64 fixture / 512 real")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--discount", type=float, default=0.997)
    ap.add_argument("--held-out", type=int, default=12,
                    help="approx number of held-out decisions for calibration/ranking")
    # search / candidate machinery (dithered-book, the deployment distribution)
    ap.add_argument("--vector-k", type=int, default=None, help="K candidates (default 8 fixture)")
    ap.add_argument("--sims", type=int, default=None, help="search sims (default 8 fixture)")
    ap.add_argument("--vector-hold-bias", type=float, default=3.0)
    ap.add_argument("--vector-dither-scale", type=float, default=1.0)
    ap.add_argument("--gumbel-scale", type=float, default=3.0)
    ap.add_argument("--leaf-rollout-r", type=int, default=3,
                    help="true-env leaf HOLD-rollout depth R (counterfactual return horizon)")
    ap.add_argument("--max-positions", type=int, default=16)
    ap.add_argument("--min-cash-threshold", type=float, default=100.0)
    # arch (mirror 832 defaults so the warm-start checkpoint restores)
    ap.add_argument("--state-dim", type=int, default=64)
    ap.add_argument("--h-cycles", type=int, default=4)
    ap.add_argument("--l-cycles", type=int, default=6)
    ap.add_argument("--reward-hybrid", dest="reward_hybrid", action="store_true", default=True)
    ap.add_argument("--no-reward-hybrid", dest="reward_hybrid", action="store_false")
    ap.add_argument("--reward-attributable", dest="reward_attributable",
                    action="store_true", default=True)
    ap.add_argument("--no-reward-attributable", dest="reward_attributable", action="store_false")
    ap.add_argument("--entropy-coef", type=float, default=0.001)   # trainer ctor (imprint parity)
    ap.add_argument("--min-entropy", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warm-start", default=None,
                    help="warm-start checkpoint dir (default: 832 ckpt_smoke fixture / "
                         "runs/832_e5/ckpt_imprint_full real)")
    ap.add_argument("--ckpt-out", default=None,
                    help="distilled checkpoint dir (default: <exp>/ckpt_smoke fixture)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke-assert", action="store_true", default=None,
                    help="assert finite+decreasing CE + freeze + calibration improvement "
                         "(default ON for the fixture db)")
    ap.add_argument("--no-smoke-assert", dest="smoke_assert", action="store_false")
    ap.add_argument("--shuffle-targets", dest="shuffle_targets", action="store_true",
                    default=False,
                    help="FALSIFICATION spot-check (default OFF): permute the training targets so "
                         "the distill is degenerate — the (i)/(ii) smoke gates must FAIL. Proves "
                         "the gates are non-hollow.")
    # TASK_860 — reward-head co-distill (default OFF = shipped 857 behavior bit-identical)
    ap.add_argument("--codistill-reward", dest="codistill_reward", action="store_true",
                    default=False,
                    help="TASK_860 (default OFF): co-distill the reward head on true-r0 "
                         "counterfactual targets ALONGSIDE the value imprint. Unfreezes the "
                         "dynamics_dense2 reward-projection columns via an added λ·reward-CE term "
                         "(SAME PopArt/two-hot form as the shipped trainer); policy head + encoder "
                         "stay frozen. OFF reproduces the shipped 857 deliverables byte-for-byte.")
    ap.add_argument("--reward-loss-weight", type=float, default=1.0,
                    help="TASK_860 λ on the reward CE term (co-distill only; default 1.0).")
    args = ap.parse_args()

    is_fixture = os.path.abspath(args.db) == os.path.abspath(FIXTURE_DB)
    if args.start is None:
        args.start = "2025-09-14" if is_fixture else "2021-01-01"
    if args.end is None:
        args.end = "2025-12-12" if is_fixture else "2025-09-01"
    if args.bar_minutes is None:
        args.bar_minutes = 240 if is_fixture else 1440
    if args.n_assets is None:
        args.n_assets = 8 if is_fixture else 64
    if args.max_decisions is None:
        args.max_decisions = 30 if is_fixture else None
    if args.iters is None:
        args.iters = 80 if is_fixture else 2000
    if args.batch_size is None:
        args.batch_size = 64 if is_fixture else 512
    if args.vector_k is None:
        args.vector_k = 8 if is_fixture else 16
    if args.sims is None:
        args.sims = 8 if is_fixture else 32
    if args.position_size_pct is None:
        args.position_size_pct = 0.5 / args.n_assets
    if args.warm_start is None:
        args.warm_start = (os.path.join(_REPO, "experiments", "832_linear_imprint", "ckpt_smoke")
                           if is_fixture else os.path.join(_REPO, "runs", "832_e5",
                                                           "ckpt_imprint_full"))
    if args.ckpt_out is None:
        args.ckpt_out = os.path.join(EXP_DIR, "ckpt_smoke" if is_fixture else "ckpt_distilled")
    if args.smoke_assert is None:
        args.smoke_assert = is_fixture
    tag = args.tag or ("fixture-smoke" if is_fixture else "real-panel")
    out = args.out
    if out is None and is_fixture:
        out = os.path.join(EXP_DIR, "results_fixture_smoke.json")
    run(args, out, tag)


if __name__ == "__main__":
    main()
