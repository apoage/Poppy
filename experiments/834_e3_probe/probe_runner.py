#!/usr/bin/env python3
"""TASK_834 / E3 — DEC-1 attribution runner (checkpoint-loading R_model vs R_true).

The REAL DEC-1 gate on SAVED training checkpoints. The shipped probe
(``tools/tests/test_vector_reward_probe.py``) is an untrained-model, in-process
sensitivity-vs-numerical-noise harness; THIS runner loads a Stage-3 snapshot
(``jax_ckpt_fast/<name>/snapshot_ep<N>/``) and computes the attribution ratio

    R = E_book[Var_action(reward)] / Var_book[E_action(reward)]

TWICE on the SAME states and SAME action vectors:

  * R_model — decoded scalar reward from the learned reward head
    (``initial_inference`` -> latent; ``recurrent_inference_vector`` -> reward
    logits -> expected-support scalar, the canon search decode).
  * R_true  — the true env reward for the identical (state, action-vector)
    pairs via ``jit_batch_step_vector_unique`` (mcts_trader/env/jax_trading_env.py).

Verdict = DEC-1 (tasks/DECISIONS.md, quoted verbatim in SPEC.md):
``R_model >= 0.5 * R_true`` (calibrated primary); fallback absolute
``R_model >= 0.5``, FLAGGED, if the R_true leg is unavailable. The Stage-B ~1/60
action-blind baseline is reported as a distance-travelled reference line.

Design (all pre-registered in SPEC.md):
  * Checkpoint load: model+trainer rebuilt from the RUN'S TrainingConfig, provided
    as ``--config-json`` (TrainingConfig field overrides; unknown keys attached via
    setattr, mirroring mcts_trader/training/cli.py's dynamic attrs). Shapes come
    from the config — nothing is hard-coded to small/64. ``trainer_metadata.json``
    in the snapshot is cross-checked against the built model before restore.
  * State distribution: N states collected by walking the run's OWN env
    (``create_jax_env_historical`` — same DB loader, ``--periods-file``/
    ``assets_file`` universe order, ``bar_minutes`` resample, fees, burn flags,
    reward_attributable) forward with actions drawn from the ZERO-logit
    dithered-book proposal (HOLD-book), legality-masked per state exactly as
    battery self-play (``jit_batch_compute_action_mask`` with the run's
    max_positions/min_cash_threshold — the mask, not the env, enforces the order
    cap), so every snapshot of a run is probed on the IDENTICAL seed-fixed,
    training-reachable state sequence. States at/after an env's done flag are
    excluded from the sample.
  * Action distribution: M i.i.d. action-vectors per state from the canon proposal
    builder ``SampledMuZeroVectorSearch._build_proposal`` (hold-bias / dither from
    the run config — the 828 Stage-3 battery operator is hold-bias 3.0, dither 1.0)
    on the checkpoint's factored policy logits (``--proposal-base zero`` = control),
    with the SAME per-state legality mask (illegal slots -> -1e9, canon root).
  * Fixture smoke (``--fixture-smoke``): fresh-init untrained model on
    ``fixtures/trading_fixture.db``, exercising the REAL save->load path
    (``trainer.save_checkpoint`` to a temp dir, reloaded by a SECOND trainer, param
    round-trip asserted) then the full probe end-to-end on CPU. Plumbing proof:
    finite values + schema; NO DEC-1 threshold applied to the untrained case.

CPU-only (JAX_PLATFORMS=cpu); fixture db by default; never GPU in the factory
loop. The real snapshot probe is a scheduled run (planner-owned), not this smoke.

Run:  JAX_PLATFORMS=cpu python experiments/834_e3_probe/probe_runner.py --fixture-smoke
      JAX_PLATFORMS=cpu python experiments/834_e3_probe/probe_runner.py \
          --checkpoint jax_ckpt_fast/<name>/snapshot_ep<N> --config-json <run>.json
"""
import os
# CPU-only, deterministic — keep JAX off the GPU (factory smoke / scheduled-CPU job).
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import dataclasses
import json
import math
import shutil
import sys
import tempfile
import time

import numpy as np
import jax
import jax.numpy as jnp

# Run from anywhere: repo root on the path so `mcts_trader` imports cleanly.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from mcts_trader.training.config import TrainingConfig
from mcts_trader.training.setup import setup_model_and_trainer
from mcts_trader.training.env_factory import create_jax_env_historical
from mcts_trader.env.jax_trading_env import (
    batch_compute_action_mask, batch_step_vector, compute_observation_27ch,
    jit_batch_step_vector_unique,
)
from mcts_trader.jax_agent.sampled_search import SampledMuZeroVectorSearch

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
FIXTURE_DB = os.path.join(_REPO, "fixtures", "trading_fixture.db")
SMOKE_OUT = os.path.join(MODULE_DIR, "results_fixture_smoke.json")

HOLD = 1
N_ACTIONS_PER_ASSET = 5
EPS = 1e-10                      # division guard (CLAUDE.md: jnp.maximum(x, 1e-10))
BATTERY_HOLD_BIAS = 3.0          # 828 Stage-3 operator (CLI default 0.0 is a trap)
STAGE_B_BASELINE_R = 1.0 / 60.0  # Stage-B reward head ~60x action-blind variance gap
DEC1_RULE = ("R_model >= 0.5 * R_true (calibrated primary, DEC-1); fallback: "
             "absolute R_model >= 0.5, FLAGGED, if the R_true leg is unavailable")

REQUIRED_KEYS = (
    "task", "mode", "checkpoint", "db", "n_states", "m_actions", "seed",
    "sampling", "config", "r_model", "r_true", "ratio_model_over_true",
    "dec1", "stage_b_baseline_R", "jax_version", "timestamp",
)


# =============================================================================
# config provision (pre-registered: --config-json = TrainingConfig overrides)
# =============================================================================

def build_training_config(config_json_path):
    """RUN config from a JSON of TrainingConfig field overrides.

    Known dataclass fields go through the constructor; unknown keys are attached
    with setattr — mirroring mcts_trader/training/cli.py, which sets dynamic
    attrs (e.g. ``reward_hybrid``, ``use_jax_env``) the dataclass doesn't declare.
    """
    with open(config_json_path) as f:
        raw = json.load(f)
    field_names = {f.name for f in dataclasses.fields(TrainingConfig)}
    known = {k: v for k, v in raw.items() if k in field_names}
    extra = {k: v for k, v in raw.items() if k not in field_names}
    cfg = TrainingConfig(**known)
    for k, v in sorted(extra.items()):
        setattr(cfg, k, v)
    if extra:
        print(f"[config] non-dataclass keys attached via setattr: {sorted(extra)}")
    return cfg


def smoke_training_config(seed):
    """Fixture-smoke config (pre-registered in SPEC.md).

    bar_minutes=60: the fixture's 1-min ``ohlcv`` table spans only ~90 days
    (2025-09-13..2025-12-12); the battery's 1440-min bars would leave too few
    rows for window(60)+episode. Hourly bars give ~427 rows on the fixture's
    common grid — enough for the walk while exercising the same resample path.
    """
    cfg = TrainingConfig(
        name="834_smoke", seed=seed, parallel_envs=2, steps=96, epochs=1,
        sims=8, batch_size=32, model_size="small",
        fee=16, slippage=5,                      # battery primary = fee16/slip5
        db_path=FIXTURE_DB,
        start_date="2025-09-14", end_date="2025-12-11",
        reward_attributable=True,                # battery: --reward-attributable
        bar_minutes=60,
        vector_action=True,
        vector_hold_bias=BATTERY_HOLD_BIAS,      # the battery operator
        vector_dither_scale=1.0,
        vector_proposal_scheme="dithered-book",
    )
    cfg.use_jax_env = True
    cfg.reward_hybrid = True                     # trainer-side only; noted in SPEC
    return cfg


# =============================================================================
# model + checkpoint
# =============================================================================

def build_model(cfg, n_assets, init_seed):
    """Model+trainer from the RUN config (shapes from config, never hard-coded)."""
    if not getattr(cfg, "vector_action", False):
        raise SystemExit("TASK_834: the run config must set vector_action=true "
                         "(this probe measures the vector reward head).")
    model, trainer = setup_model_and_trainer(cfg, n_assets)
    assert model.use_vector_action, "model must be built use_vector_action=True"
    dummy_obs = jnp.ones((1, 27, n_assets, n_assets))
    trainer.state = trainer.init_train_state(jax.random.PRNGKey(init_seed), dummy_obs)
    return model, trainer


def check_ckpt_metadata(ckpt_path, model):
    """Cross-check snapshot ``trainer_metadata.json`` vs the built model.

    ``trainer.save_checkpoint`` persists state_dim / h_cycles / l_cycles /
    use_prediction_head (trainer.py:1656); a mismatch means the config JSON does
    not describe this run — fail loudly BEFORE a confusing flax restore error.
    """
    meta_path = os.path.join(ckpt_path, "trainer_metadata.json")
    if not os.path.exists(meta_path):
        # FAIL LOUD (verify-panel MAJOR 2): flax restore_checkpoint(step=None)
        # silently returns the target UNCHANGED on a missing/empty dir — a bad
        # path would otherwise probe the fresh init and emit a false verdict.
        raise SystemExit(
            f"TASK_834: no trainer_metadata.json in {ckpt_path!r} — not a "
            "trainer.save_checkpoint snapshot (missing/empty/wrong dir?). "
            "Refusing to continue: flax restore would silently return the "
            "fresh-init params and produce a FALSE DEC-1 verdict.")
    with open(meta_path) as f:
        meta = json.load(f)
    checks = (("state_dim", model.state_dim), ("h_cycles", model.H_cycles),
              ("l_cycles", model.L_cycles),
              ("use_prediction_head", model.use_prediction_head))
    bad = [f"{k}: ckpt={meta.get(k)!r} vs config={v!r}"
           for k, v in checks if meta.get(k) is not None and meta.get(k) != v]
    if bad:
        raise SystemExit("TASK_834: checkpoint/config architecture mismatch — "
                         "fix --config-json:\n  " + "\n  ".join(bad))


# =============================================================================
# env (the run's own creation path — train_jax_fast.py:240 mapping, verbatim)
# =============================================================================

def build_env(cfg, n_assets, n_envs, seed, db_override=None):
    start_date, end_date = cfg.start_date, cfg.end_date
    specific_symbols = None
    if getattr(cfg, "periods_file", None):
        with open(cfg.periods_file) as f:
            periods_config = json.load(f)
        first_period = periods_config["periods"][0]
        start_date = first_period["start_date"]
        end_date = first_period["end_date"]
        specific_symbols = first_period.get("assets")
        print(f"[env] periods file: {cfg.periods_file} "
              f"({start_date[:10]} -> {end_date[:10]}, "
              f"{len(specific_symbols) if specific_symbols else 0} pinned symbols)")
    elif getattr(cfg, "assets_file", None):
        # train_jax_fast.py:233-236 — assets_file pins the universe when no
        # periods file is given; silently ignoring it would probe the wrong
        # universe (verify-panel MINOR 3).
        with open(cfg.assets_file) as f:
            specific_symbols = json.load(f)
        print(f"[env] assets file: {cfg.assets_file} "
              f"({len(specific_symbols)} pinned symbols)")
    db_path = db_override or cfg.db_path
    states, prices, env_cfg = create_jax_env_historical(
        db_path=db_path,
        start_date=start_date,
        end_date=end_date,
        n_envs=n_envs,
        n_assets=n_assets,
        specific_symbols=specific_symbols,
        fee_bps=float(cfg.fee),
        slippage_bps=float(cfg.slippage),
        max_episode_steps=cfg.steps,
        borrow_rate_annual=cfg.borrow_rate,
        burning_money=getattr(cfg, "burning_money", False),
        burn_market_linked=getattr(cfg, "burn_market_linked", True),
        burn_base_rate=getattr(cfg, "burn_base_rate", 0.00005),
        burn_random=getattr(cfg, "burn_random", False),
        burn_random_prob=getattr(cfg, "burn_random_prob", 0.02),
        burn_random_max=getattr(cfg, "burn_random_max", 500.0),
        burn_random_limit=getattr(cfg, "burn_random_limit", 10),
        reward_attributable=getattr(cfg, "reward_attributable", False),
        bar_minutes=getattr(cfg, "bar_minutes", 1),
        vector_action=True,
        rng_key=jax.random.PRNGKey(seed),
    )
    n_rows = int(prices.shape[0])
    need = env_cfg.window_size + env_cfg.max_steps + 11
    if n_rows <= need:
        raise SystemExit(
            f"TASK_834: panel too short ({n_rows} rows <= window+max_steps+11={need}); "
            "reduce --steps / bar_minutes or widen the window.")
    return states, prices, env_cfg, {"db": db_path, "start": start_date,
                                     "end": end_date, "n_rows": n_rows}


# =============================================================================
# state walk (pre-registered: zero-logit dithered-book proposal, HOLD book)
# =============================================================================

def make_walk_step(searcher, prices, env_cfg, n_envs, max_positions,
                   min_cash_threshold):
    """Jitted: (states, key) -> (new_states, obs, legal_mask, dones) for the
    PRE-step (decision-point) states.

    Actions come from the canon proposal builder on ZERO policy logits (book=None
    -> the HOLD slot gets the hold_bias, sampled_search.py:_build_proposal), so
    the walk is checkpoint-independent: every snapshot of a run is probed on the
    identical seed-fixed state sequence. The legality mask is computed exactly as
    battery self-play does (self_play.py:661-674 — jit_batch_compute_action_mask
    with the run's max_positions/min_cash_threshold) and masks illegal per-asset
    slots to -1e9 inside _build_proposal, keeping the walk on the
    training-reachable distribution (the env does NOT enforce max_positions —
    the mask is the only cap). Obs = compute_observation_27ch — the same simple
    27ch encoder the jax self-play path trains on.
    """
    A = env_cfg.n_assets
    zero_logits = jnp.zeros((n_envs, A * N_ACTIONS_PER_ASSET), dtype=jnp.float32)

    def _step(states, key):
        dither_key, sample_key = jax.random.split(key)
        # Current prices exactly as self-play (price_idx = start + step).
        price_idx = states.episode_start_idx + states.current_step
        current_prices = prices[price_idx]                            # (n_envs, A)
        mask = batch_compute_action_mask(
            states, current_prices, env_cfg,
            max_positions, min_cash_threshold)                        # (n_envs, A*5)
        proposal = searcher._build_proposal(zero_logits, mask, None, dither_key)
        action_vecs = jax.random.categorical(
            sample_key, proposal, axis=-1).astype(jnp.int32)          # (n_envs, A)
        obs = jax.vmap(
            lambda s: compute_observation_27ch(s, prices, env_cfg))(states)
        new_states, _, _, dones = batch_step_vector(
            states, action_vecs, prices, env_cfg)
        return new_states, obs, mask, dones

    return jax.jit(_step)


def run_walk(walk_step, init_states, seed, burn_in, collect_steps):
    """Walk burn_in+collect_steps steps; collect the PRE-step (decision-point)
    states, their observations and legality masks for the last collect_steps
    steps, plus a per-collected-state validity flag that excludes states AT or
    AFTER an env's done flag (post-done states are degenerate — MINOR 4)."""
    states = init_states
    n_envs = int(init_states.cash.shape[0])
    base_key = jax.random.PRNGKey(seed)
    kept_states, kept_obs, kept_masks, kept_valid = [], [], [], []
    done_ever = np.zeros(n_envs, dtype=bool)
    first_done = -1
    for t in range(burn_in + collect_steps):
        new_states, obs, mask, dones = walk_step(states, jax.random.fold_in(base_key, t))
        dones_np = np.asarray(dones).astype(bool)
        if t >= burn_in:
            kept_states.append(states)     # the state the obs/mask were computed on
            kept_obs.append(obs)
            kept_masks.append(mask)
            # valid = env not done before this step AND this step didn't flag done
            kept_valid.append(~(done_ever | dones_np))
        if first_done < 0 and dones_np.any():
            first_done = t
        done_ever = done_ever | dones_np
        states = new_states
    batched_states = jax.tree_util.tree_map(
        lambda *xs: jnp.concatenate(xs, axis=0), *kept_states)
    obs_all = jnp.concatenate(kept_obs, axis=0)
    masks_all = jnp.concatenate(kept_masks, axis=0)
    valid = np.concatenate(kept_valid, axis=0)
    jax.block_until_ready(obs_all)
    return batched_states, obs_all, masks_all, valid, first_done


# =============================================================================
# R_model leg (initial_inference -> recurrent_inference_vector -> decoded reward)
# =============================================================================

def make_model_probe(model, searcher, m_actions, proposal_base):
    """Jitted: (params, obs, masks, dither_key, sample_key) ->
    (rewards (N,M), actions (N,M,A)).

    Mirrors the canon root (sampled_search.py:_search) exactly: one proposal
    build per state (single dither draw) WITH the per-state legality mask
    (illegal slots -> -1e9, sampled_search.py:209-211), then M i.i.d. per-asset
    categorical draws from softmax(proposal). Reward decode =
    searcher._reward_to_scalar (expected value over the 401-bin support — the
    canon search decode).
    """
    A = searcher.n_assets

    def _probe(params, obs, masks, dither_key, sample_key):
        init = model.apply({"params": params}, obs, mode="initial_inference")
        latent = init["latent_state"]                      # (N, state_dim)
        pol = init["policy_logits"]                        # (N, A*5)
        base = pol if proposal_base == "policy" else jnp.zeros_like(pol)
        proposal = searcher._build_proposal(base, masks, None, dither_key)  # (N,A,5)
        n_states = obs.shape[0]
        samples = jax.random.categorical(
            sample_key, proposal, axis=-1, shape=(m_actions, n_states, A))
        action_vecs = jnp.transpose(samples, (1, 0, 2)).astype(jnp.int32)  # (N,M,A)
        latent_rep = jnp.repeat(latent, m_actions, axis=0)                 # (N*M,D)
        act_flat = action_vecs.reshape(n_states * m_actions, A)
        rec = model.apply({"params": params}, latent_rep, act_flat,
                          mode="recurrent_inference_vector")
        rewards = searcher._reward_to_scalar(rec["reward"])
        return rewards.reshape(n_states, m_actions), action_vecs

    return jax.jit(_probe)


# =============================================================================
# R_true leg (SAME states/actions through the true env reward)
# =============================================================================

def true_rewards(batched_states, action_vecs, prices, env_cfg, chunk_states):
    """(N,M) true env rewards via jit_batch_step_vector_unique, chunked over
    states to bound the tiled-prices footprint (prices broadcast per chunk)."""
    n_states, m_actions, A = action_vecs.shape
    out = np.zeros((n_states, m_actions), dtype=np.float64)
    for lo in range(0, n_states, chunk_states):
        hi = min(lo + chunk_states, n_states)
        tiled = jax.tree_util.tree_map(
            lambda x: jnp.repeat(x[lo:hi], m_actions, axis=0), batched_states)
        acts = action_vecs[lo:hi].reshape((hi - lo) * m_actions, A)
        prices_b = jnp.broadcast_to(prices, ((hi - lo) * m_actions,) + prices.shape)
        _, _, rewards, _ = jit_batch_step_vector_unique(tiled, acts, prices_b, env_cfg)
        jax.block_until_ready(rewards)
        out[lo:hi] = np.asarray(rewards, dtype=np.float64).reshape(hi - lo, m_actions)
    return out


# =============================================================================
# attribution decomposition + DEC-1 verdict
# =============================================================================

def attribution(rewards_nm):
    """R = E_book[Var_action] / Var_book[E_action] (population variance, ddof=0;
    identical decomposition for both legs so any bias cancels in the comparison)."""
    r = jnp.asarray(rewards_nm)
    var_within = jnp.var(r, axis=1)                    # Var_action per state
    e_book_var_action = jnp.mean(var_within)           # E_book[Var_action]
    mean_within = jnp.mean(r, axis=1)                  # E_action per state
    var_book_e_action = jnp.var(mean_within)           # Var_book[E_action]
    ratio = e_book_var_action / jnp.maximum(var_book_e_action, EPS)
    return {
        "R": float(ratio),
        "e_book_var_action": float(e_book_var_action),
        "var_book_e_action": float(var_book_e_action),
    }


def dec1_verdict(r_model, r_true, smoke):
    """Apply DEC-1 verbatim (calibrated primary; flagged absolute fallback)."""
    if smoke:
        return {
            "rule": DEC1_RULE,
            "threshold_applied": False,
            "fallback_used": False,
            "passed": None,
            "verdict": ("SMOKE (untrained fresh-init model) — plumbing proof only; "
                        "DEC-1 threshold NOT applied to the untrained case "
                        "(per TASK_834). Real verdict = scheduled run on the "
                        "828s3 mid/late snapshots."),
        }
    if r_true is not None:
        bar = 0.5 * r_true["R"]
        passed = bool(r_model["R"] >= bar)
        text = (f"DEC-1 calibrated: R_model={r_model['R']:.6g} "
                f"{'>=' if passed else '<'} 0.5 x R_true={r_true['R']:.6g} "
                f"(= {bar:.6g}) -> {'PASS' if passed else 'FAIL'}")
        return {"rule": DEC1_RULE, "threshold_applied": True,
                "fallback_used": False, "passed": passed, "verdict": text}
    passed = bool(r_model["R"] >= 0.5)
    text = (f"DEC-1 FALLBACK (R_true leg unavailable): absolute "
            f"R_model={r_model['R']:.6g} {'>=' if passed else '<'} 0.5 "
            f"-> {'PASS' if passed else 'FAIL'} — FLAGGED: fallback used, "
            "calibrated leg missing")
    return {"rule": DEC1_RULE, "threshold_applied": True,
            "fallback_used": True, "passed": passed, "verdict": text}


# =============================================================================
# probe core (shared by checkpoint mode and the fixture smoke)
# =============================================================================

def probe_core(model, params, cfg, env_pack, args, checkpoint_label, smoke,
               restored_step_count=None):
    init_states, prices, env_cfg, env_info = env_pack
    n_envs = int(init_states.cash.shape[0])
    collect_steps = math.ceil(args.n_states / n_envs)
    n_collected = collect_steps * n_envs
    if args.burn_in + collect_steps + 1 > env_cfg.max_steps:
        raise SystemExit(
            f"TASK_834: walk ({args.burn_in}+{collect_steps}+1 steps) exceeds the "
            f"run's max_steps={env_cfg.max_steps}; lower --n-states/--burn-in or "
            "raise config steps.")

    hold_bias = float(getattr(cfg, "vector_hold_bias", 0.0))
    if hold_bias == 0.0:
        print("[probe] WARNING: vector_hold_bias=0.0 — the 828 Stage-3 battery "
              f"operator is hold-bias {BATTERY_HOLD_BIAS} (the CLI default 0.0 is "
              "a documented trap; check the config JSON).")
    searcher = SampledMuZeroVectorSearch(
        model_apply_fn=None,                       # sampling machinery only (831 pattern)
        n_assets=env_cfg.n_assets,
        n_actions_per_asset=N_ACTIONS_PER_ASSET,
        num_samples=args.m_actions,
        hold_bias=hold_bias,
        dither_scale=float(getattr(cfg, "vector_dither_scale", 1.0)),
        proposal_scheme=str(getattr(cfg, "vector_proposal_scheme", "dithered-book")),
    )

    # 1. state distribution: seed-fixed env walk on the LEGAL (masked) proposal,
    #    with the run's max_positions / min_cash_threshold (train_jax_fast.py:486).
    max_positions = int(getattr(cfg, "max_positions", 16))
    min_cash_threshold = float(getattr(cfg, "min_cash_threshold", 100.0))
    walk_step = make_walk_step(searcher, prices, env_cfg, n_envs,
                               max_positions, min_cash_threshold)
    batched_states, obs_all, masks_all, valid, first_done = run_walk(
        walk_step, init_states, args.seed, args.burn_in, collect_steps)

    # MINOR 4: exclude states at/after an env's done flag (post-done states are
    # degenerate); fail loud if the surviving sample collapsed.
    keep_idx = np.nonzero(valid)[0]
    n_excluded = n_collected - len(keep_idx)
    if len(keep_idx) < max(2, args.n_states // 2):
        raise SystemExit(
            f"TASK_834: only {len(keep_idx)}/{n_collected} collected states "
            f"survive the post-done exclusion (< half the requested "
            f"{args.n_states}) — walk degenerated (first done at step "
            f"{first_done}); re-seed or shorten the walk.")
    if n_excluded:
        batched_states = jax.tree_util.tree_map(lambda x: x[keep_idx], batched_states)
        obs_all = obs_all[keep_idx]
        masks_all = masks_all[keep_idx]
    n_states = len(keep_idx)
    print(f"[walk] collected {n_collected} states "
          f"({n_envs} envs x {collect_steps} steps, burn-in {args.burn_in}); "
          f"first done flag at step {first_done}; "
          f"{n_excluded} post-done states excluded -> N={n_states}")

    # Mask / book diagnostics on the surviving sample (training-reachable check).
    masks_np = np.asarray(masks_all, dtype=np.int8)
    frac_illegal_slots = float(1.0 - masks_np.mean())
    n_orders_per_state = np.asarray(
        jnp.sum(batched_states.pos_active.astype(jnp.int32), axis=1)
        + jnp.sum(batched_states.order_active.astype(jnp.int32), axis=1))

    # 2. R_model: M proposal-sampled action vectors per state -> decoded reward,
    #    proposal masked per state exactly as the canon root.
    model_probe = make_model_probe(model, searcher, args.m_actions, args.proposal_base)
    probe_key = jax.random.fold_in(jax.random.PRNGKey(args.seed), 10_000)
    dither_key, sample_key = jax.random.split(probe_key)
    r_model_nm, action_vecs = model_probe(params, obs_all, masks_all,
                                          dither_key, sample_key)
    jax.block_until_ready(r_model_nm)
    r_model_nm = np.asarray(r_model_nm, dtype=np.float64)
    pct_hold = float(jnp.mean((action_vecs == HOLD).astype(jnp.float32)))

    # Legality verification of the sampled probe actions (must be ~0 illegal).
    acts_np = np.asarray(action_vecs)                       # (N, M, A)
    masks_pa = masks_np.reshape(n_states, env_cfg.n_assets, N_ACTIONS_PER_ASSET)
    legal = np.take_along_axis(
        np.broadcast_to(masks_pa[:, None],
                        (n_states, args.m_actions) + masks_pa.shape[1:]),
        acts_np[..., None], axis=-1)[..., 0]
    frac_illegal_vectors = float(np.mean(np.any(legal == 0, axis=-1)))
    print(f"[mask] illegal slots on collected states: {frac_illegal_slots:.4f}; "
          f"probe vectors containing an illegal action: {frac_illegal_vectors:.4f}; "
          f"orders/state mean={n_orders_per_state.mean():.2f} "
          f"max={int(n_orders_per_state.max())} (cap {max_positions})")

    r_model = attribution(r_model_nm)
    print(f"[R_model] E_book[Var_action]={r_model['e_book_var_action']:.6g}  "
          f"Var_book[E_action]={r_model['var_book_e_action']:.6g}  "
          f"R={r_model['R']:.6g}")

    # 3. R_true: SAME states/actions through the true env reward.
    r_true = None
    r_true_error = None
    if args.skip_true:
        r_true_error = "--skip-true set (fallback exercise)"
        print("[R_true] SKIPPED (--skip-true) — DEC-1 fallback leg engaged")
    else:
        try:
            r_true_nm = true_rewards(batched_states, action_vecs, prices,
                                     env_cfg, args.chunk_states)
            r_true = attribution(r_true_nm)
            print(f"[R_true ] E_book[Var_action]={r_true['e_book_var_action']:.6g}  "
                  f"Var_book[E_action]={r_true['var_book_e_action']:.6g}  "
                  f"R={r_true['R']:.6g}")
        except Exception as exc:                       # fallback path per DEC-1
            r_true_error = f"{type(exc).__name__}: {exc}"
            print(f"[R_true ] UNAVAILABLE ({r_true_error}) — DEC-1 fallback engaged")

    dec1 = dec1_verdict(r_model, r_true, smoke)
    ratio = (float(r_model["R"] / max(r_true["R"], EPS))
             if r_true is not None else None)

    result = {
        "task": "TASK_834 E3/DEC-1 attribution probe",
        "mode": "fixture-smoke" if smoke else "checkpoint",
        "checkpoint": checkpoint_label,
        "db": env_info["db"],
        "window": [env_info["start"], env_info["end"]],
        "panel_rows": env_info["n_rows"],
        "n_states": n_states,
        "n_states_collected": n_collected,
        "n_states_excluded_post_done": n_excluded,
        "m_actions": args.m_actions,
        "n_envs": n_envs,
        "burn_in": args.burn_in,
        "seed": args.seed,
        "restored_step_count": restored_step_count,
        "sampling": {
            "state_method": ("env-walk: create_jax_env_historical reset + "
                             "zero-logit dithered-book proposal actions, "
                             "legality-masked per state (jit_batch_compute_"
                             "action_mask, self_play.py operator) — "
                             "snapshot-independent, seed-fixed"),
            "action_method": (f"M i.i.d. draws/state from "
                              f"SampledMuZeroVectorSearch._build_proposal on "
                              f"{args.proposal_base} logits + per-state legality "
                              "mask (one dither draw per state, canon root "
                              "behavior)"),
            "proposal_base": args.proposal_base,
            "proposal_scheme": searcher.proposal_scheme,
            "hold_bias": hold_bias,
            "dither_scale": searcher.dither_scale,
            "mask_max_positions": max_positions,
            "mask_min_cash_threshold": min_cash_threshold,
            "mask_frac_illegal_slots": frac_illegal_slots,
            "probe_actions_illegal_vector_frac": frac_illegal_vectors,
            "orders_per_state_mean": float(n_orders_per_state.mean()),
            "orders_per_state_max": int(n_orders_per_state.max()),
            "pct_hold_probe_actions": pct_hold,
            "walk_first_done_step": first_done,
        },
        "config": {
            "model_size": cfg.model_size, "state_dim": cfg.state_dim,
            "h_cycles": cfg.h_cycles, "l_cycles": cfg.l_cycles,
            "n_assets": env_cfg.n_assets, "fee_bps": float(env_cfg.fee_bps),
            "slippage_bps": float(env_cfg.slippage_bps),
            "max_steps": int(env_cfg.max_steps),
            "window_size": int(env_cfg.window_size),
            "bar_minutes": getattr(cfg, "bar_minutes", 1),
            "reward_attributable": bool(env_cfg.reward_attributable),
            "reward_hybrid": bool(getattr(cfg, "reward_hybrid", False)),
            "vector_action": True,
        },
        "r_model": r_model,
        "r_true": r_true,
        "r_true_error": r_true_error,
        "ratio_model_over_true": ratio,
        "dec1": dec1,
        "stage_b_baseline_R": STAGE_B_BASELINE_R,
        "stage_b_note": ("Stage-B reward head measured ~60x action-blind "
                         "(R ~ 1/60) — distance-travelled reference, "
                         "project_stageb_reward_head_diseased"),
        "jax_version": jax.__version__,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    return result


# =============================================================================
# fixture smoke (REAL save->load path + end-to-end probe; no threshold)
# =============================================================================

def run_fixture_smoke(args):
    cfg = smoke_training_config(args.seed)
    n_assets = args.n_assets

    # Fresh-init model A -> save through the REAL trainer checkpoint path.
    model_a, trainer_a = build_model(cfg, n_assets, init_seed=args.seed)
    saved_params = trainer_a.state.params
    tmpdir = tempfile.mkdtemp(prefix="e3_probe_ckpt_")
    try:
        trainer_a.save_checkpoint(tmpdir)
        print(f"[smoke] checkpoint saved -> {tmpdir} ({sorted(os.listdir(tmpdir))})")

        # SECOND trainer, DIFFERENT init -> restore must overwrite its params.
        model_b, trainer_b = build_model(cfg, n_assets, init_seed=args.seed + 9999)
        fresh_leaves = jax.tree_util.tree_leaves(trainer_b.state.params)
        check_ckpt_metadata(tmpdir, model_b)
        trainer_b.state = trainer_b.load_checkpoint(tmpdir)
        restored_leaves = jax.tree_util.tree_leaves(trainer_b.state.params)
        saved_leaves = jax.tree_util.tree_leaves(saved_params)
        assert all(np.allclose(a, b) for a, b in zip(saved_leaves, restored_leaves)), \
            "SMOKE FAIL: restored params != saved params (save->load round trip broken)"
        assert any(not np.allclose(a, b) for a, b in zip(fresh_leaves, restored_leaves)), \
            "SMOKE FAIL: restore did not overwrite the fresh init (load path inert)"
        print("[smoke] save->load round trip verified (params restored, fresh init overwritten)")

        env_pack = build_env(cfg, n_assets, args.n_envs or cfg.parallel_envs,
                             args.seed, db_override=args.db)
        result = probe_core(
            model_b, trainer_b.state.params, cfg, env_pack, args,
            checkpoint_label=("fresh-init (fixture smoke; saved+reloaded via "
                              "trainer.save_checkpoint/load_checkpoint)"),
            smoke=True,
            restored_step_count=int(trainer_b.state.step_count))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    # --- smoke assertions: plumbing proof, NOT a verdict --------------------
    for leg_name, leg in (("r_model", result["r_model"]), ("r_true", result["r_true"])):
        assert leg is not None, f"SMOKE FAIL: {leg_name} leg missing"
        for k, v in leg.items():
            assert np.isfinite(v), f"SMOKE FAIL: {leg_name}.{k} not finite ({v})"
    missing = [k for k in REQUIRED_KEYS if k not in result]
    assert not missing, f"SMOKE FAIL: JSON schema incomplete, missing {missing}"
    assert result["dec1"]["threshold_applied"] is False, \
        "SMOKE FAIL: DEC-1 threshold must NOT be applied to the untrained case"
    assert result["ratio_model_over_true"] is not None and \
        np.isfinite(result["ratio_model_over_true"]), \
        "SMOKE FAIL: ratio_model_over_true not finite"
    assert result["sampling"]["probe_actions_illegal_vector_frac"] == 0.0, \
        ("SMOKE FAIL: masked proposal still sampled training-illegal actions "
         f"({result['sampling']['probe_actions_illegal_vector_frac']:.4f})")
    # Order-cap sanity: the mask is computed ONCE per vector step (same
    # semantics as self-play), so several simultaneous opens can overshoot
    # max_positions in the crossing step — training-faithful. The hard bound is
    # (max_positions - 1) + n_assets: no NEW opens are legal once the cap is hit.
    _cap_bound = (result["sampling"]["mask_max_positions"] - 1
                  + result["config"]["n_assets"])
    assert result["sampling"]["orders_per_state_max"] <= _cap_bound, \
        ("SMOKE FAIL: walk exceeded the masked order-cap bound "
         f"({result['sampling']['orders_per_state_max']} > {_cap_bound})")

    out_path = args.out or SMOKE_OUT
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1)
    print(f"[smoke] results -> {out_path}")
    print(f"[smoke] R_model={result['r_model']['R']:.6g}  "
          f"R_true={result['r_true']['R']:.6g}  "
          f"ratio={result['ratio_model_over_true']:.6g}  "
          f"(Stage-B baseline ~{STAGE_B_BASELINE_R:.4f})")
    print("[SMOKE] PASS — end-to-end plumbing verified (finite legs, schema "
          "complete, save->load exercised, no threshold applied)")
    return result


# =============================================================================
# checkpoint mode (the real DEC-1 gate on a saved snapshot)
# =============================================================================

def run_checkpoint(args):
    cfg = build_training_config(args.config_json)
    if args.db:
        print(f"[config] db override: {args.db}")
    n_assets = args.n_assets
    model, trainer = build_model(cfg, n_assets, init_seed=args.seed)
    ckpt = os.path.abspath(args.checkpoint)
    check_ckpt_metadata(ckpt, model)

    # MAJOR 2 guards: flax restore_checkpoint(step=None) returns the target
    # UNCHANGED on a missing/empty dir — assert the restore actually happened.
    fresh_leaves = jax.tree_util.tree_leaves(trainer.state.params)
    trainer.state = trainer.load_checkpoint(ckpt)
    step_count = int(trainer.state.step_count)
    if step_count <= 0:
        raise SystemExit(
            f"TASK_834: restored step_count={step_count} — a trained snapshot "
            f"must have step_count > 0. {ckpt!r} is not a trained checkpoint "
            "(or the restore was a silent no-op). Refusing to emit a verdict.")
    restored_leaves = jax.tree_util.tree_leaves(trainer.state.params)
    if all(np.allclose(a, b) for a, b in zip(fresh_leaves, restored_leaves)):
        raise SystemExit(
            "TASK_834: restored params are IDENTICAL to the fresh init — the "
            f"flax restore from {ckpt!r} was a silent no-op. Refusing to emit "
            "a verdict on a fresh-init model.")
    print(f"[ckpt] restored {ckpt} (step_count={step_count}; params differ "
          "from fresh init — restore verified)")

    env_pack = build_env(cfg, n_assets, args.n_envs or cfg.parallel_envs,
                         args.seed, db_override=args.db)
    result = probe_core(model, trainer.state.params, cfg, env_pack, args,
                        checkpoint_label=ckpt, smoke=False,
                        restored_step_count=step_count)
    out_path = args.out or os.path.join(MODULE_DIR, "results_probe.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1)
    print(f"[probe] results -> {out_path}")
    print(f"[probe] {result['dec1']['verdict']}")
    return result


# =============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fixture-smoke", action="store_true",
                    help="fresh-init untrained model on the fixture db; exercises "
                         "the real save->load path end-to-end (no DEC-1 threshold)")
    ap.add_argument("--checkpoint", default=None,
                    help="snapshot dir (jax_ckpt_fast/<name>/snapshot_ep<N>)")
    ap.add_argument("--config-json", default=None,
                    help="RUN TrainingConfig as JSON field overrides (required "
                         "with --checkpoint)")
    ap.add_argument("--db", default=None, help="override config db_path")
    ap.add_argument("--n-assets", type=int, default=None,
                    help="run n_assets (checkpoint default 64, smoke default 16)")
    ap.add_argument("--n-states", type=int, default=None,
                    help="N book/state draws (checkpoint 64, smoke 16; rounded up "
                         "to a multiple of n_envs)")
    ap.add_argument("--m-actions", type=int, default=None,
                    help="M action vectors per state (checkpoint 16, smoke 8)")
    ap.add_argument("--n-envs", type=int, default=None,
                    help="parallel walk envs (default: config parallel_envs)")
    ap.add_argument("--burn-in", type=int, default=None,
                    help="walk steps before state collection (checkpoint 8, smoke 6)")
    ap.add_argument("--seed", type=int, default=0, help="probe seed (walk/proposal)")
    ap.add_argument("--proposal-base", choices=("policy", "zero"), default="policy",
                    help="proposal base logits: the checkpoint's factored policy "
                         "(battery-faithful, default) or zeros (control)")
    ap.add_argument("--skip-true", action="store_true",
                    help="skip the R_true leg (exercises the FLAGGED DEC-1 fallback)")
    ap.add_argument("--chunk-states", type=int, default=8,
                    help="states per jit_batch_step_vector_unique chunk (memory bound)")
    ap.add_argument("--out", default=None, help="output JSON path")
    args = ap.parse_args()

    if args.fixture_smoke:
        args.n_assets = args.n_assets or 16
        args.n_states = args.n_states or 16
        args.m_actions = args.m_actions or 8
        args.burn_in = 6 if args.burn_in is None else args.burn_in
        run_fixture_smoke(args)
    else:
        if not args.checkpoint or not args.config_json:
            ap.error("checkpoint mode needs --checkpoint AND --config-json "
                     "(or use --fixture-smoke)")
        args.n_assets = args.n_assets or 64
        args.n_states = args.n_states or 64
        args.m_actions = args.m_actions or 16
        args.burn_in = 8 if args.burn_in is None else args.burn_in
        run_checkpoint(args)


if __name__ == "__main__":
    main()
