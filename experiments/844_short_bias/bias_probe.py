#!/usr/bin/env python3
"""TASK_844 — short-bias postmortem probe (where does the ~10-short standing book come from?).

Both p1b AND p1c policies converge to persistently SHORT-heavy books (≈10 shorts vs
≈1–6 longs) that profit only in bear windows (LEDGER 835/S4-p1b, 835/S4-p1c). The
objective is absolute return (reward-hybrid), so the bias is a LEARNING ARTIFACT with an
unknown locus. This read-only instrument localizes it with three pre-registered probes on
the SAME loaded checkpoint (834/836 pattern — reuse their checkpoint-restore + walk
machinery, do NOT re-derive):

  1. MIRROR-BOOK VALUE PROBE (primary). Sample N states via the 836 deterministic walk
     (weight env, run flags from --config-json). Each sampled state carries a book; take
     its SUPPORT (the assets that hold a position). Construct, from the state's OWN market
     latent (initial_inference on the leak-free 27ch obs, position-blind), the value of
     OPENING that support LONG (book B: BUY on support, HOLD elsewhere) vs its EXACT
     sign-mirror opened SHORT (−B: SELL on support, HOLD elsewhere) — via one
     recurrent_inference_vector dynamics step, decoded with the canon
     _categorical_to_scalar. −B is an element-wise long↔short flip of B on the identical
     asset set (magnitudes symmetric, same latent ⇒ same "cash"); asserted exact, leak-
     checked like 836's alignment audit. Δ = V(B) − V(−B). B is canonicalized to the LONG
     side so the sign is interpretable per the pre-registered read.
     READ (checkpoint mode only): mean Δ significantly < 0 (prefers short books) ⇒ VALUE
     locus; ≈ 0 ⇒ value is not the carrier. (The ≥2-checkpoint aggregation is the
     planner's job across scheduled runs; each run emits its own per-checkpoint read.)

  2. FLAT-BOOK POLICY MARGINALS. At M flat-book states (episode-start book = all cash, no
     positions) spread across the panel, the policy's per-asset marginal mass on
     Sell+LimitSell vs Buy+LimitBuy (raw factored policy softmax = the learned PRIOR;
     legality-masked variant reported as a diagnostic secondary).
     READ: sell-mass / buy-mass ratio > 1.5 ⇒ POLICY-prior locus.

  3. ROLLOUT LEG DECOMPOSITION. Re-run the 835 deterministic rollout (real Sampled-MuZero
     vector search, visit-argmax, same operator flags) and capture per-leg REALIZED PnL by
     side: long-legs vs short-legs realized totals + open/close counts, PLUS the per-step
     reward the TRAINING objective would assign each class — the exact reward_attributable
     term (jax_trading_env.py:1249-1259): realized log-PnL on close, long = log(exit/entry),
     short = −log(exit/entry), exit = decision-row price, entry = pre-close avg entry.
     READ: short legs realize losses but the policy keeps opening them ⇒ credit
     mis-assignment; profited in training-era self-play but not at eval ⇒ distribution shift.

INSTRUMENT INTEGRITY (the panel attacks these): the sign-mirror is asserted EXACT (only
sides flipped on a fixed support, HOLD identical elsewhere, magnitudes/cash preserved);
leg attribution is alignment-clean (each asset closes at most once per step — actions do
not flip a side in one step — pre-step entry × decision-row exit, no double-count); restore
is FAIL-LOUD (834's guard — flax restore_checkpoint(step=None) silently returns the target
UNCHANGED on a bad dir, which the 834 panel caught as a FALSE PASS); determinism is
seed-fixed (same seed ⇒ bit-identical).

CPU-only (JAX_PLATFORMS=cpu); fixture db by default; never GPU in the factory loop. The
real snapshot probe is a SCHEDULED run (planner-owned with the run flags — the 836
precedent), not this smoke.

Run:  JAX_PLATFORMS=cpu python experiments/844_short_bias/bias_probe.py --fixture-smoke
      JAX_PLATFORMS=cpu python experiments/844_short_bias/bias_probe.py \
          --checkpoint jax_ckpt_fast/<name>/snapshot_ep<N> --config-json <run>.json
"""
import os
# CPU-only, deterministic — keep JAX off the GPU (factory smoke / scheduled-CPU job).
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import importlib.util
import json
import shutil
import sys
import tempfile
import time
from datetime import date

import numpy as np
import jax
import jax.numpy as jnp

# Run from anywhere: repo root on the path so `mcts_trader` imports cleanly.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from mcts_trader.training.config import TrainingConfig
from mcts_trader.training.env_factory import create_jax_env_historical
from mcts_trader.env.jax_trading_env import (
    jit_batch_step_vector, jit_batch_compute_action_mask, compute_portfolio_value,
)
from mcts_trader.env.jax_encoder import EncoderConfig, jit_batch_encode_simple
from mcts_trader.jax_agent.sampled_search import (
    SampledMuZeroVectorSearch, create_sampled_vector_search,
)

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
FIXTURE_DB = os.path.join(_REPO, "fixtures", "trading_fixture.db")
SMOKE_OUT = os.path.join(MODULE_DIR, "results_fixture_smoke.json")

# Action encoding (mcts_trader/CLAUDE.md; jax_trading_env.py:1198-1202):
SELL, HOLD, BUY, LIMIT_SELL, LIMIT_BUY = 0, 1, 2, 3, 4
N_ACTIONS_PER_ASSET = 5
EPS = 1e-10                      # division guard (CLAUDE.md: jnp.maximum(x, 1e-10))
BATTERY_HOLD_BIAS = 3.0          # 828 Stage-3 operator (CLI default 0.0 is a trap)
MIN_N = 8                        # below this a mean/ratio is emitted but flagged under-powered

# Pre-registered interpretation thresholds (SPEC.md), quoted in every JSON.
# NOTE (MAJOR-1 calibration): the raw within-state t (VALUE_T_THRESHOLD) is UNCALIBRATED —
# VectorActionEmbedding.vec_act_proj (muzero_model.py:306-314) has independently-initialized
# BUY vs SELL columns, so even a FRESH untrained model gives a per-checkpoint mirror-Δ offset
# with |t| up to ~4 (an arbitrary architectural zero-point, sign varies by init seed). The
# value-locus verdict therefore gates on the NULL-SUBTRACTED, init-standardized statistic
# (calibrated z below), NOT the raw t. The raw t is retained only as a reference diagnostic.
VALUE_T_THRESHOLD = -2.0         # raw within-state t (UNCALIBRATED — reference only)
VALUE_Z_THRESHOLD = -2.0         # calibrated: (Δ_ckpt_offset − null_mean)/null_std_over_inits < −2
MIN_NULL = 3                     # fewer fresh-init nulls than this ⇒ calibration under-powered
POLICY_RATIO_THRESHOLD = 1.5     # sell-mass/buy-mass ratio > 1.5 ⇒ policy-prior locus

# Fresh-init null for the mirror-Δ calibration (MAJOR-1). Dedicated seed base, far from the
# typical probe seeds (0,1,7,8,...) so the null is independent of and never collides with the
# checkpoint/walk seed. Same architecture (built from the run's OWN cfg) ⇒ a MATCHED null.
NULL_INIT_SEED_BASE = 20000
N_NULL_BOOT = 2000               # bootstrap resamples of the null OVER INITS (calibrated CI)

INTERPRETATION_VERBATIM = (
    "mirror-book value mean Δ significantly < 0 (prefers short books) ⇒ VALUE locus; "
    "flat-book sell/buy policy-mass ratio > 1.5 ⇒ POLICY-prior locus; short legs realize "
    "losses but the policy keeps opening them ⇒ REWARD/credit mis-assignment (profited in "
    "training-era self-play but not at eval ⇒ distribution shift). DIAGNOSTIC — this probe "
    "localizes the ~10-short bias; it gates nothing and fixes nothing by itself. The "
    "≥2-checkpoint aggregation is the planner's job across scheduled runs.")

REQUIRED_KEYS = (
    "task", "mode", "checkpoint", "db", "window", "panel", "seed",
    "operator", "mirror_book_value", "flat_book_policy", "rollout_legs",
    "interpretation", "restored_step_count", "config", "leak_check",
    "jax_version", "timestamp",
)


# =============================================================================
# cross-experiment imports (the 832 importlib precedent):
#   834 = checkpoint-load pattern + FAIL-LOUD restore guards + zero-logit walk;
#   835 = read-only row-timestamp loader (panel-consistency mirror).
# =============================================================================

def _load_experiment_module(rel_path, name):
    path = os.path.join(_REPO, rel_path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


probe834 = _load_experiment_module("experiments/834_e3_probe/probe_runner.py", "probe834_844")
eval835 = _load_experiment_module("experiments/835_stage4_eval/eval_vector.py", "eval835_844")

# Reused 834 machinery (do not re-derive): --config-json → TrainingConfig with setattr
# extras; model+trainer build (vector assert + init_train_state); the trainer_metadata.json
# cross-check that fails loud BEFORE a silent flax no-op restore; the zero-logit dithered-
# book walk (make_walk_step + run_walk) that collects seed-fixed, training-reachable books.
build_training_config = probe834.build_training_config
build_model = probe834.build_model
make_walk_step = probe834.make_walk_step
run_walk = probe834.run_walk


def check_ckpt_metadata(ckpt_path, model):
    """834's fail-loud metadata guard, re-branded so a TASK_844 failure is attributed
    to this probe in factory logs."""
    try:
        probe834.check_ckpt_metadata(ckpt_path, model)
    except SystemExit as exc:
        raise SystemExit(f"TASK_844 (via the imported 834 guard): {exc}") from None


# =============================================================================
# fixture-smoke config (mirror of 834/836 smoke_training_config)
# =============================================================================

def smoke_training_config(seed):
    cfg = TrainingConfig(
        name="844_smoke", seed=seed, parallel_envs=1, steps=96, epochs=1,
        sims=8, batch_size=32, model_size="small",
        fee=16, slippage=5,                      # battery primary = fee16/slip5
        db_path=FIXTURE_DB,
        start_date="2025-09-14", end_date="2025-12-11",
        reward_attributable=True,                # battery: --reward-attributable
        bar_minutes=60,                          # fixture 1-min table spans ~90 days
        vector_action=True,
        vector_hold_bias=BATTERY_HOLD_BIAS,      # the battery operator
        vector_dither_scale=1.0,
        vector_proposal_scheme="dithered-book",
    )
    cfg.use_jax_env = True
    cfg.reward_hybrid = True                     # trainer-side only (834 precedent)
    # Small search operator for the CPU smoke (plumbing proof only; the checkpoint run
    # takes the FULL battery operator from --config-json). Self-described in output.
    cfg.vector_k = 8
    cfg.discount = 0.997
    cfg.gumbel_scale = 3.0
    cfg.value_scale = 1.0
    cfg.value_shift = 0.0
    return cfg


# =============================================================================
# panel — single deterministic env over the FULL timeline (836/835 fixed-start pattern)
# =============================================================================

def build_panel(cfg, n_assets, seed, db_override, max_decisions):
    start_date, end_date = cfg.start_date, cfg.end_date
    specific_symbols = None
    if getattr(cfg, "periods_file", None):       # 834/835 build_env branch, verbatim
        with open(cfg.periods_file) as f:
            first_period = json.load(f)["periods"][0]
        start_date = first_period["start_date"]
        end_date = first_period["end_date"]
        specific_symbols = first_period.get("assets")
        print(f"[env] periods file: {cfg.periods_file} "
              f"({start_date[:10]} -> {end_date[:10]})")
    elif getattr(cfg, "assets_file", None):      # train_jax_fast.py:233-236 parity
        with open(cfg.assets_file) as f:
            specific_symbols = json.load(f)
        print(f"[env] assets file: {cfg.assets_file} ({len(specific_symbols)} symbols)")
    db_path = db_override or cfg.db_path
    states0, prices, env_cfg = create_jax_env_historical(
        db_path=db_path, start_date=start_date, end_date=end_date,
        n_envs=1, n_assets=n_assets, specific_symbols=specific_symbols,
        fee_bps=float(cfg.fee), slippage_bps=float(cfg.slippage),
        max_episode_steps=cfg.steps,             # inert: overridden below (835)
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
        vector_budget_step=getattr(cfg, "vector_budget_step", False),
        vector_budget_pct=getattr(cfg, "vector_budget_pct", 0.99),
        vector_weight_rebalance=getattr(cfg, "vector_weight_rebalance", False),
        vector_weight_unit=getattr(cfg, "vector_weight_unit", -1.0),
        vector_weight_max=getattr(cfg, "vector_weight_max", 4.0),
        vector_rebalance_deadband=getattr(cfg, "vector_rebalance_deadband", -1.0),
        rng_key=jax.random.PRNGKey(seed),
    )
    row_ts, loaded_syms = eval835.load_row_timestamps(
        db_path, start_date, end_date, specific_symbols, n_assets,
        getattr(cfg, "bar_minutes", 1))
    T = int(prices.shape[0])
    if len(row_ts) != T:
        raise SystemExit(
            f"TASK_844: panel-consistency violation — reconstructed timestamp axis has "
            f"{len(row_ts)} rows but create_jax_env_historical produced {T}")
    warm = int(env_cfg.window_size)
    n_max = T - 1 - warm
    n_decisions = n_max if max_decisions is None else min(n_max, max_decisions)
    if n_decisions < 2:
        raise SystemExit(f"TASK_844: panel too short — T={T}, warm={warm}, "
                         f"n_decisions={n_decisions}")
    # Deterministic full-timeline episode: fixed start at the first obs-valid row (835).
    state_init = states0.replace(
        episode_start_idx=jnp.full_like(states0.episode_start_idx, warm),
        current_step=jnp.zeros_like(states0.current_step),
        max_episode_steps=jnp.full_like(states0.max_episode_steps, n_decisions + 2),
        initial_portfolio_value=jnp.full_like(states0.initial_portfolio_value,
                                              env_cfg.initial_cash),
        prev_portfolio_value=jnp.full_like(states0.prev_portfolio_value,
                                            env_cfg.initial_cash),
    )
    assert int(state_init.episode_start_idx[0]) == int(env_cfg.window_size), \
        ("TASK_844: walk must start at the first obs-valid row "
         "(episode_start_idx == window_size); obs window would leak/underrun")
    n_loaded = len(loaded_syms)
    col_symbols = [loaded_syms[i % n_loaded] if i < n_loaded
                   else f"{loaded_syms[i % n_loaded]}#pad{i}" for i in range(n_assets)]
    # Repo-relative db for a self-describing committed artifact (no embedded $HOME path;
    # the 835 _display_path rationale); absolute only if the db lives outside the repo.
    db_rel = os.path.relpath(os.path.abspath(db_path), _REPO)
    db_display = os.path.abspath(db_path) if db_rel.startswith("..") else db_rel
    info = {
        "db": db_display, "start": start_date, "end": end_date, "n_rows": T,
        "n_assets": int(env_cfg.n_assets), "warm": warm, "n_decisions": n_decisions,
        "n_loaded_symbols": n_loaded, "padded": n_loaded < n_assets,
        "fee_bps": float(env_cfg.fee_bps), "slippage_bps": float(env_cfg.slippage_bps),
    }
    return state_init, prices, env_cfg, row_ts, col_symbols, info


def build_searcher(cfg, env_cfg):
    """Sampling/decode machinery only (831/834 pattern; model_apply_fn=None)."""
    return SampledMuZeroVectorSearch(
        model_apply_fn=None,
        n_assets=env_cfg.n_assets, n_actions_per_asset=N_ACTIONS_PER_ASSET,
        num_samples=1, hold_bias=float(getattr(cfg, "vector_hold_bias", 0.0)),
        dither_scale=float(getattr(cfg, "vector_dither_scale", 1.0)),
        proposal_scheme=str(getattr(cfg, "vector_proposal_scheme", "dithered-book")),
        value_scale=float(getattr(cfg, "value_scale", 1.0)),
        value_shift=float(getattr(cfg, "value_shift", 0.0)))


# =============================================================================
# PROBE 1 — mirror-book value
# =============================================================================

def _mean_sem(x):
    x = np.asarray(x, dtype=np.float64)
    n = int(len(x))
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "sem": None, "t": None,
                "frac_negative": None}
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=1)) if n > 1 else 0.0
    sem = std / np.sqrt(n) if n > 1 else 0.0
    t = (mean / sem) if sem > EPS else None
    return {"n": n, "mean": mean, "std": std, "sem": sem,
            "t": (float(t) if t is not None else None),
            "frac_negative": float(np.mean(x < 0.0))}


def _null_init_seeds(n_null, seed_avoid):
    """Fixed, checkpoint-independent fresh-init seeds for the matched null (MAJOR-1),
    skipping the checkpoint/walk seed so the null never collides with the probed model."""
    seeds, s = [], NULL_INIT_SEED_BASE
    while len(seeds) < n_null:
        if s != seed_avoid:
            seeds.append(int(s))
        s += 1
    return seeds


def probe_mirror_book_value(model, params, searcher, panel, cfg, seed, args):
    """Δ = V(open support LONG) − V(open support SHORT) from the same market latent,
    over N books collected by the seed-fixed zero-logit dithered-book walk (834)."""
    state_init, prices, env_cfg, row_ts, col_symbols, info = panel
    A = env_cfg.n_assets
    max_positions = int(getattr(cfg, "max_positions", 16))
    min_cash_threshold = float(getattr(cfg, "min_cash_threshold", 100.0))

    # Collect training-reachable books via 834's zero-logit dithered-book walk.
    walk_step = make_walk_step(searcher, prices, env_cfg, 1,
                               max_positions, min_cash_threshold)
    batched_states, obs_all, masks_all, valid, first_done = run_walk(
        walk_step, state_init, seed, args.mirror_burn_in, args.mirror_states)
    keep = np.nonzero(valid)[0]

    pos_active = np.asarray(batched_states.pos_active)[keep].astype(bool)   # (M, A)
    pos_side = np.asarray(batched_states.pos_side)[keep].astype(np.int32)   # (M, A)
    obs = np.asarray(obs_all)[keep]

    # Keep only states that actually HOLD a book (support >= 1); a flat state would
    # force B == −B == all-HOLD and a trivial Δ == 0, diluting the signal.
    support = pos_active                                       # active ⇒ side in {1,2}
    has_book = support.sum(axis=1) >= 1
    idx = np.nonzero(has_book)[0]
    n_book = int(len(idx))
    if n_book == 0:
        return {
            "n_states_walked": int(len(keep)), "n_states_with_book": 0,
            "delta_stats": _mean_sem(np.empty(0)), "note": (
                "no book-bearing states collected on the walk — cannot mirror; widen "
                "--mirror-states / --mirror-burn-in or check the walk operator"),
            "walk": _walk_meta(seed, args, first_done, max_positions, min_cash_threshold),
        }
    support = support[idx]
    obs = obs[idx]

    # Build book B (support LONG) and its EXACT sign-mirror −B (support SHORT).
    a_B = np.where(support, BUY, HOLD).astype(np.int32)        # (n_book, A)
    a_mirror = np.where(support, SELL, HOLD).astype(np.int32)

    # ---- alignment audit: the mirror MUST be an exact long↔short flip -----------
    assert a_B.shape == a_mirror.shape == support.shape
    off = ~support
    assert np.array_equal(a_B[off], np.full(int(off.sum()), HOLD)), \
        "TASK_844: mirror leak — B has a non-HOLD action off the book support"
    assert np.array_equal(a_mirror[off], np.full(int(off.sum()), HOLD)), \
        "TASK_844: mirror leak — −B has a non-HOLD action off the book support"
    assert np.all(a_B[support] == BUY) and np.all(a_mirror[support] == SELL), \
        "TASK_844: mirror leak — support slots are not a clean BUY vs SELL flip"
    # Same support set, same count ⇒ same magnitudes/cash (symmetric encoding, one latent).
    assert np.array_equal((a_B == BUY).sum(axis=1), (a_mirror == SELL).sum(axis=1)), \
        "TASK_844: mirror leak — long/short leg counts differ (not a true sign-flip)"
    assert np.array_equal((a_B != HOLD), (a_mirror != HOLD)), \
        "TASK_844: mirror leak — active asset set differs between B and −B"

    @jax.jit
    def _values(p, obs_j, aB_j, aM_j):
        init = model.apply({"params": p}, obs_j, mode="initial_inference")
        z = init["latent_state"]
        vB = searcher._categorical_to_scalar(
            model.apply({"params": p}, z, aB_j, mode="recurrent_inference_vector")["value"])
        vM = searcher._categorical_to_scalar(
            model.apply({"params": p}, z, aM_j, mode="recurrent_inference_vector")["value"])
        return vB, vM

    obs_j = jnp.asarray(obs)
    aB_j = jnp.asarray(a_B)
    aM_j = jnp.asarray(a_mirror)
    vB, vM = _values(params, obs_j, aB_j, aM_j)
    jax.block_until_ready(vB)
    vB = np.asarray(vB, dtype=np.float64)
    vM = np.asarray(vM, dtype=np.float64)
    delta = vB - vM
    ckpt_offset = float(delta.mean())
    support_sizes = support.sum(axis=1).astype(np.int64)

    # ---- MAJOR-1 CALIBRATION: matched fresh-init null over the SAME book states ----------
    # vec_act_proj's BUY/SELL columns are independently initialized, so a fresh untrained
    # model already produces a per-checkpoint mirror-Δ offset (arbitrary architectural
    # zero-point). Build a few fresh-init models of the SAME architecture (run's own cfg) and
    # measure their mirror-Δ offsets on the IDENTICAL states/latents. The null distribution is
    # OVER INITS; the inference unit is the per-checkpoint offset (its mean-Δ). The value-locus
    # verdict gates on the null-subtracted, init-standardized z, NOT the raw within-state t.
    n_null = int(getattr(args, "mirror_null_inits", 0) or 0)
    null_seeds = _null_init_seeds(n_null, seed_avoid=args.seed) if n_null > 0 else []
    null_offsets = []
    for ns in null_seeds:
        _m_null, trainer_null = build_model(cfg, A, init_seed=ns)
        vBn, vMn = _values(trainer_null.state.params, obs_j, aB_j, aM_j)
        jax.block_until_ready(vBn)
        null_offsets.append(float(np.mean(np.asarray(vBn, dtype=np.float64)
                                          - np.asarray(vMn, dtype=np.float64))))
    calibrated = _calibrate_offset(ckpt_offset, np.asarray(null_offsets, dtype=np.float64),
                                   null_seeds, int(seed))

    return {
        "definition": ("Δ = V(B) − V(−B); B = open the walk-state book's support LONG "
                       "(BUY on support, HOLD elsewhere); −B = its EXACT element-wise "
                       "long↔short mirror (SELL on support). Value = one "
                       "recurrent_inference_vector step from the state's own market "
                       "latent, canon _categorical_to_scalar decode. B canonicalized "
                       "LONG so mean Δ < 0 ⇒ short-preference. RAW delta_stats (within-"
                       "state t) is UNCALIBRATED: vec_act_proj (muzero_model.py:306-314) "
                       "has independently-initialized BUY vs SELL columns, so even a fresh "
                       "untrained model yields a per-checkpoint offset with |t| up to ~4 "
                       "(arbitrary architectural zero-point). The VALUE-locus read gates on "
                       "`calibrated` (null-subtracted vs matched fresh inits), NOT raw t."),
        "n_states_walked": int(len(keep)), "n_states_with_book": n_book,
        "mean_support_size": float(support_sizes.mean()),
        "delta_stats": _mean_sem(delta),
        "calibrated": calibrated,
        "value_B_mean": float(vB.mean()), "value_mirror_mean": float(vM.mean()),
        "mirror_audit": ("EXACT: off-support both HOLD; on-support B=BUY vs −B=SELL; "
                         "per-state long-leg count == short-leg count; same active set"),
        "walk": _walk_meta(seed, args, first_done, max_positions, min_cash_threshold),
    }


def _calibrate_offset(ckpt_offset, null_offsets, null_seeds, boot_seed):
    """Null-subtracted, init-standardized calibration of the per-checkpoint mirror-Δ offset
    (MAJOR-1). Null spread is OVER FRESH INITS (the architectural zero-point confound), not
    over states. z = (ckpt_offset − null_mean)/null_std_over_inits; the null is bootstrapped
    over inits for a CI on the calibrated delta. Value-locus gates on z (see interpret)."""
    n_null = int(len(null_offsets))
    if n_null == 0:
        return {
            "definition": ("null-subtracted mirror-Δ vs a matched fresh-init null "
                           "(same architecture). DISABLED (n_null=0) — no calibration."),
            "ckpt_offset": ckpt_offset, "n_null": 0,
            "null_init_seeds": [], "null_offsets": [], "null_mean": None,
            "null_std": None, "calibrated_delta": None, "z": None,
            "bootstrap_ci95_delta": None,
        }
    null_mean = float(null_offsets.mean())
    null_std = float(null_offsets.std(ddof=1)) if n_null > 1 else 0.0
    calibrated_delta = ckpt_offset - null_mean
    z = (calibrated_delta / null_std) if null_std > EPS else None
    ci = None
    if n_null > 1:
        rng = np.random.default_rng(boot_seed)
        boot = np.array([ckpt_offset - rng.choice(null_offsets, size=n_null, replace=True).mean()
                         for _ in range(N_NULL_BOOT)], dtype=np.float64)
        ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    return {
        "definition": ("z = (Δ_ckpt_offset − null_mean) / null_std, where null_mean/null_std "
                       "are the MEAN and SPREAD of the mirror-Δ offset ACROSS FRESH INITS of "
                       "the SAME architecture, measured on the IDENTICAL book states/latents. "
                       "This removes the arbitrary vec_act_proj zero-point that the raw "
                       "within-state t cannot separate from a learned short-preference. The "
                       "null is bootstrapped OVER INITS for the CI. Value-locus ⇔ z < "
                       f"{VALUE_Z_THRESHOLD} (needs n_null ≥ {MIN_NULL})."),
        "ckpt_offset": ckpt_offset, "n_null": n_null,
        "null_init_seeds": list(null_seeds), "null_offsets": null_offsets.tolist(),
        "null_mean": null_mean, "null_std": null_std,
        "calibrated_delta": calibrated_delta, "z": (float(z) if z is not None else None),
        "bootstrap_ci95_delta": ci,
    }


def _walk_meta(seed, args, first_done, max_positions, min_cash_threshold):
    return {
        "action_rule": ("zero-logit dithered-book proposal, seed-fixed, legality-masked "
                        "per state (834 precedent) — checkpoint-independent; single env, "
                        "fixed start at the first obs-valid row"),
        "seed": seed, "burn_in": args.mirror_burn_in, "collect_steps": args.mirror_states,
        "first_done": int(first_done), "mask_max_positions": max_positions,
        "mask_min_cash_threshold": min_cash_threshold,
    }


# =============================================================================
# PROBE 2 — flat-book policy marginals
# =============================================================================

def probe_flat_book_policy(model, params, panel, cfg, args):
    """Raw per-asset policy softmax at M flat-book (all-cash) states across the panel.
    sell-mass = p(SELL)+p(LimitSell); buy-mass = p(BUY)+p(LimitBuy). No RNG (deterministic)."""
    state_init, prices, env_cfg, row_ts, col_symbols, info = panel
    A = env_cfg.n_assets
    warm = info["warm"]
    T = info["n_rows"]
    max_positions = int(getattr(cfg, "max_positions", 16))
    min_cash_threshold = float(getattr(cfg, "min_cash_threshold", 100.0))
    enc_cfg = EncoderConfig(n_assets=A, window_size=env_cfg.window_size, n_channels=27)

    # M flat-book states = the reset (all-cash) book relocated to M spread decision rows.
    # obs window is [r-window, r-1]; require r >= warm (no underrun) and r <= T-1.
    M = min(args.flat_states, T - 1 - warm)
    rows = np.unique(np.linspace(warm, T - 2, M).astype(np.int64))
    n_flat = int(len(rows))

    def _flat_state(r):
        return state_init.replace(
            episode_start_idx=jnp.full_like(state_init.episode_start_idx, int(r)),
            current_step=jnp.zeros_like(state_init.current_step))

    flat_states = jax.tree_util.tree_map(
        lambda *xs: jnp.concatenate(xs, axis=0), *[_flat_state(r) for r in rows])
    # Flat-book invariant: these states carry NO position (all cash) by construction.
    assert not bool(np.asarray(flat_states.pos_active).any()), \
        "TASK_844: flat-book states are not flat (a position leaked into the reset book)"

    price_idx = flat_states.episode_start_idx + flat_states.current_step   # (M,)
    window_starts = price_idx - enc_cfg.window_size
    obs = jit_batch_encode_simple(prices, window_starts, enc_cfg)
    current_prices = prices[price_idx]
    mask = jit_batch_compute_action_mask(
        flat_states, current_prices, env_cfg, max_positions, min_cash_threshold)

    @jax.jit
    def _marginals(p, obs_j, mask_j):
        init = model.apply({"params": p}, obs_j, mode="initial_inference")
        logits = init["policy_logits"].reshape(-1, A, N_ACTIONS_PER_ASSET)   # (M,A,5)
        raw = jax.nn.softmax(logits, axis=-1)
        mask_pa = mask_j.reshape(-1, A, N_ACTIONS_PER_ASSET).astype(bool)
        masked = jax.nn.softmax(jnp.where(mask_pa, logits, -1e9), axis=-1)
        return raw, masked

    raw, masked = _marginals(params, obs, mask)
    jax.block_until_ready(raw)
    raw = np.asarray(raw, dtype=np.float64)          # (M, A, 5)
    masked = np.asarray(masked, dtype=np.float64)

    def _summ(pr):
        sell = pr[..., SELL] + pr[..., LIMIT_SELL]
        buy = pr[..., BUY] + pr[..., LIMIT_BUY]
        hold = pr[..., HOLD]
        sm, bm, hm = float(sell.mean()), float(buy.mean()), float(hold.mean())
        return {"sell_mass": sm, "buy_mass": bm, "hold_mass": hm,
                "sell_over_buy": (sm / bm if bm > EPS else None)}

    return {
        "definition": ("raw = per-asset softmax of the factored policy head at flat-book "
                       "(all-cash) states = the learned PRIOR (primary); masked = same "
                       "under the legality mask (diagnostic secondary). sell-mass = "
                       "p(SELL)+p(LimitSell); buy-mass = p(BUY)+p(LimitBuy)."),
        "n_flat_states": n_flat, "n_assets": int(A),
        "under_powered": bool(n_flat < MIN_N),
        "raw_prior": _summ(raw), "masked": _summ(masked),
        "rows_first_last": [int(rows[0]), int(rows[-1])],
    }


# =============================================================================
# PROBE 3 — rollout leg decomposition
# =============================================================================

def _op_from_cfg(cfg):
    return {
        "vector_k": int(getattr(cfg, "vector_k", 16)),
        "sims": int(getattr(cfg, "sims", 32)),
        "discount": float(getattr(cfg, "discount", 0.997)),
        "gumbel_scale": float(getattr(cfg, "gumbel_scale", 3.0)),
        "value_scale": float(getattr(cfg, "value_scale", 1.0)),
        "value_shift": float(getattr(cfg, "value_shift", 0.0)),
        "hold_bias": float(getattr(cfg, "vector_hold_bias", 0.0)),
        "dither_scale": float(getattr(cfg, "vector_dither_scale", 1.0)),
        "proposal_scheme": str(getattr(cfg, "vector_proposal_scheme", "dithered-book")),
        "per_node_resampling": bool(getattr(cfg, "vector_per_node_resample", False)),
    }


def probe_rollout_legs(model, params, panel, cfg, seed, args):
    """835 deterministic rollout (real vector search, visit-argmax) instrumented for
    per-leg realized PnL by side. Alignment-clean: each asset closes at most once per step
    (actions never flip a side in one step); realized on close uses PRE-step avg entry ×
    the DECISION-row exit price (the row the action executes on) — no double-count."""
    state_init, prices, env_cfg, row_ts, col_symbols, info = panel
    A = env_cfg.n_assets
    op = _op_from_cfg(cfg)
    n_decisions = info["n_decisions"]
    max_positions = int(getattr(cfg, "max_positions", 16))
    min_cash_threshold = float(getattr(cfg, "min_cash_threshold", 100.0))
    enc_cfg = EncoderConfig(n_assets=A, window_size=env_cfg.window_size, n_channels=27)

    search = create_sampled_vector_search(
        model, n_assets=A, n_actions_per_asset=N_ACTIONS_PER_ASSET,
        num_samples=op["vector_k"], num_simulations=op["sims"],
        max_num_considered_actions=min(op["vector_k"], 16), discount=op["discount"],
        gumbel_scale=op["gumbel_scale"], value_scale=op["value_scale"],
        value_shift=op["value_shift"], hold_bias=op["hold_bias"],
        dither_scale=op["dither_scale"], proposal_scheme=op["proposal_scheme"],
        per_node_resampling=op["per_node_resampling"])
    book = jnp.ones((1, A), dtype=jnp.int32)                  # v1 book = HOLD (DEC-3)

    # side-classified accumulators.
    # *_all = ALL closes (pre-step active → post-step inactive) — a SUPERSET of the env's
    #         reward_attributable term (it also sweeps in limit-fill / TP-SL / margin closes).
    # *_action = the ACTION-ATTRIBUTED subset: closes the market SELL/BUY on that asset
    #         directly commanded (SELL closes a long, BUY closes a short) — a clean subset of
    #         the env reward_attributable term (jax_trading_env.py:1401 snapshots pre-ACTION,
    #         post limit/TP-SL/margin). Conservative: it excludes indirect post-action margin
    #         liquidations the env would also credit, so it NEVER over-credits.
    realized_log = {"long": 0.0, "short": 0.0}               # all-close superset (log-PnL)
    realized_dollar = {"long": 0.0, "short": 0.0}
    realized_log_action = {"long": 0.0, "short": 0.0}        # action-attributed subset
    close_count = {"long": 0, "short": 0}
    close_count_action = {"long": 0, "short": 0}
    open_count = {"long": 0, "short": 0}

    state = state_init
    key = jax.random.PRNGKey(seed)
    pvs = [float(env_cfg.initial_cash)]
    n_exec = 0
    terminated_at = -1
    for t in range(n_decisions):
        price_idx = int(state.episode_start_idx[0]) + int(state.current_step[0])
        price_dec = np.asarray(prices[price_idx])            # (A,) decision-row exec mark
        obs = jit_batch_encode_simple(
            prices, state.episode_start_idx + state.current_step - enc_cfg.window_size,
            enc_cfg)
        mask = jit_batch_compute_action_mask(
            state, prices[state.episode_start_idx + state.current_step],
            env_cfg, max_positions, min_cash_threshold)
        # pre-step book (before limit/tp_sl/action) — the leg's live entry/size
        pre_active = np.asarray(state.pos_active)[0].astype(bool)
        pre_side = np.asarray(state.pos_side)[0].astype(np.int32)
        pre_entry = np.asarray(state.pos_entry)[0].astype(np.float64)
        pre_size = np.asarray(state.pos_size)[0].astype(np.float64)

        key, step_rng = jax.random.split(key)
        action_vec, _s, _v, _st = search(params, step_rng,
                                         obs, mask, book, True)  # deterministic=True
        act = np.asarray(action_vec)[0].astype(np.int32)     # (A,) per-asset action code
        state, _, _, done = jit_batch_step_vector(state, action_vec, prices, env_cfg)

        post_active = np.asarray(state.pos_active)[0].astype(bool)
        post_side = np.asarray(state.pos_side)[0].astype(np.int32)

        # closes: active pre → inactive post (one per asset; actions don't flip in a step).
        # This is the ALL-CLOSE SUPERSET (limit-fill / TP-SL / margin closes included). The
        # ACTION-ATTRIBUTED subset = the market SELL/BUY on that asset directly closed it
        # (SELL(0) closes a long; BUY(2) closes a short). An asset pre-closed by
        # limit/TP-SL/margin is inactive at action time, so a SELL/BUY there OPENS a new
        # position (post-active) ⇒ not in `closed` ⇒ pre_side here is the pre-ACTION side.
        closed = pre_active & (~post_active)
        for i in np.nonzero(closed)[0]:
            side = int(pre_side[i])
            entry = max(float(pre_entry[i]), 1e-8)
            exit_px = max(float(price_dec[i]), 1e-8)
            long_log = float(np.log(exit_px / entry))
            a_i = int(act[i])
            action_caused = (a_i == SELL and side == 1) or (a_i == BUY and side == 2)
            if side == 1:                                    # long
                realized_log["long"] += long_log
                realized_dollar["long"] += float(pre_size[i]) * (exit_px - entry)
                close_count["long"] += 1
                if action_caused:
                    realized_log_action["long"] += long_log
                    close_count_action["long"] += 1
            elif side == 2:                                  # short (mirror, env:1255)
                realized_log["short"] += -long_log
                realized_dollar["short"] += float(pre_size[i]) * (entry - exit_px)
                close_count["short"] += 1
                if action_caused:
                    realized_log_action["short"] += -long_log
                    close_count_action["short"] += 1
        # opens: inactive pre → active post
        opened = (~pre_active) & post_active
        for i in np.nonzero(opened)[0]:
            if int(post_side[i]) == 1:
                open_count["long"] += 1
            elif int(post_side[i]) == 2:
                open_count["short"] += 1

        cur_after = prices[state.episode_start_idx + state.current_step - 1]
        pv = float(np.asarray(jax.vmap(compute_portfolio_value)(state, cur_after))[0])
        pvs.append(pv)
        n_exec += 1
        if bool(np.asarray(done)[0]):
            terminated_at = t
            break

    # still-open legs at rollout end → mark-to-market unrealized by side
    final_active = np.asarray(state.pos_active)[0].astype(bool)
    final_side = np.asarray(state.pos_side)[0].astype(np.int32)
    final_entry = np.asarray(state.pos_entry)[0].astype(np.float64)
    final_size = np.asarray(state.pos_size)[0].astype(np.float64)
    final_price = np.asarray(
        prices[int(state.episode_start_idx[0]) + int(state.current_step[0]) - 1])
    unreal_dollar = {"long": 0.0, "short": 0.0}
    open_now = {"long": 0, "short": 0}
    for i in np.nonzero(final_active)[0]:
        side = int(final_side[i])
        entry = max(float(final_entry[i]), 1e-8)
        px = max(float(final_price[i]), 1e-8)
        if side == 1:
            unreal_dollar["long"] += float(final_size[i]) * (px - entry)
            open_now["long"] += 1
        elif side == 2:
            unreal_dollar["short"] += float(final_size[i]) * (entry - px)
            open_now["short"] += 1

    return {
        "definition": ("835 deterministic vector rollout (visit-argmax); per-leg realized "
                       "PnL by side, long=log(exit/entry), short=−log(exit/entry), "
                       "exit=decision-row price, entry=pre-close avg entry. "
                       "realized_log_pnl_by_side = ALL-CLOSE realized log-PnL = a SUPERSET "
                       "of the env reward_attributable term (jax_trading_env.py:1249-1259): "
                       "it also sweeps in limit-fill / TP-SL / margin closes, whereas the "
                       "env snapshots prev_active_attr POST limit/TP-SL/margin, pre-action "
                       "(env:1401/1191) so it credits ACTION-caused closes only. "
                       "action_attributed_log_pnl_by_side = the closes the market SELL/BUY "
                       "on that asset directly commanded (a clean, conservative subset of "
                       "the env term — excludes indirect post-action margin liquidations, so "
                       "it never over-credits). credit-misassignment gates on the "
                       "ACTION-ATTRIBUTED short term. realized_dollar / unrealized_dollar are "
                       "the $ views of the all-close set."),
        "operator": op, "n_decisions": n_decisions, "n_exec": n_exec,
        "terminated_early_at": terminated_at, "final_pv": pvs[-1],
        "close_count": close_count, "close_count_action_attributed": close_count_action,
        "open_count": open_count,
        "realized_log_pnl_by_side": realized_log,
        "action_attributed_log_pnl_by_side": realized_log_action,
        "realized_dollar_by_side": realized_dollar,
        "open_at_end": open_now, "unrealized_dollar_by_side": unreal_dollar,
    }


# =============================================================================
# interpretation (checkpoint mode only; NEVER the untrained smoke)
# =============================================================================

def interpret(mirror, flat, legs, smoke):
    block = {"rule_verbatim": INTERPRETATION_VERBATIM, "applied": not smoke}
    if smoke:
        block["reading"] = ("FIXTURE-SMOKE (untrained fresh-init model) — plumbing proof "
                            "only; interpretation NOT applied. Real reading = scheduled "
                            "run on the p1c/p1b snapshots.")
        return block

    ds = mirror.get("delta_stats", {})
    n = ds.get("n", 0)
    cal = mirror.get("calibrated", {})
    z = cal.get("z")
    n_null = cal.get("n_null", 0)
    # MAJOR-1: gate on the CALIBRATED (null-subtracted, init-standardized) statistic — the raw
    # within-state t fires on unbiased fresh inits and cannot see the architectural zero-point.
    value_locus = bool(z is not None and z <= VALUE_Z_THRESHOLD
                       and n >= MIN_N and n_null >= MIN_NULL)
    value_note = (
        f"calibrated z={z} (Δ_calib={cal.get('calibrated_delta')}, ckpt_offset="
        f"{cal.get('ckpt_offset')}, null_mean={cal.get('null_mean')}, "
        f"null_std={cal.get('null_std')}, n_null={n_null}); raw within-state t="
        f"{ds.get('t')} (UNCALIBRATED, n={n})" if n else "no book-bearing states")

    ratio = flat.get("raw_prior", {}).get("sell_over_buy")
    policy_locus = bool(ratio is not None and ratio > POLICY_RATIO_THRESHOLD
                        and not flat.get("under_powered", True))

    # MAJOR-2: credit-misassignment gates on ACTION-ATTRIBUTED closes only (the subset the
    # policy's market SELL/BUY directly commanded), NOT the all-close superset — so it can
    # never fire on realized PnL the training objective never credited.
    aa = legs["action_attributed_log_pnl_by_side"]
    sl = aa["short"]
    ll = aa["long"]
    sl_all = legs["realized_log_pnl_by_side"]["short"]
    ll_all = legs["realized_log_pnl_by_side"]["long"]
    n_short_open = legs["open_count"]["short"]
    n_long_open = legs["open_count"]["long"]
    credit_misassign = bool(sl < 0.0 and n_short_open > 0)
    short_heavy_opens = bool(n_short_open > n_long_open)

    loci = []
    if value_locus:
        loci.append("VALUE")
    if policy_locus:
        loci.append("POLICY-prior")
    if credit_misassign:
        loci.append("REWARD/credit-misassignment")
    reading = (
        f"value-locus: {value_locus} ({value_note}); "
        f"policy-prior-locus: {policy_locus} (sell/buy raw ratio={ratio}); "
        f"reward-asymmetry (ACTION-ATTRIBUTED closes): short realized-log={sl:.4f} "
        f"(opens {n_short_open}) vs long realized-log={ll:.4f} (opens {n_long_open}) "
        f"[all-close superset: S={sl_all:.4f}, L={ll_all:.4f}]; "
        f"short-heavy opens: {short_heavy_opens}. "
        + ("candidate loci this checkpoint: " + ", ".join(loci) if loci
           else "no single locus tripped its threshold this checkpoint") +
        ". DIAGNOSTIC — localizes only; gates/fixes nothing.")
    block.update({
        "value_locus": value_locus, "policy_prior_locus": policy_locus,
        "reward_credit_misassignment": credit_misassign,
        "short_heavy_opens": short_heavy_opens, "candidate_loci": loci,
        "reading": reading,
    })
    return block


# =============================================================================
# result assembly + LEDGER row
# =============================================================================

def _fmt(x):
    return "n/a" if x is None else f"{x:+.4f}"


def make_ledger_row(r):
    mv = r["mirror_book_value"]["delta_stats"]
    cal = r["mirror_book_value"].get("calibrated", {})
    fp = r["flat_book_policy"]["raw_prior"]
    lg = r["rollout_legs"]
    aa = lg["action_attributed_log_pnl_by_side"]
    itp = r["interpretation"]
    smoke_note = (" FIXTURE-SMOKE — plumbing proof, untrained model, interpretation not "
                  "applied; real numbers come from the scheduled run."
                  if r["mode"] == "fixture-smoke" else f" {itp['reading']}")
    return (
        f"| 844/short-bias | {date.today().isoformat()} | postmortem | "
        f"Where is the ~10-short standing book carried (value / policy prior / realized "
        f"reward)? {r['checkpoint']} | "
        f"mirror-Δ calibrated {_fmt(cal.get('calibrated_delta'))} "
        f"(z {_fmt(cal.get('z'))}, n_null={cal.get('n_null')}; raw mean "
        f"{_fmt(mv.get('mean'))}, raw-t {_fmt(mv.get('t'))} UNCAL, n={mv.get('n')}); "
        f"flat-book sell/buy {_fmt(fp.get('sell_over_buy'))} "
        f"(sell {_fmt(fp.get('sell_mass'))} vs buy {_fmt(fp.get('buy_mass'))}); "
        f"legs action-attributed realized-log S {_fmt(aa['short'])} "
        f"(opens {lg['open_count']['short']}) vs L {_fmt(aa['long'])} "
        f"(opens {lg['open_count']['long']}) [all-close S "
        f"{_fmt(lg['realized_log_pnl_by_side']['short'])} / L "
        f"{_fmt(lg['realized_log_pnl_by_side']['long'])}].{smoke_note} | "
        f"n/a (diagnostic instrument — no corridor claim) | {r['out_pointer']} |")


def assemble(model, params, cfg, panel, args, checkpoint_label, smoke,
             searcher, restored_step_count):
    state_init, prices, env_cfg, row_ts, col_symbols, info = panel
    hold_bias = float(getattr(cfg, "vector_hold_bias", 0.0))
    if hold_bias == 0.0:
        print("[probe] WARNING: vector_hold_bias=0.0 — the 828 Stage-3 battery operator "
              f"is hold-bias {BATTERY_HOLD_BIAS} (the CLI default 0.0 is a documented "
              "trap; check the config JSON).")

    t0 = time.time()
    mirror = probe_mirror_book_value(model, params, searcher, panel, cfg, args.seed, args)
    print(f"[probe1/mirror] n_book={mirror['n_states_with_book']} "
          f"Δ mean={_fmt(mirror['delta_stats'].get('mean'))} "
          f"t={_fmt(mirror['delta_stats'].get('t'))} ({time.time()-t0:.1f}s)")

    t1 = time.time()
    flat = probe_flat_book_policy(model, params, panel, cfg, args)
    print(f"[probe2/policy] n_flat={flat['n_flat_states']} "
          f"sell/buy(raw)={_fmt(flat['raw_prior'].get('sell_over_buy'))} "
          f"(sell={_fmt(flat['raw_prior']['sell_mass'])} "
          f"buy={_fmt(flat['raw_prior']['buy_mass'])}) ({time.time()-t1:.1f}s)")

    t2 = time.time()
    legs = probe_rollout_legs(model, params, panel, cfg, args.seed, args)
    print(f"[probe3/legs] exec={legs['n_exec']}/{legs['n_decisions']} "
          f"opens L{legs['open_count']['long']}/S{legs['open_count']['short']} "
          f"realized-log L{_fmt(legs['realized_log_pnl_by_side']['long'])}/"
          f"S{_fmt(legs['realized_log_pnl_by_side']['short'])} ({time.time()-t2:.1f}s)")

    interp = interpret(mirror, flat, legs, smoke)
    result = {
        "task": "TASK_844 short-bias postmortem probe",
        "mode": "fixture-smoke" if smoke else "checkpoint",
        "checkpoint": checkpoint_label,
        "db": info["db"], "window": [info["start"], info["end"]],
        "panel": {k: info[k] for k in ("n_rows", "n_assets", "warm", "n_decisions",
                                       "n_loaded_symbols", "padded")},
        "seed": args.seed,
        "restored_step_count": restored_step_count,
        "operator": {
            "hold_bias": hold_bias,
            "dither_scale": float(getattr(cfg, "vector_dither_scale", 1.0)),
            "proposal_scheme": str(getattr(cfg, "vector_proposal_scheme", "dithered-book")),
            "rollout": _op_from_cfg(cfg),
            "fee_bps": info["fee_bps"], "slippage_bps": info["slippage_bps"],
        },
        "mirror_book_value": mirror,
        "flat_book_policy": flat,
        "rollout_legs": legs,
        "interpretation": interp,
        "leak_check": {
            "obs_rows": f"[t-{int(env_cfg.window_size)}, t-1] (position-blind 27ch)",
            "mirror_exact_flip": mirror.get("mirror_audit"),
            "leg_exit_mark": "decision-row price (the row the action executes on)",
            "leg_no_flip_in_step": ("SELL closes long / BUY closes short — a side never "
                                    "flips in one step, so each asset closes at most once "
                                    "per step (no double-count)"),
        },
        "config": {
            "model_size": cfg.model_size, "state_dim": cfg.state_dim,
            "h_cycles": cfg.h_cycles, "l_cycles": cfg.l_cycles,
            "n_assets": int(env_cfg.n_assets),
            "value_scale": searcher.value_scale, "value_shift": searcher.value_shift,
            "bar_minutes": getattr(cfg, "bar_minutes", 1),
            "window_size": int(env_cfg.window_size),
            "reward_attributable": bool(env_cfg.reward_attributable),
            "vector_action": True,
            "vector_budget_step": bool(env_cfg.vector_budget_step),
            "vector_budget_pct": float(env_cfg.vector_budget_pct),
            "vector_weight_rebalance": bool(env_cfg.vector_weight_rebalance),
            "vector_weight_unit": float(env_cfg.vector_weight_unit),
            "vector_weight_max": float(env_cfg.vector_weight_max),
            "vector_rebalance_deadband": float(env_cfg.vector_rebalance_deadband),
            "obs_is_price_only_27ch": True,
        },
        "jax_version": jax.__version__,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    return result


def _finish(result, out_path):
    result["out_name"] = os.path.basename(out_path)
    result["out_pointer"] = os.path.relpath(out_path, _REPO)
    result["ledger_row"] = make_ledger_row(result)
    missing = [k for k in REQUIRED_KEYS if k not in result]
    assert not missing, f"JSON schema incomplete, missing {missing}"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1, default=float)
    print(f"[844] results -> {out_path}")
    print("[844] LEDGER row:")
    print(result["ledger_row"])
    return result


# =============================================================================
# fixture smoke (REAL save->load path + end-to-end probe; NO interpretation applied)
# =============================================================================

# Untrained-model plumbing tolerances. The RAW mirror-Δ mean is NOT ≈0 on a fresh init
# (vec_act_proj's arbitrary BUY/SELL zero-point — MAJOR-1), so the smoke gates on the
# CALIBRATED statistic instead: a fresh-init "checkpoint" is drawn from the SAME init
# distribution as the matched null, so its null-subtracted, init-standardized z is ≈0 by
# construction and seed-robust (verified across init seeds 0,1,7,8).
SMOKE_CAL_Z_TOL = 3.0            # |calibrated z| — a fresh init is a typical draw from the null
SMOKE_RATIO_LO, SMOKE_RATIO_HI = 0.60, 1.60   # sell/buy ≈ 1.0; comfortably below >1.5


def run_fixture_smoke(args):
    cfg = smoke_training_config(args.seed)
    n_assets = args.n_assets

    model_a, trainer_a = build_model(cfg, n_assets, init_seed=args.seed)
    saved_params = trainer_a.state.params
    tmpdir = tempfile.mkdtemp(prefix="e844_bias_ckpt_")
    try:
        trainer_a.save_checkpoint(tmpdir)
        print(f"[smoke] checkpoint saved -> {tmpdir} ({sorted(os.listdir(tmpdir))})")

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
        print("[smoke] save->load round trip verified (params restored, fresh init "
              "overwritten)")

        panel = build_panel(cfg, n_assets, args.seed, db_override=args.db,
                            max_decisions=args.max_decisions)
        searcher = build_searcher(cfg, panel[2])
        result = assemble(
            model_b, trainer_b.state.params, cfg, panel, args,
            checkpoint_label=("fresh-init (fixture smoke; saved+reloaded via "
                              "trainer.save_checkpoint/load_checkpoint)"),
            smoke=True, searcher=searcher,
            restored_step_count=int(trainer_b.state.step_count))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    # --- smoke assertions: plumbing proof, NOT a verdict ---------------------
    mv = result["mirror_book_value"]
    ds = mv["delta_stats"]
    cal = mv["calibrated"]
    assert mv["n_states_with_book"] >= 1, \
        "SMOKE FAIL: no book-bearing states collected — mirror probe never exercised"
    assert ds["mean"] is not None and np.isfinite(ds["mean"]), \
        f"SMOKE FAIL: mirror-Δ mean not finite ({ds})"
    assert cal["n_null"] >= MIN_NULL, (
        f"SMOKE FAIL: matched null under-powered (n_null={cal['n_null']} < {MIN_NULL}) — "
        "the calibration cannot separate the architectural zero-point from a real signal")
    z = cal["z"]
    assert z is not None and np.isfinite(z), (
        f"SMOKE FAIL: calibrated mirror-Δ z not finite ({cal}) — null spread degenerate")
    assert abs(z) < SMOKE_CAL_Z_TOL, (
        f"SMOKE FAIL: untrained CALIBRATED mirror-Δ z {z:.4f} not ≈0 "
        f"(|·| >= {SMOKE_CAL_Z_TOL}) — a fresh init should be a typical draw from the "
        "matched fresh-init null; a large |z| means the calibration or null is broken")
    ratio = result["flat_book_policy"]["raw_prior"]["sell_over_buy"]
    assert ratio is not None and np.isfinite(ratio), \
        f"SMOKE FAIL: flat-book sell/buy ratio not finite ({ratio})"
    assert SMOKE_RATIO_LO < ratio < SMOKE_RATIO_HI, (
        f"SMOKE FAIL: untrained sell/buy ratio {ratio:.4f} not ≈1.0 "
        f"([{SMOKE_RATIO_LO}, {SMOKE_RATIO_HI}]) — marginals biased on a symmetric model")
    lg = result["rollout_legs"]
    for side in ("long", "short"):
        assert np.isfinite(lg["realized_log_pnl_by_side"][side]), \
            f"SMOKE FAIL: rollout realized-log {side} not finite"
    assert result["interpretation"]["applied"] is False, \
        "SMOKE FAIL: interpretation must NOT be applied to the untrained case"

    result = _finish(result, args.out or SMOKE_OUT)
    print(f"[SMOKE] PASS — calibrated mirror-Δ z={z:+.4f} (≈0; raw mean {ds['mean']:+.4f} "
          f"is NOT ≈0 by design, null_mean {cal['null_mean']:+.4f}±{cal['null_std']:.4f} "
          f"over {cal['n_null']} inits), sell/buy={ratio:.4f} (≈1.0), legs exec={lg['n_exec']}; "
          "end-to-end plumbing verified, interpretation not applied")
    return result


# =============================================================================
# checkpoint mode (the real postmortem on a saved snapshot — scheduled/planner-owned)
# =============================================================================

def run_checkpoint(args):
    cfg = build_training_config(args.config_json)
    if args.db:
        print(f"[config] db override: {args.db}")
    n_assets = args.n_assets
    if not getattr(cfg, "vector_action", False):
        raise SystemExit("TASK_844: the run config must set vector_action=true "
                         "(this probe decodes the vector-model value/policy).")
    model, trainer = build_model(cfg, n_assets, init_seed=args.seed)
    ckpt = os.path.abspath(args.checkpoint)
    check_ckpt_metadata(ckpt, model)

    # 834 fail-loud restore guards (probe_runner.py:740-753), mirrored verbatim.
    fresh_leaves = jax.tree_util.tree_leaves(trainer.state.params)
    trainer.state = trainer.load_checkpoint(ckpt)
    step_count = int(trainer.state.step_count)
    if step_count <= 0:
        raise SystemExit(
            f"TASK_844: restored step_count={step_count} — a trained snapshot must have "
            f"step_count > 0. {ckpt!r} is not a trained checkpoint (or the restore was a "
            "silent no-op). Refusing to emit a postmortem.")
    restored_leaves = jax.tree_util.tree_leaves(trainer.state.params)
    if all(np.allclose(a, b) for a, b in zip(fresh_leaves, restored_leaves)):
        raise SystemExit(
            "TASK_844: restored params are IDENTICAL to the fresh init — the flax restore "
            f"from {ckpt!r} was a silent no-op. Refusing to emit a postmortem on a "
            "fresh-init model.")
    print(f"[ckpt] restored {ckpt} (step_count={step_count}; params differ from fresh "
          "init — restore verified)")

    panel = build_panel(cfg, n_assets, args.seed, db_override=args.db,
                        max_decisions=args.max_decisions)
    searcher = build_searcher(cfg, panel[2])
    result = assemble(model, trainer.state.params, cfg, panel, args,
                      checkpoint_label=ckpt, smoke=False, searcher=searcher,
                      restored_step_count=step_count)
    out_path = args.out or os.path.join(MODULE_DIR, "results_probe.json")
    result = _finish(result, out_path)
    print(f"[probe] {result['interpretation']['reading']}")
    return result


# =============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fixture-smoke", action="store_true",
                    help="fresh-init untrained model on the fixture db; exercises the "
                         "real save->load path end-to-end (no interpretation)")
    ap.add_argument("--checkpoint", default=None,
                    help="snapshot dir (jax_ckpt_fast/<name>/snapshot_ep<N>)")
    ap.add_argument("--config-json", default=None,
                    help="RUN TrainingConfig as JSON field overrides (required with "
                         "--checkpoint; 834 pattern — weight/budget flags flow through)")
    ap.add_argument("--db", default=None, help="override config db_path")
    ap.add_argument("--n-assets", type=int, default=None,
                    help="run n_assets (checkpoint default 64, smoke default 16)")
    ap.add_argument("--max-decisions", type=int, default=None,
                    help="cap rollout decisions (smoke default 30; checkpoint None=full)")
    ap.add_argument("--mirror-states", type=int, default=None,
                    help="probe-1 walk collect steps (books) (smoke 24, checkpoint 64)")
    ap.add_argument("--mirror-burn-in", type=int, default=6,
                    help="probe-1 walk burn-in before book collection (834 default)")
    ap.add_argument("--mirror-null-inits", type=int, default=None,
                    help="probe-1 matched fresh-init nulls for the mirror-Δ calibration "
                         "(MAJOR-1; smoke 5, checkpoint 6). 0 disables calibration.")
    ap.add_argument("--flat-states", type=int, default=None,
                    help="probe-2 flat-book states across the panel (smoke 16, ckpt 64)")
    ap.add_argument("--seed", type=int, default=0,
                    help="probe seed (walk + rollout); multi-seed = rerun with new --seed")
    ap.add_argument("--out", default=None, help="output JSON path")
    args = ap.parse_args()

    if args.fixture_smoke:
        args.n_assets = args.n_assets or 16
        args.max_decisions = 30 if args.max_decisions is None else args.max_decisions
        args.mirror_states = args.mirror_states or 24
        args.flat_states = args.flat_states or 16
        if args.mirror_null_inits is None:
            args.mirror_null_inits = 5
        run_fixture_smoke(args)
    else:
        if not args.checkpoint or not args.config_json:
            ap.error("checkpoint mode needs --checkpoint AND --config-json "
                     "(or use --fixture-smoke)")
        args.n_assets = args.n_assets or 64
        args.mirror_states = args.mirror_states or 64
        args.flat_states = args.flat_states or 64
        if args.mirror_null_inits is None:
            args.mirror_null_inits = 6
        run_checkpoint(args)


if __name__ == "__main__":
    main()
