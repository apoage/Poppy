#!/usr/bin/env python3
"""TASK_836 / E4 — value-ranking probe (does the learned VALUE net rank states above momentum?).

E3 (834) gates the reward head; 835 grades full rollouts. Neither isolates the VALUE
net — if a trained policy underperforms we cannot tell "value can't rank states" from
"search/policy can't exploit a good value". This probe closes that gap:

  1. Load a Stage-3 snapshot with the 834 checkpoint pattern INCLUDING its fail-loud
     restore guards (imported via importlib — the 832 cross-experiment precedent).
  2. Walk the run's OWN env deterministically over the FULL panel timeline (835
     fixed-start pattern, n_envs=1), actions from the ZERO-logit dithered-book
     proposal with a fixed seed (834 precedent) — checkpoint-INDEPENDENT, so every
     snapshot is ranked on the identical seed-fixed states. (A HOLD-walk is
     pre-registered AGAINST: from the all-cash reset it never opens a position, the
     PV series is constant, and the realized-return leg degenerates to all-zeros.)
  3. Per decision-point state t: predicted ROOT value = canon decode
     ``_categorical_to_scalar(initial_inference(obs)['value'])`` (sampled_search.py:245),
     obs = training rule [t-window, t-1] via ``jit_batch_encode_simple`` (the 832 PIN b /
     835 panel-consistency rule; the 27ch obs is PRICE-ONLY — position-blind).
  4. REALIZED forward return over horizon k = pv[t+k]/pv[t] - 1 along the SAME walk —
     the k-decision continuation under the same pre-registered deterministic rule,
     computed by the TRUE env (``jit_batch_step_vector``). Realized returns are
     therefore CONDITIONAL on the walk rule, not on the trained policy. k=3 primary
     (the 831x/C value-horizon finding), k=1 / k=6 secondary.
  5. Metric: Spearman rank correlation (scipy.stats.spearmanr — pre-registered;
     present in trader_env) per window per horizon: full window + the four 831-EXT
     regime windows (definitions LOADED from runs/831_ext/<name>.json via the 835
     loader — never hardcoded), windows = timestamp-filtered subsets of the one walk.
     Baselines: trailing m=21-bar momentum (a) PRIMARY "book" — current holdings
     revalued at rows t-1 vs t-1-m (holdings-weighted trailing asset momentum;
     positions/cash/entries frozen — NOT the trailing walk PV); (b) SECONDARY
     "market" — equal-weight cross-asset trailing return (information-symmetric
     to the price-only obs); (c) SECONDARY "walk_pv" — the true trailing-PV
     time-mirror pv[t]/pv[t-m]-1 on the same walk PV series (rows <= t, the
     realized-leg mark convention; requires t >= m).

Interpretation (pre-registered, quoted verbatim from tasks/TASK_836): value beats
momentum's correlation on >=3/5 windows = the value net carries signal beyond the
obvious; below-momentum everywhere = the value net is the bottleneck. DIAGNOSTIC —
it gates nothing by itself.

Leak protocol: value obs rows [t-window, t-1]; momentum rows [t-1-m, t-1]; the
realized leg reads rows [t, t+k] — FUTURE relative to the state, which is its purpose.
Offsets are asserted explicitly (warm == window_size, warm-1-m >= 0).

CPU-only (JAX_PLATFORMS=cpu); fixture db by default; never GPU in the factory loop.
The real snapshot probe is a scheduled run (planner-owned), not this smoke.

Run:  JAX_PLATFORMS=cpu python experiments/836_value_ranking/rank_probe.py --fixture-smoke
      JAX_PLATFORMS=cpu python experiments/836_value_ranking/rank_probe.py \
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
from mcts_trader.jax_agent.sampled_search import SampledMuZeroVectorSearch

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
FIXTURE_DB = os.path.join(_REPO, "fixtures", "trading_fixture.db")
SMOKE_OUT = os.path.join(MODULE_DIR, "results_fixture_smoke.json")

HOLD = 1
N_ACTIONS_PER_ASSET = 5
EPS = 1e-10                      # division guard (CLAUDE.md: jnp.maximum(x, 1e-10))
BATTERY_HOLD_BIAS = 3.0          # 828 Stage-3 operator (CLI default 0.0 is a trap)

# --- pre-registered probe constants (SPEC.md) --------------------------------
HORIZONS = (1, 3, 6)             # k=3 primary (831x/C value-horizon finding)
PRIMARY_HORIZON = 3
MOMENTUM_M = 21                  # trailing lookback, bars (~1 month at daily bars)
MIN_N = 8                        # below this a rank correlation is emitted as null
EXT_WINDOW_NAMES = ("bear22w", "bull23w", "y2024", "recent")
INTERPRETATION_VERBATIM = (
    "value beats momentum's correlation on ≥3/5 windows = the value net carries "
    "signal beyond the obvious; below-momentum everywhere = the value net is the "
    "bottleneck (feeds the phase-2 postmortem either way — this probe is "
    "DIAGNOSTIC, it gates nothing by itself)")

REQUIRED_KEYS = (
    "task", "mode", "checkpoint", "db", "window", "panel", "seed", "burn_in",
    "walk", "momentum", "horizons", "leak_check", "spearman_source", "windows",
    "interpretation", "restored_step_count", "config", "jax_version", "timestamp",
)

# =============================================================================
# cross-experiment imports (the 832 importlib precedent, imprint.py:85-94):
# 834 = checkpoint-load pattern + fail-loud restore guards (DO NOT re-derive);
# 835 = 831-EXT window loader + panel row-timestamp mirror (LOADED, not hardcoded).
# =============================================================================

def _load_experiment_module(rel_path, name):
    path = os.path.join(_REPO, rel_path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


probe834 = _load_experiment_module("experiments/834_e3_probe/probe_runner.py", "probe834")
eval835 = _load_experiment_module("experiments/835_stage4_eval/eval_vector.py", "eval835")

# Reused 834 pieces (probe_runner.py:113-201): --config-json -> TrainingConfig with
# setattr extras; model+trainer build (vector assert + init_train_state); the
# trainer_metadata.json cross-check that fails loud BEFORE a silent flax no-op restore.
build_training_config = probe834.build_training_config
build_model = probe834.build_model


def check_ckpt_metadata(ckpt_path, model):
    """834's fail-loud metadata guard verbatim (probe_runner.py:175-201), re-branded
    so a TASK_836 failure is attributed to this probe in factory logs."""
    try:
        probe834.check_ckpt_metadata(ckpt_path, model)
    except SystemExit as exc:
        raise SystemExit(f"TASK_836 (via the imported 834 guard): {exc}") from None


# =============================================================================
# Spearman (pre-registered source: scipy.stats.spearmanr — verified in trader_env)
# =============================================================================

try:
    from scipy.stats import spearmanr as _scipy_spearmanr
    SPEARMAN_SOURCE = "scipy.stats.spearmanr"
except ImportError:                                    # pragma: no cover — trader_env has scipy
    _scipy_spearmanr = None
    SPEARMAN_SOURCE = "internal rank-correlation fallback (scipy unavailable)"


def _rank_avg(x):
    """Average ranks (ties -> mean rank), matching spearmanr's tie handling."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman_or_null(x, y):
    """{'n', 'rho', 'p', 'note'} — null rho (with note) for n<MIN_N / constant /
    non-finite inputs so a degenerate window can never masquerade as a signal."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = int(len(x))
    if n < MIN_N:
        return {"n": n, "rho": None, "p": None, "note": f"n < {MIN_N} — not evaluated"}
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        return {"n": n, "rho": None, "p": None, "note": "non-finite input"}
    if np.all(x == x[0]) or np.all(y == y[0]):
        return {"n": n, "rho": None, "p": None, "note": "degenerate (constant series)"}
    if _scipy_spearmanr is not None:
        rho, p = _scipy_spearmanr(x, y)
        rho, p = float(rho), float(p)
    else:
        rx, ry = _rank_avg(x), _rank_avg(y)
        rx -= rx.mean()
        ry -= ry.mean()
        denom = float(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
        rho, p = float((rx * ry).sum() / max(denom, EPS)), None
    if not np.isfinite(rho):
        return {"n": n, "rho": None, "p": None, "note": "non-finite correlation"}
    return {"n": n, "rho": rho, "p": (p if p is None or np.isfinite(p) else None)}


# =============================================================================
# fixture-smoke config (mirror of 834's smoke_training_config, probe_runner.py:133-156
# — same fixture slice / bar_minutes rationale; renamed for 836)
# =============================================================================

def smoke_training_config(seed):
    cfg = TrainingConfig(
        name="836_smoke", seed=seed, parallel_envs=1, steps=96, epochs=1,
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
    return cfg


# =============================================================================
# panel — the run's own env over the FULL timeline (834 config mapping + 835
# fixed-start deterministic-walk state init + timestamp-axis consistency assert)
# =============================================================================

def build_panel(cfg, n_assets, seed, db_override, max_decisions, burn_in):
    start_date, end_date = cfg.start_date, cfg.end_date
    specific_symbols = None
    if getattr(cfg, "periods_file", None):       # 834 build_env branch, verbatim
        with open(cfg.periods_file) as f:
            periods_config = json.load(f)
        first_period = periods_config["periods"][0]
        start_date = first_period["start_date"]
        end_date = first_period["end_date"]
        specific_symbols = first_period.get("assets")
        print(f"[env] periods file: {cfg.periods_file} "
              f"({start_date[:10]} -> {end_date[:10]}, "
              f"{len(specific_symbols) if specific_symbols else 0} pinned symbols)")
    elif getattr(cfg, "assets_file", None):      # train_jax_fast.py:233-236 parity
        with open(cfg.assets_file) as f:
            specific_symbols = json.load(f)
        print(f"[env] assets file: {cfg.assets_file} "
              f"({len(specific_symbols)} pinned symbols)")
    db_path = db_override or cfg.db_path
    states0, prices, env_cfg = create_jax_env_historical(
        db_path=db_path,
        start_date=start_date,
        end_date=end_date,
        n_envs=1,                                # single deterministic walk (835)
        n_assets=n_assets,
        specific_symbols=specific_symbols,
        fee_bps=float(cfg.fee),
        slippage_bps=float(cfg.slippage),
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
        # TASK_839: budget-constrained vector-step sizing, from the run config (default OFF).
        vector_budget_step=getattr(cfg, "vector_budget_step", False),
        vector_budget_pct=getattr(cfg, "vector_budget_pct", 0.99),
        # TASK_840: target-weight rebalance step, from the run config (default OFF).
        vector_weight_rebalance=getattr(cfg, "vector_weight_rebalance", False),
        vector_weight_unit=getattr(cfg, "vector_weight_unit", -1.0),
        vector_weight_max=getattr(cfg, "vector_weight_max", 4.0),
        vector_rebalance_deadband=getattr(cfg, "vector_rebalance_deadband", -1.0),
        rng_key=jax.random.PRNGKey(seed),
    )
    # Row-timestamp axis via the 835 read-only loader mirror; length-asserted
    # against the panel (the 832 PIN b panel-consistency rule).
    row_ts, loaded_syms = eval835.load_row_timestamps(
        db_path, start_date, end_date, specific_symbols, n_assets,
        getattr(cfg, "bar_minutes", 1))
    T = int(prices.shape[0])
    if len(row_ts) != T:
        raise SystemExit(
            f"TASK_836: panel-consistency violation — reconstructed timestamp axis "
            f"has {len(row_ts)} rows but create_jax_env_historical produced {T}")
    warm = int(env_cfg.window_size)
    n_max = T - 1 - warm
    n_decisions = n_max if max_decisions is None else min(n_max, max_decisions)
    need = burn_in + max(HORIZONS) + MIN_N
    if n_decisions < need:
        raise SystemExit(
            f"TASK_836: panel too short — n_decisions={n_decisions} < "
            f"burn_in+max(k)+MIN_N={need} (T={T}, warm={warm})")
    # Deterministic full-timeline walk: fixed start at the first obs-valid row (835).
    state_init = states0.replace(
        episode_start_idx=jnp.full_like(states0.episode_start_idx, warm),
        current_step=jnp.zeros_like(states0.current_step),
        max_episode_steps=jnp.full_like(states0.max_episode_steps, n_decisions + 2),
        initial_portfolio_value=jnp.full_like(states0.initial_portfolio_value,
                                              env_cfg.initial_cash),
        prev_portfolio_value=jnp.full_like(states0.prev_portfolio_value,
                                           env_cfg.initial_cash),
    )
    # ---- leak-protocol asserts (explicit horizon offsets) --------------------
    # value obs rows = [t-window, t-1]: window_starts = price_idx - window_size, so
    # the FIRST state's obs window starts at row episode_start_idx - window_size,
    # which must be exactly 0 (no leak, no underrun) — asserted on the state that
    # the walk actually uses, not on how `warm` happened to be derived.
    assert int(state_init.episode_start_idx[0]) == int(env_cfg.window_size), \
        ("TASK_836: walk must start at the first obs-valid row "
         "(episode_start_idx == window_size); obs window would leak/underrun")
    if warm - 1 - MOMENTUM_M < 0:
        raise SystemExit(
            f"TASK_836: momentum lookback m={MOMENTUM_M} unreachable — needs "
            f"warm-1-m >= 0 but warm={warm} (momentum must never see beyond t-1)")
    info = {
        "db": db_path, "start": start_date, "end": end_date, "n_rows": T,
        "n_assets": int(env_cfg.n_assets), "warm": warm,
        "n_decisions": n_decisions, "n_loaded_symbols": len(loaded_syms),
        "padded": len(loaded_syms) < n_assets,
        "fee_bps": float(env_cfg.fee_bps), "slippage_bps": float(env_cfg.slippage_bps),
    }
    return state_init, prices, env_cfg, row_ts, info


# =============================================================================
# deterministic walk + per-state measurement
# =============================================================================

def make_step_fns(model, searcher, prices, env_cfg, max_positions,
                  min_cash_threshold):
    """Jitted (params, state, key) -> (new_state, per-state scalars).

    Per decision-point state t (BEFORE the step executes):
      value      — canon root decode _categorical_to_scalar(initial_inference['value'])
                   on the training obs rule [t-window, t-1] (jit_batch_encode_simple,
                   27ch price-only — self_play.py:646-658 / 835 mirror);
      mom_book   — PRIMARY momentum: current holdings revalued at rows t-1 vs t-1-m
                   via compute_portfolio_value (holdings-weighted trailing asset
                   momentum; positions/cash/entries frozen — NOT the trailing walk
                   PV; that true time-mirror is the momentum_walk_pv column,
                   computed from the pv_pre series after the walk);
      mom_mkt    — SECONDARY momentum: equal-weight cross-asset trailing return
                   (rows t-1 vs t-1-m; padded columns double-count — diagnostic);
      pv_pre     — the realized-leg mark: PV of state t at its decision row t.
    Walk action = zero-logit dithered-book proposal sample (834 precedent: book=None
    -> the HOLD slot gets hold_bias inside _build_proposal), legality-masked per state
    exactly as battery self-play; checkpoint-independent by construction.
    """
    A = env_cfg.n_assets
    n_envs = 1
    enc_cfg = EncoderConfig(n_assets=A, window_size=env_cfg.window_size,
                            n_channels=27)               # self_play.py:627 (27ch)
    zero_logits = jnp.zeros((n_envs, A * N_ACTIONS_PER_ASSET), dtype=jnp.float32)

    def _step(params, state, key):
        price_idx = state.episode_start_idx + state.current_step          # (1,)
        window_starts = price_idx - enc_cfg.window_size                   # ends at t-1
        obs = jit_batch_encode_simple(prices, window_starts, enc_cfg)
        current_prices = prices[price_idx]                                # (1, A)
        mask = jit_batch_compute_action_mask(
            state, current_prices, env_cfg, max_positions, min_cash_threshold)
        # value leg — canon root decode (sampled_search.py:245)
        init = model.apply({"params": params}, obs, mode="initial_inference")
        value = searcher._categorical_to_scalar(init["value"])            # (1,)
        # walk action — zero-logit dithered-book proposal, seed-fixed (834)
        dither_key, sample_key = jax.random.split(key)
        proposal = searcher._build_proposal(zero_logits, mask, None, dither_key)
        action_vec = jax.random.categorical(
            sample_key, proposal, axis=-1).astype(jnp.int32)              # (1, A)
        # momentum legs — inputs end at row t-1 (never beyond t)
        p_lag1 = prices[price_idx - 1]
        p_lagm = prices[price_idx - 1 - MOMENTUM_M]
        pv_lag1 = jax.vmap(compute_portfolio_value)(state, p_lag1)
        pv_lagm = jax.vmap(compute_portfolio_value)(state, p_lagm)
        mom_book = pv_lag1 / jnp.maximum(pv_lagm, EPS) - 1.0
        mom_mkt = jnp.mean(p_lag1 / jnp.maximum(p_lagm, EPS) - 1.0, axis=-1)
        # realized-leg mark at the decision row t (forward window starts here)
        pv_pre = jax.vmap(compute_portfolio_value)(state, current_prices)
        # diagnostics
        mask_pa = mask.reshape(n_envs, A, N_ACTIONS_PER_ASSET)
        legal = jnp.take_along_axis(mask_pa, action_vec[..., None], axis=-1)[..., 0]
        illegal_any = jnp.any(legal == 0)
        pct_hold = jnp.mean((action_vec == HOLD).astype(jnp.float32))
        n_orders = (jnp.sum(state.pos_active.astype(jnp.int32))
                    + jnp.sum(state.order_active.astype(jnp.int32)))
        new_state, _, _, done = jit_batch_step_vector(
            state, action_vec, prices, env_cfg)
        scalars = (value[0], mom_book[0], mom_mkt[0], pv_pre[0],
                   pct_hold, n_orders, illegal_any, done[0])
        return new_state, scalars

    @jax.jit
    def _pv_at_current(state):
        price_idx = state.episode_start_idx + state.current_step
        return jax.vmap(compute_portfolio_value)(state, prices[price_idx])[0]

    return jax.jit(_step), _pv_at_current


def run_walk(step_fn, pv_fn, params, state_init, n_decisions, seed):
    """n_decisions sequential decisions (or fewer on env done/liquidation).
    Returns per-state arrays + the (n_exec+1)-long pre-decision PV mark series."""
    state = state_init
    base_key = jax.random.PRNGKey(seed)
    value = np.empty(n_decisions)
    mom_book = np.empty(n_decisions)
    mom_mkt = np.empty(n_decisions)
    pv_pre = np.empty(n_decisions + 1)
    pct_hold_sum = 0.0
    orders = np.empty(n_decisions, dtype=np.int64)
    illegal_steps = 0
    first_done = -1
    n_exec = 0
    for t in range(n_decisions):
        state, scalars = step_fn(params, state, jax.random.fold_in(base_key, t))
        jax.block_until_ready(scalars[0])            # device->host boundary
        v, mb, mm, pv, ph, no, ill, done = (np.asarray(s) for s in scalars)
        value[t], mom_book[t], mom_mkt[t], pv_pre[t] = v, mb, mm, pv
        pct_hold_sum += float(ph)
        orders[t] = int(no)
        illegal_steps += int(bool(ill))
        n_exec = t + 1
        if bool(done):
            first_done = t
            break
    pv_final = pv_fn(state)
    jax.block_until_ready(pv_final)
    pv_pre[n_exec] = float(np.asarray(pv_final))     # mark AFTER the last decision
    return {
        "value": value[:n_exec], "mom_book": mom_book[:n_exec],
        "mom_mkt": mom_mkt[:n_exec], "pv_pre": pv_pre[:n_exec + 1],
        "n_exec": n_exec, "first_done": first_done,
        "pct_hold_walk": pct_hold_sum / max(n_exec, 1),
        "orders_mean": float(orders[:n_exec].mean()),
        "orders_max": int(orders[:n_exec].max()),
        "illegal_action_steps": illegal_steps,
        "final_pv": float(pv_pre[n_exec]),
    }


# =============================================================================
# per-window / per-horizon Spearman table
# =============================================================================

def load_ext_windows():
    """The four 831-EXT regime windows via the 835 loader (runs/831_ext/<name>.json,
    key ``window`` — LOADED, never hardcoded). Missing file -> None (key still emitted)."""
    return eval835.load_ext_windows()


_NULL_HORIZON = {"n": 0, "value": None, "momentum_book": None,
                 "momentum_market": None, "momentum_walk_pv": None,
                 "value_beats_primary": None, "realized_std": None}


def score_windows(walk, row_ts, warm, burn_in, ext_windows, panel_window):
    """windows -> per_horizon -> {n, value, momentum_book, momentum_market,
    momentum_walk_pv, value_beats_primary, realized_std}. States enter a horizon-k
    sample iff t >= burn_in (books accumulated) and t+k <= n_exec (forward mark
    exists; post-done excluded by construction since the walk stops at done).
    The walk_pv leg additionally needs t >= m (its trailing-PV lookback); its own
    'n' reflects that sub-sample."""
    n_exec = walk["n_exec"]
    pv = walk["pv_pre"]
    ts_dec = np.asarray(row_ts[warm:warm + n_exec], dtype=np.int64)

    specs = {"full": {"window": list(panel_window), "source": "panel",
                      "start_ts": None, "end_ts": None}}
    for name in EXT_WINDOW_NAMES:
        wd = ext_windows.get(name)
        if wd is None:
            specs[name] = None
        else:
            specs[name] = {"window": wd["window"], "source": wd["source"],
                           "start_ts": wd["start_ts"], "end_ts": wd["end_ts"]}

    out = {}
    for name, spec in specs.items():
        if spec is None:
            out[name] = {"window": None, "note": "831-EXT window file missing",
                         "per_horizon": {str(k): dict(_NULL_HORIZON)
                                         for k in HORIZONS}}
            continue
        per_h = {}
        for k in HORIZONS:
            t_idx = np.arange(burn_in, max(n_exec - k + 1, burn_in))
            if spec["start_ts"] is not None:
                sel = ((ts_dec[t_idx] >= spec["start_ts"])
                       & (ts_dec[t_idx] <= spec["end_ts"]))
                t_idx = t_idx[sel]
            if len(t_idx) == 0:
                per_h[str(k)] = dict(_NULL_HORIZON)
                continue
            realized = pv[t_idx + k] / np.maximum(pv[t_idx], EPS) - 1.0
            s_val = spearman_or_null(walk["value"][t_idx], realized)
            s_book = spearman_or_null(walk["mom_book"][t_idx], realized)
            s_mkt = spearman_or_null(walk["mom_mkt"][t_idx], realized)
            # walk_pv leg: the trailing-PV mirror needs t >= m (its own sub-sample).
            t_wpv = t_idx[t_idx >= MOMENTUM_M]
            if len(t_wpv):
                realized_wpv = pv[t_wpv + k] / np.maximum(pv[t_wpv], EPS) - 1.0
                mom_wpv = pv[t_wpv] / np.maximum(pv[t_wpv - MOMENTUM_M], EPS) - 1.0
                s_wpv = spearman_or_null(mom_wpv, realized_wpv)
            else:
                s_wpv = {"n": 0, "rho": None, "p": None,
                         "note": f"no states with t >= m={MOMENTUM_M}"}
            beats = (None if s_val["rho"] is None or s_book["rho"] is None
                     else bool(s_val["rho"] > s_book["rho"]))
            per_h[str(k)] = {"n": int(len(t_idx)), "value": s_val,
                             "momentum_book": s_book, "momentum_market": s_mkt,
                             "momentum_walk_pv": s_wpv,
                             "value_beats_primary": beats,
                             "realized_std": float(np.std(realized))}
        out[name] = {"window": spec["window"], "source": spec["source"],
                     "per_horizon": per_h}
    return out


def interpret(windows, smoke):
    """Apply the pre-registered interpretation at the PRIMARY horizon/baseline.
    Never applied to the untrained smoke (plumbing proof, 834/835 precedent)."""
    k = str(PRIMARY_HORIZON)
    names = ("full",) + EXT_WINDOW_NAMES
    evaluable, beats = [], []
    windows_sampled, value_leg_notes = [], {}
    for name in names:
        ph = windows[name]["per_horizon"][k]
        b = ph["value_beats_primary"] if ph else None
        if b is not None:
            evaluable.append(name)
            if b:
                beats.append(name)
        # MINOR-3 disambiguation inputs: windows that HAD enough samples, and the
        # notes attached to any nulled value legs (degenerate vs coverage).
        if ph and ph.get("n", 0) >= MIN_N:
            windows_sampled.append(name)
        s = ph.get("value") if ph else None
        if isinstance(s, dict) and s.get("rho") is None and s.get("note"):
            value_leg_notes[name] = s["note"]
    block = {
        "rule_verbatim": INTERPRETATION_VERBATIM,
        "primary_horizon_k": PRIMARY_HORIZON,
        "primary_baseline": f"momentum_book_m{MOMENTUM_M}",
        "n_windows_total": len(names),
        "n_windows_evaluable": len(evaluable),
        "windows_evaluable": evaluable,
        "n_windows_value_beats_momentum": len(beats),
        "windows_value_beats_momentum": beats,
        "value_leg_notes": value_leg_notes or None,
        "applied": not smoke,
    }
    if smoke:
        block["reading"] = ("FIXTURE-SMOKE (untrained fresh-init model) — plumbing "
                            "proof only; interpretation NOT applied. Real reading = "
                            "scheduled run on the phase-1 cost0 snapshots.")
        return block
    nb, ne, nt = len(beats), len(evaluable), len(names)
    if ne == 0:
        # MINOR 3: distinguish WHY nothing is evaluable. A window that had enough
        # samples but a value leg nulled as degenerate/non-finite is collapse
        # evidence, not a coverage gap.
        degen = [nm for nm in windows_sampled
                 if value_leg_notes.get(nm, "").startswith(("degenerate", "non-finite"))]
        if windows_sampled and len(degen) == len(windows_sampled):
            reading = ("value output degenerate everywhere — value net is trivially "
                       "the bottleneck (strongest collapse evidence).")
        else:
            reading = "no evaluable windows — coverage failure; re-scope the probe."
    elif nb >= 3:
        reading = (f"value beats momentum in {nb}/{ne} evaluable windows (>=3/{nt}) "
                   "— the value net carries signal beyond the obvious.")
    elif nb == 0 and ne == nt:
        reading = ("below-momentum everywhere — the value net is the bottleneck "
                   "(feeds the phase-2 postmortem).")
    else:
        reading = (f"mixed/partial: value beats momentum in {nb}/{ne} evaluable "
                   f"windows (coverage {ne}/{nt}).")
    block["reading"] = reading + " DIAGNOSTIC — gates nothing by itself."
    return block


# =============================================================================
# result assembly
# =============================================================================

def _fmt_rho(s):
    if s is None or s.get("rho") is None:
        return "n/a"
    return f"{s['rho']:+.3f}"


def make_ledger_row(r):
    """One experiments/LEDGER.md-convention row. Pointer = r['out_pointer'], derived
    by the caller from the ACTUAL output path (the 835 MINOR-1 / 734-819
    stale-pointer rule). Corridor column = n/a: this instrument makes no corridor claim."""
    k = str(PRIMARY_HORIZON)
    cells = []
    for name in ("full",) + EXT_WINDOW_NAMES:
        ph = r["windows"][name]["per_horizon"][k]
        cells.append(f"{name} ρv {_fmt_rho(ph['value'])} vs ρm "
                     f"{_fmt_rho(ph['momentum_book'])} (n={ph['n']})")
    itp = r["interpretation"]
    smoke_note = (" FIXTURE-SMOKE — plumbing proof, untrained model, interpretation "
                  "not applied; real numbers come from the scheduled run."
                  if r["mode"] == "fixture-smoke" else
                  f" {itp['reading']}")
    return (
        f"| 836/E4 | {date.today().isoformat()} | H1/H2 | "
        f"Does the learned root VALUE rank states above trailing momentum "
        f"(m={MOMENTUM_M})? {r['checkpoint']}: deterministic zero-logit walk "
        f"{r['window'][0]}..{r['window'][1]}, realized = true-env k-step walk "
        f"continuation, k={PRIMARY_HORIZON} primary (k=1/6 in JSON) | "
        f"value>momentum in {itp['n_windows_value_beats_momentum']}/"
        f"{itp['n_windows_evaluable']} evaluable windows @k={PRIMARY_HORIZON}: "
        + "; ".join(cells) + f".{smoke_note} | "
        f"n/a (diagnostic instrument — no corridor claim) | {r['out_pointer']} |"
    )


def probe_core(model, params, cfg, panel, args, checkpoint_label, smoke,
               restored_step_count=None):
    state_init, prices, env_cfg, row_ts, info = panel
    hold_bias = float(getattr(cfg, "vector_hold_bias", 0.0))
    if hold_bias == 0.0:
        print("[probe] WARNING: vector_hold_bias=0.0 — the 828 Stage-3 battery "
              f"operator is hold-bias {BATTERY_HOLD_BIAS} (the CLI default 0.0 is "
              "a documented trap; check the config JSON).")
    searcher = SampledMuZeroVectorSearch(
        model_apply_fn=None,                     # sampling+decode machinery only (831/834)
        n_assets=env_cfg.n_assets,
        n_actions_per_asset=N_ACTIONS_PER_ASSET,
        num_samples=1,
        hold_bias=hold_bias,
        dither_scale=float(getattr(cfg, "vector_dither_scale", 1.0)),
        proposal_scheme=str(getattr(cfg, "vector_proposal_scheme", "dithered-book")),
        # TASK_847: policy-prior knobs (ignored unless scheme='policy-prior').
        proposal_temp=float(getattr(cfg, "vector_proposal_temp", 1.0)),
        explore_slots=int(getattr(cfg, "vector_proposal_explore_slots", 2)),
        value_scale=float(getattr(cfg, "value_scale", 1.0)),
        value_shift=float(getattr(cfg, "value_shift", 0.0)),
    )
    max_positions = int(getattr(cfg, "max_positions", 16))
    min_cash_threshold = float(getattr(cfg, "min_cash_threshold", 100.0))
    step_fn, pv_fn = make_step_fns(model, searcher, prices, env_cfg,
                                   max_positions, min_cash_threshold)
    t0 = time.time()
    walk = run_walk(step_fn, pv_fn, params, state_init, info["n_decisions"],
                    args.seed)
    print(f"[walk] {walk['n_exec']}/{info['n_decisions']} decisions in "
          f"{time.time() - t0:.1f}s; first_done={walk['first_done']}; "
          f"pct_hold={100 * walk['pct_hold_walk']:.1f}%; orders/state "
          f"mean={walk['orders_mean']:.2f} max={walk['orders_max']} "
          f"(cap {max_positions}); illegal-action steps={walk['illegal_action_steps']}; "
          f"final PV={walk['final_pv']:.0f}")

    ext_windows = load_ext_windows()
    windows = score_windows(walk, row_ts, info["warm"], args.burn_in, ext_windows,
                            (info["start"], info["end"]))
    interp = interpret(windows, smoke)
    k = str(PRIMARY_HORIZON)
    for name in ("full",) + EXT_WINDOW_NAMES:
        ph = windows[name]["per_horizon"][k]
        print(f"[spearman k={k}] {name:>8}: n={ph['n']:>5}  "
              f"rho_value={_fmt_rho(ph['value'])}  "
              f"rho_mom_book={_fmt_rho(ph['momentum_book'])}  "
              f"rho_mom_mkt={_fmt_rho(ph['momentum_market'])}  "
              f"rho_walk_pv={_fmt_rho(ph['momentum_walk_pv'])}  "
              f"value_beats={ph['value_beats_primary']}")

    result = {
        "task": "TASK_836 E4 value-ranking probe",
        "mode": "fixture-smoke" if smoke else "checkpoint",
        "checkpoint": checkpoint_label,
        "db": info["db"],
        "window": [info["start"], info["end"]],
        "panel": {kk: info[kk] for kk in ("n_rows", "n_assets", "warm", "n_decisions",
                                          "n_loaded_symbols", "padded")},
        "seed": args.seed,
        "burn_in": args.burn_in,
        "restored_step_count": restored_step_count,
        "walk": {
            "action_rule": ("zero-logit dithered-book proposal, seed-fixed, "
                            "legality-masked per state (834 precedent) — "
                            "checkpoint-independent; single env, fixed start at "
                            "the first obs-valid row (835 pattern)"),
            "hold_bias": hold_bias,
            "dither_scale": searcher.dither_scale,
            "proposal_scheme": searcher.proposal_scheme,
            "fee_bps": info["fee_bps"], "slippage_bps": info["slippage_bps"],
            "mask_max_positions": max_positions,
            "mask_min_cash_threshold": min_cash_threshold,
            "n_exec": walk["n_exec"], "first_done": walk["first_done"],
            "pct_hold_walk_actions": walk["pct_hold_walk"],
            "orders_per_state_mean": walk["orders_mean"],
            "orders_per_state_max": walk["orders_max"],
            "illegal_action_steps": walk["illegal_action_steps"],
            "final_pv": walk["final_pv"],
        },
        "momentum": {
            "m_bars": MOMENTUM_M,
            "primary": ("book: current holdings revalued at rows t-1 vs t-1-m "
                        "(compute_portfolio_value at two price rows; positions/cash/"
                        "entries frozen — holdings-weighted trailing asset momentum, "
                        "NOT the trailing walk PV)"),
            "secondary": ("market: equal-weight cross-asset mean trailing return, "
                          "rows t-1 vs t-1-m (padded columns double-count — "
                          "diagnostic only)"),
            "walk_pv": ("walk_pv: pv_pre[t]/pv_pre[t-m]-1 on the walk PV series — "
                        "the true trailing-PV time-mirror of the realized leg "
                        "(rows <= t, the realized-leg mark convention, never beyond "
                        "t; requires t >= m, earlier states excluded from this leg "
                        "only; secondary/diagnostic — the pre-registered PRIMARY "
                        "stays the book baseline)"),
            "why_m21": ("~1 month at the canonical daily bars — the standard "
                        "time-series-momentum lookback; < warm=window_size so it is "
                        "always computable without leaving the panel"),
        },
        "horizons": {
            "all": list(HORIZONS), "primary_k": PRIMARY_HORIZON,
            "realized": ("pv_pre[t+k]/pv_pre[t] - 1 along the SAME deterministic "
                         "walk — the k-decision continuation under the same "
                         "pre-registered rule, computed by the true env "
                         "(jit_batch_step_vector); CONDITIONAL on the walk rule, "
                         "not the trained policy"),
            "p_caveat": ("p-values assume independent samples; k>1 forward windows "
                         "overlap by k-1 steps (effective n ~ n/k), making p "
                         "anti-conservative — treat p as descriptive only; the "
                         "interpretation rule uses rho comparisons exclusively"),
        },
        "leak_check": {
            "value_obs_rows": (f"[t-{int(env_cfg.window_size)}, t-1] "
                               "(window_starts+window_size==price_idx)"),
            "momentum_rows": f"[t-1-{MOMENTUM_M}, t-1] (never beyond t-1)",
            "momentum_walk_pv_rows": (f"[t-{MOMENTUM_M}, t] — rows <= t, the "
                                      "realized-leg mark convention (never beyond t)"),
            "realized_rows": "[t, t+k] — FUTURE relative to the state (its purpose)",
            # derived from the actually-used values (NIT 4), not hardcoded literals
            "warm_equals_window_size": bool(info["warm"] == int(env_cfg.window_size)),
            "warm_minus_1_minus_m_nonneg": bool(info["warm"] - 1 - MOMENTUM_M >= 0),
        },
        "spearman_source": SPEARMAN_SOURCE,
        "min_n_for_correlation": MIN_N,
        "windows": windows,
        "interpretation": interp,
        "config": {
            "model_size": cfg.model_size, "state_dim": cfg.state_dim,
            "h_cycles": cfg.h_cycles, "l_cycles": cfg.l_cycles,
            "n_assets": int(env_cfg.n_assets),
            "value_scale": searcher.value_scale, "value_shift": searcher.value_shift,
            "bar_minutes": getattr(cfg, "bar_minutes", 1),
            "window_size": int(env_cfg.window_size),
            "reward_attributable": bool(env_cfg.reward_attributable),
            "vector_action": True,
            # TASK_839: record the step-sizing mode the walk actually ran under.
            "vector_budget_step": bool(env_cfg.vector_budget_step),
            "vector_budget_pct": float(env_cfg.vector_budget_pct),
            # TASK_840: record the weight-rebalance mode the walk actually ran under.
            "vector_weight_rebalance": bool(env_cfg.vector_weight_rebalance),
            "vector_weight_unit": float(env_cfg.vector_weight_unit),
            "vector_weight_max": float(env_cfg.vector_weight_max),
            "vector_rebalance_deadband": float(env_cfg.vector_rebalance_deadband),
            "obs_is_price_only_27ch": True,      # position-blind value input (SPEC note)
        },
        "jax_version": jax.__version__,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    return result


def _finish(result, out_path):
    result["out_name"] = os.path.basename(out_path)
    # LEDGER pointer from the ACTUAL output path (835 MINOR-1: hardcoded pointers
    # dangle — the documented 734-819 stale-pointer failure mode).
    result["out_pointer"] = os.path.relpath(out_path, _REPO)
    result["ledger_row"] = make_ledger_row(result)
    missing = [kk for kk in REQUIRED_KEYS if kk not in result]
    assert not missing, f"JSON schema incomplete, missing {missing}"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1, default=float)
    print(f"[836] results -> {out_path}")
    print("[836] LEDGER row:")
    print(result["ledger_row"])
    return result


# =============================================================================
# fixture smoke (REAL save->load path + end-to-end probe; NO interpretation applied)
# =============================================================================

def run_fixture_smoke(args):
    cfg = smoke_training_config(args.seed)
    n_assets = args.n_assets

    # Fresh-init model A -> save through the REAL trainer checkpoint path (834).
    model_a, trainer_a = build_model(cfg, n_assets, init_seed=args.seed)
    saved_params = trainer_a.state.params
    tmpdir = tempfile.mkdtemp(prefix="e4_rank_ckpt_")
    try:
        trainer_a.save_checkpoint(tmpdir)
        print(f"[smoke] checkpoint saved -> {tmpdir} ({sorted(os.listdir(tmpdir))})")

        # SECOND trainer, DIFFERENT init -> restore must overwrite its params (834).
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
                            max_decisions=args.max_decisions, burn_in=args.burn_in)
        result = probe_core(
            model_b, trainer_b.state.params, cfg, panel, args,
            checkpoint_label=("fresh-init (fixture smoke; saved+reloaded via "
                              "trainer.save_checkpoint/load_checkpoint)"),
            smoke=True,
            restored_step_count=int(trainer_b.state.step_count))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    # --- smoke assertions: plumbing proof, NOT a verdict --------------------
    k = str(PRIMARY_HORIZON)
    for kk in map(str, HORIZONS):
        ph = result["windows"]["full"]["per_horizon"][kk]
        for leg in ("value", "momentum_book", "momentum_market", "momentum_walk_pv"):
            s = ph[leg]
            assert s is not None and s["rho"] is not None and np.isfinite(s["rho"]), \
                f"SMOKE FAIL: full-window k={kk} {leg} correlation not finite ({s})"
    for name in EXT_WINDOW_NAMES:                    # 835 precedent: keys emitted,
        ph = result["windows"][name]["per_horizon"][k]   # n=0/null where uncovered
        assert "n" in ph, f"SMOKE FAIL: window {name} missing from schema"
    assert result["interpretation"]["applied"] is False, \
        "SMOKE FAIL: interpretation must NOT be applied to the untrained case"
    assert result["walk"]["illegal_action_steps"] == 0, \
        ("SMOKE FAIL: masked proposal sampled training-illegal walk actions "
         f"({result['walk']['illegal_action_steps']} steps)")

    result = _finish(result, args.out or SMOKE_OUT)
    print("[SMOKE] PASS — end-to-end plumbing verified (finite full-window "
          "correlations at k=1/3/6, schema complete incl. n=0 EXT windows, "
          "save->load exercised, interpretation not applied)")
    return result


# =============================================================================
# checkpoint mode (the real E4 probe on a saved snapshot — scheduled/planner-owned)
# =============================================================================

def run_checkpoint(args):
    cfg = build_training_config(args.config_json)
    if args.db:
        print(f"[config] db override: {args.db}")
    n_assets = args.n_assets
    if not getattr(cfg, "vector_action", False):
        raise SystemExit("TASK_836: the run config must set vector_action=true "
                         "(this probe decodes the vector-model root value).")
    model, trainer = build_model(cfg, n_assets, init_seed=args.seed)
    ckpt = os.path.abspath(args.checkpoint)
    check_ckpt_metadata(ckpt, model)

    # 834 restore guards, mirrored verbatim (probe_runner.py:722-739): flax
    # restore_checkpoint(step=None) returns the target UNCHANGED on a missing/empty
    # dir — assert the restore actually happened before emitting any numbers.
    fresh_leaves = jax.tree_util.tree_leaves(trainer.state.params)
    trainer.state = trainer.load_checkpoint(ckpt)
    step_count = int(trainer.state.step_count)
    if step_count <= 0:
        raise SystemExit(
            f"TASK_836: restored step_count={step_count} — a trained snapshot "
            f"must have step_count > 0. {ckpt!r} is not a trained checkpoint "
            "(or the restore was a silent no-op). Refusing to emit rankings.")
    restored_leaves = jax.tree_util.tree_leaves(trainer.state.params)
    if all(np.allclose(a, b) for a, b in zip(fresh_leaves, restored_leaves)):
        raise SystemExit(
            "TASK_836: restored params are IDENTICAL to the fresh init — the "
            f"flax restore from {ckpt!r} was a silent no-op. Refusing to emit "
            "rankings for a fresh-init model.")
    print(f"[ckpt] restored {ckpt} (step_count={step_count}; params differ "
          "from fresh init — restore verified)")

    panel = build_panel(cfg, n_assets, args.seed, db_override=args.db,
                        max_decisions=args.max_decisions, burn_in=args.burn_in)
    result = probe_core(model, trainer.state.params, cfg, panel, args,
                        checkpoint_label=ckpt, smoke=False,
                        restored_step_count=step_count)
    out_path = args.out or os.path.join(MODULE_DIR, "results_probe.json")
    result = _finish(result, out_path)
    print(f"[probe] {result['interpretation']['reading']}")
    return result


# =============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fixture-smoke", action="store_true",
                    help="fresh-init untrained model on the fixture db; exercises "
                         "the real save->load path end-to-end (no interpretation)")
    ap.add_argument("--checkpoint", default=None,
                    help="snapshot dir (jax_ckpt_fast/<name>/snapshot_ep<N>)")
    ap.add_argument("--config-json", default=None,
                    help="RUN TrainingConfig as JSON field overrides (required "
                         "with --checkpoint; 834 pattern)")
    ap.add_argument("--db", default=None, help="override config db_path")
    ap.add_argument("--n-assets", type=int, default=None,
                    help="run n_assets (checkpoint default 64, smoke default 16)")
    ap.add_argument("--max-decisions", type=int, default=None,
                    help="cap walk decisions (smoke default 96; checkpoint default "
                         "None = full timeline)")
    ap.add_argument("--burn-in", type=int, default=8,
                    help="walk decisions excluded from the ranked sample "
                         "(books accumulate; 834 default)")
    ap.add_argument("--seed", type=int, default=0,
                    help="walk seed (reset + proposal dither/sampling); multi-seed "
                         "claims = rerun with different --seed")
    ap.add_argument("--out", default=None, help="output JSON path")
    args = ap.parse_args()

    if args.fixture_smoke:
        args.n_assets = args.n_assets or 16
        if args.max_decisions is None:
            args.max_decisions = 96
        run_fixture_smoke(args)
    else:
        if not args.checkpoint or not args.config_json:
            ap.error("checkpoint mode needs --checkpoint AND --config-json "
                     "(or use --fixture-smoke)")
        args.n_assets = args.n_assets or 64
        run_checkpoint(args)


if __name__ == "__main__":
    main()
