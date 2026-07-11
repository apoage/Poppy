#!/usr/bin/env python3
"""TASK_835 / Stage-4 — faithful vector eval harness (checkpoint → corridor scorecard).

The running 828 Stage-3 battery has NO in-training held-out eval (vector mode fail-closed
skips the legacy evaluator — shipped panel finding). This harness is the instrument that
grades it: load a training checkpoint, run a DETERMINISTIC full-timeline rollout through
the REAL vector env + the SHIPPED Sampled-MuZero vector search, and score it against the
830/831 corridor conventions.

Fidelity contract (every piece mirrors the training stack — no reimplementation):
  * MODEL/TRAINER: built via ``TrainingConfig`` → ``setup_model_and_trainer`` (the exact
    training constructor), then ``trainer.init_train_state`` + ``trainer.load_checkpoint``
    (mcts_trader/jax_agent/trainer.py:1700; the train_jax_fast.py:443 load pattern).
    Run shapes (model size, n_assets, K, sims, hold-bias, …) come from CLI args that
    default to the 828 Stage-3 battery manifest values — NOT hardcoded in the loop.
  * PANEL: ``create_jax_env_historical`` (mcts_trader/training/env_factory.py) with the
    run's periods-file/db/bar-minutes — the SAME 1-min-table → ``[::bar_minutes]`` daily
    resample as training (the 832 panel-consistency rule). The row-timestamp axis is
    reconstructed with the loader's own queries (read-only mirror of
    ``VectorBTTradingEnv._load_database_prices``) and length-asserted against the panel.
  * DECISION LOOP: byte-level mirror of ``run_self_play_jax``'s vector branch
    (mcts_trader/training/self_play.py:640-705): leak-free obs window
    [t-window, t-1] via ``jit_batch_encode_simple`` (27ch), the same
    ``jit_batch_compute_action_mask``, the v1 book = HOLD everywhere (DEC-3), the
    shipped ``SampledMuZeroVectorSearch`` via ``create_sampled_vector_search`` with the
    train_jax_fast.py:375 parameters — EXCEPT ``deterministic=True`` (Stage-2 carry-note:
    eval selects the visit-count argmax, not the gumbel exploration pick). The reserved
    ``vector_gumbel_scale_eval`` knob stays untouched/unwired. Env step =
    ``jit_batch_step_vector`` (the real ``step_vector``). No exploration schedule, no
    replay buffer, pure eval pass.
  * DETERMINISM: single env; one fixed PRNGKey chain (``--eval-seed``) drives proposal
    sampling + search gumbel; ``deterministic=True`` argmax selection; env RNG seeded
    via the same key (and inert: probabilistic_fills False, burns off). Two calls with
    the same args are bit-identical.
  * COST TIERS (the 830/831 convention): gross (cost 0) and net@21 (fee16 + slip5) are
    SEPARATE rollouts — fees change the trajectory through the env (execution prices,
    cash, mask), so each tier is walked independently on the identical panel/seed and
    scored on its own realized return series.
  * SCORECARD: per-decision Sharpe via the audited ``trader.backtest._sharpe``
    annualization (mean/std·√(525600/period), period = --bar-minutes ⇒ 1440 canonical);
    full window + the four 831-EXT regime windows, whose definitions AND per-window
    floors/ceilings are LOADED from ``runs/831_ext/{bear22w,bull23w,y2024,recent}.json``
    (keys ``window``/``floor``/``ceiling``) — never hardcoded; position stats (pct_hold
    per asset + aggregate, avg long/short counts — the 821 HOLD-collapse counters,
    collapse signature = pct_hold pinned >99); final PV; corridor position vs the
    830/E0-REAL official corridor loaded from
    ``experiments/830_corridor/results_real_panel.json`` (net@21 floor +0.424 /
    engine ref +0.751 / ceiling +29.698 — pinned fallback constants below carry the
    same provenance if the file is absent). One LEDGER-convention row per run.

FACTORY / SCHEDULED split (830/831 house style): this module + the fixture smoke are the
factory deliverable. Real Stage-4 numbers (battery snapshots × seeds over the read-only
``trading_data.db`` panel) are a SCHEDULED run — never inside the factory loop. Default
db = the fixture. Applying the Stage-4 PASS verdict is the planner's job, not this file's.

Fixture smoke: ``python eval_vector.py --smoke`` — fresh-init UNTRAINED model on
``fixtures/trading_fixture.db``, CPU, end-to-end incl. a REAL ``save_checkpoint`` →
``load_checkpoint`` roundtrip (params proven different before load, identical after).
The fixture's minute table only covers 8 symbols × ~90 days (2025-09-13..2025-12-12),
so the smoke sizes its window/universe to that slice (pre-registered in SPEC.md).
PASS = runs clean + emits the FULL scorecard schema; untrained numbers are near-noise
by construction — no threshold applied (plumbing proof, not verdict).
"""
import os
# CPU-only by default, deterministic; keep JAX off the GPU (fixture smoke / scheduled job).
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import json
import sqlite3
import sys
from datetime import datetime, date

import numpy as np
import jax
import jax.numpy as jnp

# Run from anywhere: repo root on the path so `trader` / `mcts_trader` import cleanly.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from trader import backtest, config as trader_config
from mcts_trader.training.config import TrainingConfig
from mcts_trader.training.setup import setup_model_and_trainer
from mcts_trader.training.env_factory import create_jax_env_historical
from mcts_trader.jax_agent.sampled_search import create_sampled_vector_search
from mcts_trader.env.jax_trading_env import (
    jit_batch_step_vector, jit_batch_compute_action_mask,
    get_current_prices, compute_portfolio_value,
)
from mcts_trader.env.jax_encoder import EncoderConfig, jit_batch_encode_simple

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
FIXTURE_DB = os.path.join(_REPO, "fixtures", "trading_fixture.db")
CORRIDOR_JSON = os.path.join(_REPO, "experiments", "830_corridor", "results_real_panel.json")
EXT_WINDOW_NAMES = ("bear22w", "bull23w", "y2024", "recent")
EXT_WINDOW_DIR = os.path.join(_REPO, "runs", "831_ext")

GATE_COSTS = (0, 21)          # bp/side cost tiers — gross + net@21 (fee16+slip5), the 830/831 SPEC tiers
FEE_BPS, SLIP_BPS = 16.0, 5.0
HOLD = 1
N_ACTIONS_PER_ASSET = 5
HOLD_COLLAPSE_PCT = 99.0      # the 821 failure signature: pct_hold pinned >99

# 830/E0-REAL official corridor, net@21 [floor +0.424, ceil +29.698], engine ref +0.751
# (gross +0.753 / +0.956 / +31.367). Provenance: experiments/LEDGER.md row 830/E0-REAL +
# experiments/830_corridor/results_real_panel.json (loaded at runtime; these pinned
# constants are only the fallback if that file is absent, e.g. on a stripped host).
CORRIDOR_FALLBACK = {
    "21": {"floor_inenv": 0.4237793380446761, "reference": 0.7509887330376795,
           "ceiling": 29.69798287600358},
    "0":  {"floor_inenv": 0.7532467397811813, "reference": 0.9557594651716449,
           "ceiling": 31.36710570497202},
    "source": "pinned fallback (830/E0-REAL, experiments/LEDGER.md)",
}

# Fixture minute-table coverage (top-8 by row count, 2025-09-13..2025-12-12) — the
# pre-registered smoke universe (SPEC.md). The fixture's DAILY panel is full-history,
# but create_jax_env_historical reads the 1-min `ohlcv` table, so the smoke must live
# inside the minute slice.
SMOKE_SYMBOLS = ["XBT/USD", "ETH/USD", "LTC/USD", "XMR/USD",
                 "ZEC/USD", "ETC/USD", "DASH/USD", "REP/USD"]
SMOKE_START, SMOKE_END = "2025-09-13", "2025-12-12"
SMOKE_BAR_MINUTES = 60        # exercises the ::bar_minutes resample path; 447 rows > warm+decisions
SMOKE_MAX_DECISIONS = 40


# =============================================================================
# panel timestamp axis — read-only mirror of the training loader
# =============================================================================

def _to_epoch(v, default):
    """EXACT mirror of VectorBTTradingEnv._load_database_prices date parsing
    (local-timezone strptime → .timestamp(); digits pass through)."""
    if v is None:
        return default
    if isinstance(v, (int, float)):
        return int(v)
    s = str(v)
    if s.isdigit():
        return int(s)
    try:
        return int(datetime.strptime(s, "%Y-%m-%d").timestamp())
    except ValueError:
        try:
            return int(datetime.fromisoformat(s).timestamp())
        except Exception:
            return default


def load_row_timestamps(db_path, start_date, end_date, specific_symbols, n_assets, bar_minutes):
    """Reconstruct the panel's row-timestamp axis with the SAME queries the training
    loader runs (VectorBTTradingEnv._load_database_prices), then the SAME
    ``[::bar_minutes]`` stride as env_factory. Read-only; consistency is asserted by the
    caller (len must equal prices.shape[0]). Returns (row_ts int64 array, loaded_symbols)."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    s_ts = _to_epoch(start_date, 0)
    e_ts = _to_epoch(end_date, 9999999999)
    if specific_symbols:
        ph = ",".join("?" * len(specific_symbols))
        rows = cur.execute(
            f"SELECT a.id, a.symbol FROM assets a WHERE a.symbol IN ({ph})",
            specific_symbols).fetchall()
        sym_to_id = {r[1]: r[0] for r in rows}
        asset_ids, symbols = [], []
        for sym in specific_symbols:           # preserve order; skip missing (loader parity)
            if sym in sym_to_id:
                asset_ids.append(sym_to_id[sym])
                symbols.append(sym)
    else:
        rows = cur.execute(
            """SELECT a.id, a.symbol, COUNT(*) as cnt FROM ohlcv o
               JOIN assets a ON o.asset_id = a.id
               WHERE o.timestamp >= ? AND o.timestamp <= ?
               GROUP BY a.id ORDER BY cnt DESC LIMIT ?""",
            (s_ts, e_ts, n_assets)).fetchall()
        asset_ids = [r[0] for r in rows]
        symbols = [r[1] for r in rows]
    if not asset_ids:
        conn.close()
        raise RuntimeError(f"no assets resolved in {db_path} for the requested universe")
    ph = ",".join("?" * len(asset_ids))
    ts = [r[0] for r in cur.execute(
        f"""SELECT DISTINCT timestamp FROM ohlcv
            WHERE asset_id IN ({ph}) AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp""",
        (*asset_ids, s_ts, e_ts)).fetchall()]
    conn.close()
    row_ts = np.asarray(ts, dtype=np.int64)
    if bar_minutes > 1:                        # env_factory.py:369 stride, verbatim
        row_ts = row_ts[::bar_minutes]
    return row_ts, symbols


# =============================================================================
# scorecard conventions — corridor constants + 831-EXT windows (LOADED, not hardcoded)
# =============================================================================

def load_corridor_constants():
    """Official corridor from 830/E0-REAL results JSON; pinned fallback if absent."""
    if os.path.exists(CORRIDOR_JSON):
        with open(CORRIDOR_JSON) as fh:
            r = json.load(fh)
        out = {c: {k: r["corridor"][c][k] for k in ("floor_inenv", "reference", "ceiling")}
               for c in ("0", "21")}
        out["source"] = os.path.relpath(CORRIDOR_JSON, _REPO)
        return out
    return dict(CORRIDOR_FALLBACK)


def load_ext_windows():
    """The four 831-EXT regime windows: definitions + per-window floors/ceilings from
    runs/831_ext/<name>.json (keys ``window``, ``floor``, ``ceiling``). Missing file →
    None entry (schema key still emitted)."""
    out = {}
    for name in EXT_WINDOW_NAMES:
        path = os.path.join(EXT_WINDOW_DIR, f"{name}.json")
        if not os.path.exists(path):
            out[name] = None
            continue
        with open(path) as fh:
            r = json.load(fh)
        w0, w1 = r["window"]
        out[name] = {
            "window": [w0, w1],
            "start_ts": _to_epoch(w0, 0),
            "end_ts": _to_epoch(w1, 9999999999),
            "floor": {str(k): v for k, v in r["floor"].items()},
            "ceiling": {str(k): v for k, v in r["ceiling"].items()},
            "source": os.path.relpath(path, _REPO),
        }
    return out


def _sharpe(returns, period_minutes):
    """The audited 819/backtest annualization: mean/std·√(525600/period)."""
    return backtest._sharpe(np.asarray(returns, dtype=np.float64), float(period_minutes))


def _f(x):
    """JSON-safe float (None for non-finite)."""
    if x is None:
        return None
    x = float(x)
    return x if np.isfinite(x) else None


def _display_path(p):
    """Stable provenance for emitted artifacts: repo-relative when inside the repo,
    absolute otherwise (committed smoke artifacts must not embed ephemeral tmp paths)."""
    p = os.path.abspath(p)
    rel = os.path.relpath(p, _REPO)
    return p if rel.startswith("..") else rel


# =============================================================================
# checkpoint load — the TASK_834 pattern (run's TrainingConfig → trainer.load_checkpoint)
# =============================================================================

def build_model_and_trainer(args, n_assets, init_seed):
    """Build model+trainer EXACTLY as training does: TrainingConfig →
    setup_model_and_trainer → init_train_state(dummy obs). Shapes come from args."""
    cfg = TrainingConfig(
        name="835_stage4_eval",
        model_size=args.model_size,
        h_cycles=args.h_cycles,
        l_cycles=args.l_cycles,
        sims=args.sims,
        discount=args.discount,
        gumbel_scale=args.gumbel_scale,
        value_scale=args.value_scale,
        value_shift=args.value_shift,
        vector_action=True,
        vector_k=args.vector_k,
        vector_hold_bias=args.vector_hold_bias,
        vector_dither_scale=args.vector_dither_scale,
        vector_proposal_scheme=args.vector_proposal_scheme,
        # TASK_839: carried for run self-description (env-only flag; model build ignores it).
        vector_budget_step=args.vector_budget_step,
        vector_budget_pct=args.vector_budget_pct,
        # TASK_840: carried for run self-description (env-only flag; model build ignores it).
        vector_weight_rebalance=args.vector_weight_rebalance,
        vector_weight_unit=args.vector_weight_unit,
        vector_weight_max=args.vector_weight_max,
        vector_rebalance_deadband=args.vector_rebalance_deadband,
        seed=init_seed,
    )
    model, trainer = setup_model_and_trainer(cfg, n_assets)
    dummy_obs = jnp.ones((1, 27, n_assets, n_assets))         # train_jax_fast.py:439 pattern
    trainer.init_train_state(jax.random.PRNGKey(init_seed), dummy_obs)
    return model, trainer


def load_checkpoint_into(trainer, ckpt_path):
    """train_jax_fast.py:443 load pattern: load_checkpoint then assign trainer.state."""
    abspath = os.path.abspath(ckpt_path)                      # orbax requires absolute paths
    loaded = trainer.load_checkpoint(abspath)
    if loaded is None:
        raise RuntimeError(f"trainer.load_checkpoint returned None for {abspath}")
    trainer.state = loaded
    return trainer


def build_search(model, args, n_assets):
    """Mirror of train_jax_fast.py:375-385 — the shipped Stage-2/3 search operator.
    ``vector_gumbel_scale_eval`` is reserved/unwired upstream and is NOT touched here."""
    return create_sampled_vector_search(
        model, n_assets=n_assets, n_actions_per_asset=N_ACTIONS_PER_ASSET,
        num_samples=args.vector_k, num_simulations=args.sims,
        max_num_considered_actions=min(args.vector_k, 16), discount=args.discount,
        gumbel_scale=args.gumbel_scale, value_scale=args.value_scale,
        value_shift=args.value_shift,
        hold_bias=args.vector_hold_bias,
        dither_scale=args.vector_dither_scale,
        proposal_scheme=args.vector_proposal_scheme,
        # TASK_838: eval operator must match the training operator — a checkpoint
        # trained with --vector-per-node-resample must be evaluated with the SAME
        # flag or the measurement is train/eval-asymmetric (default OFF = canon).
        per_node_resampling=args.vector_per_node_resample,
    )


def _trees_allclose(a, b):
    la, lb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    if len(la) != len(lb):
        return False
    return all(np.asarray(x).shape == np.asarray(y).shape and
               np.allclose(np.asarray(x), np.asarray(y)) for x, y in zip(la, lb))


# =============================================================================
# deterministic full-timeline rollout (mirror of run_self_play_jax's vector branch)
# =============================================================================

def rollout(params, search_obj, prices, env_cfg, state0, n_assets, n_decisions,
            eval_seed, max_positions, min_cash_threshold):
    """Walk the full timeline once: obs → mask → deterministic vector search → step_vector.
    Returns per-decision portfolio returns + the HOLD-collapse / position diagnostics."""
    enc_cfg = EncoderConfig(n_assets=n_assets, window_size=env_cfg.window_size,
                            n_channels=27)                    # self_play.py:627 (27ch)
    book = jnp.ones((1, n_assets), dtype=jnp.int32)           # v1 book = HOLD (DEC-3)

    @jax.jit
    def _pv_fn(state):
        # 830 accounting convention: mark PV at the row the action just executed on
        # (current_step was advanced by step_vector, hence -1).
        cur = get_current_prices(prices, state.current_step - 1, state.episode_start_idx)
        return jax.vmap(compute_portfolio_value)(state, cur)

    initial_cash = float(env_cfg.initial_cash)
    state = state0
    key = jax.random.PRNGKey(eval_seed)
    pvs = np.empty(n_decisions + 1, dtype=np.float64)
    pvs[0] = initial_cash
    hold_count = np.zeros(n_assets, dtype=np.int64)
    action_hist = np.zeros(N_ACTIONS_PER_ASSET, dtype=np.int64)
    n_long_total = n_short_total = 0
    bankrupt_at = terminated_early_at = -1
    n_exec = 0
    for k in range(n_decisions):
        # A. leak-free obs window [t-window, t-1] + legal mask (self_play.py:645-668)
        price_idx = state.episode_start_idx + state.current_step          # (1,)
        window_starts = price_idx - enc_cfg.window_size
        obs = jit_batch_encode_simple(prices, window_starts, enc_cfg)
        current_prices = prices[price_idx]                                # (1, A)
        mask = jit_batch_compute_action_mask(
            state, current_prices, env_cfg, max_positions, min_cash_threshold)
        # B. deterministic Sampled-MuZero vector search (visit-argmax selection)
        key, step_rng = jax.random.split(key)
        action_vec, _sampled, _visit_probs, _stats = search_obj(
            params, step_rng, obs, mask, book, True)          # deterministic=True (eval)
        # C. the REAL vector env step
        state, _, _, done = jit_batch_step_vector(state, action_vec, prices, env_cfg)
        pv_arr = _pv_fn(state)
        jax.block_until_ready(pv_arr)                         # device→host boundary
        pv = float(np.asarray(pv_arr)[0])
        av = np.asarray(action_vec)[0]
        sd = np.asarray(state.pos_side)[0]
        pvs[k + 1] = pv
        hold_count += (av == HOLD)
        action_hist += np.bincount(av, minlength=N_ACTIONS_PER_ASSET)
        n_long_total += int((sd == 1).sum())
        n_short_total += int((sd == 2).sum())
        n_exec += 1
        if bankrupt_at < 0 and pv <= initial_cash * env_cfg.liquidation_threshold_pct:
            bankrupt_at = k
        if bool(np.asarray(done)[0]):
            terminated_early_at = k
            break
    pvs = pvs[:n_exec + 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        rets = pvs[1:] / np.maximum(pvs[:-1], 1e-8) - 1.0
    rets = np.where(np.isfinite(rets), rets, 0.0)
    denom = max(n_exec, 1)
    return {
        "rets": rets, "n_exec": n_exec, "final_pv": float(pvs[-1]),
        "pct_hold_per_asset": 100.0 * hold_count / denom,
        "pct_hold_agg": 100.0 * float(hold_count.sum()) / (denom * n_assets),
        "action_hist_pct": (100.0 * action_hist / max(action_hist.sum(), 1)).tolist(),
        "avg_long": n_long_total / denom, "avg_short": n_short_total / denom,
        "bankrupt_at": bankrupt_at, "terminated_early_at": terminated_early_at,
    }


def score_tier(rep, ts_dec, ext_windows, period_minutes, tier, col_symbols, steps_expected):
    """Full + per-831-EXT-window Sharpe for ONE cost tier's realized return series.
    Records steps_completed vs steps_expected — a rollout that terminated early (env
    done / bankruptcy) is flagged ``partial`` and the caller withholds corridor booleans."""
    rets = rep["rets"][:len(ts_dec)]
    per_window = {}
    for name, wd in ext_windows.items():
        if wd is None:
            per_window[name] = {"window": None, "n": 0, "sharpe": None,
                                "floor": None, "ceiling": None, "above_floor": None,
                                "note": "831-EXT window file missing"}
            continue
        sel = (ts_dec >= wd["start_ts"]) & (ts_dec <= wd["end_ts"])
        n = int(sel.sum())
        sh = _f(_sharpe(rets[sel], period_minutes)) if n >= 2 else None
        floor = _f(wd["floor"].get(str(tier)))
        ceil = _f(wd["ceiling"].get(str(tier)))
        per_window[name] = {
            "window": wd["window"], "n": n, "sharpe": sh,
            "floor": floor, "ceiling": ceil,
            "above_floor": (None if sh is None or floor is None else bool(sh >= floor)),
        }
    return {
        "sharpe_full": _f(_sharpe(rets, period_minutes)),
        "steps_completed": rep["n_exec"],
        "steps_expected": int(steps_expected),
        "partial": bool(rep["n_exec"] < steps_expected),
        "final_pv": rep["final_pv"],
        "avg_long": rep["avg_long"], "avg_short": rep["avg_short"],
        "pct_hold_agg": rep["pct_hold_agg"],
        "pct_hold_per_asset": {sym: float(p) for sym, p
                               in zip(col_symbols, rep["pct_hold_per_asset"])},
        "action_hist_pct": rep["action_hist_pct"],
        "bankrupt_at": rep["bankrupt_at"],
        "terminated_early_at": rep["terminated_early_at"],
        "per_window": per_window,
    }


# =============================================================================
# per-checkpoint evaluation
# =============================================================================

def evaluate_checkpoint(label, params, search_obj, panel, args, corridor, ext_windows):
    prices, env_cfg0, state_init, row_ts, col_symbols, panel_info = panel
    warm = int(env_cfg0.window_size)
    n_decisions = panel_info["n_decisions"]
    period_minutes = float(args.bar_minutes)
    ts_dec = row_ts[warm:warm + n_decisions]

    tiers = {}
    for c in GATE_COSTS:
        fee, slip = (0.0, 0.0) if c == 0 else (FEE_BPS, SLIP_BPS)
        # cost tier enters via the env config ONLY; identical panel + seed per tier
        # (the 830/831 separate-rollout convention — fee-aware trajectories differ).
        env_cfg = env_cfg0.replace(fee_bps=fee, slippage_bps=slip)
        rep = rollout(params, search_obj, prices, env_cfg, state_init,
                      env_cfg0.n_assets, n_decisions, args.eval_seed,
                      args.max_positions, args.min_cash_threshold)
        tiers[str(c)] = score_tier(rep, ts_dec[:rep["n_exec"]], ext_windows,
                                   period_minutes, c, col_symbols, n_decisions)

    net, gross = tiers["21"], tiers["0"]
    # Completeness guard: an early-terminated walk (env done / bankruptcy) must not emit
    # corridor PASS booleans computed on a partial timeline — withhold them (null) and
    # mark the scorecard loudly; the raw numbers stay visible for diagnosis.
    partial = bool(net["partial"] or gross["partial"])
    fl, rf, ce = (_f(corridor["21"][k]) for k in ("floor_inenv", "reference", "ceiling"))
    agent_net = net["sharpe_full"]
    span = None if (fl is None or ce is None) else max(ce - fl, 1e-10)
    corridor_block = {
        "floor_net21": fl, "reference_net21": rf, "ceiling_net21": ce,
        "floor_gross": _f(corridor["0"]["floor_inenv"]),
        "reference_gross": _f(corridor["0"]["reference"]),
        "ceiling_gross": _f(corridor["0"]["ceiling"]),
        "source": corridor["source"],
        "agent_net21": agent_net, "agent_gross": gross["sharpe_full"],
        "partial_rollout": partial,
        "position_in_corridor_net21": (None if partial or agent_net is None or span is None
                                       else float((agent_net - fl) / span)),
        "beats_floor_net21": (None if partial or agent_net is None or fl is None
                              else bool(agent_net >= fl)),
        "beats_reference_net21": (None if partial or agent_net is None or rf is None
                                  else bool(agent_net >= rf)),
        "hold_collapse": bool(net["pct_hold_agg"] > HOLD_COLLAPSE_PCT),
    }

    result = {
        "tag": args.tag, "checkpoint": label["path"], "checkpoint_label": label["name"],
        "db": args.db, "window": [args.start, args.end], "bar_minutes": args.bar_minutes,
        "eval_seed": args.eval_seed,
        "panel": panel_info,
        "operator": {
            "vector_k": args.vector_k, "sims": args.sims,
            "hold_bias": args.vector_hold_bias, "dither_scale": args.vector_dither_scale,
            "proposal_scheme": args.vector_proposal_scheme,
            "per_node_resampling": args.vector_per_node_resample,  # TASK_838
            "vector_budget_step": args.vector_budget_step,   # TASK_839
            "vector_budget_pct": args.vector_budget_pct,     # TASK_839
            "vector_weight_rebalance": args.vector_weight_rebalance,     # TASK_840
            "vector_weight_unit": args.vector_weight_unit,              # TASK_840
            "vector_weight_max": args.vector_weight_max,                # TASK_840
            "vector_rebalance_deadband": args.vector_rebalance_deadband,  # TASK_840
            "gumbel_scale": args.gumbel_scale,
            "max_num_considered_actions": min(args.vector_k, 16),
            "deterministic": True, "selection": "visit_probs argmax",
            "model_size": args.model_size, "h_cycles": args.h_cycles,
            "l_cycles": args.l_cycles, "discount": args.discount,
        },
        "cost_tiers_bp": list(GATE_COSTS),
        "annualization_period_minutes": period_minutes,
        "tiers": tiers,
        "corridor": corridor_block,
    }
    # ledger_row is rendered by the caller once the ACTUAL output path is known
    # (out_pointer) — a hardcoded pointer here would dangle for scheduled runs.
    return result


def _fmt(x):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:+.3f}"


def make_ledger_row(r):
    """One experiments/LEDGER.md-convention row: | ID | Date | H# | Question | Verdict |
    Corridor | Pointer |. Verdict text carries numbers only — applying the Stage-4 PASS
    is the planner's job (out of scope here). Pointer = r['out_pointer'], derived by the
    caller from the ACTUAL out_dir (never a hardcoded directory — scheduled runs write to
    runs/835_stage4/, the fixture smoke to this experiment dir; a stale pointer is the
    documented 734-819 audit failure mode)."""
    net, gross, cor = r["tiers"]["21"], r["tiers"]["0"], r["corridor"]
    regimes = ", ".join(
        f"{name} {_fmt(net['per_window'][name]['sharpe'])}"
        f"(fl {_fmt(net['per_window'][name]['floor'])})"
        for name in EXT_WINDOW_NAMES)
    partial_note = ""
    if cor["partial_rollout"]:
        partial_note = (
            f"PARTIAL ROLLOUT — terminated early (net {net['steps_completed']}/"
            f"{net['steps_expected']}, gross {gross['steps_completed']}/"
            f"{gross['steps_expected']} decisions); corridor booleans withheld. ")
    collapse = ("HOLD-COLLAPSE (pct_hold "
                f"{net['pct_hold_agg']:.1f}%)" if cor["hold_collapse"]
                else f"no HOLD collapse (pct_hold {net['pct_hold_agg']:.1f}%)")
    smoke_note = (" FIXTURE-SMOKE — plumbing proof, untrained near-noise, no threshold applied;"
                  " real Stage-4 numbers come from the scheduled run." if r["tag"].startswith("fixture")
                  else " Scheduled-run numbers; PASS verdict applied by the planner (gate ladder).")
    return (
        f"| 835/S4 | {date.today().isoformat()} | H1 (learned) | "
        f"Stage-4 eval {r['checkpoint_label']}: deterministic full-timeline vector rollout "
        f"({r['window'][0]}..{r['window'][1]}, bar_minutes={r['bar_minutes']}, "
        f"seed {r['eval_seed']}) vs the corridor | "
        f"{partial_note}"
        f"net@21 {_fmt(net['sharpe_full'])} · gross {_fmt(gross['sharpe_full'])}; "
        f"final PV {net['final_pv']:.0f}; avgL {net['avg_long']:.1f} avgS {net['avg_short']:.1f}; "
        f"{collapse}; regimes net@21: {regimes}.{smoke_note} | "
        f"agent {_fmt(cor['agent_net21'])} in [{_fmt(cor['floor_net21'])}, "
        f"{_fmt(cor['ceiling_net21'])}] net@21 (ref {_fmt(cor['reference_net21'])}) | "
        f"{r['out_pointer']} |"
    )


# =============================================================================
# panel construction (once per run; shared by every checkpoint + tier)
# =============================================================================

def build_panel(args):
    """The training panel, via the training factory itself. Gross config returned;
    tiers derive from it via .replace(fee/slip) — prices identical by construction."""
    states0, prices, env_cfg = create_jax_env_historical(
        db_path=args.db,
        start_date=args.start,
        end_date=args.end,
        n_envs=1,
        n_assets=args.n_assets,
        specific_symbols=args.symbols,
        fee_bps=0.0,
        slippage_bps=0.0,
        max_episode_steps=400,          # inert: episode bounds overridden below
        bar_minutes=args.bar_minutes,
        vector_action=True,
        # TASK_839: eval ENV must use the SAME step-sizing the checkpoint trained under.
        vector_budget_step=args.vector_budget_step,
        vector_budget_pct=args.vector_budget_pct,
        # TASK_840: eval ENV must use the SAME weight-rebalance mode the checkpoint trained under.
        vector_weight_rebalance=args.vector_weight_rebalance,
        vector_weight_unit=args.vector_weight_unit,
        vector_weight_max=args.vector_weight_max,
        vector_rebalance_deadband=args.vector_rebalance_deadband,
        rng_key=jax.random.PRNGKey(args.eval_seed),
    )
    row_ts, loaded_syms = load_row_timestamps(
        args.db, args.start, args.end, args.symbols, args.n_assets, args.bar_minutes)
    T = int(prices.shape[0])
    if len(row_ts) != T:
        raise RuntimeError(
            f"panel-consistency violation: reconstructed timestamp axis has {len(row_ts)} "
            f"rows but create_jax_env_historical produced {T} — loader mirror out of sync")
    warm = int(env_cfg.window_size)
    n_max = T - 1 - warm
    n_decisions = n_max if args.max_decisions is None else min(n_max, args.max_decisions)
    if n_decisions < 2:
        raise RuntimeError(f"panel too short: T={T}, warm={warm} → n_decisions={n_decisions}")
    # Deterministic full-timeline episode: fixed start at the first obs-valid row.
    state_init = states0.replace(
        episode_start_idx=jnp.full_like(states0.episode_start_idx, warm),
        current_step=jnp.zeros_like(states0.current_step),
        max_episode_steps=jnp.full_like(states0.max_episode_steps, n_decisions + 2),
        initial_portfolio_value=jnp.full_like(states0.initial_portfolio_value,
                                              env_cfg.initial_cash),
        prev_portfolio_value=jnp.full_like(states0.prev_portfolio_value,
                                           env_cfg.initial_cash),
    )
    n_loaded = len(loaded_syms)
    padded = n_loaded < args.n_assets
    col_symbols = [loaded_syms[i % n_loaded] if i < n_loaded
                   else f"{loaded_syms[i % n_loaded]}#pad{i}"
                   for i in range(args.n_assets)]
    spacing = (float(np.median(np.diff(row_ts))) / 60.0) if len(row_ts) > 1 else None
    panel_info = {
        "n_rows": T, "n_assets": int(env_cfg.n_assets), "warm": warm,
        "n_decisions": n_decisions, "n_loaded_symbols": n_loaded, "padded": padded,
        "position_size_pct": float(env_cfg.position_size_pct),
        "row_spacing_minutes_median": spacing,
    }
    return prices, env_cfg, state_init, row_ts, col_symbols, panel_info


# =============================================================================
# driver
# =============================================================================

def _ckpt_label(path):
    parts = [p for p in os.path.normpath(path).split(os.sep) if p]
    name = "__".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
    return {"path": _display_path(path), "name": name}


def print_summary(r):
    net, gross, cor = r["tiers"]["21"], r["tiers"]["0"], r["corridor"]
    print("=" * 84)
    print(f"STAGE-4 EVAL ({r['tag']}) — {r['checkpoint_label']}")
    print(f"panel: {r['panel']['n_assets']} assets × {r['panel']['n_rows']} rows "
          f"(bar_minutes={r['bar_minutes']}), warm={r['panel']['warm']}, "
          f"decisions={r['panel']['n_decisions']}, eval_seed={r['eval_seed']}")
    print(f"operator: dithered-book K={r['operator']['vector_k']} sims={r['operator']['sims']} "
          f"hold_bias={r['operator']['hold_bias']} deterministic(visit-argmax)")
    if cor["partial_rollout"]:
        print("*" * 84)
        print("*** PARTIAL ROLLOUT — a tier terminated early (env done / bankruptcy); "
              "corridor booleans WITHHELD ***")
        print("*" * 84)
    print("-" * 84)
    print(f"{'tier':>8} | {'Sharpe':>8} {'finalPV':>10} {'pct_hold':>9} "
          f"{'avgL':>6} {'avgS':>6} {'bankrupt':>9} {'steps':>10}")
    for c in GATE_COSTS:
        t = r["tiers"][str(c)]
        lab = "gross" if c == 0 else f"net@{c}"
        bk = "no" if t["bankrupt_at"] < 0 else f"YES@{t['bankrupt_at']}"
        steps = f"{t['steps_completed']}/{t['steps_expected']}" + \
                (" PART" if t["partial"] else "")
        print(f"{lab:>8} | {_fmt(t['sharpe_full']):>8} {t['final_pv']:>10.0f} "
              f"{t['pct_hold_agg']:>8.1f}% {t['avg_long']:>6.1f} {t['avg_short']:>6.1f} "
              f"{bk:>9} {steps:>10}")
    print("-" * 84)
    print("831-EXT regime windows (net@21):")
    for name in EXT_WINDOW_NAMES:
        w = net["per_window"][name]
        win = f"{w['window'][0]}..{w['window'][1]}" if w["window"] else "n/a"
        hit = {True: "YES", False: "no", None: "n/a"}[w["above_floor"]]
        print(f"  {name:>8} [{win}] n={w['n']:>4}: sharpe {_fmt(w['sharpe'])} "
              f"floor {_fmt(w['floor'])} ceil {_fmt(w['ceiling'])} >=floor? {hit}")
    print("-" * 84)
    print(f"corridor net@21 [{_fmt(cor['floor_net21'])}, {_fmt(cor['ceiling_net21'])}] "
          f"ref {_fmt(cor['reference_net21'])} ({cor['source']})")
    print(f"agent net@21 {_fmt(cor['agent_net21'])} → position "
          f"{_fmt(cor['position_in_corridor_net21'])}; beats floor: {cor['beats_floor_net21']}; "
          f"beats ref: {cor['beats_reference_net21']}; HOLD collapse: {cor['hold_collapse']}"
          + ("  [PARTIAL — booleans withheld]" if cor["partial_rollout"] else ""))
    print("=" * 84)
    print("LEDGER row:")
    print(r["ledger_row"])
    print("=" * 84)


def print_comparison(results):
    print("=" * 100)
    print(f"{'checkpoint':>40} | {'net@21':>8} {'gross':>8} | {'pct_hold':>8} "
          f"{'>=floor':>7} {'>=ref':>5} | {'bear22w':>8} {'bull23w':>8} {'y2024':>8} {'recent':>8}")
    for r in results:
        net, cor = r["tiers"]["21"], r["corridor"]
        pw = net["per_window"]
        flag = {True: "YES", False: "no", None: "n/a"}
        print(f"{r['checkpoint_label'][:40]:>40} | {_fmt(net['sharpe_full']):>8} "
              f"{_fmt(r['tiers']['0']['sharpe_full']):>8} | {net['pct_hold_agg']:>7.1f}% "
              f"{flag[cor['beats_floor_net21']]:>7} {flag[cor['beats_reference_net21']]:>5} | "
              + " ".join(f"{_fmt(pw[n]['sharpe']):>8}" for n in EXT_WINDOW_NAMES))
    print("=" * 100)


def run_smoke(args, corridor, ext_windows):
    """Fresh-init untrained model + REAL save→load checkpoint roundtrip, then the full
    scorecard end-to-end on the fixture (plumbing proof — no thresholds)."""
    print("[smoke] building fresh-init model/trainer (seed 0) ...")
    model_a, trainer_a = build_model_and_trainer(args, args.n_assets, init_seed=0)
    # Module-local, gitignored (see .gitignore: smoke_ckpt*/) — stable provenance in the
    # committed smoke artifact instead of an ephemeral tmp path.
    ckpt_dir = os.path.abspath(args.smoke_ckpt_dir or os.path.join(EXP_DIR, "smoke_ckpt"))
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"[smoke] save_checkpoint → {ckpt_dir}")
    trainer_a.save_checkpoint(ckpt_dir)

    print("[smoke] building second fresh-init trainer (seed 1) + load_checkpoint ...")
    model_b, trainer_b = build_model_and_trainer(args, args.n_assets, init_seed=1)
    differ_before = not _trees_allclose(trainer_a.state.params, trainer_b.state.params)
    load_checkpoint_into(trainer_b, ckpt_dir)
    match_after = _trees_allclose(trainer_a.state.params, trainer_b.state.params)
    roundtrip = {"ckpt_dir": _display_path(ckpt_dir),
                 "params_differ_before_load": bool(differ_before),
                 "params_match_after_load": bool(match_after)}
    print(f"[smoke] roundtrip: differ_before={differ_before} match_after={match_after}")
    if not (differ_before and match_after):
        raise RuntimeError(f"checkpoint roundtrip FAILED: {roundtrip}")

    panel = build_panel(args)
    search_obj = build_search(model_b, args, args.n_assets)
    label = {"path": _display_path(ckpt_dir),
             "name": "untrained_fresh_init(save->load roundtrip)"}
    result = evaluate_checkpoint(label, trainer_b.state.params, search_obj,
                                 panel, args, corridor, ext_windows)
    result["roundtrip"] = roundtrip
    return result


def main():
    ap = argparse.ArgumentParser(
        description="Stage-4 faithful vector eval harness (TASK_835)")
    ap.add_argument("--db", default=FIXTURE_DB,
                    help="db path (default: fixture; the scheduled run points at trading_data.db)")
    ap.add_argument("--periods-file", default=None,
                    help="training periods JSON (periods[0] start/end/assets — the "
                         "train_jax_fast.py:223 pattern; e.g. "
                         "experiments/821_planner_daily/period_821_train.json)")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--symbols", default=None,
                    help="comma-separated symbol list (overrides periods-file assets)")
    ap.add_argument("--bar-minutes", type=int, default=1440,
                    help="1-min → K-min resample stride (1440 = the battery's daily bars)")
    ap.add_argument("--n-assets", type=int, default=64)
    ap.add_argument("--checkpoint", action="append", default=[],
                    help="checkpoint dir (repeatable; jax_ckpt_fast/<name>/snapshot_ep<N>/ or final)")
    ap.add_argument("--checkpoint-list", default=None,
                    help="file with one checkpoint path per line (comparison-table mode)")
    ap.add_argument("--smoke", action="store_true",
                    help="fixture smoke: fresh-init untrained model + save->load roundtrip")
    ap.add_argument("--smoke-ckpt-dir", default=None,
                    help="where the smoke writes its roundtrip checkpoint "
                         "(default: <this dir>/smoke_ckpt — gitignored, stable provenance)")
    ap.add_argument("--max-decisions", type=int, default=None,
                    help="cap decisions (smoke); None/negative = full timeline")
    ap.add_argument("--eval-seed", type=int, default=0)
    # ---- run shapes: defaults = the 828 Stage-3 battery manifest (jobs.manifest) ----
    ap.add_argument("--model-size", default="small")
    ap.add_argument("--h-cycles", type=int, default=4)
    ap.add_argument("--l-cycles", type=int, default=6)
    ap.add_argument("--sims", type=int, default=32)
    ap.add_argument("--vector-k", type=int, default=16)
    ap.add_argument("--vector-hold-bias", type=float, default=3.0,
                    help="battery value 3.0 (TrainingConfig CLI default 0.0 is the known trap)")
    ap.add_argument("--vector-dither-scale", type=float, default=1.0)
    ap.add_argument("--vector-proposal-scheme", default="dithered-book")
    ap.add_argument("--vector-per-node-resample", action="store_true",
                    dest="vector_per_node_resample",
                    help="TASK_838: per-node action resampling in the search operator — "
                         "MUST match the flag the checkpoint was trained with "
                         "(default OFF = root-reuse, the shipped operator)")
    ap.add_argument("--vector-budget-step", action="store_true",
                    dest="vector_budget_step",
                    help="TASK_839: budget-constrained vector-step sizing in the eval ENV — "
                         "MUST match the step the checkpoint was trained under (a budget-ON "
                         "checkpoint scored under a budget-OFF env silently invalidates the "
                         "measurement); default OFF = canon single-snapshot sizing")
    ap.add_argument("--vector-budget-pct", type=float, default=0.99,
                    dest="vector_budget_pct",
                    help="TASK_839: cash-budget fraction split across simultaneous opening "
                         "trades when --vector-budget-step is set (default 0.99)")
    ap.add_argument("--vector-weight-rebalance", action="store_true",
                    dest="vector_weight_rebalance",
                    help="TASK_840: target-weight rebalance step in the eval ENV — MUST match "
                         "the step the checkpoint was trained under (a weight-ON checkpoint "
                         "scored under a weight-OFF env silently invalidates the measurement); "
                         "default OFF = canon")
    ap.add_argument("--vector-weight-unit", type=float, default=-1.0,
                    dest="vector_weight_unit",
                    help="TASK_840: per-Buy/Sell target weight step (sentinel <=0 => 1/n_assets)")
    ap.add_argument("--vector-weight-max", type=float, default=4.0,
                    dest="vector_weight_max",
                    help="TASK_840: per-asset weight cap in UNITS of w_unit (default 4)")
    ap.add_argument("--vector-rebalance-deadband", type=float, default=-1.0,
                    dest="vector_rebalance_deadband",
                    help="TASK_840: re-trim only when |target-current| exceeds this "
                         "(sentinel <0 => 0.1*w_unit)")
    ap.add_argument("--discount", type=float, default=0.997)
    ap.add_argument("--gumbel-scale", type=float, default=3.0)
    ap.add_argument("--value-scale", type=float, default=1.0)
    ap.add_argument("--value-shift", type=float, default=0.0)
    ap.add_argument("--max-positions", type=int, default=16)
    ap.add_argument("--min-cash-threshold", type=float, default=100.0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out-dir", default=None,
                    help="output dir (default: this experiment dir for fixture runs, "
                         "runs/835_stage4 otherwise)")
    args = ap.parse_args()

    if args.max_decisions is not None and args.max_decisions < 0:
        args.max_decisions = None
    if args.symbols:
        args.symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if args.periods_file:
        with open(args.periods_file) as fh:
            first = json.load(fh)["periods"][0]              # train_jax_fast.py:227 pattern
        args.start = args.start or first["start_date"]
        args.end = args.end or first["end_date"]
        if args.symbols is None:
            args.symbols = first.get("assets")

    is_fixture = os.path.abspath(args.db) == os.path.abspath(FIXTURE_DB)
    if args.smoke:
        # pre-registered smoke config (SPEC.md): fixture minute-slice universe/window
        args.db = FIXTURE_DB
        args.symbols = args.symbols or SMOKE_SYMBOLS
        args.start = args.start or SMOKE_START
        args.end = args.end or SMOKE_END
        if args.bar_minutes == 1440:
            args.bar_minutes = SMOKE_BAR_MINUTES
        if args.n_assets == 64:
            args.n_assets = len(args.symbols)
        if args.max_decisions is None:
            args.max_decisions = SMOKE_MAX_DECISIONS
        is_fixture = True
    args.tag = args.tag or ("fixture-smoke" if (args.smoke and is_fixture)
                            else "fixture" if is_fixture else "real-panel")
    out_dir = args.out_dir or (EXP_DIR if is_fixture
                               else os.path.join(_REPO, "runs", "835_stage4"))
    os.makedirs(out_dir, exist_ok=True)

    corridor = load_corridor_constants()
    ext_windows = load_ext_windows()

    if args.smoke:
        result = run_smoke(args, corridor, ext_windows)
        out_name = "results_fixture_smoke.json"
        result["out_name"] = out_name
        # LEDGER pointer derived from the ACTUAL output path (MINOR-1 fix: a hardcoded
        # directory dangles as soon as out_dir differs — the 734-819 stale-pointer hazard).
        result["out_pointer"] = os.path.relpath(os.path.join(out_dir, out_name), _REPO)
        result["ledger_row"] = make_ledger_row(result)
        print_summary(result)
        out_path = os.path.join(out_dir, out_name)
        with open(out_path, "w") as fh:
            json.dump(result, fh, indent=2, default=float)
        print(f"\n[835] wrote {out_path}")
        return

    ckpts = list(args.checkpoint)
    if args.checkpoint_list:
        with open(args.checkpoint_list) as fh:
            ckpts += [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]
    if not ckpts:
        ap.error("no checkpoint given (use --checkpoint/--checkpoint-list, or --smoke)")

    panel = build_panel(args)
    results = []
    for path in ckpts:
        label = _ckpt_label(path)
        print(f"\n[835] evaluating {label['name']} ...")
        model, trainer = build_model_and_trainer(args, args.n_assets, init_seed=0)
        load_checkpoint_into(trainer, path)
        search_obj = build_search(model, args, args.n_assets)
        result = evaluate_checkpoint(label, trainer.state.params, search_obj,
                                     panel, args, corridor, ext_windows)
        out_name = f"results_{args.tag}_{label['name'].replace('/', '__')}.json"
        result["out_name"] = out_name
        # LEDGER pointer from the ACTUAL out_dir (scheduled runs write runs/835_stage4/).
        result["out_pointer"] = os.path.relpath(os.path.join(out_dir, out_name), _REPO)
        result["ledger_row"] = make_ledger_row(result)
        print_summary(result)
        out_path = os.path.join(out_dir, out_name)
        with open(out_path, "w") as fh:
            json.dump(result, fh, indent=2, default=float)
        print(f"[835] wrote {out_path}")
        results.append(result)

    if len(results) > 1:
        print_comparison(results)
        comp = {
            "tag": args.tag, "db": args.db, "window": [args.start, args.end],
            "eval_seed": args.eval_seed, "n_checkpoints": len(results),
            "rows": [{
                "checkpoint_label": r["checkpoint_label"],
                "net21": r["tiers"]["21"]["sharpe_full"],
                "gross": r["tiers"]["0"]["sharpe_full"],
                "pct_hold_agg_net21": r["tiers"]["21"]["pct_hold_agg"],
                "beats_floor_net21": r["corridor"]["beats_floor_net21"],
                "beats_reference_net21": r["corridor"]["beats_reference_net21"],
                "hold_collapse": r["corridor"]["hold_collapse"],
                "partial_rollout": r["corridor"]["partial_rollout"],
                "per_window_net21": {n: r["tiers"]["21"]["per_window"][n]["sharpe"]
                                     for n in EXT_WINDOW_NAMES},
                "final_pv_net21": r["tiers"]["21"]["final_pv"],
                "position_in_corridor_net21": r["corridor"]["position_in_corridor_net21"],
            } for r in results],
            "ledger_rows": [r["ledger_row"] for r in results],
        }
        comp_path = os.path.join(out_dir, f"comparison_{args.tag}.json")
        with open(comp_path, "w") as fh:
            json.dump(comp, fh, indent=2, default=float)
        print(f"[835] wrote {comp_path}")


if __name__ == "__main__":
    main()
