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
import importlib.util
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
    # TASK_850: daily-source panel builds its own env states/config (the minute path goes
    # through create_jax_env_historical; the daily path reads ohlcv_daily directly).
    EnvConfig, jit_batch_reset,
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
DECISION_RULES = ("argmax", "threshold", "top-m")   # TASK_856: policy-replay decision rules

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

# TASK_848: reuse the 832 imprint runner's deterministic-replay + weight-native-teacher
# machinery verbatim (do NOT reimplement) — its dir name is not a python identifier, so it
# is imported by path (the same pattern 832 uses for the 830 corridor module).
IMPRINT_PY = os.path.join(_REPO, "experiments", "832_linear_imprint", "imprint.py")
_IMPRINT_MOD = None


def load_imprint_module():
    """Import experiments/832_linear_imprint/imprint.py by path (cached). Provides the
    pre-registered ``policy_argmax_actions`` / ``encode_training_obs`` deterministic-replay
    operator and ``_make_weight_native_teacher_fn`` / ``walk_env`` / ``_load_corridor`` —
    lifted, not re-derived (TASK_848 scope 1 + 3)."""
    global _IMPRINT_MOD
    if _IMPRINT_MOD is None:
        spec = importlib.util.spec_from_file_location("imprint832", IMPRINT_PY)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _IMPRINT_MOD = mod
    return _IMPRINT_MOD


# =============================================================================
# TASK_856 — decision rules on the 848 policy-replay operator
# =============================================================================
# The ExIt2 policy trains toward the fixed search's visit targets (~94% hold / ~6% sparse
# trades) but a per-asset ARGMAX rounds to 100% HOLD while CE stays far from a point-mass —
# the trade signal survives IN THE DISTRIBUTION. These rules re-decode that distribution
# without touching training: 'argmax' (default) is byte-identical to the 848 replay; the
# other two read the per-asset softmax MARGINALS and act only where the mass warrants it,
# HOLDing everywhere else.
#   * threshold: per asset, act on the best non-HOLD action iff its marginal mass > tau.
#   * top-m:     per step, act on the M assets with the highest non-HOLD mass, forcing each of
#                them onto its BEST NON-HOLD action (the selected assets ALWAYS trade — the
#                point of the probe: extract ~M trades/step even when the argmax collapses to
#                100% HOLD); the rest HOLD. Assigning the full argmax would re-collapse to HOLD
#                on the collapsed ep300 checkpoints and make top-m == argmax (vacuous).

def _apply_decision_rule(probs, rule, tau, m, n_assets):
    """Map per-asset action MARGINALS (probs: (N, A, 5), softmax over the 5 actions) to a
    per-decision action vector (N, A) int32 in {0..4}. HOLD == 1. 'non-HOLD mass' = the
    largest marginal over the four non-HOLD actions. Pure numpy — no model, so the property
    tests exercise it directly."""
    if rule not in DECISION_RULES:
        raise ValueError(f"unknown decision rule: {rule!r} (choices: {DECISION_RULES})")
    N = probs.shape[0]
    hold_vec = np.full((N, n_assets), HOLD, dtype=np.int32)
    # Mask HOLD out with a sentinel below any real probability so the argmax/max pick a
    # genuine non-HOLD action (argmax and max are taken over the SAME masked array => the
    # returned action's mass IS best_nonhold_mass).
    masked = np.array(probs, dtype=np.float64, copy=True)
    masked[:, :, HOLD] = -1.0
    best_nonhold_action = masked.argmax(axis=-1).astype(np.int32)        # (N, A)
    best_nonhold_mass = masked.max(axis=-1)                              # (N, A)
    if rule == "argmax":
        # Pure-numpy analogue of the logit argmax (softmax is strictly monotonic, so the
        # argmax index is identical). The model path routes 'argmax' through the verbatim
        # imprint operator instead — this branch exists only for the property tests.
        return probs.argmax(axis=-1).astype(np.int32)
    if rule == "threshold":
        acted = best_nonhold_mass > float(tau)                          # (N, A)
        return np.where(acted, best_nonhold_action, hold_vec).astype(np.int32)
    if rule == "top-m":
        out = hold_vec.copy()
        m_eff = int(max(0, min(int(m), n_assets)))
        if m_eff > 0:
            # per decision: the M assets with the highest non-HOLD mass (ties -> lower slot
            # index via a STABLE sort => deterministic). Each selected asset is forced onto its
            # BEST NON-HOLD action (never the full argmax — that would re-collapse to HOLD on a
            # HOLD-argmax checkpoint and defeat the probe); the M chosen assets ALWAYS trade.
            order = np.argsort(-best_nonhold_mass, axis=1, kind="stable")  # (N, A)
            sel = order[:, :m_eff]                                         # (N, m_eff)
            rows = np.arange(N)[:, None]
            out[rows, sel] = best_nonhold_action[rows, sel]
        return out
    raise ValueError(f"unknown decision rule: {rule!r}")                 # unreachable


def policy_decision_actions(model, params, obs_np, n_assets, rule, tau, m, block=128):
    """TASK_856: the 848 policy-replay operator under a decision rule. rule='argmax' (default)
    delegates VERBATIM to ``imprint.policy_argmax_actions`` (byte-identical to HEAD — the
    per-asset logit argmax); 'threshold'/'top-m' read the per-asset softmax marginals via the
    same batched ``initial_inference`` pass and re-decode with ``_apply_decision_rule``."""
    imp = load_imprint_module()
    if rule == "argmax":
        return imp.policy_argmax_actions(model, params, obs_np, n_assets, block=block)

    @jax.jit
    def fwd(o):
        out = model.apply({'params': params}, o, mode='initial_inference')
        logits = out['policy_logits'].reshape(o.shape[0], n_assets, N_ACTIONS_PER_ASSET)
        return jax.nn.softmax(logits, axis=-1)
    probs = []
    for i in range(0, obs_np.shape[0], block):
        p = fwd(jnp.asarray(obs_np[i:i + block]))
        jax.block_until_ready(p)
        probs.append(np.asarray(p, dtype=np.float64))
    probs = np.concatenate(probs, axis=0)                               # (N, A, 5)
    return _apply_decision_rule(probs, rule, tau, m, n_assets)


def operator_net_dependent(args):
    """TASK_856 scope 2 — does the configured operator READ the learned checkpoint at all?
    The 832/E5-exit2 vacuous-leg hole: a search operator with pw0 + true-env leaf + no v-hat
    tail + a non-policy proposal scheme touches NONE of the net's learned heads, so grading a
    checkpoint returns bit-identical (teacher-only) numbers regardless of the weights. Fail-loud
    on that config rather than silently 'grade' it.
      * policy-argmax replay -> True (it IS the policy net, under any decision rule).
      * search -> True iff it uses the learned policy prior (prior_weight != 0 OR the
        policy-prior proposal scheme), OR the learned model leaf (leaf_eval='model' => value
        head + dynamics), OR the learned v-hat tail (true-env leaf WITH --leaf-vhat-tail).
      * TASK_857: leaf_eval='vhat-distilled' (the deployable v̂-leaf — the distilled value
        head IS the leaf, no lookahead) -> True. Mechanically the model leaf (value head +
        dynamics), so it reads the learned checkpoint exactly as leaf_eval='model' does."""
    if args.operator == "policy-argmax":
        return True
    uses_policy_prior = (args.prior_weight != 0.0
                         or args.vector_proposal_scheme == "policy-prior")
    uses_model_leaf = (args.leaf_eval in ("model", "vhat-distilled"))
    uses_vhat_tail = (args.leaf_eval == "true-env" and args.leaf_vhat_tail)
    return bool(uses_policy_prior or uses_model_leaf or uses_vhat_tail)


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


def _validate_eval_date(v, which):
    """TASK_848 fix pass: STRICT window parsing for the eval --start/--end path. ``_to_epoch``
    (an exact mirror of the env loader) silently coerces an unparseable date to a SENTINEL
    (start→0 / end→9999999999) = the FULL DB window — which would fold the imprint's in-sample
    training rows into an OOS generalization read. In the eval context we refuse the fallback:
    accept only all-digit epoch seconds, ``%Y-%m-%d``, or ISO-8601; ABORT otherwise. (The env
    loader's own sentinel behavior is left untouched — we simply never hand it a wrong value.)"""
    if v is None:
        return
    s = str(v)
    if s.isdigit():
        return
    for parse in (lambda t: datetime.strptime(t, "%Y-%m-%d"), datetime.fromisoformat):
        try:
            parse(s)
            return
        except (ValueError, TypeError):
            continue
    raise RuntimeError(
        f"eval --{which}: '{v}' is not a valid date (expected YYYY-MM-DD, ISO-8601, or epoch "
        f"seconds). Refusing the full-DB sentinel fallback — that would fold in-sample rows into "
        f"the OOS read (acceptance iii: no silently-degenerate window). Fix the --{which} value.")


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
# TASK_850 — daily-source panel loader (reads ohlcv_daily directly; no resample)
# =============================================================================
# The minute path (create_jax_env_historical) resamples the dying `ohlcv` table via a
# ``[::bar_minutes]`` stride. That table's 64-asset coverage dies ~2025-10 while
# ``ohlcv_daily`` is current to 2026-06 for 376 assets — so the 10-month OOS panel exists
# but the minute loader can't reach it (LEDGER 832/E5-oos). The daily source reads the
# daily table directly with the SAME symbol selection + SAME warm/decision-row semantics
# as the minute path, and is valid ONLY with --bar-minutes 1440 (the table IS daily —
# a non-1440 request would be an incoherent second resample). LOADER-only: zero training
# path change (imprint.py stays minute-sourced — out of scope).

def _daily_table_exists(db_path):
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='ohlcv_daily'"
        ).fetchone()
    finally:
        conn.close()
    return row is not None


def _minute_ranked_symbols(db_path, start_date, end_date, n_assets):
    """The minute path's default universe: top-N symbols coverage-ranked over the MINUTE
    `ohlcv` table (COUNT(*) DESC in the window — VectorBTTradingEnv._load_database_prices'
    no-symbols branch, verbatim). The daily source and the equivalence gate BOTH route their
    no-symbols universe through this single resolver, so the daily panel is scored on the
    SAME asset set as the minute baseline (on the real DB the majors still hold the deepest
    ohlcv history even though it ends ~Oct-2025, so they still rank top-N). Fail-loud if the
    minute table is absent/empty — never silently rank over ohlcv_daily (a disjoint universe
    that corrupts the OOS benchmark)."""
    conn = sqlite3.connect(db_path)
    try:
        has = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='ohlcv'").fetchone()
        if has is None:
            raise RuntimeError(
                f"cannot resolve a universe without --symbols: no `ohlcv` (minute) table in "
                f"{db_path} to coverage-rank against. The daily source pins its universe to the "
                f"minute baseline's top-N; pass --symbols explicitly, or point --db at a DB "
                f"that carries the minute table.")
        s_ts, e_ts = _to_epoch(start_date, 0), _to_epoch(end_date, 9999999999)
        rows = conn.execute(
            """SELECT a.symbol FROM ohlcv o JOIN assets a ON o.asset_id = a.id
               WHERE o.timestamp >= ? AND o.timestamp <= ?
               GROUP BY a.id ORDER BY COUNT(*) DESC LIMIT ?""",
            (s_ts, e_ts, n_assets)).fetchall()
    finally:
        conn.close()
    syms = [r[0] for r in rows]
    if not syms:
        raise RuntimeError(
            f"cannot resolve a universe without --symbols: the minute `ohlcv` table in "
            f"{db_path} has no rows in [{start_date}..{end_date}] to rank. Pass --symbols "
            f"or widen the window.")
    return syms


def _daily_coverage_ids(db_path, asset_ids, s_ts, e_ts):
    """Subset of `asset_ids` with >=1 real ``ohlcv_daily`` row in epoch window [s_ts, e_ts].
    Shared coverage definition (TASK_852) for the daily loader's active-mask filter AND the
    equivalence gate's FLOOR-leg universe intersection — one query shape, not two divergent
    ones. Read-only; opens its own connection (mirrors ``_daily_table_exists`` style)."""
    if not asset_ids:
        return set()
    conn = sqlite3.connect(db_path)
    try:
        ph = ",".join("?" * len(asset_ids))
        # close IS NOT NULL AND != 0 (MINOR-2 hardening): row EXISTENCE alone isn't coverage —
        # a future placeholder/garbage row with a null or zero close would otherwise be waved
        # through as "covered" and produce exactly the flat-zero densified column this filter
        # exists to eliminate, just through the other door. Latent today (no such rows on the
        # real DB) but cheap to close now.
        rows = conn.execute(
            f"""SELECT asset_id FROM ohlcv_daily WHERE asset_id IN ({ph})
                AND timestamp >= ? AND timestamp <= ?
                AND close IS NOT NULL AND close != 0 GROUP BY asset_id""",
            (*asset_ids, s_ts, e_ts)).fetchall()
    finally:
        conn.close()
    return {r[0] for r in rows}


def load_daily_prices(db_path, start_date, end_date, specific_symbols, n_assets):
    """Read the panel close matrix from ``ohlcv_daily`` — a read-only mirror of
    ``VectorBTTradingEnv._load_database_prices`` (symbol selection, distinct-timestamp axis,
    close matrix, forward-fill) with the table swapped to ``ohlcv_daily`` and NO
    ``[::bar_minutes]`` stride (the rows are already daily). Returns
    (prices_np (T, n_loaded), symbols, row_ts int64 (T,)). Fail-loud: missing table / empty
    window (acceptance iii — no silently-degenerate panel)."""
    if not _daily_table_exists(db_path):
        raise RuntimeError(
            f"--panel-source daily: no `ohlcv_daily` table in {db_path}. The daily source "
            f"reads that table directly (schema asset_id/timestamp/OHLCV); it is absent here. "
            f"Use --panel-source minute, or point --db at a DB that carries ohlcv_daily.")
    # --- symbol selection: IDENTICAL universe to the minute baseline (task scope 1) ---
    # When no --symbols is given the universe is coverage-ranked over the MINUTE `ohlcv`
    # table (exactly what the minute path selects) and those symbols' PRICE ROWS are then
    # read from ohlcv_daily. Ranking over ohlcv_daily instead would resolve a DIFFERENT
    # (meme-tail) universe than the minute baseline and silently corrupt the OOS benchmark
    # this task exists to unlock — so that path is removed. Fail-loud if the minute table
    # can't resolve a universe (never silently fall back to ranking over ohlcv_daily).
    if not specific_symbols:
        specific_symbols = _minute_ranked_symbols(db_path, start_date, end_date, n_assets)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    # Window bounds via _to_epoch (LOCAL-time date parsing) — intentionally identical to the
    # minute path (VectorBTTradingEnv parses dates in local time too), so the daily and minute
    # panels apply the SAME window semantics. (The equivalence gate uses UTC bounds instead;
    # that's a gate-only choice affecting boundary-day close bucketing, not the panels.)
    s_ts = _to_epoch(start_date, 0)
    e_ts = _to_epoch(end_date, 9999999999)
    ph = ",".join("?" * len(specific_symbols))
    rows = cur.execute(
        f"SELECT a.id, a.symbol FROM assets a WHERE a.symbol IN ({ph})",
        specific_symbols).fetchall()
    sym_to_id = {r[1]: r[0] for r in rows}
    asset_ids, symbols = [], []
    for sym in specific_symbols:               # preserve order; skip missing (loader parity)
        if sym in sym_to_id:
            asset_ids.append(sym_to_id[sym])
            symbols.append(sym)
    if not asset_ids:
        conn.close()
        raise RuntimeError(
            f"--panel-source daily: no assets resolved in {db_path} ohlcv_daily for the "
            f"requested universe/window [{start_date}..{end_date}] (empty window).")
    # ACTIVE-MASK coverage filter (TASK_852, LEDGER 850/eqgate): an asset resolved in
    # `assets` can still have ZERO ohlcv_daily rows in this window (e.g. 37/64 of the
    # minute-ranked universe pre-2025 — the daily table simply wasn't backfilled that far
    # back for them). Forward-filling such a column from a nonexistent anchor keeps it at a
    # flat zero for the WHOLE window — a fake, silently-densified "asset" that never
    # actually traded here. The minute loader never has this failure mode (its default
    # universe is ranked BY minute coverage, so a zero-coverage symbol is never selected);
    # reproduce that ACTIVE-MASK semantics here by dropping zero-coverage symbols instead of
    # densifying them. Real BUT PARTIAL histories (e.g. a coin that listed mid-window) are
    # untouched — book_target_signs' own first_active/present_mask already ages those in
    # correctly once loaded.
    covered_ids = _daily_coverage_ids(db_path, asset_ids, s_ts, e_ts)
    dropped = [sym for aid, sym in zip(asset_ids, symbols) if aid not in covered_ids]
    if dropped:
        print(f"[850] --panel-source daily ACTIVE-MASK: dropping {len(dropped)}/{len(asset_ids)} "
              f"requested symbols with ZERO ohlcv_daily rows in [{start_date}..{end_date}] (no "
              f"real daily history here; not densifying a fake flat column): "
              f"{dropped[:10]}{' ...' if len(dropped) > 10 else ''}")
        kept = [(aid, sym) for aid, sym in zip(asset_ids, symbols) if aid in covered_ids]
        asset_ids = [aid for aid, _ in kept]
        symbols = [sym for _, sym in kept]
    if not asset_ids:
        conn.close()
        raise RuntimeError(
            f"--panel-source daily: every requested symbol has ZERO ohlcv_daily rows in "
            f"[{start_date}..{end_date}] after the active-mask coverage filter (empty window).")
    id_to_col = {aid: i for i, aid in enumerate(asset_ids)}
    ph = ",".join("?" * len(asset_ids))
    timestamps = [r[0] for r in cur.execute(
        f"""SELECT DISTINCT timestamp FROM ohlcv_daily
            WHERE asset_id IN ({ph}) AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp""",
        (*asset_ids, s_ts, e_ts)).fetchall()]
    if not timestamps:
        conn.close()
        raise RuntimeError(
            f"--panel-source daily: ohlcv_daily has no rows for the requested universe in "
            f"[{start_date}..{end_date}] (empty window). Widen --start/--end.")
    ts_to_idx = {ts: i for i, ts in enumerate(timestamps)}
    n_steps = len(timestamps)
    prices = np.zeros((n_steps, len(symbols)), dtype=np.float32)
    for asset_id, ts, close in cur.execute(
        f"""SELECT asset_id, timestamp, close FROM ohlcv_daily
            WHERE asset_id IN ({ph}) AND timestamp >= ? AND timestamp <= ?""",
        (*asset_ids, s_ts, e_ts)).fetchall():
        if ts in ts_to_idx and asset_id in id_to_col:
            prices[ts_to_idx[ts], id_to_col[asset_id]] = close
    conn.close()
    # forward-fill zeros — verbatim mirror of _load_database_prices:646-656
    for col in range(prices.shape[1]):
        mask = prices[:, col] == 0
        if mask.any():
            first_valid = int(np.argmax(~mask))
            if first_valid > 0:
                prices[:first_valid, col] = prices[first_valid, col]
            for i in range(1, n_steps):
                if prices[i, col] == 0:
                    prices[i, col] = prices[i - 1, col]
    return prices, symbols, np.asarray(timestamps, dtype=np.int64)


def create_daily_panel(args):
    """Daily-source analogue of ``create_jax_env_historical``: load ``ohlcv_daily`` closes
    (no stride), pad/truncate to n_assets, build the SAME EnvConfig (env_factory.py:422-451,
    the fields build_panel drives), and reset. Returns
    (states0, prices_j, env_cfg, row_ts, loaded_symbols)."""
    prices_np, symbols, row_ts = load_daily_prices(
        args.db, args.start, args.end, args.symbols, args.n_assets)
    n_assets = args.n_assets
    if prices_np.shape[1] < n_assets:
        # TASK_852: this branch was mirrored from env_factory.py:412-415's pad-by-cycling
        # fallback, which only ever fires on the MINUTE path's rare/explicit under-supply of
        # symbols. On the daily path it is now ROUTINELY reachable — the active-mask coverage
        # filter above deliberately DROPS zero-coverage symbols instead of densifying them, so
        # any real-world window with imperfect ohlcv_daily backfill will under-supply. Silently
        # cycling the survivors back up to n_assets would duplicate real assets across multiple
        # independent panel columns, each separately sized by the env's per-asset weight-cap/
        # budget logic — doubling (or worse) their effective exposure with nothing raising or
        # printing a loud signal. Fail loud instead (the exact failure class this task exists
        # to eliminate, one layer up): the caller must explicitly opt into the smaller real
        # universe rather than have it silently densified via duplication.
        raise RuntimeError(
            f"--panel-source daily: only {prices_np.shape[1]}/{n_assets} requested assets have "
            f"real ohlcv_daily coverage in [{args.start}..{args.end}] after the active-mask "
            f"filter (the rest have zero rows there and were dropped, not densified). Padding "
            f"by cycling/duplicating the {prices_np.shape[1]} real columns back up to "
            f"{n_assets} would silently double-count their exposure — refusing. Fix: pass "
            f"--symbols {','.join(symbols)} (the exact {prices_np.shape[1]} that survived this "
            f"filter) with --n-assets {prices_np.shape[1]}. NOTE: retrying with only "
            f"--n-assets {prices_np.shape[1]} (no --symbols) is NOT guaranteed to converge — "
            f"the no-symbols path RE-RANKS a fresh top-{prices_np.shape[1]} minute-coverage "
            f"universe at that N, which can have its own, different coverage gap.")
    elif prices_np.shape[1] > n_assets:
        prices_np = prices_np[:, :n_assets]
    prices_j = jnp.array(prices_np, dtype=jnp.float32)
    # EnvConfig — identical to create_jax_env_historical's block for the args build_panel
    # passes (fee/slip 0 here; tiers derive via .replace downstream, exactly as the minute
    # path). Defaults (initial_cash, window_size 60, probabilistic_fills False, …) mirror
    # env_factory.py:422-451 so the daily panel is env-identical to a minute panel of the
    # same closes.
    config = EnvConfig(
        n_assets=n_assets,
        initial_cash=100_000.0,
        fee_bps=0.0,
        slippage_bps=0.0,
        max_steps=400,
        window_size=60,
        position_size_pct=args.position_size_pct,
        vector_action=True,
        vector_budget_step=args.vector_budget_step,
        vector_budget_pct=args.vector_budget_pct,
        vector_weight_rebalance=args.vector_weight_rebalance,
        vector_weight_unit=args.vector_weight_unit,
        vector_weight_max=args.vector_weight_max,
        vector_rebalance_deadband=args.vector_rebalance_deadband,
    )
    keys = jax.random.split(jax.random.PRNGKey(args.eval_seed), 1)
    states = jit_batch_reset(keys, prices_j, config)
    return states, prices_j, config, row_ts, symbols


# =============================================================================
# TASK_853 — fixed-frame daily panel with inactive masking (the OOS-read mechanic)
# =============================================================================
# The imprint checkpoint's encoder FLATTENS assets (kernel = 27 feats x n_assets slots):
# slot k IS asset k, POSITIONALLY. In an OOS window only a subset of the 64 training assets
# have real ohlcv_daily coverage; the TASK_852 loader correctly refuses to densify (fake
# columns) or shrink to a sub-panel (which would mis-shape the net AND scramble slot
# identities — two fail-louds fired, LEDGER 832/E5-oos). The correct mechanic (same as the
# TRAINING panel's handling of mid-window listings): keep the checkpoint's full 64-slot frame
# in its EXACT recorded training ORDER, one column per slot, and mark zero-coverage slots
# INACTIVE via the env's EXISTING active-mask semantics (compute_action_mask's
# ``price_valid = current_prices > 0`` gate). A zero-coverage column forward-fills to a flat
# ZERO (no real anchor), so ``price_valid`` is False for the whole window -> the env mask
# admits only HOLD there (no fills, no fees, no n_present renormalization) — reused, not
# reinvented. Under the search operator the mask does this automatically; under the 848
# policy-argmax replay (which bypasses the mask) the same inactive slots are forced to HOLD in
# the replayed vector (evaluate_checkpoint) — an application of the SAME price_valid rule.
# FLAG-GATED (--panel-frame-from-checkpoint), default OFF; the minute/daily paths are untouched.

def _load_frame_provenance(path, n_assets, cli_symbols):
    """Read the checkpoint's recorded training symbol FRAME (one column per positional slot)
    from a provenance JSON. Fail-loud (task scope 4): file/key absent, frame length !=
    n_assets (positional slot mismatch — the 972-vs-1728 kernel error this task exists to
    avoid), or an explicit --symbols list that disagrees with the recorded order.

    Schema: a JSON object carrying an ORDERED symbol list under ``frame_symbols`` (or
    ``symbols``); any other keys (train_window/db/n_assets/source) are carried through as
    provenance for the artifact. The frame is authoritative — the panel is built column-by-slot
    in this exact order so slot identity matches the checkpoint's encoder."""
    if not os.path.exists(path):
        raise RuntimeError(
            f"--panel-frame-from-checkpoint: provenance file not found: {path}. The fixed-frame "
            f"panel needs the checkpoint's RECORDED training symbol order (one column per "
            f"encoder slot); it cannot be inferred from an OOS window (re-ranking there yields a "
            f"DIFFERENT universe/order and scrambles slot identity). Point this at the JSON that "
            f"records the imprint run's ordered symbol frame.")
    try:
        with open(path) as fh:
            prov = json.load(fh)
    except (ValueError, OSError) as e:
        raise RuntimeError(
            f"--panel-frame-from-checkpoint: cannot read provenance {path} "
            f"({type(e).__name__}: {e}).")
    if not isinstance(prov, dict):
        raise RuntimeError(
            f"--panel-frame-from-checkpoint: provenance {path} is not a JSON object "
            f"(got {type(prov).__name__}).")
    frame = prov.get("frame_symbols", prov.get("symbols"))
    if not (isinstance(frame, list) and frame and all(isinstance(s, str) and s for s in frame)):
        raise RuntimeError(
            f"--panel-frame-from-checkpoint: provenance {path} has no non-empty ordered symbol "
            f"list under 'frame_symbols' (or 'symbols'). The recorded training frame is ABSENT — "
            f"refusing to guess it (task scope 4 fail-loud).")
    if len(set(frame)) != len(frame):
        from collections import Counter
        dupes = sorted(s for s, n in Counter(frame).items() if n > 1)
        raise RuntimeError(
            f"--panel-frame-from-checkpoint: provenance {path} has DUPLICATE symbol(s) in the "
            f"frame: {dupes}. Each positional slot must be a DISTINCT asset (slot k IS asset k); "
            f"a repeated symbol silently collapses two slots onto one column and scrambles slot "
            f"identity — the exact failure this fixed-frame loader exists to prevent (scope 4).")
    if len(frame) != n_assets:
        raise RuntimeError(
            f"--panel-frame-from-checkpoint: recorded frame has {len(frame)} symbols but "
            f"--n-assets is {n_assets}. Slot k IS asset k positionally, so the frame MUST have "
            f"exactly n_assets columns — a mismatched count mis-shapes the encoder kernel "
            f"(the 27x{n_assets} vs 27x{len(frame)} failure). Pass --n-assets {len(frame)} to "
            f"match the recorded frame, or use the provenance that matches this checkpoint.")
    if cli_symbols is not None and list(cli_symbols) != list(frame):
        raise RuntimeError(
            f"--panel-frame-from-checkpoint: explicit --symbols disagrees with the recorded "
            f"training frame (order mismatch). The frame is authoritative and order-sensitive "
            f"(slot identity); drop --symbols, or make it match the recorded order exactly.")
    return list(frame), prov


def load_daily_prices_framed(db_path, start_date, end_date, frame_symbols):
    """Fixed-frame daily close matrix: ONE column per recorded frame symbol IN ORDER (no drop,
    no pad-by-cycling — the identity-preserving contrast to ``load_daily_prices``). Zero-coverage
    frame symbols are KEPT as flat-zero columns (the env's price>0 mask renders them inactive).
    Returns (prices_np (T, n_frame), frame_symbols, row_ts int64 (T,), coverage) where
    ``coverage`` lists per-slot status/spans. Fail-loud: missing daily table, an unknown symbol
    (not in ``assets``), or an empty window / all-slots-inactive (task scope 4)."""
    if not _daily_table_exists(db_path):
        raise RuntimeError(
            f"--panel-frame-from-checkpoint: no `ohlcv_daily` table in {db_path}. The fixed-frame "
            f"panel reads that table directly; it is absent here.")
    s_ts = _to_epoch(start_date, 0)
    e_ts = _to_epoch(end_date, 9999999999)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    ph = ",".join("?" * len(frame_symbols))
    rows = cur.execute(
        f"SELECT a.id, a.symbol FROM assets a WHERE a.symbol IN ({ph})",
        frame_symbols).fetchall()
    sym_to_id = {r[1]: r[0] for r in rows}
    unknown = [s for s in frame_symbols if s not in sym_to_id]
    if unknown:
        conn.close()
        raise RuntimeError(
            f"--panel-frame-from-checkpoint: {len(unknown)} recorded frame symbol(s) are not in "
            f"the `assets` table of {db_path} (unknown symbols — the frame cannot be dropped or "
            f"reordered to skip them without scrambling slot identity): "
            f"{unknown[:10]}{' ...' if len(unknown) > 10 else ''}")
    # One asset_id per slot, in recorded frame order (slot k IS asset k).
    asset_ids = [sym_to_id[s] for s in frame_symbols]
    # Shared coverage definition (TASK_852 _daily_coverage_ids): >=1 real close row in window.
    covered_ids = _daily_coverage_ids(db_path, asset_ids, s_ts, e_ts)
    covered_ids_ordered = [aid for aid in asset_ids if aid in covered_ids]
    if not covered_ids_ordered:
        conn.close()
        raise RuntimeError(
            f"--panel-frame-from-checkpoint: EVERY slot in the recorded frame has ZERO "
            f"ohlcv_daily coverage in [{start_date}..{end_date}] — the panel would be entirely "
            f"inactive (all HOLD). Nothing to evaluate; widen --start/--end or check the frame.")
    # Row axis: DISTINCT daily timestamps over the COVERED slots (verbatim mirror of
    # load_daily_prices' axis query — uncovered slots contribute no rows anyway).
    phc = ",".join("?" * len(covered_ids_ordered))
    timestamps = [r[0] for r in cur.execute(
        f"""SELECT DISTINCT timestamp FROM ohlcv_daily
            WHERE asset_id IN ({phc}) AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp""",
        (*covered_ids_ordered, s_ts, e_ts)).fetchall()]
    if not timestamps:
        conn.close()
        raise RuntimeError(
            f"--panel-frame-from-checkpoint: ohlcv_daily has no rows for the frame in "
            f"[{start_date}..{end_date}] (empty window). Widen --start/--end.")
    ts_to_idx = {ts: i for i, ts in enumerate(timestamps)}
    id_to_col = {aid: i for i, aid in enumerate(asset_ids)}     # slot == frame position
    n_steps = len(timestamps)
    prices = np.zeros((n_steps, len(frame_symbols)), dtype=np.float32)
    for asset_id, ts, close in cur.execute(
        f"""SELECT asset_id, timestamp, close FROM ohlcv_daily
            WHERE asset_id IN ({phc}) AND timestamp >= ? AND timestamp <= ?""",
        (*covered_ids_ordered, s_ts, e_ts)).fetchall():
        if ts in ts_to_idx and asset_id in id_to_col:
            prices[ts_to_idx[ts], id_to_col[asset_id]] = close
    conn.close()
    # Inactive-span report BEFORE the fill (the fill back-/forward-fills covered columns; a
    # zero-coverage column has no anchor and stays flat-zero, which is exactly the inactive
    # signal). raw_nonzero[:, col] tells real vs filled.
    raw_nonzero = prices != 0
    coverage = []
    for col, sym in enumerate(frame_symbols):
        aid = asset_ids[col]
        n_nz = int(raw_nonzero[:, col].sum())
        if aid not in covered_ids or n_nz == 0:
            coverage.append({"slot": col, "symbol": sym, "status": "inactive",
                             "reason": "zero ohlcv_daily coverage in window",
                             "span": [start_date, end_date]})
        else:
            first = int(np.argmax(raw_nonzero[:, col]))
            entry = {"slot": col, "symbol": sym, "status": "active",
                     "covered_rows": n_nz, "n_rows": n_steps}
            if first > 0:                       # listed mid-window: leading rows had no data
                entry["status"] = "partial"
                entry["first_covered_ts"] = int(timestamps[first])
                entry["leading_inactive_rows"] = first
            coverage.append(entry)
    # Fill: forward-fill INTERIOR gaps only (from each slot's first real close onward). Leading
    # pre-coverage rows are LEFT flat-zero — a DELIBERATE divergence from the coverage-ranked
    # load_daily_prices (whose universe has no uncovered leading region, so it back-fills). In the
    # fixed frame a slot may be zero-coverage (fully inactive) OR list mid-window (partial); either
    # way its no-real-price rows must read INACTIVE, not carry a fabricated pre-listing price. Left
    # flat-zero, the env price>0 mask HOLDs them under search, the teacher-floor present-mask ages
    # them in, and the policy-argmax price-validity guard (evaluate_checkpoint) HOLDs them too — no
    # tradeable pre-listing region for EITHER operator (leak-free invariant).
    for col in range(prices.shape[1]):
        col_nz = prices[:, col] != 0
        if not col_nz.any():
            continue                                    # fully-inactive slot: stays flat-zero
        first_nz = int(np.argmax(col_nz))               # first real close (0 if covered from open)
        for i in range(first_nz + 1, n_steps):          # forward-fill interior gaps post-listing
            if prices[i, col] == 0:
                prices[i, col] = prices[i - 1, col]
    return prices, list(frame_symbols), np.asarray(timestamps, dtype=np.int64), coverage


def create_daily_panel_framed(args, frame_symbols):
    """Fixed-frame analogue of ``create_daily_panel``: load the framed daily closes (one column
    per recorded slot, zero-coverage columns kept flat-zero), build the SAME EnvConfig, reset.
    Returns (states0, prices_j, env_cfg, row_ts, frame_symbols, coverage). No pad/truncate — the
    frame is exactly n_assets columns by construction (asserted by _load_frame_provenance)."""
    prices_np, symbols, row_ts, coverage = load_daily_prices_framed(
        args.db, args.start, args.end, frame_symbols)
    n_assets = args.n_assets
    if prices_np.shape[1] != n_assets:          # defensive: frame length already checked upstream
        raise RuntimeError(
            f"--panel-frame-from-checkpoint: framed panel has {prices_np.shape[1]} columns but "
            f"--n-assets is {n_assets} (should be equal by construction).")
    prices_j = jnp.array(prices_np, dtype=jnp.float32)
    config = EnvConfig(                          # identical to create_daily_panel's block
        n_assets=n_assets,
        initial_cash=100_000.0,
        fee_bps=0.0,
        slippage_bps=0.0,
        max_steps=400,
        window_size=60,
        position_size_pct=args.position_size_pct,
        vector_action=True,
        vector_budget_step=args.vector_budget_step,
        vector_budget_pct=args.vector_budget_pct,
        vector_weight_rebalance=args.vector_weight_rebalance,
        vector_weight_unit=args.vector_weight_unit,
        vector_weight_max=args.vector_weight_max,
        vector_rebalance_deadband=args.vector_rebalance_deadband,
    )
    keys = jax.random.split(jax.random.PRNGKey(args.eval_seed), 1)
    states = jit_batch_reset(keys, prices_j, config)
    return states, prices_j, config, row_ts, symbols, coverage


def _build_panel_framed(args):
    """TASK_853 fixed-frame source: build the daily panel on the checkpoint's recorded training
    symbol frame (exact order, one column per slot) and mark zero-coverage slots INACTIVE. Valid
    ONLY with --bar-minutes 1440 (the daily-table rule; fail-loud otherwise). Reports every
    inactive span to stdout AND (via panel_info) into the JSON artifact so a 36/64-live-slot
    benchmark is impossible to mistake for a 64-live one (task scope 2)."""
    if args.bar_minutes != 1440:
        raise RuntimeError(
            f"--panel-frame-from-checkpoint requires --bar-minutes 1440 (got {args.bar_minutes}); "
            f"the frame is read from the daily ohlcv_daily table (any other stride is an "
            f"incoherent second resample of already-daily rows).")
    frame_symbols, prov = _load_frame_provenance(
        args.panel_frame_from_checkpoint, args.n_assets, args.symbols)
    states0, prices, env_cfg, row_ts, loaded_syms, coverage = create_daily_panel_framed(
        args, frame_symbols)
    result = _finalize_panel(states0, prices, env_cfg, row_ts, loaded_syms, args)
    prices_f, env_cfg_f, state_init, row_ts_f, col_symbols, panel_info = result
    inactive = [c for c in coverage if c["status"] == "inactive"]
    partial = [c for c in coverage if c["status"] == "partial"]
    inactive_slots = [c["slot"] for c in inactive]
    n_active = args.n_assets - len(inactive_slots)
    # --- stdout report (task scope 2: impossible to mistake 36/64-live for 64-live) ---
    print("=" * 84)
    print(f"[853] FIXED-FRAME daily panel from checkpoint provenance "
          f"({_display_path(args.panel_frame_from_checkpoint)}): {args.n_assets} slots, "
          f"{n_active} ACTIVE / {len(inactive_slots)} INACTIVE (zero coverage) in "
          f"[{args.start}..{args.end}]")
    if inactive:
        print(f"[853] INACTIVE slots ({len(inactive)}) — no fills/fees, HOLD-only (env price>0 "
              f"active-mask); every span reported:")
        for c in inactive:
            print(f"        slot {c['slot']:>3}  {c['symbol']:<16} INACTIVE "
                  f"[{c['span'][0]}..{c['span'][1]}]  ({c['reason']})")
    if partial:
        print(f"[853] PARTIAL slots ({len(partial)}) — listed mid-window: pre-listing rows are "
              f"flat-zero => INACTIVE (HOLD under BOTH operators via price>0 / price-validity), "
              f"active from the listing day onward:")
        for c in partial:
            print(f"        slot {c['slot']:>3}  {c['symbol']:<16} first covered ts "
                  f"{c['first_covered_ts']} ({c['leading_inactive_rows']} leading inactive rows)")
    if not inactive and not partial:
        print("[853] all slots fully covered on this window (no inactive/partial spans).")
    print("=" * 84)
    panel_info["panel_frame"] = {
        "source": "checkpoint-provenance-fixed-frame (TASK_853)",
        "provenance_path": _display_path(args.panel_frame_from_checkpoint),
        "provenance_meta": {k: v for k, v in prov.items()
                            if k not in ("frame_symbols", "symbols")},
        "n_slots": int(args.n_assets),
        "n_active_slots": int(n_active),
        "n_inactive_slots": int(len(inactive_slots)),
        "n_partial_slots": int(len(partial)),
        "inactive_slots": list(inactive_slots),
        "inactive_spans": inactive,
        "partial_spans": partial,
        "coverage": coverage,
    }
    # Carried to the rollout: force these slots to HOLD on the 848 policy-argmax replay (which
    # bypasses the env mask); the search operator already masks them. See evaluate_checkpoint.
    panel_info["inactive_slots"] = list(inactive_slots)
    return prices_f, env_cfg_f, state_init, row_ts_f, col_symbols, panel_info


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
        vector_proposal_temp=args.vector_proposal_temp,
        vector_proposal_explore_slots=args.vector_proposal_explore_slots,
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


def check_checkpoint_arch(ckpt_path, args):
    """Fail-loud architecture guard (TASK_848 acceptance iii). The TRM shares weights across
    its H/L recursion cycles, so a checkpoint evaluated with the WRONG --h-cycles/--l-cycles
    loads with IDENTICAL param shapes and silently unrolls a different net than was trained
    (a wrong-operator result, not a crash — unlike an n_assets/state_dim mismatch, which the
    param-shape check already rejects on load). Verify the eval arch against the checkpoint's
    ``trainer_metadata.json`` (written by trainer.save_checkpoint). Missing metadata -> WARN
    (older checkpoint, cannot verify); mismatch -> ABORT before any rollout runs."""
    meta_path = os.path.join(os.path.abspath(ckpt_path), "trainer_metadata.json")
    if not os.path.exists(meta_path):
        print(f"[835] WARN: no trainer_metadata.json in {ckpt_path} — cannot verify checkpoint "
              f"architecture; ensure --h-cycles/--l-cycles match the training run")
        return
    try:
        with open(meta_path) as fh:
            meta = json.load(fh)
    except (ValueError, OSError) as e:
        # Corrupt/truncated metadata (MEMORY: zstd->rsync doesn't detect source corruption) is
        # NOT a hard arch mismatch — trainer.load_checkpoint tolerates it and the pre-848 search
        # path never read this file. Downgrade to the SAME WARN-and-continue as missing metadata.
        print(f"[835] WARN: trainer_metadata.json in {ckpt_path} is unreadable ({type(e).__name__}"
              f": {e}) — cannot verify checkpoint architecture; ensure --h-cycles/--l-cycles match")
        return
    if not isinstance(meta, dict):
        # Valid JSON but wrong TYPE (null / number / list — page corruption can produce these):
        # meta.get(...) would raise AttributeError. Same WARN-and-continue as corrupt/missing.
        print(f"[835] WARN: trainer_metadata.json in {ckpt_path} is not a JSON object "
              f"(got {type(meta).__name__}) — cannot verify checkpoint architecture; ensure "
              f"--h-cycles/--l-cycles match")
        return
    want = {"h_cycles": args.h_cycles, "l_cycles": args.l_cycles}
    mism = [(k, meta.get(k), v) for k, v in want.items()
            if meta.get(k) is not None and meta.get(k) != v]
    if mism:
        detail = "; ".join(f"{k}: ckpt={m} eval={v}" for k, m, v in mism)
        raise RuntimeError(
            f"checkpoint architecture mismatch ({ckpt_path}): {detail}. The TRM shares weights "
            f"across H/L recursion cycles, so a wrong --h-cycles/--l-cycles loads with identical "
            f"param shapes and would SILENTLY evaluate a different net than was trained. Pass "
            f"--h-cycles {meta.get('h_cycles')} --l-cycles {meta.get('l_cycles')} to match "
            f"(source: {os.path.relpath(meta_path, _REPO)}).")


def build_search(model, args, n_assets):
    """Mirror of train_jax_fast.py:375-385 — the shipped Stage-2/3 search operator.
    ``vector_gumbel_scale_eval`` is reserved/unwired upstream and is NOT touched here.

    TASK_857: ``--leaf-eval vhat-distilled`` is the DEPLOYABLE v̂-leaf read (the distilled
    value head IS the leaf, no true-env lookahead). Mechanically it is the canon MODEL leaf
    (value head + dynamics via mctx's recurrent inference), so the search constructor is fed
    ``leaf_eval='model'`` — the search is BIT-IDENTICAL to a ``--leaf-eval model`` run on the
    same checkpoint. The distinction is provenance only: it names the counterfactual-distilled
    read and (via ``operator_net_dependent``) records net_dependent=true even at prior_weight=0
    (the v̂ leaf is the learned component). sampled_search.py is UNTOUCHED — the new mode lives
    entirely in this operator harness (task scope 3)."""
    search_leaf_eval = "model" if args.leaf_eval == "vhat-distilled" else args.leaf_eval
    return create_sampled_vector_search(
        model, n_assets=n_assets, n_actions_per_asset=N_ACTIONS_PER_ASSET,
        num_samples=args.vector_k, num_simulations=args.sims,
        max_num_considered_actions=min(args.vector_k, 16), discount=args.discount,
        gumbel_scale=args.gumbel_scale, value_scale=args.value_scale,
        value_shift=args.value_shift,
        hold_bias=args.vector_hold_bias,
        dither_scale=args.vector_dither_scale,
        proposal_scheme=args.vector_proposal_scheme,
        # TASK_847: policy-prior knobs (ignored unless scheme='policy-prior').
        proposal_temp=args.vector_proposal_temp,
        explore_slots=args.vector_proposal_explore_slots,
        # TASK_854 (D21): learned-prior weight in candidate scoring (default 1.0 = canon;
        # 0.0 = pure-Q / E1 selection — the last learned component vs the E1 anchor).
        prior_weight=args.prior_weight,
        # TASK_838: eval operator must match the training operator — a checkpoint
        # trained with --vector-per-node-resample must be evaluated with the SAME
        # flag or the measurement is train/eval-asymmetric (default OFF = canon).
        per_node_resampling=args.vector_per_node_resample,
        # TASK_851: search leaf evaluator (default OFF = model leaf, canon).
        # TASK_857: 'vhat-distilled' -> 'model' here (mechanically the model leaf; see docstring).
        leaf_eval=search_leaf_eval,
        leaf_rollout_r=args.leaf_rollout_r,
        leaf_vhat_tail=args.leaf_vhat_tail,
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
            eval_seed, max_positions, min_cash_threshold, policy_actions=None,
            leaf_eval='model'):
    """Walk the full timeline once: obs → mask → deterministic vector search → step_vector.
    Returns per-decision portfolio returns + the HOLD-collapse / position diagnostics.

    TASK_848: when ``policy_actions`` (an (N, A) int array) is supplied the SEARCH branch is
    bypassed entirely — decision k replays the precomputed per-asset policy-argmax vector
    directly (no obs/mask/search/dither), the deterministic direct-replay operator lifted from
    832's imprint-sanity. ``policy_actions is None`` (default) keeps the search path
    character-for-character identical to the shipped operator (acceptance ii).

    TASK_851: ``leaf_eval='true-env'`` passes the REAL pre-action state/panel/config into the
    search so its true-env leaf can step a copy of the env forward (see sampled_search.py's
    module docstring). Default 'model' issues the UNCHANGED canon search call — zero new
    kwargs, bit-identical (acceptance iii)."""
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
        if policy_actions is None:
            # A. leak-free obs window [t-window, t-1] + legal mask (self_play.py:645-668)
            price_idx = state.episode_start_idx + state.current_step      # (1,)
            window_starts = price_idx - enc_cfg.window_size
            obs = jit_batch_encode_simple(prices, window_starts, enc_cfg)
            current_prices = prices[price_idx]                            # (1, A)
            mask = jit_batch_compute_action_mask(
                state, current_prices, env_cfg, max_positions, min_cash_threshold)
            # B. deterministic Sampled-MuZero vector search (visit-argmax selection)
            key, step_rng = jax.random.split(key)
            if leaf_eval == 'true-env':
                # TASK_851: true-env leaf needs the real (pre-action) env state/panel/
                # config to step a copy of it forward.
                action_vec, _sampled, _visit_probs, _stats = search_obj(
                    params, step_rng, obs, mask, book, True,      # deterministic=True (eval)
                    env_state=state, prices=prices, env_config=env_cfg)
            else:
                action_vec, _sampled, _visit_probs, _stats = search_obj(
                    params, step_rng, obs, mask, book, True)      # deterministic=True (eval)
        else:
            # B'. TASK_848 policy-argmax replay: the precomputed per-asset argmax vector,
            # no search, no mask (the 832 imprint-sanity direct-replay operator).
            action_vec = jnp.asarray(policy_actions[k], dtype=jnp.int32)[None, :]
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

def book_target_signs(prices):
    """The 819 book's per-asset target sign per row + its warmup, via the 830 corridor module
    (``book_target_signs`` on log-close). Returns (tgt, book_warm)."""
    imp = load_imprint_module()
    corridor_mod = imp._load_corridor()
    logc = np.log(np.maximum(np.asarray(prices, dtype=np.float64), 1e-12))
    tgt, _pres, book_warm = corridor_mod.book_target_signs(logc)
    return tgt, int(book_warm)


def guard_book_warm(book_warm, warm):
    """Fail-loud alignment guard (fix pass): the operator/teacher start decisions at row
    ``warm`` (= env window_size), but the 819 book has no valid target until row ``book_warm``
    (imprint replayed from max(book_warm, window_size)). If the book warmup exceeds the env
    window, the leading decision rows were never imprinted — abort rather than score them."""
    if book_warm > warm:
        raise RuntimeError(
            f"book warmup ({book_warm}) exceeds env window ({warm}) on this window — the "
            f"operator starts decisions at row {warm} but the 819 book has no valid target "
            f"until row {book_warm}; the first {book_warm - warm} decision rows were never "
            f"imprinted. Start --start earlier so every decision row has a valid book target.")


def compute_teacher_floor(prices, env_cfg0, warm, n_decisions, period_minutes):
    """TASK_848 scope 3 — the on-window teacher floor. Walks 832's weight-native ladder
    teacher (``_make_weight_native_teacher_fn``, same sentinels) on the SAME decision rows
    [warm, warm+N) the operator replays, through the SAME env config (its cost tier + weight
    flag). Returns ({cost: {sharpe, final_pv, ...}}, book_warm). Fail-loud if the book warmup
    exceeds the env window (some decision rows would read strategy-warmup signs)."""
    imp = load_imprint_module()
    n_assets = int(env_cfg0.n_assets)
    # FAIL-CLOSED (fix pass, imprint.py:479-482 precedent): the weight-native ladder emits
    # 0/1/2 as -1/0/+1-unit TARGET-WEIGHT steps; walked through the ORDER env (weight OFF)
    # those codes are misread as SELL/HOLD/BUY orders => an incoherent floor. Require the
    # weight env so the teacher walk actually runs weight semantics.
    assert env_cfg0.vector_weight_rebalance, (
        "--teacher-floor requires --vector-weight-rebalance so the weight-native ladder teacher "
        "walks WEIGHT semantics; without it the ladder's 0/1/2 target-weight codes are misread "
        "as SELL/HOLD/BUY orders (an incoherent floor). (imprint.py:479-482 precedent.)")
    tgt, book_warm = book_target_signs(prices)
    guard_book_warm(book_warm, warm)
    wn_fn = imp._make_weight_native_teacher_fn(prices, env_cfg0, tgt, warm, n_assets)
    floor = {}
    for c in GATE_COSTS:
        fee, slip = (0.0, 0.0) if c == 0 else (FEE_BPS, SLIP_BPS)
        cfg_c = env_cfg0.replace(fee_bps=fee, slippage_bps=slip)
        wrep = imp.walk_env(prices, cfg_c, warm, n_decisions, wn_fn)
        s = _sharpe(wrep["rets"], period_minutes)
        floor[c] = {"sharpe": _f(s), "final_pv": wrep["final_pv"],
                    "avg_long": wrep["avg_long"], "avg_short": wrep["avg_short"],
                    "bankrupt_at": wrep["bankrupt_at"]}
    return floor, int(book_warm)


def evaluate_checkpoint(label, model, params, search_obj, panel, args, corridor, ext_windows):
    prices, env_cfg0, state_init, row_ts, col_symbols, panel_info = panel
    warm = int(env_cfg0.window_size)
    n_decisions = panel_info["n_decisions"]
    period_minutes = float(args.bar_minutes)
    ts_dec = row_ts[warm:warm + n_decisions]

    # TASK_848: policy-argmax = the deterministic direct-replay operator lifted from 832.
    # Obs depend only on prices (state-independent), so the whole per-asset argmax vector
    # sequence batches through the net ONCE and replays across every cost tier (the policy
    # is fee-blind; only the env trajectory differs by tier).
    policy_actions = None
    if args.operator == "policy-argmax":
        imp = load_imprint_module()
        # Alignment guard fires for the BARE operator too (not only under --teacher-floor):
        # a policy-argmax replay on rows the 819 book never covered is scoring un-imprinted
        # decisions. Cheap numpy book-sign pass; abort if book_warm > window_size.
        _tgt, book_warm_op = book_target_signs(prices)
        guard_book_warm(book_warm_op, warm)
        obs_all = imp.encode_training_obs(prices, warm, n_decisions, env_cfg0)
        # TASK_856: decision rule over the policy marginals. 'argmax' (default) delegates
        # VERBATIM to imprint.policy_argmax_actions (byte-identical to HEAD); 'threshold'/
        # 'top-m' re-decode the same softmax marginals.
        policy_actions = policy_decision_actions(
            model, params, obs_all, int(env_cfg0.n_assets),
            args.decision_rule, args.decision_tau, args.decision_m)
        # TASK_853: on the fixed-frame OOS panel a slot has NO real price at a decision row when
        # it is fully-inactive (flat-zero all window) OR partial and the row is BEFORE its
        # mid-window listing (left flat-zero by the framed loader). The env action mask
        # (price_valid = price>0) HOLDs those under search; policy-argmax replay BYPASSES the mask,
        # so apply the SAME rule per (decision, slot) here. Forcing HOLD on a zero-price cell is
        # always correct (no tradeable market) and can never HOLD a genuinely-active decision (its
        # price is >0), so it is alignment-safe. Framed path only (panel_frame present) → non-framed
        # policy-argmax stays byte-identical.
        if panel_info.get("panel_frame") is not None:
            dec_prices = np.asarray(prices[warm:warm + n_decisions])   # (n_decisions, n_assets)
            _invalid = dec_prices <= 0
            if _invalid.any():
                policy_actions = np.array(policy_actions, copy=True)
                policy_actions[_invalid] = HOLD

    # TASK_848 scope 3: the on-window teacher floor (weight-native ladder, same rows/env).
    teacher_floor = book_warm = None
    if args.teacher_floor:
        teacher_floor, book_warm = compute_teacher_floor(
            prices, env_cfg0, warm, n_decisions, period_minutes)

    tiers = {}
    for c in GATE_COSTS:
        fee, slip = (0.0, 0.0) if c == 0 else (FEE_BPS, SLIP_BPS)
        # cost tier enters via the env config ONLY; identical panel + seed per tier
        # (the 830/831 separate-rollout convention — fee-aware trajectories differ).
        env_cfg = env_cfg0.replace(fee_bps=fee, slippage_bps=slip)
        rep = rollout(params, search_obj, prices, env_cfg, state_init,
                      env_cfg0.n_assets, n_decisions, args.eval_seed,
                      args.max_positions, args.min_cash_threshold,
                      policy_actions=policy_actions, leaf_eval=args.leaf_eval)
        st = score_tier(rep, ts_dec[:rep["n_exec"]], ext_windows,
                        period_minutes, c, col_symbols, n_decisions)
        # scope-3 fields: teacher floor on the SAME window + policy-vs-floor delta (per tier).
        # WITHHOLD the delta on a partial rollout — the policy Sharpe is over n_exec rows while
        # the teacher walk has no done-break (imprint.walk_env) and covers all N: subtracting
        # them is apples-to-oranges (mirror the corridor booleans' partial gating below).
        tf = teacher_floor[c]["sharpe"] if teacher_floor else None
        st["teacher_floor_inenv"] = tf
        st["corridor_position"] = (
            None if (tf is None or st["sharpe_full"] is None or st["partial"])
            else float(st["sharpe_full"] - tf))
        tiers[str(c)] = st

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
    # TASK_848 scope 3/4: the on-window teacher floor delta (the OOS read of record).
    tf21 = teacher_floor[21]["sharpe"] if teacher_floor else None
    corridor_block["teacher_floor_net21"] = tf21
    corridor_block["policy_minus_floor_net21"] = (
        None if (tf21 is None or agent_net is None or partial) else float(agent_net - tf21))

    result = {
        "tag": args.tag, "checkpoint": label["path"], "checkpoint_label": label["name"],
        "db": args.db, "window": [args.start, args.end], "bar_minutes": args.bar_minutes,
        "eval_seed": args.eval_seed,
        "panel": panel_info,
        "operator": {
            "mode": args.operator,                                   # TASK_848
            "vector_k": args.vector_k, "sims": args.sims,
            "hold_bias": args.vector_hold_bias, "dither_scale": args.vector_dither_scale,
            "proposal_scheme": args.vector_proposal_scheme,
            "proposal_temp": args.vector_proposal_temp,              # TASK_847
            "proposal_explore_slots": args.vector_proposal_explore_slots,  # TASK_847
            "prior_weight": args.prior_weight,                       # TASK_854
            "per_node_resampling": args.vector_per_node_resample,  # TASK_838
            "vector_budget_step": args.vector_budget_step,   # TASK_839
            "vector_budget_pct": args.vector_budget_pct,     # TASK_839
            "vector_weight_rebalance": args.vector_weight_rebalance,     # TASK_840
            "vector_weight_unit": args.vector_weight_unit,              # TASK_840
            "vector_weight_max": args.vector_weight_max,                # TASK_840
            "vector_rebalance_deadband": args.vector_rebalance_deadband,  # TASK_840
            "leaf_eval": args.leaf_eval,                       # TASK_851
            "leaf_rollout_r": args.leaf_rollout_r,             # TASK_851
            "leaf_vhat_tail": args.leaf_vhat_tail,              # TASK_851
            "gumbel_scale": args.gumbel_scale,
            "max_num_considered_actions": min(args.vector_k, 16),
            "deterministic": True,
            "selection": ("visit_probs argmax" if args.operator == "search"
                          else "per-asset policy-logit argmax (initial_inference, no search)"),
            "model_size": args.model_size, "h_cycles": args.h_cycles,
            "l_cycles": args.l_cycles, "discount": args.discount,
        },
        "cost_tiers_bp": list(GATE_COSTS),
        "annualization_period_minutes": period_minutes,
        "tiers": tiers,
        "corridor": corridor_block,
        # TASK_848 scope 3/4: on-window teacher floor block (None unless --teacher-floor).
        "teacher_floor": (None if teacher_floor is None else {
            "mapping": "weight-native ladder (832 _make_weight_native_teacher_fn, same sentinels)",
            "book_warm": book_warm,
            "weight_rebalance_env": bool(env_cfg0.vector_weight_rebalance),
            "same_rows_as_operator": True,
            "per_tier": {str(c): teacher_floor[c] for c in GATE_COSTS},
        }),
    }
    # TASK_853: fixed-frame inactive-mask block (present only when --panel-frame-from-checkpoint).
    # Every inactive span lands in the JSON here AND on stdout at build time (task scope 2).
    if panel_info.get("panel_frame") is not None:
        result["panel_frame"] = panel_info["panel_frame"]
    # TASK_856 scope 1: decision-rule provenance. Flag-gated (default 'argmax' adds NOTHING =>
    # byte-identical to HEAD); a non-default rule records its knobs and rewrites the operator
    # selection string so the artifact is self-describing.
    if args.operator == "policy-argmax" and args.decision_rule != "argmax":
        op = result["operator"]
        op["decision_rule"] = args.decision_rule
        op["decision_tau"] = args.decision_tau
        op["decision_m"] = args.decision_m
        op["selection"] = (
            f"per-asset policy marginal decision-rule '{args.decision_rule}' "
            f"(tau={args.decision_tau}, m={args.decision_m}; initial_inference, no search)")
    # TASK_856 scope 2: net-dependence grade. Written for EVERY run (an ADDITIVE provenance key,
    # not a scored quantity — scored keys + action arrays stay byte-identical vs HEAD): the 848
    # replay and every pw>0 / model-leaf / v-hat-leaf operator record `true`; a VACUOUS operator
    # (zero learned components — pw0 + true-env leaf + no v-hat tail + non-policy proposal)
    # records `false` AND fires a loud WARNING (the read cannot grade a checkpoint — bit-identical
    # teacher-only numbers regardless of weights, LEDGER 832/E5-exit2 item 2). run_queue consumers
    # can assert on result["net_dependent"] without a KeyError.
    result["net_dependent"] = operator_net_dependent(args)
    if not result["net_dependent"]:
        print("*" * 84)
        print("*** WARNING [856]: net_dependent=false — the configured operator has ZERO learned "
              "components")
        print("*** (prior_weight=0, true-env leaf, no v-hat tail, non-policy proposal). This read "
              "CANNOT")
        print("*** grade a checkpoint: the result is teacher/heuristic-only and bit-identical "
              "across")
        print(f"*** weights. Operator={args.operator} prior_weight={args.prior_weight} "
              f"leaf_eval={args.leaf_eval} vhat_tail={args.leaf_vhat_tail} "
              f"proposal={args.vector_proposal_scheme}")
        print("*" * 84)
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

def _finalize_panel(states0, prices, env_cfg, row_ts, loaded_syms, args):
    """Shared panel tail: assert loader/env row-count consistency, size the deterministic
    full-timeline episode, build col_symbols + panel_info. Identical for both sources so a
    daily panel and a same-close minute panel produce byte-identical downstream scorecards."""
    T = int(prices.shape[0])
    if len(row_ts) != T:
        raise RuntimeError(
            f"panel-consistency violation: reconstructed timestamp axis has {len(row_ts)} "
            f"rows but the loader produced {T} — loader mirror out of sync")
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


def _build_panel_daily(args):
    """TASK_850 daily source: read ohlcv_daily directly (no [::bar_minutes] stride) then the
    SAME finalize tail as the minute path. Valid ONLY with --bar-minutes 1440 (the table is
    daily; any other stride would be an incoherent second resample) — fail-loud otherwise
    (acceptance iii)."""
    if args.bar_minutes != 1440:
        raise RuntimeError(
            f"--panel-source daily requires --bar-minutes 1440 (got {args.bar_minutes}). "
            f"The ohlcv_daily table already IS daily; a non-1440 bar would imply a second "
            f"resample of already-daily rows (incoherent). Pass --bar-minutes 1440.")
    states0, prices, env_cfg, row_ts, loaded_syms = create_daily_panel(args)
    return _finalize_panel(states0, prices, env_cfg, row_ts, loaded_syms, args)


def build_panel(args):
    """The training panel, via the training factory itself. Gross config returned;
    tiers derive from it via .replace(fee/slip) — prices identical by construction.

    TASK_850: --panel-source daily reads ohlcv_daily directly (the minute `ohlcv` table's
    coverage dies ~2025-10); default 'minute' is byte-identical to the pre-850 path.

    TASK_853: --panel-frame-from-checkpoint builds the fixed 64-slot frame in the checkpoint's
    recorded training order with zero-coverage slots INACTIVE (the OOS-read mechanic); default
    (flag unset) is byte-identical to the minute/daily paths above."""
    _frame_path = getattr(args, "panel_frame_from_checkpoint", None)
    if _frame_path is not None and not str(_frame_path).strip():
        raise RuntimeError(
            "--panel-frame-from-checkpoint was given an EMPTY path; refusing to silently fall "
            "back to the minute/daily universe (that would score a DIFFERENT universe under the "
            "fixed-frame flag). Pass the provenance JSON path, or omit the flag entirely.")
    if _frame_path:
        return _build_panel_framed(args)
    if getattr(args, "panel_source", "minute") == "daily":
        return _build_panel_daily(args)
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
        # TASK_848: expose per-asset notional fraction (default 0.1 = the create_jax_env_
        # historical default the search path already used → search byte-identical). The
        # policy-argmax round-trip passes the imprint run's 0.5/n_assets to match the env
        # the checkpoint was replayed in.
        position_size_pct=args.position_size_pct,
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
# TASK_850 — equivalence gate (trust anchor): daily source vs minute table
# =============================================================================

def _resolve_universe(db_path, start_date, end_date, specific_symbols, n_assets):
    """The comparison universe: the explicit --symbols list, else the SAME minute-coverage
    top-N the daily loader pins to (via the shared _minute_ranked_symbols resolver). Both
    sources are then loaded on this SAME universe so the gate tests equivalence rather than
    comparing disjoint asset sets."""
    if specific_symbols:
        return list(specific_symbols)
    return _minute_ranked_symbols(db_path, start_date, end_date, n_assets)


def _utc_day_bounds(start_date, end_date):
    """Whole-UTC-day epoch bounds [s, e] for the gate's close comparison. Parsed in UTC
    (calendar.timegm) and the end extended to the last second of the end day, so a boundary
    day's minute last-of-day is NOT clipped mid-day. This is deliberately UTC-consistent
    with the daily table's midnight-UTC row timestamps — distinct from the loader's
    _to_epoch (local-time) window bounds, which the daily loader keeps for byte-parity with
    the minute path (VectorBTTradingEnv parses dates in local time too)."""
    import calendar
    def p(s, default):
        if s is None:
            return default
        s = str(s)
        if s.isdigit():
            return int(s)
        return calendar.timegm(datetime.strptime(s, "%Y-%m-%d").timetuple())
    return p(start_date, 0), p(end_date, 9999999999) + 86399


def _close_by_date(db_path, table, symbols, s_ts, e_ts):
    """{symbol: {YYYY-MM-DD (UTC): last-of-day close}} for `symbols` from `table` over the
    UTC epoch window [s_ts, e_ts]. Last-write-wins per (symbol, UTC day) with rows ordered by
    timestamp = the calendar-daily close (the aggregation ohlcv_daily is built from — its
    midnight-UTC rows collapse to one per day, so the daily table reads as-is)."""
    conn = sqlite3.connect(db_path)
    ph = ",".join("?" * len(symbols))
    rows = conn.execute(
        f"SELECT a.id, a.symbol FROM assets a WHERE a.symbol IN ({ph})", symbols).fetchall()
    id_to_sym = {r[0]: r[1] for r in rows}
    if not id_to_sym:
        conn.close()
        return {}
    aids = list(id_to_sym)
    ph2 = ",".join("?" * len(aids))
    out = {sym: {} for sym in id_to_sym.values()}
    for aid, ts, close in conn.execute(
        f"""SELECT asset_id, timestamp, close FROM {table}
            WHERE asset_id IN ({ph2}) AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp""", (*aids, s_ts, e_ts)):
        d = datetime.utcfromtimestamp(int(ts)).strftime("%Y-%m-%d")
        out[id_to_sym[aid]][d] = float(close)
    conn.close()
    return out


def run_equivalence_gate(args):
    """TASK_850 acceptance (i). Two legs on the overlap window:
      CLOSE  — daily-source closes vs the minute table's calendar-daily aggregation, per
               asset over shared UTC dates; max relative diff must be <= --equiv-close-rtol.
      FLOOR  — teacher-floor net@21 on the daily panel vs the minute panel (both @1440);
               |Δ| must be <= 0.02. Deferred when either panel is too short at 1440 (the
               fixture minute slice is — the real-db leg is the planner's scheduled run).
    Divergence beyond tolerance => data-integrity FINDING: nonzero exit, NO artifact."""
    print("=" * 84)
    print(f"TASK_850 EQUIVALENCE GATE — db={_display_path(args.db)} window=[{args.start}..{args.end}]")
    if not _daily_table_exists(args.db):
        raise RuntimeError(
            f"--equivalence-gate: no `ohlcv_daily` table in {args.db}; nothing to compare "
            f"against. (Check gated on table presence — acceptance i.)")
    universe = _resolve_universe(args.db, args.start, args.end, args.symbols, args.n_assets)
    if not universe:
        raise RuntimeError(f"--equivalence-gate: empty universe for [{args.start}..{args.end}]")
    print(f"universe ({len(universe)}): {universe[:8]}{' ...' if len(universe) > 8 else ''}")

    # ---- CLOSE leg -------------------------------------------------------------
    s_ts, e_ts = _utc_day_bounds(args.start, args.end)
    minute_cd = _close_by_date(args.db, "ohlcv", universe, s_ts, e_ts)
    daily_cd = _close_by_date(args.db, "ohlcv_daily", universe, s_ts, e_ts)
    per_asset, worst = {}, {"symbol": None, "date": None, "rel": 0.0}
    shared_dates_total = 0
    for sym in universe:
        md, dd = minute_cd.get(sym, {}), daily_cd.get(sym, {})
        shared = sorted(set(md) & set(dd))
        if not shared:
            per_asset[sym] = {"shared_dates": 0, "max_rel": None}
            continue
        mx = 0.0
        for d in shared:
            denom = max(abs(dd[d]), 1e-8)
            rel = abs(md[d] - dd[d]) / denom
            if rel > mx:
                mx = rel
            if rel > worst["rel"]:
                worst = {"symbol": sym, "date": d, "rel": rel,
                         "minute": md[d], "daily": dd[d]}
        per_asset[sym] = {"shared_dates": len(shared), "max_rel": mx}
        shared_dates_total += len(shared)
    checked = [v["max_rel"] for v in per_asset.values() if v["max_rel"] is not None]
    close_max_rel = max(checked) if checked else None
    close_pass = (close_max_rel is not None and close_max_rel <= args.equiv_close_rtol)
    print("-" * 84)
    print(f"CLOSE leg: {len(checked)}/{len(universe)} assets with shared dates "
          f"({shared_dates_total} asset-days); tol rtol<={args.equiv_close_rtol:g}")
    if close_max_rel is None:
        print("  NO shared dates between minute and daily on this window — CLOSE leg N/A "
              "(minute table does not cover the window).")
    elif worst["symbol"] is None:
        print(f"  max relative close diff = {close_max_rel:.3e} (all shared dates exact) "
              f"→ {'PASS' if close_pass else 'FAIL'}")
    else:
        print(f"  max relative close diff = {close_max_rel:.3e} "
              f"(worst {worst['symbol']} {worst['date']}: minute={worst.get('minute')} "
              f"daily={worst.get('daily')}) → {'PASS' if close_pass else 'FAIL'}")

    # ---- FLOOR leg -------------------------------------------------------------
    floor_note = None
    floor_delta = tf_daily = tf_minute = None
    if not args.vector_weight_rebalance:
        floor_note = ("teacher-floor leg SKIPPED — pass --vector-weight-rebalance (the "
                      "weight-native ladder needs weight semantics; imprint.py:479-482).")
    else:
        # Pin BOTH FLOOR builds to the gate's already-resolved comparison universe — never
        # let the two sources' no-symbols selection diverge (that would compare teacher
        # floors on DISJOINT portfolios and pass/fail spuriously). Explicit pinning tests
        # equivalence rather than relying on the loaders happening to agree.
        #
        # TASK_852: that universe is minute-coverage-ranked and can include symbols with ZERO
        # ohlcv_daily rows in this window (LEDGER 850/eqgate: 37/64 on the real 2021-2025
        # overlap). The daily loader's own active-mask filter (above) now drops those rather
        # than densifying them — but left as-is here, args.n_assets would stay pinned at the
        # ORIGINAL 64, so create_daily_panel would pad the ~27 real columns back up to 64 by
        # CYCLING (duplicating) them, multiplying their effective exposure and comparing a
        # duplicate-heavy daily book against a genuinely-distinct 64-real-asset minute book —
        # not the coverage-symmetric comparison the gate exists to run. Intersect the universe
        # against real ohlcv_daily coverage FIRST and shrink n_assets to match, so both FLOOR
        # panels are built on the identical, coverage-safe symbol set (verified in isolation:
        # this collapses the bulk of the divergence — see run log).
        s_ts0, e_ts0 = _to_epoch(args.start, 0), _to_epoch(args.end, 9999999999)
        conn0 = sqlite3.connect(args.db)
        try:
            ph0 = ",".join("?" * len(universe))
            id_rows0 = conn0.execute(
                f"SELECT a.id, a.symbol FROM assets a WHERE a.symbol IN ({ph0})",
                universe).fetchall()
        finally:
            conn0.close()
        sym_to_id0 = {sym: aid for aid, sym in id_rows0}
        uni_ids0 = [sym_to_id0[s] for s in universe if s in sym_to_id0]
        covered_ids0 = _daily_coverage_ids(args.db, uni_ids0, s_ts0, e_ts0)
        covered_syms0 = {sym for sym, aid in sym_to_id0.items() if aid in covered_ids0}
        floor_universe = [s for s in universe if s in covered_syms0]
        if len(floor_universe) < len(universe):
            # MINOR-1 (flow-verifier): name the excluded symbols, not just the count — a
            # planner acting on this diagnostic (D20) needs to audit WHICH assets got dropped
            # without writing ad hoc SQL.
            excluded0 = [s for s in universe if s not in covered_syms0]
            print(f"FLOOR leg ACTIVE-MASK (TASK_852): {len(excluded0)}/{len(universe)} "
                  f"comparison-universe symbols have ZERO ohlcv_daily rows in "
                  f"[{args.start}..{args.end}] — excluded from BOTH panels for this leg "
                  f"(coverage-symmetric comparison); {len(floor_universe)} remain. Excluded: "
                  f"{excluded0[:20]}{' ...' if len(excluded0) > 20 else ''}")
        pd_ = pm_ = None
        if not floor_universe:
            floor_note = ("no symbol in the comparison universe has any ohlcv_daily coverage "
                          "in this window — FLOOR leg DEFERRED (active-mask filter left nothing "
                          "to compare).")
        else:
            args.symbols = list(floor_universe)
            args.n_assets = len(floor_universe)
            args.panel_source = "daily"
            try:
                pd_ = build_panel(args)
            except RuntimeError as e:
                pd_ = None
                floor_note = f"daily panel could not build @1440 ({e})"
            args.panel_source = "minute"
            if pd_ is not None:
                try:
                    pm_ = build_panel(args)
                except RuntimeError as e:
                    floor_note = (f"minute panel too short @1440 on this window ({e}) — FLOOR leg "
                                  f"DEFERRED to the planner's scheduled real-db run (the fixture "
                                  f"minute slice cannot build a 1440 panel).")
        if pd_ is not None and pm_ is not None:
            pm = float(args.bar_minutes)
            tf_d, _ = compute_teacher_floor(pd_[0], pd_[1], int(pd_[1].window_size),
                                            pd_[5]["n_decisions"], pm)
            tf_m, _ = compute_teacher_floor(pm_[0], pm_[1], int(pm_[1].window_size),
                                            pm_[5]["n_decisions"], pm)
            tf_daily, tf_minute = tf_d[21]["sharpe"], tf_m[21]["sharpe"]
            if tf_daily is not None and tf_minute is not None:
                floor_delta = abs(tf_daily - tf_minute)
    floor_pass = (floor_delta is not None and floor_delta <= 0.02)
    print("-" * 84)
    if floor_delta is not None:
        print(f"FLOOR leg: teacher-floor net@21 daily={tf_daily:+.4f} minute={tf_minute:+.4f} "
              f"|Δ|={floor_delta:.4f} (tol<=0.02) → {'PASS' if floor_pass else 'FAIL'}")
    else:
        print(f"FLOOR leg: DEFERRED — {floor_note}")

    print("=" * 84)
    # data-integrity finding: any leg that RAN and diverged beyond tolerance.
    diverged = []
    if close_max_rel is not None and not close_pass:
        diverged.append(f"CLOSE max_rel {close_max_rel:.3e} > {args.equiv_close_rtol:g}")
    if floor_delta is not None and not floor_pass:
        diverged.append(f"FLOOR |Δ| {floor_delta:.4f} > 0.02")
    if diverged:
        print("*** DATA-INTEGRITY FINDING — daily and minute sources DIVERGE beyond "
              "tolerance: " + "; ".join(diverged))
        print("*** STOP: the daily source does NOT reproduce the minute panel on the overlap. "
              "Do not use --panel-source daily until reconciled.")
        print("=" * 84)
        sys.exit(2)
    ran = [n for n, ok in (("CLOSE", close_max_rel is not None),
                           ("FLOOR", floor_delta is not None)) if ok]
    deferred = [n for n in ("CLOSE", "FLOOR")
                if (n == "CLOSE" and close_max_rel is None) or (n == "FLOOR" and floor_delta is None)]
    if not ran:
        # Fail-closed: nothing was actually compared (no minute↔daily overlap AND floor
        # deferred). Never certify equivalence on zero bytes — INCONCLUSIVE, non-zero exit.
        print("*** EQUIVALENCE GATE INCONCLUSIVE — NO leg ran (no minute↔daily overlap on "
              f"this window and FLOOR deferred: {floor_note}). Nothing was compared; this is "
              "NOT a pass. Choose a window where the minute table overlaps ohlcv_daily.")
        print("=" * 84)
        sys.exit(3)
    print(f"EQUIVALENCE GATE PASS — legs run: {ran}; legs deferred: {deferred}.")
    print("=" * 84)


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
    pf = r.get("panel_frame")
    if pf is not None:
        print(f"FIXED-FRAME (TASK_853): {pf['n_slots']} slots, {pf['n_active_slots']} ACTIVE / "
              f"{pf['n_inactive_slots']} INACTIVE / {pf['n_partial_slots']} partial "
              f"(provenance {pf['provenance_path']})")
        for c in pf["inactive_spans"]:
            print(f"   INACTIVE slot {c['slot']:>3} {c['symbol']:<16} "
                  f"[{c['span'][0]}..{c['span'][1]}] ({c['reason']})")
    _op = r['operator']
    if _op.get('mode') == 'policy-argmax':
        print("operator: policy-argmax (initial_inference -> per-asset argmax -> step_vector; "
              "no search/dither)")
    else:
        _scheme = _op['proposal_scheme']
        _extra = (f" temp={_op['proposal_temp']} explore_slots={_op['proposal_explore_slots']}"
                  if _scheme == 'policy-prior' else f" hold_bias={_op['hold_bias']}")
        _pw = _op.get('prior_weight', 1.0)                        # TASK_854
        _pw_str = f" prior_weight={_pw}" if _pw != 1.0 else ""
        print(f"operator: search[{_scheme}] K={_op['vector_k']} sims={_op['sims']}"
              f"{_extra}{_pw_str} deterministic(visit-argmax)")
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
    tf = r.get("teacher_floor")
    if tf is not None:
        print("-" * 84)
        print(f"on-window teacher floor ({tf['mapping']}, book_warm={tf['book_warm']}, "
              f"weight_env={tf['weight_rebalance_env']}):")
        for c in GATE_COSTS:
            t = r["tiers"][str(c)]
            lab = "gross" if c == 0 else f"net@{c}"
            print(f"  {lab:>8}: policy {_fmt(t['sharpe_full'])}  floor "
                  f"{_fmt(t['teacher_floor_inenv'])}  policy-floor "
                  f"{_fmt(t['corridor_position'])}")
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
    search_obj = build_search(model_b, args, args.n_assets) if args.operator == "search" else None
    label = {"path": _display_path(ckpt_dir),
             "name": "untrained_fresh_init(save->load roundtrip)"}
    result = evaluate_checkpoint(label, model_b, trainer_b.state.params, search_obj,
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
    ap.add_argument("--panel-source", choices=["minute", "daily"], default="minute",
                    dest="panel_source",
                    help="TASK_850: panel price source. 'minute' (default = byte-identical to "
                         "pre-850) resamples the dying `ohlcv` 1-min table via [::bar_minutes]; "
                         "'daily' reads the current `ohlcv_daily` table directly (same symbol "
                         "selection + warm/decision semantics, valid ONLY with --bar-minutes "
                         "1440) — unlocks the OOS window the minute table no longer covers")
    ap.add_argument("--panel-frame-from-checkpoint", default=None,
                    dest="panel_frame_from_checkpoint",
                    help="TASK_853: path to a JSON recording the checkpoint's ORDERED training "
                         "symbol frame ('frame_symbols' or 'symbols'). Builds the fixed n_assets-"
                         "slot daily panel in that EXACT order (slot k == asset k, matching the "
                         "positionally universe-locked encoder) and marks zero-coverage slots "
                         "INACTIVE via the env's existing price>0 active-mask (no fills/fees, "
                         "HOLD-only) — the OOS-read mechanic. Requires --bar-minutes 1440. "
                         "Default OFF = the minute/daily paths are byte-identical to HEAD")
    ap.add_argument("--n-assets", type=int, default=64)
    ap.add_argument("--checkpoint", action="append", default=[],
                    help="checkpoint dir (repeatable; jax_ckpt_fast/<name>/snapshot_ep<N>/ or final)")
    ap.add_argument("--checkpoint-list", default=None,
                    help="file with one checkpoint path per line (comparison-table mode)")
    ap.add_argument("--smoke", action="store_true",
                    help="fixture smoke: fresh-init untrained model + save->load roundtrip")
    ap.add_argument("--equivalence-gate", dest="equivalence_gate", action="store_true",
                    help="TASK_850 trust anchor: on the overlap window compare the daily source "
                         "against the minute table — per-asset closes (daily vs minute "
                         "calendar-daily aggregation) within tolerance, and (when both build) "
                         "the teacher-floor net@21 within +/-0.02. Divergence beyond tolerance "
                         "= data-integrity finding (nonzero exit, no artifact)")
    ap.add_argument("--equiv-close-rtol", type=float, default=1e-4, dest="equiv_close_rtol",
                    help="TASK_850 equivalence gate: max per-asset relative close diff "
                         "(daily vs minute calendar-daily aggregation) over shared dates")
    ap.add_argument("--smoke-ckpt-dir", default=None,
                    help="where the smoke writes its roundtrip checkpoint "
                         "(default: <this dir>/smoke_ckpt — gitignored, stable provenance)")
    ap.add_argument("--max-decisions", type=int, default=None,
                    help="cap decisions (smoke); None/negative = full timeline")
    ap.add_argument("--eval-seed", type=int, default=0)
    # ---- TASK_848: eval OPERATOR + on-window teacher floor + window sizing ----
    ap.add_argument("--operator", choices=["search", "policy-argmax"], default="search",
                    help="TASK_848: 'search' (default = byte-identical shipped Sampled-MuZero "
                         "vector search) or 'policy-argmax' (the 832 imprint-sanity direct "
                         "replay: initial_inference -> per-asset argmax -> step_vector, no "
                         "search/dither). policy-argmax bypasses the proposal/search path "
                         "entirely (orthogonal to the TASK_847 --vector-proposal-* knobs)")
    ap.add_argument("--decision-rule", choices=list(DECISION_RULES), default="argmax",
                    dest="decision_rule",
                    help="TASK_856: decision rule for the policy-argmax replay (policy-argmax "
                         "operator only). 'argmax' (default) = byte-identical to HEAD (per-asset "
                         "logit argmax); 'threshold' acts on the best non-HOLD action per asset "
                         "iff its marginal mass > --decision-tau; 'top-m' acts on the "
                         "--decision-m assets per step with the highest non-HOLD mass (forcing "
                         "each onto its BEST NON-HOLD action, so they always trade), HOLD "
                         "elsewhere")
    ap.add_argument("--decision-tau", type=float, default=0.5, dest="decision_tau",
                    help="TASK_856: threshold rule mass cutoff (act iff best non-HOLD marginal "
                         "mass > tau; 'threshold' rule only)")
    ap.add_argument("--decision-m", type=int, default=4, dest="decision_m",
                    help="TASK_856: top-m rule trade count per step (M assets with the highest "
                         "non-HOLD mass act; ~6%%x64 search trade rate; 'top-m' rule only)")
    ap.add_argument("--teacher-floor", dest="teacher_floor", action="store_true", default=False,
                    help="TASK_848: also walk 832's weight-native ladder teacher "
                         "(_make_weight_native_teacher_fn, same sentinels) on the SAME decision "
                         "rows/env and report teacher_floor_inenv + policy-vs-floor delta per "
                         "tier — the OOS read is meaningless without the floor on the same window")
    ap.add_argument("--position-size-pct", type=float, default=0.1,
                    dest="position_size_pct",
                    help="TASK_848: per-asset notional fraction for the eval env (default 0.1 = "
                         "the create_jax_env_historical default the search path already used; "
                         "the imprint round-trip uses 0.5/n_assets to match the imprint env)")
    # ---- run shapes: defaults = the 828 Stage-3 battery manifest (jobs.manifest) ----
    ap.add_argument("--model-size", default="small")
    ap.add_argument("--h-cycles", type=int, default=4)
    ap.add_argument("--l-cycles", type=int, default=6)
    ap.add_argument("--sims", type=int, default=32)
    ap.add_argument("--vector-k", type=int, default=16)
    ap.add_argument("--vector-hold-bias", type=float, default=3.0,
                    help="battery value 3.0 (TrainingConfig CLI default 0.0 is the known trap)")
    ap.add_argument("--vector-dither-scale", type=float, default=1.0)
    ap.add_argument("--vector-proposal-scheme", default="dithered-book",
                    choices=["dithered-book", "iid-beta", "policy-prior"],
                    help="TASK_847: policy-prior puts the policy-argmax vector in every "
                         "candidate set (the read-A inversion); default dithered-book = canon")
    ap.add_argument("--vector-proposal-temp", type=float, default=1.0,
                    dest="vector_proposal_temp",
                    help="TASK_847: temperature on the per-asset marginals for the "
                         "policy-prior sampled slots (policy-prior scheme only)")
    ap.add_argument("--vector-proposal-explore-slots", type=int, default=2,
                    dest="vector_proposal_explore_slots",
                    help="TASK_847: N of K candidates from the dither machinery under "
                         "policy-prior (default 2 of K=16; policy-prior only)")
    ap.add_argument("--prior-weight", type=float, default=1.0, dest="prior_weight",
                    help="TASK_854 (D21): scalar weight on the LEARNED policy prior in the "
                         "Gumbel candidate scoring/proposal. Default 1.0 = today, "
                         "bit-identical; 0.0 = pure-Q (E1 selection — candidate scoring "
                         "invariant to policy_logits; the last learned component vs the E1 "
                         "+5.29 anchor, LEDGER 832/E5-inversion2). The inversion re-run leg "
                         "uses --prior-weight 0.0.")
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
    ap.add_argument("--leaf-eval", type=str, default="model",
                    choices=["model", "true-env", "vhat-distilled"],
                    dest="leaf_eval",
                    help="TASK_851/857: search leaf evaluator — MUST match the flag the "
                         "checkpoint's evaluation is meant to measure (default 'model' = "
                         "canon learned leaf, bit-identical; 'true-env' replaces the "
                         "per-candidate leaf with a true-env truncated rollout, the E1 "
                         "+5.29 operator, LEDGER 849/leaf; 'vhat-distilled' (TASK_857) = the "
                         "DEPLOYABLE v̂ leaf — mechanically the model leaf, bit-identical to "
                         "'model' search, but flags the counterfactual-distilled read and is "
                         "net_dependent=true even at prior_weight=0)")
    ap.add_argument("--leaf-rollout-r", type=int, default=3, dest="leaf_rollout_r",
                    help="TASK_851: further HOLD steps in the true-env leaf rollout past "
                         "the candidate action (R; default 3; true-env only)")
    ap.add_argument("--no-leaf-vhat-tail", action="store_false", dest="leaf_vhat_tail",
                    help="TASK_851: disable the discounted v̂ tail at the true-env leaf "
                         "horizon (default ON; true-env only)")
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

    # TASK_848 fix pass — fail-loud BEFORE any panel build / JSON write:
    #  (1) strict eval-window parsing (no silent full-DB sentinel), and
    #  (2) --teacher-floor coherence: the weight-native ladder is only meaningful under the
    #      weight env (imprint.py:479-482 precedent; the D17 OOS read always passes it).
    _validate_eval_date(args.start, "start")
    _validate_eval_date(args.end, "end")
    if args.teacher_floor and not args.vector_weight_rebalance:
        raise RuntimeError(
            "--teacher-floor requires --vector-weight-rebalance so the weight-native ladder "
            "teacher walks WEIGHT semantics; without it the ladder's 0/1/2 target-weight codes "
            "are misread as SELL/HOLD/BUY orders (an incoherent floor). (imprint.py:479-482 "
            "precedent; the D17 OOS read always passes --vector-weight-rebalance.)")

    if args.equivalence_gate:
        run_equivalence_gate(args)
        return

    args.tag = args.tag or ("fixture-smoke" if (args.smoke and is_fixture)
                            else "fixture" if is_fixture else "real-panel")
    out_dir = args.out_dir or (EXP_DIR if is_fixture
                               else os.path.join(_REPO, "runs", "835_stage4"))
    os.makedirs(out_dir, exist_ok=True)

    corridor = load_corridor_constants()
    ext_windows = load_ext_windows()

    if args.smoke:
        result = run_smoke(args, corridor, ext_windows)
        # Fix pass: only the EXACT default-search config owns the canonical committed artifact
        # (the config that JSON actually represents: search operator, no teacher floor, weight
        # env OFF). ANY other flag combo — incl. a SEARCH smoke run with --teacher-floor /
        # --vector-weight-rebalance (the D17 OOS workflow always passes --teacher-floor) — writes
        # a suffixed file so a verification smoke can never clobber the checked-in parity baseline.
        is_default_search = (args.operator == "search" and not args.teacher_floor
                             and not args.vector_weight_rebalance)
        if is_default_search:
            out_name = "results_fixture_smoke.json"
        else:
            suffix = args.operator
            if args.teacher_floor:
                suffix += "_tf"
            if args.vector_weight_rebalance:
                suffix += "_wr"
            out_name = f"results_fixture_smoke_{suffix}.json"
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
        check_checkpoint_arch(path, args)                    # fail-loud arch guard (acceptance iii)
        model, trainer = build_model_and_trainer(args, args.n_assets, init_seed=0)
        load_checkpoint_into(trainer, path)
        search_obj = build_search(model, args, args.n_assets) if args.operator == "search" else None
        result = evaluate_checkpoint(label, model, trainer.state.params, search_obj,
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
