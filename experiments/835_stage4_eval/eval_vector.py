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

TASK_861 (D26) — ``--collect-chosen-vectors <out>`` is a PURE RECORDER bolted onto the walk:
it writes the per-decision action vector the operator CHOSE on the net@21 rollout (plus the
realized book, diagnostics) and a provenance header (checkpoint, full operator block, db,
window, panel-frame provenance, EnvConfig fingerprint, decision count). It draws no RNG and
issues no extra env/model call, so the walk and this module's scorecard JSON are byte-identical
with and without the flag. The consumer is ``experiments/832_linear_imprint/imprint.py
--teacher-vectors``, which re-gates the whole header before imprinting on the stream.

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
import dataclasses
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, date, timezone

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
# TASK_870: the ONE _ro_connect implementation now lives in mcts_trader/utils/db.py (this
# module used to define its own; see the fix-pass note below for why it moved there instead
# of the canon env loader importing this module).
from mcts_trader.utils.db import _ro_connect, SQLITE_TIMEOUT_S

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
FIXTURE_DB = os.path.join(_REPO, "fixtures", "trading_fixture.db")
PROTECTED_DB = os.path.join(_REPO, "trading_data.db")

CORRIDOR_JSON = os.path.join(_REPO, "experiments", "830_corridor", "results_real_panel.json")
EXT_WINDOW_NAMES = ("bear22w", "bull23w", "y2024", "recent")
EXT_WINDOW_DIR = os.path.join(_REPO, "runs", "831_ext")

GATE_COSTS = (0, 21)          # bp/side cost tiers — gross + net@21 (fee16+slip5), the 830/831 SPEC tiers
FEE_BPS, SLIP_BPS = 16.0, 5.0
HOLD = 1
N_ACTIONS_PER_ASSET = 5
HOLD_COLLAPSE_PCT = 99.0      # the 821 failure signature: pct_hold pinned >99
DECISION_RULES = ("argmax", "threshold", "top-m")   # TASK_856: policy-replay decision rules

# TASK_861 — chosen-vector collection (the search-teacher imprint's target stream).
# TIER PIN: the 832 imprint recipe injects the COST-HONEST trajectory (imprint.py
# ``net_tier = GATE_COSTS[-1]``) and the +8.024 teacher read of record is the net@21 walk
# (LEDGER 832/E5-inversion3), so the recorder pins the net@21 rollout. Recording the gross
# tier as well would emit a SECOND, different action stream into one file and invite an
# accidental cross-tier imprint.
COLLECT_TIER_BP = 21
CHOSEN_VECTORS_SCHEMA = "861/chosen-vectors/v1"

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
# fix pass — READ-ONLY DB HANDLES (project invariant 2, source of record)
# =============================================================================
# Every query this harness issues is a SELECT, but a plain ``sqlite3.connect(path)`` opens the
# file READ-WRITE, and that is not a cosmetic difference on the 24GB irreplaceable
# ``trading_data.db``:
#   * HOT-JOURNAL RECOVERY REWRITES THE FILE ON THE FIRST *READ*. If a writer died mid-txn (the
#     fetcher, tools/fetch_daily_kraken.py:360, writes this same file; the host has a documented
#     hard-crash history — MEMORY workstation-syncflood), a rollback journal is left next to the
#     db. The FIRST read through a READ-WRITE handle plays that journal back into the db and
#     unlinks it — no write SQL involved. MEASURED on a copy carrying a hot journal: the file's
#     sha256 AND size changed and the journal disappeared, while a ``mode=ro`` handle refused the
#     recovery and left the file byte-identical.
#   * A READ-WRITE handle CREATES an empty db at any path that does not exist, so a typo'd or
#     symlinked ``--db`` silently produces a 0-byte file and then fails with a confusing
#     "no such table" instead of "no such file".
# ``mode=ro`` closes both, and cannot change any SELECT's result, so no numeric moves.
# ``_ro_connect`` also carries the repo's documented 60s busy timeout (see SQLITE_TIMEOUT_S) and
# refuses to be pointed at the protected source of record for anything but reading, resolving
# symlinks first (the ``_guard_out`` rule, experiments/840_weight_rebalance/acceptance_probe.py
# :373-391 — abspath does NOT resolve an innocuously named symlink to trading_data.db).
#
# SCOPE, EXACTLY (fix pass 3 — TASK_870 closed the gap fix pass 2 recorded below; kept for
# history). fix pass 2 read: ``_ro_connect`` covers every DB handle THIS harness and the
# 832/863/865 framed loaders open, but NOT the CANON UNPINNED panel path
# (``mcts_trader/env/vectorbt_env.py``'s two ``_load_database_prices*`` methods, reached via
# ``build_panel`` -> ``create_jax_env_historical`` on a DEFAULT unpinned eval/training run).
# TASK_870 routed those two sites through the SAME ``_ro_connect`` (now homed in
# ``mcts_trader/utils/db.py``, imported above, so this harness and the canon env loader share
# one implementation instead of ``vectorbt_env.py`` growing its own copy or importing this
# module — the latter would be circular: this module -> env_factory -> vectorbt_env).


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
    (digits pass through; naive dates resolve at UTC midnight).

    TASK_869 (D31 (a3) scope 1): the parse is now UTC, not local-timezone — a benchmark
    window's extent no longer depends on which machine ran it (measured pre-fix: a
    308-row-class CEST offset on the OOS window; the reason several counts differed by one
    between parties). Same resolution rule as mcts_trader/data/history.py and
    trader/backtest.py, which were always UTC. Window SEMANTICS are otherwise unchanged:
    ``end`` still resolves to MIDNIGHT of the end date (inclusivity stays as-is), only the
    timezone moved. Artifacts record the resolved ``window_utc`` epochs so cross-machine
    comparisons compare epochs, not date strings.

    FIX (run-25 panel): a TZ-AWARE ISO-8601 input (``2025-06-01T00:00:00+05:00``) keeps its
    explicit offset — it is converted to the correct instant (``astimezone(utc)``), never
    silently re-labeled UTC. Only NAIVE inputs are labeled UTC."""
    if v is None:
        return default
    if isinstance(v, (int, float)):
        return int(v)
    s = str(v)
    if s.isdigit():
        return int(s)
    try:
        return int(datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
    except ValueError:
        try:
            dt = datetime.fromisoformat(s)
            dt = dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
        except Exception:
            return default


def _window_utc(args):
    """TASK_869 (D31 (a3) scope 1): the artifact's RECORD of how --start/--end resolved —
    ``[start_epoch, end_epoch]`` under the SAME rule the loader applied (``_to_epoch`` with
    the loader's own sentinels). Cross-machine/cross-boundary comparisons compare EPOCHS,
    not date strings: two machines running the same command now produce identical windows,
    and an artifact written before the UTC re-base is distinguishable from one written
    after by comparing this field against the date strings' UTC resolution."""
    return [_to_epoch(args.start, 0), _to_epoch(args.end, 9999999999)]


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
    caller (len must equal prices.shape[0]). Returns (row_ts int64 array, loaded_symbols).

    TASK_868 (D31 a1): mirrors the canon loader's ALIAS UNION — each requested symbol
    resolves to every ``assets`` row whose normalized form matches (per-id coverage
    counts grouped and summed in the ranked branch), and the axis is the DISTINCT
    timestamps over the unioned member ids. On a DB whose duplicate spellings carry no
    rows in the window (e.g. the fixture) this is bit-identical to the pre-868 mirror."""
    from mcts_trader.data.asset_aliases import (
        ranked_minute_universe, resolve_requested_symbols)
    conn = _ro_connect(db_path)
    cur = conn.cursor()
    s_ts = _to_epoch(start_date, 0)
    e_ts = _to_epoch(end_date, 9999999999)
    if specific_symbols:
        all_asset_rows = cur.execute("SELECT id, symbol FROM assets").fetchall()
        resolution, _unknown = resolve_requested_symbols(all_asset_rows, specific_symbols)
        # preserve order; skip missing (loader parity — the canon loader warns+skips)
        asset_ids = sorted({aid for r in resolution for aid in r["asset_ids"]})
        symbols = [r["symbol"] for r in resolution]
    else:
        ranked = ranked_minute_universe(cur, s_ts, e_ts, n_assets)
        asset_ids = sorted({aid for g in ranked for aid in g["asset_ids"]})
        symbols = [g["symbol"] for g in ranked]
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
    if bar_minutes > 1:                        # env_factory.py:409 stride, verbatim
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
    conn = _ro_connect(db_path)
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
    no-symbols branch). The daily source and the equivalence gate BOTH route their
    no-symbols universe through this single resolver, so the daily panel is scored on the
    SAME asset set as the minute baseline (on the real DB the majors still hold the deepest
    ohlcv history even though it ends ~Oct-2025, so they still rank top-N). Fail-loud if the
    minute table is absent/empty — never silently rank over ohlcv_daily (a disjoint universe
    that corrupts the OOS benchmark).

    TASK_868 (D31 a1): the ranking is GROUPED by normalized symbol (per-id in-window
    counts summed across alias twins), mirroring the canon loader's repaired ranking —
    a split series now ranks on its whole history instead of each twin ranking
    separately (D31: "the split slots' coverage ranking is itself distorted today").
    Bit-identical to the pre-868 resolver wherever no alias twin carries window rows."""
    from mcts_trader.data.asset_aliases import ranked_minute_universe
    conn = _ro_connect(db_path)
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
        ranked = ranked_minute_universe(conn, s_ts, e_ts, n_assets)
        syms = [g["symbol"] for g in ranked]
    finally:
        conn.close()
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
    ones. Opens its own connection, STRICTLY read-only (``_ro_connect``; mirrors
    ``_daily_table_exists`` style)."""
    if not asset_ids:
        return set()
    conn = _ro_connect(db_path)
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
    conn = _ro_connect(db_path)
    cur = conn.cursor()
    # Window bounds via _to_epoch — intentionally identical to the minute path
    # (VectorBTTradingEnv parses dates the same way), so the daily and minute panels apply
    # the SAME window semantics. TASK_869: that shared parse is now UTC on both paths (was
    # local-time), so the panels are machine-timezone-invariant; inclusivity unchanged.
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


# Bounds on the provenance METADATA that gets copied into an artifact (fix pass). It used to be
# copied VERBATIM, so a provenance file could put anything into the read of record: a float('nan')
# or float('inf') is emitted by json.dump as the bare tokens NaN/Infinity, which makes the artifact
# NOT strict JSON (json.loads in another language, or with parse_constant refusing, fails on it); a
# 555 KB blob was accepted unbounded into every window row; and a 2000-deep nesting raised an
# UNCAUGHT RecursionError from inside json.dump, i.e. a crash with no named cause after the whole
# panel had been built. The real provenance meta is 245 bytes / depth 2, so these bounds are ~130x
# and 4x headroom, not a constraint on anything real.
PROV_META_MAX_BYTES = 32_768
PROV_META_MAX_DEPTH = 8
PROV_META_MAX_KEYS = 64


def provenance_meta_block(prov, path):
    """The provenance metadata that goes into the artifact: everything except the frame itself,
    TYPE-CHECKED and BOUNDED. Refuses (named, before the panel build) a non-finite float, a
    non-JSON-safe value, an over-deep or over-large blob, rather than letting it reach json.dump."""
    meta = {k: v for k, v in prov.items() if k not in ("frame_symbols", "symbols")}
    if len(meta) > PROV_META_MAX_KEYS:
        raise RuntimeError(
            f"--panel-frame-from-checkpoint: provenance {path} carries {len(meta)} metadata keys "
            f"(max {PROV_META_MAX_KEYS}). This block is copied into the read of record; it is "
            f"provenance, not a data dump.")

    def check(v, depth, where):
        if depth > PROV_META_MAX_DEPTH:
            raise RuntimeError(
                f"--panel-frame-from-checkpoint: provenance {path} nests deeper than "
                f"{PROV_META_MAX_DEPTH} at {where}. Copied verbatim into the artifact, a deeply "
                f"nested blob raises an UNCAUGHT RecursionError inside json.dump after the whole "
                f"panel has been built — refused here instead, by name.")
        if v is None or isinstance(v, (bool, int, str)):
            return
        if isinstance(v, float):
            if v != v or v in (float("inf"), float("-inf")):
                raise RuntimeError(
                    f"--panel-frame-from-checkpoint: provenance {path} has a NON-FINITE number "
                    f"({v!r}) at {where}. json.dump would emit the bare token NaN/Infinity and the "
                    f"artifact would no longer be strict JSON (a conforming reader rejects it).")
            return
        if isinstance(v, list):
            for i, x in enumerate(v):
                check(x, depth + 1, f"{where}[{i}]")
            return
        if isinstance(v, dict):
            for k, x in v.items():
                if not isinstance(k, str):
                    raise RuntimeError(
                        f"--panel-frame-from-checkpoint: provenance {path} has a non-string key "
                        f"{k!r} at {where}.")
                check(x, depth + 1, f"{where}.{k}")
            return
        raise RuntimeError(
            f"--panel-frame-from-checkpoint: provenance {path} has a value of unsupported type "
            f"{type(v).__name__} at {where}; only JSON scalars/lists/objects are carried into the "
            f"artifact.")

    try:
        check(meta, 1, "provenance_meta")
    except RecursionError:
        raise RuntimeError(
            f"--panel-frame-from-checkpoint: provenance {path} is nested too deeply to validate "
            f"(RecursionError). Refusing to copy it into the artifact.")
    n_bytes = len(json.dumps(meta, allow_nan=False, separators=(",", ":")))
    if n_bytes > PROV_META_MAX_BYTES:
        raise RuntimeError(
            f"--panel-frame-from-checkpoint: provenance {path} metadata serialises to {n_bytes} "
            f"bytes (max {PROV_META_MAX_BYTES}). It is copied into EVERY artifact that reads this "
            f"frame; keep it to provenance.")
    return meta


# ---------------------------------------------------------------------------------------------
# D30 (ruled 2026-07-30) — the stale-tail contract, UNIFORM across BOTH pin paths, and the
# breadth refusal that keeps a half-dead panel from producing a publishable number.
# ---------------------------------------------------------------------------------------------
STALE_TAIL_RULE = (
    "a STALE-TAIL slot's LAST REAL close precedes the window end (delisting / ticker migration). "
    "Its trailing panel rows are FORWARD-FILLED at that last real close — the fill rule the "
    "shipped daily pin has always applied, made uniform across both pin paths by D30 (ruled "
    "2026-07-30, opt (i)). The alternative — ZEROING the tail and letting price_valid force-HOLD "
    "it — was measured to FABRICATE PnL: force-HOLD gates OPENING, not VALUATION, so zeroing "
    "marks a position open at the boundary against price 0 (a long wiped, a short handed a 100% "
    "windfall) and freezes it there — MEASURED at 75,002.50 / 124,997.50 for a long/short book "
    "on flat prices at fee 0 where any book must end at exactly 100,000.00, while the forward "
    "fill is PV-exact to the cent. WHAT A FORWARD FILL CAN AND CANNOT DO, stated exactly (fix pass — the "
    "earlier blanket claim was FALSE for one of the two cases): (1) on a TRAILING span the series "
    "is constant to the window end, so a round trip nets exactly 0 at fee 0 and is a strict fee "
    "drag above it — it cannot fabricate PnL and is biased AGAINST the bar being graded; (2) on an "
    "INTERIOR span (a hole that RESUMES before the window end) the same fill is NOT harmless: the "
    "book is marked and can be opened/rebalanced at a frozen price no market offered, and the "
    "whole accumulated gap move is then delivered in ONE bar at the re-anchor. That is real, "
    "sign-indeterminate PnL — MEASURED on the 861 leg-1 window at re-anchor jumps of +64.52% "
    "(ETHBTC), +33.88% (BTCCAD), +32.69% (BTCGBP), +24.29% (BTCCHF), -7.76% (/USDTEUR). Choosing "
    "what to DO about (2) — mask / refuse / accept-with-disclosure — was left OPEN here (this "
    "loader measured and disclosed it so the ruling could be made on numbers) and is now RULED: "
    "DECISIONS D31 opt (a) — repair the ingest identity split at load time (a1) and mask the "
    "residual frozen cells causally (a2); see the TASK_868 AMENDMENT at the end of this rule. "
    "So: every frozen row — interior or trailing — is disclosed, "
    "and a slot with a frozen span is still counted inside n_active_slots, is weight-eligible, and "
    "is NOT listed in inactive_slots or (for a tail-only slot) partial_slots. Consumers COULD "
    "previously mistake a frozen price for a live market — an interior span left no trace at all "
    "(no stale_tail, no partial, covered_rows == n_rows), which is the defect the frozen_spans "
    "keys close. TRUE LIQUIDATION (close at the last real close, then mask) is PARKED as its own "
    "task (D30 opt (ii)) — it needs a state-aware forced close in walk_env AND in eval's search "
    "rollout. Asserted by the PV-conservation leg of "
    "experiments/865_minute_frame_pin/smoke_minute_frame_pin.py. TASK_868 AMENDMENT (D31 opt "
    "(a2), ruled 2026-08-01): on the MINUTE paths (this pinned minute loader AND the canon "
    "unpinned minute loader) a trailing frozen row is now force-HELD by the causal "
    "open-gate — at decision time a trailing gap and an interior hole are indistinguishable "
    "without look-ahead into whether the gap later resumes, and the ruling requires the mask to "
    "be derived from data <= the decision bar alone. VALUATION is untouched everywhere: the "
    "forward-fill stands and stays PV-exact. The DAILY pin is NOT masked (gate (v): daily paths "
    "bit-identical) — there these rows remain tradeable on the constant series, as ruled in D30.")

# TASK_868 (D31 opt (a1)) — the alias-union contract, emitted into artifacts.
ALIAS_UNION_RULE = (
    "ALIAS UNION (DECISIONS D31 opt (a1)): each requested/recorded symbol resolves to EVERY "
    "assets row whose NORMALIZED spelling matches (symbol.upper().replace('/', '') — slash-only, "
    "deliberately NOT XBT->BTC folding), and the group's ohlcv rows are unioned into the slot's "
    "single series at load time (NO DB writes — the ingest identity split is repaired at the "
    "loader). Where aliased series overlap in time the closes must AGREE (rel <= 1e-12) or the "
    "load REFUSES, naming both asset_ids and the first disagreeing timestamp — never average, "
    "never prefer silently; agreeing duplicate timestamps dedupe. Per-slot usage is disclosed "
    "(asset_ids / db_symbols / per_id_rows / overlap_agreement).")

# TASK_868 (D31 opt (a2)) — the causal open-gate contract, emitted into artifacts.
OPEN_GATE_RULE = (
    "CAUSAL OPEN-GATE (DECISIONS D31 opt (a2)): a decision cell whose slot printed NO real close "
    "anywhere inside its feature bar is force-HELD (the shipped price_valid force-HOLD mechanic), "
    "on the pinned AND canon unpinned minute loaders uniformly. The mask is CAUSAL by construction "
    "— bar r's value reads only prints inside bar r, so the mask at decision t is identical "
    "whether or not rows after t exist (no look-ahead into whether a gap later resumes; that is "
    "also why TRAILING gaps are gated — at decision time a delisting and an interior hole are "
    "indistinguishable). WHAT THE FORCE-HOLD COVERS, stated exactly (regression-panel amendment): "
    "the channel holds the WHOLE action, not only the open — on a masked cell no position can "
    "OPEN and none can CLOSE or rebalance either, so a book already holding the slot is CARRIED "
    "through the frozen span at the D30 forward-filled valuation (PV-exact to the cent — the "
    "PV-conservation smoke leg, both books, mask ON) and realizes its PnL at the NEXT REAL print. "
    "That is the sound semantics, not an oversight: a close at a frozen price would realize PnL "
    "against a price no market offered — the same fabrication class the mask exists to remove — "
    "and a state-aware close-at-the-boundary channel is D30 opt (ii) TRUE LIQUIDATION, which the "
    "ruling keeps PARKED. 'The mask only gates OPENING' in the task text is the contrast with "
    "VALUATION (the fill still values every row), not a claim that closes execute on frozen "
    "cells. Conservative by construction: a masked cell cannot trade, so it can only reduce "
    "measured performance, never flatter it.")

# D30 rider (b): the minute framed loader used to refuse ONLY when EVERY slot was uncovered, so a
# 4-of-8-inert fixture run exited 0 with a publishable net@21, and the real 2021-01-01..2022-01-01
# walk-forward fold (50 of 64 slots live) would have run green — reintroducing the exact 64->33
# breadth failure D29 exists to prevent. The default is a FRACTION of the frame's slots so that one
# number can serve a 64-slot real panel and be explicitly overridden by a small fixture frame:
#   * the 861 leg-1 pin carries rows on 64/64 slots  = 1.000 -> PASSES with margin
#   * the 2021-2022 WF fold carries rows on 50/64    = 0.781 -> REFUSES (needs an explicit override)
#   * the 8-slot fixture frame is 6/8 by design      = 0.750 -> REFUSES, so the fixture smoke opts
#     out explicitly (--require-active-slots 0) instead of the default being weakened for everyone
# 0.9 is the loosest round value that still refuses the 50/64 fold; anything >= 0.79 works, and a
# value > 1 is read as an ABSOLUTE slot count for an operator who wants to pin the number.
#
# READ THIS BEFORE QUOTING "daily-pinned paths are bit-identical" (fix pass 2, planner-accepted
# remainder MINOR-2). A FRACTION rounds UP, so on any frame with <= 10 slots this default demands
# ALL of them: ceil(0.9 * 8) = 8 of 8, ceil(0.9 * 10) = 9 of 10. Consequence, stated plainly rather
# than left to be rediscovered: the SHIPPED 863 canonical daily-pinned fixture invocation — the
# task864-parity-baseline-hygiene one — now REFUSES at the default, because that frame is 6/8
# covered by design. It passes only under `--require-active-slots 0`, an argv that does not exist
# at that baseline tag. So acceptance gate (i)'s "daily-pinned paths bit-identical" holds ONLY
# under that added flag; it is NOT unconditional parity. The direction is fail-closed (a partly
# inert panel must be asked for explicitly) and the planner ACCEPTED it, so the default is
# deliberately NOT weakened here — but no reader should take the parity claim as unqualified.
REQUIRE_ACTIVE_SLOTS_DEFAULT = 0.9


class RequireActiveSlotsError(ValueError, argparse.ArgumentTypeError):
    """Refusal for an out-of-domain ``--require-active-slots``.

    Subclasses BOTH so the message survives every route: argparse checks
    ``ArgumentTypeError`` FIRST and prints ``str(exc)`` verbatim (a plain ValueError would be
    replaced by argparse's generic "invalid ... value: '-1'", i.e. refused but NOT by name), while
    in-process callers can still catch it as a ValueError."""


def parse_require_active_slots(v):
    """Validate/normalise ``--require-active-slots``. THE one place the domain is defined, used
    both as the argparse ``type=`` (so the CLI refuses before anything runs) and by
    ``_guard_active_slots`` (so an in-process caller cannot bypass it).

    DOMAIN: ``0`` (disable) | a FRACTION in (0, 1] | an INTEGRAL slot count >= 1.
    Everything else is refused BY NAME. Before this validator the guard FAILED OPEN on nonsense:
    ``-1``, ``-0.0``, ``False``, ``1e-300`` and ``"0"`` all admitted a 50/64 panel while the
    artifact recorded a floor as though one were in force (``floor = max(1, ...)`` collapses every
    non-positive or infinitesimal request to "at least 1 active slot"), and ``nan`` / ``inf`` blew
    up inside ``np.ceil`` as a bare ValueError/OverflowError instead of a named refusal. Returns a
    float (0.0 == disabled)."""
    if isinstance(v, bool):
        raise RequireActiveSlotsError(
            f"--require-active-slots: got the boolean {v!r}. This is a breadth FLOOR, not a "
            f"toggle — pass 0 to disable it, a fraction in (0,1], or an integral slot count >= 1.")
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise RequireActiveSlotsError(
            f"--require-active-slots: {v!r} is not a number. Pass 0 (disable), a fraction in "
            f"(0,1], or an integral slot count >= 1.")
    if f != f or f in (float("inf"), float("-inf")):
        raise RequireActiveSlotsError(
            f"--require-active-slots: {v!r} is not finite. Pass 0 (disable), a fraction in (0,1], "
            f"or an integral slot count >= 1.")
    if f == 0.0:                    # normalises -0.0 too, which is otherwise recorded as "-0.0"
        return 0.0
    if f < 0.0:
        raise RequireActiveSlotsError(
            f"--require-active-slots: {v!r} is negative. A negative floor is NOT 'disabled' — it "
            f"used to collapse to 'at least 1 active slot' and pass a 1-of-64 panel while the "
            f"artifact recorded {v!r} as though a floor were in force. Pass exactly 0 to disable.")
    if f <= 1.0:
        if f < 1e-6:
            raise RequireActiveSlotsError(
                f"--require-active-slots: {v!r} is a positive fraction so small it cannot refuse "
                f"anything (it rounds up to a floor of 1 slot on any frame) while recording a "
                f"floor as though one were in force. Pass exactly 0 to disable the check.")
        return f
    if f != float(int(f)):
        raise RequireActiveSlotsError(
            f"--require-active-slots: {v!r} is > 1, so it is read as an ABSOLUTE slot count, and a "
            f"fractional slot count is meaningless. Pass an integer (e.g. 58), or a fraction <= 1.")
    return f


def _resolve_require_active(args):
    """``--require-active-slots`` off the args namespace, VALIDATED. UNSET (attribute absent, or
    None) means the shared default, NOT 'disabled' — only an explicit 0 disables the refusal.
    (``_guard_active_slots`` refuses None outright for exactly this reason: the two functions must
    not give the same value opposite meanings.)"""
    v = getattr(args, "require_active_slots", None)
    return REQUIRE_ACTIVE_SLOTS_DEFAULT if v is None else parse_require_active_slots(v)


def _guard_active_slots(coverage, frame_symbols, require_active_slots, start_date, end_date):
    """D30 rider (b) — fail LOUD on the result-producing path when too many frame slots are wholly
    inert. ``require_active_slots``: exactly 0 disables; a fraction in (0,1] is a FRACTION of the
    frame's slots; an integral value > 1 is an absolute slot count; anything else is REFUSED by
    name (``parse_require_active_slots``). ``None`` is NOT accepted here — it means "unset, use the
    shared default" at the args level (``_resolve_require_active``), and letting it also mean
    "disabled" one function over is how a floor silently stops existing. 'Active' is the loader's
    own definition (a slot with at least one real close in the window, i.e. status != 'inactive');
    a stale-tail or mid-window-listing slot counts as active because it really traded."""
    if require_active_slots is None:
        raise RuntimeError(
            "_guard_active_slots: require_active_slots is None. None means 'unset => use "
            "REQUIRE_ACTIVE_SLOTS_DEFAULT' (resolve it with _resolve_require_active), never "
            "'disabled' — pass an explicit 0 to disable the breadth refusal.")
    req = parse_require_active_slots(require_active_slots)
    if req == 0.0:
        return
    n_slots = len(frame_symbols)
    inactive = [c for c in coverage if c["status"] == "inactive"]
    n_active = n_slots - len(inactive)
    floor = int(np.ceil(req * n_slots)) if req <= 1.0 else int(np.ceil(req))
    # FIX PASS 2 — an ABSOLUTE floor above the frame's own slot count used to be CLAMPED
    # (``min(floor, n_slots)``), so ``--require-active-slots 65`` on a 64-slot frame silently became
    # 64 and a run that asked for something impossible was graded against a floor it never
    # requested, with the impossible value recorded verbatim in the artifact. Refuse by name: the
    # operator asked for a breadth this frame cannot supply at any coverage, which is a
    # configuration error, not a run to soften. (Fractions are unaffected — ceil(f*n) <= n for
    # f <= 1 — and are the form every shipped invocation uses.)
    if req > 1.0 and floor > n_slots:
        raise RequireActiveSlotsError(
            f"--require-active-slots {require_active_slots}: that is an ABSOLUTE floor of {floor} "
            f"active slots, but this frame has only {n_slots} slots, so it can NEVER be satisfied. "
            f"It used to be clamped to {n_slots} silently — the run then passed a floor it was "
            f"never asked to enforce while the artifact recorded {require_active_slots}. Pass a "
            f"value <= {n_slots}, or a fraction in (0,1] of the frame.")
    floor = max(1, floor)
    if n_active >= floor:
        return
    dead = ", ".join(f"slot {c['slot']} {c['symbol']}" for c in inactive[:12])
    raise RuntimeError(
        f"--require-active-slots {require_active_slots}: only {n_active} of {n_slots} frame slots "
        f"have ANY real price in [{start_date}..{end_date}] ({100.0 * n_active / max(n_slots, 1):.1f}"
        f"%), below the required floor of {floor} slots. The remaining {len(inactive)} slot(s) are "
        f"wholly INACTIVE (flat-zero columns, force-HOLD for the entire window), so this run would "
        f"measure a {n_active}-asset book while its frame, its artifact and every per-cell scalar "
        f"claim {n_slots} — the 64->33 breadth confound D29 exists to prevent, and it exits 0 with "
        f"a publishable number unless refused HERE (the all-slots-inactive refusal is not enough). "
        f"Inactive: {dead}{' ...' if len(inactive) > 12 else ''}. Widen --start/--end, use a frame "
        f"whose universe is live on this window, or pass --require-active-slots explicitly (0 "
        f"disables the check; a value <=1 is a fraction of the frame, >1 an absolute slot count) "
        f"to record that a partly-inert panel is intended.")


def load_daily_prices_framed(db_path, start_date, end_date, frame_symbols,
                             require_active_slots=REQUIRE_ACTIVE_SLOTS_DEFAULT):
    """Fixed-frame daily close matrix: ONE column per recorded frame symbol IN ORDER (no drop,
    no pad-by-cycling — the identity-preserving contrast to ``load_daily_prices``). Zero-coverage
    frame symbols are KEPT as flat-zero columns (the env's price>0 mask renders them inactive).
    Returns (prices_np (T, n_frame), frame_symbols, row_ts int64 (T,), coverage) where
    ``coverage`` lists per-slot status/spans. Fail-loud: missing daily table, an unknown symbol
    (not in ``assets``), or an empty window / all-slots-inactive (task scope 4), PLUS the D30
    rider (b) breadth refusal (``require_active_slots``).

    FIX PASS — rider (b) applies HERE TOO. It shipped minute-only, so the DAILY pin had no breadth
    floor at all: a daily-pinned fixture run at 5/8 active slots exited 0 with a full scorecard and
    without even a ``require_active_slots`` key, while the identical frame on the minute source was
    refused. The real daily pin is 33/64 = 0.52, and the manifest's own note says flipping the
    source to daily is a one-word change — so the floor has to hold on both paths or it does not
    hold. D30's ruling is explicit that the semantics are UNIFORM across both pin paths."""
    if not _daily_table_exists(db_path):
        raise RuntimeError(
            f"--panel-frame-from-checkpoint: no `ohlcv_daily` table in {db_path}. The fixed-frame "
            f"panel reads that table directly; it is absent here.")
    s_ts = _to_epoch(start_date, 0)
    e_ts = _to_epoch(end_date, 9999999999)
    conn = _ro_connect(db_path)
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
            real_rows = raw_nonzero[:, col]
            first = int(np.argmax(real_rows))
            entry = {"slot": col, "symbol": sym, "status": "active",
                     "covered_rows": n_nz, "n_rows": n_steps}
            if first > 0:                       # listed mid-window: leading rows had no data
                entry["status"] = "partial"
                entry["first_covered_ts"] = int(timestamps[first])
                entry["leading_inactive_rows"] = first
            # D30 amendment 4 — DISCLOSURE PARITY, additive only. This loader has always
            # forward-filled a mid-window delisting to the window end at a frozen price (the loop
            # below), and under D30 opt (i) that fill is the SANCTIONED semantics on both pin paths,
            # so the NUMERICS do not move. What was missing is that such a slot appeared in NEITHER
            # inactive_slots NOR partial_slots while sitting inside n_active_slots and staying
            # weight-eligible (measured: fixture frame slot 5 FTM/USD, 91 of 181 rows fabricated).
            # It now discloses on its own axis. Keys are APPENDED, so every pre-existing field of a
            # daily-pinned artifact keeps its value and its position.
            last_real = int(np.max(np.nonzero(real_rows)[0]))
            if last_real < n_steps - 1:
                entry["stale_tail"] = True
                entry["last_real_ts"] = int(timestamps[last_real])
                entry["last_covered_row"] = last_real
                entry["trailing_stale_rows"] = n_steps - 1 - last_real
            # Fix pass — INTERIOR frozen spans, uniform on both pin paths. The fill loop below runs
            # from ``first`` to the END, so post-fill every row >= first carries a price: an
            # INTERIOR hole is filled by exactly the same rule as a tail but produced NO disclosure
            # at all (no stale_tail, no partial, and covered_rows silently included it on the minute
            # twin). Measured and disclosed here; the fill itself is UNCHANGED (no numeric moves).
            filled_rows = np.arange(n_steps) >= first        # == (prices[:, col] != 0) post-fill
            spans, n_frozen, n_spans = _frozen_spans(real_rows, filled_rows)
            entry["real_panel_rows"] = n_nz
            entry["filled_rows"] = int(filled_rows.sum())
            entry["n_frozen_rows"] = n_frozen
            if n_frozen:
                entry["frozen_spans"] = spans
                entry["n_frozen_spans"] = n_spans
                entry["frozen_interior_rows"] = n_frozen - int(entry.get("trailing_stale_rows", 0))
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
    # D30 rider (b), fix pass: SAME refusal, SAME default, on the daily pin path too.
    _guard_active_slots(coverage, frame_symbols, require_active_slots, start_date, end_date)
    return prices, list(frame_symbols), np.asarray(timestamps, dtype=np.int64), coverage


def create_daily_panel_framed(args, frame_symbols):
    """Fixed-frame analogue of ``create_daily_panel``: load the framed daily closes (one column
    per recorded slot, zero-coverage columns kept flat-zero), build the SAME EnvConfig, reset.
    Returns (states0, prices_j, env_cfg, row_ts, frame_symbols, coverage). No pad/truncate — the
    frame is exactly n_assets columns by construction (asserted by _load_frame_provenance)."""
    prices_np, symbols, row_ts, coverage = load_daily_prices_framed(
        args.db, args.start, args.end, frame_symbols,
        require_active_slots=_resolve_require_active(args))
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
    benchmark is impossible to mistake for a 64-live one (task scope 2).

    TASK_865: --panel-frame-source minute dispatches to the MINUTE sibling below BEFORE any of
    this runs; the daily branch (the default) is untouched and byte-identical."""
    if getattr(args, "panel_frame_source", "daily") == "minute":
        return _build_panel_framed_minute(args)
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
    stale = [c for c in coverage if c.get("stale_tail")]          # D30 amendment 4 (disclosure)
    inactive_slots = [c["slot"] for c in inactive]
    n_active = args.n_assets - len(inactive_slots)
    # --- stdout report (task scope 2: impossible to mistake 36/64-live for 64-live) ---
    print("=" * 84)
    print(f"[853] FIXED-FRAME daily panel from checkpoint provenance "
          f"({_display_path(args.panel_frame_from_checkpoint)}): {args.n_assets} slots, "
          f"{n_active} ACTIVE / {len(inactive_slots)} INACTIVE (zero coverage) in "
          f"[{args.start}..{args.end}]")
    print(f"[853] ... of the {n_active} ACTIVE: {len(partial)} PARTIAL (listed mid-window, "
          f"pre-listing rows flat-zero => force-HOLD) and {len(stale)} STALE-TAIL (delisted "
          f"mid-window, trailing rows FORWARD-FILLED at a frozen price and still tradeable) — "
          f"neither is in inactive_slots, and a stale tail is not in partial_slots either")
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
    if stale:
        print(f"[853] STALE-TAIL slots ({len(stale)}) — delisted / migrated mid-window: this "
              f"loader FORWARD-FILLS the trailing rows at the last real close (the D30-sanctioned "
              f"semantics: PV-exact, cannot fabricate PnL, a strict fee drag above fee 0), so they "
              f"remain TRADEABLE at a FROZEN price and stay weight-eligible. Disclosed here "
              f"because they are in neither inactive_slots nor partial_slots:")
        for c in stale:
            print(f"        slot {c['slot']:>3}  {c['symbol']:<16} last real close ts "
                  f"{c['last_real_ts']} = panel row {c['last_covered_row']} of {c['n_rows'] - 1}; "
                  f"{c['trailing_stale_rows']} trailing rows forward-filled "
                  f"({100.0 * c['trailing_stale_rows'] / max(c['n_rows'], 1):.1f}% of the window)")
    # FIX PASS — the same CELL-level honesty the minute path emits, on the daily pin. An INTERIOR
    # frozen span was invisible here too (this loader forward-fills interior holes and derived
    # staleness from the last real row only).
    warm_d = int(env_cfg_f.window_size)
    n_dec_d = int(panel_info["n_decisions"])
    dec_px_d = np.asarray(prices_f[warm_d:warm_d + n_dec_d])
    n_cells_d = int(n_dec_d * args.n_assets)
    n_inert_d = int((dec_px_d <= 0).sum())
    fz_d = frozen_decision_cells(coverage, warm_d, n_dec_d)
    n_not_real_d = n_inert_d + fz_d["total"]
    pct_d = lambda n: (100.0 * n / n_cells_d) if n_cells_d else 0.0            # noqa: E731
    print(f"[853] DECISION GRID ({n_dec_d} x {args.n_assets} = {n_cells_d} cells): "
          f"{n_inert_d} INERT ({pct_d(n_inert_d):.2f}%, no price => force-HOLD) + "
          f"{fz_d['total']} FROZEN ({pct_d(fz_d['total']):.2f}%, forward-filled: "
          f"{fz_d['interior']} INTERIOR + {fz_d['tail']} trailing) => "
          f"{n_cells_d - n_not_real_d} LIVE ({pct_d(n_cells_d - n_not_real_d):.2f}%); "
          f"NOT REAL DATA {n_not_real_d} ({pct_d(n_not_real_d):.2f}%)")
    for r in fz_d["slots"][:12]:
        print(f"        slot {r['slot']:>3}  {r['symbol']:<16} {r['frozen_decision_cells']:>6} "
              f"frozen decision cells ({r['interior']} interior + {r['tail']} trailing) across "
              f"{r['n_frozen_spans']} span(s)")
    if not inactive and not partial and not stale and not fz_d["slots"]:
        print("[853] all slots fully covered on this window (no inactive/partial/frozen spans).")
    print("=" * 84)
    # EMITTED coverage: the same records with an over-long frozen_spans list elided for
    # readability. Computed AFTER fz_d, which reads the complete in-memory coverage — the split
    # that fix pass 2 introduced so a display bound can never drive a count.
    cov_art = coverage_for_artifact(coverage)
    inactive = [c for c in cov_art if c["status"] == "inactive"]
    partial = [c for c in cov_art if c["status"] == "partial"]
    stale = [c for c in cov_art if c.get("stale_tail")]
    panel_info["panel_frame"] = {
        "source": "checkpoint-provenance-fixed-frame (TASK_853)",
        "provenance_path": _display_path(args.panel_frame_from_checkpoint),
        "provenance_meta": provenance_meta_block(prov, args.panel_frame_from_checkpoint),
        "n_slots": int(args.n_assets),
        "n_active_slots": int(n_active),
        "n_inactive_slots": int(len(inactive_slots)),
        "n_partial_slots": int(len(partial)),
        "inactive_slots": list(inactive_slots),
        "inactive_spans": inactive,
        "partial_spans": partial,
        "coverage": cov_art,
        # D30 amendment 4 — the NAMED additive stale-tail disclosure fields. Appended LAST so every
        # key above keeps its value AND its position in a daily-pinned artifact; the daily path's
        # numerics are unchanged (this loader always forward-filled).
        "n_stale_tail_slots": int(len(stale)),
        "stale_tail_slots": [c["slot"] for c in stale],
        "stale_tail_spans": stale,
        "stale_tail_rule": STALE_TAIL_RULE,
        # FIX PASS additive keys, same names/meanings as the minute path (so a consumer reads ONE
        # schema): the honest frozen accounting incl. INTERIOR spans, and the breadth floor that is
        # now in force here too — recorded so "no floor applied" is never silence.
        "n_frozen_slots": fz_d["n_slots"],
        "frozen_slots": [r["slot"] for r in fz_d["slots"]],
        "frozen_span_slots": fz_d["slots"],
        "n_decision_cells": n_cells_d,
        "n_inert_decision_cells": n_inert_d,
        "inert_decision_cell_pct": round(pct_d(n_inert_d), 4),
        "n_frozen_decision_cells": fz_d["total"],
        "frozen_decision_cell_pct": round(pct_d(fz_d["total"]), 4),
        "n_frozen_stale_tail_decision_cells": fz_d["tail"],
        "frozen_stale_tail_decision_cell_pct": round(pct_d(fz_d["tail"]), 4),
        "n_frozen_interior_decision_cells": fz_d["interior"],
        "frozen_interior_decision_cell_pct": round(pct_d(fz_d["interior"]), 4),
        "n_not_real_decision_cells": n_not_real_d,
        "not_real_decision_cell_pct": round(pct_d(n_not_real_d), 4),
        "n_live_decision_cells": n_cells_d - n_not_real_d,
        "live_decision_cell_pct": round(pct_d(n_cells_d - n_not_real_d), 4),
        # Elision of a slot's span LIST is disclosed; the counts above are never elided.
        "frozen_spans_display_max": FROZEN_SPANS_ARTIFACT_MAX,
        "n_slots_with_elided_frozen_span_list": sum(
            1 for c in cov_art if c.get("frozen_spans_truncated")),
        "require_active_slots": _resolve_require_active(args),
    }
    # Carried to the rollout: force these slots to HOLD on the 848 policy-argmax replay (which
    # bypasses the env mask); the search operator already masks them. See evaluate_checkpoint.
    panel_info["inactive_slots"] = list(inactive_slots)
    return prices_f, env_cfg_f, state_init, row_ts_f, col_symbols, panel_info


# =============================================================================
# TASK_865 — the SAME fixed-frame mechanic on the MINUTE source (D29 ruled opt (a))
# =============================================================================
# 853/863 pin the frame through ``ohlcv_daily`` ONLY. On the 861 real leg that substitutes two
# confounds for the one D28 removed: a SOURCE confound (D20: daily is within-panel grading only)
# and a 64->33 BREADTH confound (the daily table covers 33/64 frame slots on the leg-1 window,
# the minute table 62/64). This section adds the minute sibling of the 853 loader — the SAME
# semantics, the SAME fail-louds, one column per recorded slot in the recorded order — reading
# the minute ``ohlcv`` table with the canon ``[::bar_minutes]`` stride (env_factory.py:409; the
# minute env itself is built by ``create_jax_env_historical``, env_factory.py:321, whose universe
# derivation is the ``ORDER BY COUNT(*) DESC LIMIT`` block at vectorbt_env.py:578-596).
#
# FLAG-GATED: --panel-frame-source minute (default 'daily' = the shipped 853 path, whose NUMERICS
# are untouched). Selection is recorded in the artifact's panel_frame block; the DAILY block keeps
# every field 853 shipped, at its shipped value (its ABSENCE of ``panel_frame_source`` still means
# daily) and gains only D30 amendment 4's additive stale-tail disclosure keys.
#
# THE STALE TAIL (D30, ruled 2026-07-30, opt (i)). A slot whose last REAL close falls before the end
# of the window (delisting / ticker migration) is NOT zero-coverage, so the 853 INACTIVE rule does
# not see it. Its trailing rows are FORWARD-FILLED at that last real close — identical to what the
# shipped daily loader has always done, now uniform across both pin paths. On the DAILY pin they
# stay TRADEABLE on a constant series; on the MINUTE paths TASK_868 (D31 a2) force-HOLDs them
# (whole action — see OPEN_GATE_RULE). The rejected alternative (zero the tail and let
# ``price_valid`` force-HOLD it) was
# MEASURED to FABRICATE PnL: force-HOLD gates OPENING, not VALUATION, so a position open at the
# boundary is marked against price 0 and frozen there — 75,002.50 / 124,997.50 for a long/short book
# where any book must end at exactly 100,000.00, i.e. -/+25% on 1 of 4 slots. On the 861 leg-1 window
# two slots are stale-tailed (the FTM->S / MATIC->POL migrations, both dead in the WHOLE table after
# 2025-03-31, 186 of 1734 rows each), which under vector_weight_max 4.0 / w_unit 1/64 put up to
# ~12.5% of PV of fabricated, sign-indeterminate PnL into the exact scalar graded against > +0.839.
# A forward-filled constant tail cannot do that (a round trip nets exactly 0 at fee 0 and costs fees
# above it), so it is the safe direction; it is nevertheless a FROZEN price, so it is DISCLOSED —
# dual-disclosed (stdout + artifact) on both pin paths and ASSERTED in the fixture smoke, alongside
# the ground-truth PV-conservation leg that discriminates the options. TRUE LIQUIDATION is PARKED
# (D30 opt (ii)). See ``STALE_TAIL_RULE`` / ``_ffill_interior_only``.
# NOTE, deliberately NOT done: ``POL/USD`` exists in the DB (id 763, 2023-12-21..2025-09-30) and
# ``S/USD`` does not. Splicing a successor ticker into a predecessor's slot is a UNIVERSE change
# needing a planner ruling on cross-ticker splice provenance — out of scope here; reported only.

def _minute_table_exists(db_path):
    """True iff ``db_path`` carries a minute ``ohlcv`` table. Opened through ``_ro_connect``, so a
    missing/typo'd/symlinked ``--db`` is REFUSED rather than silently created as a 0-byte database
    (which a plain read-write handle does, turning "no such file" into "no such table")."""
    conn = _ro_connect(db_path)
    try:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='ohlcv'").fetchone()
    finally:
        conn.close()
    return row is not None


def _minute_coverage_ids(db_path, asset_ids, s_ts, e_ts):
    """Subset of ``asset_ids`` with >=1 real minute ``ohlcv`` close in [s_ts, e_ts]. The minute
    twin of ``_daily_coverage_ids`` (same shape, same ``close IS NOT NULL AND close != 0``
    hardening — row existence alone is not coverage). Own connection, opened STRICTLY read-only
    (``_ro_connect`` => ``file:...?mode=ro``): the statement is a SELECT, but a read-write handle
    would still be free to replay a hot journal into the 24GB source of record on the first read."""
    if not asset_ids:
        return set()
    conn = _ro_connect(db_path)
    try:
        ph = ",".join("?" * len(asset_ids))
        rows = conn.execute(
            f"""SELECT asset_id FROM ohlcv WHERE asset_id IN ({ph})
                AND timestamp >= ? AND timestamp <= ?
                AND close IS NOT NULL AND close != 0 GROUP BY asset_id""",
            (*asset_ids, s_ts, e_ts)).fetchall()
    finally:
        conn.close()
    return {r[0] for r in rows}


def _ffill_interior_only(col):
    """THE fill rule for BOTH framed loaders (D30 opt (i), ruled 2026-07-30): forward-fill a 1-D
    price column from its FIRST real value ONWARD, all the way to the end of the window, leaving
    ONLY the leading pre-listing region flat-zero. Vectorised because the minute axis is millions
    of rows deep; semantically identical to ``load_daily_prices_framed``'s in-place loop, which is
    the rule the SHIPPED daily pin has always applied. Returns a new array.

    HEAD (flat-zero, unchanged — the 853 leak fix): a slot's pre-listing rows carry no real price
    and must read INACTIVE, so ``price_valid = price > 0`` force-HOLDs them under every operator.
    A back-filled head would be a tradeable fabricated pre-listing region, i.e. a leak.

    TAIL (forward-filled, D30 opt (i)): a slot whose last real close precedes the window end
    (delisting / ticker migration) keeps that close to the end. The alternative — zeroing the tail
    — was MEASURED to fabricate PnL: force-HOLD gates OPENING, not VALUATION
    (jax_trading_env.py:2106-2115 vs :246-258), so a position open at the boundary is marked
    against price 0 (a long wiped, a short handed a 100% windfall) and then frozen there. On flat
    prices at fee 0, where any book must end at exactly the initial cash, zeroing gives
    75,002.50 / 124,997.50 on a long/short book (+/-25% on 1 of 4 slots) while forward-filling is
    PV-exact to the cent. A TRAILING constant span cannot fabricate PnL — a round trip on it nets
    exactly 0 at fee 0 and is a strict fee drag above it, i.e. biased AGAINST the bar being graded.

    INTERIOR (forward-filled TOO — read this before quoting the paragraph above): this function
    fills from the first real value to the END, so a hole in the MIDDLE of a slot's history is
    filled by exactly the same rule as a tail, and the "cannot fabricate PnL" argument DOES NOT
    carry over to it: the span is constant but it RE-ANCHORS, so the book can be opened at a price
    no market offered and the whole gap move arrives in one bar (MEASURED re-anchor jumps on the
    861 leg-1 window: ETHBTC +64.52%, BTCCAD +33.88%, BTCGBP +32.69%, BTCCHF +24.29%, /USDTEUR
    -7.76%; 11 of 64 frame slots are near-empty 2025-03-31..2025-10-01 in many multi-day sub-gaps,
    2,013 frozen interior decision cells = 1.88% of the grid). Fix pass: those spans used to be
    INVISIBLE — staleness was derived from the last real row only, so a slot that resumed before
    the window end got no stale_tail, no partial, no inactive, and read covered_rows == n_rows.
    They are now MEASURED and DISCLOSED per slot as ``frozen_spans`` and counted into the
    cell-honesty totals. What to DO about them (mask / refuse / accept) is a D30-class semantics
    ruling, is out of this task's scope, and is deliberately NOT decided here — the fill rule
    below is UNCHANGED, so no numeric moves; only the disclosure is new.

    TRUE LIQUIDATION (close at the last real close, then mask) is the economically honest third
    option; it needs a state-aware forced close in ``walk_env`` AND in eval's search rollout and is
    PARKED as its own task (D30 opt (ii)). Asserted by the PV-conservation leg in
    experiments/865_minute_frame_pin/smoke_minute_frame_pin.py."""
    nz = col != 0
    if not nz.any():
        return col.copy()                                  # fully-inactive slot: stays flat-zero
    idx = np.where(nz, np.arange(col.shape[0], dtype=np.int64), 0)
    np.maximum.accumulate(idx, out=idx)
    out = col[idx]
    out[:int(np.argmax(nz))] = 0.0                         # leading pre-listing rows stay ZERO
    return out


# How many frozen spans a single slot may show in an EMITTED ARTIFACT before the list is elided.
# A DISPLAY bound only (see ``coverage_for_artifact``): a slot can carry hundreds of multi-day
# sub-gaps and the JSON must stay readable. It is deliberately NOT applied to the in-memory
# coverage the cell accounting reads — see the fix-pass note on ``frozen_decision_cells``.
FROZEN_SPANS_ARTIFACT_MAX = 64


def _frozen_spans(real_rows, filled_rows):
    """Contiguous PANEL-ROW ranges ``[first, last]`` where the panel carries a price that is NOT
    real data for that slot — i.e. ``filled AND NOT real``. Fix pass: THE honest definition of
    "not real data at this (row, slot)" is *the panel row does not sample a real close for that
    slot*, NOT "the row is past the slot's last real close". The second definition is blind to
    every INTERIOR hole, and interior holes are the majority of the frozen cells on both the real
    leg-1 window (2,013 cells = 1.88%, invisible before this) and the fixture (101 cells = 25.25%
    of the smoke grid, while the banner claimed 73.25% live).

    Covers head/interior/tail uniformly: a leading pre-listing row is ``NOT filled`` (flat-zero,
    price_valid force-HOLDs it) so it is INERT, not frozen — the two categories are disjoint and
    together account for every non-real cell. Returns ``(spans, n_frozen_rows, n_spans)``.

    FIX PASS 2 — the returned span list is COMPLETE. It used to be truncated to ``max_spans=64``
    here, i.e. AT THE SOURCE, and ``frozen_decision_cells`` then summed that truncated list: on a
    2-slot synthetic minute panel with one slot dark on every other bar the artifact recorded
    ``n_frozen_rows 200`` (true) beside ``n_frozen_decision_cells 34`` of a true 170 and
    ``live_decision_cell_pct 95.01%`` of a true 75.07%, exit 0, with the loader's own
    ``coverage[slot].n_frozen_rows`` contradicting the total in the same file. Truncation is now a
    pure DISPLAY transform applied once, at artifact-emission time (``coverage_for_artifact``), and
    handing a truncated list to the cell accounting is a hard STOP."""
    frozen = np.asarray(filled_rows, dtype=bool) & ~np.asarray(real_rows, dtype=bool)
    n = int(frozen.sum())
    if n == 0:
        return [], 0, 0
    # run-length decode: boundaries where the boolean flips
    idx = np.flatnonzero(frozen)
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate(([idx[0]], idx[breaks + 1]))
    ends = np.concatenate((idx[breaks], [idx[-1]]))
    spans = [[int(a), int(b)] for a, b in zip(starts, ends)]
    return spans, n, len(spans)


def coverage_for_artifact(coverage, max_spans=FROZEN_SPANS_ARTIFACT_MAX):
    """DISPLAY copy of a loader ``coverage`` list for EMISSION into a JSON artifact: identical in
    every field except that a slot's ``frozen_spans`` list is elided to the first ``max_spans``
    entries, with the elision named (``frozen_spans_truncated`` / ``frozen_spans_shown``).

    Why this exists as a separate step (fix pass 2): the truncation used to happen inside
    ``_frozen_spans``, so the TRUNCATED list was the only one that ever existed and every consumer
    — including the cell accounting — silently under-counted. Splitting it means exactly one list
    is short, it is the one nobody computes from, and ``frozen_decision_cells`` REFUSES it by name
    if it is ever fed back in. Entries are shallow copies; nothing else is rewritten, so a coverage
    list with no over-long slot round-trips field-for-field (verified on the fixture: no slot on
    either pin path reaches 64 spans, so no artifact byte moves)."""
    out = []
    for c in coverage:
        spans = c.get("frozen_spans")
        if not spans or len(spans) <= int(max_spans):
            out.append(c)
            continue
        d = dict(c)
        d["frozen_spans"] = [list(s) for s in spans[:int(max_spans)]]
        d["frozen_spans_shown"] = int(max_spans)
        d["frozen_spans_truncated"] = True
        d["frozen_spans_truncation_note"] = (
            f"DISPLAY ONLY: {len(spans)} frozen spans on this slot, first {int(max_spans)} shown. "
            f"n_frozen_rows / n_frozen_spans and every cell count in this artifact are computed "
            f"from the COMPLETE span list, never from this one. Recomputing cell counts from the "
            f"list below is a hard STOP in eval_vector.frozen_decision_cells.")
        out.append(d)
    return out


def _real_bar_mask(raw_nz, n_steps, step):
    """(n_steps,) bool — did this slot print a REAL close anywhere inside panel row r's BAR?

    THE definition of "real data at this (panel row, slot)", and the one load-bearing judgement in
    the whole frozen-span fix. Panel row r is the ``[::step]`` sample at 1-min index ``r*step``, so
    the bar it stands for is the ``step`` minutes ``((r-1)*step, r*step]`` (row 0 is the single
    minute 0). If the slot printed anywhere in that window, the panel's value IS this bar's market
    data. If it printed nothing in the WHOLE bar, the value is carried over from an earlier bar and
    is FROZEN.

    Deliberately NOT ``raw_nz[::step]`` ("did the slot print at the exact sampled minute"), which
    is the obvious reading and is WRONG on real data: minute bars only exist where a trade
    happened, so on ``trading_data.db`` the exact-minute rule marks 51.22% of the 861 leg-1
    decision grid "frozen" (40.34% live) — it is measuring sparsity, not staleness. The bar rule
    gives 2.22% frozen / 89.33% live on the same grid, and reproduces the panel's independently
    measured 2,013 interior cells (1.88%) and its exact list of 11 near-empty slots. The
    exact-minute count is still recorded, as a separate DIAGNOSTIC
    (``n_no_print_at_sampled_minute``), because it measures something real about the canon
    ``[::bar_minutes]`` operator — it samples ONE MINUTE rather than aggregating the bar — it just
    is not fabrication.

    TASK_868: the implementation is SHARED (``mcts_trader.data.asset_aliases.real_bar_mask``)
    with the canon unpinned path's open-gate (env_factory) and the duplicate sweep, so the
    pinned and canon loaders can never drift on the bar definition."""
    from mcts_trader.data.asset_aliases import real_bar_mask
    return real_bar_mask(raw_nz, n_steps, step)


def frozen_decision_cells(coverage, warm, n_decisions):
    """Fix pass — CELL-LEVEL frozen accounting over the DECISION grid, from the per-slot
    ``frozen_spans`` the framed loaders now record. Shared by both runners so the banner, the
    artifact and the smoke gate cannot drift apart.

    Returns ``{"total", "interior", "tail", "slots", "n_slots"}``, counting only panel rows inside
    ``[warm, warm + n_decisions)`` (the rows an operator actually decides on). A cell is TAIL when
    the slot is flagged ``stale_tail`` AND the cell's own panel row is past that slot's
    ``last_covered_row``, INTERIOR otherwise — the distinction matters because only the trailing
    case carries the "a constant span cannot fabricate PnL" argument (an interior span re-anchors;
    see STALE_TAIL_RULE).

    The old accounting derived the frozen count from ``last_covered_row`` alone, which is blind to
    every interior hole by construction: MEASURED 2,013 cells (1.88% of the grid) missing on the
    861 leg-1 window and 101 cells (25.25%) missing on the fixture smoke grid, where the banner
    consequently claimed 73.25% live against a measured 48.00%.

    FIX PASS 2, two defects, both measured:
      * INTERIOR-vs-TRAILING was decided PER SPAN against ``last_dec_row``
        (``stale_tail and b >= last_dec_row``), i.e. against the END OF THE GRADED WINDOW rather
        than against the slot's own last real row. Shorten the grid with ``--max-decisions`` (a
        first-class manifest window key, scorecard.py:959-960) so that it ends INSIDE an interior
        span and that span is reclassified wholesale as trailing: on a synthetic slot with an
        interior hole on rows 100..200 that RE-ANCHORS +50% and a delisting after row 280, the full
        grid reads ``total 221 / INTERIOR 101 / TRAILING 120`` and the same panel at 91 decisions
        reads ``total 51 / INTERIOR 0 / TRAILING 51`` — the banner prints "0 INTERIOR (a hole that
        RE-ANCHORS: it CAN fabricate PnL)" over 51 of exactly those cells, silencing the one axis
        DECISIONS D31 exists to expose. Classification is now PER CELL against the slot's own
        ``last_covered_row``, so it cannot depend on where the graded window is cut.
      * the count was summed over ``c["frozen_spans"]``, which used to arrive pre-truncated to 64
        entries. The list is complete now; a SHORT list is refused by name rather than quietly
        under-counted (an artifact's ``coverage`` block is display-elided, so this is reachable
        from a consumer that recomputes cells from a written artifact)."""
    lo, hi = int(warm), int(warm) + int(n_decisions)          # [lo, hi)
    last_dec_row = hi - 1
    out = {"total": 0, "interior": 0, "tail": 0, "slots": [], "n_slots": 0}
    for c in coverage:
        spans = c.get("frozen_spans") or []
        if not spans:
            continue
        # HARD STOP: never count cells off a display-elided span list. ``n_frozen_spans`` is the
        # loader's true span count; if the list is shorter, the caller is holding
        # ``coverage_for_artifact`` output (or an artifact read back off disk) and every total
        # derived from it would be silently low while the artifact's own n_frozen_rows contradicts
        # it. Refuse instead of reporting a number that looks fine.
        n_true_spans = int(c.get("n_frozen_spans", len(spans)))
        if n_true_spans > len(spans) or c.get("frozen_spans_truncated"):
            raise RuntimeError(
                f"frozen_decision_cells: slot {c.get('slot')} {c.get('symbol')!r} carries "
                f"{n_true_spans} frozen spans but only {len(spans)} are present — this is a "
                f"DISPLAY-elided coverage entry (coverage_for_artifact / a coverage block read back "
                f"out of an emitted artifact). Counting decision cells from it under-reports the "
                f"frozen total and inflates live_decision_cell_pct while the SAME record's "
                f"n_frozen_rows={c.get('n_frozen_rows')} contradicts it. Pass the loader's "
                f"in-memory coverage (complete spans) instead.")
        # A frozen span is by construction a run of NON-real rows and ``last_covered_row`` is the
        # slot's last REAL row, so a span never straddles it — but count per CELL anyway, so the
        # classification stays correct if the span representation ever changes.
        lcr = None
        if c.get("stale_tail"):
            if "last_covered_row" not in c:
                raise RuntimeError(
                    f"frozen_decision_cells: slot {c.get('slot')} {c.get('symbol')!r} is flagged "
                    f"stale_tail but carries no last_covered_row, so its frozen cells cannot be "
                    f"split into INTERIOR (can fabricate PnL) and TRAILING (cannot). Refusing to "
                    f"guess — both framed loaders record last_covered_row whenever stale_tail is "
                    f"set.")
            lcr = int(c["last_covered_row"])
        tot = it = tl = 0
        for a, b in spans:
            a_c, b_c = max(int(a), lo), min(int(b), last_dec_row)
            if b_c < a_c:
                continue
            n = b_c - a_c + 1
            tot += n
            n_tail = 0 if lcr is None else max(0, b_c - max(a_c - 1, lcr))
            tl += n_tail
            it += n - n_tail
        if not tot:
            continue
        out["total"] += tot
        out["interior"] += it
        out["tail"] += tl
        out["slots"].append({"slot": c["slot"], "symbol": c["symbol"], "frozen_decision_cells": tot,
                             "interior": it, "tail": tl,
                             "n_frozen_spans": n_true_spans,
                             # DISPLAY elision of THIS slot's span list in the emitted artifact —
                             # the counts above are always off the complete list.
                             "frozen_spans_truncated": n_true_spans > FROZEN_SPANS_ARTIFACT_MAX,
                             "no_real_panel_rows": bool(c.get("no_real_panel_rows"))})
    out["slots"].sort(key=lambda r: -r["frozen_decision_cells"])
    out["n_slots"] = len(out["slots"])
    return out


def load_minute_prices_framed(db_path, start_date, end_date, frame_symbols, bar_minutes,
                              require_active_slots=REQUIRE_ACTIVE_SLOTS_DEFAULT):
    """Fixed-frame MINUTE close matrix: ONE column per recorded frame symbol IN ORDER, read from
    the minute ``ohlcv`` table and strided ``[::bar_minutes]`` exactly as the canon minute path
    does (env_factory.py:409 — fill FIRST at 1-min resolution, stride SECOND; the reverse order
    would ffill from the previous STRIDED row and produce a different panel).

    The minute twin of ``load_daily_prices_framed``: no drop, no pad-by-cycling, zero-coverage
    frame symbols KEPT as flat-zero columns (the env's price>0 mask renders them inactive), the
    same fill rule (``_ffill_interior_only``: flat-zero head, forward-filled tail — D30 opt (i))
    and the same fail-louds (missing table, unknown symbol, empty window, all-slots-inactive),
    PLUS the D30 rider (b) breadth refusal (``require_active_slots``).

    Returns (prices_np (T, n_frame) float32, frame_symbols, row_ts int64 (T,), coverage,
    open_gate (T, n_frame) bool). ``open_gate[r, col]`` is the causal TASK_868 (D31 a2)
    open-gate: True iff the slot printed a real close anywhere inside panel row r's bar
    (== the per-slot ``real_panel`` mask the frozen accounting derives). Consumers
    force-HOLD OPENING where it is False; VALUATION (the D30 forward-fill) is untouched."""
    if int(bar_minutes) < 1:
        raise RuntimeError(
            f"--panel-frame-source minute: --bar-minutes must be >= 1 (got {bar_minutes}); the "
            f"stride is a row subsample of the 1-min ohlcv table.")
    if not _minute_table_exists(db_path):
        raise RuntimeError(
            f"--panel-frame-source minute: no `ohlcv` (minute) table in {db_path}. The minute "
            f"fixed-frame panel reads that table directly; it is absent here. Use the default "
            f"--panel-frame-source daily, or point --db at a DB that carries the minute table.")
    s_ts = _to_epoch(start_date, 0)
    e_ts = _to_epoch(end_date, 9999999999)
    conn = _ro_connect(db_path)
    cur = conn.cursor()
    # TASK_868 (D31 a1) — ALIAS UNION. Each recorded frame symbol resolves to EVERY
    # ``assets`` row whose normalized form matches (``/USDTEUR`` == ``USDTEUR`` etc.); the
    # slot's column is the UNION of the group's rows. A symbol with NO group at all is the
    # same unknown-symbol refusal as before; a KNOWN group with a missing exact spelling
    # (e.g. the frame records ``USDTEUR`` but only ``/USDTEUR`` exists) resolves to the
    # group — the alternative (refuse a slot whose data exists under a spelling variant)
    # is exactly the identity split D31 repairs.
    from mcts_trader.data.asset_aliases import (
        alias_agreement_gate, resolve_requested_symbols)
    all_asset_rows = cur.execute("SELECT id, symbol FROM assets").fetchall()
    resolution, unknown = resolve_requested_symbols(all_asset_rows, frame_symbols)
    if unknown:
        conn.close()
        raise RuntimeError(
            f"--panel-frame-source minute: {len(unknown)} recorded frame symbol(s) are not in "
            f"the `assets` table of {db_path} (unknown symbols — the frame cannot be dropped or "
            f"reordered to skip them without scrambling slot identity): "
            f"{unknown[:10]}{' ...' if len(unknown) > 10 else ''}")
    # One alias GROUP per slot, in recorded frame order (slot k IS asset k). NOTE the contrast
    # with VectorBTTradingEnv._load_database_prices' specific_symbols branch, which DROPS a
    # symbol the DB does not know with a logging.warning — that silently shortens and
    # re-indexes the panel, the exact failure a frame pin exists to refuse.
    id_groups = [r["asset_ids"] for r in resolution]
    db_symbols = [r["db_symbols"] for r in resolution]
    # FAIL-LOUD agreement gate (D31 a1), once per load BEFORE the stream: every alias pair's
    # shared-timestamp closes in the window must agree (rel <= 1e-12) — a disagreement
    # REFUSES naming both asset_ids + the first disagreeing timestamp (never average, never
    # prefer silently). Agreeing duplicates dedupe in the union below.
    gate_disclosure = alias_agreement_gate(conn, "ohlcv", id_groups, s_ts, e_ts)
    gate_by_ids = {tuple(d["asset_ids"]): d["overlaps"] for d in gate_disclosure}
    asset_ids_flat = sorted({aid for g in id_groups for aid in g})
    covered_ids = _minute_coverage_ids(db_path, asset_ids_flat, s_ts, e_ts)
    # A slot is covered iff ANY member of its alias group has a real close in the window.
    slot_covered = [any(aid in covered_ids for aid in g) for g in id_groups]
    covered_ids_ordered = sorted(aid for g, cov in zip(id_groups, slot_covered) if cov
                                 for aid in g if aid in covered_ids)
    if not covered_ids_ordered:
        conn.close()
        raise RuntimeError(
            f"--panel-frame-source minute: EVERY slot in the recorded frame has ZERO minute "
            f"`ohlcv` coverage in [{start_date}..{end_date}] — the panel would be entirely "
            f"inactive (all HOLD). Nothing to evaluate; widen --start/--end or check the frame.")
    # D30 rider (b), EARLY: zero-coverage is already known here, before the ~60M-row stream on a
    # real window, so a doomed run refuses in seconds instead of after a multi-minute panel build.
    # The authoritative check runs again on the FINAL coverage report below (which can only find
    # MORE inactive slots — the rare "no real close reaches a strided panel row" case).
    try:
        _guard_active_slots(
            [{"slot": i, "symbol": s,
              "status": "active" if slot_covered[i] else "inactive"}
             for i, s in enumerate(frame_symbols)],
            frame_symbols, require_active_slots, start_date, end_date)
    except Exception:
        # NOT just RuntimeError: the guard also raises RequireActiveSlotsError (a ValueError) for a
        # malformed or unsatisfiable floor, and that used to leak this connection.
        conn.close()
        raise
    # Row axis: DISTINCT minute timestamps over the COVERED slots' UNIONED member ids (the canon
    # minute loader's axis query; uncovered slots contribute no rows anyway).
    phc = ",".join("?" * len(covered_ids_ordered))
    timestamps = [r[0] for r in cur.execute(
        f"""SELECT DISTINCT timestamp FROM ohlcv
            WHERE asset_id IN ({phc}) AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp""",
        (*covered_ids_ordered, s_ts, e_ts)).fetchall()]
    if not timestamps:
        conn.close()
        raise RuntimeError(
            f"--panel-frame-source minute: the minute `ohlcv` table has no rows for the frame in "
            f"[{start_date}..{end_date}] (empty window). Widen --start/--end.")
    ts_to_idx = {ts: i for i, ts in enumerate(timestamps)}
    id_to_col = {aid: col for col, g in enumerate(id_groups) for aid in g}   # slot == frame position
    n_full = len(timestamps)
    raw = np.zeros((n_full, len(frame_symbols)), dtype=np.float32)
    # Per-id in-window row counts, accumulated during the stream (the D31 disclosure rider:
    # per-slot asset_ids used + per-id row counts).
    per_id_rows = {aid: 0 for aid in asset_ids_flat}
    # Streamed (no fetchall): the real leg's window is ~60M minute rows and materialising them as
    # a Python list would cost tens of GB. Same rows, same values, bounded memory.
    for asset_id, ts, close in cur.execute(
        f"""SELECT asset_id, timestamp, close FROM ohlcv
            WHERE asset_id IN ({phc}) AND timestamp >= ? AND timestamp <= ?""",
        (*covered_ids_ordered, s_ts, e_ts)):
        i = ts_to_idx.get(ts)
        c = id_to_col.get(asset_id)
        if i is not None and c is not None and close:
            # Alias union: duplicate (timestamp, group) rows AGREED at the gate above, so a
            # second write lands the same value — dedupe is numerically free.
            raw[i, c] = close
            per_id_rows[asset_id] += 1
    conn.close()
    step = int(bar_minutes) if int(bar_minutes) > 1 else 1
    n_steps = len(range(0, n_full, step))
    prices = np.zeros((n_steps, len(frame_symbols)), dtype=np.float32)
    # TASK_868 (D31 a2): the causal OPEN-GATE mask at panel resolution — open_gate[r, col]
    # is True iff the slot printed a real close anywhere inside panel row r's bar. This is
    # the per-slot ``real_panel`` mask the frozen accounting already derives, assembled as
    # one matrix so the rollout/walk can force-HOLD OPENING on every frozen cell. It is
    # CAUSAL by construction: bar r's value reads only prints inside bar r, so the mask at
    # decision t is identical whether or not rows after t exist (no look-ahead into whether
    # a gap later resumes — which is also why TRAILING gaps are gated: at decision time a
    # delisting and an interior hole are indistinguishable). The D30 forward-fill still
    # values every row; the gate only gates OPENING.
    open_gate = np.zeros((n_steps, len(frame_symbols)), dtype=bool)
    coverage = []
    for col, sym in enumerate(frame_symbols):
        raw_col = raw[:, col]
        raw_nz = raw_col != 0
        n_real = int(raw_nz.sum())
        prices[:, col] = _ffill_interior_only(raw_col)[::step]    # fill THEN stride (canon order)
        panel_nz = prices[:, col] != 0
        # TASK_868 (a1) disclosure: the alias group this slot unioned, per-id row counts,
        # and the gate's measured overlap agreement.
        alias_block = {"asset_ids": list(id_groups[col]),
                       "db_symbols": list(db_symbols[col]),
                       "aliased": len(id_groups[col]) > 1,
                       "per_id_rows": {str(aid): per_id_rows[aid] for aid in id_groups[col]},
                       "overlap_agreement": gate_by_ids.get(tuple(id_groups[col]), [])}
        if not slot_covered[col] or not panel_nz.any():
            # Zero minute coverage in the window (or no real close early enough to reach a strided
            # row): flat-zero column -> INACTIVE for the whole window, exactly like 853's daily.
            coverage.append({"slot": col, "symbol": sym, "status": "inactive",
                             "reason": ("zero minute ohlcv coverage in window"
                                        if not slot_covered[col] else
                                        "no real minute close reaches a strided panel row"),
                             "span": [start_date, end_date],
                             "real_minute_rows": n_real, **alias_block})
            continue
        first = int(np.argmax(panel_nz))
        last_real_full = int(np.max(np.nonzero(raw_nz)[0]))
        # Fix pass — REAL vs FILLED at PANEL resolution. ``_real_bar_mask`` is the honest "did this
        # slot print anywhere inside this panel row's BAR" mask (see its docstring for why the
        # exact-minute reading is wrong on real data), and it is the basis for every not-real
        # accounting below. Two defects it closes:
        #   * ``covered_rows`` used to be ``panel_nz.sum()`` = the FILLED count, which on this path
        #     is DEGENERATE (== n_rows for every active slot) while the daily twin's identically
        #     named field counts REAL rows. Same key, opposite meaning, one of them useless.
        #   * a slot can have ZERO real PANEL rows and still be fully priced by the fill (every real
        #     close lands outside every strided bar). It used to read covered_rows == n_rows and
        #     count ACTIVE toward the breadth guard with 100% fabricated cells and no disclosure.
        real_panel = _real_bar_mask(raw_nz, n_steps, step)
        open_gate[:, col] = real_panel          # TASK_868 (a2): the causal open-gate column
        n_real_panel = int(real_panel.sum())
        # DIAGNOSTIC, not a frozen count: rows whose exact sampled minute carries no print. Normal
        # on sparse minute data (the canon [::bar_minutes] operator samples one minute, it does not
        # aggregate the bar) — recorded so the distinction is visible instead of assumed.
        n_no_print_at_sampled_minute = int(n_steps - raw_nz[::step].sum())
        entry = {"slot": col, "symbol": sym, "status": "active",
                 # REAL panel rows on BOTH pin paths now (was the filled count here)
                 "covered_rows": n_real_panel, "n_rows": n_steps,
                 "real_panel_rows": n_real_panel,
                 "filled_rows": int(panel_nz.sum()),
                 "no_print_at_sampled_minute_rows": n_no_print_at_sampled_minute,
                 "real_minute_rows": n_real,
                 "first_real_ts": int(timestamps[int(np.argmax(raw_nz))]),
                 "last_real_ts": int(timestamps[last_real_full]),
                 # kept under the 853 name too: the last row this slot has a real price on
                 "last_covered_ts": int(timestamps[last_real_full])}
        if first > 0:                       # listed mid-window: leading rows had no real price
            entry["status"] = "partial"
            entry["first_covered_ts"] = int(timestamps[first * step])
            entry["leading_inactive_rows"] = first
        # STALE TAIL (TASK_865 + D30 opt (i)): the slot's real data STOPS before the window end
        # (delisting / ticker migration). Its trailing panel rows are FORWARD-FILLED at the last
        # real close — the same rule the shipped daily pin has always applied, and the only one
        # measured PV-exact (see _ffill_interior_only). TASK_868 (D31 a2): those rows are now
        # ALSO inside the causal open-gate (open_gate=False there), i.e. force-HELD
        # on this MINUTE path — at decision time a trailing gap and an interior hole are
        # indistinguishable without look-ahead, and D31 ruled the mask causal. VALUATION is
        # untouched (the fill stands; the PV-conservation leg stays green). The tail is still
        # disclosed on its OWN axis (stale_tail / stale_tail_slots / stale_tail_spans) and is
        # NOT 'partial' (every row carries a price).
        # Fix pass: ``last_covered_row`` is derived from the STRIDED panel's REAL rows, not from
        # ``last_real_full // step``. That old expression names the panel row at-or-before the last
        # 1-min close, which need NOT itself sample a real close — so it could name a row whose
        # price is fabricated, and on a slot with zero real panel rows it named a row before the
        # column's first priced row entirely.
        if n_real_panel:
            last_real_row = int(np.max(np.nonzero(real_panel)[0]))
            entry["last_covered_row"] = last_real_row
        else:
            # ZERO real panel rows: every priced cell in this column is fabricated. Not silently
            # accepted — named here, printed by the banner, and counted in full into the frozen
            # cell totals (so it cannot pass the cell-honesty gate unnoticed). Deliberately NOT
            # reclassified as inactive: that would change the breadth guard's arithmetic, which is
            # a semantics change, not a disclosure fix.
            # DEFENSIVE, not a live case: under the bar rule a slot whose first print is early
            # enough to make the column filled necessarily lands that print inside some bar, so
            # this branch is unreachable via the current mask (it WAS reachable under the
            # exact-minute reading, where a slot could miss every sampled minute). Kept so the
            # invariant "every priced cell is accounted for" holds by construction rather than by
            # argument, and so a future mask change cannot reintroduce the silent version.
            last_real_row = -1
            entry["no_real_panel_rows"] = True
            entry["real_panel_rows_note"] = (
                f"every real minute close for this slot falls BETWEEN strided panel rows at "
                f"[::{step}], so no panel row samples real data: all {int(panel_nz.sum())} priced "
                f"rows are forward-filled. Counted ACTIVE by the loader's coverage definition "
                f"(>=1 real minute close in the window) and 100% frozen at cell level.")
        trailing = n_steps - 1 - last_real_row if last_real_row >= 0 else 0
        if trailing > 0:
            entry["stale_tail"] = True
            entry["trailing_stale_rows"] = trailing
        # FROZEN SPANS (fix pass, the BLOCKER): every panel row that carries a price this slot never
        # actually printed — interior holes INCLUDED. Staleness used to be derived from the last real
        # row alone, so a slot that RESUMED before the window end was invisible on every axis.
        spans, n_frozen, n_spans = _frozen_spans(real_panel, panel_nz)
        entry["n_frozen_rows"] = n_frozen
        if n_frozen:
            entry["frozen_spans"] = spans
            entry["n_frozen_spans"] = n_spans
            entry["frozen_interior_rows"] = n_frozen - trailing
        entry.update(alias_block)               # TASK_868 (a1): per-slot alias disclosure
        coverage.append(entry)
    _guard_active_slots(coverage, frame_symbols, require_active_slots, start_date, end_date)
    return (prices, list(frame_symbols), np.asarray(timestamps[::step], dtype=np.int64),
            coverage, open_gate)


def create_minute_panel_framed(args, frame_symbols):
    """Minute analogue of ``create_daily_panel_framed``: load the framed minute closes (one column
    per recorded slot, zero-coverage columns flat-zero), build the SAME EnvConfig
    ``create_jax_env_historical`` would have built for these args, reset. Returns
    (states0, prices_j, env_cfg, row_ts, frame_symbols, coverage, open_gate)."""
    prices_np, symbols, row_ts, coverage, open_gate = load_minute_prices_framed(
        args.db, args.start, args.end, frame_symbols, args.bar_minutes,
        require_active_slots=_resolve_require_active(args))
    n_assets = args.n_assets
    if prices_np.shape[1] != n_assets:          # defensive: frame length already checked upstream
        raise RuntimeError(
            f"--panel-frame-source minute: framed panel has {prices_np.shape[1]} columns but "
            f"--n-assets is {n_assets} (should be equal by construction).")
    prices_j = jnp.array(prices_np, dtype=jnp.float32)
    config = EnvConfig(                          # identical to create_daily_panel_framed's block
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
    return states, prices_j, config, row_ts, symbols, coverage, open_gate


def _build_panel_framed_minute(args):
    """TASK_865 fixed-frame MINUTE source (--panel-frame-from-checkpoint --panel-frame-source
    minute): the checkpoint's recorded frame, exact order, one column per slot, over the minute
    ``ohlcv`` table. Zero-coverage slots INACTIVE, pre-listing rows flat-zero, stale tails
    disclosed — reported to stdout AND into the JSON artifact (the 853 dual-disclosure rule)."""
    frame_symbols, prov = _load_frame_provenance(
        args.panel_frame_from_checkpoint, args.n_assets, args.symbols)
    states0, prices, env_cfg, row_ts, loaded_syms, coverage, open_gate = \
        create_minute_panel_framed(args, frame_symbols)
    result = _finalize_panel(states0, prices, env_cfg, row_ts, loaded_syms, args)
    prices_f, env_cfg_f, state_init, row_ts_f, col_symbols, panel_info = result
    # TASK_868 (D31 a2): the causal open-gate travels in panel_info for the rollout/evaluate
    # layer (stripped before JSON emission — see evaluate_checkpoint); the scalar disclosure
    # lands in the panel_frame block below.
    panel_info["open_gate"] = open_gate
    inactive = [c for c in coverage if c["status"] == "inactive"]
    partial = [c for c in coverage if c["status"] == "partial"]
    stale = [c for c in coverage if c.get("stale_tail")]
    inactive_slots = [c["slot"] for c in inactive]
    n_active = args.n_assets - len(inactive_slots)
    # D30 amendment 5 — CELL-LEVEL COVERAGE HONESTY. ``n_active`` counts only WHOLLY-inert slots,
    # so on the 861 leg-1 frame the old banner read "64 ACTIVE / 0 INACTIVE" over a decision grid
    # that is ~10.7% not-real (22 late-listing HEADS + 2 stale tails + interior frozen spans on 11
    # slots). The banner now states the partial/stale-tail/frozen slot counts AND the
    # inert/frozen/live shares of the actual decision grid, so that reading is impossible.
    #
    # FIX PASS: the frozen count no longer comes from ``last_covered_row`` (blind to every INTERIOR
    # hole — measured 1.88% of the real grid and 25.25% of the fixture grid missing, which is why
    # the old banner claimed 73.25% live on a grid measured 48.00% live).
    warm = int(env_cfg_f.window_size)
    n_dec = int(panel_info["n_decisions"])
    dec_px = np.asarray(prices_f[warm:warm + n_dec])
    n_cells = int(n_dec * args.n_assets)
    n_inert_cells = int((dec_px <= 0).sum())
    fz = frozen_decision_cells(coverage, warm, n_dec)
    n_frozen_cells = fz["total"]
    n_frozen_tail_cells = fz["tail"]
    n_frozen_interior_cells = fz["interior"]
    n_not_real_cells = n_inert_cells + n_frozen_cells
    n_live_cells = n_cells - n_not_real_cells
    pct = lambda n: (100.0 * n / n_cells) if n_cells else 0.0                 # noqa: E731
    print("=" * 84)
    print(f"[865] FIXED-FRAME MINUTE panel from checkpoint provenance "
          f"({_display_path(args.panel_frame_from_checkpoint)}): {args.n_assets} slots, "
          f"{n_active} ACTIVE / {len(inactive_slots)} INACTIVE (zero minute coverage) in "
          f"[{args.start}..{args.end}] — minute `ohlcv` [::{args.bar_minutes}], 853 semantics")
    print(f"[865] ... of the {n_active} ACTIVE: {len(partial)} PARTIAL (listed mid-window: "
          f"pre-listing rows flat-zero => force-HOLD), {len(stale)} STALE-TAIL (delisted "
          f"mid-window: trailing rows FORWARD-FILLED at a frozen price, force-HELD "
          f"under the TASK_868 causal gate) and "
          f"{fz['n_slots']} carrying FROZEN spans on the decision grid (tails AND interior holes). "
          f"'ACTIVE' means 'has at least one real close', NOT 'live all window'.")
    aliased = [c for c in coverage if c.get("aliased")]
    if aliased:
        print(f"[865] ALIAS-UNION slots ({len(aliased)}) — TASK_868 (D31 a1): the slot's recorded "
              f"symbol resolved to MULTIPLE assets rows (ingest identity split) and their rows "
              f"were unioned into one series (overlap agreement verified, rel <= 1e-12):")
        for c in aliased:
            print(f"        slot {c['slot']:>3}  {c['symbol']:<16} asset_ids {c['asset_ids']} "
                  f"db spellings {c['db_symbols']} per-id rows {c['per_id_rows']}")
    print(f"[865] DECISION GRID ({n_dec} x {args.n_assets} = {n_cells} cells):")
    print(f"[865]   INERT  (no price at all => force-HOLD under both operators) : "
          f"{n_inert_cells} ({pct(n_inert_cells):.2f}%)")
    print(f"[865]   FROZEN (a forward-filled price no market printed, force-HELD "
          f"by the TASK_868 causal open-gate; D30 fill still values it) : "
          f"{n_frozen_cells} ({pct(n_frozen_cells):.2f}%) = {n_frozen_tail_cells} trailing "
          f"(delisted) + {n_frozen_interior_cells} INTERIOR "
          f"(a hole that RE-ANCHORS: it CAN fabricate PnL — see stale_tail_rule / DECISIONS D31)")
    print(f"[865]   LIVE   (a real close at that (decision, slot))               : "
          f"{n_live_cells} ({pct(n_live_cells):.2f}%)")
    print(f"[865]   NOT REAL DATA, total                                         : "
          f"{n_not_real_cells} ({pct(n_not_real_cells):.2f}%)")
    if fz["slots"]:
        print(f"[865] FROZEN-SPAN slots ({fz['n_slots']}) — heaviest first; INTERIOR spans are the "
              f"ones a consumer could previously not see at all (no stale_tail, no partial, "
              f"covered_rows == n_rows):")
        for r in fz["slots"][:12]:
            print(f"        slot {r['slot']:>3}  {r['symbol']:<16} {r['frozen_decision_cells']:>6} "
                  f"frozen decision cells ({r['interior']} interior + {r['tail']} trailing) across "
                  f"{r['n_frozen_spans']} span(s)"
                  + ("  [NO REAL PANEL ROW AT ALL]" if r["no_real_panel_rows"] else ""))
    if inactive:
        print(f"[865] INACTIVE slots ({len(inactive)}) — no fills/fees, HOLD-only (env price>0 "
              f"active-mask); every span reported:")
        for c in inactive:
            print(f"        slot {c['slot']:>3}  {c['symbol']:<16} INACTIVE "
                  f"[{c['span'][0]}..{c['span'][1]}]  ({c['reason']})")
    # A slot may be short at the HEAD (listed mid-window => PARTIAL, flat-zero force-HOLD rows) or
    # at the TAIL (delisted mid-window => STALE TAIL, forward-filled rows at a frozen price,
    # force-HELD by the TASK_868 causal open-gate), or both. Reported separately: different keys,
    # different causes, different treatment.
    partial_head = [c for c in coverage if c.get("leading_inactive_rows")]
    if partial_head:
        print(f"[865] PARTIAL-HEAD slots ({len(partial_head)}) — listed mid-window: pre-listing "
              f"rows are flat-zero => INACTIVE (HOLD under BOTH operators via price>0 / "
              f"price-validity), active from the listing bar onward:")
        for c in partial_head:
            print(f"        slot {c['slot']:>3}  {c['symbol']:<16} first covered ts "
                  f"{c['first_covered_ts']} ({c['leading_inactive_rows']} leading inactive rows)")
    if stale:
        print(f"[865] STALE-TAIL slots ({len(stale)}) — the slot's LAST real minute close falls "
              f"BEFORE the window end (delisting / ticker migration). The trailing rows are "
              f"FORWARD-FILLED at that last real close (D30 opt (i): the only treatment measured "
              f"PV-exact; zeroing the tail marks an open position against price 0 and fabricates "
              f"-/+25% of PV on a flat series). The fill still VALUES those rows (PV-exact), but "
              f"under TASK_868 (D31 a2) they are force-HELD: the causal open-gate "
              f"cannot distinguish a delisting from an interior hole without look-ahead — "
              f"disclosed here because they are in NEITHER inactive_slots NOR partial_slots:")
        for c in stale:
            print(f"        slot {c['slot']:>3}  {c['symbol']:<16} last real close ts "
                  f"{c['last_real_ts']} = panel row {c['last_covered_row']} of {c['n_rows'] - 1}; "
                  f"{c['trailing_stale_rows']} trailing rows forward-filled at a frozen price "
                  f"({100.0 * c['trailing_stale_rows'] / max(c['n_rows'], 1):.1f}% of the "
                  f"window)")
    no_real = [c for c in coverage if c.get("no_real_panel_rows")]
    if no_real:
        print(f"[865] ZERO-REAL-PANEL-ROW slots ({len(no_real)}) — every real minute close for these "
              f"slots falls BETWEEN strided panel rows, so 100% of their priced cells are "
              f"forward-filled. They are counted ACTIVE by the loader's coverage definition (>=1 "
              f"real minute close in the window) and are NOT reclassified here (that would change "
              f"the breadth guard's arithmetic — a semantics change, not a disclosure fix):")
        for c in no_real:
            print(f"        slot {c['slot']:>3}  {c['symbol']:<16} {c['real_minute_rows']} real "
                  f"minute rows, 0 real PANEL rows of {c['n_rows']}")
    if not inactive and not partial and not stale and not fz["slots"]:
        print("[865] all slots fully covered on this window (no inactive/partial/frozen spans).")
    print("=" * 84)
    # EMITTED coverage: display-elided frozen_spans (see coverage_for_artifact). Built AFTER every
    # count above, all of which read the complete in-memory coverage.
    cov_art = coverage_for_artifact(coverage)
    inactive = [c for c in cov_art if c["status"] == "inactive"]
    partial = [c for c in cov_art if c["status"] == "partial"]
    stale = [c for c in cov_art if c.get("stale_tail")]
    no_real = [c for c in cov_art if c.get("no_real_panel_rows")]
    panel_info["panel_frame"] = {
        "source": "checkpoint-provenance-fixed-frame (TASK_853 semantics, TASK_865 minute source)",
        # The selection, recorded explicitly (scope 2: no silent source switching). Its ABSENCE on
        # a panel_frame block means the shipped 853 DAILY source.
        "panel_frame_source": "minute",
        "panel_price_source": (f"minute `ohlcv` via eval_vector.load_minute_prices_framed, "
                               f"[::{int(args.bar_minutes)}] stride"),
        "bar_minutes": int(args.bar_minutes),
        "provenance_path": _display_path(args.panel_frame_from_checkpoint),
        "provenance_meta": provenance_meta_block(prov, args.panel_frame_from_checkpoint),
        "n_slots": int(args.n_assets),
        "n_active_slots": int(n_active),
        "n_inactive_slots": int(len(inactive_slots)),
        "n_partial_slots": int(len(partial)),
        "n_stale_tail_slots": int(len(stale)),
        "inactive_slots": list(inactive_slots),
        "inactive_spans": inactive,
        "partial_spans": partial,
        "stale_tail_slots": [c["slot"] for c in stale],
        "stale_tail_spans": stale,
        "stale_tail_rule": STALE_TAIL_RULE + (
            " On the 861 leg-1 window the two ticker migrations (FTM/USD -> S, MATIC/USD -> POL, "
            "both dead in the WHOLE table after 2025-03-31) land here. POL/USD exists in the DB "
            "and S/USD does not, but splicing a successor ticker into a predecessor's slot is a "
            "UNIVERSE change needing a planner ruling on cross-ticker splice provenance — "
            "deliberately NOT done."),
        # FIX PASS — INTERIOR frozen spans, disclosed on their own axis. Same rule as the tails
        # (they are filled identically), but the "a constant span cannot fabricate PnL" argument
        # does NOT cover them: an interior hole RE-ANCHORS. Measured on the 861 leg-1 window: 11 of
        # 64 slots are near-empty 2025-03-31..2025-10-01 in many multi-day sub-gaps (10-95 real rows
        # across 183 days), 2,013 frozen interior decision cells = 1.88% of the grid, with re-anchor
        # jumps of ETHBTC +64.52% / BTCCAD +33.88% / BTCGBP +32.69% / BTCCHF +24.29% /
        # /USDTEUR -7.76%. What to DO about them is a semantics ruling (DECISIONS D31) and is NOT
        # decided here — the fill is unchanged, only the measurement is new.
        "n_frozen_slots": fz["n_slots"],
        "frozen_slots": [r["slot"] for r in fz["slots"]],
        "frozen_span_slots": fz["slots"],
        "n_zero_real_panel_row_slots": len(no_real),
        "zero_real_panel_row_slots": [c["slot"] for c in no_real],
        # D30 amendment 5: the same honesty at CELL level, in the artifact, not only on stdout.
        "n_decision_cells": n_cells,
        "n_inert_decision_cells": n_inert_cells,
        "inert_decision_cell_pct": round(pct(n_inert_cells), 4),
        # THE honest frozen total: trailing AND interior. The stale-tail-only counters are kept
        # beside it (their name is accurate) so a consumer can see the split, but the TOTAL is what
        # the cell-honesty gate reads.
        "n_frozen_decision_cells": n_frozen_cells,
        "frozen_decision_cell_pct": round(pct(n_frozen_cells), 4),
        "n_frozen_stale_tail_decision_cells": n_frozen_tail_cells,
        "frozen_stale_tail_decision_cell_pct": round(pct(n_frozen_tail_cells), 4),
        "n_frozen_interior_decision_cells": n_frozen_interior_cells,
        "frozen_interior_decision_cell_pct": round(pct(n_frozen_interior_cells), 4),
        "n_not_real_decision_cells": n_not_real_cells,
        "not_real_decision_cell_pct": round(pct(n_not_real_cells), 4),
        "n_live_decision_cells": n_live_cells,
        "live_decision_cell_pct": round(pct(n_live_cells), 4),
        # Named so a reader of THIS block knows whether any slot's span list below is elided (the
        # counts never are — frozen_decision_cells refuses an elided list outright).
        "frozen_spans_display_max": FROZEN_SPANS_ARTIFACT_MAX,
        "n_slots_with_elided_frozen_span_list": sum(
            1 for c in cov_art if c.get("frozen_spans_truncated")),
        # TASK_868 (D31 a1) — the alias-union disclosure, panel level (per-slot detail is on
        # each coverage entry: asset_ids / db_symbols / per_id_rows / overlap_agreement).
        "n_aliased_slots": len(aliased),
        "aliased_slots": [int(c["slot"]) for c in aliased],
        "alias_union_rule": ALIAS_UNION_RULE,
        # TASK_868 (D31 a2) — the causal open-gate mask, what it gates and why it is causal.
        # The mask force-HOLDs OPENING on exactly the FROZEN decision cells above (interior
        # AND trailing); VALUATION is untouched (the D30 forward-fill stands, PV-exact).
        "open_gate_rule": OPEN_GATE_RULE,
        "n_open_gated_decision_cells": n_frozen_cells,
        "open_gated_decision_cell_pct": round(pct(n_frozen_cells), 4),
        "decision_cell_rule": (
            "INERT = no price at all at that (decision row, slot) (pre-listing head or a wholly "
            "uncovered slot) => force-HOLD under both operators. FROZEN = the panel carries a "
            "forward-filled price the market never printed at that row; force-HELD by the "
            "TASK_868 causal open-gate (interior AND trailing — at decision time the two are "
            "indistinguishable without look-ahead). The hold covers the WHOLE action — no open, "
            "no close, no rebalance — a held position is carried at the D30 forward-filled "
            "valuation (PV-exact) and realizes at the next real print (see open_gate_rule). "
            "LIVE = the panel row samples a real close for that slot. "
            "The three are disjoint and sum to n_decision_cells. 'not real data' = INERT + FROZEN."),
        "require_active_slots": _resolve_require_active(args),
        "coverage": cov_art,
    }
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
            leaf_eval='model', collect=None, open_gate=None):
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
    kwargs, bit-identical (acceptance iii).

    TASK_861: ``collect`` (default None) is a PURE RECORDER — a dict of two lists that receive
    the per-decision CHOSEN action vector and the post-step realized book. It draws no RNG,
    issues no extra env/model call and changes no control flow, so a collecting walk is
    numerically IDENTICAL to a non-collecting one (acceptance ii).

    TASK_868 (D31 a2): ``open_gate`` (N, A) bool, default None = pre-868 behaviour. Where
    ``open_gate[k]`` is False the slot printed nothing inside decision k's bar (a frozen,
    forward-filled price), so the search's legal mask is restricted to HOLD for that
    (decision, slot) — the SAME force-HOLD the env's own ``price_valid = price > 0`` rule
    applies to inert cells, extended to frozen ones, and CAUSAL (bar k's value reads only
    prints inside bar k). VALUATION is untouched: step_vector still marks the book at the
    filled price."""
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
            # TASK_868 (D31 a2): force-HOLD the frozen cells — restrict the legal mask to
            # HOLD wherever the slot printed nothing inside this decision's bar. Same
            # mechanic as the env's own price_valid gate, one layer up (the frozen price
            # is > 0, so the env's own mask cannot see it).
            if open_gate is not None and not bool(open_gate[k].all()):
                # batch_compute_action_mask returns the FLAT (n_envs, A*5) mask —
                # gate per (slot, action): every action of a gated slot illegal but HOLD.
                n_act = mask.shape[-1] // n_assets
                gated = jnp.repeat(jnp.asarray(~open_gate[k], dtype=bool), n_act)
                hold_flat = (jnp.arange(mask.shape[-1]) % n_act) == HOLD
                mask = jnp.where(gated.reshape((1,) * (mask.ndim - 1) + (-1,)),
                                 hold_flat.astype(mask.dtype), mask)
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
        # TASK_861 pure recording: both arrays are ones the diagnostics below already
        # materialized — appending them cannot perturb the walk (acceptance ii).
        if collect is not None:
            collect["chosen_vectors"].append(av.astype(np.int32).tolist())
            collect["realized_book"].append(sd.astype(np.int32).tolist())
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
    # TASK_859 — trade-activity metadata from the SAME decoded action stream (no new compute,
    # no extra pass): per-asset non-HOLD trade counts + long/short action split. Action codes
    # (jax_trading_env: [0]Sell [1]Hold [2]Buy [3]LimitSell [4]LimitBuy) => long = Buy+LimitBuy,
    # short = Sell+LimitSell. trades_per_asset[i] = n_exec - hold_count[i] (each decision assigns
    # asset i exactly one action). Emitted for BOTH operators / all decision rules (this is the
    # single decode path). Raw counts here; score_tier attaches symbols (needs col_symbols).
    trades_per_asset = (n_exec - hold_count).astype(np.int64)
    return {
        "rets": rets, "n_exec": n_exec, "final_pv": float(pvs[-1]),
        "pct_hold_per_asset": 100.0 * hold_count / denom,
        "pct_hold_agg": 100.0 * float(hold_count.sum()) / (denom * n_assets),
        "action_hist_pct": (100.0 * action_hist / max(action_hist.sum(), 1)).tolist(),
        "avg_long": n_long_total / denom, "avg_short": n_short_total / denom,
        "bankrupt_at": bankrupt_at, "terminated_early_at": terminated_early_at,
        "trades_per_asset": trades_per_asset,
        "activity_n_long": int(action_hist[2] + action_hist[4]),
        "activity_n_short": int(action_hist[0] + action_hist[3]),
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
    # TASK_859 — self-describing trade-activity block (additive; pre-existing keys untouched,
    # so the SHA over the old key set is unchanged). Built from rollout's decoded action stream:
    # per-asset non-HOLD trade counts + the long/short action split. hold_pct is a FRACTION
    # (0..1) = pct_hold_agg/100 by construction; trades_per_step is the mean non-HOLD actions per
    # decision. Post-856-fix a top-m M run trades M assets every step => trades_per_step == M; a
    # 100%-HOLD run => hold_pct == 1.0, n_trades_total == 0 (acceptance ii).
    tpa = np.asarray(rep["trades_per_asset"], dtype=np.int64)
    n_exec = rep["n_exec"]
    denom = max(n_exec, 1)
    n_assets = len(col_symbols)
    n_trades_total = int(tpa.sum())
    per_asset_traded = {sym: int(c) for sym, c in zip(col_symbols, tpa)}
    top5 = dict(sorted(((s, c) for s, c in per_asset_traded.items() if c > 0),
                       key=lambda kv: -kv[1])[:5])
    activity = {
        "hold_pct": 1.0 - n_trades_total / (denom * n_assets),
        "trades_per_step": n_trades_total / denom,
        "n_trades_total": n_trades_total,
        "n_long": int(rep["activity_n_long"]),
        "n_short": int(rep["activity_n_short"]),
        "active_assets": int((tpa >= 1).sum()),
        "top5_traded": top5,
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
        "activity": activity,
    }


# =============================================================================
# TASK_861 — chosen-vector collection (offline search-teacher target stream)
# =============================================================================
# The +8.024 true-env-leaf search (LEDGER 832/E5-inversion3) is the best teacher the program
# has measured, but its leaf is an R-step LOOKAHEAD so it can never trade live. It CAN however
# teach offline: run it once over the panel and record the action vector it CHOOSES at each
# decision — the same semantic object the env executes and the 832 weight-native BC loss
# consumes (one weight vector per decision, NOT ExIt2's visit distribution, whose per-asset
# argmax rounds to 100% HOLD — LEDGER 832/E5-exit2). This module only RECORDS; the consumer is
# imprint.py --teacher-vectors, which re-gates the whole provenance header before using it.

def _scalar(v):
    """JSON-safe python scalar from a config field (bool stays bool; jnp/np 0-d unwraps)."""
    if isinstance(v, bool):
        return bool(v)
    a = np.asarray(v)
    if a.shape != ():
        return a.tolist()
    return int(a) if np.issubdtype(a.dtype, np.integer) else float(a)


def env_flag_block(env_cfg):
    """EVERY EnvConfig field of the walked config, as plain python — the collection's env
    fingerprint. The imprint consumer compares this field-by-field so a target stream recorded
    under different env semantics can never be imprinted silently (task scope 2 fail-loud)."""
    return {f.name: _scalar(getattr(env_cfg, f.name))
            for f in dataclasses.fields(env_cfg)}


def payload_digest(decisions):
    """sha256 over the CANONICAL serialization of the decisions payload (fix pass FIX 2). The
    imprint consumer recomputes it, so a byte-level edit of the target stream — a truncated
    code, a shifted row, a hand-patched vector — is caught before any training starts."""
    blob = json.dumps({k: v for k, v in decisions.items() if k != "note"},
                      sort_keys=True, separators=(",", ":"), default=float)
    return hashlib.sha256(blob.encode()).hexdigest()


def build_chosen_vectors_doc(args, label, panel, collected, operator_block, tier_bp,
                             walk_stats):
    """Assemble the collection artifact: the recorded per-decision chosen vectors plus the
    provenance header the imprint runner gates on (checkpoint, full operator block, db, window,
    panel-frame provenance, env fingerprint, decision count).

    FIX PASS (FIX 2) — the header also carries the COLLECTION WALK'S OWN SCORE
    (``walk_net21`` / ``walk_final_pv`` / ``walk_bankrupt_at``, from the very rollout that
    produced the stream) plus a sha256 of the payload. The imprint consumer replays the stream
    in-env and asserts the replayed floor reproduces ``walk_net21``: one number that
    independently catches an off-pin operator, a corrupted payload and any future silent drift,
    because a stream that does not reproduce the recorded score is not the stream that was
    scored."""
    prices, env_cfg0, state_init, row_ts, col_symbols, panel_info = panel
    warm = int(env_cfg0.window_size)
    n_expected = int(panel_info["n_decisions"])
    n_coll = len(collected["chosen_vectors"])
    n_loaded = int(panel_info["n_loaded_symbols"])
    fee, slip = ((FEE_BPS, SLIP_BPS) if tier_bp else (0.0, 0.0))
    tier_cfg = env_cfg0.replace(fee_bps=fee, slippage_bps=slip)
    rows = list(range(warm, warm + n_coll))
    decisions = {
        "note": "chosen_vectors[k] is the per-asset action vector the operator SELECTED at "
                "decision k (panel row row_index[k]); codes are the env's "
                "[0]Sell [1]Hold [2]Buy [3]LimitSell [4]LimitBuy — the same object "
                "step_vector executes and the 832 BC loss one-hots. realized_book[k] is the "
                "post-step pos_side (0 flat / 1 long / 2 short), DIAGNOSTICS ONLY.",
        "row_index": rows,
        "timestamp": [int(row_ts[r]) for r in rows],
        "chosen_vectors": collected["chosen_vectors"],
        "realized_book": collected["realized_book"],
    }
    return {
        "schema": CHOSEN_VECTORS_SCHEMA,
        "provenance": {
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "producer": "experiments/835_stage4_eval/eval_vector.py --collect-chosen-vectors",
            "tag": args.tag,
            "checkpoint": label["path"], "checkpoint_label": label["name"],
            "db": args.db, "db_abspath": os.path.abspath(args.db),
            "window": [args.start, args.end],
            "window_utc": _window_utc(args),          # TASK_869: resolved epoch bounds
            "bar_minutes": int(args.bar_minutes),
            "n_assets": int(args.n_assets),
            "panel_source": getattr(args, "panel_source", "minute"),
            "panel_frame_from_checkpoint": getattr(args, "panel_frame_from_checkpoint", None),
            "panel_frame": panel_info.get("panel_frame"),
            # TASK_868 (D31 a1/a2 rider): the alias-union + causal open-gate disclosure on the
            # canon UNPINNED minute panel (panel_frame is null there — the frame block only
            # exists under --panel-frame-from-checkpoint, and it carries its own copy). None
            # on the daily paths (no union, no gate).
            "asset_aliases": panel_info.get("asset_aliases"),
            "n_aliased_slots": panel_info.get("n_aliased_slots"),
            "alias_union_rule": panel_info.get("alias_union_rule"),
            "open_gate_rule": panel_info.get("open_gate_rule"),
            "n_open_gated_decision_cells": panel_info.get("n_open_gated_decision_cells"),
            "symbols_cli": list(args.symbols) if args.symbols else None,
            "loaded_symbols": list(col_symbols[:n_loaded]),
            "col_symbols": list(col_symbols),
            "n_loaded_symbols": n_loaded,
            "padded": bool(panel_info["padded"]),
            "n_rows": int(panel_info["n_rows"]),
            "warm": warm,
            "start_row": warm,
            "n_decisions_expected": n_expected,
            "n_decisions_collected": n_coll,
            "partial": bool(n_coll < n_expected),
            "max_decisions": args.max_decisions,
            "position_size_pct": float(env_cfg0.position_size_pct),
            "eval_seed": int(args.eval_seed),
            "cost_tier_bp": int(tier_bp),
            "fee_bps": float(fee), "slip_bps": float(slip),
            "operator": operator_block,
            "env_flags": env_flag_block(tier_cfg),
            # FIX 2 — the walk's OWN score, from the rollout that produced this exact stream.
            # The consumer replays the stream and must reproduce walk_net21 (rel 1e-4).
            "walk_note": f"scores of the net@{int(tier_bp)} rollout that RECORDED this stream "
                         f"(835 score_tier on the same rets); imprint.py --teacher-vectors "
                         f"replays the stream in-env and refuses unless its teacher floor "
                         f"reproduces walk_net21",
            "walk_net21": walk_stats.get("sharpe_full"),
            "walk_final_pv": walk_stats.get("final_pv"),
            "walk_bankrupt_at": walk_stats.get("bankrupt_at"),
            "walk_terminated_early_at": walk_stats.get("terminated_early_at"),
            "walk_steps_completed": walk_stats.get("steps_completed"),
            "walk_pct_hold_agg": walk_stats.get("pct_hold_agg"),
            "payload_sha256": payload_digest(decisions),
        },
        "decisions": decisions,
    }


def preflight_collect_path(path):
    """FIX 3 — prove the collection output is WRITABLE before the walk starts.

    The stream is written only after the full rollout AND all scoring, so a bad path (or an
    unwritable parent) discovered at write time discards the whole teacher pass — on the real
    leg that is a multi-hour 1670-decision true-env-leaf walk. Create the parent directory and
    actually open the target for append here; remove it again if we created it. Returns the
    resolved absolute path, raises OSError/ValueError on anything unusable."""
    if not str(path).strip():
        raise ValueError(
            "--collect-chosen-vectors was given an EMPTY path; refusing to guess an output "
            "location for the teacher target stream.")
    out = os.path.abspath(path)
    if os.path.isdir(out):
        raise ValueError(f"--collect-chosen-vectors: {out} is a DIRECTORY, not a file path.")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    existed = os.path.exists(out)
    with open(out, "a"):
        pass
    if not existed:
        os.remove(out)
    return out


def write_chosen_vectors(path, doc):
    """Write the collection artifact (JSON). Fail-loud on an empty/blank path."""
    if not str(path).strip():
        raise RuntimeError(
            "--collect-chosen-vectors was given an EMPTY path; refusing to guess an output "
            "location for the teacher target stream.")
    out = os.path.abspath(path)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as fh:
        json.dump(doc, fh, indent=1, default=float)
    p = doc["provenance"]
    print(f"[861] collected {p['n_decisions_collected']}/{p['n_decisions_expected']} chosen "
          f"action vectors (cost tier net@{p['cost_tier_bp']}, rows "
          f"{p['start_row']}..{p['start_row'] + p['n_decisions_collected'] - 1}, "
          f"{p['n_assets']} slots) -> {_display_path(out)}")
    wn = "None" if p["walk_net21"] is None else f"{p['walk_net21']:+.6f}"
    print(f"[861] recorded walk score (the consumer's integrity anchor): "
          f"net@{p['cost_tier_bp']} {wn}  final_pv {p['walk_final_pv']}  "
          f"bankrupt_at {p['walk_bankrupt_at']}  payload sha256 {p['payload_sha256'][:16]}...")
    if p["partial"]:
        print(f"[861] WARN: the collection walk terminated EARLY (env done / bankruptcy) — the "
              f"stream covers {p['n_decisions_collected']} of {p['n_decisions_expected']} "
              f"decisions. imprint.py --teacher-vectors will REFUSE it unless its own panel "
              f"yields exactly that many decision rows.")
    return out


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


def compute_teacher_floor(prices, env_cfg0, warm, n_decisions, period_minutes,
                          hold_mask=None):
    """TASK_848 scope 3 — the on-window teacher floor. Walks 832's weight-native ladder
    teacher (``_make_weight_native_teacher_fn``, same sentinels) on the SAME decision rows
    [warm, warm+N) the operator replays, through the SAME env config (its cost tier + weight
    flag). Returns ({cost: {sharpe, final_pv, ...}}, book_warm). Fail-loud if the book warmup
    exceeds the env window (some decision rows would read strategy-warmup signs).

    D30 rider (a) — ``hold_mask`` (N, A) bool, True where the panel has NO real price at that
    (decision row, slot) (i.e. the polarity ``walk_env(hold_mask=...)`` wants: True => force HOLD).
    On a FRAMED panel the pre-listing HEADS are flat-zero by design (the 853 leak fix), and this
    walk hands its vector straight to ``step_vector``, bypassing the env's
    ``price_valid = current_prices > 0`` action mask — exactly like the policy-argmax replay above,
    which already applies the rule. As of TASK_869 (D31 (a3)) gated on BOTH framed pins (minute
    AND daily); ``None`` = the unmasked walk, which only the UNPINNED DAILY path keeps (the
    unpinned MINUTE path passes 868's causal open-gate as the mask — uniform there by ruling).

    WHAT THIS RIDER ACTUALLY BUYS, restated honestly (fix pass — the previous docstring asserted a
    mechanism that CANNOT OCCUR on any panel this loader emits, and quoted numbers from a build
    that no longer exists):
      * The rider was written against the ZEROED-tail build. Its cited proof (tier-0bp Sharpe
        -15.83 / final_pv 0.00 / bankrupt_at 45 unmasked, vs -1.50 / 96,094 masked) was measured on
        THAT panel. Under D30 opt (i) the tail is FORWARD-FILLED, so the real->0 transition that
        produced the blow-up no longer exists.
      * The blow-up needs a real->0 transition: ``log(max(0, 1e-12)) = -27.6`` only becomes a ~-26
        log-RETURN if the preceding row had a real price. ``_ffill_interior_only`` makes that
        impossible — only LEADING zeros survive the fill, and a leading run of zeros gives
        ``log(1e-12) - log(1e-12) = 0``.
      * On a leading flat-zero run ``imprint.book_target_signs`` returns sign 0, so the ladder emits
        HOLD on every decision of such a slot anyway (instrumented action histogram [0, 50, 0] for
        the fixture's two zero-coverage slots).
      * Consequently masked and unmasked are BIT-IDENTICAL on both fixture pins (asserted by the
        865 smoke's teacher-floor leg, which now checks exactly that instead of claiming a rescue).
    So this is DEFENCE IN DEPTH, not a live bug fix: it keeps the ladder teacher reading the SAME
    price-validity rule the search operator and the policy-argmax replay already apply, so a future
    loader (or a hand-built panel) that CAN emit a non-leading zero cannot short at price 0 through
    this door.

    TASK_869 (D31 (a3) scope 2): the mask is no longer minute-pin-only — the DAILY-framed floor
    gets the SAME mask (rider (a)'s semantics are uniform across both pin paths; the unmasked
    daily walk was the same defect class). The minute-path inertness finding did NOT transfer by
    assumption: the 865 smoke measures the daily path directly (masked == unmasked, bit-identical,
    on the shipped daily panel — the daily fill is interior-only too, so only leading zeros
    survive and those read as ladder sign 0 => HOLD anyway), with a synthetic non-leading-zero
    CONTROL proving the mask still bites there. The UNPINNED daily path stays unmasked and
    bit-identical; the unpinned MINUTE path is masked only by 868's open-gate (no
    price-validity half), exactly as the operator and the replay read it there."""
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
        wrep = imp.walk_env(prices, cfg_c, warm, n_decisions, wn_fn,
                            hold_mask=hold_mask)
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

    # TASK_868 (D31 a2): the causal OPEN-GATE, present on BOTH minute panels (the pinned
    # minute loader returns it; the canon unpinned panel captures it in build_panel). It is
    # a (T, A) bool — True iff the slot printed a real close inside that panel row's bar.
    # Where it is False the panel price is forward-filled: fine for VALUATION (D30, PV-exact)
    # but OPENING there is force-HELD, uniformly across the search operator, the policy-argmax
    # replay and the teacher-floor walk. None on the DAILY paths (bit-identical, gate (v)).
    open_gate = panel_info.get("open_gate")
    gate_dec = (np.asarray(open_gate)[warm:warm + n_decisions]
                if open_gate is not None else None)

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
        # TASK_868 (D31 a2): the causal open-gate applies to the replay too — a FROZEN cell
        # (price > 0 but no print inside the bar) is force-HELD on BOTH minute panels, the
        # SAME rule the search operator now reads via rollout's open-gate. Daily/unpinned-
        # daily replays carry no gate and are byte-identical.
        if gate_dec is not None and (~gate_dec).any():
            policy_actions = np.array(policy_actions, copy=True)
            policy_actions[~gate_dec] = HOLD

    # TASK_848 scope 3: the on-window teacher floor (weight-native ladder, same rows/env).
    teacher_floor = book_warm = None
    if args.teacher_floor:
        # D30 rider (a): on a FRAMED panel the ladder teacher reads the SAME price-validity
        # rule the search operator and the policy-argmax replay already apply, so it cannot act
        # on a cell with no market. DEFENCE IN DEPTH, not a live rescue — on every panel these
        # loaders can emit the mask is PROVABLY INERT (a flat-zero head gives the ladder sign 0
        # => HOLD, and the forward fill makes a real->0 transition impossible). See
        # compute_teacher_floor's docstring for the full measurement; the "-15.83 / 0.00 /
        # bankrupt_at 45" the rider was written against was the ZEROED-tail build, which D30
        # opt (i) replaced.
        # TASK_869 (D31 (a3) scope 2): the mask now covers the DAILY pin too — rider (a)'s
        # semantics are uniform across both pin paths (D30's uniformity ruling), and the
        # daily-framed teacher floor walking UNMASKED was the same defect class. MEASURED on
        # the daily path (the minute-path inertness finding does NOT transfer by assumption —
        # the 865 smoke's daily teacher-floor leg measures it): bit-identical masked vs
        # unmasked on the shipped daily panel, for the same structural reason (interior-only
        # forward fill leaves only LEADING zeros, and a leading flat-zero run reads as ladder
        # sign 0 => HOLD anyway), so this ships as defence-in-depth exactly as 865's minute
        # rider did. The price-validity half is FRAMED-pins-only; the UNPINNED DAILY path
        # carries no mask and is bit-identical (865 canon-parity invariant). The unpinned
        # MINUTE path is masked the way 868 made it — no price-validity half, but the D31 (a2)
        # open-gate below applies there uniformly (gate_dec is not None).
        _pf = panel_info.get("panel_frame") or {}
        _floor_mask = None
        _pv_masked_cells = 0
        # The 853 convention: an ABSENT ``panel_frame_source`` key in the frame block IS the
        # daily pin (the key exists only on the minute sibling). _pf is non-empty only under
        # --panel-frame-from-checkpoint, so the unpinned paths skip the PRICE-VALIDITY half
        # entirely (the open-gate half below still applies on the canon unpinned minute panel).
        _pf_source = (_pf.get("panel_frame_source") or "daily") if _pf else None
        if _pf_source in ("minute", "daily"):
            _dec_px = np.asarray(prices[warm:warm + n_decisions])
            _pv_mask = _dec_px <= 0
            if _pv_mask.any():
                _floor_mask = _pv_mask
                _pv_masked_cells = int(_pv_mask.sum())
        # TASK_868 (D31 a2): on the minute panels the teacher-floor walk reads the SAME causal
        # open-gate the operator and the replay read (uniform OPENING rule). Daily panels carry
        # no gate (gate_dec is None there) — the daily floor's mask is the price-validity rule
        # above only.
        _gate_masked_cells = 0
        if gate_dec is not None:
            _gate_hold = ~gate_dec
            if _gate_hold.any():
                _gate_masked_cells = int(_gate_hold.sum())
                _floor_mask = _gate_hold if _floor_mask is None else (_floor_mask | _gate_hold)
        teacher_floor, book_warm = compute_teacher_floor(
            prices, env_cfg0, warm, n_decisions, period_minutes,
            hold_mask=_floor_mask)

    # TASK_861: pure recorder on the PINNED collection tier (net@21). None everywhere else, and
    # None entirely when the flag is absent => the walks are bit-identical to HEAD.
    collect_path = getattr(args, "collect_chosen_vectors", None)
    collected = None

    tiers = {}
    for c in GATE_COSTS:
        fee, slip = (0.0, 0.0) if c == 0 else (FEE_BPS, SLIP_BPS)
        # cost tier enters via the env config ONLY; identical panel + seed per tier
        # (the 830/831 separate-rollout convention — fee-aware trajectories differ).
        env_cfg = env_cfg0.replace(fee_bps=fee, slippage_bps=slip)
        collect = ({"chosen_vectors": [], "realized_book": []}
                   if (collect_path is not None and c == COLLECT_TIER_BP) else None)
        rep = rollout(params, search_obj, prices, env_cfg, state_init,
                      env_cfg0.n_assets, n_decisions, args.eval_seed,
                      args.max_positions, args.min_cash_threshold,
                      policy_actions=policy_actions, leaf_eval=args.leaf_eval,
                      collect=collect, open_gate=gate_dec)
        if collect is not None:
            collected = collect
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
        "window_utc": _window_utc(args),              # TASK_869: resolved epoch bounds
        "eval_seed": args.eval_seed,
        # TASK_868: ``open_gate`` is the in-memory (T, A) mask for the rollout layer — NOT
        # JSON (a numpy bool matrix; its scalar disclosure is recorded beside it).
        "panel": {k: v for k, v in panel_info.items() if k != "open_gate"},
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
            # TASK_869: the walk's hold-mask DISCLOSED (D30 rider (a) + D31 (a3) scope 2) —
            # which pin paths it applies on and how many decision cells it force-HOLDs.
            "hold_mask": {
                "rule": "force-HOLD the whole action where the decision row has no real price "
                        "(flat-zero) — the SAME price-validity rule the search operator and the "
                        "policy-argmax replay apply — OR-ed with the D31 (a2) causal open-gate on "
                        "EVERY minute panel (pinned AND canon unpinned: 868's gate is uniform "
                        "there, so n_open_gate_masked_cells is legitimately > 0 on an unpinned "
                        "minute run). The price-validity half applies on the framed pins only "
                        "(minute AND daily); the unpinned DAILY path carries no mask at all and "
                        "is bit-identical.",
                "panel_frame_source": _pf_source,
                "n_price_validity_masked_cells": _pv_masked_cells,
                "n_open_gate_masked_cells": _gate_masked_cells,
                "n_masked_decision_cells": (
                    int(_floor_mask.sum()) if _floor_mask is not None else 0),
            },
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
    # TASK_861 scope 1: write the collected chosen-vector stream. Deliberately writes NOTHING
    # into ``result`` — the scorecard JSON stays byte-identical to a flag-OFF run so acceptance
    # (ii) is a plain file diff; all provenance lives in the collection artifact itself.
    if collect_path is not None:
        if collected is None:
            raise RuntimeError(
                f"--collect-chosen-vectors: no walk ran on the pinned collection tier "
                f"net@{COLLECT_TIER_BP} (cost tiers were {list(GATE_COSTS)}) — nothing recorded. "
                f"Refusing to write an empty teacher target stream.")
        write_chosen_vectors(collect_path, build_chosen_vectors_doc(
            args, label, panel, collected, result["operator"], COLLECT_TIER_BP,
            tiers[str(COLLECT_TIER_BP)]))

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
    # TASK_868 (D31 a2): the canon UNPINNED minute path captures the causal open-gate too
    # (uniform with the pinned minute loader — the ruling's (a2) is explicit that the mask
    # is uniform on BOTH minute loaders). The loader itself is repaired by (a1)'s alias
    # union; capture_open_gate makes it ALSO return the pre-fill real-print mask + the
    # per-slot alias disclosure. Canon TRAINING callers of create_jax_env_historical do
    # not pass the flag and are byte-identical (canon 3-tuple, no extra memory).
    states0, prices, env_cfg, open_gate, asset_aliases = create_jax_env_historical(
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
        capture_open_gate=True,
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
        # TASK_868 (D31 a2): the causal open-gate travels in panel_info for the
        # rollout/evaluate layer and is STRIPPED before JSON emission (see
        # evaluate_checkpoint); the scalar + alias disclosure below is what the
        # artifact records. (This is the canon UNPINNED minute panel — the framed
        # loaders route through _finalize_panel and add their own keys.)
        "open_gate": open_gate,
        # TASK_868 (D31 a1): per-slot alias-union disclosure on the canon unpinned path.
        "asset_aliases": asset_aliases,
        "n_aliased_slots": sum(1 for a in asset_aliases if a.get("aliased")),
        "alias_union_rule": ALIAS_UNION_RULE,
        "open_gate_rule": OPEN_GATE_RULE,
        "n_open_gated_decision_cells": int(
            (~np.asarray(open_gate)[warm:warm + n_decisions]).sum()),
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
    day's minute last-of-day is NOT clipped mid-day. The gate has always been UTC; as of
    TASK_869 the loaders' ``_to_epoch`` is UTC too, so the gate and the panels now differ
    ONLY in the gate's end-of-day extension (a gate-only choice affecting boundary-day
    close bucketing, not the panels)."""
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
    conn = _ro_connect(db_path)
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
        conn0 = _ro_connect(args.db)
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
                         "1440). NOTE (measured 2026-07-30): the 858 backfill carried the minute "
                         "`ohlcv` table to 2026-03-31T23:59 UTC, so 'the minute table no longer "
                         "covers the OOS window' is NO LONGER true as a blanket claim; PER-SYMBOL "
                         "minute coverage still varies (see --panel-frame-source minute's "
                         "stale-tail / late-listing disclosure), so choose the source per window on "
                         "measured coverage, not on that retired assumption")
    ap.add_argument("--panel-frame-from-checkpoint", default=None,
                    dest="panel_frame_from_checkpoint",
                    help="TASK_853: path to a JSON recording the checkpoint's ORDERED training "
                         "symbol frame ('frame_symbols' or 'symbols'). Builds the fixed n_assets-"
                         "slot daily panel in that EXACT order (slot k == asset k, matching the "
                         "positionally universe-locked encoder) and marks zero-coverage slots "
                         "INACTIVE via the env's existing price>0 active-mask (no fills/fees, "
                         "HOLD-only) — the OOS-read mechanic. Requires --bar-minutes 1440. "
                         "Default OFF = the minute/daily paths are byte-identical to HEAD")
    ap.add_argument("--panel-frame-source", choices=["daily", "minute"], default="daily",
                    dest="panel_frame_source",
                    help="TASK_865 (D29): which table the FIXED FRAME is read from. 'daily' "
                         "(default = the shipped 853 path, byte-identical) reads ohlcv_daily and "
                         "requires --bar-minutes 1440; 'minute' reads the minute `ohlcv` table "
                         "with the canon [::bar_minutes] stride — the only pin that is "
                         "apples-to-apples with the minute-panel-native +0.839 bar on universe, "
                         "source AND breadth at once (62/64 vs 33/64 frame slots on the leg-1 "
                         "window). Only meaningful with --panel-frame-from-checkpoint")
    ap.add_argument("--require-active-slots", type=parse_require_active_slots,
                    default=REQUIRE_ACTIVE_SLOTS_DEFAULT, dest="require_active_slots",
                    help=f"D30 rider (b): REFUSE a pinned run (minute OR daily) whose frame is too "
                         f"inert to produce a comparable number. A fraction in (0,1] is a FRACTION "
                         f"of the frame's slots, an integral value >1 an absolute slot count, "
                         f"exactly 0 disables; anything else (negative, fractional-above-1, "
                         f"nan/inf, non-numeric) is REFUSED by name. Default "
                         f"{REQUIRE_ACTIVE_SLOTS_DEFAULT} — the 861 leg-1 pin is 64/64 (passes), "
                         f"the 2021-2022 WF fold is 50/64 (refuses without an explicit override), "
                         f"the 8-slot fixture frame is 6/8 by design (refuses; the fixture smoke "
                         f"passes 0). Without this, a half-dead panel exits 0 with a publishable "
                         f"net@21 — the 64->33 breadth confound D29 exists to prevent")
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
    ap.add_argument("--collect-chosen-vectors", default=None, dest="collect_chosen_vectors",
                    help="TASK_861: PURE RECORDER — write the per-decision action vector the "
                         f"operator CHOOSES on the net@{COLLECT_TIER_BP} walk (plus the realized "
                         "book, diagnostics only) to this JSON, with a provenance header "
                         "(checkpoint, full operator block, db, window, panel frame, env "
                         "fingerprint, decision count). Consumed by imprint.py "
                         "--teacher-vectors as the offline search-teacher target stream. "
                         "Records only: the walk, the scorecard and this module's JSON output "
                         "are IDENTICAL with and without the flag; flag absent = byte-identical "
                         "module behavior. Requires --operator search, refuses a blank path, and "
                         "PROVES the path writable before the walk starts (fix pass FIX 3). The "
                         "header carries the walk's own net@21 / final_pv / bankrupt_at plus a "
                         "payload sha256 so the consumer can verify the stream (FIX 2)")
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
    # TASK_865 fail-closed coherence: --panel-frame-source only selects the FRAME's table, so
    # passing it without a frame pin would silently do nothing while the invocation claims a
    # source. Refuse before any panel build (the 861/863 empty-path precedent).
    if (getattr(args, "panel_frame_source", "daily") != "daily"
            and not args.panel_frame_from_checkpoint):
        ap.error(
            f"--panel-frame-source {args.panel_frame_source} requires "
            f"--panel-frame-from-checkpoint: it chooses which table the FIXED FRAME is read from "
            f"and is inert without a pin. Pass the provenance JSON, or drop the flag (the "
            f"unpinned panel source is --panel-source).")

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
    # TASK_861 fail-loud: one collection path can only hold ONE walk's target stream; with N
    # checkpoints each would silently overwrite the previous one and the imprint would train on
    # whichever ran last (a wrong-teacher result, not a crash).
    if args.collect_chosen_vectors is not None and len(ckpts) > 1:
        ap.error(
            f"--collect-chosen-vectors takes a single walk but {len(ckpts)} checkpoints were "
            f"given; each would overwrite the same file. Run one checkpoint per collection "
            f"(the teacher pass is one deterministic walk).")
    # FIX 3 (fix pass): the flag must FAIL CLOSED. `--collect-chosen-vectors ""` used to run the
    # whole walk, write nothing and exit 0 — on the pre-registered real leg 1 an unset
    # $TEACHER_OUT would silently discard a multi-hour true-env-leaf teacher pass. Validate the
    # path here, and PROVE it writable before a single decision is stepped (the stream is
    # serialized only after the walk AND all scoring, so a late failure costs the whole pass).
    if args.collect_chosen_vectors is not None:
        if not args.collect_chosen_vectors.strip():
            ap.error("--collect-chosen-vectors was given an EMPTY path; pass a real output file "
                     "(an unset shell variable would otherwise discard the whole teacher pass).")
        if args.operator != "search":
            ap.error(
                f"--collect-chosen-vectors records the SEARCH teacher's chosen vectors but "
                f"--operator is '{args.operator}'; a policy-argmax walk would silently yield a "
                f"'chosen vectors' file that is just the student's own argmax.")
        try:
            _pre = preflight_collect_path(args.collect_chosen_vectors)
        except (OSError, ValueError) as e:
            ap.error(f"--collect-chosen-vectors path is not writable: {type(e).__name__}: {e}")
        print(f"[861] collection output preflight OK (writable before the walk): "
              f"{_display_path(_pre)}")

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
            "window_utc": _window_utc(args),          # TASK_869: resolved epoch bounds
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
