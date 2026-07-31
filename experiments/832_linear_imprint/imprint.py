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

TASK_863 (D28 opt (a)) — ``--panel-frame-from-checkpoint <provenance.json>`` pins the panel to a
checkpoint's RECORDED training symbol frame with the SHIPPED TASK_853 semantics (the loader and
frame validator are CALLED from ``experiments/835_stage4_eval/eval_vector.py``, not re-derived):
a fixed ``n_assets``-slot panel in the frame's exact positional order, zero-coverage and
pre-listing PANEL CELLS left flat-zero and forced to HOLD on every in-env walk, dual-disclosed on
stdout AND in the artifact. (Fix pass FIX C — precise claim: the flat-zero thing is the PANEL
cell. The ENCODED features of a fully-inactive slot are CONSTANT across every decision but NOT
all-zero: the guarded vol-ratio channels 8-11 carry a constant 0.2 = clip(1.0,0,5)/5 out of the
0/0 volume ratio, i.e. 896 of 6048 feature cells per inactive slot on the fixture panel. Constant
=> no signal, so there is no leak; the claim is GUARDED, not merely restated, by
``check_inactive_slot_features`` on every pinned run.) Needed because the 858 backfill moved the
coverage ranking, so the
unpinned ``ORDER BY COUNT(*) DESC LIMIT n_assets`` universe no longer reproduces
``runs/832_e5/frame_provenance_imprint.json`` (59/64 positional slots differ) — and slot k IS
asset k positionally, so an unpinned imprint would train a net whose slots do not line up with
the references it is graded against. The 861 provenance gate is FAIL-CLOSED in BOTH directions:
a frame-pinned collection may only be imprinted by an identically-pinned imprint (frame identity
= symbol list AND order; the provenance PATH may differ) and vice versa. Default OFF (no flag) =
bit-identical to the shipped module.

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
import hashlib
import importlib.util
import json
import subprocess
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
    EnvConfig, reset, step_vector, get_current_prices, compute_portfolio_value,
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
# TASK_864: the CHECKED-IN parity baseline vs the untracked file every ordinary run writes.
# The baseline is the weight-ON / order-code deliverable (SPEC.md "Committed fixture artifact"),
# NOT the flagless default -- so the flagless default must never be able to land on it. Promoting
# requires --write-parity-baseline; everything else writes PARITY_LOCAL_NAME (gitignored).
PARITY_BASELINE_NAME = "results_fixture_smoke.json"
PARITY_LOCAL_NAME = "results_fixture_smoke_local.json"
# TASK_864 fix pass FIX 1: the gate above only stopped the WRONG DESTINATION; it did not pin the
# RECIPE. The CANON promoting recipe is the flagless fixture default with weight-rebalance ON --
# expressed as the argv delta below, so the EXPECTED values are re-derived from this module's own
# argparse defaults (see _canon_recipe_args) instead of being a second hardcoded copy that can
# drift. Anything else -- flagless, weight-native, off-default seed/iters/window/sizing -- REFUSES.
CANON_BASELINE_ARGV = ("--vector-weight-rebalance",)
CANON_BASELINE_RECIPE = ("python experiments/832_linear_imprint/imprint.py "
                         "--vector-weight-rebalance --write-parity-baseline")
# `out` is the destination (already refused when explicit) and the promote flag itself is the
# delta; every OTHER argparse dest is compared, because every other one reaches the artifact.
CANON_RECIPE_EXEMPT_FIELDS = ("write_parity_baseline", "out")
FEE_BPS, SLIP_BPS = 16.0, 5.0          # net@21 = fee16 (Kraken taker) + slip5 bp/side
GATE_COSTS = (0, 21)                    # cost tiers, mirroring 830/831
N_ACTIONS_PER_ASSET = 5
HOLD_ACTION_CODE = 1                    # == eval_vector.HOLD; {0,3}=short {2,4}=long
# TASK_863 fix pass FIX C — the ONLY encoder channels a fully-inactive (flat-zero panel) slot may
# carry a non-zero value on: vol_ratio_5/15/30/60 (jax_encoder.py:239-242), whose 0/0 volume ratio
# is guarded to 1.0 and normalised to clip(1.0,0,5)/5 = 0.2. Every other channel of a dead slot is
# exactly 0, and all of them are CONSTANT across decisions (=> no information, no leak). Used by
# check_inactive_slot_features, which REFUSES any pinned run that violates either half.
INACTIVE_SLOT_CONSTANT_CHANNELS = (8, 9, 10, 11)

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
# TASK_863 — PINNED-FRAME panel (--panel-frame-from-checkpoint): the 853 mechanic
# =============================================================================
# The encoder kernel is 27 feats x n_assets SLOTS, so slot k IS asset k positionally. The default
# panel above derives its universe from ``ORDER BY COUNT(*) DESC LIMIT n_assets`` over the given
# window (vectorbt_env.py:578-596) — a ranking the 858 backfill MOVED, so it no longer reproduces
# the frame ``runs/832_e5/ckpt_imprint_full`` was trained on (59/64 slots differ; D28). Under the
# flag the panel is instead built on the checkpoint's RECORDED frame, in its EXACT order, with
# zero-coverage / pre-listing cells INACTIVE.
#
# NOT re-derived: ``eval_vector._load_frame_provenance`` (frame schema + fail-loud on identity
# scrambling: absent list, duplicate symbol, wrong length) and ``eval_vector.load_daily_prices_
# framed`` (one column per slot in order, unknown symbol refused, zero-coverage columns left
# flat-zero, interior-only forward fill so pre-listing rows stay flat-zero) are CALLED here — the
# shipped, proven 853 implementations. This module adds only (a) the EnvConfig the imprint walks
# (mirroring create_jax_env_historical's block, since the framed loader replaces that loader) and
# (b) the price-validity force-HOLD, which 853 applies in eval_vector.evaluate_checkpoint: every
# in-env walk here supplies its action vector DIRECTLY to step_vector and therefore bypasses the
# env's compute_action_mask, so the SAME ``price_valid = price > 0`` rule is applied per
# (decision, slot) to the teacher targets, the injected BC labels and the sanity replay.
#
# CONSEQUENCE, disclosed: like 853's, this panel is DAILY-SOURCED (``ohlcv_daily``) and therefore
# requires --bar-minutes 1440. The unpinned path stays minute-sourced (PIN b) and untouched. Both
# 861 legs must pin (the provenance gate below refuses any pinned/unpinned mix), so leg 1's
# framed panel and leg 2's framed panel come out of the SAME loader on the SAME window.

def load_framed_panel(args):
    """Fixed-frame analogue of ``load_training_panel``. Returns
    (prices_j (T,A) float32 jnp, logc (T,A) float64 np, env_cfg, frame) where ``frame`` carries
    the ordered frame symbols, the panel's row-timestamp axis, the per-slot coverage report and
    the 853-shaped ``panel_frame`` artifact block.

    TASK_865: ``--panel-frame-source minute`` reads the frame from the MINUTE ``ohlcv`` table via
    ``eval_vector.load_minute_prices_framed`` (again CALLED, not re-derived) — same slot-positional
    semantics, no --bar-minutes 1440 requirement (the stride is the canon minute resample).
    Default 'daily' = the shipped 863 path, bit-identical."""
    ev = _load_eval_module()
    frame_source = getattr(args, "panel_frame_source", "daily")
    # D30 rider (b): unset (None / attribute absent) means "the shared default", NOT "disabled" —
    # only an explicit 0 disables the breadth refusal. Resolved through eval_vector's own helper so
    # the domain check (0 | (0,1] | integral >= 1; nan/inf/negative/"0" refused by name) is the SAME
    # one the CLI applies, and this file cannot drift from it.
    req_active = ev._resolve_require_active(args)
    if frame_source == "daily" and int(args.bar_minutes) != 1440:
        raise RuntimeError(
            f"--panel-frame-from-checkpoint requires --bar-minutes 1440 (got {args.bar_minutes}); "
            f"the frame is read from the daily `ohlcv_daily` table (the same rule as the shipped "
            f"853 path, eval_vector._build_panel_framed) and any other stride would be an "
            f"incoherent second resample of already-daily rows. (Pass --panel-frame-source minute "
            f"to pin the frame on the minute `ohlcv` table at this stride instead.)")
    frame_symbols, prov = ev._load_frame_provenance(
        args.panel_frame_from_checkpoint, int(args.n_assets), None)
    if frame_source == "minute":
        prices_np32, loaded_syms, row_ts, coverage = ev.load_minute_prices_framed(
            args.db, args.start, args.end, frame_symbols, int(args.bar_minutes),
            require_active_slots=req_active)
    else:
        # FIX PASS — rider (b) on the DAILY pin too. It shipped minute-only, so ``req_active`` was
        # computed here and then never used: a daily-pinned fixture run at 5/8 active slots exited 0
        # with a full scorecard and no ``require_active_slots`` key at all, while the identical frame
        # on the minute source was refused. The real daily pin is 33/64 = 0.52 and switching the
        # manifest's source to daily is a one-word change, so the floor must hold on both paths.
        prices_np32, loaded_syms, row_ts, coverage = ev.load_daily_prices_framed(
            args.db, args.start, args.end, frame_symbols, require_active_slots=req_active)
    # Fail-loud on identity scrambling: the framed loader must hand back EXACTLY the recorded
    # frame, one column per slot, in order. (It refuses unknown symbols upstream; this is the
    # positional assertion — a silently dropped or reordered column would misalign every slot.)
    if list(loaded_syms) != list(frame_symbols) or prices_np32.shape[1] != int(args.n_assets):
        raise RuntimeError(
            f"--panel-frame-from-checkpoint: the framed loader returned "
            f"{prices_np32.shape[1]} column(s) for {len(frame_symbols)} recorded slot(s) "
            f"(--n-assets {args.n_assets}) / a different symbol order:\n"
            f"  recorded frame {list(frame_symbols)}\n  loaded columns {list(loaded_syms)}\n"
            f"  Slot k IS asset k positionally — refusing to imprint on a scrambled frame.")
    # The EnvConfig create_jax_env_historical would have built for these arguments (env_factory.py
    # :422-450), reproduced here because the framed loader REPLACES that loader. Every field not
    # listed is the EnvConfig default, which is also env_factory's default for that field.
    env_cfg = EnvConfig(
        n_assets=int(args.n_assets),
        initial_cash=100_000.0,
        fee_bps=float(args.fee_bps),
        slippage_bps=float(args.slip_bps),
        max_steps=8,                             # walk state overrides max_episode_steps
        window_size=60,
        position_size_pct=args.position_size_pct,
        reward_attributable=args.reward_attributable,
        vector_action=True,
        vector_weight_rebalance=args.vector_weight_rebalance,
        vector_weight_unit=args.vector_weight_unit,
        vector_weight_max=args.vector_weight_max,
        vector_rebalance_deadband=args.vector_rebalance_deadband,
    )
    prices_j = jnp.asarray(prices_np32, dtype=jnp.float32)
    prices_np = np.asarray(prices_np32, dtype=np.float64)
    logc = np.log(np.maximum(prices_np, 1e-12))
    inactive = [c for c in coverage if c["status"] == "inactive"]
    partial = [c for c in coverage if c["status"] == "partial"]
    # FIX PASS 2 — what goes INTO the artifact is the display-elided copy (a slot can carry
    # hundreds of multi-day frozen sub-gaps); what the CELL ACCOUNTING reads is ``coverage``, the
    # complete one. Those used to be the same truncated list, so ``n_frozen_decision_cells`` was
    # summed off at most 64 spans per slot while ``n_frozen_rows`` in the same file was the true
    # total. ``ev.frozen_decision_cells`` now REFUSES an elided list by name.
    cov_art = ev.coverage_for_artifact(coverage)
    inactive_art = [c for c in cov_art if c["status"] == "inactive"]
    partial_art = [c for c in cov_art if c["status"] == "partial"]
    inactive_slots = [int(c["slot"]) for c in inactive]
    # Fix pass FIX A(3): a PARTIAL slot is counted inside n_active_slots and is ABSENT from
    # ``inactive_slots``, so a consumer that reads only that list sees a slot which was dead for
    # most of the window as fully live. The partial slots are therefore ALSO discoverable at top
    # level (``partial_slots``), next to ``inactive_slots``, with the spans unchanged in meaning.
    partial_slots = [int(c["slot"]) for c in partial]
    n_active = int(args.n_assets) - len(inactive_slots)
    frame = {
        "frame_symbols": list(frame_symbols),
        "row_ts": np.asarray(row_ts, dtype=np.int64),
        "coverage": coverage,
        "inactive_slots": inactive_slots,
        "partial_slots": partial_slots,
        # 853-shaped disclosure block (mirrors eval_vector._build_panel_framed's panel_frame).
        "panel_frame": {
            "source": "checkpoint-provenance-fixed-frame (TASK_853 semantics, TASK_863 port)",
            "provenance_path": _display_path(args.panel_frame_from_checkpoint),
            # BOUNDED + type-checked (fix pass): copied verbatim, a provenance file could put a
            # NaN/Infinity (making the artifact non-strict JSON), a 555 KB blob, or a 2000-deep
            # nesting (uncaught RecursionError inside json.dump) into the read of record.
            "provenance_meta": ev.provenance_meta_block(
                prov, args.panel_frame_from_checkpoint),
            "panel_price_source": "ohlcv_daily via eval_vector.load_daily_prices_framed",
            "n_slots": int(args.n_assets),
            "n_active_slots": n_active,
            "n_inactive_slots": len(inactive_slots),
            "n_partial_slots": len(partial),
            "inactive_slots": list(inactive_slots),
            # FIX A(3): partial slots at the same altitude as the inactive ones (they are INSIDE
            # n_active_slots and absent from inactive_slots — do not read one without the other).
            "partial_slots": list(partial_slots),
            "inactive_spans": inactive_art,
            "partial_spans": partial_art,
            "coverage": cov_art,
            "force_hold_rule": "price_valid = price > 0 applied per (decision row, slot) to the "
                               "teacher target vectors, the injected one-hot BC labels and the "
                               "policy-argmax sanity replay — every walk here feeds step_vector "
                               "directly and so bypasses the env's compute_action_mask",
            # FIX C: the precise, MEASURED claim about what a dead slot shows the encoder. The
            # PANEL cell is flat-zero; the ENCODED features are CONSTANT across every decision but
            # NOT all-zero (the guarded 0/0 vol-ratio channels sit at clip(1.0,0,5)/5 = 0.2).
            # Constant => carries no signal => no leak; asserted by check_inactive_slot_features.
            "inactive_slot_feature_rule":
                "an inactive slot's PANEL cells are flat-zero; its ENCODED features are CONSTANT "
                "across all decisions but NOT all-zero — the guarded vol-ratio channels "
                f"{list(INACTIVE_SLOT_CONSTANT_CHANNELS)} (vol_ratio_5/15/30/60) carry a constant "
                "clip(1.0,0,5)/5 = 0.2 from the 0/0 volume ratio. Constant across decisions => no "
                "information, so no leak; ASSERTED per pinned run by "
                "check_inactive_slot_features (any decision-varying cell, or a non-zero cell on "
                "any other channel, is a REFUSAL)",
        },
    }
    # D30 amendment 4 — the stale-tail disclosure ships on BOTH pin paths (the daily loader has
    # always forward-filled a mid-window delisting to the window end at a frozen price and
    # disclosed it NOWHERE: measured on fixture frame slot 5 FTM/USD, 91 of 181 rows). Keys are
    # APPENDED, so every field the shipped 863 daily block carries keeps its value and position;
    # the daily NUMERICS do not move (the fill rule is unchanged and is now the sanctioned one).
    stale = [c for c in coverage if c.get("stale_tail")]
    frozen = [c for c in coverage if c.get("n_frozen_rows")]
    stale_art = [c for c in cov_art if c.get("stale_tail")]
    frozen_art = [c for c in cov_art if c.get("n_frozen_rows")]
    frame["panel_frame"].update({
        "n_stale_tail_slots": len(stale),
        "stale_tail_slots": [int(c["slot"]) for c in stale],
        "stale_tail_spans": stale_art,
        "stale_tail_rule": ev.STALE_TAIL_RULE,
        # FIX PASS — FROZEN SPANS, interior INCLUDED, on both pin paths. Staleness used to be
        # derived from the last real row alone, so an INTERIOR hole (a slot that RESUMES before the
        # window end) produced no disclosure whatsoever: no stale_tail, no partial, no inactive, and
        # covered_rows == n_rows. Measured 101 undisclosed frozen cells (25.25%) on the fixture
        # smoke grid and 2,013 (1.88%) on the 861 leg-1 window. Fill rule UNCHANGED — disclosure only.
        "n_frozen_slots": len(frozen),
        "frozen_slots": [int(c["slot"]) for c in frozen],
        # display-elided lists (the counts beside them are off the complete ones)
        "frozen_spans": {str(int(c["slot"])): c.get("frozen_spans") for c in frozen_art},
        "frozen_spans_display_max": ev.FROZEN_SPANS_ARTIFACT_MAX,
        "n_slots_with_elided_frozen_span_list": sum(
            1 for c in cov_art if c.get("frozen_spans_truncated")),
        # D30 rider (b) recorded on BOTH paths — "no floor applied" must never be silence.
        "require_active_slots": req_active,
    })
    # TASK_865: the minute source is recorded EXPLICITLY (scope 2 — no silent source switching).
    # Everything below is reached ONLY under --panel-frame-source minute; a daily-pinned block
    # never carries ``panel_frame_source`` (its ABSENCE means daily).
    if frame_source == "minute":
        frame["panel_frame"].update({
            "source": "checkpoint-provenance-fixed-frame (TASK_853 semantics, TASK_865 minute "
                      "source)",
            "panel_frame_source": "minute",
            "panel_price_source": (f"minute `ohlcv` via eval_vector.load_minute_prices_framed, "
                                   f"[::{int(args.bar_minutes)}] stride"),
            "bar_minutes": int(args.bar_minutes),
        })
    tag = "865" if frame_source == "minute" else "863"
    print("=" * 84)
    print(f"[{tag}] PINNED-FRAME {'MINUTE ' if frame_source == 'minute' else ''}panel from "
          f"checkpoint provenance "
          f"({_display_path(args.panel_frame_from_checkpoint)}): {args.n_assets} slots, "
          f"{n_active} ACTIVE / {len(inactive_slots)} INACTIVE (zero coverage) in "
          f"[{args.start}..{args.end}] — 853 semantics, slot k IS asset k")
    # D30 amendment 5: "n ACTIVE / 0 INACTIVE" counts only WHOLLY-inert slots, so it reads as
    # "fully live" over a grid that can be ~10% inert. State the partial/stale-tail counts here.
    print(f"[{tag}] ... of the {n_active} ACTIVE: {len(partial)} PARTIAL (listed mid-window: "
          f"pre-listing rows flat-zero => forced HOLD), {len(stale)} STALE-TAIL (delisted "
          f"mid-window: trailing rows FORWARD-FILLED at a frozen price, still tradeable) and "
          f"{len(frozen)} carrying FROZEN spans (trailing AND INTERIOR forward-filled rows). "
          f"'ACTIVE' means 'has at least one real close', NOT 'live all window'.")
    print(f"[{tag}] breadth floor in force: --require-active-slots {req_active} "
          f"({'DISABLED — a partly-inert panel is explicitly intended' if not req_active else 'D30 rider (b)'})")
    if inactive:
        print(f"[{tag}] INACTIVE slots ({len(inactive)}) — flat-zero PANEL cells, forced HOLD on "
              f"every in-env walk (no fills/fees); their ENCODED features are CONSTANT across "
              f"every decision but NOT all-zero (guarded vol-ratio channels "
              f"{list(INACTIVE_SLOT_CONSTANT_CHANNELS)} = 0.2; constant => no signal => no leak, "
              f"asserted below); every span reported:")
        for c in inactive:
            print(f"        slot {c['slot']:>3}  {c['symbol']:<16} INACTIVE "
                  f"[{c['span'][0]}..{c['span'][1]}]  ({c['reason']})")
    # A slot may be short at the HEAD (listed mid-window => PARTIAL, flat-zero force-HOLD rows) or
    # at the TAIL (delisted mid-window => STALE TAIL, forward-filled tradeable rows at a frozen
    # price). Reported separately: different keys, different causes, different treatment. On the
    # daily path partial IS partial-head, printed exactly as 863 printed it.
    partial_head = [c for c in partial if c.get("leading_inactive_rows")]
    if partial_head:
        print(f"[{tag}] PARTIAL slots ({len(partial_head)}) — listed mid-window: pre-listing rows "
              f"are flat-zero => forced HOLD there, active from the listing day onward:")
        for c in partial_head:
            print(f"        slot {c['slot']:>3}  {c['symbol']:<16} first covered ts "
                  f"{c['first_covered_ts']} ({c['leading_inactive_rows']} leading inactive rows)")
    if stale:
        print(f"[{tag}] STALE-TAIL slots ({len(stale)}) — the slot's LAST real close falls BEFORE "
              f"the window end (delisting / ticker migration). Trailing rows are FORWARD-FILLED at "
              f"that close (D30 opt (i): the only PV-exact treatment — zeroing them marks an open "
              f"position against price 0 and fabricates -/+25% of PV on a flat series), so they "
              f"stay TRADEABLE and weight-eligible at a FROZEN price. In NEITHER inactive_slots "
              f"NOR partial_slots — disclosed only here and in stale_tail_slots:")
        for c in stale:
            print(f"        slot {c['slot']:>3}  {c['symbol']:<16} last real close ts "
                  f"{c['last_real_ts']} = panel row {c['last_covered_row']} of {c['n_rows'] - 1}; "
                  f"{c['trailing_stale_rows']} trailing rows forward-filled at a frozen price "
                  f"({100.0 * c['trailing_stale_rows'] / max(c['n_rows'], 1):.1f}% of the "
                  f"window)")
    if frozen:
        print(f"[{tag}] FROZEN-SPAN slots ({len(frozen)}) — panel rows carrying a forward-filled "
              f"price the market never printed. A TRAILING span cannot fabricate PnL (constant to "
              f"the window end); an INTERIOR span CAN (it re-anchors, delivering the whole gap move "
              f"in one bar) and used to be invisible on every axis. Disclosed, NOT treated — what "
              f"to do about interior spans is a semantics ruling (DECISIONS D31):")
        for c in sorted(frozen, key=lambda r: -r["n_frozen_rows"])[:12]:
            n_int = c["n_frozen_rows"] - int(c.get("trailing_stale_rows", 0))
            print(f"        slot {c['slot']:>3}  {c['symbol']:<16} {c['n_frozen_rows']:>6} frozen "
                  f"panel rows of {c['n_rows']} ({n_int} INTERIOR + "
                  f"{int(c.get('trailing_stale_rows', 0))} trailing) across "
                  f"{c.get('n_frozen_spans', 0)} span(s)"
                  + ("  [NO REAL PANEL ROW AT ALL]" if c.get("no_real_panel_rows") else ""))
    if not inactive and not partial and not stale and not frozen:
        print(f"[{tag}] all slots fully covered on this window (no inactive/partial/frozen spans).")
    print("=" * 84)
    return prices_j, logc, env_cfg, frame


def frame_hold_mask(prices_j, start_row, n_decisions):
    """(N, A) bool: True where the panel has NO real price at that (decision row, slot) — the
    853 ``price_valid = current_prices > 0`` rule, evaluated on the decision rows this run walks
    (``walk_env`` reads ``prices[start_row + k]`` at decision k). ``None`` when nothing is
    inactive, so a fully-covered pinned window takes the unmasked code path."""
    dec = np.asarray(prices_j[start_row:start_row + n_decisions], dtype=np.float64)
    mask = dec <= 0.0
    return mask if mask.any() else None


def check_inactive_slot_features(obs, inactive_slots):
    """TASK_863 fix pass FIX C — GUARD (not a restatement) of what a dead slot shows the encoder.

    ``obs`` is (N, 27, A, A) from ``encode_training_obs``; axis 2 is the asset SLOT and axis 3 is
    the tiled width (jax_encoder._compute_per_asset_channels: ``output.at[ch].set(tile(x[:, -1:],
    (1, n)))`` over a (n_assets, n_assets) block), so slot s occupies ``obs[:, :, s, :]``.

    For every FULLY-INACTIVE slot (flat-zero panel column) this REFUSES unless
      (a) the slot's features are IDENTICAL across all N decisions, and
      (b) the only non-zero channels are ``INACTIVE_SLOT_CONSTANT_CHANNELS`` (the guarded 0/0
          vol-ratio channels at 0.2).
    Measured on the framed fixture panel: 896 of 6048 cells per inactive slot are non-zero, all on
    channels 8-11, all 0.2, identical for both dead slots and across every decision. A future
    encoder change that leaked real (decision-varying, or any other channel's) signal into a dead
    slot would fail here instead of silently training on it. PARTIAL slots are NOT checked: their
    pre-listing rows are flat-zero and their post-listing rows are real, so decision-varying
    features are CORRECT there (the price-validity mask, not this check, protects those rows).
    Returns a disclosure dict recorded in the artifact."""
    obs = np.asarray(obs)
    report = {
        "rule": "fully-inactive slot: features CONSTANT across decisions, non-zero ONLY on the "
                "guarded vol-ratio channels",
        "allowed_nonzero_channels": list(INACTIVE_SLOT_CONSTANT_CHANNELS),
        "n_decisions": int(obs.shape[0]), "obs_shape": list(obs.shape),
        "slot_axis": 2, "checked_slots": [], "passed": True,
    }
    for s in [int(x) for x in inactive_slots]:
        f = np.asarray(obs[:, :, s, :], dtype=np.float64)             # (N, C, A)
        f0 = f[0]
        if not np.array_equal(f, np.broadcast_to(f0, f.shape)):
            k = int(np.argmax(np.any(f != f0, axis=(1, 2))))
            ch = sorted({int(c) for c in np.nonzero(f[k] != f0)[0]})
            raise RuntimeError(
                f"--panel-frame-from-checkpoint FIX-C GUARD: INACTIVE slot {s} has encoded "
                f"features that VARY across decisions (first divergence at decision {k}, "
                f"channel(s) {ch}). A slot with a flat-zero panel column must encode to the same "
                f"constant every decision — a decision-varying feature there is real signal "
                f"leaking into a dead slot, which the force-HOLD mask does NOT remove from the "
                f"policy's input. Refusing to imprint on it.")
        nz = sorted({int(c) for c in np.nonzero(f0)[0]})
        leak = [c for c in nz if c not in INACTIVE_SLOT_CONSTANT_CHANNELS]
        if leak:
            raise RuntimeError(
                f"--panel-frame-from-checkpoint FIX-C GUARD: INACTIVE slot {s} encodes a non-zero "
                f"value on channel(s) {leak}, outside the guarded vol-ratio set "
                f"{list(INACTIVE_SLOT_CONSTANT_CHANNELS)} (values "
                f"{[float(f0[c].reshape(-1)[0]) for c in leak]}). A flat-zero panel column may "
                f"only produce the guarded 0/0 vol-ratio constant; anything else means the encoder "
                f"now derives something from a dead slot. Refusing to imprint on it.")
        report["checked_slots"].append({
            "slot": s,
            "n_feature_cells": int(f.size),
            "n_nonzero_feature_cells": int((f != 0).sum()),
            "nonzero_channels": nz,
            "nonzero_channel_values": {str(c): float(np.unique(f0[c])[0]) for c in nz
                                       if np.unique(f0[c]).size == 1},
            "constant_across_decisions": True,
        })
    return report


def _display_path(p):
    """Repo-relative when inside the repo, absolute otherwise (mirror of
    eval_vector._display_path: committed artifacts must not embed ephemeral tmp paths)."""
    p = os.path.abspath(p)
    rel = os.path.relpath(p, _REPO)
    return p if rel.startswith("..") else rel


# =============================================================================
# in-env walks (teacher replay / imprinted-policy sanity replay)
# =============================================================================

def walk_env(prices_j, cfg, start_row, n_decisions, action_fn, hold_mask=None):
    """Deterministic single-episode walk from panel row ``start_row`` through
    ``step_vector`` (the corridor/831 replay pattern, on the TRAINING panel/config).
    ``action_fn(k, state) -> (A,) int32`` (state = the pre-step TradingState, so a
    state-greedy mapping can read ``pos_side`` and, for the weight-native ladder,
    reconstruct the current signed weight). Returns per-decision returns + diag.

    TASK_863: ``hold_mask`` (N, A) bool, when given, forces HOLD wherever the panel has no real
    price at that (decision, slot) — the 853 price-validity rule, applied here because this walk
    hands its vector straight to ``step_vector`` and so bypasses the env's action mask. The
    RECORDED ``actions`` are the masked ones, so the injected BC labels are what was walked.
    Default ``None`` = the unpinned path, unchanged."""
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
    # Fix pass FIX A: under the pin, the reported avg_long/avg_short average over ALL slots,
    # including the force-HOLD ones that can never carry a position — so they are not comparable
    # to an unpinned (fully-covered) reference. Count an ACTIVE-CELLS-ONLY companion alongside.
    n_long_active = n_short_active = 0
    bankrupt_at = -1
    for k in range(n_decisions):
        act = np.asarray(action_fn(k, state), dtype=np.int32)
        if hold_mask is not None:
            act = np.where(hold_mask[k], HOLD_ACTION_CODE, act).astype(np.int32)
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
        if hold_mask is not None:
            live = ~hold_mask[k]
            n_long_active += int(((sd == 1) & live).sum())
            n_short_active += int(((sd == 2) & live).sum())
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
        # FIX A companions — None on the unpinned path (nothing is masked there).
        "avg_long_active": (None if hold_mask is None
                            else n_long_active / max(n_decisions, 1)),
        "avg_short_active": (None if hold_mask is None
                             else n_short_active / max(n_decisions, 1)),
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


def order_floor_reference(prices_j, env_cfg, teacher_fn, start_row, N, bar_minutes,
                          hold_mask=None):
    """The FIXED order-semantics floor: the canonical order-code teacher
    (``corridor._to_action_vec``, via ``teacher_fn``) walked at net@21 through the ORDER
    env (weight OFF). This is the 830/831 corridor floor and the SAME reference both the
    primary and the weight-native re-gate are scored against."""
    order_cfg = env_cfg.replace(fee_bps=FEE_BPS, slippage_bps=SLIP_BPS,
                                vector_weight_rebalance=False)
    wo = walk_env(prices_j, order_cfg, start_row, N, teacher_fn, hold_mask=hold_mask)
    return {"s": _sharpe(wo["rets"], bar_minutes),
            "final_pv": wo["final_pv"], "bankrupt_at": wo["bankrupt_at"]}


def expressibility_gate(prices_j, env_cfg, weight_teacher_fn, start_row, N, bar_minutes,
                        order_ref, mapping, tol=0.10, hold_mask=None):
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
    ww = walk_env(prices_j, weight_cfg, start_row, N, weight_teacher_fn, hold_mask=hold_mask)
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


def gate_teacher_vectors(doc, path, args, tier_cfg, start_row, N, T, frame=None):
    """FAIL-LOUD provenance gate (task scope 2). The collected target stream may only be
    imprinted when the imprint run reproduces the EXACT panel and env the walk was recorded on:
    db, window, bar/universe frame, decision rows (index AND timestamp), and the whole EnvConfig
    fingerprint minus the documented ``TV_ENV_EXEMPT`` fields. Every mismatch is collected and
    reported together, then the run ABORTS — a silently-misaligned teacher is a wrong-experiment
    result, not a crash. Returns the provenance summary for the artifact.

    Fix pass: the gate ALSO pins the teacher OPERATOR and CHECKPOINT (FIX 1 —
    ``TV_OPERATOR_PIN`` / ``TV_CHECKPOINT_PIN``; TASK_861 line 35 "any deviation is a different
    experiment and must fail loud") and validates the payload's integrality (FIX 2). Only
    ``--allow-off-pin-teacher`` downgrades the pin to a loud, RECORDED deviation.

    TASK_863 scope 2: the panel-frame check is FAIL-CLOSED IN BOTH DIRECTIONS. A frame-pinned
    collection may only be imprinted by an imprint pinned to the IDENTICAL frame (symbol list AND
    order — the provenance path may differ); a pinned collection with an unpinned imprint, or a
    pinned imprint with an unpinned collection, is REFUSED with both directions named.
    Unpinned-on-both-sides stays allowed: that is the legacy path, unchanged."""
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

    # --- panel FRAME (TASK_863 scope 2): FAIL-CLOSED IN BOTH DIRECTIONS -----------------------
    # Before 863 this was a one-directional refusal ("the 853 fixed-frame panel has no counterpart
    # here"), which is no longer true: --panel-frame-from-checkpoint gives it one. The dangerous
    # case the old text could not express is the REVERSE mix (a pinned imprint on an unpinned
    # collection), and the case it must now police is a pinned/pinned pair whose frames DIFFER —
    # including a same-membership REORDER, which is invisible to any set comparison but misaligns
    # every slot (slot k IS asset k).
    coll_frame_path = p.get("panel_frame_from_checkpoint")
    # FIX D (fix pass): a TRUTHY NON-DICT panel_frame (valid JSON, wrong type — e.g. the string
    # "fixed-frame") used to reach ``.get("coverage")`` below and raise a bare AttributeError. It
    # was fail-closed in effect, but house style (eval_vector.check_checkpoint_arch handles this
    # exact valid-JSON/wrong-type case) is a NAMED refusal. Guard + name it; still fail-closed.
    coll_frame_raw = p.get("panel_frame")
    coll_frame_block = coll_frame_raw if isinstance(coll_frame_raw, dict) else {}
    # FIX 2.4 (round 2): a PRESENT but EMPTY panel_frame object was the one malformed value that
    # slipped through BOTH guards — it IS a dict (so the FIX D type check below skips it) and it is
    # FALSY (so ``coll_pinned`` read False), so a TRUNCATED 853 disclosure block silently
    # DOWNGRADED to the legacy/unpinned path instead of refusing. Every other malformed value ([],
    # false, 0, 7, "fixed-frame", true) already draws the named panel_frame TYPE refusal. What
    # makes this safe is the ABSENT/EMPTY distinction: eval_vector records ``panel_frame: null`` on
    # every UNPINNED collection (the legitimate legacy header — see
    # 861_search_teacher_imprint/chosen_vectors_fixture.json), so ONLY a present, EMPTY OBJECT is
    # treated as corrupt here; null/absent keeps the legacy path byte-for-byte.
    coll_frame_empty = isinstance(coll_frame_raw, dict) and not coll_frame_raw
    coll_pinned = bool(coll_frame_path or coll_frame_raw or coll_frame_empty)
    imp_pinned = frame is not None
    # TASK_865: WHICH TABLE the pin was read from. A daily-pinned collection imprinted under a
    # minute pin (or vice versa) is the exact SOURCE confound D20 ruled forbidden — same frame,
    # different price series — and it is invisible to every symbol-identity check below. Absent
    # ``panel_frame_source`` == the shipped 853 DAILY source (that block never carried the key).
    coll_frame_src = (coll_frame_block.get("panel_frame_source", "daily")
                      if coll_pinned else None)
    imp_frame_src = getattr(args, "panel_frame_source", "daily") if imp_pinned else None
    minute_involved = "minute" in (coll_frame_src, imp_frame_src)
    frame_gate = {
        "collection_pinned": bool(coll_pinned),
        "imprint_pinned": bool(imp_pinned),
        "collection_provenance_path": coll_frame_path,
        "imprint_provenance_path": (_display_path(args.panel_frame_from_checkpoint)
                                    if imp_pinned else None),
        "identity_matched": None,
        "identity_source": None,
        # FIX B: what the collection RECORDED as its panel source vs what this run's panel
        # actually IS. Under the pin the recorded flag is the --panel-source DEFAULT string
        # ('minute'), which eval_vector.build_panel:1827 OVERRIDES by dispatching to the framed
        # (daily) loader before the panel_source branch is even read — so the recorded value
        # verifies nothing under a frame pin and cannot be compared to 'minute' as if it did.
        "collection_panel_source_recorded": p.get("panel_source"),
        "imprint_panel_source_actual": (
            ("minute-framed" if imp_frame_src == "minute" else "daily-framed")
            if imp_pinned else "minute"),
    }
    # Additive and minute-only: a daily/daily pair records exactly the keys 863 shipped.
    if minute_involved:
        frame_gate["collection_frame_source"] = coll_frame_src
        frame_gate["imprint_frame_source"] = imp_frame_src
    if coll_pinned and imp_pinned and coll_frame_src != imp_frame_src:
        mism.append(
            f"    panel_frame SOURCE MISMATCH: collection frame read from "
            f"{coll_frame_src!r}, imprint frame read from {imp_frame_src!r} -> the two runs pin "
            f"the SAME symbols but DIFFERENT price series (`ohlcv_daily` vs the minute `ohlcv` "
            f"table). Cross-source comparison is RULED forbidden (D20: |delta|=0.0555 even "
            f"coverage-symmetric), and the two tables also differ in per-slot coverage on the "
            f"same window (33/64 vs 62/64 on the 861 leg-1 frame), so the target vectors would "
            f"be replayed against prices the collection never saw. Pass the SAME "
            f"--panel-frame-source on both legs")
    if coll_frame_raw is not None and not isinstance(coll_frame_raw, dict):
        mism.append(
            f"    panel_frame TYPE: the collection header's provenance.panel_frame is a "
            f"{type(coll_frame_raw).__name__}, not a JSON object "
            f"({str(coll_frame_raw)[:80]!r}) -> the 853 frame-disclosure block is MALFORMED, so "
            f"the frame it walked cannot be read at all (per-slot symbols, coverage, spans). "
            f"Re-collect with the current eval_vector.py; a malformed frame record is not "
            f"imprintable (the same rule as an absent one)")
        frame_gate["collection_panel_frame_type"] = type(coll_frame_raw).__name__
    elif coll_frame_empty:
        # FIX 2.4 (round 2) — the named refusal for the truncated block.
        mism.append(
            "    panel_frame EMPTY: the collection header carries provenance.panel_frame as an "
            "EMPTY JSON object -> the 853 frame-disclosure block is TRUNCATED (no per-slot "
            "symbols, no coverage, no spans), so the frame it walked cannot be read at all. An "
            "ABSENT panel_frame key — or the `null` one every UNPINNED collection records — is the "
            "legitimate legacy header and still takes the legacy path; a PRESENT but empty object "
            "is a corrupt/truncated record and is not imprintable (the same rule as a malformed or "
            "an absent frame). Re-collect with the current eval_vector.py")
        frame_gate["collection_panel_frame_empty"] = True
    # --- panel SOURCE: which loader built the panel this run walks ---
    if imp_pinned:
        # Under --panel-frame-from-checkpoint this run's panel is NOT minute-sourced: it is the
        # DAILY framed panel. A frame-pinned collection records either 'minute' (the --panel-source
        # default, left untouched because the frame path overrides it) or 'daily' — the framed
        # loader is dispatched BEFORE that branch, so the panel is bit-identical either way and
        # refusing 'daily' would reject a perfectly frame-matched pair.
        if p.get("panel_source") not in ("minute", "daily"):
            if imp_frame_src == "minute":
                mism.append(
                    f"    panel_source: collection={p.get('panel_source')!r}  "
                    f"imprint='minute-framed'  -> under --panel-frame-from-checkpoint "
                    f"--panel-frame-source minute BOTH runs build the MINUTE framed panel (the "
                    f"`ohlcv` table via eval_vector.load_minute_prices_framed, [::bar_minutes]), "
                    f"and a frame-pinned collection records 'minute' (the --panel-source default "
                    f"the frame path overrides) or 'daily' here — this value is neither, so the "
                    f"collection's panel loader is unknown")
            else:
                # Byte-unchanged from shipped 863 (only the enclosing `else:` is new).
                mism.append(
                    f"    panel_source: collection={p.get('panel_source')!r}  "
                    f"imprint='daily-framed'  -> under --panel-frame-from-checkpoint BOTH runs build "
                    f"the DAILY framed panel (ohlcv_daily via eval_vector.load_daily_prices_framed, "
                    f"--bar-minutes 1440), and a frame-pinned collection records 'minute' (the "
                    f"--panel-source default the frame path overrides) or 'daily' here — this value "
                    f"is neither, so the collection's panel loader is unknown")
    elif p.get("panel_source") != "minute":
        # Unpinned/legacy path — message and behavior byte-identical to shipped 861.
        mism.append(f"    panel_source: collection={p.get('panel_source')!r}  imprint='minute'"
                    f"  -> this runner builds its panel through create_jax_env_historical (the "
                    f"1-min ohlcv table, PIN b); it cannot reproduce a daily/framed panel")
    if coll_pinned and not imp_pinned:
        mism.append(
            f"    panel_frame PIN MISMATCH (pinned collection, UNPINNED imprint): "
            f"collection=FIXED-FRAME ({coll_frame_path!r})  imprint=coverage-ranked  -> the "
            f"collection walked a fixed positional frame while this imprint would derive its "
            f"universe from ORDER BY COUNT(*) DESC LIMIT {args.n_assets} over the window; slot k "
            f"IS asset k, so the recorded target vectors would be applied to DIFFERENT assets. "
            f"Pass --panel-frame-from-checkpoint <the collection's frame> (TASK_863 scope 2: "
            f"BOTH legs pin the frame, or NEITHER does)")
    elif imp_pinned and not coll_pinned:
        mism.append(
            f"    panel_frame PIN MISMATCH (PINNED imprint, unpinned collection): "
            f"collection=coverage-ranked  imprint=FIXED-FRAME "
            f"({_display_path(args.panel_frame_from_checkpoint)})  -> this imprint walks a fixed "
            f"positional frame the collection never used, so the recorded vectors belong to the "
            f"collection's coverage-ranked slots, not to these. Re-collect with `eval_vector.py "
            f"--panel-frame-from-checkpoint` on the SAME frame, or drop the flag here (TASK_863 "
            f"scope 2: BOTH legs pin the frame, or NEITHER does)")
    elif coll_pinned and imp_pinned:
        # Frame IDENTITY = the ordered symbol list. Prefer the per-slot record (853 writes one
        # coverage entry per slot); fall back to loaded_symbols, which the framed loader fills
        # with the full frame in order. If NEITHER is present the frame is unverifiable -> REFUSE
        # (a claim we cannot check is worse than a refusal — the FIX 6 lesson).
        want_frame = list(frame["frame_symbols"])
        cov = coll_frame_block.get("coverage")
        got_frame = src_id = cov_bad = None
        # FIX 2.2 / FIX 2.3 (round 2): the old predicate checked only that the "slot"/"symbol" KEYS
        # existed, never their TYPES — so ``int(c["slot"])`` raised a raw ValueError on "abc" and a
        # raw TypeError on null (the same valid-JSON/wrong-TYPE class FIX D exists to name; this
        # deliverable's contract is "no raw tracebacks") — and it accepted slot multisets that
        # CANNOT be a per-slot map (e.g. [0,0,2,3,4,5,6,7]: two records for slot 0, none for slot 1)
        # which sort stably into a plausible-looking symbol sequence that was then recorded as
        # ``identity_matched: true`` positional proof. Validate the types AND that the slots are
        # exactly 0..n-1. A coverage record that is PRESENT but malformed is REFUSED (it does NOT
        # silently fall back to loaded_symbols: a corrupt header cannot vouch for itself); an
        # ABSENT coverage keeps the pre-existing loaded_symbols fallback.
        if cov is not None:
            if not isinstance(cov, list) or len(cov) != len(want_frame):
                cov_bad = (f"expected a JSON list of {len(want_frame)} per-slot records, got a "
                           f"{type(cov).__name__}"
                           + (f" of length {len(cov)}" if isinstance(cov, list) else ""))
            else:
                bad_i = next((i for i, c in enumerate(cov) if not (
                    isinstance(c, dict) and isinstance(c.get("symbol"), str)
                    and isinstance(c.get("slot"), int)
                    and not isinstance(c.get("slot"), bool))), None)
                slots = None if bad_i is not None else sorted(int(c["slot"]) for c in cov)
                if bad_i is not None:
                    cov_bad = (f"record {bad_i} is not an object carrying an INT 'slot' and a STR "
                               f"'symbol' ({str(cov[bad_i])[:100]!r})")
                elif slots != list(range(len(want_frame))):
                    cov_bad = (f"the 'slot' values {slots} are not a complete per-slot map of "
                               f"0..{len(want_frame) - 1} (duplicated and/or missing slots), so "
                               f"the record cannot establish POSITIONAL identity at all")
                else:
                    got_frame = [c["symbol"] for c in sorted(cov, key=lambda c: int(c["slot"]))]
                    src_id = "provenance.panel_frame.coverage (per-slot symbol record)"
        if got_frame is None and cov_bad is None and isinstance(p.get("loaded_symbols"), list):
            got_frame = list(p["loaded_symbols"])
            src_id = "provenance.loaded_symbols"
        frame_gate["identity_source"] = src_id
        if cov_bad is not None:
            frame_gate["collection_coverage_malformed"] = cov_bad
            mism.append(
                f"    panel_frame.coverage MALFORMED: {cov_bad}. That per-slot record is the "
                f"collection's positional proof (slot k IS asset k); a record that cannot be read "
                f"as one leaves the frame it walked UNVERIFIABLE, and provenance.loaded_symbols is "
                f"NOT accepted in its place (a corrupt frame header cannot vouch for itself). "
                f"Re-collect with the current eval_vector.py; an unverifiable frame is not "
                f"imprintable")
        elif got_frame is None:
            mism.append(
                f"    panel_frame identity: the collection header records a frame pin "
                f"({coll_frame_path!r}) but carries no per-slot symbol record "
                f"(panel_frame.coverage / loaded_symbols), so the frame it walked cannot be "
                f"compared against this run's frame. Re-collect with the current eval_vector.py; "
                f"an unverifiable frame is not imprintable")
        elif got_frame != want_frame:
            k = next((i for i in range(min(len(want_frame), len(got_frame)))
                      if want_frame[i] != got_frame[i]), min(len(want_frame), len(got_frame)))
            same_membership = sorted(got_frame) == sorted(want_frame)
            mism.append(
                f"    panel_frame identity MISMATCH at slot {k}: collection="
                f"{got_frame[k] if k < len(got_frame) else None!r}  imprint="
                f"{want_frame[k] if k < len(want_frame) else None!r} (source: {src_id})\n"
                f"      collection frame {got_frame}\n      imprint    frame {want_frame}\n"
                f"      -> the two runs pin DIFFERENT frames. "
                + ("The MEMBERSHIP is identical and only the ORDER differs — that is exactly the "
                   "failure this gate exists to catch: slot k IS asset k positionally (the "
                   "encoder kernel is 27 feats x n_assets slots), so a reordered frame silently "
                   "applies every target vector to the wrong asset while every set-based check "
                   "passes. Membership-only matching is NOT accepted."
                   if same_membership else
                   "The frames differ in membership as well as position.")
                + " Point both legs at the same frame provenance (the PATH may differ; the "
                  "ordered symbol list may not)")
        else:
            frame_gate["identity_matched"] = True
    # Row-timestamp axis + universe: under the pin they come from the framed panel this run
    # actually built (the minute loader mirror would reconstruct the coverage-ranked universe and
    # the minute row axis — neither of which this run walks).
    if imp_pinned:
        row_ts = frame["row_ts"]
        loaded_syms = list(frame["frame_symbols"])
    else:
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
    if imp_pinned or coll_pinned:
        print(f"[863] frame pin gate PASS: collection PINNED={coll_pinned} imprint "
              f"PINNED={imp_pinned}; frame identity (symbol list AND order) matched on "
              f"{args.n_assets} slots via {frame_gate['identity_source']}")
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
            # TASK_863 scope 2: the both-directions frame-pin verdict. ADDITIVE and appended LAST
            # so it is ABSENT (and every key above is byte-order-unchanged) when neither side is
            # pinned — the legacy unpinned-both artifact stays bit-identical to shipped 861.
            **({"frame_pin": frame_gate} if (imp_pinned or coll_pinned) else {}),
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
    # TASK_863: --panel-frame-from-checkpoint swaps in the 853 fixed-frame panel (flag absent =>
    # the PIN-b minute panel, untouched). ``frame`` stays None on the unpinned path and every
    # frame-derived object below (hold_mask, the artifact block) is then None/absent.
    frame = None
    if args.panel_frame_from_checkpoint is not None:
        prices_j, logc, env_cfg, frame = load_framed_panel(args)
    else:
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
    # TASK_863: the 853 price-validity rule on THIS run's decision rows. None (and every walk
    # below takes the unmasked path) when unpinned or when no slot is inactive.
    hold_mask = None if frame is None else frame_hold_mask(prices_j, start_row, N)
    if frame is not None:
        n_hold_cells = 0 if hold_mask is None else int(hold_mask.sum())
        print(f"[863] force-HOLD cells (no real price at that decision row / slot): "
              f"{n_hold_cells} of {N * A} ({100.0 * n_hold_cells / max(N * A, 1):.2f}%) — "
              f"applied to the teacher targets, the injected BC labels and the sanity replay")
        frame["panel_frame"]["n_force_hold_cells"] = n_hold_cells
        frame["panel_frame"]["n_decision_cells"] = int(N * A)
        # Fix pass FIX A: the dilution FRACTION, recorded next to the counts. Every reported
        # per-cell scalar below (teacher_action_agreement, policy_action_hist, avg_long/short, the
        # 861 fit block) averages over ALL cells including these dead ones, so it is NOT comparable
        # to an unpinned reference — hence the active-cells-only companions and this fraction.
        frame["panel_frame"]["masked_cell_fraction"] = (
            float(n_hold_cells) / float(max(N * A, 1)))

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
                                       tier_cfg[GATE_COSTS[-1]], start_row, N, T, frame=frame)
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
        teacher = {c: walk_env(prices_j, tier_cfg[c], start_row, N, teacher_fn,
                               hold_mask=hold_mask)
                   for c in GATE_COSTS}
        floor = {c: _sharpe(teacher[c]["rets"], args.bar_minutes) for c in GATE_COSTS}
        print(f"[832] teacher floor in-env: gross {floor[0]:+.3f}  net@21 {floor[21]:+.3f}")

        wn_fn = _make_weight_native_teacher_fn(prices_j, env_cfg, tgt, start_row, A)
        order_ref = order_floor_reference(prices_j, env_cfg, teacher_fn, start_row, N,
                                          args.bar_minutes, hold_mask=hold_mask)
        gate = expressibility_gate(prices_j, env_cfg, teacher_fn, start_row, N, args.bar_minutes,
                                   order_ref, "order-code (corridor._to_action_vec)",
                                   hold_mask=hold_mask)
        _gate_line(gate)
        if not gate["passed"]:
            # FAIL => re-gate the weight-native ladder mapping against the SAME order-code floor
            # reference (pre-registered contingency; the same wn_fn can be IMPRINTED, below).
            wn_gate = expressibility_gate(
                prices_j, env_cfg, wn_fn, start_row, N, args.bar_minutes, order_ref,
                "weight-native ladder (sign(w_target - w_cur) | |dw|>deadband)",
                hold_mask=hold_mask)
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
        teacher = {c: walk_env(prices_j, tier_cfg[c], start_row, N, tv_fn, hold_mask=hold_mask)
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
        teacher = {c: walk_env(prices_j, tier_cfg[c], start_row, N, wn_fn, hold_mask=hold_mask)
                   for c in GATE_COSTS}
        floor = {c: _sharpe(teacher[c]["rets"], args.bar_minutes) for c in GATE_COSTS}
        print(f"[832] IMPRINT MAPPING = weight-native ladder; teacher floor in-env: "
              f"gross {floor[0]:+.3f}  net@21 {floor[21]:+.3f}")
    else:
        print(f"[832] IMPRINT MAPPING = order-code (default; byte-identical default path)")

    # ---- training observations for the SAME decision rows (PIN b) ----
    obs = encode_training_obs(prices_j, start_row, N, env_cfg)
    # Fix pass FIX C: GUARD the dead-slot feature claim on the obs actually injected (pinned runs
    # with at least one fully-inactive slot). Fails loud — see check_inactive_slot_features.
    if frame is not None and frame["inactive_slots"]:
        frame["panel_frame"]["inactive_slot_feature_check"] = check_inactive_slot_features(
            obs, frame["inactive_slots"])
        chk_ = frame["panel_frame"]["inactive_slot_feature_check"]
        for s_ in chk_["checked_slots"]:
            print(f"[863] inactive-slot feature GUARD slot {s_['slot']:>3}: constant across all "
                  f"{chk_['n_decisions']} decisions; {s_['n_nonzero_feature_cells']} of "
                  f"{s_['n_feature_cells']} feature cells non-zero, channels "
                  f"{s_['nonzero_channels']} (guarded vol-ratio) = "
                  f"{sorted(set(s_['nonzero_channel_values'].values()))} — flat-zero PANEL cells, "
                  f"CONSTANT (not all-zero) encoded features, so no signal and no leak")

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
    # TASK_863: the policy-argmax replay bypasses the env action mask, so apply the SAME
    # price-validity rule 853 applies in eval_vector.evaluate_checkpoint. Forcing HOLD on a
    # zero-price cell is always correct (no tradeable market there) and can never HOLD a
    # genuinely-active decision, so it is alignment-safe. Unpinned path: hold_mask is None.
    if hold_mask is not None:
        pol_actions = np.where(hold_mask, HOLD_ACTION_CODE, pol_actions).astype(np.int32)
    agreement = float(np.mean(pol_actions == teacher[net_tier]["actions"]))
    # TASK_861 acceptance (iv): the train-set fit measured against the NEVER-SHUFFLED targets.
    tv_fit = None
    if tv_actions_original is not None:
        tv_logp = policy_action_log_marginals(model, trainer.state.params, obs, A)
        tv_fit = teacher_vectors_fit(tv_logp, tv_actions_original)
        # Fix pass FIX A: every scalar in this block (NLL, marginal baseline, argmax agreement,
        # modal_action_rate — and therefore argmax_excess_over_modal, whose floor is SHIFTED by the
        # forced cells) is computed over ALL N x A label cells, including the force-HOLD ones. Add
        # the ACTIVE-CELLS-ONLY companion (same estimator, cells restricted to ~hold_mask) without
        # touching the existing keys. Pinned-only: absent on the legacy path.
        if hold_mask is not None:
            act_ = ~hold_mask
            if int(act_.sum()):
                tv_fit["active_cells_only"] = teacher_vectors_fit(
                    tv_logp[act_][:, None, :], tv_actions_original[act_][:, None])
            else:
                tv_fit["active_cells_only"] = None
            tv_fit["masked_cell_fraction"] = float(hold_mask.mean())
            tv_fit["active_cells_note"] = (
                "the keys above average over ALL label cells, INCLUDING the force-HOLD cells of "
                "inactive/pre-listing slots (see masked_cell_fraction). 'active_cells_only' is the "
                "SAME estimator on the ~hold_mask cells and is the figure to compare across "
                "frames; neither is comparable to an unpinned, fully-covered reference")
        print(f"[861] teacher-vector fit vs the NEVER-SHUFFLED targets "
              f"(n={tv_fit['n_label_cells']} cells): NLL {tv_fit['nll_vs_original_targets']:.4f} "
              f"vs marginal baseline {tv_fit['nll_marginal_baseline']:.4f} -> reduction "
              f"{tv_fit['nll_reduction']:+.4f} (floor {tv_fit['nll_reduction_floor']:+.2f}); "
              f"argmax agreement {tv_fit['argmax_agreement_vs_original']:.3f} vs modal rate "
              f"{tv_fit['modal_action_rate']:.3f}")
        if tv_fit.get("active_cells_only"):
            a_ = tv_fit["active_cells_only"]
            print(f"[863] the line above is DILUTED by the force-HOLD cells "
                  f"({100.0 * tv_fit['masked_cell_fraction']:.2f}% of cells). ACTIVE-CELLS-ONLY "
                  f"companion (n={a_['n_label_cells']}): NLL "
                  f"{a_['nll_vs_original_targets']:.4f} vs marginal baseline "
                  f"{a_['nll_marginal_baseline']:.4f} -> reduction {a_['nll_reduction']:+.4f}; "
                  f"argmax agreement {a_['argmax_agreement_vs_original']:.3f} vs modal rate "
                  f"{a_['modal_action_rate']:.3f} (the modal floor above is SHIFTED by the forced "
                  f"HOLDs, so argmax_excess_over_modal is measured against a moved floor)")
    sanity = {}
    for c in GATE_COSTS:
        rep = walk_env(prices_j, tier_cfg[c], start_row, N,
                       lambda k, state: pol_actions[k], hold_mask=hold_mask)
        s = _sharpe(rep["rets"], args.bar_minutes)
        sanity[c] = {
            "policy_inenv": float(s) if np.isfinite(s) else None,
            "teacher_floor_inenv": float(floor[c]) if np.isfinite(floor[c]) else None,
            "corridor_position": (float(s - floor[c])
                                  if np.isfinite(s) and np.isfinite(floor[c]) else None),
            "final_pv": rep["final_pv"], "avg_long": rep["avg_long"],
            "avg_short": rep["avg_short"], "bankrupt_at": rep["bankrupt_at"],
        }
        # Fix pass FIX A — APPENDED (so the legacy dict is byte-identical): avg_long/avg_short
        # average open positions over ALL slots, dead ones included. Active-cells-only companions.
        if hold_mask is not None:
            sanity[c]["avg_long_active_cells"] = rep["avg_long_active"]
            sanity[c]["avg_short_active_cells"] = rep["avg_short_active"]

    hist = lambda a: {str(i): int((a == i).sum()) for i in range(N_ACTIONS_PER_ASSET)}
    pol_hist = hist(pol_actions)
    # ============================ fix pass FIX 0 + FIX A (PINNED RUNS ONLY) ======================
    # FIX 0 (MAJOR): a degenerate PINNED run used to exit 0 and write a green-looking artifact. An
    # under-trained pinned imprint can collapse the policy to total HOLD, which makes every replay
    # return a zero-variance return series => Sharpe non-finite => policy_inenv/corridor_position
    # null, sitting next to all_losses_finite:true and policy_ce_decreased:true with NO failure
    # marker of any kind. With --smoke-assert OFF (the default for ANY db that is not byte-equal to
    # the repo fixture path — i.e. INCLUDING the scheduled real leg) the run exited 0 and nothing
    # downstream could tell it from a healthy one. Under the pin (and ONLY under the pin — the
    # unpinned/legacy behavior is PRE-EXISTING and deliberately untouched) a non-finite/None sanity
    # Sharpe on EITHER cost tier is now a fail-loud condition INDEPENDENT of --smoke-assert: the
    # verdict is decided HERE, before anything is written, the artifact carries an explicit marker,
    # and the run exits non-zero (see the refusal after the write below).
    # FIX 2.1 (round 2): the verdict examines EVERY Sharpe-valued sanity.per_tier field, not just
    # policy_inenv. corridor_position IS the ledger metric (ledger_row.
    # imprint_corridor_position_net21) and teacher_floor_inenv is the bar it is measured against, so
    # a null in either is just as degenerate — and a run that recorded degenerate_replay:FALSE with
    # a null corridor_position was making a POSITIVE health assertion about a degenerate run, which
    # is worse than carrying no marker at all. The TEACHER-side null gets its OWN diagnosis: it
    # means the teacher never opened a position, which no amount of extra --iters can fix.
    sanity_extra = {}
    degenerate_reason = None
    if frame is not None:
        sharpe_fields = ("policy_inenv", "teacher_floor_inenv", "corridor_position")
        null_by_tier = {c: [f for f in sharpe_fields if sanity[c][f] is None] for c in GATE_COSTS}
        degenerate = [c for c in GATE_COSTS if null_by_tier[c]]
        collapsed = [c for c in GATE_COSTS if "policy_inenv" in null_by_tier[c]]
        teacher_flat = [c for c in GATE_COSTS if "teacher_floor_inenv" in null_by_tier[c]]

        def tier_names(cs):
            return ", ".join(("gross" if c == 0 else f"net@{c}") for c in cs)

        n_cells = int(N * A)
        n_masked = 0 if hold_mask is None else int(hold_mask.sum())
        if degenerate:
            parts = [
                "PINNED-FRAME run produced a DEGENERATE sanity replay: a null (non-finite) "
                "Sharpe-valued sanity.per_tier field on cost tier(s) " + tier_names(degenerate)
                + " — " + "; ".join(f"{tier_names([c])}: " + ", ".join(null_by_tier[c]) + " null"
                                    for c in degenerate)
                + ". corridor_position is the LEDGER metric "
                  "(ledger_row.imprint_corridor_position_net21), so a null there is not a cosmetic "
                  "gap: there is no result to record."]
            if collapsed:
                parts.append(
                    "STUDENT SIDE (policy_inenv null on " + tier_names(collapsed) + f"): the "
                    f"policy-argmax action histogram is {pol_hist} out of {n_cells} decision cells "
                    f"({N_ACTIONS_PER_ASSET}-code space; {HOLD_ACTION_CODE}=HOLD)"
                    + (" — a total-HOLD policy has a zero-variance return series, so the Sharpe "
                       "is undefined, NOT zero"
                       if pol_hist[str(HOLD_ACTION_CODE)] == n_cells else "")
                    + f". {n_masked} of {n_cells} cells were force-HOLD by the frame mask, so the "
                    f"pinned panel reaches this collapse on FEWER free decisions than an unpinned "
                    f"one; under-training (too few --iters) is a known cause, but the collapse "
                    f"BAND is CONFIGURATION-dependent (teacher stream, frame, seed, --iters, "
                    f"deadband) and has to be MEASURED for the configuration at hand — there is "
                    f"NO universal --iters boundary (measured counter-example on this fixture "
                    f"frame: the "
                    f"default-teacher pinned path at --iters 20, seed 0, weight-native is already "
                    f"HEALTHY, corridor_position net@21 +1.84).")
            if teacher_flat:
                parts.append(
                    "TEACHER SIDE (teacher_floor_inenv null on " + tier_names(teacher_flat)
                    + f"): the TEACHER's OWN replay has no finite Sharpe — its action histogram "
                    f"is {hist(teacher[net_tier]['actions'])} out of {n_cells} cells, i.e. the "
                    f"teacher never opened a position on these rows, so the corridor has NO floor "
                    f"to measure the student against. This is a DIFFERENT diagnosis from a "
                    f"collapsed student and MORE --iters cannot fix it: the teacher/env "
                    f"configuration (e.g. a "
                    f"--vector-rebalance-deadband so wide that every target is HOLD), the window, "
                    f"or the frame is at fault.")
            parts.append("This artifact is NOT a result.")
            degenerate_reason = " ".join(parts)
        sanity_extra["degenerate_replay"] = bool(degenerate)
        sanity_extra["degenerate_replay_reason"] = degenerate_reason
        sanity_extra["n_decision_cells"] = n_cells
        sanity_extra["n_masked_cells"] = n_masked
        sanity_extra["masked_cell_fraction"] = float(n_masked) / float(max(n_cells, 1))
        # FIX A: active-cells-only companions for every diluted per-cell scalar above. The existing
        # keys keep their (diluted) values; these are the frame-comparable ones.
        if hold_mask is not None:
            act_cells = ~hold_mask
            n_act = int(act_cells.sum())
            sanity_extra["active_cells_only"] = {
                "n_active_cells": n_act,
                "teacher_action_agreement": (
                    float(np.mean(pol_actions[act_cells]
                                  == teacher[net_tier]["actions"][act_cells])) if n_act else None),
                "policy_action_hist": hist(pol_actions[act_cells]) if n_act else None,
                "per_tier_avg_long": {str(c): sanity[c]["avg_long_active_cells"]
                                      for c in GATE_COSTS},
                "per_tier_avg_short": {str(c): sanity[c]["avg_short_active_cells"]
                                       for c in GATE_COSTS},
            }
            sanity_extra["dilution_note"] = (
                f"teacher_action_agreement, policy_action_hist and avg_long/avg_short above are "
                f"computed over ALL {n_cells} decision cells, {n_masked} of which "
                f"({100.0 * n_masked / max(n_cells, 1):.2f}%) are force-HOLD by the frame mask on "
                f"both sides — so they are INFLATED/deflated by cells no policy could influence "
                f"and are NOT comparable to an unpinned reference (e.g. the +0.839 run's 97.5% "
                f"agreement was measured on a fully-covered minute panel). Use "
                f"'active_cells_only'.")
        else:
            sanity_extra["active_cells_only"] = None
            sanity_extra["dilution_note"] = (
                "no cell is force-HOLD on this pinned window (every slot fully covered), so the "
                "reported figures are already active-cells-only")
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
            "policy_action_hist": pol_hist,
            # TASK_863 fix pass — pinned-only, APPENDED LAST so the legacy block is byte-identical
            # (empty dict on the unpinned path): the FIX 0 degeneracy verdict + the FIX A
            # masked-cell disclosure and active-cells-only companions.
            **sanity_extra,
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

    # TASK_863: flag-gated ADDITIVE disclosure block — absent (and every key above byte-identical)
    # when --panel-frame-from-checkpoint is not passed. Present, a 6/8-live-slot imprint is
    # impossible to mistake for an 8/8-live one (the 853 dual-disclosure requirement).
    if frame is not None:
        result["panel_frame_from_checkpoint"] = _display_path(args.panel_frame_from_checkpoint)
        result["panel_frame"] = frame["panel_frame"]
        result["inactive_slots"] = list(frame["inactive_slots"])
        # FIX A(3): a PARTIAL slot is inside n_active_slots and absent from inactive_slots — a
        # consumer reading only inactive_slots would see a slot that was dead for most of the
        # window as fully live. Discoverable at top level, next to inactive_slots.
        result["partial_slots"] = list(frame["partial_slots"])
        result["frame_symbols"] = list(frame["frame_symbols"])
        # D30 amendment 4: the stale-tail slots at TOP level on BOTH pin paths (a stale tail is in
        # neither inactive_slots nor partial_slots, so a consumer reading only those two lists sees
        # a slot whose price has been frozen for part of the window as fully live). Appended after
        # every pre-existing key, so a daily-pinned artifact keeps all of its previous values.
        result["stale_tail_slots"] = list(frame["panel_frame"].get("stale_tail_slots") or [])
        # FIX PASS: the same treatment for FROZEN slots (interior spans included). A slot with an
        # interior frozen span is in NONE of inactive_slots / partial_slots / stale_tail_slots, so
        # without this a consumer reading all three still sees it as fully live.
        result["frozen_slots"] = list(frame["panel_frame"].get("frozen_slots") or [])
        # ... and the CELL-level honesty on the rows this run actually decided on, computed here
        # (load_framed_panel does not know start_row/N). Disjoint: INERT (no price => force-HOLD,
        # == n_hold_cells) + FROZEN (a forward-filled price, TRADEABLE) + LIVE = the whole grid.
        _fz = _load_eval_module().frozen_decision_cells(frame["coverage"], start_row, N)
        _n_cells = int(N * int(args.n_assets))
        _n_inert = 0 if hold_mask is None else int(hold_mask.sum())
        result["decision_cell_honesty"] = {
            "n_decision_cells": _n_cells,
            "n_inert_decision_cells": _n_inert,
            "n_frozen_decision_cells": _fz["total"],
            "n_frozen_interior_decision_cells": _fz["interior"],
            "n_frozen_stale_tail_decision_cells": _fz["tail"],
            "n_not_real_decision_cells": _n_inert + _fz["total"],
            "n_live_decision_cells": _n_cells - _n_inert - _fz["total"],
            "inert_decision_cell_pct": round(100.0 * _n_inert / max(_n_cells, 1), 4),
            "frozen_decision_cell_pct": round(100.0 * _fz["total"] / max(_n_cells, 1), 4),
            "not_real_decision_cell_pct": round(
                100.0 * (_n_inert + _fz["total"]) / max(_n_cells, 1), 4),
            "live_decision_cell_pct": round(
                100.0 * (_n_cells - _n_inert - _fz["total"]) / max(_n_cells, 1), 4),
            "frozen_span_slots": _fz["slots"],
            "rule": ("INERT = no price at all (force-HOLD via hold_mask); FROZEN = a "
                     "forward-filled price no market printed, still TRADEABLE and weight-eligible "
                     "(trailing spans cannot fabricate PnL, INTERIOR spans can — they re-anchor); "
                     "LIVE = the panel row samples a real close. Disjoint, and they sum to "
                     "n_decision_cells."),
        }
        if frame["panel_frame"].get("panel_frame_source") == "minute":
            result["panel_frame_source"] = "minute"

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

    # ---- fix pass FIX 0: PINNED degenerate replay = fail loud, regardless of --smoke-assert ----
    # The verdict was decided BEFORE the artifact was built, so whatever is on disk carries
    # sanity.degenerate_replay = true and the reason. The checkpoint was written earlier (the
    # replay needs the trained net), so it is STAMPED with the same marker here — nothing this run
    # leaves behind may read as a complete result. Unpinned/legacy runs are untouched.
    if degenerate_reason is not None:
        marker = os.path.join(ckpt_path, "DEGENERATE_REPLAY.json")
        stamped = []
        try:
            with open(marker, "w") as fh:
                json.dump({"degenerate_replay": True, "reason": degenerate_reason, "tag": tag,
                           "artifact": out_path, "iters": int(args.iters),
                           "seed": int(args.seed),
                           "panel_frame_from_checkpoint":
                               _display_path(args.panel_frame_from_checkpoint),
                           "warning": "this checkpoint is the net of a FAILED "
                                      "(degenerate-replay) pinned imprint run — it is NOT an "
                                      "ExIt warm start"}, fh, indent=2)
            stamped.append(marker)
        except OSError as e:                        # never mask the refusal with an IO error
            print(f"[863] WARN: could not stamp {marker} ({type(e).__name__}: {e})")
        raise SystemExit(
            "\n" + "=" * 84 + "\n"
            f"[863] REFUSED — {degenerate_reason}\n"
            f"  exit non-zero INDEPENDENT of --smoke-assert (smoke_assert={bool(args.smoke_assert)}"
            f"): a pinned run whose sanity replay has no finite Sharpe is not a result. FIX 2.6: "
            f"the EXIT CODE is the only wiring — as of this change nothing downstream reads "
            f"sanity.degenerate_replay or DEGENERATE_REPLAY.json (835's scorecard.py never opens "
            f"an imprint.py artifact); the marker and the stamp are ADVISORY forensics for the "
            f"operator, available for a consumer to assert on but not yet consumed by one.\n"
            + (f"  the artifact WAS written for forensics and CARRIES the marker: {out_path} "
               f"(sanity.degenerate_replay=true, sanity.degenerate_replay_reason=...)\n"
               if out_path else "  no artifact was written (no --out)\n")
            + (f"  checkpoint stamped (advisory): {', '.join(stamped)}\n" if stamped else "")
            + f"  the checkpoint directory {ckpt_path} holds the net of a FAILED run — do NOT "
              f"warm-start ExIt from it.\n" + "=" * 84)

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
    pf = r.get("panel_frame")                            # TASK_863 (absent when unpinned)
    if pf is not None:
        print("-" * 84)
        print(f"[863] PINNED FRAME: {pf['n_slots']} slots, {pf['n_active_slots']} ACTIVE / "
              f"{pf['n_inactive_slots']} INACTIVE / {pf['n_partial_slots']} partial "
              f"(provenance {pf['provenance_path']})")
        for c in pf["inactive_spans"]:
            print(f"   INACTIVE slot {c['slot']:>3} {c['symbol']:<16} "
                  f"[{c['span'][0]}..{c['span'][1]}] ({c['reason']})")
        for c in pf["partial_spans"]:
            print(f"   PARTIAL  slot {c['slot']:>3} {c['symbol']:<16} first covered ts "
                  f"{c['first_covered_ts']} ({c['leading_inactive_rows']} leading inactive rows) "
                  f"— INSIDE n_active_slots, ABSENT from inactive_slots (see partial_slots)")
        # D30 amendment 4: the stale tails, on BOTH pin paths. Forward-filled at a frozen price =>
        # tradeable and weight-eligible, and in NEITHER of the two lists above.
        for c in pf.get("stale_tail_spans") or []:
            print(f"   STALE    slot {c['slot']:>3} {c['symbol']:<16} last real close ts "
                  f"{c['last_real_ts']} ({c['trailing_stale_rows']} trailing rows FORWARD-FILLED "
                  f"at a frozen price, still tradeable) — INSIDE n_active_slots, ABSENT from BOTH "
                  f"inactive_slots and partial_slots (see stale_tail_slots)")
        print(f"   forced-HOLD cells: {pf.get('n_force_hold_cells')} of "
              f"{pf.get('n_decision_cells')} (teacher targets == BC labels == replay)")
        # FIX C: state the MEASURED claim, not "flat-zero features".
        fchk = pf.get("inactive_slot_feature_check")
        if fchk is not None:
            for s in fchk["checked_slots"]:
                print(f"   inactive slot {s['slot']:>3} features: PANEL cells flat-zero; ENCODED "
                      f"features CONSTANT across all {fchk['n_decisions']} decisions but NOT "
                      f"all-zero — {s['n_nonzero_feature_cells']} of {s['n_feature_cells']} cells "
                      f"non-zero on guarded vol-ratio channels {s['nonzero_channels']} "
                      f"(= {sorted(set(s['nonzero_channel_values'].values()))}); constant => no "
                      f"signal => no leak (GUARDED, not just claimed)")
        # FIX A(2): make the masked-cell dilution of every reported scalar explicit.
        sn = r.get("sanity") or {}
        ac = sn.get("active_cells_only")
        if sn.get("masked_cell_fraction") is not None:
            print(f"   MASKED-CELL DILUTION: {sn.get('n_masked_cells')} of "
                  f"{sn.get('n_decision_cells')} decision cells "
                  f"({100.0 * sn['masked_cell_fraction']:.2f}%) are force-HOLD on BOTH sides, so "
                  f"the figures below are NOT comparable to an unpinned reference (the +0.839 "
                  f"run's 97.5% agreement was measured on a fully-covered minute panel):")
            print(f"     teacher_action_agreement reported "
                  f"{sn.get('teacher_action_agreement'):.4f}"
                  + (f" | ACTIVE-CELLS-ONLY {ac['teacher_action_agreement']:.4f} "
                     f"(n={ac['n_active_cells']})" if ac and
                     ac.get("teacher_action_agreement") is not None else " | active-only n/a"))
            print(f"     policy_action_hist reported {sn.get('policy_action_hist')}"
                  + (f" | ACTIVE-CELLS-ONLY {ac['policy_action_hist']}" if ac and
                     ac.get("policy_action_hist") is not None else ""))
            if ac:
                print(f"     avg_long/avg_short ACTIVE-CELLS-ONLY per tier: "
                      f"long {ac['per_tier_avg_long']}  short {ac['per_tier_avg_short']}")
        tvf = ((r.get("teacher_vectors") or {}).get("fit_vs_original_targets")) or {}
        if tvf.get("active_cells_only"):
            a = tvf["active_cells_only"]
            print(f"     861 fit block is diluted the same way: NLL reduction reported "
                  f"{tvf['nll_reduction']:+.4f} | ACTIVE-CELLS-ONLY {a['nll_reduction']:+.4f}; "
                  f"argmax agreement {tvf['argmax_agreement_vs_original']:.3f} -> "
                  f"{a['argmax_agreement_vs_original']:.3f} vs modal rate "
                  f"{tvf['modal_action_rate']:.3f} -> {a['modal_action_rate']:.3f} (the modal "
                  f"floor itself moves, so argmax_excess_over_modal was measured against a "
                  f"shifted floor)")
        if sn.get("degenerate_replay"):
            print(f"   *** DEGENERATE REPLAY (863 FIX 0): {sn.get('degenerate_replay_reason')}")
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


# ---- TASK_864: tracked-parity-baseline hygiene ---------------------------------------------
# The failure mode this closes: `results_fixture_smoke.json` is TRACKED, and every ordinary
# verification run used to write it in place. A stray run therefore leaves a dirty tracked file
# that reads like a legitimate "refresh" -- and a real behavior change gets laundered into the
# parity baseline with nobody deciding the values were allowed to move. (Run 20 actually clobbered
# it via an empty-string flag bug; run 21 mis-read the resulting weight-ON-vs-weight-OFF gap as
# "74 commits of staleness".) Fix: the default output path is untracked, and promotion is an
# explicit, diff-printing, fail-closed step.

def _json_leaves(obj, prefix=""):
    """Flatten a JSON document to (path, scalar) pairs -- the per-field unit of the 864 diff."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _json_leaves(v, f"{prefix}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _json_leaves(v, f"{prefix}[{i}]")
    else:
        yield prefix, obj


def _assert_baseline_promotable(path):
    """Fail-closed gate on --write-parity-baseline: the baseline is promotable only from a
    verifiably CLEAN working copy of that one file.

    A dirty baseline means we cannot tell what we would be destroying, which is exactly the
    laundering path this guard exists to block. Anything we cannot VERIFY (git missing, not a
    work tree, status unreadable) refuses too -- fail-closed, not fail-open."""
    rel = os.path.relpath(os.path.abspath(path), _REPO)
    try:
        p = subprocess.run(["git", "-C", _REPO, "status", "--porcelain", "--", path],
                           capture_output=True, text=True, timeout=120)
        status = p.stdout if p.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        status = None
    if status is None:
        raise SystemExit(
            f"[832] REFUSING --write-parity-baseline: could not read `git status` for {rel} "
            f"(git missing, or {_REPO} is not a work tree). The tracked parity baseline is only "
            f"promotable from a verifiably clean working copy.")
    if status.strip():
        raise SystemExit(
            f"[832] REFUSING --write-parity-baseline: {rel} is git-DIRTY\n"
            f"          {status.strip()}\n"
            f"  Commit it, or `git checkout -- {rel}`, then re-run. Promoting on top of "
            f"uncommitted edits would destroy them and is the exact path by which an unreviewed "
            f"behavior change gets laundered into the parity baseline (TASK_864).")


def _resolve_run_defaults(args):
    """Apply the pre-registered fixture-vs-real defaults IN PLACE; return
    (exp_dir, is_fixture, tv_suffix, tag).

    Extracted from main() unchanged so there is exactly ONE copy: the FIX-1 canon-recipe gate
    re-derives its expected values by running this same resolution over a defaults-only
    Namespace, and so cannot drift from the defaults main() actually uses."""
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
    # TASK_863: a frame-PINNED run is likewise a different experiment from the canon smoke (a
    # different panel source AND a different universe), so it suffixes its default outputs too —
    # `results_fixture_smoke.json` / `ckpt_smoke` are the CHECKED-IN parity baseline and no
    # verification run may overwrite them.
    tv_suffix = ""
    if args.teacher_vectors is not None:
        tv_suffix = "_teacher_vectors" + ("_shuffled" if args.shuffle_teacher_vectors else "")
    if args.panel_frame_from_checkpoint is not None:
        # TASK_865: a MINUTE-sourced pin is a different experiment from the daily one (a different
        # price series over the same frame), so it gets its own suffix; the daily suffix is
        # unchanged, so every shipped 863 default output path is byte-identical.
        tv_suffix += "_frame_pin"
        if getattr(args, "panel_frame_source", "daily") == "minute":
            tv_suffix += "_minute"
    if args.ckpt_out is None:
        args.ckpt_out = os.path.join(
            exp_dir, ("ckpt_smoke" if is_fixture else "ckpt_imprint") + tv_suffix)
    if args.smoke_assert is None:
        args.smoke_assert = is_fixture
    tag = args.tag or (("fixture-smoke" if is_fixture else "real-panel")
                       + tv_suffix.replace("_", "-"))
    return exp_dir, is_fixture, tv_suffix, tag


def _canon_recipe_args(ap):
    """The CANON promoting recipe as a fully-resolved Namespace, DERIVED from this module's own
    argparse defaults: parse CANON_BASELINE_ARGV (nothing but the weight-rebalance flag) through
    the SAME parser, then run the SAME _resolve_run_defaults. No second copy of any default."""
    canon = ap.parse_args(list(CANON_BASELINE_ARGV))
    _resolve_run_defaults(canon)
    return canon


def _assert_canon_recipe(ap, args):
    """FIX 1 (BLOCKER): --write-parity-baseline must pin the RECIPE, not just the destination.

    Every argparse dest except the promote flag and `out` reaches the artifact (db -> /db,
    ckpt_out -> /checkpoint, tag -> /tag, everything else -> /imprint/config or the panel/env
    semantics), so ANY deviation from the canon recipe produces a DIFFERENT artifact that would
    silently become the parity oracle. The observed hole: the FLAGLESS default promoted onto the
    weight-ON baseline (462/545 leaves replaced, vector_weight_rebalance True -> False, exit 0),
    and there is no downstream detector -- 861/smoke_e2e.py only checks within-run byte identity
    and the 832 canon pins are hardcoded with an explicit --out. So this refuses, fail-fast,
    BEFORE the run, naming every offending field and the value the canon recipe expects."""
    canon = _canon_recipe_args(ap)
    bad = [(f, getattr(args, f, "<missing>"), want)
           for f, want in sorted(vars(canon).items())
           if f not in CANON_RECIPE_EXEMPT_FIELDS and getattr(args, f, "<missing>") != want]
    if bad:
        ap.error(
            "--write-parity-baseline REFUSED: this is not the CANON parity-baseline recipe, so "
            "promoting it would install a DIFFERENT experiment as the tracked oracle (TASK_864 "
            "FIX 1).\n"
            + "".join(f"    --{f.replace('_', '-')}: got {g!r}, canon expects {w!r}\n"
                      for f, g, w in bad)
            + f"  The canon recipe is EXACTLY: {CANON_BASELINE_RECIPE}\n"
              f"  (weight-ON, order-code mapping, every other knob at its fixture default; the "
              f"expected values above are re-derived from this module's argparse defaults, not a "
              f"second hardcoded copy). To run a different recipe, drop "
              f"--write-parity-baseline and pass --out <scratch>.json.")


def _assert_result_promotable(result, args):
    """FIX 2 (MAJOR): the run must have PASSED its own validity gate to become the oracle.

    run() writes the artifact BEFORE the --smoke-assert block, and main() used to promote on any
    exit-0 run, so `--iters 1 --no-smoke-assert --write-parity-baseline` installed an artifact
    carrying policy_ce_decreased:false with its loss trajectories truncated 80 -> 1. Asserted
    HERE on the bytes actually being promoted and INDEPENDENT of --smoke-assert, matching the
    TASK_863 FIX-0 degenerate-replay precedent."""
    im = result.get("imprint") or {}
    gate = result.get("expressibility_gate")
    per_tier = (result.get("sanity") or {}).get("per_tier") or {}
    bad = []
    if im.get("all_losses_finite") is not True:
        bad.append(f"imprint.all_losses_finite = {im.get('all_losses_finite')!r} (need true)")
    if im.get("policy_ce_decreased") is not True:
        bad.append(f"imprint.policy_ce_decreased = {im.get('policy_ce_decreased')!r} (need true)")
    # A truncated loss trajectory is the tell of the demonstrated `--iters 1` promotion: the
    # per-key summaries survive but their point series collapses (80 -> 1). `trajectory` is
    # sub-sampled only above 200 points (_loss_summary), so below that it is one point per iter.
    traj = ((im.get("losses") or {}).get("policy_loss") or {}).get("trajectory") or ()
    if int(args.iters) <= 200 and len(traj) != int(args.iters):
        bad.append(f"imprint.losses.policy_loss.trajectory has {len(traj)} point(s), expected "
                   f"{int(args.iters)} (one per --iters) — a truncated run, not a full imprint")
    elif not traj:
        bad.append("imprint.losses.policy_loss.trajectory is empty (no imprint history)")
    for c in GATE_COSTS:
        if (per_tier.get(str(c)) or {}).get("policy_inenv") is None:
            bad.append(f"sanity.per_tier.{c}.policy_inenv is missing (sanity replay Sharpe)")
    if not isinstance(gate, dict):
        bad.append("expressibility_gate is missing (the 845 deliverable)")
    else:
        for k in ("teacher_order_net21", "teacher_weight_net21"):
            if gate.get(k) is None:
                bad.append(f"expressibility_gate.{k} is missing")
    if bad:
        raise SystemExit(
            "\n" + "=" * 84 + "\n"
            "[832] REFUSING --write-parity-baseline: this run did NOT pass its own validity "
            "gate, so its artifact is not a result and must not become the parity oracle "
            "(TASK_864 FIX 2).\n"
            + "".join(f"    - {b}\n" for b in bad)
            + f"  The tracked baseline is UNCHANGED. Fix the run, then re-run the canon recipe: "
              f"{CANON_BASELINE_RECIPE}\n" + "=" * 84)


def promote_parity_baseline(result, baseline_path, source_label, args, max_show=40):
    """Print the per-field diff vs the checked-in baseline, then replace it ATOMICALLY.

    Order is deliberate: the operator sees WHAT would move before it moves. The git-dirty gate is
    re-checked HERE as well as pre-run, so an edit landing during the run is not silently lost.

    FIX 3: the promoted bytes are re-serialised from THIS run's in-memory `result` (identical
    json.dumps params to the write in run()), never re-read from the now-SHARED default output
    path -- a concurrent default run can no longer have its bytes promoted. The write goes to
    <baseline>.tmp + os.replace, so an interrupt or ENOSPC mid-write can never leave the tracked
    oracle truncated (os.replace is atomic within a filesystem, and tmp is a sibling)."""
    _assert_result_promotable(result, args)
    _assert_baseline_promotable(baseline_path)
    rel = os.path.relpath(os.path.abspath(baseline_path), _REPO)
    # Byte-for-byte what run() wrote, produced from the in-memory result (not re-read from disk).
    new_text = json.dumps(result, indent=2, default=float)
    print("\n" + "=" * 84)
    print(f"[832] --write-parity-baseline: per-field diff vs the checked-in {rel}")
    print(f"  promoting recipe: {CANON_BASELINE_RECIPE}")
    print(f"  bytes source: this run's in-memory result (mirrored to {source_label})")
    if not os.path.exists(baseline_path):
        print("  no existing baseline -- this run CREATES it (nothing to compare)")
    else:
        with open(baseline_path) as fh:
            old = dict(_json_leaves(json.load(fh)))
        new = dict(_json_leaves(json.loads(new_text)))
        moved = sorted(k for k in set(old) & set(new) if old[k] != new[k])
        removed, added = sorted(set(old) - set(new)), sorted(set(new) - set(old))
        print(f"  leaves {len(old)} -> {len(new)}   moved {len(moved)}   "
              f"removed {len(removed)}   added {len(added)}")
        for label, keys, fmt in (
                ("~", moved, lambda k: f"{old[k]!r}  ->  {new[k]!r}"),
                ("-", removed, lambda k: f"was {old[k]!r}"),
                ("+", added, lambda k: f"= {new[k]!r}")):
            for k in keys[:max_show]:
                print(f"    {label} {k}\n        {fmt(k)}")
            if len(keys) > max_show:
                print(f"    ... {len(keys) - max_show} more '{label}' field(s)")
        if not (moved or removed or added):
            print("  IDENTICAL -- the baseline already matches this run")
    tmp = baseline_path + ".tmp"
    try:
        with open(tmp, "w") as fh:
            fh.write(new_text)
        os.replace(tmp, baseline_path)                     # atomic; no truncated oracle window
    finally:
        if os.path.exists(tmp):                            # only reachable if the write failed
            os.unlink(tmp)
    print(f"[832] parity baseline PROMOTED -> {rel}  (md5 "
          f"{hashlib.md5(new_text.encode()).hexdigest()})")
    print("      A refresh commit MUST carry the per-field attribution table (every moved field "
          "traced to the commit that moved it) -- TASK_864 acceptance (iii), and SPEC.md's "
          "'Promoting recipe' record MUST be updated with the recipe + md5 above.")
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
    ap.add_argument("--panel-frame-from-checkpoint", default=None,
                    dest="panel_frame_from_checkpoint",
                    help="TASK_863 (D28): path to a JSON recording a checkpoint's ORDERED "
                         "training symbol frame ('frame_symbols' or 'symbols'). Builds the fixed "
                         "n_assets-slot panel in that EXACT order (slot k == asset k, matching "
                         "the positionally universe-locked encoder) with zero-coverage and "
                         "pre-listing PANEL CELLS left flat-zero and forced to HOLD on every "
                         "in-env walk (a dead slot's ENCODED features are CONSTANT but not "
                         "all-zero: guarded vol-ratio channels at 0.2 — asserted per run) "
                         "— the SHIPPED TASK_853 semantics (eval_vector.py's frame validator "
                         "and framed loader are called, not re-derived). Requires --bar-minutes "
                         "1440 (the frame is read from `ohlcv_daily`). The 861 provenance gate is "
                         "fail-closed in BOTH directions: a pinned collection needs an "
                         "identically-pinned imprint and vice versa. Default OFF = the "
                         "coverage-ranked minute panel, bit-identical to shipped 832/861")
    ap.add_argument("--panel-frame-source", choices=("daily", "minute"), default="daily",
                    dest="panel_frame_source",
                    help="TASK_865 (D29): which table --panel-frame-from-checkpoint reads the "
                         "PINNED frame from. 'daily' (default = the shipped 863 path, "
                         "bit-identical) reads ohlcv_daily and requires --bar-minutes 1440; "
                         "'minute' reads the minute `ohlcv` table at the canon [::bar-minutes] "
                         "stride — the only pin that is apples-to-apples with the "
                         "minute-panel-native +0.839 bar on universe, source AND breadth at once "
                         "(62/64 vs 33/64 frame slots on the 861 leg-1 window). The provenance "
                         "gate is fail-closed on the SOURCE too: a daily-pinned collection may "
                         "not be imprinted under a minute pin, or vice versa (D20)")
    ap.add_argument("--require-active-slots", type=float, default=None,
                    dest="require_active_slots",
                    help="D30 rider (b): REFUSE a MINUTE-pinned run whose frame is too inert to "
                         "produce a comparable number. <=1 is a FRACTION of the frame's slots, >1 "
                         "an absolute slot count, 0 disables. Default = eval_vector's "
                         "REQUIRE_ACTIVE_SLOTS_DEFAULT (0.9): the 861 leg-1 pin is 64/64 and "
                         "passes, the 2021-2022 WF fold is 50/64 and refuses without an explicit "
                         "override, the 8-slot fixture frame is 6/8 by design and refuses (the "
                         "fixture smokes pass 0). Daily pins are unaffected")
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
    ap.add_argument("--write-parity-baseline", dest="write_parity_baseline",
                    action="store_true", default=False,
                    help=f"TASK_864: promote this run's artifact onto the CHECKED-IN parity "
                         f"baseline {PARITY_BASELINE_NAME}. Default OFF => a fixture run writes "
                         f"the gitignored {PARITY_LOCAL_NAME} instead, so NO ordinary "
                         f"verification run can dirty the tracked baseline. Prints a per-field "
                         f"diff before promoting and REFUSES if the tracked file is git-dirty. "
                         f"The baseline recipe is the weight-ON / order-code deliverable "
                         f"(SPEC.md 'Committed fixture artifact'), NOT the flagless default: "
                         f"`imprint.py --vector-weight-rebalance --write-parity-baseline`")
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
    # TASK_863: same fail-closed rule for the frame pin (the 835 precedent, build_panel). An unset
    # shell variable must never silently fall back to the coverage-ranked universe under a flag
    # whose whole point is that the universe is PINNED.
    if (args.panel_frame_from_checkpoint is not None
            and not args.panel_frame_from_checkpoint.strip()):
        ap.error("--panel-frame-from-checkpoint was given an EMPTY path; refusing to silently "
                 "fall back to the coverage-ranked universe under the fixed-frame flag (that "
                 "would train a DIFFERENT universe while the artifact claims a pinned frame). "
                 "Pass the provenance JSON path, or omit the flag entirely.")
    # TASK_865: --panel-frame-source only selects the PIN's table, so it is inert without a pin.
    # Refuse rather than let an invocation claim a source it never used (the same fail-closed
    # rule as the empty path above, and the 835 build_panel precedent).
    if args.panel_frame_source != "daily" and args.panel_frame_from_checkpoint is None:
        ap.error(
            f"--panel-frame-source {args.panel_frame_source} requires "
            f"--panel-frame-from-checkpoint: it chooses which table the PINNED frame is read "
            f"from and does nothing without a pin (the unpinned panel is always the "
            f"coverage-ranked minute one, PIN b). Pass the provenance JSON, or drop the flag.")
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

    exp_dir, is_fixture, tv_suffix, tag = _resolve_run_defaults(args)
    # TASK_864: coherence + fail-fast dirty check BEFORE the ~1min run, so a refused promotion
    # costs nothing. The suffixed variants (861 teacher-vectors, 863 frame pin) are DIFFERENT
    # experiments with their own artifacts -- they can never be promoted onto the canon baseline.
    baseline_path = os.path.join(exp_dir, PARITY_BASELINE_NAME)
    if args.write_parity_baseline:
        if not is_fixture:
            ap.error("--write-parity-baseline is fixture-only: the checked-in baseline IS the "
                     f"fixture-smoke artifact. Got --db {args.db}.")
        if args.out is not None:
            ap.error("--write-parity-baseline conflicts with an explicit --out (it promotes onto "
                     f"{PARITY_BASELINE_NAME}). Drop one.")
        if tv_suffix:
            ap.error(f"--write-parity-baseline refuses a '{tv_suffix.lstrip('_')}' run: that is a "
                     f"DIFFERENT experiment with its own suffixed artifact, never the canon "
                     f"{PARITY_BASELINE_NAME} baseline.")
        # FIX 2 (MAJOR): promotion requires the run's OWN validity gate to be armed. Named
        # explicitly (the FIX-1 recipe check below would also catch it, on the same field).
        if not args.smoke_assert:
            ap.error("--write-parity-baseline requires --smoke-assert (fixture default ON): "
                     "--no-smoke-assert disarms the finite-loss / decreasing-CE / schema gate, "
                     "and an artifact that was never checked must not become the parity oracle "
                     "(TASK_864 FIX 2). The result is ALSO re-asserted post-run before promoting.")
        # FIX 1 (BLOCKER): pin the RECIPE, not just the destination.
        _assert_canon_recipe(ap, args)
        _assert_baseline_promotable(baseline_path)
    out = args.out
    if out is None and is_fixture:
        # TASK_864: the canon (unsuffixed) recipe now writes an UNTRACKED local file by default.
        # `out` is not recorded anywhere in the artifact, so this changes the DESTINATION only --
        # the bytes written are identical to what the pre-864 default produced.
        out = (os.path.join(exp_dir, f"results_fixture_smoke{tv_suffix}.json") if tv_suffix
               else os.path.join(exp_dir, PARITY_LOCAL_NAME))
    result = run(args, out, tag)
    if args.write_parity_baseline:
        # FIX 3: promote THIS run's in-memory result, not a re-read of `out` (which is now the
        # single shared destination of every unsuffixed fixture run, so a concurrent default
        # run's bytes could otherwise be promoted).
        promote_parity_baseline(result, baseline_path, os.path.relpath(out, _REPO), args)


if __name__ == "__main__":
    main()
